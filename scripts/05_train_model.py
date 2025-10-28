#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_train_model.py — Train/Test/Predict NFL model (snake_case output)
────────────────────────────────────────────────────────────────────────────
Trains LightGBM with temporal gating and saves feature schema for safe prediction.
All output filenames are normalized to snake_case (no spaces).
Source data pulled from normalized feature set.
Includes walk-forward hyperparameter tuning with enhanced logging.
"""

import argparse
import datetime as dt
import json
import re
import sys
from itertools import product
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from joblib import dump
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, train_test_split

from adjustment_utils import apply_edge_threshold
EXCLUDED_FEATURES = {"home_score", "away_score"}


def auto_triage_features(
    X: pd.DataFrame,
    y: pd.Series,
    gain_threshold: float,
    top_n: int,
    seed: int,
) -> tuple[pd.Index, list[str]]:
    """
    Train a lightweight booster to rank features by gain and return a filtered list.
    """
    if X.empty:
        return X.columns, []

    params = {
        "objective": "regression",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l2": 5.0,
        "seed": seed,
    }
    booster = lgb.train(
        params,
        lgb.Dataset(X.fillna(0.0), label=y),
        num_boost_round=200,
    )
    importance = booster.feature_importance(importance_type="gain")
    imp_series = pd.Series(importance, index=X.columns, dtype=float)

    keep_mask = imp_series >= gain_threshold if gain_threshold > 0 else imp_series > 0
    keep_cols = imp_series[keep_mask].index

    if top_n and len(keep_cols) < min(top_n, len(imp_series)):
        keep_cols = imp_series.sort_values(ascending=False).head(top_n).index

    dropped = sorted(set(X.columns) - set(keep_cols))
    return pd.Index(keep_cols), dropped


# --------------------------------------------------------------------------- #
def normalize_tag(tag: str) -> str:
    tag = tag.replace(" ", "_").replace("-", "_")
    tag = re.sub(r"[^0-9a-zA-Z_]", "", tag)
    return tag.lower()


def parse_gate(value):
    parts = str(value).replace("W", " ").split()
    return int(parts[0]), int(parts[1])


def slice_by_gate(df, traingate=None, testgate=None, predictgate=None):
    """Split data based on season/week cutoffs."""

    def in_gate(df, s, w, le=True):
        if le:
            return (df["season"] < s) | ((df["season"] == s) & (df["week"] <= w))
        return (df["season"] == s) & (df["week"] == w)

    s_train, w_train = parse_gate(traingate) if traingate else (9999, 9999)
    s_test, w_test = parse_gate(testgate) if testgate else (9999, 9999)
    s_pred, w_pred = parse_gate(predictgate) if predictgate else (9999, 9999)

    df_train = df[in_gate(df, s_train, w_train, le=True)].copy()
    df_test = df[in_gate(df, s_test, w_test, le=False)].copy()
    df_pred = df[in_gate(df, s_pred, w_pred, le=False)].copy()

    if testgate and predictgate and testgate == predictgate:
        print(f"🧩 testgate == predictgate ({testgate}) — merging eval & forecast.")
        df_pred = df_test.copy()
    return df_train, df_test, df_pred


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Convert categorical columns to integer codes."""
    cat_cols = df.select_dtypes(include=["object"]).columns
    if len(cat_cols) > 0:
        print(f"⚙️ Encoding categorical columns: {list(cat_cols)}")
        for c in cat_cols:
            df[c] = df[c].astype("category").cat.codes
    return df


def run_time_series_cv(
    df: pd.DataFrame,
    base_params: dict,
    param_grid: dict,
    n_splits: int = 5,
    label_col: str = "residual_target",
    baseline_col: str = "baseline_prob",
    original_target_col: str = "target",
):
    """Perform walk-forward cross-validation over the provided parameter grid for residual modeling."""
    sortable = df.sort_values(["season", "week"]).reset_index(drop=True)
    feature_cols = [c for c in sortable.columns if c not in {label_col, original_target_col}]
    X = sortable[feature_cols]
    residual = sortable[label_col].astype(float)
    baseline = sortable[baseline_col].astype(float)
    actual = sortable[original_target_col].astype(float)

    if len(sortable) < (n_splits + 1):
        print("⚠️ Not enough rows for requested CV splits; skipping hyperparameter search.")
        return None, []

    splitter = TimeSeriesSplit(n_splits=n_splits)
    results = []

    combos = list(product(
        param_grid["learning_rate"],
        param_grid["num_leaves"],
        param_grid["feature_fraction"],
        param_grid["bagging_fraction"],
        param_grid["lambda_l2"],
        param_grid["min_data_in_leaf"],
    ))

    print(f"🔍 Hyperparameter search over {len(combos)} combinations (TimeSeriesSplit={n_splits})")

    for idx, (lr, leaves, feat_frac, bag_frac, lambda_l2, min_leaf) in enumerate(combos, start=1):
        params = {
            **base_params,
            "learning_rate": lr,
            "num_leaves": leaves,
            "feature_fraction": feat_frac,
            "bagging_fraction": bag_frac,
            "lambda_l2": lambda_l2,
            "min_data_in_leaf": min_leaf,
        }

        fold_metrics = []
        for fold, (train_idx, val_idx) in enumerate(splitter.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = residual.iloc[train_idx], residual.iloc[val_idx]
            baseline_val = baseline.iloc[val_idx].values
            y_val_actual = actual.iloc[val_idx].values

            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val)

            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
            )

            best_iter = model.best_iteration or params.get("num_boost_round", 100)
            y_pred_resid = model.predict(X_val, num_iteration=best_iter)
            raw_pred = baseline_val + y_pred_resid
            y_pred_prob = apply_edge_threshold(raw_pred, baseline_val)

            fold_metrics.append({
                "fold": fold,
                "logloss": log_loss(y_val_actual, y_pred_prob, labels=[0, 1]),
                "auc": roc_auc_score(y_val_actual, y_pred_prob),
                "best_iteration": best_iter,
            })

        avg_logloss = float(np.mean([m["logloss"] for m in fold_metrics]))
        avg_auc = float(np.mean([m["auc"] for m in fold_metrics]))
        avg_iter = float(np.mean([m["best_iteration"] for m in fold_metrics]))
        result = {
            "learning_rate": lr,
            "num_leaves": leaves,
            "feature_fraction": feat_frac,
            "bagging_fraction": bag_frac,
            "lambda_l2": lambda_l2,
            "min_data_in_leaf": min_leaf,
            "logloss": avg_logloss,
            "auc": avg_auc,
            "best_iteration": avg_iter,
        }
        results.append(result)
        print(f"   [{idx}/{len(combos)}] {result}")

    if not results:
        return None, []

    results_sorted = sorted(results, key=lambda r: (r["logloss"], -r["auc"]))
    best = results_sorted[0]
    print(f"🏆 Best params: {best}")
    return best, results_sorted


# --------------------------------------------------------------------------- #
def train_model(args):
    """Train and save LightGBM model."""
    data_file = Path("data/normalized/training_features_normalized.csv")
    legacy_file = Path("data/features/training_features.csv")
    if not data_file.exists():
        if legacy_file.exists():
            print(f"⚠️ Normalized features missing at {data_file}. Falling back to {legacy_file}.")
            data_file = legacy_file
        else:
            raise FileNotFoundError(
                "Missing normalized training features. Run 04_build_features.py to regenerate them."
            )

    print(f"📘 Loading training features → {data_file}")
    df = pd.read_csv(data_file)
    df.columns = [c.lower() for c in df.columns]

    df["season"] = pd.to_numeric(df["season"], errors="coerce").fillna(0).astype(int)
    df["week"] = pd.to_numeric(df["week"], errors="coerce").fillna(0).astype(int)

    if "target" not in df.columns:
        raise ValueError("❌ 'target' column missing from training features.")

    df_targets = df[df["target"].notna()]
    if df_targets.empty:
        raise ValueError("❌ No rows with non-null target found for training/testing.")

    last_season = df_targets["season"].max()
    last_week = df_targets[df_targets["season"] == last_season]["week"].max()

    if not args.predictgate:
        next_week = last_week + 1
        inferred_predictgate = f"{last_season}W{next_week}"
        print(f"ℹ️ --predictgate not provided; inferred as next unplayed week: {inferred_predictgate}")
        predictgate = inferred_predictgate
    else:
        predictgate = args.predictgate

    df_train_test = df[df["target"].notna()].copy()
    future_rows_count = len(df) - len(df_train_test)
    print(f"ℹ️ Last completed week with target: Season {last_season} Week {last_week}")
    print(f"ℹ️ Rows with NaN target (future games) excluded from training/testing: {future_rows_count}")

    if args.rolling_seasons:
        min_season = last_season - (args.rolling_seasons - 1)
        before = len(df_train_test)
        df_train_test = df_train_test[df_train_test["season"] >= min_season]
        print(
            f"🔄 Rolling window (seasons={args.rolling_seasons}): "
            f"kept {len(df_train_test):,} rows from seasons ≥ {min_season} "
            f"(dropped {before - len(df_train_test):,})."
        )

    if args.rolling_weeks:
        df_train_test["_abs_week"] = df_train_test["season"] * 100 + df_train_test["week"]
        cutoff_abs = last_season * 100 + last_week - (args.rolling_weeks - 1)
        before = len(df_train_test)
        df_train_test = df_train_test[df_train_test["_abs_week"] >= cutoff_abs]
        df_train_test = df_train_test.drop(columns=["_abs_week"])
        print(
            f"🔄 Rolling window (weeks={args.rolling_weeks}): "
            f"kept {len(df_train_test):,} rows within the last {args.rolling_weeks} weeks "
            f"(dropped {before - len(df_train_test):,})."
        )
    else:
        df_train_test = df_train_test.drop(columns=["_abs_week"], errors="ignore")

    df_train_test_encoded = encode_categoricals(df_train_test.copy())

    # Hyperparameter tuning via walk-forward CV
    if "baseline_prob" not in df_train_test_encoded.columns:
        df_train_test_encoded["baseline_prob"] = (
            df_train_test_encoded["home_moneyline_prob"]
            .fillna(df_train_test_encoded["win_prob"])
            .fillna(0.5)
        )
    else:
        df_train_test_encoded["baseline_prob"] = (
            df_train_test_encoded["baseline_prob"]
            .fillna(df_train_test_encoded["home_moneyline_prob"])
            .fillna(df_train_test_encoded["win_prob"])
            .fillna(0.5)
        )
    df_train_test_encoded["baseline_prob"] = df_train_test_encoded["baseline_prob"].clip(0, 1)
    df_train_test_encoded["residual_target"] = df_train_test_encoded["target"] - df_train_test_encoded["baseline_prob"]

    base_params = {
        "objective": "regression",
        "metric": "l2",
        "max_depth": -1,
        "bagging_freq": 5,
        "lambda_l2": 0.0,
        "min_data_in_leaf": 30,
        "seed": args.seed,
    }
    param_grid = {
        "learning_rate": [0.05],
        "num_leaves": [31],
        "feature_fraction": [0.85, 0.9],
        "bagging_fraction": [0.7, 0.8],
        "lambda_l2": [0.0, 5.0, 10.0],
        "min_data_in_leaf": [30, 50],
    }

    best_params, cv_results = run_time_series_cv(
        df_train_test_encoded,
        base_params,
        param_grid,
    )

    if best_params:
        params_for_training = {
            **base_params,
            "learning_rate": best_params["learning_rate"],
            "num_leaves": best_params["num_leaves"],
            "feature_fraction": best_params["feature_fraction"],
            "bagging_fraction": best_params["bagging_fraction"],
            "lambda_l2": best_params["lambda_l2"],
            "min_data_in_leaf": best_params["min_data_in_leaf"],
        }
        inferred_rounds = int(round(best_params.get("best_iteration", args.num_boost_round)))
        num_boost_round = max(inferred_rounds, 100)
        print(f"🎯 Using tuned parameters: {params_for_training} (num_boost_round={num_boost_round})")
    else:
        params_for_training = {
            **base_params,
            "learning_rate": args.learning_rate,
            "num_leaves": args.num_leaves,
            "max_depth": args.max_depth,
            "feature_fraction": args.feature_fraction,
            "bagging_fraction": args.bagging_fraction,
        }
        num_boost_round = args.num_boost_round
        print("⚠️ No tuning performed; falling back to CLI/default parameters.")

    df_train_test = df_train_test_encoded

    df_train, df_test, df_pred = slice_by_gate(df_train_test, args.traingate, args.testgate, predictgate)
    print(f"📊 Train rows: {len(df_train):,} | Test rows: {len(df_test):,} | Predict rows: {len(df_pred):,}")

    if len(df_train) == 0:
        raise ValueError("❌ No training rows found. Check --traingate values.")

    feature_cols = [
        c for c in df_train.columns
        if c not in {"target", "residual_target"} and c not in EXCLUDED_FEATURES
    ]
    X_train = df_train[feature_cols]
    X_test = df_test[feature_cols]
    y_train_resid = df_train["residual_target"]
    y_test_actual = df_test["target"]
    baseline_train = df_train["baseline_prob"].values
    baseline_test = df_test["baseline_prob"].values

    if len(df_test) == 0 or X_test.empty:
        print("⚠️ No test data — using 10% validation split from training.")
        train_indices = df_train.index.to_numpy()
        stratify_labels = (df_train["target"] > 0).astype(int)
        stratify = stratify_labels if stratify_labels.nunique() > 1 else None
        idx_train, idx_test = train_test_split(
            train_indices,
            test_size=0.1,
            random_state=args.seed,
            stratify=stratify,
        )
        X_train = df_train.loc[idx_train, feature_cols]
        X_test = df_train.loc[idx_test, feature_cols]
        y_train_resid = df_train.loc[idx_train, "residual_target"]
        y_test_resid = df_train.loc[idx_test, "residual_target"]
        y_train_actual = df_train.loc[idx_train, "target"]
        y_test_actual = df_train.loc[idx_test, "target"]
        baseline_train = df_train.loc[idx_train, "baseline_prob"].values
        baseline_test = df_train.loc[idx_test, "baseline_prob"].values
    else:
        y_test_resid = df_test["residual_target"]
        y_train_actual = df_train["target"]

    triage_details = None
    if not args.disable_triage:
        selected_cols, dropped_cols = auto_triage_features(
            X_train,
            y_train_resid,
            args.feature_triage_gain,
            args.feature_triage_top_n or 0,
            args.seed,
        )
        if len(selected_cols) == 0:
            raise ValueError("❌ Feature triage removed all columns. Loosen thresholds or disable triage.")
        if dropped_cols:
            print(f"🧹 Feature triage removed {len(dropped_cols)} columns (sample: {dropped_cols[:10]})")
        feature_cols = list(selected_cols)
        X_train = X_train.reindex(columns=feature_cols)
        X_test = X_test.reindex(columns=feature_cols, fill_value=0.0)
        triage_details = {
            "enabled": True,
            "gain_threshold": args.feature_triage_gain,
            "top_n": args.feature_triage_top_n,
            "dropped": dropped_cols,
        }
    else:
        triage_details = {
            "enabled": False,
            "gain_threshold": args.feature_triage_gain,
            "top_n": args.feature_triage_top_n,
            "dropped": [],
        }

    train_data = lgb.Dataset(X_train, label=y_train_resid)
    valid_data = lgb.Dataset(X_test, label=y_test_resid, reference=train_data)

    print("🚀 Training LightGBM model...")
    lgb_model = lgb.train(
        {**params_for_training, "metric": ["l2"], "verbose": -1},
        train_data,
        valid_sets=[train_data, valid_data],
        num_boost_round=num_boost_round,
        callbacks=[
            lgb.early_stopping(stopping_rounds=args.early_stopping),
            lgb.log_evaluation(100 if not args.debug else 10),
        ],
    )

    print("🚀 Training GradientBoostingRegressor ensemble member...")
    gb_model = GradientBoostingRegressor(
        random_state=args.seed,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
    )
    gb_model.fit(X_train.fillna(0.0), y_train_resid)

    residual_pred_lgb = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration or num_boost_round)
    residual_pred_gb = gb_model.predict(X_test.fillna(0.0))
    ensemble_residual = (residual_pred_lgb + residual_pred_gb) / 2.0

    raw_pred = baseline_test + ensemble_residual
    y_pred_proba = apply_edge_threshold(raw_pred, baseline_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    acc = accuracy_score(y_test_actual, y_pred) if len(y_test_actual) > 0 else float("nan")
    logloss = log_loss(y_test_actual, y_pred_proba, labels=[0, 1]) if len(y_test_actual) > 0 else float("nan")
    try:
        auc = roc_auc_score(y_test_actual, y_pred_proba)
    except Exception:
        auc = float("nan")

    baseline_pred = np.clip(baseline_test, 1e-6, 1 - 1e-6)
    baseline_acc = accuracy_score(y_test_actual, (baseline_pred > 0.5).astype(int)) if len(y_test_actual) > 0 else float("nan")
    baseline_logloss = log_loss(y_test_actual, baseline_pred, labels=[0, 1]) if len(y_test_actual) > 0 else float("nan")
    try:
        baseline_auc = roc_auc_score(y_test_actual, baseline_pred)
    except Exception:
        baseline_auc = float("nan")

    calibration = None
    y_pred_calibrated = y_pred_proba
    if len(np.unique(y_test_actual)) > 1 and len(y_test_actual) >= 10:
        try:
            calib_model = LogisticRegression(max_iter=1000)
            calib_model.fit(y_pred_proba.reshape(-1, 1), y_test_actual)
            y_pred_calibrated = calib_model.predict_proba(y_pred_proba.reshape(-1, 1))[:, 1]
            cal_logloss = log_loss(y_test_actual, np.clip(y_pred_calibrated, 1e-6, 1 - 1e-6), labels=[0, 1])
            cal_auc = roc_auc_score(y_test_actual, y_pred_calibrated)
            cal_acc = accuracy_score(y_test_actual, (y_pred_calibrated > 0.5).astype(int))
            calibration = {
                "coef": float(calib_model.coef_[0][0]),
                "intercept": float(calib_model.intercept_[0]),
                "logloss": cal_logloss,
                "auc": cal_auc,
                "accuracy": cal_acc,
            }
            print(
                f"🔧 Calibration (Platt scaling) — Accuracy: {cal_acc:.3f}, "
                f"LogLoss: {cal_logloss:.3f}, AUC: {cal_auc:.3f}"
            )
        except Exception as exc:
            print(f"⚠️ Calibration failed: {exc}")

    raw_tag = args.tag or f"{args.traingate or 'ALL'}_{args.testgate or 'NA'}_{predictgate or 'NA'}"
    tag = normalize_tag(raw_tag)

    MODEL_DIR = Path("data/models")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR = Path("logs")
    LOG_DIR.mkdir(exist_ok=True)

    model_path = MODEL_DIR / f"nfl_model_{tag}.txt"
    gb_model_path = MODEL_DIR / f"gb_model_{tag}.joblib"
    featimp_path = MODEL_DIR / f"feature_importance_{tag}.csv"
    shap_path = MODEL_DIR / f"shap_summary_{tag}.csv"
    meta_path = MODEL_DIR / f"training_metadata_{tag}.json"
    schema_path = MODEL_DIR / "columns_schema.json"
    calibration_path = MODEL_DIR / f"calibration_{tag}.json"

    lgb_model.save_model(str(model_path))
    dump(gb_model, gb_model_path)
    importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": lgb_model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    importance.to_csv(featimp_path, index=False)

    try:
        explainer = shap.TreeExplainer(lgb_model)
        sample = X_train.sample(n=min(1000, len(X_train)), random_state=args.seed)
        shap_values = explainer.shap_values(sample)
        mean_abs = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({"feature": sample.columns, "mean_abs_shap": mean_abs})
        shap_df.sort_values("mean_abs_shap", ascending=False).to_csv(shap_path, index=False)
        print(f"📊 SHAP summary saved → {shap_path}")
    except Exception as exc:
        print(f"⚠️ SHAP computation failed: {exc}")

    schema = {col: str(dtype) for col, dtype in X_train.dtypes.items()}
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)

    if calibration:
        with open(calibration_path, "w") as f:
            json.dump(calibration, f, indent=2)
        print(f"🧮 Calibration coefficients saved → {calibration_path}")
    else:
        if calibration_path.exists():
            calibration_path.unlink()
        calibration_path = None

    meta = {
        "timestamp": dt.datetime.utcnow().isoformat(),
        "train_rows": len(df_train),
        "test_rows": len(df_test),
        "features": len(X_train.columns),
        "columns": list(X_train.columns),
        "accuracy": acc,
        "logloss": logloss,
        "auc": auc,
        "baseline_metrics": {
            "accuracy": baseline_acc,
            "logloss": baseline_logloss,
            "auc": baseline_auc,
        },
        "params": params_for_training,
        "num_boost_round": num_boost_round,
        "traingate": args.traingate,
        "testgate": args.testgate,
        "predictgate": predictgate,
        "rolling_seasons": args.rolling_seasons,
        "rolling_weeks": args.rolling_weeks,
        "cv_best_params": best_params if best_params else None,
        "calibration": calibration,
        "triage": triage_details,
        "ensemble": {
            "strategy": "mean",
            "members": [
                {"name": "lightgbm", "path": str(model_path)},
                {"name": "gradient_boosting", "path": str(gb_model_path)},
            ],
        },
        "shap_summary": str(shap_path),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Training complete — Model saved → {model_path}")
    print(f"📊 Feature importance → {featimp_path}")
    print(f"📈 SHAP summary → {shap_path}")
    print(f"📘 Metadata → {meta_path}")
    print(f"📚 Columns schema → {schema_path}")
    print(f"📈 Metrics — Accuracy: {acc:.3f}, LogLoss: {logloss:.3f}, AUC: {auc:.3f}")
    print(f"📊 Baseline — Accuracy: {baseline_acc:.3f}, LogLoss: {baseline_logloss:.3f}, AUC: {baseline_auc:.3f}")
    if calibration:
        print(
            f"📉 Calibrated — Accuracy: {calibration['accuracy']:.3f}, "
            f"LogLoss: {calibration['logloss']:.3f}, AUC: {calibration['auc']:.3f}"
        )

    if cv_results:
        cv_log_path = LOG_DIR / f"cv_results_{tag}.md"
        top5 = cv_results[:5]
        lines = [
            "# Hyperparameter Search Report",
            f"- Timestamp: {dt.datetime.utcnow().isoformat()}",
            f"- Train rows used: {len(df_train_test):,}",
            f"- Splits: {min(5, len(df_train_test) - 1)}",
            "",
            "| learning_rate | num_leaves | feature_fraction | bagging_fraction | lambda_l2 | min_data_in_leaf | logloss | auc | best_iteration |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for r in top5:
            lines.append(
                f"| {r['learning_rate']} | {r['num_leaves']} | {r['feature_fraction']} | "
                f"{r['bagging_fraction']} | {r['lambda_l2']} | {r['min_data_in_leaf']} | {r['logloss']:.5f} | {r['auc']:.5f} | {r['best_iteration']:.1f} |"
            )
        cv_log_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"📝 CV summary log → {cv_log_path}")


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Train NFL model with temporal gating (snake_case output)")
    parser.add_argument("--mode", choices=["train"], required=False)
    parser.add_argument("--traingate", type=str)
    parser.add_argument("--testgate", type=str)
    parser.add_argument("--predictgate", type=str)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--num_leaves", type=int, default=31)
    parser.add_argument("--max_depth", type=int, default=-1)
    parser.add_argument("--num_boost_round", type=int, default=1000)
    parser.add_argument("--early_stopping", type=int, default=50)
    parser.add_argument("--bagging_fraction", type=float, default=0.8)
    parser.add_argument("--feature_fraction", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable_triage", action="store_true", help="Skip automated feature triage before training.")
    parser.add_argument("--feature_triage_gain", type=float, default=5.0, help="Minimum LightGBM gain required to keep a feature during auto-triage.")
    parser.add_argument("--feature_triage_top_n", type=int, default=120, help="Fallback maximum number of features to retain during auto-triage.")
    parser.add_argument("--tag", type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--rolling_seasons", type=int, help="Restrict training data to the most recent N seasons.")
    parser.add_argument("--rolling_weeks", type=int, help="Restrict training data to the most recent N weeks across seasons.")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        print("\nExamples:")
        print("  python scripts/05_train_model.py --mode train --traingate 2025W7 --testgate 2025W7 --predictgate '2025 W8' --tag apple")
        print("  python scripts/05_train_model.py --mode train --traingate 2024W16 --testgate 2024W17 --predictgate '2024 W18'")
        print("  python scripts/05_train_model.py --mode train --rolling_weeks 200 --testgate 2025W7 --predictgate '2025 W8'")
        sys.exit(0)

    args = parser.parse_args()
    if args.mode == "train":
        train_model(args)


if __name__ == "__main__":
    main()
