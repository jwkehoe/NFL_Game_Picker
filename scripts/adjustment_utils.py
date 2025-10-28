#!/usr/bin/env python3
"""
Adjustment utilities for residual models.

Currently exposes a helper that limits residual adjustments to cases
where the model deviates meaningfully from the market baseline.
"""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np

_DEFAULT_THRESHOLD = float(os.getenv("AUTO_NFL_EDGE_THRESHOLD", "0.05"))
_DEFAULT_THRESHOLD_POS = float(os.getenv("AUTO_NFL_EDGE_THRESHOLD_POS", "0.10"))
_DEFAULT_THRESHOLD_NEG = float(os.getenv("AUTO_NFL_EDGE_THRESHOLD_NEG", "0.12"))
_EDGE_SCALE = float(os.getenv("AUTO_NFL_EDGE_SCALE", "1.0"))
_MAX_FLIP_OFFSET = float(os.getenv("AUTO_NFL_MAX_FLIP_BASELINE_OFFSET", "0.05"))


def apply_edge_threshold(
    predicted: Iterable[float],
    baseline: Iterable[float],
    threshold: float | None = None,
    threshold_pos: float | None = None,
    threshold_neg: float | None = None,
) -> np.ndarray:
    """
    Clamp adjusted win probabilities so that only residuals with magnitude
    greater than the configured threshold survive. Smaller deviations fall back
    to the original baseline probability.

    Parameters
    ----------
    predicted : array-like
        Raw model probabilities (baseline + residual).
    baseline : array-like
        Baseline probabilities (e.g., market implied).
    threshold : float, optional
        Minimum absolute difference required to keep the adjustment (applied to
        both directions). Overrides ``threshold_pos``/``threshold_neg`` when set.
    threshold_pos : float, optional
        Threshold applied when ``predicted >= baseline`` (default 0.10 or
        ``AUTO_NFL_EDGE_THRESHOLD_POS``).
    threshold_neg : float, optional
        Threshold applied when ``predicted < baseline`` (default 0.12 or
        ``AUTO_NFL_EDGE_THRESHOLD_NEG``).

    Returns
    -------
    numpy.ndarray
        Thresholded probabilities clipped to [1e-6, 1 - 1e-6].
    """
    if threshold is not None:
        thresh_pos = thresh_neg = float(threshold)
    else:
        thresh_pos = (
            _DEFAULT_THRESHOLD_POS if threshold_pos is None else float(threshold_pos)
        )
        thresh_neg = (
            _DEFAULT_THRESHOLD_NEG if threshold_neg is None else float(threshold_neg)
        )

    thresh_abs = _DEFAULT_THRESHOLD if threshold is None else float(threshold)
    pred = np.asarray(predicted, dtype=float)
    base = np.asarray(baseline, dtype=float)

    if pred.shape != base.shape:
        raise ValueError("Predicted and baseline arrays must have the same shape.")

    delta = pred - base
    if _EDGE_SCALE != 1.0:
        pred = base + delta * _EDGE_SCALE
        delta = pred - base
    mask = np.abs(delta) >= (thresh_abs if threshold is not None else 0.0)

    gated = base.copy()
    if threshold is not None:
        gated[mask] = pred[mask]
    else:
        pos_mask = (delta >= 0) & (np.abs(delta) >= thresh_pos)
        neg_mask = (delta < 0) & (np.abs(delta) >= thresh_neg)
        keep_mask = pos_mask | neg_mask
        if _MAX_FLIP_OFFSET < 1.0:
            candidate = base.copy()
            candidate[keep_mask] = pred[keep_mask]
            flip_mask = ((base >= 0.5) & (candidate < 0.5)) | ((base < 0.5) & (candidate >= 0.5))
            allow_flip = flip_mask & (np.abs(base - 0.5) <= _MAX_FLIP_OFFSET)
            # Suppress disallowed flips by removing them from masks
            block_mask = flip_mask & ~allow_flip
            if block_mask.any():
                pos_mask = pos_mask & ~block_mask
                neg_mask = neg_mask & ~block_mask
        gated[pos_mask] = pred[pos_mask]
        gated[neg_mask] = pred[neg_mask]

    return np.clip(gated, 1e-6, 1 - 1e-6)
