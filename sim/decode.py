"""Decoders: consume ONLY the CAM's binary match vector (plus the codebook's
df_grid, which is a fixed design-time constant, not a per-trial distance).

The one exception is argmin_baseline, which is explicitly a reference-only
upper bound computed from true Hamming distances -- it is kept out of the
CAM/decode data path used by the rest of the pipeline and is clearly labeled.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def any_fire(m: np.ndarray) -> bool:
    return bool(np.any(m))


def _longest_run(m: np.ndarray):
    """Return (start, end) inclusive indices of the longest contiguous run of
    True values in m, or None if m is all False. Ties broken by first run."""
    if not np.any(m):
        return None
    best_len = 0
    best = None
    run_start = None
    for i, v in enumerate(m):
        if v:
            if run_start is None:
                run_start = i
            run_len = i - run_start + 1
            if run_len > best_len:
                best_len = run_len
                best = (run_start, i)
        else:
            run_start = None
    return best


@dataclass
class DecodeResult:
    detected: bool
    df_hat: float
    run_length: int


def run_midpoint(m: np.ndarray, df_grid: np.ndarray) -> DecodeResult:
    """Primary proposed estimator: find the longest contiguous run of 1s and
    estimate df_hat as the midpoint of that run's endpoints (sub-grid
    resolution comes from the midpoint of the *span*, not a lookup)."""
    run = _longest_run(m)
    if run is None:
        return DecodeResult(detected=False, df_hat=float("nan"), run_length=0)
    lo, hi = run
    df_hat = (df_grid[lo] + df_grid[hi]) / 2.0
    run_length = hi - lo + 1
    return DecodeResult(detected=True, df_hat=df_hat, run_length=run_length)


def centroid(m: np.ndarray, df_grid: np.ndarray) -> DecodeResult:
    """df_hat = mean(df_k for k where m[k]==1). Differs from run_midpoint when
    the fired set is fragmented (noise breaks the run into multiple islands)."""
    if not np.any(m):
        return DecodeResult(detected=False, df_hat=float("nan"), run_length=0)
    df_hat = float(np.mean(df_grid[m]))
    run = _longest_run(m)
    run_length = (run[1] - run[0] + 1) if run is not None else 0
    return DecodeResult(detected=True, df_hat=df_hat, run_length=run_length)


def is_fragmented(m: np.ndarray) -> bool:
    """True iff the fired set consists of more than one contiguous island."""
    if not np.any(m):
        return False
    islands = 0
    prev = False
    for v in m:
        if v and not prev:
            islands += 1
        prev = bool(v)
    return islands > 1


def argmin_baseline(rows: np.ndarray, key: np.ndarray, df_grid: np.ndarray,
                     mask: np.ndarray | None = None) -> DecodeResult:
    """Reference-only, unachievable-in-hardware upper bound: true argmin over
    per-row Hamming distances. NOT part of the CAM data path -- operates
    directly on rows/key, bypassing CAM.query() entirely, and is labeled as
    such everywhere it is used. Included purely to quantify what the binary
    readout costs relative to full-precision distance information.
    """
    rows_b = rows.astype(bool)
    key_b = key.astype(bool)
    mismatch = rows_b != key_b[None, :]
    if mask is not None:
        mismatch = mismatch & (~mask.astype(bool))
    distance = mismatch.sum(axis=1)
    k_hat = int(np.argmin(distance))
    return DecodeResult(detected=True, df_hat=float(df_grid[k_hat]), run_length=1)
