"""Segmented match lines and the multi-dwell fire accumulator.

Splitting the W-bit match line into S sub-lines, each with its own threshold, and
combining them with a >= k-of-S popcount is what shortens the coherent interval --
and therefore what shrinks the Doppler dimension of the dictionary
(n_doppler = 4*f_max*T_coh + 1).

The trade is real and runs both ways. Hamming distance is *additive* across
segments, so summing segment distances would be exactly the unsegmented 1 ms
coherent detector and would need the full 41 Doppler rows. The Doppler tolerance
appears only because each segment is thresholded *before* combining -- hard
decisions, which break coherence between segments and cost sensitivity.

Everything here consumes booleans. `CAM.query` (and its private batch form) is the
only thing that ever sees a distance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..cam import CAM


class SegmentedCAM:
    """S independent CAMs over contiguous slices of the same rows."""

    def __init__(self, rows: np.ndarray, n_segments: int = 1):
        rows = np.asarray(rows).astype(bool)
        self.N, self.W = rows.shape
        if self.W % n_segments:
            raise ValueError(f"W={self.W} not divisible by n_segments={n_segments}")
        self.S = n_segments
        self.seg_width = self.W // n_segments
        self._cams = [CAM(rows[:, i * self.seg_width:(i + 1) * self.seg_width])
                      for i in range(n_segments)]

    def fires(self, keys: np.ndarray, tau: int) -> np.ndarray:
        """(M, W) keys -> (M, N, S) boolean per-segment fires."""
        keys = np.atleast_2d(np.asarray(keys))
        out = np.empty((len(keys), self.N, self.S), dtype=bool)
        for i, cam in enumerate(self._cams):
            out[:, :, i] = cam._query_batch(keys[:, i * self.seg_width:(i + 1) * self.seg_width], tau)
        return out


@dataclass
class Detection:
    fired: np.ndarray          # (N,) bool -- rows declared present
    fire_count: np.ndarray     # (N,) int  -- segments fired, summed over dwells


def detect(scam: SegmentedCAM, keys_per_dwell: list[np.ndarray], tau: int, k: int) -> Detection:
    """Accumulate per-segment fires over theta0 quadrants and dwells.

    `keys_per_dwell[d]` is the (4, W) stack of the four free 90-degree rotations of
    dwell d's key. Within a dwell the quadrants are OR-ed (the boolean stand-in for
    the magnitude); across dwells the segment fires are summed, which is the
    non-coherent accumulation the link budget requires anyway.
    """
    total = np.zeros(scam.N, dtype=int)
    for keys in keys_per_dwell:
        f = scam.fires(keys, tau)          # (4, N, S)
        total += f.any(axis=0).sum(axis=1)  # OR over quadrants, count segments
    return Detection(total >= k, total)
