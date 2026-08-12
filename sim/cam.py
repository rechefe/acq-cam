"""The CAM primitive -- and nothing else.

Stores N rows of W-bit words (optionally with a per-bit don't-care mask, i.e. a
ternary CAM). Given a W-bit query key, returns in "one cycle" an N-bit match
vector: m[k] = 1 iff hamming(key, row_k) <= tau over the non-masked bits.

This module is the ONLY place in the codebase allowed to compare a query key
against stored rows. Its public API returns booleans only -- no method here
returns a distance, a rank, or an argmin. Downstream code (decode.py) must
consume nothing but the match vector this class hands back.
"""
from __future__ import annotations

import numpy as np


class CAM:
    def __init__(self, rows: np.ndarray, mask: np.ndarray | None = None):
        """rows: (N, W) binary (0/1 or bool) array, one row per stored hypothesis.
        mask: optional (N, W) boolean array, True where that bit is a don't-care
        for that row (excluded from the Hamming count -- ternary CAM behaviour).
        """
        rows = np.asarray(rows)
        if rows.ndim != 2:
            raise ValueError("rows must be a 2D (N, W) array")
        self._rows = rows.astype(bool)
        self.N, self.W = self._rows.shape
        if mask is not None:
            mask = np.asarray(mask)
            if mask.shape != self._rows.shape:
                raise ValueError("mask must have the same shape as rows")
            self._mask = mask.astype(bool)
        else:
            self._mask = None

    def query(self, key: np.ndarray, tau: int) -> np.ndarray:
        """Return the N-bit boolean match vector for a single query key.

        This is the ONLY read primitive. It performs the per-row Hamming
        comparison internally (as the analog sense-amp array would) and returns
        strictly a boolean vector -- the intermediate per-row distance is a
        local value inside this function, never returned or stored.
        """
        key = np.asarray(key).astype(bool)
        if key.shape != (self.W,):
            raise ValueError(f"key must have shape ({self.W},), got {key.shape}")
        mismatch = self._rows != key[None, :]
        if self._mask is not None:
            mismatch = mismatch & (~self._mask)
        distance = mismatch.sum(axis=1)
        match = distance <= tau
        return match
