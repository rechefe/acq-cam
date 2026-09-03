"""Codebook construction: PRN x Doppler rows.

Placement principle: sweep the rotations that are free in the quantized domain,
store the ones that are not. A theta0 step of 90 degrees is a bit permutation
(`quantize.rotate_key`), so it is swept and costs no rows. A Doppler hypothesis is
a phase *ramp* -- not free on a 2-bit key -- but costs nothing on a precomputed
row, so it is stored.

The Doppler row count follows from the coherent segment length alone. A row stays
matched while the residual ramp stays under half a cycle across the coherent
window, so the spacing is df = 1/(2*T_coh) and a symmetric grid covering +-f_max
needs

    n_doppler = 2*ceil(2 * f_max * T_coh) + 1

(the un-rounded 4*f_max*T_coh + 1 understates it whenever f_max is not a whole
number of bins -- e.g. +-10 kHz at T_coh = 125 us needs 7 rows, not 6).

Note that T_coh is not freely choosable: a segment must be a whole number of
samples, so the segment count S must divide the samples per code period. At
OSR = 2 that is 2046 = 2*3*11*31, giving S in {2, 3, 6, 11, 22, 33} -- not powers
of two.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .channel import replica
from .prn import NUM_PRNS
from .quantize import full_scale_key, key_width, row_state_bits

# Rows are built at theta = 45 deg, NOT 0. At theta_k = 0 the replica is purely
# real, so the row's Q rail degenerates to a constant and contributes variance
# without signal -- half the match line is dead. At 45 deg both rails carry the
# code. The key is then swept over the four free 90 deg rotations, so the
# residual phase error is at most 45 deg: 0-3 dB of ripple, never blind.
ROW_THETA = np.pi / 4


def doppler_spacing(t_coh: float) -> float:
    return 1.0 / (2.0 * t_coh)


def doppler_grid(f_max: float, t_coh: float) -> np.ndarray:
    """Symmetric grid at spacing 1/(2*T_coh) covering +-f_max."""
    df = doppler_spacing(t_coh)
    k = int(np.ceil(f_max / df))
    return df * np.arange(-k, k + 1)


def segment_counts(n_samples: int) -> list[int]:
    """Segment counts that divide `n_samples` evenly -- a segment cannot be a
    fractional number of samples."""
    return [s for s in range(1, n_samples + 1) if n_samples % s == 0]


@dataclass
class GpsCodebook:
    rows: np.ndarray          # (N, W) bool
    prn_of_row: np.ndarray    # (N,) int
    fd_of_row: np.ndarray     # (N,) Hz
    W: int
    N: int
    n_samples: int
    mapping: str
    fd_grid: np.ndarray

    @property
    def dictionary_bits(self) -> int:
        """Stored state. Rows are full scale, so the thermometer mapping widens
        the match line without widening the memory."""
        return self.N * row_state_bits(self.n_samples)

    @property
    def match_line_cells(self) -> int:
        return self.N * self.W

    def rows_for(self, prn: int) -> np.ndarray:
        return np.flatnonzero(self.prn_of_row == prn)


def build_codebook(prns: list[int] | None = None, f_max: float = 10e3,
                   t_coh: float = 1e-3, fs: float = 2.046e6,
                   n_samples: int | None = None, mapping: str = "sign") -> GpsCodebook:
    """One row per (PRN, Doppler), each the full-scale noiseless replica at code
    phase 0 and theta0 = 0."""
    if prns is None:
        prns = list(range(1, NUM_PRNS + 1))
    if n_samples is None:
        n_samples = int(round(fs * t_coh))
    grid = doppler_grid(f_max, t_coh)
    W = key_width(n_samples, mapping)

    rows = np.zeros((len(prns) * len(grid), W), dtype=bool)
    prn_of = np.zeros(len(rows), dtype=int)
    fd_of = np.zeros(len(rows))
    i = 0
    for p in prns:
        for fd in grid:
            rows[i] = full_scale_key(replica(p, fd, fs, n_samples, theta=ROW_THETA),
                                     mapping).astype(bool)
            prn_of[i], fd_of[i] = p, fd
            i += 1
    return GpsCodebook(rows, prn_of, fd_of, W, len(rows), n_samples, mapping, grid)
