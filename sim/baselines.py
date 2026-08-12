"""Two conventional baselines for comparison against the CAM, operating on the
same received samples:

B1 -- Sequential CFO search: one correlator, tries each of N hypotheses in
turn. Latency = N cycles. Area = 1 correlator.
B2 -- Parallel correlator bank: N full correlators. Latency = 1 cycle.
Area = N correlators (each N_s complex multiply-accumulates).

Both compute the identical correlation magnitudes (the only difference between
B1 and B2 is *when* those N products are formed -- serially vs. in parallel --
not what is computed), so a single correlate_bank() is shared and the two
baselines differ only in their reported latency/area bookkeeping.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tx import tx_waveform, STANDARD_ACCESS_ADDRESS
from .channel import apply_cfo
from .codebook import cfo_grid


@dataclass
class CorrelatorBank:
    waveforms: np.ndarray  # (N, N_s) complex, unit-energy-normalized
    df_grid: np.ndarray    # (N,)
    N_s: int


def build_correlator_bank(N: int | None = None, delta: float | None = None,
                           df_min: float = -150e3, df_max: float = 150e3,
                           n_sym: int = 40, osr: int = 4,
                           access_address: int = STANDARD_ACCESS_ADDRESS) -> CorrelatorBank:
    grid = cfo_grid(df_min, df_max, delta if delta is not None else (df_max - df_min) / max(N - 1, 1))
    if N is not None:
        grid = grid[:N]
    tx = tx_waveform(n_sym=n_sym, osr=osr, access_address=access_address)
    waveforms = np.stack([apply_cfo(tx, df, osr) for df in grid])
    return CorrelatorBank(waveforms=waveforms, df_grid=grid, N_s=tx.size)


def correlate_bank(rx: np.ndarray, bank: CorrelatorBank) -> np.ndarray:
    """|sum(rx * conj(ref_k))| / N_s for each hypothesis k -- a normalized
    complex correlation magnitude, insensitive to theta0 (magnitude only)."""
    corr = bank.waveforms @ np.conj(rx)
    return np.abs(corr) / bank.N_s


@dataclass
class BaselineResult:
    detected: bool
    df_hat: float
    latency_cycles: int
    area_proxy: int  # complex MACs


def b1_sequential(rx: np.ndarray, bank: CorrelatorBank, threshold: float) -> BaselineResult:
    mags = correlate_bank(rx, bank)
    k_hat = int(np.argmax(mags))
    detected = bool(mags[k_hat] >= threshold)
    N = bank.df_grid.size
    return BaselineResult(detected=detected, df_hat=float(bank.df_grid[k_hat]) if detected else float("nan"),
                           latency_cycles=N, area_proxy=1 * bank.N_s)


def b2_parallel(rx: np.ndarray, bank: CorrelatorBank, threshold: float) -> BaselineResult:
    mags = correlate_bank(rx, bank)
    k_hat = int(np.argmax(mags))
    detected = bool(mags[k_hat] >= threshold)
    N = bank.df_grid.size
    return BaselineResult(detected=detected, df_hat=float(bank.df_grid[k_hat]) if detected else float("nan"),
                           latency_cycles=1, area_proxy=N * bank.N_s)
