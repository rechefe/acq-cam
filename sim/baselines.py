"""Three conventional baselines for comparison against the CAM, operating on
the same received samples:

B1 -- Sequential CFO search: one full-precision complex correlator, tries
each of N hypotheses in turn. Latency = N cycles. Area = 1 correlator.
B2 -- Parallel full-precision correlator bank: N correlators. Latency =
1 cycle. Area = N correlators (each N_s complex multiply-accumulates).
B3 -- Parallel 1-bit sign-sign correlator bank: same N hypotheses, same
1-cycle latency, but both the received signal and the reference waveforms
are quantized to 1-bit I/Q before correlating. A sign-sign product needs no
real multiplier (it is an XNOR/sign-flip), so this is the honest low-power
peer to the binary CAM -- B1/B2 use full-precision arithmetic the CAM was
never given, which is an unfair comparison on its own.

B1 and B2 compute the identical correlation magnitudes (the only difference
between them is *when* those N products are formed -- serially vs. in
parallel -- not what is computed), so a single correlate_bank() is shared
and the two baselines differ only in their reported latency/area
bookkeeping. B3 shares the same structure via correlate_bank_sign().
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


def sign_quantize(x: np.ndarray) -> np.ndarray:
    """1-bit I/Q quantization: sign(Re(x)) + j*sign(Im(x)), each component in
    {-1, +1} -- the cheapest possible ADC, and the natural front end for a
    sign-sign correlator (no real multiplier needed downstream)."""
    re = np.where(x.real >= 0, 1.0, -1.0)
    im = np.where(x.imag >= 0, 1.0, -1.0)
    return re + 1j * im


@dataclass
class SignCorrelatorBank:
    waveforms_q: np.ndarray  # (N, N_s) complex, each component in {-1,+1}
    df_grid: np.ndarray
    N_s: int


def build_sign_correlator_bank(bank: CorrelatorBank) -> SignCorrelatorBank:
    """Derive a 1-bit sign-sign correlator bank from a full-precision
    CorrelatorBank -- same hypothesis waveforms, quantized once at build
    time (a fixed, zero-cost hardware step; the reference patterns are
    known at design time, not computed at runtime)."""
    return SignCorrelatorBank(waveforms_q=sign_quantize(bank.waveforms),
                               df_grid=bank.df_grid, N_s=bank.N_s)


def correlate_bank_sign(rx: np.ndarray, sign_bank: SignCorrelatorBank) -> np.ndarray:
    """Sign-sign correlation magnitude for each hypothesis: both rx and the
    reference are 1-bit quantized, so each per-sample product is one of 4
    fixed {-1,+1}x{-1,+1} outcomes -- an XNOR/sign-flip, not a real multiply.
    """
    rx_q = sign_quantize(rx)
    corr = sign_bank.waveforms_q @ np.conj(rx_q)
    return np.abs(corr) / sign_bank.N_s


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


def b3_parallel_sign(rx: np.ndarray, sign_bank: SignCorrelatorBank, threshold: float) -> BaselineResult:
    """1-bit sign-sign correlator bank: same latency as B2 (1 cycle, N
    correlators in parallel), but area_proxy counts XNOR/sign-flip ops
    instead of complex MACs -- NOT directly comparable in magnitude to
    B1/B2's area_proxy (a real multiply and an XNOR are very different gate
    counts), only in op *count*. Reported alongside, not merged with, the
    complex-MAC proxy so the comparison stays honest (see docs/baseline_b3_findings.md)."""
    mags = correlate_bank_sign(rx, sign_bank)
    k_hat = int(np.argmax(mags))
    detected = bool(mags[k_hat] >= threshold)
    N = sign_bank.df_grid.size
    return BaselineResult(detected=detected,
                           df_hat=float(sign_bank.df_grid[k_hat]) if detected else float("nan"),
                           latency_cycles=1, area_proxy=N * sign_bank.N_s)
