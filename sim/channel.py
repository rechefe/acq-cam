"""AWGN + CFO + phase + timing-offset channel model. Optional flat Rayleigh fading
and CFO drift for secondary robustness figures.
"""
from __future__ import annotations

import numpy as np

from .tx import SYMBOL_RATE


def apply_cfo(signal: np.ndarray, df: float, osr: int, symbol_rate: float = SYMBOL_RATE,
              drift_rate: float = 0.0) -> np.ndarray:
    """Apply a carrier frequency offset df (Hz). Optional linear drift_rate (Hz per
    sample) for the CFO-drift robustness study."""
    ts = 1.0 / (symbol_rate * osr)
    n = np.arange(len(signal))
    inst_df = df + drift_rate * n
    phase_ramp = 2 * np.pi * np.cumsum(inst_df) * ts if drift_rate != 0.0 else 2 * np.pi * df * n * ts
    return signal * np.exp(1j * phase_ramp)


def apply_phase_offset(signal: np.ndarray, theta0: float) -> np.ndarray:
    return signal * np.exp(1j * theta0)


def apply_timing_offset(signal: np.ndarray, frac_offset: float) -> np.ndarray:
    """Fractional sub-sample delay in [0, 1) samples via linear interpolation between
    adjacent samples. Adequate at the OSR used here (>=1); the signal is band-limited
    by the Gaussian pulse shape so linear interpolation introduces only a small,
    consistent-with-hardware distortion.
    """
    if frac_offset == 0.0:
        return signal.copy()
    shifted = np.empty_like(signal)
    shifted[:-1] = (1 - frac_offset) * signal[:-1] + frac_offset * signal[1:]
    shifted[-1] = signal[-1]
    return shifted


def apply_rayleigh_fading(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Flat Rayleigh fading: single complex gain applied to the whole packet."""
    gain = (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2)
    return signal * gain


def awgn_noise_variance(ebn0_db: float, osr: int, bits_per_symbol: float = 1.0) -> float:
    """Per-complex-sample noise variance for a unit-envelope (|s|=1) GFSK signal.

    Es (energy/symbol) = osr * 1 (unit power over osr samples).
    Eb = Es / bits_per_symbol.
    N0 = Eb / (10^(EbN0dB/10)).
    Complex AWGN variance per sample = N0 (split N0/2 across I and Q).
    """
    ebn0_lin = 10.0 ** (ebn0_db / 10.0)
    es = osr * 1.0
    eb = es / bits_per_symbol
    n0 = eb / ebn0_lin
    return n0


def add_awgn(signal: np.ndarray, ebn0_db: float, osr: int, rng: np.random.Generator,
             bits_per_symbol: float = 1.0) -> np.ndarray:
    n0 = awgn_noise_variance(ebn0_db, osr, bits_per_symbol)
    std_per_rail = np.sqrt(n0 / 2.0)
    noise = std_per_rail * (rng.standard_normal(len(signal)) + 1j * rng.standard_normal(len(signal)))
    return signal + noise


def run_channel(tx: np.ndarray, df: float, osr: int, ebn0_db: float, rng: np.random.Generator,
                 theta0: float | None = None, timing_frac: float | None = None,
                 fading: bool = False, drift_rate: float = 0.0,
                 bits_per_symbol: float = 1.0) -> np.ndarray:
    """Full receive-side channel: CFO -> timing offset -> phase offset -> (fading) -> AWGN.

    theta0 and timing_frac are randomized per-call if not supplied.
    """
    if theta0 is None:
        theta0 = rng.uniform(0.0, 2 * np.pi)
    if timing_frac is None:
        timing_frac = rng.uniform(0.0, 1.0)

    sig = apply_cfo(tx, df, osr, drift_rate=drift_rate)
    sig = apply_timing_offset(sig, timing_frac)
    sig = apply_phase_offset(sig, theta0)
    if fading:
        sig = apply_rayleigh_fading(sig, rng)
    sig = add_awgn(sig, ebn0_db, osr, rng, bits_per_symbol=bits_per_symbol)
    return sig
