"""GPS L1 C/A signal generation at a specified C/N0.

r[n] = A*c[n - tau]*d*exp(j(2*pi*f_d*n*Ts + theta0)) + w[n]

with w complex Gaussian of variance sigma^2 per rail. Fixing sigma = 1 and
setting A from C/N0 keeps the ADC thresholds fixed across the whole sweep, which
is what a real front end with a settled AGC does.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .prn import CODE_LENGTH, CHIP_RATE, L1_CARRIER, ca_code_bipolar, upsample

SIGMA = 1.0


def amplitude_for_cn0(cn0_dbhz: float, fs: float) -> float:
    """Signal amplitude giving `cn0_dbhz` against unit-variance-per-rail noise."""
    return float(np.sqrt(2.0 * SIGMA ** 2 * 10.0 ** (cn0_dbhz / 10.0) / fs))


def code_doppler_chips_per_sample(fd: float, fs: float) -> float:
    """Chips advanced per sample including code Doppler (fd/L1 fractional rate)."""
    return (CHIP_RATE / fs) * (1.0 + fd / L1_CARRIER)


@dataclass
class Signal:
    samples: np.ndarray
    prn: int
    fd: float
    theta0: float
    code_phase_chips: float


def generate(prn: int, cn0_dbhz: float, fd: float, fs: float, n_samples: int,
             theta0: float | None = None, code_phase_chips: float | None = None,
             rng: np.random.Generator | None = None, noise: bool = True,
             data_bit: int = 1) -> Signal:
    """One realization of the received signal."""
    rng = rng or np.random.default_rng()
    if theta0 is None:
        theta0 = float(rng.uniform(0.0, 2.0 * np.pi))
    if code_phase_chips is None:
        code_phase_chips = float(rng.uniform(0.0, CODE_LENGTH))

    cps = code_doppler_chips_per_sample(fd, fs)
    chips = upsample(ca_code_bipolar(prn), osr=None, code_phase_chips=code_phase_chips,
                     n_samples=n_samples, chips_per_sample=cps)
    n = np.arange(n_samples)
    carrier = np.exp(1j * (2.0 * np.pi * fd * n / fs + theta0))
    s = amplitude_for_cn0(cn0_dbhz, fs) * data_bit * chips * carrier

    if noise:
        w = rng.normal(0.0, SIGMA, n_samples) + 1j * rng.normal(0.0, SIGMA, n_samples)
        s = s + w
    return Signal(s, prn, fd, theta0, code_phase_chips)


def replica(prn: int, fd: float, fs: float, n_samples: int,
            code_phase_chips: float = 0.0, theta: float = 0.0) -> np.ndarray:
    """Noiseless unit-amplitude replica -- the thing a codebook row encodes."""
    cps = code_doppler_chips_per_sample(fd, fs)
    chips = upsample(ca_code_bipolar(prn), osr=None, code_phase_chips=code_phase_chips,
                     n_samples=n_samples, chips_per_sample=cps)
    n = np.arange(n_samples)
    return chips * np.exp(1j * (2.0 * np.pi * fd * n / fs + theta))
