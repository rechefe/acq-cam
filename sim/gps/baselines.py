"""Conventional acquisition baselines.

B1 -- FFT parallel-code-phase search. The standard GNSS acquisition engine and
      the real competitor: per (PRN, Doppler) it wipes off the carrier, then
      gets all 2046 code phases from one forward FFT, a conjugate multiply and
      one inverse FFT. Full precision, non-coherent |.|^2 detection.
B2 -- the same search driven from the 2-bit quantized samples, which isolates
      how much of any CAM gap is quantization rather than the boolean readout.

Cost accounting for the area/throughput comparison is in `fft_complex_macs`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .channel import code_doppler_chips_per_sample
from .prn import ca_code_bipolar, upsample
from .quantize import DEFAULT_THRESHOLD, LEVELS, adc_2bit


@dataclass
class AcqResult:
    prn: int
    code_phase_samples: int
    fd: float
    peak: float
    metric: float          # peak / second-highest, the usual acquisition metric
    grid: np.ndarray       # (n_doppler, n_samples) correlation power


def code_replica_fft(prn: int, fs: float, n_samples: int) -> np.ndarray:
    """Conjugate spectrum of the code, cached per call site by the caller."""
    cps = code_doppler_chips_per_sample(0.0, fs)
    c = upsample(ca_code_bipolar(prn), osr=None, code_phase_chips=0.0,
                 n_samples=n_samples, chips_per_sample=cps)
    return np.conj(np.fft.fft(c))


def acquire_fft(samples: np.ndarray, prn: int, fd_grid: np.ndarray, fs: float,
                quantize: bool = False,
                threshold: float = DEFAULT_THRESHOLD) -> AcqResult:
    """Parallel-code-phase search over `fd_grid` for one PRN."""
    r = samples
    if quantize:
        r = (LEVELS[adc_2bit(samples.real, 1.0, threshold)]
             + 1j * LEVELS[adc_2bit(samples.imag, 1.0, threshold)])
    n = len(r)
    code_f = code_replica_fft(prn, fs, n)
    t = np.arange(n) / fs
    grid = np.empty((len(fd_grid), n))
    for i, fd in enumerate(fd_grid):
        wiped = r * np.exp(-1j * 2.0 * np.pi * fd * t)
        grid[i] = np.abs(np.fft.ifft(np.fft.fft(wiped) * code_f)) ** 2

    idx = np.unravel_index(np.argmax(grid), grid.shape)
    peak = grid[idx]
    # exclude +-1 chip around the peak when finding the runner-up
    guard = int(round(2 * fs / 1.023e6))
    masked = grid.copy()
    lo, hi = idx[1] - guard, idx[1] + guard + 1
    masked[:, max(0, lo):hi] = 0.0
    if lo < 0:
        masked[:, lo:] = 0.0
    if hi > n:
        masked[:, :hi - n] = 0.0
    second = masked.max()
    return AcqResult(prn, int(idx[1]), float(fd_grid[idx[0]]), float(peak),
                     float(peak / second) if second > 0 else np.inf, grid)


def fft_complex_macs(n_samples: int, n_doppler: int, n_prn: int) -> int:
    """Complex MACs for one full 2-D search: per (PRN, Doppler) a forward FFT, a
    conjugate multiply and an inverse FFT. The code spectrum is precomputed."""
    per_fft = (n_samples / 2) * np.log2(n_samples)
    return int(n_prn * n_doppler * (2 * per_fft + n_samples))


def dwells_required_fft(cn0_dbhz: float, t_coh: float, n_cells: int,
                        target_pd: float = 0.9, system_pfa: float = 0.01,
                        max_dwells: int = 500) -> tuple[int, float]:
    """Dwells the full-precision FFT search needs for the same targets.

    Non-coherent accumulation of M coherent correlations: normalising so the
    per-dwell noise is unit-variance per component, the statistic is chi2 with
    2M dof under H0 and non-central chi2 with non-centrality M*2*(C/N0)*T under
    H1. Returns (M, threshold).
    """
    from scipy.stats import chi2, ncx2
    lam = 2.0 * 10.0 ** (cn0_dbhz / 10.0) * t_coh
    for m in range(1, max_dwells + 1):
        thr = chi2.isf(system_pfa / n_cells, 2 * m)
        if ncx2.sf(thr, 2 * m, m * lam) >= target_pd:
            return m, float(thr)
    return -1, float("nan")
