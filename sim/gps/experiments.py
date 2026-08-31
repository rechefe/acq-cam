"""Monte Carlo sweeps for the GPS CAM acquisition study.

Measurement strategy: rather than sweeping all 2046 code phases every trial, we
measure the *per-cell* detection and false-alarm rates -- at the aligned cell, and
at wrong-code-phase / wrong-Doppler / wrong-PRN cells -- and combine them
analytically for the full 2-D search. That is both far cheaper and more precise
at the small probabilities the search-space size demands.

Calibration follows the trap documented in `docs/findings.md` for the BLE work:
tau is calibrated against a *structured* H0 (a real signal at the wrong cell) at
the HIGHEST C/N0 in the sweep, not against noise at a mid-SNR point. A clean
wrong-cell key can sit structurally closer to a row than noise ever does, so
false-alarm rates can get worse as C/N0 rises.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, asdict

import numpy as np

from . import channel, codebook
from .prn import CODE_LENGTH, NUM_PRNS
from .quantize import key_width, quantize_to_key, rotate_key, row_state_bits
from .segment import SegmentedCAM

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "cache")
N_QUADRANTS = 4


@dataclass
class GpsConfig:
    fs: float = 2.046e6
    f_max: float = 10e3
    n_segments: int = 1
    mapping: str = "sign"
    n_prn: int = 32
    threshold: float = 1.0

    @property
    def n_samples(self) -> int:
        return int(round(self.fs * CODE_LENGTH / 1.023e6))

    @property
    def t_coh(self) -> float:
        return self.n_samples / self.n_segments / self.fs

    @property
    def prns(self) -> list[int]:
        return list(range(1, self.n_prn + 1))

    def as_dict(self) -> dict:
        return {**asdict(self), "t_coh": self.t_coh, "n_samples": self.n_samples}


def cached(name: str, params: dict, compute_fn, force: bool = False) -> dict:
    os.makedirs(CACHE_DIR, exist_ok=True)
    blob = json.dumps(params, sort_keys=True, default=str).encode()
    path = os.path.join(CACHE_DIR, f"{name}_{hashlib.sha1(blob).hexdigest()[:16]}.npz")
    if os.path.exists(path) and not force:
        data = np.load(path, allow_pickle=True)
        return {k: data[k] for k in data.files}
    result = compute_fn()
    np.savez(path, **result)
    return result


def build(cfg: GpsConfig) -> tuple[codebook.GpsCodebook, SegmentedCAM]:
    cb = codebook.build_codebook(prns=cfg.prns, f_max=cfg.f_max, t_coh=cfg.t_coh,
                                 fs=cfg.fs, n_samples=cfg.n_samples, mapping=cfg.mapping)
    return cb, SegmentedCAM(cb.rows, cfg.n_segments)


def dwell_keys(samples: np.ndarray, cfg: GpsConfig) -> np.ndarray:
    """The four free 90-degree rotations of one dwell's key: (4, W)."""
    k = quantize_to_key(samples, 1.0, cfg.mapping, cfg.threshold)
    return np.stack([rotate_key(k, q, cfg.mapping) for q in range(N_QUADRANTS)])


def fire_counts(scam: SegmentedCAM, keys: np.ndarray, tau: int) -> np.ndarray:
    """Segments fired per row, OR-ed over the theta0 quadrants: (N,)."""
    return scam.fires(keys, tau).any(axis=0).sum(axis=1)


# ---------------------------------------------------------------- cell sampling

def _signal(cfg, prn, cn0, fd, rng, code_phase):
    return channel.generate(prn, cn0, fd, cfg.fs, cfg.n_samples,
                            code_phase_chips=code_phase, rng=rng).samples


def sample_cells(cfg: GpsConfig, cn0: float, trials: int, tau: int,
                 rng: np.random.Generator, cb=None, scam=None) -> dict:
    """Per-trial segment-fire counts at the aligned cell and at the three kinds of
    wrong cell. Returns arrays of fire counts (0..S)."""
    if cb is None:
        cb, scam = build(cfg)
    aligned_prn = 1
    row_ok = int(np.flatnonzero((cb.prn_of_row == aligned_prn) & (cb.fd_of_row == 0.0))[0])
    wrong_fd = int(np.flatnonzero((cb.prn_of_row == aligned_prn)
                                  & (cb.fd_of_row == cb.fd_grid[-1]))[0])
    wrong_prn = int(np.flatnonzero((cb.prn_of_row == 2) & (cb.fd_of_row == 0.0))[0])

    out = {k: np.zeros(trials, dtype=int) for k in
           ("aligned", "wrong_phase", "wrong_doppler", "wrong_prn")}
    for t in range(trials):
        keys = dwell_keys(_signal(cfg, aligned_prn, cn0, 0.0, rng, 0.0), cfg)
        c = fire_counts(scam, keys, tau)
        out["aligned"][t] = c[row_ok]
        out["wrong_doppler"][t] = c[wrong_fd]
        out["wrong_prn"][t] = c[wrong_prn]
        off = float(rng.uniform(2, CODE_LENGTH - 2))
        keys_off = dwell_keys(_signal(cfg, aligned_prn, cn0, 0.0, rng, off), cfg)
        out["wrong_phase"][t] = fire_counts(scam, keys_off, tau)[row_ok]
    return out


def calibrate_tau(cfg: GpsConfig, cn0_high: float, target_seg_pfa: float,
                  trials: int, rng: np.random.Generator) -> int:
    """Largest per-segment tau whose wrong-cell fire rate stays under budget.

    Calibrated at the HIGHEST C/N0 of the sweep against a structured wrong-phase
    signal -- see the module docstring.
    """
    cb, scam = build(cfg)
    dists = []
    row_ok = int(np.flatnonzero((cb.prn_of_row == 1) & (cb.fd_of_row == 0.0))[0])
    seg_w = scam.seg_width
    for _ in range(trials):
        off = float(rng.uniform(2, CODE_LENGTH - 2))
        keys = dwell_keys(_signal(cfg, 1, cn0_high, 0.0, rng, off), cfg)
        d = np.stack([[(keys[q][i * seg_w:(i + 1) * seg_w].astype(bool)
                        != cb.rows[row_ok][i * seg_w:(i + 1) * seg_w]).sum()
                       for i in range(scam.S)] for q in range(N_QUADRANTS)])
        dists.append(d.min(axis=0))
    return int(np.quantile(np.concatenate(dists), target_seg_pfa))


# ---------------------------------------------------------------- search sizing

def independent_cells(cfg: GpsConfig) -> int:
    """Chip-spaced code phases x Doppler rows x PRNs. Half-chip samples are
    correlated over the +-1 chip correlation triangle, so chips are the
    independent unit."""
    return CODE_LENGTH * len(codebook.doppler_grid(cfg.f_max, cfg.t_coh)) * cfg.n_prn


def dictionary_bits(cfg: GpsConfig) -> int:
    n_fd = len(codebook.doppler_grid(cfg.f_max, cfg.t_coh))
    return cfg.n_prn * n_fd * row_state_bits(cfg.n_samples)


def match_line_cells(cfg: GpsConfig) -> int:
    n_fd = len(codebook.doppler_grid(cfg.f_max, cfg.t_coh))
    return cfg.n_prn * n_fd * key_width(cfg.n_samples, cfg.mapping)


# ------------------------------------------------------- characterization pass

def segment_distances(cfg: GpsConfig, cn0: float, trials: int,
                      rng: np.random.Generator) -> dict:
    """Per-segment Hamming distances (minimised over the four theta0 quadrants)
    for the aligned cell and each kind of wrong cell.

    This is a *characterization and calibration* pass, the same role
    `sim/experiments.calibrate_tau` plays for the BLE study: it looks at
    distances so that a threshold can be chosen. The detector itself
    (`segment.detect`) still consumes nothing but booleans.

    Returning distances rather than fires lets tau be swept afterwards without
    re-running the Monte Carlo.
    """
    cb, scam = build(cfg)
    seg_w = scam.seg_width
    rows = {
        "aligned": int(np.flatnonzero((cb.prn_of_row == 1) & (cb.fd_of_row == 0.0))[0]),
        "wrong_doppler": int(np.flatnonzero((cb.prn_of_row == 1)
                                            & (cb.fd_of_row == cb.fd_grid[-1]))[0]),
        "wrong_prn": int(np.flatnonzero((cb.prn_of_row == 2) & (cb.fd_of_row == 0.0))[0]),
    }
    out = {k: np.zeros((trials, scam.S), dtype=int) for k in list(rows) + ["wrong_phase"]}

    def per_segment(keys, row):
        d = np.stack([[(keys[q][i * seg_w:(i + 1) * seg_w].astype(bool)
                        != cb.rows[row][i * seg_w:(i + 1) * seg_w]).sum()
                       for i in range(scam.S)] for q in range(N_QUADRANTS)])
        return d.min(axis=0)

    for t in range(trials):
        keys = dwell_keys(_signal(cfg, 1, cn0, 0.0, rng, 0.0), cfg)
        for name, row in rows.items():
            out[name][t] = per_segment(keys, row)
        off = float(rng.uniform(2, CODE_LENGTH - 2))
        out["wrong_phase"][t] = per_segment(
            dwell_keys(_signal(cfg, 1, cn0, 0.0, rng, off), cfg), rows["aligned"])
    return out


def cached_distances(cfg: GpsConfig, cn0: float, trials: int, seed: int,
                     force: bool = False) -> dict:
    """`segment_distances` behind the npz cache, so the segmentation and
    sense-margin studies share one Monte Carlo pass."""
    params = {**cfg.as_dict(), "cn0": cn0, "trials": trials, "seed": seed,
              "kind": "gps_seg_distances_v1"}

    def compute():
        return segment_distances(cfg, cn0, trials, np.random.default_rng(seed))

    return cached("gps_seg_dist", params, compute, force=force)


def dwells_required(p1: float, p0: float, n_seg_per_dwell: int, n_cells: int,
                    target_pd: float = 0.9, system_pfa: float = 0.01,
                    max_dwells: int = 500) -> tuple[int, int]:
    """Smallest number of 1 ms dwells M (and segment-count threshold k) meeting
    both targets under the validated independent-segment binomial model.

    Fire counts accumulate over M dwells, so the statistic is Binomial(M*S, p).
    k is chosen as the smallest threshold whose per-cell false-alarm rate keeps
    the whole 2-D search under `system_pfa`; M is then the smallest dwell count
    at which that same k still detects with probability `target_pd`.
    Returns (-1, -1) if unreachable within `max_dwells`.
    """
    from scipy.stats import binom
    for m in range(1, max_dwells + 1):
        n = m * n_seg_per_dwell
        k = int(binom.isf(system_pfa / n_cells, n, p0)) + 1
        k = max(1, min(k, n))
        if binom.sf(k - 1, n, p0) * n_cells > system_pfa:
            k += 1
        if k <= n and binom.sf(k - 1, n, p1) >= target_pd:
            return m, k
    return -1, -1
