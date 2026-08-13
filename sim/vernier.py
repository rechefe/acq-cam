"""Composite multi-lag ("vernier") key: concatenate a coarse-lag differential
segment (monotone/unambiguous across the full +-150 kHz range, but weakly
CFO-sensitive) with a fine-lag segment (strongly CFO-sensitive, but aliases
on its own within the range).

The disambiguation is free: Hamming distance is additive over concatenated
bit vectors, so a hypothesis row that only matches the fine segment (a false
alias) still pays the coarse segment's mismatch penalty and cannot make it
under tau. No changes to cam.py or decode.py are needed -- this is purely a
different key/codebook construction, exactly like classic vernier calipers
or coarse/fine ADC stages.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tx import tx_waveform, STANDARD_ACCESS_ADDRESS
from .channel import apply_cfo
from .encode import differential_product, CODES, code_width, assemble_key
from .codebook import cfo_grid


@dataclass
class LagSpec:
    D: int  # symbols
    B: int
    coding: str = "thermometer"


def composite_key_width(n_sym: int, osr: int, lags: list[LagSpec]) -> int:
    n_samples = n_sym * osr
    total = 0
    for lag in lags:
        diff_delay = lag.D * osr
        n_z = n_samples - diff_delay
        total += n_z * code_width(lag.B, lag.coding)
    return total


def composite_encode(r: np.ndarray, osr: int, lags: list[LagSpec]) -> np.ndarray:
    parts = []
    for lag in lags:
        z = differential_product(r, delay=lag.D * osr)
        code_fn = CODES[lag.coding]
        bits_2d = code_fn(z, lag.B)
        parts.append(assemble_key(bits_2d))
    return np.concatenate(parts)


@dataclass
class CompositeCodebook:
    rows: np.ndarray
    df_grid: np.ndarray
    W: int
    N: int
    delta: float
    lags: list[LagSpec]


def build_composite_codebook(lags: list[LagSpec], N: int | None = None, delta: float | None = None,
                              df_min: float = -150e3, df_max: float = 150e3,
                              n_sym: int = 40, osr: int = 4,
                              access_address: int = STANDARD_ACCESS_ADDRESS) -> CompositeCodebook:
    if delta is None:
        if N is None:
            raise ValueError("must supply either N or delta")
        delta = (df_max - df_min) / max(N - 1, 1)
    grid = cfo_grid(df_min, df_max, delta)
    if N is not None:
        grid = grid[:N]
    N_actual = len(grid)

    tx = tx_waveform(n_sym=n_sym, osr=osr, access_address=access_address)
    W = composite_key_width(n_sym, osr, lags)
    rows = np.zeros((N_actual, W), dtype=bool)
    for k, df in enumerate(grid):
        sig = apply_cfo(tx, df, osr)
        rows[k] = composite_encode(sig, osr, lags).astype(bool)

    return CompositeCodebook(rows=rows, df_grid=grid, W=W, N=N_actual, delta=delta, lags=lags)
