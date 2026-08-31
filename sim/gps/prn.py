"""GPS L1 C/A Gold code generation.

Two 10-stage LFSRs (G1, G2) per IS-GPS-200; the per-PRN code is G1 XOR a
delayed G2, the delay being selected by a pair of G2 taps. Everything
downstream is meaningless if these codes are wrong, so `tests/gps/test_prn.py`
gates them against the IS-GPS-200 first-10-chips octal table and against the
three-valued autocorrelation property.
"""
from __future__ import annotations

import numpy as np

CODE_LENGTH = 1023
CHIP_RATE = 1.023e6
CODE_PERIOD = CODE_LENGTH / CHIP_RATE      # 1 ms exactly
L1_CARRIER = 1575.42e6

# IS-GPS-200 Table 3-Ia: G2 phase-selector taps per PRN (1-indexed stages).
G2_TAPS: dict[int, tuple[int, int]] = {
    1: (2, 6),   2: (3, 7),   3: (4, 8),   4: (5, 9),   5: (1, 9),
    6: (2, 10),  7: (1, 8),   8: (2, 9),   9: (3, 10), 10: (2, 3),
    11: (3, 4), 12: (5, 6),  13: (6, 7),  14: (7, 8),  15: (8, 9),
    16: (9, 10), 17: (1, 4), 18: (2, 5),  19: (3, 6),  20: (4, 7),
    21: (5, 8), 22: (6, 9),  23: (1, 3),  24: (4, 6),  25: (5, 7),
    26: (6, 8), 27: (7, 9),  28: (8, 10), 29: (1, 6),  30: (2, 7),
    31: (3, 8), 32: (4, 9),
}

# IS-GPS-200 Table 3-Ia: first 10 chips as an octal word (leading 1 is the
# first chip). Used purely as a test vector.
FIRST10_OCTAL: dict[int, int] = {
    1: 0o1440,  2: 0o1620,  3: 0o1710,  4: 0o1744,  5: 0o1133,
    6: 0o1455,  7: 0o1131,  8: 0o1454,  9: 0o1626, 10: 0o1504,
    11: 0o1642, 12: 0o1750, 13: 0o1764, 14: 0o1772, 15: 0o1775,
    16: 0o1776, 17: 0o1156, 18: 0o1467, 19: 0o1633, 20: 0o1715,
    21: 0o1746, 22: 0o1763, 23: 0o1063, 24: 0o1706, 25: 0o1743,
    26: 0o1761, 27: 0o1770, 28: 0o1774, 29: 0o1127, 30: 0o1453,
    31: 0o1625, 32: 0o1712,
}

NUM_PRNS = 32


def ca_code(prn: int) -> np.ndarray:
    """Return the 1023-chip C/A code for `prn` as a uint8 array of 0/1."""
    if prn not in G2_TAPS:
        raise ValueError(f"PRN must be 1..{NUM_PRNS}, got {prn}")
    t1, t2 = G2_TAPS[prn]

    g1 = np.ones(10, dtype=np.uint8)
    g2 = np.ones(10, dtype=np.uint8)
    out = np.empty(CODE_LENGTH, dtype=np.uint8)

    for i in range(CODE_LENGTH):
        out[i] = g1[9] ^ g2[t1 - 1] ^ g2[t2 - 1]
        fb1 = g1[2] ^ g1[9]
        fb2 = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]
        g1[1:] = g1[:9]
        g1[0] = fb1
        g2[1:] = g2[:9]
        g2[0] = fb2

    return out


def ca_code_bipolar(prn: int) -> np.ndarray:
    """C/A code mapped to +-1 floats (chip 0 -> +1, chip 1 -> -1)."""
    return 1.0 - 2.0 * ca_code(prn).astype(np.float64)


def all_codes_bipolar(prns: list[int] | None = None) -> np.ndarray:
    """(len(prns), 1023) matrix of +-1 codes."""
    if prns is None:
        prns = list(range(1, NUM_PRNS + 1))
    return np.stack([ca_code_bipolar(p) for p in prns])


def upsample(code: np.ndarray, osr: int, code_phase_chips: float = 0.0,
             n_samples: int | None = None, chips_per_sample: float | None = None) -> np.ndarray:
    """Sample a chip sequence at `osr` samples/chip with a (possibly fractional)
    starting code phase, wrapping cyclically.

    `chips_per_sample` defaults to 1/osr; pass a slightly different value to
    model code Doppler. Nearest-chip (zero-order hold) sampling, which is what a
    real correlator sees from a hard-limited front end.
    """
    if chips_per_sample is None:
        chips_per_sample = 1.0 / osr
    if n_samples is None:
        n_samples = int(round(len(code) / chips_per_sample))
    idx = code_phase_chips + chips_per_sample * np.arange(n_samples)
    return code[np.floor(idx).astype(np.int64) % len(code)]
