"""2-bit I/Q front end, CAM key mappings, and the free 90-degree rotation.

Two key mappings are provided:

  sign        1 bit per rail (2 bits/sample). Hamming distance is a sign-sign
              correlation: the textbook 1-bit limiting loss of 1.96 dB.
  thermometer 3 bits per rail (6 bits/sample) encoding the four 2-bit ADC levels
              {-3,-1,+1,+3} as [x>-2, x>0, x>+2]. Hamming distance between two
              thermometer words is the L1 distance between their levels, so a key
              matched against a full-scale row gives a magnitude-weighted
              (soft-decision) correlation.

Note on area: codebook rows are always full scale, so a thermometer row only ever
takes the two values 000 and 111. The array needs 3 comparison cells per rail but
only 1 bit of stored state -- see `row_state_bits()`.

The theta0 sweep is a bit permutation, not arithmetic: rotating a sample by 90
degrees is (I,Q) -> (-Q,I), which on these codes is a rail swap plus an inversion.
`rotate_key` performs it exactly, with no re-quantization.
"""
from __future__ import annotations

import numpy as np

# 2-bit uniform quantizer. The threshold is in units of the noise sigma; ~1.0 is
# the standard GNSS choice (near-optimal for a hard-limited front end).
DEFAULT_THRESHOLD = 1.0
LEVELS = np.array([-3.0, -1.0, 1.0, 3.0])


def adc_2bit(x: np.ndarray, sigma: float, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    """Quantize a real rail to level indices 0..3 (values -3,-1,+1,+3)."""
    t = threshold * sigma
    return (np.digitize(x, [-t, 0.0, t])).astype(np.uint8)


def levels_to_sign_bits(lv: np.ndarray) -> np.ndarray:
    """Level index -> 1 sign bit (1 == negative)."""
    return (lv < 2).astype(np.uint8)[..., None]


def levels_to_thermo_bits(lv: np.ndarray) -> np.ndarray:
    """Level index -> 3 thermometer bits, so Hamming == L1 distance in levels."""
    return (lv[..., None] > np.arange(3)).astype(np.uint8)


MAPPINGS = {"sign": (levels_to_sign_bits, 1), "thermometer": (levels_to_thermo_bits, 3)}


def bits_per_rail(mapping: str) -> int:
    return MAPPINGS[mapping][1]


def key_width(n_samples: int, mapping: str) -> int:
    return n_samples * 2 * bits_per_rail(mapping)


def row_state_bits(n_samples: int) -> int:
    """Stored state per row: rows are full scale, so 1 bit per rail regardless of
    mapping. The extra thermometer cells widen the match line, not the memory."""
    return n_samples * 2


def make_key(i_lv: np.ndarray, q_lv: np.ndarray, mapping: str) -> np.ndarray:
    """Interleave I and Q level indices into a flat sample-major key."""
    fn, nb = MAPPINGS[mapping]
    both = np.stack([fn(i_lv), fn(q_lv)], axis=-2)      # (..., n, 2, nb)
    return both.reshape(*i_lv.shape[:-1], -1).astype(np.uint8)


def quantize_to_key(r: np.ndarray, sigma: float, mapping: str = "sign",
                    threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    """Complex baseband -> 2-bit I/Q -> flat CAM key."""
    return make_key(adc_2bit(r.real, sigma, threshold),
                    adc_2bit(r.imag, sigma, threshold), mapping)


def full_scale_key(c: np.ndarray, mapping: str) -> np.ndarray:
    """Codebook row: a noiseless complex replica taken to full scale (+-3), which
    makes Hamming distance against a soft key a magnitude-weighted correlation."""
    i_lv = np.where(c.real >= 0, 3, 0).astype(np.uint8)
    q_lv = np.where(c.imag >= 0, 3, 0).astype(np.uint8)
    return make_key(i_lv, q_lv, mapping)


def rotate_key(key: np.ndarray, quadrant: int, mapping: str) -> np.ndarray:
    """Rotate a key by quadrant*90 degrees, exactly, as a bit permutation.

    90 deg is (I,Q) -> (-Q,I). Negating a rail reverses and complements its
    thermometer code (or complements its sign bit). No arithmetic, no
    re-quantization, so the theta0 sweep is free in hardware.
    """
    q = quadrant % 4
    if q == 0:
        return key
    nb = bits_per_rail(mapping)
    v = key.reshape(*key.shape[:-1], -1, 2, nb)
    i, qr = v[..., 0, :], v[..., 1, :]
    neg = lambda b: 1 - b[..., ::-1]
    if q == 1:
        i2, q2 = neg(qr), i
    elif q == 2:
        i2, q2 = neg(i), neg(qr)
    else:
        i2, q2 = qr, neg(i)
    return np.stack([i2, q2], axis=-2).reshape(key.shape).astype(np.uint8)
