"""Key encoding: theta0-cancelling differential product + circular thermometer /
Gray / one-hot phase quantization, and CAM key assembly.

No angle()/arctan anywhere: the thermometer code is built purely from signs of
projections onto rotated axes, exactly as the analog comparator array would do it.
"""
from __future__ import annotations

import numpy as np


def differential_product(r: np.ndarray, delay: int = 1) -> np.ndarray:
    """z[n] = r[n] * conj(r[n-delay]).

    A constant phase offset theta0 present identically on r[n] and r[n-delay]
    cancels exactly in this product.
    """
    return r[delay:] * np.conj(r[:-delay])


def rotated_axes(B: int) -> np.ndarray:
    K = 2 ** B
    idx = np.arange(K)
    return np.exp(1j * 2 * np.pi * idx / K)


def thermometer_code(z: np.ndarray, B: int) -> np.ndarray:
    """Circular thermometer code: for each sample n and axis i in 0..2^B-1,
    bit[n, i] = 1 iff Re(z[n] * conj(u_i)) >= 0, u_i = exp(j*2*pi*i/2^B).

    Returns an (len(z), 2**B) uint8 array (no angle()/atan2 used).
    """
    u = rotated_axes(B)
    proj = np.real(z[:, None] * np.conj(u)[None, :])
    return (proj >= 0).astype(np.uint8)


def onehot_code(z: np.ndarray, B: int) -> np.ndarray:
    """One-hot sector code: same 2^B rotated axes define 2^B sectors (by nearest-axis
    projection argmax); bit i is 1 iff sector i is the argmax sector. Same code width
    as thermometer_code, for a fair Fig-3 comparison."""
    u = rotated_axes(B)
    proj = np.real(z[:, None] * np.conj(u)[None, :])
    K = 2 ** B
    sector = np.argmax(proj, axis=1)
    bits = np.zeros((len(z), K), dtype=np.uint8)
    bits[np.arange(len(z)), sector] = 1
    return bits


def _gray_code_int(n: int) -> int:
    return n ^ (n >> 1)


def gray_code(z: np.ndarray, B: int) -> np.ndarray:
    """Gray-coded sector index using the same 2^B rotated-axis sectorization
    (argmax projection), encoded as a standard B-bit Gray code per sample."""
    u = rotated_axes(B)
    proj = np.real(z[:, None] * np.conj(u)[None, :])
    sector = np.argmax(proj, axis=1)
    gray_vals = np.array([_gray_code_int(int(s)) for s in sector], dtype=np.uint32)
    bits = np.zeros((len(z), B), dtype=np.uint8)
    for b in range(B):
        bits[:, B - 1 - b] = (gray_vals >> b) & 1
    return bits


CODES = {
    "thermometer": thermometer_code,
    "gray": gray_code,
    "onehot": onehot_code,
}


def code_width(B: int, coding: str) -> int:
    if coding == "gray":
        return B
    return 2 ** B


def assemble_key(bits_2d: np.ndarray) -> np.ndarray:
    """Flatten a (num_samples, code_width) bit matrix into a single sample-major
    flat key vector of length W = num_samples * code_width."""
    return bits_2d.reshape(-1).astype(np.uint8)


def encode_waveform(r: np.ndarray, B: int, coding: str = "thermometer",
                     diff_delay: int = 1) -> np.ndarray:
    """Full encode chain: received complex baseband -> differential product ->
    phase code -> flattened binary key of length W."""
    z = differential_product(r, delay=diff_delay)
    code_fn = CODES[coding]
    bits_2d = code_fn(z, B)
    return assemble_key(bits_2d)


def key_width(n_sym: int, osr: int, B: int, coding: str = "thermometer", diff_delay: int = 1) -> int:
    n_samples = n_sym * osr
    n_z = n_samples - diff_delay
    return n_z * code_width(B, coding)
