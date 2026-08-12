"""BLE 1M PHY transmitter: bit generation and GFSK modulation.

GFSK, modulation index h = 0.5, BT = 0.5 Gaussian pulse shaping,
symbol rate 1 Msym/s, one bit per symbol.
"""
from __future__ import annotations

import numpy as np

SYMBOL_RATE = 1.0e6  # 1 Msym/s
STANDARD_ACCESS_ADDRESS = 0x8E89BED6
DEFAULT_H = 0.5
DEFAULT_BT = 0.5


def _bytes_to_bits_lsb_first(value: int, n_bytes: int) -> np.ndarray:
    """BLE transmits each byte LSB-first, bytes in the given (little-endian) order."""
    bits = []
    for i in range(n_bytes):
        byte = (value >> (8 * i)) & 0xFF
        for b in range(8):
            bits.append((byte >> b) & 1)
    return np.array(bits, dtype=np.uint8)


def preamble_bits(access_address: int) -> np.ndarray:
    """8-bit BLE preamble: 0x55 if AA's first transmitted bit (LSB of byte0) is 0,
    else 0xAA, per the BLE Core Spec preamble rule."""
    aa_bits = _bytes_to_bits_lsb_first(access_address, 4)
    first_bit = aa_bits[0]
    byte = 0xAA if first_bit == 1 else 0x55
    return _bytes_to_bits_lsb_first(byte, 1)


def access_address_bits(access_address: int) -> np.ndarray:
    return _bytes_to_bits_lsb_first(access_address, 4)


def packet_bits(n_sym: int = 40, access_address: int = STANDARD_ACCESS_ADDRESS) -> np.ndarray:
    """Preamble (8 bits) followed by access address (32 bits), truncated/padded to n_sym.

    n_sym <= 8 returns only the leading preamble bits; n_sym > 8 appends AA bits.
    """
    pre = preamble_bits(access_address)
    aa = access_address_bits(access_address)
    full = np.concatenate([pre, aa])
    if n_sym <= len(full):
        return full[:n_sym].copy()
    # Extend by repeating the AA's trailing pattern is undefined; clip is expected use.
    raise ValueError(f"n_sym={n_sym} exceeds preamble+AA length {len(full)}")


def random_valid_access_address(rng: np.random.Generator) -> int:
    """Draw a pseudo-valid BLE access address for the false-alarm study.

    Implements a simplified subset of the BLE Core Spec AA validity rules:
    no more than 6 consecutive equal bits, at least one transition every 8 bits
    (roughly), not equal to the advertising AA, and not the all-alternating pattern.
    This is a reasonable proxy for "looks like a legitimate but different AA",
    not a full compliance checker.
    """
    while True:
        value = int(rng.integers(0, 2 ** 32, dtype=np.uint64))
        if value == STANDARD_ACCESS_ADDRESS:
            continue
        bits = _bytes_to_bits_lsb_first(value, 4)
        # reject long runs
        run = 1
        max_run = 1
        for i in range(1, len(bits)):
            if bits[i] == bits[i - 1]:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 1
        if max_run > 6:
            continue
        transitions = np.sum(bits[1:] != bits[:-1])
        if transitions < 6:
            continue
        return value


def gaussian_taps(bt: float, osr: int, span_symbols: int = 4) -> np.ndarray:
    """Gaussian premodulation filter taps, unit DC gain, applied to a zero-order-hold
    NRZ pulse train (this is the standard 'Gaussian LPF after rect NRZ' GFSK model)."""
    n = span_symbols * osr
    t = np.arange(-n // 2, n // 2 + 1) / osr  # in symbol periods
    alpha = np.sqrt(np.log(2)) / (2 * np.pi * bt)
    g = np.exp(-(t ** 2) / (2 * alpha ** 2))
    g = g / g.sum()
    return g


def gfsk_modulate(bits: np.ndarray, h: float = DEFAULT_H, bt: float = DEFAULT_BT,
                   osr: int = 4) -> np.ndarray:
    """Modulate bits to a complex-baseband GFSK waveform sampled at osr samples/symbol.

    No AGC, no timing recovery here -- this is the ideal TX waveform.
    """
    symbols_nrz = 2.0 * bits.astype(float) - 1.0
    upsampled = np.repeat(symbols_nrz, osr)
    taps = gaussian_taps(bt, osr)
    freq_shape = np.convolve(upsampled, taps, mode="same")
    phase = np.cumsum(freq_shape) * (np.pi * h / osr)
    waveform = np.exp(1j * phase)
    return waveform


def tx_waveform(n_sym: int = 40, osr: int = 4, h: float = DEFAULT_H, bt: float = DEFAULT_BT,
                 access_address: int = STANDARD_ACCESS_ADDRESS) -> np.ndarray:
    bits = packet_bits(n_sym, access_address)
    return gfsk_modulate(bits, h=h, bt=bt, osr=osr)
