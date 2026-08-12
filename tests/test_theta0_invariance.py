"""theta0-invariance of the differential encoding (spec section 2.1 / 1.2).

A random uniform initial phase theta0 must cancel exactly in the differential
product, so the resulting binary key must be identical regardless of theta0
(in the noiseless case).
"""
import numpy as np
import pytest

from sim.tx import tx_waveform
from sim.channel import apply_cfo, apply_phase_offset
from sim.encode import encode_waveform


@pytest.mark.parametrize("coding", ["thermometer", "gray", "onehot"])
def test_theta0_invariance_noiseless(coding):
    rng = np.random.default_rng(0)
    osr = 4
    n_sym = 40
    B = 2
    df = 20e3

    tx = tx_waveform(n_sym=n_sym, osr=osr)
    tx_cfo = apply_cfo(tx, df, osr)

    keys = []
    for theta0 in [0.0, 0.3, 1.7, np.pi, 2 * np.pi - 0.01, rng.uniform(0, 2 * np.pi)]:
        r = apply_phase_offset(tx_cfo, theta0)
        key = encode_waveform(r, B, coding=coding)
        keys.append(key)

    for k in keys[1:]:
        assert np.array_equal(k, keys[0])


def test_differential_product_cancels_constant_phase():
    from sim.encode import differential_product
    rng = np.random.default_rng(1)
    n = 200
    base = np.exp(1j * rng.uniform(0, 2 * np.pi, n))
    for theta0 in [0.0, 1.234, 5.5]:
        z = differential_product(base * np.exp(1j * theta0))
        z0 = differential_product(base)
        assert np.allclose(z, z0, atol=1e-10)
