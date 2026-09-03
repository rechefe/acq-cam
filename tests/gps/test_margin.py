"""The gate that catches algebra errors: measured detection margin must track the
analytic 1-bit result sqrt(2*C/N0*T)*0.798*cos(phase error)."""
import math

import numpy as np
import pytest

from sim.gps import channel, codebook
from sim.gps.prn import CODE_LENGTH
from sim.gps.quantize import quantize_to_key

FS = 2.046e6
T = 1e-3
NS = int(FS * T)


def _margin(cn0, theta0, trials, seed):
    cb = codebook.build_codebook(prns=[1], f_max=0.0, t_coh=T, fs=FS, mapping="sign")
    row = cb.rows[0]
    rng = np.random.default_rng(seed)
    h1, h0 = [], []
    for _ in range(trials):
        s = channel.generate(1, cn0, 0.0, FS, NS, theta0=theta0,
                             code_phase_chips=0.0, rng=rng)
        h1.append((quantize_to_key(s.samples, 1.0, "sign").astype(bool) != row).sum())
        s0 = channel.generate(1, cn0, 0.0, FS, NS, theta0=theta0,
                              code_phase_chips=CODE_LENGTH // 2, rng=rng)
        h0.append((quantize_to_key(s0.samples, 1.0, "sign").astype(bool) != row).sum())
    h1, h0 = np.array(h1), np.array(h0)
    return (h0.mean() - h1.mean()) / h0.std()


def _theory(cn0, phase_err):
    return math.sqrt(2 * 10 ** (cn0 / 10) * T) * 0.798 * math.cos(phase_err)


@pytest.mark.parametrize("cn0", [48, 45])
def test_margin_matches_theory_at_matched_phase(cn0):
    """Rows sit at 45 degrees, so a signal at 45 degrees is the matched case."""
    meas = _margin(cn0, theta0=codebook.ROW_THETA, trials=400, seed=11)
    assert 0.85 <= meas / _theory(cn0, 0.0) <= 1.10


def test_worst_case_phase_costs_3db_not_everything():
    """A 45-degree phase error is the worst the free 90-degree sweep allows; it
    must cost cos(45deg) = 3 dB, and must never go blind."""
    matched = _margin(48, theta0=codebook.ROW_THETA, trials=400, seed=13)
    worst = _margin(48, theta0=0.0, trials=400, seed=13)
    assert 0.85 <= (worst / matched) / math.cos(math.pi / 4) <= 1.15


def test_noise_floor_sits_at_half_the_word():
    """Under H0 the expected distance is W/2: a wrong code phase is unstructured."""
    cb = codebook.build_codebook(prns=[1], f_max=0.0, t_coh=T, fs=FS, mapping="sign")
    rng = np.random.default_rng(5)
    d = []
    for _ in range(200):
        s = channel.generate(1, 45, 0.0, FS, NS, code_phase_chips=511.0, rng=rng)
        d.append((quantize_to_key(s.samples, 1.0, "sign").astype(bool) != cb.rows[0]).sum())
    assert abs(np.mean(d) - cb.W / 2) < 4 * np.std(d) / math.sqrt(len(d)) + 0.01 * cb.W
