"""The baseline must actually acquire, or it is not a baseline."""
import numpy as np
import pytest

from sim.gps import baselines, channel, codebook

FS = 2.046e6
NS = 2046


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_fft_acquires_a_planted_signal(seed):
    """Cross-validation: plant (PRN, code phase, Doppler) and the parallel
    code-phase search must find it to within a sample and a bin."""
    rng = np.random.default_rng(seed)
    grid = codebook.doppler_grid(10e3, 1e-3)
    prn = int(rng.integers(1, 33))
    cp = float(rng.integers(0, 1023))
    fd = float(rng.choice(grid))
    s = channel.generate(prn, 48, fd, FS, NS, code_phase_chips=cp, rng=rng)
    r = baselines.acquire_fft(s.samples, prn, grid, FS)

    expected = int(round((-cp * 2) % NS))
    err = min(abs(r.code_phase_samples - expected), NS - abs(r.code_phase_samples - expected))
    assert err <= 1, f"code phase off by {err} samples"
    assert abs(r.fd - fd) <= grid[1] - grid[0]
    assert r.metric > 2.0


def test_fft_finds_nothing_for_the_wrong_prn():
    rng = np.random.default_rng(4)
    grid = codebook.doppler_grid(5e3, 1e-3)
    s = channel.generate(1, 48, 0.0, FS, NS, code_phase_chips=100.0, rng=rng)
    assert baselines.acquire_fft(s.samples, 7, grid, FS).metric < 2.0


@pytest.mark.parametrize("cn0", [38, 42, 45])
def test_dwells_required_decreases_with_cn0(cn0):
    cells = 1023 * 41 * 32
    m, _ = baselines.dwells_required_fft(cn0, 1e-3, cells)
    m_better, _ = baselines.dwells_required_fft(cn0 + 3, 1e-3, cells)
    assert 0 < m_better <= m


def test_quantized_baseline_is_worse_but_still_acquires():
    """B1' must still find the signal -- it is the reference that isolates the
    1.96 dB limiting loss from the boolean-readout cost."""
    rng = np.random.default_rng(6)
    grid = codebook.doppler_grid(5e3, 1e-3)
    s = channel.generate(3, 48, 1000.0, FS, NS, code_phase_chips=200.0, rng=rng)
    full = baselines.acquire_fft(s.samples, 3, grid, FS)
    quant = baselines.acquire_fft(s.samples, 3, grid, FS, quantize=True)
    assert quant.code_phase_samples == full.code_phase_samples
    assert quant.metric > 2.0


def test_mac_count_scales_as_expected():
    a = baselines.fft_complex_macs(2046, 41, 32)
    assert baselines.fft_complex_macs(2046, 41, 64) == 2 * a
    assert baselines.fft_complex_macs(2046, 82, 32) == 2 * a
