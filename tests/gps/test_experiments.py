"""Detector plumbing and the search-sizing arithmetic."""
import numpy as np
import pytest

from sim.gps import codebook
from sim.gps.experiments import (GpsConfig, N_QUADRANTS, build, dictionary_bits,
                                 dwell_keys, dwells_required, fire_counts,
                                 independent_cells, match_line_cells)
from sim.gps.segment import SegmentedCAM, detect


def test_config_derives_t_coh_from_segments():
    assert GpsConfig(n_segments=1).n_samples == 2046
    assert GpsConfig(n_segments=6).t_coh == pytest.approx(2046 / 6 / 2.046e6)


def test_dwell_keys_are_the_four_rotations():
    cfg = GpsConfig(n_prn=2)
    rng = np.random.default_rng(0)
    r = rng.normal(0, 1, cfg.n_samples) + 1j * rng.normal(0, 1, cfg.n_samples)
    keys = dwell_keys(r, cfg)
    assert keys.shape == (N_QUADRANTS, cfg.n_samples * 2)
    assert len({tuple(k) for k in keys}) == N_QUADRANTS


def test_fire_counts_are_bounded_by_segment_count():
    cfg = GpsConfig(n_segments=6, n_prn=2)
    cb, scam = build(cfg)
    rng = np.random.default_rng(1)
    keys = dwell_keys(rng.normal(0, 1, cfg.n_samples)
                      + 1j * rng.normal(0, 1, cfg.n_samples), cfg)
    c = fire_counts(scam, keys, tau=scam.seg_width // 2)
    assert c.shape == (cb.N,) and c.min() >= 0 and c.max() <= scam.S


def test_detect_accumulates_across_dwells():
    """More dwells can only raise the accumulated fire count."""
    rows = np.random.default_rng(2).integers(0, 2, size=(5, 96))
    scam = SegmentedCAM(rows, n_segments=3)
    keys = np.random.default_rng(3).integers(0, 2, size=(N_QUADRANTS, 96))
    one = detect(scam, [keys], tau=40, k=1)
    two = detect(scam, [keys, keys], tau=40, k=1)
    assert np.all(two.fire_count == 2 * one.fire_count)


def test_dwells_required_is_monotone_in_separation():
    cells = 10 ** 6
    easy, _ = dwells_required(0.60, 0.10, 6, cells)
    hard, _ = dwells_required(0.30, 0.10, 6, cells)
    assert 0 < easy <= hard


def test_dwells_required_gives_up_when_inseparable():
    assert dwells_required(0.10, 0.10, 6, 10 ** 6) == (-1, -1)


@pytest.mark.parametrize("S,expected_rows_per_prn", [(1, 41), (2, 21), (6, 9), (11, 5)])
def test_search_sizing_tracks_the_doppler_grid(S, expected_rows_per_prn):
    cfg = GpsConfig(n_segments=S, n_prn=32)
    assert len(codebook.doppler_grid(cfg.f_max, cfg.t_coh)) == expected_rows_per_prn
    assert independent_cells(cfg) == 1023 * expected_rows_per_prn * 32
    assert dictionary_bits(cfg) == 32 * expected_rows_per_prn * 2 * 2046


def test_thermometer_widens_the_match_line_only():
    a = GpsConfig(n_segments=6, mapping="sign")
    b = GpsConfig(n_segments=6, mapping="thermometer")
    assert dictionary_bits(a) == dictionary_bits(b)
    assert match_line_cells(b) == 3 * match_line_cells(a)
