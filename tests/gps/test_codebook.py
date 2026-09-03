"""Codebook gates, including the row-phase trap."""
import numpy as np
import pytest

from sim.gps import codebook
from sim.gps.channel import replica
from sim.gps.quantize import full_scale_key


@pytest.mark.parametrize("t_coh,expected", [(1e-3, 41), (500e-6, 21), (1e-3 / 3, 15),
                                            (1e-3 / 6, 9), (1e-3 / 11, 5)])
def test_doppler_row_count_follows_the_formula(t_coh, expected):
    """n_doppler = 2*ceil(2*f_max*T_coh)+1 -- the coherent segment length is the
    only thing that sets the Doppler dimension of the dictionary. The T_coh values
    here are 1 ms / S for the S that actually divide 2046 samples."""
    assert len(codebook.doppler_grid(10e3, t_coh)) == expected


def test_segment_counts_are_divisors_of_the_code_period():
    """A segment must be a whole number of samples, so S is constrained to
    divisors of 2046 at OSR=2 -- powers of two are mostly unavailable."""
    valid = codebook.segment_counts(2046)
    assert [s for s in valid if s <= 33] == [1, 2, 3, 6, 11, 22, 31, 33]
    assert 8 not in valid and 4 not in valid


def test_doppler_grid_covers_range_and_is_symmetric():
    g = codebook.doppler_grid(10e3, 1e-3)
    assert g.min() <= -10e3 and g.max() >= 10e3
    assert np.allclose(g, -g[::-1])


def test_row_quadrature_rail_is_not_degenerate():
    """The trap: a row built at theta=0 has a purely real replica, so its Q rail
    collapses to a constant and contributes variance without signal -- half the
    match line dead. Rows must be built at 45 degrees."""
    n = 512
    at_zero = full_scale_key(replica(1, 0.0, 2.046e6, n, theta=0.0), "sign")
    q_rail_zero = at_zero.reshape(-1, 2)[:, 1]
    assert len(np.unique(q_rail_zero)) == 1, "expected the theta=0 trap to be degenerate"

    at_45 = full_scale_key(replica(1, 0.0, 2.046e6, n, theta=codebook.ROW_THETA), "sign")
    for rail in range(2):
        bits = at_45.reshape(-1, 2)[:, rail]
        assert len(np.unique(bits)) == 2, f"rail {rail} of a 45-degree row is degenerate"


def test_rows_are_unique_across_prn_and_doppler():
    cb = codebook.build_codebook(prns=[1, 2, 3], f_max=5e3, t_coh=1e-3)
    assert len({tuple(r) for r in cb.rows}) == cb.N


def test_dictionary_accounting():
    cb = codebook.build_codebook(prns=list(range(1, 33)), f_max=10e3,
                                 t_coh=1e-3 / 6, mapping="sign")
    assert cb.N == 32 * 9
    # thermometer widens the match line but not the stored state
    cbt = codebook.build_codebook(prns=list(range(1, 33)), f_max=10e3,
                                  t_coh=1e-3 / 6, mapping="thermometer")
    assert cbt.dictionary_bits == cb.dictionary_bits
    assert cbt.match_line_cells == 3 * cb.match_line_cells
