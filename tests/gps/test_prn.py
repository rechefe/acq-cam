"""Gold code gates. If these fail, nothing downstream means anything."""
import numpy as np
import pytest

from sim.gps.prn import (CODE_LENGTH, FIRST10_OCTAL, NUM_PRNS, ca_code,
                         ca_code_bipolar, upsample)

# Three-valued autocorrelation of a length-1023 Gold code (m=10).
VALID_OFF_PEAK = {-65, -1, 63}


@pytest.mark.parametrize("prn", range(1, NUM_PRNS + 1))
def test_first_ten_chips_match_is_gps_200(prn):
    got = int("".join(str(b) for b in ca_code(prn)[:10]), 2)
    assert got == FIRST10_OCTAL[prn], f"PRN {prn}: {got:o} != {FIRST10_OCTAL[prn]:o}"


@pytest.mark.parametrize("prn", [1, 5, 17, 32])
def test_code_is_balanced(prn):
    """A C/A code has exactly 512 ones and 511 zeros."""
    assert ca_code(prn).sum() == 512
    assert len(ca_code(prn)) == CODE_LENGTH


@pytest.mark.parametrize("prn", [1, 11, 23, 32])
def test_autocorrelation_is_three_valued(prn):
    c = ca_code_bipolar(prn)
    ac = np.array([np.dot(c, np.roll(c, k)) for k in range(CODE_LENGTH)])
    assert ac[0] == CODE_LENGTH
    assert set(np.unique(ac[1:]).astype(int)) <= VALID_OFF_PEAK


@pytest.mark.parametrize("a,b", [(1, 2), (5, 17), (23, 32), (7, 19)])
def test_crosscorrelation_is_bounded(a, b):
    ca, cb = ca_code_bipolar(a), ca_code_bipolar(b)
    xc = np.array([np.dot(ca, np.roll(cb, k)) for k in range(CODE_LENGTH)])
    assert set(np.unique(xc).astype(int)) <= VALID_OFF_PEAK
    assert np.abs(xc).max() == 65


def test_all_codes_distinct():
    codes = {tuple(ca_code(p)) for p in range(1, NUM_PRNS + 1)}
    assert len(codes) == NUM_PRNS


def test_upsample_wraps_cyclically_and_holds_chips():
    c = ca_code_bipolar(1)
    up = upsample(c, osr=2, n_samples=2 * CODE_LENGTH)
    assert np.array_equal(up[0::2], c) and np.array_equal(up[1::2], c)
    # wraps past one period
    assert np.array_equal(upsample(c, osr=1, n_samples=CODE_LENGTH + 10)[CODE_LENGTH:], c[:10])
