"""Composite ("vernier") key: verifies the core claim that concatenating a
coarse (D=1 symbol, collision-free) segment with a fine (D>1, individually
aliased) segment eliminates genuine far-CFO collisions -- even for D=4 alone,
which is the worst single-lag offender (see docs/vernier_findings.md).

"Genuine collision" here means two rows more than half the coverage range
apart in CFO that are nonetheless almost identical in Hamming distance --
the actual false-lock risk, as opposed to mere local resolution saturation
(neighboring rows tied because the local encoding is coarse, which is
benign) or non-monotone aggregate distance-from-reference (which affects
run-midpoint's ordering assumption but not collision safety).
"""
import numpy as np
import pytest

from sim.codebook import build_codebook
from sim.vernier import LagSpec, build_composite_codebook


def _has_genuine_collision(rows: np.ndarray, df_grid: np.ndarray, W: int,
                            far_khz: float = 75.0, near_frac: float = 0.05) -> bool:
    near_thresh = near_frac * W
    for k in range(len(rows)):
        d = np.count_nonzero(rows != rows[k][None, :], axis=1)
        far_mask = np.abs(df_grid - df_grid[k]) > far_khz * 1e3
        if np.any((d <= near_thresh) & far_mask):
            return True
    return False


def test_single_lag_D4_has_genuine_far_collisions():
    """D=4 alone is the known-worst single-lag offender: confirms the
    baseline failure mode this test module exists to fix."""
    cb = build_codebook(N=128, df_min=-150e3, df_max=150e3, n_sym=40, osr=4,
                         B=2, diff_delay=4 * 4)
    assert _has_genuine_collision(cb.rows, cb.df_grid, cb.W)


@pytest.mark.parametrize("fine_D", [2, 3, 4])
def test_composite_key_eliminates_far_collisions(fine_D):
    """Coarse (D=1) + fine (D=fine_D) concatenation must be collision-free
    across the whole +-150 kHz range, for every fine lag tested -- including
    D=4, which is catastrophic on its own (see test above)."""
    lags = [LagSpec(1, 2), LagSpec(fine_D, 2)]
    cb = build_composite_codebook(lags, N=128, df_min=-150e3, df_max=150e3,
                                   n_sym=40, osr=4)
    assert not _has_genuine_collision(cb.rows, cb.df_grid, cb.W)


def test_composite_key_width_is_additive():
    from sim.encode import key_width
    lags = [LagSpec(1, 2), LagSpec(3, 2)]
    cb = build_composite_codebook(lags, N=8, df_min=-150e3, df_max=150e3, n_sym=40, osr=4)
    expected = key_width(40, 4, 2, diff_delay=1 * 4) + key_width(40, 4, 2, diff_delay=3 * 4)
    assert cb.W == expected
