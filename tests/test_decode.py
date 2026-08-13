"""decode.py: two_stage_midpoint must never be worse than plain run_midpoint
on the loose-tau read -- same detected flag always, df_hat only ever
replaced by a tighter, more precise value when the tight-tau read also
fires. See docs/two_stage_findings.md for the Monte Carlo validation this
guarantee is built on."""
import numpy as np

from sim import decode as dec

DF_GRID = np.array([-100e3, -50e3, 0.0, 50e3, 100e3, 150e3])


def test_falls_back_to_loose_when_tight_empty():
    m_loose = np.array([False, True, True, True, False, False])
    m_tight = np.array([False, False, False, False, False, False])
    res = dec.two_stage_midpoint(m_loose, m_tight, DF_GRID)
    expected = dec.run_midpoint(m_loose, DF_GRID)
    assert res.detected == expected.detected
    assert res.df_hat == expected.df_hat


def test_uses_tight_run_when_available():
    m_loose = np.array([False, True, True, True, True, False])
    m_tight = np.array([False, False, True, True, False, False])
    res = dec.two_stage_midpoint(m_loose, m_tight, DF_GRID)
    expected_tight = dec.run_midpoint(m_tight, DF_GRID)
    assert res.detected
    assert res.df_hat == expected_tight.df_hat


def test_never_detects_when_loose_is_empty():
    """Even if (unrealistically) m_tight had bits set, an empty loose read
    means no detection -- tau2 is a refinement stage, not a rescue."""
    m_loose = np.array([False, False, False, False, False, False])
    m_tight = np.array([False, True, False, False, False, False])
    res = dec.two_stage_midpoint(m_loose, m_tight, DF_GRID)
    assert not res.detected


def test_matches_loose_detected_flag_across_random_vectors():
    rng = np.random.default_rng(0)
    for _ in range(200):
        m_loose = rng.random(6) < 0.4
        m_tight = m_loose & (rng.random(6) < 0.5)  # tight is a subset-ish, realistic
        res = dec.two_stage_midpoint(m_loose, m_tight, DF_GRID)
        expected_detected = dec.run_midpoint(m_loose, DF_GRID).detected
        assert res.detected == expected_detected
