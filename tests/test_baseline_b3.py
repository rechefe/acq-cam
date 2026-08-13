"""B3 (1-bit sign-sign correlator): structural sanity checks. Full
Pd/RMS/Pfa validation lives in sim/baseline_b3_experiment.py and
docs/baseline_b3_findings.md (needs Monte Carlo, not a fast unit test)."""
import numpy as np

from sim import baselines as bl


def test_sign_quantize_only_four_outcomes():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(50) + 1j * rng.standard_normal(50)
    q = bl.sign_quantize(x)
    assert set(np.unique(q.real)) <= {-1.0, 1.0}
    assert set(np.unique(q.imag)) <= {-1.0, 1.0}


def test_sign_bank_matches_full_bank_shape():
    bank = bl.build_correlator_bank(N=8, df_min=-150e3, df_max=150e3, n_sym=40, osr=4)
    sign_bank = bl.build_sign_correlator_bank(bank)
    assert sign_bank.waveforms_q.shape == bank.waveforms.shape
    assert sign_bank.N_s == bank.N_s
    assert np.array_equal(sign_bank.df_grid, bank.df_grid)


def test_sign_correlation_peaks_at_matching_hypothesis_noiseless():
    """Noiseless sanity check: correlating a hypothesis waveform's own sign
    pattern against the bank must peak at its own row."""
    bank = bl.build_correlator_bank(N=16, df_min=-150e3, df_max=150e3, n_sym=40, osr=4)
    sign_bank = bl.build_sign_correlator_bank(bank)
    for k in [0, 5, 10, 15]:
        mags = bl.correlate_bank_sign(bank.waveforms[k], sign_bank)
        assert int(np.argmax(mags)) == k


def test_b3_result_has_same_latency_and_area_as_b2():
    bank = bl.build_correlator_bank(N=8, df_min=-150e3, df_max=150e3, n_sym=40, osr=4)
    sign_bank = bl.build_sign_correlator_bank(bank)
    res_b2 = bl.b2_parallel(bank.waveforms[3], bank, threshold=0.5)
    res_b3 = bl.b3_parallel_sign(bank.waveforms[3], sign_bank, threshold=0.5)
    assert res_b2.latency_cycles == res_b3.latency_cycles == 1
    assert res_b2.area_proxy == res_b3.area_proxy
