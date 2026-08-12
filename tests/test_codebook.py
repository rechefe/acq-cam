"""Codebook row uniqueness and delta-vs-capture-window sanity checks.

Operating default uses a one-symbol (OSR-sample) differential lag -- see
docs/findings.md for why: the literal adjacent-sample lag (delay=1) implied
by the spec's headline W=636 example was found, via the Figure 1/2 sanity
gate, to give a maximum per-bit CFO bias of only ~0.24 rad across the whole
+-150 kHz BLE range (2*pi*df*Ts_sample), far too small to produce a
contiguous, tracking fired set. A one-symbol lag (Ts_sample -> Ts_symbol,
4x larger bias at OSR=4) fixes this while keeping the exact same
theta0-cancelling differential-product structure. diff_delay is a first-class
sweepable parameter (spec 8/encode.py); both configurations are exercised in
tests below.
"""
import numpy as np
import pytest

from sim.codebook import build_codebook, capture_window_half_width

OP_DELAY = 4  # one symbol at the default OSR=4


def test_codebook_rows_unique():
    cb = build_codebook(N=32, df_min=-150e3, df_max=150e3, n_sym=40, osr=4, B=2,
                         diff_delay=OP_DELAY)
    rows_as_tuples = {tuple(row.tolist()) for row in cb.rows}
    # resolution saturates (finite number of distinct decision-boundary
    # crossings in a fixed 40-symbol pattern); most, not all, rows at N=32
    # are expected unique.
    assert len(rows_as_tuples) >= 0.5 * cb.N


def test_codebook_shape_matches_key_width_literal_spec_example():
    """The spec's headline number (W=159*4=636) corresponds to the literal
    adjacent-sample (delay=1) differential lag; verify the arithmetic."""
    from sim.encode import key_width
    assert key_width(40, 4, 2, diff_delay=1) == 636


def test_codebook_shape_matches_key_width_operating_config():
    from sim.encode import key_width
    cb = build_codebook(N=16, df_min=-150e3, df_max=150e3, n_sym=40, osr=4, B=2,
                         diff_delay=OP_DELAY)
    assert cb.W == key_width(40, 4, 2, diff_delay=OP_DELAY)
    assert cb.W == 624


def test_codebook_dont_care_mask_shape():
    cb = build_codebook(N=8, df_min=-150e3, df_max=150e3, n_sym=40, osr=4, B=2,
                         dont_care_frac=0.1, diff_delay=OP_DELAY)
    assert cb.mask is not None
    assert cb.mask.shape == cb.rows.shape
    frac_masked = cb.mask.mean()
    assert 0.05 < frac_masked < 0.15


def test_delta_smaller_than_capture_window_gives_contiguous_overlap():
    """Empirically verify the design rule: delta should be < Wc so adjacent rows
    overlap in coverage (several rows fire per hypothesis)."""
    cb = build_codebook(N=64, df_min=-150e3, df_max=150e3, n_sym=40, osr=4, B=2,
                         diff_delay=OP_DELAY)
    tau = int(0.20 * cb.W)
    mid = cb.N // 2
    wc = capture_window_half_width(cb.df_grid, cb.rows, tau, mid)
    assert wc > cb.delta, "capture window should exceed grid spacing at this tau"


def test_literal_delay1_resolution_is_degenerate():
    """Documents the negative finding: with delay=1 (adjacent-sample lag), the
    codebook collapses to only a handful of distinguishable rows across the
    full +-150 kHz range, regardless of how many rows are requested."""
    cb = build_codebook(N=64, df_min=-150e3, df_max=150e3, n_sym=40, osr=4, B=2,
                         diff_delay=1)
    n_unique = len({tuple(r.tolist()) for r in cb.rows})
    assert n_unique <= 6
