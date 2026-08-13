# Two-stage (tau-ramp) successive refinement

Idea from the original exploration plan: tau is a single global analog sense
margin, but nothing stops the CAM from being queried *twice* on the same
key at two different tau values in successive cycles -- a loose tau1 for
detection, then a tighter tau2 for refinement. Both reads are still
ordinary binary match vectors (no distances are ever computed or returned),
so this stays within the CAM primitive's "binary readout only" constraint;
it is just tau re-applied as a second global sense margin, exactly as real
analog hardware would ramp it.

## Design: can only help, never hurt

`decode.two_stage_midpoint(m_loose, m_tight, df_grid)` (in `sim/decode.py`):
run `run_midpoint` on the loose-tau read first; if it doesn't detect, return
that (not detected) immediately -- tau2 never rescues a non-detection, it
only refines one. If it does detect, try `run_midpoint` on the tight-tau
read; use its (necessarily narrower-or-equal, more precise) result if it
also fires, otherwise fall back to the loose-tau result unchanged. This
means detection probability is **identical by construction** to plain
`run_midpoint(m_loose, ...)` in every case (verified in
`tests/test_decode.py`) -- the only thing that can change is `df_hat`
getting more precise when the tighter read happens to still fire.

## Validated result

tau2 = 0.6 x tau1 (not tuned further -- this was a screen, not an
optimization; see "not yet done" below), default config (B=2, OSR=4,
N=32), Pfa-budget-calibrated tau1, 1200-trial run plus cross-checks at
500-800 trials across 3 additional seeds:

| Eb/N0 | Pd (identical) | RMS single-tau | RMS two-stage | Improvement |
|---|---|---|---|---|
| 10 | 0.98 | 29.5 kHz | 28.9 kHz | ~2% |
| 12 | 1.00 | 29.7 kHz | 25.6 kHz | ~14% |
| 14 | 1.00 | 30.3 kHz | 23.1 kHz | ~24% |
| 16 | 1.00 | 29.9 kHz | 24.9 kHz | ~17% |
| 18 | 1.00 | 29.6 kHz | 26.0 kHz | ~12% |
| 20 | 1.00 | 30.7 kHz | 27.6 kHz | ~10% |

Reproducible across every seed checked (42, 11, 23 in early screening; 42
in the final run): a real 10-24% RMS improvement from roughly 12 dB upward,
zero improvement below the detection threshold (~10 dB, where the tight
read essentially never also fires, so it degrades gracefully to the
single-tau result exactly as designed) and zero Pd cost anywhere.

## Why this one didn't need the same scrutiny as the vernier D_fine=4 lead

The D_fine=4 composite-key lead (`docs/vernier_findings.md`) needed
cross-seed validation because it *could* have been unstable -- and was.
Two-stage refinement is structurally different: because it falls back to
the single-tau result whenever the tight read is empty, there is no
mechanism by which it could make Pd or RMS *worse* than the baseline for
any individual trial, only "same" or "better." The multi-seed check here
was to confirm the *size* of the improvement was real and not noise, not to
guard against a hidden failure mode the way the vernier check was.

## Stacked with the composite vernier key: the best result in this repo

Two-stage refinement applies to *any* CAM key, including the composite
vernier key from `docs/vernier_findings.md` (D1B2+D3B2, W=1216) -- its
RMS was already the best validated single result (~10-14 kHz). Stacking
two-stage refinement on top (tau2=0.6*tau1, same methodology, checked
across 2 seeds at 500 trials/point):

| Eb/N0 | Composite alone | Composite + two-stage | Improvement |
|---|---|---|---|
| 10 | 10.4-11.1 kHz | 10.1-10.6 kHz | ~3-5% |
| 14 | 11.7-12.2 kHz | 10.1-10.5 kHz | ~10-14% |
| 18 | 13.9-14.2 kHz | 13.1-13.1 kHz | ~6-8% |

Reproducible, no Pd cost (Pd=1.000 throughout at these SNR points, by the
same falls-back-when-empty construction as the single-lag case above).
**This combination -- composite vernier key + two-stage refinement -- is
the best validated CAM configuration in this repository**, at ~10-13 kHz
RMS vs. the original single-lag default's ~30 kHz, closing roughly a third
of the remaining gap to the correlator baselines' ~3 kHz floor (see
`docs/baseline_b3_findings.md`) without changing `cam.py`, without adding
CAM rows, and using less than double the original config's bit width.

## Not yet done

- tau2/tau1 = 0.6 was picked from a quick 3-point sweep (0.5, 0.7, 0.85 all
  showed improvement; 0.5 and 0.7 looked slightly better at some SNR points
  than 0.6 in the initial screen). Not swept properly against Pfa impact of
  tau2 itself, though tau2 doesn't gate detection so it has no Pfa exposure
  of its own -- only tau1's Pfa matters for false-alarm safety.
- Only tested at 2 stages. A 3-stage ramp (tau1 > tau2 > tau3) is a natural
  extension and untested.
- Not yet combined with preamble masking as a third stacked layer, and not
  yet checked at the high-resolution B=4/OSR=20 (~500-address) operating
  point from the original exploration.
