# Findings

This documents (1) the one deviation from the spec's literal default that the
mandated Figure 1/2 sanity gate forced, and (2) the simulation's answers to
the open questions in spec section 10. Numbers below are from the runs
committed in `figures/` and `cache/` (seed 42 unless noted); re-run with
`--force --trials <bigger>` to refresh at publication-grade sample counts.

## The sanity-gate finding: differential lag

Spec section 2.1 gives the differential product as `z[n] = r[n]*conj(r[n-1])`
and section 2.3's headline example (`W = 159*4 = 636` for the OSR=4, N_sym=40,
B=2 default) is only consistent with a **1-sample** lag (`159 = 40*4 - 1`).
Implemented literally, this lag is too short to be useful: the CFO-induced
phase bias is `2*pi*df*Ts_sample`, and at OSR=4 (`Ts_sample` = 0.25 us) the
maximum bias across the whole +-150 kHz BLE range is only **+-0.235 rad** --
far smaller than the pi/2 needed to flip a decision boundary for most bits.
Empirically, the codebook collapses:

```
tests/test_codebook.py::test_literal_delay1_resolution_is_degenerate
-> N unique rows <= 6, for ANY requested N from 8 to 128
```

Figure 2 built this way is not a diagonal band; it's three flat blocks
(very-negative / middle / very-positive), because most codebook rows in the
middle 190 kHz are bit-for-bit identical. That fails the sanity gate outright.

**Fix used throughout the rest of this simulation:** `diff_delay` is a
first-class parameter (`encode.py`, `codebook.py`), and the operating default
is **one full symbol** (`diff_delay = OSR`), not one sample. This is still
exactly the same theta0-cancelling differential-product structure -- only the
lag changes -- and raises the per-bit bias 4x (to `2*pi*df*Ts_symbol`,
+-0.94 rad max at the default OSR=4), which is enough to produce a real,
graded response. With this fix:

```
N unique rows at N=8/16/32/64/128 requested: 8/11/17/19/21 (saturates ~20)
```

and Figure 2 shows a genuine contiguous band whose center tracks the true
CFO (see `figures/fig2_fired_set.png`).

**Consequence for the headline area number:** W is 624 bits (not 636) at the
operating default (OSR=4, N_sym=40, B=2, one-symbol lag), since one fewer
sample fits before the first tap: `(40*4 - 4) * 4 = 624`. Both configurations
are exercised by unit tests (`test_codebook_shape_matches_key_width_*`); the
636 number is correct as a literal reading of the spec's arithmetic, it's
just not the configuration that passes the sanity gate.

We did not go looking for this -- it's exactly what "stop and review" in the
spec's Section 9 order-of-work is for. The lag is exposed as an ordinary
sweepable parameter, so a future run can revisit the tradeoff (e.g. against a
half-symbol lag) if the layout team has a reason to.

## Open questions (spec section 10)

**1. Is the fired set reliably contiguous, or does it fragment?**
Mostly contiguous once there's enough SNR to detect at all. At the Fig 4/Table
I operating point (tau=142, N=32, W=624):

| Eb/N0 | P(detect) | P(fragmented \| detected) |
|---|---|---|
| 6 dB | 0.010 | 0.167 |
| 10 dB | 0.977 | 0.041 |
| 14 dB | 1.000 | 0.002 |
| 20 dB | 1.000 | 0.000 |

Fragmentation is concentrated right at the detection threshold (where there's
barely enough margin to fire at all) and is negligible by 14 dB. Run-midpoint
is a reasonable default estimator; centroid mainly earns its keep in that
narrow near-threshold band, which also matches the `run_length` data --
short/fragmented runs cluster at low SNR, supporting the "run length as a
free confidence metric" claim.

**2. What does the binary readout cost versus true argmin?**
More than the spec hoped. At the Table I operating point (Eb/N0=10 dB):

| System | RMS CFO error |
|---|---|
| CAM (run-midpoint, binary readout) | ~29-31 kHz |
| CAM (argmin over true distances, upper bound) | ~26-30 kHz |
| B1/B2 (full complex correlator, argmax) | ~2.5-2.8 kHz |

The binary-readout tax specifically (run-midpoint vs. this encoding's own
argmin) is small -- a few kHz, i.e. the "under ~1 dB"-style result the spec
hoped for *does* hold for that specific comparison. But argmin over the
CAM's own Hamming distances is itself an order of magnitude worse than the
full-precision correlator baselines. That gap isn't the binary-readout tax;
it's the cost of the thermometer/differential encoding discarding amplitude
and most of the phase information before any distance is ever computed. Fig 1
already hints at this: the whole +-150 kHz sweep moves the mean Hamming
distance by only ~10-15% of W (e.g. ~131 to ~136 bits out of 624 at 10 dB),
so the Hamming-distance "SNR" available to any decoder -- argmin included --
is inherently weak. **This is the honest framing for the paper: the CAM
architecture's win is constant-latency, area-cheap coarse acquisition, not
CFO precision competitive with a correlator bank; fine CFO tracking still
needs a downstream refinement stage.**

**3. Does one global tau suffice, or do edge rows need different tolerance?**
Not quite uniform, but not in the direction one would guess. At tau=142,
nearest-row detection probability at Eb/N0=10 dB:

| Hypothesis | P(detect) |
|---|---|
| center (df=0) | 0.757 |
| edge (df=+140 kHz) | 0.915 |
| edge (df=-140 kHz) | 0.910 |

Center rows are *harder* to detect than edge rows at the same tau (the
CFO-sensitive bit population isn't uniformly distributed across the range --
see the sanity-gate discussion above). The gap closes by ~12 dB (0.985 vs.
1.000). So: a single global tau works, with a few dB of extra margin needed
near the center of the range; don't-care masking (implemented, `codebook.py`
`dont_care_frac`) is the mechanism available if a design wants to equalize
this rather than over-margin the whole array.

**4. Minimum N_sym for +-150 kHz coverage?**
Using the operating (symbol-spaced-lag) configuration, per-config-calibrated
tau (target Pd=0.6 at 10 dB), nearest-row detection:

| N_sym | W | Pd @ 8 dB | Pd @ 10 dB | Pd @ 12 dB |
|---|---|---|---|---|
| 8 (preamble only) | 112 | 0.173 | 0.547 | 0.810 |
| 20 | 304 | 0.097 | 0.643 | 0.967 |
| 32 | 496 | 0.083 | 0.657 | 0.983 |
| 40 (preamble+AA, default) | 624 | 0.080 | 0.663 | 0.993 |

All configurations can be calibrated to ~the same Pd at 10 dB (that's what
the tau calibration does), but the *transition* gets sharper with more
symbols -- N_sym=40 needs roughly 2-4 dB less margin than N_sym=8 for
comparable high-confidence detection, consistent with the ~7.4 dB
processing-gain ceiling implied by W scaling 624/112 (`10*log10(624/112)`).
**N_sym=40 (the full preamble+AA) is the right default; preamble-only
acquisition works but should budget extra SNR margin.**

**5. Is OSR=4 necessary, or does OSR=2 suffice?**
OSR=1 fails distinctly and for the reason the spec anticipated -- not lack of
sensitivity, but timing: at OSR=1 the random sub-sample timing offset (up to
1 full sample = 1 full symbol) causes samples to land on essentially
arbitrary points in the pulse trajectory. Nearest-row Pd is flat and stuck
around 0.55 **regardless of SNR** (0.553 / 0.537 / 0.560 at 8/10/12 dB) --
the signature of a floor set by misalignment, not noise:

| OSR | W | Pd @ 8 dB | Pd @ 10 dB | Pd @ 12 dB |
|---|---|---|---|---|
| 1 | 156 | 0.553 | 0.537 | 0.560 (flat -- timing-limited, not SNR-limited) |
| 2 | 312 | 0.233 | 0.570 | 0.770 |
| 4 (default) | 624 | 0.080 | 0.680 | 0.970 |
| 8 | 1248 | 0.007 | 0.623 | 1.000 |

OSR=2 works (unlike OSR=1) but has a visibly less sharp transition than
OSR=4 at matched calibration; OSR=8 doubles W again for limited extra benefit
in this range. **OSR=4 is a reasonable default; OSR=2 is a viable
area-reduction option if a design is willing to trade some margin, but OSR=1
is not usable regardless of don't-care masking on its own** (the failure is
structural misalignment, not a few noisy bits), agreeing with the spec's
expectation.

## Robustness extra: flat Rayleigh fading (spec section 1.2, optional)

Quick check with the best validated configuration (composite vernier key,
`docs/vernier_findings.md`), `channel.py`'s existing `fading=True` option
(a single random complex gain applied to the whole packet), checked across
3 seeds at 400 trials/point:

| Eb/N0 | Pd, AWGN only | Pd, +Rayleigh fading | RMS, AWGN only | RMS, +fading |
|---|---|---|---|---|
| 10 dB | 1.000 | 0.59-0.63 | ~10.5-10.9 kHz | ~10.6-11.9 kHz |
| 18 dB | 1.000 | 0.91-0.93 | ~13.8-13.9 kHz | ~13.0-13.2 kHz |

Clean and reproducible: fading costs real detection probability (a classic
Rayleigh outage effect -- deep fades drop effective SNR below the detection
threshold for that trial), but **RMS accuracy among the trials that DO
detect is essentially unchanged** (differences are within run-to-run noise,
not a systematic degradation). The likely mechanism: a fading gain is a
single random *complex* multiplier on the whole packet, i.e. a random
amplitude AND a random phase -- and the differential product that cancels
theta0 (spec section 2.1) cancels a constant multiplicative phase exactly
the same way, for exactly the same reason. So fading only ever costs SNR
margin (via the amplitude component), never accuracy (the phase component
is free). This is a real, positive property worth stating explicitly in
the paper rather than leaving as an implicit consequence of the theta0
design: the same cancellation that handles unknown carrier phase also
handles unknown fading phase, for free.
