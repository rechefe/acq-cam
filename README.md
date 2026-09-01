# CAM-Based Joint Preamble Detection and CFO Estimation -- Simulation

Python simulation of a BLE 1M-PHY acquisition front end built on an
approximate-match CAM: a bank of CFO hypotheses is stored as CAM rows and
searched in one cycle, so preamble detection and coarse CFO estimation happen
simultaneously from a single binary match vector. Produces the 5 figures and
comparison table for the ISCAS 2027 submission.

## Install

```
pip install -r requirements.txt
```

## Run the tests (do this first)

```
python -m pytest tests/ -v
```

Covers: the circular-thermometer Hamming-distance metric property (B=2,3),
theta0-cancellation of the differential product, the CAM primitive against a
brute-force reference, and codebook row uniqueness.

## Generate figures

```
python -m sim.figures 1              # Fig 1: distance profile
python -m sim.figures 2              # Fig 2: fired-set structure
python -m sim.figures 3              # Fig 3: encoding comparison
python -m sim.figures 4              # Fig 4: performance vs SNR
python -m sim.figures 5              # Fig 5: threshold/grid design
python -m sim.figures table          # Table I
python -m sim.figures all            # everything
```

Options: `--trials N` (Monte Carlo trials per operating point, default varies
by figure -- see below), `--seed N` (default 42), `--force` (bypass the npz
cache in `cache/` and recompute). Figures are written to `figures/` as both
vector PDF (IEEE two-column format: 3.5in wide, 8pt font) and PNG for quick
viewing.

Sweep results are cached to `cache/*.npz` keyed by a hash of every parameter
that affects the result, so re-running the same figure command is instant,
and changing any parameter (trial count, seed, config) automatically
recomputes rather than silently reusing stale data.

**Trial counts.** Spec section 7 asks for >=10,000 trials per detection-curve
operating point and >=100,000 for false-alarm rates below 1e-3. The commands
above default to a few hundred to a few thousand trials so the full figure
set regenerates in well under a minute on a laptop; the numbers shipped in
`figures/` were regenerated with `--trials 2000` to `--trials 5000` (`--trials
1500` for Fig 5, which sweeps a 7x5 grid). Pass a larger `--trials` for a
publication-grade run; nothing else needs to change.

## Package layout

| Module | Responsibility |
|---|---|
| `sim/tx.py` | BLE 1M GFSK transmitter, preamble + access address bits |
| `sim/channel.py` | AWGN, CFO, random theta0/timing offset, optional fading/drift |
| `sim/encode.py` | Differential product, circular thermometer/Gray/one-hot codes, key assembly |
| `sim/cam.py` | **The CAM primitive and nothing else** -- binary match vector only |
| `sim/codebook.py` | Hypothesis row construction, don't-care (ternary) masking |
| `sim/decode.py` | any-fire / run-midpoint / centroid / argmin-baseline decoders |
| `sim/baselines.py` | B1 (sequential correlator), B2 (parallel correlator bank) |
| `sim/experiments.py` | Monte Carlo sweep runners + npz caching |
| `sim/plots.py`, `sim/figures.py`, `sim/figures_more.py` | IEEE-format plotting, CLI |

`cam.py`'s `CAM.query()` is the only method on the only public class in that
module, and it returns strictly a boolean match vector -- no distance, rank,
or argmin is ever returned by that module (`tests/test_cam.py` enforces this
via `inspect`, not just convention). The one place true Hamming distances are
used is `decode.argmin_baseline`, which bypasses `CAM` entirely and is
labeled everywhere as a reference-only, unachievable-in-hardware upper bound.

## Encoder-sensitivity study (diff_delay x B)

```
python -m sim.encoder_sensitivity              # --trials 500 --seed 42 by default
```

Noiseless codebook-construction sweep over diff_delay D in {1,2,3,4} symbols
x B in {2,3,4} (N=128 requested hypotheses, +-150 kHz): unique row count,
minimum adjacent Hamming distance, total endpoint-to-endpoint Hamming swing,
W, and a monotonicity/aliasing check, plus an "encoder SNR" (swing / binomial
noise std at Eb/N0=10 dB) ranking. Writes `figures/encoder_sensitivity.md`
and `figures/encoder_sensitivity_heatmap.pdf`.

**Headline result:** a naive bulk-phase estimate suggests only D=4 should
alias within +-150 kHz; empirically **D=2 and D=3 alias too**, at every B
tested -- individual bits wrap past their own nearest decision axis (a much
finer threshold than the bulk estimate assumes) well before the bulk phase
reference completes a half turn. Only **D=1 (the operating default used
everywhere else in this repo) stays monotone across the full BLE range**,
which is reassuring given the rest of the simulation depends on it. Among
the non-aliased configurations, larger B gives a higher encoder SNR (D=1,
B=4 wins); see `figures/encoder_sensitivity.md` for the full table and the
caveat on why the aliased configs' raw ranking is not usable as-is.

## Follow-up explorations for the paper

Four further screens, run to find operating points that improve on the
project's headline numbers rather than just report them. Three panned out
cleanly, one revealed and then corrected a methodology trap along the way --
recorded because the correction matters as much as the result. **Best
validated result overall: composite vernier key + two-stage refinement
(last item below), ~10-13 kHz RMS vs. the original ~30 kHz.**

- **`python -m sim.performance_at_config ... --mask-preamble`**: marking the
  preamble-derived ~18% of the key as CAM don't-cares (only two distinct BLE
  preambles exist, so that segment barely discriminates between packets)
  improves the detection transition by ~2 dB at both the default and the
  high-resolution config, at no cost to Pfa or RMS, using fewer bits. See
  `figures/performance_*_maskpre.md`.
- **`python -m sim.vernier_experiment [--mask-preamble]`**: a composite
  ("vernier") key that concatenates a coarse D=1-symbol differential segment
  (weak but collision-free) with a fine D=3-symbol segment (strong but
  aliased alone) eliminates far-CFO false-lock collisions entirely -- even
  for D=4, which is catastrophic by itself -- and beats a single-lag
  baseline with *double* its bit budget by roughly 2x on RMS accuracy, with
  no Pfa cost. A split sweep (D_fine in {2,3,4}) initially made D_fine=4
  look even better in a quick screen, but that lead failed full validation
  (reproducible RMS *increase* with SNR -- a red flag caught before it
  shipped as a result); D_fine=3 remains the validated choice. See
  **`docs/vernier_findings.md`** for both the collision-mechanism writeup
  and that retraction.
- **`python -m sim.baseline_b3_experiment`**: B3, a 1-bit sign-sign
  correlator bank, added as a fair low-power peer to the CAM (B1/B2 use full
  complex precision, which the CAM never gets either). Once calibration was
  fixed to compare all three systems by the same Pfa-budget methodology
  (an initial run made B3 look like it beat full-precision B1/B2 outright,
  which turned out to be a calibration-methodology artifact, not a real
  result), B3 matches B1/B2's RMS accuracy almost exactly (~3 kHz either
  way) at a ~4-8 dB detection-threshold cost -- textbook-consistent for
  1-bit quantization, and it sharpens rather than softens the CAM's honesty
  gap: a binary-ish scheme *can* match full-precision accuracy, the CAM
  specifically does not. See **`docs/baseline_b3_findings.md`**.
- **`python -m sim.two_stage_experiment`**: query the CAM twice on the same
  key at two tau values in successive cycles -- a loose tau1 for detection,
  a tighter tau2 for refinement (`decode.two_stage_midpoint`). Both reads
  stay pure binary match vectors, so this is still within the "no distances,
  ever" constraint; it's tau re-applied as a second sense margin, the way
  real analog hardware would ramp it. By construction (falls back to the
  loose-tau result whenever the tight read is empty) detection probability
  is unchanged and RMS can only improve, never worsen -- confirmed a real
  10-24% RMS improvement from ~12 dB upward, reproducible across seeds.
  Stacked on top of the composite vernier key, this is the best validated
  CAM configuration in the repo (~10-13 kHz RMS). See
  **`docs/two_stage_findings.md`**.

## GPS L1 C/A acquisition (`sim/gps/`)

A second application of the same CAM primitive, and a better fit than BLE:
GNSS acquisition wants a detection plus a coarse bin index, not an estimate, so
the binary-readout cost that limits the BLE result is close to free.

- `docs/gps_algorithm_explained.md` -- plain-language version, no signal-processing
  background assumed. Start here.
- `docs/gps_findings.md` -- the results write-up for specialists.

```
python -m pytest tests/gps/ -v
python -m sim.gps.segmentation_study   [--trials 3000] [--seed 42] [--force]
python -m sim.gps.sense_margin_study   [--trials 3000] [--seed 42] [--force]
python -m sim.gps.comparison_study     [--trials 3000] [--seed 42] [--force]
```

The three studies share one Monte Carlo pass through the `cached_distances`
npz cache, so run `segmentation_study` first and the other two are instant.
Results are written to `figures/gps_*.md` plus `figures/gps_area_time_pareto.{pdf,png}`.

| Module | Responsibility |
|---|---|
| `sim/gps/prn.py` | Gold code generator, gated against the IS-GPS-200 octal table |
| `sim/gps/quantize.py` | 2-bit I/Q front end, sign and thermometer key mappings, the free 90-degree rotation |
| `sim/gps/channel.py` | Signal generation at a specified C/N0 |
| `sim/gps/codebook.py` | PRN x Doppler rows (built at theta=45 deg -- see findings) |
| `sim/gps/segment.py` | Segmented match lines, multi-dwell fire accumulator |
| `sim/gps/baselines.py` | FFT parallel-code-phase search, full-precision and 1-bit |
| `sim/gps/experiments.py` | Config, calibration, characterization pass, search sizing |

Placement principle behind the architecture: **sweep the rotations that are free
in the quantized domain, store the ones that are not.** A 90-degree carrier-phase
step is `(I,Q) -> (-Q,I)`, an exact bit permutation, so theta0 is swept and costs
no rows; a Doppler hypothesis is a phase ramp, free on a precomputed row but not
on a 2-bit key, so Doppler is stored.

Headline: the boolean-readout tax over an equivalent 1-bit correlator is
**+1.9 dB** (17 ms vs 11 ms to acquire at 38 dB-Hz), and segmentation trades
dictionary size against acquisition time along a measured curve -- 5.37 Mbit at
17 ms down to 0.65 Mbit at 65 ms.

## Read this next

**`docs/findings.md`** documents one load-bearing deviation from the spec's
literal default (the differential-product sample lag) that the Fig 1/2
sanity gate required, and the simulation's answers to the open questions in
spec section 10 -- including a negative result (the binary CAM readout costs
much more than the ~1-5 dB the spec hoped for) that matters for how the paper
frames its contribution. **`docs/vernier_findings.md`** and
**`docs/baseline_b3_findings.md`** document the two follow-ups above.
