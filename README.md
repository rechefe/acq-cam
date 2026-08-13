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

## Read this next

**`docs/findings.md`** documents one load-bearing deviation from the spec's
literal default (the differential-product sample lag) that the Fig 1/2
sanity gate required, and the simulation's answers to the open questions in
spec section 10 -- including a negative result (the binary CAM readout costs
much more than the ~1-5 dB the spec hoped for) that matters for how the paper
frames its contribution.
