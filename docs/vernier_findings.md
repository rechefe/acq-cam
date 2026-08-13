# Composite ("vernier") key: exploration and results

Follow-up to the encoder-sensitivity study (`docs/findings.md`, `figures/encoder_sensitivity.md`),
which found that any diff_delay D>1 symbol aliases within +-150 kHz — worse
than the naive "D>3" estimate, since individual bits wrap past their own
nearest decision axis well before the bulk phase reference does. That
finding used a specific test: aggregate Hamming distance from a fixed
reference row (row 0) must be non-decreasing as CFO increases. This
document distinguishes what that test actually catches from what a receiver
actually needs to avoid, and shows a fix.

## Two different failure modes, not one

"Non-monotone aggregate distance from row 0" was being used as a proxy for
"the codebook is broken," but it conflates two different things:

1. **Local resolution saturation.** A contiguous run of neighboring rows
   collapses to the same (or nearly the same) code because the encoding is
   too coarse to distinguish them. Benign: a receiver reporting "somewhere
   in this ~20 kHz neighborhood" is still correct, just imprecise.
2. **Genuine far collision.** Two rows on *opposite sides* of the range
   produce nearly identical codes. Dangerous: the receiver can lock onto
   the wrong CFO by hundreds of kHz with no indication anything is wrong.

Splitting these apart (`_has_genuine_collision` in `tests/test_vernier.py`:
flag a pair only if the rows are >75 kHz apart in CFO *and* <=5% of W apart
in Hamming distance) changes the picture substantially:

| Config | Genuine far collisions (of 128 rows) | Worst case |
|---|---|---|
| D=1 (baseline) | 20/128 | weak sensitivity, not a fold |
| D=2 | 0/128 | none |
| D=3 | 0/128 | none |
| D=4 | 48/128 | row at -150 kHz == row at +100.4 kHz (distance 0!) |

D=2 and D=3 "alias" in the sense that flagged the original study (ordering
breaks), but they do **not** produce dangerous far collisions. D=4 does,
badly. This means D=2/D=3 were unfairly tarred by the same "aliased" label
as D=4 — they just aren't safe to decode with the naive run-midpoint
ordering assumption on their own.

## The fix: concatenate a coarse and a fine segment

A composite key concatenates a coarse D=1 segment (weak sensitivity, proven
collision-free) with a fine D=3 segment (strong sensitivity, but 0/128
far-collision-free on its own too — D=3 was already safe, it's D=4 that
needed rescuing). Because Hamming distance is additive over concatenation, a
row that only matches the fine segment still pays the coarse segment's
mismatch penalty, so it can't sneak under tau. No CAM or decoder changes
are needed — `cam.py` and `decode.py` are untouched; this is purely a key
construction change (`sim/vernier.py`).

Result, tested against **D=4 alone** (the worst case) composited with D=1:

```
tests/test_vernier.py::test_composite_key_eliminates_far_collisions[4]  PASSES
```

0/128 far collisions, vs. 48/128 for D=4 alone. The fix works even for the
single-lag configuration that was completely broken by itself.

## Does it actually help under noise? Yes — a clean win, not just a safety fix

Matched-bit-budget comparison: composite D1B2+D3B2 (W=1216) vs. the closest
single-lag alternative, D1B3 (W=1248) — same N=32, same Pfa-budget tau
calibration, same seed, 1500 trials/point:

| Eb/N0 | Composite Pd | Composite RMS (kHz) | Single-lag Pd | Single-lag RMS (kHz) |
|---|---|---|---|---|
| 6 | 0.363 | 11.0 | 0.013 | 22.0 |
| 8 | 0.971 | 12.0 | 0.569 | 20.0 |
| 10 | 1.000 | 10.5 | 0.995 | 24.8 |
| 14 | 1.000 | 13.3 | 1.000 | 31.5 |
| 20 | 1.000 | 15.0 | 1.000 | 31.0 |

At the **same bit cost**, the composite key detects ~2 dB earlier and is
**roughly 2x more accurate** on CFO across the whole SNR range — and this
isn't just a safety patch for D=4, it beats D3-alone's natural single-lag
peer at matched W. False-alarm rate stays controlled for both (Pfa <= 0.018
throughout, both well under the 0.01-0.02 budget used for calibration).

Fired-set structure (noiseless-construction check, mirroring Fig 2's
methodology) also confirms the composite key gives a genuinely
contiguous, monotonically-tracking band at every true-CFO test point across
the full range — not just locally around zero.

## What this means for the paper

- The original "D>1 always aliases, avoid it" framing from the
  encoder-sensitivity study was too pessimistic. D=2 and D=3 are
  collision-safe; they just can't be decoded with the naive single-segment
  run-midpoint assumption.
- A composite/vernier key is a legitimate, low-risk way to buy back most of
  the accuracy lost to the binary readout (see `docs/findings.md`'s
  argmin-vs-correlator gap) without changing the CAM primitive, the
  decoder, or the false-alarm profile.
- This is worth its own paragraph or subsection in the paper: "aliasing" as
  reported by a naive monotonicity check overstates the actual risk, and a
  vernier construction recovers much of the lost accuracy for free (no
  extra CAM logic, just a longer, differently-composed key).

## Not yet done

- Only one composite split (B=2+B=2, D=1+D=3) was tuned by hand. A proper
  sweep over the coarse/fine bit-width ratio and fine D would likely do
  better still — this was a screen, not an optimization.
- Not tested: composite keys with 3+ segments, or pairing with the
  preamble-masking result (`figures/performance_*_maskpre.md`) — masking
  the preamble on top of the composite key is a natural next combination.
- Decoding here still uses plain run_midpoint over the full concatenated
  row ordering. A decoder that explicitly reads the coarse segment first to
  pick a coarse bin, then refines with the fine segment, was not built or
  compared -- it might do even better than the "let the CAM sort it out"
  approach used here.
