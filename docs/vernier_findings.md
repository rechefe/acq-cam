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

## Split sweep: is D1B2+D3B2 actually the best composite?

The original composite result above used a hand-picked split. A follow-up
swept the noiseless collision check (same methodology, `far_khz=75`,
`near_frac=0.05`) over D_fine in {2,3,4} x B_coarse/B_fine in {2,3,4} (18
configurations): **every single one was collision-free.** The composite
construction is robust to the split choice on the collision-safety axis.

That made W-efficiency the only apparent differentiator, and a quick screen
(~250 trials/point, heuristic tau=0.22*W instead of the project's proper
Pfa-calibrated tau) made **D_fine=4** combinations look clearly best —
smaller W, more unique rows, and RMS as low as 3-9 kHz across 10-18 dB.

**That lead did not survive full validation, and it's worth recording why.**
Re-run at the project's actual methodology (Pfa-calibrated tau at
worst-case SNR, 800-1500 trials/point, checked across 3 seeds):

| Fine lag | RMS trend with SNR | Verdict |
|---|---|---|
| D_fine=2 | 14 -> 24 kHz, worse than D_fine=3 throughout | stable but mediocre |
| **D_fine=3** | **11 -> 15 kHz, gently increasing, consistent across seeds** | **stable and best** |
| D_fine=4 | 6 kHz at low SNR -> 20-38 kHz at high SNR, non-monotone, direction varies by seed | reproducibly unstable |

D_fine=4's RMS *increasing* with SNR is backwards — accuracy should never
get worse as noise drops. The likely mechanism: run_midpoint selects the
*longest* contiguous fired run, not the closest one, and D=4's fine segment
folds aggressively enough that at low noise a specific wrong-but-stable run
can consistently win the length contest, whereas at moderate noise that
near-tie gets broken randomly (helping about as often as it hurts). This
wasn't caught by the noiseless collision check (which only asks whether two
rows are near-identical, not whether the *decoder* reliably prefers the
right one) or by the quick screen (too few trials, wrong tau methodology,
and the SNR points tested happened to sit before the degradation onset).

**Conclusion: D1B2+D3B2 (D_fine=3) remains the validated choice, and it
turns out to beat an even bigger target than initially tested.** The
comparison below adds a second baseline, single-lag D1B4 at W=2496 — *double*
the composite's bit budget:

| Eb/N0 | Composite D1B2+D3B2 (W=1216) RMS | Single D1B3 (W=1248) RMS | Single D1B4 (W=2496) RMS |
|---|---|---|---|
| 8 | 12.0 kHz | 20.0 kHz | 14.0 kHz |
| 12 | 11.5 kHz | 28.8 kHz | 27.3 kHz |
| 16 | 14.0 kHz | 31.4 kHz | 31.1 kHz |
| 20 | 15.0 kHz | 31.0 kHz | 31.5 kHz |

The composite key beats a single-lag baseline with **twice its bit budget**,
not just a matched one. **Methodological lesson for the rest of this
exploration:** a small-trial, heuristic-tau screen is only good enough to
shortlist candidates for the noiseless collision check; every RMS/Pd/Pfa
claim needs the full Pfa-calibrated, multi-hundred-trial pipeline, checked
across at least 2-3 seeds, before being reported as a result.

## Combining with preamble masking: a trade-off, not a free win

Unlike the single-lag case (`figures/performance_*_maskpre.md`, where
masking the preamble-derived ~18% of the key was a strict improvement --
better detection, same RMS, same Pfa), masking the composite key's
preamble-derived bits is a genuine trade-off. Checked at N=32, 800-1200
trials, two seeds:

| Eb/N0 | Unmasked RMS | Masked RMS | Unmasked Pd | Masked Pd |
|---|---|---|---|---|
| 6 | 10.6 kHz | 12.7 kHz | 0.27 | **0.81** |
| 10 | 10.3 kHz | 12.9 kHz | 1.00 | 1.00 |
| 16 | 13.4 kHz | 19.1 kHz | 1.00 | 1.00 |
| 20 | 14.2 kHz | 20.4 kHz | 1.00 | 1.00 |

Masking gives a real, reproducible low-SNR detection win (0.27 -> 0.81 at
6 dB, consistent across seeds) but costs 30-45% worse RMS from 10 dB
upward, also reproducible. The likely reason this differs from the
single-lag case: in the composite key, the *fine* (D=3) segment is what
carries the precise CFO information, and its preamble-derived samples,
while not useful for discriminating *which packet*, still carry real CFO
precision -- masking them removes signal that the single-lag encoding
didn't have to begin with (there, the masked segment was uniformly weak).

**Recommendation:** use the unmasked composite key as the default (best
RMS across the whole range, and still detects reasonably by 8-10 dB); only
add preamble masking if the application specifically needs faster
acquisition at very low SNR and can tolerate worse precision once acquired.

## Not yet done

- The split sweep above only tried B_coarse/B_fine in {2,3,4} with a
  2-segment key. Not tested: 3+ segment composites (e.g. coarse+medium+fine),
  or non-thermometer codings for the fine segment.
- Decoding here still uses plain run_midpoint over the full concatenated
  row ordering. A decoder that explicitly reads the coarse segment first to
  pick a coarse bin, then refines with the fine segment, was not built or
  compared -- it might do even better than the "let the CAM sort it out"
  approach used here, and might also be what actually fixes D_fine=4's
  instability (by not relying on longest-run-wins across the whole
  concatenated key).
