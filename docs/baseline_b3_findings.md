# B3: a fair low-power correlator peer for the CAM

Table I, as originally built, compares the CAM (a strictly binary readout)
against B1 and B2 -- both **full complex-precision** correlators. That's an
unfair comparison on its face: B1/B2 get amplitude and phase information the
CAM's architecture was never given access to. B3 closes that gap: same
N-hypothesis bank, same 1-cycle latency as B2, but both the received signal
and every reference waveform are quantized to 1-bit I/Q (`sign_quantize` in
`sim/baselines.py`) before correlating. Each per-sample product is then one
of four fixed {-1,+1}x{-1,+1} outcomes -- an XNOR/sign-flip, not a real
multiplier -- making B3 the honest low-power peer to a binary CAM.

## A methodology trap, caught before it produced a false result

The first run made B3 look like it *beat* B1/B2 outright (Pd=1.0 by 6-8 dB
vs. B1/B2 stuck at 0.85-0.97 even at 16 dB). That doesn't square with
communications theory -- 1-bit quantization should cost a few dB, not win.
Cause: B1/B2's threshold was calibrated with the project's existing
`calibrate_b_threshold` (a Pd-quantile method: pick the threshold so exactly
70% of matched-signal samples exceed it, evaluated once at 10 dB), while B3
used a proper Pfa-budget calibration. Those are different questions, and
comparing thresholds chosen by different methodologies is not a real
result -- it was an artifact of a Pd-quantile threshold getting pinned to a
timing-offset-induced plateau (B1/B2 never quite reaches Pd=1.0 for any
*fixed* threshold, because the random sub-sample timing offset caps peak
correlation below 1.0 regardless of SNR).

Fixed by giving B1/B2 the same Pfa-budget calibration as B3
(`_calibrate_full_threshold`, mirroring `_calibrate_sign_threshold`). While
fixing this, a second, unrelated methodology gap turned up: **the CAM's Pfa
gets worse at high SNR (established in `docs/findings.md`), but the
full-precision correlator's Pfa gets worse at LOW SNR** (measured: 9.3% Pfa
at 0 dB vs. ~0% at 16 dB, for a fixed mid-range threshold) -- the opposite
direction. A calibration that only checks the "obvious" end of the SNR
sweep silently under-protects the other end. Both `_calibrate_full_threshold`
and `_calibrate_sign_threshold` now check both ends of `ebn0_list` and take
the tighter constraint.

## Validated result (Pfa-budget calibration for all three systems, checked across 2 seeds, 800-1200 trials/point)

| System | Latency | Area proxy | Detection onset (Pd~1.0) | RMS once locked |
|---|---|---|---|---|
| CAM | 1 cycle | N x W (bitcells) | ~10-12 dB | ~25-32 kHz |
| B1/B2 (full-precision) | 1 cycle | N x N_s (complex MACs) | ~0-4 dB | ~2.7-3.2 kHz |
| B3 (1-bit sign-sign) | 1 cycle | N x N_s (XNOR ops) | ~6-8 dB | ~3.0-3.3 kHz |

Textbook-consistent and reassuring: **B3's RMS accuracy is essentially
identical to full-precision B1/B2** (~3 kHz either way) -- once a 1-bit
correlator locks on, it's just as precise as the full-precision version,
because precision here comes from the CFO grid spacing, not from amplitude
information. The cost of 1-bit quantization is entirely in the *detection
threshold*: B3 needs roughly 4-8 dB more Eb/N0 than B1/B2 to reach the same
Pd, consistent with the classical 1-bit-ADC correlation-loss literature.
Pfa stays controlled (<=3%, close to the 1% design target) for all three at
every SNR point tested.

**The CAM is far worse than both correlator variants on every axis** --
detection threshold, RMS accuracy -- which is the same honest finding as
`docs/findings.md`'s argmin-vs-correlator gap, now extended to a fairer,
lower-power peer instead of only a full-precision one. B3 demonstrates that
the accuracy gap is NOT simply "binary readout vs. full precision" -- a
different binary-ish architecture (1-bit sign correlation) gets full
correlator-like accuracy for a similar op-count to B2, while the CAM does
not. That sharpens, rather than softens, the honesty point from the
original findings: the CAM's real area story has to be about its N x W
bitcell array vs. N x N_s XNOR-op count specifically, not "binary vs.
analog" in general, since B3 shows a binary-ish scheme can still match
full-precision accuracy.

## One architectural point the op-count proxy misses

B3's area proxy matches B2's (N x N_s), but per-op cost and per-op
*output-stage* logic are not the same story. The CAM's real distinguishing
feature is that its N-row output is already a boolean match vector -- no
magnitude comparison across N correlator outputs is needed downstream,
just N independent one-bit compares against a single global sense margin.
B3 (and B1/B2) still need an argmax/magnitude-sort across N correlator
outputs to pick the best hypothesis, which is extra logic the op-count
proxy doesn't capture. This is worth a sentence in the paper's area
discussion so B3's near-parity in accuracy doesn't get read as "B3 is
strictly better and cheaper" -- it isn't cheaper on the output stage.

## Not yet done

- The area proxy still doesn't distinguish an XNOR from a complex MAC in
  actual gate count (both counted as "1 op"). A real silicon estimate would
  need a technology-specific gate-count pass, out of scope here (same
  caveat the project already carries for B1/B2's proxy).
- Only tested at the project's default (B=2, OSR=4, W=624) operating point.
  Worth checking whether B3's ~4-8 dB penalty relative to B1/B2 holds at the
  higher-resolution B=4/OSR=20 configuration too.
- 2-bit (not just 1-bit) quantization would be a natural intermediate point
  on the precision/area curve, not tried here.
