# GPS L1 C/A acquisition on the CAM -- findings

Companion to `docs/findings.md` (the BLE study). Same primitive, different
application, and it fits better -- for one specific reason. The BLE result is
limited by *estimation*: the CAM's CFO estimate is ~10x worse than a correlator
bank because the binary readout discards it. GNSS acquisition does not want an
estimate. It wants a detection plus a coarse bin index, handed to a tracking loop
that refines it anyway, so the binary-readout tax is close to free here.

Operating point throughout: +-10 kHz cold start, 38-42 dB-Hz, 2.046 MHz sampling
(OSR=2), 2-bit I/Q, 32 PRNs. All numbers from 3000 trials/point, seed 42.

## Architecture

Three search dimensions, placed by one principle: **sweep the rotations that are
free in the quantized domain, store the ones that are not.**

- **Code phase** (2046 hypotheses) -- shift register, one per clock.
- **Carrier phase theta0** (4) -- *swept*. A 90-degree rotation is
  `(I,Q) -> (-Q,I)`: on sign or thermometer codes that is a rail swap plus an
  inversion, exact, with no multiplier and no re-quantization. Free.
- **Doppler** (5-41) -- *stored as rows*. A phase ramp is not free on a 2-bit
  key (it needs a real rotator and costs re-quantization loss) but is free on a
  precomputed row.

Hamming distance *is* a 1-bit correlation: `W - 2*HD = sum key_i * row_i` in the
+-1 domain. What the array lacks is not the multiply but the **magnitude**.
OR-ing the four theta0 fires computes `max_k Re{r e^-j.theta_k}`, an L-infinity
approximation to `|r|` -- the magnitude operator built from rows and an OR
instead of squarers.

## Three things the sanity gates caught

**1. Rows must be built at theta = 45 degrees, not 0.** At theta_k = 0 the
replica is purely real, so the row's Q rail collapses to a constant: half the
match line contributes variance without signal. Measured margin was a clean
0.66x of theory, and 0.33x when the signal sat at 45 degrees -- consistent ratios
across C/N0, which is what marks a deterministic bug rather than noise. With rows
at 45 degrees both rails carry the code and measured margin tracks the analytic
`sqrt(2*C/N0*T)*0.798*cos(phase error)` to within 5%. The free 90-degree sweep
then bounds the phase error at 45 degrees: 0-3 dB of ripple, never blind.

**2. The Doppler grid needs `2*ceil(2*f_max*T_coh)+1` rows.** The un-rounded
`4*f_max*T_coh + 1` understates it whenever f_max is not a whole number of bins
(7 rows, not 6, at T_coh = 125 us).

**3. A segment must be a whole number of samples,** so the segment count divides
2046 = 2*3*11*31. The available S are {2, 3, 6, 11, 22, 31, 33} -- *not* powers
of two. The natural ladder is S = 1, 2, 3, 6, 11.

## The segmentation trade

Segmenting the match line into S sub-lines with per-segment thresholds and a
>= k-of-S popcount shortens the coherent interval, and the Doppler dimension of
the dictionary shrinks with it. Hamming distance is *additive* across segments,
so summing segment distances would be exactly the unsegmented detector and would
need all 41 rows; the Doppler tolerance appears only because each segment is
thresholded *before* combining. That hard decision is what costs sensitivity.

| S | T_coh | Doppler rows/PRN | dictionary | dwells @ 38 dB-Hz | dwells @ 42 dB-Hz |
|---|---|---|---|---|---|
| 1 | 1000 us | 41 | 5.37 Mbit | 17 ms | 6 ms |
| 2 | 500 us | 21 | 2.75 Mbit | 23 ms | 7 ms |
| 3 | 333 us | 15 | 1.96 Mbit | 27 ms | 7 ms |
| 6 | 167 us | 9 | 1.18 Mbit | 42 ms | 10 ms |
| 11 | 91 us | 5 | 0.65 Mbit | 65 ms | 13 ms |

**8.3x less dictionary costs 3.8x more time at 38 dB-Hz, 2.2x at 42.** A
pre-simulation hand estimate put the hard-decision cost at ~2 dB; measured it is
up to 5.8 dB at S=11, so the estimate was optimistic and the trade is harsher
than assumed -- though still worth taking, because cold-start acquisition time
budgets are generous while array area is not.

## Where the loss actually goes

| C/N0 | B1 full-precision FFT | B1' 1-bit FFT | CAM S=1 | CAM S=11 |
|---|---|---|---|---|
| 38 dB-Hz | 6 ms | 11 ms | 17 ms (+1.9 dB) | 65 ms (+7.7 dB) |
| 42 dB-Hz | 2 ms | 4 ms | 6 ms (+1.8 dB) | 13 ms (+5.1 dB) |

B1 -> B1' is the textbook 1.96 dB limiting loss, which any hard-limited GNSS
front end already pays and which is therefore not chargeable to the CAM.
**B1' -> CAM at S=1 is +1.9 dB: that is the true boolean-readout tax, and it is
a far better number than the BLE study's.** Everything beyond it is the
segmentation trade, bought deliberately in exchange for dictionary size.

For scale, the FFT engine costs 32.2 M complex MAC per dwell for the full 2-D
search; the CAM does all 32 PRNs and all Doppler rows every cycle.

## Required match-line resolution

The paper's central hardware question, and the sharpest difference from the BLE
work. There a match sits at ~23% of W, comfortably below chance. Here the correct
row sits *barely* below chance, because the processing gain is spread thinly
across the whole word -- so the sense amplifier must resolve a small fraction of
the match line. Measured as the tau band keeping acquisition within 10% of its
best dwell count:

| S | segment width | resolution @ 38 dB-Hz | resolution @ 42 dB-Hz |
|---|---|---|---|
| 1 | 4092 bits | 1 part in 128 | 1 part in 105 |
| 2 | 2046 bits | 1 part in 89 | 1 part in 71 |
| 3 | 1364 bits | 1 part in 97 | 1 part in 80 |
| 6 | 682 bits | 1 part in 68 | 1 part in 49 |
| 11 | 372 bits | 1 part in 53 | 1 part in 41 |

So ~1% of the match line unsegmented, relaxing to ~2.4% at S=11. **Segmentation
helps on both axes** -- fewer Doppler rows *and* a coarser sense amplifier -- and
time is what pays for both. That is the design's main degree of freedom.

## Open questions, answered

**Does a wrong PRN behave like noise, or like the structured near-collision that
`docs/findings.md` found for wrong BLE access addresses?** Like noise. Segment
fire counts under a wrong PRN are statistically indistinguishable from a wrong
code phase (p_seg 0.108-0.119 vs 0.109-0.119), and the independent-segment
binomial model holds for all wrong-cell types (variance ratio 0.97-1.02 over 6000
trials). A single chi-square initially rejected the model for wrong-PRN at
p=0.003; that did not survive more trials and was a multiple-comparison artifact.
Gold cross-correlation is bounded at 65/1023, and it shows.

**Does the sense margin exist?** Yes, at 1 part in 41-128 depending on
segmentation. Demanding, but not the show-stopper the regime suggested it might
be, and the designer can buy margin with time.

## What is not done

- theta0 grid of 8 (45-degree spacing, ~0.7 dB ripple instead of 3 dB) via a
  16-entry LUT; and the complement trick (`HD(key, ~row) = W - HD(key, row)`),
  which halves the phase dimension if the array can threshold the upper tail.
- Thermometer mapping measured end-to-end. It is implemented and gated
  (Hamming == L1 in levels) but the sweeps here all use the sign mapping.
- Multipath, interference, and front-end AGC dynamics. All absent.
- Trial counts are 3000/point. The tail probabilities the 1.3 M-cell search needs
  are reached through the *validated* binomial model, not measured directly;
  p0 carries ~2% relative error at this trial count, which propagates into k.
