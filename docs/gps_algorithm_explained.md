# GPS L1 C/A acquisition on an approximate-match CAM

*For engineers. Assumes comfort with probability and linear algebra; assumes no
GNSS or DSP background — every domain term is defined at first use. The
specialist results write-up is [`gps_findings.md`](gps_findings.md).*

---

## 1. Signal model

GPS L1 C/A broadcasts, from each satellite, a distinct 1023-symbol binary sequence
(a *Gold code*, written `c[·] ∈ {−1,+1}`) at 1.023 Mchip/s, repeating every 1 ms.
The symbols are called *chips*; the sequence is a spreading code, not data. All 32
satellites transmit simultaneously on the same carrier.

After downconversion, the complex baseband the receiver sees is

$$r[n] \;=\; A\, c\big[(n-\tau) \bmod N\big]\, d \; e^{\,j\left(2\pi f_d n / f_s \,+\, \theta_0\right)} \;+\; w[n]$$

| Symbol | Meaning | Status |
|---|---|---|
| `τ` | **code phase** — where in the 1023-chip period we currently are | unknown, must search |
| `f_d` | **Doppler shift** — carrier offset from satellite motion, ±10 kHz cold start | unknown, must search |
| `θ₀` | **carrier phase** — where the carrier sits in its cycle at n=0 | unknown, nuisance |
| `d` | navigation data bit, ±1 at 50 bps | nuisance (folds into θ₀ ± π) |
| `w[n]` | complex AWGN, variance σ² per rail | — |
| `A` | signal amplitude | set by link budget |

Note `θ₀` is *not* the GNSS carrier-phase ranging observable. It is a nuisance
parameter: a constant rotation of the whole block, which we must survive but never
want to measure.

**Sampling.** `f_s = 2.046 MHz` (2 samples/chip), so `N_s = 2046` samples per 1 ms
code period. The front end quantizes to 2 bits per rail — standard in GNSS, because
the signal is far below the noise and fine amplitude resolution buys almost nothing.

**Link budget.** Signal strength is quoted as `C/N₀` in dB-Hz (carrier power to
noise power spectral density). Per-sample SNR is

$$\gamma \;=\; \frac{C/N_0}{f_s}, \qquad \frac{A}{\sigma} \;=\; \sqrt{2\gamma}$$

At `C/N₀ = 38 dB-Hz` this gives `γ = −25.1 dB`: **the noise is 324× more powerful
than the signal, per sample.** Nothing is visible in any single sample. All detection
comes from accumulating `N_s` of them — the 30 dB of processing gain in the code.

## 2. The search space

Three unknowns, of which two must be searched over a grid:

$$N_\text{cells} \;=\; \underbrace{1023}_{\text{code phases}} \times \underbrace{n_{f_d}}_{\text{Doppler}} \times \underbrace{32}_{\text{PRNs}} \;=\; 1.34\times10^6 \quad (n_{f_d}=41)$$

Code phases are counted chip-spaced, not sample-spaced: the correlation triangle is
±1 chip wide, so half-chip samples are correlated and 1023 is the independent count.

The Doppler grid spacing follows from coherent integration length `T_coh`. A residual
`f_e` over `T_coh` costs `sinc(f_e T_coh)`; holding that to ~1 dB gives

$$\delta f \;=\; \frac{1}{2 T_\text{coh}}, \qquad n_{f_d} \;=\; 2\left\lceil 2 f_\max T_\text{coh} \right\rceil + 1$$

**This is the single most important relation in the design.** It says the Doppler
dimension is set by the coherent integration length and nothing else — a lever we
exploit in §7.

## 3. Baseline: FFT parallel-code-phase search

The conventional engine computes, per (PRN, Doppler) pair, the circular correlation
of the Doppler-wiped signal against the local code, using the FFT to get all code
phases at once:

$$R[\tau] \;=\; \mathcal{F}^{-1}\Big\{ \mathcal{F}\big\{ r[n]\, e^{-j2\pi f_d n/f_s} \big\} \cdot \overline{\mathcal{F}\{c[n]\}} \Big\}, \qquad \text{detect on } |R[\tau]|^2$$

The `|·|²` is what makes it phase-invariant: it discards `θ₀`, so the baseline never
searches it.

**Cost.** Two length-`N_s` FFTs and one multiply per (PRN, Doppler):

$$C_\text{FFT} \;=\; N_\text{PRN}\, n_{f_d}\left(2 \cdot \tfrac{N_s}{2}\log_2 N_s + N_s\right) \;=\; 32.2\ \text{M complex MAC per 1 ms dwell}$$

**Detection.** Normalising so per-dwell noise is unit-variance per component, the
non-coherent sum over `M` dwells is

$$T \sim \chi^2_{2M} \ \ (H_0), \qquad T \sim {\chi'}^2_{2M}(M\lambda) \ \ (H_1), \qquad \lambda = 2\,(C/N_0)\,T_\text{coh}$$

Multipliers dominate the area and power. That is what we are trying to remove.

## 4. The identity that replaces the multiplier

Map bits to bipolar symbols, `x̃ = 1 − 2x`. Then for `x, y ∈ {0,1}^W`:

$$\langle \tilde{x}, \tilde{y}\rangle \;=\; \sum_{i=1}^{W} \tilde{x}_i \tilde{y}_i \;=\; W - 2\, d_H(x,y)$$

where `d_H` is Hamming distance. **Correlation and Hamming distance are the same
quantity up to an affine map.** A content-addressable memory that thresholds `d_H`
against every stored row in parallel is therefore a bank of 1-bit correlators — and
it computes them inside the array, as a side-effect of the read, with no multipliers.

An *approximate-match* CAM returns, in one cycle, the boolean vector

$$m_k \;=\; \mathbb{1}\big[\, d_H(\text{key}, \text{row}_k) \le \tau \,\big], \qquad k = 1 \dots N$$

Booleans only — never a distance, rank, or argmin. That constraint is enforced by
test (`tests/test_cam.py`), not convention.

### What the CAM lacks

Not the multiply — the **magnitude**. The baseline forms `Re` and `Im` and takes
`|·|²`. A CAM gives one thresholded projection onto one replica phase. So `θ₀`, which
the baseline discards for free, must be handled explicitly. We do it by sweeping four
quadrants and OR-ing the fires:

$$\bigvee_{k=0}^{3} \mathbb{1}\big[\operatorname{Re}\{z e^{-j\theta_k}\} > T\big] \;=\; \mathbb{1}\big[\max_k \operatorname{Re}\{z e^{-j\theta_k}\} > T\big] \;\approx\; \mathbb{1}\big[|z| > T\big]$$

i.e. **the magnitude operator implemented as rows plus an OR, rather than squarers.**

## 5. Detection statistics — the master formula

Sign-quantize both rails; the key is `W = 2N_s = 4092` bits. Rows store the
full-scale replica (§6), so a row bit is `sgn(c)` on each rail.

**Per bit.** With Gaussian noise, `P(sign error) = Q(A/σ)`, and since `A/σ ≪ 1` here,

$$p \;=\; Q\!\left(\tfrac{A}{\sigma}\right) \;\approx\; \tfrac{1}{2} - \varphi(0)\tfrac{A}{\sigma}, \qquad \varphi(0)=\tfrac{1}{\sqrt{2\pi}}=0.3989$$

**Per word.** Let `θ_e` be the residual phase error after the quadrant sweep
(`|θ_e| ≤ 45°`). With rows at 45°, both rails carry the code, and the I/Q deficits add
as `cos(45°+θ_e) + sin(45°+θ_e) = √2 cos θ_e`:

$$\mathbb{E}[d_H \mid H_0] = \tfrac{W}{2}, \qquad \mathbb{E}[d_H \mid H_1] = \tfrac{W}{2} - \Delta, \qquad \Delta = N_s\,\varphi(0)\,\tfrac{A}{\sigma}\sqrt{2}\cos\theta_e$$

$$\sigma_d = \tfrac{\sqrt{W}}{2}$$

Substituting `A/σ = √(2γ)`, `γ = (C/N₀)/f_s`, `N_s = f_s T`, and `2φ(0) = √(2/π)`:

$$\boxed{\ \frac{\Delta}{\sigma_d} \;=\; \sqrt{2\,(C/N_0)\,T}\;\cdot\;\sqrt{\tfrac{2}{\pi}}\;\cdot\;\cos\theta_e\ }$$

Three factors, each with a clean reading:

- `√(2·(C/N₀)·T)` — the ideal coherent detection SNR. Sampling rate does not appear:
  oversampling widens `W` but adds no information.
- `√(2/π) = 0.7979 = −1.96 dB` — **the hard-limiting loss falls out of `φ(0)`.** This
  is the textbook 1-bit correlator penalty, derived rather than quoted.
- `cos θ_e ∈ [1/√2, 1]` — the quadrant-sweep ripple, at most 3 dB, never zero.

| C/N₀ | γ | A/σ | p | Δ | Δ/σ_d |
|---|---|---|---|---|---|
| 45 dB-Hz | −18.1 dB | 0.176 | 0.430 | 203 | 6.35 |
| 42 dB-Hz | −21.1 dB | 0.124 | 0.450 | 144 | 4.49 |
| 38 dB-Hz | −25.1 dB | 0.079 | 0.469 | 91 | 2.83 |

Read the `p` column: **each bit comparison is right only 53.1% of the time at
38 dB-Hz.** The separation is 91 bits out of 4092 — about 2% of the word. Everything
in this design is a way of counting that 2% reliably.

**Consequence.** For a `1.34×10⁶`-cell search at 1% system false alarm, the per-cell
threshold sits at 5.35σ. With `Δ/σ_d = 2.83`, a single 1 ms dwell gives `P_d ≈ 0.003`.
**One dwell never closes at these signal levels — for any architecture.** Multi-dwell
non-coherent accumulation is mandatory, which turns out to suit a boolean readout.

## 6. Placement: what goes in rows, what gets swept

Each unknown lives either in **rows** (costs area) or in a **sweep** (costs cycles).
The rule that fell out:

> **Sweep the rotations that are free in the quantized domain; store the ones that are not.**

**θ₀ → swept, free.** A 90° rotation is `(I,Q) → (−Q, I)`. On sign bits this is
`(b_I, b_Q) → (¬b_Q, b_I)` — a wire swap and an inversion, *exact*, with no
re-quantization. (For the 3-bit-per-rail thermometer mapping, negation is a reverse
plus complement; also exact.) Verified bit-exact for all four quadrants and both
mappings in `tests/gps/test_quantize.py`.

**f_d → stored.** A Doppler hypothesis is a phase *ramp*. Applying it to the key needs
a real rotator plus re-quantization; applying it to a row is free, because rows are
precomputed offline at arbitrary precision.

**τ → swept**, one code phase per clock, over a 2 ms buffer (a genuine sliding window;
a circular 1 ms buffer would introduce a seam where wrapped samples are off by
`exp(−j2π f_d · 1ms)`, several full cycles at 5 kHz).

### The row-phase trap

Rows must be built at `θ = 45°`, not 0. At `θ_k = 0` the replica `c·e^{j0}` is purely
real, so the row's Q rail is a constant — half the match line contributes variance
with no signal, costing 3 dB. Measured margin came in at `0.66×` theory across every
C/N₀ (and `0.33×` for a 45° input); *constant* ratios are the signature of a
deterministic modelling error rather than noise. At 45° both rails carry `±c`, and
measured margin tracks the boxed formula to within 5%.

$$\text{Search} = \underbrace{\text{32 PRN} \times n_{f_d}\ \text{rows}}_{\text{parallel, 1 cycle}} \;\times\; \underbrace{N_s\ \text{code phases}}_{\text{1 per clock}} \;\times\; \underbrace{4\ \text{quadrants}}_{\text{free}}$$

giving `N_s × 4 = 8184` queries per 1 ms dwell → **8.2 MHz**, well inside an array's
capability, with all 32 PRNs resolved every cycle.

## 7. Segmentation: trading area for time

Split the match line into `S` sub-lines with per-segment threshold `τ`, and combine
by counting:

$$\text{fire}_s = \mathbb{1}\big[d_H^{(s)} \le \tau\big], \qquad \text{declare} \iff \sum_{s=1}^{MS}\text{fire}_s \ge k$$

The mechanism, stated precisely. Hamming distance is **additive**:

$$d_H(\text{word}) = \sum_{s=1}^{S} d_H^{(s)}$$

so *summing* segment distances would be exactly the unsegmented 1 ms detector, and
would still need all 41 Doppler rows. The Doppler tolerance appears **only** because
each segment is thresholded before combining. Hard decisions break coherence between
segments, `T_coh → 1ms/S`, and by §2 the row count collapses. That is also where the
sensitivity goes — the two are the same act.

**Statistics.** With segments independent, the fire count over `M` dwells is Binomial.
This was validated before use (variance/binomial-variance = 0.97–1.02 over 6000
trials) — it is what lets the `10⁻⁸` tail probabilities be reached at all:

$$P_d = P\big[\mathrm{Bin}(MS, p_1) \ge k\big], \qquad P_{fa}^\text{sys} = N_\text{cells}\cdot P\big[\mathrm{Bin}(MS, p_0) \ge k\big]$$

**Sizing.** Rows are full scale, so stored state is 1 bit/rail regardless of mapping —
a thermometer key widens the match line but not the memory:

$$\text{bits} = N_\text{PRN}\, n_{f_d} \cdot 2N_s, \qquad \text{cells} = N_\text{PRN}\, n_{f_d} \cdot W$$

| S | T_coh | δf | n_fd | dictionary | M @ 38 dB-Hz | M @ 42 dB-Hz |
|---|---|---|---|---|---|---|
| 1 | 1000 µs | 500 Hz | 41 | 5.37 Mbit | 17 ms | 6 ms |
| 2 | 500 µs | 1000 Hz | 21 | 2.75 Mbit | 23 ms | 7 ms |
| 3 | 333 µs | 1500 Hz | 15 | 1.96 Mbit | 27 ms | 7 ms |
| 6 | 167 µs | 3000 Hz | 9 | 1.18 Mbit | 42 ms | 10 ms |
| 11 | 91 µs | 5500 Hz | 5 | 0.65 Mbit | 65 ms | 13 ms |

**8.3× less area for 3.8× more time** at 38 dB-Hz (2.2× at 42). A pre-simulation hand
estimate put the hard-decision cost at ~2 dB; measured it reaches 5.8 dB at S=11, so
the estimate was optimistic.

**Constraint worth knowing:** a segment must be a whole number of samples, so `S`
divides `N_s = 2046 = 2·3·11·31`. Available `S ∈ {2,3,6,11,22,31,33}` — **powers of
two are unavailable.**

## 8. Results, with the loss attributed

Dwells for `P_d = 0.9` at system `P_fa = 10⁻²` over all `1.34×10⁶` cells:

| C/N₀ | B1: full-precision FFT | B1′: 1-bit FFT | CAM (S=1) | CAM (S=11) |
|---|---|---|---|---|
| 38 dB-Hz | 6 ms | 11 ms | 17 ms (**+1.9 dB**) | 65 ms (+7.7 dB) |
| 42 dB-Hz | 2 ms | 4 ms | 6 ms (**+1.8 dB**) | 13 ms (+5.1 dB) |

The decomposition is the point:

- **B1 → B1′** is `√(2/π)`, the −1.96 dB limiting loss. Every hard-limited GNSS front
  end already pays it. Not chargeable to the CAM.
- **B1′ → CAM at S=1 is +1.9 dB.** That is the true boolean-readout tax.
- Everything beyond is the segmentation trade of §7, bought deliberately for area.

For contrast, `docs/findings.md` records the same primitive applied to BLE coming out
~10× worse than a correlator bank. The difference is what the application needs from
the readout: BLE wanted a precise frequency *estimate*, which a thresholded boolean
cannot carry; GNSS acquisition wants a detection plus a bin index, then hands off to
tracking. The binary readout is nearly free when the required output is already
coarse.

**Required match-line resolution.** Since the H₀/H₁ separation is ~2% of `W`, the
sense amplifier must resolve a small fraction of the count. Measured as the τ band
holding acquisition within 10% of its best dwell count:

| S | segment width | @ 38 dB-Hz | @ 42 dB-Hz |
|---|---|---|---|
| 1 | 4092 bits | 1 part in 128 | 1 part in 105 |
| 11 | 372 bits | 1 part in 53 | 1 part in 41 |

This is the sharpest contrast with the BLE study, where a match sits at ~23% of `W`,
comfortably below chance. Here the correct row sits *barely* below chance, because the
processing gain is spread thinly across the whole word. Segmentation relaxes it —
**the same dial buys fewer rows and a coarser sense amp**, with time paying for both.

## 9. How each claim was checked

| Claim | Gate |
|---|---|
| Gold codes correct | All 32 PRNs vs IS-GPS-200 first-10-chip octal table; autocorrelation ∈ {−65,−1,63} |
| `⟨x̃,ỹ⟩ = W − 2d_H` | Direct, `tests/gps/test_experiments.py` |
| 90° rotation exact | Rotated-then-quantized == quantized-then-permuted, bit for bit, both mappings, all 4 quadrants |
| Boxed Δ/σ formula | Monte Carlo within 5% at matched phase; the `cos θ_e` law within 15% |
| Binomial segment model | Variance ratio 0.97–1.02, 6000 trials, all wrong-cell types |
| FFT baseline valid | Acquires planted signals to ≤1 sample and 0 Hz; measured non-centrality 64.0 vs 63.3 predicted |
| Wrong PRN ≈ noise | `p_seg` 0.108–0.119 vs 0.109–0.119 for wrong code phase — Gold cross-correlation bounded at 65/1023 |

## 10. Limitations

- Simulation only. The "1 part in 100" is a *requirement* on a future sense amplifier,
  not a measurement of one.
- No multipath, interference, or AGC dynamics.
- Sweeps use the sign mapping (effectively 1-bit). The thermometer mapping
  (`d_H` = L1 distance in ADC levels, worth ~1.4 dB) is implemented and gated but not
  measured end to end.
- 3000 trials/point. Tail probabilities come from the *validated* binomial model, not
  direct measurement; `p₀` carries ~2% relative error, propagating into `k`.
- 8-hypothesis θ₀ grid (45° spacing, 0.7 dB ripple via a 16-entry LUT) and the
  complement trick — `d_H(key, ¬row) = W − d_H(key, row)` halves the phase dimension
  if the array can threshold the upper tail — are both unexplored.
