# Encoder sensitivity study

N=128 requested hypotheses, +-150 kHz, n_sym=40, osr=4.

| D (symbols) | B | W | unique rows (/N) | min adj. Hamming | total swing (bits) | monotone | alias onset (kHz) | p @10dB | encoder SNR |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 624 | 21/128 | 2 | 408 | yes | -- | 0.2148 | 39.77 |
| 1 | 3 | 1248 | 35/128 | 2 | 870 | yes | -- | 0.2008 | 61.48 |
| 1 | 4 | 2496 | 55/128 | 2 | 1532 | yes | -- | 0.1955 | 77.32 |
| 2 | 2 | 608 | 38/128 | 2 | 386 | **NO (aliased)** | 102.8 | 0.2284 | 37.29 |
| 2 | 3 | 1216 | 59/128 | 2 | 912 | **NO (aliased)** | 102.8 | 0.2044 | 64.86 |
| 2 | 4 | 2432 | 110/128 | 2 | 2058 | **NO (aliased)** | 102.8 | 0.1968 | 104.96 |
| 3 | 2 | 592 | 52/128 | 2 | 204 | **NO (aliased)** | 34.3 | 0.2270 | 20.02 |
| 3 | 3 | 1184 | 73/128 | 2 | 296 | **NO (aliased)** | 27.2 | 0.2035 | 21.37 |
| 3 | 4 | 2368 | 125/128 | 2 | 380 | **NO (aliased)** | 17.7 | 0.1966 | 19.65 |
| 4 | 2 | 576 | 53/128 | 2 | 228 | **NO (aliased)** | -22.4 | 0.2274 | 22.66 |
| 4 | 3 | 1152 | 70/128 | 4 | 332 | **NO (aliased)** | -22.4 | 0.2014 | 24.39 |
| 4 | 4 | 2304 | 118/128 | 4 | 872 | **NO (aliased)** | -22.4 | 0.1954 | 45.81 |

## Ranked by encoder SNR (total swing / binomial noise std, Eb/N0=10dB)

**Caveat:** for aliased (non-monotone) configs, total swing is endpoint-to-endpoint Hamming distance (row_first vs. row_last), which is not a meaningful resolving-power metric once the codebook has folded back on itself -- a large value there can mean the two endpoints happened to land on opposite sides of a fold, not that CFO is well resolved across the range. The unrestricted ranking is included for completeness; the monotone-only ranking below is the one that supports an actual recommendation.

### All configurations (includes aliased -- do not use to pick D, B)

| Rank | D | B | Encoder SNR | Monotone |
|---|---|---|---|---|
| 1 | 2 | 4 | 104.96 | NO |
| 2 | 1 | 4 | 77.32 | yes |
| 3 | 2 | 3 | 64.86 | NO |
| 4 | 1 | 3 | 61.48 | yes |
| 5 | 4 | 4 | 45.81 | NO |
| 6 | 1 | 2 | 39.77 | yes |
| 7 | 2 | 2 | 37.29 | NO |
| 8 | 4 | 3 | 24.39 | NO |
| 9 | 4 | 2 | 22.66 | NO |
| 10 | 3 | 3 | 21.37 | NO |
| 11 | 3 | 2 | 20.02 | NO |
| 12 | 3 | 4 | 19.65 | NO |

### Monotone-only (safe to use for picking D, B)

| Rank | D | B | Encoder SNR | Monotone |
|---|---|---|---|---|
| 1 | 1 | 4 | 77.32 | yes |
| 2 | 1 | 3 | 61.48 | yes |
| 3 | 1 | 2 | 39.77 | yes |


## Aliasing flags

A naive bulk-phase estimate (bias=2*pi*df*D*T_symbol exceeding pi) predicts only D=4 aliases within +-150 kHz. That estimate is wrong: aliasing is set by individual bits wrapping past their own nearest decision axis (spacing 2*pi/2**B), a much finer threshold than pi, crossed much earlier. Empirically **every D in {2,3,4} aliases for every B tested** -- only D=1 stays monotone across the full BLE range.

- D=2, B=2: row ordering breaks monotonicity at ~102.8 kHz.
- D=2, B=3: row ordering breaks monotonicity at ~102.8 kHz.
- D=2, B=4: row ordering breaks monotonicity at ~102.8 kHz.
- D=3, B=2: row ordering breaks monotonicity at ~34.3 kHz.
- D=3, B=3: row ordering breaks monotonicity at ~27.2 kHz.
- D=3, B=4: row ordering breaks monotonicity at ~17.7 kHz.
- D=4, B=2: row ordering breaks monotonicity at ~-22.4 kHz.
- D=4, B=3: row ordering breaks monotonicity at ~-22.4 kHz.
- D=4, B=4: row ordering breaks monotonicity at ~-22.4 kHz.
