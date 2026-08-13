# B3 (1-bit sign-sign correlator): a fair low-power peer for the CAM

tau=151, threshold_full=0.7119, threshold_sign=0.9356, W=624, N=32, N_s=160

| System | Latency (cycles) | Area proxy (op-count) | Pd@8dB | Pd@12dB | Pd@16dB | Pd@20dB |
|---|---|---|---|---|---|---|
| CAM | 1 | 19968 | 0.550 | 1.000 | 1.000 | 1.000 |
| B1/B2 (full-precision) | 1 | 5120 | 1.000 | 1.000 | 1.000 | 1.000 |
| B3 (1-bit sign-sign) | 1 | 5120 | 1.000 | 1.000 | 1.000 | 1.000 |

Area proxy note: B3's op-count matches B2's (N x N_s), but each op is a 1-bit XNOR/sign-flip, not a complex multiply-accumulate -- the two proxies are NOT directly comparable in silicon area despite the same count. B3 is the fairer *precision* peer to the CAM's binary readout; B1/B2 remain in the comparison as the full-precision upper bound.


## CAM

| Eb/N0 | Pd | RMS (kHz) | Pfa |
|---|---|---|---|
| 0 | 0.000 | nan | 0.0000 |
| 2 | 0.000 | nan | 0.0000 |
| 4 | 0.000 | nan | 0.0000 |
| 6 | 0.016 | 23.44 | 0.0000 |
| 8 | 0.550 | 26.53 | 0.0000 |
| 10 | 0.990 | 30.13 | 0.0000 |
| 12 | 1.000 | 30.32 | 0.0000 |
| 14 | 1.000 | 29.28 | 0.0067 |
| 16 | 1.000 | 30.15 | 0.0000 |
| 18 | 1.000 | 29.75 | 0.0067 |
| 20 | 1.000 | 30.11 | 0.0300 |

## B1/B2 (full-precision)

| Eb/N0 | Pd | RMS (kHz) | Pfa |
|---|---|---|---|
| 0 | 0.984 | 3.18 | 0.0067 |
| 2 | 0.998 | 3.08 | 0.0000 |
| 4 | 1.000 | 2.96 | 0.0000 |
| 6 | 1.000 | 2.86 | 0.0000 |
| 8 | 1.000 | 2.92 | 0.0000 |
| 10 | 1.000 | 2.83 | 0.0000 |
| 12 | 1.000 | 2.77 | 0.0000 |
| 14 | 1.000 | 2.84 | 0.0000 |
| 16 | 1.000 | 2.88 | 0.0000 |
| 18 | 1.000 | 2.79 | 0.0033 |
| 20 | 1.000 | 2.84 | 0.0000 |

## B3 (1-bit sign-sign)

| Eb/N0 | Pd | RMS (kHz) | Pfa |
|---|---|---|---|
| 0 | 0.004 | 1.33 | 0.0000 |
| 2 | 0.077 | 3.34 | 0.0000 |
| 4 | 0.608 | 3.28 | 0.0000 |
| 6 | 0.978 | 3.16 | 0.0000 |
| 8 | 1.000 | 3.15 | 0.0000 |
| 10 | 1.000 | 3.04 | 0.0000 |
| 12 | 1.000 | 3.01 | 0.0000 |
| 14 | 1.000 | 3.06 | 0.0033 |
| 16 | 1.000 | 3.18 | 0.0000 |
| 18 | 1.000 | 3.10 | 0.0067 |
| 20 | 1.000 | 3.26 | 0.0100 |
