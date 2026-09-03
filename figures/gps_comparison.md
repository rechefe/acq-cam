# GPS CAM vs conventional acquisition

Dwells (1 ms) for Pd=0.9 at system Pfa=1e-2 over the full 2-D search, +-10 kHz cold start. 3000 trials/point, seed 42.

B1 -> B1' is the 1.96 dB limiting loss any hard-limited front end pays; B1' -> CAM is what the boolean readout costs.

| C/N0 | S | B1 full-precision | B1' 1-bit | CAM | CAM vs B1' | dictionary |
|---|---|---|---|---|---|---|
| 38 dB-Hz | 1 | 6 ms | 11 ms | 17 ms | +1.9 dB | 5.37 Mbit |
| 38 dB-Hz | 2 | 6 ms | 11 ms | 23 ms | +3.2 dB | 2.75 Mbit |
| 38 dB-Hz | 3 | 6 ms | 11 ms | 27 ms | +3.9 dB | 1.96 Mbit |
| 38 dB-Hz | 6 | 6 ms | 11 ms | 42 ms | +5.8 dB | 1.18 Mbit |
| 38 dB-Hz | 11 | 6 ms | 11 ms | 65 ms | +7.7 dB | 0.65 Mbit |
| 42 dB-Hz | 1 | 2 ms | 4 ms | 6 ms | +1.8 dB | 5.37 Mbit |
| 42 dB-Hz | 2 | 2 ms | 4 ms | 7 ms | +2.4 dB | 2.75 Mbit |
| 42 dB-Hz | 3 | 2 ms | 4 ms | 7 ms | +2.4 dB | 1.96 Mbit |
| 42 dB-Hz | 6 | 2 ms | 4 ms | 10 ms | +4.0 dB | 1.18 Mbit |
| 42 dB-Hz | 11 | 2 ms | 4 ms | 13 ms | +5.1 dB | 0.65 Mbit |

One full 2-D search costs the FFT engine 32.2 M complex MAC per dwell.
