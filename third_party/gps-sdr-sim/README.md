# gps-sdr-sim (vendored)

GPS L1 C/A signal generator by Takuji Ebinuma, used as the signal source for
`notebooks/09_benchmark.ipynb`. It builds the constellation from real broadcast
ephemeris, so orbital Doppler, code Doppler, path loss, navigation data bits and
ionospheric delay are modelled by construction rather than by our own
approximations -- which is why it replaced the hand-written signal model.

| file | provenance |
|---|---|
| `gpssim.c`, `gpssim.h` | https://github.com/osqzss/gps-sdr-sim, branch `master`, fetched 2026-09-03 |
| `LICENSE` | MIT, Copyright (c) 2015-2025 Takuji Ebinuma, verbatim |
| `brdc0010.22n` | sample RINEX broadcast ephemeris shipped with the project (2022, day 001) |
| `truth-dump.patch` | **ours** -- see below |

Vendored rather than fetched at run time so the notebook reproduces offline.

## The patch

`gps-sdr-sim` does not expose per-satellite code phase or Doppler; `-v` reports
only azimuth, elevation, range and ionospheric delay. `truth-dump.patch` adds
`fprintf` calls that write the internal `channel_t` state -- PRN, carrier
Doppler, code frequency, code phase, gain -- once per 0.1 s block.

It is **observation only**:

- nothing in the signal path is touched;
- output is gated on the `GPSSIM_TRUTH` environment variable, so with the
  variable unset no file is opened and no branch is taken.

Verified: the patched binary produces **byte-identical** I/Q to the unpatched one
for the same arguments, both with and without `GPSSIM_TRUTH` set. The notebook
re-runs that check before using any of the data.

## Build

    cp gpssim.c gpssim_patched.c
    patch -p0 gpssim_patched.c < truth-dump.patch
    gcc gpssim_patched.c -lm -O3 -I. -o gps-sdr-sim

Generated artefacts (`gpssim_patched.c`, the binary, `*.bin`, `*.csv`) are not
committed; the notebook rebuilds them.
