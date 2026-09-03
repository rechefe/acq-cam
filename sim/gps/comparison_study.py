"""CAM against the conventional acquisition engine, with the loss decomposed.

Three reference points, so the CAM's cost is attributed rather than lumped:

  B1  full-precision FFT parallel-code-phase search -- the real competitor.
  B1' the same search driven from 1-bit samples. The gap B1 -> B1' is the
      textbook 1.96 dB limiting loss, which any hard-limited GNSS front end
      already pays and which is therefore not chargeable to the CAM.
  CAM the segmented approximate-match array. The gap B1' -> CAM is what the
      boolean readout and the hard per-segment decisions actually cost.

Reported as acquisition time (1 ms dwells) and as hardware: complex MACs per
full 2-D search for the FFT, stored bits and match-line cells for the CAM.

    python -m sim.gps.comparison_study [--trials N] [--seed N] [--force]
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from .baselines import dwells_required_fft, fft_complex_macs
from .experiments import (GpsConfig, cached_distances, dictionary_bits,
                          dwells_required, independent_cells, match_line_cells)
from .segmentation_study import CN0S, FIGDIR, SEGMENTS

# 1-bit limiting loss as a power factor: (2/pi) = -1.96 dB.
ONE_BIT_POWER_FACTOR = 2.0 / np.pi


def cam_dwells(S: int, cn0: float, trials: int, seed: int, force: bool) -> int:
    cfg = GpsConfig(n_segments=S, n_prn=3)
    cells = independent_cells(GpsConfig(n_segments=S, n_prn=32))
    d = cached_distances(cfg, cn0, trials, seed, force)
    wrong = np.concatenate([d[k].ravel() for k in
                            ("wrong_phase", "wrong_doppler", "wrong_prn")])
    aligned = d["aligned"].ravel()
    best = 10 ** 9
    for tau in np.unique(np.quantile(wrong, np.linspace(0.01, 0.45, 40)).astype(int)):
        p1, p0 = (aligned <= tau).mean(), (wrong <= tau).mean()
        if not 0 < p0 < p1:
            continue
        m, _ = dwells_required(p1, p0, S, cells)
        if m > 0:
            best = min(best, m)
    return best if best < 10 ** 9 else -1


def run(trials: int = 3000, seed: int = 42, force: bool = False) -> dict:
    rows = []
    for j, cn0 in enumerate(CN0S):
        cells_full = independent_cells(GpsConfig(n_segments=1, n_prn=32))
        m_b1, _ = dwells_required_fft(cn0, 1e-3, cells_full)
        cn0_1bit = 10 * np.log10(10 ** (cn0 / 10) * ONE_BIT_POWER_FACTOR)
        m_b1p, _ = dwells_required_fft(cn0_1bit, 1e-3, cells_full)
        for i, S in enumerate(SEGMENTS):
            rows.append((cn0, S, m_b1, m_b1p,
                         cam_dwells(int(S), cn0, trials, seed + 1000 * i + j, force)))
    return {"rows": np.array(rows, dtype=float),
            "fft_macs": fft_complex_macs(2046, 41, 32),
            "dict_bits": np.array([dictionary_bits(GpsConfig(n_segments=int(S), n_prn=32))
                                   for S in SEGMENTS]),
            "ml_cells": np.array([match_line_cells(GpsConfig(n_segments=int(S), n_prn=32))
                                  for S in SEGMENTS])}


def table(res: dict) -> str:
    lines = ["| C/N0 | S | B1 full-precision | B1' 1-bit | CAM | CAM vs B1' | dictionary |",
             "|---|---|---|---|---|---|---|"]
    for k, (cn0, S, b1, b1p, cam) in enumerate(res["rows"]):
        i = list(SEGMENTS).index(int(S))
        pen = f"{10 * np.log10(cam / b1p):+.1f} dB" if cam > 0 and b1p > 0 else "n/a"
        lines.append(f"| {cn0:.0f} dB-Hz | {int(S)} | {int(b1)} ms | {int(b1p)} ms | "
                     f"{int(cam)} ms | {pen} | {res['dict_bits'][i] / 1e6:.2f} Mbit |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    res = run(a.trials, a.seed, a.force)
    md = ("# GPS CAM vs conventional acquisition\n\n"
          "Dwells (1 ms) for Pd=0.9 at system Pfa=1e-2 over the full 2-D search, "
          f"+-10 kHz cold start. {a.trials} trials/point, seed {a.seed}.\n\n"
          "B1 -> B1' is the 1.96 dB limiting loss any hard-limited front end pays; "
          "B1' -> CAM is what the boolean readout costs.\n\n" + table(res) +
          f"\n\nOne full 2-D search costs the FFT engine "
          f"{res['fft_macs'] / 1e6:.1f} M complex MAC per dwell.\n")
    os.makedirs(FIGDIR, exist_ok=True)
    with open(os.path.join(FIGDIR, "gps_comparison.md"), "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
