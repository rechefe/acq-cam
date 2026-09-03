"""How finely must the match line resolve?

This is the architecture's central hardware question. In the BLE study a match
sits at a Hamming distance of ~23% of W -- comfortably below chance. GNSS is the
opposite regime: the correct row sits just *barely* below chance, because all the
processing gain is spread thinly across the whole word. So the sense amplifier
has to resolve a small fraction of the match line, and the question is how small.

Required resolution is defined operationally: the width of the tau band over
which acquisition stays within 10% of its best dwell count, expressed as a
fraction of the segment width. A circuit designer reads it as "the match line
must resolve one part in N".

Segmenting should relax this, since each sub-line is shorter -- that is the
sense-margin argument for segmentation, independent of the Doppler-row argument.

    python -m sim.gps.sense_margin_study [--trials N] [--seed N] [--force]
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from .experiments import (GpsConfig, cached_distances, dwells_required,
                          independent_cells)
from .segmentation_study import CN0S, FIGDIR, SEGMENTS


def run(trials: int = 3000, seed: int = 42, force: bool = False) -> dict:
    n_s, n_c = len(SEGMENTS), len(CN0S)
    best = np.zeros((n_s, n_c), dtype=int)
    lo = np.zeros((n_s, n_c), dtype=int)
    hi = np.zeros((n_s, n_c), dtype=int)
    seg_w = np.zeros(n_s, dtype=int)

    for i, S in enumerate(SEGMENTS):
        cfg = GpsConfig(n_segments=S, n_prn=3)
        cells = independent_cells(GpsConfig(n_segments=S, n_prn=32))
        seg_w[i] = cfg.n_samples // S * 2  # sign mapping: 2 bits per sample
        for j, cn0 in enumerate(CN0S):
            d = cached_distances(cfg, cn0, trials, seed + 1000 * i + j, force)
            wrong = np.concatenate([d[k].ravel() for k in
                                    ("wrong_phase", "wrong_doppler", "wrong_prn")])
            aligned = d["aligned"].ravel()
            taus = np.arange(int(np.quantile(wrong, 0.002)), int(np.quantile(wrong, 0.5)))
            dwells = []
            for tau in taus:
                p1, p0 = (aligned <= tau).mean(), (wrong <= tau).mean()
                m, _ = dwells_required(p1, p0, int(S), cells) if 0 < p0 < p1 else (-1, -1)
                dwells.append(m if m > 0 else 10 ** 9)
            dwells = np.array(dwells)
            m_best = dwells.min()
            ok = np.flatnonzero(dwells <= np.ceil(1.1 * m_best))
            best[i, j] = m_best
            lo[i, j], hi[i, j] = taus[ok[0]], taus[ok[-1]]

    frac = (hi - lo + 1) / seg_w[:, None]
    return {"segments": np.array(SEGMENTS), "cn0s": np.array(CN0S), "seg_width": seg_w,
            "tau_lo": lo, "tau_hi": hi, "best_dwells": best, "tolerance_frac": frac}


def table(res: dict) -> str:
    head = ("| S | segment width | " +
            " | ".join(f"tau band @ {c:.0f} dB-Hz | resolution @ {c:.0f} dB-Hz"
                       for c in res["cn0s"]) + " |")
    lines = [head, "|---|---|" + "---|" * (2 * len(res["cn0s"]))]
    for i, S in enumerate(res["segments"]):
        cells = []
        for j in range(len(res["cn0s"])):
            f = res["tolerance_frac"][i, j]
            cells.append(f"{res['tau_lo'][i, j]}-{res['tau_hi'][i, j]} bits")
            cells.append(f"1 part in {1 / f:.0f}" if f > 0 else "n/a")
        lines.append(f"| {S} | {res['seg_width'][i]} bits | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    res = run(a.trials, a.seed, a.force)
    os.makedirs(FIGDIR, exist_ok=True)
    md = ("# GPS CAM required match-line resolution\n\n"
          "Width of the tau band keeping acquisition within 10% of its best dwell "
          f"count. {a.trials} trials/point, seed {a.seed}.\n\n" + table(res) + "\n")
    with open(os.path.join(FIGDIR, "gps_sense_margin.md"), "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
