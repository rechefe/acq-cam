"""Segmentation study: what shortening the coherent interval buys and costs.

Splitting the match line into S segments shrinks the Doppler dimension of the
dictionary -- n_doppler = 2*ceil(2*f_max*T_coh)+1 with T_coh = 1 ms / S -- but
each segment is thresholded before combining, which is a hard decision and costs
sensitivity. This measures both sides of that trade at the chosen operating
point (+-10 kHz cold start, 38-42 dB-Hz) and reports the resulting acquisition
time against dictionary size.

    python -m sim.gps.segmentation_study [--trials N] [--seed N] [--force]
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from . import codebook
from matplotlib.ticker import NullFormatter

from ..plots import new_figure, save_figure
from .baselines import dwells_required_fft
from .experiments import (GpsConfig, cached_distances, dictionary_bits,
                          dwells_required, independent_cells)

# Segment counts must divide 2046 samples; powers of two mostly do not.
SEGMENTS = [1, 2, 3, 6, 11]
CN0S = [38.0, 42.0]
FIGDIR = os.path.join(os.path.dirname(__file__), "..", "..", "figures")


def run(trials: int = 3000, seed: int = 42, force: bool = False) -> dict:
    """Per-S best achievable dwell count, sweeping tau for each (S, C/N0)."""
    n_s, n_c = len(SEGMENTS), len(CN0S)
    best_m = np.zeros((n_s, n_c), dtype=int)
    best_k = np.zeros((n_s, n_c), dtype=int)
    best_tau = np.zeros((n_s, n_c), dtype=int)
    p1s = np.zeros((n_s, n_c))
    p0s = np.zeros((n_s, n_c))

    for i, S in enumerate(SEGMENTS):
        cfg = GpsConfig(n_segments=S, n_prn=3)
        cells = independent_cells(GpsConfig(n_segments=S, n_prn=32))
        for j, cn0 in enumerate(CN0S):
            d = cached_distances(cfg, cn0, trials, seed + 1000 * i + j, force)
            # the closest of the three wrong-cell kinds drives the false alarms
            wrong = np.concatenate([d[k].ravel() for k in
                                    ("wrong_phase", "wrong_doppler", "wrong_prn")])
            aligned = d["aligned"].ravel()
            best = (10 ** 9, -1, -1, 0.0, 0.0)
            for tau in np.unique(np.quantile(wrong, np.linspace(0.01, 0.45, 40)).astype(int)):
                p1, p0 = (aligned <= tau).mean(), (wrong <= tau).mean()
                if not 0 < p0 < p1:
                    continue
                m, k = dwells_required(p1, p0, S, cells)
                if 0 < m < best[0]:
                    best = (m, k, int(tau), p1, p0)
            best_m[i, j], best_k[i, j], best_tau[i, j], p1s[i, j], p0s[i, j] = best

    return {"segments": np.array(SEGMENTS), "cn0s": np.array(CN0S),
            "dwells": best_m, "k": best_k, "tau": best_tau, "p1": p1s, "p0": p0s,
            "dict_bits": np.array([dictionary_bits(GpsConfig(n_segments=S, n_prn=32))
                                   for S in SEGMENTS]),
            "n_doppler": np.array([len(codebook.doppler_grid(
                10e3, GpsConfig(n_segments=S).t_coh)) for S in SEGMENTS]),
            "cells": np.array([independent_cells(GpsConfig(n_segments=S, n_prn=32))
                               for S in SEGMENTS])}


def make_figure(res: dict, name: str = "gps_area_time_pareto"):
    """Dictionary size against acquisition time: the segmentation trade, with the
    conventional full-precision search as the reference point."""
    fig, ax = new_figure(height_in=2.5)
    markers = ["o", "s"]
    b1s = []
    for j, cn0 in enumerate(res["cn0s"]):
        m = res["dwells"][:, j].astype(float)
        ok = m > 0
        ax.plot(m[ok], res["dict_bits"][ok] / 1e6, markers[j] + "-",
                markersize=3.5, label=f"CAM, {cn0:.0f} dB-Hz")
        for i, S in enumerate(res["segments"]):
            if ok[i]:
                ax.annotate(f"S={S}", (m[i], res["dict_bits"][i] / 1e6),
                            textcoords="offset points", xytext=(3, 3), fontsize=5)
        b1, _ = dwells_required_fft(float(cn0), 1e-3,
                                    int(res["cells"][0]))
        b1s.append((b1, float(cn0)))
    for b1, cn0 in b1s:
        ax.axvline(b1, color="0.55", linestyle=":", linewidth=0.8)
        ax.annotate(f"FFT {cn0:.0f} dB-Hz", (b1, 0.62), rotation=90, fontsize=5,
                    color="0.35", ha="right", va="bottom",
                    textcoords="offset points", xytext=(-2, 0))

    ax.set_xscale("log")
    ax.set_xlim(1.6, 90)
    ax.set_ylim(0, 6.2)
    ax.set_xticks([2, 5, 10, 20, 50])
    ax.set_xticklabels(["2", "5", "10", "20", "50"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("acquisition time (ms) for $P_d$ = 0.9")
    ax.set_ylabel("dictionary (Mbit)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25, linewidth=0.4)
    return save_figure(fig, name)


def table(res: dict) -> str:
    lines = ["| S | T_coh | Doppler rows/PRN | dictionary | "
             + " | ".join(f"dwells @ {c:.0f} dB-Hz" for c in res["cn0s"]) + " |",
             "|---|---|---|---|" + "---|" * len(res["cn0s"])]
    for i, S in enumerate(res["segments"]):
        t = GpsConfig(n_segments=int(S)).t_coh * 1e6
        cells = [f"{res['dwells'][i, j]} ms" if res["dwells"][i, j] > 0 else "unreachable"
                 for j in range(len(res["cn0s"]))]
        lines.append(f"| {S} | {t:.0f} us | {res['n_doppler'][i]} | "
                     f"{res['dict_bits'][i] / 1e6:.2f} Mbit | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    res = run(a.trials, a.seed, a.force)
    os.makedirs(FIGDIR, exist_ok=True)
    md = ("# GPS CAM segmentation trade\n\n"
          f"+-10 kHz cold start, {a.trials} trials/point, seed {a.seed}. "
          "Dwells is the 1 ms dwell count for Pd=0.9 at system Pfa=1e-2 over the "
          "full 2-D search.\n\n" + table(res) + "\n")
    with open(os.path.join(FIGDIR, "gps_segmentation.md"), "w") as f:
        f.write(md)
    make_figure(res)
    print(md)


if __name__ == "__main__":
    main()
