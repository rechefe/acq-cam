"""Generate the paper figures. Usage: python -m sim.figures <1|2|3|4|5|table|all>
[--trials N] [--seed N] [--force]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from . import experiments as exp
from . import plots
from . import decode as dec
from .cam import CAM


def fig1(cfg: exp.Config, n_trials: int, seed: int, force: bool):
    ebn0_list = [0, 5, 10, 15, 20]
    offsets = np.linspace(-150e3, 150e3, 31)
    data = exp.distance_profile(cfg, ebn0_list, offsets, n_trials, seed, force=force)
    fig, ax = plots.new_figure()
    for i, ebn0 in enumerate(ebn0_list):
        ax.plot(data["offsets"] / 1e3, data["mean_dist"][i], marker="o", markersize=1.5,
                label=f"{ebn0} dB")
    ax.set_xlabel("CFO offset from true value (kHz)")
    ax.set_ylabel("Mean Hamming distance (bits)")
    ax.set_title(f"Fig. 1: Distance profile (W={int(data['W'])})")
    ax.legend(title="Eb/N0", ncol=2)
    return plots.save_figure(fig, "fig1_distance_profile")


def fig2(cfg: exp.Config, n_trials: int, seed: int, force: bool):
    N = 32
    cb = exp.make_codebook(cfg, N=N)
    # target_pd=0.6 on the single nearest-row detector: illustrates the fired-
    # set mechanism at a tau tight enough to resolve structure (the system-
    # level any(m) detector used elsewhere is looser -- see Fig 4/Table I).
    tau = exp.calibrate_tau_for_detection(cfg, cb, ebn0=10.0, target_pd=0.6,
                                           n_trials=400, seed=seed)
    true_cfos = np.linspace(-150e3, 150e3, 25)
    data = exp.fired_set_heatmap(cfg, true_cfos, N, tau, ebn0=10.0, n_trials=n_trials,
                                  seed=seed, force=force)
    fig, ax = plots.new_figure()
    extent = [true_cfos[0] / 1e3, true_cfos[-1] / 1e3, data["df_grid"][0] / 1e3,
              data["df_grid"][-1] / 1e3]
    im = ax.imshow(data["fire_prob"], aspect="auto", origin="lower", extent=extent,
                    cmap="viridis", vmin=0, vmax=1)
    ax.plot([true_cfos[0] / 1e3, true_cfos[-1] / 1e3],
            [true_cfos[0] / 1e3, true_cfos[-1] / 1e3], "r--", linewidth=0.6,
            label="ideal (row=true)")
    ax.set_xlabel("True CFO (kHz)")
    ax.set_ylabel("Hypothesis row CFO (kHz)")
    ax.set_title(f"Fig. 2: Fired-set structure ($\\tau$={int(data['tau'])}, "
                 f"$\\delta$={float(data['delta']) / 1e3:.1f} kHz)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("P(fire)")
    ax.legend(loc="upper left")
    return plots.save_figure(fig, "fig2_fired_set")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("which", nargs="+")
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = exp.Config()
    which = set(args.which)
    if "all" in which:
        which = {"1", "2", "3", "4", "5", "table"}

    if "1" in which:
        path = fig1(cfg, args.trials, args.seed, args.force)
        print("wrote", path)
    if "2" in which:
        path = fig2(cfg, args.trials, args.seed, args.force)
        print("wrote", path)
    if "3" in which:
        from . import figures_more
        print("wrote", figures_more.fig3(cfg, args.trials, args.seed, args.force))
    if "4" in which:
        from . import figures_more
        print("wrote", figures_more.fig4(cfg, args.trials, args.seed, args.force))
    if "5" in which:
        from . import figures_more
        print("wrote", figures_more.fig5(cfg, args.trials, args.seed, args.force))
    if "table" in which:
        from . import figures_more
        print("wrote", figures_more.table1(cfg, args.trials, args.seed, args.force))


if __name__ == "__main__":
    main()
