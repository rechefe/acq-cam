"""Full performance report (detection/miss, false alarm, CFO accuracy) at a
given (B, OSR) operating point -- e.g. the B=4/OSR=20 config found in the
encoder-sensitivity study to reach ~500 unique addresses within BLE's
+-150 kHz range.

Run: python -m sim.performance_at_config [--B 4] [--osr 20] [--N 128]
     [--trials 800] [--seed 42] [--tag b4_osr20]
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from . import experiments as exp
from . import plots


def make_report(B: int, osr: int, N: int, n_trials: int, seed: int, tag: str,
                 ebn0_list=None, mask_preamble: bool = False, force: bool = False):
    if ebn0_list is None:
        ebn0_list = list(range(0, 21, 2))
    cfg = exp.Config(osr=osr, n_sym=exp.DEFAULT_N_SYM, B=B, diff_delay=osr)
    data = exp.performance_vs_snr_full(cfg, ebn0_list, N=N, n_trials=n_trials,
                                        seed=seed, mask_preamble=mask_preamble, force=force)

    import matplotlib.pyplot as plt
    plots.set_ieee_style()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(plots.IEEE_WIDTH_IN * 2 + 0.6, 2.3))

    labels = {"cam_run_midpoint": "CAM (run-midpoint)", "cam_argmin": "CAM (argmin, upper bound)",
              "b1_sequential": "B1 sequential", "b2_parallel": "B2 parallel"}
    methods = list(data["methods"])

    for mi, m in enumerate(methods):
        pmiss = 1.0 - data["pdet"][mi]
        ax1.plot(data["ebn0_list"], pmiss, marker="o", markersize=2, label=labels[m])
    ax1.set_xlabel("Eb/N0 (dB)")
    ax1.set_ylabel("P(miss) = 1 - P(detect)")
    ax1.set_yscale("log")
    ax1.set_ylim(1e-3, 1.05)
    ax1.set_title("(a) Missed detection")
    ax1.legend(fontsize=5)

    for mi, m in enumerate(methods):
        if m == "cam_argmin":
            continue  # no false-alarm concept for argmin (always answers)
        ax2.plot(data["ebn0_list"], np.clip(data["pfa"][mi], 1e-4, 1.0), marker="o",
                  markersize=2, label=labels[m])
    ax2.set_xlabel("Eb/N0 (dB)")
    ax2.set_ylabel("P(false alarm)")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-4, 1.05)
    ax2.set_title("(b) False alarm")

    for mi, m in enumerate(methods):
        rms_plot = np.where(data["pdet"][mi] >= 0.1, data["rms"][mi], np.nan)
        ax3.plot(data["ebn0_list"], rms_plot / 1e3, marker="o", markersize=2, label=labels[m])
    ax3.set_xlabel("Eb/N0 (dB)")
    ax3.set_ylabel("RMS CFO error (kHz)")
    ax3.set_yscale("log")
    ax3.set_title("(c) CFO estimation accuracy")

    mask_note = f", preamble masked ({int(data['effective_W'])}/{int(data['W'])} active bits)" if mask_preamble else ""
    fig.suptitle(f"Performance at B={B}, OSR={osr}, N={N} "
                 f"(W={int(data['W'])}, $\\tau$={int(data['tau'])}{mask_note})", y=1.03)
    fig.tight_layout(pad=0.4)
    path = os.path.join(plots.FIGURES_DIR, f"performance_{tag}.pdf")
    os.makedirs(plots.FIGURES_DIR, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)

    # summary table
    lines = ["| Eb/N0 (dB) | " + " | ".join(f"{labels[m]} Pd" for m in methods) + " |",
             "|" + "---|" * (len(methods) + 1)]
    for ei, ebn0 in enumerate(ebn0_list):
        row = [f"{ebn0}"] + [f"{data['pdet'][mi, ei]:.3f}" for mi in range(len(methods))]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append(f"tau={int(data['tau'])}, threshold={float(data['threshold']):.4f}, "
                 f"W={int(data['W'])}, effective_W={int(data['effective_W'])}, N={N}, "
                 f"mask_preamble={mask_preamble}")
    table_path = os.path.join(plots.FIGURES_DIR, f"performance_{tag}.md")
    with open(table_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return path, table_path, data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=4)
    parser.add_argument("--osr", type=int, default=20)
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--trials", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--mask-preamble", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    tag = args.tag or f"B{args.B}_osr{args.osr}_N{args.N}"
    if args.mask_preamble:
        tag += "_maskpre"

    fig_path, table_path, data = make_report(args.B, args.osr, args.N, args.trials,
                                              args.seed, tag,
                                              mask_preamble=args.mask_preamble,
                                              force=args.force)
    print("wrote", fig_path)
    print("wrote", table_path)


if __name__ == "__main__":
    main()
