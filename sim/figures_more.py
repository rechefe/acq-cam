"""Figures 3, 4, 5 and Table I."""
from __future__ import annotations

import os

import numpy as np

from . import experiments as exp
from . import plots


def fig3(cfg: exp.Config, n_trials: int, seed: int, force: bool):
    """Thermometer's theoretical advantage (the unit-tested circular metric
    property) shows up as a materially higher any-of-N detection probability
    at a fixed per-row operating tau -- nearby hypotheses are more likely to
    co-fire, which is exactly what feeds the run-midpoint estimator's
    contiguous band. It does NOT show up as a better RMS point estimate
    (Gray/one-hot are comparable or slightly better there); both panels are
    reported so the comparison is not one-sided.
    """
    ebn0_list = list(range(0, 21, 2))
    data = exp.encoding_comparison(ebn0_list, N=32, n_trials=n_trials, seed=seed, force=force)
    import matplotlib.pyplot as plt
    plots.set_ieee_style()
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(plots.IEEE_WIDTH_IN * 2 + 0.6, 2.3))
    codings = list(data["codings"])
    colors = {"thermometer": "#4c72b0", "gray": "#dd8452", "onehot": "#55a868"}
    for ci, coding in enumerate(codings):
        ax0.plot(data["ebn0_list"], data["pdet"][ci], marker="o", markersize=2,
                  label=coding, color=colors.get(coding))
    ax0.set_xlabel("Eb/N0 (dB)")
    ax0.set_ylabel("Detection probability")
    ax0.set_title("(a) Detection (any-fire)")
    ax0.legend()

    # RMS conditioned on detection is unreliable (and biased low) where Pd is
    # small -- only a handful of lucky, easy trials get counted. Suppress those.
    for ci, coding in enumerate(codings):
        rms_plot = np.where(data["pdet"][ci] >= 0.1, data["rms"][ci], np.nan)
        ax1.plot(data["ebn0_list"], rms_plot / 1e3, marker="o", markersize=2,
                  label=coding, color=colors.get(coding))
    ax1.set_xlabel("Eb/N0 (dB)")
    ax1.set_ylabel("RMS CFO error (kHz)")
    ax1.set_title("(b) Estimation error ($P_d \\geq 0.1$)")

    ax2.bar(codings, data["W"], color=[colors.get(c) for c in codings])
    ax2.set_ylabel("Key width W (bits)")
    ax2.set_title("(c) Area cost")
    fig.suptitle("Fig. 3: Encoding comparison", y=1.02)
    fig.tight_layout(pad=0.4)
    path = os.path.join(plots.FIGURES_DIR, "fig3_encoding_comparison.pdf")
    os.makedirs(plots.FIGURES_DIR, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)
    return path


def fig4(cfg: exp.Config, n_trials: int, seed: int, force: bool):
    ebn0_list = list(range(0, 21, 2))
    data = exp.performance_vs_snr(cfg, ebn0_list, N=32, n_trials=n_trials, seed=seed, force=force)
    import matplotlib.pyplot as plt
    plots.set_ieee_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(plots.IEEE_WIDTH_IN * 2 + 0.3, 2.3))
    labels = {"cam_run_midpoint": "CAM (run-midpoint)", "cam_argmin": "CAM (argmin, upper bound)",
              "b1_sequential": "B1 sequential", "b2_parallel": "B2 parallel"}
    methods = list(data["methods"])
    for mi, m in enumerate(methods):
        ax1.plot(data["ebn0_list"], data["pdet"][mi], marker="o", markersize=2, label=labels[m])
        rms_plot = np.where(data["pdet"][mi] >= 0.1, data["rms"][mi], np.nan)
        ax2.plot(data["ebn0_list"], rms_plot / 1e3, marker="o", markersize=2, label=labels[m])
    ax1.set_xlabel("Eb/N0 (dB)")
    ax1.set_ylabel("Detection probability")
    ax1.set_title("(a) Detection")
    ax1.legend(fontsize=5)
    ax2.set_xlabel("Eb/N0 (dB)")
    ax2.set_ylabel("RMS CFO error (kHz)")
    ax2.set_title("(b) CFO estimation error")
    ax2.set_yscale("log")
    fig.suptitle(f"Fig. 4: Performance vs. SNR ($\\tau$={int(data['tau'])}, W={int(data['W'])})", y=1.02)
    fig.tight_layout(pad=0.4)
    path = os.path.join(plots.FIGURES_DIR, "fig4_performance_vs_snr.pdf")
    os.makedirs(plots.FIGURES_DIR, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)
    return path


def fig5(cfg: exp.Config, n_trials: int, seed: int, force: bool):
    tau_fracs = np.linspace(0.08, 0.32, 7)
    delta_fracs = np.array([0.25, 0.5, 1.0, 1.5, 2.0])
    data = exp.threshold_grid_sweep(cfg, tau_fracs, delta_fracs, ebn0=10.0,
                                     n_trials=n_trials, seed=seed, force=force)
    import matplotlib.pyplot as plt
    plots.set_ieee_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(plots.IEEE_WIDTH_IN * 2 + 0.3, 2.3))
    extent = [delta_fracs[0], delta_fracs[-1], tau_fracs[0], tau_fracs[-1]]
    im1 = ax1.imshow(data["rms_grid"] / 1e3, aspect="auto", origin="lower", extent=extent,
                      cmap="magma")
    ax1.set_xlabel(r"$\delta$ / $W_c$")
    ax1.set_ylabel(r"$\tau$ / W")
    ax1.set_title("(a) RMS CFO error (kHz)")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    pfa_plot = np.log10(np.clip(data["pfa_grid"], 1e-4, 1.0))
    im2 = ax2.imshow(pfa_plot, aspect="auto", origin="lower", extent=extent, cmap="viridis")
    ax2.set_xlabel(r"$\delta$ / $W_c$")
    ax2.set_ylabel(r"$\tau$ / W")
    ax2.set_title(r"(b) $\log_{10}$ false-alarm rate")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    fig.suptitle("Fig. 5: Threshold and grid design", y=1.02)
    fig.tight_layout(pad=0.4)
    path = os.path.join(plots.FIGURES_DIR, "fig5_threshold_grid.pdf")
    os.makedirs(plots.FIGURES_DIR, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)
    return path


def table1(cfg: exp.Config, n_trials: int, seed: int, force: bool):
    data = exp.table1_operating_point(cfg, N=32, ebn0=10.0, n_trials=n_trials, seed=seed,
                                       force=force)
    lines = []
    lines.append("| System | Latency (cycles) | Area proxy (op-count) | Pd @ 10dB | RMS CFO err (kHz) |")
    lines.append("|---|---|---|---|---|")
    for i, m in enumerate(data["methods"]):
        lines.append(f"| {m} | {int(data['latency'][i])} | {int(data['area_proxy'][i])} | "
                     f"{data['pdet'][i]:.3f} | {data['rms'][i] / 1e3:.2f} |")
    lines.append("")
    lines.append(f"tau={int(data['tau'])}, W={int(data['W'])}, N={int(data['N'])}, "
                 f"grid resolution={float(data['resolution_hz']) / 1e3:.2f} kHz, "
                 f"correlator threshold={float(data['threshold']):.4f}")
    text = "\n".join(lines)
    path = os.path.join(plots.FIGURES_DIR, "table1.md")
    os.makedirs(plots.FIGURES_DIR, exist_ok=True)
    with open(path, "w") as f:
        f.write(text + "\n")
    return path
