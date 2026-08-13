"""Two-stage (tau-ramp) successive refinement: query the CAM twice at the
same key with a loose tau1 (the existing Pfa-budget-calibrated detection
threshold) and a tighter tau2, refining the run-midpoint estimate when the
tighter read also fires. Both reads are ordinary binary CAM match vectors --
tau is just re-applied as a second global sense margin, exactly as real
hardware would ramp it -- so this stays within the "no distances, ever"
constraint. See sim/decode.py two_stage_midpoint() and
docs/two_stage_findings.md.

Run: python -m sim.two_stage_experiment [--trials 800] [--seed 42]
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from . import plots
from . import experiments as exp
from .channel import run_channel
from .encode import encode_waveform
from .cam import CAM
from . import decode as dec


def run(cfg=None, N=32, ebn0_list=None, n_trials=800, seed=42, target_pfa=0.01,
        tau2_frac=0.6):
    if cfg is None:
        cfg = exp.Config()
    if ebn0_list is None:
        ebn0_list = list(range(6, 21, 2))
    tx = exp.make_tx(cfg)
    cb = exp.make_codebook(cfg, N=N)
    cam = CAM(cb.rows)

    tau1 = exp.calibrate_tau(cfg, cb, ebn0=max(ebn0_list), target_pfa=target_pfa,
                              n_trials=max(400, n_trials // 2), seed=seed)
    tau2 = int(tau2_frac * tau1)

    rng = np.random.default_rng(seed + 500)
    pdet_single = np.zeros(len(ebn0_list))
    rms_single = np.zeros(len(ebn0_list))
    pdet_two = np.zeros(len(ebn0_list))
    rms_two = np.zeros(len(ebn0_list))

    for ei, ebn0 in enumerate(ebn0_list):
        errs_single, errs_two = [], []
        hits_single = hits_two = 0
        for _ in range(n_trials):
            true_df = rng.uniform(cfg.df_min, cfg.df_max)
            r = run_channel(tx, true_df, cfg.osr, ebn0, rng)
            key = encode_waveform(r, cfg.B, diff_delay=cfg.diff_delay)

            m1 = cam.query(key, tau1)
            res1 = dec.run_midpoint(m1, cb.df_grid)
            if res1.detected:
                hits_single += 1
                errs_single.append(res1.df_hat - true_df)

            m2 = cam.query(key, tau2)
            res2 = dec.two_stage_midpoint(m1, m2, cb.df_grid)
            if res2.detected:
                hits_two += 1
                errs_two.append(res2.df_hat - true_df)

        pdet_single[ei] = hits_single / n_trials
        rms_single[ei] = np.sqrt(np.mean(np.square(errs_single))) if errs_single else np.nan
        pdet_two[ei] = hits_two / n_trials
        rms_two[ei] = np.sqrt(np.mean(np.square(errs_two))) if errs_two else np.nan

    return dict(ebn0_list=np.array(ebn0_list), W=cb.W, tau1=tau1, tau2=tau2,
                pdet_single=pdet_single, rms_single=rms_single,
                pdet_two=pdet_two, rms_two=rms_two)


def make_figure(result, tag="two_stage"):
    import matplotlib.pyplot as plt
    plots.set_ieee_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(plots.IEEE_WIDTH_IN * 2 + 0.3, 2.3))
    ebn0 = result["ebn0_list"]
    ax1.plot(ebn0, result["pdet_single"], marker="o", markersize=2, color="#dd8452",
              label="Single tau (run_midpoint)")
    ax1.plot(ebn0, result["pdet_two"], marker="o", markersize=2, color="#4c72b0",
              label="Two-stage refinement")
    ax1.set_xlabel("Eb/N0 (dB)")
    ax1.set_ylabel("Detection probability")
    ax1.set_title("(a) Detection (identical by construction)")
    ax1.legend(fontsize=5)

    rms1 = np.where(result["pdet_single"] >= 0.1, result["rms_single"], np.nan)
    rms2 = np.where(result["pdet_two"] >= 0.1, result["rms_two"], np.nan)
    ax2.plot(ebn0, rms1 / 1e3, marker="o", markersize=2, color="#dd8452",
              label="Single tau (run_midpoint)")
    ax2.plot(ebn0, rms2 / 1e3, marker="o", markersize=2, color="#4c72b0",
              label="Two-stage refinement")
    ax2.set_xlabel("Eb/N0 (dB)")
    ax2.set_ylabel("RMS CFO error (kHz)")
    ax2.set_title("(b) CFO estimation accuracy")
    ax2.legend(fontsize=5)

    fig.suptitle(f"Two-stage successive refinement (tau1={result['tau1']}, "
                 f"tau2={result['tau2']}, W={result['W']})", y=1.03)
    fig.tight_layout(pad=0.4)
    path = os.path.join(plots.FIGURES_DIR, f"{tag}_comparison.pdf")
    os.makedirs(plots.FIGURES_DIR, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tau2-frac", type=float, default=0.6)
    args = parser.parse_args()
    result = run(n_trials=args.trials, seed=args.seed, tau2_frac=args.tau2_frac)

    fig_path = make_figure(result)
    print("wrote", fig_path)

    lines = ["# Two-stage (tau-ramp) successive refinement\n",
            f"tau1={result['tau1']}, tau2={result['tau2']} "
            f"(frac={args.tau2_frac}), W={result['W']}\n",
            "| Eb/N0 | Pd (both, identical) | RMS single-tau (kHz) | RMS two-stage (kHz) | Improvement |",
            "|---|---|---|---|---|"]
    for i, e in enumerate(result["ebn0_list"]):
        r1, r2 = result["rms_single"][i], result["rms_two"][i]
        pct = f"{(1 - r2/r1)*100:.0f}%" if r1 and not np.isnan(r1) and not np.isnan(r2) else "n/a"
        lines.append(f"| {e} | {result['pdet_single'][i]:.3f} | {r1/1e3:.2f} | {r2/1e3:.2f} | {pct} |")
    md_path = os.path.join(plots.FIGURES_DIR, "two_stage_comparison.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", md_path)


if __name__ == "__main__":
    main()
