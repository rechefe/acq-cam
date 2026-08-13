"""Composite ("vernier") key vs. matched-bit-budget single-lag baseline.

The encoder-sensitivity study found that any diff_delay D>1 symbol aliases
within +-150 kHz (spec's own D>3 rule-of-thumb was optimistic -- see
docs/findings.md). But "aliases" there meant a specific thing: aggregate
Hamming distance from a fixed reference row stops being monotone in CFO.
That breaks the simple midpoint-of-contiguous-run assumption, but it is NOT
the same as two well-separated CFOs producing near-identical codes -- that
distinction matters, and this experiment is built on it (see
docs/vernier_findings.md for the full nearest-neighbor collision analysis
that separates the two failure modes).

A composite key concatenates a coarse D=1 segment (weak sensitivity, but
provably collision-free across the whole range) with a fine D=3 segment
(strong sensitivity, but aliases badly alone). Because Hamming distance is
additive over concatenation, a hypothesis row that only matches the fine
segment still pays the coarse segment's mismatch penalty -- so the composite
key is empirically collision-free too (0/128 far false-lock pairs, vs 48/128
for D=3 alone), while keeping most of the fine segment's sensitivity.

Run: python -m sim.vernier_experiment [--trials 600] [--seed 42]
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from . import plots
from .vernier import LagSpec, build_composite_codebook, composite_encode
from .codebook import build_codebook
from .tx import tx_waveform, random_valid_access_address
from .channel import run_channel
from .encode import encode_waveform
from .cam import CAM
from . import decode as dec


def _false_alarm_rate(encode_fn, cam, tau, n_sym, osr, ebn0, n_trials, rng):
    fa = 0
    for _ in range(n_trials):
        aa = random_valid_access_address(rng)
        tx = tx_waveform(n_sym=n_sym, osr=osr, access_address=aa)
        df = rng.uniform(-150e3, 150e3)
        r = run_channel(tx, df, osr, ebn0, rng)
        key = encode_fn(r)
        if dec.any_fire(cam.query(key, tau)):
            fa += 1
    return fa / n_trials


def _calibrate_tau(encode_fn, cam, tau_range, n_sym, osr, ebn0_worst, target_pfa,
                    n_trials, seed):
    rng = np.random.default_rng(seed)
    best = tau_range[0]
    for tau in tau_range:
        pfa = _false_alarm_rate(encode_fn, cam, tau, n_sym, osr, ebn0_worst, n_trials, rng)
        if pfa <= target_pfa:
            best = tau
        else:
            break
    return best


def _sweep(encode_fn, rows, df_grid, tau, tx, n_sym, osr, ebn0_list, n_trials, seed):
    rng = np.random.default_rng(seed + 500)
    cam = CAM(rows)
    pdet = np.zeros(len(ebn0_list))
    rms = np.zeros(len(ebn0_list))
    pfa = np.zeros(len(ebn0_list))
    for ei, ebn0 in enumerate(ebn0_list):
        errs = []
        hits = 0
        for _ in range(n_trials):
            true_df = rng.uniform(-150e3, 150e3)
            r = run_channel(tx, true_df, osr, ebn0, rng)
            key = encode_fn(r)
            m = cam.query(key, tau)
            res = dec.run_midpoint(m, df_grid)
            if res.detected:
                hits += 1
                errs.append(res.df_hat - true_df)
        pdet[ei] = hits / n_trials
        rms[ei] = np.sqrt(np.mean(np.square(errs))) if errs else np.nan
        pfa[ei] = _false_alarm_rate(encode_fn, cam, tau, n_sym, osr, ebn0, max(100, n_trials // 3), rng)
    return pdet, rms, pfa


def run(n_sym=40, osr=4, N=32, ebn0_list=None, n_trials=600, seed=42,
        target_pfa=0.01, tau_frac=0.22):
    if ebn0_list is None:
        ebn0_list = list(range(6, 21, 2))
    tx = tx_waveform(n_sym=n_sym, osr=osr)

    lags = [LagSpec(1, 2), LagSpec(3, 2)]
    cb_comp = build_composite_codebook(lags, N=N, df_min=-150e3, df_max=150e3,
                                        n_sym=n_sym, osr=osr)
    encode_comp = lambda r: composite_encode(r, osr, lags)
    cam_comp = CAM(cb_comp.rows)
    tau_range_comp = range(int(0.10 * cb_comp.W), int(0.40 * cb_comp.W), max(1, cb_comp.W // 150))
    tau_comp = _calibrate_tau(encode_comp, cam_comp, list(tau_range_comp), n_sym, osr,
                               ebn0_worst=max(ebn0_list), target_pfa=target_pfa,
                               n_trials=max(300, n_trials // 2), seed=seed)

    cb_single = build_codebook(N=N, df_min=-150e3, df_max=150e3, n_sym=n_sym, osr=osr,
                                B=3, diff_delay=osr)
    encode_single = lambda r: encode_waveform(r, 3, diff_delay=osr)
    cam_single = CAM(cb_single.rows)
    tau_range_single = range(int(0.10 * cb_single.W), int(0.40 * cb_single.W),
                              max(1, cb_single.W // 150))
    tau_single = _calibrate_tau(encode_single, cam_single, list(tau_range_single), n_sym, osr,
                                 ebn0_worst=max(ebn0_list), target_pfa=target_pfa,
                                 n_trials=max(300, n_trials // 2), seed=seed)

    pdet_c, rms_c, pfa_c = _sweep(encode_comp, cb_comp.rows, cb_comp.df_grid, tau_comp,
                                   tx, n_sym, osr, ebn0_list, n_trials, seed)
    pdet_s, rms_s, pfa_s = _sweep(encode_single, cb_single.rows, cb_single.df_grid, tau_single,
                                   tx, n_sym, osr, ebn0_list, n_trials, seed)

    return dict(
        ebn0_list=np.array(ebn0_list),
        composite=dict(W=cb_comp.W, tau=tau_comp, pdet=pdet_c, rms=rms_c, pfa=pfa_c,
                        label="Composite D1B2+D3B2"),
        single=dict(W=cb_single.W, tau=tau_single, pdet=pdet_s, rms=rms_s, pfa=pfa_s,
                    label="Single-lag D1B3 (matched W)"),
    )


def make_figure(result, tag="vernier"):
    import matplotlib.pyplot as plt
    plots.set_ieee_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(plots.IEEE_WIDTH_IN * 2 + 0.3, 2.3))
    ebn0 = result["ebn0_list"]
    for key, color in [("composite", "#4c72b0"), ("single", "#dd8452")]:
        d = result[key]
        ax1.plot(ebn0, d["pdet"], marker="o", markersize=2, color=color, label=d["label"])
        rms_plot = np.where(d["pdet"] >= 0.1, d["rms"], np.nan)
        ax2.plot(ebn0, rms_plot / 1e3, marker="o", markersize=2, color=color, label=d["label"])
    ax1.set_xlabel("Eb/N0 (dB)")
    ax1.set_ylabel("Detection probability")
    ax1.set_title("(a) Detection")
    ax1.legend(fontsize=5)
    ax2.set_xlabel("Eb/N0 (dB)")
    ax2.set_ylabel("RMS CFO error (kHz)")
    ax2.set_title("(b) CFO estimation accuracy")
    Wc, Ws = result["composite"]["W"], result["single"]["W"]
    fig.suptitle(f"Composite vernier key vs. matched-W single-lag "
                 f"(W={Wc} vs W={Ws})", y=1.03)
    fig.tight_layout(pad=0.4)
    path = os.path.join(plots.FIGURES_DIR, f"{tag}_comparison.pdf")
    os.makedirs(plots.FIGURES_DIR, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run(n_trials=args.trials, seed=args.seed)

    fig_path = make_figure(result)
    print("wrote", fig_path)

    lines = ["# Vernier composite-key vs. matched-W single-lag\n"]
    for key in ["composite", "single"]:
        d = result[key]
        lines.append(f"## {d['label']} (W={d['W']}, tau={d['tau']})\n")
        lines.append("| Eb/N0 | Pd | RMS (kHz) | Pfa |")
        lines.append("|---|---|---|---|")
        for i, e in enumerate(result["ebn0_list"]):
            lines.append(f"| {e} | {d['pdet'][i]:.3f} | {d['rms'][i]/1e3:.2f} | {d['pfa'][i]:.4f} |")
        lines.append("")
    text = "\n".join(lines)
    md_path = os.path.join(plots.FIGURES_DIR, "vernier_comparison.md")
    with open(md_path, "w") as f:
        f.write(text)
    print("wrote", md_path)


if __name__ == "__main__":
    main()
