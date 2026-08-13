"""B3: a fair, low-power correlator peer for the CAM.

Table I as originally built compares the CAM (a binary readout) against B1
and B2, both FULL COMPLEX-PRECISION correlators -- an unfair comparison,
since the CAM never gets that precision either. B3 is the same N-hypothesis,
1-cycle-latency parallel bank as B2, but both the received signal and the
reference waveforms are quantized to 1-bit I/Q before correlating (a
sign-sign correlator: each per-sample product is an XNOR/sign-flip, not a
real multiply -- the natural low-power peer to a binary CAM).

Run: python -m sim.baseline_b3_experiment [--trials 800] [--seed 42]
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from . import plots
from . import experiments as exp
from . import baselines as bl
from . import decode as dec
from .tx import tx_waveform, random_valid_access_address
from .channel import run_channel
from .encode import encode_waveform
from .cam import CAM


def _false_alarm_rate_sign(sign_bank, threshold, n_sym, osr, ebn0, n_trials, rng):
    fa = 0
    for _ in range(n_trials):
        aa = random_valid_access_address(rng)
        tx = tx_waveform(n_sym=n_sym, osr=osr, access_address=aa)
        df = rng.uniform(-150e3, 150e3)
        r = run_channel(tx, df, osr, ebn0, rng)
        mags = bl.correlate_bank_sign(r, sign_bank)
        if bool(mags.max() >= threshold):
            fa += 1
    return fa / n_trials


def _calibrate_sign_threshold(sign_bank, n_sym, osr, ebn0_list, target_pfa, n_trials, seed,
                               thresh_range=None):
    """Pfa-based calibration for B3, checking both ends of ebn0_list (see
    _calibrate_full_threshold docstring for why a single assumed-worst-case
    point is not safe to rely on without checking)."""
    rng = np.random.default_rng(seed)
    if thresh_range is None:
        thresh_range = np.linspace(0.3, 1.8, 60)
    candidates = []
    for ebn0_check in (min(ebn0_list), max(ebn0_list)):
        for thr in thresh_range:
            pfa = _false_alarm_rate_sign(sign_bank, thr, n_sym, osr, ebn0_check, n_trials, rng)
            if pfa <= target_pfa:
                candidates.append(float(thr))
                break
        else:
            candidates.append(float(thresh_range[-1]))
    return max(candidates)


def _false_alarm_rate_full(bank, threshold, n_sym, osr, ebn0, n_trials, rng):
    fa = 0
    for _ in range(n_trials):
        aa = random_valid_access_address(rng)
        tx = tx_waveform(n_sym=n_sym, osr=osr, access_address=aa)
        df = rng.uniform(-150e3, 150e3)
        r = run_channel(tx, df, osr, ebn0, rng)
        mags = bl.correlate_bank(r, bank)
        if bool(mags.max() >= threshold):
            fa += 1
    return fa / n_trials


def _calibrate_full_threshold(bank, n_sym, osr, ebn0_list, target_pfa, n_trials, seed,
                               thresh_range=None):
    """Pfa-based calibration for B1/B2, mirroring _calibrate_sign_threshold,
    so B1/B2 and B3 are compared at operating points chosen by the SAME
    methodology. The project's existing calibrate_b_threshold (used by
    performance_vs_snr_full/Table I) is Pd-quantile-based instead -- fine
    for those figures since B1/B2's Pfa was already ~0 there regardless of
    calibration method, but NOT fine for a head-to-head against B3, where a
    Pd-quantile threshold (pinned to exactly target_pd at one SNR by
    construction) vs. a Pfa-budget threshold are answering different
    questions and would make any comparison an artifact of the mismatched
    methodology, not a real result.

    UNLIKE the CAM (Pfa worst at high SNR, see experiments.py), the
    full-precision correlator's Pfa is worst at LOW SNR (more independent
    noisy hypotheses get a chance to cross threshold by chance when the
    whole bank is noisy) -- confirmed empirically (9.3% Pfa at 0dB vs ~0%
    at 16dB for a mid-range threshold). So this checks BOTH ends of
    ebn0_list and returns the threshold satisfying the tighter constraint,
    rather than assuming a direction the way the CAM calibration can.
    """
    rng = np.random.default_rng(seed)
    if thresh_range is None:
        thresh_range = np.linspace(0.3, 1.2, 60)
    candidates = []
    for ebn0_check in (min(ebn0_list), max(ebn0_list)):
        for thr in thresh_range:
            pfa = _false_alarm_rate_full(bank, thr, n_sym, osr, ebn0_check, n_trials, rng)
            if pfa <= target_pfa:
                candidates.append(float(thr))
                break
        else:
            candidates.append(float(thresh_range[-1]))
    return max(candidates)


def run(n_sym=40, osr=4, N=32, ebn0_list=None, n_trials=800, seed=42, target_pfa=0.01):
    if ebn0_list is None:
        ebn0_list = list(range(0, 21, 2))
    cfg = exp.Config(osr=osr, n_sym=n_sym, B=2, diff_delay=osr)
    tx = tx_waveform(n_sym=n_sym, osr=osr)

    cb = exp.make_codebook(cfg, N=N)
    cam = CAM(cb.rows)
    tau = exp.calibrate_tau(cfg, cb, ebn0=max(ebn0_list), target_pfa=target_pfa,
                             n_trials=max(400, n_trials // 2), seed=seed)

    bank = bl.build_correlator_bank(N=N, df_min=cfg.df_min, df_max=cfg.df_max,
                                     n_sym=n_sym, osr=osr)
    sign_bank = bl.build_sign_correlator_bank(bank)

    threshold_full = _calibrate_full_threshold(bank, n_sym, osr, ebn0_list=ebn0_list,
                                                target_pfa=target_pfa, n_trials=max(400, n_trials // 2),
                                                seed=seed + 1)
    threshold_sign = _calibrate_sign_threshold(sign_bank, n_sym, osr, ebn0_list=ebn0_list,
                                                target_pfa=target_pfa, n_trials=max(400, n_trials // 2),
                                                seed=seed + 2)

    methods = ["cam", "b1_b2", "b3_sign"]
    pdet = {m: np.zeros(len(ebn0_list)) for m in methods}
    rms = {m: np.zeros(len(ebn0_list)) for m in methods}
    pfa = {m: np.zeros(len(ebn0_list)) for m in methods}

    rng = np.random.default_rng(seed + 500)
    for ei, ebn0 in enumerate(ebn0_list):
        errs = {m: [] for m in methods}
        hits = {m: 0 for m in methods}
        for _ in range(n_trials):
            true_df = rng.uniform(cfg.df_min, cfg.df_max)
            r = run_channel(tx, true_df, osr, ebn0, rng)

            key = encode_waveform(r, cfg.B, diff_delay=cfg.diff_delay)
            m_vec = cam.query(key, tau)
            res = dec.run_midpoint(m_vec, cb.df_grid)
            if res.detected:
                hits["cam"] += 1
                errs["cam"].append(res.df_hat - true_df)

            mags_full = bl.correlate_bank(r, bank)
            k_full = int(np.argmax(mags_full))
            if mags_full[k_full] >= threshold_full:
                hits["b1_b2"] += 1
                errs["b1_b2"].append(bank.df_grid[k_full] - true_df)

            mags_sign = bl.correlate_bank_sign(r, sign_bank)
            k_sign = int(np.argmax(mags_sign))
            if mags_sign[k_sign] >= threshold_sign:
                hits["b3_sign"] += 1
                errs["b3_sign"].append(sign_bank.df_grid[k_sign] - true_df)

        fa_trials = max(150, n_trials // 4)
        for m in methods:
            pdet[m][ei] = hits[m] / n_trials
            rms[m][ei] = np.sqrt(np.mean(np.square(errs[m]))) if errs[m] else np.nan

        fa_counts = {m: 0 for m in methods}
        for _ in range(fa_trials):
            aa = random_valid_access_address(rng)
            tx_wrong = tx_waveform(n_sym=n_sym, osr=osr, access_address=aa)
            df_wrong = rng.uniform(cfg.df_min, cfg.df_max)
            r_wrong = run_channel(tx_wrong, df_wrong, osr, ebn0, rng)

            key_wrong = encode_waveform(r_wrong, cfg.B, diff_delay=cfg.diff_delay)
            if dec.any_fire(cam.query(key_wrong, tau)):
                fa_counts["cam"] += 1

            mags_full = bl.correlate_bank(r_wrong, bank)
            if mags_full.max() >= threshold_full:
                fa_counts["b1_b2"] += 1

            mags_sign = bl.correlate_bank_sign(r_wrong, sign_bank)
            if mags_sign.max() >= threshold_sign:
                fa_counts["b3_sign"] += 1

        for m in methods:
            pfa[m][ei] = fa_counts[m] / fa_trials

    N_s = bank.N_s
    W = cb.W
    area = dict(cam=N * W, b1_seq=1 * N_s, b2_par=N * N_s, b3_sign=N * N_s)
    latency = dict(cam=1, b1_seq=N, b2_par=1, b3_sign=1)

    return dict(
        ebn0_list=np.array(ebn0_list), methods=methods,
        pdet=pdet, rms=rms, pfa=pfa,
        tau=tau, threshold_full=threshold_full, threshold_sign=threshold_sign,
        W=W, N=N, N_s=N_s, area=area, latency=latency,
    )


def make_figure(result, tag="baseline_b3"):
    import matplotlib.pyplot as plt
    plots.set_ieee_style()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(plots.IEEE_WIDTH_IN * 2 + 0.6, 2.3))
    labels = {"cam": "CAM", "b1_b2": "B1/B2 (full-precision)", "b3_sign": "B3 (1-bit sign-sign)"}
    colors = {"cam": "#4c72b0", "b1_b2": "#55a868", "b3_sign": "#c44e52"}
    ebn0 = result["ebn0_list"]
    for m in result["methods"]:
        pmiss = 1.0 - result["pdet"][m]
        ax1.plot(ebn0, pmiss, marker="o", markersize=2, color=colors[m], label=labels[m])
        ax2.plot(ebn0, np.clip(result["pfa"][m], 1e-4, 1.0), marker="o", markersize=2,
                  color=colors[m], label=labels[m])
        rms_plot = np.where(result["pdet"][m] >= 0.1, result["rms"][m], np.nan)
        ax3.plot(ebn0, rms_plot / 1e3, marker="o", markersize=2, color=colors[m], label=labels[m])
    ax1.set_xlabel("Eb/N0 (dB)")
    ax1.set_ylabel("P(miss)")
    ax1.set_yscale("log")
    ax1.set_ylim(1e-3, 1.05)
    ax1.set_title("(a) Missed detection")
    ax1.legend(fontsize=5)
    ax2.set_xlabel("Eb/N0 (dB)")
    ax2.set_ylabel("P(false alarm)")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-4, 1.05)
    ax2.set_title("(b) False alarm")
    ax3.set_xlabel("Eb/N0 (dB)")
    ax3.set_ylabel("RMS CFO error (kHz)")
    ax3.set_yscale("log")
    ax3.set_title("(c) CFO accuracy")
    fig.suptitle("CAM vs. full-precision (B1/B2) vs. 1-bit sign-sign (B3) correlator", y=1.03)
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
    args = parser.parse_args()
    result = run(n_trials=args.trials, seed=args.seed)

    fig_path = make_figure(result)
    print("wrote", fig_path)

    lines = ["# B3 (1-bit sign-sign correlator): a fair low-power peer for the CAM\n"]
    lines.append(f"tau={result['tau']}, threshold_full={result['threshold_full']:.4f}, "
                 f"threshold_sign={result['threshold_sign']:.4f}, W={result['W']}, "
                 f"N={result['N']}, N_s={result['N_s']}\n")
    lines.append("| System | Latency (cycles) | Area proxy (op-count) | " +
                 " | ".join(f"Pd@{e}dB" for e in [8, 12, 16, 20]) + " |")
    lines.append("|---|---|---|" + "---|" * 4)
    name_map = {"cam": "CAM", "b1_b2": "B1/B2 (full-precision)", "b3_sign": "B3 (1-bit sign-sign)"}
    area_map = {"cam": result["area"]["cam"], "b1_b2": result["area"]["b2_par"],
                "b3_sign": result["area"]["b3_sign"]}
    latency_map = {"cam": 1, "b1_b2": 1, "b3_sign": 1}
    ebn0_list = list(result["ebn0_list"])
    for m in result["methods"]:
        idxs = [ebn0_list.index(e) for e in [8, 12, 16, 20] if e in ebn0_list]
        pdvals = " | ".join(f"{result['pdet'][m][i]:.3f}" for i in idxs)
        lines.append(f"| {name_map[m]} | {latency_map[m]} | {area_map[m]} | {pdvals} |")
    lines.append("\nArea proxy note: B3's op-count matches B2's (N x N_s), but each op is a "
                 "1-bit XNOR/sign-flip, not a complex multiply-accumulate -- the two proxies "
                 "are NOT directly comparable in silicon area despite the same count. B3 is "
                 "the fairer *precision* peer to the CAM's binary readout; B1/B2 remain in "
                 "the comparison as the full-precision upper bound.\n")

    for m in result["methods"]:
        lines.append(f"\n## {name_map[m]}\n")
        lines.append("| Eb/N0 | Pd | RMS (kHz) | Pfa |")
        lines.append("|---|---|---|---|")
        for i, e in enumerate(result["ebn0_list"]):
            lines.append(f"| {e} | {result['pdet'][m][i]:.3f} | "
                         f"{result['rms'][m][i]/1e3:.2f} | {result['pfa'][m][i]:.4f} |")

    md_path = os.path.join(plots.FIGURES_DIR, "baseline_b3_comparison.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", md_path)


if __name__ == "__main__":
    main()
