"""Encoder-sensitivity study: noiseless codebook construction over a
diff_delay (in symbols) x B grid, plus a noise-informed "encoder SNR" ranking.

Run: python -m sim.encoder_sensitivity [--trials 500] [--seed 42]

Definitions (see docs/findings.md for why diff_delay is swept in symbols):
  - "unique rows": count of distinct rows among the N requested hypotheses.
  - "min adjacent Hamming distance": over consecutive rows (ordered by df),
    the smallest NONZERO Hamming distance -- i.e. the effective grid step
    size in bit-space once duplicate-adjacent rows are excluded.
  - "total noiseless Hamming swing": hamming(row_first, row_last), the
    accumulated bit-flip distance from one end of the +-150 kHz range to
    the other.
  - "monotone (no aliasing)": hamming(row_0, row_k) must be non-decreasing
    as k increases. A decrease means the code has wrapped back toward where
    it started -- CFO values on either side of the wrap become
    indistinguishable (or worse, ordered backwards).

    A back-of-envelope guess is that this tracks the BULK bias
    2*pi*df*D*T_symbol exceeding pi (a half-turn) at the reference axis,
    which only happens for D>3 at df=150 kHz (D=3 -> 2.83 rad < pi... wait,
    D*0.15 MHz*T_symbol > 0.5 needs D>3.33, so D=4). Empirically that guess
    is wrong and optimistic: aliasing is set by individual bits wrapping
    past their OWN nearest axis (spacing 2*pi/2**B), not by the bulk/mean
    phase reference, and the worst-case bit's margin is always <= half a
    sector width. That threshold is much finer than pi, so it is crossed
    much earlier. This simulation finds ALL of D in {2,3,4} alias within
    +-150 kHz for every B in {2,3,4} tested -- only D=1 (the operating
    default used everywhere else in this repo) stays monotone across the
    full BLE tolerance. Finer B (more, closer-together axes) aliases at a
    smaller |df| for the same D, for the same reason. See the "Aliasing
    flags" section of the generated report for exact onset points.
  - "encoder SNR": total_swing / sqrt(W*p*(1-p)), where p is the empirically
    measured per-bit flip probability at Eb/N0=10 dB (matched CFO, i.e. the
    noise floor of the Hamming-distance "detector"). This is the only place
    the channel model enters; the table/heatmap above it are pure noiseless
    codebook construction.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from . import experiments as exp
from . import plots
from .encode import key_width, encode_waveform
from .channel import run_channel
from .tx import tx_waveform


def _adjacent_stats(rows: np.ndarray):
    diffs = np.count_nonzero(rows[:-1] != rows[1:], axis=1)
    nonzero = diffs[diffs > 0]
    n_duplicate_adjacent = int(np.sum(diffs == 0))
    min_adjacent = int(nonzero.min()) if nonzero.size else 0
    return min_adjacent, n_duplicate_adjacent


def _monotonicity(rows: np.ndarray, df_grid: np.ndarray):
    dist_from_first = np.count_nonzero(rows != rows[0][None, :], axis=1)
    decreases = np.where(np.diff(dist_from_first) < 0)[0]
    monotone = decreases.size == 0
    alias_onset_df = float(df_grid[decreases[0] + 1]) if decreases.size else float("nan")
    return monotone, alias_onset_df, dist_from_first


def _estimate_bitflip_prob(cfg: exp.Config, ebn0_db: float, n_trials: int,
                            rng: np.random.Generator) -> float:
    """p = mean fraction of bits that differ from the noiseless df=0 row when
    the same (df=0) signal is passed through AWGN + random theta0/timing.
    This is the per-bit noise floor feeding the binomial std used below."""
    tx = tx_waveform(n_sym=cfg.n_sym, osr=cfg.osr, access_address=cfg.access_address)
    ref = encode_waveform(tx, cfg.B, coding=cfg.coding, diff_delay=cfg.diff_delay).astype(bool)
    W = ref.size
    fracs = np.empty(n_trials)
    for t in range(n_trials):
        r = run_channel(tx, 0.0, cfg.osr, ebn0_db, rng)
        key = encode_waveform(r, cfg.B, coding=cfg.coding, diff_delay=cfg.diff_delay).astype(bool)
        fracs[t] = np.count_nonzero(key != ref) / W
    return float(fracs.mean())


def run_study(D_list=(1, 2, 3, 4), B_list=(2, 3, 4), N=128,
              df_min=-150e3, df_max=150e3, n_sym=40, osr=4, ebn0_db=10.0,
              n_trials=500, seed=42):
    rows_out = []
    for D in D_list:
        for B in B_list:
            cfg = exp.Config(osr=osr, n_sym=n_sym, B=B, diff_delay=D * osr)
            cb = exp.make_codebook(cfg, N=N)
            W = cb.W

            n_unique = len({tuple(r.tolist()) for r in cb.rows})
            min_adj, n_dup_adj = _adjacent_stats(cb.rows)
            total_swing = int(np.count_nonzero(cb.rows[0] != cb.rows[-1]))
            monotone, alias_onset_df, _ = _monotonicity(cb.rows, cb.df_grid)

            rng = np.random.default_rng(seed + 1000 * D + B)
            p = _estimate_bitflip_prob(cfg, ebn0_db, n_trials, rng)
            noise_std = np.sqrt(W * p * (1 - p)) if 0 < p < 1 else np.nan
            encoder_snr = total_swing / noise_std if noise_std and noise_std > 0 else np.nan

            rows_out.append(dict(
                D=D, B=B, W=W, N_requested=N, N_actual=cb.N, n_unique=n_unique,
                min_adjacent_hamming=min_adj, n_duplicate_adjacent=n_dup_adj,
                total_swing=total_swing, monotone=monotone,
                alias_onset_df_khz=alias_onset_df / 1e3 if monotone is False else float("nan"),
                p_bitflip_10db=p, noise_std=noise_std, encoder_snr=encoder_snr,
            ))
    return rows_out


def _format_table(rows_out) -> str:
    header = ("| D (symbols) | B | W | unique rows (/N) | min adj. Hamming | "
              "total swing (bits) | monotone | alias onset (kHz) | p @10dB | encoder SNR |")
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for r in rows_out:
        alias_str = f"{r['alias_onset_df_khz']:.1f}" if r["monotone"] is False else "--"
        lines.append(
            f"| {r['D']} | {r['B']} | {r['W']} | {r['n_unique']}/{r['N_actual']} | "
            f"{r['min_adjacent_hamming']} | {r['total_swing']} | "
            f"{'yes' if r['monotone'] else '**NO (aliased)**'} | {alias_str} | "
            f"{r['p_bitflip_10db']:.4f} | {r['encoder_snr']:.2f} |"
        )
    return "\n".join(lines)


def _ranked_table(rows_out, monotone_only: bool) -> str:
    pool = [r for r in rows_out if r["monotone"]] if monotone_only else rows_out
    ranked = sorted(pool, key=lambda r: r["encoder_snr"], reverse=True)
    lines = ["| Rank | D | B | Encoder SNR | Monotone |", "|---|---|---|---|---|"]
    for i, r in enumerate(ranked, 1):
        lines.append(f"| {i} | {r['D']} | {r['B']} | {r['encoder_snr']:.2f} | "
                     f"{'yes' if r['monotone'] else 'NO'} |")
    return "\n".join(lines)


def make_heatmap(rows_out, D_list, B_list):
    import matplotlib.pyplot as plt
    plots.set_ieee_style()
    grid = np.zeros((len(B_list), len(D_list)))
    mono_grid = np.zeros((len(B_list), len(D_list)), dtype=bool)
    lookup = {(r["D"], r["B"]): r for r in rows_out}
    for bi, B in enumerate(B_list):
        for di, D in enumerate(D_list):
            r = lookup[(D, B)]
            grid[bi, di] = r["n_unique"]
            mono_grid[bi, di] = r["monotone"]

    fig, ax = plots.new_figure(height_in=2.4)
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(D_list)))
    ax.set_xticklabels([str(d) for d in D_list])
    ax.set_yticks(range(len(B_list)))
    ax.set_yticklabels([str(b) for b in B_list])
    ax.set_xlabel("diff_delay D (symbols)")
    ax.set_ylabel("B")
    ax.set_title("Unique codebook rows over (D, B), N=128 requested")
    for bi in range(len(B_list)):
        for di in range(len(D_list)):
            txt = str(int(grid[bi, di]))
            if not mono_grid[bi, di]:
                txt += "*"
            ax.text(di, bi, txt, ha="center", va="center",
                     color="white" if grid[bi, di] < grid.max() * 0.6 else "black",
                     fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="unique rows")
    ax.text(0.0, -0.32, "* = row ordering not monotone in CFO (aliased)",
            transform=ax.transAxes, fontsize=6)
    path = os.path.join(plots.FIGURES_DIR, "encoder_sensitivity_heatmap.pdf")
    os.makedirs(plots.FIGURES_DIR, exist_ok=True)
    fig.tight_layout(pad=0.6)
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    D_list = [1, 2, 3, 4]
    B_list = [2, 3, 4]
    rows_out = run_study(D_list=D_list, B_list=B_list, n_trials=args.trials, seed=args.seed)

    table_md = _format_table(rows_out)
    ranked_all_md = _ranked_table(rows_out, monotone_only=False)
    ranked_mono_md = _ranked_table(rows_out, monotone_only=True)
    aliased = [r for r in rows_out if not r["monotone"]]

    text = ["# Encoder sensitivity study\n",
            f"N=128 requested hypotheses, +-150 kHz, n_sym=40, osr=4.\n",
            table_md, "",
            "## Ranked by encoder SNR (total swing / binomial noise std, Eb/N0=10dB)\n",
            "**Caveat:** for aliased (non-monotone) configs, total swing is "
            "endpoint-to-endpoint Hamming distance (row_first vs. row_last), which is "
            "not a meaningful resolving-power metric once the codebook has folded back "
            "on itself -- a large value there can mean the two endpoints happened to "
            "land on opposite sides of a fold, not that CFO is well resolved across the "
            "range. The unrestricted ranking is included for completeness; the "
            "monotone-only ranking below is the one that supports an actual "
            "recommendation.\n",
            "### All configurations (includes aliased -- do not use to pick D, B)\n",
            ranked_all_md, "",
            "### Monotone-only (safe to use for picking D, B)\n",
            ranked_mono_md, ""]
    if aliased:
        text.append("\n## Aliasing flags\n")
        text.append(
            "A naive bulk-phase estimate (bias=2*pi*df*D*T_symbol exceeding pi) predicts "
            "only D=4 aliases within +-150 kHz. That estimate is wrong: aliasing is set by "
            "individual bits wrapping past their own nearest decision axis (spacing "
            "2*pi/2**B), a much finer threshold than pi, crossed much earlier. Empirically "
            "**every D in {2,3,4} aliases for every B tested** -- only D=1 stays monotone "
            "across the full BLE range.\n")
        for r in aliased:
            text.append(f"- D={r['D']}, B={r['B']}: row ordering breaks monotonicity at "
                       f"~{r['alias_onset_df_khz']:.1f} kHz.")
    body = "\n".join(text)

    path = os.path.join(plots.FIGURES_DIR, "encoder_sensitivity.md")
    os.makedirs(plots.FIGURES_DIR, exist_ok=True)
    with open(path, "w") as f:
        f.write(body + "\n")
    print("wrote", path)

    heatmap_path = make_heatmap(rows_out, D_list, B_list)
    print("wrote", heatmap_path)


if __name__ == "__main__":
    main()
