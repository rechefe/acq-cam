"""Monte Carlo sweep runners with disk caching (npz) so figures can be
re-rendered without re-running the sweeps.

Operating defaults: OSR=4, N_sym=40, B=2, one-symbol (OSR-sample) differential
lag. See docs/findings.md for why the differential lag departs from the
adjacent-sample (delay=1) default implied by the spec's headline W=636
example -- that configuration was found, via the Figure 1/2 sanity gate, to
have a maximum per-bit CFO-induced bias of only ~0.24 rad across the full
+-150 kHz BLE range, which collapses the codebook to a handful of
distinguishable rows. A one-symbol lag (still fully theta0-cancelling)
restores a graded, monotonic response while changing nothing else about the
architecture.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, asdict

import numpy as np

from .tx import tx_waveform, STANDARD_ACCESS_ADDRESS, random_valid_access_address
from .channel import run_channel
from .encode import encode_waveform, key_width, CODES
from .codebook import build_codebook, capture_window_half_width
from .cam import CAM
from . import decode as dec
from . import baselines as bl

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")

DEFAULT_OSR = 4
DEFAULT_N_SYM = 40
DEFAULT_B = 2
DEFAULT_DIFF_DELAY = DEFAULT_OSR  # one symbol -- see module docstring
DEFAULT_DF_MIN = -150e3
DEFAULT_DF_MAX = 150e3
DEFAULT_N_ROWS = 32
BLE_TOLERANCE_HZ = 150e3


def _cache_key(name: str, params: dict) -> str:
    blob = json.dumps(params, sort_keys=True, default=str).encode()
    h = hashlib.sha1(blob).hexdigest()[:16]
    return f"{name}_{h}.npz"


def cached(name: str, params: dict, compute_fn, force: bool = False):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, _cache_key(name, params))
    if os.path.exists(path) and not force:
        data = np.load(path, allow_pickle=True)
        return {k: data[k] for k in data.files}
    result = compute_fn()
    np.savez(path, **result)
    return result


@dataclass
class Config:
    osr: int = DEFAULT_OSR
    n_sym: int = DEFAULT_N_SYM
    B: int = DEFAULT_B
    diff_delay: int = DEFAULT_DIFF_DELAY
    coding: str = "thermometer"
    access_address: int = STANDARD_ACCESS_ADDRESS
    df_min: float = DEFAULT_DF_MIN
    df_max: float = DEFAULT_DF_MAX

    def as_dict(self):
        return asdict(self)

    def W(self) -> int:
        return key_width(self.n_sym, self.osr, self.B, self.coding, self.diff_delay)


def make_tx(cfg: Config) -> np.ndarray:
    return tx_waveform(n_sym=cfg.n_sym, osr=cfg.osr, access_address=cfg.access_address)


def make_codebook(cfg: Config, N: int | None = None, delta: float | None = None,
                   dont_care_frac: float = 0.0):
    return build_codebook(N=N, delta=delta, df_min=cfg.df_min, df_max=cfg.df_max,
                           n_sym=cfg.n_sym, osr=cfg.osr, B=cfg.B, coding=cfg.coding,
                           access_address=cfg.access_address,
                           dont_care_frac=dont_care_frac, diff_delay=cfg.diff_delay)


# ---------------------------------------------------------------------------
# Fig 1: distance profile
# ---------------------------------------------------------------------------

def distance_profile(cfg: Config, ebn0_list, offsets, n_trials: int, seed: int,
                      force: bool = False):
    params = dict(cfg=cfg.as_dict(), ebn0_list=list(ebn0_list), offsets=list(offsets),
                  n_trials=n_trials, seed=seed, kind="distance_profile")

    def compute():
        rng = np.random.default_rng(seed)
        tx = make_tx(cfg)
        ref_row = encode_waveform(tx, cfg.B, coding=cfg.coding, diff_delay=cfg.diff_delay).astype(bool)
        mean_dist = np.zeros((len(ebn0_list), len(offsets)))
        std_dist = np.zeros((len(ebn0_list), len(offsets)))
        for i, ebn0 in enumerate(ebn0_list):
            for j, off in enumerate(offsets):
                d = np.empty(n_trials)
                for t in range(n_trials):
                    r = run_channel(tx, off, cfg.osr, ebn0, rng)
                    key = encode_waveform(r, cfg.B, coding=cfg.coding, diff_delay=cfg.diff_delay).astype(bool)
                    d[t] = np.count_nonzero(key != ref_row)
                mean_dist[i, j] = d.mean()
                std_dist[i, j] = d.std()
        return dict(ebn0_list=np.array(ebn0_list), offsets=np.array(offsets),
                    mean_dist=mean_dist, std_dist=std_dist, W=np.array(cfg.W()))

    return cached("fig1_distance_profile", params, compute, force=force)


# ---------------------------------------------------------------------------
# Fig 2: fired-set structure
# ---------------------------------------------------------------------------

def fired_set_heatmap(cfg: Config, true_cfos, N: int, tau: int, ebn0: float,
                       n_trials: int, seed: int, force: bool = False):
    params = dict(cfg=cfg.as_dict(), true_cfos=list(true_cfos), N=N, tau=tau,
                  ebn0=ebn0, n_trials=n_trials, seed=seed, kind="fired_set_heatmap")

    def compute():
        rng = np.random.default_rng(seed)
        tx = make_tx(cfg)
        cb = make_codebook(cfg, N=N)
        cam = CAM(cb.rows)
        fire_prob = np.zeros((N, len(true_cfos)))
        for j, tcf in enumerate(true_cfos):
            counts = np.zeros(N)
            for _ in range(n_trials):
                r = run_channel(tx, tcf, cfg.osr, ebn0, rng)
                key = encode_waveform(r, cfg.B, coding=cfg.coding, diff_delay=cfg.diff_delay)
                m = cam.query(key, tau)
                counts += m
            fire_prob[:, j] = counts / n_trials
        return dict(true_cfos=np.array(true_cfos), df_grid=cb.df_grid,
                    fire_prob=fire_prob, W=np.array(cb.W), delta=np.array(cb.delta),
                    tau=np.array(tau))

    return cached("fig2_fired_set", params, compute, force=force)


# ---------------------------------------------------------------------------
# tau calibration to a target false-alarm rate
# ---------------------------------------------------------------------------

def false_alarm_rate(cfg: Config, cb, tau: int, ebn0: float, n_trials: int,
                      rng: np.random.Generator) -> float:
    """Pfa = P(any row fires) when the input is NOT the codebook's access
    address (random valid AA, random CFO/theta0/timing/noise)."""
    cam = CAM(cb.rows)
    fa = 0
    for _ in range(n_trials):
        aa = random_valid_access_address(rng)
        tx = tx_waveform(n_sym=cfg.n_sym, osr=cfg.osr, access_address=aa)
        df = rng.uniform(cfg.df_min, cfg.df_max)
        r = run_channel(tx, df, cfg.osr, ebn0, rng)
        key = encode_waveform(r, cfg.B, coding=cfg.coding, diff_delay=cfg.diff_delay)
        m = cam.query(key, tau)
        if dec.any_fire(m):
            fa += 1
    return fa / n_trials


def calibrate_tau(cfg: Config, cb, ebn0: float, target_pfa: float, n_trials: int,
                   seed: int, tau_range=None):
    """Pfa(tau) is monotonically non-decreasing in tau (a looser threshold can
    only admit more false alarms), so the best operating point is the
    LARGEST tau that still satisfies the false-alarm budget -- loosest
    threshold consistent with the Pfa target maximizes detection probability.
    """
    rng = np.random.default_rng(seed)
    W = cb.W
    if tau_range is None:
        tau_range = range(int(0.05 * W), int(0.45 * W), max(1, W // 100))
    tau_range = list(tau_range)
    best_tau = tau_range[0]
    for tau in tau_range:
        pfa = false_alarm_rate(cfg, cb, tau, ebn0, n_trials, rng)
        if pfa <= target_pfa:
            best_tau = tau
        else:
            break
    return best_tau


def detection_prob_at_tau(cfg: Config, cb, tau: int, ebn0: float, n_trials: int,
                           rng: np.random.Generator, true_df: float = 0.0) -> float:
    """P(nearest-grid-row fires) at true_df. Deliberately checks the single
    nearest row rather than any(m) over the whole bank: an any-of-N criterion
    is biased loose (only the best of N hypotheses needs to cross tau), which
    would pick an operating tau far looser than what actually gives a tight,
    resolving fired-set band."""
    cam = CAM(cb.rows)
    tx = make_tx(cfg)
    nearest = int(np.argmin(np.abs(cb.df_grid - true_df)))
    hits = 0
    for _ in range(n_trials):
        r = run_channel(tx, true_df, cfg.osr, ebn0, rng)
        key = encode_waveform(r, cfg.B, coding=cfg.coding, diff_delay=cfg.diff_delay)
        if cam.query(key, tau)[nearest]:
            hits += 1
    return hits / n_trials


def calibrate_tau_for_detection(cfg: Config, cb, ebn0: float, target_pd: float,
                                 n_trials: int, seed: int, tau_range=None) -> int:
    """Operating tau chosen from the detection side: the SMALLEST tau that
    achieves >= target_pd against the true (on-grid) signal. A tighter tau
    gives better CFO resolution (narrower fired-set bands), so -- subject to
    meeting the detection target -- tighter is better. This is the dual of
    calibrate_tau (false-alarm side); Fig 5 explores the joint (tau, delta)
    trade-off explicitly.
    """
    rng = np.random.default_rng(seed)
    W = cb.W
    if tau_range is None:
        tau_range = range(int(0.05 * W), int(0.45 * W), max(1, W // 200))
    for tau in tau_range:
        pd = detection_prob_at_tau(cfg, cb, tau, ebn0, n_trials, rng)
        if pd >= target_pd:
            return tau
    return list(tau_range)[-1]


def calibrate_b_threshold(bank: bl.CorrelatorBank, cfg: Config, ebn0: float,
                           target_pd: float, n_trials: int, rng: np.random.Generator) -> float:
    """Correlator-bank analogue of calibrate_tau_for_detection: smallest
    magnitude threshold such that the nearest-hypothesis correlator peak
    exceeds it with probability >= target_pd, at true_df=0."""
    tx = make_tx(cfg)
    nearest = int(np.argmin(np.abs(bank.df_grid)))
    mags = np.empty(n_trials)
    for t in range(n_trials):
        r = run_channel(tx, 0.0, cfg.osr, ebn0, rng)
        mags[t] = bl.correlate_bank(r, bank)[nearest]
    # threshold = the value at the (1-target_pd) quantile from below
    return float(np.quantile(mags, 1 - target_pd))


# ---------------------------------------------------------------------------
# Fig 3: encoding comparison (thermometer vs Gray vs one-hot)
# ---------------------------------------------------------------------------

def encoding_comparison(ebn0_list, N: int, n_trials: int, seed: int,
                         osr: int = DEFAULT_OSR, n_sym: int = DEFAULT_N_SYM,
                         B: int = DEFAULT_B, diff_delay: int = DEFAULT_DIFF_DELAY,
                         force: bool = False):
    codings = ["thermometer", "gray", "onehot"]
    params = dict(ebn0_list=list(ebn0_list), N=N, n_trials=n_trials, seed=seed,
                  osr=osr, n_sym=n_sym, B=B, diff_delay=diff_delay, codings=codings,
                  kind="encoding_comparison")

    def compute():
        rms = np.zeros((len(codings), len(ebn0_list)))
        pdet = np.zeros((len(codings), len(ebn0_list)))
        widths = np.zeros(len(codings))
        for ci, coding in enumerate(codings):
            cfg = Config(osr=osr, n_sym=n_sym, B=B, diff_delay=diff_delay, coding=coding)
            widths[ci] = cfg.W()
            rng = np.random.default_rng(seed + ci)
            tx = make_tx(cfg)
            cb = make_codebook(cfg, N=N)
            tau = calibrate_tau_for_detection(cfg, cb, ebn0=10.0, target_pd=0.6,
                                               n_trials=300, seed=seed + ci)
            cam = CAM(cb.rows)
            for ei, ebn0 in enumerate(ebn0_list):
                errs = []
                hits = 0
                for _ in range(n_trials):
                    true_df = rng.uniform(cfg.df_min, cfg.df_max)
                    r = run_channel(tx, true_df, cfg.osr, ebn0, rng)
                    key = encode_waveform(r, cfg.B, coding=cfg.coding, diff_delay=cfg.diff_delay)
                    m = cam.query(key, tau)
                    res = dec.run_midpoint(m, cb.df_grid)
                    if res.detected:
                        hits += 1
                        errs.append(res.df_hat - true_df)
                pdet[ci, ei] = hits / n_trials
                rms[ci, ei] = np.sqrt(np.mean(np.square(errs))) if errs else np.nan
        return dict(codings=np.array(codings), ebn0_list=np.array(ebn0_list),
                    rms=rms, pdet=pdet, W=widths)

    return cached("fig3_encoding_comparison", params, compute, force=force)


# ---------------------------------------------------------------------------
# Fig 4: performance vs SNR -- CAM (run-midpoint), CAM (argmin upper bound),
# B1 (sequential), B2 (parallel)
# ---------------------------------------------------------------------------

def performance_vs_snr(cfg: Config, ebn0_list, N: int, n_trials: int, seed: int,
                        force: bool = False):
    params = dict(cfg=cfg.as_dict(), ebn0_list=list(ebn0_list), N=N,
                  n_trials=n_trials, seed=seed, kind="performance_vs_snr")

    def compute():
        rng = np.random.default_rng(seed)
        tx = make_tx(cfg)
        cb = make_codebook(cfg, N=N)
        cam = CAM(cb.rows)
        tau = calibrate_tau_for_detection(cfg, cb, ebn0=10.0, target_pd=0.7,
                                           n_trials=400, seed=seed)

        bank = bl.build_correlator_bank(N=N, df_min=cfg.df_min, df_max=cfg.df_max,
                                         n_sym=cfg.n_sym, osr=cfg.osr,
                                         access_address=cfg.access_address)
        threshold = calibrate_b_threshold(bank, cfg, ebn0=10.0, target_pd=0.7,
                                           n_trials=400, rng=np.random.default_rng(seed + 1))

        methods = ["cam_run_midpoint", "cam_argmin", "b1_sequential", "b2_parallel"]
        pdet = np.zeros((len(methods), len(ebn0_list)))
        rms = np.zeros((len(methods), len(ebn0_list)))
        run_len_mean = np.zeros(len(ebn0_list))

        for ei, ebn0 in enumerate(ebn0_list):
            errs = {m: [] for m in methods}
            hits = {m: 0 for m in methods}
            run_lens = []
            for _ in range(n_trials):
                true_df = rng.uniform(cfg.df_min, cfg.df_max)
                r = run_channel(tx, true_df, cfg.osr, ebn0, rng)
                key = encode_waveform(r, cfg.B, coding=cfg.coding, diff_delay=cfg.diff_delay)

                m_vec = cam.query(key, tau)
                res_rm = dec.run_midpoint(m_vec, cb.df_grid)
                if res_rm.detected:
                    hits["cam_run_midpoint"] += 1
                    errs["cam_run_midpoint"].append(res_rm.df_hat - true_df)
                    run_lens.append(res_rm.run_length)

                res_am = dec.argmin_baseline(cb.rows, key, cb.df_grid)
                hits["cam_argmin"] += 1
                errs["cam_argmin"].append(res_am.df_hat - true_df)

                mags = bl.correlate_bank(r, bank)
                k_hat = int(np.argmax(mags))
                b_detected = bool(mags[k_hat] >= threshold)
                for bm in ("b1_sequential", "b2_parallel"):
                    if b_detected:
                        hits[bm] += 1
                        errs[bm].append(bank.df_grid[k_hat] - true_df)

            for mi, m in enumerate(methods):
                pdet[mi, ei] = hits[m] / n_trials
                rms[mi, ei] = np.sqrt(np.mean(np.square(errs[m]))) if errs[m] else np.nan
            run_len_mean[ei] = np.mean(run_lens) if run_lens else np.nan

        return dict(methods=np.array(methods), ebn0_list=np.array(ebn0_list),
                    pdet=pdet, rms=rms, run_len_mean=run_len_mean,
                    tau=np.array(tau), threshold=np.array(threshold), W=np.array(cb.W))

    return cached("fig4_performance_vs_snr", params, compute, force=force)


def performance_vs_snr_full(cfg: Config, ebn0_list, N: int, n_trials: int, seed: int,
                             target_pfa: float = 0.01, force: bool = False):
    """Like performance_vs_snr, but additionally sweeps false-alarm probability
    per Eb/N0 (random valid-but-wrong access address, random CFO/theta0/timing)
    at the SAME calibrated tau/threshold used for detection -- for reporting
    a full misdetect/false-alarm/RMS-accuracy picture at a given operating
    point (e.g. a new (B, OSR) configuration), not just Pd and RMS.

    IMPORTANT: for the CAM, Pfa against a *different, valid* access address is
    NOT monotonically safe at low tau the way one might assume, and is NOT
    worst-case at low SNR either -- empirically it gets WORSE as SNR
    increases, because a clean (low-noise) wrong-AA key can sit structurally
    closer to a codebook row than noise would ever put it (cross-AA Hamming
    distance for this differential/thermometer encoding is measured at
    ~30-40% of W, not the ~50% a good discriminating code would give, mostly
    because all BLE packets share the same 8-bit preamble and the
    differential/Gaussian-filtered encoding has limited local diversity).
    So tau is calibrated here against Pfa at the WORST-case (highest) Eb/N0 in
    ebn0_list, not against a single mid-SNR detection-probability target --
    the latter (used by performance_vs_snr/Table I) does not actually bound
    high-SNR Pfa and was never checked against it before this function
    existed. B1/B2 do not show this problem (full complex correlation
    discriminates cross-AA cleanly; empirically Pfa~0 at every SNR tested),
    so their threshold keeps the original detection-side calibration.

    cam_argmin has no natural false-alarm concept (it always returns an
    estimate, never abstains) so its Pfa column is NaN.
    """
    params = dict(cfg=cfg.as_dict(), ebn0_list=list(ebn0_list), N=N,
                  n_trials=n_trials, seed=seed, target_pfa=target_pfa,
                  kind="performance_vs_snr_full_v2")

    def compute():
        rng = np.random.default_rng(seed)
        tx = make_tx(cfg)
        cb = make_codebook(cfg, N=N)
        cam = CAM(cb.rows)
        ebn0_worst = max(ebn0_list)
        tau = calibrate_tau(cfg, cb, ebn0=ebn0_worst, target_pfa=target_pfa,
                             n_trials=max(n_trials, 400), seed=seed)

        bank = bl.build_correlator_bank(N=N, df_min=cfg.df_min, df_max=cfg.df_max,
                                         n_sym=cfg.n_sym, osr=cfg.osr,
                                         access_address=cfg.access_address)
        threshold = calibrate_b_threshold(bank, cfg, ebn0=10.0, target_pd=0.7,
                                           n_trials=400, rng=np.random.default_rng(seed + 1))

        methods = ["cam_run_midpoint", "cam_argmin", "b1_sequential", "b2_parallel"]
        pdet = np.zeros((len(methods), len(ebn0_list)))
        rms = np.zeros((len(methods), len(ebn0_list)))
        pfa = np.full((len(methods), len(ebn0_list)), np.nan)
        run_len_mean = np.zeros(len(ebn0_list))

        for ei, ebn0 in enumerate(ebn0_list):
            errs = {m: [] for m in methods}
            hits = {m: 0 for m in methods}
            run_lens = []
            for _ in range(n_trials):
                true_df = rng.uniform(cfg.df_min, cfg.df_max)
                r = run_channel(tx, true_df, cfg.osr, ebn0, rng)
                key = encode_waveform(r, cfg.B, coding=cfg.coding, diff_delay=cfg.diff_delay)

                m_vec = cam.query(key, tau)
                res_rm = dec.run_midpoint(m_vec, cb.df_grid)
                if res_rm.detected:
                    hits["cam_run_midpoint"] += 1
                    errs["cam_run_midpoint"].append(res_rm.df_hat - true_df)
                    run_lens.append(res_rm.run_length)

                res_am = dec.argmin_baseline(cb.rows, key, cb.df_grid)
                hits["cam_argmin"] += 1
                errs["cam_argmin"].append(res_am.df_hat - true_df)

                mags = bl.correlate_bank(r, bank)
                k_hat = int(np.argmax(mags))
                b_detected = bool(mags[k_hat] >= threshold)
                for bm in ("b1_sequential", "b2_parallel"):
                    if b_detected:
                        hits[bm] += 1
                        errs[bm].append(bank.df_grid[k_hat] - true_df)

            fa_counts = {m: 0 for m in methods}
            for _ in range(n_trials):
                aa = random_valid_access_address(rng)
                tx_wrong = tx_waveform(n_sym=cfg.n_sym, osr=cfg.osr, access_address=aa)
                df_wrong = rng.uniform(cfg.df_min, cfg.df_max)
                r_wrong = run_channel(tx_wrong, df_wrong, cfg.osr, ebn0, rng)
                key_wrong = encode_waveform(r_wrong, cfg.B, coding=cfg.coding,
                                             diff_delay=cfg.diff_delay)
                if dec.any_fire(cam.query(key_wrong, tau)):
                    fa_counts["cam_run_midpoint"] += 1

                mags_wrong = bl.correlate_bank(r_wrong, bank)
                if bool(mags_wrong.max() >= threshold):
                    fa_counts["b1_sequential"] += 1
                    fa_counts["b2_parallel"] += 1

            for mi, m in enumerate(methods):
                pdet[mi, ei] = hits[m] / n_trials
                rms[mi, ei] = np.sqrt(np.mean(np.square(errs[m]))) if errs[m] else np.nan
                if m != "cam_argmin":
                    pfa[mi, ei] = fa_counts[m] / n_trials
            run_len_mean[ei] = np.mean(run_lens) if run_lens else np.nan

        return dict(methods=np.array(methods), ebn0_list=np.array(ebn0_list),
                    pdet=pdet, rms=rms, pfa=pfa, run_len_mean=run_len_mean,
                    tau=np.array(tau), threshold=np.array(threshold), W=np.array(cb.W))

    return cached("performance_vs_snr_full", params, compute, force=force)


# ---------------------------------------------------------------------------
# Fig 5: threshold (tau) and grid (delta) design surface
# ---------------------------------------------------------------------------

def threshold_grid_sweep(cfg: Config, tau_fracs, delta_fracs_of_wc, ebn0: float,
                          n_trials: int, seed: int, force: bool = False):
    """RMS CFO error and false-alarm rate over a (tau, delta) grid. delta is
    expressed as a fraction of an estimated capture-window half-width Wc so
    the sweep is meaningful across tau values (Wc itself depends on tau)."""
    params = dict(cfg=cfg.as_dict(), tau_fracs=list(tau_fracs),
                  delta_fracs_of_wc=list(delta_fracs_of_wc), ebn0=ebn0,
                  n_trials=n_trials, seed=seed, kind="threshold_grid_sweep")

    def compute():
        rng = np.random.default_rng(seed)
        tx = make_tx(cfg)
        W = cfg.W()
        rms_grid = np.full((len(tau_fracs), len(delta_fracs_of_wc)), np.nan)
        pfa_grid = np.full((len(tau_fracs), len(delta_fracs_of_wc)), np.nan)
        for ti, tau_frac in enumerate(tau_fracs):
            tau = int(tau_frac * W)
            # reference codebook (fine grid) to estimate Wc at this tau
            cb_ref = make_codebook(cfg, N=65)
            wc = capture_window_half_width(cb_ref.df_grid, cb_ref.rows, tau, len(cb_ref.df_grid) // 2)
            for di, dfrac in enumerate(delta_fracs_of_wc):
                delta = max(dfrac * max(wc, 1.0), (cfg.df_max - cfg.df_min) / 96)
                cb = make_codebook(cfg, delta=delta)
                if cb.N < 2:
                    continue
                cam = CAM(cb.rows)
                errs = []
                fa = 0
                for _ in range(n_trials):
                    true_df = rng.uniform(cfg.df_min, cfg.df_max)
                    r = run_channel(tx, true_df, cfg.osr, ebn0, rng)
                    key = encode_waveform(r, cfg.B, coding=cfg.coding, diff_delay=cfg.diff_delay)
                    m = cam.query(key, tau)
                    res = dec.run_midpoint(m, cb.df_grid)
                    if res.detected:
                        errs.append(res.df_hat - true_df)
                    aa = random_valid_access_address(rng)
                    tx_wrong = tx_waveform(n_sym=cfg.n_sym, osr=cfg.osr, access_address=aa)
                    r_wrong = run_channel(tx_wrong, rng.uniform(cfg.df_min, cfg.df_max),
                                           cfg.osr, ebn0, rng)
                    key_wrong = encode_waveform(r_wrong, cfg.B, coding=cfg.coding,
                                                 diff_delay=cfg.diff_delay)
                    if dec.any_fire(cam.query(key_wrong, tau)):
                        fa += 1
                rms_grid[ti, di] = np.sqrt(np.mean(np.square(errs))) if errs else np.nan
                pfa_grid[ti, di] = fa / n_trials
        return dict(tau_fracs=np.array(tau_fracs), delta_fracs_of_wc=np.array(delta_fracs_of_wc),
                    rms_grid=rms_grid, pfa_grid=pfa_grid, W=np.array(W))

    return cached("fig5_threshold_grid", params, compute, force=force)


# ---------------------------------------------------------------------------
# Table I: CAM vs B1 vs B2 at a fixed operating point
# ---------------------------------------------------------------------------

def table1_operating_point(cfg: Config, N: int, ebn0: float, n_trials: int, seed: int,
                            force: bool = False):
    params = dict(cfg=cfg.as_dict(), N=N, ebn0=ebn0, n_trials=n_trials, seed=seed,
                  kind="table1")

    def compute():
        rng = np.random.default_rng(seed)
        tx = make_tx(cfg)
        cb = make_codebook(cfg, N=N)
        cam = CAM(cb.rows)
        tau = calibrate_tau_for_detection(cfg, cb, ebn0=ebn0, target_pd=0.7,
                                           n_trials=400, seed=seed)
        bank = bl.build_correlator_bank(N=N, df_min=cfg.df_min, df_max=cfg.df_max,
                                         n_sym=cfg.n_sym, osr=cfg.osr,
                                         access_address=cfg.access_address)
        threshold = calibrate_b_threshold(bank, cfg, ebn0=ebn0, target_pd=0.7,
                                           n_trials=400, rng=np.random.default_rng(seed + 1))

        methods = ["CAM", "B1_sequential", "B2_parallel"]
        pdet = {m: 0 for m in methods}
        errs = {m: [] for m in methods}
        for _ in range(n_trials):
            true_df = rng.uniform(cfg.df_min, cfg.df_max)
            r = run_channel(tx, true_df, cfg.osr, ebn0, rng)
            key = encode_waveform(r, cfg.B, coding=cfg.coding, diff_delay=cfg.diff_delay)
            m_vec = cam.query(key, tau)
            res = dec.run_midpoint(m_vec, cb.df_grid)
            if res.detected:
                pdet["CAM"] += 1
                errs["CAM"].append(res.df_hat - true_df)

            mags = bl.correlate_bank(r, bank)
            k_hat = int(np.argmax(mags))
            b_detected = bool(mags[k_hat] >= threshold)
            for bm in ("B1_sequential", "B2_parallel"):
                if b_detected:
                    pdet[bm] += 1
                    errs[bm].append(bank.df_grid[k_hat] - true_df)

        W = cb.W
        N_s = bank.N_s
        latency = {"CAM": 1, "B1_sequential": N, "B2_parallel": 1}
        area = {"CAM": N * W, "B1_sequential": 1 * N_s, "B2_parallel": N * N_s}
        rows = []
        for m in methods:
            rms = float(np.sqrt(np.mean(np.square(errs[m])))) if errs[m] else float("nan")
            rows.append((m, latency[m], area[m], pdet[m] / n_trials, rms))
        return dict(
            methods=np.array([r[0] for r in rows]),
            latency=np.array([r[1] for r in rows]),
            area_proxy=np.array([r[2] for r in rows]),
            pdet=np.array([r[3] for r in rows]),
            rms=np.array([r[4] for r in rows]),
            tau=np.array(tau), threshold=np.array(threshold),
            W=np.array(W), N=np.array(N), resolution_hz=np.array(cb.delta),
        )

    return cached("table1", params, compute, force=force)
