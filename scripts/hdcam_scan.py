#!/usr/bin/env python3
"""Run the simplest HDCAM acquisition model over a gps-sdr-sim scenario.

Reads a scenario pair produced by gps-sdr-sim (see notebooks/09_benchmark.ipynb):

    <name>.bin   int16 interleaved I/Q at 2.046 MHz
    <name>.csv   per-block truth: block, gps_sec, prn, f_carr_hz,
                 f_code_chips_s, code_phase_chip, gain

and reports, for every 1 ms dwell, which satellites the CAM declared.

The model is deliberately the plain one: sign-quantize the samples, compare
against a (PRN x Doppler) codebook of full-scale replicas, and read out a single
boolean per cell -- "did the Hamming distance fall at or below tau". No ranking,
no argmin: the Hamming distance is local to fires() and never leaves it, exactly
as in sim/cam.py.

Examples
--------
    python scripts/hdcam_scan.py notebooks/_scen/s00.bin notebooks/_scen/s00.csv
    python scripts/hdcam_scan.py notebooks/_scen/s00.bin notebooks/_scen/s00.csv \
        --dwells 20 --cn0 42 --verbose
    python scripts/hdcam_scan.py notebooks/_scen/s03.bin notebooks/_scen/s03.csv \
        --accumulate 4 --k 3
"""
from __future__ import annotations

import argparse
import csv as csvmod
import sys
import time
from collections import defaultdict
from functools import lru_cache

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# ----------------------------------------------------------------- constants
FS = 2.046e6                 # sampling rate, 2 samples per chip
NS = 2046                    # samples in one 1 ms code period
CL = 1023                    # chips per C/A code period
L1 = 1575.42e6               # L1 carrier
DWELL = 2 * NS               # 2 ms buffer: 1 ms of key + 1 ms of sliding window
BLK = int(0.1 * FS)          # gps-sdr-sim rewrites its truth every 0.1 s
GRID = np.arange(-20, 21) * 500.0        # +-10 kHz in 500 Hz bins
PRNS = list(range(1, 33))

HIT_CP, HIT_FD = 1.0, 500.0    # a declaration is a hit within this of a real satellite
HALO_CP, HALO_FD = 2.0, 2000.0  # ...and a "halo" (skirt of a real one) within this

# IS-GPS-200 Table 3-Ia: the G2 phase-selector taps per PRN.
G2 = {1: (2, 6), 2: (3, 7), 3: (4, 8), 4: (5, 9), 5: (1, 9), 6: (2, 10),
      7: (1, 8), 8: (2, 9), 9: (3, 10), 10: (2, 3), 11: (3, 4), 12: (5, 6),
      13: (6, 7), 14: (7, 8), 15: (8, 9), 16: (9, 10), 17: (1, 4), 18: (2, 5),
      19: (3, 6), 20: (4, 7), 21: (5, 8), 22: (6, 9), 23: (1, 3), 24: (4, 6),
      25: (5, 7), 26: (6, 8), 27: (7, 9), 28: (8, 10), 29: (1, 6), 30: (2, 7),
      31: (3, 8), 32: (4, 9)}


# ------------------------------------------------------------------ the model
@lru_cache(None)
def ca_code(prn: int) -> np.ndarray:
    """The satellite's 1023-chip Gold code as +1/-1."""
    t1, t2 = G2[prn]
    g1 = np.ones(10, np.uint8)
    g2 = np.ones(10, np.uint8)
    out = np.empty(CL, np.uint8)
    for i in range(CL):
        out[i] = g1[9] ^ g2[t1 - 1] ^ g2[t2 - 1]
        f1 = g1[2] ^ g1[9]
        f2 = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]
        g1[1:], g1[0] = g1[:9], f1
        g2[1:], g2[0] = g2[:9], f2
    return 1.0 - 2.0 * out


def build_codebook(prns, grid):
    """One row per (PRN, Doppler): the noiseless replica, sign-quantized.

    Rows sit at theta = 45 degrees so both rails carry the code; at theta = 0 the
    replica is purely real and half the match line would be dead.
    """
    rows, tags = [], []
    for p in prns:
        code = ca_code(p)
        for f in grid:
            idx = (1.023e6 / FS) * (1 + f / L1) * np.arange(NS)
            chips = code[np.floor(idx).astype(np.int64) % CL]
            x = chips * np.exp(1j * (2 * np.pi * f * np.arange(NS) / FS + np.pi / 4))
            rows.append(np.stack([x.real < 0, x.imag < 0], -1).ravel())
            tags.append((p, f))
    return np.array(rows), tags


class Cam:
    """The array: rows of bits, and a single boolean readout per cell."""

    def __init__(self, prns=PRNS, grid=GRID):
        self.rows, tags = build_codebook(prns, grid)
        self.rpm = (1.0 - 2.0 * self.rows).astype(np.float32)
        self.tag_prn = np.array([t[0] for t in tags])
        self.tag_fd = np.array([t[1] for t in tags])
        self.W = self.rows.shape[1]
        self.shifts = np.arange(NS)          # half-chip code phases
        self.cps = self.shifts / 2.0         # ...in chips
        self.n_cells = len(self.shifts) * self.rows.shape[0]

    def fires(self, dwell: np.ndarray, tau: int) -> np.ndarray:
        """Which cells matched within tau. Booleans out.

        The per-cell Hamming distance is computed here -- as the sense amplifier
        does -- and never leaves this function, so nothing downstream can rank
        cells. Rotating 180 degrees complements every bit, so d_180 = W - d and
        two rotations cover all four.
        """
        b = np.stack([dwell.real < 0, dwell.imag < 0], -1)
        out = np.zeros((len(self.shifts), self.rows.shape[0]), bool)
        for q in (0, 1):
            v = b if q == 0 else np.stack([~b[:, 1], b[:, 0]], -1)
            K = sliding_window_view(v.reshape(-1), self.W)[::2][: len(self.shifts)]
            d = (self.W - (1.0 - 2.0 * K.astype(np.float32)) @ self.rpm.T) / 2
            out |= d <= tau              # rotation q
            out |= self.W - d <= tau     # rotation q + 2
        return out

    def declare(self, fired: np.ndarray):
        """Fired cells as (prn, code_phase_chips, doppler_hz) triples."""
        return [(int(self.tag_prn[b]), float(self.cps[a]), float(self.tag_fd[b]))
                for a, b in np.argwhere(fired)]


# ------------------------------------------------------------------- scenario
def load_scenario(bin_path: str, csv_path: str):
    """gps-sdr-sim output: int16 interleaved I/Q, plus its per-block truth dump."""
    raw = np.fromfile(bin_path, dtype=np.int16)
    if raw.size % 2:
        raise ValueError(f"{bin_path}: odd int16 count, not interleaved I/Q?")
    x = raw[0::2].astype(np.float64) + 1j * raw[1::2].astype(np.float64)

    truth, amp = defaultdict(list), {}
    with open(csv_path, newline="") as fh:
        for r in csvmod.DictReader(fh):
            prn = int(r["prn"])
            truth[int(r["block"])].append({
                "prn": prn,
                "fd": float(r["f_carr_hz"]),
                "fcode": float(r["f_code_chips_s"]),
                "cp_g": float(r["code_phase_chip"]),
            })
            amp[prn] = max(amp.get(prn, 0.0), float(r["gain"]))
    if not truth:
        raise ValueError(f"{csv_path}: no truth rows")
    return {"x": x, "truth": dict(truth), "amp": amp}


def sats_at(scen, offset: int):
    """Truth for the dwell starting at `offset` samples, with code phase drifted.

    gps-sdr-sim states its truth once per 0.1 s block; within a block the code
    phase advances at f_code chips/second, which is where code Doppler lives.
    """
    blk = offset // BLK + 1
    if blk not in scen["truth"]:
        blk = max(b for b in scen["truth"] if b <= blk)
    dt = (offset - (blk - 1) * BLK) / FS
    out = []
    for t in scen["truth"][blk]:
        cp_g = (t["cp_g"] + t["fcode"] * dt) % CL
        out.append({"prn": t["prn"], "fd": t["fd"],
                    "cp": (CL - cp_g) % CL})     # our convention: cp = CL - gps cp
    return out


def sigma_for(scen, cn0_strongest: float) -> float:
    """Noise sigma placing the scenario's strongest satellite at a given C/N0."""
    return float(np.sqrt(max(scen["amp"].values()) ** 2 * FS
                         / (2 * 10 ** (cn0_strongest / 10))))


def cn0_of(scen, prn: int, sigma: float) -> float:
    return float(10 * np.log10(scen["amp"][prn] ** 2 / (2 * sigma ** 2) * FS))


def classify(dec, sats):
    """hit = on a real satellite; halo = its correlation skirt; false = spurious.

    Separating halo from false matters: a cell one Doppler bin out on a satellite
    that IS there is the detection leaking through the sinc skirt, not an error.
    A receiver hands the whole cluster to tracking and loses nothing.
    """
    prn, cp, fd = dec
    for tol_cp, tol_fd, kind in ((HIT_CP, HIT_FD, "hit"), (HALO_CP, HALO_FD, "halo")):
        for s in sats:
            if s["prn"] != prn:
                continue
            e = abs(cp - s["cp"])
            if min(e, CL - e) <= tol_cp and abs(fd - s["fd"]) <= tol_fd:
                return kind, s
    return "false", None


# ----------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bin_file", help="gps-sdr-sim .bin (int16 I/Q at 2.046 MHz)")
    ap.add_argument("csv_file", help="matching truth .csv")
    ap.add_argument("--dwells", type=int, default=10, help="detection steps to run; each consumes --accumulate dwells (default 10)")
    ap.add_argument("--start", type=int, default=0, help="first dwell index (default 0)")
    ap.add_argument("--cn0", type=float, default=45.0,
                    help="C/N0 in dB-Hz of the STRONGEST satellite; sets the noise "
                         "level, since gps-sdr-sim output carries no thermal noise "
                         "(default 45)")
    ap.add_argument("--tau", type=int, default=1870,
                    help="match threshold in bits, out of 4092 (default 1870)")
    ap.add_argument("--accumulate", type=int, default=1, metavar="M",
                    help="declare on M dwells accumulated rather than 1 (default 1)")
    ap.add_argument("--k", type=int, default=1,
                    help="fires needed out of M to declare (default 1)")
    ap.add_argument("--prns", type=str, default="",
                    help="restrict the codebook, e.g. 5,10,13 (default: all 32)")
    ap.add_argument("--seed", type=int, default=0, help="noise seed (default 0)")
    ap.add_argument("--verbose", action="store_true",
                    help="list every declared cell, not just the PRNs")
    a = ap.parse_args(argv)

    if a.k > a.accumulate:
        ap.error(f"--k {a.k} exceeds --accumulate {a.accumulate}")

    prns = [int(p) for p in a.prns.split(",")] if a.prns else PRNS

    t0 = time.time()
    try:
        scen = load_scenario(a.bin_file, a.csv_file)
    except FileNotFoundError as e:
        print(f"error: {e.filename} not found", file=sys.stderr)
        print("scenario pairs live in notebooks/_scen/ as sNN.bin + sNN.csv; "
              "notebooks/09_benchmark.ipynb generates them.", file=sys.stderr)
        return 1
    cam = Cam(prns, GRID)
    rng = np.random.default_rng(a.seed)
    sigma = sigma_for(scen, a.cn0)

    total = scen["x"].size // DWELL
    n = min(a.dwells, total - a.start)
    if n <= 0:
        print(f"nothing to scan: file holds {total} dwells, --start {a.start}")
        return 1

    print(f"file      : {a.bin_file}  ({scen['x'].size/FS:.2f} s, {total} dwells of 2 ms)")
    print(f"codebook  : {cam.rows.shape[0]} rows x {cam.W} bits "
          f"({len(prns)} PRN x {len(GRID)} Doppler), {cam.n_cells:,} cells")
    print(f"noise     : strongest satellite at {a.cn0:.0f} dB-Hz -> sigma {sigma:.0f}")
    print(f"detector  : tau {a.tau}, {a.k} of {a.accumulate} dwell(s)")
    print(f"setup     : {time.time()-t0:.1f}s\n")

    sats0 = sats_at(scen, a.start * DWELL)
    print(f"satellites present ({len(sats0)}):")
    for s in sorted(sats0, key=lambda s: -cn0_of(scen, s["prn"], sigma)):
        print(f"   PRN {s['prn']:2d}   {cn0_of(scen, s['prn'], sigma):5.1f} dB-Hz   "
              f"fd {s['fd']:+8.1f} Hz   cp {s['cp']:7.2f} chips")
    print()

    hdr = (f"{'dwell':>5} {'t (ms)':>8}  {'found':<34} {'missed':<28} "
           f"{'halo':>5} {'false':>6}")
    print(hdr)
    print("-" * len(hdr))

    seen, present_all = set(), set()
    n_false_tot = n_halo_tot = n_dec_tot = 0
    for i in range(n):
        d0 = a.start + i * a.accumulate
        if (d0 + a.accumulate) * DWELL > scen["x"].size:
            break
        acc = np.zeros((len(cam.shifts), cam.rows.shape[0]), np.int16)
        for m in range(a.accumulate):
            off = (d0 + m) * DWELL
            seg = scen["x"][off:off + DWELL].copy()
            seg += rng.normal(0, sigma, DWELL) + 1j * rng.normal(0, sigma, DWELL)
            acc += cam.fires(seg, a.tau)

        off = d0 * DWELL
        sats = sats_at(scen, off)
        present = {s["prn"] for s in sats}
        present_all |= present
        decs = cam.declare(acc >= a.k)
        n_dec_tot += len(decs)

        found, n_halo, n_false = set(), 0, 0
        for dec in decs:
            kind, s = classify(dec, sats)
            if kind == "hit":
                found.add(s["prn"])
            elif kind == "halo":
                n_halo += 1
            else:
                n_false += 1
        seen |= found
        n_false_tot += n_false
        n_halo_tot += n_halo

        fs = " ".join(f"{p:2d}" for p in sorted(found)) or "-"
        ms = " ".join(f"{p:2d}" for p in sorted(present - found)) or "-"
        print(f"{d0:>5} {off/FS*1e3:>8.1f}  {fs:<34} {ms:<28} {n_halo:>5} {n_false:>6}")

        if a.verbose:
            for dec in sorted(decs):
                kind, s = classify(dec, sats)
                mark = f"{kind} PRN {s['prn']}" if s else "FALSE"
                print(f"         PRN {dec[0]:2d}  cp {dec[1]:7.1f} chips  "
                      f"fd {dec[2]:+7.0f} Hz   [{mark}]")

    print("-" * len(hdr))
    print(f"\nscanned {n} step(s) of {a.accumulate} dwell(s) in {time.time()-t0:.1f}s")
    print(f"satellites present    : {len(present_all)}  ({' '.join(str(p) for p in sorted(present_all))})")
    print(f"acquired at least once: {len(seen)}  ({' '.join(str(p) for p in sorted(seen)) or '-'})")
    print(f"never acquired        : {' '.join(str(p) for p in sorted(present_all - seen)) or '-'}")
    print(f"declarations          : {n_dec_tot}  "
          f"({n_dec_tot - n_halo_tot - n_false_tot} on target, {n_halo_tot} halo, "
          f"{n_false_tot} false)")
    print(f"false alarms per step : {n_false_tot/max(n,1):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
