#!/usr/bin/env python3
"""Reviewer-defence analysis on the FRESH-24 cells of llf_cd_confirm_v1, after the primary read (08403874) is filed (descriptive; no label, no new cell).

The analysis blocks are `llf_reviewer_defence.py`'s, unchanged and imported (tail p50/p95/p99 per task and per fan-out request, charged-decision share, free-decisions contrasts, mean
decomposition); only the cell loading is new, because the fresh 24 live in several directories. Statistic and runaway rule are the parent's (confirm_rulings.json): per topology the
median over g0, g1 of the paired %, then the median over the 24 topologies, faster count, exact Wilcoxon; "all" = every cell, "no_runaway" = a pair is dropped when either cell is a
runaway (> 10 x the arm's median in the rung, any seed flags the cell; whole-run flag re-used for every statistic).

Views of the learned arms (gnn, wide, mlp = the static e40 arms): seeds 1-3 (the primary's), seeds 1-6, seeds 4-6 alone. bar, random, reactive: s0. (b) re-encode gnn and its GPU twin: seed 1
(d_conf_reenc2). Cells absent (random's watchdog stalls) are left out of that arm's medians and their pairs are dropped; they are listed.

  llf_reviewer_fresh24.py --d-cd DIR --d-s456 DIR --react DIR --reenc DIR --inputs IN --ids ids.txt --rulings confirm_rulings.json --out-dir DIR
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import statistics as st
import sys
import types

import numpy as np
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llf_confirm_read as R  # noqa: E402
import llf_reviewer_defence as LD  # noqa: E402

IDS_SHA256 = "8d9d17a1b7e5c45ea4eb8fa99e11ddca46ff3f32248f7ce1e727d44424d1c563"
KINDS = {"gnn": R.ARMS["gnn"], "wide": R.ARMS["wide"], "mlp": R.ARMS["mlp"], "bar": R.BAR, "random": "random", "reactive": "reactive",
         "re_gnn": "ra_gnn_eng_combfl_linkconv_nosplit_k100", "re_twin": "ra_twin_eng_combfl_link_gpu_nosplit_k100"}
PIN = {"gnn": "02668ecf", "wide": "02668ecf", "mlp": "02668ecf", "bar": "0ec7cdcb", "random": "0ec7cdcb", "reactive": "0ec7cdcb", "re_gnn": "0ec7cdcb", "re_twin": "0ec7cdcb"}
S0 = ("bar", "random", "reactive")
ONE = ("re_gnn", "re_twin")
LEARNED = ("gnn", "wide", "mlp")
VIEWS = {"s123": (1, 2, 3), "s1-6": (1, 2, 3, 4, 5, 6), "s456": (4, 5, 6)}
STATS = ("whole", "a360")
PAIRS_BASE = (("gnn", "bar"), ("gnn", "wide"), ("gnn", "mlp"), ("gnn", "random"), ("gnn", "reactive"), ("bar", "reactive"))
PAIRS_RE = (("re_gnn", "gnn"), ("re_gnn", "re_twin"), ("re_gnn", "bar"), ("re_gnn", "wide"))
SHIM = types.SimpleNamespace(ARMS=KINDS, RUNGS=R.RUNGS, WINDOWS=R.WINDOWS, Grid=R.Grid)
LD.R = SHIM


class Grid(R.Grid):
    view: Tuple[int, ...] = (1, 2, 3)

    def seeds(self, arm):
        return (0,) if arm in S0 else (1,) if arm in ONE else self.view

    def raw(self, arm, seed, t, w, rung, stat):
        return self.V.get((self.arms[arm], seed, t, w, rung), {}).get(stat, float("nan"))

    def arm_median(self, arm, seed, rung, mstat):
        key = (arm, rung, mstat)
        if key not in self.med:
            vals = [self.raw(arm, s, t, w, rung, mstat) for s in self.seeds(arm) for t in self.topos for w in R.WINDOWS]
            self.med[key] = st.median(v for v in vals if v == v)
        return self.med[key]

    def cell(self, arm, t, w, rung, stat, reading, only_seed=None):
        ss = self.seeds(arm) if only_seed is None else (only_seed,)
        if any(self.raw(arm, s, t, w, rung, stat) != self.raw(arm, s, t, w, rung, stat) for s in ss):
            return float("nan"), True
        return super().cell(arm, t, w, rung, stat, reading, only_seed)

    def runaways(self):
        out = []
        for a in self.arms:
            for s in self.seeds(a):
                for rung in R.RUNGS:
                    for t in self.topos:
                        for w in R.WINDOWS:
                            if self.raw(a, s, t, w, rung, "whole") != self.raw(a, s, t, w, rung, "whole"):
                                continue
                            for stat in STATS:
                                if self.seed_flagged(a, s, t, w, rung, stat):
                                    out.append({"arm": a, "seed": s, "topology": t, "window": w, "rung": rung, "stat": stat})
        return out


def load(dirs: Dict[str, str], ids: Tuple[int, ...]):
    """V by (kind, seed, topology, window, rung), the file stems, and the cells absent from the expected grid (never an unexpected or duplicated one)."""
    allowed_seeds = {a: ((0,) if a in S0 else (1,) if a in ONE else (1, 2, 3, 4, 5, 6)) for a in KINDS}
    by_kind = {v: a for a, v in KINDS.items()}
    V, stems, links, commits = {}, {}, {}, set()
    for name, root in dirs.items():
        for f in sorted(glob.glob(os.path.join(root, "**", "cc40s*.summary.json"), recursive=True)):
            m = R.NAME.match(os.path.basename(f))
            if not m or m.group(4) not in by_kind or m.group(2) not in R.WINDOWS:
                continue
            arm = by_kind[m.group(4)]
            key = (m.group(4), int(m.group(5)), int(m.group(1)), m.group(2), m.group(3))
            if key in V:
                raise SystemExit(f"FAIL LOUD: {key} appears twice ({f})")
            if key[1] not in allowed_seeds[arm] or key[2] not in ids:
                raise SystemExit(f"FAIL LOUD: unexpected cell {key} ({f})")
            s = json.load(open(f))
            code = s.get("code") or {}
            if code.get("dirty") is not False or not str(code.get("commit", "")).startswith(PIN[arm]) or s.get("servers") != 24:
                raise SystemExit(f"FAIL LOUD: {f} ran at {code.get('commit')} dirty={code.get('dirty')} servers={s.get('servers')}, not {PIN[arm]} clean at 24 servers")
            a360 = ((s.get("steady_state") or {}).get("after_360s") or {}).get("mean")
            if s.get("averageElapsedTime") is None or a360 is None:
                raise SystemExit(f"FAIL LOUD: {f} has no averageElapsedTime / after_360s.mean")
            V[key] = {"whole": float(s["averageElapsedTime"]), "a360": float(a360)}
            stems[key] = f[:-len(".summary.json")]
            fs, ov = s.get("peerExchangeFairshare") or {}, s.get("peerExchangeOverlap") or {}
            links[key] = {"transfers": fs.get("transfers"), "flow_seconds": fs.get("flow_seconds"), "peak_flows": fs.get("peak_flows_on_a_link"),
                          "extra_seconds": fs.get("extra_seconds"), "overlap_share": ov.get("overlap_seconds_share")}
            commits.add(code["commit"])
    missing = {}
    for a, kind in KINDS.items():
        for s in allowed_seeds[a]:
            for t in ids:
                for w in R.WINDOWS:
                    for r in R.RUNGS:
                        if (kind, s, t, w, r) not in V:
                            missing.setdefault(a, []).append([s, t, w, r])
    return V, stems, links, sorted(commits), missing


class _NP:
    """numpy for llf_reviewer_defence.cell_extras, except that a `tasks_bsz.npy` whose group-size column is -1 (the arms that do not batch by peer group: reactive, random) gets the
    workload's own group size there, so the script's other checks (the id permutation, mean latency) still run for them."""
    def __init__(self, wl, rung, w):
        self.wl, self.rung, self.w = wl, rung, w

    def __getattr__(self, name):
        return getattr(np, name)

    def load(self, path, *a, **k):
        arr = np.load(path, *a, **k)
        if str(path).endswith("tasks_bsz.npy") and (arr[:, 3] < 0).all():
            _ts, inv, cnt = self.wl.get(self.rung, self.w)
            arr = arr.copy()
            arr[:, 3] = cnt[inv[arr[:, 0].astype(int)]]
        return arr


def augment(V, stems, inputs):
    wl = LD.Workloads(inputs)
    D = {}
    for key, stem in stems.items():
        s = json.load(open(stem + ".summary.json"))
        LD.np = _NP(wl, key[4], key[3])
        stats, dec = LD.cell_extras(stem, s, wl, key[4], key[3])
        V[key].update(stats)
        D[key] = dec
    return D


def link_block(links, ids) -> Dict[str, Any]:
    """transfers, seconds per transfer, peak flows on a link, overlap share: median [min, max] over the cells of each arm and rung (g0+g1; all seeds of the arm)."""
    out = {}
    by_kind = {v: a for a, v in KINDS.items()}
    for arm in KINDS:
        for rung in R.RUNGS:
            rows = [v for k, v in links.items() if by_kind[k[0]] == arm and k[4] == rung and v["transfers"]]
            if not rows:
                continue
            q = lambda xs: {"median": st.median(xs), "min": min(xs), "max": max(xs)}
            out[f"{arm}|{rung}"] = {"cells": len(rows), "transfers": q([r["transfers"] for r in rows]), "seconds_per_transfer": q([r["flow_seconds"] / r["transfers"] for r in rows]),
                                    "peak_flows_on_a_link": q([r["peak_flows"] for r in rows]), "overlap_share": q([r["overlap_share"] for r in rows]),
                                    "extra_over_flow_seconds": q([r["extra_seconds"] / r["flow_seconds"] for r in rows])}
    return out


def blocks(V, D, rul, ids, view, pairs, show) -> Dict[str, Any]:
    g = Grid(V, ids, rul, KINDS)
    g.view = view
    LD.SHOW, LD.PAIRS = show, pairs
    return {"tail": LD.tail_block(g), "free_decisions": LD.free_block(g), "decision_share": LD.share_block(V, D, g), "decomposition": LD.decomposition_block(D, g), "grid": g}


def runaway_counts(g: Grid) -> Dict[str, Any]:
    ra = g.runaways()
    out = {}
    for a in KINDS:
        for rung in R.RUNGS:
            tot = len(g.seeds(a)) * len(g.topos) * len(R.WINDOWS)
            absent = sum(1 for s in g.seeds(a) for t in g.topos for w in R.WINDOWS if g.raw(a, s, t, w, rung, "whole") != g.raw(a, s, t, w, rung, "whole"))
            row = {"seed_cells": tot, "absent": absent}
            for stat in STATS:
                hit = [r for r in ra if r["arm"] == a and r["rung"] == rung and r["stat"] == stat]
                row[f"flagged_{stat}"] = len(hit)
                row[f"cells_flagged_{stat}_(topology,window)"] = len({(r["topology"], r["window"]) for r in hit})
            out[f"{a}|{rung}"] = row
    return out


def re_rows(V, rul, ids) -> Dict[str, Any]:
    out = {}
    for view_name, view in VIEWS.items():
        g = Grid(V, ids, rul, KINDS)
        g.view = view
        for a, b in PAIRS_RE:
            for mode in ("seed1", "pooled"):
                if mode == "pooled" and b not in LEARNED:
                    continue
                for stat in ("whole", "a360", "t_p50", "t_p99", "r_p50", "r_p99"):
                    for rung in R.RUNGS:
                        for reading in ("all", "no_runaway"):
                            c = g.contrast(a, b, rung, stat, reading, seed=1 if mode == "seed1" else None)
                            out[f"{view_name}|{a}_vs_{b}|{mode}|{stat}|{rung}|{reading}"] = LD.clean(c)
    return {k: v for k, v in out.items()}


def exchange_block(D, ids) -> Dict[str, Any]:
    """Peer-exchange seconds per task (the decomposition's `exchange`), median [min, max] over the arm's cells in the rung, g0+g1, all its seeds."""
    by_kind = {v: a for a, v in KINDS.items()}
    out = {}
    for arm in KINDS:
        for rung in R.RUNGS:
            xs = [d["exchange"] for k, d in D.items() if by_kind[k[0]] == arm and k[4] == rung]
            if xs:
                out[f"{arm}|{rung}"] = {"cells": len(xs), "median": st.median(xs), "min": min(xs), "max": max(xs), "mean_latency_median": st.median(d["mean"] for k, d in D.items() if by_kind[k[0]] == arm and k[4] == rung)}
    return out


def summary(res: Dict[str, Any], ids) -> str:
    f = lambda x: "n/a" if x is None else f"{x:+.2f}"
    line = lambda d: f"{f(d['median_pct'])} % ({d['faster']}/{d['n']}, p {('%.3g' % d['p']) if d['p'] is not None else 'n/a'})"
    out = [f"fresh-24 reviewer-defence, g0+g1, {len(ids)} topologies, per-topology median over windows then median over topologies, exact Wilcoxon; 'no runaway' drops pairs with a flagged cell; descriptive"]
    for vn in VIEWS:
        v = res["views"][vn]
        for rung in R.RUNGS:
            F, T = v["free_decisions"], v["tail"]
            out.append(f"[{vn}] {rung}: mean gnn vs bar {line(F['whole|gnn_vs_bar|%s|no_runaway' % rung])} no-runaway / {line(F['whole|gnn_vs_bar|%s|all' % rung])} all; vs wide {line(F['whole|gnn_vs_wide|%s|no_runaway' % rung])} / {line(F['whole|gnn_vs_wide|%s|all' % rung])}"
                       f"; free decisions vs bar {line(F['whole_free|gnn_vs_bar|%s|no_runaway' % rung])}, vs wide {line(F['whole_free|gnn_vs_wide|%s|no_runaway' % rung])}")
            out.append(f"[{vn}] {rung}: task p50/p99 gnn vs bar {line(T['task|p50|gnn_vs_bar|%s|no_runaway' % rung])} / {line(T['task|p99|gnn_vs_bar|%s|no_runaway' % rung])}; vs wide {line(T['task|p50|gnn_vs_wide|%s|no_runaway' % rung])} / {line(T['task|p99|gnn_vs_wide|%s|no_runaway' % rung])}"
                       f"; request p50/p99 vs bar {line(T['request_fanout|p50|gnn_vs_bar|%s|no_runaway' % rung])} / {line(T['request_fanout|p99|gnn_vs_bar|%s|no_runaway' % rung])}; vs wide {line(T['request_fanout|p50|gnn_vs_wide|%s|no_runaway' % rung])} / {line(T['request_fanout|p99|gnn_vs_wide|%s|no_runaway' % rung])}")
    S = res["views"]["s123"]["decision_share"]
    for rung in R.RUNGS:
        out.append(f"[s123] {rung}: decision charge, median-cell share of mean latency (felt ms/task): " + ", ".join(f"{a} {100 * S[a + '|' + rung]['share_median_cell']:.2f} % ({1000 * S[a + '|' + rung]['felt_s_per_task_mean']:.0f})" for a in ("gnn", "wide", "mlp", "bar")))
    out.append("runaway seed-cells (whole) flagged / seed-cells: " + "; ".join(f"{k} {v['flagged_whole']}/{v['seed_cells']}" for k, v in res["runaway_counts"]["s1-6"].items() if v["flagged_whole"]))
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("d-cd", "d-s456", "react", "reenc", "inputs", "ids", "rulings", "out-dir"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    ids = tuple(int(x) for x in open(a.ids).read().split())
    sha = hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()
    if sha != IDS_SHA256 or len(set(ids)) != len(ids):
        raise SystemExit("FAIL LOUD: the id list is not the filed fresh-24 one")
    rul = json.load(open(a.rulings))
    R.check_rulings_cd(rul)
    V, stems, links, commits, missing = load({"d_cd": getattr(a, "d_cd"), "d_s456": a.d_s456, "react": a.react, "reenc": a.reenc}, ids)
    for arm in KINDS:
        if arm != "random" and arm in missing:
            raise SystemExit(f"FAIL LOUD: {arm} cells missing, first {missing[arm][:3]}")
    D = augment(V, stems, a.inputs)
    views = {}
    for vn, view in VIEWS.items():
        b = blocks(V, D, rul, ids, view, PAIRS_BASE, ("gnn", "wide", "mlp", "bar", "reactive"))
        g = b.pop("grid")
        views[vn] = {**b, "runaway_counts": runaway_counts(g)}
    reb = blocks(V, D, rul, ids, (1, 2, 3), PAIRS_RE, ("gnn", "wide", "bar", "re_gnn", "re_twin"))
    reb.pop("grid")
    res = {"family": "llf_cd_confirm_v1 reviewer-defence analysis on the fresh-24 (descriptive, no label)", "topologies": list(ids), "ids_sha256": sha, "windows": list(R.WINDOWS), "commits": commits,
           "rulings": rul, "arms": KINDS, "views": views, "runaway_counts": {vn: v["runaway_counts"] for vn, v in views.items()}, "reenc": {"contrasts": re_rows(V, rul, ids), **reb},
           "link_load": link_block(links, ids), "exchange_per_task": exchange_block(D, ids), "missing_cells_left_out": missing, "cells_read": len(stems),
           "denominators": "learned arm heavy seed-cells = 24 topologies x 2 windows x seeds = 144 at seeds 1-3, 144 at seeds 4-6, 288 at seeds 1-6; moderate the same",
           "scripts_sha256": {os.path.basename(p): hashlib.sha256(open(p, "rb").read()).hexdigest() for p in (os.path.abspath(__file__), LD.__file__)}}
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, "reviewer_defence.json")
    with open(out + ".tmp", "w") as fh:
        json.dump(res, fh, indent=1, default=str)
    os.replace(out + ".tmp", out)
    txt = summary(res, ids)
    open(os.path.join(a.out_dir, "summary.txt"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
