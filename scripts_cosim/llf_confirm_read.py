#!/usr/bin/env python3
"""live_link_features_v1 CONFIRMATION reader (S5 draft; read once, then frozen by blob by SCIENTIST). Strictly the node's registered text
(docs/lineages/live_link_features_v1.md, "Confirmation (read once)"): option D, the sealed 12 ids, g0 + g1, both rungs, seeds 1-3.

Statistic: the paired % per cell, A vs B at the same topology, window and rung = 100 (A / B - 1) on the task mean; a learned cell is the mean over seeds of the
per-seed cell value; per topology the median over the 2 windows; then the median over the topologies, topologies faster (< 0) and the exact two-sided Wilcoxon
over topologies (zeros dropped; r1_attribution_v1_read.exact_wilcoxon). A cell is a runaway if it is > 10x its arm's median in that rung; every contrast is read
with all cells and without runaways (dropping a pair when either cell is flagged). Primary: v7 gnn vs v7 twin on 2 tests (moderate, heavy), Holm over the 2, labelled
TWIN-FASTER > MP-ATTRIBUTED > MP-DIRECTION > TIE by the first that holds; the label is the weaker of the two readings. Secondary: descriptive, raw p, both rungs, both readings,
the after-360 s row beside the whole run.

Where the node text leaves a point open this reader does NOT choose: every such point is a key of the --rulings JSON (written by SCIENTIST) and the reader refuses to run
without all of them. The points are listed in RULING_POINTS below.

  llf_confirm_read.py --cells <dir with the cell subdirs> --rulings rulings.json --pin <code.commit prefix | static=P,refs=P,bar=P> --out read.json
  llf_confirm_read.py --mode cd --cells <root>[,<root>...] --react-cells <flat reactive dir> --ids <ids> --ids-sha256 <sha> --rulings rulings.json --pin static=P,bar=P --out read.json
    (llf_cd_confirm_v1: learned arms at seeds 1-6, amendment A2)

The cell subdirectories (any names) hold cc40s<topology>__<window><rung>__<kind>_s<seed>.summary.json. A missing, duplicated, unexpected, dirty or wrong-pin cell fails loud.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r1_attribution_v1_read import exact_wilcoxon, holm  # noqa: E402

IDS = (30285, 30286, 30287, 30289, 30290, 30291, 30292, 30293, 30294, 30295, 30296, 30297)
IDS_SHA256 = "f165536615a17be19f9bb5a01736dacb7f30255f85370d32846b3b177dbf179a"
WINDOWS, RUNGS, SEEDS = ("g0", "g1"), ("moderate", "heavy"), (1, 2, 3)
BAR = "cd_ib0_inflightfull_k100"
ARMS = {  # name -> gate kind (the static variant's e40 best-val arms, and the v6 fair-share-label references)
    "gnn": "ra_gnn_eng_combfl_linkstatic_e40_nosplit_k100", "twin": "ra_twin_eng_combfl_linkstatic_e40_nosplit_k100",
    "mlp": "ra_mlp_same_combfl_linkstatic_e40_nosplit_k100", "wide": "ra_mlp_wide_combfl_linkstatic_e40_nosplit_k100",
    "fs_gnn": "ra_gnn_eng_fslabel_combfl_nosplit_k100", "fs_twin": "ra_twin_eng_fslabel_combfl_nosplit_k100",
    "fs_mlp": "ra_mlp_same_fslabel_combfl_nosplit_k100", "fs_wide": "ra_mlp_wide_fslabel_combfl_nosplit_k100", "bar": BAR}
CD_ARMS = {  # --mode cd (llf_cd_confirm): the static epoch-40 gnn and the MLPs against the CD bar, reactive and random; no twin, no fs_*
    "gnn": ARMS["gnn"], "wide": ARMS["wide"], "mlp": ARMS["mlp"], "bar": BAR, "reactive": "reactive", "random": "random"}
S0_ARMS = ("bar", "reactive", "random")   # single-seed (s0) arms, the rest run at seeds 1-3
CD_SEEDS = (1, 2, 3, 4, 5, 6)              # llf_cd_confirm_v1 amendment A2 (the user's POOL): learned arms at seeds 1-6
CD_SEED_BLOCKS = ((1, 2, 3), (4, 5, 6))   # descriptive per-block rows
CD_PRIMARY = (("gnn", "bar"), ("gnn", "wide"))
LABELS = ("MP-ATTRIBUTED", "MP-DIRECTION", "TIE", "TWIN-FASTER")
RUNAWAY_X = 10.0
STATS = ("whole", "a360")

# every point the node text leaves open; the value must be given in --rulings
RULING_POINTS = {
    "primary_metric": ("whole", "a360"),                      # which statistic the primary label is read on (the node says only "as d_read"; the secondary shows the after-360 s row beside the whole run)
    "runaway_metric": ("whole", "same_as_stat"),               # which metric the 10x test uses
    "arm_median_pool": ("all_seeds_pooled", "per_seed"),       # "its arm's median in that rung": over all of an arm's cells in the rung (seeds pooled) or per seed
    "runaway_unit": ("any_seed_flags_cell", "unflagged_seeds_mean"),  # a seed-mean learned cell is flagged if any seed is, or is the mean of its unflagged seeds (flagged only if all are)
    "no_window_left": ("drop_topology",),                      # a topology with no unflagged window pair is dropped from the median, the sign count and the Wilcoxon
    "weaker_order": None,                                      # a permutation of LABELS, strongest first (the last is the weakest); precedent: uncut_arm_gate_read.TW_ORDER
    "not_slower_than_mlp_pct": None,                           # a number: gnn vs MLP median (%) must be <= this on both rungs (0 reads "not slower")
    "seed_clause_reading": ("same_reading_as_label",),         # "each seed alone gives a median <= -1 %": seed-matched pairs, computed within the reading being labelled
    "mp_direction_min_faster": None,                           # an integer: topologies faster on at least one rung (the node says 8 of 12)
}


def check_rulings(r):
    miss = [k for k in RULING_POINTS if k not in r]
    if miss:
        raise SystemExit(f"FAIL LOUD: --rulings lacks {miss}; SCIENTIST rules on every open point before the freeze")
    for k, ok in RULING_POINTS.items():
        if ok is not None and r[k] not in ok:
            raise SystemExit(f"FAIL LOUD: ruling {k}={r[k]!r} is not one of {ok}")
    if sorted(r["weaker_order"]) != sorted(LABELS):
        raise SystemExit(f"FAIL LOUD: weaker_order must be a permutation of {LABELS}")
    if not isinstance(r["not_slower_than_mlp_pct"], (int, float)) or not isinstance(r["mp_direction_min_faster"], int):
        raise SystemExit("FAIL LOUD: not_slower_than_mlp_pct must be a number and mp_direction_min_faster an integer")
    return r


NAME = re.compile(r"^cc40s(\d+)__(g\d)(moderate|heavy)__(.+)_s(\d+)\.summary\.json$")


def load(root, topos, pin, arms=None, check=None, seeds=SEEDS, flat=None):
    """{(kind, seed, topology, window, rung): {'whole', 'a360'}}: exactly the expected grid, nothing else.
    pin: one code.commit prefix for every cell, or 'group=prefix,...' with the groups static (gnn, twin, mlp, wide), refs (the four fair-share-label arms) and bar: each cell must
    match its own group's prefix and every listed prefix must be used. Returns (cells, the commits seen, the commits seen per group).
    root: one directory of cell subdirectories, or a list of them. flat: optional (dir, kind): cells of that kind directly in dir, only for WINDOWS (other files there are ignored)."""
    arms = ARMS if arms is None else arms
    group_of = {arms[a]: ("bar" if a in S0_ARMS else "refs" if a.startswith("fs_") else "static") for a in arms}
    groups = tuple(sorted(set(group_of.values()), key=("static", "refs", "bar").index))
    if "=" in pin:
        want_pin = dict(p.split("=", 1) for p in pin.split(","))
        if set(want_pin) != set(groups) or not all(want_pin.values()):
            raise SystemExit(f"FAIL LOUD: --pin groups must be exactly {', '.join(groups[:-1])} and {groups[-1]}, got {sorted(want_pin)}")
    else:
        want_pin = {g: pin for g in groups}
    want = {(arms[a], 0 if a in S0_ARMS else s, t, w, r) for a in arms for s in ((0,) if a in S0_ARMS else seeds) for t in topos for w in WINDOWS for r in RUNGS}
    V, pins, by_group = {}, set(), {g: set() for g in want_pin}
    roots = [root] if isinstance(root, str) else list(root)
    files = [f for r_ in roots for f in sorted(glob.glob(os.path.join(r_, "*", "cc40s*.summary.json")))]
    if flat is not None:
        fdir, fkind = flat
        for f in sorted(glob.glob(os.path.join(fdir, "cc40s*.summary.json"))):
            m = NAME.match(os.path.basename(f))
            if m and m.group(4) == fkind and m.group(2) in WINDOWS:
                files.append(f)
    for f in files:
        m = NAME.match(os.path.basename(f))
        if not m:
            continue
        key = (m.group(4), int(m.group(5)), int(m.group(1)), m.group(2), m.group(3))
        if key in V:
            raise SystemExit(f"FAIL LOUD: {key} appears twice (second copy {f})")
        if key not in want:
            raise SystemExit(f"FAIL LOUD: unexpected cell {key} ({f})")
        s = json.load(open(f))
        code = s.get("code") or {}
        grp = group_of[key[0]]
        if code.get("dirty") is not False or not str(code.get("commit", "")).startswith(want_pin[grp]) or s.get("servers") != 24:
            raise SystemExit(f"FAIL LOUD: {f} ran at {code.get('commit')} dirty={code.get('dirty')} servers={s.get('servers')}, not {want_pin[grp]} clean at 24 servers")
        if check is not None:
            check(key, s, f)
        pins.add(code["commit"])
        by_group[grp].add(code["commit"])
        a360 = ((s.get("steady_state") or {}).get("after_360s") or {}).get("mean")
        if s.get("averageElapsedTime") is None or a360 is None:
            raise SystemExit(f"FAIL LOUD: {f} has no averageElapsedTime / steady_state.after_360s.mean")
        V[key] = {"whole": float(s["averageElapsedTime"]), "a360": float(a360)}
    miss = sorted(want - set(V))
    if miss:
        raise SystemExit(f"FAIL LOUD: {len(miss)} of {len(want)} cells missing, first {miss[:3]}")
    unused = sorted(p for p in set(want_pin.values()) if not any(c.startswith(p) for c in pins))
    if unused:
        raise SystemExit(f"FAIL LOUD: the listed pin(s) {unused} are used by no cell")
    return V, sorted(pins), {g: sorted(c) for g, c in by_group.items()}


class Grid:
    def __init__(self, V, topos, rul, arms=None, seeds=SEEDS):
        self.V, self.topos, self.rul, self.seed_set = V, tuple(topos), rul, tuple(seeds)
        self.arms = ARMS if arms is None else arms
        self.flag = {}   # (kind, seed, t, w, rung, stat) -> bool
        self.med = {}

    def seeds(self, arm):
        return (0,) if arm in S0_ARMS else self.seed_set

    def raw(self, arm, seed, t, w, rung, stat):
        return self.V[(self.arms[arm], seed, t, w, rung)][stat]

    def arm_median(self, arm, seed, rung, mstat):
        pool = self.rul["arm_median_pool"]
        key = (arm, None if pool == "all_seeds_pooled" else seed, rung, mstat)
        if key not in self.med:
            ss = self.seeds(arm) if pool == "all_seeds_pooled" else (seed,)
            self.med[key] = st.median(self.raw(arm, s, t, w, rung, mstat) for s in ss for t in self.topos for w in WINDOWS)
        return self.med[key]

    def seed_flagged(self, arm, seed, t, w, rung, stat):
        mstat = "whole" if self.rul["runaway_metric"] == "whole" else stat
        return self.raw(arm, seed, t, w, rung, mstat) > RUNAWAY_X * self.arm_median(arm, seed, rung, mstat)

    def cell(self, arm, t, w, rung, stat, reading, only_seed=None):
        """(value, flagged): the seed-mean learned cell (or the single seed, or tuple of seeds, asked for)."""
        ss = self.seeds(arm) if only_seed is None else tuple(only_seed) if isinstance(only_seed, tuple) else (only_seed,)
        vals = [(self.raw(arm, s, t, w, rung, stat), self.seed_flagged(arm, s, t, w, rung, stat)) for s in ss]
        if reading == "all":
            return st.fmean(v for v, _ in vals), False
        if self.rul["runaway_unit"] == "any_seed_flags_cell":
            return st.fmean(v for v, _ in vals), any(f for _, f in vals)
        ok = [v for v, f in vals if not f]
        return (st.fmean(ok), False) if ok else (st.fmean(v for v, _ in vals), True)

    def contrast(self, a, b, rung, stat, reading, seed=None):
        """Per topology: median over the unflagged windows of the paired % of A to B; then the median, the sign count and the exact Wilcoxon over the topologies kept."""
        per = {}
        for t in self.topos:
            pcts = []
            for w in WINDOWS:
                x, fx = self.cell(a, t, w, rung, stat, reading, seed if a not in S0_ARMS else None)
                y, fy = self.cell(b, t, w, rung, stat, reading, seed if b not in S0_ARMS else None)
                if not (fx or fy):
                    pcts.append(100.0 * (x / y - 1.0))
            per[t] = st.median(pcts) if pcts else None
        kept = [v for v in per.values() if v is not None]
        if not kept:
            return {"median_pct": None, "faster": 0, "n": 0, "p": None, "per_topology": per}
        return {"median_pct": st.median(kept), "faster": sum(1 for v in kept if v < 0), "n": len(kept), "p": exact_wilcoxon(kept), "per_topology": per}

    def runaways(self):
        out = []
        for a in self.arms:
            for s in self.seeds(a):
                for rung in RUNGS:
                    for t in self.topos:
                        for w in WINDOWS:
                            for stat in STATS:
                                if self.seed_flagged(a, s, t, w, rung, stat):
                                    out.append({"arm": a, "seed": s, "topology": t, "window": w, "rung": rung, "stat": stat, "value_s": self.raw(a, s, t, w, rung, stat),
                                                "arm_median_s": self.arm_median(a, s, rung, "whole" if self.rul["runaway_metric"] == "whole" else stat)})
        return out


def primary_label(g, reading):
    """One reading's label and its evidence."""
    rul, stat = g.rul, g.rul["primary_metric"]
    rows = {r: g.contrast("gnn", "twin", r, stat, reading) for r in RUNGS}
    hp = holm({r: rows[r]["p"] for r in RUNGS})
    ev = {"rows": {r: {k: v for k, v in rows[r].items() if k != "per_topology"} | {"holm_p": hp[r]} for r in RUNGS}}
    sig = lambda r: hp[r] is not None and hp[r] < 0.05
    if any(rows[r]["median_pct"] is not None and rows[r]["median_pct"] >= 2.0 and sig(r) for r in RUNGS):
        return "TWIN-FASTER", ev
    seed_med = {s: {r: g.contrast("gnn", "twin", r, stat, reading, seed=s)["median_pct"] for r in RUNGS} for s in SEEDS}
    mlp = {r: g.contrast("gnn", "mlp", r, stat, reading)["median_pct"] for r in RUNGS}
    ev["per_seed_median_pct"] = seed_med
    ev["gnn_vs_mlp_median_pct"] = mlp
    attributed = (all(rows[r]["median_pct"] is not None and rows[r]["median_pct"] <= -2.0 and sig(r) for r in RUNGS)
                  and all(seed_med[s][r] is not None and seed_med[s][r] <= -1.0 for s in SEEDS for r in RUNGS)
                  and all(mlp[r] is not None and mlp[r] <= rul["not_slower_than_mlp_pct"] for r in RUNGS))
    if attributed:
        return "MP-ATTRIBUTED", ev
    if all(rows[r]["median_pct"] is not None and rows[r]["median_pct"] < 0 for r in RUNGS) and any(rows[r]["faster"] >= rul["mp_direction_min_faster"] for r in RUNGS):
        return "MP-DIRECTION", ev
    return "TIE", ev


def secondary(g):
    out = {}
    pairs = (("gnn", "mlp"), ("gnn", "wide"), ("gnn", "bar"), ("gnn", "fs_gnn"), ("gnn", "fs_twin"), ("gnn", "fs_mlp"), ("gnn", "fs_wide"), ("twin", "mlp"), ("gnn", "twin"))
    for reading in ("all", "no_runaway"):
        for stat in STATS:
            for a, b in pairs:
                for r in RUNGS:
                    out[f"{reading}|{stat}|{a}_vs_{b}|{r}"] = g.contrast(a, b, r, stat, reading)
    return out


CD_MODE_POINTS = ("primary_metric", "runaway_metric", "arm_median_pool", "runaway_unit", "no_window_left")


def check_rulings_cd(r):
    """The cd mode has no label, so only the statistic-defining rulings are required (same keys and values as check_rulings)."""
    miss = [k for k in CD_MODE_POINTS if k not in r]
    if miss:
        raise SystemExit(f"FAIL LOUD: --rulings lacks {miss}; SCIENTIST rules on every open point before the freeze")
    for k in CD_MODE_POINTS:
        if r[k] not in RULING_POINTS[k]:
            raise SystemExit(f"FAIL LOUD: ruling {k}={r[k]!r} is not one of {RULING_POINTS[k]}")
    return r


CD_CKPT = {  # arm -> checkpoint basename stem (seed appended): llf_cd_confirm_v1 "Arms"
    "gnn": "uncut-linkstatic-gnn-eng-e40-seed", "wide": "uncut-linkstatic-mlp-wide-e40-seed", "mlp": "uncut-linkstatic-mlp-same-e40-seed"}
CD_CHARGE = {  # (arm, rung) -> {bucket: ms}: the parent primary's measured decision charge, llf_cd_confirm_v1 "Decision charge"
    ("gnn", "moderate"): {"le10": 5.01465, "11_50": 2.254181, "gt50": 2.191336}, ("gnn", "heavy"): {"le10": 6.151998, "11_50": 3.103575, "gt50": 3.678685},
    ("wide", "moderate"): {"le10": 4.527073, "11_50": 3.585821, "gt50": 7.915672}, ("wide", "heavy"): {"le10": 5.008803, "11_50": 5.134591, "gt50": 10.944349},
    ("mlp", "moderate"): {"le10": 4.275423, "11_50": 1.91878, "gt50": 2.003687}, ("mlp", "heavy"): {"le10": 5.181011, "11_50": 3.17074, "gt50": 3.602202}}
CD_BAR_CHARGE = 0.01098


def cd_check(key, s, f):
    """Fail loud when a cell's served checkpoint or decision charge is not the registered one."""
    kind, seed, _t, _w, rung = key
    env = s.get("env") or {}
    arm = next(a for a, k in CD_ARMS.items() if k == kind)
    if arm in CD_CKPT:
        want = f"{CD_CKPT[arm]}{seed}.pt"
        got = os.path.basename(str(env.get("GNN_MODEL_PATH", "")))
        if got != want:
            raise SystemExit(f"FAIL LOUD: {f} served {got!r}, not {want}")
        raw = env.get("HEROSIM_DECISION_COST_MS_BY_BUCKET")
        try:
            have = json.loads(raw)
        except (TypeError, ValueError):
            have = None
        if have != CD_CHARGE[(arm, rung)]:
            raise SystemExit(f"FAIL LOUD: {f} decision charge {raw!r}, not {CD_CHARGE[(arm, rung)]}")
    elif arm == "bar":
        if float(env.get("HEROSIM_DECISION_COST_MS", "nan")) != CD_BAR_CHARGE:
            raise SystemExit(f"FAIL LOUD: {f} bar decision charge {env.get('HEROSIM_DECISION_COST_MS')!r}, not {CD_BAR_CHARGE}")


def cd_label(nr, al):
    """CONFIRMED: without-runaways median < 0, Holm p < .05 and the all-cells median < 0; DIRECTION: median < 0 otherwise; NOT-FASTER: median >= 0 or no topology left."""
    if nr["median_pct"] is None or nr["median_pct"] >= 0:
        return "NOT-FASTER"
    if nr["holm_p"] is not None and nr["holm_p"] < 0.05 and al["median_pct"] is not None and al["median_pct"] < 0:
        return "CONFIRMED"
    return "DIRECTION"


def run_cd(root, rulings, topos, pin, out_path, ids_sha=None, seeds=CD_SEEDS, flat=None):
    """llf_cd_confirm_v1: gnn vs bar and gnn vs wide x {moderate, heavy}; Holm over the 4 in each reading; the label reads without runaways with the all-cells sign guard."""
    rul = check_rulings_cd(rulings)
    if out_path and os.path.exists(out_path):
        raise SystemExit(f"FAIL LOUD: {out_path} exists (read once)")
    V, pins, pins_by_group = load(root, topos, pin, CD_ARMS, check=cd_check, seeds=seeds, flat=flat)
    g = Grid(V, topos, rul, CD_ARMS, seeds=seeds)
    stat = rul["primary_metric"]
    primary = {}
    for reading in ("all", "no_runaway"):
        rows = {f"{a}_vs_{b}|{r}": g.contrast(a, b, r, stat, reading) for a, b in CD_PRIMARY for r in RUNGS}
        hp = holm({k: v["p"] for k, v in rows.items()})
        primary[reading] = {k: {kk: vv for kk, vv in v.items() if kk != "per_topology"} | {"holm_p": hp[k]} for k, v in rows.items()}
    labels = {k: cd_label(primary["no_runaway"][k], primary["all"][k]) for k in primary["no_runaway"]}
    per_seed = {f"{a}_vs_{b}|{r}|{rd}": {s_: g.contrast(a, b, r, stat, rd, seed=s_)["median_pct"] for s_ in seeds} for a, b in CD_PRIMARY for r in RUNGS for rd in ("all", "no_runaway")}
    blocks = [b_ for b_ in CD_SEED_BLOCKS if set(b_) <= set(seeds)] if tuple(seeds) != tuple(SEEDS) else []
    per_block = {f"{a}_vs_{b}|{r}|{rd}": {"s" + "".join(map(str, b_)): {k: v for k, v in g.contrast(a, b, r, stat, rd, seed=b_).items() if k != "per_topology"} for b_ in blocks}
                 for a, b in CD_PRIMARY for r in RUNGS for rd in ("all", "no_runaway")}
    names = list(CD_ARMS)
    sec = {}
    for reading in ("all", "no_runaway"):
        for st_ in STATS:
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    for r in RUNGS:
                        sec[f"{reading}|{st_}|{a}_vs_{b}|{r}"] = g.contrast(a, b, r, st_, reading)
    res = {"family": "llf_cd_confirm_v1 (read once)", "mode": "cd", "topologies": list(topos), "ids_sha256": ids_sha, "pins": pins, "pins_by_group": pins_by_group, "cells": len(V),
           "rulings": rul, "primary_metric": stat, "labels": labels, "primary": primary, "seeds": list(seeds), "per_seed_median_pct": per_seed, "per_seed_block": per_block, "runaways": g.runaways(), "secondary": sec,
           "reader_sha256": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()}
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path + ".tmp", "w") as fh:
            json.dump(res, fh, indent=1, default=str)
        os.replace(out_path + ".tmp", out_path)
    return res


def show_cd(res):
    fm = lambda x: "n/a" if x is None else f"{x:.4g}"
    print(f"llf_cd_confirm_v1; primary metric {res['primary_metric']}; cells {res['cells']}; pins {res['pins']}")
    for k, lab in res["labels"].items():
        print(f"  {k:22s} {lab}")
        for reading in ("no_runaway", "all"):
            row = res["primary"][reading][k]
            print(f"      [{reading:10s}] median {fm(row['median_pct'])} % faster {row['faster']}/{row['n']} p {fm(row['p'])} holm(4) {fm(row['holm_p'])}")
    print(f"runaway cells: {len(res['runaways'])}")


def run(root, rulings, topos, pin, out_path, ids_sha=None):
    rul = check_rulings(rulings)
    if out_path and os.path.exists(out_path):
        raise SystemExit(f"FAIL LOUD: {out_path} exists (read once)")
    V, pins, pins_by_group = load(root, topos, pin)
    g = Grid(V, topos, rul)
    labels, evidence = {}, {}
    for reading in ("all", "no_runaway"):
        labels[reading], evidence[reading] = primary_label(g, reading)
    final = max(labels.values(), key=rul["weaker_order"].index)
    res = {"family": "live_link_features_v1 confirmation (read once)", "topologies": list(topos), "ids_sha256": ids_sha, "pins": pins, "pins_by_group": pins_by_group, "cells": len(V), "rulings": rul,
           "primary_metric": rul["primary_metric"], "label": final, "label_by_reading": labels, "evidence": evidence, "runaways": g.runaways(), "secondary": secondary(g),
           "reader_sha256": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()}
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path + ".tmp", "w") as fh:
            json.dump(res, fh, indent=1, default=str)
        os.replace(out_path + ".tmp", out_path)
    return res


def show(res):
    print(f"LABEL {res['label']} (all cells {res['label_by_reading']['all']}; without runaways {res['label_by_reading']['no_runaway']}); primary metric {res['primary_metric']}; cells {res['cells']}; pins {res['pins']}")
    for reading in ("all", "no_runaway"):
        for r, row in res["evidence"][reading]["rows"].items():
            fm = lambda x: "n/a" if x is None else f"{x:.4g}"
            print(f"  gnn vs twin [{reading:10s}] {r:8s} median {fm(row['median_pct'])} % faster {row['faster']}/{row['n']} p {fm(row['p'])} holm {fm(row['holm_p'])}")
    print(f"runaway cells: {len(res['runaways'])}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", required=True)
    ap.add_argument("--rulings", required=True)
    ap.add_argument("--pin", required=True, help="one code.commit prefix for every cell, or static=<prefix>,refs=<prefix>,bar=<prefix>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=("confirm", "cd"), default="confirm")
    ap.add_argument("--ids", help="cd mode: comma-separated topology ids")
    ap.add_argument("--ids-sha256", help="cd mode: sha256 of the comma-joined id list, as filed")
    ap.add_argument("--react-cells", help="cd mode: a flat directory holding the reactive_s0 cells (the admission screen); only WINDOWS are read")
    a = ap.parse_args()
    if a.mode == "cd":
        if not a.ids or not a.ids_sha256:
            raise SystemExit("FAIL LOUD: --mode cd needs --ids and --ids-sha256")
        ids = tuple(int(x) for x in a.ids.split(","))
        sha = hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()
        if sha != a.ids_sha256 or len(set(ids)) != len(ids):
            raise SystemExit("FAIL LOUD: the id list does not match the filed sha256 (or repeats an id)")
        flat = (a.react_cells, CD_ARMS["reactive"]) if a.react_cells else None
        show_cd(run_cd(a.cells.split(","), json.load(open(a.rulings)), ids, a.pin, a.out, sha, flat=flat))
        return
    sha = hashlib.sha256(",".join(map(str, IDS)).encode()).hexdigest()
    if sha != IDS_SHA256:
        raise SystemExit("FAIL LOUD: the id list in this reader is not the sealed one")
    show(run(a.cells, json.load(open(a.rulings)), IDS, a.pin, a.out, sha))


if __name__ == "__main__":
    main()
