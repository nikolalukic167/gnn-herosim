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
  llf_confirm_read.py --mode cd --cells <root>[,<root>...] --react-cells <flat reactive dir> [--allow-missing <random cells>] --ids <ids> --ids-sha256 <sha> --rulings rulings.json --pin static=P,bar=P --out read.json
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
CD_ARM_REGISTRY = {**CD_ARMS, "twin": ARMS["twin"], "heft": "heft_k100"}   # --arms picks from these (knative_stable_v1, scale_d_v1: heft; mp_confirm_v1: twin)
S0_ARMS = S0_ARMS + ("heft",)
LABELS = ("MP-ATTRIBUTED", "MP-DIRECTION", "TIE", "TWIN-FASTER")
# outbound_link_features_v1 (--arm-set v9): the four partial_state_v9 e40 arms, the v8 static e40 gnn as the secondary arm, and the bar. All served at one v9 serving pin.
V9_ARM_REGISTRY = {"gnn": "ra_gnn_eng_combfl_linkout_e40_nosplit_k100", "twin": "ra_twin_eng_combfl_linkout_e40_nosplit_k100", "mlp": "ra_mlp_same_combfl_linkout_e40_nosplit_k100",
                   "wide": "ra_mlp_wide_combfl_linkout_e40_nosplit_k100", "v8": ARMS["gnn"], "bar": BAR}
V9_CKPT = {"gnn": "uncut-linkout-gnn-eng-e40-seed", "twin": "uncut-linkout-twin-e40-seed", "mlp": "uncut-linkout-mlp-same-e40-seed", "wide": "uncut-linkout-mlp-wide-e40-seed",
           "v8": "uncut-linkstatic-gnn-eng-e40-seed"}
V9_PRIMARY = (("gnn", "bar"), ("gnn", "twin"), ("gnn", "wide"), ("gnn", "mlp"))
V9_SECONDARY = (("gnn", "v8"),)
V9DEV_ARM_REGISTRY = {"gnn": "ra_gnn_eng_combfl_linkout_nosplit_k100", "twin": "ra_twin_eng_combfl_linkout_nosplit_k100", "mlp": "ra_mlp_same_combfl_linkout_nosplit_k100",
                      "wide": "ra_mlp_wide_combfl_linkout_nosplit_k100", "bar": BAR}   # the 20-epoch development arms (--arm-set v9dev): bare stems, seeds 1-3, no v8 arm, no secondary family
V9DEV_CKPT = {"gnn": "uncut-linkout-gnn-eng-seed", "twin": "uncut-linkout-twin-seed", "mlp": "uncut-linkout-mlp-same-seed", "wide": "uncut-linkout-mlp-wide-seed"}
V9DEV_SEEDS = (1, 2, 3)
ARM_SETS = {"v9": (V9_ARM_REGISTRY, V9_CKPT, V9_SECONDARY), "v9dev": (V9DEV_ARM_REGISTRY, V9DEV_CKPT, ())}
V9_CHARGED = ("gnn", "twin", "mlp", "wide")   # arms whose per-(arm, rung) decision charge comes from --charges (measured on the v9 code); the v8 arm keeps its llf_cd_confirm_v1 charge
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


NAME = re.compile(r"^cc40s(\d+)__(g\d)([A-Za-z][A-Za-z0-9.]*)__(.+)_s(\d+)\.summary\.json$")  # rung: moderate, heavy or a registered tag (knative_stable_v1: f<fraction>)


def load(root, topos, pin, arms=None, check=None, seeds=SEEDS, flat=None, allow_missing=(), rungs=RUNGS, servers=24, clients=None, failed_learned=(), extra=None):
    """{(kind, seed, topology, window, rung): {'whole', 'a360'}}: exactly the expected grid, nothing else.
    pin: one code.commit prefix for every cell, or 'group=prefix,...' with the groups static (gnn, twin, mlp, wide), refs (the four fair-share-label arms) and bar: each cell must
    match its own group's prefix and every listed prefix must be used. Returns (cells, the commits seen, the commits seen per group).
    root: one directory of cell subdirectories, or a list of them. flat: optional (dir, kind): cells of that kind directly in dir, only for WINDOWS (other files there are ignored).
    failed_learned: learned-arm cells known to have failed twice (scale_d_v1 v3), accepted as absent like allow_missing. extra: optional {} filled with
    the per-cell fields admission and the charge ratio need (only when given, so the filed reads are unchanged)."""
    arms = ARMS if arms is None else arms
    group_of = {arms[a]: ("bar" if a in S0_ARMS else "refs" if a.startswith("fs_") else "static") for a in arms}
    groups = tuple(sorted(set(group_of.values()), key=("static", "refs", "bar").index))
    if "=" in pin:
        want_pin = dict(p.split("=", 1) for p in pin.split(","))
        if set(want_pin) != set(groups) or not all(want_pin.values()):
            raise SystemExit(f"FAIL LOUD: --pin groups must be exactly {', '.join(groups[:-1])} and {groups[-1]}, got {sorted(want_pin)}")
    else:
        want_pin = {g: pin for g in groups}
    want = {(arms[a], 0 if a in S0_ARMS else s, t, w, r) for a in arms for s in ((0,) if a in S0_ARMS else seeds) for t in topos for w in WINDOWS for r in rungs}
    V, pins, by_group = {}, set(), {g: set() for g in want_pin}
    roots = [root] if isinstance(root, str) else list(root)
    files = [f for r_ in roots for f in sorted(glob.glob(os.path.join(r_, "*", "cc40s*.summary.json")))]
    if flat is not None:
        fdir, fkind = flat
        for f in sorted(glob.glob(os.path.join(fdir, "cc40s*.summary.json"))):
            m = NAME.match(os.path.basename(f))
            if m and m.group(4) == fkind and m.group(2) in WINDOWS and m.group(3) in rungs:
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
        if code.get("dirty") is not False or not str(code.get("commit", "")).startswith(want_pin[grp]) or s.get("servers") != servers:
            raise SystemExit(f"FAIL LOUD: {f} ran at {code.get('commit')} dirty={code.get('dirty')} servers={s.get('servers')}, not {want_pin[grp]} clean at {servers} servers")
        if clients is not None and s.get("clients") != clients:
            raise SystemExit(f"FAIL LOUD: {f} has clients={s.get('clients')}, not {clients}")
        if check is not None:
            check(key, s, f)
        pins.add(code["commit"])
        by_group[grp].add(code["commit"])
        a360 = ((s.get("steady_state") or {}).get("after_360s") or {}).get("mean")
        if s.get("averageElapsedTime") is None or a360 is None:
            raise SystemExit(f"FAIL LOUD: {f} has no averageElapsedTime / steady_state.after_360s.mean")
        V[key] = {"whole": float(s["averageElapsedTime"]), "a360": float(a360)}
        if extra is not None:
            extra[key] = cell_extra(s)
    allow = set(allow_missing)
    bad = sorted(k for k in allow if k not in want or k[0] != "random")
    if bad:
        raise SystemExit(f"FAIL LOUD: --allow-missing names {bad}: only cells of the random arm in the grid may be allowed missing")
    learned_kinds = {arms[a] for a in arms if a not in S0_ARMS}
    badf = sorted(k for k in failed_learned if k not in want or k[0] not in learned_kinds)
    if badf:
        raise SystemExit(f"FAIL LOUD: --failed-learned names {badf}: only learned-arm cells in the grid")
    allow |= set(failed_learned)
    present = sorted(allow & set(V))
    if present:
        raise SystemExit(f"FAIL LOUD: --allow-missing names {present}, which exist; the list is stale")
    miss = sorted(want - set(V) - allow)
    if miss:
        raise SystemExit(f"FAIL LOUD: {len(miss)} of {len(want)} cells missing, first {miss[:3]}")
    unused = sorted(p for p in set(want_pin.values()) if not any(c.startswith(p) for c in pins))
    if unused:
        raise SystemExit(f"FAIL LOUD: the listed pin(s) {unused} are used by no cell")
    return V, sorted(pins), {g: sorted(c) for g, c in by_group.items()}


def cell_extra(s):
    """The fields knative_stable_v1's admission and scale_d_v1's charge ratio read (None where absent)."""
    fs, ae, dt = s.get("peerExchangeFairshare") or {}, s.get("arrival_end") or {}, s.get("decisionTiming") or {}
    env = s.get("env") or {}
    return {"endTime": s.get("endTime"), "last_arrival_s": ae.get("last_arrival_s"), "flow_seconds": fs.get("flow_seconds"), "extra_seconds": fs.get("extra_seconds"),
            "queue_share": s.get("queue_share"), "num_tasks": s.get("num_tasks"), "by_bucket": dt.get("by_bucket"), "charge_by_bucket": env.get("HEROSIM_DECISION_COST_MS_BY_BUCKET")}


STABLE_DRAIN, STABLE_SLOWDOWN, STABLE_QUEUE = 0.10, 3.0, 0.80   # knative_stable_v1 "Stability rule"


def reactive_stable(x):
    """knative_stable_v1: drain (end - last arrival) / last arrival <= 0.10, fair-share flow / solo seconds <= 3, queue_share <= 0.80; unknown is not a pass."""
    try:
        end, last, flow, extra_s, qs = (float(x[k]) for k in ("endTime", "last_arrival_s", "flow_seconds", "extra_seconds", "queue_share"))
    except (TypeError, ValueError, KeyError):
        return False
    solo = flow - extra_s
    if not (last > 0 and solo > 0) or any(v != v for v in (end, last, flow, extra_s, qs)):
        return False
    return (end - last) / last <= STABLE_DRAIN and flow / solo <= STABLE_SLOWDOWN and qs <= STABLE_QUEUE


class Grid:
    def __init__(self, V, topos, rul, arms=None, seeds=SEEDS, rungs=RUNGS, excluded=()):
        self.V, self.topos, self.rul, self.seed_set = V, tuple(topos), rul, tuple(seeds)
        self.rungs, self.excluded = tuple(rungs), set(excluded)   # excluded: (topology, window, rung) not admitted; every arm's cell there is absent
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
            self.med[key] = st.median(self.raw(arm, s, t, w, rung, mstat) for s in ss for t in self.topos for w in WINDOWS if self.has(arm, s, t, w, rung))
        return self.med[key]

    def seed_flagged(self, arm, seed, t, w, rung, stat):
        mstat = "whole" if self.rul["runaway_metric"] == "whole" else stat
        return self.raw(arm, seed, t, w, rung, mstat) > RUNAWAY_X * self.arm_median(arm, seed, rung, mstat)

    def has(self, arm, seed, t, w, rung):
        return (self.arms[arm], seed, t, w, rung) in self.V and (t, w, rung) not in self.excluded

    def cell(self, arm, t, w, rung, stat, reading, only_seed=None):
        """(value, flagged): the seed-mean learned cell (or the single seed, or tuple of seeds, asked for); (None, None) for an allowed-missing cell."""
        ss = self.seeds(arm) if only_seed is None else tuple(only_seed) if isinstance(only_seed, tuple) else (only_seed,)
        if not all(self.has(arm, s, t, w, rung) for s in ss):
            return None, None
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
                if x is not None and y is not None and not (fx or fy):
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
                for rung in self.rungs:
                    for t in self.topos:
                        for w in WINDOWS:
                            for stat in STATS:
                                if self.has(a, s, t, w, rung) and self.seed_flagged(a, s, t, w, rung, stat):
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
    "gnn": "uncut-linkstatic-gnn-eng-e40-seed", "wide": "uncut-linkstatic-mlp-wide-e40-seed", "mlp": "uncut-linkstatic-mlp-same-e40-seed",
    "twin": "uncut-linkstatic-twin-e40-seed"}
CD_CHARGE = {  # (arm, rung) -> {bucket: ms}: the parent primary's measured decision charge, llf_cd_confirm_v1 "Decision charge"
    ("gnn", "moderate"): {"le10": 5.01465, "11_50": 2.254181, "gt50": 2.191336}, ("gnn", "heavy"): {"le10": 6.151998, "11_50": 3.103575, "gt50": 3.678685},
    ("wide", "moderate"): {"le10": 4.527073, "11_50": 3.585821, "gt50": 7.915672}, ("wide", "heavy"): {"le10": 5.008803, "11_50": 5.134591, "gt50": 10.944349},
    ("mlp", "moderate"): {"le10": 4.275423, "11_50": 1.91878, "gt50": 2.003687}, ("mlp", "heavy"): {"le10": 5.181011, "11_50": 3.17074, "gt50": 3.602202},
    ("twin", "moderate"): {"le10": 4.323723, "11_50": 1.878764, "gt50": 2.129705}, ("twin", "heavy"): {"le10": 4.826589, "11_50": 2.525467, "gt50": 2.983758}}  # twin: the parent primary's cells
CD_BAR_CHARGE = {"moderate": 0.0116, "heavy": 0.01098}   # cd_ib0_inflightfull w per rung (decision_cost_w.json), as the parent primary's bar cells
CD_HEFT_CHARGE = {"moderate": 0.003035, "heavy": 0.002807}   # heft w per rung (decision_cost_w.json; scale_d_v1, knative_stable_v1)
CHARGE_RATIO_REF24 = {("gnn", "moderate"): 1.45, ("gnn", "heavy"): 1.22, ("wide", "moderate"): 1.41, ("wide", "heavy"): 1.41,
                      ("mlp", "moderate"): 1.42, ("mlp", "heavy"): 1.22}   # scale_d_v1 v3: median measured/charged at 24 servers (llf_cd_confirm_v1 cells)
CHARGE_LIMITED_X = 1.5


def make_cd_check(arms=None, charge_rung=None, ckpt=None, charge=None):
    """cd_check over `arms`; charge_rung maps a cell's rung to the rung whose registered charge it must carry (knative_stable_v1: f<x> -> moderate)."""
    arms = CD_ARMS if arms is None else arms
    charge_rung = charge_rung or {}

    def check(key, s, f):
        kind, seed, t, w, rung = key
        return cd_check(key, s, f, arms, charge_rung.get(rung, rung), ckpt, charge)
    return check


def cd_check(key, s, f, arms=None, crung=None, ckpt=None, charge=None):
    """Fail loud when a cell's served checkpoint or decision charge is not the registered one."""
    kind, seed, _t, _w, rung = key
    arms = CD_ARMS if arms is None else arms
    rung = rung if crung is None else crung
    env = s.get("env") or {}
    ckpt = CD_CKPT if ckpt is None else ckpt
    charge = CD_CHARGE if charge is None else charge
    arm = next(a for a, k in arms.items() if k == kind)
    if arm in ckpt:
        want = f"{ckpt[arm]}{seed}.pt"
        got = os.path.basename(str(env.get("GNN_MODEL_PATH", "")))
        if got != want:
            raise SystemExit(f"FAIL LOUD: {f} served {got!r}, not {want}")
        raw = env.get("HEROSIM_DECISION_COST_MS_BY_BUCKET")
        try:
            have = json.loads(raw)
        except (TypeError, ValueError):
            have = None
        if have != charge[(arm, rung)]:
            raise SystemExit(f"FAIL LOUD: {f} decision charge {raw!r}, not {charge[(arm, rung)]}")
    elif arm in ("bar", "heft"):
        want = (CD_BAR_CHARGE if arm == "bar" else CD_HEFT_CHARGE)[rung]
        if float(env.get("HEROSIM_DECISION_COST_MS", "nan")) != want:
            raise SystemExit(f"FAIL LOUD: {f} {arm} decision charge {env.get('HEROSIM_DECISION_COST_MS')!r}, not {want}")


def cd_label(nr, al, blocked=False, min_n=None):
    """CONFIRMED: without-runaways median < 0, Holm p < .05 and the all-cells median < 0; DIRECTION: median < 0 otherwise; NOT-FASTER: median >= 0 or no topology left.
    blocked (scale_d_v1 v3: a gnn cell failed at this rung): never CONFIRMED. min_n (scale_d_v1 v3): a test that is not CONFIRMED with n < min_n is UNDERPOWERED."""
    if nr["median_pct"] is None or nr["median_pct"] >= 0:
        lab = "NOT-FASTER"
    elif not blocked and nr["holm_p"] is not None and nr["holm_p"] < 0.05 and al["median_pct"] is not None and al["median_pct"] < 0:
        return "CONFIRMED"
    else:
        lab = "DIRECTION"
    if min_n is not None and (nr.get("n") or 0) < min_n:
        return "UNDERPOWERED"
    return lab


def charge_ratios(extra, arms, rungs):
    """scale_d_v1 v3: per (arm, rung) the median over cells of sum(measured bucket seconds) / sum(charged w x plan_evals), and it normalised by the arm's 24-server median."""
    out = {}
    for a in arms:
        if a not in CD_CKPT:
            continue
        for r in rungs:
            vals = []
            for (kind, _s, _t, _w, rung), x in extra.items():
                if kind != arms[a] or rung != r:
                    continue
                try:
                    w = json.loads(x["charge_by_bucket"])
                    meas = sum(float(b["total_s"]) for b in x["by_bucket"].values()) * 1000.0
                    chg = sum(float(w[k]) * float(b["plan_evals"]) for k, b in x["by_bucket"].items())
                except (TypeError, ValueError, KeyError, AttributeError):
                    raise SystemExit(f"FAIL LOUD: {kind} {r}: decisionTiming.by_bucket or the charge string is unreadable")
                vals.append(meas / chg)
            if vals:
                ref = CHARGE_RATIO_REF24.get((a, r))
                med = st.median(vals)
                out[f"{a}|{r}"] = {"median": med, "n": len(vals), "ref24": ref, "normalised": None if ref is None else med / ref}
    flags = {}
    for r in rungs:
        g_, w_ = out.get(f"gnn|{r}"), out.get(f"wide|{r}")
        if g_ and g_["normalised"] is not None:
            flags[r] = bool(g_["normalised"] > CHARGE_LIMITED_X and (w_ is None or w_["normalised"] is None or g_["normalised"] > w_["normalised"]))
    return {"ratios": out, "gnn_charge_limited": flags}


def run_cd(root, rulings, topos, pin, out_path, ids_sha=None, seeds=CD_SEEDS, flat=None, allow_missing=(), arms=None, primary_pairs=None, rungs=RUNGS,
           servers=24, clients=None, failed_learned=(), admit_reactive=False, charge_rung=None, min_n=None, charge_ratio=False, tag=None, arm_set_name=None, charges=None, secondary_pairs=None, charges_sha256=None):
    """llf_cd_confirm_v1: gnn vs bar and gnn vs wide x {moderate, heavy}; Holm over the 4 in each reading; the label reads without runaways with the all-cells sign guard.
    The keyword options extend it (every one off by default, so the filed llf_cd_confirm_v1 read is reproduced byte for byte):
      arms / primary_pairs / rungs   the arm set (names from CD_ARM_REGISTRY), the primary contrasts and the rungs; Holm over len(pairs) x len(rungs)
      servers / clients              the size every cell must carry (scale_d_v1)
      failed_learned                 learned cells that failed twice: absent, and a gnn one blocks CONFIRMED at its rung (scale_d_v1 v3)
      admit_reactive                 a (topology, window, rung) enters only if its reactive cell is stable (knative_stable_v1); computed here
      charge_rung                    {cell rung: rung whose registered charge it carries} (knative_stable_v1: the moderate copies)
      min_n                          UNDERPOWERED below this n_eff unless CONFIRMED (scale_d_v1 v3)
      charge_ratio                   report the measured/charged decision-time ratios and the gnn charge-limited flag (scale_d_v1 v3)
      tag                            prefix every label (mp_confirm_v1: INTERIM for the seeds 1-3 read)
      arm_set_name="v9"|"v9dev"      outbound_link_features_v1: the V9_ARM_REGISTRY arms (default all six), V9_PRIMARY (4 contrasts) and the secondary family V9_SECONDARY with its own Holm; v9dev is the
                                     development read (V9DEV_ARM_REGISTRY, seeds 1-3, no v8 arm, no secondary), tag DEV required. Both take one --pin prefix for every arm;
                                     charges is the parsed --charges file {"<arm>|<rung>": {bucket: ms}} for the four v9 learned arms (measured on the v9 code; required)
      secondary_pairs                a second family (descriptive in the filed reads, here labelled): Holm over len(pairs) x len(rungs) on its own, the same label rule and blocks"""
    rul = check_rulings_cd(rulings)
    if out_path and os.path.exists(out_path):
        raise SystemExit(f"FAIL LOUD: {out_path} exists (read once)")
    if arm_set_name is not None and arm_set_name not in ARM_SETS:
        raise SystemExit(f"FAIL LOUD: unknown arm set {arm_set_name!r}")
    v9 = arm_set_name is not None
    if v9:
        if "=" in pin:
            raise SystemExit(f"FAIL LOUD: --arm-set {arm_set_name} serves every arm at one pin; the group form of --pin is refused")
        if arm_set_name == "v9dev" and (tag != "DEV" or tuple(seeds) != V9DEV_SEEDS):
            raise SystemExit("FAIL LOUD: --arm-set v9dev is the development read: --tag DEV and seeds 1,2,3")
    registry, v9_ckpt, v9_secondary = ARM_SETS[arm_set_name] if v9 else (CD_ARM_REGISTRY, None, ())
    arm_set = (registry if v9 else CD_ARMS) if arms is None else {a: registry[a] for a in arms}
    pairs = (V9_PRIMARY if v9 else CD_PRIMARY) if primary_pairs is None else tuple(primary_pairs)
    sec_pairs = v9_secondary if secondary_pairs is None else tuple(secondary_pairs)
    v9_charge = None
    if v9:
        if charges is None:
            raise SystemExit("FAIL LOUD: the v9 arm set needs --charges (each v9 arm's own measured decision charge, never a v8 value)")
        stale = sorted(k for k, c in charges.items() if c in list(CD_CHARGE.values()))
        if stale:
            raise SystemExit(f"FAIL LOUD: --charges {stale} equal a v8 (CD_CHARGE) value; each v9 arm carries its own measured charge")
        v9_charge = {(a, r): charges[f"{a}|{r}"] for a in V9_CHARGED if a in arm_set for r in rungs if f"{a}|{r}" in charges}
        miss_c = sorted(f"{a}|{r}" for a in V9_CHARGED if a in arm_set for r in rungs if (a, r) not in v9_charge)
        if miss_c:
            raise SystemExit(f"FAIL LOUD: --charges lacks {miss_c}")
        if "v8" in arm_set:   # the v8 arm keeps its llf_cd_confirm_v1 charge
            v9_charge.update({("v8", r): CD_CHARGE[("gnn", r)] for r in rungs})
    for a, b in sec_pairs:
        if a not in arm_set or b not in arm_set:
            raise SystemExit(f"FAIL LOUD: secondary pair {a} vs {b} names an arm outside {sorted(arm_set)}")
    for a, b in pairs:
        if a not in arm_set or b not in arm_set:
            raise SystemExit(f"FAIL LOUD: primary pair {a} vs {b} names an arm outside {sorted(arm_set)}")
    if admit_reactive and "reactive" not in arm_set:
        raise SystemExit("FAIL LOUD: admission reads the reactive cells, but reactive is not an arm")
    want_extra = admit_reactive or charge_ratio
    extra = {} if want_extra else None
    check = cd_check if (arms is None and not charge_rung and not v9) else make_cd_check(arm_set, charge_rung, v9_ckpt, v9_charge)
    V, pins, pins_by_group = load(root, topos, pin, arm_set, check=check, seeds=seeds, flat=flat, allow_missing=allow_missing, rungs=rungs,
                                  servers=servers, clients=clients, failed_learned=failed_learned, extra=extra)
    excluded, admission = set(), None
    if admit_reactive:
        rk = arm_set["reactive"]
        admitted = {(t, w, r): (rk, 0, t, w, r) in extra and reactive_stable(extra[(rk, 0, t, w, r)]) for t in topos for w in WINDOWS for r in rungs}
        excluded = {k for k, ok in admitted.items() if not ok}
        admission = {"admitted": sorted(f"{t}:{w}:{r}" for (t, w, r), ok in admitted.items() if ok), "excluded": sorted(f"{t}:{w}:{r}" for t, w, r in excluded),
                     "topologies_with_no_admitted_window": {r: sorted(t for t in topos if all((t, w, r) in excluded for w in WINDOWS)) for r in rungs},
                     "rule": {"drain_max": STABLE_DRAIN, "slowdown_max": STABLE_SLOWDOWN, "queue_share_max": STABLE_QUEUE}}
    g = Grid(V, topos, rul, arm_set, seeds=seeds, rungs=rungs, excluded=excluded)
    blocked = {r: any(k[0] == arm_set.get("gnn") and k[4] == r for k in failed_learned) for r in rungs}
    stat = rul["primary_metric"]
    primary = {}
    for reading in ("all", "no_runaway"):
        rows = {f"{a}_vs_{b}|{r}": g.contrast(a, b, r, stat, reading) for a, b in pairs for r in rungs}
        hp = holm({k: v["p"] for k, v in rows.items()})
        primary[reading] = {k: {kk: vv for kk, vv in v.items() if kk != "per_topology"} | {"holm_p": hp[k]} for k, v in rows.items()}
    labels = {k: cd_label(primary["no_runaway"][k], primary["all"][k], blocked=blocked[k.split("|")[1]], min_n=min_n) for k in primary["no_runaway"]}
    secondary_family = None
    if sec_pairs:
        secondary_family = {}
        for reading in ("all", "no_runaway"):
            rows = {f"{a}_vs_{b}|{r}": g.contrast(a, b, r, stat, reading) for a, b in sec_pairs for r in rungs}
            hp = holm({k: v["p"] for k, v in rows.items()})
            secondary_family[reading] = {k: {kk: vv for kk, vv in v.items() if kk != "per_topology"} | {"holm_p": hp[k]} for k, v in rows.items()}
        labels |= {f"secondary|{k}": cd_label(secondary_family["no_runaway"][k], secondary_family["all"][k], blocked=blocked[k.split("|")[1]], min_n=min_n) for k in secondary_family["no_runaway"]}
    if tag:
        labels = {k: f"{tag}-{v}" for k, v in labels.items()}
    per_seed = {f"{a}_vs_{b}|{r}|{rd}": {s_: g.contrast(a, b, r, stat, rd, seed=s_)["median_pct"] for s_ in seeds} for a, b in pairs for r in rungs for rd in ("all", "no_runaway")}
    blocks = [b_ for b_ in CD_SEED_BLOCKS if set(b_) <= set(seeds)] if tuple(seeds) != tuple(SEEDS) else []
    per_block = {f"{a}_vs_{b}|{r}|{rd}": {"s" + "".join(map(str, b_)): {k: v for k, v in g.contrast(a, b, r, stat, rd, seed=b_).items() if k != "per_topology"} for b_ in blocks}
                 for a, b in pairs for r in rungs for rd in ("all", "no_runaway")}
    names = list(arm_set)
    sec = {}
    for reading in ("all", "no_runaway"):
        for st_ in STATS:
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    for r in rungs:
                        sec[f"{reading}|{st_}|{a}_vs_{b}|{r}"] = g.contrast(a, b, r, st_, reading)
    res = {"family": "llf_cd_confirm_v1 (read once)", "mode": "cd", "topologies": list(topos), "ids_sha256": ids_sha, "pins": pins, "pins_by_group": pins_by_group, "cells": len(V),
           "rulings": rul, "allowed_missing": sorted(f"cc40s{t}__{w}{r}__{k}_s{s_}" for k, s_, t, w, r in allow_missing), "primary_metric": stat, "labels": labels, "primary": primary, "seeds": list(seeds), "per_seed_median_pct": per_seed, "per_seed_block": per_block, "runaways": g.runaways(), "secondary": sec,
           "reader_sha256": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()}
    if secondary_family is not None:
        res["secondary_family"] = secondary_family
    if v9:
        res["charges"] = {f"{a}|{r}": c for (a, r), c in sorted(v9_charge.items())}
        res["charges_sha256"] = charges_sha256
    opts = {"arm_set": arm_set_name, "secondary_pairs": None if secondary_pairs is None else [list(p_) for p_ in sec_pairs], "arms": arms, "primary_pairs": None if primary_pairs is None else [list(p_) for p_ in pairs], "rungs": None if tuple(rungs) == RUNGS else list(rungs),
            "servers": None if servers == 24 else servers, "clients": clients, "failed_learned": sorted(f"cc40s{t}__{w}{r}__{k}_s{s_}" for k, s_, t, w, r in failed_learned) or None,
            "admit_reactive": admit_reactive or None, "charge_rung": charge_rung or None, "min_n": min_n, "charge_ratio": charge_ratio or None, "tag": tag}
    opts = {k: v for k, v in opts.items() if v is not None}
    if opts:   # only an extended read carries these keys; the filed llf_cd_confirm_v1 read has none of them
        res["options"] = opts
        res["gnn_blocked_by_failure"] = blocked
        if admission is not None:
            res["admission"] = admission
        if charge_ratio:
            res["charge_ratio"] = charge_ratios(extra, arm_set, rungs)
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
            fam, key = (res["secondary_family"], k.split("|", 1)[1]) if k.startswith("secondary|") else (res["primary"], k)
            row = fam[reading][key]
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
    ap.add_argument("--allow-missing", default="", help="cd mode: comma-separated random cells known not to exist (cc40s<t>__<w><rung>__random_s0); any other missing cell fails loud")
    ap.add_argument("--react-cells", help="cd mode: a flat directory holding the reactive_s0 cells (the admission screen); only WINDOWS are read")
    ap.add_argument("--arms", help="cd mode: comma-separated arms from CD_ARM_REGISTRY (default the llf_cd_confirm_v1 set)")
    ap.add_argument("--primary", help="cd mode: comma-separated A:B primary contrasts (default gnn:bar,gnn:wide)")
    ap.add_argument("--rungs", help="cd mode: comma-separated rungs (default moderate,heavy; knative_stable_v1: its chosen f<x> tag)")
    ap.add_argument("--charge-rung", help="cd mode: comma-separated RUNG=REGISTERED_RUNG charge copies (knative_stable_v1: f<x>=moderate)")
    ap.add_argument("--seeds", help="cd mode: comma-separated learned seeds (default 1-6)")
    ap.add_argument("--servers", type=int, default=24, help="cd mode: the server count every cell must carry")
    ap.add_argument("--clients", type=int, help="cd mode: the client count every cell must carry")
    ap.add_argument("--failed-learned", default="", help="cd mode: learned cells that failed twice (scale_d_v1 v3); a gnn one blocks CONFIRMED at its rung")
    ap.add_argument("--admit-reactive", action="store_true", help="cd mode: knative_stable_v1 cell admission from the reactive cells")
    ap.add_argument("--min-n", type=int, help="cd mode: UNDERPOWERED below this n_eff unless CONFIRMED (scale_d_v1 v3: 8)")
    ap.add_argument("--charge-ratio", action="store_true", help="cd mode: report measured/charged decision-time ratios (scale_d_v1 v3)")
    ap.add_argument("--tag", help="cd mode: prefix every label (mp_confirm_v1: INTERIM)")
    ap.add_argument("--arm-set", choices=tuple(ARM_SETS), help="cd mode: outbound_link_features_v1 arms: v9 (gnn/twin/wide/mlp e40, the v8 gnn as secondary, the bar) or v9dev (the 20-epoch development arms, seeds 1-3, --tag DEV)")
    ap.add_argument("--charges", help="cd mode, --arm-set v9/v9dev: JSON {\"<arm>|<rung>\": {bucket: ms}} of each v9 arm's measured decision charge; its sha256 is recorded")
    ap.add_argument("--secondary", help="cd mode: comma-separated A:B pairs of a second family with its own Holm (v9 default gnn:v8)")
    a = ap.parse_args()
    if a.mode == "cd":
        if not a.ids or not a.ids_sha256:
            raise SystemExit("FAIL LOUD: --mode cd needs --ids and --ids-sha256")
        ids = tuple(int(x) for x in a.ids.split(","))
        sha = hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()
        if sha != a.ids_sha256 or len(set(ids)) != len(ids):
            raise SystemExit("FAIL LOUD: the id list does not match the filed sha256 (or repeats an id)")
        flat = (a.react_cells, CD_ARMS["reactive"]) if a.react_cells else None

        def cells(arg, flag):
            out = []
            for c in filter(None, arg.split(",")):
                m = NAME.match(c + ".summary.json")
                if not m:
                    raise SystemExit(f"FAIL LOUD: {flag} entry {c!r} is not cc40s<t>__<w><rung>__<kind>_s<seed>")
                out.append((m.group(4), int(m.group(5)), int(m.group(1)), m.group(2), m.group(3)))
            return out
        kw = {}
        if a.arm_set:
            kw["arm_set_name"] = a.arm_set
            if not a.charges:
                raise SystemExit("FAIL LOUD: --arm-set v9 needs --charges")
            kw["charges"] = json.load(open(a.charges))
            kw["charges_sha256"] = hashlib.sha256(open(a.charges, "rb").read()).hexdigest()
            if a.arm_set == "v9dev" and not a.seeds:
                kw["seeds"] = V9DEV_SEEDS
        elif a.charges:
            raise SystemExit("FAIL LOUD: --charges is only read with --arm-set v9 or v9dev")
        if a.secondary:
            kw["secondary_pairs"] = [tuple(x.split(":")) for x in a.secondary.split(",")]
        if a.arms:
            kw["arms"] = a.arms.split(",")
            registry = ARM_SETS[a.arm_set][0] if a.arm_set else CD_ARM_REGISTRY
            bad = [x for x in kw["arms"] if x not in registry]
            if bad:
                raise SystemExit(f"FAIL LOUD: --arms {bad} not in {sorted(registry)}")
        if a.primary:
            kw["primary_pairs"] = [tuple(x.split(":")) for x in a.primary.split(",")]
        if a.rungs:
            kw["rungs"] = tuple(a.rungs.split(","))
        if a.charge_rung:
            kw["charge_rung"] = dict(x.split("=", 1) for x in a.charge_rung.split(","))
        if a.seeds:
            kw["seeds"] = tuple(int(x) for x in a.seeds.split(","))
        if a.clients is not None:
            kw["clients"] = a.clients
        if a.failed_learned:
            kw["failed_learned"] = cells(a.failed_learned, "--failed-learned")
        if a.admit_reactive:
            kw["admit_reactive"] = True
        if a.min_n is not None:
            kw["min_n"] = a.min_n
        if a.charge_ratio:
            kw["charge_ratio"] = True
        if a.tag:
            kw["tag"] = a.tag
        show_cd(run_cd(a.cells.split(","), json.load(open(a.rulings)), ids, a.pin, a.out, sha, flat=flat, allow_missing=cells(a.allow_missing, "--allow-missing"),
                       servers=a.servers, **kw))
        return
    sha = hashlib.sha256(",".join(map(str, IDS)).encode()).hexdigest()
    if sha != IDS_SHA256:
        raise SystemExit("FAIL LOUD: the id list in this reader is not the sealed one")
    show(run(a.cells, json.load(open(a.rulings)), IDS, a.pin, a.out, sha))


if __name__ == "__main__":
    main()
