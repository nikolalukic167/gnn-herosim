#!/usr/bin/env python3
"""llf_cd_confirm_v1 descriptive read (no freeze): the static epoch-40 gnn (seeds 1-3) against reactive Knative and random (s0), with the CD bar and the wide MLP for context,
on the sealed 12. Statistic and rules are the parent reader's (llf_confirm_read.Grid: paired % per topology and window, median over the windows kept, median over topologies,
exact Wilcoxon, runaway > 10x the arm's median in the rung, any seed flags a cell). Cells that did not finish (random watchdog stalls) are absent: a pair with a missing cell is
dropped and the cell is left out of its arm's median.

  llf_refs_read.py --refs2 DIR --final DIR --bar DIR --rulings confirm_rulings.json --out refs_sealed12.json

gnn, wide and bar were run on windows g0, g1 only; reactive and random on g0-g3. So the g0+g1 block has every contrast and the g0-g3 block has reactive vs random only.
"""
import argparse
import glob
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llf_confirm_read as R  # noqa: E402

ALLW = ("g0", "g1", "g2", "g3")
KINDS = {"gnn": R.ARMS["gnn"], "wide": R.ARMS["wide"], "bar": R.BAR, "reactive": "reactive", "random": "random"}
SRC = {"gnn": ("final", "02668ecf", R.SEEDS, ("g0", "g1")), "wide": ("final", "02668ecf", R.SEEDS, ("g0", "g1")), "bar": ("bar", "0ec7cdcb", (0,), ("g0", "g1")),
       "reactive": ("refs2", "0ec7cdcb", (0,), ALLW), "random": ("refs2", "0ec7cdcb", (0,), ALLW)}
CONTRASTS = (("gnn", "reactive"), ("gnn", "random"), ("gnn", "bar"), ("gnn", "wide"), ("bar", "reactive"), ("bar", "random"), ("wide", "reactive"), ("reactive", "random"))


class Grid(R.Grid):
    def seeds(self, arm):
        return SRC[arm][2]

    def raw(self, arm, seed, t, w, rung, stat):
        return self.V.get((self.arms[arm], seed, t, w, rung), {}).get(stat, float("nan"))

    def arm_median(self, arm, seed, rung, mstat):
        key = (arm, rung, mstat, R.WINDOWS)
        if key not in self.med:
            vals = [self.raw(arm, s, t, w, rung, mstat) for s in self.seeds(arm) for t in self.topos for w in R.WINDOWS]
            self.med[key] = st.median(v for v in vals if v == v)
        return self.med[key]

    def cell(self, arm, t, w, rung, stat, reading, only_seed=None):
        ss = self.seeds(arm) if only_seed is None else (only_seed,)
        if any(self.raw(arm, s, t, w, rung, stat) != self.raw(arm, s, t, w, rung, stat) for s in ss):
            return float("nan"), True
        return super().cell(arm, t, w, rung, stat, reading, only_seed)


def load(dirs):
    V, commits = {}, set()
    for a, kind in KINDS.items():
        root, pin, seeds, wins = SRC[a]
        for f in sorted(glob.glob(os.path.join(dirs[root], "*", f"cc40s*__{kind}_s*.summary.json"))):
            m = R.NAME.match(os.path.basename(f))
            if not m or m.group(4) != kind:
                continue
            key = (kind, int(m.group(5)), int(m.group(1)), m.group(2), m.group(3))
            if key in V:
                raise SystemExit(f"FAIL LOUD: {key} appears twice ({f})")
            if key[1] not in seeds or key[2] not in R.IDS or key[3] not in wins:
                raise SystemExit(f"FAIL LOUD: unexpected cell {key} ({f})")
            s = json.load(open(f))
            code = s.get("code") or {}
            if code.get("dirty") is not False or not str(code.get("commit", "")).startswith(pin) or s.get("servers") != 24:
                raise SystemExit(f"FAIL LOUD: {f} ran at {code.get('commit')} dirty={code.get('dirty')} servers={s.get('servers')}, not {pin} clean at 24 servers")
            a360 = ((s.get("steady_state") or {}).get("after_360s") or {}).get("mean")
            if s.get("averageElapsedTime") is None or a360 is None:
                raise SystemExit(f"FAIL LOUD: {f} has no averageElapsedTime / after_360s.mean")
            V[key] = {"whole": float(s["averageElapsedTime"]), "a360": float(a360)}
            commits.add(code["commit"])
    missing = {}
    for a, kind in KINDS.items():
        root, pin, seeds, wins = SRC[a]
        for s in seeds:
            for t in R.IDS:
                for w in wins:
                    for r in R.RUNGS:
                        if (kind, s, t, w, r) not in V:
                            missing.setdefault(a, []).append([s, t, w, r])
    return V, sorted(commits), missing


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("refs2", "final", "bar", "rulings", "out"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"FAIL LOUD: {a.out} exists")
    rul = json.load(open(a.rulings))
    R.check_rulings_cd(rul)
    V, commits, missing = load({"refs2": a.refs2, "final": a.final, "bar": a.bar})
    for arm in ("gnn", "wide", "bar", "reactive"):
        if arm in missing:
            raise SystemExit(f"FAIL LOUD: {arm} cells missing: {missing[arm][:3]}")
    rows, runs = {}, {}
    for block, wins in (("g0+g1", ("g0", "g1")), ("g0-g3", ALLW)):
        R.WINDOWS = wins
        g = Grid(V, R.IDS, rul, KINDS)
        for x, y in CONTRASTS:
            if not all(w in SRC[x][3] and w in SRC[y][3] for w in wins):
                continue
            for stat in R.STATS:
                for rung in R.RUNGS:
                    for reading in ("all", "no_runaway"):
                        c = g.contrast(x, y, rung, stat, reading)
                        rows[f"{block}|{x}_vs_{y}|{stat}|{rung}|{reading}"] = {k: v for k, v in c.items() if k != "per_topology"}
        counts = {}
        for r in g.runaways():
            if r["arm"] in ("reactive", "random") and r["window"] not in wins:
                continue
            k = f"{r['arm']}|{r['rung']}|{r['stat']}"
            counts[k] = counts.get(k, 0) + 1
        runs[block] = counts
    res = {"family": "llf_cd_confirm_v1 descriptive read: static e40 gnn vs reactive and random, sealed 12 (no freeze)", "topologies": list(R.IDS), "commits": commits,
           "rulings": rul, "contrasts": rows, "runaway_counts_by_block": runs, "missing_cells_left_out": missing,
           "windows_run": {k: list(v[3]) for k, v in SRC.items()}}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1, default=str)
    fm = lambda x: "n/a" if x is None else f"{x:+.2f}"
    for key, v in rows.items():
        if "|whole|" in key:
            print(f"{key:62s} {fm(v['median_pct'])}% {v['faster']}/{v['n']} p {None if v['p'] is None else round(v['p'], 4)}")
    print("missing:", json.dumps(missing)); print("runaways:", json.dumps(runs, sort_keys=True))


if __name__ == "__main__":
    main()
