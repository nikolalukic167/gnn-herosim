#!/usr/bin/env python3
"""warm_surge_s0_v1 Phase 0b reader (docs/lineages/warm_surge_s0_v1.md, with the cd-healthy admission amendment). Written before the data lands.

Statistic: mean latency (done - dispatched) over the tasks with dispatched >= t_s, from <cell>.tasks.npy. Secondary windows by dispatch time:
onset [t_s, t_s+300), settled [t_s+360, end), pre-switch warm [600, t_s). The whole-run averageElapsedTime is printed beside the primary.

Order: counts (summaries, failed, completeness); leak check 2 (done < t_s rows identical across k for every arm at seed s); cd-health admission
per (rung, k, topology) from the cd cells (backlog ratio <= 1.5, end - last arrival <= 120 s, requestFailures 0); a topology enters a condition
only if healthy at that k AND at k = 1; a condition stays in the read only with >= 10 topologies.

Contrasts, per rung and k, paired per topology, pct = 100 * (first / second - 1) on the primary statistic (negative = first faster):
  PRIMARY    ib0 vs exactS: the pct at k, and Delta(k) = pct(k) - pct(1) per topology, median over topologies, sign count (topologies where Delta < 0).
  SECONDARY  GNN deficit vs ib0 for SS, NS (seed-mean per topology) and twin-NS, the same Delta(k) = deficit(k) - deficit(1); BEST = the lower-median of SS and NS at that k.
  Descriptive: twin-NS vs NS (the MP contrast), cd vs ib0.
A move counts only if |median Delta| > 1 pp AND sign-consistent in >= 9 of the topologies (of 12). Kill rule printed as flags, interpreted in the node.

  warm_surge_0b_read.py --gate0b <dir> --gate0a <dir> --k1cd <dir> --topos 19420,...,19431 [--t-switch 1200] [--out read.json]
"""
import argparse
import json
import os
import statistics as st
import sys

import numpy as np

RUNGS = ("heavy", "moderate")
KS = (1, 2, 3)
IB0, EX, CD = "cd_exactS_ib0", "cd_exactS", "cd"
SS, NS, TW = "ra_gnn_eng_selfsearch", "ra_gnn_eng_nosplit", "ra_twin_eng_nosplit"
CLASSICAL = (IB0, EX)
LEARNED = (SS, NS, TW)
SEEDS = (1, 2)
NEED, MOVE_PP, SIGN_MIN = 10, 1.0, 9
GROW_MAX, TAIL_MAX = 1.5, 120.0


def stem(d, topo, rung, k, arm, seed):
    return os.path.join(d, f"cc40s{topo}__g0{rung}k{k}__{arm}_s{seed}")


def window_mean(rows, lo, hi=None):
    d, done = rows[:, 0].astype(np.float64), rows[:, 1].astype(np.float64)
    m = d >= lo
    if hi is not None:
        m &= d < hi
    return float((done[m] - d[m]).mean()) if m.any() else None


def pct(a, b):
    return None if a is None or b is None or b == 0 else 100.0 * (a / b - 1.0)


def backlog_ratio(rows, t_s, last):
    grid = np.linspace(t_s, last, 400)
    edges = np.linspace(t_s, last, 5)
    cnt = np.array([np.count_nonzero((rows[:, 0] <= t) & (rows[:, 1] > t)) for t in grid])
    q = [cnt[(grid >= edges[i]) & (grid <= edges[i + 1])].mean() for i in range(4)]
    return q[3] / q[1] if q[1] > 0 else None


def wilcoxon_p(x):
    from scipy.stats import wilcoxon

    x = [v for v in x if v is not None]
    return None if len(x) < 6 or all(v == 0 for v in x) else float(wilcoxon(x, method="exact").pvalue)


def summarize(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return {"n": 0, "median": None, "wins": 0, "p": None}
    return {"n": len(v), "median": st.median(v), "wins": sum(1 for x in v if x < 0), "p": wilcoxon_p(v)}


def moved(s, n_topos):
    return bool(s["median"] is not None and abs(s["median"]) > MOVE_PP and max(s["wins"], s["n"] - s["wins"]) >= min(SIGN_MIN, n_topos))


class Cells:
    def __init__(self, gate0b, gate0a, k1cd, t_s):
        self.g, self.a, self.k1, self.t_s = gate0b, gate0a, k1cd, t_s
        self._rows, self._sum = {}, {}

    def path(self, topo, rung, k, arm, seed):
        d = self.g
        if arm == CD:
            d = self.k1 if k == 1 else self.a
        return stem(d, topo, rung, k, arm, seed)

    def summary(self, *key):
        if key not in self._sum:
            p = self.path(*key) + ".summary.json"
            self._sum[key] = json.load(open(p)) if os.path.exists(p) else None
        return self._sum[key]

    def rows(self, *key):
        if key not in self._rows:
            p = self.path(*key) + ".tasks.npy"
            self._rows[key] = np.load(p) if os.path.exists(p) else None
        return self._rows[key]


def seed_of(arm):
    return 0 if arm in CLASSICAL + (CD,) else None


def arm_mean(c, topo, rung, k, arm, lo, hi=None):
    seeds = (0,) if arm in CLASSICAL + (CD,) else SEEDS
    vals = []
    for s in seeds:
        r = c.rows(topo, rung, k, arm, s)
        if r is None:
            return None
        vals.append(window_mean(r, lo, hi))
    return None if None in vals else sum(vals) / len(vals)


def counts(c, topos):
    out, bad = {}, []
    for rung in RUNGS:
        for k in KS:
            for arm in (*CLASSICAL, CD, *LEARNED):
                seeds = (0,) if arm in CLASSICAL + (CD,) else SEEDS
                n = 0
                for t in topos:
                    for s in seeds:
                        sm = c.summary(t, rung, k, arm, s)
                        if sm is not None:
                            n += 1
                            if sm.get("num_tasks") != 50000 or sm.get("requestFailures") != 0 or sm.get("task_times_rows") != 50000:
                                bad.append((t, rung, k, arm, s))
                out[f"{rung}k{k}|{arm}"] = {"summaries": n, "expected": len(topos) * len(seeds)}
    return out, bad


def leak_check_2(c, topos):
    """done < t_s rows identical across k, per arm, seed, rung and topology (deterministic arms); returns the failures."""
    fails, checked = [], 0
    for rung in RUNGS:
        for arm in (*CLASSICAL, CD, *LEARNED):
            for s in ((0,) if arm in CLASSICAL + (CD,) else SEEDS):
                for t in topos:
                    ref = None
                    for k in KS:
                        r = c.rows(t, rung, k, arm, s)
                        if r is None:
                            continue
                        p = r[r[:, 1] < c.t_s]
                        p = p[np.lexsort((p[:, 1], p[:, 0]))]
                        if ref is None:
                            ref = p
                        elif p.shape != ref.shape or not (p == ref).all():
                            fails.append((t, rung, k, arm, s))
                    checked += ref is not None
    return checked, fails


def health(c, topo, rung, k):
    sm, r = c.summary(topo, rung, k, CD, 0), c.rows(topo, rung, k, CD, 0)
    if sm is None or r is None:
        return False, "missing"
    last = float(sm["arrival_end"]["last_arrival_s"])
    ratio = backlog_ratio(r.astype(np.float64), c.t_s, last)
    tail = float(sm["endTime"]) - last
    why = []
    if ratio is None or ratio > GROW_MAX:
        why.append(f"backlog ratio {ratio}")
    if tail > TAIL_MAX:
        why.append(f"tail {tail:.0f}s")
    if sm["requestFailures"]:
        why.append("requestFailures")
    return not why, "; ".join(why)


def read(c, topos):
    out = {"topologies": topos, "t_switch_s": c.t_s}
    out["counts"], out["incomplete_cells"] = counts(c, topos)
    out["leak2_checked"], out["leak2_fails"] = leak_check_2(c, topos)
    adm = {}
    for rung in RUNGS:
        for k in KS:
            ok = {t: health(c, t, rung, k) for t in topos}
            adm[(rung, k)] = ok
    out["health"] = {f"{r}k{k}": {str(t): v[1] for t, v in d.items() if not v[0]} for (r, k), d in adm.items()}
    cond = {}
    for rung in RUNGS:
        base = {t for t, v in adm[(rung, 1)].items() if v[0]}
        for k in KS:
            keep = sorted(t for t in topos if adm[(rung, k)][t][0] and t in base)
            cond[(rung, k)] = {"topos": keep, "in_read": len(keep) >= NEED}
    out["conditions"] = {f"{r}k{k}": {"topologies": v["topos"], "n": len(v["topos"]), "in_read": v["in_read"]} for (r, k), v in cond.items()}
    res = {}
    windows = {"primary": (c.t_s, None), "onset": (c.t_s, c.t_s + 300), "settled": (c.t_s + 360, None), "pre_warm": (600, c.t_s)}
    for rung in RUNGS:
        for wname, (lo, hi) in windows.items():
            for k in KS:
                cd = cond[(rung, k)]
                if not cd["in_read"]:
                    continue
                keep = cd["topos"]

                def p(a, b, kk):
                    return {t: pct(arm_mean(c, t, rung, kk, a, lo, hi), arm_mean(c, t, rung, kk, b, lo, hi)) for t in keep}

                def delta(a, b):
                    pk, p1 = p(a, b, k), p(a, b, 1)
                    return {t: (None if pk[t] is None or p1[t] is None else pk[t] - p1[t]) for t in keep}

                row = {}
                for name, (a, b) in {"ib0_vs_exactS": (IB0, EX), "cd_vs_ib0": (CD, IB0), "SS_vs_ib0": (SS, IB0), "NS_vs_ib0": (NS, IB0),
                                     "twinNS_vs_ib0": (TW, IB0), "twinNS_vs_NS": (TW, NS)}.items():
                    pk = p(a, b, k)
                    row[name] = {"pct": summarize(list(pk.values())), "per_topology": {str(t): v for t, v in pk.items()}}
                    if k != 1:
                        d = delta(a, b)
                        s = summarize(list(d.values()))
                        s["moved"] = moved(s, len(keep))
                        row[name]["delta_vs_k1"] = s
                        row[name]["delta_per_topology"] = {str(t): v for t, v in d.items()}
                if k != 1:
                    best = min(("SS_vs_ib0", "NS_vs_ib0"), key=lambda n: row[n]["pct"]["median"] if row[n]["pct"]["median"] is not None else 1e9)
                    row["BEST_gnn_vs_ib0"] = {"arm": best, **row[best]["delta_vs_k1"], "pct_median": row[best]["pct"]["median"]}
                res[f"{rung}k{k}|{wname}"] = row
        for k in KS:
            if cond[(rung, k)]["in_read"]:
                whole = {t: pct(c.summary(t, rung, k, IB0, 0)["averageElapsedTime"], c.summary(t, rung, k, EX, 0)["averageElapsedTime"])
                         for t in cond[(rung, k)]["topos"] if c.summary(t, rung, k, IB0, 0) and c.summary(t, rung, k, EX, 0)}
                res[f"{rung}k{k}|whole_run_ib0_vs_exactS"] = summarize(list(whole.values()))
    out["results"] = res
    prim = [(rung, k) for rung in RUNGS for k in (2, 3) if cond[(rung, k)]["in_read"]]
    out["PRIMARY_MOVED"] = {f"{r}k{k}": res[f"{r}k{k}|primary"]["ib0_vs_exactS"]["delta_vs_k1"]["moved"] for r, k in prim}
    out["GNN_MOVED"] = {f"{r}k{k}": moved({kk: res[f"{r}k{k}|primary"]["BEST_gnn_vs_ib0"][kk] for kk in ("median", "wins", "n")}, len(cond[(r, k)]["topos"])) for r, k in prim}
    out["SURGE_NOT_A_LEVER"] = not any(out["PRIMARY_MOVED"].values()) and not any(out["GNN_MOVED"].values())
    return out


def print_report(o):
    f = lambda v, n=2: "-" if v is None else f"{v:+.{n}f}"
    print("-- counts (summaries / expected) per rung+k, arm")
    for k, v in o["counts"].items():
        print(f"  {k:36s} {v['summaries']}/{v['expected']}{'' if v['summaries'] == v['expected'] else '   INCOMPLETE'}")
    print(f"-- incomplete/bad cells (num_tasks, failures, rows): {o['incomplete_cells']}")
    print(f"-- leak check 2: {o['leak2_checked']} (rung, arm, seed, topology) series compared, failures {o['leak2_fails']}")
    print("-- cd-health drops:", {k: v for k, v in o["health"].items() if v})
    print("-- conditions:", {k: (v["n"], v["in_read"]) for k, v in o["conditions"].items()})
    for key, row in o["results"].items():
        if key.endswith("whole_run_ib0_vs_exactS"):
            print(f"{key}: median {f(row['median'])} % wins {row['wins']}/{row['n']} p={row['p']}")
            continue
        print(f"== {key}")
        for name, v in row.items():
            if name == "BEST_gnn_vs_ib0":
                print(f"   {name}: {v['arm']} pct_median {f(v['pct_median'])}  Delta(k) median {f(v['median'])} wins {v['wins']}/{v['n']} moved={v.get('moved')}")
                continue
            p = v["pct"]
            line = f"   {name:16s} pct median {f(p['median'])} wins {p['wins']}/{p['n']} p={p['p']}"
            if "delta_vs_k1" in v:
                d = v["delta_vs_k1"]
                line += f" | Delta(k)-Delta(1) median {f(d['median'])} wins {d['wins']}/{d['n']} moved={d['moved']}"
            print(line)
    print(f"-- PRIMARY_MOVED {o['PRIMARY_MOVED']}   GNN_MOVED {o['GNN_MOVED']}   SURGE_NOT_A_LEVER {o['SURGE_NOT_A_LEVER']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for a in ("--gate0b", "--gate0a", "--k1cd", "--topos"):
        ap.add_argument(a, required=True)
    ap.add_argument("--t-switch", type=float, default=1200.0)
    ap.add_argument("--out")
    a = ap.parse_args()
    topos = [int(x) for x in a.topos.split(",")]
    if len(topos) != 12:
        raise SystemExit("FAIL LOUD: expected 12 topologies")
    o = read(Cells(a.gate0b, a.gate0a, a.k1cd, a.t_switch), topos)
    if a.out:
        json.dump(o, open(a.out, "w"), indent=1)
    print_report(o)
    if o["leak2_fails"] or o["incomplete_cells"]:
        print("LEAK/COMPLETENESS FAILURE: nothing above is a result")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
