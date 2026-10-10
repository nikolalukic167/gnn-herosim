#!/usr/bin/env python3
"""cost_to_go_v1 rung 1: the hand V and the reader that fits its single lambda.

V(state, plan) prices what a plan leaves behind for the next H seconds, beyond the batch's own cost (that is already in S).
Two terms in seconds, no free weights, one lambda that scales their sum in the policy  argmin_plans  S + lambda * V :

  load  = sum_p need[p] * load_after[p]
          load_after[p]: platform p's committed backlog after the plan commits (queue drain + in-flight remainder + the plan's
          own service on p). need[p]: expected arrivals in (t0, t0 + H] that would be served on p, i.e. for each task type its
          arrival rate * H spread evenly over the platforms that can serve it (p's share is 1/|candidates(type)| if p is one).
  cold  = sum over replicas the plan leaves idle whose idle time + H reaches the scale-in time:
          rate(type) * H * (1 / replicas(type)) * cold_cost(type)   (a type's next arrival lands on a scaled-in replica with
          probability ~ its share, and pays that type's cold start).

Arrival rates use trace events with timestamp <= t0 only; nothing after t0 is read (tested).

The V target is PAIRED, never a separate return: A_k = Q_H(plan_k) - Q_H(policy plan) on the same (ds, H) future, where the policy
plan is cd_exactS's own decision. The reader never forms a score from a single plan's return.

  cost_to_go_handv.py read --s0 ROWS.jsonl --tops TOPS.jsonl --features FEATS.jsonl
                           --fit-topos 16301,16302 --eval-topos 16251,16252 --gate-topos 16251,16252,... [--lookback 60]

lambda is fit on --fit-topos only, which must be disjoint from --eval-topos and from --gate-topos. S0's rows are held-out
topologies only (S6, 2026-10-10), so a fit set needs states rolled on other topologies: the reader refuses an empty or overlapping
fit set rather than fit on what it reports.

FEATURES (one JSON row per (ds, slot), written by cost_to_go_handv_features.py from a replay of the batch decision):
  {ds, slot, t0, workload (the cell's trace JSON path), load_after: {"node:platform": seconds},
   type_platforms: {type: ["node:platform", ...]}, replicas: [{key, type, idle_s, used}],
   cold_cost_s: {type: seconds}, scale_in_after_s: seconds}
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cost_to_go_s0 import _plan_set, spearman  # noqa: E402

CELL_RE = re.compile(r"cc40s(\d+)_(heavy|moderate)_(g\d+)")
HS = (5.0, 15.0, 30.0)


def event_type(ev):
    return next(iter(ev["application"]["dag"]))


def type_rates(events, t0, lookback_s):
    """Arrivals per second by task type from events with timestamp <= t0, over (t0 - lookback, t0]. No event after t0 is read."""
    lo = max(0.0, t0 - lookback_s)
    span = t0 - lo
    if span <= 0:
        return {}
    n = defaultdict(int)
    for ev in events:
        ts = float(ev["timestamp"])
        if lo < ts <= t0:
            n[event_type(ev)] += 1
    return {t: c / span for t, c in n.items()}


def hand_v(feat, rates, H):
    """{'v', 'load', 'cold'} in seconds for one plan's post-commit features and the pre-t0 type rates."""
    need = defaultdict(float)
    for typ, keys in feat["type_platforms"].items():
        if not keys:
            continue
        exp = rates.get(typ, 0.0) * H
        for k in keys:
            need[k] += exp / len(keys)
    load = sum(need[k] * float(v) for k, v in feat["load_after"].items())
    n_rep = defaultdict(int)
    for r in feat["replicas"]:
        n_rep[r["type"]] += 1
    cold = 0.0
    for r in feat["replicas"]:
        if r["used"] or float(r["idle_s"]) + H < float(feat["scale_in_after_s"]):
            continue
        cold += rates.get(r["type"], 0.0) * H / n_rep[r["type"]] * float(feat["cold_cost_s"].get(r["type"], 0.0))
    return {"v": load + cold, "load": load, "cold": cold}


def choose(cands, lam):
    """cands: {slot: (dS, dV)} paired against the policy plan (policy is (0, 0)). argmin dS + lam*dV; ties go to the policy, then slot order."""
    return min(cands, key=lambda s: (cands[s][0] + lam * cands[s][1], s != "policy", s))


def build_states(rows, tops_by_ds, feats, rates_of, cell_of):
    """-> {(ds, H): {'cell','cands':{slot:(dS,dV)},'dq':{slot:A}}} over the 6-plan set minus the GNN slot (its S is not recorded)."""
    by = defaultdict(dict)
    for r in rows:
        if "error" in r:
            continue
        if float(r.get("eps") or 0.0) != 0.0 or r.get("continuation") != "cd_exacts":
            raise ValueError(f"{r['ds']}: only eps=0 / cd_exacts rows pair on the same future, got eps={r.get('eps')} {r.get('continuation')}")
        by[(r["ds"], float(r["H"]))][r["tag"].split("|", 1)[1]] = r
    out = {}
    for (ds, H), slots in by.items():
        ps = _plan_set(slots)
        if ps is None or ds not in tops_by_ds:
            continue
        top = tops_by_ds[ds]
        s_of = {"policy": float(top["s"])}
        for slot in ps:
            if slot.startswith("s") and slot != "policy":
                s_of[slot] = float(top["top_s"][int(slot[1:])]["s"])
        keep = [s for s in ps if s in s_of]
        if len(keep) < 2 or any((ds, s) not in feats for s in keep):
            continue
        rates = rates_of(ds)
        vv = {s: hand_v(feats[(ds, s)], rates, H)["v"] for s in keep}
        q0 = ps["policy"]["q"]
        out[(ds, H)] = {"cell": cell_of(ds),
                        "cands": {s: (s_of[s] - s_of["policy"], vv[s] - vv["policy"]) for s in keep},
                        "dq": {s: ps[s]["q"] - q0 for s in keep}}
    return out


def lam_grid():
    return [0.0] + [10 ** (e / 4.0) for e in range(-16, 17)]


def score(states, lam):
    """Held-out numbers for one lambda over a set of states."""
    n = hit = hit0 = 0
    got = head = 0.0
    for st in states:
        c, dq = st["cands"], st["dq"]
        pick = choose(c, lam)
        best = min(dq, key=lambda s: (dq[s], s != "policy", s))
        n += 1
        hit += pick == best
        hit0 += choose(c, 0.0) == best
        got += -dq[pick]
        head += -dq[best]
    return {"n": n, "hit": hit / n if n else float("nan"), "hit_lambda0": hit0 / n if n else float("nan"),
            "captured_q": got, "headroom_q": head, "captured_share": got / head if head > 0 else float("nan")}


def fit_lambda(fit_states):
    """lambda maximising summed Q_H saved on FIT states (ties -> smallest), and the near-tie-breaker lambda: the largest on the
    grid that changes the lambda=0 pick on at most 10 % of fit states."""
    grid = lam_grid()
    best_l, best_v = None, None
    for lam in grid:
        v = score(fit_states, lam)["captured_q"]
        if best_v is None or v > best_v + 1e-12:
            best_l, best_v = lam, v
    small = 0.0
    for lam in grid:
        changed = sum(choose(s["cands"], lam) != choose(s["cands"], 0.0) for s in fit_states)
        if fit_states and changed / len(fit_states) <= 0.10:
            small = lam
    return best_l, small


def spearmans(states):
    """Pooled and mean per-state Spearman of dV vs dQ over the non-policy slots."""
    xs, ys, per = [], [], []
    for st in states:
        slots = [s for s in st["cands"] if s != "policy"]
        x = [st["cands"][s][1] for s in slots]
        y = [st["dq"][s] for s in slots]
        xs += x
        ys += y
        if len(slots) >= 3 and len(set(y)) > 1 and len(set(x)) > 1:
            per.append(spearman(x, y))
    return {"pooled": spearman(xs, ys) if len(xs) > 2 else float("nan"),
            "per_state_mean": sum(per) / len(per) if per else float("nan"), "per_state_n": len(per)}


def check_split(fit, ev, gate):
    fit, ev, gate = set(fit), set(ev), set(gate)
    if not fit:
        raise ValueError("FAIL LOUD: empty --fit-topos; lambda is fit on states disjoint from the ones reported")
    if not ev:
        raise ValueError("FAIL LOUD: empty --eval-topos")
    for name, other in (("eval", ev), ("gate", gate)):
        both = fit & other
        if both:
            raise ValueError(f"FAIL LOUD: --fit-topos overlaps {name} topologies {sorted(both)}")


def topo_of_cell(cell):
    m = CELL_RE.search(cell)
    if not m:
        raise ValueError(f"cannot read a topology from cell {cell!r}")
    return int(m.group(1))


def read(a):
    fit_t = [int(x) for x in a.fit_topos.split(",") if x]
    ev_t = [int(x) for x in a.eval_topos.split(",") if x]
    gate_t = [int(x) for x in (a.gate_topos or "").split(",") if x]
    check_split(fit_t, ev_t, gate_t)
    rows = [json.loads(l) for l in open(a.s0) if l.strip()]
    tops = {}
    for l in open(a.tops):
        r = json.loads(l)
        if "error" not in r and r.get("argmin_s"):
            tops[r["ds"]] = r["argmin_s"]
    feats = {}
    for l in open(a.features):
        r = json.loads(l)
        feats[(r["ds"], r["slot"])] = r
    wl_cache = {}

    def cell_of(ds):
        argv = json.load(open(os.path.join(ds, "generation_provenance.json")))["argv"]
        return os.path.basename(argv[argv.index("--snapshots") + 1])

    def rates_of(ds):
        f = feats[(ds, "policy")]
        if ds not in wl_cache:
            wl_cache[ds] = json.load(open(f["workload"]))["events"]
        return type_rates(wl_cache[ds], float(f["t0"]), a.lookback)

    states = build_states(rows, tops, feats, rates_of, cell_of)
    print(f"{len(states)} (state, H) pairs with a paired S, V and Q_H for >= 2 plans")
    for H in sorted({k[1] for k in states}):
        fit = [s for k, s in states.items() if k[1] == H and topo_of_cell(s["cell"]) in fit_t]
        ev = [s for k, s in states.items() if k[1] == H and topo_of_cell(s["cell"]) in ev_t]
        print(f"\nH = {H:g} s: fit states {len(fit)}, eval states {len(ev)}")
        if not fit or not ev:
            print("  no fit or no eval states at this H: nothing reported")
            continue
        lam, small = fit_lambda(fit)
        print(f"  lambda fit on fit topologies only: best {lam:.4g}, near-tie breaker {small:.4g}")
        sp = spearmans(ev)
        print(f"  held-out Spearman of dV vs dQ_H: pooled {sp['pooled']:+.3f}, per-state mean {sp['per_state_mean']:+.3f} (n={sp['per_state_n']})")
        for name, l in (("lambda=0 (argmin S)", 0.0), ("small lambda (tie breaker)", small), ("best lambda", lam)):
            m = score(ev, l)
            print(f"  [{name}, lambda {l:.4g}] picks the Q_H-best plan on {100 * m['hit']:.1f} % of states "
                  f"(argmin S alone {100 * m['hit_lambda0']:.1f} %); Q_H saved vs policy {m['captured_q']:.2f} s of {m['headroom_q']:.2f} s "
                  f"available ({100 * m['captured_share']:.1f} %)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("read")
    r.add_argument("--s0", required=True)
    r.add_argument("--tops", required=True)
    r.add_argument("--features", required=True)
    r.add_argument("--fit-topos", required=True)
    r.add_argument("--eval-topos", required=True)
    r.add_argument("--gate-topos", default="")
    r.add_argument("--lookback", type=float, default=60.0)
    a = ap.parse_args()
    read(a)


if __name__ == "__main__":
    main()
