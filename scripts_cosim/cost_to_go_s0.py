#!/usr/bin/env python3
"""cost_to_go_v1 hard stops S0 (headroom) and S1 (stability): job builders and the reader. Offline, held-out steady states.

  cost_to_go_s0.py jobs-s0  OUT.jsonl --argmin TOPS.jsonl --gnn EVAL.json --tables DIR --cells CELLS.txt [--hs 5,15,30] [--max-states N]
  cost_to_go_s0.py jobs-s1  OUT.jsonl --s0 ROWS.jsonl --n-states 100 [--eps 0.001,0.01]
  cost_to_go_s0.py read     --s0 ROWS.jsonl [--s1 ROWS.jsonl]

Per state the plan set is: cd_exactS's own decision on the sweep's candidates (the 'policy' row, the baseline), the next 4
distinct plans by S (from fidelity_cd_plans --argmin-s top_s, the policy plan removed), and the GNN plan (g4-seed1-best,
descriptive). Only states whose corpus label replay made ONE decision are used (fidelity_cd_plans' S is valid there only).
S0: headroom = Q_H(cd_exactS) - min over the plans of Q_H, as a share of the state's Q_H(cd_exactS); STOP if the summed
headroom is below 3 % of the summed Q_H(cd_exactS). S1: the unperturbed plans are forced on the state with every in-flight
remaining time x (1 +- eps); STOP if the median Spearman of the plan advantages is below 0.9 or the best plan changes on more
than 10 % of states.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from statistics import median


def _rows(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def jobs_s0(a):
    tops = {}
    for r in _rows(a.argmin):
        if "error" in r or int(r.get("decisions") or 0) != 1 or not r.get("argmin_s"):
            continue
        tops[r["ds"]] = r["argmin_s"]["top_s"]
    gnn = {os.path.basename(r["dataset_id"]): r.get("combo") for r in json.load(open(a.gnn))["records"]}
    cells = {}
    for line in open(a.cells):
        c, cfg, wl = line.split()
        cells[c] = (cfg, wl)
    hs = [float(x) for x in a.hs.split(",")]
    out, skipped = [], defaultdict(int)
    names = sorted(tops)
    if a.max_states:
        names = names[: a.max_states]
    for ds in names:
        if int((json.load(open(os.path.join(ds, "fidelity_replay.json"))).get("slate") or {}).get("of") or 1) > 1:
            skipped["sub-batch view of a split group (siblings decided live at t0, absent from the label)"] += 1
            continue
        prov = json.load(open(os.path.join(ds, "generation_provenance.json")))["argv"]
        cell = os.path.basename(prov[prov.index("--snapshots") + 1]).replace("shard_", "").replace(".jsonl", "")
        table = os.path.join(a.tables, cell + ".table.json")
        chk = os.path.join(a.tables, cell + ".check.json")
        if not (os.path.exists(chk) and json.load(open(chk)).get("ok")):
            skipped["cell re-run missing or not reproducing the corpus snapshots"] += 1
            continue
        _cfg, wl = cells[cell]
        plans = [("policy", "policy")]
        seen = []
        for k, t in enumerate(tops[ds][:5]):
            if t["plan"] not in seen:
                seen.append(t["plan"])
                plans.append((f"s{k}", t["plan"]))
        g = gnn.get(os.path.basename(ds))
        if g:
            plans.append(("gnn", [list(map(int, p)) for p in g]))
        for H in hs:
            for slot, plan in plans:
                out.append(dict(ds=ds, table=table, workload=wl, H=H, plan=plan, continuation="cd_exacts", eps=0.0, tag=f"s0|{slot}"))
    with open(a.out, "w") as fh:
        for j in out:
            fh.write(json.dumps(j) + "\n")
    print(f"{len(out)} jobs, {len(names)} states with one decision and an S ranking; skipped {dict(skipped)}")


def _by_state(rows):
    st = defaultdict(dict)
    for r in rows:
        if "error" in r:
            continue
        st[(r["ds"], r["H"], r.get("eps") or 0.0)][r["tag"].split("|", 1)[1]] = r
    return st


def _plan_set(slots):
    """{slot: row} -> the 6-plan set: policy, the first 4 S-ranked plans different from it, the GNN plan."""
    pol = slots.get("policy")
    if pol is None or pol.get("plan") is None:
        return None
    keep = {"policy": pol}
    n = 0
    for k in sorted(s for s in slots if s.startswith("s")):
        if slots[k]["plan"] != pol["plan"] and n < 4:
            keep[k] = slots[k]
            n += 1
    if "gnn" in slots:
        keep["gnn"] = slots["gnn"]
    return keep


def jobs_s1(a):
    st = _by_state(_rows(a.s0))
    keys = sorted({k[0] for k in st})
    random.Random(0).shuffle(keys)
    pick = set(keys[: a.n_states])
    out = []
    for (ds, H, eps), slots in sorted(st.items()):
        if ds not in pick or eps:
            continue
        ps = _plan_set(slots)
        if ps is None:
            continue
        for e in [float(x) for x in a.eps.split(",")]:
            for sign in (1, -1):
                for slot, r in ps.items():
                    out.append(dict(ds=ds, table=r_table(r, slots), workload=r_wl(r, slots), H=H, plan=r["plan"], continuation="cd_exacts",
                                    eps=sign * e, tag=f"s1|{slot}"))
    with open(a.out, "w") as fh:
        for j in out:
            fh.write(json.dumps(j) + "\n")
    print(f"{len(out)} jobs over {len(pick)} states")


def r_table(r, slots):
    return r["table"]


def r_wl(r, slots):
    return r["workload"]


def spearman(x, y):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                rk[o[k]] = (i + j) / 2
            i = j + 1
        return rk
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((p - mx) * (q - my) for p, q in zip(rx, ry))
    den = (sum((p - mx) ** 2 for p in rx) * sum((q - my) ** 2 for q in ry)) ** 0.5
    return num / den if den else 1.0


def read(a):
    rows = _rows(a.s0)
    errs = [r for r in rows if "error" in r]
    print(f"S0 rows {len(rows)}, errors {len(errs)}")
    for r in errs[:5]:
        print("  ERROR", r["tag"], r["ds"][-9:], r["H"], r["error"][:200])
    st = _by_state(rows)
    for H in sorted({k[1] for k in st}):
        sets = {k[0]: _plan_set(v) for k, v in st.items() if k[1] == H and not k[2]}
        sets = {k: v for k, v in sets.items() if v and len(v) >= 2}
        share, wins, sum_head, sum_q, cut = [], defaultdict(int), 0.0, 0.0, []
        fallback = 0
        for ds, ps in sets.items():
            q0 = ps["policy"]["q"]
            best_slot = min(ps, key=lambda s: (ps[s]["q"], s != "policy"))
            head = q0 - ps[best_slot]["q"]
            wins[best_slot if head > 1e-9 else "policy"] += 1
            share.append(100 * head / q0 if q0 > 0 else 0.0)
            sum_head += head
            sum_q += q0
            p = ps["policy"]
            if p.get("n_siblings"):
                raise RuntimeError(f"{ds}: the cd_exactS rollout found {p['n_siblings']} sub-batch siblings in a non-split view")
            if p.get("pairs_touching_scored"):
                cut.append(100 * p["pairs_touching_scored_cut"] / p["pairs_touching_scored"])
            fallback += p.get("exact_fallbacks", 0) > 0
        no_gnn = []
        for ds, ps in sets.items():
            sub = {s: r for s, r in ps.items() if s != "gnn"}
            no_gnn.append(ps["policy"]["q"] - min(r["q"] for r in sub.values()))
        n = len(share)
        if not n:
            continue
        s_sorted = sorted(share)
        print(f"\nH = {H:g} s: {n} states, window tasks median {median(ps['policy']['n_window'] for ps in sets.values()):.0f}")
        print(f"  headroom (best of the plan set vs cd_exactS) as a share of the state's Q_H: median {median(share):.3f} %  "
              f"p75 {s_sorted[int(.75 * (n - 1))]:.3f} %  p90 {s_sorted[int(.9 * (n - 1))]:.3f} %")
        print(f"  summed headroom / summed Q_H(cd_exactS): {100 * sum_head / sum_q:.3f} %  (without the GNN slot "
              f"{100 * sum(no_gnn) / sum_q:.3f} %)  -> S0 {'PASS' if 100 * sum_head / sum_q >= 3 else 'STOP'} (bar 3 %)")
        print(f"  states each plan wins (strictly better than cd_exactS; ties to cd_exactS): {dict(sorted(wins.items()))}")
        if cut:
            print(f"  partner pairs of the scored tasks that reach beyond the horizon (stubbed at their live placement): median {median(cut):.1f} %  mean {sum(cut) / len(cut):.1f} %")
        print(f"  states whose cd_exactS rollout hit the plan cap somewhere (expansion fallback): {fallback}")
    if a.s1:
        s1 = _by_state(_rows(a.s1))
        s1err = sum(1 for r in _rows(a.s1) if "error" in r)
        print(f"\nS1 rows errors {s1err}")
        for H in sorted({k[1] for k in s1}):
            for eps in sorted({abs(k[2]) for k in s1 if k[1] == H}):
                rho, flip, n = [], 0, 0
                for (ds, h, e), slots in s1.items():
                    if h != H or abs(e) != eps:
                        continue
                    base = _plan_set(st.get((ds, H, 0.0), {}))
                    if not base:
                        continue
                    common = [s for s in base if s in slots]
                    if len(common) < 3 or "policy" not in common:
                        continue
                    a0 = [base[s]["q"] - base["policy"]["q"] for s in common]
                    a1 = [slots[s]["q"] - slots["policy"]["q"] for s in common]
                    rho.append(spearman(a0, a1))
                    b0 = min(common, key=lambda s: (base[s]["q"], s != "policy"))
                    b1 = min(common, key=lambda s: (slots[s]["q"], s != "policy"))
                    flip += b0 != b1
                    n += 1
                if n:
                    ok = median(rho) >= 0.9 and flip / n <= 0.10
                    print(f"  H {H:g} eps {eps:g} (both signs): {n} perturbed states, Spearman of advantages median {median(rho):.3f} min "
                          f"{min(rho):.3f}, best plan changed {flip}/{n} ({100 * flip / n:.1f} %) -> {'PASS' if ok else 'STOP'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("jobs-s0", "jobs-s1", "read"))
    ap.add_argument("out", nargs="?")
    ap.add_argument("--argmin"); ap.add_argument("--gnn"); ap.add_argument("--tables"); ap.add_argument("--cells")
    ap.add_argument("--hs", default="5,15,30"); ap.add_argument("--max-states", type=int, default=0)
    ap.add_argument("--s0"); ap.add_argument("--s1"); ap.add_argument("--n-states", type=int, default=100)
    ap.add_argument("--eps", default="0.001,0.01")
    a = ap.parse_args()
    {"jobs-s0": jobs_s0, "jobs-s1": jobs_s1, "read": read}[a.mode](a)


if __name__ == "__main__":
    main()
