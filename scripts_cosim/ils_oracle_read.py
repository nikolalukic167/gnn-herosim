#!/usr/bin/env python3
"""uncut_groups_s0_v1 `ils_oracle_live` read, under the amended read filed at 9f119675 (docs/lineages/uncut_groups_s0_v1.md),
following SCIENTIST's read-rule checklist (items 1-14).

  ils_oracle_read.py --oracle <ils12 OUT> --oracle-pin 5109d07d --ref <gate_k0> --topos 19461,...,19466
                     [--exact10 <OUT> --exact10-pin P --exact10-identity-pass] [--descriptive] [--out read.json]

Contrast A: ils_oracle_live vs cd_ib0. Contrast B: ils_oracle_exact10 vs cd_exactS_ib0. Reference arms and `cd` come from --ref (gate_k0).
g0 only, seed 0, k = 0. Paired per topology x rung (one window: no median over windows).

Keys, per cell summary:
  whole-run mean latency    averageElapsedTime
  after-360 s mean latency  steady_state.after_360s.mean        (decides the label)
  exchange s/task           totalPeerExchangeTime / num_tasks   } WHOLE-RUN ONLY: the per-task file (.tasks.npy) holds latency and arrival time,
  queue s/task              averageQueueTime                    } nothing finer, so there is no after-360 s mechanism; the whole-run one is used
  cap hits by bucket        pg_ils_batches_<b>, pg_ils_cap_hits_<b>, b in 11_50 / 51_100 / 101_226
  ILS work                  pg_ils_batches, pg_ils_tasks_moved, pg_ils_gain_seconds, pg_ils_budget_stops
  served plan != bar plan   no such counter in these summaries: printed "not available"
  provenance                code.commit (must start with the pin), code.dirty (must be false)
  collapse                  mean > 3 x the same topology x rung's `cd` mean, whole-run and after 360 s tested separately
  tail flag                 endTime - arrival_end.last_arrival_s > 120 s
  WALL-INFEASIBLE           a .failed.json whose `why` starts "watchdog: projected": reported, never imputed, never rerun

Per contrast x rung x statistic: paired % = 100 (arm - bar) / bar per topology; the median over topologies present; the sign count (a missing
topology counts as not faster); an exact two-sided Wilcoxon (r1_attribution_v1_read.exact_wilcoxon).
Mechanism per contrast x rung: holds when -median(d exchange) > median(d queue), medians over topologies.

Labels (after 360 s decides), applied only when every oracle cell (and every exact10 cell unless B-NOT-RUN) is in, as a summary or a failure:
  TRANSFERS     A <= -6 % at both rungs, >= 5/6 topologies faster at both rungs, B <= -2 % at both rungs, mechanism holds in A and B at both rungs
  MYOPIC-LIVE   A > -2 % at either rung, OR B > 0 % at either rung, OR >= 2/6 collapses (either statistic) in either oracle arm at a rung,
                OR the mechanism fails in either contrast at a rung
  PARTIAL       otherwise
  + CEILING-TRUNCATED when, for either oracle arm at either rung, more than 20 % of its > 50-task batches (pooled within arm x rung) hit the cap
  + B-NOT-RUN   exact10 absent, incomplete or its identity not passed: the B and B-mechanism clauses are dropped from TRANSFERS and MYOPIC-LIVE.
--descriptive, or any oracle cell missing, prints everything with no label ("interim, n cells of 12").
"""
import argparse
import hashlib
import json
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r1_attribution_v1_read import exact_wilcoxon  # noqa: E402

AMENDMENT_SHA = "9f119675"
MECH_NOTE = "mechanism read on whole-run (after-360 s unavailable in summaries; decided before any ils12 cell)"
RUNGS = ("moderate", "heavy")
WIN = "g0"
ORACLE, EXACT10, BAR, EXS, CD = "ils_oracle_live", "ils_oracle_exact10", "cd_ib0", "cd_exactS_ib0", "cd"
STATS = ("whole", "after360")
COLLAPSE_X, TAIL_S, CAP_TRUNC = 3.0, 120.0, 0.20
A_TRANSFER, A_MIN_WINS, B_TRANSFER, A_MYOPIC, B_MYOPIC, COLLAPSE_MAX = -6.0, 5, -2.0, -2.0, 0.0, 2
BUCKETS = ("11_50", "51_100", "101_226")
BIG = ("51_100", "101_226")
NAME = re.compile(r"^cc40s(\d+)__(g\d)(moderate|heavy)__(.+)_s(\d+)\.(summary|failed)\.json$")


def load(dirs):
    cells, failed, paths = {}, {}, {}
    for d in [x for x in (dirs or "").split(",") if x]:
        for f in sorted(os.listdir(d)):
            m = NAME.match(f)
            if not m or m.group(2) != WIN:
                continue
            key = (m.group(4), int(m.group(1)), m.group(3))
            if key in cells or key in failed:
                raise SystemExit(f"FAIL LOUD: duplicate cell {key} in {dirs}")
            p = os.path.join(d, f)
            with open(p, "rb") as fh:
                raw = fh.read()
            (cells if m.group(6) == "summary" else failed)[key] = json.loads(raw)
            paths[(key, m.group(6))] = (p, hashlib.md5(raw).hexdigest())
    return cells, failed, paths


def check_pin(cells, arm, pin):
    for (a, t, r), s in cells.items():
        if a != arm:
            continue
        code = s.get("code") or {}
        if not str(code.get("commit", "")).startswith(pin) or code.get("dirty") is not False:
            raise SystemExit(f"FAIL LOUD: {arm} {t} {r} code {code.get('commit')} dirty={code.get('dirty')} is not the pin {pin}")


def means(s):
    a = ((s.get("steady_state") or {}).get("after_360s") or {}).get("mean")
    if a is None:
        raise SystemExit(f"FAIL LOUD: {s.get('cell')} has no steady_state.after_360s.mean")
    return {"whole": float(s["averageElapsedTime"]), "after360": float(a)}


def mech(s):
    return {"exchange": float(s["totalPeerExchangeTime"]) / int(s["num_tasks"]), "queue": float(s["averageQueueTime"])}


def ils(s):
    c = s.get("schedulerCounters") or {}
    if "pg_ils_batches" not in c:
        return None
    by = {b: {"batches": int(c[f"pg_ils_batches_{b}"]), "cap_hits": int(c[f"pg_ils_cap_hits_{b}"])} for b in BUCKETS}
    return {"by_bucket": by, "batches": int(c["pg_ils_batches"]), "tasks_moved": int(c["pg_ils_tasks_moved"]),
            "gain_seconds": float(c["pg_ils_gain_seconds"]), "budget_stops": int(c["pg_ils_budget_stops"]),
            "served_differs_from_bar": "not available"}


def cell_row(s, ref, cd):
    m, r, ms, mr = means(s), means(ref), mech(s), mech(ref)
    c = means(cd) if cd is not None else None
    return {"pct": {k: 100.0 * (m[k] - r[k]) / r[k] for k in STATS}, "mean": m, "bar_mean": r,
            "d_exchange": ms["exchange"] - mr["exchange"], "d_queue": ms["queue"] - mr["queue"],
            "exchange": ms["exchange"], "queue": ms["queue"], "bar_exchange": mr["exchange"], "bar_queue": mr["queue"],
            "collapse": {k: (m[k] > COLLAPSE_X * c[k]) if c else None for k in STATS},
            "tail_flag": float(s["endTime"]) - float(s["arrival_end"]["last_arrival_s"]) > TAIL_S, "ils": ils(s)}


def contrast(arm, bar, oc, of, rc, topos):
    out = {}
    for rung in RUNGS:
        per, missing, fails = {}, [], []
        for t in topos:
            s, ref, cd = oc.get((arm, t, rung)), rc.get((bar, t, rung)), rc.get((CD, t, rung))
            if s is None:
                if (arm, t, rung) in of:
                    why = str(of[(arm, t, rung)].get("why"))[:120]
                    fails.append({"topo": t, "why": why, "wall_infeasible": why.startswith("watchdog: projected")})
                else:
                    missing.append(t)
                continue
            if ref is None:
                raise SystemExit(f"FAIL LOUD: no {bar} reference cell for {t} {rung} in --ref")
            per[t] = cell_row(s, ref, cd)
        agg = {"arm": arm, "bar": bar, "n": len(per), "of": len(topos), "missing": missing, "failed": fails,
               "wall_infeasible": sum(f["wall_infeasible"] for f in fails)}
        for k in STATS:
            v = [r["pct"][k] for r in per.values()]
            agg[k] = {"median_pct": st.median(v) if v else None, "faster": sum(1 for x in v if x < 0), "of": len(topos),
                      "wilcoxon_p": exact_wilcoxon(v) if v else None,
                      "per_topology": {str(t): round(r["pct"][k], 3) for t, r in sorted(per.items())}}
        dx, dq = [r["d_exchange"] for r in per.values()], [r["d_queue"] for r in per.values()]
        agg["mechanism_whole_run"] = {"median_d_exchange": st.median(dx) if dx else None, "median_d_queue": st.median(dq) if dq else None,
                                      "holds": bool(dx) and -st.median(dx) > st.median(dq),
                                      "per_topology": {str(t): {"d_exchange": round(r["d_exchange"], 4), "d_queue": round(r["d_queue"], 4)}
                                                       for t, r in sorted(per.items())},
                                      "note": "whole-run only; no after-360 s exchange/queue in the per-task file"}
        agg["collapse"] = {k: sum(1 for r in per.values() if r["collapse"][k]) for k in STATS}
        agg["collapse"]["any"] = sum(1 for r in per.values() if any(r["collapse"].values()))
        agg["tail_flags"] = sum(1 for r in per.values() if r["tail_flag"])
        il = [r["ils"] for r in per.values() if r["ils"]]
        if il:
            byb = {b: {"batches": sum(x["by_bucket"][b]["batches"] for x in il), "cap_hits": sum(x["by_bucket"][b]["cap_hits"] for x in il)} for b in BUCKETS}
            for b in BUCKETS:
                byb[b]["share"] = byb[b]["cap_hits"] / byb[b]["batches"] if byb[b]["batches"] else None
            hits, tot = sum(byb[b]["cap_hits"] for b in BIG), sum(byb[b]["batches"] for b in BIG)
            agg["cap"] = {"by_bucket": byb, "gt50_hits": hits, "gt50_batches": tot, "gt50_share": hits / tot if tot else None,
                          "truncated": bool(tot) and hits / tot > CAP_TRUNC}
            agg["ils_work"] = {k: sum(x[k] for x in il) for k in ("batches", "tasks_moved", "gain_seconds", "budget_stops")}
            agg["ils_work"]["served_differs_from_bar"] = "not available"
        else:
            agg["cap"], agg["ils_work"] = None, None
        out[rung] = agg
    return out


def complete(arm, oc, of, topos):
    return all((arm, t, r) in oc or (arm, t, r) in of for t in topos for r in RUNGS)


def label(A, B, b_run):
    a = {r: A[r]["after360"]["median_pct"] for r in RUNGS}
    if any(v is None for v in a.values()):
        return "NO-DATA"
    myopic = any(a[r] > A_MYOPIC for r in RUNGS) or any(A[r]["collapse"]["any"] >= COLLAPSE_MAX for r in RUNGS) \
        or any(not A[r]["mechanism_whole_run"]["holds"] for r in RUNGS)
    transfers = all(a[r] <= A_TRANSFER and A[r]["after360"]["faster"] >= A_MIN_WINS for r in RUNGS) \
        and all(A[r]["mechanism_whole_run"]["holds"] for r in RUNGS)
    arms = [A]
    if b_run:
        b = {r: B[r]["after360"]["median_pct"] for r in RUNGS}
        myopic = myopic or any(b[r] is None or b[r] > B_MYOPIC for r in RUNGS) or any(B[r]["collapse"]["any"] >= COLLAPSE_MAX for r in RUNGS) \
            or any(not B[r]["mechanism_whole_run"]["holds"] for r in RUNGS)
        transfers = transfers and all(b[r] is not None and b[r] <= B_TRANSFER for r in RUNGS) and all(B[r]["mechanism_whole_run"]["holds"] for r in RUNGS)
        arms.append(B)
    lab = "MYOPIC-LIVE" if myopic else ("TRANSFERS" if transfers else "PARTIAL")
    if any(X[r]["cap"] and X[r]["cap"]["truncated"] for X in arms for r in RUNGS):
        lab += " + CEILING-TRUNCATED"
    if not b_run:
        lab += " + B-NOT-RUN (B and B-mechanism clauses dropped from TRANSFERS and MYOPIC-LIVE)"
    return lab


def table(res):
    lines = [f"amended read {AMENDMENT_SHA}; oracle pin {res['oracle_pin']}; {res['oracle_cells_in']}; B {res['B_status']}"]
    for name in ("A", "B", "oracle_vs_cd_exactS_ib0 (confounded on <= 10 batches)", "exact10_vs_cd_ib0 (confounded on <= 10 batches)"):
        X = res.get(name)
        if not X:
            continue
        for r in RUNGS:
            x = X[r]
            cap = x["cap"]["gt50_share"] if x["cap"] else None
            lines.append(f"{name[:40]:40s} {r:8s} n={x['n']}/{x['of']} after360 {x['after360']['median_pct']} ({x['after360']['faster']} faster, "
                         f"p={x['after360']['wilcoxon_p']}) whole {x['whole']['median_pct']} | dExch {x['mechanism_whole_run']['median_d_exchange']} "
                         f"dQueue {x['mechanism_whole_run']['median_d_queue']} holds={x['mechanism_whole_run']['holds']} | collapse {x['collapse']} "
                         f"tail {x['tail_flags']} | cap>50 {cap} | WI {x['wall_infeasible']} missing {x['missing']}")
    lines.append(f"LABEL: {res['label']}  [{MECH_NOTE}]")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--oracle-pin", required=True)
    ap.add_argument("--exact10", default="")
    ap.add_argument("--exact10-pin", default="")
    ap.add_argument("--exact10-identity-pass", action="store_true")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--topos", required=True)
    ap.add_argument("--descriptive", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    topos = [int(x) for x in a.topos.split(",")]
    print(f"amended read {AMENDMENT_SHA}; oracle pin {a.oracle_pin}")
    oc, of, op = load(a.oracle)
    check_pin(oc, ORACLE, a.oracle_pin)
    if a.exact10:
        ec, ef, ep = load(a.exact10)
        if a.exact10_pin:
            check_pin(ec, EXACT10, a.exact10_pin)
        oc.update(ec)
        of.update(ef)
    rc, _, rp = load(a.ref)
    res = {"read": f"ils_oracle_live (amended {AMENDMENT_SHA})", "oracle_pin": a.oracle_pin, "topos": topos,
           "bar_cells": {f"{k[0][0]}|{k[0][1]}|{k[0][2]}": {"path": v[0], "md5": v[1]} for k, v in sorted(rp.items())
                         if k[1] == "summary" and k[0][0] in (BAR, EXS, CD) and k[0][1] in topos},
           "A": contrast(ORACLE, BAR, oc, of, rc, topos)}
    n_in = sum(1 for t in topos for r in RUNGS if (ORACLE, t, r) in oc or (ORACLE, t, r) in of)
    res["oracle_cells_in"] = f"interim, {n_in} cells of {len(topos) * len(RUNGS)}" if n_in < len(topos) * len(RUNGS) else f"{n_in} cells of {n_in}"
    res["oracle_vs_cd_exactS_ib0 (confounded on <= 10 batches)"] = contrast(ORACLE, EXS, oc, of, rc, topos)
    b_run = bool(a.exact10) and a.exact10_identity_pass and complete(EXACT10, oc, of, topos)
    if a.exact10:
        res["B"] = contrast(EXACT10, EXS, oc, of, rc, topos)
        res["exact10_vs_cd_ib0 (confounded on <= 10 batches)"] = contrast(EXACT10, BAR, oc, of, rc, topos)
    res["B_status"] = "RUN" if b_run else "B-NOT-RUN"
    res["mechanism_note"] = MECH_NOTE
    if a.descriptive or not complete(ORACLE, oc, of, topos):
        res["label"] = f"DESCRIPTIVE ({res['oracle_cells_in']}); no label"
    else:
        res["label"] = label(res["A"], res.get("B"), b_run)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1, default=str)
    print(table(res))
    return res


if __name__ == "__main__":
    main(sys.argv[1:])
