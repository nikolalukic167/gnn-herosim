#!/usr/bin/env python3
"""prune_g8_s0_v1 offline S screen: over-cap share and the S gain of the pruned optimum against the expansion plan, per rung and order.
No new simulations beyond the decision replay of each existing big_groups_s0_v1 G=8 state (rollout_replay, batch-only).

  prune_g8_screen.py run OUT.jsonl --states <states/top5 dir> --order cost|gnn [--workers 16] [--tag G8]
  prune_g8_screen.py read ROWS.jsonl [ROWS.jsonl ...]

run: for each state of the given G (the dataset's warm_snapshot source tag ends in _G8) replay the cd_exactS decision with the over-cap batches pruned
(cost order: continuation cd_exacts_prc), and record the scheduler's own prune books for that one batch decision. A state whose batch is under the
cap has pg_exact_prune_batches == 0 and is not over-cap.

read: per (rung, order), the states, the over-cap share, and over the over-cap states the gain `pg_exact_prune_gain_fb_seconds` (S of the expansion plan
minus S of the pruned optimum; positive = the pruned optimum is cheaper on S): median, mean, and the shares > 0, = 0, < 0 (a tolerance of 1e-9 s); the same
against the pass plan; and how often the pruned optimum differs from the expansion plan. The registered GO bar (node): median gain > 0 at both
rungs, and the GNN order's median greater than the cost order's at both rungs.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts_cosim"))

CONTINUATION = {"cost": "cd_exacts_prc", "gnn": "cd_exacts_prg"}
TOL = 1e-9


def tag_of(ds: Path):
    tag = json.load(open(ds / "warm_snapshot.json"))["provenance"]["source_tag"]
    _, rung, g = tag.split("_")
    return rung, g


def _state(job):
    try:
        import os

        import rollout_replay

        if job["order"] == "gnn":
            os.environ.update(GNN_MODEL_PATH=job["ckpt"], GNN_PREFIX_ALPHA_KEY="inf", PARTIAL_STATE_CONTRACT="partial_state_v5",
                              PARTIAL_STATE_LOAD_SECONDS="1", PARTIAL_STATE_EXCHANGE_SECONDS="1")
        r = rollout_replay._one(dict(ds=job["ds"], plan="policy", H=-1.0, continuation=CONTINUATION[job["order"]], table=None, eps=0.0))
        if "error" in r:
            return dict(ds=job["ds"], order=job["order"], rung=job["rung"], error=r["error"][:300], tb=r.get("tb", "")[-1500:])
        return dict(ds=job["ds"], order=job["order"], rung=job["rung"], n=r["n_batch"] if "n_batch" in r else len(r["plan"]), prune=r["prune"],
                    exact_batches=r["exact_batches"], exact_fallbacks=r["exact_fallbacks"], wall=r["wall"])
    except Exception as e:
        import traceback

        return dict(ds=job["ds"], order=job["order"], rung=job["rung"], error=f"{type(e).__name__}: {str(e)[:300]}", tb=traceback.format_exc()[-1500:])


def run(a):
    states = sorted(p for p in Path(a.states).iterdir() if p.name.startswith("ds_"))
    jobs = []
    for p in states:
        rung, g = tag_of(p)
        if g == a.tag:
            jobs.append(dict(ds=str(p), rung=rung, order=a.order, ckpt=a.ckpt))
    if a.order == "gnn" and not a.ckpt:
        raise SystemExit("FAIL LOUD: --order gnn needs --ckpt")
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"{len(jobs)} {a.tag} states of {len(states)}", flush=True)
    with get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool, open(a.out, "w") as fh:
        for r in pool.imap_unordered(_state, jobs):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
    print(f"run: {len(jobs)} states -> {a.out}")


def summarise(rows):
    out = {}
    by = defaultdict(list)
    errors = defaultdict(int)
    for r in rows:
        if "error" in r:
            errors[(r["rung"], r["order"])] += 1
        else:
            by[(r["rung"], r["order"])].append(r)
    for (rung, order), rs in sorted(by.items()):
        over = [r for r in rs if int(r["prune"].get("pg_exact_prune_batches") or 0) > 0]
        fb = [float(r["prune"]["pg_exact_prune_gain_fb_seconds"]) / int(r["prune"]["pg_exact_prune_batches"]) for r in over]
        pa = [float(r["prune"]["pg_exact_prune_gain_pass_seconds"]) / int(r["prune"]["pg_exact_prune_batches"]) for r in over]
        differs = sum(1 for r in over if int(r["prune"].get("pg_exact_prune_differs") or 0) > 0)
        sh = lambda v, f: sum(1 for x in v if f(x)) / len(v) if v else None
        out[f"{rung}|{order}"] = {
            "states": len(rs), "errors": errors.get((rung, order), 0), "over_cap_states": len(over), "over_cap_share": len(over) / len(rs) if rs else None,
            "gain_vs_expansion": {"median": st.median(fb) if fb else None, "mean": st.mean(fb) if fb else None, "pos": sh(fb, lambda x: x > TOL),
                                  "zero": sh(fb, lambda x: abs(x) <= TOL), "neg": sh(fb, lambda x: x < -TOL)},
            "gain_vs_pass": {"median": st.median(pa) if pa else None, "mean": st.mean(pa) if pa else None},
            "differs_share": differs / len(over) if over else None,
            "topk_set_differs_share": (sum(int(r["prune"].get("pg_exact_prune_set_differs") or 0) for r in over)
                                       / sum(int(r["prune"]["pg_exact_prune_batches"]) for r in over)) if over else None}
    return out


def read(a):
    rows = [json.loads(l) for f in a.rows for l in open(f) if l.strip()]
    s = summarise(rows)
    print(f"{len(rows)} rows")
    for k, v in s.items():
        print(k, json.dumps(v))
    go = {}
    for rung in ("moderate", "heavy"):
        c, g = s.get(f"{rung}|cost"), s.get(f"{rung}|gnn")
        go[rung] = {"cost_median_gain_pos": bool(c and c["gain_vs_expansion"]["median"] is not None and c["gain_vs_expansion"]["median"] > 0),
                    "gnn_median_gain_pos": bool(g and g["gain_vs_expansion"]["median"] is not None and g["gain_vs_expansion"]["median"] > 0),
                    "gnn_beats_cost": bool(c and g and None not in (c["gain_vs_expansion"]["median"], g["gain_vs_expansion"]["median"])
                                           and g["gain_vs_expansion"]["median"] > c["gain_vs_expansion"]["median"])}
    print("GO bar legs", json.dumps(go))
    print("SCREEN_GO", all(v["gnn_median_gain_pos"] and v["gnn_beats_cost"] for v in go.values()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("out")
    r.add_argument("--states", required=True)
    r.add_argument("--order", choices=sorted(CONTINUATION), required=True)
    r.add_argument("--workers", type=int, default=16)
    r.add_argument("--tag", default="G8")
    r.add_argument("--ckpt", default=None, help="the gnn_eng checkpoint (.pt) for --order gnn")
    r.add_argument("--limit", type=int, default=None, help="first N states only (dry check)")
    r.set_defaults(fn=run)
    d = sub.add_parser("read")
    d.add_argument("rows", nargs="+")
    d.set_defaults(fn=read)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
