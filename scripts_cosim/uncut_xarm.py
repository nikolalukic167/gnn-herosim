"""K1(b) analysis passes on the sealed states (no new physics): for each replayable first-capture state of a K1(b) run,
(A) the standalone-cost rank (1 = best; the declared slate is rank <= 5) of every task's placement in the ILS plan and in the hand plan, from a decision-only pass under the
    hand arm (the re-decided hand plan must equal the recorded one, bit for bit), and
(B) the plan another arm decides on the same snapshot (default cd_exactS_ib0), replayed by the fidelity replay like the hand and ILS plans.
Each row also carries the cross-node partner-pair count of every plan (in-batch pairs of the snapshot). Rows are keyed by state id; joined to the K1(b) rows by `summarize`."""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts_cosim import uncut_ils as U  # noqa: E402


def pair_counts(snap: Dict[str, Any], plan: Sequence[Sequence[int]]) -> Dict[str, Any]:
    gids = [int(r["gid"]) for r in sorted(snap["fidelity"]["batch"], key=lambda r: int(r["gid"]))]
    node = {g: int(p[0]) for g, p in zip(gids, plan)}
    same = cross = 0
    for a, b, _pay in snap["fidelity"]["pairs"]:
        if int(a) in node and int(b) in node:
            same, cross = (same + 1, cross) if node[int(a)] == node[int(b)] else (same, cross + 1)
    return {"same_node": same, "cross_node": cross, "nodes_used": len(set(node.values()))}


def run_job(job: Dict[str, Any], rec: Dict[str, Any], arm_b: str) -> Dict[str, Any]:
    a = U.decision_pass(dict(job, rank_plans={"ils": rec["best"], "hand": rec["hand"]}), "cd_ib0", 0, 5, 200, 30.0)
    if a["hand"] != rec["hand"]:
        raise RuntimeError(f"FAIL LOUD: state {job['state_id']}: the re-decided hand plan differs from the recorded one")
    b = U.decision_pass(dict(job), arm_b, 0, 5, 200, 30.0, check_arm=False)
    rt, _setup, _runs = U.replay_rtts(b["spec"], b["types"], [b["hand"]])
    row = {"state_id": job["state_id"], "bucket": rec["bucket"], "n_batch": rec["n_batch"], "ranks": a["ranks"], "arm_b": arm_b, "plan_b": b["hand"], "rtt_b": rt[0],
           "b_cap_hit": b["exact_cap_hit"], "b_equals_hand": b["hand"] == rec["hand"], "b_equals_ils": b["hand"] == rec["best"], "rtt_hand": rec["rtt_hand"], "rtt_ils": rec["rtt_ils"],
           "pairs": {"hand": pair_counts(a["snap"], rec["hand"]), "ils": pair_counts(a["snap"], rec["best"]), arm_b: pair_counts(a["snap"], b["hand"])}}
    return row


def summarize(rows: Sequence[Dict[str, Any]], arm_b: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    names = [n for n, _l, _h in U.BUCKETS]
    for name in ["pooled"] + names:
        rs = [r for r in rows if name == "pooled" or U._canon_bucket([r])[0]["bucket"] == name]
        if not rs:
            continue
        ranks = [x for r in rs for x in r["ranks"]["ils"]]
        hr = [x for r in rs for x in r["ranks"]["hand"]]
        per_state = [sum(1 for x, _n in r["ranks"]["ils"] if x == 0 or x > 5) / len(r["ranks"]["ils"]) for r in rs]
        reg = lambda key: [(r[key] - r["rtt_ils"]) / r["rtt_ils"] for r in rs]
        rb, rh = reg("rtt_b"), reg("rtt_hand")
        tw = lambda key: (sum(r[key] for r in rs) - sum(r["rtt_ils"] for r in rs)) / sum(r["rtt_ils"] for r in rs)
        out[name] = {"n_states": len(rs), "n_tasks": len(ranks),
                     "ils_outside_top5_task_share": sum(1 for x, _n in ranks if x == 0 or x > 5) / len(ranks), "hand_outside_top5_task_share": sum(1 for x, _n in hr if x == 0 or x > 5) / len(hr),
                     "ils_rank1_task_share": sum(1 for x, _n in ranks if x == 1) / len(ranks), "ils_not_a_candidate_tasks": sum(1 for x, _n in ranks if x == 0),
                     "states_with_any_ils_task_outside_top5": sum(s > 0 for s in per_state) / len(rs), "mean_state_share_outside_top5": st.fmean(per_state),
                     "states_all_ils_tasks_in_top5": sum(s == 0 for s in per_state) / len(rs),
                     f"{arm_b}_mean_regret_pct": 100 * st.fmean(rb), f"{arm_b}_median_pct": 100 * st.median(rb), f"{arm_b}_task_weighted_pct": 100 * tw("rtt_b"),
                     "hand_mean_regret_pct": 100 * st.fmean(rh), "hand_median_pct": 100 * st.median(rh), "hand_task_weighted_pct": 100 * tw("rtt_hand"),
                     f"{arm_b}_equals_hand_share": sum(r["b_equals_hand"] for r in rs) / len(rs), f"{arm_b}_equals_ils_share": sum(r["b_equals_ils"] for r in rs) / len(rs),
                     f"{arm_b}_cap_hit_share": sum(bool(r["b_cap_hit"]) for r in rs) / len(rs),
                     "cross_node_pairs_mean": {k: st.fmean(r["pairs"][k]["cross_node"] for r in rs) for k in ("hand", "ils", arm_b)},
                     "states_with_zero_cross_node_pairs": {k: sum(r["pairs"][k]["cross_node"] == 0 for r in rs) / len(rs) for k in ("hand", "ils", arm_b)}}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("jobs"); r.add_argument("rows", help="the K1(b) rows (hand/best plans and replayed RTTs)"); r.add_argument("out"); r.add_argument("--arm-b", default="cd_exactS_ib0")
    s = sub.add_parser("summarize")
    s.add_argument("xrows"); s.add_argument("--arm-b", default="cd_exactS_ib0"); s.add_argument("--json", default=None)
    a = ap.parse_args()
    if a.cmd == "run":
        recs = {}
        for l in open(a.rows):
            x = json.loads(l)
            if x.get("role", "first") == "first":
                recs[x["state_id"]] = x
        with open(a.out, "a") as fh:
            for line in open(a.jobs):
                job = json.loads(line)
                if job.get("role", "first") != "first" or job["state_id"] not in recs:
                    continue
                fh.write(json.dumps(run_job(job, recs[job["state_id"]], a.arm_b), separators=(",", ":")) + "\n"); fh.flush()
                print(job["state_id"], flush=True)
        return 0
    rows = [json.loads(l) for l in open(a.xrows) if l.strip()]
    res = summarize(rows, a.arm_b)
    print(json.dumps(res, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
