#!/usr/bin/env python3
"""workload_fix_v1 W2+W3: wired+wired share of peer transfers under random pairing, against each arm's observed share.

Baseline, exact (a count, not a simulation), per (topology, rung): every peer pair {i, j} of the rung's four window
workloads has task types (a, b). Each task's server is drawn uniformly and independently from S(type), the servers
that carry at least one platform listed for that type in data/nofs-ids/task-types.json. A transfer exists only when the
two servers differ (`Platform._peer_exchange_time` skips a co-located peer), so over the pair

    P(transfer)           = (|S_a| |S_b| - |S_a & S_b|) / (|S_a| |S_b|)
    P(transfer, wired+wired) = (|W_a| |W_b| - |W_a & W_b|) / (|S_a| |S_b|),   W = the wired members of S

and the baseline share is the sum of the second over pairs divided by the sum of the first. Observed: an arm's
`wired+wired` transfers over all its transfers, pooled over the topology's four windows, from the summaries'
`peerExchangeByAccessClass`. Per arm and rung: observed minus baseline per topology (percentage points), the median
over topologies and the number of topologies above the baseline. Descriptive; no test.

  workload_fix_v1_w3_wired_baseline.py --inputs <stage_w23 inputs> --gate-dir <gate> --selected <selected.json> --out <json>
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import workload_fix_v1_read as R  # noqa: E402
import workload_fix_v1_w3_build as B  # noqa: E402

WINDOWS = {w: f"grounded_{w}_n50000.json" for w in R.WINDOWS}


def task_types(events: List[dict]) -> List[str]:
    """Type of each global task id, in the order Orchestrator.create_application assigns ids (task_id += len(app.tasks))."""
    out: List[str] = []
    for ev in events:
        dag = ev["application"]["dag"]
        if len(dag) != 1:
            raise SystemExit(f"FAIL LOUD: event with {len(dag)} tasks; the id order of a multi-task application is not handled here")
        out.append(next(iter(dag)))
    return out


def server_sets(infra: dict, task_types_json: dict, types: Set[str]) -> Dict[str, Dict[str, Set[str]]]:
    classes = {n: s["class"] for n, s in infra["link_topology"]["access_classes"].items()}
    servers = [n for n in infra["nodes"] if not n["node_name"].startswith("client_node")]
    out: Dict[str, Dict[str, Set[str]]] = {}
    for ty in sorted(types):
        listed = set(task_types_json[ty]["platforms"])
        s = {n["node_name"] for n in servers if listed & set(n["platforms"])}
        out[ty] = {"S": s, "W": {x for x in s if classes[x] == "wired"}}
    return out


def baseline(pairs: Dict[tuple, int], sets: Dict[str, Dict[str, Set[str]]]) -> Dict[str, float]:
    num = den = 0.0
    for (a, b), n in pairs.items():
        sa, sb, wa, wb = sets[a]["S"], sets[b]["S"], sets[a]["W"], sets[b]["W"]
        tot = len(sa) * len(sb)
        den += n * (tot - len(sa & sb)) / tot
        num += n * (len(wa) * len(wb) - len(wa & wb)) / tot
    return {"share": num / den, "expected_transfers_per_pair": den}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--gate-dir", required=True)
    ap.add_argument("--selected", required=True)
    ap.add_argument("--rungs", default="lo,hi")
    ap.add_argument("--sim-input", type=Path, default=ROOT / "data" / "nofs-ids")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    topos = json.load(open(a.selected))["topologies"]
    rungs = a.rungs.split(",")
    ok, bad = R.load(a.gate_dir)
    if bad:
        raise SystemExit(f"FAIL LOUD: {len(bad)} failed runs in {a.gate_dir}")
    tt = json.load(open(a.sim_input / "task-types.json"))

    res: Dict[str, Any] = {"definition": "S(type) = servers with a platform listed for the type; independent uniform draws; "
                                         "transfer only if the two servers differ", "rungs": {}}
    for rung in rungs:
        per_topo: Dict[int, Any] = {}
        for t in topos:
            cfg = json.loads((a.inputs / f"wf1_{rung}" / "cfg" / f"cc40s{t}.json").read_text())
            infra = B.jsonable(B.live_infra(cfg, a.sim_input))
            pairs: Dict[tuple, int] = {}
            for w, fname in WINDOWS.items():
                wl = json.loads((a.inputs / f"wf1_{rung}" / "wl" / fname).read_text())
                types = task_types(wl["events"])
                for i, j, _ in wl["peer_exchange"]:
                    if not (0 <= int(i) < len(types) and 0 <= int(j) < len(types)):
                        raise SystemExit(f"FAIL LOUD: peer id outside the {len(types)} tasks of {fname}")
                    key = tuple(sorted((types[int(i)], types[int(j)])))
                    pairs[key] = pairs.get(key, 0) + 1
            sets = server_sets(infra, tt, {x for k in pairs for x in k})
            base = baseline(pairs, sets)
            uni = {"S": {n["node_name"] for n in infra["nodes"] if not n["node_name"].startswith("client_node")}}
            wired_all = [n for n in infra["link_topology"]["access_classes"] if not n.startswith("client_node")
                         and infra["link_topology"]["access_classes"][n]["class"] == "wired"]
            m, w = len(uni["S"]), len(wired_all)
            row: Dict[str, Any] = {"baseline_share": base["share"], "n_pairs": sum(pairs.values()),
                                   "pair_types": {"+".join(k): v for k, v in sorted(pairs.items())},
                                   "n_servers": m, "n_wired_servers": w,
                                   "uniform_over_all_servers_share": w * (w - 1) / (m * (m - 1)),
                                   "compatible_servers": {ty: len(v["S"]) for ty, v in sets.items()}, "arms": {}}
            for arm in R.ARMS:
                tr = trw = 0
                for g in R.WINDOWS:
                    r = ok[(t, f"{g}{rung}", arm)]
                    for pair, v in r["peerExchangeByAccessClass"].items():
                        tr += v["transfers"]
                        trw += v["transfers"] if pair == "wired+wired" else 0
                row["arms"][arm] = {"transfers": tr, "observed_share": trw / tr}
            per_topo[t] = row
        summary = {}
        for arm in R.ARMS:
            diffs = [100 * (per_topo[t]["arms"][arm]["observed_share"] - per_topo[t]["baseline_share"]) for t in topos]
            summary[arm] = {"median_diff_pp": st.median(diffs), "n_above": sum(d > 0 for d in diffs), "n": len(diffs),
                            "median_observed": st.median(per_topo[t]["arms"][arm]["observed_share"] for t in topos)}
        res["rungs"][rung] = {"median_baseline": st.median(per_topo[t]["baseline_share"] for t in topos),
                              "min_baseline": min(per_topo[t]["baseline_share"] for t in topos),
                              "max_baseline": max(per_topo[t]["baseline_share"] for t in topos),
                              "arms": summary, "topologies": per_topo}
    Path(a.out).write_text(json.dumps(res, indent=1))
    for rung, r in res["rungs"].items():
        print(f"== {rung}: baseline wired+wired share median {100 * r['median_baseline']:.1f}% (range {100 * r['min_baseline']:.1f}-{100 * r['max_baseline']:.1f}%)")
        for arm, s in r["arms"].items():
            print(f"  {arm:12s} observed median {100 * s['median_observed']:.1f}%  observed-baseline {s['median_diff_pp']:+.1f} pp  above in {s['n_above']}/{s['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
