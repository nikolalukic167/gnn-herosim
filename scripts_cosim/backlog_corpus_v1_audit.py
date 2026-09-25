#!/usr/bin/env python3
"""backlog_corpus_v1 corpus audit: does the synthetic backlog reach the label?

For every dataset: its rung, how much of the candidate slate is busy, and -- against its
joint_burst_v2 twin (same snapshot, same candidate draw, no backlog) -- whether the raw-RTT
optimum moved and how much of the optimum's work sits on busy replicas versus a uniform
random plan. Offline and descriptive: it orders nothing (rule 6).

    python3 scripts_cosim/backlog_corpus_v1_audit.py \
        --corpus simulation_data/gnn_datasets_backlog_corpus_v1_train simulation_data/gnn_datasets_backlog_corpus_v1_heldout \
        --twins simulation_data/gnn_datasets_joint_burst_v2_train simulation_data/gnn_datasets_joint_burst_v2_heldout \
        --out simulation_data/backlog_corpus_v1/audit.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def _sweep(ds: Path) -> Tuple[List[Tuple[Tuple[Tuple[int, int], ...], float]], int]:
    rows = []
    with open(ds / "placements" / "placements.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            p = r["placement_plan"]
            rows.append((tuple(tuple(int(v) for v in p[str(t)]) for t in range(len(p))), float(r["rtt"])))
    return rows, len(rows[0][0])


def _key(ds: Path) -> Tuple[str, int]:
    prov = json.loads((ds / "warm_snapshot.json").read_text())["provenance"]
    tag = str(prov["source_tag"]).rsplit("_b", 1)[0] if "synthetic_backlog" in prov else str(prov["source_tag"])
    return tag, int(prov["snapshot_id"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", type=Path, nargs="+", required=True)
    ap.add_argument("--twins", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    twins: Dict[Tuple[str, int], Path] = {}
    for d in args.twins:
        for ds in sorted(d.glob("ds_*")):
            if (ds / "warm_snapshot.json").is_file():
                twins[_key(ds)] = ds

    by_rung: Dict[float, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    missing_twin = 0
    for d in args.corpus:
        for ds in sorted(d.glob("ds_*")):
            if not (ds / "best.json").is_file():
                continue
            prov = json.loads((ds / "warm_snapshot.json").read_text())["provenance"]
            syn = prov["synthetic_backlog"]
            rung = float(syn["mean_seconds"])
            busy = {q for q, (k, s) in syn["per_key"].items() if s > 0}
            rows, n = _sweep(ds)
            pids = {c[1] for combo, _ in rows for c in combo}
            infra = json.loads((ds / "infrastructure.json").read_text())
            name_of = {}
            for specs in infra["live_snapshot_seed"]["replicas_by_type"].values():
                for sp in specs:
                    name_of[int(sp["platform_id"])] = f"{sp['node_name']}:{int(sp['platform_id'])}"
            busy_pids = {p for p in pids if name_of.get(p) in busy}
            opt_combo, opt = min(rows, key=lambda r: r[1])
            rec = by_rung[rung]
            rec["busy_candidate_fraction"].append(len(busy_pids) / max(1, len(pids)))
            rec["opt_on_busy"].append(sum(c[1] in busy_pids for c in opt_combo) / n)
            rec["random_on_busy"].append(
                statistics.fmean(sum(c[1] in busy_pids for c in combo) / n for combo, _ in rows)
            )
            twin = twins.get(_key(ds))
            if twin is None:
                missing_twin += 1
                continue
            trows, _ = _sweep(twin)
            tbest = min(trows, key=lambda r: r[1])
            rec["opt_moved_vs_twin"].append(float(tbest[0] != opt_combo))
            rec["opt_rtt_ratio_vs_twin"].append(opt / tbest[1] if tbest[1] > 0 else float("nan"))
            tmap = dict(rows)
            if tbest[0] in tmap:
                rec["twin_plan_regret_s"].append(tmap[tbest[0]] - opt)
    out = {"missing_twin": missing_twin, "rungs": {}}
    for rung, rec in sorted(by_rung.items()):
        out["rungs"][f"{rung:g}"] = {
            "n": len(rec["opt_on_busy"]),
            **{k: statistics.fmean(v) for k, v in rec.items() if v},
            "twin_plan_regret_s_median": statistics.median(rec["twin_plan_regret_s"]) if rec["twin_plan_regret_s"] else None,
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
