#!/usr/bin/env python3
"""drainable_debug_v1 D1 -- on the states the arms were actually served, how good is each plan?

Registered in `docs/lineages/drainable_debug_v1.md`. Bars are the module constants below and
were committed before any dataset was cut.

This is the read that arbitrates the lineage's three hypotheses. For every captured live batch
state, brute-forced into a co-sim dataset, it scores four plans against the same enumerated
sweep:

  optimum        the best plan in the sweep (the label the supervised arms were trained toward)
  shortest_queue the plan reactive Knative's rule produces, recomputed from the snapshot
  gnn / mpoff    the plans the checkpoints decode on that state

and reads the three regrets against each other:

  H-env      shortest-queue regret is already ~0  -> the one-step label has nothing to teach
             here, and no supervised arm can win by being a better one-step scorer.
  H-gap      the checkpoint's regret is large and worse than shortest-queue's -> the model
             does not reach its own label's optimum on served states.
  H-myopia   the checkpoint's regret is BELOW shortest-queue's, yet it loses the live gate
             -> one-step quality is not the live problem, and the objective is.

The shortest-queue plan is a pure function of the snapshot -- each task's candidate list
carries `queue_length`, `node_id`, `platform_id` and `initialized`, which is everything
`src/policy/knative_network/scheduler.py:180-192` reads -- so it can be recomputed for EVERY
source, including states captured while a learned arm was driving. That is what makes the
comparison fair: all four plans are scored on identical states.

A plan that is not present in its dataset's enumerated sweep is a failure, not a missing value.
The sweep is the complete enumeration of the candidate slate; a plan outside it means the
candidate subsampling dropped a replica the policy would have used, and silently skipping such
datasets would bias every statistic toward the states where the policy agreed with the subset.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts_cosim.score_route_b_contention import load_rows  # noqa: E402

# --- bars (drainable_debug_v1, signed 2026-09-14, before any dataset was cut) ----------------
# H-env: at or below this median regret, the reactive rule is already near-optimal one-step.
D1A_SHORTEST_QUEUE_MAX_PCT = 3.0
# H-gap: at or above this median regret, AND no better than shortest-queue, the checkpoint
# fails to reach its own label's optimum on served states.
D1B_CHECKPOINT_MIN_PCT = 10.0
# A source is not read below this many scored datasets.
D1_MIN_DATASETS = 40
# --------------------------------------------------------------------------------------------

Plan = Dict[int, Tuple[int, int]]


def shortest_queue_plan(snapshot: Dict[str, Any], cosim_id_of: Dict[int, int]) -> Plan:
    """The plan reactive Knative's scheduler produces on this state.

    Mirrors `KnativeScheduler.placement` exactly: prefer replicas whose platform is already
    initialized, then take the minimum of `(queue depth, node id, platform id)`. Depth is
    incremented as tasks are placed, because the live scheduler pulls ONE task at a time and
    each placement is visible to the next -- that incrementality is the whole difference
    between this rule and a batch decode against a frozen snapshot, so dropping it would
    flatter the reactive arm.

    Tasks are placed in ascending trace id, which is arrival order; the co-sim task ids are a
    re-grouping by application and are not the order the live scheduler saw.
    """
    depth: Dict[str, int] = {}
    plan: Plan = {}
    tasks = sorted(snapshot.get("tasks") or [], key=lambda t: int(t["task_id"]))
    for task in tasks:
        trace_id = int(task["task_id"])
        cosim_id = cosim_id_of.get(trace_id)
        if cosim_id is None:
            raise SystemExit(
                f"FAIL LOUD: snapshot task {trace_id} has no co-sim id in this dataset's "
                "workload (trace_task_ids); the snapshot and the dataset disagree"
            )
        candidates = task.get("candidates") or []
        if not candidates:
            raise SystemExit(f"FAIL LOUD: snapshot task {trace_id} has no candidates")
        for cand in candidates:
            depth.setdefault(cand["queue_key"], int(cand.get("queue_length", 0) or 0))
        initialized = [c for c in candidates if c.get("initialized")]
        pool = initialized or candidates
        chosen = min(
            pool,
            key=lambda c: (depth[c["queue_key"]], int(c["node_id"]), int(c["platform_id"])),
        )
        plan[cosim_id] = (int(chosen["node_id"]), int(chosen["platform_id"]))
        depth[chosen["queue_key"]] += 1
    return plan


def plan_cost(plan: Plan, index: Dict[Tuple[Tuple[int, int, int], ...], float], where: str) -> float:
    key = tuple(sorted((t, n, p) for t, (n, p) in plan.items()))
    cost = index.get(key)
    if cost is None:
        raise SystemExit(
            f"FAIL LOUD: {where}: the plan is not in the enumerated sweep. The sweep is the "
            "complete enumeration of this dataset's candidate slate, so a plan outside it "
            "means candidate subsampling dropped a replica the policy uses. Re-cut the corpus "
            "with --force-candidates-from-plans; skipping the dataset would bias the read "
            "toward states where the policy happened to agree with the subset."
        )
    return cost


def index_rows(rows: Sequence[Tuple[Plan, float]]) -> Dict[Tuple[Tuple[int, int, int], ...], float]:
    index: Dict[Tuple[Tuple[int, int, int], ...], float] = {}
    for plan, value in rows:
        index[tuple(sorted((t, n, p) for t, (n, p) in plan.items()))] = value
    return index


def regret_pct(cost: float, optimum: float) -> float:
    if optimum <= 0:
        raise SystemExit(f"FAIL LOUD: non-positive sweep optimum {optimum!r}")
    return 100.0 * (cost - optimum) / optimum


def read_dataset(
    ds_dir: Path,
    objective: str,
    decoded: Optional[Dict[str, Plan]] = None,
) -> Dict[str, Any]:
    warm = json.loads((ds_dir / "warm_snapshot.json").read_text())
    snapshot = warm.get("snapshot") or warm
    workload = json.loads((ds_dir / "workload.json").read_text())
    trace_ids = workload.get("trace_task_ids")
    if not trace_ids:
        raise SystemExit(
            f"FAIL LOUD: {ds_dir}/workload.json carries no trace_task_ids, so snapshot task "
            "ids cannot be mapped onto the sweep's task ids"
        )
    cosim_id_of = {int(g): i for i, g in enumerate(trace_ids)}

    rows = load_rows(ds_dir, objective)
    index = index_rows(rows)
    optimum = min(value for _plan, value in rows)

    jsq = shortest_queue_plan(snapshot, cosim_id_of)
    out: Dict[str, Any] = {
        "dataset": ds_dir.name,
        "source": (warm.get("provenance") or {}).get("source_tag"),
        "n_rows": len(rows),
        "optimum": optimum,
        "shortest_queue": regret_pct(plan_cost(jsq, index, f"{ds_dir.name}/shortest_queue"), optimum),
    }
    for arm, plan in (decoded or {}).items():
        out[arm] = regret_pct(plan_cost(plan, index, f"{ds_dir.name}/{arm}"), optimum)
    return out


def summarise(rows: List[Dict[str, Any]], arms: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n_datasets": len(rows)}
    for key in ("shortest_queue", *arms):
        vals = [r[key] for r in rows if key in r]
        if not vals:
            continue
        out[key] = {
            "n": len(vals),
            "median_pct": st.median(vals),
            "mean_pct": sum(vals) / len(vals),
            "max_pct": max(vals),
            "pct_above_2": 100.0 * sum(1 for v in vals if v > 2.0) / len(vals),
            "exactly_optimal_pct": 100.0 * sum(1 for v in vals if v <= 1e-9) / len(vals),
        }
    return out


def verdict(summary: Dict[str, Any], arm: str = "gnn") -> Dict[str, Any]:
    n = summary.get("n_datasets", 0)
    if n < D1_MIN_DATASETS:
        return {"verdict": "VOID", "why": f"{n} datasets < {D1_MIN_DATASETS}"}
    jsq = summary.get("shortest_queue", {}).get("median_pct")
    ckpt = summary.get(arm, {}).get("median_pct")
    out: Dict[str, Any] = {
        "shortest_queue_median_pct": jsq,
        f"{arm}_median_pct": ckpt,
        "bars": {
            "d1a_shortest_queue_max_pct": D1A_SHORTEST_QUEUE_MAX_PCT,
            "d1b_checkpoint_min_pct": D1B_CHECKPOINT_MIN_PCT,
        },
    }
    if jsq is None:
        return {**out, "verdict": "VOID", "why": "no shortest-queue regret"}
    if jsq <= D1A_SHORTEST_QUEUE_MAX_PCT:
        out["d1a"] = "H-ENV-SUPPORTED"
    if ckpt is not None:
        if ckpt < jsq:
            out["d1c"] = "H-MYOPIA-SUPPORTED"
        if ckpt >= D1B_CHECKPOINT_MIN_PCT and ckpt >= jsq:
            out["d1b"] = "H-GAP-SUPPORTED"
    fired = [k for k in ("d1a", "d1b", "d1c") if k in out]
    out["verdict"] = "+".join(out[k] for k in fired) if fired else "NONE-FIRED"
    return out


def load_decoded(path: Optional[Path]) -> Dict[str, Dict[str, Plan]]:
    """{dataset: {arm: plan}} from a decode dump, keyed the way the sweep keys plans."""
    if path is None:
        return {}
    raw = json.loads(path.read_text())
    out: Dict[str, Dict[str, Plan]] = {}
    for ds, arms in raw.items():
        out[ds] = {
            arm: {int(t): (int(v[0]), int(v[1])) for t, v in plan.items()}
            for arm, plan in arms.items()
        }
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, required=True, help="directory of cut datasets (ds_*)")
    ap.add_argument("--objective", default="rtt", choices=("rtt", "makespan"))
    ap.add_argument("--decoded", type=Path,
                    help="JSON {dataset: {arm: {task_id: [node_id, platform_id]}}} of checkpoint plans")
    ap.add_argument("--arm", default="gnn", help="which decoded arm the verdict is read on")
    ap.add_argument("--limit", type=int, help="read at most this many datasets (smoke only)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)

    decoded = load_decoded(args.decoded)
    dirs = sorted(d for d in args.corpus.iterdir() if d.is_dir() and d.name.startswith("ds_"))
    if args.limit:
        dirs = dirs[: args.limit]
    if not dirs:
        raise SystemExit(f"FAIL LOUD: no ds_* directories under {args.corpus}")

    rows = [read_dataset(d, args.objective, decoded.get(d.name)) for d in dirs]
    arms = sorted({k for r in rows for k in r if k not in
                   ("dataset", "source", "n_rows", "optimum", "shortest_queue")})

    by_source: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_source.setdefault(row.get("source") or "unknown", []).append(row)

    reading = {
        "lineage": "drainable_debug_v1",
        "read": "D1",
        "corpus": str(args.corpus),
        "objective": args.objective,
        "bars": {
            "d1a_shortest_queue_max_pct": D1A_SHORTEST_QUEUE_MAX_PCT,
            "d1b_checkpoint_min_pct": D1B_CHECKPOINT_MIN_PCT,
            "d1_min_datasets": D1_MIN_DATASETS,
        },
        "per_source": {},
        "pooled": summarise(rows, arms),
        "per_dataset": rows,
    }
    for source, srows in sorted(by_source.items()):
        summary = summarise(srows, arms)
        reading["per_source"][source] = {"summary": summary, "verdict": verdict(summary, args.arm)}
        print(f"[{source}] n={summary['n_datasets']}")
        for key in ("shortest_queue", *arms):
            if key in summary:
                s = summary[key]
                print(f"    {key:>16}: median {s['median_pct']:7.3f}%  mean {s['mean_pct']:7.3f}%  "
                      f"exactly optimal {s['exactly_optimal_pct']:5.1f}%  >2% {s['pct_above_2']:5.1f}%")
        print(f"    verdict: {reading['per_source'][source]['verdict']['verdict']}")
    reading["pooled_verdict"] = verdict(reading["pooled"], args.arm)
    print(f"[pooled] {reading['pooled_verdict']['verdict']}")

    if args.out:
        args.out.write_text(json.dumps(reading, indent=1))
        print(f"[wrote] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
