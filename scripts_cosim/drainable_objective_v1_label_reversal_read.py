"""A2 (drainable_objective_v1): does the shaped label prefer what wins the stream?

The cheapest possible falsification of this lineage, and it costs no training. D1 scored
four plans per captured state under the ONE-STEP label and found the checkpoints beat the
reactive shortest-queue rule everywhere (0.35 % vs 16.06 % median regret) while losing the
live gate by -226 %. If the shaped label is the right objective, then under it the ordering
must flip: the plan the reactive rule makes -- the one that wins live -- should score at
least as well as the checkpoint's.

BARS (signed in docs/lineages/drainable_objective_v1.md before this ran):

    A2      at V = 1, shortest_queue regret <= gnn regret on >= 60 % of the states that
            have real choice                     =>  LABEL-AGREES-WITH-STREAM
    A2-rank median Spearman of the full plan ranking between V = 1 and V = 2, and between
            V = 1 and V = 0.5, >= 0.80 on BOTH. This is the control inherited from the
            horizon-return stop (lessons.md L32): a label whose ranking does not survive a
            change of its own parameter is not a stable property of the (state, action)
            pair, and a V that fails it is NOT TRAINED.

Inputs are exactly D1's: the captured co-sim datasets (their sweeps are already on the
live clock, because make_warm_corpus replays each snapshot's measured drain) and D1's
decoded-plan JSON. The shortest-queue plan is recomputed from each snapshot by the same
function D1 used, so the two reads cannot disagree about what the reactive rule does.

Usage:

    python3 scripts_cosim/drainable_objective_v1_label_reversal_read.py \
        --corpus simulation_data/gnn_datasets_drainable_debug_v1_d1_knb \
        --decoded simulation_data/drainable_debug_v1/d1_decoded.json \
        --arrival-rate 0.46 \
        --out simulation_data/drainable_objective_v1/a2_knb.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from scripts_cosim.drainable_debug_one_step_read import (
    Plan,
    shortest_queue_plan,
)
from scripts_cosim.drift_label import (
    DriftLabelError,
    build_state_context,
    externality_seconds,
)

# ---- BARS. Committed before the data exists; do not edit after. ------------
A2_PRIMARY_V = 1.0
A2_V_LADDER = (0.5, 1.0, 2.0)
A2_REVERSAL_MIN_PCT = 60.0
A2_RANK_STABILITY_MIN = 0.80
# A state with a handful of sweep rows scores every plan the same; D1 measured the
# reactive source's median whole-group plan space at 8 rows. Both bars are read on the
# states with real choice, and the full table is printed beside it.
A2_CHOICE_ROWS_MIN = 100
A2_MIN_DATASETS = 40
# ---------------------------------------------------------------------------


class LabelReversalError(RuntimeError):
    """Fail loud."""


def spearman(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """Rank correlation, average ranks for ties. None when either side is constant."""
    n = len(a)
    if n != len(b) or n < 3:
        return None

    def ranks(xs: Sequence[float]) -> List[float]:
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        out = [0.0] * len(xs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    if da == 0.0 or db == 0.0:
        return None
    return num / (da * db)


def regret_pct(value: float, optimum: float) -> float:
    if optimum <= 0:
        raise LabelReversalError(f"non-positive optimum {optimum}")
    return 100.0 * (value - optimum) / optimum


def plan_key(plan: Plan) -> Tuple[Tuple[int, int, int], ...]:
    return tuple(sorted((int(t), int(p[0]), int(p[1])) for t, p in plan.items()))


def read_dataset(
    ds_dir: Path,
    decoded_for_ds: Dict[str, Dict[str, Any]],
    task_types_db: Dict[str, Any],
    arrival_rate: float,
) -> Optional[Dict[str, Any]]:
    """One captured state, scored under every V in the ladder."""
    from scripts_cosim.score_route_b_contention import load_rows

    snap_path = ds_dir / "warm_snapshot.json"
    if not snap_path.exists():
        raise LabelReversalError(f"{ds_dir}: warm_snapshot.json missing")
    snapshot = json.loads(snap_path.read_text())
    workload = json.loads((ds_dir / "workload.json").read_text())
    trace_task_ids = workload.get("trace_task_ids")
    if not trace_task_ids:
        raise LabelReversalError(f"{ds_dir}: workload carries no trace_task_ids")
    cosim_id_of = {int(t): i for i, t in enumerate(trace_task_ids)}

    rows = load_rows(ds_dir, "rtt")
    ctx = build_state_context(
        ds_dir, task_types_db, arrival_rate=arrival_rate,
        task_type_names=None,
    )

    # Shaped value of every sweep row at every V, in one pass over the externality.
    ext = [externality_seconds(plan, ctx) for plan, _v in rows]
    base = [float(v) for _p, v in rows]
    index_by_key: Dict[Tuple[Tuple[int, int, int], ...], int] = {}
    for i, (plan, _v) in enumerate(rows):
        index_by_key.setdefault(plan_key(plan), i)

    plans: Dict[str, Plan] = {"shortest_queue": shortest_queue_plan(snapshot, cosim_id_of)}
    for arm, per_task in (decoded_for_ds or {}).items():
        plans[arm] = {int(t): (int(p[0]), int(p[1])) for t, p in per_task.items()}

    row_index: Dict[str, int] = {}
    for name, plan in plans.items():
        idx = index_by_key.get(plan_key(plan))
        if idx is None:
            raise LabelReversalError(
                f"{ds_dir}: the {name} plan is not in the enumerated sweep -- re-cut the "
                "corpus with --force-candidates-from-plans; scoring it against a sweep "
                "that never contained it would be an invented number"
            )
        row_index[name] = idx

    out: Dict[str, Any] = {
        "dataset": ds_dir.name,
        "n_rows": len(rows),
        "has_real_choice": len(rows) >= A2_CHOICE_ROWS_MIN,
        "per_v": {},
    }
    values_by_v: Dict[float, List[float]] = {}
    for v in A2_V_LADDER:
        values = [b + v * e for b, e in zip(base, ext)]
        values_by_v[v] = values
        optimum = min(values)
        out["per_v"][f"{v:g}"] = {
            "optimum": optimum,
            "regret_pct": {name: regret_pct(values[i], optimum) for name, i in row_index.items()},
        }
    # V = 0 (the one-step label) for reference; this reproduces D1's own column.
    opt0 = min(base)
    out["per_v"]["0"] = {
        "optimum": opt0,
        "regret_pct": {name: regret_pct(base[i], opt0) for name, i in row_index.items()},
    }

    # A2-rank: does the ordering of the whole plan space survive a change of V?
    out["rank_stability"] = {
        "v1_vs_v2": spearman(values_by_v[1.0], values_by_v[2.0]),
        "v1_vs_v05": spearman(values_by_v[1.0], values_by_v[0.5]),
        "v1_vs_v0": spearman(values_by_v[1.0], base),
    }
    return out


def summarise(per_dataset: List[Dict[str, Any]], arm: str) -> Dict[str, Any]:
    choice = [d for d in per_dataset if d["has_real_choice"]]

    def reversal_rate(rows: List[Dict[str, Any]]) -> Optional[float]:
        usable = [
            d
            for d in rows
            if arm in d["per_v"][f"{A2_PRIMARY_V:g}"]["regret_pct"]
            and "shortest_queue" in d["per_v"][f"{A2_PRIMARY_V:g}"]["regret_pct"]
        ]
        if not usable:
            return None
        wins = sum(
            1
            for d in usable
            if d["per_v"][f"{A2_PRIMARY_V:g}"]["regret_pct"]["shortest_queue"]
            <= d["per_v"][f"{A2_PRIMARY_V:g}"]["regret_pct"][arm] + 1e-9
        )
        return 100.0 * wins / len(usable)

    def medians(rows: List[Dict[str, Any]], v: str) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for name in ("shortest_queue", "gnn", "mpoff"):
            vals = [
                d["per_v"][v]["regret_pct"][name]
                for d in rows
                if name in d["per_v"][v]["regret_pct"]
            ]
            if vals:
                out[name] = st.median(vals)
        return out

    def stability(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
        vals = [d["rank_stability"][key] for d in rows if d["rank_stability"][key] is not None]
        return st.median(vals) if vals else None

    rate = reversal_rate(choice)
    s12 = stability(choice, "v1_vs_v2")
    s105 = stability(choice, "v1_vs_v05")

    if len(per_dataset) < A2_MIN_DATASETS:
        verdict = "VOID"
    elif not choice:
        verdict = "VOID-NO-CHOICE"
    elif rate is None:
        verdict = "VOID-NO-ARM"
    elif rate >= A2_REVERSAL_MIN_PCT:
        verdict = "LABEL-AGREES-WITH-STREAM"
    else:
        verdict = "LABEL-DOES-NOT-AGREE"

    rank_ok = (
        s12 is not None
        and s105 is not None
        and s12 >= A2_RANK_STABILITY_MIN
        and s105 >= A2_RANK_STABILITY_MIN
    )
    return {
        "arm": arm,
        "n_datasets": len(per_dataset),
        "n_with_real_choice": len(choice),
        "reversal_rate_pct": rate,
        "medians_all": {v: medians(per_dataset, v) for v in ("0", "0.5", "1", "2")},
        "medians_choice": {v: medians(choice, v) for v in ("0", "0.5", "1", "2")},
        "rank_stability_median": {"v1_vs_v2": s12, "v1_vs_v05": s105},
        "rank_stability_verdict": "STABLE" if rank_ok else "UNSTABLE",
        "verdict": verdict,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--decoded", type=Path, required=True)
    ap.add_argument("--arm", default="gnn", help="the checkpoint arm the bar is read on")
    ap.add_argument("--arrival-rate", type=float, required=True)
    ap.add_argument("--task-types", type=Path, default=Path("data/nofs-ids/task-types.json"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    db = json.loads(args.task_types.read_text())
    decoded = json.loads(args.decoded.read_text())
    corpus_name = args.corpus.name

    ds_dirs = sorted(args.corpus.glob("ds_*"))
    if args.limit:
        ds_dirs = ds_dirs[: args.limit]
    if not ds_dirs:
        raise LabelReversalError(f"no datasets under {args.corpus}")

    per_dataset: List[Dict[str, Any]] = []
    for ds in ds_dirs:
        full_id = f"{corpus_name}/{ds.name}"
        if full_id not in decoded:
            raise LabelReversalError(
                f"{full_id} has no decoded plans in {args.decoded} -- D1 keyed plans by "
                "the full dataset id after a collision collapsed 151 datasets into 84; "
                "refusing to score a dataset against another one's plan"
            )
        row = read_dataset(ds, decoded[full_id], db, args.arrival_rate)
        if row is not None:
            per_dataset.append(row)

    summary = summarise(per_dataset, args.arm)
    payload = {
        "bars": {
            "A2_PRIMARY_V": A2_PRIMARY_V,
            "A2_REVERSAL_MIN_PCT": A2_REVERSAL_MIN_PCT,
            "A2_RANK_STABILITY_MIN": A2_RANK_STABILITY_MIN,
            "A2_CHOICE_ROWS_MIN": A2_CHOICE_ROWS_MIN,
            "A2_MIN_DATASETS": A2_MIN_DATASETS,
        },
        "corpus": str(args.corpus),
        "arrival_rate": args.arrival_rate,
        "summary": summary,
        "per_dataset": per_dataset,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))

    print(f"[A2 {corpus_name}] n={summary['n_datasets']} "
          f"({summary['n_with_real_choice']} with real choice)")
    for v in ("0", "0.5", "1", "2"):
        med = summary["medians_choice"].get(v) or {}
        cells = "  ".join(f"{k}: {val:7.2f}%" for k, val in sorted(med.items()))
        print(f"    V={v:<4} {cells}")
    rate = summary["reversal_rate_pct"]
    print(
        f"    reversal rate at V={A2_PRIMARY_V:g}: "
        f"{rate if rate is not None else float('nan'):.1f}% "
        f"against a {A2_REVERSAL_MIN_PCT}% bar -> {summary['verdict']}"
    )
    rs = summary["rank_stability_median"]
    print(
        f"    rank stability: v1~v2 {rs['v1_vs_v2']}, v1~v0.5 {rs['v1_vs_v05']} "
        f"(bar {A2_RANK_STABILITY_MIN}) -> {summary['rank_stability_verdict']}"
    )
    print(f"[A2] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
