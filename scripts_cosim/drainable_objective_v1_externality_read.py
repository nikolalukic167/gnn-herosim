"""A3 (drainable_objective_v1): does the closed form track the externality that actually
happened?

The label's shaped term is a fluid prediction: putting A seconds of work on a platform
holding B seconds makes later arrivals wait (lambda/2)[(B+A)^2 - B^2] extra. That is an
approximation of a discrete arrival process on a cluster whose replica set moves under the
autoscaler, so it is measured rather than assumed.

The measurement uses D2's retained per-task records, which carry, for every task, when it
was committed, when it reached its platform, and when it finished. For each COMMIT GROUP
(the tasks a scheduler committed at the same instant -- a batch), the realized externality
is the queue time that LATER tasks provably spent waiting behind that group's tasks on the
same platform: exactly the overlap `drainable_debug_pertask_read.serialization()` computes,
restricted to predecessors from this group and successors from outside it.

BAR A3 (signed in docs/lineages/drainable_objective_v1.md before this ran):
    Spearman(predicted, realized) >= 0.50 over >= 5,000 batches per arm.

The fitted slope is reported as a calibration of lambda_p and is NOT used to choose the
gated V -- V = 1 is fixed in advance, so this read cannot tune the treatment it is meant
to check.

Input is a raw gate result kept with KEEP_RAW=1 KEEP_RAW_QSNAP=1 (SIM_FORCE_FULL_STATS=1
plus GNN_CAPTURE_DATASET_STATE=1), which is what D2's six arms already are.

Usage:

    python3 scripts_cosim/drainable_objective_v1_externality_read.py \
        --raw gnn_s1=.../dbg_D2_f4000_pg16/gnn_s1.json \
        --raw knative_network=.../dbg_D2_f4000_pg16/knative_network.json \
        --out simulation_data/drainable_objective_v1/a3_externality.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scripts_cosim.drainable_objective_v1_label_reversal_read import spearman

# ---- BARS. Committed before the data exists; do not edit after. ------------
A3_SPEARMAN_MIN = 0.50
A3_MIN_BATCHES = 5000
# A batch whose tasks inflicted nothing measurable on anyone carries no information about
# a monotone prediction and would just pile mass on (predicted>0, realized=0). They are
# counted and reported; the correlation is read on the batches that did inflict something.
A3_MIN_NONZERO_BATCHES = 500
# ---------------------------------------------------------------------------


class ExternalityReadError(RuntimeError):
    """Fail loud."""


def _num(rec: Dict[str, Any], key: str, default: float = 0.0) -> float:
    """`float(rec.get(k) or 0.0)` is wrong here and the D2 tests caught it: a legitimate
    0.0 is falsy, so a batch committed at simulated time 0 reads as "no timestamp"."""
    val = rec.get(key)
    return default if val is None else float(val)


def load_task_results(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise ExternalityReadError(f"raw result missing: {path}")
    doc = json.loads(path.read_text())
    stats = doc.get("stats") or doc
    recs = stats.get("taskResults")
    if not recs:
        raise ExternalityReadError(
            f"{path} carries no taskResults -- re-run the arm with KEEP_RAW=1, which "
            "exports SIM_FORCE_FULL_STATS=1; the streaming stats path writes an empty "
            "list above 10,000 events and this read would otherwise report zeros"
        )
    return recs


def platform_key(rec: Dict[str, Any]) -> str:
    return f"{rec.get('executionNode')}:{rec.get('executionPlatform')}"


def analyse(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per commit group: predicted backlog-squared increment vs realized inflicted wait."""
    by_platform: Dict[str, List[Dict[str, Any]]] = {}
    for rec in recs:
        by_platform.setdefault(platform_key(rec), []).append(rec)
    for items in by_platform.values():
        items.sort(key=lambda r: _num(r, "scheduledTime"))

    # Group tasks by (platform, commit instant) -- one scheduler decision's worth of work
    # on one platform, which is the unit the shaped term charges.
    groups: Dict[Tuple[str, float], List[Dict[str, Any]]] = {}
    for key, items in by_platform.items():
        for rec in items:
            groups.setdefault((key, _num(rec, "scheduledTime")), []).append(rec)

    predicted: List[float] = []
    realized: List[float] = []
    n_zero = 0
    for (key, commit_t), members in groups.items():
        items = by_platform[key]
        # A_p: the service time this group put on the platform. Measured, not modelled:
        # each member's own busy span on that platform.
        added = sum(_num(m, "doneTime") - _num(m, "arrivedTime") for m in members)
        if added <= 0.0:
            continue
        # B_p: the backlog already committed to this platform and not yet finished at the
        # commit instant -- the seconds of work standing in front of this group.
        backlog = 0.0
        for other in items:
            if _num(other, "scheduledTime") >= commit_t:
                continue
            done = _num(other, "doneTime")
            if done <= commit_t:
                continue
            backlog += done - max(_num(other, "arrivedTime"), commit_t)
        # Realized: queue time that tasks committed LATER spent provably behind this
        # group's members on this platform.
        member_ids = {m.get("taskId") for m in members}
        spans = [(_num(m, "arrivedTime"), _num(m, "doneTime")) for m in members]
        inflicted = 0.0
        for other in items:
            if other.get("taskId") in member_ids:
                continue
            if _num(other, "scheduledTime") <= commit_t:
                continue
            q0 = _num(other, "scheduledTime")
            q1 = _num(other, "arrivedTime")
            if q1 <= q0:
                continue
            for s0, s1 in spans:
                lo, hi = max(q0, s0), min(q1, s1)
                if hi > lo:
                    inflicted += hi - lo
        # The closed form's shape, with lambda folded into the fitted slope: the bar is on
        # the RANK correlation, so any positive lambda_p gives the same verdict.
        predicted.append(0.5 * ((backlog + added) ** 2 - backlog**2))
        realized.append(inflicted)
        if inflicted <= 0.0:
            n_zero += 1

    nonzero = [(p, r) for p, r in zip(predicted, realized) if r > 0.0]
    rho_all = spearman(predicted, realized)
    rho_nz = spearman([p for p, _r in nonzero], [r for _p, r in nonzero])

    slope: Optional[float] = None
    if nonzero:
        num = sum(p * r for p, r in nonzero)
        den = sum(p * p for p, _r in nonzero)
        slope = num / den if den > 0 else None

    n = len(predicted)
    if n < A3_MIN_BATCHES:
        verdict = "VOID"
    elif len(nonzero) < A3_MIN_NONZERO_BATCHES:
        verdict = "VOID-NO-EXTERNALITY"
    elif rho_nz is not None and rho_nz >= A3_SPEARMAN_MIN:
        verdict = "CLOSED-FORM-TRACKS"
    else:
        verdict = "CLOSED-FORM-DOES-NOT-TRACK"

    return {
        "n_batches": n,
        "n_batches_with_realized_externality": len(nonzero),
        "n_batches_zero_externality": n_zero,
        "spearman_all": rho_all,
        "spearman_nonzero": rho_nz,
        "implied_lambda_slope": slope,
        "predicted_median": st.median(predicted) if predicted else None,
        "realized_median": st.median(realized) if realized else None,
        "realized_median_nonzero": st.median([r for _p, r in nonzero]) if nonzero else None,
        "verdict": verdict,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    arms: Dict[str, Any] = {}
    for spec in args.raw:
        if "=" not in spec:
            raise ExternalityReadError(f"--raw wants NAME=PATH, got {spec!r}")
        name, _, path = spec.partition("=")
        recs = load_task_results(Path(path))
        arms[name] = analyse(recs)
        arms[name]["n_task_records"] = len(recs)
        arms[name]["source"] = path

    payload = {
        "bars": {
            "A3_SPEARMAN_MIN": A3_SPEARMAN_MIN,
            "A3_MIN_BATCHES": A3_MIN_BATCHES,
            "A3_MIN_NONZERO_BATCHES": A3_MIN_NONZERO_BATCHES,
        },
        "arms": arms,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))

    for name, a in arms.items():
        rho = a["spearman_nonzero"]
        print(
            f"[A3 {name:>22}] batches {a['n_batches']:>7} "
            f"({a['n_batches_with_realized_externality']} inflicted something)  "
            f"rho {rho if rho is not None else float('nan'):.3f}  "
            f"implied lambda {a['implied_lambda_slope']}  -> {a['verdict']}"
        )
    print(f"[A3] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
