#!/usr/bin/env python3
"""offline_live_transfer_v1 -- R2: is there a cheap offline statistic that DOES rank runs?

Registered in docs/lineages/offline_live_transfer_v1.md on 2026-09-15. Bars are the module
constants below and were committed before any candidate was correlated with live latency.

R1 measured that the offline score the selector uses -- held-out plan regret -- carries no
across-run signal (pooled-z rho = -0.030, p = 0.772, n = 96). R2 asks whether some other
number, still computable without a live run, does.

Four candidates, declared in the node before any was computed, each a property of the
CHECKPOINT measured on one common yardstick (drainable_debug_v1's 151 captured live states,
brute-forced so every plan can be scored against the true optimum):

  depth              mean places the decoded plan sits above the shallowest legal replica
  concentration      most tasks the plan puts on one platform, as a fraction of the batch
  one_step_regret    the decoded plan's regret against that state's sweep optimum
  range_sensitivity  fraction of placements that move when platform queue depth is scaled
                     toward the live range the corpus never contains

BARS

  A candidate is a SURROGATE when, against live latency:
    * pooled-z |rho| >= 0.50 (z-scored within each (family, arm) cell, as in R1), AND
    * its p-value survives HOLM correction over R2_HOLM_N candidates at alpha = 0.05, AND
    * within-family |rho| >= 0.40 in >= 2 of the 3 families.

  AT MOST ONE candidate is promoted -- the one with the larger pooled-z |rho|. Any other that
  clears is reported and NOT used. This was fixed in advance so a surrogate cannot be chosen
  after seeing which one flatters a downstream result.

  Holm always divides by R2_HOLM_N = 4, even if a candidate could not be computed. Shrinking
  the family after seeing the data would make the correction weaker than the one registered.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts_cosim.offline_live_transfer_v1_pairing_read import (  # noqa: E402
    ARMS,
    FAMILIES,
    TransferReadError,
    load_live,
    spearman,
    zscore,
)

# --------------------------------------------------------------------------- BARS
CANDIDATES: Tuple[str, ...] = (
    "depth", "concentration", "one_step_regret", "range_sensitivity",
)
R2_POOLED_Z_MIN_ABS_RHO = 0.50
R2_WITHIN_MIN_ABS_RHO = 0.40
R2_WITHIN_MIN_FAMILIES = 2
R2_ALPHA = 0.05
R2_HOLM_N = 4                  # never shrunk to the number actually computed
R2_MAX_PROMOTED = 1
# The scale the range-sensitivity decode used. Recorded here so the statistic's meaning is
# pinned to a number rather than to whatever the sbatch happened to pass.
R2_QUEUE_SCALE = 50.0

# Family tag -> the checkpoint-name prefix its decoded files carry.
FAMILY_TAGS: Dict[str, str] = {
    "F1_x800p2": "peer-affinity-v1-x800p2",
    "F2_x800p3": "peer-affinity-v1-x800p3",
    "F3_warm": "peer-affinity-v1-warm",
}

Plan = Dict[int, Tuple[int, int]]


# ------------------------------------------------------------------- the statistics
def concentration(plan: Plan) -> float:
    """Most tasks on one platform, as a fraction of the batch."""
    if not plan:
        raise TransferReadError("empty plan has no concentration")
    counts = Counter(plan.values())
    return max(counts.values()) / len(plan)


def depth_above_shallowest(
    plan: Plan, snapshot: Dict[str, Any], cosim_id_of: Dict[int, int]
) -> float:
    """Mean queue depth of the chosen replica minus the shallowest legal one.

    Depth is read from the snapshot the state was captured with, exactly the field the
    reactive scheduler reads (`queue_length` on each candidate). Unlike the reactive rule
    this does NOT increment depth as tasks are placed: the question is how deep the model
    chose to go relative to what was on offer, not how a sequential rule would have gone.
    """
    if not plan:
        raise TransferReadError("empty plan has no depth")
    gaps: List[float] = []
    for task in snapshot.get("tasks") or []:
        trace_id = int(task["task_id"])
        cosim_id = cosim_id_of.get(trace_id)
        if cosim_id is None or cosim_id not in plan:
            continue
        candidates = task.get("candidates") or []
        if not candidates:
            raise TransferReadError(f"snapshot task {trace_id} has no candidates")
        by_place = {
            (int(c["node_id"]), int(c["platform_id"])): float(c.get("queue_length", 0) or 0)
            for c in candidates
        }
        chosen = plan[cosim_id]
        if chosen not in by_place:
            raise TransferReadError(
                f"task {trace_id}: the decoded placement {chosen} is not among this task's "
                "candidates in the snapshot -- the plan and the state disagree"
            )
        gaps.append(by_place[chosen] - min(by_place.values()))
    if not gaps:
        raise TransferReadError("no task in this plan could be matched to the snapshot")
    return sum(gaps) / len(gaps)


def range_sensitivity(unscaled: Plan, scaled: Plan) -> float:
    """Fraction of placements that move when queue depth is scaled by R2_QUEUE_SCALE."""
    shared = set(unscaled) & set(scaled)
    if not shared:
        raise TransferReadError("the scaled and unscaled plans share no tasks")
    moved = sum(1 for t in shared if unscaled[t] != scaled[t])
    return moved / len(shared)


# ------------------------------------------------------------------- Holm correction
def holm(pvalues: Dict[str, float], n: int) -> Dict[str, float]:
    """Holm-Bonferroni adjusted p-values over a family of size `n`.

    `n` is the REGISTERED family size, which may exceed len(pvalues) when a candidate could
    not be computed. Passing the smaller number would weaken the correction.
    """
    if n < len(pvalues):
        raise TransferReadError(f"Holm family size {n} is smaller than {len(pvalues)} tests")
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    adjusted: Dict[str, float] = {}
    running = 0.0
    for i, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (n - i) * p))
        adjusted[name] = running
    return adjusted


# ------------------------------------------------------------------- the read
def score_candidate(
    values: Dict[Tuple[str, str, int], float],
    live: Dict[Tuple[str, str, int], float],
) -> Dict[str, Any]:
    """Pooled-z and per-family correlation of one candidate against live latency."""
    per_family: Dict[str, Dict[str, Any]] = {}
    zx: List[float] = []
    zy: List[float] = []
    for fam in FAMILIES:
        fx: List[float] = []
        fy: List[float] = []
        for arm in ARMS:
            keys = sorted(k for k in values if k[0] == fam and k[1] == arm and k in live)
            if len(keys) < 3:
                continue
            cx = [values[k] for k in keys]
            cy = [live[k] for k in keys]
            zx.extend(zscore(cx))
            zy.extend(zscore(cy))
            fx.extend(cx)
            fy.extend(cy)
        if len(fx) >= 3:
            rho, p = spearman(fx, fy)
            per_family[fam] = {"n": len(fx), "rho": rho, "p": p,
                               "clears": abs(rho) >= R2_WITHIN_MIN_ABS_RHO}
    if len(zx) < 3:
        raise TransferReadError("not enough paired points to score this candidate")
    z_rho, z_p = spearman(zx, zy)
    families_clearing = sum(1 for v in per_family.values() if v["clears"])
    return {
        "n": len(zx),
        "pooled_z_rho": z_rho,
        "pooled_z_p": z_p,
        "per_family": per_family,
        "families_clearing": families_clearing,
        "clears_rho_bar": abs(z_rho) >= R2_POOLED_Z_MIN_ABS_RHO,
        "clears_family_bar": families_clearing >= R2_WITHIN_MIN_FAMILIES,
    }


def read(
    candidate_values: Dict[str, Dict[Tuple[str, str, int], float]],
    live: Dict[Tuple[str, str, int], float],
) -> Dict[str, Any]:
    unknown = set(candidate_values) - set(CANDIDATES)
    if unknown:
        raise TransferReadError(f"unregistered candidate(s): {sorted(unknown)}")

    scored: Dict[str, Any] = {}
    not_computed = [c for c in CANDIDATES if c not in candidate_values]
    for name in CANDIDATES:
        if name in candidate_values:
            scored[name] = score_candidate(candidate_values[name], live)

    adjusted = holm({k: v["pooled_z_p"] for k, v in scored.items()}, R2_HOLM_N)
    for name, row in scored.items():
        row["p_holm"] = adjusted[name]
        row["clears_p_bar"] = adjusted[name] < R2_ALPHA
        row["is_surrogate"] = bool(
            row["clears_rho_bar"] and row["clears_p_bar"] and row["clears_family_bar"]
        )

    clearing = sorted(
        (n for n, r in scored.items() if r["is_surrogate"]),
        key=lambda n: abs(scored[n]["pooled_z_rho"]), reverse=True,
    )
    promoted = clearing[:R2_MAX_PROMOTED]
    return {
        "lineage": "offline_live_transfer_v1",
        "read": "R2",
        "bars": {
            "pooled_z_min_abs_rho": R2_POOLED_Z_MIN_ABS_RHO,
            "within_min_abs_rho": R2_WITHIN_MIN_ABS_RHO,
            "within_min_families": R2_WITHIN_MIN_FAMILIES,
            "alpha": R2_ALPHA,
            "holm_family_size": R2_HOLM_N,
            "max_promoted": R2_MAX_PROMOTED,
            "queue_scale": R2_QUEUE_SCALE,
        },
        "candidates": scored,
        "not_computed": not_computed,
        "clearing": clearing,
        "promoted": promoted,
        "reported_not_used": clearing[R2_MAX_PROMOTED:],
        "verdict": f"SURROGATE:{promoted[0]}" if promoted else "NO-SURROGATE",
    }


# ------------------------------------------------------------------- loading
def load_decoded(path: Path) -> Dict[str, Dict[str, Plan]]:
    raw = json.loads(path.read_text())
    out: Dict[str, Dict[str, Plan]] = {}
    for ds_id, arms in raw.items():
        if ds_id.startswith("_"):
            continue
        out[ds_id] = {
            arm: {int(t): (int(p[0]), int(p[1])) for t, p in plan.items()}
            for arm, plan in arms.items()
        }
    if not out:
        raise TransferReadError(f"{path}: no decoded datasets")
    return out


def load_state(sim_root: Path, ds_id: str) -> Tuple[Dict[str, Any], Dict[int, int]]:
    ds_dir = sim_root / ds_id
    warm = json.loads((ds_dir / "warm_snapshot.json").read_text())
    snapshot = warm.get("snapshot") or warm
    workload = json.loads((ds_dir / "workload.json").read_text())
    trace_ids = workload.get("trace_task_ids")
    if not trace_ids:
        raise TransferReadError(f"{ds_dir}/workload.json carries no trace_task_ids")
    return snapshot, {int(g): i for i, g in enumerate(trace_ids)}


def main(argv: Optional[List[str]] = None) -> int:
    from scripts_cosim.drainable_debug_one_step_read import (  # noqa: PLC0415
        index_rows, plan_cost, regret_pct,
    )
    from scripts_cosim.score_route_b_contention import load_rows  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decoded-dir", type=Path, required=True,
                    help="unscaled decodes, <tag>-seed<N>.json")
    ap.add_argument("--scaled-dir", type=Path,
                    help="decodes at --queue-scale R2_QUEUE_SCALE; without it, "
                         "range_sensitivity is NOT-COMPUTED and Holm still divides by 4")
    ap.add_argument("--gate-root", type=Path,
                    default=Path("simulation_data/peer_affinity_live_gate/results"))
    ap.add_argument("--sim-root", type=Path, default=Path("simulation_data"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    live: Dict[Tuple[str, str, int], float] = {}
    for fam, spec in FAMILIES.items():
        for (arm, seed), row in load_live(args.gate_root, spec["results"]).items():
            live[(fam, arm, seed)] = row["averageElapsedTime"]

    values: Dict[str, Dict[Tuple[str, str, int], float]] = {
        c: {} for c in ("depth", "concentration", "one_step_regret")
    }
    if args.scaled_dir:
        values["range_sensitivity"] = {}

    state_cache: Dict[str, Any] = {}
    for fam, tag in FAMILY_TAGS.items():
        for seed in range(1, 17):
            path = args.decoded_dir / f"{tag}-seed{seed}.json"
            if not path.is_file():
                raise TransferReadError(f"missing decode: {path}")
            decoded = load_decoded(path)
            scaled = (load_decoded(args.scaled_dir / f"{tag}-seed{seed}.json")
                      if args.scaled_dir else None)
            per_arm: Dict[str, Dict[str, List[float]]] = {
                a: {k: [] for k in values} for a in ARMS
            }
            for ds_id, arms in decoded.items():
                if ds_id not in state_cache:
                    snapshot, cosim_id_of = load_state(args.sim_root, ds_id)
                    rows = load_rows(args.sim_root / ds_id, objective="rtt")
                    index = index_rows(rows)
                    state_cache[ds_id] = (snapshot, cosim_id_of, index, min(index.values()))
                snapshot, cosim_id_of, index, optimum = state_cache[ds_id]
                for arm, plan in arms.items():
                    if arm not in per_arm:
                        continue
                    per_arm[arm]["depth"].append(
                        depth_above_shallowest(plan, snapshot, cosim_id_of))
                    per_arm[arm]["concentration"].append(concentration(plan))
                    per_arm[arm]["one_step_regret"].append(
                        regret_pct(plan_cost(plan, index, f"{ds_id}/{arm}"), optimum))
                    if scaled is not None:
                        other = (scaled.get(ds_id) or {}).get(arm)
                        if other is None:
                            raise TransferReadError(
                                f"{ds_id}/{arm}: decoded unscaled but not scaled; the two "
                                "decodes must cover the same states")
                        per_arm[arm]["range_sensitivity"].append(
                            range_sensitivity(plan, other))
            for arm, series in per_arm.items():
                for cand, xs in series.items():
                    if not xs:
                        raise TransferReadError(f"{tag} seed {seed} {arm}: no {cand} values")
                    values[cand][(fam, arm, seed)] = sum(xs) / len(xs)
            print(f"[r2] {tag} seed {seed}: "
                  + "  ".join(f"{c}={values[c][(fam,'gnn',seed)]:.4f}" for c in values),
                  flush=True)

    result = read(values, live)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=str))

    print(f"\n[R2] bars: pooled-z |rho| >= {R2_POOLED_Z_MIN_ABS_RHO}, Holm over "
          f"{R2_HOLM_N} at alpha {R2_ALPHA}, within-family |rho| >= "
          f"{R2_WITHIN_MIN_ABS_RHO} in >= {R2_WITHIN_MIN_FAMILIES}")
    for name in CANDIDATES:
        row = result["candidates"].get(name)
        if row is None:
            print(f"     {name:20s} NOT-COMPUTED (Holm still divides by {R2_HOLM_N})")
            continue
        fams = "  ".join(f"{f.split('_')[0]}={v['rho']:+.3f}"
                         for f, v in sorted(row["per_family"].items()))
        print(f"     {name:20s} n={row['n']:3d}  pooled-z rho={row['pooled_z_rho']:+.3f}"
              f"  p={row['pooled_z_p']:.4f}  p_holm={row['p_holm']:.4f}"
              f"  families_clearing={row['families_clearing']}/3   {fams}"
              f"   -> {'SURROGATE' if row['is_surrogate'] else 'no'}")
    if result["reported_not_used"]:
        print(f"[R2] also cleared, reported and NOT used: {result['reported_not_used']}")
    print(f"[R2] {result['verdict']}")
    print(f"[wrote] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
