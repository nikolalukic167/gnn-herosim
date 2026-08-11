#!/usr/bin/env python3
"""Quantify how much *irreducible joint coupling* exists in the co-sim oracle labels.

Central question for "is a GNN necessary":
  A pointwise edge scorer f(task, platform) -> logit picks argmax per task,
  INDEPENDENTLY. A GNN can (in principle) make the score of one edge depend on
  the rest of the batch. The GNN can only beat a pointwise model when the OPTIMAL
  placement is NOT recoverable by independent per-task decisions.

We measure three label-grounded quantities per dataset, from placements/placements.jsonl
(the full (placement_plan, rtt) brute-force sweep):

  (M1) marginal-greedy regret:
       pi(t) = argmin_p [ min RTT over all combos with task t on platform p ]
       regret = RTT(joint combo of pi) - RTT(optimal).
       If ~0 everywhere, each task's best platform is independent of the others
       => separable => pointwise MLP suffices.

  (M2) identical-task symmetry:
       tasks with identical (type, source_node) get IDENTICAL features, hence a
       pointwise model (and a frozen-decode GNN) MUST assign them the same platform.
       - frac of datasets with >=2 identical tasks
       - among those, does the optimum SPREAD them?
       - regret of forcing identical tasks to co-locate (pointwise lower bound):
         min RTT over combos where every identical group shares one platform,
         minus optimal RTT.

  (M3) collision in optimum:
       fraction of optimal combos that place 2+ tasks on the SAME platform
       (i.e., the optimum itself "double books") vs. spreads.

Run:
  pipenv run python3 scripts_cosim/separability_diagnostic.py <corpus_dir> [--limit N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_workload_task_sigs(ds_dir: Path) -> Optional[List[Tuple[str, str]]]:
    """Return per-task (type_name, source_node) in event order = task index order."""
    wl = ds_dir / "workload.json"
    if not wl.exists():
        return None
    try:
        data = json.loads(wl.read_text())
    except Exception:
        return None
    sigs: List[Tuple[str, str]] = []
    for ev in data.get("events", []):
        app = ev.get("application", {})
        # task type is the dag key, e.g. {"dnn2": []}
        dag = app.get("dag", {})
        ttype = next(iter(dag.keys()), app.get("name", "?"))
        src = str(ev.get("node_name", "?"))
        sigs.append((str(ttype), src))
    return sigs


def load_combos(ds_dir: Path) -> Optional[List[Tuple[Dict[int, Tuple[int, int]], float]]]:
    jp = ds_dir / "placements" / "placements.jsonl"
    if not jp.exists():
        return None
    combos: List[Tuple[Dict[int, Tuple[int, int]], float]] = []
    with jp.open() as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception as exc:
                raise RuntimeError(
                    f"{jp}:{line_number}: invalid JSON"
                ) from exc
            plan = rec.get("placement_plan")
            rtt = rec.get("rtt")
            if plan is None or rtt is None:
                raise RuntimeError(
                    f"{jp}:{line_number}: missing placement_plan or rtt"
                )
            pp: Dict[int, Tuple[int, int]] = {}
            for k, v in plan.items():
                try:
                    if isinstance(v, (list, tuple)) and len(v) >= 2:
                        pp[int(k)] = (int(v[0]), int(v[1]))
                    else:
                        raise ValueError("placement must contain node and platform")
                except Exception as exc:
                    raise RuntimeError(
                        f"{jp}:{line_number}: invalid placement plan"
                    ) from exc
            if not pp:
                raise RuntimeError(f"{jp}:{line_number}: empty placement plan")
            combos.append((pp, float(rtt)))
    return combos or None


def analyze_dataset(ds_dir: Path) -> Optional[dict]:
    combos = load_combos(ds_dir)
    if not combos:
        return None
    sigs = load_workload_task_sigs(ds_dir)

    n_tasks = max(len(pp) for pp, _ in combos)
    # task indices present
    task_ids = sorted({t for pp, _ in combos for t in pp.keys()})
    if len(task_ids) < 1:
        return None

    # optimal
    opt_plan, opt_rtt = min(combos, key=lambda x: x[1])
    if opt_rtt <= 0:
        return None

    # --- M1: marginal greedy ---
    # marginal_min[t][p] = min rtt over combos with task t -> p
    marginal_min: Dict[int, Dict[Tuple[int, int], float]] = defaultdict(lambda: defaultdict(lambda: float("inf")))
    combo_lookup: Dict[Tuple[Tuple[int, int], ...], float] = {}
    for pp, rtt in combos:
        key = tuple(pp[t] for t in task_ids)
        # keep min rtt if duplicate combo keys
        if key not in combo_lookup or rtt < combo_lookup[key]:
            combo_lookup[key] = rtt
        for t in task_ids:
            p = pp[t]
            if rtt < marginal_min[t][p]:
                marginal_min[t][p] = rtt

    greedy_choice: Dict[int, Tuple[int, int]] = {}
    for t in task_ids:
        greedy_choice[t] = min(marginal_min[t].items(), key=lambda kv: kv[1])[0]
    greedy_key = tuple(greedy_choice[t] for t in task_ids)
    greedy_rtt = combo_lookup.get(greedy_key)  # may be None if combo not enumerated
    m1_regret = None
    m1_regret_rel = None
    greedy_in_sweep = greedy_rtt is not None
    if greedy_rtt is not None:
        m1_regret = greedy_rtt - opt_rtt
        m1_regret_rel = m1_regret / opt_rtt

    # --- M2: identical-task symmetry (needs sigs) ---
    m2 = {
        "has_identical": False,
        "n_identical_groups": 0,
        "max_group_size": 0,
        "opt_spreads_identical": None,
        "colocate_regret_rel": None,
    }
    if sigs is not None and len(sigs) >= len(task_ids):
        groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        for t in task_ids:
            if t < len(sigs):
                groups[sigs[t]].append(t)
        ident_groups = [g for g in groups.values() if len(g) >= 2]
        m2["n_identical_groups"] = len(ident_groups)
        m2["has_identical"] = len(ident_groups) > 0
        m2["max_group_size"] = max((len(g) for g in ident_groups), default=0)
        if ident_groups:
            # does optimum place each identical group on a single platform?
            opt_spreads = False
            for g in ident_groups:
                plats = {opt_plan[t] for t in g}
                if len(plats) > 1:
                    opt_spreads = True
                    break
            m2["opt_spreads_identical"] = opt_spreads
            # pointwise lower bound: best combo where each identical group co-locates
            best_colo = float("inf")
            for key, rtt in combo_lookup.items():
                ok = True
                for g in ident_groups:
                    plats = {key[task_ids.index(t)] for t in g}
                    if len(plats) > 1:
                        ok = False
                        break
                if ok and rtt < best_colo:
                    best_colo = rtt
            if best_colo < float("inf"):
                m2["colocate_regret_rel"] = (best_colo - opt_rtt) / opt_rtt

    # --- M3: collision in optimum ---
    opt_plats = [opt_plan[t] for t in task_ids]
    opt_has_collision = len(set(opt_plats)) < len(opt_plats)
    opt_unique_plats = len(set(opt_plats))

    return {
        "n_tasks": len(task_ids),
        "n_combos": len(combos),
        "opt_rtt": opt_rtt,
        "greedy_in_sweep": greedy_in_sweep,
        "m1_regret_rel": m1_regret_rel,
        "m1_greedy_eq_opt": (greedy_key == tuple(opt_plan[t] for t in task_ids)),
        "m2": m2,
        "opt_has_collision": opt_has_collision,
        "opt_unique_plats": opt_unique_plats,
    }


def pctl(vals: List[float], q: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    return s[min(len(s) - 1, int(q * len(s)))]


def summarize_results(results: List[dict]) -> dict[str, Any]:
    if not results:
        raise ValueError("Cannot summarize empty separability results")
    n = len(results)
    m1_rel = [
        r["m1_regret_rel"]
        for r in results
        if r["m1_regret_rel"] is not None
    ]
    greedy_eq = sum(1 for r in results if r["m1_greedy_eq_opt"])
    greedy_in = sum(1 for r in results if r["greedy_in_sweep"])
    has_ident = sum(1 for r in results if r["m2"]["has_identical"])
    spreads = sum(
        1 for r in results if r["m2"]["opt_spreads_identical"] is True
    )
    colo_rel = [
        r["m2"]["colocate_regret_rel"]
        for r in results
        if r["m2"]["colocate_regret_rel"] is not None
    ]
    opt_coll = sum(1 for r in results if r["opt_has_collision"])
    multitask = [r for r in results if r["n_tasks"] >= 2]

    def distribution(values: List[float]) -> dict[str, float | int | None]:
        if not values:
            return {
                "count": 0,
                "mean": None,
                "median": None,
                "p90": None,
                "p99": None,
                "max": None,
            }
        return {
            "count": len(values),
            "mean": sum(values) / len(values),
            "median": pctl(values, 0.5),
            "p90": pctl(values, 0.9),
            "p99": pctl(values, 0.99),
            "max": max(values),
        }

    return {
        "datasets_analyzed": n,
        "multitask_datasets": len(multitask),
        "mean_n_combos": sum(r["n_combos"] for r in results) / n,
        "m1_marginal_greedy": {
            "greedy_in_sweep_count": greedy_in,
            "greedy_in_sweep_fraction": greedy_in / n,
            "greedy_exact_optimum_count": greedy_eq,
            "greedy_exact_optimum_fraction": greedy_eq / n,
            "regret_relative": distribution(m1_rel),
            "coupled_gt_1pct_count": sum(value > 0.01 for value in m1_rel),
            "coupled_gt_1pct_fraction": (
                sum(value > 0.01 for value in m1_rel) / len(m1_rel)
                if m1_rel
                else None
            ),
            "coupled_gt_5pct_count": sum(value > 0.05 for value in m1_rel),
            "coupled_gt_5pct_fraction": (
                sum(value > 0.05 for value in m1_rel) / len(m1_rel)
                if m1_rel
                else None
            ),
            "coupled_gt_10pct_count": sum(value > 0.10 for value in m1_rel),
            "coupled_gt_10pct_fraction": (
                sum(value > 0.10 for value in m1_rel) / len(m1_rel)
                if m1_rel
                else None
            ),
        },
        "m2_identical_tasks": {
            "has_identical_count": has_ident,
            "has_identical_fraction": has_ident / n,
            "optimum_spreads_identical_count": spreads,
            "optimum_spreads_identical_fraction": (
                spreads / has_ident if has_ident else None
            ),
            "forced_colocation_regret_relative": distribution(colo_rel),
        },
        "m3_optimum_collision": {
            "collision_count": opt_coll,
            "collision_fraction": opt_coll / n,
            "avg_unique_platforms_multitask": (
                sum(r["opt_unique_plats"] for r in multitask) / len(multitask)
                if multitask
                else None
            ),
            "avg_tasks_multitask": (
                sum(r["n_tasks"] for r in multitask) / len(multitask)
                if multitask
                else None
            ),
        },
    }


def print_summary(base_name: str, summary: dict[str, Any]) -> None:
    n = summary["datasets_analyzed"]
    m1 = summary["m1_marginal_greedy"]
    m2 = summary["m2_identical_tasks"]
    m3 = summary["m3_optimum_collision"]
    regret = m1["regret_relative"]
    colo = m2["forced_colocation_regret_relative"]

    print(f"\n===== Separability diagnostic: {base_name} =====")
    print(
        f"Datasets analyzed: {n} "
        f"(multi-task >=2: {summary['multitask_datasets']})"
    )
    print(f"Mean n_combos: {summary['mean_n_combos']:.0f}")
    print("\n--- M1: marginal-greedy (independent per-task best) vs joint optimum ---")
    print(
        f"  greedy combo present in sweep: {m1['greedy_in_sweep_count']}/{n} "
        f"({100 * m1['greedy_in_sweep_fraction']:.1f}%)"
    )
    print(
        f"  greedy == optimum (exact):     {m1['greedy_exact_optimum_count']}/{n} "
        f"({100 * m1['greedy_exact_optimum_fraction']:.1f}%)"
    )
    if regret["count"]:
        print(
            "  regret_rel (greedy vs opt): "
            f"mean={100 * regret['mean']:.2f}%  "
            f"median={100 * regret['median']:.2f}%  "
            f"p90={100 * regret['p90']:.2f}%  "
            f"p99={100 * regret['p99']:.2f}%  "
            f"max={100 * regret['max']:.1f}%"
        )
        print(
            "  coupled datasets: "
            f">1%={100 * m1['coupled_gt_1pct_fraction']:.1f}%  "
            f">5%={100 * m1['coupled_gt_5pct_fraction']:.1f}%  "
            f">10%={100 * m1['coupled_gt_10pct_fraction']:.1f}%"
        )
    print("\n--- M2: identical (type,src) tasks => pointwise MUST co-assign ---")
    print(
        f"  datasets with >=2 identical tasks: {m2['has_identical_count']}/{n} "
        f"({100 * m2['has_identical_fraction']:.1f}%)"
    )
    if m2["has_identical_count"]:
        print(
            "  among identical: optimum SPREADS them: "
            f"{m2['optimum_spreads_identical_count']}/{m2['has_identical_count']} "
            f"({100 * m2['optimum_spreads_identical_fraction']:.1f}%)"
        )
    if colo["count"]:
        print(
            "  forced-colocation regret_rel (pointwise floor): "
            f"mean={100 * colo['mean']:.2f}%  "
            f"median={100 * colo['median']:.2f}%  "
            f"p90={100 * colo['p90']:.2f}%  "
            f"max={100 * colo['max']:.1f}%"
        )
    print("\n--- M3: does the OPTIMUM itself collide (2+ tasks same platform)? ---")
    print(
        f"  optimal combo has collision: {m3['collision_count']}/{n} "
        f"({100 * m3['collision_fraction']:.1f}%)"
    )
    print(
        "  avg unique platforms in optimum: "
        f"{m3['avg_unique_platforms_multitask']:.2f} / "
        f"avg n_tasks {m3['avg_tasks_multitask']:.2f} (multi-task)"
    )


def _load_integrity_manifest(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("status") != "clean" or manifest.get("clean") is not True:
        raise RuntimeError(f"Integrity manifest is not clean: {path}")
    return manifest, hashlib.sha256(raw).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus_dir", nargs="+", type=Path)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--integrity-manifest", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    integrity_manifest = None
    integrity_sha256 = None
    if args.integrity_manifest:
        integrity_manifest, integrity_sha256 = _load_integrity_manifest(
            args.integrity_manifest
        )

    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "integrity_manifest": (
            str(args.integrity_manifest) if args.integrity_manifest else None
        ),
        "integrity_manifest_sha256": integrity_sha256,
        "corpora": {},
    }
    for base in args.corpus_dir:
        if not base.is_dir():
            raise FileNotFoundError(f"Corpus directory not found: {base}")
        if integrity_manifest is not None:
            corpus_inventory = integrity_manifest["corpora"].get(base.name)
            if corpus_inventory is None:
                raise RuntimeError(
                    f"Corpus absent from integrity manifest: {base.name}"
                )
            retained_names = sorted(corpus_inventory["datasets"])
            ds_dirs = [base / name for name in retained_names]
            excluded = corpus_inventory["excluded_datasets"]
        else:
            ds_dirs = sorted(d for d in base.glob("ds_*") if d.is_dir())
            excluded = []
        if args.limit:
            ds_dirs = ds_dirs[: args.limit]
        if not ds_dirs:
            raise RuntimeError(f"No retained datasets under {base}")

        results: list[dict] = []
        for dataset_dir in ds_dirs:
            result = analyze_dataset(dataset_dir)
            if result is None:
                raise RuntimeError(f"Dataset is not analyzable: {dataset_dir}")
            result["dataset_id"] = dataset_dir.name
            results.append(result)
        summary = summarize_results(results)
        if summary["m1_marginal_greedy"]["greedy_in_sweep_count"] != len(results):
            raise RuntimeError(
                f"{base.name}: marginal greedy combo absent from one or more "
                "retained full sweeps"
            )
        print_summary(base.name, summary)
        report["corpora"][base.name] = {
            "path": str(base),
            "excluded_datasets": excluded,
            "summary": summary,
            "datasets": results,
        }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"\nFrozen report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
