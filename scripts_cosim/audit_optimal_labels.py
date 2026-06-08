#!/usr/bin/env python3
"""
Audit optimal_result.json labels against placements/placements.jsonl.

Compares the placement combo stored in optimal_result.json to the min-RTT
combo(s) in placements.jsonl, broken down by task count and mismatch severity.

Default paths target the 3705 and 1060 four-task corpora under run_queue_big.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BASES = [
    PROJECT_ROOT / "simulation_data/artifacts/run_queue_big/gnn_datasets_4tasks",
    PROJECT_ROOT / "simulation_data/artifacts/run_queue_big/gnn_datasets_4tasks_1060",
]


def combo_from_plan(plan: Dict[str, Any]) -> Tuple[Tuple[int, int], ...]:
    keys = sorted(plan.keys(), key=lambda k: int(k))
    return tuple(tuple(plan[k]) for k in keys)


def load_label_plan(optimal_result_path: Path) -> Tuple[Dict[str, Any], Tuple[Tuple[int, int], ...], int]:
    with open(optimal_result_path, "r") as f:
        opt = json.load(f)
    plan = opt.get("sample", {}).get("placement_plan")
    if not plan:
        plan = opt["config"]["infrastructure"]["forced_placements"]
    return opt, combo_from_plan(plan), len(plan)


def scan_placements(placements_path: Path) -> Tuple[Optional[float], Optional[Tuple[Tuple[int, int], ...]], set]:
    min_rtt: Optional[float] = None
    min_combo: Optional[Tuple[Tuple[int, int], ...]] = None
    min_combos: set = set()

    with open(placements_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            combo = combo_from_plan(rec["placement_plan"])
            rtt = float(rec["rtt"])
            if min_rtt is None or rtt < min_rtt:
                min_rtt = rtt
                min_combo = combo
                min_combos = {combo}
            elif min_rtt is not None and abs(rtt - min_rtt) <= 1e-4:
                min_combos.add(combo)

    return min_rtt, min_combo, min_combos


def lookup_opt_rtt(placements_path: Path, opt_combo: Tuple[Tuple[int, int], ...]) -> Optional[float]:
    rtts: List[float] = []
    with open(placements_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if combo_from_plan(rec["placement_plan"]) == opt_combo:
                rtts.append(float(rec["rtt"]))
    return min(rtts) if rtts else None


def classify_mismatch(delta: Optional[float]) -> str:
    if delta is None:
        return "not in placements"
    if delta <= 1e-4:
        return "tie-ish"
    if delta < 0.01:
        return "worse <0.01s"
    if delta < 0.1:
        return "worse 0.01-0.1s"
    if delta < 1.0:
        return "worse 0.1-1s"
    return "worse >1s"


def analyze_base(base: Path, label: Optional[str] = None) -> Dict[str, Any]:
    if not base.is_dir():
        raise FileNotFoundError(f"Dataset base not found: {base}")

    title = label or base.name
    by_n_tasks = Counter()
    match_by_n: Counter = Counter()
    mismatch_by_n: Counter = Counter()
    subopt_cats: Counter = Counter()
    skipped = Counter()
    best_rtt_mismatch = 0
    match_count = 0
    mismatch_count = 0
    single_row_mismatch = 0
    total = 0

    for ds in sorted(base.glob("ds_*")):
        optimal_path = ds / "optimal_result.json"
        placements_path = ds / "placements/placements.jsonl"
        best_path = ds / "best.json"

        if not optimal_path.exists():
            skipped["no optimal_result.json"] += 1
            continue
        if not placements_path.exists():
            skipped["no placements/placements.jsonl"] += 1
            continue
        if not best_path.exists():
            skipped["no best.json"] += 1
            continue

        total += 1
        _, opt_combo, n_plan = load_label_plan(optimal_path)
        by_n_tasks[n_plan] += 1

        with open(best_path, "r") as f:
            best_rtt = float(json.load(f)["rtt"])

        min_rtt, min_combo, min_combos = scan_placements(placements_path)
        if min_rtt is None or min_combo is None:
            skipped["empty placements.jsonl"] += 1
            total -= 1
            continue

        if abs(best_rtt - min_rtt) > 1e-4:
            best_rtt_mismatch += 1

        if opt_combo != min_combo:
            single_row_mismatch += 1

        if opt_combo in min_combos:
            match_count += 1
            match_by_n[n_plan] += 1
        else:
            mismatch_count += 1
            mismatch_by_n[n_plan] += 1
            opt_rtt = lookup_opt_rtt(placements_path, opt_combo)
            delta = None if opt_rtt is None else opt_rtt - min_rtt
            subopt_cats[classify_mismatch(delta)] += 1

    task_rows = []
    for n in sorted(by_n_tasks):
        m = match_by_n.get(n, 0)
        mm = mismatch_by_n.get(n, 0)
        task_rows.append(
            {
                "num_tasks": n,
                "total": by_n_tasks[n],
                "match": m,
                "mismatch": mm,
                "mismatch_rate_pct": 100.0 * mm / (m + mm) if (m + mm) else 0.0,
            }
        )

    return {
        "label": title,
        "base": str(base),
        "analyzed": total,
        "skipped": dict(skipped),
        "task_counts": task_rows,
        "match": match_count,
        "match_pct": 100.0 * match_count / total if total else 0.0,
        "mismatch": mismatch_count,
        "mismatch_pct": 100.0 * mismatch_count / total if total else 0.0,
        "single_min_row_mismatch": single_row_mismatch,
        "single_min_row_mismatch_pct": 100.0 * single_row_mismatch / total if total else 0.0,
        "best_json_rtt_mismatch": best_rtt_mismatch,
        "mismatch_breakdown": dict(subopt_cats),
    }


def print_report(result: Dict[str, Any]) -> None:
    print(f"\n{'=' * 60}")
    print(result["label"])
    print(f"{'=' * 60}")
    print(f"Base: {result['base']}")
    print(f"Analyzed: {result['analyzed']}")
    if result["skipped"]:
        print(f"Skipped: {result['skipped']}")

    print("\nTask count (len placement_plan in optimal_result.json):")
    for row in result["task_counts"]:
        print(
            f"  {row['num_tasks']} tasks: {row['total']} total | "
            f"match={row['match']} mismatch={row['mismatch']} "
            f"({row['mismatch_rate_pct']:.1f}% mismatch)"
        )

    print(
        f"\nVs min-RTT set: match={result['match']} ({result['match_pct']:.1f}%) "
        f"mismatch={result['mismatch']} ({result['mismatch_pct']:.1f}%)"
    )
    if result["best_json_rtt_mismatch"]:
        print(f"best.json RTT != min placements RTT: {result['best_json_rtt_mismatch']}")
    print(
        f"Single min-row mismatch: {result['single_min_row_mismatch']}/"
        f"{result['analyzed']} ({result['single_min_row_mismatch_pct']:.2f}%)"
    )

    if result["mismatch"]:
        print(f"\nMismatch detail ({result['mismatch']}):")
        for key, count in sorted(
            result["mismatch_breakdown"].items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            pct = 100.0 * count / result["mismatch"]
            print(f"  {count:4d} ({pct:5.1f}%) {key}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit optimal_result.json labels vs placements/placements.jsonl",
    )
    parser.add_argument(
        "bases",
        nargs="*",
        type=Path,
        help="Dataset base directories (default: 3705 + 1060 four-task corpora)",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        help="Optional labels aligned with bases (same count as bases)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of text reports",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bases = args.bases or DEFAULT_BASES
    labels = args.labels or []

    if labels and len(labels) != len(bases):
        print("error: --labels count must match bases count", file=sys.stderr)
        return 2

    results = []
    for i, base in enumerate(bases):
        label = labels[i] if i < len(labels) else None
        try:
            result = analyze_base(base.resolve(), label=label)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        results.append(result)
        if not args.json:
            print_report(result)

    if args.json:
        print(json.dumps(results, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
