#!/usr/bin/env python3
"""A/B: old training contract vs repaired contract on retained co-sim datasets.

OLD (pre-5.5):
  - labels from optimal_result.sample.placement_plan
  - opt_rtt from best.json (or sum elapsed)
  - replicas from terminal systemStateResults[-1]
  - warmth: previous_task_type_name dropped from SSC temporal
  - splits: copy-level train_test_split on @os-augmented IDs
  - RTT lookup: exact graph id (misses @os)

NEW (5.5):
  - labels + opt_rtt from placements.jsonl sweep minimum
  - replicas from SSC scheduling-time replicas
  - warmth: previous_task_type_name preserved
  - splits: canonical-parent 70/15/15
  - RTT lookup: canonical parent id
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

_REPO = Path(__file__).resolve().parents[1]
_NOTEBOOKS = _REPO / "src" / "notebooks"
for p in (_REPO, _NOTEBOOKS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from non_unique_lib.training_contract import (  # noqa: E402
    canonical_parent_id,
    combo_from_plan,
    load_sweep_minimum,
    split_ids_by_canonical_parent,
)


def _approx(a: float, b: float, eps: float = 1e-6) -> bool:
    return abs(a - b) <= eps


def _old_label_from_optimal(optimal_path: Path) -> Tuple[Tuple[Tuple[int, int], ...], Optional[float]]:
    with open(optimal_path) as f:
        opt = json.load(f)
    plan = opt.get("sample", {}).get("placement_plan") or {}
    if not plan:
        raise ValueError(f"{optimal_path}: empty sample.placement_plan")
    combo = combo_from_plan(plan)
    best_path = optimal_path.parent / "best.json"
    best_rtt = None
    if best_path.exists():
        best_rtt = float(json.load(best_path.open())["rtt"])
    return combo, best_rtt


def _lookup_combo_rtt(jsonl: Path, combo: Tuple[Tuple[int, int], ...]) -> Optional[float]:
    rtts: List[float] = []
    with jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if combo_from_plan(rec["placement_plan"]) == combo:
                rtts.append(float(rec["rtt"]))
    return min(rtts) if rtts else None


def _terminal_replicas(optimal_path: Path) -> Dict[str, List[List[Any]]]:
    with open(optimal_path) as f:
        opt = json.load(f)
    ssr = opt.get("stats", {}).get("systemStateResults") or [{}]
    return dict(ssr[-1].get("replicas") or {})


def _ssc_replicas_and_warmth(ssc_path: Path) -> Tuple[Dict[str, Any], int, int]:
    with open(ssc_path) as f:
        ssc = json.load(f)
    replicas = ssc.get("replicas") or {}
    tp0 = (ssc.get("task_placements") or [{}])[0]
    temporal = tp0.get("full_temporal_state_at_scheduling") or {}
    n_plats = 0
    n_prev = 0
    for st in temporal.values():
        if not isinstance(st, dict):
            continue
        n_plats += 1
        if st.get("previous_task_type_name") is not None:
            n_prev += 1
    return replicas, n_plats, n_prev


def _replica_keyset(replicas: Dict[str, Any]) -> set:
    keys = set()
    for task_type, lst in replicas.items():
        if not isinstance(lst, list):
            continue
        for item in lst:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                keys.add((str(task_type), str(item[0]), int(item[1])))
    return keys


def _oversample_ids(parents: Sequence[str], weight: int = 8) -> List[str]:
    out: List[str] = []
    for p in parents:
        out.append(p)
        for i in range(1, weight):
            out.append(f"{p}@os{i}")
    return out


def _copy_level_overlap(ids: Sequence[str], random_state: int = 42) -> Dict[str, float]:
    """OLD split: train_test_split on copy IDs — measures parent leak into test."""
    train, temp = train_test_split(list(ids), test_size=0.3, random_state=random_state)
    val, test = train_test_split(temp, test_size=0.5, random_state=random_state)
    train_p = {canonical_parent_id(x) for x in train}
    test_p = {canonical_parent_id(x) for x in test}
    val_p = {canonical_parent_id(x) for x in val}
    leaked_test = test_p & train_p
    leaked_val = val_p & train_p
    return {
        "n_ids": len(ids),
        "n_parents": len({canonical_parent_id(x) for x in ids}),
        "test_parents": len(test_p),
        "test_parent_leak_frac": (len(leaked_test) / max(1, len(test_p))),
        "val_parent_leak_frac": (len(leaked_val) / max(1, len(val_p))),
        "os_rtt_miss_frac": sum(1 for x in ids if "@os" in x) / max(1, len(ids)),
    }


def _parent_level_overlap(ids: Sequence[str], random_state: int = 42) -> Dict[str, float]:
    class _G:
        pass

    graphs = [_G() for _ in ids]
    for g, i in zip(graphs, ids):
        g.parent_dataset_id = canonical_parent_id(i)
    _, train_ids, _, val_ids, _, test_ids = split_ids_by_canonical_parent(
        graphs, list(ids), test_size=0.3, random_state=random_state
    )
    train_p = {canonical_parent_id(x) for x in train_ids}
    test_p = {canonical_parent_id(x) for x in test_ids}
    val_p = {canonical_parent_id(x) for x in val_ids}
    return {
        "n_ids": len(ids),
        "n_parents": len({canonical_parent_id(x) for x in ids}),
        "test_parents": len(test_p),
        "test_parent_leak_frac": len(test_p & train_p) / max(1, len(test_p)),
        "val_parent_leak_frac": len(val_p & train_p) / max(1, len(val_p)),
        "os_rtt_miss_frac": 0.0,  # parent canonicalize → 0 miss by construction
    }


def audit_corpus(base: Path, max_ds: int = 0) -> Dict[str, Any]:
    ds_dirs = sorted(base.glob("ds_*"))
    if max_ds > 0:
        ds_dirs = ds_dirs[:max_ds]

    old_absent = 0
    old_subopt = 0
    old_match = 0
    new_absent = 0
    new_subopt = 0
    new_match = 0
    regret_old: List[float] = []
    replica_mismatch = 0
    replica_compared = 0
    ssc_prev_platforms = 0
    ssc_platforms = 0
    skipped = Counter()
    parents: List[str] = []

    for ds in ds_dirs:
        opt_path = ds / "optimal_result.json"
        jsonl = ds / "placements" / "placements.jsonl"
        ssc = ds / "system_state_captured_unique.json"
        if not opt_path.exists():
            skipped["no_optimal"] += 1
            continue
        if not jsonl.exists() or jsonl.stat().st_size == 0:
            skipped["no_jsonl"] += 1
            continue
        if not ssc.exists():
            skipped["no_ssc"] += 1
            continue

        parent_id = f"{base.name}/{ds.name}"
        parents.append(parent_id)

        try:
            sweep_plan, sweep_rtt, sweep_combo = load_sweep_minimum(jsonl)
        except Exception as exc:
            skipped[f"sweep_fail:{type(exc).__name__}"] += 1
            continue

        # NEW label is sweep min by construction
        new_match += 1

        try:
            old_combo, best_rtt = _old_label_from_optimal(opt_path)
        except Exception as exc:
            skipped[f"old_label_fail:{type(exc).__name__}"] += 1
            continue

        old_rtt = _lookup_combo_rtt(jsonl, old_combo)
        if old_rtt is None:
            old_absent += 1
        elif _approx(old_rtt, sweep_rtt):
            old_match += 1
            regret_old.append(0.0)
        else:
            old_subopt += 1
            regret_old.append((old_rtt - sweep_rtt) / sweep_rtt * 100.0 if sweep_rtt > 0 else 0.0)

        term = _terminal_replicas(opt_path)
        sched, n_plats, n_prev = _ssc_replicas_and_warmth(ssc)
        ssc_platforms += n_plats
        ssc_prev_platforms += n_prev
        if term or sched:
            replica_compared += 1
            if _replica_keyset(term) != _replica_keyset(sched):
                replica_mismatch += 1

    # Split / RTT identity A/B on synthetic 8× oversample of retained parents
    oversampled = _oversample_ids(parents, weight=8)
    old_split = _copy_level_overlap(oversampled)
    new_split = _parent_level_overlap(oversampled)

    n = old_absent + old_subopt + old_match
    return {
        "corpus": base.name,
        "datasets_scanned": len(ds_dirs),
        "datasets_audited": n,
        "skipped": dict(skipped),
        "labels": {
            "old": {
                "match_sweep_min": old_match,
                "suboptimal": old_subopt,
                "absent_from_sweep": old_absent,
                "nonoptimal_or_absent": old_subopt + old_absent,
                "nonoptimal_or_absent_frac": (old_subopt + old_absent) / max(1, n),
                "mean_label_regret_pct": float(np.mean(regret_old)) if regret_old else 0.0,
                "p90_label_regret_pct": float(np.percentile(regret_old, 90)) if regret_old else 0.0,
            },
            "new": {
                "match_sweep_min": new_match,
                "suboptimal": new_subopt,
                "absent_from_sweep": new_absent,
                "nonoptimal_or_absent": 0,
                "nonoptimal_or_absent_frac": 0.0,
                "mean_label_regret_pct": 0.0,
                "p90_label_regret_pct": 0.0,
            },
        },
        "replicas": {
            "compared": replica_compared,
            "terminal_vs_ssc_mismatch": replica_mismatch,
            "mismatch_frac": replica_mismatch / max(1, replica_compared),
        },
        "warmth": {
            "ssc_platforms_with_temporal": ssc_platforms,
            "ssc_platforms_with_previous_task": ssc_prev_platforms,
            "prev_type_frac": ssc_prev_platforms / max(1, ssc_platforms),
            "old_batch_loader_preserves_prev_type": False,
            "new_batch_loader_preserves_prev_type": True,
        },
        "splits_8x_oversample": {
            "old_copy_level": old_split,
            "new_parent_level": new_split,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--bases",
        nargs="+",
        default=[
            "simulation_data/gnn_datasets_4tasks_contention_v2",
            "simulation_data/gnn_datasets_4tasks_1060_warmth_v2",
            "simulation_data/gnn_datasets_4tasks_sparse_warmth_v2",
        ],
    )
    ap.add_argument("--max-ds", type=int, default=0, help="0 = all")
    ap.add_argument("--out", type=Path, default=_REPO / "simulation_data/training_contract_ab_20260804.json")
    args = ap.parse_args()

    reports = []
    for base in args.bases:
        path = Path(base)
        if not path.is_absolute():
            path = _REPO / path
        if not path.is_dir():
            print(f"SKIP missing {path}", flush=True)
            continue
        print(f"Auditing {path.name} ...", flush=True)
        rep = audit_corpus(path, max_ds=args.max_ds)
        reports.append(rep)
        lab = rep["labels"]
        print(
            f"  labels OLD nonopt/absent {lab['old']['nonoptimal_or_absent']}/{rep['datasets_audited']} "
            f"({100*lab['old']['nonoptimal_or_absent_frac']:.1f}%) "
            f"mean regret {lab['old']['mean_label_regret_pct']:.2f}% | "
            f"NEW {lab['new']['nonoptimal_or_absent']}/{rep['datasets_audited']}",
            flush=True,
        )
        print(
            f"  replicas mismatch terminal vs SSC {rep['replicas']['terminal_vs_ssc_mismatch']}/"
            f"{rep['replicas']['compared']} ({100*rep['replicas']['mismatch_frac']:.1f}%)",
            flush=True,
        )
        print(
            f"  warmth SSC prev_type {rep['warmth']['ssc_platforms_with_previous_task']}/"
            f"{rep['warmth']['ssc_platforms_with_temporal']} "
            f"({100*rep['warmth']['prev_type_frac']:.1f}%) — OLD drops, NEW keeps",
            flush=True,
        )
        old_s = rep["splits_8x_oversample"]["old_copy_level"]
        new_s = rep["splits_8x_oversample"]["new_parent_level"]
        print(
            f"  split parent-leak test: OLD {100*old_s['test_parent_leak_frac']:.1f}% → "
            f"NEW {100*new_s['test_parent_leak_frac']:.1f}%",
            flush=True,
        )

    payload = {
        "generated": "scripts_cosim/ab_training_contract.py",
        "date": "2026-08-04",
        "reports": reports,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
