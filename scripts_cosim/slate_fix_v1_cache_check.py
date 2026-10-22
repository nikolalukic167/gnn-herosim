#!/usr/bin/env python3
"""slate_fix_v1 pre-training check (docs/lineages/archive/slate_fix_v1.md): the recached uncapped corpus is the capped accel_replica_v1 cache with
only the candidate slate changed. Reads two metadata.json files and the split artifact; no graphs are loaded, so it is login-node safe.

  slate_fix_v1_cache_check.py --new <uncapped cache dir> --old <accel_replica_v1 cache dir> --split experiments/accel_replica_v1_split.json
      [--out check.json]

Exit 0 only when every check passes; the failures are printed one per line.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

UNCAPPED = "declared_pruning_v1_uncapped"
CAPPED = "declared_pruning_v1"
SPLIT_SHA = "f5bc4f51106d71a1830febec8c06dca66121a81336acd5662cf24b7997cf7e53"  # the split accel_replica_v1's g4 and twin checkpoints record
# Everything a checkpoint's contract or the feature builder reads from the cache, other than the slate, must be unchanged.
SAME_KEYS = ("partial_state_contract", "physics_env", "dag_primary_alpha_key", "inference_feature_layout", "queue_feature_contract",
             "topology_feature_contract", "platform_feature_dim", "dag_task_type_vocab", "label_objective", "peer_exchange_block",
             "dag_partial_state", "queue_norm_mode", "training_contract")


def check(new: Dict[str, Any], old: Dict[str, Any], split: Dict[str, Any], split_sha: str) -> List[str]:
    bad = []
    if new.get("candidate_slate") != UNCAPPED:
        bad.append(f"new cache candidate_slate={new.get('candidate_slate')!r}, expected {UNCAPPED!r}")
    rules = new.get("candidate_slate_rules")
    if rules is not None and sorted(rules) != sorted((CAPPED, UNCAPPED)):
        bad.append(f"new cache candidate_slate_rules={rules!r}; the only mixture allowed is {[CAPPED, UNCAPPED]}")
    if old.get("candidate_slate") != CAPPED:
        bad.append(f"old cache candidate_slate={old.get('candidate_slate')!r}, expected {CAPPED!r}")
    for k in SAME_KEYS:
        if new.get(k) != old.get(k):
            bad.append(f"{k} differs: new {json.dumps(new.get(k))[:120]} old {json.dumps(old.get(k))[:120]}")
    if split_sha != SPLIT_SHA:
        bad.append(f"split sha256 {split_sha[:12]}... is not the registered {SPLIT_SHA[:12]}...")
    new_parents, old_parents = set(new.get("parent_dataset_ids") or []), set(old.get("parent_dataset_ids") or [])
    if new_parents != old_parents:
        bad.append(f"parent ids differ: {len(new_parents - old_parents)} only in new, {len(old_parents - new_parents)} only in old")
    covered = set(split.get("train") or []) | set(split.get("val") or []) | set(split.get("test") or [])
    missing = new_parents - covered
    if missing:
        bad.append(f"{len(missing)} new-cache parents are not in the split (e.g. {sorted(missing)[:3]})")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("--old", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if "S7_FILLS_THIS" in str(a.new):
        print(f"FAIL LOUD: --new is still the placeholder {a.new}", file=sys.stderr)
        return 1
    new = json.loads((a.new / "metadata.json").read_text())
    old = json.loads((a.old / "metadata.json").read_text())
    raw = a.split.read_bytes()
    split = json.loads(raw)
    bad = check(new, old, split, hashlib.sha256(raw).hexdigest())
    report = {
        "new": str(a.new.resolve()), "old": str(a.old.resolve()), "split": str(a.split),
        "candidate_slate_rules": new.get("candidate_slate_rules"),
        "num_graphs": {"new": new.get("num_graphs"), "old": old.get("num_graphs")},
        "num_rtt_combo_rows": {"new": new.get("num_rtt_combo_rows"), "old": old.get("num_rtt_combo_rows")},
        "statistics": {"new": new.get("statistics"), "old": old.get("statistics")},
        "split_counts": {k: len(split.get(k) or []) for k in ("train", "val", "test")},
        "failures": bad,
    }
    if a.out:
        a.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("num_graphs", "num_rtt_combo_rows", "split_counts", "candidate_slate_rules")}))
    for b in bad:
        print("FAIL:", b)
    print("PASS" if not bad else f"FAIL LOUD: {len(bad)} check(s) failed")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
