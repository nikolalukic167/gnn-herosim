"""Plan index -> task id for a co-sim dataset.

A dataset's plan is keyed by the task index of its `workload.json`, whose events are in application order. The snapshot's
`fidelity.batch` lists the same tasks in the live batch order, which differs in most datasets (425 of 604 in corpus_b2_ext).
Reading plan index i as snapshot batch[i] puts a task on another type's platform. The mapping is the one
`wf1_fidelity_parity.py` uses: take the snapshot's tasks of each type in snapshot order, in the dataset's type order."""
from typing import Any, Dict, List


def plan_task_ids(workload_events: List[Dict[str, Any]], fidelity_batch: List[Dict[str, Any]]) -> List[int]:
    types = [next(iter(e["application"]["dag"])) for e in workload_events]
    by_type: Dict[str, List[int]] = {}
    for rec in fidelity_batch:
        by_type.setdefault(rec["fn"], []).append(int(rec["gid"]))
    if sorted(types) != sorted(rec["fn"] for rec in fidelity_batch):
        raise ValueError(f"dataset task types {types} do not match the snapshot batch {[r['fn'] for r in fidelity_batch]}")
    return [by_type[t].pop(0) for t in types]
