"""workload_fix_v1 W4 -- task-type mix relabelling.

Opt-in. Relabels the events of a workload into all defined task types in an equal mix, from a dedicated random
stream. Only the type labels change: ``application.name``, the single-node ``application.dag`` and the key of
``application.demand_scale`` (its value stays). Timestamps, origins, peer groups, QoS and ``peer_exchange`` are
untouched, so a relabelled window is the W2 window with different types.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence

TASK_MIXES = ("none", "wf1_v1")

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "nofs-ids"


def task_mix_rng(seed: int) -> random.Random:
    """Dedicated stream: no other draw of the workload or the generator moves when the mix is switched on."""
    return random.Random(f"{seed}:task_mix_wf1_v1")


def require_task_mix(name: str) -> str:
    if name not in TASK_MIXES:
        raise ValueError(f"task mix {name!r}; expected one of {TASK_MIXES}")
    return name


def defined_application_types(data_dir: Path = DATA_DIR) -> Dict[str, Dict[str, Any]]:
    """``{task type: application}`` for every type in task-types.json, from application-types.json (the
    application whose DAG is that type alone)."""
    types = list(json.loads((data_dir / "task-types.json").read_text()))
    apps = json.loads((data_dir / "application-types.json").read_text())
    by_type: Dict[str, Dict[str, Any]] = {}
    for app in apps.values():
        dag = app["dag"]
        if len(dag) == 1 and not next(iter(dag.values())):
            by_type[next(iter(dag))] = app
    missing = [t for t in types if t not in by_type]
    if missing:
        raise ValueError(f"task type(s) {missing} have no single-node application in application-types.json")
    return {t: by_type[t] for t in types}


def balanced_labels(n: int, types: Sequence[str], rng: random.Random) -> List[str]:
    """``n`` labels with every type's count within one of the others, in a seeded random order."""
    labels = [types[i % len(types)] for i in range(n)]
    rng.shuffle(labels)
    return labels


def relabel_events(events: Sequence[Dict[str, Any]], seed: int, applications: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Events with the type labels redrawn; everything else is the input's, key for key."""
    types = list(applications)
    labels = balanced_labels(len(events), types, task_mix_rng(seed))
    out = []
    for event, new_type in zip(events, labels):
        app = event["application"]
        dag = app.get("dag")
        if not isinstance(dag, dict) or len(dag) != 1 or next(iter(dag.values())):
            raise ValueError(f"event application {app.get('name')!r} is not a single-node DAG; cannot relabel: {dag!r}")
        (old_type,) = dag
        new_event = copy.deepcopy(event)
        new_app = new_event["application"]
        new_app["name"] = applications[new_type]["name"]
        new_app["dag"] = copy.deepcopy(applications[new_type]["dag"])
        if "demand_scale" in app:
            if set(app["demand_scale"]) != {old_type}:
                raise ValueError(f"demand_scale keys {sorted(app['demand_scale'])} != the event's type {old_type!r}")
            new_app["demand_scale"] = {new_type: app["demand_scale"][old_type]}
        out.append(new_event)
    return out


def task_mix_meta(applications: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    return {"task_mix": "wf1_v1", "types": list(applications), "stream": "<seed>:task_mix_wf1_v1", "mix": "equal"}
