#!/usr/bin/env python3
"""peer_affinity_warm_v1: cut brute-force-labelled co-sim datasets from LIVE cluster snapshots.

Every peer_affinity corpus so far captured its state from a cluster that the autoscaler had
not built yet: one replica per type per host, chosen first-suitable (so never `xavierGpu`),
queues drawn from Poisson(2). Served, the same checkpoints meet a cluster the autoscaler
DID build under a 2,650-arrival/s trace -- 14 replicas per type, `xavierGpu` among them,
queues 11k-19k deep (docs/lineages/peer_affinity_v1.md, H5). This script closes that gap
at the source: it takes live-audit snapshots (LIVE_AUDIT_SNAPSHOT_PATH, one JSON per
batch, written by the knative_network_batch / gnn schedulers) and turns each into a
standard dataset directory whose infrastructure carries the snapshot's cluster state as a
`live_snapshot_seed` (src/placement/live_snapshot_seed.py) and whose workload is the
snapshot's own peer group, then runs the ordinary brute-force sweep on it
(generate_gnn_datasets_fast.generate_single_dataset) so every downstream tool -- SSC
rewrite, alpha pre-scan, prepare_graphs_cache, the split artifact, the trainers -- sees
exactly the artifacts a generated dataset has.

Candidate subsampling. A warm batch task has 6-10 reachable replicas (measured), so the
full product is ~10^8 plans and cannot be enumerated (the generated corpora sit at a
median 20k). Per task TYPE a random subset of R live replicas is marked `candidate: true`
and the rest `candidate: false`: the latter are replayed with their backlog (they load the
cluster exactly as live) but are not offered to the sweep and therefore are not candidate
edges in the cache -- so the label is the optimum over exactly the slate the model is
trained on. Among the draws whose per-task reachable-candidate product fits
--target-combos the most balanced one is taken (largest minimum per task, then most tasks
with >= 2); the draw is seeded by (--seed, snapshot_id) and recorded in the dataset.

Alignment. Only batches that ARE a peer group (consecutive ids, group-aligned, every peer
inside) are usable: a pair that straddles two batches cannot be priced by a batch co-sim.
Capture with GNN_BATCH_BY_PEER_GROUP=1 (gnn arms) or KNATIVE_BATCH_BY_PEER_GROUP=1.

Usage (physics env exactly as the generation sbatch sets it -- asserted):
  HEROSIM_PEER_EXCHANGE=1 HEROSIM_RETAIN_TASK_TIMES=1 HEROSIM_RETAIN_PEER_STATS=1 \\
  GNN_CAPTURE_DATASET_STATE=0 COSIM_SUPPRESS_SIM_PRINTS=1 \\
  python3 scripts_cosim/make_warm_corpus.py \\
      --snapshots <snapshots.jsonl> --trace <trace-with-peer_exchange.json> \\
      --cell-config <space_with_network.json of the cell the snapshots were captured on> \\
      --output-dir simulation_data/gnn_datasets_peer_affinity_warm_v1_<tag> \\
      --source-tag knb_c9001 --start-index 0 --workers 31
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts_cosim.generate_gnn_datasets_fast import (  # noqa: E402
    generate_single_dataset,
    json_dumps_pretty,
)
from src.placement.live_snapshot_seed import build_live_snapshot_seed, inject_synthetic_backlog  # noqa: E402

REQUIRED_ENV = {"HEROSIM_PEER_EXCHANGE": "1"}
WARM_SNAPSHOT_FILE = "warm_snapshot.json"


class SnapshotRejected(ValueError):
    """The snapshot cannot become a dataset; the message says why (recorded, never hidden)."""


# ---------------------------------------------------------------------------------------
# alignment + workload
# ---------------------------------------------------------------------------------------

def batch_task_ids(snapshot: Dict[str, Any]) -> List[int]:
    return [int(t["task_id"]) for t in snapshot.get("tasks", [])]


def check_aligned_peer_group(snapshot: Dict[str, Any], group_size: int) -> List[int]:
    ids = batch_task_ids(snapshot)
    if len(ids) != group_size:
        raise SnapshotRejected(f"batch has {len(ids)} tasks, need {group_size}")
    if ids != list(range(ids[0], ids[0] + group_size)):
        raise SnapshotRejected(f"batch ids are not consecutive: {ids}")
    if ids[0] % group_size != 0:
        raise SnapshotRejected(f"batch starts at id {ids[0]}, not a multiple of {group_size}")
    return ids


def build_batch_workload(
    snapshot: Dict[str, Any], trace: Dict[str, Any], ids: Sequence[int], app_order: Sequence[str]
) -> Dict[str, Any]:
    """The snapshot's peer group as a co-sim workload: the trace's own events for those
    ids (type, source, qos, demand_scale) at t = 0 like every generated corpus, and the
    trace's peer pairs among them re-indexed. A pair reaching outside the batch is a
    rejection, not a dropped edge.

    Event ORDER is the wsc application order (`app_order`), stable within an application:
    the sweep regroups a workload per application before the simulator assigns task ids
    and fails loud if that changes the order (executecosimulation.flatten_workloads), so
    the co-sim's task 0..k-1 is the grouped order, not the trace's arrival order. Peer ids
    are remapped to it; `task_ids` in the provenance keep the trace ids."""
    events_src = trace["events"]
    rank = {str(app): i for i, app in enumerate(app_order)}
    ordered: List[Tuple[int, int]] = []  # (gid, snapshot task index)
    for k, gid in enumerate(ids):
        if gid >= len(events_src):
            raise SnapshotRejected(f"task id {gid} beyond the trace ({len(events_src)} events)")
        app = str(events_src[gid]["application"].get("name"))
        if app not in rank:
            raise SnapshotRejected(f"event {gid} application {app!r} is not in the cell's wsc {list(rank)}")
        ordered.append((gid, k))
    ordered.sort(key=lambda t: rank[str(events_src[t[0]]["application"]["name"])])
    local = {gid: i for i, (gid, _k) in enumerate(ordered)}
    events: List[Dict[str, Any]] = []
    for gid, k in ordered:
        task = snapshot["tasks"][k]
        if gid >= len(events_src):
            raise SnapshotRejected(f"task id {gid} beyond the trace ({len(events_src)} events)")
        ev = events_src[gid]
        dag = ev["application"]["dag"]
        if not isinstance(dag, dict) or len(dag) != 1:
            raise SnapshotRejected(f"event {gid} is not a single-task application: {dag!r}")
        (ev_type,) = dag.keys()
        if str(ev_type) != str(task["task_type"]):
            raise SnapshotRejected(
                f"task {gid}: snapshot type {task['task_type']!r} != trace type {ev_type!r} "
                "(is this the trace the snapshots were captured on?)"
            )
        if str(ev.get("node_name")) != str(task.get("source_node")):
            raise SnapshotRejected(
                f"task {gid}: snapshot source {task.get('source_node')!r} != trace "
                f"{ev.get('node_name')!r}"
            )
        events.append(
            {
                "timestamp": 0.0,
                "application": deepcopy(ev["application"]),
                "qos": deepcopy(ev.get("qos") or {"name": "medium", "maxDurationDeviation": 15}),
                "node_name": str(ev["node_name"]),
            }
        )
    pairs: List[List[Any]] = []
    id_set = set(ids)
    outside = 0
    for i, j, payload in trace.get("peer_exchange") or []:
        i, j = int(i), int(j)
        if i in id_set and j in id_set:
            pairs.append([local[i], local[j], float(payload)])
        elif i in id_set or j in id_set:
            outside += 1
    if outside:
        raise SnapshotRejected(f"{outside} peer pair(s) reach outside the batch")
    if not pairs:
        raise SnapshotRejected("batch carries no peer pairs")
    return {
        "rps": len(events), "duration": 1, "events": events, "peer_exchange": pairs,
        # co-sim task id -> trace task id, so a row of this dataset can be traced back
        "trace_task_ids": [gid for gid, _k in ordered],
    }


# ---------------------------------------------------------------------------------------
# candidate subsampling
# ---------------------------------------------------------------------------------------

def _qkey(spec: Dict[str, Any]) -> str:
    return f"{spec['node_name']}:{int(spec['platform_id'])}"


CAP_ALPHA_TIGHTEST = 2.0  # tightest rung of DAG_ALPHA_LADDER; monotone, so it implies every rung


def batch_demands(
    snapshot: Dict[str, Any],
    ids: Sequence[int],
    trace: Dict[str, Any],
    task_types_db: Dict[str, dict],
) -> List[Tuple[str, Dict[str, Tuple[str, float]]]]:
    """Per batch task: (type, {queue_key: (node_name, demand)}) over its LIVE candidates,
    with demand = demand_scale x memoryRequirements[type][platform_type] -- the scorer's
    and the cache's formula (score_route_b_contention.Dataset), so a draw judged feasible
    here is feasible under training-contract 5.5. Fails loud on a missing table entry."""
    events = trace["events"]
    out: List[Tuple[str, Dict[str, Tuple[str, float]]]] = []
    for k, gid in enumerate(ids):
        task = snapshot["tasks"][k]
        ttype = str(task["task_type"])
        app = events[gid]["application"]
        scales = app.get("demand_scale") or {}
        scale = float(scales.get(ttype, 1.0))
        mem = task_types_db[ttype].get("memoryRequirements", {})
        cands: Dict[str, Tuple[str, float]] = {}
        for c in task.get("candidates", []):
            ptype = c.get("platform_type")
            if ptype not in mem:
                raise RuntimeError(
                    f"task {gid}: no memoryRequirements[{ttype}][{ptype}] -- refusing to invent a demand"
                )
            cands[str(c["queue_key"])] = (str(c["node_name"]), scale * float(mem[ptype]))
        out.append((ttype, cands))
    return out


def cap_feasible(
    demands: Sequence[Tuple[str, Dict[str, Tuple[str, float]]]],
    subset: Optional[Dict[str, Set[str]]],
    alpha: float = CAP_ALPHA_TIGHTEST,
) -> bool:
    """True iff some plan over the slate keeps every node's load <= alpha x the largest
    single candidate demand on that node (Dataset.node_caps 'alpha_max' + plan_feasible;
    a node whose candidate demands are all zero is uncapped). Exhaustive with pruning --
    10 tasks x <= 9 candidates."""
    slate: List[List[Tuple[str, float]]] = []
    for ttype, cands in demands:
        keep = [v for k, v in cands.items() if subset is None or k in subset.get(ttype, set())]
        if not keep:
            return False
        slate.append(keep)
    peak: Dict[str, float] = {}
    for options in slate:
        for node, d in options:
            peak[node] = max(peak.get(node, 0.0), d)
    caps = {node: alpha * d for node, d in peak.items() if d > 0.0}
    order = sorted(range(len(slate)), key=lambda i: len(slate[i]))
    load: Dict[str, float] = {}
    eps = 1e-9

    def rec(pos: int) -> bool:
        if pos == len(order):
            return True
        for node, d in slate[order[pos]]:
            new = load.get(node, 0.0) + d
            if new <= caps.get(node, math.inf) + eps:
                load[node] = new
                if rec(pos + 1):
                    return True
                load[node] = new - d
        return False

    return rec(0)


def reactive_plan_keys(snapshot: Dict[str, Any]) -> Dict[str, Set[str]]:
    """Per task type, the replica keys the live shortest-queue rule would use on this state.

    drainable_debug_v1 D1 (2026-09-14). The candidate subsample is drawn for sweep size, so it
    can drop the very replica a policy would have chosen -- and then that policy's plan is
    absent from the enumerated sweep and cannot be scored at all. Skipping such datasets is
    not an option: it keeps only the states where the policy agreed with the draw, which are
    the states it looks best on. Forcing these keys into the subset costs at most one replica
    per task type and makes the comparison scoreable on every dataset.

    Mirrors `KnativeScheduler.placement` (`src/policy/knative_network/scheduler.py:180-192`),
    including the depth increment between tasks and the arrival (trace id) ordering.
    """
    depth: Dict[str, int] = {}
    keys: Dict[str, Set[str]] = {}
    for task in sorted(snapshot.get("tasks") or [], key=lambda t: int(t["task_id"])):
        cands = task.get("candidates") or []
        if not cands:
            raise SnapshotRejected(f"task {task.get('task_id')} has no live candidate")
        for c in cands:
            depth.setdefault(c["queue_key"], int(c.get("queue_length", 0) or 0))
        pool = [c for c in cands if c.get("initialized")] or cands
        chosen = min(
            pool, key=lambda c: (depth[c["queue_key"]], int(c["node_id"]), int(c["platform_id"]))
        )
        keys.setdefault(str(task["task_type"]), set()).add(chosen["queue_key"])
        depth[chosen["queue_key"]] += 1
    return keys


def choose_candidates(
    snapshot: Dict[str, Any],
    rng: random.Random,
    target_combos: int,
    max_combos: int,
    attempts: int = 200,
    demands: Optional[Sequence[Tuple[str, Dict[str, Tuple[str, float]]]]] = None,
    force_keys: Optional[Dict[str, Set[str]]] = None,
    min_choice_fraction: float = 0.5,
) -> Tuple[Dict[str, Set[str]], Dict[str, Any]]:
    """Per task type, the subset of live replicas offered to the sweep.

    Pool = replicas of the type reachable by at least one batch task (the snapshot's own
    candidate lists, computed live by the scheduler's reachability rule). R is searched
    from the largest pool size down; the first draw whose per-task product
    prod_t |subset ∩ reach(t)| is <= target_combos with every task keeping >= 1 candidate
    and at least half keeping >= 2 is taken. With `demands`, a draw must also admit a plan
    under the alpha=2.0 node caps (`cap_feasible`): W0 attempt 2 (2026-09-13) lost 46/100
    datasets at the cache because the balanced draw packed a type's candidates onto too
    few nodes for any plan to fit, and a dataset without a feasible row carries no label.
    Returns (subset by type, record)."""
    tasks = snapshot["tasks"]
    reach: List[Tuple[str, Set[str]]] = [
        (str(t["task_type"]), {c["queue_key"] for c in t.get("candidates", [])}) for t in tasks
    ]
    pools: Dict[str, List[str]] = {}
    for ttype, keys in reach:
        pools.setdefault(ttype, [])
        for k in sorted(keys):
            if k not in pools[ttype]:
                pools[ttype].append(k)
    if any(not keys for _, keys in reach):
        raise SnapshotRejected("a batch task has no live candidate")
    full_product = math.prod(len(keys) for _, keys in reach)
    r_max = max(len(p) for p in pools.values())
    # Every feasible draw across every R is scored; the most BALANCED slate wins (largest
    # minimum per-task count, then most tasks with >= 2, then most plans). Taking the
    # first feasible draw at the largest R gave slates like [1, 6, 1, 1, 6, 6, 3, 6, 1, 4]
    # -- half the tasks forced -- because reachability from the sources is uneven.
    best: Optional[Tuple[Tuple[int, int, int], int, Dict[str, Set[str]], List[int]]] = None
    cap_rejected = 0
    for r in range(r_max, 0, -1):
        for _ in range(attempts):
            subset = {t: set(rng.sample(pool, min(r, len(pool)))) for t, pool in pools.items()}
            if force_keys:
                # Keys the reactive rule uses are added, never substituted, so the draw's
                # balance is preserved and the product can only grow -- the target_combos
                # test below still gates it.
                for ttype, forced in force_keys.items():
                    subset.setdefault(ttype, set()).update(k for k in forced if k in pools.get(ttype, []))
            counts = [len(subset[t] & keys) for t, keys in reach]
            if min(counts) < 1:
                continue
            product = math.prod(counts)
            if product > target_combos:
                continue
            key = (min(counts), sum(1 for c in counts if c >= 2), product)
            if best is not None and key <= best[0]:
                continue
            if demands is not None and not cap_feasible(demands, subset):
                cap_rejected += 1
                continue
            best = (key, r, subset, counts)
    # `min_choice_fraction` is how many of the batch's tasks must keep >= 2 candidates. The
    # default 0.5 is the training-corpus rule: a dataset where most tasks are forced teaches
    # nothing. A DIAGNOSTIC read over served states wants the opposite (drainable_debug_v1 D1,
    # 2026-09-14) -- at a drainable load the reactive policy drives the cluster into states
    # whose whole-peer-group plan space has a median size of 8, and dropping those would
    # measure a cluster that policy never actually produces.
    if best is None or best[0][1] < min_choice_fraction * len(reach):
        full_ok = None if demands is None else cap_feasible(demands, None)
        raise SnapshotRejected(
            f"no candidate subset fits target_combos={target_combos} with at least "
            f"{min_choice_fraction:.0%} of tasks keeping >= 2 candidates (live product {full_product}; "
            f"{cap_rejected} draws failed the alpha={CAP_ALPHA_TIGHTEST} cap, full live slate "
            f"feasible={full_ok})"
        )
    _key, r, subset, counts = best
    record = {
        "R": r,
        "per_task_candidates": counts,
        "num_combos": math.prod(counts),
        "full_live_product": full_product,
        "pool_sizes": {t: len(p) for t, p in pools.items()},
        "subset": {t: sorted(s) for t, s in subset.items()},
        "cap_feasible_alpha": None if demands is None else CAP_ALPHA_TIGHTEST,
        "cap_rejected_draws": cap_rejected,
        "forced_keys": {t: sorted(k) for t, k in (force_keys or {}).items()} or None,
        "min_choice_fraction": min_choice_fraction,
    }
    return subset, record


def flag_candidates(snapshot: Dict[str, Any], subset: Dict[str, Set[str]]) -> Dict[str, Any]:
    """A copy of the snapshot whose replicas_by_type specs carry `candidate`, and whose
    task candidate lists are restricted to the chosen subset (so build_live_snapshot_seed
    and anything reading `tasks[].candidates` agree on the slate)."""
    out = deepcopy(snapshot)
    for ttype, specs in out.get("replicas_by_type", {}).items():
        chosen = subset.get(ttype, set())
        for spec in specs:
            spec["candidate"] = _qkey(spec) in chosen
    for task in out["tasks"]:
        chosen = subset.get(str(task["task_type"]), set())
        task["candidates"] = [c for c in task.get("candidates", []) if c["queue_key"] in chosen]
        task["candidate_count"] = len(task["candidates"])
    return out


# ---------------------------------------------------------------------------------------
# infrastructure
# ---------------------------------------------------------------------------------------

def cell_base_infrastructure(cell_config: Path, sim_input: Path, seed: int, scratch: Path) -> Dict[str, Any]:
    """The cell's topology (network maps, link fabric) exactly as the LIVE run built it.

    Not `generate_deterministic_infrastructure`: the corpus generator repairs client->server
    reachability for ITS OWN replica plan after drawing the topology (the "live topology is
    a subgraph of the corpus topology" gap), and on the prototype that handed two batch
    tasks a candidate their live source could not reach -- 1536 sweep plans against the
    384 the snapshot's candidate lists span. The snapshots were captured on
    `prepare_infrastructure_for_real_simulation`'s graph, so that is the graph the replay
    runs on; replica/queue tables are the snapshot's (live_snapshot_seed)."""
    from src.executesimulation import prepare_infrastructure_for_real_simulation

    out = scratch / f"cell_s{seed}_live_infrastructure.json"
    if not out.exists():
        space = json.loads(Path(cell_config).read_text())
        live = prepare_infrastructure_for_real_simulation(space, seed=seed, sim_input_path=sim_input)
        base = {
            "network_maps": {n["node_name"]: dict(n.get("network_map") or {}) for n in live["nodes"]},
            "link_topology": live.get("link_topology"),
            "compute_slots_per_node": live.get("compute_slots_per_node"),
            "ingress_bandwidth_mbps": live.get("ingress_bandwidth_mbps"),
            "metadata": {
                "seed": seed, "config_file": str(cell_config), "topology_source": "live",
                "generation_time": time.strftime("%Y-%m-%dT%H:%M:%S"), "warmth_physics": None,
            },
        }
        # Atomic: two array tasks (one per behaviour source) generate into the same corpus
        # dir for the same cell and both reach here; a reader must never see a half file.
        tmp = out.with_name(f"{out.name}.tmp.{os.getpid()}")
        tmp.write_text(json.dumps(base))
        os.replace(tmp, out)
    with open(out) as fh:
        return json.load(fh)


def build_infrastructure(
    base: Dict[str, Any], flagged_snapshot: Dict[str, Any], provenance: Dict[str, Any],
    synthetic: Optional[Tuple[random.Random, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    seed_block = build_live_snapshot_seed(flagged_snapshot)
    if synthetic is not None:
        rng, params = synthetic
        provenance["synthetic_backlog"] = inject_synthetic_backlog(seed_block, rng, **params)
    # The seed carries the WHOLE live cluster (candidates and occupied non-candidates) and
    # is what the simulator replays. The two classic tables list the CANDIDATE replicas
    # only -- the slate the sweep enumerates -- because that is what the scoring tools
    # read them for (score_route_b_contention.load_platform_map -> alpha pre-scan, cache
    # caps). The simulator never uses them here: start_simulation takes the
    # live_snapshot_seed branch before it looks at deterministic placements.
    type_by_key: Dict[str, str] = {}
    for task in flagged_snapshot.get("tasks", []):
        for c in task.get("candidates", []):
            type_by_key[str(c["queue_key"])] = str(c.get("platform_type", ""))
    replica_placements: Dict[str, List[Dict[str, Any]]] = {}
    queue_distributions: Dict[str, Dict[str, int]] = {}
    for task_type, specs in seed_block["replicas_by_type"].items():
        for spec in specs:
            if not spec.get("candidate", True):
                continue
            key = _qkey(spec)
            platform_type = str(spec.get("platform_type") or type_by_key.get(key) or "")
            if not platform_type:
                raise SnapshotRejected(f"candidate {key} has no platform_type in the snapshot")
            replica_placements.setdefault(task_type, []).append(
                {"node_name": spec["node_name"], "platform_id": int(spec["platform_id"]),
                 "platform_type": platform_type}
            )
            queue_distributions.setdefault(task_type, {})[key] = int(spec.get("queue_length", 0) or 0) + int(
                spec.get("synthetic_queue_length", 0) or 0
            )
    if not replica_placements:
        raise SnapshotRejected("seed carries no candidate replica")
    infra = {
        "network_maps": base["network_maps"],
        "replica_placements": replica_placements,
        "queue_distributions": queue_distributions,
        "compute_slots_per_node": base.get("compute_slots_per_node"),
        "ingress_bandwidth_mbps": base.get("ingress_bandwidth_mbps"),
        "link_topology": base.get("link_topology"),
        "live_snapshot_seed": seed_block,
        "metadata": {**base.get("metadata", {}), "warm_snapshot": provenance},
    }
    return infra


# ---------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------

def load_snapshots(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in paths:
        with open(p) as fh:
            for line in fh:
                if line.strip():
                    s = json.loads(line)
                    s["_source_file"] = str(p)
                    out.append(s)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--snapshots", type=Path, nargs="+", required=True, help="live-audit JSONL file(s)")
    ap.add_argument("--trace", type=Path, required=True, help="the trace the snapshots were captured on (peer_exchange + demand_scale)")
    ap.add_argument("--cell-config", type=Path, required=True, help="space_with_network.json of the capture cell (carries the topology seed)")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--sim-input", type=Path, default=PROJECT_ROOT / "data" / "nofs-ids")
    ap.add_argument("--source-tag", required=True, help="behaviour policy + cell, recorded per dataset (e.g. knb_c9001)")
    ap.add_argument("--group-size", type=int, default=10)
    ap.add_argument("--start-index", type=int, default=0, help="first dataset number (ds_XXXXX)")
    ap.add_argument("--limit", type=int, default=None, help="stop after this many datasets")
    ap.add_argument("--min-time", type=float, default=0.0, help="skip snapshots captured before this sim time")
    ap.add_argument("--target-combos", type=int, default=20000)
    ap.add_argument("--max-combos", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=9001, help="base seed of the candidate draw")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="write configs + candidate records, run no sweep")
    ap.add_argument(
        "--no-cap-filter", action="store_true",
        help="joint_burst_v2 (2026-09-20): do NOT reject a snapshot whose live slate admits no plan "
             "under the alpha=2.0 node caps. 579 of the 634 burst snapshots joint_burst_v1 rejected "
             "were rejected for exactly this -- the loaded states the served model meets and was "
             "never trained on. A dataset built this way carries a label only at the alpha rungs "
             "that have a feasible row (prepare_graphs_cache records an empty label set for the "
             "others), so train it at NEAR_RTT_DAG_ALPHA_KEY=inf.")
    ap.add_argument(
        "--min-choice-fraction", type=float, default=0.5,
        help="fraction of a batch's tasks that must keep >= 2 candidates for the draw to be "
             "accepted (default 0.5, the training-corpus rule). Pass 0 for a diagnostic read "
             "over served states, where forced tasks are part of what is being measured",
    )
    ap.add_argument(
        "--force-candidates-from-plans", action="store_true",
        help="keep the replicas the live shortest-queue rule would use in the candidate subset "
             "(drainable_debug_v1 D1), so that plan is present in the enumerated sweep and can "
             "be scored on every dataset rather than only where the draw happened to include it",
    )
    ap.add_argument(
        "--synthetic-backlog-rungs", type=float, nargs="+", default=None,
        help="backlog_corpus_v1: per dataset, draw one mean backlog (seconds per busy platform) from "
             "this list and inject fake queued work on the replayed cluster (0 = none). Off when unset.")
    ap.add_argument("--synthetic-backlog-busy-prob", type=float, default=0.5)
    ap.add_argument("--synthetic-backlog-task-seconds", type=float, default=4.5,
                    help="mean drain of one fake queued task (live execution + peer exchange scale)")
    ap.add_argument("--synthetic-backlog-seed", type=int, default=7001)
    args = ap.parse_args()
    if args.synthetic_backlog_rungs is not None and (
        not 0.0 < args.synthetic_backlog_busy_prob <= 1.0 or args.synthetic_backlog_task_seconds <= 0.0
        or any(r < 0.0 for r in args.synthetic_backlog_rungs)
    ):
        raise SystemExit("FAIL LOUD: synthetic backlog needs busy-prob in (0, 1], task-seconds > 0, rungs >= 0")

    for k, v in REQUIRED_ENV.items():
        if os.environ.get(k) != v:
            raise SystemExit(f"FAIL LOUD: {k}={v} is the physics the snapshots were taken under; set it")

    cell = json.loads(args.cell_config.read_text())
    cell_seed = int(cell["network"]["topology"]["seed"])
    trace = json.loads(args.trace.read_text())
    task_types_db = json.loads((args.sim_input / "task-types.json").read_text())
    snapshots = load_snapshots(args.snapshots)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scratch = args.output_dir / ".warm_scratch"
    scratch.mkdir(exist_ok=True)
    base_infra = cell_base_infrastructure(args.cell_config, args.sim_input, cell_seed, scratch)

    sim_data = PROJECT_ROOT / "simulation_data"
    sample_json_file = sim_data / "sample_simple.json"
    samples_file = sim_data / "lhs_samples_simple.npy"
    mapping_file = sim_data / "lhs_samples_simple_mapping.pkl"

    manifest_path = args.output_dir / "warm_manifest.jsonl"
    made = 0
    idx = args.start_index
    t0 = time.time()
    for snap in snapshots:
        if args.limit is not None and made >= args.limit:
            break
        sid = int(snap.get("snapshot_id", -1))
        entry: Dict[str, Any] = {
            "snapshot_id": sid, "source_file": snap.get("_source_file"), "source_tag": args.source_tag,
            "time": float(snap.get("time", 0.0)), "policy": snap.get("policy"),
        }
        try:
            if float(snap.get("time", 0.0)) < args.min_time:
                raise SnapshotRejected(f"time {snap.get('time')} < --min-time {args.min_time}")
            ids = check_aligned_peer_group(snap, args.group_size)
            workload = build_batch_workload(snap, trace, ids, list(cell["wsc"].keys()))
            rng = random.Random(args.seed * 1_000_003 + sid)
            demands = batch_demands(snap, ids, trace, task_types_db)
            force_keys = reactive_plan_keys(snap) if args.force_candidates_from_plans else None
            subset, record = choose_candidates(
                snap, rng, args.target_combos, args.max_combos,
                demands=None if args.no_cap_filter else demands,
                force_keys=force_keys, min_choice_fraction=args.min_choice_fraction,
            )
            flagged = flag_candidates(snap, subset)
            provenance = {
                "source_tag": args.source_tag, "snapshot_id": sid, "snapshot_time": float(snap.get("time", 0.0)),
                "policy": snap.get("policy"), "trigger_task_id": snap.get("trigger_task_id"),
                "task_ids": ids, "trace": str(args.trace), "cell_config": str(args.cell_config),
                "cell_seed": cell_seed, "candidate_draw": record, "candidate_seed": args.seed,
            }
            synthetic = None
            if args.synthetic_backlog_rungs is not None:
                brng = random.Random(args.synthetic_backlog_seed * 1_000_003 + sid)
                synthetic = (brng, {
                    "mean_seconds": brng.choice(args.synthetic_backlog_rungs),
                    "busy_prob": args.synthetic_backlog_busy_prob,
                    "task_seconds": args.synthetic_backlog_task_seconds,
                })
                provenance["synthetic_backlog_seed"] = args.synthetic_backlog_seed
            infra = build_infrastructure(base_infra, flagged, provenance, synthetic)
        except SnapshotRejected as exc:
            entry["status"] = "rejected"
            entry["reason"] = str(exc)
            with open(manifest_path, "a") as fh:
                fh.write(json.dumps(entry) + "\n")
            if not args.quiet:
                print(f"[warm] snapshot {sid}: rejected -- {exc}", flush=True)
            continue

        dataset_id = f"ds_{idx:05d}"
        out_dir = args.output_dir / dataset_id
        if (out_dir / "placements" / "placements.jsonl").is_file() and (out_dir / "best.json").is_file():
            if not args.quiet:
                print(f"[warm] {dataset_id} exists, skipping", flush=True)
            idx += 1
            made += 1
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        infra_path = scratch / f"{dataset_id}_infrastructure.json"
        infra_path.write_text(json_dumps_pretty(infra))
        workload_path = scratch / f"{dataset_id}_workload.json"
        workload_path.write_text(json_dumps_pretty(workload))
        config = deepcopy(cell)
        config.setdefault("scheduler", {})
        config["scheduler"]["batch_size"] = args.group_size
        config["scheduler"].setdefault("batch_timeout", 0.1)
        config["warm_snapshot"] = provenance
        with open(out_dir / WARM_SNAPSHOT_FILE, "w") as fh:
            json.dump({"snapshot": flagged, "provenance": provenance}, fh)

        entry.update({"dataset_id": dataset_id, "candidate_draw": record})
        if args.dry_run:
            (out_dir / "space_with_network.json").write_text(json_dumps_pretty(config))
            shutil.copy2(infra_path, out_dir / "infrastructure.json")
            shutil.copy2(workload_path, out_dir / "workload.json")
            entry["status"] = "dry-run"
        else:
            status, rtt, secs = generate_single_dataset(
                dataset_id, out_dir, config, workload_path, args.sim_input,
                sample_json_file, samples_file, mapping_file,
                seed=cell_seed, max_workers=args.workers, quiet=args.quiet,
                fast_forward_warmup=True, fast_forward_threshold=1,
                allow_non_unique_replicas=True, warmth_physics="node_disk_v2",
                grid_name=f"warm_snapshot:{args.source_tag}", num_tasks=args.group_size,
                infrastructure_override=infra_path,
            )
            entry.update({"status": status, "rtt": rtt, "seconds": secs})
            if not args.quiet:
                print(f"[warm] {dataset_id} <- snapshot {sid} (t={snap.get('time'):.1f}s, "
                      f"{record['num_combos']} combos): {status} rtt={rtt:.1f} in {secs:.0f}s", flush=True)
        with open(manifest_path, "a") as fh:
            fh.write(json.dumps(entry) + "\n")
        if entry["status"] not in ("success", "dry-run"):
            # Fail loud (CLAUDE.md rule 4). generate_single_dataset already refines the
            # engine's 'success' into 'truncated' when placement_metadata.json says the
            # sweep lost rows (2026-09-13: 81 of 95 W0 datasets came back with 3-30 % of
            # their plans after the worker pool was OOM-killed, and the manifest said
            # done). A truncated sweep's best.json is not a label; stop here so the
            # array task fails instead of the read discovering it.
            raise RuntimeError(
                f"{dataset_id} (snapshot {sid}): generation status {entry['status']!r} "
                f"-- see {out_dir / 'placement_metadata.json'}"
            )
        idx += 1
        made += 1
    print(f"[warm] done: {made} dataset(s) in {time.time() - t0:.0f}s -> {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
