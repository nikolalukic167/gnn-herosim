#!/usr/bin/env python3
"""peer_affinity_v1 stage 3 — is the live serving path the offline read, bit for bit?

The T1/T1b reads scored every checkpoint offline: cache graph -> prefix-conditioned masked
decode -> sweep lookup. Serving the same checkpoint LIVE goes through a different builder
(`feature_builder.build_pyg_inference_graph` + `prefix_serving.attach_live_prefix_block`),
a different batch path (`GNNScheduler._process_task_batch_prefix`) and the real engine.
This script runs that live path on the held-out co-sim datasets — the one setting where
the right answer is known to the last digit — and asserts, per dataset:

  1. the live scheduler formed ONE batch holding exactly the dataset's tasks, in order;
  2. the graph it served equals the cache graph attribute by attribute (features, edges,
     one-hot, peer edges, every partial_state_ctx ingredient);
  3. the plan it decoded equals the offline report's `decoded_combo`;
  4. the engine's total_rtt equals the sweep row for that plan (and, when given, the
     replay gate's number for the same arm/seed).

Any mismatch is listed, never summarised away; the exit code is 1 if one exists.

Usage:
  PIPENV_IGNORE_VIRTUALENVS=1 VIRTUAL_ENV= PYTHONPATH=. pipenv run python3 \
    scripts_cosim/peer_affinity_live_serve_check.py \
      --checkpoint models/peer-affinity-v1-t1b-gnn-lr2e3-seed1.pt \
      --report simulation_data/peer_affinity_t1b_reports/peer-affinity-v1-t1b-gnn-lr2e3-seed1.json \
      --live-replay simulation_data/peer_affinity_t1b_live_replay.json --arm-key gnn_s1 \
      --output simulation_data/peer_affinity_live_serve_check_gnn_s1.json
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import multiprocessing as mp
import os
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PHYSICS_ENV = {
    "HEROSIM_PEER_EXCHANGE": "1",
    "HEROSIM_COSIM_KEEP_ALIVE": "1000000",
    "COSIM_SUPPRESS_SIM_PRINTS": "1",
    "SIM_FORCE_FULL_STATS": "1",
    "GNN_DECODE_MODE": "masked_topo",
    "HEROSIM_GNN_DEVICE": "cpu",
    "PYTHONHASHSEED": "0",
}
TENSOR_FIELDS = (
    "task_features", "platform_features", "edge_index", "edge_attr", "node_edge_index",
    "task_type_onehot4", "peer_edge_index", "peer_edge_attr",
)
PSC_FIELDS = (
    "node_caps", "demand", "task_type_index", "parents", "route_hops_bneck", "payload_bytes",
    "transfer_norm", "node_rank", "ingress_links", "core_links", "peer_pairs", "node_exchange",
    "peer_norm", "cand_nodes",
)


def _apply_env() -> None:
    os.environ.update(PHYSICS_ENV)
    for name in ("GNN_BATCH_SIZE", "HEROSIM_DATA_LOCALITY", "GNN_BATCH_BY_PEER_GROUP"):
        os.environ.pop(name, None)


def _norm(value: Any) -> Any:
    """JSON-ish normalisation so cache and live containers compare structurally."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(_norm(k)): _norm(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_norm(v) for v in value]
        return sorted(items, key=repr) if isinstance(value, (set, frozenset)) else items
    if isinstance(value, float):
        return value
    return value


def _close(a: Any, b: Any, tol: float = 1e-9) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            fa, fb = float(a), float(b)
        except (TypeError, ValueError):
            return False
        if math.isinf(fa) or math.isinf(fb):
            return fa == fb
        return abs(fa - fb) <= tol * max(1.0, abs(fa), abs(fb))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k], tol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y, tol) for x, y in zip(a, b))
    return a == b


def _canon(graph: Any, n_tasks: int) -> Dict[str, Any]:
    """Order-free view of a graph: rows keyed by (task, node_id, platform_id)."""
    meta = graph.queue_key_to_platform_meta
    by_pos = {int(m["platform_pos"]): (int(m["node_id"]), int(m["platform_id"])) for m in meta.values()}
    out: Dict[str, Any] = {}
    out["task_features"] = _norm(graph.task_features)
    out["task_type_onehot4"] = _norm(getattr(graph, "task_type_onehot4", None))
    out["peer_edge_index"] = _norm(getattr(graph, "peer_edge_index", None))
    out["peer_edge_attr"] = _norm(getattr(graph, "peer_edge_attr", None))
    pf = graph.platform_features.detach().cpu().tolist()
    out["platform_features"] = {str(by_pos[i]): row for i, row in enumerate(pf)}
    ei = graph.edge_index.detach().cpu().tolist()
    ea = graph.edge_attr.detach().cpu().tolist() if getattr(graph, "edge_attr", None) is not None else None
    edges: Dict[str, Any] = {}
    for k, (src, dst) in enumerate(zip(ei[0], ei[1])):
        if src < n_tasks <= dst:
            key = str((int(src),) + by_pos[int(dst) - n_tasks])
            edges[key] = ea[k] if ea is not None else True
    out["edges"] = edges
    nei = getattr(graph, "node_edge_index", None)
    if nei is not None:
        pairs = set()
        for src, dst in zip(*nei.detach().cpu().tolist()):
            pairs.add(str(tuple(sorted((by_pos[int(src) - n_tasks], by_pos[int(dst) - n_tasks])))))
        out["node_edges"] = sorted(pairs)
    tl = graph.task_logit_to_placement
    out["candidates"] = {str(t): sorted(str(tuple(int(v) for v in c)) for c in tl[t]) for t in range(n_tasks)}
    return out


def _sweep_lookup(ds_dir: Path) -> Dict[Tuple[Tuple[int, int], ...], float]:
    out: Dict[Tuple[Tuple[int, int], ...], float] = {}
    with open(ds_dir / "placements/placements.jsonl") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            plan = {int(k): (int(v[0]), int(v[1])) for k, v in row["placement_plan"].items()}
            out[tuple(plan[i] for i in sorted(plan))] = float(row["rtt"])
    return out


def _one_dataset(job: dict) -> dict:
    ds_id: str = job["dataset_id"]
    _apply_env()
    import torch

    from src.executecosimulation import QUEUE_LENGTH, cosim_keep_alive, execute_simulation
    from src.executesimulation import load_gnn_model

    ds_dir = REPO_ROOT / "simulation_data" / ds_id
    o = json.load(open(ds_dir / "optimal_result.json"))
    n_tasks = len(o["config"]["workload"]["events"])
    mismatches: List[str] = []
    result: Dict[str, Any] = {"dataset_id": ds_id, "n_tasks": n_tasks}

    trace = tempfile.NamedTemporaryFile(prefix="prefix_trace_", suffix=".pkl", delete=False)
    trace.close()
    os.environ["GNN_PREFIX_TRACE_PATH"] = trace.name
    try:
        model, device = load_gnn_model(Path(job["checkpoint"]), space_config=None)
        cfg = copy.deepcopy(o["config"])
        cfg["infrastructure"].pop("forced_placements", None)
        models = {"gnn_model": model, "device": device, "task_types_data": o["sim_inputs"]["task_types"]}
        stats = execute_simulation(
            cfg, o["sim_inputs"], "gnn_gnn", models=models, cache_policy="fifo",
            task_priority="fifo", keep_alive=cosim_keep_alive(), queue_length=QUEUE_LENGTH,
        )["stats"]
        records = []
        with open(trace.name, "rb") as fh:
            while True:
                try:
                    records.append(pickle.load(fh))
                except EOFError:
                    break
    finally:
        os.unlink(trace.name)

    # 1. one batch, the dataset's tasks, in order
    result["n_batches"] = len(records)
    if len(records) != 1:
        mismatches.append(f"batches: live formed {len(records)} batches, expected 1")
    if records and records[0]["task_ids"] != list(range(n_tasks)):
        mismatches.append(f"batch order: {records[0]['task_ids']} != {list(range(n_tasks))}")
    if int(stats.get("num_tasks") or -1) != n_tasks:
        mismatches.append(f"num_tasks: {stats.get('num_tasks')} != {n_tasks}")

    if records:
        live = records[0]["graph"]
        cache = job["cache_graph"]
        # 2. graph parity, keyed by ids. Platform POSITION is history-dependent on both
        # sides (the cache takes nodeResults order from the optimal replay, the live
        # builder takes FilterStore order at batch time), and nothing the model computes
        # depends on it (GIN aggregation is permutation-equivariant, edges are scored
        # per pair, the decoder ties break by ids), so the comparison is by
        # (task, node_id, platform_id), not by row.
        lc, cc = _canon(live, n_tasks), _canon(cache, n_tasks)
        for name in sorted(set(lc) | set(cc)):
            lv, cv = lc.get(name), cc.get(name)
            if lv is None or cv is None:
                mismatches.append(f"{name}: live {'absent' if lv is None else 'present'}, cache {'absent' if cv is None else 'present'}")
            elif not _close(lv, cv, tol=1e-6):
                mismatches.append(f"{name}: differ — live {str(lv)[:240]} … cache {str(cv)[:240]}")
        lpsc, cpsc = live.partial_state_ctx, cache.partial_state_ctx
        for name in PSC_FIELDS:
            lv, cv = _norm(lpsc.get(name)), _norm(cpsc.get(name))
            if name == "cand_nodes":
                lv = {k: sorted(v) for k, v in lv.items()}
                cv = {k: sorted(v) for k, v in cv.items()}
            if not _close(lv, cv):
                mismatches.append(f"partial_state_ctx.{name}: live {str(lv)[:200]} != cache {str(cv)[:200]}")
        result["peers_outside_batch"] = records[0]["diag"].get("peers_outside_batch")
        # 3. decoded plan vs the offline report
        combo = [tuple(int(v) for v in c) for c in records[0]["combo"]]
        result["live_combo"] = [list(c) for c in combo]
        if job.get("report_combo") is not None:
            rep = [tuple(int(v) for v in c) for c in job["report_combo"]]
            result["report_combo"] = [list(c) for c in rep]
            if combo != rep:
                mismatches.append(f"decoded_combo: live {combo} != report {rep}")
        # 4. engine rtt vs the sweep row (and the replay gate)
        sweep = _sweep_lookup(ds_dir)
        live_rtt = float(stats["total_rtt"])
        result["live_total_rtt"] = live_rtt
        result["peer_exchange_time"] = float(stats.get("totalPeerExchangeTime") or 0.0)
        result["peer_rendezvous_wait"] = float(stats.get("totalPeerRendezvousWait") or 0.0)
        row = sweep.get(tuple(combo))
        result["sweep_rtt"] = row
        if row is None:
            mismatches.append("sweep: live combo is not a sweep row")
        elif abs(live_rtt - row) > 1e-9 * max(1.0, abs(row)):
            mismatches.append(f"total_rtt: live {live_rtt!r} != sweep {row!r} (Δ={live_rtt - row:+.6e})")
        if job.get("replay_rtt") is not None:
            result["replay_rtt"] = job["replay_rtt"]
            if abs(live_rtt - job["replay_rtt"]) > 1e-9 * max(1.0, abs(job["replay_rtt"])):
                mismatches.append(f"total_rtt: live {live_rtt!r} != replay gate {job['replay_rtt']!r}")
    result["mismatches"] = mismatches
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "simulation_data/graphs_cache_peer_affinity_v1_t1b")
    ap.add_argument("--split-artifact", type=Path, default=REPO_ROOT / "experiments/peer_affinity_v1_t1b_split.json")
    ap.add_argument("--split", default="test")
    ap.add_argument("--report", type=Path, default=None, help="eval_route_b_stage2_arm.py report for this checkpoint")
    ap.add_argument("--live-replay", type=Path, default=None, help="peer_affinity_live_replay_gate.py output")
    ap.add_argument("--arm-key", default=None, help="key inside the replay gate's rtt map, e.g. gnn_s1")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    ids = list(json.loads(args.split_artifact.read_text())[args.split])
    if args.limit:
        ids = ids[: args.limit]
    dataset_ids = pickle.load(open(args.cache_dir / "dataset_ids.pkl", "rb"))
    graphs = pickle.load(open(args.cache_dir / "graphs.pkl", "rb"))
    index = {str(d): i for i, d in enumerate(dataset_ids)}
    report_by_ds: Dict[str, Any] = {}
    if args.report:
        for entry in json.loads(args.report.read_text())["per_dataset"]:
            report_by_ds[entry["dataset_id"]] = entry.get("decoded_combo")
    replay_by_ds: Dict[str, float] = {}
    if args.live_replay and args.arm_key:
        for entry in json.loads(args.live_replay.read_text())["per_dataset"]:
            if args.arm_key in entry["rtt"]:
                replay_by_ds[entry["dataset_id"]] = float(entry["rtt"][args.arm_key])

    jobs = []
    for ds in ids:
        if ds not in index:
            raise SystemExit(f"{ds} not in cache {args.cache_dir}")
        jobs.append({
            "dataset_id": ds,
            "checkpoint": str(args.checkpoint),
            "cache_graph": graphs[index[ds]],
            "report_combo": report_by_ds.get(ds),
            "replay_rtt": replay_by_ds.get(ds),
        })
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        results = pool.map(_one_dataset, jobs)

    bad = [r for r in results if r["mismatches"]]
    summary = {
        "checkpoint": str(args.checkpoint),
        "cache_dir": str(args.cache_dir),
        "split": args.split,
        "n_datasets": len(results),
        "n_clean": len(results) - len(bad),
        "n_with_mismatches": len(bad),
        "per_dataset": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(f"[live-serve-check] {len(results) - len(bad)}/{len(results)} datasets bit-identical; report {args.output}")
    for r in bad:
        print(f"  {r['dataset_id']}:")
        for m in r["mismatches"]:
            print(f"    - {m}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
