#!/usr/bin/env python3
"""r1_attribution_v1: train/serve parity for a fidelity-mode corpus (make_warm_corpus --fidelity).

The cache graph of a fidelity dataset is built from the dataset's files; the graph a LIVE scheduler would serve for the same
decision is built from live state. This script produces the second one the way the first is meant to be reproduced:
the snapshot's fidelity replay (queued + batch tasks, queued state restored: ghosts resumed, pulls pending, KPA state) is run
under the real GNN scheduler (`gnn_gnn`, masked_topo, peer-group batching, R1 environment, live_run_params). The queued tasks'
batches are decided from their live placements (the only patch: `GNNScheduler._prefix_inference` returns them for a batch made
only of queued tasks), they go through the scheduler's own enqueue path, and the dataset's batch is then decided by an
UNTRAINED v5 checkpoint through the live builder (`feature_builder`) and `prefix_serving`. The served graph (the scheduler's
GNN_PREFIX_TRACE_PATH record) is compared with the cache graph attribute by attribute, keyed by (task, node_id, platform_id):
task / platform columns (incl. the four-type columns), edges, peer edges, node_exchange, backlog_s / service_s, every
partial_state_ctx ingredient. Any mismatch is listed; exit 1 if there is one.

  wf1_fidelity_parity.py --datasets <gnn_datasets dir> --cache-dir <v5 cache> --checkpoint <untrained v5 .pt> --out report.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _one(job: Dict[str, Any]) -> Dict[str, Any]:
    import contextlib
    import io

    os.environ["GNN_DECODE_MODE"] = "masked_topo"
    os.environ["GNN_BATCH_BY_PEER_GROUP"] = "1"
    os.environ["HEROSIM_GNN_DEVICE"] = "cpu"
    for name in ("GNN_BATCH_SIZE", "HEROSIM_DATA_LOCALITY"):
        os.environ.pop(name, None)
    import peer_affinity_live_serve_check as P
    from src.executesimulation import execute_simulation, load_gnn_model, load_simulation_inputs, prepare_infrastructure_for_real_simulation
    from src.placement import snapshot_fidelity as SF
    from src.placement.live_snapshot_seed import build_live_snapshot_seed
    from src.policy.gnn.scheduler import GNNScheduler

    d = Path(job["dataset_dir"])
    out: Dict[str, Any] = {"dataset": d.name}
    mism: List[str] = []
    infra_ds = json.loads((d / "infrastructure.json").read_text())
    spec = infra_ds["live_snapshot_seed"]["fidelity_replay"]
    snap = deepcopy(spec["snapshot"])
    fid = snap["fidelity"]
    wl_ds = json.loads((d / "workload.json").read_text())
    types = [next(iter(e["application"]["dag"])) for e in wl_ds["events"]]
    n_batch = len(types)
    # the live batch must list its tasks in the dataset's order (events grouped by application), as the cache graph does
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for rec in sorted(fid["batch"], key=lambda r: int(r["gid"])):
        by_type.setdefault(rec["fn"], []).append(rec)
    fid["batch"] = [by_type[t].pop(0) for t in types]
    wl, forced, ids = SF.replay_workload(fid)
    batch_local = [ids[int(r["gid"])] for r in fid["batch"]]
    out.update(queued=len(forced), batch=n_batch, batch_ids=batch_local)

    space = json.loads(Path(spec["cell_config"]).read_text())
    infra = prepare_infrastructure_for_real_simulation(space, seed=None, sim_input_path=Path(spec["sim_input"]))
    infra["live_snapshot_seed"] = build_live_snapshot_seed(snap)
    infra["forced_placements"] = {}
    infra["fast_forward_warmup"] = True
    infra["fast_forward_threshold"] = 1
    infra["scheduler"] = {"batch_size": max(len(wl["events"]), 1), "batch_timeout": 0.02}
    kw = SF.live_run_params()

    orig = GNNScheduler._prefix_inference

    def patched(self, batch_tasks, system_state, queue_snapshot, temporal_state):
        tids = [int(t.id) for t in batch_tasks]
        queued = [t in forced for t in tids]
        if all(queued):
            return {i: (int(forced[t][0]), int(forced[t][1])) for i, t in enumerate(tids)}
        if any(queued):
            raise RuntimeError(f"queued and batch tasks formed one scheduler batch: {tids}")
        return orig(self, batch_tasks, system_state, queue_snapshot, temporal_state)

    GNNScheduler._prefix_inference = patched
    trace = tempfile.NamedTemporaryFile(prefix="fid_trace_", suffix=".pkl", delete=False)
    trace.close()
    os.environ["GNN_PREFIX_TRACE_PATH"] = trace.name
    records: List[Dict[str, Any]] = []
    try:
        model, device = load_gnn_model(Path(job["checkpoint"]), space_config=None)
        sim_inputs = load_simulation_inputs(Path(spec["sim_input"]))
        models = {"gnn_model": model, "device": device, "task_types_data": sim_inputs["task_types"]}
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            stats = execute_simulation({"infrastructure": infra, "workload": wl}, sim_inputs, "gnn_gnn",
                                       cache_policy="fifo", task_priority="fifo", models=models, **kw)["stats"]
        with open(trace.name, "rb") as fh:
            while True:
                try:
                    records.append(pickle.load(fh))
                except EOFError:
                    break
    except Exception as exc:  # recorded by name
        out["mismatches"] = [f"replay failed: {type(exc).__name__}: {str(exc)[:300]}"]
        return out
    finally:
        os.unlink(trace.name)
        GNNScheduler._prefix_inference = orig

    out["scheduler_batches"] = [r["task_ids"] for r in records]
    want = [int(x) for x in batch_local]
    rec = [r for r in records if [int(x) for x in r["task_ids"]] == want]
    if len(rec) != 1:
        mism.append(f"expected one served batch with tasks {want}; served batches {out['scheduler_batches']}")
        out["mismatches"] = mism
        return out
    live = rec[0]["graph"]
    cache = job["cache_graph"]
    lc, cc = P._canon(live, n_batch), P._canon(cache, n_batch)
    for name in sorted(set(lc) | set(cc)):
        lv, cv = lc.get(name), cc.get(name)
        if lv is None or cv is None:
            mism.append(f"{name}: live {'absent' if lv is None else 'present'}, cache {'absent' if cv is None else 'present'}")
        elif not P._close(lv, cv, tol=1e-6):
            mism.append(f"{name}: differ -- live {str(lv)[:260]} ... cache {str(cv)[:260]}")
    lp, cp = live.partial_state_ctx, cache.partial_state_ctx
    for name in P.PSC_FIELDS:
        lv, cv = P._norm(lp.get(name)), P._norm(cp.get(name))
        if name == "cand_nodes":
            lv = {k: sorted(v) for k, v in lv.items()}
            cv = {k: sorted(v) for k, v in cv.items()}
        if not P._close(lv, cv):
            mism.append(f"partial_state_ctx.{name}: live {str(lv)[:260]} != cache {str(cv)[:260]}")
    out["task_width"] = int(live.task_features.size(-1))
    out["platform_width"] = int(live.platform_features.size(-1))
    out["n_task_results"] = len(stats.get("taskResults") or [])
    out["mismatches"] = mism
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if os.environ.get("HEROSIM_SNAPSHOT_FIDELITY") != "1":
        raise SystemExit("FAIL LOUD: export HEROSIM_SNAPSHOT_FIDELITY=1, PARTIAL_STATE_CONTRACT=partial_state_v5 and the R1 environment")
    ids = pickle.load(open(a.cache_dir / "dataset_ids.pkl", "rb"))
    graphs = pickle.load(open(a.cache_dir / "graphs.pkl", "rb"))
    index = {str(i): k for k, i in enumerate(ids)}
    jobs = []
    for d in sorted(a.datasets.glob("ds_*")):
        key = f"{a.datasets.name}/{d.name}"
        if key not in index:
            raise SystemExit(f"{key} is not in the cache {a.cache_dir}")
        jobs.append({"dataset_dir": str(d), "checkpoint": str(a.checkpoint), "cache_graph": graphs[index[key]]})
    with mp.get_context("spawn").Pool(a.workers) as pool:
        results = pool.map(_one, jobs)
    bad = [r for r in results if r["mismatches"]]
    a.out.write_text(json.dumps({"n": len(results), "clean": len(results) - len(bad), "per_dataset": results}, indent=1))
    print(f"[fidelity parity] {len(results) - len(bad)}/{len(results)} datasets identical; report {a.out}")
    for r in results:
        print(f"  {r['dataset']}: queued {r.get('queued')}, batch {r.get('batch')}, widths task {r.get('task_width')} platform "
              f"{r.get('platform_width')}, scheduler batches {len(r.get('scheduler_batches') or [])}, "
              f"{'identical' if not r['mismatches'] else str(len(r['mismatches'])) + ' mismatches'}")
        for m in r["mismatches"]:
            print("     -", m[:420])
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
