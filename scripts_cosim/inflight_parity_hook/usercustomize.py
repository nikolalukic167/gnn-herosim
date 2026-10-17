"""partial_state_v6 live-parity hook (installed as usercustomize.py in a private PYTHONUSERBASE by inflight_parity_live.sbatch; active only when INFLIGHT_PARITY_OUT is set).

At every INFLIGHT_PARITY_EVERY-th batch decision of the peer-greedy batch scheduler (the base of cd / cd_ib0 / batched / heft) it takes the fidelity capture of the live state,
the function that wrote the corpus states' `fidelity.ghosts`, and compares, platform by platform:
  (a) the cache-side numbers (`state_from_ghosts` over the capture's ghost records and the platform's queue length) with the live ones (`live_state` over the platform objects);
  (b) the live builder's own graph columns (`build_pyg_inference_graph` under partial_state_v6, fed by `StateCaptureHelper`'s temporal state) with `columns(...)` over the
      cache-side numbers and the graph's queue divisor, after the float32 cast the graph applies.
Counts must be equal, work seconds within 1e-9, the builder's columns exactly equal. Everything is tallied in INFLIGHT_PARITY_OUT (json) at exit; any mismatch is listed.
The hook changes no decision: it only reads, then calls the scheduler's own method."""
import atexit
import json
import os

_OUT = os.environ.get("INFLIGHT_PARITY_OUT")
if _OUT:
    os.environ.setdefault("HEROSIM_SNAPSHOT_FIDELITY", "1")
    _EVERY = int(os.environ.get("INFLIGHT_PARITY_EVERY", "300"))
    _TALLY = {"decisions": 0, "sampled": 0, "platforms_compared": 0, "platforms_with_inflight": 0, "max_abs_work_diff": 0.0, "count_mismatch": 0,
              "work_mismatch": 0, "builder_mismatch": 0, "builder_graphs": 0, "max_count": 0, "max_work": 0.0, "mismatches": []}

    def _dump():
        # one file per process (the gate's parent process loads this hook too); only a process that saw decisions, or failed to install, leaves a tally
        if _TALLY["decisions"] or _TALLY.get("install_error"):
            json.dump(_TALLY, open(f"{_OUT}.{os.getpid()}", "w"), indent=1)

    atexit.register(_dump)

    def _install():
        import numpy as np
        import torch

        from src.placement import inflight_features as IF
        from src.placement import snapshot_fidelity
        from src.policy.peer_greedy_network import scheduler as S
        from src.policy.tabular.feature_builder import build_pyg_inference_graph

        repo = os.environ["INFLIGHT_PARITY_REPO"]
        task_types = json.load(open(os.path.join(repo, "data/nofs-ids/task-types.json")))
        orig = S.PeerGreedyNetworkBatchScheduler._pg_decide

        def compare(sched, batch_tasks, system_state):
            fid = snapshot_fidelity.capture(sched, system_state, batch_tasks)
            by_q = {}
            for g in fid["ghosts"]:
                by_q.setdefault(str(g["q"]), []).append(g)
            temporal = sched._capture_temporal_state_snapshot()   # the capture the GNN scheduler feeds the builder with (state_capture's helper)
            cache_side = {}
            for node in sched.nodes.items:
                for platform in node.platforms.items:
                    key = f"{node.node_name}:{platform.id}"
                    live = IF.live_state(platform)
                    cap = IF.state_from_ghosts(by_q.get(key, ()), platform.queue_length(), task_types, platform.type["shortName"])
                    cache_side[key] = cap
                    _TALLY["platforms_compared"] += 1
                    if platform.inflight or platform.admitted is not None:
                        _TALLY["platforms_with_inflight"] += 1
                    _TALLY["max_count"] = max(_TALLY["max_count"], live[0])
                    _TALLY["max_work"] = max(_TALLY["max_work"], live[1])
                    d = abs(live[1] - cap[1])
                    _TALLY["max_abs_work_diff"] = max(_TALLY["max_abs_work_diff"], d)
                    if live[0] != cap[0]:
                        _TALLY["count_mismatch"] += 1
                        _TALLY["mismatches"].append({"t": float(sched.env.now), "q": key, "live": list(live), "cache": list(cap)})
                    elif d > 1e-9:
                        _TALLY["work_mismatch"] += 1
                        _TALLY["mismatches"].append({"t": float(sched.env.now), "q": key, "live": list(live), "cache": list(cap)})
            # (b) the live builder's columns
            qsnap = {f"{n.node_name}:{p.id}": p.queue_length() for n in sched.nodes.items for p in n.platforms.items}
            graph, _ = build_pyg_inference_graph(batch_tasks, system_state, qsnap, nodes=list(sched.nodes.items), task_types_data=task_types,
                                                 queue_norm_mode=os.environ.get("GNN_QUEUE_NORM_MODE", "scheduler_adaptive"), temporal_state=temporal)
            if graph is not None:
                _TALLY["builder_graphs"] += 1
                meta = graph.queue_key_to_platform_meta
                for key, m in meta.items():
                    want = IF.columns(cache_side[key][0], cache_side[key][1], float(graph.queue_norm_divisor))
                    got = graph.platform_features[int(m["platform_pos"]), -IF.INFLIGHT_DIM:]
                    if not torch.equal(got, torch.as_tensor(want, dtype=torch.float32)):
                        _TALLY["builder_mismatch"] += 1
                        _TALLY["mismatches"].append({"t": float(sched.env.now), "q": key, "builder": got.tolist(), "cache_columns": want.tolist()})

        def wrapped(self, batch_tasks, system_state):
            _TALLY["decisions"] += 1
            if _TALLY["decisions"] % _EVERY == 0 or _TALLY["decisions"] <= 5:
                _TALLY["sampled"] += 1
                compare(self, batch_tasks, system_state)
            return orig(self, batch_tasks, system_state)

        S.PeerGreedyNetworkBatchScheduler._pg_decide = wrapped

    try:
        _install()
    except Exception as e:  # the hook must fail loud: an interpreter that cannot install it produces no tally
        _TALLY["install_error"] = repr(e)
        _dump()
        raise
