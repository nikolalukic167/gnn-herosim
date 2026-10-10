#!/usr/bin/env python3
"""The CD greedy's plan (plain, or with cd_expand's alpha-expansion) on fidelity-replay label tables, against the exact optimum.

Each dataset is replayed exactly as its labels were (fidelity_replay.FidelityReplay: the snapshot's ghosts, pulls and queued tasks), but
with the batch decided by `peer_greedy_network_cd` instead of forced: the snapshot's queued tasks keep their recorded replicas (their
planned nodes are set before the decision, so CD prices them as placed partners) and CD chooses the dataset's batch tasks, each restricted
to the candidate set of the dataset's own sweep (`_pg_allowed`), so the plan is always a sweep row. Self-check: the replayed total_rtt must
equal that row's label to 1e-9 (the decision ran on the label's state). HEROSIM_PG_CD_EXPANSION=1 selects cd_expand. Read-only.

  fidelity_cd_plans.py OUT.jsonl LISTFILE [--expand] [--workers N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import get_context
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def label_of(rows, platforms):
    """The sweep's label for the plan with these platform ids (task-index order); None if the plan is not a sweep row. The CD replay
    path schedules a task up to a few ms apart from the forced label path, so the plan is scored by its own row, on the optimum's scale."""
    key = [int(p) for p in platforms]
    hit = [float(r["rtt"]) for r in rows if [int(r["placement_plan"][str(i)][1]) for i in range(len(key))] == key]
    return min(hit) if hit else None


def _one(job):
    ds, expand, extra = Path(job[0]), job[1], (job[2] if len(job) > 2 else None) or {}
    try:
        os.environ["HEROSIM_SNAPSHOT_FIDELITY"] = "1"
        prov = json.load(open(ds / "generation_provenance.json"))
        for k, v in (prov.get("physics_env") or {}).items():
            if k.startswith("HEROSIM_") or k in ("COSIM_AUTOSCALER_RECONCILE_INTERVAL",):
                os.environ.setdefault(k, str(v))
        os.environ.update(GNN_DECODE_MODE="masked_topo", GNN_BATCH_BY_PEER_GROUP="1", PYTHONHASHSEED="0")
        for k in ("GNN_SERVE_CANDIDATE_SLATE", "HEROSIM_PG_EXCHANGE_SCALE", "HEROSIM_PG_CD_PASSES", "HEROSIM_PG_BATCH_BLIND"):
            os.environ.pop(k, None)
        if expand:
            os.environ["HEROSIM_PG_CD_EXPANSION"] = "1"
        else:
            os.environ.pop("HEROSIM_PG_CD_EXPANSION", None)
        from copy import deepcopy

        from src.executesimulation import execute_simulation
        from src.placement import fidelity_replay
        from src.policy.peer_greedy_network import scheduler as PG

        seed = json.load(open(ds / "infrastructure.json"))["live_snapshot_seed"]
        spec = dict(seed["fidelity_replay"])
        if not Path(spec["sim_input"]).exists():
            spec["sim_input"] = str(REPO / "data" / "nofs-ids")
        types = [next(iter(e["application"]["dag"])) for e in json.load(open(ds / "workload.json"))["events"]]
        offered = {t: [(sp["node_name"], sp["platform_id"]) for sp in specs if sp.get("candidate", True)]
                   for t, specs in seed["replicas_by_type"].items()}
        fr = fidelity_replay.FidelityReplay(spec, types, offered)
        rows = [r for r in (json.loads(l) for l in open(ds / "placements" / "placements.jsonl") if l.strip()) if "placement_plan" in r]
        allowed = {}
        for r in rows:
            for k, v in r["placement_plan"].items():
                allowed.setdefault(int(k), set()).add((int(v[0]), int(v[1])))
        local_of = {fr.batch_local[i]: i for i in range(len(fr.batch_local))}   # replay task id -> dataset task index
        forced = dict(fr.queued_forced)
        seen = {"decisions": 0, "decided": [], "queued_scheduled_at_decision": None, "s_extra": {}, "s_own": None}

        orig = PG.PeerGreedyNetworkBatchScheduler._prefix_inference

        def patched(self, batch_tasks, system_state, queue_snapshot, temporal_state):
            out = {}
            free = []
            node_by_id = {n.id: n for n in self.nodes.items}
            for i, t in enumerate(batch_tasks):
                f = forced.get(int(t.id))
                if f is not None:
                    out[i] = (int(f[0]), int(f[1]))
                    t.planned_node_name = node_by_id[int(f[0])].node_name
                else:
                    free.append(i)
            if free:
                missing = [int(batch_tasks[i].id) for i in free if int(batch_tasks[i].id) not in local_of]
                if missing:
                    raise RuntimeError(f"tasks {missing} are neither queued-forced nor batch tasks")
                orch = self._pg_orchestrator()
                seen["queued_scheduled_at_decision"] = sum(
                    1 for tid in forced if getattr(orch, "task_by_id", {}).get(tid) is not None
                    and orch.task_by_id[tid].scheduled.triggered)
                self._pg_allowed = {int(batch_tasks[i].id): allowed[local_of[int(batch_tasks[i].id)]] for i in free}
                free_tasks = [batch_tasks[i] for i in free]
                couple = {(int(n.id), int(p.id)): (n, p) for n in self.nodes.items for p in n.platforms.items}

                def s_of(plan_by_ds_index):
                    trial = [couple[(int(plan_by_ds_index[local_of[int(t.id)]][0]), int(plan_by_ds_index[local_of[int(t.id)]][1]))]
                             for t in free_tasks]
                    return float(self._pg_plan_cost(free_tasks, trial, orch, {}, self.nodes.items)[0])

                for key, plan in extra.items():
                    seen["s_extra"][key] = s_of(plan)
                try:
                    sub = self._pg_decide(free_tasks, system_state)
                finally:
                    self._pg_allowed = None
                own = {}
                for k, i in enumerate(free):
                    own[local_of[int(batch_tasks[i].id)]] = sub[k]
                seen["s_own"] = s_of(own)
                for k, i in enumerate(free):
                    out[i] = sub[k]
                seen["decisions"] += 1
                seen["decided"] += [int(batch_tasks[i].id) for i in free]
            return out

        PG.PeerGreedyNetworkBatchScheduler._prefix_inference = patched
        try:
            infra = deepcopy(fr.base_infra)
            infra["live_snapshot_seed"] = fr.seed
            infra["fast_forward_warmup"] = True
            infra["fast_forward_threshold"] = 1
            infra["scheduler"] = {"batch_size": max(len(fr.wl["events"]), 1), "batch_timeout": 0.02, "exact_batch": True}
            res = execute_simulation({"infrastructure": infra, "workload": fr.wl}, fr.sim_inputs,
                                     scheduling_strategy="peer_greedy_network_cd_peer_greedy_network_cd", cache_policy="fifo",
                                     task_priority="fifo", **fr.kw)
        finally:
            PG.PeerGreedyNetworkBatchScheduler._prefix_inference = orig
        if sorted(seen["decided"]) != sorted(fr.batch_local):
            raise RuntimeError(f"CD decided {sorted(seen['decided'])}, the batch is {sorted(fr.batch_local)}")
        trs = {tr["taskId"]: tr for tr in res["stats"]["taskResults"] if tr.get("taskId", -1) >= 0}
        label = sum(float(trs[b]["doneTime"]) - float(trs[b]["scheduledTime"]) for b in fr.batch_local)
        names = {b: (str(trs[b]["executionNode"]), int(trs[b]["executionPlatform"])) for b in fr.batch_local}
        off = [q for q, (n, p) in forced.items() if q in trs and int(trs[q]["executionPlatform"]) != int(p)]
        if off:
            raise RuntimeError(f"queued tasks {off[:5]} did not run on their recorded replicas")
        opt = min(float(r["rtt"]) for r in rows)
        match = [r for r in rows if abs(float(r["rtt"]) - label) <= 1e-9 * max(1.0, abs(label))]
        sc = res["stats"].get("schedulerCounters") or {}
        return dict(ds=str(ds), arm="cd_expand" if expand else "cd", rtt=label, opt=opt, regret_rel=(label - opt) / opt if opt > 0 else 0.0,
                    matched_rows=len(match), n_plans=len(rows), n_tasks=len(fr.batch_local), decisions=seen["decisions"],
                    queued=len(forced), queued_scheduled_at_decision=seen["queued_scheduled_at_decision"],
                    plan_nodes=[names[b][0] for b in fr.batch_local], plan_platforms=[names[b][1] for b in fr.batch_local],
                    label_of_plan=label_of(rows, [names[b][1] for b in fr.batch_local]),
                    expand_moves=int(sc.get("pg_expand_moves") or 0), cd_moves=int(sc.get("pg_cd_moves") or 0),
                    s_own=seen["s_own"], s_extra=seen["s_extra"])
    except Exception as e:  # recorded, never swallowed
        return dict(ds=str(ds), arm="cd_expand" if expand else "cd", error=f"{type(e).__name__}: {str(e)[:300]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out"); ap.add_argument("listfile")
    ap.add_argument("--expand", action="store_true"); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--extra-plans", default=None, help="JSON {dataset dir name: {label: [[node_id, platform_id] per task index]}}: "
                    "CD's whole-plan surrogate S of each is recorded at the decision, on the same state (s_extra), with s_own for its own plan")
    a = ap.parse_args()
    dss = [l.strip() for l in open(a.listfile) if l.strip()]
    extra = json.load(open(a.extra_plans)) if a.extra_plans else {}
    with get_context("spawn").Pool(a.workers) as pool, open(a.out, "w") as fh:
        for r in pool.imap(_one, [(d, a.expand, extra.get(Path(d).name)) for d in dss], chunksize=2):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
