"""Step 0 (offline, no training): a learned checkpoint's own decision on each sealed K1(b) snapshot, served as the gate serves it (masked_topo prefix-conditioned decode,
GNN_SLATE_NO_SPLIT=1, the candidate-slate rule the checkpoint's contract records), replayed by the fidelity replay like the hand and ILS plans.
Per task, the rank of the ILS plan's (node, platform) among the GNN's scores, two ways: FREE-RUN (the step's scores under the GNN's own committed prefix) and TEACHER-FORCED
(the step's scores under the ILS plan's earlier tasks as the prefix). Top-k recall is the share of tasks whose ILS placement ranks <= k; whole-plan match is plan equality."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts_cosim import uncut_ils as U  # noqa: E402
from scripts_cosim.uncut_xarm import pair_counts  # noqa: E402

MODELS = Path("/home/nikola.lukic/gnn-herosim/models")
CKPT = {"gnn": MODELS / "slate-fix-v1-gnn-eng-seed1.pt", "twin": MODELS / "slate-fix-v1-twin-eng-seed1.pt"}


def serve_env(kind: str) -> Dict[str, str]:
    ck = CKPT[kind]
    rule = json.load(open(str(ck)[:-3] + ".contract.json")).get("candidate_slate")
    if not rule:
        raise RuntimeError(f"FAIL LOUD: {ck} records no candidate_slate")
    env = {"GNN_MODEL_PATH": str(ck), "GNN_DECODE_MODE": "masked_topo", "GNN_BATCH_BY_PEER_GROUP": "1", "GNN_PREFIX_ALPHA_KEY": "inf", "PARTIAL_STATE_CONTRACT": "partial_state_v5",
           "PARTIAL_STATE_LOAD_SECONDS": "1", "PARTIAL_STATE_EXCHANGE_SECONDS": "1", "GNN_SERVE_CANDIDATE_SLATE": rule, "GNN_SLATE_NO_SPLIT": "1"}
    if kind == "twin":
        env["GNN_DISABLE_MESSAGE_PASSING"] = "1"
    return env


def learned_decision(job: Dict[str, Any], kind: str, ils_plan: Sequence[Sequence[int]], models: Dict[str, Any]) -> Dict[str, Any]:
    """Replay the snapshot under the learned policy; at the whole-batch decision return its plan (snapshot batch order) and the rank diagnostics against `ils_plan`."""
    U._setup_env("cd", job, check_arm=False)
    os.environ.update(serve_env(kind))
    from src.executesimulation import execute_simulation
    from src.placement import fidelity_replay, snapshot_fidelity
    from src.placement.live_snapshot_seed import build_live_snapshot_seed
    from src.policy.gnn import orchestrator as GO
    from src.policy.gnn import prefix_serving as PS
    from src.policy.gnn import scheduler as GS

    snap = U._load_snapshot(job)
    spec = {"snapshot": snap, "cell_config": job["cell_config"], "sim_input": job["sim_input"], "live_run_params": snapshot_fidelity.live_run_params()}
    replicas_by_type = build_live_snapshot_seed(snap)["replicas_by_type"]
    types = [r["fn"] for r in sorted(snap["fidelity"]["batch"], key=lambda r: int(r["gid"]))]
    offered = {t: [(sp["node_name"], sp["platform_id"]) for sp in specs if sp.get("candidate", True)] for t, specs in replicas_by_type.items()}
    fr = fidelity_replay.FidelityReplay(spec, types, offered)
    local_of = {fr.batch_local[i]: i for i in range(len(fr.batch_local))}
    forced = dict(fr.queued_forced)
    t0 = float(snap["time"])
    phase = (math.ceil(t0) - t0) % 1.0
    orig_inf, orig_monitor, orig_mk = GS.GNNScheduler._prefix_inference, GO.GNNOrchestrator.monitor_process, PS.make_partial_state_score_fn
    seen: Dict[str, Any] = {"instances": [], "calls": []}

    def mk(model, graph, ctx):
        fn = orig_mk(model, graph, ctx)
        inst = {"graph_tl": {int(t): [(int(c[0]), int(c[1])) for c in cs] for t, cs in graph.task_logit_to_placement.items()}, "fn": fn, "steps": []}
        seen["instances"].append(inst)

        def rec(t, committed):
            sc = fn(t, committed)
            inst["steps"].append((int(t), {int(k): (int(v[0]), int(v[1])) for k, v in committed.items()}, [float(x) for x in sc]))
            return sc
        return rec

    def patched(self, batch_tasks, system_state, queue_snapshot, temporal_state):
        ids = {int(t.id) for t in batch_tasks}
        whole = set(local_of) <= ids
        seen["calls"].append((round(float(self.env.now), 3), len(batch_tasks), len(set(local_of) & ids)))
        if whole:
            seen["instances"].clear()
        res = orig_inf(self, batch_tasks, system_state, queue_snapshot, temporal_state)
        node_by_id = {nd.id: nd for nd in self.nodes.items}
        for i, t in enumerate(batch_tasks):
            f = forced.get(int(t.id))
            if f is not None:
                res[i] = (int(f[0]), int(f[1]))
                t.planned_node_name = node_by_id[int(f[0])].node_name
        if whole and "out" not in seen:
            order = sorted(((local_of[int(t.id)], i) for i, t in enumerate(batch_tasks) if int(t.id) in local_of))
            plan = [[int(res[i][0]), int(res[i][1])] for _b, i in order]
            seen["out"] = {"plan": plan, "n_decodes": len(seen["instances"]), "calls": seen["calls"], "diag": _diagnostics(seen["instances"], order, ils_plan, len(batch_tasks))}
            raise U._Done(seen["out"])
        return res

    def monitor(self):
        if phase > 0:
            yield self.env.timeout(phase)
        yield self.env.timeout(0)
        yield from orig_monitor(self)

    GS.GNNScheduler._prefix_inference = patched
    GO.GNNOrchestrator.monitor_process = monitor
    PS.make_partial_state_score_fn = mk
    try:
        infra = deepcopy(fr.base_infra)
        seed_live = deepcopy(fr.seed)
        seed_live["fidelity"] = snap["fidelity"]
        infra["live_snapshot_seed"] = seed_live
        infra["fast_forward_warmup"] = True
        infra["fast_forward_threshold"] = 1
        try:
            execute_simulation({"infrastructure": infra, "workload": deepcopy(fr.wl)}, fr.sim_inputs, scheduling_strategy="gnn_gnn", cache_policy="fifo", task_priority="fifo",
                               models=models, **fr.kw)
        except U._Done as d:
            out = d.out
        else:
            raise RuntimeError(f"FAIL LOUD: the simulation ended without the batch being decided whole: calls {seen['calls'][:12]}, t0 {t0}")
    finally:
        GS.GNNScheduler._prefix_inference = orig_inf
        GO.GNNOrchestrator.monitor_process = orig_monitor
        PS.make_partial_state_score_fn = orig_mk
    out.update(snap=snap, spec=spec, types=types)
    return out


def _diagnostics(instances: List[Dict[str, Any]], order: Sequence[Any], ils_plan: Sequence[Sequence[int]], n_batch_tasks: int) -> Dict[str, Any]:
    """Per task (snapshot batch order): [free-run rank, teacher-forced rank, n candidates] of the ILS placement; 0 = the ILS placement is not among the task's candidates."""
    if len(instances) != 1:
        return {"error": f"{len(instances)} decode instances (expected one whole-batch decode)"}
    inst = instances[0]
    if len(inst["graph_tl"]) != n_batch_tasks:
        return {"error": f"decode over {len(inst['graph_tl'])} tasks, batch of {n_batch_tasks}"}
    ils_by_idx = {i: tuple(ils_plan[k]) for k, (_b, i) in enumerate(order)}
    free = {t: sc for t, _c, sc in inst["steps"]}
    step_order = [t for t, _c, _s in inst["steps"]]
    ranks = {}
    committed: Dict[int, Any] = {}
    for t in step_order:
        cands = inst["graph_tl"][t]
        tf = [float(x) for x in inst["fn"](t, dict(committed))]
        pick = ils_by_idx[t]
        def rank(sc):
            if pick not in cands:
                return 0
            j = cands.index(pick)
            return 1 + sum(1 for i, x in enumerate(sc) if x > sc[j] or (x == sc[j] and cands[i] < pick))
        ranks[t] = [rank(free[t]), rank(tf), len(cands)]
        committed[t] = pick
    return {"ranks": [ranks[i] for _b, i in order]}


def run(jobs_path: str, rows_path: str, out_path: str, kind: str) -> None:
    recs = {}
    for l in open(rows_path):
        x = json.loads(l)
        if x.get("role", "first") == "first":
            recs[x["state_id"]] = x
    models = None
    with open(out_path, "a") as fh:
        for line in open(jobs_path):
            job = json.loads(line)
            if job.get("role", "first") != "first" or job["state_id"] not in recs:
                continue
            rec = recs[job["state_id"]]
            if models is None:
                U._setup_env("cd", job, check_arm=False)
                os.environ.update(serve_env(kind))
                from src.executesimulation import load_gnn_model, load_task_types_data

                gm, gd = load_gnn_model(CKPT[kind], space_config=json.load(open(job["cell_config"])))
                models = {"gnn_model": gm, "device": gd, "task_types_data": load_task_types_data(Path(job["sim_input"]))}
            d = learned_decision(job, kind, rec["best"], models)
            rt, _s, _r = U.replay_rtts(d["spec"], d["types"], [d["plan"]])
            row = {"state_id": job["state_id"], "bucket": rec["bucket"], "n_batch": rec["n_batch"], "kind": kind, "plan": d["plan"], "rtt": rt[0], "rtt_hand": rec["rtt_hand"],
                   "rtt_ils": rec["rtt_ils"], "n_decodes": d["n_decodes"], "diag": d["diag"], "equals_ils": d["plan"] == rec["best"], "equals_hand": d["plan"] == rec["hand"],
                   "pairs": {kind: pair_counts(d["snap"], d["plan"]), "hand": pair_counts(d["snap"], rec["hand"]), "ils": pair_counts(d["snap"], rec["best"])}}
            fh.write(json.dumps(row, separators=(",", ":")) + "\n"); fh.flush()
            print(job["state_id"], flush=True)


def summarize(rows: Sequence[Dict[str, Any]], kind: str) -> Dict[str, Any]:
    import statistics as st

    out: Dict[str, Any] = {}
    names = [n for n, _l, _h in U.BUCKETS]
    for name in ["pooled"] + names:
        rs = [r for r in rows if name == "pooled" or U._canon_bucket([r])[0]["bucket"] == name]
        if not rs:
            continue
        good = [r for r in rs if "ranks" in r["diag"]]
        reg = [(r["rtt"] - r["rtt_ils"]) / r["rtt_ils"] for r in rs]
        hreg = [(r["rtt_hand"] - r["rtt_ils"]) / r["rtt_ils"] for r in rs]
        tw = lambda key: (sum(r[key] for r in rs) - sum(r["rtt_ils"] for r in rs)) / sum(r["rtt_ils"] for r in rs)
        ent = [x for r in good for x in r["diag"]["ranks"]]
        rec = lambda col, k: sum(1 for x in ent if 0 < x[col] <= k) / len(ent) if ent else None
        out[name] = {"n_states": len(rs), "n_states_with_rank_diag": len(good), "n_tasks": len(ent),
                     f"{kind}_mean_regret_pct": 100 * st.fmean(reg), f"{kind}_median_pct": 100 * st.median(reg), f"{kind}_p90_pct": 100 * sorted(reg)[min(len(reg) - 1, int(0.9 * len(reg)))],
                     f"{kind}_task_weighted_pct": 100 * tw("rtt"), "hand_mean_regret_pct": 100 * st.fmean(hreg), "hand_median_pct": 100 * st.median(hreg),
                     "hand_task_weighted_pct": 100 * tw("rtt_hand"),
                     "recall_free_top1": rec(0, 1), "recall_free_top5": rec(0, 5), "recall_teacher_top1": rec(1, 1), "recall_teacher_top5": rec(1, 5),
                     "ils_not_a_candidate_tasks": sum(1 for x in ent if x[0] == 0), "mean_candidates": st.fmean(x[2] for x in ent) if ent else None,
                     "whole_plan_match_ils": sum(r["equals_ils"] for r in rs) / len(rs), "whole_plan_match_hand": sum(r["equals_hand"] for r in rs) / len(rs),
                     "cross_node_pairs_mean": {k: st.fmean(r["pairs"][k]["cross_node"] for r in rs) for k in (kind, "hand", "ils")},
                     "nodes_used_mean": {k: st.fmean(r["pairs"][k]["nodes_used"] for r in rs) for k in (kind, "hand", "ils")}}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("jobs"); r.add_argument("rows"); r.add_argument("out"); r.add_argument("--kind", choices=sorted(CKPT), required=True)
    s = sub.add_parser("summarize")
    s.add_argument("xrows"); s.add_argument("--kind", choices=sorted(CKPT), required=True); s.add_argument("--json", default=None)
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.jobs, a.rows, a.out, a.kind)
        return 0
    rows = [json.loads(l) for l in open(a.xrows) if l.strip()]
    res = summarize(rows, a.kind)
    print(json.dumps(res, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
