"""uncut_groups_s0_v1 Arm 1: turn a labelled state into a corpus dataset directory the trainers read (ds_NNNNN: infrastructure, workload, space_with_network, warm_snapshot,
fidelity_replay, generation_provenance, placement_metadata, best.json, optimal_result.json, placements/placements.jsonl).

The dataset is the whole batch (no sub-batching) over every candidate (declared_pruning_v1_uncapped: what the checkpoints serve). Its placements.jsonl is a SAMPLED file under
the headed convention (docs/notes/placements_jsonl_required.md): a header line, then the label plan and, when the label is the ILS plan, the hand plan, each replayed by the fidelity
replay with the corpus builder's own row format. The label plan is the minimum-RTT row by construction: an ILS label has r >= 0.10 over the hand plan; a hand label
carries no other row. best.json and optimal_result.json are the label plan's replay. States the corpus builder rejects (one candidate node, no choice, no peer pairs) are dropped and counted."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts_cosim import uncut_ils as U  # noqa: E402

SIM_INPUT = REPO / "data" / "nofs-ids"
TEMPLATE_PROV = "/share/nikola.lukic/workload_fix_v1/accel_replica_corpus/gnn_datasets_wf1_train/ds_00000/generation_provenance.json"


def dataset_plan(plan_gid_order: Sequence[Sequence[int]], gids_sorted: Sequence[int], trace_task_ids: Sequence[int]) -> Dict[str, List[int]]:
    """A plan in snapshot gid order -> the dataset's task indexing (events grouped by application, stable): {str(dataset task i): [node, platform]}."""
    pos = {g: k for k, g in enumerate(gids_sorted)}
    return {str(i): [int(plan_gid_order[pos[int(g)]][0]), int(plan_gid_order[pos[int(g)]][1])] for i, g in enumerate(trace_task_ids)}


def _rtt(stats: Dict[str, Any]) -> float:
    from src.executecosimulation import rtt_from_stats

    return float(rtt_from_stats(stats))


def build_state(label: Dict[str, Any], state: Dict[str, Any], root: str, out_dir: Path, idx: int, caches: Dict[str, Any]) -> Dict[str, Any]:
    from src.executecosimulation import DataclassJSONEncoder
    from src.placement import declared_slate, fidelity_replay
    from scripts_cosim import make_warm_corpus as M

    job = {"snapfile": state["snapfile"], "trigger_task_id": int(state["trigger"]), "time": state["time"],
           "cell_summary": f"{root}/cap/A/cc40s{state['topology']}__{state['window']}{state['rung']}__cd_ib0_s0.summary.json",
           "cell_config": f"{root}/inputs{state['set']}/inputs_group/wf1_{state['rung']}/cfg/cc40s{state['topology']}.json", "sim_input": str(SIM_INPUT)}
    U._setup_env("cd_ib0", job, check_arm=False)
    os.environ.update(HEROSIM_RETAIN_TASK_TIMES="1", HEROSIM_RETAIN_PEER_STATS="1")
    snap = U._load_snapshot(job)
    cell_path = Path(job["cell_config"])
    key = str(cell_path)
    if key not in caches:
        cell = json.loads(cell_path.read_text())
        scratch = Path(os.environ.get("TMPDIR", "/tmp")) / f"arm1_scratch_{os.getpid()}"
        scratch.mkdir(parents=True, exist_ok=True)
        caches[key] = {"cell": cell, "seed": int(cell["network"]["topology"]["seed"]),
                       "base": M.cell_base_infrastructure(cell_path, SIM_INPUT, int(cell["network"]["topology"]["seed"]), scratch)}
    c = caches[key]
    trace_path = f"{root}/inputs{state['set']}/inputs_group/wf1_{state['rung']}/wl/grounded_{state['window']}_n50000.json"
    if trace_path not in caches:
        caches[trace_path] = json.load(open(trace_path))
    trace = caches[trace_path]
    try:
        ids = M.check_consecutive_batch(snap, 1)
        workload = M.build_batch_workload(snap, trace, ids, list(c["cell"]["wsc"].keys()))
        fid = snap["fidelity"]
        waiting = sorted({int(o) for row in (fid.get("peers") or {}).values() for o, node, _p in row if node is None and str(o) not in (fid.get("future") or {})})
        if waiting:
            raise M.SnapshotRejected(f"{M.UNPLACED_PARTNER_REASON}: partner(s) {waiting} outside the batch were not placed at the snapshot")
        pairs0 = [(int(i), int(j)) for i, j, _ in trace.get("peer_exchange") or [] if int(i) in set(ids) and int(j) in set(ids)]
        sl = declared_slate.slate(snap["tasks"], pairs0, rule=declared_slate.RULE_UNCAPPED)
        gid_of = [int(t["task_id"]) for t in snap["tasks"]]
        kept = {gid_of[p]: {str(cc["queue_key"]) for cc in sl.kept[p]} for p in range(len(gid_of))}
        subset: Dict[str, set] = {}
        for t in snap["tasks"]:
            subset.setdefault(str(t["task_type"]), set()).update(kept[int(t["task_id"])])
        n_choice = sum(1 for t in snap["tasks"] if len(kept[int(t["task_id"])]) >= 2)
        if n_choice < 0.5 * len(snap["tasks"]) or sl.plans < 2:
            raise M.SnapshotRejected(f"{M.NO_CHOICE_REASON}: {n_choice} of {len(snap['tasks'])} tasks keep >= 2 candidates")
        if len(M.candidate_nodes(snap, subset)) < 2:
            raise M.SnapshotRejected(f"{M.SINGLE_NODE_REASON}: all offered candidates sit on one node")
    except M.SnapshotRejected as exc:
        return {"status": "rejected", "reason": str(exc)}
    flagged = M.flag_candidates_per_task(snap, kept)
    record = {"rule": declared_slate.RULE_UNCAPPED, "plans": sl.plans, "num_combos": sl.plans, "per_task_candidates": [len(kept[g]) for g in gid_of], "task_ids": gid_of}
    provenance = {"source_tag": f"uncut_arm1_{state['set']}", "snapshot_id": int(snap.get("snapshot_id", -1)), "snapshot_time": float(snap.get("time", 0.0)), "policy": snap.get("policy"),
                  "trigger_task_id": snap.get("trigger_task_id"), "task_ids": ids, "trace": trace_path, "cell_config": str(cell_path), "cell_seed": c["seed"], "candidate_draw": record,
                  "candidate_seed": 0}
    infra = M.build_infrastructure(c["base"], {k: v for k, v in flagged.items() if k != "fidelity"}, provenance, None)
    spec = fidelity_replay.build_spec({k: v for k, v in snap.items() if not k.startswith("_")}, cell_path, SIM_INPUT)
    by_gid = {int(t["task_id"]): t for t in flagged["tasks"]}
    spec["candidate_slate"] = declared_slate.RULE_UNCAPPED
    spec["task_candidates"] = [[[int(cc["node_id"]), int(cc["platform_id"])] for cc in by_gid[int(g)]["candidates"]] for g in workload["trace_task_ids"]]
    spec["slate"] = {k: v for k, v in record.items() if k != "task_ids"}
    spec["exposure"] = fidelity_replay.exposure(snap, [cc for g in workload["trace_task_ids"] for cc in by_gid[int(g)]["candidates"]])
    infra["live_snapshot_seed"][fidelity_replay.SPEC_KEY] = spec
    provenance["fidelity_replay"] = {k: v for k, v in spec.items() if k != "snapshot"}
    config = deepcopy(c["cell"])
    config.setdefault("scheduler", {})
    config["scheduler"]["batch_size"] = len(ids)
    config["scheduler"].setdefault("batch_timeout", 0.1)
    config["warm_snapshot"] = provenance

    gids_sorted = sorted(int(r["gid"]) for r in snap["fidelity"]["batch"])
    plans = [("label", label["label"])]
    if label["source"] == "ils" and label.get("hand") is not None and label["hand"] != label["label"]:
        plans.append(("hand", label["hand"]))
    apps = list(c["cell"]["wsc"].keys())
    fr = fidelity_replay.FidelityReplay(spec, [next(iter(e["application"]["dag"])) for e in workload["events"]],
                                        {t: [(sp["node_name"], sp["platform_id"]) for sp in specs if sp.get("candidate", True)] for t, specs in infra["live_snapshot_seed"]["replicas_by_type"].items()})
    s_of = {"label": label.get("label_S") if label["source"] == "exact" else (label.get("best_S") if label["source"] == "ils" else label.get("start_S")), "hand": label.get("start_S")}
    rows, results = [], {}
    for name, p in plans:
        dp = dataset_plan(p, gids_sorted, workload["trace_task_ids"])
        res = fr.cut_to_batch(fr.run({int(k): (int(v[0]), int(v[1])) for k, v in dp.items()}))
        stats = res["stats"]
        rtt = _rtt(stats)
        tt = [[int(tr["taskId"]), float(tr["dispatchedTime"]), float(tr["doneTime"])] for tr in stats["taskResults"] if tr.get("taskId") is not None and tr["taskId"] >= 0]
        rows.append({"name": name, "row": {"placement_plan": dp, "S": s_of[name], "iter": (label.get("iters") if (name == "label" and label["source"] == "ils") else 0), "rtt": rtt, "task_times": tt, "makespan": max(d for _, _, d in tt) - min(s for _, s, _ in tt),
                                           "peer_exchange_total": float(stats["totalPeerExchangeTime"])}})
        res["config"] = {"infrastructure": infra, "workload": workload}
        res["sample"] = {"apps": apps, "placement_plan": dp}
        results[name] = res
    label_row = rows[0]["row"]
    if len(rows) > 1 and rows[1]["row"]["rtt"] < label_row["rtt"]:
        # the registered rule (r >= 0.10 on the label stage's replay) makes the ILS plan the cheaper row; a re-replay that disagrees would silently flip the label
        raise RuntimeError(f"FAIL LOUD: {label['state_id']}: the hand plan replays cheaper ({rows[1]['row']['rtt']}) than the ILS label ({label_row['rtt']})")
    ds = out_dir / f"ds_{idx:05d}"
    if ds.exists():
        shutil.rmtree(ds)
    (ds / "placements").mkdir(parents=True)
    ev_file = label.get("evals_file")
    if ev_file and os.path.exists(ev_file):
        shutil.copyfile(ev_file, ds / "placements" / "ils_evaluated.jsonl.gz")
    header = {"header": True, "exhaustive": False, "plan_space_size": label.get("plan_space"), "n_tasks": len(ids), "label_source": label["source"],
              "search": "ils" if label["source"] in ("ils", "hand") else "exact_enumeration" if label["source"] == "exact" else "served_plan", "seed": 0, "cap": U.CAP,
              "s_pin": {"surrogate": "peer_greedy_network _pg_plan_cost", "batch_alpha": 0.0, "drain_alpha": 1.0},
              "evaluated_plans_file": "placements/ils_evaluated.jsonl.gz" if ev_file else None,
              "evaluated_plans_note": "every ILS-evaluated plan with its S and iteration, plan lists in snapshot gid order (trace_task_ids maps dataset task i to its gid); kept beside placements.jsonl, "
              "not in it, so the training cache parses 2 rows instead of tens of thousands (deviation from 'every evaluated plan goes in the file', disclosed)",
              "trace_task_ids": workload["trace_task_ids"], "snapshot_gids_sorted": gids_sorted, "label_rule": "ILS iff (RTT_hand - RTT_ILS) / RTT_hand >= 0.10",
              "r": label.get("r"), "eval_budget": label.get("budget"), "evals_done": label.get("n_evals"), "ils_version": U.ILS_VERSION, "state_id": label["state_id"],
              "replayed_rows_are": "the label plan and, for an ILS label, the hand plan; S on every row", "candidate_slate": declared_slate.RULE_UNCAPPED}
    with open(ds / "placements" / "placements.jsonl", "w") as fh:
        fh.write(json.dumps(header, separators=(",", ":")) + "\n")
        for r in rows:
            fh.write(json.dumps(r["row"], separators=(",", ":")) + "\n")
    with open(ds / "optimal_result.json", "w") as fh:
        json.dump(results["label"], fh, cls=DataclassJSONEncoder)
    (ds / "best.json").write_text(json.dumps({"file": "simulation_1_optimal.json", "rtt": label_row["rtt"]}))
    (ds / "infrastructure.json").write_text(json.dumps(infra))
    (ds / "workload.json").write_text(json.dumps(workload))
    (ds / "space_with_network.json").write_text(json.dumps(config))
    (ds / "warm_snapshot.json").write_text(json.dumps({"snapshot": flagged, "provenance": provenance}))
    (ds / "fidelity_replay.json").write_text(json.dumps({"fidelity": True, **provenance["fidelity_replay"], "snapshot_id": provenance["snapshot_id"], "snapshot_time": provenance["snapshot_time"]}, indent=1))
    prov = json.load(open(TEMPLATE_PROV))
    prov.update({"dataset_id": ds.name, "grid": f"uncut_arm1:{state['key']}", "num_tasks": len(ids), "seed": c["seed"], "infrastructure_override": str(ds / "infrastructure.json")})
    prov.pop("generated_at", None)
    prov["argv"] = ["--source-tag", f"wf1p_cc40s{state['topology']}_{state['rung']}_{state['window']}", "--fidelity"]      # make_r1a_split reads the topology from this tag
    (ds / "generation_provenance.json").write_text(json.dumps(prov, indent=1))
    n_rows = len(rows)
    (ds / "placement_metadata.json").write_text(json.dumps({"num_placements": n_rows, "completed": n_rows, "rows_written": n_rows, "timed_out": 0, "worker_failed": 0, "worker_exception": 0,
                                                            "early_terminated": False, "sweep_complete": True, "sampled": True, "exhaustive": False,
                                                            "fidelity_replay": {k: spec[k] for k in ("label", "live_run_params", "queued_tasks", "batch_tasks")}}))
    (ds / "placement_progress.txt").write_text(f"{n_rows}/{n_rows}\nBest RTT: {label_row['rtt']:.3f}s\nStatus: COMPLETE\n")
    return {"status": "built", "dataset": ds.name, "rtt": label_row["rtt"], "n_rows": n_rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ids"); ap.add_argument("labels", nargs="+", help="label JSONL files (uncut_arm1.py label)"); ap.add_argument("--root", default="/share/nikola.lukic/uncut_arm1")
    ap.add_argument("--out", required=True); ap.add_argument("--shard", default="0/1"); ap.add_argument("--manifest", required=True)
    a = ap.parse_args()
    states = json.load(open(a.ids))["states"]
    labels = {}
    for f in a.labels:
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                labels[r["state_id"]] = r
    k, m = (int(x) for x in a.shard.split("/"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    caches: Dict[str, Any] = {}
    done = set()
    if os.path.exists(a.manifest):
        done = {json.loads(l)["state_id"] for l in open(a.manifest) if l.strip()}
    with open(a.manifest, "a") as fh:
        for i, st_ in enumerate(states):
            if i % m != k or st_["key"] in done:
                continue
            lab = labels.get(st_["key"])
            if lab is None:
                raise SystemExit(f"FAIL LOUD: state {st_['key']} has no label")
            if lab["source"] == "dropped":
                rec = {"status": "dropped_at_label", "reason": lab.get("reason")}
            else:
                rec = build_state(lab, st_, a.root, out, i, caches)
            rec.update(state_id=st_["key"], idx=i, bucket=st_["bucket"], topology=st_["topology"], early=st_["early"], source=lab["source"])
            fh.write(json.dumps(rec, separators=(",", ":")) + "\n"); fh.flush()
            print(st_["key"], rec["status"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
