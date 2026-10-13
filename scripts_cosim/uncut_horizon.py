"""Myopia test (SCIENTIST, offline): the replay of a K1(b) state with the next N trace groups behind the batch.
The batch is forced to a plan (cd_ib0, cd_exactS_ib0's own decision, the ILS plan, or a hand plan one zero-dS move away: the chaos floor); the later groups are placed by cd_ib0.
PHASE 1 runs the simulation under peer_greedy_network_cd (cd_ib0) with the queued and batch tasks forced and the later groups free, and records where cd_ib0 put each later task.
PHASE 2 replays batch + later groups through the same determined/forced path as the batch-only replay (fidelity_replay.FidelityReplay.run) with every placement forced,
so with no later group it is the batch-only replay itself. One phase-2 run with the maximum horizon gives every smaller horizon: the later tasks cannot influence earlier ones.
Metric per plan and horizon N: sum over the batch and the tasks of the next N groups of (doneTime - scheduledTime)."""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics as st
import sys
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from hashlib import sha256
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts_cosim import uncut_ils as U  # noqa: E402

HORIZONS = (1, 3, 10)
HORIZON_SPECS = ("N50", "W")          # N<k>: the next k trace groups; W: every group within max(120 s, the hand plan's batch makespan) of the batch
_TRACES: Dict[str, Any] = {}


def choose_ids(rows: Sequence[Dict[str, Any]], seed: int, n_top: int = 50, n_rand: int = 50) -> Dict[str, Any]:
    firsts = [r for r in rows if r.get("role", "first") == "first" and r.get("regret") is not None]
    top = sorted(firsts, key=lambda r: (-r["regret"], r["state_id"]))[:n_top]
    taken = {r["state_id"] for r in top}
    pool = sorted(r["state_id"] for r in firsts if r["regret"] <= 1.0 and r["state_id"] not in taken)
    rand = random.Random(f"uncut_horizon:{seed}").sample(pool, n_rand)
    return {"seed": seed, "rule": f"top {n_top} by cd_ib0 regret + {n_rand} random of the {len(pool)} other states with regret <= 100 % (sorted ids, random.Random('uncut_horizon:{seed}').sample)",
            "top": [r["state_id"] for r in top], "random": sorted(rand), "n_pool": len(pool)}


def trace_extension(job: Dict[str, Any], fr: Any, snap: Dict[str, Any], n_groups: Optional[int] = None, window_s: Optional[float] = None) -> Dict[str, Any]:
    """The workload of the replay with the next `n_groups` trace groups appended (timestamps relative to the batch's last arrival; peer pairs among them)."""
    name = json.load(open(job["cell_summary"]))["workload"]
    wl_path = Path(job["cell_config"]).parents[1] / "wl" / f"{name}.json"
    trace = _TRACES.get(str(wl_path)) or _TRACES.setdefault(str(wl_path), json.load(open(wl_path)))
    events = trace["events"]
    gids = [int(r["gid"]) for r in snap["fidelity"]["batch"]]
    base = max(float(events[g]["timestamp"]) for g in gids)
    batch_groups = {events[g]["peer_group"] for g in gids}
    first_ts: Dict[Any, float] = {}
    members: Dict[Any, List[int]] = {}
    for i, e in enumerate(events):
        g = e["peer_group"]
        first_ts.setdefault(g, float(e["timestamp"]))
        members.setdefault(g, []).append(i)
    later = sorted((g for g in members if g not in batch_groups and first_ts[g] > base and (window_s is None or first_ts[g] - base <= window_s)), key=lambda g: (first_ts[g], str(g)))
    later = later[:n_groups] if n_groups is not None else later
    chosen = [i for g in later for i in members[g]]
    pos = {gid: k for k, gid in enumerate(chosen)}
    wl = deepcopy(fr.wl)
    n_base = len(wl["events"])
    for gid in chosen:
        e = deepcopy(events[gid])
        e.pop("peer_group", None)
        e["timestamp"] = float(events[gid]["timestamp"]) - base
        wl["events"].append(e)
    pairs = [[n_base + pos[a], n_base + pos[b], pay] for a, b, pay in trace["peer_exchange"] if a in pos and b in pos]
    wl["peer_exchange"] = list(wl.get("peer_exchange") or []) + pairs
    wl["rps"] = len(wl["events"])
    return {"wl": wl, "n_base": n_base, "group_of": {n_base + pos[gid]: k + 1 for k, g in enumerate(later) for gid in members[g]}, "n_later": len(chosen),
            "groups": [{"group": g, "size": len(members[g]), "arrival_after_batch_s": first_ts[g] - base} for g in later]}


def _context(job: Dict[str, Any]):
    from src.placement import fidelity_replay, snapshot_fidelity
    from src.placement.live_snapshot_seed import build_live_snapshot_seed

    snap = U._load_snapshot(job)
    spec = {"snapshot": snap, "cell_config": job["cell_config"], "sim_input": job["sim_input"], "live_run_params": snapshot_fidelity.live_run_params()}
    rbt = build_live_snapshot_seed(snap)["replicas_by_type"]
    types = [r["fn"] for r in sorted(snap["fidelity"]["batch"], key=lambda r: int(r["gid"]))]
    offered = {t: [(sp["node_name"], sp["platform_id"]) for sp in specs if sp.get("candidate", True)] for t, specs in rbt.items()}
    return snap, spec, types, offered


def phase1(job: Dict[str, Any], plan: Sequence[Sequence[int]], ext: Dict[str, Any]) -> Dict[str, Any]:
    """cd_ib0 places the later tasks while the queued and batch tasks are forced. Returns the placements of the later tasks ({replay task id: [node id, platform id]}) and the
    summed (doneTime - scheduledTime) of the batch and of each later group, read from this same run: a later task may sit on a replica the autoscaler created during the run, which a
    forced replay of the same placements would not have (phase 2 fails on it), so the horizon metric is this run's."""
    U._setup_env("cd_ib0", job)
    from src.executesimulation import execute_simulation
    from src.placement import fidelity_replay
    from src.policy.gnn import orchestrator as GO
    from src.policy.peer_greedy_network import scheduler as PG

    snap, spec, types, offered = _context(job)
    fr = fidelity_replay.FidelityReplay(spec, types, offered)
    forced = dict(fr.queued_forced)
    forced.update({fr.batch_local[i]: (int(p[0]), int(p[1])) for i, p in enumerate(plan)})
    t0 = float(snap["time"])
    phase = (math.ceil(t0) - t0) % 1.0
    orig_inf, orig_monitor = PG.PeerGreedyNetworkBatchScheduler._prefix_inference, GO.GNNOrchestrator.monitor_process
    placed: Dict[int, List[int]] = {}

    def patched(self, batch_tasks, system_state, queue_snapshot, temporal_state):
        node_by_id = {nd.id: nd for nd in self.nodes.items}
        out, free = {}, []
        for i, t in enumerate(batch_tasks):
            f = forced.get(int(t.id))
            if f is not None:
                out[i] = (int(f[0]), int(f[1]))
                t.planned_node_name = node_by_id[int(f[0])].node_name
            else:
                free.append(i)
        if not out:
            res = orig_inf(self, batch_tasks, system_state, queue_snapshot, temporal_state)
        else:
            res = {}
            if free:
                sub = self._pg_decide([batch_tasks[i] for i in free], system_state)
                for k, i in enumerate(free):
                    res[i] = sub[k]
            res.update(out)
        for i, t in enumerate(batch_tasks):
            if int(t.id) >= ext["n_base"]:
                placed[int(t.id)] = [int(res[i][0]), int(res[i][1])]
        return res

    def monitor(self):
        if phase > 0:
            yield self.env.timeout(phase)
        yield self.env.timeout(0)
        yield from orig_monitor(self)

    PG.PeerGreedyNetworkBatchScheduler._prefix_inference = patched
    GO.GNNOrchestrator.monitor_process = monitor
    try:
        infra = deepcopy(fr.base_infra)
        seed_live = deepcopy(fr.seed)
        seed_live["fidelity"] = snap["fidelity"]
        infra["live_snapshot_seed"] = seed_live
        infra["fast_forward_warmup"] = True
        infra["fast_forward_threshold"] = 1
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            result = execute_simulation({"infrastructure": infra, "workload": deepcopy(ext["wl"])}, fr.sim_inputs, scheduling_strategy=U.ARMS["cd_ib0"][0], cache_policy="fifo",
                                        task_priority="fifo", models=None, **fr.kw)
    finally:
        PG.PeerGreedyNetworkBatchScheduler._prefix_inference = orig_inf
        GO.GNNOrchestrator.monitor_process = orig_monitor
    missing = [i for i in ext["group_of"] if i not in placed]
    if missing:
        raise RuntimeError(f"FAIL LOUD: phase 1 never placed later tasks {missing[:5]} (of {len(missing)})")
    trs = {tr["taskId"]: tr for tr in result["stats"]["taskResults"] if tr.get("taskId", -1) >= 0}
    missing = [b for b in list(fr.batch_local) + list(ext["group_of"]) if b not in trs]
    if missing:
        raise RuntimeError(f"FAIL LOUD: the horizon run finished without results for tasks {missing[:5]} (of {len(missing)})")
    label = lambda b: float(trs[b]["doneTime"]) - float(trs[b]["scheduledTime"])
    groups: Dict[int, float] = {}
    for b, k in ext["group_of"].items():
        groups[k] = groups.get(k, 0.0) + label(b)
    def lock(ids):
        w = [float(trs[b]["computeStartTime"]) - float(trs[b]["ioEndTime"]) for b in ids if trs[b].get("computeStartTime") is not None and trs[b].get("ioEndTime") is not None]
        if ids and not w:
            raise RuntimeError("FAIL LOUD: taskResults carry no computeStartTime/ioEndTime; lock wait cannot be read")
        return {"sum": sum(w), "n": len(w), "n_tasks": len(list(ids))}

    return {"lock_wait": {"batch": lock(list(fr.batch_local)), "later": lock(list(ext["group_of"]))}, "placed": placed, "batch": sum(label(b) for b in fr.batch_local), "batch_makespan": max(label(b) for b in fr.batch_local), "groups": [groups.get(k, 0.0) for k in range(1, max(ext["group_of"].values(), default=0) + 1)]}


def phase2(job: Dict[str, Any], plan: Sequence[Sequence[int]], ext: Dict[str, Any], placed: Dict[int, List[int]]) -> Dict[str, Any]:
    """Replay with every placement forced. Returns the batch sum (as plan_oracle._label_chunk sums it) and the per-group sums of the later tasks."""
    U._setup_env("cd_ib0", job)
    from src.placement.fidelity_replay import FidelityReplay

    snap, spec, types, _offered = _context(job)
    spec = dict(spec, queued_tasks=len(snap["fidelity"]["queued"]), batch_tasks=len(snap["fidelity"]["batch"]))
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        replay = FidelityReplay(spec, types, {})
        replay.queued_forced = {**replay.queued_forced, **{int(i): (int(p[0]), int(p[1])) for i, p in placed.items()}}
        replay.wl = deepcopy(ext["wl"])
        result = replay.run({i: (int(p[0]), int(p[1])) for i, p in enumerate(plan)})
    trs = {tr["taskId"]: tr for tr in result["stats"]["taskResults"] if tr.get("taskId", -1) >= 0}
    missing = [b for b in list(replay.batch_local) + list(ext["group_of"]) if b not in trs]
    if missing:
        raise RuntimeError(f"FAIL LOUD: horizon replay finished without results for tasks {missing[:5]} (of {len(missing)})")
    label = lambda b: float(trs[b]["doneTime"]) - float(trs[b]["scheduledTime"])
    groups: Dict[int, float] = {}
    for b, k in ext["group_of"].items():
        groups[k] = groups.get(k, 0.0) + label(b)
    return {"batch": sum(label(b) for b in replay.batch_local), "groups": [groups.get(k, 0.0) for k in range(1, max(ext["group_of"].values(), default=0) + 1)]}


def run_state(job: Dict[str, Any], plans: Dict[str, Sequence[Sequence[int]]]) -> Dict[str, Any]:
    """Every plan at N = 0 (the batch alone) and at each horizon N, each with its own phase 1 and phase 2: later arrivals share links with the batch, so the batch's own
    RTT depends on how much follows it, and a smaller horizon cannot be read off a larger run."""
    from src.placement import fidelity_replay

    U._setup_env("cd_ib0", job)
    snap, spec, types, offered = _context(job)
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        fr = fidelity_replay.FidelityReplay(spec, types, offered)
    exts = {"0": {"wl": fr.wl, "n_base": len(fr.wl["events"]), "group_of": {}, "n_later": 0, "groups": []}}
    hand0 = phase1(job, plans["hand"], exts["0"])
    window = max(120.0, hand0["batch_makespan"])          # every group arriving within max(120 s, the hand plan's batch makespan) after the batch
    for spec_ in HORIZON_SPECS:
        exts[spec_] = trace_extension(job, fr, snap, n_groups=int(spec_[1:]) if spec_.startswith("N") else None, window_s=window if spec_ == "W" else None)
    out: Dict[str, Any] = {"state_id": job["state_id"], "window_s": window, "n_later": {k: v["n_later"] for k, v in exts.items()},
                           "groups": {k: v["groups"] for k, v in exts.items() if k != "0"}, "plans": {}}
    for name, plan in plans.items():
        if plan is None:
            out["plans"][name] = None
            continue
        per_n = {}
        for key, ext in exts.items():
            r = hand0 if (key == "0" and name == "hand") else phase1(job, plan, ext)
            per_n[key] = {"batch": r["batch"], "later": sum(r["groups"]), "total": r["batch"] + sum(r["groups"]), "n_later_tasks": ext["n_later"], "batch_makespan": r["batch_makespan"], "lock_wait": r["lock_wait"]}
        out["plans"][name] = {"by_n": per_n, "plan": [list(p) for p in plan]}
    return out


def summarize(rows: Sequence[Dict[str, Any]], ids: Dict[str, Any]) -> Dict[str, Any]:
    """Per stratum and pooled, for ILS and cd_exactS_ib0 against the hand, at every horizon key the rows carry: the advantage in summed RTT seconds (hand - arm), the share of the
    batch-only advantage kept, the share of states with the batch-only sign, the median per-state advantage, the chaos floor (median |total(hand) - total(hand one zero-dS move away)|),
    and the later-task-only difference (hand - arm; the pile-up signal), split into states where the arm's plan uses one node and the rest."""
    strata = {"pooled": [r["state_id"] for r in rows], "top50": ids["top"], "random50": ids["random"]}
    by = {r["state_id"]: r for r in rows}
    keys = [k for k in rows[0]["plans"]["hand"]["by_n"] if k != "0"]
    one_node = lambda r, arm: len({p[0] for p in r["plans"][arm]["plan"]}) == 1
    out: Dict[str, Any] = {}
    for sname, sids in strata.items():
        rs = [by[s] for s in sids if s in by]
        if not rs:
            continue
        blk: Dict[str, Any] = {"n_states": len(rs), "window_s_median": st.median(r["window_s"] for r in rs)}
        for arm in ("ils", "exactS"):
            res: Dict[str, Any] = {}
            adv0 = [r["plans"]["hand"]["by_n"]["0"]["total"] - r["plans"][arm]["by_n"]["0"]["total"] for r in rs]
            for key in ["0"] + keys:
                tot = lambda r, plan: r["plans"][plan]["by_n"][key]["total"]
                adv = [tot(r, "hand") - tot(r, arm) for r in rs]
                chaos = [abs(tot(r, "hand") - tot(r, "chaos")) for r in rs if r["plans"].get("chaos")]
                cell = {"sum_advantage_s": sum(adv), "kept_vs_batch_only": sum(adv) / sum(adv0) if sum(adv0) else None,
                        "sign_agrees_with_batch_only": sum((x > 0) == (y > 0) for x, y in zip(adv, adv0)) / len(rs), "median_advantage_s": st.median(adv),
                        "mean_hand_total_s": st.fmean(tot(r, "hand") for r in rs), "median_chaos_floor_s": st.median(chaos) if chaos else None, "n_chaos_states": len(chaos),
                        "median_later_tasks": st.median(r["plans"]["hand"]["by_n"][key]["n_later_tasks"] for r in rs)}
                if key != "0":
                    later = [r["plans"]["hand"]["by_n"][key]["later"] - r["plans"][arm]["by_n"][key]["later"] for r in rs]
                    cell["later_only_hand_minus_arm_s"] = {"sum": sum(later), "median": st.median(later), "hand_later_worse_states": sum(x > 1e-9 for x in later),
                                                          "arm_later_worse_states": sum(x < -1e-9 for x in later)}
                    for tag, sel in (("one_node", [i for i, r in enumerate(rs) if one_node(r, arm)]), ("rest", [i for i, r in enumerate(rs) if not one_node(r, arm)])):
                        cell["later_only_hand_minus_arm_s"][tag] = {"n_states": len(sel), "sum": sum(later[i] for i in sel), "hand_later_worse": sum(later[i] > 1e-9 for i in sel),
                                                                    "arm_later_worse": sum(later[i] < -1e-9 for i in sel), "median": st.median([later[i] for i in sel]) if sel else None}
                res[key] = cell
            blk[arm] = res
        out[sname] = blk
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("choose")
    c.add_argument("rows"); c.add_argument("out"); c.add_argument("--seed", type=int, required=True)
    r = sub.add_parser("run")
    r.add_argument("jobs"); r.add_argument("rows"); r.add_argument("xrows", help="uncut_xarm rows (the cd_exactS_ib0 plans)"); r.add_argument("ids"); r.add_argument("out")
    r.add_argument("--shard", default="0/1", help="k/m: every m-th selected state from k")
    r.add_argument("--no-horizon", action="store_true", help="identity check only: the batch-only replay through this module's phase 2")
    s = sub.add_parser("summarize")
    s.add_argument("hrows"); s.add_argument("ids"); s.add_argument("--json", default=None)
    a = ap.parse_args()
    if a.cmd == "choose":
        rows = [json.loads(l) for l in open(a.rows) if l.strip()]
        ids = choose_ids(rows, a.seed)
        text = json.dumps(ids, indent=1)
        Path(a.out).write_text(text)
        print(text, "\nsha256", sha256(text.encode()).hexdigest())
        return 0
    if a.cmd == "summarize":
        rows = [json.loads(l) for l in open(a.hrows) if l.strip()]
        res = summarize(rows, json.load(open(a.ids)))
        print(json.dumps(res, indent=1))
        if a.json:
            Path(a.json).write_text(json.dumps(res, indent=1))
        return 0
    ids = json.load(open(a.ids))
    want = ids["top"] + ids["random"]
    k, m = (int(x) for x in a.shard.split("/"))
    want = want[k::m]
    recs = {json.loads(l)["state_id"]: json.loads(l) for l in open(a.rows) if json.loads(l).get("role", "first") == "first"}
    xrecs = {json.loads(l)["state_id"]: json.loads(l) for l in open(a.xrows)}
    jobs = {json.loads(l)["state_id"]: json.loads(l) for l in open(a.jobs) if json.loads(l).get("role", "first") == "first"}
    with open(a.out, "a") as fh:
        for sid in want:
            job, rec = jobs.get(sid), recs[sid]
            if job is None:
                continue
            zm = U.decision_pass(dict(job, zero_move=True), "cd_ib0", 0, 5, 200, 30.0)
            if zm["hand"] != rec["hand"]:
                raise RuntimeError(f"FAIL LOUD: state {sid}: the re-decided hand plan differs from the recorded one")
            plans = {"hand": rec["hand"], "ils": rec["best"], "exactS": xrecs[sid]["plan_b"], "chaos": zm["zero_move"]["plan"] if zm["zero_move"] else None}
            if a.no_horizon:
                plans = {k_: v for k_, v in plans.items() if k_ in ("hand", "ils", "exactS")}
            row = run_state(dict(job), plans) if not a.no_horizon else _identity(job, plans)
            row["bucket"], row["n_batch"] = rec["bucket"], rec["n_batch"]
            row["rows_rtt"] = {"hand": rec["rtt_hand"], "ils": rec["rtt_ils"], "exactS": xrecs[sid]["rtt_b"]}
            row["zero_move"] = zm["zero_move"]
            fh.write(json.dumps(row, separators=(",", ":")) + "\n"); fh.flush()
            print(sid, flush=True)
    return 0


def _identity(job: Dict[str, Any], plans: Dict[str, Sequence[Sequence[int]]]) -> Dict[str, Any]:
    """No later group: phase 2 with the base workload. Its batch sums must equal the batch-only rows bit for bit."""
    from src.placement import fidelity_replay

    U._setup_env("cd_ib0", job)
    snap, spec, types, offered = _context(job)
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        fr = fidelity_replay.FidelityReplay(spec, types, offered)
    ext = {"wl": fr.wl, "n_base": len(fr.wl["events"]), "group_of": {}}
    return {"state_id": job["state_id"], "plans": {n: {"by_n": {"0": {"batch": phase2(job, p, ext, {})["batch"]}}} for n, p in plans.items() if p is not None}}


if __name__ == "__main__":
    sys.exit(main())
