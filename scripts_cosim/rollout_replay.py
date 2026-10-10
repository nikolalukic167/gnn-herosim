#!/usr/bin/env python3
"""cost_to_go_v1: a fidelity replay of one batch decision continued over the next H seconds of the cell's own trace.

Q_H(state, plan) = the batch's latency + the summed latency of every task of the rollout window, where the window is every
trace task with index below n(t0 + H) that the live run had not scheduled by the snapshot instant t0 (tasks that arrived
and were still waiting, at offset 0, and tasks arriving in (t0, t0 + H], at their own offsets). Latency is
doneTime - scheduledTime, the corpus label's quantity. The batch is forced to the given plan (or decided by the
continuation policy restricted to the dataset's sweep candidates, `--batch policy`), the snapshot's queued tasks keep
their recorded replicas, and every window task is decided by the continuation policy: cd_exactS (HEROSIM_PG_CD_EXACT=1
with the expansion as its over-cap fallback, the gate arm's env), plain CD, or the live capture policy
(peer_greedy_network_batch, for the self-check). Arrivals after t0 + H are absent, as in the cut live run the self-check
compares against (HEROSIM_MAX_EVENTS keeps the peer pairs whose both ends survive; so does the rollout).

Partners of window tasks that the live run had already placed (in flight or done at t0) are stubs at their live node, as
the replay's own `apply_orchestrator` stubs the snapshot tasks' outside partners. The scheduled times and nodes come from
a TABLE of the full live run (`--make-table`), which must reproduce the corpus' snapshots (checked by the sbatch).

  rollout_replay.py --make-table RAW.json TABLE.json
  rollout_replay.py OUT.jsonl JOBS.jsonl [--workers N]
      each job: {ds, table, workload, H, plan: [[node_id, platform_id] per dataset task] | "policy" | "live",
                 continuation: cd_exacts | cd | live, eps: 0.0, tag}
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CONTINUATION = {
    "cd_exacts": ("peer_greedy_network_cd_peer_greedy_network_cd", {"HEROSIM_PG_CD_EXACT": "1", "HEROSIM_PG_CD_EXPANSION": "1"}),
    "cd": ("peer_greedy_network_cd_peer_greedy_network_cd", {}),
    "live": ("peer_greedy_network_batch_peer_greedy_network_batch", {}),
}
DETAIL = ("scheduledTime", "arrivedTime", "startedTime", "doneTime", "coldStartTime", "coldStarted", "peerExchangeTime", "pullWaitTime",
          "peerRendezvousWait", "linkWaitTime", "executionNode", "executionPlatform")
PERTURBED = ("compute_remaining", "io_remaining", "net_remaining", "hold_remaining", "cold_remaining")


def _log_mutex(orchestrator, path: str, offset: float, until: float) -> None:
    """Debug (CTG_MUTEX_LOG): every request, acquisition and release of the orchestrator mutex up to `until`, with the
    calling generator, on the rollout's clock (truth times minus `offset`)."""
    m, env = orchestrator.mutex, orchestrator.env
    fh = open(path, "w")
    get0, put0 = m.get, m.put

    def get(*a, **kw):
        who = sys._getframe(1).f_code.co_name
        t_req = env.now
        ev = get0(*a, **kw)
        if t_req - offset <= until:
            ev.callbacks.append(lambda _e: fh.write(f"{t_req - offset:.6f} {env.now - offset:.6f} GET {who}\n"))
        return ev

    def put(*a, **kw):
        if env.now - offset <= until:
            fh.write(f"{env.now - offset:.6f} {env.now - offset:.6f} PUT {sys._getframe(1).f_code.co_name}\n")
        return put0(*a, **kw)

    m.get, m.put = get, put


class _Every:
    def __contains__(self, _key) -> bool:
        return True


class _Restrict(dict):
    """The sweep's candidates for the dataset's batch tasks; any other task sharing the decision is unrestricted."""

    def __missing__(self, _key):
        return _Every()


def make_table(raw_path: str, out_path: str) -> None:
    res = json.load(open(raw_path))
    trs = [tr for tr in res["stats"]["taskResults"] if tr.get("taskId", -1) >= 0]
    n = max(int(tr["taskId"]) for tr in trs) + 1
    tab = {"n": n, "scheduled": [None] * n, "done": [None] * n, "node": [None] * n, "platform": [None] * n}
    for tr in trs:
        i = int(tr["taskId"])
        tab["scheduled"][i] = float(tr["scheduledTime"])
        tab["done"][i] = float(tr["doneTime"])
        tab["node"][i] = str(tr["executionNode"])
        tab["platform"][i] = int(tr["executionPlatform"])
    missing = sum(1 for x in tab["done"] if x is None)
    if missing:
        raise RuntimeError(f"{raw_path}: {missing} of {n} tasks have no result")
    tab["code"] = (res.get("run_provenance") or {}).get("code")
    json.dump(tab, open(out_path, "w"))


def perturb(fid, eps: float) -> None:
    """Every in-flight remaining time x (1 + eps): ghost stages, image-pull holds, platforms' pull_remaining."""
    for g in fid.get("ghosts") or []:
        for k in PERTURBED:
            if g.get(k) is not None and float(g[k]) > 0:
                g[k] = float(g[k]) * (1 + eps)
    for p in fid.get("pulls") or []:
        p["own"] = float(p.get("own") or 0) * (1 + eps)
    for rec in (fid.get("platforms") or {}).values():
        if rec.get("pull_remaining") is not None:
            rec["pull_remaining"] = float(rec["pull_remaining"]) * (1 + eps)


def window_of(fid, t0: float, H: float, arrivals, tab, trace_pairs):
    """(window task ids in trace order, cut n): every trace task below n(t0 + H) the live run had not scheduled by t0. Fails loud when
    the table's history disagrees with the snapshot (a task scheduled before t0 that is neither done, in flight nor in the snapshot)."""
    n_cut = bisect.bisect_right(arrivals, t0 + max(H, 0.0))
    snap = {int(r["gid"]) for r in fid["queued"]} | {int(r["gid"]) for r in fid["batch"]}
    ghosts = {int(g["tid"]) for g in fid.get("ghosts") or []}
    tol = 1e-9
    window, bad = [], []
    for i in range(n_cut):
        if i in snap or i in ghosts:
            continue
        s, d = tab["scheduled"][i], tab["done"][i]
        if s < t0 - tol:
            if d > t0 + tol:
                bad.append(i)
            continue
        if H >= 0:
            window.append(i)
    if bad:
        raise RuntimeError(f"table history disagrees with the snapshot: tasks {bad[:5]} were scheduled before t0 and not done, "
                           "but are neither queued, batch nor in flight in it")
    for i in snap:
        if abs(tab["scheduled"][i] - t0) > 1e-6 and i in {int(r["gid"]) for r in fid["batch"]}:
            raise RuntimeError(f"batch task {i} was scheduled at {tab['scheduled'][i]} in the table, the snapshot is at {t0}")
    return window, n_cut


def _one(job):
    t_start = time.time()
    ds = Path(job["ds"])
    H, eps = float(job["H"]), float(job.get("eps") or 0.0)
    base = dict(ds=str(ds), H=H, eps=eps, plan_kind=job["plan"] if isinstance(job["plan"], str) else "given",
                continuation=job["continuation"], tag=job.get("tag"))
    try:
        os.environ["HEROSIM_SNAPSHOT_FIDELITY"] = "1"
        prov = json.load(open(ds / "generation_provenance.json"))
        for k, v in (prov.get("physics_env") or {}).items():
            if k.startswith("HEROSIM_") or k in ("COSIM_AUTOSCALER_RECONCILE_INTERVAL",):
                os.environ.setdefault(k, str(v))
        os.environ.update(GNN_DECODE_MODE="masked_topo", GNN_BATCH_BY_PEER_GROUP="1", PYTHONHASHSEED="0", SIM_FORCE_FULL_STATS="1")
        for k in ("GNN_SERVE_CANDIDATE_SLATE", "HEROSIM_PG_EXCHANGE_SCALE", "HEROSIM_PG_CD_PASSES", "HEROSIM_PG_BATCH_BLIND",
                  "HEROSIM_PG_CD_EXPANSION", "HEROSIM_PG_CD_EXACT", "GNN_SLATE_NO_SPLIT", "HEROSIM_MAX_EVENTS"):
            os.environ.pop(k, None)
        strategy, extra_env = CONTINUATION[job["continuation"]]
        os.environ.update(extra_env)
        from copy import deepcopy

        from src.executesimulation import execute_simulation
        from src.placement import fidelity_replay, snapshot_fidelity
        from src.policy.peer_greedy_network import scheduler as PG

        seed = json.load(open(ds / "infrastructure.json"))["live_snapshot_seed"]
        spec = deepcopy(seed["fidelity_replay"])
        if not Path(spec["sim_input"]).exists():
            spec["sim_input"] = str(REPO / "data" / "nofs-ids")
        if eps:
            perturb(spec["snapshot"]["fidelity"], eps)
        fid = spec["snapshot"]["fidelity"]
        t0 = float(spec["snapshot"]["time"])
        types = [next(iter(e["application"]["dag"])) for e in json.load(open(ds / "workload.json"))["events"]]
        offered = {t: [(sp["node_name"], sp["platform_id"]) for sp in specs if sp.get("candidate", True)]
                   for t, specs in seed["replicas_by_type"].items()}
        fr = fidelity_replay.FidelityReplay(spec, types, offered)
        rows = [r for r in (json.loads(l) for l in open(ds / "placements" / "placements.jsonl") if l.strip()) if "placement_plan" in r]

        trace = json.load(open(job["workload"]))
        arrivals = [float(e["timestamp"]) for e in trace["events"]]
        tab = json.load(open(job["table"]))
        window, n_cut = window_of(fid, t0, H, arrivals, tab, None)
        if tab["n"] < n_cut:
            raise RuntimeError(f"table has {tab['n']} tasks, the rollout needs {n_cut}")

        ids = snapshot_fidelity.local_ids(fid)  # live id -> replay id for queued + batch
        wl = deepcopy(fr.wl)
        nxt = len(wl["events"])
        for g in window:
            ev = trace["events"][g]
            wl["events"].append({"timestamp": max(0.0, arrivals[g] - t0), "application": deepcopy(ev["application"]),
                                 "qos": deepcopy(ev["qos"]), "node_name": ev["node_name"]})
            ids[g] = nxt
            nxt += 1
        wl["rps"] = len(wl["events"])
        wl["duration"] = max(1, int(H) + 1)
        win = set(window)
        snapset = set(ids) - win
        pairs = list(wl.get("peer_exchange") or [])
        stub_rows = {}  # replay id -> [(live partner id, node name, payload, placement delay from t0 or None if placed by t0)]
        beyond = 0
        scored_ids = win | {int(r["gid"]) for r in fid["batch"]}
        pairs_touch = pairs_cut = 0
        batch_gids = {int(r["gid"]) for r in fid["batch"]}
        ghost_ids = {int(g["tid"]) for g in fid.get("ghosts") or []}  # in flight at t0, possibly decided at t0 itself before the batch
        siblings = set()
        for a, b, payload in trace.get("peer_exchange") or []:
            a, b = int(a), int(b)
            if a in scored_ids or b in scored_ids:
                pairs_touch += 1
                pairs_cut += a >= n_cut or b >= n_cut
            for x, y in ((a, b), (b, a)):
                if x in batch_gids and y in win and abs(tab["scheduled"][y] - t0) <= 1e-6:
                    siblings.add(y)  # decided live in the batch's own decision: the dataset is a sub-batch view
            if a not in win and b not in win:
                continue  # pairs among snapshot tasks are the snapshot's own; pairs among placed tasks do not matter
            if a in ids and b in ids:
                if a in snapset and b in snapset:
                    continue
                pairs.append([ids[a], ids[b], float(payload)])
                continue
            inside, other = (a, b) if a in ids else (b, a)
            if other >= n_cut:
                # a partner beyond the horizon: absent as a task, but placed where and when the full live run placed it, so a
                # window task rendezvouses and pays the exchange to it as live (the cut truth run uses the same convention)
                beyond += 1
                stub_rows.setdefault(ids[inside], []).append((other, tab["node"][other], float(payload), tab["scheduled"][other] - t0))
                continue
            if other not in ghost_ids and tab["scheduled"][other] >= t0 - 1e-9 and tab["done"][other] > t0:
                raise RuntimeError(f"window task {inside}'s partner {other} is unscheduled at t0 but not in the window")
            stub_rows.setdefault(ids[inside], []).append((other, tab["node"][other], float(payload), None))
        if pairs:
            wl["peer_exchange"] = pairs
        # snapshot tasks' outside partners that are now window tasks become real pairs (none in the accel corpus: no open peers)
        for gid, prow in list((fid.get("peers") or {}).items()):
            keep = [r for r in prow if int(r[0]) not in win]
            for r in prow:
                if int(r[0]) in win:
                    wl.setdefault("peer_exchange", []).append([ids[int(gid)], ids[int(r[0])], float(r[2])])
            fid["peers"][gid] = keep

        local_of = {fr.batch_local[i]: i for i in range(len(fr.batch_local))}  # replay id -> dataset task index
        forced = dict(fr.queued_forced)
        plan_kind = job["plan"]
        if plan_kind == "live":
            batch_gids = [int(r["gid"]) for r in sorted(fid["batch"], key=lambda r: int(r["gid"]))]
            forced_names = {ids[g]: (tab["node"][g], int(tab["platform"][g])) for g in batch_gids}
        elif isinstance(plan_kind, list):
            for i, p in enumerate(plan_kind):
                forced[fr.batch_local[i]] = (int(p[0]), int(p[1]))
            forced_names = {}
        else:
            forced_names = {}
        allowed = {}
        for r in rows:
            for k, v in r["placement_plan"].items():
                allowed.setdefault(int(k), set()).add((int(v[0]), int(v[1])))
        seen = {"decisions": 0, "batch_plan": None, "batch_decided": False}

        orig_inf = PG.PeerGreedyNetworkBatchScheduler._prefix_inference
        orig_apply = snapshot_fidelity.apply_orchestrator

        def patched(self, batch_tasks, system_state, queue_snapshot, temporal_state):
            seen["decisions"] += 1
            node_by_id = {n.id: n for n in self.nodes.items}
            if forced_names:  # the live plan names nodes; resolved to ids on the first decision
                by_name = {n.node_name: n.id for n in self.nodes.items}
                for rid, (name, pid) in list(forced_names.items()):
                    forced[rid] = (int(by_name[name]), int(pid))
                forced_names.clear()
            out, free = {}, []
            for i, t in enumerate(batch_tasks):
                f = forced.get(int(t.id))
                if f is not None:
                    out[i] = (int(f[0]), int(f[1]))
                    t.planned_node_name = node_by_id[int(f[0])].node_name
                else:
                    free.append(i)
            if not out:
                restrict = _Restrict({int(t.id): allowed[local_of[int(t.id)]] for t in batch_tasks if int(t.id) in local_of})
                self._pg_allowed = restrict or None
                try:
                    res = orig_inf(self, batch_tasks, system_state, queue_snapshot, temporal_state)
                finally:
                    self._pg_allowed = None
            else:
                res = {}
                if free:
                    free_tasks = [batch_tasks[i] for i in free]
                    restrict = _Restrict({int(t.id): allowed[local_of[int(t.id)]] for t in free_tasks if int(t.id) in local_of})
                    self._pg_allowed = restrict or None
                    try:
                        sub = self._pg_decide(free_tasks, system_state)
                    finally:
                        self._pg_allowed = None
                    for k, i in enumerate(free):
                        res[i] = sub[k]
                res.update(out)
            for i, t in enumerate(batch_tasks):
                if int(t.id) in local_of and int(t.id) not in fr.queued_forced:
                    seen.setdefault("batch_rows", {})[local_of[int(t.id)]] = [int(res[i][0]), int(res[i][1])]
            return res

        def apply(orchestrator, fidelity):
            never = orig_apply(orchestrator, fidelity)
            if job.get("mutex_log"):
                _log_mutex(orchestrator, job["mutex_log"], 0.0, H + 1)
            env = orchestrator.env
            for rid, prow in stub_rows.items():
                for other, node_name, payload, delay in prow:
                    sid = snapshot_fidelity.STUB_BASE + int(other)
                    stub = orchestrator.task_by_id.get(sid)
                    if stub is None:
                        stub = snapshot_fidelity.PeerStub(env, int(other), node_name if delay is None else None)
                        orchestrator.task_by_id[sid] = stub
                        if delay is not None:
                            env.process(snapshot_fidelity._place_later(env, stub, max(0.0, float(delay)), node_name))
                    orchestrator.peer_exchange.setdefault(rid, {})[sid] = payload
            return never

        # the orchestrator's 1-s monitor ran on the live run's absolute grid (started at live t = 0); the KPA tick's env.step()
        # meets it at the same instant there, so the replay's monitor must keep that phase or the tick phase drifts
        from src.policy.gnn import orchestrator as GO

        orig_monitor = GO.GNNOrchestrator.monitor_process
        phase = (math.ceil(t0) - t0) % 1.0

        def monitor(self):
            if phase > 0:
                yield self.env.timeout(phase)
            yield from orig_monitor(self)

        PG.PeerGreedyNetworkBatchScheduler._prefix_inference = patched
        snapshot_fidelity.apply_orchestrator = apply
        GO.GNNOrchestrator.monitor_process = monitor
        try:
            infra = deepcopy(fr.base_infra)
            seed_live = deepcopy(fr.seed)
            seed_live["fidelity"] = fid
            infra["live_snapshot_seed"] = seed_live
            infra["fast_forward_warmup"] = True
            infra["fast_forward_threshold"] = 1
            res = execute_simulation({"infrastructure": infra, "workload": wl}, fr.sim_inputs, scheduling_strategy=strategy,
                                     cache_policy="fifo", task_priority="fifo", **fr.kw)
        finally:
            PG.PeerGreedyNetworkBatchScheduler._prefix_inference = orig_inf
            snapshot_fidelity.apply_orchestrator = orig_apply
            GO.GNNOrchestrator.monitor_process = orig_monitor
        trs = {tr["taskId"]: tr for tr in res["stats"]["taskResults"] if tr.get("taskId", -1) >= 0}
        lat = lambda rid: float(trs[rid]["doneTime"]) - float(trs[rid]["scheduledTime"])
        missing = [rid for rid in list(fr.batch_local) + [ids[g] for g in window] if rid not in trs]
        if missing:
            raise RuntimeError(f"rollout finished without results for {len(missing)} tasks, e.g. {missing[:5]}")
        off = [q for q, (n, p) in fr.queued_forced.items() if int(trs[q]["executionPlatform"]) != int(p)]
        if off:
            raise RuntimeError(f"queued tasks {off[:5]} did not run on their recorded replicas")
        q_batch = sum(lat(b) for b in fr.batch_local)
        q_win = sum(lat(ids[g]) for g in window)
        br = seen.get("batch_rows") or {}
        plan = [br.get(i) for i in range(len(fr.batch_local))]
        label = None
        if all(p is not None for p in plan):
            hit = [float(r["rtt"]) for r in rows if all(list(map(int, r["placement_plan"][str(i)])) == plan[i] for i in range(len(plan)))]
            label = min(hit) if hit else None
        sc = res["stats"].get("schedulerCounters") or {}
        return dict(base, t0=t0, n_cut=n_cut, n_window=len(window), n_window_pending=sum(1 for g in window if arrivals[g] <= t0),
                    partners_beyond_cut=beyond, n_siblings=len(siblings), pairs_touching_scored=pairs_touch, pairs_touching_scored_cut=pairs_cut, stubs=sum(len(v) for v in stub_rows.values()),
                    q=q_batch + q_win, q_batch=q_batch, q_window=q_win, plan=plan, label_of_plan=label,
                    window_lat={str(g): lat(ids[g]) for g in window}, batch_lat=[lat(b) for b in fr.batch_local],
                    batch_detail=[{k: trs[b].get(k) for k in DETAIL} for b in fr.batch_local],
                    window_detail={str(g): {k: trs[ids[g]].get(k) for k in DETAIL} for g in window} if job.get("detail") else None,
                    scale_events=[[round(e["timestamp"], 3), e.get("action"), e.get("name"), e.get("cause")] for e in res["stats"].get("scaleEvents", [])][:300] if job.get("detail") else None,
                    decisions=seen["decisions"], exact_batches=int(sc.get("pg_exact_batches") or 0),
                    exact_fallbacks=int(sc.get("pg_exact_fallbacks") or 0), wall=time.time() - t_start)
    except Exception as e:  # recorded, never swallowed
        import traceback
        return dict(base, error=f"{type(e).__name__}: {str(e)[:300]}", tb=traceback.format_exc()[-1500:])


def truth_one(job):
    """The self-check's truth: the live capture run cut at n(t0 + H) (HEROSIM_MAX_EVENTS), the environment inherited from the
    caller (the capture's). Its history before t0 must equal the table's, and its batch must be decided at t0 on the table's plan."""
    import subprocess
    import tempfile

    t_start = time.time()
    ds, H = Path(job["ds"]), float(job["H"])
    base = dict(ds=str(ds), H=H, tag=job.get("tag"), kind="truth")
    try:
        snap = json.load(open(ds / "infrastructure.json"))["live_snapshot_seed"]["fidelity_replay"]["snapshot"]
        fid, t0 = snap["fidelity"], float(snap["time"])
        arrivals = [float(e["timestamp"]) for e in json.load(open(job["workload"]))["events"]]
        tab = json.load(open(job["table"]))
        n_cut = bisect.bisect_right(arrivals, t0 + H)
        window, _ = window_of(fid, t0, H, arrivals, tab, None)
        batch = [int(r["gid"]) for r in fid["batch"]]
        with tempfile.TemporaryDirectory(dir=os.environ.get("HEROSIM_RAW_DIR")) as tmp:
            # the live run on the trace cut at n(t0 + H), every pair with an end below the cut KEPT: a partner beyond the cut is a stub
            # placed where and when the full run placed it, so batching waits and rendezvous happen as live and the history before
            # t0 is the full run's (checked below, not assumed)
            trace = json.load(open(job["workload"]))
            cut = {k: v for k, v in trace.items() if k not in ("events", "peer_exchange")}
            cut["events"] = trace["events"][:n_cut]
            cut["peer_exchange"] = [p for p in trace.get("peer_exchange") or [] if min(int(p[0]), int(p[1])) < n_cut]
            later = sorted({int(x) for p in cut["peer_exchange"] for x in p[:2] if int(x) >= n_cut})
            wl_cut = os.path.join(tmp, "wl.json")
            json.dump(cut, open(wl_cut, "w"))
            spec = os.path.join(tmp, "spec.json")
            json.dump({"mutex_log": job.get("mutex_log"), "t0": t0, "H": H, "stubs": {str(j): [tab["scheduled"][j], tab["node"][j]] for j in later},
                       "argv": ["src/executesimulation.py", "--config", job["cfg"], "--workload", wl_cut, "--policy",
                                "peer_greedy_network_batch", "--output", os.path.join(tmp, "cut.json")]}, open(spec, "w"))
            env = dict(os.environ, SIM_FORCE_FULL_STATS="1", LIVE_AUDIT_SNAPSHOT_PATH=os.path.join(tmp, "snap.jsonl"))
            env.pop("HEROSIM_MAX_EVENTS", None)
            out = os.path.join(tmp, "cut.json")
            proc = subprocess.run([sys.executable, str(REPO / "scripts_cosim/rollout_replay.py"), "--truth-run", spec],
                                  env=env, cwd=str(REPO), capture_output=True, text=True)
            if proc.returncode != 0 or not os.path.exists(out):
                raise RuntimeError(f"cut live run failed (rc {proc.returncode}): {proc.stderr[-300:]}")
            _st = json.load(open(out))["stats"]
            trs = {int(tr["taskId"]): tr for tr in _st["taskResults"] if tr.get("taskId", -1) >= 0}
            stats_scale = [e for e in _st.get("scaleEvents", []) if t0 - 30 <= e["timestamp"] <= t0 + H + 30]
        hist = [abs(float(trs[i]["scheduledTime"]) - tab["scheduled"][i]) for i in range(n_cut) if tab["scheduled"][i] < t0 - 1e-9]
        hist_off = [[i, tab["scheduled"][i], float(trs[i]["scheduledTime"]), float(trs[i]["doneTime"]) - tab["done"][i]]
                    for i in range(n_cut) if tab["scheduled"][i] < t0 - 1e-9 and abs(float(trs[i]["scheduledTime"]) - tab["scheduled"][i]) > 1e-9][:10]
        moved = [g for g in batch if (str(trs[g]["executionNode"]), int(trs[g]["executionPlatform"])) != (tab["node"][g], tab["platform"][g])
                 or abs(float(trs[g]["scheduledTime"]) - t0) > 1e-6]
        snapset = {int(r["gid"]) for r in fid["queued"]} | set(batch) | {int(g["tid"]) for g in fid.get("ghosts") or []}
        win_truth = [i for i in range(n_cut) if i not in snapset and float(trs[i]["scheduledTime"]) >= t0 - 1e-9]
        lat = lambda i: float(trs[i]["doneTime"]) - float(trs[i]["scheduledTime"])
        return dict(base, t0=t0, n_cut=n_cut, partners_beyond_cut=len(later), history_max_abs=max(hist) if hist else 0.0, history_off=hist_off, batch_moved=moved,
                    window_same=sorted(win_truth) == sorted(window), n_window=len(win_truth),
                    q=sum(lat(g) for g in batch) + sum(lat(i) for i in win_truth), q_batch=sum(lat(g) for g in batch),
                    q_window=sum(lat(i) for i in win_truth), batch_detail=[{k: trs[g].get(k) for k in DETAIL} for g in batch],
                    window_detail={str(i): {k: trs[i].get(k) for k in DETAIL} for i in win_truth} if job.get("detail") else None,
                    scale_events=[[round(e["timestamp"] - t0, 3), e.get("action"), e.get("name"), e.get("cause")] for e in stats_scale] if job.get("detail") else None, window_lat={str(i): lat(i) for i in win_truth}, wall=time.time() - t_start)
    except Exception as e:  # recorded, never swallowed
        return dict(base, error=f"{type(e).__name__}: {str(e)[:300]}")


def truth_run(spec_path: str) -> None:
    """Inside the truth subprocess: stub the partners beyond the cut on every orchestrator, then run the live driver."""
    import runpy

    from src.placement import orchestrator as O
    from src.placement import snapshot_fidelity

    spec = json.load(open(spec_path))
    stubs = {int(k): (float(v[0]), str(v[1])) for k, v in spec["stubs"].items()}
    orig = O.Orchestrator.__init__

    def init(self, *a, **kw):
        orig(self, *a, **kw)
        if spec.get("mutex_log"):
            _log_mutex(self, spec["mutex_log"], spec["t0"], spec["H"] + 1)
        for j, (when, node) in stubs.items():
            if j in self.task_by_id:
                raise RuntimeError(f"stub {j} collides with a real task")
            stub = snapshot_fidelity.PeerStub(self.env, j, None)
            stub.id = j
            self.task_by_id[j] = stub
            self.env.process(snapshot_fidelity._place_later(self.env, stub, when, node))

    O.Orchestrator.__init__ = init
    sys.argv = spec["argv"]
    runpy.run_path(str(REPO / "src/executesimulation.py"), run_name="__main__")


def _dispatch(job):
    return truth_one(job) if job.get("kind") == "truth" else _one(job)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", nargs="?"); ap.add_argument("jobs", nargs="?")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--make-table", nargs=2, metavar=("RAW", "TABLE"))
    ap.add_argument("--truth-run", metavar="SPEC")
    a = ap.parse_args()
    if a.truth_run:
        truth_run(a.truth_run)
        return 0
    if a.make_table:
        make_table(*a.make_table)
        return 0
    jobs = [json.loads(l) for l in open(a.jobs) if l.strip()]
    with get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool, open(a.out, "w") as fh:
        for r in pool.imap_unordered(_dispatch, jobs, chunksize=1):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
