#!/usr/bin/env python3
"""fresh_topo_burst_v1 -- local driver for the screen, the venue-parity check and the gate.
See docs/lineages/fresh_topo_burst_v1.md.

Every task runs `src/executesimulation.py` once, in its own memory-capped cgroup scope with a
wall-clock timeout (a hung run is a FAILED task, never a skipped one), and writes a summary in
`selfpredict_burst_v1_gate.sbatch`'s format. Existing summaries are not re-run.

  fresh_topo_burst_v1_gate.py screen --inputs DIR --out DIR
  fresh_topo_burst_v1_gate.py parity --inputs DIR --out DIR
  fresh_topo_burst_v1_gate.py gate   --inputs DIR --out DIR --selection selected.json

--inputs holds cfg/ (cc40s<topo>.json), wl/ (burst workloads), models/ (jb2 checkpoints + sidecars)
and joint_burst_v2_split.json, all md5-verified against datalab before use.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WINDOWS = ("w0", "w1", "w2", "w3")
SEEDS = (1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 14, 15, 16)
OLD_POOL = (9001, 9002, 9003, 9005, 9102, 9103, 9104, 9105, 9107, 9108, 9109, 9110, 9111, 9112, 9113,
            9115, 9117, 9118, 9119, 9120, 9121, 9122, 9123, 9124)
NEW_POOL = tuple(range(9401, 9425))
EXT_POOL = tuple(range(9425, 9473))  # amendment A2
CANDIDATES = OLD_POOL + NEW_POOL + EXT_POOL
RULE_POLICY = {
    "reactive": "knative_network",
    "batched": "peer_greedy_network_batch",
    "selfpredict": "peer_greedy_selfpredict_network",
    "cd": "peer_greedy_network_cd",
    "cd_blind": "peer_greedy_network_cd",  # cd_gap_v1 D1: HEROSIM_PG_BATCH_BLIND=1
    "cd_slate": "peer_greedy_network_cd",  # cd_gap_v1 D4: GNN_SERVE_CORPUS_SLATE=1
}
SUFFIXES = ("_spread", "_slate", "_cdshadow", "_cdapply", "_selfref", "_se")
GATE_RULES = ("selfpredict", "cd", "batched", "reactive")
V4_KINDS = ("v4load", "v4twin")  # load_repr_v1: partial_state_v4, load columns on / zeroed
# backlog_corpus_v1: v4load's recipe and its MP-OFF twin on the synthetic-backlog corpus, always served
# with the in-flight capture fix; a "_se" suffix serves any other arm with it (Amendment 1: v4load_se)
BC1_KINDS = ("bc1load", "bc1mpoff")
LOAD_KINDS = V4_KINDS + BC1_KINDS
LEARNED_KINDS = ("gnnedge0", "mpoff", "cdimit") + LOAD_KINDS
SERVICE_END = "service_end_v1"
N_TASKS = 50000
PY = shlex.split(os.environ.get("HEROSIM_PY", "pipenv run python3"))
# SLURM compute nodes have no systemd-run: the job's own memory allocation caps the runs instead.
NO_SCOPE = False


def task(topo: int, window: str, kind: str, seed: int = 0) -> Dict[str, object]:
    return {"topo": topo, "window": window, "kind": kind, "seed": seed}


def tasks_for(phase: str, selection: Optional[dict]) -> List[Dict[str, object]]:
    if phase == "screen":
        return [task(t, w, k) for k in ("reactive", "batched") for t in CANDIDATES for w in WINDOWS]
    if phase == "parity":
        return [task(9101, "w1", k) for k in ("selfpredict", "cd")] + \
               [task(9101, "w1", k, 1) for k in ("gnnedge0", "mpoff")]
    topos = selection["topologies"]
    if phase == "d1":
        return [task(t, w, "cd_blind") for t in topos for w in WINDOWS]
    if phase == "a":
        return [task(t, w, "cdimit", s) for s in (1, 2, 3, 4) for t in topos for w in WINDOWS]
    if phase == "v4":
        return [task(t, w, k, s) for k in V4_KINDS for s in (1, 2, 3, 4) for t in topos for w in WINDOWS]
    if phase == "bc1":
        return [task(t, w, k, s) for k in BC1_KINDS + ("v4load_se",) for s in (1, 2, 3, 4)
                for t in topos for w in WINDOWS]
    if phase == "fix":
        # load_repr_v1 Amendment 2: every learned arm again after the uncapped-rung rank fix (721d44f)
        return [task(t, w, k, s) for k in V4_KINDS for s in (1, 2, 3, 4) for t in topos for w in WINDOWS] + \
               [task(t, w, k, s) for k in ("gnnedge0", "mpoff") for s in SEEDS for t in topos for w in WINDOWS]
    if phase == "d6":
        return [task(t, w, "gnnedge0_selfref", s) for s in SEEDS for t in topos for w in WINDOWS]
    if phase == "d5":
        return [task(t, w, k, s) for k in ("gnnedge0_cdapply", "mpoff_cdapply") for s in SEEDS
                for t in topos for w in WINDOWS] + \
               [task(t, w, "gnnedge0_cdshadow", s) for s in (1, 2, 3) for t in topos for w in WINDOWS]
    if phase == "d4":
        return [task(t, w, "cd_slate") for t in topos for w in WINDOWS] + \
               [task(t, w, "gnnedge0_slate", s) for t in topos for w in WINDOWS for s in SEEDS]
    if phase == "d2":
        return [task(t, w, k, s) for t in topos for w in WINDOWS for k in ("gnnedge0_spread", "mpoff_spread")
                for s in SEEDS]
    out = [task(t, w, k) for t in topos for w in WINDOWS for k in GATE_RULES]
    out += [task(t, w, k, s) for t in topos for w in WINDOWS for k in ("gnnedge0", "mpoff") for s in SEEDS]
    return out


def arm_name(t: Dict[str, object]) -> str:
    return f"cc40s{t['topo']}__{t['window']}__{t['kind']}_s{t['seed']}"


def _reactive_disqualified(topo: int, out_dir: str) -> bool:
    """Screen shortcut that cannot change admission: a failed or saturated reactive window already
    rules the topology out, so its batch path is not run."""
    for w in WINDOWS:
        base = os.path.join(out_dir, f"cc40s{topo}__{w}__reactive_s0")
        if os.path.exists(base + ".failed.json"):
            return True
        if os.path.exists(base + ".summary.json"):
            share = json.load(open(base + ".summary.json")).get("queue_share")
            if share is None or float(share) > 0.80:
                return True
    return False


def run_one(t: Dict[str, object], inputs: str, out_dir: str, mem: str, timeout_s: int) -> str:
    name = arm_name(t)
    summary = os.path.join(out_dir, name + ".summary.json")
    failed = os.path.join(out_dir, name + ".failed.json")
    if os.path.exists(summary) or os.path.exists(failed):
        return f"[skip] {name}"
    kind, seed, window = str(t["kind"]), int(t["seed"]), str(t["window"])
    if kind == "batched" and _reactive_disqualified(int(t["topo"]), out_dir):
        return f"[skip, reactive already disqualifies] {name}"
    wl_name = "burst_drainable_f4000_n50000" if window == "w0" else f"burst_drainable_{window}_n50000"
    cfg = os.path.join(inputs, "cfg", f"cc40s{t['topo']}.json")
    wl = os.path.join(inputs, "wl", wl_name + ".json")
    env = dict(os.environ)
    for k in ("GNN_MODEL_PATH", "GNN_DISABLE_MESSAGE_PASSING", "GNN_MP_PLATFORM_EDGES_OFF", "GNN_BATCH_TIMEOUT",
              "GNN_BATCH_SIZE", "GNN_DECODE_MODE", "GNN_BATCH_BY_PEER_GROUP", "GNN_PREFIX_ALPHA_KEY",
              "LIVE_AUDIT_SNAPSHOT_PATH", "HEROSIM_ROLLOUT_SCORER", "HEROSIM_PG_EXCHANGE_SCALE",
              "HEROSIM_PG_ORACLE_NODES", "HEROSIM_PG_CD_PASSES", "HEROSIM_MAX_EVENTS", "HEROSIM_FORCED_PLACEMENTS",
              "HEROSIM_EXEC_PHYSICS", "HEROSIM_EXEC_SEED", "HEROSIM_PG_EXEC_KNOWLEDGE", "HEROSIM_PG_BATCH_BLIND",
              "GNN_PREFIX_SIBLING_SPREAD", "GNN_SERVE_CORPUS_SLATE", "NEAR_RTT_LABEL_OVERRIDE_JSON", "GNN_CD_REFINE",
              "GNN_PREFIX_SELF_REFINE", "HEROSIM_POLICY_TIME_SCALE", "PARTIAL_STATE_CONTRACT",
              "PARTIAL_STATE_LOAD_SECONDS", "PARTIAL_STATE_PEER_MASS", "HEROSIM_INFLIGHT_CAPTURE"):
        env.pop(k, None)
    # cd_gap_v1 B': a rate-stretched cell scales keep_alive and the reconcile interval by its own factor
    time_scale = float((json.load(open(cfg)).get("cd_gap_v1_rate_scale") or {}).get("factor", 1.0))
    if time_scale != 1.0:
        env["HEROSIM_POLICY_TIME_SCALE"] = repr(time_scale)
    env.update(HEROSIM_PEER_EXCHANGE="1", HEROSIM_SERVER_ONLY_REPLICAS="1", HEROSIM_WARMTH_PHYSICS="node_disk_v2",
               PYTHONHASHSEED="0", HEROSIM_GNN_DEVICE="cpu", SIM_FORCE_FULL_STATS="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="", PYTHONPATH=REPO)
    base_kind = next((kind[:-len(s)] for s in SUFFIXES if kind.endswith(s)), kind)
    if base_kind in LEARNED_KINDS:
        # cd_gap_v1 A: the CD imitator is the gnnedge0 architecture on the jb2 corpus and split
        if base_kind == "cdimit":
            stem = "cd-gap-v1-cdimit-gnnedge0"
        elif base_kind in V4_KINDS:
            stem = f"load-repr-v1-{base_kind}-gnnedge0"
        elif base_kind in BC1_KINDS:
            stem = f"backlog-corpus-v1-{base_kind}"
        else:
            stem = f"joint-burst-v2-{base_kind}"
        ck = os.path.join(inputs, "models", f"{stem}-lr2e3-seed{seed}.pt")
        side = ck[:-3] + ".contract.json"
        check_kind = "gnnedge0" if base_kind == "cdimit" else base_kind
        split = "backlog_corpus_v1_split.json" if base_kind in BC1_KINDS else "joint_burst_v2_split.json"
        rc = subprocess.run(PY + [os.path.join(REPO, "scripts_cosim/joint_burst_v2_sidecheck.py"), side, check_kind,
                                  os.path.join(inputs, split), "inf"], env=env, cwd=REPO)
        if rc.returncode != 0:
            raise SystemExit(f"FAIL LOUD: sidecheck failed for {ck}")
        env.update(GNN_MODEL_PATH=ck, GNN_DECODE_MODE="masked_topo", GNN_BATCH_BY_PEER_GROUP="1",
                   GNN_PREFIX_ALPHA_KEY="inf")
        if base_kind in ("mpoff", "bc1mpoff"):
            env["GNN_DISABLE_MESSAGE_PASSING"] = "1"
        if base_kind in LOAD_KINDS:
            # exported, not adopted, so run_provenance records them; the loader verifies the sidecar
            env.update(PARTIAL_STATE_CONTRACT="partial_state_v4",
                       PARTIAL_STATE_LOAD_SECONDS="0" if base_kind == "v4twin" else "1")
        if base_kind in BC1_KINDS or kind.endswith("_se"):
            env["HEROSIM_INFLIGHT_CAPTURE"] = SERVICE_END
        if kind.endswith("_spread"):
            env["GNN_PREFIX_SIBLING_SPREAD"] = "1"
        if kind.endswith("_slate"):
            env["GNN_SERVE_CORPUS_SLATE"] = "1"
        if kind.endswith("_cdshadow"):
            env["GNN_CD_REFINE"] = "shadow"
        if kind.endswith("_cdapply"):
            env["GNN_CD_REFINE"] = "apply"
        if kind.endswith("_selfref"):
            env["GNN_PREFIX_SELF_REFINE"] = "3"
        policy = "gnn"
    else:
        policy = RULE_POLICY[kind]
        if kind in ("batched", "cd", "cd_blind"):
            env.update(GNN_DECODE_MODE="masked_topo", GNN_BATCH_BY_PEER_GROUP="1")
        if kind == "cd_blind":
            env["HEROSIM_PG_BATCH_BLIND"] = "1"
        if kind == "cd_slate":
            env.update(GNN_DECODE_MODE="masked_topo", GNN_BATCH_BY_PEER_GROUP="1", GNN_SERVE_CORPUS_SLATE="1")
    raw = os.path.join(out_dir, name + ".raw.json")
    log = os.path.join(out_dir, name + ".log")
    scope = [] if NO_SCOPE else ["systemd-run", "--scope", "-q", "-p", f"MemoryMax={mem}", "-p", "MemorySwapMax=0"]
    cmd = scope + ["timeout", str(timeout_s)] + PY + [os.path.join(REPO, "src/executesimulation.py"), "--config", cfg,
                                                      "--workload", wl, "--policy", policy, "--output", raw]
    start = time.time()
    with open(log, "w") as fh:
        rc = subprocess.run(cmd, env=env, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode
    wall = int(time.time() - start)
    if rc != 0 or not os.path.exists(raw):
        json.dump({"arm": name, "returncode": rc, "wallclock_s": wall,
                   "why": "timeout" if rc == 124 else ("memory cap or crash" if rc != 0 else "no output")},
                  open(failed, "w"))
        if os.path.exists(raw):
            os.remove(raw)
        return f"[FAILED rc={rc} {wall}s] {name}"
    doc = json.load(open(raw))
    st = doc.get("stats") or doc
    out = {k: st.get(k) for k in ("num_tasks", "total_rtt", "averageElapsedTime", "averageQueueTime",
                                  "averageWaitTime", "totalPeerExchangeTime", "totalPeerRendezvousWait", "endTime",
                                  "schedulerCounters")}
    c = out.get("schedulerCounters") or {}
    for k in ("residence_tasks", "residence_batches", "queue_range_records"):
        c.pop(k, None)
    arm_kind = RULE_POLICY.get(kind, kind)
    out.update(arm=name, cell=f"cc40s{t['topo']}", topology=int(t["topo"]), window=window, clients=40, servers=6,
               rung="C40", lever="burst", workload=wl_name, corpus=("jb2-cdlabel" if base_kind == "cdimit" else "bc1" if base_kind in BC1_KINDS else "jb2") if base_kind in LEARNED_KINDS else "none",
               arm_kind=arm_kind, checkpoint_seed=seed, policy_name=policy, wallclock_s=wall)
    out["env"] = {k: v for k, v in (doc.get("run_provenance") or {}).get("env", {}).items() if v}
    out["code"] = (doc.get("run_provenance") or {}).get("code")
    n = out.get("num_tasks")
    problems = []
    if time_scale != 1.0 and out["env"].get("HEROSIM_POLICY_TIME_SCALE") != repr(time_scale):
        problems.append(f"policy time scale not recorded: {out['env'].get('HEROSIM_POLICY_TIME_SCALE')!r}")
    if n is None or int(n) != N_TASKS:
        problems.append(f"num_tasks={n!r}")
    if kind == "selfpredict" and (int(c.get("pg_decisions") or 0) != N_TASKS or int(c.get("pg_lookahead_priced") or 0) == 0):
        problems.append(f"rule instrument off: {c.get('pg_decisions')}/{c.get('pg_lookahead_priced')}")
    if kind.endswith("_selfref") and int(c.get("prefix_self_refine_batches") or 0) == 0:
        problems.append("self-refine instrument off: prefix_self_refine_batches == 0")
    if not kind.endswith("_selfref") and int(c.get("prefix_self_refine_batches") or 0):
        problems.append("unrefined arm self-refined")
    if kind.endswith(("_cdshadow", "_cdapply")) and int(c.get("cdr_batches") or 0) == 0:
        problems.append("cd-refine instrument off: cdr_batches == 0")
    if not kind.endswith(("_cdshadow", "_cdapply")) and int(c.get("cdr_batches") or 0):
        problems.append("unrefined arm was refined")
    if kind.endswith("_slate") and int(c.get("slate_batches") or 0) == 0:
        problems.append("slate instrument off: slate_batches == 0")
    if not kind.endswith("_slate") and int(c.get("slate_batches") or 0):
        problems.append("unslated arm was slated")
    if kind in ("batched", "cd", "cd_blind", "cd_slate") and int(c.get("pg_batches") or 0) == 0:
        problems.append("decoded no batches")
    if kind == "cd_blind" and int(c.get("pg_partners_blinded") or 0) == 0:
        problems.append("blind instrument off: pg_partners_blinded == 0")
    if kind == "cd" and int(c.get("pg_partners_blinded") or 0) != 0:
        problems.append("CD ran blind")
    if kind.endswith("_spread") and int(c.get("prefix_sibling_moves") or 0) == 0:
        problems.append("spread instrument off: prefix_sibling_moves == 0")
    want_capture = SERVICE_END if (base_kind in BC1_KINDS or kind.endswith("_se")) else None
    if out["env"].get("HEROSIM_INFLIGHT_CAPTURE") != want_capture:
        problems.append(f"served HEROSIM_INFLIGHT_CAPTURE={out['env'].get('HEROSIM_INFLIGHT_CAPTURE')!r}, "
                        f"{kind} needs {want_capture!r}")
    if base_kind in LOAD_KINDS:
        want_ls = "0" if base_kind == "v4twin" else "1"
        if out["env"].get("PARTIAL_STATE_LOAD_SECONDS", "") != want_ls:
            problems.append(f"served PARTIAL_STATE_LOAD_SECONDS={out['env'].get('PARTIAL_STATE_LOAD_SECONDS')!r}, "
                            f"{base_kind} needs {want_ls}")
        if int(c.get("v4_backlog_batches") or 0) == 0:
            problems.append("v4 instrument off: v4_backlog_batches == 0")
        if base_kind != "v4twin" and int(c.get("v4_backlog_nonzero") or 0) == 0:
            problems.append("v4 instrument off: no candidate ever had a backlog")
    elif int(c.get("v4_backlog_batches") or 0):
        problems.append("a pre-v4 arm computed v4 backlogs")
    if base_kind in LEARNED_KINDS and not kind.endswith("_spread") and int(c.get("prefix_sibling_moves") or 0):
        problems.append("unspread arm moved siblings")
    if problems:
        json.dump({"arm": name, "why": "; ".join(problems), "wallclock_s": wall}, open(failed, "w"))
        os.remove(raw)
        return f"[FAILED {problems}] {name}"
    e, q = float(out["averageElapsedTime"]), float(out["averageQueueTime"])
    out["queue_share"] = q / e if e > 0 else None
    json.dump(out, open(summary + ".partial", "w"), indent=1)
    os.replace(summary + ".partial", summary)
    os.remove(raw)
    return f"[done {wall}s] {name} elapsed={e:.3f} share={out['queue_share']:.3f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=("screen", "parity", "gate", "d1", "d2", "d4", "d5", "d6", "a", "v4", "fix", "bc1"))
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--selection", default=None)
    ap.add_argument("--parallel", type=int, default=12)
    ap.add_argument("--mem", default="4G")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--no-scope", action="store_true", help="no per-run systemd scope (SLURM nodes)")
    a = ap.parse_args()
    global NO_SCOPE
    NO_SCOPE = a.no_scope
    selection = None
    if a.phase in ("gate", "d1", "d2", "d4", "d5", "d6", "a", "v4", "fix", "bc1"):
        selection = json.load(open(a.selection))
        if selection.get("verdict") != "DESIGN-READY":
            raise SystemExit(f"FAIL LOUD: selection verdict {selection.get('verdict')!r}")
    os.makedirs(a.out, exist_ok=True)
    todo = tasks_for(a.phase, selection)
    print(f"[{a.phase}] {len(todo)} tasks, {a.parallel} parallel, MemoryMax={a.mem}, timeout {a.timeout}s", flush=True)
    with ThreadPoolExecutor(max_workers=a.parallel) as pool:
        for line in pool.map(lambda t: run_one(t, a.inputs, a.out, a.mem, a.timeout), todo):
            print(line, flush=True)
    done = sum(1 for t in todo if os.path.exists(os.path.join(a.out, arm_name(t) + ".summary.json")))
    print(f"=== {a.phase}: {done}/{len(todo)} summaries ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
