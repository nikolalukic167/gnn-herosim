#!/usr/bin/env python3
"""Offline screen of simulator-checked labels against the current formula labels (docs/lineages/uncut_groups_s0_v1.md, Record 2026-10-17
"Simulator-label screen: REGISTERED"). No training. MYOPIC-REPLAY, INTERIM-DESCRIPTIVE.

  uncut_label_screen.py draw   --split experiments/uncut_arm1_split.json --corpus /share/nikola.lukic/uncut_arm1/corpus --out DRAW.json
  uncut_label_screen.py replay --draw DRAW.json --corpus CORPUS --out OUT.json [--shard i/n] [--limit-per-bucket K]    replay the registered plans of each state
  uncut_label_screen.py read   --in OUT.json [OUT2.json ...] [--out READ.json]                                         per-bucket metrics and the decision rule

`draw` reads only dataset headers (n_tasks, plan_space_size, label_source, state_id); it reads no replayed RTT and no label outcome.
It takes 125 states per group-size bucket (<=10, 11-50, 51-100, 101-256 tasks) from the TRAIN key of the Arm 1 split, seed 20261017, or all of a bucket
that has fewer, and fails loud if any drawn state sits on a gate topology (19537-19548), a TE3 id (19554-19589) or an accel_replica_v1 test id.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

SEED = 20261017
PER_BUCKET = 125
BUCKETS = (("le10", 1, 10), ("11_50", 11, 50), ("51_100", 51, 100), ("101_256", 101, 256))
FORBIDDEN_TOPOLOGIES = set(range(19537, 19549)) | set(range(19554, 19590))
SMALL_PLAN_SPACE = 600


def bucket_of(n: int) -> str:
    for name, lo, hi in BUCKETS:
        if lo <= n <= hi:
            return name
    raise ValueError(f"FAIL LOUD: group of {n} tasks is outside the registered buckets 1-256")


def topology_of(state_id: str) -> int:
    m = re.search(r"cc40s(\d+)__", state_id)
    if not m:
        raise ValueError(f"FAIL LOUD: cannot read a topology from state_id {state_id!r}")
    return int(m.group(1))


def draw(frame: Dict[str, Dict[str, Any]], accel_test_ids: Sequence[str] = (), seed: int = SEED, per_bucket: int = PER_BUCKET) -> Dict[str, Any]:
    """frame: dataset id -> {"n_tasks", "state_id", "plan_space_size", "label_source"} for the TRAIN key only. Deterministic in (frame, seed)."""
    rng = random.Random(seed)
    by_bucket: Dict[str, List[str]] = {name: [] for name, _, _ in BUCKETS}
    for gid in sorted(frame):
        by_bucket[bucket_of(int(frame[gid]["n_tasks"]))].append(gid)
    picked: List[str] = []
    counts: Dict[str, Dict[str, int]] = {}
    for name, _, _ in BUCKETS:
        pool = by_bucket[name]
        take = sorted(rng.sample(pool, per_bucket)) if len(pool) > per_bucket else list(pool)
        counts[name] = {"frame": len(pool), "drawn": len(take)}
        picked += take
    bad = [g for g in picked if topology_of(frame[g]["state_id"]) in FORBIDDEN_TOPOLOGIES]
    if bad:
        raise SystemExit(f"FAIL LOUD: drawn states on forbidden topologies (gate 19537-19548 or TE3 19554-19589): {bad[:5]}")
    clash = sorted(set(picked) & set(accel_test_ids))
    if clash:
        raise SystemExit(f"FAIL LOUD: drawn states that are accel_replica_v1 test ids: {clash[:5]}")
    states = [{"gid": g, "bucket": bucket_of(int(frame[g]["n_tasks"])), "n_tasks": int(frame[g]["n_tasks"]), "topology": topology_of(frame[g]["state_id"]),
               "plan_space_size": frame[g]["plan_space_size"], "label_source": frame[g]["label_source"], "state_id": frame[g]["state_id"],
               "small": frame[g]["plan_space_size"] is not None and int(frame[g]["plan_space_size"]) <= SMALL_PLAN_SPACE} for g in picked]
    sha = hashlib.sha256("\n".join(picked).encode()).hexdigest()
    return {"seed": seed, "per_bucket": per_bucket, "counts": counts, "n": len(states), "state_list_sha256": sha, "states": states}


def load_frame(split_path: str, corpus: str) -> Dict[str, Dict[str, Any]]:
    train = json.load(open(split_path))["train"]
    frame = {}
    for gid in train:
        kind, ds = gid.split("/")
        sub = kind.replace("gnn_datasets_wf1_", "")
        h = json.loads(open(f"{corpus}/{sub}/{ds}/placements/placements.jsonl").readline())
        if not h.get("header"):
            raise SystemExit(f"FAIL LOUD: {gid}: first line of placements.jsonl is not the header")
        frame[gid] = {k: h.get(k) for k in ("n_tasks", "plan_space_size", "label_source", "state_id")}
    return frame


# ---- plan lists ------------------------------------------------------------------------------------------------------------------------------
TOP_K = 8
REL_TOL = 1e-9
SIM_INPUT = "/home/nikola.lukic/gnn-herosim/data/nofs-ids"
ENUM_CAP = 100000


def plan_key(plan: Sequence[Sequence[int]]):
    return tuple((int(a), int(b)) for a, b in plan)


def ds_order_from_gid_order(plan: Sequence[Sequence[int]], trace_ids: Sequence[int], gids_sorted: Sequence[int]) -> List[List[int]]:
    """ils_evaluated.jsonl.gz keeps plans in snapshot gid order; the engine's batch order for a dataset job is the dataset order, where task i has gid trace_ids[i]."""
    pos = {int(g): k for k, g in enumerate(gids_sorted)}
    return [[int(x) for x in plan[pos[int(g)]]] for g in trace_ids]


def top_k_distinct(rows: Sequence[Dict[str, Any]], k: int = TOP_K) -> List[Dict[str, Any]]:
    """The k best distinct plans by S (ties by evaluation index); `rows` carry S, eval, plan."""
    seen, out = set(), []
    for r in sorted(rows, key=lambda r: (float(r["S"]), int(r["eval"]))):
        key = plan_key(r["plan"])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
        if len(out) == k:
            break
    return out


def plan_list(label: Sequence[Sequence[int]], small: bool, enum: Sequence[Dict[str, Any]] = (), ils_rows: Sequence[Dict[str, Any]] = ()) -> List[Dict[str, Any]]:
    """The registered plans of one state, in dataset order, each tagged; element 0 is always the current label.
    small (plan space <= 600): every plan of the product (`enum` = all of them); large: the TOP_K best by S (`enum` for the exact-labelled states without a trajectory,
    `ils_rows` for the others) plus the current label."""
    out, seen = [{"plan": [list(map(int, x)) for x in label], "tag": "label"}], {plan_key(label)}
    if small:
        cand = [{"plan": e["plan"], "tag": "enum"} for e in enum]
    elif ils_rows:
        cand = [{"plan": r["plan"], "tag": "ils_top"} for r in top_k_distinct(ils_rows)]
    else:
        cand = [{"plan": e["plan"], "tag": "enum_top"} for e in sorted(enum, key=lambda e: (float(e["S"]), int(e["k"])))[:TOP_K]]
    for c in cand:
        key = plan_key(c["plan"])
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def decide(rtts: Sequence[float]) -> Dict[str, Any]:
    """rtts[0] is the current label's replayed RTT. The simulator label is the lowest; it differs only if it is lower by more than a relative REL_TOL."""
    best = min(range(len(rtts)), key=lambda i: (rtts[i], i))
    differs = rtts[best] < rtts[0] * (1.0 - REL_TOL)
    return {"best_index": best if differs else 0, "differs": bool(differs), "gain": (rtts[0] - rtts[best]) / rtts[0] if differs else 0.0}


# ---- the reader -----------------------------------------------------------------------------------------------------------------------------
def bootstrap_mean_ci(values: Sequence[float], seed: int = SEED, n_boot: int = 10000):
    import numpy as np

    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def read_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """rows: one per replayed state with bucket, differs, gain. Per bucket: (i) share differing, (ii) mean/median gain with a bootstrap interval, (iii) share gain >= 5 %."""
    import statistics

    res: Dict[str, Any] = {}
    for name, _, _ in BUCKETS:
        rs = [r for r in rows if r["bucket"] == name]
        if not rs:
            continue
        gains = [float(r["gain"]) for r in rs]
        lo, hi = bootstrap_mean_ci(gains)
        mean = sum(gains) / len(gains)
        i_share = sum(1 for r in rs if r["differs"]) / len(rs)
        res[name] = {"n": len(rs), "share_differs": i_share, "mean_gain": mean, "median_gain": statistics.median(gains), "mean_gain_ci95": [lo, hi],
                     "share_gain_ge_5pct": sum(1 for g in gains if g >= 0.05) / len(gains),
                     "meets_rule": bool(i_share >= 0.20 and mean >= 0.02 and lo > 0.0)}
    res["verdict"] = "PROCEED-TO-TRAIN" if any(v["meets_rule"] for k, v in res.items() if k != "verdict") else "LABELS-AGREE"
    return res


# ---- the replay driver (compute node; needs the corpus, the engine and the gnn environment) --------------------------------------------------
def _stored_rows(ds: str):
    rows = [json.loads(line) for line in open(f"{ds}/placements/placements.jsonl")]
    return rows[0], [r for r in rows[1:] if r.get("rtt") is not None]


def _plan_from_row(row: Dict[str, Any], n: int) -> List[List[int]]:
    return [[int(v) for v in row["placement_plan"][str(i)]] for i in range(n)]


def _dataset_dir(corpus: str, gid: str) -> str:
    kind, ds = gid.split("/")
    return f"{corpus}/{kind.replace('gnn_datasets_wf1_', '')}/{ds}"


def replay_state(corpus: str, s: Dict[str, Any]) -> Dict[str, Any]:
    import gzip

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import uncut_ils as U

    ds = _dataset_dir(corpus, s["gid"])
    head, rows = _stored_rows(ds)
    n = int(head["n_tasks"])
    trace, gids = [int(x) for x in head["trace_task_ids"]], [int(x) for x in head["snapshot_gids_sorted"]]
    label, stored_rtt = _plan_from_row(rows[0], n), float(rows[0]["rtt"])
    small = bool(s["small"])
    ils_rows = []
    if head.get("evaluated_plans_file"):
        ils_rows = [{"S": r["S"], "eval": r["eval"], "plan": ds_order_from_gid_order(r["plan"], trace, gids)}
                    for r in (json.loads(line) for line in gzip.open(f"{ds}/{head['evaluated_plans_file']}", "rt"))]
        best = top_k_distinct(ils_rows, 1)[0]
        if head["label_source"] == "ils" and plan_key(best["plan"]) != plan_key(label):
            raise SystemExit(f"FAIL LOUD: {s['gid']}: the lowest-S stored ILS plan is not the stored label plan (plan order or trace mapping is wrong)")
    t0 = time.time()
    job = {"dataset": ds, "sim_input": SIM_INPUT}
    if small or not ils_rows:
        job["enum_cap"] = ENUM_CAP
    d = U.decision_pass(job, "cd_ib0", 0, 1, 1, None, check_arm=False)
    if head["label_source"] == "hand" and plan_key(d["hand"]) != plan_key(label):
        raise SystemExit(f"FAIL LOUD: {s['gid']}: the engine's hand plan is not the stored (hand) label plan")
    enum = d["enum"] or []
    if (small or not ils_rows) and not enum:
        raise SystemExit(f"FAIL LOUD: {s['gid']}: no enumeration returned for a plan space of {head['plan_space_size']}")
    if small and len(enum) != int(head["plan_space_size"]):
        raise SystemExit(f"FAIL LOUD: {s['gid']}: enumerated {len(enum)} plans, header plan space is {head['plan_space_size']}")
    plans = plan_list(label, small, enum, ils_rows)
    rt, setup, runs = U.replay_rtts(d["spec"], d["types"], [p["plan"] for p in plans])
    if abs(rt[0] - stored_rtt) > REL_TOL * abs(stored_rtt):
        raise SystemExit(f"FAIL LOUD: {s['gid']}: the current label's replayed RTT {rt[0]!r} is not the stored {stored_rtt!r}")
    dec = decide(rt)
    return {"gid": s["gid"], "bucket": s["bucket"], "n_tasks": n, "small": small, "label_source": head["label_source"], "n_plans": len(plans),
            "rtt_label": rt[0], "stored_label_rtt": stored_rtt, "rtt_sim": rt[dec["best_index"]], "best_tag": plans[dec["best_index"]]["tag"], **dec,
            "tags": [p["tag"] for p in plans], "rtts": rt, "plans": [p["plan"] for p in plans] if not small else None,
            "replay_setup_s": setup, "wall_s": time.time() - t0}


def cmd_replay(a):
    draw = json.load(open(a.draw))
    states = draw["states"]
    if a.limit_per_bucket:
        kept, seen = [], {}
        for st in states:
            seen[st["bucket"]] = seen.get(st["bucket"], 0) + 1
            if seen[st["bucket"]] <= a.limit_per_bucket:
                kept.append(st)
        states = kept
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        states = states[i::n]
    out = []
    for st in states:
        r = replay_state(a.corpus, st)
        out.append(r)
        print(f"[{len(out)}/{len(states)}] {r['gid']} {r['bucket']} plans={r['n_plans']} wall={r['wall_s']:.0f}s differs={r['differs']} gain={r['gain']:.4f}", flush=True)
        json.dump({"state_list_sha256": draw["state_list_sha256"], "rows": out}, open(a.out, "w"))
    print(f"replay: {len(out)} states, {sum(r['wall_s'] for r in out):.0f} s")


def cmd_read(a):
    rows, shas = [], set()
    for f in a.inp:
        d = json.load(open(f))
        shas.add(d["state_list_sha256"])
        rows += d["rows"]
    if len(shas) != 1:
        raise SystemExit(f"FAIL LOUD: inputs come from different draws {sorted(shas)}")
    ids = [r["gid"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("FAIL LOUD: a state appears twice in the inputs")
    res = {"state_list_sha256": shas.pop(), "n_states": len(rows), "buckets": read_rows(rows)}
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draw")
    d.add_argument("--split", required=True)
    d.add_argument("--corpus", required=True)
    d.add_argument("--accel-split", default=str(Path(__file__).resolve().parents[1] / "experiments" / "accel_replica_v1_split.json"))
    d.add_argument("--out", required=True)
    r = sub.add_parser("replay")
    r.add_argument("--draw", required=True)
    r.add_argument("--corpus", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--shard", default="")
    r.add_argument("--limit-per-bucket", type=int, default=0)
    q = sub.add_parser("read")
    q.add_argument("--in", dest="inp", nargs="+", required=True)
    q.add_argument("--out", default="")
    a = ap.parse_args(argv)
    if a.cmd == "replay":
        return cmd_replay(a) or 0
    if a.cmd == "read":
        return cmd_read(a) or 0
    accel = json.load(open(a.accel_split))["test"]
    res = draw(load_frame(a.split, a.corpus), accel_test_ids=accel)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("seed", "n", "counts", "state_list_sha256")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
