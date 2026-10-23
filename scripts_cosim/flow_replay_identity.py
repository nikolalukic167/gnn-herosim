#!/usr/bin/env python3
"""Served-plan replay vs live (check 3 of live_link_features_v1), for the stored-flow replay.

For each captured decision of a live cell (a `<stem>.snapshots.jsonl` written under HEROSIM_SNAPSHOT_FIDELITY=1 + HEROSIM_EXCHANGE_FAIRSHARE=1) whose served plan
placed the whole batch (no deferral): replay the snapshot with the SERVED plan (`<stem>.snapshots.jsonl.chosen.jsonl`, batch order = gid ascending) and compare the
replayed batch RTT, sum of (doneTime - scheduledTime) as `plan_oracle._label_chunk` scores it, with the live one, the sum over the batch of (completion time - the
snapshot instant) from `<stem>.tasks.npy` (column 1 is the completion time; the batch is scheduled at the snapshot instant). Each state is replayed twice, with
HEROSIM_REPLAY_STORED_FLOWS unset (the replay as it was) and set (the stored flows replayed), so the report shows what the switch changes.

  flow_replay_identity.py run  --cell-dir D --stem S --cell-config C --sim-input I --n 240 --workers 16 --out rows.jsonl
  flow_replay_identity.py read --rows rows.jsonl [--out read.json]

Per row: the live RTT, both replayed RTTs, the signed relative gaps, and the ghost classes of the state (stored exchange records by planned flag, `unplanned`, rendezvous
ghosts, active flows). When the cell also logged flow starts (S6's `<stem>.flowstarts.jsonl`, optional), `later_flow_on_batch_links` marks states where a flow that
began after the capture used a link the batch's served plan uses.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def pick_states(chosen_path: str, n: int) -> List[Dict[str, Any]]:
    """Decisions that placed their whole batch, spread evenly in time order down to `n`."""
    recs = []
    for line in open(chosen_path):
        r = json.loads(line)
        if r["chosen"] and not r.get("deferred"):
            recs.append(r)
    if len(recs) <= n:
        return recs
    step = len(recs) / n
    return [recs[int(i * step)] for i in range(n)]


def pick_ids(chosen_path: str, ids: List[int]) -> List[Dict[str, Any]]:
    """The listed snapshot ids, each of which must be a decision that placed its whole batch."""
    want = {int(i) for i in ids}
    recs = [json.loads(l) for l in open(chosen_path)]
    out = [r for r in recs if int(r["snapshot_id"]) in want and r["chosen"] and not r.get("deferred")]
    if len(out) != len(want):
        raise SystemExit(f"FAIL LOUD: {len(want) - len(out)} of the listed snapshots are not whole-batch decisions of {chosen_path}")
    return out


def classes(snap: Dict[str, Any]) -> Dict[str, Any]:
    fid = snap["fidelity"]
    flows = fid.get("flows") or {}
    ex = flows.get("exchanges") or []
    ghosts = fid.get("ghosts") or []
    return {"records_input_io": sum(not e["planned"] for e in ex), "records_planned": sum(bool(e["planned"]) for e in ex),
            "unplanned": len(flows.get("unplanned") or []), "rendezvous_ghosts": sum(g["stage"] == "rendezvous" for g in ghosts),
            "active_flows": len(flows.get("active") or []), "ghosts": len(ghosts), "batch": len(fid["batch"])}


_FLOWS: Dict[str, List[Any]] = {}


def later_flow_join(flow_log: str, snap: Dict[str, Any], gids: List[int], t_end: float) -> Dict[str, Any]:
    """Which flows begin after the capture, inside the batch's live window (t, t_end], from a task the snapshot does not represent (not in the batch and with no
    stored exchange record: the replay cannot have them), and whether any of them uses a directed link of the batch's own transfers (`on_batch`) or of the batch's
    or the stored active flows (`on_batch_or_stored`)."""
    if flow_log not in _FLOWS:
        _FLOWS[flow_log] = [json.loads(l) for l in open(flow_log) if l.strip()]
    flows = _FLOWS[flow_log]
    t = float(snap["time"])
    fl = snap["fidelity"].get("flows") or {}
    ghost_owners = {int(e["owner_tid"]) for e in fl.get("exchanges") or []}
    batch = set(gids)
    batch_links = {k for ts, o, i, hops, nb in flows if ts >= t and int(o) in batch for k in hops}
    stored_links = {h[0] for e in fl.get("exchanges") or [] for tr in e["transfers"] if tr["status"] == "active" for h in tr["hops"]}
    later = [(ts, o, hops, nb) for ts, o, i, hops, nb in flows if t < ts <= t_end and int(o) not in batch and int(o) not in ghost_owners]
    on_batch = [x for x in later if batch_links & set(x[2])]
    on_any = [x for x in later if (batch_links | stored_links) & set(x[2])]
    return {"later_flows_in_window": len(later), "later_flows_on_batch_links": len(on_batch), "later_flows_on_batch_or_stored_links": len(on_any),
            "later_bytes_on_batch_links": float(sum(x[3] for x in on_batch)), "later_flow_on_batch_links": bool(on_batch),
            "later_flow_on_batch_or_stored_links": bool(on_any), "window_end": t_end}


def _work(job: Dict[str, Any]) -> Dict[str, Any]:
    import numpy as np

    import uncut_ils as U
    from src.placement import flow_replay

    snap = job["snap"]
    rec = job["rec"]
    t = float(snap["time"])
    batch = sorted(snap["fidelity"]["batch"], key=lambda r: int(r["gid"]))
    gids = [int(r["gid"]) for r in batch]
    live_plan = {int(a): [int(b), int(c)] for a, b, c in rec["chosen"]}
    if set(live_plan) != set(gids):
        raise RuntimeError(f"FAIL LOUD: state {rec['snapshot_id']}: the served plan does not cover the batch")
    plan = [live_plan[g] for g in gids]
    done = np.load(job["tasks_npy"])[:, 1]
    live = float(sum(float(done[g]) - t for g in gids))
    ms = float((json.load(open(job["cell_summary"])).get("env") or {}).get("HEROSIM_DECISION_COST_MS", 0.0))
    live_adj = live - len(gids) * float(rec.get("plan_evals") or 0) * ms / 1000.0       # live charges the decision time to each task of the batch; replay and labels do not
    U._setup_env("cd_ib0", {"cell_summary": job["cell_summary"]}, check_arm=False)
    os.environ.update(HEROSIM_EXCHANGE_FAIRSHARE="1", HEROSIM_EXCHANGE_TCP="1")
    from src.placement import snapshot_fidelity

    spec = {"snapshot": snap, "cell_config": job["cell_config"], "sim_input": job["sim_input"], "live_run_params": snapshot_fidelity.live_run_params()}
    types = [r["fn"] for r in batch]
    out: Dict[str, Any] = {"snapshot_id": rec["snapshot_id"], "trigger_task_id": rec["trigger_task_id"], "time": t, "n_batch": len(gids), "live_rtt": live, "live_rtt_adj": live_adj, "plan_evals": rec.get("plan_evals"), **classes(snap)}
    for name, flag in (("off", None), ("on", "1")):
        if flag is None:
            os.environ.pop(flow_replay.ENV, None)
        else:
            os.environ[flow_replay.ENV] = flag
        t0 = time.time()
        try:
            rt, _setup, _runs = U.replay_rtts(spec, types, [plan])
        except RuntimeError as err:
            if "has no placement time" not in str(err):
                raise
            out["unreplayable"] = str(err)[:200]          # a batch partner outside the snapshot that live had not placed: the replay would wait on it for ever (K1 census class)
            return out
        out[f"rtt_{name}"] = float(rt[0])
        out[f"gap_{name}"] = (float(rt[0]) - live) / live if live else None
        out[f"gap_adj_{name}"] = (float(rt[0]) - live_adj) / live_adj if live_adj else None
        out[f"wall_{name}"] = time.time() - t0
    if job.get("flow_log"):
        out.update(later_flow_join(job["flow_log"], snap, gids, float(max(done[g] for g in gids))))
    return out


def cmd_run(a):
    if a.allow_old_unplanned:
        os.environ["HEROSIM_REPLAY_UNPLANNED_OLD"] = "1"          # inherited by the spawned workers
    import concurrent.futures as cf
    import multiprocessing as mp

    stem = Path(a.cell_dir) / a.stem
    chosen = pick_states(f"{stem}.snapshots.jsonl.chosen.jsonl", a.n) if not a.ids_file else pick_ids(f"{stem}.snapshots.jsonl.chosen.jsonl", json.load(open(a.ids_file)))
    want = {int(r["snapshot_id"]): r for r in chosen}
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["snapshot_id"] for l in open(a.out) if l.strip()}
    jobs = []
    with open(f"{stem}.snapshots.jsonl") as fh:
        for line in fh:
            if not want:
                break
            head = line[:80]
            sid = int(head.split('"snapshot_id":')[1].split(",")[0]) if '"snapshot_id":' in head else None
            if sid is None or sid not in want:
                continue
            snap = json.loads(line)
            rec = want.pop(sid)
            if sid in done:
                continue
            jobs.append({"snap": snap, "rec": rec, "tasks_npy": f"{stem}.tasks.npy", "cell_summary": f"{stem}.summary.json", "cell_config": a.cell_config, "sim_input": a.sim_input, "flow_log": a.flow_log})
    if want:
        raise SystemExit(f"FAIL LOUD: snapshots {sorted(want)[:5]} of the chosen file are not in the snapshot file")
    print(f"{len(jobs)} states to replay ({len(done)} already done)", flush=True)
    ctx = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(a.workers, mp_context=ctx) as pool, open(a.out, "a") as fh:
        futs = [pool.submit(_work, j) for j in jobs]
        for k, f in enumerate(cf.as_completed(futs)):
            try:
                row = f.result(timeout=a.state_timeout)
            except Exception as err:
                print(f"FAIL LOUD: {type(err).__name__}: {err}", flush=True)
                os._exit(2)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if k % 10 == 0:
                print(f"[{k + 1}/{len(jobs)}] state {row['snapshot_id']} live {row['live_rtt']:.2f} off {row['rtt_off']:.2f} on {row['rtt_on']:.2f}", flush=True)


def cmd_census(a):
    """Ghost classes of every snapshot of a cell (no replay): one row per decision, with whether the decision placed its whole batch."""
    stem = Path(a.cell_dir) / a.stem
    placed = {}
    for line in open(f"{stem}.snapshots.jsonl.chosen.jsonl"):
        r = json.loads(line)
        placed[int(r["snapshot_id"])] = bool(r["chosen"]) and not r.get("deferred")
    n = 0
    with open(f"{stem}.snapshots.jsonl") as fh, open(a.out, "w") as out:
        for line in fh:
            snap = json.loads(line)
            sid = int(snap["snapshot_id"])
            out.write(json.dumps({"cell": a.stem, "snapshot_id": sid, "whole_batch": placed.get(sid, False), **classes(snap)}) + "\n")
            n += 1
    print(f"{a.stem}: {n} snapshots", flush=True)


def _q(xs: List[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def _dist(xs: List[float]) -> Dict[str, Any]:
    if not xs:
        return {"n": 0}
    ab = [abs(x) for x in xs]
    return {"n": len(xs), "median_abs_pct": 100 * st.median(ab), "p90_abs_pct": 100 * _q(ab, 0.9), "max_abs_pct": 100 * max(ab), "mean_signed_pct": 100 * st.fmean(xs),
            "share_within_1e-9": sum(x <= 1e-9 for x in ab) / len(ab), "share_within_1pct": sum(x <= 0.01 for x in ab) / len(ab)}


def cmd_read(a):
    allrows = [json.loads(l) for l in open(a.rows) if l.strip()]
    rows = [r for r in allrows if not r.get("unreplayable")]
    out: Dict[str, Any] = {"n": len(rows), "unreplayable": len(allrows) - len(rows), "n_all": len(allrows)}
    for key in ("records_input_io", "records_planned", "unplanned", "rendezvous_ghosts", "active_flows", "ghosts", "n_batch"):
        v = [r[key] for r in rows]
        out[f"per_state_{key}"] = {"median": st.median(v), "p90": _q(v, 0.9), "max": max(v), "share_states_nonzero": sum(x > 0 for x in v) / len(v)}
    groups = {"all": rows,
              "no_planned_or_unplanned": [r for r in rows if not r["records_planned"] and not r["unplanned"] and not r["rendezvous_ghosts"]],
              "any_planned": [r for r in rows if r["records_planned"]],
              "any_unplanned_or_rendezvous": [r for r in rows if r["unplanned"] or r["rendezvous_ghosts"]],
              "any_flow_active": [r for r in rows if r["active_flows"]],
              "no_flow_active": [r for r in rows if not r["active_flows"]]}
    if all("later_flow_on_batch_links" in r for r in rows):
        groups["no_later_flow_on_batch_links"] = [r for r in rows if not r["later_flow_on_batch_links"]]
        groups["later_flow_on_batch_links"] = [r for r in rows if r["later_flow_on_batch_links"]]
        groups["no_later_flow_on_batch_or_stored_links"] = [r for r in rows if not r["later_flow_on_batch_or_stored_links"]]
        groups["later_flow_on_batch_or_stored_links"] = [r for r in rows if r["later_flow_on_batch_or_stored_links"]]
        groups["no_later_flow_in_window_at_all"] = [r for r in rows if not r["later_flows_in_window"]]
    out["gap_vs_live_decision_time_removed"] = {g: {"switch_off": _dist([r["gap_adj_off"] for r in rs]), "switch_on": _dist([r["gap_adj_on"] for r in rs])} for g, rs in groups.items()}
    out["gap_vs_live_raw"] = {g: {"switch_off": _dist([r["gap_off"] for r in rs]), "switch_on": _dist([r["gap_on"] for r in rs])} for g, rs in groups.items()}
    print(json.dumps(out, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--cell-dir", required=True)
    r.add_argument("--stem", required=True)
    r.add_argument("--cell-config", required=True)
    r.add_argument("--sim-input", required=True)
    r.add_argument("--allow-old-unplanned", action="store_true", help="a capture before schema v1.1: replay its unplanned ghosts the old way (sets HEROSIM_REPLAY_UNPLANNED_OLD=1)")
    r.add_argument("--n", type=int, default=240)
    r.add_argument("--flow-log", default="", help="the cell's HEROSIM_FLOW_LOG file: adds the later-flow join to every row")
    r.add_argument("--ids-file", default="", help="JSON list of snapshot ids to replay instead of the evenly spread --n")
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--state-timeout", type=float, default=7200.0)
    r.add_argument("--out", required=True)
    c = sub.add_parser("census")
    c.add_argument("--cell-dir", required=True)
    c.add_argument("--stem", required=True)
    c.add_argument("--out", required=True)
    q = sub.add_parser("read")
    q.add_argument("--rows", required=True)
    q.add_argument("--out", default="")
    a = ap.parse_args(argv)
    {"run": cmd_run, "read": cmd_read, "census": cmd_census}[a.cmd](a)


if __name__ == "__main__":
    main()
