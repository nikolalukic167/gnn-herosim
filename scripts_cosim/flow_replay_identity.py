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


def classes(snap: Dict[str, Any]) -> Dict[str, Any]:
    fid = snap["fidelity"]
    flows = fid.get("flows") or {}
    ex = flows.get("exchanges") or []
    ghosts = fid.get("ghosts") or []
    return {"records_input_io": sum(not e["planned"] for e in ex), "records_planned": sum(bool(e["planned"]) for e in ex),
            "unplanned": len(flows.get("unplanned") or []), "rendezvous_ghosts": sum(g["stage"] == "rendezvous" for g in ghosts),
            "active_flows": len(flows.get("active") or []), "ghosts": len(ghosts), "batch": len(fid["batch"])}


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
    U._setup_env("cd_ib0", {"cell_summary": job["cell_summary"]}, check_arm=False)
    os.environ.update(HEROSIM_EXCHANGE_FAIRSHARE="1", HEROSIM_EXCHANGE_TCP="1")
    from src.placement import snapshot_fidelity

    spec = {"snapshot": snap, "cell_config": job["cell_config"], "sim_input": job["sim_input"], "live_run_params": snapshot_fidelity.live_run_params()}
    types = [r["fn"] for r in batch]
    out: Dict[str, Any] = {"snapshot_id": rec["snapshot_id"], "trigger_task_id": rec["trigger_task_id"], "time": t, "n_batch": len(gids), "live_rtt": live, **classes(snap)}
    for name, flag in (("off", None), ("on", "1")):
        if flag is None:
            os.environ.pop(flow_replay.ENV, None)
        else:
            os.environ[flow_replay.ENV] = flag
        t0 = time.time()
        rt, _setup, _runs = U.replay_rtts(spec, types, [plan])
        out[f"rtt_{name}"] = float(rt[0])
        out[f"gap_{name}"] = (float(rt[0]) - live) / live if live else None
        out[f"wall_{name}"] = time.time() - t0
    out["links"] = job.get("links")
    return out


def cmd_run(a):
    import concurrent.futures as cf
    import multiprocessing as mp

    stem = Path(a.cell_dir) / a.stem
    chosen = pick_states(f"{stem}.snapshots.jsonl.chosen.jsonl", a.n)
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
            jobs.append({"snap": snap, "rec": rec, "tasks_npy": f"{stem}.tasks.npy", "cell_summary": f"{stem}.summary.json", "cell_config": a.cell_config, "sim_input": a.sim_input})
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
    rows = [json.loads(l) for l in open(a.rows) if l.strip()]
    out: Dict[str, Any] = {"n": len(rows)}
    for key in ("records_input_io", "records_planned", "unplanned", "rendezvous_ghosts", "active_flows", "ghosts", "n_batch"):
        v = [r[key] for r in rows]
        out[f"per_state_{key}"] = {"median": st.median(v), "p90": _q(v, 0.9), "max": max(v), "share_states_nonzero": sum(x > 0 for x in v) / len(v)}
    groups = {"all": rows,
              "no_planned_or_unplanned": [r for r in rows if not r["records_planned"] and not r["unplanned"] and not r["rendezvous_ghosts"]],
              "any_planned": [r for r in rows if r["records_planned"]],
              "any_unplanned_or_rendezvous": [r for r in rows if r["unplanned"] or r["rendezvous_ghosts"]],
              "any_flow_active": [r for r in rows if r["active_flows"]],
              "no_flow_active": [r for r in rows if not r["active_flows"]]}
    if all("later_flow_on_batch_links" in r for r in rows) and any(r["later_flow_on_batch_links"] is not None for r in rows):
        groups["no_later_flow_on_batch_links"] = [r for r in rows if r["later_flow_on_batch_links"] is False]
        groups["later_flow_on_batch_links"] = [r for r in rows if r["later_flow_on_batch_links"] is True]
    out["gap_vs_live"] = {g: {"switch_off": _dist([r["gap_off"] for r in rs]), "switch_on": _dist([r["gap_on"] for r in rs])} for g, rs in groups.items()}
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
    r.add_argument("--n", type=int, default=240)
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--state-timeout", type=float, default=7200.0)
    r.add_argument("--out", required=True)
    q = sub.add_parser("read")
    q.add_argument("--rows", required=True)
    q.add_argument("--out", default="")
    a = ap.parse_args(argv)
    {"run": cmd_run, "read": cmd_read}[a.cmd](a)


if __name__ == "__main__":
    main()
