#!/usr/bin/env python3
"""Label probe for partial_state_v9 (descriptive; scratch, not committed).

Question: does the fidelity replay that produced the labels charge the reverse pull (partner pulls from the candidate over the candidate's uplink)?

For a captured decision with a served plan P, a batch task A with remote in-batch partners, and a wired candidate x and a cellular candidate y of A that have the
same platform type, are initialised and idle (so execution, cold start and queue are equal): replay P[A->x] and P[A->y]. Read per task peerExchangeTime from the
replay's taskResults. Delta_in = change of A's own exchange time, Delta_out = change of the partners' exchange times, Delta_rtt = change of the label.
Predicted (solo, pipelined, from the cell's link_topology): Delta_in_pred = sum_p [s(p->y) - s(p->x)], Delta_out_pred = sum_p [s(y->p) - s(x->p)], s = b / (min directed bw) + latency.

  label_probe.py --cell-dir D --stem S --cell-config C --sim-input I --out rows.jsonl [--n-states 12] [--max-batch 14]
"""
import argparse
import json
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

REPO = Path(os.environ["PROBE_REPO"])
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts_cosim"))

MIB = 1024 ** 2


def solo(lt, src, dst, b):
    from src.placement.network_fabric import directed_hops_from_topology, link_key

    hops = directed_hops_from_topology(lt, src, dst)
    path = lt["routes"][src][dst]
    lat = sum(float(lt["links"][link_key(p, q)]["latency"]) for p, q in zip(path, path[1:]))
    return b / (min(bw for _k, bw in hops) * MIB) + lat


def choose_states(chosen_path, n, max_batch):
    recs = []
    for line in open(chosen_path):
        r = json.loads(line)
        if r["chosen"] and not r.get("deferred") and 3 <= len(r["chosen"]) <= max_batch:
            recs.append(r)
    if len(recs) <= n:
        return recs
    step = len(recs) / n
    return [recs[int(i * step)] for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-dir", required=True)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--cell-config", required=True)
    ap.add_argument("--sim-input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-states", type=int, default=12)
    ap.add_argument("--max-batch", type=int, default=14)
    ap.add_argument("--per-state", type=int, default=2)
    a = ap.parse_args()

    import uncut_ils as U
    stem = Path(a.cell_dir) / a.stem
    summary = f"{stem}.summary.json"
    classes = json.load(open(summary))["accessClasses"]
    chosen = {int(r["snapshot_id"]): r for r in choose_states(f"{stem}.snapshots.jsonl.chosen.jsonl", a.n_states, a.max_batch)}
    U._setup_env("cd_ib0", {"cell_summary": summary}, check_arm=False)
    os.environ.update(HEROSIM_EXCHANGE_FAIRSHARE="1", HEROSIM_EXCHANGE_TCP="1", HEROSIM_REPLAY_STORED_FLOWS="1")
    from src.placement import snapshot_fidelity
    from src.placement.fidelity_replay import FidelityReplay

    rows = []
    with open(f"{stem}.snapshots.jsonl") as fh, open(a.out, "a") as out:
        for line in fh:
            if not chosen:
                break
            head = line[:80]
            if '"snapshot_id":' not in head:
                continue
            sid = int(head.split('"snapshot_id":')[1].split(",")[0])
            if sid not in chosen:
                continue
            rec = chosen.pop(sid)
            snap = json.loads(line)
            fid = snap["fidelity"]
            batch = sorted(fid["batch"], key=lambda r: int(r["gid"]))
            gids = [int(r["gid"]) for r in batch]
            served = {int(t): (int(n), int(p)) for t, n, p in rec["chosen"]}
            if set(served) != set(gids):
                continue
            pairs = {}
            for i, j, b in fid["pairs"]:
                if int(i) in served and int(j) in served:
                    pairs.setdefault(int(i), []).append((int(j), float(b)))
                    pairs.setdefault(int(j), []).append((int(i), float(b)))
            node_name_of = {}
            for lst in snap["replicas_by_type"].values():
                for c in lst:
                    node_name_of[int(c["node_id"])] = c["node_name"]
            spec = {"snapshot": snap, "cell_config": a.cell_config, "sim_input": a.sim_input, "live_run_params": snapshot_fidelity.live_run_params()}
            types = [r["fn"] for r in batch]
            try:
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    replay = FidelityReplay(spec, types, {})
            except RuntimeError as err:
                print(f"skip state {sid}: {str(err)[:120]}", flush=True)
                continue
            lt = replay.base_infra["link_topology"]
            done = 0
            for ai, g in enumerate(gids):
                if done >= a.per_state:
                    break
                partners = [(j, b) for j, b in pairs.get(g, []) if node_name_of.get(served[j][0]) != node_name_of.get(served[g][0])]
                if not partners:
                    continue
                pnodes = {node_name_of[served[j][0]] for j, _ in pairs.get(g, [])}
                fn = batch[ai]["fn"]
                cands = [c for c in snap["replicas_by_type"][fn] if c.get("initialized") and not c.get("queue_length") and not c.get("queue_drain_seconds")
                         and c["node_name"] not in pnodes]
                wired = [c for c in cands if classes.get(c["node_name"]) == "wired"]
                cell = [c for c in cands if classes.get(c["node_name"]) == "cellular"]
                combos = [(x, y) for x in sorted(wired, key=lambda c: c["node_id"]) for y in sorted(cell, key=lambda c: c["node_id"])
                          if x["platform_type"] == y["platform_type"]]
                cache = {}

                def run_cand(c):
                    key = (int(c["node_id"]), int(c["platform_id"]))
                    if key in cache:
                        return cache[key]
                    plan = {i: served[gg] for i, gg in enumerate(gids)}
                    plan[ai] = key
                    buf = StringIO()
                    try:
                        with redirect_stdout(buf), redirect_stderr(buf):
                            result = replay.run(plan)
                    except (RuntimeError, SystemExit) as err:
                        print(f"skip cand {key} state {sid} task {g}: {type(err).__name__} {buf.getvalue()[-90:]!r}", flush=True)
                        cache[key] = None
                        return None
                    trs = {tr["taskId"]: tr for tr in result["stats"]["taskResults"] if tr.get("taskId", -1) >= 0}
                    loc = {gg: replay.batch_local[i] for i, gg in enumerate(gids)}
                    cache[key] = {"rtt": sum(float(trs[loc[gg]]["doneTime"]) - float(trs[loc[gg]]["scheduledTime"]) for gg in gids),
                                  "ex": {gg: float(trs[loc[gg]].get("peerExchangeTime", 0.0) or 0.0) for gg in gids}}
                    return cache[key]

                res = None
                for x, y in combos[:8]:
                    rx, ry = run_cand(x), run_cand(y)
                    if rx is not None and ry is not None:
                        res = {"x": rx, "y": ry}
                        break
                if res is None:
                    continue
                pg = [j for j, _ in partners]
                d_in = res["y"]["ex"][g] - res["x"]["ex"][g]
                d_out = sum(res["y"]["ex"][j] - res["x"]["ex"][j] for j in pg)
                px, py = x["node_name"], y["node_name"]
                pred_in = sum(solo(lt, node_name_of[served[j][0]], py, b) - solo(lt, node_name_of[served[j][0]], px, b) for j, b in partners)
                pred_out = sum(solo(lt, py, node_name_of[served[j][0]], b) - solo(lt, px, node_name_of[served[j][0]], b) for j, b in partners)
                row = {"stem": a.stem, "snapshot_id": sid, "task": g, "n_batch": len(gids), "n_remote_partners": len(partners), "bytes": sum(b for _, b in partners),
                       "partner_classes": sorted({classes.get(node_name_of[served[j][0]], "?") for j in pg}),
                       "x": px, "y": py, "platform_type": x["platform_type"], "rtt_x": res["x"]["rtt"], "rtt_y": res["y"]["rtt"], "d_rtt": res["y"]["rtt"] - res["x"]["rtt"],
                       "d_in": d_in, "d_out": d_out, "pred_in": pred_in, "pred_out": pred_out}
                out.write(json.dumps(row) + "\n")
                out.flush()
                rows.append(row)
                done += 1
    print(f"{a.stem}: {len(rows)} probe rows", flush=True)


if __name__ == "__main__":
    import traceback

    try:
        main()
    except BaseException:
        traceback.print_exc(file=sys.__stderr__)
        raise
