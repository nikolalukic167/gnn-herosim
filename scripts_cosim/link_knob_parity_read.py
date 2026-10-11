#!/usr/bin/env python3
"""link_bound_s0_v1 knob parity: the live run and the fidelity replay at a reduced backbone bandwidth.

Each job replays one captured decision (plan "live": the batch forced to the nodes the live run chose, H = 0) and is compared with
the same tasks in the live run itself: executed (node, platform), latency doneTime - scheduledTime, linkWaitTime. Latencies match
within TOL (the replay's known 1 ms poll offset). The pass is not vacuous only if the core links of the reduced config really are
slower than the default config's and the replay ran on the same ones the live run did.

  link_knob_parity_read.py --rows rows.jsonl --table T.json --raw live.raw.json --snaps snaps.jsonl --cfg-low C --cfg-default D --bw LOW [--out o.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import median

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
TOL = 1.0e-3 + 1e-6


def core_bw(cfg_path: str) -> float:
    from src.executesimulation import prepare_infrastructure_for_real_simulation

    infra = prepare_infrastructure_for_real_simulation(json.load(open(cfg_path)), seed=None, sim_input_path=REPO / "data" / "nofs-ids")
    links = infra["link_topology"]["links"]
    core = sorted({float(v["bandwidth_mbps"]) for k, v in links.items() if all(e.startswith("core") for e in k.split("|"))})
    if len(core) != 1:
        raise SystemExit(f"{cfg_path}: core links carry {core}, expected one rate")
    return core[0]


def read(rows, tab, raw, snaps, bw_low: float, bw_default: float, live_cfg_bw: float) -> dict:
    trs = {int(t["taskId"]): t for t in raw["stats"]["taskResults"] if t.get("taskId", -1) >= 0}
    batch = {}
    for line in snaps:
        r = json.loads(line)
        batch[int(r["snapshot_id"])] = sorted(int(b["gid"]) for b in r["fidelity"]["batch"])
    out = {"states": 0, "errors": [], "plan_mismatch": [], "n_tasks": 0, "diffs": [], "link_wait_live_nonzero": 0,
           "link_wait_replay_nonzero": 0, "link_wait_diffs": [], "core_bw_replay": set()}
    for r in rows:
        if "error" in r:
            out["errors"].append([r.get("tag"), r["error"][:200]])
            continue
        sid = int(r["tag"].rsplit("|", 1)[1])
        gids = batch[sid]
        det = r["batch_detail"]
        if len(det) != len(gids):
            out["errors"].append([r["tag"], f"batch {len(det)} vs snapshot {len(gids)}"])
            continue
        out["states"] += 1
        out["core_bw_replay"].add(r.get("core_bw"))
        for g, d, lat in zip(gids, det, r["batch_lat"]):
            live = (str(trs[g]["executionNode"]), int(trs[g]["executionPlatform"]))
            if (str(d["executionNode"]), int(d["executionPlatform"])) != live:
                out["plan_mismatch"].append([r["tag"], g, live, [d["executionNode"], d["executionPlatform"]]])
            out["n_tasks"] += 1
            out["diffs"].append(abs(lat - (tab["done"][g] - tab["scheduled"][g])))
            lw_live, lw_rep = float(trs[g].get("linkWaitTime") or 0.0), float(d.get("linkWaitTime") or 0.0)
            out["link_wait_live_nonzero"] += lw_live > 0
            out["link_wait_replay_nonzero"] += lw_rep > 0
            out["link_wait_diffs"].append(abs(lw_rep - lw_live))
    dif, lwd = out.pop("diffs"), out.pop("link_wait_diffs")
    out["core_bw_replay"] = sorted(x for x in out["core_bw_replay"] if x is not None)
    out.update(
        max_abs_latency_diff_s=max(dif, default=None), median_abs_latency_diff_s=median(dif) if dif else None,
        tasks_within_tol=sum(d <= TOL for d in dif), max_abs_link_wait_diff_s=max(lwd, default=None),
        core_bw_low_cfg=bw_low, core_bw_default_cfg=bw_default, core_bw_live=live_cfg_bw)
    ok = (out["states"] > 0 and not out["errors"] and not out["plan_mismatch"] and out["tasks_within_tol"] == out["n_tasks"]
          and bw_low < bw_default and live_cfg_bw == bw_low and out["core_bw_replay"] == [bw_low])
    out["verdict"] = "PASS" if ok else "FAIL"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True); ap.add_argument("--table", required=True); ap.add_argument("--raw", required=True)
    ap.add_argument("--snaps", required=True); ap.add_argument("--cfg-low", required=True); ap.add_argument("--cfg-default", required=True)
    ap.add_argument("--bw", type=float, required=True); ap.add_argument("--out")
    a = ap.parse_args()
    raw = json.load(open(a.raw))
    low, default = core_bw(a.cfg_low), core_bw(a.cfg_default)
    if low != a.bw:
        raise SystemExit(f"cfg-low carries core rate {low}, expected {a.bw}")
    res = read([json.loads(l) for l in open(a.rows) if l.strip()], json.load(open(a.table)), raw, open(a.snaps).read().splitlines(), a.bw, default, core_bw(raw["config_file"]))
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    return 0 if res["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
