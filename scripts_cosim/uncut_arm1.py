"""uncut_groups_s0_v1 Arm 1 label build (SCIENTIST's corpus spec, 2026-10-13): pool planning, state drawing, label driver.

  plan   : from the minted corpus workloads, choose the capture pool per (size bucket x time stratum) = POOL_MULT x its draw, as (window set, window, rung, group, topology)
           pairs, seeded; write one R1A_SNAPSHOT_PLAN per window set and the pool file. Prints the pool census and stops (exit 3) when any cell has < 3 x its draw.
  openings: write an inputs tree whose workloads keep only the first OPEN_S seconds of arrival (opening-only capture cells: a cell's opening does not depend on what follows).
Buckets are the served batch size: <= 10, 11-50, 51-100, 101-256 (larger batches are counted and not labelled). Time stratum: batch arrival t < 360 s (30 % of every bucket's draw) or later."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

BUCKETS = (("le10", 1, 10), ("11_50", 11, 50), ("51_100", 51, 100), ("101_256", 101, 256))
DRAWS = {"le10": 4800, "11_50": 4600, "51_100": 2000, "101_256": 600}       # 12,000 labelled states (SCIENTIST, as corrected)
EARLY_SHARE, EARLY_T, POOL_MULT, MIN_MULT = 0.30, 360.0, 5, 3
WINDOWS, RUNGS = ("g0", "g1", "g2", "g3"), ("moderate", "heavy")


def bucket_of(size: int):
    return next((n for n, lo, hi in BUCKETS if lo <= size <= hi), None)


def groups_of(wl: Dict[str, Any]) -> List[Tuple[int, int, float]]:
    """(trigger task id = the group's first event index, size, first arrival time) per peer group, in arrival order."""
    first: Dict[Any, int] = {}
    size: collections.Counter = collections.Counter()
    for i, e in enumerate(wl["events"]):
        g = e["peer_group"]
        size[g] += 1
        first.setdefault(g, i)
    return [(first[g], size[g], float(wl["events"][first[g]]["timestamp"])) for g in sorted(first, key=lambda g: first[g])]


def cell_draw(bucket: str, early: bool) -> int:
    n_early = round(DRAWS[bucket] * EARLY_SHARE)
    return n_early if early else DRAWS[bucket] - n_early


def plan_pool(wls: Dict[Tuple[str, str, str], Dict[str, Any]], topos: Sequence[int], seed: int, mult: int = POOL_MULT, thin_le10: float = 0.05) -> Dict[str, Any]:
    """wls[(window set, window, rung)] -> workload. Returns the pool, per-cell census and the plans per window set."""
    cells: Dict[Tuple[str, bool], List[Tuple[Tuple[str, str, str], int, int, float]]] = collections.defaultdict(list)
    over256 = collections.Counter()
    for key, wl in sorted(wls.items()):
        for trig, size, t in groups_of(wl):
            b = bucket_of(size)
            if b is None:
                if size > 256:
                    over256[key[2]] += 1
                continue
            cells[(b, t < EARLY_T)].append((key, trig, size, t))
    rng = random.Random(f"arm1_pool:{seed}")
    pool, census = [], []
    T = len(topos)
    for (b, early), items in sorted(cells.items()):
        draw = cell_draw(b, early)
        n_pairs = len(items) * T
        avail = int(n_pairs * thin_le10) if b == "le10" else n_pairs        # <= 10: a seeded 5 % of batches
        take = min(mult * draw, avail)
        idx = rng.sample(range(n_pairs), take)
        for i in sorted(idx):
            key, trig, size, t = items[i // T]
            pool.append({"set": key[0], "window": key[1], "rung": key[2], "topology": int(topos[i % T]), "trigger": trig, "size": size, "t": t, "bucket": b, "early": early})
        census.append({"bucket": b, "early": early, "draw": draw, "available": avail, "pool": take, "ratio_available_to_draw": avail / draw if draw else None, "ok": avail >= MIN_MULT * draw})
    plans: Dict[str, Dict[str, List[int]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for p in pool:
        plans[p["set"]][f"{p['topology']}__{p['window']}{p['rung']}"].append(p["trigger"])
    plans = {s: {k: sorted(set(v)) for k, v in d.items()} for s, d in plans.items()}
    return {"pool": pool, "census": census, "plans": plans, "groups_over_256": dict(over256)}


def truncate_workload(wl: Dict[str, Any], open_s: float) -> Dict[str, Any]:
    """The workload cut to events arriving by open_s (ids unchanged: events are in arrival order); pairs touching a dropped event go."""
    keep = sum(1 for e in wl["events"] if float(e["timestamp"]) <= open_s)
    out = {k: v for k, v in wl.items() if k not in ("events", "peer_exchange")}
    out["events"] = wl["events"][:keep]
    out["peer_exchange"] = [p for p in wl.get("peer_exchange", []) if p[0] < keep and p[1] < keep]
    out["rps"] = keep
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--sets", nargs="+", required=True, help="NAME=inputs_group_dir ... (the inputs tree whose wf1_<rung>/wl windows are the corpus windows)")
    p.add_argument("--topos", nargs="+", type=int, required=True); p.add_argument("--seed", type=int, required=True); p.add_argument("--out", required=True)
    o = sub.add_parser("openings")
    o.add_argument("--src", required=True); o.add_argument("--dst", required=True); o.add_argument("--open-s", type=float, default=420.0)
    a = ap.parse_args()
    if a.cmd == "openings":
        if os.path.exists(a.dst):
            raise SystemExit(f"FAIL LOUD: {a.dst} exists")
        shutil.copytree(a.src, a.dst, ignore=shutil.ignore_patterns("wl"))
        for rung in RUNGS:
            os.makedirs(f"{a.dst}/wf1_{rung}/wl")
            for w in WINDOWS:
                f = f"wf1_{rung}/wl/grounded_{w}_n50000.json"
                wl = truncate_workload(json.load(open(f"{a.src}/{f}")), a.open_s)
                json.dump(wl, open(f"{a.dst}/{f}", "w"), separators=(",", ":"))
                print(f"[open] {rung} {w}: {wl['rps']} events kept (arrival <= {a.open_s} s)")
        return 0
    wls = {}
    for spec in a.sets:
        name, d = spec.split("=", 1)
        for rung in RUNGS:
            for w in WINDOWS:
                wls[(name, w, rung)] = json.load(open(f"{d}/wf1_{rung}/wl/grounded_{w}_n50000.json"))
    res = plan_pool(wls, a.topos, a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for s, d in res["plans"].items():
        (out / f"snapshot_plan_{s}.json").write_text(json.dumps(d, sort_keys=True))
    (out / "pool.json").write_text(json.dumps(res["pool"], separators=(",", ":")))
    (out / "census.json").write_text(json.dumps({"census": res["census"], "groups_over_256": res["groups_over_256"], "seed": a.seed, "topologies": a.topos}, indent=1))
    for c in res["census"]:
        print(f"{c['bucket']:8s} {'early' if c['early'] else 'late ':5s} draw {c['draw']:5d} available {c['available']:9d} pool {c['pool']:6d} {'' if c['ok'] else '  <-- under 3x the draw'}")
    print("groups over 256 (counted, not labelled):", res["groups_over_256"])
    print("pool sha256", hashlib.sha256((out / "pool.json").read_bytes()).hexdigest())
    return 0 if all(c["ok"] for c in res["census"]) else 3


if __name__ == "__main__":
    sys.exit(main())
