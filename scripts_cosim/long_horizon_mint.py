#!/usr/bin/env python3
"""long_horizon_s0_v1 -- mint the stationary 24 h window L24 at one accel rung (docs/lineages/long_horizon_s0_v1.md).

A cycle is one 50,000-task window built by the accel gate's own chain, in its order and with its seeds:
  grounded_workload_v1_mint.mint (K=1, legacy payloads)  ->  wf1_v1 payload resample + wf1_v1 task mix (seeded by the window seed)
  ->  timestamps x rung factor (cd_gap_v1_build_b)  ->  single origin per group (client_local_v1_single_origin.rewrite)
  ->  the W3 step (configs only; workloads untouched)  ->  the 20 origins spread over 160 clients (scale_probe_remap_clients).
Cycle c uses library (0, 50, 100, 130)[c % 4] and base window (w0..w3)[c % 4] with seed 7400 + c.
  * Cycles 0-3 are the gate windows g0-g3: the x1 window on disk is re-minted and must be equal, and the built window must equal the
    accel gate's file at this rung (events and pairs). So the first 50,000 tasks of L24 are grounded_g0_n50000.
  * Cycles >= 4 take a seeded offset (random.Random("<seed>:long_horizon_offset")): a contiguous run of the library's groups starting
    at a drawn group, and the base events' type / demand / QoS / client rotated by a drawn index (base timestamps unchanged, so the
    rate stretch is the mint's own).
Cycles are concatenated in time: cycle c starts one mean inter-group gap (of cycle c - 1) after cycle c - 1's last arrival; group ids
and pair indices are offset. Groups whose first arrival is at or after --horizon-s are dropped (whole groups). Stationary by
construction: no rate envelope.

  long_horizon_mint.py --rung heavy --manifest <accel inputs>/manifest_heavy.json --gate-wl <accel inputs>/wf1_heavy/wl \\
      --x1 <x1 grounded wl dir> --lib-dir docs/lineages/grounded_workload_v1 --base-dir <drainable windows> \\
      --cfg <cfg cc40s16900.json at this rung> --out <dir>/wf1_heavy
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts_cosim"))

from src.placement.workload_payloads import resample_peer_exchange  # noqa: E402
from src.placement.workload_task_mix import defined_application_types, relabel_events  # noqa: E402

LIBS = (0, 50, 100, 130)
BASES = ("drainable_f4000_n50000", "drainable_w1_n50000", "drainable_w2_n50000", "drainable_w3_n50000")
SEED0 = 7400
N_TASKS = 50000
CLIENTS = 160
WL_NAME = "long_horizon_L24.json"


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def remap_clients(wl: dict, clients: int) -> dict:
    """scale_probe_remap_clients.py, as a function (that script is top-level code); the same rule and the same metadata."""
    used = sorted({e["node_name"] for e in wl["events"]}, key=lambda s: int(s.replace("client_node", "")))
    base = len(used)
    if clients % base:
        raise SystemExit(f"FAIL LOUD: {clients} clients is not a multiple of the {base} origin clients")
    fold = clients // base
    rank = {n: i for i, n in enumerate(used)}
    for e in wl["events"]:
        e["node_name"] = f"client_node{rank[e['node_name']] + base * (int(e['peer_group']) % fold)}"
    c = Counter(e["node_name"] for e in wl["events"])
    wl["scale_probe_remap"] = {"origin_clients_before": base, "clients": clients, "fold": fold, "clients_used": len(c),
                               "max_tasks_per_client": max(c.values()), "min_tasks_per_client": min(c.values())}
    return wl


def offset_inputs(lib: dict, base: dict, seed: int) -> Tuple[dict, dict, Dict[str, int]]:
    rng = random.Random(f"{seed}:long_horizon_offset")
    groups = lib["groups"]
    tail, max_start = 0, 0
    for i in range(len(groups) - 1, -1, -1):
        tail += len(groups[i]["offsets_ms"])
        if tail >= N_TASKS:
            max_start = i
            break
    else:
        raise SystemExit("FAIL LOUD: library holds fewer than N_TASKS tasks")
    g0 = rng.randrange(max_start + 1)
    ev = base["events"]
    o = rng.randrange(len(ev))
    rot = []
    for k, e in enumerate(ev):
        src = ev[(k + o) % len(ev)]
        rot.append({**copy.deepcopy(src), "timestamp": e["timestamp"]})
    return {**lib, "groups": groups[g0:]}, {**base, "events": rot}, {"library_group_offset": g0, "base_event_offset": o}


def build_cycle(c: int, x1_dir: Path, lib_dir: Path, base_dir: Path, factor: float, applications: dict) -> Tuple[dict, dict]:
    from grounded_workload_v1_mint import mint
    from client_local_v1_single_origin import rewrite
    seed = SEED0 + c
    lib = json.load(gzip.open(lib_dir / f"fanout_lib_{LIBS[c % 4]}.json.gz", "rt"))
    base = json.load(open(base_dir / f"{BASES[c % 4]}.json"))
    meta = {"cycle": c, "seed": seed, "library": f"fanout_lib_{LIBS[c % 4]}.json.gz", "base": f"{BASES[c % 4]}.json"}
    if c < 4:
        wl, _m = mint(lib, base, seed, N_TASKS)
        stored = json.load(open(x1_dir / f"grounded_g{c}_n50000.json"))
        if wl["events"] != stored["events"] or wl["peer_exchange"] != stored["peer_exchange"]:
            raise SystemExit(f"FAIL LOUD: re-minting g{c} (seed {seed}) does not reproduce {x1_dir}/grounded_g{c}_n50000.json")
        wl = stored
        meta["x1_sha256"] = sha256(x1_dir / f"grounded_g{c}_n50000.json")
    else:
        lib_o, base_o, offs = offset_inputs(lib, base, seed)
        wl, m = mint(lib_o, base_o, seed, N_TASKS)
        wl["grounded_workload_v1"] = {**m, "library": meta["library"], "base": meta["base"], "long_horizon_offset": offs}
        meta.update(offs)
    # workload_fix_v1_build.apply_sampler (wf1_v1 payloads, wf1_v1 task mix), then cd_gap_v1_build_b, single origin, remap
    wl["peer_exchange"] = resample_peer_exchange(wl["peer_exchange"], seed)
    wl["events"] = relabel_events(wl["events"], seed, applications)
    for e in wl["events"]:
        e["timestamp"] = float(e["timestamp"]) * factor
    wl["duration"] = float(wl["duration"]) * factor
    wl = rewrite(wl)
    wl = remap_clients(wl, CLIENTS)
    meta.update(n_tasks=len(wl["events"]), n_groups=len({e["peer_group"] for e in wl["events"]}),
                first_s=wl["events"][0]["timestamp"], last_s=wl["events"][-1]["timestamp"])
    return wl, meta


def concatenate(windows: List[dict], horizon_s: float) -> Tuple[dict, List[dict]]:
    events: List[dict] = []
    pairs: List[list] = []
    cycles: List[dict] = []
    prev_last, prev_gap, group_base = None, None, 0
    for c, wl in enumerate(windows):
        ev = wl["events"]
        first = float(ev[0]["timestamp"])
        shift = 0.0 if c == 0 else prev_last + prev_gap - first
        gids = sorted({int(e["peer_group"]) for e in ev})
        gstart: Dict[int, float] = {}
        for e in ev:
            gstart.setdefault(int(e["peer_group"]), float(e["timestamp"]) + shift)
        keep = {g for g, t in gstart.items() if t < horizon_s}
        gmap = {g: group_base + i for i, g in enumerate(gids)}
        if c == 0 and any(g != v for g, v in gmap.items()):
            raise SystemExit("FAIL LOUD: cycle 0's group ids are not 0..n-1; the prefix would not be byte-identical")
        idx: Dict[int, int] = {}
        for k, e in enumerate(ev):
            if int(e["peer_group"]) in keep:
                idx[k] = len(events)
                ne = dict(e) if c == 0 else {**e, "timestamp": float(e["timestamp"]) + shift}
                ne["peer_group"] = e["peer_group"] if c == 0 else gmap[int(e["peer_group"])]
                events.append(ne)
        for i, j, b in wl["peer_exchange"]:
            if i in idx and j in idx:
                pairs.append([idx[i], idx[j], b])
        cycles.append({"cycle": c, "shift_s": shift, "n_tasks": len(ev), "n_tasks_kept": len(idx), "n_groups_kept": len(keep)})
        ts = [float(e["timestamp"]) for e in ev]
        prev_last = ts[-1] + shift
        prev_gap = (ts[-1] - ts[0]) / max(1, len(gids) - 1)
        group_base += len(gids)
        if len(keep) < len(gids):
            break
    if any(float(a["timestamp"]) > float(b["timestamp"]) for a, b in zip(events, events[1:])):
        raise SystemExit("FAIL LOUD: concatenated events are not time-ordered")
    return {"events": events, "peer_exchange": pairs}, cycles


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rung", required=True)
    ap.add_argument("--manifest", type=Path, required=True, help="accel gate manifest_<rung>.json (the rung factor and gate sha256)")
    ap.add_argument("--gate-wl", type=Path, required=True, help="accel gate wf1_<rung>/wl: cycles 0-3 must equal these")
    ap.add_argument("--x1", type=Path, required=True)
    ap.add_argument("--lib-dir", type=Path, required=True)
    ap.add_argument("--base-dir", type=Path, required=True)
    ap.add_argument("--cfg", type=Path, required=True, help="the topology's cfg at this rung (copied into <out>/cfg)")
    ap.add_argument("--horizon-s", type=float, default=86400.0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"FAIL LOUD: {a.out} exists; a minted window is frozen")
    man = json.load(open(a.manifest))
    factor = float(man["rungs"][a.rung]["factor"])
    for name, want in man["rungs"][a.rung]["wl_sha256"].items():
        got = sha256(a.gate_wl / name)
        if got != want:
            raise SystemExit(f"FAIL LOUD: gate window {name} sha256 {got} != manifest {want}")
    cfg = json.load(open(a.cfg))
    cfg_factor = float((cfg.get("cd_gap_v1_rate_scale") or {}).get("factor", float("nan")))
    if abs(cfg_factor - factor) > 1e-12 or float(cfg["scheduler"]["batch_timeout"]) != 1.0:
        raise SystemExit(f"FAIL LOUD: cfg {a.cfg} rate factor {cfg_factor} / batch_timeout {cfg['scheduler']['batch_timeout']} "
                         f"is not rung {a.rung} ({factor}, 1.0)")
    applications = defined_application_types()
    windows, metas = [], []
    span = None
    c = 0
    while True:
        wl, meta = build_cycle(c, a.x1, a.lib_dir, a.base_dir, factor, applications)
        if c < 4:
            gate = json.load(open(a.gate_wl / f"grounded_g{c}_n50000.json"))
            if wl["events"] != gate["events"] or wl["peer_exchange"] != gate["peer_exchange"]:
                raise SystemExit(f"FAIL LOUD: cycle {c} built here differs from the gate window grounded_g{c}_n50000.json at {a.rung}")
            meta["gate_sha256"] = sha256(a.gate_wl / f"grounded_g{c}_n50000.json")
        windows.append({"events": wl["events"], "peer_exchange": wl["peer_exchange"]})
        metas.append(meta)
        span = (span or 0.0) + (meta["last_s"] - meta["first_s"])
        print(f"[cycle {c}] seed {meta['seed']} {meta['library']} {meta['base']} tasks {meta['n_tasks']} groups {meta['n_groups']} "
              f"span {meta['last_s'] - meta['first_s']:.1f} s (total {span:.0f} s)", flush=True)
        if span >= a.horizon_s:
            break
        c += 1
    doc, cycles = concatenate(windows, a.horizon_s)
    for m, cy in zip(metas, cycles):
        m.update(cy)
    ts = [e["timestamp"] for e in doc["events"]]
    first_gate = windows[0]["events"]
    if doc["events"][:len(first_gate)] != first_gate:
        raise SystemExit("FAIL LOUD: L24 does not start with grounded_g0_n50000")
    out = {"rps": 150, "duration": a.horizon_s, "events": doc["events"], "peer_exchange": doc["peer_exchange"],
           "long_horizon_v1": {"rung": a.rung, "factor": factor, "horizon_s": a.horizon_s, "envelope": "none (stationary)",
                               "n_tasks": len(ts), "n_groups": len({e["peer_group"] for e in doc["events"]}),
                               "n_pairs": len(doc["peer_exchange"]), "first_s": ts[0], "last_s": ts[-1],
                               "rate_per_s": len(ts) / (ts[-1] - ts[0]), "cycles": metas,
                               "prefix": "first 50,000 tasks and their pairs = accel gate grounded_g0_n50000 at this rung"}}
    (a.out / "wl").mkdir(parents=True)
    (a.out / "cfg").mkdir()
    tmp = a.out / "wl" / (WL_NAME + ".partial")
    with open(tmp, "w") as fh:
        json.dump(out, fh)
    os.replace(tmp, a.out / "wl" / WL_NAME)
    with open(a.out / "cfg" / a.cfg.name, "w") as fh:
        json.dump(cfg, fh, indent=1)
    summary = {k: v for k, v in out["long_horizon_v1"].items() if k != "cycles"}
    summary.update(sha256=sha256(a.out / "wl" / WL_NAME), cfg=str(a.cfg), cfg_sha256=sha256(a.cfg), n_cycles=len(metas))
    print(json.dumps(summary, indent=1))
    with open(a.out / "mint.json", "w") as fh:
        json.dump({**summary, "cycles": metas}, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
