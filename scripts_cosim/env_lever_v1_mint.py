"""env_lever_v1 -- mint the levered workloads and cells for burst_groups_v1, payload_scale_v1
and backbone_sparsity_v1. Every artefact is a one-field transform of a study input that
already exists; nothing else about it changes, and the transform is recorded in the file.

    python3 scripts_cosim/env_lever_v1_mint.py --gate simulation_data/peer_affinity_live_gate \
        --topologies 9001,9002,... [--levers burst,pk0.1,pk10,p04,bw250] [--check]

Workload levers (one output per study window; input = the C40 study's `drainable_*_n50000`):
  burst   every event's timestamp becomes the EARLIEST timestamp of its peer group, so a group
          arrives as one burst; the mean rate is unchanged (`arrival_rescale` is kept and
          `lever` records the group span before and after)
  pk<k>   every `peer_exchange` payload is multiplied by k (0.1 -> 20 MB scale, 3 -> 600 MB, 10 -> 2 GB)
Cell levers (one output per study topology; input = `cc40s<seed>.json`):
  p04     network.topology.connection_probability 0.6 -> 0.4 (sparser client-server graph)
  bw250   network.backbone.bandwidth_mbps 1000 -> 250 (slower fabric)

A minted file is written atomically and is refused if it exists with different content, so a
re-run is a no-op and a drift is an error.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List

WINDOWS = {"w0": "drainable_f4000_n50000", "w1": "drainable_w1_n50000",
           "w2": "drainable_w2_n50000", "w3": "drainable_w3_n50000"}
WORKLOAD_LEVERS = ("burst", "pk0.1", "pk3", "pk10")   # pk3: payload_scale_v1 Amendment 1
CELL_LEVERS = ("p04", "bw250")
LEVERS = WORKLOAD_LEVERS + CELL_LEVERS


def lever_workload_name(lever: str, window: str) -> str:
    base = WINDOWS[window]
    return base if lever in CELL_LEVERS else f"{lever}_{base}"


def lever_cell_name(lever: str, seed: int) -> str:
    return f"cc40s{seed}" if lever in WORKLOAD_LEVERS else f"cc40s{seed}_{lever}"


def _write_atomic(path: Path, doc: dict) -> str:
    text = json.dumps(doc, separators=(",", ":"))
    if path.exists():
        if path.read_text() == text:
            return "exists"
        raise SystemExit(f"FAIL LOUD: {path} exists with different content; refusing to overwrite")
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(text)
    os.replace(tmp, path)
    return "written"


def burst_events(events: List[dict]) -> Dict[str, float]:
    """In place: each event's timestamp := the earliest timestamp of its `peer_group`.
    Returns the group span before and after (medians / p90 in seconds)."""
    first: Dict[int, float] = {}
    last: Dict[int, float] = {}
    for ev in events:
        g = ev.get("peer_group")
        if g is None:
            raise SystemExit("FAIL LOUD: an event carries no peer_group; the burst lever needs the group id")
        t = float(ev["timestamp"])
        first[g] = min(first.get(g, t), t)
        last[g] = max(last.get(g, t), t)
    spans = sorted(last[g] - first[g] for g in first)
    for ev in events:
        ev["timestamp"] = first[ev["peer_group"]]
    ts = [float(ev["timestamp"]) for ev in events]
    if any(b < a for a, b in zip(ts, ts[1:])):
        raise SystemExit("FAIL LOUD: burst timestamps are not non-decreasing")
    return {"n_groups": len(spans), "span_before_median_s": st.median(spans),
            "span_before_p90_s": spans[int(0.9 * (len(spans) - 1))], "span_after_s": 0.0}


def scale_payloads(peer_exchange: List[list], k: float) -> Dict[str, float]:
    before = [float(p[2]) for p in peer_exchange]
    for p in peer_exchange:
        p[2] = float(p[2]) * k
    after = [float(p[2]) for p in peer_exchange]
    return {"k": k, "n_pairs": len(before), "median_bytes_before": st.median(before),
            "median_bytes_after": st.median(after)}


def mint_workload(gate: Path, lever: str, window: str) -> str:
    src = gate / "workloads" / f"{WINDOWS[window]}.json"
    dst = gate / "workloads" / f"{lever_workload_name(lever, window)}.json"
    if dst.exists():
        return "exists"
    doc = json.load(open(src))
    if lever == "burst":
        info = burst_events(doc["events"])
    elif lever.startswith("pk"):
        info = scale_payloads(doc["peer_exchange"], float(lever[2:]))
    else:
        raise SystemExit(f"not a workload lever: {lever}")
    doc["arrival_rescale"]["lever"] = {"name": lever, "source": str(src), **info}
    return _write_atomic(dst, doc)


def mint_cell(gate: Path, lever: str, seed: int) -> str:
    src = gate / "configs" / f"cc40s{seed}.json"
    dst = gate / "configs" / f"{lever_cell_name(lever, seed)}.json"
    cfg = json.load(open(src))
    out = copy.deepcopy(cfg)
    if lever == "p04":
        if float(out["network"]["topology"]["connection_probability"]) != 0.6:
            raise SystemExit(f"FAIL LOUD: {src} does not carry connection_probability 0.6")
        out["network"]["topology"]["connection_probability"] = 0.4
    elif lever == "bw250":
        if float(out["network"]["backbone"]["bandwidth_mbps"]) != 1000.0:
            raise SystemExit(f"FAIL LOUD: {src} does not carry backbone bandwidth 1000")
        out["network"]["backbone"]["bandwidth_mbps"] = 250.0
    else:
        raise SystemExit(f"not a cell lever: {lever}")
    out.setdefault("gate_cell", {})["lever"] = {"name": lever, "source": str(src)}
    # exactly one field moved
    a, b = copy.deepcopy(cfg), copy.deepcopy(out)
    b.pop("gate_cell", None); a.pop("gate_cell", None)
    diffs = _diff(a, b)
    if len(diffs) != 1:
        raise SystemExit(f"FAIL LOUD: lever {lever} moved {diffs}, expected exactly one field")
    return _write_atomic(dst, out)


def _diff(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in set(a) | set(b):
            out += _diff(a.get(k), b.get(k), f"{path}.{k}")
        return out
    return [] if a == b else [path]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gate", type=Path, required=True)
    ap.add_argument("--topologies", required=True, help="study topology seeds for the cell levers, e.g. 9001,9003")
    ap.add_argument("--levers", default=",".join(LEVERS))
    ap.add_argument("--check", action="store_true", help="only print what would be minted")
    a = ap.parse_args(argv)
    seeds = [int(s) for s in a.topologies.split(",") if s]
    for lever in a.levers.split(","):
        if lever not in LEVERS:
            raise SystemExit(f"unknown lever {lever!r}; known: {LEVERS}")
        if lever in WORKLOAD_LEVERS:
            for w in WINDOWS:
                dst = a.gate / "workloads" / f"{lever_workload_name(lever, w)}.json"
                if a.check:
                    print(f"[check] {lever} {w} -> {dst} ({'exists' if dst.exists() else 'to mint'})"); continue
                print(f"[mint] {lever} {w} -> {dst}: {mint_workload(a.gate, lever, w)}", flush=True)
        else:
            for s in seeds:
                dst = a.gate / "configs" / f"{lever_cell_name(lever, s)}.json"
                if a.check:
                    print(f"[check] {lever} {s} -> {dst} ({'exists' if dst.exists() else 'to mint'})"); continue
                print(f"[mint] {lever} {s} -> {dst}: {mint_cell(a.gate, lever, s)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
