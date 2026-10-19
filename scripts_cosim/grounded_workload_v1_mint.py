#!/usr/bin/env python3
"""grounded_workload_v1 -- mint a study workload whose group structure comes from Alibaba's call graphs
(docs/lineages/grounded_workload_v1.md).

From the fan-out library (grounded_workload_v1_extract.py) and a pre-burst study window (drainable_*_n50000):
  * groups, in request-arrival order, take the library's fan-out sizes until --n-tasks tasks are placed
    (the last group is cut to fit);
  * a group's dispatch time is its request's arrival, with the library's inter-request times stretched by one
    factor so the task rate equals the base window's (the admissible x1 load of the study);
  * a member's time is its group's dispatch time plus its REAL sibling offset (ms, never stretched);
  * each task takes the k-th base event's type, demand scale, QoS and client, so the type mix and sources are
    the study's;
  * peers are drawn with the study's rule (make_peer_affinity_production_workload.py): each member draws
    min(2, size - 1) distinct partners in its group, payload 200 MB * 10**U(-1, 1).
Events are sorted by time (stable) and peer indices refer to the sorted order.

  grounded_workload_v1_mint.py --lib fanout_lib_0.json --base drainable_f4000_n50000.json --seed 11 --out g0.json

--merge-k K (scale_sweep_v1): K consecutive library groups become one group, dispatched at the first one's arrival;
each member keeps its request's raw ms lag behind that arrival plus its sibling offset (neither is stretched), so
the task rate is unchanged and a decision batch holds ~K times the tasks. Partners are drawn over the merged group,
and one bridging pair joins any components the draw leaves apart, so the batched seat's peer closure is the whole
group. K = 1 is byte-identical to the original mint.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
import random
import statistics as st
from typing import Dict, List, Tuple

from src.placement.workload_payloads import (
    PAYLOAD_SAMPLERS, payload_rng, payload_sampler_meta, require_sampler, sample_payload_bytes,
)

PARTNERS = 2
X_SCALE_BYTES = 200e6
LOG10_SPREAD = 1.0
NESTED_MAX = 8
BIG_PAIR_BYTES = 50e6


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def merge_groups(groups: List[dict], k: int) -> List[dict]:
    if k == 1:
        return groups
    out = []
    for i in range(0, len(groups), k):
        chunk = groups[i:i + k]
        t0 = int(chunk[0]["t0_ms"])
        offs = sorted(int(g["t0_ms"]) - t0 + int(o) for g in chunk for o in g["offsets_ms"])
        out.append({"t0_ms": t0, "offsets_ms": offs})
    return out


def regroup_to_size(groups: List[dict], size: int) -> List[dict]:
    """big_groups_s0_v1: peer groups of exactly `size` tasks (the last one shorter). The library's tasks, in arrival order, are cut into
    consecutive chunks of `size`; a chunk is dispatched at its first task's arrival and every member keeps its real lag behind it (ms,
    never stretched), as --merge-k does. The task sequence and so the task rate are those of the K=1 mint."""
    if size < 1:
        raise ValueError(f"group size {size} < 1")
    abs_ms = sorted(int(g["t0_ms"]) + int(o) for g in groups for o in g["offsets_ms"])
    out = []
    for i in range(0, len(abs_ms), size):
        chunk = abs_ms[i:i + size]
        out.append({"t0_ms": chunk[0], "offsets_ms": [t - chunk[0] for t in chunk]})
    return out


def plan_groups(groups: List[dict], n_tasks: int) -> List[Tuple[int, List[int]]]:
    out: List[Tuple[int, List[int]]] = []
    placed = 0
    for g in groups:
        offs = list(g["offsets_ms"])[: n_tasks - placed]
        out.append((int(g["t0_ms"]), offs))
        placed += len(offs)
        if placed == n_tasks:
            return out
    raise SystemExit(f"FAIL LOUD: library holds only {placed} tasks < {n_tasks}")


def mint(lib: dict, base: dict, seed: int, n_tasks: int, merge_k: int = 1,
         payload_sampler: str = "legacy", group_size: int = 0, start_group: int = 0, partners_nested: int = 0) -> Tuple[dict, dict]:
    require_sampler(payload_sampler)
    if partners_nested and (not 1 <= partners_nested <= NESTED_MAX or payload_sampler != "wf1_v1" or merge_k != 1):
        raise SystemExit(f"FAIL LOUD: --partners-nested K needs 1 <= K <= {NESTED_MAX}, --payload-sampler wf1_v1 and --merge-k 1")
    if group_size and merge_k != 1:
        raise SystemExit("FAIL LOUD: --group-size and --merge-k are two ways to set the group size; give one")
    if start_group and (merge_k != 1 or group_size):
        raise SystemExit("FAIL LOUD: --start-group counts library groups; it is defined for the K=1 mint only")
    if not 0 <= start_group < len(lib["groups"]):
        raise SystemExit(f"FAIL LOUD: --start-group {start_group} outside the library's {len(lib['groups'])} groups")
    groups = lib["groups"][start_group:]
    plan = plan_groups(regroup_to_size(groups, group_size) if group_size else merge_groups(groups, merge_k), n_tasks)
    bev = base["events"][:n_tasks]
    if len(bev) < n_tasks:
        raise SystemExit("FAIL LOUD: base window shorter than --n-tasks")
    bts = [float(e["timestamp"]) for e in bev]
    rate = n_tasks / (bts[-1] - bts[0])
    rng = random.Random(seed)
    payload_rng_ = payload_rng(seed) if payload_sampler == "wf1_v1" else None

    def payload() -> float:
        # wf1_v1 still consumes the legacy draw so the partner draws after it do not move: the stages differ in
        # payloads only, on an identical pair graph. Its own values are drawn in a post-pass (below).
        return X_SCALE_BYTES * (10.0 ** rng.uniform(-LOG10_SPREAD, LOG10_SPREAD))

    # The trace stores whole milliseconds; a stretch of ~4000x would put every arrival on a ~4 s lattice and
    # dispatch same-millisecond requests at one instant. Spread each arrival uniformly inside its millisecond.
    t0s = sorted(t0 + rng.random() for t0, _offs in plan)
    t_first = t0s[0]
    arrival_span_s = (t0s[-1] - t_first) / 1000.0
    if arrival_span_s <= 0:
        raise SystemExit("FAIL LOUD: degenerate arrival span")
    stretch = (n_tasks / rate) / arrival_span_s

    raw: List[dict] = []
    k = 0
    for gi, (t0, (_t0_int, offs)) in enumerate(zip(t0s, plan)):
        anchor = bts[0] + (t0 - t_first) / 1000.0 * stretch
        for o in offs:
            src = bev[k]
            ev = {"timestamp": anchor + o / 1000.0,
                  "application": copy.deepcopy(src["application"]),
                  "qos": copy.deepcopy(src["qos"]), "node_name": src["node_name"], "peer_group": gi}
            raw.append(ev)
            k += 1
    order = sorted(range(len(raw)), key=lambda i: (raw[i]["timestamp"], i))
    new_idx = {old: new for new, old in enumerate(order)}
    events = [raw[i] for i in order]

    pairs: Dict[Tuple[int, int], float] = {}
    nested = None
    if partners_nested:
        nested = _nested_pairs(plan, new_idx, events, seed, partners_nested, bool(group_size))
        pairs = nested["pairs"]
    k = 0
    for gi, (_t0, offs) in enumerate(plan):
        members = [new_idx[k + j] for j in range(len(offs))]
        k += len(offs)
        if nested:
            continue
        p = min(PARTNERS, len(members) - 1)
        for i in members:
            for j in (rng.sample([m for m in members if m != i], p) if p > 0 else []):
                key = (min(i, j), max(i, j))
                if key not in pairs:
                    pairs[key] = payload()
        if (merge_k > 1 or group_size) and len(members) > 1:
            root = {m: m for m in members}

            def find(x: int) -> int:
                while root[x] != x:
                    root[x] = root[root[x]]
                    x = root[x]
                return x
            for (a, b) in pairs:
                if a in root and b in root:
                    root[find(a)] = find(b)
            comps = sorted({find(m) for m in members})
            for a, b in zip(comps, comps[1:]):
                key = (min(a, b), max(a, b))
                pairs[key] = payload()
                root[find(a)] = find(b)

    if nested:
        pairs, nested_meta = _nested_bytes(nested, events, plan, seed, partners_nested)
    elif payload_rng_ is not None:
        # sorted pair order = the order of the written `peer_exchange`, so resampling a legacy file with the same
        # seed (workload_payloads.resample_peer_exchange) gives these exact bytes
        for key in sorted(pairs):
            pairs[key] = sample_payload_bytes(payload_rng_)

    sizes = [len(offs) for _t, offs in plan]
    spans = [offs[-1] for _t, offs in plan if len(offs) > 1]
    interleaved = sum(1 for a, b in zip(order, order[1:]) if raw[a]["peer_group"] > raw[b]["peer_group"])
    ts = [e["timestamp"] for e in events]
    meta = {
        "n_tasks": n_tasks, "n_groups": len(plan), "n_pairs": len(pairs), "seed": seed,
        "rate_per_s": n_tasks / (ts[-1] - ts[0]), "base_rate_per_s": rate, "arrival_stretch": stretch,
        "mean_group_size": st.mean(sizes), "singleton_share": sizes.count(1) / len(sizes),
        "group_span_ms": {"median": st.median(spans), "p90": sorted(spans)[int(0.9 * (len(spans) - 1))],
                          "max": max(spans)},
        "interleaved_boundaries": interleaved,
        "peer_rule": {"partners": PARTNERS, "x_scale_bytes": X_SCALE_BYTES, "log10_spread": LOG10_SPREAD},
    }
    if merge_k > 1:
        meta["merge_k"] = merge_k
    if group_size:
        meta["group_size"] = group_size
    if start_group:
        meta["start_group"] = start_group
    if payload_sampler != "legacy":
        meta["payload_sampler"] = payload_sampler_meta()
    if nested:
        meta["peer_rule"] = {**meta["peer_rule"], "partners": partners_nested, "nested": True, "nested_max": NESTED_MAX,
                             "bytes_rule": "wf1_v1 draws, each group scaled to its level-2 byte total", "x_scale_bytes": None}
        meta["partner_nested"] = {**nested_meta, "level": partners_nested, "nonpair_sha256": nonpair_sha(
            {"rps": base["rps"], "duration": base["duration"], "events": events})}
    doc = {"rps": base["rps"], "duration": base["duration"], "events": events,
           "peer_exchange": [[i, j, b] for (i, j), b in sorted(pairs.items())]}
    return doc, meta


def nonpair_sha(doc: dict) -> str:
    """Digest of a workload with `peer_exchange` (and the mint metadata) removed: equal across partner levels."""
    body = {k: v for k, v in doc.items() if k not in ("peer_exchange", "grounded_workload_v1")}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def _components_bridge(members: List[int], keys: Dict[Tuple[int, int], float], draw) -> int:
    root = {m: m for m in members}

    def find(x: int) -> int:
        while root[x] != x:
            root[x] = root[root[x]]
            x = root[x]
        return x
    for (a, b) in keys:
        if a in root and b in root:
            root[find(a)] = find(b)
    comps = sorted({find(m) for m in members})
    added = 0
    for a, b in zip(comps, comps[1:]):
        keys[(min(a, b), max(a, b))] = draw()
        root[find(a)] = find(b)
        added += 1
    return added


def _nested_pairs(plan, new_idx, events, seed: int, level: int, bridge: bool) -> dict:
    """Per task ONE ordered sample of min(NESTED_MAX, size - 1) partners from a stream of its own (the partner draws share
    nothing with the payload draws); level k keeps the first min(k, size - 1). The sample depends on neither k nor the
    payload sampler, so level k's pairs are a subset of level k+1's before the bridging step."""
    rng = random.Random(seed * 1000003 + 8)
    samples: List[List[Tuple[int, List[int]]]] = []
    k = 0
    for _t0, offs in plan:
        members = [new_idx[k + j] for j in range(len(offs))]
        k += len(offs)
        m = min(NESTED_MAX, len(members) - 1)
        samples.append([(i, rng.sample([x for x in members if x != i], m) if m > 0 else []) for i in members])
    out = {}
    for lv in sorted({2, level}):
        pairs: Dict[Tuple[int, int], float] = {}
        bridges = 0
        for samp in samples:
            members = [i for i, _ in samp]
            gp: Dict[Tuple[int, int], float] = {}
            for i, ps in samp:
                for j in ps[:lv]:
                    gp[(min(i, j), max(i, j))] = 0.0
            if bridge and len(members) > 1:
                bridges += _components_bridge(members, gp, lambda: 0.0)
            pairs.update(gp)
        out[lv] = (pairs, bridges)
    return {"pairs": out[level][0], "by_level": out, "level": level}


def _bytes_for(keys, seed: int) -> Dict[Tuple[int, int], float]:
    r = payload_rng(seed)
    return {key: sample_payload_bytes(r) for key in sorted(keys)}


def _nested_bytes(nested: dict, events: List[dict], plan, seed: int, level: int):
    """wf1_v1 bytes drawn as today over the level's sorted pairs, then every pair of a group is scaled by
    (the group's byte total at level 2) / (its total at this level), so a group carries exactly its level-2 bytes."""
    grp = lambda key: events[key[0]]["peer_group"]  # noqa: E731
    ref = _bytes_for(nested["by_level"][2][0], seed)
    cur = _bytes_for(nested["pairs"], seed)
    tot2: Dict[int, float] = {}
    totk: Dict[int, float] = {}
    for key, b in ref.items():
        tot2[grp(key)] = tot2.get(grp(key), 0.0) + b
    for key, b in cur.items():
        totk[grp(key)] = totk.get(grp(key), 0.0) + b
    pairs = {key: b * (tot2[grp(key)] / totk[grp(key)]) if level != 2 else b for key, b in cur.items()}
    n = len(events)
    sizes = [len(offs) for _t, offs in plan]
    total_bytes = sum(pairs.values())
    meta = {"realised_mean_degree": 2.0 * len(pairs) / n, "pairs_per_task": len(pairs) / n,
            "connector_pairs": nested["by_level"][level][1], "connector_pairs_level2": nested["by_level"][2][1],
            "share_tasks_in_groups_le_k_plus_1": sum(s for s in sizes if s <= level + 1) / n,
            "share_bytes_in_pairs_gt_50MB": sum(b for b in pairs.values() if b > BIG_PAIR_BYTES) / total_bytes,
            "total_bytes": total_bytes, "total_bytes_level2": sum(tot2.values())}
    return pairs, meta


def consumed_groups(meta: dict) -> Tuple[int, int]:
    """[first, end) library group indices a K=1 mint consumed, from its metadata (the printed grounded_workload_v1 block)."""
    if meta.get("merge_k", 1) != 1 or meta.get("group_size"):
        raise SystemExit("FAIL LOUD: overlap is defined by library group index for K=1 mints only")
    first = int(meta.get("start_group", 0))
    return first, first + int(meta["n_groups"])


def check_disjoint(start_group: int, n_groups: int, source_md5: str, exclude: List[str]) -> None:
    """uncut_groups_s0_v1: a corpus mint shares no request (library group) with the gate windows it names. Groups are
    one per request in arrival order, so the group index is the request; the libraries must come from one source CSV."""
    mine = (start_group, start_group + n_groups)
    for path in exclude:
        other = json.load(open(path))
        other = other.get("grounded_workload_v1", other)
        if other["library_source"]["md5"] != source_md5:
            raise SystemExit(f"FAIL LOUD: {path} was minted from another source CSV; overlap by group index is undefined")
        lo, hi = consumed_groups(other)
        if mine[0] < hi and lo < mine[1]:
            raise SystemExit(f"FAIL LOUD: library groups {mine[0]}..{mine[1] - 1} overlap {path}'s {lo}..{hi - 1}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-tasks", type=int, default=50000)
    ap.add_argument("--merge-k", type=int, default=1)
    ap.add_argument("--group-size", type=int, default=0,
                    help="big_groups_s0_v1: peer groups of exactly this many tasks (0 = the library's own groups, the K=1 mint)")
    ap.add_argument("--start-group", type=int, default=0,
                    help="uncut_groups_s0_v1 corpus: first library group to mint from (0 = today's mint, byte-identical)")
    ap.add_argument("--exclude-mint", nargs="*", default=[],
                    help="mint metadata JSONs of windows this mint must not overlap (by library group index, same source CSV)")
    ap.add_argument("--rate-tol", type=float, default=0.02,
                    help="fail loud when the minted task rate differs from the base window's by more than this fraction")
    ap.add_argument("--payload-sampler", choices=PAYLOAD_SAMPLERS, default="legacy",
                    help="legacy: 200 MB * 10**U(-1,1). wf1_v1: workload_fix_v1 W2 (log-normal 4 MB + heavy tier)")
    ap.add_argument("--partners-nested", type=int, default=0, metavar="K",
                    help="partner_structure_v1: K in 1..8 partners per task from one nested ordered draw (needs --payload-sampler wf1_v1); "
                         "per-group bytes held at the K=2 total. 0 = off, the registered rule, byte-identical")
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"FAIL LOUD: {a.out} exists; a gate workload is frozen once minted")
    lib = json.load(gzip.open(a.lib, "rt") if a.lib.endswith(".gz") else open(a.lib))
    base = json.load(open(a.base))
    if a.merge_k < 1:
        raise SystemExit("FAIL LOUD: --merge-k must be >= 1")
    doc, meta = mint(lib, base, a.seed, a.n_tasks, a.merge_k, a.payload_sampler, a.group_size, a.start_group, a.partners_nested)
    check_disjoint(a.start_group, meta["n_groups"], lib["source"]["md5"], a.exclude_mint)
    if abs(meta["rate_per_s"] / meta["base_rate_per_s"] - 1) > a.rate_tol:
        raise SystemExit(f"FAIL LOUD: minted rate {meta['rate_per_s']:.5f}/s differs from the base {meta['base_rate_per_s']:.5f}/s by more than {a.rate_tol:.0%}")
    doc["grounded_workload_v1"] = {**meta, "library": os.path.basename(a.lib), "library_sha256": _sha(a.lib),
                                   "library_source": lib["source"], "base": os.path.basename(a.base),
                                   "base_sha256": _sha(a.base)}
    tmp = a.out + ".partial"
    with open(tmp, "w") as fh:
        json.dump(doc, fh, separators=(",", ":"))
    os.replace(tmp, a.out)
    print(json.dumps(doc["grounded_workload_v1"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
