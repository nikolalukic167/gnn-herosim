#!/usr/bin/env python3
"""cd_gap_v1 B -- build the lighter-rate input set for the burst load ladder (docs/lineages/archive/cd_gap_v1.md).

Every event timestamp of each burst window workload is multiplied by --factor (groups stay dispatched
together: members share a timestamp, and a product of equal numbers is equal). Every policy time
constant scales by the same factor (the `drainable_regime_v1` hard-stop rule): the cells'
`scheduler.batch_timeout`. Checkpoints and the jb2 split are linked, not copied. The output keeps the
source file names so `fresh_topo_burst_v1_gate.py` runs unchanged against --out as its --inputs.

  cd_gap_v1_build_b.py --src <fresh inputs> --out <dir> --factor 2 --topologies 9119 9414 ...

warm_surge_s0_v1 (docs/lineages/warm_surge_s0_v1.md): `--surge-at-s T --surge-k K` is a different mode. It leaves the factor path alone and
post-passes ONE frozen rung's workload files (--src is that rung's directory, holding wl/ and cfg/): peer groups whose earliest member
is at or after T are compressed toward T by K, everything earlier is untouched, and the cfg files are copied byte for byte.

  cd_gap_v1_build_b.py --src <inputs>/wf1_heavy --out <inputs'>/wf1_heavyk2 --surge-at-s 1200 --surge-k 2 --wl-name grounded_g0_n50000
"""
from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import os
import shutil
import sys


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def surge_after(wl: dict, t_switch_s: float, k: float) -> dict:
    """warm_surge_s0_v1: compress the arrivals after t_switch_s by k, group-atomically; k = 1 is the identity.

    A peer group is anchored at its earliest member. A group anchored before t_switch_s is unchanged; any other group has every member
    moved to t_s + (t - t_s) / k. Event order is kept (peer_exchange indexes events), so a non-monotone result fails loud.
    """
    if k < 1:
        raise SystemExit("FAIL LOUD: --surge-k must be >= 1")
    ev = wl["events"]
    ts = [float(e["timestamp"]) for e in ev]
    anchor = {}
    for e, t in zip(ev, ts):
        g = e.get("peer_group")
        anchor[g] = min(anchor.get(g, t), t)
    out = copy.deepcopy(wl)
    n_pre = n_post = 0
    for e, t in zip(out["events"], ts):
        if anchor[e.get("peer_group")] < t_switch_s:
            n_pre += 1
        else:
            e["timestamp"] = t_switch_s + (t - t_switch_s) / k
            n_post += 1
    new = [float(e["timestamp"]) for e in out["events"]]
    bad = sum(1 for a, b in zip(new, new[1:]) if b < a)
    if bad:
        raise SystemExit(f"FAIL LOUD: surge_after broke the event order at {bad} places (a group straddles the switch)")
    last = {}
    for e, t in zip(ev, ts):
        last[e.get("peer_group")] = max(last.get(e.get("peer_group"), t), t)
    straddle = sum(1 for g, a0 in anchor.items() if a0 < t_switch_s <= last[g])
    out["duration"] = float(wl["duration"]) if k == 1 else t_switch_s + (float(wl["duration"]) - t_switch_s) / k
    out["warm_surge_s0_v1"] = {"t_switch_s": t_switch_s, "k": k, "n_pre": n_pre, "n_post": n_post, "straddling_groups": straddle}
    return out


def main_surge(a) -> int:
    for sub in ("wl", "cfg"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    names = [a.wl_name + ".json"] if a.wl_name else [os.path.basename(p) for p in sorted(glob.glob(os.path.join(a.src, "wl", "*.json")))]
    for name in names:
        path = os.path.join(a.src, "wl", name)
        out = os.path.join(a.out, "wl", name)
        if a.surge_k == 1.0:
            shutil.copyfile(path, out + ".partial")
            os.replace(out + ".partial", out)
            print(f"wl {name}: k = 1, byte copy")
            continue
        wl = surge_after(json.load(open(path)), a.surge_at_s, a.surge_k)
        wl["warm_surge_s0_v1"]["source_sha256"] = _sha(path)
        json.dump(wl, open(out + ".partial", "w"))
        os.replace(out + ".partial", out)
        print(f"wl {name}: {wl['warm_surge_s0_v1']}")
    for path in sorted(glob.glob(os.path.join(a.src, "cfg", "*.json"))):
        shutil.copyfile(path, os.path.join(a.out, "cfg", os.path.basename(path)))
    print(f"cfg: {len(glob.glob(os.path.join(a.src, 'cfg', '*.json')))} files copied unchanged")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--factor", type=float)
    ap.add_argument("--topologies", type=int, nargs="+")
    ap.add_argument("--surge-at-s", type=float, default=None)
    ap.add_argument("--surge-k", type=float, default=None)
    ap.add_argument("--wl-name", default=None, help="surge mode: one workload file stem (default: every wl/*.json)")
    a = ap.parse_args()
    if (a.surge_at_s is None) != (a.surge_k is None):
        raise SystemExit("FAIL LOUD: --surge-at-s and --surge-k go together")
    if a.surge_at_s is not None:
        if a.factor is not None or a.topologies:
            raise SystemExit("FAIL LOUD: surge mode takes neither --factor nor --topologies")
        return main_surge(a)
    if a.factor is None or not a.topologies:
        raise SystemExit("FAIL LOUD: --factor and --topologies are required")
    if a.factor <= 0:
        raise SystemExit("FAIL LOUD: --factor must be > 0")
    for sub in ("cfg", "wl"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    for name in ("models", "joint_burst_v2_split.json"):
        dst = os.path.join(a.out, name)
        if not os.path.lexists(dst):
            os.symlink(os.path.abspath(os.path.join(a.src, name)), dst)
    for path in sorted(glob.glob(os.path.join(a.src, "wl", "*.json"))):
        wl = json.load(open(path))
        before = [float(e["timestamp"]) for e in wl["events"]]
        for e in wl["events"]:
            e["timestamp"] = float(e["timestamp"]) * a.factor
        wl["duration"] = float(wl["duration"]) * a.factor
        wl["cd_gap_v1_rate_scale"] = {"factor": a.factor, "source": os.path.basename(path), "source_sha256": _sha(path),
                                      "span_before_s": max(before) - min(before),
                                      "span_after_s": (max(before) - min(before)) * a.factor}
        groups = {}
        for e, t0 in zip(wl["events"], before):
            groups.setdefault(e.get("peer_group"), set()).add(e["timestamp"])
        spread = max(len(v) for v in groups.values())
        out = os.path.join(a.out, "wl", os.path.basename(path))
        json.dump(wl, open(out + ".partial", "w"))
        os.replace(out + ".partial", out)
        print(f"wl {os.path.basename(path)}: {len(before)} events, span {max(before)-min(before):.0f} -> "
              f"{(max(before)-min(before))*a.factor:.0f} s, max distinct timestamps per group {spread}")
    for t in a.topologies:
        src = os.path.join(a.src, "cfg", f"cc40s{t}.json")
        cfg = json.load(open(src))
        new = copy.deepcopy(cfg)
        old_to = float(new["scheduler"]["batch_timeout"])
        new["scheduler"]["batch_timeout"] = old_to * a.factor
        new.setdefault("cd_gap_v1_rate_scale", {})
        new["cd_gap_v1_rate_scale"] = {"factor": a.factor, "source_sha256": _sha(src),
                                       "batch_timeout_before": old_to, "batch_timeout_after": old_to * a.factor}
        json.dump(new, open(os.path.join(a.out, "cfg", f"cc40s{t}.json"), "w"), indent=1)
    print(f"cfg: {len(a.topologies)} cells, batch_timeout x{a.factor}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
