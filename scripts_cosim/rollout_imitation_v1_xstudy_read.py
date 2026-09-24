#!/usr/bin/env python3
"""rollout_imitation_v1 CROSS-STUDY read: put the rollout learned arm and joint_burst_v2 `gnnedge0`
into ONE paired table per study, so the two arms are directly subtractable.

Two studies, each paired per (rung, env) on total_rtt (averageElapsedTime alongside):
  UNBURST (unsaturated_edge_v1, dir results/rollout_imitation_v1_gate):
     focus arms = rollout learned (peer_greedy_learned_network, s0) and gnnedge0 (median over its
     16 seed-checkpoints, arm gnnedge0unburst); baselines = rule / reactive / CD / random.
  BURST (joint_burst_v2, dir results/jb_v2/gate):
     focus arms = rollout learnedbatch (s0) and gnnedge0 (median over seeds, arm jb2_gnnedge0);
     baselines = batched greedy / CD / random / immediate rule.

A multi-seed arm is summarised per env by the MEDIAN over its available seeds; a deterministic arm
is its single s0 value. Each contrast reports the paired median %delta (focus - baseline)/baseline,
the count of envs where the focus arm is faster, and an exact two-sided binomial sign-test p.

Usage (datalab or local after rsync, PYTHONPATH=.):
  python3 scripts_cosim/rollout_imitation_v1_xstudy_read.py \
      --unburst-dir simulation_data/peer_affinity_live_gate/results/rollout_imitation_v1_gate \
      --burst-dir   simulation_data/peer_affinity_live_gate/results/jb_v2/gate \
      --out simulation_data/rollout_imitation_v1/xstudy_read.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from statistics import median
from typing import Dict, Optional, Sequence, Tuple

_SEED_RE = re.compile(r"_s(\d+)$")
METRICS = ("total_rtt", "averageElapsedTime")


def _parse(fname: str) -> Optional[Tuple[str, str, str, int]]:
    """(cell, window, kind, seed) from '<cell>__<window>__<kind>_s<seed>.summary.json'."""
    base = os.path.basename(fname)
    if not base.endswith(".summary.json"):
        return None
    stem = base[: -len(".summary.json")]
    parts = stem.split("__")
    if len(parts) != 3:
        return None
    cell, window, rest = parts
    m = _SEED_RE.search(rest)
    if not m:
        return None
    return cell, window, rest[: m.start()], int(m.group(1))


def _load(gate_dir: str) -> Dict[str, Dict[Tuple[str, str, str], Dict[int, dict]]]:
    """kind -> {(rung, topo, window) -> {seed -> summary}}."""
    out: Dict[str, Dict[Tuple[str, str, str], Dict[int, dict]]] = {}
    for f in sorted(glob.glob(os.path.join(gate_dir, "cc*__*__*.summary.json"))):
        p = _parse(f)
        if p is None:
            continue
        _cell, _window, kind, seed = p
        s = json.load(open(f))
        key = (str(s.get("rung")), str(s.get("topology")), str(s.get("window")))
        out.setdefault(kind, {}).setdefault(key, {})[seed] = s
    return out


def _per_env(arm: Dict[Tuple[str, str, str], Dict[int, dict]], metric: str
             ) -> Dict[Tuple[str, str, str], float]:
    """One value per env: median over seeds of the metric (drops missing)."""
    out = {}
    for key, by_seed in arm.items():
        vals = [float(s[metric]) for s in by_seed.values() if s.get(metric) is not None]
        if vals:
            out[key] = median(vals)
    return out


def _binom_two_sided_p(k: int, n: int) -> Optional[float]:
    if n == 0:
        return None
    kk = min(k, n - k)
    p = 2.0 * sum(math.comb(n, i) for i in range(0, kk + 1)) / (2 ** n)
    return min(1.0, p)


def _contrast(focus: Dict, base: Dict, rung: Optional[str]) -> dict:
    keys = sorted(set(focus) & set(base))
    if rung is not None:
        keys = [k for k in keys if k[0] == rung]
    deltas, faster, n = [], 0, 0
    for k in keys:
        fo, ba = focus[k], base[k]
        if ba == 0:
            continue
        deltas.append(100.0 * (fo - ba) / ba)
        faster += 1 if fo < ba else 0
        n += 1
    return {"n": n, "median_pct": median(deltas) if deltas else None,
            "focus_faster": faster, "sign_p": _binom_two_sided_p(faster, n)}


STUDIES = {
    "unburst_unsaturated_edge_v1": {
        "dir_key": "unburst",
        "rungs": ["C40", "C80"],
        "focus": {"rollout_learned": "peer_greedy_learned_network", "gnnedge0": "gnnedge0unburst"},
        "baselines": {"rule": "peer_greedy_network", "reactive": "knative_network_ect",
                      "cd_greedy": "peer_greedy_network_cd", "random": "random_network"},
    },
    "burst_joint_burst_v2": {
        "dir_key": "burst",
        "rungs": ["C40"],
        "focus": {"rollout_learnedbatch": "learnedbatch", "gnnedge0": "jb2_gnnedge0"},
        "baselines": {"batched_greedy": "peer_greedy_network_batch", "cd_greedy": "peer_greedy_network_cd",
                      "random": "random_network", "immediate_rule": "peer_greedy_network"},
    },
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--unburst-dir", required=True)
    ap.add_argument("--burst-dir", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    loaded = {"unburst": _load(a.unburst_dir), "burst": _load(a.burst_dir)}
    report = {}
    for study, cfg in STUDIES.items():
        arms = loaded[cfg["dir_key"]]
        report[study] = {}
        print(f"\n=== {study} ===")
        for metric in METRICS:
            focus_env = {name: _per_env(arms.get(kind, {}), metric) for name, kind in cfg["focus"].items()}
            base_env = {name: _per_env(arms.get(kind, {}), metric) for name, kind in cfg["baselines"].items()}
            for rung in cfg["rungs"]:
                for fname, fmap in focus_env.items():
                    for bname, bmap in base_env.items():
                        c = _contrast(fmap, bmap, rung)
                        report[study].setdefault(metric, {}).setdefault(rung, {}).setdefault(fname, {})[bname] = c
                        if metric == METRICS[0]:
                            mp = c["median_pct"]
                            print(f"  [{rung}] {fname:22s} vs {bname:16s} n={c['n']:2d} "
                                  f"median={mp if mp is None else round(mp,2)}% "
                                  f"faster={c['focus_faster']}/{c['n']} "
                                  f"p={c['sign_p'] if c['sign_p'] is None else round(c['sign_p'],4)}")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(report, open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
