#!/usr/bin/env python3
"""Build oversample manifest for merged warmth+sparse+contention_v2 cache.

Dataset keys match prepare_graphs_cache.py: ``{corpus_dir_name}/ds_XXXXX``.
Coupled datasets (greedy regret > threshold) get higher repeat weight.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_lut(ds_dir: Path) -> Optional[Dict[Tuple, float]]:
    jp = ds_dir / "placements" / "placements.jsonl"
    if not jp.is_file() or jp.stat().st_size == 0:
        return None
    lut: Dict[Tuple, float] = {}
    for line in jp.open():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        plan = rec.get("placement_plan")
        rtt = rec.get("rtt")
        if plan is None or rtt is None:
            continue
        key = tuple(sorted((int(k), (int(v[0]), int(v[1]))) for k, v in plan.items()))
        rt = float(rtt)
        if key not in lut or rt < lut[key]:
            lut[key] = rt
    return lut or None


def greedy_regret_rel(lut: Dict[Tuple, float]) -> Optional[float]:
    opt = min(lut.values())
    if opt <= 0:
        return None
    marg: Dict[int, Dict[Tuple[int, int], float]] = defaultdict(lambda: defaultdict(lambda: float("inf")))
    for key, rtt in lut.items():
        for t, pl in key:
            if rtt < marg[t][pl]:
                marg[t][pl] = rtt
    plan = {t: min(dd.items(), key=lambda kv: kv[1])[0] for t, dd in marg.items()}
    k = tuple(sorted(plan.items()))
    if k not in lut:
        return None
    return (lut[k] - opt) / opt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", action="append", required=True, help="Path to gnn_datasets_* dir")
    ap.add_argument("--coupled-threshold", type=float, default=0.01)
    ap.add_argument("--coupled-weight", type=int, default=8)
    ap.add_argument("--base-weight", type=int, default=1)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    weights: Dict[str, int] = {}
    coupled = 0
    total = 0
    for corpus in args.corpus:
        base = Path(corpus)
        if not base.is_dir():
            raise FileNotFoundError(f"Missing corpus: {base}")
        corp_name = base.name
        for ds_dir in sorted(base.glob("ds_*")):
            lut = load_lut(ds_dir)
            if lut is None:
                continue
            ds_id = f"{corp_name}/{ds_dir.name}"
            reg = greedy_regret_rel(lut)
            w = args.base_weight
            if reg is not None and reg > args.coupled_threshold:
                w = args.coupled_weight
                coupled += 1
            weights[ds_id] = w
            total += 1

    payload = {
        "weights": weights,
        "meta": {
            "coupled_threshold": args.coupled_threshold,
            "coupled_weight": args.coupled_weight,
            "base_weight": args.base_weight,
            "n_datasets": total,
            "n_coupled": coupled,
            "n_graph_slots": sum(weights.values()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.output}: {total} ds, {coupled} coupled @>{args.coupled_threshold*100:.0f}%, "
          f"{payload['meta']['n_graph_slots']} graph slots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
