#!/usr/bin/env python3
"""Greedy evaluation of labelled checkpoints across cells — dev selection and the gate.

objective_pivot_v1 Phase 3, Amendment D. One script serves both jobs on purpose: the
selection statistic and the verdict statistic must be produced by identical code, or a
configuration can be chosen under one measurement and judged under another.

Everything here runs the **argmax** policy — the configuration the live gates serve. The
sampled decode exists only inside training; nothing that produces a reported number ever
samples.

Arms are given as `label=path` pairs, plus the bare word `knative` for the reactive
baseline (which takes no checkpoint). Every arm sees the same cells and the same trace, so
differences are the policy and nothing else.

    evaluate_policy.py --cells ... --workload ... \
        --arm gnn:frozen=models/gnn-linkmp-lgon-s8.pt \
        --arm gnn:cl_s1=.../cl_gnn_step020.pt \
        --arm knative:knative \
        --out results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts_cosim.closed_loop.episode import ARM_POLICY, TRAINABLE_ARMS, run_episode


def parse_arm(spec: str) -> Dict[str, Any]:
    """`arm:label=path`, or `knative:label` for the checkpoint-free baseline."""
    if ":" not in spec:
        raise SystemExit(f"FAIL LOUD: --arm {spec!r} is not 'arm:label=path'")
    arm, rest = spec.split(":", 1)
    if arm not in ARM_POLICY:
        raise SystemExit(f"FAIL LOUD: unknown arm {arm!r} in {spec!r}")
    if arm in TRAINABLE_ARMS:
        if "=" not in rest:
            raise SystemExit(f"FAIL LOUD: --arm {spec!r} needs a checkpoint (label=path)")
        label, path = rest.split("=", 1)
        model = Path(path)
        if not model.exists():
            raise SystemExit(f"FAIL LOUD: missing checkpoint {model} for arm {label!r}")
        return {"arm": arm, "label": label, "model": model}
    return {"arm": arm, "label": rest or arm, "model": None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-dir", type=Path, required=True)
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--workload", type=Path, required=True)
    ap.add_argument("--arm", dest="arms", action="append", required=True,
                    help="Repeatable. 'gnn:label=path', 'mlp:label=path', or 'knative:label'.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--keep-episode-json", action="store_true")
    args = ap.parse_args()

    arms = [parse_arm(s) for s in args.arms]
    labels = [a["label"] for a in arms]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"FAIL LOUD: duplicate arm labels {labels}; results would overwrite")

    work = args.out.parent / (args.out.stem + "_episodes")
    work.mkdir(parents=True, exist_ok=True)

    jobs = []
    for a in arms:
        for cell in args.cells:
            cfg = args.sweep_dir / "configs" / f"{cell}.json"
            if not cfg.exists():
                raise SystemExit(f"FAIL LOUD: missing cell config {cfg}")
            jobs.append(dict(
                config=cfg, workload=args.workload, model=a["model"],
                out_json=work / f"{a['label']}__{cell}.json",
                cell=cell, arm=a["arm"], temperature=None, timeout_s=args.timeout,
            ))

    def run(job):
        return run_episode(**job)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        results = list(pool.map(run, jobs))

    # `jobs` was built arm-major (every cell of arm 0, then arm 1, ...) and ThreadPoolExecutor
    # .map preserves input order, so results slice back the same way.
    n_cells = len(args.cells)
    rows: List[Dict[str, Any]] = []
    for i, a in enumerate(arms):
        for ep in results[i * n_cells:(i + 1) * n_cells]:
            rows.append({
                "label": a["label"], "arm": a["arm"],
                "model": str(a["model"]) if a["model"] else None,
                "cell": ep.cell, "total_rtt": ep.total_rtt,
                "num_tasks": ep.num_tasks, "wall_s": ep.wall_s,
            })
            print(f"[eval] {a['label']:<24} {ep.cell:<20} total_rtt={ep.total_rtt:,.1f} "
                  f"({ep.wall_s}s)", flush=True)
            if not args.keep_episode_json:
                (work / f"{a['label']}__{ep.cell}.json").unlink(missing_ok=True)

    payload = {
        "workload": str(args.workload),
        "sweep_dir": str(args.sweep_dir),
        "cells": args.cells,
        "decode": "argmax",
        "rows": rows,
    }
    # Per-arm mean across cells, the selection statistic for dev and the per-arm summary
    # for the gate. The gate's own verdict is a PAIRED test over training seeds and lives
    # in analyze_gate.py; a mean over cells is a description, not a claim.
    by_label: Dict[str, List[float]] = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r["total_rtt"])
    payload["mean_total_rtt"] = {k: sum(v) / len(v) for k, v in by_label.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))

    print("\n=== mean total_rtt across cells (lower is better) ===")
    for label, mean in sorted(payload["mean_total_rtt"].items(), key=lambda kv: kv[1]):
        print(f"  {label:<28} {mean:,.1f}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
