#!/usr/bin/env python3
"""Phase 3 Increment 1 — the registered sampling-feasibility probe.

Registered in `docs/lineages/objective_pivot_v1.md` (Phase 3 registration, 2026-09-01)
BEFORE this script existed. It measures the two quantities that decide whether a
policy-gradient loop is affordable at all, and nothing else:

  1. exploration cost  d(T) = (sampled RTT - argmax RTT) / argmax RTT, per temperature
  2. per-episode noise sd  = paired std-dev of sampled episodes at fixed T and trace

Reading (fixed before the run, reproduced here so the verdict cannot drift):
  GO            some T has d(T) <= 0.10 AND sd <= 0.05
  NO-GO         every T has d(T) > 0.25, OR sd > 0.15
  INDETERMINATE anything else -- returns the numbers to the user, NOT resolved by
                picking a T after seeing the outcome.

On GO the pilot size is n >= (2 * sd * 2.8 / 0.03)^2 for a minimum detectable effect
of 3%, reported alongside.

Each episode is one full run of `executesimulation.py` on the inner-loop trace, so the
probe reuses the serving path exactly -- no separate inference code that could drift
from what the gates run.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

PYTHON = os.environ.get("HEROSIM_PY", "python3")


def run_episode(
    *,
    config: Path,
    workload: Path,
    model: Path,
    out_json: Path,
    temperature: Optional[float],
    seed: Optional[int],
    timeout_s: int,
) -> Dict[str, Any]:
    """One full live episode. temperature=None => the deterministic argmax policy."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["PYTHONHASHSEED"] = "0"
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
    env["HEROSIM_WARMTH_PHYSICS"] = "node_disk_v2"
    env["GNN_BATCH_SIZE"] = "4"
    env["GNN_BATCH_TIMEOUT"] = "0.002"
    env["GNN_MODEL_PATH"] = str(model)
    # INFERENCE_FEATURE_LAYOUT / NETWORK_GRAPH_CONTRACT / QUEUE_FEATURE_CONTRACT are
    # deliberately NOT exported: load_gnn_model adopts them from the checkpoint's
    # .contract.json and raises on a conflicting export.
    if temperature is None:
        env["GNN_DECODE_MODE"] = "argmax"
        env.pop("GNN_SAMPLE_TEMPERATURE", None)
        env.pop("GNN_SAMPLE_SEED", None)
    else:
        env["GNN_DECODE_MODE"] = "sample"
        env["GNN_SAMPLE_TEMPERATURE"] = str(temperature)
        env["GNN_SAMPLE_SEED"] = str(seed)

    cmd = [
        *PYTHON.split(),
        "src/executesimulation.py",
        "--policy", "gnn",
        "--config", str(config),
        "--workload", str(workload),
        "--output", str(out_json),
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True,
        timeout=timeout_s,
    )
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-2500:]
        raise RuntimeError(
            f"FAIL LOUD: episode failed (rc={proc.returncode})\ncmd: {' '.join(cmd)}\n{tail}"
        )
    if not out_json.exists():
        raise RuntimeError(f"FAIL LOUD: episode wrote no result at {out_json}")
    # Result JSONs are large; total_rtt/num_tasks sit near the top, but parse the whole
    # document -- a truncated file whose PREFIX parses is a known failure mode here.
    with open(out_json) as f:
        res = json.load(f)
    rtt = res.get("total_rtt")
    n = res.get("num_tasks")
    if rtt is None or not n:
        raise RuntimeError(f"FAIL LOUD: {out_json} has no total_rtt/num_tasks")
    return {"total_rtt": float(rtt), "num_tasks": int(n), "wall_s": round(elapsed, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep-dir", type=Path, required=True,
                    help="A minted cell sweep dir (configs/ + cell_infrastructure/)")
    ap.add_argument("--cells", nargs="+", required=True, help="Cell names, >= 3 as registered")
    ap.add_argument("--model", type=Path,
                    default=Path("models/gnn-linkmp-lgon-s8.pt"))
    ap.add_argument("--workload", type=Path,
                    default=Path("data/nofs-ids/traces/workload-150-100-30k.json"))
    ap.add_argument("--temperatures", type=float, nargs="+", default=[0.1, 0.3, 1.0],
                    help="Registered: 0.1 0.3 1.0")
    ap.add_argument("--noise-seeds", type=int, default=5,
                    help="Sampled episodes per (cell, T) for the paired sd")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--timeout", type=int, default=7200)
    args = ap.parse_args()

    if len(args.cells) < 3:
        raise SystemExit("FAIL LOUD: the registration requires >= 3 backbone cells")
    if not args.model.exists():
        raise SystemExit(f"FAIL LOUD: missing model {args.model}")
    sidecar = args.model.with_suffix(".contract.json")
    if not sidecar.exists():
        raise SystemExit(
            f"FAIL LOUD: {args.model} has no .contract.json — a checkpoint without a "
            "sidecar is not evidence."
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir = args.out_dir / "episodes"
    episodes_dir.mkdir(exist_ok=True)

    records: List[Dict[str, Any]] = []
    for cell in args.cells:
        config = args.sweep_dir / "configs" / f"{cell}.json"
        if not config.exists():
            raise SystemExit(f"FAIL LOUD: missing cell config {config}")

        base = run_episode(
            config=config, workload=args.workload, model=args.model,
            out_json=episodes_dir / f"{cell}__argmax.json",
            temperature=None, seed=None, timeout_s=args.timeout,
        )
        print(f"[probe] {cell} argmax: total_rtt={base['total_rtt']:.1f} "
              f"tasks={base['num_tasks']} ({base['wall_s']}s)", flush=True)
        records.append({"cell": cell, "arm": "argmax", "temperature": None,
                        "seed": None, **base})

        for temp in args.temperatures:
            for seed in range(1, args.noise_seeds + 1):
                ep = run_episode(
                    config=config, workload=args.workload, model=args.model,
                    out_json=episodes_dir / f"{cell}__T{temp}__s{seed}.json",
                    temperature=temp, seed=seed, timeout_s=args.timeout,
                )
                rel = (ep["total_rtt"] - base["total_rtt"]) / base["total_rtt"]
                print(f"[probe] {cell} T={temp} seed={seed}: total_rtt={ep['total_rtt']:.1f} "
                      f"rel_vs_argmax={rel:+.4f} ({ep['wall_s']}s)", flush=True)
                records.append({"cell": cell, "arm": "sample", "temperature": temp,
                                "seed": seed, "rel_vs_argmax": rel, **ep})
                (args.out_dir / "records.json").write_text(json.dumps(records, indent=1))

    # ---- registered readout ----
    summary: Dict[str, Any] = {"per_temperature": {}, "model": str(args.model),
                               "workload": str(args.workload), "cells": args.cells}
    for temp in args.temperatures:
        rels = [r["rel_vs_argmax"] for r in records
                if r["arm"] == "sample" and r["temperature"] == temp]
        # sd is computed WITHIN cell (episodes share the trace and cell => paired),
        # then pooled: across-cell spread is a property of the cells, not the policy.
        sds = []
        for cell in args.cells:
            vals = [r["total_rtt"] for r in records
                    if r["arm"] == "sample" and r["temperature"] == temp and r["cell"] == cell]
            b = next(r["total_rtt"] for r in records
                     if r["arm"] == "argmax" and r["cell"] == cell)
            if len(vals) >= 2:
                sds.append(st.stdev([v / b for v in vals]))
        summary["per_temperature"][str(temp)] = {
            "d_mean": st.mean(rels) if rels else None,
            "d_median": st.median(rels) if rels else None,
            "sd_within_cell_pooled": st.mean(sds) if sds else None,
            "n_episodes": len(rels),
        }

    go, nogo = [], []
    for temp, v in summary["per_temperature"].items():
        d, sd = v["d_mean"], v["sd_within_cell_pooled"]
        if d is None or sd is None:
            continue
        if d <= 0.10 and sd <= 0.05:
            go.append((temp, d, sd))
        if d > 0.25:
            nogo.append(temp)
    worst_sd = max((v["sd_within_cell_pooled"] or 0) for v in summary["per_temperature"].values())

    if go:
        temp, d, sd = min(go, key=lambda x: x[1])
        n = (2 * sd * 2.8 / 0.03) ** 2
        summary["verdict"] = "GO"
        summary["chosen_temperature"] = temp
        summary["required_n_paired_episodes"] = int(n) + 1
    elif len(nogo) == len(summary["per_temperature"]) or worst_sd > 0.15:
        summary["verdict"] = "NO-GO"
    else:
        summary["verdict"] = "INDETERMINATE"

    (args.out_dir / "probe_summary.json").write_text(json.dumps(summary, indent=1))
    print("\n=== REGISTERED PROBE READOUT ===")
    for temp, v in summary["per_temperature"].items():
        print(f"  T={temp:>5}  d_mean={v['d_mean']:+.4f}  d_median={v['d_median']:+.4f}  "
              f"sd={v['sd_within_cell_pooled']:.4f}  (n={v['n_episodes']})")
    print(f"  VERDICT: {summary['verdict']}"
          + (f"  T={summary['chosen_temperature']}, "
             f"n>={summary['required_n_paired_episodes']} paired episodes"
             if summary["verdict"] == "GO" else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
