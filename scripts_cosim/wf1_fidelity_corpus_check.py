#!/usr/bin/env python3
"""Acceptance check for a fidelity-mode warm corpus (make_warm_corpus --fidelity), r1_attribution_v1.

For every dataset: the sweep is complete (every enumerated plan has a row, none lost), the sidecar records the fidelity flag
and live_run_params, best.json's label is the minimum of the rows, optimal_result's tasks are the batch alone numbered
0..k-1, and a sample of plans re-simulated by the INDEPENDENT harness (wf1_fidelity_cost.py, written against
physics_audit/i11_replay.py rather than against the corpus code) gives the label the sweep recorded.

  wf1_fidelity_corpus_check.py --datasets <dir> [--sample 25]
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=Path, required=True)
    ap.add_argument("--sample", type=int, default=25)
    a = ap.parse_args()
    if os.environ.get("HEROSIM_SNAPSHOT_FIDELITY") != "1":
        raise SystemExit("FAIL LOUD: export HEROSIM_SNAPSHOT_FIDELITY=1 and the R1 environment")
    import wf1_fidelity_cost as W
    from src.executesimulation import prepare_infrastructure_for_real_simulation

    bad = 0
    rng = random.Random(3)
    for d in sorted(a.datasets.glob("ds_*")):
        problems = []
        meta = json.loads((d / "placement_metadata.json").read_text())
        rows = [json.loads(l) for l in open(d / "placements" / "placements.jsonl") if l.strip()]
        if not meta.get("sweep_complete") or len(rows) != meta["num_placements"] or meta.get("worker_failed"):
            problems.append(f"sweep incomplete: {len(rows)} rows of {meta['num_placements']}, failed {meta.get('worker_failed')}")
        side = json.loads((d / "fidelity_replay.json").read_text())
        if not side.get("fidelity") or "live_run_params" not in side:
            problems.append("sidecar lacks the fidelity flag or live_run_params")
        if "fidelity_replay" not in meta:
            problems.append("placement_metadata.json does not record the fidelity replay")
        best = json.loads((d / "best.json").read_text())["rtt"]
        if abs(best - min(r["rtt"] for r in rows)) > 1e-9 * max(1.0, best):
            problems.append(f"best.json {best} != min(rows) {min(r['rtt'] for r in rows)}")
        opt = json.loads((d / "optimal_result.json").read_text())
        ids = sorted(tr["taskId"] for tr in opt["stats"]["taskResults"] if tr.get("taskId", -1) >= 0)
        wl = json.loads((d / "workload.json").read_text())
        if ids != list(range(len(wl["events"]))):
            problems.append(f"optimal_result tasks {ids} are not the batch 0..{len(wl['events']) - 1}")
        # independent re-simulation
        infra = json.loads((d / "infrastructure.json").read_text())
        spec = infra["live_snapshot_seed"]["fidelity_replay"]
        W._CTX[d.name] = {
            "snapshot": spec["snapshot"],
            "base_infra": prepare_infrastructure_for_real_simulation(json.loads(Path(spec["cell_config"]).read_text()),
                                                                     seed=None, sim_input_path=Path(spec["sim_input"])),
            "sim_input": spec["sim_input"],
            "dataset_types": [next(iter(e["application"]["dag"])) for e in wl["events"]],
        }
        pick = rows if len(rows) <= a.sample else rng.sample(rows, a.sample)
        worst = 0.0
        for r in pick:
            plan = {int(k): tuple(v) for k, v in r["placement_plan"].items()}
            out = W._one_plan({"dataset": d.name, "plan_index": -1, "plan": plan})
            if "error" in out:
                problems.append(f"independent replay failed: {out['error'][:120]}")
                break
            worst = max(worst, abs(out["batch_latency"] - r["rtt"]))
        if worst > 1e-9:
            problems.append(f"independent replay differs from the sweep's label by up to {worst:.3e}")
        print(f"{d.name}: {len(rows)} rows, best {best:.3f}, {len(pick)} plans re-simulated, max |diff| {worst:.2e}"
              + ("" if not problems else "  PROBLEMS: " + "; ".join(problems)))
        bad += bool(problems)
    print(f"datasets with problems: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
