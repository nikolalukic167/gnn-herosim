#!/usr/bin/env python3
"""r1_attribution_v1: choose each arm's configuration on VALIDATION topologies only and stage its checkpoints for the gate.

Registered rule (docs/lineages/r1_attribution_v1.md, "Fair-comparison budget"): every arm has the same 6-configuration grid and 3 training seeds; the
configuration per arm is chosen on validation topologies only, and the best learned arm is declared from validation before any test read.

For each arm this reads models/r1-attribution-v1-<arm>-g<k>-seed<s>.val.json (written by train_near_rtt.py: the validation score of the saved checkpoint,
regret_masked_topo seconds on the held-out-by-topology validation split), requires all 6 configs x 3 seeds, takes the configuration with the lowest MEAN
validation score over seeds (ties: the lower config index), and copies that configuration's three checkpoints and sidecars to
<inputs>/models/r1-attribution-v1-<arm>-seed<s>.pt (the names fresh_topo_burst_v1_gate.py reads for the ra_* kinds). A run that was scored on the held-out
topologies (test_evaluated) is refused: the choice must precede any test read. The report names the best learned arm by validation.

  r1a_select_checkpoints.py --models-dir ~/gnn-herosim/models --inputs-dir <gate inputs> --split experiments/r1_attribution_v1_split.json --out selection.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts_cosim"))

ARMS = ("gnn_eng", "twin_eng", "mlp_same", "gnn_raw", "twin_raw", "gnn_eng_physmp", "set_transformer")
CONFIGS = range(6)
SEEDS = (1, 2, 3)
DEFAULT_PREFIX = "r1-attribution-v1"
PREFIX = DEFAULT_PREFIX  # the lineage's run-name prefix (--prefix); a second lineage must not collide in the shared models dir


def run_stem(arm: str, k: int) -> str:
    return f"{PREFIX}-{arm.replace('_', '-')}-g{k}"


class Incomplete(Exception):
    pass


def load_scores(models: Path, arm: str, *, allow_incomplete: bool = False) -> Dict[int, Dict[int, Dict[str, Any]]]:
    out: Dict[int, Dict[int, Dict[str, Any]]] = {}
    missing: List[str] = []
    for k in CONFIGS:
        for s in SEEDS:
            p = models / f"{run_stem(arm, k)}-seed{s}.val.json"
            if not p.is_file():
                missing.append(p.name)
                continue
            v = json.loads(p.read_text())
            if v.get("test_evaluated"):
                raise SystemExit(f"FAIL LOUD: {p.name} was scored on the held-out topologies; the selection must precede any test read")
            out.setdefault(k, {})[s] = v
    if missing:
        if allow_incomplete:
            raise Incomplete(f"{arm}: {len(missing)} of 18 runs have no validation record yet")
        raise SystemExit(f"FAIL LOUD: {arm}: {len(missing)} of 18 runs have no validation record: {missing[:4]}...")
    return out


def stage(src: Path, dst: Path) -> None:
    """Copy through a temporary name and rename, so the gate never reads a half-written checkpoint."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def main() -> int:
    global PREFIX
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", type=Path, required=True)
    ap.add_argument("--inputs-dir", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--arm", action="append", default=None, help="per-arm run: the same rule on one arm (repeatable); same picks as the all-arms run")
    ap.add_argument("--if-complete", action="store_true", help="exit 3 quietly when an arm still lacks validation records")
    ap.add_argument("--prefix", default=DEFAULT_PREFIX, help="run-name prefix of the lineage (scale_160_v1: scale-160-v1)")
    a = ap.parse_args()
    PREFIX = a.prefix
    a.arms = (a.arm or []) + (a.arms or []) or list(ARMS)
    import joint_burst_v2_sidecheck as sc

    split_sha = hashlib.sha256(a.split.read_bytes()).hexdigest()
    dest = a.inputs_dir / "models"
    chosen: Dict[str, Any] = {}
    plan: List[tuple] = []
    for arm in a.arms:
        try:
            scores = load_scores(a.models_dir, arm, allow_incomplete=a.if_complete)
        except Incomplete as exc:
            print(f"incomplete: {exc}")
            return 3
        table = {k: {"seeds": {s: scores[k][s]["best_val"] for s in SEEDS},
                     "mean": statistics.fmean(scores[k][s]["best_val"] for s in SEEDS),
                     "sd": statistics.pstdev(scores[k][s]["best_val"] for s in SEEDS),
                     "metric": scores[k][1]["checkpoint_metric"]} for k in CONFIGS}
        if len({t["metric"] for t in table.values()}) != 1:
            raise SystemExit(f"FAIL LOUD: {arm}: configurations were selected on different checkpoint metrics")
        best = min(CONFIGS, key=lambda k: (table[k]["mean"], k))
        chosen[arm] = {"config": best, "val_mean": table[best]["mean"], "val_sd": table[best]["sd"], "metric": table[best]["metric"], "table": table}
        for s in SEEDS:
            src = a.models_dir / f"{run_stem(arm, best)}-seed{s}.pt"
            side = json.loads(src.with_suffix(".contract.json").read_text())
            if (side.get("split_artifact") or {}).get("sha256") != split_sha:
                raise SystemExit(f"FAIL LOUD: {src.name} was trained on a different split than {a.split}")
            if sc.check_ra(side, f"ra_{arm}", str(a.split), "inf"):
                raise SystemExit(f"FAIL LOUD: sidecheck failed for {src.name}")
            plan.append((src, dest / f"{PREFIX}-{arm.replace('_', '-')}-seed{s}.pt"))
    best_arm = min(chosen, key=lambda r: (chosen[r]["val_mean"], r))
    # a per-arm run and the all-arms run apply one rule; if both exist their picks must agree
    for arm, c in chosen.items():
        prior = a.out.parent / f"selection_{arm}.json"
        if prior.is_file() and a.out.name != prior.name:
            before = json.loads(prior.read_text())["arms"][arm]["config"]
            if before != c["config"]:
                raise SystemExit(f"FAIL LOUD: {arm}: the per-arm selection picked g{before}, this run picks g{c['config']}")
    dest.mkdir(parents=True, exist_ok=True)
    for src, dst in plan:
        for suffix in (".pt", ".contract.json"):
            stage(src.with_suffix(suffix), dst.with_suffix(suffix))
    report = {"split_sha256": split_sha, "rule": "lowest mean validation score over 3 seeds; validation topologies only; no test read",
              "best_learned_arm_by_validation": best_arm, "arms": chosen}
    a.out.write_text(json.dumps(report, indent=1))
    print(f"best learned arm by validation: {best_arm} ({chosen[best_arm]['val_mean']:.4f} {chosen[best_arm]['metric']})")
    for arm, c in chosen.items():
        print(f"  {arm:16s} config g{c['config']}  val {c['val_mean']:.4f} +- {c['val_sd']:.4f}")
    print(f"staged {len(plan)} checkpoints under {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
