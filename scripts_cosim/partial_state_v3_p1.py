"""partial_state_v3 P1 -- the offline tie, v3 vs v2 held-out regret paired by training seed.

Runs on datalab, where the W&B run directories and the SLURM logs live:

    python3 scripts_cosim/partial_state_v3_p1.py --v3-job 769631 --v2-job 766897 \
        --cache simulation_data/graphs_cache_drainable_objective_v1_v1_psv3 \
        --split experiments/peer_affinity_v1_t1b_split.json \
        --out simulation_data/partial_state_v3/p1.json

Each training task's log names its arm and seed (`[TASK t/n] arm=gnn seed=4 ...`) and its
W&B run (`.../runs/<id>`); the run's summary carries `final/test/regret_masked_topo` in
seconds at the SELECTED checkpoint (the trainer reloads the best weights before the final
eval). The statistic is that regret as a percentage of the held-out split's mean optimal
RTT -- the same denominator for both representations, which P0 guarantees is identical
between the two caches -- so the paired difference is in percentage points, as registered.
Ordering only: P1 never closes the lineage (rule 6).
"""
from __future__ import annotations

import argparse
import glob
import json
import pickle
import re
import statistics as st
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

from scripts_cosim.partial_state_v3_read import read_p1

TASK_RE = re.compile(r"^\[TASK \d+/\d+\].*\barm=(?P<arm>gnn|mpoff|peeronly)\b.*\bseed=(?P<seed>\d+)\b")
RUN_RE = re.compile(r"https://wandb\.ai/\S+/runs/(?P<id>[A-Za-z0-9]+)")
REGRET_KEY = "final/test/regret_masked_topo"


def parse_task_log(text: str) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """(arm, seed, wandb run id) from one task's .out, any of them None when absent."""
    arm = seed = run_id = None
    for line in text.splitlines():
        m = TASK_RE.match(line)
        if m and arm is None:
            arm, seed = m.group("arm"), int(m.group("seed"))
        r = RUN_RE.search(line)
        if r:
            run_id = r.group("id")
    return arm, seed, run_id


def mean_test_opt_rtt(cache: Path, split: Path) -> float:
    opt = pickle.load(open(cache / "optimal_rtt.pkl", "rb"))
    test_ids = list(json.load(open(split)).get("test") or [])
    if not test_ids:
        raise ValueError(f"{split} has no test ids")
    missing = [i for i in test_ids if i not in opt]
    if missing:
        raise ValueError(f"{len(missing)} test ids absent from {cache}/optimal_rtt.pkl, e.g. {missing[:2]}")
    return float(st.fmean(float(opt[i]) for i in test_ids))


def collect(job: int, prefix: str, tasks: Sequence[int], denom: float, *, logs: Path, wandb_dir: Path) -> Dict[str, Dict[int, dict]]:
    from scripts_cosim.read_training_curves import _read_wandb_dir  # heavy import, kept local
    out: Dict[str, Dict[int, dict]] = {}
    for t in tasks:
        log = logs / f"{prefix}-{job}_{t}.out"
        if not log.is_file():
            raise FileNotFoundError(f"FAIL LOUD: {log} missing")
        # The task line is on stdout; W&B prints its run URL on STDERR. Read both.
        err = log.with_suffix(".err")
        text = log.read_text(errors="replace") + ("\n" + err.read_text(errors="replace") if err.is_file() else "")
        arm, seed, run_id = parse_task_log(text)
        if arm is None or run_id is None:
            raise ValueError(f"FAIL LOUD: {log} names no arm/seed or no W&B run")
        runs = sorted(glob.glob(str(wandb_dir / f"*run-*-{run_id}")))
        if len(runs) != 1:
            raise ValueError(f"FAIL LOUD: {len(runs)} wandb dirs for run {run_id} ({log})")
        rows, summ, _cfg = _read_wandb_dir(Path(runs[0]))
        if REGRET_KEY not in summ:
            raise KeyError(f"FAIL LOUD: {REGRET_KEY} absent from {runs[0]} summary")
        hist = [x["val/regret_masked_topo"] for x in rows if "val/regret_masked_topo" in x]
        out.setdefault(arm, {})[seed] = {
            "regret_s": float(summ[REGRET_KEY]),
            "regret_pct": 100.0 * float(summ[REGRET_KEY]) / denom,
            "selected_epoch": (hist.index(min(hist)) if hist else None),
            "epochs_run": len(hist),
            "stop_reason": summ.get("early_stop/reason"),
            "run": Path(runs[0]).name,
        }
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3-job", type=int, required=True)
    ap.add_argument("--v2-job", type=int, required=True)
    ap.add_argument("--v3-prefix", default="psv3-train")
    ap.add_argument("--v2-prefix", default="dobj-train")
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--logs", type=Path, default=Path("logs"))
    ap.add_argument("--wandb-dir", type=Path, default=Path("wandb"))
    ap.add_argument("--out", type=Path, required=True)
    # peer_only_v1 A1 reuses this read: the "v3" side is any job/task range, and each
    # comparison names (arm in the v3 job):(arm in the v2 job), e.g. peeronly:gnn.
    ap.add_argument("--v3-tasks", default="0-31", help="task index range in the v3 job")
    ap.add_argument("--v2-tasks", default="0-31", help="task index range in the v2 job")
    ap.add_argument("--compare", default="gnn:gnn,mpoff:mpoff",
                    help="comma-separated v3arm:v2arm pairs to read")
    args = ap.parse_args(argv)

    def _range(spec: str) -> range:
        lo, hi = spec.split("-")
        return range(int(lo), int(hi) + 1)

    denom = mean_test_opt_rtt(args.cache, args.split)
    v3 = collect(args.v3_job, args.v3_prefix, _range(args.v3_tasks), denom, logs=args.logs, wandb_dir=args.wandb_dir)
    v2 = collect(args.v2_job, args.v2_prefix, _range(args.v2_tasks), denom, logs=args.logs, wandb_dir=args.wandb_dir)
    res = {"denominator_mean_test_opt_rtt_s": denom, "arms": {}}
    print(f"P1 -- offline tie, {args.v3_prefix} {args.v3_job} vs {args.v2_prefix} {args.v2_job}, held-out regret at "
          f"the selected checkpoint (% of mean test optimal RTT {denom:.2f} s), paired by training seed")
    for pair in args.compare.split(","):
        a3, a2 = pair.split(":")
        if a3 not in v3 or a2 not in v2:
            raise KeyError(f"FAIL LOUD: comparison {pair}: arms present are v3={sorted(v3)} v2={sorted(v2)}")
        r = read_p1({s: d["regret_pct"] for s, d in v3[a3].items()},
                    {s: d["regret_pct"] for s, d in v2[a2].items()})
        sel3 = [d["selected_epoch"] for d in v3[a3].values()]
        sel2 = [d["selected_epoch"] for d in v2[a2].values()]
        stops = sum(1 for d in v3[a3].values() if d["stop_reason"] == "patience")
        res["arms"][pair] = {"read": r, "v3": v3[a3], "v2": v2[a2]}
        med3 = st.median(d["regret_pct"] for d in v3[a3].values())
        med2 = st.median(d["regret_pct"] for d in v2[a2].values())
        print(f"  {a3:8s} vs {a2:5s}  {med3:6.2f}% vs {med2:6.2f}%  paired median {r.get('median', float('nan')):+.2f} pp  "
              f"p={r.get('p', float('nan')):.4f}  ahead {r.get('v3_ahead')}/{r.get('n')}  -> {r['verdict']}")
        print(f"        selected epoch {a3} {min(sel3)}-{max(sel3)} (median {st.median(sel3):.0f}), "
              f"{a2} {min(sel2)}-{max(sel2)} (median {st.median(sel2):.0f}); patience stops {stops}/{len(sel3)}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"[wrote] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
