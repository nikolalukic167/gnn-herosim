#!/usr/bin/env python3
"""rollout_imitation_v1 -- the forced-placement rollout label engine and Phase A (rank stability).
See docs/lineages/rollout_imitation_v1.md.

For one training topology (cell config) + an unburst workload, run the immediate peer_greedy rule
once to capture each decision's candidate set, then LABEL each decision by one step of policy
improvement: for candidate c, re-run the trace from t=0 with the rule and the decision task forced
to c, truncated N arrivals after the decision; the group-local cost is

    cost(c, N) = elapsed(i) + sum elapsed(peers of i) + sum elapsed(tasks queued behind i on i's
                 chosen platform, arriving within the horizon)

all read from taskResults of the forced run. Label = argmin over c. Horizons N in {20, 50, 100}.

Phase A (R0, blocking): over the pooled decisions, the candidate RANKING at N=20 vs 50 vs 100 --
median Spearman >= 0.80 and the argmin agreeing on >= 80%. Otherwise the label is downstream chaos
(as objective_pivot_v1 P3 found a horizon return to be) and the lineage closes LABEL-IS-CHAOS.

Load note: the study's arrival rate saturates a 6-server training cell over the horizon (the rule
then places onto an unreachable/full node and the physics fails loud), so the label runs use a
non-saturating unburst rate (drainable_f700); this is a rate choice for a well-defined rollout
cost, disclosed, not the study's serving rate.

Usage (datalab, micromamba gnn, PYTHONPATH=.):
  python3 scripts_cosim/rollout_imitation_v1_label.py --seed 9201 \
      --config  simulation_data/peer_affinity_live_gate/configs/cc40s9201.json \
      --workload simulation_data/peer_affinity_live_gate/workloads/drainable_f700_n50000.json \
      --out results/rollout_imitation_v1/labels/s9201.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HORIZONS = (20, 50, 100)

# env the rule is served under (peer physics on, full per-task stats). NOT the batch env: the
# immediate per-arrival rule is the one whose one-step improvement we label.
PHYSICS_ENV = {
    "HEROSIM_PEER_EXCHANGE": "1", "SIM_FORCE_FULL_STATS": "1",
    "COSIM_SUPPRESS_SIM_PRINTS": "1", "HEROSIM_GNN_DEVICE": "cpu", "PYTHONHASHSEED": "0",
}
for _k, _v in PHYSICS_ENV.items():
    os.environ.setdefault(_k, _v)

import logging  # noqa: E402
from src.executesimulation import run_simulation  # noqa: E402

_LOG = logging.getLogger("rollout_label")
_SIM_INPUT = REPO_ROOT / "data" / "nofs-ids"


def _run(config: Path, workload: Path, policy: str, seed: int, *, max_events: Optional[int],
         forced: Optional[Dict[int, list]], capture: Optional[Path]) -> dict:
    """One rule run through the real engine; returns the stats dict (with taskResults)."""
    if max_events is not None:
        os.environ["HEROSIM_MAX_EVENTS"] = str(int(max_events))
    else:
        os.environ.pop("HEROSIM_MAX_EVENTS", None)
    if forced is not None:
        os.environ["HEROSIM_FORCED_PLACEMENTS"] = json.dumps({str(k): v for k, v in forced.items()})
    else:
        os.environ.pop("HEROSIM_FORCED_PLACEMENTS", None)
    if capture is not None:
        capture.unlink(missing_ok=True)
        os.environ["HEROSIM_PG_CAPTURE_PATH"] = str(capture)
    else:
        os.environ.pop("HEROSIM_PG_CAPTURE_PATH", None)
    with tempfile.NamedTemporaryFile("r", suffix=".json", delete=False) as tf:
        out = Path(tf.name)
    ok = run_simulation(config, workload, out, _SIM_INPUT, _LOG, policy, seed=seed)
    if not ok:
        out.unlink(missing_ok=True)
        raise RuntimeError(f"run_simulation failed (seed={seed}, forced={forced}, N={max_events})")
    d = json.load(open(out))
    out.unlink(missing_ok=True)
    return d.get("stats", d)


def _task_rows(stats: dict) -> Dict[int, dict]:
    return {int(t["taskId"]): t for t in stats.get("taskResults", [])}


def _peer_groups(workload_path: Path) -> Dict[int, List[int]]:
    """Peers of each task from the workload's peer_exchange pairs ([i, j, payload])."""
    w = json.load(open(workload_path))
    peers: Dict[int, set] = {}
    for p in w.get("peer_exchange", []):
        i, j = int(p[0]), int(p[1])
        peers.setdefault(i, set()).add(j)
        peers.setdefault(j, set()).add(i)
    return {k: sorted(v) for k, v in peers.items()}


def _cost(stats: dict, decision_id: int, peers: Sequence[int]) -> Optional[float]:
    """Group-local cost of the forced run: elapsed(i) + elapsed(peers present) + elapsed(tasks
    queued behind i on i's chosen platform within the truncated horizon). None if i did not run."""
    rows = _task_rows(stats)
    di = rows.get(decision_id)
    if di is None:
        return None
    total = float(di["elapsedTime"])
    for p in peers:
        rp = rows.get(p)
        if rp is not None:
            total += float(rp["elapsedTime"])
    node, plat, disp = di.get("executionNode"), di.get("executionPlatform"), float(di["dispatchedTime"])
    for tid, r in rows.items():
        if tid == decision_id:
            continue
        if (r.get("executionNode") == node and r.get("executionPlatform") == plat
                and float(r["dispatchedTime"]) >= disp):
            total += float(r["elapsedTime"])
    return total


def _spearman(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    n = len(a)
    if n < 2:
        return None

    def ranks(x):
        order = sorted(range(n), key=lambda k: x[k])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and x[order[j + 1]] == x[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    ra, rb = ranks(a), ranks(b)
    dbar_a = sum(ra) / n
    dbar_b = sum(rb) / n
    num = sum((ra[k] - dbar_a) * (rb[k] - dbar_b) for k in range(n))
    da = sum((ra[k] - dbar_a) ** 2 for k in range(n)) ** 0.5
    db = sum((rb[k] - dbar_b) ** 2 for k in range(n)) ** 0.5
    if da == 0 or db == 0:
        return None
    return num / (da * db)


def label_decisions(config: Path, workload: Path, seed: int, *, capture_n: int,
                    decision_max: int, max_decisions: int) -> dict:
    peers_of = _peer_groups(workload)
    cap = Path(tempfile.gettempdir()) / f"rollout_cap_{os.getpid()}_{seed}.jsonl"
    _run(config, workload, "peer_greedy_network", seed, max_events=capture_n, forced=None, capture=cap)
    decisions = []
    for line in open(cap):
        rec = json.loads(line)
        tid, cands = int(rec["task_id"]), rec["candidates"]
        if len(cands) >= 2 and tid + max(HORIZONS) <= capture_n and tid < decision_max:
            decisions.append((tid, cands))
    cap.unlink(missing_ok=True)
    decisions = decisions[:max_decisions]

    out_decisions = []
    for tid, cands in decisions:
        peers = peers_of.get(tid, [])
        costs_by_h: Dict[int, List[float]] = {}
        ok = True
        for N in HORIZONS:
            row = []
            for c in cands:
                node_id, plat_id = int(c[0]), int(c[1])
                stats = _run(config, workload, "peer_greedy_network", seed,
                             max_events=tid + N, forced={tid: [node_id, plat_id]}, capture=None)
                cost = _cost(stats, tid, peers)
                if cost is None:
                    ok = False
                    break
                row.append(cost)
            if not ok:
                break
            costs_by_h[N] = row
        if not ok:
            continue
        argmins = {N: min(range(len(costs_by_h[N])), key=lambda k: costs_by_h[N][k]) for N in HORIZONS}
        rhos = {
            "20_50": _spearman(costs_by_h[20], costs_by_h[50]),
            "20_100": _spearman(costs_by_h[20], costs_by_h[100]),
            "50_100": _spearman(costs_by_h[50], costs_by_h[100]),
        }
        out_decisions.append({
            "task_id": tid, "n_candidates": len(cands), "peers": len(peers),
            "costs": costs_by_h, "argmins": argmins, "spearman": rhos,
        })

    return {"seed": seed, "config": str(config), "workload": str(workload),
            "n_decisions": len(out_decisions), "decisions": out_decisions}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--workload", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    # Multi-candidate decisions only appear after replicas warm (~pos 248 on a 6-server training
    # cell) and top out at 2 candidates; capture through the warm region and label those.
    ap.add_argument("--capture-n", type=int, default=620, help="events kept while capturing decisions")
    ap.add_argument("--decision-max", type=int, default=500, help="only decisions whose id < this")
    ap.add_argument("--max-decisions", type=int, default=15)
    a = ap.parse_args(argv)

    res = label_decisions(a.config, a.workload, a.seed, capture_n=a.capture_n,
                          decision_max=a.decision_max, max_decisions=a.max_decisions)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)

    # local summary (the pooled Phase A read is a separate driver over all seeds)
    rhos = [d["spearman"]["20_100"] for d in res["decisions"] if d["spearman"]["20_100"] is not None]
    agree = [1 for d in res["decisions"] if d["argmins"][20] == d["argmins"][100]]
    n = res["n_decisions"]
    print(f"seed {a.seed}: {n} decisions; "
          f"median rho(20,100)={median(rhos):.3f} " if rhos else f"seed {a.seed}: {n} decisions; ",
          f"argmin(20==100)={sum(agree)}/{n}" if n else "")
    print(f"   wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
