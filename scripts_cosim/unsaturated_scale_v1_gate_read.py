"""unsaturated_scale_v1 -- the gate read. Bars live in unsaturated_scale_v1_read.py; this file
only finds the summaries, pairs the arms and prints.

    # S1: the baseline sweep -> the selected factor
    python3 scripts_cosim/unsaturated_scale_v1_gate_read.py s1 \
        simulation_data/peer_affinity_live_gate/results/us_v1 \
        simulation_data/peer_affinity_live_gate/results/psv3_p3

    # S2: the learned arms at the selected rung
    python3 scripts_cosim/unsaturated_scale_v1_gate_read.py s2 \
        simulation_data/peer_affinity_live_gate/results/us_v1 \
        simulation_data/peer_affinity_live_gate/results/psv3_p3 \
        simulation_data/peer_affinity_live_gate/results/po_v1 --factor 1000
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts_cosim.unsaturated_scale_v1_read import (  # noqa: E402
    K_ARMS, K_CELLS, K_FACTORS, K_GNN, K_GRAPH, K_TWIN, V_UNREADABLE,
    collapse_to_seed, read_k1, read_k2, read_k3, read_k4, read_s1, select_factor,
)
from scripts_cosim.peer_only_v1_read import read_pair_pct  # noqa: E402

U80_RUNG = "U80"     # the rung tag S2 writes into results/po_v1
RELABEL = {"v3ext": "516"}


def _load(d: str, pattern: str) -> list:
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, pattern)))]


# --- S1 -----------------------------------------------------------------------------------

def s1_shares(us_dir: str, p3_dir: str) -> Dict[int, Dict[str, float]]:
    """factor -> cell -> reactive queue share. f300 comes from psv3_p3's R3 reactive arms; a
    reactive arm that did not finish 50,000 tasks is dropped with a warning, never used."""
    out: Dict[int, Dict[str, float]] = {}

    def put(factor, row, src):
        n = row.get("num_tasks")
        if n is None or int(n) != 50000:
            print(f"[WARN] {src}: num_tasks={n!r}, not a rung-defining arm; dropped", file=sys.stderr)
            return
        e, q = float(row["averageElapsedTime"]), float(row["averageQueueTime"])
        out.setdefault(int(factor), {})[row["cell"]] = q / e

    for row in _load(us_dir, "cs80s*__F*__reactive_s0.summary.json"):
        put(row["factor"], row, row["arm"])
    for row in _load(p3_dir, "cs80s*__R3__reactive_s0.summary.json"):
        assert row.get("workload") == "drainable_f300_n50000", row.get("workload")
        put(300, row, row["arm"])
    return out


def report_s1(us_dir: str, p3_dir: str) -> dict:
    shares = s1_shares(us_dir, p3_dir)
    s1 = read_s1(shares)
    sel = select_factor(s1)
    print("S1 -- reactive knative_network, 80 servers, queue share = queue / elapsed")
    print(f"{'factor':>7} {'rate/s':>7} {'share':>7} {'sat?':>5} {'band?':>5}  per-cell")
    for f in K_FACTORS:
        r = s1["factors"][f]
        if "queue_share" not in r:
            print(f"{f:>7} {'':>7} {'UNREADABLE':>7}  missing {r['missing']}")
            continue
        cells = " ".join(f"{v:.3f}" for v in r["per_cell"].values())
        print(f"f{f:<6} {1841.0 / f:>7.3f} {r['queue_share']:>7.3f} {str(r['saturated']):>5} "
              f"{str(r['in_band']):>5}  {cells}")
    print(f"monotone in f: {s1['monotone']}")
    print(f"\nSELECTION: {sel['verdict']}  factor=f{sel.get('factor')}  "
          f"share={sel.get('queue_share', float('nan')):.3f}  primary={sel.get('primary')}")
    if sel.get("why"):
        print(f"  {sel['why']}")
    return {"s1": s1, "selection": sel}


# --- S2 -----------------------------------------------------------------------------------

def s2_table(po_dir: str) -> Dict[str, Dict[tuple, float]]:
    """label -> {(cell, seed): elapsed} for every U80 arm; v3ext relabelled to 516."""
    table: Dict[str, Dict[tuple, float]] = {}
    for row in _load(po_dir, f"cs80s*__{U80_RUNG}__*.summary.json"):
        n = row.get("num_tasks")
        if n is None or int(n) != 50000:
            print(f"[WARN] {row['arm']}: num_tasks={n!r}; dropped", file=sys.stderr)
            continue
        corpus = RELABEL.get(row["corpus"], row["corpus"])
        label = f"{corpus}_{row['arm_kind']}"
        table.setdefault(label, {})[(row["cell"], int(row["checkpoint_seed"]))] = float(row["averageElapsedTime"])
    return table


def _collapse(table, label) -> Optional[Dict[int, float]]:
    arm = table.get(label)
    if not arm:
        return None
    try:
        return collapse_to_seed(arm, rung_cells=list(K_CELLS))
    except ValueError as e:
        print(f"[WARN] {label}: {e}", file=sys.stderr)
        return None


def report_s2(us_dir: str, p3_dir: str, po_dir: str, factor: int) -> dict:
    s1 = report_s1(us_dir, p3_dir)
    sel = s1["selection"]
    if sel.get("factor") != factor:
        raise SystemExit(f"FAIL LOUD: S2 was run at f{factor} but S1 selects f{sel.get('factor')}; "
                         "the rung is chosen by the rule, not by the caller")
    shares = s1_shares(us_dir, p3_dir)[factor]
    # reactive is deterministic at seed 0: one value per cell, collapsed then replicated
    react_rows = {r["cell"]: float(r["averageElapsedTime"])
                  for r in _load(us_dir, f"cs80s*__F{factor}__reactive_s0.summary.json")
                  if int(r.get("num_tasks") or 0) == 50000}
    if factor == 300:
        react_rows = {r["cell"]: float(r["averageElapsedTime"])
                      for r in _load(p3_dir, "cs80s*__R3__reactive_s0.summary.json")}
    missing = [c for c in K_CELLS if c not in react_rows]
    if missing:
        raise SystemExit(f"FAIL LOUD: reactive at f{factor} is missing cells {missing}")
    from statistics import median
    react_one = median(react_rows[c] for c in K_CELLS)

    table = s2_table(po_dir)
    print(f"\nS2 -- learned arms at f{factor} (80 servers), reactive elapsed {react_one:.3f} s, "
          f"queue share {median(shares.values()):.3f}")
    by_seed: Dict[str, Dict[int, float]] = {}
    k1: Dict[str, dict] = {}
    for label in K_ARMS:
        s = _collapse(table, label)
        if s is None:
            k1[label] = {"verdict": V_UNREADABLE, "why": "arm absent or ragged"}
            print(f"  {label:<18} UNREADABLE ({len(table.get(label, {}))} cell-seed pairs)")
            continue
        by_seed[label] = s
        react = {seed: react_one for seed in s}
        r = read_k1(s, react)
        k1[label] = r
        n_ahead = r.get("v3_ahead", r.get("ahead"))
        print(f"  {label:<18} K1 {r['verdict']:<40} median {r.get('median', float('nan')):+7.2f} % "
              f"p={r.get('p', float('nan')):.4f} ahead {n_ahead}/{len(s)}  "
              f"elapsed {median(s.values()):.3f} s")
    k4 = read_k4(k1, primary=bool(sel.get("primary")))
    out = {"s1": s1, "k1": k1, "k4": k4}
    if K_GRAPH in by_seed and K_TWIN in by_seed:
        out["k2"] = read_k2(by_seed[K_GRAPH], by_seed[K_TWIN])
        print(f"\n  K2 {K_GRAPH} vs {K_TWIN}: {out['k2']['verdict']}  median "
              f"{out['k2'].get('median', float('nan')):+.2f} % p={out['k2'].get('p', float('nan')):.4f}")
    if K_GNN in by_seed and K_GRAPH in by_seed:
        out["k3"] = read_k3(by_seed[K_GNN], by_seed[K_GRAPH])
        print(f"  K3 {K_GNN} vs {K_GRAPH}: {out['k3']['verdict']}  median "
              f"{out['k3'].get('median', float('nan')):+.2f} % p={out['k3'].get('p', float('nan')):.4f}")
    print(f"\nK4: {k4['verdict']}  winners={k4.get('winners')} losers={k4.get('losers')} "
          f"missing={k4.get('missing')} secondary={k4.get('secondary')}")
    print(f"  {k4.get('why', '')}")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("s1", "s2"))
    ap.add_argument("us_dir"); ap.add_argument("p3_dir")
    ap.add_argument("po_dir", nargs="?")
    ap.add_argument("--factor", type=int)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    if a.step == "s1":
        res = report_s1(a.us_dir, a.p3_dir)
    else:
        if not a.po_dir or a.factor is None:
            ap.error("s2 needs po_dir and --factor")
        res = report_s2(a.us_dir, a.p3_dir, a.po_dir, a.factor)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
