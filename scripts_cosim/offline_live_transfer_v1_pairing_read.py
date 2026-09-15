#!/usr/bin/env python3
"""offline_live_transfer_v1 -- R0 (positive control) and R1 (the primary read).

Registered in docs/lineages/offline_live_transfer_v1.md on 2026-09-15. Every bar below is
a module constant and was committed BEFORE the registered families were read.

The question: does the offline score -- held-out plan regret at the selected checkpoint,
the number the checkpoint selector actually uses -- have any ability to rank checkpoints
by how they perform when served?

R0 POSITIVE CONTROL. The read must detect a correlation known to be there, or it is not
   measuring anything. Live latency against live queue time, per family.
   Bar: Spearman >= 0.90 in ALL THREE families, else every other read here is VOID.

R1 PRIMARY. Two registered readings of the same pairing:

   * within-arm -- Spearman per (family, arm), n = 16 each. Underpowered on its own
     (~47 % power at rho = 0.5), which is why it is not the only reading.
     Bar: |rho| >= 0.50, p < 0.05, same sign, in >= 2 of 3 families, for >= 1 arm.

   * pooled-z -- z-score BOTH variables within each (family, arm) cell, which removes the
     arm and family offsets and leaves only seed-level signal, then correlate all 96
     points. This is the powered reading (~80 % power at rho = 0.30).
     Bar: |rho| >= 0.30, p < 0.05.

   OFFLINE-INFORMATIVE if either bar fires; OFFLINE-UNINFORMATIVE otherwise.
   Registered expectation: OFFLINE-UNINFORMATIVE.

   A plain pooled correlation (arms not z-scored apart) is reported for continuity with
   how the reversal has been quoted in this record, and DECIDES NOTHING: it is dominated
   by the two arm means, which is the very thing under suspicion.

BURNED FAMILIES. `peer_affinity_v1` T1b and `drainable_objective_v1` V = 1 were read on
this statistic on 2026-09-15 before the bars were written. They are disclosed in the node
as the motivating measurement and this tool REFUSES to score them -- a bar cannot be
applied to data it was fitted to.

Usage:

    python3 scripts_cosim/offline_live_transfer_v1_pairing_read.py \
        --gate-root simulation_data/peer_affinity_live_gate/results \
        --wandb-root wandb \
        --out simulation_data/offline_live_transfer_v1/r0_r1.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import statistics as st
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# BARS -- signed 2026-09-15, before the registered families were read.
# ---------------------------------------------------------------------------
R0_CONTROL_MIN_RHO = 0.90        # live latency vs live queue time, every family
R1_WITHIN_MIN_ABS_RHO = 0.50     # within-arm
R1_WITHIN_MIN_FAMILIES = 2       # ... in at least this many families, same sign
R1_POOLED_Z_MIN_ABS_RHO = 0.30   # pooled-z, the powered reading
R1_ALPHA = 0.05
MIN_SEEDS_PER_CELL = 12          # a (family, arm) cell below this is VOID, not read
REGISTERED_EXPECTATION = "OFFLINE-UNINFORMATIVE"

# The offline statistic is the SELECTED checkpoint's held-out score -- what the selector
# uses. The test-split twin is reported as a secondary and never substituted for it.
OFFLINE_PRIMARY_KEY = "final/val/regret_masked_topo"
OFFLINE_SECONDARY_KEY = "final/test/regret_masked_topo"
LIVE_KEY = "averageElapsedTime"
LIVE_CONTROL_KEY = "averageQueueTime"

# ---------------------------------------------------------------------------
# The three registered families. Each: 16 gnn + 16 mpoff, lr 2e-3, per-seed live
# results with matching W&B runs. None had been read on this statistic.
# ---------------------------------------------------------------------------
FAMILIES: Dict[str, Dict[str, str]] = {
    "F1_x800p2": {
        "results": "x800p2_uncapped_gate",
        "run_re": r"peer-affinity-v1-x800p2-(gnn|mpoff)-lr2e3-seed(\d+)",
    },
    "F2_x800p3": {
        "results": "x800p3_uncapped_gate",
        "run_re": r"peer-affinity-v1-x800p3-(gnn|mpoff)-lr2e3-seed(\d+)",
    },
    "F3_warm": {
        "results": "warm_uncapped_gate",
        "run_re": r"peer-affinity-v1-warm-(gnn|mpoff)-lr2e3-seed(\d+)",
    },
}

# Refused on purpose. See the node's "motivating measurement" section.
BURNED_RESULT_DIRS = {"drain_f4000_E_pg16", "dobj_f4000_pg16"}
BURNED_RUN_PATTERNS = ("peer-affinity-v1-t1b-", "drainable-objective-v1-")

ARMS = ("gnn", "mpoff")


class TransferReadError(RuntimeError):
    """Fail loud: a silent hole here is a bar applied to the wrong data."""


# ---------------------------------------------------------------------------
# Statistics -- pure, so the tests can pin them without touching the filesystem.
# ---------------------------------------------------------------------------
def _ranks(xs: Sequence[float]) -> List[float]:
    """Average ranks, so ties do not silently bias rho."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        shared = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n != len(ys):
        raise TransferReadError(f"paired series differ in length: {n} vs {len(ys)}")
    if n < 3:
        raise TransferReadError(f"need >= 3 points to correlate, got {n}")
    mx, my = st.fmean(xs), st.fmean(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        raise TransferReadError("a series is constant -- correlation is undefined, not 0")
    return sxy / math.sqrt(sxx * syy)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, float]:
    """Spearman rho and a two-sided p-value from the t approximation."""
    rho = pearson(_ranks(xs), _ranks(ys))
    n = len(xs)
    if n <= 2 or abs(rho) >= 1.0:
        return rho, 0.0 if abs(rho) >= 1.0 else 1.0
    t = rho * math.sqrt((n - 2) / (1.0 - rho * rho))
    return rho, _t_sf_two_sided(abs(t), n - 2)


def _t_sf_two_sided(t: float, df: int) -> float:
    """Two-sided tail of Student's t, via the regularised incomplete beta."""
    x = df / (df + t * t)
    return _betainc(df / 2.0, 0.5, x)


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b) -- continued fraction, Lentz."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1.0 - x) * b - lbeta) / a
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _betainc(b, a, 1.0 - x)
    f, c, d = 1.0, 1.0, 0.0
    for i in range(0, 300):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2.0 * m - 1.0) * (a + 2.0 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2.0 * m) * (a + 2.0 * m + 1.0))
        d = 1.0 + num * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d
        c = 1.0 + num / c
        c = 1e-30 if abs(c) < 1e-30 else c
        delta = c * d
        f *= delta
        if abs(1.0 - delta) < 1e-12:
            break
    return front * (f - 1.0)


def zscore(xs: Sequence[float]) -> List[float]:
    """Centre and scale one (family, arm) cell. A constant cell is a fail, not zeros."""
    if len(xs) < 2:
        raise TransferReadError(f"cannot z-score {len(xs)} point(s)")
    m = st.fmean(xs)
    sd = st.pstdev(xs)
    if sd <= 0:
        raise TransferReadError("cell is constant -- z-scoring it would invent signal")
    return [(x - m) / sd for x in xs]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_live(gate_root: Path, results_dir: str) -> Dict[Tuple[str, int], Dict[str, float]]:
    if results_dir in BURNED_RESULT_DIRS:
        raise TransferReadError(
            f"{results_dir} is a BURNED family: it was read on this statistic before the "
            "bars were written and is disclosed in the node as the motivating "
            "measurement. Scoring it here would apply a bar to the data it was fitted to."
        )
    root = gate_root / results_dir
    if not root.is_dir():
        raise TransferReadError(f"live results missing: {root}")
    out: Dict[Tuple[str, int], Dict[str, float]] = {}
    for path in sorted(root.glob("*.summary.json")):
        summary = json.loads(path.read_text())
        arm = summary.get("arm") or ""
        match = re.fullmatch(r"(gnn|mpoff)_s(\d+)", arm)
        if not match:
            continue  # reactive baselines carry no checkpoint and pair with nothing
        for key in (LIVE_KEY, LIVE_CONTROL_KEY):
            if summary.get(key) is None:
                raise TransferReadError(f"{path}: missing {key}")
        out[(match.group(1), int(match.group(2)))] = {
            LIVE_KEY: float(summary[LIVE_KEY]),
            LIVE_CONTROL_KEY: float(summary[LIVE_CONTROL_KEY]),
        }
    if not out:
        raise TransferReadError(f"{root}: no per-seed learned arms found")
    return out


def load_offline(wandb_root: Path, run_re: str) -> Dict[Tuple[str, int], Dict[str, float]]:
    for burned in BURNED_RUN_PATTERNS:
        if burned in run_re:
            raise TransferReadError(f"{run_re!r} names a BURNED family; see the node")
    pattern = re.compile(run_re)
    out: Dict[Tuple[str, int], Dict[str, float]] = {}
    for run_dir in sorted(wandb_root.glob("run-*")):
        log = run_dir / "files" / "output.log"
        summary_path = run_dir / "files" / "wandb-summary.json"
        if not log.is_file() or not summary_path.is_file():
            continue
        match = pattern.search(log.read_text(errors="ignore"))
        if not match:
            continue
        summary = json.loads(summary_path.read_text())
        primary = summary.get(OFFLINE_PRIMARY_KEY)
        if primary is None:
            raise TransferReadError(
                f"{run_dir}: no {OFFLINE_PRIMARY_KEY}. An unfinished run cannot be paired, "
                "and skipping it would silently reshape the sample."
            )
        key = (match.group(1), int(match.group(2)))
        if key in out:
            raise TransferReadError(
                f"two W&B runs match {key} for {run_re!r} -- the pairing is ambiguous"
            )
        out[key] = {
            "offline": float(primary),
            "offline_test": (
                float(summary[OFFLINE_SECONDARY_KEY])
                if summary.get(OFFLINE_SECONDARY_KEY) is not None else float("nan")
            ),
        }
    if not out:
        raise TransferReadError(f"no W&B runs matched {run_re!r} under {wandb_root}")
    return out


def pair_family(
    gate_root: Path, wandb_root: Path, name: str, spec: Dict[str, str]
) -> Dict[str, Any]:
    live = load_live(gate_root, spec["results"])
    offline = load_offline(wandb_root, spec["run_re"])
    cells: Dict[str, Dict[str, List[float]]] = {}
    for arm in ARMS:
        seeds = sorted(s for (a, s) in live if a == arm and (a, s) in offline)
        if len(seeds) < MIN_SEEDS_PER_CELL:
            raise TransferReadError(
                f"{name}/{arm}: only {len(seeds)} paired seeds, bar is "
                f"{MIN_SEEDS_PER_CELL}. Read the missing runs or drop the family in the "
                "node by amendment -- do not read it short."
            )
        cells[arm] = {
            "seeds": [float(s) for s in seeds],
            "offline": [offline[(arm, s)]["offline"] for s in seeds],
            "offline_test": [offline[(arm, s)]["offline_test"] for s in seeds],
            "live": [live[(arm, s)][LIVE_KEY] for s in seeds],
            "live_queue": [live[(arm, s)][LIVE_CONTROL_KEY] for s in seeds],
        }
    return {"family": name, "results_dir": spec["results"], "cells": cells}


# ---------------------------------------------------------------------------
# The reads
# ---------------------------------------------------------------------------
def read_r0(families: List[Dict[str, Any]]) -> Dict[str, Any]:
    per_family = {}
    for fam in families:
        xs = [v for arm in ARMS for v in fam["cells"][arm]["live_queue"]]
        ys = [v for arm in ARMS for v in fam["cells"][arm]["live"]]
        rho, p = spearman(xs, ys)
        per_family[fam["family"]] = {"rho": rho, "p": p, "n": len(xs),
                                     "passes": rho >= R0_CONTROL_MIN_RHO}
    passes = all(v["passes"] for v in per_family.values())
    return {
        "bar": R0_CONTROL_MIN_RHO,
        "per_family": per_family,
        "passes": passes,
        "verdict": "CONTROL-PASSES" if passes else "CONTROL-FAILS-EVERYTHING-VOID",
    }


def read_r1(families: List[Dict[str, Any]]) -> Dict[str, Any]:
    within: Dict[str, Dict[str, Any]] = {}
    for fam in families:
        for arm in ARMS:
            cell = fam["cells"][arm]
            rho, p = spearman(cell["offline"], cell["live"])
            within[f"{fam['family']}/{arm}"] = {
                "n": len(cell["seeds"]), "rho": rho, "p": p,
                "fires": abs(rho) >= R1_WITHIN_MIN_ABS_RHO and p < R1_ALPHA,
                "offline_range": [min(cell["offline"]), max(cell["offline"])],
                "live_range": [min(cell["live"]), max(cell["live"])],
            }
    # The within-arm bar needs the SAME arm firing in the same direction across families.
    within_fires = False
    for arm in ARMS:
        for sign in (1, -1):
            hits = sum(
                1 for k, v in within.items()
                if k.endswith("/" + arm) and v["fires"] and (v["rho"] > 0) == (sign > 0)
            )
            within_fires = within_fires or hits >= R1_WITHIN_MIN_FAMILIES

    zx: List[float] = []
    zy: List[float] = []
    for fam in families:
        for arm in ARMS:
            cell = fam["cells"][arm]
            zx.extend(zscore(cell["offline"]))
            zy.extend(zscore(cell["live"]))
    z_rho, z_p = spearman(zx, zy)
    pooled_fires = abs(z_rho) >= R1_POOLED_Z_MIN_ABS_RHO and z_p < R1_ALPHA

    # Reported for continuity with how the reversal has been quoted. Decides nothing.
    raw_x = [v for fam in families for arm in ARMS for v in fam["cells"][arm]["offline"]]
    raw_y = [v for fam in families for arm in ARMS for v in fam["cells"][arm]["live"]]
    raw_rho, raw_p = spearman(raw_x, raw_y)

    fires = bool(within_fires or pooled_fires)
    return {
        "within_arm": within,
        "within_arm_bar": {"min_abs_rho": R1_WITHIN_MIN_ABS_RHO,
                           "min_families": R1_WITHIN_MIN_FAMILIES, "fires": within_fires},
        "pooled_z": {"n": len(zx), "rho": z_rho, "p": z_p,
                     "bar": R1_POOLED_Z_MIN_ABS_RHO, "fires": pooled_fires},
        "pooled_raw_decides_nothing": {"n": len(raw_x), "rho": raw_rho, "p": raw_p},
        "fires": fires,
        "verdict": "OFFLINE-INFORMATIVE" if fires else "OFFLINE-UNINFORMATIVE",
        "registered_expectation": REGISTERED_EXPECTATION,
    }


def read(families: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(families) != len(FAMILIES):
        raise TransferReadError(
            f"{len(families)} families supplied, {len(FAMILIES)} registered. The bars are "
            "stated over the registered set; reading a subset is an amendment."
        )
    r0 = read_r0(families)
    out: Dict[str, Any] = {
        "lineage": "offline_live_transfer_v1",
        "offline_statistic": OFFLINE_PRIMARY_KEY,
        "live_statistic": LIVE_KEY,
        "families": {f["family"]: f["results_dir"] for f in families},
        "R0": r0,
    }
    if not r0["passes"]:
        out["R1"] = {"verdict": "VOID-R0-FAILED", "fires": None}
        out["outcome"] = "VOID-R0-FAILED"
        return out
    r1 = read_r1(families)
    out["R1"] = r1
    out["outcome"] = r1["verdict"]
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate-root", type=Path,
                    default=Path("simulation_data/peer_affinity_live_gate/results"))
    ap.add_argument("--wandb-root", type=Path, default=Path("wandb"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    families = [pair_family(args.gate_root, args.wandb_root, name, spec)
                for name, spec in FAMILIES.items()]
    result = read(families)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))

    print(f"[R0] bar rho >= {R0_CONTROL_MIN_RHO} in every family")
    for fam, row in result["R0"]["per_family"].items():
        print(f"     {fam:12s} n={row['n']:3d}  rho={row['rho']:+.4f}  p={row['p']:.2e}"
              f"  -> {'pass' if row['passes'] else 'FAIL'}")
    print(f"[R0] {result['R0']['verdict']}")
    if result["R1"].get("fires") is None:
        print(f"[R1] {result['R1']['verdict']}")
    else:
        print(f"[R1] within-arm (bar |rho| >= {R1_WITHIN_MIN_ABS_RHO}, p < {R1_ALPHA}, "
              f"same arm and sign in >= {R1_WITHIN_MIN_FAMILIES} families)")
        for cell, row in result["R1"]["within_arm"].items():
            print(f"     {cell:18s} n={row['n']:2d}  rho={row['rho']:+.3f}  p={row['p']:.3f}"
                  f"   offline {row['offline_range'][0]:.2f}-{row['offline_range'][1]:.2f}"
                  f"   live {row['live_range'][0]:.2f}-{row['live_range'][1]:.2f}")
        pz = result["R1"]["pooled_z"]
        print(f"     POOLED-Z (powered)  n={pz['n']}  rho={pz['rho']:+.3f}  p={pz['p']:.4f}"
              f"  bar {pz['bar']} -> {'FIRES' if pz['fires'] else 'no'}")
        raw = result["R1"]["pooled_raw_decides_nothing"]
        print(f"     pooled raw (decides nothing)  rho={raw['rho']:+.3f}  p={raw['p']:.4f}")
        print(f"[R1] {result['R1']['verdict']}  (registered expectation "
              f"{REGISTERED_EXPECTATION})")
    print(f"[OUTCOME] {result['outcome']}")
    print(f"[wrote] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
