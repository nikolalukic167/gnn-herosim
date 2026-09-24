"""Phase C (drainable_objective_v1): the live gate read. This is what closes the lineage.

Rule 6: a lineage ends with a live gate, never with an offline read. Phase A's reads order
the work; this one decides it.

BARS (signed in docs/lineages/drainable_objective_v1.md before any arm ran):

  C0 assembly control   every learned arm's OWN counters must show the batching its name
                        claims: mean batch size >= 4 and incomplete peer-group batches
                        <= 20 % of prefix_batches. An arm that fails is CONFOUNDED and is
                        not read. Absent counters fail loud. This is the bar that caught
                        three confounded reads in drainable_regime_v1, one of which was
                        wrong by 30x.
  C1 behaviour control  the V = 1 arms must place above the shallowest legal replica in
                        <= 18 % of choosable placements (half of T1b's 35.5 %). Read from
                        two raw-retained seeds per arm. If the label did not change the
                        behaviour it was built to change, C2/C3 are read as CONFOUNDED --
                        a latency win from an arm that still concentrates is not evidence
                        about the label.
  C2 clock defect       WITHDRAWN by Amendment 1 and replaced by a measurement: repricing
                        the backlog on the measured clock moves the median sweep RTT ~1 %
                        and leaves the ARGMIN PLAN unmoved, so a V = 0 arm is a copy of
                        T1b and this bar could not have fired by construction.
  C3 label lever        V = 1 vs the T1b lr2e3 checkpoints at this cell (the V = 0 control,
     (PRIMARY)          same corpus, same split, same seeds, same lr -- only the label
                        differs): median lower, Mann-Whitney p < 0.05, >= 12/16
                        =>  LABEL-HELPS.
  C4 vs reactive        V = 1 vs knative_network: median below AND >= 12/16 seeds below
     (HEADLINE)         =>  LEARNED-BEATS-REACTIVE, which no measurement in this program
                        has produced at a drainable load.
  C5 graph question     V = 1 gnn vs V = 1 mpoff. Registered prediction: TIE. The shaped
                        term is inside count competitor v2, so a GNN-NEEDED reading here
                        would contradict the composition theorem and is reported as an
                        anomaly to investigate, never as a headline.

Input is the gate's summary JSONs (the live sbatch writes one per arm). Latency is
`averageElapsedTime`, the statistic every drainable_* reading in this program quotes.

Usage:

    python3 scripts_cosim/drainable_objective_v1_gate_read.py \
        --results simulation_data/peer_affinity_live_gate/results/dobj_f4000_pg16 \
        --raw-read simulation_data/drainable_objective_v1/c1_pertask.json \
        --out simulation_data/drainable_objective_v1/c_gate_read.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---- BARS. Committed before any arm ran; do not edit after. ----------------
C0_MEAN_BATCH_MIN = 4.0
C0_INCOMPLETE_MAX_PCT = 20.0
C1_ABOVE_MIN_MAX_PCT = 18.0
C2_MIN_SEEDS = 12
C3_MIN_SEEDS = 12
C3_ALPHA = 0.05
C4_MIN_SEEDS = 12
SEEDS_PER_ARM = 16
# The T1b control at this exact cell (drainable_serving_config_v1, config E, 16 s window).
# Its PER-SEED results are on disk at results/drain_f4000_E_pg16 and are loaded with
# --control-results, so C3 is a proper 16-vs-16 test rather than a comparison against a
# constant. These medians are the fallback and the cross-check: a control dir whose median
# disagrees with them is not the run the record describes, and the read says so.
T1B_BASELINE_S = {"gnn": 53.45, "mpoff": 51.32}
T1B_BASELINE_TOLERANCE_S = 0.05
KNATIVE_BASELINE_S = 25.95
CONTROL_RESULTS_DEFAULT = "simulation_data/peer_affinity_live_gate/results/drain_f4000_E_pg16"
# ---------------------------------------------------------------------------


class GateReadError(RuntimeError):
    """Fail loud."""


def mann_whitney_u_p(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """Two-sided Mann-Whitney U via the normal approximation with a tie correction.

    n = 16 per arm is comfortably inside the range where the normal approximation is
    standard; the exact test is not worth a dependency here, and the bar is 0.05 against
    effects the parent lineage measured at p <= 1e-3.
    """
    na, nb = len(a), len(b)
    if na < 3 or nb < 3:
        return None
    combined = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks: List[float] = [0.0] * (na + nb)
    i = 0
    tie_term = 0.0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        t = j - i + 1
        tie_term += t**3 - t
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    ra = sum(r for r, (_v, g) in zip(ranks, combined) if g == 0)
    ua = ra - na * (na + 1) / 2.0
    ub = na * nb - ua
    u = min(ua, ub)
    mu = na * nb / 2.0
    n = na + nb
    sigma_sq = (na * nb / 12.0) * ((n + 1) - tie_term / (n * (n - 1)))
    if sigma_sq <= 0:
        return None
    z = (u - mu) / math.sqrt(sigma_sq)
    return 2.0 * 0.5 * math.erfc(abs(z) / math.sqrt(2.0))


def load_arms(results_dir: Path) -> Dict[str, Dict[str, Any]]:
    if not results_dir.is_dir():
        raise GateReadError(f"results dir missing: {results_dir}")
    arms: Dict[str, Dict[str, Any]] = {}
    for path in sorted(results_dir.glob("*.summary.json")):
        payload = json.loads(path.read_text())
        arm = payload.get("arm") or path.name.replace(".summary.json", "")
        arms[arm] = payload
    if not arms:
        raise GateReadError(f"no *.summary.json under {results_dir}")
    return arms


def latency(summary: Dict[str, Any], arm: str) -> float:
    value = summary.get("averageElapsedTime")
    if value is None:
        raise GateReadError(
            f"{arm}: no averageElapsedTime in its summary -- the gate's own reducer writes "
            "it; a summary without it is a truncated write, not a fast arm"
        )
    return float(value)


def counters_ok(summary: Dict[str, Any], arm: str) -> Dict[str, Any]:
    """C0: the arm's own counters must show the batching its name claims."""
    counters = summary.get("schedulerCounters") or {}
    if not counters:
        raise GateReadError(
            f"{arm}: no schedulerCounters -- C0 cannot be read and an unread C0 is how "
            "three reads in drainable_regime_v1 were confounded, one by 30x"
        )
    batches = float(counters.get("prefix_batches") or 0)
    decoded = float(counters.get("prefix_tasks_decoded") or 0)
    incomplete = float(counters.get("peer_group_incomplete_batches") or 0)
    if batches <= 0:
        raise GateReadError(f"{arm}: prefix_batches is 0 -- this arm decoded nothing")
    mean_batch = decoded / batches
    incomplete_pct = 100.0 * incomplete / batches
    return {
        "mean_batch_size": mean_batch,
        "incomplete_pct": incomplete_pct,
        "passes": mean_batch >= C0_MEAN_BATCH_MIN and incomplete_pct <= C0_INCOMPLETE_MAX_PCT,
    }


def arm_group(arms: Dict[str, Dict[str, Any]], prefix: str) -> List[Tuple[str, float]]:
    """Every seed of one arm, as (name, latency), sorted by name."""
    out = []
    for name, summary in arms.items():
        if name.startswith(prefix + "_s"):
            out.append((name, latency(summary, name)))
    return sorted(out)


def compare(
    treatment: List[Tuple[str, float]],
    reference_median: float,
    min_seeds: int,
) -> Dict[str, Any]:
    values = [v for _n, v in treatment]
    if not values:
        return {"n": 0, "verdict": "VOID-NO-ARM"}
    med = st.median(values)
    better = sum(1 for v in values if v < reference_median)
    return {
        "n": len(values),
        "median_s": med,
        "reference_median_s": reference_median,
        "delta_pct": 100.0 * (reference_median - med) / reference_median,
        "seeds_better": better,
        "seeds_better_bar": min_seeds,
        "fires": med < reference_median and better >= min_seeds,
    }


def load_control(results_dir: Optional[Path]) -> Dict[str, List[float]]:
    """T1b's per-seed latencies at this cell, checked against the recorded medians.

    The control is a real 16-seed arm, not a number in a comment. A control directory whose
    median disagrees with what the record says is a different run, and reading a contrast
    against it would silently redefine the baseline.
    """
    if results_dir is None:
        return {}
    if not results_dir.is_dir():
        raise GateReadError(f"control results dir missing: {results_dir}")
    out: Dict[str, List[float]] = {}
    for arm in ("gnn", "mpoff"):
        vals = []
        for path in sorted(results_dir.glob(f"{arm}_s*.summary.json")):
            vals.append(float(json.loads(path.read_text())["averageElapsedTime"]))
        if not vals:
            raise GateReadError(f"no {arm}_s*.summary.json under {results_dir}")
        med = st.median(vals)
        expected = T1B_BASELINE_S[arm]
        if abs(med - expected) > T1B_BASELINE_TOLERANCE_S:
            raise GateReadError(
                f"control {arm} median {med:.2f} s != the recorded T1b baseline "
                f"{expected:.2f} s -- {results_dir} is not the run the record describes"
            )
        out[arm] = vals
    return out


def read(
    arms: Dict[str, Dict[str, Any]],
    c1: Optional[Dict[str, Any]],
    control: Optional[Dict[str, List[float]]] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"bars": {
        "C0_MEAN_BATCH_MIN": C0_MEAN_BATCH_MIN,
        "C0_INCOMPLETE_MAX_PCT": C0_INCOMPLETE_MAX_PCT,
        "C1_ABOVE_MIN_MAX_PCT": C1_ABOVE_MIN_MAX_PCT,
        "C3_ALPHA": C3_ALPHA,
        "seeds_bar": C3_MIN_SEEDS,
    }}

    # --- C0, per learned arm -------------------------------------------------
    c0: Dict[str, Any] = {}
    confounded: set = set()
    for name, summary in arms.items():
        if not any(name.startswith(p) for p in ("v1_", "v05_", "v2_")):
            continue
        c0[name] = counters_ok(summary, name)
        if not c0[name]["passes"]:
            confounded.add(name)
    out["C0"] = {"per_arm": c0, "confounded_arms": sorted(confounded)}

    def group(prefix: str) -> List[Tuple[str, float]]:
        return [(n, v) for n, v in arm_group(arms, prefix) if n not in confounded]

    # --- C1, from the per-task read -----------------------------------------
    if c1:
        per_arm = {
            k: v.get("chosen_queue_vs_min", {}).get("pct_above_min")
            for k, v in (c1.get("arms") or {}).items()
        }
        v1_arms = {k: v for k, v in per_arm.items() if k.startswith("v1_") and v is not None}
        worst = max(v1_arms.values()) if v1_arms else None
        out["C1"] = {
            "pct_above_min_per_arm": per_arm,
            "worst_v1_arm_pct": worst,
            "bar": C1_ABOVE_MIN_MAX_PCT,
            "fires": worst is not None and worst <= C1_ABOVE_MIN_MAX_PCT,
        }
    else:
        out["C1"] = {"fires": None, "note": "no --raw-read given; C2/C3 are read without "
                                            "the behaviour control and say so"}

    behaviour_ok = out["C1"].get("fires")

    # --- C2 (withdrawn) / C3 / C4 / C5 ---------------------------------------
    out["C2"] = {
        "verdict": "WITHDRAWN",
        "note": ("Amendment 1: repricing the backlog on the measured clock moves the median "
                 "sweep RTT ~1 % and leaves the argmin plan unmoved, so V = 0 is the T1b "
                 "checkpoints and this bar could not fire by construction. Replaced by that "
                 "measurement, recorded in the node."),
    }

    c3: Dict[str, Any] = {}
    for arm in ("gnn", "mpoff"):
        v1 = group(f"v1_{arm}")
        if not v1:
            c3[arm] = {"verdict": "VOID-NO-ARM"}
            continue
        # The control is T1b at this cell: its 16 per-seed latencies when they are on
        # disk (a proper 16-vs-16 test), the recorded median otherwise (seed count only,
        # no p). A V = 0 arm trained here would be the same checkpoint by Amendment 1.
        control_vals = (control or {}).get(arm)
        v0_med = st.median(control_vals) if control_vals else T1B_BASELINE_S[arm]
        row = compare(v1, v0_med, C3_MIN_SEEDS)
        row["control"] = (
            f"T1b {arm}, {len(control_vals)} seeds at this cell" if control_vals
            else f"T1b {arm} recorded median at this cell (per-seed values not supplied)"
        )
        row["p"] = (
            mann_whitney_u_p([v for _n, v in v1], control_vals) if control_vals else None
        )
        # Without per-seed control values there is no p, and the bar explicitly requires
        # one: a seed-count majority against a single number is not a significance test.
        row["fires"] = bool(row.get("fires")) and row["p"] is not None and row["p"] < C3_ALPHA
        row["verdict"] = "LABEL-HELPS" if row["fires"] else "LABEL-DOES-NOT-HELP"
        if behaviour_ok is False:
            row["verdict"] = "CONFOUNDED-C1"
        c3[arm] = row
    out["C3"] = c3

    c4: Dict[str, Any] = {}
    kn = arms.get("knative_network")
    kn_latency = latency(kn, "knative_network") if kn else KNATIVE_BASELINE_S
    for arm in ("gnn", "mpoff"):
        row = compare(group(f"v1_{arm}"), kn_latency, C4_MIN_SEEDS)
        row["verdict"] = (
            "LEARNED-BEATS-REACTIVE" if row.get("fires") else "REACTIVE-STILL-WINS"
        )
        if behaviour_ok is False:
            row["verdict"] = "CONFOUNDED-C1"
        c4[arm] = row
    out["C4"] = {"knative_latency_s": kn_latency, "per_arm": c4}

    g, m = group("v1_gnn"), group("v1_mpoff")
    if g and m:
        gv, mv = [v for _n, v in g], [v for _n, v in m]
        p = mann_whitney_u_p(gv, mv)
        out["C5"] = {
            "gnn_median_s": st.median(gv),
            "mpoff_median_s": st.median(mv),
            "delta_pct": 100.0 * (st.median(mv) - st.median(gv)) / st.median(mv),
            "p": p,
            "registered_prediction": "TIE",
            "verdict": (
                "TIE" if (p is None or p >= C3_ALPHA)
                else ("GNN-NEEDED-ANOMALY" if st.median(gv) < st.median(mv) else "POINTWISE-BETTER")
            ),
        }
    else:
        out["C5"] = {"verdict": "VOID-NO-ARM"}

    # --- the lineage's outcome ----------------------------------------------
    c3_fires = any(v.get("fires") for v in c3.values())
    c4_fires = any(v.get("fires") for v in c4.values())
    if behaviour_ok is False:
        outcome = "CONFOUNDED-C1"
    elif c3_fires and c4_fires:
        outcome = "OBJECTIVE-WAS-THE-LEVER"
    elif c3_fires:
        outcome = "LABEL-HELPS-NOT-ENOUGH"
    else:
        outcome = "OBJECTIVE-NOT-THE-LEVER"
    out["outcome"] = outcome
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--raw-read", type=Path, default=None,
                    help="drainable_debug_pertask_read.py output, for C1")
    ap.add_argument("--control-results", type=Path, default=Path(CONTROL_RESULTS_DEFAULT),
                    help="T1b's 16-seed results at this cell, the C3 control")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    arms = load_arms(args.results)
    c1 = json.loads(args.raw_read.read_text()) if args.raw_read else None
    control = load_control(args.control_results)
    result = read(arms, c1, control)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))

    print("[C0] confounded arms:", result["C0"]["confounded_arms"] or "none")
    print(f"[C1] worst V=1 arm above the shallowest replica: "
          f"{result['C1'].get('worst_v1_arm_pct')}% (bar {C1_ABOVE_MIN_MAX_PCT}) -> "
          f"{result['C1'].get('fires')}")
    # C2 is a flat {"verdict": ..., "note": ...} since Amendment 1 withdrew it, so it is
    # printed on its own -- iterating it as {arm: row} walks the strings and crashes.
    print(f"[C2] {result['C2'].get('verdict')}")
    for arm, row in result["C3"].items():
        print(f"[C3 {arm:>5}] median {row.get('median_s')} vs "
              f"{row.get('reference_median_s')}  "
              f"{row.get('seeds_better')}/{row.get('n')} seeds  "
              f"p={row.get('p')}  -> {row.get('verdict')}")
    for arm, row in result["C4"]["per_arm"].items():
        print(f"[C4 {arm:>5}] median {row.get('median_s')} vs knative "
              f"{result['C4']['knative_latency_s']}  {row.get('seeds_better')}/{row.get('n')}"
              f"  -> {row.get('verdict')}")
    print(f"[C5] {result['C5'].get('verdict')} "
          f"(predicted {result['C5'].get('registered_prediction')})")
    print(f"[OUTCOME] {result['outcome']}")
    print(f"[C] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
