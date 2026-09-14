#!/usr/bin/env python3
"""drainable_regime_v1 S1 read -- the registered live-gate read at the x4000 rung.

Bars are in docs/lineages/drainable_regime_v1.md, stage 2, and are not repeated as prose
here; the constants below ARE the bars and must not move once a number has been read.

Sign convention is the live one used everywhere in peer_affinity_v1: "% vs X" is
100 * (X - arm) / X, so positive means the arm is faster than X.
"""
import argparse, json, sys
from pathlib import Path

from scipy.stats import wilcoxon

ALPHA = 0.05
SEED_BAR = 12          # B1/B2: seeds that must fall the right way
N_SEEDS = 16
B4_PEER_SHARE_MIN_PCT = 20.0
B4_COOLDOWN_MAX_PCT = 5.0
B5_DIM7_P90_MAX = 42.0
B6_INCOMPLETE_BATCH_MAX_PCT = 20.0   # peer-group assembly control (S1 Amendment 1)


def load_arm(d: Path, name: str) -> dict:
    p = d / f"{name}.summary.json"
    if not p.exists():
        raise SystemExit(f"FAIL LOUD: missing arm summary {p}")
    return json.loads(p.read_text())


def seed_series(d: Path, arm: str) -> list:
    return [load_arm(d, f"{arm}_s{s}")["total_rtt"] for s in range(1, N_SEEDS + 1)]


def paired(a: list, b: list) -> dict:
    """a vs b, paired by seed. Positive pct means a is faster than b."""
    diffs = [y - x for x, y in zip(a, b)]          # b - a, positive when a faster
    stat, p = wilcoxon(a, b)
    wins = sum(1 for v in diffs if v > 0)
    losses = sum(1 for v in diffs if v < 0)
    med_a = sorted(a)[len(a) // 2]
    med_b = sorted(b)[len(b) // 2]
    return {"pct": 100.0 * (med_b - med_a) / med_b, "p": float(p),
            "wins": wins, "losses": losses, "n": len(a),
            "median_arm": med_a, "median_ref": med_b}


def one_sample(a: list, ref: float) -> dict:
    """Each seed of a against a single reference run. Positive pct means a is faster."""
    diffs = [ref - x for x in a]
    stat, p = wilcoxon(diffs)
    wins = sum(1 for v in diffs if v > 0)
    losses = sum(1 for v in diffs if v < 0)
    med_a = sorted(a)[len(a) // 2]
    return {"pct": 100.0 * (ref - med_a) / ref, "p": float(p),
            "wins": wins, "losses": losses, "n": len(a),
            "median_arm": med_a, "median_ref": ref}


def verdict(r: dict, better: str, worse: str) -> str:
    if r["p"] < ALPHA and r["wins"] >= SEED_BAR:
        return better
    if r["p"] < ALPHA and r["losses"] >= SEED_BAR:
        return worse
    return "TIE"


def read_b6(d: Path, label: str) -> dict:
    """Peer-group assembly control. GNN_BATCH_TIMEOUT is a policy time constant: at a
    stretched arrival rate a window sized for the landed gate expires before a group can
    co-arrive, the decoder sees singletons, and B1/B2 then compare two arms that both ran
    pointwise. Measured from the arms' own counters, not inferred (S1 Amendment 1)."""
    worst = None
    for arm in ("gnn", "mpoff"):
        for s in range(1, N_SEEDS + 1):
            c = load_arm(d, f"{arm}_s{s}").get("schedulerCounters") or {}
            batches = c.get("prefix_batches")
            if not batches:
                raise SystemExit(
                    f"FAIL LOUD: {arm}_s{s} has no prefix_batches counter; B6 cannot be read")
            pct = 100.0 * (c.get("peer_group_incomplete_batches") or 0) / batches
            if worst is None or pct > worst[0]:
                worst = (pct, f"{arm}_s{s}", batches, c.get("peer_group_incomplete_batches"))
    holds = worst[0] <= B6_INCOMPLETE_BATCH_MAX_PCT
    print(f"[{label}] B6 worst incomplete peer-group batches {worst[0]:.2f}% ({worst[1]}: "
          f"{worst[3]}/{worst[2]}) (<= {B6_INCOMPLETE_BATCH_MAX_PCT}) -> holds={holds}")
    return {"worst_pct": worst[0], "worst_arm": worst[1], "holds": holds}


def read_config(d: Path, label: str) -> dict:
    gnn, mpoff = seed_series(d, "gnn"), seed_series(d, "mpoff")
    kn = load_arm(d, "knative_network")["total_rtt"]
    knb = load_arm(d, "knative_network_batch")["total_rtt"]
    best_kn, best_kn_name = (kn, "knative_network") if kn <= knb else (knb, "knative_network_batch")

    b1 = paired(gnn, mpoff)
    b1["verdict"] = verdict(b1, "GNN-NEEDED", "POINTWISE-BETTER")
    b2 = one_sample(gnn, best_kn)
    b2["verdict"] = verdict(b2, "LEARNED-WINS", "REACTIVE-WINS")
    b2["reference"] = best_kn_name
    # descriptive, not a bar: the pointwise twin against the same reactive reference
    b2m = one_sample(mpoff, best_kn)
    b2m["reference"] = best_kn_name

    print(f"[{label}] B1 gnn vs mpoff      {b1['pct']:+7.2f}%  p={b1['p']:.4g}  "
          f"{b1['wins']}/{b1['n']} -> {b1['verdict']}")
    print(f"[{label}] B2 gnn vs {b2['reference']:<21s} {b2['pct']:+7.2f}%  p={b2['p']:.4g}  "
          f"{b2['wins']}/{b2['n']} -> {b2['verdict']}")
    print(f"[{label}] .. mpoff vs {b2m['reference']:<19s} {b2m['pct']:+7.2f}%  p={b2m['p']:.4g}  "
          f"{b2m['wins']}/{b2m['n']} (descriptive)")
    return {"b1": b1, "b2": b2, "mpoff_vs_reactive": b2m}


def read_b4(d: Path, arrival_span_s: float) -> dict:
    s = load_arm(d, "knative_network")
    peer = 100.0 * s["totalPeerExchangeTime"] / s["total_rtt"]
    cool = 100.0 * (1.0 - arrival_span_s / s["endTime"])
    holds = peer >= B4_PEER_SHARE_MIN_PCT and cool < B4_COOLDOWN_MAX_PCT
    print(f"[B4] peer {peer:.2f}% (>= {B4_PEER_SHARE_MIN_PCT}) "
          f"cooldown {cool:.2f}% (< {B4_COOLDOWN_MAX_PCT}) -> holds={holds}")
    return {"peer_share_pct": peer, "cooldown_pct": cool, "holds": holds}


def read_b5(snapshots: Path) -> dict:
    if snapshots is None or not snapshots.exists():
        print("[B5] no snapshot file given -- B1/B2 recorded without the queue-range control")
        return {"dim7_p90": None, "holds": None}
    # LIVE_AUDIT snapshots carry queue depth in `full_queue_snapshot` (queue_key -> depth);
    # `candidates` is the co-sim dataset schema and does NOT exist here. Reading the wrong
    # key returns an empty busy set, which would pass this bar vacuously -- so an empty
    # parse is a loud failure, not a pass (2026-09-14).
    busy, rows = [], 0
    for line in snapshots.read_text().splitlines():
        if not line.strip():
            continue
        rows += 1
        for depth in (json.loads(line).get("full_queue_snapshot") or {}).values():
            if isinstance(depth, (int, float)) and depth > 0:
                busy.append(depth)
    if rows == 0:
        raise SystemExit(f"FAIL LOUD: no snapshots parsed from {snapshots}")
    if not busy:
        print(f"[B5] {rows} snapshots, no platform ever carries a queue -- "
              f"the bar is vacuous at this rate, recorded as NOT-APPLICABLE")
        return {"dim7_p90": None, "n_busy": 0, "n_snapshots": rows, "holds": None}
    busy.sort()
    p90 = busy[min(len(busy) - 1, int(0.9 * len(busy)))]
    holds = p90 <= B5_DIM7_P90_MAX
    print(f"[B5] dim-7 p90 over {len(busy)} busy platform-queues = {p90} "
          f"(<= {B5_DIM7_P90_MAX}) -> holds={holds}")
    return {"dim7_p90": p90, "n_busy": len(busy), "n_snapshots": rows, "holds": holds}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capped-dir", required=True, type=Path)
    ap.add_argument("--uncapped-dir", required=True, type=Path)
    ap.add_argument("--snapshots", type=Path)
    ap.add_argument("--arrival-span-s", type=float, required=True)
    ap.add_argument("--output", required=True, type=Path)
    a = ap.parse_args()

    out = {"bars": {"alpha": ALPHA, "seed_bar": SEED_BAR, "n_seeds": N_SEEDS,
                    "b4_peer_min_pct": B4_PEER_SHARE_MIN_PCT,
                    "b4_cooldown_max_pct": B4_COOLDOWN_MAX_PCT,
                    "b5_dim7_p90_max": B5_DIM7_P90_MAX}}
    out["b4"] = read_b4(a.capped_dir, a.arrival_span_s)
    out["b5"] = read_b5(a.snapshots)
    if not out["b4"]["holds"]:
        out["verdict"] = "VOID"
        print("[verdict] VOID -- B4 failed; B1-B3 are not read")
        a.output.write_text(json.dumps(out, indent=2))
        return 0

    out["b6"] = {"capped": read_b6(a.capped_dir, "capped"),
                 "uncapped": read_b6(a.uncapped_dir, "uncapped")}
    if not (out["b6"]["capped"]["holds"] and out["b6"]["uncapped"]["holds"]):
        out["verdict"] = "CONFOUNDED"
        print("[verdict] CONFOUNDED -- the peer groups never assembled, so B1/B2 would "
              "compare two arms that both decoded singletons")
        a.output.write_text(json.dumps(out, indent=2))
        return 0

    out["capped"] = read_config(a.capped_dir, "capped")
    out["uncapped"] = read_config(a.uncapped_dir, "uncapped")

    same_b1 = out["capped"]["b1"]["verdict"] == out["uncapped"]["b1"]["verdict"]
    same_b2 = out["capped"]["b2"]["verdict"] == out["uncapped"]["b2"]["verdict"]
    out["b3_cap_contingent"] = not (same_b1 and same_b2)
    print(f"[B3] cap-contingent = {out['b3_cap_contingent']}")

    head = out["capped"]["b1"]["verdict"]
    if out["b5"]["holds"] is False:
        head = f"{head} (OUT-OF-RANGE: B5 failed)"
    if out["b3_cap_contingent"]:
        head = f"{head}, CAP-CONTINGENT"
    out["verdict"] = head
    print(f"[verdict] {head}")
    a.output.write_text(json.dumps(out, indent=2))
    print(f"[read] wrote {a.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
