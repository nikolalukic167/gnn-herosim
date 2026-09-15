#!/usr/bin/env python3
"""drainable_serving_config_v1 read -- C1..C5 over the batching configurations.

Bars are in docs/lineages/drainable_serving_config_v1.md; the constants below ARE the
bars and must not move once a number has been read. Sign convention is the live one:
"% vs X" = 100 * (X - arm) / X, positive means the arm is faster.
"""
import argparse, json, statistics as st, sys
from pathlib import Path

from scipy.stats import wilcoxon

ALPHA = 0.05
SEED_BAR = 12
N_SEEDS = 16
C1_EXPLAINS_MAX_PCT = 25.0      # gnn within 25% of Knative under A -> BATCHING-EXPLAINS
C1_DOES_NOT_MIN_PCT = 100.0     # still >100% slower -> BATCHING-DOES-NOT-EXPLAIN
C4_NOBATCH_MAX_MEAN = 1.05
C4_BATCH_MIN_MEAN = 4.0
C4_PAIR_RETENTION_MIN_PCT = 60.0
C5_PEER_SHARE_MIN_PCT = 20.0
C5_COOLDOWN_MAX_PCT = 5.0


def arm(d: Path, name: str) -> dict:
    p = d / f"{name}.summary.json"
    if not p.exists():
        raise SystemExit(f"FAIL LOUD: missing arm summary {p}")
    return json.loads(p.read_text())


def series(d: Path, a: str) -> list:
    return [arm(d, f"{a}_s{s}")["total_rtt"] for s in range(1, N_SEEDS + 1)]


def _verdict(p, wins, losses, better, worse):
    if p < ALPHA and wins >= SEED_BAR:
        return better
    if p < ALPHA and losses >= SEED_BAR:
        return worse
    return "TIE"


def paired(a, b, better, worse):
    diffs = [y - x for x, y in zip(a, b)]
    _, p = wilcoxon(a, b)
    wins, losses = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    ma, mb = st.median(a), st.median(b)
    return {"pct": 100.0 * (mb - ma) / mb, "p": float(p), "wins": wins, "losses": losses,
            "verdict": _verdict(p, wins, losses, better, worse)}


def vs_ref(a, ref, better, worse):
    diffs = [ref - x for x in a]
    _, p = wilcoxon(diffs)
    wins, losses = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    return {"pct": 100.0 * (ref - st.median(a)) / ref, "p": float(p), "wins": wins,
            "losses": losses, "verdict": _verdict(p, wins, losses, better, worse)}


def c4(d: Path, kind: str, ref_pairs: float) -> dict:
    mb, pin = [], []
    for a in ("gnn", "mpoff"):
        for s in range(1, N_SEEDS + 1):
            c = arm(d, f"{a}_s{s}").get("schedulerCounters") or {}
            if not c.get("prefix_batches"):
                raise SystemExit(f"FAIL LOUD: {a}_s{s} has no prefix_batches; C4 cannot be read")
            mb.append(c["prefix_tasks_decoded"] / c["prefix_batches"])
            pin.append(c["prefix_pairs_in_batch"])
    mean_batch, pairs = st.median(mb), st.median(pin)
    retention = 100.0 * pairs / ref_pairs if ref_pairs else 0.0
    if kind == "nobatch":
        holds, why = mean_batch <= C4_NOBATCH_MAX_MEAN, f"mean batch {mean_batch:.2f} <= {C4_NOBATCH_MAX_MEAN}"
    elif kind == "timewindow":
        holds, why = mean_batch >= C4_BATCH_MIN_MEAN, f"mean batch {mean_batch:.2f} >= {C4_BATCH_MIN_MEAN}"
    elif kind == "peergroup":
        holds = mean_batch >= C4_BATCH_MIN_MEAN and retention >= C4_PAIR_RETENTION_MIN_PCT
        why = (f"mean batch {mean_batch:.2f} >= {C4_BATCH_MIN_MEAN} and pair retention "
               f"{retention:.1f}% >= {C4_PAIR_RETENTION_MIN_PCT}%")
    else:
        raise SystemExit(f"FAIL LOUD: unknown configuration kind {kind!r}")
    return {"mean_batch": mean_batch, "pairs_in_batch": pairs, "pair_retention_pct": retention,
            "kind": kind, "holds": holds, "why": why}


def read_config(d: Path, label: str, kind: str, ref_pairs: float) -> dict:
    out = {"c4": c4(d, kind, ref_pairs)}
    print(f"[{label}] C4 {out['c4']['why']} -> holds={out['c4']['holds']}")
    if not out["c4"]["holds"]:
        out["verdict"] = "VOID"
        print(f"[{label}] VOID -- did not batch the way the configuration names; not read")
        return out
    gnn, mpoff = series(d, "gnn"), series(d, "mpoff")
    kn, knb = arm(d, "knative_network")["total_rtt"], arm(d, "knative_network_batch")["total_rtt"]
    best = min(kn, knb)
    out["c2"] = paired(gnn, mpoff, "GNN-NEEDED", "POINTWISE-BETTER")
    out["c3_gnn"] = vs_ref(gnn, best, "LEARNED-WINS", "REACTIVE-WINS")
    out["c3_mpoff"] = vs_ref(mpoff, best, "LEARNED-WINS", "REACTIVE-WINS")
    for k, t in (("c2", "C2 gnn vs mpoff   "), ("c3_gnn", "C3 gnn vs reactive"),
                 ("c3_mpoff", "C3 mpoff vs react ")):
        r = out[k]
        print(f"[{label}] {t} {r['pct']:+9.2f}%  p={r['p']:.4g}  {r['wins']}/{N_SEEDS} -> {r['verdict']}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", required=True, metavar="LABEL:KIND:DIR",
                    help="e.g. 'A nobatch:nobatch:/path/to/dir'; kind is nobatch|timewindow|peergroup")
    ap.add_argument("--reference-pairs", type=float, required=True,
                    help="prefix_pairs_in_batch of the full-window peer-group config (D)")
    ap.add_argument("--arrival-span-s", type=float, required=True)
    ap.add_argument("--c1-config", required=True, help="label of the nobatch configuration for C1")
    ap.add_argument("--output", required=True, type=Path)
    a = ap.parse_args()

    out, first = {"configs": {}}, None
    for spec in a.config:
        label, kind, d = spec.split(":", 2)
        out["configs"][label] = read_config(Path(d), label, kind, a.reference_pairs)
        first = first or Path(d)

    s = arm(first, "knative_network")
    peer = 100.0 * s["totalPeerExchangeTime"] / s["total_rtt"]
    cool = 100.0 * (1.0 - a.arrival_span_s / s["endTime"])
    out["c5"] = {"peer_share_pct": peer, "cooldown_pct": cool,
                 "holds": peer >= C5_PEER_SHARE_MIN_PCT and cool < C5_COOLDOWN_MAX_PCT}
    print(f"[C5] peer {peer:.2f}% cooldown {cool:.2f}% -> holds={out['c5']['holds']}")
    if not out["c5"]["holds"]:
        out["verdict"] = "VOID"
        print("[verdict] VOID -- C5 regime control failed")
        a.output.write_text(json.dumps(out, indent=2)); return 0

    nb = out["configs"].get(a.c1_config)
    if nb is None or nb.get("verdict") == "VOID":
        out["c1"] = {"verdict": "UNREADABLE"}
        print("[C1] UNREADABLE -- the nobatch configuration was not read")
    else:
        pct = nb["c3_gnn"]["pct"]        # negative == slower than reactive
        if pct >= -C1_EXPLAINS_MAX_PCT:
            v = "BATCHING-EXPLAINS"
        elif pct <= -C1_DOES_NOT_MIN_PCT:
            v = "BATCHING-DOES-NOT-EXPLAIN"
        else:
            v = "PARTIAL"
        out["c1"] = {"gnn_vs_reactive_pct": pct, "verdict": v}
        print(f"[C1] gnn vs reactive with zero batch wait: {pct:+.2f}% -> {v}")

    rescued = [l for l, c in out["configs"].items()
               if c.get("verdict") != "VOID"
               and c.get("c2", {}).get("verdict") == "GNN-NEEDED"
               and c.get("c3_gnn", {}).get("verdict") == "LEARNED-WINS"]
    out["rescued_configs"] = rescued
    out["verdict"] = (f"{out['c1']['verdict']}; rescued: {', '.join(rescued)}" if rescued
                      else f"{out['c1']['verdict']}; no configuration rescues the graph arm")
    print(f"[verdict] {out['verdict']}")
    a.output.write_text(json.dumps(out, indent=2))
    print(f"[read] wrote {a.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
