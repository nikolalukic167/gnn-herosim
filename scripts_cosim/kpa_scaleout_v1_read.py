#!/usr/bin/env python3
"""kpa_scaleout_v1 read (docs/lineages/kpa_scaleout_v1.md).

Per condition x rung, under HEROSIM_SCALEOUT=kpa: each arm's latency, queue share and exchange (median over the 19
topologies of per-topology means), and the paired % vs CD (per topology: median over windows x seeds of the paired %;
then the median over topologies, exact two-sided Wilcoxon over topologies). Primary family: the 5 non-CD arms
(reactive, selfpredict, locality, batched, so1load zero-shot) x 3 rungs x 4 conditions = 60 tests, Holm across all.
Also the scale-out instrumentation (share of load-caused creations, short-lived replicas, panic entries) and the
legacy reference from transfer_physics_v1's tp1_read.json for predictions 2 and 5.

  kpa_scaleout_v1_read.py [--out kpa_read.json]       (run on datalab; TP1_EXCLUDE=9485 for the sensitivity read)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st

from scipy.stats import wilcoxon

B = "/home/nikola.lukic/gnn-herosim/simulation_data/"
EXCLUDE = {int(x) for x in os.environ.get("TP1_EXCLUDE", "").split(",") if x}
TOPOS = [t for t in json.load(open(B + "client_local_v1/inputs_so_server/selected.json"))["topologies"] if t not in EXCLUDE]
CONDS = {"sf_held": "gate_replay_kpa", "pipe": "gate_pipe_kpa", "release": "gate_release_kpa",
         "pipe_release": "gate_pipe_release_kpa"}
ARMS = ["reactive", "selfpredict", "locality", "batched", "cd", "so1load_selfref"]
FAMILY = ["reactive", "selfpredict", "locality", "batched", "so1load_selfref"]
SEEDS = {"so1load_selfref": (1, 2)}
RUNGS = ("20", "30", "50")


def load(d):
    s = {}
    for f in glob.glob(B + "kpa_scaleout_v1/" + d + "/*.summary.json"):
        r = json.load(open(f))
        k = r["arm"].split("__")[2].rsplit("_s", 1)[0]
        s[(int(r["topology"]), r["window"], k, int(r["checkpoint_seed"]))] = r
    return s


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    a = ap.parse_args()
    legacy = json.load(open(B + "transfer_physics_v1/tp1_read.json"))
    out, tests = {}, []
    for cond, d in CONDS.items():
        s = load(d)
        out[cond] = {}
        for rung in RUNGS:
            ws = [f"g{i}x{rung}" for i in range(4)]
            rows = {}
            for arm in ARMS:
                seeds = SEEDS.get(arm, (0,))
                lat, qs, ex, pc, load_share, short, panic = [], [], [], [], [], [], []
                for t in TOPOS:
                    rs = [s[(t, w, arm, sd)] for w in ws for sd in seeds if (t, w, arm, sd) in s]
                    if not rs:
                        continue
                    lat.append(st.mean(r["averageElapsedTime"] for r in rs))
                    qs.append(st.mean(r["queue_share"] for r in rs))
                    ex.append(st.mean(r["totalPeerExchangeTime"] / r["num_tasks"] for r in rs))
                    for r in rs:
                        so = r["scaleOut"]
                        c = so["scale_ups_by_cause"]
                        load_share.append(c["load"] / max(1, c["load"] + c["reachability"]))
                        short.append(so["replica_lifetimes_within_stable_window"] / max(1, so["replica_lifetimes_closed"]))
                        panic.append(so["panic_entries"])
                    if arm != "cd":
                        p = [100 * (s[(t, w, arm, sd)]["averageElapsedTime"] / s[(t, w, "cd", 0)]["averageElapsedTime"] - 1)
                             for w in ws for sd in seeds if (t, w, arm, sd) in s and (t, w, "cd", 0) in s]
                        if p:
                            pc.append(st.median(p))
                row = {"n": len(lat), "lat": st.median(lat) if lat else None, "qshare": st.median(qs) if qs else None,
                       "exch": st.median(ex) if ex else None, "load_caused_share": st.median(load_share) if load_share else None,
                       "short_lived_share": st.median(short) if short else None,
                       "panic_entries": st.median(panic) if panic else None}
                if arm != "cd" and pc:
                    row.update(vs_cd=st.median(pc), faster=sum(x < 0 for x in pc), n_pc=len(pc),
                               p=float(wilcoxon(pc).pvalue) if any(pc) else 1.0)
                    if arm in FAMILY:
                        tests.append((cond, rung, arm))
                leg = (legacy.get(cond) or {}).get(rung, {}).get(arm)
                if leg:
                    row["legacy_lat"], row["legacy_qshare"], row["legacy_vs_cd"] = leg["lat"], leg["qshare"], leg.get("vs_cd")
                rows[arm] = row
            out[cond][rung] = rows
    adj = holm([out[c][r][k]["p"] for c, r, k in tests])
    for (c, r, k), pa in zip(tests, adj):
        row = out[c][r][k]
        row["p_holm"] = pa
        row["label"] = ("CONFIRMED" if row["vs_cd"] <= -5 and pa < 0.05 else
                        "CD-FASTER" if row["vs_cd"] > 0 and pa < 0.05 else
                        "DIRECTION" if pa < 0.05 else "NOT-SEPARATED")
    out["_meta"] = {"family_size": len(tests), "topologies": TOPOS, "excluded": sorted(EXCLUDE)}
    text = json.dumps(out, indent=1)
    if a.out:
        open(a.out, "w").write(text)
    for cond in CONDS:
        for rung in RUNGS:
            print(f"== {cond} x{rung[0]}.{rung[1]}")
            for arm, r in out[cond][rung].items():
                vs = f"vs CD {r['vs_cd']:+6.1f}% ({r['faster']}/{r['n_pc']} faster, Holm p {r['p_holm']:.2g}, {r['label']})" if "p_holm" in r else ""
                leg = f" | legacy lat {r['legacy_lat']:.2f} q {r['legacy_qshare']:.2f}" if "legacy_lat" in r else ""
                print(f"  {arm:16s} lat {r['lat']:7.2f} q {r['qshare']:.2f} exch {r['exch']:.2f} load% {r['load_caused_share']:.2f} "
                      f"short {r['short_lived_share']:.2f} panic {r['panic_entries']:.0f} {vs}{leg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
