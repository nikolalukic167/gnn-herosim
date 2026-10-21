#!/usr/bin/env python3
"""Fair-share diagnostic report (uncut_groups_s0_v1; reads anomaly/fairshare/rho3/<size>/<arm>/<window>_<rung>_<topology>.json from rho_extract2.py). Descriptive.
Arms: bar (cd_ib0_inflightfull_k100 s0), gnn (FL gnn s1), mlp (FL MLP s1), each a switch-off cell re-run with the audit trace and checked equal to its filed cell.
rho(link) = sum over the run's transfers of bytes / (bandwidth x 1024**2) on the directed link / endTime. Per size, load, arm and window (g0-g3) and pooled over the
four windows, over the 12 topologies:
  PEER exchange (server -> server; placement-determined): server access links with rho > 1 / (2 x servers per cell); CELLULAR-server uplinks with rho > 1 /
    cellular servers per cell; max rho; share of exchange BYTES (and transfers) that cross at least one rho > 1 link, and by the type of the over-1 link crossed.
  INGRESS (client -> server input; the first hop is the client's own uplink): client uplinks with rho > 1 (FORCED: the bytes leaving a client are the same wherever the
    task is placed; checked equal across the arms), server downlinks with rho > 1 (placement-driven), core links with rho > 1; share of ingress bytes crossing
    an over-1 link.
Counts of a window are summed over its 12 topologies; the pooled row sums the windows present. Missing cells or arms fail loud. The report also prints the node and
the physical cores of the jobs that produced the cells (rho3/<size>/node.txt)."""
import argparse, json, os, statistics as st, sys

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/share/nikola.lukic/uncut_groups/anomaly/fairshare/rho3")
ap.add_argument("--out", default="/share/nikola.lukic/uncut_groups/anomaly/fairshare/rho3_report.txt")
ap.add_argument("--topos", type=int, default=12)
ap.add_argument("--sizes", default="24,64,128")
ap.add_argument("--windows", default="g0,g1,g2,g3")
a = ap.parse_args()
TOP = {24: 19537, 64: 30129, 128: 30241}
ARMS = ("bar", "gnn", "mlp")
WINS = a.windows.split(",")
out = [__doc__.strip(), ""]


def load(size, arm, w, rung):
    cells = []
    for i in range(a.topos):
        p = f"{a.root}/{size}/{arm}/{w}_{rung}_{TOP[size] + i}.json"
        if not os.path.exists(p):
            raise SystemExit(f"FAIL LOUD: missing {p}")
        cells.append(json.load(open(p)))
    return cells


def stats(cells):
    """counts summed over cells"""
    s = {"srv_links": 0, "srv_over": 0, "cel_srv": 0, "cel_over": 0, "max_peer": 0.0, "peer_b": 0.0, "peer_cross_b": 0.0, "peer_n": 0, "peer_cross_n": 0,
         "by_type": {}, "cli_links": 0, "cli_over": 0, "cli_cel": 0, "cli_cel_over": 0, "srvin_over": 0, "core_over": 0, "ing_b": 0.0, "ing_cross_b": 0.0,
         "ing_n": 0, "ing_cross_n": 0, "max_ing": 0.0, "cli_usage": {}}
    for c in cells:
        nodes = c["nodes"]
        servers = {n for n, (r, k) in nodes.items() if r == "server"}
        clients = {n for n, (r, k) in nodes.items() if r == "client"}
        s["srv_links"] += 2 * len(servers); s["cli_links"] += len(clients)
        s["cel_srv"] += sum(1 for n in servers if nodes[n][1] == "cellular"); s["cli_cel"] += sum(1 for n in clients if nodes[n][1] == "cellular")
        pk, ik = c["kinds"]["peer"], c["kinds"]["ingress"]
        for d, v in pk["links"].items():
            s["max_peer"] = max(s["max_peer"], v["rho"])
            if v["type"].startswith("server") and v["rho"] > 1:
                s["srv_over"] += 1
                if v["type"] == "server cellular out":
                    s["cel_over"] += 1
        s["peer_b"] += pk["bytes"]; s["peer_cross_b"] += pk["cross_bytes"]; s["peer_n"] += pk["transfers"]; s["peer_cross_n"] += pk["cross_transfers"]
        for t, v in pk["cross_by_type"].items():
            x = s["by_type"].setdefault(t, 0.0); s["by_type"][t] = x + v["bytes"]
        for d, v in ik["links"].items():
            s["max_ing"] = max(s["max_ing"], v["rho"])
            if v["rho"] > 1:
                if v["type"].startswith("client") and v["type"].endswith("out"):
                    s["cli_over"] += 1
                    if "cellular" in v["type"]:
                        s["cli_cel_over"] += 1
                elif v["type"].startswith("server"):
                    s["srvin_over"] += 1
                elif v["type"] == "core":
                    s["core_over"] += 1
            if v["type"].startswith("client") and v["type"].endswith("out"):
                s["cli_usage"][d] = s["cli_usage"].get(d, 0.0) + v["usage_s"]
        s["ing_b"] += ik["bytes"]; s["ing_cross_b"] += ik["cross_bytes"]; s["ing_n"] += ik["transfers"]; s["ing_cross_n"] += ik["cross_transfers"]
    return s


def pct(x, y):
    return f"{100 * x / y:5.1f} %" if y else "  n/a  "


def merge(ss):
    m = {}
    for s in ss:
        for k, v in s.items():
            if k == "by_type":
                for t, b in v.items():
                    m.setdefault(k, {}); m[k][t] = m[k].get(t, 0.0) + b
            elif k == "cli_usage":
                m.setdefault(k, {})
                for d, u in v.items():
                    m[k][d] = m[k].get(d, 0.0) + u
            elif k.startswith("max"):
                m[k] = max(m.get(k, 0.0), v)
            else:
                m[k] = m.get(k, 0) + v
    return m


for size in [int(x) for x in a.sizes.split(",")]:
    node = f"{a.root}/{size}/node.txt"
    out.append(f"##### {size} servers; producing job: " + (open(node).read().strip() if os.path.exists(node) else "node file missing"))
    for rung in ("moderate", "heavy"):
        per = {(arm, w): stats(load(size, arm, w, rung)) for arm in ARMS for w in WINS}
        out.append(f"=== {size} servers, {rung} ({a.topos} topologies per window)")
        out.append("  PEER exchange: server access links rho>1 | CELLULAR-server uplinks rho>1 | max rho | bytes crossing a rho>1 link | transfers crossing")
        for arm in ARMS:
            for w in WINS + ["pooled"]:
                s = merge([per[(arm, x)] for x in WINS]) if w == "pooled" else per[(arm, w)]
                bt = "; ".join(f"{t} {pct(b, s['peer_b']).strip()}" for t, b in sorted(s.get("by_type", {}).items(), key=lambda kv: -kv[1])[:4])
                out.append(f"    {arm:3s} {w:6s} srv {s['srv_over']:4d}/{s['srv_links']:5d} {pct(s['srv_over'], s['srv_links'])} | cel-up {s['cel_over']:4d}/{s['cel_srv']:4d} "
                           f"{pct(s['cel_over'], s['cel_srv'])} | max {s['max_peer']:6.2f} | bytes {pct(s['peer_cross_b'], s['peer_b'])} | transfers {pct(s['peer_cross_n'], s['peer_n'])} | by type: {bt}")
        out.append("  INGRESS (client -> server): FORCED client uplinks rho>1 (cellular among them) | placement-driven server downlinks rho>1 | core links rho>1 | max rho | bytes crossing a rho>1 link")
        for arm in ARMS:
            for w in WINS + ["pooled"]:
                s = merge([per[(arm, x)] for x in WINS]) if w == "pooled" else per[(arm, w)]
                out.append(f"    {arm:3s} {w:6s} client-up {s['cli_over']:4d}/{s['cli_links']:5d} (cellular {s['cli_cel_over']:4d}/{s['cli_cel']:4d}) | server-down {s['srvin_over']:4d} | core {s['core_over']:4d} "
                           f"| max {s['max_ing']:6.2f} | bytes {pct(s['ing_cross_b'], s['ing_b'])}")
        ok = all(abs(per[(arm, w)]["cli_usage"].get(d, 0.0) - per[("bar", w)]["cli_usage"].get(d, 0.0)) <= 1e-6 * max(1.0, per[("bar", w)]["cli_usage"].get(d, 0.0))
                 for arm in ("gnn", "mlp") for w in WINS for d in set(per[(arm, w)]["cli_usage"]) | set(per[("bar", w)]["cli_usage"]))
        out.append(f"  client uplink usage equal across the three arms in every window (forced = placement-independent): {ok}")
    out.append("")
txt = "\n".join(out); print(txt)
if os.path.exists(a.out):
    raise SystemExit("FAIL LOUD: written once")
open(a.out + ".partial", "w").write(txt + "\n"); os.replace(a.out + ".partial", a.out)
