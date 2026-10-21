#!/usr/bin/env python3
"""Fair-share utilisation report v3 (uncut_groups_s0_v1; descriptive; no rerun). Reads the same cells as rho_report2.py
(anomaly/fairshare/rho3/<size>/<arm>/<window>_<rung>_<topology>.json from rho_extract2.py) but sums the two kinds on each directed link:
rho_sum(link) = (peer usage_s + ingress usage_s) / endTime, where usage_s = sum bytes / (bw x 1024**2) as in rho_extract2.py.
Per size, load and arm (bar, gnn, mlp), pooled over windows g0-g3 and the 12 topologies: for every link type (role + access class + direction; core) the
number of links with rho_sum > 1 over the links of that type (access links: every node's in and out link; core: links that carry any transfer, the cells
do not list unused core links), the max rho_sum and the type and name of the link that attains it. Also the peer-only count of CELLULAR-server uplinks with
rho > 1 next to the summed count, and the check that no ingress transfer uses a server uplink (so the two counts must be equal).
Missing cells fail loud."""
import json, os, sys

ROOT = "/share/nikola.lukic/uncut_groups/anomaly/fairshare/rho3"
OUT = "/share/nikola.lukic/uncut_groups/anomaly/fairshare/rho3_report_v3.txt"
TOP = {24: 19537, 64: 30129, 128: 30241}
ARMS, WINS, N = ("bar", "gnn", "mlp"), ("g0", "g1", "g2", "g3"), 12
out = [__doc__.strip(), ""]


def ltype(role, cls, d):
    return f"{role} {cls} {d}"


def stats(cells):
    s = {"n": {}, "over": {}, "max": (0.0, "", ""), "cel_peer_over": 0, "cel_sum_over": 0, "cel_n": 0, "ing_on_server_up": 0.0}
    for c in cells:
        nodes = c["nodes"]
        for node, (role, cls) in nodes.items():
            for d in ("out", "in"):
                t = ltype(role, cls, d); s["n"][t] = s["n"].get(t, 0) + 1
        pk, ik = c["kinds"]["peer"]["links"], c["kinds"]["ingress"]["links"]
        end = c["endTime"]
        tot = {}
        for d, v in pk.items():
            tot[d] = [v["usage_s"], v["type"]]
        for d, v in ik.items():
            if d in tot:
                if tot[d][1] != v["type"]:
                    raise SystemExit(f"FAIL LOUD: link {d} typed {tot[d][1]!r} and {v['type']!r}")
                tot[d][0] += v["usage_s"]
            else:
                tot[d] = [v["usage_s"], v["type"]]
            if v["type"].startswith("server") and v["type"].endswith("out"):
                s["ing_on_server_up"] += v["usage_s"]
        for d, (u, t) in tot.items():
            r = u / end
            if r > 1:
                s["over"][t] = s["over"].get(t, 0) + 1
            if r > s["max"][0]:
                s["max"] = (r, t, d)
        for d, (u, t) in tot.items():
            if t == "core":
                k = "core (links carrying any transfer)"
                s["n"][k] = s["n"].get(k, 0) + 1
        s["cel_n"] += sum(1 for n, (role, cls) in nodes.items() if role == "server" and cls == "cellular")
        for d, v in pk.items():
            if v["type"] == "server cellular out" and v["rho"] > 1:
                s["cel_peer_over"] += 1
        for d, (u, t) in tot.items():
            if t == "server cellular out" and u / end > 1:
                s["cel_sum_over"] += 1
    return s


def merge(ss):
    m = {"n": {}, "over": {}, "max": (0.0, "", ""), "cel_peer_over": 0, "cel_sum_over": 0, "cel_n": 0, "ing_on_server_up": 0.0}
    for s in ss:
        for k in ("n", "over"):
            for t, v in s[k].items():
                m[k][t] = m[k].get(t, 0) + v
        for k in ("cel_peer_over", "cel_sum_over", "cel_n", "ing_on_server_up"):
            m[k] += s[k]
        if s["max"][0] > m["max"][0]:
            m["max"] = s["max"]
    return m


allok = True
for size in (24, 64, 128):
    out.append(f"##### {size} servers; producing job: " + open(f"{ROOT}/{size}/node.txt").read().strip())
    for rung in ("moderate", "heavy"):
        out.append(f"=== {size} servers, {rung}: links with rho_sum > 1 / links of the type, pooled over g0-g3 x {N} topologies (types with none over 1 omitted)")
        for arm in ARMS:
            cells = []
            for w in WINS:
                for i in range(N):
                    p = f"{ROOT}/{size}/{arm}/{w}_{rung}_{TOP[size] + i}.json"
                    if not os.path.exists(p):
                        raise SystemExit(f"FAIL LOUD: missing {p}")
                    cells.append(json.load(open(p)))
            s = stats(cells)
            rows = "; ".join(f"{t} {s['over'][t]}/{s['n'][t]} ({100 * s['over'][t] / s['n'][t]:.2f} %)" for t in sorted(s["over"]))
            r, t, d = s["max"]
            out.append(f"    {arm:3s} {rows or 'none over 1'} | max rho_sum {r:.2f} on a '{t}' link ({d})")
            ok = s["cel_peer_over"] == s["cel_sum_over"] and s["ing_on_server_up"] == 0.0
            allok &= ok
            out.append(f"        cellular-server uplinks rho>1: peer-only {s['cel_peer_over']}/{s['cel_n']}, summed {s['cel_sum_over']}/{s['cel_n']}; ingress usage on server uplinks {s['ing_on_server_up']:.1f} s; unchanged: {ok}")
    out.append("")
out.append(f"cellular-server uplink counts unchanged by the sum, and no ingress usage on any server uplink, in every size/load/arm: {allok}")
txt = "\n".join(out); print(txt)
if os.path.exists(OUT):
    raise SystemExit("FAIL LOUD: written once")
open(OUT + ".partial", "w").write(txt + "\n"); os.replace(OUT + ".partial", OUT)
