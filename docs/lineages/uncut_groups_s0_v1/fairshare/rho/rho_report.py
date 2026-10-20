#!/usr/bin/env python3
"""Offered link utilisation, descriptive (SCIENTIST's A1 ruling 3; uncut_groups_s0_v1). Reads anomaly/fairshare/rho/<size>/<rung>_<topology>.json written by
rho_extract.py (the bar's switch-off g0 cells re-run with the audit trace, each checked equal to its filed cell). rho = sum over the run's remote peer
exchanges of bytes / (bandwidth x 1024**2) on the directed link / endTime. Per size and load (12 topologies each):
  server access links (every server has an 'out' and an 'in' link, 2 x servers per cell): share with rho > 1 and with rho >= 1, by access class and direction;
  links carrying traffic: count, share with rho > 1 / >= 1, core links apart; max and median rho.
Size-wise verdict line: whether any link of the heavy load has rho >= 1 (then the read labels every number of that size as overload)."""
import glob, json, os, statistics as st, sys

U = "/share/nikola.lukic/uncut_groups"; R = f"{U}/anomaly/fairshare/rho"
TOPOS = {24: range(19537, 19549), 64: range(30129, 30141), 128: range(30241, 30253)}
out = [__doc__.strip(), ""]
verdict = []
for size, topos in TOPOS.items():
    for rung in ("moderate", "heavy"):
        cells = []
        for t in topos:
            p = f"{R}/{size}/{rung}_{t}.json"
            if not os.path.exists(p):
                raise SystemExit(f"FAIL LOUD: missing {p}")
            cells.append(json.load(open(p)))
        n = len(cells)
        acc = {}   # (class, dir) -> [total links, links>1, links>=1]
        core = [0, 0, 0]
        rhos, allrho = [], []
        for c in cells:
            servers = c["servers"]
            for node, cls in c["classes_of_nodes"].items():
                for d in ("out", "in"):
                    acc.setdefault((cls, d), [0, 0, 0])[0] += 1
            for d, v in c["links"].items():
                r = v["rho"]; allrho.append(r)
                if v["class"] == "core":
                    core[0] += 1; core[1] += r > 1; core[2] += r >= 1
                else:
                    cls, dr = v["class"].split()
                    acc[(cls, dr)][1] += r > 1; acc[(cls, dr)][2] += r >= 1
        tot = sum(a[0] for a in acc.values()); gt1 = sum(a[1] for a in acc.values()); ge1 = sum(a[2] for a in acc.values())
        out.append(f"=== {size} servers, {rung}: {n} topologies (g0); {sum(len(c['links']) for c in cells)} directed links carried traffic; "
                   f"max rho {max(allrho):.3f}, median over links with traffic {st.median(allrho):.3f}")
        out.append(f"    server access links (2 x servers per cell): {gt1}/{tot} = {100 * gt1 / tot:.1f} % with rho > 1; {ge1}/{tot} = {100 * ge1 / tot:.1f} % with rho >= 1")
        for (cls, d), (tt, g, ge) in sorted(acc.items()):
            out.append(f"      {cls:9s} {d:3s}: {g:4d}/{tt:4d} with rho > 1 ({100 * g / tt:5.1f} %), {ge:4d} with rho >= 1")
        out.append(f"    core links with traffic: {core[0]} ; rho > 1: {core[1]} ; rho >= 1: {core[2]}")
        top = sorted(((v["rho"], v["class"], d, t) for t, c in zip(topos, cells) for d, v in c["links"].items()), reverse=True)[:5]
        out.append("    five largest: " + "; ".join(f"{r:.2f} ({cl}, topology {t})" for r, cl, d, t in top))
        verdict.append((size, rung, ge1 + core[2]))
out.append("")
for size in TOPOS:
    h = [x for x in verdict if x[0] == size and x[1] == "heavy"][0]
    out.append(f"size {size}: heavy load has {h[2]} links with rho >= 1 over the 12 topologies -> {'OVERLOAD label applies' if h[2] else 'no overload label'}")
txt = "\n".join(out); print(txt)
dst = f"{U}/anomaly/fairshare/rho_report.txt"
if os.path.exists(dst):
    raise SystemExit("FAIL LOUD: written once")
open(dst + ".partial", "w").write(txt + "\n"); os.replace(dst + ".partial", dst)
