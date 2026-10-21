#!/usr/bin/env python3
"""Offered utilisation per directed link from one audit trace (HEROSIM_AUDIT_TRACE) of a switch-off bar cell: usage = sum over the cell's
remote peer exchanges of bytes / (bandwidth x 1024**2) on every directed link of the exchange's route; rho = usage / endTime.
usage: rho_extract.py <trace.jsonl> <new summary.json> <filed summary.json> <out.json>. Identity: endTime, total_rtt, num_tasks,
averageElapsedTime of the re-run equal the filed cell (FAIL LOUD otherwise). Link direction is recovered by walking the route from the
sender: each link key 'x|y' is traversed from the current node to the other end."""
import json, sys

trace, new_p, filed_p, out_p = sys.argv[1:5]
new, filed = json.load(open(new_p)), json.load(open(filed_p))
for k in ("endTime", "total_rtt", "num_tasks", "averageElapsedTime"):
    if new[k] != filed[k]:
        raise SystemExit(f"FAIL LOUD: {k} re-run {new[k]!r} filed {filed[k]!r}")
classes = new["accessClasses"]
usage, count, bw_of = {}, {}, {}
n = 0
with open(trace) as fh:
    for line in fh:
        if '"xfer"' not in line:
            continue
        r = json.loads(line)
        if r.get("k") != "xfer" or r.get("kind") != "peer":
            continue
        n += 1
        cur = r["src"]
        for key, bw in r["route"]:
            a, b = key.split("|")
            if cur not in (a, b):
                raise SystemExit(f"FAIL LOUD: route of {r['task']} not a path from {r['src']} at {key}")
            nxt = b if cur == a else a
            d = f"{cur}>{nxt}"
            usage[d] = usage.get(d, 0.0) + r["bytes"] / (bw * 1024 ** 2)
            count[d] = count.get(d, 0) + 1
            bw_of[d] = bw
            cur = nxt
        if cur != r["dst"]:
            raise SystemExit(f"FAIL LOUD: route of {r['task']} ends at {cur}, not {r['dst']}")
want = sum(v["transfers"] for v in new["peerExchangeByAccessClass"].values())
if n != want:
    raise SystemExit(f"FAIL LOUD: {n} peer transfers in the trace, {want} in the summary")
end = float(new["endTime"])
links = {}
for d, u in usage.items():
    a, b = d.split(">")
    if a in classes:
        cls = f"{classes[a]} out"
    elif b in classes:
        cls = f"{classes[b]} in"
    else:
        cls = "core"
    links[d] = {"rho": u / end, "usage_s": u, "bw": bw_of[d], "transfers": count[d], "class": cls}
json.dump({"endTime": end, "servers": new["servers"], "transfers": n, "classes_of_nodes": classes, "links": links}, open(out_p, "w"))
print(f"[rho] {n} transfers, {len(links)} directed links with traffic, max rho {max(v['rho'] for v in links.values()):.3f}")
