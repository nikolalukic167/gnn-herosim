#!/usr/bin/env python3
"""Offered utilisation per directed link from one audit trace of a switch-off cell (uncut_groups_s0_v1 fair-share diagnostic; v2 of rho_extract.py).
usage: rho_extract2.py <trace.jsonl> <new summary.json> <filed summary.json> <out.json>
Identity: endTime, total_rtt, num_tasks, averageElapsedTime of the re-run equal the filed cell (FAIL LOUD otherwise), and the peer transfers in the trace equal the
summary's count. Two kinds of transfer are kept apart: 'peer' (server -> server exchange, placement-determined) and 'ingress' (client -> server input, whose first
hop is the client's own uplink: the bytes leaving a client are the same wherever the task is placed). Per kind and directed link: usage = sum bytes / (bw x 1024**2),
rho = usage / endTime, bytes, transfers, access class and role (server / client access link, or core). Link direction is recovered by walking the route from the
sender. Per kind also: bytes and transfers in total, bytes and transfers that cross at least one link with rho > 1 of that kind, and the same split by the type of
the over-1 links crossed (role + class + direction)."""
import json, sys

trace, new_p, filed_p, out_p = sys.argv[1:5]
new, filed = json.load(open(new_p)), json.load(open(filed_p))
for k in ("endTime", "total_rtt", "num_tasks", "averageElapsedTime"):
    if new[k] != filed[k]:
        raise SystemExit(f"FAIL LOUD: {k} re-run {new[k]!r} filed {filed[k]!r}")
classes = new["accessClasses"]
end = float(new["endTime"])
KINDS = ("peer", "ingress")
transfers = {k: [] for k in KINDS}
with open(trace) as fh:
    for line in fh:
        if '"xfer"' not in line:
            continue
        r = json.loads(line)
        if r.get("k") != "xfer" or r.get("kind") not in KINDS:
            continue
        cur, path = r["src"], []
        for key, bw in r["route"]:
            a, b = key.split("|")
            if cur not in (a, b):
                raise SystemExit(f"FAIL LOUD: route of {r['task']} not a path from {r['src']} at {key}")
            nxt = b if cur == a else a
            path.append((f"{cur}>{nxt}", bw)); cur = nxt
        if cur != r["dst"]:
            raise SystemExit(f"FAIL LOUD: route of {r['task']} ends at {cur}, not {r['dst']}")
        transfers[r["kind"]].append((r["bytes"], path))
want = sum(v["transfers"] for v in new["peerExchangeByAccessClass"].values())
if len(transfers["peer"]) != want:
    raise SystemExit(f"FAIL LOUD: {len(transfers['peer'])} peer transfers in the trace, {want} in the summary")


def role(node):
    if node not in classes:
        return None
    return "client" if node.startswith("client_") else "server"


def link_type(d):
    a, b = d.split(">")
    if a in classes:
        return f"{role(a)} {classes[a]} out"
    if b in classes:
        return f"{role(b)} {classes[b]} in"
    return "core"


res = {"endTime": end, "servers": new["servers"], "nodes": {n: [role(n), c] for n, c in classes.items()}, "kinds": {}}
for kind, trs in transfers.items():
    usage, nbytes, cnt, bwo = {}, {}, {}, {}
    for b, path in trs:
        for d, bw in path:
            usage[d] = usage.get(d, 0.0) + b / (bw * 1024 ** 2)
            nbytes[d] = nbytes.get(d, 0.0) + b
            cnt[d] = cnt.get(d, 0) + 1
            bwo[d] = bw
    over = {d for d, u in usage.items() if u / end > 1.0}
    tot_b = sum(b for b, _ in trs)
    cross_b = cross_n = 0
    by_type = {}
    for b, path in trs:
        tags = {link_type(d) for d, _ in path if d in over}
        if tags:
            cross_b += b; cross_n += 1
            for t in tags:
                s = by_type.setdefault(t, [0.0, 0]); s[0] += b; s[1] += 1
    res["kinds"][kind] = {"transfers": len(trs), "bytes": tot_b, "cross_bytes": cross_b, "cross_transfers": cross_n,
                          "cross_by_type": {t: {"bytes": v[0], "transfers": v[1]} for t, v in by_type.items()},
                          "links": {d: {"rho": usage[d] / end, "usage_s": usage[d], "bytes": nbytes[d], "bw": bwo[d], "transfers": cnt[d],
                                        "type": link_type(d)} for d in usage}}
json.dump(res, open(out_p, "w"))
print("[rho2] " + "; ".join(f"{k}: {len(v['links'])} links, max rho {max((x['rho'] for x in v['links'].values()), default=0):.3f}, "
                            f"{100 * v['cross_bytes'] / max(v['bytes'], 1):.1f} % of bytes cross a rho>1 link" for k, v in res["kinds"].items()))
