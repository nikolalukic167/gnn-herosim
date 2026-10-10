"""Scale probe (2026-10-09, descriptive): spread a single-origin WF1 window over N clients. The production windows use 20 origin
clients (client_local_v1_single_origin.clients_used); group g keeps its origin's rank r among those and moves to
client_node(r + 20 * (g mod N/20)). Timestamps, groups, payloads and types are untouched, so the arrival process is the
same and each client sends 20/N as often. Disclosed as a probe-only transform.
usage: scale_probe_remap_clients.py SRC_WL DST_WL --clients N"""
import argparse, json
from collections import Counter
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("dst"); ap.add_argument("--clients", type=int, required=True)
a = ap.parse_args()
wl = json.loads(Path(a.src).read_text())
used = sorted({e["node_name"] for e in wl["events"]}, key=lambda s: int(s.replace("client_node", "")))
base = len(used)
if a.clients % base:
    raise SystemExit(f"FAIL LOUD: {a.clients} clients is not a multiple of the {base} origin clients")
fold = a.clients // base
rank = {n: i for i, n in enumerate(used)}
for e in wl["events"]:
    e["node_name"] = f"client_node{rank[e['node_name']] + base * (int(e['peer_group']) % fold)}"
c = Counter(e["node_name"] for e in wl["events"])
wl["scale_probe_remap"] = {"origin_clients_before": base, "clients": a.clients, "fold": fold, "clients_used": len(c),
                           "max_tasks_per_client": max(c.values()), "min_tasks_per_client": min(c.values())}
Path(a.dst).write_text(json.dumps(wl))
print(json.dumps(wl["scale_probe_remap"]))
