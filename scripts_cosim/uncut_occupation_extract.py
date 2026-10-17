"""Platform compute occupancy from kept raw results (docs/lineages/uncut_groups_s0_v1.md, Record 2026-10-17 "Load descriptors of the moderate rung").

Reads <RAW>/*.raw.json written by r1a_gate.sbatch with KEEP_RAW_DIR=<RAW>; for each cell takes stats.nodeResults[].platformResults[].idleProportion
(occupation = 100 - idleProportion = 100 * load_time / env.now, src/placement/infrastructure.py:910,925,1946 at d6ae87c2) and writes per-cell counts and
occupations to <OUT>. nodeId 0-159 are the 160 clients, 160-183 the 24 servers (the accessClasses order). Run on a compute node (each raw file is ~250 MB):
  python3 scripts_cosim/uncut_occupation_extract.py
"""
import json, glob, os, sys
RAW = "/share/nikola.lukic/uncut_arm1/ld_raw"; OUT = "/share/nikola.lukic/uncut_arm1/ld_occupation.json"
rows = []
first = True
for f in sorted(glob.glob(RAW + "/*.raw.json")):
    full = json.load(open(f)); d = full["stats"]; del full
    nr = d["nodeResults"]
    if first:
        first = False
        print("nodeResult keys:", sorted(nr[0].keys())[:30]); pr0 = nr[0]["platformResults"]
        print("platformResult keys:", sorted(pr0[0].keys()) if pr0 else None, "n nodes", len(nr))
        print("node name candidates:", {k: nr[0][k] for k in nr[0] if "ame" in k})
    namekey = next(k for k in ("nodeName", "node_name", "name", "nodeId") if k in nr[0])
    plats = []
    for n in nr:
        for p in n["platformResults"]:
            plats.append((int(n[namekey]), p["platformId"], p["platformType"]["shortName"], 100.0 - p["idleProportion"]))
    srv = [p for p in plats if p[0] >= 160]  # nodeId 0-159 are the clients, 160-183 the 24 servers (accessClasses order)
    occ = lambda L: (sum(p[3] for p in L) / len(L)) if L else None
    used = [p for p in srv if p[3] > 0]
    r = {"cell": os.path.basename(f).replace(".raw.json", ""), "end_time_s": d["endTime"], "total_rtt": d["total_rtt"],
         "averageOccupation_key": d["averageOccupation"], "averageOccupation_all_recomputed": occ(plats),
         "n_nodes": len(nr), "n_platforms_all": len(plats), "n_platforms_server": len(srv), "n_platforms_server_used": len(used),
         "n_server_nodes": len({p[0] for p in srv}), "n_platforms_client": len(plats) - len(srv),
         "occupation_server_mean_pct": occ(srv), "occupation_server_used_mean_pct": occ(used),
         "occupation_server_max_pct": max((p[3] for p in srv), default=None),
         "by_type_server": {t: {"n": sum(1 for p in srv if p[2] == t), "mean_pct": occ([p for p in srv if p[2] == t])} for t in sorted({p[2] for p in srv})}}
    rows.append(r); print(r["cell"], r["n_platforms_all"], r["n_platforms_server"], round(r["averageOccupation_key"], 4), round(r["occupation_server_mean_pct"], 4), flush=True)
json.dump(rows, open(OUT, "w"), indent=1)
print("wrote", OUT, len(rows))
