"""Aggregation of the audit-trace re-runs of the sealed-12 confirmation cells (llf_cd_confirm_v1 mechanism diagnostic; see mechanism_diag.py). Reads anomaly/fairshare/mech/<arm>/<window>_<rung>_<topology>.json
(mech_extract.py; each re-run was checked equal to its filed cell). Arms: bar, gnn (seed 1), wide (seed 1), and gnn2 / gnn3 (the seed-2/3 gnn runaway cells only). Writes
  mechanism_rho.json       per arm, rung, window: median over topologies of per-link offered usage statistics (peer + ingress summed on a link), and gnn-vs-bar / gnn-vs-wide paired contrasts
  mechanism_runaways.json  per runaway gnn cell, next to the bar and wide cell of the same (topology, window, rung): onset and growth of tasks in the system, in the exchange stage and waiting for the compute lock,
                           the busiest link (type, rho, share of peer transfers) and whether its offered rate keeps rising
  mechanism_groups.json    per-group concentration of the served placement (top-node share, normalised Herfindahl) at 11+ and 51+ tasks"""
import json, os, statistics as st
import numpy as np

R = "/share/nikola.lukic/uncut_groups/anomaly/fairshare/mech"
TOPOS = [30285, 30286, 30287, 30289, 30290, 30291, 30292, 30293, 30294, 30295, 30296, 30297]; WINS = ("g0", "g1"); RUNGS = ("moderate", "heavy")

def load(arm, w, rung, t):
    p = f"{R}/{arm}/{w}_{rung}_{t}.json"
    return json.load(open(p)) if os.path.exists(p) else None

def stats(c):
    nodes = c["nodes"]; end = c["endTime"]; pk, ik = c["kinds"]["peer"]["links"], c["kinds"]["ingress"]["links"]
    tot = {}
    for d, v in pk.items(): tot[d] = [v["rho"], v["type"]]
    for d, v in ik.items():
        if d in tot: tot[d][0] += v["rho"]
        else: tot[d] = [v["rho"], v["type"]]
    n_cel = sum(1 for role, cls in nodes.values() if role == "server" and cls == "cellular"); n_srv = sum(1 for role, cls in nodes.values() if role == "server")
    cel_over = sum(1 for d, (r, t) in tot.items() if t == "server cellular out" and r > 1.0)
    srv_over = sum(1 for d, (r, t) in tot.items() if t.startswith("server") and r > 1.0)
    worst = max(tot.items(), key=lambda kv: kv[1][0]) if tot else (None, [0.0, ""])
    top = c["top_links"][0] if c["top_links"] else {"rho": 0.0, "share_of_peer_transfers": 0.0, "type": "", "link": ""}
    peer = c["kinds"]["peer"]
    return {"cellular_server_uplinks": n_cel, "cellular_uplinks_over_1_share": cel_over / n_cel if n_cel else None, "server_access_links_over_1_share": srv_over / (2 * n_srv) if n_srv else None,
            "max_rho": worst[1][0], "max_rho_type": worst[1][1], "max_rho_link": worst[0], "peer_bytes_crossing_rho_gt_1_share": peer["cross_bytes"] / peer["bytes"] if peer["bytes"] else 0.0,
            "busiest_link_rho": top["rho"], "busiest_link_share_of_peer_transfers": top["share_of_peer_transfers"], "busiest_link_type": top["type"], "busiest_link": top["link"],
            "n_links_over_1": sum(1 for r, t in tot.values() if r > 1.0), "end_time_s": end}

NUM = ["cellular_uplinks_over_1_share", "server_access_links_over_1_share", "max_rho", "peer_bytes_crossing_rho_gt_1_share", "busiest_link_rho", "busiest_link_share_of_peer_transfers", "n_links_over_1"]

def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None

def onset(series, base_n=5, factor=2.0):
    """first sample at or above factor x the baseline (the median of the first base_n samples, at least 1); None when it never gets there."""
    s = np.array(series, dtype=float)
    if len(s) < base_n + 3: return None, None
    base = max(float(np.median(s[:base_n])), 1.0)
    idx = np.nonzero(s >= factor * base)[0]
    return (int(idx[0]) if len(idx) else None), base

def shape(series, t, base_n=5, factor=2.0):
    s = np.array(series, dtype=float); i, base = onset(series, base_n, factor); pk = int(np.argmax(s))
    back = next((j for j in range(pk, len(s)) if s[j] <= factor * base), None) if base is not None else None
    return {"baseline": base, "first_ge_2x_time_s": (t[i] if i is not None else None), "peak_time_s": t[pk], "peak": int(s[pk]), "back_to_2x_time_s": (t[back] if back is not None else None), "last": int(s[-1])}

def growth(series):
    s = np.array(series, dtype=float); q = max(len(s) // 4, 1)
    return (float(s[-q:].mean()) / max(float(s[:q].mean()), 1.0))

def run(out):
    res = {"label": "sealed-12 confirmation cells re-run under audit trace; seed 1 for gnn and wide (seed 2/3 gnn runaway cells separately); windows g0, g1", "arms": {}, "contrasts": {}}
    cache = {}
    for arm in ("bar", "gnn", "wide"):
        for rung in RUNGS:
            for w in WINS:
                for t in TOPOS:
                    c = load(arm, w, rung, t)
                    if c: cache[(arm, rung, w, t)] = c
    missing = [(a, r, w, t) for a in ("bar", "gnn", "wide") for r in RUNGS for w in WINS for t in TOPOS if (a, r, w, t) not in cache]
    res["missing_cells"] = len(missing); res["missing_list"] = missing[:30]
    S = {k: stats(c) for k, c in cache.items()}
    for arm in ("bar", "gnn", "wide"):
        for rung in RUNGS:
            for wl, wins in (("g0+g1", WINS), ("g0", ("g0",)), ("g1", ("g1",))):
                ks = [(arm, rung, w, t) for w in wins for t in TOPOS if (arm, rung, w, t) in S]
                res["arms"][f"{arm}|{rung}|{wl}"] = {"cells": len(ks), **{m: med([S[k][m] for k in ks]) for m in NUM}}
    for B in ("bar", "wide"):
        for rung in RUNGS:
            for wl, wins in (("g0+g1", WINS), ("g0", ("g0",)), ("g1", ("g1",))):
                row = {}
                for m in NUM:
                    vals = []
                    for t in TOPOS:
                        ds = [S[("gnn", rung, w, t)][m] - S[(B, rung, w, t)][m] for w in wins if ("gnn", rung, w, t) in S and (B, rung, w, t) in S and S[("gnn", rung, w, t)][m] is not None and S[(B, rung, w, t)][m] is not None]
                        if ds: vals.append(st.median(ds))
                    if vals: row[m] = {"median_difference": st.median(vals), "gnn_lower_of_n": f"{sum(v < 0 for v in vals)}/{len(vals)}", "per_topology": vals}
                res["contrasts"][f"gnn_minus_{B}|{rung}|{wl}"] = row
    json.dump(res, open(f"{out}/mechanism_rho.json", "w"), indent=1)
    # runaway series
    cd = json.load(open(f"{out}/mechanism_contrasts.json")); rows = []
    for c in cd["runaway_cells"]:
        if c["arm"] != "gnn" or c["rung"] != "heavy": continue
        arm = "gnn" if c["seed"] == 1 else f"gnn{c['seed']}"
        cell = load(arm, c["window"], "heavy", c["topology"])
        if not cell: rows.append({"cell": c, "trace": "missing"}); continue
        ser = cell["series"]; tl = cell["timeline"]; top = cell["top_links"][0]; tb = tl["top_link_bytes_started"][top["link"]]
        cap = next((v["bw"] for v in cell["kinds"]["peer"]["links"].values() if False), None)
        bw = cell["kinds"]["peer"]["links"][top["link"]]["bw"]; offered = [b / (bw * 1024 ** 2 * tl["bin_s"]) for b in tb]
        entry = {"cell": c, "end_time_s": cell["endTime"], "series_t": ser["t"]}
        for name in ("in_system", "in_exchange", "waiting_for_compute"):
            entry[name] = {**shape(ser[name], ser["t"]), "growth_last_over_first_quarter": growth(ser[name])}
        wl_s = json.load(open(f"/share/nikola.lukic/uncut_groups/anomaly/fairshare/d_conf_final/{'r1' if c['topology'] == 30297 else 'main'}_{c['window']}heavy_static_s{c['seed']}/cc40s{c['topology']}__{c['window']}heavy__ra_gnn_eng_combfl_linkstatic_e40_nosplit_k100_s{c['seed']}.summary.json"))
        entry["last_arrival_s"] = (wl_s.get("arrival_end") or {}).get("last_arrival_s")
        entry["busiest_link"] = {"link": top["link"], "type": top["type"], "rho": top["rho"], "share_of_peer_transfers": top["share_of_peer_transfers"], "offered_rho_by_bin": offered,
                                 "first_bin_offered_gt_1": next((i for i, o in enumerate(offered) if o > 1.0), None), "bins_offered_gt_1": sum(1 for o in offered if o > 1.0), "bin_s": tl["bin_s"], "peer_started_by_bin": tl["peer_started"]}
        entry["links_over_1"] = sum(1 for kind in ("peer", "ingress") for v in cell["kinds"][kind]["links"].values() if v["rho"] > 1.0)
        for o in ("bar", "wide"):
            oc = load(o, c["window"], "heavy", c["topology"])
            if oc:
                oser = oc["series"]; entry[f"matched_{o}"] = {"end_time_s": oc["endTime"], "in_system_growth": growth(oser["in_system"]), "in_exchange_growth": growth(oser["in_exchange"]), "in_system_max": int(max(oser["in_system"])), "busiest_link_rho": oc["top_links"][0]["rho"]}
        rows.append(entry)
    json.dump(rows, open(f"{out}/mechanism_runaways.json", "w"), indent=1)
    groups = {}
    for arm in ("bar", "gnn", "wide"):
        for rung in RUNGS:
            for w in WINS:
                for key in ("size_ge_11", "size_ge_51"):
                    vs = [cache[(arm, rung, w, t)]["groups"][key] for t in TOPOS if (arm, rung, w, t) in cache and cache[(arm, rung, w, t)]["groups"][key]]
                    if vs: groups[f"{arm}|{rung}|{w}|{key}"] = {"cells": len(vs), **{m: st.median(v[m] for v in vs) for m in ("mean_top_node_share", "mean_norm_hhi", "share_with_half_on_one_node", "mean_size", "groups")}}
    json.dump(groups, open(f"{out}/mechanism_groups.json", "w"), indent=1)
    print("written rho / runaways / groups;", "missing cells:", len(missing), "runaway rows:", len(rows))
