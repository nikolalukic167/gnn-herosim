"""Open-loop replay of a HEROSIM_FLOW_LOG under the engine's equal-split rule and under max-min, with per-link load.
usage: flow_replay_check.py <flows.jsonl> <bw.json {directed link id: MB/s}> <summary.json> <out.json> [--no-maxmin]"""
import json, sys
import numpy as np

MB = 1024 ** 2


def load(path, bwmap):
    t, hops, nb = [], [], []
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            t.append(r[0]); hops.append(r[3]); nb.append(r[4])
    links = sorted({k for h in hops for k in h})
    idx = {k: i for i, k in enumerate(links)}
    missing = [k for k in links if k not in bwmap]
    if missing:
        raise SystemExit(f"FAIL LOUD: {len(missing)} links with no bandwidth, e.g. {missing[:3]}")
    H = max(len(h) for h in hops)
    hm = np.full((len(t), H), -1, dtype=np.int64)
    for i, h in enumerate(hops):
        hm[i, :len(h)] = [idx[k] for k in h]
    bw = np.array([bwmap[k] * MB for k in links])
    return np.array(t), hm, np.array(nb, dtype=float), links, bw


def rates_equal(act_h, counts, bw):
    share = np.where(act_h >= 0, bw[np.maximum(act_h, 0)] / np.maximum(counts[np.maximum(act_h, 0)], 1), np.inf)
    return share.min(axis=1)


def rates_maxmin(act_h, bw, L):
    n = act_h.shape[0]
    rate = np.zeros(n)
    frozen = np.zeros(n, bool)
    resid = bw.copy()
    valid = act_h >= 0
    while not frozen.all():
        live = ~frozen
        cnt = np.bincount(act_h[live][valid[live]], minlength=L).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            fair = np.where(cnt > 0, resid / cnt, np.inf)
        lvl = fair.min()
        sat = np.isclose(fair, lvl, rtol=1e-12, atol=0)
        hit = live & (valid & sat[np.maximum(act_h, 0)]).any(axis=1)
        rate[hit] = lvl
        frozen |= hit
        used = np.bincount(act_h[hit][valid[hit]], minlength=L).astype(float) * lvl
        resid = np.maximum(resid - used, 0.0)
    return rate


def replay(t0, hm, nb, bw, mode):
    L = len(bw)
    n = len(t0)
    rem = nb.copy()
    end = np.full(n, np.nan)
    active = np.zeros(0, dtype=np.int64)
    busy = np.zeros(L); delivered = np.zeros(L); peak = np.zeros(L, dtype=np.int64)
    now, nxt = 0.0, 0
    while nxt < n or active.size:
        if active.size:
            ah = hm[active]
            counts = np.bincount(ah[ah >= 0], minlength=L)
            peak = np.maximum(peak, counts)
            r = rates_equal(ah, counts, bw) if mode == "equal" else rates_maxmin(ah, bw, L)
            dt_done = (rem[active] / r).min()
        else:
            counts = np.zeros(L, dtype=np.int64); r = None; dt_done = np.inf
        dt_start = t0[nxt] - now if nxt < n else np.inf
        dt = max(0.0, min(dt_done, dt_start))
        if active.size and dt > 0:
            rem[active] -= r * dt
            on = counts > 0
            busy[on] += dt
            ah = hm[active]
            moved = np.repeat(r * dt, ah.shape[1]).reshape(ah.shape)
            np.add.at(delivered, ah[ah >= 0], moved[ah >= 0])
        now += dt
        if active.size:
            fin = (rem[active] <= 1e-9 * np.maximum(nb[active], 1.0)) | (now + rem[active] / r == now)
            end[active[fin]] = now
            active = active[~fin]
        while nxt < n and t0[nxt] <= now:
            active = np.append(active, nxt); nxt += 1
    return end - t0, busy, delivered, peak


def main():
    flows, bwf, summ, out = sys.argv[1:5]
    t0, hm, nb, links, bw = load(flows, json.load(open(bwf)))
    s = json.load(open(summ))
    fs = s["peerExchangeFairshare"]
    bottleneck = np.where(hm >= 0, bw[np.maximum(hm, 0)], np.inf).min(axis=1)
    alone = (nb / bottleneck).sum()
    res = {"flows": len(t0), "summary_transfers": fs["shared_transfers"], "links": len(links)}
    dur_eq, busy, deliv, peak = replay(t0, hm, nb, bw, "equal")
    # engine: extra = flow_seconds - sum(solo); solo = b + bytes/bottleneck + latency, so sum(byte phase) - sum(bytes/bottleneck) = extra
    res["equal_split"] = {"replay_extra_s": float(dur_eq.sum() - alone), "summary_extra_s": fs["extra_seconds"],
                          "rel_err": float(abs(dur_eq.sum() - alone - fs["extra_seconds"]) / fs["extra_seconds"]),
                          "peak_flows": int(peak.max()), "summary_peak": fs["peak_flows_on_a_link"], "byte_phase_s": float(dur_eq.sum())}
    span = t0.max() - t0.min()
    offered = np.zeros(len(bw)); np.add.at(offered, hm[hm >= 0], np.repeat(nb, hm.shape[1]).reshape(hm.shape)[hm >= 0])
    rho = offered / (bw * span)
    util = np.where(busy > 0, deliv / (bw * np.maximum(busy, 1e-12)), np.nan)
    order = np.argsort(-rho)
    res["links_top_rho"] = [{"link": links[i], "MBps": bw[i] / MB, "rho_offered": float(rho[i]), "util_while_busy": float(util[i]),
                             "busy_s": float(busy[i]), "peak": int(peak[i])} for i in order[:15]]
    res["links_top_peak"] = [{"link": links[i], "MBps": bw[i] / MB, "rho_offered": float(rho[i]), "util_while_busy": float(util[i]),
                              "peak": int(peak[i])} for i in np.argsort(-peak)[:10]]
    res["n_links_rho_gt1"] = int((rho > 1).sum())
    res["wide_links_util"] = {"n": int((bw / MB > 50).sum()),
                              "median_util_while_busy": float(np.nanmedian(util[bw / MB > 50])),
                              "max_rho": float(rho[bw / MB > 50].max())}
    if "--no-maxmin" not in sys.argv:
        dur_mm, busy_mm, _, peak_mm = replay(t0, hm, nb, bw, "maxmin")
        solo_sum = fs["flow_seconds"] - fs["extra_seconds"]
        res["maxmin"] = {"byte_phase_s": float(dur_mm.sum()), "extra_s": float(dur_mm.sum() - alone),
                         "slowdown_equal": float(fs["flow_seconds"] / solo_sum),
                         "slowdown_maxmin": float((solo_sum + dur_mm.sum() - alone) / solo_sum),
                         "flow_ratio_median": float(np.median(dur_eq / dur_mm)), "flow_ratio_p99": float(np.percentile(dur_eq / dur_mm, 99)),
                         "last_end_equal": float(np.nanmax(t0 + dur_eq)), "last_end_maxmin": float(np.nanmax(t0 + dur_mm)),
                         "peak_flows": int(peak_mm.max())}
        saved = float((dur_eq - dur_mm).sum())
        n_tasks = s["num_tasks"]
        res["per_task"] = {"num_tasks": n_tasks,
                           "exchange_s_per_task_equal": s["totalPeerExchangeTime"] / n_tasks,
                           "exchange_s_per_task_maxmin": (s["totalPeerExchangeTime"] - saved) / n_tasks,
                           "flow_seconds_equal": fs["flow_seconds"], "flow_seconds_maxmin": fs["flow_seconds"] - saved,
                           "mean_latency_equal": s["averageElapsedTime"],
                           "mean_latency_minus_saved_exchange": s["averageElapsedTime"] - saved / n_tasks}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1)[:4000])


if __name__ == "__main__":
    main()
