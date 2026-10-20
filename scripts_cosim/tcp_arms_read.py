#!/usr/bin/env python3
"""TCP-on arm study read (uncut_groups_s0_v1, HEROSIM_EXCHANGE_TCP; registration scientist/tcp-arms-reg 80ad5181, cell list
tcp_arms_cells.txt md5 3e94b9af3bcba01dbe87bfec79fce174). Extends the filed tcp_read (c4caac08) to every arm and size. Descriptive, no label.

Sizes: 24 (19537-19548), 64 (30129-30140), 128 (30241-30252); one size per run (--size), each read once, when its grid is complete (Holm families are per size). Arms: bar cd_ib0_inflightfull_k100 s0; FL gnn, FL MLP, twin (all combfl, _k100) s1-2.
Modes: "off" (switch unset; filed cells at their own pins, new twin cells at e8b3f53c) and "on" (R1A_EXCHANGE_TCP=1 at e8b3f53c). The mode is a parameter of
every function (MODES, ROOTS); a third exchange mode adds roots and nothing else. Nothing here is built for it yet.

PRIMARY ("primary (Table I)") is the frozen reader's own statistic: uncut_arm_gate_read.contrast, imported, blob-checked: per cell 100 (arm - cmp) / cmp with a
learned arm the mean of its two seeds, the median over the four windows per topology, the median over the 12 topologies, exact two-sided Wilcoxon over
topologies. Holm within one size and one mode: three families of four tests (2 loads x whole / after 360 s): F-bar (gnn vs bar), F-mlp (gnn vs MLP), F-twin
(gnn vs twin). Nine families on, nine off (off for reference); no correction across families, sizes or modes. The off -> on shift of every contrast is
descriptive: per topology on minus off of the primary statistic, its median, exact Wilcoxon raw p, topologies where on > off (toward the comparator).
SECONDARY ("window-mean") is the filed tcp_read statistic: per topology, latencies averaged over windows and seeds, then the %; median over topologies.
It is printed beside the primary, never Holm-ed. The filed 24-server bar and FL gnn switch-on cells are recomputed under the primary.

Per cell, asserted (FAIL LOUD): exact grid (every cell present, none failed; the on-cells of the tcp_arms roots equal the cell list's, identity cells are never
opened); _k100 only; code.commit is the root's pin and clean; servers = size; peerExchangeTcp present with slow_start_v1 / 1448 / 1514 / 10 / per_partner
(on) or absent (off); scaleOut.target 0.7 and no knative_panic; no HEROSIM_SNAPSHOT_FIDELITY / PG_ORACLE / KPA_KNATIVE_PANIC / LIVE_AUDIT_* in env; learned
cells' GNN_MODEL_PATH basename and file md5 against comb/md5_comball.json (gnn, MLP) or fl3/md5_twin.json (twin). Latency keys only; never
total_rtt_plus_inference. Output is written once. Prediction items 1-5 of the registration are evaluated mechanically at the end, not labelled."""
import argparse
import hashlib
import json
import os
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FROZEN_BLOBS = {"uncut_arm_gate_read.py": "04e368af8fbd9eb462ae68945676cc850ea808a9",
                "r1_attribution_v1_read.py": "f72bbb8fbbc423b70b073b7e6f3cc431b68bc5fc"}
CELLS_MD5 = "3e94b9af3bcba01dbe87bfec79fce174"


def git_blob(path):
    b = open(path, "rb").read()
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


for _f, _want in FROZEN_BLOBS.items():
    _got = git_blob(os.path.join(HERE, _f))
    if _got != _want:
        raise SystemExit(f"FAIL LOUD: frozen {_f} blob {_got}, want {_want}")
import uncut_arm_gate_read as G  # noqa: E402
from r1_attribution_v1_read import exact_wilcoxon  # noqa: E402

PIN_ON = "e8b3f53c8b2e1e3d8d5a586064fc4f15b39b2e50"
MODES = ("off", "on")
ARMS = ("bar", "gnn", "mlp", "twin")
ARM_NAME = {"bar": "cd_ib0_inflightfull_k100", "gnn": "ra_gnn_eng_combfl_nosplit_k100", "mlp": "ra_mlp_same_combfl_nosplit_k100",
            "twin": "ra_twin_eng_combfl_nosplit_k100"}
SEEDS = {"bar": (0,), "gnn": (1, 2), "mlp": (1, 2), "twin": (1, 2)}
CKPT_FILE = {"gnn": "uncut-comb-fl-gnn-eng-seed{}.pt", "mlp": "uncut-comb-fl-mlp-same-seed{}.pt", "twin": "uncut-comb-fl-twin-seed{}.pt"}
CKPT_KEY = {"gnn": "ra_gnn_eng_combfl_nosplit", "mlp": "ra_mlp_same_combfl_nosplit", "twin": "ra_twin_eng_combfl_nosplit"}
TCP_BLOCK = {"model": "slow_start_v1", "mss": 1448, "frame": 1514, "initcwnd_segments": 10, "per_partner": True}
BAD_ENV = ("HEROSIM_SNAPSHOT_FIDELITY", "HEROSIM_PG_ORACLE", "HEROSIM_KPA_KNATIVE_PANIC")
RUNGS, WINDOWS, STATS = G.RUNGS, G.WINDOWS, G.STATS
CONTRASTS = (("F-bar", "gnn", "bar"), ("F-mlp", "gnn", "mlp"), ("F-twin", "gnn", "twin"))
DESC_CONTRASTS = (("mlp-vs-bar", "mlp", "bar"), ("twin-vs-bar", "twin", "bar"))
PER_ARM_KEYS = ("elapsed", "elapsed_after360", "X_exchange", "transfers_per_task", "s_per_transfer", "coloc_share", "L_lock_wait",
                "replica_time_mean", "K_cold", "tcp_extra_s_per_task")
TOPOS = {24: list(range(19537, 19549)), 64: list(range(30129, 30141)), 128: list(range(30241, 30253))}


def make_cfg(U):
    A = f"{U}/anomaly"
    roots, pins = {}, {}

    def put(mode, size, arm, rung, d, pin):
        roots[(mode, size, arm, rung)] = d
        pins[d] = pin

    for r, short in (("moderate", "mod"), ("heavy", "heavy")):
        put("on", 24, "bar", r, f"{A}/cells_tcp_ref_{short}", PIN_ON)
        put("on", 24, "gnn", r, f"{A}/cells_tcp_gnn_{short}", PIN_ON)
        put("off", 24, "bar", r, f"{A}/dose_7cb483c5/cells_ref_{short}070", "7cb483c5")
        put("off", 24, "gnn", r, f"{A}/dose_7cb483c5/cells_gnn_{short}070", "7cb483c5")
        put("off", 24, "mlp", r, f"{U}/comb/cells_comb", "a8681537")
        put("on", 24, "mlp", r, f"{A}/tcp_arms/on24", PIN_ON)
        put("on", 24, "twin", r, f"{A}/tcp_arms/on24", PIN_ON)
        put("off", 24, "twin", r, f"{A}/tcp_arms/off24", PIN_ON)
        for arm in ARMS:
            put("on", 64, arm, r, f"{A}/tcp_arms/on64", PIN_ON)
            put("on", 128, arm, r, f"{A}/tcp_arms/on128", PIN_ON)
        put("off", 64, "bar", r, f"{U}/size64/cells_bar", "a459dd98")
        put("off", 64, "gnn", r, f"{U}/size64/cells_learned", "74ddf70e")
        put("off", 64, "mlp", r, f"{U}/size64/cells_desc", "74ddf70e")
        put("off", 64, "twin", r, f"{U}/size64/cells_twin", "f8a55ff0")
        put("off", 128, "bar", r, f"{U}/size128/cells_bar", "a459dd98")
        put("off", 128, "gnn", r, f"{U}/size128/cells_learned", "74ddf70e")
        put("off", 128, "mlp", r, f"{U}/size128/mlp_cells", "74ddf70e")
        put("off", 128, "twin", r, f"{A}/tcp_arms/off128", PIN_ON)
    return {"U": U, "A": A, "roots": roots, "pins": pins, "topos": TOPOS,
            "wl": {24: f"{U}/comb/in_comball", 64: f"{U}/size64/in_size64", 128: f"{U}/size128/in_size128"},
            "ckpt": {"comb": f"{U}/comb/md5_comball.json", "twin": f"{U}/fl3/md5_twin.json"}}


def stem_of(cfg, mode, size, arm, rung, t, w, sd):
    d = cfg["roots"][(mode, size, arm, rung)]
    return f"{d}/cc40s{t}__{w}{rung}__{ARM_NAME[arm]}_s{sd}"


def grid(cfg):
    """Every expected cell: (mode, size, arm, seed, topology, window, rung) -> stem."""
    out = {}
    for mode in MODES:
        for size, topos in cfg["topos"].items():
            for arm in ARMS:
                for r in RUNGS:
                    for t in topos:
                        for w in WINDOWS:
                            for sd in SEEDS[arm]:
                                out[(mode, size, arm, sd, t, w, r)] = stem_of(cfg, mode, size, arm, r, t, w, sd)
    return out


def list_size(root):
    """24, 64 or 128 from a list root such as tcp_arms/on64, tcp_arms/off128 or tcp_arms/ident24."""
    return int(re.search(r"(\d+)$", root).group(1))


def check_list(cfg, cells_file, expect):
    """The tcp_arms cells of the grid equal the registered list's entries of the sizes in cfg (identity entries excluded and never opened). The whole list
    must still match the registered md5."""
    got = hashlib.md5(open(cells_file, "rb").read()).hexdigest()
    if expect and got != expect:
        raise SystemExit(f"FAIL LOUD: cell list md5 {got}, want {expect}")
    listed, ident = set(), 0
    for line in open(cells_file).read().splitlines():
        root, name, flag, *rest = line.split("\t")
        if list_size(root) not in cfg["topos"]:
            continue
        if root.startswith("tcp_arms/ident"):
            ident += 1
            continue
        listed.add((f"{cfg['A']}/{root}", name, flag))
    want = set()
    for key, stem in grid(cfg).items():
        d, name = os.path.dirname(stem), os.path.basename(stem)
        if d.startswith(f"{cfg['A']}/tcp_arms/"):
            want.add((d, name, "TCP=1" if key[0] == "on" else "TCP=0"))
    if listed != want:
        raise SystemExit(f"FAIL LOUD: cell list and grid differ: {len(listed - want)} listed only {sorted(listed - want)[:3]}, "
                         f"{len(want - listed)} grid only {sorted(want - listed)[:3]}")
    return len(listed), ident


class Ckpt:
    def __init__(self, cfg, md5s=None):
        self.m = md5s if md5s is not None else {**json.load(open(cfg["ckpt"]["comb"])), **json.load(open(cfg["ckpt"]["twin"]))}
        self.seen = {}

    def check(self, arm, sd, s, where):
        path = s["env"].get("GNN_MODEL_PATH")
        if path is None or os.path.basename(path) != CKPT_FILE[arm].format(sd):
            raise SystemExit(f"FAIL LOUD: model path {where}: {path}")
        if path not in self.seen:
            h = hashlib.md5()
            with open(path, "rb") as fh:
                for b in iter(lambda: fh.read(1 << 20), b""):
                    h.update(b)
            self.seen[path] = h.hexdigest()
        if self.seen[path] != self.m[f"{CKPT_KEY[arm]}|s{sd}"]:
            raise SystemExit(f"FAIL LOUD: checkpoint md5 {where}")


def one(cfg, key, stem, ckpt, rel):
    mode, size, arm, sd, t, w, r = key
    if "tcp_arms/ident" in stem:
        raise SystemExit(f"FAIL LOUD: identity path in the grid: {stem}")
    if os.path.exists(stem + ".failed.json"):
        raise SystemExit(f"FAIL LOUD: failed cell {stem}")
    if not os.path.exists(stem + ".summary.json"):
        raise SystemExit(f"FAIL LOUD: missing cell {stem}")
    s = json.load(open(stem + ".summary.json"))
    s["_path"] = stem + ".summary.json"
    n = s["num_tasks"]
    pin = cfg["pins"][os.path.dirname(stem)]
    c = s["code"]
    if not c["commit"].startswith(pin) or c["dirty"]:
        raise SystemExit(f"FAIL LOUD: pin {stem}: {c['commit'][:8]} dirty={c['dirty']}, want {pin}")
    if s["servers"] != size or s["topology"] != t:
        raise SystemExit(f"FAIL LOUD: servers/topology {stem}")
    b = s.get("peerExchangeTcp")
    if mode == "on":
        if b is None or any(b[k] != v for k, v in TCP_BLOCK.items()) or not b["transfers"] > 0:
            raise SystemExit(f"FAIL LOUD: TCP block {stem}")
    elif b is not None:
        raise SystemExit(f"FAIL LOUD: TCP block in a switch-off cell {stem}")
    if "knative_panic" in s["scaleOut"] or s["scaleOut"]["target"] != 0.7:
        raise SystemExit(f"FAIL LOUD: scaleOut {stem}")
    bad = [k for k in s["env"] if k in BAD_ENV or k.startswith("LIVE_AUDIT_")]
    if bad:
        raise SystemExit(f"FAIL LOUD: env {stem}: {bad}")
    if arm != "bar":
        ckpt.check(arm, sd, s, stem)
    cls = s["peerExchangeByAccessClass"]
    tr = sum(v["transfers"] for v in cls.values())
    sec = sum(v["seconds"] for v in cls.values())
    row = {"elapsed": float(s["averageElapsedTime"]), "elapsed_after360": float(s["steady_state"]["after_360s"]["mean"]),
           "X_exchange": s["totalPeerExchangeTime"] / n, "transfers_per_task": tr / n, "s_per_transfer": sec / tr if tr else 0.0,
           "coloc_share": 1 - tr / rel[(w, r)], "L_lock_wait": s["lock_wait"]["mean"],
           "replica_time_mean": s["replica_count_series"]["time_mean"], "K_cold": s["averageColdStartTime"],
           "tcp_extra_s_per_task": (b["extra_seconds"] / n) if b else 0.0,
           "tail": float(s["endTime"]) - float(s["arrival_end"]["last_arrival_s"]) > G.TAIL_S}
    try:
        row["excess_pp"] = G.charge_miss(s, stem)["excess_pp"]
    except SystemExit:
        row["excess_pp"] = None
    return s, row


def load_all(cfg, ckpt):
    cells = {m: {sz: {} for sz in cfg["topos"]} for m in MODES}
    rows = {m: {sz: {} for sz in cfg["topos"]} for m in MODES}
    rel = {sz: {(w, r): 2 * len(json.load(open(f"{cfg['wl'][sz]}/wf1_{r}/wl/grounded_{w}_n50000.json"))["peer_exchange"])
                for w in WINDOWS for r in RUNGS} for sz in cfg["topos"]}
    g = grid(cfg)
    for key, stem in g.items():
        mode, size, arm, sd, t, w, r = key
        s, row = one(cfg, key, stem, ckpt, rel[size])
        cells[mode][size][(arm, sd, t, w, r)] = s
        rows[mode][size][(arm, sd, t, w, r)] = row
    return cells, rows, len(g)


def topo_vals(cells, a, a_seeds, b, b_seeds, rung, topos, stat):
    """contrast()'s per-topology values, unrounded (the frozen contrast rounds them to 3 decimals)."""
    out = {}
    for t in topos:
        v = []
        for w in WINDOWS:
            ma, mb = G.cell_mean(cells, a, a_seeds, t, w, rung), G.cell_mean(cells, b, b_seeds, t, w, rung)
            if ma is None or mb is None:
                raise SystemExit(f"FAIL LOUD: missing pair {a}/{b} {t}{w}{rung}")
            v.append(100.0 * (ma[stat] - mb[stat]) / mb[stat])
        out[t] = st.median(v)
    return out


def window_mean_vals(rows, a, b, rung, topos, key):
    """The filed tcp_read statistic: per topology, the mean over windows (and seeds) of each arm's latency, then the %."""
    out = {}
    for t in topos:
        va = [rows[(a, sd, t, w, rung)][key] for w in WINDOWS for sd in SEEDS[a]]
        vb = [rows[(b, sd, t, w, rung)][key] for w in WINDOWS for sd in SEEDS[b]]
        ma, mb = sum(va) / len(va), sum(vb) / len(vb)
        out[t] = 100.0 * (ma - mb) / mb
    return out


def primary(cells, a, b, topos):
    es = [G.contrast(cells, a, SEEDS[a], b, SEEDS[b], r, topos, s) for r in RUNGS for s in STATS]
    for e in es:
        tv = topo_vals(cells, a, SEEDS[a], b, SEEDS[b], e["rung"], topos, e["stat"])
        if abs(st.median(tv.values()) - e["median_pct"]) > 1e-9:
            raise SystemExit("FAIL LOUD: own primary differs from the frozen contrast")
        e["tv"] = tv
    return es


def build(cfg, cells, rows):
    res = {}
    for sz, topos in cfg["topos"].items():
        res[sz] = {"contrasts": {}, "desc": {}}
        for name, a, b in CONTRASTS + DESC_CONTRASTS:
            per = {}
            for mode in MODES:
                es = primary(cells[mode][sz], a, b, topos)
                if name.startswith("F-"):
                    G.family(es)
                for e in es:
                    wm = window_mean_vals(rows[mode][sz], a, b, e["rung"], topos, "elapsed" if e["stat"] == "whole" else "elapsed_after360")
                    e["wm_median"] = st.median(wm.values())
                    e["wm_faster"] = sum(1 for x in wm.values() if x < 0)
                    e["wm"] = wm
                per[mode] = es
            shifts = []
            for i in range(len(per["on"])):
                eo, en = per["off"][i], per["on"][i]
                d = [en["tv"][t] - eo["tv"][t] for t in topos]
                shifts.append({"rung": en["rung"], "stat": en["stat"], "shift_median": st.median(d), "p_raw": exact_wilcoxon(d),
                               "toward_cmp": sum(1 for x in d if x > 0), "wm_shift": en["wm_median"] - eo["wm_median"]})
            (res[sz]["contrasts"] if name.startswith("F-") else res[sz]["desc"])[name] = {"per": per, "shift": shifts}
    return res


def health(cfg, cells, rows):
    out = {}
    for mode in MODES:
        for sz, topos in cfg["topos"].items():
            for arm in ARMS:
                for r in RUNGS:
                    coll = {sd: 0 for sd in SEEDS[arm]}
                    tail, ex, n = 0, [], 0
                    for t in topos:
                        for w in WINDOWS:
                            ref = G.cell_mean(cells[mode][sz], "bar", (0,), t, w, r)
                            for sd in SEEDS[arm]:
                                m = G.means(cells[mode][sz][(arm, sd, t, w, r)])
                                coll[sd] += any(m[k] > G.COLLAPSE_X * ref[k] for k in STATS)
                                rw = rows[mode][sz][(arm, sd, t, w, r)]
                                tail += rw["tail"]
                                n += 1
                                if rw["excess_pp"] is not None:
                                    ex.append(rw["excess_pp"])
                    out[(mode, sz, arm, r)] = {"collapse": max(coll.values()), "tail": tail, "cells": n,
                                               "excess_max": max(ex) if ex else None, "excess_cells": len(ex)}
    return out


def arm_table(cfg, rows):
    out = {}
    for sz, topos in cfg["topos"].items():
        for arm in ARMS:
            for r in RUNGS:
                for k in PER_ARM_KEYS:
                    v = {}
                    for mode in MODES:
                        pt = [sum(rows[mode][sz][(arm, sd, t, w, r)][k] for w in WINDOWS for sd in SEEDS[arm]) / (len(WINDOWS) * len(SEEDS[arm]))
                              for t in topos]
                        v[mode] = pt
                    d = [b - a for a, b in zip(v["off"], v["on"])]
                    out[(sz, arm, r, k)] = (sum(v["off"]) / len(topos), sum(v["on"]) / len(topos), sum(1 for x in d if x > 0), exact_wilcoxon(d))
    return out


def fmt_es(es, sh):
    L = []
    for i, e in enumerate(es):
        L.append(f"      {e['rung']:8s} {e['stat']:8s} median {e['median_pct']:+8.3f} % ({e['faster']:2d}/{e['n_topologies']})  p {e['p']:.4g}"
                 + (f"  holm {e['p_holm']:.4g}" if 'p_holm' in e else "") + f"   | window-mean {e['wm_median']:+8.3f} % ({e['wm_faster']}/{e['n_topologies']})")
    return L


def report(cfg, res, hl, at, nlist, ncells):
    L = [__doc__.strip(), "", f"=== cells read and asserted: {ncells}; cell list entries {nlist[0]} (identity entries {nlist[1]} never opened)"]
    pred = {}
    for sz in cfg["topos"]:
        L.append(f"\n==================== {sz} servers ====================")
        for name, _, _ in CONTRASTS + DESC_CONTRASTS:
            d = (res[sz]["contrasts"] if name.startswith("F-") else res[sz]["desc"])[name]
            L.append(f"  -- {name}   (primary (Table I) | window-mean)")
            for mode in ("on", "off"):
                L.append(f"    switch {mode}{'  [reference]' if mode == 'off' else ''}" + ("  Holm within family of 4" if name.startswith("F-") else "  descriptive, raw p"))
                L += fmt_es(d["per"][mode], d["shift"])
            L.append("    off -> on shift of the primary (positive = toward the comparator): median, topologies on > off, exact Wilcoxon raw p | window-mean shift")
            for s_ in d["shift"]:
                L.append(f"      {s_['rung']:8s} {s_['stat']:8s} {s_['shift_median']:+8.3f} pp  {s_['toward_cmp']:2d}/{len(cfg['topos'][sz])}  p {s_['p_raw']:.4g}   | {s_['wm_shift']:+8.3f} pp")
        tw = res[sz]["contrasts"]["F-twin"]["per"]["on"]
        L.append(f"  MP rule at {sz} (switch on): " + ("MP-ATTRIBUTED" if all(e["median_pct"] <= -2.0 and e["p_holm"] < 0.05 for e in tw)
                                                      else "no message-passing claim") + " (<= -2 % with Holm p < .05 on all four F-twin tests)")
        L.append("  -- health: collapse (any seed > 3 x the same-mode bar, either statistic; worse seed), tail flags (end - last arrival > 120 s), decision-charge excess max (pp; cells with a served rate)")
        for mode in MODES:
            for arm in ARMS:
                for r in RUNGS:
                    h = hl[(mode, sz, arm, r)]
                    L.append(f"    {mode:3s} {arm:4s} {r:8s} collapse {h['collapse']}  tail {h['tail']}/{h['cells']}  excess max "
                             + (f"{h['excess_max']:+.3f} ({h['excess_cells']} cells)" if h["excess_max"] is not None else "n/a"))
        L.append("  -- per arm x load, mean over 12 topologies: off -> on, topologies on > off, exact Wilcoxon raw p")
        for arm in ARMS:
            for r in RUNGS:
                for k in PER_ARM_KEYS:
                    o, n, up, p = at[(sz, arm, r, k)]
                    L.append(f"    {arm:4s} {r:8s} {k:22s} {o:11.4f} -> {n:11.4f}  {n - o:+10.4f}  {up:2d}/12  p {p if p is None else format(p, '.4g')}")
    L.append("\n=== prediction items (registration), evaluated mechanically; no label")
    for sz in cfg["topos"]:
        fb = res[sz]["contrasts"]["F-bar"]
        mod = [(e, s_) for e, s_ in zip(fb["per"]["on"], fb["shift"]) if e["rung"] == "moderate"]
        L.append(f"  1 @{sz}: F-bar moderate on: median < 0 and >= 10/12 on both statistics: "
                 f"{all(e['median_pct'] < 0 and e['faster'] >= 10 for e, _ in mod)}; off -> on shift toward the bar on both: {all(s_['shift_median'] > 0 for _, s_ in mod)} "
                 f"({', '.join(s_['stat'] + format(s_['shift_median'], '+.2f') + ' pp' for _, s_ in mod)})")
        hw = [(e, s_) for e, s_ in zip(fb["per"]["on"], fb["shift"]) if e["rung"] == "heavy" and e["stat"] == "whole"][0]
        L.append(f"  2 @{sz}: F-bar heavy whole on {hw[0]['median_pct']:+.2f} % ({hw[0]['faster']}/12), shift {hw[1]['shift_median']:+.2f} pp "
                 f"(64: shift toward the bar {hw[1]['shift_median'] > 0}; 128: median >= 0 {hw[0]['median_pct'] >= 0})")
        fm = res[sz]["contrasts"]["F-mlp"]["shift"]
        L.append(f"  3 @{sz}: F-mlp off -> on shift within +-1.5 pp on every test: {all(abs(s_['shift_median']) <= 1.5 for s_ in fm)} "
                 f"({', '.join(s_['rung'][:3] + ' ' + s_['stat'] + format(s_['shift_median'], '+.2f') for s_ in fm)})")
        tw = res[sz]["contrasts"]["F-twin"]["per"]["on"]
        L.append(f"  4 @{sz}: MP-ATTRIBUTED: {all(e['median_pct'] <= -2.0 and e['p_holm'] < 0.05 for e in tw)}")
        up = {k: all(at[(sz, a, r, k)][1] > at[(sz, a, r, k)][0] for a in ARMS for r in RUNGS) for k in ("X_exchange", "tcp_extra_s_per_task", "replica_time_mean", "L_lock_wait")}
        L.append(f"  5 @{sz}: rises in every arm x load: exchange per task {up['X_exchange']}, TCP extra {up['tcp_extra_s_per_task']}, replicas {up['replica_time_mean']}; "
                 f"lock wait rises in every arm x load: {up['L_lock_wait']}")
    return "\n".join(L)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("cells")
    ap.add_argument("--size", type=int, required=True, choices=sorted(TOPOS))
    ap.add_argument("--out")
    a = ap.parse_args(argv[1:])
    U = os.environ.get("TCP_ARMS_U", "/share/nikola.lukic/uncut_groups")
    out = a.out or f"{U}/anomaly/tcp_arms_read_{a.size}.txt"
    cfg = make_cfg(U)
    cfg["topos"] = {a.size: TOPOS[a.size]}
    nlist = check_list(cfg, a.cells, CELLS_MD5)
    cells, rows, ncells = load_all(cfg, Ckpt(cfg))
    res = build(cfg, cells, rows)
    txt = report(cfg, res, health(cfg, cells, rows), arm_table(cfg, rows), nlist, ncells)
    print(txt)
    if os.path.exists(out):
        raise SystemExit("FAIL LOUD: output exists, each size is read once")
    open(out + ".partial", "w").write(txt + "\n")
    os.replace(out + ".partial", out)


if __name__ == "__main__":
    main(sys.argv)
