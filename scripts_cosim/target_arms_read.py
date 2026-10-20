#!/usr/bin/env python3
"""Autoscaler-target arm study read (uncut_groups_s0_v1; registration scientist/target-arms-reg 03f29334, cell list target_arms_cells.txt
md5 55ad6434f9d5bb58fb9fbc5e557c327e). Extends S4's dose read (dose_read.py) to FL gnn vs the bar at each KPA target. Descriptive, no label.

Cells (24 servers, topologies 19537-19548, windows g0-g3, MODERATE only, frozen _k100, pin 7cb483c5, TCP switch unset), KPA target T in {0.7, 0.45, 0.294}:
  bar  cd_ib0_inflightfull_k100 s0:  anomaly/dose_7cb483c5/cells_ref_mod{070,045,0294}  (filed dose cells)
  FL gnn ra_gnn_eng_combfl_nosplit_k100 s1-2:  cells_gnn_mod070 (filed) and cells_gnn_mod045, cells_gnn_mod0294 (new, the 192 cells of the list).
PRIMARY ("primary (Table I)"): the frozen reader's own uncut_arm_gate_read.contrast, imported and blob-checked, called per (arm, comparator): per cell 100 (arm -
comparator) / comparator on the seed mean, median over the four windows per topology, median over 12 topologies, exact two-sided Wilcoxon over topologies; each
per-topology value is recomputed unrounded and the median must equal contrast()'s. SECONDARY ("window-mean"): per topology the means over windows and seeds, then
the %; printed beside the primary, never Holm-ed.
F-target (Holm 4): FL gnn vs the bar at the same target, targets 0.45 and 0.294 x {whole, after 360 s}. Descriptive, raw p, no Holm: the same contrast at 0.7; the
per-topology change of that contrast 0.7 -> 0.45 -> 0.294 (median, topologies toward the bar, p); FL gnn at each target vs the bar at 0.294; FL gnn at 0.7 vs the
bar at 0.294. Beside every latency row: mean replicas and replica-seconds per task of both arms (replica_count_series.time_mean x endTime / num_tasks; the series'
last time must equal endTime). Per arm x target: replicas, replica-seconds per task, share of series samples at the 120 cap, exchange per task, transfers per
task, seconds per transfer, co-location share, lock wait, cold start. Health: collapse (a seed cell over 3 x the bar at its target, either statistic), tail flags,
per arm x target. Latency keys only; never total_rtt_plus_inference.
Per cell, asserted (FAIL LOUD): the exact grid (the 192 new cells equal the list's entries; every filed cell present; none failed); code.commit 7cb483c5 clean;
scaleOut mode kpa and target equal to the arm's registered value; no peerExchangeTcp, knative_panic, HEROSIM_EXCHANGE_TCP / SNAPSHOT_FIDELITY / PG_ORACLE / LIVE_AUDIT_*;
servers 24 and the topology; for each FL gnn cell the checkpoint basename and file md5 against comb/md5_comball.json. Output written once.
Training distribution: FL gnn was trained at target 0.7 (registration); a train/serve shift and a limit of the method are not separable here. SCIENTIST's prediction items
1-5 are evaluated mechanically at the end, not labelled. Heavy load and the FL MLP at lower targets are out of scope."""
import argparse
import hashlib
import json
import os
import statistics as st
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FROZEN_BLOBS = {"uncut_arm_gate_read.py": "04e368af8fbd9eb462ae68945676cc850ea808a9",
                "r1_attribution_v1_read.py": "f72bbb8fbbc423b70b073b7e6f3cc431b68bc5fc"}
CELLS_MD5 = "55ad6434f9d5bb58fb9fbc5e557c327e"


def git_blob(path):
    b = open(path, "rb").read()
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


for _f, _want in FROZEN_BLOBS.items():
    _got = git_blob(os.path.join(HERE, _f))
    if _got != _want:
        raise SystemExit(f"FAIL LOUD: frozen {_f} blob {_got}, want {_want}")
import uncut_arm_gate_read as G  # noqa: E402
from r1_attribution_v1_read import exact_wilcoxon  # noqa: E402

PIN = "7cb483c5"
SIZE, TOPOS, RUNG = 24, list(range(19537, 19549)), "moderate"
WINDOWS, STATS = G.WINDOWS, G.STATS
TARGETS = (0.7, 0.45, 0.294)
TAG = {0.7: "070", 0.45: "045", 0.294: "0294"}
ARM_NAME = {"bar": "cd_ib0_inflightfull_k100", "gnn": "ra_gnn_eng_combfl_nosplit_k100"}
SEEDS = {"bar": (0,), "gnn": (1, 2)}
CKPT_FILE = "uncut-comb-fl-gnn-eng-seed{}.pt"
CKPT_KEY = "ra_gnn_eng_combfl_nosplit"
BAD_ENV = ("HEROSIM_EXCHANGE_TCP", "HEROSIM_SNAPSHOT_FIDELITY", "HEROSIM_PG_ORACLE", "HEROSIM_KPA_KNATIVE_PANIC")
CAP = 120
NEW = {("gnn", 0.45), ("gnn", 0.294)}
PER_ARM_KEYS = ("elapsed", "elapsed_after360", "replica_time_mean", "replica_s_per_task", "at_cap_share", "X_exchange", "transfers_per_task",
                "s_per_transfer", "coloc_share", "L_lock_wait", "K_cold")


def name(arm, target):
    return f"{arm}@{target}"


def make_cfg(U):
    D = f"{U}/anomaly/dose_7cb483c5"
    roots = {}
    for t in TARGETS:
        roots[("bar", t)] = f"{D}/cells_ref_mod{TAG[t]}"
        roots[("gnn", t)] = f"{D}/cells_gnn_mod{TAG[t]}"
    return {"U": U, "D": D, "roots": roots, "topos": TOPOS, "wl": f"{U}/comb/in_comball", "ckpt": f"{U}/comb/md5_comball.json"}


def grid(cfg):
    out = {}
    for (arm, t), d in cfg["roots"].items():
        for tp in cfg["topos"]:
            for w in WINDOWS:
                for sd in SEEDS[arm]:
                    out[(arm, t, sd, tp, w)] = f"{d}/cc40s{tp}__{w}{RUNG}__{ARM_NAME[arm]}_s{sd}"
    return out


def check_list(cfg, cells_file, expect):
    got = hashlib.md5(open(cells_file, "rb").read()).hexdigest()
    if expect and got != expect:
        raise SystemExit(f"FAIL LOUD: cell list md5 {got}, want {expect}")
    listed = set()
    for line in open(cells_file).read().splitlines():
        root, nm, qflag, tflag = line.split("\t")
        listed.add((f"{cfg['U']}/anomaly/{root}", nm, qflag, tflag))
    want = set()
    for (arm, t, sd, tp, w), stem in grid(cfg).items():
        if (arm, t) in NEW:
            want.add((os.path.dirname(stem), os.path.basename(stem), f"R1A_QUEUE_LENGTH={t}", "TCP=0"))
    if listed != want:
        raise SystemExit(f"FAIL LOUD: cell list and grid differ: {len(listed - want)} listed only {sorted(listed - want)[:3]}, "
                         f"{len(want - listed)} grid only {sorted(want - listed)[:3]}")
    return len(listed)


class Ckpt:
    def __init__(self, cfg, md5s=None):
        self.m = md5s if md5s is not None else json.load(open(cfg["ckpt"]))
        self.seen = {}

    def check(self, sd, s, where):
        path = s["env"].get("GNN_MODEL_PATH")
        if path is None or os.path.basename(path) != CKPT_FILE.format(sd):
            raise SystemExit(f"FAIL LOUD: model path {where}: {path}")
        if path not in self.seen:
            h = hashlib.md5()
            with open(path, "rb") as fh:
                for b in iter(lambda: fh.read(1 << 20), b""):
                    h.update(b)
            self.seen[path] = h.hexdigest()
        if self.seen[path] != self.m[f"{CKPT_KEY}|s{sd}"]:
            raise SystemExit(f"FAIL LOUD: checkpoint md5 {where}")


def one(cfg, key, stem, ckpt, rel):
    arm, t, sd, tp, w = key
    if os.path.exists(stem + ".failed.json"):
        raise SystemExit(f"FAIL LOUD: failed cell {stem}")
    if not os.path.exists(stem + ".summary.json"):
        raise SystemExit(f"FAIL LOUD: missing cell {stem}")
    s = json.load(open(stem + ".summary.json"))
    s["_path"] = stem + ".summary.json"
    n = s["num_tasks"]
    c = s["code"]
    if not c["commit"].startswith(PIN) or c["dirty"]:
        raise SystemExit(f"FAIL LOUD: pin {stem}: {c['commit'][:8]} dirty={c['dirty']}")
    if s["servers"] != SIZE or s["topology"] != tp:
        raise SystemExit(f"FAIL LOUD: servers/topology {stem}")
    so = s["scaleOut"]
    if so.get("mode") != "kpa" or abs(so["target"] - t) > 1e-12 or "knative_panic" in so:
        raise SystemExit(f"FAIL LOUD: scaleOut {stem}: {so.get('mode')} {so.get('target')} want {t}")
    if "peerExchangeTcp" in s or [k for k in s["env"] if k in BAD_ENV or k.startswith("LIVE_AUDIT_")]:
        raise SystemExit(f"FAIL LOUD: TCP block or excluded env {stem}")
    if arm == "gnn":
        ckpt.check(sd, s, stem)
    cls = s["peerExchangeByAccessClass"]
    tr = sum(v["transfers"] for v in cls.values())
    sec = sum(v["seconds"] for v in cls.values())
    rc = s["replica_count_series"]
    if abs(float(rc["t"][-1]) - float(s["endTime"])) > 1e-6 * max(1.0, float(s["endTime"])):
        raise SystemExit(f"FAIL LOUD: replica series ends at {rc['t'][-1]}, endTime {s['endTime']} {stem}")
    tot = np.asarray(rc["total"][1:])
    row = {"elapsed": float(s["averageElapsedTime"]), "elapsed_after360": float(s["steady_state"]["after_360s"]["mean"]),
           "replica_time_mean": float(rc["time_mean"]), "replica_s_per_task": float(rc["time_mean"]) * float(s["endTime"]) / n,
           "at_cap_share": float((tot >= CAP).mean()), "X_exchange": s["totalPeerExchangeTime"] / n, "transfers_per_task": tr / n,
           "s_per_transfer": sec / tr if tr else 0.0, "coloc_share": 1 - tr / rel[w], "L_lock_wait": s["lock_wait"]["mean"],
           "K_cold": s["averageColdStartTime"], "tail": float(s["endTime"]) - float(s["arrival_end"]["last_arrival_s"]) > G.TAIL_S}
    return s, row


def load_all(cfg, ckpt):
    rel = {w: 2 * len(json.load(open(f"{cfg['wl']}/wf1_{RUNG}/wl/grounded_{w}_n50000.json"))["peer_exchange"]) for w in WINDOWS}
    cells, rows = {}, {}
    g = grid(cfg)
    for key, stem in g.items():
        arm, t, sd, tp, w = key
        s, row = one(cfg, key, stem, ckpt, rel)
        cells[(name(arm, t), sd, tp, w, RUNG)] = s
        rows[(name(arm, t), sd, tp, w)] = row
    return cells, rows, len(g)


def arm_seeds(a):
    return SEEDS[a.split("@")[0]]


def topo_vals(cells, a, b, stat):
    out = {}
    for tp in TOPOS:
        v = []
        for w in WINDOWS:
            ma, mb = G.cell_mean(cells, a, arm_seeds(a), tp, w, RUNG), G.cell_mean(cells, b, arm_seeds(b), tp, w, RUNG)
            if ma is None or mb is None:
                raise SystemExit(f"FAIL LOUD: missing pair {a}/{b} {tp}{w}")
            v.append(100.0 * (ma[stat] - mb[stat]) / mb[stat])
        out[tp] = st.median(v)
    return out


def window_mean(rows, a, b, key):
    out = {}
    for tp in TOPOS:
        va = [rows[(a, sd, tp, w)][key] for w in WINDOWS for sd in arm_seeds(a)]
        vb = [rows[(b, sd, tp, w)][key] for w in WINDOWS for sd in arm_seeds(b)]
        out[tp] = 100.0 * (sum(va) / len(va) - sum(vb) / len(vb)) / (sum(vb) / len(vb))
    return out


def primary(cells, rows, a, b):
    es = []
    for stat in STATS:
        e = G.contrast(cells, a, arm_seeds(a), b, arm_seeds(b), RUNG, TOPOS, stat)
        tv = topo_vals(cells, a, b, stat)
        if abs(st.median(tv.values()) - e["median_pct"]) > 1e-9:
            raise SystemExit("FAIL LOUD: own primary differs from the frozen contrast")
        wm = window_mean(rows, a, b, "elapsed" if stat == "whole" else "elapsed_after360")
        e.update({"tv": tv, "a": a, "b": b, "wm_median": st.median(wm.values()), "wm_faster": sum(1 for x in wm.values() if x < 0)})
        es.append(e)
    return es


def arm_mean(rows, a, key):
    pt = [sum(rows[(a, sd, tp, w)][key] for w in WINDOWS for sd in arm_seeds(a)) / (len(WINDOWS) * len(arm_seeds(a))) for tp in TOPOS]
    return sum(pt) / len(pt), pt


def build(cells, rows):
    res = {"ft": [], "desc": {}, "change": {}}
    for t in (0.45, 0.294):
        res["ft"] += primary(cells, rows, name("gnn", t), name("bar", t))
    G.family(res["ft"])
    for t in TARGETS:
        res["desc"][("gnn", t, "bar", t)] = primary(cells, rows, name("gnn", t), name("bar", t))
        res["desc"][("gnn", t, "bar", 0.294)] = primary(cells, rows, name("gnn", t), name("bar", 0.294))
    for stat_i, stat in enumerate(STATS):
        for t0, t1 in ((0.7, 0.45), (0.45, 0.294), (0.7, 0.294)):
            a = res["desc"][("gnn", t0, "bar", t0)][stat_i]["tv"]
            b = res["desc"][("gnn", t1, "bar", t1)][stat_i]["tv"]
            d = [b[tp] - a[tp] for tp in TOPOS]
            res["change"][(stat, t0, t1)] = (st.median(d), sum(1 for x in d if x > 0), exact_wilcoxon(d))
    return res


def health(cells, rows):
    out = {}
    for arm in ("bar", "gnn"):
        for t in TARGETS:
            coll = {sd: 0 for sd in SEEDS[arm]}
            tail, n = 0, 0
            for tp in TOPOS:
                for w in WINDOWS:
                    ref = G.cell_mean(cells, name("bar", t), (0,), tp, w, RUNG)
                    for sd in SEEDS[arm]:
                        m = G.means(cells[(name(arm, t), sd, tp, w, RUNG)])
                        coll[sd] += any(m[k] > G.COLLAPSE_X * ref[k] for k in STATS)
                        tail += rows[(name(arm, t), sd, tp, w)]["tail"]
                        n += 1
            out[(arm, t)] = (max(coll.values()), tail, n)
    return out


def fmt(e, rows):
    a, b = e["a"], e["b"]
    ra, _ = arm_mean(rows, a, "replica_time_mean")
    rb, _ = arm_mean(rows, b, "replica_time_mean")
    sa, _ = arm_mean(rows, a, "replica_s_per_task")
    sb, _ = arm_mean(rows, b, "replica_s_per_task")
    return (f"    {a:11s} vs {b:11s} {e['stat']:8s} median {e['median_pct']:+8.3f} % ({e['faster']:2d}/{e['n_topologies']})  p {e['p']:.4g}"
            + (f"  holm {e['p_holm']:.4g}" if "p_holm" in e else "") + f"  | window-mean {e['wm_median']:+8.3f} % ({e['wm_faster']}/{e['n_topologies']})"
            + f"  | replicas {ra:6.2f} vs {rb:6.2f}, replica-s/task {sa:7.2f} vs {sb:7.2f} ({100 * (sa - sb) / sb:+6.1f} %)")


def report(res, rows, hl, nlist, ncells):
    L = [__doc__.strip(), "", f"=== cells read and asserted: {ncells}; new cells in the list {nlist}"]
    L.append("=== F-target (Holm 4): FL gnn vs the bar at the same target, moderate; primary (Table I) | window-mean | replicas")
    L += [fmt(e, rows) for e in res["ft"]]
    L.append("=== Descriptive (raw p, no Holm)")
    for (a, ta, b, tb), es in res["desc"].items():
        L.append(f"  -- FL gnn @{ta} vs bar @{tb}" + ("  [same target]" if ta == tb else ""))
        L += [fmt(e, rows) for e in es]
    L.append("=== Change of the FL-gnn-vs-bar contrast with the target: per-topology (later - earlier) of the primary, median pp, topologies where it rose (toward the bar), exact Wilcoxon raw p")
    for (stat, t0, t1), (m, up, p) in res["change"].items():
        L.append(f"    {stat:8s} {t0} -> {t1}: {m:+8.3f} pp  {up:2d}/12  p {p:.4g}")
    L.append("=== Per arm x target, mean over 12 topologies (moderate)")
    for arm in ("bar", "gnn"):
        for t in TARGETS:
            L.append(f"  {arm:3s} @{t}")
            for k in PER_ARM_KEYS:
                L.append(f"      {k:20s} {arm_mean(rows, name(arm, t), k)[0]:11.4f}")
    L.append("=== Health per arm x target: collapse (a seed cell over 3 x the bar at its target, either statistic; worse seed), tail flags (end - last arrival > 120 s)")
    for (arm, t), (c, tail, n) in hl.items():
        L.append(f"    {arm:3s} @{t}: collapse {c}  tail {tail}/{n}")
    L.append("=== Prediction items (registration), evaluated mechanically; no label")
    d7, d45, d294 = (res["desc"][("gnn", t, "bar", t)] for t in TARGETS)
    med = lambda es: {e["stat"]: e["median_pct"] for e in es}
    m7, m45, m294 = med(d7), med(d45), med(d294)
    L.append(f"  1 gain shrinks as the target falls (whole median less negative at each step): {m45['whole'] > m7['whole'] and m294['whole'] > m45['whole']} "
             f"(0.7 {m7['whole']:+.2f}, 0.45 {m45['whole']:+.2f}, 0.294 {m294['whole']:+.2f}); after 360 s: {m45['after360'] > m7['after360'] and m294['after360'] > m45['after360']} "
             f"({m7['after360']:+.2f}, {m45['after360']:+.2f}, {m294['after360']:+.2f})")
    L.append(f"  2 @0.45: both medians < 0 and >= 9/12: {all(e['median_pct'] < 0 and e['faster'] >= 9 for e in d45)}; whole in [-5, -1] %: {-5 <= m45['whole'] <= -1}; "
             f"less negative than at 0.7: {m45['whole'] > m7['whole']}")
    L.append(f"  3 @0.294: |median| <= 3 % on both: {all(abs(e['median_pct']) <= 3 for e in d294)}; whole {m294['whole']:+.2f} % ({d294[0]['faster']}/12 ahead)")
    s = res["desc"][("gnn", 0.7, "bar", 0.294)][0]
    L.append(f"  4 FL gnn @0.7 slower than the bar @0.294 on 12/12, whole: {s['faster'] == 0} ({12 - s['faster']}/12 slower, median {s['median_pct']:+.2f} %)")
    rr = {t: arm_mean(rows, name("gnn", t), "replica_time_mean")[0] / arm_mean(rows, name("bar", t), "replica_time_mean")[0] - 1 for t in TARGETS}
    L.append(f"  5 FL gnn mean replicas within +-10 % of the bar's at each target: {all(abs(v) <= 0.10 for v in rr.values())} "
             f"({', '.join(f'{t}: {100 * v:+.1f} %' for t, v in rr.items())})")
    return "\n".join(L)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("cells")
    ap.add_argument("--out")
    a = ap.parse_args(argv[1:])
    U = os.environ.get("TARGET_ARMS_U", "/share/nikola.lukic/uncut_groups")
    out = a.out or f"{U}/anomaly/target_arms_read.txt"
    cfg = make_cfg(U)
    nlist = check_list(cfg, a.cells, CELLS_MD5)
    cells, rows, ncells = load_all(cfg, Ckpt(cfg))
    txt = report(build(cells, rows), rows, health(cells, rows), nlist, ncells)
    print(txt)
    if os.path.exists(out):
        raise SystemExit("FAIL LOUD: output exists, written once")
    open(out + ".partial", "w").write(txt + "\n")
    os.replace(out + ".partial", out)


if __name__ == "__main__":
    main(sys.argv)
