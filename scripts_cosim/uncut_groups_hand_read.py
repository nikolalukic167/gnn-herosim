#!/usr/bin/env python3
"""uncut_groups_s0_v1 hand-arm readers (docs/lineages/uncut_groups_s0_v1.md, K0 and K1). Written before any hand cell exists.

  k0  --gate <dir> --topos t1,...,t12 [--arms cd,cd_expand,cd_exactS,cd_exactS_ib0,...] --bar-arm A
      Counts first (summaries and failed against 12 topologies x g0-g3 x 2 rungs per arm). Then, per arm and rung, the COLLAPSE count (whole-run and
      after-360 s mean > 3x the same topology x window x rung's `cd` mean) and the TAIL FLAG count (end - last arrival > 120 s), separately. K0 health
      on g0: `cd` and `cd_exactS` healthy (no collapse on either statistic, effective_queue_share <= 0.80) on >= 10/12 at heavy, else the moderate
      rung only, else NO-SEAT. Uncut admissibility on the bar arm: tasks in groups > 10 served in a whole-group batch >= 90 % pooled over the four
      windows (per window and rung beside it), from the always-on schedulerCounters pgb_whole_<b> / pgb_split_<b> (tasks served in a batch holding
      their whole peer group / in a smaller batch, <b> = the group's ORIGINAL size: le10, 11_50, 51_100, gt100); the share is
      sum(pgb_whole over 11_50, 51_100, gt100) / (that + sum(pgb_split over the same)); a missing counter fails loud.
  k1  --gate <dir> --topos ... --suffix _k100 [--replay rows.json --sample ids.json --sample-sha256 H]
      Hand arm per rung: the best arm by the median over topologies of the per-topology median whole-run latency; if it collapses (any cell, either
      statistic), the best non-collapsing arm decides. (a) decision-time headroom from the summaries' own counters: (sched_queue_wait_seconds +
      sched_busy_seconds) / num_tasks / averageElapsedTime, and the same over the after-360 s mean, median over the arm's cells, < 3 % at both rungs.
      (b) the MEAN over the 200 sealed states of (RTT(hand) - RTT(ILS)) / RTT(ILS) < 1 %, with the share within 1 %, p90 and per-bucket means; the
      replay file's state ids must equal the sealed sample's and its hash must match. HAND-SUFFICES only if (a) and (b) hold.

  Cell files: <gate>/cc40s<topo>__g<i><rung>__<arm>_s<seed>.summary.json; arms at k > 0 carry the clock suffix (_k100, _k025).
"""
import argparse
import hashlib
import json
import os
import re
import statistics as st
import sys

RUNGS = ("heavy", "moderate")
WINDOWS = ("g0", "g1", "g2", "g3")
COLLAPSE_X, TAIL_S, QS_MAX = 3.0, 120.0, 0.80
K0_MIN_HEALTHY, WG_MIN, HEADROOM_MAX, REGRET_MAX = 10, 0.90, 0.03, 0.01
STAT = ("whole", "after360")
NAME = re.compile(r"^cc40s(\d+)__(g\d)(moderate|heavy)__(.+)_s(\d+)\.(summary|failed)\.json$")


def load_gate(gate):
    cells, failed = {}, {}
    for f in sorted(os.listdir(gate)):
        m = NAME.match(f)
        if not m:
            continue
        key = (m.group(4), int(m.group(1)), m.group(2), m.group(3))
        (cells if m.group(6) == "summary" else failed)[key] = os.path.join(gate, f)
    return {k: json.load(open(p)) for k, p in cells.items()}, failed


def means(s):
    ss = (s.get("steady_state") or {}).get("after_360s") or {}
    return {"whole": float(s["averageElapsedTime"]), "after360": ss.get("mean")}


def tail(s):
    return float(s["endTime"]) - float(s["arrival_end"]["last_arrival_s"])


def collapsed(s, ref, stat):
    a, b = means(s)[stat], means(ref)[stat] if ref else None
    return bool(a is not None and b and a > COLLAPSE_X * b)


def health_table(cells, arms, topos, suffix=""):
    out = {}
    for arm in arms:
        for rung in RUNGS:
            row = {"cells": 0, "collapse_whole": 0, "collapse_after360": 0, "collapse_any": 0, "tail_flag": 0, "collapsed_cells": []}
            for t in topos:
                for w in WINDOWS:
                    s, ref = cells.get((arm + suffix, t, w, rung)), cells.get(("cd" + suffix, t, w, rung))
                    if s is None:
                        continue
                    row["cells"] += 1
                    cw, ca = collapsed(s, ref, "whole"), collapsed(s, ref, "after360")
                    row["collapse_whole"] += cw
                    row["collapse_after360"] += ca
                    if cw or ca:
                        row["collapse_any"] += 1
                        row["collapsed_cells"].append((t, w))
                    row["tail_flag"] += tail(s) > TAIL_S
            out[f"{arm}|{rung}"] = row
    return out


def counts(cells, failed, arms, topos, suffix=""):
    exp = len(topos) * len(WINDOWS)
    return {f"{a}|{r}": {"finished": sum(1 for k in cells if k[0] == a + suffix and k[3] == r),
                         "failed": sum(1 for k in failed if k[0] == a + suffix and k[3] == r), "expected": exp}
            for a in arms for r in RUNGS}


def wall_infeasible(failed, arms, topos, suffix=""):
    """Third health count: watchdog-projected kills (projected wall > 1.5 x limit). Never pooled into collapse, never counted healthy;
    every other failure is listed apart as `other_failed`."""
    out = {}
    for arm in arms:
        for rung in RUNGS:
            row = {"wall_infeasible": 0, "wall_infeasible_g0": 0, "other_failed": 0, "cells": [], "other": []}
            for (a, t, w, r), path in sorted(failed.items()):
                if a != arm + suffix or r != rung or t not in topos:
                    continue
                d = json.load(open(path))
                if str(d.get("why", "")).startswith("watchdog: projected"):
                    row["wall_infeasible"] += 1
                    row["wall_infeasible_g0"] += w == "g0"
                    wd = d.get("watchdog") or {}
                    proj, last = wd.get("projected_s"), d.get("last_arrival_s")
                    row["cells"].append({"topo": t, "window": w, "detail": wd.get("detail"), "limit_s": d.get("limit_s"), "sim_now_at_kill": wd.get("sim_now"),
                                         "last_arrival_s": last, "projected_wall_s": proj,
                                         "wall_over_sim_length": proj / last if proj and last else None})
                else:
                    row["other_failed"] += 1
                    row["other"].append({"topo": t, "window": w, "why": str(d.get("why"))[:120]})
            out[f"{arm}|{rung}"] = row
    return out


def k0_health(cells, topos):
    """g0 health of cd and cd_exactS per rung: no collapse on either statistic and effective_queue_share <= 0.80."""
    out = {}
    for rung in RUNGS:
        res = {}
        for arm in ("cd", "cd_exactS"):
            ok = 0
            for t in topos:
                s, ref = cells.get((arm, t, "g0", rung)), cells.get(("cd", t, "g0", rung))
                if s is None or ref is None:
                    continue
                qs = s.get("effective_queue_share")
                ok += bool(not collapsed(s, ref, "whole") and not collapsed(s, ref, "after360") and qs is not None and qs <= QS_MAX)
            res[arm] = ok
        out[rung] = {"healthy_g0": res, "pass": all(v >= K0_MIN_HEALTHY for v in res.values())}
    verdict = "K0_OK_BOTH_RUNGS" if all(v["pass"] for v in out.values()) else \
        ("K0_MODERATE_ONLY" if out["moderate"]["pass"] else "NO-SEAT")
    return out, verdict


BIG_BUCKETS = ("11_50", "51_100", "gt100")


def _pgb(c, who, key):
    names = [f"pgb_{key}_{b}" for b in BIG_BUCKETS]
    for n in names:
        if n not in c:
            raise SystemExit(f"FAIL LOUD: counter {n!r} missing ({who})")
    return sum(int(c[n]) for n in names)


def whole_group_share(cells, bar, topos):
    out = {}
    for rung in RUNGS:
        per, tb, tw = {}, 0, 0
        for w in WINDOWS:
            b = wl = 0
            for t in topos:
                s = cells.get((bar, t, w, rung))
                if s is None:
                    continue
                c = s["schedulerCounters"]
                who = f"{bar} t{t} {w} {rung}"
                wh, sp = _pgb(c, who, "whole"), _pgb(c, who, "split")
                b += wh + sp
                wl += wh
            per[w] = wl / b if b else None
            tb += b
            tw += wl
        out[rung] = {"per_window": per, "pooled": tw / tb if tb else None, "pass": bool(tb and tw / tb >= WG_MIN)}
    return out


def split_counts(cells, arms, topos):
    return {f"{a}|{r}": sum(int(cells[(a, t, w, r)]["schedulerCounters"].get("peer_group_incomplete_batches") or 0)
                            for t in topos for w in WINDOWS if (a, t, w, r) in cells) for a in arms for r in RUNGS}


def pick_hand_arm(cells, arms, topos, suffix):
    """Per rung: arms ranked by the median over topologies of the per-topology median whole-run latency; a collapsing arm (any cell, either
    statistic) is passed over for the best non-collapsing one."""
    out = {}
    ht = health_table(cells, arms, topos, suffix)
    for rung in RUNGS:
        rank = []
        for arm in arms:
            per = []
            for t in topos:
                v = [means(cells[(arm + suffix, t, w, rung)])["whole"] for w in WINDOWS if (arm + suffix, t, w, rung) in cells]
                if v:
                    per.append(st.median(v))
            if per:
                rank.append((st.median(per), arm))
        rank.sort()
        chosen = next(((m, a) for m, a in rank if ht[f"{a}|{rung}"]["collapse_any"] == 0), None)
        out[rung] = {"ranking": rank, "median_best": rank[0][1] if rank else None, "median_best_collapses": bool(rank and ht[f"{rank[0][1]}|{rung}"]["collapse_any"]),
                     "decides": chosen[1] if chosen else None}
    return out, ht


def headroom(cells, arm, suffix, topos, rung):
    r = {"whole": [], "after360": []}
    for t in topos:
        for w in WINDOWS:
            s = cells.get((arm + suffix, t, w, rung))
            if s is None:
                continue
            c = s["schedulerCounters"]
            for k in ("sched_queue_wait_seconds", "sched_busy_seconds"):
                if k not in c:
                    raise SystemExit(f"FAIL LOUD: {k} missing in {arm}{suffix} t{t} {w} {rung}")
            per_task = (float(c["sched_queue_wait_seconds"]) + float(c["sched_busy_seconds"])) / int(s["num_tasks"])
            m = means(s)
            for stat in STAT:
                if m[stat]:
                    r[stat].append(per_task / m[stat])
    return {stat: {"median": st.median(v) if v else None, "n": len(v)} for stat, v in r.items()}


def replay_regret(rows, sample_ids, sample_sha, sample_path, arms_by_rung):
    if hashlib.sha256(open(sample_path, "rb").read()).hexdigest() != sample_sha:
        raise SystemExit("FAIL LOUD: the sealed sample file's hash differs from the recorded one")
    ids = {r["state_id"] for r in rows}
    if ids != set(sample_ids) or len(rows) != len(sample_ids):
        raise SystemExit("FAIL LOUD: replay rows do not match the sealed 200 states")
    for r in rows:
        if r["arm"] != arms_by_rung[r["rung"]]:
            raise SystemExit(f"FAIL LOUD: replay row for {r['arm']} at {r['rung']}, the deciding hand arm is {arms_by_rung[r['rung']]}")
    reg = [(r["hand_rtt"] - r["ils_rtt"]) / r["ils_rtt"] for r in rows]
    srt = sorted(reg)
    by = {}
    for r, x in zip(rows, reg):
        by.setdefault(r.get("bucket", "?"), []).append(x)
    return {"n": len(reg), "mean": sum(reg) / len(reg), "share_within_1pct": sum(1 for x in reg if x < 0.01) / len(reg),
            "p90": srt[int(0.9 * (len(srt) - 1))], "per_bucket_mean": {b: sum(v) / len(v) for b, v in by.items()},
            "pass": sum(reg) / len(reg) < REGRET_MAX}


ROB_MEDIAN, ROB_ALPHA = -2.0, 0.05


def win_rule(by_rung):
    """by_rung: {rung: {"median": %, "p_holm": p}} for one arm against one reference; WIN rule: median <= -2 % and Holm p < .05 at both rungs."""
    return all(by_rung[r]["median"] is not None and by_rung[r]["median"] <= ROB_MEDIAN and by_rung[r]["p_holm"] is not None and by_rung[r]["p_holm"] < ROB_ALPHA
               for r in RUNGS)


def bar_faster(by_rung):
    """E-FASTER: the bar Holm-faster at both rungs (the arm's % against it positive)."""
    return all(by_rung[r]["median"] is not None and by_rung[r]["median"] > 0 and by_rung[r]["p_holm"] is not None and by_rung[r]["p_holm"] < ROB_ALPHA
               for r in RUNGS)


def f1_label(vs_bar, vs_robust, arm_collapse, bar_collapse, anchor_ok):
    """F1 label for one learned arm at k = 1. vs_bar / vs_robust: {rung: {median, p_holm}} against the median-chosen bar and against the best
    zero-collapse hand arm per rung (cd is always a candidate; when no hand arm is clean the caller passes cd). arm_collapse / bar_collapse:
    {rung: count}. anchor_ok: the crossover-k rule. Registered robust-bar clause: where the median-chosen bar has collapse > 0 in a rung, a WIN
    needs the WIN rule against the robust arm too (WIN-ROBUST); clearing only the brittle bar is WIN-OVER-BRITTLE-BAR, never quoted as WIN."""
    if not win_rule(vs_bar):
        return "E-FASTER" if bar_faster(vs_bar) else "TIES"
    if any(arm_collapse[r] > bar_collapse[r] for r in RUNGS):
        return "TIES"
    if not anchor_ok:
        return "NO-GO"
    if any(bar_collapse[r] > 0 for r in RUNGS):
        return "WIN-ROBUST" if win_rule(vs_robust) else "WIN-OVER-BRITTLE-BAR"
    return "WIN"


def read_k0(a):
    topos = [int(x) for x in a.topos.split(",")]
    arms = a.arms.split(",")
    cells, failed = load_gate(a.gate)
    out = {"counts": counts(cells, failed, arms, topos), "health": health_table(cells, arms, topos)}
    out["wall_infeasible"] = wall_infeasible(failed, arms, topos)
    out["k0_health"], out["verdict"] = k0_health(cells, topos)
    out["uncut_share"] = whole_group_share(cells, a.bar_arm, topos)
    out["split_counts"] = split_counts(cells, arms, topos)
    return out


def read_k1(a):
    topos = [int(x) for x in a.topos.split(",")]
    arms = a.arms.split(",")
    cells, failed = load_gate(a.gate)
    out = {"counts": counts(cells, failed, arms, topos, a.suffix), "wall_infeasible": wall_infeasible(failed, arms, topos, a.suffix)}
    pick, ht = pick_hand_arm(cells, arms, topos, a.suffix)
    out["hand_arm"], out["health"] = pick, ht
    out["headroom"] = {r: headroom(cells, pick[r]["decides"], a.suffix, topos, r) for r in RUNGS if pick[r]["decides"]}
    out["a_pass"] = bool(len(out["headroom"]) == 2 and all(v[s]["median"] is not None and v[s]["median"] < HEADROOM_MAX for v in out["headroom"].values() for s in STAT))
    if a.replay:
        sample = json.load(open(a.sample))
        out["b"] = replay_regret(json.load(open(a.replay)), sample, a.sample_sha256, a.sample, {r: pick[r]["decides"] for r in RUNGS})
        out["HAND_SUFFICES"] = bool(out["a_pass"] and out["b"]["pass"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("k0", "k1"):
        p = sub.add_parser(n)
        p.add_argument("--gate", required=True)
        p.add_argument("--topos", required=True)
        p.add_argument("--arms", default="cd,cd_ib0,cd_expand,cd_exactS,cd_exactS_ib0,cd_exactS_ib0_prune")
        p.add_argument("--out")
    p = sub.choices["k0"]
    p.add_argument("--bar-arm", default="cd_expand")
    p = sub.choices["k1"]
    p.add_argument("--suffix", default="_k100")
    p.add_argument("--replay")
    p.add_argument("--sample")
    p.add_argument("--sample-sha256")
    a = ap.parse_args()
    if a.cmd == "k1" and a.replay and not (a.sample and a.sample_sha256):
        raise SystemExit("FAIL LOUD: --replay needs --sample and --sample-sha256")
    o = read_k0(a) if a.cmd == "k0" else read_k1(a)
    if a.out:
        json.dump(o, open(a.out, "w"), indent=1, default=str)
    print(json.dumps(o, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
