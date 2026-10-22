#!/usr/bin/env python3
"""Own-state (DAgger) corpus plumbing, generic for any learned arm: captured snapshots -> state list -> (label, build, relabel: the existing Arm 1 tools) -> combined corpus view and split.

  dagger_corpus.py states --arm A --cells DIR --token T --seed S --plan PLAN.json (keys <topo>__<window><rung>, as the gate writes them) --inputs INPUTS_GROUP --allow-topos 16201 ... --out STATES.json
      One state per captured trigger of the plan, from the arm's own cells DIR/cc40s<topo>__<window><rung>__<T>_s<S>.{snapshots.jsonl,summary.json}. The first capture of a trigger is the
      state (later same-trigger captures are sub-batches). Groups over 256 tasks are counted, not labelled. Uncaptured triggers are counted per cell (`missing`) with no top-up; below 80 % captured the run stops (A2). Fails loud on a topology outside --allow-topos. The states carry check_arm False (the decision is re-decided under the hand arm, not the capturing arm).
      Then:  uncut_arm1.py label STATES.json --per 50 ...   ->  uncut_arm1_build.py STATES.json LABELS... --out CORPUS/dagger_<A> (its parent ids are gnn_datasets_wf1_dagger_<A>/ds_N)  ->
             uncut_label_screen.py relabel/merge on the NEW datasets only (split from `split --new-only`)  ->  `view` + `split` -> the v6 recache over the combined view.
  dagger_corpus.py plan  --inputs INPUTS_GROUP --topos 16201 16202 16203 16204 --out PLAN.json     (amendment A2: 1,000 ids, seed 20261022, half per load; run before capture)
      The pre-drawn R1A_SNAPSHOT_PLAN: arm-independent, Arm 1 stratification (size buckets; 30 % at t < 360 s), groups over 256 excluded; a shortfall is reported, never topped up.
  dagger_corpus.py view  --old-view VIEW --new-corpus CORPUS --arm A [A2 ...] --out COMBINED_VIEW     symlinks: every old base dir as it is, plus gnn_datasets_wf1_dagger_<A> per arm name
  dagger_corpus.py merge-labels --old OLD_SIM_LABELS.json --new NEW_RELABEL.json --out MERGED.json     the one label-override json of the combined cache (ids disjoint)
  dagger_corpus.py split --old-split OLD.json --new-corpus CORPUS --arm A [A2 ...] --out SPLIT.json [--new-only]
      The old split with every new dataset added to TRAIN (val and test unchanged), or with --new-only a split holding only the new datasets (the relabel's frame). The new ids are
      disjoint from the old ones by construction (their own base dir); fails loud otherwise."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

EARLY_T = 360.0
CAP = 256
BUCKETS = (("le10", 1, 10), ("11_50", 11, 50), ("51_100", 51, 100), ("101_256", 101, CAP))


def bucket_of(size: int):
    return next((n for n, lo, hi in BUCKETS if lo <= size <= hi), None)


SHARES = {"le10": 4800, "11_50": 4600, "51_100": 2000, "101_256": 600}       # Arm 1's bucket draw (12,000); the proportions, not the counts, are used
EARLY_SHARE = 0.30
SEED = 20261022          # amendment A2
TOTAL = 1000             # trigger ids per arm and seed (A2); both seeds capture each, so about 2,000 states per arm
EARLY_FLOOR = 0.25       # A2: the corpus-wide share of drawn triggers at t < 360 s; below it the draw fails loud
MIN_CAPTURED = 0.80      # A2: an arm x seed capturing fewer of its drawn triggers stops labelling
WINDOWS, RUNGS = ("g0", "g1", "g2", "g3"), ("moderate", "heavy")


def _largest_remainder(total: int, weights: Dict[Any, float]) -> Dict[Any, int]:
    """Integer quotas summing to `total`, proportional to the weights; ties go to the sorted-first key."""
    z = float(sum(weights.values()))
    raw = {k: total * w / z for k, w in weights.items()}
    out = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: (-(raw[k] - out[k]), str(k)))[: total - sum(out.values())]:
        out[k] += 1
    return out


def draw_plan(wls: Dict[Any, Dict[str, Any]], total: int = TOTAL, seed: int = SEED) -> Dict[str, Any]:
    """The pre-drawn trigger plan (amendment A2): `total` trigger ids per arm over the cells (topology, window, rung) in `wls` (their workloads), as an R1A_SNAPSHOT_PLAN
    {"<topo>__<window><rung>": [trigger ids]}. Arm-independent (workloads only): both arms get the same seed and ids. Quotas follow Arm 1's stratification: size buckets in the Arm 1
    proportions, 30 % of each bucket's draw at group arrival t < 360 s, groups over 256 tasks excluded up front. Every cell draws its quota of each (bucket, stratum) with
    random.Random on (seed, cell, bucket, stratum); a cell with fewer groups than its quota gives all it has and the shortfall is reported (no top-up)."""
    from scripts_cosim.uncut_arm1 import groups_of

    cells = sorted(wls)
    cell_q = _largest_remainder(total, {b: SHARES[b] for b in SHARES})
    plan: Dict[str, List[int]] = {}
    census, short = [], 0
    rungs = sorted({c[2] for c in cells})
    per_cell: Dict[str, Dict[Any, int]] = {}
    for b, q in cell_q.items():                         # half per load first, then equally over that load's cells
        per_cell[b] = {}
        for r, qr in _largest_remainder(q, {r: 1.0 for r in rungs}).items():
            per_cell[b].update(_largest_remainder(qr, {c: 1.0 for c in cells if c[2] == r}))
    for c in cells:
        topo, window, rung = c
        gs = [(trig, size, t) for trig, size, t in groups_of(wls[c]) if bucket_of(size) is not None]
        ids: List[int] = []
        for b in SHARES:
            n_e = round(EARLY_SHARE * per_cell[b][c])
            for early, want in ((True, n_e), (False, per_cell[b][c] - n_e)):
                pool = sorted(trig for trig, size, t in gs if bucket_of(size) == b and (t < EARLY_T) == early)
                got = random.Random(f"dagger_plan:{seed}:{topo}:{window}:{rung}:{b}:{int(early)}").sample(pool, min(want, len(pool)))
                ids += got
                short += want - len(got)
                census.append({"cell": f"{topo}__{window}{rung}", "bucket": b, "early": early, "want": want, "pool": len(pool), "drawn": len(got)})
        plan[f"{topo}__{window}{rung}"] = sorted(ids)
    drawn = sum(len(v) for v in plan.values())
    early = sum(c["drawn"] for c in census if c["early"]) / drawn if drawn else 0.0
    per_rung = {r: sum(len(v) for k, v in plan.items() if k.endswith(r)) for r in RUNGS}
    return {"plan": plan, "census": census, "total_drawn": drawn, "shortfall": short, "quotas": cell_q, "seed": seed, "early_share": early, "early_floor_ok": early >= EARLY_FLOOR,
            "per_rung": per_rung}


def base_name(arm: str) -> str:
    return f"gnn_datasets_wf1_dagger_{arm}"


def first_captures(snapfile: str) -> Dict[int, Dict[str, Any]]:
    """trigger task id -> {time, n_batch} of the FIRST capture led by that trigger, in file order."""
    out: Dict[int, Dict[str, Any]] = {}
    for line in open(snapfile):
        if not line.strip():
            continue
        rec = json.loads(line)
        out.setdefault(int(rec["trigger_task_id"]), {"time": float(rec["time"]), "n_batch": len(rec["fidelity"]["batch"])})
    return out


def make_states(arm: str, cells: Path, token: str, seed: int, plan: Dict[str, Sequence[int]], inputs: Path, allow_topos: Sequence[int], rungs: Sequence[str] = ("moderate", "heavy"),
                min_captured: float = MIN_CAPTURED) -> Dict[str, Any]:
    states: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []
    over = 0
    for key in sorted(plan):                      # the gate's plan key is <topo>__<window><rung>, e.g. 16201__g0moderate
        topo, wr = key.split("__")
        rung = next((r for r in rungs if wr.endswith(r)), None)
        if rung is None:
            raise SystemExit(f"FAIL LOUD: plan key {key}: no rung suffix in {tuple(rungs)}")
        window = wr[:-len(rung)]
        if int(topo) not in set(allow_topos):
            raise SystemExit(f"FAIL LOUD: plan key {key}: topology {topo} is not one of the allowed training topologies {sorted(allow_topos)}")
        stem = cells / f"cc40s{topo}__{window}{rung}__{token}_s{seed}"
        snapfile, summary = str(stem) + ".snapshots.jsonl", str(stem) + ".summary.json"
        for f in (snapfile, summary):
            if not os.path.exists(f):
                raise SystemExit(f"FAIL LOUD: {f} does not exist")
        cfg = inputs / f"wf1_{rung}" / "cfg" / f"cc40s{topo}.json"
        trace = inputs / f"wf1_{rung}" / "wl" / f"grounded_{window}_n50000.json"
        for f in (cfg, trace):
            if not f.exists():
                raise SystemExit(f"FAIL LOUD: {f} does not exist")
        caps = first_captures(snapfile)
        for trig in plan[key]:
            cap = caps.get(int(trig))
            if cap is None:
                missing.append({"topology": int(topo), "window": window, "rung": rung, "trigger": int(trig)})
                continue
            b = bucket_of(cap["n_batch"])
            if b is None:
                over += 1
                continue
            states.append({"key": f"dagger_{arm}:{topo}:{window}:{rung}:{int(trig)}", "set": f"dagger_{arm}", "side": arm, "topology": int(topo), "window": window, "rung": rung,
                           "trigger": int(trig), "time": cap["time"], "n_batch": cap["n_batch"], "bucket": b, "early": cap["time"] < EARLY_T, "snapfile": snapfile,
                           "cell_summary": summary, "cell_config": str(cfg), "trace": str(trace), "check_arm": False})
    drawn = sum(len(v) for v in plan.values()) if plan else 0
    share = (drawn - len(missing)) / drawn if drawn else 0.0
    if share < min_captured:
        raise SystemExit(f"FAIL LOUD (A2 stop rule): arm {arm} seed {seed} captured {share:.3f} of its {drawn} drawn triggers, below {min_captured}; stop labelling and report to SCIENTIST. "
                         f"Missing {len(missing)}, first {missing[:3]}")
    counts: Dict[str, int] = {}
    for st in states:
        counts[st["bucket"]] = counts.get(st["bucket"], 0) + 1
    return {"arm": arm, "token": token, "seed": seed, "states": states, "counts": counts, "groups_over_256": over, "missing": missing, "captured_share": share, "drawn": drawn}


def new_parents(new_corpus: Path, arms: Any) -> List[str]:
    """The new datasets of one arm name or several (e.g. twin_s1, twin_s2: one base dir per seed, since a state key does not contain the seed)."""
    out: List[str] = []
    for arm in ([arms] if isinstance(arms, str) else list(arms)):
        d = new_corpus / f"dagger_{arm}"
        if not d.is_dir():
            raise SystemExit(f"FAIL LOUD: {d} is not a directory (the relabel frame reads CORPUS/<base name without gnn_datasets_wf1_>/ds_*)")
        out += sorted(f"{base_name(arm)}/{p.name}" for p in d.iterdir() if p.name.startswith("ds_") and (p / "placements" / "placements.jsonl").exists())
    return sorted(out)


ACCEL_SPLIT = REPO / "experiments" / "accel_replica_v1_split.json"


def accel_test_names(path: Path = ACCEL_SPLIT) -> List[str]:
    return [str(x) for x in json.load(open(path))["test"]]


def combined_split(old: Dict[str, Any], parents: Sequence[str], new_only: bool, accel_test: Sequence[str] = ()) -> Dict[str, Any]:
    """`accel_test`: the 57 accel_replica_v1 test dataset names. No new dataset id may be one of them, and none of them may be in the resulting TRAIN (SCIENTIST, 2026-10-22)."""
    if not parents:
        raise SystemExit("FAIL LOUD: no new datasets")
    hit = set(parents) & set(accel_test)
    if hit:
        raise SystemExit(f"FAIL LOUD: {len(hit)} new dataset ids are accel_replica_v1 test names, e.g. {sorted(hit)[:3]}")
    out = _combined_split(old, parents, new_only)
    leak = set(out["train"]) & set(accel_test)
    if leak:
        raise SystemExit(f"FAIL LOUD: {len(leak)} accel_replica_v1 test names are in the split's TRAIN, e.g. {sorted(leak)[:3]}")
    return out


def _combined_split(old: Dict[str, Any], parents: Sequence[str], new_only: bool) -> Dict[str, Any]:
    if new_only:
        return {"schema": "split_artifact_v1", "n_parents": len(parents), "train": sorted(parents), "val": [], "test": [], "test_placeholder": False, "dagger_new_only": True}
    seen = set(old["train"]) | set(old["val"]) | set(old["test"])
    clash = seen & set(parents)
    if clash:
        raise SystemExit(f"FAIL LOUD: {len(clash)} new datasets share an id with the old corpus, e.g. {sorted(clash)[:3]}")
    out = {k: v for k, v in old.items()}
    out["train"] = sorted(set(old["train"]) | set(parents))
    out["n_parents"] = len(seen) + len(parents)
    out["dagger"] = {"new_train": len(parents), "base": parents[0].split("/")[0], "old_n_parents": int(old["n_parents"])}
    return out


def write_split(payload: Dict[str, Any], out: Path) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def k1_commit_of_arm1(label_script: str, expect: str) -> str:
    """The K1-read commit the Arm 1 labels were made under, read from the launch script that made them (`--k1-read-commit <hash>`); asserted equal to `expect` (SCIENTIST: ca47cb62)."""
    import re

    found = set(re.findall(r"--k1-read-commit\s+([0-9a-f]{7,40})", open(label_script).read()))
    if found != {expect}:
        raise SystemExit(f"FAIL LOUD: {label_script} names K1-read commits {sorted(found)}, expected exactly {expect}")
    return expect


def label_report(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Counts of a label file: by source and bucket, the replayed-vs-served mismatches (reported, A1), and every drop with its reason. Any drop is listed; a drop that is not an
    expected one (SPLIT / UNPLACED, which take the served plan) makes `ok` False: nothing is dropped silently."""
    by_source: Dict[str, int] = {}
    mismatch = {"checked": 0, "mismatched": 0}
    drops: List[Dict[str, Any]] = []
    for r in rows:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
        if r.get("chosen_match") is not None:
            mismatch["checked"] += 1
            mismatch["mismatched"] += int(not r["chosen_match"])
        if r["source"] == "dropped":
            drops.append({"state_id": r["state_id"], "reason": r.get("reason"), "error": str(r.get("error"))[:200]})
    return {"n": len(rows), "by_source": by_source, "served_vs_redecision": mismatch, "dropped": drops, "ok": not drops}


def merge_labels(old: Path, new: Path, out: Path) -> Dict[str, Any]:
    """One NEW_RTT label-override json for the combined cache: the old simulator labels (old ids only) plus the new relabel (new ids only); ids must be disjoint (A1)."""
    o, n = json.load(open(old)), json.load(open(new))
    om, nm = o.pop("_meta", None), n.pop("_meta", None)
    clash = set(o) & set(n)
    if clash:
        raise SystemExit(f"FAIL LOUD: {len(clash)} state ids are in both label jsons, e.g. {sorted(clash)[:3]}")
    body = {"_meta": {"merged_from": {"old": str(old), "new": str(new)}, "n_old": len(o), "n_new": len(n), "old_meta": om, "new_meta": nm}, **dict(sorted({**o, **n}.items()))}
    raw = json.dumps(body, indent=0) + "\n"
    out.write_text(raw)
    return {"n_old": len(o), "n_new": len(n), "sha256": hashlib.sha256(raw.encode()).hexdigest()}


def graphs_equal(g: Any, h: Any) -> List[str]:
    """Attribute names on which two cached graphs differ (tensors by torch.equal with dtype and shape, anything else by ==); empty when identical."""
    import torch

    bad = []
    keys = set(g.keys() if callable(getattr(g, "keys", None)) else g.keys) | set(h.keys() if callable(getattr(h, "keys", None)) else h.keys)
    for k in sorted(keys):
        a, b = getattr(g, k, None), getattr(h, k, None)
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
                bad.append(k)
        elif isinstance(a, torch.Tensor) != isinstance(b, torch.Tensor):
            bad.append(k)
        else:
            try:
                same = bool(a == b)
            except Exception:
                same = repr(a) == repr(b)
            if not same:
                bad.append(k)
    return bad


def check_cache(old: Path, new: Path) -> Dict[str, Any]:
    """Every graph of the old cache is, graph for graph, in the new cache unchanged (matched by dataset id); the new cache may hold more ids. Fails loud on any difference."""
    import pickle

    og, oi = pickle.load(open(old / "graphs.pkl", "rb")), [str(x) for x in pickle.load(open(old / "dataset_ids.pkl", "rb"))]
    ng, ni = pickle.load(open(new / "graphs.pkl", "rb")), [str(x) for x in pickle.load(open(new / "dataset_ids.pkl", "rb"))]
    by_new = dict(zip(ni, ng))
    missing = [i for i in oi if i not in by_new]
    if missing:
        raise SystemExit(f"FAIL LOUD: {len(missing)} old ids are not in the new cache, e.g. {missing[:3]}")
    diff = {}
    for i, g in zip(oi, og):
        bad = graphs_equal(g, by_new[i])
        if bad:
            diff[i] = bad
    if diff:
        raise SystemExit(f"FAIL LOUD: {len(diff)} old graphs differ in the new cache, e.g. {dict(list(diff.items())[:3])}")
    return {"old_graphs": len(oi), "new_graphs": len(ni), "added": len(ni) - len(oi), "old_graphs_identical": True}


def make_view(old_view: Path, new_corpus: Path, arms: Any, out: Path) -> List[str]:
    if out.exists():
        raise SystemExit(f"FAIL LOUD: {out} exists")
    out.mkdir(parents=True)
    names = []
    for b in sorted(old_view.iterdir()):
        if b.is_dir():
            os.symlink(os.path.realpath(b), out / b.name)
            names.append(b.name)
    for arm in ([arms] if isinstance(arms, str) else list(arms)):
        nb = base_name(arm)
        if nb in names:
            raise SystemExit(f"FAIL LOUD: the old view already has {nb}")
        os.symlink(os.path.realpath(new_corpus / f"dagger_{arm}"), out / nb)
        names.append(nb)
    return names


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("states")
    s.add_argument("--arm", required=True); s.add_argument("--cells", required=True); s.add_argument("--token", required=True); s.add_argument("--seed", type=int, required=True)
    s.add_argument("--plan", required=True); s.add_argument("--inputs", required=True); s.add_argument("--allow-topos", type=int, nargs="+", required=True)
    s.add_argument("--out", required=True)
    d = sub.add_parser("plan")
    d.add_argument("--inputs", required=True, help="inputs_group dir (wf1_<rung>/wl/grounded_<window>_n50000.json)"); d.add_argument("--topos", type=int, nargs="+", required=True)
    d.add_argument("--total", type=int, default=TOTAL); d.add_argument("--seed", type=int, default=SEED); d.add_argument("--out", required=True)
    k = sub.add_parser("k1"); k.add_argument("--arm1-label-script", required=True); k.add_argument("--expect", required=True)
    r_ = sub.add_parser("report"); r_.add_argument("labels", nargs="+")
    v = sub.add_parser("view")
    v.add_argument("--old-view", required=True); v.add_argument("--new-corpus", required=True); v.add_argument("--arm", nargs="+", required=True); v.add_argument("--out", required=True)
    cc = sub.add_parser("check-cache"); cc.add_argument("--old", required=True); cc.add_argument("--new", required=True)
    ml = sub.add_parser("merge-labels"); ml.add_argument("--old", required=True); ml.add_argument("--new", required=True); ml.add_argument("--out", required=True)
    p = sub.add_parser("split")
    p.add_argument("--old-split", required=True); p.add_argument("--new-corpus", required=True); p.add_argument("--arm", nargs="+", required=True); p.add_argument("--out", required=True)
    p.add_argument("--new-only", action="store_true"); p.add_argument("--accel-split", default=str(ACCEL_SPLIT))
    a = ap.parse_args(argv)
    if a.cmd == "states":
        res = make_states(a.arm, Path(a.cells), a.token, a.seed, json.load(open(a.plan)), Path(a.inputs), a.allow_topos)
        Path(a.out).write_text(json.dumps(res, indent=1))
        print(json.dumps({"states": len(res["states"]), "counts": res["counts"], "groups_over_256": res["groups_over_256"], "missing": len(res["missing"])}))
        return 0
    if a.cmd == "plan":
        wls = {(t, w, r): json.load(open(f"{a.inputs}/wf1_{r}/wl/grounded_{w}_n50000.json")) for t in a.topos for w in WINDOWS for r in RUNGS}
        res = draw_plan(wls, a.total, a.seed)
        raw = json.dumps(res["plan"], sort_keys=True, separators=(",", ":"))
        Path(a.out).write_text(raw)
        Path(a.out + ".census.json").write_text(json.dumps({k: v for k, v in res.items() if k != "plan"}, indent=1))
        print(json.dumps({"total_drawn": res["total_drawn"], "shortfall": res["shortfall"], "quotas": res["quotas"], "per_rung": res["per_rung"], "early_share": round(res["early_share"], 4),
                          "plan_sha256": hashlib.sha256(raw.encode()).hexdigest()}))
        if not res["early_floor_ok"]:
            print(f"FAIL LOUD: the draw has {res['early_share']:.3f} of its triggers at t < 360 s, below the {EARLY_FLOOR} floor")
            return 3
        return 0
    if a.cmd == "k1":
        print(k1_commit_of_arm1(a.arm1_label_script, a.expect))
        return 0
    if a.cmd == "report":
        rows = [json.loads(l) for f in a.labels for l in open(f) if l.strip()]
        res = label_report(rows)
        print(json.dumps(res, indent=1))
        return 0 if res["ok"] else 4
    if a.cmd == "view":
        print(json.dumps(make_view(Path(a.old_view), Path(a.new_corpus), a.arm, Path(a.out))))
        return 0
    if a.cmd == "check-cache":
        print(json.dumps(check_cache(Path(a.old), Path(a.new))))
        return 0
    if a.cmd == "merge-labels":
        print(json.dumps(merge_labels(Path(a.old), Path(a.new), Path(a.out))))
        return 0
    parents = new_parents(Path(a.new_corpus), a.arm)
    payload = combined_split(json.load(open(a.old_split)), parents, a.new_only, accel_test_names(Path(a.accel_split)))
    print(json.dumps({"n_new": len(parents), "n_parents": payload["n_parents"], "sha256": write_split(payload, Path(a.out))}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
