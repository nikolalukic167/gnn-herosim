#!/usr/bin/env python3
"""Own-state (DAgger) corpus plumbing, generic for any learned arm: captured snapshots -> state list -> (label, build, relabel: the existing Arm 1 tools) -> combined corpus view and split.

  dagger_corpus.py states --arm A --cells DIR --token T --seed S --plan PLAN.json (keys <topo>__<window><rung>, as the gate writes them) --inputs INPUTS_GROUP --allow-topos 16201 ... --out STATES.json
      One state per captured trigger of the plan, from the arm's own cells DIR/cc40s<topo>__<window><rung>__<T>_s<S>.{snapshots.jsonl,summary.json}. The first capture of a trigger is the
      state (later same-trigger captures are sub-batches). Groups over 256 tasks are counted, not labelled. A trigger of the plan that was never captured is listed (`missing`) and fails
      loud unless --allow-missing. Fails loud on a topology outside --allow-topos. The states carry check_arm False (the decision is re-decided under the hand arm, not the capturing arm).
      Then:  uncut_arm1.py label STATES.json --per 50 ...   ->  uncut_arm1_build.py STATES.json LABELS... --out CORPUS/dagger_<A> (its parent ids are gnn_datasets_wf1_dagger_<A>/ds_N)  ->
             uncut_label_screen.py relabel/merge on the NEW datasets only (split from `split --new-only`)  ->  `view` + `split` -> the v6 recache over the combined view.
  dagger_corpus.py view  --old-view VIEW --new-corpus CORPUS --arm A --out COMBINED_VIEW     symlinks: every old base dir as it is, plus gnn_datasets_wf1_dagger_<A>
  dagger_corpus.py split --old-split OLD.json --new-corpus CORPUS --arm A --out SPLIT.json [--new-only]
      The old split with every new dataset added to TRAIN (val and test unchanged), or with --new-only a split holding only the new datasets (the relabel's frame). The new ids are
      disjoint from the old ones by construction (their own base dir); fails loud otherwise."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
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
                allow_missing: bool = False) -> Dict[str, Any]:
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
    if missing and not allow_missing:
        raise SystemExit(f"FAIL LOUD: {len(missing)} planned triggers were never captured, first {missing[:3]}")
    counts: Dict[str, int] = {}
    for st in states:
        counts[st["bucket"]] = counts.get(st["bucket"], 0) + 1
    return {"arm": arm, "token": token, "seed": seed, "states": states, "counts": counts, "groups_over_256": over, "missing": missing}


def new_parents(new_corpus: Path, arm: str) -> List[str]:
    d = new_corpus / f"dagger_{arm}"
    if not d.is_dir():
        raise SystemExit(f"FAIL LOUD: {d} is not a directory (the relabel frame reads CORPUS/<base name without gnn_datasets_wf1_>/ds_*)")
    return sorted(f"{base_name(arm)}/{p.name}" for p in d.iterdir() if p.name.startswith("ds_") and (p / "placements" / "placements.jsonl").exists())


def combined_split(old: Dict[str, Any], parents: Sequence[str], new_only: bool) -> Dict[str, Any]:
    if not parents:
        raise SystemExit("FAIL LOUD: no new datasets")
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


def make_view(old_view: Path, new_corpus: Path, arm: str, out: Path) -> List[str]:
    if out.exists():
        raise SystemExit(f"FAIL LOUD: {out} exists")
    out.mkdir(parents=True)
    names = []
    for b in sorted(old_view.iterdir()):
        if b.is_dir():
            os.symlink(os.path.realpath(b), out / b.name)
            names.append(b.name)
    nb = base_name(arm)
    if nb in names:
        raise SystemExit(f"FAIL LOUD: the old view already has {nb}")
    os.symlink(os.path.realpath(new_corpus / f"dagger_{arm}"), out / nb)
    return names + [nb]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("states")
    s.add_argument("--arm", required=True); s.add_argument("--cells", required=True); s.add_argument("--token", required=True); s.add_argument("--seed", type=int, required=True)
    s.add_argument("--plan", required=True); s.add_argument("--inputs", required=True); s.add_argument("--allow-topos", type=int, nargs="+", required=True)
    s.add_argument("--allow-missing", action="store_true"); s.add_argument("--out", required=True)
    v = sub.add_parser("view")
    v.add_argument("--old-view", required=True); v.add_argument("--new-corpus", required=True); v.add_argument("--arm", required=True); v.add_argument("--out", required=True)
    p = sub.add_parser("split")
    p.add_argument("--old-split", required=True); p.add_argument("--new-corpus", required=True); p.add_argument("--arm", required=True); p.add_argument("--out", required=True)
    p.add_argument("--new-only", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "states":
        res = make_states(a.arm, Path(a.cells), a.token, a.seed, json.load(open(a.plan)), Path(a.inputs), a.allow_topos, allow_missing=a.allow_missing)
        Path(a.out).write_text(json.dumps(res, indent=1))
        print(json.dumps({"states": len(res["states"]), "counts": res["counts"], "groups_over_256": res["groups_over_256"], "missing": len(res["missing"])}))
        return 0
    if a.cmd == "view":
        print(json.dumps(make_view(Path(a.old_view), Path(a.new_corpus), a.arm, Path(a.out))))
        return 0
    parents = new_parents(Path(a.new_corpus), a.arm)
    payload = combined_split(json.load(open(a.old_split)), parents, a.new_only)
    print(json.dumps({"n_new": len(parents), "n_parents": payload["n_parents"], "sha256": write_split(payload, Path(a.out))}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
