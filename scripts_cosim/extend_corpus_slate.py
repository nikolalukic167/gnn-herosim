#!/usr/bin/env python3
"""Lift the declared top-5 candidate cap on a fidelity warm corpus without re-simulating the plans it already has.

A dataset of the accel corpus (scripts_cosim/make_warm_corpus.py --fidelity) offers each task its top 5 candidates
(src/placement/declared_slate.py) and holds a replay-scored (plan, rtt) row for every plan of that slate. The full candidate lists
are in the dataset itself: infrastructure.json -> live_snapshot_seed.fidelity_replay.snapshot is the raw snapshot (restricted to the
sub-batch's tasks), whose task candidates are the 9 / 13 / 7 reachable replicas the slate was cut from. The fidelity replay of one plan
is deterministic, so the plans already scored keep their rtt; only the plans the wider slate adds are simulated.

For each dataset under --max-plans full plans:
  1. rebuild the full-slate view (raw snapshot, every candidate kept) with make_warm_corpus.flag_candidates_per_task, after checking that
     the same call with the old slate reproduces the stored warm_snapshot.json snapshot and the stored per-task candidate lists;
  2. rebuild infrastructure.json from it (make_warm_corpus.build_infrastructure), the fidelity spec's task_candidates and slate widened;
  3. run the ordinary sweep (generate_single_dataset) on that infrastructure with COSIM_SKIP_PLANS_FILE = the old placements.jsonl, so
     the sweep's own enumeration drops the plans already scored and simulates the rest;
  4. re-score --verify old plans through the same sweep on the new infrastructure (COSIM_ONLY_PLANS_FILE) and fail the dataset unless
     every rtt matches the stored one within --verify-tol;
  5. merge: placements.jsonl = the old rows verbatim + the new rows; best.json / optimal_result.json from the better of the two sweeps
     (ties by plan, as the sweep breaks them); placement_metadata.json, warm_snapshot.json, fidelity_replay.json, generation_provenance.json
     and slate_extension.json rewritten with the rule declared_pruning_v1_uncapped, the old slate, and the plan counts before and after.
A dataset above --max-plans, or whose slate was never pruned, is not touched (listed in extend_manifest.jsonl; --link-untouched symlinks
it into --out so the new corpus directory is complete). Nothing is written into --src.

  extend_corpus_slate.py --src <corpus dir> --out <new corpus dir> [--datasets ds_00000 ...] [--first N] [--shard i/n]
      [--max-plans 100000] [--workers 8] [--sim-input data/nofs-ids] [--verify 20] [--link-untouched] [--census]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RULE_UNCAPPED = "declared_pruning_v1_uncapped"
PLAN_FILES = ("best.json", "optimal_result.json", "placement_metadata.json", "placement_progress.txt", "space_with_network.json",
              "workload.json", "infrastructure.json", "warm_snapshot.json", "fidelity_replay.json", "generation_provenance.json")


def plan_key(plan: Dict[Any, Any]) -> Tuple[Tuple[int, int, int], ...]:
    """A plan as the sweep's skip list sees it: (task id, node id, platform id) triples, sorted."""
    return tuple(sorted((int(k), int(v[0]), int(v[1])) for k, v in plan.items()))


def plan_combo(plan: Dict[Any, Any]) -> Tuple[Tuple[int, int], ...]:
    """The sweep's tie-break key (executecosimulation._plan_combo): (node, platform) per task in task order."""
    return tuple((int(v[0]), int(v[1])) for _, v in sorted(plan.items(), key=lambda kv: int(kv[0])))


def read_rows(path: Path) -> List[Tuple[str, Dict[str, Any]]]:
    out = []
    with open(path) as fh:
        for line in fh:
            if line.strip():
                out.append((line.rstrip("\n"), json.loads(line)))
    return out


def best_of(rows: Sequence[Tuple[str, Dict[str, Any]]]) -> Tuple[float, Tuple[Tuple[int, int], ...]]:
    return min((float(r["rtt"]), plan_combo(r["placement_plan"])) for _, r in rows)


def slate_view(dataset: Path) -> Dict[str, Any]:
    """Everything the extension needs, read from the dataset itself (symlinks followed)."""
    ws = json.loads((dataset / "warm_snapshot.json").read_text())
    infra = json.loads((dataset / "infrastructure.json").read_text())
    wl = json.loads((dataset / "workload.json").read_text())
    spec = infra["live_snapshot_seed"]["fidelity_replay"]
    raw = spec["snapshot"]
    draw = ws["provenance"]["candidate_draw"]
    full_sizes = {int(t["task_id"]): len(t["candidates"]) for t in raw["tasks"]}
    old_sizes = {int(t["task_id"]): len(t["candidates"]) for t in ws["snapshot"]["tasks"]}
    return {"ws": ws, "infra": infra, "wl": wl, "spec": spec, "raw": raw, "draw": draw,
            "full_plans": math.prod(full_sizes.values()), "old_plans": math.prod(old_sizes.values()),
            "full_sizes": full_sizes, "old_sizes": old_sizes}


def full_kept(raw: Dict[str, Any]) -> Dict[int, set]:
    return {int(t["task_id"]): {c["queue_key"] for c in t["candidates"]} for t in raw["tasks"]}


def old_kept(ws_snapshot: Dict[str, Any]) -> Dict[int, set]:
    return {int(t["task_id"]): {c["queue_key"] for c in t["candidates"]} for t in ws_snapshot["tasks"]}


def rebuild_checks(v: Dict[str, Any]) -> None:
    """The reconstruction must reproduce what the dataset stores before it is trusted for the wider slate."""
    from scripts_cosim.make_warm_corpus import flag_candidates_per_task

    raw, ws = v["raw"], v["ws"]
    ok, fk = old_kept(ws["snapshot"]), full_kept(raw)
    if set(ok) != set(fk):
        raise RuntimeError(f"FAIL LOUD: stored slate covers tasks {sorted(ok)}, the raw snapshot {sorted(fk)}")
    for g in ok:
        if not ok[g] <= fk[g]:
            raise RuntimeError(f"FAIL LOUD: task {g}: the stored slate is not a subset of the raw candidates ({sorted(ok[g] - fk[g])})")
    if flag_candidates_per_task(raw, ok) != ws["snapshot"]:
        raise RuntimeError("FAIL LOUD: flag_candidates_per_task(raw snapshot, stored slate) does not reproduce warm_snapshot.json's snapshot")
    tids = [int(g) for g in v["wl"]["trace_task_ids"]]
    by_gid = {int(t["task_id"]): t for t in ws["snapshot"]["tasks"]}
    want = [[[int(c["node_id"]), int(c["platform_id"])] for c in by_gid[g]["candidates"]] for g in tids]
    if want != v["spec"]["task_candidates"]:
        raise RuntimeError("FAIL LOUD: the stored task_candidates are not the stored slate in trace_task_ids order")


def new_provenance(v: Dict[str, Any], plans_before: int, plans_after: int) -> Dict[str, Any]:
    prov = deepcopy(v["ws"]["provenance"])
    old_draw = prov["candidate_draw"]
    sizes = v["full_sizes"]
    order = old_draw["task_ids"]
    prov["candidate_draw"] = {
        "rule": RULE_UNCAPPED, "plans": plans_after, "num_combos": plans_after, "full_plans": v["full_plans"], "pruned": False,
        "sub_batch": old_draw.get("sub_batch"), "of": old_draw.get("of"),
        "per_task_candidates": [sizes[int(g)] for g in order], "task_ids": list(order),
        "old_slate": {k: old_draw.get(k) for k in ("rule", "plans", "num_combos", "per_task_candidates", "pruned")},
        "plans_before": plans_before, "plans_after": plans_after,
    }
    return prov


def build_new_infra(v: Dict[str, Any], prov: Dict[str, Any], sim_input: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    from scripts_cosim.make_warm_corpus import build_infrastructure, flag_candidates_per_task

    infra, raw = v["infra"], v["raw"]
    flagged = flag_candidates_per_task(raw, full_kept(raw))
    base = {k: infra[k] for k in ("network_maps", "compute_slots_per_node", "ingress_bandwidth_mbps", "link_topology", "replica_placement_rule") if k in infra}
    base["metadata"] = {k: val for k, val in infra["metadata"].items() if k != "warm_snapshot"}
    new = build_infrastructure(base, {k: val for k, val in flagged.items() if k != "fidelity"}, prov, None)
    spec = deepcopy(v["spec"])
    by_gid = {int(t["task_id"]): t for t in flagged["tasks"]}
    spec["task_candidates"] = [[[int(c["node_id"]), int(c["platform_id"])] for c in by_gid[int(g)]["candidates"]] for g in v["wl"]["trace_task_ids"]]
    spec["candidate_slate"] = RULE_UNCAPPED
    spec["slate"] = {k: val for k, val in prov["candidate_draw"].items() if k != "task_ids"}
    spec["sim_input_before_extension"] = spec.get("sim_input")
    spec["sim_input"] = str(sim_input)
    new["live_snapshot_seed"][_spec_key()] = spec
    prov["fidelity_replay"] = {k: val for k, val in spec.items() if k != "snapshot"}
    new["metadata"]["warm_snapshot"] = prov
    return new, flagged


def _spec_key() -> str:
    from src.placement import fidelity_replay

    return fidelity_replay.SPEC_KEY


def run_sweep(tag: str, workdir: Path, wl_path: Path, v: Dict[str, Any], config: Dict[str, Any], infra_path: Path, args, env: Dict[str, str]) -> Tuple[str, float, float]:
    from scripts_cosim.generate_gnn_datasets_fast import generate_single_dataset

    sim_data = PROJECT_ROOT / "simulation_data"
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        return generate_single_dataset(
            tag, workdir, config, wl_path, args.sim_input,
            sim_data / "sample_simple.json", sim_data / "lhs_samples_simple.npy", sim_data / "lhs_samples_simple_mapping.pkl",
            seed=int(v["ws"]["provenance"]["cell_seed"]), max_workers=args.workers, quiet=True,
            fast_forward_warmup=True, fast_forward_threshold=1, allow_non_unique_replicas=True, warmth_physics="node_disk_v2",
            grid_name=f"warm_snapshot:{v['ws']['provenance']['source_tag']}", num_tasks=len(v["wl"]["events"]),
            infrastructure_override=infra_path)
    finally:
        for k, old in saved.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old


def check_env(dataset: Path) -> None:
    rec = json.loads((dataset / "generation_provenance.json").read_text()).get("physics_env") or {}
    bad = {k: (val, os.environ.get(k)) for k, val in rec.items() if k != "HEROSIM_PY" and os.environ.get(k) != val}
    if bad:
        raise SystemExit(f"FAIL LOUD: the corpus was built under physics env {rec}; this process differs on {bad}")


def sweep_ok(workdir: Path, expect_rows: int) -> Dict[str, Any]:
    meta = json.loads((workdir / "placement_metadata.json").read_text())
    rows = sum(1 for line in open(workdir / "placements" / "placements.jsonl") if line.strip())
    if not meta.get("sweep_complete") or rows != expect_rows or meta.get("num_placements") != expect_rows:
        raise RuntimeError(f"FAIL LOUD: sweep of {workdir.name} incomplete: {rows} rows, expected {expect_rows}, metadata {meta}")
    return meta


def rate_of(workdir: Path) -> Optional[float]:
    for line in (workdir / "placement_progress.txt").read_text().splitlines():
        if line.startswith("Rate:"):
            return float(line.split()[1])
    return None


def extend_one(src: Path, dst: Path, args) -> Dict[str, Any]:
    name = src.name
    v = slate_view(src)
    rec: Dict[str, Any] = {"dataset": name, "plans_old_slate": v["old_plans"], "full_plans": v["full_plans"],
                           "per_task_full": [v["full_sizes"][int(g)] for g in v["draw"]["task_ids"]]}
    if v["full_plans"] > args.max_plans:
        rec["status"] = "untouched_over_cap"
        return rec
    if v["full_plans"] == v["old_plans"]:
        rec["status"] = "untouched_already_full"
        return rec
    if args.census:
        rec["status"] = "would_extend"
        rec["new_plans"] = v["full_plans"] - v["old_plans"]
        return rec
    check_env(src)
    rebuild_checks(v)
    old_rows = read_rows(src / "placements" / "placements.jsonl")
    if len(old_rows) != v["old_plans"]:
        raise RuntimeError(f"FAIL LOUD: {name} stores {len(old_rows)} rows for an old slate of {v['old_plans']} plans")
    if len({plan_key(r["placement_plan"]) for _, r in old_rows}) != len(old_rows):
        raise RuntimeError(f"FAIL LOUD: {name} has duplicate plans in placements.jsonl")

    work = args.scratch / name
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    prov = new_provenance(v, len(old_rows), v["full_plans"])
    infra, flagged = build_new_infra(v, prov, args.sim_input)
    infra_path = work / "infrastructure_override.json"
    infra_path.write_text(json.dumps(infra))
    wl_src = work / "sweep_workload.json"
    wl_src.write_text(json.dumps(v["wl"]))
    config = json.loads((src / "space_with_network.json").read_text())
    config["warm_snapshot"] = prov
    skip = work / "old_plans.jsonl"
    shutil.copyfile(src / "placements" / "placements.jsonl", skip)

    # a) the new plans
    new_dir = work / "new"
    t0 = time.time()
    status, new_best_rtt, secs = run_sweep("new", new_dir, wl_src, v, config, infra_path, args, {"COSIM_SKIP_PLANS_FILE": str(skip)})
    if status != "success":
        raise RuntimeError(f"FAIL LOUD: {name}: the sweep of the added plans returned {status!r}")
    report = json.loads((new_dir / "skip_plans_report.json").read_text())
    new_rows = read_rows(new_dir / "placements" / "placements.jsonl")
    sweep_ok(new_dir, report["to_simulate"])
    old_keys = {plan_key(r["placement_plan"]) for _, r in old_rows}
    new_keys = {plan_key(r["placement_plan"]) for _, r in new_rows}
    if old_keys & new_keys:
        raise RuntimeError(f"FAIL LOUD: {name}: {len(old_keys & new_keys)} plans scored twice")
    if report["listed_in_enumeration"] != len(old_rows):
        raise RuntimeError(f"FAIL LOUD: {name}: {len(old_rows) - report['listed_in_enumeration']} stored plans are not in the full slate's enumeration")
    total = len(old_rows) + len(new_rows)
    if total != report["enumerated"]:
        raise RuntimeError(f"FAIL LOUD: {name}: {total} rows after the merge for {report['enumerated']} enumerated plans")
    sweep_rate = rate_of(new_dir)

    # b) the old plans re-scored on the new infrastructure
    verified: Dict[str, Any] = {"n": 0, "max_abs_diff": 0.0}
    if args.verify > 0:
        step = max(1, len(old_rows) // args.verify)
        picked = old_rows[::step][:args.verify]
        only = work / "verify_plans.jsonl"
        only.write_text("".join(line + "\n" for line, _ in picked))
        vdir = work / "verify"
        vstatus, _, _ = run_sweep("verify", vdir, wl_src, v, config, infra_path, args, {"COSIM_ONLY_PLANS_FILE": str(only)})
        if vstatus != "success":
            raise RuntimeError(f"FAIL LOUD: {name}: the re-score sweep returned {vstatus!r}")
        again = {plan_key(r["placement_plan"]): float(r["rtt"]) for _, r in read_rows(vdir / "placements" / "placements.jsonl")}
        if set(again) != {plan_key(r["placement_plan"]) for _, r in picked}:
            raise RuntimeError(f"FAIL LOUD: {name}: the re-score returned {len(again)} of {len(picked)} plans")
        diffs = [abs(again[plan_key(r["placement_plan"])] - float(r["rtt"])) for _, r in picked]
        verified = {"n": len(picked), "max_abs_diff": max(diffs)}
        if max(diffs) > args.verify_tol:
            raise RuntimeError(f"FAIL LOUD: {name}: re-scoring stored plans on the extended infrastructure moved an rtt by {max(diffs):.3e} (> {args.verify_tol})")

    # c) merge
    out = dst / name
    if out.exists():
        shutil.rmtree(out)
    (out / "placements").mkdir(parents=True)
    with open(out / "placements" / "placements.jsonl", "w") as fh:
        for line, _ in old_rows + new_rows:
            fh.write(line + "\n")
    old_best, new_best = best_of(old_rows), best_of(new_rows)
    stored_best = json.loads((src / "best.json").read_text())["rtt"]
    if abs(stored_best - old_best[0]) > 1e-12:
        raise RuntimeError(f"FAIL LOUD: {name}: best.json says {stored_best}, the stored rows' minimum is {old_best[0]}")
    new_wins = new_best < old_best
    winner = new_dir if new_wins else src
    shutil.copyfile(winner / "optimal_result.json", out / "optimal_result.json")
    best_info = json.loads((winner / "best.json").read_text())
    (out / "best.json").write_text(json.dumps({"file": best_info["file"], "rtt": min(old_best, new_best)[0]}))
    meta_old = json.loads((src / "placement_metadata.json").read_text())
    ext = {"rule": RULE_UNCAPPED, "plans_before": len(old_rows), "plans_added": len(new_rows), "plans_after": total,
           "best_before": old_best[0], "best_after": min(old_best, new_best)[0], "new_best_wins": bool(new_wins),
           "verified_old_plans": verified, "per_task_candidates_before": v["draw"]["per_task_candidates"],
           "per_task_candidates_after": prov["candidate_draw"]["per_task_candidates"], "old_slate": prov["candidate_draw"]["old_slate"],
           "sweep_seconds_added": secs, "sim_per_second_added": sweep_rate, "workers": args.workers,
           "code_commit": os.environ.get("EXTEND_CODE_COMMIT")}
    meta = dict(meta_old)
    meta.update({"num_placements": total, "completed": total, "rows_written": total, "sweep_complete": True, "slate_extension": ext})
    (out / "placement_metadata.json").write_text(json.dumps(meta))
    (out / "placement_progress.txt").write_text(f"{total}/{total}\nBest RTT: {min(old_best, new_best)[0]:.3f}s\nStatus: COMPLETE\n")
    (out / "slate_extension.json").write_text(json.dumps(ext, indent=1))
    (out / "warm_snapshot.json").write_text(json.dumps({"snapshot": flagged, "provenance": prov}))
    shutil.copyfile(new_dir / "infrastructure.json", out / "infrastructure.json")
    shutil.copyfile(new_dir / "space_with_network.json", out / "space_with_network.json")
    shutil.copyfile(src / "workload.json", out / "workload.json")
    fr = json.loads((src / "fidelity_replay.json").read_text())
    fr.update({"candidate_slate": RULE_UNCAPPED, "slate_extension": {k: ext[k] for k in ("plans_before", "plans_added", "plans_after")}})
    (out / "fidelity_replay.json").write_text(json.dumps(fr, indent=1))
    gp = json.loads((src / "generation_provenance.json").read_text())
    gp["slate_extension"] = ext
    (out / "generation_provenance.json").write_text(json.dumps(gp, indent=1))
    from src.placement.sweep_status import sweep_status

    done, why = sweep_status(out)
    if not done:
        raise RuntimeError(f"FAIL LOUD: {name}: the merged dataset is not a complete sweep ({why})")
    if not args.keep_scratch:
        shutil.rmtree(work, ignore_errors=True)
    rec.update({"status": "extended", "plans_before": len(old_rows), "plans_added": len(new_rows), "plans_after": total,
                "best_before": old_best[0], "best_after": min(old_best, new_best)[0], "new_best_wins": bool(new_wins),
                "verified": verified, "sweep_seconds": secs, "wall_seconds": time.time() - t0, "sim_per_second": sweep_rate,
                "workers": args.workers,
                "worker_seconds_per_plan": (args.workers / sweep_rate) if sweep_rate else None})
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--first", type=int, default=None, help="only the first N datasets in name order (after --shard)")
    ap.add_argument("--shard", default=None, help="i/n: every n-th dataset starting at i")
    ap.add_argument("--max-plans", type=int, default=100_000, help="full plans above which a dataset is left untouched")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--sim-input", type=Path, default=PROJECT_ROOT / "data" / "nofs-ids")
    ap.add_argument("--verify", type=int, default=20, help="stored plans re-scored on the extended infrastructure per dataset (0 = off)")
    ap.add_argument("--verify-tol", type=float, default=1e-9)
    ap.add_argument("--link-untouched", action="store_true", help="symlink untouched datasets into --out")
    ap.add_argument("--census", action="store_true", help="count what would be extended; simulate nothing")
    ap.add_argument("--keep-scratch", action="store_true")
    args = ap.parse_args()
    if args.out.resolve() == args.src.resolve() or str(args.out.resolve()).startswith(str(args.src.resolve()) + os.sep):
        raise SystemExit("FAIL LOUD: --out must be a new directory outside --src; the source corpus is never written")
    names = sorted(p.name for p in args.src.iterdir() if p.name.startswith("ds_") and p.is_dir())
    if args.datasets:
        missing = [d for d in args.datasets if d not in names]
        if missing:
            raise SystemExit(f"FAIL LOUD: not in {args.src}: {missing}")
        names = sorted(args.datasets)
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        names = names[i::n]
    if args.first is not None:
        names = names[:args.first]
    args.out.mkdir(parents=True, exist_ok=True)
    args.scratch = args.out / ".extend_scratch"
    args.scratch.mkdir(exist_ok=True)
    manifest = args.out / ("extend_census.jsonl" if args.census else f"extend_manifest{'.' + args.shard.replace('/', 'of') if args.shard else ''}.jsonl")
    counts: Dict[str, int] = {}
    for name in names:
        src = args.src / name
        if not args.census and (args.out / name).exists() and (args.out / name / "slate_extension.json").exists():
            rec = {"dataset": name, "status": "done_earlier"}
        else:
            try:
                rec = extend_one(src, args.out, args)
            except Exception as exc:  # one failed dataset must not stop the run; it is listed and the exit code says so
                rec = {"dataset": name, "status": "failed", "error": f"{type(exc).__name__}: {exc}"[:600]}
                print(f"[extend] {name}: FAILED {rec['error']}", flush=True)
            if rec["status"].startswith("untouched") and args.link_untouched and not args.census:
                link = args.out / name
                if not link.exists() and not link.is_symlink():
                    link.symlink_to(src.resolve())
        counts[rec["status"]] = counts.get(rec["status"], 0) + 1
        with open(manifest, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        if rec["status"] == "extended":
            print(f"[extend] {name}: {rec['plans_before']} -> {rec['plans_after']} plans, best {rec['best_before']:.4f} -> {rec['best_after']:.4f}, "
                  f"{rec['worker_seconds_per_plan']:.3f} worker-s/plan", flush=True)
    print(f"[extend] {len(names)} datasets: {counts}", flush=True)
    return 1 if counts.get("failed") else 0


if __name__ == "__main__":
    sys.exit(main())
