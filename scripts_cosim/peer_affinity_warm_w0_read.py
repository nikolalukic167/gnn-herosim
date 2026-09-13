#!/usr/bin/env python3
"""peer_affinity_warm_v1 -- the W0 screen read (docs/lineages/peer_affinity_warm_v1.md).

Three registered reads on the warm-snapshot cache, bars fixed 2026-09-13 before any
dataset existed:

  W0.a support closure   xavierGpu is a candidate in >= 20 % of datasets and >= 2 % of
                         candidate slots; the dim-7 queue column over busy platforms
                         reaches p90 >= 20 and max >= 150 across datasets.
  W0.b joint structure   score_route_b_contention `r_exact_band.mean_tied` at alpha 2.5:
                         NO-GO for W1 if median <= 1.0 % AND <= 20 % of datasets > 2.0 %.
                         Read from the scorer's JSON (--scorer-json, one per source).
  W0.c cold checkpoints  the T1b gnn/mpoff lr2e3 checkpoints (16 seeds each) evaluated
                         on the warm cache with the serving decoder, regret vs the sweep
                         optimum at alpha 2.5, paired by seed. Descriptive.

Usage (datalab, after the W0 cache exists):
  python3 scripts_cosim/peer_affinity_warm_w0_read.py \\
      --cache-dir simulation_data/graphs_cache_peer_affinity_warm_v1_w0 \\
      --scorer-json simulation_data/peer_affinity_warm_v1/w0_score_knb.json \\
                    simulation_data/peer_affinity_warm_v1/w0_score_gnn.json \\
      --out simulation_data/peer_affinity_warm_v1/w0_read.json
"""
from __future__ import annotations

import argparse
import collections
import json
import pickle
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src" / "notebooks")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

BARS = {
    "a_xaviergpu_dataset_share_min": 0.20,
    "a_xaviergpu_slot_share_min": 0.02,
    "a_dim7_busy_p90_min": 20.0,
    "a_dim7_busy_max_min": 150.0,
    "b_nogo_median_max_pct": 1.0,
    "b_nogo_frac_gt2_max": 0.20,
}


def _source_of(dataset_id: str) -> str:
    # cache dataset ids are "<base_dir_name>/ds_XXXXX"
    base = str(dataset_id).split("/")[0]
    return base.rsplit("_", 1)[-1] if "_w0_" in base else base


def read_support(graphs: List[Any], dataset_ids: List[str]) -> Dict[str, Any]:
    per_source: Dict[str, Dict[str, Any]] = {}
    pooled = {"datasets": 0, "with_gpu": 0, "slots": 0, "gpu_slots": 0, "dim7_busy": [], "dim7_max": []}
    for g, did in zip(graphs, dataset_ids):
        src = _source_of(did)
        acc = per_source.setdefault(src, {"datasets": 0, "with_gpu": 0, "slots": 0, "gpu_slots": 0, "dim7_busy": [], "dim7_max": []})
        meta = g.queue_key_to_platform_meta
        plat_type = {}
        for qk, m in meta.items():
            plat_type[qk] = str(m.get("platform_type", ""))
        n_slots = 0
        n_gpu = 0
        has_gpu = False
        tl = g.task_logit_to_queue_key
        keys_per_task = tl.values() if isinstance(tl, dict) else tl
        for keys in keys_per_task:
            for qk in keys:
                n_slots += 1
                if plat_type.get(qk) == "xavierGpu":
                    n_gpu += 1
                    has_gpu = True
        pf = g.platform_features.numpy()
        busy = pf[pf[:, 7] > 0, 7]
        for a in (acc, pooled):
            a["datasets"] += 1
            a["with_gpu"] += int(has_gpu)
            a["slots"] += n_slots
            a["gpu_slots"] += n_gpu
            a["dim7_busy"].extend(busy.tolist())
            a["dim7_max"].append(float(pf[:, 7].max()))

    def _fin(a: Dict[str, Any]) -> Dict[str, Any]:
        busy = np.array(a["dim7_busy"]) if a["dim7_busy"] else np.array([0.0])
        return {
            "n_datasets": a["datasets"],
            "xaviergpu_dataset_share": a["with_gpu"] / max(1, a["datasets"]),
            "xaviergpu_slot_share": a["gpu_slots"] / max(1, a["slots"]),
            "dim7_busy_p50": float(np.median(busy)),
            "dim7_busy_p90": float(np.percentile(busy, 90)),
            "dim7_busy_max": float(busy.max()),
            "dim7_max_over_datasets": float(max(a["dim7_max"])) if a["dim7_max"] else 0.0,
            "busy_platforms_per_dataset_mean": float(len(a["dim7_busy"]) / max(1, a["datasets"])),
        }

    out = {"per_source": {k: _fin(v) for k, v in per_source.items()}, "pooled": _fin(pooled)}
    p = out["pooled"]
    out["pass"] = bool(
        p["xaviergpu_dataset_share"] >= BARS["a_xaviergpu_dataset_share_min"]
        and p["xaviergpu_slot_share"] >= BARS["a_xaviergpu_slot_share_min"]
        and p["dim7_busy_p90"] >= BARS["a_dim7_busy_p90_min"]
        and p["dim7_max_over_datasets"] >= BARS["a_dim7_busy_max_min"]
    )
    return out


def read_structure(scorer_jsons: List[Path], alpha: str) -> Dict[str, Any]:
    """score_route_b_contention writes a LIST of reports (one per --corpus); per_dataset
    and per_alpha are keyed by the alpha string."""
    per_source: Dict[str, Any] = {}
    pooled: List[float] = []
    for path in scorer_jsons:
        loaded = json.loads(Path(path).read_text())
        reports = loaded if isinstance(loaded, list) else [loaded]
        for rep in reports:
            per_alpha = rep.get("per_alpha", {})
            key = next((k for k in per_alpha if k != "None" and float(k) == float(alpha)), None)
            if key is None:
                raise SystemExit(f"{path}: alpha {alpha} not in {list(per_alpha)}")
            block = per_alpha[key]
            entries = (rep.get("per_dataset") or {}).get(key, [])
            vals = [float(r["r_exact_band"]["mean_tied"]) for r in entries if "r_exact_band" in r]
            pooled.extend(vals)
            per_source[f"{Path(path).stem}:{Path(rep.get('corpus', '')).name}"] = {
                "n": len(vals),
                "n_datasets_in_report": rep.get("n_datasets"),
                "failures": rep.get("failures"),
                "mean_tied_median_pct": statistics.median(vals) if vals else None,
                "frac_gt_2pct": (sum(1 for v in vals if v > 2.0) / len(vals)) if vals else None,
                "frac_gt_1pct": (sum(1 for v in vals if v > 1.0) / len(vals)) if vals else None,
                "scorer_summary_r_exact_band": block.get("r_exact_band"),
                "scorer_summary_repairs": {
                    k: block.get(k) for k in (
                        "r_exact_repaired_1int", "r_exact_repaired_kint", "r_exact_repaired_lnk",
                        "r_exact_repaired_t1", "r_exact_repaired_t1lnk", "r_exact_ls", "r_greedy",
                    ) if k in block
                },
                "n_exact_scored": block.get("n_exact_scored"),
                "no_feasible_rows": block.get("no_feasible_rows"),
            }
    med = statistics.median(pooled) if pooled else None
    frac2 = (sum(1 for v in pooled if v > 2.0) / len(pooled)) if pooled else None
    nogo = bool(pooled) and med <= BARS["b_nogo_median_max_pct"] and frac2 <= BARS["b_nogo_frac_gt2_max"]
    return {
        "alpha": alpha, "per_source": per_source,
        "pooled": {"n": len(pooled), "mean_tied_median_pct": med, "frac_gt_2pct": frac2,
                   "frac_gt_1pct": (sum(1 for v in pooled if v > 1.0) / len(pooled)) if pooled else None,
                   "values_sorted": sorted(pooled)},
        "verdict": "NO-GO" if nogo else ("GO" if pooled else "UNREAD"),
    }


def read_cold_checkpoints(graphs, dataset_ids, args) -> Dict[str, Any]:
    import os

    from scripts_cosim.eval_route_b_stage2_arm import evaluate_dataset, load_arm

    # The serving decoder the T1b arms were gated with: replica reuse on, relax on
    # (experiments/peer_affinity_v1_t1b_*: NEAR_RTT_DECODE_REPLICA_REUSE=1, _RELAX=1). The
    # evaluator's defaults are route_b's no-reuse decoder, which has NO plan on a warm
    # dataset (two types x five tasks over ~3 candidates per type) and reads "infeasible".
    os.environ.setdefault("EVAL_DECODE_REPLICA_REUSE", "1")
    os.environ.setdefault("EVAL_DECODE_RELAX", "1")
    task_types_db = json.loads(Path(args.task_types).read_text())
    sim_root = REPO_ROOT / "simulation_data"
    per_ck: Dict[str, Any] = {}
    for arm_name in args.arms:
        for seed in args.seeds:
            ck = args.models_dir / f"{args.tag}-{arm_name}-{args.lr}-seed{seed}.pt"
            if not ck.is_file():
                raise SystemExit(f"missing checkpoint: {ck}")
            # load_arm checks the sidecar's GNN_DISABLE_MESSAGE_PASSING against the
            # environment (a mismatch is a train/serve error, not an ablation), so the
            # flag follows the arm being loaded.
            os.environ["GNN_DISABLE_MESSAGE_PASSING"] = "1" if arm_name == "mpoff" else "0"
            arm = load_arm(ck)
            regrets: Dict[str, List[float]] = collections.defaultdict(list)
            infeasible = 0
            for graph, did in zip(graphs, dataset_ids):
                res = evaluate_dataset(graph, did, arm=arm, alpha_key=args.alpha_key,
                                       simulation_data_root=sim_root, task_types_db=task_types_db)
                if res["infeasible"]:
                    infeasible += 1
                else:
                    regrets[_source_of(did)].append(res["decode_regret_pct"]["registered"])
                    regrets["pooled"].append(res["decode_regret_pct"]["registered"])
            if not regrets["pooled"]:
                raise SystemExit(f"FAIL LOUD: {ck} decoded no feasible plan on any of {len(graphs)} datasets")
            per_ck[f"{arm_name}|seed{seed}"] = {
                "arm": arm_name, "seed": seed, "infeasible": infeasible,
                **{f"median_regret_pct_{k}": statistics.median(v) for k, v in regrets.items()},
                **{f"mean_regret_pct_{k}": statistics.fmean(v) for k, v in regrets.items()},
                "n": len(regrets["pooled"]),
            }
            print(f"[w0.c] {arm_name} seed {seed}: median {per_ck[f'{arm_name}|seed{seed}']['median_regret_pct_pooled']:.2f}% "
                  f"(infeasible {infeasible})", flush=True)
    summary: Dict[str, Any] = {}
    for scope in sorted({k.split("median_regret_pct_", 1)[1] for r in per_ck.values() for k in r if k.startswith("median_regret_pct_")}):
        per_arm = {a: {r["seed"]: r.get(f"median_regret_pct_{scope}") for r in per_ck.values() if r["arm"] == a} for a in args.arms}
        entry: Dict[str, Any] = {a: statistics.median([v for v in per_arm[a].values() if v is not None]) for a in args.arms}
        if len(args.arms) == 2:
            a, b = args.arms
            seeds = sorted(set(per_arm[a]) & set(per_arm[b]))
            diffs = [per_arm[b][s] - per_arm[a][s] for s in seeds if per_arm[a][s] is not None and per_arm[b][s] is not None]
            entry["n_pairs"] = len(diffs)
            entry[f"{b}_minus_{a}_pp_median"] = statistics.median(diffs) if diffs else None
            entry[f"{a}_better_seeds"] = sum(1 for d in diffs if d > 0)
            if len(diffs) > 1:
                from scipy.stats import wilcoxon
                entry["p_exact_wilcoxon"] = float(wilcoxon(diffs, alternative="two-sided", mode="exact").pvalue)
        summary[scope] = entry
    return {"per_checkpoint": per_ck, "summary": summary, "tag": args.tag, "lr": args.lr, "alpha_key": args.alpha_key}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--scorer-json", type=Path, nargs="*", default=[])
    ap.add_argument("--alpha-key", default="2.5")
    ap.add_argument("--models-dir", type=Path, default=Path("models"))
    ap.add_argument("--tag", default="peer-affinity-v1-t1b")
    ap.add_argument("--lr", default="lr2e3")
    ap.add_argument("--arms", nargs="+", default=["gnn", "mpoff"])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 17)))
    ap.add_argument("--task-types", default="data/nofs-ids/task-types.json")
    ap.add_argument("--skip-checkpoints", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    with open(args.cache_dir / "graphs.pkl", "rb") as fh:
        graphs = pickle.load(fh)
    with open(args.cache_dir / "dataset_ids.pkl", "rb") as fh:
        dataset_ids = pickle.load(fh)
    print(f"[w0] {len(graphs)} graphs in {args.cache_dir}", flush=True)

    out: Dict[str, Any] = {"lineage": "peer_affinity_warm_v1", "stage": "W0", "bars": BARS,
                           "cache_dir": str(args.cache_dir), "n_datasets": len(graphs)}
    out["w0a_support"] = read_support(graphs, dataset_ids)
    print("[w0.a]", json.dumps({k: v for k, v in out["w0a_support"]["pooled"].items()}, indent=None), "PASS" if out["w0a_support"]["pass"] else "FAIL", flush=True)
    if args.scorer_json:
        out["w0b_structure"] = read_structure(args.scorer_json, args.alpha_key)
        print("[w0.b]", json.dumps({k: v for k, v in out["w0b_structure"]["pooled"].items() if k != "values_sorted"}), out["w0b_structure"]["verdict"], flush=True)
    if not args.skip_checkpoints:
        out["w0c_cold_checkpoints"] = read_cold_checkpoints(graphs, dataset_ids, args)
        print("[w0.c]", json.dumps(out["w0c_cold_checkpoints"]["summary"], indent=1), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[w0] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
