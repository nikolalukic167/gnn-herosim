#!/usr/bin/env python3
"""Build strategic merge weights: warmth + sparse + contention_v2 (no v3/skew).

Oversamples coupled cells (greedy regret > 1%); upweights contention_v2 base vs warmth/sparse.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts_cosim.separability_diagnostic import analyze_dataset  # noqa: E402

COUPLED_THRESH = 0.01  # 1% greedy regret

CORPORA = [
    ("gnn_datasets_4tasks_1060_warmth_v2", "warmth"),
    ("gnn_datasets_4tasks_sparse_warmth_v2", "sparse"),
    ("gnn_datasets_4tasks_contention_v2", "contention"),
]

# repeat counts (not equal dump)
WEIGHTS = {
    "contention": {"separable": 2, "collide": 3, "coupled": 5},
    "warmth": {"separable": 1, "collide": 2, "coupled": 4},
    "sparse": {"separable": 1, "collide": 2, "coupled": 4},
}


def bucket(corpus_tag: str, r: dict) -> str:
    regret = r.get("m1_regret_rel")
    coupled = regret is not None and regret > COUPLED_THRESH
    collide = bool(r.get("opt_has_collision"))
    if coupled:
        return "coupled"
    if collide:
        return "collide"
    return "separable"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-root", type=Path, default=_REPO / "simulation_data")
    ap.add_argument(
        "--out",
        type=Path,
        default=_REPO / "simulation_data" / "strategic_merge_weights.json",
    )
    ap.add_argument("--limit", type=int, default=0, help="Per-corpus limit (debug)")
    args = ap.parse_args()

    weights: dict[str, int] = {}
    records: list[dict] = []
    stats = {tag: {"separable": 0, "collide": 0, "coupled": 0, "skip": 0} for _, tag in CORPORA}

    for dir_name, tag in CORPORA:
        base = args.sim_root / dir_name
        if not base.is_dir():
            raise FileNotFoundError(f"Missing corpus: {base}")
        ds_dirs = sorted(d for d in base.glob("ds_*") if d.is_dir())
        if args.limit:
            ds_dirs = ds_dirs[: args.limit]

        for ds_dir in ds_dirs:
            r = analyze_dataset(ds_dir)
            if r is None:
                stats[tag]["skip"] += 1
                continue
            b = bucket(tag, r)
            stats[tag][b] += 1
            w = WEIGHTS[tag][b]
            ds_key = f"{dir_name}/{ds_dir.name}"
            weights[ds_key] = w
            records.append(
                {
                    "dataset_id": ds_key,
                    "corpus": tag,
                    "bucket": b,
                    "repeat": w,
                    "greedy_regret_rel": r.get("m1_regret_rel"),
                    "opt_collision": bool(r.get("opt_has_collision")),
                    "n_combos": r.get("n_combos"),
                }
            )

    total_base = len(records)
    total_graphs = sum(weights.values())
    payload = {
        "version": 1,
        "corpora": [c[0] for c in CORPORA],
        "excluded": ["gnn_datasets_4tasks_contention_v3", "gnn_datasets_4tasks_skew_warmth_v2"],
        "coupled_threshold": COUPLED_THRESH,
        "weight_table": WEIGHTS,
        "stats": stats,
        "total_datasets": total_base,
        "total_graph_slots": total_graphs,
        "weights": weights,
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))

    print(f"Wrote {args.out}")
    print(f"  base datasets: {total_base}  graph slots after oversample: {total_graphs}")
    for tag in ("warmth", "sparse", "contention"):
        s = stats[tag]
        n = s["separable"] + s["collide"] + s["coupled"]
        print(
            f"  {tag}: n={n} separable={s['separable']} collide={s['collide']} "
            f"coupled={s['coupled']} skip={s['skip']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
