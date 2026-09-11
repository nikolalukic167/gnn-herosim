#!/usr/bin/env python3
"""peer_affinity_v1 -- find datasets that cannot carry a label at the tightest alpha rung.

`prepare_graphs_cache.py --dag-partial-state` raises `no feasible sweep rows at alpha=2.0`
for a dataset whose full sweep has no plan under the alpha_max caps, and the whole build
dies with it (T1b lost four of 350 that way and set them aside by hand). This scan applies
the same rule (caps = alpha x max candidate demand per node, demand = demand_scale x
memoryRequirements, over the sweep's own rows) BEFORE the build and, with --set-aside-dir,
moves the offenders out of the corpus so the build runs once. The ladder is monotone in
alpha (same peaks, caps scale), so feasibility at the tightest rung implies every rung.

Usage:
  python3 scripts_cosim/peer_affinity_alpha_prescan.py <corpus_dir>... [--alpha 2.0]
      [--set-aside-dir simulation_data/<corpus>_setaside]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts_cosim.score_route_b_contention import Dataset, load_task_types  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("corpora", nargs="+", type=Path)
    ap.add_argument("--alpha", type=float, default=2.0, help="tightest rung of DAG_ALPHA_LADDER")
    ap.add_argument("--task-types", type=Path, default=REPO_ROOT / "data/nofs-ids/task-types.json")
    ap.add_argument("--set-aside-dir", type=Path, default=None,
                    help="move infeasible datasets here (one subdir per corpus); default: report only")
    args = ap.parse_args()
    tt = load_task_types(args.task_types)
    total = infeasible = 0
    offenders = []
    for corpus in args.corpora:
        dirs = sorted(p for p in corpus.glob("ds_*") if p.is_dir())
        if not dirs:
            raise SystemExit(f"FAIL LOUD: no ds_* under {corpus}")
        for ds_dir in dirs:
            total += 1
            ds = Dataset(ds_dir, tt, "rtt")
            caps = ds.node_caps(args.alpha)
            if any(ds.plan_feasible(plan, caps) for plan, _v in ds.rows):
                continue
            infeasible += 1
            offenders.append(ds_dir)
            print(f"[INFEASIBLE alpha={args.alpha}] {ds_dir} ({len(ds.rows)} sweep rows)")
    print(f"[prescan] {total} datasets, {infeasible} infeasible at alpha={args.alpha}")
    if args.set_aside_dir is not None:
        for ds_dir in offenders:
            dst = args.set_aside_dir / ds_dir.parent.name / ds_dir.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(ds_dir), str(dst))
            print(f"[SET ASIDE] {ds_dir} -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
