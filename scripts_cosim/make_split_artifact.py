"""Generate the B6 shared train/val/test split artifact from a batch cache.

Every arm of a paired comparison (route_b stage 2: A1 GNN vs A2/A3 MLP) must load
the SAME parent-level split. Run this once per corpus, commit the JSON, and point
both trainers at it (GNN: NEAR_RTT_SPLIT_ARTIFACT; MLP: --split-artifact).

Usage:
    PIPENV_IGNORE_VIRTUALENVS=1 PYTHONPATH=. pipenv run python3 \
        scripts_cosim/make_split_artifact.py \
        --cache-dir simulation_data/graphs_cache_route_b_pilot_s_dag \
        --output experiments/route_b_stage2_split_v1.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = REPO_ROOT / "src" / "notebooks"
for p in (str(REPO_ROOT), str(NOTEBOOKS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from non_unique_lib.training_contract import (  # noqa: E402
    SPLIT_NAMES,
    load_split_artifact,
    write_split_artifact,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--val-fraction-of-holdout", type=float, default=0.5)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(
            f"Refusing to overwrite {args.output} — a split artifact is frozen once "
            f"runs depend on it. Delete it explicitly if regeneration is intended."
        )

    payload, sha256 = write_split_artifact(
        args.cache_dir,
        args.output,
        test_size=args.test_size,
        val_fraction_of_holdout=args.val_fraction_of_holdout,
        random_state=args.random_state,
    )
    # Round-trip through the loader so a freshly generated artifact is proven loadable.
    reloaded, reloaded_sha = load_split_artifact(args.output)
    if reloaded_sha != sha256 or reloaded != payload:
        raise SystemExit(f"Round-trip mismatch on {args.output}")

    sizes = " / ".join(f"{name}={len(payload[name])}" for name in SPLIT_NAMES)
    print(f"[split-artifact] wrote {args.output}")
    print(f"[split-artifact] parents: {payload['n_parents']} ({sizes})")
    print(f"[split-artifact] sha256: {sha256}")


if __name__ == "__main__":
    main()
