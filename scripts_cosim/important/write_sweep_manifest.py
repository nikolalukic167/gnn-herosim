#!/usr/bin/env python3
"""Write a live-sweep provenance manifest.

Split out of the sweep runners so that sweeps executed cell-by-cell (datalab
SLURM arrays call `run_sealed_holdout_one.sh` per cell and never run the
whole-sweep runner) can still produce the same manifest. Without this, a sweep
directory arrives back from the cluster as bare `results/` with no record of the
physics, code commit, or checkpoint hashes it was produced under.

Fails loudly on a missing model/workload rather than writing a manifest that
claims provenance it cannot verify.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head(repo: Path) -> Optional[str]:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def model_entry(raw: str) -> Dict[str, str]:
    path = Path(raw)
    if not path.is_file():
        raise SystemExit(f"ERROR: model missing, refusing to write manifest: {path}")
    return {"path": raw, "md5": md5(path)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", type=Path, required=True)
    ap.add_argument("--kind", required=True)
    ap.add_argument("--note", default="")
    ap.add_argument("--physics", required=True)
    ap.add_argument("--workload", required=True)
    ap.add_argument("--seeds", required=True, help="comma-separated seeds")
    ap.add_argument("--configs", required=True, help="comma-separated config names")
    ap.add_argument("--gnn-model", required=True)
    ap.add_argument("--mlp-model", required=True)
    ap.add_argument(
        "--extra-json",
        default="{}",
        help="JSON object merged into the manifest (wins on key conflict)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing manifest.json",
    )
    args = ap.parse_args()

    if not args.sweep_dir.is_dir():
        raise SystemExit(f"ERROR: sweep dir does not exist: {args.sweep_dir}")

    out = args.sweep_dir / "manifest.json"
    if out.exists() and not args.force:
        raise SystemExit(
            f"ERROR: {out} already exists; pass --force to overwrite "
            "(refusing to silently replace an existing provenance record)"
        )

    try:
        extra: Any = json.loads(args.extra_json)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"ERROR: --extra-json is not valid JSON: {exc}") from exc
    if not isinstance(extra, dict):
        raise SystemExit(
            f"ERROR: --extra-json must be a JSON object, got {type(extra).__name__}"
        )

    workload = Path(args.workload)
    if not workload.is_file():
        raise SystemExit(f"ERROR: workload missing: {workload}")

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise SystemExit("ERROR: --seeds produced an empty seed list")
    configs = [c for c in args.configs.split(",") if c.strip()]
    if not configs:
        raise SystemExit("ERROR: --configs produced an empty config list")

    repo = Path(__file__).resolve().parents[2]
    manifest: Dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "kind": args.kind,
        "note": args.note,
        "code_head": git_head(repo),
        "warmth_physics": args.physics,
        "workload": args.workload,
        "seeds": seeds,
        "configs": configs,
        "gnn_model": model_entry(args.gnn_model),
        "mlp_model": model_entry(args.mlp_model),
        "policies": ["knative", "mlp", "gnn"],
    }
    manifest.update(extra)

    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {out}")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
