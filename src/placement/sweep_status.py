"""A co-sim dataset is usable only if its sweep finished: placement_metadata.json says `sweep_complete`, placements.jsonl has
every plan, and best.json is non-empty. A file merely existing is not that (2026-09-13: 81 of 95 datasets carried 3-30 % of their
plans). One definition for the corpus builder's resume skip, the cache loader and the corpus check."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple


def sweep_status(dataset_dir: Path) -> Tuple[bool, str]:
    meta_path = dataset_dir / "placement_metadata.json"
    rows_path = dataset_dir / "placements" / "placements.jsonl"
    best_path = dataset_dir / "best.json"
    if not meta_path.is_file():
        return False, "no placement_metadata.json"
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return False, "unreadable placement_metadata.json"
    if meta.get("sweep_complete") is not True:
        return False, "sweep_complete is not true"
    if not best_path.is_file() or best_path.stat().st_size == 0:
        return False, "best.json missing or empty"
    if not rows_path.is_file():
        return False, "no placements.jsonl"
    with rows_path.open("rb") as fh:
        rows = sum(1 for line in fh if line.strip())
    if rows != int(meta.get("num_placements", -1)):
        return False, f"{rows} rows for {meta.get('num_placements')} plans"
    return True, "complete"


def sweep_complete(dataset_dir: Path) -> bool:
    return sweep_status(dataset_dir)[0]
