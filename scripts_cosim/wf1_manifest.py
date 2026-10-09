"""r1_attribution_v1: the warm-corpus manifests of one output directory. A build run with --manifest-per-source writes
`warm_manifest.<source_tag>.jsonl`, one file per cell and one writer per file, so concurrent build tasks (on a network filesystem) never
append to the same file. Readers take the union of `warm_manifest.jsonl` and every `warm_manifest.<tag>.jsonl`."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def manifest_paths(directory: Path) -> List[Path]:
    paths = sorted(directory.glob("warm_manifest.jsonl")) + sorted(directory.glob("warm_manifest.*.jsonl"))
    return paths


def read_manifests(directory: Path, strict: bool = True) -> Tuple[List[Dict[str, Any]], int]:
    """(entries, skipped): the union of the directory's manifests. strict raises on a line that does not parse; otherwise such lines are
    skipped and counted."""
    entries: List[Dict[str, Any]] = []
    skipped = 0
    for p in manifest_paths(directory):
        for n, line in enumerate(open(p), 1):
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                if strict:
                    raise ValueError(f"FAIL LOUD: {p}:{n} is not valid JSON (a garbled manifest line): {line[:120]!r}")
                skipped += 1
    return entries, skipped
