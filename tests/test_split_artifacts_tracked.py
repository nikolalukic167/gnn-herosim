"""Every split a training config or checkpoint sidecar names is a tracked file with the recorded sha (audit 2026-10-12, M2).

The accel g4 and slate_fix_v1 sidecars named experiments/accel_replica_v1_split.json (sha256 f5bc4f51...) while no branch tracked it.
Sidecars live under the gitignored models/, so the sidecar check runs on HEROSIM_SIDECAR_DIRS (colon-separated) when it is set.
"""
import glob
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
REGISTERED = {"experiments/accel_replica_v1_split.json": "f5bc4f51106d71a1830febec8c06dca66121a81336acd5662cf24b7997cf7e53"}
# untracked splits other configs name, found by the audit; this list may only shrink
KNOWN_UNTRACKED = {"experiments/joint_burst_v2_split.json", "experiments/r1_attribution_v1_split.json",
                   "experiments/r1_attribution_v1_smoke_split.json", "experiments/scale_160_v1_split.json"}


def _tracked() -> set:
    return set(subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split())


def _sha(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def test_registered_splits_are_tracked_with_their_sha():
    tracked = _tracked()
    for rel, sha in REGISTERED.items():
        assert rel in tracked, f"{rel} is not tracked"
        assert _sha(rel) == sha


def test_config_splits_are_tracked():
    tracked = _tracked()
    untracked = set()
    for f in sorted(glob.glob(str(ROOT / "experiments/*.yaml"))):
        rel = ((yaml.safe_load(open(f)) or {}).get("env") or {}).get("NEAR_RTT_SPLIT_ARTIFACT")
        if rel and rel not in tracked:
            untracked.add(rel)
    assert untracked <= KNOWN_UNTRACKED, f"configs name untracked splits: {sorted(untracked - KNOWN_UNTRACKED)}"


@pytest.mark.skipif(not os.environ.get("HEROSIM_SIDECAR_DIRS"), reason="set HEROSIM_SIDECAR_DIRS to the checkpoint directories")
def test_sidecar_splits_resolve_to_tracked_files():
    tracked = _tracked()
    seen = 0
    for d in os.environ["HEROSIM_SIDECAR_DIRS"].split(":"):
        for f in sorted(glob.glob(os.path.join(d, "*.contract.json"))):
            sa = json.load(open(f)).get("split_artifact")
            if not sa:
                continue
            seen += 1
            assert sa["path"] in tracked, f"{f}: {sa['path']} is not tracked"
            assert _sha(sa["path"]) == sa["sha256"], f"{f}: {sa['path']} sha differs from the recorded {sa['sha256'][:8]}"
    assert seen, "no sidecar with a split_artifact found"
