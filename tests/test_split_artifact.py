"""B6: the shared split artifact.

§3 of docs/lineages/route_b_v1/stage2-preregistration.md requires a "draw" to vary initialisation and
batch order ONLY. Before B6, the MLP drew its split from --random-state (so a seed
sweep moved the split) and the GNN drew its own with a hardcoded 42 — two different
splits, and a paired test confounded by both. These tests pin the artifact that
replaces the draws: its determinism, its validation, the parent-identity derivation
both trainers share, and the fail-loud paths in each consumer.

The GNN trainer trains at import time, so its consumer paths are exercised as
fast-fail subprocesses (the guards fire at the split site, before any training);
the success path is covered by the MLP end-to-end case plus the artifact-vs-cache
partition test on the primitives the trainer's inline code composes.
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = REPO_ROOT / "src" / "notebooks"
for _p in (str(REPO_ROOT), str(NOTEBOOKS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from non_unique_lib.training_contract import (  # noqa: E402
    SPLIT_NAMES,
    assert_split_artifact_covers,
    canonical_parent_id,
    load_split_artifact,
    write_split_artifact,
)
from src.policy.tabular.reduced_features import parent_dataset_id  # noqa: E402

SMOKE_CACHE = REPO_ROOT / "simulation_data/graphs_cache_route_b_smoke_s_dag"
BATCH_CACHE = REPO_ROOT / "simulation_data/graphs_cache_regime_b_oracle_split_cosim"

needs_smoke = pytest.mark.skipif(
    not SMOKE_CACHE.is_dir(), reason=f"cache not present at {SMOKE_CACHE}"
)
needs_batch = pytest.mark.skipif(
    not BATCH_CACHE.is_dir(), reason=f"cache not present at {BATCH_CACHE}"
)


def _fake_cache(tmp_path: Path, dataset_ids, parent_ids=None) -> Path:
    cache = tmp_path / "cache"
    cache.mkdir()
    with (cache / "dataset_ids.pkl").open("wb") as f:
        pickle.dump(list(dataset_ids), f)
    meta = {"version": "test"}
    if parent_ids is not None:
        meta["parent_dataset_ids"] = list(parent_ids)
    (cache / "metadata.json").write_text(json.dumps(meta))
    return cache


# --------------------------------------------------------------------------------------
# Producer + loader
# --------------------------------------------------------------------------------------


def test_producer_is_deterministic_byte_identical(tmp_path):
    cache = _fake_cache(tmp_path, [f"corpus/ds_{i:05d}" for i in range(20)])
    _, sha_a = write_split_artifact(cache, tmp_path / "a.json", random_state=42)
    _, sha_b = write_split_artifact(cache, tmp_path / "b.json", random_state=42)
    assert sha_a == sha_b
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()


def test_round_trip_preserves_payload_and_hash(tmp_path):
    cache = _fake_cache(tmp_path, [f"corpus/ds_{i:05d}" for i in range(12)])
    payload, sha = write_split_artifact(cache, tmp_path / "split.json")
    reloaded, reloaded_sha = load_split_artifact(tmp_path / "split.json")
    assert reloaded == payload
    assert reloaded_sha == sha
    all_parents = set().union(*(set(reloaded[n]) for n in SPLIT_NAMES))
    assert len(all_parents) == reloaded["n_parents"] == 12


def test_augmented_instances_collapse_to_one_parent(tmp_path):
    """@os/@seq instances of one parent must land in one split, never straddle two."""
    ids = []
    for i in range(10):
        base = f"corpus/ds_{i:05d}"
        ids.extend([base, f"{base}@os1", f"{base}@seq2"])
    cache = _fake_cache(tmp_path, ids)
    payload, _ = write_split_artifact(cache, tmp_path / "split.json")
    assert payload["n_parents"] == 10
    for name in SPLIT_NAMES:
        assert all("@" not in p for p in payload[name])


def test_seed_moves_the_split(tmp_path):
    cache = _fake_cache(tmp_path, [f"corpus/ds_{i:05d}" for i in range(30)])
    a, _ = write_split_artifact(cache, tmp_path / "a.json", random_state=42)
    b, _ = write_split_artifact(cache, tmp_path / "b.json", random_state=43)
    assert a["train"] != b["train"]


def test_loader_rejects_wrong_schema(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema": "something_else", "train": ["a"]}))
    with pytest.raises(RuntimeError, match="split_artifact_v1"):
        load_split_artifact(path)


def test_loader_rejects_parent_overlap(tmp_path):
    path = tmp_path / "leak.json"
    path.write_text(
        json.dumps(
            {
                "schema": "split_artifact_v1",
                "train": ["p1", "p2"],
                "val": ["p2"],
                "test": ["p3"],
            }
        )
    )
    with pytest.raises(RuntimeError, match="overlap"):
        load_split_artifact(path)


def test_loader_rejects_empty_split(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(
        json.dumps(
            {"schema": "split_artifact_v1", "train": ["p1"], "val": [], "test": ["p2"]}
        )
    )
    with pytest.raises(RuntimeError, match="non-empty"):
        load_split_artifact(path)


def test_coverage_assert_requires_exact_parent_set(tmp_path):
    cache = _fake_cache(tmp_path, [f"corpus/ds_{i:05d}" for i in range(10)])
    payload, _ = write_split_artifact(cache, tmp_path / "split.json")
    parents = [f"corpus/ds_{i:05d}" for i in range(10)]
    assert_split_artifact_covers(payload, parents)  # exact match passes
    # augmented instance ids canonicalize before comparison
    assert_split_artifact_covers(payload, [f"{p}@os1" for p in parents])
    with pytest.raises(RuntimeError, match="in data but not artifact"):
        assert_split_artifact_covers(payload, parents + ["corpus/ds_99999"])
    with pytest.raises(RuntimeError, match="in artifact but not data"):
        assert_split_artifact_covers(payload, parents[:-1])


# --------------------------------------------------------------------------------------
# The parent identity both trainers share
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "instance_id",
    [
        "corpus/ds_00000",
        "corpus/ds_00000@os3",
        "corpus/ds_00000@seq12",
        "corpus/ds_00000@os1@seq2",
    ],
)
def test_parent_derivation_agrees_across_trainers(instance_id):
    """The whole scheme keys on canonical parents, derived by TWO copies of the
    stripping logic — training_contract (GNN side) and reduced_features (MLP side).
    If they ever diverge, one artifact quietly means two different splits."""
    assert canonical_parent_id(instance_id) == parent_dataset_id(instance_id)


# --------------------------------------------------------------------------------------
# Partition semantics against a real cache (the primitives the GNN's inline code uses)
# --------------------------------------------------------------------------------------


@needs_smoke
def test_artifact_partitions_a_real_cache_completely_and_disjointly(tmp_path):
    payload, _ = write_split_artifact(SMOKE_CACHE, tmp_path / "split.json")
    with (SMOKE_CACHE / "dataset_ids.pkl").open("rb") as f:
        dataset_ids = pickle.load(f)
    assignment = {p: n for n in SPLIT_NAMES for p in payload[n]}
    buckets = {n: [] for n in SPLIT_NAMES}
    for dsid in dataset_ids:
        buckets[assignment[canonical_parent_id(dsid)]].append(dsid)
    assert sum(len(b) for b in buckets.values()) == len(dataset_ids)
    assert all(buckets[n] for n in SPLIT_NAMES)


# --------------------------------------------------------------------------------------
# Consumers, via subprocess (fail-loud paths must exit nonzero BEFORE training)
# --------------------------------------------------------------------------------------


def _run(args, env_extra=None, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
            "PIPENV_IGNORE_VIRTUALENVS": "1",
            "PYTHONPATH": str(REPO_ROOT),
            "WANDB_MODE": "disabled",
        }
    )
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, *args], cwd=str(cwd), env=env, capture_output=True, text=True
    )


@needs_batch
def test_mlp_trains_under_a_pinned_artifact_and_stamps_it(tmp_path):
    import torch

    _, sha = write_split_artifact(BATCH_CACHE, tmp_path / "split.json")
    out = tmp_path / "mlp.pt"
    result = _run(
        [
            "src/policy/tabular/train_mlp_dim22_from_batch.py",
            "--cache-dir", str(BATCH_CACHE),
            "--output", str(out),
            "--epochs", "1",
            "--random-state", "4242",
            "--split-artifact", str(tmp_path / "split.json"),
        ]
    )
    assert result.returncode == 0, (
        f"trainer exited {result.returncode}\n--- stdout ---\n{result.stdout[-3000:]}"
        f"\n--- stderr ---\n{result.stderr[-3000:]}"
    )
    meta = json.loads((tmp_path / "mlp.pt.meta.json").read_text())
    assert meta["split_artifact"] == {"path": str(tmp_path / "split.json"), "sha256": sha}
    assert "init/batch order only" in meta["split_note"]
    assert torch.load(out, map_location="cpu", weights_only=False)["torch_seeded"] is True


@needs_batch
def test_mlp_refuses_an_artifact_from_a_different_cache(tmp_path):
    cache = _fake_cache(tmp_path, [f"other_corpus/ds_{i:05d}" for i in range(10)])
    write_split_artifact(cache, tmp_path / "wrong.json")
    result = _run(
        [
            "src/policy/tabular/train_mlp_dim22_from_batch.py",
            "--cache-dir", str(BATCH_CACHE),
            "--output", str(tmp_path / "mlp.pt"),
            "--epochs", "1",
            "--split-artifact", str(tmp_path / "wrong.json"),
        ]
    )
    assert result.returncode != 0
    assert "does not match this corpus" in result.stderr


@needs_batch
def test_mlp_refuses_size_knobs_alongside_an_artifact(tmp_path):
    write_split_artifact(BATCH_CACHE, tmp_path / "split.json")
    result = _run(
        [
            "src/policy/tabular/train_mlp_dim22_from_batch.py",
            "--cache-dir", str(BATCH_CACHE),
            "--output", str(tmp_path / "mlp.pt"),
            "--epochs", "1",
            "--split-artifact", str(tmp_path / "split.json"),
            "--test-size", "0.2",
        ]
    )
    assert result.returncode != 0
    assert "no effect" in result.stderr


@needs_smoke
def test_gnn_refuses_train_all_alongside_an_artifact(tmp_path):
    """The trainer's train=val=test bypasses would silently contradict the pinned
    split the sidecar then claims; both are refused at the split site."""
    write_split_artifact(SMOKE_CACHE, tmp_path / "split.json")
    result = _run(
        ["src/notebooks/train_near_rtt.py", "--cache-dir", str(SMOKE_CACHE)],
        env_extra={
            "NEAR_RTT_SPLIT_ARTIFACT": str(tmp_path / "split.json"),
            "NEAR_RTT_TRAIN_ALL": "1",
        },
    )
    assert result.returncode != 0
    assert "NEAR_RTT_TRAIN_ALL=1 would bypass" in result.stderr


@needs_smoke
def test_gnn_refuses_an_artifact_from_a_different_cache(tmp_path):
    cache = _fake_cache(tmp_path, [f"other_corpus/ds_{i:05d}" for i in range(10)])
    write_split_artifact(cache, tmp_path / "wrong.json")
    result = _run(
        ["src/notebooks/train_near_rtt.py", "--cache-dir", str(SMOKE_CACHE)],
        env_extra={"NEAR_RTT_SPLIT_ARTIFACT": str(tmp_path / "wrong.json")},
    )
    assert result.returncode != 0
    assert "does not match this corpus" in result.stderr
