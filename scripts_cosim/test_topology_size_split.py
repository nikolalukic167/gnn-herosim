"""Contract tests for the topology_transfer_v1 `topology_size` split (Phase 3).

The point: train/test must never mix topology sizes, or "does this transfer to a
LARGER topology" silently becomes "does this generalize across a random split" --
the exact ambiguity `canonical_parent` cannot resolve for this lineage.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "notebooks"))

from non_unique_lib.training_contract import (  # noqa: E402
    split_ids_by_topology_size,
    topology_size_of_dataset,
    topology_sizes_by_parent,
)


def _write_infra(root: Path, dataset_id: str, n_clients: int, n_servers: int) -> None:
    d = root / dataset_id
    d.mkdir(parents=True, exist_ok=True)
    nodes = {f"client_node{i}": {} for i in range(n_clients)}
    nodes.update({f"node{i}": {} for i in range(n_servers)})
    (d / "infrastructure.json").write_text(json.dumps({"network_maps": nodes}))


# ---------------------------------------------------------------- reading size on disk
def test_topology_size_of_dataset_counts_non_client_nodes(tmp_path):
    _write_infra(tmp_path, "ds_00000", n_clients=20, n_servers=28)
    assert topology_size_of_dataset("ds_00000", tmp_path) == 28


def test_topology_size_of_dataset_uses_the_canonical_parent(tmp_path):
    """An augmented graph id (@os0) must resolve to its parent's infrastructure.json."""
    _write_infra(tmp_path, "ds_00000", n_clients=20, n_servers=60)
    assert topology_size_of_dataset("ds_00000@os0", tmp_path) == 60


def test_topology_sizes_by_parent_reads_each_parent_once(tmp_path):
    _write_infra(tmp_path, "ds_00000", n_clients=20, n_servers=20)
    _write_infra(tmp_path, "ds_00001", n_clients=20, n_servers=80)
    sizes = topology_sizes_by_parent(
        ["ds_00000", "ds_00000@seq1", "ds_00001"], tmp_path
    )
    assert sizes == {"ds_00000": 20, "ds_00001": 80}


# --------------------------------------------------------------------------- the split
def _ids(sizes_by_id, n_per_id=1):
    """`n_per_id` graph instances per dataset id, ids like '<id>@os<k>' for k>0."""
    graphs, ids = [], []
    for dsid in sizes_by_id:
        for k in range(n_per_id):
            gid = dsid if k == 0 else f"{dsid}@os{k}"
            graphs.append(object())
            ids.append(gid)
    return graphs, ids


def _sizes_by_parent(train_ids, held_out_ids, train_size=20, held_out_size=60):
    return {**{i: train_size for i in train_ids}, **{i: held_out_size for i in held_out_ids}}


def test_train_and_test_never_mix_sizes():
    train_pool = [f"ds_{i:03d}" for i in range(10)]
    held_out_pool = [f"ds_{i:03d}" for i in range(10, 15)]
    graphs, ids = _ids(train_pool + held_out_pool)
    sizes = _sizes_by_parent(train_pool, held_out_pool)

    train_g, train_ids, val_g, val_ids, test_g, test_ids = split_ids_by_topology_size(
        graphs, ids, sizes, train_sizes=[20], held_out_sizes=[60]
    )
    assert set(train_ids) <= set(train_pool)
    assert set(val_ids) <= set(train_pool)
    assert set(test_ids) == set(held_out_pool)  # ALL held-out parents go to test
    assert not (set(train_ids) & set(val_ids))


def test_val_is_a_holdout_slice_of_train_sizes_only():
    train_pool = [f"ds_{i:03d}" for i in range(20)]
    held_out_pool = [f"ds_{i:03d}" for i in range(20, 25)]
    graphs, ids = _ids(train_pool + held_out_pool)
    sizes = _sizes_by_parent(train_pool, held_out_pool)

    *_ , val_g, val_ids, test_g, test_ids = split_ids_by_topology_size(
        graphs, ids, sizes, train_sizes=[20], held_out_sizes=[60], val_fraction_of_train=0.2
    )
    assert len(val_ids) == pytest.approx(4, abs=1)  # 20% of 20 train-size parents
    assert not (set(val_ids) & set(held_out_pool))


def test_multiple_augmented_instances_of_one_parent_stay_together():
    """@os/@seq copies of the same dataset must land in the same split as their parent."""
    graphs, ids = _ids(["ds_00000"], n_per_id=3)  # ds_00000, ds_00000@os1, ds_00000@os2
    other_train_graphs, other_train_ids = _ids(["ds_00001"])
    held_out_graphs, held_out_ids = _ids(["ds_00099"])
    graphs += other_train_graphs + held_out_graphs
    ids += other_train_ids + held_out_ids
    sizes = {"ds_00000": 20, "ds_00001": 20, "ds_00099": 60}
    train_g, train_ids, val_g, val_ids, test_g, test_ids = split_ids_by_topology_size(
        graphs, ids, sizes, train_sizes=[20], held_out_sizes=[60], val_fraction_of_train=0.5,
    )
    ds_00000_instances = {i for i in ids if i == "ds_00000" or i.startswith("ds_00000@")}
    landed = set(train_ids) | set(val_ids)
    assert ds_00000_instances <= set(train_ids) or ds_00000_instances <= set(val_ids)
    assert ds_00000_instances <= landed


def test_unregistered_size_raises():
    graphs, ids = _ids(["ds_00000", "ds_00001"])
    sizes = {"ds_00000": 20, "ds_00001": 50}  # 50 is not on the train/held-out ladder
    with pytest.raises(RuntimeError, match="outside the registered ladder"):
        split_ids_by_topology_size(graphs, ids, sizes, train_sizes=[20], held_out_sizes=[60])


def test_missing_size_raises_key_error():
    graphs, ids = _ids(["ds_00000"])
    with pytest.raises(KeyError):
        split_ids_by_topology_size(graphs, ids, {}, train_sizes=[20], held_out_sizes=[60])


def test_empty_train_pool_raises():
    graphs, ids = _ids(["ds_00000"])
    sizes = {"ds_00000": 60}
    with pytest.raises(RuntimeError, match="no parents at the train sizes"):
        split_ids_by_topology_size(graphs, ids, sizes, train_sizes=[20], held_out_sizes=[60])


def test_empty_held_out_pool_raises():
    graphs, ids = _ids(["ds_00000"])
    sizes = {"ds_00000": 20}
    with pytest.raises(RuntimeError, match="no parents at the held-out sizes"):
        split_ids_by_topology_size(graphs, ids, sizes, train_sizes=[20], held_out_sizes=[60])


def test_mismatched_graphs_and_ids_length_raises():
    with pytest.raises(RuntimeError, match="graphs"):
        split_ids_by_topology_size([object(), object()], ["a"], {"a": 20},
                                    train_sizes=[20], held_out_sizes=[60])
