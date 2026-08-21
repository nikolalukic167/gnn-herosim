"""The ablation harness must persist weights AND a contract, or the run is unusable later.

`topology_transfer_v1` ran a pre-registered 5-seed gate through
`scripts_cosim/gnn_necessity_ablation.py` and ended with zero deployable weights, because the
harness had no `torch.save` anywhere -- so the lineage's own stated next step (a live gate
across topology sizes) had nothing to run against. Saving weights alone would not have been
enough either: `executesimulation._read_checkpoint_sidecar` returns `{}` for a checkpoint with
no `.contract.json`, and every contract check downstream then silently adopts its default,
which is how `regime_b`'s five checkpoints ended up on disk and untrustworthy.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_cosim"))

from scripts_cosim.gnn_necessity_ablation import (  # noqa: E402
    AblationModel,
    save_arm_checkpoint,
)

DIMS = (3, 14, 2)


def _args(tmp_path, split_mode="topology_size"):
    return SimpleNamespace(
        seed=42,
        epochs=120,
        nondeterministic=False,
        cache=str(tmp_path / "cache"),
        corpus_root="simulation_data",
        split_mode=split_mode,
        train_sizes=[20, 28, 40],
        held_out_sizes=[60, 80],
    )


def _save(tmp_path, arm="gnn_base", cfg=None, split_mode="topology_size"):
    cfg = cfg if cfg is not None else dict(use_gin=True, use_node_edges=False)
    model = AblationModel(*DIMS, **cfg)
    out = tmp_path / "ckpts"
    path = save_arm_checkpoint(
        model=model,
        out_dir=out,
        arm_name=arm,
        arm_cfg=cfg,
        args=_args(tmp_path, split_mode),
        dims=DIMS,
        n_train=100,
        n_val=20,
        n_test=30,
    )
    contract = json.loads((out / f"{arm}_seed42.contract.json").read_text())
    return path, contract, model


def test_weights_round_trip_into_the_same_architecture(tmp_path):
    path, contract, model = _save(tmp_path)
    assert path.is_file()
    reloaded = AblationModel(*DIMS, **contract["arm_config"])
    reloaded.load_state_dict(torch.load(path, map_location="cpu"))
    for (ka, va), (kb, vb) in zip(
        model.state_dict().items(), reloaded.state_dict().items()
    ):
        assert ka == kb
        assert torch.equal(va, vb)


def test_contract_records_the_held_out_sizes(tmp_path):
    _, contract, _ = _save(tmp_path)
    # A checkpoint that does not say which sizes it never saw cannot be used to test
    # transfer to those sizes -- which is the entire hypothesis of this lineage.
    assert contract["split_mode"] == "topology_size"
    assert contract["train_sizes"] == [20, 28, 40]
    assert contract["held_out_sizes"] == [60, 80]


def test_contract_records_every_axis_the_weights_cannot(tmp_path):
    _, contract, _ = _save(tmp_path)
    for key in (
        "queue_feature_contract",
        "topology_feature_contract",
        "network_graph_contract",
        "platform_feature_dim",
        "arm_config",
        "seed",
        "model_class",
    ):
        assert key in contract, key
    assert contract["platform_feature_dim"] == DIMS[1]
    assert contract["model_class"] == "AblationModel"


def test_topology_blind_arm_records_network_contract_off(tmp_path):
    _, contract, _ = _save(tmp_path, arm="pointwise", cfg=dict(use_gin=False, use_node_edges=False))
    # gnn_base/gnn_node/pointwise never see network entities; recording "off" is what stops a
    # later reader assuming they did.
    assert contract["network_graph_contract"] == "off"


def test_non_topology_split_records_null_sizes(tmp_path):
    _, contract, _ = _save(tmp_path, split_mode="canonical_parent")
    assert contract["train_sizes"] is None
    assert contract["held_out_sizes"] is None


def test_arms_and_seeds_do_not_overwrite_each_other(tmp_path):
    _save(tmp_path, arm="gnn_base")
    _save(tmp_path, arm="pointwise", cfg=dict(use_gin=False, use_node_edges=False))
    names = sorted(p.name for p in (tmp_path / "ckpts").glob("*.pt"))
    assert names == ["gnn_base_seed42.pt", "pointwise_seed42.pt"]


@pytest.mark.parametrize("arm,cfg", [
    ("pointwise", dict(use_gin=False, use_node_edges=False)),
    ("gnn_base", dict(use_gin=True, use_node_edges=False)),
    ("gnn_node", dict(use_gin=True, use_node_edges=True)),
])
def test_every_gateable_arm_saves(tmp_path, arm, cfg):
    path, contract, _ = _save(tmp_path, arm=arm, cfg=cfg)
    assert path.is_file()
    assert contract["arm"] == arm
    assert contract["arm_config"] == cfg
