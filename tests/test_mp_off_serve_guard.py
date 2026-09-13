"""The live-serving loader had the same MP-OFF sidecar hole as the offline evaluator.

`GNN_DISABLE_MESSAGE_PASSING` skips the GIN forward but still constructs the module
(gnn_model.py:295 reads it once at __init__), so the choice is invisible in the weights.
`executesimulation.load_gnn_model` already refused a `mp_node_edges` / `mp_dag_edges`
mismatch between sidecar and serving env (both weight-invisible, same class of risk) --
but `checkpoint_mp_config`'s key whitelist never included `disable_message_passing`, so
`mp_cfg.get("disable_message_passing")` was silently `None` for every checkpoint and the
loader had nothing to check a serving-env mismatch against. This is the identical bug
`eval_route_b_stage2_arm.py` was found and fixed for on 2026-09-03 (a 5.7x train-regret
error, 12.67% matched vs 72.23% mismatched) -- but that fix only covered the offline
route_b evaluator, not this loader, which every live gate goes through.

These tests build a real, loadable TaskPlacementGNN (task_dim=2 / platform_dim=14, the
current constants, so no inference-layout ambiguity) to exercise `load_gnn_model` itself
rather than a mocked sidecar.
"""

import json
import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.executesimulation import load_gnn_model  # noqa: E402
from src.policy.gnn.gnn_model import TaskPlacementGNN  # noqa: E402

DISABLE_MP_ENV = "GNN_DISABLE_MESSAGE_PASSING"
QUEUE_CONTRACT_ENV = "QUEUE_FEATURE_CONTRACT"


def _checkpoint(tmp_path: Path, *, disable_mp) -> Path:
    torch.manual_seed(0)
    model = TaskPlacementGNN(
        task_feature_dim=2, platform_feature_dim=14, embedding_dim=64,
        hidden_dim=64, num_layers=3, edge_dim=5,
    )
    ck = tmp_path / "arm.pt"
    torch.save(model.state_dict(), ck)
    sidecar = {"queue_feature_contract": "legacy_v0"}
    if disable_mp is not None:
        sidecar["disable_message_passing"] = disable_mp
    ck.with_suffix(".contract.json").write_text(json.dumps(sidecar))
    return ck


@pytest.fixture(autouse=True)
def _clean_env():
    # The sidecar declares queue_feature_contract=legacy_v0; pin serving to match so an
    # earlier test's leaked QUEUE_FEATURE_CONTRACT=scale_invariant_v1 (module-level
    # os.environ state persists across test files in one process) does not turn this
    # into an unrelated contract-mismatch failure.
    keys = (DISABLE_MP_ENV, "INFERENCE_FEATURE_LAYOUT", QUEUE_CONTRACT_ENV)
    saved = {k: os.environ.get(k) for k in keys}
    os.environ[QUEUE_CONTRACT_ENV] = "legacy_v0"
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def test_mp_off_checkpoint_served_mp_on_raises(tmp_path):
    os.environ.pop(DISABLE_MP_ENV, None)
    ck = _checkpoint(tmp_path, disable_mp=True)
    with pytest.raises(ValueError, match="disable_message_passing"):
        load_gnn_model(ck)


def test_mp_on_checkpoint_served_mp_off_raises(tmp_path):
    os.environ[DISABLE_MP_ENV] = "1"
    ck = _checkpoint(tmp_path, disable_mp=False)
    with pytest.raises(ValueError, match="disable_message_passing"):
        load_gnn_model(ck)


@pytest.mark.parametrize("declared,env", [(True, "1"), (False, None), (None, None)])
def test_matched_flag_loads_cleanly(tmp_path, declared, env):
    """Matched pairs (including a sidecar-less / pre-2026-09-03 checkpoint) must load."""
    if env is None:
        os.environ.pop(DISABLE_MP_ENV, None)
    else:
        os.environ[DISABLE_MP_ENV] = env
    ck = _checkpoint(tmp_path, disable_mp=declared)
    load_gnn_model(ck)
