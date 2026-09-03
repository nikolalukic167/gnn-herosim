"""The MP-OFF train/serve guard added 2026-09-03.

`GNN_DISABLE_MESSAGE_PASSING` skips the GIN forward but still constructs the module, so
the choice is invisible in the weights and `load_state_dict(strict=True)` cannot catch a
mismatch. Until this fix it lived only in the environment: the live gates asserted it via
`run_provenance` (score_mp_ablation.py / score_link_mp_v1.py), the offline evaluator did
not. Measured cost on the route_b DAG corpus: 12.67% train regret matched vs 72.23%
mismatched — an error that reads as a decisive ablation result rather than a bug.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts_cosim.eval_route_b_stage2_arm import EvalError, load_arm  # noqa: E402


def _gnn_checkpoint(tmp_path: Path, *, disable_mp) -> Path:
    """A bare state_dict plus a sidecar — enough to reach the guard, not to score."""
    import torch

    ck = tmp_path / "arm.pt"
    torch.save({"encoder.weight": torch.zeros(2, 2)}, ck)
    sidecar = {
        "partial_state_edge_features": True,
        "partial_state_contract": "route_b_v1",
        "queue_feature_contract": "scale_invariant_v1",
    }
    if disable_mp is not None:
        sidecar["disable_message_passing"] = disable_mp
    ck.with_suffix(".contract.json").write_text(json.dumps(sidecar))
    return ck


def test_mp_off_checkpoint_scored_with_mp_on_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("GNN_DISABLE_MESSAGE_PASSING", raising=False)
    ck = _gnn_checkpoint(tmp_path, disable_mp=True)
    with pytest.raises(EvalError) as exc:
        load_arm(ck)
    assert "train/serve mismatch" in str(exc.value)


def test_mp_on_checkpoint_scored_with_mp_off_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("GNN_DISABLE_MESSAGE_PASSING", "1")
    ck = _gnn_checkpoint(tmp_path, disable_mp=False)
    with pytest.raises(EvalError) as exc:
        load_arm(ck)
    assert "train/serve mismatch" in str(exc.value)


@pytest.mark.parametrize("declared,env", [(True, "1"), (False, None)])
def test_matched_flag_passes_the_guard(tmp_path, monkeypatch, declared, env):
    """A matched pair must get past the guard (it then fails later, on the real load)."""
    if env is None:
        monkeypatch.delenv("GNN_DISABLE_MESSAGE_PASSING", raising=False)
    else:
        monkeypatch.setenv("GNN_DISABLE_MESSAGE_PASSING", env)
    ck = _gnn_checkpoint(tmp_path, disable_mp=declared)
    with pytest.raises(Exception) as exc:
        load_arm(ck)
    assert "train/serve mismatch" not in str(exc.value)


def test_sidecar_without_the_key_is_not_blocked(tmp_path, monkeypatch):
    """Pre-2026-09-03 sidecars carry no key; they must stay loadable, not be refused."""
    monkeypatch.delenv("GNN_DISABLE_MESSAGE_PASSING", raising=False)
    ck = _gnn_checkpoint(tmp_path, disable_mp=None)
    with pytest.raises(Exception) as exc:
        load_arm(ck)
    assert "train/serve mismatch" not in str(exc.value)
