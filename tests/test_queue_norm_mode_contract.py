"""A checkpoint's cache was built with ONE queue-depth divisor; serving must use that one.

`queue_feature_contract` fixes the dim7/dim13 *formulas*; `queue_norm_mode` fixes the
*divisor* they take -- `scheduler_adaptive` (p90 over every platform), `adaptive_nonzero`
(p90 over the busy ones), `fixed`. Only the contract was ever enforced. The live graph
builder read `GNN_QUEUE_NORM_MODE` with a hard-coded `adaptive` default
(`GNNScheduler._build_inference_graph`) and never consulted the sidecar, so a cache built
under any other mode served under a divisor it was never fitted on, silently -- item (4)
of the 2026-09-06 audit row in docs/gates/gate-tools.md, open until this fix.

`adaptive` and `scheduler_adaptive` are the same arithmetic today, which is why no gate
has broken on it yet. That is luck, not a guarantee, and it is exactly the shape of
failure the layout and MP-flag guards already exist to stop.
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.executesimulation import apply_checkpoint_queue_norm_mode  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env():
    """The helper both reads and WRITES the var, so a leak would make a later case pass
    on an earlier one's declaration."""
    saved = os.environ.pop("GNN_QUEUE_NORM_MODE", None)
    yield
    os.environ.pop("GNN_QUEUE_NORM_MODE", None)
    if saved is not None:
        os.environ["GNN_QUEUE_NORM_MODE"] = saved


def _checkpoint_with_sidecar(tmp_path: Path, payload: dict) -> Path:
    ckpt = tmp_path / "arm.pt"
    ckpt.write_bytes(b"")  # never loaded: the helper reads the sidecar only
    ckpt.with_suffix(".contract.json").write_text(json.dumps(payload))
    return ckpt


def test_silent_environment_adopts_the_cached_mode(tmp_path):
    ckpt = _checkpoint_with_sidecar(tmp_path, {"queue_norm_mode": "adaptive_nonzero"})
    apply_checkpoint_queue_norm_mode(ckpt, "test arm")
    assert os.environ["GNN_QUEUE_NORM_MODE"] == "adaptive_nonzero"


def test_a_conflicting_declaration_fails_loud(tmp_path):
    ckpt = _checkpoint_with_sidecar(tmp_path, {"queue_norm_mode": "adaptive_nonzero"})
    os.environ["GNN_QUEUE_NORM_MODE"] = "scheduler_adaptive"
    with pytest.raises(ValueError, match="queue_norm_mode"):
        apply_checkpoint_queue_norm_mode(ckpt, "test arm")


def test_a_matching_declaration_is_kept(tmp_path):
    ckpt = _checkpoint_with_sidecar(tmp_path, {"queue_norm_mode": "scheduler_adaptive"})
    os.environ["GNN_QUEUE_NORM_MODE"] = "scheduler_adaptive"
    apply_checkpoint_queue_norm_mode(ckpt, "test arm")
    assert os.environ["GNN_QUEUE_NORM_MODE"] == "scheduler_adaptive"


def test_a_sidecar_without_the_key_leaves_the_environment_alone(tmp_path):
    """Pre-2026-08 sidecars carry no queue_norm_mode; they must stay servable."""
    ckpt = _checkpoint_with_sidecar(tmp_path, {"queue_feature_contract": "legacy_v0"})
    apply_checkpoint_queue_norm_mode(ckpt, "test arm")
    assert "GNN_QUEUE_NORM_MODE" not in os.environ
