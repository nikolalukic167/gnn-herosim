"""A task_dim=3 / platform_dim=14 checkpoint must not be served under a guessed layout.

Both `atomic21` and `dim22` are structurally valid for that shape -- they assign different
meanings to the same platform columns (dim22 normalizes the queue features, atomic21 does
not) -- so `load_state_dict` succeeds either way and the forward pass raises nothing. The
loader used to default an undeclared layout to `atomic21`, which is how the `prefixctl`
and `tempfix` live gates came to serve a different layout than every deployed-checkpoint
gate (whose sidecar declares `dim22`), leaving only a banner line to say so. Their
`run_provenance.env` recorded `INFERENCE_FEATURE_LAYOUT: None` while the deployed runs
recorded `dim22` -- a serving difference sitting underneath a result read as a
training-draw effect.

These tests pin the three cases apart: declared by sidecar, declared by env, declared by
neither (which must fail loud rather than pick one).
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.executesimulation import load_gnn_model  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SIDECAR_DECLARED = REPO / "models" / "near-rtt-v2-full-corpus-siv1-dim14-ce-only.pt"
SIDECAR_SILENT = (
    REPO / "models" / "near-rtt-v2-full-corpus-siv1-dim14-ce-only-prefixctl.pt"
)


@pytest.fixture(autouse=True)
def _clean_layout_env():
    """The loader both reads and WRITES this var, so leaking it across tests would make
    a later case pass on an earlier one's declaration."""
    saved = os.environ.pop("INFERENCE_FEATURE_LAYOUT", None)
    yield
    os.environ.pop("INFERENCE_FEATURE_LAYOUT", None)
    if saved is not None:
        os.environ["INFERENCE_FEATURE_LAYOUT"] = saved


@pytest.mark.skipif(not SIDECAR_SILENT.is_file(), reason="prefixctl checkpoint not present")
def test_undeclared_layout_on_the_ambiguous_shape_fails_loud():
    with pytest.raises(ValueError, match="declares neither"):
        load_gnn_model(SIDECAR_SILENT)


@pytest.mark.skipif(not SIDECAR_SILENT.is_file(), reason="prefixctl checkpoint not present")
def test_an_explicit_env_declaration_is_enough_to_serve():
    os.environ["INFERENCE_FEATURE_LAYOUT"] = "dim22"
    load_gnn_model(SIDECAR_SILENT)
    assert os.environ["INFERENCE_FEATURE_LAYOUT"] == "dim22"


@pytest.mark.skipif(
    not SIDECAR_DECLARED.is_file(), reason="deployed checkpoint not present"
)
def test_a_sidecar_declaration_is_enough_to_serve():
    """The deployed checkpoint's .contract.json says dim22, so it needs no env var --
    this is why the guard does not break the existing gate scripts, which deliberately
    do not export the variable."""
    load_gnn_model(SIDECAR_DECLARED)
    assert os.environ["INFERENCE_FEATURE_LAYOUT"] == "dim22"
