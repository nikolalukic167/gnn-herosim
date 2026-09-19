"""dim47crk — the MLP partial-state layout under partial_state_v3.

`pointwise_baseline_v1` needs the pointwise model class to read the SAME representation the
graph arms read, and until 2026-09-19 `_batch_edge_feature_dims` refused `partial_state_v3`
outright — on the stated grounds that "the MLP is not an arm of that lineage", which is an
organisational reason and not a physical one. `partial_state_columns` has been fully
contract-aware the whole time; only the MLP's total width was wired in as a constant.

This file exists because widening a feature layout is the thing CLAUDE.md names as silently
breaking a checkpoint. What it pins:

  * v1/v2 are BYTE-IDENTICAL to before — same width, same columns, same layout name;
  * v3 gets a SEPARATE NAME (`dim47crk`), never a widened `dim63crk`. A checkpoint declares
    `inference_feature_layout`, so a distinct name is the only thing preventing a v2
    checkpoint from being served v3 features; widening in place would have removed that guard;
  * the two widths are exactly dim25cr + the contract's own block width, so they cannot drift.

Run: pipenv run python3 -m pytest tests/test_dim47crk_layout.py -q
"""
from __future__ import annotations

import pytest

from src.policy.tabular import feature_builder as fb
from src.policy.tabular.reduced_features import (
    DIM25CR_FEATURE_DIM,
    DIM47CRK_FEATURE_DIM,
    DIM63CRK_FEATURE_DIM,
    PARTIAL_STATE_CONTRACT_ENV,
    PartialStateContractMismatchError,
    _batch_edge_feature_dims,
    partial_state_feature_dim,
    partial_state_mlp_layout,
    require_matching_partial_state_contract,
)

V1, V2, V3 = "partial_state_v1", "partial_state_v2", "partial_state_v3"


def test_v1_and_v2_are_unchanged():
    """The regression that would matter: every MLP checkpoint on disk is a v1/v2 one."""
    for c in (V1, V2):
        dim, names, layout = partial_state_mlp_layout(c)
        assert (dim, layout) == (63, "dim63crk")
        assert dim == DIM63CRK_FEATURE_DIM == len(names)


def test_v3_gets_its_own_layout_name_not_a_widened_dim63crk():
    dim, names, layout = partial_state_mlp_layout(V3)
    assert (dim, layout) == (47, "dim47crk")
    assert dim == DIM47CRK_FEATURE_DIM == len(names)
    # The names must differ, or a checkpoint's declared layout stops being a guard.
    assert layout != partial_state_mlp_layout(V2)[2]


def test_both_widths_are_dim25cr_plus_that_contracts_own_block():
    """Derived, not hardcoded — the two layouts cannot drift apart from the block builder."""
    for c in (V1, V2, V3):
        dim, _names, _layout = partial_state_mlp_layout(c)
        assert dim == DIM25CR_FEATURE_DIM + partial_state_feature_dim(c)


def test_batch_edge_feature_dims_follows_the_env_contract(monkeypatch):
    monkeypatch.setenv(PARTIAL_STATE_CONTRACT_ENV, V3)
    assert _batch_edge_feature_dims(14, candidate_relative=True, partial_state=True)[2] \
        == "dim47crk"
    monkeypatch.setenv(PARTIAL_STATE_CONTRACT_ENV, V2)
    assert _batch_edge_feature_dims(14, candidate_relative=True, partial_state=True)[2] \
        == "dim63crk"


def test_the_old_refusals_that_are_still_refusals(monkeypatch):
    """Widening the contract must not have loosened anything else."""
    monkeypatch.setenv(PARTIAL_STATE_CONTRACT_ENV, V3)
    with pytest.raises(ValueError, match="requires candidate_relative"):
        _batch_edge_feature_dims(14, candidate_relative=False, partial_state=True)
    with pytest.raises(ValueError, match="platform width"):
        _batch_edge_feature_dims(16, candidate_relative=True, partial_state=True)


def test_feature_builder_agrees_with_reduced_features_on_both_names():
    for layout, want in (("dim63crk", DIM63CRK_FEATURE_DIM),
                         ("dim47crk", DIM47CRK_FEATURE_DIM)):
        assert fb._uses_partial_state_layout(layout)
        assert fb._uses_candidate_relative_layout(layout)
        assert fb._uses_dim22_layout(layout)
        assert fb._expected_feature_dim_for_layout(layout) == want


def test_serving_still_refuses_a_cross_contract_checkpoint():
    """The point of the separate name: v2 weights must never meet v3 features."""
    with pytest.raises(PartialStateContractMismatchError):
        require_matching_partial_state_contract(V2, V3, model_label="mlp")
    with pytest.raises(PartialStateContractMismatchError):
        require_matching_partial_state_contract(V3, V2, model_label="mlp")
    # and a contract-less checkpoint is still not evidence
    with pytest.raises(PartialStateContractMismatchError):
        require_matching_partial_state_contract(None, V3, model_label="mlp")
    require_matching_partial_state_contract(V3, V3, model_label="mlp")
