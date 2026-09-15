"""Tests for the cell-config derivation used by serving_stability_v1's replication."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts_cosim.derive_treated_cell_config import (  # noqa: E402
    ConfigDeriveError, delta, derive, flatten,
)


def _base(seed=1):
    return {"network": {"topology": {"seed": seed, "nodes": 8}},
            "scheduler": {"batch_timeout": 0.02, "queue_length": 100}}


def _treated(seed=1):
    d = _base(seed)
    d["scheduler"]["batch_timeout"] = 16.0
    return d


def test_flatten_uses_dotted_paths():
    assert flatten(_base())["scheduler.batch_timeout"] == 0.02


def test_delta_names_only_the_changed_key():
    assert set(delta(_base(), _treated())) == {"scheduler.batch_timeout"}


def test_the_treatment_is_applied_to_the_new_base():
    out = derive(_base(), _treated(), _base(seed=9001), ["network.topology.seed"])
    assert out["scheduler"]["batch_timeout"] == 16.0
    assert out["network"]["topology"]["seed"] == 9001


def test_nothing_else_moves():
    out = derive(_base(), _treated(), _base(seed=9001), ["network.topology.seed"])
    assert set(delta(_base(seed=9001), out)) == {"scheduler.batch_timeout"}


def test_a_new_base_that_differs_elsewhere_is_refused():
    other = _base(seed=9001)
    other["network"]["topology"]["nodes"] = 16
    with pytest.raises(ConfigDeriveError, match="not the same cell"):
        derive(_base(), _treated(), other, ["network.topology.seed"])


def test_an_already_treated_new_base_is_refused():
    with pytest.raises(ConfigDeriveError, match="already carries"):
        derive(_base(), _treated(), _treated(seed=9001), ["network.topology.seed"])


def test_an_empty_treatment_is_refused():
    with pytest.raises(ConfigDeriveError, match="no treatment"):
        derive(_base(), _base(), _base(seed=9001), ["network.topology.seed"])


def test_a_treatment_that_removes_a_key_is_refused():
    stripped = _treated()
    del stripped["scheduler"]["queue_length"]
    with pytest.raises(ConfigDeriveError, match="REMOVES"):
        derive(_base(), stripped, _base(seed=9001), ["network.topology.seed"])


def test_the_allowed_difference_must_be_named_explicitly():
    """Without --allow-differ the seed itself makes the two cells 'not the same cell'."""
    with pytest.raises(ConfigDeriveError, match="not the same cell"):
        derive(_base(), _treated(), _base(seed=9001), [])


def test_derived_matches_the_treated_example_except_the_draw():
    out = derive(_base(), _treated(), _base(seed=9001), ["network.topology.seed"])
    assert set(delta(_treated(), out)) == {"network.topology.seed"}
