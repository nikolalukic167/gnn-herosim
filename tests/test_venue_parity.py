"""Regression guard for the venue-parity fixture.

The fixture and its reference are only useful if they still describe the checkpoint that is
actually deployed. A silent drift — someone re-mints the checkpoint, or the graph builder
changes shape — turns `verify_venue_parity.py --assert` into a check that passes because it
is comparing nothing. These tests fail loudly in that case instead.

Skips rather than fails when the checkpoint is absent: `models/` is gitignored, so a fresh
clone legitimately has the fixture but not the weights.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts_cosim.verify_venue_parity import (  # noqa: E402
    DEFAULT_FIXTURE,
    DEFAULT_MODEL,
    DEFAULT_REFERENCE,
    compare_logits,
    forward_fixture,
    load_fixture,
)

requires_checkpoint = pytest.mark.skipif(
    not DEFAULT_MODEL.is_file(), reason=f"checkpoint not present at {DEFAULT_MODEL}"
)


def test_fixture_and_reference_are_committed():
    assert DEFAULT_FIXTURE.is_file(), f"missing fixture {DEFAULT_FIXTURE}"
    assert DEFAULT_REFERENCE.is_file(), f"missing reference {DEFAULT_REFERENCE}"


def test_fixture_carries_no_pickled_objects():
    """The whole point is loading the same bytes under a different torch_geometric.

    `allow_pickle=False` is what makes the fixture version-agnostic; a pickled PyG `Data`
    would either fail to load or silently migrate under a different PyG, which is precisely
    the failure this tool exists to detect.
    """
    archive = np.load(DEFAULT_FIXTURE, allow_pickle=False)
    assert int(archive["n_graphs"]) > 0


def test_reference_records_the_env_that_produced_it():
    """A reference with no env stamp cannot tell you which stack blessed these numbers."""
    import json

    archive = np.load(DEFAULT_REFERENCE, allow_pickle=False)
    assert "env" in archive, "reference has no env stamp; re-mint with --write-reference"
    env = json.loads(str(archive["env"]))
    for key in ("torch", "numpy", "torch_geometric", "python_version", "executable"):
        assert env.get(key), f"reference env is missing {key!r}"


@requires_checkpoint
def test_current_venue_reproduces_the_reference_exactly():
    """This venue must decide identically to the reference, or its total_rtt is not comparable."""
    import torch

    torch.set_num_threads(1)
    graphs = load_fixture(DEFAULT_FIXTURE)
    actual = forward_fixture(graphs, DEFAULT_MODEL, "cpu")
    report = compare_logits(actual, DEFAULT_REFERENCE)

    assert report["argmax_flips"] == 0, (
        f"{report['argmax_flips']}/{report['n_decisions']} argmax flips vs the reference "
        f"(max|delta|={report['max_abs_delta']:.3e}). This venue does not decide identically; "
        f"see PARITY.md."
    )
