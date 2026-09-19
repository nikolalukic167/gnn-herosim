"""The autoscaler may be restricted to the platform types the training corpus contained.

Audit 2026-09-13. The `peer_affinity_v1` co-sim corpora carry `xavierGpu` and `xavierDla`
platform rows but never make either a replica, so neither is ever a candidate the model
ranks -- 516 of 516 datasets. Live, the same cell under a 450,729-task trace scales up far
enough that `xavierGpu` becomes a replica and 7.6 % of live candidates are of a type the
checkpoint never saw as one. The damage is not only the unseen one-hot column:
`node_caps[n] = alpha x max single candidate demand on n`, and the GPU demand is 1.739
against a corpus maximum of 0.213, so one such candidate inflates that node's cap ~8x and
the capacity mask stops binding for the whole batch (measured: 46.4 % of live node caps
exceed the largest cap in training).

`HEROSIM_REPLICA_PLATFORM_TYPES` restores the corpus's action space for every arm at once.
Default unset = no restriction, so every result recorded before this date is bit-identical.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.placement.autoscaler import (  # noqa: E402
    replica_platform_type_allowed,
    replica_platform_types_allowed,
)


@pytest.fixture(autouse=True)
def _clean_env():
    saved = os.environ.pop("HEROSIM_REPLICA_PLATFORM_TYPES", None)
    yield
    os.environ.pop("HEROSIM_REPLICA_PLATFORM_TYPES", None)
    if saved is not None:
        os.environ["HEROSIM_REPLICA_PLATFORM_TYPES"] = saved


def test_unset_allows_every_platform_type():
    assert replica_platform_types_allowed() is None
    for name in ("rpiCpu", "xavierCpu", "pynqFpga", "xavierGpu", "xavierDla"):
        assert replica_platform_type_allowed(name)


def test_an_allow_list_admits_only_what_it_names():
    os.environ["HEROSIM_REPLICA_PLATFORM_TYPES"] = "rpiCpu,xavierCpu,pynqFpga"
    assert replica_platform_types_allowed() == frozenset({"rpiCpu", "xavierCpu", "pynqFpga"})
    assert replica_platform_type_allowed("rpiCpu")
    assert not replica_platform_type_allowed("xavierGpu")
    assert not replica_platform_type_allowed("xavierDla")


def test_whitespace_and_ordering_do_not_matter():
    os.environ["HEROSIM_REPLICA_PLATFORM_TYPES"] = " pynqFpga , rpiCpu ,xavierCpu "
    assert replica_platform_types_allowed() == frozenset({"rpiCpu", "xavierCpu", "pynqFpga"})


def test_an_empty_allow_list_fails_loud_rather_than_silently_allowing_everything():
    """A typo that collapses the list must not read as "no restriction" -- that is the
    difference between a restricted gate and an unrestricted one, and it would be invisible
    in the result."""
    os.environ["HEROSIM_REPLICA_PLATFORM_TYPES"] = " , ,"
    with pytest.raises(ValueError, match="lists no platform type"):
        replica_platform_types_allowed()


def test_every_autoscaler_consults_the_allow_list():
    """The base `scale_up` filter is not enough: each policy's `create_first_replica`
    builds its own hardware-target set, and a target the allow-list forbids would make
    `scale_up` return StopIteration for a task that has no other option."""
    import inspect

    from src.policy.gnn import autoscaler as gnn_as
    from src.policy.knative import autoscaler as kn_as
    from src.policy.knative_network import autoscaler as knn_as
    from src.placement import autoscaler as base_as

    for mod in (base_as, gnn_as, kn_as, knn_as):
        assert "replica_platform_type_allowed" in inspect.getsource(mod), mod.__name__
