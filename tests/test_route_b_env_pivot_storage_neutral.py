"""route_b_env_pivot_v1 AMENDMENT 1, build item A1 — HEROSIM_STORAGE_NEUTRAL.

The screen's paired "separable control" was defined as Arm S with HEROSIM_DATA_LOCALITY /
HEROSIM_OUTPUT_SIZE_BYTES unset, and that ablation does NOT produce separable physics: the
always-on storage-tier branch (infrastructure.py:1244-1252 / :1265-1280) prices a child's
input read from flashCard when every parent ran on this node and from someRemote otherwise
— 0.0156 s per task charged as a function of where the PARENTS ran. These tests lock in
that HEROSIM_STORAGE_NEUTRAL=1 removes that cost difference, that it is inert when unset,
and that it touches nothing but the read path.

Teeth: test_storage_neutral_zeroes_parent_locality_cost fails without the override (the
two arms differ by the someRemote/flashCard read gap).
"""

import importlib
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.executecosimulation as ec  # noqa: E402


# data/nofs-ids/storage-types.json, the file every corpus shares.
LOCAL_READ_MBPS = 235.0
LOCAL_READ_LATENCY = 0.00012
REMOTE_READ_MBPS = 108.0
REMOTE_READ_LATENCY = 0.015
# data/nofs-ids/task-types.json — welded input for the nofs applications.
INPUT_BYTES = 153600.0


def storage_types():
    """A faithful copy of the two tiers the local_dependencies branch selects between."""
    return {
        "flashCard": {
            "name": "Flash Card",
            "remote": False,
            "capacity": 64,
            "iops": {"read": 92500, "write": 60170},
            "throughput": {"read": LOCAL_READ_MBPS, "write": 171},
            "latency": {"read": LOCAL_READ_LATENCY, "write": LOCAL_READ_LATENCY},
        },
        "someRemote": {
            "name": "Some Remote Storage",
            "remote": True,
            "capacity": 10 ** 12,
            "iops": {"read": 92500, "write": 60170},
            "throughput": {"read": REMOTE_READ_MBPS, "write": 108},
            "latency": {"read": REMOTE_READ_LATENCY, "write": REMOTE_READ_LATENCY},
        },
    }


# data/nofs-ids/infrastructure.json — every node in the corpus.
NODE_BANDWIDTH_MBPS = 100.0


def read_cost(tier, bandwidth_mbps=NODE_BANDWIDTH_MBPS, payload_bytes=INPUT_BYTES):
    """The quantity infrastructure.py:1288-1301 charges for the input read.

    Recomputed here from the tier dict rather than imported, so this test fails if the
    override silently stops moving the fields the simulator actually reads. Note the
    remote arm is clamped by the node's NIC (`min(throughput.read, bandwidth)`) and the
    local arm is not — the asymmetry that makes equalising throughput insufficient.
    """
    speed = (
        tier["throughput"]["read"]
        if not tier["remote"]
        else min(tier["throughput"]["read"], bandwidth_mbps)
    )
    return payload_bytes / (speed * 1024 * 1024) + tier["latency"]["read"]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("HEROSIM_STORAGE_NEUTRAL", raising=False)
    yield


def test_parent_locality_gap_exists_without_the_override(monkeypatch):
    """The defect AMENDMENT 1 documents: the two arms cost 0.0156 s apart.

    This is the teeth for the whole file — if this ever reads ~0, the coupling the control
    needed to ablate is gone for some other reason and the amendment's premise is stale.
    """
    sim_inputs = {"storage_types": storage_types()}
    applied = ec.apply_storage_neutral_override(sim_inputs)
    assert applied is False

    st = sim_inputs["storage_types"]
    gap = read_cost(st["someRemote"]) - read_cost(st["flashCard"])
    assert gap == pytest.approx(0.015722, abs=1e-6)


def test_storage_neutral_zeroes_parent_locality_cost(monkeypatch):
    """With the override on, local and remote reads cost EXACTLY the same."""
    monkeypatch.setenv("HEROSIM_STORAGE_NEUTRAL", "1")
    sim_inputs = {"storage_types": storage_types()}
    applied = ec.apply_storage_neutral_override(sim_inputs)
    assert applied is True

    st = sim_inputs["storage_types"]
    # Exactly zero, not merely small — this gate decides VOID-S0, so a "negligible"
    # residual is not good enough.
    assert read_cost(st["someRemote"]) == read_cost(st["flashCard"])
    assert st["someRemote"]["throughput"]["read"] == st["flashCard"]["throughput"]["read"]
    assert st["someRemote"]["latency"]["read"] == st["flashCard"]["latency"]["read"]


def test_storage_neutral_survives_the_node_bandwidth_clamp(monkeypatch):
    """The remote arm is clamped by the NIC; equalising throughput alone is NOT enough.

    Regression for the 2026-08-27 near-miss: setting someRemote's read to flashCard's
    235 MB/s left 0.00084 s of coupling at the corpus's real 100 Mbps nodes (an 18.7x
    reduction, but not zero) because `min(throughput.read, bandwidth)` still bit the
    remote arm while the local arm read unclamped. Both tiers are pinned at or below the
    fabric bandwidth so the clamp is a no-op.
    """
    monkeypatch.setenv("HEROSIM_STORAGE_NEUTRAL", "1")
    sim_inputs = {"storage_types": storage_types()}
    ec.apply_storage_neutral_override(sim_inputs)
    st = sim_inputs["storage_types"]

    # At or above the pinned read speed the clamp is a no-op and the arms are identical.
    # The corpus's real fabric is 100 Mbps, which is exactly the default.
    for bandwidth in (100.0, 235.0, 1000.0):
        gap = read_cost(st["someRemote"], bandwidth) - read_cost(st["flashCard"], bandwidth)
        assert gap == 0.0, f"parent-locality coupling survives at {bandwidth} Mbps: {gap}"


def test_storage_neutral_read_mbps_is_overridable(monkeypatch):
    monkeypatch.setenv("HEROSIM_STORAGE_NEUTRAL", "1")
    monkeypatch.setenv("HEROSIM_STORAGE_NEUTRAL_READ_MBPS", "50")
    sim_inputs = {"storage_types": storage_types()}
    ec.apply_storage_neutral_override(sim_inputs)
    st = sim_inputs["storage_types"]

    assert st["flashCard"]["throughput"]["read"] == 50.0
    assert st["someRemote"]["throughput"]["read"] == 50.0
    assert read_cost(st["someRemote"]) == read_cost(st["flashCard"])


def test_storage_neutral_read_mbps_rejects_nonpositive(monkeypatch):
    monkeypatch.setenv("HEROSIM_STORAGE_NEUTRAL", "1")
    monkeypatch.setenv("HEROSIM_STORAGE_NEUTRAL_READ_MBPS", "0")
    with pytest.raises(ValueError, match="must be positive"):
        ec.apply_storage_neutral_override({"storage_types": storage_types()})


def test_storage_neutral_refuses_a_read_speed_above_the_fabric(monkeypatch):
    """The clamp only vanishes when the pinned speed is <= every node's bandwidth.

    Pinning above the fabric silently reintroduces the coupling on the remote arm — the
    exact near-miss this lever was rewritten to close — so it must be refused rather than
    trusted. `infrastructure["network"]["bandwidth"]` is the single shared block every
    node's clamp reads (simulation.py:142).
    """
    monkeypatch.setenv("HEROSIM_STORAGE_NEUTRAL", "1")
    monkeypatch.setenv("HEROSIM_STORAGE_NEUTRAL_READ_MBPS", "235")
    sim_inputs = {
        "storage_types": storage_types(),
        "infrastructure": {"network": {"bandwidth": 100}},
    }
    with pytest.raises(RuntimeError, match="bandwidth"):
        ec.apply_storage_neutral_override(sim_inputs)


def test_storage_neutral_accepts_a_read_speed_at_the_fabric(monkeypatch):
    monkeypatch.setenv("HEROSIM_STORAGE_NEUTRAL", "1")
    sim_inputs = {
        "storage_types": storage_types(),
        "infrastructure": {"network": {"bandwidth": 100}},
    }
    assert ec.apply_storage_neutral_override(sim_inputs) is True
    st = sim_inputs["storage_types"]
    assert read_cost(st["someRemote"], 100.0) == read_cost(st["flashCard"], 100.0)


@pytest.mark.parametrize("value", [None, "0", "", "false"])
def test_storage_neutral_is_inert_when_unset(monkeypatch, value):
    """Default-off contract: sim_inputs must come back untouched."""
    if value is None:
        monkeypatch.delenv("HEROSIM_STORAGE_NEUTRAL", raising=False)
    else:
        monkeypatch.setenv("HEROSIM_STORAGE_NEUTRAL", value)

    before = storage_types()
    sim_inputs = {"storage_types": storage_types()}
    applied = ec.apply_storage_neutral_override(sim_inputs)

    assert applied is False
    assert sim_inputs["storage_types"] == before


def test_storage_neutral_leaves_write_path_untouched(monkeypatch):
    """Only the read path moves — warmth/disk semantics must not be perturbed.

    node_disk_v2's pull and the FilterStore contention price the WRITE path and the
    `remote` flag; moving those would change Arm S physics through the back door.
    """
    monkeypatch.setenv("HEROSIM_STORAGE_NEUTRAL", "1")
    before = storage_types()
    sim_inputs = {"storage_types": storage_types()}
    ec.apply_storage_neutral_override(sim_inputs)

    after = sim_inputs["storage_types"]
    for tier in ("flashCard", "someRemote"):
        assert after[tier]["throughput"]["write"] == before[tier]["throughput"]["write"]
        assert after[tier]["latency"]["write"] == before[tier]["latency"]["write"]
        assert after[tier]["remote"] == before[tier]["remote"]
        assert after[tier]["capacity"] == before[tier]["capacity"]
        assert after[tier]["iops"] == before[tier]["iops"]
        assert after[tier]["name"] == before[tier]["name"]


@pytest.mark.parametrize("missing", ["flashCard", "someRemote"])
def test_storage_neutral_fails_loudly_on_missing_tier(monkeypatch, missing):
    """No silent skip — a control corpus that looks amended but isn't is the one
    failure mode this lever must not have."""
    monkeypatch.setenv("HEROSIM_STORAGE_NEUTRAL", "1")
    st = storage_types()
    del st[missing]
    sim_inputs = {"storage_types": st}

    with pytest.raises(RuntimeError, match=missing):
        ec.apply_storage_neutral_override(sim_inputs)


def test_bandwidth_guard_reads_the_real_infrastructure_schema(monkeypatch):
    """Pin the guard against the actual on-disk file, not a hand-built dict.

    The first implementation looked for a per-node `network.bandwidth`, which does not
    exist — `data/nofs-ids/infrastructure.json` carries one shared
    `network.bandwidth: 100`. A guard that reads the wrong key finds nothing and never
    fires, which is worse than no guard at all.
    """
    import json

    repo = Path(__file__).resolve().parent.parent
    infrastructure = json.loads((repo / "data/nofs-ids/infrastructure.json").read_text())
    assert infrastructure["network"]["bandwidth"] == 100

    monkeypatch.setenv("HEROSIM_STORAGE_NEUTRAL", "1")
    monkeypatch.setenv("HEROSIM_STORAGE_NEUTRAL_READ_MBPS", "235")
    with pytest.raises(RuntimeError, match="bandwidth"):
        ec.apply_storage_neutral_override(
            {"storage_types": storage_types(), "infrastructure": infrastructure})


def test_storage_neutral_is_wired_into_the_sim_input_loader():
    """The override must run where sim_inputs is assembled, not only on demand.

    A lever that exists but is never called is how a control corpus silently keeps its
    coupling; pin the call site rather than trusting it.
    """
    import inspect

    source = inspect.getsource(ec.load_simulation_inputs)
    assert "apply_storage_neutral_override" in source
