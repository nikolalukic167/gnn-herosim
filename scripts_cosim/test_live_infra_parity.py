#!/usr/bin/env python3
"""Tests for scripts_cosim/verify_live_infra_parity.py.

The controls matter as much as the positive cases: a parity gate that cannot fail is
worse than no gate (see LINEAGES.md "GATE TOOLS"). Every fatal class below has a test
that asserts it *does* fail, and the `--seed` control asserts the documented
seed-override divergence is caught rather than tolerated.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts_cosim.verify_live_infra_parity import (  # noqa: E402
    DEFAULT_SIM_INPUT,
    _classify_corpus_only_edges,
    compare_topology,
    is_client,
    verify_dataset,
)

CORPUS = REPO_ROOT / "simulation_data" / "gnn_datasets_4tasks_contention_v2"
DATASET = CORPUS / "ds_00000"

pytestmark = pytest.mark.skipif(
    not (DATASET / "infrastructure.json").is_file(),
    reason="contention_v2 corpus not present on this machine",
)


def _load(path: Path):
    with open(path, "r") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def dataset_files():
    return (
        _load(DATASET / "space_with_network.json"),
        _load(DATASET / "infrastructure.json"),
    )


# --- the expected-divergence classifier -----------------------------------------


def test_client_prefix_comes_from_the_shared_helper():
    assert is_client("client_node0")
    assert not is_client("node0")


def test_repair_signature_requires_both_tier_crossing_and_base_latency():
    base = 0.1
    edges = {
        ("client_node1", "node2"): 0.1,  # repair: crosses tiers, at base
        ("node2", "client_node1"): 0.1,  # repair: reverse direction
        ("client_node1", "client_node2"): 0.1,  # not repair: same tier
        ("client_node1", "node3"): 0.0731,  # not repair: not at base latency
    }
    repair, unexplained = _classify_corpus_only_edges(edges, base)
    assert set(repair) == {("client_node1", "node2"), ("node2", "client_node1")}
    assert set(unexplained) == {
        ("client_node1", "client_node2"),
        ("client_node1", "node3"),
    }


def test_backbone_repair_edges_are_recognized_by_their_route_sum():
    """Under a backbone a repair edge's latency is a path sum, not base_latency, so the
    base-latency signature alone reports genuine repair edges as unexplained (which is
    what --allow-backbone-latency-divergence used to paper over). With the fabric passed
    in, the structural check identifies them exactly — and still rejects an edge whose
    latency does not match its own route."""
    base = 0.1
    backbone = {
        "links": {
            "client_node1|core0": {"latency": 0.02},
            "core0|core1": {"latency": 0.004},
            "core1|node2": {"latency": 0.021},
            "core0|node3": {"latency": 0.019},
        },
        "routes": {
            "client_node1": {
                "node2": ["client_node1", "core0", "core1", "node2"],  # sum = 0.045
                "node3": ["client_node1", "core0", "node3"],           # sum = 0.039
            }
        },
    }
    edges = {
        ("client_node1", "node2"): 0.045,   # repair: matches its route sum
        ("node2", "client_node1"): 0.045,   # repair: reverse direction
        ("client_node1", "node3"): 0.077,   # NOT repair: route says 0.039
        ("client_node1", "client_node2"): 0.045,  # NOT repair: same tier
    }
    repair, unexplained = _classify_corpus_only_edges(edges, base, backbone=backbone)
    assert set(repair) == {("client_node1", "node2"), ("node2", "client_node1")}
    assert set(unexplained) == {
        ("client_node1", "node3"),
        ("client_node1", "client_node2"),
    }

    # Without the fabric the same genuine repair edges are unexplained — the exact gap.
    repair_blind, unexplained_blind = _classify_corpus_only_edges(edges, base)
    assert not repair_blind
    assert len(unexplained_blind) == 4


# --- positive case ---------------------------------------------------------------


def test_dataset_regenerates_to_a_parity_matching_topology():
    result = verify_dataset(DATASET, DEFAULT_SIM_INPUT)
    assert result.ok, result.findings
    assert result.stats["live_only_edges"] == 0
    assert result.stats["latency_mismatches"] == 0
    assert result.stats["unexplained_corpus_only_edges"] == 0


def test_the_only_divergence_is_replica_reachability_repair():
    """Live is a strict subgraph; the missing edges are all repair edges."""
    result = verify_dataset(DATASET, DEFAULT_SIM_INPUT)
    stats = result.stats
    assert stats["repair_edges"] > 0, "this dataset is expected to exercise the repair path"
    assert stats["live_directed_edges"] + stats["repair_edges"] == stats["corpus_directed_edges"]
    assert any("replica_reachability_repair" in note for note in result.notes)


def test_repair_fraction_stays_small():
    """A large repair fraction would mean live physics differs materially from the corpus."""
    result = verify_dataset(DATASET, DEFAULT_SIM_INPUT)
    assert result.stats["repair_edge_fraction"] < 0.10


def test_several_datasets_pass():
    datasets = sorted(d for d in CORPUS.glob("ds_*") if d.is_dir())[:5]
    assert datasets
    for dataset in datasets:
        assert verify_dataset(dataset, DEFAULT_SIM_INPUT).ok


# --- controls: each fatal class must actually fail -------------------------------


def test_seed_override_fails_parity():
    """The live CLI's --seed overrides the config's topology seed and changes the graph.

    This is the control for the finding that a seed sweep is a *topology* sweep.
    """
    result = verify_dataset(DATASET, DEFAULT_SIM_INPUT, seed=42)
    assert not result.ok
    assert any("live-only edge" in f for f in result.findings)


def test_wrong_node_count_fails(dataset_files):
    space_config, corpus_infra = dataset_files
    bigger = copy.deepcopy(space_config)
    bigger["nodes"]["client_nodes"]["count"] = 40
    bigger["nodes"]["server_nodes"]["count"] = 40
    result = compare_topology(DATASET, bigger, corpus_infra, DEFAULT_SIM_INPUT)
    assert not result.ok
    assert any("node set mismatch" in f for f in result.findings)


def test_wrong_connection_probability_fails(dataset_files):
    """Same node count, different density — the sealed-holdout mismatch class."""
    space_config, corpus_infra = dataset_files
    denser = copy.deepcopy(space_config)
    denser["network"]["topology"]["connection_probability"] = 0.5
    result = compare_topology(DATASET, denser, corpus_infra, DEFAULT_SIM_INPUT)
    assert not result.ok


def test_live_only_edge_is_fatal(dataset_files):
    """Dropping a corpus edge makes live a superset — never acceptable."""
    space_config, corpus_infra = dataset_files
    trimmed = copy.deepcopy(corpus_infra)
    maps = trimmed["network_maps"]
    src = next(n for n in maps if maps[n])
    dst = next(iter(maps[src]))
    del maps[src][dst]
    result = compare_topology(DATASET, space_config, trimmed, DEFAULT_SIM_INPUT)
    assert not result.ok
    assert any("live-only edge" in f for f in result.findings)


def test_latency_mismatch_is_fatal(dataset_files):
    space_config, corpus_infra = dataset_files
    perturbed = copy.deepcopy(corpus_infra)
    maps = perturbed["network_maps"]
    src = next(n for n in maps if maps[n])
    dst = next(iter(maps[src]))
    maps[src][dst] = float(maps[src][dst]) + 0.05
    result = compare_topology(DATASET, space_config, perturbed, DEFAULT_SIM_INPUT)
    assert not result.ok
    assert any("latency" in f for f in result.findings)


def test_seed_provenance_mismatch_is_fatal(dataset_files):
    space_config, corpus_infra = dataset_files
    relabelled = copy.deepcopy(corpus_infra)
    relabelled["metadata"]["seed"] = 999999
    result = compare_topology(DATASET, space_config, relabelled, DEFAULT_SIM_INPUT)
    assert not result.ok
    assert any("seed provenance" in f for f in result.findings)


def test_same_tier_corpus_only_edge_is_not_excused(dataset_files):
    """A client<->client extra edge is not the repair signature and must fail."""
    space_config, corpus_infra = dataset_files
    tampered = copy.deepcopy(corpus_infra)
    base = space_config["network"]["latency"]["base_latency"]
    tampered["network_maps"]["client_node0"]["client_node1"] = base
    tampered["network_maps"]["client_node1"]["client_node0"] = base
    result = compare_topology(DATASET, space_config, tampered, DEFAULT_SIM_INPUT)
    assert not result.ok
    assert any("replica-reachability signature" in f for f in result.findings)
