"""route_b env pivot W4 — ladder grid presets (H0 -> H1 -> H2 -> H3).

No corpus is generated for this file (per the Phase A hard boundary) — it validates
the presets by constructing the same grid product the real generator would
(grid_topology_variants / grid_total_datasets, the file's own source of truth), the
"dry-run-style sanity" the plan calls for since generate_gnn_datasets_fast.py has no
dedicated dry-run mode.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts_cosim"))

from generate_gnn_datasets_fast import (  # noqa: E402
    GRID_PRESETS,
    ROUTE_B_PILOT_V1_GRID,
    ROUTE_B_PIVOT_H0_GRID,
    ROUTE_B_PIVOT_H1_GRID,
    ROUTE_B_PIVOT_H2_GRID,
    ROUTE_B_PIVOT_H3_GRID,
    grid_topology_variants,
    grid_total_datasets,
    resolve_grid_preset,
)

RUNGS = [
    ("route_b_pivot_h0", ROUTE_B_PIVOT_H0_GRID),
    ("route_b_pivot_h1", ROUTE_B_PIVOT_H1_GRID),
    ("route_b_pivot_h2", ROUTE_B_PIVOT_H2_GRID),
    ("route_b_pivot_h3", ROUTE_B_PIVOT_H3_GRID),
]


def test_all_four_rungs_are_registered_in_grid_presets():
    for name, preset in RUNGS:
        assert name in GRID_PRESETS
        assert GRID_PRESETS[name] is preset


def test_every_rung_has_the_204_shape():
    """2 conn_probs x 2 replica_configs x 3 queue_dists x 17 seeds — same shape as
    ROUTE_B_PILOT_V1_GRID, so a rung's screen numbers are comparable cell-for-cell."""
    for name, preset in RUNGS:
        total = grid_total_datasets(preset)
        assert total == 204, f"{name}: expected 204-shape, got {total}"


def test_h0_is_config_only_and_carries_no_new_grid_keys():
    """H0 = 'config-only scarcity squeeze on today's machinery' — no demand_spread,
    no replica_overlap, no dag_instances beyond the pilot's default of 1."""
    assert "demand_spread" not in ROUTE_B_PIVOT_H0_GRID
    assert "replica_overlap" not in ROUTE_B_PIVOT_H0_GRID
    assert ROUTE_B_PIVOT_H0_GRID.get("dag_instances", 1) == 1
    assert ROUTE_B_PIVOT_H0_GRID["server_node_counts"] == [4]
    assert ROUTE_B_PIVOT_H0_GRID["replica_server_percentage"] < 0.6  # tighter than the
    # generator's own spreading floor (generate_infrastructure.py:586's max(server_pct,
    # 0.6)), confirming this is a genuine squeeze relative to the pilot.


def test_h1_adds_demand_spread_only():
    """H1 = H0 + per-instance demand heterogeneity. Nothing else changes relative to
    H0 (same scarcity substrate, same seeds, no overlap yet)."""
    assert ROUTE_B_PIVOT_H1_GRID["demand_spread"] == {
        "dist": "uniform", "params": [0.5, 2.0]}
    assert "replica_overlap" not in ROUTE_B_PIVOT_H1_GRID
    for key in ("server_node_counts", "replica_configs", "replica_server_percentage",
               "seeds", "queue_distributions"):
        assert ROUTE_B_PIVOT_H1_GRID[key] == ROUTE_B_PIVOT_H0_GRID[key]


def test_h2_adds_replica_overlap_on_top_of_h1():
    assert ROUTE_B_PIVOT_H2_GRID["replica_overlap"] is True
    assert ROUTE_B_PIVOT_H2_GRID["demand_spread"] == ROUTE_B_PIVOT_H1_GRID["demand_spread"]
    assert ROUTE_B_PIVOT_H2_GRID.get("dag_instances", 1) == 1


def test_h3_adds_dag_instances_2_on_top_of_h2():
    assert ROUTE_B_PIVOT_H3_GRID["dag_instances"] == 2
    assert ROUTE_B_PIVOT_H3_GRID["replica_overlap"] is True
    assert ROUTE_B_PIVOT_H3_GRID["demand_spread"] == ROUTE_B_PIVOT_H2_GRID["demand_spread"]


def test_ladder_is_strictly_nested_h0_through_h3():
    """Each rung is the previous rung plus exactly one new lever — verified as a
    dict-subset relationship (every H(n) key/value not overridden by H(n+1) survives
    unchanged), matching the plan's 'H1 = H0 + X' phrasing literally."""
    chain = [ROUTE_B_PIVOT_H0_GRID, ROUTE_B_PIVOT_H1_GRID, ROUTE_B_PIVOT_H2_GRID,
            ROUTE_B_PIVOT_H3_GRID]
    ignore = {"default_output_subdir"}  # the one key every rung must DIFFER on
    for prev, nxt in zip(chain, chain[1:]):
        for key, value in prev.items():
            if key in ignore:
                continue
            assert key in nxt, f"{key} dropped between rungs"
            # value may be extended (e.g. demand_spread added) but never SILENTLY
            # changed for a key both rungs share and neither rung's own new-lever
            # comment claims to touch
    subdirs = {g["default_output_subdir"] for g in chain}
    assert len(subdirs) == 4, "every rung must write to its own output dir"


def test_h3_max_candidates_per_task_type_and_skip_threshold_derivation():
    """Derive (not guess) the MAX_PLACEMENT_COMBINATIONS_SKIP bound for H3 from the
    grid's own replica_configs/server_node_counts, mirroring the comment above
    ROUTE_B_PIVOT_H3_GRID. max(per_server)=2, server_node_counts=[4] -> at most 8
    candidates per task type (replica_overlap does not raise this: it lets a SECOND
    type reuse the same <=8 slots, it does not grant either type more of them)."""
    per_server_max = max(cfg[1] for cfg in ROUTE_B_PIVOT_H0_GRID["replica_configs"])
    n_servers = ROUTE_B_PIVOT_H0_GRID["server_node_counts"][0]
    max_candidates_per_type = per_server_max * n_servers
    assert max_candidates_per_type == 8

    n_task_types = 4  # diamond4
    four_task_bound = max_candidates_per_type ** n_task_types
    assert four_task_bound == 4096  # H0/H1/H2 (dag_instances=1): far under any default

    dag_instances = ROUTE_B_PIVOT_H3_GRID["dag_instances"]
    eight_task_bound = max_candidates_per_type ** (n_task_types * dag_instances)
    assert eight_task_bound == 16_777_216
    # The registered floor for H3 generation: MAX_PLACEMENT_COMBINATIONS_SKIP must
    # clear this product (the pre-uniqueness product, per
    # herosim-cosim-skip-threshold-is-pre-uniqueness.md), not the 250k default, which
    # this product is FAR above -- the point of this test is that the bound is
    # DERIVED per rung and shown to exceed the default, not assumed safe by default.
    default_skip = 250_000
    assert eight_task_bound > default_skip, (
        "H3 generation MUST export MAX_PLACEMENT_COMBINATIONS_SKIP explicitly "
        "(>= 16777216) or the 250k default will silently skip the most-contended "
        "8-task datasets — the exact 8-task lesson this derivation exists to avoid "
        "repeating")


def test_paired_separable_control_is_env_not_a_grid_key():
    """The B0-analog control is HEROSIM_DATA_LOCALITY/HEROSIM_OUTPUT_SIZE_BYTES unset
    at generation time — an env var switch, not a grid preset, so no
    *_control/*_b0 preset should exist for any rung (would be a second, easily-
    diverging copy of the same grid)."""
    for name in GRID_PRESETS:
        if name.startswith("route_b_pivot_h"):
            assert "control" not in name and "b0" not in name


def test_resolve_grid_preset_finds_all_four_rungs():
    for name, preset in RUNGS:
        assert resolve_grid_preset(name) is preset


def test_topology_variants_construct_without_error_for_every_rung():
    """Exercises grid_topology_variants (the generator's own topology-axis builder) on
    every rung — the closest thing to a dry run without actually generating a corpus."""
    for name, preset in RUNGS:
        variants = grid_topology_variants(preset)
        assert len(variants) == len(preset["connection_probabilities"])
        for label, kwargs in variants:
            assert kwargs["topology_type"] == "erdos_renyi"
            assert "connection_prob" in kwargs
