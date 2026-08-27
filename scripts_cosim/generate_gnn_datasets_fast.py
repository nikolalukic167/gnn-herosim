#!/usr/bin/env python3
"""
Optimized GNN Dataset Generation Script

MANDATORY OUTPUT: placements/placements.jsonl
  Every successful dataset MUST persist the full brute-force sweep:
  one JSONL line per (placement_plan, rtt). prepare_graphs_cache* builds
  rtt_chunk_*.pkl from this file for counterfactual RTT / near-RTT training.

  refresh_optimal_full_stats.py --repair does NOT create JSONL.
  Repair + recache alone leaves ~40% of CE rows without counterfactual RTT lookup.

  Never delete .bf_scratch until placements/placements.jsonl exists and is non-empty.
  --resume must not skip datasets that have best.json but lack placements.jsonl.
  See docs/notes/placements_jsonl_required.md

This Python script replaces generate_gnn_datasets.sh with significant performance improvements:
1. Eliminates jq overhead (native Python JSON handling)
2. Eliminates subprocess spawning for infrastructure generation
3. Single Python process for all operations
4. Supports --quiet mode for faster execution
5. Uses orjson for faster JSON serialization when available

Usage:
    python scripts_cosim/generate_gnn_datasets_fast.py [--quiet] [--max-datasets N] [--workers N]

Example:
    # Generate up to 100 datasets with quiet mode
    python scripts_cosim/generate_gnn_datasets_fast.py --quiet --max-datasets 100
"""

import argparse
import json
import logging
import os
import random
import shutil
import signal
import subprocess
import sys
import time
import re
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Dict, List, Any, Sequence, Tuple, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Try to use orjson for faster JSON
try:
    import orjson
    
    def _convert_keys_to_str(obj):
        """Recursively convert dict keys to strings for orjson compatibility."""
        if isinstance(obj, dict):
            return {str(k): _convert_keys_to_str(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_convert_keys_to_str(item) for item in obj]
        elif isinstance(obj, tuple):
            return [_convert_keys_to_str(item) for item in obj]
        return obj
    
    def json_dumps(obj):
        return orjson.dumps(_convert_keys_to_str(obj)).decode('utf-8')
    def json_dumps_pretty(obj):
        return orjson.dumps(_convert_keys_to_str(obj), option=orjson.OPT_INDENT_2).decode('utf-8')
    HAS_ORJSON = True
except ImportError:
    def json_dumps(obj):
        return json.dumps(obj, separators=(',', ':'))
    def json_dumps_pretty(obj):
        return json.dumps(obj, indent=2)
    HAS_ORJSON = False

from src.generate_infrastructure import generate_deterministic_infrastructure
from src.executecosimulation import execute_brute_force_optimized, load_simulation_inputs
from src.sample_loader import ensure_workload_params, load_primary_sample_and_mapping

# Timeout for brute-force simulation (1 hour per dataset)
SIMULATION_TIMEOUT = 900  # seconds
# Skip datasets with excessive placement combinations (OOM guard)
MAX_PLACEMENT_COMBINATIONS_SKIP_DEFAULT = 250000


# =============================================================================
# CONFIGURATION GRIDS
# =============================================================================

GridPreset = Dict[str, Any]

# warmth_v2: fixed conn=0.50, moderate-to-heavy replicas/queues (500 ds).
WARMTH_V2_GRID: GridPreset = {
    "connection_probabilities": [0.50],
    "replica_configs": [
        (1, 2, 0.0, 0.0),
        (2, 2, 0.0, 0.0),
        (1, 3, 0.3, 0.5),
        (2, 3, 0.3, 0.5),
        (2, 2, 0.5, 0.7),
    ],
    "queue_distributions": [
        ("pois16", "poisson", 16, 0, 0, 42, 1),
        ("norm22", "normal", 22, 7, 0, 56, 1),
        ("pois28", "poisson", 28, 0, 0, 72, 1),
        ("norm35", "normal", 35, 11, 0, 96, 1),
        ("uniform20_80", "uniform", 20, 80, 0, 120, 1),
    ],
    "seeds": list(range(101, 121)),
    "default_output_subdir": "gnn_datasets_4tasks_1060_warmth_v2",
}

# sparse_warmth_v2 Option A: sparse topology + light replicas + low/mid queues (351 ds).
SPARSE_WARMTH_V2_GRID: GridPreset = {
    "connection_probabilities": [0.25, 0.30, 0.35],
    "replica_configs": [
        (1, 1, 0.0, 0.0),
        (1, 2, 0.0, 0.0),
        (2, 2, 0.0, 0.0),
    ],
    "queue_distributions": [
        ("pois12", "poisson", 12, 0, 0, 24, 1),
        ("norm16", "normal", 16, 5, 0, 32, 1),
        ("pois16", "poisson", 16, 0, 0, 42, 1),
    ],
    "seeds": list(range(121, 134)),
    "default_output_subdir": "gnn_datasets_4tasks_sparse_warmth_v2",
}

# skew_warmth_v2: degree_skewed_core hub topology + 5/30ms asymmetric latency (288 ds).
SKEW_WARMTH_V2_GRID: GridPreset = {
    "topology_type": "degree_skewed_core",
    "k_core_values": [4, 6, 8],
    "hub_seeker_fractions": [0.35, 0.50, 0.65],
    "latency_core_ms": 5,
    "latency_periphery_ms": 30,
    "connection_probabilities": [],
    "replica_configs": [
        (1, 2, 0.0, 0.0),
        (2, 2, 0.0, 0.0),
    ],
    "queue_distributions": [
        ("pois16", "poisson", 16, 0, 0, 42, 1),
        ("pois28", "poisson", 28, 0, 0, 72, 1),
    ],
    "seeds": list(range(141, 149)),
    "default_output_subdir": "gnn_datasets_4tasks_skew_warmth_v2",
}

# contention_v1: deliberately non-separable regime for GNN-necessity study.
# Levers (verified to drive joint coupling, see separability_diagnostic):
#   - sparse topology (few reachable platforms) => tasks must compete for the same options
#   - all-cold replicas (client_pct=server_pct=0) => simultaneous pulls serialize on node
#     FilterStore (N x pullTime), so co-locating cold tasks on one node is a real cliff
#   - heavy queues => stacking two tasks on one platform serializes execution
# MUST be generated with --allow-non-unique-replicas so the oracle can express collisions.
CONTENTION_V1_GRID: GridPreset = {
    "connection_probabilities": [0.25, 0.35],
    "replica_configs": [
        (1, 1, 0.0, 0.0),
        (1, 2, 0.0, 0.0),
        (2, 2, 0.0, 0.0),
    ],
    "queue_distributions": [
        ("pois28", "poisson", 28, 0, 0, 72, 1),
        ("norm35", "normal", 35, 11, 0, 96, 1),
        ("uniform20_80", "uniform", 20, 80, 0, 120, 1),
    ],
    "seeds": list(range(201, 215)),
    "default_output_subdir": "gnn_datasets_4tasks_contention_v1",
}

# contention_v2: SCARCE-WARM-RESOURCE regime. contention_v1 (all-cold) showed that
# cranking collision frequency does NOT break per-task greedy (regret stayed 0.00%):
# when colliding is optimal, both tasks independently prefer the same platform anyway.
# Greedy only breaks when two tasks share a #1 platform AND co-locating there is
# EXPENSIVE, forcing the optimum to split them. That needs a scarce *attractive* resource:
#   - warm replicas (high preinit) => a few platforms are clearly best => tasks compete
#   - heavy/pre-loaded queues => stacking two tasks on the warm platform serializes,
#     destroying its advantage so the optimum must split => anti-correlated preferences
#   - sparse topology => few fallback platforms => the split is non-trivial
# MUST be generated with --allow-non-unique-replicas.
# shallow_longexec_v1: the inverse of the contention series' core lever.
#
# Measured 2026-08-17 on all 899 contention_v2 sweeps: queue depth PREDICTS separability,
# monotonically. Shallowest quartile (depth 27.6) -> additive R^2 0.97822, collision gain
# +1.986pp; deepest quartile (depth 50.8) -> 0.99803, +0.181pp. The coupling is 11x weaker
# when queues are deep.
#
# The reason is arithmetic. The additive term is `depth x exec_time` and grows with depth;
# the interaction term is `added_in_batch x exec_time` and does NOT. Deepening queues
# therefore dilutes the only coupling the corpus has -- which is why contention_v4/v5
# landed at R^2 0.9997.
#
# So invert both factors: make queues shallow so the additive term is small, and use
# long-execution task types so each collision is large. cnn runs 0.706s on xavierCpu and
# 3.086s on rpiCpu vs dnn2's 0.024s.
# MUST be generated with --allow-non-unique-replicas so the oracle can express collisions.
SHALLOW_LONGEXEC_V1_GRID: GridPreset = {
    "connection_probabilities": [0.25, 0.35],
    "replica_configs": [
        (1, 1, 0.7, 0.9),
        (1, 2, 0.7, 0.9),
        (2, 2, 0.5, 0.7),
    ],
    "queue_distributions": [
        ("shallow_pois2", "poisson", 2, 0, 0, 8, 1),
        ("shallow_uniform0_4", "uniform", 0, 4, 0, 8, 1),
        ("shallow_norm3", "normal", 3, 1, 0, 8, 1),
    ],
    "seeds": list(range(701, 751)),
    "task_type_pair": ("cnn", "rf"),
    "default_output_subdir": "gnn_datasets_4tasks_shallow_longexec_v1",
}

# shallow_v1: the queue-depth half of shallow_longexec_v1, on the stock dnn1/dnn2 apps.
# Isolates the lever the 899-dataset measurement directly supports, with no new task types
# (which additionally require workload params in the sampled space).
SHALLOW_V1_GRID: GridPreset = {
    "connection_probabilities": [0.25, 0.35],
    "replica_configs": [
        (1, 1, 0.7, 0.9),
        (1, 2, 0.7, 0.9),
        (2, 2, 0.5, 0.7),
    ],
    "queue_distributions": [
        ("shallow_pois2", "poisson", 2, 0, 0, 8, 1),
        ("shallow_uniform0_4", "uniform", 0, 4, 0, 8, 1),
        ("shallow_norm3", "normal", 3, 1, 0, 8, 1),
    ],
    "seeds": list(range(701, 751)),
    "default_output_subdir": "gnn_datasets_4tasks_shallow_v1",
}

# route_a_pilot_v1: the first grid whose applications are a real DAG.
#
# Shallow queues are copied from shallow_v1 deliberately, not for variety: the additive
# term is depth x exec_time and dilutes every interaction, which is what falsified
# contention_v4/v5. Against shallow queues the parent->child transfer is not drowned.
#
# `state_size_bytes` is the lever and MUST be set from the 4e scaling probe rather than
# guessed. At the welded 153,600 B the dependency read is ~1.2% of the queue term. Unlike
# link bandwidth — where the additive and interaction terms both scale as 1/bandwidth and
# the ratio is invariant — the coupled term scales with stateSize while queue work does
# not, so the ratio MOVES. That is the whole reason this lever is worth pulling.
#
# `server_mesh` is required: without it a parent and child that both land on servers have
# no distance and no route, and the transfer term fails loud rather than charging 0.0.
#
# Seeds 801+ do not overlap any existing range (101-148, 201-214, 701-750).
ROUTE_A_PILOT_V1_GRID: GridPreset = {
    "connection_probabilities": [0.25, 0.35],
    "replica_configs": [
        (1, 1, 0.7, 0.9),
        (1, 2, 0.7, 0.9),
        (2, 2, 0.5, 0.7),
    ],
    "queue_distributions": [
        ("shallow_pois2", "poisson", 2, 0, 0, 8, 1),
        ("shallow_uniform0_4", "uniform", 0, 4, 0, 8, 1),
        ("shallow_norm3", "normal", 3, 1, 0, 8, 1),
    ],
    "seeds": list(range(801, 851)),
    "dag_shape": "diamond4",
    # Four DISTINCT types: a dag node name must be a unique key in task-types.json.
    "dag_task_types": ("dnn1", "dnn2", "rf", "cnn"),
    "server_mesh": True,
    # REQUIRED, not decorative. Without a fabric there is no parent->child ROUTE, and
    # `_payload_transfer_time` falls back to the child's own NIC — which makes the
    # magnitude-carrying half of the transfer separable by construction and produces a
    # guaranteed 0.000% regret regardless of payload size (measured 2026-08-25 over a
    # 100,000x range). Hop count over the backbone is what makes distance carry magnitude.
    "backbone_defaults": {"link_bandwidth_mbps": 1000.0},
    "default_output_subdir": "gnn_datasets_dag4_route_a_pilot_v1",
}

# route_b_pilot_v1: route A's stacked machinery + SCARCE placement substrate, for the
# free-choice (contention) hypothesis. See route_b_v1 PRE-REGISTRATION in LINEAGES.md.
#
# The memory-knapsack constraint itself is applied at SCORING time
# (scripts_cosim/score_route_b_contention.py) — it changes no physics, so one corpus
# serves the whole capacity-tightness ladder. What this grid must supply is the
# *competition substrate* the pre-probe found missing (free-choice plans collided in only
# 10% of m3-pilot datasets against ~22 candidate hosts):
#   - server_node_counts [4]: four servers for four tasks, so individual favourites
#     genuinely overlap;
#   - per_client = 0: every task crosses the network (netc_multihop_v1's lesson — a
#     client-local corner makes everything more separable, not less);
#   - four DISTINCT task types: the knapsack demands are type-asymmetric (GPU: dnn1/dnn2
#     0.9, rf 1.5, cnn 1.3), so WHICH types co-reside determines feasibility, not just
#     how many — the count-vector collapse the five co-location mechanisms died of does
#     not describe this constraint.
#
# The two arms are generated from THIS grid with the same seeds, differing only in env:
#   Arm S  (primary): HEROSIM_DATA_LOCALITY=1 HEROSIM_OUTPUT_SIZE_BYTES=800000000
#   Arm B0 (control): both unset (separable physics; theorem-predicted R_exact == 0)
#
# Seeds 901+ do not overlap any existing range (101-148, 201-214, 701-750, 801-850).
ROUTE_B_PILOT_V1_GRID: GridPreset = {
    "connection_probabilities": [0.25, 0.35],
    "server_node_counts": [6],
    "replica_configs": [
        (0, 2, 0.7, 0.9),
        (0, 3, 0.7, 0.9),
    ],
    "queue_distributions": [
        ("shallow_pois2", "poisson", 2, 0, 0, 8, 1),
        ("deepvar_uniform0_12", "uniform", 0, 12, 0, 16, 1),
        ("deepvar_pois4", "poisson", 4, 0, 0, 16, 1),
    ],
    "seeds": list(range(901, 918)),
    "dag_shape": "diamond4",
    "dag_task_types": ("dnn1", "dnn2", "rf", "cnn"),
    "server_mesh": True,
    "backbone_defaults": {"link_bandwidth_mbps": 1000.0},
    "default_output_subdir": "gnn_datasets_dag4_route_b_pilot_v1",
}

# route_b_pilot_v1_8task: the 8-task probe named in the route_b_v1 handover plan
# (2026-08-25) — same grid as ROUTE_B_PILOT_V1_GRID (2 conn_probs x 2 replica_configs x 3
# queue_dists x 17 seeds = 204 datasets), differing only in `dag_instances: 2`. Two
# diamond4 DAG instances submitted from different client nodes, co-decided in one episode
# (see generate_workload_templates), doubling the joint decision from 4 to 8 tasks. Tests
# whether pooled `krank` closure (0.790 at 4 tasks, LINEAGES §9c) survives the doubling.
ROUTE_B_PILOT_V1_8TASK_GRID: GridPreset = {
    **ROUTE_B_PILOT_V1_GRID,
    "dag_instances": 2,
    "default_output_subdir": "gnn_datasets_dag4_route_b_pilot_v1_8task",
}

# route_c_link_screen: the route_c_link_transfer_v1 SCREEN grid (registration in
# LINEAGES.md, 2026-08-26). ROUTE_B_PILOT_V1_GRID physics with the backbone squeezed to
# the measured link_contention_v1 coupling peak (n_core=4, attach=1, chords=0 — crossings
# per segment are the ratio lever; bandwidth is a null lever on wait/transfer but moves
# link cost's share of RTT). Bandwidth per rung comes from --link-bandwidth-mbps at the
# CLI (R1: 1000, R2+: 100, 25); the contended payload comes from
# HEROSIM_INPUT_SIZE_BYTES in the env (the fabric transmits INPUT over ingress routes —
# see apply_state_size_override). Same seeds as route_b so datasets pair with the Arm S
# anchor corpus.
ROUTE_C_LINK_SCREEN_GRID: GridPreset = {
    **ROUTE_B_PILOT_V1_GRID,
    "backbone_defaults": {
        "link_bandwidth_mbps": 1000.0,
        "n_core": 4,
        "attach_degree": 1,
        "chord_count": 0,
    },
    "default_output_subdir": "gnn_datasets_dag4_route_c_link_screen",
}

# route_c_link_screen_8task: the screen's registered CONTINGENCY rung. The 4-task ladder
# measured a STRUCTURAL ceiling on link-wait share (wait/(wait+transfer) median 4-6%, max
# 8.8% — under the 10% manipulation bar at ANY bandwidth): one client and a diamond DAG
# cap concurrent transfers at 2. Two diamond4 instances from independently drawn clients
# double the joint decision to 8 tasks and the peak transfer concurrency to 4+ — the
# 7-14x amplifier the link_contention_v1 real-trace A/B measured. Sweeps are ~100x the
# 4-task rungs; generate on datalab (route_b_8task_probe.sbatch pattern).
ROUTE_C_LINK_SCREEN_8TASK_GRID: GridPreset = {
    **ROUTE_C_LINK_SCREEN_GRID,
    "dag_instances": 2,
    "default_output_subdir": "gnn_datasets_route_c_link_screen_8task",
}

# route_b_pivot_h{0..3}: the route_b ENV PIVOT ladder (docs/lineages/route_b_env_pivot_v1/screen-preregistration.md,
# W4 of the pivot plan; registered rung order H0 -> H1 -> H2 -> H3, fixed, no post-hoc
# rungs). Each rung is the SAME 204-shape (2 conn_probs x 2 replica_configs x 3
# queue_dists x 17 seeds) as ROUTE_B_PILOT_V1_GRID, so a rung's screen numbers are
# comparable to the frozen pilot/stage-1 numbers cell-for-cell. Do NOT generate these
# corpora as part of Phase A — presets only, sign-off (W5) gates any actual generation.
#
# H0: config-only scarcity squeeze on TODAY's machinery (no new grid keys at all) —
# calibrates the screen: if S1 (structure exists) already fails here, the later rungs'
# comparison point is known. server_node_counts drops from the pilot's [6] to [4] and
# replica_server_percentage is pushed low (0.5, vs the pilot's default-derived ~0.6+)
# to concentrate replicas onto fewer hosts.
#
# CORRECTION 2026-08-27: this comment used to read "four servers for four task types, so
# individual favourites collide more directly", reasoning as if 4 servers meant 4 HOSTING
# nodes. It does not. server_node_counts=[4] x replica_server_percentage=0.5 puts replicas
# on exactly **2** hosting nodes — measured histogram {2: 204} on H0, H0_ctrl AND H1. The
# squeeze is twice as tight as the sentence implied, and that single fact drives three
# observed effects: alpha=1.5 is pigeonhole-infeasible (4 tasks over 2 nodes at
# cap = 1.5 x max_single_demand), the ~50% greedy_stuck rate (the 64-row arm always has
# exactly 2 task types confined to one node, which strands a non-backtracking greedy), and
# the marginal degeneracy behind the R_exact tie artifact. The GRID IS REGISTERED AND
# UNCHANGED — this is a comment correction only. See tests/test_route_b_env_pivot_w4.py.
#
# replica_configs keeps the pilot's TWO-arm shape (204 = 2x2x3x17, comparable cell-for-cell
# to the frozen pilot) but at TIGHTER absolute counts (1-2 per server, vs the pilot's 2-3)
# -- fewer replicas per node is a squeeze relative to the pilot at matched shape.
ROUTE_B_PIVOT_H0_GRID: GridPreset = {
    **ROUTE_B_PILOT_V1_GRID,
    "server_node_counts": [4],
    "replica_configs": [
        (0, 1, 0.7, 0.5),
        (0, 2, 0.7, 0.5),
    ],
    "replica_server_percentage": 0.5,
    # docs/lineages/route_b_env_pivot_v1/screen-preregistration.md §3: fresh seed block, none previously used. Without
    # this override the preset silently inherited ROUTE_B_PILOT_V1_GRID's 901-917 (the
    # frozen pilot's own seeds) via **ROUTE_B_PILOT_V1_GRID above -- caught in pre-flight
    # 2026-08-27, before any rung was scored (see LINEAGES route_b_env_pivot_v1 outcome).
    "seeds": list(range(3001, 3018)),
    "default_output_subdir": "gnn_datasets_dag4_route_b_pivot_h0",
}
# Paired separable control (B0-analog): HEROSIM_DATA_LOCALITY / HEROSIM_OUTPUT_SIZE_BYTES
# unset at generation time (same grid, config-identical) -- run as a SEPARATE generation
# pass with those env vars absent; R_exact must be ~0 on it (S0 gate). Not a distinct
# preset because the physics switch is an env var, not a grid key -- see W6's command
# sequence and CLAUDE.md's Arm S / Arm B0 convention.

# H1: H0 + per-instance demand heterogeneity (the packing hypothesis, minimal) +
# cap_mode alpha_mean (an independent-tightness cap that does not auto-scale away the
# scarcity heterogeneous demand would otherwise erase — score_route_b_contention.py's
# node_caps cap_mode option). demand_spread starts at uniform [0.5, 2.0] per the plan;
# cap_mode itself is a SCORING-time flag (--cap-mode on score_route_b_contention.py),
# not a generator grid key, so it is not stored in this preset — recorded here as the
# rung's registered scoring parameter.
ROUTE_B_PIVOT_H1_GRID: GridPreset = {
    **ROUTE_B_PIVOT_H0_GRID,
    "demand_spread": {"dist": "uniform", "params": [0.5, 2.0]},
    "seeds": list(range(3101, 3118)),  # §3: fresh block, distinct from H0's 3001-3017
    "default_output_subdir": "gnn_datasets_dag4_route_b_pivot_h1",
}
# Registered scoring parameter for this rung and H2/H3 below: --cap-mode alpha_mean.

# H2: H1 + overlapping eligibility (the assignment hypothesis) — task types share
# contested replica hosts/platforms (generate_infrastructure.py's preinit.
# replica_overlap, plumbed via the replica_overlap grid key).
ROUTE_B_PIVOT_H2_GRID: GridPreset = {
    **ROUTE_B_PIVOT_H1_GRID,
    "replica_overlap": True,
    "seeds": list(range(3201, 3218)),  # §3: fresh block, distinct from H0/H1
    "default_output_subdir": "gnn_datasets_dag4_route_b_pivot_h2",
}

# H3: H2 + dag_instances=2 (8-task joint decision, the largest rung), alpha at the
# registered doubling correspondence (see ROUTE_B_PILOT_V1_8TASK_GRID's own alpha
# note — the 8-task probe's alpha ladder mirrors the 4-task one 1:1 rather than
# doubling the cap, since cap_node is already per-node not per-task-count).
#
# MAX_PLACEMENT_COMBINATIONS_SKIP derivation (the 8-task lesson: 1M was not enough,
# the 250k default silently skips the most-contended datasets) — DERIVED, not guessed:
#   max candidates per task type here = max(per_server across replica_configs) *
#   server_node_counts = max(1, 2) * 4 = 8 (H0-H3's replica_configs top out at
#   per_server=2; replica_overlap in H2/H3 does not raise this per-type max, it only
#   lets a SECOND type reuse the same up-to-8 slots -- overlap changes which
#   platforms are shared, not how many candidates one type can have).
#   4-task (H0-H2): max Pi n_t = 8^4 = 4,096 -- far under any default, unaffected.
#   8-task (H3): two diamond4 instances, 8 tasks total, each with up to 8 candidates
#     (replica_overlap means instance 2's tasks compete for the SAME <=8-per-type
#     slots instance 1's tasks used, not a disjoint second set) ->
#     max Pi n_t = 8^8 = 16,777,216. This is the PRODUCT the skip threshold must clear
#     (herosim-cosim-skip-threshold-is-pre-uniqueness.md: the threshold tests the
#     product BEFORE the unique-replica-per-plan reduction, which is smaller but not
#     computed until the enumeration runs) -- so H3 generation must export
#     MAX_PLACEMENT_COMBINATIONS_SKIP >= 16777216 (e.g. 20000000 for headroom), FAR
#     above the 250k default (the ROUTE_B_PILOT_V1_8TASK_GRID lesson repeated: this
#     rung needs datalab, not a local run, per W6's H3 note).
ROUTE_B_PIVOT_H3_GRID: GridPreset = {
    **ROUTE_B_PIVOT_H2_GRID,
    "dag_instances": 2,
    "seeds": list(range(3301, 3318)),  # §3: fresh block, distinct from H0/H1/H2
    "default_output_subdir": "gnn_datasets_dag4_route_b_pivot_h3",
}

# netc_multihop_v1: shallow queues + NO client-local replicas, for link_contention_v1.
#
# The first matched pilot ran link_contention_v1 on the stock shallow_v1 grid and all three
# arms failed the gate on headroom (additive-argmin regret 2.51% off / 1.07% bw5p0 / 1.09%
# bw1p5, threshold 5%). The backbone made things *more* separable, not less.
#
# Mechanism: shallow_v1 keeps per_client >= 1, so most tasks can run on their own source
# node and never touch the network at all. A cost that only prices *remoteness* then pushes
# the optimum further toward the local corner the additive fit already picks -- the same
# shape as netc_scarce_v1, where penalising co-location pushed the optimum toward the corner
# greedy already picked. Measured on the bw5p0 arm: the optimum left 0 or 1 task remote in
# 5/16 datasets.
#
# per_client = 0 is the single-variable fix: every task must cross the network, so the
# per-link pipes are on the critical path for all four. Deliberately NOT combined with a
# replica_server_percentage cut -- netc_hotspot_v1 moved that and per_client together and
# its "cliff" turned out to be one node-occupancy integer over the only 2 hosts that
# existed. Server spread stays at the 0.6 floor so contention has somewhere to spread.
NETC_MULTIHOP_V1_GRID: GridPreset = {
    "connection_probabilities": [0.25, 0.35],
    "replica_configs": [
        (0, 1, 0.7, 0.9),
        (0, 2, 0.7, 0.9),
    ],
    "queue_distributions": [
        ("shallow_pois2", "poisson", 2, 0, 0, 8, 1),
        ("shallow_uniform0_4", "uniform", 0, 4, 0, 8, 1),
        ("shallow_norm3", "normal", 3, 1, 0, 8, 1),
    ],
    "seeds": list(range(701, 751)),
    "default_output_subdir": "gnn_datasets_4tasks_netc_multihop_v1",
}

# topo_transfer_v1: the topology-SIZE axis, for topology_transfer_v1.
#
# Every corpus in this repo before this one was generated at exactly one size --
# space_with_network.json's 20 clients + 20 servers -- so nothing could be held out to ask
# whether a model transfers across infrastructure. (`cluster_size` in sample_simple.json
# looks like the size knob but is inert: `calculate_device_counts` is defined in
# executecosimulation.py and never called. Node counts come from the config.)
#
# Inherits netc_multihop_v1's two deliberate choices, both load-bearing here:
#   - per_client = 0, so every task must cross the network. A grid where tasks can run on
#     their own source node makes topology irrelevant to the optimum, which is precisely
#     how the first link_contention_v1 pilot failed.
#   - shallow queues, which keep the pointwise ceiling low (the deep-queue arithmetic in
#     graph_structure_physics dilutes every interaction term).
#
# The size axis holds the CLIENT tier fixed at 20 and scales only servers, so matched arms
# differ in candidate-set size and nothing else; scaling clients would move the task-source
# draw itself.
#
# Ladder chosen from a measured combination-count probe (1 dataset per size, conn=0.25,
# rps=1, seed 801, shallow_pois2), because generating past the enumeration cap silently
# SKIPS datasets and would bias a held-out size toward its easier half. That cap is
# MAX_PLACEMENT_COMBINATIONS_SKIP_DEFAULT = 250,000 (this file, exported as
# $MAX_PLACEMENT_COMBINATIONS); earlier revisions of this comment said "100k", which was
# never the value in code.
#
# ⚠ The original probe table (2026-08-18) DOES NOT REPRODUCE. Re-measured 2026-08-19 on a
# 32-core box at --workers 8, both plan counts and times differ:
#
#     servers   plans (orig)   plans (re-run)   time (orig)   time (re-run)
#          20             32               18         0.8s          0.4s
#          28             48               44         0.8s          0.5s
#          40            432              343         2.0s          3.3s
#          60          2,730            2,231         9.3s         23.0s
#          80          9,828            8,698        39.0s        117.2s
#
# Suspected cause, NOT confirmed: the 2026-08-18 workload-seeding fix changed the draw, so
# the two tables enumerate different workloads. Use the re-run numbers for budgeting -- the
# top of the ladder is ~3x more expensive than recorded, and the LOW end is coarser than
# recorded (18 plans at 20 servers, not 32), which is what the ladder's low-end cutoff was
# justified on. Both tables are kept so a future re-run can tell which one it matches.
#
# The sweep grows ~quartically (4 tasks x candidates each), so full enumeration stays well
# inside the cap up to ~100 servers -- the ceiling is not the binding constraint here (the
# re-run peak, 8,698 plans, is 3.5% of the cap). The low end is: at 10-14 servers a sweep of
# 16 plans makes regret far too coarse to measure a degradation curve against. Hence a
# ladder starting at 20 rather than 10, giving train sizes {20, 28, 40} and held-out sizes
# {60, 80} at 1.5-4x the largest size seen in training, every label still a true sweep
# minimum.
#
# The candidates/task floor worry that motivated the probe is unfounded: geometric-mean
# candidates/task grows 2.38 -> 9.96 strictly monotonically across the ladder (replica-host
# nodes 7 -> 29), because `replica_server_pct = max(server_pct, 0.6)` is a PERCENTAGE and
# scales with server count.
TOPO_TRANSFER_V1_GRID: GridPreset = {
    "connection_probabilities": [0.25, 0.35],
    "server_node_counts": [20, 28, 40, 60, 80],
    "replica_configs": [
        (0, 1, 0.7, 0.9),
        (0, 2, 0.7, 0.9),
    ],
    "queue_distributions": [
        ("shallow_pois2", "poisson", 2, 0, 0, 8, 1),
        ("shallow_uniform0_4", "uniform", 0, 4, 0, 8, 1),
        ("shallow_norm3", "normal", 3, 1, 0, 8, 1),
    ],
    # 75 seeds -> 2*2*3*75 = 900 datasets/server_node_count, i.e. `tier_launch` in
    # gate_statistics.PHASE4_TIERS (registered 2026-08-19 as the Phase 4 launch tier;
    # was 30 seeds / 360 per size = tier_0.02, which the ladder arithmetic showed
    # cannot resolve either observed shallow_v1 win_rate effect). The size axis has
    # no per-size override, so this seed count applies uniformly across all five
    # server_node_counts, not just the two held-out sizes (60, 80) the gate reads.
    "seeds": list(range(801, 876)),
    "default_output_subdir": "gnn_datasets_4tasks_topo_transfer_v1",
    # BACKBONE ON BY DEFAULT (decided 2026-08-19). Without this the preset produced
    # `link_topology: null`, because the backbone block was only written when
    # --link-bandwidth-mbps was passed -- and `build_network_graph_block` treats a
    # missing fabric as a legitimate silent no-op. Training that corpus under
    # NETWORK_GRAPH_CONTRACT=core_v1 would have yielded zero network entities and
    # zero network edges without a word of warning: two topology-blind models, which
    # is precisely the failure Phase 2 exists to prevent. A grid whose whole question
    # is topology must not depend on the operator remembering a flag.
    #
    # 1000 MB/s is deliberately NON-BINDING: it buys routing STRUCTURE (routes, core
    # segments, shared-segment adjacency) without the link-contention effect, which
    # `link_contention_v1` already measured as real but small (0.08-0.35% regret).
    # Stacking a known-small, known-noisy mechanism on top of a signal being resolved
    # at MDG ~0.02 is how netc_hotspot_v1 lost attribution -- it moved percentage and
    # per_client together and could not say which produced the cliff. Contention under
    # transfer is a follow-on lineage, not a rider on this one.
    #
    # n_core stays FIXED at the argparse default (12) and does NOT scale with servers.
    # That makes the transfer axis candidate-set growth (2.38 -> 9.96 candidates/task,
    # 4.19x) over a fixed-complexity fabric: measured core links/route 3.13 -> 3.02
    # from 20 to 80 servers, i.e. more nodes hang off the same ring without lengthening
    # routes. The claim this corpus can support is therefore "generalizes across
    # candidate-set growth", NOT "generalizes to larger networks" -- narrower, and
    # labelled as such. Scaling n_core is defensible but untested against Phase 2's
    # aggregation-invariance property (GIN sums, so any degree growing with N shifts
    # embedding magnitudes with N); it would need the degree-bound asserts re-run at
    # every rung, which is a separate phase with its own budget.
    "backbone_defaults": {"link_bandwidth_mbps": 1000.0},
}

# netc_scarce_v1: shallow_v1 queues + a SCARCE candidate set, for network_contention_v1.
#
# The 12-dataset matched pilot showed ingress contention moves every M4 metric monotonically
# with bandwidth (additive R^2 0.9667 -> 0.9596 -> 0.9478 at unset/1.5/0.5 MB/s) but leaves
# M1 marginal-greedy regret at exactly 0% -- greedy finds the joint optimum 12/12 in every
# arm. The reason is structural, not statistical: tasks had 4.56 candidate NODES each and
# 0/12 datasets lacked a fully-spread plan, so co-location was never forced. A cost that
# only PENALISES co-location then pushes the optimum toward the corner greedy already picks.
#
# For a joint decision to exist, tasks must compete for scarce good options. This grid cuts
# connectivity and holds replicas at one per client/server so the candidate set per task
# shrinks toward the batch size.
NETC_SCARCE_V1_GRID: GridPreset = {
    "connection_probabilities": [0.12, 0.18],
    "replica_configs": [
        (1, 1, 0.7, 0.9),
    ],
    "queue_distributions": [
        ("shallow_pois2", "poisson", 2, 0, 0, 8, 1),
        ("shallow_uniform0_4", "uniform", 0, 4, 0, 8, 1),
        ("shallow_norm3", "normal", 3, 1, 0, 8, 1),
    ],
    "seeds": list(range(701, 751)),
    "default_output_subdir": "gnn_datasets_4tasks_netc_scarce_v1",
}

# netc_funnel_v1: shallow queues + a FUNNELING topology, for network_contention_v1.
#
# netc_scarce_v1 established that cutting candidate COUNT does not create a joint decision
# (4.56 -> 3.23 candidate nodes/task left M1 at 0%). The spreading-slack pre-check explains
# why: what matters is whether tasks' cheap sets OVERLAP, not how large they are. Four tasks
# with three candidates each still spread perfectly if those sets are disjoint -- and they
# were: each task's single favourite node was already distinct in ~3.9 of 4 tasks, so
# theta* (the premium needed to spread) was 0 in 92% of datasets.
#
# degree_skewed_core makes a few core nodes cheap for MANY clients at once (latency_core_ms
# 5 vs periphery 30, p_core 0.95 vs p_periphery 0.15), so the cheap sets collide by
# construction. Measured on the existing skew_warmth_v2 corpus: free-spreading drops to
# 61.2% (vs 76.0% on shallow_v1) and 28/98 datasets need a >25% premium to spread (vs
# 11/200). Hub-seeker fractions here are pushed above that corpus's 0.35-0.65 to sharpen it.
NETC_FUNNEL_V1_GRID: GridPreset = {
    "topology_type": "degree_skewed_core",
    "k_core_values": [3, 4],
    "hub_seeker_fractions": [0.70, 0.90],
    "latency_core_ms": 5,
    "latency_periphery_ms": 30,
    "connection_probabilities": [],
    "replica_configs": [
        (1, 1, 0.7, 0.9),
    ],
    "queue_distributions": [
        ("shallow_pois2", "poisson", 2, 0, 0, 8, 1),
        ("shallow_uniform0_4", "uniform", 0, 4, 0, 8, 1),
        ("shallow_norm3", "normal", 3, 1, 0, 8, 1),
    ],
    "seeds": list(range(701, 751)),
    "default_output_subdir": "gnn_datasets_4tasks_netc_funnel_v1",
}

# netc_hotspot_v1: shallow queues + DENSE connectivity + replicas concentrated on few hosts.
#
# This inverts the two failed attempts. netc_scarce_v1 (cut connectivity) and
# netc_funnel_v1 (funnel to hubs) both left M1 at 0%, and the overlap measurement says why:
# they made tasks' candidate sets more DISJOINT, not more shared -- mean pairwise overlap
# fell 0.93 -> 0.36 -> 0.14 of 4 tasks, so every task kept a private favourite node and
# spreading stayed free (theta* = 0 in 92-100% of datasets).
#
# Overlap needs the opposite: dense connectivity so every client can reach every host, and
# FEW hosts so they must all use the same ones. The blocker was
# generate_infrastructure.py's `replica_server_pct = max(server_pct, 0.6)` floor, which
# spread replicas over >=60% of servers no matter what the grid asked for; this grid sets
# preinit.replica_server_percentage to override it.
NETC_HOTSPOT_V1_GRID: GridPreset = {
    "connection_probabilities": [0.85],
    "replica_configs": [
        (0, 1, 0.2, 0.15),
    ],
    "replica_server_percentage": 0.15,
    "queue_distributions": [
        ("shallow_pois2", "poisson", 2, 0, 0, 8, 1),
        ("shallow_uniform0_4", "uniform", 0, 4, 0, 8, 1),
        ("shallow_norm3", "normal", 3, 1, 0, 8, 1),
    ],
    "seeds": list(range(701, 751)),
    "default_output_subdir": "gnn_datasets_4tasks_netc_hotspot_v1",
}

CONTENTION_V2_GRID: GridPreset = {
    "connection_probabilities": [0.25, 0.35],
    "replica_configs": [
        (1, 1, 0.7, 0.9),
        (1, 2, 0.7, 0.9),
        (2, 2, 0.5, 0.7),
    ],
    "queue_distributions": [
        ("norm35", "normal", 35, 11, 0, 96, 1),
        ("uniform20_80", "uniform", 20, 80, 0, 120, 1),
        ("pois28", "poisson", 28, 0, 0, 72, 1),
    ],
    "seeds": list(range(301, 351)),
    "default_output_subdir": "gnn_datasets_4tasks_contention_v2",
}

# contention_v4_deepq: MATCH THE LIVE QUEUE REGIME.
# Measured 2026-08-13: the live failure of 873/v5.5 GNN/MLP is pure queueing, not pull
# serialization — live GNN sparse_p35 s42 has averageQueueTime 503.4s out of 503.4s
# averageElapsedTime, with averagePullTime 0.017s and averageColdStartTime 0.0003s.
# Meanwhile every contention_v2 label is warm-path: 0/900 datasets have any cold start or
# pull, and optimum averageQueueTime spans only 0.27-5.47s. The trained models therefore
# never see a queue deep enough for co-locating a batch to be catastrophic, which is exactly
# the mistake they make live (penaltyProportion ~80%).
# Lever: keep contention_v2's scarce-warm attractor (a few clearly-best warm platforms) but
# push pre-seeded queue depth deep enough that stacking two tasks on the attractive platform
# costs tens of seconds, so the optimum MUST split them.
# Depth is calibrated to the queue band the WINNING live policy occupies, not the failing one:
# live seed-42 averageQueueTime is Knative 7.7/30.6/44.9s vs MLP 25.2/79.5/216.1s vs GNN
# 87.9/210.0/503.4s on skew/p25/p35. Target label band ~5-50s => ~10x contention_v2's mean 35
# warmup tasks (which yielded only 0.27-5.47s). Overshooting to mean 1200 was measured at
# >150s/dataset locally because warmup tasks are materialized per placement, so keep the mean
# in the hundreds.
# MUST be generated with --allow-non-unique-replicas.
CONTENTION_V4_DEEPQ_GRID: GridPreset = {
    "connection_probabilities": [0.25, 0.35],
    "replica_configs": [
        (1, 1, 0.7, 0.9),
        (1, 2, 0.7, 0.9),
        (2, 2, 0.5, 0.7),
    ],
    "queue_distributions": [
        ("norm350", "normal", 350, 120, 0, 900, 1),
        ("uniform150_700", "uniform", 150, 700, 0, 900, 1),
        ("pois400", "poisson", 400, 0, 0, 900, 1),
    ],
    "seeds": list(range(601, 651)),
    "default_output_subdir": "gnn_datasets_4tasks_contention_v4_deepq",
}

# contention_v5_quick_test: QUICK TEST for deep queues + coupling optimization.
# Hypothesis: norm450/pois500 (12.9x deeper than v2's norm35) + scarce warm resources
# should push coupling rate from v2's 7.1% to 15-25%, giving GNN measurable advantage.
# Target: 216 datasets, 7-10 min walltime with 15 parallel workers on datalab CPU.
# MUST be generated with --allow-non-unique-replicas.
CONTENTION_V5_QUICK_TEST_GRID: GridPreset = {
    "connection_probabilities": [0.25, 0.30, 0.35],
    "replica_configs": [
        (1, 1, 0.7, 0.9),  # Minimal replicas, high warmth (strongest scarcity)
        (1, 2, 0.7, 0.9),  # Slightly more replicas, high warmth
    ],
    "queue_distributions": [
        ("norm450", "normal", 450, 150, 0, 1200, 1),
        ("uniform200_800", "uniform", 200, 800, 0, 1200, 1),
        ("pois500", "poisson", 500, 0, 0, 1200, 1),
    ],
    "seeds": list(range(701, 713)),  # 12 seeds
    "default_output_subdir": "gnn_datasets_4tasks_contention_v5_quick_test",
}

# contention_v3: push coupling further — sparser topology + heavier queues.
# Target: coupled (>1% greedy regret) fraction >25% (v2 was 7.2%).
# MUST be generated with --allow-non-unique-replicas.
CONTENTION_V3_GRID: GridPreset = {
    "connection_probabilities": [0.15, 0.20],
    "replica_configs": [
        (1, 1, 0.7, 0.9),
        (1, 2, 0.7, 0.9),
        (2, 2, 0.5, 0.7),
    ],
    "queue_distributions": [
        ("norm40", "normal", 40, 13, 0, 110, 1),
        ("uniform25_90", "uniform", 25, 90, 0, 130, 1),
        ("pois32", "poisson", 32, 0, 0, 84, 1),
    ],
    "seeds": list(range(401, 451)),
    "default_output_subdir": "gnn_datasets_4tasks_contention_v3",
}

# regime_b_cold_burst_v1: training labels for the frozen Regime B problem.
# Live gate is N=12 under platform_reuse_v1 (see archive/regime_b/scripts_cosim/regime_b_problem_spec.py).
# Co-sim stays at 4-task BF (placement space); physics + scarce-warm lever match live.
# MUST use --warmth-physics platform_reuse_v1 (node_disk_v2 kills FilterStore headroom).
# MUST --allow-non-unique-replicas. Cartesian: 2×3×3×25 = 450.
REGIME_B_COLD_BURST_V1_GRID: GridPreset = {
    "connection_probabilities": [0.25, 0.35],
    "replica_configs": [
        (1, 1, 0.7, 0.9),
        (1, 2, 0.7, 0.9),
        (2, 2, 0.5, 0.7),
    ],
    "queue_distributions": [
        ("norm35", "normal", 35, 11, 0, 96, 1),
        ("uniform20_80", "uniform", 20, 80, 0, 120, 1),
        ("pois28", "poisson", 28, 0, 0, 72, 1),
    ],
    "seeds": list(range(501, 526)),
    "default_output_subdir": "gnn_datasets_4tasks_regime_b_cold_burst_v1",
    "required_warmth_physics": "platform_reuse_v1",
}

GRID_PRESETS: Dict[str, GridPreset] = {
    "warmth_v2": WARMTH_V2_GRID,
    "sparse_warmth_v2": SPARSE_WARMTH_V2_GRID,
    "skew_warmth_v2": SKEW_WARMTH_V2_GRID,
    "contention_v1": CONTENTION_V1_GRID,
    "contention_v2": CONTENTION_V2_GRID,
    "shallow_v1": SHALLOW_V1_GRID,
    "netc_multihop_v1": NETC_MULTIHOP_V1_GRID,
    "topo_transfer_v1": TOPO_TRANSFER_V1_GRID,
    "netc_scarce_v1": NETC_SCARCE_V1_GRID,
    "netc_funnel_v1": NETC_FUNNEL_V1_GRID,
    "netc_hotspot_v1": NETC_HOTSPOT_V1_GRID,
    "shallow_longexec_v1": SHALLOW_LONGEXEC_V1_GRID,
    "contention_v3": CONTENTION_V3_GRID,
    "contention_v4_deepq": CONTENTION_V4_DEEPQ_GRID,
    "contention_v5_quick_test": CONTENTION_V5_QUICK_TEST_GRID,
    "regime_b_cold_burst_v1": REGIME_B_COLD_BURST_V1_GRID,
    "route_a_pilot_v1": ROUTE_A_PILOT_V1_GRID,
    "route_b_pilot_v1": ROUTE_B_PILOT_V1_GRID,
    "route_b_pilot_v1_8task": ROUTE_B_PILOT_V1_8TASK_GRID,
    "route_c_link_screen": ROUTE_C_LINK_SCREEN_GRID,
    "route_c_link_screen_8task": ROUTE_C_LINK_SCREEN_8TASK_GRID,
    "route_b_pivot_h0": ROUTE_B_PIVOT_H0_GRID,
    "route_b_pivot_h1": ROUTE_B_PIVOT_H1_GRID,
    "route_b_pivot_h2": ROUTE_B_PIVOT_H2_GRID,
    "route_b_pivot_h3": ROUTE_B_PIVOT_H3_GRID,
}


def resolve_grid_preset(grid_name: str) -> GridPreset:
    if grid_name not in GRID_PRESETS:
        known = ", ".join(sorted(GRID_PRESETS))
        raise ValueError(f"Unknown grid {grid_name!r}; expected one of: {known}")
    return GRID_PRESETS[grid_name]


def grid_server_node_counts(preset: GridPreset) -> List[Optional[int]]:
    """The topology-size axis. `[None]` means "leave the base config's count alone".

    Every grid written before topology_transfer_v1 omits the key and therefore keeps
    generating at the base config's 20 servers, unchanged.
    """
    counts = preset.get("server_node_counts")
    if not counts:
        return [None]
    return list(counts)


def grid_total_datasets(preset: GridPreset) -> int:
    # grid_topology_variants already crosses shape x size, so it is the single source of
    # truth for the topology axis -- do not multiply the size axis in again here.
    return (
        len(grid_topology_variants(preset))
        * len(preset["replica_configs"])
        * len(preset["seeds"])
        * len(preset["queue_distributions"])
    )


def _grid_topology_shape_variants(preset: GridPreset) -> List[Tuple[str, Dict[str, Any]]]:
    """The topology *shape* axis, before the size axis is crossed in."""
    if preset.get("topology_type") == "degree_skewed_core":
        variants: List[Tuple[str, Dict[str, Any]]] = []
        for k_core in preset["k_core_values"]:
            for hub_frac in preset["hub_seeker_fractions"]:
                seek_pct = int(round(hub_frac * 100))
                variants.append(
                    (
                        f"k{k_core}_seek{seek_pct:02d}",
                        {
                            "topology_type": "degree_skewed_core",
                            "k_core": k_core,
                            "hub_seeker_fraction": hub_frac,
                            "latency_core_ms": preset.get("latency_core_ms", 5),
                            "latency_periphery_ms": preset.get("latency_periphery_ms", 30),
                        },
                    )
                )
        return variants
    return [
        (
            f"conn={conn_prob}",
            {"topology_type": "erdos_renyi", "connection_prob": conn_prob},
        )
        for conn_prob in preset["connection_probabilities"]
    ]


def grid_topology_variants(preset: GridPreset) -> List[Tuple[str, Dict[str, Any]]]:
    """Ordered (label, kwargs for create_config_for_iteration topology fields).

    Topology shape x topology size. The size axis rides here rather than as its own loop
    level because these kwargs are already splatted straight into
    `create_config_for_iteration`, so `server_node_count` needs no separate plumbing and
    the size lands in the dataset label for free.
    """
    variants: List[Tuple[str, Dict[str, Any]]] = []
    for shape_label, shape_kwargs in _grid_topology_shape_variants(preset):
        for server_count in grid_server_node_counts(preset):
            if server_count is None:
                variants.append((shape_label, dict(shape_kwargs)))
            else:
                variants.append(
                    (
                        f"{shape_label},srv={server_count}",
                        {**shape_kwargs, "server_node_count": server_count},
                    )
                )
    return variants

# Task type ratios: (dnn1%, dnn2%)
TASK_TYPE_RATIOS = [
    (0, 100), (50, 50), (100, 0)
]

# Which two task types the workload mixes, in TASK_TYPE_RATIOS proportions.
# Grid presets may override via "task_type_pair". The default (dnn1, dnn2) is what every
# existing corpus was generated with.
#
# Why this is a lever: the coupling in the co-sim target is `added_in_batch x exec_time`,
# so it scales with execution time. dnn2 runs in 0.024s on xavierCpu, while cnn runs in
# 0.706s there and 3.086s on rpiCpu -- 30-130x more interaction per collision.
DEFAULT_TASK_TYPE_PAIR = ("dnn1", "dnn2")

# Workload parameters (can be overridden via --num-tasks)
NUM_TASKS = 4
NUM_CLIENT_NODES = 20  # matches space_with_network.json client_nodes.count
NUM_WORKLOAD_TEMPLATES = 10

# Seed for the workload task-source draw. Fixed so a grid regenerates identically and
# matched A/B arms (e.g. network_contention_v1's baseline vs ingress-bandwidth arms)
# differ only in the variable under test. Override with --workload-seed to resample.
DEFAULT_WORKLOAD_SEED = 42


def log(msg: str, quiet: bool = False, force: bool = False):
    """Print message unless in quiet mode."""
    if not quiet or force:
        print(msg)


def _run_shard_tag() -> str:
    """Stable per-process tag so concurrent SLURM array shards / local procs don't collide."""
    job_id = (
        os.environ.get("SLURM_ARRAY_JOB_ID")
        or os.environ.get("SLURM_JOB_ID")
        or str(os.getpid())
    )
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", str(os.getpid()))
    return f"{job_id}_{task_id}"


def workload_templates_dir_for_run(sim_input_path: Path) -> Path:
    """Per-SLURM-task template dir — parallel array shards must not share gnn_templates/."""
    return sim_input_path / "traces" / f"gnn_templates_{_run_shard_tag()}"


def workload_base_file_for_run(sim_input_path: Path) -> Path:
    """Per-shard workload-10 copy. The BF reads this path; concurrent array shards each
    overwrite their OWN copy so they cannot clobber each other's workload mid-run."""
    return sim_input_path / "traces" / f"workload-10_{_run_shard_tag()}.json"


def _diamond4_dag(task_types: Sequence[str]) -> Dict[str, List[str]]:
    """`A -> {B, C} -> D` over four DISTINCT task types.

    Distinct because `Orchestrator.create_application` keys tasks by function name and
    looks each one up in `task-types.json`, so a dag cannot use the same type twice.

    Why a diamond rather than a plain fan-out: the fan-out gives siblings that are
    co-decidable (route A needs the parent and child in one jointly-decided plan), and the
    fan-in gives a genuine `max` over *coupled* branch costs — which is where the
    composition theorem stops applying, and the only place Decima's `g(·)` argument
    actually transfers. A pure fan-out has no fan-in term at all.
    """
    if len(set(task_types)) != 4:
        raise ValueError(
            f"diamond4 needs 4 distinct task types, got {task_types!r}; a DAG node name "
            f"must be a unique key in task-types.json"
        )
    a, b, c, d = task_types
    return {a: [], b: [a], c: [a], d: [b, c]}


DAG_SHAPES = {"diamond4": _diamond4_dag}


def _draw_demand_scale(rng: random.Random, demand_spread: Optional[Dict[str, Any]]) -> float:
    """One seeded per-instance demand_scale draw. Absent config (None) -> 1.0, so a
    dataset generated without demand_spread is byte-identical to before this option
    existed — no rng.* call happens at all, and downstream demand = 1.0 * table value.

    dist='uniform': params [low, high]. Kept intentionally minimal (route_b env pivot
    W2 rung H1 uses uniform [0.5, 2.0]); extend with more dists only when a rung needs
    one, matching the rest of this file's grid-key conventions."""
    if demand_spread is None:
        return 1.0
    dist = demand_spread["dist"]
    params = demand_spread["params"]
    if dist == "uniform":
        low, high = params
        return rng.uniform(low, high)
    raise ValueError(f"unknown demand_spread dist {dist!r}; known: ['uniform']")


def generate_workload_templates(
    base_workload_path: Path,
    output_dir: Path,
    num_templates: int = NUM_WORKLOAD_TEMPLATES,
    quiet: bool = False,
    task_type_pair: Tuple[str, str] = DEFAULT_TASK_TYPE_PAIR,
    workload_seed: int = DEFAULT_WORKLOAD_SEED,
    dag_shape: Optional[str] = None,
    dag_task_types: Optional[Sequence[str]] = None,
    dag_instances: int = 1,
    demand_spread: Optional[Dict[str, Any]] = None,
) -> List[Path]:
    """
    Generate workload templates with varied task type ratios.

    Task source nodes are drawn from a LOCAL seeded RNG. They previously came from the
    unseeded global `random`, which meant two runs of the same grid with the same seeds
    produced different workloads and therefore different RTTs — corpora were not
    reproducible from their recorded seed, and matched A/B arms could not be built at all
    (any measured difference would be confounded by a different workload draw). A local
    Random keeps this independent of any other module's use of the global RNG.

    `demand_spread` (route_b env pivot W2, label-side only): when set, draws a seeded
    per-DAG-task-instance `demand_scale` from this same local rng, written into each
    event's `application.demand_scale` dict ({task_type_name: scale}). Absent (default
    None) -> no draw happens, no key is written, and generated datasets are byte-
    identical to before this option existed. Only wired for the dag_shape path (the
    non-DAG legacy path has no route_b consumer and is left untouched).

    Returns list of paths to generated template files.
    """
    rng = random.Random(workload_seed)
    with open(base_workload_path, 'r') as f:
        base_workload = json.load(f)
    
    templates = []
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for template_idx in range(num_templates):
        # Cycle through task type ratios
        # The type-ratio axis is meaningless for a DAG: its node types are fixed by the
        # shape, and varying their mix would change the dag rather than the workload.
        if dag_shape:
            task_types = list(dag_task_types or ())
            num_first = num_second = 0
            first_name = second_name = ""
        else:
            first_pct, _second_pct = TASK_TYPE_RATIOS[template_idx % len(TASK_TYPE_RATIOS)]
            first_name, second_name = task_type_pair

            num_first = NUM_TASKS * first_pct // 100
            num_second = NUM_TASKS - num_first

            # Create task types list
            task_types = [first_name] * num_first + [second_name] * num_second
        
        # Random client node assignments
        client_nodes = [rng.randint(0, NUM_CLIENT_NODES - 1) for _ in range(NUM_TASKS)]
        
        # Create workload with improved duration for queue accumulation
        workload = {
            'rps': base_workload.get('rps', 1),
            'duration': 1,
            'events': []
        }
        
        base_events = base_workload.get('events', [])
        if dag_shape:
            # ONE application containing a real DAG, instead of NUM_TASKS independent
            # single-task applications. This is what makes the tasks co-decidable: co-sim
            # enumerates a placement_plan over every task in the event before the episode
            # runs, so a parent and its child are chosen jointly. If they were decided
            # separately, "distance from the parent's node" would be ordinary known state
            # and a pointwise model would recover optimality — see route_a in LINEAGES.
            if dag_shape not in DAG_SHAPES:
                raise ValueError(f"unknown dag_shape {dag_shape!r}; known: {sorted(DAG_SHAPES)}")
            types = list(dag_task_types or [])
            # Multiple instances of the same DAG shape land in ONE workload's events list
            # and are therefore co-decided by the same co-sim episode (see route_b_v1
            # 8-task probe, LINEAGES.md) — the tasks across instances compete for the same
            # platforms, not just within one DAG.
            #
            # Instance client nodes are INDEPENDENT draws, not distinct ones: two instances
            # share a client node in ~1/NUM_CLIENT_NODES of templates. That is deliberate.
            # Distinctness would be the wrong trade twice over:
            #   - it buys nothing physically. route_b's grid sets per_client = 0, so no task
            #     runs on its own client node — every task crosses the network regardless of
            #     which client emitted it. A shared source node is not a shared *host*.
            #   - it would cost matched-seed comparability. `inst = 0` draws client_nodes[0],
            #     which is exactly what the 1-instance (4-task) corpus draws for its single
            #     instance at the same workload_seed, so the 8-task corpus is a strict
            #     perturbation of the 4-task one: instance 0 identical, instance 1 added.
            #     Resampling to force distinctness would move instance 0 too, and the
            #     4-vs-8-task closure comparison is the entire point of the probe.
            for inst in range(dag_instances):
                dag = DAG_SHAPES[dag_shape](types)
                base_event = deepcopy(base_events[0])
                base_event['application']['name'] = f"nofs-{dag_shape}"
                base_event['application']['dag'] = dag
                base_event['node_name'] = f"client_node{client_nodes[inst % len(client_nodes)]}"
                if demand_spread is not None:
                    # Drawn per (template, instance, task type) from the SAME local rng
                    # stream client_nodes already consumes — deterministic from
                    # workload_seed, and drawn in a fixed order (types, sorted) so the
                    # stream position is independent of dict iteration order.
                    base_event['application']['demand_scale'] = {
                        ttype: _draw_demand_scale(rng, demand_spread)
                        for ttype in sorted(types)
                    }
                workload['events'].append(base_event)
        else:
            for idx in range(NUM_TASKS):
                base_event = deepcopy(base_events[idx % len(base_events)])
                task_type = task_types[idx]
                client_node = client_nodes[idx]

                base_event['application']['name'] = f"nofs-{task_type}"
                base_event['application']['dag'] = {task_type: []}
                base_event['node_name'] = f"client_node{client_node}"

                workload['events'].append(base_event)
        
        # Save template
        template_path = output_dir / f"workload_template_{template_idx}.json"
        with open(template_path, 'w') as f:
            f.write(json_dumps_pretty(workload))
        
        templates.append(template_path)
        
        if not quiet:
            if dag_shape:
                log(f"  Template {template_idx}: {dag_shape} over {list(task_types)}")
            else:
                log(
                    f"  Template {template_idx}: {num_first} {first_name} "
                    f"+ {num_second} {second_name}"
                )
    
    return templates


def create_config_for_iteration(
    base_config: Dict[str, Any],
    replica_cfg: Tuple[int, int, float, float],
    seed: int,
    queue_dist: Tuple[str, str, int, int, int, int, int],
    batch_size: int = 4,
    topology_type: str = "erdos_renyi",
    connection_prob: Optional[float] = None,
    k_core: Optional[int] = None,
    hub_seeker_fraction: Optional[float] = None,
    latency_core_ms: float = 5.0,
    latency_periphery_ms: float = 30.0,
    task_type_pair: Tuple[str, str] = DEFAULT_TASK_TYPE_PAIR,
    replica_server_percentage: Optional[float] = None,
    server_node_count: Optional[int] = None,
    dag_task_types: Optional[Sequence[str]] = None,
    replica_overlap: bool = False,
) -> Dict[str, Any]:
    """
    Create a modified config for a specific iteration.
    
    Args:
        batch_size: Batch size for determined scheduler (should match num_tasks)
    """
    config = deepcopy(base_config)
    
    per_client, per_server, client_pct, server_pct = replica_cfg
    qname, qtype, qp1, qp2, qmin, qmax, qstep = queue_dist
    
    # Network topology
    if 'network' not in config:
        config['network'] = {}
    if 'topology' not in config['network']:
        config['network']['topology'] = {}
    topo = config['network']['topology']
    if topology_type == "degree_skewed_core":
        if k_core is None or hub_seeker_fraction is None:
            raise ValueError(
                "degree_skewed_core requires k_core and hub_seeker_fraction"
            )
        topo.update(
            {
                "type": "degree_skewed_core",
                "k_core": k_core,
                "hub_seeker_fraction": hub_seeker_fraction,
                "p_core": 0.95,
                "p_periphery": 0.15,
                "latency_core_ms": latency_core_ms,
                "latency_periphery_ms": latency_periphery_ms,
                "seed": seed,
            }
        )
    else:
        if connection_prob is None:
            raise ValueError("erdos_renyi topology requires connection_prob")
        topo["connection_probability"] = connection_prob
        topo["seed"] = seed
    
    # Topology SIZE. Only the server tier scales: servers are the placement substrate, so
    # they are the axis a transfer study is actually about. Holding the client tier fixed
    # keeps the workload's task-source draw identical across sizes, so matched arms differ
    # in candidate-set size and nothing else -- scaling clients too would move the tasks
    # themselves and confound the comparison. Unset leaves the base config untouched, so
    # every pre-existing grid regenerates bit-identically.
    if server_node_count is not None:
        if server_node_count < 1:
            raise ValueError(f"server_node_count must be >= 1, got {server_node_count}")
        config.setdefault('nodes', {}).setdefault('server_nodes', {})['count'] = int(
            server_node_count
        )

    # Preinit configuration
    config['preinit'] = {
        'client_percentage': client_pct,
        'server_percentage': server_pct
    }
    # Optional: concentrate replicas onto few server hosts, overriding the 0.6 spreading
    # floor in generate_infrastructure. This is what makes tasks compete for the SAME nodes
    # rather than each owning a private favourite.
    if replica_server_percentage is not None:
        config['preinit']['replica_server_percentage'] = replica_server_percentage
    # route_b env pivot (2026-08-27), W3: task types may share replica hosts/platforms
    # (generate_infrastructure.py's preinit.replica_overlap). Default False -> the key
    # is never written, so every existing grid reproduces byte-identically.
    if replica_overlap:
        config['preinit']['replica_overlap'] = True

    # Replica configuration. Keyed by the grid's task types, not a hardcoded dnn1/dnn2 --
    # this dict REPLACES whatever main() synthesized, so hardcoding it left a grid with a
    # substituted pair holding replicas for task types its workload never asks for.
    # A DAG's node types all need replicas: every one of them is a real task that has to
    # land somewhere, and a type with no replica anywhere fails the episode with
    # "No valid replicas for task N".
    replica_task_types = tuple(dag_task_types) if dag_task_types else task_type_pair
    config['replicas'] = {
        task_type: {'per_client': per_client, 'per_server': per_server}
        for task_type in replica_task_types
    }
    
    # Queue distribution parameters
    if qtype == "constant":
        q_params = {'type': 'constant', 'value': qp1, 'min': qmin, 'max': qmax, 'step': qstep}
    elif qtype == "poisson":
        q_params = {'type': 'poisson', 'lambda': qp1, 'min': qmin, 'max': qmax, 'step': qstep}
    elif qtype == "normal":
        stddev = qp2 if qp2 != 0 else 1
        q_params = {'type': 'normal', 'mean': qp1, 'stddev': stddev, 'min': qmin, 'max': qmax, 'step': qstep}
    elif qtype == "uniform":
        q_params = {'type': 'uniform', 'low': qp1, 'high': qp2, 'min': qmin, 'max': qmax, 'step': qstep}
    else:
        q_params = {'type': 'poisson', 'lambda': 4, 'min': qmin, 'max': qmax, 'step': qstep}
    
    config['prewarm'] = {
        task_type: {
            'distribution': 'none',
            'queue_distribution': 'statistical',
            'queue_distribution_params': q_params
        }
        for task_type in replica_task_types
    }
    
    # Set scheduler batch_size to match num_tasks (for determined scheduler)
    # This ensures scheduler processes tasks in batches matching the workload
    if 'scheduler' not in config:
        config['scheduler'] = {}
    config['scheduler']['batch_size'] = batch_size
    config['scheduler']['batch_timeout'] = 0.1
    
    return config


def generate_single_dataset(
    dataset_id: str,
    output_dir: Path,
    config: Dict[str, Any],
    workload_template: Path,
    sim_input_path: Path,
    sample_json_file: Path,
    samples_file: Path,
    mapping_file: Path,
    seed: int,
    max_workers: int,
    quiet: bool = False,
    fast_forward_warmup: bool = True,
    fast_forward_threshold: int = 1,
    allow_non_unique_replicas: bool = True,
    warmth_physics: str = "node_disk_v2",
) -> Tuple[bool, float, float]:
    """
    Generate a single GNN dataset.
    
    Returns (success, rtt, duration_seconds)
    """
    start_time = time.time()
    
    try:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        if os.environ.get("GNN_CAPTURE_DATASET_STATE", "0") != "1":
            ssc_stale = output_dir / "system_state_captured_unique.json"
            if ssc_stale.exists():
                ssc_stale.unlink()
        
        # Save config for this dataset
        config_path = output_dir / "space_with_network.json"
        with open(config_path, 'w') as f:
            f.write(json_dumps_pretty(config))
        
        # Copy workload template
        workload_path = output_dir / "workload.json"
        with open(workload_template, 'r') as f:
            workload = json.load(f)
        with open(workload_path, 'w') as f:
            f.write(json_dumps_pretty(workload))
        
        # Copy workload to a per-shard location for simulation (parallel-array safe).
        traces_dir = sim_input_path / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)
        shard_workload_file = workload_base_file_for_run(sim_input_path)
        with open(shard_workload_file, 'w') as f:
            f.write(json_dumps_pretty(workload))
        
        # Generate infrastructure
        infra_file = output_dir / "infrastructure.json"
        log(f"  Generating infrastructure...", quiet)
        generate_deterministic_infrastructure(
            str(config_path),
            sim_input_path,
            str(infra_file),
            seed
        )
        
        # Load one scenario sample (JSON preferred, .npy/.pkl fallback)
        sample, mapping, sample_source = load_primary_sample_and_mapping(
            sample_json_path=sample_json_file,
            samples_npy_path=samples_file,
            mapping_pkl_path=mapping_file,
        )
        log(f"  Sample source: {sample_source}", quiet)
        
        # Load apps from config
        apps = list(config['wsc'].keys())
        # A grid naming new task types gets synthesized wsc entries but no sampled workload
        # factor; grow this run's copy of the sample rather than the shared input files.
        sample, mapping = ensure_workload_params(sample, mapping, apps)
        
        # Per-dataset scratch dir — parallel-safe (shared initial_results_simple races across shards).
        # placements.jsonl lives here during BF; MUST be copied to placements/ before rmtree.
        results_dir = output_dir / ".bf_scratch"
        if results_dir.exists():
            shutil.rmtree(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Run optimized brute-force simulation
        # NOTE: No timeout wrapper here because execute_brute_force_optimized uses ProcessPoolExecutor internally.
        # If it hangs, the process will need to be killed externally. The simulation itself should complete
        # within reasonable time for most datasets. If it hangs, check for deadlocks in the simulation code.
        log(f"  Running brute-force optimization (max {SIMULATION_TIMEOUT}s expected)...", quiet)
        
        sim_start = time.time()
        suppress_sim_prints = (
            os.environ.get("COSIM_SUPPRESS_SIM_PRINTS", "0") == "1"
            or os.environ.get("GNN_CAPTURE_DATASET_STATE", "0") != "1"
        )
        bf_kwargs = dict(
            apps=apps,
            config_file=str(config_path),
            mapping_file=str(mapping_file),
            output_dir=results_dir,
            sample=sample,
            sim_input_path=sim_input_path,
            workload_base_file=str(shard_workload_file),
            max_workers=max_workers,
            infrastructure_file=infra_file,
            quiet=quiet,
            final_dataset_dir=output_dir,  # Write progress files to final dataset directory
            fast_forward_warmup=fast_forward_warmup,
            fast_forward_threshold=fast_forward_threshold,
            allow_non_unique_replicas=allow_non_unique_replicas,
            mapping_override=mapping,
            warmth_physics=warmth_physics,
        )
        if suppress_sim_prints:
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                result_paths = execute_brute_force_optimized(**bf_kwargs)
        else:
            result_paths = execute_brute_force_optimized(**bf_kwargs)
        sim_duration = time.time() - sim_start
        
        # Warn if simulation took too long (but don't fail - it might be legitimate)
        if sim_duration > SIMULATION_TIMEOUT:
            log(f"  WARNING: Simulation took {sim_duration:.1f}s (exceeded {SIMULATION_TIMEOUT}s threshold)", quiet, force=True)
        
        # Check for results and copy to dataset directory
        best_json = results_dir / "best.json"
        placements_file = results_dir / "placements.jsonl"
        
        if best_json.exists():
            # Copy results to dataset directory
            with open(best_json, 'r') as f:
                best_info = json.load(f)
            
            optimal_rtt = best_info.get('rtt', float('inf'))
            optimal_file = best_info.get('file', '')
            
            # Copy best.json
            with open(output_dir / "best.json", 'w') as f:
                f.write(json_dumps(best_info))
            
            # Copy optimal result (use stdlib json to handle numpy types)
            optimal_src = results_dir / optimal_file
            if optimal_src.exists():
                shutil.copy2(optimal_src, output_dir / "optimal_result.json")
                capture_state = os.environ.get("GNN_CAPTURE_DATASET_STATE", "0") == "1"
                ssc_path = output_dir / "system_state_captured_unique.json"
                if capture_state:
                    try:
                        from src.executecosimulation import build_system_state_captured
                        from src.placement.model import DataclassJSONEncoder

                        with open(output_dir / "optimal_result.json", "r") as opt_f:
                            optimal_data = json.load(opt_f)
                        opt_stats = optimal_data.get("stats")
                        if opt_stats and opt_stats.get("taskResults"):
                            captured_state = build_system_state_captured(opt_stats)
                            with open(ssc_path, "w") as ssc_f:
                                json.dump(captured_state, ssc_f, indent=2, cls=DataclassJSONEncoder)
                    except Exception as ssc_exc:
                        log(f"  WARNING: Failed to write system_state_captured_unique.json: {ssc_exc}", quiet, force=True)
                        raise RuntimeError(
                            f"SSC export failed for {output_dir.name}: {ssc_exc}"
                        ) from ssc_exc
                elif ssc_path.exists():
                    ssc_path.unlink()
            
            # REQUIRED: persist full placement sweep for RTT-hash / near-RTT training.
            pub_placements = output_dir / "placements" / "placements.jsonl"
            if not placements_file.exists():
                raise RuntimeError(
                    f"{dataset_id}: best.json exists but .bf_scratch/placements.jsonl missing — "
                    "cannot train counterfactual RTT; keep scratch and re-run BF"
                )
            (output_dir / "placements").mkdir(exist_ok=True)
            shutil.copy2(placements_file, pub_placements)
            if pub_placements.stat().st_size == 0:
                raise RuntimeError(
                    f"{dataset_id}: placements/placements.jsonl is empty after copy"
                )
            
            # Copy placement metadata if it exists (will be written by execute_brute_force_optimized)
            metadata_src = results_dir / "placement_metadata.json"
            if metadata_src.exists():
                shutil.copy2(metadata_src, output_dir / "placement_metadata.json")
            
            # Copy placement progress if it exists
            progress_src = results_dir / "placement_progress.txt"
            if progress_src.exists():
                shutil.copy2(progress_src, output_dir / "placement_progress.txt")
            
            # Only remove scratch after public JSONL is verified (see placements_jsonl_required.md)
            shutil.rmtree(results_dir, ignore_errors=True)

            duration = time.time() - start_time
            return 'success', optimal_rtt, duration
        else:
            # No results - check if this was an infeasible scenario (placements.jsonl empty or missing)
            duration = time.time() - start_time
            if placements_file.exists() and placements_file.stat().st_size == 0:
                # Empty placements file = skipped scenario. The engine records WHY in
                # skip_reason.json (infeasible_no_candidates vs too_many_combinations);
                # keep that distinction in the log and in the dataset dir.
                skip_reason_src = results_dir / "skip_reason.json"
                if skip_reason_src.exists():
                    try:
                        reason = json.loads(skip_reason_src.read_text()).get("reason", "unknown")
                    except (json.JSONDecodeError, OSError):
                        reason = "unreadable"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(skip_reason_src, output_dir / "skip_reason.json")
                else:
                    reason = "unknown (pre-skip_reason engine)"
                log(f"  SKIP REASON: {reason}", quiet, force=True)
                shutil.rmtree(results_dir, ignore_errors=True)
                return 'skipped', float('inf'), duration
            else:
                # Some other issue
                shutil.rmtree(results_dir, ignore_errors=True)
                return 'failed', float('inf'), duration
            
    except Exception as e:
        duration = time.time() - start_time
        log(f"  ERROR: {e}", quiet, force=True)
        # The message alone routinely says nothing useful ("System state capture FAILED"),
        # and the traceback is the only thing that names the line that actually raised.
        if os.environ.get("HEROSIM_TRACE_DATASET_ERRORS", "0") == "1":
            import traceback
            traceback.print_exc()
        return 'failed', float('inf'), duration


def main():
    parser = argparse.ArgumentParser(
        description="Optimized GNN Dataset Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--quiet', '-q', default=True, action='store_true',
                        help='Suppress per-placement logging (default: True)')
    parser.add_argument('--no-quiet', action='store_false', dest='quiet',
                        help='Disable quiet mode (show per-placement logging)')
    parser.add_argument('--max-datasets', '-n', type=int, default=300000,
                        help='Maximum number of datasets to generate')
    parser.add_argument('--workers', '-w', type=int, default=None,
                        help='Number of parallel workers (default: CPU count - 1)')
    parser.add_argument(
        '--resume',
        action='store_true',
        help=(
            'Skip datasets that already have best.json AND non-empty '
            'placements/placements.jsonl (JSONL required for RTT-hash training)'
        ),
    )
    parser.add_argument(
        '--only-missing-jsonl',
        action='store_true',
        help=(
            'Only re-run BF for datasets that have best.json but lack non-empty '
            'placements/placements.jsonl (skips tail datasets with no best.json)'
        ),
    )
    parser.add_argument('--start-from', type=int, default=0,
                        help='Start from dataset index (e.g., 118 to start from ds_00118)')
    parser.add_argument('--fast-forward-warmup', default=True, action='store_true',
                        help='Enable fast-forward warmup for queues > 1 task (default: True)')
    parser.add_argument('--no-fast-forward-warmup', action='store_false', dest='fast_forward_warmup',
                        help='Disable fast-forward warmup')
    parser.add_argument('--fast-forward-threshold', type=int, default=1,
                        help='Threshold for fast-forward warmup (default: 1)')
    parser.add_argument(
        '--warmth-physics',
        type=str,
        default='node_disk_v2',
        choices=['platform_reuse_v1', 'node_disk_v2'],
        help='Warmth model for co-sim labels (default: node_disk_v2)',
    )
    parser.add_argument('--allow-non-unique-replicas', action='store_true',
                        help='Allow multiple tasks to share the same replica')
    parser.add_argument('--compute-slots-per-node', type=int, default=None,
                        help='node_contention_v3: shared execution slots per node, so '
                             'co-located platforms contend. Unset keeps node_disk_v2 '
                             'physics (platforms fully independent).')
    parser.add_argument('--ingress-bandwidth-mbps', type=float, default=None,
                        help='network_contention_v1: shared inbound bandwidth (MB/s) per '
                             'node, so tasks placed on the same node serialize their '
                             'input transfers. Unset keeps node_disk_v2 physics (no '
                             'ingress pipe, no transmission time).')
    parser.add_argument('--link-bandwidth-mbps', type=float, default=None,
                        help='link_contention_v1: per-link capacity (MB/s) over a core '
                             'backbone, so tasks whose ROUTES cross a shared segment '
                             'serialize even when they land on different nodes. Unset '
                             'keeps node_disk_v2 physics (no backbone, one-hop latency).')
    parser.add_argument('--backbone-n-core', type=int, default=None,
                        help='link_contention_v1: core routers in the ring (default 12, '
                             'chosen by scripts_cosim/link_overlap_precheck.py; a grid '
                             'preset may carry its own in backbone_defaults["n_core"]).')
    parser.add_argument('--backbone-attach-degree', type=int, default=None,
                        help='link_contention_v1: cores each node attaches to (default 1; '
                             '2 lets paths diverge and collapses route overlap; preset: '
                             'backbone_defaults["attach_degree"]).')
    parser.add_argument('--backbone-chord-count', type=int, default=None,
                        help='link_contention_v1: chords across the core ring (default 0; '
                             'chords let traffic bypass shared segments; preset: '
                             'backbone_defaults["chord_count"]).')
    parser.add_argument('--backbone-rng-stream', choices=('legacy_v0', 'independent_v1'),
                        default='independent_v1',
                        help='Which rng stream draws backbone access-link jitter. '
                             'independent_v1 (default for new corpora) derives it from the '
                             'topology seed alone, so corpus and live generation agree '
                             'exactly (no --allow-backbone-latency-divergence waiver). '
                             'legacy_v0 reproduces pre-2026-08-22 corpora, whose jitter '
                             'stream was offset by the replica-reachability repair.')
    parser.add_argument('--replica-server-percentage', type=float, default=None,
                        help='Override the grid/default fraction of server nodes hosting '
                             'replicas. Lower concentrates replicas so tasks compete for '
                             'the same hosts (overlapping candidate sets). Overrides the '
                             '0.6 spreading floor in generate_infrastructure.')
    parser.add_argument('--workload-seed', type=int, default=DEFAULT_WORKLOAD_SEED,
                        help=f'Seed for the workload task-source draw (default: '
                             f'{DEFAULT_WORKLOAD_SEED}). Fixed so a grid regenerates '
                             f'identically and matched A/B arms differ only in the '
                             f'variable under test.')
    parser.add_argument('--num-tasks', type=int, choices=[1, 2, 3, 4, 5], default=4,
                        help='Number of tasks per workload (1-5). Sets batch_size accordingly.')
    parser.add_argument(
        '--grid',
        type=str,
        default='warmth_v2',
        choices=sorted(GRID_PRESETS),
        help='Dataset grid preset (default: warmth_v2)',
    )
    parser.add_argument(
        '--output-subdir',
        type=str,
        default=None,
        help='Output subdirectory under simulation_data (default from --grid preset)',
    )
    parser.add_argument(
        '--progress-log-name',
        type=str,
        default=None,
        help='Progress log filename under logs/ (default: progress_{num_tasks}tasks.txt or derived from --output-subdir)',
    )
    args = parser.parse_args()
    if args.only_missing_jsonl and not args.resume:
        args.resume = True

    if os.environ.get("COSIM_SUPPRESS_SIM_PRINTS", "0") == "1":
        logging.getLogger().setLevel(logging.ERROR)
        logging.getLogger("simulation").setLevel(logging.ERROR)
    
    # OOM guard for brute-force placement generation.
    # Keep user-provided value if already exported in environment.
    os.environ.setdefault(
        "MAX_PLACEMENT_COMBINATIONS_SKIP",
        str(MAX_PLACEMENT_COMBINATIONS_SKIP_DEFAULT),
    )
    
    quiet = args.quiet
    grid_preset = resolve_grid_preset(args.grid)
    required_physics = grid_preset.get("required_warmth_physics")
    if required_physics and args.warmth_physics != required_physics:
        raise SystemExit(
            f"FAIL LOUD: grid {args.grid!r} requires --warmth-physics {required_physics} "
            f"(got {args.warmth_physics!r}). Regime B / FilterStore headroom collapses under "
            f"node_disk_v2 same-image — see archive/regime_b/scripts_cosim/regime_b_problem_spec.py."
        )
    topology_variants = grid_topology_variants(grid_preset)
    replica_configs = grid_preset["replica_configs"]
    queue_distributions = grid_preset["queue_distributions"]
    seeds = grid_preset["seeds"]
    grid_total = grid_total_datasets(grid_preset)

    # max_datasets is relative to start_from (e.g., --start-from xyz --max-datasets 1 means generate ds_xyz only)
    max_datasets = args.start_from + args.max_datasets
    cpu_count = os.cpu_count()
    max_workers = args.workers or (cpu_count - 1 if cpu_count and cpu_count > 1 else 1)
    
    # Set NUM_TASKS based on argument
    global NUM_TASKS
    NUM_TASKS = args.num_tasks
    batch_size = NUM_TASKS  # Match batch_size to num_tasks
    
    # Paths
    base_dir = PROJECT_ROOT / "simulation_data"
    config_path = base_dir / "space_with_network.json"
    sample_json_file = base_dir / "sample_simple.json"
    default_output_subdir = grid_preset.get("default_output_subdir") or f"gnn_datasets_{NUM_TASKS}tasks"
    output_subdir = args.output_subdir or default_output_subdir
    output_base = base_dir / output_subdir
    sim_input_path = PROJECT_ROOT / "data" / "nofs-ids"
    samples_file = base_dir / "lhs_samples_simple.npy"
    mapping_file = base_dir / "lhs_samples_simple_mapping.pkl"
    workload_base_file = sim_input_path / "traces" / "workload-10.json"
    workload_templates_dir = workload_templates_dir_for_run(sim_input_path)
    if args.progress_log_name:
        progress_log_name = args.progress_log_name
    elif args.output_subdir:
        safe_subdir = re.sub(r'[^A-Za-z0-9_.-]+', '_', output_subdir)
        progress_log_name = f"progress_{safe_subdir}.txt"
    else:
        progress_log_name = f"progress_{NUM_TASKS}tasks.txt"
    progress_log = PROJECT_ROOT / "logs" / progress_log_name
    
    # Create directories
    output_base.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    
    log(f"=== Optimized GNN Dataset Generation ===", quiet)
    log(f"Grid: {args.grid} ({grid_total} combos)", quiet)
    log(f"Num tasks: {NUM_TASKS} (batch_size={batch_size})", quiet)
    log(f"Max datasets: {args.max_datasets} (up to ds_{max_datasets-1:05d})", quiet)
    log(f"Workers: {max_workers}", quiet)
    log(f"Using orjson: {HAS_ORJSON}", quiet)
    log(f"Quiet mode: {quiet}", quiet)
    log(
        f"MAX_PLACEMENT_COMBINATIONS_SKIP: {os.environ.get('MAX_PLACEMENT_COMBINATIONS_SKIP')}",
        quiet,
    )
    
    # Load base config
    with open(config_path, 'r') as f:
        base_config = json.load(f)

    # node_contention_v3: co-located platforms contend for a shared pool of node
    # execution slots. Left unset the corpus keeps node_disk_v2 physics exactly.
    # build_config() deepcopies base_config, so this propagates to every dataset.
    if args.compute_slots_per_node is not None:
        base_config.setdefault('nodes', {})['compute_slots_per_node'] = (
            args.compute_slots_per_node
        )
        log(
            f"node_contention_v3: {args.compute_slots_per_node} shared execution "
            f"slot(s) per node",
            quiet,
        )

    # network_contention_v1: inbound transfers to a node share one pipe, so co-placement
    # on a node costs queueing time. Left unset the corpus keeps node_disk_v2 physics.
    if args.ingress_bandwidth_mbps is not None:
        base_config.setdefault('nodes', {})['ingress_bandwidth_mbps'] = (
            args.ingress_bandwidth_mbps
        )
        log(
            f"network_contention_v1: {args.ingress_bandwidth_mbps} MB/s shared ingress "
            f"bandwidth per node",
            quiet,
        )

    # link_contention_v1: route every logical edge over a core backbone whose segments
    # have finite capacity. Unlike the ingress pipe this is not indexed by destination, so
    # two tasks on DIFFERENT nodes can contend — the coupling a node-occupancy count
    # cannot express. Left unset the corpus keeps node_disk_v2 physics.
    # A grid may REQUIRE a backbone (topo_transfer_v1 does: contract core_v1 on a
    # fabric-less corpus is a silent no-op, not an error, so the grid has to carry the
    # default rather than trusting the operator to pass the flag).
    grid_backbone = grid_preset.get("backbone_defaults") or {}
    link_bandwidth_mbps = args.link_bandwidth_mbps
    if link_bandwidth_mbps is None:
        link_bandwidth_mbps = grid_backbone.get("link_bandwidth_mbps")
        if link_bandwidth_mbps is not None:
            log(
                f"grid {args.grid!r} declares a backbone by default "
                f"({link_bandwidth_mbps} MB/s); override with --link-bandwidth-mbps",
                quiet,
            )
    if link_bandwidth_mbps is not None:
        # Topology knobs: explicit CLI wins, then the preset's backbone_defaults, then
        # the historical defaults (12/1/0) — a grid whose question IS the topology must
        # not depend on the operator remembering three flags (topo_transfer_v1's rule,
        # extended to n_core/attach/chords for the route_c screen).
        backbone_n_core = (args.backbone_n_core if args.backbone_n_core is not None
                           else grid_backbone.get("n_core", 12))
        backbone_attach_degree = (
            args.backbone_attach_degree if args.backbone_attach_degree is not None
            else grid_backbone.get("attach_degree", 1))
        backbone_chord_count = (
            args.backbone_chord_count if args.backbone_chord_count is not None
            else grid_backbone.get("chord_count", 0))
        base_config.setdefault('network', {})['backbone'] = {
            'n_core': backbone_n_core,
            'attach_degree': backbone_attach_degree,
            'chord_count': backbone_chord_count,
            'core_link_latency_ms': 4.0,
            'access_link_latency_ms': 20.0,
            'bandwidth_mbps': link_bandwidth_mbps,
            'rng_stream': args.backbone_rng_stream,
        }
        log(
            f"link_contention_v1: {link_bandwidth_mbps} MB/s per link over a "
            f"{backbone_n_core}-core ring (attach={backbone_attach_degree}, "
            f"chords={backbone_chord_count})",
            quiet,
        )

    # route_a: server<->server reachability. Every edge the generator makes is
    # client<->server, so a parent and child that both land on servers have no distance
    # and no route at all — and `_dependency_transfer_time` fails loud rather than
    # charging 0.0 for one. A DAG grid therefore REQUIRES this, the same way
    # topo_transfer_v1 has to carry its backbone default rather than trust the operator.
    if grid_preset.get("server_mesh"):
        base_config.setdefault('network', {})['server_mesh'] = True
        log("route_a: server<->server mesh enabled (parent->child distances)", quiet)

    # A grid may name task types the base config has no application entries for
    # (space_with_network.json ships only nofs-dnn1/nofs-dnn2). Synthesize them from an
    # existing entry so the shared config stays untouched for every other grid.
    task_type_pair = tuple(grid_preset.get("task_type_pair", DEFAULT_TASK_TYPE_PAIR))
    # route_a: a DAG preset names its own (distinct) task types, which replace the pair as
    # the set needing wsc/prewarm/replicas entries below.
    dag_shape = grid_preset.get("dag_shape")
    dag_task_types = tuple(grid_preset.get("dag_task_types", ())) if dag_shape else ()
    # A DAG names 4 types, all of which need wsc/prewarm/replicas entries — but
    # `task_type_pair` stays a PAIR, because the ratio arithmetic in
    # generate_workload_templates unpacks exactly two names.
    types_needing_entries = dag_task_types if dag_shape else task_type_pair
    template_prewarm = next(iter(base_config['prewarm'].values()))
    template_replicas = next(iter(base_config['replicas'].values()))
    for task_type_name in types_needing_entries:
        # prewarm/replicas are keyed by TASK TYPE — every DAG node type needs them, or it
        # has no replica anywhere and the placement enumerator finds no candidate for it.
        if task_type_name not in base_config['prewarm']:
            base_config['prewarm'][task_type_name] = deepcopy(template_prewarm)
            base_config['replicas'][task_type_name] = deepcopy(template_replicas)
            log(f"Added prewarm/replica entries for task type {task_type_name}", quiet)
        # wsc is keyed by APPLICATION. For a flat grid each task type is its own
        # application; for a DAG they are all nodes of ONE application, handled below.
        if dag_shape:
            continue
        app_name = f"nofs-{task_type_name}"
        if app_name in base_config.get('wsc', {}):
            continue
        base_config['wsc'][app_name] = deepcopy(next(iter(base_config['wsc'].values())))
        log(f"Added application entries for {app_name}", quiet)

    if dag_shape:
        # `apps` is derived from wsc.keys(), and prepare_workloads keeps only events whose
        # application.name is in it — so without this the DAG event is filtered out and the
        # run dies later with the uninformative "No workload events available for state
        # capture". The DAG application replaces the per-type ones rather than joining
        # them: the trace contains no `nofs-<type>` events at all, and each surviving one
        # would demand its own sampled workload factor for zero events.
        dag_app_name = f"nofs-{dag_shape}"
        base_config['wsc'] = {
            dag_app_name: deepcopy(next(iter(base_config['wsc'].values())))
        }
        log(f"DAG application {dag_app_name} is the only wsc entry", quiet)

    # Generate workload templates
    log(f"\nGenerating workload templates...", quiet)
    templates = generate_workload_templates(
        workload_base_file,
        workload_templates_dir,
        NUM_WORKLOAD_TEMPLATES,
        quiet,
        task_type_pair=task_type_pair,
        workload_seed=args.workload_seed,
        dag_shape=dag_shape,
        dag_task_types=dag_task_types,
        dag_instances=grid_preset.get("dag_instances", 1),
        demand_spread=grid_preset.get("demand_spread"),
    )
    log(f"Generated {len(templates)} workload templates "
        f"(workload_seed={args.workload_seed})", quiet)
    
    # Generate datasets
    log(f"\n=== Starting Dataset Generation ===", quiet)
    
    dataset_idx = args.start_from
    if dataset_idx > 0:
        log(f"Starting from dataset index: {dataset_idx} (ds_{dataset_idx:05d})", quiet)
    template_idx = 0
    total_time = 0
    successful = 0
    skipped = 0
    failed = 0
    
    total_combinations = grid_total
    log(f"Total possible combinations: {total_combinations}", quiet)
    
    start_time = time.time()
    
    # Calculate starting positions for nested loops
    start_from = args.start_from
    current_idx = 0
    
    for topo_label, topo_kwargs in topology_variants:
        for replica_cfg in replica_configs:
            for seed in seeds:
                for queue_dist in queue_distributions:
                    # Skip until we reach the starting index
                    if current_idx < start_from:
                        current_idx += 1
                        template_idx = (template_idx + 1) % NUM_WORKLOAD_TEMPLATES
                        continue
                    
                    if dataset_idx >= max_datasets:
                        break
                    
                    dataset_id = f"ds_{dataset_idx:05d}"
                    output_dir = output_base / dataset_id
                    
                    # Resume only when full BF artifacts exist (JSONL = placement–RTT universe)
                    pub_jsonl = output_dir / "placements" / "placements.jsonl"
                    has_jsonl = pub_jsonl.exists() and pub_jsonl.stat().st_size > 0
                    has_best = (output_dir / "best.json").exists()

                    if args.only_missing_jsonl:
                        if has_jsonl:
                            log(
                                f"[{dataset_id}] Skipping (--only-missing-jsonl: JSONL ok)",
                                quiet,
                            )
                            dataset_idx += 1
                            current_idx += 1
                            template_idx = (template_idx + 1) % NUM_WORKLOAD_TEMPLATES
                            continue
                        if not has_best:
                            log(
                                f"[{dataset_id}] Skipping (--only-missing-jsonl: no best.json)",
                                quiet,
                            )
                            dataset_idx += 1
                            current_idx += 1
                            template_idx = (template_idx + 1) % NUM_WORKLOAD_TEMPLATES
                            continue
                        log(
                            f"[{dataset_id}] BF repair: best.json without placements.jsonl",
                            quiet,
                            force=True,
                        )
                        if pub_jsonl.exists() and pub_jsonl.stat().st_size == 0:
                            pub_jsonl.unlink()
                    elif args.resume and has_best and has_jsonl:
                        log(
                            f"[{dataset_id}] Skipping (best.json + placements.jsonl)",
                            quiet,
                        )
                        dataset_idx += 1
                        current_idx += 1
                        template_idx = (template_idx + 1) % NUM_WORKLOAD_TEMPLATES
                        continue
                    elif args.resume and has_best and not has_jsonl:
                        log(
                            f"[{dataset_id}] Re-running: best.json exists but "
                            "placements/placements.jsonl missing or empty",
                            quiet,
                            force=True,
                        )
                        if pub_jsonl.exists() and pub_jsonl.stat().st_size == 0:
                            pub_jsonl.unlink()
                    elif not has_best and has_jsonl:
                        log(
                            f"[{dataset_id}] Re-running: stale placements.jsonl without best.json",
                            quiet,
                            force=True,
                        )
                        pub_jsonl.unlink()
                    
                    # Get workload template
                    template = templates[template_idx]
                    
                    # Create config for this iteration (with batch_size matching num_tasks)
                    config = create_config_for_iteration(
                        base_config,
                        replica_cfg,
                        seed,
                        queue_dist,
                        batch_size=batch_size,
                        task_type_pair=task_type_pair,
                        dag_task_types=dag_task_types or None,
                        replica_server_percentage=(
                            args.replica_server_percentage
                            if args.replica_server_percentage is not None
                            else grid_preset.get("replica_server_percentage")
                        ),
                        replica_overlap=bool(grid_preset.get("replica_overlap", False)),
                        **topo_kwargs,
                    )
                    
                    qname = queue_dist[0]
                    per_client, per_server, client_pct, server_pct = replica_cfg
                    
                    log(
                        f"\n[{dataset_id}] topo={topo_label} rpc={per_client} "
                        f"rps={per_server} cpct={client_pct} spct={server_pct} "
                        f"seed={seed} q={qname}",
                        quiet,
                    )
                    
                    # Generate dataset
                    status, rtt, duration = generate_single_dataset(
                        dataset_id=dataset_id,
                        output_dir=output_dir,
                        config=config,
                        workload_template=template,
                        sim_input_path=sim_input_path,
                        sample_json_file=sample_json_file,
                        samples_file=samples_file,
                        mapping_file=mapping_file,
                        seed=seed,
                        max_workers=max_workers,
                        quiet=quiet,
                        fast_forward_warmup=args.fast_forward_warmup,
                        fast_forward_threshold=args.fast_forward_threshold,
                        allow_non_unique_replicas=args.allow_non_unique_replicas,
                        warmth_physics=args.warmth_physics,
                    )
                    
                    total_time += duration
                    
                    # Match logs/non_unique_progress_* line shape: existing= new= best_rtt=
                    # (brute-force run: no prior placements file → existing=0, new=completed sims)
                    num_existing = 0
                    num_new = 0
                    metadata_file = output_dir / "placement_metadata.json"
                    if metadata_file.exists():
                        try:
                            with open(metadata_file, 'r') as mf:
                                metadata = json.load(mf)
                            num_new = int(metadata.get('completed', metadata.get('num_placements', 0)))
                        except Exception:
                            pass
                    
                    if status == 'success':
                        successful += 1
                        log(f"  SUCCESS: RTT={rtt:.3f}s ({duration:.1f}s)", quiet)
                        with open(progress_log, 'a') as f:
                            f.write(
                                f"{dataset_id} SUCCESS {datetime.now().isoformat()} "
                                f"{duration:.1f}s existing={num_existing} new={num_new} "
                                f"best_rtt={rtt:.3f}s\n"
                            )
                    elif status == 'skipped':
                        skipped += 1
                        log(f"  SKIPPED: infeasible configuration ({duration:.1f}s)", quiet)
                        with open(progress_log, 'a') as f:
                            f.write(
                                f"{dataset_id} SKIPPED {datetime.now().isoformat()} "
                                f"infeasible\n"
                            )
                    else:
                        failed += 1
                        log(f"  FAILED ({duration:.1f}s)", quiet)
                        with open(progress_log, 'a') as f:
                            f.write(
                                f"{dataset_id} FAILED {datetime.now().isoformat()} "
                                f"{duration:.1f}s\n"
                            )
                    
                    dataset_idx += 1
                    current_idx += 1
                    template_idx = (template_idx + 1) % NUM_WORKLOAD_TEMPLATES
                    
                    # Progress update
                    if dataset_idx % 10 == 0:
                        elapsed = time.time() - start_time
                        rate = dataset_idx / elapsed if elapsed > 0 else 0
                        log(f"\n--- Progress: {dataset_idx}/{max_datasets} "
                            f"({100*dataset_idx/max_datasets:.1f}%) - "
                            f"{rate:.2f} datasets/min ---", quiet)
                
                if dataset_idx >= max_datasets:
                    break
            if dataset_idx >= max_datasets:
                break
        if dataset_idx >= max_datasets:
            break
    
    # Summary
    total_elapsed = time.time() - start_time
    
    log(f"\n=== Generation Complete ===", quiet, force=True)
    log(f"Total attempted: {dataset_idx}", quiet, force=True)
    log(f"Successful: {successful}", quiet, force=True)
    log(f"Skipped (infeasible): {skipped}", quiet, force=True)
    log(f"Failed: {failed}", quiet, force=True)
    if successful > 0:
        log(f"Success rate: {100*successful/(successful+failed+skipped):.1f}%", quiet, force=True)
    log(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)", quiet, force=True)
    log(f"Average time per dataset: {total_elapsed/max(1, dataset_idx):.1f}s", quiet, force=True)
    log(f"Output directory: {output_base}", quiet, force=True)
    log(f"Progress log: {progress_log}", quiet, force=True)


if __name__ == "__main__":
    main()
