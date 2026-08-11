"""
Frozen Regime B problem spec — cold-burst env with ≥10× oracle–greedy headroom.

Machine-checkable source of truth. Do not reopen RQ3 / hub9 decode polish.
Primary score = max-burst elapsed (see regime_b_metrics.py); never total_rtt.
"""

from __future__ import annotations

from typing import Any, Dict

# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

PROBLEM_ID = "regime_b_cold_burst_v1"
SPEC_VERSION = "1.0.0"
FROZEN_DATE = "2026-08-11"

# ---------------------------------------------------------------------------
# Metric contract (non-negotiable)
# ---------------------------------------------------------------------------

PRIMARY_SCORE_KEY = "regime_b_primary_score_s"
BANNED_PRIMARY_METRICS = ("total_rtt", "sum_elapsed", "total_rtt_s")
MIN_ORACLE_GREEDY_RATIO = 10.0
# Toy harness (N=4 platform_reuse_v1) stays as regression; target is the gate.
TOY_N_TASKS = 4
TOY_MIN_RATIO = 3.5

# ---------------------------------------------------------------------------
# Physics — why hub9/n64 failed the gate
# ---------------------------------------------------------------------------

# platform_reuse_v1: each cold replica pulls through node FilterStore → N× T_pull.
# node_disk_v2 same-image: first pull warms node disk → co-located siblings ~free
# (hub9/n64 primary stayed ~18–30s). Gate physics MUST be platform_reuse_v1.
GATE_WARMTH_PHYSICS = "platform_reuse_v1"
FORBIDDEN_GATE_PHYSICS = ("node_disk_v2",)

# Theory constants (data/nofs-ids dnn1 @ flashCard / 100 MB/s) — storage_contention.md
T_PULL_S = 31.3038
T_BASELINE_S = 31.6367  # T_pull + cold_start + exec

# ---------------------------------------------------------------------------
# Cluster + trace (target gate cell)
# ---------------------------------------------------------------------------

# N simultaneous cold tasks. Ratio ≈ N * T_pull / T_baseline → N=12 ≈ 11.87×.
TARGET_N_TASKS = 12
TARGET_TASK_TYPE = "dnn1"
TARGET_BURST_ID = "cold_burst_n12"
# One burst at t=0; multi-burst traces are optional later, not the gate.
TARGET_BURST_TIMES_S = (0.0,)
TARGET_CLIENT = "client_node0"

# Cluster: 1 client + N servers; scarce attractor = node0 (all N cold platforms);
# oracle spreads one cold pull per server flashCard.
TARGET_SERVER_COUNT = TARGET_N_TASKS
TARGET_PLATFORMS_ON_SCARCE_NODE = TARGET_N_TASKS
TARGET_PLATFORMS_ON_OTHER_NODES = 1
TARGET_STORAGE = ("flashCard", "someRemote")

# Primary design lever (closes pending Q in memory.md):
# scarce attractor + FilterStore co-location cost — NOT over-provision alone.
# Over-provision without a scarce preferred site leaves greedy near-optimal (contention_v1).
PRIMARY_LEVER = "scarce_attractor_filterstore"
ANTI_CORRELATED_PREFS = True  # greedy #1 site is expensive under co-location

# ---------------------------------------------------------------------------
# Expected headroom (closed form; sim must match within tolerance)
# ---------------------------------------------------------------------------


def expected_oracle_primary_s() -> float:
    return T_BASELINE_S


def expected_greedy_primary_s(n: int = TARGET_N_TASKS) -> float:
    # Contended last-task ≈ N * T_pull + cold + exec (cold+exec ≈ T_BASELINE - T_PULL)
    return n * T_PULL_S + (T_BASELINE_S - T_PULL_S)


def expected_oracle_greedy_ratio(n: int = TARGET_N_TASKS) -> float:
    return expected_greedy_primary_s(n) / expected_oracle_primary_s()


TARGET_EXPECTED_RATIO = expected_oracle_greedy_ratio(TARGET_N_TASKS)
TARGET_RATIO_TOLERANCE = 0.5  # absolute on ratio for near-theory check
TARGET_SCORE_TOLERANCE_S = 2.0  # absolute seconds vs theory

# ---------------------------------------------------------------------------
# Co-sim / live follow-on (specified, not yet generated)
# ---------------------------------------------------------------------------

COSIM_GRID_NAME = "regime_b_cold_burst_v1"
COSIM_OUTPUT_SUBDIR = "gnn_datasets_4tasks_regime_b_cold_burst_v1"
COSIM_NUM_TASKS = 4  # BF joint labels; live gate remains TARGET_N_TASKS=12
COSIM_N_DATASETS = 450  # 2 conn × 3 rep × 3 queue × 25 seeds
COSIM_REQUIRE_PLACEMENTS_JSONL = True
COSIM_ALLOW_NON_UNIQUE_REPLICAS = True
COSIM_WARMTH_PHYSICS = GATE_WARMTH_PHYSICS
# Live baselines only after gate PASS: Kn / MLP / GNN zero-shot on this env.
BASELINE_POLICIES = ("knative_network", "mlp_batch", "gnn")
LIVE_STUB_DIR = "simulation_data/regime_b_cold_burst_v1/live_stub"

# Hard stops carried from memory §2
HARD_STOPS = (
    "reopen_rq3",
    "hub9_skew_bipartite_decode",
    "merge_or_contention_v3_deploy",
    "more_sealed_er_cells",
    "cache_5_5_ablation_without_new_env",
    "total_rtt_as_primary",
    "gate_under_node_disk_v2_same_image",
)


def as_dict() -> Dict[str, Any]:
    return {
        "problem_id": PROBLEM_ID,
        "spec_version": SPEC_VERSION,
        "frozen_date": FROZEN_DATE,
        "metric": {
            "primary": PRIMARY_SCORE_KEY,
            "banned_primary": list(BANNED_PRIMARY_METRICS),
            "min_oracle_greedy_ratio": MIN_ORACLE_GREEDY_RATIO,
        },
        "physics": {
            "gate_warmth_physics": GATE_WARMTH_PHYSICS,
            "forbidden_gate_physics": list(FORBIDDEN_GATE_PHYSICS),
            "t_pull_s": T_PULL_S,
            "t_baseline_s": T_BASELINE_S,
        },
        "cluster": {
            "server_count": TARGET_SERVER_COUNT,
            "platforms_on_scarce_node": TARGET_PLATFORMS_ON_SCARCE_NODE,
            "platforms_on_other_nodes": TARGET_PLATFORMS_ON_OTHER_NODES,
            "storage": list(TARGET_STORAGE),
            "primary_lever": PRIMARY_LEVER,
            "anti_correlated_prefs": ANTI_CORRELATED_PREFS,
        },
        "trace": {
            "n_tasks": TARGET_N_TASKS,
            "task_type": TARGET_TASK_TYPE,
            "burst_id": TARGET_BURST_ID,
            "burst_times_s": list(TARGET_BURST_TIMES_S),
            "client": TARGET_CLIENT,
        },
        "expected": {
            "oracle_primary_s": expected_oracle_primary_s(),
            "greedy_primary_s": expected_greedy_primary_s(),
            "oracle_greedy_ratio": TARGET_EXPECTED_RATIO,
        },
        "cosim": {
            "grid_name": COSIM_GRID_NAME,
            "output_subdir": COSIM_OUTPUT_SUBDIR,
            "num_tasks": COSIM_NUM_TASKS,
            "n_datasets": COSIM_N_DATASETS,
            "warmth_physics": COSIM_WARMTH_PHYSICS,
            "require_placements_jsonl": COSIM_REQUIRE_PLACEMENTS_JSONL,
            "allow_non_unique_replicas": COSIM_ALLOW_NON_UNIQUE_REPLICAS,
            "baseline_policies": list(BASELINE_POLICIES),
            "live_stub_dir": LIVE_STUB_DIR,
        },
        "hard_stops": list(HARD_STOPS),
        "toy_regression": {
            "n_tasks": TOY_N_TASKS,
            "min_ratio": TOY_MIN_RATIO,
        },
    }


def assert_gate_ratio(ratio: float, *, context: str = "") -> None:
    """Fail loud if oracle–greedy headroom collapses below the frozen gate."""
    if ratio < MIN_ORACLE_GREEDY_RATIO:
        where = f" ({context})" if context else ""
        raise AssertionError(
            f"Regime B gate FAIL{where}: oracle_greedy_ratio={ratio:.3f}x "
            f"< required {MIN_ORACLE_GREEDY_RATIO:.1f}x "
            f"(problem={PROBLEM_ID} n={TARGET_N_TASKS} physics={GATE_WARMTH_PHYSICS})"
        )


def assert_primary_metric_key(key: str) -> None:
    if key in BANNED_PRIMARY_METRICS or key != PRIMARY_SCORE_KEY:
        raise AssertionError(
            f"Banned/wrong primary metric {key!r}; use {PRIMARY_SCORE_KEY!r} "
            f"(never {BANNED_PRIMARY_METRICS})"
        )
