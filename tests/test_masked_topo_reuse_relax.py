"""peer_affinity_v1 T1: the shared masked decoder's two new options.

Default (no options): bit-identical to the registered route_b decoder -- replica reuse
forbidden, a stuck state returns None. `allow_replica_reuse` lifts the reuse mask only.
`relax_on_stuck` backtracks one step at a time and, failing that, takes the least-over-cap
candidate and COUNTS it; the returned plan is never silently infeasible.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.policy.gnn.seq_decode import GnnDecodeRunStats, decode_masked_topo_placement  # noqa: E402

# two tasks, both prefer platform (1, 10) on node 1; node 1's cap admits ONE unit
TL = {0: [(1, 10), (2, 20)], 1: [(1, 10), (2, 20)]}
PARENTS = {0: [], 1: []}
LOGITS = [[2.0, 1.0], [2.0, 1.0]]
DEM = {0: [1.0, 1.0], 1: [1.0, 1.0]}


def decode(**kw):
    stats = GnnDecodeRunStats()
    combo = decode_masked_topo_placement(LOGITS, TL, 2, dag_parents=PARENTS, node_caps=kw.pop("caps"),
                                         demands=DEM, stats=stats, **kw)
    return combo, stats


def test_default_forbids_reuse_and_is_unchanged():
    combo, _ = decode(caps={})
    assert combo == ((1, 10), (2, 20))  # task 1 cannot reuse (1, 10)


def test_reuse_allowed_lets_both_take_the_best_platform():
    combo, _ = decode(caps={}, allow_replica_reuse=True)
    assert combo == ((1, 10), (1, 10))


def test_capacity_mask_still_applies_with_reuse():
    combo, _ = decode(caps={1: 1.0}, allow_replica_reuse=True)
    assert combo == ((1, 10), (2, 20))


def test_stuck_returns_none_without_relaxation():
    # node 2 capped at 0: task 1 has no feasible candidate once task 0 fills node 1
    combo, stats = decode(caps={1: 1.0, 2: 0.0})
    assert combo is None and stats.masked_topo_failed_decodes == 1


def test_backtracking_finds_the_feasible_plan_before_relaxing():
    # node 1 admits one, node 2 admits one but task 0 must NOT take node 2's slot first:
    # task 0 prefers node 1 (fine), task 1 then prefers node 1 (full) -> node 2 (ok).
    # Make node 2 unusable for task 1 only by capping it at 0.5 with task-1 demand 1.0
    # while task 0's demand on node 2 is 0.5 -> stuck for task 1 -> backtrack task 0 to
    # node 2 -> task 1 gets node 1.
    dem = {0: [1.0, 0.5], 1: [1.0, 1.0]}
    stats = GnnDecodeRunStats()
    combo = decode_masked_topo_placement(LOGITS, TL, 2, dag_parents=PARENTS, node_caps={1: 1.0, 2: 0.5},
                                         demands=dem, stats=stats, relax_on_stuck=True)
    assert combo == ((2, 20), (1, 10))
    assert stats.masked_topo_backtracks == 1 and stats.masked_topo_relaxed_steps == 0


def test_relaxation_is_counted_when_no_feasible_plan_exists():
    stats = GnnDecodeRunStats()
    combo = decode_masked_topo_placement(LOGITS, TL, 2, dag_parents=PARENTS, node_caps={1: 1.0, 2: 0.0},
                                         demands=DEM, stats=stats, relax_on_stuck=True)
    assert combo is not None
    assert stats.masked_topo_relaxed_steps >= 1 and stats.masked_topo_relaxed_decodes == 1
