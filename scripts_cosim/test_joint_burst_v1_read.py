"""joint_burst_v1: the registered bars behave as signed, before any burst-trained checkpoint exists.

Run: pipenv run python3 -m pytest scripts_cosim/test_joint_burst_v1_read.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts_cosim.joint_burst_v1_read import (  # noqa: E402
    J_MIN_DATASETS, J_MIN_TRAIN_CELLS, J_STUDY_CANDIDATES, J_STUDY_TOPOLOGIES, J_TRAIN_SEEDS,
    J_WINDOWS, V_ARM_BEATS_REACTIVE, V_CORPUS_READY, V_CORPUS_SHORT, V_DESIGN_READY,
    V_DESIGN_SHORT, V_GNN_BEATS_GREEDY_ONLY, V_GNN_WIN, V_GRAPH_BEATS_GREEDY, V_GREEDY_FASTER,
    V_NO_GNN_WIN, V_NOT_SEP, V_SERVED_TRAINING_HELPS, broadcast_rule, environments,
    pair_checkpoint_stats, read_j0, read_j1, read_j5, read_j6, select_burst_topologies,
)
from scripts_cosim.peer_only_v1_read import V_UNREADABLE  # noqa: E402

SEEDS = list(range(1, 17))


def test_training_and_study_pools_are_disjoint():
    assert not set(J_TRAIN_SEEDS) & set(J_STUDY_CANDIDATES)
    assert len(J_STUDY_CANDIDATES) == 28 and len(J_TRAIN_SEEDS) == 24
    assert J_MIN_TRAIN_CELLS == 16 and J_MIN_DATASETS == 192 and J_STUDY_TOPOLOGIES == 4


def test_selection_needs_both_paths_on_all_windows_and_unknown_is_not_a_pass():
    share = {(40, t, w): 0.5 for t in J_STUDY_CANDIDATES for w in J_WINDOWS}
    done = {(40, t, w): True for t in J_STUDY_CANDIDATES for w in J_WINDOWS}
    r = select_burst_topologies(share, done)
    assert r["verdict"] == V_DESIGN_READY and r["topologies"] == [9001, 9002, 9003, 9005]
    done[(40, 9001, "w2")] = False                     # batch path spins on one window
    share.pop((40, 9002, "w0"))                        # reactive unknown on one window
    share[(40, 9003, "w1")] = 0.85                     # reactive saturated
    r = select_burst_topologies(share, done)
    assert r["topologies"] == [9005, 9101, 9102, 9103]
    for t in J_STUDY_CANDIDATES:
        done[(40, t, "w3")] = False
    assert select_burst_topologies(share, done)["verdict"] == V_DESIGN_SHORT


def test_j0_needs_the_corpus_as_well_as_the_design():
    sel = {"verdict": V_DESIGN_READY}
    assert read_j0(sel, 20, 240)["corpus"] == V_CORPUS_READY
    assert read_j0(sel, 15, 240)["corpus"] == V_CORPUS_SHORT
    assert read_j0(sel, 20, 100)["corpus"] == V_CORPUS_SHORT


def test_j1_pairs_the_graph_arm_against_the_broadcast_greedy():
    envs = environments(40, [9001, 9003, 9101, 9104])
    greedy = {e: 6.5 for e in envs}
    graph = {(e, s): 6.5 * (1 - 0.08 - 0.001 * s) for e in envs for s in SEEDS}
    st = pair_checkpoint_stats(graph, broadcast_rule(greedy, SEEDS), envs)
    assert read_j1(st)["verdict"] == V_GRAPH_BEATS_GREEDY
    slow = {(e, s): 6.5 * (1 + 0.25 + 0.001 * s) for e in envs for s in SEEDS}
    assert read_j1(pair_checkpoint_stats(slow, broadcast_rule(greedy, SEEDS), envs))["verdict"] == V_GREEDY_FASTER
    tie = {(e, s): 6.5 * (1 + 0.01 * ((s % 2) * 2 - 1)) for e in envs for s in SEEDS}
    assert read_j1(pair_checkpoint_stats(tie, broadcast_rule(greedy, SEEDS), envs))["verdict"] == V_NOT_SEP
    assert read_j1({s: -8.0 for s in SEEDS[:15]})["verdict"] == V_UNREADABLE


def test_j5_and_j6_compose_as_signed():
    assert read_j5({s: -10.0 - 0.1 * s for s in SEEDS})["verdict"] == V_SERVED_TRAINING_HELPS
    assert read_j6({"verdict": V_GRAPH_BEATS_GREEDY}, {"verdict": V_ARM_BEATS_REACTIVE})["verdict"] == V_GNN_WIN
    assert read_j6({"verdict": V_GRAPH_BEATS_GREEDY}, {"verdict": V_NOT_SEP})["verdict"] == V_GNN_BEATS_GREEDY_ONLY
    assert read_j6({"verdict": V_NOT_SEP}, {"verdict": V_ARM_BEATS_REACTIVE})["verdict"] == V_NO_GNN_WIN
    assert read_j6({"verdict": V_UNREADABLE}, {"verdict": V_ARM_BEATS_REACTIVE})["verdict"] == V_UNREADABLE
