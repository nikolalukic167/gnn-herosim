"""joint_burst_v1 -- the bars, signed 2026-09-20 before any burst-trained checkpoint exists.

`burst_groups_v1` showed that with every peer group arriving as one burst the learned arm's
wait is gone and it still only ties reactive Knative, while a greedy on the physics reads
-26 %. Every checkpoint that has ever been served was trained on COLD generated states (no
standing load) and a batch label the served stream never reproduces. Under bursts the served
decision IS the batch decision -- one whole group, decided at once, from a loaded state -- so a
corpus of exactly those states, labelled by the brute-force sweep on the measured clock, is the
first corpus whose label matches what serving asks. This lineage builds it (captured from the
batched rule's own live runs on 24 training topologies), trains `gnnedge0` and its MP-OFF twin
on it, and gates them under bursts against the greedy that beat every cold checkpoint.

Study cells are chosen by the pool's own screen: a topology enters only if reactive Knative is
admissible (queue share <= 0.80) on all four windows AND the batch path finishes on all four
(under bursts it spins on topologies reactive survives). Training topologies (9201-9224) are
disjoint from every study candidate (<= 9124).

Units as in peer_greedy_live_v1: learned arm reads per CHECKPOINT (n = 16), rule reads per
ENVIRONMENT (n = 16), rule-vs-learned by broadcasting the rule over checkpoints. Chain's bars:
|median| >= 5 %, p < 0.05, two-sided signed-rank.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts_cosim.peer_only_v1_read import V_UNREADABLE  # noqa: E402
from scripts_cosim.unsaturated_edge_v1_read import (  # noqa: E402
    E_MIN_CHECKPOINTS, E_TARGET_SHARE, E_WINDOWS, environments, checkpoint_stats,
    pair_checkpoint_stats,
)
from scripts_cosim.peer_greedy_live_v1_read import P_MIN_ENVIRONMENTS, broadcast_rule, env_stats  # noqa: E402
from scripts_cosim.unsaturated_scale_v2_read import _one_sample  # noqa: E402

__all__ = [
    "J_RUNG", "J_WINDOWS", "J_STUDY_CANDIDATES", "J_TRAIN_SEEDS", "J_STUDY_TOPOLOGIES",
    "J_MIN_TRAIN_CELLS", "J_MIN_DATASETS", "J_GRAPH_BT", "J_TWIN_BT", "J_GRAPH_COLD", "J_IMMEDIATE",
    "J_BATCHED", "J_REACTIVE", "J_RANDOM",
    "V_DESIGN_READY", "V_DESIGN_SHORT", "V_CORPUS_READY", "V_CORPUS_SHORT",
    "V_GRAPH_BEATS_GREEDY", "V_GREEDY_FASTER", "V_NOT_SEP", "V_GRAPH_BEATS_RULE", "V_RULE_FASTER",
    "V_ARM_BEATS_REACTIVE", "V_REACTIVE_FASTER", "V_GRAPH_FASTER_THAN_TWIN", "V_POINTWISE_FASTER",
    "V_SERVED_TRAINING_HELPS", "V_SERVED_TRAINING_HURTS", "V_GNN_WIN", "V_GNN_BEATS_GREEDY_ONLY",
    "V_NO_GNN_WIN",
    "environments", "checkpoint_stats", "pair_checkpoint_stats", "broadcast_rule", "env_stats",
    "select_burst_topologies", "read_j0", "read_j1", "read_j2", "read_j3", "read_j4", "read_j5",
    "read_j6",
]

J_RUNG = 40
J_WINDOWS = E_WINDOWS
J_STUDY_CANDIDATES = (9001, 9002, 9003, 9005, 9101, 9102, 9103, 9104, 9105, 9106, 9107, 9108,
                      *range(9109, 9125))
J_TRAIN_SEEDS = tuple(range(9201, 9225))
J_STUDY_TOPOLOGIES = 4
J_MIN_TRAIN_CELLS = 16           # capture runs that finish (the batch path can spin)
J_MIN_DATASETS = 192             # complete sweeps in the training corpus (16 cells x 12)

J_GRAPH_BT = "jb1_gnnedge0"      # burst-trained gnnedge0 (this lineage's checkpoints)
J_TWIN_BT = "jb1_mpoff"          # burst-trained MP-OFF twin
J_GRAPH_COLD = "be1670_gnnedge0" # the cold-corpus checkpoints, served under bursts
J_IMMEDIATE = "peer_greedy_network"
J_BATCHED = "peer_greedy_network_batch"
J_REACTIVE = "knative_network"
J_RANDOM = "random_network"

V_DESIGN_READY = "DESIGN-READY"
V_DESIGN_SHORT = "TOO-FEW-SERVABLE-ENVIRONMENTS"
V_CORPUS_READY = "CORPUS-READY"
V_CORPUS_SHORT = "CORPUS-TOO-SMALL"
V_NOT_SEP = "NOT-SEPARATED"
V_GRAPH_BEATS_GREEDY = "GRAPH-ARM-BEATS-BATCHED-GREEDY"
V_GREEDY_FASTER = "BATCHED-GREEDY-FASTER-THAN-GRAPH-ARM"
V_GRAPH_BEATS_RULE = "GRAPH-ARM-BEATS-IMMEDIATE-RULE"
V_RULE_FASTER = "IMMEDIATE-RULE-FASTER-THAN-GRAPH-ARM"
V_ARM_BEATS_REACTIVE = "ARM-BEATS-REACTIVE-UNDER-BURSTS"
V_REACTIVE_FASTER = "REACTIVE-FASTER-UNDER-BURSTS"
V_GRAPH_FASTER_THAN_TWIN = "GRAPH-FASTER-THAN-TWIN-ON-THE-SERVED-CORPUS"
V_POINTWISE_FASTER = "POINTWISE-FASTER-THAN-GRAPH-ON-THE-SERVED-CORPUS"
V_SERVED_TRAINING_HELPS = "SERVED-DISTRIBUTION-TRAINING-HELPS"
V_SERVED_TRAINING_HURTS = "SERVED-DISTRIBUTION-TRAINING-HURTS"
V_GNN_WIN = "GNN-WIN"
V_GNN_BEATS_GREEDY_ONLY = "GNN-BEATS-GREEDY-NOT-REACTIVE"
V_NO_GNN_WIN = "NO-GNN-WIN"


# --- J0: the pool's screen and the corpus size --------------------------------------------

def select_burst_topologies(share_by_env: Mapping[tuple, Optional[float]],
                            batched_done: Mapping[tuple, bool],
                            candidates: Sequence[int] = J_STUDY_CANDIDATES) -> dict:
    """The study topologies: the `J_STUDY_TOPOLOGIES` lowest-numbered candidates on which
    reactive is admissible on ALL four windows AND the batch path finishes on ALL four.
    Missing or None is a failure on that window (unknown is not a pass)."""
    def ok(t):
        return all(share_by_env.get((J_RUNG, t, w)) is not None
                   and float(share_by_env[(J_RUNG, t, w)]) <= E_TARGET_SHARE
                   and bool(batched_done.get((J_RUNG, t, w), False)) for w in J_WINDOWS)
    qualified = [t for t in candidates if ok(t)]
    if len(qualified) < J_STUDY_TOPOLOGIES:
        return {"verdict": V_DESIGN_SHORT, "qualified": qualified,
                "why": f"{len(qualified)} of {len(candidates)} candidates servable by both paths on all "
                       f"{len(J_WINDOWS)} windows; {J_STUDY_TOPOLOGIES} needed"}
    chosen = sorted(qualified)[:J_STUDY_TOPOLOGIES]
    return {"verdict": V_DESIGN_READY, "topologies": chosen, "qualified": qualified,
            "environments": environments(J_RUNG, chosen),
            "why": f"{len(qualified)} candidates servable by both paths; the {J_STUDY_TOPOLOGIES} lowest ids are {chosen}"}


def read_j0(selection: Mapping[str, object], n_train_cells: int, n_datasets: int) -> dict:
    """Design and corpus readiness. Both must pass before a checkpoint is trained/gated."""
    corpus = (V_CORPUS_READY if (n_train_cells >= J_MIN_TRAIN_CELLS and n_datasets >= J_MIN_DATASETS)
              else V_CORPUS_SHORT)
    return {"design": selection.get("verdict"), "corpus": corpus, "n_train_cells": n_train_cells,
            "n_datasets": n_datasets,
            "why": f"design {selection.get('verdict')}; corpus {n_train_cells} cells / {n_datasets} datasets "
                   f"against {J_MIN_TRAIN_CELLS} / {J_MIN_DATASETS}"}


# --- the reads ---------------------------------------------------------------------------

def read_j1(graph_vs_batched_greedy: Mapping[int, float], *, min_n: Optional[int] = None) -> dict:
    """PRIMARY: burst-trained gnnedge0 vs the batched greedy, GRAPH FIRST, paired on environment
    and checkpoint (`pair_checkpoint_stats(graph, broadcast_rule(greedy, seeds), envs)`).
    Negative = the graph arm is faster."""
    return _one_sample(graph_vs_batched_greedy, min_n=E_MIN_CHECKPOINTS if min_n is None else min_n,
                       faster_a=V_GRAPH_BEATS_GREEDY, faster_b=V_GREEDY_FASTER, tie=V_NOT_SEP)


def read_j2(graph_vs_immediate_rule: Mapping[int, float], *, min_n: Optional[int] = None) -> dict:
    return _one_sample(graph_vs_immediate_rule, min_n=E_MIN_CHECKPOINTS if min_n is None else min_n,
                       faster_a=V_GRAPH_BEATS_RULE, faster_b=V_RULE_FASTER, tie=V_NOT_SEP)


def read_j3(arm_vs_reactive: Mapping[int, float], *, min_n: Optional[int] = None) -> dict:
    return _one_sample(arm_vs_reactive, min_n=E_MIN_CHECKPOINTS if min_n is None else min_n,
                       faster_a=V_ARM_BEATS_REACTIVE, faster_b=V_REACTIVE_FASTER, tie=V_NOT_SEP)


def read_j4(graph_vs_twin: Mapping[int, float], *, min_n: Optional[int] = None) -> dict:
    """Model class on the served corpus: burst-trained gnnedge0 vs burst-trained mpoff, GRAPH FIRST."""
    return _one_sample(graph_vs_twin, min_n=E_MIN_CHECKPOINTS if min_n is None else min_n,
                       faster_a=V_GRAPH_FASTER_THAN_TWIN, faster_b=V_POINTWISE_FASTER, tie=V_NOT_SEP)


def read_j5(bt_vs_cold: Mapping[int, float], *, min_n: Optional[int] = None) -> dict:
    """Does training on the served distribution help? burst-trained vs cold gnnedge0, both served
    under bursts, paired on environment and checkpoint SEED, BURST-TRAINED FIRST."""
    return _one_sample(bt_vs_cold, min_n=E_MIN_CHECKPOINTS if min_n is None else min_n,
                       faster_a=V_SERVED_TRAINING_HELPS, faster_b=V_SERVED_TRAINING_HURTS, tie=V_NOT_SEP)


def read_j6(j1: Mapping[str, object], j3: Mapping[str, object]) -> dict:
    """The composite the lineage exists for: GNN-WIN iff the graph arm beats the batched greedy
    (J1) AND reactive (J3); GNN-BEATS-GREEDY-NOT-REACTIVE if only J1; otherwise NO-GNN-WIN.
    Unreadable if J1 is."""
    if j1.get("verdict") == V_UNREADABLE:
        return {"verdict": V_UNREADABLE, "why": "J1 unreadable"}
    if j1.get("verdict") == V_GRAPH_BEATS_GREEDY:
        v = V_GNN_WIN if j3.get("verdict") == V_ARM_BEATS_REACTIVE else V_GNN_BEATS_GREEDY_ONLY
    else:
        v = V_NO_GNN_WIN
    return {"verdict": v, "j1": j1.get("verdict"), "j3": j3.get("verdict")}
