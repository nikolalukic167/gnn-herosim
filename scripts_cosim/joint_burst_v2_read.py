#!/usr/bin/env python3
"""joint_burst_v2 -- the bars, signed 2026-09-20 before any v2 checkpoint exists. See
docs/lineages/joint_burst_v2.md.

joint_burst_v1 closed NO-GNN-WIN: the served-distribution corpus took gnnedge0 past reactive and
its cold twin but not past the batched greedy in its own seat (+16.8 %). Step 0 found two causes:
(1) the model's decoder was CAPPED (alpha 2.5) out of the label's co-location move on loaded
states its corpus had REJECTED (the alpha=2.0 cap filter dropped 579/634 snapshots); (2) the
room above the 1-pass greedy is joint structure a coordinate-descent greedy captures (regret
33.5 -> 11.2 s). v2 rebuilds the corpus keeping the loaded states, trains and serves UNCAPPED,
and gates against the CD greedy (the honest bar) as well as the 1-pass greedy (the reachable one).

Units as in joint_burst_v1 / peer_greedy_live_v1: learned arm per CHECKPOINT (n = 16), rule per
ENVIRONMENT (n = 16), rule-vs-learned by broadcasting the rule over checkpoint seeds. Bars:
|median| >= 5 %, p < 0.05, two-sided signed-rank.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts_cosim.peer_only_v1_read import V_UNREADABLE  # noqa: E402
from scripts_cosim.unsaturated_edge_v1_read import E_MIN_CHECKPOINTS  # noqa: E402
from scripts_cosim.peer_greedy_live_v1_read import (  # noqa: E402
    P_MIN_ENVIRONMENTS, broadcast_rule, env_stats,
)
from scripts_cosim.unsaturated_scale_v2_read import _one_sample  # noqa: E402

__all__ = [
    "K_GRAPH", "K_TWIN", "K_V1_UNCAPPED", "K_CD", "K_BATCHED", "K_REACTIVE", "K_RANDOM", "K_X2",
    "V_GRAPH_BEATS_GREEDY", "V_GREEDY_FASTER", "V_NOT_SEP", "V_GRAPH_BEATS_CD", "V_CD_FASTER",
    "V_ARM_BEATS_REACTIVE", "V_REACTIVE_FASTER", "V_GRAPH_FASTER_THAN_TWIN", "V_POINTWISE_FASTER",
    "V_CORPUS_HELPS", "V_CORPUS_NEUTRAL", "V_UNCAP_BEATS_GREEDY", "V_UNCAP_NOT",
    "V_CD_BEATS_1PASS", "V_CD_NOT", "V_MORE_COLO_HELPS", "V_MORE_COLO_HURTS", "V_MORE_COLO_NEUTRAL",
    "V_GNN_BEATS_GREEDY", "V_GNN_BEATS_CD", "V_NO_GNN_WIN",
    "read_k1", "read_k2", "read_k3", "read_k4", "read_k5", "read_k6", "read_k7", "read_k8", "read_k9",
]

K_GRAPH = "jb2_gnnedge0"          # this lineage's uncapped-trained checkpoints
K_TWIN = "jb2_mpoff"
K_V1_UNCAPPED = "jb1uncapped_gnnedge0"  # v1 weights served uncapped (isolates corpus from cap)
K_CD = "peer_greedy_network_cd"   # coordinate-descent greedy (the honest bar)
K_BATCHED = "peer_greedy_network_batch"  # the 1-pass greedy (the reachable bar)
K_REACTIVE = "knative_network"
K_RANDOM = "random_network"
K_X2 = "peer_greedy_network_batch_x2scale"

V_NOT_SEP = "NOT-SEPARATED"
V_GRAPH_BEATS_GREEDY = "GRAPH-BEATS-GREEDY"
V_GREEDY_FASTER = "GREEDY-FASTER-THAN-GRAPH"
V_GRAPH_BEATS_CD = "GRAPH-BEATS-CD"
V_CD_FASTER = "CD-FASTER-THAN-GRAPH"
V_ARM_BEATS_REACTIVE = "ARM-BEATS-REACTIVE"
V_REACTIVE_FASTER = "REACTIVE-FASTER"
V_GRAPH_FASTER_THAN_TWIN = "GRAPH-FASTER-THAN-TWIN"
V_POINTWISE_FASTER = "POINTWISE-FASTER-THAN-GRAPH"
V_CORPUS_HELPS = "CORPUS-HELPS-BEYOND-UNCAPPING"
V_CORPUS_NEUTRAL = "CORPUS-NEUTRAL-BEYOND-UNCAPPING"
V_UNCAP_BEATS_GREEDY = "UNCAPPING-ALONE-BEATS-GREEDY"
V_UNCAP_NOT = "UNCAPPING-ALONE-NOT-ENOUGH"
V_CD_BEATS_1PASS = "CD-BEATS-1PASS"
V_CD_NOT = "CD-NOT-BETTER-THAN-1PASS"
V_MORE_COLO_HELPS = "MORE-COLOCATION-HELPS"
V_MORE_COLO_HURTS = "MORE-COLOCATION-HURTS"
V_MORE_COLO_NEUTRAL = "MORE-COLOCATION-NEUTRAL"
V_GNN_BEATS_GREEDY = "GNN-BEATS-GREEDY"
V_GNN_BEATS_CD = "GNN-BEATS-CD"
V_NO_GNN_WIN = "NO-GNN-WIN"


def read_k1(graph_vs_1pass: Mapping, *, min_n: Optional[int] = None) -> dict:
    """PRIMARY reachable bar: jb2 gnnedge0 vs the 1-pass batched greedy, GRAPH FIRST, paired on
    env and checkpoint. Negative = the graph arm is faster."""
    return _one_sample(graph_vs_1pass, min_n=E_MIN_CHECKPOINTS if min_n is None else min_n,
                       faster_a=V_GRAPH_BEATS_GREEDY, faster_b=V_GREEDY_FASTER, tie=V_NOT_SEP)


def read_k2(graph_vs_cd: Mapping, *, min_n: Optional[int] = None) -> dict:
    """The honest bar: jb2 gnnedge0 vs the coordinate-descent greedy, GRAPH FIRST."""
    return _one_sample(graph_vs_cd, min_n=E_MIN_CHECKPOINTS if min_n is None else min_n,
                       faster_a=V_GRAPH_BEATS_CD, faster_b=V_CD_FASTER, tie=V_NOT_SEP)


def read_k3(graph_vs_reactive: Mapping, *, min_n: Optional[int] = None) -> dict:
    return _one_sample(graph_vs_reactive, min_n=E_MIN_CHECKPOINTS if min_n is None else min_n,
                       faster_a=V_ARM_BEATS_REACTIVE, faster_b=V_REACTIVE_FASTER, tie=V_NOT_SEP)


def read_k4(graph_vs_twin: Mapping, *, min_n: Optional[int] = None) -> dict:
    return _one_sample(graph_vs_twin, min_n=E_MIN_CHECKPOINTS if min_n is None else min_n,
                       faster_a=V_GRAPH_FASTER_THAN_TWIN, faster_b=V_POINTWISE_FASTER, tie=V_NOT_SEP)


def read_k5(jb2_vs_v1uncapped: Mapping, *, min_n: Optional[int] = None) -> dict:
    """Does the corpus (loaded states) help BEYOND just uncapping? jb2 vs v1-served-uncapped,
    both uncapped, paired on env and seed, JB2 FIRST."""
    return _one_sample(jb2_vs_v1uncapped, min_n=E_MIN_CHECKPOINTS if min_n is None else min_n,
                       faster_a=V_CORPUS_HELPS, faster_b=V_CORPUS_NEUTRAL, tie=V_CORPUS_NEUTRAL)


def read_k6(v1uncapped_vs_1pass: Mapping, *, min_n: Optional[int] = None) -> dict:
    """Does uncapping the EXISTING v1 checkpoint alone beat the 1-pass greedy? v1-served-uncapped
    vs the 1-pass greedy, GRAPH FIRST."""
    return _one_sample(v1uncapped_vs_1pass, min_n=E_MIN_CHECKPOINTS if min_n is None else min_n,
                       faster_a=V_UNCAP_BEATS_GREEDY, faster_b=V_UNCAP_NOT, tie=V_UNCAP_NOT)


def read_k7(cd_vs_1pass_by_env: Mapping) -> dict:
    """The joint-structure read: CD greedy vs the 1-pass greedy, per environment (n = 16),
    CD FIRST. `cd_vs_1pass_by_env` is the per-env % delta (from `env_stats(cd, one_pass, envs)`).
    Negative = CD is faster."""
    return _one_sample(cd_vs_1pass_by_env, min_n=P_MIN_ENVIRONMENTS,
                       faster_a=V_CD_BEATS_1PASS, faster_b=V_CD_NOT, tie=V_CD_NOT)


def read_k8(x2_vs_1pass_by_env: Mapping) -> dict:
    """Label-validity read: the x2-exchange probe vs the 1-pass greedy, per environment,
    x2 FIRST (per-env % delta). Negative (x2 faster) = more co-location than the rule is
    live-valid."""
    return _one_sample(x2_vs_1pass_by_env, min_n=P_MIN_ENVIRONMENTS,
                       faster_a=V_MORE_COLO_HELPS, faster_b=V_MORE_COLO_HURTS, tie=V_MORE_COLO_NEUTRAL)


def read_k9(k1: Mapping, k2: Mapping) -> dict:
    """Composite: GNN-BEATS-CD iff K2 fires; GNN-BEATS-GREEDY iff K1 fires (but not K2);
    otherwise NO-GNN-WIN. Unreadable if K1 is."""
    if k1.get("verdict") == V_UNREADABLE:
        return {"verdict": V_UNREADABLE, "why": "K1 unreadable"}
    if k2.get("verdict") == V_GRAPH_BEATS_CD:
        v = V_GNN_BEATS_CD
    elif k1.get("verdict") == V_GRAPH_BEATS_GREEDY:
        v = V_GNN_BEATS_GREEDY
    else:
        v = V_NO_GNN_WIN
    return {"verdict": v, "k1": k1.get("verdict"), "k2": k2.get("verdict")}
