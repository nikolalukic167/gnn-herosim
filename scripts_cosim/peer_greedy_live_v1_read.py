"""peer_greedy_live_v1 -- the bars, signed 2026-09-20 before any rule arm is served.

`unsaturated_edge_v1` and `batch_window_edge_v1` left the learned arms with one measured
placement gain -- about 1.05 s of peer exchange per task from co-location -- and one cost that
eats it: the 7.1 s a task waits for its peer group before the decoder sees it. The graph arm's
only surviving edge is over its own MP-OFF twin, served with the same batching. Nothing in the
record puts a HAND RULE with the same information on the live gate, in either serving
configuration. This lineage does, on the 16 (topology, window) environments per rung that
`unsaturated_edge_v1` selected, against the reactive, ECT and random baselines and against
`gnnedge0`'s 16 checkpoints, with nothing retrained and nothing re-tuned.

Three rule arms (src/policy/peer_greedy_network/scheduler.py):
  peer_greedy_network        per arrival, no wait: drain(p) + cold + exec + latency + exchange
                             to every peer whose node is already known
  drain_greedy_network       the same with the exchange term off (the ablation)
  peer_greedy_network_batch  the same score on the learned arms' stack: peer-group batching at
                             the cell's 16 s window, greedy in task-id order, in-batch
                             commitments charged, `planned_node_name` pre-pass

The unit. A rule has no training draw, so a rule-vs-baseline read has ONE value per
environment: the unit is the ENVIRONMENT (n = 16 per rung), the statistic is the paired
relative % on the same environment, and the bar is the chain's otherwise (|median| >= 5 %,
p < 0.05, two-sided signed-rank). A rule-vs-`gnnedge0` read keeps the chain's unit -- the
checkpoint -- by pairing the rule's one value per environment against every checkpoint on that
environment (`pair_checkpoint_stats` with the rule broadcast over seeds).
"""
from __future__ import annotations

import sys
from pathlib import Path
from statistics import median
from typing import Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts_cosim.peer_only_v1_read import V_UNREADABLE  # noqa: E402
from scripts_cosim.unsaturated_edge_v1_read import (  # noqa: E402
    E_ALPHA, E_MIN_CHECKPOINTS, E_SEPARATE_PCT, E_WINDOWS, E_GRAPH, E_TWIN, E_REACTIVE,
    E_RANDOM, environments,
)
from scripts_cosim.unsaturated_scale_v2_read import _one_sample, pair_checkpoint_stats  # noqa: E402

__all__ = [
    "P_SEPARATE_PCT", "P_ALPHA", "P_MIN_ENVIRONMENTS", "P_MIN_CHECKPOINTS", "P_RUNGS",
    "P_PRIMARY_RUNG", "P_WINDOWS", "P_IMMEDIATE", "P_DRAIN", "P_BATCHED", "P_RULES",
    "P_REACTIVE", "P_ECT", "P_RANDOM", "P_GRAPH", "P_TWIN", "P_MECHANISM_MIN_ENVS",
    "V_IMMEDIATE_BEATS_REACTIVE", "V_REACTIVE_FASTER_THAN_IMMEDIATE", "V_NOT_SEP",
    "V_BATCHED_BEATS_REACTIVE", "V_REACTIVE_FASTER_THAN_BATCHED",
    "V_RULE_FASTER_THAN_GRAPH", "V_GRAPH_FASTER_THAN_RULE", "V_RULE_MATCHES_GRAPH",
    "V_IMMEDIATE_BEATS_RANDOM", "V_RANDOM_FASTER_THAN_IMMEDIATE", "V_RANDOM_NOT_SEP",
    "V_IMMEDIATE_FASTER_THAN_BATCHED", "V_BATCHED_FASTER_THAN_IMMEDIATE",
    "V_EXCHANGE_HELPS", "V_EXCHANGE_HURTS", "V_EXCHANGE_NOT_SEP",
    "V_COLOCATION_DELIVERED", "V_COLOCATION_NOT_DELIVERED",
    "V_RULE_WINS", "V_RULE_TIES", "V_RULE_LOSES",
    "environments", "env_stats", "broadcast_rule", "pair_checkpoint_stats",
    "read_g1", "read_g2", "read_g3", "read_g4", "read_g5", "read_g6", "read_g7", "read_g8",
]

# --- the bars: the chain's, with the unit named per read --------------------------------
P_SEPARATE_PCT = E_SEPARATE_PCT          # 5.0
P_ALPHA = E_ALPHA                        # 0.05
P_MIN_ENVIRONMENTS = 16                  # unit of a rule-vs-baseline read: the environment
P_MIN_CHECKPOINTS = E_MIN_CHECKPOINTS    # unit of a rule-vs-gnnedge0 read: the checkpoint
P_MECHANISM_MIN_ENVS = 12                # G7: exchange must fall on >= 12 of 16 environments

P_RUNGS = (40, 80)
P_PRIMARY_RUNG = 40
P_WINDOWS = E_WINDOWS

P_IMMEDIATE = "peer_greedy_network"
P_DRAIN = "drain_greedy_network"
P_BATCHED = "peer_greedy_network_batch"
P_RULES = (P_IMMEDIATE, P_DRAIN, P_BATCHED)

P_REACTIVE = E_REACTIVE                  # knative_network, from the ue_v1 screen
P_ECT = "knative_network_ect"            # from the ue_v1 study
P_RANDOM = E_RANDOM                      # random_network, from the ue_v1 study
P_GRAPH = E_GRAPH                        # be1670_gnnedge0, 16 checkpoints, ue_v1 study
P_TWIN = E_TWIN                          # 1670_mpoff (descriptive only)

# --- verdicts --------------------------------------------------------------------------
V_NOT_SEP = "NOT-SEPARATED"
V_IMMEDIATE_BEATS_REACTIVE = "IMMEDIATE-RULE-BEATS-REACTIVE"
V_REACTIVE_FASTER_THAN_IMMEDIATE = "REACTIVE-FASTER-THAN-IMMEDIATE-RULE"
V_BATCHED_BEATS_REACTIVE = "BATCHED-RULE-BEATS-REACTIVE"
V_REACTIVE_FASTER_THAN_BATCHED = "REACTIVE-FASTER-THAN-BATCHED-RULE"
V_RULE_FASTER_THAN_GRAPH = "RULE-FASTER-THAN-GRAPH-ARM"
V_GRAPH_FASTER_THAN_RULE = "GRAPH-ARM-FASTER-THAN-RULE"
V_RULE_MATCHES_GRAPH = "RULE-MATCHES-GRAPH-ARM"
V_IMMEDIATE_BEATS_RANDOM = "IMMEDIATE-RULE-BEATS-RANDOM"
V_RANDOM_FASTER_THAN_IMMEDIATE = "RANDOM-FASTER-THAN-IMMEDIATE-RULE"
V_RANDOM_NOT_SEP = "RANDOM-NOT-SEPARATED"
V_IMMEDIATE_FASTER_THAN_BATCHED = "IMMEDIATE-FASTER-THAN-BATCHED"
V_BATCHED_FASTER_THAN_IMMEDIATE = "BATCHED-FASTER-THAN-IMMEDIATE"
V_EXCHANGE_HELPS = "EXCHANGE-TERM-HELPS"
V_EXCHANGE_HURTS = "EXCHANGE-TERM-HURTS"
V_EXCHANGE_NOT_SEP = "EXCHANGE-TERM-NOT-SEPARATED"
V_COLOCATION_DELIVERED = "CO-LOCATION-DELIVERED"
V_COLOCATION_NOT_DELIVERED = "CO-LOCATION-NOT-DELIVERED"
V_RULE_WINS = "HAND-RULE-BEATS-REACTIVE-WITHOUT-WAITING"
V_RULE_TIES = "HAND-RULE-TIES-REACTIVE-WITHOUT-WAITING"
V_RULE_LOSES = "HAND-RULE-LOSES-TO-REACTIVE-WITHOUT-WAITING"


# --- statistics ------------------------------------------------------------------------

def env_stats(arm_by_env: Mapping[tuple, float], ref_by_env: Mapping[tuple, float],
              envs: Sequence[tuple]) -> dict:
    """{environment: 100 * (arm / ref - 1)} paired on the SAME environment. Negative = the
    arm is faster. Fails loud on a missing environment: a rule-vs-baseline read on a ragged
    design is a different design."""
    out = {}
    for e in envs:
        if e not in arm_by_env or e not in ref_by_env:
            raise ValueError(f"environment {e} missing from {'arm' if e not in arm_by_env else 'reference'}")
        out[e] = 100.0 * (float(arm_by_env[e]) / float(ref_by_env[e]) - 1.0)
    return out


def broadcast_rule(rule_by_env: Mapping[tuple, float], seeds: Sequence[int]) -> dict:
    """A deterministic rule as an `arm_by_env_seed` table: the same value under every
    checkpoint seed, so `pair_checkpoint_stats` pairs it against each checkpoint."""
    return {(e, int(s)): float(v) for e, v in rule_by_env.items() for s in seeds}


def _env_sample(stats: Mapping[tuple, float], *, faster_a: str, faster_b: str, tie: str) -> dict:
    return _one_sample(stats, min_n=P_MIN_ENVIRONMENTS, faster_a=faster_a, faster_b=faster_b, tie=tie)


# --- the reads -------------------------------------------------------------------------

def read_g1(immediate_vs_reactive: Mapping[tuple, float]) -> dict:
    """THE question: the no-wait rule vs reactive Knative, unit = environment."""
    return _env_sample(immediate_vs_reactive, faster_a=V_IMMEDIATE_BEATS_REACTIVE,
                       faster_b=V_REACTIVE_FASTER_THAN_IMMEDIATE, tie=V_NOT_SEP)


def read_g2(batched_vs_reactive: Mapping[tuple, float]) -> dict:
    """The batched rule vs reactive: the learned arms' serving configuration with a rule in
    the decoder's seat."""
    return _env_sample(batched_vs_reactive, faster_a=V_BATCHED_BEATS_REACTIVE,
                       faster_b=V_REACTIVE_FASTER_THAN_BATCHED, tie=V_NOT_SEP)


def read_g3(batched_rule_vs_graph: Mapping[int, float], *, min_n: Optional[int] = None) -> dict:
    """Does a hand rule with the same information match `gnnedge0` at the same window? Built
    with `pair_checkpoint_stats(broadcast_rule(rule, seeds), gnnedge0, envs)`, RULE FIRST;
    unit = checkpoint. Negative = the rule is faster. NOT-SEPARATED is the `matches` verdict:
    within the 5 % bar the model has learned nothing the rule does not encode."""
    return _one_sample(batched_rule_vs_graph, min_n=P_MIN_CHECKPOINTS if min_n is None else min_n,
                       faster_a=V_RULE_FASTER_THAN_GRAPH, faster_b=V_GRAPH_FASTER_THAN_RULE,
                       tie=V_RULE_MATCHES_GRAPH)


def read_g4(immediate_vs_random: Mapping[tuple, float]) -> dict:
    return _env_sample(immediate_vs_random, faster_a=V_IMMEDIATE_BEATS_RANDOM,
                       faster_b=V_RANDOM_FASTER_THAN_IMMEDIATE, tie=V_RANDOM_NOT_SEP)


def read_g5(immediate_vs_batched: Mapping[tuple, float]) -> dict:
    """The same score with and without the group wait, paired on the environment: the read
    that predicts the no-wait DECODER (the next lineage) before it is built."""
    return _env_sample(immediate_vs_batched, faster_a=V_IMMEDIATE_FASTER_THAN_BATCHED,
                       faster_b=V_BATCHED_FASTER_THAN_IMMEDIATE, tie=V_NOT_SEP)


def read_g6(immediate_vs_drain: Mapping[tuple, float]) -> dict:
    """The exchange term's own ablation: peer_greedy vs drain_greedy, paired on the
    environment. Negative = the exchange term helps."""
    return _env_sample(immediate_vs_drain, faster_a=V_EXCHANGE_HELPS,
                       faster_b=V_EXCHANGE_HURTS, tie=V_EXCHANGE_NOT_SEP)


def read_g7(exchange_by_env: Mapping[tuple, tuple], queue_by_env: Mapping[tuple, tuple]) -> dict:
    """Mechanism, per environment: (rule, reactive) peer exchange per task and platform queue
    per task. CO-LOCATION-DELIVERED iff the rule's exchange is below reactive's on at least
    `P_MECHANISM_MIN_ENVS` of the environments; the queue delta is carried as the cost side.
    Descriptive: it explains G1, never overrides it."""
    envs = sorted(exchange_by_env)
    if len(envs) < P_MIN_ENVIRONMENTS:
        return {"verdict": V_UNREADABLE, "n": len(envs),
                "reason": f"{len(envs)} environments < {P_MIN_ENVIRONMENTS}"}
    d_ex = {e: float(exchange_by_env[e][0]) - float(exchange_by_env[e][1]) for e in envs}
    d_q = {e: float(queue_by_env[e][0]) - float(queue_by_env[e][1]) for e in envs}
    fell = sum(v < 0 for v in d_ex.values())
    return {"verdict": V_COLOCATION_DELIVERED if fell >= P_MECHANISM_MIN_ENVS else V_COLOCATION_NOT_DELIVERED,
            "n": len(envs), "exchange_fell_on": fell, "bar": P_MECHANISM_MIN_ENVS,
            "median_exchange_delta_s": median(d_ex.values()),
            "median_queue_delta_s": median(d_q.values()),
            "why": (f"exchange per task fell on {fell}/{len(envs)} environments (median "
                    f"{median(d_ex.values()):+.3f} s); platform queue per task moved "
                    f"{median(d_q.values()):+.3f} s")}


def read_g8(g1: Mapping[str, object], g3: Mapping[str, object], g5: Mapping[str, object]) -> dict:
    """The composite, on the IMMEDIATE rule: WINS / TIES / LOSES against reactive without
    waiting, carrying whether the batched rule matched the graph arm (G3) and whether the
    no-wait configuration beat the batched one (G5). Unreadable if G1 is."""
    v1 = g1.get("verdict")
    if v1 == V_UNREADABLE:
        return {"verdict": V_UNREADABLE, "why": "G1 unreadable"}
    verdict = {V_IMMEDIATE_BEATS_REACTIVE: V_RULE_WINS, V_NOT_SEP: V_RULE_TIES,
               V_REACTIVE_FASTER_THAN_IMMEDIATE: V_RULE_LOSES}[v1]
    return {"verdict": verdict, "g1": v1, "graph_arm_matched": g3.get("verdict"),
            "no_wait_helps": g5.get("verdict"),
            "why": (f"immediate rule vs reactive: {v1}; batched rule vs gnnedge0: {g3.get('verdict')}; "
                    f"immediate vs batched: {g5.get('verdict')}")}
