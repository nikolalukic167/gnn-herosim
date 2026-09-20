"""env_lever_v1 -- the bars shared by burst_groups_v1, payload_scale_v1 and backbone_sparsity_v1,
signed 2026-09-20 before any levered arm ran.

`unsaturated_edge_v1` / `batch_window_edge_v1` / `peer_greedy_live_v1` fixed the mechanism at
the study's operating point: a task's peer group arrives over 13.8 s (median; p90 32 s), the
learned arms wait ~7 s per task to see it, and co-location is worth ~1 s of exchange per task at
the 200 MB payload scale over a 1 Gbps backbone. Three levers move the ENVIRONMENT, one field
each, and re-run the same 16-environment gate (`scripts_cosim/env_lever_v1_mint.py`):

  burst   peer groups dispatched as one burst (the environmental twin of no-wait decoding)
  pk0.1 / pk10   payload scale 20 MB / 2 GB (the exchange-vs-queue ratio; the crossing point)
  p04 / bw250    sparser client-server graph / a 4x slower backbone (distance)

Every lever changes reactive's load, so each is screened for admissibility on its own
(`read_m0` / `select_topologies` from unsaturated_edge_v1 at C40; unknown is not a pass) and
its study runs on the 4 topologies its own screen selects. Arms per lever: reactive (screen),
random, the two rule arms, `gnnedge0` x 16 checkpoints; `mpoff` x 16 for burst only.

Units as in peer_greedy_live_v1: a learned arm's read is per CHECKPOINT (median over the 16
environments of its paired relative %, n = 16); a rule's read is per ENVIRONMENT (n = 16); a
rule-vs-learned read pairs the rule's value against every checkpoint. Bars are the chain's:
|median| >= 5 %, p < 0.05, two-sided signed-rank.
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
    E_GRAPH, E_MIN_CHECKPOINTS, E_RANDOM, E_REACTIVE, E_TWIN, E_WINDOWS, V_DESIGN_READY,
    V_DESIGN_SHORT, checkpoint_stats, environments, pair_checkpoint_stats, read_m0,
    select_topologies,
)
from scripts_cosim.peer_greedy_live_v1_read import (  # noqa: E402
    P_BATCHED, P_IMMEDIATE, P_MIN_ENVIRONMENTS, broadcast_rule, env_stats,
)
from scripts_cosim.unsaturated_scale_v2_read import _one_sample  # noqa: E402

__all__ = [
    "L_LEVERS", "L_LINEAGE_OF", "L_RUNG", "L_WINDOWS", "L_GRAPH", "L_TWIN", "L_IMMEDIATE",
    "L_BATCHED", "L_RANDOM", "L_REACTIVE", "L_ARMS", "L_WAIT_COLLAPSED_S", "L_PAYLOAD_SCALES",
    "V_DESIGN_READY", "V_DESIGN_SHORT", "V_ARM_BEATS_REACTIVE", "V_REACTIVE_FASTER", "V_NOT_SEP",
    "V_RULE_BEATS_REACTIVE", "V_REACTIVE_FASTER_THAN_RULE", "V_GRAPH_FASTER_THAN_RULE",
    "V_RULE_FASTER_THAN_GRAPH", "V_GRAPH_MATCHES_RULE", "V_GRAPH_FASTER", "V_POINTWISE_FASTER",
    "V_WAIT_COLLAPSED", "V_WAIT_DID_NOT_COLLAPSE", "V_REGIME_FROM", "V_NO_REGIME",
    "environments", "checkpoint_stats", "pair_checkpoint_stats", "env_stats", "broadcast_rule",
    "read_m0", "select_topologies", "read_l0", "read_l1", "read_l2", "read_l3", "read_l4",
    "read_l5", "read_l6",
]

L_LEVERS = ("burst", "pk0.1", "pk10", "p04", "bw250")
L_LINEAGE_OF = {"burst": "burst_groups_v1", "pk0.1": "payload_scale_v1", "pk10": "payload_scale_v1",
                "p04": "backbone_sparsity_v1", "bw250": "backbone_sparsity_v1"}
L_RUNG = 40
L_WINDOWS = E_WINDOWS
L_GRAPH = E_GRAPH                # be1670_gnnedge0
L_TWIN = E_TWIN                  # 1670_mpoff (burst only)
L_IMMEDIATE = P_IMMEDIATE
L_BATCHED = P_BATCHED
L_RANDOM = E_RANDOM
L_REACTIVE = E_REACTIVE
L_ARMS = {lever: (L_RANDOM, L_IMMEDIATE, L_BATCHED, L_GRAPH) + ((L_TWIN,) if lever == "burst" else ())
          for lever in L_LEVERS}
L_WAIT_COLLAPSED_S = 1.0         # burst L4: gnnedge0's median scheduler wait per task must fall below this
L_PAYLOAD_SCALES = (0.1, 1.0, 10.0)   # k = 1 is the study's own operating point (ue_v1 / pg_v1)

V_ARM_BEATS_REACTIVE = "ARM-BEATS-REACTIVE-UNDER-THE-LEVER"
V_REACTIVE_FASTER = "REACTIVE-FASTER-UNDER-THE-LEVER"
V_NOT_SEP = "NOT-SEPARATED"
V_RULE_BEATS_REACTIVE = "RULE-BEATS-REACTIVE-UNDER-THE-LEVER"
V_REACTIVE_FASTER_THAN_RULE = "REACTIVE-FASTER-THAN-RULE-UNDER-THE-LEVER"
V_GRAPH_FASTER_THAN_RULE = "GRAPH-ARM-FASTER-THAN-RULE"
V_RULE_FASTER_THAN_GRAPH = "RULE-FASTER-THAN-GRAPH-ARM"
V_GRAPH_MATCHES_RULE = "GRAPH-ARM-MATCHES-RULE"
V_GRAPH_FASTER = "GRAPH-FASTER-THAN-TWIN-UNDER-THE-LEVER"
V_POINTWISE_FASTER = "POINTWISE-FASTER-THAN-GRAPH-UNDER-THE-LEVER"
V_WAIT_COLLAPSED = "WAIT-COLLAPSED"
V_WAIT_DID_NOT_COLLAPSE = "WAIT-DID-NOT-COLLAPSE"
V_REGIME_FROM = "CO-LOCATION-PAYS-FROM"
V_NO_REGIME = "CO-LOCATION-PAYS-AT-NO-MEASURED-SCALE"


def read_l0(share_by_env: Mapping[tuple, Optional[float]]) -> dict:
    """The lever's own admissibility screen at C40: `read_m0` + `select_topologies`, the
    registered rule verbatim. Missing = inadmissible."""
    m0 = read_m0(share_by_env)
    sel = select_topologies(m0, L_RUNG)
    return {"m0": m0, "selection": sel, "verdict": sel["verdict"]}


def read_l1(arm_stats: Mapping[int, float], *, min_n: Optional[int] = None) -> dict:
    """A learned arm vs reactive under the lever, unit = checkpoint. Negative = arm faster."""
    return _one_sample(arm_stats, min_n=E_MIN_CHECKPOINTS if min_n is None else min_n,
                       faster_a=V_ARM_BEATS_REACTIVE, faster_b=V_REACTIVE_FASTER, tie=V_NOT_SEP)


def read_l2(rule_vs_reactive: Mapping[tuple, float]) -> dict:
    """A rule arm vs reactive under the lever, unit = environment."""
    return _one_sample(rule_vs_reactive, min_n=P_MIN_ENVIRONMENTS, faster_a=V_RULE_BEATS_REACTIVE,
                       faster_b=V_REACTIVE_FASTER_THAN_RULE, tie=V_NOT_SEP)


def read_l3(graph_vs_rule: Mapping[int, float], *, min_n: Optional[int] = None) -> dict:
    """`gnnedge0` vs the immediate rule, GRAPH FIRST, paired on environment and checkpoint.
    Negative = the graph arm is faster. The tie is `GRAPH-ARM-MATCHES-RULE`."""
    return _one_sample(graph_vs_rule, min_n=E_MIN_CHECKPOINTS if min_n is None else min_n,
                       faster_a=V_GRAPH_FASTER_THAN_RULE, faster_b=V_RULE_FASTER_THAN_GRAPH,
                       tie=V_GRAPH_MATCHES_RULE)


def read_l4(wait_by_env_seed: Mapping[tuple, float]) -> dict:
    """burst only: did the lever do what its name says? Median over (environment, checkpoint)
    of `gnnedge0`'s scheduler wait per task against `L_WAIT_COLLAPSED_S`. If it did not
    collapse, L1 is not a burst result."""
    if not wait_by_env_seed:
        return {"verdict": V_UNREADABLE, "reason": "no wait values"}
    m = median(float(v) for v in wait_by_env_seed.values())
    return {"verdict": V_WAIT_COLLAPSED if m < L_WAIT_COLLAPSED_S else V_WAIT_DID_NOT_COLLAPSE,
            "median_wait_s": m, "bar_s": L_WAIT_COLLAPSED_S, "n": len(wait_by_env_seed)}


def read_l5(graph_vs_twin: Mapping[int, float], *, min_n: Optional[int] = None) -> dict:
    """burst only: `gnnedge0` vs `mpoff` under the lever, GRAPH FIRST."""
    return _one_sample(graph_vs_twin, min_n=E_MIN_CHECKPOINTS if min_n is None else min_n,
                       faster_a=V_GRAPH_FASTER, faster_b=V_POINTWISE_FASTER, tie=V_NOT_SEP)


def read_l6(verdict_by_scale: Mapping[float, str], *, beats: str) -> dict:
    """payload_scale_v1's regime read: over k in ascending order, the SMALLEST scale at which
    the named `beats` verdict fires, given that it also fires at every larger measured scale;
    otherwise no regime. Descriptive -- it composes L1/L2 across scales, it does not test."""
    scales = sorted(verdict_by_scale)
    firing = [k for k in scales if verdict_by_scale[k] == beats]
    if not firing:
        return {"verdict": V_NO_REGIME, "scales": {k: verdict_by_scale[k] for k in scales}}
    k0 = min(firing)
    monotone = all(verdict_by_scale[k] == beats for k in scales if k >= k0)
    return {"verdict": f"{V_REGIME_FROM}-x{k0:g}" if monotone else V_NO_REGIME,
            "from_scale": k0 if monotone else None, "monotone": monotone,
            "scales": {k: verdict_by_scale[k] for k in scales}}
