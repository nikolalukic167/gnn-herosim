"""unsaturated_edge_v1 -- the bars, signed 2026-09-19 before any learned arm is served.

The programme's standing headline is `gnnedge0` vs reactive Knative at 6 servers / 40 clients:
-20.8 % (15/16), unsaturated. It was read on FOUR cells, all on arrival window w0.
`unsaturated_scale_v2` then showed, on the 80-server rung, that a 4-environment read on this
apparatus keeps every DIRECTION and inflates every MAGNITUDE 4-10x, because w0 is the burstiest
of the four windows the trace offers. That caution applies to the headline verbatim, and nothing
in the record has tested it there.

So this lineage gives the 6-server CLIENT rungs exactly v2's treatment and changes nothing else:
16 (topology, arrival window) environments per rung, fully crossed, the same checkpoints, the
same statistic, the same bars. 40 clients is the PRIMARY rung (where the headline lives; every
registered arm runs there); 80 clients is the SECONDARY rung (the graph arm alone, to read the
shape of the margin across load). `random_network` is a registered comparison in its own right,
not a decoration -- v2 measured it +8.4 % behind reactive with a bimodal tail to +782 %.
"""
from __future__ import annotations

import sys
from pathlib import Path
from statistics import median, pstdev
from typing import Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts_cosim.peer_only_v1_read import V_UNREADABLE  # noqa: E402
from scripts_cosim.unsaturated_scale_v2_read import (  # noqa: E402
    M_ALPHA, M_MIN_CHECKPOINTS, M_SEPARATE_PCT, M_DESIGN_SD_BAR, M_TARGET_SHARE,
    M_SATURATION_SHARE, _one_sample, checkpoint_stats, pair_checkpoint_stats, saturated,
)

__all__ = [
    "E_SEPARATE_PCT", "E_ALPHA", "E_MIN_CHECKPOINTS", "E_SERVERS", "E_RUNGS", "E_PRIMARY_RUNG",
    "E_TOPOLOGY_CANDIDATES", "E_WINDOWS", "E_STUDY_TOPOLOGIES", "E_TARGET_SHARE",
    "E_SATURATION_SHARE", "E_DESIGN_SD_BAR", "E_GRAPH", "E_PEER", "E_TWIN", "E_GNN",
    "E_ARMS_BY_RUNG", "E_REACTIVE", "E_RANDOM", "E_BASELINES",
    "V_DESIGN_READY", "V_DESIGN_SHORT", "V_ARM_BEATS_REACTIVE", "V_REACTIVE_FASTER", "V_NOT_SEP",
    "V_GRAPH_FASTER", "V_POINTWISE_FASTER", "V_SUM_COSTS", "V_SUM_HELPS", "V_SUM_NOT_SEP",
    "V_ARM_BEATS_RANDOM", "V_RANDOM_FASTER", "V_RANDOM_NOT_SEP",
    "V_POWER_DELIVERED", "V_POWER_NOT_DELIVERED", "V_WINDOW_CONSISTENT", "V_WINDOW_FLIPS",
    "V_HEADLINE_BOTH", "V_HEADLINE_PRIMARY", "V_HEADLINE_BELOW_BAR",
    "checkpoint_stats", "pair_checkpoint_stats", "saturated", "read_m0", "select_topologies",
    "read_e1", "read_e2", "read_e3", "read_e4", "read_e5", "read_e6", "read_e7",
]

# --- the bars: v2's, unchanged, so the two lineages are one measurement at two rungs ----
E_SEPARATE_PCT = M_SEPARATE_PCT          # 5.0
E_ALPHA = M_ALPHA                        # 0.05
E_MIN_CHECKPOINTS = M_MIN_CHECKPOINTS    # 16; the unit is the checkpoint
E_TARGET_SHARE = M_TARGET_SHARE          # 0.80 admits an environment
E_SATURATION_SHARE = M_SATURATION_SHARE  # 0.90 is saturated; "unknown is not a pass"
E_DESIGN_SD_BAR = M_DESIGN_SD_BAR        # 10 pp; v2 delivered 4.48 at 80 servers

E_SERVERS = 6
E_RUNGS = (40, 80)                       # clients; the 20-client rung is where EVERY arm loses
E_PRIMARY_RUNG = 40
# 12 candidates per rung. 9004 is absent on purpose: it hangs on every policy (peer_only_v1).
E_TOPOLOGY_CANDIDATES = (9001, 9002, 9003, 9005,
                         9101, 9102, 9103, 9104, 9105, 9106, 9107, 9108)
E_WINDOWS = ("w0", "w1", "w2", "w3")
E_STUDY_TOPOLOGIES = 4                   # 4 topologies x 4 windows = 16 environments per rung

E_GRAPH = "be1670_gnnedge0"              # the bipartite arm that carries the headline
E_PEER = "1670_peeronly"
E_TWIN = "1670_mpoff"                    # gnnedge0's corpus-matched MP-OFF twin
E_GNN = "1670_gnn"                       # the GIN-sum bipartite arm
E_ARMS_BY_RUNG = {40: (E_GRAPH, E_PEER, E_TWIN, E_GNN), 80: (E_GRAPH,)}

E_REACTIVE = "knative_network"
E_RANDOM = "random_network"
E_BASELINES = (E_REACTIVE, "knative_network_ect", E_RANDOM)

# --- verdicts --------------------------------------------------------------------------
V_DESIGN_READY = "DESIGN-READY"
V_DESIGN_SHORT = "TOO-FEW-UNSATURATED-ENVIRONMENTS"

V_ARM_BEATS_REACTIVE = "ARM-BEATS-REACTIVE-AT-THE-EDGE-RUNG"
V_REACTIVE_FASTER = "REACTIVE-FASTER-AT-THE-EDGE-RUNG"
V_NOT_SEP = "NOT-SEPARATED"

V_GRAPH_FASTER = "GRAPH-FASTER-AT-THE-EDGE-RUNG"
V_POINTWISE_FASTER = "POINTWISE-FASTER-AT-THE-EDGE-RUNG"

V_SUM_COSTS = "SUM-COSTS-AT-THE-EDGE-RUNG"
V_SUM_HELPS = "SUM-HELPS-AT-THE-EDGE-RUNG"
V_SUM_NOT_SEP = "SUM-NOT-SEPARATED-AT-THE-EDGE-RUNG"

V_ARM_BEATS_RANDOM = "ARM-BEATS-RANDOM-AT-THE-EDGE-RUNG"
V_RANDOM_FASTER = "RANDOM-FASTER-AT-THE-EDGE-RUNG"
V_RANDOM_NOT_SEP = "RANDOM-NOT-SEPARATED-AT-THE-EDGE-RUNG"

V_POWER_DELIVERED = "POWER-DELIVERED"
V_POWER_NOT_DELIVERED = "POWER-NOT-DELIVERED"

V_WINDOW_CONSISTENT = "SIGN-CONSISTENT-ACROSS-WINDOWS"
V_WINDOW_FLIPS = "SIGN-FLIPS-ACROSS-WINDOWS"

V_HEADLINE_BOTH = "HEADLINE-HOLDS-AT-POWER-ON-BOTH-RUNGS"
V_HEADLINE_PRIMARY = "HEADLINE-HOLDS-AT-POWER-ON-THE-PRIMARY-RUNG"
V_HEADLINE_BELOW_BAR = "HEADLINE-SHRINKS-BELOW-THE-BAR-AT-POWER"


def environments(rung: int, topologies: Sequence[int]) -> list:
    """The crossed environment set for one rung: [(rung, topology, window), ...]."""
    return [(int(rung), int(t), w) for t in topologies for w in E_WINDOWS]


# --- M0: the screen, and the selection rule --------------------------------------------

def read_m0(share_by_env: Mapping[tuple, Optional[float]]) -> dict:
    """Per (rung, topology, window): is reactive healthy enough for this to be a readable rung?

    A None or a missing entry is INADMISSIBLE, never assumed fine.
    """
    rows = {}
    for nc in E_RUNGS:
        for t in E_TOPOLOGY_CANDIDATES:
            for w in E_WINDOWS:
                qs = share_by_env.get((nc, t, w))
                rows[(nc, t, w)] = {
                    "queue_share": qs,
                    "saturated": saturated(qs),
                    "admissible": qs is not None and float(qs) <= E_TARGET_SHARE,
                }
    return {"environments": rows,
            "n_admissible": sum(r["admissible"] for r in rows.values()),
            "n_screened": len(rows)}


def select_topologies(m0: Mapping[str, object], rung: int) -> dict:
    """THE selection rule, per rung: the `E_STUDY_TOPOLOGIES` LOWEST-NUMBERED topology seeds
    admissible on ALL FOUR windows at that rung. v2's rule verbatim -- the seeds are arbitrary
    labels, so ordering by them cannot be steered by any arm; requiring all four windows keeps
    the design fully crossed."""
    rows = m0["environments"]
    qualified = [t for t in E_TOPOLOGY_CANDIDATES
                 if all(rows[(rung, t, w)]["admissible"] for w in E_WINDOWS)]
    if len(qualified) < E_STUDY_TOPOLOGIES:
        return {"verdict": V_DESIGN_SHORT, "rung": rung, "qualified": qualified,
                "why": f"C{rung}: only {len(qualified)} of {len(E_TOPOLOGY_CANDIDATES)} topologies "
                       f"are unsaturated on all {len(E_WINDOWS)} windows; {E_STUDY_TOPOLOGIES} are "
                       "needed for the crossed design"}
    chosen = sorted(qualified)[:E_STUDY_TOPOLOGIES]
    return {"verdict": V_DESIGN_READY, "rung": rung, "topologies": chosen, "qualified": qualified,
            "environments": environments(rung, chosen),
            "why": f"C{rung}: {len(qualified)} topologies unsaturated on all {len(E_WINDOWS)} "
                   f"windows; the {E_STUDY_TOPOLOGIES} lowest seed ids are {chosen}"}


# --- the reads ---------------------------------------------------------------------------
# The statistic is v2's: one value per checkpoint, the median over environments of its
# relative % against the reference ON THE SAME ENVIRONMENT (`checkpoint_stats`), or of arm A
# against arm B paired on environment AND checkpoint (`pair_checkpoint_stats`). Every read is
# then a one-sample signed-rank against zero under |median| >= 5 %, p < 0.05, n >= 16.

def read_e1(arm_stats: Mapping[int, float], *, min_n: Optional[int] = None) -> dict:
    """One arm vs reactive at one rung. Negative = the arm is faster."""
    return _one_sample(arm_stats, min_n=min_n, faster_a=V_ARM_BEATS_REACTIVE,
                       faster_b=V_REACTIVE_FASTER, tie=V_NOT_SEP)


def read_e2(graph_vs_twin: Mapping[int, float], *, min_n: Optional[int] = None) -> dict:
    """`gnnedge0` vs `mpoff`, both 1,670: the model-class contrast at matched corpus. Build with
    `pair_checkpoint_stats(gnnedge0, mpoff, envs)`, GRAPH ARM FIRST."""
    return _one_sample(graph_vs_twin, min_n=min_n, faster_a=V_GRAPH_FASTER,
                       faster_b=V_POINTWISE_FASTER, tie=V_NOT_SEP)


def read_e3(gnn_vs_gnnedge0: Mapping[int, float], *, min_n: Optional[int] = None) -> dict:
    """`gnn` (GIN, sum) vs `gnnedge0` (mean), GNN FIRST; positive = sum costs. bipartite_aggr_v1
    predicted and measured NO separation at 6 servers (3.55 candidates/task); this is that
    prediction at 16 environments."""
    return _one_sample(gnn_vs_gnnedge0, min_n=min_n, faster_a=V_SUM_HELPS,
                       faster_b=V_SUM_COSTS, tie=V_SUM_NOT_SEP)


def read_e4(arm_vs_random: Mapping[int, float], *, min_n: Optional[int] = None) -> dict:
    """One arm vs `random_network`, built with `checkpoint_stats(arm, random_by_env, envs)`.
    Negative = the arm is faster than random placement."""
    return _one_sample(arm_vs_random, min_n=min_n, faster_a=V_ARM_BEATS_RANDOM,
                       faster_b=V_RANDOM_FASTER, tie=V_RANDOM_NOT_SEP)


def read_e5(stats_by_arm: Mapping[str, Mapping[int, float]], rung: int) -> dict:
    """Did the design buy its power at this rung? sd of the checkpoint statistic, per registered
    arm, against `E_DESIGN_SD_BAR`. A bar on the APPARATUS, read before E7 is interpreted."""
    sds = {}
    for a in E_ARMS_BY_RUNG[rung]:
        s = stats_by_arm.get(a)
        sds[a] = None if not s or len(s) < E_MIN_CHECKPOINTS else pstdev(list(s.values()))
    known = {a: v for a, v in sds.items() if v is not None}
    if not known:
        return {"verdict": V_UNREADABLE, "rung": rung, "sd": sds,
                "why": "no arm carries a full checkpoint set"}
    worst = max(known.values())
    return {"verdict": V_POWER_DELIVERED if worst <= E_DESIGN_SD_BAR else V_POWER_NOT_DELIVERED,
            "rung": rung, "sd": sds, "worst_sd": worst, "bar": E_DESIGN_SD_BAR,
            "why": f"C{rung}: worst-arm sd {worst:.2f} pp against the registered "
                   f"{E_DESIGN_SD_BAR:.1f} pp; v2 delivered 4.48 at 80 servers"}


def read_e6(per_window_median: Mapping[str, Mapping[str, float]], rung: int) -> dict:
    """Does each arm keep its sign vs reactive across the four windows at this rung? v2 found
    `peeronly` and `mpoff_516` flipping. Descriptive: it qualifies E1, never overrides it."""
    rows = {}
    for a in E_ARMS_BY_RUNG[rung]:
        per = per_window_median.get(a) or {}
        missing = [w for w in E_WINDOWS if w not in per]
        if missing:
            rows[a] = {"verdict": V_UNREADABLE, "missing": missing}
            continue
        vals = [float(per[w]) for w in E_WINDOWS]
        rows[a] = {"verdict": V_WINDOW_CONSISTENT if len({v < 0 for v in vals}) == 1
                   else V_WINDOW_FLIPS,
                   "per_window": dict(zip(E_WINDOWS, vals)), "spread_pp": max(vals) - min(vals)}
    flips = sorted(a for a, r in rows.items() if r.get("verdict") == V_WINDOW_FLIPS)
    return {"rung": rung, "arms": rows, "flips": flips,
            "why": (f"C{rung}: every arm keeps its sign across the four arrival windows"
                    if not flips else
                    f"C{rung}: {', '.join(flips)} change sign between windows; their E1 number "
                    "is an average over a sign change and must never be quoted without this")}


def read_e7(e1_by_rung: Mapping[int, Mapping[str, Mapping[str, object]]],
            e5_by_rung: Mapping[int, Mapping[str, object]]) -> dict:
    """The composite, on the GRAPH ARM: does the headline hold at power?

    BOTH if `gnnedge0` clears E1 at 40 AND 80 clients; PRIMARY if at 40 only; otherwise
    BELOW-BAR. A BELOW-BAR read is `interpretable` only if E5 delivered the power at the
    primary rung -- an under-powered negative is reported as such, never as a tie. A positive
    stands either way: a bar that fires, fires. Other arms' E1 results are carried, not composed.
    """
    def _fires(nc):
        r = (e1_by_rung.get(nc) or {}).get(E_GRAPH) or {}
        return r.get("verdict") == V_ARM_BEATS_REACTIVE
    primary, secondary = _fires(E_PRIMARY_RUNG), _fires(80)
    powered = (e5_by_rung.get(E_PRIMARY_RUNG) or {}).get("verdict") == V_POWER_DELIVERED
    graph_readable = ((e1_by_rung.get(E_PRIMARY_RUNG) or {}).get(E_GRAPH) or {}).get("verdict") \
        not in (None, V_UNREADABLE)
    if not graph_readable:
        return {"verdict": V_UNREADABLE, "why": f"{E_GRAPH} is unreadable at C{E_PRIMARY_RUNG}"}
    if primary and secondary:
        return {"verdict": V_HEADLINE_BOTH, "interpretable": True,
                "why": f"{E_GRAPH} beats reactive under the chain's bar on 16 environments at "
                       "BOTH 40 and 80 clients"}
    if primary:
        return {"verdict": V_HEADLINE_PRIMARY, "interpretable": True,
                "why": f"{E_GRAPH} beats reactive under the chain's bar on 16 environments at "
                       "40 clients; not at 80"}
    why = (f"{E_GRAPH} does not clear the 5 % bar against reactive on 16 environments at "
           f"C{E_PRIMARY_RUNG}; the -20.8 % headline was a w0 number")
    if not powered:
        why += ("; but E5 says the delivered sd is above the registered design bar, so this is "
                "UNINTERPRETABLE, not a tie")
    return {"verdict": V_HEADLINE_BELOW_BAR, "interpretable": powered, "why": why}
