"""unsaturated_scale_v2 -- the bars, signed 2026-09-19 before any datum exists.

`unsaturated_scale_v1` closed `NO-LEARNED-ARM-BEATS-REACTIVE-AT-UNSATURATED-SCALE` with
`gnnedge0` at -6.46 % (p = 0.47) and `peeronly` at -7.97 % (p = 0.12) -- effects the right size
to matter, at a power that could not resolve them. Its own post-hoc decomposition says why, and
says it precisely:

  * the checkpoint statistic's sd scales as EXACTLY 1/sqrt(environments) for every arm, so the
    whole of it is (checkpoint x environment) interaction and none is a stable "this checkpoint
    is good" component -- 4 environments gives 14-29 pp against a 5-8 pp effect;
  * the simulator is deterministic, so 50,000 tasks carries ZERO sampling noise: task count is
    not a lever and running longer cannot tighten anything;
  * between the two halves of the single arrival window every gate has used, the arms' median
    against reactive swings 19 pp and FLIPS SIGN.

So v2 changes the replication unit and nothing else. An ENVIRONMENT is a (topology, arrival
window) pair; the study runs 4 x 4 = 16 of them, fully crossed, against v1's 4. Same rung, same
checkpoints, same bars, same statistic -- only n_environments moves, which is what makes v2 a
power replication of v1 rather than a new question.

Two things measured while building it, both recorded here because both changed the design:

  * `knative_network_batch` pays 0.002 s of batch wait, not the learned arms' 6.82 s. It is
    NOT a batching control. It is carried as a second reactive variant and labelled as one.
  * at a common rescale factor the four windows deliver 0.460/0.747/0.753/0.757 arrivals/s --
    window 0 is 1.6x slower than the rest of the trace -- so each window carries the factor
    that matches w0's rate and the window axis is arrival PATTERN, not load.
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
from scripts_cosim.partial_state_v3_read import wilcoxon_p  # noqa: E402

__all__ = [
    "M_SEPARATE_PCT", "M_ALPHA", "M_MIN_CHECKPOINTS", "M_SERVERS",
    "M_TOPOLOGY_CANDIDATES", "M_WINDOWS", "M_STUDY_TOPOLOGIES",
    "M_TARGET_SHARE", "M_SATURATION_SHARE", "M_DESIGN_SD_BAR",
    "M_ARMS", "M_GRAPH", "M_TWIN", "M_GNN", "M_BASELINES", "M_REACTIVE",
    "V_DESIGN_READY", "V_DESIGN_SHORT",
    "V_ARM_BEATS_REACTIVE", "V_REACTIVE_FASTER", "V_NOT_SEP",
    "V_GRAPH_FASTER", "V_POINTWISE_FASTER",
    "V_SUM_COSTS", "V_SUM_HELPS", "V_SUM_NOT_SEP",
    "V_LEARNED_BEATS_REACTIVE", "V_NO_LEARNED_BEATS_REACTIVE",
    "V_POWER_DELIVERED", "V_POWER_NOT_DELIVERED",
    "V_WINDOW_CONSISTENT", "V_WINDOW_FLIPS",
    "checkpoint_stats", "pair_checkpoint_stats", "read_m0", "select_topologies",
    "read_l1", "read_l2", "read_l3", "read_l4", "read_l5", "read_l6", "saturated",
]

# --- the bars, signed 2026-09-19 -------------------------------------------------------
M_SEPARATE_PCT = 5.0        # K/G/F/J_SEPARATE_PCT, unchanged down the whole chain
M_ALPHA = 0.05
M_MIN_CHECKPOINTS = 16      # the unit is the checkpoint; this does NOT move in v2

M_SERVERS = 80
# 12 candidates. 9004 is absent on purpose: it hangs on every policy (peer_only_v1).
M_TOPOLOGY_CANDIDATES = (9001, 9002, 9003, 9005,
                         9101, 9102, 9103, 9104, 9105, 9106, 9107, 9108)
M_WINDOWS = ("w0", "w1", "w2", "w3")
M_STUDY_TOPOLOGIES = 4      # 4 topologies x 4 windows = 16 environments

# v1's K_TARGET_SHARE / K_SATURATION_SHARE, unchanged. "Unknown is not a pass."
M_TARGET_SHARE = 0.80
M_SATURATION_SHARE = 0.90

# THE DESIGN-VALIDATION BAR. v1's decomposition projects sd 7-9 pp for the checkpoint statistic
# at 16 environments (it measured 14-29 pp at 4, scaling as 1/sqrt(n)). If the delivered sd is
# above this, the design did NOT buy the power it was built for and every L1 non-separation is
# reported as UNINTERPRETABLE, never as a tie. Signed before the data so that a disappointing
# sd cannot be re-described after the fact.
M_DESIGN_SD_BAR = 10.0

M_GRAPH = "be1670_gnnedge0"
M_TWIN = "1670_mpoff"
M_GNN = "1670_gnn"
M_ARMS = (M_GRAPH, "1670_peeronly", M_TWIN, "516_mpoff", M_GNN)

M_REACTIVE = "knative_network"
# `knative_network_batch` is a second REACTIVE VARIANT, not a batching control: it was measured
# at 0.002 s of batch wait against the learned arms' 6.82 s before it entered this list.
M_BASELINES = (M_REACTIVE, "knative_network_batch", "knative_network_ect", "random_network")

# --- verdicts --------------------------------------------------------------------------
V_DESIGN_READY = "DESIGN-READY"
V_DESIGN_SHORT = "TOO-FEW-UNSATURATED-ENVIRONMENTS"

V_ARM_BEATS_REACTIVE = "ARM-BEATS-REACTIVE-AT-UNSATURATED-SCALE"
V_REACTIVE_FASTER = "REACTIVE-FASTER-AT-UNSATURATED-SCALE"
V_NOT_SEP = "NOT-SEPARATED"

V_GRAPH_FASTER = "GRAPH-FASTER-AT-UNSATURATED-SCALE"
V_POINTWISE_FASTER = "POINTWISE-FASTER-AT-UNSATURATED-SCALE"

V_SUM_COSTS = "SUM-COSTS-AT-UNSATURATED-SCALE"
V_SUM_HELPS = "SUM-HELPS-AT-UNSATURATED-SCALE"
V_SUM_NOT_SEP = "SUM-NOT-SEPARATED-AT-UNSATURATED-SCALE"

V_LEARNED_BEATS_REACTIVE = "LEARNED-ARM-BEATS-REACTIVE-AT-UNSATURATED-SCALE"
V_NO_LEARNED_BEATS_REACTIVE = "NO-LEARNED-ARM-BEATS-REACTIVE-AT-UNSATURATED-SCALE"

V_POWER_DELIVERED = "POWER-DELIVERED"
V_POWER_NOT_DELIVERED = "POWER-NOT-DELIVERED"

V_WINDOW_CONSISTENT = "SIGN-CONSISTENT-ACROSS-WINDOWS"
V_WINDOW_FLIPS = "SIGN-FLIPS-ACROSS-WINDOWS"


def saturated(queue_share: Optional[float]) -> Optional[bool]:
    """None stays None: an unknown share renders UNKNOWN, never as a pass."""
    if queue_share is None:
        return None
    return float(queue_share) >= M_SATURATION_SHARE


# --- M0: the screen, and the selection rule --------------------------------------------

def read_m0(share_by_env: Mapping[tuple, Optional[float]]) -> dict:
    """Per environment: is reactive healthy enough for this to be a readable rung?

    `share_by_env` is {(topology_seed, window): reactive queue share}; a None or a missing
    entry is INADMISSIBLE, never assumed fine -- an arm that could not finish, or was never
    run, must not be able to admit an environment.
    """
    rows = {}
    for t in M_TOPOLOGY_CANDIDATES:
        for w in M_WINDOWS:
            qs = share_by_env.get((t, w))
            rows[(t, w)] = {
                "queue_share": qs,
                "saturated": saturated(qs),
                "admissible": qs is not None and float(qs) <= M_TARGET_SHARE,
            }
    n_ok = sum(r["admissible"] for r in rows.values())
    return {"environments": rows, "n_admissible": n_ok, "n_screened": len(rows)}


def select_topologies(m0: Mapping[str, object]) -> dict:
    """THE selection rule: the `M_STUDY_TOPOLOGIES` LOWEST-NUMBERED topology seeds that are
    admissible on ALL FOUR windows.

    Lowest-numbered, not best-performing and not closest-to-target: the seeds are arbitrary
    labels fixed by the topology generator, so ordering by them is independent of anything any
    arm does. Requiring all four windows keeps the design fully crossed -- a topology admissible
    on three windows would make the environment set ragged, and `collapse_to_seed`'s whole point
    is that two checkpoints are never compared on different environments.
    """
    rows = m0["environments"]
    qualified = [t for t in M_TOPOLOGY_CANDIDATES
                 if all(rows[(t, w)]["admissible"] for w in M_WINDOWS)]
    if len(qualified) < M_STUDY_TOPOLOGIES:
        return {"verdict": V_DESIGN_SHORT, "qualified": qualified,
                "why": f"only {len(qualified)} of {len(M_TOPOLOGY_CANDIDATES)} topologies are "
                       f"unsaturated on all {len(M_WINDOWS)} windows; {M_STUDY_TOPOLOGIES} are "
                       "needed for the crossed design. The study does not run on a ragged "
                       "environment set -- that is the finding, and it is about the baseline"}
    chosen = sorted(qualified)[:M_STUDY_TOPOLOGIES]
    return {"verdict": V_DESIGN_READY, "topologies": chosen, "qualified": qualified,
            "environments": [(t, w) for t in chosen for w in M_WINDOWS],
            "why": f"{len(qualified)} topologies unsaturated on all {len(M_WINDOWS)} windows; "
                   f"the {M_STUDY_TOPOLOGIES} lowest seed ids are {chosen}"}


# --- the statistic ----------------------------------------------------------------------

def checkpoint_stats(arm_by_env_seed: Mapping[tuple, float],
                     reactive_by_env: Mapping[tuple, float],
                     environments: Sequence[tuple]) -> dict:
    """One value per checkpoint: the median over environments of its RELATIVE % vs reactive
    ON THAT SAME ENVIRONMENT.

    v1 took the median of raw elapsed across cells and divided by reactive's median -- a ratio
    of medians, which does not pair. Here reactive's own elapsed ranges 20-46 s across the
    environments, so the pairing is where most of the leverage is, and a paired test deserves a
    paired statistic. Fails loud on a ragged arm rather than averaging over a partial set.
    """
    by_seed: dict = {}
    for (env, seed), v in arm_by_env_seed.items():
        by_seed.setdefault(int(seed), {})[env] = float(v)
    out = {}
    for seed, envs in by_seed.items():
        missing = [e for e in environments if e not in envs]
        if missing:
            raise ValueError(f"FAIL LOUD: checkpoint seed {seed} is missing environments {missing}")
        rel = []
        for e in environments:
            r = reactive_by_env.get(e)
            if r is None or float(r) <= 0:
                raise ValueError(f"FAIL LOUD: no reactive baseline for environment {e}")
            rel.append(100.0 * (envs[e] / float(r) - 1.0))
        out[seed] = median(rel)
    return out


def pair_checkpoint_stats(a_by_env_seed: Mapping[tuple, float],
                          b_by_env_seed: Mapping[tuple, float],
                          environments: Sequence[tuple]) -> dict:
    """One value per checkpoint: the median over environments of A vs B in %, paired on BOTH
    the environment and the checkpoint. Negative = A faster.

    Arm-vs-arm cannot be read by differencing two already-relative statistics: that is a ratio
    of ratios, it divides by a number that is near zero whenever an arm ties reactive, and its
    units are not the % the chain's 5 % bar is written in. Pairing the raw elapsed times is both
    correct and exactly the quantity the bar names.
    """
    def _by_seed(t):
        out: dict = {}
        for (env, seed), v in t.items():
            out.setdefault(int(seed), {})[env] = float(v)
        return out
    a_s, b_s = _by_seed(a_by_env_seed), _by_seed(b_by_env_seed)
    out = {}
    for seed in sorted(set(a_s) & set(b_s)):
        rel = []
        for e in environments:
            if e not in a_s[seed] or e not in b_s[seed]:
                raise ValueError(f"FAIL LOUD: checkpoint seed {seed} is missing environment {e}")
            b = b_s[seed][e]
            if b <= 0:
                raise ValueError(f"FAIL LOUD: non-positive baseline elapsed at {e}, seed {seed}")
            rel.append(100.0 * (a_s[seed][e] / b - 1.0))
        out[seed] = median(rel)
    return out


def _one_sample(stats: Mapping[int, float], *, min_n, faster_a, faster_b, tie) -> dict:
    """The chain's bar on a set of per-checkpoint percentages: |median| >= 5 %, p < 0.05,
    n >= 16, two-sided signed-rank against zero. Negative median favours the FIRST-named arm."""
    need = M_MIN_CHECKPOINTS if min_n is None else min_n
    seeds = sorted(stats)
    if len(seeds) < need:
        return {"verdict": V_UNREADABLE, "n": len(seeds),
                "reason": f"{len(seeds)} checkpoints < {need}"}
    vals = [float(stats[s]) for s in seeds]
    med = median(vals)
    p = wilcoxon_p(vals)
    if p is None:                       # every checkpoint exactly zero: no evidence either way
        p = 1.0
    if p < M_ALPHA and med <= -M_SEPARATE_PCT:
        verdict = faster_a
    elif p < M_ALPHA and med >= M_SEPARATE_PCT:
        verdict = faster_b
    else:
        verdict = tie
    return {"verdict": verdict, "n": len(seeds), "median": med, "p": p,
            "ahead": sum(v < 0 for v in vals),
            "bar": {"separate_pct": M_SEPARATE_PCT, "alpha": M_ALPHA,
                    "min_checkpoints": M_MIN_CHECKPOINTS}}


def read_l1(arm_stats: Mapping[int, float], *, min_n: Optional[int] = None) -> dict:
    """One arm vs reactive. `arm_stats` comes from `checkpoint_stats`, which already pairs each
    checkpoint against reactive on its own environment, so this is a ONE-SAMPLE test against
    zero. Negative = the arm is faster."""
    return _one_sample(arm_stats, min_n=min_n, faster_a=V_ARM_BEATS_REACTIVE,
                       faster_b=V_REACTIVE_FASTER, tie=V_NOT_SEP)


def read_l2(graph_vs_twin: Mapping[int, float], *, min_n: Optional[int] = None) -> dict:
    """`gnnedge0` vs `mpoff`, both 1,670 -- the model-class contrast at matched corpus. Build
    the argument with `pair_checkpoint_stats(gnnedge0, mpoff, envs)`, GRAPH ARM FIRST; a
    negative median then means the graph arm is faster."""
    return _one_sample(graph_vs_twin, min_n=min_n, faster_a=V_GRAPH_FASTER,
                       faster_b=V_POINTWISE_FASTER, tie=V_NOT_SEP)


def read_l3(gnn_vs_gnnedge0: Mapping[int, float], *, min_n: Optional[int] = None) -> dict:
    """`gnn` (GIN, sum) vs `gnnedge0` (mean). Build with
    `pair_checkpoint_stats(gnn, gnnedge0, envs)`, GNN FIRST; a POSITIVE median means sum costs.
    v1 read +26.69 % (p = 0.039) at 4 environments, so this is a REPLICATION at power of a bar
    that already fired, not a fresh look at a null."""
    return _one_sample(gnn_vs_gnnedge0, min_n=min_n, faster_a=V_SUM_HELPS,
                       faster_b=V_SUM_COSTS, tie=V_SUM_NOT_SEP)


def read_l4(l1_by_arm: Mapping[str, Mapping[str, object]], l5: Mapping[str, object]) -> dict:
    """The composite. Fires positive if ANY registered arm beats reactive under L1's bar.

    If L5 says the design did not deliver its power, a composite NEGATIVE is reported as
    UNINTERPRETABLE rather than as "no arm beats reactive" -- the whole point of v2 is that
    v1's nulls were under-powered, and repeating that mistake with a bigger n would be worse,
    not better. A composite POSITIVE stands either way: a bar that fires, fires.
    """
    missing = [a for a in M_ARMS
               if a not in l1_by_arm or l1_by_arm[a].get("verdict") == V_UNREADABLE]
    readable = {a: r for a, r in l1_by_arm.items()
                if a in M_ARMS and r.get("verdict") != V_UNREADABLE}
    if not readable:
        return {"verdict": V_UNREADABLE, "missing": missing,
                "why": "no registered arm is readable"}
    winners = sorted(a for a, r in readable.items() if r["verdict"] == V_ARM_BEATS_REACTIVE)
    losers = sorted(a for a, r in readable.items() if r["verdict"] == V_REACTIVE_FASTER)
    out = {"winners": winners, "losers": losers, "missing": missing}
    if winners:
        return {**out, "verdict": V_LEARNED_BEATS_REACTIVE,
                "why": f"{', '.join(winners)} beat reactive at 80 servers under the chain's bar, "
                       f"on {M_STUDY_TOPOLOGIES * len(M_WINDOWS)} environments where reactive is "
                       "not saturated"}
    powered = l5.get("verdict") == V_POWER_DELIVERED
    why = ("no registered arm beats reactive where the baseline is healthy, at "
           f"{M_STUDY_TOPOLOGIES * len(M_WINDOWS)} environments")
    if not powered:
        why += ("; but L5 says the delivered sd is above the registered design bar, so this is "
                "UNINTERPRETABLE, not a tie -- the same defect v1 had, at a larger n")
    return {**out, "verdict": V_NO_LEARNED_BEATS_REACTIVE, "interpretable": powered, "why": why}


def read_l5(stats_by_arm: Mapping[str, Mapping[int, float]]) -> dict:
    """Did the design buy the power it was built for? The registered quantity is the sd of the
    checkpoint statistic, per arm, against `M_DESIGN_SD_BAR`.

    This is a bar on the APPARATUS, not on any arm, and it is read before L4 is interpreted.
    """
    sds = {}
    for a in M_ARMS:
        s = stats_by_arm.get(a)
        if not s or len(s) < M_MIN_CHECKPOINTS:
            sds[a] = None
            continue
        sds[a] = pstdev(list(s.values()))
    known = {a: v for a, v in sds.items() if v is not None}
    if not known:
        return {"verdict": V_UNREADABLE, "sd": sds, "why": "no arm carries a full checkpoint set"}
    worst = max(known.values())
    ok = worst <= M_DESIGN_SD_BAR
    return {"verdict": V_POWER_DELIVERED if ok else V_POWER_NOT_DELIVERED,
            "sd": sds, "worst_sd": worst, "bar": M_DESIGN_SD_BAR,
            "why": (f"worst-arm sd {worst:.2f} pp against the registered {M_DESIGN_SD_BAR:.1f} pp; "
                    f"v1 measured 14-29 pp at 4 environments")}


def read_l6(per_window_median: Mapping[str, Mapping[str, float]]) -> dict:
    """Does the arm-vs-reactive result hold across arrival windows, or flip with the window?

    `per_window_median` is {arm: {window: median relative %}}. v1's split-half found a 19 pp
    swing and a SIGN FLIP inside one window, which is the reason the window axis exists. An arm
    whose sign is not the same on all four windows is reported as such, and any quote of its L1
    number must carry that. Descriptive by registration -- it qualifies L1, never overrides it.
    """
    rows = {}
    for a in M_ARMS:
        per = per_window_median.get(a) or {}
        missing = [w for w in M_WINDOWS if w not in per]
        if missing:
            rows[a] = {"verdict": V_UNREADABLE, "missing": missing}
            continue
        vals = [float(per[w]) for w in M_WINDOWS]
        signs = {v < 0 for v in vals}
        rows[a] = {"verdict": V_WINDOW_CONSISTENT if len(signs) == 1 else V_WINDOW_FLIPS,
                   "per_window": dict(zip(M_WINDOWS, vals)),
                   "spread_pp": max(vals) - min(vals)}
    flips = sorted(a for a, r in rows.items() if r.get("verdict") == V_WINDOW_FLIPS)
    return {"arms": rows, "flips": flips,
            "why": ("every arm keeps its sign across the four arrival windows"
                    if not flips else
                    f"{', '.join(flips)} change sign between windows; their L1 number is an "
                    "average over a sign change and must never be quoted without this")}
