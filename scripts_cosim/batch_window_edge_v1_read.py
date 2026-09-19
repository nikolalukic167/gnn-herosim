"""batch_window_edge_v1 -- the bars, signed 2026-09-19 before any arm at a new window is served.

Every learned arm in this programme collects a task's whole peer group (10 tasks) before it
decodes, waiting up to `scheduler.batch_timeout` = 16 s for members that have not arrived. That
wait is 6.8 s per task at every operating point measured (`peer_only_v1` B7, `unsaturated_scale_v2`),
and reactive Knative pays 0.000 s of it. Group arrival spans are median 13.8 s, p90 32.3 s, so a
16 s window closes about half the groups complete and times out on the rest. At 80 servers the
tax is 37 % of reactive's elapsed and the arms end up +-2 % of reactive; at 6 servers / 40
clients on window w0 the arm halves reactive's queue and still keeps 20 % after the tax.

The window is a POLICY CONSTANT of the learned arm, never retrained, never tuned: it was set once
(`drainable_regime_v1`) and `drainable_serving_config_v1` swept it only at the 20-client rung,
where every learned arm loses to reactive by >= 70 % at every window. Nobody has moved it where
the arm is competitive. A shorter window closes groups earlier and decodes the late members in a
later batch conditioned on the prefix already placed (partial_state_v3) -- less wait, more peers
placed apart. Whether that trade is a net win is an empirical question, and this lineage asks it
with a SCREEN / CONFIRM split so the window is chosen on environments the verdict never sees.

Design (6 servers, 40 clients, the primary rung of `unsaturated_edge_v1`):
  * SCREEN: `gnnedge0` at 2 / 4 / 8 s on the LOWEST-SEED study topology x 4 windows, 16
    checkpoints; the 16 s arm is the main study's. The window with the lowest median checkpoint
    statistic vs reactive is chosen. 16 s winning closes the lineage WINDOW-NOT-THE-LEVER.
  * CONFIRM: the chosen window on the 12 HELD-OUT environments (the other 3 topologies x 4
    windows), `gnnedge0` and its MP-OFF twin, 16 checkpoints each. Reactive and random do not
    batch, so their study runs are the reference unchanged.
Bars are the chain's: |median| >= 5 %, p < 0.05, n >= 16 checkpoints, signed-rank against zero.
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
from scripts_cosim.unsaturated_scale_v2_read import _one_sample  # noqa: E402
from scripts_cosim.unsaturated_edge_v1_read import (  # noqa: E402
    E_ALPHA, E_GRAPH, E_MIN_CHECKPOINTS, E_SEPARATE_PCT, E_TWIN, E_WINDOWS,
    checkpoint_stats, pair_checkpoint_stats,
)

__all__ = [
    "W_SEPARATE_PCT", "W_ALPHA", "W_MIN_CHECKPOINTS", "W_RUNG", "W_WINDOWS_S", "W_BASE_WINDOW_S",
    "W_GRAPH", "W_TWIN", "W_SCREEN_TOPOLOGIES", "W_HELD_OUT_TOPOLOGIES",
    "V_WINDOW_NOT_THE_LEVER", "V_WINDOW_SELECTED",
    "V_SHORTER_BEATS_REACTIVE", "V_REACTIVE_FASTER", "V_NOT_SEP",
    "V_SHORTER_FASTER", "V_LONGER_FASTER",
    "V_GRAPH_FASTER", "V_POINTWISE_FASTER",
    "V_SHORTER_BEATS_RANDOM", "V_RANDOM_FASTER", "V_RANDOM_NOT_SEP",
    "V_WAIT_FELL", "V_WAIT_DID_NOT_FALL",
    "checkpoint_stats", "pair_checkpoint_stats", "split_environments", "select_window",
    "read_w1", "read_w2", "read_w3", "read_w4", "read_w5", "read_w6",
]

# --- the bars: the chain's, unchanged -----------------------------------------------------
W_SEPARATE_PCT = E_SEPARATE_PCT        # 5.0
W_ALPHA = E_ALPHA                      # 0.05
W_MIN_CHECKPOINTS = E_MIN_CHECKPOINTS  # 16

W_RUNG = 40                            # clients, 6 servers: unsaturated_edge_v1's primary rung
W_WINDOWS_S = (2.0, 4.0, 8.0)          # the screened windows; 16 s is the incumbent
W_BASE_WINDOW_S = 16.0
W_GRAPH = E_GRAPH                      # be1670_gnnedge0
W_TWIN = E_TWIN                        # 1670_mpoff
W_SCREEN_TOPOLOGIES = 1                # the LOWEST-seed study topology screens; the other 3 confirm
W_HELD_OUT_TOPOLOGIES = 3

# --- verdicts --------------------------------------------------------------------------
V_WINDOW_NOT_THE_LEVER = "WINDOW-NOT-THE-LEVER"
V_WINDOW_SELECTED = "WINDOW-SELECTED"

V_SHORTER_BEATS_REACTIVE = "SHORTER-WINDOW-BEATS-REACTIVE-HELD-OUT"
V_REACTIVE_FASTER = "REACTIVE-FASTER-HELD-OUT"
V_NOT_SEP = "NOT-SEPARATED"

V_SHORTER_FASTER = "SHORTER-WINDOW-FASTER-THAN-16S-HELD-OUT"
V_LONGER_FASTER = "16S-FASTER-THAN-SHORTER-WINDOW-HELD-OUT"

V_GRAPH_FASTER = "GRAPH-FASTER-AT-MATCHED-WINDOW"
V_POINTWISE_FASTER = "POINTWISE-FASTER-AT-MATCHED-WINDOW"

V_SHORTER_BEATS_RANDOM = "SHORTER-WINDOW-BEATS-RANDOM-HELD-OUT"
V_RANDOM_FASTER = "RANDOM-FASTER-HELD-OUT"
V_RANDOM_NOT_SEP = "RANDOM-NOT-SEPARATED-HELD-OUT"

V_WAIT_FELL = "BATCH-WAIT-FELL"
V_WAIT_DID_NOT_FALL = "BATCH-WAIT-DID-NOT-FALL"


def split_environments(study_topologies: Sequence[int]) -> dict:
    """THE split: the lowest-numbered study topology screens, the other three confirm. Seeds are
    arbitrary labels fixed by the generator, so the split cannot be steered by any arm."""
    topos = sorted(int(t) for t in study_topologies)
    if len(topos) != W_SCREEN_TOPOLOGIES + W_HELD_OUT_TOPOLOGIES:
        raise ValueError(f"FAIL LOUD: {len(topos)} study topologies, the design needs "
                         f"{W_SCREEN_TOPOLOGIES + W_HELD_OUT_TOPOLOGIES}")
    screen, held = topos[:W_SCREEN_TOPOLOGIES], topos[W_SCREEN_TOPOLOGIES:]
    return {"screen_topologies": screen, "held_out_topologies": held,
            "screen": [(W_RUNG, t, w) for t in screen for w in E_WINDOWS],
            "held_out": [(W_RUNG, t, w) for t in held for w in E_WINDOWS]}


def select_window(stats_by_window: Mapping[float, Mapping[int, float]]) -> dict:
    """THE selection rule, on the SCREEN environments only: the window whose median checkpoint
    statistic vs reactive is lowest, among the windows carrying a full checkpoint set. A window
    that could not deliver all 16 checkpoints (a hang, a failure) is ineligible -- unknown is not
    a pass. If the incumbent 16 s wins, there is nothing to confirm."""
    medians = {}
    for w, st in stats_by_window.items():
        if st and len(st) >= W_MIN_CHECKPOINTS:
            medians[float(w)] = median(st.values())
    if W_BASE_WINDOW_S not in medians:
        return {"verdict": V_UNREADABLE, "medians": medians,
                "why": "the incumbent 16 s window carries no full checkpoint set on the screen"}
    best = min(medians, key=lambda w: (medians[w], w))
    if best == W_BASE_WINDOW_S:
        return {"verdict": V_WINDOW_NOT_THE_LEVER, "window_s": best, "medians": medians,
                "why": f"16 s has the lowest screen median ({medians[best]:+.2f} %); no shorter "
                       "window beats it, nothing to confirm"}
    return {"verdict": V_WINDOW_SELECTED, "window_s": best, "medians": medians,
            "why": f"{best:g} s has the lowest screen median ({medians[best]:+.2f} % vs "
                   f"{medians[W_BASE_WINDOW_S]:+.2f} % at 16 s); confirm on the held-out 12"}


def read_w1(arm_stats: Mapping[int, float], *, min_n: Optional[int] = None) -> dict:
    """`gnnedge0` at the chosen window vs reactive on the HELD-OUT environments. Negative = the
    arm is faster. This is the read the lineage exists for."""
    return _one_sample(arm_stats, min_n=min_n, faster_a=V_SHORTER_BEATS_REACTIVE,
                       faster_b=V_REACTIVE_FASTER, tie=V_NOT_SEP)


def read_w2(short_vs_base: Mapping[int, float], *, min_n: Optional[int] = None) -> dict:
    """`gnnedge0` at the chosen window vs `gnnedge0` at 16 s, paired on environment AND
    checkpoint, held-out. Build with `pair_checkpoint_stats(short, base, envs)`, SHORT FIRST."""
    return _one_sample(short_vs_base, min_n=min_n, faster_a=V_SHORTER_FASTER,
                       faster_b=V_LONGER_FASTER, tie=V_NOT_SEP)


def read_w3(graph_vs_twin: Mapping[int, float], *, min_n: Optional[int] = None) -> dict:
    """`gnnedge0` vs `mpoff`, BOTH at the chosen window, held-out. GRAPH FIRST. The twin pays the
    same window, so this is the model-class contrast with the tax matched."""
    return _one_sample(graph_vs_twin, min_n=min_n, faster_a=V_GRAPH_FASTER,
                       faster_b=V_POINTWISE_FASTER, tie=V_NOT_SEP)


def read_w4(arm_vs_random: Mapping[int, float], *, min_n: Optional[int] = None) -> dict:
    """`gnnedge0` at the chosen window vs `random_network`, held-out."""
    return _one_sample(arm_vs_random, min_n=min_n, faster_a=V_SHORTER_BEATS_RANDOM,
                       faster_b=V_RANDOM_FASTER, tie=V_RANDOM_NOT_SEP)


def read_w5(per_window_median: Mapping[str, float]) -> dict:
    """Sign per arrival window on the held-out set, descriptive: qualifies W1, never overrides."""
    missing = [w for w in E_WINDOWS if w not in per_window_median]
    if missing:
        return {"verdict": V_UNREADABLE, "missing": missing}
    vals = [float(per_window_median[w]) for w in E_WINDOWS]
    return {"verdict": ("SIGN-CONSISTENT-ACROSS-WINDOWS" if len({v < 0 for v in vals}) == 1
                        else "SIGN-FLIPS-ACROSS-WINDOWS"),
            "per_window": dict(zip(E_WINDOWS, vals)), "spread_pp": max(vals) - min(vals)}


def read_w6(wait_short: Mapping[tuple, float], wait_base: Mapping[tuple, float]) -> dict:
    """The mechanism check: the median per-task batch wait at the chosen window must be below
    the 16 s arm's, on the same (environment, checkpoint) keys. If the wait did not fall, the
    window was not what moved and W1 must not be read as a batching result."""
    keys = sorted(set(wait_short) & set(wait_base))
    if not keys:
        return {"verdict": V_UNREADABLE, "why": "no shared (environment, checkpoint) keys"}
    ms, mb = median(wait_short[k] for k in keys), median(wait_base[k] for k in keys)
    return {"verdict": V_WAIT_FELL if ms < mb else V_WAIT_DID_NOT_FALL,
            "median_wait_short_s": ms, "median_wait_base_s": mb, "n": len(keys)}
