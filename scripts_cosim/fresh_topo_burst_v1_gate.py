#!/usr/bin/env python3
"""fresh_topo_burst_v1 -- local driver for the screen, the venue-parity check and the gate.
See docs/lineages/fresh_topo_burst_v1.md.

Every task runs `src/executesimulation.py` once, in its own memory-capped cgroup scope with a
wall-clock timeout (a hung run is a FAILED task, never a skipped one), and writes a summary in
`selfpredict_burst_v1_gate.sbatch`'s format. Existing summaries are not re-run.

  fresh_topo_burst_v1_gate.py screen --inputs DIR --out DIR
  fresh_topo_burst_v1_gate.py parity --inputs DIR --out DIR
  fresh_topo_burst_v1_gate.py gate   --inputs DIR --out DIR --selection selected.json

--inputs holds cfg/ (cc40s<topo>.json), wl/ (burst workloads), models/ (jb2 checkpoints + sidecars)
and joint_burst_v2_split.json, all md5-verified against datalab before use.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import shutil
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

from capped_log import cap_bytes, run_logged
import progress_watchdog as pw

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WINDOWS = ("w0", "w1", "w2", "w3")
SEEDS = (1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 14, 15, 16)
OLD_POOL = (9001, 9002, 9003, 9005, 9102, 9103, 9104, 9105, 9107, 9108, 9109, 9110, 9111, 9112, 9113,
            9115, 9117, 9118, 9119, 9120, 9121, 9122, 9123, 9124)
NEW_POOL = tuple(range(9401, 9425))
EXT_POOL = tuple(range(9425, 9473))  # amendment A2
CANDIDATES = OLD_POOL + NEW_POOL + EXT_POOL
# raw_plan_v2 Phase C: unseen confirmation topologies, screened under the same rule (phase rp2screen); 9521-9568 is
# its Amendment A1 extension (0 of 9473-9520 admitted)
RP2C_POOL = tuple(range(9473, 9569))
RULE_POLICY = {
    "reactive": "knative_network",
    "batched": "peer_greedy_network_batch",
    "selfpredict": "peer_greedy_selfpredict_network",
    "cd": "peer_greedy_network_cd",
    "cd_blind": "peer_greedy_network_cd",  # cd_gap_v1 D1: HEROSIM_PG_BATCH_BLIND=1
    "cd_slate": "peer_greedy_network_cd",  # cd_gap_v1 D4: GNN_SERVE_CORPUS_SLATE=1
    "cd_inflight": "peer_greedy_network_cd",  # burst_ladder_v1: HEROSIM_PG_INFLIGHT=1
    "cd_pull": "peer_greedy_network_cd",  # accel pull-hold control: HEROSIM_PG_PULL_HOLD=1 + HEROSIM_PULL_LEDGER=1
    "cd_ledger": "peer_greedy_network_cd",  # identity check: the pull ledger alone (HEROSIM_PULL_LEDGER=1) must leave CD bit-identical
    "cd_declared": "peer_greedy_network_cd",  # r1_attribution_v1: CD over the learned arms' declared slate (GNN_SERVE_CANDIDATE_SLATE=declared_pruning_v1); descriptive
    "cd_expand": "peer_greedy_network_cd",  # cd_expand: CD + exact-move alpha-expansion in its refine (HEROSIM_PG_CD_EXPANSION=1); descriptive, the stronger hand control
    "cd_exactS": "peer_greedy_network_cd",  # cd_exactS: CD + exact search over S on the top-5 slate (HEROSIM_PG_CD_EXACT=1, expansion as the over-cap fallback); the hand bar
    "cd_exactS_a0": "peer_greedy_network_cd",  # plan_rerank_v1 descriptive: cd_exactS with S's whole queue term scaled to 0 (HEROSIM_PG_DRAIN_ALPHA=0, S6's S_alpha)
    "cd_exactS_ib0_d05": "peer_greedy_network_cd",  # ib0_confirm_v1 diagnostic S_drain_weight: ib0 with the platform's own drain weighted 0.5 (HEROSIM_PG_DRAIN_ALPHA=0.5, BATCH_ALPHA=0)
    "cd_exactS_a025": "peer_greedy_network_cd",  # warm_surge_s0_v1 0c: batch alpha 0.25
    "cd_exactS_a05": "peer_greedy_network_cd",  # ib0_confirm_v1 descriptive: batch alpha 0.5, to show alpha = 0 is not a knife-edge
    "cdx_ib0": "peer_greedy_network_cd",  # ib0_confirm_v1 hand-seed control: the cd_expand flow (greedy first pass, passes, one expansion, passes) on the ib0 surrogate
    **{k: "peer_greedy_network_cd" for k in (  # sparse_obs_s0_v1: cd_exactS_ib0's search on a degraded observation (src/policy/peer_greedy_network/observation.py)
        "cd_obs_queue_naive", "cd_obs_queue_ec", "cd_obs_bundle_naive", "cd_obs_bundle_ec", "cd_obs_exec", "cd_obs_partners_naive", "cd_obs_partners_ec",
        "cd_obs_a0plus", "cd_obs_w1_naive", "cd_obs_w1_ec")},
    "cd_ib0": "peer_greedy_network_cd",  # decision_cost_s0_v1 hand frontier: plain CD greedy on the ib0 surrogate (HEROSIM_PG_BATCH_ALPHA=0)
    "cd_exactS_ib0_prune": "peer_greedy_network_cd",  # decision_cost_s0_v1 hand frontier: cd_exactS_ib0 with cost-ordered top-k pruning of over-cap batches (HEROSIM_PG_CD_EXACT_PRUNE=cost)
    "cd_exactS_ib0": "peer_greedy_network_cd",  # plan_rerank_v1 descriptive: cd_exactS with only the in-batch stacking term zeroed (HEROSIM_PG_BATCH_ALPHA=0)
    "cd_oracle_ib0": "peer_greedy_network_cd",  # ib0_confirm_v1 full-coverage oracle: the ib0 surrogate's plan, the exact label on covered batches, a replay-scored local search on over-cap ones
    "cd_oracle": "peer_greedy_network_cd",  # plan_rerank_v1 Phase 0: cd_exactS with the exact search scored by the real single-batch RTT (HEROSIM_PG_ORACLE=1); a measurement arm
    "cd_random_seed": "peer_greedy_network_cd_random_seed",  # r1_attribution_v1 CD<-random: random plan in the declared slate, then CD<-GNN's refine
    "decima": "decima_wfair_network",  # decima_rule_v1: alpha from the driver's HEROSIM_DECIMA_ALPHA (the tuned value)
    "random": "random_network",
    "drain": "drain_greedy_network",  # rule_baselines_v1: least-loaded (drain-time shortest queue, per arrival, no exchange term)
    "locality": "peer_greedy_network_batch",  # rule_baselines_v1: locality-first (exchange term x LOCALITY_SCALE)
    "offload": "offload_network",  # client_local_v1: never on the source client, shortest queue over reachable servers
    "localfirst": "local_first_network",  # client_local_v1: the source client when it hosts a replica, else shortest queue
    "cdext": "peer_greedy_network_cd",  # peak_controls_v1: CD + the label's externality at the label's rate
    "cdextr": "peer_greedy_network_cd",  # peak_controls_v1: CD + the externality at the rung's offered rate
}
# peak_controls_v1: HEROSIM_PG_EXT_RATE for the externality arms. The label's constant (drift_label, 0.46/s)
# is the study's x1 rate; cdextr multiplies it by the rung (1 / HEROSIM_POLICY_TIME_SCALE).
EXT_LABEL_RATE = 0.46
# rule_baselines_v1: the locality-first rule is the batched peer-greedy with its exchange term weighted 100x, so
# co-locating with placed partners dominates and queue drain only breaks ties
LOCALITY_SCALE = 100.0
RB1_KINDS = ("random", "drain", "locality", "decima", "batched", "selfpredict")
# client_local_v1: clients host replicas and may run their own calls (cells carry "client_local_v1": true)
CL1_KINDS = ("reactive", "cd", "locality", "selfpredict", "batched", "offload", "localfirst")
CL1_LEARNED = ("sb1load", "sb1mpoff", "lf1gnn", "lf1mlp")  # served zero-shot: trained on server-only candidates
EXT_KINDS = ("cdext", "cdextr")
# decima_rule_v1 tuning arms: Decima's tuned weighted fair at a fixed alpha (a0 = fair, a1 = naive weighted fair)
DECIMA_TUNE_ALPHAS = {"decima_am2": -2.0, "decima_am1": -1.0, "decima_am05": -0.5, "decima_a0": 0.0,
                      "decima_a05": 0.5, "decima_a1": 1.0}
RULE_POLICY.update({k: "decima_wfair_network" for k in DECIMA_TUNE_ALPHAS})
# Tuning never touches the study: non-study candidates from the fresh pool (disclosed: not screened for
# reactive admission).
DECIMA_TUNE_TOPOS = (9101, 9102, 9103, 9104)
# burst_ladder_v1: w0 at three arrival intensities; the window label carries the rung, the inputs live in
# <inputs>/ladder/x<rung> (cd_gap_v1_build_b.py, factor 1/intensity on timestamps and policy time constants)
LADDER_RUNGS = {"w0x10": "x10", "w0x15": "x15", "w0x20": "x20"}
# burst_ladder_v1 Amendment 1: K perturbed draws of a rung (burst_ladder_jitter.py), the same draw for every arm
JIT_DRAWS = (1, 2, 3, 4)
JIT_RUNGS = {f"w0{r}d{k}": f"{r}d{k}" for r in ("x10", "x15") for k in JIT_DRAWS}
LADDER_RUNGS.update(JIT_RUNGS)
# capacity_sweep_v1: x1.1 / x1.2 / x1.3 rungs with the same 4 draws, and the x1 draws served with replica keep_alive
# removed (window suffix "ka": the x1 draw's inputs, HEROSIM_KEEP_ALIVE=CAP_KEEP_ALIVE for every arm)
CAP_RUNGS = ("x11", "x12", "x13")
LADDER_RUNGS.update({f"w0{r}d{k}": f"{r}d{k}" for r in CAP_RUNGS for k in JIT_DRAWS})
KA_WINDOWS = {f"w0x10d{k}ka": f"x10d{k}" for k in JIT_DRAWS}
LADDER_RUNGS.update(KA_WINDOWS)
CAP_KEEP_ALIVE = "1e9"
CAP_ARMS = ("reactive", "cd")
LADDER_ARMS = ("reactive", "cd", "cd_inflight", "selfpredict")
LADDER_LEARNED = ("xs1load_selfref", "xs1load_cdapply")
SUFFIXES = ("_spread", "_slate", "_cdshadow", "_cdapply", "_cdxapply", "_cdxib0", "_cdxhib0", "_cdxexg", "_cdxpra", "_cdxprb", "_selfref", "_selfrefkw", "_se", "_nosplit", "_selfsearch")
# gnn_selfsearch: "_selfsearch" = "_nosplit" plus GNN_SELF_SEARCH=1 (the slate plan with the highest GNN plan score)
# accel_nosplit_v1: "_nosplit" serves a declared-slate arm with GNN_SLATE_NO_SPLIT=1 (an over-100k batch decoded whole)
# ra_twin_eng_cdxapply: the same refine seeded by the separately trained MP-OFF twin (GNN_DISABLE_MESSAGE_PASSING via RA_MP_OFF; checkpoint <R1A_STEM>-twin-eng-seed<N>)
# cdxapply: "_cdxapply" = the NO-SPLIT learned plan (GNN_SLATE_NO_SPLIT=1, so the refine sees whole groups), then the CD refine
# WITH exact-move expansion (GNN_CD_REFINE=apply + HEROSIM_PG_CD_EXPANSION=1): does a learned seed still help the strongest search?
# cd_exactS_gnn: "_cdxexg" = the cdxapply seed (no-split learned plan) into the CD refine with the EXACT search over S (HEROSIM_PG_CD_EXACT=1, expansion as the
# over-cap fallback), exact S-ties broken by the GNN's plan score (HEROSIM_PG_CD_EXACT_GNN=1)
# cdxprune: "_cdxpra" / "_cdxprb" = the cdxapply seed into the CD refine with the exact search over S, and in over-cap batches the slate cut to its top-k (largest k
# that fits 100k plans) and enumerated exactly (HEROSIM_PG_CD_EXACT_PRUNE=gnn by the model's logit / cost by the slate's standalone-cost order); no GNN tie-break
# replica_guard_v1: "_selfrefkw" = self-refine plus the GNN_REPLICA_KEEPWARM serving guard (registered parameters)
KEEPWARM_ENV = {"GNN_REPLICA_KEEPWARM": "1", "GNN_REPLICA_KEEPWARM_MAX_REPLICAS": "4", "GNN_REPLICA_KEEPWARM_MARGIN_S": "5"}
GATE_RULES = ("selfpredict", "cd", "batched", "reactive")
V4_KINDS = ("v4load", "v4twin")  # load_repr_v1: partial_state_v4, load columns on / zeroed
# backlog_corpus_v1: v4load's recipe and its MP-OFF twin on the synthetic-backlog corpus, always served
# with the in-flight capture fix; a "_se" suffix serves any other arm with it (Amendment 1: v4load_se)
# fc1load: fullctx_refine_v1; xs1load: exchange_seconds_v1; xs1mpoff: its MP-OFF twin (peak_controls_v1); rawgnn /
# rawmlp: raw_plan_v1; RAW_V2: raw_plan_v2 (rawStwin is the no-convolution twin). Same cache and split.
RAW_V2 = ("rawE", "rawS", "rawES", "rawStwin")
RAW_KINDS = ("rawgnn", "rawmlp") + RAW_V2
SB1_KINDS = ("sb1load", "sb1mpoff")  # small_batch_v1: xs1load / xs1mpoff recipes trained on live-sized batches
# local_features_v1: rawS / rawStwin / rawmlp plus the static per-candidate columns (plan_raw_local), small-batch corpus
LF1_KINDS = ("lf1gnn", "lf1twin", "lf1mlp")
# sum aggregation over the bipartite convs: sb1load's and lf1gnn's recipes with NEAR_RTT_MP_BIPARTITE_AGGR=sum
AGG_KINDS = ("sb1sum", "lf1sum")
# hetero_conv_v1: lf1gnn's recipe with per-relation / per-node-type bipartite weights (twin: lf1twin's existing runs)
HET_KINDS = ("lf1het",)
# small_batch_so_v1: sb1load / sb1mpoff's recipes retrained on the single-origin corpus
TP1_CONDS = {"replay": ("store_forward", "0"), "pipe": ("pipelined", "0"), "release": ("store_forward", "1"),
             "pipe_release": ("pipelined", "1")}
SO1_KINDS = ("so1load", "so1mpoff", "so1lfgnn", "so1lfmlp")  # so1lf*: lf1gnn / lf1mlp recipes (Amendment 1)
BC1_KINDS = ("bc1load", "bc1mpoff", "fc1load", "xs1load", "xs1mpoff") + RAW_KINDS + SB1_KINDS + LF1_KINDS + AGG_KINDS + HET_KINDS + SO1_KINDS
LOAD_KINDS = V4_KINDS + BC1_KINDS
# r1_attribution_v1: the seven learned arms, retrained on R1 + WF1 under partial_state_v5. The checkpoint each arm serves is the
# configuration chosen on validation topologies only, copied to <inputs>/models/r1-attribution-v1-<arm>-seed<N>.pt by the selection step.
# scale_160_v1 reuses the arms: R1A_STEM / R1A_SPLIT name its checkpoint prefix and split file, R1A_TOPOS its topologies.
RA_ARMS = ("gnn_eng", "twin_eng", "mlp_same", "gnn_raw", "twin_raw", "gnn_eng_physmp", "set_transformer")
RA_KINDS = tuple(f"ra_{a}" for a in RA_ARMS)
RA_MP_OFF = ("ra_twin_eng", "ra_twin_raw")  # GNN_DISABLE_MESSAGE_PASSING=1; the graph-free arms have no such switch
# phase r1a (the r1_attribution_v1 live gate): classical arms run once (seed 0); learned arms and the seeded-CD arms (CD refine applied
# to the learned plan, "_cdapply") run at the checkpoint seeds in R1A_SEEDS. CD<-random has no implementation in the tree.
R1A_CLASSICAL = ("cd", "cd_declared", "locality", "batched", "selfpredict", "reactive")
R1A_SEEDED_CD = ("ra_gnn_eng_cdapply", "ra_twin_eng_cdapply", "cd_random_seed")
R1A_RANDOM = ("random",)  # plain random_network scheduler, descriptive and outside the families; run at the seeds in R1A_SEEDS
EXACT_ALPHA = {"cd_exactS_a0": {"HEROSIM_PG_DRAIN_ALPHA": "0"}, "cd_exactS_ib0": {"HEROSIM_PG_BATCH_ALPHA": "0"}, "cd_exactS_a05": {"HEROSIM_PG_BATCH_ALPHA": "0.5"}, "cd_exactS_a025": {"HEROSIM_PG_BATCH_ALPHA": "0.25"},
               "cd_exactS_ib0_d05": {"HEROSIM_PG_DRAIN_ALPHA": "0.5", "HEROSIM_PG_BATCH_ALPHA": "0"},
               "cd_exactS_ib0_prune": {"HEROSIM_PG_BATCH_ALPHA": "0"}}  # plan_rerank_v1 S_alpha arms; warm_surge/drain-weight arms; decision_cost_s0_v1 prune arm
_BUNDLE = {"HEROSIM_OBS_QUEUE": "count", "HEROSIM_OBS_EXEC_SIGMA": "0.3", "HEROSIM_OBS_PARTNERS": "arrival"}
OBS_ARMS = {  # sparse_obs_s0_v1 degradations (the estimator is the naive/ec arm); every one searches exactly as cd_exactS_ib0
    "cd_obs_queue_naive": {"HEROSIM_OBS_QUEUE": "count", "HEROSIM_OBS_ESTIMATOR": "naive"},
    "cd_obs_queue_ec": {"HEROSIM_OBS_QUEUE": "count", "HEROSIM_OBS_ESTIMATOR": "ec"},
    "cd_obs_bundle_naive": {**_BUNDLE, "HEROSIM_OBS_ESTIMATOR": "naive"},
    "cd_obs_bundle_ec": {**_BUNDLE, "HEROSIM_OBS_ESTIMATOR": "ec"},
    "cd_obs_exec": {"HEROSIM_OBS_EXEC_SIGMA": "0.3", "HEROSIM_OBS_ESTIMATOR": "naive"},
    "cd_obs_partners_naive": {"HEROSIM_OBS_PARTNERS": "arrival", "HEROSIM_OBS_ESTIMATOR": "naive"},
    "cd_obs_partners_ec": {"HEROSIM_OBS_PARTNERS": "arrival", "HEROSIM_OBS_ESTIMATOR": "ec"},
    "cd_obs_a0plus": {"HEROSIM_PG_COLD_TAIL": "1"},  # stage 2: A0 with the cold term priced against the type the task follows
    "cd_obs_w1_naive": {"HEROSIM_PG_COLD_TAIL": "1", "HEROSIM_OBS_WARM": "seconds", "HEROSIM_OBS_ESTIMATOR": "naive"},
    "cd_obs_w1_ec": {"HEROSIM_PG_COLD_TAIL": "1", "HEROSIM_OBS_WARM": "seconds", "HEROSIM_OBS_ESTIMATOR": "ec"},
}
OBS_ENV = ("HEROSIM_OBS_QUEUE", "HEROSIM_OBS_EXEC_SIGMA", "HEROSIM_OBS_INFLIGHT", "HEROSIM_OBS_PARTNERS", "HEROSIM_OBS_WARM", "HEROSIM_PG_COLD_TAIL", "HEROSIM_OBS_ESTIMATOR", "HEROSIM_OBS_SEED")
EXACT_ALPHA.update({k: {"HEROSIM_PG_BATCH_ALPHA": "0"} for k in OBS_ARMS})  # sparse_obs_s0_v1
ORACLE_ENV = ("GNN_CD_SEED", "HEROSIM_PG_DRAIN_ALPHA", "HEROSIM_PG_BATCH_ALPHA", "HEROSIM_PG_ORACLE", "HEROSIM_PG_ORACLE_MAX_PLANS", "HEROSIM_PG_ORACLE_SEARCH", "HEROSIM_PG_ORACLE_SEARCH_BUDGET", "HEROSIM_ORACLE_CELL_CONFIG", "HEROSIM_ORACLE_SIM_INPUT", "HEROSIM_ORACLE_WORKERS", "HEROSIM_SNAPSHOT_FIDELITY")  # scrubbed per cell; cd_oracle sets them


def ib0_alpha(kind: str) -> dict:
    """ib0_confirm_v1: the batch alpha the expansion arms on the ib0 surrogate must serve (none for every other arm)."""
    return {"HEROSIM_PG_BATCH_ALPHA": "0"} if kind in ("cdx_ib0", "cd_oracle_ib0", "cd_ib0") or kind.endswith(("_cdxib0", "_cdxhib0")) else {}


def check_alias(kind: str) -> str:
    """ib0_confirm_v1: the registered arm whose instrument checks an ib0-surrogate expansion arm shares (the alpha and seed are checked on top)."""
    if kind == "cdx_ib0":
        return "cd_expand"
    if kind == "cd_oracle_ib0":
        return "cd_oracle"
    if kind == "cd_ib0":
        return "cd"
    for suffix in ("_cdxib0", "_cdxhib0"):
        if kind.endswith(suffix):
            return kind[:-len(suffix)] + "_cdxapply"
    return kind
# decision_cost_s0_v1 (Amendment 3): a cell kind "<arm>_k<centi>" serves <arm> with the scheduler's decision time charged as simulated time, arm-calibrated:
# t_batch = k * w[arm][rung] ms * plans_evaluated, so HEROSIM_DECISION_COST_MS = k * w (src/placement/decision_cost.py). w is the measured wall ms per plan,
# fixed in scripts_cosim/decision_cost_w.json before any cell counts. The file name keeps the suffix; every arm check runs on the base arm.
DC_K = {"k025": 0.25, "k050": 0.5, "k100": 1.0, "k200": 2.0}
DC_W_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "decision_cost_w.json")


# uncut_groups_s0_v1 K1 (identity (m)): R1A_SNAPSHOT_PLAN names a JSON file {"<topology>__<window>": [trigger task ids]} sealed
# before any cell runs. Every cell of that topology and window captures exactly those batches (fidelity snapshots, for the
# replay); the summary reports how many it wanted and wrote. Unset: nothing is captured, as before.
SNAP_ENV = ("LIVE_AUDIT_TRIGGER_IDS", "LIVE_AUDIT_MAX_SNAPSHOTS", "LIVE_AUDIT_STRIDE", "LIVE_AUDIT_MIN_BATCH_SIZE", "LIVE_AUDIT_MIN_CANDIDATES")


def sealed_triggers(topo: int, window: str) -> Optional[List[int]]:
    path = os.environ.get("R1A_SNAPSHOT_PLAN")
    if not path:
        return None
    ids = json.load(open(path)).get(f"{topo}__{window}")
    if ids is not None and not (isinstance(ids, list) and all(isinstance(i, int) for i in ids)):
        raise SystemExit(f"FAIL LOUD: R1A_SNAPSHOT_PLAN {path} entry {topo}__{window} must be a list of task ids")
    return sorted(ids) if ids else None


def split_dc(kind: str) -> Tuple[str, Optional[float]]:
    for tag, k in DC_K.items():
        if kind.endswith("_" + tag):
            return kind[:-len(tag) - 1], k
    return kind, None


def dc_cost_ms(kind: str, k: Optional[float], rung: str) -> Optional[str]:
    """The per-plan cost (ms, as the env string) cell `kind` at rung serves under k: k x the arm's calibrated w. None when k is not set."""
    if k is None:
        return None
    w = json.load(open(DC_W_FILE))["w_ms_per_plan"]
    if kind not in w or rung not in w[kind]:
        raise SystemExit(f"FAIL LOUD: no calibrated w (ms per plan) for {kind!r} at rung {rung!r} in {DC_W_FILE}")
    return repr(k * float(w[kind][rung]))


R1A_DIAG = (*OBS_ARMS, "cd_ib0", "cd_exactS_ib0_prune", "cd_pull", "cd_ledger", "cd_expand", "cd_exactS", "cd_oracle", "cd_exactS_a0", "cd_exactS_ib0", "cd_exactS_a05", "cd_exactS_a025", "cd_exactS_ib0_d05", "cdx_ib0", "cd_oracle_ib0")  # diagnostic classical arms, seed 0, never in the default grid (name them in R1A_ARMS)
R1A_NOSPLIT = ("ra_gnn_eng_nosplit", "ra_gnn_eng_physmp_nosplit", "ra_twin_eng_nosplit")  # accel_nosplit_v1 primary arms (+ slate_fix_v1's twin), checkpoint seeds; name them in R1A_ARMS
R1A_SELFSEARCH = ("ra_gnn_eng_selfsearch", "ra_gnn_eng_physmp_selfsearch", "ra_twin_eng_selfsearch")  # gnn_selfsearch arms, checkpoint seeds; name them in R1A_ARMS
R1A_CDX = ("ra_gnn_eng_cdxapply", "ra_gnn_eng_physmp_cdxapply", "ra_gnn_eng_cdxexg", "ra_gnn_eng_cdxpra", "ra_gnn_eng_cdxprb", "ra_twin_eng_cdxpra", "ra_gnn_eng_cdxib0", "ra_gnn_eng_cdxhib0", "ra_twin_eng_cdxib0", "ra_twin_eng_cdxapply")  # cdxapply diagnostics, checkpoint seeds; name them in R1A_ARMS
R1A_ARMS = R1A_CLASSICAL + RA_KINDS + R1A_SEEDED_CD + R1A_RANDOM + R1A_DIAG + R1A_NOSPLIT + R1A_SELFSEARCH + R1A_CDX
# slate_fix_v1: an r1a checkpoint is served under the candidate-slate rule its sidecar records. R1A_SLATE names the rule a gate
# submission expects (unset: declared_pruning_v1, every checkpoint before slate_fix_v1); a sidecar with another rule fails the cell.
def r1a_slate() -> str:
    from src.placement.declared_slate import RULE, RULES
    want = os.environ.get("R1A_SLATE", "").strip() or RULE
    if want not in RULES:
        raise SystemExit(f"FAIL LOUD: R1A_SLATE={want!r}; expected one of {RULES}")
    return want


def ra_serving_slate(side: str) -> str:
    rule = json.load(open(side)).get("candidate_slate")
    if rule != r1a_slate():
        raise SystemExit(f"FAIL LOUD: {side} records candidate_slate={rule!r}; this gate serves R1A_SLATE={r1a_slate()!r}")
    return rule


R1A_ON = False  # set by main() for phase r1a: progress watchdog and the 5 % pause line
R1A_LIMIT_S = 8100
_LAST_ARRIVAL: Dict[str, float] = {}
LEARNED_KINDS = ("gnnedge0", "mpoff", "cdimit") + LOAD_KINDS + RA_KINDS
SERVICE_END = "service_end_v1"
# grounded_workload_v1: study windows whose group sizes, sibling offsets and arrival process come from Alibaba's
# 2021 call graphs (grounded_workload_v1_mint.py), at the study's x1 rate; files live in <inputs>/grounded/wl
GROUNDED_WINDOWS = {f"g{i}": f"grounded_g{i}_n50000" for i in range(4)}
GROUNDED_RULES = ("random", "reactive", "batched", "selfpredict", "cd", "decima")
GROUNDED_LEARNED = ("xs1load_selfref", "xs1load", "gnnedge0", "mpoff")
# peak_load_v1: 1.5x the mean rate (peak hour; Azure's hourly peak/mean is ~1.6, Shahrad et al. ATC'20 Fig. 4),
# built with the ladder protocol (timestamps and policy time constants x 1/1.5) into <inputs>/grounded_x15
GROUNDED_X15 = {f"g{i}x15": f"grounded_g{i}_n50000" for i in range(4)}
X15_FILL_RULES = ("random", "batched", "decima")
# peak_load_v2: the same grounded windows at x2 / x3 / x5 (load sweep to saturation, as DeathStarBench, Gan et al.
# ASPLOS'19), built the same way into <inputs>/grounded_x20 etc.
GROUNDED_RUNGS = {"x20": 0.5, "x30": 1 / 3, "x50": 0.2}
GROUNDED_LADDER = {f"g{i}{r}": (f"grounded_g{i}_n50000", f"grounded_{r}") for r in GROUNDED_RUNGS for i in range(4)}
# workload_fix_v1: rungs at arbitrary multipliers, built by workload_fix_v1_build.py into <inputs>/wf1_<tag>;
# WF1_RUNGS="tagA,tagB" names them and the window label is g<i><tag>. Empty unless the env is set.
WF1_TAGS = [t for t in os.environ.get("WF1_RUNGS", "").split(",") if t]
WF1_LADDER = {f"g{i}{t}": (f"grounded_g{i}_n50000", f"wf1_{t}") for t in WF1_TAGS for i in range(4)}
# scale_sweep_v1: the ladder at x3 / x5 with ~2x / ~4x tasks per decision batch (m2 / m4: grounded_workload_v1_mint.py
# --merge-k, cell batch_size 16) and on 12-server cells (s12), each built into <inputs>/grounded_<rung><cond>
SCALE_CONDS = ("m2", "m4", "s12")
SCALE_LADDER = {f"g{i}{r}{c}": (f"grounded_g{i}_n50000", f"grounded_{r}{c}")
                for c in SCALE_CONDS for r in ("x30", "x50") for i in range(4)}
# peak_controls_v1: the two controls on the same ladder cells -- CD with the label's externality (two rates) and the
# MP-OFF twin of the GNN trained on the same corpus and recipe, served with the same self-refine
PEAKCTL_WITNESS = ((9119, "g0x30"), (9420, "g0x30"))
N_TASKS = 50000
PY = shlex.split(os.environ.get("HEROSIM_PY", "pipenv run python3"))
# SLURM compute nodes have no systemd-run: the job's own memory allocation caps the runs instead.
NO_SCOPE = False


def queue_drift(task_results: Optional[List[dict]]) -> Optional[Dict[str, object]]:
    """capacity_sweep_v1: mean queue time per quarter of the tasks in dispatch order, and last / first quarter.
    Recorded so stability can be read after the raw per-task file is deleted."""
    if not task_results:
        return None
    rows = sorted((float(r["dispatchedTime"]), float(r["queueTime"])) for r in task_results
                  if r.get("taskId") is None or int(r["taskId"]) >= 0)
    n = len(rows)
    if n < 4:
        return None
    q = [rows[i * n // 4:(i + 1) * n // 4] for i in range(4)]
    means = [sum(x for _, x in part) / len(part) if part else 0.0 for part in q]
    return {"quarter_mean_queue": means,
            "last_over_first": (means[3] / means[0]) if means[0] > 0 else None}


def latency_percentiles(task_results: Optional[List[dict]]) -> Optional[Dict[str, object]]:
    """physics_audit_v1: P50 / P95 / P99 / max of the per-task latency (done - dispatched, the quantity whose mean is
    averageElapsedTime), nearest-rank. Recorded so the tail can be read after the raw per-task file is deleted."""
    if not task_results:
        return None
    lat = sorted(float(r["doneTime"]) - float(r["dispatchedTime"]) for r in task_results
                 if r.get("taskId") is None or int(r["taskId"]) >= 0)
    n = len(lat)
    if not n:
        return None

    def rank(q: float) -> float:
        return lat[min(n - 1, max(0, math.ceil(q * n) - 1))]

    return {"n": n, "p50": rank(0.50), "p95": rank(0.95), "p99": rank(0.99), "max": lat[-1]}


def placement_wait(task_results: Optional[List[dict]]) -> Optional[Dict[str, object]]:
    """load_recalibration_v1: arrival-to-placement time per task (scheduledTime - dispatchedTime, the quantity whose mean is
    averageWaitTime): mean, p95 (nearest-rank) and max. A queue share cannot see a wait that happens before execution."""
    if not task_results:
        return None
    w = sorted(float(r["scheduledTime"]) - float(r["dispatchedTime"]) for r in task_results
               if r.get("taskId") is None or int(r["taskId"]) >= 0)
    n = len(w)
    if not n:
        return None
    return {"n": n, "mean": sum(w) / n, "p95": w[min(n - 1, max(0, math.ceil(0.95 * n) - 1))], "max": w[-1]}


def backlog_profile(task_results: Optional[List[dict]], with_lock_wait: bool = False) -> Optional[Dict[str, object]]:
    """load_recalibration_v1 backlog guard. Per task, backlog = (scheduledTime - dispatchedTime) + queueTime, so a wait
    before placement counts; with ``with_lock_wait`` (amendment 2026-10-09) the wait for the replica's compute lock,
    computeStartTime - ioEndTime, is added. Quarter means in arrival (dispatchedTime) order, last over the mean of the
    middle two, and the tasks in the system (dispatched, not done) at 1/2 and 3/4 of the last arrival time, 3/4 over 1/2."""
    if not task_results:
        return None
    def lock(r):
        if not with_lock_wait or r.get("ioEndTime") is None or r.get("computeStartTime") is None:
            return 0.0
        return float(r["computeStartTime"]) - float(r["ioEndTime"])

    rows = sorted((float(r["dispatchedTime"]), float(r["scheduledTime"]) - float(r["dispatchedTime"]) + float(r["queueTime"]) + lock(r),
                   float(r["doneTime"])) for r in task_results if r.get("taskId") is None or int(r["taskId"]) >= 0)
    n = len(rows)
    if n < 4:
        return None
    q = [rows[i * n // 4:(i + 1) * n // 4] for i in range(4)]
    means = [sum(x[1] for x in part) / len(part) for part in q]
    mid = (means[1] + means[2]) / 2
    arrivals = [x[0] for x in rows]
    done = sorted(x[2] for x in rows)
    last = arrivals[-1]
    in_system = {}
    for tag, frac in (("half", 0.5), ("three_quarter", 0.75)):
        t = frac * last
        in_system[tag] = bisect.bisect_right(arrivals, t) - bisect.bisect_right(done, t)
    return {"quarter_mean_backlog": means, "last_over_mid": means[3] / mid if mid > 0 else (math.inf if means[3] > 0 else 1.0),
            "in_system_at": in_system,
            "in_system_ratio": in_system["three_quarter"] / in_system["half"] if in_system["half"] > 0
            else (math.inf if in_system["three_quarter"] > 0 else 1.0)}


def lock_wait_profile(task_results: Optional[List[dict]]) -> Optional[Dict[str, object]]:
    """load_recalibration_v1 (amendment 2026-10-09): the wait for the replica's compute lock, compute_start - io_end per task.
    `started` is stamped before it, so the platform queue's FIFO backlog on a saturated replica is filed under compute
    time and queue_share misses it. effective_queue_share = (queue time + lock wait) / elapsed over the same tasks;
    queue_share is kept alongside. A task without the stamps (never served) counts as no lock wait and is reported in
    n_unstamped. None when the result has no task results."""
    if not task_results:
        return None
    rows = [r for r in task_results if r.get("taskId") is None or int(r["taskId"]) >= 0]
    if not rows:
        return None
    w = sorted((float(r["computeStartTime"]) - float(r["ioEndTime"]))
               if r.get("ioEndTime") is not None and r.get("computeStartTime") is not None else 0.0 for r in rows)
    n = len(w)
    elapsed = sum(float(r["doneTime"]) - float(r["dispatchedTime"]) for r in rows)
    queue = sum(float(r["queueTime"]) for r in rows)
    return {"n": n, "n_unstamped": sum(1 for r in rows if r.get("computeStartTime") is None),
            "mean": sum(w) / n, "p95": w[min(n - 1, max(0, math.ceil(0.95 * n) - 1))], "max": w[-1],
            "effective_queue_share": (queue + sum(w)) / elapsed if elapsed > 0 else None,
            "queue_share_same_tasks": queue / elapsed if elapsed > 0 else None}


STEADY_T_S = 360.0  # the corpus MIN_TIME


def _workload_rows(task_results: Optional[List[dict]]) -> List[tuple]:
    return sorted((float(r["dispatchedTime"]), float(r["doneTime"])) for r in task_results or ()
                  if r.get("taskId") is None or int(r["taskId"]) >= 0)


def steady_state_latency(task_results: Optional[List[dict]]) -> Optional[Dict[str, object]]:
    """The live path autoscales from zero replicas with no warm-up discard, so averageElapsedTime carries a whole-system cold
    start (expert's ruling, 2026-10-12). Mean latency (done - dispatched, the averageElapsedTime quantity) over the tasks
    dispatched at or after T, for T = 360 s and T = the dispatch time of the 5th-percentile task (nearest rank), and the mean
    of each tenth of the tasks in dispatch order."""
    rows = _workload_rows(task_results)
    n = len(rows)
    if n < 10:
        return None

    def after(t_s: float) -> Dict[str, object]:
        lat = [done - d for d, done in rows if d >= t_s]
        return {"T_s": t_s, "n": len(lat), "mean": sum(lat) / len(lat) if lat else None}

    t5 = rows[min(n - 1, max(0, math.ceil(0.05 * n) - 1))][0]
    deciles = [rows[i * n // 10:(i + 1) * n // 10] for i in range(10)]
    return {"after_360s": after(STEADY_T_S), "after_p5_dispatch": after(t5),
            "decile_mean": [sum(done - d for d, done in part) / len(part) for part in deciles]}


def write_task_times(task_results: Optional[List[dict]], path: str) -> Optional[int]:
    """Per-task (dispatchedTime, doneTime) as float32, in dispatch order, so any warm-up cut can be read after the raw file is
    deleted (~400 KB per 50k-task cell). Written atomically; returns the row count."""
    import numpy as np

    rows = _workload_rows(task_results)
    if not rows:
        return None
    with open(path + ".partial", "wb") as fh:
        np.save(fh, np.asarray(rows, dtype=np.float32))
    os.replace(path + ".partial", path)
    return len(rows)


def arrival_end(workload_path: str, end_time: Optional[float]) -> Optional[Dict[str, object]]:
    """load_recalibration_v1: the run's end time against the last arrival in its workload file."""
    last = max(float(e["timestamp"]) for e in json.load(open(workload_path))["events"])
    if end_time is None:
        return None
    return {"last_arrival_s": last, "end_time_s": float(end_time), "end_over_last_arrival": float(end_time) / last if last > 0 else None}


def replica_count_series(system_events: Optional[List[dict]], end_time: Optional[float],
                         points: int = 120) -> Optional[Dict[str, object]]:
    """physics_audit_v1: live replicas per function and in total on a uniform grid over the run, from the autoscaler's
    one-second systemEvents, plus the time-mean and peak of the total. None when the run has no such record."""
    if not system_events or not end_time or end_time <= 0:
        return None
    by_fn: Dict[str, List[tuple]] = {}
    for e in system_events:
        by_fn.setdefault(str(e["name"]), []).append((float(e["timestamp"]), int(e["count"])))
    for v in by_fn.values():
        v.sort(key=lambda x: x[0])
    grid = [float(end_time) * i / points for i in range(points + 1)]
    series: Dict[str, List[int]] = {}
    for fn, v in by_fn.items():
        times = [x[0] for x in v]
        series[fn] = [v[i - 1][1] if (i := bisect.bisect_right(times, g)) > 0 else 0 for g in grid]
    total = [sum(series[fn][i] for fn in series) for i in range(len(grid))]
    # time-mean on the one-second record itself, not on the coarse grid
    last = {fn: 0 for fn in by_fn}
    area, prev_t, cur = 0.0, 0.0, 0
    for t, fn, c in sorted((t, fn, c) for fn, v in by_fn.items() for t, c in v):
        area += cur * (t - prev_t)
        cur += c - last[fn]
        last[fn] = c
        prev_t = t
    area += cur * max(0.0, float(end_time) - prev_t)
    return {"t": grid, "total": total, "by_function": series, "peak": max(total), "time_mean": area / float(end_time)}


def task(topo: int, window: str, kind: str, seed: int = 0) -> Dict[str, object]:
    return {"topo": topo, "window": window, "kind": kind, "seed": seed}


def r1a_tasks(selection: Optional[dict]) -> List[Dict[str, object]]:
    """r1_attribution_v1 live gate cells: R1A_TOPOS (default the selection's 19 test topologies) x the WF1_RUNGS rungs x R1A_WINDOWS
    x R1A_ARMS. R1A_SHARD="i/n" keeps every n-th cell (array shards; the list is ordered rung, topology, window, arm, seed)."""
    # knative_diag: R1A_KNATIVE_DIAG=1 lets the replica-release physics be switched off for a diagnostic cell (reactive only, not a gate)
    need = {"HEROSIM_TRANSFER_MODEL": "pipelined", "HEROSIM_REPLICA_RELEASE": "0" if os.environ.get("R1A_KNATIVE_DIAG") == "1" else "1",
            "HEROSIM_SCALEOUT": "kpa", "GATE_FIXED_POLICY_TIME_SCALE": "1.0", "HEROSIM_SHARED_AUTOSCALER": "0"}
    if os.environ.get("R1A_KNATIVE_DIAG") == "1" and os.environ.get("HEROSIM_REPLICA_RELEASE") not in ("0", "1"):
        raise SystemExit("FAIL LOUD: R1A_KNATIVE_DIAG=1 needs HEROSIM_REPLICA_RELEASE set to 0 or 1")
    if os.environ.get("R1A_KNATIVE_DIAG") == "1":
        need["HEROSIM_REPLICA_RELEASE"] = os.environ["HEROSIM_REPLICA_RELEASE"]
    bad = {k: os.environ.get(k) for k, v in need.items() if os.environ.get(k) != v}
    if bad or not WF1_LADDER:
        raise SystemExit(f"FAIL LOUD: r1a runs on R1.1 (want {need}); got {bad}; WF1_RUNGS={WF1_TAGS}")
    topos = [int(x) for x in os.environ.get("R1A_TOPOS", "").split(",") if x] or list(selection["topologies"])
    wins = [w for w in os.environ.get("R1A_WINDOWS", "g0,g1,g2,g3").split(",") if w]
    arms = [k for k in os.environ.get("R1A_ARMS", ",".join(k for k in R1A_ARMS if k not in R1A_RANDOM + R1A_DIAG + R1A_NOSPLIT + R1A_SELFSEARCH + R1A_CDX)).split(",") if k]
    seeds = [int(x) for x in os.environ.get("R1A_SEEDS", "1,2").split(",") if x]
    unknown = [k for k in arms if split_dc(k)[0] not in R1A_ARMS]
    if unknown:
        raise SystemExit(f"FAIL LOUD: R1A_ARMS {unknown}; arms are {R1A_ARMS}")
    single = R1A_CLASSICAL + R1A_DIAG
    cells = [task(t, f"{w}{tag}", k, 0 if split_dc(k)[0] in single else sd)
             for tag in WF1_TAGS for t in topos for w in wins for k in arms for sd in ([0] if split_dc(k)[0] in single else seeds)]
    shard = os.environ.get("R1A_SHARD", "")
    if shard:
        i, n = (int(x) for x in shard.split("/"))
        if not 0 <= i < n:
            raise SystemExit(f"FAIL LOUD: R1A_SHARD={shard!r}")
        cells = cells[i::n]
    return cells


def tasks_for(phase: str, selection: Optional[dict]) -> List[Dict[str, object]]:
    if phase in ("screen", "rp2screen"):
        pool = RP2C_POOL if phase == "rp2screen" else CANDIDATES
        return [task(t, w, k) for k in ("reactive", "batched") for t in pool for w in WINDOWS]
    if phase == "parity":
        return [task(9101, "w1", k) for k in ("selfpredict", "cd")] + \
               [task(9101, "w1", k, 1) for k in ("gnnedge0", "mpoff")]
    if phase == "decimatune":
        return [task(t, w, k) for k in DECIMA_TUNE_ALPHAS for t in DECIMA_TUNE_TOPOS for w in WINDOWS] + \
               [task(t, w, "cd") for t in DECIMA_TUNE_TOPOS for w in WINDOWS]
    if phase in ("wf1", "wf1cal"):
        # workload_fix_v1: classical arms only. wf1cal: CD alone on the calibration topologies (WF1_TOPOS) at the
        # windows in WF1_CAL_WINDOWS; wf1: the five rules on the selection's topologies at every WF1_RUNGS rung.
        if not WF1_LADDER:
            raise SystemExit("FAIL LOUD: phase %s needs WF1_RUNGS" % phase)
        if os.environ.get("HEROSIM_TRANSFER_MODEL") != "pipelined" or os.environ.get("HEROSIM_REPLICA_RELEASE") != "1" \
                or os.environ.get("HEROSIM_SCALEOUT") != "kpa" or os.environ.get("GATE_FIXED_POLICY_TIME_SCALE") != "1.0":
            raise SystemExit("FAIL LOUD: workload_fix_v1 runs on R1 (pipelined, release, kpa, time scale 1.0)")
        if phase == "wf1cal":
            cal = [int(x) for x in os.environ.get("WF1_TOPOS", "").split(",") if x]
            wins = [w for w in os.environ.get("WF1_CAL_WINDOWS", "g0,g1").split(",") if w]
            if not cal:
                raise SystemExit("FAIL LOUD: wf1cal needs WF1_TOPOS")
            kinds = [k for k in os.environ.get("WF1_CAL_KINDS", "cd").split(",") if k]
            if any(k not in ("cd", "reactive", "cd_declared") + RA_KINDS + R1A_SEEDED_CD for k in kinds):  # ra_*: the r1_attribution_v1 serve smoke
                raise SystemExit(f"FAIL LOUD: WF1_CAL_KINDS={kinds!r}")
            return [task(t, f"{w}{tag}", k) for tag in WF1_TAGS for t in cal for w in wins for k in kinds]
        rules = ("reactive", "selfpredict", "locality", "batched", "cd")
        only = [k for k in os.environ.get("WF1_ARMS", "").split(",") if k]  # amendment WB reruns only the batching arms
        if any(k not in rules for k in only):
            raise SystemExit(f"FAIL LOUD: WF1_ARMS={only!r}; arms are {rules}")
        # opt-in window subset for smoke runs (windows "g0,g1"); unset = the full registered grid
        only_w = [x for x in os.environ.get("WF1_ARRIVAL_WINDOWS", "").split(",") if x]
        ladder = [w for w in WF1_LADDER if not only_w or w[:2] in only_w]
        if not ladder:
            raise SystemExit(f"FAIL LOUD: WF1_ARRIVAL_WINDOWS={only_w!r} selects no window")
        return [task(t, w, k) for k in rules if not only or k in only for t in selection["topologies"] for w in ladder]
    topos = selection["topologies"]
    if phase == "r1a":
        return r1a_tasks(selection)
    if phase == "d1":
        return [task(t, w, "cd_blind") for t in topos for w in WINDOWS]
    if phase == "a":
        return [task(t, w, "cdimit", s) for s in (1, 2, 3, 4) for t in topos for w in WINDOWS]
    if phase == "v4":
        return [task(t, w, k, s) for k in V4_KINDS for s in (1, 2, 3, 4) for t in topos for w in WINDOWS]
    if phase == "x15fill":
        # peak_load_v1 (a): the rules burst_ladder_v1 Amendment 1 did not run at x1.5, on the same four draws; the
        # witness reruns CD and xs1load_selfref seed 1 on two cells, which must equal the ladderjit runs to the digit
        x15 = [f"w0x15d{k}" for k in JIT_DRAWS]
        return [task(t, "w0x15d1", "cd") for t in (9119, 9420)] + \
               [task(t, "w0x15d1", "xs1load_selfref", 1) for t in (9119, 9420)] + \
               [task(t, w, k) for k in X15_FILL_RULES for t in topos for w in x15]
    if phase == "groundedx15":
        # peak_load_v1 (b): every arm on the Alibaba-grounded windows at x1.5
        gw = tuple(GROUNDED_X15)
        return [task(t, w, k) for k in GROUNDED_RULES for t in topos for w in gw] + \
               [task(t, w, k, s) for k in GROUNDED_LEARNED for s in (1, 2, 3, 4) for t in topos for w in gw]
    if phase == "peakctl":
        gw = tuple(GROUNDED_LADDER)
        # witness: CD on two x3 cells must equal the groundedladder runs to the digit (the externality code is inert when unset)
        return [task(t, w, "cd") for t, w in PEAKCTL_WITNESS] + \
               [task(t, w, k) for k in EXT_KINDS for t in topos for w in gw]
    if phase == "w0mlp":
        # peak_controls_v1 Amendment 1: the MP-OFF twin on the w0 draws where the GNN has 4 seeds (x1.5 ladderjit,
        # x1.1 capacity + x11confirm); witness: GNN seed 1 on two x1.5 cells must equal ladderjit to the digit
        w0 = tuple(f"w0{r}d{k}" for r in ("x15", "x11") for k in JIT_DRAWS)
        return [task(t, "w0x15d1", "xs1load_selfref", 1) for t in (9119, 9420)] + \
               [task(t, w, "xs1mpoff_selfref", s) for s in (1, 2, 3, 4) for t in topos for w in w0]
    if phase in ("rawplan", "rawgnn", "rawmlp"):
        # raw_plan_v1: rawgnn / rawmlp run one arm each, so whichever finishes training first is not held back
        gw = tuple(GROUNDED_LADDER)
        kinds = ("rawgnn_selfref", "rawmlp_selfref") if phase == "rawplan" else (f"{phase}_selfref",)
        return [task(t, w, k, s) for k in kinds for s in (1, 2, 3, 4) for t in topos for w in gw]
    if phase == "rp2conf":
        # raw_plan_v2 Phase C on the unseen topologies: the dev-selected arm (RP2_SELECTED), its twin, the plain MLP,
        # and the rules. The arm is named by the operator from the dev read; a typo fails here, not in a run.
        sel = os.environ.get("RP2_SELECTED", "")
        if sel not in ("rawE", "rawS", "rawES"):
            raise SystemExit(f"FAIL LOUD: RP2_SELECTED={sel!r}; must be the dev read's selected arm (rawE|rawS|rawES)")
        twin = "rawmlp" if sel == "rawE" else "rawStwin"
        gw = tuple(GROUNDED_LADDER)
        learned = tuple(dict.fromkeys((sel, twin, "rawmlp")))
        return [task(t, w, k) for k in ("cd", "cdextr", "reactive") for t in topos for w in gw] + \
               [task(t, w, f"{k}_selfref", s) for k in learned for s in (1, 2, 3, 4) for t in topos for w in gw]
    if phase == "sb1dev":
        # small_batch_v1 B: the xs1load / xs1mpoff recipes retrained on live-sized batches, on the development cells
        gw = tuple(GROUNDED_LADDER)
        return [task(t, w, f"{k}_selfref", s) for k in SB1_KINDS for s in (1, 2, 3, 4) for t in topos for w in gw]
    if phase == "sbconf":
        # small_batch_confirm_v1 on the 19 unseen admitted topologies: sb1load and its MP-OFF twin sb1mpoff at 8 seeds,
        # the rules, and xs1load at its 4 seeds (descriptive). SBCONF_KINDS picks a subset so the parts run on separate
        # nodes; every part reads the same selection and inputs.
        gw = tuple(GROUNDED_LADDER)
        parts = {"sb1load": [("sb1load_selfref", s) for s in range(1, 9)],
                 "sb1mpoff": [("sb1mpoff_selfref", s) for s in range(1, 9)],
                 "rules": [(k, 0) for k in ("cd", "cdextr", "reactive")] + [("xs1load_selfref", s) for s in (1, 2, 3, 4)]}
        want = os.environ.get("SBCONF_KINDS", "sb1load,sb1mpoff,rules").split(",")
        if not want or any(w not in parts for w in want):
            raise SystemExit(f"FAIL LOUD: SBCONF_KINDS={want!r}; parts are {sorted(parts)}")
        return [task(t, w, k, s) for p in want for k, s in parts[p] for t in topos for w in gw]
    if phase == "agg1":
        # sum vs mean bipartite aggregation on the 19 topologies; the mean arms are sb1load / lf1gnn's existing runs
        kinds = os.environ.get("AGG_KINDS_RUN", ",".join(AGG_KINDS)).split(",")
        if not kinds or any(k not in AGG_KINDS for k in kinds):
            raise SystemExit(f"FAIL LOUD: AGG_KINDS_RUN={kinds!r}; kinds are {AGG_KINDS}")
        return [task(t, w, f"{k}_selfref", s) for k in kinds for s in range(1, 9) for t in topos for w in tuple(GROUNDED_LADDER)]
    if phase == "so1":
        # small_batch_so_v1 on the single-origin cells (--inputs client_local_v1/inputs_so_server), seeds 1-4
        smoke = os.environ.get("SO1_SMOKE", "")
        if smoke:
            return [task(topos[0], "g0x30", f"{smoke}{os.environ.get('SO1_SUFFIX', '_selfref')}", 1)]
        kinds = os.environ.get("SO1_KINDS_RUN", "so1load,so1mpoff").split(",")
        if not kinds or any(k not in SO1_KINDS for k in kinds):
            raise SystemExit(f"FAIL LOUD: SO1_KINDS_RUN={kinds!r}; kinds are {SO1_KINDS}")
        # _cdapply: CD's refine passes started from the learned plan (seeded_cd_xs1_v1's arm, single origin)
        suffix = os.environ.get("SO1_SUFFIX", "_selfref")
        if suffix not in ("_selfref", "_cdapply"):
            raise SystemExit(f"FAIL LOUD: SO1_SUFFIX={suffix!r}")
        return [task(t, w, f"{k}{suffix}", s) for k in kinds for s in (1, 2, 3, 4) for t in topos for w in tuple(GROUNDED_LADDER)]
    if phase == "tp1":
        # transfer_physics_v1: rules + zero-shot so1load on the single-origin server cells under one physics
        # condition (TP1_COND); the driver's env carries HEROSIM_TRANSFER_MODEL / HEROSIM_REPLICA_RELEASE
        cond = os.environ.get("TP1_COND", "")
        want = TP1_CONDS.get(cond)
        if want is None:
            raise SystemExit(f"FAIL LOUD: TP1_COND={cond!r}; conditions are {sorted(TP1_CONDS)}")
        got = (os.environ.get("HEROSIM_TRANSFER_MODEL", "store_forward"), os.environ.get("HEROSIM_REPLICA_RELEASE", "0"))
        if got != want:
            raise SystemExit(f"FAIL LOUD: TP1_COND={cond} needs (transfer, release)={want}, env has {got}")
        smoke = os.environ.get("TP1_SMOKE", "")
        if smoke:
            return [task(topos[0], "g0x20", k, 1 if k.endswith("_selfref") else 0) for k in smoke.split(",")]
        # kpa_scaleout_v1: under kpa the store-and-forward / held cell is a new condition and runs in full
        shared = os.environ.get("HEROSIM_SHARED_AUTOSCALER", "0") == "1"
        if cond == "replay" and os.environ.get("HEROSIM_SCALEOUT", "legacy") == "legacy" and not shared:
            return [task(t, w, k, 0) for t in topos[:4] for w in ("g0x20", "g1x20") for k in ("cd", "batched", "reactive")]
        rules = ("reactive", "selfpredict", "locality", "batched", "cd")
        # kpa_scaleout_v1 A4 control: TP1_ARMS restricts the arms (e.g. the two that changed autoscaler)
        only = [k for k in os.environ.get("TP1_ARMS", "").split(",") if k]
        if any(k not in rules + ("so1load_selfref",) for k in only):
            raise SystemExit(f"FAIL LOUD: TP1_ARMS={only!r}")
        keep = (lambda k: k in only) if only else (lambda k: True)
        return ([task(t, w, k, 0) for k in rules if keep(k) for t in topos for w in tuple(GROUNDED_LADDER)]
                + [task(t, w, "so1load_selfref", s) for s in (1, 2) if keep("so1load_selfref") for t in topos
                   for w in tuple(GROUNDED_LADDER)])
    if phase == "het1":
        # hetero vs plain bipartite convs on the 19 topologies; lf1gnn / lf1twin / lf1mlp are local_features_v1's runs
        smoke = os.environ.get("HET_SMOKE", "")
        if smoke:
            return [task(topos[0], "g0x30", "lf1het_selfref", 1)]
        return [task(t, w, "lf1het_selfref", s) for s in range(1, 9) for t in topos for w in tuple(GROUNDED_LADDER)]
    if phase == "cl1":
        kinds = os.environ.get("CL1_KINDS_RUN", ",".join(CL1_KINDS)).split(",")
        if not kinds or any(k not in CL1_KINDS + CL1_LEARNED for k in kinds):
            raise SystemExit(f"FAIL LOUD: CL1_KINDS_RUN={kinds!r}; kinds are {CL1_KINDS + CL1_LEARNED}")
        seeds = [int(x) for x in os.environ.get("CL1_SEEDS", "1,2,3,4").split(",")]
        gw = tuple(GROUNDED_LADDER)
        return ([task(t, w, k) for k in kinds if k in CL1_KINDS for t in topos for w in gw] +
                [task(t, w, f"{k}_selfref", s) for k in kinds if k in CL1_LEARNED for s in seeds for t in topos for w in gw])
    if phase == "rb1":
        # rule_baselines_v1: six hand rules on small_batch_confirm_v1's 19 topologies, the grounded x2/x3/x5 ladder
        kinds = os.environ.get("RB1_KINDS_RUN", ",".join(RB1_KINDS)).split(",")
        if not kinds or any(k not in RB1_KINDS for k in kinds):
            raise SystemExit(f"FAIL LOUD: RB1_KINDS_RUN={kinds!r}; kinds are {RB1_KINDS}")
        return [task(t, w, k) for k in kinds for t in topos for w in tuple(GROUNDED_LADDER)]
    if phase == "scale":
        # scale_sweep_v1 (exploratory, development topologies): SCALE_CONDS_RUN picks the conditions of this part
        conds = os.environ.get("SCALE_CONDS_RUN", ",".join(SCALE_CONDS)).split(",")
        if not conds or any(c not in SCALE_CONDS for c in conds):
            raise SystemExit(f"FAIL LOUD: SCALE_CONDS_RUN={conds!r}; conditions are {SCALE_CONDS}")
        gw = tuple(w for w in SCALE_LADDER if any(w.endswith(c) for c in conds))
        learned = [(f"{k}_selfref", s) for k in ("xs1load", "xs1mpoff") + SB1_KINDS for s in (1, 2, 3, 4)]
        return [task(t, w, k, s) for k, s in [("cd", 0), ("cdextr", 0)] + learned for t in topos for w in gw]
    if phase == "lf1conf":
        # local_features_v1 on small_batch_confirm_v1's 19 topologies: the three arms at 8 seeds (the references --
        # CD, cdextr, sb1load, sb1mpoff -- are read from that gate's directory)
        smoke = os.environ.get("LF1_SMOKE", "")
        if smoke:
            if smoke not in LF1_KINDS:
                raise SystemExit(f"FAIL LOUD: LF1_SMOKE={smoke!r}")
            return [task(topos[0], "g0x30", f"{smoke}_selfref", 1)]
        kinds = os.environ.get("LF1_KINDS_RUN", ",".join(LF1_KINDS)).split(",")
        if not kinds or any(k not in LF1_KINDS for k in kinds):
            raise SystemExit(f"FAIL LOUD: LF1_KINDS_RUN={kinds!r}")
        gw = tuple(GROUNDED_LADDER)
        return [task(t, w, f"{k}_selfref", s) for k in kinds for s in range(1, 9) for t in topos for w in gw]
    if phase == "rp2dev" or phase in RAW_V2:
        # raw_plan_v2 Phase D: the development cells; one phase per arm so arms are not held back by each other
        gw = tuple(GROUNDED_LADDER)
        kinds = tuple(f"{k}_selfref" for k in RAW_V2) if phase == "rp2dev" else (f"{phase}_selfref",)
        return [task(t, w, k, s) for k in kinds for s in (1, 2, 3, 4) for t in topos for w in gw]
    if phase == "peakmlp":
        gw = tuple(GROUNDED_LADDER)
        return [task(t, w, "xs1mpoff_selfref", s) for s in (1, 2, 3, 4) for t in topos for w in gw]
    if phase == "groundedladder":
        # peak_load_v2: CD and the GNN first so the primary can be read early, then every other rule
        gw = tuple(GROUNDED_LADDER)
        return [task(t, w, "cd") for t in topos for w in gw] + \
               [task(t, w, "xs1load_selfref", s) for s in (1, 2, 3, 4) for t in topos for w in gw] + \
               [task(t, w, k) for k in GROUNDED_RULES if k != "cd" for t in topos for w in gw]
    if phase == "grounded":
        gw = tuple(GROUNDED_WINDOWS)
        # witness: xs1load_selfref seed 1 on two x1.1 draws must equal capacity_sweep_v1's runs to the digit
        return [task(t, "w0x11d1", "xs1load_selfref", 1) for t in (9119, 9420)] + \
               [task(t, w, k) for k in GROUNDED_RULES for t in topos for w in gw] + \
               [task(t, w, k, s) for k in GROUNDED_LEARNED for s in (1, 2, 3, 4) for t in topos for w in gw]
    if phase == "x11confirm":
        # x11_confirm_v1: xs1load_selfref seeds 2-4 on the x1.1 draws (seed 1, CD and reactive are capacity_sweep_v1's
        # runs); plus seed 1 on two cells at this commit, the witness that the served path is unchanged.
        x11 = [f"w0x11d{k}" for k in JIT_DRAWS]
        return [task(t, w, "xs1load_selfref", s) for s in (2, 3, 4) for w in x11 for t in topos] + \
               [task(t, "w0x11d1", "xs1load_selfref", 1) for t in (9119, 9420)]
    if phase == "decima":
        return [task(t, w, "decima") for t in topos for w in WINDOWS]
    if phase == "jitsmoke":
        return [task(9119, f"w0x15d{k}", "cd") for k in (1, 2, 3)]
    if phase == "capacity":
        # capacity_sweep_v1 (docs/lineages/capacity_sweep_v1.md). C1: the new rungs, plus the x1 draws' missing
        # reactive and self-predict runs; C2: every arm at x1 with keep_alive removed.
        new = [f"w0{r}d{k}" for r in CAP_RUNGS for k in JIT_DRAWS]
        x10 = [f"w0x10d{k}" for k in JIT_DRAWS]
        return [task(t, w, k) for w in new for t in topos for k in CAP_ARMS] + \
               [task(t, w, "xs1load_selfref", 1) for w in new for t in topos] + \
               [task(t, w, k) for w in x10 for t in topos for k in ("reactive", "selfpredict")] + \
               [task(t, w, k) for w in KA_WINDOWS for t in topos for k in ("reactive", "cd", "selfpredict")] + \
               [task(t, w, "xs1load_selfref", 1) for w in KA_WINDOWS for t in topos]
    if phase == "guard":
        # replica_guard_v1: guard vs its twin vs CD on the x1 perturbed draws; 9434 w0 as run in every gate;
        # w1-w3 unperturbed, guard vs twin, seeds 1-2
        x10 = [f"w0x10d{k}" for k in JIT_DRAWS]
        return [task(t, w, "cd") for w in x10 for t in topos] + \
               [task(t, w, k, s) for w in x10 for k in ("xs1load_selfref", "xs1load_selfrefkw")
                for s in (1, 2, 3, 4) for t in topos] + \
               [task(9434, "w0", k, s) for k in ("xs1load_selfref", "xs1load_selfrefkw") for s in (1, 2, 3, 4)] + \
               [task(t, w, k, s) for w in ("w1", "w2", "w3") for k in ("xs1load_selfref", "xs1load_selfrefkw")
                for s in (1, 2) for t in topos]
    if phase == "ladderjit":
        x15 = [f"w0x15d{k}" for k in JIT_DRAWS]
        x10 = [f"w0x10d{k}" for k in JIT_DRAWS]
        return [task(t, w, k) for w in x15 for t in topos for k in LADDER_ARMS] + \
               [task(t, w, "xs1load_selfref", s) for w in x15 for s in (1, 2, 3, 4) for t in topos] + \
               [task(t, w, k) for w in x10 for t in topos for k in ("cd", "cd_inflight")] + \
               [task(t, w, "xs1load_selfref", 1) for w in x10 for t in topos]
    if phase == "ladder":
        return [task(t, w, k) for w in LADDER_RUNGS for t in topos for k in LADDER_ARMS] + \
               [task(t, w, k, s) for w in LADDER_RUNGS for k in LADDER_LEARNED for s in (1, 2, 3, 4) for t in topos]
    if phase == "xs1cd":
        # seeded_cd_xs1_v1: CD's refine passes started from the xs1load plan, and from the MP-OFF bc1 plan
        return [task(t, w, k, s) for k in ("xs1load_cdapply", "bc1mpoff_cdapply") for s in (1, 2, 3, 4)
                for t in topos for w in WINDOWS]
    if phase == "xs1":
        # exchange_seconds_v1: the exchange-in-seconds arm, self-refined (the registered arm) and plain
        return [task(t, w, k, s) for k in ("xs1load_selfref", "xs1load") for s in (1, 2, 3, 4)
                for t in topos for w in WINDOWS]
    if phase == "fc1":
        # fullctx_refine_v1: the full-context-trained arm, self-refined (the registered arm) and plain
        return [task(t, w, k, s) for k in ("fc1load_selfref", "fc1load") for s in (1, 2, 3, 4)
                for t in topos for w in WINDOWS]
    if phase == "bc1selfref":
        # backlog_corpus_v1 Amendment 3: bc1load with 3 self-refine passes on its own score
        return [task(t, w, "bc1load_selfref", s) for s in (1, 2, 3, 4) for t in topos for w in WINDOWS]
    if phase == "bc1":
        return [task(t, w, k, s) for k in BC1_KINDS + ("v4load_se",) for s in (1, 2, 3, 4)
                for t in topos for w in WINDOWS]
    if phase == "fix":
        # load_repr_v1 Amendment 2: every learned arm again after the uncapped-rung rank fix (721d44f)
        return [task(t, w, k, s) for k in V4_KINDS for s in (1, 2, 3, 4) for t in topos for w in WINDOWS] + \
               [task(t, w, k, s) for k in ("gnnedge0", "mpoff") for s in SEEDS for t in topos for w in WINDOWS]
    if phase == "d6":
        return [task(t, w, "gnnedge0_selfref", s) for s in SEEDS for t in topos for w in WINDOWS]
    if phase == "d5":
        return [task(t, w, k, s) for k in ("gnnedge0_cdapply", "mpoff_cdapply") for s in SEEDS
                for t in topos for w in WINDOWS] + \
               [task(t, w, "gnnedge0_cdshadow", s) for s in (1, 2, 3) for t in topos for w in WINDOWS]
    if phase == "d4":
        return [task(t, w, "cd_slate") for t in topos for w in WINDOWS] + \
               [task(t, w, "gnnedge0_slate", s) for t in topos for w in WINDOWS for s in SEEDS]
    if phase == "d2":
        return [task(t, w, k, s) for t in topos for w in WINDOWS for k in ("gnnedge0_spread", "mpoff_spread")
                for s in SEEDS]
    out = [task(t, w, k) for t in topos for w in WINDOWS for k in GATE_RULES]
    out += [task(t, w, k, s) for t in topos for w in WINDOWS for k in ("gnnedge0", "mpoff") for s in SEEDS]
    return out


def arm_name(t: Dict[str, object]) -> str:
    return f"cc40s{t['topo']}__{t['window']}__{t['kind']}_s{t['seed']}"


def _reactive_disqualified(topo: int, out_dir: str) -> bool:
    """Screen shortcut that cannot change admission: a failed or saturated reactive window already
    rules the topology out, so its batch path is not run."""
    for w in WINDOWS:
        base = os.path.join(out_dir, f"cc40s{topo}__{w}__reactive_s0")
        if os.path.exists(base + ".failed.json"):
            return True
        if os.path.exists(base + ".summary.json"):
            share = json.load(open(base + ".summary.json")).get("queue_share")
            if share is None or float(share) > 0.80:
                return True
    return False


def run_one(t: Dict[str, object], inputs: str, out_dir: str, mem: str, timeout_s: int) -> str:
    name = arm_name(t)
    summary = os.path.join(out_dir, name + ".summary.json")
    failed = os.path.join(out_dir, name + ".failed.json")
    if os.path.exists(summary) or os.path.exists(failed):
        return f"[skip] {name}"
    kind, seed, window = str(t["kind"]), int(t["seed"]), str(t["window"])
    kind, dc_k = split_dc(kind)
    dc_ms = dc_cost_ms(kind, dc_k, window[2:])
    if R1A_ON:
        state = pw.rung_state(out_dir, window[2:])
        if pw.paused(state):
            return f"[PAUSED rung {window[2:]}: {state['hung']} of {state['decided']} decided cells hung, line {pw.PAUSE_SHARE:.0%}] {name}"
    if kind == "batched" and _reactive_disqualified(int(t["topo"]), out_dir):
        return f"[skip, reactive already disqualifies] {name}"
    wl_dir = None
    inputs_root = inputs
    if window in GROUNDED_LADDER or window in SCALE_LADDER or window in WF1_LADDER:
        wl_name, sub = GROUNDED_LADDER.get(window) or SCALE_LADDER.get(window) or WF1_LADDER[window]
        inputs_root, inputs = inputs, os.path.join(inputs, sub)
    elif window in GROUNDED_X15:
        wl_name = GROUNDED_X15[window]
        inputs = os.path.join(inputs, "grounded_x15")
    elif window in GROUNDED_WINDOWS:
        wl_name = GROUNDED_WINDOWS[window]
        wl_dir = os.path.join(inputs, "grounded", "wl")
    elif window in LADDER_RUNGS:
        inputs = os.path.join(inputs, "ladder", LADDER_RUNGS[window])
        wl_name = "burst_drainable_f4000_n50000"
    else:
        wl_name = "burst_drainable_f4000_n50000" if window == "w0" else f"burst_drainable_{window}_n50000"
    cfg = os.path.join(inputs, "cfg", f"cc40s{t['topo']}.json")
    wl = os.path.join(wl_dir or os.path.join(inputs, "wl"), wl_name + ".json")
    env = dict(os.environ)
    for k in ("GNN_MODEL_PATH", "GNN_DISABLE_MESSAGE_PASSING", "GNN_MP_PLATFORM_EDGES_OFF", "GNN_BATCH_TIMEOUT",
              "GNN_BATCH_SIZE", "GNN_DECODE_MODE", "GNN_BATCH_BY_PEER_GROUP", "GNN_PREFIX_ALPHA_KEY",
              "LIVE_AUDIT_SNAPSHOT_PATH", "HEROSIM_ROLLOUT_SCORER", "HEROSIM_PG_EXCHANGE_SCALE",
              "HEROSIM_PG_ORACLE_NODES", "HEROSIM_PG_CD_PASSES", "HEROSIM_DECIMA_ALPHA", "HEROSIM_MAX_EVENTS", "HEROSIM_FORCED_PLACEMENTS",
              "HEROSIM_EXEC_PHYSICS", "HEROSIM_EXEC_SEED", "HEROSIM_PG_EXEC_KNOWLEDGE", "HEROSIM_PG_BATCH_BLIND",
              "GNN_PREFIX_SIBLING_SPREAD", "GNN_SERVE_CORPUS_SLATE", "NEAR_RTT_LABEL_OVERRIDE_JSON", "GNN_CD_REFINE",
              "GNN_PREFIX_SELF_REFINE", "HEROSIM_POLICY_TIME_SCALE", "PARTIAL_STATE_CONTRACT",
              "PARTIAL_STATE_LOAD_SECONDS", "PARTIAL_STATE_PEER_MASS", "HEROSIM_INFLIGHT_CAPTURE",
              "HEROSIM_PG_INFLIGHT", "HEROSIM_KEEP_ALIVE", "HEROSIM_PG_EXT_RATE", "HEROSIM_PROGRESS_FILE", "HEROSIM_CD_RANDOM_SEED",
              "HEROSIM_PG_PULL_HOLD", "HEROSIM_PULL_LEDGER", "HEROSIM_PG_CD_EXPANSION", "HEROSIM_PG_CD_EXACT", "HEROSIM_PG_CD_EXACT_GNN", "HEROSIM_PG_CD_EXACT_PRUNE", "HEROSIM_PG_SHARED_LINK", "HEROSIM_PG_INPUT_LINK", "GNN_SLATE_NO_SPLIT", "GNN_SELF_SEARCH", *KEEPWARM_ENV, *ORACLE_ENV):
        env.pop(k, None)
    for k in OBS_ENV:
        env.pop(k, None)
    env.pop("HEROSIM_DECISION_COST_MS", None)
    if dc_ms is not None:
        env["HEROSIM_DECISION_COST_MS"] = dc_ms
    for k in SNAP_ENV:
        env.pop(k, None)
    sealed = sealed_triggers(int(t["topo"]), window)
    if sealed is not None:
        trig = os.path.join(out_dir, name + ".triggers.json")
        with open(trig, "w") as fh:
            json.dump(sealed, fh)
        env.update(LIVE_AUDIT_SNAPSHOT_PATH=os.path.join(out_dir, name + ".snapshots.jsonl"), LIVE_AUDIT_TRIGGER_IDS=trig,
                   LIVE_AUDIT_MAX_SNAPSHOTS=str(len(sealed)), HEROSIM_SNAPSHOT_FIDELITY="1")
    if window in KA_WINDOWS:
        env["HEROSIM_KEEP_ALIVE"] = CAP_KEEP_ALIVE
    # cd_gap_v1 B': a rate-stretched cell scales keep_alive and the reconcile interval by its own factor
    rate_scale = float((json.load(open(cfg)).get("cd_gap_v1_rate_scale") or {}).get("factor", 1.0))
    # physics_audit_v1 B1: R1 runs one policy time scale on every rung; GATE_FIXED_POLICY_TIME_SCALE pins it
    # (the cell's rate factor still sets cdextr's per-rung label rate)
    fixed = os.environ.get("GATE_FIXED_POLICY_TIME_SCALE")
    time_scale = float(fixed) if fixed else rate_scale
    if time_scale != 1.0:
        env["HEROSIM_POLICY_TIME_SCALE"] = repr(time_scale)
    env.update(HEROSIM_PEER_EXCHANGE="1", HEROSIM_SERVER_ONLY_REPLICAS="1", HEROSIM_WARMTH_PHYSICS="node_disk_v2",
               PYTHONHASHSEED="0", HEROSIM_GNN_DEVICE="cpu", SIM_FORCE_FULL_STATS="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="", PYTHONPATH=REPO)
    client_local = bool(json.load(open(cfg)).get("client_local_v1"))
    if client_local:
        env["HEROSIM_SERVER_ONLY_REPLICAS"] = "0"
    base_kind = next((kind[:-len(s)] for s in SUFFIXES if kind.endswith(s)), kind)
    if base_kind in LEARNED_KINDS:
        # cd_gap_v1 A: the CD imitator is the gnnedge0 architecture on the jb2 corpus and split
        if base_kind == "cdimit":
            stem = "cd-gap-v1-cdimit-gnnedge0"
        elif base_kind in V4_KINDS:
            stem = f"load-repr-v1-{base_kind}-gnnedge0"
        elif base_kind == "fc1load":
            stem = "fullctx-refine-v1-fc1load"
        elif base_kind == "xs1load":
            stem = "exchange-seconds-v1-xs1load"
        elif base_kind == "xs1mpoff":
            stem = "peak-controls-v1-xs1mpoff"
        elif base_kind in ("rawgnn", "rawmlp"):
            stem = f"raw-plan-v1-{base_kind}"
        elif base_kind in RAW_V2:
            stem = f"raw-plan-v2-{base_kind}"
        elif base_kind in SB1_KINDS:
            stem = f"small-batch-v1-{base_kind}"
        elif base_kind in LF1_KINDS:
            stem = f"local-features-v1-{base_kind}"
        elif base_kind in AGG_KINDS:
            stem = f"aggr-sum-v1-{base_kind}"
        elif base_kind in SO1_KINDS:
            stem = f"small-batch-so-v1-{base_kind}"
        elif base_kind in HET_KINDS:
            stem = f"hetero-conv-v1-{base_kind}"
        elif base_kind in BC1_KINDS:
            stem = f"backlog-corpus-v1-{base_kind}"
        elif base_kind in RA_KINDS:
            stem = os.environ.get("R1A_STEM", "r1-attribution-v1") + "-" + base_kind[3:].replace("_", "-")
        else:
            stem = f"joint-burst-v2-{base_kind}"
        # checkpoints and the split file sit at the inputs root (r1a: one set for every rung); an older layout kept them in the rung directory
        mroot = inputs_root if os.path.isdir(os.path.join(inputs_root, "models")) else inputs
        ck = os.path.join(mroot, "models", f"{stem}-seed{seed}.pt" if base_kind in RA_KINDS else f"{stem}-lr2e3-seed{seed}.pt")
        side = ck[:-3] + ".contract.json"
        check_kind = "gnnedge0" if base_kind == "cdimit" else {"sb1load": "xs1load", "sb1mpoff": "xs1mpoff", "so1load": "xs1load", "so1mpoff": "xs1mpoff", "so1lfgnn": "lf1gnn", "so1lfmlp": "lf1mlp"}.get(base_kind, base_kind)
        split = (os.environ.get("R1A_SPLIT", "r1_attribution_v1_split.json") if base_kind in RA_KINDS else "small_batch_so_v1_split.json" if base_kind in SO1_KINDS else "small_batch_v1_split.json" if base_kind in SB1_KINDS + LF1_KINDS + AGG_KINDS + HET_KINDS + SO1_KINDS else "backlog_corpus_v1_split.json" if base_kind in BC1_KINDS
                 else "joint_burst_v2_split.json")
        rc = subprocess.run(PY + [os.path.join(REPO, "scripts_cosim/joint_burst_v2_sidecheck.py"), side, check_kind,
                                  os.path.join(mroot, split), "inf"], env=env, cwd=REPO)
        if rc.returncode != 0:
            raise SystemExit(f"FAIL LOUD: sidecheck failed for {ck}")
        env.update(GNN_MODEL_PATH=ck, GNN_DECODE_MODE="masked_topo", GNN_BATCH_BY_PEER_GROUP="1",
                   GNN_PREFIX_ALPHA_KEY="inf")
        if base_kind in ("mpoff", "bc1mpoff", "xs1mpoff", "sb1mpoff", "so1mpoff", "so1lfmlp", "rawmlp", "rawStwin", "lf1twin", "lf1mlp") + RA_MP_OFF:
            env["GNN_DISABLE_MESSAGE_PASSING"] = "1"
        if base_kind in RA_KINDS:
            # exported, not adopted, so run_provenance records them; the loader verifies the sidecar. The graph-free arms
            # additionally refuse a serving environment that names another arm.
            env.update(PARTIAL_STATE_CONTRACT="partial_state_v5", PARTIAL_STATE_LOAD_SECONDS="1",
                       PARTIAL_STATE_EXCHANGE_SECONDS="1", GNN_SERVE_CANDIDATE_SLATE=ra_serving_slate(side))
            if base_kind in ("ra_mlp_same", "ra_set_transformer"):
                env["GNN_ARM_KIND"] = base_kind[3:]
        if base_kind in LOAD_KINDS:
            # exported, not adopted, so run_provenance records them; the loader verifies the sidecar
            env.update(PARTIAL_STATE_CONTRACT="partial_state_v4",
                       PARTIAL_STATE_LOAD_SECONDS="0" if base_kind == "v4twin" else "1",
                       PARTIAL_STATE_EXCHANGE_SECONDS="1" if base_kind in ("xs1load", "xs1mpoff") + RAW_KINDS + SB1_KINDS + LF1_KINDS + AGG_KINDS + HET_KINDS + SO1_KINDS else "0")
        if base_kind in BC1_KINDS or kind.endswith("_se"):
            env["HEROSIM_INFLIGHT_CAPTURE"] = SERVICE_END
        if kind.endswith("_spread"):
            env["GNN_PREFIX_SIBLING_SPREAD"] = "1"
        if kind.endswith("_slate"):
            env["GNN_SERVE_CORPUS_SLATE"] = "1"
        if kind.endswith("_cdshadow"):
            env["GNN_CD_REFINE"] = "shadow"
        if kind.endswith("_cdapply"):
            env["GNN_CD_REFINE"] = "apply"
        if kind.endswith("_cdxapply"):
            env.update(GNN_CD_REFINE="apply", HEROSIM_PG_CD_EXPANSION="1", GNN_SLATE_NO_SPLIT="1")
        if kind.endswith(("_cdxib0", "_cdxhib0")):
            env.update(GNN_CD_REFINE="apply", HEROSIM_PG_CD_EXPANSION="1", GNN_SLATE_NO_SPLIT="1", **ib0_alpha(kind))
        if kind.endswith("_cdxhib0"):
            env["GNN_CD_SEED"] = "hand"
        if kind.endswith("_cdxexg"):
            env.update(GNN_CD_REFINE="apply", HEROSIM_PG_CD_EXPANSION="1", HEROSIM_PG_CD_EXACT="1", HEROSIM_PG_CD_EXACT_GNN="1", GNN_SLATE_NO_SPLIT="1")
        if kind.endswith(("_cdxpra", "_cdxprb")):
            env.update(GNN_CD_REFINE="apply", HEROSIM_PG_CD_EXPANSION="1", HEROSIM_PG_CD_EXACT="1", GNN_SLATE_NO_SPLIT="1",
                       HEROSIM_PG_CD_EXACT_PRUNE="gnn" if kind.endswith("_cdxpra") else "cost")
        if kind.endswith(("_nosplit", "_selfsearch")):
            env["GNN_SLATE_NO_SPLIT"] = "1"
        if kind.endswith("_selfsearch"):
            env["GNN_SELF_SEARCH"] = "1"
        if kind.endswith(("_selfref", "_selfrefkw")):
            env["GNN_PREFIX_SELF_REFINE"] = "3"
        if kind.endswith("_selfrefkw"):
            env.update(KEEPWARM_ENV)
        policy = "gnn"
    else:
        policy = RULE_POLICY[kind]
        if kind in ("batched", "locality", "cd", "cd_blind", "cd_inflight", "cd_pull", "cd_ledger", "cd_declared", "cd_expand", "cd_exactS", "cd_oracle", "cd_ib0", *EXACT_ALPHA, "cd_random_seed", *EXT_KINDS) or policy == "decima_wfair_network":
            env.update(GNN_DECODE_MODE="masked_topo", GNN_BATCH_BY_PEER_GROUP="1")
        if kind == "cd_declared":
            env["GNN_SERVE_CANDIDATE_SLATE"] = "declared_pruning_v1"
        if kind == "cd_random_seed":
            env.update(GNN_SERVE_CANDIDATE_SLATE="declared_pruning_v1", GNN_CD_REFINE="apply", HEROSIM_CD_RANDOM_SEED=str(seed))
        if kind == "locality":
            env["HEROSIM_PG_EXCHANGE_SCALE"] = repr(LOCALITY_SCALE)
        if kind == "cdext":
            env["HEROSIM_PG_EXT_RATE"] = repr(EXT_LABEL_RATE)
        if kind == "cdextr":
            env["HEROSIM_PG_EXT_RATE"] = repr(EXT_LABEL_RATE / rate_scale)
        if kind == "cd_inflight":
            env["HEROSIM_PG_INFLIGHT"] = "1"
        if kind == "cd_pull":
            env.update(HEROSIM_PG_PULL_HOLD="1", HEROSIM_PULL_LEDGER="1")
        if kind == "cd_ledger":
            env["HEROSIM_PULL_LEDGER"] = "1"
        if kind == "cd_expand":
            env["HEROSIM_PG_CD_EXPANSION"] = "1"
        if kind == "cdx_ib0":
            env.update(GNN_DECODE_MODE="masked_topo", GNN_BATCH_BY_PEER_GROUP="1", HEROSIM_PG_CD_EXPANSION="1", **ib0_alpha(kind))
        if kind == "cd_oracle_ib0":
            env.update(GNN_DECODE_MODE="masked_topo", GNN_BATCH_BY_PEER_GROUP="1", HEROSIM_PG_CD_EXACT="1", HEROSIM_PG_CD_EXPANSION="1", HEROSIM_PG_ORACLE="1",
                       HEROSIM_PG_ORACLE_SEARCH="1", HEROSIM_SNAPSHOT_FIDELITY="1", HEROSIM_ORACLE_CELL_CONFIG=os.path.abspath(cfg),
                       HEROSIM_ORACLE_SIM_INPUT=os.path.join(REPO, "data/nofs-ids"), HEROSIM_ORACLE_WORKERS=os.environ.get("R1A_ORACLE_WORKERS", "8"), **ib0_alpha(kind))
            env["ORACLE_DIAG_DIR"] = os.path.join(out_dir, "oracle_diag", name)
            if os.environ.get("R1A_ORACLE_TERMS") == "1":
                env["ORACLE_DIAG_TERMS"] = "1"
            for src_env, dst in (("R1A_ORACLE_MAX_PLANS", "HEROSIM_PG_ORACLE_MAX_PLANS"), ("R1A_ORACLE_SEARCH_BUDGET", "HEROSIM_PG_ORACLE_SEARCH_BUDGET")):
                if os.environ.get(src_env):
                    env[dst] = os.environ[src_env]
        if kind == "cd_exactS":
            env.update(HEROSIM_PG_CD_EXACT="1", HEROSIM_PG_CD_EXPANSION="1")
        if kind in EXACT_ALPHA:
            env.update(HEROSIM_PG_CD_EXACT="1", HEROSIM_PG_CD_EXPANSION="1", **EXACT_ALPHA[kind])
        if kind in OBS_ARMS:
            env.update(OBS_ARMS[kind])
            if "HEROSIM_OBS_ESTIMATOR" in OBS_ARMS[kind]:
                env["HEROSIM_OBS_SEED"] = str(seed)
        if kind == "cd_exactS_ib0_prune":
            env["HEROSIM_PG_CD_EXACT_PRUNE"] = "cost"
        if kind == "cd_ib0":
            env.update(ib0_alpha(kind))
        if kind == "cd_oracle":
            env.update(HEROSIM_PG_CD_EXACT="1", HEROSIM_PG_CD_EXPANSION="1", HEROSIM_PG_ORACLE="1", HEROSIM_SNAPSHOT_FIDELITY="1", HEROSIM_ORACLE_CELL_CONFIG=os.path.abspath(cfg),
                       HEROSIM_ORACLE_SIM_INPUT=os.path.join(REPO, "data/nofs-ids"),
                       HEROSIM_ORACLE_WORKERS=os.environ.get("R1A_ORACLE_WORKERS", "8"))
            env["ORACLE_DIAG_DIR"] = os.path.join(out_dir, "oracle_diag", name)
            if os.environ.get("R1A_ORACLE_MAX_PLANS"):
                env["HEROSIM_PG_ORACLE_MAX_PLANS"] = os.environ["R1A_ORACLE_MAX_PLANS"]
        if kind in DECIMA_TUNE_ALPHAS:
            env["HEROSIM_DECIMA_ALPHA"] = repr(DECIMA_TUNE_ALPHAS[kind])
        if kind == "decima":
            alpha = os.environ.get("HEROSIM_DECIMA_ALPHA", "").strip()
            if not alpha:
                raise SystemExit("FAIL LOUD: the decima study arm needs HEROSIM_DECIMA_ALPHA (the tuned alpha)")
            env["HEROSIM_DECIMA_ALPHA"] = alpha
        if kind == "cd_blind":
            env["HEROSIM_PG_BATCH_BLIND"] = "1"
        if kind == "cd_slate":
            env.update(GNN_DECODE_MODE="masked_topo", GNN_BATCH_BY_PEER_GROUP="1", GNN_SERVE_CORPUS_SLATE="1")
    # The ~250 MB per-task result is deleted once summarised; HEROSIM_RAW_DIR keeps it off the home quota (node-local).
    raw = os.path.join(os.environ.get("HEROSIM_RAW_DIR") or out_dir, name + ".raw.json")
    log = os.path.join(out_dir, name + ".log")
    scope = [] if NO_SCOPE else ["systemd-run", "--scope", "-q", "-p", f"MemoryMax={mem}", "-p", "MemorySwapMax=0"]
    cmd = scope + ["timeout", str(timeout_s)] + PY + [os.path.join(REPO, "src/executesimulation.py"), "--config", cfg,
                                                      "--workload", wl, "--policy", policy, "--output", raw]
    start = time.time()
    dog = None
    if R1A_ON:
        prog = os.path.join(os.environ.get("HEROSIM_RAW_DIR") or out_dir, name + ".progress")
        env["HEROSIM_PROGRESS_FILE"] = prog
        if wl not in _LAST_ARRIVAL:
            _LAST_ARRIVAL[wl] = max(float(e["timestamp"]) for e in json.load(open(wl))["events"])
        rule = {k: float(os.environ[e]) for k, e in (("warmup_s", "R1A_GRACE_S"), ("stall_s", "R1A_STALL_S")) if e in os.environ}
        dog = pw.Watchdog(prog, _LAST_ARRIVAL[wl], float(timeout_s), poll_s=float(os.environ.get("R1A_POLL_S", pw.POLL_S)), **rule)
        if "R1A_POLL_S" in os.environ:  # a test run with short cells: report as often as the watchdog polls
            env["HEROSIM_PROGRESS_INTERVAL_S"] = os.environ["R1A_POLL_S"]
    rc = run_logged(cmd, env, REPO, log, cap_bytes(), on_start=dog.attach if dog else None)  # capped: a hung run's log used to reach several GB
    wall = int(time.time() - start)
    if dog:
        dog.stop()
        if os.path.exists(prog):
            os.remove(prog)
    if dog and dog.verdict:
        json.dump({"arm": name, "returncode": rc, "wallclock_s": wall, "why": f"watchdog: {dog.verdict['reason']}",
                   "watchdog": dog.verdict, "limit_s": timeout_s, "last_arrival_s": _LAST_ARRIVAL[wl]}, open(failed, "w"))
        if os.path.exists(raw):
            os.remove(raw)
        return f"[FAILED watchdog {dog.verdict['reason']} projected={dog.verdict.get('projected_s')} {wall}s] {name}"
    if rc != 0 or not os.path.exists(raw):
        json.dump({"arm": name, "returncode": rc, "wallclock_s": wall,
                   "why": "timeout" if rc == 124 else ("memory cap or crash" if rc != 0 else "no output")},
                  open(failed, "w"))
        if os.path.exists(raw):
            os.remove(raw)
        return f"[FAILED rc={rc} {wall}s] {name}"
    doc = json.load(open(raw))
    st = doc.get("stats") or doc
    drift = queue_drift(st.get("taskResults"))
    out = {k: st.get(k) for k in ("num_tasks", "total_rtt", "averageElapsedTime", "averageQueueTime",
                                  "averageWaitTime", "totalPeerExchangeTime", "totalPeerRendezvousWait", "endTime",
                                  "schedulerCounters", "offloadingRate", "total_rtt_plus_inference",
                                  "requestFailures", "requestTimeoutS", "decisionTiming")}
    if WF1_LADDER:  # workload_fix_v1: the latency decomposition needs the stage times; other phases keep their keys
        for k in ("averageColdStartTime", "averageInitializationTime", "averagePullTime", "averageExecutionTime",
                  "averageComputeTime", "averageCommunicationsTime", "averageNetworkLatency"):
            out[k] = st.get(k)
    for k in ("peerExchangeByAccessClass", "accessClasses"):  # workload_fix_v1 W3 telemetry; absent otherwise
        if k in st:
            out[k] = st[k]
    c = out.get("schedulerCounters") or {}
    for k in ("residence_tasks", "residence_batches", "queue_range_records"):
        c.pop(k, None)
    arm_kind = RULE_POLICY.get(kind, kind)
    out.update(arm=name, cell=f"cc40s{t['topo']}", topology=int(t["topo"]), window=window, clients=int(json.load(open(cfg))["nodes"]["client_nodes"]["count"]),
               servers=int(json.load(open(cfg))["nodes"]["server_nodes"]["count"]),
               rung="C40", lever="burst", workload=wl_name, corpus=("jb2-cdlabel" if base_kind == "cdimit" else "r1a" if base_kind in RA_KINDS else "bc1" if base_kind in BC1_KINDS else "jb2") if base_kind in LEARNED_KINDS else "none",
               arm_kind=arm_kind, checkpoint_seed=seed, policy_name=policy, wallclock_s=wall)
    out["env"] = {k: v for k, v in (doc.get("run_provenance") or {}).get("env", {}).items() if v}
    out["queue_drift"] = drift
    out["code"] = (doc.get("run_provenance") or {}).get("code")
    out["python_env"] = (doc.get("run_provenance") or {}).get("python_env")  # the raw JSON is deleted at job exit; PARITY.md reads the interpreter from here
    out["env_fingerprint"] = (doc.get("run_provenance") or {}).get("env_fingerprint")
    n = out.get("num_tasks")
    problems = []
    orig_kind, kind = kind, check_alias(kind)
    for k, default in (("HEROSIM_TRANSFER_MODEL", "store_forward"), ("HEROSIM_REPLICA_RELEASE", "0"),
                       ("HEROSIM_SCALEOUT", "legacy"), ("HEROSIM_SHARED_AUTOSCALER", "0")):
        if out["env"].get(k, default) != os.environ.get(k, default):
            problems.append(f"physics not recorded as driven: {k}={out['env'].get(k)!r}, driver {os.environ.get(k)!r}")
    # kpa_scaleout_v1: the autoscaler's own record of the rule it ran; legacy runs carry none
    out["scaleOut"] = st.get("scaleOut")
    # reference_physics_programme metrics: cold-start share (percent of tasks) travels with every summary
    out["cold_start_pct"] = st.get("coldStartProportion")
    out["latency_percentiles"] = latency_percentiles(st.get("taskResults"))
    out["placement_wait"] = placement_wait(st.get("taskResults"))
    out["backlog_profile"] = backlog_profile(st.get("taskResults"))
    out["backlog_profile_v2"] = backlog_profile(st.get("taskResults"), with_lock_wait=True)
    out["lock_wait"] = lock_wait_profile(st.get("taskResults"))
    out["arrival_end"] = arrival_end(wl, st.get("endTime"))
    out["replica_count_series"] = replica_count_series(st.get("systemEvents"), st.get("endTime"))
    out["steady_state"] = steady_state_latency(st.get("taskResults"))
    if os.environ.get("HEROSIM_SCALEOUT", "legacy") == "kpa":
        want = {"mode": "kpa", "target": 0.7, "stable_window_s": 60.0 * time_scale,
                "panic_window_s": 6.0 * time_scale, "panic_threshold": 2.0}
        got = {k: (out["scaleOut"] or {}).get(k) for k in want}
        if any(not isinstance(got[k], (int, float)) or abs(got[k] - v) > 1e-9 for k, v in want.items() if k != "mode") \
                or got["mode"] != "kpa":
            problems.append(f"kpa scale-out not served as registered: {got}, want {want}")
    elif out["scaleOut"] is not None:
        problems.append(f"legacy run carries a scaleOut block: {out['scaleOut']}")
    if out["env"].get("HEROSIM_SERVER_ONLY_REPLICAS") != ("0" if client_local else "1"):
        problems.append(f"served HEROSIM_SERVER_ONLY_REPLICAS={out['env'].get('HEROSIM_SERVER_ONLY_REPLICAS')!r}, "
                        f"cell client_local_v1={client_local}")
    if kind == "offload" and float(out.get("offloadingRate") or 0.0) < 100.0:
        problems.append(f"offload arm ran {100.0 - float(out.get('offloadingRate') or 0.0):.2f} % of tasks locally")
    out["client_local_v1"] = client_local
    if time_scale != 1.0 and out["env"].get("HEROSIM_POLICY_TIME_SCALE") != repr(time_scale):
        problems.append(f"policy time scale not recorded: {out['env'].get('HEROSIM_POLICY_TIME_SCALE')!r}")
    if time_scale == 1.0 and out["env"].get("HEROSIM_POLICY_TIME_SCALE") not in (None, "1.0"):
        problems.append(f"policy time scale should be 1.0, served {out['env'].get('HEROSIM_POLICY_TIME_SCALE')!r}")
    out["policy_time_scale"] = time_scale
    if n is None or int(n) != N_TASKS:
        problems.append(f"num_tasks={n!r}")
    want_ka = CAP_KEEP_ALIVE if window in KA_WINDOWS else None
    if out["env"].get("HEROSIM_KEEP_ALIVE") != want_ka:
        problems.append(f"served HEROSIM_KEEP_ALIVE={out['env'].get('HEROSIM_KEEP_ALIVE')!r}, {window} needs {want_ka!r}")
    if window in LADDER_RUNGS and window.startswith(("w0x11", "w0x12", "w0x13", "w0x10d")) and drift is None:
        problems.append("queue drift not computable: no taskResults in the raw result")
    if kind == "selfpredict" and (int(c.get("pg_decisions") or 0) != N_TASKS or int(c.get("pg_lookahead_priced") or 0) == 0):
        problems.append(f"rule instrument off: {c.get('pg_decisions')}/{c.get('pg_lookahead_priced')}")
    if kind.endswith("_selfrefkw") and int(c.get("keepwarm_armed") or 0) != 1:
        problems.append("keep-warm guard not armed")
    if not kind.endswith("_selfrefkw") and int(c.get("keepwarm_armed") or 0):
        problems.append("keep-warm guard armed on an unguarded arm")
    if kind.endswith(("_selfref", "_selfrefkw")) and int(c.get("prefix_self_refine_batches") or 0) == 0:
        problems.append("self-refine instrument off: prefix_self_refine_batches == 0")
    if not kind.endswith(("_selfref", "_selfrefkw")) and int(c.get("prefix_self_refine_batches") or 0):
        problems.append("unrefined arm self-refined")
    if kind.endswith(("_cdshadow", "_cdapply", "_cdxapply", "_cdxexg", "_cdxpra", "_cdxprb")) and int(c.get("cdr_batches") or 0) == 0:
        problems.append("cd-refine instrument off: cdr_batches == 0")
    if not kind.endswith(("_cdshadow", "_cdapply", "_cdxapply", "_cdxexg", "_cdxpra", "_cdxprb")) and kind != "cd_random_seed" and int(c.get("cdr_batches") or 0):
        problems.append("unrefined arm was refined")
    expands = kind.endswith(("_cdxapply", "_cdxexg", "_cdxpra", "_cdxprb")) or kind == "cd_expand"
    prune_kind = kind.endswith(("_cdxpra", "_cdxprb")) or kind == "cd_exactS_ib0_prune"
    if prune_kind and (out["env"].get("HEROSIM_PG_CD_EXPANSION") != "1" or int(c.get("pg_expand_batches") or 0) != 0):
        # the prune arms run the expansion only as a dry counterfactual whose counters are restored; applied expansion must read 0
        problems.append(f"prune arm expansion flag/counter wrong: HEROSIM_PG_CD_EXPANSION={out['env'].get('HEROSIM_PG_CD_EXPANSION')!r}, pg_expand_batches {c.get('pg_expand_batches')} (want 0)")
    if expands and not prune_kind and (out["env"].get("HEROSIM_PG_CD_EXPANSION") != "1" or int(c.get("pg_expand_batches") or 0) == 0):
        problems.append(f"expansion instrument off: HEROSIM_PG_CD_EXPANSION={out['env'].get('HEROSIM_PG_CD_EXPANSION')!r}, pg_expand_batches {c.get('pg_expand_batches')}")
    if not expands and kind != "cd_exactS" and kind != "cd_oracle" and kind not in EXACT_ALPHA and (out["env"].get("HEROSIM_PG_CD_EXPANSION") or int(c.get("pg_expand_batches") or 0)):
        problems.append("a non-expansion arm expanded")
    if kind == "cd_exactS" and (out["env"].get("HEROSIM_PG_CD_EXACT") != "1" or int(c.get("pg_exact_batches") or 0) == 0):
        problems.append(f"exact-search instrument off: HEROSIM_PG_CD_EXACT={out['env'].get('HEROSIM_PG_CD_EXACT')!r}, pg_exact_batches {c.get('pg_exact_batches')}")
    if kind in EXACT_ALPHA and (out["env"].get("HEROSIM_PG_CD_EXACT") != "1" or int(c.get("pg_exact_batches") or 0) == 0):
        problems.append(f"exact-search instrument off: HEROSIM_PG_CD_EXACT={out['env'].get('HEROSIM_PG_CD_EXACT')!r}, pg_exact_batches {c.get('pg_exact_batches')}")
    obs_want = OBS_ARMS.get(kind, {})
    for var in OBS_ENV:
        want_o = obs_want.get(var, str(seed) if (var == "HEROSIM_OBS_SEED" and "HEROSIM_OBS_ESTIMATOR" in obs_want) else None)
        if out["env"].get(var) != want_o:
            problems.append(f"{var} served {out['env'].get(var)!r}, want {want_o!r} for {kind}")
    used = {n: int(c.get(n) or 0) for n in ("obs_drain_degraded", "obs_drain_exact", "obs_exec_degraded", "obs_inflight_degraded", "obs_partner_hidden", "obs_partner_predicted", "obs_cold_degraded", "obs_cold_exact")}
    tail_calls = int(c.get("pg_cold_tail_calls") or 0)
    if bool(obs_want.get("HEROSIM_PG_COLD_TAIL")) != bool(tail_calls):
        problems.append(f"cold-tail instrument: HEROSIM_PG_COLD_TAIL={obs_want.get('HEROSIM_PG_COLD_TAIL')!r} but pg_cold_tail_calls {tail_calls}")
    if obs_want.get("HEROSIM_OBS_WARM") and (not used["obs_cold_degraded"] or used["obs_cold_exact"]):
        problems.append(f"warmth channel: degraded cold prices {used['obs_cold_degraded']}, exact {used['obs_cold_exact']}")
    if not obs_want.get("HEROSIM_OBS_WARM") and used["obs_cold_degraded"]:
        problems.append("cold seconds degraded on an arm that does not degrade them")
    if not obs_want and any(used.values()):
        problems.append(f"an exact-observation arm used degraded channels: {used}")
    if obs_want:
        if obs_want.get("HEROSIM_OBS_QUEUE") and (not used["obs_drain_degraded"] or used["obs_drain_exact"]):
            problems.append(f"queue channel: degraded drains {used['obs_drain_degraded']}, exact walks {used['obs_drain_exact']}")
        if not obs_want.get("HEROSIM_OBS_QUEUE") and used["obs_drain_degraded"]:
            problems.append("queue degraded on an arm that does not degrade it")
        if obs_want.get("HEROSIM_OBS_EXEC_SIGMA") and not used["obs_exec_degraded"]:
            problems.append("exec channel off: obs_exec_degraded == 0")
        if obs_want.get("HEROSIM_OBS_PARTNERS") and not used["obs_partner_hidden"]:
            problems.append("partner channel off: obs_partner_hidden == 0")
        if obs_want.get("HEROSIM_OBS_ESTIMATOR") == "ec" and obs_want.get("HEROSIM_OBS_PARTNERS") and not used["obs_partner_predicted"]:
            problems.append("ec partner prediction off: obs_partner_predicted == 0")
        if not obs_want.get("HEROSIM_OBS_INFLIGHT") and used["obs_inflight_degraded"]:
            problems.append("in-flight degraded on an arm that does not degrade it")
    for var in ("HEROSIM_PG_DRAIN_ALPHA", "HEROSIM_PG_BATCH_ALPHA"):
        want = EXACT_ALPHA.get(kind, ib0_alpha(orig_kind)).get(var)
        if out["env"].get(var) != want:
            problems.append(f"{var} served {out['env'].get(var)!r}, want {want!r} for {kind}")
    # decision_cost_s0_v1: the cost served is the cost named, the plan counter always runs, the scheduler server's books exist exactly when c > 0
    served_dc = out["env"].get("HEROSIM_DECISION_COST_MS")
    if (float(served_dc) if served_dc else None) != (float(dc_ms) if dc_ms else None):  # dc_ms is k x w for this arm and rung
        problems.append(f"HEROSIM_DECISION_COST_MS served {served_dc!r}, want {dc_ms!r}")
    if int(c.get("sched_plan_evals") or 0) == 0 and int(c.get("pg_batches") or 0) + int(c.get("prefix_batches") or 0) > 0:
        problems.append("plan-evaluation meter off: sched_plan_evals == 0")
    charged = int(c.get("sched_batches_charged") or 0)
    if dc_ms and (charged == 0 or float(c.get("sched_busy_seconds") or 0) <= 0):
        problems.append(f"scheduler server idle at c = {dc_ms}: sched_batches_charged {charged}, sched_busy_seconds {c.get('sched_busy_seconds')}")
    if not dc_ms and (charged or c.get("sched_busy_seconds")):
        problems.append("scheduler server charged time at c = 0")
    exact_arm = kind in ("cd_exactS", "cd_oracle", *EXACT_ALPHA) or kind.endswith(("_cdxexg", "_cdxpra", "_cdxprb"))
    prune_arm = kind.endswith(("_cdxpra", "_cdxprb")) or kind == "cd_exactS_ib0_prune"
    if prune_arm and (out["env"].get("HEROSIM_PG_CD_EXACT") != "1" or int(c.get("pg_exact_batches") or 0) == 0):
        problems.append(f"exact-search instrument off: HEROSIM_PG_CD_EXACT={out['env'].get('HEROSIM_PG_CD_EXACT')!r}, pg_exact_batches {c.get('pg_exact_batches')}")
    want_prune = "gnn" if kind.endswith("_cdxpra") else "cost"
    if prune_arm and (out["env"].get("HEROSIM_PG_CD_EXACT_PRUNE") != want_prune or int(c.get("pg_exact_prune_batches") or 0) == 0):
        problems.append(f"prune instrument off: HEROSIM_PG_CD_EXACT_PRUNE={out['env'].get('HEROSIM_PG_CD_EXACT_PRUNE')!r} (want {want_prune}), pg_exact_prune_batches {c.get('pg_exact_prune_batches')}")
    if prune_arm and int(c.get("pg_exact_prune_batches") or 0) != int(c.get("pg_exact_fallbacks") or 0):
        problems.append(f"pruned {c.get('pg_exact_prune_batches')} batches but {c.get('pg_exact_fallbacks')} were over the cap")
    if not prune_arm and (out["env"].get("HEROSIM_PG_CD_EXACT_PRUNE") or int(c.get("pg_exact_prune_batches") or 0)):
        problems.append("a non-prune arm pruned the slate")
    if kind.endswith("_cdxexg") and (out["env"].get("HEROSIM_PG_CD_EXACT") != "1" or int(c.get("pg_exact_batches") or 0) == 0):
        problems.append(f"exact-search instrument off: HEROSIM_PG_CD_EXACT={out['env'].get('HEROSIM_PG_CD_EXACT')!r}, pg_exact_batches {c.get('pg_exact_batches')}")
    if kind.endswith("_cdxexg") and (out["env"].get("HEROSIM_PG_CD_EXACT_GNN") != "1" or int(c.get("pg_exact_gnn_scored") or 0) == 0):
        problems.append(f"exact-GNN instrument off: HEROSIM_PG_CD_EXACT_GNN={out['env'].get('HEROSIM_PG_CD_EXACT_GNN')!r}, pg_exact_gnn_scored {c.get('pg_exact_gnn_scored')}")
    if not kind.endswith("_cdxexg") and (out["env"].get("HEROSIM_PG_CD_EXACT_GNN") or int(c.get("pg_exact_gnn_scored") or 0)):
        problems.append("a non-exact-GNN arm scored plans with the GNN")
    if not exact_arm and (out["env"].get("HEROSIM_PG_CD_EXACT") or int(c.get("pg_exact_batches") or 0)):
        problems.append("a non-exact arm ran the exact search")
    if kind == "cd_oracle":
        if out["env"].get("HEROSIM_PG_CD_EXACT") != "1" or int(c.get("pg_exact_batches") or 0) == 0:
            problems.append(f"exact-search instrument off: HEROSIM_PG_CD_EXACT={out['env'].get('HEROSIM_PG_CD_EXACT')!r}, pg_exact_batches {c.get('pg_exact_batches')}")
        if out["env"].get("HEROSIM_PG_ORACLE") != "1" or int(c.get("pg_oracle_batches") or 0) == 0:
            problems.append(f"oracle instrument off: HEROSIM_PG_ORACLE={out['env'].get('HEROSIM_PG_ORACLE')!r}, pg_oracle_batches {c.get('pg_oracle_batches')}")
        if int(c.get("pg_oracle_seen_batches") or 0) != int(c.get("pg_oracle_batches") or 0) + int(c.get("pg_oracle_over_cap_batches") or 0) + int(c.get("pg_oracle_unplaced_batches") or 0) + int(c.get("pg_oracle_search_batches") or 0):
            problems.append(f"oracle books do not close: seen {c.get('pg_oracle_seen_batches')} != served {c.get('pg_oracle_batches')} + over cap {c.get('pg_oracle_over_cap_batches')} + unplaced {c.get('pg_oracle_unplaced_batches')} + searched {c.get('pg_oracle_search_batches')}")
        want_search = orig_kind == "cd_oracle_ib0"
        if (out["env"].get("HEROSIM_PG_ORACLE_SEARCH") == "1") != want_search or bool(int(c.get("pg_oracle_search_batches") or 0)) != want_search:
            problems.append(f"over-cap search: HEROSIM_PG_ORACLE_SEARCH={out['env'].get('HEROSIM_PG_ORACLE_SEARCH')!r}, searched {c.get('pg_oracle_search_batches')}, want {want_search}")
    elif out["env"].get("HEROSIM_PG_ORACLE") or int(c.get("pg_oracle_batches") or 0):
        problems.append("a non-oracle arm ran the live oracle")
    if kind == "cd_random_seed":
        if int(c.get("pg_random_seed_batches") or 0) == 0 or int(c.get("cdr_batches") or 0) != int(c.get("pg_random_seed_batches") or 0):
            problems.append(f"random-seed instrument off: seeded {c.get('pg_random_seed_batches')} batches, refined {c.get('cdr_batches')}")
        if out["env"].get("HEROSIM_CD_RANDOM_SEED") != str(seed):
            problems.append(f"served HEROSIM_CD_RANDOM_SEED={out['env'].get('HEROSIM_CD_RANDOM_SEED')!r}, cell seed {seed}")
    elif int(c.get("pg_random_seed_batches") or 0):
        problems.append("a non-random-seed arm drew a random seed plan")
    if kind.endswith("_slate") and int(c.get("slate_batches") or 0) == 0:
        problems.append("slate instrument off: slate_batches == 0")
    if not kind.endswith("_slate") and int(c.get("slate_batches") or 0):
        problems.append("unslated arm was slated")
    if RULE_POLICY.get(kind) == "decima_wfair_network":
        if int(c.get("decima_batches") or 0) == 0:
            problems.append("decima instrument off: decima_batches == 0")
        if not out["env"].get("HEROSIM_DECIMA_ALPHA"):
            problems.append("served without HEROSIM_DECIMA_ALPHA in provenance")
    if kind in ("batched", "locality", "cd", "cd_blind", "cd_slate", "cd_inflight", "cd_pull", "cd_ledger", "cd_declared", *EXT_KINDS) and int(c.get("pg_batches") or 0) == 0:
        problems.append("decoded no batches")
    # cd_declared: the slate must have been applied to every batch, and nowhere else
    declared = int(c.get("pg_declared_batches") or 0)
    if kind == "cd_declared":
        if out["env"].get("GNN_SERVE_CANDIDATE_SLATE") != "declared_pruning_v1" or declared == 0 or int(c.get("pg_declared_groups") or 0) < declared:
            problems.append("declared-slate instrument off: GNN_SERVE_CANDIDATE_SLATE not served or no batch went through declared_slate")
    elif declared and not kind.startswith("ra_") and kind != "cd_random_seed":
        problems.append("a non-declared arm applied the declared slate")
    want_scale = repr(LOCALITY_SCALE) if kind == "locality" else None
    if out["env"].get("HEROSIM_PG_EXCHANGE_SCALE") != want_scale:
        problems.append(f"served HEROSIM_PG_EXCHANGE_SCALE={out['env'].get('HEROSIM_PG_EXCHANGE_SCALE')!r}, {kind} needs {want_scale!r}")
    if kind in EXT_KINDS:
        want_rate = repr(EXT_LABEL_RATE if kind == "cdext" else EXT_LABEL_RATE / rate_scale)
        if out["env"].get("HEROSIM_PG_EXT_RATE") != want_rate:
            problems.append(f"served HEROSIM_PG_EXT_RATE={out['env'].get('HEROSIM_PG_EXT_RATE')!r}, {kind} needs {want_rate}")
        if int(c.get("pg_ext_batches") or 0) != int(c.get("pg_batches") or 0) or int(c.get("pg_ext_charged") or 0) == 0:
            problems.append(f"externality instrument off: {c.get('pg_ext_batches')}/{c.get('pg_batches')} batches, "
                            f"{c.get('pg_ext_charged')} charged")
    elif out["env"].get("HEROSIM_PG_EXT_RATE") or int(c.get("pg_ext_batches") or 0):
        problems.append("a non-externality arm charged the externality")
    if kind == "cd_blind" and int(c.get("pg_partners_blinded") or 0) == 0:
        problems.append("blind instrument off: pg_partners_blinded == 0")
    if kind == "cd" and int(c.get("pg_partners_blinded") or 0) != 0:
        problems.append("CD ran blind")
    if kind == "cd_inflight" and (out["env"].get("HEROSIM_PG_INFLIGHT") != "1" or int(c.get("pg_inflight_charged") or 0) == 0):
        problems.append("inflight instrument off: HEROSIM_PG_INFLIGHT not served or pg_inflight_charged == 0")
    if kind != "cd_inflight" and (out["env"].get("HEROSIM_PG_INFLIGHT") or int(c.get("pg_inflight_charged") or 0)):
        problems.append("a non-inflight arm charged the in-flight task")
    if kind == "cd_pull" and (out["env"].get("HEROSIM_PG_PULL_HOLD") != "1" or out["env"].get("HEROSIM_PULL_LEDGER") != "1"
                              or int(c.get("pg_pull_charged") or 0) == 0):
        problems.append("pull-hold instrument off: HEROSIM_PG_PULL_HOLD / HEROSIM_PULL_LEDGER not served or pg_pull_charged == 0")
    unsplit, sub_b = int(c.get("slate_declared_unsplit") or 0), int(c.get("slate_declared_sub_batched") or 0)
    if kind.endswith(("_nosplit", "_cdxapply", "_cdxexg", "_cdxpra", "_cdxprb", "_selfsearch")) and (out["env"].get("GNN_SLATE_NO_SPLIT") != "1" or unsplit != sub_b):
        problems.append(f"no-split instrument off: GNN_SLATE_NO_SPLIT={out['env'].get('GNN_SLATE_NO_SPLIT')!r}, unsplit {unsplit} of {sub_b} sub-batched")
    if not kind.endswith(("_nosplit", "_cdxapply", "_cdxexg", "_cdxpra", "_cdxprb", "_selfsearch")) and (out["env"].get("GNN_SLATE_NO_SPLIT") or unsplit):
        problems.append("a split arm decoded a batch whole")
    ss = int(c.get("ss_batches") or 0)
    if kind.endswith("_selfsearch") and (out["env"].get("GNN_SELF_SEARCH") != "1" or ss == 0 or ss != int(c.get("slate_declared_batches") or -1)):
        problems.append(f"self-search instrument off: GNN_SELF_SEARCH={out['env'].get('GNN_SELF_SEARCH')!r}, searched {ss} of {c.get('slate_declared_batches')} batches")
    if not kind.endswith("_selfsearch") and (out["env"].get("GNN_SELF_SEARCH") or ss):
        problems.append("a non-self-search arm ran the plan search")
    if kind != "cd_pull" and (out["env"].get("HEROSIM_PG_PULL_HOLD") or int(c.get("pg_pull_charged") or 0)):
        problems.append("a non-pull arm charged the pull hold")
    if (out["env"].get("HEROSIM_PULL_LEDGER") == "1") != (kind in ("cd_pull", "cd_ledger")):
        problems.append(f"HEROSIM_PULL_LEDGER={out['env'].get('HEROSIM_PULL_LEDGER')!r} served for {kind}")
    if kind.endswith("_spread") and int(c.get("prefix_sibling_moves") or 0) == 0:
        problems.append("spread instrument off: prefix_sibling_moves == 0")
    want_capture = SERVICE_END if (base_kind in BC1_KINDS or kind.endswith("_se")) else None
    if out["env"].get("HEROSIM_INFLIGHT_CAPTURE") != want_capture:
        problems.append(f"served HEROSIM_INFLIGHT_CAPTURE={out['env'].get('HEROSIM_INFLIGHT_CAPTURE')!r}, "
                        f"{kind} needs {want_capture!r}")
    if base_kind in LOAD_KINDS:
        want_ls = "0" if base_kind == "v4twin" else "1"
        if out["env"].get("PARTIAL_STATE_LOAD_SECONDS", "") != want_ls:
            problems.append(f"served PARTIAL_STATE_LOAD_SECONDS={out['env'].get('PARTIAL_STATE_LOAD_SECONDS')!r}, "
                            f"{base_kind} needs {want_ls}")
        want_xs = "1" if base_kind in ("xs1load", "xs1mpoff") + RAW_KINDS + SB1_KINDS + LF1_KINDS + AGG_KINDS + HET_KINDS + SO1_KINDS else "0"
        if out["env"].get("PARTIAL_STATE_EXCHANGE_SECONDS", "") != want_xs:
            problems.append(f"served PARTIAL_STATE_EXCHANGE_SECONDS={out['env'].get('PARTIAL_STATE_EXCHANGE_SECONDS')!r}, "
                            f"{base_kind} needs {want_xs}")
        if int(c.get("v4_backlog_batches") or 0) == 0:
            problems.append("v4 instrument off: v4_backlog_batches == 0")
        if base_kind != "v4twin" and int(c.get("v4_backlog_nonzero") or 0) == 0:
            problems.append("v4 instrument off: no candidate ever had a backlog")
    elif base_kind in RA_KINDS:
        for key, want in (("PARTIAL_STATE_CONTRACT", "partial_state_v5"), ("PARTIAL_STATE_LOAD_SECONDS", "1"),
                          ("PARTIAL_STATE_EXCHANGE_SECONDS", "1"), ("GNN_SERVE_CANDIDATE_SLATE", r1a_slate())):
            if out["env"].get(key, "") != want:
                problems.append(f"served {key}={out['env'].get(key)!r}, {base_kind} needs {want!r}")
    elif int(c.get("v4_backlog_batches") or 0):
        problems.append("a pre-v4 arm computed v4 backlogs")
    if base_kind in LEARNED_KINDS and not kind.endswith("_spread") and int(c.get("prefix_sibling_moves") or 0):
        problems.append("unspread arm moved siblings")
    want_seed = "hand" if orig_kind.endswith("_cdxhib0") else None
    if out["env"].get("GNN_CD_SEED") != want_seed:
        problems.append(f"GNN_CD_SEED served {out['env'].get('GNN_CD_SEED')!r}, want {want_seed!r} for {orig_kind}")
    kind = orig_kind
    if problems:
        json.dump({"arm": name, "why": "; ".join(problems), "wallclock_s": wall}, open(failed, "w"))
        os.remove(raw)
        return f"[FAILED {problems}] {name}"
    e, q = float(out["averageElapsedTime"]), float(out["averageQueueTime"])
    out["queue_share"] = q / e if e > 0 else None
    out["effective_queue_share"] = (out["lock_wait"] or {}).get("effective_queue_share")
    out["task_times_rows"] = write_task_times(st.get("taskResults"), os.path.join(out_dir, name + ".tasks.npy"))
    sealed = sealed_triggers(int(t["topo"]), window)
    if sealed is not None:
        snap = os.path.join(out_dir, name + ".snapshots.jsonl")
        got = [json.loads(line)["trigger_task_id"] for line in open(snap)] if os.path.exists(snap) else []
        out["sealed_snapshots"] = {"wanted": len(sealed), "written": len(got), "missing": sorted(set(sealed) - set(got))}
    json.dump(out, open(summary + ".partial", "w"), indent=1)
    os.replace(summary + ".partial", summary)
    keep = os.environ.get("KEEP_RAW_DIR")
    if keep:
        # regime decomposition (accel_replica_v1 idea 1): the per-task result outlives the summary, on /share, never on /home
        os.makedirs(keep, exist_ok=True)
        shutil.move(raw, os.path.join(keep, os.path.basename(raw)))
    else:
        os.remove(raw)
    return f"[done {wall}s] {name} elapsed={e:.3f} share={out['queue_share']:.3f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=("screen", "rp2screen", "parity", "gate", "d1", "d2", "d4", "d5", "d6", "a", "v4", "fix", "bc1", "bc1selfref", "fc1", "xs1", "xs1cd", "ladder", "jitsmoke", "ladderjit", "capacity", "guard", "decimatune", "decima", "x11confirm", "grounded", "x15fill", "groundedx15", "groundedladder", "peakctl", "peakmlp", "rawplan", "rawgnn", "rawmlp", "w0mlp", "rp2dev", "rp2conf", "sb1dev", "sbconf", "scale", "lf1conf", "rb1", "agg1", "het1", "so1", "cl1", "tp1", "wf1", "wf1cal", "r1a") + RAW_V2)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--selection", default=None)
    ap.add_argument("--parallel", type=int, default=12)
    ap.add_argument("--mem", default="4G")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--no-scope", action="store_true", help="no per-run systemd scope (SLURM nodes)")
    a = ap.parse_args()
    global NO_SCOPE, R1A_ON
    NO_SCOPE = a.no_scope
    R1A_ON = a.phase == "r1a"
    selection = None
    if a.phase in ("gate", "d1", "d2", "d4", "d5", "d6", "a", "v4", "fix", "bc1", "bc1selfref", "fc1", "xs1", "xs1cd", "ladder", "jitsmoke", "ladderjit", "capacity", "guard", "decimatune", "decima", "x11confirm", "grounded", "x15fill", "groundedx15", "groundedladder", "peakctl", "peakmlp", "rawplan", "rawgnn", "rawmlp", "w0mlp", "rp2dev", "rp2conf", "sb1dev", "sbconf", "scale", "lf1conf", "rb1", "agg1", "het1", "so1", "cl1", "tp1", "wf1", "r1a") + RAW_V2:
        selection = json.load(open(a.selection))
        if selection.get("verdict") != "DESIGN-READY":
            raise SystemExit(f"FAIL LOUD: selection verdict {selection.get('verdict')!r}")
    os.makedirs(a.out, exist_ok=True)
    todo = tasks_for(a.phase, selection)
    print(f"[{a.phase}] {len(todo)} tasks, {a.parallel} parallel, MemoryMax={a.mem}, timeout {a.timeout}s", flush=True)
    with ThreadPoolExecutor(max_workers=a.parallel) as pool:
        for line in pool.map(lambda t: run_one(t, a.inputs, a.out, a.mem, a.timeout), todo):
            print(line, flush=True)
    done = sum(1 for t in todo if os.path.exists(os.path.join(a.out, arm_name(t) + ".summary.json")))
    print(f"=== {a.phase}: {done}/{len(todo)} summaries ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
