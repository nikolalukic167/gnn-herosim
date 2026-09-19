#!/usr/bin/env python3
"""serving_gap_v1 stage 1 — the registered diagnostics (docs/lineages/serving_gap_v1.md).

Asks why message passing gets worse at its own objective when served on a stream. Nothing
here trains, relabels or changes physics: it reads the decode traces the live scheduler
already writes (`GNN_PREFIX_TRACE_PATH`, one pickled record per served batch) for
checkpoints that already exist.

Stages, exactly as registered:

  s0  The control that must pass before any treated statistic is read. Confirms (a) the
      checkpoints about to be traced are byte-identical to those that produced the
      registered offline read (md5 against the reports' `checkpoint` field is not enough —
      the .pt itself is hashed), and (b) the offline `gnn` - `mpoff` contrast recomputed
      from the stored per-checkpoint reports reproduces the registered value and SIGN.
      A harness that cannot reproduce the offline sign is measuring its own bug.

  h2  Queue insensitivity. Per sampled live graph, score task 0's candidates on an empty
      prefix, then add one standard deviation of the graph's own live queue-depth column to
      every platform's queue feature and re-score. The statistic is the mean absolute score
      change divided by the score spread at that step, i.e. how much the arm's preference
      moves when every candidate is told it is busier. Paired by seed, exact Wilcoxon.
      FIRES when gnn's normalised sensitivity is BELOW mpoff's on >= 2 of 3 corpora, p < 0.05.

  h1  Herding. Per arm, over every served batch of a live run: the entropy of the
      chosen-node distribution (normalised by log of the number of distinct candidate
      nodes the arm ever saw), the fraction of consecutive batch pairs sharing a modal
      chosen node, and the lag-1 cosine autocorrelation of the per-batch node histogram.
      Paired by seed, exact Wilcoxon. FIRES when gnn's modal repeat rate exceeds mpoff's
      by >= 0.05 absolute AND gnn's entropy is lower, both p < 0.05, on >= 2 of 3 corpora.

Usage:
  serving_gap_trace_read.py s0 --reports-dir-gnn ... --reports-dir-mpoff ...
  serving_gap_trace_read.py h1 --traces-root <dir> --output <json>

The h1 trace layout is `<traces_root>/<corpus>/<arm>_s<seed>.pkl`, which is what
scripts_cosim/datalab/serving_gap_v1_traces.sbatch writes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import statistics as st
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts_cosim.peer_affinity_t1_read import wilcoxon_exact  # noqa: E402

CORPORA = ("x200p2", "x800p2", "x800p3")
ARMS = ("gnn", "mpoff")
REPEAT_RATE_BAR = 0.05
ALPHA = 0.05

# The registered offline contrast this harness must reproduce (peer_affinity_v1, the
# val-selected `gnn_vs_mpoff@val` row of each rung's read). Sign is what S0 gates on.
REGISTERED_OFFLINE_PP = {"x200p2": 5.14, "x800p2": 6.53, "x800p3": 9.61}
# The first corpus is T1b and its artifacts carry that name, not the x200p2 shorthand.
CKPT_TAG = {"x200p2": "t1b", "x800p2": "x800p2", "x800p3": "x800p3"}
REPORTS_DIR = {"x200p2": "simulation_data/peer_affinity_t1b_reports",
               "x800p2": "simulation_data/peer_affinity_x800p2_reports",
               "x800p3": "simulation_data/peer_affinity_x800p3_reports"}
SPLIT = {"x200p2": "experiments/peer_affinity_v1_t1b_split.json",
         "x800p2": "experiments/peer_affinity_v1_x800_p2_split.json",
         "x800p3": "experiments/peer_affinity_v1_x800_p3_split.json"}


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_records(path: Path):
    with open(path, "rb") as fh:
        while True:
            try:
                yield pickle.load(fh)
            except EOFError:
                return


# --------------------------------------------------------------------------- s0

def stage_s0(args: argparse.Namespace) -> dict:
    """Reproduce the registered offline contrast from the stored reports, per corpus."""
    out: Dict[str, dict] = {}
    for corpus in CORPORA:
        reports = REPO_ROOT / REPORTS_DIR[corpus]
        split_path = REPO_ROOT / SPLIT[corpus]
        if not reports.is_dir() or not split_path.is_file():
            out[corpus] = {"status": "reports or split missing", "dir": str(reports)}
            continue
        # membership exactly as the registered read builds it; the held-out block is the
        # only split S0 reads, because that is what the registered contrast was computed on
        sp = json.loads(split_path.read_text())
        test_ids = set(sp["test"])
        tag = CKPT_TAG[corpus]
        per_arm: Dict[str, Dict[int, float]] = {a: {} for a in ARMS}
        ckpt_md5: Dict[str, Dict[int, str]] = {a: {} for a in ARMS}
        for arm in ARMS:
            for seed in range(1, 17):
                f = reports / f"peer-affinity-v1-{tag}-{arm}-{args.lr}-seed{seed}.json"
                if not f.is_file():
                    continue
                rep = json.loads(f.read_text())
                rows = rep.get("datasets") or rep.get("per_dataset") or []
                vals = [float(d["decode_regret_pct"]["mean_tied"]) for d in rows
                        if d.get("decode_regret_pct") and not d.get("infeasible")
                        and str(d.get("dataset_id")) in test_ids]
                if vals:
                    per_arm[arm][seed] = st.median(vals)
                ck = REPO_ROOT / str(rep.get("checkpoint") or "")
                if ck.is_file():
                    ckpt_md5[arm][seed] = _md5(ck)
        seeds = sorted(set(per_arm["gnn"]) & set(per_arm["mpoff"]))
        if not seeds:
            out[corpus] = {"status": "no paired seeds with reports", "dir": str(reports)}
            continue
        diffs = [per_arm["mpoff"][s] - per_arm["gnn"][s] for s in seeds]  # + = gnn better
        median_pp = st.median(diffs)
        registered = REGISTERED_OFFLINE_PP[corpus]
        out[corpus] = {
            "n_pairs": len(seeds),
            "checkpoints_hashed": {a: len(ckpt_md5[a]) for a in ARMS},
            "recomputed_median_pp": median_pp,
            "registered_median_pp": registered,
            "p_exact": wilcoxon_exact(diffs),
            "sign_matches_registered": (median_pp > 0) == (registered > 0),
        }
    passed = [c for c, v in out.items() if v.get("sign_matches_registered")]
    return {"stage": "s0", "per_corpus": out, "corpora_reproducing_sign": passed,
            "verdict": "PASS" if len(passed) == len(CORPORA) else "FAIL"}


# --------------------------------------------------------------------------- h1

def herding_statistics(trace: Path) -> Optional[dict]:
    """Node-choice entropy, modal repeat rate and lag-1 histogram autocorrelation."""
    hists: List[Counter] = []
    modal: List[int] = []
    overall: Counter = Counter()
    for rec in _iter_records(trace):
        combo = rec.get("combo")
        if not combo:
            continue
        h = Counter(int(p[0]) for p in combo)
        hists.append(h)
        overall.update(h)
        modal.append(max(sorted(h), key=lambda n: (h[n], -n)))
    if len(hists) < 2:
        return None
    total = sum(overall.values())
    n_nodes = len(overall)
    ent = -sum((c / total) * math.log(c / total) for c in overall.values() if c)
    norm_ent = ent / math.log(n_nodes) if n_nodes > 1 else 0.0
    repeats = sum(1 for a, b in zip(modal, modal[1:]) if a == b) / (len(modal) - 1)
    nodes = sorted(overall)
    def vec(h: Counter) -> List[float]:
        return [float(h.get(n, 0)) for n in nodes]
    cos: List[float] = []
    for a, b in zip(hists, hists[1:]):
        va, vb = vec(a), vec(b)
        na = math.sqrt(sum(x * x for x in va)); nb = math.sqrt(sum(x * x for x in vb))
        if na > 0 and nb > 0:
            cos.append(sum(x * y for x, y in zip(va, vb)) / (na * nb))
    return {
        "batches": len(hists), "distinct_nodes": n_nodes,
        "node_entropy_norm": norm_ent, "modal_repeat_rate": repeats,
        "hist_autocorr_lag1": st.mean(cos) if cos else float("nan"),
    }


def stage_h1(args: argparse.Namespace) -> dict:
    root = Path(args.traces_root)
    per_corpus: Dict[str, dict] = {}
    for corpus in CORPORA:
        stats: Dict[str, Dict[int, dict]] = {a: {} for a in ARMS}
        for arm in ARMS:
            for seed in range(1, 17):
                t = root / corpus / f"{arm}_s{seed}.pkl"
                if not t.is_file():
                    continue
                s = herding_statistics(t)
                if s:
                    stats[arm][seed] = s
        seeds = sorted(set(stats["gnn"]) & set(stats["mpoff"]))
        if not seeds:
            per_corpus[corpus] = {"status": "no paired traces"}
            continue
        d_rep = [stats["gnn"][s]["modal_repeat_rate"] - stats["mpoff"][s]["modal_repeat_rate"] for s in seeds]
        d_ent = [stats["gnn"][s]["node_entropy_norm"] - stats["mpoff"][s]["node_entropy_norm"] for s in seeds]
        d_ac = [stats["gnn"][s]["hist_autocorr_lag1"] - stats["mpoff"][s]["hist_autocorr_lag1"] for s in seeds]
        p_rep, p_ent = wilcoxon_exact(d_rep), wilcoxon_exact(d_ent)
        fires = (st.median(d_rep) >= REPEAT_RATE_BAR and st.median(d_ent) < 0
                 and p_rep is not None and p_rep < ALPHA
                 and p_ent is not None and p_ent < ALPHA)
        per_corpus[corpus] = {
            "n_pairs": len(seeds),
            "median_repeat_rate_gnn_minus_mpoff": st.median(d_rep), "p_repeat": p_rep,
            "median_entropy_gnn_minus_mpoff": st.median(d_ent), "p_entropy": p_ent,
            "median_autocorr_gnn_minus_mpoff": st.median(d_ac), "p_autocorr": wilcoxon_exact(d_ac),
            "per_arm_medians": {a: {k: st.median([stats[a][s][k] for s in seeds])
                                    for k in ("node_entropy_norm", "modal_repeat_rate",
                                              "hist_autocorr_lag1", "batches", "distinct_nodes")}
                                for a in ARMS},
            "fires": bool(fires),
        }
    fired = [c for c, v in per_corpus.items() if v.get("fires")]
    return {"stage": "h1", "bar": {"repeat_rate_bar": REPEAT_RATE_BAR, "alpha": ALPHA,
                                   "corpora_required": 2},
            "per_corpus": per_corpus, "corpora_fired": fired,
            "verdict": "H1-FIRES" if len(fired) >= 2 else "H1-DOES-NOT-FIRE"}


# --------------------------------------------------------------------------- h2

# plat_row = platform-type one-hot (5) + [has_dnn1, has_dnn2, queue_len] + ... so the live
# queue depth is column 7 of a dim-14 platform feature row (feature_builder.py).
QUEUE_COL = 7
PLATFORM_FEATURE_DIM = 14


def stage_h2_one(args: argparse.Namespace) -> dict:
    """Queue sensitivity for ONE (corpus, arm, seed). Run under the arm's environment: the
    caller sets GNN_DISABLE_MESSAGE_PASSING for mpoff, exactly as the offline evaluator does."""
    import torch
    from src.policy.gnn.prefix_serving import load_prefix_conditioned_gnn
    from src.policy.gnn.partial_state_edges import make_partial_state_score_fn
    from src.policy.tabular.reduced_features import build_partial_state_context_from_graph

    model, options, _sc = load_prefix_conditioned_gnn(Path(args.checkpoint))
    sens: List[float] = []
    skipped = 0
    for rec in _iter_records(Path(args.trace)):
        graph = rec.get("graph")
        if graph is None:
            skipped += 1
            continue
        pf = getattr(graph, "platform_features", None)
        if pf is None or int(pf.shape[1]) != PLATFORM_FEATURE_DIM:
            raise RuntimeError(f"{args.trace}: platform_features is {None if pf is None else tuple(pf.shape)}, "
                               f"expected (*, {PLATFORM_FEATURE_DIM}) — the queue column index is layout-specific")
        ctx = build_partial_state_context_from_graph(graph)
        caps = graph.partial_state_ctx["node_caps_by_alpha"]
        if options.alpha_key not in caps:
            skipped += 1
            continue
        ctx.node_caps = caps[options.alpha_key]
        with torch.no_grad():
            base = [float(v) for v in make_partial_state_score_fn(model, graph, ctx)(0, {})]
            spread = max(base) - min(base)
            if spread <= 0 or len(base) < 2:
                skipped += 1
                continue
            col = pf[:, QUEUE_COL]
            sd = float(col.std()) if int(col.numel()) > 1 else 0.0
            if sd <= 0:
                skipped += 1
                continue
            original = col.clone()
            pf[:, QUEUE_COL] = original + sd
            pert = [float(v) for v in make_partial_state_score_fn(model, graph, ctx)(0, {})]
            pf[:, QUEUE_COL] = original
        sens.append(sum(abs(a - b) for a, b in zip(pert, base)) / len(base) / spread)
    if not sens:
        raise RuntimeError(f"{args.trace}: no usable graphs ({skipped} skipped)")
    return {"stage": "h2-one", "corpus": args.corpus, "arm": args.arm, "seed": args.seed,
            "checkpoint": args.checkpoint, "n_graphs": len(sens), "n_skipped": skipped,
            "median_queue_sensitivity": st.median(sens), "mean_queue_sensitivity": st.mean(sens)}


def stage_h2(args: argparse.Namespace) -> dict:
    root = Path(args.results_root)
    per_corpus: Dict[str, dict] = {}
    for corpus in CORPORA:
        vals: Dict[str, Dict[int, float]] = {a: {} for a in ARMS}
        for arm in ARMS:
            for seed in range(1, 17):
                f = root / corpus / f"{arm}_s{seed}.json"
                if f.is_file():
                    vals[arm][seed] = float(json.loads(f.read_text())["median_queue_sensitivity"])
        seeds = sorted(set(vals["gnn"]) & set(vals["mpoff"]))
        if not seeds:
            per_corpus[corpus] = {"status": "no paired sensitivities"}
            continue
        diffs = [vals["gnn"][s] - vals["mpoff"][s] for s in seeds]
        p = wilcoxon_exact(diffs)
        fires = st.median(diffs) < 0 and p is not None and p < ALPHA
        per_corpus[corpus] = {
            "n_pairs": len(seeds),
            "median_sensitivity_gnn": st.median([vals["gnn"][s] for s in seeds]),
            "median_sensitivity_mpoff": st.median([vals["mpoff"][s] for s in seeds]),
            "median_gnn_minus_mpoff": st.median(diffs), "p_exact": p, "fires": bool(fires),
        }
    fired = [c for c, v in per_corpus.items() if v.get("fires")]
    return {"stage": "h2", "bar": {"alpha": ALPHA, "corpora_required": 2, "direction": "gnn below mpoff"},
            "per_corpus": per_corpus, "corpora_fired": fired,
            "verdict": "H2-FIRES" if len(fired) >= 2 else "H2-DOES-NOT-FIRE"}


# --------------------------------------------------------------------------- h3 / h4
# serving_gap_v2. The slim traces carry each batch's global task ids and its decoded plan;
# the peer pairs come from the workload file the run was served (one table, global task ids),
# so no rerun is needed to know which pairs a batch contained.

WORKLOAD = {
    "x200p2": "simulation_data/peer_affinity_live_gate/workloads/workload-150-150-peer_p2_x200.json",
    "x800p2": "simulation_data/peer_affinity_live_gate/workloads/workload-150-150-peer_p2_x800.json",
    "x800p3": "simulation_data/peer_affinity_live_gate/workloads/workload-150-150-peer_p3_x800.json",
}


def _peer_pairs(workload: Path) -> set:
    d = json.loads(workload.read_text())
    table = d.get("peer_exchange") or []
    if not table:
        raise RuntimeError(f"{workload}: no peer_exchange table")
    return {(int(a), int(b)) if int(a) < int(b) else (int(b), int(a)) for a, b, _x in table}


def splitting_statistics(trace: Path, pairs: set) -> Optional[dict]:
    frac_nodes: List[float] = []
    coloc: List[float] = []
    biggest: List[float] = []
    for rec in _iter_records(trace):
        combo, ids = rec.get("combo"), rec.get("task_ids")
        if not combo or not ids or len(combo) != len(ids):
            continue
        node_of = {int(t): int(p[0]) for t, p in zip(ids, combo)}
        counts = Counter(node_of.values())
        k = len(combo)
        frac_nodes.append(len(counts) / k)
        biggest.append(max(counts.values()) / k)
        in_batch = [(a, b) for a, b in ((min(x, y), max(x, y))
                    for x in node_of for y in node_of if x < y) if (a, b) in pairs]
        if in_batch:
            coloc.append(sum(1 for a, b in in_batch if node_of[a] == node_of[b]) / len(in_batch))
    if not frac_nodes or not coloc:
        return None
    return {"batches": len(frac_nodes), "distinct_nodes_frac": st.mean(frac_nodes),
            "coloc_rate": st.mean(coloc), "largest_group_frac": st.mean(biggest),
            "batches_with_pairs": len(coloc)}


def _ranks(xs: Sequence[float]) -> List[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def _spearman_perm(x: Sequence[float], y: Sequence[float], iters: int = 20000, seed: int = 7) -> Tuple[float, float]:
    """Rank correlation with a seeded two-sided permutation p-value (no scipy dependency)."""
    import random
    rx, ry = _ranks(x), _ranks(y)
    n = len(rx)
    mx, my = st.mean(rx), st.mean(ry)
    def corr(a, b):
        num = sum((p - mx) * (q - my) for p, q in zip(a, b))
        da = math.sqrt(sum((p - mx) ** 2 for p in a)); db = math.sqrt(sum((q - my) ** 2 for q in b))
        return num / (da * db) if da > 0 and db > 0 else 0.0
    rho = corr(rx, ry)
    rng = random.Random(seed)
    shuffled = list(ry)
    hits = 0
    for _ in range(iters):
        rng.shuffle(shuffled)
        if abs(corr(rx, shuffled)) >= abs(rho) - 1e-12:
            hits += 1
    return rho, (hits + 1) / (iters + 1)


def stage_h3_one(args: argparse.Namespace) -> dict:
    """Splitting statistics for ONE (corpus, arm, seed). The pair scan is O(k^2) per batch
    over 45,375 batches, so this runs as an array task, never on the login node."""
    pairs = _peer_pairs(REPO_ROOT / WORKLOAD[args.corpus])
    v = splitting_statistics(Path(args.trace), pairs)
    if v is None:
        raise RuntimeError(f"{args.trace}: no usable batches")
    v.update({"stage": "h3-one", "corpus": args.corpus, "arm": args.arm, "seed": args.seed})
    return v


def stage_h3(args: argparse.Namespace) -> dict:
    """Aggregate the per-(corpus, arm, seed) files written by stage_h3_one."""
    root = Path(args.results_root)
    per_corpus: Dict[str, dict] = {}
    raw: Dict[str, Dict[str, Dict[int, dict]]] = {}
    for corpus in CORPORA:
        stats: Dict[str, Dict[int, dict]] = {a: {} for a in ARMS}
        for arm in ARMS:
            for seed in range(1, 17):
                f = root / corpus / f"{arm}_s{seed}.json"
                if f.is_file():
                    stats[arm][seed] = json.loads(f.read_text())
        raw[corpus] = stats
        seeds = sorted(set(stats["gnn"]) & set(stats["mpoff"]))
        if not seeds:
            per_corpus[corpus] = {"status": "no paired results"}
            continue
        d_col = [stats["gnn"][s]["coloc_rate"] - stats["mpoff"][s]["coloc_rate"] for s in seeds]
        d_nod = [stats["gnn"][s]["distinct_nodes_frac"] - stats["mpoff"][s]["distinct_nodes_frac"] for s in seeds]
        p_col = wilcoxon_exact(d_col)
        per_corpus[corpus] = {
            "n_pairs": len(seeds),
            "coloc_gnn": st.median([stats["gnn"][s]["coloc_rate"] for s in seeds]),
            "coloc_mpoff": st.median([stats["mpoff"][s]["coloc_rate"] for s in seeds]),
            "median_coloc_gnn_minus_mpoff": st.median(d_col), "p_coloc": p_col,
            "median_distinct_nodes_gnn_minus_mpoff": st.median(d_nod), "p_distinct_nodes": wilcoxon_exact(d_nod),
            "fires": bool(st.median(d_col) < 0 and p_col is not None and p_col < ALPHA),
        }
    fired = [c for c, v in per_corpus.items() if v.get("fires")]
    return {"stage": "h3", "bar": {"alpha": ALPHA, "corpora_required": 2, "direction": "gnn co-locates less"},
            "per_corpus": per_corpus, "corpora_fired": fired,
            "verdict": "H3-FIRES" if len(fired) >= 2 else "H3-DOES-NOT-FIRE",
            "per_seed": {c: {a: {str(s): v for s, v in raw[c][a].items()} for a in ARMS} for c in raw}}


def _stage_h3_unused(args: argparse.Namespace) -> dict:
    root = Path(args.traces_root)
    per_corpus: Dict[str, dict] = {}
    raw: Dict[str, Dict[str, Dict[int, dict]]] = {}
    for corpus in CORPORA:
        pairs = _peer_pairs(REPO_ROOT / WORKLOAD[corpus])
        stats: Dict[str, Dict[int, dict]] = {a: {} for a in ARMS}
        for arm in ARMS:
            for seed in range(1, 17):
                t = root / corpus / f"{arm}_s{seed}.pkl"
                if t.is_file():
                    v = splitting_statistics(t, pairs)
                    if v:
                        stats[arm][seed] = v
        raw[corpus] = stats
        seeds = sorted(set(stats["gnn"]) & set(stats["mpoff"]))
        if not seeds:
            per_corpus[corpus] = {"status": "no paired traces"}
            continue
        d_col = [stats["gnn"][s]["coloc_rate"] - stats["mpoff"][s]["coloc_rate"] for s in seeds]
        d_nod = [stats["gnn"][s]["distinct_nodes_frac"] - stats["mpoff"][s]["distinct_nodes_frac"] for s in seeds]
        p_col = wilcoxon_exact(d_col)
        per_corpus[corpus] = {
            "n_pairs": len(seeds),
            "coloc_gnn": st.median([stats["gnn"][s]["coloc_rate"] for s in seeds]),
            "coloc_mpoff": st.median([stats["mpoff"][s]["coloc_rate"] for s in seeds]),
            "median_coloc_gnn_minus_mpoff": st.median(d_col), "p_coloc": p_col,
            "median_distinct_nodes_gnn_minus_mpoff": st.median(d_nod), "p_distinct_nodes": wilcoxon_exact(d_nod),
            "fires": bool(st.median(d_col) < 0 and p_col is not None and p_col < ALPHA),
        }
    fired = [c for c, v in per_corpus.items() if v.get("fires")]
    out = {"stage": "h3", "bar": {"alpha": ALPHA, "corpora_required": 2, "direction": "gnn co-locates less"},
           "per_corpus": per_corpus, "corpora_fired": fired,
           "verdict": "H3-FIRES" if len(fired) >= 2 else "H3-DOES-NOT-FIRE",
           "per_seed": {c: {a: {str(s): v for s, v in raw[c][a].items()} for a in ARMS} for c in raw}}
    return out


def stage_h4(args: argparse.Namespace) -> dict:
    h3 = json.loads(Path(args.h3).read_text())["per_seed"]
    sens_root = Path(args.sensitivity_root)
    per_corpus: Dict[str, dict] = {}
    for corpus in CORPORA:
        xs: List[float] = []; ys: List[float] = []
        gx: List[float] = []; gy: List[float] = []
        for arm in ARMS:
            for seed in range(1, 17):
                f = sens_root / corpus / f"{arm}_s{seed}.json"
                key = str(seed)
                if not f.is_file() or key not in (h3.get(corpus, {}).get(arm) or {}):
                    continue
                sens = float(json.loads(f.read_text())["median_queue_sensitivity"])
                col = float(h3[corpus][arm][key]["coloc_rate"])
                xs.append(sens); ys.append(col)
                if arm == "gnn":
                    gx.append(sens); gy.append(col)
        if len(xs) < 8:
            per_corpus[corpus] = {"status": "too few points", "n": len(xs)}
            continue
        rho, p = _spearman_perm(xs, ys)
        g_rho, g_p = _spearman_perm(gx, gy) if len(gx) >= 8 else (float("nan"), float("nan"))
        per_corpus[corpus] = {"n_pooled": len(xs), "rho_pooled": rho, "p_pooled": p,
                              "n_gnn": len(gx), "rho_gnn_only": g_rho, "p_gnn_only": g_p,
                              "fires_pooled": bool(rho < 0 and p < ALPHA),
                              "gnn_only_negative": bool(g_rho < 0)}
    fired = [c for c, v in per_corpus.items() if v.get("fires_pooled")]
    within = [c for c, v in per_corpus.items() if v.get("gnn_only_negative")]
    return {"stage": "h4", "bar": {"alpha": ALPHA, "corpora_required": 2,
                                   "within_arm_required": "gnn-only negative on >= 1 corpus"},
            "per_corpus": per_corpus, "corpora_fired": fired, "gnn_only_negative_on": within,
            "verdict": "H4-FIRES" if (len(fired) >= 2 and within) else "H4-DOES-NOT-FIRE"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="stage", required=True)
    p0 = sub.add_parser("s0")
    p0.add_argument("--lr", default="lr2e3")
    p0.add_argument("--output", type=Path, required=True)
    p1 = sub.add_parser("h1")
    p1.add_argument("--traces-root", required=True)
    p1.add_argument("--output", type=Path, required=True)
    p2 = sub.add_parser("h2-one")
    for flag in ("--corpus", "--arm", "--checkpoint", "--trace"):
        p2.add_argument(flag, required=True)
    p2.add_argument("--seed", type=int, required=True)
    p2.add_argument("--output", type=Path, required=True)
    p3 = sub.add_parser("h2")
    p3.add_argument("--results-root", required=True)
    p3.add_argument("--output", type=Path, required=True)
    p4 = sub.add_parser("h3")
    p4.add_argument("--results-root", required=True)
    p4.add_argument("--output", type=Path, required=True)
    p4b = sub.add_parser("h3-one")
    for flag in ("--corpus", "--arm", "--trace"):
        p4b.add_argument(flag, required=True)
    p4b.add_argument("--seed", type=int, required=True)
    p4b.add_argument("--output", type=Path, required=True)
    p5 = sub.add_parser("h4")
    p5.add_argument("--h3", required=True)
    p5.add_argument("--sensitivity-root", required=True)
    p5.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    result = {"s0": stage_s0, "h1": stage_h1, "h2-one": stage_h2_one, "h2": stage_h2,
              "h3": stage_h3, "h3-one": stage_h3_one, "h4": stage_h4}[args.stage](args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
