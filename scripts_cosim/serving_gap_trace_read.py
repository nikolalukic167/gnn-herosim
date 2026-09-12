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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="stage", required=True)
    p0 = sub.add_parser("s0")
    p0.add_argument("--lr", default="lr2e3")
    p0.add_argument("--output", type=Path, required=True)
    p1 = sub.add_parser("h1")
    p1.add_argument("--traces-root", required=True)
    p1.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    result = stage_s0(args) if args.stage == "s0" else stage_h1(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
