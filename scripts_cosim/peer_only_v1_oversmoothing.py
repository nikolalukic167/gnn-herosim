#!/usr/bin/env python3
"""peer_only_v1 C3 (AMENDMENT 8) — what the bipartite GIN does to platform state.

C1 and C2 ask whether the bipartite task<->platform stage costs. This asks WHY, and it is the
only one of the three that speaks to whether a bipartite graph could ever work in this
environment rather than whether this one does.

Method. For every `gnn` checkpoint, run the model's OWN ``_encode`` twice over the same cached
graphs:

  post = _encode(...) with ``mp_platform_edges = True``   — the platform block after the GIN
  pre  = _encode(...) with ``mp_platform_edges = False``  — the encoder's platform block,
                                                            untouched, which is literally what
                                                            `peeronly` and `mpoff` hand to the
                                                            EdgeScorer

Nothing is reconstructed: the contrast IS the two arms, computed by the code the gate serves.
Toggling the flag on a loaded model is safe precisely because it is weight-invisible — the GIN
is constructed either way and simply not run.

Two measurements, both registered as bars in ``peer_only_v1_read.py``:

  separation  mean pairwise cosine distance among the platform rows, post / pre. Below
              C3_SEPARATION_RATIO the GIN has made platforms materially harder to tell apart.
  retention   R^2 of a ridge probe recovering each platform's raw queue column from its
              embedding, post / pre, FIT AND SCORED ON DISJOINT GRAPHS. This is the
              decision-relevant half: platforms can stay far apart while the queue axis is
              washed out, and the two bars are counted separately so that case is visible.

Usage (datalab, CPU, ~minutes):
  python3 scripts_cosim/peer_only_v1_oversmoothing.py \
      --cache simulation_data/graphs_cache_peer_only_v1_1670_psv3 \
      --models models --tag peer-only-v1-1670-gnn-lr2e3 --out results_c3.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts_cosim.peer_only_v1_read import (  # noqa: E402
    C3_N_GRAPHS, C5_MIN_GRAPHS, read_c3, read_c5,
)
from scripts_cosim.scheduler_residence_v1_r3_read import spearman  # noqa: E402

# The queue columns of the legacy_v0 platform block. Reported by name when the block is wide
# enough to have them; the probe itself sweeps EVERY column regardless, so a layout change
# cannot silently turn this into a probe of the wrong thing.
QUEUE_COLUMNS = (7, 13)


def load_graphs(cache_dir: Path, n_graphs: int) -> List[Any]:
    """`n_graphs` cached graphs, strided across the cache rather than a head slice.

    Cached graphs are written in dataset order, so the first N are one corner of the corpus
    (one collection, one density). Same reasoning as verify_venue_parity.capture_fixture.
    """
    graphs_pkl = cache_dir / "graphs.pkl"
    if not graphs_pkl.is_file():
        raise FileNotFoundError(f"FAIL LOUD: no graphs.pkl under {cache_dir}")
    with graphs_pkl.open("rb") as handle:
        graphs = pickle.load(handle)
    if not isinstance(graphs, list) or not graphs:
        raise ValueError(f"FAIL LOUD: {graphs_pkl} is not a non-empty list of graphs")
    stride = max(1, len(graphs) // n_graphs)
    picked = [graphs[i] for i in range(0, len(graphs), stride)][:n_graphs]
    if len(picked) < n_graphs:
        raise ValueError(
            f"FAIL LOUD: cache holds {len(graphs)} graphs, stride {stride} yields only "
            f"{len(picked)} of the requested {n_graphs}"
        )
    print(f"[cache] {len(graphs)} graphs, took {len(picked)} at stride {stride}", flush=True)
    return picked


def mean_pairwise_cosine_distance(block: np.ndarray) -> Optional[float]:
    """Mean 1 - cos(u, v) over distinct rows. None when there are fewer than two rows.

    Cosine rather than Euclidean on purpose: the GIN is free to rescale the whole block, and a
    uniform rescaling is not a loss of information. What matters is whether the platforms still
    point in different directions.
    """
    if block.shape[0] < 2:
        return None
    norms = np.linalg.norm(block, axis=1, keepdims=True)
    # A dead row (all zeros) has no direction; it cannot enter a cosine and is dropped.
    keep = norms[:, 0] > 1e-12
    if int(keep.sum()) < 2:
        return None
    unit = block[keep] / norms[keep]
    sims = unit @ unit.T
    n = unit.shape[0]
    off = ~np.eye(n, dtype=bool)
    return float(np.mean(1.0 - sims[off]))


def ridge_r2(x_train: np.ndarray, y_train: np.ndarray,
             x_test: np.ndarray, y_test: np.ndarray, *, lam: float = 1e-3) -> float:
    """Held-out R^2 of a ridge probe y ~ x, standardised on the TRAIN moments only.

    Clipped at 0: a probe that does worse than predicting the mean has recovered nothing, and
    letting it go negative would make the post/pre ratio meaningless.
    """
    if x_train.shape[0] < 8 or x_test.shape[0] < 8:
        return 0.0
    mu, sd = x_train.mean(axis=0), x_train.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    xt = np.hstack([(x_train - mu) / sd, np.ones((x_train.shape[0], 1))])
    xv = np.hstack([(x_test - mu) / sd, np.ones((x_test.shape[0], 1))])
    ym = float(y_train.mean())
    a = xt.T @ xt + lam * np.eye(xt.shape[1])
    w = np.linalg.solve(a, xt.T @ (y_train - ym))
    pred = xv @ w + ym
    ss_res = float(np.sum((y_test - pred) ** 2))
    ss_tot = float(np.sum((y_test - float(y_test.mean())) ** 2))
    if ss_tot < 1e-12:
        return 0.0
    return max(0.0, 1.0 - ss_res / ss_tot)


def encode_both_ways(model, graphs: Sequence[Any]) -> Dict[str, Any]:
    """Per-graph platform blocks pre- and post-GIN, plus the raw platform features."""
    import torch

    if not getattr(model, "mp_platform_edges", False):
        raise ValueError(
            "FAIL LOUD: this checkpoint already has mp_platform_edges=False — it is a "
            "`peeronly` arm and has no bipartite stage to probe."
        )
    if getattr(model, "_disable_mp", False):
        raise ValueError("FAIL LOUD: this checkpoint has message passing disabled entirely")

    pre_blocks, post_blocks, raw_feats = [], [], []
    with torch.no_grad():
        for data in graphs:
            model.mp_platform_edges = True
            _, platform_post = model._encode(data)
            model.mp_platform_edges = False
            _, platform_pre = model._encode(data)
            model.mp_platform_edges = True          # restore, always
            post_blocks.append(platform_post.detach().cpu().numpy().astype(np.float64))
            pre_blocks.append(platform_pre.detach().cpu().numpy().astype(np.float64))
            raw_feats.append(data.platform_features.detach().cpu().numpy().astype(np.float64))
    return {"pre": pre_blocks, "post": post_blocks, "raw": raw_feats}


def probe_checkpoint(model, graphs: Sequence[Any]) -> Dict[str, Any]:
    """The two registered measurements for one checkpoint."""
    enc = encode_both_ways(model, graphs)
    pre, post, raw = enc["pre"], enc["post"], enc["raw"]

    sep_pre = [d for d in (mean_pairwise_cosine_distance(b) for b in pre) if d is not None]
    sep_post = [d for d in (mean_pairwise_cosine_distance(b) for b in post) if d is not None]
    if not sep_pre or not sep_post:
        raise ValueError("FAIL LOUD: no graph had two live platform rows to compare")
    mean_pre, mean_post = float(np.mean(sep_pre)), float(np.mean(sep_post))
    separation_ratio = mean_post / mean_pre if mean_pre > 1e-12 else float("nan")

    # Fit and score on DISJOINT graphs (even/odd), so a probe cannot memorise one cluster's
    # platforms and be scored on them.
    n = len(graphs)
    tr = [i for i in range(n) if i % 2 == 0]
    te = [i for i in range(n) if i % 2 == 1]
    width = raw[0].shape[1]
    cols = [c for c in QUEUE_COLUMNS if c < width]

    def stack(blocks, idx):
        return np.vstack([blocks[i] for i in idx])

    per_column: Dict[str, Dict[str, float]] = {}
    for col in range(width):
        y_tr = stack(raw, tr)[:, col]
        y_te = stack(raw, te)[:, col]
        if float(np.std(y_tr)) < 1e-9:
            continue                                  # a constant column is not recoverable
        per_column[str(col)] = {
            "pre": ridge_r2(stack(pre, tr), y_tr, stack(pre, te), y_te),
            "post": ridge_r2(stack(post, tr), y_tr, stack(post, te), y_te),
        }
    # The registered statistic is the queue columns; when neither varies in this cache the
    # probe says so loudly rather than substituting a different column's number.
    q = [per_column[str(c)] for c in cols if str(c) in per_column]
    if not q:
        raise ValueError(
            f"FAIL LOUD: no queue column among {cols} varies in this cache (platform block "
            f"width {width}). The retention half of C3 is not measurable here."
        )
    # C5 (AMENDMENT 10): the same numbers per GRAPH, so the compression can be correlated
    # with that graph's platform count. Free -- it is the data C3 already computed, not
    # aggregated away.
    per_graph = []
    for i in range(n):
        a = mean_pairwise_cosine_distance(pre[i])
        b = mean_pairwise_cosine_distance(post[i])
        if a is None or b is None or a <= 1e-12:
            continue
        per_graph.append({"n_platforms": int(raw[i].shape[0]),
                          "separation_pre": a, "separation_post": b, "ratio": b / a})
    rho, rho_p = (None, None)
    if len(per_graph) >= C5_MIN_GRAPHS:
        rho, rho_p = spearman([g["n_platforms"] for g in per_graph],
                              [g["ratio"] for g in per_graph])

    return {
        "separation_ratio": separation_ratio,
        "per_graph": per_graph, "c5_rho": rho, "c5_rho_p": rho_p,
        "separation_pre": mean_pre, "separation_post": mean_post,
        "queue_r2_pre": float(np.mean([m["pre"] for m in q])),
        "queue_r2_post": float(np.mean([m["post"] for m in q])),
        "queue_columns": cols, "platform_feature_width": width,
        "per_column_r2": per_column,
        "n_graphs": n,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, help="graphs cache directory")
    ap.add_argument("--models", default="models", help="checkpoint directory")
    ap.add_argument("--tag", default="peer-only-v1-1670-gnn-lr2e3",
                    help="checkpoint stem before -seed<N>.pt")
    ap.add_argument("--seeds", default="1-16", help="e.g. 1-16 or 1,2,3")
    ap.add_argument("--n-graphs", type=int, default=C3_N_GRAPHS)
    ap.add_argument("--out", default="peer_only_v1_c3.json")
    args = ap.parse_args(argv)

    if "-" in args.seeds:
        lo, hi = args.seeds.split("-")
        seeds = list(range(int(lo), int(hi) + 1))
    else:
        seeds = [int(s) for s in args.seeds.split(",")]

    from src.executesimulation import load_gnn_model

    graphs = load_graphs(Path(args.cache), args.n_graphs)

    per_ckpt: Dict[int, Dict[str, Any]] = {}
    for s in seeds:
        ck = Path(args.models) / f"{args.tag}-seed{s}.pt"
        if not ck.is_file():
            raise FileNotFoundError(f"FAIL LOUD: missing checkpoint {ck}")
        model, _ = load_gnn_model(ck)
        model.eval()
        r = probe_checkpoint(model, graphs)
        per_ckpt[s] = r
        print(f"[seed {s:>2}] separation {r['separation_pre']:.4f} -> {r['separation_post']:.4f} "
              f"(ratio {r['separation_ratio']:.3f})   queue R^2 {r['queue_r2_pre']:.3f} -> "
              f"{r['queue_r2_post']:.3f}", flush=True)

    verdict = read_c3(per_ckpt)
    c5 = read_c5({s: r.get("c5_rho") for s, r in per_ckpt.items()})
    doc = {"verdict": verdict, "c5": c5, "per_checkpoint": per_ckpt,
           "cache": args.cache, "tag": args.tag, "n_graphs": args.n_graphs}
    Path(args.out).write_text(json.dumps(doc, indent=1))
    print("\n=== C3 ===")
    print(f"verdict            : {verdict['verdict']}")
    print(f"median separation  : {verdict.get('median_separation_ratio')}  "
          f"(bar <= {verdict['bar']['separation_ratio']}, "
          f"{verdict.get('n_below_separation_bar')}/{verdict.get('n_checkpoints')} below)")
    print(f"median retention   : {verdict.get('median_retention_ratio')}  "
          f"(bar <= {verdict['bar']['retention_ratio']}, "
          f"{verdict.get('n_below_retention_bar')}/{verdict.get('n_retention_usable')} below)")
    print("\n=== C5 — does the compression scale with platform count? ===")
    print(f"verdict            : {c5['verdict']}")
    print(f"median rho         : {c5.get('median_rho')}  (bar <= {c5['bar']['rho']}, "
          f"{c5.get('n_negative')}/{c5.get('n_checkpoints')} negative)"
          if c5.get("verdict") != "UNREADABLE" else f"  {c5.get('reason')}")
    print(f"written            : {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
