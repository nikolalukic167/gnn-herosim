#!/usr/bin/env python3
"""Timing of the v7 gnn training loss on one synthetic graph: the as-built batched teacher-forced loss (prefix-free encode) against the re-encode-per-placement loss (link_conv_loss, one big graph of every
distinct prefix and full-context state), forward + backward, on CPU or CUDA. Sparse peers (mean degree ~4, as the corpus: ~2n unordered pairs per group), a real SimPy fair-share fabric with random flows, the real
model size (hidden 128, emb 128, 3 layers, peer edges, edge conv). A speed and memory instrument only: no accuracy is read from it.

  bench_link_conv.py --device cpu|cuda --sizes 12 24 60 120 240 [--reps 3]"""
import argparse
import os
import sys
import time

import torch

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tests"))
os.environ.setdefault("PARTIAL_STATE_CONTRACT", "partial_state_v7")
os.environ.setdefault("PARTIAL_STATE_LOAD_SECONDS", "1")
os.environ.setdefault("PARTIAL_STATE_EXCHANGE_SECONDS", "1")
os.environ.setdefault("PARTIAL_STATE_PEER_MASS", "1")


class _Env:
    def setenv(self, k, v):
        os.environ[k] = v


def graph_on(device, n, seed, cands):
    import test_batched_tf_link as TB

    # 6 replicas per server (30 platforms) so that candidate sets of the corpus's size (mean ~12, max 23 under the uncapped slate) exist
    TB.REPLICAS[:] = [(node, 10 * node + k) for node in range(5) for k in range(6)]
    rnd, ctx, g, place = TB.scenario(seed, _Env(), n_tasks=n, density=min(0.4, 4.0 / n), n_flows=(8, 40), n_cands=cands)
    gen = torch.Generator().manual_seed(seed)
    n_p = len(TB.REPLICAS)
    g.task_features = torch.rand(n, 5, generator=gen)
    g.platform_features = torch.rand(n_p, 21, generator=gen)
    g.edge_attr = torch.rand(g.edge_index.size(1), 5, generator=gen)
    g.task_type_onehot4 = torch.eye(4)[torch.arange(n) % 4]
    g.dag_parents = {t: [] for t in range(n)}
    keys = sorted(ctx.peer_pairs)
    g.peer_edge_index = torch.tensor([[i for i, _j in keys], [j for _i, j in keys]], dtype=torch.long)
    g.peer_edge_attr = torch.tensor([[torch.log1p(torch.tensor(ctx.peer_pairs[k] / 1e6)).item()] for k in keys])
    plans = [[rnd.randrange(len(place[t])) for t in range(n)]]
    return ctx, g.to(device), plans, len(keys)


def model_on(device, link):
    from src.policy.gnn.gnn_model import TaskPlacementGNN

    torch.manual_seed(0)
    return TaskPlacementGNN(task_feature_dim=5, platform_feature_dim=21, embedding_dim=128, hidden_dim=128, num_layers=3, edge_dim=5, task_type_onehot_dim=4, partial_state_edge_dim=30,
                            mp_peer_edges=True, mp_bipartite_edge_conv=True, mp_bipartite_edge_attr_zero=True, mp_link_edge_conv=link).to(device).train()


def timed(fn, device, reps):
    ts, peak = [], 0.0
    for r in range(reps + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak = max(peak, torch.cuda.max_memory_allocated(device) / 2 ** 20)
        if r:
            ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2], peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sizes", type=int, nargs="+", default=[12, 24, 60, 120, 240])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--cands", type=int, nargs=2, default=[3, 6], help="min max candidates per task (the corpus: mean 12, max 23)")
    a = ap.parse_args()
    device = torch.device(a.device)
    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(device))
    print(f"threads {torch.get_num_threads()}  device {device}")
    print(f"{'T':>4} {'pairs':>6} {'states':>6} | {'as-built ms':>12} {'MB':>7} | {'link-conv ms':>12} {'MB':>7} | ratio")
    from src.policy.gnn.batched_tf import batched_tf_loss
    from src.policy.gnn.link_conv import link_conv_loss

    for n in a.sizes:
        ctx, g, plans, pairs = graph_on(device, n, 3, tuple(a.cands))
        order = list(range(n))
        base, conv = model_on(device, False), model_on(device, True)

        def f_base():
            base.zero_grad(set_to_none=True)
            batched_tf_loss(base, g, ctx, plans, order, 0.5, device).backward()

        def f_conv():
            conv.zero_grad(set_to_none=True)
            link_conv_loss(conv, g, ctx, plans, order, 0.5, device).backward()

        tb, mb = timed(f_base, device, a.reps)
        tc, mc = timed(f_conv, device, a.reps)
        print(f"{n:4d} {pairs:6d} {2 * n:6d} | {tb * 1000:12.1f} {mb:7.0f} | {tc * 1000:12.1f} {mc:7.0f} | {tc / tb:5.1f}x", flush=True)


if __name__ == "__main__":
    main()
