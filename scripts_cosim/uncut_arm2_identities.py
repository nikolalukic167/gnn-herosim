"""uncut_groups_s0_v1 Arm 2, identity 2: with the re-encode period beyond the batch and the group anchor off, the Arm 2
scorer decodes bit-equal plans to the single-encode path.

The checkpoint is a flags-off slate_fix_v1 / Arm 1 checkpoint. The Arm 2 model is the same checkpoint plus RANDOM weights
for the B1 modules (committed_task_enc / committed_platform_enc), so the equality is a property of the code (an empty
committed state adds exactly zero), not of zero-initialised parameters. Both models decode the same cached graphs through
the registered masked_topo decoder; every step's logits and every plan must match exactly.

    python3 scripts_cosim/uncut_arm2_identities.py --cache <cache_dir> --checkpoint <ckpt.pt> --n 50 --out id2.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
import tempfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.policy.gnn.partial_state_edges import make_partial_state_score_fn  # noqa: E402
from src.policy.gnn.prefix_serving import load_prefix_conditioned_gnn  # noqa: E402
from src.policy.gnn.seq_decode import decode_masked_topo_placement  # noqa: E402
from src.policy.tabular.reduced_features import build_partial_state_context_from_graph  # noqa: E402


def _arm2_twin(checkpoint: Path, workdir: Path, period: str) -> Path:
    """A copy of the checkpoint with B1 weights added at random and the sidecar declaring ``period``."""
    state = torch.load(checkpoint, map_location="cpu")
    emb = int(state["task_encoder.net.4.weight"].shape[0])
    hidden = int(state["task_encoder.net.0.weight"].shape[0])
    g = torch.Generator().manual_seed(0)
    state["committed_task_enc.weight"] = torch.randn(emb, 1, generator=g)
    state["committed_platform_enc.0.weight"] = torch.randn(hidden, 2, generator=g)
    state["committed_platform_enc.2.weight"] = torch.randn(emb, hidden, generator=g)
    out = workdir / checkpoint.name
    torch.save(state, out)
    side = json.loads(checkpoint.with_suffix(".contract.json").read_text())
    side["reencode_every"] = period
    out.with_suffix(".contract.json").write_text(json.dumps(side))
    return out


def _decode(model, options, graph):
    ctx = build_partial_state_context_from_graph(graph)
    ctx.node_caps = graph.partial_state_ctx["node_caps_by_alpha"][options.alpha_key]
    n = int(graph.n_tasks)
    tl = graph.task_logit_to_placement
    demands = {t: [float(ctx.demand[(t, (int(c[0]), int(c[1])))]) for c in tl[t]] for t in range(n)}
    score = make_partial_state_score_fn(model, graph, ctx)
    trace = []

    def traced(t, committed):
        logits = score(t, committed).detach().clone()
        trace.append(logits)
        return logits

    with torch.no_grad():
        combo = decode_masked_topo_placement(
            [None] * n, tl, n, dag_parents=graph.dag_parents, node_caps=ctx.node_caps, demands=demands,
            score_fn=traced, allow_replica_reuse=options.allow_replica_reuse, relax_on_stuck=options.relax_on_stuck,
            initial_load=ctx.base_load or None,
        )
    return combo, trace


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ckpt = Path(args.checkpoint)
    with open(Path(args.cache) / "graphs.pkl", "rb") as fh:
        graphs = pickle.load(fh)
    chosen = [g for g in graphs if int(g.n_tasks) >= 2 and getattr(g, "partial_state_ctx", None)][: args.n]
    if len(chosen) < args.n:
        raise SystemExit(f"FAIL LOUD: only {len(chosen)} usable graphs in {args.cache}, wanted {args.n}")
    base, options, _ = load_prefix_conditioned_gnn(ckpt)
    with tempfile.TemporaryDirectory() as tmp:
        twin_path = _arm2_twin(ckpt, Path(tmp), "100000")
        twin, _, _ = load_prefix_conditioned_gnn(twin_path)
        if not twin.committed_inputs or twin.group_anchor:
            raise SystemExit("FAIL LOUD: the twin was not built with B1 only")
        rows, bad = [], 0
        for g in chosen:
            a, ta = _decode(base, options, g)
            for name in ("arm2_task_committed", "arm2_platform_state", "arm2_edge_keep", "partial_state_edge_attr", "_arm2_layout"):
                if hasattr(g, name):
                    delattr(g, name)
            b, tb = _decode(twin, options, g)
            same_plan = tuple(a) == tuple(b)
            same_logits = len(ta) == len(tb) and all(torch.equal(x, y) for x, y in zip(ta, tb))
            bad += 0 if (same_plan and same_logits) else 1
            rows.append({"dataset_id": str(getattr(g, "dataset_id", "?")), "n_tasks": int(g.n_tasks),
                         "steps": len(ta), "plan_equal": same_plan, "logits_equal": same_logits})
    result = {"identity": "arm2_id2_single_encode", "checkpoint": str(ckpt), "n_graphs": len(rows),
              "n_mismatch": bad, "rows": rows}
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(f"identity 2: {len(rows) - bad}/{len(rows)} graphs bit-equal (plans and every step's logits)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
