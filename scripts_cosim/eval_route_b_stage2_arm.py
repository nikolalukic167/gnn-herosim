#!/usr/bin/env python3
"""route_b stage 2, W4 — evaluate a stage-2 checkpoint through the §4 shared decoder.

Offline from the DAG cache: no live simulation, no serving path. Reuses, never
re-types, the §2/§4 machinery both trainers and the decoder already share:

  * ``decode_masked_topo_placement`` / ``topological_task_order``
    (``src/policy/gnn/seq_decode.py``) — the ONE decoder every arm decodes through.
  * ``partial_state_columns`` / ``build_partial_state_context_from_graph`` /
    ``_extract_dim22_rows_for_task`` (``src/policy/tabular/reduced_features.py``) —
    the single-source feature builders A2's per-step recompute and A3's static
    scores both call.
  * ``make_partial_state_score_fn`` (``src/policy/gnn/partial_state_edges.py``) —
    the ONE prefix-conditioned scorer A1's decode uses, identical to what
    ``train_near_rtt._masked_topo_regret_for_graph`` calls at train time (the
    reference decode call shape this script's GNN path mirrors).

Arm type comes from the checkpoint's meta/sidecar, never guessed from a successful
``load_state_dict`` (memory ``herosim-checkpoint-contract-holes``,
``herosim-sidecar-keys-need-serving-whitelist``): an MLP checkpoint with
``inference_feature_layout`` dim63crk or dim25cr, or a GNN checkpoint whose
``<model>.contract.json`` sidecar exists. A sidecar-less/meta-less checkpoint is
refused outright (stage-1 rule: a checkpoint without a contract is not evidence).

Per dataset: decode regret % vs the alpha=2.0 constrained optimum (the tied-optimal
RTT set the cache carries), the decoded combo, split membership (train/val/test from
the split artifact, sha256 verified), and infeasible completion counted loudly —
never relaxed (§4's registered prohibition; ``uniq_platform``'s relax-to-argmax path
does not exist in this mode).

Tie rule (§4): the decoder's own step ties already break deterministically
(lowest placement id); per-step tie WIDTHS are reported. Where the decoded plan's
score exactly ties one or more OTHER feasible plans at the constrained optimum's own
tie group, the regret is reported as a band [pessimistic, mean_tied, optimistic]
over that tie group — expected to collapse to a point for a float-scored model, but
the band must exist for any statistic that feeds a reading.

Usage::

    PIPENV_IGNORE_VIRTUALENVS=1 PYTHONPATH=. pipenv run python3 \\
        scripts_cosim/eval_route_b_stage2_arm.py \\
        --checkpoint models/tabular/route_b_stage2_a2_dim63crk.pt \\
        --cache-dir simulation_data/graphs_cache_route_b_pilot_s_dag \\
        --split-artifact experiments/route_b_stage2_split_v1.json \\
        --alpha-key 2.0 \\
        --report simulation_data/route_b_stage2_a2_eval.json

    # Positive control (§4 acceptance target, reused from
    # verify_route_b_scorer_agreement.py --check-decoder):
    PIPENV_IGNORE_VIRTUALENVS=1 PYTHONPATH=. pipenv run python3 \\
        scripts_cosim/eval_route_b_stage2_arm.py --self-check \\
        --corpus simulation_data/gnn_datasets_dag4_route_b_pilot_v1_arm_s \\
        --report simulation_data/route_b_pilot_v1_arm_s_rtt.json

    # Aggregate several per-draw reports (the §9 abort statistic / §6 sigma):
    PIPENV_IGNORE_VIRTUALENVS=1 PYTHONPATH=. pipenv run python3 \\
        scripts_cosim/eval_route_b_stage2_arm.py --aggregate \\
        --reports simulation_data/route_b_stage2_a2_eval_seed{1,2,3,4}.json \\
        --report simulation_data/route_b_stage2_a2_aggregate.json
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = REPO_ROOT / "src" / "notebooks"
for _p in (str(REPO_ROOT), str(NOTEBOOKS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from non_unique_lib.training_contract import (  # noqa: E402
    canonical_parent_id,
    load_split_artifact,
)
from src.placement.env_fingerprint import (  # noqa: E402
    describe_code_provenance,
    describe_python_env,
    env_fingerprint,
)
from src.policy.gnn.partial_state_edges import make_partial_state_score_fn  # noqa: E402
from src.policy.gnn.seq_decode import (  # noqa: E402
    decode_masked_topo_placement,
    topological_task_order,
)
from src.policy.tabular.reduced_features import (  # noqa: E402
    DIM25CR_FEATURE_DIM,
    DIM63CRK_FEATURE_DIM,
    PARTIAL_STATE_FEATURE_DIM,
    _extract_dim22_rows_for_task,
    build_partial_state_context_from_graph,
    partial_state_columns,
    require_matching_partial_state_contract,
    resolve_partial_state_contract,
)


class EvalError(RuntimeError):
    """Any malformed input. Never swallowed — fail loudly."""


# ---------------------------------------------------------------------------
# Checkpoint / arm identification — sidecar-driven, never inferred from a load.
# ---------------------------------------------------------------------------


def load_arm(checkpoint_path: Path) -> Dict[str, Any]:
    """Identify and construct the scorer for one checkpoint.

    Returns a dict: {"arm_type": "mlp_dim63crk"|"mlp_dim25cr"|"gnn", "meta": {...},
    "score_fn_factory": callable(graph, ctx) -> score_fn(task_idx, committed)}.

    Refuses a sidecar-less checkpoint outright (stage-1 rule): an MLP checkpoint
    with no ``inference_feature_layout`` in its (bare-dict) payload, or a GNN
    checkpoint (bare state_dict) with no ``<model>.contract.json`` sidecar, is not
    evidence and this function raises rather than guessing from tensor shapes.
    """
    import torch

    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    sidecar_path = checkpoint_path.with_suffix(".contract.json")

    if isinstance(payload, dict) and "model_state_dict" in payload:
        layout = payload.get("inference_feature_layout")
        if not layout:
            raise EvalError(
                f"{checkpoint_path}: MLP checkpoint declares no "
                "inference_feature_layout — a checkpoint without a contract is not "
                "evidence (stage-1 rule); refusing to guess dim63crk vs dim25cr "
                "from input_dim."
            )
        layout = str(layout).strip().lower()
        if layout not in ("dim63crk", "dim25cr"):
            raise EvalError(
                f"{checkpoint_path}: unsupported MLP layout {layout!r} for stage-2 "
                "eval (expected dim63crk or dim25cr)"
            )
        from src.policy.tabular.mlp_model import PointwiseEdgeMLP

        input_dim = int(payload["input_dim"])
        expected = DIM63CRK_FEATURE_DIM if layout == "dim63crk" else DIM25CR_FEATURE_DIM
        if input_dim != expected:
            raise EvalError(
                f"{checkpoint_path}: layout {layout!r} declares input_dim={input_dim}, "
                f"expected {expected}"
            )
        model = PointwiseEdgeMLP(
            input_dim=input_dim, hidden_dim=int(payload.get("hidden_dim", 64))
        )
        model.load_state_dict(payload["model_state_dict"])
        model.eval()

        include_partial_state = layout == "dim63crk"
        if include_partial_state:
            trained_contract = payload.get("partial_state_contract")
            require_matching_partial_state_contract(
                trained_contract, resolve_partial_state_contract(),
                model_label=str(checkpoint_path),
            )

        def make_mlp_score_fn(graph, ctx):
            n_tasks = int(graph.n_tasks)
            parent_id = str(
                getattr(graph, "parent_dataset_id", None) or checkpoint_path.stem
            )
            tl = graph.task_logit_to_placement

            def score(task_idx: int, committed: Mapping[int, Any]):
                candidates = [tuple(c) for c in tl[task_idx]]
                block = (
                    partial_state_columns(ctx, task_idx, candidates, committed)
                    if include_partial_state
                    else None
                )
                rows, skip_reason = _extract_dim22_rows_for_task(
                    graph,
                    "eval",
                    parent_id,
                    task_idx,
                    n_tasks,
                    candidate_relative=True,
                    partial_state_block=block,
                    target_override=0,
                )
                if skip_reason:
                    raise EvalError(f"feature extraction failed: {skip_reason}")
                import numpy as np

                x = torch.from_numpy(
                    np.stack([r.features for r in rows]).astype("float32")
                )
                with torch.no_grad():
                    return model(x).tolist()

            return score

        return {
            "arm_type": f"mlp_{layout}",
            "meta": payload,
            "score_fn_factory": make_mlp_score_fn,
            "prefix_conditioned": include_partial_state,
        }

    # GNN: bare state_dict, contract sidecar mandatory.
    if not sidecar_path.is_file():
        raise EvalError(
            f"{checkpoint_path}: no .contract.json sidecar — a GNN checkpoint "
            "without one is not evidence (stage-1 rule) and its architecture "
            "cannot be inferred from a successful load_state_dict "
            "(herosim-checkpoint-contract-holes)."
        )
    sidecar = json.loads(sidecar_path.read_text())
    # Whether the GIN forward is skipped is weight-invisible and lives in the environment,
    # so a checkpoint trained MP-OFF and scored here with the flag unset silently
    # message-passes through weights that were never fitted with it. Measured on the
    # route_b DAG corpus 2026-09-03: train regret 12.67% (matched) vs 72.23% (mismatched)
    # — an error large enough to read as a decisive ablation result. Sidecars written
    # before that date carry no key; those checkpoints predate the fix and are only safe
    # if the flag was unset at train time, which is the historical default.
    declared_mp_off = sidecar.get("disable_message_passing")
    serving_mp_off = os.environ.get(
        "GNN_DISABLE_MESSAGE_PASSING", "").strip().lower() in ("1", "true", "yes")
    if declared_mp_off is not None and bool(declared_mp_off) != serving_mp_off:
        raise EvalError(
            f"{checkpoint_path}: sidecar declares disable_message_passing="
            f"{bool(declared_mp_off)} but this process has "
            f"GNN_DISABLE_MESSAGE_PASSING {'set' if serving_mp_off else 'unset'}. "
            "Scoring an MP-OFF checkpoint with message passing on (or the reverse) is a "
            "train/serve mismatch, not an ablation — export the flag to match the sidecar."
        )
    if not sidecar.get("partial_state_edge_features"):
        raise EvalError(
            f"{checkpoint_path}: sidecar does not declare "
            "partial_state_edge_features=true — this is not a stage-2 T2 (A1) "
            "checkpoint (masked_topo eval here assumes prefix conditioning)."
        )
    trained_contract = sidecar.get("partial_state_contract")
    require_matching_partial_state_contract(
        trained_contract, resolve_partial_state_contract(), model_label=str(checkpoint_path)
    )
    from src.notebooks.prepare_graphs_cache import DAG_TASK_TYPE_VOCAB
    from src.policy.gnn.gnn_model import TaskPlacementGNN

    onehot_dim = int(sidecar.get("task_type_onehot_dim") or 0)
    if onehot_dim and list(sidecar.get("dag_task_type_vocab") or []) != list(
        DAG_TASK_TYPE_VOCAB
    ):
        raise EvalError(
            f"{checkpoint_path}: sidecar dag_task_type_vocab "
            f"{sidecar.get('dag_task_type_vocab')!r} != the live vocab "
            f"{list(DAG_TASK_TYPE_VOCAB)!r} — a vocab reorder would silently "
            "permute task types"
        )
    partial_state_dim = int(sidecar.get("partial_state_feature_dim") or 0)
    if partial_state_dim != PARTIAL_STATE_FEATURE_DIM:
        raise EvalError(
            f"{checkpoint_path}: sidecar partial_state_feature_dim="
            f"{partial_state_dim} != live PARTIAL_STATE_FEATURE_DIM="
            f"{PARTIAL_STATE_FEATURE_DIM}"
        )

    state_dict = payload
    task_w = state_dict["task_encoder.net.0.weight"]
    plat_w = state_dict["platform_encoder.net.0.weight"]
    task_feature_dim_total = int(task_w.shape[1])
    platform_feature_dim = int(plat_w.shape[1])
    task_feature_dim = task_feature_dim_total - onehot_dim
    # hidden_dim/embedding_dim ARE weight-visible (they set every Linear's shape), so —
    # unlike mp_dag_edges/task_type_onehot_dim, which are not — they are read off the
    # state_dict rather than guessed from the constructor defaults (128/64), which would
    # otherwise silently mismatch any checkpoint trained with different widths (e.g. the
    # A1 smoke, hidden_dim=64) and fail loudly via a shape-mismatch load_state_dict.
    hidden_dim = int(task_w.shape[0])
    embedding_dim = int(state_dict["task_encoder.net.4.weight"].shape[0])
    num_layers = sum(
        1 for k in state_dict if k.startswith("gin.convs.") and k.endswith(".nn.lins.0.weight")
    )
    if num_layers <= 0:
        raise EvalError(f"{checkpoint_path}: could not infer num_layers from gin.convs.* keys")

    def build_model(task_input_dim: int, platform_input_dim: int) -> "TaskPlacementGNN":
        m = TaskPlacementGNN(
            task_feature_dim=task_input_dim,
            platform_feature_dim=platform_input_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            mp_residual=bool(sidecar.get("mp_residual", False)),
            mp_node_edges=bool(sidecar.get("mp_node_edges", False)),
            mp_node_edges_candidates_only=bool(
                sidecar.get("mp_node_edges_candidates_only", True)
            ),
            mp_network_entities=bool(sidecar.get("mp_network_entities", False)),
            mp_dag_edges=bool(sidecar.get("mp_dag_edges", False)),
            task_type_onehot_dim=onehot_dim,
            partial_state_edge_dim=partial_state_dim,
            normalize_platform_inputs=sidecar.get("feature_dim") == 21,
        )
        m.load_state_dict(state_dict)
        m.eval()
        return m

    model = build_model(task_feature_dim, platform_feature_dim)

    def make_gnn_score_fn(graph, ctx):
        return make_partial_state_score_fn(model, graph, ctx)

    return {
        "arm_type": "gnn",
        "meta": sidecar,
        "score_fn_factory": make_gnn_score_fn,
        "prefix_conditioned": True,
    }


# ---------------------------------------------------------------------------
# Per-dataset decode + regret
# ---------------------------------------------------------------------------


def _demands_and_caps(graph: Any, alpha_key: str) -> Tuple[Dict[int, List[float]], Dict[int, float]]:
    ctx = build_partial_state_context_from_graph(graph)
    caps_by_alpha = graph.partial_state_ctx["node_caps_by_alpha"]
    if alpha_key not in caps_by_alpha:
        raise EvalError(f"alpha_key {alpha_key!r} not in node_caps_by_alpha")
    ctx.node_caps = caps_by_alpha[alpha_key]
    n_tasks = int(graph.n_tasks)
    demands = {
        t: [float(ctx.demand[(t, tuple(int(v) for v in c))]) for c in graph.task_logit_to_placement[t]]
        for t in range(n_tasks)
    }
    return demands, ctx


def _load_sweep(simulation_data_root: Path, dataset_id: str, task_types_db: Dict[str, dict]):
    """dataset_id is ``<corpus_name>/ds_XXXXX``, exactly the layout on disk under
    simulation_data/ (dataset_ids.pkl records it verbatim — see
    prepare_graphs_cache.py)."""
    from scripts_cosim.score_route_b_contention import Dataset

    ds_dir = simulation_data_root / dataset_id
    if not ds_dir.is_dir():
        raise EvalError(f"dataset dir not found: {ds_dir}")
    return Dataset(ds_dir, task_types_db, "rtt")


def evaluate_dataset(
    graph: Any,
    dataset_id: str,
    *,
    arm: Dict[str, Any],
    alpha_key: str,
    simulation_data_root: Path,
    task_types_db: Dict[str, dict],
) -> Dict[str, Any]:
    n_tasks = int(graph.n_tasks)
    demands, ctx = _demands_and_caps(graph, alpha_key)
    dag_parents = graph.dag_parents
    tl = graph.task_logit_to_placement

    score_fn = arm["score_fn_factory"](graph, ctx)
    combo = decode_masked_topo_placement(
        [None] * n_tasks,
        tl,
        n_tasks,
        dag_parents=dag_parents,
        node_caps=ctx.node_caps,
        demands=demands,
        score_fn=score_fn,
    )

    ds = _load_sweep(simulation_data_root, dataset_id, task_types_db)
    caps = ds.node_caps(float(alpha_key))
    feasible_rows = [(p, v) for p, v in ds.rows if ds.plan_feasible(p, caps)]
    if not feasible_rows:
        raise EvalError(f"{dataset_id}: no feasible rows at alpha={alpha_key}")
    opt_value = min(v for _p, v in feasible_rows)
    tie_group = [v for _p, v in feasible_rows if abs(v - opt_value) <= 1e-9]
    opt_band = {
        "pessimistic": max(tie_group),
        "mean_tied": sum(tie_group) / len(tie_group),
        "optimistic": min(tie_group),
        "tie_width": len(tie_group),
    }

    result: Dict[str, Any] = {
        "dataset_id": dataset_id,
        "alpha_key": alpha_key,
        "constrained_optimum": opt_band,
        "infeasible": combo is None,
    }
    if combo is None:
        result["decode_regret_pct"] = None
        result["decoded_combo"] = None
        return result

    decoded_plan = {t: (int(combo[t][0]), int(combo[t][1])) for t in range(n_tasks)}
    lookup = {tuple(sorted(p.items())): v for p, v in ds.rows}
    key = tuple(sorted(decoded_plan.items()))
    if key not in lookup:
        raise EvalError(
            f"{dataset_id}: decoded plan {key} not present in the placement sweep — "
            "the decoder produced a combination outside the enumerated space"
        )
    decoded_value = lookup[key]
    if not ds.plan_feasible(decoded_plan, caps):
        raise EvalError(
            f"{dataset_id}: decoder returned an INFEASIBLE plan {decoded_plan} — "
            "the mask should have made this impossible (fail loud, never relaxed)"
        )
    decoded_tie_group = [
        v for p, v in feasible_rows if abs(v - decoded_value) <= 1e-9
    ]

    def regret_pct(value: float) -> float:
        return 100.0 * (value - opt_value) / opt_value

    result["decoded_combo"] = [[int(a), int(b)] for a, b in decoded_plan.values()]
    result["decoded_rtt"] = decoded_value
    result["decode_regret_pct"] = {
        "registered": regret_pct(decoded_value),
        "pessimistic": regret_pct(max(decoded_tie_group)),
        "mean_tied": regret_pct(sum(decoded_tie_group) / len(decoded_tie_group)),
        "optimistic": regret_pct(min(decoded_tie_group)),
        "tie_width": len(decoded_tie_group),
    }
    return result


# ---------------------------------------------------------------------------
# Split membership
# ---------------------------------------------------------------------------


def split_membership(split_artifact_path: Path) -> Tuple[Dict[str, str], str]:
    payload, sha256 = load_split_artifact(split_artifact_path)
    membership: Dict[str, str] = {}
    for name in ("train", "val", "test"):
        for parent in payload[name]:
            membership[parent] = name
    return membership, sha256


# ---------------------------------------------------------------------------
# Main per-checkpoint eval
# ---------------------------------------------------------------------------


def run_eval(args: argparse.Namespace) -> Dict[str, Any]:
    cache_dir = Path(args.cache_dir)
    checkpoint_path = Path(args.checkpoint)
    with open(cache_dir / "graphs.pkl", "rb") as fh:
        graphs = pickle.load(fh)
    with open(cache_dir / "dataset_ids.pkl", "rb") as fh:
        dataset_ids = pickle.load(fh)
    if len(graphs) != len(dataset_ids):
        raise EvalError(f"graphs ({len(graphs)}) != dataset_ids ({len(dataset_ids)})")

    arm = load_arm(checkpoint_path)

    membership: Dict[str, str] = {}
    split_sha256 = None
    if args.split_artifact:
        membership, split_sha256 = split_membership(Path(args.split_artifact))

    task_types_db = json.loads(Path(args.task_types).read_text())
    simulation_data_root = REPO_ROOT / "simulation_data"

    per_dataset: List[Dict[str, Any]] = []
    n_infeasible = 0
    for graph, dataset_id in zip(graphs, dataset_ids):
        row = evaluate_dataset(
            graph,
            dataset_id,
            arm=arm,
            alpha_key=args.alpha_key,
            simulation_data_root=simulation_data_root,
            task_types_db=task_types_db,
        )
        parent = canonical_parent_id(dataset_id)
        row["parent_dataset_id"] = parent
        row["split"] = membership.get(parent, "unknown") if membership else None
        per_dataset.append(row)
        if row["infeasible"]:
            n_infeasible += 1

    feasible_regrets = [
        r["decode_regret_pct"]["registered"]
        for r in per_dataset
        if not r["infeasible"]
    ]
    train_regrets = [
        r["decode_regret_pct"]["registered"]
        for r in per_dataset
        if not r["infeasible"] and r.get("split") == "train"
    ]

    python_env = describe_python_env()
    report = {
        "checkpoint": str(checkpoint_path),
        "cache_dir": str(cache_dir),
        "arm_type": arm["arm_type"],
        "alpha_key": args.alpha_key,
        "split_artifact": (
            {"path": str(args.split_artifact), "sha256": split_sha256}
            if args.split_artifact
            else None
        ),
        "num_datasets": len(per_dataset),
        "num_infeasible": n_infeasible,
        "mean_regret_pct_all": (
            statistics.fmean(feasible_regrets) if feasible_regrets else None
        ),
        "median_regret_pct_all": (
            statistics.median(feasible_regrets) if feasible_regrets else None
        ),
        "mean_regret_pct_train": (
            statistics.fmean(train_regrets) if train_regrets else None
        ),
        "median_regret_pct_train": (
            statistics.median(train_regrets) if train_regrets else None
        ),
        "per_dataset": per_dataset,
        "run_provenance": {
            "code": describe_code_provenance(),
            "python_env": python_env,
            "env_fingerprint": env_fingerprint(python_env),
        },
    }
    return report


# ---------------------------------------------------------------------------
# Self-check: the §4 acceptance target, reused (not re-typed) from
# verify_route_b_scorer_agreement.check_decoder.
# ---------------------------------------------------------------------------


def run_self_check(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(REPO_ROOT / "scripts_cosim"))
    from verify_route_b_scorer_agreement import check_decoder

    alphas = tuple(args.alpha) if args.alpha else ("2.0", "3.0")
    return check_decoder(args.corpus, args.report, args.task_types, alphas)


# ---------------------------------------------------------------------------
# Aggregate mode
# ---------------------------------------------------------------------------


def run_aggregate(report_paths: Sequence[Path], out_path: Path) -> Dict[str, Any]:
    reports = [json.loads(p.read_text()) for p in report_paths]
    if not reports:
        raise EvalError("--aggregate needs at least one --reports entry")

    arm_types = {r["arm_type"] for r in reports}
    if len(arm_types) != 1:
        raise EvalError(f"--aggregate reports mix arm types: {sorted(arm_types)}")

    per_draw_train_regret: List[float] = []
    per_dataset_by_draw: List[Dict[str, float]] = []
    for r in reports:
        train_rows = [
            row for row in r["per_dataset"]
            if not row["infeasible"] and row.get("split") == "train"
        ]
        if train_rows:
            per_draw_train_regret.append(
                statistics.fmean(row["decode_regret_pct"]["registered"] for row in train_rows)
            )
        per_dataset_by_draw.append(
            {
                row["dataset_id"]: row["decode_regret_pct"]["registered"]
                for row in r["per_dataset"]
                if not row["infeasible"]
            }
        )

    # Pooled per-dataset paired-difference sigma across draws (§6 calibration): for
    # each dataset present in every draw, the spread of its regret across draws.
    common_ids = set.intersection(*(set(d) for d in per_dataset_by_draw)) if per_dataset_by_draw else set()
    per_dataset_sigma: Dict[str, float] = {}
    for ds_id in sorted(common_ids):
        values = [d[ds_id] for d in per_dataset_by_draw]
        if len(values) > 1:
            per_dataset_sigma[ds_id] = statistics.stdev(values)
    pooled_sigma = (
        statistics.fmean(per_dataset_sigma.values()) if per_dataset_sigma else None
    )

    result = {
        "arm_type": next(iter(arm_types)),
        "num_draws": len(reports),
        "reports": [str(p) for p in report_paths],
        "per_draw_train_regret_pct": per_draw_train_regret,
        "mean_train_regret_pct": (
            statistics.fmean(per_draw_train_regret) if per_draw_train_regret else None
        ),
        "median_train_regret_pct": (
            statistics.median(per_draw_train_regret) if per_draw_train_regret else None
        ),
        "pooled_per_dataset_paired_diff_sigma": pooled_sigma,
        "num_datasets_common_to_all_draws": len(common_ids),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--cache-dir", type=Path)
    ap.add_argument("--split-artifact", type=Path, default=None)
    ap.add_argument("--alpha-key", default="2.0")
    ap.add_argument("--task-types", default="data/nofs-ids/task-types.json")
    ap.add_argument("--report", type=Path, required=True,
                    help="output report path (single-eval / self-check), or the "
                         "aggregate output path under --aggregate")
    ap.add_argument("--self-check", action="store_true",
                    help="positive control: fed true min-marginals, the decoder "
                         "must reproduce the frozen greedy_masked_plan plans to "
                         "1e-9 (same target as --check-decoder)")
    ap.add_argument("--corpus", help="--self-check only")
    ap.add_argument("--alpha", action="append", help="--self-check only")
    ap.add_argument("--aggregate", action="store_true",
                    help="aggregate multiple --reports into one summary at --report")
    ap.add_argument("--reports", nargs="+", type=Path, help="--aggregate only")
    args = ap.parse_args()

    if args.self_check:
        if not args.corpus:
            raise SystemExit("--self-check needs --corpus")
        return run_self_check(args)

    if args.aggregate:
        if not args.reports:
            raise SystemExit("--aggregate needs --reports")
        result = run_aggregate(args.reports, args.report)
        print(
            f"[aggregate] {result['arm_type']} draws={result['num_draws']} "
            f"mean_train_regret_pct={result['mean_train_regret_pct']} "
            f"median_train_regret_pct={result['median_train_regret_pct']} "
            f"pooled_sigma={result['pooled_per_dataset_paired_diff_sigma']}"
        )
        return 0

    if not args.checkpoint or not args.cache_dir:
        raise SystemExit("--checkpoint and --cache-dir are required")
    report = run_eval(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"[{report['arm_type']}] {report['num_datasets']} datasets "
        f"({report['num_infeasible']} infeasible) "
        f"mean_regret_pct_all={report['mean_regret_pct_all']} "
        f"mean_regret_pct_train={report['mean_regret_pct_train']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
