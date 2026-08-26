"""Two runs of a trainer at the same seed must produce the same weights.

This is the guard for the defect that retired three published-track claims: the MLP
trainer seeded its data split and batch order but never `torch.manual_seed`, so weight
init came from OS entropy and every MLP checkpoint before 2026-08-24 was an
unreproducible draw. The GNN trainer seeded correctly the whole time. The asymmetry
survived for months because the two trainers were never diffed against each other, so
the test that prevents a recurrence has to cover *every* trainer, not the one that broke.

Seeding is necessary but not sufficient: at a fixed seed the GIN autograd path was measured
diverging run to run even on CPU (2026-08-19, see scripts_cosim/gnn_necessity_ablation.py),
which is why the trainers also set `torch.use_deterministic_algorithms`.

What each kind of test here actually proves, because the split is not obvious:

  * The *training* tests catch an unseeded trainer — the defect that actually happened.
    Verified to have teeth: drop `torch.manual_seed` and 28/31 tensors diverge.
  * They do NOT catch removal of `use_deterministic_algorithms`. The GIN nondeterminism
    did not reproduce on this machine at any size tried (12-200 graphs, 2-5 epochs, with
    and without node edges, flag on and off — all bit-identical), so a dynamic test cannot
    discriminate here. It is hardware- and build-dependent, and absence on one box is not
    evidence it is gone.
  * The *static* tests cover that gap: they assert the line is present in every trainer.
    A guard that reads the source is weaker than one that measures behaviour, but for a
    defect whose signature is "a missing line that only sometimes shows up at runtime" it
    is the check that fires reliably.

Skips rather than fails when a cache is absent: `simulation_data/` is not part of a fresh
clone, the same reason tests/test_venue_parity.py skips on a missing checkpoint.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The cheapest caches on disk that each trainer will actually accept. These are not
# interchangeable: TINY_CACHE has no batch-regime rows, so the batch MLP trainer exits
# with "No dim22/dim24 rows extracted" on it, and BATCH_CACHE is the smallest that works.
TINY_CACHE = REPO_ROOT / "simulation_data/graphs_cache_regime_b_ect_pull_distill_oracle_split_v1"
BATCH_CACHE = REPO_ROOT / "simulation_data/graphs_cache_regime_b_oracle_split_cosim"

SEED = 4242


def _weights_of(checkpoint) -> Dict[str, torch.Tensor]:
    """The tensor parameters out of a checkpoint, whatever wrapper it uses.

    The MLP trainers save a dict of weights *plus* scalar metadata (`torch_seeded`,
    `input_dim`, ...) under `model_state_dict`; the GNN trainer saves a bare state_dict.
    """
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("model_state_dict", "state_dict"):
        inner = checkpoint.get(key)
        if isinstance(inner, dict):
            return {k: v for k, v in inner.items() if isinstance(v, torch.Tensor)}
    return {k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)}


def _assert_state_dicts_identical(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor], label: str) -> None:
    assert set(a) == set(b), f"{label}: parameter names differ"
    mismatched = [
        key
        for key in sorted(a)
        if not torch.equal(a[key].detach().cpu(), b[key].detach().cpu())
    ]
    if mismatched:
        worst = max(
            (a[k].detach().cpu() - b[k].detach().cpu()).abs().max().item() for k in mismatched
        )
        raise AssertionError(
            f"{label}: {len(mismatched)}/{len(a)} tensors differ between two runs at "
            f"seed {SEED} (max |delta| {worst:.3e}); first: {mismatched[:3]}"
        )


def _run_trainer(args, env_extra: Dict[str, str], cwd: Path = REPO_ROOT) -> None:
    env = dict(os.environ)
    # One thread and a fixed hash seed remove the two variables that are not the trainer's
    # own doing. Neither is the cause of the divergence this test guards against, but a
    # failure should point at the trainer rather than at the environment.
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
            "PIPENV_IGNORE_VIRTUALENVS": "1",
            "PYTHONPATH": str(REPO_ROOT),
            "WANDB_MODE": "disabled",
        }
    )
    env.update(env_extra)
    result = subprocess.run(
        [sys.executable, *args], cwd=str(cwd), env=env, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise AssertionError(
            f"trainer exited {result.returncode}\n"
            f"--- stdout tail ---\n{result.stdout[-3000:]}\n"
            f"--- stderr tail ---\n{result.stderr[-3000:]}"
        )


# --------------------------------------------------------------------------------------
# MLP — the trainer that actually broke
# --------------------------------------------------------------------------------------


@pytest.mark.skipif(not BATCH_CACHE.is_dir(), reason=f"cache not present at {BATCH_CACHE}")
def test_mlp_batch_trainer_is_reproducible_at_a_fixed_seed(tmp_path):
    out_a = tmp_path / "mlp_a.pt"
    out_b = tmp_path / "mlp_b.pt"
    for out in (out_a, out_b):
        _run_trainer(
            [
                "src/policy/tabular/train_mlp_dim22_from_batch.py",
                "--cache-dir", str(BATCH_CACHE),
                "--output", str(out),
                "--epochs", "2",
                "--random-state", str(SEED),
            ],
            env_extra={},
        )
    a = _weights_of(torch.load(out_a, map_location="cpu", weights_only=False))
    b = _weights_of(torch.load(out_b, map_location="cpu", weights_only=False))
    _assert_state_dicts_identical(a, b, "train_mlp_dim22_from_batch")

    # The stamp is what lets a later gate tell a seeded checkpoint from a drawn one, so a
    # checkpoint that is reproducible but unstamped still fails the study's pre-run assert.
    meta = torch.load(out_a, map_location="cpu", weights_only=False)
    assert meta.get("torch_seeded") is True, "checkpoint does not record torch_seeded"


@pytest.mark.skipif(not BATCH_CACHE.is_dir(), reason=f"cache not present at {BATCH_CACHE}")
def test_mlp_partial_state_flag_refuses_contractless_cache(tmp_path):
    """B2 (route_b stage 2): --partial-state on a cache with no partial_state_contract
    must fail loudly BEFORE extraction — silently training a dim25cr model under a
    dim63crk-declaring flag is exactly the layout-mismatch class the sidecar rules
    exist to prevent. The reproducibility run for the dim63crk path itself is added
    when a B3 stage-2 cache exists to pin."""
    with pytest.raises(AssertionError, match="partial_state_contract"):
        _run_trainer(
            [
                "src/policy/tabular/train_mlp_dim22_from_batch.py",
                "--cache-dir", str(BATCH_CACHE),
                "--output", str(tmp_path / "mlp_ps.pt"),
                "--epochs", "1",
                "--random-state", str(SEED),
                "--candidate-relative-queue",
                "--partial-state",
            ],
            env_extra={},
        )


@pytest.mark.parametrize(
    "trainer",
    [
        "src/policy/tabular/train_mlp.py",
        "src/policy/tabular/train_mlp_ce_reduced.py",
        "src/policy/tabular/train_mlp_dim22_from_seq.py",
        "src/policy/tabular/train_mlp_dim22_from_batch.py",
    ],
)
def test_every_mlp_trainer_seeds_torch(trainer):
    """Static guard: the 2026-08-24 fix reached one of four trainers and stopped there.

    A trainer that seeds only `--random-state` into the split looks correct in review —
    the bug is the *absence* of a line, which is exactly what review misses.
    """
    source = (REPO_ROOT / trainer).read_text()
    assert "torch.manual_seed" in source, (
        f"{trainer} never calls torch.manual_seed, so its weight init comes from OS "
        f"entropy and its checkpoints are unreproducible draws"
    )
    assert '"torch_seeded"' in source, (
        f"{trainer} does not stamp torch_seeded, so a seeded checkpoint cannot be told "
        f"from a drawn one after the fact"
    )


# --------------------------------------------------------------------------------------
# GNN — seeding is not enough; this covers use_deterministic_algorithms
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "trainer",
    [
        "src/notebooks/train_near_rtt.py",
        "src/notebooks/train.py",
        "src/notebooks/train_ram.py",
        "src/notebooks/train_seq.py",
    ],
)
def test_every_gnn_trainer_pins_deterministic_algorithms(trainer):
    """cudnn.deterministic does not reach the GIN autograd nondeterminism; this does."""
    source = (REPO_ROOT / trainer).read_text()
    assert "torch.use_deterministic_algorithms" in source, (
        f"{trainer} sets seeds but not use_deterministic_algorithms; at a fixed seed the "
        f"GIN path still diverges run to run, so a seed sweep through it is uninterpretable"
    )


@pytest.mark.skipif(not TINY_CACHE.is_dir(), reason=f"cache not present at {TINY_CACHE}")
def test_gnn_training_is_bit_identical_at_a_fixed_seed():
    """In-process, via the ablation harness — the one GNN trainer that is importable.

    Proves the seeding half only; see the module docstring for why the
    use_deterministic_algorithms half is a static check instead.

    train_near_rtt.py trains at import time and cannot be exercised this way; it is
    covered by the static checks plus the checkpoint's `train_seed` /
    `deterministic_algorithms` contract fields, which the draw-study gate asserts before
    spending a run.
    """
    import pickle
    import random

    import numpy as np

    from scripts_cosim.gnn_necessity_ablation import AblationModel, train_model

    graphs_path = TINY_CACHE / "graphs.pkl"
    with open(graphs_path, "rb") as handle:
        graphs = pickle.load(handle)
    graphs = graphs[:8]
    assert graphs, "cache produced no graphs"

    device = torch.device("cpu")
    torch.set_num_threads(1)

    def one_run() -> Dict[str, torch.Tensor]:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.use_deterministic_algorithms(True, warn_only=True)
        sample = graphs[0]
        model = AblationModel(
            task_dim=int(sample.task_features.shape[1]),
            plat_dim=int(sample.platform_features.shape[1]),
            edge_dim=int(sample.edge_attr.shape[1]),
            use_gin=True,
            use_node_edges=False,
        ).to(device)
        train_model(model, list(graphs), device, epochs=2)
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    _assert_state_dicts_identical(one_run(), one_run(), "gnn_necessity_ablation (GIN)")


def test_a1_teacher_forced_loss_is_bit_identical_at_a_fixed_seed():
    """route_b stage-2 arm A1: the prefix-conditioned path, seeded.

    Higher nondeterminism risk than the static trainers and worth its own case: the loss
    runs many small forwards, memoizes on a dict keyed by tuples, and reduces with
    logsumexp over a variable-length stack. Iteration order over plans and over the
    topological task order must be deterministic (it is — lists plus a Kahn heap), but
    "must be" is what the 2026-08-24 MLP seed defect also assumed.

    Exercised in-process, since train_near_rtt.py trains at import time.
    """
    import pickle

    import numpy as np

    from src.policy.gnn.gnn_model import TaskPlacementGNN
    from src.policy.gnn.partial_state_edges import make_partial_state_score_fn
    from src.policy.gnn.seq_decode import topological_task_order
    from src.policy.tabular.reduced_features import (
        PARTIAL_STATE_FEATURE_DIM,
        build_partial_state_context_from_graph,
    )

    cache = REPO_ROOT / "simulation_data" / "graphs_cache_route_b_smoke_s_dag" / "graphs.pkl"
    if not cache.exists():
        pytest.skip(f"no --dag-partial-state cache at {cache}")
    with open(cache, "rb") as handle:
        graphs = pickle.load(handle)[:4]

    torch.set_num_threads(1)
    device = torch.device("cpu")

    def one_run():
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        sample = graphs[0]
        model = TaskPlacementGNN(
            task_feature_dim=int(sample.task_features.shape[1]),
            platform_feature_dim=int(sample.platform_features.shape[1]),
            edge_dim=int(sample.edge_attr.shape[1]),
            mp_dag_edges=True,
            task_type_onehot_dim=4,
            partial_state_edge_dim=PARTIAL_STATE_FEATURE_DIM,
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(2):
            for graph in graphs:
                ctx = build_partial_state_context_from_graph(graph)
                ctx.node_caps = graph.partial_state_ctx["node_caps_by_alpha"]["2.0"]
                order = topological_task_order(int(graph.n_tasks), graph.dag_parents)
                score = make_partial_state_score_fn(model, graph, ctx)
                plan_logps = []
                for plan in graph.tied_optimal_logit_plans["2.0"]:
                    committed, logp = {}, torch.zeros((), device=device)
                    for task_idx in order:
                        lp = torch.log_softmax(score(task_idx, committed), dim=-1)
                        logp = logp + lp[int(plan[task_idx])]
                        committed[task_idx] = tuple(
                            int(v)
                            for v in graph.task_logit_to_placement[task_idx][plan[task_idx]]
                        )
                    plan_logps.append(logp)
                loss = -torch.logsumexp(torch.stack(plan_logps), dim=0)
                opt.zero_grad()
                loss.backward()
                opt.step()
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    _assert_state_dicts_identical(one_run(), one_run(), "route_b A1 teacher-forced CE")
