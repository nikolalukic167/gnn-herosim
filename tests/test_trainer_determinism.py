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
from typing import Dict, Optional

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
        [sys.executable, *args], cwd=str(cwd), env=env, capture_output=True, text=True, timeout=120
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
        "src/policy/workflow/train.py",
        "src/policy/residency/train.py",
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
        "src/policy/workflow/train.py",
        "src/policy/residency/train.py",
    ],
)
def test_every_gnn_trainer_pins_deterministic_algorithms(trainer):
    """cudnn.deterministic does not reach the GIN autograd nondeterminism; this does."""
    source = (REPO_ROOT / trainer).read_text()
    assert "torch.use_deterministic_algorithms" in source, (
        f"{trainer} sets seeds but not use_deterministic_algorithms; at a fixed seed the "
        f"GIN path still diverges run to run, so a seed sweep through it is uninterpretable"
    )


def test_the_closed_loop_trainer_seeds_and_pins_determinism():
    """objective_pivot_v1 Phase 3's trainer is a trainer, and this file covers every trainer.

    It is not in either parametrized list above because it fits neither: it trains a GNN
    *or* an MLP through one loop, and it has no cache to train on — its data is episodes
    of the live simulator. The properties it has to have are the same ones, though, and
    it is more exposed than the supervised trainers, not less: with no labels anywhere, a
    trainer whose weight init came from OS entropy would produce a training curve nobody
    could tell from a hard problem.
    """
    trainer = "scripts_cosim/closed_loop/train_closed_loop.py"
    source = (REPO_ROOT / trainer).read_text()
    assert "torch.manual_seed" in source, f"{trainer} never calls torch.manual_seed"
    assert "torch.use_deterministic_algorithms" in source, (
        f"{trainer} seeds but does not pin the nondeterministic autograd kernels"
    )
    assert '"torch_seeded"' in source, (
        f"{trainer} does not stamp torch_seeded, so a seeded closed-loop checkpoint "
        f"cannot be told from a drawn one after the fact"
    )


def test_the_closed_loop_episode_seed_is_derived_not_drawn():
    """Two arms at the same --seed must see the same episode seeds.

    That pairing IS the registered primary statistic (common random numbers). A seed
    taken from the clock, the PID or an unseeded RNG would leave the arms comparable only
    in expectation, at an n the pilot cannot afford.
    """
    source = (REPO_ROOT / "scripts_cosim/closed_loop/train_closed_loop.py").read_text()
    assert "ep_seed = (args.seed" in source, (
        "the episode seed is no longer derived from --seed and the step; CRN pairing "
        "between CL-GNN and CL-MLP is silently gone"
    )
    for forbidden in ("time.time()", "os.getpid()", "random.randint", "uuid"):
        assert forbidden not in source.split("def main")[-1], (
            f"the closed-loop trainer draws from {forbidden}, so its episodes are not "
            f"reproducible and cannot be paired across arms"
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


# --------------------------------------------------------------------------------------
# run_experiment.py --seed, end to end through train_near_rtt.py itself
#
# The tests above exercise trainer LOGIC in-process (gnn_necessity_ablation.AblationModel,
# or a hand-built TaskPlacementGNN loop) precisely because "train_near_rtt.py trains at
# import time and cannot be exercised this way" — but that means none of them go through
# train_near_rtt.py's OWN import chain. That chain had a real defect (found 2026-08-26,
# route_b stage-2 A1 draws): `from src.notebooks.prepare_graphs_cache import
# DAG_TASK_TYPE_VOCAB` at train_near_rtt.py:343 pulled in prepare_graphs_cache.py, whose
# module body used to call `torch.manual_seed(42)` unconditionally at IMPORT time — after
# train_near_rtt.py's own NEAR_RTT_TRAIN_SEED-derived seed block (lines ~103-110), so it
# always ran last and clobbered every requested seed back to 42. All 4 of a route_b A1
# seed sweep (seeds 1-4) produced bit-identical weights and byte-identical wandb summaries
# to full precision; run_experiment.py --seed correctly set the env var (verified via
# --dry-run and the checkpoint sidecar's `train_seed` field), so nothing short of actually
# training end to end through the real import chain would have caught this. Fixed by
# moving prepare_graphs_cache.py's seed calls from module scope into its own `main()` —
# a module must never reseed the global RNG as an import side effect.
# --------------------------------------------------------------------------------------

SMOKE_DAG_CACHE = REPO_ROOT / "simulation_data" / "graphs_cache_route_b_smoke_s_dag"


def _run_a1_via_run_experiment(
    seed: int, tmp_path: Path, env_extra: Optional[Dict[str, str]] = None
) -> Dict[str, torch.Tensor]:
    """One real train_near_rtt.py run through run_experiment.py --seed, on the tiny
    12-graph smoke DAG cache, 1 epoch. Returns the saved checkpoint's tensor weights.

    `env_extra` (drainable_objective_v1) carries label-config env into the run, so a knob
    that changes what the trainer fits is covered by the same determinism gate as the
    seed itself."""
    import shutil

    import yaml

    cfg = yaml.safe_load((REPO_ROOT / "experiments/route_b_stage2_a1.yaml").read_text())
    # Point at the tiny smoke cache and drop the pilot-204 split artifact (it does not
    # cover this cache's parents) so the run is fast and self-contained; everything else
    # (the A1 flags, ce_only objective) stays exactly as registered.
    cfg["cache_dir"] = "simulation_data/graphs_cache_route_b_smoke_s_dag"
    cfg["env"].pop("NEAR_RTT_SPLIT_ARTIFACT", None)
    cfg["args"]["epochs"] = 1
    # This twelve-graph in-process cache needs no forked workers; fork-after-Torch can hang.
    cfg["args"]["num-dataloader-workers"] = 0
    cfg.pop("wandb", None)  # WANDB_MODE=disabled (via _run_trainer's env) makes this moot
    config_path = tmp_path / "a1_smoke.yaml"
    config_path.write_text(yaml.dump(cfg))

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    # train_near_rtt.py writes to Path("models") relative to CWD — run from tmp_path so
    # parallel test runs / repeated seeds never collide with real checkpoints.
    _run_trainer(
        [str(REPO_ROOT / "run_experiment.py"), str(config_path), "--seed", str(seed)],
        {"WANDB_CACHE_DIR": str(tmp_path / "wandb-cache"), "WANDB_DATA_DIR": str(tmp_path / "wandb-data"), **(env_extra or {})},
        cwd=tmp_path,
    )
    # Disabled W&B supplies a random run name; inspect only this test's isolated directory.
    candidates = sorted(
        models_dir.glob("*.pt"),
        key=lambda p: p.stat().st_mtime,
    )
    assert candidates, "no checkpoint written under models/"
    newest = candidates[-1]
    checkpoint = torch.load(str(newest), map_location="cpu", weights_only=False)
    weights = _weights_of(checkpoint)
    newest.unlink()
    sidecar = newest.with_suffix(".contract.json")
    if sidecar.is_file():
        sidecar.unlink()
    return weights


@pytest.mark.skipif(
    not SMOKE_DAG_CACHE.is_dir(), reason=f"cache not present at {SMOKE_DAG_CACHE}"
)
def test_run_experiment_seed_same_seed_is_reproducible_for_a1():
    """Regression-shaped companion to the different-seed test below: same seed twice
    through the REAL run_experiment.py -> train_near_rtt.py path must still agree."""
    import tempfile

    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        w1 = _run_a1_via_run_experiment(4242, Path(d1))
        w2 = _run_a1_via_run_experiment(4242, Path(d2))
    _assert_state_dicts_identical(w1, w2, "run_experiment.py --seed 4242 (A1, twice)")


@pytest.mark.skipif(
    not SMOKE_DAG_CACHE.is_dir(), reason=f"cache not present at {SMOKE_DAG_CACHE}"
)
def test_run_experiment_seed_different_seeds_diverge_for_a1():
    """THE inverse regression test for the import-time reseed bug: two DIFFERENT
    --seed values through the real train_near_rtt.py path must produce DIFFERENT
    weights. Before the fix this failed (seeds 1-4 were bit-identical, see the module
    note above) because prepare_graphs_cache.py's import-time torch.manual_seed(42)
    always ran after train_near_rtt.py's own seed block and won."""
    import tempfile

    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        w1 = _run_a1_via_run_experiment(1, Path(d1))
        w2 = _run_a1_via_run_experiment(2, Path(d2))

    assert set(w1) == set(w2), "parameter names differ between the two runs"
    differing = [k for k in w1 if not torch.equal(w1[k], w2[k])]
    assert differing, (
        "run_experiment.py --seed 1 vs --seed 2 produced BIT-IDENTICAL weights for "
        "train_near_rtt.py (A1) -- the seed had no effect on training. This is the "
        "exact signature of the import-time reseed clobber: something imported after "
        "train_near_rtt.py's own seed block is calling torch.manual_seed with a "
        "constant. Check for module-level random.seed/np.random.seed/torch.manual_seed "
        "calls anywhere on train_near_rtt.py's import path (prepare_graphs_cache.py is "
        "the known offender; a NEW one would reproduce this exact failure)."
    )


@pytest.mark.skipif(
    not SMOKE_DAG_CACHE.is_dir(), reason=f"cache not present at {SMOKE_DAG_CACHE}"
)
def test_label_objective_env_does_not_break_determinism(tmp_path):
    """drainable_objective_v1: a run that declares a shaped label is still bit-identical
    at a fixed seed.

    The smoke cache was built on the one-step label, so the trainer's targets here do not
    move -- what this pins is that the env knob itself introduces no nondeterminism (a
    label transform that iterated a set, say, would surface exactly here). Whether the
    label CHANGES the targets is tested against real sweeps in tests/test_drift_label.py
    and by the cache-build equivalence check recorded in the lineage node.
    """
    env = {
        "NEAR_RTT_LABEL_OBJECTIVE": "rtt_drift:1.0",
        "NEAR_RTT_LABEL_ARRIVAL_RATE": "0.46",
    }
    dir_a, dir_b = tmp_path / "a", tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    first = _run_a1_via_run_experiment(4242, dir_a, env)
    second = _run_a1_via_run_experiment(4242, dir_b, env)
    assert set(first) == set(second)
    for key in first:
        assert torch.equal(first[key], second[key]), f"{key} differs between runs"

@pytest.mark.parametrize('arm', ['gnn', 'mpoff', 'mlp_hand'])
def test_workflow_trainer_same_seed_same_weights(arm):
    from src.policy.workflow.train import seed_everything, train_epoch
    from src.policy.workflow.model import features, WorkflowAssignmentNet
    from src.placement.workflow_planning import instance, rollout, initial
    import numpy as np
    xs=[];es=[];ys=[]
    for seed in (71,72,73,74):
        p=instance(seed,jobs=2,ops=3,machines=3);x,e=features(p,arm=='mlp_hand');labels=np.zeros((2,3),dtype=np.int64)
        for j,k,h in rollout(initial(p),p)[3]:labels[j,k]=h
        xs.append(x);es.append(e);ys.append(labels)
    x=torch.tensor(np.stack(xs));e=torch.tensor(np.stack(es));y=torch.tensor(np.stack(ys));runs=[]
    for _ in range(2):
        seed_everything(SEED)
        model=WorkflowAssignmentNet(x.shape[-1],machines=3,hidden=16,layers=2,mp=arm=='gnn')
        opt=torch.optim.AdamW(model.parameters(),lr=.001)
        generator=torch.Generator().manual_seed(SEED)
        for epoch in range(3):train_epoch(model,opt,x,e,y,2,generator)
        runs.append({k:v.detach().clone() for k,v in model.state_dict().items()})
    _assert_state_dicts_identical(runs[0],runs[1],f'workflow {arm}')

@pytest.mark.parametrize('arm',['gnn','mpoff','mlp_hand'])
def test_mixed_trainer_same_seed_same_weights(arm,tmp_path):
    import numpy as np
    from src.policy.mixed.train import train_epoch,seed_everything
    from src.policy.mixed.model import MixedNet,features
    from src.placement.radical.environment import problem,initial
    from src.placement.radical.mixed import MixedNative
    engine=MixedNative(tmp_path/'build');problems=[problem(s,jobs=2,ops=3) for s in (31,32,33,34)];data=[features(b,arm=='mlp_hand') for b in problems];x=torch.tensor(np.stack([r[0] for r in data]));e=torch.tensor(np.stack([r[1] for r in data]));adj=torch.tensor(np.stack([r[2] for r in data]));y=torch.tensor(np.stack([initial(b,'fastest').ravel() for b in problems]));states=[]
    for _ in range(2):
        seed_everything(SEED);model=MixedNet(x.shape[-1],hidden=8,layers=2,mp=arm=='gnn');opt=torch.optim.AdamW(model.parameters(),lr=.001);generator=torch.Generator().manual_seed(SEED)
        for policy_gradient in (False,True):train_epoch(model,opt,x,e,adj,y,problems,engine,2,generator,policy_gradient)
        states.append({k:v.detach().clone() for k,v in model.state_dict().items()})
    _assert_state_dicts_identical(states[0],states[1],f'mixed {arm}')

@pytest.mark.parametrize('arm',['gnn','mpoff','mlp_hand'])
def test_pair_selector_trainer_is_deterministic(arm):
    import numpy as np
    from src.placement.radical.environment import problem,initial
    from src.policy.pair_selector.model import PairNet,features
    from src.policy.pair_selector.train import train_epoch
    from src.policy.workflow.train import seed_everything
    torch.set_num_threads(1)
    b=problem(17,jobs=3,ops=3)
    x,adj,pair=features(b,initial(b,'affinity'),np.zeros((3,3)),arm=='mlp_hand')
    x=torch.from_numpy(x)[None];adj=torch.from_numpy(adj)[None];pair=torch.from_numpy(pair)[None]
    costs=torch.arange(37).float()[None]+100.;states=[]
    for _ in range(2):
        seed_everything(SEED);model=PairNet(x.shape[-1],nodes=9,mp=arm=='gnn')
        opt=torch.optim.AdamW(model.parameters(),lr=.001)
        for _ in range(2):train_epoch(model,opt,x,adj,pair,costs,torch.Generator().manual_seed(SEED))
        states.append({k:v.clone() for k,v in model.state_dict().items()})
    _assert_state_dicts_identical(*states,'pair-selector-'+arm)


@pytest.mark.parametrize('arm', ['gnn', 'mpoff', 'mlp_hand'])
def test_dispatch_priority_trainer_is_deterministic(arm):
    import numpy as np
    from src.placement.radical.environment import initial, problem
    from src.policy.dispatch_priority.model import PriorityNet, features
    from src.policy.dispatch_priority.train import train_epoch
    from src.policy.workflow.train import seed_everything

    torch.set_num_threads(1)
    physical = problem(29, jobs=3, ops=3)
    x, adjacency = features(physical, initial(physical, 'affinity'), arm == 'mlp_hand')
    x = torch.from_numpy(x)[None]
    adjacency = torch.from_numpy(adjacency)[None]
    target = torch.arange(9).float()[None] / 8
    states = []
    for _ in range(2):
        seed_everything(SEED)
        model = PriorityNet(x.shape[-1], hidden=8, layers=2, mp=arm == 'gnn')
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        generator = torch.Generator().manual_seed(SEED)
        for _ in range(2):
            train_epoch(model, optimizer, x, adjacency, target, generator)
        states.append({key: value.clone() for key, value in model.state_dict().items()})
    _assert_state_dicts_identical(*states, 'dispatch-priority-' + arm)


@pytest.mark.parametrize('arm', ['gnn', 'mpoff', 'mlp_hand'])
def test_dispatch_state_trainer_is_deterministic(arm):
    import numpy as np
    from src.placement.radical.dispatch_state import trace_priority
    from src.placement.radical.environment import initial, problem
    from src.policy.dispatch_priority.model import features
    from src.policy.dispatch_state.model import ReadySetNet
    from src.policy.dispatch_state.train import train_epoch
    from src.policy.workflow.train import seed_everything

    physical = problem(39, jobs=3, ops=3)
    assignment = initial(physical, 'affinity')
    x, adjacency = features(physical, assignment, arm == 'mlp_hand')
    priority = np.arange(assignment.size).reshape(assignment.shape)
    _, rows = trace_priority(physical, assignment, priority)
    tensors = [torch.from_numpy(value)[None] for value in
               (x, adjacency, np.stack([row['dynamic'] for row in rows]),
                np.stack([row['feasible'] for row in rows]),
                np.asarray([row['target'] for row in rows]))]
    states = []
    for _ in range(2):
        seed_everything(SEED)
        model = ReadySetNet(x.shape[-1], hidden=8, layers=2, mp=arm == 'gnn')
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        generator = torch.Generator().manual_seed(SEED)
        for _ in range(2):
            train_epoch(model, optimizer, tensors, generator)
        states.append({key: value.clone() for key, value in model.state_dict().items()})
    _assert_state_dicts_identical(*states, 'dispatch-state-' + arm)


@pytest.mark.parametrize('arm', ['gnn', 'mpoff', 'mlp_hand'])
def test_dispatch_value_trainer_is_deterministic(arm):
    from src.policy.dispatch_value.model import ReadySetNet
    from src.policy.dispatch_value.train import train_epoch
    from src.policy.workflow.train import seed_everything

    torch.manual_seed(SEED)
    x = torch.randn(3, 6, 5)
    adjacency = torch.randn(3, 4, 6, 6)
    dynamic = torch.randn(3, 4, 6, 13)
    feasible = torch.zeros(3, 4, 6, dtype=torch.bool)
    feasible[..., :3] = True
    costs = torch.full((3, 4, 6), float('nan'))
    costs[..., :3] = torch.tensor([10.0, 12.0, 15.0])
    rule = torch.full((3,), 15.0)
    tensors = [x, adjacency, dynamic, feasible, costs, rule]
    states = []
    for _ in range(2):
        seed_everything(SEED)
        model = ReadySetNet(5, hidden=8, layers=2, mp=arm == 'gnn')
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        generator = torch.Generator().manual_seed(SEED)
        for _ in range(2):
            train_epoch(model, optimizer, tensors, generator)
        states.append({key: value.clone() for key, value in model.state_dict().items()})
    _assert_state_dicts_identical(*states, 'dispatch-value-' + arm)


@pytest.mark.parametrize('arm', ['gnn', 'mpoff', 'hand_mlp'])
def test_dag_memory_trainer_is_deterministic(arm):
    from src.policy.dag_memory.model import MemoryValueNet
    from src.policy.dag_memory.train import seed_everything, train_epoch

    torch.manual_seed(SEED)
    tensors = {'x': torch.randn(3, 8, 12),
               'graph': torch.randn(3, 4, 8, 8),
               'mask': torch.ones(3, 8),
               'target': torch.randn(3, 8)}
    states = []
    for _ in range(2):
        seed_everything(SEED)
        model = MemoryValueNet(12, hidden=8, layers=2, arm=arm)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        generator = torch.Generator().manual_seed(SEED)
        for _ in range(2):
            train_epoch(model, optimizer, tensors, generator, 2)
        states.append({key: value.detach().clone() for key, value in model.state_dict().items()})
    _assert_state_dicts_identical(*states, 'dag-memory-' + arm)


@pytest.mark.parametrize('arm', ['gnn', 'mpoff', 'hand_mlp'])
def test_residency_trainer_is_deterministic(arm):
    from src.policy.residency.model import ResidencyNet
    from src.policy.residency.train import seed_everything, train_epoch

    torch.manual_seed(SEED)
    x = torch.randn(9, 8, 3, 20)
    q = torch.rand(9, 3)
    states = []
    for _ in range(2):
        seed_everything(SEED)
        model = ResidencyNet(arm=arm, hidden=8, layers=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        generator = torch.Generator().manual_seed(SEED)
        for _ in range(2):
            train_epoch(model, optimizer, x, q, generator, batch_size=3)
        states.append({key: value.detach().clone() for key, value in model.state_dict().items()})
    _assert_state_dicts_identical(*states, 'residency-' + arm)
