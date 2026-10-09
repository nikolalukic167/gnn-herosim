"""r1_attribution_v1 graph-free arms: MLP-same and Set-transformer (src/policy/gnn/arm_models.py).

What is pinned:

1. Same inputs. The GNN, MLP-same and Set-transformer read their per-edge columns through one function
   (``gather_scorable_edges``), and MLP-same's flat vector is exactly [task | platform | edge_attr | partial state].
2. No graph. Neither arm's encoding reads edge_index, edge_attr or the partial-state block; a set-transformer
   task embedding does not change when the candidate edges are scrambled; encodes are permutation-equivariant.
3. Train/serve parity. A checkpoint written the way the trainer writes it loads through the serving loader and
   reproduces the trained model's logits bit for bit.
4. Refusals. A sidecar that names another arm than its weights, a graph flag on a graph-free arm, and a serving
   environment that asks for a different arm all fail loudly.
5. Seeded training is bit-identical (a supervised loop on synthetic graphs; the end-to-end trainer check is the smoke
   job, scripts_cosim/datalab/r1a_arms_smoke.sbatch).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.policy.gnn.arm_models import (  # noqa: E402
    ARM_GNN, ARM_MLP_SAME, ARM_SET_TRANSFORMER, TaskPlacementMLPSame, TaskPlacementSetTransformer,
    arm_kind_from_state_dict, build_graph_free_arm, infer_graph_free_dims, require_arm_matches_weights,
)
from src.policy.gnn.gnn_model import TaskPlacementGNN, gather_scorable_edges  # noqa: E402

N_TASKS, N_PLATFORMS, TASK_DIM, PLAT_DIM, EDGE_DIM, PS_DIM, ONEHOT = 4, 6, 5, 16, 5, 27, 4


@pytest.fixture(autouse=True)
def _restore_environment():
    snapshot = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(snapshot)


def make_graph(seed: int = 0, n_tasks: int = N_TASKS, n_platforms: int = N_PLATFORMS) -> Data:
    g = torch.Generator().manual_seed(seed)
    fwd = [(t, n_tasks + p) for t in range(n_tasks) for p in range(n_platforms) if (t + p) % 3 != 0 or p == t]
    src = [a for a, _ in fwd] + [b for _, b in fwd]
    dst = [b for _, b in fwd] + [a for a, _ in fwd]
    ei = torch.tensor([src, dst], dtype=torch.long)
    n_edges = ei.size(1)
    data = Data(
        task_features=torch.randn(n_tasks, TASK_DIM, generator=g),
        platform_features=torch.randn(n_platforms, PLAT_DIM, generator=g),
        edge_index=ei,
        edge_attr=torch.randn(n_edges, EDGE_DIM, generator=g),
    )
    data.partial_state_edge_attr = torch.randn(n_edges, PS_DIM, generator=g)
    data.task_type_onehot4 = torch.nn.functional.one_hot(torch.arange(n_tasks) % ONEHOT, ONEHOT).float()
    data.n_tasks = n_tasks
    data.n_platforms = n_platforms
    return data


def make_arm(kind: str, seed: int = 0):
    torch.manual_seed(seed)
    return build_graph_free_arm(
        kind, task_feature_dim=TASK_DIM, platform_feature_dim=PLAT_DIM, embedding_dim=32, hidden_dim=48,
        num_layers=2, dropout=0.0, task_type_onehot_dim=ONEHOT, partial_state_edge_dim=PS_DIM, heads=4, inducing=3,
    ).eval()


def make_gnn(seed: int = 0, disable_mp: bool = True) -> TaskPlacementGNN:
    os.environ["GNN_DISABLE_MESSAGE_PASSING"] = "1" if disable_mp else "0"
    torch.manual_seed(seed)
    return TaskPlacementGNN(
        task_feature_dim=TASK_DIM, platform_feature_dim=PLAT_DIM, embedding_dim=32, hidden_dim=48, num_layers=2,
        dropout=0.0, post_gin_dropout=0.0, task_type_onehot_dim=ONEHOT, partial_state_edge_dim=PS_DIM,
    ).eval()


# 1. same inputs ----------------------------------------------------------------------

def test_every_arm_reads_the_same_edge_columns():
    g = make_graph()
    ti, pj, e_attr = gather_scorable_edges(g, PS_DIM)
    assert e_attr.shape == (ti.numel(), EDGE_DIM + PS_DIM)
    captured = {}

    def grab(name):
        def hook(_module, args):
            captured[name] = args[2].detach().clone()
        return hook

    gnn, st = make_gnn(), make_arm(ARM_SET_TRANSFORMER)
    gnn.edge_scorer.register_forward_pre_hook(grab("gnn"))
    st.edge_scorer.register_forward_pre_hook(grab("set"))
    gnn(g)
    st(g)
    assert torch.equal(captured["gnn"], e_attr) and torch.equal(captured["set"], e_attr)


def test_mlp_same_input_is_task_platform_edge_and_partial_state():
    g = make_graph()
    arm = make_arm(ARM_MLP_SAME)
    task_in, plat_in = arm._encode(g)
    ti, pj, x = arm.edge_inputs(task_in, plat_in, g)
    expect = torch.cat([
        torch.cat([g.task_features, g.task_type_onehot4], dim=-1)[ti], g.platform_features[pj],
        g.edge_attr[: g.edge_index.size(1)][(g.edge_index[0] < N_TASKS)], g.partial_state_edge_attr[(g.edge_index[0] < N_TASKS)],
    ], dim=-1)
    assert x.shape[1] == TASK_DIM + ONEHOT + PLAT_DIM + EDGE_DIM + PS_DIM
    assert torch.equal(x, expect)


def test_scores_are_one_logit_per_candidate_edge():
    g = make_graph()
    for kind in (ARM_MLP_SAME, ARM_SET_TRANSFORMER):
        logits = make_arm(kind)(g)
        counts = [int(((g.edge_index[0] == t) & (g.edge_index[1] >= N_TASKS)).sum()) for t in range(N_TASKS)]
        assert [int(x.numel()) for x in logits] == counts and all(torch.isfinite(x).all() for x in logits)


# 2. no graph -------------------------------------------------------------------------

def test_encode_does_not_read_edges_or_partial_state():
    g = make_graph()
    h = make_graph()
    h.edge_index = h.edge_index[:, torch.randperm(h.edge_index.size(1), generator=torch.Generator().manual_seed(3))]
    h.edge_attr = torch.zeros_like(h.edge_attr)
    h.partial_state_edge_attr = torch.zeros_like(h.partial_state_edge_attr)
    for kind in (ARM_MLP_SAME, ARM_SET_TRANSFORMER):
        arm = make_arm(kind)
        for a, b in zip(arm._encode(g), arm._encode(h)):
            assert torch.equal(a, b), f"{kind}: encode read the edges or the partial state"


def test_set_transformer_is_equivariant_in_tasks_and_platforms():
    g = make_graph()
    arm = make_arm(ARM_SET_TRANSFORMER)
    task, plat = arm._encode(g)
    tp = torch.tensor([2, 0, 3, 1])
    pp = torch.tensor([5, 3, 1, 0, 4, 2])
    h = make_graph()
    h.task_features, h.task_type_onehot4 = g.task_features[tp], g.task_type_onehot4[tp]
    h.platform_features = g.platform_features[pp]
    task2, plat2 = arm._encode(h)
    assert torch.allclose(task2, task[tp], atol=1e-5) and torch.allclose(plat2, plat[pp], atol=1e-5)


def test_set_transformer_context_reaches_every_member_and_mlp_same_has_none():
    g = make_graph()
    h = make_graph()
    h.task_features = g.task_features.clone()
    h.task_features[3] += 5.0
    st, mlp = make_arm(ARM_SET_TRANSFORMER), make_arm(ARM_MLP_SAME)
    assert not torch.allclose(st._encode(g)[0][0], st._encode(h)[0][0]), "no set context between tasks"
    assert torch.equal(mlp._encode(g)[0][0], mlp._encode(h)[0][0])
    assert torch.equal(mlp(g)[0], mlp(h)[0]), "MLP-same scored task 0 differently when task 3 changed"


def test_set_transformer_pools_by_mean_not_sum():
    """Duplicating a whole set leaves a softmax-attention average unchanged; a sum would double it."""
    g = make_graph()
    arm = make_arm(ARM_SET_TRANSFORMER)
    block = arm.task_blocks[0].mab
    x = arm.task_encoder(torch.cat([g.task_features, g.task_type_onehot4], dim=-1)).unsqueeze(0)
    once = block.attn(x, x, x, need_weights=False)[0]
    twice = block.attn(x, torch.cat([x, x], dim=1), torch.cat([x, x], dim=1), need_weights=False)[0]
    assert torch.allclose(once, twice, atol=1e-5)


# 3. train/serve parity ---------------------------------------------------------------

def _write_checkpoint(tmp_path: Path, kind: str, model) -> Path:
    from src.policy.gnn.prefix_serving import _task_type_vocab
    from src.placement.cache_physics import current_physics_env

    path = tmp_path / f"{kind}.pt"
    torch.save(model.state_dict(), path)
    sidecar = {
        "arm_kind": kind, "partial_state_edge_features": True, "partial_state_contract": "partial_state_v5",
        "partial_state_feature_dim": PS_DIM, "peer_mass": True, "load_seconds": True, "exchange_seconds": True,
        "task_type_onehot_dim": ONEHOT, "dag_task_type_vocab": list(_task_type_vocab()), "dag_alpha_key": "inf",
        "task_feature_dim": TASK_DIM, "platform_feature_dim": PLAT_DIM, "edge_dim": EDGE_DIM,
        "set_heads": 4 if kind == ARM_SET_TRANSFORMER else None, "disable_message_passing": False,
        "physics_env": current_physics_env(), "candidate_slate": None, "feature_dim": None,
    }
    path.with_suffix(".contract.json").write_text(json.dumps(sidecar))
    return path


@pytest.mark.parametrize("kind", [ARM_MLP_SAME, ARM_SET_TRANSFORMER])
def test_checkpoint_round_trips_through_the_serving_loader_bit_for_bit(tmp_path, kind):
    from src.policy.gnn.prefix_serving import load_prefix_conditioned_gnn

    for name in ("GNN_DISABLE_MESSAGE_PASSING", "GNN_ARM_KIND", "PARTIAL_STATE_CONTRACT"):
        os.environ.pop(name, None)
    trained = make_arm(kind, seed=7)
    ckpt = _write_checkpoint(tmp_path, kind, trained)
    served, options, sidecar = load_prefix_conditioned_gnn(ckpt, device=torch.device("cpu"))
    assert type(served) is type(trained) and sidecar["arm_kind"] == kind and options.alpha_key == "inf"
    g = make_graph(seed=11)
    for a, b in zip(trained(g), served(g)):
        assert torch.equal(a, b)
    assert served.partial_state_edge_dim == PS_DIM and served.task_type_onehot_dim == ONEHOT


def test_dims_are_recovered_from_weights_and_sidecar():
    sd = make_arm(ARM_SET_TRANSFORMER).state_dict()
    sc = {"partial_state_feature_dim": PS_DIM, "edge_dim": EDGE_DIM, "set_heads": 4}
    dims = infer_graph_free_dims(ARM_SET_TRANSFORMER, sd, ONEHOT, sc)
    assert dims["task_feature_dim"] == TASK_DIM and dims["platform_feature_dim"] == PLAT_DIM
    assert (dims["embedding_dim"], dims["hidden_dim"], dims["num_layers"], dims["heads"], dims["inducing"]) == (32, 48, 2, 4, 3)
    sd = make_arm(ARM_MLP_SAME).state_dict()
    sc = {"partial_state_feature_dim": PS_DIM, "edge_dim": EDGE_DIM, "task_feature_dim": TASK_DIM, "platform_feature_dim": PLAT_DIM}
    assert infer_graph_free_dims(ARM_MLP_SAME, sd, ONEHOT, sc)["hidden_dim"] == 48
    with pytest.raises(ValueError, match="first layer takes"):
        infer_graph_free_dims(ARM_MLP_SAME, sd, ONEHOT, {**sc, "platform_feature_dim": PLAT_DIM + 1})


# 4. refusals -------------------------------------------------------------------------

def test_arm_is_told_from_the_weights():
    assert arm_kind_from_state_dict(make_arm(ARM_MLP_SAME).state_dict()) == ARM_MLP_SAME
    assert arm_kind_from_state_dict(make_arm(ARM_SET_TRANSFORMER).state_dict()) == ARM_SET_TRANSFORMER
    assert arm_kind_from_state_dict(make_gnn().state_dict()) == ARM_GNN


def test_sidecar_and_weights_must_name_the_same_arm():
    for sidecar_arm, sd_arm in ((ARM_MLP_SAME, ARM_SET_TRANSFORMER), (ARM_SET_TRANSFORMER, ARM_MLP_SAME), (ARM_GNN, ARM_MLP_SAME)):
        with pytest.raises(ValueError, match="weights are a"):
            require_arm_matches_weights({"arm_kind": sidecar_arm}, make_arm(sd_arm).state_dict(), "x.pt")
    with pytest.raises(ValueError, match="weights are a"):
        require_arm_matches_weights({}, make_arm(ARM_MLP_SAME).state_dict(), "legacy.pt")
    assert require_arm_matches_weights({}, make_gnn().state_dict(), "legacy.pt") == ARM_GNN


def test_graph_flags_on_a_graph_free_arm_are_refused():
    sd = make_arm(ARM_MLP_SAME).state_dict()
    for flag in ("mp_peer_edges", "mp_bipartite_edge_conv", "plan_raw", "disable_message_passing", "mp_node_edges"):
        with pytest.raises(ValueError, match="has no graph"):
            require_arm_matches_weights({"arm_kind": ARM_MLP_SAME, flag: True}, sd, "x.pt")


def test_serving_environment_cannot_ask_for_another_arm(monkeypatch):
    sd = make_arm(ARM_SET_TRANSFORMER).state_dict()
    monkeypatch.setenv("GNN_ARM_KIND", ARM_MLP_SAME)
    with pytest.raises(ValueError, match="GNN_ARM_KIND"):
        require_arm_matches_weights({"arm_kind": ARM_SET_TRANSFORMER}, sd, "x.pt")
    monkeypatch.setenv("GNN_ARM_KIND", ARM_SET_TRANSFORMER)
    assert require_arm_matches_weights({"arm_kind": ARM_SET_TRANSFORMER}, sd, "x.pt") == ARM_SET_TRANSFORMER


def test_loader_refuses_a_mismatched_sidecar(tmp_path):
    from src.policy.gnn.prefix_serving import PrefixServingError, load_prefix_conditioned_gnn

    for name in ("GNN_DISABLE_MESSAGE_PASSING", "GNN_ARM_KIND", "PARTIAL_STATE_CONTRACT"):
        os.environ.pop(name, None)
    ckpt = _write_checkpoint(tmp_path, ARM_SET_TRANSFORMER, make_arm(ARM_SET_TRANSFORMER))
    sc = json.loads(ckpt.with_suffix(".contract.json").read_text())
    sc["arm_kind"] = ARM_MLP_SAME
    ckpt.with_suffix(".contract.json").write_text(json.dumps(sc))
    with pytest.raises(PrefixServingError, match="weights are a"):
        load_prefix_conditioned_gnn(ckpt, device=torch.device("cpu"))


def test_trainer_arm_flag_is_validated():
    from src.policy.gnn.arm_models import resolve_arm_kind

    assert resolve_arm_kind("") == ARM_GNN and resolve_arm_kind("mlp_same") == ARM_MLP_SAME
    with pytest.raises(ValueError, match="NEAR_RTT_ARM"):
        resolve_arm_kind("transformer")


# 5. determinism ----------------------------------------------------------------------

def train_arm_once(kind: str, seed: int = 4242):
    """A short seeded supervised run on synthetic graphs; tests/test_trainer_determinism.py calls this too."""
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(1)
    graphs = [make_graph(seed=s, n_tasks=3 + s % 2, n_platforms=5 + s) for s in range(4)]
    torch.manual_seed(seed)
    model = build_graph_free_arm(
        kind, task_feature_dim=TASK_DIM, platform_feature_dim=PLAT_DIM, embedding_dim=32, hidden_dim=48,
        num_layers=2, dropout=0.1, task_type_onehot_dim=ONEHOT, partial_state_edge_dim=PS_DIM, heads=4, inducing=3,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(3):
        for g in graphs:
            loss = sum(-torch.log_softmax(x, -1)[0] for x in model(g) if x.numel())
            opt.zero_grad()
            loss.backward()
            opt.step()
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


@pytest.mark.parametrize("kind", [ARM_MLP_SAME, ARM_SET_TRANSFORMER])
def test_seeded_training_is_bit_identical(kind):
    a, b = train_arm_once(kind), train_arm_once(kind)
    assert set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)
    c = train_arm_once(kind, seed=4243)
    assert any(not torch.equal(a[k], c[k]) for k in a), "a different seed gave the same weights: the seed does nothing"


# 6. configs and sidecheck ----------------------------------------------------------------

def test_committed_configs_are_what_the_generator_writes():
    import subprocess

    rc = subprocess.run([sys.executable, str(REPO_ROOT / "scripts_cosim/make_r1a_arm_configs.py"), "--check"],
                        capture_output=True, text=True)
    assert rc.returncode == 0, rc.stderr


def test_every_arm_has_the_same_six_point_grid_and_the_epoch_cap():
    import yaml

    grids = {}
    for arm in ("gnn_eng", "twin_eng", "mlp_same", "gnn_raw", "twin_raw", "gnn_eng_physmp", "set_transformer"):
        points = []
        for k in range(6):
            cfg = yaml.safe_load((REPO_ROOT / f"experiments/r1_attribution_v1_{arm}_g{k}.yaml").read_text())
            assert cfg["args"]["epochs"] <= 100 and cfg["args"]["patience"] == 60 and cfg["lineage"] == "r1_attribution_v1"
            assert cfg["env"]["PARTIAL_STATE_CONTRACT"] == "partial_state_v5"
            points.append((cfg["args"]["learning-rate"], cfg["args"]["embedding-dim"], cfg["args"]["hidden-dim"]))
        grids[arm] = points
        assert len(set(points)) == 6
        assert (REPO_ROOT / f"experiments/r1_attribution_v1_{arm}_smoke.yaml").is_file()
    assert len({tuple(v) for v in grids.values()}) == 1, "arms must share one grid (Errica et al. fair-budget rule)"


def _good_sidecar(arm: str) -> dict:
    sys.path.insert(0, str(REPO_ROOT / "scripts_cosim"))
    import joint_burst_v2_sidecheck as sc_mod

    kind, mp_off, raw, conv, phys = sc_mod.RA_ARMS[f"ra_{arm}"]
    graph_free = kind != "gnn"
    return {
        "arm_kind": kind, "disable_message_passing": mp_off, "partial_state_contract": "partial_state_v5", "load_seconds": True,
        "exchange_seconds": True, "full_context_ce_weight": 1.0, "dag_alpha_key": "inf", "mp_residual": False, "plan_raw": raw,
        "plan_raw_sum": False, "plan_raw_local": False, "mp_bipartite_edge_conv": conv and not graph_free,
        "mp_bipartite_edge_attr_zero": conv and not phys and not graph_free, "mp_peer_edges": not graph_free,
        "partial_state_feature_dim": 2 if raw else 27, "candidate_slate": "declared_pruning_v1", "physics_env": {"x": "y"},
        "set_heads": 4 if kind == "set_transformer" else None, "split_artifact": {"sha256": "abc"},
    }


@pytest.mark.parametrize("arm", ["gnn_eng", "twin_eng", "mlp_same", "gnn_raw", "twin_raw", "gnn_eng_physmp", "set_transformer"])
def test_sidecheck_accepts_each_arm_and_refuses_its_neighbours(tmp_path, arm):
    import hashlib

    sys.path.insert(0, str(REPO_ROOT / "scripts_cosim"))
    import joint_burst_v2_sidecheck as sc_mod

    split = tmp_path / "split.json"
    split.write_text("{}")
    good = _good_sidecar(arm)
    good["split_artifact"] = {"sha256": hashlib.sha256(split.read_bytes()).hexdigest()}
    assert sc_mod.check_ra(good, f"ra_{arm}", str(split), "inf") == 0
    others = [a for a in ("gnn_eng", "twin_eng", "mlp_same", "gnn_raw", "twin_raw", "gnn_eng_physmp", "set_transformer") if a != arm]
    for other in others:
        assert sc_mod.check_ra(good, f"ra_{other}", str(split), "inf") == 1, f"a {arm} sidecar passed as {other}"


def test_loader_refuses_an_inflight_capture_mismatch(tmp_path, monkeypatch):
    from src.policy.gnn.prefix_serving import PrefixServingError, load_prefix_conditioned_gnn

    for name in ("GNN_DISABLE_MESSAGE_PASSING", "GNN_ARM_KIND", "PARTIAL_STATE_CONTRACT", "HEROSIM_INFLIGHT_CAPTURE"):
        monkeypatch.delenv(name, raising=False)
    ckpt = _write_checkpoint(tmp_path, ARM_MLP_SAME, make_arm(ARM_MLP_SAME))
    load_prefix_conditioned_gnn(ckpt, device=torch.device("cpu"))  # legacy sidecar, legacy run: fine
    monkeypatch.setenv("HEROSIM_INFLIGHT_CAPTURE", "service_end_v1")
    with pytest.raises(PrefixServingError, match="inflight_capture"):
        load_prefix_conditioned_gnn(ckpt, device=torch.device("cpu"))
