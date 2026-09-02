#!/usr/bin/env python3
"""Per-arm glue for the Phase 3 closed loop: load, replay, save.

The loop itself is arm-agnostic — sample an episode, score it against the greedy
baseline, replay a subsample of its decisions with grad. Only three things differ
between CL-GNN and CL-MLP, and they live here:

  * how the checkpoint becomes a module (and, critically, becomes the *same* module the
    serving path builds — both loaders below are the serving loaders, not re-implementations),
  * how a stored replay payload becomes `logits_per_task`,
  * what a checkpoint file looks like when written back.

`load` deliberately routes through `src.executesimulation.load_gnn_model` and the
`MLPBatchScheduler` loading rules rather than reconstructing an architecture from
hyperparameters. A checkpoint without a `.contract.json` sidecar is not evidence, and
architecture flags that are invisible in weight shapes (`mp_residual`, `mp_node_edges`,
`network_graph_contract`) only exist in that sidecar; a hand-rolled loader here would
silently adopt defaults and train a different model than the one being gated.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class LoadedPolicy:
    model: torch.nn.Module
    device: torch.device
    arm: str


# ---------------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------------


# The two arms declare their contract in different places, and conflating them is how a
# checkpoint comes to serve under a layout it was never trained on. The GNN's lives in a
# `.contract.json` sidecar; the MLP's lives inside the `.pt` itself (that is what
# `MLPBatchScheduler.set_models` reads, and no MLP checkpoint in the tree has a sidecar).
# Either way the rule is the same: a checkpoint that cannot state its own contract is not
# evidence, so refuse it rather than adopt a default.
MLP_REQUIRED_CONTRACT_KEYS = ("input_dim", "inference_feature_layout", "queue_feature_contract")


def require_contract(arm: str, model_path: Path) -> Dict[str, Any]:
    """Return the checkpoint's self-declared contract, or fail loudly."""
    if arm == "gnn":
        sidecar = model_path.with_suffix(".contract.json")
        if not sidecar.exists():
            raise SystemExit(
                f"FAIL LOUD: {model_path} has no .contract.json — architecture flags that "
                "are invisible in weight shapes (mp_residual, mp_node_edges, "
                "network_graph_contract) live only there, and every check downstream "
                "would silently adopt its default."
            )
        return json.loads(sidecar.read_text())
    if arm == "mlp":
        ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
        missing = [k for k in MLP_REQUIRED_CONTRACT_KEYS if k not in ckpt]
        if missing:
            raise SystemExit(
                f"FAIL LOUD: MLP checkpoint {model_path} declares no {missing}. The "
                "layouts give the same tensor shapes different meanings, so load and "
                "forward would both succeed on the wrong one."
            )
        return {k: ckpt[k] for k in MLP_REQUIRED_CONTRACT_KEYS if k in ckpt}
    raise ValueError(f"FAIL LOUD: unknown arm {arm!r}")


def load_policy(arm: str, model_path: Path, space_config: Optional[Dict[str, Any]] = None) -> LoadedPolicy:
    contract = require_contract(arm, model_path)
    if arm == "gnn":
        from src.executesimulation import load_gnn_model

        model, device = load_gnn_model(model_path, space_config)
        return LoadedPolicy(model=model, device=device, arm=arm)
    if arm == "mlp":
        from src.policy.tabular.mlp_model import PointwiseEdgeMLP

        ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
        model = PointwiseEdgeMLP(
            input_dim=int(contract["input_dim"]),
            hidden_dim=int(ckpt.get("hidden_dim", 64)),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        # Serving on CPU by parity with the gates, not for speed.
        return LoadedPolicy(model=model, device=torch.device("cpu"), arm=arm)
    raise ValueError(f"FAIL LOUD: unknown arm {arm!r}")


# ---------------------------------------------------------------------------------
# replay: stored payload -> logits_per_task, WITH grad
# ---------------------------------------------------------------------------------


def replay_logits(policy: LoadedPolicy, payload: Any) -> Sequence[Tensor]:
    """Recompute one decode batch's logits from its stored inputs.

    This is the half of the two-pass estimator that carries the autograd graph. A full
    episode is ~30k decisions across ~7.5k forward passes and cannot hold a graph across
    all of them, so pass 1 ran under `no_grad` and recorded only what it chose; this
    replays a uniform subsample of those forward passes.
    """
    if policy.arm == "gnn":
        graph = payload.to(policy.device) if hasattr(payload, "to") else payload
        return policy.model(graph)
    if policy.arm == "mlp":
        feat, boundaries = payload
        x = feat.to(policy.device).to(torch.float32)
        expected = int(policy.model.input_dim)
        if x.shape[1] != expected:
            raise ValueError(
                f"FAIL LOUD: replay matrix has {x.shape[1]} columns, model expects "
                f"{expected}. The stored payload was built under a different feature layout."
            )
        scores = policy.model(x)
        return [
            scores[start:end] if end > start else scores.new_empty(0)
            for start, end in boundaries
        ]
    raise ValueError(f"FAIL LOUD: unknown arm {policy.arm!r}")


def batch_logprob(
    policy: LoadedPolicy, payload: Any, chosen: Sequence[int], temperature: float
) -> Tensor:
    """Sum of log pi(a_t | s_t) over one replayed decode batch, differentiable.

    Uses the same float64 log-softmax over the candidate axis as the sampler, so the
    replayed value is comparable to the pass-1 record at the tolerance the check in
    `train_closed_loop` applies.
    """
    logits_per_task = replay_logits(policy, payload)
    if len(logits_per_task) < len(chosen):
        raise RuntimeError(
            f"FAIL LOUD: replay produced {len(logits_per_task)} task logit rows but the "
            f"episode recorded {len(chosen)} decisions in this batch. The stored payload "
            "does not correspond to the decode it came from."
        )
    total = None
    for t_idx, idx in enumerate(chosen):
        row = logits_per_task[t_idx].to(torch.float64).reshape(-1)
        if idx >= row.numel():
            raise RuntimeError(
                f"FAIL LOUD: recorded action {idx} is outside the {row.numel()} candidates "
                f"the replay reconstructed for task {t_idx}."
            )
        lp = torch.log_softmax(row / float(temperature), dim=0)[idx]
        total = lp if total is None else total + lp
    if total is None:
        raise RuntimeError("FAIL LOUD: replayed a batch with no recorded decisions")
    return total


# ---------------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------------


def save_policy(
    policy: LoadedPolicy,
    out_path: Path,
    *,
    init_path: Path,
    provenance: Dict[str, Any],
) -> None:
    """Write a checkpoint the serving path can load, sidecar included.

    Closed-loop training never changes the architecture — it only moves the weights the
    warm start came with — so the init's sidecar is carried forward verbatim and the
    closed-loop fields are added alongside. Writing weights without the sidecar would
    produce something that loads and serves *and is not evidence*.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in policy.model.state_dict().items()}
    if policy.arm == "gnn":
        torch.save(state, out_path)
        contract = json.loads(init_path.with_suffix(".contract.json").read_text())
        # `train_seed` on a warm start describes the SUPERVISED run that produced these
        # weights, and stays true of their ancestry. Say so explicitly, so nobody reads a
        # closed-loop checkpoint as a supervised one whose seed is reproducible.
        contract["warm_start_train_seed"] = contract.get("train_seed")
        contract["trained_by"] = "scripts_cosim/closed_loop/train_closed_loop.py"
        contract["closed_loop"] = provenance
        out_path.with_suffix(".contract.json").write_text(json.dumps(contract, indent=2))
    else:
        # The MLP's contract travels inside the .pt, so the init is loaded and its
        # declared fields carried forward with the new weights on top.
        ckpt = torch.load(str(init_path), map_location="cpu", weights_only=False)
        ckpt["model_state_dict"] = state
        ckpt["torch_seeded"] = True
        ckpt["trained_by"] = "scripts_cosim/closed_loop/train_closed_loop.py"
        ckpt["closed_loop"] = provenance
        torch.save(ckpt, out_path)


def copy_checkpoint(src: Path, dst: Path) -> None:
    """Copy a checkpoint together with its sidecar, when it has one.

    GNN checkpoints are never separable from their `.contract.json`. MLP checkpoints
    carry their contract inside the `.pt` and have no sidecar in this tree, so demanding
    one here would block the CL-MLP arm outright — but a sidecar that exists and is left
    behind is the actual failure mode, so copy it whenever it is there.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    src_sidecar = src.with_suffix(".contract.json")
    if src_sidecar.exists():
        shutil.copyfile(src_sidecar, dst.with_suffix(".contract.json"))
