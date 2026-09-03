#!/usr/bin/env python3
from __future__ import annotations

"""
Train non-unique task placement with near-optimal exact RTT ranking.

The older structured regret loss mostly sampled catastrophic negatives. This
trainer uses valid_combos_map.pkl from prepare_graphs_cache_near_rtt.py and
samples heavily from near-optimal RTT bands so the loss remains informative
after the model has learned to avoid obviously bad placements.
"""

import gc
import itertools
import json
import os
import random
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_NOTEBOOKS_DIR = Path(__file__).resolve().parent
if str(_NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOKS_DIR))
# runpy launches this file with sys.path[0] = src/notebooks, so absolute `src.*` imports
# need the repo root explicitly (same treatment as prepare_graphs_cache.py).
_REPO_ROOT = _NOTEBOOKS_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from non_unique_lib.training_contract import (  # noqa: E402
    assert_split_artifact_covers,
    assert_zero_parent_overlap,
    canonical_parent_id,
    load_split_artifact,
    split_ids_by_canonical_parent,
)
from torch import Tensor
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn.models import GIN
from tqdm import tqdm
import wandb

from non_unique_lib.cache_io import (
    ExactRttLookupMap,
    PlacementToLogitMap,
    build_capped_valid_combos_map_from_chunked_cache,
    build_exact_rtt_index_lookups,
    build_valid_combos_map_from_chunked_cache,
    create_cache_context,
    load_capped_valid_combos_map,
    load_graphs_from_cache,
    load_optimal_rtt_from_cache,
    load_valid_combos_map,
    save_capped_valid_combos_map,
    save_valid_combos_map,
)
from non_unique_lib.soft_combo_loss import concentration_penalty, soft_combo_ce_loss
from non_unique_lib.training_config import parse_training_config
from src.placement.network_graph import (
    NETWORK_GRAPH_CONTRACT_OFF,
    resolve_network_graph_contract,
)
from src.placement.corpus_provenance import derive_corpus_provenance
from src.placement.queue_features import (
    DEFAULT_QUEUE_FEATURE_CONTRACT,
    queue_depth_norm,
    usage_ratio_feature,
    validate_queue_feature_contract,
)
from src.placement.topology_features import resolve_topology_feature_contract
from src.policy.tabular.constants import (
    CACHE_VERSION as ATOMIC_CACHE_VERSION,
    PLATFORM_FEATURE_DIM,
    TASK_FEATURE_DIM,
)


PlacementCombo = Tuple[Tuple[int, int], ...]
RttByCombo = Dict[str, Dict[PlacementCombo, float]]


def parent_dataset_id(dataset_id: Any) -> str:
    return str(dataset_id or "").split("@seq", 1)[0]


def lookup_dataset_id(data: Data) -> str:
    parent_id = getattr(data, "parent_dataset_id", None)
    if parent_id:
        return parent_dataset_id(parent_id)
    return parent_dataset_id(getattr(data, "dataset_id", ""))


# Init/shuffling seed only. The canonical-parent split is seeded separately (random_state=42
# at the split call) and must stay fixed, so varying this measures weight-init variance alone.
_TRAIN_SEED = int(os.environ.get("NEAR_RTT_TRAIN_SEED", "42"))
random.seed(_TRAIN_SEED)
np.random.seed(_TRAIN_SEED)
torch.manual_seed(_TRAIN_SEED)
torch.cuda.manual_seed_all(_TRAIN_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Seeding alone does NOT make this trainer reproducible. Measured 2026-08-19 in
# gnn_necessity_ablation.py: at a fixed seed on CPU the GIN autograd path diverges run to
# run (mean_ce 0.9604 vs 0.9601 at epoch 5) — it is not cudnn, not intra-op threading
# (OMP_NUM_THREADS=1 still diverges) and not PYTHONHASHSEED. cudnn.deterministic above
# cannot reach it. Without this line a seed sweep conflates seed effects with run-to-run
# noise, and the noise is the larger of the two.
_NONDETERMINISTIC = os.environ.get("NEAR_RTT_NONDETERMINISTIC", "").strip().lower() in (
    "1",
    "true",
    "yes",
)
if not _NONDETERMINISTIC:
    torch.use_deterministic_algorithms(True, warn_only=True)


@dataclass(frozen=True)
class NearRttConfig:
    loss_variant: str = os.environ.get("NEAR_RTT_LOSS_VARIANT", "near-rtt-v1")
    sidecar_name: str = os.environ.get("NEAR_RTT_SIDECAR_NAME", "valid_combos_near_rtt_capped.pkl")
    top_k_decode: int = int(os.environ.get("NEAR_RTT_TOP_K", "5"))
    pairs_per_graph: int = int(os.environ.get("NEAR_RTT_PAIRS_PER_GRAPH", "32"))
    near_margin_floor: float = float(os.environ.get("NEAR_RTT_MARGIN_FLOOR", "0.05"))
    margin_cap: float = float(os.environ.get("NEAR_RTT_MARGIN_CAP", "1.0"))
    margin_mode: str = os.environ.get("NEAR_RTT_MARGIN_MODE", "linear")
    margin_exp_scale: float = float(os.environ.get("NEAR_RTT_MARGIN_EXP_SCALE", "0.75"))
    margin_exp_clip: float = float(os.environ.get("NEAR_RTT_MARGIN_EXP_CLIP", "4.0"))
    trash_delta: float = float(os.environ.get("NEAR_RTT_TRASH_DELTA", "5.0"))
    near_weight: float = float(os.environ.get("NEAR_RTT_NEAR_WEIGHT", "3.0"))
    close_weight: float = float(os.environ.get("NEAR_RTT_CLOSE_WEIGHT", "2.0"))
    mid_weight: float = float(os.environ.get("NEAR_RTT_MID_WEIGHT", "1.0"))
    far_weight: float = float(os.environ.get("NEAR_RTT_FAR_WEIGHT", "0.25"))
    trash_weight: float = float(os.environ.get("NEAR_RTT_TRASH_WEIGHT", "0.0"))
    dropout: float = float(os.environ.get("NEAR_RTT_DROPOUT", "0.1"))
    train_all: bool = os.environ.get("NEAR_RTT_TRAIN_ALL", "0") == "1"
    unmapped_penalty: float = float(os.environ.get("NEAR_RTT_UNMAPPED_PENALTY", "1.0"))
    train_objective: str = os.environ.get("NEAR_RTT_TRAIN_OBJECTIVE", "").strip().lower()
    soft_combo_tau: float = float(os.environ.get("NEAR_RTT_SOFT_COMBO_TAU", "0.25"))
    soft_combo_max_combos: int = int(os.environ.get("NEAR_RTT_SOFT_COMBO_MAX_COMBOS", "4096"))
    conc_gamma: float = float(os.environ.get("NEAR_RTT_CONC_GAMMA", "0.02"))
    conc_cap: float = float(os.environ.get("NEAR_RTT_CONC_CAP", "1.5"))
    # Message-passing architecture. Both are recorded in the checkpoint contract sidecar
    # so serving cannot silently diverge from what was trained (see save_checkpoint).
    mp_residual: bool = os.environ.get("NEAR_RTT_MP_RESIDUAL", "0") == "1"
    mp_node_edges: bool = os.environ.get("NEAR_RTT_MP_NODE_EDGES", "0") == "1"
    mp_node_edges_candidates_only: bool = (
        os.environ.get("NEAR_RTT_MP_NODE_EDGES_CANDIDATES_ONLY", "1") == "1"
    )
    # Network entities (physical nodes + core links + route edges). Off unless the cache
    # was built with them; $NETWORK_GRAPH_CONTRACT names which graph that was, and the
    # sidecar records it so serving resolves the same one.
    mp_network_entities: bool = os.environ.get("NEAR_RTT_MP_NETWORK_ENTITIES", "0") == "1"

    # route_b stage 2 (arm A1, the genuine T2 GNN). All three default OFF, so an
    # unflagged run is bit-identical to today's trainer.
    #   mp_dag_edges         workload-DAG task<->task edges into message passing
    #   task_type_onehot     the 4-way task-type one-hot (a fairness repair: the T1 MLP
    #                        already sees task type via krank), and a hard prerequisite
    #                        of mp_dag_edges
    #   partial_state_edges  per-step prefix conditioning + teacher-forced any-of-K CE
    mp_dag_edges: bool = os.environ.get("NEAR_RTT_MP_DAG_EDGES", "0") == "1"
    task_type_onehot: bool = os.environ.get("NEAR_RTT_TASK_TYPE_ONEHOT", "0") == "1"
    partial_state_edges: bool = os.environ.get("NEAR_RTT_PARTIAL_STATE_EDGES", "0") == "1"
    dag_alpha_key: str = os.environ.get("NEAR_RTT_DAG_ALPHA_KEY", "2.0")
    # 0 = use every tied-optimal plan. Any other value CHANGES THE LOSS DEFINITION, so
    # it is recorded in the sidecar and applied deterministically (first N in cache
    # order — never a random sample).
    tied_max_plans: int = int(os.environ.get("NEAR_RTT_TIED_MAX_PLANS", "0"))
    # B6: path to the shared split artifact (scripts_cosim/make_split_artifact.py).
    # When set, the trainer loads the pinned parent-level split instead of drawing
    # one, so a "draw" varies initialisation and batch order ONLY (§3). Every arm of
    # a paired comparison must point at the same file.
    split_artifact: str = os.environ.get("NEAR_RTT_SPLIT_ARTIFACT", "").strip()


_DEFAULT_NEAR_RTT_WANDB_PROJECT = "gnn-near-rtt-jun2026"

RUNTIME_CONFIG = parse_training_config()
if "--wandb-project" not in sys.argv:
    RUNTIME_CONFIG = replace(
        RUNTIME_CONFIG, wandb_project=_DEFAULT_NEAR_RTT_WANDB_PROJECT
    )
CACHE_CTX = create_cache_context(RUNTIME_CONFIG.cache_dir)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NEAR_CFG = NearRttConfig()

EMBEDDING_DIM = RUNTIME_CONFIG.embedding_dim
HIDDEN_DIM = RUNTIME_CONFIG.hidden_dim
LEARNING_RATE = RUNTIME_CONFIG.learning_rate
BATCH_SIZE = RUNTIME_CONFIG.batch_size
NUM_GIN_LAYERS = RUNTIME_CONFIG.num_gin_layers
WEIGHT_DECAY = RUNTIME_CONFIG.weight_decay
EPOCHS = RUNTIME_CONFIG.epochs
RTT_SCALE_FACTOR = RUNTIME_CONFIG.rtt_scale_factor
REGRET_LOSS_WEIGHT = RUNTIME_CONFIG.regret_loss_weight
CE_LOSS_WEIGHT = RUNTIME_CONFIG.ce_loss_weight
TRAIN_OBJECTIVE = NEAR_CFG.train_objective or ("ce_only" if REGRET_LOSS_WEIGHT <= 0.0 else "ranking")
SOFT_COMBO_TRAINING = TRAIN_OBJECTIVE in {"soft_combo", "soft_combo_conc"}
CONCENTRATION_TRAINING = TRAIN_OBJECTIVE == "soft_combo_conc"
CE_ONLY_TRAINING = TRAIN_OBJECTIVE == "ce_only"
# route_b stage 2 arm A1: prefix conditioning implies the teacher-forced any-of-K CE.
TEACHER_FORCED = NEAR_CFG.partial_state_edges
if TEACHER_FORCED and TRAIN_OBJECTIVE != "ce_only":
    raise ValueError(
        f"NEAR_RTT_PARTIAL_STATE_EDGES=1 registers arm A1 as CE-only, but "
        f"TRAIN_OBJECTIVE resolves to {TRAIN_OBJECTIVE!r}. The teacher-forced any-of-K "
        "CE is the registered objective; a ranking/regret term is not part of it."
    )
if TEACHER_FORCED and not NEAR_CFG.mp_dag_edges:
    raise ValueError(
        "NEAR_RTT_PARTIAL_STATE_EDGES=1 without NEAR_RTT_MP_DAG_EDGES=1 is arm A3 "
        "(pointwise scoring under a masked decoder) wearing A1's name — the model would "
        "get the prefix columns but no graph structure. Set NEAR_RTT_MP_DAG_EDGES=1."
    )
NUM_DATALOADER_WORKERS = RUNTIME_CONFIG.num_dataloader_workers
PHASE_B_CHECKPOINT_METRIC = os.environ.get(
    "NEAR_RTT_PHASE_B_CHECKPOINT_METRIC",
    "seq_reforward_regret" if bool(os.environ.get("TRAIN_INIT_CHECKPOINT")) and not CE_ONLY_TRAINING else "",
).strip().lower()

if (
    SOFT_COMBO_TRAINING
    and CE_LOSS_WEIGHT <= 0.0
    and os.environ.get("NEAR_RTT_ALLOW_UNANCHORED_SOFT_COMBO", "0") != "1"
):
    raise ValueError(
        "soft_combo objectives require a CE anchor (ce_loss_weight > 0). "
        "Set NEAR_RTT_ALLOW_UNANCHORED_SOFT_COMBO=1 only for explicit ablations."
    )

print(f"Cache directory: {CACHE_CTX.cache_dir}")
print(f"Device: {DEVICE}")
print(f"Near RTT config: {NEAR_CFG}")

_feature_dim: Optional[int] = None
_queue_feature_contract = DEFAULT_QUEUE_FEATURE_CONTRACT
_corpus_provenance: Optional[Dict[str, Any]] = None
# The platform-feature layout is a property of the CACHE's feature construction, not of
# whatever the training shell happened to export. Reading it from the environment (as this
# script did until 2026-08-23) writes `null` into the sidecar whenever an sbatch forgets to
# export it — which is exactly how the tempfix and prefixctl checkpoints came to declare no
# layout and then serve as `atomic21` while the deployed checkpoint served `dim22`, a
# difference worth up to 40.8% of live total_rtt. The cache already records the answer.
_inference_feature_layout: Optional[str] = None
_topology_feature_contract: Optional[str] = None
_queue_norm_mode: Optional[str] = None
_required_cache_version = os.environ.get("NEAR_RTT_REQUIRE_CACHE_VERSION", "").strip()
_metadata_path = CACHE_CTX.cache_dir / "metadata.json"
if _metadata_path.exists():
    with open(_metadata_path, "r", encoding="utf-8") as _mf:
        _cache_meta = json.load(_mf)
    _cache_version = _cache_meta.get("cache_version") or _cache_meta.get("version")
    # Caches older than CACHE_VERSION 5.7 predate the field and are legacy_v0 by construction.
    _queue_feature_contract = validate_queue_feature_contract(
        _cache_meta.get("queue_feature_contract") or DEFAULT_QUEUE_FEATURE_CONTRACT
    )
    if _required_cache_version and _cache_version != _required_cache_version:
        raise ValueError(
            f"Cache version mismatch: metadata has {_cache_version!r}, "
            f"expected {_required_cache_version!r}"
        )
    _feature_dim = _cache_meta.get("feature_dim")
    if _required_cache_version == ATOMIC_CACHE_VERSION and _feature_dim not in (None, 21):
        raise ValueError(f"Expected feature_dim=21 in cache metadata, got {_feature_dim!r}")
    _cache_layout = (_cache_meta.get("inference_feature_layout") or "").strip().lower()
    _env_layout = os.environ.get("INFERENCE_FEATURE_LAYOUT", "").strip().lower()
    if _cache_layout and _env_layout and _cache_layout != _env_layout:
        raise ValueError(
            f"Cache {CACHE_CTX.cache_dir} was built with inference_feature_layout="
            f"{_cache_layout!r} but this run exports INFERENCE_FEATURE_LAYOUT="
            f"{_env_layout!r}. The layouts assign different meanings to the same platform "
            "columns; training against one and recording the other produces a checkpoint "
            "that serves wrong numbers with no error."
        )
    _inference_feature_layout = _cache_layout or _env_layout or None
    # Same rule for the topology contract: the cache's value wins, a conflicting shell
    # export is an error, and a cache that predates the field leaves it to the resolver.
    _cache_topo = (_cache_meta.get("topology_feature_contract") or "").strip()
    _env_topo = os.environ.get("TOPOLOGY_FEATURE_CONTRACT", "").strip()
    if _cache_topo and _env_topo and _cache_topo != _env_topo:
        raise ValueError(
            f"Cache {CACHE_CTX.cache_dir} was built under topology_feature_contract="
            f"{_cache_topo!r} but this run exports TOPOLOGY_FEATURE_CONTRACT="
            f"{_env_topo!r}. Task dim 2 means different quantities under the two contracts."
        )
    _topology_feature_contract = _cache_topo or None
    _queue_norm_mode = _cache_meta.get("queue_norm_mode")
    print(
        f"Cache metadata: version={_cache_version}, feature_dim={_feature_dim}, "
        f"queue_feature_contract={_queue_feature_contract}, "
        f"inference_feature_layout={_inference_feature_layout}, "
        f"topology_feature_contract={_topology_feature_contract or '(pre-field cache)'}, "
        f"queue_norm_mode={_queue_norm_mode}"
    )
    # Which infrastructure the corpus spans, for the checkpoint sidecar. Derived from the
    # cache's own dataset list so it cannot drift from the data it describes.
    _corpus_provenance = derive_corpus_provenance(_cache_meta)
    print(
        f"Corpus provenance: {_corpus_provenance.get('n_datasets')} datasets, "
        f"{_corpus_provenance.get('client_node_count')}c/"
        f"{_corpus_provenance.get('server_node_count')}s, "
        f"warmth={_corpus_provenance.get('warmth_physics')}"
    )
elif _required_cache_version:
    raise FileNotFoundError(
        f"NEAR_RTT_REQUIRE_CACHE_VERSION={_required_cache_version!r} but metadata.json missing"
    )


# The model definition lives in src/policy/gnn/gnn_model.py and is IMPORTED, not copied.
# This file used to declare its own TaskPlacementGNN; the two copies drifted and the
# served model ended up message-passing over a graph its weights had never seen
# (2026-08-16: same-node edges outnumbered bipartite ~30:1, 87.5% of argmax decisions
# flipped, 12.4x live RTT on sparse_p35). One definition, imported by both sides, is the
# structural fix. Do not re-declare these classes here.
from src.policy.gnn.gnn_model import TaskPlacementGNN  # noqa: E402
from src.policy.gnn.partial_state_edges import (  # noqa: E402
    make_partial_state_score_fn,
)
from src.policy.gnn.seq_decode import (  # noqa: E402
    decode_masked_topo_placement,
    topological_task_order,
)
from src.policy.tabular.reduced_features import (  # noqa: E402
    PARTIAL_STATE_FEATURE_DIM,
    build_partial_state_context_from_graph,
    resolve_partial_state_contract,
)

# The 4-way DAG task-type vocabulary. Imported rather than restated so a vocab change
# cannot silently disagree with the cache that built the one-hot.
from src.notebooks.prepare_graphs_cache import DAG_TASK_TYPE_VOCAB  # noqa: E402

DAG_TASK_TYPE_ONEHOT_DIM = len(DAG_TASK_TYPE_VOCAB)


def combo_score(logits_per_task: List[Tensor], indices: List[int]) -> Tensor:
    score = torch.zeros((), device=logits_per_task[0].device)
    for task_idx, logit_idx in enumerate(indices):
        if task_idx >= len(logits_per_task) or logit_idx >= logits_per_task[task_idx].numel():
            return score.new_tensor(float("nan"))
        score = score + logits_per_task[task_idx][logit_idx]
    return score


def loss_original_ce(logits_per_task: List[Tensor], data: Data, device: torch.device) -> Tuple[Tensor, int]:
    loss_total = torch.zeros((), device=device)
    valid_tasks = 0
    for task_idx, logits_t in enumerate(logits_per_task):
        if logits_t.numel() == 0:
            continue
        target = data.y[task_idx].long()
        if target.item() < 0 or target.item() >= logits_t.numel():
            continue
        loss_total = loss_total + F.cross_entropy(logits_t.unsqueeze(0), target.view(1))
        valid_tasks += 1
    return loss_total / max(1, valid_tasks), valid_tasks


def _prefix_free_prefix_block(data: Data) -> None:
    """Zero the prefix block so a plain ``model(data)`` is well-defined for arm A1.

    Used only for the static per-task metrics (top-1, greedy) so they stay comparable
    across arms. A1's actual decision rule is the prefix-conditioned masked_topo decode;
    an all-zero prefix is NOT that, and nothing that gates the arm may read these.
    """
    n_edges = int(data.edge_index.size(1))
    data.partial_state_edge_attr = torch.zeros(
        (n_edges, PARTIAL_STATE_FEATURE_DIM),
        dtype=torch.float32,
        device=data.edge_index.device,
    )


def _masked_topo_regret_for_graph(
    model: nn.Module, data: Data
) -> Optional[Tuple[int, ...]]:
    """Arm A1's real decision: the §4 shared masked decoder driven by the per-step
    prefix-conditioned scorer. Returns the decoded combo, or None if the decode failed
    (which §4 forbids relaxing — a failure stays a failure)."""
    alpha_key = str(
        NEAR_CFG.dag_alpha_key or getattr(data, "dag_primary_alpha_key", "2.0")
    )
    ctx = build_partial_state_context_from_graph(data)
    ctx.node_caps = data.partial_state_ctx["node_caps_by_alpha"][alpha_key]
    n_tasks = int(data.n_tasks)
    demands = {
        t: [float(ctx.demand[(t, tuple(int(v) for v in c))])
            for c in data.task_logit_to_placement[t]]
        for t in range(n_tasks)
    }
    return decode_masked_topo_placement(
        [torch.empty(0)] * n_tasks,
        data.task_logit_to_placement,
        n_tasks,
        dag_parents=data.dag_parents,
        node_caps=ctx.node_caps,
        demands=demands,
        score_fn=make_partial_state_score_fn(model, data, ctx),
    )


def loss_tied_teacher_forced_ce(
    model: nn.Module, data: Data, device: torch.device
) -> Tuple[Tensor, int]:
    """The §5 any-of-K marginalized CE for arm A1: ``-log Σ_k Π_t p_t^{(k)}``.

    Tasks are walked in the SAME topological order the §4 masked decoder uses
    (``topological_task_order``, imported not re-typed), the prefix at each step is
    plan k's own committed placements, and each step's logits come from
    ``make_partial_state_score_fn`` — the same closure the decoder calls, so train-time
    and decode-time prefixes agree by construction rather than by two implementations
    staying in sync.

    Two costs are collapsed:
      * one GIN pass per graph, reused across every step and plan (valid because the
        prefix columns enter at the EdgeScorer only — asserted in the score_fn), and
      * a prefix trie: tied plans share prefixes, so a step is scored once per DISTINCT
        (task, committed-prefix) rather than once per (plan, task).

    Note on dropout: the shared encode means every step of every plan sees ONE dropout
    draw on the node embeddings. That is deliberate — a single consistent graph
    representation per graph per batch — but it is a real difference from scoring each
    step independently, so it is stated rather than discovered.
    """
    alpha_key = str(
        NEAR_CFG.dag_alpha_key or getattr(data, "dag_primary_alpha_key", "2.0")
    )
    tied = data.tied_optimal_logit_plans
    if alpha_key not in tied:
        raise ValueError(
            f"loss_tied_teacher_forced_ce: alpha_key {alpha_key!r} not in "
            f"tied_optimal_logit_plans {sorted(tied)} for graph "
            f"{getattr(data, 'dataset_id', '?')}"
        )
    plans = tied[alpha_key]
    if not plans:
        return torch.zeros((), device=device), 0
    if NEAR_CFG.tied_max_plans > 0:
        # Deterministic: the first N in cache order. Never a random sample — this
        # changes the loss definition and is recorded in the sidecar.
        plans = plans[: NEAR_CFG.tied_max_plans]

    caps_by_alpha = data.partial_state_ctx["node_caps_by_alpha"]
    if alpha_key not in caps_by_alpha:
        raise ValueError(
            f"loss_tied_teacher_forced_ce: alpha_key {alpha_key!r} not in "
            f"node_caps_by_alpha {sorted(caps_by_alpha)}"
        )
    ctx = build_partial_state_context_from_graph(data)
    ctx.node_caps = caps_by_alpha[alpha_key]

    n_tasks = int(data.n_tasks)
    order = topological_task_order(n_tasks, data.dag_parents)
    score = make_partial_state_score_fn(model, data, ctx)
    placements = data.task_logit_to_placement

    memo: Dict[Tuple[Any, ...], Tensor] = {}

    def log_probs(task_idx: int, committed: Dict[int, Tuple[int, int]]) -> Tensor:
        key = (task_idx, tuple(sorted(committed.items())))
        cached = memo.get(key)
        if cached is None:
            cached = F.log_softmax(score(task_idx, committed), dim=-1)
            memo[key] = cached
        return cached

    plan_logps: List[Tensor] = []
    for plan in plans:
        committed: Dict[int, Tuple[int, int]] = {}
        logp = torch.zeros((), device=device)
        for t in order:
            lp = log_probs(t, committed)
            idx = int(plan[t])
            if idx < 0 or idx >= lp.numel():
                raise ValueError(
                    f"loss_tied_teacher_forced_ce: tied plan logit index {idx} out of "
                    f"range for task {t} ({lp.numel()} candidates)"
                )
            logp = logp + lp[idx]
            committed[t] = tuple(int(v) for v in placements[t][idx])
        plan_logps.append(logp)

    # -log Σ_k Π_t p: any tied-optimal member counts as correct (§5).
    return -torch.logsumexp(torch.stack(plan_logps), dim=0), n_tasks


class NearRttRankingLoss(nn.Module):
    def __init__(
        self,
        exact_rtt_map: ExactRttLookupMap,
        rtt_scale: float,
        cfg: NearRttConfig,
    ) -> None:
        super().__init__()
        self.exact_rtt_map = exact_rtt_map
        self.rtt_scale = max(float(rtt_scale), 1e-9)
        self.cfg = cfg

    def _sample_band(
        self,
        rows: List[Tuple[List[int], float]],
        opt_rtt: float,
        low: float,
        high: float,
        count: int,
    ) -> List[Tuple[List[int], float]]:
        band = [row for row in rows if low < row[1] - opt_rtt <= high]
        if not band:
            return []
        if len(band) <= count:
            return band
        return random.sample(band, count)

    def _margin_for_gap(self, gap: float) -> float:
        scaled_gap = max(0.0, float(gap) / self.rtt_scale)
        if self.cfg.margin_mode == "exp":
            clipped = min(scaled_gap, max(0.0, self.cfg.margin_exp_clip))
            margin = self.cfg.near_margin_floor + self.cfg.margin_exp_scale * (np.exp(clipped) - 1.0)
        else:
            margin = scaled_gap
        return min(self.cfg.margin_cap, max(self.cfg.near_margin_floor, float(margin)))

    def forward(
        self,
        logits_per_task: List[Tensor],
        data: Data,
        device: torch.device,
    ) -> Tuple[Tensor, int, Dict[str, Any]]:
        rows = self.exact_rtt_map.get(lookup_dataset_id(data), [])
        if len(rows) < 2:
            return torch.zeros((), device=device), 0, {}

        opt_indices, opt_rtt = rows[0]
        opt_score = combo_score(logits_per_task, opt_indices)
        if torch.isnan(opt_score):
            return torch.zeros((), device=device), 0, {}

        # Make the post-epoch plateau visible to the loss: most pairs come from
        # <=0.3s regret, with a few mid/far pairs retained to avoid regressions.
        n = max(4, self.cfg.pairs_per_graph)
        sampled: List[Tuple[List[int], float, float]] = []
        for indices, rtt in self._sample_band(rows, opt_rtt, 0.0, 0.05, max(1, n // 4)):
            sampled.append((indices, rtt, self.cfg.near_weight))
        for indices, rtt in self._sample_band(rows, opt_rtt, 0.05, 0.30, max(1, n // 2)):
            sampled.append((indices, rtt, self.cfg.close_weight))
        for indices, rtt in self._sample_band(rows, opt_rtt, 0.30, 1.00, max(1, n // 4)):
            sampled.append((indices, rtt, self.cfg.mid_weight))
        for indices, rtt in self._sample_band(rows, opt_rtt, 1.00, self.cfg.trash_delta, max(1, n // 8)):
            sampled.append((indices, rtt, self.cfg.far_weight))
        if self.cfg.trash_weight > 0.0:
            for indices, rtt in self._sample_band(rows, opt_rtt, self.cfg.trash_delta, float("inf"), max(1, n // 8)):
                sampled.append((indices, rtt, self.cfg.trash_weight))

        if not sampled:
            return torch.zeros((), device=device), 0, {}

        loss = torch.zeros((), device=device)
        active = 0
        used = 0
        for neg_indices, neg_rtt, weight in sampled:
            neg_score = combo_score(logits_per_task, neg_indices)
            if torch.isnan(neg_score):
                continue
            gap = max(0.0, float(neg_rtt) - float(opt_rtt))
            margin = self._margin_for_gap(gap)
            pair_loss = F.softplus(neg_score - opt_score + margin) * float(weight)
            if pair_loss.item() > 1e-12:
                active += 1
            loss = loss + pair_loss
            used += 1

        if used == 0:
            return torch.zeros((), device=device), 0, {}

        return loss / used, 1, {
            "pairs": used,
            "active_pairs": active,
            "opt_rtt": float(opt_rtt),
        }


class GraphRttDataset(torch.utils.data.Dataset):
    def __init__(self, graphs: List[Data], dataset_ids: List[str], optimal_rtt_map: Dict[str, float]) -> None:
        self.graphs = graphs
        self.dataset_ids = dataset_ids
        self.optimal_rtt_map = optimal_rtt_map

    def __len__(self) -> int:
        return len(self.dataset_ids)

    def __getitem__(self, idx: int) -> Data:
        graph = self.graphs[idx]
        dataset_id = self.dataset_ids[idx]
        parent_id = parent_dataset_id(getattr(graph, "parent_dataset_id", None) or dataset_id)
        graph.dataset_id = dataset_id
        graph.parent_dataset_id = parent_id
        graph.opt_rtt = float(
            self.optimal_rtt_map.get(dataset_id, self.optimal_rtt_map.get(parent_id, 0.0))
        )
        graph.task_logit_to_placement = getattr(
            graph,
            "task_logit_to_placement",
            getattr(graph, "_task_logit_to_placement", {}),
        )
        return graph


def collate_graphs(items: List[Data]) -> List[Data]:
    return items


def create_loader(dataset: GraphRttDataset, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_DATALOADER_WORKERS,
        collate_fn=collate_graphs,
        persistent_workers=NUM_DATALOADER_WORKERS > 0 and RUNTIME_CONFIG.persistent_dataloader_workers,
    )


def decode_greedy(logits_per_task: List[Tensor], data: Data) -> Optional[PlacementCombo]:
    mapping = getattr(data, "task_logit_to_placement", getattr(data, "_task_logit_to_placement", None))
    if mapping is None:
        return None
    combo: List[Tuple[int, int]] = []
    for task_idx, logits_t in enumerate(logits_per_task):
        if logits_t.numel() == 0 or task_idx not in mapping:
            return None
        idx = int(logits_t.argmax().item())
        if idx >= len(mapping[task_idx]):
            return None
        combo.append(tuple(mapping[task_idx][idx]))
    return tuple(combo)


def _queue_norm_from_values(queue_values: List[int]) -> float:
    mode = os.environ.get("NEAR_RTT_SEQ_VAL_QUEUE_NORM_MODE", os.environ.get("GNN_QUEUE_NORM_MODE", "scheduler_adaptive")).strip()
    return queue_depth_norm(
        queue_values,
        mode,
        _queue_feature_contract,
        fixed_factor=float(os.environ.get("NEAR_RTT_SEQ_VAL_QUEUE_NORM_FACTOR", "50.0")),
    )


def _require_seq_val_metadata(data: Data) -> Tuple[Dict[int, List[Tuple[int, int]]], Dict[int, List[str]], Dict[str, Dict[str, Any]], Dict[str, int]]:
    mapping = getattr(data, "task_logit_to_placement", getattr(data, "_task_logit_to_placement", None))
    keys_map = getattr(data, "task_logit_to_queue_key", None)
    meta = getattr(data, "queue_key_to_platform_meta", None)
    queue_snapshot = getattr(data, "queue_snapshot", None)
    if not mapping or not keys_map or not meta or queue_snapshot is None:
        dataset_id = getattr(data, "dataset_id", "<unknown>")
        raise RuntimeError(
            f"Sequential validation requires placement, queue-key, metadata, and queue snapshot fields; missing on {dataset_id}."
        )
    return mapping, keys_map, meta, {str(k): int(v) for k, v in dict(queue_snapshot).items()}


def _refresh_queue_dependent_platform_features(
    data: Data,
    live_queues: Dict[str, int],
    meta: Dict[str, Dict[str, Any]],
) -> None:
    platform_features = data.platform_features
    if platform_features.size(-1) < 14:
        raise RuntimeError(
            f"Sequential validation expects 14-dim platform features; got {platform_features.size(-1)}."
        )
    atomic21 = _feature_dim == 21
    queue_norm: Optional[float] = None
    if not atomic21:
        queue_values = [int(live_queues.get(str(key), 0)) for key in meta.keys()]
        queue_norm = _queue_norm_from_values(queue_values)
    for queue_key, info in meta.items():
        if "platform_pos" not in info:
            raise RuntimeError(f"Sequential validation metadata missing platform_pos for {queue_key}.")
        pos = int(info["platform_pos"])
        if pos < 0 or pos >= int(data.n_platforms):
            raise RuntimeError(f"Sequential validation platform_pos out of range for {queue_key}: {pos}.")
        raw_q = float(live_queues.get(str(queue_key), 0))
        if atomic21:
            platform_features[pos, 7] = raw_q
            platform_features[pos, 13] = 0.0
        else:
            target_concurrency = max(float(info.get("target_concurrency", 1.0)), 1e-9)
            platform_features[pos, 7] = raw_q / float(queue_norm)
            platform_features[pos, 13] = usage_ratio_feature(
                raw_q, target_concurrency, _queue_feature_contract
            )


@torch.no_grad()
def decode_sequential_reforward(
    model: nn.Module,
    data: Data,
) -> Optional[PlacementCombo]:
    mapping, keys_map, meta, live_queues = _require_seq_val_metadata(data)
    combo: List[Tuple[int, int]] = []
    original_platform_features = data.platform_features
    data.platform_features = original_platform_features.clone()
    try:
        _refresh_queue_dependent_platform_features(data, live_queues, meta)
        for task_idx in range(int(data.n_tasks)):
            if task_idx not in mapping or task_idx not in keys_map:
                return None
            logits_per_task = model(data)
            if task_idx >= len(logits_per_task):
                return None
            logits_t = logits_per_task[task_idx]
            if logits_t.numel() == 0:
                return None
            idx = int(logits_t.argmax().item())
            if idx >= len(mapping[task_idx]) or idx >= len(keys_map[task_idx]):
                return None
            queue_key = str(keys_map[task_idx][idx])
            combo.append(tuple(mapping[task_idx][idx]))
            live_queues[queue_key] = live_queues.get(queue_key, 0) + 1
            _refresh_queue_dependent_platform_features(data, live_queues, meta)
    finally:
        data.platform_features = original_platform_features
    return tuple(combo)


def build_worst_regret_by_dataset(rtt_by_dataset: RttByCombo) -> Dict[str, float]:
    """Per-dataset max regret in the capped sidecar; used as unmapped-combo penalty floor."""
    worst: Dict[str, float] = {}
    for dataset_id, combos in rtt_by_dataset.items():
        if not combos:
            continue
        opt_rtt = min(rtt for rtt in combos.values())
        worst[dataset_id] = max(float(rtt) - opt_rtt for rtt in combos.values())
    return worst


def regret_for_combo(
    combo: Optional[PlacementCombo],
    rtt_map: Dict[PlacementCombo, float],
    opt_rtt: float,
    worst_regret: float,
    unmapped_penalty: float,
) -> Optional[float]:
    if combo is None:
        return None
    if combo in rtt_map:
        return float(rtt_map[combo]) - opt_rtt
    return max(float(worst_regret), float(unmapped_penalty))


def decode_topk_joint(
    logits_per_task: List[Tensor],
    data: Data,
    rtt_by_combo: Dict[PlacementCombo, float],
    k: int,
) -> Tuple[Optional[PlacementCombo], Optional[PlacementCombo]]:
    mapping = getattr(data, "task_logit_to_placement", getattr(data, "_task_logit_to_placement", None))
    if mapping is None:
        return None, None

    choices: List[List[Tuple[int, float, Tuple[int, int]]]] = []
    for task_idx, logits_t in enumerate(logits_per_task):
        if logits_t.numel() == 0 or task_idx not in mapping:
            return None, None
        top_k = min(k, logits_t.numel(), len(mapping[task_idx]))
        values, indices = torch.topk(logits_t.float(), k=top_k)
        choices.append([
            (int(idx.item()), float(val.item()), tuple(mapping[task_idx][int(idx.item())]))
            for val, idx in zip(values, indices)
        ])

    best_model_combo: Optional[PlacementCombo] = None
    best_model_score = float("-inf")
    best_oracle_combo: Optional[PlacementCombo] = None
    best_oracle_rtt = float("inf")

    for product in itertools.product(*choices):
        combo = tuple(item[2] for item in product)
        rtt = rtt_by_combo.get(combo)
        if rtt is None:
            continue
        score = sum(item[1] for item in product)
        if score > best_model_score:
            best_model_score = score
            best_model_combo = combo
        if rtt < best_oracle_rtt:
            best_oracle_rtt = rtt
            best_oracle_combo = combo

    return best_model_combo, best_oracle_combo


def move_graph_to_device(data: Data, device: torch.device) -> Data:
    saved = {
        "dataset_id": getattr(data, "dataset_id", None),
        "parent_dataset_id": getattr(data, "parent_dataset_id", None),
        "opt_rtt": getattr(data, "opt_rtt", None),
        "queue_snapshot": getattr(data, "queue_snapshot", None),
        "task_logit_to_queue_key": getattr(data, "task_logit_to_queue_key", None),
        "queue_key_to_platform_meta": getattr(data, "queue_key_to_platform_meta", None),
        "task_logit_to_placement": getattr(
            data,
            "task_logit_to_placement",
            getattr(data, "_task_logit_to_placement", {}),
        ),
        # route_b stage-2 DAG block. These are plain Python containers, so `.to(device)`
        # drops them; the tensors alongside them (dag_edge_index, task_type_onehot4,
        # partial_state_edge_attr) move on their own and need no entry here.
        "dag_parents": getattr(data, "dag_parents", None),
        "partial_state_ctx": getattr(data, "partial_state_ctx", None),
        "tied_optimal_logit_plans": getattr(data, "tied_optimal_logit_plans", None),
        "tied_optimal_rtts": getattr(data, "tied_optimal_rtts", None),
        "node_caps_by_alpha": getattr(data, "node_caps_by_alpha", None),
        "dag_primary_alpha_key": getattr(data, "dag_primary_alpha_key", None),
        "dag_task_type_vocab": getattr(data, "dag_task_type_vocab", None),
    }
    data = data.to(device)
    for key, value in saved.items():
        setattr(data, key, value)
    if TEACHER_FORCED:
        # A cache built without --dag-partial-state must not degrade into an arm that
        # trains on an all-zero prefix block; say so here rather than 40 epochs later.
        missing = [
            name
            for name in ("partial_state_ctx", "tied_optimal_logit_plans", "dag_parents")
            if getattr(data, name, None) is None
        ]
        if missing:
            raise ValueError(
                f"FAIL LOUD: NEAR_RTT_PARTIAL_STATE_EDGES=1 but graph "
                f"{getattr(data, 'dataset_id', '?')} is missing {missing}. Rebuild the "
                "cache with prepare_graphs_cache.py --dag-partial-state."
            )
    return data


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: NearRttRankingLoss,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    running_ce = 0.0
    running_rank = 0.0
    running_combo = 0.0
    running_conc = 0.0
    running_total = 0.0
    steps = 0
    valid_rank = 0
    valid_combo = 0
    valid_conc = 0
    active_pairs = 0
    total_pairs = 0
    combo_count_total = 0.0
    combo_model_regret_total = 0.0
    combo_model_entropy_total = 0.0
    conc_max_load_total = 0.0
    conc_mean_cap_total = 0.0

    for batch in tqdm(loader, desc=f"Epoch {epoch:3d} [Train]", leave=False):
        optimizer.zero_grad()
        loss_ce_total = torch.zeros((), device=DEVICE)
        loss_rank_total = torch.zeros((), device=DEVICE)
        loss_combo_total = torch.zeros((), device=DEVICE)
        loss_conc_total = torch.zeros((), device=DEVICE)
        n_ce = 0
        n_rank = 0
        n_combo = 0
        n_conc = 0

        for graph in batch:
            data = move_graph_to_device(graph, DEVICE)
            if TEACHER_FORCED:
                # Arm A1: no single static forward exists — each step is scored under
                # its own prefix inside the loss.
                logits = None
                loss_ce, valid_ce = loss_tied_teacher_forced_ce(model, data, DEVICE)
            else:
                logits = model(data)
                loss_ce, valid_ce = loss_original_ce(logits, data, DEVICE)
            if valid_ce > 0 and torch.isfinite(loss_ce):
                loss_ce_total = loss_ce_total + loss_ce
                n_ce += 1

            if TRAIN_OBJECTIVE == "ranking":
                loss_rank, valid, stats = criterion(logits, data, DEVICE)
                if valid > 0 and torch.isfinite(loss_rank):
                    loss_rank_total = loss_rank_total + loss_rank
                    n_rank += 1
                    valid_rank += 1
                    active_pairs += int(stats.get("active_pairs", 0))
                    total_pairs += int(stats.get("pairs", 0))

            if SOFT_COMBO_TRAINING:
                loss_combo, valid_combo_graph, combo_stats = soft_combo_ce_loss(
                    logits,
                    data,
                    EXACT_RTT_MAP,
                    tau=NEAR_CFG.soft_combo_tau,
                    max_combos=NEAR_CFG.soft_combo_max_combos,
                )
                if valid_combo_graph > 0 and torch.isfinite(loss_combo):
                    loss_combo_total = loss_combo_total + loss_combo
                    n_combo += 1
                    valid_combo += 1
                    combo_count_total += float(combo_stats.get("combos", 0.0))
                    combo_model_regret_total += float(combo_stats.get("model_regret", 0.0))
                    combo_model_entropy_total += float(combo_stats.get("model_entropy", 0.0))

                if CONCENTRATION_TRAINING:
                    loss_conc, valid_conc_graph, conc_stats = concentration_penalty(
                        logits,
                        data,
                        cap=NEAR_CFG.conc_cap,
                    )
                    if valid_conc_graph > 0 and torch.isfinite(loss_conc):
                        loss_conc_total = loss_conc_total + loss_conc
                        n_conc += 1
                        valid_conc += 1
                        conc_max_load_total += float(conc_stats.get("max_expected_load", 0.0))
                        conc_mean_cap_total += float(conc_stats.get("mean_adaptive_cap", 0.0))

        if n_ce == 0 and n_rank == 0 and n_combo == 0:
            continue

        ce_avg = loss_ce_total / max(1, n_ce)
        rank_avg = loss_rank_total / max(1, n_rank)
        combo_avg = loss_combo_total / max(1, n_combo)
        conc_avg = loss_conc_total / max(1, n_conc)
        if SOFT_COMBO_TRAINING:
            loss = combo_avg + CE_LOSS_WEIGHT * ce_avg
            if CONCENTRATION_TRAINING:
                loss = loss + NEAR_CFG.conc_gamma * conc_avg
        else:
            regret_w = effective_regret_weight(epoch)
            loss = CE_LOSS_WEIGHT * ce_avg + regret_w * rank_avg
        if not torch.isfinite(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_ce += float(ce_avg.item())
        running_rank += float(rank_avg.item())
        running_combo += float(combo_avg.item())
        running_conc += float(conc_avg.item())
        running_total += float(loss.item())
        steps += 1

    return {
        "ce": running_ce / max(1, steps),
        "rank": running_rank / max(1, steps),
        "soft_combo": running_combo / max(1, steps),
        "concentration": running_conc / max(1, steps),
        "total": running_total / max(1, steps),
        "valid_rank": float(valid_rank),
        "valid_combo": float(valid_combo),
        "valid_conc": float(valid_conc),
        "active_pair_frac": active_pairs / max(1, total_pairs),
        "combo_count": combo_count_total / max(1, valid_combo),
        "combo_model_regret": combo_model_regret_total / max(1, valid_combo),
        "combo_model_entropy": combo_model_entropy_total / max(1, valid_combo),
        "conc_max_load": conc_max_load_total / max(1, valid_conc),
        "conc_mean_cap": conc_mean_cap_total / max(1, valid_conc),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    rtt_by_dataset: RttByCombo,
    worst_regret_by_dataset: Dict[str, float],
    split_name: str,
) -> Dict[str, float]:
    model.eval()
    ce_total = 0.0
    valid_tasks = 0
    graph_correct = 0
    graphs = 0
    tasks_correct = 0
    tasks_total = 0
    regret_greedy: List[float] = []
    regret_seq_reforward: List[float] = []
    regret_topk: List[float] = []
    regret_oracle_topk: List[float] = []
    greedy_mapped = 0
    greedy_total = 0
    seq_reforward_mapped = 0
    seq_reforward_total = 0
    topk_mapped = 0
    topk_total = 0
    regret_masked_topo: List[float] = []
    masked_topo_mapped = 0
    masked_topo_total = 0

    for batch in tqdm(loader, desc=f"Evaluating {split_name}", leave=False):
        for graph in batch:
            data = move_graph_to_device(graph, DEVICE)
            if TEACHER_FORCED:
                # A prefix-free forward for the static per-task metrics below: score
                # every task against the EMPTY prefix. It is not this arm's decision
                # rule (that is masked_topo below), but it keeps top-1/greedy
                # comparable with the other arms instead of silently absent.
                _prefix_free_prefix_block(data)
            logits = model(data)
            if TEACHER_FORCED:
                loss_ce, valid_ce = loss_tied_teacher_forced_ce(model, data, DEVICE)
            else:
                loss_ce, valid_ce = loss_original_ce(logits, data, DEVICE)
            if valid_ce > 0:
                ce_total += float(loss_ce.item()) * valid_ce
                valid_tasks += valid_ce

            all_correct = True
            local_valid = 0
            for task_idx, logits_t in enumerate(logits):
                if logits_t.numel() == 0:
                    continue
                target = int(data.y[task_idx].item())
                if target < 0 or target >= logits_t.numel():
                    continue
                pred = int(logits_t.argmax().item())
                tasks_correct += int(pred == target)
                tasks_total += 1
                local_valid += 1
                if pred != target:
                    all_correct = False

            graphs += 1
            if all_correct and local_valid == int(data.n_tasks):
                graph_correct += 1

            dataset_id = lookup_dataset_id(data)
            opt_rtt = float(getattr(data, "opt_rtt", 0.0))
            rtt_map = rtt_by_dataset.get(dataset_id, {})
            worst_regret = worst_regret_by_dataset.get(
                dataset_id,
                NEAR_CFG.unmapped_penalty,
            )
            greedy_combo = decode_greedy(logits, data)
            if greedy_combo is not None:
                greedy_total += 1
                if greedy_combo in rtt_map:
                    greedy_mapped += 1
                greedy_regret = regret_for_combo(
                    greedy_combo,
                    rtt_map,
                    opt_rtt,
                    worst_regret,
                    NEAR_CFG.unmapped_penalty,
                )
                if greedy_regret is not None:
                    regret_greedy.append(greedy_regret)

            if TEACHER_FORCED:
                # Arm A1 is judged on the prefix-conditioned masked_topo decode, so the
                # checkpoint-selection metric must be that decode and not the static
                # greedy above.
                mt_combo = _masked_topo_regret_for_graph(model, data)
                if mt_combo is not None:
                    masked_topo_total += 1
                    if mt_combo in rtt_map:
                        masked_topo_mapped += 1
                    mt_regret = regret_for_combo(
                        mt_combo,
                        rtt_map,
                        opt_rtt,
                        worst_regret,
                        NEAR_CFG.unmapped_penalty,
                    )
                    if mt_regret is not None:
                        regret_masked_topo.append(mt_regret)

            if PHASE_B_CHECKPOINT_METRIC == "seq_reforward_regret":
                seq_combo = decode_sequential_reforward(model, data)
                if seq_combo is not None:
                    seq_reforward_total += 1
                    if seq_combo in rtt_map:
                        seq_reforward_mapped += 1
                    seq_regret = regret_for_combo(
                        seq_combo,
                        rtt_map,
                        opt_rtt,
                        worst_regret,
                        NEAR_CFG.unmapped_penalty,
                    )
                    if seq_regret is not None:
                        regret_seq_reforward.append(seq_regret)

            topk_combo, oracle_combo = decode_topk_joint(logits, data, rtt_map, NEAR_CFG.top_k_decode)
            if topk_combo is not None:
                topk_total += 1
                if topk_combo in rtt_map:
                    topk_mapped += 1
                topk_regret = regret_for_combo(
                    topk_combo,
                    rtt_map,
                    opt_rtt,
                    worst_regret,
                    NEAR_CFG.unmapped_penalty,
                )
                if topk_regret is not None:
                    regret_topk.append(topk_regret)
            if oracle_combo in rtt_map:
                regret_oracle_topk.append(float(rtt_map[oracle_combo]) - opt_rtt)

    def avg(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    metrics = {
        "ce": ce_total / max(1, valid_tasks),
        "acc": graph_correct / max(1, graphs),
        "task_acc": tasks_correct / max(1, tasks_total),
        "regret_greedy": avg(regret_greedy),
        "regret_masked_topo": avg(regret_masked_topo),
        "count_regret_masked_topo": float(len(regret_masked_topo)),
        "masked_topo_mapped_rate": masked_topo_mapped / max(1, masked_topo_total),
        "masked_topo_decoded": float(masked_topo_total),
        "regret_seq_reforward": avg(regret_seq_reforward),
        "regret_topk": avg(regret_topk),
        "regret_oracle_topk": avg(regret_oracle_topk),
        "count_regret_greedy": float(len(regret_greedy)),
        "count_regret_seq_reforward": float(len(regret_seq_reforward)),
        "count_regret_topk": float(len(regret_topk)),
        "greedy_sidecar_coverage": greedy_mapped / max(1, greedy_total),
        "seq_reforward_sidecar_coverage": seq_reforward_mapped / max(1, seq_reforward_total),
        "seq_reforward_unmapped": float(seq_reforward_total - seq_reforward_mapped),
        "topk_sidecar_coverage": topk_mapped / max(1, topk_total),
        "greedy_unmapped": float(greedy_total - greedy_mapped),
    }
    seq_msg = ""
    if PHASE_B_CHECKPOINT_METRIC == "seq_reforward_regret":
        seq_msg = (
            f" seq_reforward={metrics['regret_seq_reforward']:.4f}s "
            f"(sidecar_hit={metrics['seq_reforward_sidecar_coverage']*100:.1f}%, "
            f"unmapped={int(metrics['seq_reforward_unmapped'])})"
        )
    print(
        f"[{split_name}] acc={metrics['acc']*100:.1f}% "
        f"task_acc={metrics['task_acc']*100:.1f}% "
        f"greedy_regret={metrics['regret_greedy']:.4f}s "
        f"(sidecar_hit={metrics['greedy_sidecar_coverage']*100:.1f}%, "
        f"unmapped={int(metrics['greedy_unmapped'])}) "
        f"{seq_msg} "
        f"top{NEAR_CFG.top_k_decode}_regret={metrics['regret_topk']:.4f}s "
        f"oracle_top{NEAR_CFG.top_k_decode}={metrics['regret_oracle_topk']:.4f}s"
    )
    return metrics


def prefix(metrics: Dict[str, float], name: str) -> Dict[str, float]:
    return {f"{name}/{k}": float(v) for k, v in metrics.items()}


def effective_regret_weight(epoch: int) -> float:
    if os.environ.get("NEAR_RTT_REGRET_RAMP", "0") != "1":
        return REGRET_LOSS_WEIGHT
    ramp_epoch = int(os.environ.get("NEAR_RTT_REGRET_RAMP_EPOCH", "10"))
    start_w = float(os.environ.get("NEAR_RTT_REGRET_RAMP_START", "0.05"))
    end_w = float(os.environ.get("NEAR_RTT_REGRET_RAMP_END", str(REGRET_LOSS_WEIGHT)))
    if epoch < ramp_epoch:
        return start_w
    return end_w


def is_phase_b_ce_init() -> bool:
    return bool(os.environ.get("TRAIN_INIT_CHECKPOINT")) and not CE_ONLY_TRAINING


def phase_b_acc_collapse_floor() -> Optional[float]:
    if not is_phase_b_ce_init():
        return None
    baseline = float(os.environ.get("NEAR_RTT_CE_BASELINE_ACC", "0.244"))
    rel_drop = float(os.environ.get("NEAR_RTT_ACC_COLLAPSE_REL", "0.05"))
    return baseline * (1.0 - rel_drop)


def ranking_checkpoint_metric(val_metrics: Dict[str, float]) -> float:
    if TEACHER_FORCED:
        # Arm A1 is gated on the prefix-conditioned masked_topo decode, so that is what
        # selects its checkpoint. Selecting on the static greedy would pick the weights
        # that are best at a decision rule this arm never uses.
        if val_metrics["count_regret_masked_topo"] <= 0:
            raise RuntimeError(
                "Arm A1 validation produced no masked_topo regret samples — every "
                "decode failed or no combo mapped to a known RTT."
            )
        return float(val_metrics["regret_masked_topo"])
    if is_phase_b_ce_init() and PHASE_B_CHECKPOINT_METRIC == "seq_reforward_regret":
        if val_metrics["count_regret_seq_reforward"] <= 0:
            raise RuntimeError("Phase B sequential validation produced no regret samples.")
        return float(val_metrics["regret_seq_reforward"])
    if is_phase_b_ce_init():
        return float(val_metrics["regret_greedy"])
    if val_metrics["count_regret_topk"] > 0:
        return float(val_metrics["regret_topk"])
    return float(val_metrics["regret_greedy"])


def phase_b_collapse_reason(
    val_metrics: Dict[str, float],
    baseline: Optional[Dict[str, float]],
) -> Optional[str]:
    if baseline is None:
        return None
    val_acc = float(val_metrics["acc"])
    acc_floor = phase_b_acc_collapse_floor()
    if acc_floor is not None and val_acc < acc_floor:
        return (
            f"val acc {val_acc * 100:.1f}% < floor {acc_floor * 100:.1f}%"
        )
    baseline_greedy = float(baseline.get("regret_greedy", 0.0))
    greedy_rel = float(os.environ.get("NEAR_RTT_GREEDY_COLLAPSE_REL", "0.10"))
    val_greedy = float(val_metrics["regret_greedy"])
    if baseline_greedy > 0.0 and val_greedy > baseline_greedy * (1.0 + greedy_rel):
        return (
            f"greedy {val_greedy:.4f}s > baseline {baseline_greedy:.4f}s "
            f"* {1.0 + greedy_rel:.2f}"
        )
    if PHASE_B_CHECKPOINT_METRIC == "seq_reforward_regret":
        baseline_seq = float(baseline.get("regret_seq_reforward", 0.0))
        val_seq = float(val_metrics["regret_seq_reforward"])
        if baseline_seq > 0.0 and val_seq > baseline_seq * (1.0 + greedy_rel):
            return (
                f"seq_reforward {val_seq:.4f}s > baseline {baseline_seq:.4f}s "
                f"* {1.0 + greedy_rel:.2f}"
            )
    return None


def load_or_build_valid_combos() -> Tuple[PlacementToLogitMap, ExactRttLookupMap, RttByCombo]:
    graphs_for_ids, dataset_ids_for_ids = load_graphs_from_cache(CACHE_CTX)
    parent_ids = set(dataset_ids_for_ids)
    valid_combos_map = load_capped_valid_combos_map(CACHE_CTX.cache_dir, sidecar_name=NEAR_CFG.sidecar_name)
    if valid_combos_map is None and os.environ.get("NEAR_RTT_ALLOW_FULL_COMBOS", "0") == "1":
        valid_combos_map = load_valid_combos_map(CACHE_CTX.cache_dir)
    if valid_combos_map is None:
        valid_combos_map = build_capped_valid_combos_map_from_chunked_cache(
            CACHE_CTX.cache_dir,
            parent_ids,
            sidecar_name=NEAR_CFG.sidecar_name,
        )
        save_capped_valid_combos_map(CACHE_CTX.cache_dir, valid_combos_map, sidecar_name=NEAR_CFG.sidecar_name)

    if os.environ.get("NEAR_RTT_ALLOW_FULL_COMBOS", "0") == "1" and not valid_combos_map:
        valid_combos_map = build_valid_combos_map_from_chunked_cache(CACHE_CTX.cache_dir, parent_ids)
        save_valid_combos_map(CACHE_CTX.cache_dir, valid_combos_map)

    placement_to_logit, exact_rtt_map = build_exact_rtt_index_lookups(
        graphs_for_ids,
        dataset_ids_for_ids,
        valid_combos_map,
    )
    rtt_by_dataset = {
        dataset_id: {combo: float(rtt) for combo, rtt in combos}
        for dataset_id, combos in valid_combos_map.items()
    }
    del graphs_for_ids, dataset_ids_for_ids, valid_combos_map
    gc.collect()
    return placement_to_logit, exact_rtt_map, rtt_by_dataset


graphs, dataset_ids = load_graphs_from_cache(CACHE_CTX)
DATA_OPTIMAL_RTT = load_optimal_rtt_from_cache(CACHE_CTX)
PLACEMENT_TO_LOGIT_MAP, EXACT_RTT_MAP, RTT_BY_DATASET = load_or_build_valid_combos()
WORST_REGRET_BY_DATASET = build_worst_regret_by_dataset(RTT_BY_DATASET)

print(f"Loaded {len(graphs)} graphs")
_task_feature_dim = int(graphs[0].task_features.size(-1))
_platform_feature_dim = int(graphs[0].platform_features.size(-1))
if _task_feature_dim != TASK_FEATURE_DIM:
    print(
        f"Using cache task_feature_dim={_task_feature_dim} "
        f"(constants TASK_FEATURE_DIM={TASK_FEATURE_DIM})"
    )
if _platform_feature_dim != PLATFORM_FEATURE_DIM:
    print(
        f"Using cache platform_feature_dim={_platform_feature_dim} "
        f"(constants PLATFORM_FEATURE_DIM={PLATFORM_FEATURE_DIM})"
    )
print(f"Exact RTT datasets: {len(EXACT_RTT_MAP)}, combos: {sum(len(v) for v in EXACT_RTT_MAP.values()):,}")

ys = np.concatenate([g.y.numpy() for g in graphs])
print("Valid labels:", int(np.sum(ys >= 0)), "/", len(ys))
print("Avg edges:", float(np.mean([g.edge_index.size(1) for g in graphs])))

if NEAR_CFG.split_artifact:
    # B6: the pinned split. The bypasses below would silently produce a different
    # split than the sidecar claims, so both are refused outright while an
    # artifact is set.
    if NEAR_CFG.train_all:
        raise RuntimeError(
            "NEAR_RTT_SPLIT_ARTIFACT is set but NEAR_RTT_TRAIN_ALL=1 would bypass "
            "the split — the sidecar would then claim a split this run did not use. "
            "Unset one of them."
        )
    if len(graphs) < 10:
        raise RuntimeError(
            f"NEAR_RTT_SPLIT_ARTIFACT is set but the cache has only {len(graphs)} "
            f"graphs, which would trigger the train=val=test small-corpus bypass. "
            f"A pinned split on a corpus this small is not meaningful."
        )
    _split_path = Path(NEAR_CFG.split_artifact)
    if not _split_path.is_absolute() and not _split_path.is_file():
        # run_experiment configs carry repo-relative paths; the trainer may be
        # launched from elsewhere (sbatch cd's around), so fall back to the root.
        _split_path = _REPO_ROOT / NEAR_CFG.split_artifact
    _split_payload, _split_sha256 = load_split_artifact(_split_path)
    _graph_parents = [
        canonical_parent_id(getattr(g, "parent_dataset_id", None) or gid)
        for g, gid in zip(graphs, dataset_ids)
    ]
    assert_split_artifact_covers(
        _split_payload, _graph_parents, artifact_path=str(_split_path)
    )
    _parent_to_split = {
        parent: name
        for name in ("train", "val", "test")
        for parent in _split_payload[name]
    }
    _buckets: Dict[str, Tuple[list, list]] = {
        "train": ([], []), "val": ([], []), "test": ([], [])
    }
    for g, gid, parent in zip(graphs, dataset_ids, _graph_parents):
        bucket_graphs, bucket_ids = _buckets[_parent_to_split[parent]]
        bucket_graphs.append(g)
        bucket_ids.append(gid)
    train_graphs, train_ids = _buckets["train"]
    val_graphs, val_ids = _buckets["val"]
    test_graphs, test_ids = _buckets["test"]
    assert_zero_parent_overlap(train_ids, val_ids, test_ids)
    SPLIT_ARTIFACT_PROVENANCE = {
        "path": str(_split_path),
        "sha256": _split_sha256,
    }
    print(
        f"Split (B6 artifact {_split_path}, sha256={_split_sha256[:12]}…): "
        f"train={len(train_graphs)} val={len(val_graphs)} test={len(test_graphs)}"
    )
elif NEAR_CFG.train_all or len(graphs) < 10:
    train_graphs, val_graphs, test_graphs = graphs, graphs, graphs
    train_ids, val_ids, test_ids = dataset_ids, dataset_ids, dataset_ids
    # Record the bypass honestly — a sidecar claiming a split this run did not
    # perform is worse than no record.
    SPLIT_ARTIFACT_PROVENANCE = {"mode": "train_all"}
else:
    (
        train_graphs,
        train_ids,
        val_graphs,
        val_ids,
        test_graphs,
        test_ids,
    ) = split_ids_by_canonical_parent(
        graphs,
        dataset_ids,
        test_size=0.3,
        val_fraction_of_holdout=0.5,
        random_state=42,
    )
    assert_zero_parent_overlap(train_ids, val_ids, test_ids)
    SPLIT_ARTIFACT_PROVENANCE = {
        "mode": "split_ids_by_canonical_parent",
        "random_state": 42,
    }
    print(
        f"Split (canonical-parent 70/15/15): "
        f"train={len(train_graphs)} val={len(val_graphs)} test={len(test_graphs)}"
    )

train_dataset = GraphRttDataset(train_graphs, train_ids, DATA_OPTIMAL_RTT)
val_dataset = GraphRttDataset(val_graphs, val_ids, DATA_OPTIMAL_RTT)
test_dataset = GraphRttDataset(test_graphs, test_ids, DATA_OPTIMAL_RTT)

train_loader = create_loader(train_dataset, shuffle=True)
val_loader = create_loader(val_dataset, shuffle=False)
test_loader = create_loader(test_dataset, shuffle=False)

if RUNTIME_CONFIG.wandb_api_key:
    os.environ["WANDB_API_KEY"] = RUNTIME_CONFIG.wandb_api_key

wandb.init(
    project=RUNTIME_CONFIG.wandb_project,
    entity=RUNTIME_CONFIG.wandb_entity,
    name=os.environ.get("WANDB_RUN_NAME") or os.environ.get("WANDB_NAME") or None,
    config={
        "embedding_dim": int(EMBEDDING_DIM),
        "hidden_dim": int(HIDDEN_DIM),
        "lr": float(LEARNING_RATE),
        "epochs": int(EPOCHS),
        "batch_size": int(BATCH_SIZE),
        "num_gin_layers": int(NUM_GIN_LAYERS),
        "weight_decay": float(WEIGHT_DECAY),
        "device": str(DEVICE),
        "train_seed": int(_TRAIN_SEED),
        "deterministic_algorithms": bool(not _NONDETERMINISTIC),
        "ce_weight": float(CE_LOSS_WEIGHT),
        "regret_weight": float(REGRET_LOSS_WEIGHT),
        "rtt_scale_factor": float(RTT_SCALE_FACTOR),
        "loss_type": str(TRAIN_OBJECTIVE),
        "queue_feature_contract": str(_queue_feature_contract),
        "loss_variant": str(NEAR_CFG.loss_variant),
        "sidecar_name": str(NEAR_CFG.sidecar_name),
        "near_rtt_training": True,
        "soft_combo_tau": float(NEAR_CFG.soft_combo_tau),
        "soft_combo_max_combos": int(NEAR_CFG.soft_combo_max_combos),
        "concentration_gamma": float(NEAR_CFG.conc_gamma),
        "concentration_cap": float(NEAR_CFG.conc_cap),
        "top_k_decode": int(NEAR_CFG.top_k_decode),
        "pairs_per_graph": int(NEAR_CFG.pairs_per_graph),
        "near_margin_floor": float(NEAR_CFG.near_margin_floor),
        "margin_cap": float(NEAR_CFG.margin_cap),
        "margin_mode": str(NEAR_CFG.margin_mode),
        "margin_exp_scale": float(NEAR_CFG.margin_exp_scale),
        "margin_exp_clip": float(NEAR_CFG.margin_exp_clip),
        "near_weight": float(NEAR_CFG.near_weight),
        "close_weight": float(NEAR_CFG.close_weight),
        "mid_weight": float(NEAR_CFG.mid_weight),
        "far_weight": float(NEAR_CFG.far_weight),
        "trash_weight": float(NEAR_CFG.trash_weight),
        "trash_delta": float(NEAR_CFG.trash_delta),
        "num_datasets": int(len(graphs)),
        "task_feature_dim": int(_task_feature_dim),
        "platform_feature_dim": int(_platform_feature_dim),
        "num_train": int(len(train_graphs)),
        "num_val": int(len(val_graphs)),
        "num_test": int(len(test_graphs)),
        "num_exact_rtt_combo_rows": int(sum(len(v) for v in EXACT_RTT_MAP.values())),
        "unmapped_penalty": float(NEAR_CFG.unmapped_penalty),
        "cache_dir": str(CACHE_CTX.cache_dir),
        "phase_b_checkpoint_metric": str(PHASE_B_CHECKPOINT_METRIC),
        "seq_val_queue_norm_mode": str(
            os.environ.get(
                "NEAR_RTT_SEQ_VAL_QUEUE_NORM_MODE",
                os.environ.get("GNN_QUEUE_NORM_MODE", "scheduler_adaptive"),
            )
        ),
    },
    tags=[t for t in os.environ.get("WANDB_TAGS", "near-rtt").split(",") if t],
)

model = TaskPlacementGNN(
    task_feature_dim=_task_feature_dim,
    platform_feature_dim=_platform_feature_dim,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_GIN_LAYERS,
    dropout=NEAR_CFG.dropout,
    post_gin_dropout=NEAR_CFG.dropout,
    normalize_platform_inputs=_feature_dim == 21,
    mp_residual=NEAR_CFG.mp_residual,
    mp_node_edges=NEAR_CFG.mp_node_edges,
    mp_node_edges_candidates_only=NEAR_CFG.mp_node_edges_candidates_only,
    mp_network_entities=NEAR_CFG.mp_network_entities,
    # _task_feature_dim stays the cache-derived width; the model adds the one-hot
    # itself, so the printed provenance above stays honest about the cache.
    mp_dag_edges=NEAR_CFG.mp_dag_edges,
    task_type_onehot_dim=DAG_TASK_TYPE_ONEHOT_DIM if NEAR_CFG.task_type_onehot else 0,
    partial_state_edge_dim=(
        PARTIAL_STATE_FEATURE_DIM if NEAR_CFG.partial_state_edges else 0
    ),
).to(DEVICE)
print(
    f"Message passing: residual={NEAR_CFG.mp_residual} node_edges={NEAR_CFG.mp_node_edges} "
    f"candidates_only={NEAR_CFG.mp_node_edges_candidates_only} "
    f"network_entities={NEAR_CFG.mp_network_entities} "
    f"dag_edges={NEAR_CFG.mp_dag_edges} "
    f"task_type_onehot={NEAR_CFG.task_type_onehot} "
    f"partial_state_edges={NEAR_CFG.partial_state_edges} "
    f"({resolve_network_graph_contract()})"
)
if NEAR_CFG.mp_network_entities and (
    resolve_network_graph_contract() == NETWORK_GRAPH_CONTRACT_OFF
):
    raise ValueError(
        "NEAR_RTT_MP_NETWORK_ENTITIES=1 but NETWORK_GRAPH_CONTRACT resolves to 'off', so "
        "the cache carries no network entities for the model to message-pass over. Set "
        "NETWORK_GRAPH_CONTRACT=core_v1 and rebuild the cache."
    )


def init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)


model.apply(init_weights)

_init_ckpt = os.environ.get("TRAIN_INIT_CHECKPOINT")
if _init_ckpt:
    _init_path = Path(_init_ckpt)
    if not _init_path.is_absolute():
        _init_path = Path.cwd() / _init_path
    if _init_path.exists():
        model.load_state_dict(torch.load(str(_init_path), map_location=DEVICE))
        print(f"[INFO] Loaded init checkpoint: {_init_path}")
    else:
        print(f"[WARN] TRAIN_INIT_CHECKPOINT not found: {_init_path}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion = NearRttRankingLoss(EXACT_RTT_MAP, RTT_SCALE_FACTOR, NEAR_CFG)

model_path = Path("models") / f"{wandb.run.name}.pt"


def save_checkpoint(state_dict: Dict[str, Any], path: Path) -> None:
    """Save weights plus a contract sidecar.

    The GNN checkpoint is a bare state_dict, so platform dims 7/13 scaling cannot be
    inferred from weight shapes at load time. `executesimulation.load_gnn_model` reads this
    sidecar and refuses to serve a checkpoint under a different contract.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, path)
    sidecar = path.with_suffix(".contract.json")
    sidecar.write_text(
        json.dumps(
            {
                "queue_feature_contract": _queue_feature_contract,
                "cache_dir": str(CACHE_CTX.cache_dir),
                "cache_version": _cache_version if _metadata_path.exists() else None,
                "feature_dim": _feature_dim,
                # Serving must message-pass over the same graph this was fitted on.
                # `mp_residual` is also recoverable from the `mp_gate` weight, but record
                # it for provenance; `mp_node_edges` is recoverable ONLY from here.
                "mp_residual": NEAR_CFG.mp_residual,
                "mp_node_edges": NEAR_CFG.mp_node_edges,
                "mp_node_edges_candidates_only": NEAR_CFG.mp_node_edges_candidates_only,
                # Whether the GIN forward was SKIPPED during training. Weight-invisible
                # (the module is still constructed and initialised), and it lived only in
                # the environment until 2026-09-03 — so an MP-OFF checkpoint served
                # without the flag silently message-passes through weights that were never
                # fitted with it. Measured cost of that mismatch on the route_b DAG corpus:
                # train regret 12.67% -> 72.23%, a 5.7x error that reads as a plausible
                # ablation result. The live gates were protected by a run_provenance
                # assertion (score_mp_ablation.py, score_link_mp_v1.py); the offline
                # evaluators were not.
                "disable_message_passing": bool(
                    os.environ.get("GNN_DISABLE_MESSAGE_PASSING", "").strip().lower()
                    in ("1", "true", "yes")
                ),
                # Which network entities were in the training graph. The encoders show up
                # in the weights; the contract that built their *features* does not.
                "network_graph_contract": (
                    resolve_network_graph_contract()
                    if NEAR_CFG.mp_network_entities
                    else NETWORK_GRAPH_CONTRACT_OFF
                ),
                # Task feature dim 2 means different things under each contract and is
                # invisible in the weights, so serving cannot infer it. From the CACHE
                # when it records one; the resolver only covers pre-field caches.
                "topology_feature_contract": (
                    _topology_feature_contract or resolve_topology_feature_contract()
                ),
                # Weight shapes pin the platform feature *count*, not which layout assigns
                # meaning to those columns.
                # From the CACHE, not the environment — see the note at _inference_feature_layout.
                "inference_feature_layout": _inference_feature_layout,
                # dim7's divisor semantics (adaptive p90 vs fixed factor) — a cache built
                # with one and served under another is a silent divisor mismatch.
                "queue_norm_mode": _queue_norm_mode,
                # Which infrastructure this was actually fitted on, so a live run can say
                # whether it is in-distribution instead of guessing.
                "corpus": _corpus_provenance,
                # Which draw this is. A seeded-draw study attributes variance to the seed,
                # so serving the wrong checkpoint would be silent — the filename is not
                # evidence. `deterministic_algorithms` says whether the seed was actually
                # sufficient to reproduce these weights (see the seed block at the top).
                "train_seed": _TRAIN_SEED,
                "deterministic_algorithms": not _NONDETERMINISTIC,
                # route_b stage 2. mp_dag_edges is weight-invisible (recoverable ONLY
                # from here); the one-hot and prefix widths ARE weight-visible, but a
                # VOCAB REORDER is not — it stays 4 columns and would silently permute
                # the types — hence dag_task_type_vocab.
                "mp_dag_edges": NEAR_CFG.mp_dag_edges,
                "mp_dag_edges_undirected": True if NEAR_CFG.mp_dag_edges else None,
                "task_type_onehot_dim": (
                    DAG_TASK_TYPE_ONEHOT_DIM if NEAR_CFG.task_type_onehot else 0
                ),
                "dag_task_type_vocab": (
                    list(DAG_TASK_TYPE_VOCAB) if NEAR_CFG.task_type_onehot else None
                ),
                # Prefix conditioning. `partial_state_edge_features` is what makes
                # executesimulation refuse to serve this checkpoint: its scores are a
                # function of the committed decode prefix, and live prefix construction
                # is stage 3.
                "partial_state_edge_features": NEAR_CFG.partial_state_edges,
                "partial_state_contract": (
                    resolve_partial_state_contract()
                    if NEAR_CFG.partial_state_edges
                    else None
                ),
                "partial_state_feature_dim": (
                    PARTIAL_STATE_FEATURE_DIM if NEAR_CFG.partial_state_edges else None
                ),
                # Which capacity rung the labels AND the capacity columns came from —
                # they move together, so this names both.
                "dag_alpha_key": NEAR_CFG.dag_alpha_key if TEACHER_FORCED else None,
                "tied_label_mode": (
                    "any_of_k_marginalized" if TEACHER_FORCED else None
                ),
                # Non-zero CHANGES the loss definition, so it is recorded, not implied.
                "tied_max_plans": NEAR_CFG.tied_max_plans if TEACHER_FORCED else None,
                # §3 requires a draw to vary initialisation and batch order ONLY.
                # Under NEAR_RTT_SPLIT_ARTIFACT (B6) this is {"path", "sha256"} of the
                # shared artifact; otherwise it names the split this run actually drew.
                # Registered draws must carry the artifact form — a paired test whose
                # split moves between arms is confounded. Analysis provenance only:
                # not consumed by checkpoint_mp_config at serve time, so no serving
                # whitelist entry is needed.
                "split_artifact": SPLIT_ARTIFACT_PROVENANCE,
                # Message passing sees DAG structure but NOT the prefix: the 38 columns
                # enter at the EdgeScorer. Still strictly T2 (the head interacts
                # graph-derived embeddings with the prefix), and it is §2's fairness
                # bargain — the GNN gets the same 38 pointwise columns as T1, plus
                # structure. The prefix-into-node-features variant is a §3 sensitivity
                # row, not this arm.
                "prefix_conditioning_scope": (
                    "edge_scorer_only" if TEACHER_FORCED else None
                ),
            },
            indent=2,
        )
        + "\n"
    )


best_val_regret = float("inf")
best_val_acc = 0.0
best_val_metrics: Dict[str, float] = {}
checkpoint_saved = False
phase_b_baseline: Optional[Dict[str, float]] = None
checkpoint_metric_name = "regret_topk"
if TEACHER_FORCED:
    checkpoint_metric_name = "regret_masked_topo"
    print(f"[route_b A1] Checkpoint metric: val/{checkpoint_metric_name}")

if is_phase_b_ce_init():
    checkpoint_metric_name = (
        "regret_seq_reforward"
        if PHASE_B_CHECKPOINT_METRIC == "seq_reforward_regret"
        else "regret_greedy"
    )
    print(f"[Phase B] Checkpoint metric: val/{checkpoint_metric_name}")
    phase_b_baseline = evaluate(
        model, val_loader, RTT_BY_DATASET, WORST_REGRET_BY_DATASET, "phase-b/baseline"
    )
    best_val_regret = ranking_checkpoint_metric(phase_b_baseline)
    best_val_metrics = phase_b_baseline
    save_checkpoint(model.state_dict(), model_path)
    checkpoint_saved = True
    print(
        f"[Phase B baseline] acc={phase_b_baseline['acc'] * 100:.1f}% "
        f"seq_reforward={phase_b_baseline['regret_seq_reforward']:.4f}s "
        f"greedy={phase_b_baseline['regret_greedy']:.4f}s "
        f"top{NEAR_CFG.top_k_decode}={phase_b_baseline['regret_topk']:.4f}s "
        f"(seed checkpoint saved)"
    )
    wandb.summary["phase_b_baseline_seq_reforward"] = float(
        phase_b_baseline["regret_seq_reforward"]
    )
    wandb.summary["phase_b_baseline_greedy"] = float(phase_b_baseline["regret_greedy"])
    wandb.summary["phase_b_baseline_acc"] = float(phase_b_baseline["acc"])

print("=" * 80)
print(f"TRAINING ({TRAIN_OBJECTIVE})")
print("=" * 80)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

for epoch in range(EPOCHS):
    start = time.perf_counter()
    train_metrics = train_epoch(model, train_loader, optimizer, criterion, epoch)
    val_metrics = evaluate(model, val_loader, RTT_BY_DATASET, WORST_REGRET_BY_DATASET, "val")

    log_dict: Dict[str, float] = {}
    log_dict.update(prefix(train_metrics, "train"))
    log_dict.update(prefix(val_metrics, "val"))
    log_dict["lr"] = float(optimizer.param_groups[0]["lr"])
    if not CE_ONLY_TRAINING:
        log_dict["train/effective_regret_weight"] = float(effective_regret_weight(epoch))
    wandb.log(log_dict, step=epoch)

    if TEACHER_FORCED:
        # Arm A1 is CE-only, but it must NOT select on the CE-only branch's acc/top-k:
        # both are read off the prefix-free forward, which is not this arm's decision
        # rule. Select on the prefix-conditioned masked_topo decode it is gated on.
        val_target = ranking_checkpoint_metric(val_metrics)
        if val_target < best_val_regret:
            best_val_regret = val_target
            best_val_metrics = val_metrics
            best_val_acc = max(best_val_acc, float(val_metrics["acc"]))
            save_checkpoint(model.state_dict(), model_path)
            checkpoint_saved = True
            print(
                f"  *** New best val {checkpoint_metric_name}: {best_val_regret:.4f}s "
                f"(decoded={int(val_metrics['masked_topo_decoded'])}, "
                f"mapped={val_metrics['masked_topo_mapped_rate'] * 100:.1f}%, "
                f"ce={val_metrics['ce']:.4f})"
            )
    elif CE_ONLY_TRAINING:
        val_target_acc = float(val_metrics["acc"])
        val_topk = float(val_metrics["regret_topk"])
        improved = False
        reason = ""
        if val_target_acc > best_val_acc:
            best_val_acc = val_target_acc
            improved = True
            reason = f"val acc={best_val_acc * 100:.1f}%"
        elif best_val_acc <= 0.0 and val_topk < best_val_regret:
            # Small/hard corpora can keep joint combo acc at 0 while top-k regret moves.
            best_val_regret = val_topk
            improved = True
            reason = f"val top{NEAR_CFG.top_k_decode} regret={best_val_regret:.4f}s (acc still 0)"
        if improved:
            best_val_metrics = val_metrics
            save_checkpoint(model.state_dict(), model_path)
            checkpoint_saved = True
            print(
                f"  *** New best {reason} "
                f"(top{NEAR_CFG.top_k_decode}={val_metrics['regret_topk']:.4f}s "
                f"task_acc={float(val_metrics.get('task_acc', 0.0)) * 100:.1f}%)"
            )
    else:
        val_target = ranking_checkpoint_metric(val_metrics)
        val_acc = float(val_metrics["acc"])
        collapse_reason = phase_b_collapse_reason(val_metrics, phase_b_baseline)
        if val_target < best_val_regret:
            if collapse_reason is not None:
                print(
                    f"  [COLLAPSE GUARD] skip save: {collapse_reason} "
                    f"(seq_reforward={val_metrics['regret_seq_reforward']:.4f}s, "
                    f"top{NEAR_CFG.top_k_decode}={val_metrics['regret_topk']:.4f}s, "
                    f"greedy={val_metrics['regret_greedy']:.4f}s)"
                )
            else:
                best_val_regret = val_target
                best_val_metrics = val_metrics
                save_checkpoint(model.state_dict(), model_path)
                checkpoint_saved = True
                print(
                    f"  *** New best val {checkpoint_metric_name}: {best_val_regret:.4f}s "
                    f"(seq_reforward={val_metrics['regret_seq_reforward']:.4f}s, "
                    f"greedy={val_metrics['regret_greedy']:.4f}s, "
                    f"top{NEAR_CFG.top_k_decode}={val_metrics['regret_topk']:.4f}s, "
                    f"acc={val_acc * 100:.1f}%)"
                )

    if epoch % 5 == 0 or epoch == EPOCHS - 1:
        print(
            f"Epoch {epoch:3d}/{EPOCHS} "
            f"Train CE={train_metrics['ce']:.4f} "
            f"Rank={train_metrics['rank']:.4f} "
            f"SoftCombo={train_metrics['soft_combo']:.4f} "
            f"Conc={train_metrics['concentration']:.4f} "
            f"Val seq_reforward={val_metrics['regret_seq_reforward']:.4f}s "
            f"Val greedy={val_metrics['regret_greedy']:.4f}s "
            f"Val top{NEAR_CFG.top_k_decode}={val_metrics['regret_topk']:.4f}s "
            f"active_pair_frac={train_metrics['active_pair_frac']:.2f} "
            f"({time.perf_counter() - start:.1f}s)"
        )

if not checkpoint_saved:
    raise RuntimeError("No near-RTT checkpoint was saved.")

# Opt-in: also keep the LAST-epoch weights, with their own sidecar. The served
# checkpoint is always the val-selected one above; the final weights exist for
# fit-ceiling questions ("can this arm memorise the training split at all?"), which
# a val-selected checkpoint cannot answer once validation has plateaued.
if os.environ.get("NEAR_RTT_SAVE_FINAL", "0") == "1":
    final_path = model_path.with_name(f"{model_path.stem}-final.pt")
    save_checkpoint(model.state_dict(), final_path)
    print(f"[final] last-epoch weights saved to {final_path} (NEAR_RTT_SAVE_FINAL=1)")

model.load_state_dict(torch.load(model_path, map_location=DEVICE))
train_final = evaluate(model, train_loader, RTT_BY_DATASET, WORST_REGRET_BY_DATASET, "final/train")
val_final = evaluate(model, val_loader, RTT_BY_DATASET, WORST_REGRET_BY_DATASET, "final/val")
test_final = evaluate(model, test_loader, RTT_BY_DATASET, WORST_REGRET_BY_DATASET, "final/test")

final_log: Dict[str, float] = {}
final_log.update(prefix(train_final, "final/train"))
final_log.update(prefix(val_final, "final/val"))
final_log.update(prefix(test_final, "final/test"))
wandb.log(final_log)

if CE_ONLY_TRAINING:
    wandb.summary["best_val_acc"] = float(best_val_acc)
elif is_phase_b_ce_init():
    wandb.summary["best_val_checkpoint_target"] = float(best_val_regret)
    wandb.summary["best_val_regret_greedy"] = float(
        best_val_metrics.get("regret_greedy", 0.0)
    )
    wandb.summary["best_val_regret_seq_reforward"] = float(
        best_val_metrics.get("regret_seq_reforward", 0.0)
    )
    wandb.summary["best_val_regret_topk"] = float(best_val_metrics.get("regret_topk", 0.0))
    wandb.summary["checkpoint_metric"] = str(checkpoint_metric_name)
else:
    wandb.summary["best_val_regret_topk"] = float(best_val_regret)
    wandb.summary["checkpoint_metric"] = "regret_topk"
wandb.summary["best_val_regret_greedy"] = float(best_val_metrics.get("regret_greedy", 0.0))
wandb.summary["final_test_regret_topk"] = float(test_final["regret_topk"])
wandb.summary["final_test_regret_greedy"] = float(test_final["regret_greedy"])
wandb.summary["final_test_oracle_topk"] = float(test_final["regret_oracle_topk"])

artifact = wandb.Artifact("placement-gnn-near-rtt", type="model")
artifact.add_file(str(model_path))
wandb.log_artifact(artifact)
wandb.finish()

print("=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print(f"Model saved to: {model_path}")
if CE_ONLY_TRAINING:
    print(f"Best val acc: {best_val_acc * 100:.1f}%")
else:
    print(f"Best val {checkpoint_metric_name}: {best_val_regret:.4f}s")
print(
    f"Final test: greedy={test_final['regret_greedy']:.4f}s, "
    f"top{NEAR_CFG.top_k_decode}={test_final['regret_topk']:.4f}s, "
    f"oracle_top{NEAR_CFG.top_k_decode}={test_final['regret_oracle_topk']:.4f}s"
)
