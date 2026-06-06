# %%
# %%
#!/usr/bin/env python3
"""
GNN for Task-to-Platform Placement Prediction - EXACT RTT RANKING TRAINING.

Trains on sequential counterfactual graphs (prepare_graphs_cache_exact_rtt.py).
Uses a combined loss:
  Loss = alpha * CrossEntropy + beta * ExactRttRankingLoss

ExactRttRankingLoss (no random sampling):
1. Loads every co-sim (combo, exact RTT) for the parent dataset
2. Sorts combos by RTT ascending
3. Pairwise margin loss on adjacent pairs: lower RTT must score higher by exact ΔRTT

NON-UNIQUE PLACEMENTS:
- Multiple tasks can be placed on the same replica (node_id, platform_id)
- Decoder uses greedy per-task selection (no uniqueness constraint)
"""

import gc
import os
import sys
import time
import numpy as np
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.nn.models import GIN
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb

# Temporary timing logs — delete this block when no longer needed.
_TRAIN_LOG_BATCH_EVERY = 25
_TRAIN_LOG_SLOW_STEP_SEC = 20.0

from non_unique_lib.cache_io import (
    ExactRttLookupMap,
    PlacementToLogitMap,
    build_exact_rtt_index_lookups,
    build_valid_combos_map_from_chunked_cache,
    create_cache_context,
    load_graphs_from_cache,
    load_optimal_rtt_from_cache,
    load_valid_combos_map,
)
from non_unique_lib.training_config import parse_training_config
from non_unique_lib.seq_training_utils import (
    decode_sequential_argmax_placement,
    initial_queue_snapshot_for_graph,
    is_final_sequential_graph,
)


np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# %%
# Configuration (default: filtered high-queue sequential cache with exact RTT sidecar)
_DEFAULT_EXACT_CACHE_DIR = (
    Path(__file__).resolve().parents[2]
    / "simulation_data"
    / "artifacts"
    / "run_queue_big"
    / "graphs_cache_gnn_datasets_4tasks_seq_filtered"
)

RUNTIME_CONFIG = parse_training_config()
if "--cache-dir" not in sys.argv:
    RUNTIME_CONFIG = replace(RUNTIME_CONFIG, cache_dir=_DEFAULT_EXACT_CACHE_DIR)

CACHE_CTX = create_cache_context(RUNTIME_CONFIG.cache_dir)
SEQUENTIAL_CACHE = bool(CACHE_CTX.metadata.get("sequential_counterfactual", False))
_CACHE_VERSION = CACHE_CTX.metadata.get("version", "unknown")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IS_MERGED_CACHE = CACHE_CTX.is_merged_cache or RUNTIME_CONFIG.use_merged_cache
TASK_COUNT_DIST = CACHE_CTX.task_count_dist

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
NUM_DATALOADER_WORKERS = RUNTIME_CONFIG.num_dataloader_workers
PRECOMPUTE_RTT_LOOKUPS = RUNTIME_CONFIG.precompute_rtt_lookups
print(
    f"[INFO] Cache version={_CACHE_VERSION} sequential={SEQUENTIAL_CACHE} "
    f"exact_rtt_training={CACHE_CTX.metadata.get('exact_rtt_training', False)}"
)

print(f"Cache directory: {CACHE_CTX.cache_dir}")
print(f"Cache mode: {'MERGED' if IS_MERGED_CACHE else 'SINGLE'}")
print(f"Sequential counterfactual cache: {SEQUENTIAL_CACHE}")
if not SEQUENTIAL_CACHE:
    print(
        "WARNING: metadata.sequential_counterfactual is false; "
        "run prepare_graphs_cache_exact_rtt.py and pass --cache-dir to match."
    )
if TASK_COUNT_DIST:
    print("Task count distribution in cache:")
    for n_tasks, count in sorted(TASK_COUNT_DIST.items(), key=lambda x: int(x[0])):
        print(f"  {n_tasks} tasks: {count} graphs")

print(
    f"DataLoader num_workers={NUM_DATALOADER_WORKERS}, "
    f"precompute_rtt_lookups={PRECOMPUTE_RTT_LOOKUPS}"
)


# %%
# ============================================================================
# GNN MODEL (same as before)
# ============================================================================

class MLPEncoder(nn.Module):
    """Generic 2-layer MLP encoder with LayerNorm."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class EdgeScorer(nn.Module):
    """2-layer MLP to score task-platform edges with optional edge attributes."""
    def __init__(self, embedding_dim: int, hidden_dim: int, edge_dim: int = 0) -> None:
        super().__init__()
        in_dim = 2 * embedding_dim + (edge_dim if edge_dim else 0)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(
        self,
        e_task: Tensor,
        e_platform: Tensor,
        e_attr: Optional[Tensor] = None,
    ) -> Tensor:
        x = torch.cat([e_task, e_platform] + ([e_attr] if e_attr is not None else []), dim=-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze(-1)


class TaskPlacementGNN(nn.Module):
    """
    1. Encode task and platform features separately
    2. GIN to produce node embeddings
    3. Edge MLP to score task-platform compatibility
    4. Masked softmax to predict placement probabilities
    """
    def __init__(
        self,
        task_feature_dim: int,
        platform_feature_dim: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.task_encoder = MLPEncoder(task_feature_dim, hidden_dim, embedding_dim)
        self.platform_encoder = MLPEncoder(platform_feature_dim, hidden_dim, embedding_dim)
        
        self.gin = GIN(
            in_channels=embedding_dim,
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            out_channels=embedding_dim
        )
        self.post_gin_dropout = nn.Dropout(p=0.2)
        self.edge_scorer = EdgeScorer(embedding_dim, hidden_dim, edge_dim=5)  # 5 edge dims (exec, latency, warm, energy, comm)

    def forward(self, data: Data) -> List[Tensor]:
        n_tasks: int = int(data.n_tasks)
        n_platforms: int = int(data.n_platforms)

        # Encode features (inputs are finite if graphs were built with prepare_graphs_cache.py)
        task_embeddings = self.task_encoder(data.task_features)
        platform_embeddings = self.platform_encoder(data.platform_features)

        # Message passing
        x = torch.cat([task_embeddings, platform_embeddings], dim=0)
        x = self.gin(x, data.edge_index)
        x = self.post_gin_dropout(x)
        
        task_emb = x[:n_tasks]
        platform_emb = x[n_tasks:]

        # Score edges
        ei = data.edge_index
        if ei.numel() == 0:
            return [torch.empty(0, device=x.device) for _ in range(n_tasks)]

        ti = ei[0]
        pj = ei[1] - n_tasks
        valid = (pj >= 0) & (pj < n_platforms)
        ti = ti[valid]
        pj = pj[valid]
        if ti.numel() == 0:
            return [torch.empty(0, device=x.device) for _ in range(n_tasks)]

        e_task = task_emb[ti]
        e_platform = platform_emb[pj]
        e_attr: Optional[Tensor] = None
        if hasattr(data, 'edge_attr') and data.edge_attr.numel() > 0:
            try:
                e_attr = data.edge_attr[valid]
            except (IndexError, RuntimeError):
                e_attr = None
        edge_scores = self.edge_scorer(e_task, e_platform, e_attr)

        # Split scores per task
        logits_per_task = []
        for t in range(n_tasks):
            mask_t = (ti == t)
            logits_t = edge_scores[mask_t]
            logits_per_task.append(logits_t)

        return logits_per_task


# %%
# ============================================================================
# SEQUENTIAL CE MASK (B1: one active task per step, all tasks on final step)
# ============================================================================

def ce_label_mask_for_graph(data: Data) -> Tensor:
    mask = getattr(data, "ce_label_mask", None)
    if mask is not None:
        return mask
    step = getattr(data, "seq_step", None)
    n = getattr(data, "seq_n_tasks", None)
    if step is not None and n is not None:
        n_int = int(n)
        step_int = int(step)
        out = torch.zeros(n_int, dtype=torch.bool, device=data.y.device)
        if step_int == n_int - 1:
            out[:] = True
        elif 0 <= step_int < n_int:
            out[step_int] = True
        return out
    return data.y >= 0


def ensure_graph_ce_label_mask(graph: Data) -> None:
    if getattr(graph, "ce_label_mask", None) is not None:
        return
    graph.ce_label_mask = ce_label_mask_for_graph(graph)


def task_in_ce_loss(data: Data, task_idx: int) -> bool:
    mask = ce_label_mask_for_graph(data)
    if task_idx < 0 or task_idx >= mask.numel():
        return False
    return bool(mask[task_idx].item())


def print_sequential_label_statistics(graphs: List[Data]) -> None:
    ys = np.concatenate([g.y.numpy() for g in graphs])
    placement_valid = int(np.sum(ys >= 0))
    total_slots = len(ys)

    ce_active = 0
    ce_missing = 0
    for g in graphs:
        ensure_graph_ce_label_mask(g)
        mask = g.ce_label_mask.numpy()
        ce_active += int(mask.sum())
        y_np = g.y.numpy()
        for t_idx in range(len(y_np)):
            if mask[t_idx] and y_np[t_idx] < 0:
                ce_missing += 1

    cache_ver = CACHE_CTX.metadata.get("version", "?")
    print(f"Cache version: {cache_ver}")
    print(
        f"Placement labels (y>=0, all tasks): {placement_valid} / {total_slots} "
        f"({100 * placement_valid / total_slots:.1f}%)"
    )
    print(
        f"CE training targets (sequential mask): {ce_active} / {total_slots} "
        f"({100 * ce_active / total_slots:.1f}%) — NOT missing data; one task/step + all on final"
    )
    if ce_missing:
        print(f"WARNING: CE targets without placement label: {ce_missing} / {ce_active}")
    ce_per_graph = [int(g.ce_label_mask.sum()) for g in graphs]
    print(f"Avg CE targets per graph: {np.mean(ce_per_graph):.2f}")
    print(f"Min/Max CE targets per graph: {min(ce_per_graph)} / {max(ce_per_graph)}")


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def loss_original_ce(logits_per_task, data, device):
    """
    Original cross-entropy loss with one-hot labels (optimal placement).
    """
    loss_total = torch.zeros(1, device=device)
    valid_tasks = 0
    
    for task_idx, logits_t in enumerate(logits_per_task):
        if logits_t.numel() == 0:
            continue
        if not task_in_ce_loss(data, task_idx):
            continue

        logits = logits_t.unsqueeze(0)
        target = data.y[task_idx].long()
        if target.ndim == 0:
            target = target.unsqueeze(0)

        if target.item() < 0 or target.item() >= logits.size(1):
            continue
        
        loss_total = loss_total + F.cross_entropy(logits, target)
        valid_tasks += 1
    
    if valid_tasks == 0:
        return torch.zeros(1, device=device), 0
    
    return loss_total / valid_tasks, valid_tasks


class ExactRttRankingLoss(nn.Module):
    """
    Exact RTT ranking via pairwise margins on the full co-sim combo list.

    For parent dataset combos sorted by ascending RTT (a, b, c, ...):
      loss += max(0, (RTT_b - RTT_a)/scale - (score_a - score_b))

    Every combo uses its exact co-sim RTT; no random negatives or subsampling.
    """

    def __init__(
        self,
        rtt_scale: float,
        exact_rtt_map: ExactRttLookupMap,
    ) -> None:
        super().__init__()
        self.rtt_scale = rtt_scale
        self.exact_rtt_map = exact_rtt_map

    def _combo_score(
        self,
        logits_per_task: List[torch.Tensor],
        indices: List[int],
    ) -> torch.Tensor:
        score = torch.tensor(0.0, device=logits_per_task[0].device)
        for t_idx, logit_idx in enumerate(indices):
            if logit_idx >= logits_per_task[t_idx].numel():
                return score.new_tensor(float("nan"))
            score = score + logits_per_task[t_idx][logit_idx]
        return score

    def forward(
        self,
        logits_per_task: List[torch.Tensor],
        data: Data,
        device: torch.device,
    ) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        parent_dataset_id = getattr(data, "parent_dataset_id", None)
        if not parent_dataset_id:
            return torch.tensor(0.0, device=device), 0, {}

        n_tasks = int(data.n_tasks)
        seq_step = getattr(data, "seq_step", None)
        seq_n_tasks = getattr(data, "seq_n_tasks", None)
        if seq_step is not None and seq_n_tasks is not None and int(seq_step) != int(seq_n_tasks) - 1:
            return torch.tensor(0.0, device=device), 0, {}

        entries = self.exact_rtt_map.get(parent_dataset_id, [])
        if len(entries) < 2:
            return torch.tensor(0.0, device=device), 0, {}

        scores: List[torch.Tensor] = []
        rtts: List[float] = []
        for indices, rtt in entries:
            if len(indices) != n_tasks:
                continue
            score = self._combo_score(logits_per_task, indices)
            if torch.isnan(score):
                continue
            scores.append(score)
            rtts.append(float(rtt))

        if len(scores) < 2:
            return torch.tensor(0.0, device=device), 0, {}

        loss = torch.tensor(0.0, device=device)
        n_active = 0
        for k in range(len(scores) - 1):
            rtt_gap = (rtts[k + 1] - rtts[k]) / self.rtt_scale
            if rtt_gap <= 0.0:
                continue
            margin = scores[k] - scores[k + 1]
            pair_loss = F.relu(torch.tensor(rtt_gap, device=device) - margin)
            if pair_loss.item() > 1e-12:
                n_active += 1
            loss = loss + pair_loss

        n_pairs = len(scores) - 1
        loss = loss / max(1, n_pairs)

        stats = {
            "n_combos": len(scores),
            "n_pairs": n_pairs,
            "n_active_pairs": n_active,
            "opt_rtt": rtts[0],
            "worst_rtt": rtts[-1],
        }
        return loss, 1, stats


class GraphRttDataset(torch.utils.data.Dataset):
    """Lightweight dataset: attaches per-step metadata needed after batching."""

    def __init__(
        self,
        graphs: List[Data],
        dataset_ids: List[str],
        optimal_rtt_map: Dict[str, float],
    ) -> None:
        self.graphs = graphs
        self.dataset_ids = dataset_ids
        self.optimal_rtt_map = optimal_rtt_map
        self._task_map_by_dataset: Dict[str, Dict[int, List[Tuple[int, int]]]] = {}
        for graph, dataset_id in zip(self.graphs, self.dataset_ids):
            task_map = getattr(graph, "task_logit_to_placement", None)
            if task_map is None:
                task_map = getattr(graph, "_task_logit_to_placement", None)
            if task_map is not None:
                self._task_map_by_dataset[dataset_id] = task_map

    def __len__(self) -> int:
        return len(self.dataset_ids)

    def __getitem__(self, idx: int) -> Data:
        graph = self.graphs[idx]
        dataset_id = self.dataset_ids[idx]
        parent_dataset_id = getattr(graph, "parent_dataset_id", None)
        rtt_key = parent_dataset_id or dataset_id
        graph.dataset_id = dataset_id
        graph.parent_dataset_id = parent_dataset_id
        graph.opt_rtt = float(
            self.optimal_rtt_map.get(
                dataset_id, self.optimal_rtt_map.get(rtt_key, 0.0)
            )
        )
        graph.task_logit_to_placement = self._task_map_by_dataset.get(
            dataset_id,
            self._task_map_by_dataset.get(rtt_key, {}),
        )
        graph.seq_step = getattr(graph, "seq_step", None)
        graph.seq_n_tasks = getattr(graph, "seq_n_tasks", None)
        graph.prefix_augment = bool(getattr(graph, "prefix_augment", False))
        graph.queue_snapshot = getattr(graph, "queue_snapshot", None)
        graph.initial_queue_snapshot = getattr(graph, "initial_queue_snapshot", None)
        graph.task_logit_to_queue_key = getattr(graph, "task_logit_to_queue_key", None)
        ce_mask = getattr(graph, "ce_label_mask", None)
        if ce_mask is not None:
            graph.ce_label_mask = ce_mask.clone()
        return graph


# %%
# ============================================================================
# CUSTOM COLLATE AND ATTRIBUTE RESTORATION
# ============================================================================

# Non-tensor / heterogeneous attrs — stripped before Batch.from_data_list, restored via sidecars.
_SIDECAR_ATTR_KEYS = (
    "dataset_id",
    "parent_dataset_id",
    "opt_rtt",
    "seq_step",
    "seq_n_tasks",
    "prefix_augment",
    "task_logit_to_placement",
    "_task_logit_to_placement",
    "task_logit_to_queue_key",
    "queue_snapshot",
    "initial_queue_snapshot",
)


def _extract_graph_sidecar(data: Data) -> Dict[str, Any]:
    sidecar: Dict[str, Any] = {}
    for key in _SIDECAR_ATTR_KEYS:
        val = getattr(data, key, None)
        if val is None and key in data:
            val = data[key]
        sidecar[key] = val
    if sidecar.get("prefix_augment") is None:
        sidecar["prefix_augment"] = False
    return sidecar


def _apply_graph_sidecar(data: Data, sidecar: Dict[str, Any]) -> Data:
    for key, val in sidecar.items():
        if val is not None:
            setattr(data, key, val)
    return data


def sequential_collate(batch: List[Data]) -> Batch:
    """Batch tensor graph fields only; keep metadata on batch._graph_sidecars."""
    sidecars = [_extract_graph_sidecar(data) for data in batch]
    stripped: List[Data] = []
    for data in batch:
        clean = data.clone()
        for key in _SIDECAR_ATTR_KEYS:
            if key in clean:
                del clean[key]
        stripped.append(clean)
    out = Batch.from_data_list(stripped)
    out._graph_sidecars = sidecars
    return out


def graphs_from_batch(batch: Batch) -> List[Data]:
    sidecars = getattr(batch, "_graph_sidecars", None)
    graphs = batch.to_data_list()
    if sidecars is None:
        return graphs
    return [_apply_graph_sidecar(g, sc) for g, sc in zip(graphs, sidecars)]


def _regret_lookup_graphs(
    graphs: List[Data],
    dataset_ids: List[str],
) -> Tuple[List[Data], List[str]]:
    """Use one final-step graph per parent dataset for RTT hash / hard-negative lookups."""
    by_parent: Dict[str, Tuple[Data, str]] = {}
    for graph, graph_id in zip(graphs, dataset_ids):
        parent_id = getattr(graph, "parent_dataset_id", None)
        if parent_id is None:
            by_parent[graph_id] = (graph, graph_id)
            continue
        step = int(getattr(graph, "seq_step", -1))
        n_tasks = int(getattr(graph, "seq_n_tasks", 0))
        if step == n_tasks - 1:
            by_parent[parent_id] = (graph, parent_id)
    if not by_parent:
        return graphs, dataset_ids
    rep_graphs = [pair[0] for pair in by_parent.values()]
    rep_ids = [pair[1] for pair in by_parent.values()]
    return rep_graphs, rep_ids


def _alias_seq_exact_lookups(
    graphs: List[Data],
    dataset_ids: List[str],
    exact_rtt_map: ExactRttLookupMap,
) -> None:
    """Copy parent exact RTT lookup tables onto each sequential graph id."""
    for graph, graph_id in zip(graphs, dataset_ids):
        parent_id = getattr(graph, "parent_dataset_id", None)
        if parent_id and parent_id in exact_rtt_map:
            exact_rtt_map[graph_id] = exact_rtt_map[parent_id]


def create_dataloader(dataset, *, shuffle: bool, pin_memory: bool) -> DataLoader:
    kw: Dict[str, Any] = dict(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_DATALOADER_WORKERS,
        pin_memory=pin_memory,
        collate_fn=sequential_collate,
    )
    if NUM_DATALOADER_WORKERS > 0:
        kw["prefetch_factor"] = RUNTIME_CONFIG.dataloader_prefetch_factor
        kw["persistent_workers"] = False
    return DataLoader(**kw)


def release_eval_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_final_split_eval(model, dataset, split_name: str) -> Dict[str, Any]:
    """Evaluate one split and free loader/CUDA cache before the next split."""
    print(f"[final eval] starting {split_name}...", flush=True)
    loader = create_dataloader(dataset, shuffle=False, pin_memory=False)
    metrics = evaluate(
        model, loader, DEVICE, RTT_LOOKUP_FOR_EVAL, is_last_epoch=True
    )
    del loader
    release_eval_memory()
    print(f"[final eval] completed {split_name}", flush=True)
    return metrics


# %%
# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_epoch(
    model, 
    train_loader, 
    optimizer, 
    device, 
    epoch_num,
    exact_rtt_criterion: ExactRttRankingLoss,
    ce_weight: float,
    regret_weight: float,
    is_last_epoch: bool = False
):
    model.train()
    running_ce = 0.0
    running_regret = 0.0
    running_total = 0.0
    n_steps = 0
    n_regret_steps = 0
    n_valid_exact = 0
    n_exact_active = 0
    n_exact_pairs = 0
    running_ce_final = 0.0
    n_final_ce = 0
    final_tasks_correct = 0
    final_tasks_total = 0
    dataset_ids_processed = set()

    print(
        f"[train_epoch] Epoch {epoch_num}: fetching first batch from DataLoader "
        f"(num_workers={getattr(train_loader, 'num_workers', 0)}, "
        f"this can take a while on first batch)...",
        flush=True,
    )
    _t_loader = time.perf_counter()
    try:
        _n_batches = len(train_loader)
    except TypeError:
        _n_batches = -1

    for step, batch in enumerate(
        tqdm(train_loader, desc=f"Epoch {epoch_num:3d} [Train]", leave=is_last_epoch)
    ):
        t_step = time.perf_counter()
        if step == 0:
            print(
                f"[train_epoch] First batch received in {t_step - _t_loader:.2f}s, "
                f"num_graphs={batch.num_graphs}",
                flush=True,
            )

        stepped = False
        try:
            optimizer.zero_grad()
            graphs_in_batch = graphs_from_batch(batch)

            loss_ce_total = torch.zeros(1, device=device)
            loss_exact_total = torch.zeros(1, device=device)
            n_graphs_ce = 0
            n_graphs_exact = 0

            for data in graphs_in_batch:
                dataset_id_saved = getattr(data, 'dataset_id', None)
                parent_dataset_id_saved = getattr(data, 'parent_dataset_id', None)
                seq_step_saved = getattr(data, 'seq_step', None)
                seq_n_tasks_saved = getattr(data, 'seq_n_tasks', None)
                opt_rtt_saved = getattr(data, 'opt_rtt', None)
                task_map_saved = getattr(
                    data,
                    'task_logit_to_placement',
                    getattr(data, '_task_logit_to_placement', {}),
                )
                queue_keys_saved = getattr(data, "task_logit_to_queue_key", {})
                initial_queue_saved = getattr(data, "initial_queue_snapshot", None)

                data = data.to(device)

                data.dataset_id = dataset_id_saved
                data.parent_dataset_id = parent_dataset_id_saved
                data.seq_step = seq_step_saved
                data.seq_n_tasks = seq_n_tasks_saved
                data.opt_rtt = opt_rtt_saved
                data.task_logit_to_placement = task_map_saved
                data.task_logit_to_queue_key = queue_keys_saved
                if initial_queue_saved is not None:
                    data.initial_queue_snapshot = initial_queue_saved

                dataset_id = getattr(data, 'dataset_id', None)
                if dataset_id:
                    dataset_ids_processed.add(dataset_id)
                            
                logits_per_task = model(data)

                # Cross-entropy loss
                loss_ce, valid_ce = loss_original_ce(logits_per_task, data, device)
                if valid_ce > 0 and not (torch.isnan(loss_ce) or torch.isinf(loss_ce)):
                    loss_ce_total = loss_ce_total + loss_ce
                    n_graphs_ce += 1
                    if is_final_sequential_graph(data):
                        running_ce_final += loss_ce.item()
                        n_final_ce += 1
                        for task_idx, task_logits in enumerate(logits_per_task):
                            if task_logits.numel() == 0 or not task_in_ce_loss(data, task_idx):
                                continue
                            target = data.y[task_idx].long()
                            if target.item() < 0 or target.item() >= task_logits.size(0):
                                continue
                            final_tasks_total += 1
                            if int(task_logits.argmax().item()) == int(target.item()):
                                final_tasks_correct += 1

                # Exact RTT pairwise ranking loss (final step only)
                loss_exact, valid_exact, stats = exact_rtt_criterion(
                    logits_per_task, data, device
                )
                if valid_exact > 0 and not (torch.isnan(loss_exact) or torch.isinf(loss_exact)):
                    loss_exact_total = loss_exact_total + loss_exact
                    n_graphs_exact += 1
                    n_exact_pairs += int(stats.get("n_pairs", 0))
                    if int(stats.get("n_active_pairs", 0)) > 0:
                        n_exact_active += 1

            if n_graphs_ce == 0:
                continue

            # Average losses
            loss_ce_avg = loss_ce_total / n_graphs_ce
            if n_graphs_exact > 0:
                loss_exact_avg = loss_exact_total / n_graphs_exact
            else:
                loss_exact_avg = torch.zeros(1, device=device)
            
            # Combined loss
            loss = ce_weight * loss_ce_avg + regret_weight * loss_exact_avg

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            stepped = True

            running_ce += loss_ce_avg.item()
            if n_graphs_exact > 0:
                running_regret += loss_exact_avg.item()
                n_regret_steps += 1
            running_total += loss.item()
            n_steps += 1
            n_valid_exact += n_graphs_exact

        finally:
            step_dt = time.perf_counter() - t_step
            is_last = _n_batches >= 0 and step + 1 == _n_batches
            periodic = (
                _TRAIN_LOG_BATCH_EVERY > 0
                and (step % _TRAIN_LOG_BATCH_EVERY == 0 or is_last)
            )
            unusual = step_dt >= _TRAIN_LOG_SLOW_STEP_SEC
            if periodic or unusual:
                tag = " SLOW" if unusual and not periodic else ""
                print(
                    f"[train_epoch] epoch={epoch_num} step={step}{tag} "
                    f"batch_secs={step_dt:.2f} stepped={stepped}",
                    flush=True,
                )
    
    print(f"\n[Epoch {epoch_num}] Processed {len(dataset_ids_processed)} datasets, exact RTT graphs: {n_valid_exact}")

    return {
        'ce': running_ce / max(1, n_steps),
        'regret_loss': running_regret / max(1, n_regret_steps),
        'total': running_total / max(1, n_steps),
        'n_valid_exact': n_valid_exact,
        'n_exact_active': n_exact_active,
        'n_exact_pairs': n_exact_pairs,
        'n_regret_steps': n_regret_steps,
        'regret_active_fraction': n_exact_active / max(1, n_valid_exact),
        'ce_final': running_ce_final / max(1, n_final_ce),
        'acc_final': final_tasks_correct / max(1, final_tasks_total),
        'n_final_ce_graphs': n_final_ce,
        'n_final_tasks': final_tasks_total,
    }


@torch.no_grad()
def decode_inference_placement(logits_per_task, data):
    """
    Greedy decoder: for each task, select the highest scoring platform.
    Non-unique version: multiple tasks can be placed on the same replica.
    """
    n_tasks = int(data.n_tasks)
    if len(logits_per_task) != n_tasks:
        return None

    task_logit_to_placement = getattr(
        data,
        'task_logit_to_placement',
        getattr(data, '_task_logit_to_placement', None),
    )
    if task_logit_to_placement is None:
        return None

    combo_list = []
    for t_idx in range(n_tasks):
        if t_idx not in task_logit_to_placement:
            return None

        logits_t = logits_per_task[t_idx].float()
        if logits_t.numel() == 0:
            return None

        best_logit_idx = logits_t.argmax().item()
        if best_logit_idx >= len(task_logit_to_placement[t_idx]):
            return None

        combo_list.append(tuple(task_logit_to_placement[t_idx][best_logit_idx]))

    return tuple(combo_list)


@torch.no_grad()
def evaluate(model, loader, device, placement_rtt_hash_table, is_last_epoch=False):
    model.eval()
    total_loss_ce = 0.0
    total_valid_tasks = 0
    correct_graphs = 0
    total_graphs = 0
    total_tasks_correct = 0
    total_tasks = 0
    
    sum_regret = 0.0
    sum_regret_pct = 0.0
    count_regret = 0
    sum_regret_decode = 0.0
    sum_regret_pct_decode = 0.0
    count_regret_decode = 0
    correct_graphs_decode = 0
    total_graphs_decode = 0
    correct_graphs_final = 0
    total_graphs_final = 0
    final_tasks_correct = 0
    final_tasks_total = 0
    count_final_graphs = 0

    per_task_count_stats = {}

    def _ensure_task_bucket(task_count: int) -> None:
        if task_count not in per_task_count_stats:
            per_task_count_stats[task_count] = {
                'correct': 0, 'total': 0, 'regret_sum': 0.0, 'regret_count': 0,
            }

    def _update_accuracy_counts(data_obj, logits_per_task_obj):
        graph_all_correct = True
        graph_valid_tasks = 0
        local_total_tasks = 0
        local_tasks_correct = 0

        for task_idx, task_logits in enumerate(logits_per_task_obj):
            if task_logits.numel() == 0:
                continue
            if not task_in_ce_loss(data_obj, task_idx):
                continue

            target = data_obj.y[task_idx].long()
            if target.ndim == 0:
                target = target.unsqueeze(0)
            if target.item() < 0 or target.item() >= task_logits.size(0):
                continue

            pred = task_logits.argmax().item()
            is_correct = int(pred == target.item())
            local_tasks_correct += is_correct
            local_total_tasks += 1
            graph_valid_tasks += 1
            if not is_correct:
                graph_all_correct = False

        labeled_tasks = int(ce_label_mask_for_graph(data_obj).sum().item())
        graph_correct = int(
            graph_all_correct and graph_valid_tasks == labeled_tasks and labeled_tasks > 0
        )
        return local_tasks_correct, local_total_tasks, graph_correct

    def _combo_rtt_regret(data_obj, combo_tuple, dataset_id_obj):
        if combo_tuple is None or not dataset_id_obj:
            return None
        opt_rtt = getattr(data_obj, "opt_rtt", None)
        if opt_rtt is None:
            return None
        try:
            opt_rtt_val = float(opt_rtt)
        except (TypeError, ValueError):
            return None
        hash_dataset_id = getattr(data_obj, "parent_dataset_id", None) or dataset_id_obj
        pred_rtt = placement_rtt_hash_table.get((hash_dataset_id, combo_tuple))
        if pred_rtt is None:
            return None
        pred_rtt_val = float(pred_rtt)
        regret_val = pred_rtt_val - opt_rtt_val
        regret_pct_val = (regret_val / opt_rtt_val) * 100.0 if opt_rtt_val > 0 else 0.0
        return regret_val, regret_pct_val, int(data_obj.n_tasks)

    def _compute_regret_metrics(dataset_id_obj, n_tasks_obj, data_obj, logits_per_task_obj):
        combo_tuple = decode_inference_placement(logits_per_task_obj, data_obj)
        return _combo_rtt_regret(data_obj, combo_tuple, dataset_id_obj)

    def _compute_regret_metrics_sequential_argmax(dataset_id_obj, data_obj, logits_per_task_obj):
        """Final-step regret after sequential argmax + queue roll-forward."""
        if not is_final_sequential_graph(data_obj):
            return None
        task_map = getattr(
            data_obj,
            "task_logit_to_placement",
            getattr(data_obj, "_task_logit_to_placement", None),
        )
        if task_map is None:
            return None
        n_tasks = int(data_obj.n_tasks)
        combo_tuple = decode_sequential_argmax_placement(
            logits_per_task_obj,
            task_map,
            n_tasks,
            initial_queue_snapshot_for_graph(data_obj),
            getattr(data_obj, "task_logit_to_queue_key", None),
        )
        return _combo_rtt_regret(data_obj, combo_tuple, dataset_id_obj)

    for batch in tqdm(loader, desc="Evaluating", leave=is_last_epoch):
        graphs_in_batch = graphs_from_batch(batch)

        for data in graphs_in_batch:
            task_logit_to_placement_orig = getattr(
                data,
                'task_logit_to_placement',
                getattr(data, '_task_logit_to_placement', {}),
            )
            dataset_id_orig = getattr(data, 'dataset_id', None)
            parent_dataset_id_orig = getattr(data, 'parent_dataset_id', None)
            seq_step_orig = getattr(data, 'seq_step', None)
            seq_n_tasks_orig = getattr(data, 'seq_n_tasks', None)
            opt_rtt_orig = getattr(data, 'opt_rtt', None)
            queue_keys_orig = getattr(data, "task_logit_to_queue_key", {})
            initial_queue_orig = getattr(data, "initial_queue_snapshot", None)

            data = data.to(device)

            data.task_logit_to_placement = task_logit_to_placement_orig
            data.dataset_id = dataset_id_orig
            data.parent_dataset_id = parent_dataset_id_orig
            data.seq_step = seq_step_orig
            data.seq_n_tasks = seq_n_tasks_orig
            data.opt_rtt = opt_rtt_orig
            data.task_logit_to_queue_key = queue_keys_orig
            if initial_queue_orig is not None:
                data.initial_queue_snapshot = initial_queue_orig

            dataset_id = getattr(data, "dataset_id", None)
            n_tasks = int(data.n_tasks)
            logits_per_task = model(data)

            loss_ce, valid_ce = loss_original_ce(logits_per_task, data, device)
            if valid_ce > 0:
                total_loss_ce += loss_ce.item() * valid_ce
                total_valid_tasks += valid_ce
                total_graphs += 1

                _ensure_task_bucket(n_tasks)
                local_tasks_correct, local_total_tasks, graph_correct = _update_accuracy_counts(data, logits_per_task)
                total_tasks_correct += local_tasks_correct
                total_tasks += local_total_tasks
                per_task_count_stats[n_tasks]['total'] += 1
                if graph_correct:
                    correct_graphs += 1
                    per_task_count_stats[n_tasks]['correct'] += 1

                if is_final_sequential_graph(data):
                    count_final_graphs += 1
                    total_graphs_final += 1
                    if graph_correct:
                        correct_graphs_final += 1
                    for task_idx, task_logits in enumerate(logits_per_task):
                        if task_logits.numel() == 0 or not task_in_ce_loss(data, task_idx):
                            continue
                        target = data.y[task_idx].long()
                        if target.item() < 0 or target.item() >= task_logits.size(0):
                            continue
                        final_tasks_total += 1
                        if int(task_logits.argmax().item()) == int(target.item()):
                            final_tasks_correct += 1

                    regret_metrics = _compute_regret_metrics(dataset_id, n_tasks, data, logits_per_task)
                    if regret_metrics is not None:
                        regret, regret_pct, regret_task_count = regret_metrics
                        sum_regret += regret
                        sum_regret_pct += regret_pct
                        count_regret += 1
                        per_task_count_stats[regret_task_count]['regret_sum'] += regret
                        per_task_count_stats[regret_task_count]['regret_count'] += 1

                    seq_metrics = _compute_regret_metrics_sequential_argmax(
                        dataset_id, data, logits_per_task
                    )
                    if seq_metrics is not None:
                        d_regret, d_regret_pct, d_n = seq_metrics
                        sum_regret_decode += d_regret
                        sum_regret_pct_decode += d_regret_pct
                        count_regret_decode += 1
                        total_graphs_decode += 1
                        if d_regret <= 1e-9:
                            correct_graphs_decode += 1

    avg_loss_ce = total_loss_ce / max(1, total_valid_tasks)
    acc = correct_graphs / max(1, total_graphs)
    regret = sum_regret / max(1, count_regret)
    regret_pct = sum_regret_pct / max(1, count_regret)
    regret_decode = sum_regret_decode / max(1, count_regret_decode)
    regret_pct_decode = sum_regret_pct_decode / max(1, count_regret_decode)
    acc_decode = correct_graphs_decode / max(1, total_graphs_decode)
    acc_final = correct_graphs_final / max(1, total_graphs_final)
    per_task_acc_final = final_tasks_correct / max(1, final_tasks_total)
    regret_coverage = count_regret / max(1, count_final_graphs)

    print(f"\n[Evaluation] Graphs: {total_graphs}, Correct: {correct_graphs} ({acc*100:.1f}%)")
    print(
        f"  Final-step graphs: {total_graphs_final}, graph acc: {acc_final*100:.1f}%, "
        f"per-task acc: {per_task_acc_final*100:.1f}%, regret coverage: {regret_coverage*100:.1f}%"
    )
    print(f"  Per-task accuracy: {total_tasks_correct}/{total_tasks} ({total_tasks_correct/max(1,total_tasks)*100:.1f}%)")
    print(f"  Regret (frozen argmax, final-step): {count_regret} samples, Avg: {regret:.4f}s ({regret_pct:.2f}%)")
    print(
        f"  Regret (sequential argmax + queue roll, final-step): {count_regret_decode} samples, "
        f"Avg: {regret_decode:.4f}s ({regret_pct_decode:.2f}%), "
        f"optimal-batch rate: {acc_decode*100:.1f}%"
    )
    
    if IS_MERGED_CACHE and len(per_task_count_stats) > 1:
        print(f"\n  Per-task-count breakdown:")
        for n_tasks in sorted(per_task_count_stats.keys()):
            stats = per_task_count_stats[n_tasks]
            acc_n = stats['correct'] / max(1, stats['total'])
            regret_n = stats['regret_sum'] / max(1, stats['regret_count']) if stats['regret_count'] > 0 else 0.0
            print(
                f"    {n_tasks} tasks: {stats['correct']}/{stats['total']} ({acc_n*100:.1f}%), "
                f"regret: {regret_n:.4f}s ({stats['regret_count']} samples)"
            )
    
    return {
        'ce': avg_loss_ce,
        'acc': acc,
        'acc_final': acc_final,
        'per_task_acc_final': per_task_acc_final,
        'count_final_graphs': count_final_graphs,
        'regret_coverage': regret_coverage,
        'regret': regret,
        'regret_pct': regret_pct,
        'count_regret': count_regret,
        'regret_seq': regret_decode,
        'regret_pct_seq': regret_pct_decode,
        'count_regret_seq': count_regret_decode,
        'acc_seq_optimal': acc_decode,
        # Back-compat keys for older wandb dashboards
        'regret_decode': regret_decode,
        'regret_pct_decode': regret_pct_decode,
        'count_regret_decode': count_regret_decode,
        'acc_decode_optimal': acc_decode,
        'per_task_count_stats': per_task_count_stats if IS_MERGED_CACHE else {},
    }


# %%
# ========================================================================
# Load graphs from cache (filtered high-queue sequential; no overnight merge)
# ========================================================================
graphs, dataset_ids = load_graphs_from_cache(CACHE_CTX)

if len(graphs) == 0:
    print("ERROR: No graphs loaded from cache!")
    exit(1)

print(
    "[INFO] Running exact RTT training backend "
    f"('{CACHE_CTX.rtt_combos_backend}')"
)

DATA_OPTIMAL_RTT = load_optimal_rtt_from_cache(CACHE_CTX)

parent_ids_in_graphs = {
    ds_id.split("@seq")[0] for ds_id in dataset_ids
}

EXACT_RTT_MAP: ExactRttLookupMap = {}
PLACEMENT_TO_LOGIT_MAP: PlacementToLogitMap = {}
PLACEMENT_RTT_HASH_TABLE: Dict[Tuple[str, Tuple], float] = {}

if PRECOMPUTE_RTT_LOOKUPS:
    valid_combos_map = load_valid_combos_map(CACHE_CTX.cache_dir)
    if valid_combos_map is None:
        print("[INFO] valid_combos_map.pkl missing; streaming RTT chunks to build exact combos...")
        valid_combos_map = build_valid_combos_map_from_chunked_cache(
            CACHE_CTX.cache_dir,
            parent_ids_in_graphs,
        )
    else:
        valid_combos_map = {
            k: v for k, v in valid_combos_map.items() if k in parent_ids_in_graphs
        }

    for ds_id, combos in valid_combos_map.items():
        for combo, rtt in combos:
            PLACEMENT_RTT_HASH_TABLE[(ds_id, combo)] = float(rtt)
    print(
        f"[INFO] Built in-memory exact RTT hash for {len(parent_ids_in_graphs)} parent datasets "
        f"({len(PLACEMENT_RTT_HASH_TABLE):,} combos in RAM)"
    )

    regret_graphs, regret_ids = _regret_lookup_graphs(graphs, dataset_ids)
    PLACEMENT_TO_LOGIT_MAP, EXACT_RTT_MAP = build_exact_rtt_index_lookups(
        regret_graphs,
        regret_ids,
        valid_combos_map,
    )
    _alias_seq_exact_lookups(graphs, dataset_ids, EXACT_RTT_MAP)
    del valid_combos_map
    gc.collect()
else:
    print("[WARN] precompute_rtt_lookups disabled; exact RTT training will produce zero valid samples")

RTT_LOOKUP_FOR_EVAL = PLACEMENT_RTT_HASH_TABLE

for g in graphs:
    ensure_graph_ce_label_mask(g)
print_sequential_label_statistics(graphs)
print("Graphs with no edges:", sum([g.edge_index.numel() == 0 for g in graphs]), "/", len(graphs))
print("Avg edges:", np.mean([g.edge_index.size(1) for g in graphs]))

print(f"\nLoaded {len(graphs)} graphs from cache")

# ========================================================================
# Train/Val/Test Split (80/10/10) — group sequential graphs by parent dataset
# ========================================================================
def _split_graphs_by_parent(
    graphs: List[Data],
    dataset_ids: List[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[Data], List[str], List[Data], List[str], List[Data], List[str]]:
    by_parent: Dict[str, List[Tuple[Data, str]]] = {}
    for graph, graph_id in zip(graphs, dataset_ids):
        parent_id = getattr(graph, "parent_dataset_id", None) or graph_id
        by_parent.setdefault(parent_id, []).append((graph, graph_id))

    parent_keys = list(by_parent.keys())
    train_parents, temp_parents = train_test_split(
        parent_keys, test_size=test_size, random_state=random_state
    )
    val_parents, test_parents = train_test_split(
        temp_parents, test_size=0.5, random_state=random_state
    )

    def _flatten(keys: List[str]) -> Tuple[List[Data], List[str]]:
        out_g: List[Data] = []
        out_ids: List[str] = []
        for key in keys:
            for graph, graph_id in by_parent[key]:
                out_g.append(graph)
                out_ids.append(graph_id)
        return out_g, out_ids

    train_graphs, train_ids = _flatten(train_parents)
    val_graphs, val_ids = _flatten(val_parents)
    test_graphs, test_ids = _flatten(test_parents)
    return train_graphs, train_ids, val_graphs, val_ids, test_graphs, test_ids


train_graphs, train_ids, val_graphs, val_ids, test_graphs, test_ids = _split_graphs_by_parent(
    graphs, dataset_ids
)

print("Dataset split:")
print(f"  Train: {len(train_graphs)} datasets ({len(train_graphs)/len(graphs)*100:.1f}%)")
print(f"  Val:   {len(val_graphs)} datasets ({len(val_graphs)/len(graphs)*100:.1f}%)")
print(f"  Test:  {len(test_graphs)} datasets ({len(test_graphs)/len(graphs)*100:.1f}%)")

# Print task count distribution per split if merged
if IS_MERGED_CACHE:
    for split_name, split_graphs in [("Train", train_graphs), ("Val", val_graphs), ("Test", test_graphs)]:
        task_dist = {}
        for g in split_graphs:
            n = int(g.n_tasks)
            task_dist[n] = task_dist.get(n, 0) + 1
        print(f"  {split_name} task distribution: " + ", ".join([f"{n}t: {c}" for n, c in sorted(task_dist.items())]))
print()

train_dataset = GraphRttDataset(
    graphs=train_graphs,
    dataset_ids=train_ids,
    optimal_rtt_map=DATA_OPTIMAL_RTT,
)
val_dataset = GraphRttDataset(
    graphs=val_graphs,
    dataset_ids=val_ids,
    optimal_rtt_map=DATA_OPTIMAL_RTT,
)
test_dataset = GraphRttDataset(
    graphs=test_graphs,
    dataset_ids=test_ids,
    optimal_rtt_map=DATA_OPTIMAL_RTT,
)

# %%
if RUNTIME_CONFIG.wandb_api_key:
    os.environ['WANDB_API_KEY'] = RUNTIME_CONFIG.wandb_api_key

wandb.init(
    project=RUNTIME_CONFIG.wandb_project,
    entity=RUNTIME_CONFIG.wandb_entity,
    config={
        "embedding_dim": int(EMBEDDING_DIM),
        "hidden_dim": int(HIDDEN_DIM),
        "lr": float(LEARNING_RATE),
        "epochs": int(EPOCHS),
        "batch_size": int(BATCH_SIZE),
        "num_gin_layers": int(NUM_GIN_LAYERS),
        "weight_decay": float(WEIGHT_DECAY),
        "device": str(DEVICE),
        "ce_weight": float(CE_LOSS_WEIGHT),
        "regret_weight": float(REGRET_LOSS_WEIGHT),
        "rtt_scale_factor": float(RTT_SCALE_FACTOR),
        "loss_type": "CE + ExactRttRanking",
        "exact_rtt_training": True,
        "sequential_counterfactual": bool(SEQUENTIAL_CACHE),
        "cache_mode": "merged" if IS_MERGED_CACHE else "single",
        "task_count_distribution": {str(k): int(v) for k, v in TASK_COUNT_DIST.items()} if TASK_COUNT_DIST else {},
        "non_unique_placements": True,  # Flag to indicate non-unique support
        "precompute_rtt_lookups": bool(PRECOMPUTE_RTT_LOOKUPS),
        "num_exact_rtt_parent_datasets": len(EXACT_RTT_MAP),
        "num_exact_rtt_combo_rows": sum(len(v) for v in EXACT_RTT_MAP.values()),
        "rtt_combos_backend": CACHE_CTX.rtt_combos_backend,
        "num_datasets": int(len(graphs)),
        "num_train": int(len(train_graphs)),
        "num_val": int(len(val_graphs)),
        "num_test": int(len(test_graphs)),
        "init_checkpoint": os.environ.get("TRAIN_INIT_CHECKPOINT", ""),
    },
    tags=[t for t in os.environ.get("WANDB_TAGS", "exact-rtt,filtered-883").split(",") if t],
)

MODEL_FILENAME = f"{wandb.run.name}.pt"

# %%
# ========================================================================
# Initialize model
# ========================================================================
# Updated feature dimensions for HRC-parity features
task_feature_dim = 3  # [task_type_onehot(2), source_node(1)]
platform_feature_dim = 13  # [type_onehot(5), has_dnn1(1), has_dnn2(1), queue(1), temporal_state(3), target_concurrency(1), usage_ratio(1)]

model = TaskPlacementGNN(
    task_feature_dim=task_feature_dim,
    platform_feature_dim=platform_feature_dim,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_GIN_LAYERS
).to(DEVICE)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

model.apply(init_weights)

_init_ckpt = os.environ.get("TRAIN_INIT_CHECKPOINT")
if _init_ckpt:
    _init_path = Path(_init_ckpt)
    if _init_path.exists():
        model.load_state_dict(torch.load(str(_init_path), map_location=DEVICE))
        print(f"[INFO] Loaded init checkpoint: {_init_path}")
    else:
        print(f"[WARN] TRAIN_INIT_CHECKPOINT not found: {_init_path}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

exact_rtt_criterion = ExactRttRankingLoss(
    rtt_scale=RTT_SCALE_FACTOR,
    exact_rtt_map=EXACT_RTT_MAP,
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print()

# ========================================================================
# Helper function for safe logging
# ========================================================================
def safe_float(val):
    """Convert to float and handle NaN/Inf for WandB logging."""
    f = float(val)
    return f if np.isfinite(f) else 0.0


def prefix_metric_dict(metrics: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """Prefix flat scalar metrics for structured logging."""
    out: Dict[str, Any] = {}
    scalar_keys = (
        "ce", "acc", "acc_final", "per_task_acc_final", "regret_coverage", "count_final_graphs",
        "regret", "regret_pct", "count_regret",
        "regret_decode", "regret_pct_decode", "count_regret_decode", "acc_decode_optimal",
    )
    count_keys = {"count_regret", "count_regret_decode", "count_final_graphs"}
    for key in scalar_keys:
        if key in metrics:
            value = metrics[key]
            out[f"{prefix}/{key}"] = int(value) if key in count_keys else safe_float(value)
    return out

# ========================================================================
# Training loop
# ========================================================================
print("="*80)
print("TRAINING (CE + Exact RTT Pairwise Ranking)")
print("="*80)
print(f"CE Weight: {CE_LOSS_WEIGHT}, Exact RTT Weight: {REGRET_LOSS_WEIGHT}")
print()

if os.environ.get("WANDB_WATCH", "0") == "1":
    wandb.watch(model, log="gradients", log_freq=100)
else:
    print("[INFO] wandb.watch disabled (set WANDB_WATCH=1 to log gradients)")

best_val_regret = float('inf')  # Minimize regret
best_val_acc = 0
checkpoint_saved = False
model_path = Path("models") / MODEL_FILENAME

train_loader = create_dataloader(train_dataset, shuffle=True, pin_memory=(DEVICE.type == "cuda"))
val_loader = create_dataloader(val_dataset, shuffle=False, pin_memory=(DEVICE.type == "cuda"))
test_loader = create_dataloader(test_dataset, shuffle=False, pin_memory=(DEVICE.type == "cuda"))

for epoch in range(EPOCHS):
    is_last_epoch = (epoch == EPOCHS - 1)
    
    # Train
    train_losses = train_epoch(
        model, train_loader, optimizer, DEVICE, epoch,
        exact_rtt_criterion=exact_rtt_criterion,
        ce_weight=CE_LOSS_WEIGHT,
        regret_weight=REGRET_LOSS_WEIGHT,
        is_last_epoch=is_last_epoch
    )
    
    # Evaluate
    val_metrics = evaluate(
        model, val_loader, DEVICE, RTT_LOOKUP_FOR_EVAL,
        is_last_epoch=is_last_epoch
    )
    
    weighted_ce = CE_LOSS_WEIGHT * train_losses['ce']
    weighted_regret = REGRET_LOSS_WEIGHT * train_losses['regret_loss']
    loss_total = train_losses['total']
    regret_fraction = weighted_regret / loss_total if loss_total > 1e-12 else 0.0

    # Wandb logging
    log_dict = {
        "train/loss_ce": safe_float(train_losses['ce']),
        "train/loss_ce_final": safe_float(train_losses.get('ce_final', 0.0)),
        "train/acc_final": safe_float(train_losses.get('acc_final', 0.0)),
        "train/n_final_ce_graphs": int(train_losses.get('n_final_ce_graphs', 0)),
        "train/loss_regret": safe_float(train_losses['regret_loss']),
        "train/loss_total": safe_float(loss_total),
        "train/weighted_ce": safe_float(weighted_ce),
        "train/weighted_regret": safe_float(weighted_regret),
        "train/regret_loss_fraction": safe_float(regret_fraction),
        "train/regret_active_fraction": safe_float(train_losses['regret_active_fraction']),
        "train/n_valid_exact": int(train_losses.get('n_valid_exact', 0)),
        "train/n_exact_active": int(train_losses.get('n_exact_active', 0)),
        "train/n_exact_pairs": int(train_losses.get('n_exact_pairs', 0)),
        "train/n_regret_steps": int(train_losses.get('n_regret_steps', 0)),
        "config/ce_loss_weight": float(CE_LOSS_WEIGHT),
        "config/regret_loss_weight": float(REGRET_LOSS_WEIGHT),
        "val/loss_ce": safe_float(val_metrics['ce']),
        "val/acc": safe_float(val_metrics['acc']),
        "val/acc_final": safe_float(val_metrics.get('acc_final', 0.0)),
        "val/per_task_acc_final": safe_float(val_metrics.get('per_task_acc_final', 0.0)),
        "val/regret_coverage": safe_float(val_metrics.get('regret_coverage', 0.0)),
        "val/count_final_graphs": int(val_metrics.get('count_final_graphs', 0)),
        "val/regret": safe_float(val_metrics['regret']),
        "val/regret_pct": safe_float(val_metrics['regret_pct']),
        "val/count_regret": int(val_metrics['count_regret']),
        "val/regret_seq": safe_float(val_metrics.get('regret_seq', 0.0)),
        "val/regret_pct_seq": safe_float(val_metrics.get('regret_pct_seq', 0.0)),
        "val/acc_seq_optimal": safe_float(val_metrics.get('acc_seq_optimal', 0.0)),
        "lr": safe_float(optimizer.param_groups[0]["lr"]),
    }
    
    # Add per-task-count statistics if merged cache
    if IS_MERGED_CACHE:
        per_task_stats = val_metrics.get('per_task_count_stats', {})
        for n_tasks, stats in per_task_stats.items():
            acc_n = stats['correct'] / max(1, stats['total'])
            regret_n = stats['regret_sum'] / max(1, stats['regret_count']) if stats['regret_count'] > 0 else 0.0
            log_dict[f"val/{n_tasks}tasks_acc"] = safe_float(acc_n)
            log_dict[f"val/{n_tasks}tasks_regret"] = safe_float(regret_n)
            log_dict[f"val/{n_tasks}tasks_count"] = int(stats['total'])
    
    wandb.log(log_dict, step=epoch)
    
    # Checkpoint on frozen argmax regret (model decides placements on final graph).
    regret_for_ckpt = val_metrics['regret']
    count_for_ckpt = val_metrics['count_regret']
    if count_for_ckpt > 0 and regret_for_ckpt < best_val_regret:
        best_val_regret = regret_for_ckpt
        best_val_acc = val_metrics['acc']
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), str(model_path))
        checkpoint_saved = True
        print(
            f"  *** New best model (frozen argmax regret): regret={best_val_regret:.4f}s, "
            f"acc={best_val_acc*100:.1f}%"
        )

    if epoch % 10 == 0 or epoch == EPOCHS - 1:
        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"Train CE: {train_losses['ce']:.4f} | "
              f"Train CE final: {train_losses.get('ce_final', 0):.4f} | "
              f"Train acc final: {train_losses.get('acc_final', 0)*100:.1f}% | "
              f"Train Exact RTT: {train_losses['regret_loss']:.4f} ({train_losses.get('n_valid_exact', 0)} graphs) | "
              f"Exact active: {train_losses['regret_active_fraction']*100:.1f}% | "
              f"Val acc final: {val_metrics.get('acc_final', 0)*100:.2f}% | "
              f"Val Regret: {val_metrics['regret']:.4f}s (coverage {val_metrics.get('regret_coverage', 0)*100:.0f}%) | "
              f"Val Regret seq: {val_metrics.get('regret_seq', val_metrics.get('regret_decode', 0)):.4f}s")

print()
if checkpoint_saved:
    print(f"Best validation regret: {best_val_regret:.4f}s (acc: {best_val_acc*100:.2f}%)")
else:
    print("Best validation regret: no valid regret checkpoint saved")

# ========================================================================
# Final Evaluation
# ========================================================================
print()
print("="*80)
print("FINAL EVALUATION")
print("="*80)

if not checkpoint_saved or not model_path.exists():
    raise RuntimeError(
        "No regret-valid checkpoint was saved. "
        "Training produced zero valid regret samples; fix regret data path before evaluating."
    )
model.load_state_dict(torch.load(str(model_path), map_location=DEVICE))
release_eval_memory()

train_metrics = run_final_split_eval(model, train_dataset, "train")
val_metrics_final = run_final_split_eval(model, val_dataset, "val")
test_metrics = run_final_split_eval(model, test_dataset, "test")

# ========================================================================
# WANDB
# ========================================================================
wandb.log({
    "data/num_datasets_total": int(len(graphs)),
    "data/num_train": int(len(train_graphs)),
    "data/num_val": int(len(val_graphs)),
    "data/num_test": int(len(test_graphs)),
})

final_metrics_log = {}
final_metrics_log.update(prefix_metric_dict(train_metrics, "final/train"))
final_metrics_log.update(prefix_metric_dict(val_metrics_final, "final/val"))
final_metrics_log.update(prefix_metric_dict(test_metrics, "final/test"))

# Add per-task-count statistics if merged cache
if IS_MERGED_CACHE:
    for split_name, metrics in [("train", train_metrics), ("val", val_metrics_final), ("test", test_metrics)]:
        per_task_stats = metrics.get('per_task_count_stats', {})
        for n_tasks, stats in per_task_stats.items():
            acc_n = stats['correct'] / max(1, stats['total'])
            regret_n = stats['regret_sum'] / max(1, stats['regret_count']) if stats['regret_count'] > 0 else 0.0
            final_metrics_log[f"final/{split_name}/{n_tasks}tasks_acc"] = safe_float(acc_n)
            final_metrics_log[f"final/{split_name}/{n_tasks}tasks_regret"] = safe_float(regret_n)
            final_metrics_log[f"final/{split_name}/{n_tasks}tasks_count"] = int(stats['total'])

wandb.log(final_metrics_log)

wandb.summary["train_dataset_ids"] = train_ids
wandb.summary["val_dataset_ids"] = val_ids
wandb.summary["test_dataset_ids"] = test_ids
wandb.summary["best_val_regret"] = float(best_val_regret)
wandb.summary["best_val_acc"] = float(best_val_acc)
wandb.summary["final_test_acc"] = float(test_metrics['acc'])
wandb.summary["final_test_regret"] = float(test_metrics['regret'])
wandb.summary["final_test_regret_pct"] = float(test_metrics['regret_pct'])

# Add per-task-count summary if merged cache
if IS_MERGED_CACHE:
    per_task_stats = test_metrics.get('per_task_count_stats', {})
    for n_tasks, stats in per_task_stats.items():
        acc_n = stats['correct'] / max(1, stats['total'])
        regret_n = stats['regret_sum'] / max(1, stats['regret_count']) if stats['regret_count'] > 0 else 0.0
        wandb.summary[f"test_{n_tasks}tasks_acc"] = float(acc_n)
        wandb.summary[f"test_{n_tasks}tasks_regret"] = float(regret_n)

artifact = wandb.Artifact("placement-gnn-regret", type="model")
artifact.add_file(str(model_path))
wandb.log_artifact(artifact)

wandb.finish()

# ========================================================================
# Local logging
# ========================================================================
print(f"\nTrain: CE={train_metrics['ce']:.4f}, Acc={train_metrics['acc']*100:.2f}%, Regret={train_metrics['regret']:.4f}s")
print(f"Val:   CE={val_metrics_final['ce']:.4f}, Acc={val_metrics_final['acc']*100:.2f}%, Regret={val_metrics_final['regret']:.4f}s")
print(f"Test:  CE={test_metrics['ce']:.4f}, Acc={test_metrics['acc']*100:.2f}%, Regret={test_metrics['regret']:.4f}s")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"Model saved to: {model_path}")
print(f"Best validation regret: {best_val_regret:.4f}s")
print(f"Best validation accuracy: {best_val_acc*100:.2f}%")
