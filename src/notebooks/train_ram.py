# %%
#!/usr/bin/env python3
"""
GNN for Task-to-Platform Placement Prediction - REGRET-FOCUSED TRAINING (NON-UNIQUE VERSION)

**RAM cache path:** uses ``prepare_graphs_ram.py`` output: RTT combos are **embedded** in each graph
inside ``graphs.pkl``. Training loads them with ``pickle.load`` once; ``__getitem__`` does not read
LMDB or extra per-sample files (only sets ``dataset_id`` / ``opt_rtt`` on the shared graph objects).

This script **requires** ``metadata.json`` with ``rtt_combos_backend: embedded_in_graphs``.
Copy the prepared cache to the cluster under ``/share/...`` and optionally point ``--cache-dir`` at
node-local scratch for a single fast read at process start.

This version uses a combined loss:
  Loss = alpha * CrossEntropy + beta * StructuredRegretLoss

The StructuredRegretLoss:
1. Samples negative placements from the RTT hash table (valid but suboptimal combos)
2. Computes margin loss: max(0, Regret - (Score_Opt - Score_Neg))
3. Directly optimizes for lower regret, not just classification accuracy

NON-UNIQUE PLACEMENTS:
- Multiple tasks can be placed on the same replica (node_id, platform_id)
- Decoder uses greedy per-task selection (no uniqueness constraint)
- Supports datasets: gnn_datasets_2tasks, gnn_datasets_3tasks, and gnn_datasets_4tasks
"""

import os
import pickle
import random
import json
import time
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.nn.models import GIN
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb

# Temporary timing logs — delete this block when no longer needed.
_TRAIN_LOG_BATCH_EVERY = 25
_TRAIN_LOG_SLOW_STEP_SEC = 20.0
_TRAIN_LOG_HUGE_COMBOS = 200_000

from non_unique_lib.cache_io import (
    create_cache_context,
    load_graphs_from_cache,
    load_optimal_rtt_from_cache,
)
from non_unique_lib.training_config import parse_training_config


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# %%
# Configuration
RUNTIME_CONFIG = parse_training_config()
CACHE_CTX = create_cache_context(RUNTIME_CONFIG.cache_dir)

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
DATALOADER_PREFETCH_FACTOR = RUNTIME_CONFIG.dataloader_prefetch_factor
PERSISTENT_DATALOADER_WORKERS = RUNTIME_CONFIG.persistent_dataloader_workers
PRECOMPUTE_RTT_LOOKUPS = RUNTIME_CONFIG.precompute_rtt_lookups
HARD_NEGATIVE_FRACTION = RUNTIME_CONFIG.hard_negative_fraction

if RUNTIME_CONFIG.torch_threads > 0:
    torch.set_num_threads(RUNTIME_CONFIG.torch_threads)
elif os.environ.get("SLURM_CPUS_PER_TASK"):
    torch.set_num_threads(max(1, int(os.environ["SLURM_CPUS_PER_TASK"])))
if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

print(f"Cache directory: {CACHE_CTX.cache_dir}")
print(f"Cache mode: {'MERGED' if IS_MERGED_CACHE else 'SINGLE'}")
if TASK_COUNT_DIST:
    print("Task count distribution in cache:")
    for n_tasks, count in sorted(TASK_COUNT_DIST.items(), key=lambda x: int(x[0])):
        print(f"  {n_tasks} tasks: {count} graphs")

_dl_help = "embedded RTT on graphs (single pickle load; no LMDB)"
print(f"DataLoader num_workers={NUM_DATALOADER_WORKERS} ({_dl_help})")
print(f"torch num_threads={torch.get_num_threads()}")
print(
    f"RAM lookup precompute={PRECOMPUTE_RTT_LOOKUPS}, "
    f"hard_negative_fraction={HARD_NEGATIVE_FRACTION}"
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


class StructuredRegretLoss(nn.Module):
    """
    Margin-based loss that directly optimizes for regret.
    
    Loss = max(0, Regret(Negative) - (Score(Optimal) - Score(Negative)))
    
    This encourages the model to:
    1. Score the optimal placement higher than negative placements
    2. Make the score gap proportional to the regret (RTT difference)
    """
    
    def __init__(self, rtt_scale: float):
        super().__init__()
        self.rtt_scale = rtt_scale
    
    def forward(
        self,
        logits_per_task: List[torch.Tensor],
        data: Data,
        device: torch.device
    ) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        """
        Compute structured regret loss.
        
        Returns:
            loss: The margin loss tensor
            valid: 1 if loss was computed, 0 otherwise
            stats: Dictionary with debugging stats
        """
        dataset_id = getattr(data, 'dataset_id', None)
        valid_combos = getattr(data, 'valid_combos', [])
        opt_rtt = getattr(data, 'opt_rtt', None)
        
        # Get task_logit_to_placement mapping
        task_logit_to_placement = getattr(data, '_task_logit_to_placement', None)
        
        if not dataset_id or task_logit_to_placement is None or opt_rtt is None or not valid_combos:
            return torch.tensor(0.0, device=device), 0, {}
        
        n_tasks = int(data.n_tasks)
        
        # Check all tasks have valid labels
        for t_idx in range(n_tasks):
            if data.y[t_idx].item() == -1:
                return torch.tensor(0.0, device=device), 0, {}
            if t_idx not in task_logit_to_placement:
                return torch.tensor(0.0, device=device), 0, {}
        
        # 1. Calculate Score of Optimal Path (sum of logits for optimal indices)
        score_opt = torch.tensor(0.0, device=device)
        opt_indices = []
        
        for t_idx in range(n_tasks):
            opt_idx = data.y[t_idx].item()
            if opt_idx >= logits_per_task[t_idx].numel():
                return torch.tensor(0.0, device=device), 0, {}
            score_opt = score_opt + logits_per_task[t_idx][opt_idx]
            opt_indices.append(opt_idx)
        
        # 2. Sample a Negative Path from valid combos (RAM-precomputed when available)
        opt_combo_tuple = tuple(
            task_logit_to_placement[t_idx][opt_indices[t_idx]]
            if opt_indices[t_idx] < len(task_logit_to_placement[t_idx])
            else (-1, -1)
            for t_idx in range(n_tasks)
        )
        hard_negative_combos = getattr(data, '_hard_negative_combos', None)
        if hard_negative_combos:
            neg_combo, neg_rtt = random.choice(hard_negative_combos)
        else:
            non_optimal_combos = [
                (combo, rtt) for combo, rtt in valid_combos
                if combo != opt_combo_tuple
            ]
            if not non_optimal_combos:
                return torch.tensor(0.0, device=device), 0, {}

            # Sample one, preferring high-regret ones (harder negatives)
            non_optimal_combos.sort(key=lambda x: x[1], reverse=True)
            top_half = non_optimal_combos[:max(1, len(non_optimal_combos) // 2)]
            neg_combo, neg_rtt = random.choice(top_half)

        # Map combo back to logit indices
        neg_indices = []
        placement_to_logit_by_task = getattr(data, '_placement_to_logit_by_task', None)
        for t_idx in range(n_tasks):
            target_node_id, target_plat_id = neg_combo[t_idx]
            if placement_to_logit_by_task and t_idx < len(placement_to_logit_by_task):
                found_idx = placement_to_logit_by_task[t_idx].get((target_node_id, target_plat_id))
            else:
                found_idx = None
                for logit_idx, (node_id, plat_id) in enumerate(task_logit_to_placement[t_idx]):
                    if node_id == target_node_id and plat_id == target_plat_id:
                        found_idx = logit_idx
                        break
            if found_idx is None:
                return torch.tensor(0.0, device=device), 0, {}
            neg_indices.append(found_idx)
        
        # 3. Calculate Score of Negative Path
        score_neg = torch.tensor(0.0, device=device)
        for t_idx in range(n_tasks):
            neg_idx = neg_indices[t_idx]
            if neg_idx >= logits_per_task[t_idx].numel():
                return torch.tensor(0.0, device=device), 0, {}
            score_neg = score_neg + logits_per_task[t_idx][neg_idx]
        
        # 4. Calculate Regret (normalized)
        regret = (neg_rtt - opt_rtt) / self.rtt_scale
        regret = max(0.0, regret)
        
        # 5. Compute Margin Loss
        # We want: Score_Opt > Score_Neg + Regret
        # Loss = max(0, Regret - (Score_Opt - Score_Neg))
        margin = score_opt - score_neg
        loss = F.relu(torch.tensor(regret, device=device) - margin)
        
        stats = {
            'regret': regret,
            'margin': margin.item(),
            'score_opt': score_opt.item(),
            'score_neg': score_neg.item(),
        }
        
        return loss, 1, stats


class GraphRttEmbeddedDataset(torch.utils.data.Dataset):
    """Graphs already carry ``valid_combos`` from ``prepare_graphs_ram.py`` (no LMDB I/O)."""

    def __init__(
        self,
        graphs: List[Data],
        dataset_ids: List[str],
        optimal_rtt_map: Dict[str, float],
    ) -> None:
        self.graphs = graphs
        self.dataset_ids = dataset_ids
        self.optimal_rtt_map = optimal_rtt_map

    def __len__(self) -> int:
        return len(self.dataset_ids)

    def __getitem__(self, idx: int) -> Data:
        graph = self.graphs[idx]
        dataset_id = self.dataset_ids[idx]
        graph.dataset_id = dataset_id
        graph.opt_rtt = float(self.optimal_rtt_map.get(dataset_id, 0.0))
        vc = getattr(graph, "valid_combos", None)
        if vc is None:
            graph.valid_combos = []
        return graph


def _optimal_combo_from_graph(graph: Data) -> Optional[Tuple[Tuple[int, int], ...]]:
    task_map = getattr(graph, '_task_logit_to_placement', None)
    if task_map is None:
        return None
    n_tasks = int(graph.n_tasks)
    combo: List[Tuple[int, int]] = []
    for t_idx in range(n_tasks):
        if t_idx not in task_map:
            return None
        opt_idx = int(graph.y[t_idx].item())
        placements = task_map[t_idx]
        if opt_idx < 0 or opt_idx >= len(placements):
            return None
        combo.append(placements[opt_idx])
    return tuple(combo)


def prepare_graphs_for_ram_training(
    graphs: List[Data],
    *,
    precompute_rtt_lookups: bool,
    hard_negative_fraction: float,
) -> None:
    """
    Spend RAM once so every epoch avoids sorting/scanning huge embedded RTT lists.
    This is intended for high-memory SLURM nodes and keeps training semantics unchanged.
    """
    if not precompute_rtt_lookups:
        return

    start = time.perf_counter()
    total_combo_rows = 0
    total_hard_negatives = 0
    combo_maps_built = 0
    hard_negative_fraction = min(1.0, max(0.0, hard_negative_fraction))

    for graph in tqdm(graphs, desc="Precomputing RAM training lookups", unit="graph"):
        task_map = getattr(graph, '_task_logit_to_placement', None)
        if task_map is not None:
            graph._placement_to_logit_by_task = [
                {placement: idx for idx, placement in enumerate(task_map.get(t_idx, []))}
                for t_idx in range(int(graph.n_tasks))
            ]

        valid_combos = getattr(graph, "valid_combos", None) or []
        total_combo_rows += len(valid_combos)
        if not valid_combos:
            continue

        graph._combo_to_rtt = dict(valid_combos)
        combo_maps_built += 1

        opt_combo = _optimal_combo_from_graph(graph)
        hard_candidates = [
            (combo, rtt) for combo, rtt in valid_combos
            if opt_combo is None or combo != opt_combo
        ]
        if not hard_candidates:
            graph._hard_negative_combos = []
            continue

        hard_candidates.sort(key=lambda x: x[1], reverse=True)
        keep_n = max(1, int(len(hard_candidates) * hard_negative_fraction))
        graph._hard_negative_combos = hard_candidates[:keep_n]
        total_hard_negatives += keep_n

    print(
        "Precomputed RAM training lookups: "
        f"{combo_maps_built} combo maps, {total_combo_rows:,} RTT rows, "
        f"{total_hard_negatives:,} hard negatives in {time.perf_counter() - start:.2f}s",
        flush=True,
    )


# %%
# ============================================================================
# CUSTOM COLLATE AND ATTRIBUTE RESTORATION
# ============================================================================

def custom_collate(data_list):
    """Batch graphs while preserving non-tensor custom attributes."""
    preserved_names = (
        '_task_logit_to_placement',
        'dataset_id',
        'valid_combos',
        'opt_rtt',
        '_placement_to_logit_by_task',
        '_hard_negative_combos',
        '_combo_to_rtt',
    )
    preserved: List[Dict[str, Any]] = []
    for data in data_list:
        attrs: Dict[str, Any] = {}
        for name in preserved_names:
            if hasattr(data, name):
                attrs[name] = getattr(data, name)
                delattr(data, name)
        preserved.append(attrs)

    try:
        batch = Batch.from_data_list(data_list)
    finally:
        for data, attrs in zip(data_list, preserved):
            for name, value in attrs.items():
                setattr(data, name, value)

    task_maps = [attrs.get('_task_logit_to_placement', {}) for attrs in preserved]
    dataset_ids = [attrs.get('dataset_id') for attrs in preserved]
    valid_combos = [attrs.get('valid_combos', []) for attrs in preserved]
    opt_rtts = [attrs.get('opt_rtt') for attrs in preserved]
    placement_to_logit = [attrs.get('_placement_to_logit_by_task') for attrs in preserved]
    hard_negatives = [attrs.get('_hard_negative_combos') for attrs in preserved]
    combo_to_rtt = [attrs.get('_combo_to_rtt') for attrs in preserved]
    batch._task_logit_to_placement_list = task_maps
    batch.dataset_id_list = dataset_ids
    batch.valid_combos_list = valid_combos
    batch.opt_rtt_list = opt_rtts
    batch._placement_to_logit_by_task_list = placement_to_logit
    batch._hard_negative_combos_list = hard_negatives
    batch._combo_to_rtt_list = combo_to_rtt
    return batch


def restore_custom_attrs(batch, graphs):
    """Restore custom attrs from collate metadata lists."""
    task_maps = getattr(batch, '_task_logit_to_placement_list', [])
    dataset_ids = getattr(batch, 'dataset_id_list', [])
    valid_combos = getattr(batch, 'valid_combos_list', [])
    opt_rtts = getattr(batch, 'opt_rtt_list', [])
    placement_to_logit = getattr(batch, '_placement_to_logit_by_task_list', [])
    hard_negatives = getattr(batch, '_hard_negative_combos_list', [])
    combo_to_rtt = getattr(batch, '_combo_to_rtt_list', [])

    for idx, graph in enumerate(graphs):
        if idx < len(task_maps):
            graph._task_logit_to_placement = task_maps[idx]
        if idx < len(dataset_ids):
            graph.dataset_id = dataset_ids[idx]
        if idx < len(valid_combos):
            graph.valid_combos = valid_combos[idx]
        if idx < len(opt_rtts):
            graph.opt_rtt = opt_rtts[idx]
        if idx < len(placement_to_logit):
            graph._placement_to_logit_by_task = placement_to_logit[idx]
        if idx < len(hard_negatives):
            graph._hard_negative_combos = hard_negatives[idx]
        if idx < len(combo_to_rtt):
            graph._combo_to_rtt = combo_to_rtt[idx]
    return graphs


def create_dataloader(dataset, *, shuffle: bool, pin_memory: bool) -> DataLoader:
    """``prefetch_factor=1`` when using workers limits RAM from huge embedded ``valid_combos``."""
    kw: Dict[str, Any] = dict(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_DATALOADER_WORKERS,
        pin_memory=pin_memory,
        collate_fn=custom_collate,
    )
    if NUM_DATALOADER_WORKERS > 0:
        kw["prefetch_factor"] = DATALOADER_PREFETCH_FACTOR
        kw["persistent_workers"] = PERSISTENT_DATALOADER_WORKERS
    return DataLoader(**kw)


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
    regret_criterion: StructuredRegretLoss,
    ce_weight: float,
    regret_weight: float,
    is_last_epoch: bool = False
):
    model.train()
    running_ce = 0.0
    running_regret = 0.0
    running_total = 0.0
    n_steps = 0
    n_valid_regret = 0
    dataset_ids_processed = set()

    _first_batch_help = "embedded RTT (already in memory after load_graphs_from_cache)"
    print(
        f"[train_epoch] Epoch {epoch_num}: fetching first batch from DataLoader "
        f"(num_workers={getattr(train_loader, 'num_workers', 0)}, "
        f"this can take a while on first batch due to {_first_batch_help})...",
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
        max_n = 0
        slowest_id = None
        try:
            optimizer.zero_grad()
            graphs_in_batch = batch.to_data_list()
            graphs_in_batch = restore_custom_attrs(batch, graphs_in_batch)

            combo_sizes = [len(getattr(g, "valid_combos", None) or []) for g in graphs_in_batch]
            if combo_sizes:
                max_n = max(combo_sizes)
                i_mx = combo_sizes.index(max_n)
                slowest_id = getattr(graphs_in_batch[i_mx], "dataset_id", None)

            loss_ce_total = torch.zeros(1, device=device)
            loss_regret_total = torch.zeros(1, device=device)
            n_graphs_ce = 0
            n_graphs_regret = 0

            for data in graphs_in_batch:
                dataset_id_saved = getattr(data, 'dataset_id', None)
                valid_combos_saved = getattr(data, 'valid_combos', [])
                opt_rtt_saved = getattr(data, 'opt_rtt', None)
                task_map_saved = getattr(data, '_task_logit_to_placement', {})
                placement_to_logit_saved = getattr(data, '_placement_to_logit_by_task', None)
                hard_negative_combos_saved = getattr(data, '_hard_negative_combos', None)

                data = data.to(device)

                data.dataset_id = dataset_id_saved
                data.valid_combos = valid_combos_saved
                data.opt_rtt = opt_rtt_saved
                data._task_logit_to_placement = task_map_saved
                data._placement_to_logit_by_task = placement_to_logit_saved
                data._hard_negative_combos = hard_negative_combos_saved

                dataset_id = getattr(data, 'dataset_id', None)
                if dataset_id:
                    dataset_ids_processed.add(dataset_id)
                            
                logits_per_task = model(data)

                # Cross-entropy loss
                loss_ce, valid_ce = loss_original_ce(logits_per_task, data, device)
                if valid_ce > 0 and not (torch.isnan(loss_ce) or torch.isinf(loss_ce)):
                    loss_ce_total = loss_ce_total + loss_ce
                    n_graphs_ce += 1

                # Structured regret loss
                loss_regret, valid_regret, stats = regret_criterion(
                    logits_per_task, data, device
                )
                if valid_regret > 0 and not (torch.isnan(loss_regret) or torch.isinf(loss_regret)):
                    loss_regret_total = loss_regret_total + loss_regret
                    n_graphs_regret += 1

            if n_graphs_ce == 0:
                continue

            # Average losses
            loss_ce_avg = loss_ce_total / n_graphs_ce
            loss_regret_avg = loss_regret_total / max(1, n_graphs_regret)
            
            # Combined loss
            loss = ce_weight * loss_ce_avg + regret_weight * loss_regret_avg

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            stepped = True

            running_ce += loss_ce_avg.item()
            running_regret += loss_regret_avg.item()
            running_total += loss.item()
            n_steps += 1
            n_valid_regret += n_graphs_regret

        finally:
            step_dt = time.perf_counter() - t_step
            is_last = _n_batches >= 0 and step + 1 == _n_batches
            periodic = (
                _TRAIN_LOG_BATCH_EVERY > 0
                and (step % _TRAIN_LOG_BATCH_EVERY == 0 or is_last)
            )
            unusual = step_dt >= _TRAIN_LOG_SLOW_STEP_SEC or max_n >= _TRAIN_LOG_HUGE_COMBOS
            if periodic or unusual:
                tag = ""
                if unusual and not periodic:
                    tag = " SLOW_OR_HUGE"
                print(
                    f"[train_epoch] epoch={epoch_num} step={step}{tag} "
                    f"batch_secs={step_dt:.2f} max_n_combos={max_n} "
                    f"largest={slowest_id!r} stepped={stepped}",
                    flush=True,
                )
    
    print(f"\n[Epoch {epoch_num}] Processed {len(dataset_ids_processed)} datasets, valid regret samples: {n_valid_regret}")

    return {
        'ce': running_ce / max(1, n_steps),
        'regret_loss': running_regret / max(1, n_steps),
        'total': running_total / max(1, n_steps),
        'n_valid_regret': n_valid_regret,
    }


@torch.no_grad()
def decode_inference_placement(logits_per_task, data):
    """
    Greedy decoder: for each task, select the highest scoring platform.
    Non-unique version: multiple tasks can be placed on the same replica.
    """
    dataset_id = getattr(data, 'dataset_id', None)
    if not dataset_id:
        return None

    n_tasks = int(data.n_tasks)
    if len(logits_per_task) != n_tasks:
        return None

    task_logit_to_placement = getattr(data, '_task_logit_to_placement', None)
    if task_logit_to_placement is None:
        return None

    # For each task, select the highest scoring platform (greedy per-task)
    combo_list = []
    for t_idx in range(n_tasks):
        if t_idx not in task_logit_to_placement:
            return None
        
        logits_t = logits_per_task[t_idx].float().clone()
        if logits_t.numel() == 0:
            return None
        
        # Pick highest scoring platform for this task
        best_logit_idx = logits_t.argmax().item()
        
        if best_logit_idx >= len(task_logit_to_placement[t_idx]):
            return None
        
        node_id, plat_id = task_logit_to_placement[t_idx][best_logit_idx]
        combo_list.append((node_id, plat_id))

    return tuple(combo_list)


@torch.no_grad()
def evaluate(model, loader, device, is_last_epoch=False):
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

    # Per-task-count statistics (for merged datasets)
    per_task_count_stats = {}  # {n_tasks: {correct: int, total: int, regret_sum: float, regret_count: int}}

    def _ensure_task_bucket(task_count: int) -> None:
        if task_count not in per_task_count_stats:
            per_task_count_stats[task_count] = {
                'correct': 0, 'total': 0, 'regret_sum': 0.0, 'regret_count': 0
            }

    def _update_accuracy_counts(data_obj, logits_per_task_obj):
        graph_all_correct = True
        graph_valid_tasks = 0
        local_total_tasks = 0
        local_tasks_correct = 0

        for task_idx, task_logits in enumerate(logits_per_task_obj):
            if task_logits.numel() == 0:
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

        graph_correct = int(graph_all_correct and graph_valid_tasks == data_obj.n_tasks)
        return local_tasks_correct, local_total_tasks, graph_correct

    def _compute_regret_metrics(dataset_id_obj, n_tasks_obj, data_obj, logits_per_task_obj):
        if not dataset_id_obj:
            return None
        combo_tuple = decode_inference_placement(logits_per_task_obj, data_obj)
        if combo_tuple is None:
            return None
        valid_combos = getattr(data_obj, "valid_combos", [])
        opt_rtt = getattr(data_obj, "opt_rtt", None)
        if opt_rtt is None or not valid_combos:
            return None
        combo_to_rtt = getattr(data_obj, "_combo_to_rtt", None)
        if combo_to_rtt is None:
            combo_to_rtt = {combo: rtt for combo, rtt in valid_combos}
        pred_rtt = combo_to_rtt.get(combo_tuple)
        if pred_rtt is None or opt_rtt is None:
            return None
        regret_val = float(pred_rtt - opt_rtt)
        regret_pct_val = (regret_val / opt_rtt) * 100.0 if opt_rtt > 0 else 0.0
        return regret_val, regret_pct_val, n_tasks_obj

    for batch in tqdm(loader, desc="Evaluating", leave=is_last_epoch):
        graphs_in_batch = batch.to_data_list()
        graphs_in_batch = restore_custom_attrs(batch, graphs_in_batch)
        
        for data in graphs_in_batch:
            # Preserve custom attributes (lost on device transfer)
            task_logit_to_placement_orig = getattr(data, '_task_logit_to_placement', {})
            dataset_id_orig = getattr(data, 'dataset_id', None)
            valid_combos_orig = getattr(data, 'valid_combos', [])
            opt_rtt_orig = getattr(data, 'opt_rtt', None)
            combo_to_rtt_orig = getattr(data, '_combo_to_rtt', None)

            data = data.to(device)

            data._task_logit_to_placement = task_logit_to_placement_orig
            data.dataset_id = dataset_id_orig
            data.valid_combos = valid_combos_orig
            data.opt_rtt = opt_rtt_orig
            data._combo_to_rtt = combo_to_rtt_orig

            dataset_id = data.dataset_id
            n_tasks = int(data.n_tasks)
            logits_per_task = model(data)

            # CE loss
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

                regret_metrics = _compute_regret_metrics(dataset_id, n_tasks, data, logits_per_task)
                if regret_metrics is not None:
                    regret, regret_pct, regret_task_count = regret_metrics
                    sum_regret += regret
                    sum_regret_pct += regret_pct
                    count_regret += 1
                    per_task_count_stats[regret_task_count]['regret_sum'] += regret
                    per_task_count_stats[regret_task_count]['regret_count'] += 1

    avg_loss_ce = total_loss_ce / max(1, total_valid_tasks)
    acc = correct_graphs / max(1, total_graphs)
    regret = sum_regret / max(1, count_regret)
    regret_pct = sum_regret_pct / max(1, count_regret)

    print(f"\n[Evaluation] Graphs: {total_graphs}, Correct: {correct_graphs} ({acc*100:.1f}%)")
    print(f"  Per-task accuracy: {total_tasks_correct}/{total_tasks} ({total_tasks_correct/max(1,total_tasks)*100:.1f}%)")
    print(f"  Regret: {count_regret} samples, Avg: {regret:.4f}s ({regret_pct:.2f}%)")
    
    # Print per-task-count statistics if merged cache
    if IS_MERGED_CACHE and len(per_task_count_stats) > 1:
        print(f"\n  Per-task-count breakdown:")
        for n_tasks in sorted(per_task_count_stats.keys()):
            stats = per_task_count_stats[n_tasks]
            acc_n = stats['correct'] / max(1, stats['total'])
            regret_n = stats['regret_sum'] / max(1, stats['regret_count']) if stats['regret_count'] > 0 else 0.0
            print(f"    {n_tasks} tasks: {stats['correct']}/{stats['total']} ({acc_n*100:.1f}%), "
                  f"regret: {regret_n:.4f}s ({stats['regret_count']} samples)")
    
    return {
        'ce': avg_loss_ce,
        'acc': acc,
        'regret': regret,
        'regret_pct': regret_pct,
        'count_regret': count_regret,
        'per_task_count_stats': per_task_count_stats if IS_MERGED_CACHE else {},
    }


# %%
# ========================================================================
# Load graphs from cache
# ========================================================================
graphs, dataset_ids = load_graphs_from_cache(CACHE_CTX)

if len(graphs) == 0:
    print("ERROR: No graphs loaded from cache!")
    exit(1)

if not CACHE_CTX.rtt_combos_embedded:
    raise FileNotFoundError(
        f"train_ram.py requires an embedded RTT cache (prepare_graphs_ram.py). "
        f"Expected metadata.json rtt_combos_backend='embedded_in_graphs' under {CACHE_CTX.cache_dir!s}. "
        f"For LMDB caches use train.py instead."
    )
print("Using RTT combos: embedded in graphs.pkl (prepare_graphs_ram.py); no LMDB.")

DATA_OPTIMAL_RTT = load_optimal_rtt_from_cache(CACHE_CTX)
prepare_graphs_for_ram_training(
    graphs,
    precompute_rtt_lookups=PRECOMPUTE_RTT_LOOKUPS,
    hard_negative_fraction=HARD_NEGATIVE_FRACTION,
)

# Compute statistics
ys = np.concatenate([g.y.numpy() for g in graphs])
print("Valid labels:", np.sum(ys >= 0), "/", len(ys))
print("Graphs with no edges:", sum([g.edge_index.numel() == 0 for g in graphs]), "/", len(graphs))
print("Avg edges:", np.mean([g.edge_index.size(1) for g in graphs]))
print("Avg valid tasks:", np.mean([(g.y >= 0).sum().item() for g in graphs]))
print("Max valid tasks:", np.max([(g.y >= 0).sum().item() for g in graphs]))
print("Min valid tasks:", np.min([(g.y >= 0).sum().item() for g in graphs]))

print(f"\nLoaded {len(graphs)} graphs from cache")

# ========================================================================
# Train/Val/Test Split (80/10/10)
# ========================================================================
train_graphs, temp_graphs, train_ids, temp_ids = train_test_split(
    graphs, dataset_ids, test_size=0.2, random_state=42
)
val_graphs, test_graphs, val_ids, test_ids = train_test_split(
    temp_graphs, temp_ids, test_size=0.5, random_state=42
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

train_dataset = GraphRttEmbeddedDataset(train_graphs, train_ids, DATA_OPTIMAL_RTT)
val_dataset = GraphRttEmbeddedDataset(val_graphs, val_ids, DATA_OPTIMAL_RTT)
test_dataset = GraphRttEmbeddedDataset(test_graphs, test_ids, DATA_OPTIMAL_RTT)

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
        "loss_type": "CE + StructuredRegret",
        "cache_mode": "merged" if IS_MERGED_CACHE else "single",
        "task_count_distribution": {str(k): int(v) for k, v in TASK_COUNT_DIST.items()} if TASK_COUNT_DIST else {},
        "non_unique_placements": True,  # Flag to indicate non-unique support
        "rtt_backend": "embedded_in_graphs",
        "precompute_rtt_lookups": bool(PRECOMPUTE_RTT_LOOKUPS),
        "hard_negative_fraction": float(HARD_NEGATIVE_FRACTION),
        "num_dataloader_workers": int(NUM_DATALOADER_WORKERS),
        "dataloader_prefetch_factor": int(DATALOADER_PREFETCH_FACTOR),
        "persistent_dataloader_workers": bool(PERSISTENT_DATALOADER_WORKERS),
        "torch_threads": int(torch.get_num_threads()),
        "num_datasets": int(len(graphs)),
        "num_train": int(len(train_graphs)),
        "num_val": int(len(val_graphs)),
        "num_test": int(len(test_graphs)),
    }
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

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

regret_criterion = StructuredRegretLoss(rtt_scale=RTT_SCALE_FACTOR)

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
    for key in ("ce", "acc", "regret", "regret_pct", "count_regret"):
        if key in metrics:
            value = metrics[key]
            out[f"{prefix}/{key}"] = int(value) if key == "count_regret" else safe_float(value)
    return out

# ========================================================================
# Training loop
# ========================================================================
print("="*80)
print("TRAINING (CE + Structured Regret Loss)")
print("="*80)
print(f"CE Weight: {CE_LOSS_WEIGHT}, Regret Weight: {REGRET_LOSS_WEIGHT}")
print()

wandb.watch(model, log="gradients", log_freq=100)

best_val_regret = float('inf')  # Minimize regret
best_val_acc = 0

train_loader = create_dataloader(train_dataset, shuffle=True, pin_memory=(DEVICE.type == "cuda"))
val_loader = create_dataloader(val_dataset, shuffle=False, pin_memory=(DEVICE.type == "cuda"))
test_loader = create_dataloader(test_dataset, shuffle=False, pin_memory=(DEVICE.type == "cuda"))

for epoch in range(EPOCHS):
    is_last_epoch = (epoch == EPOCHS - 1)
    
    # Train
    train_losses = train_epoch(
        model, train_loader, optimizer, DEVICE, epoch,
        regret_criterion=regret_criterion,
        ce_weight=CE_LOSS_WEIGHT,
        regret_weight=REGRET_LOSS_WEIGHT,
        is_last_epoch=is_last_epoch
    )
    
    # Evaluate
    val_metrics = evaluate(
        model, val_loader, DEVICE,
        is_last_epoch=is_last_epoch
    )
    
    # Wandb logging
    log_dict = {
        "train/loss_ce": safe_float(train_losses['ce']),
        "train/loss_regret": safe_float(train_losses['regret_loss']),
        "train/loss_total": safe_float(train_losses['total']),
        "train/n_valid_regret": int(train_losses['n_valid_regret']),
        "val/loss_ce": safe_float(val_metrics['ce']),
        "val/acc": safe_float(val_metrics['acc']),
        "val/regret": safe_float(val_metrics['regret']),
        "val/regret_pct": safe_float(val_metrics['regret_pct']),
        "val/count_regret": int(val_metrics['count_regret']),
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
    
    # Save best model only when healthy and better regret
    if val_metrics['regret'] < best_val_regret and val_metrics['count_regret'] > 0:
        best_val_regret = val_metrics['regret']
        best_val_acc = val_metrics['acc']
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/" + MODEL_FILENAME)
        print(f"  *** New best model: regret={best_val_regret:.4f}s, acc={best_val_acc*100:.1f}%")

    if epoch % 10 == 0 or epoch == EPOCHS - 1:
        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"Train CE: {train_losses['ce']:.4f} | "
              f"Train Regret: {train_losses['regret_loss']:.4f} | "
              f"Val Acc: {val_metrics['acc']*100:.2f}% | "
              f"Val Regret: {val_metrics['regret']:.4f}s")

print()
print(f"Best validation regret: {best_val_regret:.4f}s (acc: {best_val_acc*100:.2f}%)")

# ========================================================================
# Final Evaluation
# ========================================================================
print()
print("="*80)
print("FINAL EVALUATION")
print("="*80)

model.load_state_dict(torch.load("models/" + MODEL_FILENAME))

train_loader_eval = create_dataloader(train_dataset, shuffle=False, pin_memory=False)
val_loader_eval = create_dataloader(val_dataset, shuffle=False, pin_memory=False)
test_loader_eval = create_dataloader(test_dataset, shuffle=False, pin_memory=False)

train_metrics = evaluate(model, train_loader_eval, DEVICE, is_last_epoch=True)
val_metrics_final = evaluate(model, val_loader_eval, DEVICE, is_last_epoch=True)
test_metrics = evaluate(model, test_loader_eval, DEVICE, is_last_epoch=True)

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
artifact.add_file("models/" + MODEL_FILENAME)
wandb.log_artifact(artifact)

wandb.finish()

# ========================================================================
# Local logging
# ========================================================================
print(f"\nTrain: CE={train_metrics['ce']:.4f}, Acc={train_metrics['acc']*100:.2f}%, Regret={train_metrics['regret']:.4f}s ({train_metrics['regret_pct']:.2f}%)")
print(f"Val:   CE={val_metrics_final['ce']:.4f}, Acc={val_metrics_final['acc']*100:.2f}%, Regret={val_metrics_final['regret']:.4f}s ({val_metrics_final['regret_pct']:.2f}%)")
print(f"Test:  CE={test_metrics['ce']:.4f}, Acc={test_metrics['acc']*100:.2f}%, Regret={test_metrics['regret']:.4f}s ({test_metrics['regret_pct']:.2f}%)")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"Model saved to: models/{MODEL_FILENAME}")
print(f"Best validation regret: {best_val_regret:.4f}s")
print(f"Best validation accuracy: {best_val_acc*100:.2f}%")
