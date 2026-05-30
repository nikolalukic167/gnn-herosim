# %%
# %%
#!/usr/bin/env python3
"""
GNN for Task-to-Platform Placement Prediction - SEQUENTIAL COUNTERFACTUAL TRAINING.

Trains on graphs built by prepare_graphs_cache_seq.py (one graph per batch decision step).
Uses a combined loss:
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

import gc
import os
import random
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
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb

# Temporary timing logs — delete this block when no longer needed.
_TRAIN_LOG_BATCH_EVERY = 25
_TRAIN_LOG_SLOW_STEP_SEC = 20.0

from non_unique_lib.cache_io import (
    HardNegativeMap,
    PlacementToLogitMap,
    build_regret_training_lookups_from_hash_table,
    create_cache_context,
    load_graphs_from_cache,
    load_optimal_rtt_from_cache,
    load_rtt_hash_table_from_cache,
)
from non_unique_lib.training_config import parse_training_config


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# %%
# Configuration (default cache: sequential counterfactual graphs from prepare_graphs_cache_seq.py)
_DEFAULT_SEQ_CACHE_DIR = (
    Path(__file__).resolve().parents[2]
    / "simulation_data"
    / "artifacts"
    / "run_queue_big"
    / "graphs_cache_gnn_datasets_4tasks_seq"
)

RUNTIME_CONFIG = parse_training_config()
if "--cache-dir" not in sys.argv:
    RUNTIME_CONFIG = replace(RUNTIME_CONFIG, cache_dir=_DEFAULT_SEQ_CACHE_DIR)

CACHE_CTX = create_cache_context(RUNTIME_CONFIG.cache_dir)
SEQUENTIAL_CACHE = bool(CACHE_CTX.metadata.get("sequential_counterfactual", False))

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
HARD_NEGATIVE_FRACTION = RUNTIME_CONFIG.hard_negative_fraction

print(f"Cache directory: {CACHE_CTX.cache_dir}")
print(f"Cache mode: {'MERGED' if IS_MERGED_CACHE else 'SINGLE'}")
print(f"Sequential counterfactual cache: {SEQUENTIAL_CACHE}")
if not SEQUENTIAL_CACHE:
    print(
        "WARNING: metadata.sequential_counterfactual is false; "
        "run prepare_graphs_cache_seq.py and pass --cache-dir to match."
    )
if TASK_COUNT_DIST:
    print("Task count distribution in cache:")
    for n_tasks, count in sorted(TASK_COUNT_DIST.items(), key=lambda x: int(x[0])):
        print(f"  {n_tasks} tasks: {count} graphs")

print(
    f"DataLoader num_workers={NUM_DATALOADER_WORKERS}, "
    f"precompute_rtt_lookups={PRECOMPUTE_RTT_LOOKUPS}, "
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

    Hard-negative pools and placement->logit maps are precomputed once at startup
    so each forward pass only does O(1) hash-map lookups.
    """

    def __init__(
        self,
        rtt_scale: float,
        hard_negative_map: HardNegativeMap,
        placement_to_logit_map: PlacementToLogitMap,
    ) -> None:
        super().__init__()
        self.rtt_scale = rtt_scale
        self.hard_negative_map = hard_negative_map
        self.placement_to_logit_map = placement_to_logit_map

    def forward(
        self,
        logits_per_task: List[torch.Tensor],
        data: Data,
        device: torch.device,
    ) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        dataset_id = getattr(data, 'dataset_id', None)
        opt_rtt = getattr(data, 'opt_rtt', None)
        task_logit_to_placement = getattr(
            data,
            'task_logit_to_placement',
            getattr(data, '_task_logit_to_placement', None),
        )
        parent_dataset_id = getattr(data, "parent_dataset_id", None)
        regret_lookup_id = parent_dataset_id or dataset_id
        hard_negative_combos = self.hard_negative_map.get(regret_lookup_id or "", [])

        if (
            not dataset_id
            or task_logit_to_placement is None
            or opt_rtt is None
            or not hard_negative_combos
        ):
            return torch.tensor(0.0, device=device), 0, {}

        n_tasks = int(data.n_tasks)
        seq_step = getattr(data, "seq_step", None)
        seq_n_tasks = getattr(data, "seq_n_tasks", None)
        if seq_step is not None and seq_n_tasks is not None and int(seq_step) != int(seq_n_tasks) - 1:
            return torch.tensor(0.0, device=device), 0, {}

        for t_idx in range(n_tasks):
            if data.y[t_idx].item() == -1:
                return torch.tensor(0.0, device=device), 0, {}
            if t_idx not in task_logit_to_placement:
                return torch.tensor(0.0, device=device), 0, {}

        score_opt = torch.tensor(0.0, device=device)
        opt_indices = []

        for t_idx in range(n_tasks):
            opt_idx = data.y[t_idx].item()
            if opt_idx >= logits_per_task[t_idx].numel():
                return torch.tensor(0.0, device=device), 0, {}
            score_opt = score_opt + logits_per_task[t_idx][opt_idx]
            opt_indices.append(opt_idx)

        neg_combo, neg_rtt = random.choice(hard_negative_combos)

        placement_to_logit_by_task = self.placement_to_logit_map.get(
            regret_lookup_id or dataset_id
        )
        neg_indices = []
        for t_idx in range(n_tasks):
            target_node_id, target_plat_id = neg_combo[t_idx]
            found_idx = None
            if placement_to_logit_by_task and t_idx < len(placement_to_logit_by_task):
                found_idx = placement_to_logit_by_task[t_idx].get((target_node_id, target_plat_id))
            if found_idx is None:
                for logit_idx, (node_id, plat_id) in enumerate(task_logit_to_placement[t_idx]):
                    if node_id == target_node_id and plat_id == target_plat_id:
                        found_idx = logit_idx
                        break
            if found_idx is None:
                return torch.tensor(0.0, device=device), 0, {}
            neg_indices.append(found_idx)

        score_neg = torch.tensor(0.0, device=device)
        for t_idx in range(n_tasks):
            neg_idx = neg_indices[t_idx]
            if neg_idx >= logits_per_task[t_idx].numel():
                return torch.tensor(0.0, device=device), 0, {}
            score_neg = score_neg + logits_per_task[t_idx][neg_idx]

        try:
            opt_rtt_val = float(opt_rtt)
        except (TypeError, ValueError):
            return torch.tensor(0.0, device=device), 0, {}
        regret = (float(neg_rtt) - opt_rtt_val) / self.rtt_scale
        regret = max(0.0, regret)

        margin = score_opt - score_neg
        loss = F.relu(torch.tensor(regret, device=device) - margin)

        stats = {
            'regret': regret,
            'margin': margin.item(),
            'score_opt': score_opt.item(),
            'score_neg': score_neg.item(),
        }

        return loss, 1, stats


class GraphRttDataset(torch.utils.data.Dataset):
    """Lightweight dataset: only attaches dataset_id, opt_rtt, and task placement map."""

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
        graph.opt_rtt = float(self.optimal_rtt_map.get(dataset_id, self.optimal_rtt_map.get(rtt_key, 0.0)))
        graph.task_logit_to_placement = self._task_map_by_dataset.get(
            dataset_id,
            self._task_map_by_dataset.get(rtt_key, {}),
        )
        return graph


# %%
# ============================================================================
# CUSTOM COLLATE AND ATTRIBUTE RESTORATION
# ============================================================================

def custom_collate(data_list):
    """Batch graphs while preserving non-tensor custom attributes."""
    task_maps = [
        getattr(d, 'task_logit_to_placement', getattr(d, '_task_logit_to_placement', {}))
        for d in data_list
    ]
    dataset_ids = [getattr(d, 'dataset_id', None) for d in data_list]
    parent_dataset_ids = [getattr(d, 'parent_dataset_id', None) for d in data_list]
    seq_steps = [getattr(d, 'seq_step', None) for d in data_list]
    seq_n_tasks_list = [getattr(d, 'seq_n_tasks', None) for d in data_list]
    opt_rtts = [getattr(d, 'opt_rtt', None) for d in data_list]
    batch = Batch.from_data_list(data_list)
    batch.task_logit_to_placement_list = task_maps
    batch.dataset_id_list = dataset_ids
    batch.parent_dataset_id_list = parent_dataset_ids
    batch.seq_step_list = seq_steps
    batch.seq_n_tasks_list = seq_n_tasks_list
    batch.opt_rtt_list = opt_rtts
    return batch


def restore_custom_attrs(batch, graphs):
    """Restore custom attrs from collate metadata lists."""
    task_maps = getattr(batch, 'task_logit_to_placement_list', [])
    dataset_ids = getattr(batch, 'dataset_id_list', [])
    parent_dataset_ids = getattr(batch, 'parent_dataset_id_list', [])
    seq_steps = getattr(batch, 'seq_step_list', [])
    seq_n_tasks_list = getattr(batch, 'seq_n_tasks_list', [])
    opt_rtts = getattr(batch, 'opt_rtt_list', [])

    for idx, graph in enumerate(graphs):
        if idx < len(task_maps):
            graph.task_logit_to_placement = task_maps[idx]
        if idx < len(dataset_ids):
            graph.dataset_id = dataset_ids[idx]
        if idx < len(parent_dataset_ids):
            graph.parent_dataset_id = parent_dataset_ids[idx]
        if idx < len(seq_steps):
            graph.seq_step = seq_steps[idx]
        if idx < len(seq_n_tasks_list):
            graph.seq_n_tasks = seq_n_tasks_list[idx]
        if idx < len(opt_rtts):
            graph.opt_rtt = opt_rtts[idx]
    return graphs


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


def _alias_seq_regret_lookups(
    graphs: List[Data],
    dataset_ids: List[str],
    placement_to_logit: PlacementToLogitMap,
    hard_negative_map: HardNegativeMap,
) -> None:
    """Copy parent regret lookup tables onto each sequential graph id."""
    for graph, graph_id in zip(graphs, dataset_ids):
        parent_id = getattr(graph, "parent_dataset_id", None)
        if not parent_id:
            continue
        if parent_id in hard_negative_map:
            hard_negative_map[graph_id] = hard_negative_map[parent_id]
        if parent_id in placement_to_logit:
            placement_to_logit[graph_id] = placement_to_logit[parent_id]
        elif graph_id not in placement_to_logit:
            task_map = getattr(graph, "task_logit_to_placement", None)
            if task_map:
                placement_to_logit[graph_id] = [
                    {placement: idx for idx, placement in enumerate(task_map.get(t_idx, []))}
                    for t_idx in range(int(graph.n_tasks))
                ]


def create_dataloader(dataset, *, shuffle: bool, pin_memory: bool) -> DataLoader:
    kw: Dict[str, Any] = dict(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_DATALOADER_WORKERS,
        pin_memory=pin_memory,
        collate_fn=custom_collate,
    )
    if NUM_DATALOADER_WORKERS > 0:
        kw["prefetch_factor"] = RUNTIME_CONFIG.dataloader_prefetch_factor
        kw["persistent_workers"] = RUNTIME_CONFIG.persistent_dataloader_workers
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
        model, loader, DEVICE, PLACEMENT_RTT_HASH_TABLE, is_last_epoch=True
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
    n_regret_active = 0
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
            graphs_in_batch = batch.to_data_list()
            graphs_in_batch = restore_custom_attrs(batch, graphs_in_batch)

            loss_ce_total = torch.zeros(1, device=device)
            loss_regret_total = torch.zeros(1, device=device)
            n_graphs_ce = 0
            n_graphs_regret = 0

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

                data = data.to(device)

                data.dataset_id = dataset_id_saved
                data.parent_dataset_id = parent_dataset_id_saved
                data.seq_step = seq_step_saved
                data.seq_n_tasks = seq_n_tasks_saved
                data.opt_rtt = opt_rtt_saved
                data.task_logit_to_placement = task_map_saved

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
                    if loss_regret.item() > 1e-8:
                        n_regret_active += 1

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
            unusual = step_dt >= _TRAIN_LOG_SLOW_STEP_SEC
            if periodic or unusual:
                tag = " SLOW" if unusual and not periodic else ""
                print(
                    f"[train_epoch] epoch={epoch_num} step={step}{tag} "
                    f"batch_secs={step_dt:.2f} stepped={stepped}",
                    flush=True,
                )
    
    print(f"\n[Epoch {epoch_num}] Processed {len(dataset_ids_processed)} datasets, valid regret samples: {n_valid_regret}")

    return {
        'ce': running_ce / max(1, n_steps),
        'regret_loss': running_regret / max(1, n_steps),
        'total': running_total / max(1, n_steps),
        'n_valid_regret': n_valid_regret,
        'n_regret_active': n_regret_active,
        'regret_active_fraction': n_regret_active / max(1, n_valid_regret),
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

        labeled_tasks = sum(
            1 for task_idx in range(int(data_obj.n_tasks))
            if data_obj.y[task_idx].item() >= 0
        )
        graph_correct = int(
            graph_all_correct and graph_valid_tasks == labeled_tasks and labeled_tasks > 0
        )
        return local_tasks_correct, local_total_tasks, graph_correct

    def _compute_regret_metrics(dataset_id_obj, n_tasks_obj, data_obj, logits_per_task_obj):
        if not dataset_id_obj:
            return None
        combo_tuple = decode_inference_placement(logits_per_task_obj, data_obj)
        if combo_tuple is None:
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
        return regret_val, regret_pct_val, n_tasks_obj

    for batch in tqdm(loader, desc="Evaluating", leave=is_last_epoch):
        graphs_in_batch = batch.to_data_list()
        graphs_in_batch = restore_custom_attrs(batch, graphs_in_batch)
        
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

            data = data.to(device)

            data.task_logit_to_placement = task_logit_to_placement_orig
            data.dataset_id = dataset_id_orig
            data.parent_dataset_id = parent_dataset_id_orig
            data.seq_step = seq_step_orig
            data.seq_n_tasks = seq_n_tasks_orig
            data.opt_rtt = opt_rtt_orig

            dataset_id = data.dataset_id
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

print(
    "[INFO] Running preloaded RTT hash-table backend "
    f"('{CACHE_CTX.rtt_combos_backend}')"
)

PLACEMENT_RTT_HASH_TABLE = load_rtt_hash_table_from_cache(CACHE_CTX.cache_dir)
DATA_OPTIMAL_RTT = load_optimal_rtt_from_cache(CACHE_CTX)
print(f"[dbg] placement_rtt combos: {len(PLACEMENT_RTT_HASH_TABLE):,}")

if PRECOMPUTE_RTT_LOOKUPS:
    regret_graphs, regret_ids = _regret_lookup_graphs(graphs, dataset_ids)
    PLACEMENT_TO_LOGIT_MAP, HARD_NEGATIVE_MAP = build_regret_training_lookups_from_hash_table(
        regret_graphs,
        regret_ids,
        PLACEMENT_RTT_HASH_TABLE,
        hard_negative_fraction=HARD_NEGATIVE_FRACTION,
    )
    _alias_seq_regret_lookups(graphs, dataset_ids, PLACEMENT_TO_LOGIT_MAP, HARD_NEGATIVE_MAP)
    gc.collect()
else:
    print("[WARN] precompute_rtt_lookups disabled; regret training will produce zero valid samples")
    PLACEMENT_TO_LOGIT_MAP = {}
    HARD_NEGATIVE_MAP = {}

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
        "loss_type": "CE + StructuredRegret",
        "sequential_counterfactual": bool(SEQUENTIAL_CACHE),
        "cache_mode": "merged" if IS_MERGED_CACHE else "single",
        "task_count_distribution": {str(k): int(v) for k, v in TASK_COUNT_DIST.items()} if TASK_COUNT_DIST else {},
        "non_unique_placements": True,  # Flag to indicate non-unique support
        "precompute_rtt_lookups": bool(PRECOMPUTE_RTT_LOOKUPS),
        "hard_negative_fraction": float(HARD_NEGATIVE_FRACTION),
        "rtt_combos_backend": CACHE_CTX.rtt_combos_backend,
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

regret_criterion = StructuredRegretLoss(
    rtt_scale=RTT_SCALE_FACTOR,
    hard_negative_map=HARD_NEGATIVE_MAP,
    placement_to_logit_map=PLACEMENT_TO_LOGIT_MAP,
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
        "ce", "acc",
        "regret", "regret_pct", "count_regret",
    )
    count_keys = {"count_regret"}
    for key in scalar_keys:
        if key in metrics:
            value = metrics[key]
            out[f"{prefix}/{key}"] = int(value) if key in count_keys else safe_float(value)
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
        regret_criterion=regret_criterion,
        ce_weight=CE_LOSS_WEIGHT,
        regret_weight=REGRET_LOSS_WEIGHT,
        is_last_epoch=is_last_epoch
    )
    
    # Evaluate
    val_metrics = evaluate(
        model, val_loader, DEVICE, PLACEMENT_RTT_HASH_TABLE,
        is_last_epoch=is_last_epoch
    )
    
    weighted_ce = CE_LOSS_WEIGHT * train_losses['ce']
    weighted_regret = REGRET_LOSS_WEIGHT * train_losses['regret_loss']
    loss_total = train_losses['total']
    regret_fraction = weighted_regret / loss_total if loss_total > 1e-12 else 0.0

    # Wandb logging
    log_dict = {
        "train/loss_ce": safe_float(train_losses['ce']),
        "train/loss_regret": safe_float(train_losses['regret_loss']),
        "train/loss_total": safe_float(loss_total),
        "train/weighted_ce": safe_float(weighted_ce),
        "train/weighted_regret": safe_float(weighted_regret),
        "train/regret_loss_fraction": safe_float(regret_fraction),
        "train/regret_active_fraction": safe_float(train_losses['regret_active_fraction']),
        "train/n_valid_regret": int(train_losses['n_valid_regret']),
        "train/n_regret_active": int(train_losses['n_regret_active']),
        "config/ce_loss_weight": float(CE_LOSS_WEIGHT),
        "config/regret_loss_weight": float(REGRET_LOSS_WEIGHT),
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
    
    # Save best model only on valid regret improvement.
    if val_metrics['count_regret'] > 0 and val_metrics['regret'] < best_val_regret:
        best_val_regret = val_metrics['regret']
        best_val_acc = val_metrics['acc']
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), str(model_path))
        checkpoint_saved = True
        print(
            f"  *** New best model (regret): regret={best_val_regret:.4f}s, "
            f"acc={best_val_acc*100:.1f}%"
        )

    if epoch % 10 == 0 or epoch == EPOCHS - 1:
        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"Train CE: {train_losses['ce']:.4f} | "
              f"Train Regret: {train_losses['regret_loss']:.4f} | "
              f"Regret active: {train_losses['regret_active_fraction']*100:.1f}% | "
              f"Regret frac: {regret_fraction*100:.1f}% | "
              f"Val Acc: {val_metrics['acc']*100:.2f}% | "
              f"Val Regret: {val_metrics['regret']:.4f}s")

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
