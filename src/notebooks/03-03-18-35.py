# %%
#!/usr/bin/env python3
"""
GNN for Task-to-Platform Placement Prediction - REGRET-FOCUSED TRAINING (NON-UNIQUE VERSION)

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
import random
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn.models import GIN
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb
from non_unique_lib.cache_io import (
    build_valid_combos_map,
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

print(f"Cache directory: {CACHE_CTX.cache_dir}")
print(f"Cache mode: {'MERGED' if IS_MERGED_CACHE else 'SINGLE'}")
if TASK_COUNT_DIST:
    print("Task count distribution in cache:")
    for n_tasks, count in sorted(TASK_COUNT_DIST.items(), key=lambda x: int(x[0])):
        print(f"  {n_tasks} tasks: {count} graphs")


# %%
# ============================================================================
# GNN MODEL (same as before)
# ============================================================================

class TaskEncoder(nn.Module):
    """2-layer MLP encoder for task features with LayerNorm for training stability."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class PlatformEncoder(nn.Module):
    """2-layer MLP encoder for platform features with LayerNorm for training stability."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class EdgeScorer(nn.Module):
    """2-layer MLP to score task-platform edges with optional edge attributes."""
    def __init__(self, embedding_dim, hidden_dim, edge_dim=0):
        super().__init__()
        in_dim = 2 * embedding_dim + (edge_dim if edge_dim else 0)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, e_task, e_platform, e_attr=None):
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
    def __init__(self, task_feature_dim, platform_feature_dim, embedding_dim=64, hidden_dim=128, num_layers=3):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.task_encoder = TaskEncoder(task_feature_dim, hidden_dim, embedding_dim)
        self.platform_encoder = PlatformEncoder(platform_feature_dim, hidden_dim, embedding_dim)
        
        self.gin = GIN(
            in_channels=embedding_dim,
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            out_channels=embedding_dim
        )
        self.post_gin_dropout = nn.Dropout(p=0.2)
        self.edge_scorer = EdgeScorer(embedding_dim, hidden_dim, edge_dim=5)  # 5 edge dims (exec, latency, warm, energy, comm)

    def forward(self, data):
        n_tasks = data.n_tasks
        n_platforms = data.n_platforms

        # Handle NaN/Inf in input
        if torch.isnan(data.task_features).any() or torch.isinf(data.task_features).any():
            data.task_features = torch.nan_to_num(data.task_features, nan=0.0, posinf=1e6, neginf=-1e6)
        if torch.isnan(data.platform_features).any() or torch.isinf(data.platform_features).any():
            data.platform_features = torch.nan_to_num(data.platform_features, nan=0.0, posinf=1e6, neginf=-1e6)

        # Encode features
        task_embeddings = self.task_encoder(data.task_features)
        platform_embeddings = self.platform_encoder(data.platform_features)
        
        if torch.isnan(task_embeddings).any() or torch.isinf(task_embeddings).any():
            task_embeddings = torch.nan_to_num(task_embeddings, nan=0.0, posinf=1e6, neginf=-1e6)
        if torch.isnan(platform_embeddings).any() or torch.isinf(platform_embeddings).any():
            platform_embeddings = torch.nan_to_num(platform_embeddings, nan=0.0, posinf=1e6, neginf=-1e6)

        # Message passing
        x = torch.cat([task_embeddings, platform_embeddings], dim=0)
        x = self.gin(x, data.edge_index)
        x = self.post_gin_dropout(x)
        
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.clamp(x, min=-50.0, max=50.0)
        
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
        e_attr = None
        if hasattr(data, 'edge_attr') and data.edge_attr.numel() > 0:
            try:
                e_attr = data.edge_attr[valid]
            except Exception:
                e_attr = None
        edge_scores = self.edge_scorer(e_task, e_platform, e_attr)
        
        if torch.isnan(edge_scores).any() or torch.isinf(edge_scores).any():
            edge_scores = torch.clamp(edge_scores, min=-50.0, max=50.0)

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
        valid_combos_map: Dict[str, List[Tuple[Tuple[Tuple[int, int], ...], float]]],
        optimal_rtt_map: Dict[str, float],
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
        
        # Get task_logit_to_placement mapping
        task_logit_to_placement = getattr(data, '_task_logit_to_placement', None)
        
        if not dataset_id or dataset_id not in optimal_rtt_map or task_logit_to_placement is None:
            return torch.tensor(0.0, device=device), 0, {}
        
        n_tasks = int(data.n_tasks)
        opt_rtt = optimal_rtt_map[dataset_id]
        
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
        
        # 2. Sample a Negative Path from valid combos (hash table)
        valid_combos = valid_combos_map.get(dataset_id, [])
        opt_combo_tuple = tuple(
            task_logit_to_placement[t_idx][opt_indices[t_idx]]
            if opt_indices[t_idx] < len(task_logit_to_placement[t_idx])
            else (-1, -1)
            for t_idx in range(n_tasks)
        )
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
        for t_idx in range(n_tasks):
            target_node_id, target_plat_id = neg_combo[t_idx]
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


# %%
# ============================================================================
# CUSTOM COLLATE AND ATTRIBUTE RESTORATION
# ============================================================================

def restore_custom_attrs(batch, graphs):
    """Restore custom attributes from global dictionary using dataset_id."""
    global GRAPH_CUSTOM_ATTRS
    
    for graph in graphs:
        dataset_id = getattr(graph, 'dataset_id', None)
        if dataset_id and dataset_id in GRAPH_CUSTOM_ATTRS:
            attrs = GRAPH_CUSTOM_ATTRS[dataset_id]
            graph._task_logit_to_placement = attrs.get('_task_logit_to_placement', {})
            graph.dataset_id = attrs.get('dataset_id', dataset_id)
    
    return graphs


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
    valid_combos_map: Dict,
    optimal_rtt_map: Dict,
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

    for batch in tqdm(train_loader, desc=f"Epoch {epoch_num:3d} [Train]", leave=is_last_epoch):
        optimizer.zero_grad()
        graphs_in_batch = batch.to_data_list()
        graphs_in_batch = restore_custom_attrs(batch, graphs_in_batch)

        loss_ce_total = torch.zeros(1, device=device)
        loss_regret_total = torch.zeros(1, device=device)
        n_graphs_ce = 0
        n_graphs_regret = 0

        for data in graphs_in_batch:
            data = data.to(device)

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
                logits_per_task, data, valid_combos_map, optimal_rtt_map, device
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

        running_ce += loss_ce_avg.item()
        running_regret += loss_regret_avg.item()
        running_total += loss.item()
        n_steps += 1
        n_valid_regret += n_graphs_regret
    
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
def evaluate(model, loader, device, placement_rtt_hash_table, optimal_rtt_map, is_last_epoch=False):
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

    for batch in tqdm(loader, desc="Evaluating", leave=is_last_epoch):
        graphs_in_batch = batch.to_data_list()
        graphs_in_batch = restore_custom_attrs(batch, graphs_in_batch)
        
        for data in graphs_in_batch:
            # Preserve custom attributes (lost on device transfer)
            task_logit_to_placement_orig = getattr(data, '_task_logit_to_placement', {})
            dataset_id_orig = getattr(data, 'dataset_id', None)
            
            data = data.to(device)
            
            data._task_logit_to_placement = task_logit_to_placement_orig
            data.dataset_id = dataset_id_orig

            dataset_id = data.dataset_id
            n_tasks = int(data.n_tasks)
            logits_per_task = model(data)

            # CE loss
            loss_ce, valid_ce = loss_original_ce(logits_per_task, data, device)
            if valid_ce > 0:
                total_loss_ce += loss_ce.item() * valid_ce
                total_valid_tasks += valid_ce
                total_graphs += 1

                # Initialize per-task-count stats if needed
                if n_tasks not in per_task_count_stats:
                    per_task_count_stats[n_tasks] = {
                        'correct': 0, 'total': 0, 'regret_sum': 0.0, 'regret_count': 0
                    }

                # Accuracy: per-task argmax over logits
                graph_all_correct = True
                graph_valid_tasks = 0
                
                for task_idx, task_logits in enumerate(logits_per_task):
                    if task_logits.numel() == 0:
                        continue

                    target = data.y[task_idx].long()
                    if target.ndim == 0:
                        target = target.unsqueeze(0)
                    if target.item() < 0 or target.item() >= task_logits.size(0):
                        continue

                    pred = task_logits.argmax().item()
                    is_correct = int(pred == target.item())
                    total_tasks_correct += is_correct
                    total_tasks += 1
                    graph_valid_tasks += 1
                    
                    if not is_correct:
                        graph_all_correct = False
                
                # Count as correct if all valid tasks are correctly predicted
                per_task_count_stats[n_tasks]['total'] += 1
                if graph_all_correct and graph_valid_tasks == data.n_tasks:
                    correct_graphs += 1
                    per_task_count_stats[n_tasks]['correct'] += 1

                # Compute regret
                if dataset_id and dataset_id in optimal_rtt_map:
                    combo_tuple = decode_inference_placement(logits_per_task, data)
                    if combo_tuple is not None:
                        hash_key = (dataset_id, combo_tuple)
                        pred_rtt = placement_rtt_hash_table.get(hash_key)
                        opt_rtt = optimal_rtt_map.get(dataset_id)
                        if pred_rtt is not None and opt_rtt is not None:
                            regret = float(pred_rtt - opt_rtt)
                            regret_pct = (regret / opt_rtt) * 100.0 if opt_rtt > 0 else 0.0
                            sum_regret += regret
                            sum_regret_pct += regret_pct
                            count_regret += 1
                            # Per-task-count regret
                            per_task_count_stats[n_tasks]['regret_sum'] += regret
                            per_task_count_stats[n_tasks]['regret_count'] += 1

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

# Load helper maps
DATA_OPTIMAL_RTT = load_optimal_rtt_from_cache(CACHE_CTX)
placement_rtt_hash_table = load_rtt_hash_table_from_cache(CACHE_CTX)
print(f"[dbg] placement_rtt combos: {len(placement_rtt_hash_table)}")

VALID_COMBOS_MAP = build_valid_combos_map(placement_rtt_hash_table)

# Compute statistics
ys = np.concatenate([g.y.numpy() for g in graphs])
print("Valid labels:", np.sum(ys >= 0), "/", len(ys))
print("Graphs with no edges:", sum([g.edge_index.numel() == 0 for g in graphs]), "/", len(graphs))
print("Avg edges:", np.mean([g.edge_index.size(1) for g in graphs]))
print("Avg valid tasks:", np.mean([(g.y >= 0).sum().item() for g in graphs]))
print("Max valid tasks:", np.max([(g.y >= 0).sum().item() for g in graphs]))
print("Min valid tasks:", np.min([(g.y >= 0).sum().item() for g in graphs]))

print(f"\nLoaded {len(graphs)} graphs from cache")

# Store custom attributes globally (only those needed at train/eval time)
GRAPH_CUSTOM_ATTRS = {}
for graph in graphs:
    dataset_id = getattr(graph, 'dataset_id', None)
    if dataset_id:
        GRAPH_CUSTOM_ATTRS[dataset_id] = {
            '_task_logit_to_placement': getattr(graph, '_task_logit_to_placement', {}),
            'dataset_id': dataset_id
        }

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

train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

for epoch in range(EPOCHS):
    is_last_epoch = (epoch == EPOCHS - 1)
    
    # Train
    train_losses = train_epoch(
        model, train_loader, optimizer, DEVICE, epoch,
        regret_criterion=regret_criterion,
        valid_combos_map=VALID_COMBOS_MAP,
        optimal_rtt_map=DATA_OPTIMAL_RTT,
        ce_weight=CE_LOSS_WEIGHT,
        regret_weight=REGRET_LOSS_WEIGHT,
        is_last_epoch=is_last_epoch
    )
    
    # Evaluate
    val_metrics = evaluate(
        model, val_loader, DEVICE, 
        placement_rtt_hash_table, DATA_OPTIMAL_RTT,
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

train_loader_eval = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
val_loader_eval = DataLoader(val_graphs, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
test_loader_eval = DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

train_metrics = evaluate(model, train_loader_eval, DEVICE, placement_rtt_hash_table, DATA_OPTIMAL_RTT, is_last_epoch=True)
val_metrics_final = evaluate(model, val_loader_eval, DEVICE, placement_rtt_hash_table, DATA_OPTIMAL_RTT, is_last_epoch=True)
test_metrics = evaluate(model, test_loader_eval, DEVICE, placement_rtt_hash_table, DATA_OPTIMAL_RTT, is_last_epoch=True)

# ========================================================================
# WANDB
# ========================================================================
wandb.log({
    "data/num_datasets_total": int(len(graphs)),
    "data/num_train": int(len(train_graphs)),
    "data/num_val": int(len(val_graphs)),
    "data/num_test": int(len(test_graphs)),
})

final_metrics_log = {
    "final/train/loss_ce": safe_float(train_metrics['ce']),
    "final/train/acc": safe_float(train_metrics['acc']),
    "final/train/regret": safe_float(train_metrics['regret']),
    "final/train/regret_pct": safe_float(train_metrics['regret_pct']),
    "final/val/loss_ce": safe_float(val_metrics_final['ce']),
    "final/val/acc": safe_float(val_metrics_final['acc']),
    "final/val/regret": safe_float(val_metrics_final['regret']),
    "final/val/regret_pct": safe_float(val_metrics_final['regret_pct']),
    "final/test/loss_ce": safe_float(test_metrics['ce']),
    "final/test/acc": safe_float(test_metrics['acc']),
    "final/test/regret": safe_float(test_metrics['regret']),
    "final/test/regret_pct": safe_float(test_metrics['regret_pct']),
}

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
