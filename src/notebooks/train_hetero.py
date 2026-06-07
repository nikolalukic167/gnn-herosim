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
from typing import Any, Dict, Iterable, List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
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
from non_unique_lib.training_config import parse_training_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.policy.gnn_hetero.data import FORWARD_EDGE_TYPE
from src.policy.gnn_hetero.gnn_model import TaskPlacementGNN as HeteroTaskPlacementGNN


PlacementCombo = Tuple[Tuple[int, int], ...]
RttByCombo = Dict[str, Dict[PlacementCombo, float]]


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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


_DEFAULT_NEAR_RTT_WANDB_PROJECT = "gnn-hetero-near-rtt-jun2026"

RUNTIME_CONFIG = parse_training_config()
if "--cache-dir" not in sys.argv:
    RUNTIME_CONFIG = replace(
        RUNTIME_CONFIG,
        cache_dir=(
            RUNTIME_CONFIG.project_root
            / "simulation_data"
            / "artifacts"
            / "run_queue_big"
            / "graphs_cache_gnn_datasets_4tasks_1060_scheduler_adaptive_hetero"
        ),
    )
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
NUM_DATALOADER_WORKERS = RUNTIME_CONFIG.num_dataloader_workers

print(f"Cache directory: {CACHE_CTX.cache_dir}")
print(f"Device: {DEVICE}")
print(f"Near RTT config: {NEAR_CFG}")


class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout_p: float = 0.1) -> None:
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
    def __init__(self, embedding_dim: int, hidden_dim: int, edge_dim: int = 0) -> None:
        super().__init__()
        in_dim = 2 * embedding_dim + (edge_dim if edge_dim else 0)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(p=NEAR_CFG.dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, e_task: Tensor, e_platform: Tensor, e_attr: Optional[Tensor] = None) -> Tensor:
        x = torch.cat([e_task, e_platform] + ([e_attr] if e_attr is not None else []), dim=-1)
        return self.fc2(self.dropout(F.relu(self.fc1(x)))).squeeze(-1)


class TaskPlacementGNN(nn.Module):
    def __init__(
        self,
        task_feature_dim: int,
        platform_feature_dim: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.task_encoder = MLPEncoder(task_feature_dim, hidden_dim, embedding_dim, NEAR_CFG.dropout)
        self.platform_encoder = MLPEncoder(platform_feature_dim, hidden_dim, embedding_dim, NEAR_CFG.dropout)
        self.gin = GIN(
            in_channels=embedding_dim,
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            out_channels=embedding_dim,
        )
        self.post_gin_dropout = nn.Dropout(p=NEAR_CFG.dropout)
        self.edge_scorer = EdgeScorer(embedding_dim, hidden_dim, edge_dim=5)

    def forward(self, data: Data) -> List[Tensor]:
        n_tasks = int(data.n_tasks)
        n_platforms = int(data.n_platforms)

        task_embeddings = self.task_encoder(data.task_features)
        platform_embeddings = self.platform_encoder(data.platform_features)
        x = torch.cat([task_embeddings, platform_embeddings], dim=0)
        x = self.post_gin_dropout(self.gin(x, data.edge_index))

        task_emb = x[:n_tasks]
        platform_emb = x[n_tasks:]
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

        e_attr: Optional[Tensor] = None
        if hasattr(data, "edge_attr") and data.edge_attr.numel() > 0:
            e_attr = data.edge_attr[valid]

        scores = self.edge_scorer(task_emb[ti], platform_emb[pj], e_attr)
        return [scores[ti == task_idx] for task_idx in range(n_tasks)]


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
        dataset_id = getattr(data, "dataset_id", None)
        rows = self.exact_rtt_map.get(dataset_id or "", [])
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
        graph.dataset_id = dataset_id
        graph.opt_rtt = float(self.optimal_rtt_map.get(dataset_id, 0.0))
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
        "opt_rtt": getattr(data, "opt_rtt", None),
        "task_logit_to_placement": getattr(
            data,
            "task_logit_to_placement",
            getattr(data, "_task_logit_to_placement", {}),
        ),
    }
    data = data.to(device)
    for key, value in saved.items():
        setattr(data, key, value)
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
    running_total = 0.0
    steps = 0
    valid_rank = 0
    active_pairs = 0
    total_pairs = 0

    for batch in tqdm(loader, desc=f"Epoch {epoch:3d} [Train]", leave=False):
        optimizer.zero_grad()
        loss_ce_total = torch.zeros((), device=DEVICE)
        loss_rank_total = torch.zeros((), device=DEVICE)
        n_ce = 0
        n_rank = 0

        for graph in batch:
            data = move_graph_to_device(graph, DEVICE)
            logits = model(data)

            loss_ce, valid_ce = loss_original_ce(logits, data, DEVICE)
            if valid_ce > 0 and torch.isfinite(loss_ce):
                loss_ce_total = loss_ce_total + loss_ce
                n_ce += 1

            loss_rank, valid, stats = criterion(logits, data, DEVICE)
            if valid > 0 and torch.isfinite(loss_rank):
                loss_rank_total = loss_rank_total + loss_rank
                n_rank += 1
                valid_rank += 1
                active_pairs += int(stats.get("active_pairs", 0))
                total_pairs += int(stats.get("pairs", 0))

        if n_ce == 0 and n_rank == 0:
            continue

        ce_avg = loss_ce_total / max(1, n_ce)
        rank_avg = loss_rank_total / max(1, n_rank)
        loss = CE_LOSS_WEIGHT * ce_avg + REGRET_LOSS_WEIGHT * rank_avg
        if not torch.isfinite(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_ce += float(ce_avg.item())
        running_rank += float(rank_avg.item())
        running_total += float(loss.item())
        steps += 1

    return {
        "ce": running_ce / max(1, steps),
        "rank": running_rank / max(1, steps),
        "total": running_total / max(1, steps),
        "valid_rank": float(valid_rank),
        "active_pair_frac": active_pairs / max(1, total_pairs),
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
    regret_topk: List[float] = []
    regret_oracle_topk: List[float] = []
    greedy_mapped = 0
    greedy_total = 0
    topk_mapped = 0
    topk_total = 0

    for batch in tqdm(loader, desc=f"Evaluating {split_name}", leave=False):
        for graph in batch:
            data = move_graph_to_device(graph, DEVICE)
            logits = model(data)
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

            dataset_id = getattr(data, "dataset_id", None) or ""
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
        "regret_topk": avg(regret_topk),
        "regret_oracle_topk": avg(regret_oracle_topk),
        "count_regret_greedy": float(len(regret_greedy)),
        "count_regret_topk": float(len(regret_topk)),
        "greedy_sidecar_coverage": greedy_mapped / max(1, greedy_total),
        "topk_sidecar_coverage": topk_mapped / max(1, topk_total),
        "greedy_unmapped": float(greedy_total - greedy_mapped),
    }
    print(
        f"[{split_name}] acc={metrics['acc']*100:.1f}% "
        f"task_acc={metrics['task_acc']*100:.1f}% "
        f"greedy_regret={metrics['regret_greedy']:.4f}s "
        f"(sidecar_hit={metrics['greedy_sidecar_coverage']*100:.1f}%, "
        f"unmapped={int(metrics['greedy_unmapped'])}) "
        f"top{NEAR_CFG.top_k_decode}_regret={metrics['regret_topk']:.4f}s "
        f"oracle_top{NEAR_CFG.top_k_decode}={metrics['regret_oracle_topk']:.4f}s"
    )
    return metrics


def prefix(metrics: Dict[str, float], name: str) -> Dict[str, float]:
    return {f"{name}/{k}": float(v) for k, v in metrics.items()}


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
print(f"Exact RTT datasets: {len(EXACT_RTT_MAP)}, combos: {sum(len(v) for v in EXACT_RTT_MAP.values()):,}")

ys = np.concatenate([g.y.numpy() for g in graphs])
print("Valid labels:", int(np.sum(ys >= 0)), "/", len(ys))
print("Avg edges:", float(np.mean([g[FORWARD_EDGE_TYPE].edge_index.size(1) for g in graphs])))

if NEAR_CFG.train_all or len(graphs) < 10:
    train_graphs, val_graphs, test_graphs = graphs, graphs, graphs
    train_ids, val_ids, test_ids = dataset_ids, dataset_ids, dataset_ids
else:
    train_graphs, temp_graphs, train_ids, temp_ids = train_test_split(
        graphs, dataset_ids, test_size=0.2, random_state=42
    )
    val_graphs, test_graphs, val_ids, test_ids = train_test_split(
        temp_graphs, temp_ids, test_size=0.5, random_state=42
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
        "ce_weight": float(CE_LOSS_WEIGHT),
        "regret_weight": float(REGRET_LOSS_WEIGHT),
        "rtt_scale_factor": float(RTT_SCALE_FACTOR),
        "loss_type": "CE + NearExactRttRanking",
        "model_type": "hetero",
        "loss_variant": str(NEAR_CFG.loss_variant),
        "sidecar_name": str(NEAR_CFG.sidecar_name),
        "near_rtt_training": True,
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
        "num_train": int(len(train_graphs)),
        "num_val": int(len(val_graphs)),
        "num_test": int(len(test_graphs)),
        "num_exact_rtt_combo_rows": int(sum(len(v) for v in EXACT_RTT_MAP.values())),
        "unmapped_penalty": float(NEAR_CFG.unmapped_penalty),
        "cache_dir": str(CACHE_CTX.cache_dir),
    },
    tags=[t for t in os.environ.get("WANDB_TAGS", "near-rtt").split(",") if t],
)

model = HeteroTaskPlacementGNN(
    task_feature_dim=3,
    platform_feature_dim=13,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_GIN_LAYERS,
).to(DEVICE)


def init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)


model.apply(init_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion = NearRttRankingLoss(EXACT_RTT_MAP, RTT_SCALE_FACTOR, NEAR_CFG)

model_path = Path("models") / f"{wandb.run.name}.pt"
best_val_regret = float("inf")
best_val_metrics: Dict[str, float] = {}
checkpoint_saved = False

print("=" * 80)
print("TRAINING (CE + Near Exact RTT Ranking)")
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
    wandb.log(log_dict, step=epoch)

    val_target = val_metrics["regret_topk"] if val_metrics["count_regret_topk"] > 0 else val_metrics["regret_greedy"]
    if val_target < best_val_regret:
        best_val_regret = val_target
        best_val_metrics = val_metrics
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        metadata_path = model_path.with_suffix(".metadata.json")
        with open(metadata_path, "w") as fh:
            json.dump(
                {
                    "model_type": "hetero",
                    "task_feature_dim": 3,
                    "platform_feature_dim": 13,
                    "embedding_dim": int(EMBEDDING_DIM),
                    "hidden_dim": int(HIDDEN_DIM),
                    "num_layers": int(NUM_GIN_LAYERS),
                    "cache_dir": str(CACHE_CTX.cache_dir),
                },
                fh,
                indent=2,
            )
        checkpoint_saved = True
        print(
            f"  *** New best top-k regret: {best_val_regret:.4f}s "
            f"(greedy={val_metrics['regret_greedy']:.4f}s)"
        )

    if epoch % 5 == 0 or epoch == EPOCHS - 1:
        print(
            f"Epoch {epoch:3d}/{EPOCHS} "
            f"Train CE={train_metrics['ce']:.4f} "
            f"Rank={train_metrics['rank']:.4f} "
            f"Val greedy={val_metrics['regret_greedy']:.4f}s "
            f"Val top{NEAR_CFG.top_k_decode}={val_metrics['regret_topk']:.4f}s "
            f"active_pair_frac={train_metrics['active_pair_frac']:.2f} "
            f"({time.perf_counter() - start:.1f}s)"
        )

if not checkpoint_saved:
    raise RuntimeError("No near-RTT checkpoint was saved.")

model.load_state_dict(torch.load(model_path, map_location=DEVICE))
train_final = evaluate(model, train_loader, RTT_BY_DATASET, WORST_REGRET_BY_DATASET, "final/train")
val_final = evaluate(model, val_loader, RTT_BY_DATASET, WORST_REGRET_BY_DATASET, "final/val")
test_final = evaluate(model, test_loader, RTT_BY_DATASET, WORST_REGRET_BY_DATASET, "final/test")

final_log: Dict[str, float] = {}
final_log.update(prefix(train_final, "final/train"))
final_log.update(prefix(val_final, "final/val"))
final_log.update(prefix(test_final, "final/test"))
wandb.log(final_log)

wandb.summary["best_val_regret_topk"] = float(best_val_regret)
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
print(f"Best val top-k regret: {best_val_regret:.4f}s")
print(
    f"Final test: greedy={test_final['regret_greedy']:.4f}s, "
    f"top{NEAR_CFG.top_k_decode}={test_final['regret_topk']:.4f}s, "
    f"oracle_top{NEAR_CFG.top_k_decode}={test_final['regret_oracle_topk']:.4f}s"
)
