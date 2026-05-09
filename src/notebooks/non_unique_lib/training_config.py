from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    project_root: Path
    cache_dir: Path
    use_merged_cache: bool
    embedding_dim: int
    hidden_dim: int
    learning_rate: float
    batch_size: int
    num_gin_layers: int
    weight_decay: float
    epochs: int
    rtt_scale_factor: float
    regret_loss_weight: float
    ce_loss_weight: float
    wandb_project: str
    wandb_entity: str
    wandb_api_key: str | None
    num_dataloader_workers: int


def parse_training_config() -> TrainingConfig:
    default_project_root = Path(__file__).resolve().parents[3]
    default_cache_dir = (
        default_project_root
        / "simulation_data"
        / "artifacts"
        / "run_queue_big"
        / "graphs_cache_gnn_datasets_4tasks_overnight_260422"
    )

    parser = argparse.ArgumentParser(description="Train non-unique task placement GNN.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=default_project_root,
        help="Repository root path.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=default_cache_dir,
        help="Path to prepared cache directory.",
    )
    parser.add_argument("--use-merged-cache", action="store_true", help="Flag metadata only (for logging).")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-gin-layers", type=int, default=3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--rtt-scale-factor", type=float, default=1.0)
    parser.add_argument("--regret-loss-weight", type=float, default=0.3)
    parser.add_argument("--ce-loss-weight", type=float, default=1.0)
    parser.add_argument("--wandb-project", type=str, default="2-3-4-tasks-non-unique")
    parser.add_argument("--wandb-entity", type=str, default="nikolalukic167-tu-wien")
    parser.add_argument(
        "--wandb-api-key",
        type=str,
        default=None,
        help="Optional WandB API key; if omitted, existing environment auth is used.",
    )
    parser.add_argument(
        "--num-dataloader-workers",
        type=int,
        default=0,
        help=(
            "DataLoader worker processes. Default 0 (load in main process): each worker "
            "holds full LMDB-unpickled valid_combos in RAM — many workers OOMs on large RTT lists. "
            "Try 2–4 only on high-memory hosts."
        ),
    )
    args = parser.parse_args()

    return TrainingConfig(
        project_root=args.project_root,
        cache_dir=args.cache_dir,
        use_merged_cache=args.use_merged_cache,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_gin_layers=args.num_gin_layers,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        rtt_scale_factor=args.rtt_scale_factor,
        regret_loss_weight=args.regret_loss_weight,
        ce_loss_weight=args.ce_loss_weight,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_api_key=args.wandb_api_key,
        num_dataloader_workers=max(0, args.num_dataloader_workers),
    )
