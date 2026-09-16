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
    dataloader_prefetch_factor: int
    persistent_dataloader_workers: bool
    torch_threads: int
    precompute_rtt_lookups: bool
    hard_negative_fraction: float
    # Early stopping on the checkpoint-selection metric. patience=0 is OFF (every
    # registration before 2026-09-16 ran the full --epochs; a default that stopped early
    # would silently change what "last epoch" means in offline_live_transfer_v1 R4).
    patience: int = 0
    min_epochs: int = 0


def should_stop_early(epoch: int, last_improvement_epoch: int, patience: int, min_epochs: int) -> bool:
    """True when `patience` epochs have passed since the selection metric last improved.

    `epoch` is 0-based and has just finished. A run never stops before `min_epochs`
    epochs have run, and never when patience is 0. Pure so it can be tested without a
    trainer: measured on the 2026-09-15 cold-corpus arms the selected epoch is 19-67 and
    on the warm corpus 34-283, so a fixed cap is wrong on one of them by construction.
    """
    if patience <= 0:
        return False
    if epoch + 1 < min_epochs:
        return False
    return (epoch - last_improvement_epoch) >= patience


def parse_training_config() -> TrainingConfig:
    default_project_root = Path(__file__).resolve().parents[3]
    default_cache_dir = (
        default_project_root
        / "simulation_data"
        / "artifacts"
        / "run_queue_big"
        / "graphs_cache_gnn_datasets_4tasks_1060_scheduler_adaptive"
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
    parser.add_argument(
        "--patience", type=int, default=0,
        help="Stop when the checkpoint-selection metric has not improved for this many "
             "epochs. 0 (default) runs every --epochs, as every pre-2026-09-16 run did.",
    )
    parser.add_argument(
        "--min-epochs", type=int, default=0,
        help="With --patience: never stop before this many epochs have run.",
    )
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
        default=4,
        help=(
            "DataLoader worker processes. Default 4; safe with preloaded hash-table "
            "lookups because __getitem__ only attaches lightweight metadata."
        ),
    )
    parser.add_argument(
        "--dataloader-prefetch-factor",
        type=int,
        default=1,
        help="Prefetch factor when DataLoader workers are enabled.",
    )
    parser.add_argument(
        "--persistent-dataloader-workers",
        action="store_true",
        help="Keep DataLoader workers alive across epochs when --num-dataloader-workers > 0.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=0,
        help="torch.set_num_threads value. 0 means use SLURM_CPUS_PER_TASK when present.",
    )
    parser.add_argument(
        "--no-precompute-rtt-lookups",
        action="store_true",
        help="Disable extra RAM-heavy lookup caches for faster regret training/evaluation.",
    )
    parser.add_argument(
        "--hard-negative-fraction",
        type=float,
        default=0.5,
        help="Fraction of highest-RTT non-optimal combos kept for random hard-negative sampling.",
    )
    args = parser.parse_args()
    hard_negative_fraction = min(1.0, max(0.0, args.hard_negative_fraction))

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
        dataloader_prefetch_factor=max(1, args.dataloader_prefetch_factor),
        persistent_dataloader_workers=bool(args.persistent_dataloader_workers),
        torch_threads=max(0, args.torch_threads),
        precompute_rtt_lookups=not args.no_precompute_rtt_lookups,
        hard_negative_fraction=hard_negative_fraction,
        patience=max(0, int(args.patience)),
        min_epochs=max(0, int(args.min_epochs)),
    )
