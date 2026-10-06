"""Deterministic, matched value training for DAG memory-retention moves."""
import argparse
import hashlib
import json
import os
import random
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn

from src.placement.radical.dag_memory import problem
from src.placement.radical.dag_reservation_bridge import BridgedDag
from src.policy.dag_memory.model import CONTRACT, MemoryValueNet, serve
from scripts_cosim.dag_memory_lifetime_s0 import make_case


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_data(root, split):
    meta = json.loads((root / 'METADATA.json').read_text())
    if meta['contract'] != CONTRACT or sha(root / f'{split}.npz') != meta['files'][f'{split}.npz']:
        raise ValueError('memory data contract or fingerprint mismatch')
    with np.load(root / f'{split}.npz') as source:
        data = {name: source[name].copy() for name in source.files}
    if not all(np.isfinite(value).all() for value in data.values()):
        raise ValueError('nonfinite memory training data')
    return data, meta


def validation_cost(model, meta, engine, budget_ms):
    values = []
    for item in meta['cases']:
        if item['split'] != 'validation':
            continue
        b, a, rank, simple, initial_rule, prep_ms = make_case(engine, item['seed'], meta['shape'])
        result = serve(model, b, a, rank, simple[initial_rule], prep_ms, budget_ms)
        values.append(result['cost'])
    return float(np.mean(values))


def train_epoch(model, optimizer, tensors, generator, batch_size):
    model.train()
    order = torch.randperm(len(tensors['x']), generator=generator)
    total = 0.
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        optimizer.zero_grad(set_to_none=True)
        prediction = model(tensors['x'][idx], tensors['graph'][idx], tensors['mask'][idx])
        valid = tensors['mask'][idx] > 0
        objective = nn.functional.smooth_l1_loss(prediction[valid], tensors['target'][idx][valid])
        objective.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        total += float(objective.detach()) * len(idx)
    return total / len(order)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--arm', choices=['gnn', 'mpoff', 'hand_mlp'], required=True)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--budget-ms', type=float, default=250.)
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--wandb-project', default='herosim-dag-memory-value')
    parser.add_argument('--wandb-mode', choices=['offline', 'online'], default='offline')
    args = parser.parse_args()
    seed_everything(args.seed)
    torch.set_num_threads(1)
    train, meta = load_data(args.cache_dir, 'train')
    validation, other = load_data(args.cache_dir, 'validation')
    if meta != other or train['x'].shape[1:] != validation['x'].shape[1:]:
        raise ValueError('memory split incompatibility')
    architecture = {'feature_dim': train['x'].shape[-1], 'hidden': args.hidden,
                    'layers': args.layers, 'arm': args.arm}
    model = MemoryValueNet(**architecture)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    tensors = {key: torch.as_tensor(value) for key, value in train.items()}
    generator = torch.Generator().manual_seed(args.seed)
    import wandb
    run = wandb.init(project=args.wandb_project,
                     name=os.environ.get('WANDB_RUN_NAME', f'{args.arm}-seed{args.seed}'),
                     tags=os.environ.get('WANDB_TAGS', '').split(','),
                     config={**{key: str(value) if isinstance(value, Path) else value
                                for key, value in vars(args).items()},
                             'contract': CONTRACT, 'architecture': architecture}, mode=args.wandb_mode)
    if run is None:
        raise RuntimeError('W&B run did not initialize')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / f'{args.arm}_seed{args.seed}.pt'
    if checkpoint.exists():
        raise ValueError('checkpoint exists; preserve earlier run')
    engine = BridgedDag(args.output_dir / 'build')
    best = float('inf')
    best_epoch = -1
    history = []
    try:
        for epoch in range(args.epochs + 1):
            loss = None
            if epoch:
                loss = train_epoch(model, optimizer, tensors, generator, args.batch_size)
            model.eval()
            value = validation_cost(model, meta, engine, args.budget_ms)
            row = {'epoch': epoch, 'validation_mean_job_completion_ms': value}
            if loss is not None:
                row['train_loss'] = loss
            history.append(row)
            wandb.log(row, step=epoch)
            if value < best:
                best, best_epoch = value, epoch
                torch.save({key: tensor.detach().cpu() for key, tensor in model.state_dict().items()}, checkpoint)
                repo = Path(__file__).resolve().parents[3]
                sources = [Path(__file__), Path(__file__).with_name('model.py'),
                           repo / 'src/placement/radical/dag_memory.py']
                sidecar = {'contract': CONTRACT, 'arm': args.arm, 'seed': args.seed,
                           'architecture': architecture, 'selected_epoch': epoch,
                           'selection': 'minimum_complete_served_validation_cost',
                           'validation_mean_job_completion_ms': value,
                           'data_metadata_sha256': sha(args.cache_dir / 'METADATA.json'),
                           'wandb_mode': args.wandb_mode, 'wandb_run_dir': run.dir,
                           'sources': {str(path.relative_to(repo)): sha(path) for path in sources}}
                checkpoint.with_suffix('.contract.json').write_text(json.dumps(sidecar, indent=2) + '\n')
            print('TRAIN', args.arm, args.seed, row, 'best', best_epoch, flush=True)
        checkpoint.with_suffix('.training.json').write_text(json.dumps(
            {'history': history, 'best_epoch': best_epoch, 'best_validation_cost': best,
             'wandb_run_dir': run.dir}, indent=2) + '\n')
        wandb.summary.update({'best_epoch': best_epoch, 'best_validation_cost': best})
    finally:
        wandb.finish()


if __name__ == '__main__':
    main()
