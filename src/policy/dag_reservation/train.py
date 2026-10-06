"""Deterministic matched training for complete DAG reservation proposals."""
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
from src.policy.dag_reservation.model import CONTRACT, ReservationProposalNet, decode
from src.placement.radical.dag_reservation import problem
from src.placement.radical.dag_reservation_bridge import BridgedDag


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_split(root, split):
    meta = json.loads((root / 'METADATA.json').read_text())
    if meta['contract'] != CONTRACT or sha(root / f'{split}.npz') != meta['files'][f'{split}.npz']:
        raise ValueError('dataset contract or fingerprint mismatch')
    with np.load(root / f'{split}.npz') as data:
        result = {k: data[k].copy() for k in data.files}
    if not np.isfinite(result['x']).all() or not np.isfinite(result['graph']).all():
        raise ValueError('invalid features')
    return result, meta


def train_epoch(model, optimizer, tensors, batch_size, generator):
    model.train(); x, graph, eligible, assignment, rank = tensors
    order = torch.randperm(len(x), generator=generator); total = 0.
    for start in range(0, len(x), batch_size):
        idx = order[start:start + batch_size]
        optimizer.zero_grad(set_to_none=True)
        logits, score = model(x[idx], graph[idx], eligible[idx])
        ce = nn.functional.cross_entropy(logits.flatten(0, 1), assignment[idx].flatten())
        normalized = torch.sigmoid(score)
        loss = ce + .5 * nn.functional.smooth_l1_loss(normalized, rank[idx])
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step(); total += float(loss.detach()) * len(idx)
    return total / len(x)


def validation_cost(model, data, shape, engine, device):
    values = []
    for i, seed in enumerate(data['seeds']):
        b = problem(int(seed), *shape)
        a, rank = decode(model, data['x'][i], data['graph'][i], data['eligible'][i], shape[0], shape[1], device)
        values.append(min(engine.plan(b, a, rank, mode)[0] for mode in (0, 1)))
    return float(np.mean(values))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache-dir', type=Path, required=True)
    ap.add_argument('--arm', choices=['gnn', 'mpoff'], required=True)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--learning-rate', type=float, default=.001)
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    ap.add_argument('--seed', type=int, default=101)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--wandb-project', default='herosim-dag-reservation')
    ap.add_argument('--wandb-mode', choices=['offline', 'online'], default='offline')
    args = ap.parse_args(); seed_everything(args.seed); torch.set_num_threads(1)
    if args.device == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('requested GPU unavailable')
    train, meta = load_split(args.cache_dir, 'train')
    val, other = load_split(args.cache_dir, 'validation')
    if meta != other or set(train['seeds']) & set(val['seeds']): raise ValueError('split metadata mismatch or overlap')
    device = torch.device(args.device)
    tensors = tuple(torch.as_tensor(train[k], device=device) for k in ('x', 'graph', 'eligible', 'assignment', 'rank_target'))
    architecture = {'feature_dim': train['x'].shape[-1], 'hosts': meta['shape'][2],
                    'hidden': args.hidden, 'layers': args.layers, 'mp': args.arm == 'gnn'}
    model = ReservationProposalNet(**architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(args.seed)
    import wandb
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    run = wandb.init(project=args.wandb_project,
                     name=os.environ.get('WANDB_RUN_NAME', f'{args.arm}-seed{args.seed}'),
                     tags=os.environ.get('WANDB_TAGS', '').split(','),
                     config={**config, 'contract': CONTRACT, 'architecture': architecture,
                             'chance_host_accuracy': .5}, mode=args.wandb_mode)
    if run is None: raise RuntimeError('W&B run was not initialized')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / f'{args.arm}_seed{args.seed}.pt'
    if checkpoint.exists(): raise ValueError('checkpoint exists; preserve previous run')
    engine = BridgedDag(args.output_dir / 'build')
    best = float('inf'); best_epoch = -1; history = []
    try:
        for epoch in range(args.epochs + 1):
            loss = None if epoch == 0 else train_epoch(model, optimizer, tensors, args.batch_size, generator)
            value = validation_cost(model, val, meta['shape'], engine, device)
            row = {'epoch': epoch, 'validation_mean_job_completion_ms': value}
            if loss is not None: row['train_loss'] = loss
            history.append(row); wandb.log(row, step=epoch)
            if value < best:
                best = value; best_epoch = epoch
                torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, checkpoint)
                sources = [Path(__file__), Path(__file__).with_name('model.py'),
                           Path(__file__).resolve().parents[3] / 'src/placement/radical/dag_reservation_bridge.py']
                sidecar = {'contract': CONTRACT, 'torch_seeded': True, 'deterministic_algorithms': True,
                           'arm': args.arm, 'seed': args.seed, 'architecture': architecture,
                           'selected_epoch': epoch, 'selection': 'minimum_mean_complete_validation_cost',
                           'validation_mean_job_completion_ms': value,
                           'data_metadata_sha256': sha(args.cache_dir / 'METADATA.json'),
                           'wandb_mode': args.wandb_mode, 'wandb_run_dir': run.dir,
                           'python_env': {'python': os.sys.version, 'torch': torch.__version__, 'numpy': np.__version__},
                           'sources': {str(p.relative_to(Path(__file__).resolve().parents[3])): sha(p) for p in sources}}
                checkpoint.with_suffix('.contract.json').write_text(json.dumps(sidecar, indent=2) + '\n')
            if epoch % 5 == 0: print('TRAIN', args.arm, args.seed, row, 'best', best_epoch, flush=True)
        report = {'arm': args.arm, 'seed': args.seed, 'selected_epoch': best_epoch,
                  'validation_mean_job_completion_ms': best, 'history': history,
                  'checkpoint': str(checkpoint), 'wandb_mode': args.wandb_mode, 'wandb_run_dir': run.dir}
        checkpoint.with_suffix('.training.json').write_text(json.dumps(report, indent=2) + '\n')
        wandb.summary.update({'best_epoch': best_epoch, 'best_validation_cost': best})
    finally: wandb.finish()


if __name__ == '__main__': main()
