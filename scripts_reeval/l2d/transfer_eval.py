"""Evaluate converged L2D checkpoints on a different job-shop size."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts_reeval.l2d.l2d_bridge import load_l2d, upstream_commit
from scripts_reeval.l2d.arm_features import INPUT_DIM
from scripts_reeval.l2d.train_arm import greedy_eval, make_policy
from scripts_reeval.l2d.pdr import run_pdr


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'simulation_data/literature_reeval_v1/l2d'
RUNS = DATA / 'runs'
ARMS = ('gnn', 'mpoff', 'mlp_t1')


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint(runs_root: Path, source: int, arm: str, seed: int) -> Path:
    return runs_root / f'l2d_{source}x{source}_{arm}_lr0.0005_up30000_s{seed}' / 'best_val.pth'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=int, default=6)
    parser.add_argument('--target', type=int, required=True)
    parser.add_argument('--instances', type=int, default=100)
    parser.add_argument('--seeds', type=int, default=8)
    parser.add_argument('--test-path', type=Path, required=True)
    parser.add_argument('--runs-root', type=Path, default=RUNS)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.source == args.target or args.source != 6 or args.target not in (10, 15):
        raise ValueError('this pilot requires 6x6 training and a 10x10 or 15x15 target')
    if not (1 <= args.instances <= 100 and 1 <= args.seeds <= 8):
        raise ValueError('instances must be 1..100 and seeds 1..8')
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    device = torch.device('cpu')
    L = load_l2d(args.target, args.target, 'cpu')
    data_path = args.test_path.resolve()
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    frozen = np.load(data_path)
    if frozen.ndim != 4 or frozen.shape[1:] != (2, args.target, args.target) or frozen.shape[0] < args.instances:
        raise RuntimeError(f'wrong target data shape: {frozen.shape}')
    instances = [(frozen[i][0], frozen[i][1]) for i in range(args.instances)]
    if len(instances) != args.instances:
        raise RuntimeError('incomplete target split')
    output = {
        'contract': 'l2d_size_transfer_pilot_v1',
        'source': args.source, 'target': args.target,
        'instances': args.instances, 'seeds': args.seeds,
        'upstream_commit': upstream_commit(),
        'test_path': str(data_path), 'test_sha256': sha256(data_path),
        'checkpoint_protocol': 'best_val, source-size validation only',
        'arms': {},
    }
    for arm in ARMS:
        output['arms'][arm] = []
        for seed in range(args.seeds):
            path = checkpoint(args.runs_root, args.source, arm, seed)
            sidecar = Path(str(path) + '.contract.json')
            if not path.is_file() or not sidecar.is_file():
                raise FileNotFoundError(f'checkpoint or sidecar missing: {path}')
            meta = json.loads(sidecar.read_text())
            expected = {'arm': arm, 'n_j': args.source, 'n_m': args.source,
                        'study_seed': seed, 'input_dim': INPUT_DIM[arm],
                        'max_updates': 30000, 'lr': 0.0005,
                        'upstream_commit': upstream_commit(), 'checkpoint': 'best_val'}
            for key, value in expected.items():
                if meta.get(key) != value:
                    raise RuntimeError(f'{path}: {key} contract mismatch')
            policy = make_policy(L, arm, args.target, args.target, device)
            policy.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
            policy.eval()
            makespans = greedy_eval(L, arm, policy, instances, args.target, args.target, device)
            if len(makespans) != args.instances or not np.isfinite(makespans).all():
                raise RuntimeError(f'{arm} seed {seed}: invalid evaluation')
            output['arms'][arm].append({
                'seed': seed, 'checkpoint_path': str(path),
                'checkpoint_sha256': sha256(path), 'sidecar_sha256': sha256(sidecar),
                'selected_update': meta['selected_update'],
                'mean_makespan': float(makespans.mean()),
                'makespans': [float(x) for x in makespans],
            })
            print(f'{arm} seed={seed} target={args.target} mean={makespans.mean():.3f}', flush=True)
    pdr_env = L.SJSSP(n_j=args.target, n_m=args.target)
    pdr = np.array([run_pdr('FDD_MWKR', pdr_env, data) for data in instances])
    if not np.isfinite(pdr).all():
        raise RuntimeError('invalid PDR evaluation')
    output['fdd_mwkr'] = {'mean_makespan': float(pdr.mean()),
                          'makespans': [float(x) for x in pdr]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + '\n')
    print(f'FDD/MWKR target={args.target} mean={pdr.mean():.3f}', flush=True)


if __name__ == '__main__':
    main()
