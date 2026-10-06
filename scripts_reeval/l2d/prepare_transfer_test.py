"""Generate a frozen, fresh L2D test split using the upstream generator."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts_reeval.l2d.l2d_bridge import load_l2d, upstream_commit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, choices=(10, 15), required=True)
    parser.add_argument('--seed', type=int, default=301)
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    L = load_l2d(args.size, args.size, 'cpu')
    np.random.seed(args.seed)
    data = np.array([L.uni_instance_gen(n_j=args.size, n_m=args.size,
                                        low=L.configs.low, high=L.configs.high)
                     for _ in range(args.count)])
    if data.shape != (args.count, 2, args.size, args.size):
        raise RuntimeError(f'unexpected instance shape: {data.shape}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, data)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps({'path': str(args.output), 'sha256': digest,
                      'shape': list(data.shape), 'seed': args.seed,
                      'upstream_commit': upstream_commit()}))


if __name__ == '__main__':
    main()
