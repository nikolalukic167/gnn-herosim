"""Post-hoc check of guarded learned proposals on the opened live-gate inputs."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from src.placement.radical.dag_reservation import problem
from src.placement.radical.dag_reservation_bridge import BridgedDag, baseline
from src.policy.dag_reservation.model import features, decode
from scripts_cosim.evaluate_dag_reservation_learning import load_models, CORPUS
from scripts_cosim.mixed_joint_dispatch_screen import read, save, sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gate', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(); torch.set_num_threads(1)
    meta = read(CORPUS / 'METADATA.json'); models, receipts = load_models(meta)
    gate = read(args.gate / 'read.json')
    if gate['checkpoints'] != receipts or gate['inputs'] != 16:
        raise ValueError('not the frozen complete gate')
    engine = BridgedDag(args.gate / 'build'); rows = []
    for seed in range(162000, 162016):
        b = problem(seed, *meta['shape'])
        base, a, rank, _, _ = baseline(engine, b)
        x, graph, eligible = features(b, a, rank)
        observed = read(args.gate / 'fresh' / str(seed) / 'read.json')
        values = {}
        for name, model in models.items():
            predicted_a, predicted_rank = decode(model, x, graph, eligible, *meta['shape'][:2])
            mode_cost = [engine.plan(b, predicted_a, predicted_rank, mode)[0] for mode in (0, 1)]
            source = observed['models'][name]['source']
            if (min(mode_cost) < base) != (source == 'model'):
                raise ValueError('recorded guard decision mismatch')
            values[name] = {'nondelay_cost': mode_cost[0], 'reservation_cost': mode_cost[1],
                            'best_cost': min(mode_cost), 'accepted': source == 'model'}
        rows.append({'seed': seed, 'hand_initializer_cost': base, 'models': values,
                     'gate_read_sha256': sha(args.gate / 'fresh' / str(seed) / 'read.json')})
    result = {'scope': 'post-hoc diagnostic on already-opened gate; no model selection',
              'gate_report_sha256': sha(args.gate / 'read.json'),
              'source_sha256': sha(Path(__file__)), 'checkpoint_receipts': receipts,
              'accepted_count': {name: sum(r['models'][name]['accepted'] for r in rows) for name in models},
              'median_raw_regret_pct': {name: float(np.median([(r['models'][name]['best_cost'] / r['hand_initializer_cost'] - 1) * 100 for r in rows])) for name in models},
              'rows': rows}
    save(args.out, result); print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))


if __name__ == '__main__': main()
