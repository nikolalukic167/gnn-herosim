"""Generate disjoint exact flip-value labels for the DAG memory pilot."""
import argparse
import json
from pathlib import Path

import numpy as np

from src.placement.radical.dag_memory import problem, replay
from src.placement.radical.dag_reservation_bridge import BridgedDag
from src.policy.dag_memory.model import CONTRACT, features
from scripts_cosim.dag_memory_lifetime_s0 import identity, make_case
from scripts_cosim.mixed_joint_dispatch_screen import ROOT, read, save, sha

PROTOCOL = ROOT / 'experiments/dag_memory_value_learning_v1_protocol.json'


def labelled_state(b, assignment, rank, keep):
    x, graph, mask = features(b, assignment, rank, keep)
    cost = replay(b, assignment, rank, keep)['objective']
    values = np.zeros(keep.size, dtype=np.float32)
    for node in np.flatnonzero(mask):
        trial = keep.copy()
        trial.flat[node] ^= 1
        values[node] = 100 * (cost - replay(b, assignment, rank, trial)['objective']) / cost
    return {'x': x, 'graph': graph, 'mask': mask, 'target': values}, cost


def run(out):
    if out.exists():
        raise ValueError('preserve existing memory corpus')
    out.mkdir(parents=True)
    protocol = read(PROTOCOL)
    sources = [PROTOCOL, Path(__file__).resolve(), ROOT / 'src/policy/dag_memory/model.py',
               ROOT / 'src/placement/radical/dag_memory.py']
    save(out / 'protocol_before_run.json', {'protocol': protocol,
         'sources': {str(p.relative_to(ROOT)): sha(p) for p in sources}})
    engine = BridgedDag(out / 'build')
    save(out / 'native.json', engine.provenance)
    seen = set()
    cases = []
    for split, key in (('train', 'train_seeds'), ('validation', 'validation_seeds'),
                       ('test', 'sealed_test_seeds')):
        samples = []
        low, high = protocol[key]
        for seed in range(low, high + 1):
            b = problem(seed, *protocol['shape'])
            digest = identity(b)
            if digest in seen:
                raise RuntimeError('duplicate memory environment')
            seen.add(digest)
            row = {'seed': seed, 'split': split, 'identity': digest}
            if split != 'test':
                b, a, rank, simple, initial_rule, _ = make_case(engine, seed, protocol['shape'])
                keep = np.asarray(simple[initial_rule]['retain'], dtype=np.int8)
                first, first_cost = labelled_state(b, a, rank, keep)
                samples.append(first)
                best_cost, next_keep = first_cost, keep
                for node in np.flatnonzero(first['mask']):
                    trial = keep.copy()
                    trial.flat[node] ^= 1
                    value = replay(b, a, rank, trial)['objective']
                    if value < best_cost:
                        best_cost, next_keep = value, trial
                if best_cost < first_cost:
                    second, _ = labelled_state(b, a, rank, next_keep)
                    samples.append(second)
                row.update({'initial_cost': first_cost, 'best_single_flip_cost': best_cost,
                            'labelled_states': 2 if best_cost < first_cost else 1})
            cases.append(row)
            print('DATA', split, seed, row.get('labelled_states', 0), flush=True)
        if samples:
            np.savez_compressed(out / f'{split}.npz', **{
                key: np.stack([sample[key] for sample in samples]) for key in samples[0]})
    for name, digest in read(out / 'protocol_before_run.json')['sources'].items():
        assert sha(ROOT / name) == digest, name
    save(out / 'METADATA.json', {'contract': CONTRACT, 'shape': protocol['shape'],
         'cases': cases, 'files': {path.name: sha(path) for path in out.glob('*.npz')},
         'test_unopened': True, 'source_fingerprints': read(out / 'protocol_before_run.json')['sources']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.out)
