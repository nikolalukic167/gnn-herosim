"""Generate disjoint searched-schedule labels for the DAG reservation pilot."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from src.placement.radical.dag_reservation import problem
from src.placement.radical.dag_reservation_bridge import BridgedDag, baseline
from src.placement.radical.environment import serialized, pack
from src.policy.dag_reservation.model import features, CONTRACT
from scripts_cosim.dag_reservation_screen import plan, verify_plan
from scripts_cosim.mixed_joint_dispatch_screen import ROOT, read, save, sha

PROTOCOL = ROOT / 'experiments/dag_reservation_learning_v1_protocol.json'


def make_case(engine, seed, shape, split, root):
    b = problem(seed, *shape); cell = root / split / str(seed)
    (cell / 'placements').mkdir(parents=True)
    save(cell / 'input.json', serialized(b))
    cost, a, rank, decoder, rule = baseline(engine, b)
    original = {**plan(engine, b, a, rank, decoder, cost), 'rule': rule}
    best = original; alternatives = []
    if split != 'test':
        for mode in (0, 2):
            value, aa, rr, dd, count = engine.search(b, a, rank, 8192, mode, .001, 29)
            candidate = {**plan(engine, b, aa, rr, dd, value), 'mode': mode, 'steps': count}
            verify_plan(engine, b, candidate); alternatives.append(candidate)
            if candidate['cost'] < best['cost']: best = candidate
        with (cell / 'placements/placements.jsonl').open('w') as trace:
            for name, p in [('baseline', original), *[(f'reference_{p["mode"]}', p) for p in alternatives]]:
                trace.write(json.dumps({'arm': name, **p}) + '\n')
        save(cell / 'labels.json', {'baseline': original, 'candidates': alternatives, 'best': best})
    identity = hashlib.sha256(pack(b).tobytes() + b['predecessors'].tobytes()).hexdigest()
    result = {'seed': seed, 'split': split, 'identity': identity, 'input_sha256': sha(cell / 'input.json')}
    if split == 'test': return result, None
    x, graph, eligible = features(b, a, rank)
    target = np.asarray(best['assignment']).ravel()
    starts = np.asarray(best['starts']).ravel()
    priority = np.asarray(best['priority']).ravel()
    order = np.lexsort((priority, starts))
    rank_target = np.empty(len(order), dtype=np.float32)
    rank_target[order] = np.arange(len(order)) / max(len(order) - 1, 1)
    result.update({'teacher_cost': best['cost'], 'baseline_cost': original['cost'],
                   'labels_sha256': sha(cell / 'labels.json'),
                   'placements_sha256': sha(cell / 'placements/placements.jsonl')})
    return result, {'x': x, 'graph': graph, 'eligible': eligible,
                    'assignment': target.astype(np.int64), 'rank_target': rank_target}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--train-count', type=int, default=64)
    ap.add_argument('--validation-count', type=int, default=16)
    args = ap.parse_args()
    if args.out.exists(): raise ValueError('preserve existing corpus; use a new output path')
    protocol = read(PROTOCOL); args.out.mkdir(parents=True)
    inherited = read(ROOT / 'simulation_data/gnn_environment_search_v1/dag_reservation_bridge_v1/protocol_before_run.json')['sources']
    for name, digest in inherited.items():
        if sha(ROOT / name) != digest: raise RuntimeError('historical source changed: ' + name)
    own = [Path(__file__).resolve(), PROTOCOL, ROOT / 'src/policy/dag_reservation/model.py']
    sources = {**inherited, **{str(p.relative_to(ROOT)): sha(p) for p in own}}
    save(args.out / 'protocol_before_run.json', {'protocol': protocol, 'sources': sources,
         'train_count': args.train_count, 'validation_count': args.validation_count})
    engine = BridgedDag(args.out / 'build'); save(args.out / 'native.json', engine.provenance)
    counts = {'train': args.train_count, 'validation': args.validation_count,
              'test': protocol['sealed_test_seeds'][1] - protocol['sealed_test_seeds'][0] + 1}
    rows = []; seen = set(); shape = protocol['shape']
    for split, key in [('train', 'train_seeds'), ('validation', 'validation_seeds'), ('test', 'sealed_test_seeds')]:
        lo, hi = protocol[key]
        if counts[split] < 1 or lo + counts[split] - 1 > hi: raise ValueError('invalid split count')
        samples = []
        for seed in range(lo, lo + counts[split]):
            row, arrays = make_case(engine, seed, shape, split, args.out)
            if row['identity'] in seen: raise RuntimeError('duplicate source identity')
            seen.add(row['identity']); rows.append(row)
            if arrays is not None: samples.append(arrays)
            print('DATA', split, seed, flush=True)
        if samples:
            np.savez_compressed(args.out / f'{split}.npz', seeds=np.arange(lo, lo + len(samples)),
                                **{k: np.stack([p[k] for p in samples]) for k in samples[0]})
    for name, digest in sources.items():
        if sha(ROOT / name) != digest: raise RuntimeError('source changed during generation: ' + name)
    files = {p.name: sha(p) for p in args.out.glob('*.npz')}
    save(args.out / 'METADATA.json', {'contract': CONTRACT, 'shape': shape,
         'label': 'best_observed_complete_schedule_not_optimal', 'counts': counts,
         'cases': rows, 'files': files, 'sources': sources,
         'test_unopened': True, 'native_sha256': sha(args.out / 'native.json')})
    print('DATA COMPLETE', len(rows), flush=True)


if __name__ == '__main__': main()
