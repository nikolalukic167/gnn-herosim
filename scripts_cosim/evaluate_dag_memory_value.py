"""Sealed, paired actual-HeROsim gate for learned DAG memory moves."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts_cosim.dag_memory_lifetime_s0 import identity, live_case, make_case, record, search
from scripts_cosim.mixed_joint_dispatch_screen import ROOT, read, save, sha
from scripts_cosim.workflow_proposal_gate import configure_environment
from src.placement.radical.dag_reservation_bridge import BridgedDag
from src.placement.radical.environment import serialized
from src.policy.dag_memory.model import CONTRACT, MemoryValueNet, serve

PROTOCOL = ROOT / 'experiments/dag_memory_value_learning_v1_protocol.json'
CORPUS = ROOT / 'simulation_data/gnn_environment_search_v1/dag_memory_value_corpus_v1'
CHECKPOINTS = ROOT / 'models/dag_memory_value_learning_v1'


def load_models(protocol, metadata):
    models = {}
    receipts = {}
    digest = sha(CORPUS / 'METADATA.json')
    for arm in protocol['arms']:
        for seed in protocol['draws']:
            name = f'{arm}_seed{seed}'
            path = CHECKPOINTS / f'{name}.pt'
            sidecar = read(path.with_suffix('.contract.json'))
            assert sidecar['contract'] == CONTRACT
            assert sidecar['arm'] == arm and sidecar['seed'] == seed
            assert sidecar['data_metadata_sha256'] == digest
            assert sidecar['selection'] == 'minimum_complete_served_validation_cost'
            assert path.with_suffix('.training.json').is_file()
            for source, fingerprint in sidecar['sources'].items():
                assert sha(ROOT / source) == fingerprint, source
            model = MemoryValueNet(**sidecar['architecture'])
            model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
            model.eval()
            models[name] = model
            receipts[name] = {'checkpoint_sha256': sha(path), 'sidecar_sha256': sha(path.with_suffix('.contract.json')),
                              'selected_epoch': sidecar['selected_epoch'],
                              'validation_cost': sidecar['validation_mean_job_completion_ms']}
    return models, receipts


def interval(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(290170)
    resample = rng.integers(0, len(values), size=(20000, len(values)))
    statistics = np.median(values[resample], axis=1)
    return {'median_pct': float(np.median(values)),
            'bootstrap_95_pct': [float(x) for x in np.quantile(statistics, [.025, .975])]}


def run(out):
    protocol = read(PROTOCOL)
    metadata = read(CORPUS / 'METADATA.json')
    assert metadata['contract'] == CONTRACT and metadata['test_unopened']
    assert metadata['shape'] == protocol['shape']
    expected = {item['seed']: item['identity'] for item in metadata['cases'] if item['split'] == 'test'}
    assert sorted(expected) == list(range(protocol['sealed_test_seeds'][0], protocol['sealed_test_seeds'][1] + 1))
    for source, fingerprint in metadata['source_fingerprints'].items():
        assert sha(ROOT / source) == fingerprint, source
    for filename, fingerprint in metadata['files'].items():
        assert sha(CORPUS / filename) == fingerprint, filename
    torch.set_num_threads(1)
    models, receipts = load_models(protocol, metadata)
    out.mkdir(parents=True, exist_ok=False)
    sources = [PROTOCOL, Path(__file__).resolve(),
               ROOT / 'scripts_cosim/dag_memory_lifetime_s0.py',
               ROOT / 'src/placement/radical/dag_memory_live.py',
               ROOT / 'src/placement/radical/dag_reservation.py',
               ROOT / 'src/placement/radical/dag_reservation.cpp']
    save(out / 'protocol_before_run.json', {'protocol': protocol,
         'sources': {str(path.relative_to(ROOT)): sha(path) for path in sources},
         'corpus_metadata_sha256': sha(CORPUS / 'METADATA.json'), 'checkpoints': receipts})
    engine = BridgedDag(out / 'build')
    save(out / 'native.json', engine.provenance)
    configure_environment()
    rows = []
    live_runs = operations = 0
    for seed in sorted(expected):
        b, a, rank, simple, initial_rule, prep_ms = make_case(engine, seed, protocol['shape'])
        assert identity(b) == expected[seed], seed
        initial = simple[initial_rule]
        hand = search(b, a, rank, initial, 'greedy', seed + 31,
                      seconds=max(0, protocol['decision_budget_ms'] - prep_ms - 8) / 1000)
        hand['search_ms'] = hand['time_ms']
        hand['time_ms'] += prep_ms
        references = [search(b, a, rank, source, 'anneal', seed + 71 + offset, steps=2048)
                      for offset, source in enumerate((initial, hand))]
        reference = min([initial, hand, *references], key=lambda item: item['cost'])
        learned = {}
        for name, model in models.items():
            served = serve(model, b, a, rank, initial, prep_ms, protocol['decision_budget_ms'])
            exact = record(b, a, rank, served['retain'])
            assert served['cost'] == exact['cost']
            learned[name] = {**exact, **served}
        row = {'seed': seed, 'identity': expected[seed], 'assignment': a.tolist(),
               'priority': rank.tolist(), 'initial_rule': initial_rule, 'simple': simple,
               'preparation_ms': prep_ms, 'hand': hand, 'references': references,
               'reference': reference, 'learned': learned}
        cell = out / 'test' / str(seed)
        cell.mkdir(parents=True)
        save(cell / 'input.json', serialized(b))
        save(cell / 'read.json', row)
        for name, plan in [('initial', initial), ('hand', hand), ('reference', reference), *learned.items()]:
            operations += live_case(b, a, rank, plan, cell, name)
            live_runs += 1
        rows.append(row)
        print('LIVE', seed, {'hand': hand['cost'], 'reference': reference['cost'],
                             **{name: plan['cost'] for name, plan in learned.items()}}, flush=True)
    comparisons = {}
    for arm in protocol['arms']:
        for control in ('mpoff', 'hand_mlp', 'hand'):
            if arm != 'gnn' or arm == control:
                continue
            values = []
            for row in rows:
                draws = []
                for draw in protocol['draws']:
                    treatment = row['learned'][f'gnn_seed{draw}']['cost']
                    baseline = row['hand']['cost'] if control == 'hand' else row['learned'][f'{control}_seed{draw}']['cost']
                    draws.append(100 * (baseline - treatment) / baseline)
                values.append(float(np.mean(draws)))
            comparisons[f'gnn_vs_{control}'] = interval(values)
    comparisons['reference_vs_hand'] = interval([100 * (row['hand']['cost'] - row['reference']['cost']) /
                                                 row['hand']['cost'] for row in rows])
    mean_cost = {name: float(np.mean([row['learned'][name]['cost'] for row in rows])) for name in models}
    report = {'cases': len(rows), 'live_runs': live_runs, 'live_operations': operations,
              'mean_cost_ms': mean_cost, 'mean_hand_cost_ms': float(np.mean([row['hand']['cost'] for row in rows])),
              'mean_reference_cost_ms': float(np.mean([row['reference']['cost'] for row in rows])),
              'comparisons': comparisons,
              'maximum_time_ms': {name: max(row['learned'][name]['time_ms'] for row in rows) for name in models},
              'maximum_hand_time_ms': max(row['hand']['time_ms'] for row in rows),
              'all_within_budget': bool(all(row['learned'][name]['time_ms'] <= protocol['decision_budget_ms']
                                            for row in rows for name in models) and
                                        all(row['hand']['time_ms'] <= protocol['decision_budget_ms'] for row in rows))}
    save(out / 'read.json', report)
    receipt = read(out / 'protocol_before_run.json')
    for source, fingerprint in receipt['sources'].items():
        assert sha(ROOT / source) == fingerprint, source
    for name, item in receipt['checkpoints'].items():
        path = CHECKPOINTS / f'{name}.pt'
        assert sha(path) == item['checkpoint_sha256']
        assert sha(path.with_suffix('.contract.json')) == item['sidecar_sha256']
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    run(parser.parse_args().out)
