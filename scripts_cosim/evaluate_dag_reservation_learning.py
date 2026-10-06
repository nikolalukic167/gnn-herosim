"""Fresh matched live gate for learned irregular-DAG reservation proposals."""
import argparse
import itertools
import json
import time
from pathlib import Path
import numpy as np
import torch
from src.placement.radical.dag_reservation import problem
from src.placement.radical.dag_reservation_bridge import BridgedDag, baseline
from src.policy.dag_reservation.model import CONTRACT, ReservationProposalNet, features, decode
from scripts_cosim.dag_reservation_bridge_screen import control, base_plan
from scripts_cosim.dag_reservation_screen import PLAN_KEYS, plan, live, verify_plan, verify_live
from scripts_cosim.mixed_joint_dispatch_screen import ROOT, read, save, sha, gains
from scripts_cosim.audit_radical_physics import load_problem
from scripts_cosim.workflow_proposal_gate import configure_environment

PROTOCOL = ROOT / 'experiments/dag_reservation_learning_v1_protocol.json'
CORPUS = ROOT / 'simulation_data/gnn_environment_search_v1/dag_reservation_learning_corpus_v1'
MODELS = ROOT / 'models/dag_reservation_learning_v1'
BUDGET_MS = 250


def load_models(meta):
    result = {}; receipts = {}
    for arm, seed in itertools.product(('gnn', 'mpoff'), (101, 102)):
        stem = MODELS / f'{arm}_seed{seed}'
        sidecar = read(stem.with_suffix('.contract.json'))
        if sidecar['contract'] != CONTRACT or sidecar['arm'] != arm or sidecar['seed'] != seed:
            raise ValueError('checkpoint contract mismatch')
        if sidecar['data_metadata_sha256'] != sha(CORPUS / 'METADATA.json'):
            raise ValueError('checkpoint data fingerprint mismatch')
        for name, digest in sidecar['sources'].items():
            if sha(ROOT / name) != digest: raise ValueError('checkpoint source changed: ' + name)
        arch = sidecar['architecture']
        if arch['mp'] != (arm == 'gnn') or arch['hosts'] != meta['shape'][2]:
            raise ValueError('checkpoint architecture mismatch')
        model = ReservationProposalNet(**arch)
        model.load_state_dict(torch.load(stem.with_suffix('.pt'), map_location='cpu', weights_only=True), strict=True)
        model.eval(); result[f'{arm}_{seed}'] = model
        receipts[f'{arm}_{seed}'] = {'checkpoint': sha(stem.with_suffix('.pt')),
                                    'sidecar': sha(stem.with_suffix('.contract.json')),
                                    'training': sha(stem.with_suffix('.training.json'))}
    return result, receipts


def learned(engine, b, model, budget_ms):
    tick = time.perf_counter()
    start_cost, start_a, start_rank, start_decoder, rule = baseline(engine, b)
    x, graph, eligible = features(b, start_a, start_rank)
    a, rank = decode(model, x, graph, eligible, *b['p'].shape[:2])
    proposals = [(engine.plan(b, a, rank, mode)[0], a, rank, mode, 'model') for mode in (0, 1)]
    proposals.append((start_cost, start_a, start_rank, start_decoder, f'hand:{rule}'))
    cost, a, rank, decoder, source = min(proposals, key=lambda p: p[0])
    remaining = budget_ms / 1000 - (time.perf_counter() - tick) - .006
    count = 0
    if remaining > .001:
        cost, a, rank, decoder, count = engine.search(b, a, rank, 2**31 - 1, 2, .001, 77, remaining)
    result = plan(engine, b, a, rank, decoder, cost)
    return {**result, 'source': source, 'scores': count, 'time_ms': (time.perf_counter() - tick) * 1000}


def hand_envelope(engine, b, budget_ms):
    result = {}
    for mode, temperature in itertools.product((0, 1, 2), (0., .001, .01)):
        name = f'{mode}_{temperature:g}'
        result[name] = control(engine, b, mode, temperature, budget_ms)
    return result


def summarize(rows, budget):
    keys = ('gnn_101', 'gnn_102', 'mpoff_101', 'mpoff_102')
    series = {name: np.array([row['models'][name]['cost'] for row in rows]) for name in keys}
    gnn = (series['gnn_101'] + series['gnn_102']) / 2
    mpoff = (series['mpoff_101'] + series['mpoff_102']) / 2
    hand = np.array([row['hand']['cost'] for row in rows])
    envelope = np.array([min(p['cost'] for p in row['hand_envelope'].values() if p['time_ms'] <= budget) for row in rows])
    reference = np.array([row['reference']['cost'] for row in rows])
    timing = {name: max(row['models'][name]['time_ms'] for row in rows) for name in keys}
    timing['hand'] = max(row['hand']['time_ms'] for row in rows)
    comparisons = {'mpoff': gains(gnn, mpoff), 'hand': gains(gnn, hand),
                   'hand_envelope': gains(gnn, envelope), 'reference': gains(gnn, reference)}
    return {'inputs': len(rows), 'budget_ms': budget, 'timing_max_ms': timing,
            'all_timing_valid': all(v <= budget for v in timing.values()),
            'mean_job_completion_ms': {**{n: float(v.mean()) for n, v in series.items()},
                                       'gnn_draw_mean': float(gnn.mean()), 'mpoff_draw_mean': float(mpoff.mean()),
                                       'hand': float(hand.mean()), 'hand_envelope': float(envelope.mean()),
                                       'reference': float(reference.mean())},
            'gnn_gain_pct': comparisons,
            'exploratory_gnn_win': bool(all(comparisons[n]['median_pct'] > 0 and comparisons[n]['ci95_pct'][0] > 0
                                             for n in ('mpoff', 'hand')) and all(v <= budget for v in timing.values()))}


def run(out):
    torch.set_num_threads(1)
    if out.exists(): raise ValueError('preserve existing gate; use new output path')
    out.mkdir(parents=True); protocol = read(PROTOCOL); meta = read(CORPUS / 'METADATA.json')
    if meta['counts'] != {'train': 64, 'validation': 16, 'test': 16} or not meta['test_unopened']:
        raise ValueError('full sealed corpus required')
    for name, digest in meta['sources'].items():
        if sha(ROOT / name) != digest: raise ValueError('corpus source changed: ' + name)
    for name, digest in meta['files'].items():
        if sha(CORPUS / name) != digest: raise ValueError('corpus file changed: ' + name)
    models, receipts = load_models(meta)
    sources = {**meta['sources'], **{str(p.relative_to(ROOT)): sha(p) for p in
               (Path(__file__).resolve(), PROTOCOL, ROOT / 'src/policy/dag_reservation/train.py')}}
    save(out / 'protocol_before_run.json', {'protocol': protocol, 'sources': sources,
         'corpus_metadata_sha256': sha(CORPUS / 'METADATA.json'), 'checkpoints': receipts})
    engine = BridgedDag(out / 'build'); save(out / 'native.json', engine.provenance)
    configure_environment(); rows = []; shape = protocol['shape']; runs = operations = 0
    for seed in range(protocol['sealed_test_seeds'][0], protocol['sealed_test_seeds'][1] + 1):
        b = problem(seed, *shape); cell = out / 'fresh' / str(seed)
        cell.mkdir(parents=True); save(cell / 'input.json', read(CORPUS / 'test' / str(seed) / 'input.json'))
        if sha(cell / 'input.json') != next(p['input_sha256'] for p in meta['cases'] if p['seed'] == seed):
            raise ValueError('sealed input mismatch')
        base = base_plan(engine, b)
        model_plans = {name: learned(engine, b, model, BUDGET_MS) for name, model in models.items()}
        hand = control(engine, b, 0, .001, BUDGET_MS)
        envelope = hand_envelope(engine, b, BUDGET_MS)
        refs = []
        for source, mode, temperature in itertools.product(('hand', 'baseline'), (0, 1, 2), (0., .001, .01)):
            start = hand if source == 'hand' else base
            cost, a, rank, decoder, count = engine.search(b, np.array(start['assignment']),
                    np.array(start['priority']), 8192, mode, temperature, 29)
            refs.append({**plan(engine, b, a, rank, decoder, cost), 'start': source,
                         'search_mode': mode, 'temperature': temperature, 'steps': count})
        reference = min([base, hand, *envelope.values(), *refs], key=lambda p: p['cost'])
        row = {'seed': seed, 'models': model_plans, 'hand': hand, 'hand_envelope': envelope,
               'baseline': base, 'references': refs,
               'reference': {key: reference[key] for key in PLAN_KEYS}}
        save(cell / 'read.json', row); rows.append(row)
        print('GATE SEARCH', seed, {n: p['cost'] for n, p in model_plans.items()},
              'hand', hand['cost'], 'reference', reference['cost'], flush=True)
    save(out / 'screen_before_live.json', summarize(rows, BUDGET_MS))
    for row in rows:
        seed = row['seed']; b = problem(seed, *shape); cell = out / 'fresh' / str(seed)
        (cell / 'placements').mkdir()
        with (cell / 'placements/placements.jsonl').open('w') as trace:
            for name, p in [(n, row['models'][n]) for n in models] + [('hand', row['hand']), ('reference', row['reference'])]:
                counts = live(engine, b, p, cell, name); runs += counts['runs']; operations += counts['operations']
                trace.write(json.dumps({'arm': name, **{k: p[k] for k in PLAN_KEYS}}) + '\n')
        print('GATE LIVE', seed, flush=True)
    report = summarize(rows, BUDGET_MS)
    report.update({'live_runs': runs, 'live_operations': operations,
                   'checkpoints': receipts, 'corpus_metadata_sha256': sha(CORPUS / 'METADATA.json')})
    save(out / 'read.json', report)
    for name, digest in sources.items():
        if sha(ROOT / name) != digest: raise ValueError('source changed during gate: ' + name)
    save(out / 'artifacts.json', {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*'))
                                  if p.is_file() and p.suffix in ('.json', '.jsonl')})
    print(json.dumps(report, indent=2), flush=True)


def audit(out):
    torch.set_num_threads(1)
    pre = read(out / 'protocol_before_run.json')
    for name, digest in pre['sources'].items():
        if sha(ROOT / name) != digest: raise ValueError('source changed: ' + name)
    for name, digest in read(out / 'artifacts.json').items():
        if sha(out / name) != digest: raise ValueError('artifact changed: ' + name)
    if sha(CORPUS / 'METADATA.json') != pre['corpus_metadata_sha256']:
        raise ValueError('corpus changed')
    models, receipts = load_models(read(CORPUS / 'METADATA.json'))
    if receipts != pre['checkpoints']: raise ValueError('checkpoints changed')
    engine = BridgedDag(out / 'build')
    if engine.provenance != read(out / 'native.json'): raise ValueError('native changed')
    rows = []; runs = operations = proposals = 0
    for seed in range(pre['protocol']['sealed_test_seeds'][0], pre['protocol']['sealed_test_seeds'][1] + 1):
        cell = out / 'fresh' / str(seed); b = load_problem(cell / 'input.json'); row = read(cell / 'read.json')
        if row['seed'] != seed: raise ValueError('seed mismatch')
        for name, p in [(n, row['models'][n]) for n in models] + [('hand', row['hand']), ('reference', row['reference'])]:
            verify_plan(engine, b, p)
            served = read(cell / f'{name}_live.json'); verify_live(b, p, served)
            runs += 1; operations += b['p'].shape[0] * b['p'].shape[1]
        for p in row['hand_envelope'].values(): verify_plan(engine, b, p)
        if base_plan(engine, b) != row['baseline']: raise ValueError('baseline mismatch')
        for ref in row['references']:
            start = row['hand'] if ref['start'] == 'hand' else row['baseline']
            value, a, rank, decoder, count = engine.search(b, np.array(start['assignment']),
                   np.array(start['priority']), 8192, ref['search_mode'], ref['temperature'], 29)
            if plan(engine, b, a, rank, decoder, value) != {k: ref[k] for k in PLAN_KEYS} or count != ref['steps']:
                raise ValueError('reference reproduction failed')
            proposals += count
        best = min([row['baseline'], row['hand'], *row['hand_envelope'].values(), *row['references']], key=lambda p: p['cost'])
        if row['reference'] != {k: best[k] for k in PLAN_KEYS}: raise ValueError('reference selection mismatch')
        plans = [json.loads(line) for line in (cell / 'placements/placements.jsonl').read_text().splitlines()]
        if [p['arm'] for p in plans] != [*models, 'hand', 'reference']: raise ValueError('plan coverage mismatch')
        rows.append(row); print('GATE AUDIT', seed, flush=True)
    expected = summarize(rows, BUDGET_MS); report = read(out / 'read.json')
    if any(report[k] != v for k, v in expected.items()): raise ValueError('report mismatch')
    if (runs, operations) != (96, 24576) or report['live_runs'] != runs or report['live_operations'] != operations:
        raise ValueError('live coverage mismatch')
    result = {'status': 'PASS', 'live_runs': runs, 'live_operations': operations,
              'reference_proposals_reexecuted': proposals, 'report_sha256': sha(out / 'read.json')}
    save(out / 'AUDIT.json', result); print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--audit-only', action='store_true'); args = ap.parse_args()
    audit(args.out) if args.audit_only else run(args.out)
