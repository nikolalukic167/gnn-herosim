"""Frozen CPU capacity calibration, block-search qualification, and mandatory live replay."""
import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from src.placement.radical.block_moves import baseline, scaled_problem, graph_features, move_pool, apply_move, order_moves
from src.placement.radical.dispatch import priorities
from src.placement.radical.dispatch_adaptive import AdaptiveDispatch
from src.placement.radical.dispatch_live import replay, herosim
from src.placement.radical.environment import pack, serialized, initial
from src.policy.dispatch_priority.model import PriorityNet
from scripts_cosim.workflow_proposal_gate import configure_environment

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'experiments/mixed_dispatch_block_screen_v1.json'


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank_hash(rank):
    return hashlib.sha256(np.asarray(rank, dtype=np.int64).tobytes()).hexdigest()


def timed(fn, repeats):
    expected, times = None, []
    for _ in range(repeats + 1):
        start = time.perf_counter()
        result = np.asarray(fn())
        times.append((time.perf_counter() - start) * 1000)
        if expected is not None and not np.array_equal(result, expected):
            raise ValueError('nondeterministic calibration')
        expected = result.copy()
    return {'median_ms': float(np.median(times[1:])), 'samples_ms': times[1:]}


def parity(engine, b, a, rank, cell, name, live=False):
    cost, ends = engine.score_priority(b, a, rank)
    ref = replay(b, a, rank)
    if cost != ref['objective'] or not np.array_equal(ends, ref['ends']):
        raise ValueError('native-independent replay mismatch')
    rec = {'objective': cost, 'ends': ends.tolist(), 'events': ref['events'],
           'assignment': a.tolist(), 'priority': rank.tolist()}
    save(cell / f'{name}_replay.json', rec)
    if live:
        with (cell / f'{name}_live.log').open('w') as log:
            actual = herosim(b, a, rank, log, int(b['seed']))
        if abs(actual['total_rtt'] * 1000 - cost) > 1e-6:
            raise ValueError('live objective mismatch')
        seen = set()
        for task in actual['tasks']:
            j, k = map(int, task['taskType']['name'][2:].split('_'))
            if (j, k) in seen or task['executionNode'] != f'node{a[j,k]}' or abs(task['doneTime'] * 1000 - ends[j,k]) > 1e-6:
                raise ValueError('live task mismatch')
            seen.add((j, k))
        if len(seen) != a.size or actual['mixed_execution']['dispatch_contract'] != 'mixed_ready_priority_v1':
            raise ValueError('live coverage mismatch')
        save(cell / f'{name}_live.json', actual)
    starts = [e for e in ref['events'] if e['event'] == 'start']
    active = sum(e['end'] - e['start'] for e in starts) / float(ends.max())
    p = np.take_along_axis(b['p'], a[..., None], -1)[..., 0]
    return {'mean_active_hosts': active, 'utilization': active / b['p'].shape[-1],
            'stretch': float(np.mean(ends[:, -1] / p.sum(1)))}


def hand_search(engine, b, protocol, method, wall):
    start = time.perf_counter()
    deadline = start + protocol['budget_ms'] / 1000
    cost, a, rank = baseline(engine, b)
    start_rank = rank.copy()
    trace, scores = [], 0
    safety = .003
    if method == 'native':
        if not wall:
            cost, rank = engine.search_priority(b, a, rank, protocol['count_budget'])
            scores = protocol['count_budget']
            trace.append({'steps': scores, 'cost': cost, 'priority': rank.tolist()})
        else:
            best = rank.copy()
            for iteration in range(100000):
                if time.perf_counter() + safety >= deadline:
                    break
                tick = time.perf_counter()
                mode = iteration % 13
                if mode < 4:
                    candidate = priorities(b, a, ('srpt', 'spt', 'longest', 'type')[mode]) if iteration < 13 else best.copy()
                else:
                    _, candidate = engine.adaptive_plan(b, a, mode - 4)
                value, candidate = engine.search_priority(b, a, candidate, 64)
                scores += 64
                if value < cost:
                    cost, best = value, candidate.copy()
                trace.append({'mode': mode, 'steps': 64, 'cost': value, 'priority': candidate.tolist()})
                safety = max(safety, (time.perf_counter() - tick) * 1.5)
            rank = best
    else:
        _, _, _, context = graph_features(b, a)
        pool, _ = move_pool(b, a, rank, protocol['pool_per_family'], tuple(protocol['block_lengths']))
        order = order_moves(b, a, rank, pool, method, context)
        for index in order:
            if wall and time.perf_counter() + safety >= deadline or not wall and scores >= protocol['count_budget']:
                break
            tick = time.perf_counter()
            candidate = apply_move(rank, pool[index])
            value = engine.score_priority(b, a, candidate)[0]
            accepted = value < cost
            trace.append({'move': pool[index].record(), 'cost': value, 'accepted': accepted})
            scores += 1
            if accepted:
                cost, rank = value, candidate
            safety = max(safety, (time.perf_counter() - tick) * 1.5)
    elapsed = (time.perf_counter() - start) * 1000
    return {'cost': cost, 'priority': rank.tolist(), 'assignment': a.tolist(), 'initial_priority': start_rank.tolist(),
            'scores': scores, 'time_ms': elapsed, 'trace': trace}


def reference(engine, b, a, rank, protocol, cell):
    best_cost = engine.score_priority(b, a, rank)[0]
    rounds, scored = [], 0
    with (cell / 'candidates.jsonl').open('w') as output:
        for round_id in range(protocol['reference_rounds']):
            pool, _ = move_pool(b, a, rank, protocol['pool_per_family'], tuple(protocol['block_lengths']))
            initial_cost, winner, winner_rank = best_cost, None, rank.copy()
            for index, move in enumerate(pool):
                candidate = apply_move(rank, move)
                cost = engine.score_priority(b, a, candidate)[0]
                output.write(json.dumps({'round': round_id, 'index': index, 'move': move.record(),
                                         'cost': cost, 'delta': cost - initial_cost, 'priority_sha256': rank_hash(candidate)}) + '\n')
                scored += 1
                if cost < best_cost:
                    best_cost, winner, winner_rank = cost, index, candidate
            rounds.append({'round': round_id, 'initial_priority': rank.tolist(), 'initial_cost': initial_cost,
                           'candidate_count': len(pool), 'winner': winner, 'final_cost': best_cost,
                           'final_priority': winner_rank.tolist()})
            rank = winner_rank
            if winner is None:
                break
    save(cell / 'reference_rounds.json', rounds)
    return {'cost': best_cost, 'priority': rank.tolist(), 'scores': scored}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(PROTOCOL.read_text())
    torch.set_num_threads(1)
    configure_environment()
    prior = ROOT / 'simulation_data/gnn_environment_search_v1/dispatch_scale_s0_v1/protocol_before_run.json'
    old_hashes = json.loads(prior.read_text())['sources']
    for name, digest in old_hashes.items():
        if sha(ROOT / name) != digest:
            raise ValueError('historical fingerprint changed: ' + name)
    paths = {**old_hashes, **{str(p.relative_to(ROOT)): sha(p) for p in
             (PROTOCOL, Path(__file__).resolve(), ROOT / 'src/placement/radical/block_moves.py')}}
    save(out / 'protocol_before_run.json', {'protocol': protocol, 'sources': paths, 'platform': platform.platform(),
                                         'threads': {k: os.getenv(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}})
    engine = AdaptiveDispatch(out / 'build')
    save(out / 'native.json', engine.provenance)
    identities = set()
    def make_input(seed, shape, directory):
        b = scaled_problem(seed, *shape)
        identity = hashlib.sha256(pack(b).tobytes()).hexdigest()
        if identity in identities or not np.array_equal(pack(b), pack(scaled_problem(seed, *shape))):
            raise ValueError('duplicate or nondeterministic input')
        identities.add(identity)
        directory.mkdir(parents=True)
        save(directory / 'input.json', serialized(b))
        return b
    for seed in (139900, 139901):
        cell = out / 'parity' / str(seed)
        b = make_input(seed, (4, 3, 8), cell)
        _, a, rank = baseline(engine, b)
        parity(engine, b, a, rank, cell, 'baseline', live=True)
    print('SMALL PARITY PASS', flush=True)
    calibration = []
    with torch.inference_mode():
        for shape, first in zip(protocol['scales'], protocol['calibration_seed_starts']):
            for seed in range(first, first + protocol['calibration_cases']):
                cell = out / 'calibration' / str(seed)
                b = make_input(seed, shape, cell)
                _, a, rank = baseline(engine, b)
                metrics = parity(engine, b, a, rank, cell, 'baseline', live=seed == first)
                x, adj, _, _ = graph_features(b, a)
                torch.manual_seed(0)
                model = PriorityNet(x.shape[-1], hidden=16, layers=2, mp=True).eval()
                def proxy():
                    _, aa, rr = baseline(engine, b)
                    xx, adjacency, _, context = graph_features(b, aa)
                    model(torch.from_numpy(xx)[None], torch.from_numpy(adjacency)[None])
                    pool, _ = move_pool(b, aa, rr, protocol['pool_per_family'], tuple(protocol['block_lengths']))
                    indices = order_moves(b, aa, rr, pool, 'hop4', context)[:protocol['count_budget']]
                    return np.array([engine.score_priority(b, aa, apply_move(rr, pool[i]))[0] for i in indices])
                timings = {
                    'baseline': timed(lambda: np.stack(baseline(engine, b)[1:]), protocol['timing_repeats']),
                    'features_124': timed(lambda: graph_features(b, a)[2], protocol['timing_repeats']),
                    'full_proxy': timed(proxy, protocol['timing_repeats']),
                }
                row = {'seed': seed, 'shape': shape, 'capacity': metrics, 'timings': timings,
                       'physical_sha256': hashlib.sha256(pack(b).tobytes()).hexdigest()}
                save(cell / 'read.json', row)
                calibration.append(row)
                print('CAL', seed, shape, round(timings['full_proxy']['median_ms'], 2), flush=True)
    small_stretch = float(np.median([r['capacity']['stretch'] for r in calibration if r['shape'] == protocol['scales'][0]]))
    summaries = []
    for shape in protocol['scales']:
        rows = [r for r in calibration if r['shape'] == shape]
        p95 = float(np.quantile([r['timings']['full_proxy']['median_ms'] for r in rows], .95))
        utilization = float(np.median([r['capacity']['utilization'] for r in rows]))
        stretch = float(np.median([r['capacity']['stretch'] for r in rows]))
        summaries.append({'shape': shape, 'p95_ms': p95, 'utilization': utilization, 'stretch': stretch,
                          'eligible': p95 <= protocol['budget_ms'] * protocol['calibration_fraction'] and
                          utilization >= protocol['capacity_min_host_utilization'] and
                          stretch <= small_stretch * protocol['capacity_max_stretch_ratio_to_small']})
    eligible = [r for r in summaries if r['eligible']]
    selected = max(eligible, key=lambda r: r['shape'][0] * r['shape'][1]) if eligible else None
    save(out / 'calibration.json', {'scales': summaries, 'selected': selected, 'rows': calibration})
    print('CALIBRATION', json.dumps(summaries), flush=True)
    if selected is None:
        save(out / 'read.json', {'status': 'S0-NO-GO', 'training_allowed': False})
    else:
        rows = []
        methods = [*protocol['hand_proposers'], 'native']
        for seed in range(protocol['qualification_seeds'][0], protocol['qualification_seeds'][1] + 1):
            cell = out / 'qualification' / str(seed)
            b = make_input(seed, selected['shape'], cell)
            cost, a, rank = baseline(engine, b)
            arms = {}
            for method in methods:
                arms['count_' + method] = hand_search(engine, b, protocol, method, wall=False)
                repeats = [hand_search(engine, b, protocol, method, wall=True) for _ in range(protocol['wall_repeats'])]
                chosen = max(repeats, key=lambda r: r['cost'])
                arms['wall_' + method] = {**chosen, 'repeats': repeats,
                    'all_within_budget': all(r['time_ms'] <= protocol['budget_ms'] for r in repeats)}
            structured = reference(engine, b, a, rank, protocol, cell)
            best = min([structured, *arms.values()], key=lambda r: r['cost'])
            row = {'seed': seed, 'baseline': {'cost': cost, 'assignment': a.tolist(), 'priority': rank.tolist()},
                   'arms': arms, 'structured': structured, 'reference': {'cost': best['cost'], 'priority': best['priority']}}
            save(cell / 'read.json', row)
            rows.append(row)
            print('SCREEN', seed, 'gain_pct', round((cost - best['cost']) / cost * 100, 3), flush=True)
        base = np.array([r['baseline']['cost'] for r in rows])
        ref = np.array([r['reference']['cost'] for r in rows])
        gain = (base - ref) / base * 100
        denominator = float(np.sum(base - ref))
        capture = {name: float(np.sum(base - np.array([r['arms'][name]['cost'] for r in rows])) / denominator) if denominator > 0 else 1.
                   for name in rows[0]['arms']}
        best_wall = max((name for name in capture if name.startswith('wall_')), key=lambda name: capture[name])
        best_count = max((name for name in capture if name.startswith('count_')), key=lambda name: capture[name])
        headroom = float(np.median(gain)) >= protocol['headroom_bar_median_pct']
        hardness = max(capture.values()) < protocol['hardness_bar_max_capture_fraction']
        timing = all(r['arms'][name]['all_within_budget'] for r in rows for name in r['arms'] if name.startswith('wall_'))
        report = {'status': 'S2-PENDING-LIVE', 'median_headroom_pct': float(np.median(gain)), 'gain_pct': gain.tolist(),
                  'capture': capture, 'best_wall': best_wall, 'best_count': best_count,
                  'headroom_pass': headroom, 'hardness_pass': hardness, 'wall_timing_pass': timing,
                  'residual_information': 'NOT-RUN', 'training_allowed': False, 'live_runs': 0, 'live_operations': 0}
        save(out / 'read.json', report)
        for row in rows:
            cell = out / 'qualification' / str(row['seed'])
            b = scaled_problem(row['seed'], *selected['shape'])
            a = np.array(row['baseline']['assignment'])
            plans = {'baseline': row['baseline'], 'reference': row['reference'], 'hand': row['arms'][best_wall]}
            (cell / 'placements').mkdir()
            with (cell / 'placements/placements.jsonl').open('w') as output:
                for name, arm in plans.items():
                    rank = np.array(arm['priority'])
                    parity(engine, b, a, rank, cell, name, live=True)
                    output.write(json.dumps({'arm': name, 'placement_plan': a.tolist(), 'priority': rank.tolist(), 'objective': arm['cost']}) + '\n')
                    report['live_runs'] += 1
                    report['live_operations'] += a.size
            print('LIVE', row['seed'], flush=True)
        report['status'] = 'RESIDUAL-GATE-REQUIRED' if headroom and hardness and timing else 'PRETRAINING-NO-GO'
        save(out / 'read.json', report)
        print(json.dumps(report, indent=2), flush=True)
    for name, digest in paths.items():
        if sha(ROOT / name) != digest:
            raise ValueError('source changed during run: ' + name)
    save(out / 'artifacts.json', {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.suffix in ('.json', '.jsonl')})


if __name__ == '__main__':
    main()
