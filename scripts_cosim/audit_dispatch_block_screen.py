"""Audit all structured candidates, hand traces, coverage, live events and gate arithmetic."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from src.placement.radical.block_moves import scaled_problem, baseline, move_pool
from src.placement.radical.dispatch import priorities
from src.placement.radical.dispatch_adaptive import AdaptiveDispatch
from src.placement.radical.dispatch_live import replay
from src.placement.radical.environment import pack
from scripts_cosim.audit_radical_physics import load_problem

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def independently_apply(rank, spec):
    order = sorted(range(rank.size), key=lambda i: int(rank.flat[i]))
    first, second = list(spec['block']), list(spec['other'])
    assert len(first) == len(set(first)) and set(first) <= set(order)
    if second:
        assert not set(first) & set(second)
        replacement = {min(order.index(i) for i in first): second, min(order.index(i) for i in second): first}
        result = []
        for position, node in enumerate(order):
            result.extend(replacement.get(position, []))
            if node not in first and node not in second:
                result.append(node)
    else:
        assert spec['anchor'] not in first
        remainder = [i for i in order if i not in first]
        position = remainder.index(spec['anchor'])
        result = remainder[:position] + first + remainder[position:]
    assert sorted(result) == list(range(rank.size))
    return np.array([result.index(i) for i in range(rank.size)], dtype=np.int64).reshape(rank.shape)


def audit(out):
    pre = read(out / 'protocol_before_run.json')
    protocol = pre['protocol']
    for name, digest in pre['sources'].items():
        assert sha(ROOT / name) == digest, name
    for name, digest in read(out / 'artifacts.json').items():
        assert sha(out / name) == digest, name
    engine = AdaptiveDispatch(out / 'build')
    assert engine.provenance == read(out / 'native.json')
    identities, live_runs, live_operations = set(), 0, 0
    for path in sorted(out.glob('*/*/input.json')):
        b = load_problem(path)
        seed = b['seed']
        assert np.array_equal(pack(b), pack(scaled_problem(seed, *b['p'].shape)))
        identity = hashlib.sha256(pack(b).tobytes()).hexdigest()
        assert identity not in identities
        identities.add(identity)
        for replay_path in sorted(path.parent.glob('*_replay.json')):
            rec = read(replay_path)
            a, rank = np.array(rec['assignment']), np.array(rec['priority'])
            cost, ends = engine.score_priority(b, a, rank)
            independent = replay(b, a, rank)
            assert cost == rec['objective'] == independent['objective']
            assert np.array_equal(ends, rec['ends']) and np.array_equal(ends, independent['ends'])
            assert rec['events'] == independent['events']
            live_path = replay_path.with_name(replay_path.name.replace('_replay.json', '_live.json'))
            if live_path.exists():
                actual = read(live_path)
                assert abs(actual['total_rtt'] * 1000 - cost) < 1e-6
                assert actual['mixed_execution']['dispatch_contract'] == 'mixed_ready_priority_v1'
                seen = set()
                for task in actual['tasks']:
                    j, k = map(int, task['taskType']['name'][2:].split('_'))
                    assert (j, k) not in seen
                    seen.add((j, k))
                    assert task['executionNode'] == f'node{a[j,k]}'
                    assert abs(task['doneTime'] * 1000 - ends[j,k]) < 1e-6
                assert len(seen) == a.size
                live_runs += 1
                live_operations += a.size
    calibration = read(out / 'calibration.json')
    assert len(calibration['rows']) == len(protocol['scales']) * protocol['calibration_cases']
    small = float(np.median([r['capacity']['stretch'] for r in calibration['rows'] if r['shape'] == protocol['scales'][0]]))
    for scale in calibration['scales']:
        rows = [r for r in calibration['rows'] if r['shape'] == scale['shape']]
        for r in rows:
            cell = out / 'calibration' / str(r['seed'])
            assert r == read(cell / 'read.json')
            for timing in r['timings'].values():
                assert len(timing['samples_ms']) == protocol['timing_repeats']
                assert timing['median_ms'] == float(np.median(timing['samples_ms']))
            b = load_problem(cell / 'input.json')
            rec = read(cell / 'baseline_replay.json')
            ends, a = np.array(rec['ends']), np.array(rec['assignment'])
            active = sum(e['end'] - e['start'] for e in rec['events'] if e['event'] == 'start') / float(ends.max())
            p = np.take_along_axis(b['p'], a[..., None], -1)[..., 0]
            assert r['capacity'] == {'mean_active_hosts': active, 'utilization': active / b['p'].shape[-1],
                                    'stretch': float(np.mean(ends[:, -1] / p.sum(1)))}
        assert scale['p95_ms'] == float(np.quantile([r['timings']['full_proxy']['median_ms'] for r in rows], .95))
        assert scale['utilization'] == float(np.median([r['capacity']['utilization'] for r in rows]))
        assert scale['stretch'] == float(np.median([r['capacity']['stretch'] for r in rows]))
        expected = scale['p95_ms'] <= protocol['budget_ms'] * protocol['calibration_fraction'] and scale['utilization'] >= protocol['capacity_min_host_utilization'] and scale['stretch'] <= small * protocol['capacity_max_stretch_ratio_to_small']
        assert scale['eligible'] == expected
    eligible = [r for r in calibration['scales'] if r['eligible']]
    assert calibration['selected'] == (max(eligible, key=lambda r: r['shape'][0] * r['shape'][1]) if eligible else None)
    report = read(out / 'read.json')
    candidates, trace_scores = 0, 0
    rows = []
    if calibration['selected'] is not None:
        expected_seeds = set(range(protocol['qualification_seeds'][0], protocol['qualification_seeds'][1] + 1))
        assert {int(p.name) for p in (out / 'qualification').iterdir()} == expected_seeds
        for seed in sorted(expected_seeds):
            cell = out / 'qualification' / str(seed)
            b, row = load_problem(cell / 'input.json'), read(cell / 'read.json')
            cost, a, rank = baseline(engine, b)
            assert row['baseline'] == {'cost': cost, 'assignment': a.tolist(), 'priority': rank.tolist()}
            rounds = read(cell / 'reference_rounds.json')
            groups = [[] for _ in rounds]
            with (cell / 'candidates.jsonl').open() as stream:
                for line in stream:
                    item = json.loads(line)
                    groups[item['round']].append(item)
            previous = rank.copy()
            for rec, items in zip(rounds, groups):
                assert np.array_equal(previous, rec['initial_priority'])
                initial_cost = engine.score_priority(b, a, previous)[0]
                assert initial_cost == rec['initial_cost']
                pool, _ = move_pool(b, a, previous, protocol['pool_per_family'], tuple(protocol['block_lengths']))
                assert len(items) == len(pool) == rec['candidate_count']
                winner, best, best_rank = None, initial_cost, previous.copy()
                for index, item in enumerate(items):
                    assert item['index'] == index and item['move'] == pool[index].record()
                    candidate = independently_apply(previous, item['move'])
                    measured = engine.score_priority(b, a, candidate)[0]
                    assert measured == item['cost'] and item['delta'] == measured - initial_cost
                    assert item['priority_sha256'] == hashlib.sha256(candidate.tobytes()).hexdigest()
                    if measured < best:
                        winner, best, best_rank = index, measured, candidate
                    candidates += 1
                assert rec['winner'] == winner and rec['final_cost'] == best
                assert np.array_equal(rec['final_priority'], best_rank)
                assert replay(b, a, best_rank)['objective'] == best
                previous = best_rank
            assert row['structured']['cost'] == rounds[-1]['final_cost']
            assert np.array_equal(row['structured']['priority'], previous)
            assert row['structured']['scores'] == sum(len(g) for g in groups)
            for name, chosen in row['arms'].items():
                repetitions = chosen.get('repeats', [chosen])
                if name.startswith('wall_'):
                    assert len(repetitions) == protocol['wall_repeats']
                    assert chosen['cost'] == max(r['cost'] for r in repetitions)
                    assert chosen['all_within_budget'] == all(r['time_ms'] <= protocol['budget_ms'] for r in repetitions)
                for arm in repetitions:
                    assert np.array_equal(arm['assignment'], a) and np.array_equal(arm['initial_priority'], rank)
                    current, current_cost, scores = rank.copy(), cost, 0
                    if name.endswith('_native'):
                        if name.startswith('count_'):
                            value, candidate = engine.search_priority(b, a, current, protocol['count_budget'])
                            assert arm['trace'] == [{'steps': protocol['count_budget'], 'cost': value, 'priority': candidate.tolist()}]
                            current, current_cost, scores = candidate, value, protocol['count_budget']
                        else:
                            for iteration, entry in enumerate(arm['trace']):
                                mode = iteration % 13
                                assert entry['mode'] == mode and entry['steps'] == 64
                                if mode < 4:
                                    candidate = priorities(b, a, ('srpt','spt','longest','type')[mode]) if iteration < 13 else current.copy()
                                else:
                                    _, candidate = engine.adaptive_plan(b, a, mode - 4)
                                value, candidate = engine.search_priority(b, a, candidate, 64)
                                assert value == entry['cost'] and np.array_equal(candidate, entry['priority'])
                                if value < current_cost:
                                    current, current_cost = candidate, value
                                scores += 64
                    else:
                        for entry in arm['trace']:
                            candidate = independently_apply(current, entry['move'])
                            value = engine.score_priority(b, a, candidate)[0]
                            assert value == entry['cost'] and entry['accepted'] == (value < current_cost)
                            if entry['accepted']:
                                current, current_cost = candidate, value
                            scores += 1
                    assert scores == arm['scores']
                    if name.startswith('count_'):
                        assert scores == protocol['count_budget']
                    assert current_cost == arm['cost'] and np.array_equal(current, arm['priority'])
                    assert replay(b, a, current)['objective'] == current_cost
                    trace_scores += scores
            best = min([row['structured'], *row['arms'].values()], key=lambda r: r['cost'])
            assert row['reference'] == {'cost': best['cost'], 'priority': best['priority']}
            assert len((cell / 'placements/placements.jsonl').read_text().splitlines()) == 3
            for arm_name, arm in [('baseline', row['baseline']), ('reference', row['reference']), ('hand', row['arms'][report['best_wall']])]:
                assert read(cell / f'{arm_name}_replay.json')['priority'] == arm['priority']
                assert (cell / f'{arm_name}_live.json').exists()
            rows.append(row)
            print('AUDIT', seed, flush=True)
        base = np.array([r['baseline']['cost'] for r in rows])
        ref = np.array([r['reference']['cost'] for r in rows])
        gain = (base - ref) / base * 100
        denominator = float(np.sum(base - ref))
        capture = {name: float(np.sum(base - np.array([r['arms'][name]['cost'] for r in rows])) / denominator) if denominator > 0 else 1.
                   for name in rows[0]['arms']}
        assert capture == report['capture'] and report['gain_pct'] == gain.tolist()
        assert report['median_headroom_pct'] == float(np.median(gain))
        assert report['headroom_pass'] == (float(np.median(gain)) >= protocol['headroom_bar_median_pct'])
        assert report['hardness_pass'] == (max(capture.values()) < protocol['hardness_bar_max_capture_fraction'])
        assert report['best_wall'] == max((n for n in capture if n.startswith('wall_')), key=lambda n: capture[n])
        assert report['best_count'] == max((n for n in capture if n.startswith('count_')), key=lambda n: capture[n])
        timing = all(r['arms'][name]['all_within_budget'] for r in rows for name in r['arms'] if name.startswith('wall_'))
        assert report['wall_timing_pass'] == timing
        expected_status = 'RESIDUAL-GATE-REQUIRED' if report['headroom_pass'] and report['hardness_pass'] and timing else 'PRETRAINING-NO-GO'
        assert report['status'] == expected_status and not report['training_allowed']
        assert report['live_runs'] == len(rows) * 3
        assert report['live_operations'] == len(rows) * 3 * np.prod(calibration['selected']['shape'][:2])
        assert live_runs == report['live_runs'] + 6
    else:
        assert report['status'] == 'S0-NO-GO' and not report['training_allowed']
    return {'status': 'PASS', 'unique_inputs': len(identities), 'structured_candidates_checked': candidates,
            'hand_search_proposals_reexecuted': trace_scores, 'live_runs': live_runs, 'live_operations': live_operations,
            'report_sha256': sha(out / 'read.json'), 'audit_source_sha256': sha(Path(__file__))}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out
    result = audit(out)
    (out / 'AUDIT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
