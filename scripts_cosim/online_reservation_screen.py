"""Small online reservation feasibility screen with hidden background arrivals.

This is an event-level diagnostic, not a learned policy or a HeROsim callback.
The incumbent and the lookahead arm see exactly the jobs that have arrived.
"""
import argparse
import copy
import io
import json
import time
from pathlib import Path

import numpy as np

from src.placement.radical.dag_reservation import problem


RULES = ('short_job', 'critical_path', 'setup_aware', 'lock_pressure')


def initial(b):
    jobs, ops, hosts = b['p'].shape
    return {'now': 0., 'started': np.zeros((jobs, ops), bool),
            'done': np.zeros((jobs, ops), bool), 'starts': np.full((jobs, ops), np.nan),
            'ends': np.full((jobs, ops), np.nan), 'assignment': np.full((jobs, ops), -1, int),
            'active': {}, 'last': {}, 'events': []}


def visible(state, arrivals):
    return arrivals <= state['now'] + 1e-9


def actions(b, state, arrivals):
    jobs, ops, hosts = b['p'].shape
    present = visible(state, arrivals)
    result = []
    for j in range(jobs):
        if not present[j]:
            continue
        for k in range(ops):
            if state['started'][j, k]:
                continue
            pred = int(b['predecessors'][j, k])
            if any(pred & (1 << z) and not state['done'][j, z] for z in range(k)):
                continue
            for h in np.flatnonzero(np.isfinite(b['p'][j, k])):
                h = int(h)
                if h in state['active']:
                    continue
                if any(int(b['locks'][j, k]) & int(b['locks'][r[0], r[1]])
                       for r in state['active'].values()):
                    continue
                if any(b['domains'][h, g] and
                       sum(b['domains'][busy, g] for busy in state['active']) >= 2
                       for g in range(b['domains'].shape[1])):
                    continue
                result.append((j, k, h))
    if state['active'] or np.any((arrivals > state['now'] + 1e-9) & np.isfinite(arrivals)):
        result.append(None)
    return result


def advance(b, state, arrivals):
    future = [r[2] for r in state['active'].values()]
    future += [float(x) for x in arrivals if x > state['now'] + 1e-9 and np.isfinite(x)]
    if not future:
        raise RuntimeError('online execution deadlocked')
    state['now'] = min(future)
    for h, (j, k, end, setup) in list(state['active'].items()):
        if end <= state['now'] + 1e-9:
            state['done'][j, k] = True
            state['ends'][j, k] = state['now']
            state['last'][h] = int(b['types'][j, k])
            del state['active'][h]
            state['events'].append({'event': 'done', 'job': j, 'operation': k,
                                    'host': h, 'time': state['now'], 'setup': setup})


def apply(b, state, action, arrivals):
    if action is None:
        advance(b, state, arrivals)
        return
    j, k, h = action
    if action not in actions(b, state, arrivals):
        raise ValueError('infeasible online action')
    setup = float(b['setup'][state['last'][h], b['types'][j, k]]) if h in state['last'] else 0.
    end = state['now'] + float(b['p'][j, k, h]) + setup
    state['started'][j, k] = True
    state['starts'][j, k] = state['now']
    state['assignment'][j, k] = h
    state['active'][h] = (j, k, end, setup)
    state['events'].append({'event': 'start', 'job': j, 'operation': k,
                            'host': h, 'time': state['now'], 'setup': setup})


def scores(b, state, arrivals, choices):
    jobs, ops, _ = b['p'].shape
    result = {rule: [] for rule in RULES}
    for action in choices:
        if action is None:
            for rule in RULES:
                result[rule].append((float('inf'), action))
            continue
        j, k, h = action
        duration = float(b['p'][j, k, h])
        setup = float(b['setup'][state['last'][h], b['types'][j, k]]) if h in state['last'] else 0.
        remaining = sum(np.nanmin(b['p'][j, z]) for z in range(ops) if not state['done'][j, z])
        tail = duration
        for z in range(k + 1, ops):
            if not state['done'][j, z]:
                tail += float(np.nanmin(b['p'][j, z]))
        pressure = sum(bool(int(b['locks'][j, k]) & int(mask))
                       for mask in b['locks'][np.isfinite(arrivals)].ravel())
        for rule, value in [('short_job', remaining + .001 * duration),
                            ('critical_path', -tail), ('setup_aware', duration + setup),
                            ('lock_pressure', -pressure + .001 * duration)]:
            result[rule].append((value, action))
    return result


def greedy(b, state, arrivals, rule):
    choices = actions(b, state, arrivals)
    actual = [a for a in choices if a is not None]
    if not actual:
        return None
    return min(scores(b, state, arrivals, actual)[rule], key=lambda row: (row[0], row[1]))[1]


def rollout(b, source, arrivals, rule, forced=()):
    state = copy.deepcopy(source)
    forced = list(forced)
    present = np.isfinite(arrivals)
    while not state['done'][present].all():
        action = forced.pop(0) if forced else greedy(b, state, arrivals, rule)
        apply(b, state, action, arrivals)
    return float(state['ends'][present, -1].sum()), state


def choose(b, state, arrivals, mode, cap=32):
    if mode == 'hand':
        return greedy(b, state, arrivals, 'short_job'), 0
    # A hand portfolio receives the same exact simulator and evaluation cap.
    candidates = actions(b, state, arrivals)
    rows = []
    if mode == 'beam2v2':
        actual = [action for action in candidates if action is not None]
        ranked = scores(b, state, arrivals, actual)
        shortlist = []
        for rule in RULES:
            action = min(ranked[rule], key=lambda row: (row[0], row[1]))[1] if actual else None
            if action is not None and action not in shortlist:
                shortlist.append(action)
        if None in candidates:
            shortlist = shortlist[:3] + [None]
        for _, action in sorted(ranked['short_job'], key=lambda row: (row[0], row[1])):
            if len(shortlist) >= 4:
                break
            if action not in shortlist:
                shortlist.append(action)
        candidates = shortlist[:4]
    first_limit = cap // 2 if mode in ('beam2', 'beam2v2') else cap
    for rule in RULES:
        for action in candidates:
            if len(rows) >= first_limit:
                break
            value, _ = rollout(b, state, arrivals, rule, (action,))
            rows.append((value, action, rule))
        if len(rows) >= first_limit:
            break
    winner = min(rows, key=lambda row: (row[0], row[1] is None, str(row[1]), row[2]))
    if mode == 'portfolio':
        return winner[1], len(rows)
    if mode not in ('beam2', 'beam2v2'):
        raise ValueError(mode)
    firsts = sorted({row[1] for row in rows}, key=lambda action: (
        min(row[0] for row in rows if row[1] == action), action is None, str(action)))[:4]
    deeper = []
    for first in firsts:
        if len(rows) + len(deeper) >= cap:
            break
        after = copy.deepcopy(state)
        apply(b, after, first, arrivals)
        if after['done'][np.isfinite(arrivals)].all():
            continue
        for second in actions(b, after, arrivals):
            for rule in RULES:
                if len(rows) + len(deeper) >= cap:
                    break
                value, _ = rollout(b, after, arrivals, rule, (second,))
                deeper.append((value, first, rule))
            if len(rows) + len(deeper) >= cap:
                break
    selected = min(rows + deeper, key=lambda row: (row[0], row[1] is None, str(row[1]), row[2]))
    return selected[1], len(rows) + len(deeper)


def execute(b, arrivals, mode):
    state = initial(b)
    evaluations = 0
    started = time.perf_counter()
    while not state['done'].all():
        present = visible(state, arrivals)
        projection = np.where(present, arrivals, np.inf)
        if not np.any(present) or state['done'][present].all():
            advance(b, state, arrivals)
            continue
        action, count = choose(b, state, projection, mode)
        evaluations += count
        if action is None:
            advance(b, state, arrivals)
        else:
            apply(b, state, action, arrivals)
    return {'objective': float(state['ends'][:, -1].sum()), 'state': state,
            'evaluations': evaluations, 'planning_ms': (time.perf_counter() - started) * 1000}


def audit_live(b, arrivals, result):
    from scripts_cosim.workflow_proposal_gate import configure_environment
    from src.placement.radical.dag_reservation_live import herosim
    state = result['state']
    rank = np.empty(state['assignment'].shape, dtype=np.int64)
    starts = [e for e in state['events'] if e['event'] == 'start']
    for index, event in enumerate(starts):
        rank[event['job'], event['operation']] = index
    configure_environment()
    actual = herosim(b, state['assignment'], rank, state['starts'],
                     io.StringIO(), arrivals=arrivals)
    assert len(actual['tasks']) == state['started'].size
    assert abs(actual['job_completion_sum'] * 1000 - result['objective']) < 1e-6
    for task in actual['tasks']:
        j, k = map(int, task['taskType']['name'][2:].split('_'))
        assert abs(task['doneTime'] * 1000 - state['ends'][j, k]) < 1e-6
        assert task['executionNode'] == f"node{state['assignment'][j, k]}"
    live_events = actual['mixed_execution']['events']
    assert len(live_events) == 2 * state['started'].size
    expected = {(e['event'], e['job'], e['operation']): e for e in state['events']}
    for event in live_events:
        key = event['event'], event['job'], event['operation']
        ref = expected[key]
        assert event['host'] == f"node{ref['host']}"
        assert abs(event['time'] * 1000 - ref['time']) < 1e-6
        assert abs(event['setup'] * 1000 - ref['setup']) < 1e-6
    return {'tasks': len(actual['tasks']), 'events': len(live_events)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=274000)
    parser.add_argument('--count', type=int, default=16)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--retry', action='store_true')
    args = parser.parse_args()
    rows = []
    for seed in range(args.start, args.start + args.count):
        b = problem(seed, jobs=4, ops=4, hosts=3)
        arrivals = np.array([0., 0., 0., 4. + (seed % 9)], dtype=float)
        hand = execute(b, arrivals, 'hand')
        portfolio = execute(b, arrivals, 'portfolio')
        candidate_name = 'beam2v2' if args.retry else 'beam2'
        beam = execute(b, arrivals, candidate_name)
        live = {mode: audit_live(b, arrivals, value) for mode, value in
                [('hand', hand), ('portfolio', portfolio), (candidate_name, beam)]} if args.live else None
        rows.append({'seed': seed, 'arrivals': arrivals.tolist(),
                     'hand': hand['objective'], 'portfolio': portfolio['objective'],
                     'candidate': candidate_name, 'candidate_cost': beam['objective'],
                     'portfolio_evaluations': portfolio['evaluations'],
                     'candidate_evaluations': beam['evaluations'],
                     'portfolio_planning_ms': portfolio['planning_ms'],
                     'candidate_planning_ms': beam['planning_ms'],
                     'live_audit': live,
                     'portfolio_gain_pct': 100 * (hand['objective'] - portfolio['objective']) / hand['objective'],
                     'gain_pct': 100 * (portfolio['objective'] - beam['objective']) / portfolio['objective']})
    result = {'scope': 'online event-level feasibility only; no GNN or live callback',
              'rows': rows, 'median_gain_pct': float(np.median([r['gain_pct'] for r in rows]))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'count': len(rows), 'median_gain_pct': result['median_gain_pct'],
                      'positive': sum(r['gain_pct'] > 0 for r in rows)}, indent=2))


if __name__ == '__main__':
    main()
