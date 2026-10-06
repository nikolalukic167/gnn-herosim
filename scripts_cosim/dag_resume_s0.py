"""Bounded checkpoint-continuation screen with immutable history and live replay."""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np

from scripts_cosim.dag_block_proposer_128_s0 import strong_start
from scripts_cosim.workflow_proposal_gate import configure_environment
from src.placement.radical.dag_reservation import problem, hand_ranks
from src.placement.radical.dag_reservation_bridge import BridgedDag
from src.placement.radical.dag_reservation_live import replay, herosim
from src.placement.radical.dag_resume import (
    snapshot, snapshot_from_result, resume, InfeasibleContinuation,
)
from src.placement.radical.environment import pack, serialized

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'experiments/dag_resume_s0_v1.json'


def save(path, value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    path.write_text(json.dumps(value, default=convert, allow_nan=False, indent=2) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan(a, r, release, result):
    return {'assignment': np.array(a, copy=True), 'rank': np.array(r, copy=True),
            'release': np.array(release, copy=True), **result}


def protected(state):
    mask = state.started.copy()
    mask.ravel()[list(state.committed)] = True
    return mask


def check_history(state, candidate):
    mask = protected(state)
    for key, expected in [('assignment', state.assignment), ('starts', state.starts), ('ends', state.ends)]:
        if not np.array_equal(candidate[key][mask], expected[mask]):
            raise AssertionError(f'immutable {key} changed')
    original = {(e['job'], e['operation']): e for e in state.events if e['event'] == 'start'}
    actual = {(e['job'], e['operation']): e for e in candidate['events'] if e['event'] == 'start'}
    for node in zip(*np.where(mask)):
        if actual[node] != original[node]:
            raise AssertionError('immutable resource event changed')


class Budget:
    def __init__(self, engine, b, state, incumbent, cap, milliseconds=None):
        self.engine, self.b, self.state = engine, b, state
        self.best = incumbent
        self.cap, self.milliseconds = cap, milliseconds
        self.tick = time.perf_counter()
        self.longest_ms = 5.
        self.rows, self.feasible = [], []

    def available(self):
        elapsed = (time.perf_counter() - self.tick) * 1000
        return len(self.rows) < self.cap and (self.milliseconds is None or
               elapsed + max(15., 2.5 * self.longest_ms) < self.milliseconds)

    def evaluate(self, a, r, release, label, state=None, expected_first=None):
        if not self.available():
            return None
        state = self.state if state is None else state
        tick = time.perf_counter()
        row = {'label': label, 'assignment': a.copy(), 'rank': r.copy(),
               'release': release.copy(), 'checkpoint_time': state.now,
               'evaluation_index': len(self.rows)}
        if state is not self.state:
            row['source_state'] = {'assignment': state.assignment, 'rank': state.rank,
                                   'release': state.release, 'now': state.now,
                                   'committed': list(state.committed),
                                   'include_starts_at_cut': state.include_starts_at_cut}
        try:
            result = resume(self.b, state, a, r, release)
            candidate = plan(a, r, release, result)
            if expected_first is not None:
                mutable_starts = [e for e in result['events'] if e['event'] == 'start'
                                  and not state.started[e['job'], e['operation']]]
                first = mutable_starts[0]
                if first['job'] * a.shape[1] + first['operation'] != expected_first:
                    raise AssertionError('forced action did not start next')
            check_history(self.state, candidate)
            check_history(state, candidate)
            cost, starts, ends = self.engine.replay(self.b, a, r, result['starts'])
            if (cost != result['objective'] or not np.array_equal(starts, result['starts'])
                    or not np.array_equal(ends, result['ends'])):
                raise AssertionError('native materialized continuation mismatch')
            row.update(status='feasible', objective=cost, starts=starts, ends=ends)
            self.feasible.append(candidate)
            if cost < self.best['objective']:
                self.best = candidate
        except InfeasibleContinuation as exc:
            row.update(status='infeasible', reason=str(exc))
            candidate = None
        row['evaluation_ms'] = (time.perf_counter() - tick) * 1000
        self.longest_ms = max(self.longest_ms, row['evaluation_ms'])
        self.rows.append(row)
        return candidate

    def finish(self):
        elapsed = (time.perf_counter() - self.tick) * 1000
        return {'plan': self.best, 'evaluations': len(self.rows), 'elapsed_ms': elapsed,
                'timing_valid': self.milliseconds is None or elapsed <= self.milliseconds,
                'trace': self.rows}


def actions(b, state, incumbent):
    started = state.started
    done = state.ends <= state.now
    ready = []
    ops = started.shape[1]
    for j, k in zip(*np.where(~protected(state))):
        if all(done[j, z] for z in range(k) if int(b['predecessors'][j, k]) & (1 << z)):
            ready.append(j * ops + k)
    priority = hand_ranks(b, incumbent['assignment'])['graph2'].ravel()
    ready.sort(key=lambda i: (int(priority[i]), i))
    result = []
    active = list(zip(*np.where(started & ~done)))
    busy_hosts = {int(state.assignment[node]) for node in active}
    for i in ready:
        node = divmod(i, ops)
        if any(int(b['locks'][node]) & int(b['locks'][other]) for other in active):
            continue
        for host in np.flatnonzero(np.isfinite(b['p'].reshape(-1, b['p'].shape[-1])[i])):
            if host in busy_hosts or any(b['domains'][host, g] and
                    sum(b['domains'][h, g] for h in busy_hosts) >= 2
                    for g in range(b['domains'].shape[1])):
                continue
            result.append((i, int(host)))
    active_ends = state.ends[started & ~done]
    if len(active_ends):
        result.insert(0, ('wait', float(active_ends.min())))
    return result


def proposals(b, state, incumbent, rules):
    mask = protected(state)
    choices = actions(b, state, incumbent)
    cached = {}
    for offset in range(len(rules)):
        for index, action in enumerate(choices):
            if index not in cached:
                a = incumbent['assignment'].copy()
                release = state.release.copy()
                release[~mask] = 0.
                forced = []
                if action[0] == 'wait':
                    release[~mask] = action[1]
                else:
                    i, host = action
                    a.ravel()[i] = host
                    forced = [i]
                cached[index] = a, release, forced, hand_ranks(b, a)
            a, release, forced, ranks = cached[index]
            rule = rules[(index + offset) % len(rules)]
            rank = incumbent['rank'] if rule == 'incumbent' else ranks[rule]
            order = forced + [int(i) for i in np.argsort(rank.ravel()) if int(i) not in forced]
            r = np.empty(a.size, dtype=np.int64)
            r[order] = np.arange(a.size)
            yield a.copy(), r.reshape(a.shape), release.copy(), f'{action}:{rule}', (forced[0] if forced else None)


def lookahead(engine, b, state, incumbent, protocol, depth, reference=False):
    cap = protocol['decision_evaluation_cap']
    if reference:
        cap = protocol['reference_evaluation_cap'] if depth == 1 else protocol['retry_reference_evaluation_cap']
    budget = Budget(engine, b, state, incumbent, cap,
                    None if reference else protocol['decision_budget_ms'])
    first_cap = cap if depth == 1 else (cap // 2 if reference else protocol['retry_first_evaluation_cap'])
    for a, r, release, label, chosen in proposals(b, state, incumbent, protocol['continuation_rules']):
        if len(budget.rows) >= first_cap or not budget.available():
            break
        budget.evaluate(a, r, release, 'first:' + label, expected_first=chosen)
    first_evaluations = len(budget.rows)
    beam_count = 0
    if depth == 2:
        beam, seen = [], set()
        for selected in sorted(budget.feasible, key=lambda p: p['objective']):
            identity = tuple(selected[key].tobytes() for key in ('assignment', 'starts', 'ends'))
            if identity not in seen:
                seen.add(identity)
                beam.append(selected)
            if len(beam) >= protocol['beam_width']:
                break
        beam_count = len(beam)
        generators = []
        for selected in beam:
            if not budget.available():
                break
            future = selected['ends'][selected['ends'] > state.now]
            if not len(future):
                continue
            second = snapshot_from_result(b, selected['assignment'], selected['rank'],
                                          selected['release'], selected, float(future.min()),
                                          committed=state.committed, include_starts_at_cut=False)
            generators.append((second, proposals(b, second, selected, protocol['continuation_rules'])))
        while generators and budget.available():
            remaining = []
            for second, generator in generators:
                if not budget.available():
                    break
                item = next(generator, None)
                if item is not None:
                    a, r, release, label, chosen = item
                    budget.evaluate(a, r, release, 'second:' + label, second, expected_first=chosen)
                    remaining.append((second, generator))
            generators = remaining
    result = budget.finish()
    result.update(first_evaluations=first_evaluations,
                  second_evaluations=len(budget.rows) - first_evaluations, distinct_beam_schedules=beam_count)
    return result


def hand(engine, b, state, incumbent, protocol):
    budget = Budget(engine, b, state, incumbent, protocol['decision_evaluation_cap'],
                    protocol['decision_budget_ms'])
    mask = protected(state)
    mutable = np.flatnonzero(~mask.ravel())
    a = incumbent['assignment'].copy()
    release = state.release.copy()
    release[~mask] = 0.
    ranks = hand_ranks(b, a)
    for name, rank in ranks.items():
        if not budget.available():
            break
        budget.evaluate(a, rank, release, 'rule:' + name)
    rng = np.random.default_rng(79)
    current = budget.best
    while len(mutable) and budget.available():
        a, rank = current['assignment'].copy(), current['rank'].copy()
        width = min(len(mutable), int(rng.integers(1, 5)))
        selected = rng.choice(mutable, width, replace=False)
        for i in selected:
            if rng.random() < .5:
                hosts = np.flatnonzero(np.isfinite(b['p'].reshape(-1, b['p'].shape[-1])[i]))
                a.ravel()[i] = next(int(h) for h in hosts if h != a.ravel()[i])
            other = int(rng.choice(mutable))
            rank.ravel()[i], rank.ravel()[other] = rank.ravel()[other], rank.ravel()[i]
        candidate = budget.evaluate(a, rank, release, 'anneal')
        if candidate is not None:
            temperature = max(.001, .02 * incumbent['objective'] *
                              (1 - len(budget.rows) / budget.cap))
            difference = candidate['objective'] - current['objective']
            if difference <= 0 or rng.random() < np.exp(-difference / temperature):
                current = candidate
    return budget.finish()


def live_check(engine, b, candidate, cell, name):
    a, rank, starts = candidate['assignment'], candidate['rank'], candidate['starts']
    reference = replay(b, a, rank, starts)
    if (reference['objective'] != candidate['objective'] or
            not np.array_equal(reference['starts'], starts) or
            not np.array_equal(reference['ends'], candidate['ends'])):
        raise AssertionError('independent final plan mismatch')
    with (cell / f'{name}_live.log').open('w') as log:
        actual = herosim(b, a, rank, starts, log)
    if abs(actual['job_completion_sum'] * 1000 - candidate['objective']) > 1e-6:
        raise AssertionError('actual objective mismatch')
    seen = set()
    for task in actual['tasks']:
        j, k = map(int, task['taskType']['name'][2:].split('_'))
        if (j, k) in seen or task['executionNode'] != f'node{a[j, k]}':
            raise AssertionError('actual task identity or placement mismatch')
        seen.add((j, k))
        if abs(task['doneTime'] * 1000 - candidate['ends'][j, k]) > 1e-6:
            raise AssertionError('actual completion mismatch')
    expected = {(e['job'], e['operation'], e['event']): e for e in candidate['events']}
    actual_events = actual['mixed_execution']['events']
    if len(actual_events) != len(expected) or len(seen) != a.size:
        raise AssertionError('incomplete actual resource events')
    event_keys = set()
    for event in actual_events:
        key = event['job'], event['operation'], event['event']
        if key in event_keys:
            raise AssertionError('duplicate actual resource event')
        event_keys.add(key)
        exp = expected[key]
        when = exp['start'] if event['event'] == 'start' else exp['end']
        if (event['host'] != f'node{exp["host"]}' or
                abs(event['time'] * 1000 - when) > 1e-6 or
                abs(event['setup'] * 1000 - exp['setup']) > 1e-6):
            raise AssertionError('actual resource event mismatch')
    save(cell / f'{name}_live.json', actual)


def case(engine, seed, protocol, depth, cell, serve):
    cell.mkdir(parents=True, exist_ok=False)
    b = problem(seed, *protocol['shape'])
    save(cell / 'input.json', serialized(b))
    identity = hashlib.sha256(pack(b).tobytes() + b['predecessors'].tobytes()).hexdigest()
    initial = strong_start(engine, b, protocol)
    a, rank = np.array(initial['assignment']), np.array(initial['priority'])
    starts = np.array(initial['starts'])
    original = replay(b, a, rank, starts)
    at = float(np.sort(original['ends'].ravel())[int(a.size * protocol['checkpoint_completed_fraction']) - 1])
    tick = time.perf_counter()
    state = snapshot(b, a, rank, starts, at, include_starts_at_cut=False)
    incumbent = plan(a, rank, starts, resume(b, state, a, rank, starts))
    snapshot_ms = (time.perf_counter() - tick) * 1000
    if incumbent['objective'] != original['objective']:
        raise AssertionError('unchanged continuation changed cost')
    control = hand(engine, b, state, incumbent, protocol)
    proposed = lookahead(engine, b, state, incumbent, protocol, depth)
    reference = lookahead(engine, b, state, incumbent, protocol, depth, True)
    reference['plan'] = min((reference['plan'], control['plan'], proposed['plan']),
                            key=lambda p: p['objective'])
    arms = {'incumbent': incumbent, 'hand': control['plan'],
            'candidate': proposed['plan'], 'reference': reference['plan']}
    row = {'seed': seed, 'identity': identity, 'depth': depth, 'checkpoint_time': at,
           'started': int(np.sum(state.started)), 'completed': int(np.sum(state.ends <= at)),
           'common_start_ms': initial['time_ms'], 'snapshot_ms': snapshot_ms,
           'arms': arms, 'search': {'hand': control, 'candidate': proposed, 'reference': reference}}
    (cell / 'placements').mkdir()
    for name, reading in row['search'].items():
        with (cell / 'placements' / f'{name}.jsonl').open('w') as handle:
            for record in reading.pop('trace'):
                handle.write(json.dumps(record, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item()) + '\n')
    save(cell / 'read.json', row)
    if serve:
        for name, candidate in arms.items():
            live_check(engine, b, candidate, cell, name)
    return row


def summary(rows, protocol):
    gains = lambda arm: [100 * (r['arms']['hand']['objective'] - r['arms'][arm]['objective']) /
                         r['arms']['hand']['objective'] for r in rows]
    candidate, reference = gains('candidate'), gains('reference')
    timing_valid = all(r['search'][name]['timing_valid'] for r in rows for name in ('hand', 'candidate'))
    advance = (timing_valid and np.median(candidate) >= protocol['candidate_gain_bar_pct'] and
               np.median(reference) >= protocol['reference_headroom_bar_pct'])
    return {'cases': len(rows), 'candidate_gain_pct': candidate, 'reference_gain_pct': reference,
            'median_gain_over_incumbent_pct': {arm: float(np.median([
                100 * (r['arms']['incumbent']['objective'] - r['arms'][arm]['objective']) /
                r['arms']['incumbent']['objective'] for r in rows]))
                for arm in ('hand', 'candidate', 'reference')},
            'median_candidate_gain_pct': float(np.median(candidate)),
            'median_reference_gain_pct': float(np.median(reference)), 'timing_valid': timing_valid,
            'candidate_wins': sum(g > 0 for g in candidate), 'advance': bool(advance),
            'live_runs': len(rows) * 4, 'live_operations': len(rows) * 4 * np.prod(protocol['shape'][:2]).item()}


def run(out, phase):
    out.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(PROTOCOL.read_text())
    paths = [PROTOCOL, Path(__file__).resolve()]
    paths += list((ROOT / 'src/placement/radical').glob('*.py'))
    paths += list((ROOT / 'src/placement/radical').glob('*.cpp'))
    paths += [ROOT / 'scripts_cosim/dag_block_proposer_128_s0.py',
              ROOT / 'scripts_cosim/workflow_live.py', ROOT / 'src/placement/infrastructure.py']
    sources = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    save(out / 'protocol_before_run.json', {'protocol': protocol, 'phase': phase, 'sources': sources})
    for name in sources:
        target = out / 'source_snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    engine = BridgedDag(out / 'build')
    save(out / 'native.json', engine.provenance)
    configure_environment()
    rows = []
    if phase == 'calibration':
        for seed in protocol['calibration_seeds']:
            row = case(engine, seed, protocol, 2, out / str(seed), True)
            rows.append(row)
            print('CALIBRATION', seed, {k: round(v['elapsed_ms'], 2) for k, v in row['search'].items()}, flush=True)
        report = summary(rows, protocol)
    else:
        depth = 1 if phase == 'primary' else 2
        seeds = protocol['primary_seeds'] if phase == 'primary' else protocol['retry_seeds']
        for seed in range(seeds[0], seeds[1] + 1):
            row = case(engine, seed, protocol, depth, out / str(seed), True)
            rows.append(row)
            print(phase.upper(), seed, {k: v['objective'] for k, v in row['arms'].items()}, flush=True)
        report = summary(rows, protocol)
    if len({r['identity'] for r in rows}) != len(rows):
        raise AssertionError('duplicate physical workflows')
    for name, old in sources.items():
        if digest(ROOT / name) != old:
            raise AssertionError('source changed during run: ' + name)
    report.update(phase=phase, training_allowed=False, sources_unchanged=True)
    save(out / 'report.json', report)
    save(out / 'artifacts.json', {str(path.relative_to(out)): digest(path)
         for path in sorted(out.rglob('*')) if path.is_file() and path.suffix in ('.json', '.jsonl')})
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--phase', choices=('calibration', 'primary', 'retry'), required=True)
    args = parser.parse_args()
    run(args.out, args.phase)
