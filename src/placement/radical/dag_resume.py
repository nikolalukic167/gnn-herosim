"""Reference event continuation with immutable executed and committed reservations."""
from dataclasses import dataclass
import copy

import numpy as np

from .dag_reservation import checked
from .dag_reservation_live import replay


class InfeasibleContinuation(ValueError):
    """A continuation conflicts with execution history or a fixed reservation."""


@dataclass(frozen=True)
class Snapshot:
    now: float
    assignment: np.ndarray
    rank: np.ndarray
    release: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    events: tuple
    committed: tuple
    problem: dict
    include_starts_at_cut: bool = True

    @property
    def started(self):
        return self.starts <= self.now if self.include_starts_at_cut else self.starts < self.now


def _release(value, shape):
    result = np.array(value, dtype=float, copy=True)
    if result.shape != shape or not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError('invalid reservation release times')
    return result


def snapshot(b, assignment, rank, release, at, committed=(), include_starts_at_cut=True):
    if type(include_starts_at_cut) is not bool:
        raise ValueError('include_starts_at_cut must be boolean')
    _, _, a, r, _ = checked(b, assignment, rank)
    release = _release(release, a.shape)
    now = float(at)
    if not np.isfinite(now) or now < 0:
        raise ValueError('snapshot time must be finite and nonnegative')
    ids = []
    for value in committed:
        if not isinstance(value, (int, np.integer)) or not 0 <= value < a.size:
            raise ValueError('committed operations must be valid flat integer ids')
        ids.append(int(value))
    result = replay(b, a, r, release)
    arrays = [a.copy(), r.copy(), release, result['starts'].copy(), result['ends'].copy()]
    for array in arrays:
        array.setflags(write=False)
    return Snapshot(now, *arrays, tuple(copy.deepcopy(result['events'])),
                    tuple(sorted(set(ids))), copy.deepcopy(b), include_starts_at_cut)


def resume(b, state, assignment, rank, release):
    if not isinstance(state, Snapshot):
        raise ValueError('expected a Snapshot')
    if set(b) != set(state.problem) or any(
            not np.array_equal(np.asarray(b[key]), np.asarray(state.problem[key]))
            for key in b):
        raise ValueError('snapshot problem changed')
    _, pred, a, r, _ = checked(b, assignment, rank)
    release = _release(release, a.shape)
    # Reconstruct from the saved reference to reject tampered or fabricated state.
    reference = replay(b, state.assignment, state.rank, state.release)
    if (type(state.include_starts_at_cut) is not bool or
            not np.isfinite(state.now) or state.now < 0 or
            not np.array_equal(state.starts, reference['starts']) or
            not np.array_equal(state.ends, reference['ends']) or
            list(state.events) != reference['events'] or
            any(type(i) is not int or not 0 <= i < a.size for i in state.committed)):
        raise ValueError('invalid snapshot state')
    jobs, ops = a.shape
    records = {(e['job'], e['operation']): e for e in state.events if e['event'] == 'start'}
    started = set(zip(*np.where(state.started)))
    committed = {divmod(i, ops) for i in state.committed} - started
    frozen = started | committed
    if any(a[node] != state.assignment[node] for node in frozen):
        raise InfeasibleContinuation('assignment changes an immutable operation')
    if any(release[node] > records[node]['start'] + 1e-9 for node in frozen):
        raise InfeasibleContinuation('release postpones an immutable operation')
    starts, ends = np.full(a.shape, np.nan), np.full(a.shape, np.nan)
    done, active, last = {}, {}, {}
    events = []
    for event in state.events:
        node = event['job'], event['operation']
        when = event['start'] if event['event'] == 'start' else state.ends[node]
        if when > state.now or (event['event'] == 'start' and node not in started):
            continue
        rec = {key: value for key, value in event.items() if key != 'event'}
        events.append(dict(event))
        if event['event'] == 'start':
            starts[node] = rec['start']; active[rec['host']] = rec
        else:
            ends[node] = state.ends[node]; done[node] = state.ends[node]
            last[rec['host']] = int(b['types'][node]); active.pop(rec['host'], None)
    parents = {(j, k): {(j, z) for z in range(k) if int(pred[j, k]) & (1 << z)}
               for j in range(jobs) for k in range(ops)}
    todo = set(parents) - started
    now = state.now

    def available(node):
        host, lock = int(a[node]), int(b['locks'][node])
        if host in active or not parents[node].issubset(done):
            return False
        if any(lock & int(b['locks'][rec['job'], rec['operation']]) for rec in active.values()):
            return False
        return not any(b['domains'][host, g] and
                       sum(b['domains'][h, g] for h in active) >= 2
                       for g in range(b['domains'].shape[1]))

    # History already includes completed events; an arbitrary cut is not a completion event.
    first_tick = True
    while len(done) < a.size:
        epsilon = 0. if first_tick else 1e-9
        for host, rec in list(active.items()):
            if not first_tick and rec['end'] <= now + 1e-9:
                node = rec['job'], rec['operation']
                done[node] = now; ends[node] = now
                last[host] = int(b['types'][node]); del active[host]
                events.append({'event': 'done', **rec})
        if len(done) == a.size:
            break
        due = sorted((node for node in committed & todo if records[node]['start'] <= now),
                     key=lambda node: int(r[node]))
        candidates = due + sorted(todo - committed, key=lambda node: int(r[node]))
        for node in candidates:
            fixed = node in committed
            if fixed and records[node]['start'] != now:
                raise InfeasibleContinuation('missed immutable reservation')
            if release[node] > now + epsilon or not available(node):
                if fixed:
                    raise InfeasibleContinuation('immutable reservation resource or dependency conflict')
                continue
            host = int(a[node])
            setup = float(b['setup'][last[host], b['types'][node]]) if host in last else 0.
            rec = {'job': node[0], 'operation': node[1], 'host': host, 'start': now,
                   'end': now + float(b['p'][node][host]) + setup, 'setup': setup}
            if fixed and any(rec[key] != records[node][key] for key in rec):
                raise InfeasibleContinuation('immutable reservation timing or setup changed')
            active[host] = rec; starts[node] = now; todo.remove(node)
            events.append({'event': 'start', **rec})
        next_times = [rec['end'] for rec in active.values()]
        next_times.extend(records[node]['start'] for node in committed & todo
                          if records[node]['start'] > now)
        next_times.extend(float(release[node]) for node in todo - committed
                          if release[node] > now + epsilon and parents[node].issubset(done))
        if not next_times:
            raise InfeasibleContinuation('continuation deadlock')
        now = min(next_times)
        first_tick = False
    return {'objective': float(ends[:, -1].sum()), 'starts': starts, 'ends': ends, 'events': events}


def snapshot_from_result(b, assignment, rank, release, result, at, committed=(),
                         include_starts_at_cut=True):
    """Freeze a materialized continuation using its starts as replay reservations."""
    _, _, a, r, _ = checked(b, assignment, rank)
    lower = _release(release, a.shape)
    starts = _release(result['starts'], a.shape)
    if np.any(starts < lower - 1e-9):
        raise ValueError('materialized schedule violates release times')
    reference = replay(b, a, r, starts)
    if (not np.array_equal(reference['starts'], starts) or
            not np.array_equal(reference['ends'], result['ends']) or
            reference['objective'] != result['objective']):
        raise ValueError('materialized continuation is not independently replayable')
    def event_key(event):
        return (event['job'], event['operation'], event['event'])
    if sorted(reference['events'], key=event_key) != sorted(result['events'], key=event_key):
        raise ValueError('materialized continuation event mismatch')
    return snapshot(b, a, r, starts, at, committed, include_starts_at_cut)
