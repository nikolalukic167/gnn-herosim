"""Opt-in earliest-start reservations, separate from frozen non-delay dispatch."""
import numpy as np
import simpy
from . import coordinator
from .dispatch_live import PriorityExecution
from scripts_cosim.mixed_physics_live import run as mixed_run


def checked(b, assignment, priority, not_before):
    a, rank, release = np.asarray(assignment), np.asarray(priority), np.asarray(not_before, dtype=float)
    shape = b['p'].shape[:2]
    if a.shape != shape or rank.shape != shape or release.shape != shape:
        raise ValueError('reservation plan shape mismatch')
    if not np.issubdtype(a.dtype, np.integer) or not np.issubdtype(rank.dtype, np.integer):
        raise ValueError('assignment and priority must be integer arrays')
    if np.any(a < 0) or np.any(a >= b['p'].shape[2]) or not np.isfinite(np.take_along_axis(b['p'], a[...,None], -1)).all():
        raise ValueError('ineligible reservation placement')
    if sorted(rank.ravel().tolist()) != list(range(rank.size)):
        raise ValueError('priority must be a permutation')
    if not np.isfinite(release).all() or np.any(release < 0):
        raise ValueError('not-before times must be finite and nonnegative')
    return a.copy(), rank.copy(), release.copy()


class ReservationExecution(PriorityExecution):
    def dispatch(self, event):
        now = float(self.env.now)
        deferred = [r for r in self.pending if self.not_before[r['rule']['job'], r['rule']['operation']] > now + 1e-12]
        self.pending = [r for r in self.pending if self.not_before[r['rule']['job'], r['rule']['operation']] <= now + 1e-12]
        super().dispatch(event)
        self.pending.extend(deferred)
        if deferred:
            at = min(self.not_before[r['rule']['job'], r['rule']['operation']] for r in deferred)
            if at not in self.reservation_timers:
                self.reservation_timers.add(at)
                def release_ready(_event, at=at):
                    self.reservation_timers.remove(at)
                    self.wake()
                timer = self.env.timeout(at - now)
                timer.callbacks.append(release_ready)

    def stats(self):
        return {**super().stats(), 'dispatch_contract':'mixed_ready_reservation_v1',
                'not_before_seconds':self.not_before.tolist()}


def herosim(b, assignment, priority, not_before, log, seed):
    a, rank, release = checked(b, assignment, priority, not_before)
    previous = coordinator.MixedExecution
    class BoundReservation(ReservationExecution):
        def __init__(self, env, config):
            super().__init__(env, config)
            self.priority, self.not_before = rank.copy(), release / 1000
            self.reservation_timers = set()
    coordinator.MixedExecution = BoundReservation
    try:
        return mixed_run(b, a, log, seed)
    finally:
        coordinator.MixedExecution = previous


def replay(b, assignment, priority, not_before):
    a, rank, release = checked(b, assignment, priority, not_before)
    jobs, ops = a.shape
    env = simpy.Environment()
    next_op = np.zeros(jobs, dtype=int)
    active, last, done, events = {}, {}, {}, []
    def process():
        while len(done) < a.size:
            now = float(env.now)
            for host, rec in list(active.items()):
                if rec['end'] <= now + 1e-9:
                    j,k = rec['job'],rec['operation']
                    done[j,k] = now
                    next_op[j] += 1
                    last[host] = int(b['types'][j,k])
                    del active[host]
                    events.append({'event':'done',**rec})
            if len(done) == a.size:
                break
            busy = {rec['job'] for rec in active.values()}
            ready = [j for j in range(jobs) if j not in busy and next_op[j] < ops]
            for j in sorted(ready,key=lambda j:(rank[j,next_op[j]],j)):
                k = next_op[j]
                host, lock = int(a[j,k]), int(b['locks'][j,k])
                if now + 1e-9 < release[j,k] or host in active:
                    continue
                if any(lock & int(b['locks'][r['job'],r['operation']]) for r in active.values()):
                    continue
                if any(b['domains'][host,g] and sum(b['domains'][h,g] for h in active) >= 2 for g in range(b['domains'].shape[1])):
                    continue
                setup = float(b['setup'][last[host],b['types'][j,k]]) if host in last else 0.
                rec = {'job':int(j),'operation':int(k),'host':host,'start':now,
                       'end':now+float(b['p'][j,k,host])+setup,'setup':setup}
                active[host] = rec
                events.append({'event':'start',**rec})
            next_times = [r['end'] for r in active.values()]
            next_times += [release[j,next_op[j]] for j in ready if release[j,next_op[j]] > now+1e-9]
            if not next_times:
                raise RuntimeError('reservation deadlock')
            yield env.timeout(min(next_times)-now)
    env.process(process())
    env.run()
    ends = np.array([[done[j,k] for k in range(ops)] for j in range(jobs)])
    return {'objective':float(ends[:,-1].sum()),'ends':ends,'events':events}
