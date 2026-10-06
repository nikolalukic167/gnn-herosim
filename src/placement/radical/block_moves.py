"""Capacity-scaled inputs and exact-accepted block proposals; historical sources untouched."""
from dataclasses import dataclass
import numpy as np
from .environment import problem, initial
from .dispatch import priorities
from .dispatch_live import replay

CAPACITY_CONTRACT = 'locks_equal_hosts_v1'


def scaled_problem(seed, jobs=8, ops=6, hosts=4):
    if not 2 <= hosts <= 28:
        raise ValueError('capacity contract needs 2..28 native-safe lock bits')
    b = problem(seed, jobs, ops, hosts)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 991]))
    locks = np.zeros((jobs, ops), dtype=np.int64)
    for j in range(jobs):
        for k in range(ops):
            for bit in rng.choice(hosts, int(rng.integers(1, 3)), replace=False):
                locks[j, k] |= 1 << int(bit)
    b['locks'] = locks
    b['domains'] = np.eye((hosts + 3) // 4, dtype=float)[np.arange(hosts) // 4]
    b['capacity_contract'] = CAPACITY_CONTRACT
    return b


def baseline(engine, b):
    _, a = engine.search(b, initial(b, 'affinity'), 1)
    c, rank = engine.search_priority(b, a, priorities(b, a, 'srpt'), 128)
    return c, a, rank


def graph_features(b, a):
    jobs, ops, hosts = b['p'].shape
    n = jobs * ops
    p = np.take_along_axis(b['p'], a[..., None], -1)[..., 0]
    tail = np.flip(np.cumsum(np.flip(p, 1), 1), 1)
    locks = ((b['locks'].reshape(-1, 1) >> np.arange(hosts)) & 1).astype(np.float32)
    types = np.eye(4, dtype=np.float32)[b['types'].ravel()]
    host = np.eye(hosts, dtype=np.float32)[a.ravel()]
    domain = host @ b['domains']
    counts = locks.sum(0)
    pressure = locks @ counts
    x = np.column_stack([p.ravel() / 20, tail.ravel() / (20 * ops),
                         np.tile(np.arange(ops) / ops, jobs), pressure / n,
                         types, locks, host, domain,
                         np.broadcast_to(counts / n, (n, hosts)),
                         np.broadcast_to(b['setup'].ravel() / 20, (n, 16))]).astype(np.float32)
    adj = np.zeros((4, n, n), dtype=np.float32)
    for j in range(jobs):
        for k in range(ops - 1):
            i = j * ops + k
            adj[0, i + 1, i] = 1
            adj[1, i, i + 1] = 1
    adj[2] = locks @ locks.T
    adj[3] = (host @ host.T + domain @ domain.T).astype(np.float32)
    for relation in adj:
        np.fill_diagonal(relation, 0)
    adj /= np.maximum(adj.sum(-1, keepdims=True), 1)
    one = adj @ x
    two = adj @ one
    four = adj @ (adj @ two)
    hand = np.concatenate([x, *one, *two, *four], axis=-1)
    return x, adj, hand, {'tail': tail.ravel(), 'pressure': pressure, 'one': one, 'two': two, 'four': four}


@dataclass(frozen=True)
class Move:
    family: str
    block: tuple
    anchor: int
    other: tuple = ()

    def record(self):
        return {'family': self.family, 'block': list(self.block), 'anchor': self.anchor, 'other': list(self.other)}


def apply_move(rank, move):
    order = np.argsort(rank.ravel(), kind='stable').tolist()
    block = set(move.block)
    if not block or len(block) != len(move.block) or not block.issubset(range(rank.size)):
        raise ValueError('invalid block')
    if move.other:
        other = set(move.other)
        if block & other or len(other) != len(move.other) or not other.issubset(range(rank.size)):
            raise ValueError('invalid second block')
        first, second = min(order.index(i) for i in block), min(order.index(i) for i in other)
        out = []
        for index, node in enumerate(order):
            if index == first:
                out.extend(move.other)
            if index == second:
                out.extend(move.block)
            if node not in block and node not in other:
                out.append(node)
    else:
        if move.anchor in block or move.anchor not in order:
            raise ValueError('invalid block anchor')
        out = [i for i in order if i not in block]
        at = out.index(move.anchor)
        out[at:at] = move.block
    result = np.empty(rank.size, dtype=np.int64)
    result[np.asarray(out)] = np.arange(rank.size)
    return result.reshape(rank.shape)


def _windows(sequence, lengths):
    for length in lengths:
        for start in range(len(sequence) - length + 1):
            yield tuple(sequence[start:start + length])


def move_pool(b, a, rank, per_family=1024, lengths=(2, 4, 8)):
    ref = replay(b, a, rank)
    jobs, ops, hosts = b['p'].shape
    starts = [e for e in ref['events'] if e['event'] == 'start']
    starts.sort(key=lambda e: (e['start'], int(rank[e['job'], e['operation']])))
    sequence = [e['job'] * ops + e['operation'] for e in starts]
    lookup = {e['job'] * ops + e['operation']: e for e in starts}
    # A tight realized predecessor chain can cross jobs through a host or lock.
    chains = []
    for job in range(jobs):
        chain, node = [], job * ops + ops - 1
        while node not in chain:
            chain.append(node)
            event = lookup[node]
            predecessors = [i for i in sequence if lookup[i]['end'] == event['start'] and
                            (i == node - 1 and node % ops or a.flat[i] == a.flat[node] or
                             int(b['locks'].flat[i]) & int(b['locks'].flat[node]))]
            if not predecessors:
                break
            node = min(predecessors)
        chains.append(list(reversed(chain)))
        chains.append(list(range(job * ops, (job + 1) * ops)))
    groups = {
        'critical_chain': chains,
        'lock_chain': [[i for i in sequence if int(b['locks'].flat[i]) & (1 << bit)] for bit in range(hosts)],
        'setup_class': [[i for i in sequence if a.flat[i] == h and b['types'].flat[i] == t]
                        for h in range(hosts) for t in range(4)],
    }
    order = np.argsort(rank.ravel(), kind='stable').tolist()
    rng = np.random.default_rng(np.random.SeedSequence([int(b['seed']), 431, int(rank.ravel()[:8].sum())]))
    result = []
    for family, sequences in groups.items():
        blocks = sorted(set(block for group in sequences for block in _windows(group, lengths)))
        if not blocks:
            continue
        seen = set()
        for _ in range(per_family * 12):
            block = blocks[int(rng.integers(len(blocks)))]
            anchor = order[int(rng.integers(len(order)))]
            if anchor in block:
                continue
            move = Move(family, block, anchor)
            if move in seen:
                continue
            seen.add(move)
            result.append(move)
            if len(seen) == per_family:
                break
    seen = set()
    for _ in range(per_family * 12):
        length = int(rng.choice(lengths))
        if 2 * length > len(order):
            continue
        x, y = sorted(rng.choice(len(order) - length + 1, 2, replace=False).tolist())
        if x + length > y:
            continue
        move = Move('block_swap', tuple(order[x:x + length]), -1, tuple(order[y:y + length]))
        if move in seen:
            continue
        seen.add(move)
        result.append(move)
        if len(seen) == per_family:
            break
    return result, ref


def order_moves(b, a, rank, pool, method, context):
    if method == 'random':
        return np.random.default_rng(int(b['seed']) + 117).permutation(len(pool)).tolist()
    tail, pressure = context['tail'], context['pressure']
    utility = 1 / np.maximum(tail, 1)
    if method == 'lock_low':
        utility /= 1 + pressure
    elif method == 'lock_high':
        utility *= 1 + pressure
    elif method.startswith('hop'):
        hop = context[{'hop1': 'one', 'hop2': 'two', 'hop4': 'four'}[method]]
        utility = utility * (1 + hop[:, :, 3].sum(0)) / (1 + hop[:, :, 1].sum(0))
    elif method not in ('critical', 'setup'):
        raise ValueError('unknown proposer')
    def score(move):
        block = np.array(move.block)
        anchor_rank = min(rank.flat[i] for i in move.other) if move.other else rank.flat[move.anchor]
        displacement = float(np.mean(rank.ravel()[block]) - anchor_rank)
        gain = displacement * float(np.mean(utility[block]))
        if method == 'setup':
            types = b['types'].ravel()[block]
            saving = float(np.mean(b['setup'][:, types])) * max(0, len(block) - 1)
            gain += saving if move.family == 'setup_class' else 0
        return gain
    return sorted(range(len(pool)), key=lambda i: (-score(pool[i]), i))
