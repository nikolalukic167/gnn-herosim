"""Deterministic multi-operation proposals for irregular DAG reservation schedules."""
from dataclasses import dataclass

import numpy as np

from .block_moves import Move, apply_move


@dataclass(frozen=True)
class Proposal:
    family: str
    operations: tuple[int, ...]
    anchor: int
    decoder: int
    flip_hosts: bool = False


def candidate_pool(problem, assignment, rank, starts, limit_per_family=96):
    jobs, ops, hosts = problem['p'].shape
    flat_start = np.asarray(starts).ravel()
    order = np.lexsort((rank.ravel(), flat_start)).tolist()
    predecessors = problem['predecessors']
    locks = problem['locks'].ravel()
    types = problem['types'].ravel()
    placed = assignment.ravel()
    groups = {'critical_chain': [], 'lock_chain': [], 'setup_class': [], 'block_swap': []}
    durations = np.take_along_axis(problem['p'], assignment[..., None], axis=-1)[..., 0]
    tail = durations.copy()
    for job in range(jobs):
        for op in range(ops - 1, -1, -1):
            for child in range(op + 1, ops):
                if int(predecessors[job, child]) & (1 << op):
                    tail[job, op] = max(tail[job, op], durations[job, op] + tail[job, child])
        node = 0
        chain = [job * ops]
        while node != ops - 1:
            children = [child for child in range(node + 1, ops)
                        if int(predecessors[job, child]) & (1 << node)]
            if not children:
                break
            node = max(children, key=lambda child: (tail[job, child], -child))
            chain.append(job * ops + node)
        groups['critical_chain'].append(chain)
    for bit in range(hosts):
        groups['lock_chain'].append([i for i in order if int(locks[i]) & (1 << bit)])
    for host in range(hosts):
        for kind in range(problem['setup'].shape[0]):
            groups['setup_class'].append([i for i in order if placed[i] == host and types[i] == kind])

    result = []
    for family, sequences in groups.items():
        if family == 'block_swap':
            continue
        seen = set()
        for sequence in sequences:
            for length in (2, 4, 8):
                for start in range(0, len(sequence) - length + 1):
                    block = tuple(sequence[start:start + length])
                    for anchor in (order[0], order[len(order) // 4], order[len(order) // 2], order[-1]):
                        if anchor in block:
                            continue
                        proposal = Proposal(family, block, anchor, 1)
                        if proposal not in seen:
                            seen.add(proposal)
                            result.append(proposal)
        family_proposals = [item for item in result if item.family == family]
        if len(family_proposals) > limit_per_family:
            chosen = np.linspace(0, len(family_proposals) - 1, limit_per_family, dtype=int)
            result = [item for item in result if item.family != family]
            result.extend(family_proposals[i] for i in chosen)
        if family == 'critical_chain':
            selected = [item for item in result if item.family == family]
            result.extend(Proposal(item.family, item.operations, item.anchor, item.decoder, True)
                          for item in selected[::3])
    seen = set()
    for length in (2, 4, 8):
        stride = max(1, length // 2)
        for left in range(0, len(order) - 2 * length + 1, stride):
            right = min(len(order) - length, left + 2 * length)
            if left + length > right:
                continue
            block = tuple(order[left:left + length])
            other = tuple(order[right:right + length])
            proposal = Proposal('block_swap', block + other, -1, 1)
            if proposal not in seen:
                seen.add(proposal)
                result.append(proposal)
            if len(seen) >= limit_per_family:
                break
        if len(seen) >= limit_per_family:
            break
    return result


def apply_proposal(problem, assignment, rank, proposal):
    new_assignment = assignment.copy()
    if proposal.family == 'block_swap':
        half = len(proposal.operations) // 2
        new_rank = apply_move(rank, Move('block_swap', proposal.operations[:half], -1,
                                         proposal.operations[half:]))
    else:
        new_rank = apply_move(rank, Move(proposal.family, proposal.operations, proposal.anchor))
    if proposal.flip_hosts:
        p = problem['p'].reshape(-1, problem['p'].shape[-1])
        flat = new_assignment.ravel()
        for operation in proposal.operations:
            choices = np.flatnonzero(np.isfinite(p[operation]))
            flat[operation] = int(choices[0] if choices[1] == flat[operation] else choices[1])
    return new_assignment, new_rank
