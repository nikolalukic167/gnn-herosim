"""Exact bounded-region repair; immutable joint-search sources are included as-is."""
import ctypes
import hashlib
import math
import subprocess
from pathlib import Path
import numpy as np
from .joint_dispatch import JointDispatch


class RegionDispatch(JointDispatch):
    def __init__(self, directory):
        super().__init__(directory)
        source = Path(__file__).with_suffix('.cpp')
        flags = ['-O3', '-std=c++17', '-shared', '-fPIC']
        compiler = subprocess.check_output(['c++', '--version'], text=True)
        sources = [source.with_name(p) for p in ('region_dispatch.cpp', 'joint_dispatch.cpp',
                   'dispatch_adaptive.cpp', 'dispatch.cpp', 'mixed.cpp', 'kernel.cpp')]
        key = hashlib.sha256(b''.join(p.read_bytes() for p in sources) + compiler.encode() + str(flags).encode()).hexdigest()
        library = Path(directory) / f'region_dispatch_{key[:16]}.so'
        if not library.exists():
            subprocess.run(['c++', *flags, str(source), '-o', str(library)], check=True, capture_output=True)
        self.region = ctypes.CDLL(str(library.resolve()))
        f = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        a = np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS')
        self.region.region_dispatch_repair.argtypes = [f, a, a, *([ctypes.c_int] * 5), a, ctypes.c_int]
        self.region.region_dispatch_repair.restype = ctypes.c_double
        self.provenance = {'joint': self.provenance, 'region_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                           'library_sha256': hashlib.sha256(library.read_bytes()).hexdigest(), 'compiler': compiler, 'flags': flags}

    def repair(self, b, assignment, rank, region):
        nodes = np.asarray(region)
        if nodes.ndim != 1 or not np.issubdtype(nodes.dtype, np.integer):
            raise ValueError('region must contain integer operation indices')
        if not 1 <= len(nodes) <= 4 or len(set(nodes.tolist())) != len(nodes):
            raise ValueError('region must have one to four distinct operations')
        if np.any(nodes < 0) or np.any(nodes >= np.size(assignment)):
            raise ValueError('operation index out of range')
        nodes = np.ascontiguousarray(nodes, dtype=np.int64)
        q, a, dims, r = self.checked_priority(b, np.array(assignment, copy=True), np.array(rank, copy=True))
        cost = self.region.region_dispatch_repair(q, a, r, *dims, nodes, len(nodes))
        if not np.isfinite(cost):
            raise RuntimeError('region repair did not terminate')
        return float(cost), a, r


def region_pool(b, assignment, rank, limit=128):
    """Mixed precedence/resource regions with deterministic coverage and random controls."""
    jobs, ops, _ = b['p'].shape
    n = jobs * ops
    if n < 4 or type(limit) is not int or not 1 <= limit <= math.comb(n, 4):
        raise ValueError('invalid four-operation region pool size')
    locks = b['locks'].ravel()
    shared = (locks[:, None] & locks[None, :]) != 0
    same_host = assignment.ravel()[:, None] == assignment.ravel()[None, :]
    setup_type = b['types'].ravel()
    weights = 2 * shared.astype(float) + same_host + (setup_type[:, None] == setup_type[None, :]) * .25
    for j in range(jobs):
        for k in range(ops - 1):
            x = j * ops + k
            weights[x, x+1] += 2
            weights[x+1, x] += 2
    np.fill_diagonal(weights, 0)
    rng = np.random.default_rng(np.random.SeedSequence([int(b['seed']), 883]))
    candidates = set()
    for root in rng.permutation(n):
        nodes = [int(root)]
        for _ in range(3):
            scores = weights[nodes].sum(0)
            scores[nodes] = 0
            if scores.sum() == 0:
                scores[:] = 1
                scores[nodes] = 0
            nodes.append(int(rng.choice(n, p=scores / scores.sum())))
        candidates.add(tuple(sorted(nodes)))
        if len(candidates) >= limit * 3 // 4:
            break
    while len(candidates) < limit:
        candidates.add(tuple(sorted(rng.choice(n, 4, replace=False).tolist())))
    return sorted(candidates)
