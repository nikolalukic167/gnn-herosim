"""Checked native joint placement/order search with retained best feasible plans."""
import ctypes
import hashlib
import subprocess
from pathlib import Path
import numpy as np
from .dispatch_adaptive import AdaptiveDispatch


class JointDispatch(AdaptiveDispatch):
    def __init__(self, directory):
        super().__init__(directory)
        source = Path(__file__).with_suffix('.cpp')
        flags = ['-O3', '-std=c++17', '-shared', '-fPIC']
        compiler = subprocess.check_output(['c++', '--version'], text=True)
        sources = [source.with_name(p) for p in ('joint_dispatch.cpp', 'dispatch_adaptive.cpp', 'dispatch.cpp', 'mixed.cpp', 'kernel.cpp')]
        key = hashlib.sha256(b''.join(p.read_bytes() for p in sources) + compiler.encode() + str(flags).encode()).hexdigest()
        library = Path(directory) / f'joint_dispatch_{key[:16]}.so'
        if not library.exists():
            subprocess.run(['c++', *flags, str(source), '-o', str(library)], check=True, capture_output=True)
        self.joint = ctypes.CDLL(str(library.resolve()))
        f = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        a = np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS')
        self.joint.joint_dispatch_search.argtypes = [f, a, a, *([ctypes.c_int] * 8), ctypes.c_uint64]
        self.joint.joint_dispatch_search.restype = ctypes.c_double
        self.joint.joint_dispatch_timed.argtypes = [f, a, a, *([ctypes.c_int] * 7), ctypes.c_uint64, ctypes.c_double, ctypes.POINTER(ctypes.c_int64)]
        self.joint.joint_dispatch_timed.restype = ctypes.c_double
        self.provenance = {'dispatch': self.provenance, 'joint_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                           'library_sha256': hashlib.sha256(library.read_bytes()).hexdigest(), 'compiler': compiler, 'flags': flags}

    def joint_search(self, b, assignment, rank, steps, mode=3, anneal=True, seed=77):
        if type(steps) is not int or steps < 0 or type(mode) is not int or mode not in range(4):
            raise ValueError('invalid joint search control')
        if type(seed) is not int or not 0 <= seed < 2**64:
            raise ValueError('invalid search seed')
        q, a, dims, r = self.checked_priority(b, np.array(assignment, copy=True), np.array(rank, copy=True))
        cost = self.joint.joint_dispatch_search(q, a, r, *dims, steps, mode, int(anneal), seed)
        if not np.isfinite(cost):
            raise RuntimeError('joint search did not terminate')
        return float(cost), a, r

    def joint_timed(self, b, assignment, rank, seconds, mode=3, anneal=True, seed=77):
        if not np.isfinite(seconds) or seconds <= 0 or type(mode) is not int or mode not in range(4):
            raise ValueError('invalid joint time budget')
        if type(seed) is not int or not 0 <= seed < 2**64:
            raise ValueError('invalid search seed')
        q, a, dims, r = self.checked_priority(b, np.array(assignment, copy=True), np.array(rank, copy=True))
        count = ctypes.c_int64()
        cost = self.joint.joint_dispatch_timed(q, a, r, *dims, mode, int(anneal), seed, float(seconds), ctypes.byref(count))
        if not np.isfinite(cost):
            raise RuntimeError('joint timed search did not terminate')
        return float(cost), a, r, count.value
