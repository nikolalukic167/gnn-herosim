"""Exact two-operation perturbation portfolios, with explicit priced hand selectors."""
import ctypes
import hashlib
import subprocess
from itertools import combinations
from pathlib import Path
import numpy as np
from .mixed import MixedNative
from .environment import initial

class PairNative(MixedNative):
    def __init__(self, directory):
        super().__init__(directory)
        source=Path(__file__).with_name('pairs.cpp')
        compiler=subprocess.check_output(['c++','--version'],text=True)
        flags=['-O3','-std=c++17','-shared','-fPIC']
        sources=[source,source.with_name('mixed.cpp'),source.with_name('kernel.cpp')]
        key=hashlib.sha256(b''.join(p.read_bytes() for p in sources)+compiler.encode()+str(flags).encode()).hexdigest()
        library=Path(directory)/f'pairs_{key[:16]}.so'
        if not library.exists():
            subprocess.run(['c++',*flags,str(source),'-o',str(library)],check=True,capture_output=True)
        self.pairs_lib=ctypes.CDLL(str(library.resolve()))
        f=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        a=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
        i=ctypes.c_int
        self.pairs_lib.mixed_pairs.argtypes=[f,a,i,i,i,i,i,a,i,i,f,a]
        self.pairs_lib.mixed_pairs.restype=None
        self.provenance={'mixed':self.provenance,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                         'library_sha256':hashlib.sha256(library.read_bytes()).hexdigest()}

    def candidates(self,b,start,pairs,rounds=2):
        q,a,dims=self.checked(b,start)
        raw=np.asarray(pairs)
        if raw.ndim!=2 or raw.shape[1]!=2 or not np.issubdtype(raw.dtype,np.integer):
            raise ValueError('expected integer operation pairs')
        pairs=np.ascontiguousarray(raw,dtype=np.int64)
        if np.any(pairs<0) or np.any(pairs>=a.size) or np.any(pairs[:,0]==pairs[:,1]):
            raise ValueError('invalid operation pair')
        if not isinstance(rounds,int) or rounds<0:raise ValueError('invalid refinement rounds')
        costs=np.empty(len(pairs));plans=np.empty((len(pairs),*a.shape),dtype=np.int64)
        self.pairs_lib.mixed_pairs(q,a,*dims,pairs,len(pairs),rounds,costs,plans)
        if not np.isfinite(costs).all():raise RuntimeError('nonterminating pair schedule')
        return costs,plans

    def incumbent(self,b):
        return self.search(b,initial(b,'affinity'),64)

    def plan(self,b,method):
        if method=='anneal256':return self.search(b,initial(b,'affinity'),256,True)
        cost,a=self.incumbent(b)
        if method=='local':return cost,a
        pairs=rank_pairs(b,a,'graph' if method.startswith('graph') else 'random')
        count=int(method[5:] if method.startswith('graph') else method[6:])
        costs,plans=self.candidates(b,a,pairs[:count]);idx=int(costs.argmin())
        return (float(costs[idx]),plans[idx]) if costs[idx]<cost else (cost,a)

def rank_pairs(b,a,kind):
    pairs=np.array(list(combinations(range(a.size),2)),dtype=np.int64)
    if kind=='random':return pairs[np.random.default_rng(77).permutation(len(pairs))]
    if kind!='graph':raise ValueError('unknown pair ranking')
    locks=b['locks'].ravel();hosts=a.ravel();ops=a.shape[1]
    order=sorted(range(len(pairs)),key=lambda z:(
        -int(locks[pairs[z,0]]&locks[pairs[z,1]]).bit_count(),
        -int(pairs[z,0]//ops==pairs[z,1]//ops),
        -int(hosts[pairs[z,0]]==hosts[pairs[z,1]]),*pairs[z]))
    return pairs[order]
