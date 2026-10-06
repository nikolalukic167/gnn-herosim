"""Checked static-priority dispatch and non-neural priority search."""
import ctypes,hashlib,subprocess
from pathlib import Path
import numpy as np
from .mixed import MixedNative
from .environment import initial

def priorities(b,a,method):
    p=np.take_along_axis(b['p'],a[...,None],-1)[...,0]
    tail=np.flip(np.cumsum(np.flip(p,axis=1),axis=1),axis=1)
    bits=((b['locks'][...,None]>>np.arange(6))&1).astype(float)
    pressure=bits@bits.sum((0,1))
    if method=='spt':score=p
    elif method=='srpt':score=tail
    elif method=='longest':score=-tail
    elif method=='type':score=b['types']*1000+p
    elif method=='lock_low':score=pressure
    elif method=='lock_high':score=-pressure
    elif method=='graph_tail':score=tail/np.maximum(np.flip(np.cumsum(np.flip(pressure,axis=1),axis=1),axis=1),1)
    else:raise ValueError('unknown priority rule')
    order=np.argsort(score.ravel(),kind='stable');rank=np.empty(a.size,dtype=np.int64);rank[order]=np.arange(a.size)
    return rank.reshape(a.shape)

class DispatchNative(MixedNative):
    def __init__(self,directory):
        super().__init__(directory);source=Path(__file__).with_name('dispatch.cpp');compiler=subprocess.check_output(['c++','--version'],text=True);flags=['-O3','-std=c++17','-shared','-fPIC']
        key=hashlib.sha256(b''.join(source.with_name(p).read_bytes() for p in ('dispatch.cpp','mixed.cpp','kernel.cpp'))+compiler.encode()+str(flags).encode()).hexdigest();lib=Path(directory)/f'dispatch_{key[:16]}.so'
        if not lib.exists():subprocess.run(['c++',*flags,str(source),'-o',str(lib)],check=True,capture_output=True)
        self.dispatch_lib=ctypes.CDLL(str(lib.resolve()));f=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');a=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS');i=ctypes.c_int
        self.dispatch_lib.dispatch_score.argtypes=[f,a,a,i,i,i,i,i,f];self.dispatch_lib.dispatch_score.restype=ctypes.c_double
        self.dispatch_lib.dispatch_search.argtypes=[f,a,a,i,i,i,i,i,i,ctypes.c_uint64];self.dispatch_lib.dispatch_search.restype=ctypes.c_double
        self.provenance={'mixed':self.provenance,'dispatch_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'library_sha256':hashlib.sha256(lib.read_bytes()).hexdigest()}
    def checked_priority(self,b,a,r):
        q,a,dims=self.checked(b,a);raw=np.asarray(r)
        if raw.shape!=a.shape or not np.issubdtype(raw.dtype,np.integer) or not np.array_equal(np.sort(raw.ravel()),np.arange(a.size)):raise ValueError('priority must be an integer permutation')
        return q,a,dims,np.ascontiguousarray(raw,dtype=np.int64)
    def score_priority(self,b,a,r):
        q,a,dims,r=self.checked_priority(b,a,r);ends=np.empty(a.size);cost=self.dispatch_lib.dispatch_score(q,a,r,*dims,ends)
        if not np.isfinite(cost):raise RuntimeError('nonterminating priority schedule')
        return float(cost),ends.reshape(a.shape)
    def search_priority(self,b,a,r,steps):
        if not isinstance(steps,int) or steps<0:raise ValueError('invalid search length')
        q,a,dims,r=self.checked_priority(b,a,np.array(r,copy=True));cost=self.dispatch_lib.dispatch_search(q,a,r,*dims,steps,77)
        if not np.isfinite(cost):raise RuntimeError('nonterminating priority search')
        return float(cost),r
    def plan(self,b,method):
        rule,steps=method.split(':');_,a=self.search(b,initial(b,'affinity'),64)
        cost,r=self.search_priority(b,a,priorities(b,a,rule),int(steps))
        return cost,np.stack([a,r])
