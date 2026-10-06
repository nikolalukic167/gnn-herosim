"""State-adaptive hand rules materialized as equivalent static dispatch plans."""
import ctypes,hashlib,subprocess
from pathlib import Path
import numpy as np
from .dispatch import DispatchNative
from .environment import initial

class AdaptiveDispatch(DispatchNative):
    def __init__(self,directory):
        super().__init__(directory);src=Path(__file__).with_name('dispatch_adaptive.cpp');flags=['-O3','-std=c++17','-shared','-fPIC'];compiler=subprocess.check_output(['c++','--version'],text=True)
        key=hashlib.sha256(b''.join(src.with_name(p).read_bytes() for p in ('dispatch_adaptive.cpp','dispatch.cpp','mixed.cpp','kernel.cpp'))+compiler.encode()+str(flags).encode()).hexdigest();lib=Path(directory)/f'adaptive_{key[:16]}.so'
        if not lib.exists():subprocess.run(['c++',*flags,str(src),'-o',str(lib)],check=True,capture_output=True)
        self.adaptive=ctypes.CDLL(str(lib.resolve()));f=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');a=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS');i=ctypes.c_int
        self.adaptive.adaptive_priority.argtypes=[f,a,a,i,i,i,i,i,i];self.adaptive.adaptive_priority.restype=ctypes.c_double
        self.provenance={'static':self.provenance,'adaptive_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'library_sha256':hashlib.sha256(lib.read_bytes()).hexdigest()}
    def adaptive_plan(self,b,a,mode):
        if not isinstance(mode,int) or not 0<=mode<=8:raise ValueError('unknown adaptive priority')
        q,a,dims=self.checked(b,a);r=np.empty_like(a);cost=self.adaptive.adaptive_priority(q,a,r,*dims,mode)
        if not np.isfinite(cost):raise RuntimeError('adaptive dispatch failed')
        return float(cost),r
    def plan(self,b,method):
        if not method.startswith('dyn'):return super().plan(b,method)
        rule,steps=method.split(':');_,a=self.search(b,initial(b,'affinity'),64);c,r=self.adaptive_plan(b,a,int(rule[3:]))
        if int(steps):c,r=self.search_priority(b,a,r,int(steps))
        return c,np.stack([a,r])
