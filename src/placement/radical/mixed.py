"""Specialized, bounded and checked control for setup/atomic-lock/domain physics."""
import ctypes,hashlib,subprocess
from pathlib import Path
import numpy as np
from .environment import Native

class MixedNative(Native):
    def __init__(self,directory):
        super().__init__(directory);source=Path(__file__).with_name('mixed.cpp');base=source.with_name('kernel.cpp');compiler=subprocess.check_output(['c++','--version'],text=True);flags=['-O3','-std=c++17','-shared','-fPIC'];sha=lambda x:hashlib.sha256(x).hexdigest();key=sha(source.read_bytes()+base.read_bytes()+compiler.encode()+str(flags).encode());library=Path(directory)/f'mixed_{key[:16]}.so'
        if not library.exists():subprocess.run(['c++',*flags,str(source),'-o',str(library)],check=True,capture_output=True)
        self.mixed=ctypes.CDLL(str(library.resolve()));f=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');a=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS');i=ctypes.c_int
        self.mixed.mixed_score.argtypes=[f,a,i,i,i,i,i,f];self.mixed.mixed_score.restype=ctypes.c_double
        self.mixed.mixed_search.argtypes=[f,a,i,i,i,i,i,i,i,ctypes.c_uint64];self.mixed.mixed_search.restype=ctypes.c_double
        self.provenance={'generic':self.provenance,'source_sha256':sha(source.read_bytes()),'library_sha256':sha(library.read_bytes()),'compiler':compiler,'flags':flags}
    def checked(self,b,a):
        p=np.asarray(b['p']);shape=p.shape
        if p.ndim!=3:raise ValueError('expected jobs, operations, hosts')
        j,o,h=shape;g=np.asarray(b['domains']).shape[-1];k=np.asarray(b['setup']).shape[0]
        if not (1<=j<=64 and 1<=o and j*o<=4096 and 2<=h<=32 and 1<=g<=32 and k>=1):raise ValueError('unsupported native dimensions')
        expected={'types':(j,o),'locks':(j,o),'domains':(h,g),'thermal':(h,h),'maintenance':(h,2),'deadlines':(j,),'power':(j,o),'setup':(k,k)}
        for key,s in expected.items():
            v=np.asarray(b[key])
            if v.shape!=s or not np.isfinite(v).all():raise ValueError('invalid '+key)
        for key,hi in (('types',k),('locks',1<<30)):
            v=np.asarray(b[key])
            if np.any(v<0) or np.any(v>=hi) or np.any(v!=np.floor(v)):raise ValueError('invalid '+key)
        if not np.isin(b['domains'],[0,1]).all() or np.any(b['setup']<0):raise ValueError('invalid resource contract')
        if not np.array_equal(a,np.asarray(a,dtype=np.int64)):raise ValueError('noninteger assignment')
        return super().checked(b,a)
    def score(self,b,a,mechanism='mixed'):
        if mechanism!='mixed':raise ValueError('mixed physics only')
        q,a,dims=self.checked(b,a);ends=np.empty(a.size);c=self.mixed.mixed_score(q,a,*dims,ends)
        if not np.isfinite(c):raise RuntimeError('nonterminating mixed schedule')
        return float(c),ends.reshape(a.shape)
    def search(self,b,a,steps,anneal=False,seed=77):
        if not isinstance(steps,int) or steps<0:raise ValueError('invalid step count')
        q,a,dims=self.checked(b,np.array(a,copy=True));c=self.mixed.mixed_search(q,a,*dims,steps,int(anneal),seed)
        if not np.isfinite(c):raise RuntimeError('nonterminating mixed search')
        return float(c),a
