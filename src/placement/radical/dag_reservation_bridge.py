"""Schedule-preserving initialization and semantic translation between DAG decoders."""
import ctypes
import hashlib
import subprocess
from pathlib import Path
import numpy as np
from .dag_reservation import DagReservation,checked,hand_ranks
from .environment import initial


def chronological(rank,starts):
    order=np.lexsort((np.asarray(rank).ravel(),np.asarray(starts).ravel()))
    result=np.empty(order.size,dtype=np.int64);result[order]=np.arange(order.size)
    return result.reshape(np.shape(rank))


class BridgedDag(DagReservation):
    def __init__(self,directory):
        super().__init__(directory)
        source=Path(__file__).with_suffix('.cpp');flags=['-O3','-std=c++17','-shared','-fPIC']
        compiler=subprocess.check_output(['c++','--version'],text=True)
        sources=[source,source.with_name('dag_reservation.cpp'),source.with_name('mixed.cpp'),source.with_name('kernel.cpp')]
        key=hashlib.sha256(b''.join(p.read_bytes() for p in sources)+compiler.encode()+str(flags).encode()).hexdigest()
        library=Path(directory)/f'dag_bridge_{key[:16]}.so'
        if not library.exists():subprocess.run(['c++',*flags,str(source),'-o',str(library)],check=True,capture_output=True)
        self.bridge=ctypes.CDLL(str(library.resolve()))
        self.bridge.dag_bridge_search.argtypes=self.lib.dag_search.argtypes
        self.bridge.dag_bridge_search.restype=ctypes.c_double
        self.provenance={'parent':self.provenance,'bridge_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                         'library_sha256':hashlib.sha256(library.read_bytes()).hexdigest(),'compiler':compiler,'flags':flags}

    def search(self,b,assignment,rank,steps=8192,mode=2,temperature=.005,seed=77,seconds=0.):
        if type(steps) is not int or not 0<=steps<2**31 or mode not in (0,1,2) or not np.isfinite(temperature) or temperature<0 or not np.isfinite(seconds) or seconds<0:
            raise ValueError('invalid bridge search controls')
        if type(seed) is not int or not 0<=seed<2**64:raise ValueError('invalid seed')
        q,d,a,r,dims=checked(b,assignment,rank);decoder,count=ctypes.c_int(),ctypes.c_int64()
        cost=self.bridge.dag_bridge_search(q,d,a,r,*dims,steps,mode,temperature,seed,seconds,ctypes.byref(decoder),ctypes.byref(count))
        if not np.isfinite(cost):raise RuntimeError('bridged DAG search failed')
        return float(cost),a,r,decoder.value,count.value


def baseline(engine,b,mode=2):
    best=None
    for method in ('fastest','affinity'):
        a=initial(b,method)
        for name,rank in hand_ranks(b,a).items():
            cost,starts,_=engine.plan(b,a,rank,0)
            candidates=[] if mode==1 else [(cost,rank,0,'nondelay')]
            if mode!=0:
                mapped=chronological(rank,starts)
                mapped_cost,_,_=engine.plan(b,a,mapped,1)
                if mapped_cost>cost:raise RuntimeError('schedule-preserving initialization worsened the incumbent')
                raw_cost,_,_=engine.plan(b,a,rank,1)
                candidates.extend([(mapped_cost,mapped,1,'mapped'),(raw_cost,rank,1,'raw')])
            for value,priority,decoder,kind in candidates:
                if best is None or value<best[0]:best=(value,a.copy(),priority.copy(),decoder,f'{method}:{name}:{kind}')
    return best
