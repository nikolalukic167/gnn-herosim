"""Checked Python interface to exact native FIFO single-move evaluation."""
import ctypes
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np

SOURCE=Path(__file__).with_name('native')/'workflow.cpp'
FLAGS=['-O3','-std=c++17','-shared','-fPIC']

class ExactWorkflow:
    def __init__(self,build_dir):
        build_dir=Path(build_dir);build_dir.mkdir(parents=True,exist_ok=True)
        compiler=subprocess.check_output(['c++','--version'],text=True)
        source_sha=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
        key=hashlib.sha256((source_sha+compiler+str(FLAGS)).encode()).hexdigest()[:20]
        library=build_dir/f'workflow_{key}.so'
        if not library.exists():subprocess.run(['c++',*FLAGS,str(SOURCE),'-o',str(library)],check=True,capture_output=True,text=True)
        self.provenance={'source_sha256':source_sha,'compiler':compiler,'flags':FLAGS,'library_sha256':hashlib.sha256(library.read_bytes()).hexdigest()}
        self.lib=ctypes.CDLL(str(library.resolve()))
        floats=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');ints=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS');integer=ctypes.c_int
        self.lib.workflow_ect.argtypes=[floats,integer,integer,integer,ints];self.lib.workflow_ect.restype=ctypes.c_double
        self.lib.workflow_rank.argtypes=[floats,ints,ints,integer,integer,integer,floats];self.lib.workflow_rank.restype=None

    @staticmethod
    def problem(p):
        p=np.ascontiguousarray(p,dtype=np.float64)
        if p.ndim!=3 or min(p.shape)<1 or np.isnan(p).any() or np.any(p<=0) or not np.all(np.isfinite(p).sum(-1)==2):
            raise ValueError('positive durations and exactly two eligible hosts required')
        return p

    def greedy(self,p):
        p=self.problem(p);a=np.empty(p.shape[:2],dtype=np.int64)
        return float(self.lib.workflow_ect(p,*p.shape,a)),a

    def rank(self,p,a):
        p=self.problem(p);a=np.ascontiguousarray(a,dtype=np.int64)
        if a.shape!=p.shape[:2] or np.any(a<0) or np.any(a>=p.shape[-1]) or not np.isfinite(np.take_along_axis(p,a[...,None],-1)).all():
            raise ValueError('invalid complete assignment')
        allowed=np.argsort(p,axis=-1)[...,:2]
        alternatives=np.ascontiguousarray(np.where(a==allowed[...,0],allowed[...,1],allowed[...,0]),dtype=np.int64)
        costs=np.empty(a.size,dtype=np.float64)
        self.lib.workflow_rank(p,a,alternatives,*p.shape,costs)
        return costs,alternatives

    def descent(self,p,rounds=64,observer=None):
        score,a=self.greedy(p);used=0
        if observer is not None:observer(a.copy(),score)
        for _ in range(rounds):
            costs,alternatives=self.rank(p,a);used+=1
            if observer is not None:
                for i,c in enumerate(costs):
                    candidate=a.copy();candidate.ravel()[i]=alternatives.ravel()[i];observer(candidate,float(c))
            i=int(np.argmin(costs))
            if costs[i]>=score:break
            a.ravel()[i]=alternatives.ravel()[i];score=float(costs[i])
        return score,a,used
