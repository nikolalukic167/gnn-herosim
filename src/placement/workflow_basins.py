"""Finite initializer portfolio and exact descent, shared by learned and hand selectors."""
import ctypes,hashlib,subprocess
from pathlib import Path
import numpy as np
from src.placement.workflow_exact import ExactWorkflow,FLAGS,SOURCE

class BasinPortfolio:
    def __init__(self,build_dir):
        self.exact=ExactWorkflow(build_dir);source=SOURCE.with_name('workflow_starts.cpp')
        compiler=subprocess.check_output(['c++','--version'],text=True)
        key=hashlib.sha256(source.read_bytes()+SOURCE.read_bytes()+compiler.encode()+str(FLAGS).encode()).hexdigest()[:20]
        library=Path(build_dir)/f'starts_{key}.so'
        if not library.exists():subprocess.run(['c++',*FLAGS,str(source),'-o',str(library)],check=True,capture_output=True,text=True)
        self.lib=ctypes.CDLL(str(library.resolve()));f=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');a=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS');i=ctypes.c_int
        self.lib.workflow_weighted_start.argtypes=[f,i,i,i,ctypes.c_double,a];self.lib.workflow_weighted_start.restype=ctypes.c_double
        self.lib.workflow_score.argtypes=[f,a,i,i,i];self.lib.workflow_score.restype=ctypes.c_double
        self.lib.workflow_refine.argtypes=[f,a,i,i,i,i,ctypes.POINTER(i)];self.lib.workflow_refine.restype=ctypes.c_double
        self.provenance={'exact':self.exact.provenance,'portfolio_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'library_sha256':hashlib.sha256(library.read_bytes()).hexdigest(),'compiler':compiler,'flags':FLAGS}

    def score(self,p,a):
        p=self.exact.problem(p);a=np.ascontiguousarray(a,dtype=np.int64)
        if a.shape!=p.shape[:2] or np.any(a<0) or np.any(a>=p.shape[-1]) or not np.isfinite(np.take_along_axis(p,a[...,None],-1)).all():raise ValueError('invalid initializer')
        return float(self.lib.workflow_score(p,a,*p.shape))

    def start(self,p,index):
        if not 0<=index<12:raise ValueError('unknown initializer')
        p=self.exact.problem(p);a=np.empty(p.shape[:2],dtype=np.int64)
        weight=(0.,.25,.5,1.)[index] if index<4 else 0.
        cost=float(self.lib.workflow_weighted_start(p,*p.shape,weight,a))
        if index>=4:
            allowed=np.argsort(p,axis=-1)[...,:2];alternate=np.where(a==allowed[...,0],allowed[...,1],allowed[...,0])
            mask=np.random.default_rng((index-4)%4).random(a.shape)<(.1 if index<8 else .25)
            a=np.where(mask,alternate,a).astype(np.int64);cost=self.score(p,a)
        return cost,a

    def starts(self,p):
        p=self.exact.problem(p);plans=[];costs=[]
        for weight in (0.,.25,.5,1.):
            a=np.empty(p.shape[:2],dtype=np.int64);cost=self.lib.workflow_weighted_start(p,*p.shape,weight,a);plans.append(a);costs.append(float(cost))
        base=plans[0];allowed=np.argsort(p,axis=-1)[...,:2];alternate=np.where(base==allowed[...,0],allowed[...,1],allowed[...,0])
        for rate in (.1,.25):
            for seed in range(4):
                mask=np.random.default_rng(seed).random(base.shape)<rate;a=np.where(mask,alternate,base).astype(np.int64);plans.append(a);costs.append(self.score(p,a))
        return np.stack(plans),np.array(costs)

    def refine(self,p,a,rounds=64,observer=None):
        p=self.exact.problem(p);a=np.array(a,dtype=np.int64,copy=True);score=self.score(p,a);used=0
        if observer is None:
            counter=ctypes.c_int();score=float(self.lib.workflow_refine(p,a,*p.shape,rounds,ctypes.byref(counter)))
            return score,a,counter.value
        if observer:observer(a.copy(),score)
        for _ in range(rounds):
            costs,alternate=self.exact.rank(p,a);used+=1
            if observer:
                for i,c in enumerate(costs):
                    b=a.copy();b.ravel()[i]=alternate.ravel()[i];observer(b,float(c))
            i=int(np.argmin(costs))
            if costs[i]>=score:break
            a.ravel()[i]=alternate.ravel()[i];score=float(costs[i])
        return score,a,used

    def select(self,p,rule,fixed=0):
        if rule=='ect':return self.refine(p,self.exact.greedy(p)[1])
        if rule=='fixed':return self.refine(p,self.start(p,fixed)[1])
        plans,costs=self.starts(p)
        choices=[fixed] if rule=='fixed' else np.argsort(costs,kind='stable')[:2 if rule=='top2' else 1]
        if rule not in ('fixed','min_initial','top2'):raise ValueError('unknown selector')
        outcomes=[self.refine(p,plans[int(i)]) for i in choices]
        return min(outcomes,key=lambda x:x[0])
