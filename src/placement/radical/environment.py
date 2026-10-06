"""Isolated announced-workload physics sandbox; no production simulator mutations."""
import ctypes,hashlib,subprocess
from pathlib import Path
import numpy as np
FLAGS={'none':0,'setup':1,'locks':2,'power':4,'thermal':8,'maintenance':16,'deadlines':32,'carbon':64,'mixed':7}

def problem(seed,jobs=8,ops=6,hosts=4):
    rng=np.random.default_rng(seed);p=rng.integers(3,19,(jobs,ops,hosts)).astype(float)
    for j in range(jobs):
        for k in range(ops):p[j,k,rng.choice(hosts,hosts-2,replace=False)]=np.inf
    types=rng.integers(0,4,(jobs,ops));locks=np.zeros((jobs,ops),int)
    for j in range(jobs):
        for k in range(ops):
            for l in rng.choice(6,rng.integers(1,3),replace=False):locks[j,k]|=1<<int(l)
    domains=(rng.random((hosts,3))<.7).astype(float);thermal=rng.uniform(.05,.3,(hosts,hosts));np.fill_diagonal(thermal,.6)
    maint=np.column_stack((rng.integers(20,55,hosts),rng.integers(70,105,hosts)))
    deadlines=np.min(p,axis=-1).sum(1)*rng.uniform(1.2,2.2,jobs)
    setup=rng.integers(3,21,(4,4)).astype(float);np.fill_diagonal(setup,0)
    return {'seed':seed,'p':p,'types':types,'locks':locks,'domains':domains,'thermal':thermal,'maintenance':maint,'deadlines':deadlines,'power':rng.uniform(.5,1.5,(jobs,ops)),'setup':setup}

def pack(b):return np.ascontiguousarray(np.concatenate([np.asarray(b[k],dtype=float).ravel() for k in ('p','types','locks','domains','thermal','maintenance','deadlines','power','setup')]))
def serialized(b):
    result={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in b.items()};p=b['p'].astype(object);p[~np.isfinite(b['p'])]=None;result['p']=p.tolist();return result

def initial(b,method):
    p=b['p'];j,o,h=p.shape;a=np.argmin(p,axis=-1);load=np.zeros(h);last=np.full(h,-1)
    if method=='fastest':return a.astype(np.int64)
    for k in range(o):
        for job in range(j):
            costs=p[job,k].copy()
            if method=='load':costs+=load
            elif method=='affinity':
                for host in range(h):
                    if last[host]>=0:costs[host]+=b['setup'][last[host],b['types'][job,k]]
                costs+=.5*load
            else:raise ValueError('unknown start')
            host=int(costs.argmin());a[job,k]=host;load[host]+=p[job,k,host];last[host]=b['types'][job,k]
    return a.astype(np.int64)

class Native:
    def __init__(self,directory):
        src=Path(__file__).with_name('kernel.cpp');flags=['-O3','-std=c++17','-shared','-fPIC'];compiler=subprocess.check_output(['c++','--version'],text=True);sha=lambda v:hashlib.sha256(v).hexdigest();key=sha(src.read_bytes()+compiler.encode()+str(flags).encode());directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);lib=directory/f'{key[:16]}.so'
        if not lib.exists():subprocess.run(['c++',*flags,str(src),'-o',str(lib)],check=True,capture_output=True)
        self.provenance={'source_sha256':sha(src.read_bytes()),'library_sha256':sha(lib.read_bytes()),'compiler':compiler,'flags':flags};self.lib=ctypes.CDLL(str(lib.resolve()));f=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');a=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS');i=ctypes.c_int
        self.lib.radical_score.argtypes=[f,a,i,i,i,i,i,i,f];self.lib.radical_score.restype=ctypes.c_double
        self.lib.radical_search.argtypes=[f,a,i,i,i,i,i,i,i,i,ctypes.c_uint64];self.lib.radical_search.restype=ctypes.c_double
    def checked(self,b,a):
        p=b['p'];a=np.ascontiguousarray(a,dtype=np.int64)
        if p.ndim!=3 or a.shape!=p.shape[:2] or np.isnan(p).any() or np.any(p<=0) or not np.all(np.isfinite(p).sum(-1)==2):raise ValueError('invalid problem')
        if np.any(a<0) or np.any(a>=p.shape[2]) or not np.isfinite(np.take_along_axis(p,a[...,None],-1)).all():raise ValueError('invalid placement')
        return pack(b),a,(*p.shape,b['domains'].shape[1],b['setup'].shape[0])
    def score(self,b,a,mechanism):
        q,a,dims=self.checked(b,a);ends=np.empty(a.size);score=self.lib.radical_score(q,a,*dims,FLAGS[mechanism],ends)
        if not np.isfinite(score):raise RuntimeError('nonterminating schedule')
        return float(score),ends.reshape(a.shape)
    def plan(self,b,mechanism,method):
        if method in ('fastest','load','affinity'):
            a=initial(b,method);return self.score(b,a,mechanism)[0],a
        a=initial(b,'load');q,a,dims=self.checked(b,a)
        anneal=method.startswith('anneal');steps=int(method[6:]) if anneal else 6
        if not anneal and method!='descent':raise ValueError('unknown search')
        score=self.lib.radical_search(q,a,*dims,FLAGS[mechanism],steps,int(anneal),77)
        if not np.isfinite(score):raise RuntimeError('nonterminating search')
        return float(score),a
