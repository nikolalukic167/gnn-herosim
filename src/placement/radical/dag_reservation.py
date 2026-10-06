"""Irregular fork/join workflows with serial reservations and non-delay controls."""
import ctypes
import hashlib
import subprocess
from pathlib import Path
import numpy as np
from .block_moves import scaled_problem
from .environment import pack, initial


def problem(seed, jobs=16, ops=16, hosts=16):
    if not 4 <= ops <= 32:
        raise ValueError('irregular DAG requires 4..32 operations per job')
    b = scaled_problem(seed,jobs,ops,hosts)
    rng = np.random.default_rng(np.random.SeedSequence([seed,1207]))
    pred = np.zeros((jobs,ops),dtype=np.uint64)
    for j in range(jobs):
        # Vary the widths and depths while retaining one source and one sink.
        middle = list(range(1,ops-1))
        layers = [[0]]
        while middle:
            width = min(len(middle),int(rng.integers(1,min(5,ops-1))))
            layers.append(middle[:width]);middle=middle[width:]
        layers.append([ops-1])
        for left,right in zip(layers,layers[1:]):
            edges = {(u,int(rng.choice(right))) for u in left}
            edges.update((int(rng.choice(left)),v) for v in right)
            for u in left:
                for v in right:
                    if rng.random()<.25:edges.add((u,v))
            for u,v in edges:pred[j,v] |= np.uint64(1<<u)
        for v in range(2,ops):
            if rng.random()<.25:pred[j,v] |= np.uint64(1<<int(rng.integers(v)))
    b['predecessors'] = pred
    b['dag_contract'] = 'irregular_single_source_sink_v1'
    return b


def checked(b, assignment, rank):
    p = np.asarray(b['p'])
    a,r = np.asarray(assignment),np.asarray(rank)
    if p.ndim!=3 or not 1<=p.shape[0]<=64 or not 2<=p.shape[1]<=32 or not 2<=p.shape[2]<=28:
        raise ValueError('invalid DAG dimensions')
    if a.shape!=p.shape[:2] or r.shape!=a.shape or not np.issubdtype(a.dtype,np.integer) or not np.issubdtype(r.dtype,np.integer):
        raise ValueError('invalid DAG plan arrays')
    if np.isnan(p).any() or np.any(p<=0) or not np.all(np.isfinite(p).sum(-1)==2):
        raise ValueError('exactly two positive eligible durations required')
    if np.any(a<0) or np.any(a>=p.shape[2]) or not np.isfinite(np.take_along_axis(p,a[...,None],-1)).all():
        raise ValueError('invalid DAG placement')
    if sorted(r.ravel().tolist())!=list(range(a.size)):
        raise ValueError('priority must be a permutation')
    pred=np.asarray(b['predecessors'])
    if pred.shape!=a.shape or not np.issubdtype(pred.dtype,np.integer):
        raise ValueError('invalid DAG predecessor array')
    for row in pred:
        children=np.zeros(p.shape[1],dtype=int)
        for k,value in enumerate(row):
            bits=int(value)
            if not (bits==0 if k==0 else 0<bits<(1<<k)):
                raise ValueError('predecessors must precede operation and have unique root')
            for z in range(k):children[z] += bool(bits&(1<<z))
        if np.any(children[:-1]==0):raise ValueError('unique terminal sink required')
    if np.any(b['locks']<0) or np.any(b['locks']>=1<<28):raise ValueError('invalid lock mask')
    domains=np.asarray(b['domains'])
    if domains.ndim!=2 or domains.shape[0]!=p.shape[2] or not 1<=domains.shape[1]<=32 or not np.isin(domains,[0,1]).all():raise ValueError('invalid domains')
    setup=np.asarray(b['setup'])
    if setup.ndim!=2 or setup.shape[0]!=setup.shape[1] or not np.isfinite(setup).all() or np.any(setup<0):raise ValueError('invalid setup')
    if np.asarray(b['types']).shape!=a.shape or np.any(b['types']<0) or np.any(b['types']>=len(setup)):raise ValueError('invalid setup types')
    return pack(b),np.ascontiguousarray(pred,dtype=np.uint64),np.array(a,dtype=np.int64,order='C'),np.array(r,dtype=np.int64,order='C'),(*p.shape,domains.shape[1],len(setup))


class DagReservation:
    def __init__(self,directory):
        source=Path(__file__).with_suffix('.cpp')
        compiler=subprocess.check_output(['c++','--version'],text=True)
        flags=['-O3','-std=c++17','-shared','-fPIC']
        sources=[source,source.with_name('mixed.cpp'),source.with_name('kernel.cpp')]
        key=hashlib.sha256(b''.join(p.read_bytes() for p in sources)+compiler.encode()+str(flags).encode()).hexdigest()
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        library=directory/f'dag_reservation_{key[:16]}.so'
        if not library.exists():subprocess.run(['c++',*flags,str(source),'-o',str(library)],check=True,capture_output=True)
        self.lib=ctypes.CDLL(str(library.resolve()))
        f=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        a=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
        d=np.ctypeslib.ndpointer(dtype=np.uint64,flags='C_CONTIGUOUS')
        i=ctypes.c_int
        self.lib.dag_plan.argtypes=[f,d,a,a,*([i]*6),f,f];self.lib.dag_plan.restype=ctypes.c_double
        self.lib.dag_replay.argtypes=[f,d,a,a,f,*([i]*5),f,f];self.lib.dag_replay.restype=ctypes.c_double
        self.lib.dag_search.argtypes=[f,d,a,a,*([i]*7),ctypes.c_double,ctypes.c_uint64,ctypes.c_double,ctypes.POINTER(i),ctypes.POINTER(ctypes.c_int64)]
        self.lib.dag_search.restype=ctypes.c_double
        self.provenance={'sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},'compiler':compiler,'flags':flags,'library_sha256':hashlib.sha256(library.read_bytes()).hexdigest()}

    def plan(self,b,assignment,rank,mode=1):
        if mode not in (0,1):raise ValueError('invalid decoder')
        q,d,a,r,dims=checked(b,assignment,rank)
        starts,ends=np.empty(a.shape),np.empty(a.shape)
        cost=self.lib.dag_plan(q,d,a,r,*dims,mode,starts,ends)
        if not np.isfinite(cost):raise RuntimeError('DAG planner failed')
        return float(cost),starts,ends

    def replay(self,b,assignment,rank,release):
        q,d,a,r,dims=checked(b,assignment,rank)
        release=np.ascontiguousarray(release,dtype=float)
        if release.shape!=a.shape or not np.isfinite(release).all() or np.any(release<0):raise ValueError('invalid reservation')
        starts,ends=np.empty(a.shape),np.empty(a.shape)
        cost=self.lib.dag_replay(q,d,a,r,release,*dims,starts,ends)
        if not np.isfinite(cost):raise RuntimeError('DAG replay failed')
        return float(cost),starts,ends

    def search(self,b,assignment,rank,steps=8192,mode=2,temperature=.005,seed=77,seconds=0.):
        if type(steps) is not int or not 0<=steps<2**31 or mode not in (0,1,2) or not np.isfinite(temperature) or temperature<0 or not np.isfinite(seconds) or seconds<0:
            raise ValueError('invalid search controls')
        if type(seed) is not int or not 0<=seed<2**64:raise ValueError('invalid seed')
        q,d,a,r,dims=checked(b,assignment,rank)
        decoder,count=ctypes.c_int(),ctypes.c_int64()
        cost=self.lib.dag_search(q,d,a,r,*dims,steps,mode,temperature,seed,seconds,ctypes.byref(decoder),ctypes.byref(count))
        if not np.isfinite(cost):raise RuntimeError('DAG search failed')
        return float(cost),a,r,decoder.value,count.value


def hand_ranks(b,a):
    jobs,ops,hosts=b['p'].shape
    duration=np.take_along_axis(b['p'],a[...,None],-1)[...,0]
    successors=[[[] for _ in range(ops)] for _ in range(jobs)]
    for j in range(jobs):
        for k in range(ops):
            for z in range(k):
                if int(b['predecessors'][j,k])&(1<<z):successors[j][z].append(k)
    tail=duration.copy();down=duration.copy()
    for j in range(jobs):
        for k in range(ops-1,-1,-1):
            children=successors[j][k]
            tail[j,k]+=max([tail[j,z] for z in children]+[0.])
            reachable=set(children)
            for z in range(k+1,ops):
                if z in reachable:reachable.update(successors[j][z])
            down[j,k]+=sum(duration[j,z] for z in reachable)
    lock=((b['locks'].ravel()[:,None]>>np.arange(hosts))&1).astype(float)
    pressure=lock@lock.sum(0)
    graph=(lock@lock.T>0).astype(float)
    for j in range(jobs):
        for k in range(ops):
            for z in successors[j][k]:graph[j*ops+k,j*ops+z]+=1
    np.fill_diagonal(graph,0)
    graph/=np.maximum(graph.sum(-1,keepdims=True),1)
    signal=tail.ravel();hop1=graph@signal;hop2=graph@hop1;hop4=graph@(graph@hop2)
    job_work=np.repeat(duration.sum(1),ops)
    values={'short_job':job_work+duration.ravel()*.001,'critical_path':-tail.ravel(),
            'downstream_work':-down.ravel(),'short_operation':duration.ravel(),
            'lock_pressure':-pressure,'graph1':-(signal+hop1),'graph2':-(signal+hop2),'graph4':-(signal+hop4)}
    ranks={}
    for name,score in values.items():
        rank=np.empty(a.size,dtype=np.int64);rank[np.argsort(score,kind='stable')]=np.arange(a.size)
        ranks[name]=rank.reshape(a.shape)
    return ranks


def baseline(engine,b,mode=2):
    best=None
    for method in ('fastest','affinity'):
        a=initial(b,method)
        for name,rank in hand_ranks(b,a).items():
            for decoder in ((0,1) if mode==2 else (mode,)):
                cost,_,_=engine.plan(b,a,rank,decoder)
                if best is None or cost<best[0]:best=(cost,a.copy(),rank.copy(),decoder,f'{method}:{name}')
    return best
