"""Complete FIFO workflow planning with explicit future operation manifests."""
import heapq,time,json,math
import numpy as np
from contextvars import ContextVar

# Optional complete-plan trace used by dataset generation; no search decision reads it.
plan_observer = ContextVar('workflow_plan_observer', default=None)

def observe(p,plan,cost):
 callback=plan_observer.get()
 if callback is not None:
  a=np.zeros(p.shape[:2],dtype=np.int16)
  for j,k,h in plan:a[j,k]=h
  callback(a,float(cost))

def instance(seed,jobs=8,ops=10,machines=4):
 rng=np.random.default_rng(seed);p=rng.integers(1,31,(jobs,ops,machines)).astype(float)+rng.uniform(0,.001,(jobs,ops,machines))
 for j in range(jobs):
  for k in range(ops):
   forbidden=rng.choice(machines,size=machines-2,replace=False);p[j,k,forbidden]=np.inf
 return p

def initial(p):return (tuple((0.,j,0) for j in range(len(p))),tuple([0.]*p.shape[2]),0.,())
def step(s,p,h):
 heap,free,total,plan=s;heap=list(heap);time_,j,k=heapq.heappop(heap);free=list(free);end=max(time_,free[h])+p[j,k,h];free[h]=end
 if k+1<p.shape[1]:heapq.heappush(heap,(end,j,k+1))
 else:total+=end
 if not heap:observe(p,plan+((j,k,h),),total)
 return tuple(heap),tuple(free),total,plan+((j,k,h),)
def rollout(s,p):
 while s[0]:
  t,j,k=s[0][0];h=int(np.argmin(np.maximum(t,s[1])+p[j,k]));s=step(s,p,h)
 return s

def beam(p,width,use_rollout=False):
 suffix=np.flip(np.cumsum(np.flip(p.min(2),axis=1),axis=1),axis=1)
 frontier=[initial(p)];best=rollout(frontier[0],p)
 for _ in range(p.shape[0]*p.shape[1]):
  candidates=[]
  for s in frontier:
   t,j,k=s[0][0]
   for h in np.flatnonzero(np.isfinite(p[j,k])):
    q=step(s,p,int(h));lb=q[2]+sum(time_+suffix[jj,kk] for time_,jj,kk in q[0])
    if lb>best[2]+1e-9:continue
    if use_rollout:
     full=rollout(q,p)
     if full[2]<best[2]:best=full
     rank=full[2]
    else:rank=lb
    candidates.append((rank,q))
  if not candidates:break
  candidates.sort(key=lambda x:x[0]);frontier=[q for rank,q in candidates[:width]]
  if not frontier[0][0]:
   q=min(frontier,key=lambda x:x[2])
   if q[2]<best[2]:best=q
   break
 return best


def evaluate(p,assignment):
 jobs,ops,machines=p.shape;free=[0.]*machines;heap=[(0.,j,0) for j in range(jobs)];total=0.
 while heap:
  release,j,k=heapq.heappop(heap);h=assignment[j,k];end=max(release,free[h])+p[j,k,h];free[h]=end
  if k+1<ops:heapq.heappush(heap,(end,j,k+1))
  else:total+=end
 callback=plan_observer.get()
 if callback is not None:callback(np.asarray(assignment,dtype=np.int16),float(total))
 return total

def anneal(p,plan,seed,steps):
 rng=np.random.default_rng(seed);n=p.shape[0]*p.shape[1];allowed=[np.flatnonzero(np.isfinite(x)).tolist() for x in p.reshape(n,p.shape[2])]
 a=np.zeros(p.shape[:2],dtype=int)
 for j,k,h in plan:a[j,k]=h
 assert all(a.ravel()[i] in allowed[i] for i in range(n))
 score=evaluate(p,a);best=score;bestplan=a.copy();flat=a.ravel();mut=rng.integers(n,size=(steps,2));uniform=rng.random(steps);temps=np.geomspace(score*.02,score*.0001,1000)
 for t,(i,j) in enumerate(mut):
  oldi,oldj=flat[i],flat[j];flat[i]=allowed[i][1-allowed[i].index(oldi)]
  if t%3==0 and j!=i:flat[j]=allowed[j][1-allowed[j].index(oldj)]
  value=evaluate(p,a);delta=value-score
  if delta<0 or uniform[t]<math.exp(-min(700,delta/temps[t%1000])):score=value
  else:flat[i],flat[j]=oldi,oldj
  if score<best:best=score;bestplan=a.copy()
  if t%1000==999:a[:]=bestplan;score=best
 return best,bestplan


def route_greedy(p,weight):
    eligible=np.isfinite(p);mass=np.where(eligible,p,0)/eligible.sum(2,keepdims=True)
    suffix=np.flip(np.cumsum(np.flip(mass,axis=1),axis=1),axis=1)
    s=initial(p)
    while s[0]:
        t,j,k=s[0][0]
        work=sum((suffix[jj,kk] for _,jj,kk in s[0]),start=np.zeros(p.shape[2]))
        scores=np.maximum(t,s[1])+p[j,k]+weight*work
        s=step(s,p,int(np.argmin(scores)))
    return s
