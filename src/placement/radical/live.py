"""Independent SimPy replay for the new-physics sandbox, with full event records."""
import numpy as np
import simpy
from .environment import FLAGS

def run(b,assignment,mechanism):
    flags=FLAGS[mechanism];p=b['p'];jobs,ops,hosts=p.shape;a=np.asarray(assignment);env=simpy.Environment()
    nextop=[0]*jobs;ready=[0.]*jobs;active={};last=[None]*hosts;heat=np.zeros(hosts);completed={};events=[];energy=0.;waits={'locks':0,'power':0,'maintenance':0};peak=0.
    def dispatcher():
        nonlocal heat,energy,peak
        while len(completed)<jobs*ops:
            now=float(env.now)
            for host in sorted(list(active)):
                rec=active[host]
                if rec['end']<=now+1e-9:
                    j,k=rec['job'],rec['operation'];completed[j,k]=now;nextop[j]+=1;ready[j]=now;last[host]=int(b['types'][j,k]);events.append({'event':'done','time':now,**rec});del active[host]
            if len(completed)==jobs*ops:break
            heads=[]
            for host in range(hosts):
                if host in active:continue
                eligible=[j for j in range(jobs) if nextop[j]<ops and int(a[j,nextop[j]])==host and all(r['job']!=j for r in active.values())]
                if eligible:
                    j=min(eligible,key=lambda j:(ready[j],j));heads.append((ready[j],j,host))
            for _,j,host in sorted(heads):
                k=nextop[j]
                if flags&2 and any(int(b['locks'][j,k])&int(b['locks'][r['job'],r['operation']]) for r in active.values()):waits['locks']+=1;continue
                if flags&4 and any(b['domains'][host,g] and sum(bool(b['domains'][h,g]) for h in active)>=2 for g in range(b['domains'].shape[1])):waits['power']+=1;continue
                base=float(p[j,k,host]);setup=0.
                if flags&1 and last[host] is not None:setup=float(b['setup'][last[host],b['types'][j,k]])
                duration=(base+setup)*(1+.04*heat[host] if flags&8 else 1)
                lo,hi=b['maintenance'][host]
                if flags&16 and now<hi-1e-9 and now+duration>lo+1e-9:waits['maintenance']+=1;continue
                rec={'job':j,'operation':k,'host':host,'start':now,'end':now+duration,'base':base,'setup':setup,'heat':float(heat[host])};active[host]=rec;events.append({'event':'start','time':now,**rec});energy+=duration*b['power'][j,k]*(1+(int(now/40)+host)%3)
            future=[r['end'] for r in active.values()]
            if flags&16:future.extend(float(hi) for _,hi in b['maintenance'] if hi>now+1e-9)
            if not future:raise RuntimeError('sandbox deadlock')
            delta=min(future)-now
            if delta<=0:raise RuntimeError('nonpositive event step')
            source=np.zeros(hosts)
            for h,r in active.items():source+=b['thermal'][:,h]*b['power'][r['job'],r['operation']]
            decay=np.exp(-delta/25);heat=heat*decay+source*25*(1-decay);peak=max(peak,float(heat.max()))
            yield env.timeout(delta)
    env.process(dispatcher());env.run();ends=np.array([[completed[j,k] for k in range(ops)] for j in range(jobs)])
    completion=float(ends[:,-1].sum());tardiness=float(np.maximum(0,ends[:,-1]-b['deadlines']).sum());score=completion+(3*tardiness if flags&32 else 0)+(.2*energy if flags&64 else 0)
    return {'objective':score,'completion_sum':completion,'tardiness':tardiness,'energy_cost':float(energy),'ends':ends.tolist(),'events':events,'blocking_observations':waits,'peak_heat':peak}
