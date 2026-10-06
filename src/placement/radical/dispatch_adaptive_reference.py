"""Independent event-driven definition of the nine adaptive priority controls."""
import numpy as np
import simpy

def adaptive_replay(b,a,mode):
    j,o,h=b['p'].shape;env=simpy.Environment();nextop=np.zeros(j,int);ready_time=np.zeros(j);active={};last={};ends=np.zeros((j,o));rank=np.empty((j,o),dtype=np.int64);started=done=0
    durations=np.take_along_axis(b['p'],a[...,None],-1)[...,0]
    tail=np.flip(np.cumsum(np.flip(durations,axis=1),axis=1),axis=1)
    bits=((b['locks'][...,None]>>np.arange(30))&1).astype(float)
    pressure=bits@bits.sum((0,1));pressure=np.flip(np.cumsum(np.flip(pressure,axis=1),axis=1),axis=1)
    def process():
        nonlocal started,done
        while done<j*o:
            now=float(env.now)
            for host,rec in list(active.items()):
                job,op,end=rec
                if end<=now+1e-9:
                    ends[job,op]=now;nextop[job]+=1;ready_time[job]=now;last[host]=int(b['types'][job,op]);del active[host];done+=1
            if done==j*o:break
            busy={v[0] for v in active.values()};ready=[job for job in range(j) if job not in busy and nextop[job]<o]
            def key(job):
                op=nextop[job];host=int(a[job,op]);setup=float(b['setup'][last[host],b['types'][job,op]]) if host in last else 0.;remaining=tail[job,op]
                overlap=sum(bool(int(b['locks'][job,op])&int(b['locks'][other,nextop[other]])) for other in ready if other!=job)
                value=[durations[job,op]+setup,remaining+setup,remaining+2*setup,remaining+.5*setup,-remaining+setup,(remaining+setup)/(1+overlap),10000*setup+remaining,(remaining+setup)/max(1,pressure[job,op]),ready_time[job]][mode]
                return value,job
            ordered=sorted(ready,key=key)
            for job in ordered:
                op=nextop[job];host=int(a[job,op])
                if host in active or any(int(b['locks'][job,op])&int(b['locks'][x,y]) for x,y,_ in active.values()):continue
                if any(b['domains'][host,g] and sum(b['domains'][hh,g] for hh in active)>=2 for g in range(b['domains'].shape[1])):continue
                setup=float(b['setup'][last[host],b['types'][job,op]]) if host in last else 0.
                rank[job,op]=started;started+=1;active[host]=(job,int(op),now+durations[job,op]+setup)
            if not active:raise RuntimeError('adaptive reference deadlock')
            yield env.timeout(min(v[2] for v in active.values())-now)
    env.process(process());env.run()
    return float(ends[:,-1].sum()),rank,ends
