"""Independent SimPy priority replay and opt-in actual HeROsim arbitration adapter."""
import numpy as np
import simpy
from . import coordinator
from scripts_cosim.mixed_physics_live import run as mixed_run

class PriorityExecution(coordinator.MixedExecution):
    def dispatch(self,event):
        self.barrier=None
        ordered=sorted(self.pending,key=lambda r:(self.priority[r['rule']['job'],r['rule']['operation']],r['rule']['job']))
        for r in ordered:
            host=r['host'];rule=r['rule']
            if host in self.active:continue
            if any(rule['locks']&v['rule']['locks'] for v in self.active.values()):continue
            if any(member and sum(self.domains[h][g] for h in self.active)>=2 for g,member in enumerate(self.domains[host])):continue
            setup=self.setup[self.last[host]][rule['type']] if host in self.last else 0.
            r['setup']=setup;r['start']=float(self.env.now);self.active[host]=r;self.pending.remove(r)
            self.events.append({'event':'start','job':rule['job'],'operation':rule['operation'],'host':host,'time':float(self.env.now),'setup':setup,'base':r['base']})
            r['grant'].succeed(r['base']+setup)
    def stats(self):
        return {**super().stats(),'dispatch_contract':'mixed_ready_priority_v1','priority':self.priority.tolist()}

def herosim(b,a,priority,log,seed):
    old=coordinator.MixedExecution
    class BoundPriority(PriorityExecution):
        def __init__(self,env,config):
            super().__init__(env,config);self.priority=np.array(priority,copy=True)
    coordinator.MixedExecution=BoundPriority
    try:return mixed_run(b,a,log,seed)
    finally:coordinator.MixedExecution=old

def replay(b,a,priority):
    env=simpy.Environment();j,o,h=b['p'].shape;nextop=np.zeros(j,int);active={};last={};done={};events=[]
    def process():
        while len(done)<j*o:
            now=float(env.now)
            for host,rec in list(active.items()):
                if rec['end']<=now+1e-9:
                    job,op=rec['job'],rec['operation'];done[job,op]=now;nextop[job]+=1;last[host]=int(b['types'][job,op]);del active[host];events.append({'event':'done',**rec})
            if len(done)==j*o:break
            busy={rec['job'] for rec in active.values()}
            ready=[job for job in range(j) if job not in busy and nextop[job]<o]
            for job in sorted(ready,key=lambda job:(priority[job,nextop[job]],job)):
                op=nextop[job];host=int(a[job,op]);lock=int(b['locks'][job,op])
                if host in active or any(lock&int(b['locks'][rec['job'],rec['operation']]) for rec in active.values()):continue
                if any(b['domains'][host,g] and sum(b['domains'][hh,g] for hh in active)>=2 for g in range(b['domains'].shape[1])):continue
                setup=float(b['setup'][last[host],b['types'][job,op]]) if host in last else 0.
                rec={'job':int(job),'operation':int(op),'host':host,'start':now,'end':now+float(b['p'][job,op,host])+setup,'setup':setup}
                active[host]=rec;events.append({'event':'start',**rec})
            if not active:raise RuntimeError('priority deadlock')
            yield env.timeout(min(rec['end'] for rec in active.values())-now)
    env.process(process());env.run();ends=np.array([[done[x,y] for y in range(o)] for x in range(j)])
    return {'objective':float(ends[:,-1].sum()),'ends':ends,'events':events}
