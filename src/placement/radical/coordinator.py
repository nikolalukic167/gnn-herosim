"""Opt-in atomic setup/lock/power service for real HeROsim platforms."""
import math
from heapq import heappush
import simpy
from src.placement.availability import service_phase

CONTRACT='mixed_execution_v1'
class MixedEnvironment(simpy.Environment):
    """The opt-in contract uses a picosecond clock to make equal-time events equal."""
    def schedule(self,event,priority=1,delay=0):
        heappush(self._queue,(round(self._now+delay,12),priority,next(self._eid),event))

class MixedExecution:
    def __init__(self,env,config):
        if config.get('contract')!=CONTRACT:raise ValueError('unknown mixed execution contract')
        self.env=env;self.rules=config['task_rules'];self.domains=config['domains'];self.setup=config['setup'];self.pending=[];self.active={};self.last={};self.events=[];self.barrier=None
        k=len(self.setup)
        if not k or any(len(row)!=k or any(not math.isfinite(v) or v<0 for v in row) for row in self.setup):raise ValueError('invalid setup matrix')
        widths={len(row) for row in self.domains.values()}
        if len(widths)!=1 or not widths or 0 in widths or any(v not in (0,1) for row in self.domains.values() for v in row):raise ValueError('invalid power domains')
        for name,r in self.rules.items():
            if not isinstance(r['type'],int) or not 0<=r['type']<k or not isinstance(r['locks'],int) or r['locks']<0:raise ValueError('invalid task resource rule')
            if name!=f"op{r['job']}_{r['operation']}":raise ValueError('unsupported workflow task identity')
    def wake(self):
        if self.barrier is not None:return
        event=simpy.Event(self.env);event._ok=True;event._value=None;event.callbacks.append(self.dispatch);self.barrier=event
        # Wait for the complete zero-duration arrival/completion cascade at this timestamp.
        self.env.schedule(event,priority=2,delay=0)
    def dispatch(self,event):
        self.barrier=None;heads={}
        for r in self.pending:
            host=r['host']
            if host in self.active:continue
            key=(r['ready'],r['rule']['job'])
            if host not in heads or key<(heads[host]['ready'],heads[host]['rule']['job']):heads[host]=r
        for r in sorted(heads.values(),key=lambda r:(r['ready'],r['rule']['job'])):
            host=r['host'];rule=r['rule']
            if any(rule['locks']&v['rule']['locks'] for v in self.active.values()):continue
            if any(member and sum(self.domains[h][g] for h in self.active)>=2 for g,member in enumerate(self.domains[host])):continue
            setup=self.setup[self.last[host]][rule['type']] if host in self.last else 0.
            r['setup']=setup;r['start']=float(self.env.now);self.active[host]=r;self.pending.remove(r)
            self.events.append({'event':'start','job':rule['job'],'operation':rule['operation'],'host':host,'time':float(self.env.now),'setup':setup,'base':r['base']})
            r['grant'].succeed(r['base']+setup)
    def execute(self,platform,task,base_duration):
        host=platform.node.node_name;name=task.type['name']
        if host not in self.domains or name not in self.rules:raise ValueError('task outside mixed execution manifest')
        if not math.isfinite(base_duration) or base_duration<=0:raise ValueError('invalid execution duration')
        grant=self.env.event();r={'host':host,'rule':self.rules[name],'ready':float(self.env.now),'grant':grant,'base':base_duration};self.pending.append(r);self.wake()
        service_phase(platform,'execution_wait',event=grant,unresolved=('mixed_host_lock_power',));duration=yield grant
        wait=self.env.now-r['ready'];task.node_contention_time=wait;platform.node.contention_time+=wait
        # Canonicalize physically simultaneous completions; no artificial batching delay.
        end=round(float(self.env.now)+duration,12)
        service_phase(platform,'execution',seconds=end-self.env.now);yield self.env.timeout(end-self.env.now)
        self.last[host]=r['rule']['type'];del self.active[host]
        self.events.append({'event':'done','job':r['rule']['job'],'operation':r['rule']['operation'],'host':host,'time':float(self.env.now),'setup':r['setup'],'base':base_duration,'wait':float(wait)})
        self.wake();return duration
    def stats(self):
        if self.pending or self.active:raise RuntimeError('unfinished mixed execution')
        return {'contract':CONTRACT,'events':self.events,'setup_total':sum(e['setup'] for e in self.events if e['event']=='done'),'resource_wait_total':sum(e['wait'] for e in self.events if e['event']=='done')}

def config_for(b):
    j,o,h=b['p'].shape
    return {'contract':CONTRACT,'task_rules':{f'op{x}_{y}':{'job':x,'operation':y,'type':int(b['types'][x,y]),'locks':int(b['locks'][x,y])} for x in range(j) for y in range(o)},'domains':{f'node{x}':list(map(int,b['domains'][x])) for x in range(h)},'setup':(b['setup']/1000).tolist()}
