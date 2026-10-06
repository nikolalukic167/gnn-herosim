"""Check actual resource-event starts and charges against frozen DAG schedules."""
import argparse
from pathlib import Path
import numpy as np
from scripts_cosim.mixed_joint_dispatch_screen import ROOT,read,save,sha
from scripts_cosim.audit_radical_physics import load_problem


def audit(out):
    source=Path(__file__).resolve()
    pre=read(out/'protocol_before_run.json')
    for name,digest in pre['sources'].items():assert sha(ROOT/name)==digest,name
    manifest=read(out/'artifacts.json')
    runs=operations=0
    for path in sorted(out.glob('*/*/*_live.json')):
        assert sha(path)==manifest[str(path.relative_to(out))]
        actual=read(path)
        replay_path=path.with_name(path.name.replace('_live.json','_replay.json'))
        assert sha(replay_path)==manifest[str(replay_path.relative_to(out))]
        expected=read(replay_path);b=load_problem(path.parent/'input.json')
        starts,ends,a,release=(np.array(expected[k]) for k in ('starts','ends','assignment','release'))
        seen={'start':set(),'done':set()}
        for event in actual['mixed_execution']['events']:
            kind=event['event'];node=(event['job'],event['operation'])
            assert kind in seen and node not in seen[kind],(path,kind,node)
            seen[kind].add(node)
            assert event['host']==f'node{a[node]}'
            assert abs(event['time']*1000-(starts[node] if kind=='start' else ends[node]))<1e-6,(path,kind,node)
            base=b['p'][node][a[node]]
            assert abs(event['base']*1000-base)<1e-6
            assert abs(event['setup']*1000-(ends[node]-starts[node]-base))<1e-6
        target=set(np.ndindex(a.shape))
        assert seen['start']==seen['done']==target
        assert np.all(starts+1e-9>=release)
        assert all(event['timestamp']==0 for event in actual['config']['workload']['events'])
        assert abs(sum(t['elapsedTime'] for t in actual['tasks'])-actual['total_task_rtt'])<1e-9
        assert abs(actual['job_completion_sum']*1000-ends[:,-1].sum())<1e-6
        runs+=1;operations+=a.size
    report=read(out/'read.json')
    assert (runs,operations)==(report['live_runs'],report['live_operations'])
    result={'status':'PASS','live_runs':runs,'live_operations':operations,'resource_events':operations*2,
            'auditor_sha256':sha(source),'report_sha256':sha(out/'read.json')}
    save(out/'EVENT_AUDIT.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    print(audit(parser.parse_args().out))
