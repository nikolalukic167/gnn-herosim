"""Verify retained pair-search evidence and price a free perfect-selector diagnostic."""
import argparse
import json
from pathlib import Path
import numpy as np
from src.placement.radical.pairs import PairNative
from scripts_cosim.mixed_pair_shallow import ShallowPairs
from scripts_cosim.mixed_pair_exploration import verify
from scripts_cosim.mixed_physics_control_gate import timed
from scripts_cosim.audit_radical_physics import load_problem
from scripts_cosim.radical_physics_screen import save,sha
from scripts_cosim.evaluate_workflow import paired_summary
ROOT=Path(__file__).resolve().parents[1]

def audit(root):
    pre=json.loads((root/'protocol_before_run.json').read_text());meta=json.loads((root/'source_validation.json').read_text())
    verify(root,pre,meta)
    shallow='mixed_pair_shallow_v1.json' in pre['sources'] or any(p.endswith('/mixed_pair_shallow_v1.json') for p in pre['sources'])
    engine=(ShallowPairs if shallow else PairNative)(root/'audit_build')
    live=json.loads((root/'live_READ.json').read_text())
    if live['status']!='PASS' or any(sha(ROOT/p)!=h for p,h in live['sources'].items()):raise ValueError('live source changed')
    total=0;diagnostic=[];oracle_costs=[];controls=[]
    for row in live['rows']:
        seed=row['seed'];cell=root/'test'/str(seed);read=json.loads((cell/'read.json').read_text());b=load_problem(root/'inputs'/f'{seed}.json')
        path=cell/'placements/placements.jsonl'
        if sha(path)!=read['candidate_sha256']:raise ValueError('candidate artifact changed')
        records=[json.loads(line) for line in path.read_text().splitlines()]
        base_cost,base=engine.incumbent(b)
        if records[0]['pair'] is not None or records[0]['objective']!=base_cost or records[0]['placement_plan']!=base.tolist():raise ValueError('incumbent mismatch')
        seen=set();minimum=base_cost
        for r in records[1:]:
            pair=tuple(r['pair'])
            if pair in seen or not 0<=pair[0]<pair[1]<base.size:raise ValueError('invalid or duplicate pair')
            seen.add(pair)
            c,_=engine.score(b,np.array(r['placement_plan']))
            if c!=r['objective']:raise ValueError('candidate score mismatch')
            minimum=min(minimum,c);total+=1
        if len(seen)!=base.size*(base.size-1)//2 or minimum!=read['methods']['oracle']['cost']:raise ValueError('incomplete or incorrect oracle')
        pairs=np.array([r['pair'] for r in records[1:]])
        scores,plans=engine.candidates(b,base,pairs)
        for score,plan,record in zip(scores,plans,records[1:]):
            if score!=record['objective'] or plan.tolist()!=record['placement_plan']:raise ValueError('candidate recipe mismatch')
        source=root/'live'/str(seed)/'results.jsonl'
        if sha(source)!=row['sha256']:raise ValueError('live artifact changed')
        stored=[json.loads(line) for line in source.read_text().splitlines()]
        if set(r['arm'] for r in stored)!=set(read['methods']) or len(stored)!=3:raise ValueError('incomplete live arms')
        for r in stored:
            arm=r['arm'];stats=r['stats'];expected=read['methods'][arm]
            if stats['assignment']!=expected['plan'] or abs(stats['total_rtt']*1000-expected['cost'])>1e-6:raise ValueError('live read mismatch')
            if abs(row['costs'][arm]-expected['cost'])>1e-6:raise ValueError('live report mismatch')
        pair=read['methods']['oracle']['pair']
        def free_selector():
            cost,a=engine.incumbent(b)
            if pair is None:return cost,a
            values,assignments=engine.candidates(b,a,np.array([pair]))
            return (float(values[0]),assignments[0]) if values[0]<cost else (cost,a)
        receipt=timed(free_selector,5)
        if receipt['cost']!=minimum:raise ValueError('free selector mismatch')
        diagnostic.append({'seed':seed,'time_ms':receipt['time_ms'],'timings_ms':receipt['timings_ms']})
        oracle_costs.append(minimum);controls.append(read['methods']['control']['cost'])
    gain=paired_summary(np.array([oracle_costs]),np.array([controls]))
    read=json.loads((root/'read.json').read_text())
    if gain!=read['gain']:raise ValueError('paired statistics mismatch')
    report={'status':'PASS','candidate_plans_replayed':total,'live_runs':live['runs'],'tasks':live['tasks'],
            'gain':gain,'free_selector_p95_ms':float(np.quantile([r['time_ms'] for r in diagnostic],.95)),
            'timing_scope':'posthoc optimistic cost of incumbent plus only the known best pair; selector itself is free, no GNN claim',
            'timing_rows':diagnostic,'audit_source_sha256':sha(Path(__file__)),
            'artifact_sha256':{name:sha(root/name) for name in ('read.json','live_READ.json','calibration.json','source_validation.json','protocol_before_run.json')}}
    save(root/'AUDIT.json',report);print(json.dumps({k:v for k,v in report.items() if k!='timing_rows'},indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);audit(parser.parse_args().root)
