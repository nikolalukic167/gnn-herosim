"""Post-hoc decoder-translation diagnostic; never a fresh qualification result."""
import argparse
from pathlib import Path
import numpy as np
from src.placement.radical.dag_reservation import DagReservation
from src.placement.radical.dag_reservation_bridge import chronological
from src.placement.radical.dag_reservation_live import replay
from scripts_cosim.audit_radical_physics import load_problem
from scripts_cosim.mixed_joint_dispatch_screen import ROOT,read,save,sha


def diagnose(parent,out):
    for name,digest in read(parent/'protocol_before_run.json')['sources'].items():
        assert sha(ROOT/name)==digest,name
    engine=DagReservation(parent/'build');assert engine.provenance==read(parent/'native.json')
    report=read(parent/'read.json');rows=[]
    for cell in sorted((parent/'fresh').iterdir()):
        b=load_problem(cell/'input.json');row=read(cell/'read.json');p=row['arms'][report['selected']]
        a,rank=np.array(p['assignment']),np.array(p['priority'])
        cost,starts,_=engine.plan(b,a,rank,p['decoder']);assert cost==p['cost']
        raw,_,_=engine.plan(b,a,rank,1);mapped=chronological(rank,starts)
        value,release,ends=engine.plan(b,a,mapped,1)
        independent=replay(b,a,mapped,release)
        assert value==independent['objective'] and np.array_equal(ends,independent['ends'])
        rows.append({'seed':row['seed'],'incumbent':cost,'raw_serial':raw,'mapped_serial':value,
                     'input_sha256':sha(cell/'input.json'),'read_sha256':sha(cell/'read.json')})
    result={'scope':'post-hoc representation diagnostic, not fresh qualification',
            'parent_report_sha256':sha(parent/'read.json'),'auditor_sha256':sha(Path(__file__)),
            'translation_source_sha256':sha(ROOT/'src/placement/radical/dag_reservation_bridge.py'),
            'raw_median_excess_pct':float(np.median([(r['raw_serial']/r['incumbent']-1)*100 for r in rows])),
            'mapped_equal_count':sum(r['mapped_serial']==r['incumbent'] for r in rows),'rows':rows}
    save(out,result)
    print({k:v for k,v in result.items() if k!='rows'})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();diagnose(args.parent,args.out)
