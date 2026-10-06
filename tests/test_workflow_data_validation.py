import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from scripts_cosim.validate_workflow_data import validate,digest
from src.policy.workflow.model import CONTRACT,features
from src.placement.workflow_planning import instance,initial,rollout


@pytest.fixture
def corpus(tmp_path):
    cases=[];files={}
    for seed,split in enumerate(('train','validation','test')):
        p=instance(seed,jobs=2,ops=3,machines=4);state=rollout(initial(p),p);a=np.zeros((2,3),dtype=int)
        for j,k,h in state[3]:a[j,k]=h
        x,e=features(p);xh,_=features(p,True)
        cell=tmp_path/f'ds_{seed}';(cell/'placements').mkdir(parents=True)
        source=cell/'problem.json';source.write_text(json.dumps({'seed':seed,'processing_ms':np.where(np.isfinite(p),p,None).tolist()}))
        trace=cell/'placements/placements.jsonl';trace.write_text(json.dumps({'placement_plan':a.tolist(),'rtt_ms':state[2]})+'\n')
        (cell/'best.json').write_text(json.dumps({'placement_plan':a.tolist(),'rtt_ms':state[2]}))
        file=tmp_path/f'{split}.npz';np.savez(file,p=p[None],labels=a[None],x=x[None],xh=xh[None],eligible=e[None],teacher=[state[2]],seeds=[seed]);files[file.name]=digest(file)
        cases.append({'seed':seed,'split':split,'source_sha256':digest(source),'placement_sha256':digest(trace),'instance_sha256':hashlib.sha256(p.tobytes()).hexdigest()})
    (tmp_path/'METADATA.json').write_text(json.dumps({'contract':CONTRACT,'sources':{'src/placement/workflow_planning.py':digest('src/placement/workflow_planning.py')},'cases':cases,'files':files}))
    return tmp_path


def test_validates_actual_sources(corpus):
    assert validate(corpus)['actual_source_identity_unique']==3
    source=corpus/'ds_1/problem.json';source.write_text(source.read_text()+' ')
    with pytest.raises(ValueError,match='actual artifact fingerprint mismatch'):validate(corpus)


def test_duplicate_actual_infrastructure_fails_despite_distinct_seeds(corpus):
    source=corpus/'ds_1/problem.json';first=json.loads((corpus/'ds_0/problem.json').read_text());first['seed']=1;source.write_text(json.dumps(first))
    meta=json.loads((corpus/'METADATA.json').read_text());meta['cases'][1]['source_sha256']=digest(source);meta['cases'][1]['instance_sha256']=meta['cases'][0]['instance_sha256'];(corpus/'METADATA.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='duplicate or mismatched actual infrastructure'):validate(corpus)


def test_declared_code_hash_does_not_override_actual_source(corpus):
    meta=json.loads((corpus/'METADATA.json').read_text());meta['sources']['src/placement/workflow_planning.py']='0'*64;(corpus/'METADATA.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='source fingerprint mismatch'):validate(corpus)
