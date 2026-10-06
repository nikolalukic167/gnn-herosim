from itertools import combinations
import numpy as np
import pytest
from src.placement.radical.environment import problem,initial
from src.placement.radical.pairs import PairNative,rank_pairs
from src.placement.radical.live import run

@pytest.fixture(scope='module')
def engine(tmp_path_factory):return PairNative(tmp_path_factory.mktemp('pairs'))

@pytest.mark.parametrize('seed',[7,19,106006])
def test_pair_portfolio_matches_individual_search_and_simpy(engine,seed):
    b=problem(seed,jobs=3,ops=3);a=initial(b,'affinity');before=a.copy()
    pairs=np.array(list(combinations(range(a.size),2)),dtype=np.int64)
    costs,plans=engine.candidates(b,a,pairs)
    for pair,c,plan in zip(pairs,costs,plans):
        changed=a.copy()
        for i in pair:
            changed.flat[i]=next(h for h in range(b['p'].shape[2]) if h!=a.flat[i] and np.isfinite(b['p'].reshape(-1,4)[i,h]))
        expected,expected_plan=engine.search(b,changed,2)
        assert c==expected and np.array_equal(plan,expected_plan)
        assert run(b,plan,'mixed')['objective']==pytest.approx(c)
    assert np.array_equal(a,before)

@pytest.mark.parametrize('pairs',[[[0,0]],[[0,9]],[[0,-1]],[[0.,1.]], [0,1]])
def test_invalid_pair_rejected(engine,pairs):
    b=problem(7,jobs=3,ops=3)
    with pytest.raises(ValueError):engine.candidates(b,initial(b,'affinity'),pairs)

def test_ranking_is_label_free_and_complete():
    b=problem(7);a=initial(b,'affinity')
    for kind in ('random','graph'):
        pairs=rank_pairs(b,a,kind)
        assert len(set(map(tuple,pairs)))==a.size*(a.size-1)//2
        assert np.array_equal(pairs,rank_pairs(b,a,kind))

@pytest.mark.parametrize('seed',[7,19])
def test_shallow_portfolio_matches_declared_one_round(engine,seed):
    from scripts_cosim.mixed_pair_shallow import ShallowPairs
    b=problem(seed,jobs=3,ops=3)
    shallow=object.__new__(ShallowPairs);shallow.__dict__.update(engine.__dict__)
    c,a=shallow.incumbent(b);d,z=engine.search(b,initial(b,'affinity'),2)
    assert c==d and np.array_equal(a,z)
    pair=rank_pairs(b,a,'graph')[:1]
    scores,plans=engine.candidates(b,a,pair,1)
    got,plan=shallow.plan(b,'graph1')
    expected=(float(scores[0]),plans[0]) if scores[0]<c else (c,a)
    assert got==expected[0] and np.array_equal(plan,expected[1])

@pytest.mark.parametrize('corruption',['source','input','duplicate'])
def test_gate_rejects_bad_provenance_before_simulation(tmp_path,monkeypatch,corruption):
    import hashlib
    import json
    import scripts_cosim.mixed_pair_exploration as gate
    from src.placement.radical.environment import pack,serialized
    monkeypatch.setattr(gate,'ROOT',tmp_path)
    (tmp_path/'inputs').mkdir();src=tmp_path/'code.py';src.write_text('original')
    b=problem(7);path=tmp_path/'inputs/7.json';path.write_text(json.dumps(serialized(b)))
    pre={'sources':{'code.py':gate.sha(src)}}
    meta={'inputs':{'7':{'identity':hashlib.sha256(pack(b).tobytes()).hexdigest(),'sha256':gate.sha(path)}}}
    gate.verify(tmp_path,pre,meta)
    if corruption=='source':src.write_text('changed')
    if corruption=='input':path.write_text('{}')
    if corruption=='duplicate':
        (tmp_path/'inputs/8.json').write_bytes(path.read_bytes());meta['inputs']['8']=dict(meta['inputs']['7'])
    with pytest.raises((ValueError,KeyError)):gate.verify(tmp_path,pre,meta)
