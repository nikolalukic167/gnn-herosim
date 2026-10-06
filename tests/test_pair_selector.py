import numpy as np
import pytest
import torch
from src.placement.radical.environment import problem,initial
from src.policy.pair_selector.model import PairNet,features
from src.policy.pair_selector.train import train_epoch
from src.policy.workflow.train import seed_everything

@pytest.mark.parametrize('arm',['gnn','mpoff','mlp_hand'])
def test_training_determinism_and_input_isolation(arm):
    torch.set_num_threads(1);b=problem(17,jobs=3,ops=3);a=initial(b,'affinity');ends=np.arange(9).reshape(3,3)
    x,adj,pair=features(b,a,ends,arm=='mlp_hand');tx=torch.from_numpy(np.stack([x,x]));ta=torch.from_numpy(np.stack([adj,adj]));tp=torch.from_numpy(np.stack([pair,pair]))
    costs=torch.arange(37).float()[None].repeat(2,1)+100.;states=[]
    for _ in range(2):
        seed_everything(42);m=PairNet(x.shape[-1],nodes=9,mp=arm=='gnn');opt=torch.optim.AdamW(m.parameters(),lr=.001)
        train_epoch(m,opt,tx,ta,tp,costs,torch.Generator().manual_seed(42),2);states.append({k:v.clone() for k,v in m.state_dict().items()})
    assert all(torch.equal(states[0][k],states[1][k]) for k in states[0])
    b['teacher_plan']=np.ones_like(a);b['future_labels']=np.arange(37)
    again=features(b,a,ends,arm=='mlp_hand')
    assert all(np.array_equal(u,v) for u,v in zip((x,adj,pair),again))

def test_mp_off_ignores_adjacency_but_gnn_can_distinguish_it():
    seed_everything(42);b=problem(17,jobs=3,ops=3);x,adj,pair=features(b,initial(b,'affinity'),np.zeros((3,3)))
    x=torch.from_numpy(x)[None];a=torch.from_numpy(adj)[None];p=torch.from_numpy(pair)[None]
    for mp in (False,True):
        model=PairNet(x.shape[-1],nodes=9,mp=mp).eval()
        with torch.inference_mode():first=model(x,a,p);second=model(x,a*0,p)
        assert torch.equal(first,second)==(not mp)

def test_curve_reader_treats_pair_success_as_higher_is_better(tmp_path,capsys):
    from scripts_cosim.read_training_curves import main
    path=tmp_path/'history.csv'
    path.write_text('epoch,val/workflow_rtt_ms,val/pair_optimal_fraction\n0,100,0\n1,90,0.5\n2,95,0.25\n')
    assert main(['--csv',str(path)])==0
    assert 'val/pair_optimal_fraction peaks at epoch 0' not in capsys.readouterr().out
