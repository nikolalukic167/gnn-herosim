"""Exercise deterministic CUDA training before the workflow pilot."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import json
import numpy as np
import torch
from src.policy.workflow.model import WorkflowAssignmentNet, features
from src.policy.workflow.train import seed_everything, train_epoch
from src.placement.workflow_planning import instance

torch.set_num_threads(1)
assert torch.cuda.is_available(), 'GPU allocation has no usable CUDA device'
for arm in ('gnn', 'mpoff', 'mlp_hand'):
    state=[]
    for repeat in range(2):
        seed_everything(771)
        pairs=[features(instance(i,jobs=2,ops=3,machines=4),hand=arm=='mlp_hand') for i in range(4)]
        x=torch.tensor(np.stack([p[0] for p in pairs]),device='cuda')
        e=torch.tensor(np.stack([p[1] for p in pairs]),device='cuda')
        y=e.argmax(-1)
        model=WorkflowAssignmentNet(x.shape[-1],mp=arm=='gnn').cuda()
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        generator=torch.Generator().manual_seed(771)
        for epoch in range(3):train_epoch(model,optimizer,x,e,y,2,generator)
        state.append({k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
    assert all(torch.equal(state[0][k],state[1][k]) for k in state[0]), arm
print(json.dumps({'gpu_determinism':'PASS','torch':torch.__version__,'device':torch.cuda.get_device_name(0)}),flush=True)
