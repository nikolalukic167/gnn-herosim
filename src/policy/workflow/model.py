"""Matched MP-on/off assignment scorers; hand arm adds cheap plan columns."""
import numpy as np
import torch
from torch import nn
from src.placement.workflow_planning import initial,rollout,beam

CONTRACT='workflow_manifest_assignment_v1'

def features(p,hand=False):
    eligible=np.isfinite(p);x=np.where(eligible,p,0.)/30.;j,o,m=p.shape
    minimal=np.min(p,axis=2)/30.
    prefix=np.cumsum(minimal,axis=1)-minimal;suffix=np.flip(np.cumsum(np.flip(minimal,axis=1),axis=1),axis=1)
    mass=eligible.sum((0,1))/(j*o);work=x.sum((0,1))/(j*o)
    localwork=x/np.maximum(eligible.sum(2,keepdims=True),1)
    future=np.flip(np.cumsum(np.flip(localwork,axis=1),axis=1),axis=1)/o
    past=(np.cumsum(localwork,axis=1)-localwork)/o
    scalars=np.stack([np.broadcast_to(np.arange(j)[:,None]/j,(j,o)),np.broadcast_to(np.arange(o)[None,:]/o,(j,o)),minimal,prefix/o,suffix/o],axis=2)
    columns=[x,eligible.astype(float),future,past,scalars,np.broadcast_to(mass,(j,o,m)),np.broadcast_to(work,(j,o,m))]
    if hand:
        for state in (rollout(initial(p),p),beam(p,1)):
            labels=np.zeros((j,o,m))
            for a,b,h in state[3]:labels[a,b,h]=1
            columns.append(labels)
    return np.concatenate(columns,axis=2).astype(np.float32),eligible.astype(np.float32)

class WorkflowAssignmentNet(nn.Module):
    def __init__(self,feature_dim,machines=4,hidden=64,layers=4,mp=True):
        super().__init__();self.mp=mp;self.machines=machines
        self.input=nn.Linear(feature_dim,hidden)
        self.layers=nn.ModuleList([nn.Sequential(nn.Linear(hidden*4,hidden),nn.ReLU(),nn.Linear(hidden,hidden)) for _ in range(layers)])
        self.norms=nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.output=nn.Linear(hidden,machines)
    def forward(self,x,eligible):
        h=torch.relu(self.input(x))
        for layer,norm in zip(self.layers,self.norms):
            if self.mp:
                zero=torch.zeros_like(h[:,:,:1]);prev=torch.cat((zero,h[:,:,:-1]),2);nxt=torch.cat((h[:,:,1:],zero),2)
                weight=eligible/(1+x[...,:self.machines])
                resource=torch.einsum('bjom,bjoh->bmh',weight,h)/weight.sum((1,2)).clamp_min(1).unsqueeze(-1)
                back=torch.einsum('bjom,bmh->bjoh',weight,resource)/weight.sum(-1).clamp_min(1).unsqueeze(-1)
            else:
                prev=torch.zeros_like(h);nxt=prev;back=prev
            h=norm(h+layer(torch.cat((h,prev,nxt,back),-1)))
        return self.output(h).masked_fill(eligible==0,-1e9)
