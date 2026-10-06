"""Symmetric pair scoring with matched message-passing and hand-graph controls."""
import numpy as np
import torch
from torch import nn
from src.policy.mixed.model import features as base_features
CONTRACT='mixed_pair_selector_v1'

def features(b,a,ends,hand=False):
    x,eligible,adj=base_features(b)
    n=len(x);hosts=a.ravel();ops=a.shape[1]
    x=np.concatenate([x,np.eye(b['p'].shape[2],dtype=np.float32)[hosts],
                      np.asarray(ends,dtype=np.float32).reshape(n,1)/500],-1)
    if hand:
        one=adj@x;two=adj@one;x=np.concatenate([x,*one,*two],-1)
    i,j=np.triu_indices(n,1);locks=b['locks'].ravel()
    shared=np.zeros(len(i),dtype=np.float32)
    for bit in range(6):shared+=((locks[i]&locks[j])>>bit)&1
    pair=np.column_stack([shared/6,(i//ops==j//ops),(hosts[i]==hosts[j]),np.abs(i%ops-j%ops)/ops]).astype(np.float32)
    return np.ascontiguousarray(x,dtype=np.float32),adj,pair

class PairNet(nn.Module):
    def __init__(self,feature_dim,hidden=16,layers=2,mp=True,nodes=48):
        super().__init__();self.mp=mp;self.input=nn.Linear(feature_dim,hidden)
        self.layers=nn.ModuleList([nn.Sequential(nn.Linear(hidden*5,hidden),nn.ReLU(),nn.Linear(hidden,hidden)) for _ in range(layers)])
        self.norms=nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.head=nn.Sequential(nn.Linear(hidden*3+4,hidden),nn.ReLU(),nn.Linear(hidden,1))
        self.none=nn.Linear(hidden,1)
        self.register_buffer('pairs',torch.triu_indices(nodes,nodes,1),persistent=False)
    def forward(self,x,adj,pair):
        h=torch.relu(self.input(x))
        for layer,norm in zip(self.layers,self.norms):
            messages=adj@h[:,None] if self.mp else h.new_zeros((len(h),4,h.shape[1],h.shape[2]))
            h=norm(h+layer(torch.cat([h,*messages.unbind(1)],-1)))
        left,right=h[:,self.pairs[0]],h[:,self.pairs[1]]
        scores=self.head(torch.cat([left+right,torch.abs(left-right),left*right,pair],-1)).squeeze(-1)
        return torch.cat([self.none(h.mean(1)),scores],-1)
