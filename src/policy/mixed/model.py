"""Typed workflow/resource message passing with matched and engineered controls."""
import numpy as np
import torch
from torch import nn
CONTRACT='mixed_execution_assignment_v1'

def features(b,hand=False):
    p=b['p'];j,o,h=p.shape;n=j*o;e=np.isfinite(p).astype(np.float32).reshape(n,h);dur=np.where(np.isfinite(p),p,0)/20
    types=np.eye(4)[b['types']];locks=((b['locks'][...,None]>>np.arange(6))&1).astype(float)
    def suffix(x):return np.flip(np.cumsum(np.flip(x,axis=1),axis=1),axis=1)/o
    columns=[dur,e.reshape(j,o,h),types,locks,suffix(dur),suffix(types),suffix(locks),np.broadcast_to(np.arange(o)[None,:,None]/o,(j,o,1)),np.broadcast_to(b['setup'].reshape(1,1,-1)/20,(j,o,16)),np.broadcast_to(b['domains'].reshape(1,1,-1),(j,o,h*3)),np.broadcast_to(types.sum((0,1))/(j*o),(j,o,4)),np.broadcast_to(locks.sum((0,1))/(j*o),(j,o,6))]
    x=np.concatenate(columns,-1).reshape(n,-1).astype(np.float32);adj=np.zeros((4,n,n),np.float32)
    for job in range(j):
        for op in range(o):
            i=job*o+op
            if op:adj[0,i,i-1]=1
            if op+1<o:adj[1,i,i+1]=1
    l=locks.reshape(n,6);adj[2]=l@l.T;np.fill_diagonal(adj[2],0)
    membership=e@b['domains'];adj[3]=membership@membership.T;np.fill_diagonal(adj[3],0)
    adj/=np.maximum(adj.sum(-1,keepdims=True),1)
    if hand:
        one=adj@x;two=adj@one;x=np.concatenate([x,*one,*two],axis=-1)
    return np.ascontiguousarray(x),e,adj

class MixedNet(nn.Module):
    def __init__(self,feature_dim,hidden=32,layers=2,mp=True):
        super().__init__();self.mp=mp;self.input=nn.Linear(feature_dim,hidden);self.layers=nn.ModuleList([nn.Sequential(nn.Linear(hidden*5,hidden),nn.ReLU(),nn.Linear(hidden,hidden)) for _ in range(layers)]);self.norms=nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)]);self.output=nn.Linear(hidden,4)
    def forward(self,x,e,adj):
        h=torch.relu(self.input(x))
        for layer,norm in zip(self.layers,self.norms):
            messages=torch.matmul(adj,h[:,None]) if self.mp else h.new_zeros((h.shape[0],4,h.shape[1],h.shape[2]))
            h=norm(h+layer(torch.cat([h,*messages.unbind(1)],-1)))
        return self.output(h).masked_fill(e==0,-1e9)
