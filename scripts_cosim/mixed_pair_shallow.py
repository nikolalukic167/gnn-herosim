"""Cheaper pair-selector follow-up, reusing the frozen experiment and live harness."""
import argparse
from pathlib import Path
import numpy as np
from src.placement.radical.environment import initial
from src.placement.radical.pairs import PairNative
import scripts_cosim.mixed_pair_exploration as harness

class ShallowPairs(PairNative):
    def incumbent(self,b):
        return self.search(b,initial(b,'affinity'),2)

    def candidates(self,b,start,pairs,rounds=1):
        return super().candidates(b,start,pairs,rounds)

    def plan(self,b,method):
        if method=='anneal256':return self.search(b,initial(b,'affinity'),256,True)
        cost,a=self.incumbent(b)
        if method=='local':return cost,a
        i,j=np.triu_indices(a.size,1)
        if method.startswith('graph'):
            locks=b['locks'].ravel();hosts=a.ravel();ops=a.shape[1]
            shared=np.zeros(len(i),dtype=int)
            for bit in range(6):shared+=((locks[i]&locks[j])>>bit)&1
            order=np.lexsort((j,i,-(hosts[i]==hosts[j]).astype(int),-(i//ops==j//ops).astype(int),-shared))
            count=int(method[5:])
        elif method.startswith('random'):
            order=np.random.default_rng(77).permutation(len(i));count=int(method[6:])
        else:raise ValueError('unknown shallow control')
        pairs=np.column_stack((i[order[:count]],j[order[:count]]))
        costs,plans=self.candidates(b,a,pairs);idx=int(costs.argmin())
        return (float(costs[idx]),plans[idx]) if costs[idx]<cost else (cost,a)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['screen','live']);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    harness.PROTOCOL='experiments/mixed_pair_shallow_v1.json'
    harness.SOURCES=[*harness.SOURCES,harness.PROTOCOL,'scripts_cosim/mixed_pair_shallow.py']
    harness.PairNative=ShallowPairs
    (harness.screen if args.mode=='screen' else harness.live)(args.out)
