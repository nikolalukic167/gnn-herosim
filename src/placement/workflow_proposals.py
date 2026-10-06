"""Incumbent-preserving complete-plan search with interchangeable proposal weights."""
import numpy as np
from src.placement.workflow_planning import initial,rollout,evaluate


def hand_probabilities(p,temperature=None):
    eligible=np.isfinite(p)
    if temperature is None:
        return eligible/eligible.sum(-1,keepdims=True)
    weight=np.exp(-(p-p.min(-1,keepdims=True))/temperature)
    return weight/weight.sum(-1,keepdims=True)


def improve(p,probabilities,moves,seed):
    if probabilities.shape!=p.shape or not np.isfinite(probabilities).all():
        raise ValueError('invalid proposal probabilities')
    if not np.all(np.isfinite(p).sum(-1)==2):
        raise ValueError('proposal contract requires two eligible hosts per operation')
    if np.any(probabilities<0) or np.any(probabilities[~np.isfinite(p)]!=0) or not np.allclose(probabilities.sum(-1),1):
        raise ValueError('probabilities must cover only eligible hosts and sum to one')
    state=rollout(initial(p),p)
    best=np.zeros(p.shape[:2],dtype=int)
    for j,k,h in state[3]:best[j,k]=h
    score=state[2]
    greedy=probabilities.argmax(-1);cost=evaluate(p,greedy)
    if cost<score:score,best=cost,greedy.copy()
    rng=np.random.default_rng(seed)
    allowed=np.argsort(p,axis=-1)[...,:2].reshape(-1,2)
    flatprob=probabilities.reshape(-1,p.shape[-1]);positions=np.arange(len(allowed))
    sizes=(1,1,2,4)
    for step in range(moves):
        flat=best.ravel();alternate=np.where(flat==allowed[:,0],allowed[:,1],allowed[:,0])
        weights=flatprob[positions,alternate]+1e-6;weights=weights/weights.sum()
        selected=rng.choice(len(flat),size=min(sizes[step%4],len(flat)),replace=False,p=weights)
        candidate=best.copy();candidate.ravel()[selected]=alternate[selected]
        value=evaluate(p,candidate)
        if value<score:score,best=value,candidate
    return float(score),best
