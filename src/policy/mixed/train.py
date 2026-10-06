"""Matched imitation warmup and complete-plan self-critical training."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from src.policy.workflow.train import seed_everything
from src.policy.mixed.model import MixedNet,features,CONTRACT
from src.placement.radical.mixed import MixedNative
from src.placement.radical.environment import initial
from scripts_cosim.audit_radical_physics import load_problem
from scripts_cosim.radical_physics_screen import sha,save
ROOT=Path(__file__).resolve().parents[3]

def load_split(root,split,hand=False):
    meta=json.loads((root/'METADATA.json').read_text())
    if meta['contract']!=CONTRACT or sha(root/f'{split}.npz')!=meta['files'][f'{split}.npz']:raise ValueError('cache contract/fingerprint mismatch')
    for p,h in meta['sources'].items():
        if sha(ROOT/p)!=h:raise ValueError('data source changed: '+p)
    with np.load(root/f'{split}.npz') as f:data={k:f[k].copy() for k in f.files}
    records={r['seed']:r for r in meta['cases'] if r['split']==split};problems=[]
    for seed in data['seeds']:
        path=root/f'ds_{seed}/problem.json'
        if sha(path)!=records[int(seed)]['source_sha256']:raise ValueError('physical source changed')
        problems.append(load_problem(path))
    return data,problems,torch.tensor(data['xh' if hand else 'x']),torch.tensor(data['eligible']),torch.tensor(data['adj']),torch.tensor(data['labels'])

def refine_predictions(engine,problems,predictions):
    return np.array([engine.search(b,a.reshape(b['p'].shape[:2]),1)[0] for b,a in zip(problems,predictions)])

def train_epoch(model,opt,x,e,adj,y,problems,engine,batch_size,generator,policy_gradient):
    model.train();order=torch.randperm(len(x),generator=generator);total=0.
    for offset in range(0,len(x),batch_size):
        ids=order[offset:offset+batch_size];opt.zero_grad(set_to_none=True);logits=model(x[ids],e[ids],adj[ids]);ce=nn.functional.cross_entropy(logits.flatten(0,1),y[ids].flatten())
        if policy_gradient:
            distribution=torch.distributions.Categorical(logits=logits);sample=distribution.sample();greedy=logits.argmax(-1);bs=[problems[i] for i in ids.tolist()]
            sampled=refine_predictions(engine,bs,sample.detach().numpy());baseline=refine_predictions(engine,bs,greedy.detach().numpy());adv=torch.tensor((sampled-baseline)/np.maximum(baseline,1),dtype=logits.dtype)
            loss=(adv*distribution.log_prob(sample).sum(-1)).mean()+.1*ce
        else:loss=ce
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();total+=float(loss.detach())*len(ids)
    return total/len(x)

def guarded(engine,b,a):
    choices=[a,initial(b,'fastest'),initial(b,'affinity')];best=min(choices,key=lambda p:engine.score(b,p)[0]);return engine.search(b,best,1)

def serve(model,engine,b,hand=False):
    x,e,adj=features(b,hand)
    with torch.inference_mode():a=model(torch.from_numpy(x)[None],torch.from_numpy(e)[None],torch.from_numpy(adj)[None]).argmax(-1)[0].numpy().reshape(b['p'].shape[:2])
    return guarded(engine,b,a)

def validate(model,engine,problems,x,e,adj,y):
    model.eval();pred=[]
    with torch.inference_mode():
        for i in range(0,len(x),32):pred.extend(model(x[i:i+32],e[i:i+32],adj[i:i+32]).argmax(-1).numpy())
    raw=refine_predictions(engine,problems,pred);served=np.array([guarded(engine,b,a.reshape(b['p'].shape[:2]))[0] for b,a in zip(problems,pred)])
    return {'validation_mean_served_rtt_ms':float(served.mean()),'validation_raw_refined_rtt_ms':float(raw.mean()),'validation_accuracy':float(np.mean(np.array(pred)==y.numpy()))}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache-dir',type=Path,required=True);ap.add_argument('--arm',choices=['gnn','mpoff','mlp_hand'],required=True);ap.add_argument('--seed',type=int,default=201);ap.add_argument('--epochs',type=int,default=60);ap.add_argument('--warmup-epochs',type=int,default=15);ap.add_argument('--patience',type=int,default=20);ap.add_argument('--batch-size',type=int,default=32);ap.add_argument('--learning-rate',type=float,default=.001);ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--wandb-project',default='herosim-mixed-physics');args=ap.parse_args()
    torch.set_num_threads(1);seed_everything(args.seed);engine=MixedNative(args.output_dir/'build');train,problems,x,e,adj,y=load_split(args.cache_dir,'train',args.arm=='mlp_hand');val,vproblems,vx,ve,va,vy=load_split(args.cache_dir,'validation',args.arm=='mlp_hand')
    if set(train['seeds'])&set(val['seeds']):raise ValueError('overlapping splits')
    architecture={'feature_dim':x.shape[-1],'hidden':32,'layers':2,'mp':args.arm=='gnn'};model=MixedNet(**architecture);opt=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=1e-4);generator=torch.Generator().manual_seed(args.seed)
    args.output_dir.mkdir(parents=True,exist_ok=True);checkpoint=args.output_dir/f'{args.arm}_seed{args.seed}.pt'
    if checkpoint.exists():raise ValueError('preserve previous checkpoint')
    import wandb
    run=wandb.init(project=args.wandb_project,name=os.environ.get('WANDB_RUN_NAME',f'{args.arm}-{args.seed}'),config={**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'architecture':architecture,'contract':CONTRACT,'chance_accuracy':.5},mode='offline')
    if run is None:raise RuntimeError('missing W&B logging')
    sources={p:sha(ROOT/p) for p in ['src/policy/mixed/train.py','src/policy/mixed/model.py','src/placement/radical/mixed.py','src/placement/radical/mixed.cpp','src/placement/radical/kernel.cpp','src/placement/radical/environment.py']};history=[];best=float('inf');best_epoch=-1;t=time.monotonic()
    try:
        for epoch in range(args.epochs+1):
            loss=None if epoch==0 else train_epoch(model,opt,x,e,adj,y,problems,engine,args.batch_size,generator,epoch>args.warmup_epochs)
            row={'epoch':epoch,**validate(model,engine,vproblems,vx,ve,va,vy)}
            if loss is not None:row['train_loss']=loss
            history.append(row);run.log(row,step=epoch);metric=row['validation_mean_served_rtt_ms']
            if metric<best:
                best=metric;best_epoch=epoch;torch.save(model.state_dict(),checkpoint);save(checkpoint.with_suffix('.contract.json'),{'contract':CONTRACT,'physics':'mixed_execution_v1','arm':args.arm,'seed':args.seed,'architecture':architecture,'selected_epoch':epoch,'selection':'minimum_mean_validation_served_rtt','validation_metric':best,'data_metadata_sha256':sha(args.cache_dir/'METADATA.json'),'sources':sources,'wandb_mode':'offline','wandb_run_dir':run.dir,'torch_seeded':True,'deterministic_algorithms':True,'python_env':{'python':os.sys.version,'torch':torch.__version__,'numpy':np.__version__}})
            if epoch%5==0:print(args.arm,args.seed,row,'best',best_epoch,flush=True)
            if epoch>=args.warmup_epochs+args.patience and epoch-best_epoch>=args.patience:break
        save(checkpoint.with_suffix('.training.json'),{'history':history,'selected_epoch':best_epoch,'selected_metric':best,'wall_s':time.monotonic()-t,'wandb_run_dir':run.dir});run.summary.update({'selected_epoch':best_epoch,'selected_validation_rtt_ms':best})
    finally:run.finish()
if __name__=='__main__':main()
