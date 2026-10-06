"""Config-driven finite pair-selector pilot with exhaustive guarded-cost targets."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
import torch
from src.policy.workflow.train import seed_everything
from src.policy.pair_selector.model import PairNet,features,CONTRACT
from src.placement.radical.pairs import PairNative
from scripts_cosim.audit_radical_physics import load_problem
from scripts_cosim.radical_physics_screen import sha,save
ROOT=Path(__file__).resolve().parents[3]

def load_split(root,split,hand=False):
    meta=json.loads((root/'METADATA.json').read_text())
    if meta['contract']!=CONTRACT or sha(root/f'{split}.npz')!=meta['files'][f'{split}.npz']:raise ValueError('cache mismatch')
    if any(sha(ROOT/p)!=h for p,h in meta['sources'].items()):raise ValueError('data source changed')
    with np.load(root/f'{split}.npz') as f:data={k:f[k].copy() for k in f.files}
    records={r['seed']:r for r in meta['cases'] if r['split']==split};problems=[]
    for seed in data['seeds']:
        path=root/f'ds_{seed}/problem.json'
        if sha(path)!=records[int(seed)]['sha256']:raise ValueError('physical input changed')
        problems.append(load_problem(path))
    return data,problems,torch.tensor(data['xh' if hand else 'x']),torch.tensor(data['adj']),torch.tensor(data['pair']),torch.tensor(data['costs'])

def train_epoch(model,opt,x,adj,pair,costs,generator,batch_size=16):
    model.train();order=torch.randperm(len(x),generator=generator);total=0.
    for ids in order.split(batch_size):
        logits=model(x[ids],adj[ids],pair[ids]);target=torch.softmax(-(costs[ids]-costs[ids].min(-1,keepdim=True).values)/(.01*costs[ids,0,None]),-1)
        loss=-(target*torch.log_softmax(logits,-1)).sum(-1).mean()
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();total+=float(loss.detach())*len(ids)
    return total/len(x)

def predict(model,x,adj,pair):
    model.eval()
    with torch.inference_mode():return torch.cat([model(x[i:i+16],adj[i:i+16],pair[i:i+16]).argmax(-1) for i in range(0,len(x),16)]).numpy()

def serve(model,engine,b,hand=False):
    c,a=engine.incumbent(b);_,ends=engine.score(b,a);x,adj,pair=features(b,a,ends,hand)
    model.eval()
    with torch.inference_mode():idx=int(model(torch.from_numpy(x)[None],torch.from_numpy(adj)[None],torch.from_numpy(pair)[None]).argmax(-1)[0])
    if idx==0:return c,a
    indices=model.pairs[:,idx-1].cpu().numpy()[None]
    values,plans=engine.candidates(b,a,indices)
    return (float(values[0]),plans[0]) if values[0]<c else (c,a)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache-dir',type=Path,required=True);ap.add_argument('--arm',choices=['gnn','mpoff','mlp_hand'],required=True);ap.add_argument('--seed',type=int,default=301);ap.add_argument('--epochs',type=int,default=30);ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--wandb-project',default='herosim-pair-selector');args=ap.parse_args()
    torch.set_num_threads(1);seed_everything(args.seed)
    train,_,x,adj,pair,costs=load_split(args.cache_dir,'train',args.arm=='mlp_hand');val,_,vx,va,vp,vc=load_split(args.cache_dir,'validation',args.arm=='mlp_hand')
    if set(train['seeds'])&set(val['seeds']):raise ValueError('split overlap')
    architecture={'feature_dim':x.shape[-1],'hidden':16,'layers':2,'mp':args.arm=='gnn','nodes':x.shape[1]}
    model=PairNet(**architecture);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4);generator=torch.Generator().manual_seed(args.seed)
    args.output_dir.mkdir(parents=True,exist_ok=True);checkpoint=args.output_dir/f'{args.arm}_seed{args.seed}.pt'
    if checkpoint.exists():raise ValueError('preserve checkpoint')
    sources={p:sha(ROOT/p) for p in ['src/policy/pair_selector/train.py','src/policy/pair_selector/model.py','run_experiment.py']}
    import wandb
    run=wandb.init(project=args.wandb_project,name=f'pair-{args.arm}-{args.seed}',mode='offline',config={'arm':args.arm,'seed':args.seed,'contract':CONTRACT,'architecture':architecture,'epochs':args.epochs,'uniform_ce_floor':float(np.log(costs.shape[1]))})
    if run is None:raise RuntimeError('missing W&B')
    history=[];best=float('inf');selected=-1;started=time.monotonic()
    try:
        for epoch in range(args.epochs+1):
            loss=None if epoch==0 else train_epoch(model,opt,x,adj,pair,costs,generator)
            ids=predict(model,vx,va,vp);values=val['costs'][np.arange(len(ids)),ids];metric=float(values.mean())
            row={'epoch':epoch,'validation_mean_served_rtt_ms':metric,'validation_oracle_rtt_ms':float(val['costs'].min(-1).mean()),'validation_rule_rtt_ms':float(val['rule'].mean()),'validation_optimal_fraction':float(np.mean(values==val['costs'].min(-1))), 'uniform_choice_oracle_fraction':float(np.mean(val['costs']==val['costs'].min(-1,keepdims=True)))}
            if loss is not None:row['train_loss']=loss
            run.log(row,step=epoch);history.append(row)
            if metric<best:
                best=metric;selected=epoch;torch.save(model.state_dict(),checkpoint)
                save(checkpoint.with_suffix('.contract.json'),{'contract':CONTRACT,'physics':'mixed_execution_v1','arm':args.arm,'seed':args.seed,'architecture':architecture,'selected_epoch':epoch,'selection':'minimum_validation_mean_guarded_candidate_cost','data_metadata_sha256':sha(args.cache_dir/'METADATA.json'),'sources':sources,'wandb_mode':'offline','wandb_run_dir':run.dir,'torch_seeded':True,'deterministic_algorithms':True})
            if epoch%5==0:print(args.arm,args.seed,row,flush=True)
        save(checkpoint.with_suffix('.training.json'),{'history':history,'selected_epoch':selected,'selected_metric':best,'wall_s':time.monotonic()-started,'wandb_run_dir':run.dir})
        run.summary.update({'selected_epoch':selected,'selected_validation_rtt_ms':best})
    finally:run.finish()
if __name__=='__main__':main()
