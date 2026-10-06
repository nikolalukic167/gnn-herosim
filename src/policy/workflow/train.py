"""Reusable supervised trainer for complete-workflow assignment models."""
import argparse,copy,hashlib,json,os,random,time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from torch import nn
from src.policy.workflow.model import WorkflowAssignmentNet,CONTRACT
from src.placement.workflow_planning import evaluate


def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_split(root,split,hand=False):
    meta=json.loads((root/'METADATA.json').read_text())
    if meta['contract']!=CONTRACT or sha(root/f'{split}.npz')!=meta['files'][f'{split}.npz']:raise ValueError('dataset contract or fingerprint mismatch')
    with np.load(root/f'{split}.npz') as z:data={k:z[k].copy() for k in z.files}
    x=data['xh' if hand else 'x'];e=data['eligible'];labels=data['labels']
    if not np.isfinite(x).all() or not np.all(np.take_along_axis(e,labels[...,None],axis=-1)==1):raise ValueError('invalid features or teacher choices')
    return data,x,e,labels

def train_epoch(model,optimizer,x,e,y,batch_size,generator):
    model.train();order=torch.randperm(len(x),generator=generator);total=0.
    for start in range(0,len(x),batch_size):
        idx=order[start:start+batch_size].to(x.device);optimizer.zero_grad(set_to_none=True)
        logits=model(x[idx],e[idx]);loss=nn.functional.cross_entropy(logits.flatten(0,2),y[idx].flatten())
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();total+=float(loss.detach())*len(idx)
    return total/len(x)

def predict(model,x,e,batch_size=64):
    model.eval();out=[]
    with torch.no_grad():
        for start in range(0,len(x),batch_size):out.append(model(x[start:start+batch_size],e[start:start+batch_size]).argmax(-1).cpu().numpy())
    return np.concatenate(out)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--cache-dir',type=Path,required=True);ap.add_argument('--arm',choices=['gnn','mpoff','mlp_hand'],required=True)
    ap.add_argument('--epochs',type=int,default=150);ap.add_argument('--patience',type=int,default=40);ap.add_argument('--learning-rate',type=float,default=.001);ap.add_argument('--batch-size',type=int,default=32);ap.add_argument('--hidden',type=int,default=64);ap.add_argument('--layers',type=int,default=4);ap.add_argument('--device',default='cuda');ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--seed',type=int,default=101);ap.add_argument('--wandb-project',default='herosim-workflow-amortized');ap.add_argument('--wandb-mode',choices=['offline','online'],default='offline')
    args=ap.parse_args();seed_everything(args.seed);torch.set_num_threads(1)
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('requested GPU unavailable')
    train,x,e,y=load_split(args.cache_dir,'train',args.arm=='mlp_hand');val,vx,ve,vy=load_split(args.cache_dir,'validation',args.arm=='mlp_hand')
    if set(train['seeds'])&set(val['seeds']):raise ValueError('train/validation overlap')
    device=torch.device(args.device);x=torch.tensor(x,device=device);e=torch.tensor(e,device=device);y=torch.tensor(y,device=device);vx=torch.tensor(vx,device=device);ve=torch.tensor(ve,device=device)
    arch={'feature_dim':x.shape[-1],'machines':e.shape[-1],'hidden':args.hidden,'layers':args.layers,'mp':args.arm=='gnn'}
    model=WorkflowAssignmentNet(**arch).to(device);opt=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=1e-4);generator=torch.Generator().manual_seed(args.seed)
    import wandb
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    run=wandb.init(project=args.wandb_project,name=os.environ.get('WANDB_RUN_NAME',f'{args.arm}-seed{args.seed}'),tags=os.environ.get('WANDB_TAGS','').split(','),config={**config,'contract':CONTRACT,'architecture':arch,'chance_accuracy':.5},mode=args.wandb_mode)
    if run is None:raise RuntimeError('WandB run not initialized')
    args.output_dir.mkdir(parents=True,exist_ok=True);checkpoint=args.output_dir/f'{args.arm}_seed{args.seed}.pt'
    if checkpoint.exists():raise ValueError('checkpoint exists; preserve previous run')
    best=float('inf');best_epoch=-1;history=[];started=time.monotonic()
    try:
        for epoch in range(args.epochs+1):
            loss=None if epoch==0 else train_epoch(model,opt,x,e,y,args.batch_size,generator)
            predictions=predict(model,vx,ve);costs=np.array([evaluate(p,a) for p,a in zip(val['p'],predictions)]);metric=float(costs.mean())
            row={'epoch':epoch,'validation_mean_rtt_ms':metric,'validation_teacher_regret_pct':float(np.mean(costs/val['teacher']-1)*100),'validation_accuracy':float((predictions==vy).mean())}
            if loss is not None:row['train_ce']=loss
            history.append(row);wandb.log(row,step=epoch)
            if metric<best:
                best=metric;best_epoch=epoch;torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},checkpoint)
                sidecar={"torch_seeded":True,"deterministic_algorithms":True,'contract':CONTRACT,'arm':args.arm,'seed':args.seed,'architecture':arch,'selected_epoch':epoch,'selection':'minimum_mean_complete_validation_RTT','validation_mean_rtt_ms':metric,'data_metadata_sha256':sha(args.cache_dir/'METADATA.json'),'wandb_run_url':run.url if args.wandb_mode=='online' else None,'wandb_mode':args.wandb_mode,'wandb_run_dir':run.dir,'python_env':{'python':os.sys.version,'torch':torch.__version__,'numpy':np.__version__},'sources':{str(Path(p).relative_to(Path(__file__).resolve().parents[3])):sha(p) for p in [Path(__file__),Path(__file__).with_name('model.py'),Path(__file__).resolve().parents[2]/'placement/workflow_planning.py']}}
                checkpoint.with_suffix('.contract.json').write_text(json.dumps(sidecar,indent=2)+'\n')
            if epoch%10==0:print(args.arm,args.seed,row,'best_epoch',best_epoch,flush=True)
            if epoch-best_epoch>=args.patience:break
        report={'arm':args.arm,'seed':args.seed,'best_epoch':best_epoch,'best_validation_rtt_ms':best,'history':history,'wall_s':time.monotonic()-started,'checkpoint':str(checkpoint),'wandb_run_url':run.url if args.wandb_mode=='online' else None,'wandb_mode':args.wandb_mode,'wandb_run_dir':run.dir}
        checkpoint.with_suffix('.training.json').write_text(json.dumps(report,indent=2)+'\n');wandb.summary.update({'best_epoch':best_epoch,'best_validation_rtt_ms':best})
    finally:wandb.finish()
if __name__=='__main__':main()
