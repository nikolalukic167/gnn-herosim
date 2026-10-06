"""Run nine config-driven small pair-selector models, with local offline W&B."""
import os,subprocess,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'simulation_data/gnn_environment_search_v1';logs=base/'mixed_pair_training_logs'
if logs.exists():raise ValueError('preserve earlier logs')
logs.mkdir()
for name in ('wandb','wandb-cache','wandb-config','wandb-data'):(logs/name).mkdir()
env={**os.environ,'WANDB_DIR':str(logs/'wandb'),'WANDB_CACHE_DIR':str(logs/'wandb-cache'),'WANDB_CONFIG_DIR':str(logs/'wandb-config'),'WANDB_DATA_DIR':str(logs/'wandb-data'),'WANDB_MODE':'offline','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
validation=json.loads((base/'mixed_pair_corpus/VALIDATION.json').read_text())
if validation['status']!='PASS':raise ValueError('unvalidated data')
for arm in ('gnn','mpoff','mlp_hand'):
 for seed in (301,302,303):
  print('TRAIN',arm,seed,flush=True)
  with (logs/f'{arm}_{seed}.log').open('w') as log:
   subprocess.run([sys.executable,str(ROOT/'run_experiment.py'),str(ROOT/f'experiments/mixed_pair_learning_v1_{arm}.yaml'),'--seed',str(seed)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  r=json.loads((base/'mixed_pair_models'/f'{arm}_seed{seed}.training.json').read_text())
  print('DONE',arm,seed,'selected',r['selected_epoch'],'validation',r['selected_metric'],'seconds',round(r['wall_s'],1),flush=True)
