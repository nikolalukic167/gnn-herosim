"""Run the registered config-driven nine-model CPU pilot with isolated offline logs."""
import argparse,os,subprocess,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--log-subdir',default='mixed_physics_training_logs');args=ap.parse_args()
base=ROOT/'simulation_data/gnn_environment_search_v1';logs=base/args.log_subdir;logs.mkdir(parents=True,exist_ok=True)
for name in ('wandb','wandb-cache','wandb-config','wandb-data'):(logs/name).mkdir(exist_ok=True)
env={**os.environ,'WANDB_DIR':str(logs/'wandb'),'WANDB_CACHE_DIR':str(logs/'wandb-cache'),'WANDB_CONFIG_DIR':str(logs/'wandb-config'),'WANDB_DATA_DIR':str(logs/'wandb-data'),'WANDB_MODE':'offline','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
validation=json.loads((base/'mixed_physics_corpus/VALIDATION.json').read_text())
if validation['status']!='PASS':raise RuntimeError('dataset not validated')
for arm in ('gnn','mpoff','mlp_hand'):
 for seed in (201,202,203):
  path=logs/f'{arm}_{seed}.log'
  if path.exists():raise ValueError('preserve previous training log')
  print('TRAIN',arm,seed,flush=True)
  with path.open('w') as log:subprocess.run([sys.executable,str(ROOT/'run_experiment.py'),str(ROOT/f'experiments/mixed_physics_learning_v1_{arm}.yaml'),'--seed',str(seed)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  report=json.loads((base/'mixed_physics_models'/f'{arm}_seed{seed}.training.json').read_text());print('DONE',arm,seed,'epoch',report['selected_epoch'],'validation',report['selected_metric'],'seconds',round(report['wall_s'],1),flush=True)
