"""Reconcile cached model choices with actual final plans and incumbent guards."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from src.policy.pair_selector.model import PairNet
from src.policy.pair_selector.train import load_split,predict
from scripts_cosim.radical_physics_screen import save,sha
from scripts_cosim.evaluate_workflow import paired_summary

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache-dir',type=Path,required=True);ap.add_argument('--models',type=Path,required=True);ap.add_argument('--gate',type=Path,required=True);args=ap.parse_args();torch.set_num_threads(1)
    frozen=json.loads((args.gate/'frozen_before_test.json').read_text());result={};reference=[];rule=[]
    for arm in ('gnn','mpoff','mlp_hand'):
        data,_,x,adj,pair,_=load_split(args.cache_dir,'test',arm=='mlp_hand')
        for seed in (301,302,303):
            name=f'{arm}_seed{seed}';path=args.models/f'{name}.pt';side=json.loads(path.with_suffix('.contract.json').read_text())
            if sha(path)!=frozen['models'][name]['weights_sha256']:raise ValueError('changed weights')
            model=PairNet(**side['architecture']);model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True);indices=predict(model,x,adj,pair)
            unchanged=improved=raw_worse=0;choices=[]
            for i,case in enumerate(data['seeds']):
                r=json.loads((args.gate/'test'/str(case)/'read.json').read_text());cost=float(data['costs'][i,indices[i]])
                if cost!=r['methods'][name]['cost']:raise ValueError('cache/live selection mismatch')
                records=[json.loads(line) for line in (args.cache_dir/f'ds_{case}/placements/placements.jsonl').read_text().splitlines()]
                chosen=records[int(indices[i])];inc=records[0]
                plan=chosen['placement_plan'] if chosen['objective']<inc['objective'] else inc['placement_plan']
                if plan!=r['methods'][name]['plan']:raise ValueError('prediction/guard plan mismatch')
                unchanged+=int(plan==inc['placement_plan']);improved+=int(cost<inc['objective']);raw_worse+=int(chosen['objective']>inc['objective'])
                choices.append({'seed':int(case),'candidate':int(indices[i]),'raw_cost':chosen['objective'],'served_cost':cost,'incumbent_cost':inc['objective']})
                if arm=='gnn' and seed==301:reference.append(r['methods']['oracle']['cost']);rule.append(r['methods']['graph1']['cost'])
            result[name]={'unchanged_final_plans':unchanged,'improved_vs_incumbent':improved,'raw_proposals_worse_than_incumbent':raw_worse,'explicit_no_move':int((indices==0).sum()),'optimal_candidate_fraction':float(np.mean(data['costs'][np.arange(len(indices)),indices]==data['costs'].min(-1))),'choices':choices}
    report={'status':'PASS','models':result,'oracle_vs_rule':paired_summary(np.array([reference]),np.array([rule])),'diagnostic_source_sha256':sha(Path(__file__))}
    save(args.gate/'selection_diagnostic.json',report)
    print(json.dumps({'status':'PASS','models':{k:{a:b for a,b in v.items() if a!='choices'} for k,v in result.items()},'oracle_vs_rule':report['oracle_vs_rule']},indent=2))
if __name__=='__main__':main()
