"""Read-back audit of the 128-operation broad block proposal screen."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.placement.radical.dag_reservation import problem
from src.placement.radical.dag_reservation_bridge import BridgedDag
from src.placement.radical.environment import pack, serialized
from scripts_cosim.dag_block_stepwise_s0 import scored_pool
from scripts_cosim.dag_reservation_screen import verify_live, verify_plan
from scripts_cosim.mixed_joint_dispatch_screen import ROOT, read, save, sha


def audit(out):
    registration = read(out/'protocol_before_run.json')
    protocol = registration['protocol']
    for name, digest in registration['sources'].items():
        assert sha(ROOT/name) == digest, name
    engine = BridgedDag(out/'audit_build')
    assert engine.provenance == read(out/'native.json')
    seen = set()
    rows = []
    runs = operations = proposals = 0
    for phase, seeds in (('calibration', range(protocol['opened_calibration_seeds'][0], protocol['opened_calibration_seeds'][1]+1)),
                         ('fresh', range(protocol['fresh_seeds'][0], protocol['fresh_seeds'][1]+1))):
        for seed in seeds:
            cell = out/phase/str(seed)
            row = read(cell/'read.json')
            b = problem(seed,*protocol['shape'])
            assert read(cell/'input.json') == serialized(b)
            identity = hashlib.sha256(pack(b).tobytes()+b['predecessors'].tobytes()).hexdigest()
            assert row['identity'] == identity and identity not in seen
            seen.add(identity)
            assert row['shape'] == protocol['shape'] and row['seed'] == seed
            for name in ('strong_start','hand','oracle'):
                verify_plan(engine,b,row[name])
            assert row['strong_start']['time_ms'] <= protocol['common_start_budget_ms']
            assert row['hand']['time_ms'] <= protocol['post_start_budget_ms']
            assert row['hand']['scores'] <= protocol['post_start_exact_evaluation_cap']
            current = row['strong_start']
            for item in row['passes']:
                best, records, size = scored_pool(engine,b,current,protocol['candidate_limit_per_family'])
                assert item['source_cost'] == current['cost'] and item['best_cost'] == best['cost']
                assert item['pool_size'] == size and item['evaluated'] == len(records)
                assert item['candidate_costs'] == records
                proposals += len(records)
                if best['cost'] >= current['cost']:
                    break
                current = best
            assert current['cost'] == row['oracle']['cost']
            for name in ('strong_start','hand','oracle'):
                path = cell/(name+'_live.json')
                if path.exists():
                    verify_live(b,row[name],read(path))
                    assert read(cell/(name+'_replay.json'))['cost'] == row[name]['cost']
                    runs += 1
                    operations += np.size(row[name]['assignment'])
            if phase == 'fresh':
                rows.append(row)
    assert runs == 1+3*len(rows) == read(out/'read.json')['live_runs']
    report = read(out/'read.json')
    gains = [100*(r['strong_start']['cost']-r['oracle']['cost'])/r['strong_start']['cost'] for r in rows]
    captures = [(r['strong_start']['cost']-r['hand']['cost'])/(r['strong_start']['cost']-r['oracle']['cost'])
                if r['strong_start']['cost'] > r['oracle']['cost'] else 1.0 for r in rows]
    assert report['median_oracle_gain_pct'] == float(np.median(gains))
    assert report['median_hand_capture'] == float(np.median(captures))
    assert report['total_exact_block_evaluations'] == sum(sum(p['evaluated'] for p in r['passes']) for r in rows)
    assert report['live_operations'] == operations
    result = {'status':'PASS','unique_workflows':len(seen),'proposal_costs_rechecked':proposals,
              'live_runs':runs,'live_operations':operations,'report_sha256':sha(out/'read.json')}
    save(out/'AUDIT.json',result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    audit(args.out)
