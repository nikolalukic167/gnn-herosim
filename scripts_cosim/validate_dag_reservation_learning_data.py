"""Independently validate the DAG proposal corpus and searched labels."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from src.placement.radical.dag_reservation import problem
from src.placement.radical.dag_reservation_bridge import BridgedDag
from src.placement.radical.environment import pack, serialized
from src.policy.dag_reservation.model import features, CONTRACT
from scripts_cosim.dag_reservation_screen import verify_plan
from scripts_cosim.mixed_joint_dispatch_screen import ROOT, read, save, sha


def validate(root):
    pre = read(root / 'protocol_before_run.json'); meta = read(root / 'METADATA.json')
    for name, digest in pre['sources'].items():
        if sha(ROOT / name) != digest: raise ValueError('source changed: ' + name)
    if meta['contract'] != CONTRACT or meta['sources'] != pre['sources']:
        raise ValueError('metadata contract mismatch')
    engine = BridgedDag(root / 'build')
    if engine.provenance != read(root / 'native.json') or sha(root / 'native.json') != meta['native_sha256']:
        raise ValueError('native provenance mismatch')
    for name, digest in meta['files'].items():
        if sha(root / name) != digest: raise ValueError('array fingerprint mismatch')
    identities = set(); case_counts = {'train': 0, 'validation': 0, 'test': 0}; samples = {}
    for split in ('train', 'validation'):
        with np.load(root / f'{split}.npz') as z: samples[split] = {k: z[k].copy() for k in z.files}
    for row in meta['cases']:
        split, seed = row['split'], row['seed']; cell = root / split / str(seed)
        b = problem(seed, *meta['shape'])
        if serialized(b) != read(cell / 'input.json') or sha(cell / 'input.json') != row['input_sha256']:
            raise ValueError('input mismatch')
        identity = hashlib.sha256(pack(b).tobytes() + b['predecessors'].tobytes()).hexdigest()
        if identity != row['identity'] or identity in identities: raise ValueError('duplicate or changed input')
        identities.add(identity); i = case_counts[split]; case_counts[split] += 1
        if split == 'test':
            if (cell / 'labels.json').exists() or (cell / 'placements/placements.jsonl').exists():
                raise ValueError('sealed test labels opened')
            continue
        labels = read(cell / 'labels.json')
        if sha(cell / 'labels.json') != row['labels_sha256'] or sha(cell / 'placements/placements.jsonl') != row['placements_sha256']:
            raise ValueError('label fingerprint mismatch')
        candidates = [labels['baseline'], *labels['candidates']]
        if labels['best']['cost'] != min(p['cost'] for p in candidates): raise ValueError('teacher selection mismatch')
        for p in candidates: verify_plan(engine, b, p)
        trace = [json.loads(line) for line in (cell / 'placements/placements.jsonl').read_text().splitlines()]
        if len(trace) != 3 or [p['cost'] for p in trace] != [p['cost'] for p in candidates]:
            raise ValueError('searched-plan trace mismatch')
        data = samples[split]
        if data['seeds'][i] != seed: raise ValueError('array seed mismatch')
        a, rank = np.array(labels['baseline']['assignment']), np.array(labels['baseline']['priority'])
        x, graph, eligible = features(b, a, rank)
        for key, actual in [('x', x), ('graph', graph), ('eligible', eligible)]:
            if not np.array_equal(data[key][i], actual): raise ValueError('feature mismatch: ' + key)
        if not np.array_equal(data['assignment'][i], np.array(labels['best']['assignment']).ravel()):
            raise ValueError('teacher assignment mismatch')
        starts = np.array(labels['best']['starts']).ravel()
        priority = np.array(labels['best']['priority']).ravel()
        order = np.lexsort((priority, starts)); expected = np.empty(len(order), dtype=np.float32)
        expected[order] = np.arange(len(order)) / max(len(order) - 1, 1)
        if not np.array_equal(data['rank_target'][i], expected):
            raise ValueError('teacher order mismatch')
    if case_counts != meta['counts'] or case_counts['test'] != 16:
        raise ValueError('split coverage mismatch')
    result = {'status': 'PASS', 'split_counts': case_counts, 'unique_inputs': len(identities),
              'metadata_sha256': sha(root / 'METADATA.json'), 'validator_sha256': sha(Path(__file__))}
    save(root / 'VALIDATION.json', result); print(json.dumps(result, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--root', type=Path, required=True)
    validate(ap.parse_args().root)
