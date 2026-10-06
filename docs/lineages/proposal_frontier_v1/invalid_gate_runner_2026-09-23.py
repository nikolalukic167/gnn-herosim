"""Run the frozen proposal-frontier live gate, auditing and archiving each seed."""

import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
OUT = Path('/tmp/proposal-frontier-v1')
FREEZE = ROOT / 'docs/lineages/proposal_frontier_v1/gate_freeze_2026-09-23.json'
ARMS = ('gnn', 'mpoff', 'handmulti')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_freeze(freeze):
    for name, expected in freeze['artifacts'].items():
        path = Path(name)
        if digest(path) != expected:
            raise RuntimeError(f'frozen artifact changed: {path}')


def archive(path):
    original_sha = digest(path)
    zipped = Path(str(path) + '.gz')
    temporary = Path(str(zipped) + '.tmp')
    if zipped.exists() or temporary.exists():
        raise RuntimeError(f'archive already exists: {zipped}')
    with path.open('rb') as source, gzip.open(temporary, 'wb', compresslevel=1) as target:
        for chunk in iter(lambda: source.read(1 << 20), b''):
            target.write(chunk)
    h = hashlib.sha256()
    with gzip.open(temporary, 'rb') as source:
        for chunk in iter(lambda: source.read(1 << 20), b''):
            h.update(chunk)
    if h.hexdigest() != original_sha:
        raise RuntimeError(f'archive verification failed: {path}')
    os.replace(temporary, zipped)
    path.unlink()
    return {'path': str(zipped), 'sha256': digest(zipped), 'raw_sha256': original_sha,
            'bytes': zipped.stat().st_size}


def main():
    freeze = json.loads(FREEZE.read_text())
    selected = json.loads(Path('/tmp/proposal_frontier_v1/preflight.json').read_text())['selected']
    if selected != freeze['seeds']:
        raise RuntimeError('selected seeds differ from freeze')
    OUT.mkdir(exist_ok=True)
    for seed in selected:
        verify_freeze(freeze)
        audit_path = OUT / f'seed-{seed}-audit.json'
        if audit_path.exists():
            raise RuntimeError(f'refusing to overwrite existing audit: {audit_path}')
        for arm in ARMS:
            print(f'RUN seed={seed} arm={arm}', flush=True)
            subprocess.run(['bash', 'scripts_cosim/important/run_proposal_frontier_v1.sh',
                            str(seed), arm, str(OUT)], cwd=ROOT, check=True)
        audit = subprocess.run(['pipenv', 'run', 'python3',
                                'scripts_cosim/important/audit_proposal_frontier_v1.py',
                                str(seed), '--root', str(OUT), '--output', str(audit_path)],
                               cwd=ROOT, env={**os.environ, 'PIPENV_IGNORE_VIRTUALENVS': '1',
                                              'VIRTUAL_ENV': '', 'PYTHONPATH': str(ROOT)},
                               capture_output=True, text=True)
        if audit.returncode:
            raise RuntimeError(f'audit failed seed={seed}: {audit.stdout}{audit.stderr}')
        archives = {arm: archive(OUT / f'seed-{seed}-{arm}.json') for arm in ARMS}
        (OUT / f'seed-{seed}-archives.json').write_text(json.dumps(archives, indent=2, sort_keys=True) + '\n')
        print(f'AUDIT PASS seed={seed}', flush=True)


if __name__ == '__main__':
    main()
