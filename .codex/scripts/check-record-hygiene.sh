#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
PIPENV_IGNORE_VIRTUALENVS=1 VIRTUAL_ENV= pipenv run python3 -m pytest tests/test_record_hygiene.py -q "$@"
