#!/bin/bash
set -uo pipefail
eval "$(micromamba shell hook --shell bash)" && micromamba activate gnn
cd /share/nikola.lukic/uncut_groups/anomaly && python3 request_stats.py
