#!/usr/bin/env python3
"""r1_attribution_v1 preparation: run the UNMODIFIED corpus generator on a throwaway grid built from the peer-affinity
c3 x200 preset (four task types, peer exchange, server mesh, backbone) at the calibration topology seeds, to see what
the pipeline records. Registers the grid in-process only; nothing in the repo's presets changes.

  wf1_corpus_smoke.py <generate_gnn_datasets_fast.py arguments...>   (--grid wf1_corpus_smoke)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_gnn_datasets_fast as G  # noqa: E402

base = G.PEER_AFFINITY_SCREEN_C3_X200_GRID
G.GRID_PRESETS["wf1_corpus_smoke"] = {**base, "seeds": [9601, 9607], "default_output_subdir": "wf1_corpus_smoke"}

if __name__ == "__main__":
    sys.exit(G.main())
