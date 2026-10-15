"""Byte-identity proof for a gate-kind addition: dump what the gate lists for every EXISTING kind, from one repo tree.

python3 uncut_gate_listing_proof.py <repo_root> <new_kind>[,<new_kind>...] > out.json

Imports scripts_cosim/fresh_topo_burst_v1_gate.py from <repo_root>, sets the r1a env, and prints one JSON object: R1A_ARMS, RA_KINDS, RA_MP_OFF,
R1A_NOSPLIT with the new kinds (and their _nosplit / _kr forms) removed, and the r1a_tasks cell listing for the default grid and for each existing
learned kind named alone, again without cells of the new kinds. Run it on the old and the new tree: the two outputs must be byte-identical.
"""
import json
import os
import sys

root, new = sys.argv[1], [k for k in sys.argv[2].split(",") if k]
sys.path.insert(0, root)
os.environ.update({"HEROSIM_TRANSFER_MODEL": "pipelined", "HEROSIM_REPLICA_RELEASE": "1", "HEROSIM_SCALEOUT": "kpa",
                   "GATE_FIXED_POLICY_TIME_SCALE": "1.0", "HEROSIM_SHARED_AUTOSCALER": "0"})
os.environ.setdefault("WF1_RUNGS", os.environ.get("WF1_RUNGS", "moderate,heavy"))
from scripts_cosim import fresh_topo_burst_v1_gate as G  # noqa: E402

drop = set(new) | {f"{k}_nosplit" for k in new} | {f"{k}_nosplit_kr" for k in new} | {f"{k}_nosplit_k100" for k in new}


def keep(seq):
    return [k for k in seq if k not in drop]


sel = {"topologies": list(range(19537, 19549))}
out = {"R1A_ARMS": keep(G.R1A_ARMS), "RA_KINDS": keep(G.RA_KINDS), "RA_MP_OFF": keep(G.RA_MP_OFF), "R1A_NOSPLIT": keep(G.R1A_NOSPLIT)}
listings = {}
for name, arms in [("default", "")] + [(k, k) for k in keep(G.R1A_NOSPLIT)] + [(k + "_kr", k + "_kr") for k in keep(G.R1A_NOSPLIT) if k.endswith("_nosplit") and "_arm1" in k]:
    for k_env in ("R1A_ARMS",):
        os.environ.pop(k_env, None)
        if arms:
            os.environ[k_env] = arms
        try:
            cells = G.r1a_tasks(sel)
        except SystemExit as e:
            cells = f"SystemExit: {e}".replace(root, "<root>")
        listings[name] = [c for c in cells if c["kind"] not in drop] if isinstance(cells, list) else cells
out["listings"] = listings
json.dump(out, sys.stdout, sort_keys=True, indent=1)
