#!/usr/bin/env python3
"""FL MP-OFF twin (uncut_groups_s0_v1, registered 2026-10-19), the parity test required before any twin cell.

Loads each checkpoint through the live serving loader (src.executesimulation.load_gnn_model) with the serving environment the gate gives
its kind (GNN_DISABLE_MESSAGE_PASSING=1 for the MP-OFF twin, unset for FL gnn; the loader refuses a sidecar that disagrees), encodes one
fixed synthetic graph, then permutes the features of every task except task 0 among themselves and encodes again. A model without message
passing must give task 0 and every platform the same embedding bit for bit (each node sees only its own features); a model with message
passing must not (task 0 and the platforms see the permuted neighbours). This catches a twin served with message passing on, the
2026-09-06 MP-OFF loader hole, and an FL checkpoint served with it off. Exit 1 on any expectation that fails.

  mp_twin_parity.py --mp-on models/uncut-comb-fl-gnn-eng-seed1-final.pt [...] --mp-off models/uncut-comb-fl-twin-seed1-final.pt [...] --out parity.json
"""
import argparse
import hashlib
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# the gate's serving environment for a learned RA cell (fresh_topo_burst_v1_gate.py: GNN_DECODE_MODE masked_topo, peer-group batches, prefix
# alpha key inf, the v6 partial-state contract with load and exchange seconds)
SERVE_ENV = {"PARTIAL_STATE_CONTRACT": "partial_state_v6", "PARTIAL_STATE_LOAD_SECONDS": "1", "PARTIAL_STATE_EXCHANGE_SECONDS": "1",
             "GNN_DECODE_MODE": "masked_topo", "GNN_BATCH_BY_PEER_GROUP": "1", "GNN_PREFIX_ALPHA_KEY": "inf"}
N_TASKS, N_PLATFORMS = 6, 5


def graph(model, seed=0):
    """A fixed bipartite graph sized to the model: every task a candidate of every platform (both directions, edge_attr row-aligned), a peer
    ring over the tasks touching task 0, a 4-way one-hot."""
    from torch_geometric.data import Data

    g = torch.Generator().manual_seed(seed)
    oh = int(getattr(model, "task_type_onehot_dim", 0) or 0)
    tdim = int(model.task_encoder.net[0].in_features) - oh
    pdim = int(model.platform_encoder.net[0].in_features)
    edim = int(model.edge_scorer.fc1.in_features) - 2 * int(model.embedding_dim) - int(getattr(model, "partial_state_edge_dim", 0) or 0)
    src, dst = [], []
    for t in range(N_TASKS):
        for p in range(N_PLATFORMS):
            src += [t, N_TASKS + p]
            dst += [N_TASKS + p, t]
    ea = torch.rand((N_TASKS * N_PLATFORMS, edim), generator=g)
    peer = [(t, (t + 1) % N_TASKS) for t in range(N_TASKS)]
    peer += [(b, a) for a, b in peer]
    d = Data(task_features=torch.randn((N_TASKS, tdim), generator=g), platform_features=torch.randn((N_PLATFORMS, pdim), generator=g),
             edge_index=torch.tensor([src, dst]), edge_attr=ea.repeat_interleave(2, dim=0),
             peer_edge_index=torch.tensor(peer).t().contiguous(), peer_edge_attr=torch.rand((len(peer), int(getattr(model, "peer_edge_dim", 1))), generator=g),
             node_edge_index=torch.empty((2, 0), dtype=torch.long))
    d.n_tasks, d.n_platforms = N_TASKS, N_PLATFORMS
    if oh:
        d.task_type_onehot4 = torch.nn.functional.one_hot(torch.arange(N_TASKS) % oh, oh).float()
    return d


def permuted(d):
    """Task 0 fixed; tasks 1..n-1 rotated by one (a derangement), features and one-hot together."""
    e = d.clone()
    perm = torch.tensor([0] + [1 + (i + 1) % (N_TASKS - 1) for i in range(N_TASKS - 1)])
    e.task_features = d.task_features[perm]
    if hasattr(d, "task_type_onehot4"):
        e.task_type_onehot4 = d.task_type_onehot4[perm]
    return e


def probe(model):
    with torch.no_grad():
        d = graph(model)
        t0, p0 = model._encode(d)
        t1, p1 = model._encode(permuted(d))
    return {"task0_max_abs_diff": float((t0[0] - t1[0]).abs().max()), "platform_max_abs_diff": float((p0 - p1).abs().max()),
            "disable_mp": bool(getattr(model, "_disable_mp", False))}


PHYSICS_ENV_VARS = {"transfer_model": "HEROSIM_TRANSFER_MODEL", "replica_release": "HEROSIM_REPLICA_RELEASE", "scaleout": "HEROSIM_SCALEOUT",
                    "inflight_capture": "HEROSIM_INFLIGHT_CAPTURE", "warmth_physics": "HEROSIM_WARMTH_PHYSICS"}


def physics_from_sidecar(path):
    """Serve under the physics the checkpoint records (the gate's cells must match it too, or the loader refuses them): the five environment
    switches, and the replica placement rule through replica_rule.set_rule. A sidecar without physics_env fails loud."""
    from src.placement import replica_rule

    with open(path[:-3] + ".contract.json") as fh:
        pe = json.load(fh).get("physics_env")
    if not isinstance(pe, dict) or any(k not in pe for k in (*PHYSICS_ENV_VARS, "replica_placement_rule")):
        raise SystemExit(f"FAIL LOUD: {path} sidecar has no complete physics_env ({pe!r})")
    for k, var in PHYSICS_ENV_VARS.items():
        os.environ[var] = str(pe[k])
    replica_rule.set_rule(pe["replica_placement_rule"])
    with open(path[:-3] + ".contract.json") as fh:
        slate = json.load(fh).get("candidate_slate")
    if not slate:
        raise SystemExit(f"FAIL LOUD: {path} sidecar records no candidate_slate")
    os.environ["GNN_SERVE_CANDIDATE_SLATE"] = slate  # as the gate serves it (ra_serving_slate: the sidecar's rule)
    return pe


def load(path, mp_off):
    from src.executesimulation import load_gnn_model

    os.environ.update(SERVE_ENV)
    physics_from_sidecar(str(path))
    if mp_off:
        os.environ["GNN_DISABLE_MESSAGE_PASSING"] = "1"
    else:
        os.environ.pop("GNN_DISABLE_MESSAGE_PASSING", None)
    model, _ = load_gnn_model(__import__("pathlib").Path(path))
    model.eval()
    return model


def check(name, r, mp_off):
    """MP off: both diffs exactly 0. MP on: task 0 or a platform moves."""
    if mp_off:
        ok = r["disable_mp"] and r["task0_max_abs_diff"] == 0.0 and r["platform_max_abs_diff"] == 0.0
    else:
        ok = (not r["disable_mp"]) and (r["task0_max_abs_diff"] > 0.0 or r["platform_max_abs_diff"] > 0.0)
    print(f"{'PASS' if ok else 'FAIL'} {name}: {'MP off' if mp_off else 'MP on'} {r}")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mp-on", nargs="+", required=True, help="FL gnn checkpoints (served with message passing)")
    ap.add_argument("--mp-off", nargs="+", required=True, help="MP-OFF twin checkpoints (served with GNN_DISABLE_MESSAGE_PASSING=1)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    res, ok = {}, True
    for paths, mp_off in ((a.mp_on, False), (a.mp_off, True)):
        for p in paths:
            r = probe(load(p, mp_off))
            r["md5"] = hashlib.md5(open(p, "rb").read()).hexdigest()
            res[p] = r
            ok &= check(os.path.basename(p), r, mp_off)
    res["_verdict"] = "PARITY OK" if ok else "PARITY FAILED"
    if os.path.exists(a.out):
        raise SystemExit(f"FAIL LOUD: {a.out} exists")
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    print(res["_verdict"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
