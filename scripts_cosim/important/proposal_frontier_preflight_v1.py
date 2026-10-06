"""Select fresh fixed-replica topologies using coverage alone."""

import hashlib
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
OUT = Path("/tmp/proposal_frontier_v1")
BASE = ROOT / "experiments/gnn_seeded_cd_fixed_v1_base.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    base = json.loads(BASE.read_text())
    env = os.environ.copy()
    env.update({
        "PIPENV_IGNORE_VIRTUALENVS": "1", "VIRTUAL_ENV": "", "PYTHONPATH": str(ROOT),
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONHASHSEED": "0",
        "HEROSIM_PG_EXCHANGE_SCALE": "4", "HEROSIM_PG_CD_PASSES": "6",
        "HEROSIM_PEER_EXCHANGE": "1", "HEROSIM_SERVER_ONLY_REPLICAS": "1",
        "HEROSIM_WARMTH_PHYSICS": "node_disk_v2", "GNN_DECODE_MODE": "masked_topo",
        "GNN_BATCH_BY_PEER_GROUP": "1", "GNN_PREFIX_ALPHA_KEY": "inf",
        "QUEUE_FEATURE_CONTRACT": "legacy_v0", "GNN_QUEUE_NORM_MODE": "scheduler_adaptive",
        "HEROSIM_GNN_DEVICE": "cpu", "FIXED_REPLICA_PREFLIGHT": "1",
    })
    selected, rejected = [], []
    for seed in range(10301, 10341):
        config = json.loads(json.dumps(base))
        config["network"]["topology"]["seed"] = seed
        config["scheduler"]["batch_size"] = 32
        config["gate_cell"] = {
            "lineage": "proposal_frontier_v1", "phase": "sealed_gate", "topology_seed": seed,
        }
        cfg = Path(f"/tmp/fixed-g32-config-{seed}.json")
        if cfg.exists():
            raise RuntimeError(f"refusing to overwrite {cfg}")
        cfg.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        env["FIXED_REPLICA_MANIFEST_PATH"] = str(OUT / f"preflight-{seed}-manifest.json")
        cmd = ["pipenv", "run", "python3", "/tmp/fixed_g32_runner.py",
               "--config", str(cfg), "--workload", "/tmp/fixed-g32-weighted_bridge-5120.json",
               "--policy", "peer_greedy_network_cd", "--queue-length", "100",
               "--output", str(OUT / f"preflight-{seed}.json")]
        result = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
        log = OUT / f"preflight-{seed}.log"
        log.write_text(result.stdout + result.stderr)
        if result.returncode == 0:
            selected.append(seed)
            print("ELIGIBLE", seed, flush=True)
        elif "fixed-replica pool does not cover every source" in log.read_text():
            rejected.append(seed)
            print("INELIGIBLE", seed, flush=True)
        else:
            raise RuntimeError(f"unexpected preflight failure for {seed}: {log}")
        (OUT / "preflight.json").write_text(json.dumps({
            "selected": selected, "rejected": rejected, "base_sha256": digest(BASE),
            "workload_sha256": digest(Path("/tmp/fixed-g32-weighted_bridge-5120.json")),
            "config_sha256": {str(s): digest(Path(f"/tmp/fixed-g32-config-{s}.json")) for s in selected},
        }, indent=2, sort_keys=True) + "\n")
        if len(selected) == 8:
            break
    if len(selected) != 8:
        raise RuntimeError(f"only {len(selected)} eligible seeds")


if __name__ == "__main__":
    main()
