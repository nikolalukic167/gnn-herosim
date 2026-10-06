"""Run the registered nine-model CPU dispatch pilot with isolated offline logs."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
base = ROOT / "simulation_data/gnn_environment_search_v1"
parser = argparse.ArgumentParser()
parser.add_argument("--log-subdir", default="mixed_dispatch_training_logs")
arguments = parser.parse_args()
logs = base / arguments.log_subdir
if logs.exists():
    raise ValueError("preserve earlier training logs")
logs.mkdir(parents=True)
for name in ("wandb", "wandb-cache", "wandb-config", "wandb-data"):
    (logs / name).mkdir()
environment = {
    **os.environ,
    "WANDB_DIR": str(logs / "wandb"),
    "WANDB_CACHE_DIR": str(logs / "wandb-cache"),
    "WANDB_CONFIG_DIR": str(logs / "wandb-config"),
    "WANDB_DATA_DIR": str(logs / "wandb-data"),
    "WANDB_MODE": "offline",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}
validation = json.loads((base / "mixed_dispatch_corpus/VALIDATION.json").read_text())
if validation["status"] != "PASS":
    raise RuntimeError("dataset validation failed")
for arm in ("gnn", "mpoff", "mlp_hand"):
    for seed in (401, 402, 403):
        print("TRAIN", arm, seed, flush=True)
        path = logs / f"{arm}_{seed}.log"
        with path.open("w") as stream:
            subprocess.run(
                [sys.executable, str(ROOT / "run_experiment.py"),
                 str(ROOT / f"experiments/mixed_dispatch_learning_v1_{arm}.yaml"),
                 "--seed", str(seed)],
                cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True,
            )
        report = json.loads((base / "mixed_dispatch_models" / f"{arm}_seed{seed}.training.json").read_text())
        print("DONE", arm, seed, "epoch", report["selected_epoch"],
              "validation", report["selected_metric"], "seconds", round(report["wall_s"], 1), flush=True)
