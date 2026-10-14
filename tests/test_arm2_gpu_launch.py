"""The GPU variant of the K=2 GNN run changes the device and the W&B identity, nothing else."""
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def _load(name):
    return yaml.safe_load((REPO / "experiments" / name).read_text())


def test_gpu_yaml_differs_from_cpu_yaml_only_in_run_name_and_tags():
    cpu, gpu = _load("uncut_arm2_gnn_eng.yaml"), _load("uncut_arm2_gnn_eng_gpu.yaml")
    assert gpu["wandb"]["run_name"] == "uncut-arm2-gnn-eng-gpu" != cpu["wandb"]["run_name"]
    assert gpu["wandb"]["tags"][:len(cpu["wandb"]["tags"])] == cpu["wandb"]["tags"]
    assert set(gpu["wandb"]["tags"]) - set(cpu["wandb"]["tags"]) == {"device-l40s", "gpu-variant"}
    for c, g in ((cpu, gpu),):
        c, g = dict(c), dict(g)
        cw, gw = c.pop("wandb"), g.pop("wandb")
        assert c == g
        assert {k: v for k, v in cw.items() if k not in ("run_name", "tags")} == {k: v for k, v in gw.items() if k not in ("run_name", "tags")}


def test_gpu_sbatch_is_gpu_resume_capable_and_guarded():
    s = (REPO / "scripts_cosim/datalab/uncut_arm2_gpu_train.sbatch").read_text()
    assert "--partition=GPU-l40s" in s and "--gres=gpu:l40s:1" in s
    assert 'CUDA_VISIBLE_DEVICES=""' not in s
    assert "NEAR_RTT_RESUME=1" in s and "TIMEOUT) ;;" in s
    assert "OMP_NUM_THREADS=6" in s and "uncut_${TAG}_${ARM}_gpu.yaml" in s
    assert "--exclusive" not in s.replace("not --exclusive", "")
