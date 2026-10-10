"""NEAR_RTT_CE_GRAPH_WEIGHT checks on the REAL accel cache and split (the tiny smoke cache of tests/test_trainer_determinism.py exists on no machine).
  python3 accel_stakes_checks.py run <repo> <out.pt> [stakes]   1-epoch physmp g3 run (seed 4242) through run_experiment.py in <repo>; flag on when 'stakes' is given; saves the weights
  python3 accel_stakes_checks.py compare a.pt b.pt              bit-identical or not
<repo>/models must be a real directory (the newest non -final .pt in it is taken); W&B is disabled (a check, not a training run)."""
import subprocess, sys, tempfile, os
from pathlib import Path
import torch, yaml

if sys.argv[1] == "compare":
    a, b = torch.load(sys.argv[2]), torch.load(sys.argv[3])
    nd = sum(1 for k in a if k in b and not torch.equal(a[k], b[k]))
    same = set(a) == set(b) and nd == 0
    print(f"{sys.argv[2]} vs {sys.argv[3]}: {'BIT-IDENTICAL' if same else 'DIFFER'} ({len(a)} tensors, {nd} differ)")
    sys.exit(0 if same else 1)
repo = Path(sys.argv[2]); out = sys.argv[3]; stakes = len(sys.argv) > 4 and sys.argv[4] == "stakes"
assert not (repo / "models").is_symlink(), "models/ must be a real directory"
cfg = yaml.safe_load((repo / "experiments/accel_replica_v1_gnn_eng_physmp_g3.yaml").read_text())
cfg["args"].update({"epochs": 1, "min-epochs": 1, "patience": 1}); cfg.pop("wandb", None)
cfg["env"].pop("NEAR_RTT_SAVE_FINAL", None)
if stakes:
    cfg["env"]["NEAR_RTT_CE_GRAPH_WEIGHT"] = "stakes"
before = {p for p in (repo / "models").glob("*.pt")}
with tempfile.TemporaryDirectory() as d:
    c = Path(d) / "chk.yaml"; c.write_text(yaml.dump(cfg))
    env = dict(os.environ, WANDB_MODE="disabled", PYTHONPATH=str(repo))
    r = subprocess.run([sys.executable, "-u", "run_experiment.py", str(c), "--seed", "4242"], cwd=repo, env=env)
    assert r.returncode == 0, f"run_experiment exited {r.returncode}"
new = sorted((p for p in (repo / "models").glob("*.pt") if p not in before and "-final" not in p.name), key=lambda p: p.stat().st_mtime)
assert new, "no checkpoint written"
ck = torch.load(str(new[-1]), map_location="cpu", weights_only=False)
w = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
w = {k: v for k, v in w.items() if torch.is_tensor(v)}
torch.save(w, out); print("saved", out, len(w), "tensors", "stakes" if stakes else "off", "from", new[-1].name)
for p in list(new) + [p.with_suffix(".contract.json") for p in new] + [p.with_suffix(".val.json") for p in new]:
    if p.exists(): p.unlink()
