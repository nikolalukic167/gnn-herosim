"""NEAR_RTT_CE_GRAPH_WEIGHT checks, through the REAL run_experiment.py -> train_near_rtt.py path on the tiny smoke cache (tests/test_trainer_determinism.py harness).
  python3 accel_stakes_checks.py <repo> <out.pt> [stakes]   run seed 4242 once in <repo> (flag on when 'stakes' is given), save the weights
  python3 accel_stakes_checks.py compare a.pt b.pt          print whether the two weight sets are bit-identical"""
import sys, tempfile
from pathlib import Path
import torch

if sys.argv[1] == "compare":
    a, b = torch.load(sys.argv[2]), torch.load(sys.argv[3])
    same = set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)
    print(f"{sys.argv[2]} vs {sys.argv[3]}: {'BIT-IDENTICAL' if same else 'DIFFER'} ({len(a)} tensors, {sum(1 for k in a if k in b and not torch.equal(a[k], b[k]))} differ)")
    sys.exit(0 if same else 1)
repo = Path(sys.argv[1]); sys.path.insert(0, str(repo)); sys.path.insert(0, str(repo / "tests"))
import test_trainer_determinism as T
env = {"NEAR_RTT_CE_GRAPH_WEIGHT": "stakes"} if len(sys.argv) > 3 and sys.argv[3] == "stakes" else {}
with tempfile.TemporaryDirectory() as d:
    w = T._run_a1_via_run_experiment(4242, Path(d), env_extra=env)
torch.save(w, sys.argv[2]); print("saved", sys.argv[2], len(w), "tensors", "flag=" + str(env))
