"""Capture env+argv an old wrapper would hand the trainer, without training."""
import os, sys, runpy, json
from pathlib import Path
REPO = Path("/root/projects/my-herosim")
wrapper = sys.argv[1]
before = dict(os.environ)
captured = {}
real = runpy.run_path
def fake(path, run_name=None, **kw):
    captured["trainer"] = path
    captured["argv"] = list(sys.argv)
    captured["env"] = {k: v for k, v in os.environ.items()
                       if k.startswith(("NEAR_RTT_", "WANDB_", "MLP_", "TRAIN_"))}
    captured["unset"] = [k for k in before
                         if k not in os.environ and k.startswith(("NEAR_RTT_","WANDB_","TRAIN_"))]
    raise SystemExit(0)
runpy.run_path = fake
try:
    real(str(REPO / wrapper), run_name="__main__")
except SystemExit:
    pass
runpy.run_path = real
print(json.dumps(captured, indent=1, sort_keys=True))
