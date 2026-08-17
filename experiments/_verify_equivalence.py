import json, subprocess, sys, re
from pathlib import Path
REPO=Path("/root/projects/my-herosim")
S=Path(sys.argv[0]).resolve().parent
PAIRS=[("experiments/contention_v2_gnn.yaml","src/notebooks/train_near_rtt_v2_contention_v2_dim14_ce_only.py"),
       ("experiments/contention_v3_gnn.yaml","src/notebooks/train_near_rtt_v2_contention_v3_dim14_ce_only.py"),
       ("experiments/contention_v2_mlp.yaml","src/notebooks/train_mlp_contention_v2_dim22_batchcache.py"),
       ("experiments/contention_v3_mlp.yaml","src/notebooks/train_mlp_contention_v3_dim22_batchcache.py")]
ok=True
for cfg, wrapper in PAIRS:
    old=json.loads(subprocess.check_output([sys.executable,str(S/"_verify_equivalence_capture.py"),wrapper],cwd=REPO,text=True))
    new=subprocess.check_output([sys.executable,"run_experiment.py",cfg,"--dry-run"],cwd=REPO,text=True)
    # parse dry-run
    nenv={}; nargv=None
    sec=None
    for line in new.split("\n"):
        if line.startswith("env:"): sec="env"; continue
        if line.startswith("argv:"): sec="argv"; continue
        if not line.startswith("    "): sec=None; continue
        t=line.strip()
        if sec=="env" and "=" in t and not t.startswith("(unset)"):
            k,v=t.split("=",1); nenv[k]=v
        elif sec=="argv": nargv=t.split(" ")
    nenv={k:v for k,v in nenv.items() if not k.startswith("HEROSIM_")}
    # normalise the cfg: tag we deliberately add
    if "WANDB_TAGS" in nenv:
        nenv["WANDB_TAGS"]=re.sub(r",cfg:[^,]+$","",nenv["WANDB_TAGS"])
    oenv=old["env"]
    # MLP wrapper puts model path in argv --output; old wrapper env has no MLP_ vars set
    ediff={k:(oenv.get(k),nenv.get(k)) for k in set(oenv)|set(nenv) if oenv.get(k)!=nenv.get(k)}
    adiff = old["argv"]!=nargv
    status = "MATCH" if not ediff and not adiff else "DIFF"
    if status=="DIFF": ok=False
    print(f"\n=== {cfg}  vs  {Path(wrapper).name}  ->  {status}")
    if ediff:
        print("  env differences:")
        for k,(o,n) in sorted(ediff.items()): print(f"    {k}\n      old={o}\n      new={n}")
    if adiff:
        print("  argv old:", " ".join(old["argv"]))
        print("  argv new:", " ".join(nargv or []))
print("\nOVERALL:", "ALL EQUIVALENT" if ok else "DIFFERENCES FOUND")
