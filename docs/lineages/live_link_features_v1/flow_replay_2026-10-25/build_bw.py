import json, os, sys
from pathlib import Path
summ = json.load(open(sys.argv[2]))
for k, v in summ["env"].items():
    os.environ[k] = str(v)
sys.path.insert(0, sys.argv[4])
from src.executesimulation import prepare_infrastructure_for_real_simulation
from src.placement.network_fabric import link_key, directed_bandwidth
cfg = json.load(open(sys.argv[1]))
infra = prepare_infrastructure_for_real_simulation(cfg, seed=None, sim_input_path=Path(sys.argv[4]) / "data" / "nofs-ids")
lt = infra["link_topology"]
out = {}
for key, a in lt["links"].items():
    l, _, r = key.partition("|")
    for x, y in ((l, r), (r, l)):
        out[f"{x}>{y}"] = directed_bandwidth(a, x, y) if a.get("access_node") is not None else float(a["bandwidth_mbps"])
json.dump(out, open(sys.argv[3], "w"))
print(len(lt["links"]), "links;", "classes", sorted(set(json.dumps(v["class"]) for v in (lt.get("access_classes") or {}).values())))
print("ac match summary:", {k: v["class"] for k, v in (lt.get("access_classes") or {}).items()} == {k: v for k, v in summ["accessClasses"].items()} if "accessClasses" in summ else "n/a")
