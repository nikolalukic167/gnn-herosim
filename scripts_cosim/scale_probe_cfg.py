"""Scale probe (2026-10-09, descriptive): mint one cell config from a production x1 cell config with the node counts, the
client->server connection probability and the topology seed changed; nothing else moves. Named cc40s<seed>.json because
workload_fix_v1_build.py / cd_gap_v1_build_b.py glob that name; the counts live in the file.
usage: scale_probe_cfg.py BASE_CFG OUT_DIR --clients N --servers M --p P --seed S"""
import argparse, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("base"); ap.add_argument("out")
ap.add_argument("--clients", type=int, required=True); ap.add_argument("--servers", type=int, required=True)
ap.add_argument("--p", type=float, required=True); ap.add_argument("--seed", type=int, required=True)
a = ap.parse_args()
cfg = json.loads(Path(a.base).read_text())
cfg.pop("warm_snapshot", None)
cfg["nodes"]["client_nodes"]["count"] = a.clients
cfg["nodes"]["server_nodes"]["count"] = a.servers
cfg["network"]["topology"]["connection_probability"] = a.p
cfg["network"]["topology"]["seed"] = a.seed
cfg["scale_probe"] = {"base": str(a.base), "clients": a.clients, "servers": a.servers, "connection_probability": a.p}
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
(out / f"cc40s{a.seed}.json").write_text(json.dumps(cfg, indent=1))
print(out / f"cc40s{a.seed}.json")
