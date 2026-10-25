import json, sys, glob
from collections import defaultdict
for f in sorted(glob.glob("*.flows.jsonl")):
    t = f[5:10]; bw = json.load(open(f"bw_{t}.json")); s = json.load(open(f.replace(".flows.jsonl", ".summary.json")))
    T = s["arrival_end"]["last_arrival_s"]; off = defaultdict(float)
    for line in open(f):
        r = json.loads(line)
        if r[0] <= T:
            for k in r[3]: off[k] += r[4]
    rho = sorted(((v / 2**20 / bw[k] / T, k) for k, v in off.items()), reverse=True)
    up = [x for x in rho if x[1].startswith("node") and ">core" in x[1]]
    print(f[:58], "| top uplink", up[0][1], f"{bw[up[0][1]]:.1f} MB/s rho {up[0][0]:.2f} | uplinks rho>1:", sum(r > 1 for r, _ in up), "| any link rho>1:", sum(r > 1 for r, _ in rho))
