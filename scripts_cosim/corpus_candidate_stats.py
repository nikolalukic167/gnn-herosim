"""Per-dataset shape of a finished co-sim corpus (read-only): tasks per batch, candidates per task slot (distinct (node, platform)
values over the enumerated plans), distinct nodes per slot, plan count, sweep rate from placement_progress.txt.
usage: corpus_candidate_stats.py OUT.jsonl DATASET_ROOT [--workers N]"""
import argparse, json, re
from multiprocessing import Pool
from pathlib import Path


def one(path):
    d = Path(path)
    try:
        meta = json.loads((d / "placement_metadata.json").read_text())
        rung = json.load(open(d / "warm_snapshot.json"))["provenance"]["cell_config"].split("/inputs/")[1].split("/")[0].replace("wf1_", "")
        slots = {}
        nodes = {}
        n = 0
        for l in open(d / "placements" / "placements.jsonl"):
            r = json.loads(l)
            if "placement_plan" not in r:
                continue
            n += 1
            for k, v in r["placement_plan"].items():
                slots.setdefault(k, set()).add(tuple(v)); nodes.setdefault(k, set()).add(v[0])
        rate = None
        m = re.search(r"Rate: ([\d.]+) sim/s", (d / "placement_progress.txt").read_text()) if (d / "placement_progress.txt").exists() else None
        if m:
            rate = float(m.group(1))
        return dict(ds=d.name, rung=rung, complete=meta.get("sweep_complete") is True, n_plans=n, n_tasks=len(slots),
                    cand=[len(slots[k]) for k in sorted(slots, key=int)], cand_nodes=[len(nodes[k]) for k in sorted(nodes, key=int)], rate=rate)
    except Exception as e:
        return dict(ds=d.name, status="error", err=repr(e)[:200])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("root"); ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    ds = sorted(str(p) for p in Path(a.root).glob("ds_*"))
    with Pool(a.workers) as pool, open(a.out, "w") as f:
        for r in pool.imap(one, ds, chunksize=8):
            f.write(json.dumps(r) + "\n")
