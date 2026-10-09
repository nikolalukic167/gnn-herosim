"""Re-evaluate plans of existing fidelity-replay datasets with the current replay code and compare with the stored labels.
usage: relabel_plans.py OUT.json --datasets LISTFILE [--max-plans 400] [--workers 24]
Per dataset: every plan if it has <= max-plans, else the best 60 % by stored label plus a seeded random remainder."""
import argparse, json, os, random, sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, os.getcwd())


def one(args):
    ds, max_plans = args
    ds = Path(ds)
    try:
        os.environ["HEROSIM_SNAPSHOT_FIDELITY"] = "1"
        prov = json.load(open(ds / "generation_provenance.json"))
        for k, v in (prov.get("physics_env") or {}).items():
            if k.startswith("HEROSIM_") or k in ("GNN_DECODE_MODE", "GNN_BATCH_BY_PEER_GROUP", "COSIM_AUTOSCALER_RECONCILE_INTERVAL"):
                os.environ.setdefault(k, str(v))
        from src.placement import fidelity_replay
        seed = json.load(open(ds / "infrastructure.json"))["live_snapshot_seed"]
        spec = dict(seed["fidelity_replay"])
        if not Path(spec["sim_input"]).exists():
            spec["sim_input"] = "data/nofs-ids"
        types = [next(iter(e["application"]["dag"])) for e in json.load(open(ds / "workload.json"))["events"]]
        offered = {t: [(sp["node_name"], sp["platform_id"]) for sp in specs if sp.get("candidate", True)]
                   for t, specs in seed["replicas_by_type"].items()}
        fr = fidelity_replay.FidelityReplay(spec, types, offered)
        plans = [r for r in (json.loads(l) for l in open(ds / "placements" / "placements.jsonl")) if "placement_plan" in r]
        order = sorted(range(len(plans)), key=lambda i: plans[i]["rtt"])
        if len(plans) > max_plans:
            head = order[: int(0.6 * max_plans)]
            rest = order[int(0.6 * max_plans):]
            random.Random(hash(ds.name) % 1000).shuffle(rest)
            idx = head + rest[: max_plans - len(head)]
        else:
            idx = order
        rows = []
        for i in idx:
            plan = {int(k): tuple(v) for k, v in plans[i]["placement_plan"].items()}
            res = fr.cut_to_batch(fr.run(plan))
            rows.append(dict(plan_idx=i, old=plans[i]["rtt"], new=float(res["stats"]["total_rtt"])))
        return dict(ds=str(ds), n_plans=len(plans), rows=rows)
    except Exception as e:  # recorded, never swallowed
        return dict(ds=str(ds), error="%s: %s" % (type(e).__name__, str(e)[:300]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--datasets", required=True)
    ap.add_argument("--max-plans", type=int, default=400); ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    dss = [l.strip() for l in open(a.datasets) if l.strip()]
    with Pool(a.workers) as p:
        res = p.map(one, [(d, a.max_plans) for d in dss], chunksize=1)
    json.dump(res, open(a.out, "w"))
    for r in res:
        if "error" in r:
            print("ERROR", r["ds"], r["error"]); continue
        rows = r["rows"]; d = [abs(x["new"] - x["old"]) / x["old"] for x in rows]
        ao = min(rows, key=lambda x: (x["old"], x["plan_idx"])); an = min(rows, key=lambda x: (x["new"], x["plan_idx"]))
        print("%s plans %d/%d  changed(>0.1%%) %d  max rel %.3f%%  argmin old=%d new=%d%s" % (
            Path(r["ds"]).name, len(rows), r["n_plans"], sum(1 for x in d if x > 1e-3), 100 * max(d), ao["plan_idx"], an["plan_idx"],
            "  CHANGED" if abs(an["new"] - ao["new"]) / ao["new"] > 1e-9 and an["plan_idx"] != ao["plan_idx"] else ""))
