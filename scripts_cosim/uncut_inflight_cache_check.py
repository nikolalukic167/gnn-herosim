#!/usr/bin/env python3
"""partial_state_v6 cache check: against the partial_state_v5 cache built from the same states, every graph (matched by dataset id) must be identical except platform_features,
whose first W columns equal the v5 ones and whose last INFLIGHT_DIM columns equal an independent recomputation from the stored state (warm_snapshot.json ghosts) with the shared
functions; and the metadata must differ only in the contract. Fails loud on anything else."""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src" / "notebooks"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--view", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from src.placement.inflight_features import INFLIGHT_DIM, state_from_ghosts

    old_g, old_ids = pickle.load(open(Path(a.old) / "graphs.pkl", "rb")), pickle.load(open(Path(a.old) / "dataset_ids.pkl", "rb"))
    new_g, new_ids = pickle.load(open(Path(a.new) / "graphs.pkl", "rb")), pickle.load(open(Path(a.new) / "dataset_ids.pkl", "rb"))
    old_by = {str(i): g for g, i in zip(old_g, old_ids)}
    mo, mn = json.load(open(Path(a.old) / "metadata.json")), json.load(open(Path(a.new) / "metadata.json"))
    meta_diff = {k: [mo.get(k), mn.get(k)] for k in set(mo) | set(mn) if mo.get(k) != mn.get(k) and k not in ("dataset_ids", "num_graphs", "num_datasets", "num_combo_datasets", "base_dirs", "num_rtt_combo_rows", "statistics", "rtt_hash_chunks_meta_rel_path")}
    if set(meta_diff) != {"partial_state_contract"} or meta_diff["partial_state_contract"] != ["partial_state_v5", "partial_state_v6"]:
        raise SystemExit(f"FAIL LOUD: metadata differs in more than the contract: {meta_diff}")
    task_types = json.load(open(REPO / "data" / "nofs-ids" / "task-types.json"))
    n_ok, n_nonzero, max_cnt, max_work = 0, 0, 0.0, 0.0
    for g, did in zip(new_g, new_ids):
        o = old_by.get(str(did))
        if o is None:
            raise SystemExit(f"FAIL LOUD: {did} is not in the v5 cache")
        pf_o, pf_n = o.platform_features, g.platform_features
        if pf_n.shape[1] != pf_o.shape[1] + INFLIGHT_DIM or pf_n.shape[0] != pf_o.shape[0]:
            raise SystemExit(f"FAIL LOUD: {did}: platform_features {tuple(pf_n.shape)} vs v5 {tuple(pf_o.shape)}")
        if not torch.equal(pf_n[:, :-INFLIGHT_DIM], pf_o):
            raise SystemExit(f"FAIL LOUD: {did}: the first {pf_o.shape[1]} platform columns differ from the v5 cache")
        for k in sorted(set(o.keys()) | set(g.keys())):
            if k == "platform_features":
                continue
            vo, vn = getattr(o, k, None), getattr(g, k, None)
            same = torch.equal(vo, vn) if isinstance(vo, torch.Tensor) and isinstance(vn, torch.Tensor) else vo == vn
            if not same and not (isinstance(vo, np.ndarray) and np.array_equal(vo, vn)):
                raise SystemExit(f"FAIL LOUD: {did}: attribute {k} differs from the v5 cache")
        # independent recomputation of the two columns from the stored state
        base = Path(a.view) / str(did)
        snap = json.load(open(base / "warm_snapshot.json"))["snapshot"]
        by_q = {}
        for gh in snap["fidelity"]["ghosts"]:
            by_q.setdefault(str(gh["q"]), []).append(gh)
        queue = {k: int(v) for k, v in snap["full_queue_snapshot"].items()}
        meta = g.queue_key_to_platform_meta
        want = np.zeros((pf_n.shape[0], INFLIGHT_DIM))
        for pos, (key, m) in enumerate(meta.items()):          # the metadata is built in platform-position order
            cnt, work = state_from_ghosts(by_q.get(key, ()), queue.get(key, 0), task_types, str(m["platform_type"]))
            want[pos] = [cnt, work]
        got = pf_n[:, -INFLIGHT_DIM:].double().numpy()
        # the count column is count / queue_norm (the batch's divisor): recover the divisor from the queue column's own scale and compare the work column exactly
        if not np.allclose(got[:, 1], np.log1p(np.maximum(want[:, 1], 0.0)), rtol=0, atol=1e-6):
            raise SystemExit(f"FAIL LOUD: {did}: work column differs from the recomputation (max abs {np.abs(got[:, 1] - np.log1p(want[:, 1])).max():.3g})")
        if want[:, 0].max() > 0 and (got[:, 0] <= 0).sum() != (want[:, 0] <= 0).sum():
            raise SystemExit(f"FAIL LOUD: {did}: count column zeros differ from the recomputation")
        if want[:, 0].max() > 0:
            ratio = got[want[:, 0] > 0, 0] / want[want[:, 0] > 0, 0]
            if not np.allclose(ratio, ratio[0], rtol=1e-6, atol=0):
                raise SystemExit(f"FAIL LOUD: {did}: count column is not a single divisor times the count")
        n_ok += 1
        n_nonzero += int(want[:, 0].max() > 0)
        max_cnt, max_work = max(max_cnt, float(want[:, 0].max())), max(max_work, float(want[:, 1].max()))
    res = {"graphs_checked": n_ok, "graphs_with_inflight_work": n_nonzero, "metadata_diff": meta_diff, "max_count": max_cnt, "max_work_seconds": max_work}
    print(json.dumps(res, indent=1))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
