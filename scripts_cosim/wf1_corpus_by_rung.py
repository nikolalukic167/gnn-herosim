"""r1_attribution_v1: the corpus' pause-line quantities split by rung. For every `gnn_datasets_wf1_{train,heldout}` under --root: per rung the
single-node zero-exchange share of the optima, the plan-count distribution, and from the warm summaries the offered / rejected counts, the
no_choice share, the unplaced_partner rate and the sub-batched share of batches. The rung is the third field of the source tag
(wf1p_cc40s<topology>_<rung>_<window>)."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from scripts_cosim.wf1_corpus_optima_read import dataset_row, summarise
from scripts_cosim.wf1_manifest import read_manifests


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    rows = defaultdict(list)
    counts = defaultdict(lambda: defaultdict(int))
    for split in ("train", "heldout"):
        base = a.root / f"gnn_datasets_wf1_{split}"
        if not base.is_dir():
            continue
        for f in sorted(base.glob("warm_summary_*.json")):
            s = json.loads(f.read_text())
            c = counts[s["source_tag"].split("_")[2]]
            for k in ("offered", "rejected", "batches_made", "no_choice", "single_candidate_node", "disconnected_batch",
                      "no_peer_pairs", "unplaced_partner", "discarded"):
                c[k] += int(s.get(k, 0) or 0)
        batches = defaultdict(set)
        for m in read_manifests(base)[0]:
            if "dataset_id" not in m:
                continue
            rung = m["source_tag"].split("_")[2]
            batches[rung].add((m["source_tag"], m["snapshot_id"]))
            row = dataset_row(base / m["dataset_id"], split)
            if row is not None:
                row["sub_batch_of"] = (m.get("slate") or {}).get("of", 1)
                row["batch"] = (m["source_tag"], m["snapshot_id"])
                rows[rung].append(row)
    out = {}
    for rung in sorted(set(rows) | set(counts)):
        c = dict(counts[rung])
        n_batches = len({r["batch"] for r in rows[rung]})
        sub = len({r["batch"] for r in rows[rung] if r["sub_batch_of"] > 1})
        off = c.get("offered", 0)
        out[rung] = {"optima": summarise(rows[rung]), "counts": c,
                     "batches": n_batches, "sub_batched_batches": sub,
                     "sub_batched_share": sub / n_batches if n_batches else None,
                     "no_choice_share_of_offered": c.get("no_choice", 0) / off if off else None,
                     "unplaced_partner_share_of_offered": c.get("unplaced_partner", 0) / off if off else None}
    print(json.dumps(out, indent=1))
    if a.out:
        a.out.write_text(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
