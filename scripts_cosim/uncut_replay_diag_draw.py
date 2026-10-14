"""Draw the groups of the Arm 1b replay diagnostic BEFORE any output is read (spec: docs/lineages/uncut_groups_s0_v1.md, "Arm 1b replay diagnostic").

Reads only the group metadata (split artifact, the first line of each val group's placements.jsonl: label_source, n_tasks, r), never a replay or a decode.
  train: 300 groups, uniform without replacement from the split's train ids (seed string below)
  val:   all heldout groups
  ILS subsample (LABEL-SEARCH): 50 ILS-labelled val groups in each of 11-50 / 51-100 / 101-256 (all of them if fewer, n reported); a bucket with fewer than 40
  val groups is topped up to 40 from the ILS-labelled TRAIN groups of that bucket (a label property is tested, so a train group is valid; flagged "topup_train")
  hand-fallback matched subsample: 50 hand-fallback val groups in each of the same large buckets
"""
import hashlib
import json
import random
import sys

SPLIT = "/share/nikola.lukic/uncut_arm1/cache_run/split.json"
CORPUS = "/share/nikola.lukic/uncut_arm1/corpus"
BUCKETS = (("le10", 1, 10), ("11_50", 11, 50), ("51_100", 51, 100), ("101_256", 101, 256))


def bucket(n):
    return next(b for b, lo, hi in BUCKETS if lo <= n <= hi)


def header(group):
    kind, ds = group.split("/")
    d = "train" if "train" in kind else "heldout"
    with open(f"{CORPUS}/{d}/{ds}/placements/placements.jsonl") as fh:
        return json.loads(fh.readline())


def main(out):
    sp = json.load(open(SPLIT))
    rng = random.Random("uncut-replay-diag:v1:train")
    train = sorted(rng.sample(sorted(sp["train"]), 300))
    val = sorted(sp["val"])
    meta = {}
    for g in train + val:
        h = header(g)
        meta[g] = {"n_tasks": int(h["n_tasks"]), "bucket": bucket(int(h["n_tasks"])), "label": h["label_source"], "r": h.get("r"), "state_id": h.get("state_id")}
    rng2 = random.Random("uncut-replay-diag:v1:ils")
    ils, fallback, topup = {}, {}, {}
    all_train = sorted(sp["train"])
    cache = {}

    def train_meta(g):
        if g not in cache:
            h = header(g)
            cache[g] = {"n_tasks": int(h["n_tasks"]), "bucket": bucket(int(h["n_tasks"])), "label": h["label_source"], "r": h.get("r"), "state_id": h.get("state_id")}
        return cache[g]

    for b in ("11_50", "51_100", "101_256"):
        pool = [g for g in val if meta[g]["bucket"] == b and meta[g]["label"] == "ils"]
        ils[b] = sorted(rng2.sample(pool, min(50, len(pool))))
        if len(ils[b]) < 40:
            tp = sorted(g for g in all_train if g not in train)
            tp = [g for g in tp if train_meta(g)["bucket"] == b and train_meta(g)["label"] == "ils"]
            topup[b] = sorted(rng2.sample(tp, min(40 - len(ils[b]), len(tp))))
            for g in topup[b]:
                meta[g] = train_meta(g)
        pool = [g for g in val if meta[g]["bucket"] == b and meta[g]["label"] == "hand"]
        fallback[b] = sorted(rng2.sample(pool, min(50, len(pool))))
    doc = {"split_sha256": hashlib.sha256(open(SPLIT, "rb").read()).hexdigest(), "train": train, "val": val, "ils_subsample": ils, "ils_topup_train": topup, "fallback_subsample": fallback, "meta": meta,
           "counts": {"val": {b: {k: sum(1 for g in val if meta[g]["bucket"] == b and meta[g]["label"] == k) for k in ("exact", "ils", "hand")} for b, _, _ in BUCKETS},
                      "train": {b: {k: sum(1 for g in train if meta[g]["bucket"] == b and meta[g]["label"] == k) for k in ("exact", "ils", "hand")} for b, _, _ in BUCKETS}},
           "ils_n": {b: len(v) + len(topup.get(b, [])) for b, v in ils.items()}, "ils_val_n": {b: len(v) for b, v in ils.items()}, "fallback_n": {b: len(v) for b, v in fallback.items()}}
    json.dump(doc, open(out, "w"), indent=1)
    print(json.dumps({k: doc[k] for k in ("ils_n", "ils_val_n", "fallback_n")}))


if __name__ == "__main__":
    main(sys.argv[1])
