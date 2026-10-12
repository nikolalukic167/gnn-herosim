"""scale_160_v1 preparation: the held-out topologies become the validation split, a placeholder test keeps the loader satisfied, and the
160-client source tag parses to its topology."""
import json
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts_cosim"))
import make_r1a_split as ms  # noqa: E402


def _corpus(tmp_path, tag):
    """train topologies 9001-9006 (2 datasets each), held-out 9101-9102 (2 each); returns (cache_dir, corpus split json)."""
    ids, bases = [], []
    for corpus, topos in (("gnn_datasets_wf1_train_X", range(9001, 9007)), ("gnn_datasets_wf1_heldout_X", range(9101, 9103))):
        base = tmp_path / corpus
        bases.append(str(base))
        for t in topos:
            for j in range(2):
                d = base / f"ds_{t}{j}"
                d.mkdir(parents=True)
                (d / "generation_provenance.json").write_text(json.dumps({"argv": ["--source-tag", tag.format(t=t)]}))
                ids.append(f"{corpus}/ds_{t}{j}")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "metadata.json").write_text(json.dumps({"base_dirs": bases}))
    pickle.dump(ids, open(cache / "dataset_ids.pkl", "wb"))
    split = tmp_path / "corpus_split.json"
    split.write_text(json.dumps({"train": [str(t) for t in range(9001, 9007)], "heldout": ["9101", "9102"]}))
    return cache, split


@pytest.mark.parametrize("tag", ["wf1p_cc40s{t}_heavy_g0", "wf1p_c160s24p0.6s{t}_moderate_g0"])
def test_source_tag_topology_parses_both_corpus_shapes(tmp_path, tag):
    d = tmp_path / "ds"
    d.mkdir()
    (d / "generation_provenance.json").write_text(json.dumps({"argv": ["--source-tag", tag.format(t=9903)]}))
    assert ms.topology_of(d) == "9903"


def test_heldout_as_val_makes_the_heldout_topologies_the_validation_split(tmp_path):
    cache, corpus = _corpus(tmp_path, "wf1p_c160s24p0.6s{t}_moderate_g0")
    out = tmp_path / "split.json"
    run = subprocess.run([sys.executable, str(REPO / "scripts_cosim/make_r1a_split.py"), "--cache-dir", str(cache), "--corpus-split", str(corpus),
                          "--out", str(out), "--heldout-as-val"], capture_output=True, text=True, cwd=REPO)
    assert run.returncode == 0, run.stderr
    sp = json.loads(out.read_text())
    top = sp["topologies"]
    assert top["val"] == ["9101", "9102"] and len(sp["test"]) == 2 and len(top["test"]) == 1
    assert not set(top["train"]) & set(top["val"]) and not set(top["test"]) & set(top["train"])
    assert set(top["train"]) | set(top["test"]) == {str(t) for t in range(9001, 9007)} and len(top["train"]) == 5
    assert sp["heldout_as_val"] is True and sp["test_placeholder"] is True and sp["dry_run"] is False
    assert all("heldout" in p for p in sp["val"]) and not any("heldout" in p for p in sp["train"] + sp["test"])


def test_default_mode_is_unchanged_heldout_is_test(tmp_path):
    cache, corpus = _corpus(tmp_path, "wf1p_cc40s{t}_heavy_g0")
    out = tmp_path / "split.json"
    run = subprocess.run([sys.executable, str(REPO / "scripts_cosim/make_r1a_split.py"), "--cache-dir", str(cache), "--corpus-split", str(corpus),
                          "--out", str(out)], capture_output=True, text=True, cwd=REPO)
    assert run.returncode == 0, run.stderr
    sp = json.loads(out.read_text())
    assert sp["topologies"]["test"] == ["9101", "9102"] and "heldout_as_val" not in sp
