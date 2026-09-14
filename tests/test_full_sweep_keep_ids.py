"""The trainer's full-sweep load must scale with what it queries, not with the corpus.

`build_full_sweep_rtt_by_dataset` held every parent's sweep rows for the whole run. At the
peer_affinity T1b cache that is 19.66M rows, ~21 GB resident at ~1.09 KB/row, against a 64 GB
allocation; a 2,000-dataset corpus projects to ~83 GB and would not start. The per-epoch
validation only ever queries VAL parents, so the other ~80 % was paid for nothing.

These tests pin the filtering and, more importantly, that a requested parent which is absent
fails loud -- a silent miss would make the exact-regret metric fall back to a constant floor
for that dataset, which is precisely the checkpoint-selection bug route_b_v1 Phase 2 spent a
retrain discovering.
"""

import json
import pickle
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _cache(tmp_path: Path, rows_by_chunk):
    total = 0
    for i, rows in enumerate(rows_by_chunk):
        with open(tmp_path / f"rtt_chunk_{i}.pkl", "wb") as fh:
            pickle.dump(rows, fh)
        total += len(rows)
    (tmp_path / "rtt_chunks_meta.json").write_text(
        json.dumps({"num_chunks": len(rows_by_chunk), "total_entries": total})
    )
    return tmp_path


@pytest.fixture(scope="module")
def loader():
    """`train_near_rtt.py` is a script: importing it parses argv and starts training. Lift the
    one function under test out of the source instead, which also proves it still exists with
    the signature these tests assume."""
    import ast
    import textwrap
    from typing import Dict, Iterable, Optional, Tuple  # noqa: F401

    src = (Path(__file__).resolve().parents[1] / "src/notebooks/train_near_rtt.py").read_text()
    tree = ast.parse(src)
    fn = next(
        (n for n in tree.body
         if isinstance(n, ast.FunctionDef) and n.name == "build_full_sweep_rtt_by_dataset"),
        None,
    )
    assert fn is not None, "build_full_sweep_rtt_by_dataset is gone from train_near_rtt.py"
    ns: Dict[str, object] = {
        "Path": Path, "Optional": Optional, "Iterable": Iterable,
        "RttByCombo": Dict[str, Dict[Tuple[int, ...], float]],
    }
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<train_near_rtt>", "exec"), ns)
    return ns["build_full_sweep_rtt_by_dataset"]


def test_without_keep_ids_every_parent_is_returned(tmp_path, loader):
    _cache(tmp_path, [{("ds_a", (0, 1)): 10.0, ("ds_b", (0, 1)): 20.0}])
    out = loader(tmp_path)
    assert set(out) == {"ds_a", "ds_b"}


def test_keep_ids_drops_the_other_parents_rows(tmp_path, loader):
    _cache(tmp_path, [
        {("ds_a", (0, 1)): 10.0, ("ds_b", (0, 1)): 20.0},
        {("ds_a", (1, 1)): 11.0, ("ds_c", (0, 1)): 30.0},
    ])
    out = loader(tmp_path, keep_ids=["ds_a"])
    assert set(out) == {"ds_a"}
    assert out["ds_a"] == {(0, 1): 10.0, (1, 1): 11.0}


def test_a_requested_parent_with_no_rows_fails_loud(tmp_path, loader):
    """Silently returning nothing for a val parent would drop its exact regret back to a
    constant floor and corrupt checkpoint selection without any error."""
    _cache(tmp_path, [{("ds_a", (0, 1)): 10.0}])
    with pytest.raises(RuntimeError) as exc:
        loader(tmp_path, keep_ids=["ds_a", "ds_missing"])
    assert "no full-sweep rows" in str(exc.value)


def test_the_chunk_count_check_still_covers_rows_that_were_filtered_out(tmp_path, loader):
    """Filtering must not weaken the truncation check: a short chunk file is a corrupt cache
    whether or not the split being loaded wanted its parents."""
    _cache(tmp_path, [{("ds_a", (0, 1)): 10.0, ("ds_b", (0, 1)): 20.0}])
    (tmp_path / "rtt_chunks_meta.json").write_text(
        json.dumps({"num_chunks": 1, "total_entries": 99})
    )
    with pytest.raises(RuntimeError) as exc:
        loader(tmp_path, keep_ids=["ds_a"])
    assert "disagree" in str(exc.value)


def test_the_training_loop_loads_only_the_val_split():
    """A source-level check: the per-epoch metric queries val parents, so the startup load
    must be scoped to them. If this line loses its keep_ids, resident memory silently goes
    back to scaling with the whole corpus and only a 2,000-dataset OOM would reveal it."""
    src = (Path(__file__).resolve().parents[1] / "src/notebooks/train_near_rtt.py").read_text()
    assert "build_full_sweep_rtt_by_dataset(\n        CACHE_CTX.cache_dir, keep_ids=val_ids\n    )" in src
    assert "def _final_eval(loader, ids, tag):" in src, (
        "the final train/test evaluations must load their own split and free it"
    )
