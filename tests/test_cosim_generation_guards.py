"""Guards added to the co-sim generator on 2026-09-03.

Both close protocol-only gaps recorded in docs/gates/gate-tools.md:

* 2026-08-27 — a corpus generated without HEROSIM_COSIM_KEEP_ALIVE reports 204/204
  SUCCESS while carrying truncated sweeps (`sweep_complete: false`), and nothing on disk
  records the physics environment a dataset was generated under.
* 2026-08-28 — a task type starved to zero replicas dies later as an unlabelled
  `System state capture FAILED` (covered in tests/test_route_b_env_pivot_w3.py).
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts_cosim.generate_gnn_datasets_fast import (  # noqa: E402
    GENERATION_PROVENANCE_FILE,
    classify_generation_outcome,
    write_generation_provenance,
)


def test_complete_sweep_stays_success():
    meta = {"num_placements": 64, "rows_written": 64, "sweep_complete": True}
    assert classify_generation_outcome("success", meta) == "success"


def test_incomplete_sweep_is_truncated_not_success():
    meta = {"num_placements": 64, "rows_written": 51, "sweep_complete": False,
            "worker_exception": 13}
    assert classify_generation_outcome("success", meta) == "truncated"


def test_pre_metadata_engine_output_is_left_alone():
    assert classify_generation_outcome("success", None) == "success"


@pytest.mark.parametrize("status", ["skipped", "failed"])
def test_non_success_statuses_pass_through_untouched(status):
    assert classify_generation_outcome(status, {"sweep_complete": False}) == status


def test_provenance_records_physics_env_and_inputs(tmp_path):
    environ = {
        "HEROSIM_DATA_LOCALITY": "1",
        "HEROSIM_COSIM_KEEP_ALIVE": "1000000",
        "HEROSIM_STORAGE_NEUTRAL": "1",
        "PATH": "/usr/bin",             # not physics: must not be recorded
        "WANDB_API_KEY": "secret",      # not physics: must not be recorded
    }
    path = write_generation_provenance(
        tmp_path,
        dataset_id="ds_00007",
        seed=3401,
        grid_name="route_b_pivot_h2",
        num_tasks=4,
        allow_non_unique_replicas=False,
        warmth_physics="node_disk_v2",
        fast_forward_warmup=True,
        fast_forward_threshold=1,
        argv=["generate_gnn_datasets_fast.py", "--grid", "route_b_pivot_h2"],
        environ=environ,
    )
    assert path == tmp_path / GENERATION_PROVENANCE_FILE
    payload = json.loads(path.read_text())
    assert payload["physics_env"] == {
        "HEROSIM_COSIM_KEEP_ALIVE": "1000000",
        "HEROSIM_DATA_LOCALITY": "1",
        "HEROSIM_STORAGE_NEUTRAL": "1",
    }
    assert "PATH" not in json.dumps(payload["physics_env"])
    assert "secret" not in path.read_text()
    assert payload["seed"] == 3401
    assert payload["grid"] == "route_b_pivot_h2"
    assert payload["num_tasks"] == 4
    assert payload["allow_non_unique_replicas"] is False
    assert payload["warmth_physics"] == "node_disk_v2"
    assert payload["argv"][-1] == "route_b_pivot_h2"
    # Code provenance rides along so two corpora can be triaged as same/different tree.
    assert "commit" in payload["code"]


def test_provenance_env_absent_is_recorded_as_absent(tmp_path):
    """A paired control is defined by which HEROSIM_* vars are UNSET; an empty block
    must be written, not omitted, so absence is a recorded fact."""
    path = write_generation_provenance(
        tmp_path, dataset_id="ds_00000", seed=1, grid_name=None, num_tasks=4,
        allow_non_unique_replicas=True, warmth_physics="node_disk_v2",
        fast_forward_warmup=True, fast_forward_threshold=1, argv=[], environ={},
    )
    payload = json.loads(path.read_text())
    assert payload["physics_env"] == {}
    assert payload["grid"] is None
