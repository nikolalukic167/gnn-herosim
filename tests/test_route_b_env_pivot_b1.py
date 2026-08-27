"""route_b_env_pivot_v1 screen — B1 (registered build item, §4): hetdem/futureint
single-block ablation arms in route_b_coefficient_transfer.ablation(), plus
verify_route_b_scorer_agreement.check_blocks's independent recomputation of them.

S4's attribution table needs these two arms to answer "does hetdem/futureint alone
carry a rung's closure" the same way the existing parent-coupling arm answers it for
hop+coupling. Verified here on the repo's own toy DAG+network rig (imported from
test_route_b_repair_fixtures.py, no re-derivation) and, in the session report, on the
existing pilot corpus (score an EXISTING corpus, not generation).
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts_cosim"))

from route_b_coefficient_transfer import Cell, ablation  # noqa: E402
import verify_route_b_scorer_agreement as verifier  # noqa: E402

# Reuse the repo's own toy DAG+network rig rather than re-deriving one.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_route_b_repair_fixtures import (  # noqa: E402
    TOY_TASK_TYPES,
    coupled_rtt,
    write_toy,
)

TOL = 1e-9


def toy_cell(tmp_path: Path, alpha: float = 2.0) -> Cell:
    ds_dir = write_toy(tmp_path, coupled_rtt)
    task_types_path = tmp_path / "task_types.json"
    with open(task_types_path, "w") as fh:
        json.dump(TOY_TASK_TYPES, fh)
    return Cell(ds_dir, TOY_TASK_TYPES, alpha)


def test_ablation_includes_hetdem_and_futureint_single_block_arms(tmp_path):
    cell = toy_cell(tmp_path)
    out = ablation([cell])
    assert "hetdem" in out
    assert "futureint" in out
    assert out["hetdem"]["blocks"] == ["hetdem"]
    assert out["futureint"]["blocks"] == ["futureint"]
    # Every pre-existing arm is still present and unchanged in shape.
    for name in ("kint", "quad", "kint+quad", "occupancy (kint+quad+cap)",
                "parent-coupling (hop+coupling)", "parent-coupling incl kint",
                "full T1"):
        assert name in out
        for key in ("median_fraction", "mean_fraction", "n_closed_ge_half",
                    "residual_gt_material", "fractions"):
            assert key in out[name]


def test_ablation_hetdem_futureint_fractions_are_well_formed(tmp_path):
    cell = toy_cell(tmp_path)
    out = ablation([cell])
    for name in ("hetdem", "futureint"):
        row = out[name]
        assert len(row["fractions"]) == 1
        # fraction() = 1 - repaired/r_base; on a single-cell arm this is finite and
        # matches Cell.repair's own accounting exactly (not re-derived independently
        # here -- that is what test_check_blocks_agrees_on_hetdem_futureint does).
        repaired, _beta = cell.repair((name,))
        expected_fraction = cell.fraction(repaired)
        assert row["fractions"][0] == pytest.approx(expected_fraction, abs=TOL)
        assert row["median_fraction"] == pytest.approx(expected_fraction, abs=TOL)


def test_check_blocks_independently_recomputes_hetdem_futureint_arms(tmp_path):
    """The verifier-side agreement B1 requires: route_b_coefficient_transfer's report
    (with the new hetdem/futureint ablation arms) is independently reproduced by
    verify_route_b_scorer_agreement.check_blocks to 1e-9, using check_blocks's own
    solver and its own from-scratch column recomputation (repaired_r_exact/t1_columns),
    never importing the transfer script's fitted values."""
    ds_dir = write_toy(tmp_path, coupled_rtt)
    task_types_path = tmp_path / "task_types.json"
    with open(task_types_path, "w") as fh:
        json.dump(TOY_TASK_TYPES, fh)
    alpha = 2.0
    cell = Cell(ds_dir, TOY_TASK_TYPES, alpha)

    ablation_out = ablation([cell])
    pooled_blocks = ("quad", "cap", "hop", "coupling")
    cell_b_repaired, _beta_b = cell.repair(pooled_blocks)

    # Build a minimal §9b-shaped transfer report by hand (the real CLI writes a wider
    # one; check_blocks only reads per_dataset/pooled_blocks/ablation/alpha).
    report = {
        "alpha": alpha,
        "pooled_blocks": list(pooled_blocks),
        "per_dataset": [{
            "ds": ds_dir.name,
            "r_exact_pct": cell.r_base,
            "cell_a_fraction": cell.fraction(cell.repair(tuple(verifier.T1_BLOCKS))[0]),
            "cell_b_fraction": cell.fraction(cell_b_repaired),
        }],
        "ablation": ablation_out,
    }
    report_path = tmp_path / "transfer_report.json"
    with open(report_path, "w") as fh:
        json.dump(report, fh)

    # check_blocks prints "OK: ..." and returns 0 on success, or calls fail() (sys.exit)
    # on any 1e-9 disagreement -- so a clean run IS the assertion.
    rc = verifier.check_blocks(str(ds_dir.parent), str(report_path), str(task_types_path))
    assert rc == 0
