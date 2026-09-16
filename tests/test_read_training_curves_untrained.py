"""The curve reader anchors every curve on the UNTRAINED value when the trainer logged one.

Every per-epoch history row is logged after that epoch's gradient steps, so the first row
is already one pass over the training set; the untrained read lives in the run summary
under untrained/val_* and the reader must surface it beside the chance floor.
"""
from scripts_cosim.read_training_curves import (
    REFERENCE_PREFIXES, reference_floors, untrained_key,
)


def test_untrained_is_a_reference_prefix_like_baseline():
    assert "untrained/" in REFERENCE_PREFIXES
    assert "baseline/" in REFERENCE_PREFIXES and "scale/" in REFERENCE_PREFIXES


def test_reference_floors_keeps_untrained_and_drops_curves():
    summary = {
        "baseline/val_chance_task_acc": 0.385,
        "untrained/val_task_acc": 0.39,
        "scale/val_opt_rtt": 480.0,
        "val/task_acc": 0.85,          # a curve's last value, not a reference line
        "peak_val_task_acc": 0.86,
        "_step": 299,
    }
    floors = reference_floors(summary)
    assert set(floors) == {
        "baseline/val_chance_task_acc", "untrained/val_task_acc", "scale/val_opt_rtt"}


def test_untrained_key_maps_a_curve_to_its_summary_entry():
    assert untrained_key("val/task_acc") == "untrained/val_task_acc"
    assert untrained_key("val/ce") == "untrained/val_ce"
    assert untrained_key("val/acc") == "untrained/val_acc"


def test_untrained_key_for_a_train_curve_does_not_alias_a_val_entry():
    # The trainer evaluates the val split only, so a train/ curve must never pick up
    # the val split's untrained value by accident.
    assert untrained_key("train/ce") == "untrained/train_ce"
    assert untrained_key("train/ce") != untrained_key("val/ce")
