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


def test_workflow_offline_metric_names_and_logged_floor(monkeypatch, capsys):
    from scripts_cosim import read_training_curves as reader
    rows=[{'epoch':0,'validation_accuracy':.5,'validation_mean_rtt_ms':100.},
          {'epoch':1,'train_ce':.6,'validation_accuracy':.7,'validation_mean_rtt_ms':70.},
          {'epoch':2,'train_ce':.4,'validation_accuracy':.8,'validation_mean_rtt_ms':80.}]
    monkeypatch.setattr(reader,'_read_wandb_dir',lambda path:(rows,{}, {'contract':'workflow_manifest_assignment_v1','chance_accuracy':.5}))
    assert reader.main(['unused'])==0
    out=capsys.readouterr().out
    assert 'chance=0.5' in out and 'val/workflow_rtt_ms; best at epoch 1' in out
    assert reader._epoch_label(rows,'train_ce',1)=='2'


def test_residency_soft_targets_and_validation_selector(monkeypatch, capsys):
    from scripts_cosim import read_training_curves as reader
    rows = [{'epoch': 0, 'validation_regret': 2., 'train_soft_ce': None},
            {'epoch': 1, 'validation_regret': .2, 'train_soft_ce': 1.},
            {'epoch': 2, 'validation_regret': .3, 'train_soft_ce': .9}]
    monkeypatch.setattr(reader, '_read_wandb_dir', lambda path: (rows, {},
                        {'contract': 'resident_container_fifo_v1', 'chance_ce': 1.0986}))
    assert reader.main(['unused']) == 0
    out = capsys.readouterr().out
    assert 'chance=1.0986' in out
    assert 'val/residency_regret; best at epoch 1' in out
