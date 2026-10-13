"""The curve reader's peak direction and chance comparison, on saved runs (wandb curve audit, 2026-10-12).

Fixtures are real runs trimmed to what the reader consumes (history and summary records, plain config keys):
candidate-pilot-v2 fsq9m4c5 (20 epochs, val/mt_plan_exact and val/mt_task_acc_choice rise to their last epochs) and
grounded n0jcowgj (full-context CE weight 1.0, so train/ce sits at ~2x the one-term val floor while val/ce beats it).
"""
from pathlib import Path

from scripts_cosim.read_training_curves import CHANCE_FLOOR, higher_is_better, main

FIX = Path(__file__).parent / "fixtures" / "wandb_curves"


def _read(name, capsys):
    main([str(FIX / name)])
    return capsys.readouterr().out


def test_exact_choice_and_agreement_metrics_peak_at_their_max():
    for key in ("val/mt_plan_exact", "val/mt_task_acc_choice", "val/task_acc_choice", "val/top1_agreement", "val/acc",
                "val/acc_top3", "val/exact_rate", "val/pair_optimal_fraction"):
        assert higher_is_better(key), key
    for key in ("val/ce", "val/regret_masked_topo", "val/regret_topk_frac", "train/ce", "train/total_loss",
                "val/regret_exact", "val/choice_ce"):
        assert not higher_is_better(key), key


def test_rising_exact_plan_metrics_are_not_reported_as_epoch_0_peaks(capsys):
    out = _read("candidate_pilot_v2_fsq9m4c5", capsys)
    assert "best at epoch 13 of 20" in out
    assert "NOTE val/mt_plan_exact peaks at epoch 19, not 13" in out
    assert "NOTE val/mt_task_acc_choice peaks at epoch 12" not in out  # within 5 epochs of the selection: no note
    assert "peaks at epoch 0" not in out
    # val/task_acc_choice reaches its minimum at epoch 18; that is not its peak
    assert "NOTE val/task_acc_choice peaks at epoch 1, not 13" in out and "task_acc_choice peaks at epoch 18" not in out


def test_train_ce_is_not_read_against_the_one_term_val_floor(capsys):
    assert "train/ce" not in CHANCE_FLOOR
    out = _read("grounded_n0jcowgj", capsys)
    assert "NEVER BEATS CHANCE" not in out
    assert "val/ce                         untrained=5.0945 after-ep1=4.2313 best=4.2313 chance=4.9113 -> beats chance" in out
    assert "train/ce                       not compared to chance" in out and "NEAR_RTT_FULL_CONTEXT_CE_WEIGHT" in out
