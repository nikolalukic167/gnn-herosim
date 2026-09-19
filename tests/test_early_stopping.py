"""--patience early stopping: off by default, pure stop rule, parsed into TrainingConfig.

Measured before this existed (2026-09-16): on the cold-corpus arms the selected epoch is
19-67 of 300 and on the warm corpus 34-283, so a fixed epoch cap is wrong on one corpus by
construction. Patience adapts; a floor protects the late selectors.
"""
import sys

from src.notebooks.non_unique_lib.training_config import (
    parse_training_config, should_stop_early,
)


# --- the stop rule ---------------------------------------------------------------------

def test_patience_zero_never_stops():
    for epoch in range(0, 1000, 37):
        assert should_stop_early(epoch, -1, patience=0, min_epochs=0) is False


def test_stops_exactly_patience_epochs_after_the_last_improvement():
    # improved at epoch 34, patience 60 -> epoch 94 is the first stop
    assert should_stop_early(93, 34, patience=60, min_epochs=0) is False
    assert should_stop_early(94, 34, patience=60, min_epochs=0) is True


def test_never_stops_before_the_floor():
    # 60 epochs without improvement by epoch 60, but the floor is 100 -> keep going
    assert should_stop_early(60, 0, patience=60, min_epochs=100) is False
    assert should_stop_early(98, 0, patience=60, min_epochs=100) is False
    assert should_stop_early(99, 0, patience=60, min_epochs=100) is True   # 100 epochs run


def test_a_run_that_never_improves_stops_after_patience_from_the_start():
    assert should_stop_early(58, -1, patience=60, min_epochs=0) is False
    assert should_stop_early(59, -1, patience=60, min_epochs=0) is True


def test_cold_corpus_shape_stops_well_before_300():
    """Selected epoch 67 (the latest measured on the cold corpus) + 60 patience -> 127."""
    stops = [e for e in range(300) if should_stop_early(e, 67, 60, 100)]
    assert stops[0] == 127


def test_warm_corpus_late_selector_runs_to_the_end():
    """Selected epoch 283 (measured on the warm corpus) + 60 patience > 300 -> no stop."""
    assert not any(should_stop_early(e, 283, 60, 100) for e in range(300))


# --- the parser ------------------------------------------------------------------------

def _parse(argv):
    saved = sys.argv
    sys.argv = ["train"] + argv
    try:
        return parse_training_config()
    finally:
        sys.argv = saved


def test_defaults_are_off():
    cfg = _parse([])
    assert cfg.patience == 0 and cfg.min_epochs == 0


def test_flags_are_parsed_and_clamped():
    cfg = _parse(["--patience", "60", "--min-epochs", "100", "--epochs", "300"])
    assert (cfg.patience, cfg.min_epochs, cfg.epochs) == (60, 100, 300)
    cfg = _parse(["--patience", "-5", "--min-epochs", "-1"])
    assert cfg.patience == 0 and cfg.min_epochs == 0
