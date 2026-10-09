import pytest

from scripts_cosim.ranking_mapping import plan_task_ids


def _ev(t):
    return {"application": {"dag": {t: []}}}


def _rec(gid, fn):
    return {"gid": gid, "fn": fn}


def test_ds_01600_dataset_order_differs_from_the_snapshot_batch_order():
    # corpus_b2_ext ds_01600: the dataset's tasks are dnn1, rf, cnn; the snapshot batch is cnn 2077, rf 2078, dnn1 2079
    assert plan_task_ids([_ev("dnn1"), _ev("rf"), _ev("cnn")], [_rec(2077, "cnn"), _rec(2078, "rf"), _rec(2079, "dnn1")]) == [2079, 2078, 2077]


def test_same_type_tasks_keep_snapshot_order():
    # ds_03200: three dnn1 tasks
    assert plan_task_ids([_ev("dnn1")] * 3, [_rec(1240, "dnn1"), _rec(1241, "dnn1"), _rec(1242, "dnn1")]) == [1240, 1241, 1242]


def test_mixed_repeats():
    assert plan_task_ids([_ev("rf"), _ev("cnn"), _ev("rf")], [_rec(5, "cnn"), _rec(6, "rf"), _rec(7, "rf")]) == [6, 5, 7]


def test_a_type_mismatch_raises():
    with pytest.raises(ValueError):
        plan_task_ids([_ev("rf")], [_rec(1, "cnn")])
