import copy

import pytest

from src.placement import flow_snapshot as F


def _t(pid, status, rem=100.0, **kw):
    t = {"partner_id": pid, "src_node": "a", "payload": 100.0, "latency": 0.01, "hops": [["a>c", 10.0], ["c>b", 5.0]], "b": 0.0, "rest": 100.0, "scale": 1.0,
         "status": status, "startup_remaining": 0.0, "rem": rem, "rate": 0.0, "latency_remaining": 0.01}
    t.update(kw)
    return t


OK = {
    "version": 1, "model": "processor_sharing_v1", "partners": "serial", "tcp": True, "fairshare_last": 0.0, "active": [[5, 1]],
    "exchanges": [
        {"owner_tid": 5, "dst_node": "b", "stage": "input_io", "planned": False, "read_remaining": 0.0,
         "transfers": [_t(3, "done", 0.0, latency_remaining=0.0), _t(7, "active", 40.0, rate=5.0), _t(9, "pending")]},
        {"owner_tid": 6, "dst_node": "b", "stage": "input_io", "planned": False, "read_remaining": 1.5, "transfers": [_t(3, "pending")]},
    ],
}


def test_valid_block_passes():
    F.validate(OK)


@pytest.mark.parametrize("edit", [
    lambda b: b.update(version=2),
    lambda b: b.update(fairshare_last=0.5),
    lambda b: b["active"].clear(),
    lambda b: b["active"].append([5, 1]),
    lambda b: b["exchanges"][0]["transfers"][1].update(status="bogus"),
    lambda b: b["exchanges"][0]["transfers"][1].update(rem=101.0),
    lambda b: b["exchanges"][0]["transfers"][1].update(rate=0.0),
    lambda b: b["exchanges"][0]["transfers"].reverse(),
    lambda b: b["exchanges"][1]["transfers"][0].update(status="startup"),
    lambda b: b["exchanges"].append(copy.deepcopy(b["exchanges"][0])),
])
def test_broken_block_fails_loudly(edit):
    b = copy.deepcopy(OK)
    edit(b)
    with pytest.raises(ValueError):
        F.validate(b)
