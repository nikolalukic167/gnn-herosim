import pickle
import threading

from src.notebooks.non_unique_lib import cache_io


def test_save_is_atomic_for_concurrent_readers(tmp_path):
    big = {i: [(j, 0.5) for j in range(2000)] for i in range(400)}
    cache_io.save_capped_valid_combos_map(tmp_path, {0: [(0, 0.0)]})
    path = cache_io._capped_sidecar_path(tmp_path)
    errors, stop = [], threading.Event()

    def read():
        while not stop.is_set():
            try:
                with open(path, "rb") as f:
                    pickle.load(f)
            except Exception as err:
                errors.append(err)

    t = threading.Thread(target=read)
    t.start()
    for _ in range(5):
        cache_io.save_capped_valid_combos_map(tmp_path, big)
    stop.set()
    t.join()
    assert not errors
    assert cache_io.load_capped_valid_combos_map(tmp_path) == big
    assert sorted(p.name for p in tmp_path.iterdir()) == [path.name]
