"""Import a scripts_cosim module at a frozen git blob, so a read stays frozen when the working file later gains additions.

load_frozen(here, "uncut_arm_gate_read.py", "04e368af...") returns the module built from that exact blob: the working file when it still is that
blob, else the blob from this checkout's git object store (`git cat-file`). The bytes are re-hashed either way; a missing blob or a hash miss fails
loud. The module is registered as "<name>@<blob[:8]>", so it never shadows the working module of the same name in one process.
"""
import hashlib
import importlib.util
import os
import subprocess
import sys


def git_blob(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def load_frozen(here, fname, want):
    path = os.path.join(here, fname)
    with open(path, "rb") as fh:
        src = fh.read()
    if git_blob(src) != want:
        r = subprocess.run(["git", "-C", here, "cat-file", "blob", want], capture_output=True)
        if r.returncode:
            raise SystemExit(f"FAIL LOUD: {fname} is not the frozen blob {want} and this checkout's git has no such blob: {r.stderr.decode().strip()}")
        src = r.stdout
    if git_blob(src) != want:
        raise SystemExit(f"FAIL LOUD: frozen {fname} blob {git_blob(src)}, want {want}")
    name = f"{fname[:-3]}@{want[:8]}"
    if name in sys.modules:
        return sys.modules[name]
    mod = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, loader=None, origin=path))
    mod.__file__ = path
    mod.__frozen_blob__ = want
    sys.modules[name] = mod
    try:
        exec(compile(src, f"{path}@{want[:8]}", "exec"), mod.__dict__)
    except BaseException:
        del sys.modules[name]
        raise
    return mod
