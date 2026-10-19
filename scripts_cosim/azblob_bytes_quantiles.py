#!/usr/bin/env python3
"""Derive the empirical BlobBytes quantile table of the Azure Functions Blob Dataset 2020 (azurefunctions-accesses-2020.csv.bz2, CC-BY) for the
workload_payloads 'azblob' sampler: BlobBytes is the per-access blob size, used as a proxy for a payload, NOT function-to-function exchange.
usage: azblob_bytes_quantiles.py <trace .csv.bz2> <expected sha256 of the trace> <out .json>
Verifies the trace's sha256, streams the CSV (column BlobBytes only), writes quantiles at probabilities i / 1000 (linear interpolation) plus summary numbers."""
import hashlib, json, sys
import numpy as np
import pandas as pd

path, want, out = sys.argv[1:4]
h = hashlib.sha256()
with open(path, "rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 22), b""):
        h.update(chunk)
if h.hexdigest() != want:
    raise SystemExit(f"FAIL LOUD: sha256 {h.hexdigest()} != {want}")
parts, rows = [], 0
for ch in pd.read_csv(path, compression="bz2", usecols=["BlobBytes"], dtype={"BlobBytes": "float64"}, chunksize=5_000_000):
    v = ch["BlobBytes"].to_numpy()
    if np.isnan(v).any() or (v < 0).any():
        raise SystemExit("FAIL LOUD: BlobBytes has NaN or negative values")
    parts.append(v); rows += len(v)
v = np.concatenate(parts); del parts
N = 1000
q = np.quantile(v, np.arange(N + 1) / N, method="linear")
named = {f"p{p}": float(np.quantile(v, p / 100)) for p in (1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9)}
doc = {"quantiles": [float(x) for x in q], "n_quantile_steps": N, "source": "azurefunctions-accesses-2020.csv.bz2 (Azure Functions Blob Dataset 2020, CC-BY)", "source_sha256": want,
       "column": "BlobBytes", "rows": rows, "zero_bytes": int((v == 0).sum()), "min": float(v.min()), "max": float(v.max()), "mean": float(v.mean()), "named": named,
       "note": "per-access blob size as a proxy, not function-to-function exchange"}
with open(out, "w") as fh:
    json.dump(doc, fh, indent=1, sort_keys=True)
print(json.dumps({k: doc[k] for k in ("rows", "zero_bytes", "min", "max", "mean", "named")}, indent=1))
