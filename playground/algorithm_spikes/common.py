"""Minimal shared numerical and result helpers for isolated research studies."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy import linalg as la


def sym(a):
    return (a + a.T) * 0.5


def basis(a, rank, against=None, relative_cutoff=1e-11):
    a = np.asarray(a, dtype=float).copy()
    if against is not None and against.size:
        for _ in range(2):
            a -= against @ (against.T @ a)
    if not a.size or la.norm(a) < 1e-14:
        return np.empty((a.shape[0], 0))
    u, s, _ = la.svd(a, full_matrices=False, check_finite=False)
    keep = min(rank, int(np.count_nonzero(s > relative_cutoff*s[0])))
    return np.ascontiguousarray(u[:, :keep])


def extend(base, candidates, rank, fallback):
    extra = basis(candidates, rank-base.shape[1], against=base)
    v = np.column_stack((base, extra))
    if v.shape[1] < rank:
        v = np.column_stack((v, basis(fallback, rank-v.shape[1], against=v)))
    return v


def fingerprint(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def write_results(directory, rows, metadata):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    with (directory/'results.json').open('w') as f:
        json.dump({'metadata': metadata, 'rows': rows}, f, indent=2, allow_nan=False)
    if rows:
        keys = list(dict.fromkeys(k for row in rows for k in row))
        with (directory/'results.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader(); writer.writerows(rows)
