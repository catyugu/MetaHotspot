"""Fixed-state-budget energy POD ablation on the already computed responses.

For a collocated SPD response, g^T(x-x_r) = ||x-x_r||_A^2. This motivates
compressing snapshots in a thermal energy metric instead of Euclidean norm.
This script changes compression only, with an explicit constant mode.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.linalg as la
from data import NOMINAL, load_model, matrix
from evaluate import evaluate_basis, references
from threadpoolctl import threadpool_limits


def energy_basis(X, A, order):
    n = X.shape[0]
    one = np.ones(n)
    a_one = A @ one
    one /= np.sqrt(one @ a_one)
    a_one = A @ one
    AX = A @ X
    norms = np.sqrt(np.einsum("ij,ij->j", X, AX))
    Y = X / norms
    Y -= one[:, None] * (a_one @ Y)[None, :]
    AY = A @ Y
    gram = Y.T @ AY
    vals, W = la.eigh((gram + gram.T) * 0.5)
    selected = np.argsort(vals)[::-1][: order - 1]
    assert vals[selected[-1]] > 0
    Z = Y @ (W[:, selected] / np.sqrt(vals[selected]))
    V, _ = la.qr(np.column_stack([np.ones(n) / np.sqrt(n), Z]), mode="economic")
    return V


def run(folder, out, selected=None):
    model = load_model()
    refs = references(model, out)
    X = np.load(folder / "snapshots.npy")
    order = np.load(folder / "basis.npy", mmap_mode="r").shape[1]
    metrics = {
        "K_min": matrix(model, [1.0, 1.0, 1.0]),
        "K_nominal": matrix(model, NOMINAL),
        "C": model.core.C,
    }
    path = folder / "energy_pod.json"
    records = json.loads(path.read_text()) if path.exists() else {}
    for name, A in metrics.items():
        if selected and name not in selected:
            continue
        started = time.perf_counter()
        V = energy_basis(X, A, order)
        compression_seconds = time.perf_counter() - started
        record = evaluate_basis(model, V, refs, False)
        record["compression_seconds"] = compression_seconds
        records[name] = record
        np.save(folder / f"energy_{name}.npy", V)
        print(
            name,
            {
                k: v
                for k, v in record.items()
                if k not in ("transient_junction_K", "errors_by_point_K")
            },
            flush=True,
        )
    path.write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--metric", nargs="+", choices=["K_min", "K_nominal", "C"])
    a = p.parse_args()
    with threadpool_limits(limits=1):
        run(a.run, a.out, a.metric)
