"""Run the unchanged adaptive extraction with instrumented spectrum/solves."""

import argparse
import hashlib
import json
import platform
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np
from data import load_model
from model import global_bounds
from threadpoolctl import threadpool_limits

from metahotspot.macromodel import utils


def run(method, seed, out):
    out.mkdir(parents=True, exist_ok=True)
    model = load_model()
    print(
        "cells",
        model.full_cell_count,
        "areas",
        [float(H.diagonal().sum()) for H in model.boundary_terms],
        flush=True,
    )
    print(
        "capacitance",
        model.core.C.diagonal().sum(),
        "source sums",
        model.source_shape.sum(axis=0),
        flush=True,
    )
    original_bounds = utils.port_eigenvalue_bounds
    original_solve = utils.spd_solve
    original_svd = utils._snapshot_svd_basis
    spectral = []
    solve_metrics = []
    if method == "global":
        bounds, record = global_bounds(model)
        spectral.append(record)
        print("global spectrum", record, flush=True)

        def bound_fn(*args, **kwargs):
            return bounds
    else:

        def bound_fn(*args, **kwargs):
            t = time.perf_counter()
            val = original_bounds(*args, **kwargs)
            spectral.append(
                {
                    "lambda_min": val[0],
                    "lambda_max": val[1],
                    "seconds": time.perf_counter() - t,
                }
            )
            print("port spectrum", spectral[-1], flush=True)
            return val

    def solve(A, b, **kwargs):
        t = time.perf_counter()
        x = original_solve(A, b, **kwargs)
        solve_metrics.append(
            {
                "seconds": time.perf_counter() - t,
                "residual": float(np.linalg.norm(A @ x - b) / np.linalg.norm(b)),
            }
        )
        if len(solve_metrics) % 10 == 0:
            print(
                "full solves", len(solve_metrics), "last", solve_metrics[-1], flush=True
            )
        return x

    def capture(X, tol):
        np.save(out / "snapshots.npy", X)
        return original_svd(X, tol)

    utils.port_eigenvalue_bounds = bound_fn
    utils.spd_solve = solve
    utils._snapshot_svd_basis = capture
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        V, summary = utils.build_parametric_basis(
            model.core,
            model.source_shape,
            model.boundary_terms,
            model.h_ranges(),
            tolerance=1.0e-3,
            max_order=1024,
            probe_rounds=10,
            seed=seed,
        )
    elapsed = time.perf_counter() - started
    np.save(out / "basis.npy", V)
    total = elapsed + (spectral[0]["seconds"] if method == "global" else 0.0)
    record = {
        "method": method,
        "seed": seed,
        "cells": model.full_cell_count,
        "seconds": total,
        "extraction_seconds": elapsed,
        "seconds_definition": (
            "spectral planning + adaptive extraction + closing SVD; "
            "excludes geometry assembly and validation"
        ),
        "summary": summary,
        "spectrum": spectral,
        "solves": solve_metrics,
        "warnings": [str(w.message) for w in caught],
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "source_hash": hashlib.sha256(Path(utils.__file__).read_bytes()).hexdigest(),
    }
    (out / "training.json").write_text(json.dumps(record, indent=2))
    print("DONE", method, seed, V.shape, total, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["baseline", "global"], required=True)
    p.add_argument("--seed", type=int, default=20260805)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    with threadpool_limits(limits=1):
        run(a.method, a.seed, a.out)
