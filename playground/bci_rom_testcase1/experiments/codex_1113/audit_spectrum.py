"""Retain original LOBPCG eigenvectors to distinguish convergence from planning."""

import json
from pathlib import Path

import numpy as np
from data import load_model
from threadpoolctl import threadpool_limits

from metahotspot.macromodel import utils


def run(out):
    model = load_model()
    saved = utils.spla.lobpcg
    records = []

    def capture(A, *args, **kwargs):
        values, vectors = saved(A, *args, **kwargs)
        K, C = model.core.K, model.core.C
        shift = max(1.0e-6, np.median(C.diagonal()) * 1.0e-6)
        for val, u in zip(values - shift, vectors.T, strict=True):
            Ku, Cu = K @ u, C @ u
            records.append(
                {
                    "value": float(val),
                    "absolute_residual": float(np.linalg.norm(Ku - val * Cu)),
                    "relative_residual": float(
                        np.linalg.norm(Ku - val * Cu) / max(np.linalg.norm(Ku), 1.0e-30)
                    ),
                    "c_norm": float(np.sqrt(u @ (C @ u))),
                }
            )
        return values, vectors

    utils.spla.lobpcg = capture
    bounds = utils.port_eigenvalue_bounds(
        model.core.K, model.core.C, model.source_shape[:, 0]
    )
    record = {
        "bounds": list(bounds),
        "eigenpairs": records,
        "null_residual": float(
            np.linalg.norm(model.core.K @ np.ones(model.full_cell_count))
        ),
    }
    out.write_text(json.dumps(record, indent=2))
    print(record)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run(Path(__file__).parent / "results" / "spectrum_audit.json")
