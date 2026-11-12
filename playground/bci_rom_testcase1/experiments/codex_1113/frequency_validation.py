"""Independent positive-real response checks, including dynamics below dt=20 s."""

import json
from pathlib import Path

import numpy as np
import scipy.linalg as la
from data import NOMINAL, load_model, matrix
from evaluate import full_solve
from model import direct_projection
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent


def run():
    model = load_model()
    shifts = np.geomspace(1.0e-5, 1.0e4, 12)
    parameters = np.array([NOMINAL, [1.0, 1.0, 1.0], [1.0e4, 1.0e4, 1.0e4]])
    variants = {}
    for method in ("baseline", "global"):
        folder = HERE / "results" / f"{method}_20260805"
        variants[method] = np.load(folder / "basis.npy")
    energy = HERE / "results" / "global_20260805" / "energy_K_min.npy"
    if energy.exists():
        variants["global_energy_K_min"] = np.load(energy)
    projections = {name: direct_projection(model, V) for name, V in variants.items()}
    records = []
    for h in parameters:
        Kh = matrix(model, h)
        for shift in shifts:
            A = Kh + shift * model.core.C
            X, res = full_solve(A, model.source_shape)
            Z = model.source_shape.T @ X
            errors = {}
            for name, V in variants.items():
                C, K, G, H = projections[name]
                Ar = K + shift * C
                for p, Hj in zip(model.physical_to_effective(h), H, strict=True):
                    Ar += p * Hj
                xr = la.solve(Ar, G, assume_a="pos")
                errors[name] = {
                    "relative_transfer": float(
                        np.linalg.norm(G.T @ xr - Z) / np.linalg.norm(Z)
                    ),
                    "relative_field": float(
                        np.linalg.norm(V @ xr - X) / np.linalg.norm(X)
                    ),
                }
            records.append(
                {
                    "h": h.tolist(),
                    "shift": float(shift),
                    "errors": errors,
                    "max_solve_residual": max(res),
                }
            )
    summary = {
        name: {
            metric: max(r["errors"][name][metric] for r in records)
            for metric in ("relative_transfer", "relative_field")
        }
        for name in variants
    }
    out = {
        "shifts": shifts.tolist(),
        "parameters": parameters.tolist(),
        "summary": summary,
        "records": records,
        "validation_full_solves": 144,
        "note": (
            "Exploratory independent frequency grid; does not certify H-infinity error"
        ),
    }
    (HERE / "results" / "frequency_validation.json").write_text(
        json.dumps(out, indent=2)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
