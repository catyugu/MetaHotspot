"""Held-out full-order solves, FloTHERM comparison, and compression ablations."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from data import NOMINAL, POWER, evaluation_points, flotherm, load_model, matrix
from model import direct_projection, svd_basis
from threadpoolctl import threadpool_limits

from metahotspot.macromodel import utils


def full_solve(A, rhs):
    pre = utils._rs_preconditioner(A.tocsr())
    if rhs.ndim == 1:
        rhs = rhs[:, None]
    X, residuals = [], []
    for b in rhs.T:
        x, info = sla.cg(A, b, M=pre, rtol=1.0e-11, atol=0.0, maxiter=3000)
        if info:
            raise RuntimeError(f"held-out CG failed: {info}")
        X.append(x)
        residuals.append(float(np.linalg.norm(A @ x - b) / np.linalg.norm(b)))
    return np.column_stack(X), residuals


def trajectory(C, K, G, *, full=False):
    """Common BDF1, dt=20 s, t=0..1000 s. Full solve warm starts each step."""
    C, K = sp.csr_matrix(C), sp.csr_matrix(K)
    A = K + C / 20.0
    if full:
        pre = utils._rs_preconditioner(A)

        def solve(rhs, x):
            y, info = sla.cg(A, rhs, x0=x, M=pre, rtol=1.0e-11, atol=0.0, maxiter=3000)
            if info:
                raise RuntimeError(f"transient CG failed {info}")
            return y
    else:
        factor = la.cho_factor(A.toarray(), check_finite=False)

        def solve(rhs, x):
            return la.cho_solve(factor, rhs, check_finite=False)

    x = np.zeros(K.shape[0])
    hist = [x.copy()]
    for _ in range(50):
        x = solve(C @ x / 20.0 + G @ POWER, x)
        hist.append(x.copy())
    return np.asarray(hist)


def references(model, out):
    path = out / "reference.npz"
    if path.exists():
        return dict(np.load(path))
    started = time.perf_counter()
    points = evaluation_points()
    fields, junction, responses, residuals = [], [], [], []
    for h in points:
        X, res = full_solve(matrix(model, h), model.source_shape)
        fields.append(X @ POWER)
        junction.append(model.source_shape.T @ X @ POWER)
        responses.append(model.source_shape.T @ X)
        residuals.extend(res)
        print("reference", h.tolist(), junction[-1].tolist(), flush=True)
    hist = trajectory(
        model.core.C, matrix(model, NOMINAL), model.source_shape, full=True
    )
    Kf, Cf, Gf, Hf = flotherm()
    fl_junction = []
    for h in points:
        Af = Kf.copy()
        for v, H in zip(h, Hf, strict=True):
            Af += v * H
        fl_junction.append(Gf.T @ sla.spsolve(Af, Gf @ POWER))
    Af = Kf.copy()
    for v, H in zip(NOMINAL, Hf, strict=True):
        Af += v * H
    fl_hist = trajectory(Cf, Af, Gf) @ Gf
    refs = {
        "points": points,
        "fields": np.asarray(fields),
        "junction": np.asarray(junction),
        "transfer": np.asarray(responses),
        "hist": hist,
        "transient_junction": hist @ model.source_shape,
        "fl_junction": np.asarray(fl_junction),
        "fl_hist": fl_hist,
        "max_residual": max(residuals),
        "seconds": time.perf_counter() - started,
    }
    np.savez(path, **refs)
    return refs


def evaluate_basis(model, V, refs, boundary_compressed):
    C, K, G, H = direct_projection(model, V)
    if boundary_compressed:
        Cb, Kb, Gb, F, A = utils.project_bci(
            model.core, model.source_shape, model.boundary_terms, V
        )
        C, K, G = Cb.toarray(), Kb.toarray(), Gb
        H = [F @ Ak @ F.T for Ak in A]
    ss, field_errors, transfer_errors, residuals = [], [], [], []
    for i, h in enumerate(refs["points"]):
        A = K.copy()
        for p, Hj in zip(model.physical_to_effective(h), H, strict=True):
            A += p * Hj
        X = la.solve(A, G, assume_a="pos")
        full_X = V @ X
        R = matrix(model, h) @ full_X - model.source_shape
        residuals.append(
            np.linalg.norm(R, axis=0) / np.linalg.norm(model.source_shape, axis=0)
        )
        rise = X @ POWER
        ss.append(G.T @ rise)
        field_errors.append(float(np.max(np.abs(V @ rise - refs["fields"][i]))))
        transfer_errors.append(float(np.max(np.abs(G.T @ X - refs["transfer"][i]))))
    A = K.copy()
    for p, Hj in zip(model.physical_to_effective(NOMINAL), H, strict=True):
        A += p * Hj
    hist = trajectory(C, A, G)
    j_hist = hist @ G
    errs = np.abs(np.asarray(ss) - refs["junction"])
    return {
        "order": V.shape[1],
        "boundary_compressed": boundary_compressed,
        "nominal_junction_K": (308.15 + np.asarray(ss[0])).tolist(),
        "nominal_junction_error_K": float(errs[0].max()),
        "worst_junction_error_K": float(errs.max()),
        "worst_relative_junction_error": float(
            np.max(errs / np.maximum(np.abs(refs["junction"]), 1.0e-12))
        ),
        "nominal_relative_junction_error": float(
            np.max(errs[0] / np.abs(refs["junction"][0]))
        ),
        "errors_by_point_K": errs.max(axis=1).tolist(),
        "worst_field_error_K": max(field_errors),
        "nominal_field_error_K": field_errors[0],
        "worst_transfer_error_K_per_W": max(transfer_errors),
        "nominal_final_response_residual": float(np.max(residuals[0])),
        "worst_final_response_residual": float(np.max(residuals)),
        "transient_junction_error_K": float(
            np.max(np.abs(j_hist - refs["transient_junction"]))
        ),
        "transient_field_error_K": float(np.max(np.abs(hist @ V.T - refs["hist"]))),
        "transient_junction_K": (308.15 + j_hist).tolist(),
    }


def run(dirs, out, ablate=False):
    out.mkdir(parents=True, exist_ok=True)
    model = load_model()
    refs = references(model, out)
    for folder in dirs:
        V = np.load(folder / "basis.npy")
        records = {
            "direct": evaluate_basis(model, V, refs, False),
            "bci": evaluate_basis(model, V, refs, True),
        }
        if ablate:
            X = np.load(folder / "snapshots.npy")
            for tol in (1.0e-4, 1.0e-5, 1.0e-6):
                Vt, _ = svd_basis(X, tol)
                records[f"svd_{tol:g}"] = evaluate_basis(model, Vt, refs, False)
        record = {
            "method": folder.name,
            "records": records,
            "nominal_full_K": (308.15 + refs["junction"][0]).tolist(),
            "nominal_flotherm_K": (308.15 + refs["fl_junction"][0]).tolist(),
            "points": refs["points"].tolist(),
            "flotherm_errors_by_point_K": np.max(
                np.abs(refs["fl_junction"] - refs["junction"]), axis=1
            ).tolist(),
            "flotherm_transient_error_K": float(
                np.max(np.abs(refs["fl_hist"] - refs["transient_junction"]))
            ),
            "reference_residual": float(refs["max_residual"]),
        }
        (folder / "evaluation.json").write_text(json.dumps(record, indent=2))
        print(
            folder.name,
            json.dumps(
                {
                    k: {
                        kk: vv
                        for kk, vv in v.items()
                        if kk not in ("transient_junction_K", "errors_by_point_K")
                    }
                    for k, v in records.items()
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ablate", action="store_true")
    a = p.parse_args()
    with threadpool_limits(limits=1):
        run(a.runs, a.out, a.ablate)
