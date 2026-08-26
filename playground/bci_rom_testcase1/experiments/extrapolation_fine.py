r"""Affine-parameter extrapolation re-run at the finer 1.0 mm mesh (134,640 cells).

Mirrors the 2.5 mm extrapolation experiment: two affine boundary parameters
(h_crown, h_fr4), each swept over {1e-2,1e-1,1,1e2,1e4,1e5,1e6} (49 combos,
spanning at least two decades beyond the training range on both sides), for
three training ranges.  Only the steady junction error is needed, so the
reference is the direct (unreduced) steady solve of the affine operator.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[3]  # repo root
CASE = PROJECT / "playground" / "bci_rom_testcase1"
sys.path[:0] = [str(CASE)]
from model_case1 import Case1Config, Case1Model  # noqa: E402
from metahotspot.macromodel.utils import (  # noqa: E402
    assemble_reduced_k,
    build_parametric_basis,
    project_bci,
    solve_rom_steady,
    spd_solve,
)

OUT = PROJECT / "results" / "experiments"
POWER = np.array([0.1, 0.2, 0.3, 0.4])
MM = 1.0
CASES = [
    ("1e-2..1e6", ((1e-2, 1e6), (1e-2, 1e6))),
    ("1..1e4", ((1, 1e4), (1, 1e4))),
    ("10..1e3", ((10, 1e3), (10, 1e3))),
]
TESTS = [
    (a, b)
    for a in [1e-2, 1e-1, 1, 1e2, 1e4, 1e5, 1e6]
    for b in [1e-2, 1e-1, 1, 1e2, 1e4, 1e5, 1e6]
]


def make_model(h_ranges):
    return Case1Model(
        Case1Config(
            max_xy_cell_mm=MM,
            max_z_cell_mm=MM,
            duration_s=100.0,
            dt_s=5.0,
            h_ranges=tuple(h_ranges),
        )
    )


def steady_reference(model, h):
    """Unreduced steady solve of the affine operator at physical ``h``."""
    core = model.core_operators()
    K = core.K.tocsc()
    for pk, Hk in zip(model.physical_to_effective(h), model.boundary_terms()):
        K = K + float(pk) * Hk.tocsc()
    K = (0.5 * (K + K.T)).tocsc()
    x = spd_solve(K, model.source_shape() @ POWER)  # rise above ambient
    return model.ambient_K + model.source_shape().T @ x  # absolute junction


def main():
    rows, extraction = [], []
    for name, ranges in CASES:
        model = make_model(ranges)
        core, G, terms = (
            model.core_operators(),
            model.source_shape(),
            model.boundary_terms(),
        )
        t0 = time.perf_counter()
        basis, summary = build_parametric_basis(
            core,
            G,
            terms,
            model.h_ranges(),
            tolerance=1e-3,
            max_order=1024,
            probe_rounds=2,
            seed=20260805,
        )
        ext = time.perf_counter() - t0
        extraction.append(
            dict(
                training=name,
                seconds=ext,
                order=int(basis.shape[1]),
                cells=int(model.full_cell_count),
                candidates=summary["processed_candidate_count"],
            )
        )
        C, K0, F, Fb, Ab = project_bci(core, G, terms, basis)
        ambient = model.ambient_K
        for h in TESTS:
            refj = steady_reference(model, h)
            K = assemble_reduced_k(K0, Fb, Ab, model.physical_to_effective(h))
            theta = solve_rom_steady(K, F, POWER)
            romj = ambient + F.T @ theta
            denom = max(float(np.max(np.abs(refj - ambient))), 1e-12)
            err = 100.0 * float(np.max(np.abs(romj - refj))) / denom
            rows.append(
                dict(
                    training=name,
                    h1=h[0],
                    h2=h[1],
                    error_pct=err,
                    extraction_s=ext,
                    order=int(basis.shape[1]),
                )
            )
        print(f"[{name}] extracted order={basis.shape[1]} in {ext:.1f}s")

    result = dict(
        mesh_mm=MM,
        cells=int(model.full_cell_count),
        extraction=extraction,
        points=rows,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"extrapolation_{MM:g}mm.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print("\n=== summary ===")
    print(
        f"{'training':<12}{'order':>6}{'extract_s':>10}{'max_err%':>10}{'mean_err%':>10}"
    )
    for name, _ in CASES:
        r = [x for x in rows if x["training"] == name]
        ex = next(e for e in extraction if e["training"] == name)
        print(
            f"{name:<12}{ex['order']:>6}{ex['seconds']:>10.1f}"
            f"{max(x['error_pct'] for x in r):>10.4f}"
            f"{sum(x['error_pct'] for x in r)/len(r):>10.5f}"
        )


if __name__ == "__main__":
    main()
