#!/usr/bin/env python3
"""Verify the component model and the condensation identities before any ROM claim.

Every full-order solve is AMG-preconditioned CG, so these identities hold only
up to the iterative tolerance (``STATIC_RTOL = 1e-12``).  The thresholds below
encode that; machine-precision identities would require a direct factorisation,
which this sandbox deliberately does not use.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=../../python \
        python test_reduction.py
"""

from __future__ import annotations

import numpy as np

import package_model as pm
import reduction as rd
from metahotspot.macromodel.utils import normalized_operators

REFERENCE = 340.0  # K, linearization point for the linear analysis
IDENTITY = 1.0e-8  # relative tolerance implied by STATIC_RTOL


def relative(matrix, reference) -> float:
    return float(np.linalg.norm(matrix - reference) / np.linalg.norm(reference))


def check(label, gap, tolerance=IDENTITY):
    status = "ok " if gap <= tolerance else "FAIL"
    print(f"  [{status}] {label:<48}{gap:.3e}")
    if gap > tolerance:
        raise AssertionError(f"{label}: {gap:.3e} > {tolerance:.1e}")


def main() -> None:
    package = pm.build_package()
    law = pm.default_interface(package)
    operators = normalized_operators(package.K, package.C, np.zeros(package.size))
    part = rd.partition(operators, package.interface)
    m = part.port_count
    psi = rd.static_modes(part)

    conductance = law.linear_conductance(REFERENCE, package.size)
    print(
        f"MODEL cells={package.size} interface={m} interior={part.interior_count} "
        f"port_split={[port.cells.size for port in package.ports]}"
    )
    print(
        f"  total capacity           = {float(package.C.sum()):.4f} J/K\n"
        f"  linearized top port      = {conductance[package.ports[0].cells].sum():.4e} W/K\n"
        f"  linearized bottom port   = {conductance[package.ports[1].cells].sum():.4e} W/K"
    )
    if np.any(conductance[package.interface] <= 0.0):
        raise AssertionError("interface conductance must be positive on every port cell")
    if np.any(conductance[np.setdiff1d(np.arange(package.size), package.interface)] != 0.0):
        raise AssertionError("the interface law must be supported on the ports only")

    print("\n1-2  STATIC CONSTRAINT MODES")
    residual = part.Kii @ psi + part.KiG.toarray()
    check("K_ii Psi + K_iG = 0", float(np.linalg.norm(residual) / np.linalg.norm(part.KiG.toarray())))
    check("Psi is the s = 0 interface response", relative(rd.response(part, 0.0), psi))

    print("\n3  OPEN vs CLOSED INTERFACE OPERATOR")
    schur = part.KGG + part.KiG.toarray().T @ psi
    scale = schur + np.diag(conductance[part.interface])
    null_gap = float(np.linalg.eigvalsh(schur)[0] / np.linalg.norm(schur))
    scale_min = float(np.linalg.eigvalsh(scale)[0])
    print(f"  [info] open-loop S(0) min eig / norm       {null_gap:.3e} (singular, as expected)")
    print(f"  [info] closed-loop min eig                 {scale_min:.3e} (SPD scale)")
    if scale_min <= 0.0:
        raise AssertionError("the closed-loop interface operator must be SPD")

    print("\n4-5  PROJECTED OPERATORS")
    plan = rd.ResponsePlan.build(part, np.geomspace(1.0e-2, 1.0e4, 6))
    for name, enrichment in (
        ("guyan", np.empty((part.interior_count, 0))),
        ("craig_bampton", rd.craig_bampton(part, 6)),
        ("rational_krylov", rd.rational_krylov(part, 12, plan)),
    ):
        model = rd.assemble(operators, part, enrichment, psi)
        off = float(np.linalg.norm(model.K[:m, m:]) / np.linalg.norm(model.K))
        print(
            f"  {name:<18} dim={model.dimension:<5} q={model.interior_dimension:<4} "
            f"off-block={off:.2e} S(0) gap={relative(model.K[:m, :m], schur):.2e}"
        )
        check(f"{name}: K_r block diagonal in the interface", off)
        check(f"{name}: K_r,GG = S(0)", relative(model.K[:m, :m], schur))

    guyan = rd.assemble(operators, part, np.empty((part.interior_count, 0)), psi)
    derivative = (
        part.CGG
        + part.CiG.toarray().T @ psi
        + psi.T @ part.CiG.toarray()
        + psi.T @ (part.Cii @ psi)
    )
    check("Guyan C_GG = dS/ds(0)", relative(guyan.C[:m, :m], derivative))

    print("\n6  NONLINEAR STEADY INTERFACE IS EXACT FOR ANY FLUX, ANY W")
    rng = np.random.default_rng(20261021)
    flux = rng.normal(size=m)
    closed = np.linalg.solve(scale, -flux)
    for name, enrichment in (
        ("guyan", np.empty((part.interior_count, 0))),
        ("craig_bampton", rd.craig_bampton(part, 4)),
        ("rational_krylov", rd.rational_krylov(part, 12, plan)),
    ):
        model = rd.assemble(operators, part, enrichment, psi)
        reduced = (model.K[:m, :m] + np.diag(conductance[part.interface])) @ closed
        check(
            f"{name}: interface residual, arbitrary flux",
            float(np.linalg.norm(reduced + flux) / np.linalg.norm(flux)),
        )

    print("\n7  A PORT FLUX PRODUCES EXACTLY THE STATIC CONSTRAINT MODE")
    load = -np.linalg.solve(part.Kii.toarray(), part.KiG.toarray())
    check("K_ii^-1 f_G = Psi (no interface-load direction needed)", relative(load, psi), tolerance=1.0e-6)

    print("\n8  INTERNAL SOURCE RESPONSE: where CB actually loses")
    source_flux = np.asarray(package.source) @ np.array([1.0, 0.4])
    exact_interior = np.linalg.solve(part.Kii.toarray(), source_flux[part.interior])
    source_basis = rd.source_directions(part, package.source)
    for name, enrichment in (
        ("guyan", np.empty((part.interior_count, 0))),
        ("craig_bampton_12", rd.craig_bampton(part, 12)),
        ("craig_bampton_48", rd.craig_bampton(part, 48)),
        ("with_source_directions", source_basis),
    ):
        model = rd.assemble(operators, part, enrichment, psi)
        if enrichment.shape[1]:
            z = np.linalg.solve(model.K[m:, m:], enrichment.T @ source_flux[part.interior])
        else:
            z = np.zeros(0)
        interior = enrichment @ z
        print(f"  {name:<24} interior source response error = {relative(interior, exact_interior):.2e}")

    print("\nAll structural identities hold (to the CG tolerance).")


if __name__ == "__main__":
    main()
