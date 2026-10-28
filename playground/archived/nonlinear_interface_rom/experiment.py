#!/usr/bin/env python3
"""Embeddable transient ROM for a component with a nonlinear interface.

Three questions, each measured against a full-order reference:

Q1 (linear)     At a fixed interface dimension, how do Guyan, Craig-Bampton and
                rational-Krylov enrichment compare on the closed-loop interface
                operator and on the source-to-probe transfer?
Q2 (nonlinear)  Does that ranking survive a *nonlinear* interface driven into
                regimes that were never used for training?
Q3 (cost)       What do the reduced models actually buy, in AMG setups and CG
                iterations rather than wall-clock?

All full-order solves are AMG-preconditioned CG on positive real shifts, so
every operator is SPD and no direct factorization is used.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=../../python \
        python experiment.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import scipy.linalg
import scipy.sparse as sp

import package_model as pm
import reduction as rd
import transient as tr
from linalg import Cost, SPDSolver, dense_spd_solve
from metahotspot.macromodel.utils import normalized_operators

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results"
REFERENCE_K = 340.0
SHIFTS = np.geomspace(0.05, 3.0e4, 9)


def scenario(package, *, board, ambient, emissivity, coefficient, reference_h=1500.0):
    top, bottom = package.ports
    return pm.Interface(
        laws=(
            pm.RadiationConvection(top, emissivity=emissivity, coefficient=coefficient, ambient=ambient),
            pm.ContactConductance(bottom, reference_h=reference_h, board=board),
        )
    )


SCENARIOS = {
    "nominal": dict(board=320.0, ambient=300.0, emissivity=0.85, coefficient=6.0),
    "hot_board": dict(board=345.0, ambient=305.0, emissivity=0.85, coefficient=6.0),
    "poor_sink": dict(board=320.0, ambient=300.0, emissivity=0.20, coefficient=1.0),
    "strong_sink": dict(board=315.0, ambient=295.0, emissivity=0.95, coefficient=40.0),
}


def build_models(package, operators, part, plan, plan_full, orders, cost):
    """One entry per method and interior order.

    ``rk``   rational Krylov on the interface coupling only.
    ``rkl``  rational Krylov on the interface coupling *and* the internal
             sources -- the load-aware space this study argues for.
    """
    m = part.port_count
    psi = rd.static_modes(part, cost)
    source = rd.source_directions(part, package.source, cost)
    models = {"guyan": rd.assemble(operators, part, np.empty((part.interior_count, 0)), psi)}
    for order in orders:
        models[f"cb_{order}"] = rd.assemble(operators, part, rd.craig_bampton(part, order), psi)
        models[f"rk_{order}"] = rd.assemble(operators, part, rd.rational_krylov(part, order, plan), psi)
        models[f"rkl_{order}"] = rd.assemble(operators, part, rd.rational_krylov(part, order, plan_full), psi)
        rk = rd.rational_krylov(part, order, plan)
        models[f"rk_src_{order}"] = rd.assemble(
            operators, part, np.column_stack((rk, rd.orthonormal(source, against=rk))), psi
        )
    return models


def linear_metrics(package, part, models, extra, h_full, scale, shifts, cost):
    """Closed-loop interface operator and source-to-probe transfer at real shifts.

    The open-loop ``K`` of a component whose only boundary is its ports is
    singular (uniform temperature), so the physically meaningful linear
    operator is the one closed by the linearized interface conductance
    ``diag(extra)``.  Every method is compared on that same closed operator.

    Both errors are **absolute** errors normalised by the DC gain, not by the
    local response.  A relative-to-local metric is degenerate at large shifts,
    where every response decays to the roundoff floor and the ratio is
    meaningless (measured: 1e-13 K signals reporting a 5e4 "relative error").
    """
    m = part.port_count
    probe = package.probes
    driven = package.source
    dc_scale = rd.scaled_norm(rd.exact_interface_operator(part, extra, 0.0, cost), scale)
    reference_operator, reference_transfer = [], []
    for shift in shifts:
        reference_operator.append(rd.exact_interface_operator(part, extra, shift, cost))
        a = (package.K + sp.diags(h_full) + shift * package.C).tocsc()
        reference_transfer.append(SPDSolver(a, cost=cost).solve(driven, label="transfer reference")[probe])
    dc_transfer = float(np.linalg.norm(reference_transfer[0]))
    # Band of engineering interest: where the closed-loop transfer still carries
    # at least 1% of its DC magnitude.  Above that the interface is purely
    # inertial, every response is negligible, and the ratio is roundoff.
    magnitudes = np.array([np.linalg.norm(item) for item in reference_transfer])
    inside = np.flatnonzero(magnitudes >= 0.01 * dc_transfer)
    band = slice(int(inside[0]), int(inside[-1]) + 1) if inside.size else slice(0, len(shifts))

    rows = []
    for name, model in models.items():
        errors, transfer = [], []
        for shift, ref, exact in zip(shifts, reference_operator, reference_transfer):
            errors.append(
                rd.scaled_norm(rd.reduced_interface_operator(model, extra, shift) - ref, scale) / dc_scale
            )
            reduced_system = model.K + shift * model.C
            reduced_system = reduced_system.copy()
            reduced_system[np.arange(m), np.arange(m)] += extra
            reduced = model.V[probe, :] @ dense_spd_solve(reduced_system, model.V.T @ driven, cost=cost)
            transfer.append(float(np.linalg.norm(reduced - exact) / dc_transfer))
        rows.append(
            {
                "method": name,
                "dimension": model.dimension,
                "interior_dimension": model.interior_dimension,
                "interface_error_band": float(max(errors[band])),
                "interface_error_slow": float(errors[0]),
                "interface_error_full": float(max(errors)),
                "transfer_error_band": float(max(transfer[band])),
                "transfer_error_full": float(max(transfer)),
            }
        )
    return rows


def interface_mass_diagnostics(package, part, models):
    """Why the interface operator differs between methods.

    ``C`` is diagonal, so ``C_Gi = 0`` and the exact interface operator has the
    high-shift asymptote ``s C_GG``.  Static condensation instead gives
    ``s C_r,GG`` with ``C_r,GG = C_GG + Psi^T C_ii Psi``, which is *larger* by a
    fixed amount that no enrichment can touch.  The only handle an interior
    space has is the subtraction ``C_Gz C_zz^-1 C_zG >= 0``.  The reported
    defect is therefore

        C_r,GG - C_Gz C_zz^-1 C_zG - C_GG,

    and zero is the exact target.
    """
    m = part.port_count
    exact = float(np.mean(np.diag(part.CGG)))
    rows = []
    for name, model in models.items():
        q = model.interior_dimension
        reduced = float(np.mean(np.diag(model.C[:m, :m])))
        if q:
            c_zz = model.C[m:, m:]
            c_zG = model.C[m:, :m]
            subtraction = float(np.mean(np.diag(c_zG.T @ np.linalg.solve(c_zz, c_zG))))
        else:
            subtraction = 0.0
        rows.append(
            {
                "method": name,
                "interior_dimension": q,
                "exact_interface_capacity": exact,
                "reduced_interface_capacity": reduced,
                "enrichment_subtraction": subtraction,
                "capacity_defect": reduced - subtraction - exact,
            }
        )
    return rows


def transient_metrics(reference, candidate):
    port = np.abs(candidate.port_temperature - reference.port_temperature)
    probe = np.abs(candidate.probe_temperature - reference.probe_temperature)
    rise = np.abs(reference.probe_temperature - reference.probe_temperature[0]).max(axis=0)
    rise = np.where(rise > 1.0e-9, rise, 1.0)
    power = np.abs(candidate.interface_power.sum(axis=1) - reference.interface_power.sum(axis=1))
    scale = max(np.abs(reference.interface_power.sum(axis=1)).max(), 1.0e-12)
    return {
        "port_error_K": float(port.max()),
        "probe_error_K": float(probe.max()),
        "probe_error_relative": float((probe / rise).max()),
        "interface_power_error_relative": float(power.max() / scale),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orders", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    args = parser.parse_args()

    package = pm.build_package()
    nominal = scenario(package, **SCENARIOS["nominal"])
    operators = normalized_operators(package.K, package.C, np.zeros(package.size))
    part = rd.partition(operators, package.interface)
    m = part.port_count

    extra = nominal.linear_conductance(REFERENCE_K, package.size)[part.interface]
    h_full = nominal.linear_conductance(REFERENCE_K, package.size)
    psi = rd.static_modes(part)
    schur = part.KGG + part.KiG.toarray().T @ psi
    scale = schur + np.diag(extra)
    closed_loop = float(package.C.sum()) / float(extra.sum())
    eigenvalues = scipy.linalg.eigh(part.Kii.toarray(), part.Cii.toarray(), eigvals_only=True)
    slow, fast = float(eigenvalues[0]), float(eigenvalues[-1])
    shifts = np.geomspace(0.5 / closed_loop, fast / 2.0, args.samples)

    print(
        "MODEL "
        + json.dumps(
            {
                "cells": package.size,
                "port_dofs": m,
                "interior_dofs": part.interior_count,
                "interior_time_constants_s": [1.0 / fast, 1.0 / slow],
                "closed_loop_time_constant_s": closed_loop,
                "capacity_J_per_K": float(package.C.sum()),
                "top_port_conductance_W_per_K": float(extra[: m // 2].sum()),
                "bottom_port_conductance_W_per_K": float(extra[m // 2 :].sum()),
                "shift_band_rad_per_s": [float(shifts[0]), float(shifts[-1])],
            }
        ),
        flush=True,
    )

    cost = Cost()
    plan = rd.ResponsePlan.build(part, SHIFTS, cost)
    inputs_full = np.column_stack(
        [(part.KiG + 0.0 * part.CiG).toarray(), np.asarray(package.source)[part.interior, :]]
    )
    plan_full = rd.ResponsePlan.build(part, SHIFTS, cost, inputs=inputs_full)
    models = build_models(package, operators, part, plan, plan_full, args.orders, cost)
    print(f"offline: {json.dumps(cost.as_dict())} ({2 * len(SHIFTS)} AMG setups for the two plans)", flush=True)

    rows = linear_metrics(package, part, models, extra, h_full, scale, shifts, cost)
    print("\nQ1  LINEAR, CLOSED-LOOP OPERATOR (absolute error normalised by the DC gain)")
    header = f"{'method':<14}{'dim':>5}{'q':>5}{'iface':>11}{'iface all':>11}{'transfer':>11}{'transf all':>11}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['method']:<14}{row['dimension']:>5}{row['interior_dimension']:>5}"
            f"{row['interface_error_band']:>11.3e}{row['interface_error_full']:>11.3e}"
            f"{row['transfer_error_band']:>11.3e}{row['transfer_error_full']:>11.3e}"
        )

    power = np.array([2.0, 0.8])
    common = dict(dt=args.dt, duration=args.duration, port_cells=package.interface)

    print("\nQ1b INTERFACE CAPACITY DEFECT (the only part an interior basis controls)")
    mass_rows = interface_mass_diagnostics(package, part, models)
    print(f"  {'method':<14}{'q':>5}{'C_r,GG':>13}{'subtraction':>14}{'defect':>13}{'defect %':>10}")
    for row in mass_rows:
        percent = 100.0 * row["capacity_defect"] / row["reduced_interface_capacity"]
        print(
            f"  {row['method']:<14}{row['interior_dimension']:>5}"
            f"{row['reduced_interface_capacity']:>13.4e}{row['enrichment_subtraction']:>14.3e}"
            f"{row['capacity_defect']:>13.3e}{percent:>10.1f}"
        )

    print("\nQ0  NONLINEAR STEADY STATE: port temperatures must be exact for every enrichment")
    for label in args.scenarios:
        settings = SCENARIOS[label]
        law = scenario(package, **settings)
        reference = tr.steady(
            operators, law, package.source, package.probes, power, ambient=settings["ambient"]
        )
        worst = 0.0
        for name, model in models.items():
            candidate = tr.steady_reduced(
                model, law, package.source, package.probes, power, ambient=settings["ambient"]
            )
            full = model.V @ candidate
            worst = max(worst, float(np.abs(full[part.interface] - reference[part.interface]).max()))
        print(f"  [{label}] worst port temperature deviation over all methods: {worst:.3e} K")
        if worst > 1.0e-6:
            raise AssertionError(f"steady interface relation is not exact in {label}: {worst:.3e} K")

    print(f"\nQ2  NONLINEAR TRANSIENT, full order vs reduced (dt={args.dt:g} s, {args.duration:g} s)")
    transient_rows, reference_costs = [], {}
    for label in args.scenarios:
        settings = SCENARIOS[label]
        law = scenario(package, **settings)
        reference_cost = Cost()
        started = time.perf_counter()
        reference = tr.integrate(
            operators, law, package.source, package.probes, power,
            ambient=settings["ambient"], cost=reference_cost, **common,
        )
        seconds = time.perf_counter() - started
        reference_costs[label] = dict(reference_cost.as_dict(), seconds=round(seconds, 2))
        print(
            f"\n  [{label}] full order {seconds:.1f} s, {reference_cost.as_dict()}, "
            f"peak port {reference.port_temperature.max():.2f} K, "
            f"peak junction {reference.probe_temperature[:, 0].max():.2f} K"
        )
        for name, model in models.items():
            reduced_cost = Cost()
            candidate = tr.integrate_reduced(
                model, law, package.source, package.probes, power,
                ambient=settings["ambient"], cost=reduced_cost, **common,
            )
            metrics = transient_metrics(reference, candidate)
            metrics.update(
                {
                    "scenario": label,
                    "method": name,
                    "dimension": model.dimension,
                    "dense_solves": reduced_cost.dense_cholesky,
                }
            )
            transient_rows.append(metrics)
            print(
                f"    {name:<14} dim={model.dimension:<5}"
                f" port={metrics['port_error_K']:.3e} K"
                f" probe={metrics['probe_error_K']:.3e} K"
                f" rel={metrics['probe_error_relative']:.3e}"
                f" power={metrics['interface_power_error_relative']:.3e}"
            )

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "summary.json").write_text(
        json.dumps(
            {
                "model": {
                    "cells": package.size,
                    "port_dofs": m,
                    "interior_dofs": part.interior_count,
                    "closed_loop_time_constant_s": closed_loop,
                },
                "orders": args.orders,
                "dt": args.dt,
                "duration": args.duration,
                "offline_cost": cost.as_dict(),
                "reference_cost": reference_costs,
                "linear": rows,
                "mass": mass_rows,
                "transient": transient_rows,
            },
            indent=2,
        )
    )
    for name, table in (("linear", rows), ("transient", transient_rows)):
        with (RESULTS / f"{name}.csv").open("w") as handle:
            columns = list(table[0])
            handle.write(",".join(columns) + "\n")
            for row in table:
                handle.write(",".join(str(row[key]) for key in columns) + "\n")
    print(f"\nwrote {RESULTS / 'summary.json'}, linear.csv, transient.csv")


if __name__ == "__main__":
    main()
