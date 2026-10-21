#!/usr/bin/env python3
"""Four-component embeddable ROM under BCI parameterisation.

Q0  Steady state over the whole BCI box.  If the BCI coefficient is a
    port-diagonal term, one extraction must reproduce the monolithic steady
    state at *every* h in the box.  This is the central claim.
Q1  Nonlinear transient at the box corners and the nominal point.
Q2  Port compression: what it costs once the interface law is nonlinear.
Q3  Cost: AMG setups and CG iterations, not wall clock.

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

import assemble as asm
import package_assembly as pa
import reduction as rd
import transient as tr
from linalg import Cost

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results"
SHIFTS = np.geomspace(1.0e-2, 1.0e5, 8)
NOMINAL = {"lid": 25.0, "substrate": 3000.0}
CORNERS = (
    {"lid": 5.0, "substrate": 50.0},
    {"lid": 5.0, "substrate": 2.0e4},
    {"lid": 5.0e3, "substrate": 50.0},
    {"lid": 5.0e3, "substrate": 2.0e4},
)
POWER = pa.SOURCE_POWER
AMBIENT_K = 300.0
METHODS = (("guyan", 0), ("craig_bampton", 8), ("craig_bampton", 24), ("rk_source", 12))


def box_points(samples: int) -> list[dict]:
    lid = np.geomspace(*asm.BCI_BOX["lid"], samples)
    board = np.geomspace(*asm.BCI_BOX["substrate"], samples)
    return [{"lid": float(a), "substrate": float(b)} for a in lid for b in board]


def build_models(package, shifts, orders, cost):
    plans = {c.name: asm.build_plans(c, shifts, cost) for c in package.components}
    models = {}
    for method, order in orders:
        models[f"{method}_{order}"] = {
            c.name: asm.extract(c, method=method, order=order,
                                plan=plans[c.name][0], plan_source=plans[c.name][1],
                                cost=cost)
            for c in package.components
        }
    return models, plans


def train_port_basis(package, coefficients_list, rank, cost):
    """Per-component port basis from monolithic port-temperature snapshots."""
    snapshots = {c.name: [] for c in package.components}
    for coefficients in coefficients_list:
        system = asm.monolithic(package, coefficients)
        state = tr.steady(system, ambient=AMBIENT_K, power=POWER, cost=cost)
        for component in package.components:
            # The whole port vector of the component, in its declared order.
            snapshots[component.name].append(
                state[component.cells[component.ports]] - AMBIENT_K
            )
    bases = {}
    for component in package.components:
        matrix = np.column_stack(snapshots[component.name])
        u, s, _ = np.linalg.svd(matrix, full_matrices=False)
        keep = min(rank, u.shape[1])
        bases[component.name] = np.ascontiguousarray(u[:, :keep])
    return bases


def interface_coupling(package, models):
    """``|C_Gz|`` per component: the only channel through which enrichment can
    change the interface operator.

    With ``C`` diagonal and uniform inside a component, ``C_ii = c I``.  Any
    enrichment orthogonal (in the Euclidean sense) to the static modes then has
    ``C_Gz = Psi^T C_ii W = c Psi^T W = 0``, so the reduced interface operator
    collapses to ``S(0) + s C_r,GG`` -- exactly the Guyan operator -- and the
    enrichment coordinates are never excited.  Craig-Bampton escapes this because
    its eigenmodes are C-orthonormal but not Euclidean-orthogonal to ``Psi``.
    This table is the mechanism behind the transient results, so it is reported
    rather than inferred.
    """
    rows = []
    for component in package.components:
        model = models[component.name]
        m = model.m
        if model.q:
            coupling = float(np.linalg.norm(model.C[:m, m:]))
        else:
            coupling = 0.0
        rows.append({"component": component.name, "q": model.q,
                     "capacity_coupling": coupling,
                     "capacity_scale": float(np.linalg.norm(model.C[:m, :m]))})
    return rows


def interface_capacity_defect(package, models):
    """Two numbers per component that explain the transient results.

    ``|C_Gz|`` is the only channel through which an enrichment space can change
    the interface operator.  With ``C`` diagonal and uniform inside a component,
    ``C_ii = c I``; any enrichment orthogonal (Euclidean) to the static modes has
    ``C_Gz = Psi^T C_ii W = c Psi^T W = 0``, so the reduced interface operator
    collapses to ``S(0) + s C_r,GG`` -- exactly the Guyan operator -- and the
    enrichment coordinates are never excited.  Craig-Bampton escapes this because
    its eigenmodes are C-orthonormal but not Euclidean-orthogonal to ``Psi``.

    ``mass_defect`` is ``|C_r,GG - C_GG| / |C_GG|``: static condensation adds
    ``Psi^T C_ii Psi`` to the interface capacity, and no enrichment can remove
    it.  That defect, not the interior basis, sets the residual transient error.
    """
    rows = []
    for component in package.components:
        model = models[component.name]
        m = model.m
        coupling = float(np.linalg.norm(model.C[:m, m:])) if model.q else 0.0
        physical = component.C[component.ports][:, component.ports].toarray()
        reduced = model.C[:m, :m]
        defect = float(np.linalg.norm(reduced - physical) / np.linalg.norm(physical))
        rows.append({"component": component.name, "q": model.q,
                     "capacity_coupling": coupling, "mass_defect": defect,
                     "interface_capacity": float(np.linalg.norm(physical))})
    return rows


def interior_energy_error(package, models, partitions, candidate_state, reference_state):
    """Relative energy-norm interior error, the quantity Galerkin minimizes.

    Comparing methods on a single point value is not meaningful for a nearly
    isothermal component: the exact interior source response is ~1e-6 K while the
    junction rise is ~38 K, so a point metric measures noise.  The ``K_ii``-norm
    of the interior error is the well-posed alternative and is the norm the
    projection actually controls.
    """
    offsets = {}
    cursor = 0
    for component in package.components:
        offsets[component.name] = cursor
        cursor += models[component.name].dimension

    error_squared = 0.0
    scale_squared = 0.0
    for component in package.components:
        model = models[component.name]
        part = partitions[component.name]
        base = offsets[component.name]
        state = candidate_state[base : base + model.dimension]
        lifted = model.V[part.interior, :] @ state
        exact = reference_state[component.cells[part.interior]]
        error = lifted - exact
        error_squared += float(error @ (part.Kii @ error))
        scale_squared += float(exact @ (part.Kii @ exact))
    return float(np.sqrt(error_squared / max(scale_squared, 1.0e-300)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--ranks", type=int, nargs="+", default=[4, 8, 16])
    args = parser.parse_args()

    started = time.perf_counter()
    package = pa.build()
    cost = Cost()
    points = box_points(args.samples)
    print(
        "MODEL "
        + json.dumps(
            {
                "cells": package.cell_count,
                "grid": package.grid,
                "components": [
                    {"name": c.name, "cells": c.size, "ports": int(c.ports.size),
                     "interior": c.size - int(c.ports.size)} for c in package.components
                ],
                "probes": list(package.probe_labels),
                "bci_box": {k: list(v) for k, v in asm.BCI_BOX.items()},
                "box_points": len(points),
            }
        ),
        flush=True,
    )

    models, plans = build_models(package, SHIFTS, METHODS, cost)
    partitions = {
        c.name: rd.partition(asm.Operators(c.K.tocsc(), c.C.tocsc()), c.ports)
        for c in package.components
    }
    print("offline " + json.dumps(cost.as_dict()), flush=True)

    print("\nQ-1  INTERFACE CAPACITY: enrichment channel and static-condensation defect")
    print(f"  {'method':<20}{'component':<11}{'q':>4}{'|C_Gz|':>12}{'mass defect':>14}")
    coupling_rows = {}
    for name, model_set in models.items():
        rows = interface_capacity_defect(package, model_set)
        coupling_rows[name] = rows
        for row in rows:
            print(f"  {name:<20}{row['component']:<11}{row['q']:>4}"
                  f"{row['capacity_coupling']:>12.3e}{row['mass_defect']:>14.3e}")

    # ---- Q0: steady state over the BCI box -------------------------------
    # The BCI claim is about the INTERFACE, not the interior.  Port temperatures
    # must be exact at every h with one extraction; interior probes are a
    # separate ROM-accuracy question and are reported separately.
    print("\nQ0a  PORT TEMPERATURES OVER THE BCI BOX (the BCI claim)")
    header = f"{'method':<20}{'dim':>5}" + "".join(
        f"{'h=(%g,%g)' % (p['lid'], p['substrate']):>18}" for p in points[:3]
    ) + f"{'  max over box':>16}"
    print(header)
    print("-" * len(header))
    q0_rows = []
    for name, model_set in models.items():
        port_gaps, probe_gaps = [], []
        dimension = 0
        for coefficients in points:
            # Same extraction, only the BCI coefficient changes: the assembly is
            # rebuilt (cheap, no solve) but nothing is re-extracted.
            candidate = asm.assemble(package, model_set, coefficients=coefficients, label=name)
            reference = asm.monolithic(package, coefficients)
            dimension = candidate.dimension
            reference_state = tr.steady(reference, ambient=AMBIENT_K, power=POWER, cost=cost)
            candidate_state = tr.steady(candidate, ambient=AMBIENT_K, power=POWER, cost=cost)
            reference_ports = reference.ports(reference_state)
            candidate_ports = candidate.ports(candidate_state)
            scale = max(
                float(np.max(np.abs(value - AMBIENT_K)))
                for value in reference_ports.values()
            )
            scale = max(scale, 1.0e-12)
            port_gaps.append(
                max(float(np.max(np.abs(candidate_ports[label] - value)))
                    for label, value in reference_ports.items()) / scale
            )
            reference_probe = reference.probe_map @ reference_state
            candidate_probe = candidate.probe_map @ candidate_state
            rise = max(float(np.max(np.abs(reference_probe - AMBIENT_K))), 1.0e-12)
            probe_gaps.append(
                float(np.max(np.abs(candidate_probe - reference_probe)) / rise)
            )
        q0_rows.append({
            "method": name, "dimension": dimension,
            "port_max_relative": max(port_gaps),
            "probe_max_relative": max(probe_gaps),
        })
        shown = "".join(f"{g:>18.3e}" for g in port_gaps[:3])
        print(f"{name:<20}{dimension:>5}{shown}{max(port_gaps):>16.3e}")

    worst = max(row["port_max_relative"] for row in q0_rows)
    print(f"\n  worst PORT deviation over {len(points)} BCI points: {worst:.3e}")
    if worst > 1.0e-6:
        raise AssertionError("BCI port steady state is not exact; the port-diagonal claim fails")

    print("\nQ0b  INTERIOR ACCURACY OVER THE BCI BOX (ROM accuracy, not BCI)")
    print("     The die is nearly isothermal (k/h/L ~ 289), so a point source")
    print("     produces a ~1e-6 K interior response: point errors must be")
    print("     normalised by that interior signal, not by the junction rise.")
    print(f"  {'method':<20}{'dim':>5}{'energy':>12}{'point':>12}{'interior signal':>17}")
    interior_rows = []
    for name, model_set in models.items():
        energies, point_errors = [], []
        signal = 0.0
        for coefficients in points:
            candidate = asm.assemble(package, model_set, coefficients=coefficients, label=name)
            reference = asm.monolithic(package, coefficients)
            reference_state = tr.steady(reference, ambient=AMBIENT_K, power=POWER, cost=cost)
            candidate_state = tr.steady(candidate, ambient=AMBIENT_K, power=POWER, cost=cost)
            energies.append(interior_energy_error(package, model_set, partitions,
                                                  candidate_state, reference_state))
            difference = np.abs(candidate.probe_map @ candidate_state
                                - reference.probe_map @ reference_state)
            point_errors.append(float(difference.max()))
            signal = max(signal, float(np.max(np.abs(
                reference.probe_map @ reference_state - AMBIENT_K))))
        interior_rows.append({
            "method": name,
            "energy_relative": max(energies),
            "point_error_K": max(point_errors),
            "interior_signal_K": signal,
        })
        print(f"  {name:<20}{model_set[package.components[0].name].dimension:>5}"
              f"{max(energies):>12.3e}{max(point_errors):>12.3e}{signal:>17.3e}")

    # ---- Q1: nonlinear transient ----------------------------------------
    # Scenario: the boundary condition changes while the power is constant --
    # the BCI use case.  Starting from the nominal steady state avoids exciting
    # the microsecond through-thickness modes, whose implicit-Euler error would
    # otherwise dominate the comparison.
    scenarios = [("nominal", NOMINAL)] + [(f"corner{i}", c) for i, c in enumerate(CORNERS)]
    window = 0.5
    print(f"\nQ1  NONLINEAR TRANSIENT: step change of the boundary condition")
    print(f"    (dt={args.dt:g} s, {args.duration:g} s, error taken for t >= {window:g} s)")
    q1_rows = []
    for label, coefficients in scenarios:
        reference_nominal = asm.monolithic(package, NOMINAL)
        reference_start = tr.steady(reference_nominal, ambient=AMBIENT_K, power=POWER, cost=cost)
        reference = asm.monolithic(package, coefficients)
        reference_cost = Cost()
        reference_history = tr.integrate(
            reference, ambient=AMBIENT_K, power=POWER, dt=args.dt, duration=args.duration,
            initial=reference_start, cost=reference_cost)
        mask = reference_history.times >= window
        baseline = reference_history.probe_temperature[0]
        rise = np.max(np.abs(reference_history.probe_temperature[mask] - baseline), axis=0)
        rise = np.where(rise > 1.0e-9, rise, 1.0)
        print(f"\n  [{label}] h_lid={coefficients['lid']:g} h_board={coefficients['substrate']:g}"
              f"  {json.dumps(reference_cost.as_dict())}")
        for name, model_set in models.items():
            nominal_system = asm.assemble(package, model_set, coefficients=NOMINAL, label=name)
            start = tr.steady(nominal_system, ambient=AMBIENT_K, power=POWER, cost=cost)
            system = asm.assemble(package, model_set, coefficients=coefficients, label=name)
            candidate_cost = Cost()
            candidate = tr.integrate(system, ambient=AMBIENT_K, power=POWER,
                                     dt=args.dt, duration=args.duration,
                                     initial=start, cost=candidate_cost)
            error = np.abs(candidate.probe_temperature - reference_history.probe_temperature)[mask]
            q1_rows.append({
                "scenario": label, "method": name, "dimension": system.dimension,
                "lid": coefficients["lid"], "substrate": coefficients["substrate"],
                "probe_error_K": float(error.max()),
                "probe_error_relative": float((error / rise).max()),
                "cg_solves": candidate_cost.cg_solves,
            })
            print(f"    {name:<20} dim={system.dimension:<5}"
                  f" max probe {error.max():.3e} K  rel {(error / rise).max():.3e}")

    # ---- Q2: port compression -------------------------------------------
    print("\nQ2  PORT COMPRESSION (basis trained on the BCI box)")
    train = [{"lid": a, "substrate": b}
             for a in np.geomspace(5.0, 5.0e3, 5)
             for b in np.geomspace(50.0, 2.0e4, 5)]
    q2_rows = []
    for rank in args.ranks:
        bases = train_port_basis(package, train, rank, cost)
        achieved = min(bases[c.name].shape[1] for c in package.components)
        compressed = {
            c.name: asm.extract(c, method="rk_source", order=12,
                                plan=plans[c.name][0], plan_source=plans[c.name][1],
                                port_basis=bases[c.name], cost=cost)
            for c in package.components
        }
        for label, coefficients in (("nominal", NOMINAL), ("corner3", CORNERS[3])):
            system = asm.assemble(package, compressed, coefficients=coefficients,
                                  label=f"compressed{rank}")
            reference = asm.monolithic(package, coefficients)
            reference_state = tr.steady(reference, ambient=AMBIENT_K, power=POWER, cost=cost)
            candidate_state = tr.steady(system, ambient=AMBIENT_K, power=POWER, cost=cost)
            reference_ports = reference.ports(reference_state)
            candidate_ports = system.ports(candidate_state)
            scale = max(float(np.max(np.abs(v - AMBIENT_K))) for v in reference_ports.values())
            port_gap = max(float(np.max(np.abs(candidate_ports[l] - v)))
                           for l, v in reference_ports.items()) / max(scale, 1.0e-12)
            reference_probe = reference.probe_map @ reference_state
            candidate_probe = system.probe_map @ candidate_state
            rise = max(float(np.max(np.abs(reference_probe - AMBIENT_K))), 1.0e-12)
            probe_gap = float(np.max(np.abs(candidate_probe - reference_probe)) / rise)
            q2_rows.append({"rank": achieved, "scenario": label,
                            "dimension": system.dimension,
                            "port_relative": port_gap, "probe_relative": probe_gap})
            print(f"  rank {achieved:<3} [{label}] dim={system.dimension:<5} "
                  f"port deviation {port_gap:.3e}  probe {probe_gap:.3e}")
        print("    ports per component: " + ", ".join(
            f"{c.name}={compressed[c.name].m}/{c.ports.size}" for c in package.components))

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "summary.json").write_text(
        json.dumps(
            {
                "model": {"cells": package.cell_count, "grid": package.grid},
                "bci_box": {k: list(v) for k, v in asm.BCI_BOX.items()},
                "dt": args.dt, "duration": args.duration,
                "offline_cost": cost.as_dict(),
                "interface_coupling": coupling_rows,
                "q0_steady_over_box": q0_rows,
                "q1_transient": q1_rows,
                "q2_compression": q2_rows,
                "seconds": round(time.perf_counter() - started, 1),
            },
            indent=2,
        )
    )
    for name, rows in (("q0_steady", q0_rows), ("q1_transient", q1_rows), ("q2_compression", q2_rows)):
        with (RESULTS / f"{name}.csv").open("w") as handle:
            columns = list(rows[0])
            handle.write(",".join(columns) + "\n")
            for row in rows:
                handle.write(",".join(str(row[key]) for key in columns) + "\n")
    print(f"\nwrote {RESULTS / 'summary.json'} and three CSVs "
          f"({round(time.perf_counter() - started, 1)} s)")


if __name__ == "__main__":
    main()
