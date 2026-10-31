#!/usr/bin/env python3
"""Time-step convergence for the nonlinear transient comparison.

The method ranking is only a result if it survives step refinement.  This
script refines the step until the full-order solution itself stops moving by
more than the between-method differences, and reports both sequences.
"""

from __future__ import annotations

import json

import numpy as np

import package_model as pm
import reduction as rd
import transient as tr
from metahotspot.macromodel.utils import normalized_operators

SHIFTS = np.geomspace(0.05, 3.0e4, 9)
DURATION = 3.0
STEPS = (0.02, 0.01, 0.005, 0.0025)


def main() -> None:
    package = pm.build_package()
    law = pm.default_interface(package)
    operators = normalized_operators(package.K, package.C, np.zeros(package.size))
    part = rd.partition(operators, package.interface)
    psi = rd.static_modes(part)
    plan = rd.ResponsePlan.build(part, SHIFTS)
    source = rd.source_directions(part, package.source)

    models = {
        "guyan": rd.assemble(operators, part, np.empty((part.interior_count, 0)), psi),
        "cb_16": rd.assemble(operators, part, rd.craig_bampton(part, 16), psi),
        "rk_src_8": rd.assemble(
            operators,
            part,
            np.column_stack((rd.rational_krylov(part, 8, plan), rd.orthonormal(source))),
            psi,
        ),
    }

    power = np.array([2.0, 0.8])
    histories = {}
    for dt in STEPS:
        common = dict(dt=dt, duration=DURATION, port_cells=package.interface, ambient=300.0)
        histories[dt] = {
            "full": tr.integrate(operators, law, package.source, package.probes, power, **common)
        }
        for name, model in models.items():
            histories[dt][name] = tr.integrate_reduced(
                model, law, package.source, package.probes, power, **common
            )
        print(f"  integrated dt={dt:g}", flush=True)

    def on_coarse_grid(fine, coarse):
        ratio = (fine.times.size - 1) // (coarse.times.size - 1)
        index = np.arange(coarse.times.size) * ratio
        return fine.port_temperature[index], fine.probe_temperature[index]

    def compare(coarse_history, fine_history):
        port, probe = on_coarse_grid(fine_history, coarse_history)
        return (
            float(np.abs(port - coarse_history.port_temperature).max()),
            float(np.abs(probe - coarse_history.probe_temperature).max()),
        )

    print("\nFull-order self-convergence (successive refinement):")
    for coarse_dt, fine_dt in zip(STEPS[:-1], STEPS[1:]):
        port, probe = compare(histories[coarse_dt]["full"], histories[fine_dt]["full"])
        print(f"  dt {coarse_dt:>7g} -> {fine_dt:<7g} port {port:.3e} K   probe {probe:.3e} K")

    print("\nReduced-vs-full, port channel:")
    print(f"  {'method':<12}" + "".join(f"{'dt=' + str(dt):>13}" for dt in STEPS))
    summary = {}
    for name in models:
        row = f"  {name:<12}"
        for dt in STEPS:
            port, probe = compare(histories[dt]["full"], histories[dt][name])
            summary[f"{name}@{dt}"] = {"port": port, "probe": probe}
            row += f"{port:>13.3e}"
        print(row)

    print("\nReduced-vs-full, probe channel:")
    print(f"  {'method':<12}" + "".join(f"{'dt=' + str(dt):>13}" for dt in STEPS))
    for name in models:
        row = f"  {name:<12}"
        for dt in STEPS:
            row += f"{summary[f'{name}@{dt}']['probe']:>13.3e}"
        print(row)

    print("\n" + json.dumps(summary, indent=None))


if __name__ == "__main__":
    main()
