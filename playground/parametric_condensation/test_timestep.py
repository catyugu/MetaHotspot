#!/usr/bin/env python3
"""Time-step convergence for the four-component transient comparison.

A method ranking is only a result if it survives step refinement: the
full-order solution must move by less than the between-method differences.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=../../python \
        python test_timestep.py
"""

from __future__ import annotations

import numpy as np

import assemble as asm
import package_assembly as pa
import transient as tr

NOMINAL = {"lid": 25.0, "substrate": 3000.0}
SHIFTS = np.geomspace(1.0e-2, 1.0e5, 8)
STEPS = (0.04, 0.02, 0.01, 0.005)
DURATION = 3.0
METHODS = (("guyan", 0), ("craig_bampton", 24), ("rk_source", 12))


def main() -> None:
    package = pa.build()
    plans = {c.name: asm.build_plans(c, SHIFTS) for c in package.components}
    models = {
        f"{method}_{order}": {
            c.name: asm.extract(c, method=method, order=order,
                                plan=plans[c.name][0], plan_source=plans[c.name][1])
            for c in package.components
        }
        for method, order in METHODS
    }

    histories = {}
    for dt in STEPS:
        # Steady state at the nominal boundary condition, then a step change to
        # the corner: the response to a boundary-condition change is the BCI
        # scenario, and it does not excite the microsecond through-thickness
        # modes that made the ambient-start comparison step-limited.
        monolithic = asm.monolithic(package, NOMINAL)
        start = tr.steady(monolithic, ambient=300.0, power=pa.SOURCE_POWER)
        corner = {"lid": 5.0, "substrate": 2.0e4}
        histories[dt] = {
            "monolithic": tr.integrate(asm.monolithic(package, corner), ambient=300.0,
                                       power=pa.SOURCE_POWER, dt=dt, duration=DURATION,
                                       initial=start)
        }
        for name, model_set in models.items():
            system = asm.assemble(package, model_set, coefficients=NOMINAL, label=name)
            start_reduced = tr.steady(system, ambient=300.0, power=pa.SOURCE_POWER)
            target = asm.assemble(package, model_set, coefficients=corner, label=name)
            histories[dt][name] = tr.integrate(target, ambient=300.0, power=pa.SOURCE_POWER,
                                               dt=dt, duration=DURATION,
                                               initial=start_reduced)
        print(f"  integrated dt={dt:g}", flush=True)

    def on_coarse_grid(fine, coarse):
        ratio = (fine.times.size - 1) // (coarse.times.size - 1)
        index = np.arange(coarse.times.size) * ratio
        return fine.probe_temperature[index], fine.times[index]

    def compare(coarse_history, fine_history, window: float = 0.0):
        values, times = on_coarse_grid(fine_history, coarse_history)
        mask = times >= window
        return float(np.abs(values[mask] - coarse_history.probe_temperature[mask]).max())

    for window in (0.0, 0.1, 0.5):
        print(f"\n=== window t >= {window:g} s ===")
        print("Monolithic self-convergence (successive refinement):")
        for coarse_dt, fine_dt in zip(STEPS[:-1], STEPS[1:]):
            print(f"  dt {coarse_dt:>6g} -> {fine_dt:<6g} "
                  f"{compare(histories[coarse_dt]['monolithic'], histories[fine_dt]['monolithic'], window):.3e} K")

        print("Reduced vs monolithic probe error (K):")
        print(f"  {'method':<16}" + "".join(f"{'dt=' + str(dt):>13}" for dt in STEPS))
        for name in models:
            row = f"  {name:<16}"
            for dt in STEPS:
                row += f"{compare(histories[dt]['monolithic'], histories[dt][name], window):>13.3e}"
            print(row)

    print("\nIdentity: rational Krylov vs Guyan (C_Gz = 0, so only the source")
    print("response can differ; the interface dynamics must coincide)")
    for dt in STEPS:
        gap = float(np.abs(histories[dt]["rk_source_12"].probe_temperature
                           - histories[dt]["guyan_0"].probe_temperature).max())
        print(f"  dt={dt:g}: max |rk_source - guyan| = {gap:.3e} K")


if __name__ == "__main__":
    main()
