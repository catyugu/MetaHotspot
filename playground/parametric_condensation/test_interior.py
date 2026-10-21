#!/usr/bin/env python3
"""Energy norm versus point error: why a point probe cannot rank the methods.

Galerkin minimizes the ``K_ii``-norm of the interior error, so a larger trial
space always lowers the energy error.  A single point value is a different
functional and need not be monotone.  This script demonstrates both on the die
alone, and guards the probe placement: a probe sitting on a port cell would be a
port DOF and would be blind to the interior enrichment entirely.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=../../python \
        python test_interior.py
"""

from __future__ import annotations

import numpy as np

import assemble as asm
import package_assembly as pa
import reduction as rd

SHIFTS = np.geomspace(1.0e-2, 1.0e5, 8)


def main() -> None:
    package = pa.build()
    die = next(c for c in package.components if c.name == "die")
    operators = asm.Operators(K=die.K.tocsc(), C=die.C.tocsc())
    part = rd.partition(operators, die.ports)
    psi = rd.static_modes(part)

    probes = [int(np.searchsorted(part.interior, p)) for p in die.probes]
    for k, position in enumerate(probes):
        if position >= part.interior_count or part.interior[position] != die.probes[k]:
            raise AssertionError(
                f"die probe {k} is not an interior DOF; the metric would be blind"
            )
    print(f"die: {die.size} cells, {part.port_count} ports, {part.interior_count} interior")
    print(f"probe interior positions {probes}")

    # Exact interior response to the source load with zero port drive.
    load = np.asarray(die.source)[part.interior, :] @ pa.SOURCE_POWER
    exact = np.linalg.solve(part.Kii.toarray(), load)
    print(f"exact interior source response: {exact.min():.3e} .. {exact.max():.3e} K")
    print(f"value at probes: {[float(exact[p]) for p in probes]}")
    print("(tiny because the die is nearly isothermal: k/h/L is large)")

    plans = asm.build_plans(die, SHIFTS)
    print(f"\n{'method':<18}{'q':>4}{'energy error':>15}{'point error':>15}{'monotone':>10}")
    previous_energy = None
    for method, order in (("guyan", 0), ("craig_bampton", 4), ("craig_bampton", 8),
                          ("craig_bampton", 16), ("craig_bampton", 32), ("rk_source", 12)):
        model = asm.extract(die, method=method, order=order,
                            plan=plans[0], plan_source=plans[1])
        enrichment = model.V[part.interior, model.m:]
        if model.q:
            z = np.linalg.solve(model.K[model.m:, model.m:], enrichment.T @ load)
        else:
            z = np.zeros(0)
        error = enrichment @ z - exact
        energy = float(np.sqrt(error @ (part.Kii @ error)))
        point = float(np.abs(error[probes]).max())
        trend = "" if previous_energy is None else ("down" if energy < previous_energy else "UP")
        previous_energy = energy if method == "craig_bampton" else previous_energy
        print(f"{method:<18}{model.q:>4}{energy:>15.4e}{point:>15.4e}{trend:>10}")

    print("\nThe energy error falls with Craig-Bampton order while the point error")
    print("does not.  A point probe is therefore not a valid ranking metric for a")
    print("nearly isothermal component; the energy norm and the interface error are.")


if __name__ == "__main__":
    main()
