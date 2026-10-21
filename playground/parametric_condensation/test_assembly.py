#!/usr/bin/env python3
"""Structural checks for the four-component assembly, before any ROM claim.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=../../python \
        python test_assembly.py
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

import assemble as asm
import package_assembly as pa
import reduction as rd
import transient as tr

NOMINAL = {"lid": 25.0, "substrate": 3000.0}
IDENTITY = 1.0e-9


def check(label, gap, tolerance=IDENTITY):
    status = "ok " if gap <= tolerance else "FAIL"
    print(f"  [{status}] {label:<52}{gap:.3e}")
    if gap > tolerance:
        raise AssertionError(f"{label}: {gap:.3e} > {tolerance:.1e}")


def main() -> None:
    package = pa.build()
    print(
        f"MODEL cells={package.cell_count} grid={package.grid} "
        f"area={package.area:.4e} m^2 probes={package.probe_labels}"
    )
    for component in package.components:
        faces = {face.label: face.cells.size for face in component.interfaces + component.ambient}
        print(
            f"  {component.name:<10} cells={component.size:<5} "
            f"ports={component.ports.size:<5} interior={component.size - component.ports.size:<5} "
            f"faces={faces}"
        )

    print("\n1  MONOLITHIC SYSTEM IS WELL POSED")
    reference = asm.monolithic(package, NOMINAL)
    operator = reference.linear_operator(NOMINAL)
    eigenvalues = np.linalg.eigvalsh(operator.toarray())
    print(f"  [info] smallest eigenvalue of K + diag(h A)  {eigenvalues[0]:.3e}")
    if eigenvalues[0] <= 0.0:
        raise AssertionError("the closed-loop operator must be SPD")

    print("\n2  INTERFACE CONDUCTANCE REPRODUCES THE MONOLITHIC FACE")
    worst = 0.0
    for lower, upper in pa.INTERFACES:
        low_component = next(c for c in package.components if c.name == lower)
        high_component = next(c for c in package.components if c.name == upper)
        low_face = next(f for f in low_component.interfaces if f.label == f"to_{upper}")
        high_face = next(f for f in high_component.interfaces if f.label == f"to_{lower}")
        for k in range(low_face.ij.shape[0]):
            g_low = float(low_face.conductance()[k])
            g_high = float(high_face.conductance()[k])
            series = g_low * g_high / (g_low + g_high)
            a = int(low_component.cells[low_face.cells[k]])
            b = int(high_component.cells[high_face.cells[k]])
            monolithic = -float(package.K[a, b])
            worst = max(worst, abs(series - monolithic) / abs(monolithic))
        print(f"  {lower}-{upper}: {low_face.ij.shape[0]} faces")
    check("series half-cell == monolithic face conductance", worst)

    print("\n3  FULL-PORT ROM REPRODUCES THE MONOLITHIC INTERFACE")
    power = pa.SOURCE_POWER
    reference_state = tr.steady(reference, ambient=300.0, power=power)
    reference_ports = reference.ports(reference_state)
    reference_probe = reference.probe_map @ reference_state
    print(f"  [info] monolithic junction rise  {reference_probe[0] - 300.0:.4f} K")

    shifts = np.geomspace(1.0e-2, 1.0e5, 8)
    plans = {c.name: asm.build_plans(c, shifts) for c in package.components}
    for method, order in (("guyan", 0), ("craig_bampton", 8), ("craig_bampton", 24),
                          ("rational_krylov", 12), ("rk_source", 12)):
        models = {
            component.name: asm.extract(
                component, method=method, order=order,
                plan=plans[component.name][0], plan_source=plans[component.name][1],
            )
            for component in package.components
        }
        assembled = asm.assemble(package, models, coefficients=NOMINAL, label=method)
        state = tr.steady(assembled, ambient=300.0, power=power)
        ports = assembled.ports(state)
        scale = max(float(np.max(np.abs(v - 300.0))) for v in reference_ports.values())
        port_gap = max(
            float(np.max(np.abs(ports[label] - value)))
            for label, value in reference_ports.items()
        ) / max(scale, 1.0e-12)
        probe_gap = float(
            np.max(np.abs(assembled.probe_map @ state - reference_probe))
            / max(np.max(np.abs(reference_probe - 300.0)), 1.0e-12)
        )
        print(
            f"  {method:<16} q={order:<3} dim={assembled.dimension:<5} "
            f"port {port_gap:.3e}   interior probe {probe_gap:.3e}"
        )
        # The interface must be exact for ANY enrichment; the interior is a
        # separate, finite ROM-accuracy question and is only reported.
        check(f"{method} q={order}: assembled PORT temperatures == monolithic", port_gap, 1.0e-7)

    print("\n4  BCI PARAMETER IS EXACTLY A PORT-DIAGONAL TERM")
    guyan = {c.name: asm.extract(c, method="guyan", order=0) for c in package.components}
    worst = 0.0
    for component in package.components:
        model = guyan[component.name]
        part = rd.partition(asm.Operators(component.K.tocsc(), component.C.tocsc()),
                            component.ports)
        psi = rd.static_modes(part)
        schur = part.KGG + part.KiG.toarray().T @ psi
        gap = float(np.linalg.norm(model.K[: model.m, : model.m] - schur) / np.linalg.norm(schur))
        worst = max(worst, gap)
        # Every component's open-loop S(0) is singular: its only boundary is its
        # own ports, so the uniform-temperature mode has no sink.  That is why
        # the error metrics must be taken on the *assembled*, closed system.
        relative_null = float(np.linalg.eigvalsh(schur)[0] / np.linalg.norm(schur))
        print(f"  {component.name:<10} q={model.q:<3} |S(0) - K_r,GG| = {gap:.2e}  "
              f"open-loop null {relative_null:.1e}")
    check("component port block == static Schur complement S(0)", worst, 1.0e-9)

    print("\n5  REUSE: A DIFFERENT LID RE-EXTRACTS ONLY THE LID")
    thick = tuple(
        pa.LayerSpec(spec.name, spec.conductivity, spec.volumetric_capacity,
                     spec.thickness * 2.0 if spec.name == "lid" else spec.thickness,
                     spec.cells, spec.fraction)
        for spec in pa.STACK
    )
    variant = pa.build(stack=thick)
    variant_models = dict(guyan)
    lid = next(c for c in variant.components if c.name == "lid")
    variant_models["lid"] = asm.extract(lid, method="guyan", order=0)
    variant_system = asm.assemble(variant, variant_models, coefficients=NOMINAL, label="thick lid")
    variant_reference = asm.monolithic(variant, NOMINAL)
    reference_variant = tr.steady(variant_reference, ambient=300.0, power=power)
    candidate = tr.steady(variant_system, ambient=300.0, power=power)
    reference_ports = variant_reference.ports(reference_variant)
    candidate_ports = variant_system.ports(candidate)
    scale = max(float(np.max(np.abs(v - 300.0))) for v in reference_ports.values())
    gap = max(
        float(np.max(np.abs(candidate_ports[label] - value)))
        for label, value in reference_ports.items()
    ) / max(scale, 1.0e-12)
    check("thick lid: assembled PORT temperatures == monolithic", gap, 1.0e-7)
    print(f"  [info] reused components: substrate, die, tim; re-extracted: lid")

    print("\nAll assembly identities hold.")


if __name__ == "__main__":
    main()
