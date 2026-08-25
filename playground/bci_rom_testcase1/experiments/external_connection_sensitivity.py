#!/usr/bin/env python3
"""Test whether an extracted ROM is independent of its external load."""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from interface_coupling_experiments import (
    BOUNDARY_H,
    POWER_W,
    build_full_system,
    build_interface,
    build_rom,
    build_side,
    identity_basis,
    make_model,
    run_case,
)

RESULT_PATH = (
    Path(__file__).resolve().parents[3]
    / "results"
    / "experiments"
    / "external_connection_sensitivity.json"
)


def lower_bottom_pattern(model, lower):
    """Return a lower side with a spatially varying bottom HTC."""
    bottom_area = lower.boundary_terms[1].diagonal()
    centers = model.cell_layout.centers[lower.cells]
    parity = (np.floor((centers[:, 0] + 0.03) / 0.005) + np.floor((centers[:, 1] + 0.05) / 0.005)) % 2
    factor = np.where(parity == 0, 0.1, 1.9)
    effective_bottom_h = model.physical_to_effective(BOUNDARY_H)[1]
    delta = effective_bottom_h * (factor - 1.0) * bottom_area
    stiffness = lower.stiffness + sp.diags(delta)
    return replace(lower, stiffness=stiffness.tocsc())


def lower_active_source(model, lower):
    """Return a lower side with one additional localized source port."""
    centers = model.cell_layout.centers[lower.cells]
    target = np.array([0.0, 0.0, 0.005])
    cell = int(np.argmin(np.linalg.norm(centers - target, axis=1)))
    source = lower.source.copy()
    source[cell, 0] = 1.0
    return replace(lower, source=source)


def upper_junctions(upper, lower, interface, upper_basis, lower_basis):
    result = run_case(
        upper,
        lower,
        interface,
        upper_basis,
        lower_basis,
        np.zeros(upper.source.shape[1] * 2),
    )
    return np.asarray(result["steady_junction_K"][: upper.source.shape[1]])


def evaluate_case(name, upper, lower, interface, rom_basis, identity_lower, reference):
    rom = upper_junctions(upper, lower, interface, rom_basis, identity_lower)
    detailed = upper_junctions(
        upper,
        lower,
        interface,
        identity_basis(upper)[0],
        identity_lower,
    )
    return {
        "reference_upper_junction_K": reference.tolist(),
        "detailed_upper_junction_K": detailed.tolist(),
        "rom_upper_junction_K": rom.tolist(),
        "detailed_vs_reference_max_error_K": float(np.max(np.abs(detailed - reference))),
        "rom_vs_detailed_max_error_K": float(np.max(np.abs(rom - detailed))),
    }


def run():
    model = make_model(2.5)
    stiffness, capacitance, source = build_full_system(model)
    upper = build_side(model, stiffness, capacitance, source, True)
    lower = build_side(model, stiffness, capacitance, source, False)
    interface = build_interface(upper, lower)
    identity_upper, _ = identity_basis(upper)
    identity_lower, _ = identity_basis(lower)
    area_by_cell = np.bincount(
        upper.interface_cells[interface.upper_cells],
        weights=interface.areas,
        minlength=upper.cells.size,
    )
    rom, summary = build_rom(upper, area_by_cell)

    baseline_reference = upper_junctions(
        upper, lower, interface, identity_upper, identity_lower
    )
    patterned_lower = lower_bottom_pattern(model, lower)
    patterned_interface = build_interface(upper, patterned_lower)
    active_lower = lower_active_source(model, lower)
    active_interface = build_interface(upper, active_lower)

    results = {
        "baseline_uniform_external": evaluate_case(
            "baseline", upper, lower, interface, rom, identity_lower, baseline_reference
        ),
        "nonuniform_external_boundary": evaluate_case(
            "patterned", upper, patterned_lower, patterned_interface, rom,
            identity_basis(patterned_lower)[0],
            upper_junctions(upper, patterned_lower, patterned_interface,
                            identity_upper, identity_basis(patterned_lower)[0]),
        ),
        "active_external_subdomain": evaluate_case(
            "active", upper, active_lower, active_interface, rom,
            identity_basis(active_lower)[0],
            upper_junctions(upper, active_lower, active_interface,
                            identity_upper, identity_basis(active_lower)[0]),
        ),
    }
    payload = {
        "question": "Does changing the external connection change the result of a fixed ROM?",
        "extraction": {
            "external_structure_used": "uniform interface port plus declared boundary groups",
            "basis_order": int(rom.shape[1]),
            "seconds": float(summary["seconds"]),
            "relative_response_error": float(summary["relative_response_error"]),
        },
        "results": results,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    run()
