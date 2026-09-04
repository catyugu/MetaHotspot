#!/usr/bin/env python3
"""Test whether an *extracted* embeddable ROM is independent of its external load."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = ROOT / "playground" / "bci_rom_testcase1"
sys.path[:0] = [str(CASE_DIR)]

from model_case1 import Case1Config, Case1Model  # noqa: E402
from metahotspot.macromodel import embeddable as er  # noqa: E402

AMBIENT_K = 308.15
BOUNDARY_H = (5.0e1, 1.0e3)
POWER_W = np.array([0.1, 0.2, 0.3, 0.4])
INTERFACE_Z = 10.0e-3
DT_S = 5.0
DURATION_S = 100.0

RESULT_PATH = ROOT / "results" / "experiments" / "external_connection_sensitivity.json"


def split_cells(model, upper):
    z = model.cell_layout.centers[:, 2]
    if upper:
        return np.flatnonzero(z >= INTERFACE_Z - 1.0e-12)
    return np.flatnonzero(z < INTERFACE_Z - 1.0e-12)


def make_model(cell_size_mm):
    return Case1Model(
        Case1Config(
            max_xy_cell_mm=cell_size_mm,
            max_z_cell_mm=2.5,
            dt_s=DT_S,
            duration_s=DURATION_S,
        )
    )


def build_patterned_lower(model, lower_cells):
    """Lower side with a spatially varying bottom HTC (folded via ambient_diag)."""
    full = model.full_cell_count
    effective_bottom_h = model.physical_to_effective(BOUNDARY_H)[1]
    bot_cells = np.asarray(model.boundary_groups()[1].cells, dtype=np.int64)
    area = np.asarray(model.boundary_terms[1].diagonal()).ravel()
    centers = model.cell_layout.centers[bot_cells]
    parity = (
        np.floor((centers[:, 0] + 0.03) / 0.005)
        + np.floor((centers[:, 1] + 0.05) / 0.005)
    ) % 2
    factor = np.where(parity == 0, 0.1, 1.9)
    diag = np.zeros(full)
    diag[bot_cells] = effective_bottom_h * factor * area[bot_cells]
    return er.build_subdomain(
        model, lower_cells, name="patterned_lower", ambient_diag=diag
    )


def build_active_lower(model, lower_cells):
    """Lower side with one additional localized source port (column 0)."""
    lower = er.build_subdomain(
        model, lower_cells, name="active_lower", physical_h=BOUNDARY_H
    )
    centers = model.cell_layout.centers[lower.cells]
    target = np.array([0.0, 0.0, 0.005])
    cell = int(np.argmin(np.linalg.norm(centers - target, axis=1)))
    source = lower.source.copy()
    source[cell, 0] = 1.0
    lower.source = source
    return lower


def upper_junctions(upper_side, lower, lport_label="z-", rport_label="z+"):
    """Upper-side junction temperatures (K) for a given external lower side."""
    K, C, rhs, left_order, right_order, npatch = er.connect(
        upper_side,
        lower,
        upper_side.port(lport_label),
        lower.port(rport_label),
        power=POWER_W,
    )
    steady, _ = er.solve_system(K, C, rhs, DT_S, DURATION_S)
    return AMBIENT_K + er.side_junction_rise(steady, upper_side, 0)


def evaluate_case(upper, upper_identity, lower, rom):
    """Compare detailed (identity upper) vs embedded-ROM upper junction."""
    detailed = upper_junctions(upper_identity, lower)
    rom_j = upper_junctions(upper, lower)
    return {
        "detailed_upper_junction_K": detailed.tolist(),
        "rom_upper_junction_K": rom_j.tolist(),
        "rom_vs_detailed_max_error_K": float(np.max(np.abs(rom_j - detailed))),
    }


def run():
    model = make_model(2.5)
    upper_cells = split_cells(model, True)
    lower_cells = split_cells(model, False)

    upper_identity = er.build_subdomain(
        model, upper_cells, name="upper", physical_h=BOUNDARY_H
    )
    uniform_lower = er.build_subdomain(
        model, lower_cells, name="uniform_lower", physical_h=BOUNDARY_H
    )

    # Extract the whole-subdomain EmbeddableRom once and reuse its physical
    # interface traces for every external structure.
    rom = er.extract_rom(
        upper_identity,
        tolerance=1.0e-2,
        max_order=2048,
        probe_rounds=2,
        seed=20260825,
    )
    summary = rom.summary

    patterned_lower = build_patterned_lower(model, lower_cells)
    active_lower = build_active_lower(model, lower_cells)

    results = {
        "baseline_uniform_external": evaluate_case(
            rom, upper_identity, uniform_lower, rom
        ),
        "nonuniform_external_boundary": evaluate_case(
            rom, upper_identity, patterned_lower, rom
        ),
        "active_external_subdomain": evaluate_case(
            rom, upper_identity, active_lower, rom
        ),
    }
    payload = {
        "question": (
            "Does changing the external connection change the result of a fixed, "
            "once-extracted embeddable ROM?"
        ),
        "extraction": {
            "external_structure_used": "same upper ROM reused verbatim across all "
            "three lower-side structures (uniform / patterned BC / active subdomain)",
            "basis_order": int(rom.m),
            "seconds": float(summary["seconds"]),
            "relative_response_error": float(summary["relative_response_error"]),
            "ports": [p.normal for p in rom.ports],
        },
        "results": results,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    run()
