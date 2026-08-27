#!/usr/bin/env python3
"""Interface-node coupling experiments for the Case-1 BCI-ROM model.

Migrated onto the *embeddable* ROM mechanism (``metahotspot.macromodel.embeddable``):
the source-bearing upper subdomain is reduced **once** and exposes all its
boundary faces as connectable ports; each case below reuses that single
extraction and merely connects the bottom cut port ``z-`` to a different
external side:

* ``conforming``      — fine upper (identity) ↔ fine lower (identity)
* ``nonconforming``   — fine upper (identity) ↔ coarse lower (identity)
* ``rom_fvm``         — reduced upper ↔ fine lower (identity)
"""
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
RESULT_PATH = ROOT / "results" / "experiments" / "interface_coupling_experiments.json"


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


def monolithic_reference(model):
    """Native monolithic junction temperatures (K) at the declared BCs."""
    full = model.full_reference(BOUNDARY_H)
    return model.junction_temperature(full.steady_temperature)


def upper_junction(state, side, _side_order):
    """Left-side junction temperatures (K) from a coupled state (block at offset 0)."""
    return AMBIENT_K + er.side_junction_rise(state, side, 0)


def lower_junction(state, right, offset):
    return AMBIENT_K + er.side_junction_rise(state, right, offset)


def run_case(left, right, lport, rport, reference):
    K, C, rhs, left_order, right_order, npatch = er.connect(
        left, right, lport, rport, power=POWER_W
    )
    steady, history = er.solve_system(K, C, rhs, DT_S, DURATION_S)
    junction = np.r_[
        upper_junction(steady, left, left_order),
        lower_junction(steady, right, left_order + npatch),
    ]
    interface_state = steady[left_order : left_order + npatch]
    return {
        "basis_order": [int(left.order), int(right.order)],
        "interface_patches": int(npatch),
        "steady_junction_K": junction.tolist(),
        "reference_junction_K": np.asarray(reference).tolist(),
        "steady_interface_peak_rise_K": float(interface_state.max()),
        "max_junction_error_K": float(np.max(np.abs(junction - reference))),
        "transient_steps": int(history.shape[0]),
    }


def run():
    fine_model = make_model(2.5)
    coarse_model = make_model(3.0)
    fine_upper_cells = split_cells(fine_model, True)
    fine_lower_cells = split_cells(fine_model, False)
    coarse_lower_cells = split_cells(coarse_model, False)

    fine_upper = er.build_subdomain(
        fine_model, fine_upper_cells, name="fine_upper", physical_h=BOUNDARY_H
    )
    fine_lower = er.build_subdomain(
        fine_model, fine_lower_cells, name="fine_lower", physical_h=BOUNDARY_H
    )
    coarse_lower = er.build_subdomain(
        coarse_model, coarse_lower_cells, name="coarse_lower", physical_h=BOUNDARY_H
    )

    # Extract the whole-subdomain EmbeddableRom once and reuse its physical
    # interface traces in every coupling case.
    rom = er.extract_rom(
        fine_upper,
        tolerance=1.0e-2,
        max_order=2048,
        probe_rounds=2,
        seed=20260825,
    )
    summary = rom.summary

    monolithic = monolithic_reference(fine_model)[:4]  # upper ports S0..S3
    reference_full = np.r_[monolithic, np.full(4, AMBIENT_K)]  # lower: ambient
    conforming = run_case(
        fine_upper,
        fine_lower,
        fine_upper.port("z-"),
        fine_lower.port("z+"),
        reference_full,
    )
    nonconforming = run_case(
        fine_upper,
        coarse_lower,
        fine_upper.port("z-"),
        coarse_lower.port("z+"),
        reference_full,
    )
    rom_fvm = run_case(
        rom, fine_lower, rom.port("z-"), fine_lower.port("z+"), reference_full
    )


    payload = {
        "method": "EmbeddableRom (extract once, connect everywhere): "
        "whole-subdomain cells reduced by one basis; independent interface "
        "face nodes theta_if coupled through the physical trace V_if; "
        "non-conforming grids area-weighted (E, xi) at model-definition level; "
        f"per-face conductance g = k*A/half; upper ports = "
        f"{[p.normal for p in rom.ports]}",
        "rom_extraction": {
            "basis_order": int(rom.order),
            "seconds": float(summary["seconds"]),
            "relative_response_error": float(summary["relative_response_error"]),
            "ports": [p.normal for p in rom.ports],
        },
        "identity_vs_monolithic_max_error_K": float(
            np.max(np.abs(np.asarray(conforming["steady_junction_K"])[:4] - monolithic))
        ),
        "results": {
            "conforming": conforming,
            "nonconforming": nonconforming,
            "rom_fvm": rom_fvm,
        },
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    run()
