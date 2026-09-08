"""Research experiments for the 2025-08-25 weekly report.

The experiment records parameter extrapolation, extraction amortization.
It writes machine-readable data and figures;
the report presents the scientific conclusions rather than implementation
names.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[3]  # repo root
CASE = PROJECT / "playground" / "bci_rom_testcase1"
sys.path[:0] = [str(CASE)]
from model_case1 import Case1Config, Case1Model
from metahotspot.macromodel.utils import (
    assemble_reduced_k,
    build_parametric_basis,
    project_bci,
    solve_rom_steady,
    solve_rom_transient,
)

OUT = PROJECT / "results" / "experiments"
AMB = 308.15
POWER = np.array([0.1, 0.2, 0.3, 0.4])


def make_model(h_ranges):
    return Case1Model(
        Case1Config(
            max_xy_cell_mm=2.5,
            max_z_cell_mm=2.5,
            duration_s=100.0,
            dt_s=5.0,
            h_ranges=tuple(h_ranges),
        )
    )


def run_rom(model, basis, h):
    core, G, terms = (
        model.core,
        model.source_shape,
        model.boundary_terms,
    )
    C, K0, F, Fb, Ab = project_bci(core, G, terms, basis)
    K = assemble_reduced_k(K0, Fb, Ab, model.physical_to_effective(h))
    t0 = time.perf_counter()
    ss = solve_rom_steady(K, F, POWER)
    online = time.perf_counter() - t0
    _, hist = solve_rom_transient(
        C, K, F, lambda _: POWER, model.dt, model.config.duration_s
    )
    return ss, hist, online, F


def max_transient_junction_error_pct(reference_steady, reference_history, rom_history):
    """Worst junction trajectory error, normalized by each steady rise."""
    steady_rise = np.maximum(np.abs(reference_steady - AMB), 1e-12)
    per_junction = (
        100 * np.max(np.abs(rom_history - reference_history), axis=0) / steady_rise
    )
    return float(np.max(per_junction))


def extrapolation():
    cases = [
        ("1e-2..1e6", ((1e-2, 1e6), (1e-2, 1e6))),
        ("1..1e4", ((1, 1e4), (1, 1e4))),
        ("10..1e3", ((10, 1e3), (10, 1e3))),
    ]
    tests = [
        (a, b)
        for a in [1e-2, 1e-1, 1, 1e2, 1e4, 1e5, 1e6]
        for b in [1e-2, 1e-1, 1, 1e2, 1e4, 1e5, 1e6]
    ]
    rows = []
    extraction = []
    for name, ranges in cases:
        model = make_model(ranges)
        core, G, terms = (
            model.core,
            model.source_shape,
            model.boundary_terms,
        )
        t0 = time.perf_counter()
        basis, summary = build_parametric_basis(
            core,
            G,
            terms,
            model.h_ranges(),
            tolerance=1e-3,
            max_order=256,
            probe_rounds=2,
            seed=20260805,
        )
        ext = time.perf_counter() - t0
        extraction.append(
            dict(
                training=name,
                seconds=ext,
                order=int(basis.shape[1]),
                cells=int(model.full_cell_count),
                candidates=summary["processed_candidate_count"],
            )
        )
        for h in tests:
            ref = model.full_reference(h)
            ss, hist, online, F = run_rom(model, basis, h)
            refj = model.junction_temperature(ref.steady_temperature)
            romj = AMB + F.T @ ss
            steady_error = float(
                100 * np.max(np.abs(romj - refj)) / max(np.max(refj - AMB), 1e-12)
            )
            transient_error = max_transient_junction_error_pct(
                refj,
                model.junction_temperature(ref.history),
                AMB + hist @ F,
            )
            rows.append(
                dict(
                    training=name,
                    h1=h[0],
                    h2=h[1],
                    steady_error_pct=steady_error,
                    transient_max_error_pct=transient_error,
                    online_s=online,
                    extraction_s=ext,
                    order=int(basis.shape[1]),
                )
            )
    return rows, extraction


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, extraction = extrapolation()
    (OUT / "results.json").write_text(
        json.dumps(dict(extraction=extraction, extrapolation=rows), indent=2),
        encoding="utf-8",
    )
    # Steady and transient worst junction errors over the extrapolation plane.
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    training_names = ["1e-2..1e6", "1..1e4", "10..1e3"]
    metrics = [
        ("steady_error_pct", "steady junction error (%)"),
        ("transient_max_error_pct", "transient max junction error (%)"),
    ]
    for row_axes, (metric, label) in zip(axes, metrics):
        for ax, name in zip(row_axes, training_names):
            a = [r for r in rows if r["training"] == name]
            xs = sorted(set(r["h1"] for r in a))
            ys = sorted(set(r["h2"] for r in a))
            Z = np.array(
                [
                    [
                        next(r[metric] for r in a if r["h1"] == x and r["h2"] == y)
                        for x in xs
                    ]
                    for y in ys
                ]
            )
            im = ax.imshow(
                Z,
                origin="lower",
                aspect="auto",
                extent=[
                    np.log10(xs[0]),
                    np.log10(xs[-1]),
                    np.log10(ys[0]),
                    np.log10(ys[-1]),
                ],
                cmap="magma",
                vmin=0,
            )
            ax.set_title(name)
            ax.set_xlabel("log10(h1)")
            ax.set_ylabel("log10(h2)")
            fig.colorbar(im, ax=ax, label=label)
    fig.savefig(OUT / "extrapolation_error.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = [x["training"] for x in extraction]
    ext = [x["seconds"] for x in extraction]
    orders = [x["order"] for x in extraction]
    ax.bar(names, ext, color=["#4472c4", "#ed7d31", "#70ad47"])
    ax.set_ylabel("basis extraction time (s)")
    ax.set_title("Training range: extraction cost")
    ax2 = ax.twinx()
    ax2.plot(names, orders, "ko-", label="order")
    ax2.set_ylabel("basis order")
    fig.tight_layout()
    fig.savefig(OUT / "extraction_cost.png", dpi=180)
    plt.close(fig)
    print(json.dumps(dict(extraction=extraction), indent=2))


if __name__ == "__main__":
    main()
