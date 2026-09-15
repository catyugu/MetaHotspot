#!/usr/bin/env python3
"""Compare one reusable interface-enriched MetaHotspot EROM with FloTHERM."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SIMPLE_CASE = REPO_ROOT / "playground" / "simple_erom_case1"
sys.path.insert(0, str(SIMPLE_CASE))

import erom_attach_lib as attach  # noqa: E402
from stress_erom_attach import CASES  # noqa: E402
from metahotspot.macromodel.embeddable import build_subdomain  # noqa: E402
from metahotspot.macromodel.interface_enrichment import (  # noqa: E402
    clone_embeddable_rom,
    extract_interface_enriched_rom,
)

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "baseline" / "flotherm_erom" / "simple_case1_stress_baseline.csv"
OUT = ROOT / "results" / "simple_erom"
DEFAULT_CASES = (
    "baseline_copper",
    "bottom_htc_strong",
    "external_source_200w",
    "all_stress",
    "layered_extreme_source",
)


def read_baseline() -> dict[str, dict]:
    with BASELINE.open(newline="", encoding="utf-8") as f:
        return {row["case"]: row for row in csv.DictReader(f)}


def _trim_zero_sources(subdomain) -> None:
    source = np.asarray(subdomain.source, dtype=np.float64)
    nonzero = np.flatnonzero(np.abs(source).sum(axis=0) > 0.0)
    if nonzero.size == 0:
        raise ValueError("reusable cube EROM has no physical heat source")
    if nonzero.size < source.shape[1]:
        subdomain.source = source[:, nonzero]


def build_reusable_cube_rom(
    *,
    tolerance: float,
    contour_degree: int,
    probe_rounds: int,
):
    """Extract the cube once, independent of every external attachment case."""
    cfg = attach.AttachConfig()
    model = attach.AttachModel(cfg)
    cells = model._full.cells
    zc = attach.CellGeometry(cells).centers[:, 2]
    H_m = cfg.ext_thickness_mm * 1.0e-3
    cube_idx = np.flatnonzero(zc >= H_m - 1.0e-12)
    cube = build_subdomain(
        model,
        cube_idx,
        name="erom",
        physical_h=[attach.TOP_HTC],
    )
    _trim_zero_sources(cube)
    return extract_interface_enriched_rom(
        cube,
        tolerance=tolerance,
        probe_rounds=probe_rounds,
        interface_contour_degree=contour_degree,
        interface_labels=("z-",),
    )


def _run_with_fixed_rom(cfg: attach.AttachConfig, outdir: Path, reusable_rom):
    """Use the already-extracted cube ROM without re-training for this case."""
    previous = attach.extract_rom

    def fixed_extract(_subdomain, **_kwargs):
        return clone_embeddable_rom(reusable_rom)

    attach.extract_rom = fixed_extract
    try:
        return attach.run_attached(cfg, outdir)
    finally:
        attach.extract_rom = previous


def _write_order_sweep(
    path: Path,
    tolerances,
    contour_degree: int,
    probe_rounds: int,
):
    rows = []
    for tolerance in tolerances:
        rom = build_reusable_cube_rom(
            tolerance=float(tolerance),
            contour_degree=contour_degree,
            probe_rounds=probe_rounds,
        )
        rows.append(
            {
                "tolerance": float(tolerance),
                "order": int(rom.m),
                "pre_svd_order": int(rom.summary["pre_svd_order"]),
                "svd_kept_order": int(rom.summary["svd_kept_order"]),
                "physical_sources": int(rom.summary["physical_source_count"]),
                "interface_training_sources": int(
                    rom.summary["interface_training_source_count"]
                ),
            }
        )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument(
        "--cases", nargs="*", choices=tuple(CASES), default=list(DEFAULT_CASES)
    )
    parser.add_argument("--duration", type=float, default=1000.0)
    parser.add_argument("--dt", type=float, default=10.0)
    parser.add_argument("--tolerance", type=float, default=1.0e-3)
    parser.add_argument("--interface-contour-degree", type=int, default=0)
    parser.add_argument("--probe-rounds", type=int, default=3)
    parser.add_argument(
        "--sweep-tolerances",
        type=float,
        nargs="*",
        default=[1.0e-3],
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for case in args.cases:
        overrides = CASES[case]
        if "ext_thickness_mm" in overrides or "ext_vox_mm" in overrides:
            raise ValueError(
                f"{case}: external mesh changes invalidate the one-ROM reuse contract"
            )

    reusable_rom = build_reusable_cube_rom(
        tolerance=args.tolerance,
        contour_degree=args.interface_contour_degree,
        probe_rounds=args.probe_rounds,
    )
    (args.out / "extraction_summary.json").write_text(
        json.dumps(reusable_rom.summary, indent=2), encoding="utf-8"
    )
    sweep = _write_order_sweep(
        args.out / "order_sweep.csv",
        args.sweep_tolerances,
        args.interface_contour_degree,
        args.probe_rounds,
    )
    print("order sweep", json.dumps(sweep), flush=True)

    baseline = read_baseline()
    rows = []
    observed_orders = set()
    for case in args.cases:
        overrides = dict(CASES[case])
        overrides["duration_s"] = args.duration
        overrides["dt_s"] = args.dt
        report = _run_with_fixed_rom(
            attach.AttachConfig(**overrides), args.out / case, reusable_rom
        )
        metrics = report["metrics"]
        observed_orders.add(int(metrics["rom_order"]))
        ref = baseline[case]
        row = {
            "case": case,
            "flotherm_order": int(ref["flotherm_order"]),
            "metahotspot_order": int(metrics["rom_order"]),
            "metahotspot_training_sources": int(
                reusable_rom.summary["training_source_count"]
            ),
            "metahotspot_interface_contour_degree": int(
                args.interface_contour_degree
            ),
            "flotherm_steady_global_pct": float(ref["steady_global_pct"]),
            "metahotspot_steady_global_pct": 100.0
            * float(metrics["global_field_max_relative_error"]),
            "flotherm_transient_global_pct": float(ref["transient_global_pct"]),
            "metahotspot_transient_global_pct": 100.0
            * float(metrics["global_traj_max_relative_error"]),
            "metahotspot_junction_pct": float(metrics["junction_error_abs_pct"]),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    if observed_orders != {int(reusable_rom.m)}:
        raise RuntimeError(
            "simple_erom cases did not reuse exactly one extracted ROM: "
            f"expected {reusable_rom.m}, observed {sorted(observed_orders)}"
        )

    with (args.out / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "summary.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
