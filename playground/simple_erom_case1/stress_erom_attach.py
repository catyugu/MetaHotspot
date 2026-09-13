#!/usr/bin/env python3
"""Probe the FloTHERM-style EROM capability boundary over external attachments.

The external body is a second 100x100x100 mm cube directly below the ROM cube —
geometrically identical to the BCI-ROM domain (same size, centre 50x50x50 mm
source).  Only three families of external factor are varied per case:

* material constitution  (high-k vs low-k -> interface flux non-uniformity);
* bottom boundary condition  (adiabatic vs bottom HTC -> net interface heat flow);
* active internal source strength  (reverse heat injection into the interface).

Reduction order and per-observable error are reported separately — never
inferred from interface closure alone.  Full field / trajectory artifacts for
every case are under ``results/<tag>/``.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from erom_attach_lib import OUT, AttachConfig, run_attached  # noqa: E402

CASES = {
    "baseline_copper": dict(),
    "lowk_insulator": dict(ext_k=0.2, ext_rho=1200.0, ext_c=1500.0),
    "highk_aluminum": dict(ext_k=200.0, ext_rho=2700.0, ext_c=900.0),
    "bottom_htc_weak": dict(ext_bottom_h=100.0),
    "bottom_htc_strong": dict(ext_bottom_h=1000.0),
    "external_source_50w": dict(ext_source_w=50.0),
    "external_source_200w": dict(ext_source_w=200.0),
    "lowk_source": dict(
        ext_k=2.0,  # glass-fibre board (moderate insulator) carrying the source
        ext_rho=1200.0,
        ext_c=1500.0,
        ext_source_w=50.0,
        ext_bottom_h=100.0,  # a heat sink path so the exterior source is realistic
    ),
    "all_stress": dict(
        ext_k=3.0,
        ext_rho=1200.0,
        ext_c=1500.0,
        ext_bottom_h=300.0,
        ext_source_w=50.0,
    ),
    # --- z-layered external materials, strongly disparate conductivities ---
    # three bands top->bottom (z_hi decreasing) but listed bottom->top.  Each
    # case keeps the identical 100x100x100 outer cube + centre 50x50x50 source.
    "layered_soft": dict(
        ext_layers=(  # bottom Al high-k / mid alumina / top Si high-k
            (0.0, 30.0, 205.0, 2700.0, 900.0),
            (30.0, 60.0, 1.4, 3950.0, 880.0),
            (60.0, 100.0, 149.0, 2330.0, 712.0),
        )
    ),
    "layered_soft_source": dict(
        ext_layers=(  # bottom Al / mid alumina / top Si, source in mid low-k band
            (0.0, 30.0, 205.0, 2700.0, 900.0),
            (30.0, 60.0, 1.4, 3950.0, 880.0),
            (60.0, 100.0, 149.0, 2330.0, 712.0),
        ),
        ext_source_w=40.0,
        ext_bottom_h=50.0,  # heat-sink path for the interior source
    ),
    "layered_extreme": dict(
        # 3 dex spread: hard hi-k outer, near-vacuum centre insulator
        ext_layers=(
            (0.0, 25.0, 40.0, 7870.0, 490.0),  # steel
            (25.0, 75.0, 0.05, 120.0, 700.0),  # aerogel-like
            (75.0, 100.0, 385.0, 8930.0, 385.0),  # copper
        )
    ),
    "layered_extreme_source": dict(
        # aerogel surrounds the centre source but a bottom sink keeps the
        # steady-state rise physical (~tens of K), still stressing the EROM.
        ext_layers=(
            (0.0, 25.0, 385.0, 8930.0, 385.0),  # copper heat-spreader under source
            (25.0, 75.0, 0.2, 250.0, 900.0),  # low-k matrix around centre source
            (75.0, 100.0, 385.0, 8930.0, 385.0),  # copper cap at interface
        ),
        ext_source_w=30.0,
        ext_bottom_h=500.0,  # strong bottom sink -> realistic interior-source rise
    ),
}

SUMMARY_FIELDS = [
    "rom_order",
    "n_interface_nodes",
    "steady_max_rise_reference_K",
    "steady_max_rise_coupled_K",
    "junction_error_pct",
    "junction_error_abs_pct",
    "junction_error_K",
    "interface_trace_maxerr_K",
    "erom_field_maxerr_K",
    "external_field_maxerr_K",
    "global_field_maxerr_K",
    "global_field_max_relative_error",
    "junction_traj_maxerr_K",
    "junction_traj_max_relative_error",
    "global_traj_max_relative_error",
    "top_flux_reference_W",
    "top_flux_coupled_W",
    "interface_flux_erom_W",
    "interface_flux_external_W",
    "interface_flux_balance_W",
]


def main() -> None:
    rows = []
    for tag, overrides in CASES.items():
        overrides = dict(overrides)
        overrides.setdefault("duration_s", 1000.0)
        overrides.setdefault("dt_s", 10.0)
        cfg = AttachConfig(**overrides)
        try:
            m = run_attached(cfg, OUT / tag)["metrics"]
            row = {"tag": tag, **{f: m.get(f) for f in SUMMARY_FIELDS}}
            rows.append(row)
            print(
                f"[{tag:24s}] order={m['rom_order']:3d}  "
                f"dTmax={m['steady_max_rise_reference_K']:.3f} K  "
                f"junction_rel={m['junction_error_abs_pct']:.3f}%  "
                f"global_rel={100.0 * m['global_field_max_relative_error']:.3f}%  "
                f"transient_junction_rel={100.0 * m['junction_traj_max_relative_error']:.3f}%  "
                f"transient_global_rel={100.0 * m['global_traj_max_relative_error']:.3f}%",
                flush=True,
            )
        except Exception as exc:  # pragma: no cover - a failed case is a result
            print(f"[{tag:24s}] FAILED: {type(exc).__name__}: {exc}", flush=True)
            rows.append({"tag": tag, "error": f"{type(exc).__name__}: {exc}"})

    cols = ["tag"] + SUMMARY_FIELDS
    csv_path = OUT / "stress_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    (OUT / "stress_summary.json").write_text(
        json.dumps(rows, indent=2, default=float), encoding="utf-8"
    )
    print("\nsummary table ->", csv_path)


if __name__ == "__main__":
    main()
