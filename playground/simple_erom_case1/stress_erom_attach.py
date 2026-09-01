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
    "lowk_source_100w": dict(
        ext_k=0.5, ext_rho=1200.0, ext_c=1500.0, ext_source_w=100.0
    ),
    "all_stress": dict(
        ext_k=0.5, ext_rho=1200.0, ext_c=1500.0, ext_bottom_h=300.0, ext_source_w=200.0
    ),
}

SUMMARY_FIELDS = [
    "rom_order",
    "n_interface_nodes",
    "junction_error_pct",
    "junction_error_K",
    "interface_trace_maxerr_K",
    "erom_field_maxerr_K",
    "external_field_maxerr_K",
    "global_field_maxerr_K",
    "junction_traj_maxerr_K",
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
        overrides.setdefault("duration_s", 160.0)  # bounded transient window
        overrides.setdefault("dt_s", 4.0)
        cfg = AttachConfig(**overrides)
        try:
            m = run_attached(cfg, OUT / tag)["metrics"]
            row = {"tag": tag, **{f: m.get(f) for f in SUMMARY_FIELDS}}
            rows.append(row)
            print(
                f"[{tag:24s}] order={m['rom_order']:3d}  "
                f"junction_err={m['junction_error_pct']:+.2f}%  "
                f"gbl={m['global_field_maxerr_K']:.4f} K  "
                f"ext={m['external_field_maxerr_K']:.4f} K  "
                f"iface_trace={m['interface_trace_maxerr_K']:.4f} K",
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
