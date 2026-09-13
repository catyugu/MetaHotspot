#!/usr/bin/env python3
"""Run one EROM-attached external-body case and print the report.

    python run_erom_attach.py [tag] [k=v ...]

e.g.  python run_erom_attach.py baseline
      python run_erom_attach.py my_run ext_k=0.5 ext_footprint=0.5 ext_source_w=50

Each ``k=v`` overrides an :class:`AttachConfig` field (float); the external HTC
``ext_bottom_h`` uses the special value ``none`` to mean adiabatic.  Outputs are
written to ``out/<tag>/`` (VTU fields, probe trajectory, report.json).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from erom_attach_lib import OUT, AttachConfig, run_attached  # noqa: E402


def _parse_overrides(tokens: list[str]) -> dict:
    over = {}
    for tok in tokens:
        key, _, val = tok.partition("=")
        over[key] = None if val.lower() == "none" else float(val)
    return over


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    cfg = AttachConfig(**_parse_overrides(sys.argv[2:]))
    rep = run_attached(cfg, OUT / tag)
    m = rep["metrics"]
    print(
        f"[{tag}] EROM order={m['rom_order']}  interface_nodes={m['n_interface_nodes']} "
        f"cells={m['detailed_cells']}"
    )
    print(
        f"  heat flux: inject {m['injected_W']:.0f} W | top ref {m['top_flux_reference_W']:.2f} W "
        f"| coupled {m['top_flux_coupled_W']:.2f} W"
    )
    print(
        f"  interface flux: erom {m['interface_flux_erom_W']:+.2f} W | "
        f"ext {m['interface_flux_external_W']:+.2f} W | balance {m['interface_flux_balance_W']:+.2e} W"
    )
    print(
        f"  matrix symmetry {m['matrix_symmetry']:.2e}  min-eig {m['matrix_PD_min_eig']:.3e}"
    )
    print(
        f"  junction stable {m['junction_coupled_K']:.4f}/{m['junction_reference_K']:.4f} K  "
        f"err {m['junction_error_pct']:+.2f} %  (traj max {m['junction_traj_maxerr_K']:.3e} K)"
    )
    print(f"  interface trace max err {m['interface_trace_maxerr_K']:.3e} K")
    print(f"  EROM field max err      {m['erom_field_maxerr_K']:.3e} K")
    print(f"  external field max err  {m['external_field_maxerr_K']:.3e} K")
    print(f"  global field max err    {m['global_field_maxerr_K']:.3e} K")
    for k, v in rep["artifacts"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
