#!/usr/bin/env python3
"""
HotSpot example2 reproduction using MetaHotSpot.

Reproduces HotSpot's example2:

1. 4-layer package: chip (silicon) -> TIM -> heat spreader -> heat sink (aluminum)
2. 30 functional-unit blocks per the Alpha EV6 floorplan
3. Power traces from HotSpot gcc.ptrace, registered as periodic piecewise
   constant functions (period = sampling_intvl = 0.01 s, from example.config)
4. Unit conversion: HotSpot .ptrace gives total power (W) per block;
   MetaHotSpot heat_source expects volumetric power density (W/m3).
"""

from __future__ import annotations

import json
import sys
import time as _time
from pathlib import Path

import numpy as np

import metahotspot
from metahotspot import enums

# =========================================================================
#  HotSpot example2 configuration  (from example.config)
# =========================================================================

# Package dimensions
T_CHIP = 0.00015  # silicon die (m)
T_INTERFACE = 2.0e-05  # TIM (m)
T_SPREADER = 0.001  # heat spreader (m)
T_SINK = 0.0069  # heat sink (m)
TOTAL_Z = T_CHIP + T_INTERFACE + T_SPREADER + T_SINK

# Sink & spreader size (from example.config)
S_SINK = 0.06  # heat sink side (m)
S_SPREADER = 0.03  # heat spreader side (m)

# EV6 chip footprint (from ev6.flp - bounding box)
# The chip sits at the corner of the larger package.
CHIP_X_MIN, CHIP_X_MAX = 0.0, 0.016
CHIP_Y_MIN, CHIP_Y_MAX = 0.0, 0.016
A_SINK = S_SINK * S_SINK  # sink top area (m2)
A_SPREADER = S_SPREADER * S_SPREADER  # spreader area (m2)

# X / Y mesh: 64 fine cells in the chip footprint, 1 coarse cell per gap beyond.
#   chip   = [0, 0.016]      → NX_CHIP × NY_CHIP = 64 × 64
NX_CHIP, NY_CHIP = 64, 64
NX, NY = NX_CHIP + 2 + 2, NY_CHIP + 2 + 2

CHIP_L = 0.016

X_VERTS = np.concatenate(
    [
        np.array([-(S_SINK - CHIP_L) / 2]),
        np.array([-(S_SPREADER - CHIP_L) / 2]),
        np.linspace(0, CHIP_L, NX_CHIP + 1, endpoint=True),
        np.array([(S_SPREADER + CHIP_L) / 2]),
        np.array([(S_SINK + CHIP_L) / 2]),
    ]
)
Y_VERTS = np.concatenate(
    [
        np.array([-(S_SINK - CHIP_L) / 2]),
        np.array([-(S_SPREADER - CHIP_L) / 2]),
        np.linspace(0, CHIP_L, NY_CHIP + 1, endpoint=True),
        np.array([(S_SPREADER + CHIP_L) / 2]),
        np.array([(S_SINK + CHIP_L) / 2]),
    ]
)

# Z-layer resolution
NZ_CHIP = 1  # silicon die
NZ_INTERFACE = 1  # TIM
NZ_SPREADER = 2  # heat spreader
NZ_SINK = 4  # heat sink
NZ = NZ_CHIP + NZ_INTERFACE + NZ_SPREADER + NZ_SINK

# Time stepping
SAMPLING_INTVL = 0.01  # seconds  (example.config line 90)
NUM_SAMPLES = 100
DURATION = NUM_SAMPLES * SAMPLING_INTVL  # 1.0 s

# Ambient temperature
T_AMB = 318.15

# Convection: r_convec = 0.1 K/W over the heat-sink top surface.
# Distributed h = 1 / (r_convec * A_sink)
H_CONV = 1.0 / (0.1 * A_SINK)

# =========================================================================
#  EV6 floorplan  (from ev6.flp)
# =========================================================================
EV6_BLOCKS = [
    ("L2_left", 0.004900, 0.006200, 0.000000, 0.009800),
    ("L2", 0.016000, 0.009800, 0.000000, 0.000000),
    ("L2_right", 0.004900, 0.006200, 0.011100, 0.009800),
    ("Icache", 0.003100, 0.002600, 0.004900, 0.009800),
    ("Dcache", 0.003100, 0.002600, 0.008000, 0.009800),
    ("Bpred_0", 0.001033, 0.000700, 0.004900, 0.012400),
    ("Bpred_1", 0.001033, 0.000700, 0.005933, 0.012400),
    ("Bpred_2", 0.001033, 0.000700, 0.006967, 0.012400),
    ("DTB_0", 0.001033, 0.000700, 0.008000, 0.012400),
    ("DTB_1", 0.001033, 0.000700, 0.009033, 0.012400),
    ("DTB_2", 0.001033, 0.000700, 0.010067, 0.012400),
    ("FPAdd_0", 0.001100, 0.000900, 0.004900, 0.013100),
    ("FPAdd_1", 0.001100, 0.000900, 0.006000, 0.013100),
    ("FPReg_0", 0.000550, 0.000380, 0.004900, 0.014000),
    ("FPReg_1", 0.000550, 0.000380, 0.005450, 0.014000),
    ("FPReg_2", 0.000550, 0.000380, 0.006000, 0.014000),
    ("FPReg_3", 0.000550, 0.000380, 0.006550, 0.014000),
    ("FPMul_0", 0.001100, 0.000950, 0.004900, 0.014380),
    ("FPMul_1", 0.001100, 0.000950, 0.006000, 0.014380),
    ("FPMap_0", 0.001100, 0.000670, 0.004900, 0.015330),
    ("FPMap_1", 0.001100, 0.000670, 0.006000, 0.015330),
    ("IntMap", 0.000900, 0.001350, 0.007100, 0.014650),
    ("IntQ", 0.001300, 0.001350, 0.008000, 0.014650),
    ("IntReg_0", 0.000900, 0.000670, 0.009300, 0.015330),
    ("IntReg_1", 0.000900, 0.000670, 0.010200, 0.015330),
    ("IntExec", 0.001800, 0.002230, 0.009300, 0.013100),
    ("FPQ", 0.000900, 0.001550, 0.007100, 0.013100),
    ("LdStQ", 0.001300, 0.000950, 0.008000, 0.013700),
    ("ITB_0", 0.000650, 0.000600, 0.008000, 0.013100),
    ("ITB_1", 0.000650, 0.000600, 0.008650, 0.013100),
]
EV6_NAMES = [b[0] for b in EV6_BLOCKS]
EV6_DIM = {b[0]: (b[1], b[2], b[3], b[4]) for b in EV6_BLOCKS}


# =========================================================================
#  Helpers
# =========================================================================


def load_ptrace(path: str) -> tuple[np.ndarray, list[str]]:
    """Load HotSpot .ptrace -> (data[t,b], header)."""
    with open(path) as f:
        header = f.readline().strip().split("\t")
        data = []
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= len(header):
                data.append([float(v) for v in parts])
    return np.array(data), header


def build_model(
    heat_sources: dict[str, str],  # block-name -> heat-source expression string
    model: metahotspot.Model | None = None,
) -> metahotspot.Model:
    if model is None:
        model = metahotspot.Model()
    m = model

    m.set_settings(
        study=enums.Study.STEADY,
        length_unit=enums.LengthUnit.METER,
        initial_temperature_K=T_AMB,
    )

    # Mesh: full sink XY extent, 4 Z layers
    # Z-mesh: multiple cells per package layer for better heat-spreading accuracy
    z_chip = np.linspace(0, T_CHIP, NZ_CHIP + 1, endpoint=True)
    z_interface = np.linspace(
        T_CHIP, T_CHIP + T_INTERFACE, NZ_INTERFACE + 1, endpoint=True
    )[1:]
    z_spreader = np.linspace(
        T_CHIP + T_INTERFACE,
        T_CHIP + T_INTERFACE + T_SPREADER,
        NZ_SPREADER + 1,
        endpoint=True,
    )[1:]
    z_sink = np.linspace(
        T_CHIP + T_INTERFACE + T_SPREADER, TOTAL_Z, NZ_SINK + 1, endpoint=True
    )[1:]
    z_verts = np.concatenate([z_chip, z_interface, z_spreader, z_sink])

    m.set_mesh(x=X_VERTS, y=Y_VERTS, z=z_verts)

    # Materials
    #   silicon:   k=130,  rho*c = 1,630,300 J/(m3-K)
    #   TIM:       k=4,    rho*c = 4,000,000
    #   aluminum:  k=237,  rho*c = 2,422,000
    m.add_material("silicon", kx="130", ky="130", kz="130", rho="2330", c="700")
    m.add_material("tim", kx="4", ky="4", kz="4", rho="1200", c="3333.33")
    m.add_material("aluminum", kx="237", ky="237", kz="237", rho="2700", c="897")

    # Layers (first-added = top)
    lid_sink = m.add_layer(thickness=str(T_SINK))
    lid_spreader = m.add_layer(thickness=str(T_SPREADER))
    lid_tim = m.add_layer(thickness=str(T_INTERFACE))
    lid_chip = m.add_layer(thickness=str(T_CHIP))

    # -- Passive package layers --
    # Sink: full sink extent
    _bid = m.add_block(lid_sink, "aluminum")
    m.add_rect(
        _bid,
        x=str(-(S_SINK - CHIP_L) / 2),
        y=str(-(S_SINK - CHIP_L) / 2),
        width=str(S_SINK),
        height=str(S_SINK),
    )

    # Spreader: spreader extent
    _bid = m.add_block(lid_spreader, "aluminum")
    m.add_rect(
        _bid,
        x=str(-(S_SPREADER - CHIP_L) / 2),
        y=str(-(S_SPREADER - CHIP_L) / 2),
        width=str(S_SPREADER),
        height=str(S_SPREADER),
    )

    # TIM: chip extent [0, 0.016] x [0, 0.016]
    _bid = m.add_block(lid_tim, "tim")
    m.add_rect(
        _bid,
        x="0",
        y="0",
        width=str(CHIP_X_MAX - CHIP_X_MIN),
        height=str(CHIP_Y_MAX - CHIP_Y_MIN),
    )

    # -- Chip layer: one block per FU --
    for name in EV6_NAMES:
        w, h, lx, by = EV6_DIM[name]
        hs = heat_sources.get(name, "0")
        bid = m.add_block(
            lid_chip, "silicon", heat_source=hs, x_offset=str(lx), y_offset=str(by)
        )
        m.add_rect(bid, x="0", y="0", width=str(w), height=str(h))

    # -- Convection at sink top (z = TOTAL_Z), full sink XY --
    bc_id = m.add_boundary()
    m.set_convection(bc_id, coefficient=str(H_CONV), ambient_temperature=str(T_AMB))
    m.add_face_region(
        bc_id,
        axis=enums.Axis.Z,
        coordinate=TOTAL_Z,
        a_min=-(S_SINK - CHIP_L) / 2,
        a_max=(S_SINK + CHIP_L) / 2,
        b_min=-(S_SINK - CHIP_L) / 2,
        b_max=(S_SINK + CHIP_L) / 2,
    )

    return m


# =========================================================================
#  Main
# =========================================================================


def main():
    HOTSPOT_DIR = (
        Path(__file__).resolve().parent.parent / "cases" / "hotspot_reproduction_cases"
    )
    # --- Locate HotSpot data ----------------------------------------------
    ptrace_path = HOTSPOT_DIR / "gcc.ptrace"
    if not ptrace_path.exists():
        ptrace_path = Path("gcc.ptrace")
    if not ptrace_path.exists():
        print(f"\n  ERROR: gcc.ptrace not found at {HOTSPOT_DIR / 'gcc.ptrace'}")
        return 1
    print(f"\n  ptrace: {ptrace_path}")

    ptrace, header = load_ptrace(str(ptrace_path))
    n_tsteps, n_blocks = ptrace.shape
    print(f"  Power trace: {n_tsteps} steps x {n_blocks} blocks")
    print(f"  sampling_intvl = {SAMPLING_INTVL}s  -> duration = {DURATION}s")

    # --- Per-block stats -------------------------------------------------
    print("\n" + "-" * 72)
    print("Per-block power statistics")
    print("-" * 72)
    for name in EV6_NAMES:
        w, h, *_ = EV6_DIM[name]
        col = EV6_NAMES.index(name)
        pwr = ptrace[:, col]
        print(
            f"  {name:<12s}  area={w*h*1e6:>8.4f} mm2  "
            f"avg={pwr.mean():>7.3f} W  max={pwr.max():>7.3f} W"
        )

    steady_hs: dict[str, str] = {}
    for name in EV6_NAMES:
        w, h, *_ = EV6_DIM[name]
        vol = w * h * T_CHIP
        col = EV6_NAMES.index(name)
        q_vol = ptrace[:, col].mean() / vol
        steady_hs[name] = str(q_vol)

    t0 = _time.perf_counter()
    steady_model = build_model(steady_hs)
    c_steady = steady_model.compile()
    print(
        f"  Active cells: {c_steady.cell_count()},  "
        f"Grid cells: {c_steady.grid_count()}"
    )
    sol_steady = c_steady.solve()
    steady_state = sol_steady.states().copy()
    steady_temp = sol_steady.cell_temperatures().copy()
    t_steady = _time.perf_counter() - t0
    print(
        f"  Steady T in [{steady_temp.min():.4f}, "
        f"{steady_temp.max():.4f}] K  ({t_steady:.3f}s)"
    )
    sol_steady.close()
    c_steady.close()
    steady_model.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
