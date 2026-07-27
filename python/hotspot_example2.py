#!/usr/bin/env python3
"""
HotSpot example2 full reproduction (steady + transient) using MetaHotSpot.

Reproduces the HotSpot grid-model transient from examples/example2:

  1. 4-layer package: chip (silicon) -> TIM -> heat spreader -> heat sink (Al)
  2. 30 EV6 floorplan blocks
  3. Power traces from gcc.ptrace, registered as periodic piecewise constant
     per-block functions (period = sampling_intvl = 0.01 s)
  4. Unit conversion: HotSpot .ptrace gives block-total power (W); MetaHotSpot
     heat_source expects volumetric power density (W/m3).
  5. Steady-state solve -> chain initial state -> transient solve (100 x sampling_intvl).
"""

from __future__ import annotations

import sys
import time as _time
from pathlib import Path

import numpy as np

import metahotspot
from metahotspot import enums


# ====================================================================
#  HotSpot example2 configuration  (from example.config)
# ====================================================================

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
CHIP_X_MIN, CHIP_X_MAX = 0.0, 0.016
CHIP_Y_MIN, CHIP_Y_MAX = 0.0, 0.016
CHIP_L = 0.016
A_SINK = S_SINK * S_SINK  # sink top area (m2)
A_SPREADER = S_SPREADER * S_SPREADER  # spreader area (m2)

# -- X / Y mesh --
# Core region (chip footprint): 64 x 64 uniform cells.
# Spreader extends 7 mm beyond chip, sink extends 22 mm beyond chip.
# We add exactly one cell per gap (spreader left/right, sink left/right,
# bottom/top) for the extra regions - minimal extra mesh.
NX_CHIP, NY_CHIP = 64, 64

# Mesh vertices: [sink_min, spreader_min, chip_linspace(0, L, NX_CHIP+1),
#                 spreader_max, sink_max]
X_VERTS = np.concatenate(
    [
        np.array([-(S_SINK - CHIP_L) / 2.0]),  # sink left
        np.array([-(S_SPREADER - CHIP_L) / 2.0]),  # spreader left
        np.linspace(0, CHIP_L, NX_CHIP + 1, endpoint=True),  # chip (64 cells)
        np.array([(S_SPREADER + CHIP_L) / 2.0]),  # spreader right
        np.array([(S_SINK + CHIP_L) / 2.0]),  # sink right
    ]
)
Y_VERTS = np.concatenate(
    [
        np.array([-(S_SINK - CHIP_L) / 2.0]),
        np.array([-(S_SPREADER - CHIP_L) / 2.0]),
        np.linspace(0, CHIP_L, NY_CHIP + 1, endpoint=True),
        np.array([(S_SPREADER + CHIP_L) / 2.0]),
        np.array([(S_SINK + CHIP_L) / 2.0]),
    ]
)

# Z-layer resolution
NZ_CHIP = 1  # silicon die
NZ_INTERFACE = 1  # TIM
NZ_SPREADER = 2  # heat spreader
NZ_SINK = 4  # heat sink
NZ = NZ_CHIP + NZ_INTERFACE + NZ_SPREADER + NZ_SINK

# -- Time stepping --
SAMPLING_INTVL = 0.01  # seconds (example.config line 90)
NUM_SAMPLES = 100
DURATION = NUM_SAMPLES * SAMPLING_INTVL  # 1.0 s

OUTPUT_INTERVAL = SAMPLING_INTVL  # one output per sample

# Ambient / initial temperature
T_AMB = 318.15

# Convection: r_convec = 0.1 K/W over the heat-sink top surface.
# Distributed h = 1 / (r_convec * A_sink)
H_CONV = 1.0 / (0.1 * A_SINK)


# ====================================================================
#  EV6 floorplan  (from ev6.flp)
# ====================================================================

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
# (width, height, left-x, bottom-y)
EV6_DIM = {b[0]: (b[1], b[2], b[3], b[4]) for b in EV6_BLOCKS}
# Centroid (for probe placement)
EV6_CENTER = {b[0]: (b[3] + b[1] / 2, b[4] + b[2] / 2) for b in EV6_BLOCKS}


# ====================================================================
#  Helpers
# ====================================================================


def load_ptrace(path: str) -> tuple[np.ndarray, list[str]]:
    """Load HotSpot .ptrace -> (data[t, b], header)."""
    with open(path) as f:
        header = f.readline().strip().split("\t")
        data = []
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= len(header):
                data.append([float(v) for v in parts])
    return np.array(data), header


def build_model(
    heat_sources: dict[str, str] | None = None,
    model: metahotspot.Model | None = None,
) -> metahotspot.Model:
    """Build the MetaHotSpot model for example2."""
    if model is None:
        model = metahotspot.Model()
    m = model

    m.set_settings(
        study=enums.Study.STEADY,
        length_unit=enums.LengthUnit.METER,
        initial_temperature_K=T_AMB,
    )

    # -- Mesh: full sink XY extent, 4 Z layers --
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

    # -- Materials --
    m.add_material("silicon", kx="130", ky="130", kz="130", rho="2330", c="700")
    m.add_material("tim", kx="4", ky="4", kz="4", rho="1200", c="3333.33")
    m.add_material("aluminum", kx="237", ky="237", kz="237", rho="2700", c="897")

    # -- Layers (first-added = top) --
    lid_sink = m.add_layer(thickness=str(T_SINK))
    lid_spreader = m.add_layer(thickness=str(T_SPREADER))
    lid_tim = m.add_layer(thickness=str(T_INTERFACE))
    lid_chip = m.add_layer(thickness=str(T_CHIP))

    # Sink: full sink XY extent
    bid = m.add_block(lid_sink, "aluminum")
    m.add_rect(
        bid,
        x=str(-(S_SINK - CHIP_L) / 2),
        y=str(-(S_SINK - CHIP_L) / 2),
        width=str(S_SINK),
        height=str(S_SINK),
    )

    # Spreader: spreader extent
    bid = m.add_block(lid_spreader, "aluminum")
    m.add_rect(
        bid,
        x=str(-(S_SPREADER - CHIP_L) / 2),
        y=str(-(S_SPREADER - CHIP_L) / 2),
        width=str(S_SPREADER),
        height=str(S_SPREADER),
    )

    # TIM: chip footprint [0, 0.016] x [0, 0.016]
    bid = m.add_block(lid_tim, "tim")
    m.add_rect(
        bid,
        x="0",
        y="0",
        width=str(CHIP_X_MAX - CHIP_X_MIN),
        height=str(CHIP_Y_MAX - CHIP_Y_MIN),
    )

    # Chip: one block per functional unit
    for name in EV6_NAMES:
        w, h, lx, by = EV6_DIM[name]
        hs = heat_sources.get(name, "0") if heat_sources else "0"
        bid = m.add_block(
            lid_chip, "silicon", heat_source=hs, x_offset=str(lx), y_offset=str(by)
        )
        m.add_rect(bid, x="0", y="0", width=str(w), height=str(h))

    # -- Convection at sink top (z = TOTAL_Z) --
    _regions = [
        (
            enums.Axis.Z,
            TOTAL_Z,
            -(S_SINK - CHIP_L) / 2,
            (S_SINK + CHIP_L) / 2,
            -(S_SINK - CHIP_L) / 2,
            (S_SINK + CHIP_L) / 2,
        ),
    ]
    m.add_convection(
        coefficient=str(H_CONV), ambient_temperature=str(T_AMB), regions=_regions
    )

    # -- Probes at each block centroid (in chip layer z=0+) --
    probe_z = T_CHIP / 2.0  # mid-chip
    for name in EV6_NAMES:
        cx, cy = EV6_CENTER[name]
        m.add_probe(f"chip_{name}", cx, cy, probe_z)
    m.add_probe("tim_center", CHIP_L / 2, CHIP_L / 2, T_CHIP + T_INTERFACE / 2)

    return m


# ====================================================================
#  Main
# ====================================================================


def main():
    HOTSPOT_DIR = (
        Path(__file__).resolve().parent.parent / "cases" / "hotspot_reproduction_cases"
    )
    ptrace_path = HOTSPOT_DIR / "gcc.ptrace"
    if not ptrace_path.exists():
        ptrace_path = Path("gcc.ptrace")
    if not ptrace_path.exists():
        print(f"\n  ERROR: gcc.ptrace not found at " f"{HOTSPOT_DIR / 'gcc.ptrace'}")
        return 1
    print(f"\n  ptrace: {ptrace_path}")

    ptrace, header = load_ptrace(str(ptrace_path))
    n_tsteps, n_blocks = ptrace.shape
    print(f"  Power trace: {n_tsteps} steps x {n_blocks} blocks")
    print(f"  sampling_intvl = {SAMPLING_INTVL}s  -> duration = {DURATION}s")
    assert n_blocks == len(
        EV6_NAMES
    ), f"Expected {len(EV6_NAMES)} blocks, got {n_blocks}"

    # -- Per-block stats --
    block_volume = {}
    print("\n" + "-" * 72)
    print("Per-block power statistics " "(HotSpot ptrace -> volumetric density)")
    print("-" * 72)
    for name in EV6_NAMES:
        w, h, *_ = EV6_DIM[name]
        vol = w * h * T_CHIP  # block volume (m3)
        block_volume[name] = vol
        col = EV6_NAMES.index(name)
        pwr = ptrace[:, col]  # total power (W) per sample
        q_vol = pwr / vol  # -> W/m3 per sample
        print(
            f"  {name:<12s}  area={w*h*1e6:>8.4f} mm2  "
            f"P_avg={pwr.mean():>7.3f} W  "
            f"q_avg={q_vol.mean():>7.3e} W/m3"
        )

    # ================================================================
    #  1. STEADY solve (average-power per block)
    # ================================================================
    print("\n" + "=" * 72)
    print("Phase 1: Steady-state solve (block-average power)")
    print("=" * 72)

    steady_hs = {}
    for name in EV6_NAMES:
        col = EV6_NAMES.index(name)
        q_vol = ptrace[:, col].mean() / block_volume[name]
        steady_hs[name] = str(q_vol)

    t0 = _time.perf_counter()
    steady_model = build_model(steady_hs)
    c_steady = steady_model.compile()
    print(
        f"  Active cells: {c_steady.metadata().cell_count},  "
        f"Grid cells: {c_steady.metadata().grid_count}"
    )
    sol_steady = c_steady.solve()
    steady_state = sol_steady.view().states.copy()
    steady_temp = sol_steady.view().cell_temperatures.copy()
    t_steady = _time.perf_counter() - t0
    print(
        f"  Steady T in [{steady_temp.min():.4f}, "
        f"{steady_temp.max():.4f}] K  ({t_steady:.3f}s)"
    )

    # -- Export steady-state VTU --
    vtu_path = HOTSPOT_DIR / "example2_steady.vtu"
    sol_steady.write_vtu(str(vtu_path))
    print(f"  VTU exported: {vtu_path}")

    sol_steady.close()
    c_steady.close()
    steady_model.close()

    # ================================================================
    #  2. TRANSIENT solve (periodic piecewise-constant per-block)
    # ================================================================
    print("\n" + "=" * 72)
    print("Phase 2: Transient solve")
    print("=" * 72)

    t0 = _time.perf_counter()
    transient_model = metahotspot.Model()

    # Register periodic piecewise-constant functions first.
    # HotSpot ptrace: total power (W). Convert to volumetric density (W/m3).
    for name in EV6_NAMES:
        col = EV6_NAMES.index(name)
        q_vol_samples = ptrace[:, col] / block_volume[name]
        transient_model.add_function_periodic_piecewise_constant(
            name=f"power_{name}",
            values=q_vol_samples,
            period=SAMPLING_INTVL,
        )

    # Build model with function references
    transient_heat_sources = {name: f"power_{name}(t)" for name in EV6_NAMES}
    transient_model = build_model(transient_heat_sources, model=transient_model)

    # Override to TRANSIENT study
    transient_model.set_settings(
        study=enums.Study.TRANSIENT,
        length_unit=enums.LengthUnit.METER,
        initial_temperature_K=T_AMB,
        duration=DURATION,
        output_interval=OUTPUT_INTERVAL,
    )

    # Chain steady-state as initial condition — pass directly to solve()
    c_transient = transient_model.compile()
    sol_transient = c_transient.solve(state=steady_state)
    t_transient = _time.perf_counter() - t0

    final_temp = sol_transient.view().cell_temperatures.copy()
    print(
        f"  Transient final T in [{final_temp.min():.4f}, "
        f"{final_temp.max():.4f}] K  ({t_transient:.3f}s)"
    )

    # -- Transient probe traces --
    n_probes = sol_transient.probe_count()
    print(f"\n  Probes recorded: {n_probes}")
    for pi in range(min(n_probes, 5)):
        pv = sol_transient.probe_view(pi)
        if pv.times is not None and len(pv.times) > 0:
            print(
                f"    {pv.name:<20s}  t={pv.times[0]:.4f}..{pv.times[-1]:.4f}  "
                f"T={pv.values[0]:.2f}..{pv.values[-1]:.2f} K  "
                f"({pv.record_count} records)"
            )

    # Compare with HotSpot reference (gcc.init)
    ref_path = HOTSPOT_DIR / "gcc.init"
    if ref_path.exists():
        ref_temps = {}
        with open(ref_path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    ref_temps[parts[0]] = float(parts[1])
        chip_probes = {
            k: v
            for k, v in ref_temps.items()
            if not any(k.startswith(p) for p in ["iface_", "hsp_", "hsink_", "inode_"])
        }
        print(
            f"\n  Reference steady block temperatures "
            f"(HotSpot gcc.init, {len(chip_probes)} blocks):"
        )
        hottest_ref = max(chip_probes.values())
        print(f"    T_max: {hottest_ref:.2f} K (IntReg)")
        print(f"    Our steady T_max: {steady_temp.max():.2f} K")

    sol_transient.close()
    c_transient.close()
    transient_model.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
