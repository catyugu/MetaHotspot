#!/usr/bin/env python3
"""
Spreader+Sink macro-model extraction for HotSpot example2.

Simplified using the new API: c.mesh() queries geometry, c.step() replaces
the full-transient manual backward-Euler loop.

Training: steady solves under different per-block power patterns → port
temperature snapshots → SVD → reduced basis U_r.

Experiments:
  1. Steady ROM vs full solve  (held-out power patterns from ptrace)
  2. Transient ROM vs full BE   (uniform T0=318.15K, 1 s, backward Euler,
     uses c.step() for full reference)
  3. BCI cross-h variation      (basis trained at nominal h, test at
     different convection coefficients)
"""

from __future__ import annotations

import sys
import time as _time
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import splu

import metahotspot
from metahotspot import enums

# ====================================================================
#  Configuration  (same geometry as hotspot_example2.py)
# ====================================================================

T_CHIP = 0.00015
T_INTERFACE = 2.0e-05
T_SPREADER = 0.001
T_SINK = 0.0069
TOTAL_Z = T_CHIP + T_INTERFACE + T_SPREADER + T_SINK

S_SINK = 0.06
S_SPREADER = 0.03
CHIP_L = 0.016
A_SINK = S_SINK * S_SINK

NX_CHIP, NY_CHIP = 64, 64
NZ_CHIP = 1
NZ_INTERFACE = 1
NZ_SPREADER = 2
NZ_SINK = 4

SAMPLING_INTVL = 0.01
TRANSIENT_DURATION = 1.0
N_TRANSIENT_STEPS = int(TRANSIENT_DURATION / SAMPLING_INTVL)

T_AMB = 318.15
H_CONV_NOMINAL = 1.0 / (0.1 * A_SINK)

MACRO_LAYERS = (0, 1)
MACRO_BLOCK = 0
ENERGY_THR = 0.9999

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

# ====================================================================
#  Helpers
# ====================================================================


def load_ptrace(path: str) -> tuple[np.ndarray, list[str]]:
    with open(path) as f:
        header = f.readline().strip().split("\t")
        data = [
            [float(v) for v in line.strip().split("\t") if v]
            for line in f
            if line.strip()
        ]
    return np.array(data), header


def build_example2_model(
    heat_sources: dict[str, str] | None = None,
    model: metahotspot.Model | None = None,
    h_conv: float | None = None,
) -> metahotspot.Model:
    """Build the MetaHotSpot example2 model."""
    _h = h_conv if h_conv is not None else H_CONV_NOMINAL
    if model is None:
        model = metahotspot.Model()
    m = model

    m.set_settings(
        study=enums.Study.STEADY,
        length_unit=enums.LengthUnit.METER,
        initial_temperature_K=T_AMB,
    )

    # Mesh vertices
    x = np.concatenate(
        [
            np.array([-(S_SINK - CHIP_L) / 2.0]),
            np.array([-(S_SPREADER - CHIP_L) / 2.0]),
            np.linspace(0, CHIP_L, NX_CHIP + 1, endpoint=True),
            np.array([(S_SPREADER + CHIP_L) / 2.0]),
            np.array([(S_SINK + CHIP_L) / 2.0]),
        ]
    )
    y = np.concatenate(
        [
            np.array([-(S_SINK - CHIP_L) / 2.0]),
            np.array([-(S_SPREADER - CHIP_L) / 2.0]),
            np.linspace(0, CHIP_L, NY_CHIP + 1, endpoint=True),
            np.array([(S_SPREADER + CHIP_L) / 2.0]),
            np.array([(S_SINK + CHIP_L) / 2.0]),
        ]
    )
    z_chip = np.linspace(0, T_CHIP, NZ_CHIP + 1, endpoint=True)
    z_iface = np.linspace(
        T_CHIP, T_CHIP + T_INTERFACE, NZ_INTERFACE + 1, endpoint=True
    )[1:]
    z_spr = np.linspace(
        T_CHIP + T_INTERFACE,
        T_CHIP + T_INTERFACE + T_SPREADER,
        NZ_SPREADER + 1,
        endpoint=True,
    )[1:]
    z_sink = np.linspace(
        T_CHIP + T_INTERFACE + T_SPREADER, TOTAL_Z, NZ_SINK + 1, endpoint=True
    )[1:]
    z_verts = np.concatenate([z_chip, z_iface, z_spr, z_sink])
    m.set_mesh(x=x, y=y, z=z_verts)

    # Materials
    m.add_material("silicon", kx="130", ky="130", kz="130", rho="2330", c="700")
    m.add_material("tim", kx="4", ky="4", kz="4", rho="1200", c="3333.33")
    m.add_material("aluminum", kx="237", ky="237", kz="237", rho="2700", c="897")

    # Layers (first-added = top → 0=sink, 1=spreader, 2=TIM, 3=chip)
    lid_sink = m.add_layer(thickness=str(T_SINK))
    lid_spreader = m.add_layer(thickness=str(T_SPREADER))
    lid_tim = m.add_layer(thickness=str(T_INTERFACE))
    lid_chip = m.add_layer(thickness=str(T_CHIP))

    bid = m.add_block(lid_sink, "aluminum")
    m.add_rect(
        bid,
        x=str(-(S_SINK - CHIP_L) / 2),
        y=str(-(S_SINK - CHIP_L) / 2),
        width=str(S_SINK),
        height=str(S_SINK),
    )

    bid = m.add_block(lid_spreader, "aluminum")
    m.add_rect(
        bid,
        x=str(-(S_SPREADER - CHIP_L) / 2),
        y=str(-(S_SPREADER - CHIP_L) / 2),
        width=str(S_SPREADER),
        height=str(S_SPREADER),
    )

    bid = m.add_block(lid_tim, "tim")
    m.add_rect(bid, x="0", y="0", width=str(CHIP_L), height=str(CHIP_L))

    for name in EV6_NAMES:
        w, h, lx, by = EV6_DIM[name]
        hs = heat_sources.get(name, "0") if heat_sources else "0"
        bid = m.add_block(
            lid_chip, "silicon", heat_source=hs, x_offset=str(lx), y_offset=str(by)
        )
        m.add_rect(bid, x="0", y="0", width=str(w), height=str(h))

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
        coefficient=str(_h), ambient_temperature=str(T_AMB), regions=_regions
    )
    return m


def partition(
    K: csc_matrix,
    nc: int,
    layer_ids: np.ndarray,
    block_ids: np.ndarray,
    macro_layers: tuple[int, ...],
    macro_block: int,
):
    """Split mesh → external (e), port (p), internal (i)."""
    is_macro = np.zeros(nc, dtype=bool)
    for ml in macro_layers:
        is_macro |= (layer_ids == ml) & (block_ids == macro_block)
    macro_set = set(np.where(is_macro)[0])

    port_set = set()
    for i in macro_set:
        for j in K.indices[K.indptr[i] : K.indptr[i + 1]]:
            if j not in macro_set:
                port_set.add(i)
                break

    is_port = np.zeros(nc, dtype=bool)
    for p in port_set:
        is_port[p] = True

    e_idx = np.sort(np.where(~is_macro)[0])
    p_idx = np.sort(np.where(is_port)[0])
    i_idx = np.sort(np.where(is_macro & ~is_port)[0])
    return e_idx, p_idx, i_idx


# ====================================================================
#  BCI-ROM core routines
# ====================================================================


def bci_schur_complement(A: csc_matrix, e_idx, p_idx, i_idx, U_r: np.ndarray) -> tuple:
    """Compute BCI-ROM Schur complement blocks for system matrix A.

    Returns (A_ii_lu, A_epU, A_rom_pp, A_ip_U, A_pi_csr).
    """
    A_ep = A[e_idx, :][:, p_idx].tocsc()
    A_pp = A[p_idx, :][:, p_idx]
    A_pi = A[p_idx, :][:, i_idx].tocsr()
    A_ip = A[i_idx, :][:, p_idx].tocsc()
    A_ii = A[i_idx, :][:, i_idx].tocsc()
    A_ii_lu = splu(A_ii)

    A_pp_proj = U_r.T @ (A_pp @ U_r)
    A_ip_U = A_ip @ U_r
    T = np.column_stack([A_ii_lu.solve(A_ip_U[:, k]) for k in range(U_r.shape[1])])
    correction = U_r.T @ (A_pi @ T)
    A_rom_pp = A_pp_proj - correction
    A_epU = A_ep @ U_r
    return A_ii_lu, A_epU, A_rom_pp, A_ip_U, A_pi


def build_reduced_matrix(A_ee, A_epU, A_rom_pp, ne, r):
    """Build the sparse (ne+r)×(ne+r) reduced system matrix."""
    ne_r = ne + r
    rows, cols, vals = [], [], []

    Ace = A_ee.tocoo()
    rows.extend(Ace.row.tolist())
    cols.extend(Ace.col.tolist())
    vals.extend(Ace.data.tolist())

    for j in range(r):
        for i in range(ne):
            v = A_epU[i, j]
            if v != 0.0:
                rows.append(i)
                cols.append(ne + j)
                vals.append(v)
                rows.append(ne + j)
                cols.append(i)
                vals.append(v)

    for j in range(r):
        for i in range(r):
            v = A_rom_pp[i, j]
            if v != 0.0:
                rows.append(ne + i)
                cols.append(ne + j)
                vals.append(v)

    return coo_matrix((vals, (rows, cols)), shape=(ne_r, ne_r)).tocsc()


def bci_steady_solve(
    K: csc_matrix,
    f: np.ndarray,
    e_idx,
    p_idx,
    i_idx,
    U_r: np.ndarray,
    blocks: tuple | None = None,
) -> np.ndarray:
    """BCI-ROM steady-state solve.  Returns full temperature field (nc,)."""
    ne, ni = len(e_idx), len(i_idx)
    r = U_r.shape[1]

    if blocks is not None:
        K_ii_lu, K_epU, K_rom_pp, K_ip_U, K_pi = blocks
    else:
        K_ii_lu, K_epU, K_rom_pp, K_ip_U, K_pi = bci_schur_complement(
            K, e_idx, p_idx, i_idx, U_r
        )

    f_i = f[i_idx]
    f_i_schur = K_ii_lu.solve(f_i)
    f_rom_p = U_r.T @ (f[p_idx] - K_pi.dot(f_i_schur))
    f_rom = np.concatenate([f[e_idx], f_rom_p])

    K_ee = K[e_idx, :][:, e_idx]
    A_rom = build_reduced_matrix(K_ee, K_epU, K_rom_pp, ne, r)
    A_rom_lu = splu(A_rom)
    sol = A_rom_lu.solve(f_rom)
    u_e = sol[:ne]
    alpha = sol[ne:]
    u_p = U_r @ alpha
    K_ip_mat = K[i_idx, :][:, p_idx].tocsc()
    u_i = K_ii_lu.solve(f_i - (K_ip_mat @ u_p).ravel())

    T = np.zeros(K.shape[0], dtype=np.float64)
    T[e_idx] = u_e
    T[p_idx] = u_p
    T[i_idx] = u_i
    return T


def bci_transient_setup(
    K: csc_matrix, C: csc_matrix, e_idx, p_idx, i_idx, U_r: np.ndarray, dt: float
):
    """Pre-compute factorizations for BCI-ROM backward-Euler transient.

    Builds A = C + dt*K then projects via U_r.
    """
    ne, r = len(e_idx), U_r.shape[1]
    A = C + dt * K

    A_ee = A[e_idx, :][:, e_idx]
    A_ep = A[e_idx, :][:, p_idx].tocsc()
    A_pp = A[p_idx, :][:, p_idx]
    A_pi = A[p_idx, :][:, i_idx].tocsr()
    A_ip = A[i_idx, :][:, p_idx].tocsc()
    A_ii = A[i_idx, :][:, i_idx].tocsc()
    A_ii_lu = splu(A_ii)

    A_pp_proj = U_r.T @ (A_pp @ U_r)
    A_ip_U = A_ip @ U_r
    T = np.column_stack([A_ii_lu.solve(A_ip_U[:, k]) for k in range(r)])
    A_rom_pp = A_pp_proj - U_r.T @ (A_pi @ T)
    A_epU = A_ep @ U_r
    A_rom = build_reduced_matrix(A_ee, A_epU, A_rom_pp, ne, r)
    A_rom_lu = splu(A_rom)

    return dict(
        A_ii_lu=A_ii_lu,
        A_ip_U=A_ip_U,
        A_rom_lu=A_rom_lu,
        A_pi_csr=A_pi,
        C=C,
        dt=dt,
        ne=ne,
        r=r,
    )


def bci_transient_step(
    T_curr: np.ndarray, f_np1: np.ndarray, ctx: dict, e_idx, p_idx, i_idx, U_r
) -> np.ndarray:
    """One backward-Euler step via BCI-ROM."""
    ne, r = ctx["ne"], ctx["r"]
    dt = ctx["dt"]
    C = ctx["C"]
    A_ii_lu = ctx["A_ii_lu"]
    A_ip_U = ctx["A_ip_U"]
    A_rom_lu = ctx["A_rom_lu"]
    A_pi_csr = ctx["A_pi_csr"]

    b = C @ T_curr + dt * f_np1
    b_i = b[i_idx]
    b_i_schur = A_ii_lu.solve(b_i)
    b_rom_p = U_r.T @ (b[p_idx] - A_pi_csr.dot(b_i_schur))
    b_rom = np.concatenate([b[e_idx], b_rom_p])
    sol = A_rom_lu.solve(b_rom)
    u_e = sol[:ne]
    alpha = sol[ne:]
    u_p = U_r @ alpha
    u_i = A_ii_lu.solve(b_i - (A_ip_U @ alpha).ravel())

    T_new = np.empty(C.shape[0], dtype=np.float64)
    T_new[e_idx] = u_e
    T_new[p_idx] = u_p
    T_new[i_idx] = u_i
    return T_new


# ====================================================================
#  Main
# ====================================================================


def main():
    _ROOT = Path(__file__).resolve().parent.parent
    ptrace_path = _ROOT / "cases" / "hotspot_reproduction_cases" / "gcc.ptrace"
    if not ptrace_path.exists():
        ptrace_path = Path("gcc.ptrace")
    if not ptrace_path.exists():
        print("\n  ERROR: gcc.ptrace not found")
        return 1

    print(f"\n  ptrace: {ptrace_path}")
    ptrace, header = load_ptrace(str(ptrace_path))
    n_tsteps, n_blocks = ptrace.shape
    print(f"  Power trace: {n_tsteps} steps × {n_blocks} blocks")
    print(f"  Macro region: layers {MACRO_LAYERS} (sink + spreader)")

    block_volume = {name: w * h * T_CHIP for name, w, h, *_ in EV6_BLOCKS}

    # ================================================================
    #  Phase 0: Reference model → compile → partition
    # ================================================================
    print("\n" + "=" * 72)
    print("Phase 0: Reference model, compile, partition")
    print("=" * 72)

    ref_model = build_example2_model(h_conv=H_CONV_NOMINAL)
    ref_c = ref_model.compile()
    meta = ref_c.metadata()
    nc = meta.cell_count

    # Query mesh from compiled model.
    nx, ny, nz, xv, yv, zv = (
        meta.nx,
        meta.ny,
        meta.nz,
        meta.x_verts,
        meta.y_verts,
        meta.z_verts,
    )
    print(f"  Mesh: {nx}×{ny}×{nz} cells  ({len(xv)}×{len(yv)}×{len(zv)} vertices)")

    K_ref = ref_c.assemble().stiffness_matrix()
    layer_ids = meta.layer_ids.copy()
    block_ids = meta.block_ids.copy()

    e_idx, p_idx, i_idx = partition(
        K_ref, nc, layer_ids, block_ids, MACRO_LAYERS, MACRO_BLOCK
    )
    ne, np_, ni = len(e_idx), len(p_idx), len(i_idx)
    print(f"  Active cells: {nc}")
    print(f"  External (TIM + chip):    {ne:>6}")
    print(f"  Port (spread-sink iface): {np_:>6}")
    print(f"  Internal (spread+sink):   {ni:>6}")

    ref_c.close()
    ref_model.close()

    # ================================================================
    #  Phase 1: Training — port snapshots at random power patterns
    # ================================================================
    print("\n" + "-" * 72)
    rng = np.random.default_rng(42)
    perm = rng.permutation(n_tsteps)
    n_train = min(50, n_tsteps // 2)
    n_test = min(50, n_tsteps - n_train)
    train_idx = perm[:n_train]
    test_idx = perm[n_train : n_train + n_test]
    print(f"Phase 1: Training  ({n_train} port snapshots)")
    print("-" * 72)

    snapshots = []
    for k, ti in enumerate(train_idx):
        t0 = _time.perf_counter()
        hs = {
            name: str(ptrace[ti, EV6_NAMES.index(name)] / block_volume[name])
            for name in EV6_NAMES
        }
        m = build_example2_model(hs, h_conv=H_CONV_NOMINAL)
        sol = m.compile().solve()
        T = sol.cell_temperatures().copy()
        sol.close()
        snapshots.append(T[p_idx].copy())
        print(
            f"  [{k+1:3d}/{n_train}]  T_p ∈ [{T[p_idx].min():.2f}, {T[p_idx].max():.2f}]  "
            f"({_time.perf_counter()-t0:.2f}s)"
        )

    # SVD
    print("\n" + "-" * 72)
    print("SVD on port-snapshot matrix")
    print("-" * 72)
    S = np.column_stack(snapshots)
    print(f"  Snapshot matrix: {S.shape[0]} ports × {S.shape[1]} samples")

    U, s, Vt = np.linalg.svd(S, full_matrices=False)
    cum = np.cumsum(s**2) / np.sum(s**2)
    r = min(max(int(np.searchsorted(cum, ENERGY_THR) + 1), 2), len(s))
    for k in range(min(8, len(s))):
        mark = "  ← r" if k == r else ""
        print(f"    s[{k}] = {s[k]:.6e}  cum. = {cum[k]*100:.4f}%{mark}")
    print(f"  Selected rank r = {r}  (>{ENERGY_THR*100:.1f}% energy)")
    U_r = U[:, :r]
    del snapshots, S, U, s, Vt

    # ================================================================
    #  Phase 2: Steady ROM evaluation
    # ================================================================
    print("\n" + "=" * 72)
    print(f"Phase 2: Steady ROM — {n_test} test samples (nominal h)")
    print("=" * 72)

    t0 = _time.perf_counter()
    blocks_steady = bci_schur_complement(K_ref, e_idx, p_idx, i_idx, U_r)
    print(f"  Pre-computed Schur complement: {_time.perf_counter()-t0:.3f}s")

    steady_errors = []
    for k, ti in enumerate(test_idx):
        t0 = _time.perf_counter()
        hs = {
            name: str(ptrace[ti, EV6_NAMES.index(name)] / block_volume[name])
            for name in EV6_NAMES
        }
        m = build_example2_model(hs, h_conv=H_CONV_NOMINAL)
        c = m.compile()
        sol = c.solve()
        T_full = sol.cell_temperatures().copy()
        f_test = c.assemble().rhs()
        sol.close()
        c.close()
        m.close()

        T_rom = bci_steady_solve(
            K_ref, f_test, e_idx, p_idx, i_idx, U_r, blocks=blocks_steady
        )
        diff = T_rom - T_full
        rel = np.linalg.norm(diff) / np.linalg.norm(T_full)
        mx = np.max(np.abs(diff))
        steady_errors.append((rel, mx))
        print(
            f"  [{k+1:3d}/{n_test}]  rel.err = {rel:.2e}  max|err| = {mx:.4e}  "
            f"({_time.perf_counter()-t0:.2f}s)"
        )

    rels = [e[0] for e in steady_errors]
    mxs = [e[1] for e in steady_errors]
    print(f"\n  Steady ROM ({n_test} cases):")
    print(f"    mean rel. error = {np.mean(rels):.2e}  max = {np.max(rels):.2e}")
    print(f"    mean max|err|   = {np.mean(mxs):.4e}  max = {np.max(mxs):.4e}")

    # ================================================================
    #  Phase 3: Transient ROM  (uniform T0 = 318.15 K)
    #  Full transient uses c.step() — the new single-step API.
    # ================================================================
    print("\n" + "=" * 72)
    print(f"Phase 3: Transient — {N_TRANSIENT_STEPS} steps × {SAMPLING_INTVL}s")
    print("         Uniform T0 = 318.15 K (not steady pre-heated)")
    print("         Full ref uses c.step() — new single-step API")
    print("=" * 72)

    check_step = 100
    checkpts = list(range(0, N_TRANSIENT_STEPS + 1, check_step))
    if checkpts[-1] != N_TRANSIENT_STEPS:
        checkpts.append(N_TRANSIENT_STEPS)

    # --- 3a. Full transient (via c.step()) ---
    print("\n  --- Full transient (c.step()) ---")
    t0 = _time.perf_counter()

    m_full_t = metahotspot.Model()
    for name in EV6_NAMES:
        col = EV6_NAMES.index(name)
        q_v = ptrace[:, col] / block_volume[name]
        m_full_t.add_function_periodic_piecewise_constant(
            name=f"power_{name}",
            values=q_v,
            period=SAMPLING_INTVL,
        )
    hs_fn = {name: f"power_{name}(t)" for name in EV6_NAMES}
    build_example2_model(hs_fn, model=m_full_t, h_conv=H_CONV_NOMINAL)
    m_full_t.set_settings(
        study=enums.Study.TRANSIENT,
        length_unit=enums.LengthUnit.METER,
        initial_temperature_K=T_AMB,
    )
    c_full_t = m_full_t.compile()
    T0 = np.full(nc, T_AMB, dtype=np.float64)

    T = T0.copy()
    full_check = {0: T0.copy()}
    for n in range(N_TRANSIENT_STEPS):
        T, _info = c_full_t.step(T, time=n * SAMPLING_INTVL, dt=SAMPLING_INTVL)
        if (n + 1) in checkpts:
            full_check[n + 1] = T.copy()

    t_full = _time.perf_counter() - t0
    for step in checkpts:
        Ta = full_check[step]
        print(
            f"    t = {step*SAMPLING_INTVL:5.2f}s  T ∈ [{Ta.min():.2f}, {Ta.max():.2f}] K"
        )
    c_full_t.close()
    m_full_t.close()
    print(f"  Full transient time: {t_full:.1f}s")

    # --- 3b. ROM transient (BCI-ROM BE, unchanged) ---
    print("\n  --- ROM transient (BCI-ROM BE) ---")
    t0 = _time.perf_counter()

    m_rom_t = metahotspot.Model()
    for name in EV6_NAMES:
        col = EV6_NAMES.index(name)
        q_v = ptrace[:, col] / block_volume[name]
        m_rom_t.add_function_periodic_piecewise_constant(
            name=f"power_{name}",
            values=q_v,
            period=SAMPLING_INTVL,
        )
    build_example2_model(hs_fn, model=m_rom_t, h_conv=H_CONV_NOMINAL)
    m_rom_t.set_settings(
        study=enums.Study.TRANSIENT,
        length_unit=enums.LengthUnit.METER,
        initial_temperature_K=T_AMB,
    )
    c_rom_t = m_rom_t.compile()
    K_rom = c_rom_t.assemble().stiffness_matrix()
    C_rom = c_rom_t.assemble().capacity_matrix()

    ctx = bci_transient_setup(K_rom, C_rom, e_idx, p_idx, i_idx, U_r, SAMPLING_INTVL)

    rom_check = {0: T0.copy()}
    T_rom_arr = T0.copy()
    for n in range(N_TRANSIENT_STEPS):
        t_np1 = (n + 1) * SAMPLING_INTVL
        f = c_rom_t.assemble(time=t_np1).rhs()
        T_rom_arr = bci_transient_step(T_rom_arr, f, ctx, e_idx, p_idx, i_idx, U_r)
        if (n + 1) in checkpts:
            rom_check[n + 1] = T_rom_arr.copy()
            Tr, Tf = rom_check[n + 1], full_check[n + 1]
            rel = np.linalg.norm(Tr - Tf) / np.linalg.norm(Tf)
            print(
                f"    t = {t_np1:5.2f}s  T ∈ [{Tr.min():.2f}, {Tr.max():.2f}] K  "
                f"rel.err = {rel:.2e}"
            )

    t_rom = _time.perf_counter() - t0
    c_rom_t.close()
    m_rom_t.close()
    print(f"  ROM transient time: {t_rom:.1f}s  (speedup = {t_full/t_rom:.1f}×)")

    # Transient error table
    print(f"\n  {'t (s)':>8s}  {'rel.error':>12s}  {'max|err| (K)':>14s}")
    print("  " + "-" * 38)
    for step in sorted(full_check):
        if step == 0:
            continue
        Tr, Tf = rom_check[step], full_check[step]
        diff = Tr - Tf
        print(
            f"  {step*SAMPLING_INTVL:>8.2f}  {np.linalg.norm(diff)/np.linalg.norm(Tf):>12.2e}  "
            f"{np.max(np.abs(diff)):>14.4e}"
        )

    # ================================================================
    #  Phase 4: BCI cross-h — different convection coefficients
    # ================================================================
    print("\n" + "=" * 72)
    print("Phase 4: BCI — cross-h convection variation")
    print("=" * 72)
    print(f"  Basis trained at nominal h = {H_CONV_NOMINAL:.4e} W/m2K")
    print(f"  Testing at other h values (average-power steady)")

    h_ratios = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    h_tests = [H_CONV_NOMINAL * s for s in h_ratios]
    bci_errors = []

    hs_avg = {
        name: str(ptrace[:, EV6_NAMES.index(name)].mean() / block_volume[name])
        for name in EV6_NAMES
    }

    for h_test in h_tests:
        t0 = _time.perf_counter()
        m_f = build_example2_model(hs_avg, h_conv=h_test)
        c_f = m_f.compile()
        sol = c_f.solve()
        T_full_h = sol.cell_temperatures().copy()
        f_h = c_f.assemble().rhs()
        K_h = c_f.assemble().stiffness_matrix()
        sol.close()
        c_f.close()
        m_f.close()

        T_rom_h = bci_steady_solve(K_h, f_h, e_idx, p_idx, i_idx, U_r)
        diff = T_rom_h - T_full_h
        rel = np.linalg.norm(diff) / np.linalg.norm(T_full_h)
        mx = np.max(np.abs(diff))
        bci_errors.append((h_test, rel, mx))
        print(
            f"  h = {h_test:10.2e}  rel.err = {rel:.2e}  max|err| = {mx:.4e}  "
            f"({_time.perf_counter()-t0:.2f}s)"
        )

    print(f"\n  {'h (W/m2K)':>14s}  {'rel.error':>12s}  {'max|err| (K)':>14s}")
    print("  " + "-" * 42)
    for h_val, rel, mx in bci_errors:
        print(f"  {h_val:>14.2e}  {rel:>12.2e}  {mx:>14.4e}")

    # Summary
    max_trans = max(
        (
            np.linalg.norm(rom_check[s] - full_check[s]) / np.linalg.norm(full_check[s])
            for s in full_check
            if s > 0
        ),
        default=0.0,
    )

    print("\n" + "=" * 72)
    print("  SUMMARY — Spreader+Sink BCI-ROM macro-model")
    print("=" * 72)
    print(f"  DOFs:  e={ne}  p={np_}  i={ni}  total={nc}")
    print(f"  ROM:   e={ne}  r={r}  total={ne+r}  ({100*(ne+r)/nc:.1f}%)")
    print()
    print(f"  Steady ({n_test} test cases):")
    print(f"    mean rel. err = {np.mean(rels):.2e}  max = {np.max(rels):.2e}")
    print()
    print(f"  Transient ({N_TRANSIENT_STEPS} steps):")
    print(f"    max checkpoint rel. err = {max_trans:.2e}")
    print(f"    speedup = {t_full/t_rom:.1f}×")
    print()
    print(f"  BCI cross-h ({len(bci_errors)} test values):")
    print(f"    max rel. err = {np.max([e[1] for e in bci_errors]):.2e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
