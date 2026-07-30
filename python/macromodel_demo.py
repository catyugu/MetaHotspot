#!/usr/bin/env python3
"""
Macro-model demo: Guyan static condensation through the metahotspot Python package.

Workflow:
  1. Load XML → compile → assemble K, f
  2. Full solve → reference (using compiled.solve())
  3. Split the reference into detailed block, macro block, and interface
  4. Condense the standalone macro's internal(i) DoFs to its port(p)
  5. Couple the detailed block to that port operator, solve, recover internals
  6. Verify ||T_rec - T_ref||
"""
import sys
import time as _time
from pathlib import Path

import numpy as np
from scipy.sparse import bmat, csc_matrix, diags
from scipy.sparse.linalg import splu

import metahotspot


def partition_regions(K, nc, layer_ids, block_ids, macro_layer, macro_block):
    """Partition the reference mesh into detailed, macro-port, and macro-internal sets.

    Port = macro cells adjacent to at least one detailed cell.
    Internal = macro cells not adjacent to a detailed cell.

    These indices are used only to split the monolithic reference. The
    resulting macro operator has local ``[p, i]`` indices and no detailed DoF.
    """
    is_macro = (layer_ids == macro_layer) & (block_ids == macro_block)
    macro_set = set(np.where(is_macro)[0])

    port_set = set()
    for i in macro_set:
        rs, re = K.indptr[i], K.indptr[i + 1]
        for j in K.indices[rs:re]:
            if j not in macro_set:
                port_set.add(i)
                break

    is_port = np.zeros(nc, dtype=bool)
    for p in port_set:
        is_port[p] = True

    detailed_idx = np.sort(np.where(~is_macro)[0])
    p_idx = np.sort(np.where(is_port)[0])
    i_idx = np.sort(np.where(is_macro & ~is_port)[0])
    return detailed_idx, p_idx, i_idx


def split_reference_operators(K, f, detailed_idx, p_idx, i_idx):
    """Split a monolithic reference into independently owned operators.

    This is reference-fixture preparation, not macro condensation. It removes
    the four conservative interface contributions from the two diagonal
    subdomains and returns a standalone macro matrix ordered as ``[p, i]``.
    """
    K_dp = K[detailed_idx, :][:, p_idx].tocsc()
    K_pd = K[p_idx, :][:, detailed_idx].tocsc()
    D_d = diags(-np.asarray(K_dp.sum(axis=1)).ravel(), format="csc")
    D_p = diags(-np.asarray(K_pd.sum(axis=1)).ravel(), format="csc")

    K_detailed = K[detailed_idx, :][:, detailed_idx].tocsc() - D_d
    K_macro = bmat(
        [
            [K[p_idx, :][:, p_idx].tocsc() - D_p, K[p_idx, :][:, i_idx]],
            [K[i_idx, :][:, p_idx], K[i_idx, :][:, i_idx]],
        ],
        format="csc",
    )
    f_macro = np.concatenate([f[p_idx], f[i_idx]])
    interface = (D_d, K_dp, K_pd, D_p)
    return K_detailed, f[detailed_idx], K_macro, f_macro, interface


def condense_macro(K_macro, f_macro, port_count):
    """Condense a standalone ``[p, i]`` macro to its physical port ``p``."""
    p = np.arange(port_count)
    i = np.arange(port_count, K_macro.shape[0])
    K_pp = K_macro[p, :][:, p]
    K_pi = K_macro[p, :][:, i]
    K_ip = K_macro[i, :][:, p]
    K_ii = K_macro[i, :][:, i].tocsc()
    f_i = f_macro[i]

    print(f"    factoring K_ii ({len(i)}x{len(i)})...", end=" ", flush=True)
    K_ii_lu = splu(K_ii)
    print("done", flush=True)

    K_ip_csc = K_ip.tocsc()
    K_pi_csr = K_pi.tocsr()
    inverse_Kip = np.column_stack(
        [
            K_ii_lu.solve(K_ip_csc[:, column].toarray().ravel())
            for column in range(K_ip_csc.shape[1])
        ]
    )
    K_port = csc_matrix(K_pp - K_pi_csr @ inverse_Kip)
    f_port = np.asarray(f_macro[p] - K_pi_csr @ K_ii_lu.solve(f_i)).ravel()
    return K_port, f_port, K_ii_lu, K_ip_csc, f_i


def build_coupled_reduced_system(
    K_detailed,
    f_detailed,
    K_port,
    f_port,
    interface,
):
    """Connect the detailed region to a port-only condensed macro."""
    D_d, K_dp, K_pd, D_p = interface
    K_reduced = bmat(
        [
            [K_detailed + D_d, K_dp],
            [K_pd, K_port + D_p],
        ],
        format="csc",
    )
    f_reduced = np.concatenate([f_detailed, f_port])
    return K_reduced, f_reduced


def main():
    case_path = (
        Path(__file__).resolve().parent.parent
        / "cases/simple_steady_cases/simple_steady_case2.xml"
    )
    if not case_path.exists():
        print(f"ERROR: case not found: {case_path}", file=sys.stderr)
        return 1

    MACRO_LAYER = 0
    MACRO_BLOCK = 0

    # ── Step 1: Load → compile → assemble ──────────────────────
    print("=" * 60)
    print("Step 1: Load XML → compile → assemble K, f")
    print("=" * 60)

    model = metahotspot.Model()
    model.read_xml(str(case_path))
    compiled = model.compile()
    nc = compiled.cell_count
    layer_ids = compiled.layer_ids.copy()
    block_ids = compiled.block_ids.copy()

    K, C, f = compiled.assemble()
    print(f"  Active cells: {nc},  K: {K.shape[0]}x{K.shape[1]}, {K.nnz} NNZ")

    # ── Step 2: Full solve → reference ─────────────────────────
    print("\n" + "=" * 60)
    print("Step 2: Full solve → reference (compiled.solve())")
    print("=" * 60)

    solution = compiled.solve()
    T_ref = solution.temperature.copy()
    print(f"  T in [{T_ref.min():.4f}, {T_ref.max():.4f}] K")

    # ── Step 3: Partition ──────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Step 3: Partition  layer={MACRO_LAYER}, block={MACRO_BLOCK}")
    print("=" * 60)

    detailed_idx, p_idx, i_idx = partition_regions(
        K, nc, layer_ids, block_ids, MACRO_LAYER, MACRO_BLOCK
    )
    nd, np_, ni = len(detailed_idx), len(p_idx), len(i_idx)
    print(f"  Detailed: {nd} DOFs,  Ports: {np_} DOFs,  Internal: {ni} DOFs")

    if ni == 0:
        print("  No internal DOFs, nothing to condense.")
        return 0

    # ── Step 4: Guyan static condensation ──────────────────────
    print("\n" + "=" * 60)
    print("Step 4: Port-only macro condensation")
    print("=" * 60)

    K_detailed, f_detailed, K_macro, f_macro, interface = split_reference_operators(
        K, f, detailed_idx, p_idx, i_idx
    )
    K_port, f_port, K_ii_lu, K_ip, f_i = condense_macro(K_macro, f_macro, np_)
    K_reduced, f_reduced = build_coupled_reduced_system(
        K_detailed, f_detailed, K_port, f_port, interface
    )
    print(
        f"    factoring coupled detailed+port system "
        f"({K_reduced.shape[0]}x{K_reduced.shape[1]})...",
        end=" ",
        flush=True,
    )
    K_reduced_lu = splu(K_reduced)
    print("done", flush=True)

    # ── Step 5: Solve reduced system + recover ─────────────────
    print("\n" + "=" * 60)
    print("Step 5: Couple detailed region to macro port → solve + recover")
    print("=" * 60)

    sol_ep = K_reduced_lu.solve(f_reduced)
    u_d = sol_ep[:nd]
    u_p = sol_ep[nd:]
    u_i = K_ii_lu.solve(f_i - K_ip @ u_p)

    T_rec = np.zeros(nc, dtype=np.float64)
    T_rec[detailed_idx] = u_d
    T_rec[p_idx] = u_p
    T_rec[i_idx] = u_i

    diff = T_rec - T_ref
    rel_err = np.linalg.norm(diff) / np.linalg.norm(T_ref)
    print(f"  ||T_rec - T_ref|| / ||T_ref|| = {rel_err:.2e}")
    print(f"  max|diff| = {np.max(np.abs(diff)):.6e}")

    # ── Step 6: RHS sweep benchmark ───────────────────────────
    print("\n" + "=" * 60)
    print("Step 6: RHS sweep benchmark (50 sweeps)")
    print("=" * 60)
    N_SWEEP = 50

    t0 = _time.perf_counter()
    K_full_lu = splu(K.tocsc())
    for k in range(N_SWEEP):
        f_k = f.copy()
        f_k[detailed_idx] *= 1.0 + 0.1 * np.sin(k)
        _ = K_full_lu.solve(f_k)
    t_full_sw = _time.perf_counter() - t0

    t0 = _time.perf_counter()
    for k in range(N_SWEEP):
        f_reduced_k = f_reduced.copy()
        f_reduced_k[:nd] *= 1.0 + 0.1 * np.sin(k)
        sol_k = K_reduced_lu.solve(f_reduced_k)
        _ = K_ii_lu.solve(f_i - K_ip @ sol_k[nd:])
    t_guyan_sw = _time.perf_counter() - t0

    print(
        f"  {'Method':<12s}  {'Sweep(s)':>10s}",
    )
    print("  " + "-" * 24)
    print(f"  {'Full':<12s}  {t_full_sw:>10.3f}")
    print(f"  {'Port macro':<12s}  {t_guyan_sw:>10.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
