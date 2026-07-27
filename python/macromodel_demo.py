#!/usr/bin/env python3
"""
Macro-model demo: Guyan static condensation through the metahotspot Python package.

Workflow:
  1. Load XML → compile → assemble K, f
  2. Full solve → reference (using compiled.solve())
  3. Select macro region by layer/block ID → partition into ports + internals
  4. Guyan static condensation on (e+p) block: condense internal(i) → port(p)
  5. Solve reduced (e+p) system → recover internal DOFs
  6. Verify ||T_rec - T_ref||
"""
import sys
import time as _time
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

import metahotspot


def partition_macro(K, nc, layer_ids, block_ids, macro_layer, macro_block):
    """Partition mesh into external (e), port (p), and internal (i) sets.

    Port = macro cells adjacent to at least one external cell.
    Internal = macro cells NOT adjacent to any external cell.
    External = all non-macro cells.

    Returns (e_idx, p_idx, i_idx) — sorted ascending index arrays.
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

    e_idx = np.sort(np.where(~is_macro)[0])
    p_idx = np.sort(np.where(is_port)[0])
    i_idx = np.sort(np.where(is_macro & ~is_port)[0])
    return e_idx, p_idx, i_idx


def guyan_reduce(K, f, e_idx, p_idx, i_idx):
    """Guyan static condensation: condense internal(i) → (e+p) block.

    Builds the (e+p)×(e+p) Schur-complement system and factors K_ii.
    Returns (K_s_lu, K_ii_lu, f_s, K_ip) for forward/recovery.
    """
    ne, np_, ni = len(e_idx), len(p_idx), len(i_idx)
    N_ep = ne + np_

    K_ee = K[e_idx, :][:, e_idx]
    K_ep = K[e_idx, :][:, p_idx]
    K_pe = K[p_idx, :][:, e_idx]
    K_pp = K[p_idx, :][:, p_idx]
    K_pi = K[p_idx, :][:, i_idx]
    K_ip = K[i_idx, :][:, p_idx]
    K_ii = K[i_idx, :][:, i_idx]
    f_e = f[e_idx]
    f_p = f[p_idx]
    f_i = f[i_idx]

    print(f"    factoring K_ii ({ni}x{ni})...", end=" ", flush=True)
    K_ii_lu = splu(K_ii.tocsc())
    print("done", flush=True)

    # Assemble Schur complement for (e+p) block
    rows, cols, vals = [], [], []

    def _add(M, roff, coff):
        Mc = M.tocoo()
        rows.extend((Mc.row + roff).tolist())
        cols.extend((Mc.col + coff).tolist())
        vals.extend(Mc.data.tolist())

    _add(K_ee, 0, 0)
    _add(K_ep, 0, ne)
    _add(K_pe, ne, 0)
    _add(K_pp, ne, ne)

    K_ip_csc = K_ip.tocsc()
    K_pi_csr = K_pi.tocsr()

    print(f"    building K_s ({np_} port columns)...", end=" ", flush=True)
    for j in range(np_):
        col = K_ip_csc[:, j].toarray().ravel()
        x = K_ii_lu.solve(col)
        delta = K_pi_csr.dot(x)
        nonz = np.where(np.abs(delta) > 1e-30)[0]
        for k in nonz:
            rows.append(ne + k)
            cols.append(ne + j)
            vals.append(-delta[k])

    K_s = coo_matrix((vals, (rows, cols)), shape=(N_ep, N_ep)).tocsc()
    K_s.eliminate_zeros()
    print(f"done (NNZ={K_s.nnz})", flush=True)

    print(f"    factoring K_s ({N_ep}x{N_ep})...", end=" ", flush=True)
    K_s_lu = splu(K_s)
    print("done", flush=True)

    f_s = np.zeros(N_ep, dtype=np.float64)
    f_s[:ne] = f_e
    f_s[ne:] = f_p - K_pi @ K_ii_lu.solve(f_i)

    return K_s_lu, K_ii_lu, f_s, K_ip


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
    meta = compiled.metadata()
    nc = meta.cell_count
    layer_ids = meta.layer_ids.copy()
    block_ids = meta.block_ids.copy()

    K, C, f = compiled.assemble(compiled.default_state())
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

    e_idx, p_idx, i_idx = partition_macro(
        K, nc, layer_ids, block_ids, MACRO_LAYER, MACRO_BLOCK
    )
    ne, np_, ni = len(e_idx), len(p_idx), len(i_idx)
    print(f"  External: {ne} DOFs,  Ports: {np_} DOFs,  Internal: {ni} DOFs")

    if ni == 0:
        print("  No internal DOFs, nothing to condense.")
        return 0

    # ── Step 4: Guyan static condensation ──────────────────────
    print("\n" + "=" * 60)
    print("Step 4: Guyan static condensation")
    print("=" * 60)

    K_s_lu, K_ii_lu, f_s, K_ip = guyan_reduce(K, f, e_idx, p_idx, i_idx)

    # ── Step 5: Solve reduced system + recover ─────────────────
    print("\n" + "=" * 60)
    print("Step 5: Solve reduced (e+p) system → recover internal")
    print("=" * 60)

    sol_ep = K_s_lu.solve(f_s)
    u_e = sol_ep[:ne]
    u_p = sol_ep[ne:]
    u_i = K_ii_lu.solve(f[i_idx] - K_ip @ u_p)

    T_rec = np.zeros(nc, dtype=np.float64)
    T_rec[e_idx] = u_e
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
        f_k[e_idx] *= 1.0 + 0.1 * np.sin(k)
        _ = K_full_lu.solve(f_k)
    t_full_sw = _time.perf_counter() - t0

    t0 = _time.perf_counter()
    for k in range(N_SWEEP):
        f_s_k = f_s.copy()
        f_s_k[:ne] *= 1.0 + 0.1 * np.sin(k)
        sol_k = K_s_lu.solve(f_s_k)
        _ = K_ii_lu.solve(f[i_idx] - K_ip @ sol_k[ne:])
    t_guyan_sw = _time.perf_counter() - t0

    print(
        f"  {'Method':<12s}  {'Sweep(s)':>10s}",
    )
    print("  " + "-" * 24)
    print(f"  {'Full':<12s}  {t_full_sw:>10.3f}")
    print(f"  {'Guyan (e+p)':<12s}  {t_guyan_sw:>10.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
