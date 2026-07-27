#!/usr/bin/env python3
"""
Macro-model demo: Guyan static condensation through the metahotspot Python package.

Workflow:
  1. Load XML -> compile -> assemble K, f
  2. Full solve -> reference
  3. Select macro region by layer/block ID -> partition into ports + internals
  4. Guyan static condensation on (e+p) block: condense internal(i) → port(p)
  5. Solve reduced (e+p) system -> recover internal DOFs
  6. Verify ||u - u_ref||
  7. RHS sweep inference
"""
import sys
import time as _time
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import splu

import metahotspot

# ===========================================================================
#  Partitioning
# ===========================================================================


def _partition_macro(K, nc, layer_ids, block_ids, macro_layer, macro_block):
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


# ===========================================================================
#  Guyan static condensation
# ===========================================================================


def _guyan_reduce(K, rhs, e_idx, p_idx, i_idx):
    """Guyan static condensation: condense internal(i) → (e+p) block.

    Builds the (e+p)×(e+p) Schur-complement system:
      K_s = [K_ee  K_ep;  K_pe  K_pp - K_pi*inv(K_ii)*K_ip]

    Since K_ei = K_ie = 0 (structurally — internal never contacts external),
    only the (p,p) block is modified.
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
    f_e = rhs[e_idx]
    f_p = rhs[p_idx]
    f_i = rhs[i_idx]

    t0 = _time.perf_counter()
    print("    factoring K_ii ({}x{})...".format(ni, ni), end=" ", flush=True)
    K_ii_lu = splu(K_ii.tocsc())
    print("done ({:.2f}s)".format(_time.perf_counter() - t0), flush=True)

    # Build (e+p) Schur complement
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

    t0_build = _time.perf_counter()
    print("    building K_s ({} port cols) ...".format(np_), end=" ", flush=True)
    for j in range(np_):
        col = K_ip_csc[:, j].toarray().ravel()
        x = K_ii_lu.solve(col)
        delta = K_pi_csr.dot(x)
        nonz = np.where(np.abs(delta) > 1e-30)[0]
        for k in nonz:
            rows.append(ne + k)
            cols.append(ne + j)
            vals.append(-delta[k])
    print(
        "done ({:.2f}s, NNZ~{})".format(_time.perf_counter() - t0_build, len(vals)),
        flush=True,
    )

    K_s = coo_matrix((vals, (rows, cols)), shape=(N_ep, N_ep)).tocsc()
    K_s.eliminate_zeros()

    t0 = _time.perf_counter()
    print(
        "    factoring K_s ({}x{}, {} NNZ)...".format(N_ep, N_ep, K_s.nnz),
        end=" ",
        flush=True,
    )
    K_s_lu = splu(K_s)
    print("done ({:.2f}s)".format(_time.perf_counter() - t0), flush=True)

    f_s = np.zeros(N_ep, dtype=np.float64)
    f_s[:ne] = f_e
    f_s[ne:] = f_p - K_pi @ K_ii_lu.solve(f_i)

    return K_s_lu, K_ii_lu, f_s, K_ip


# ===========================================================================
#  Main
# ===========================================================================
def main():
    case_path = (
        Path(__file__).resolve().parent.parent
        / "cases/simple_steady_cases/simple_steady_case2.xml"
    )
    if not case_path.exists():
        print("ERROR: case not found: {}".format(case_path), file=sys.stderr)
        return 1

    MACRO_LAYER = 0
    MACRO_BLOCK = 0

    # ==================================================================
    #  Step 1: Load -> compile -> assemble
    # ==================================================================
    print("=" * 60)
    print("Step 1: Load XML -> compile -> assemble")
    print("=" * 60)

    model = metahotspot.Model()
    model.read_xml(case_path)
    compiled = model.compile()
    meta = compiled.metadata()
    nc = meta.cell_count
    print("  Active cells: {}".format(nc))

    layer_ids = meta.layer_ids.copy()
    block_ids = meta.block_ids.copy()

    assembly = compiled.assemble(compiled.default_state())
    K, C, rhs = assembly.K, assembly.C, assembly.f
    assembly.close()
    print("  K: {}x{}, {} NNZ".format(K.shape[0], K.shape[1], K.nnz))

    # ==================================================================
    #  Step 2: Full solve -> reference
    # ==================================================================
    print("\n" + "=" * 60)
    print("Step 2: Full solve -> reference")
    print("=" * 60)

    solution = compiled.solve()
    T_ref = solution.temperature.copy()
    print("  T in [{:.4f}, {:.4f}] K".format(T_ref.min(), T_ref.max()))

    # ==================================================================
    #  Step 3: Partition — correct port detection
    # ==================================================================
    print("\n" + "=" * 60)
    print(
        "Step 3: Partition macro region by layer={}, block={}".format(
            MACRO_LAYER, MACRO_BLOCK
        )
    )
    print("=" * 60)
    e_idx, p_idx, i_idx = _partition_macro(
        K, nc, layer_ids, block_ids, MACRO_LAYER, MACRO_BLOCK
    )
    ne, np_, ni = len(e_idx), len(p_idx), len(i_idx)
    print("  External (layers 1&2, kept full):     {:>6} DOFs".format(ne))
    print("  Macro ports (surface interface):      {:>6} DOFs".format(np_))
    print("  Macro internal (condensed):           {:>6} DOFs".format(ni))

    if ni == 0:
        print("  No internal DOFs, nothing to condense.")
        solution.close()
        compiled.close()
        model.close()
        return 0

    # ==================================================================
    #  Step 4: Guyan static condensation
    # ==================================================================
    print("\n" + "=" * 60)
    print("Step 4: Guyan static condensation")
    print("=" * 60)
    guyan_t0 = _time.perf_counter()
    K_s_lu, K_ii_lu, f_s, K_ip = _guyan_reduce(K, rhs, e_idx, p_idx, i_idx)

    sol_ep = K_s_lu.solve(f_s)
    u_e = sol_ep[:ne]
    u_p = sol_ep[ne:]
    u_i = K_ii_lu.solve(rhs[i_idx] - K_ip @ u_p)
    guyan_time = _time.perf_counter() - guyan_t0

    T_guyan = np.zeros(nc, dtype=np.float64)
    T_guyan[e_idx] = u_e
    T_guyan[p_idx] = u_p
    T_guyan[i_idx] = u_i
    diff = T_guyan - T_ref
    rel_err = np.linalg.norm(diff) / np.linalg.norm(T_ref)
    print("  ||T_rec - T_ref|| / ||T_ref|| = {:.2e}".format(rel_err))
    print("  max|diff| = {:.6e}".format(np.max(np.abs(diff))))

    # ==================================================================
    #  Step 5: RHS sweep inference
    # ==================================================================
    print("\n" + "=" * 60)
    print("Step 5: RHS sweep inference ({} sweeps)".format(50))
    print("=" * 60)
    N_SWEEP = 50

    # Full sweeps
    t0 = _time.perf_counter()
    K_full_lu = splu(K.tocsc())
    for k in range(N_SWEEP):
        f_k = rhs.copy()
        f_k[e_idx] *= 1.0 + 0.1 * np.sin(k)
        _ = K_full_lu.solve(f_k)
    t_full_sw = _time.perf_counter() - t0

    # Guyan sweeps
    t0 = _time.perf_counter()
    for k in range(N_SWEEP):
        f_s_k = f_s.copy()
        f_s_k[:ne] *= 1.0 + 0.1 * np.sin(k)
        sol_k = K_s_lu.solve(f_s_k)
        _ = K_ii_lu.solve(rhs[i_idx] - K_ip @ sol_k[ne:])
    t_guyan_sw = _time.perf_counter() - t0

    print(
        "  {:<12s}  {:>12s}  {:>10s}  {:>14s}".format(
            "Method", "Offline(s)", "Sweep(s)", "Total+RHS(52)"
        )
    )
    print("  " + "-" * 54)
    print(
        "  {:<12s}  {:>12.3f}  {:>10.3f}  {:>14.3f}".format(
            "Full", 0.0, t_full_sw, t_full_sw
        )
    )
    print(
        "  {:<12s}  {:>12.3f}  {:>10.3f}  {:>14.3f}".format(
            "Guyan (e+p)", guyan_time, t_guyan_sw, guyan_time + t_guyan_sw
        )
    )

    # ==================================================================
    #  Cleanup
    # ==================================================================
    solution.close()
    compiled.close()
    model.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
