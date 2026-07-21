#!/usr/bin/env python3
"""
Macro-model demo: Guyan static condensation with correct port detection
entirely through the MetaHotspot C API (ctypes), no files exported.

Workflow:
  1. Load XML -> compile -> assemble K, f
  2. Full solve -> reference
  3. Select macro region by layer/block ID -> partition into ports + internals
  4. Guyan static condensation on (e+p) block: condense internal(i) → port(p)
     using Schur complement K_s = [K_ee K_ep; K_pe K_pp - K_pi*inv(K_ii)*K_ip]
  5. Solve reduced (e+p) system -> recover internal DOFs
  6. Verify ||u - u_ref||
  7. RHS sweep inference
"""
import ctypes
import sys
import time as _time
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import splu

# ---------------------------------------------------------------------------
#  ctypes wrapper
# ---------------------------------------------------------------------------
_LIB_DIR = Path(__file__).resolve().parent.parent / "build" / "src" / "api"
if sys.platform.startswith("linux"):
    _LIB_NAME = "libmhs_c_api.so"
elif sys.platform == "darwin":
    _LIB_NAME = "libmhs_c_api.dylib"
elif sys.platform == "win32":
    _LIB_NAME = "mhs_c_api.dll"
else:
    raise OSError("Unsupported platform for MetaHotspot C API: {}".format(sys.platform))

DLL = ctypes.CDLL(str(_LIB_DIR / _LIB_NAME))


class MhsModel(ctypes.Structure):
    pass


class MhsCompiled(ctypes.Structure):
    pass


class MhsSolution(ctypes.Structure):
    pass


class MhsAssembly(ctypes.Structure):
    pass


DLL.mhs_model_create.restype = ctypes.c_int32
DLL.mhs_model_create.argtypes = [ctypes.POINTER(ctypes.POINTER(MhsModel))]
DLL.mhs_model_destroy.restype = ctypes.c_int32
DLL.mhs_model_destroy.argtypes = [ctypes.POINTER(MhsModel)]
DLL.mhs_model_read_xml.restype = ctypes.c_int32
DLL.mhs_model_read_xml.argtypes = [ctypes.POINTER(MhsModel), ctypes.c_char_p]
DLL.mhs_model_compile.restype = ctypes.c_int32
DLL.mhs_model_compile.argtypes = [
    ctypes.POINTER(MhsModel),
    ctypes.POINTER(ctypes.POINTER(MhsCompiled)),
]
DLL.mhs_compiled_destroy.restype = ctypes.c_int32
DLL.mhs_compiled_destroy.argtypes = [ctypes.POINTER(MhsCompiled)]
DLL.mhs_compiled_cell_count.restype = ctypes.c_int32
DLL.mhs_compiled_cell_count.argtypes = [ctypes.POINTER(MhsCompiled)]
DLL.mhs_compiled_layer_ids.restype = ctypes.POINTER(ctypes.c_int32)
DLL.mhs_compiled_layer_ids.argtypes = [ctypes.POINTER(MhsCompiled)]
DLL.mhs_compiled_block_ids.restype = ctypes.POINTER(ctypes.c_int32)
DLL.mhs_compiled_block_ids.argtypes = [ctypes.POINTER(MhsCompiled)]
DLL.mhs_compiled_assemble.restype = ctypes.c_int32
DLL.mhs_compiled_assemble.argtypes = [
    ctypes.POINTER(MhsCompiled),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_double,
    ctypes.POINTER(ctypes.POINTER(MhsAssembly)),
]
DLL.mhs_assembly_destroy.restype = ctypes.c_int32
DLL.mhs_assembly_destroy.argtypes = [ctypes.POINTER(MhsAssembly)]
DLL.mhs_assembly_n.restype = ctypes.c_int32
DLL.mhs_assembly_n.argtypes = [ctypes.POINTER(MhsAssembly)]
DLL.mhs_assembly_nnz.restype = ctypes.c_int32
DLL.mhs_assembly_nnz.argtypes = [ctypes.POINTER(MhsAssembly)]
DLL.mhs_assembly_outer_indices.restype = ctypes.POINTER(ctypes.c_int32)
DLL.mhs_assembly_outer_indices.argtypes = [ctypes.POINTER(MhsAssembly)]
DLL.mhs_assembly_inner_indices.restype = ctypes.POINTER(ctypes.c_int32)
DLL.mhs_assembly_inner_indices.argtypes = [ctypes.POINTER(MhsAssembly)]
DLL.mhs_assembly_values.restype = ctypes.POINTER(ctypes.c_double)
DLL.mhs_assembly_values.argtypes = [ctypes.POINTER(MhsAssembly)]
DLL.mhs_assembly_rhs.restype = ctypes.POINTER(ctypes.c_double)
DLL.mhs_assembly_rhs.argtypes = [ctypes.POINTER(MhsAssembly)]
DLL.mhs_compiled_solve.restype = ctypes.c_int32
DLL.mhs_compiled_solve.argtypes = [
    ctypes.POINTER(MhsCompiled),
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.POINTER(MhsSolution)),
]
DLL.mhs_solution_destroy.restype = ctypes.c_int32
DLL.mhs_solution_destroy.argtypes = [ctypes.POINTER(MhsSolution)]
DLL.mhs_solution_cell_temperatures.restype = ctypes.POINTER(ctypes.c_double)
DLL.mhs_solution_cell_temperatures.argtypes = [ctypes.POINTER(MhsSolution)]
DLL.mhs_last_error.restype = ctypes.c_char_p
DLL.mhs_last_error.argtypes = []


def _check(stat, ctx="(no context)"):
    if stat != 0:
        err = DLL.mhs_last_error()
        if isinstance(err, bytes):
            err = err.decode("utf-8", errors="replace")
        raise RuntimeError("API error {} in {}: {}".format(stat, ctx, err))


def _assembly_to_scipy(ah):
    """Return (K_csc, f) from an mhs_assembly_t handle."""
    n = DLL.mhs_assembly_n(ah)
    nnz = DLL.mhs_assembly_nnz(ah)
    outer = DLL.mhs_assembly_outer_indices(ah)
    inner = DLL.mhs_assembly_inner_indices(ah)
    vals = DLL.mhs_assembly_values(ah)
    rhs_p = DLL.mhs_assembly_rhs(ah)
    K = csc_matrix(
        (
            np.ctypeslib.as_array(vals, shape=(nnz,)).copy(),
            np.ctypeslib.as_array(inner, shape=(nnz,)).copy(),
            np.ctypeslib.as_array(outer, shape=(n + 1,)).copy(),
        ),
        shape=(n, n),
    )
    return K, np.ctypeslib.as_array(rhs_p, shape=(n,)).copy()


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
#  Guyan static condensation (corrected)
# ===========================================================================


def _guyan_reduce(K, rhs, e_idx, p_idx, i_idx):
    """Guyan static condensation: condense internal(i) → (e+p) block.

    Builds the (e+p)×(e+p) Schur-complement system:
      K_s = [K_ee  K_ep;  K_pe  K_pp - K_pi*inv(K_ii)*K_ip]

    Since K_ei = K_ie = 0 (structurally — internal never contacts external),
    only the (p,p) block is modified.  No column-by-column inflation of the
    external block is needed (this was the bug in the original code).
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

    # Build (e+p) Schur complement.
    # K_s = K_bb - K_bi * inv(K_ii) * K_ib
    # K_bi = [0; K_pi] → only the (p,p) block is modified
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


def _guyan_forward_reduce(K, rhs, e_idx, p_idx, i_idx):
    """Direct (e+p+i) solve without condensation — for timing comparison.

    Equivalent to assembling the full (e+p+i) system and factoring it once.
    """
    N = len(e_idx) + len(p_idx) + len(i_idx)
    rows, cols, vals = [], [], []

    def _add(M, roff, coff):
        Mc = M.tocoo()
        rows.extend((Mc.row + roff).tolist())
        cols.extend((Mc.col + coff).tolist())
        vals.extend(Mc.data.tolist())

    _add(K[e_idx, :][:, e_idx], 0, 0)
    _add(K[e_idx, :][:, p_idx], 0, len(e_idx))
    _add(K[e_idx, :][:, i_idx], 0, len(e_idx) + len(p_idx))
    _add(K[p_idx, :][:, e_idx], len(e_idx), 0)
    _add(K[p_idx, :][:, p_idx], len(e_idx), len(e_idx))
    _add(K[p_idx, :][:, i_idx], len(e_idx), len(e_idx) + len(p_idx))
    _add(K[i_idx, :][:, e_idx], len(e_idx) + len(p_idx), 0)
    _add(K[i_idx, :][:, p_idx], len(e_idx) + len(p_idx), len(e_idx))
    _add(K[i_idx, :][:, i_idx], len(e_idx) + len(p_idx), len(e_idx) + len(p_idx))

    K_sys = coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsc()
    K_sys.eliminate_zeros()

    t0 = _time.perf_counter()
    K_sys_lu = splu(K_sys)
    t_factor = _time.perf_counter() - t0

    rhs_all = np.concatenate([rhs[e_idx], rhs[p_idx], rhs[i_idx]])
    sol_all = K_sys_lu.solve(rhs_all)
    return sol_all, t_factor


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
    mp = ctypes.POINTER(MhsModel)()
    _check(DLL.mhs_model_create(ctypes.byref(mp)), "create")
    m = mp

    _check(DLL.mhs_model_read_xml(m, str(case_path).encode("utf-8")), "read_xml")

    cp = ctypes.POINTER(MhsCompiled)()
    _check(DLL.mhs_model_compile(m, ctypes.byref(cp)), "compile")
    c = cp

    nc = DLL.mhs_compiled_cell_count(c)
    print("  Active cells: {}".format(nc))

    layer_p = DLL.mhs_compiled_layer_ids(c)
    block_p = DLL.mhs_compiled_block_ids(c)
    layer_ids = np.ctypeslib.as_array(layer_p, shape=(nc,)).copy()
    block_ids = np.ctypeslib.as_array(block_p, shape=(nc,)).copy()

    ap = ctypes.POINTER(MhsAssembly)()
    _check(DLL.mhs_compiled_assemble(c, None, 0.0, ctypes.byref(ap)), "assemble")
    K, rhs = _assembly_to_scipy(ap)
    print("  K: {}x{}, {} NNZ".format(K.shape[0], K.shape[1], K.nnz))
    DLL.mhs_assembly_destroy(ap)

    # ==================================================================
    #  Step 2: Full solve -> reference
    # ==================================================================
    print("\n" + "=" * 60)
    print("Step 2: Full solve -> reference")
    print("=" * 60)
    sp = ctypes.POINTER(MhsSolution)()
    _check(DLL.mhs_compiled_solve(c, None, ctypes.byref(sp)), "solve")
    T_ref_ptr = DLL.mhs_solution_cell_temperatures(sp)
    T_ref = np.ctypeslib.as_array(T_ref_ptr, shape=(nc,)).copy()
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
        DLL.mhs_solution_destroy(sp)
        DLL.mhs_compiled_destroy(c)
        DLL.mhs_model_destroy(m)
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
    DLL.mhs_solution_destroy(sp)
    DLL.mhs_compiled_destroy(c)
    DLL.mhs_model_destroy(m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
