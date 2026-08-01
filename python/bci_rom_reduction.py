"""Sparse localized BCI-ROM extraction, projection, and benchmarking."""
from __future__ import annotations

from bci_rom_model import *

def orthonormal_range(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError('matrix must be two-dimensional')
    if matrix.shape[1] == 0:
        return np.empty((matrix.shape[0], 0))
    q, r, _ = scipy.linalg.qr(matrix, mode='economic', pivoting=True, check_finite=False)
    diagonal = np.abs(np.diag(r))
    if not diagonal.size or diagonal[0] == 0.0:
        return np.empty((matrix.shape[0], 0))
    tolerance = np.finfo(np.float64).eps * max(matrix.shape) * diagonal[0]
    return np.ascontiguousarray(q[:, diagonal > tolerance])

def projected_static_residual(Kii: sp.csc_matrix, Kip: sp.csc_matrix, W: sp.csc_matrix, block_size: int) -> float:
    Krr = (W.T @ Kii @ W).tocsc()
    lu = spla.splu(Krr)
    reduced_rhs = -(W.T @ Kip).tocsc()
    numerator = 0.0
    denominator = float(np.dot(Kip.data, Kip.data))
    for start in range(0, Kip.shape[1], block_size):
        stop = min(Kip.shape[1], start + block_size)
        coordinates = lu.solve(reduced_rhs[:, start:stop].toarray())
        residual = Kii @ (W @ coordinates) + Kip[:, start:stop].toarray()
        numerator += float(np.linalg.norm(residual, ord='fro') ** 2)
    return math.sqrt(numerator / max(denominator, np.finfo(float).tiny))

def build_basis(sample: Sample, run: Run) -> Basis:
    """Build a sparse localized spectral KMS basis without external inputs."""
    started = time.perf_counter()
    _, _, Kip, Kii, Cii, _, _ = split(sample)
    columns = macro_columns(sample.compiled)
    if len(columns) != sample.ports.port_count:
        raise RuntimeError('one exact physical port is required per macro column')
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    column_orders: list[int] = []
    retained_eigenvalues: list[float] = []
    local_residuals: list[float] = []
    offset = 0
    for port, cells in enumerate(columns):
        local_k = Kii[cells, :][:, cells].toarray()
        local_c = Cii[cells, :][:, cells].toarray()
        local_coupling = Kip[cells, port].toarray().ravel()
        static_shape = -scipy.linalg.solve(local_k, local_coupling, assume_a='sym', check_finite=False)
        eigenvalues, eigenvectors = scipy.linalg.eigh(local_k, local_c, check_finite=False)
        eligible = np.flatnonzero(eigenvalues <= run.modal_cutoff_per_s)
        dynamic = eligible[:run.dynamic_modes_per_column]
        candidates = np.column_stack((static_shape, np.ones(cells.size, dtype=np.float64), eigenvectors[:, dynamic]))
        local_basis = orthonormal_range(candidates)
        if local_basis.shape[1] == 0:
            raise RuntimeError('empty local basis')
        coupling_scale = max(np.linalg.norm(local_coupling), np.finfo(float).tiny)
        local_residuals.append(float(np.linalg.norm(local_k @ static_shape + local_coupling) / coupling_scale))
        retained_eigenvalues.extend(eigenvalues[dynamic].tolist())
        column_orders.append(local_basis.shape[1])
        for local_row, cell in enumerate(cells):
            nonzero = np.flatnonzero(np.abs(local_basis[local_row]) > 1e-14)
            rows.extend([int(cell)] * nonzero.size)
            cols.extend((offset + nonzero).tolist())
            values.extend(local_basis[local_row, nonzero].tolist())
        offset += local_basis.shape[1]
    W = sp.csc_matrix((values, (rows, cols)), shape=(Kii.shape[0], offset))
    gram = (W.T @ W).tocsc()
    identity = sp.eye(W.shape[1], format='csc')
    orthogonality_error = float(spla.norm(gram - identity))
    transfer_residual = projected_static_residual(Kii, Kip, W, run.residual_block_size)
    return Basis(W, np.asarray(column_orders, dtype=np.int64), np.asarray(retained_eigenvalues, dtype=np.float64), max(local_residuals), transfer_residual, orthogonality_error, time.perf_counter() - started)

def project(sample: Sample, W: sp.csc_matrix) -> Reduced:
    started = time.perf_counter()
    Kpp, Kpi, Kip, Kii, Cii, fp, fi = split(sample)
    p, r = (Kpp.shape[0], W.shape[1])
    Kpr = (Kpi @ W).tocsc()
    Krp = (W.T @ Kip).tocsc()
    Krr = (W.T @ Kii @ W).tocsc()
    Crr = (W.T @ Cii @ W).tocsc()
    Krr = (0.5 * (Krr + Krr.T)).tocsc()
    Crr = (0.5 * (Crr + Crr.T)).tocsc()
    zero_pp = sp.csc_matrix((p, p))
    zero_pr = sp.csc_matrix((p, r))
    K = sp.bmat(((Kpp, Kpr), (Krp, Krr)), format='csc')
    C = sp.bmat(((zero_pp, zero_pr), (zero_pr.T, Crr)), format='csc')
    K.eliminate_zeros()
    C.eliminate_zeros()
    f = np.r_[fp, np.asarray(W.T @ fi).ravel()]
    return Reduced(K, C, f, W, time.perf_counter() - started)

def options(run: Run, transient: bool) -> SolveOptions:
    dt = run.dt_s if transient else 1.0
    return SolveOptions(linear_solver='EigenSparseLU', linear_tolerance=1e-12, linear_max_iterations=5000, nonlinear_max_iterations=30, nonlinear_relative_tolerance=1e-11, nonlinear_absolute_tolerance=1e-11, integrator='Bdf1', step_strategy='Fixed', error_abs_tol=1e-09, min_dt=dt, max_dt=dt, fixed_dt=dt)

def csc_bytes(matrix: sp.csc_matrix) -> int:
    matrix = matrix.tocsc()
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)

def reference(cfg: Package, run: Run, h: float) -> Reference:
    compile_started = time.perf_counter()
    steady_compiled = build_package(cfg, run, True, Study.STEADY, h).compile()
    transient_compiled = build_package(cfg, run, True, Study.TRANSIENT, h).compile()
    compile_s = time.perf_counter() - compile_started
    operators = transient_compiled.assemble()
    started = time.perf_counter()
    with steady_compiled.solve(opts=options(run, False)) as steady_solution:
        steady = np.asarray(steady_solution.temperature).copy()
    steady_s = time.perf_counter() - started
    started = time.perf_counter()
    with transient_compiled.solve(opts=options(run, True)) as transient_solution:
        times = np.asarray(transient_solution.history_times).copy()
        transient = np.asarray(transient_solution.temperature_history).copy()
    transient_s = time.perf_counter() - started
    steady_compiled.close()
    transient_compiled.close()
    return Reference(steady, times, transient, compile_s, steady_s, transient_s, operators.K.shape[0], operators.K.nnz, operators.C.nnz, csc_bytes(operators.K) + csc_bytes(operators.C))

def evaluate(data: Data, cfg: Package, run: Run, W: sp.csc_matrix, h: float, ref: Reference):
    sample = next((x for x in data.samples if x.h == h))
    reduced = project(sample, W)
    p = cfg.ports
    internal0 = np.asarray(W.T @ np.full(W.shape[0], cfg.ambient_K)).ravel()

    def run_solve(transient: bool):
        compiled = data.detail_transient if transient else data.detail_steady
        ports_ = data.detail_ports_transient if transient else data.detail_ports_steady
        state = np.r_[np.full(compiled.cell_count, cfg.ambient_K), np.full(p, cfg.ambient_K), internal0]
        model = DtNModel((reduced.K, reduced.C, reduced.f))
        started = time.perf_counter()
        with solve_macro(compiled, model, ports_, state, options(run, transient)) as solution:
            elapsed = time.perf_counter() - started
            if transient:
                return (np.asarray(solution.history_times).copy(), np.asarray(solution.state_history).copy(), elapsed)
            return (np.asarray(solution.state).copy(), elapsed)
    steady_state, steady_s = run_solve(False)
    times, transient_states, transient_s = run_solve(True)
    detail_n = data.detail_steady.cell_count

    def recover(states):
        states = np.atleast_2d(states)
        macro = states[:, detail_n:]
        out = np.empty((states.shape[0], data.full_layout.cell_count))
        out[:, data.detail_cells] = states[:, :detail_n]
        out[:, data.macro_cells] = (W @ macro[:, p:].T).T
        return out
    steady_error = float(np.max(np.abs(recover(steady_state)[0] - ref.steady)))
    if times.shape != ref.times.shape or not np.allclose(times, ref.times, atol=1e-12, rtol=0):
        raise RuntimeError('full and reduced solvers returned different output times')
    transient_error = float(np.max(np.abs(recover(transient_states) - ref.transient)))
    reduced_bytes = csc_bytes(reduced.K) + csc_bytes(reduced.C)
    return {'h_W_m2K': h, 'steady_error_K': steady_error, 'transient_error_K': transient_error, 'transient_records': int(times.size), 'projection_s': reduced.projection_s, 'full_compile_s': ref.compile_s, 'full_steady_solve_s': ref.steady_solve_s, 'reduced_steady_solve_s': steady_s, 'steady_speedup': ref.steady_solve_s / max(steady_s, np.finfo(float).tiny), 'full_transient_solve_s': ref.transient_solve_s, 'reduced_transient_solve_s': transient_s, 'transient_speedup': ref.transient_solve_s / max(transient_s, np.finfo(float).tiny), 'full_operator_order': ref.operator_order, 'full_operator_k_nnz': ref.operator_k_nnz, 'full_operator_c_nnz': ref.operator_c_nnz, 'full_operator_bytes': ref.operator_bytes, 'reduced_macro_order': int(reduced.K.shape[0]), 'reduced_online_order': int(detail_n + reduced.K.shape[0]), 'reduced_macro_k_nnz': int(reduced.K.nnz), 'reduced_macro_c_nnz': int(reduced.C.nnz), 'reduced_macro_operator_bytes': reduced_bytes}
