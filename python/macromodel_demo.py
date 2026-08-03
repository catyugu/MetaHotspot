#!/usr/bin/env python3
"""Extract and validate an adaptive transient BCI thermal macromodel.

The substrate/bump/die domain remains full order.  The TIM/spreader/cold-plate
component is reduced while every interface temperature remains an exact physical
port.  Uniform cold-plate convection is represented affinely, and one global
basis is selected by a residual-greedy parametric tangential rational Krylov
procedure over the requested convection and time-scale ranges.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
from functools import cached_property
from pathlib import Path

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import metahotspot
from metahotspot.compiled import Operators, SolveOptions
from metahotspot.enums import Axis, Face, GeometryOp, LengthUnit, Study
from metahotspot.macromodel import PortMap, PortPatch, solve as solve_macro


POWER_MAP = np.asarray(
    (
        (0.10, 0.15, 0.20, 0.15),
        (0.15, 0.50, 1.20, 0.20),
        (0.10, 0.80, 8.55, 0.25),
        (0.10, 0.20, 1.20, 0.45),
    ),
    dtype=np.float64,
)
POWER_MAP /= POWER_MAP.mean()
CHIPLET_POWER_SCALE = (1.00, 0.72, 1.25, 0.55)
MATERIALS = (
    ("organic", ".65", ".65", ".55", "1900", "1100"),
    ("underfill", ".8", ".8", ".8", "1550", "1000"),
    ("copper", "390", "390", "390", "8960", "385"),
    ("silicon", "130", "130", "115", "2330", "700"),
    ("tim", "4", "4", "3", "2500", "900"),
    ("aluminum", "180", "180", "180", "2700", "900"),
)
ACTIVITY_TRACES = (
    (
        (0.00, 0.20),
        (0.10, 1.00),
        (0.35, 0.65),
        (0.58, 1.30),
        (0.82, 0.40),
        (1.00, 0.90),
    ),
    (
        (0.00, 0.75),
        (0.18, 1.20),
        (0.40, 0.30),
        (0.64, 1.05),
        (0.88, 0.55),
        (1.00, 0.80),
    ),
    (
        (0.00, 0.10),
        (0.08, 1.45),
        (0.28, 0.50),
        (0.52, 1.15),
        (0.76, 0.25),
        (1.00, 1.00),
    ),
    (
        (0.00, 0.55),
        (0.22, 0.35),
        (0.44, 1.25),
        (0.70, 0.60),
        (0.90, 1.10),
        (1.00, 0.70),
    ),
)
BOUNDARIES = (
    ("uniform-low", 500.0),
    ("uniform-medium", 2500.0),
    ("uniform-high", 8000.0),
)


@dataclass(frozen=True)
class Config:
    ambient_K: float = 300.0
    cold_plate_size_mm: float = 60.0
    spreader_size_mm: float = 50.0
    substrate_size_mm: float = 50.0
    bump_region_size_mm: float = 36.0
    die_size_mm: float = 32.0
    tim_size_mm: float = 32.0
    substrate_mm: float = 1.2
    bump_mm: float = 0.24
    die_mm: float = 0.60
    tim_mm: float = 0.18
    spreader_mm: float = 1.2
    cold_plate_mm: float = 1.5
    substrate_cells: int = 6
    bump_cells: int = 2
    die_cells: int = 4
    tim_cells: int = 2
    spreader_cells: int = 6
    cold_plate_cells: int = 8
    max_xy_cell_mm: float = 4.0
    bump_rows: int = 12
    bump_columns: int = 12
    bump_width_mm: float = 0.60
    chiplet_size_mm: float = 12.0
    chiplet_power_W: float = 25.0
    error_K: float = 0.25
    duration_s: float = 0.5
    dt_s: float = 0.025
    affine_anchor_h: float = 2500.0
    krylov_parameter_samples: int = 3
    krylov_frequency_samples: int = 6
    krylov_residual_tolerance: float = 2.0e-3
    krylov_block_size: int = 16
    krylov_max_order: int = 512
    speedup_target: float = 1.5
    compression_target: float = 2.5
    report: Path = Path("results/bci_rom_parametric_krylov_results.json")

    @property
    def detail_layers(self):
        return (
            (self.substrate_mm, self.substrate_cells),
            (self.bump_mm, self.bump_cells),
            (self.die_mm, self.die_cells),
        )

    @property
    def macro_layers(self):
        return (
            (self.tim_mm, self.tim_cells),
            (self.spreader_mm, self.spreader_cells),
            (self.cold_plate_mm, self.cold_plate_cells),
        )

    @property
    def detail_nz(self) -> int:
        return sum(cells for _, cells in self.detail_layers)

    @property
    def nz(self) -> int:
        return self.detail_nz + sum(cells for _, cells in self.macro_layers)

    @property
    def detail_height_mm(self) -> float:
        return sum(thickness for thickness, _ in self.detail_layers)

    @property
    def macro_height_mm(self) -> float:
        return sum(thickness for thickness, _ in self.macro_layers)

    @property
    def total_height_mm(self) -> float:
        return self.detail_height_mm + self.macro_height_mm

    @property
    def chiplet_origins_mm(self) -> tuple[tuple[float, float], ...]:
        low = -self.die_size_mm / 2.0 + 2.0
        high = self.die_size_mm / 2.0 - 2.0 - self.chiplet_size_mm
        return ((low, low), (high, low), (low, high), (high, high))

    @cached_property
    def axis_vertices_mm(self) -> np.ndarray:
        half_sizes = (
            self.cold_plate_size_mm / 2.0,
            self.spreader_size_mm / 2.0,
            self.bump_region_size_mm / 2.0,
            self.die_size_mm / 2.0,
        )
        points = [-value for value in half_sizes]
        points.extend((0.0, *reversed(half_sizes)))
        tile = self.chiplet_size_mm / 4.0
        for origin, _ in self.chiplet_origins_mm:
            points.extend(origin + tile * np.arange(5, dtype=np.float64))

        unique = np.unique(np.asarray(points, dtype=np.float64))
        vertices = [float(unique[0])]
        for left, right in zip(unique[:-1], unique[1:]):
            pieces = max(1, math.ceil((right - left) / self.max_xy_cell_mm))
            vertices.extend(np.linspace(left, right, pieces + 1)[1:])
        return np.asarray(vertices)

    @cached_property
    def port_indices(self) -> np.ndarray:
        vertices = self.axis_vertices_mm
        half = self.tim_size_mm / 2.0
        tolerance = 1.0e-10 * max(1.0, self.tim_size_mm)
        return np.flatnonzero(
            (vertices[:-1] >= -half - tolerance) & (vertices[1:] <= half + tolerance)
        ).astype(np.int64)

    @property
    def nx(self) -> int:
        return self.axis_vertices_mm.size - 1

    @property
    def ports(self) -> int:
        return self.port_indices.size**2

    @property
    def nominal_power_W(self) -> float:
        return self.chiplet_power_W * float(sum(CHIPLET_POWER_SCALE))


QUICK_OVERRIDES = dict(
    substrate_cells=3,
    bump_cells=1,
    die_cells=2,
    tim_cells=1,
    spreader_cells=3,
    cold_plate_cells=4,
    max_xy_cell_mm=6.0,
    bump_rows=8,
    bump_columns=8,
    error_K=0.35,
    duration_s=0.20,
    krylov_frequency_samples=4,
    krylov_residual_tolerance=1.0e-2,
    krylov_block_size=24,
    krylov_max_order=384,
    speedup_target=1.0,
    compression_target=2.0,
)


def z_vertices(layers) -> np.ndarray:
    vertices = [0.0]
    z = 0.0
    for thickness, cells in layers:
        for _ in range(cells):
            z += thickness / cells
            vertices.append(z)
    return np.asarray(vertices)


def add_square(model, block: int, size_mm: float) -> None:
    half = size_mm / 2.0
    model.add_rect(
        block,
        GeometryOp.ADD,
        f"{-half:.17g}",
        f"{-half:.17g}",
        f"{size_mm:.17g}",
        f"{size_mm:.17g}",
    )


def build_model(
    cfg: Config,
    study: Study,
    *,
    detail: bool,
    macro: bool,
    convection_h: float | None = None,
):
    """Build a full, detail-only, or macro-only model on one shared mesh."""
    if not detail and not macro:
        raise ValueError("at least one domain must be enabled")
    if convection_h is not None and convection_h < 0.0:
        raise ValueError("convection coefficient must be non-negative")

    model = metahotspot.Model()
    layers = (
        (*cfg.detail_layers, *cfg.macro_layers)
        if detail and macro
        else (cfg.detail_layers if detail else cfg.macro_layers)
    )
    transient = study == Study.TRANSIENT
    model.set_settings(
        study=study,
        length_unit=LengthUnit.MILLIMETER,
        initial_temperature_K=cfg.ambient_K,
        duration=cfg.duration_s if transient else 0.0,
        output_interval=cfg.dt_s if transient else 0.0,
    )
    model.set_mesh(cfg.axis_vertices_mm, cfg.axis_vertices_mm, z_vertices(layers))
    for material in MATERIALS:
        model.add_material(*material)

    if macro:
        for thickness, material, size in (
            (cfg.cold_plate_mm, "aluminum", cfg.cold_plate_size_mm),
            (cfg.spreader_mm, "copper", cfg.spreader_size_mm),
            (cfg.tim_mm, "tim", cfg.tim_size_mm),
        ):
            layer = model.add_layer(str(thickness))
            add_square(model, model.add_block(layer, material), size)

    if detail:
        die = model.add_layer(str(cfg.die_mm))
        add_square(model, model.add_block(die, "silicon"), cfg.die_size_mm)
        if transient:
            for index, trace in enumerate(ACTIVITY_TRACES):
                model.add_function_piecewise(
                    f"activity_{index}",
                    np.asarray(
                        [
                            (fraction * cfg.duration_s, value)
                            for fraction, value in trace
                        ]
                    ),
                )

        tile = cfg.chiplet_size_mm / 4.0
        tile_volume_m3 = tile * tile * cfg.die_mm * 1.0e-9
        for chiplet, ((x0, y0), scale) in enumerate(
            zip(cfg.chiplet_origins_mm, CHIPLET_POWER_SCALE)
        ):
            for iy in range(4):
                for ix in range(4):
                    tile_power = (
                        cfg.chiplet_power_W * scale * POWER_MAP[iy, ix] / POWER_MAP.size
                    )
                    source = f"{tile_power / tile_volume_m3:.17g}"
                    if transient:
                        source += f"*activity_{(chiplet + 2 * ix + iy) % 4}(x)"
                    block = model.add_block(die, "silicon", heat_source=source)
                    model.add_rect(
                        block,
                        GeometryOp.ADD,
                        f"{x0 + ix * tile:.17g}",
                        f"{y0 + iy * tile:.17g}",
                        f"{tile:.17g}",
                        f"{tile:.17g}",
                    )

        bump = model.add_layer(str(cfg.bump_mm))
        add_square(
            model,
            model.add_block(bump, "underfill"),
            cfg.bump_region_size_mm,
        )
        pitch_x = cfg.die_size_mm / cfg.bump_columns
        pitch_y = cfg.die_size_mm / cfg.bump_rows
        origin = -cfg.die_size_mm / 2.0
        for iy in range(cfg.bump_rows):
            for ix in range(cfg.bump_columns):
                x = origin + (ix + 0.5) * pitch_x - cfg.bump_width_mm / 2.0
                y = origin + (iy + 0.5) * pitch_y - cfg.bump_width_mm / 2.0
                block = model.add_block(bump, "copper")
                model.add_rect(
                    block,
                    GeometryOp.ADD,
                    f"{x:.17g}",
                    f"{y:.17g}",
                    f"{cfg.bump_width_mm:.17g}",
                    f"{cfg.bump_width_mm:.17g}",
                )

        substrate = model.add_layer(str(cfg.substrate_mm))
        add_square(
            model,
            model.add_block(substrate, "organic"),
            cfg.substrate_size_mm,
        )

    model.set_default_neumann("0")
    if macro and convection_h:
        half = cfg.cold_plate_size_mm / 2.0
        top_z = cfg.total_height_mm if detail else cfg.macro_height_mm
        model.add_convection(
            str(float(convection_h)),
            str(cfg.ambient_K),
            [(Axis.Z, top_z, -half, half, -half, half)],
        )
    return model


def port_patches(cfg: Config, face: Face, z_m: float) -> list[PortPatch]:
    vertices = cfg.axis_vertices_mm * 1.0e-3
    return [
        PortPatch(
            int(face),
            z_m,
            (vertices[ix], vertices[ix + 1], vertices[iy], vertices[iy + 1]),
        )
        for ix in cfg.port_indices
        for iy in cfg.port_indices
    ]


def normalized_operators(K, C, f) -> Operators:
    K = sp.csc_matrix(K)
    C = sp.csc_matrix(C)
    K.eliminate_zeros()
    C.eliminate_zeros()
    return Operators(K, C, np.asarray(f, dtype=np.float64).copy())


def affine_operators(base: Operators, delta: Operators, alpha: float) -> Operators:
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("normalized convection coordinate must be non-negative")
    return normalized_operators(
        base.K + alpha * delta.K,
        base.C + alpha * delta.C,
        np.asarray(base.f) + alpha * delta.f,
    )


def grid_cells(compiled) -> np.ndarray:
    return compiled.grid_to_cell.reshape(compiled.nx, compiled.ny, compiled.nz)


def coordinate_map(source, target, z_offset: int, label: str) -> np.ndarray:
    if source.nx != target.nx or source.ny != target.ny:
        raise RuntimeError(f"{label}: lateral meshes differ")
    source_grid = grid_cells(source)
    target_grid = grid_cells(target)[:, :, z_offset : z_offset + source.nz]
    if target_grid.shape != source_grid.shape:
        raise RuntimeError(f"{label}: z range differs")
    valid = source_grid >= 0
    if not np.array_equal(valid, target_grid >= 0):
        raise RuntimeError(f"{label}: geometry occupancy differs")

    source_ids = source_grid[valid]
    target_ids = target_grid[valid]
    if (
        source_ids.size != source.cell_count
        or np.unique(source_ids).size != source.cell_count
    ):
        raise RuntimeError(f"{label}: source cell IDs are incomplete")
    mapping = np.empty(source.cell_count, dtype=np.int64)
    mapping[source_ids] = target_ids
    if np.unique(mapping).size != mapping.size:
        raise RuntimeError(f"{label}: target mapping is not one-to-one")
    return mapping


def symmetric_dense(matrix) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    return 0.5 * (matrix + matrix.T)


def eigenpairs_descending(matrix) -> tuple[np.ndarray, np.ndarray]:
    """Return the non-negative eigenpairs of a symmetric Gram matrix."""
    values, vectors = scipy.linalg.eigh(
        symmetric_dense(matrix),
        check_finite=False,
    )
    order = np.argsort(values)[::-1]
    return np.maximum(values[order], 0.0), vectors[:, order]


def training_points(cfg: Config, boundaries):
    h_min = min(float(h) for _, h in boundaries)
    h_max = max(float(h) for _, h in boundaries)
    h_values = np.geomspace(h_min, h_max, cfg.krylov_parameter_samples)
    h_values = np.unique(np.r_[h_values, cfg.affine_anchor_h])

    low = 1.0 / cfg.duration_s
    bdf1 = 1.0 / cfg.dt_s
    high = 2.0 / cfg.dt_s
    interior_count = max(0, cfg.krylov_frequency_samples - 4)
    interior = (
        np.geomspace(low, high, interior_count + 2)[1:-1]
        if interior_count
        else np.empty(0)
    )
    shifts = np.unique(np.r_[0.0, low, interior, bdf1, high])
    return h_values, shifts


def internal_blocks(operators: Operators, ports: int):
    return (
        operators.K[ports:, ports:].tocsc(),
        operators.C[ports:, ports:].tocsc(),
        operators.K[ports:, :ports].tocsc(),
        operators.C[ports:, :ports].tocsc(),
    )


def orthonormalize_block(basis: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Remove the current space and return an orthonormal independent block."""
    block = np.asarray(vectors, dtype=np.float64).copy()
    for _ in range(2):
        if basis.shape[1]:
            block -= basis @ (basis.T @ block)
    q, r, _ = scipy.linalg.qr(
        block,
        mode="economic",
        pivoting=True,
        check_finite=False,
    )
    diagonal = np.abs(np.diag(r))
    if not diagonal.size or diagonal[0] == 0.0:
        return np.empty((block.shape[0], 0), dtype=np.float64)
    keep = diagonal > np.finfo(float).eps * max(block.shape) * diagonal[0]
    return np.ascontiguousarray(q[:, keep])


def build_krylov_basis(
    cfg: Config,
    boundaries,
    base: Operators,
    delta: Operators,
    ports: int,
):
    """Build a global basis by block residual-greedy rational interpolation.

    At each training pair (s, h), the exact internal response block is

        X(s, h) = -(K_ii(h) + s C_ii(h))^-1
                    (K_ip(h) + s C_ip(h)).

    Errors are normalized separately at every training point.  The worst point
    contributes several dominant tangential error directions per enrichment,
    rather than one direction at a time.
    """
    started = time.perf_counter()
    K0, C0, B0, D0 = internal_blocks(base, ports)
    K1, C1, B1, D1 = internal_blocks(delta, ports)
    h_values, shifts = training_points(cfg, boundaries)
    candidates = []

    for h_value in h_values:
        mu = float(h_value / cfg.affine_anchor_h)
        for shift in shifts:
            A = (K0 + mu * K1 + shift * (C0 + mu * C1)).tocsc()
            A = (0.5 * (A + A.T)).tocsc()
            B = (B0 + mu * B1 + shift * (D0 + mu * D1)).tocsc()
            response = np.asarray(spla.splu(A).solve(-B.toarray()))
            gram = symmetric_dense(response.T @ (A @ response))
            reference_values, _ = eigenpairs_descending(gram)
            candidates.append(
                {
                    "h_W_m2K": float(h_value),
                    "shift_per_s": float(shift),
                    "A": A,
                    "B": B,
                    "response": response,
                    "reference_eigenvalue": max(
                        float(reference_values[0]), np.finfo(float).tiny
                    ),
                }
            )

    internal_order = K0.shape[0]
    max_order = min(cfg.krylov_max_order, internal_order)
    basis = np.empty((internal_order, 0), dtype=np.float64)
    history = []
    converged = False

    while True:
        best = None
        for candidate in candidates:
            if basis.shape[1]:
                reduced_A = symmetric_dense(basis.T @ (candidate["A"] @ basis))
                reduced_B = basis.T @ candidate["B"]
                reduced_response = scipy.linalg.solve(
                    reduced_A,
                    -reduced_B,
                    assume_a="sym",
                    check_finite=False,
                )
                error_response = candidate["response"] - basis @ reduced_response
            else:
                error_response = candidate["response"]

            error_gram = symmetric_dense(
                error_response.T @ (candidate["A"] @ error_response)
            )
            error_values, tangents = eigenpairs_descending(error_gram)
            score = math.sqrt(
                float(error_values[0]) / candidate["reference_eigenvalue"]
            )
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "candidate": candidate,
                    "error_response": error_response,
                    "error_values": error_values,
                    "tangents": tangents,
                }

        entry = {
            "order": basis.shape[1],
            "relative_response_error": float(best["score"]),
            "h_W_m2K": best["candidate"]["h_W_m2K"],
            "shift_per_s": best["candidate"]["shift_per_s"],
            "added_directions": 0,
        }
        history.append(entry)
        if best["score"] <= cfg.krylov_residual_tolerance:
            converged = True
            break
        if basis.shape[1] >= max_order:
            break

        relative_directions = np.sqrt(
            best["error_values"] / best["candidate"]["reference_eigenvalue"]
        )
        requested = int(
            np.count_nonzero(relative_directions > cfg.krylov_residual_tolerance)
        )
        count = min(
            max(1, requested),
            cfg.krylov_block_size,
            max_order - basis.shape[1],
        )
        vectors = best["error_response"] @ best["tangents"][:, :count]
        block = orthonormalize_block(basis, vectors)
        if not block.shape[1]:
            raise RuntimeError("rational Krylov block enrichment stalled")
        remaining = max_order - basis.shape[1]
        block = block[:, :remaining]
        basis = np.column_stack((basis, block))
        entry["added_directions"] = int(block.shape[1])

    orthogonality_error = np.linalg.norm(
        basis.T @ basis - np.eye(basis.shape[1]), ord=2
    )
    if orthogonality_error > 1.0e-10:
        raise RuntimeError("rational Krylov basis lost orthogonality")

    summary = {
        "parameter_samples_W_m2K": h_values.tolist(),
        "frequency_shifts_per_s": shifts.tolist(),
        "candidate_count": len(candidates),
        "full_port_tangential_search": True,
        "error_normalization": "relative at each parameter-frequency point",
        "block_size": cfg.krylov_block_size,
        "basis_order": basis.shape[1],
        "maximum_order": max_order,
        "orthogonality_error": orthogonality_error,
        "relative_response_error": history[-1]["relative_response_error"],
        "residual_tolerance": cfg.krylov_residual_tolerance,
        "converged": converged,
        "history": history,
        "seconds": time.perf_counter() - started,
    }
    return basis, summary


def project_operators(
    operators: Operators,
    ports: int,
    basis: np.ndarray,
    ambient_K: float,
) -> Operators:
    """Project internal temperature rise while retaining absolute port states."""
    internal_offset = np.full(operators.K.shape[0] - ports, ambient_K)
    shifted_f = np.asarray(
        operators.f - operators.K[:, ports:] @ internal_offset
    ).ravel()

    def project_matrix(matrix):
        reduced = sp.bmat(
            (
                (
                    matrix[:ports, :ports].tocsc(),
                    sp.csc_matrix(matrix[:ports, ports:] @ basis),
                ),
                (
                    sp.csc_matrix(basis.T @ matrix[ports:, :ports]),
                    sp.csc_matrix(basis.T @ matrix[ports:, ports:] @ basis),
                ),
            ),
            format="csc",
        )
        reduced = (0.5 * (reduced + reduced.T)).tocsc()
        reduced.eliminate_zeros()
        return reduced

    return Operators(
        project_matrix(operators.K),
        project_matrix(operators.C),
        np.r_[shifted_f[:ports], basis.T @ shifted_f[ports:]],
    )


def verify_ambient_balance(
    operators: Operators,
    ports: int,
    reduced_order: int,
    ambient_K: float,
    label: str,
) -> None:
    state = np.r_[np.full(ports, ambient_K), np.zeros(reduced_order)]
    defect = np.asarray(operators.K @ state - operators.f).ravel()
    scale = max(
        np.linalg.norm(operators.K @ state),
        np.linalg.norm(operators.f),
        1.0,
    )
    if np.linalg.norm(defect) > 1.0e-10 * scale:
        raise RuntimeError(f"{label} reduced operator violates ambient balance")


def solve_options(cfg: Config, transient: bool) -> SolveOptions:
    dt = cfg.dt_s if transient else 1.0
    return SolveOptions(
        linear_solver="EigenSparseLU",
        linear_tolerance=1.0e-12,
        linear_max_iterations=5000,
        nonlinear_max_iterations=30,
        nonlinear_relative_tolerance=1.0e-11,
        nonlinear_absolute_tolerance=1.0e-11,
        integrator="Bdf1",
        step_strategy="Fixed",
        error_abs_tol=1.0e-9,
        min_dt=dt,
        max_dt=dt,
        fixed_dt=dt,
    )


def full_reference(cfg: Config, convection_h: float):
    started = time.perf_counter()
    with ExitStack() as stack:
        steady = build_model(
            cfg,
            Study.STEADY,
            detail=True,
            macro=True,
            convection_h=convection_h,
        ).compile()
        stack.callback(steady.close)
        transient = build_model(
            cfg,
            Study.TRANSIENT,
            detail=True,
            macro=True,
            convection_h=convection_h,
        ).compile()
        stack.callback(transient.close)
        compile_s = time.perf_counter() - started

        started = time.perf_counter()
        with steady.solve(opts=solve_options(cfg, False)) as solution:
            steady_temperature = np.asarray(solution.temperature).copy()
        steady_s = time.perf_counter() - started

        started = time.perf_counter()
        with transient.solve(opts=solve_options(cfg, True)) as solution:
            times = np.asarray(solution.history_times).copy()
            history = np.asarray(solution.temperature_history).copy()
        transient_s = time.perf_counter() - started

        return (
            steady_temperature,
            times,
            history,
            compile_s,
            steady_s,
            transient_s,
            transient.cell_count,
        )


def run_experiment(cfg: Config, boundaries, strict: bool) -> dict:
    offline_started = time.perf_counter()
    with ExitStack() as stack:
        full_layout = build_model(cfg, Study.STEADY, detail=True, macro=True).compile()
        stack.callback(full_layout.close)
        detail_steady = build_model(
            cfg, Study.STEADY, detail=True, macro=False
        ).compile()
        stack.callback(detail_steady.close)
        detail_transient = build_model(
            cfg, Study.TRANSIENT, detail=True, macro=False
        ).compile()
        stack.callback(detail_transient.close)

        detail_patches = port_patches(cfg, Face.ZP, cfg.detail_height_mm * 1.0e-3)
        detail_ports_steady = PortMap(detail_steady, detail_patches)
        stack.callback(detail_ports_steady.close)
        detail_ports_transient = PortMap(detail_transient, detail_patches)
        stack.callback(detail_ports_transient.close)

        macro_started = time.perf_counter()
        macro_compiled = build_model(
            cfg, Study.STEADY, detail=False, macro=True
        ).compile()
        stack.callback(macro_compiled.close)
        macro_ports = PortMap(macro_compiled, port_patches(cfg, Face.ZM, 0.0))
        stack.callback(macro_ports.close)
        base = normalized_operators(*macro_ports.assemble())

        anchor_compiled = build_model(
            cfg,
            Study.STEADY,
            detail=False,
            macro=True,
            convection_h=cfg.affine_anchor_h,
        ).compile()
        stack.callback(anchor_compiled.close)
        anchor_ports = PortMap(anchor_compiled, port_patches(cfg, Face.ZM, 0.0))
        stack.callback(anchor_ports.close)
        anchor = normalized_operators(*anchor_ports.assemble())
        if anchor.K.shape != base.K.shape:
            raise RuntimeError("convection changed macro state ordering")
        delta = normalized_operators(
            anchor.K - base.K,
            anchor.C - base.C,
            np.asarray(anchor.f) - base.f,
        )
        macro_extraction_s = time.perf_counter() - macro_started

        ambient = np.full(base.K.shape[0], cfg.ambient_K)
        balance_error = np.linalg.norm(delta.K @ ambient - delta.f)
        balance_scale = max(np.linalg.norm(delta.f), np.finfo(float).tiny)
        if spla.norm(delta.C) > 1.0e-11 * max(spla.norm(base.C), 1.0):
            raise RuntimeError("convection unexpectedly changed macro capacitance")
        if balance_error > 1.0e-10 * balance_scale:
            raise RuntimeError("affine convection component violates ambient balance")

        port_count = macro_ports.port_count
        if port_count != cfg.ports:
            raise RuntimeError("configured interface port count is inconsistent")

        detail_to_full = coordinate_map(detail_steady, full_layout, 0, "detail/full")
        transient_to_full = coordinate_map(
            detail_transient, full_layout, 0, "transient/full"
        )
        if not np.array_equal(detail_to_full, transient_to_full):
            raise RuntimeError("steady and transient detail orderings differ")
        macro_to_full = coordinate_map(
            macro_compiled, full_layout, cfg.detail_nz, "macro/full"
        )
        combined = np.r_[detail_to_full, macro_to_full]
        if (
            combined.size != full_layout.cell_count
            or np.unique(combined).size != combined.size
        ):
            raise RuntimeError("detail and macro maps do not partition the full model")

        basis, basis_summary = build_krylov_basis(
            cfg, boundaries, base, delta, port_count
        )
        if not basis_summary["converged"]:
            raise RuntimeError(
                "Krylov extraction did not converge: "
                f"order={basis_summary['basis_order']}, "
                "worst relative response error="
                f"{basis_summary['relative_response_error']:.3e}, "
                f"target={basis_summary['residual_tolerance']:.3e}"
            )
        projection_started = time.perf_counter()
        reduced_base = project_operators(base, port_count, basis, cfg.ambient_K)
        reduced_delta = project_operators(delta, port_count, basis, cfg.ambient_K)
        projection_s = time.perf_counter() - projection_started
        verify_ambient_balance(
            reduced_base,
            port_count,
            basis.shape[1],
            cfg.ambient_K,
            "base",
        )
        verify_ambient_balance(
            reduced_delta,
            port_count,
            basis.shape[1],
            cfg.ambient_K,
            "convection increment",
        )
        offline_s = time.perf_counter() - offline_started

        full_macro_order = port_count + basis.shape[0]
        reduced_macro_order = port_count + basis.shape[1]
        compression = full_macro_order / reduced_macro_order
        print(
            f"Grid {cfg.nx}x{cfg.nx}x{cfg.nz}; exact ports={port_count}; "
            f"macro states {full_macro_order:,}->{reduced_macro_order:,} "
            f"({compression:.2f}x); Krylov residual="
            f"{basis_summary['relative_response_error']:.3e}"
        )

        results = []
        detail_n = detail_steady.cell_count
        for name, convection_h in boundaries:
            reference = full_reference(cfg, convection_h)
            (
                reference_steady,
                reference_times,
                reference_history,
                full_compile_s,
                full_steady_s,
                full_transient_s,
                full_order,
            ) = reference

            assembly_started = time.perf_counter()
            reduced = affine_operators(
                reduced_base,
                reduced_delta,
                convection_h / cfg.affine_anchor_h,
            )
            online_assembly_s = time.perf_counter() - assembly_started

            def solve_reduced(transient: bool):
                compiled = detail_transient if transient else detail_steady
                ports = detail_ports_transient if transient else detail_ports_steady
                state = np.r_[
                    np.full(compiled.cell_count + port_count, cfg.ambient_K),
                    np.zeros(basis.shape[1]),
                ]
                started = time.perf_counter()
                with solve_macro(
                    compiled,
                    reduced,
                    ports,
                    state,
                    solve_options(cfg, transient),
                ) as solution:
                    elapsed = time.perf_counter() - started
                    if transient:
                        return (
                            np.asarray(solution.history_times).copy(),
                            np.asarray(solution.state_history).copy(),
                            elapsed,
                        )
                    return np.asarray(solution.state).copy(), elapsed

            steady_state, reduced_steady_s = solve_reduced(False)
            times, transient_states, reduced_transient_s = solve_reduced(True)
            if times.shape != reference_times.shape or not np.allclose(
                times, reference_times, atol=1.0e-12, rtol=0.0
            ):
                raise RuntimeError("full and reduced output times differ")

            def recover(states):
                states = np.atleast_2d(states)
                temperature = np.empty((states.shape[0], full_layout.cell_count))
                temperature[:, detail_to_full] = states[:, :detail_n]
                reduced_internal = states[:, detail_n + port_count :]
                temperature[:, macro_to_full] = (
                    cfg.ambient_K + (basis @ reduced_internal.T).T
                )
                return temperature

            steady_error = float(
                np.max(np.abs(recover(steady_state)[0] - reference_steady))
            )
            transient_error = float(
                np.max(np.abs(recover(transient_states) - reference_history))
            )
            speedup = full_transient_s / max(reduced_transient_s, np.finfo(float).tiny)
            accuracy_passed = max(steady_error, transient_error) <= cfg.error_K
            speedup_passed = speedup >= cfg.speedup_target if strict else True
            result = {
                "name": name,
                "h_W_m2K": convection_h,
                "steady_error_K": steady_error,
                "transient_error_K": transient_error,
                "online_reduced_assembly_s": online_assembly_s,
                "full_compile_s": full_compile_s,
                "full_steady_solve_s": full_steady_s,
                "reduced_steady_solve_s": reduced_steady_s,
                "full_transient_solve_s": full_transient_s,
                "reduced_transient_solve_s": reduced_transient_s,
                "transient_speedup": speedup,
                "full_order": full_order,
                "reduced_online_order": detail_n + reduced.K.shape[0],
                "reduced_macro_k_nnz": reduced.K.nnz,
                "reduced_macro_c_nnz": reduced.C.nnz,
                "accuracy_passed": accuracy_passed,
                "speedup_passed": speedup_passed,
                "passed": accuracy_passed and speedup_passed,
            }
            results.append(result)
            print(
                f"{name:>16s}: h={convection_h:g} W/(m^2 K), "
                f"error steady/transient={steady_error:.5f}/"
                f"{transient_error:.5f} K; full/ROM="
                f"{full_transient_s:.3f}/{reduced_transient_s:.3f}s, "
                f"speedup={speedup:.2f}x "
                f"{'PASS' if result['passed'] else 'FAIL'}"
            )

        compression_passed = compression >= cfg.compression_target
        basis_passed = basis_summary["converged"]
        return {
            "schema_version": 20,
            "method": (
                "exact-port affine-parametric adaptive tangential rational "
                "Krylov BCI-ROM"
            ),
            "configuration": {
                **asdict(cfg),
                "report": str(cfg.report),
                "nx": cfg.nx,
                "ny": cfg.nx,
                "nz": cfg.nz,
                "ports": cfg.ports,
                "port_shape": [
                    cfg.port_indices.size,
                    cfg.port_indices.size,
                ],
                "nominal_power_W": cfg.nominal_power_W,
                "power_map_normalized": POWER_MAP.tolist(),
                "chiplet_power_scale": list(CHIPLET_POWER_SCALE),
            },
            "affine_boundary": {
                "family": "A(h)=A0+(h/anchor_h)*DeltaA_h",
                "region": "entire cold-plate top surface",
                "anchor_h_W_m2K": cfg.affine_anchor_h,
                "full_order_offline_assemblies": 2,
                "full_order_online_assemblies_per_case": 0,
                "extraction_s": macro_extraction_s,
                "projection_s": projection_s,
            },
            "reduction": {
                "full_macro_order": full_macro_order,
                "reduced_macro_order": reduced_macro_order,
                "internal_full_order": basis.shape[0],
                "internal_reduced_order": basis.shape[1],
                "compression_ratio": compression,
                "compression_target": cfg.compression_target,
                "compression_passed": compression_passed,
                "basis_dense": True,
                "temperature_coordinates": (
                    "absolute physical port temperatures and internal "
                    "temperature rise above ambient"
                ),
                "krylov": basis_summary,
                "basis_passed": basis_passed,
            },
            "passivity": {
                "preserved_structurally": True,
                "reason": "symmetric Galerkin congruence with exact ports",
            },
            "offline_s": offline_s,
            "boundary_reuse": results,
            "passed": bool(
                all(result["passed"] for result in results)
                and compression_passed
                and basis_passed
            ),
        }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true", help="small smoke experiment")
    mode.add_argument("--strict", action="store_true", help="full benchmark gates")
    args = parser.parse_args(argv)

    cfg = replace(Config(), **QUICK_OVERRIDES) if args.quick else Config()
    boundaries = (BOUNDARIES[0], BOUNDARIES[-1]) if args.quick else BOUNDARIES

    print("=" * 96)
    print("Transient BCI-ROM extraction - adaptive parametric rational Krylov")
    print("=" * 96)
    print(
        "Footprints cold plate/spreader/substrate/bump/die/TIM="
        f"{cfg.cold_plate_size_mm:g}/{cfg.spreader_size_mm:g}/"
        f"{cfg.substrate_size_mm:g}/{cfg.bump_region_size_mm:g}/"
        f"{cfg.die_size_mm:g}/{cfg.tim_size_mm:g} mm"
    )
    print(
        f"Nominal die power={cfg.nominal_power_W:.2f} W; "
        f"tile peak/mean density={POWER_MAP.max():.2f}x"
    )

    report = run_experiment(cfg, boundaries, args.strict)
    report["mode"] = "quick" if args.quick else "strict"
    cfg.report.parent.mkdir(parents=True, exist_ok=True)
    cfg.report.write_text(
        json.dumps(report, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"Report: {cfg.report}")
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
