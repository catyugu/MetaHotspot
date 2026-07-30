#!/usr/bin/env python3
"""
Nonlinear detailed FVM block coupled to an SVD-reduced macro port.

This experiment deliberately keeps the three responsibilities separate:

1. ``DetailedNonlinearBlock`` is a standalone MetaHotspot model.
2. ``CondensedMacroBlock`` is another standalone model. Eliminating its
   internal cells produces a physical port Dirichlet-to-Neumann operator.
   An SVD of the port compliance retains only a few online port modes.
3. The C++ modal-port assembler connects detailed boundary cells to the macro
   physical port. It updates the FVM-side half conductance and projects the
   interface to retained modal coordinates on every nonlinear linearization.

The full two-block MetaHotspot model is used only as a reference solution.
The macro reduction never sees or includes the detailed-region DoFs.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

import metahotspot
from metahotspot import enums


INITIAL_TEMPERATURE = 300.0
DETAIL_K0 = 15.0
DETAIL_K_SLOPE = 0.015
MACRO_K = 120.0
CELL_LENGTH_M = 1.0e-3
FACE_AREA_M2 = 1.0e-6
DETAIL_LENGTH_MM = 8.0
MACRO_LENGTH_MM = 24.0
DOMAIN_HEIGHT_MM = 24.0
DOMAIN_THICKNESS_MM = 2.0
FULL_LENGTH_MM = DETAIL_LENGTH_MM + MACRO_LENGTH_MM
HEAT_SOURCE = (
    "8e7*(1"
    " + 0.45*sin(261.7993877991494*y)"
    " + 0.20*cos(523.5987755982989*y)"
    " + 0.15*cos(1570.796326794897*z))"
)

FULL_DETAIL_BLOCK_ID = 0
FULL_MACRO_BLOCK_ID = 1
PORT_MODE_COUNTS = (1, 2, 4, 8, 12)


def add_materials(model: metahotspot.Model) -> None:
    model.add_material(
        "nonlinear",
        kx="15*(1 + 0.015*(T - 300))",
        ky="15*(1 + 0.015*(T - 300))",
        kz="15*(1 + 0.015*(T - 300))",
        rho="0",
        c="0",
    )
    model.add_material(
        "macro",
        kx="120",
        ky="120",
        kz="120",
        rho="0",
        c="0",
    )


def set_common_settings(model: metahotspot.Model, x_vertices: np.ndarray) -> None:
    model.set_settings(
        study=enums.Study.STEADY,
        length_unit=enums.LengthUnit.MILLIMETER,
        initial_temperature_K=INITIAL_TEMPERATURE,
    )
    model.set_mesh(
        x=x_vertices,
        y=np.arange(0.0, DOMAIN_HEIGHT_MM + 1.0),
        z=np.arange(0.0, DOMAIN_THICKNESS_MM + 1.0),
    )
    model.set_default_neumann("0")


def x_face(coordinate: float):
    return [
        (
            enums.Axis.X,
            coordinate,
            0.0,
            DOMAIN_HEIGHT_MM,
            0.0,
            DOMAIN_THICKNESS_MM,
        )
    ]


def build_full_reference_model() -> metahotspot.Model:
    model = metahotspot.Model()
    set_common_settings(model, np.arange(0.0, FULL_LENGTH_MM + 1.0))
    add_materials(model)

    layer = model.add_layer(thickness=str(DOMAIN_THICKNESS_MM))
    detail = model.add_block(layer, "nonlinear", heat_source=HEAT_SOURCE)
    macro = model.add_block(layer, "macro")
    assert detail == FULL_DETAIL_BLOCK_ID
    assert macro == FULL_MACRO_BLOCK_ID
    model.add_rect(
        detail,
        x="0",
        y="0",
        width=str(DETAIL_LENGTH_MM),
        height=str(DOMAIN_HEIGHT_MM),
    )
    model.add_rect(
        macro,
        x=str(DETAIL_LENGTH_MM),
        y="0",
        width=str(MACRO_LENGTH_MM),
        height=str(DOMAIN_HEIGHT_MM),
    )
    model.add_dirichlet("300", x_face(0.0))
    model.add_dirichlet("300", x_face(FULL_LENGTH_MM))
    return model


def build_detailed_nonlinear_model() -> metahotspot.Model:
    """Detailed block with an adiabatic placeholder at its X+ interface."""
    model = metahotspot.Model()
    set_common_settings(model, np.arange(0.0, DETAIL_LENGTH_MM + 1.0))
    add_materials(model)

    layer = model.add_layer(thickness=str(DOMAIN_THICKNESS_MM))
    detail = model.add_block(layer, "nonlinear", heat_source=HEAT_SOURCE)
    model.add_rect(
        detail,
        x="0",
        y="0",
        width=str(DETAIL_LENGTH_MM),
        height=str(DOMAIN_HEIGHT_MM),
    )
    model.add_dirichlet("300", x_face(0.0))
    return model


def build_macro_model() -> metahotspot.Model:
    """Macro block alone; its X- port is left adiabatic before coupling."""
    model = metahotspot.Model()
    set_common_settings(
        model,
        np.arange(DETAIL_LENGTH_MM, FULL_LENGTH_MM + 1.0),
    )
    add_materials(model)

    layer = model.add_layer(thickness=str(DOMAIN_THICKNESS_MM))
    macro = model.add_block(layer, "macro")
    model.add_rect(
        macro,
        x=str(DETAIL_LENGTH_MM),
        y="0",
        width=str(MACRO_LENGTH_MM),
        height=str(DOMAIN_HEIGHT_MM),
    )
    model.add_dirichlet("300", x_face(FULL_LENGTH_MM))
    return model


def boundary_cells(metadata, x_cell: int) -> np.ndarray:
    """Return active-cell indices on one X-normal cell patch."""
    grid = metadata.grid_to_cell.reshape(metadata.nx, metadata.ny, metadata.nz)
    cells = np.asarray(grid[x_cell, :, :]).ravel().astype(np.int64)
    if np.unique(cells).size != cells.size:
        raise RuntimeError("boundary patch contains invalid or repeated cells")
    return cells


def take(A: csc_matrix, rows: np.ndarray, cols: np.ndarray) -> csc_matrix:
    return A[rows, :][:, cols].tocsc()


@dataclass
class CondensedMacroBlock:
    """Port-only macro operator plus the data needed for field recovery."""

    port_cells: np.ndarray
    internal_cells: np.ndarray
    K_port: csc_matrix
    f_port: np.ndarray
    K_ip: csc_matrix
    K_ii_lu: object
    f_i: np.ndarray

    def recover(self, port_temperature: np.ndarray) -> np.ndarray:
        return self.K_ii_lu.solve(self.f_i - self.K_ip @ port_temperature)


@dataclass
class ModalMacroBlock:
    """Few-mode representation of the condensed physical macro port."""

    physical: CondensedMacroBlock
    basis: np.ndarray
    singular_values: np.ndarray
    K: csc_matrix
    f: np.ndarray

    @property
    def mode_count(self) -> int:
        return self.basis.shape[1]

    def port_temperature(self, modal_state: np.ndarray) -> np.ndarray:
        return self.basis @ modal_state

    def recover(self, modal_state: np.ndarray) -> np.ndarray:
        return self.physical.recover(self.port_temperature(modal_state))


@dataclass
class ReductionResult:
    mode_count: int
    online_dofs: int
    retained_energy: float
    relative_error: float
    max_error: float
    elapsed_seconds: float
    initial_interface_conductance: float
    final_interface_conductance: float


def condense_macro(compiled) -> CondensedMacroBlock:
    """Condense a standalone macro from ``p + i`` to ``p`` only."""
    metadata = compiled.metadata()
    port = boundary_cells(metadata, 0)
    all_cells = np.arange(metadata.cell_count, dtype=np.int64)
    internal = np.setdiff1d(all_cells, port)

    operators = compiled.assemble(compiled.default_state())
    K_pp = take(operators.K, port, port)
    K_pi = take(operators.K, port, internal)
    K_ip = take(operators.K, internal, port)
    K_ii = take(operators.K, internal, internal)
    K_ii_lu = splu(K_ii)

    inverse_Kip = np.column_stack(
        [
            K_ii_lu.solve(K_ip[:, column].toarray().ravel())
            for column in range(K_ip.shape[1])
        ]
    )
    K_port = csc_matrix(K_pp - K_pi @ inverse_Kip)
    f_i = operators.f[internal]
    f_port = operators.f[port] - K_pi @ K_ii_lu.solve(f_i)
    return CondensedMacroBlock(
        port_cells=port,
        internal_cells=internal,
        K_port=K_port,
        f_port=np.asarray(f_port).ravel(),
        K_ip=K_ip,
        K_ii_lu=K_ii_lu,
        f_i=f_i,
    )


def reduce_port_modes(
    macro: CondensedMacroBlock,
    mode_count: int,
) -> ModalMacroBlock:
    """Retain dominant SVD modes of the physical port compliance.

    The compliance maps port heat flow to port temperature, so its dominant
    left-singular vectors are the most responsive thermal port patterns.
    """
    port_count = macro.port_cells.size
    if mode_count < 1 or mode_count > port_count:
        raise ValueError("mode_count must be between 1 and the physical port count")

    port_lu = splu(macro.K_port.tocsc())
    compliance = port_lu.solve(np.eye(port_count))
    basis, singular_values, _ = np.linalg.svd(compliance, full_matrices=False)
    basis = basis[:, :mode_count]

    K_modal = csc_matrix(basis.T @ macro.K_port @ basis)
    f_modal = np.asarray(basis.T @ macro.f_port).ravel()
    return ModalMacroBlock(
        physical=macro,
        basis=basis,
        singular_values=singular_values,
        K=K_modal,
        f=f_modal,
    )


def total_interface_conductance(
    detailed_temperature: np.ndarray,
    detailed_cells: np.ndarray,
) -> float:
    """Evaluate the demo's physical series conductance for reporting."""
    k_detail = DETAIL_K0 * (
        1.0
        + DETAIL_K_SLOPE
        * (detailed_temperature[detailed_cells] - INITIAL_TEMPERATURE)
    )
    conductance = FACE_AREA_M2 / (
        0.5 * CELL_LENGTH_M / k_detail
        + 0.5 * CELL_LENGTH_M / MACRO_K
    )
    return float(conductance.sum())


def reference_solution() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Solve the monolithic model and return its detail/macro index maps."""
    started = perf_counter()
    with build_full_reference_model() as model:
        with model.compile() as compiled:
            options = metahotspot.SolverOpts.default()
            options.nonlinear_relative_tolerance = 1.0e-10
            with compiled.solve(opts=options) as solution:
                temperature = solution.temperature.copy()
            block_ids = compiled.metadata().block_ids.copy()
    elapsed = perf_counter() - started
    detail = np.flatnonzero(block_ids == FULL_DETAIL_BLOCK_ID)
    macro = np.flatnonzero(block_ids == FULL_MACRO_BLOCK_ID)
    return temperature, detail, macro, elapsed


def solve_reduced_case(
    detailed,
    detailed_interface: np.ndarray,
    physical_macro: CondensedMacroBlock,
    reference: np.ndarray,
    full_detail_cells: np.ndarray,
    full_macro_cells: np.ndarray,
    mode_count: int,
) -> ReductionResult:
    started = perf_counter()
    macro = reduce_port_modes(physical_macro, mode_count=mode_count)
    initial_modes = macro.basis.T @ np.full(
        physical_macro.port_cells.size,
        INITIAL_TEMPERATURE,
    )
    initial_state = np.concatenate([detailed.default_state(), initial_modes])
    macro_operators = metahotspot.Operators(
        K=macro.K,
        C=csc_matrix((macro.mode_count, macro.mode_count)),
        f=macro.f,
    )
    exterior_half_conductance = np.full(
        physical_macro.port_cells.size,
        MACRO_K * FACE_AREA_M2 / (0.5 * CELL_LENGTH_M),
    )
    options = metahotspot.SolverOpts.default()
    options.nonlinear_relative_tolerance = 1.0e-10
    with metahotspot.macromodel.solve(
        detailed,
        macro=macro_operators,
        basis=macro.basis,
        model_cells=detailed_interface,
        model_face=enums.Face.XP,
        exterior_half_conductance=exterior_half_conductance,
        state=initial_state,
        opts=options,
    ) as solution:
        reduced = solution.state.copy()

    detail_count = full_detail_cells.size
    macro_temperature = np.empty(full_macro_cells.size)
    macro_temperature[physical_macro.port_cells] = macro.port_temperature(
        reduced[detail_count:]
    )
    macro_temperature[physical_macro.internal_cells] = macro.recover(
        reduced[detail_count:]
    )

    recovered = np.empty_like(reference)
    recovered[full_detail_cells] = reduced[:detail_count]
    recovered[full_macro_cells] = macro_temperature
    difference = recovered - reference
    singular_energy = macro.singular_values**2
    return ReductionResult(
        mode_count=mode_count,
        online_dofs=detail_count + mode_count,
        retained_energy=float(
            np.sum(singular_energy[:mode_count]) / np.sum(singular_energy)
        ),
        relative_error=float(
            np.linalg.norm(difference) / np.linalg.norm(reference)
        ),
        max_error=float(np.max(np.abs(difference))),
        elapsed_seconds=perf_counter() - started,
        initial_interface_conductance=total_interface_conductance(
            initial_state[:detail_count],
            detailed_interface,
        ),
        final_interface_conductance=total_interface_conductance(
            reduced[:detail_count],
            detailed_interface,
        ),
    )


def main() -> int:
    (
        reference,
        full_detail_cells,
        full_macro_cells,
        reference_seconds,
    ) = reference_solution()

    with build_detailed_nonlinear_model() as detailed_model:
        with detailed_model.compile() as detailed:
            detailed_interface = boundary_cells(
                detailed.metadata(),
                detailed.metadata().nx - 1,
            )

            with build_macro_model() as macro_model:
                with macro_model.compile() as macro_compiled:
                    physical_macro = condense_macro(macro_compiled)
            results = [
                solve_reduced_case(
                    detailed,
                    detailed_interface,
                    physical_macro,
                    reference,
                    full_detail_cells,
                    full_macro_cells,
                    mode_count,
                )
                for mode_count in PORT_MODE_COUNTS
            ]

    detail_count = full_detail_cells.size
    print("=" * 86)
    print("Standalone nonlinear FVM + SVD port-modal macro experiment")
    print("=" * 86)
    print(
        f"Full problem: {reference.size} DoFs; detailed={detail_count}; "
        f"macro={full_macro_cells.size}"
    )
    print(
        f"Physical port={physical_macro.port_cells.size}; "
        f"macro internal={physical_macro.internal_cells.size}; "
        f"monolithic solve={reference_seconds:.3f}s"
    )
    print(
        "Total interface conductance: "
        f"{results[-1].initial_interface_conductance:.6e} -> "
        f"{results[-1].final_interface_conductance:.6e} W/K"
    )
    print()
    print(" modes | online/full | SVD energy | relative L2 | max error [K] | solve [s]")
    print("-" * 86)
    for result in results:
        print(
            f"{result.mode_count:6d} | "
            f"{result.online_dofs:4d}/{reference.size:<4d} | "
            f"{result.retained_energy:10.5%} | "
            f"{result.relative_error:11.3e} | "
            f"{result.max_error:13.3e} | "
            f"{result.elapsed_seconds:8.3f}"
        )
    print()
    print("Macro contract:")
    print("  state : retained modal coefficients q")
    print("  port  : physical temperature T_p = Phi @ q")
    print("  output: projected heat flow Phi.T @ (K_hat_p @ T_p - f_hat_p)")
    print("  solve : nonlinear iteration and FVM/interface reassembly run in C++")

    best = results[-1]
    if best.relative_error > 1.0e-3 or best.max_error > 1.0:
        print("FAIL: 12 retained modes are insufficient for this nonuniform case")
        return 1
    print("PASS: the mode sweep converges within the practical demo tolerance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
