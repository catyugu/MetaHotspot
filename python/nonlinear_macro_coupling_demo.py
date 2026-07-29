#!/usr/bin/env python3
"""
Nonlinear detailed FVM block coupled to a condensed linear macro block.

This experiment deliberately keeps the three responsibilities separate:

1. ``DetailedNonlinearBlock`` is a standalone MetaHotspot model.
2. ``CondensedMacroBlock`` is another standalone model. It contains only
   physical port cells ``p`` and macro-internal cells ``i``; eliminating ``i``
   produces a port-only Dirichlet-to-Neumann operator.
3. ``InterfaceCoupling`` connects the detailed boundary cells to the macro
   port cells. Its conductance changes during nonlinear iteration.

The full two-block MetaHotspot model is used only as a reference solution.
The macro condensation never sees or includes the detailed-region DoFs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import bmat, csc_matrix, coo_matrix, diags
from scipy.sparse.linalg import splu, spsolve

import metahotspot
from metahotspot import enums


INITIAL_TEMPERATURE = 300.0
DETAIL_K0 = 15.0
DETAIL_K_SLOPE = 0.015
MACRO_K = 120.0
CELL_LENGTH_M = 1.0e-3
FACE_AREA_M2 = 1.0e-6

FULL_DETAIL_BLOCK_ID = 0
FULL_MACRO_BLOCK_ID = 1


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
        y=np.linspace(0.0, 4.0, 5),
        z=np.linspace(0.0, 1.0, 2),
    )
    model.set_default_neumann("0")


def x_face(coordinate: float):
    return [(enums.Axis.X, coordinate, 0.0, 4.0, 0.0, 1.0)]


def build_full_reference_model() -> metahotspot.Model:
    model = metahotspot.Model()
    set_common_settings(model, np.linspace(0.0, 8.0, 9))
    add_materials(model)

    layer = model.add_layer(thickness="1")
    detail = model.add_block(layer, "nonlinear", heat_source="8e7")
    macro = model.add_block(layer, "macro")
    assert detail == FULL_DETAIL_BLOCK_ID
    assert macro == FULL_MACRO_BLOCK_ID
    model.add_rect(detail, x="0", y="0", width="4", height="4")
    model.add_rect(macro, x="4", y="0", width="4", height="4")
    model.add_dirichlet("300", x_face(0.0))
    model.add_dirichlet("300", x_face(8.0))
    return model


def build_detailed_nonlinear_model() -> metahotspot.Model:
    """Detailed block with an adiabatic placeholder at the X=4 interface."""
    model = metahotspot.Model()
    set_common_settings(model, np.linspace(0.0, 4.0, 5))
    add_materials(model)

    layer = model.add_layer(thickness="1")
    detail = model.add_block(layer, "nonlinear", heat_source="8e7")
    model.add_rect(detail, x="0", y="0", width="4", height="4")
    model.add_dirichlet("300", x_face(0.0))
    return model


def build_macro_model() -> metahotspot.Model:
    """Macro block alone; the X=4 port face is left adiabatic before coupling."""
    model = metahotspot.Model()
    set_common_settings(model, np.linspace(4.0, 8.0, 5))
    add_materials(model)

    layer = model.add_layer(thickness="1")
    macro = model.add_block(layer, "macro")
    model.add_rect(macro, x="4", y="0", width="4", height="4")
    model.add_dirichlet("300", x_face(8.0))
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


@dataclass
class InterfaceCoupling:
    detailed_cells: np.ndarray
    macro_port_count: int

    def conductivity(self, temperature: np.ndarray) -> np.ndarray:
        return DETAIL_K0 * (
            1.0 + DETAIL_K_SLOPE * (temperature - INITIAL_TEMPERATURE)
        )

    def conductance(
        self,
        detailed_temperature: np.ndarray,
        macro_port_temperature: np.ndarray,
    ) -> np.ndarray:
        k_detail = self.conductivity(
            detailed_temperature[self.detailed_cells]
        )
        k_macro = np.full_like(macro_port_temperature, MACRO_K)
        return FACE_AREA_M2 / (
            0.5 * CELL_LENGTH_M / k_detail
            + 0.5 * CELL_LENGTH_M / k_macro
        )

    def stiffness(
        self,
        detailed_temperature: np.ndarray,
        macro_port_temperature: np.ndarray,
    ) -> tuple[csc_matrix, np.ndarray]:
        """Return the four-block interface matrix and per-pair conductance."""
        nd = detailed_temperature.size
        np_ = self.macro_port_count
        if self.detailed_cells.size != np_:
            raise ValueError("one-to-one interface patch sizes do not match")

        g = self.conductance(detailed_temperature, macro_port_temperature)
        port_columns = np.arange(np_, dtype=np.int64)
        K_dp = coo_matrix(
            (-g, (self.detailed_cells, port_columns)),
            shape=(nd, np_),
        ).tocsc()
        K_pd = K_dp.transpose().tocsc()
        D_detail = coo_matrix(
            (g, (self.detailed_cells, self.detailed_cells)),
            shape=(nd, nd),
        ).tocsc()
        D_port = diags(g, format="csc")
        return (
            bmat(
                [[D_detail, K_dp], [K_pd, D_port]],
                format="csc",
            ),
            g,
        )


def condensed_nonlinear_solve(
    detailed_compiled,
    macro: CondensedMacroBlock,
    interface: InterfaceCoupling,
    max_iterations: int = 100,
    relative_tolerance: float = 1.0e-10,
):
    """Picard solve over ``[detailed Model DoFs, macro port DoFs]``."""
    nd = detailed_compiled.metadata().cell_count
    np_ = macro.port_cells.size
    state = np.full(nd + np_, INITIAL_TEMPERATURE, dtype=np.float64)
    conductance_history = []

    zero_dd = csc_matrix((nd, nd))
    zero_dp = csc_matrix((nd, np_))
    zero_pd = csc_matrix((np_, nd))
    K_macro_fixed = bmat(
        [[zero_dd, zero_dp], [zero_pd, macro.K_port]],
        format="csc",
    )

    for iteration in range(max_iterations):
        T_detail = state[:nd]
        T_port = state[nd:]
        detailed = detailed_compiled.assemble(T_detail)
        K_interface, pair_conductance = interface.stiffness(T_detail, T_port)

        K_model = bmat(
            [
                [detailed.K, zero_dp],
                [zero_pd, csc_matrix((np_, np_))],
            ],
            format="csc",
        )
        K = K_model + K_macro_fixed + K_interface
        f = np.concatenate([detailed.f, macro.f_port])
        conductance_history.append(float(pair_conductance.sum()))

        residual = f - K @ state
        residual_scale = max(np.linalg.norm(f, ord=np.inf), 1.0)
        if (
            iteration > 0
            and np.linalg.norm(residual, ord=np.inf)
            <= relative_tolerance * residual_scale
        ):
            return state, iteration, conductance_history

        next_state = np.asarray(spsolve(K, f))
        update = np.linalg.norm(next_state - state, ord=np.inf)
        state = next_state
        if update <= relative_tolerance * max(
            np.linalg.norm(state, ord=np.inf),
            1.0,
        ):
            return state, iteration + 1, conductance_history

    raise RuntimeError("condensed nonlinear Picard solve did not converge")


def reference_solution() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the monolithic model and return its detail/macro index maps."""
    with build_full_reference_model() as model:
        with model.compile() as compiled:
            options = metahotspot.SolverOpts.default()
            options.nonlinear_relative_tolerance = 1.0e-10
            with compiled.solve(opts=options) as solution:
                temperature = solution.temperature.copy()
            block_ids = compiled.metadata().block_ids.copy()
    detail = np.flatnonzero(block_ids == FULL_DETAIL_BLOCK_ID)
    macro = np.flatnonzero(block_ids == FULL_MACRO_BLOCK_ID)
    return temperature, detail, macro


def main() -> int:
    reference, full_detail_cells, full_macro_cells = reference_solution()

    with build_detailed_nonlinear_model() as detailed_model:
        with detailed_model.compile() as detailed:
            detailed_interface = boundary_cells(
                detailed.metadata(),
                detailed.metadata().nx - 1,
            )

            with build_macro_model() as macro_model:
                with macro_model.compile() as macro_compiled:
                    macro = condense_macro(macro_compiled)

            interface = InterfaceCoupling(
                detailed_cells=detailed_interface,
                macro_port_count=macro.port_cells.size,
            )
            reduced, iterations, conductance_history = (
                condensed_nonlinear_solve(detailed, macro, interface)
            )

    nd = full_detail_cells.size
    macro_temperature = np.empty(full_macro_cells.size)
    macro_temperature[macro.port_cells] = reduced[nd:]
    macro_temperature[macro.internal_cells] = macro.recover(reduced[nd:])

    recovered = np.empty_like(reference)
    recovered[full_detail_cells] = reduced[:nd]
    recovered[full_macro_cells] = macro_temperature

    difference = recovered - reference
    relative_error = np.linalg.norm(difference) / np.linalg.norm(reference)
    max_error = np.max(np.abs(difference))

    print("=" * 72)
    print("Standalone nonlinear FVM + port-only condensed macro experiment")
    print("=" * 72)
    print(
        f"DoFs: detailed={nd}, macro port={macro.port_cells.size}, "
        f"macro internal={macro.internal_cells.size}"
    )
    print(
        f"Online system: {nd + macro.port_cells.size} / {reference.size} DoFs"
    )
    print(f"Picard iterations: {iterations}")
    print(
        "Total interface conductance: "
        f"{conductance_history[0]:.6e} -> "
        f"{conductance_history[-1]:.6e} W/K"
    )
    print(f"Relative error vs monolithic nonlinear solve: {relative_error:.3e}")
    print(f"Maximum absolute error: {max_error:.3e} K")
    print()
    print("Macro contract:")
    print("  input : physical port temperature T_p")
    print("  output: port heat flow K_hat_p @ T_p - f_hat_p")
    print("  state : only retained macro-port DoFs are stored online")

    if relative_error > 1.0e-8 or max_error > 1.0e-5:
        print("FAIL: condensed solution does not match the monolithic reference")
        return 1
    print("PASS: port-only macro condensation matches the reference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
