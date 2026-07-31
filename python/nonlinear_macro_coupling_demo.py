#!/usr/bin/env python3
"""Nonlinear detailed FVM block coupled to an SVD-reduced DtN macro.

The detailed and macro blocks are compiled independently. C++ resolves the
geometric port patches, evaluates temperature-dependent half conductances,
assembles the coupled system, and runs the nonlinear solve. Python only
condenses the isolated macro DtN operator and constructs its reduced port map.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

import metahotspot
from metahotspot import enums
from metahotspot.macromodel import DtNModel, PortMap, PortPatch


INITIAL_TEMPERATURE = 300.0
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
Y_VERTICES_MM = np.arange(0.0, DOMAIN_HEIGHT_MM + 1.0)
Z_VERTICES_MM = np.arange(0.0, DOMAIN_THICKNESS_MM + 1.0)


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
    model.set_mesh(x=x_vertices, y=Y_VERTICES_MM, z=Z_VERTICES_MM)
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


def x_port_patches(face: enums.Face, coordinate_mm: float) -> list[PortPatch]:
    """Create one physical port per exposed Y-Z cell face."""
    scale = 1.0e-3
    return [
        PortPatch(
            face=int(face),
            coordinate=coordinate_mm * scale,
            rectangle=(y0 * scale, y1 * scale, z0 * scale, z1 * scale),
        )
        for y0, y1 in zip(Y_VERTICES_MM[:-1], Y_VERTICES_MM[1:], strict=True)
        for z0, z1 in zip(Z_VERTICES_MM[:-1], Z_VERTICES_MM[1:], strict=True)
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
    """Detailed block with an exposed X+ DtN coupling boundary."""
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
    """Macro block with an exposed X- DtN port and fixed far boundary."""
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


@dataclass
class CondensedMacroBlock:
    """Physical face DtN operator with recoverable macro cell temperatures."""

    face_count: int
    cell_count: int
    K_face: csc_matrix
    f_face: np.ndarray
    K_cell_face: csc_matrix
    K_cell_cell_lu: object
    f_cell: np.ndarray

    def recover(self, face_temperature: np.ndarray) -> np.ndarray:
        """Recover every macro FVM cell from physical face temperatures."""
        return self.K_cell_cell_lu.solve(
            self.f_cell - self.K_cell_face @ face_temperature
        )


@dataclass
class ModalMacroBlock:
    """Reduced representation of the physical face DtN operator."""

    physical: CondensedMacroBlock
    basis: np.ndarray
    retained_energy: float
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
    port_temperature_min: float
    port_temperature_max: float


def condense_macro(compiled, ports: PortMap) -> CondensedMacroBlock:
    """Condense the C++-assembled isolated macro DtN system to its face ports."""
    operators = ports.assemble()
    face_count = ports.port_count
    expected = face_count + compiled.cell_count
    if operators.K.shape != (expected, expected):
        raise RuntimeError("unexpected C++ DtN operator dimension")

    K = operators.K.tocsc()
    K_face_face = K[:face_count, :face_count].tocsc()
    K_face_cell = K[:face_count, face_count:].tocsc()
    K_cell_face = K[face_count:, :face_count].tocsc()
    K_cell_cell = K[face_count:, face_count:].tocsc()
    f_face = np.asarray(operators.f[:face_count], dtype=np.float64)
    f_cell = np.asarray(operators.f[face_count:], dtype=np.float64)

    cell_lu = splu(K_cell_cell)
    cell_response = cell_lu.solve(K_cell_face.toarray())
    condensed_K = np.asarray(
        K_face_face.toarray() - K_face_cell @ cell_response,
        dtype=np.float64,
    )
    condensed_K = 0.5 * (condensed_K + condensed_K.T)
    condensed_f = np.asarray(
        f_face - K_face_cell @ cell_lu.solve(f_cell),
        dtype=np.float64,
    ).ravel()

    eigenvalues = np.linalg.eigvalsh(condensed_K)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    if float(eigenvalues[0]) <= 1.0e-12 * scale:
        raise RuntimeError("condensed macro DtN operator is not positive definite")

    return CondensedMacroBlock(
        face_count=face_count,
        cell_count=compiled.cell_count,
        K_face=csc_matrix(condensed_K),
        f_face=condensed_f,
        K_cell_face=K_cell_face,
        K_cell_cell_lu=cell_lu,
        f_cell=f_cell,
    )


def reduce_port_modes(
    macro: CondensedMacroBlock,
    mode_count: int,
) -> ModalMacroBlock:
    """Build a stable port basis with the uniform-temperature mode retained."""
    if mode_count < 1 or mode_count > macro.face_count:
        raise ValueError("mode_count must be between 1 and the physical port count")

    compliance = splu(macro.K_face).solve(np.eye(macro.face_count))
    compliance = 0.5 * (compliance + compliance.T)
    left_vectors, _, _ = np.linalg.svd(compliance, full_matrices=False)

    uniform = np.ones((macro.face_count, 1), dtype=np.float64)
    uniform /= np.linalg.norm(uniform)
    candidates = np.hstack((uniform, left_vectors))
    basis, _ = np.linalg.qr(candidates, mode="reduced")
    basis = np.ascontiguousarray(basis[:, :mode_count], dtype=np.float64)

    projected = basis @ (basis.T @ compliance)
    denominator = float(np.linalg.norm(compliance, ord="fro") ** 2)
    retained_energy = float(np.linalg.norm(projected, ord="fro") ** 2 / denominator)

    modal_K = np.asarray(basis.T @ (macro.K_face @ basis), dtype=np.float64)
    modal_K = 0.5 * (modal_K + modal_K.T)
    modal_f = np.asarray(basis.T @ macro.f_face, dtype=np.float64).ravel()
    return ModalMacroBlock(
        physical=macro,
        basis=basis,
        retained_energy=retained_energy,
        K=csc_matrix(modal_K),
        f=modal_f,
    )


def reference_solution() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Solve the monolithic model and return its detail/macro index maps."""
    started = perf_counter()
    with build_full_reference_model() as model:
        with model.compile() as compiled:
            options = metahotspot.SolveOptions.default()
            options.nonlinear_relative_tolerance = 1.0e-10
            with compiled.solve(opts=options) as solution:
                temperature = solution.temperature.copy()
            block_ids = compiled.block_ids.copy()
    elapsed = perf_counter() - started
    detail = np.flatnonzero(block_ids == FULL_DETAIL_BLOCK_ID)
    macro = np.flatnonzero(block_ids == FULL_MACRO_BLOCK_ID)
    return temperature, detail, macro, elapsed


def solve_reduced_case(
    detailed,
    detailed_ports: PortMap,
    physical_macro: CondensedMacroBlock,
    reference: np.ndarray,
    full_detail_cells: np.ndarray,
    full_macro_cells: np.ndarray,
    mode_count: int,
) -> ReductionResult:
    started = perf_counter()
    macro = reduce_port_modes(physical_macro, mode_count=mode_count)
    initial_modes = macro.basis.T @ np.full(
        physical_macro.face_count,
        INITIAL_TEMPERATURE,
    )
    initial_state = np.concatenate((detailed.default_state(), initial_modes))
    operators = metahotspot.Operators(
        K=macro.K,
        C=csc_matrix((macro.mode_count, macro.mode_count)),
        f=macro.f,
    )
    dtn = DtNModel(operators=operators, port_basis=macro.basis)

    options = metahotspot.SolveOptions.default()
    options.nonlinear_relative_tolerance = 1.0e-10
    with metahotspot.macromodel.solve(
        detailed,
        dtn=dtn,
        ports=detailed_ports,
        state=initial_state,
        opts=options,
    ) as solution:
        reduced = solution.state.copy()

    detail_count = full_detail_cells.size
    modal_state = reduced[detail_count:]
    macro_temperature = macro.recover(modal_state)
    if macro_temperature.size != full_macro_cells.size:
        raise RuntimeError("recovered macro size differs from the reference mapping")

    recovered = np.empty_like(reference)
    recovered[full_detail_cells] = reduced[:detail_count]
    recovered[full_macro_cells] = macro_temperature
    difference = recovered - reference
    port_temperature = macro.port_temperature(modal_state)
    return ReductionResult(
        mode_count=mode_count,
        online_dofs=detail_count + mode_count,
        retained_energy=macro.retained_energy,
        relative_error=float(np.linalg.norm(difference) / np.linalg.norm(reference)),
        max_error=float(np.max(np.abs(difference))),
        elapsed_seconds=perf_counter() - started,
        port_temperature_min=float(np.min(port_temperature)),
        port_temperature_max=float(np.max(port_temperature)),
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
            with PortMap(
                detailed,
                x_port_patches(enums.Face.XP, DETAIL_LENGTH_MM),
            ) as detailed_ports:
                with build_macro_model() as macro_model:
                    with macro_model.compile() as macro_compiled:
                        with PortMap(
                            macro_compiled,
                            x_port_patches(enums.Face.XM, DETAIL_LENGTH_MM),
                        ) as macro_ports:
                            physical_macro = condense_macro(
                                macro_compiled,
                                macro_ports,
                            )
                if physical_macro.cell_count != full_macro_cells.size:
                    raise RuntimeError(
                        "standalone macro cell ordering differs from the reference"
                    )
                results = [
                    solve_reduced_case(
                        detailed,
                        detailed_ports,
                        physical_macro,
                        reference,
                        full_detail_cells,
                        full_macro_cells,
                        mode_count,
                    )
                    for mode_count in PORT_MODE_COUNTS
                ]

    detail_count = full_detail_cells.size
    print("=" * 90)
    print("Standalone nonlinear FVM + SVD-reduced C++ DtN macro experiment")
    print("=" * 90)
    print(
        f"Full problem: {reference.size} DoFs; detailed={detail_count}; "
        f"macro={full_macro_cells.size}"
    )
    print(
        f"Physical ports={physical_macro.face_count}; "
        f"macro cells={physical_macro.cell_count}; "
        f"monolithic solve={reference_seconds:.3f}s"
    )
    print()
    print(
        " modes | online/full | captured compliance | relative L2 | "
        "max error [K] | port range [K] | solve [s]"
    )
    print("-" * 100)
    for result in results:
        print(
            f"{result.mode_count:6d} | "
            f"{result.online_dofs:4d}/{reference.size:<4d} | "
            f"{result.retained_energy:18.5%} | "
            f"{result.relative_error:11.3e} | "
            f"{result.max_error:13.3e} | "
            f"{result.port_temperature_min:7.2f}..{result.port_temperature_max:7.2f} | "
            f"{result.elapsed_seconds:8.3f}"
        )
    print()
    print("Macro contract:")
    print("  offline: C++ assembles [physical face ports, macro FVM cells]")
    print("  reduce : Python condenses macro cells and projects the DtN map")
    print("  online : physical port temperature T_p = Phi @ q")
    print("  solve  : C++ updates nonlinear conductance, assembly, and scheduling")

    best = results[-1]
    if best.relative_error > 1.0e-3 or best.max_error > 1.0:
        print("FAIL: 12 retained modes are insufficient for this nonuniform case")
        return 1
    print("PASS: the mode sweep converges within the practical demo tolerance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
