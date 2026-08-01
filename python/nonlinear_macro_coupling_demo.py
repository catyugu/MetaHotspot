#!/usr/bin/env python3
"""Couple a nonlinear detailed FVM block to a condensed exact-port DtN macro."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

import metahotspot
from metahotspot import enums
from metahotspot.macromodel import PortMap, PortPatch


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
        "macro", kx="120", ky="120", kz="120", rho="0", c="0"
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
    scale = 1e-3
    return [
        PortPatch(
            int(face),
            coordinate_mm * scale,
            (y0 * scale, y1 * scale, z0 * scale, z1 * scale),
        )
        for y0, y1 in zip(Y_VERTICES_MM[:-1], Y_VERTICES_MM[1:], strict=True)
        for z0, z1 in zip(Z_VERTICES_MM[:-1], Z_VERTICES_MM[1:], strict=True)
    ]


def add_domain_block(
    model: metahotspot.Model,
    material: str,
    x: float,
    width: float,
    heat_source: str | None = None,
) -> int:
    layer = model.add_layer(thickness=str(DOMAIN_THICKNESS_MM))
    block = (
        model.add_block(layer, material, heat_source=heat_source)
        if heat_source is not None
        else model.add_block(layer, material)
    )
    model.add_rect(
        block,
        x=str(x),
        y="0",
        width=str(width),
        height=str(DOMAIN_HEIGHT_MM),
    )
    return block


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
    model = metahotspot.Model()
    set_common_settings(model, np.arange(0.0, DETAIL_LENGTH_MM + 1.0))
    add_materials(model)
    add_domain_block(model, "nonlinear", 0.0, DETAIL_LENGTH_MM, HEAT_SOURCE)
    model.add_dirichlet("300", x_face(0.0))
    return model


def build_macro_model() -> metahotspot.Model:
    model = metahotspot.Model()
    set_common_settings(
        model, np.arange(DETAIL_LENGTH_MM, FULL_LENGTH_MM + 1.0)
    )
    add_materials(model)
    add_domain_block(model, "macro", DETAIL_LENGTH_MM, MACRO_LENGTH_MM)
    model.add_dirichlet("300", x_face(FULL_LENGTH_MM))
    return model


@dataclass
class CondensedMacroBlock:
    face_count: int
    K_face: csc_matrix
    f_face: np.ndarray
    K_cell_face: csc_matrix
    K_cell_cell_lu: object
    f_cell: np.ndarray

    def recover(self, face_temperature: np.ndarray) -> np.ndarray:
        return self.K_cell_cell_lu.solve(
            self.f_cell - self.K_cell_face @ face_temperature
        )


@dataclass
class CouplingResult:
    online_dofs: int
    relative_error: float
    max_error: float
    elapsed_seconds: float
    port_temperature_min: float
    port_temperature_max: float


def condense_macro(compiled, ports: PortMap) -> CondensedMacroBlock:
    operators = ports.assemble()
    face_count = ports.port_count
    expected = face_count + compiled.cell_count
    if operators.K.shape != (expected, expected):
        raise RuntimeError("unexpected C++ DtN operator dimension")

    K = operators.K.tocsc()
    K_face_face = K[:face_count, :face_count].tocsc()
    K_face_cell = K[:face_count, face_count:].tocsc()
    K_cell_face = K[face_count:, :face_count].tocsc()
    K_cell_cell_lu = splu(K[face_count:, face_count:].tocsc())
    f_face = np.asarray(operators.f[:face_count], dtype=np.float64)
    f_cell = np.asarray(operators.f[face_count:], dtype=np.float64)

    response = K_cell_cell_lu.solve(K_cell_face.toarray())
    condensed_K = np.asarray(K_face_face.toarray() - K_face_cell @ response)
    condensed_K = 0.5 * (condensed_K + condensed_K.T)
    condensed_f = np.asarray(
        f_face - K_face_cell @ K_cell_cell_lu.solve(f_cell), dtype=np.float64
    ).ravel()
    return CondensedMacroBlock(
        face_count,
        csc_matrix(condensed_K),
        condensed_f,
        K_cell_face,
        K_cell_cell_lu,
        f_cell,
    )


def reference_solution() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    started = perf_counter()
    with build_full_reference_model() as model:
        with model.compile() as compiled:
            options = metahotspot.SolveOptions.default()
            options.nonlinear_relative_tolerance = 1e-10
            with compiled.solve(opts=options) as solution:
                temperature = solution.temperature.copy()
            block_ids = compiled.block_ids.copy()
    detail = np.flatnonzero(block_ids == FULL_DETAIL_BLOCK_ID)
    macro = np.flatnonzero(block_ids == FULL_MACRO_BLOCK_ID)
    return temperature, detail, macro, perf_counter() - started


def solve_condensed_case(
    detailed,
    detailed_ports: PortMap,
    macro: CondensedMacroBlock,
    reference: np.ndarray,
    full_detail_cells: np.ndarray,
    full_macro_cells: np.ndarray,
) -> CouplingResult:
    started = perf_counter()
    initial_state = np.r_[
        detailed.default_state(),
        np.full(macro.face_count, INITIAL_TEMPERATURE),
    ]
    operators = metahotspot.Operators(
        K=macro.K_face,
        C=csc_matrix((macro.face_count, macro.face_count)),
        f=macro.f_face,
    )
    options = metahotspot.SolveOptions.default()
    options.nonlinear_relative_tolerance = 1e-10
    with metahotspot.macromodel.solve(
        detailed,
        operators=operators,
        ports=detailed_ports,
        state=initial_state,
        opts=options,
    ) as solution:
        reduced = solution.state.copy()

    detail_count = full_detail_cells.size
    port_temperature = reduced[detail_count:]
    recovered = np.empty_like(reference)
    recovered[full_detail_cells] = reduced[:detail_count]
    recovered[full_macro_cells] = macro.recover(port_temperature)
    difference = recovered - reference
    return CouplingResult(
        detail_count + macro.face_count,
        float(np.linalg.norm(difference) / np.linalg.norm(reference)),
        float(np.max(np.abs(difference))),
        perf_counter() - started,
        float(np.min(port_temperature)),
        float(np.max(port_temperature)),
    )


def main() -> int:
    reference, detail_cells, macro_cells, reference_s = reference_solution()
    with build_detailed_nonlinear_model() as detailed_model:
        with detailed_model.compile() as detailed:
            with PortMap(
                detailed, x_port_patches(enums.Face.XP, DETAIL_LENGTH_MM)
            ) as detailed_ports:
                with build_macro_model() as macro_model:
                    with macro_model.compile() as macro_compiled:
                        with PortMap(
                            macro_compiled,
                            x_port_patches(enums.Face.XM, DETAIL_LENGTH_MM),
                        ) as macro_ports:
                            macro = condense_macro(macro_compiled, macro_ports)
                result = solve_condensed_case(
                    detailed,
                    detailed_ports,
                    macro,
                    reference,
                    detail_cells,
                    macro_cells,
                )

    print("=" * 86)
    print("Nonlinear FVM + exact-port condensed C++ DtN macro experiment")
    print("=" * 86)
    print(
        f"Full DoFs={reference.size}; detail={detail_cells.size}; "
        f"macro cells={macro_cells.size}; physical ports={macro.face_count}"
    )
    print(
        f"Online DoFs={result.online_dofs}; monolithic={reference_s:.3f}s; "
        f"coupled={result.elapsed_seconds:.3f}s"
    )
    print(
        f"Relative L2={result.relative_error:.3e}; max error={result.max_error:.6f} K; "
        f"port range=[{result.port_temperature_min:.3f}, "
        f"{result.port_temperature_max:.3f}] K"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
