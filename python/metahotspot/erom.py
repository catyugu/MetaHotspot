"""Standalone PTA-to-.mherom extraction using the native Project compiler."""

from __future__ import annotations

import argparse
import os
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from metahotspot.macromodel.contour import contour_training_sources
from metahotspot.macromodel.utils import (
    build_parametric_basis,
    normalized_operators,
    project_bci,
)

GEOMETRY = np.dtype(
    [("axis", "<u4"), ("normal", "<u4"), ("plane", "<f8"), ("bounds", "<f8", (4,))]
)
TRAINED = np.dtype([("geometry", GEOMETRY), ("cell", "<u4"), ("resistance", "<f8")])
WEIGHT = np.dtype([("cell", "<u4"), ("value", "<f8")])


@dataclass(frozen=True)
class Surface:
    id: int
    name: str
    facets: np.ndarray
    # Minimum, maximum, default HTC and default ambient temperature.
    htc: tuple[float, float, float, float] | None

    @property
    def geometry(self):
        return self.facets if self.htc is None else self.facets["geometry"]

    def conductance(self, h):
        rects = self.geometry["bounds"]
        area = (rects[:, 1] - rects[:, 0]) * (rects[:, 3] - rects[:, 2])
        return area * h / (1.0 + self.facets["resistance"] * h)


@dataclass(frozen=True)
class Source:
    id: int
    name: str
    power: float


@dataclass(frozen=True)
class Output:
    id: int
    name: str
    offset: float
    location: tuple[float, float, float] | int  # position or source ID


@dataclass(frozen=True)
class ExtractionProject:
    K: sp.csc_matrix
    C: sp.csc_matrix
    B: np.ndarray
    L: sp.csr_matrix
    surfaces: list[Surface]
    sources: list[Source]
    outputs: list[Output]
    reference: float
    initial: float


class _PreparedReader:
    def __init__(self, stream: BinaryIO):
        self.stream = stream

    def read(self, size):
        data = self.stream.read(size)
        if len(data) != size:
            raise ValueError("truncated native extraction input")
        return data

    def unpack(self, fmt):
        fmt = struct.Struct("<" + fmt)
        return fmt.unpack(self.read(fmt.size))

    def name(self):
        (size,) = self.unpack("I")
        return self.read(size).decode("utf-8")

    def array(self, dtype, count):
        dtype = np.dtype(dtype)
        return np.frombuffer(self.read(dtype.itemsize * count), dtype)

    def weights(self):
        (count,) = self.unpack("I")
        return self.array(WEIGHT, count)


def read_prepared_project(path: Path) -> ExtractionProject:
    with path.open("rb") as stream:
        read = _PreparedReader(stream)
        if read.read(8) != b"MHPREP01":
            raise ValueError("incompatible native extraction helper")
        n, nnz, ng, ns, no, reference, initial = read.unpack("5I2d")
        ptr = read.array("<u4", n + 1)
        cols = read.array("<u4", nnz)
        values = read.array("<f8", nnz)
        K = sp.csr_matrix((values, cols, ptr), shape=(n, n)).tocsc()
        C = sp.diags(read.array("<f8", n), format="csc")
        surfaces = []
        for _ in range(ng):
            (id_,) = read.unpack("I")
            name = read.name()
            (trained,) = read.unpack("I")
            htc = read.unpack("4d") if trained else None
            (count,) = read.unpack("I")
            surfaces.append(
                Surface(
                    id_, name, read.array(TRAINED if trained else GEOMETRY, count), htc
                )
            )
        sources = []
        B = np.zeros((n, ns))
        for i in range(ns):
            (id_,) = read.unpack("I")
            name = read.name()
            (power,) = read.unpack("d")
            sources.append(Source(id_, name, power))
            weights = read.weights()
            B[weights["cell"], i] = weights["value"]
        outputs = []
        output_ptr = [0]
        output_cols = []
        output_data = []
        for _ in range(no):
            (id_,) = read.unpack("I")
            name = read.name()
            kind, offset = read.unpack("Id")
            location = read.unpack("3d") if kind == 0 else read.unpack("I")[0]
            outputs.append(Output(id_, name, offset, location))
            weights = read.weights()
            output_cols.extend(weights["cell"])
            output_data.extend(weights["value"])
            output_ptr.append(len(output_cols))
        L = sp.csr_matrix((output_data, output_cols, output_ptr), shape=(no, n))
        if stream.read(1):
            raise ValueError("unexpected data after native extraction input")
    return ExtractionProject(K, C, B, L, surfaces, sources, outputs, reference, initial)


@dataclass(frozen=True)
class ReducedProject:
    full: ExtractionProject
    K: np.ndarray
    C: np.ndarray
    B: np.ndarray
    L: np.ndarray
    Vb: np.ndarray
    uniform: np.ndarray
    tolerance: float
    max_output_error: float


def _planes(surfaces):
    for surface in surfaces:
        f = surface.geometry
        keys = np.rec.fromarrays([f["axis"], f["normal"], f["plane"]])
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        cuts = np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1
        for i, ids in enumerate(np.split(order, cuts)):
            yield f"{surface.id}-{i}", surface.facets["cell"][ids], f["bounds"][ids]


def _diagonal(surface, h, n):
    return np.bincount(
        surface.facets["cell"], weights=surface.conductance(h), minlength=n
    )


def _training(project, surfaces, degree):
    # Interpolate exact condensed conductances at both ends of each HTC range.
    # The BCI basis builder sees an affine family. Validation uses the exact law.
    n = project.K.shape[0]
    offset = np.zeros(n)
    operators, ranges = [], []
    for surface in surfaces:
        lo, hi, _, _ = surface.htc
        g_lo = _diagonal(surface, lo, n)
        if hi == lo:
            slope = g_lo / lo
        else:
            slope = (_diagonal(surface, hi, n) - g_lo) / (hi - lo)
            offset += g_lo - lo * slope
        operators.append(sp.diags(slope, format="csc"))
        ranges.append((lo, hi))
    contour, _ = contour_training_sources(n, _planes(surfaces), degree)
    G = np.column_stack((project.B, contour))
    return project.K + sp.diags(offset, format="csc"), operators, ranges, G


def _validate_outputs(project, surfaces, basis, stiffness, B, L, tolerance, domains):
    n = project.K.shape[0]
    loads = np.array([source.power for source in project.sources])
    cases = [
        [s.htc[0] for s in surfaces],
        [np.sqrt(s.htc[0]) * np.sqrt(s.htc[1]) for s in surfaces],
        [s.htc[1] for s in surfaces],
    ]
    if len(surfaces) > 1:
        cases.extend(
            [s.htc[(i + j) % 2] for i, s in enumerate(surfaces)] for j in (0, 1)
        )
    cases.append([s.htc[2] for s in surfaces])
    worst = 0.0
    for index, coefficients in enumerate(cases):
        default = index == len(cases) - 1
        diagonal = np.zeros(n)
        ambient_loads = []
        for surface, h in zip(surfaces, coefficients, strict=True):
            g = _diagonal(surface, h, n)
            diagonal += g
            # Independent 1 K environment changes during validation.
            ambient_loads.append(g)
        if default and np.unique(domains[diagonal > 0]).size <= domains.max():
            # Embedding can provide the missing thermal reference later.
            continue
        A = project.K + sp.diags(diagonal, format="csc")
        if default:
            rhs = project.B @ loads
            for surface, g in zip(surfaces, ambient_loads, strict=True):
                rhs += g * (surface.htc[3] - project.reference)
            inputs = rhs[:, None]
            reduced_inputs = basis.T @ inputs
        else:
            inputs = np.column_stack((project.B, *ambient_loads))
            reduced_inputs = np.column_stack(
                (B, basis.T @ np.column_stack(ambient_loads))
            )
        full = spla.splu(A).solve(inputs)
        reduced = scipy.linalg.solve(
            stiffness + basis.T @ (diagonal[:, None] * basis),
            reduced_inputs,
            assume_a="pos",
        )
        exact, approx = project.L @ full, L @ reduced
        error = np.max(
            np.abs(exact - approx) / np.maximum(1.0, np.abs(exact)), initial=0
        )
        if not np.isfinite(error):
            raise ValueError("nonfinite validation temperature")
        worst = max(worst, float(error))
    if worst > tolerance:
        raise ValueError(
            f"output validation error {worst:.6g} exceeds limit {tolerance:.6g}"
        )
    return worst


def extract_erom(
    project: ExtractionProject,
    *,
    tolerance: float = 1e-3,
    max_order: int = 512,
    contour_degree: int = 2,
    output_error: float = 0.02,
) -> ReducedProject:
    """Extract from compiled project physics, independent of the input file type."""
    if (
        not 0 < tolerance < 1
        or max_order < 1
        or contour_degree < 0
        or not 0 < output_error < 1
    ):
        raise ValueError(
            "invalid EROM extraction tolerance, order, contour degree or output error"
        )
    trained = [s for s in project.surfaces if s.htc is not None]
    if not trained:
        raise ValueError("EROM extraction requires HTC surfaces")
    n = project.K.shape[0]
    n_domains, domains = sp.csgraph.connected_components(project.K, directed=False)
    owned = np.unique(domains[np.concatenate([s.facets["cell"] for s in trained])])
    if owned.size != n_domains:
        raise ValueError(
            "each thermally disconnected FOM region needs an HTC training surface"
        )

    K_train, terms, ranges, inputs = _training(project, trained, contour_degree)
    operators = normalized_operators(project.K, project.C, np.zeros(n))
    basis, _ = build_parametric_basis(
        normalized_operators(K_train, project.C, np.zeros(n)),
        inputs,
        terms,
        ranges,
        tolerance=tolerance,
        max_order=max_order,
    )
    C, K, B, _ = project_bci(operators, project.B, (), basis)
    C, K = C.toarray(), K.toarray()
    # Remove roundoff asymmetry and negative roundoff on the diagonal.
    C, K = (C + C.T) * 0.5, (K + K.T) * 0.5
    K[np.diag_indices_from(K)] = np.maximum(K.diagonal(), 0)
    L = project.L @ basis
    boundary_cells = np.concatenate([s.facets["cell"] for s in trained])
    uniform = scipy.linalg.solve(C, basis.T @ project.C.diagonal(), assume_a="pos")
    error = _validate_outputs(project, trained, basis, K, B, L, output_error, domains)
    Vb = basis[boundary_cells]
    if any(not np.isfinite(array).all() for array in (K, C, B, L, Vb, uniform)):
        raise ValueError("extraction produced a nonfinite reduced operator")
    return ReducedProject(project, K, C, B, L, Vb, uniform, tolerance, error)


def _outline(geometry):
    """Merge adjacent non-numerical rectangles on each coplanar surface."""
    facets = geometry.copy()
    for lo in (0, 2):
        other = 2 - lo
        b = facets["bounds"]
        order = np.lexsort(
            (
                b[:, lo],
                b[:, other + 1],
                b[:, other],
                facets["plane"],
                facets["normal"],
                facets["axis"],
            )
        )
        facets = facets[order]
        b = facets["bounds"]
        same = (
            (facets["axis"][1:] == facets["axis"][:-1])
            & (facets["normal"][1:] == facets["normal"][:-1])
            & (facets["plane"][1:] == facets["plane"][:-1])
            & (b[1:, other] == b[:-1, other])
            & (b[1:, other + 1] == b[:-1, other + 1])
            & (b[1:, lo] == b[:-1, lo + 1])
        )
        starts = np.r_[0, np.flatnonzero(~same) + 1]
        ends = np.r_[starts[1:] - 1, facets.size - 1]
        merged = facets[starts]
        merged["bounds"][:, lo + 1] = facets["bounds"][ends, lo + 1]
        facets = merged
    return facets


def write_erom(path: Path, rom: ReducedProject) -> None:
    """Write native .mherom v1 atomically after successful extraction."""
    project = rom.full
    surfaces = sorted(project.surfaces, key=lambda s: s.htc is None)
    geometry = [s.geometry if s.htc else _outline(s.geometry) for s in surfaces]
    filename = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            filename = stream.name

            def pack(fmt, *values):
                stream.write(struct.pack("<" + fmt, *values))

            def name(value):
                encoded = value.encode("utf-8")
                pack("I", len(encoded))
                stream.write(encoded)

            stream.write(b"MHEROM\r\n")
            pack(
                "7I3dI",
                1,
                rom.K.shape[0],
                len(project.sources),
                len(project.outputs),
                rom.Vb.shape[0],
                sum(len(g) for g in geometry),
                len(surfaces),
                project.reference,
                project.initial,
                rom.tolerance,
                1,
            )
            for array in (rom.K, rom.C, rom.B, rom.L, rom.Vb):
                stream.write(np.asarray(array, dtype="<f8").tobytes())
            for surface in surfaces:
                if surface.htc:
                    stream.write(surface.facets["resistance"].tobytes())
            stream.write(
                np.asarray(
                    rom.uniform * (project.initial - project.reference), dtype="<f8"
                ).tobytes()
            )
            stream.write(np.asarray(rom.uniform, dtype="<f8").tobytes())
            for facets in geometry:
                stream.write(facets.tobytes())
            first = 0
            for surface, facets in zip(surfaces, geometry, strict=True):
                pack("I", surface.id)
                name(surface.name)
                pack("I", surface.htc is not None)
                if surface.htc:
                    pack("4d", *surface.htc)
                pack("I", len(facets))
                stream.write(
                    np.arange(first, first + len(facets), dtype="<u4").tobytes()
                )
                first += len(facets)
            for source in project.sources:
                pack("I", source.id)
                name(source.name)
                pack("d", source.power)
            for output in project.outputs:
                point = isinstance(output.location, tuple)
                pack("I", output.id)
                name(output.name)
                pack("Id", 0 if point else 1, output.offset)
                if point:
                    pack("3d", *output.location)
                else:
                    pack("I", output.location)
        os.replace(filename, path)
    finally:
        if filename:
            Path(filename).unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="metahotspot-erom", description=__doc__)
    parser.add_argument("pta", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument(
        "--prepare", default="metahotspot-erom-prepare", help="native helper executable"
    )
    parser.add_argument("--list-surfaces", action="store_true")
    parser.add_argument(
        "--htc", action="append", nargs=3, metavar=("SURFACE", "MIN", "MAX"), default=[]
    )
    probes = parser.add_mutually_exclusive_group()
    probes.add_argument(
        "--probe",
        action="append",
        default=[],
        help="keep this named point probe (repeatable)",
    )
    probes.add_argument("--no-point-probes", action="store_true")
    parser.add_argument("--no-source-temperatures", action="store_true")
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--max-order", type=int, default=512)
    parser.add_argument("--contour-degree", type=int, default=2)
    parser.add_argument("--output-error", type=float, default=0.02)
    args = parser.parse_args(argv)
    command = [args.prepare, str(args.pta)]
    try:
        if args.list_surfaces:
            subprocess.run([*command, "--list-surfaces"], check=True)
            return
        if args.output is None or args.output.suffix != ".mherom" or not args.htc:
            parser.error("provide a .mherom output path and at least one --htc group")
        if args.output.resolve() == args.pta.resolve():
            parser.error("the output path must differ from the PTA input path")
        with tempfile.TemporaryDirectory() as tmp:
            prepared = Path(tmp) / "project.bin"
            command.append(str(prepared))
            for group in args.htc:
                command.extend(("--htc", *group))
            for probe in args.probe:
                command.extend(("--probe", probe))
            if args.no_point_probes:
                command.append("--no-point-probes")
            if args.no_source_temperatures:
                command.append("--no-source-temperatures")
            subprocess.run(command, check=True)
            project = read_prepared_project(prepared)
        rom = extract_erom(
            project,
            tolerance=args.tolerance,
            max_order=args.max_order,
            contour_degree=args.contour_degree,
            output_error=args.output_error,
        )
        write_erom(args.output, rom)
        print(
            f"{args.output}  order={rom.K.shape[0]}  "
            f"output error={rom.max_output_error:.6g}"
        )
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"metahotspot-erom: {error}\n")


if __name__ == "__main__":
    main()
