#!/usr/bin/env python3
"""Four-component package assembly: substrate, die, TIM, lid.

Scenario: component reuse.  Each of the four solids is a *component* that is
extracted once and then assembled into whatever package a user builds.  The die
is smaller than the substrate, the lid overhangs the die, and the surfaces that
are not connection surfaces (the cavity above the substrate) are adiabatic.

Two surfaces are ambient boundaries and therefore the BCI parameters:

* ``lid`` top           -- radiation plus convection, ``h_top`` in the BCI box,
* ``substrate`` bottom  -- convection to the board, ``h_bot`` in the BCI box.

Three surfaces are genuine component-to-component interfaces.  They are trimmed
to the in-plane overlap of the two facing layers, so a small die under a large
lid couples only over the die footprint.  The in-plane grid is shared, so every
interface is conforming and couples one-to-one; that is a stated simplification,
not a hidden one.

Units are SI: W/(m K), J/(m^3 K), m, K, W.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

STEFAN_BOLTZMANN = 5.670374419e-8


@dataclass(frozen=True)
class LayerSpec:
    """One solid in the stack; each layer is one reusable component."""

    name: str
    conductivity: float  # W/(m K)
    volumetric_capacity: float  # J/(m^3 K)
    thickness: float  # m
    cells: int  # through-thickness cells
    fraction: float  # in-plane extent relative to the footprint


STACK = (
    LayerSpec("substrate", 12.0, 2.20e6, 400e-6, 4, 1.0),
    LayerSpec("die", 130.0, 1.65e6, 150e-6, 5, 0.6),
    LayerSpec("tim", 3.5, 2.00e6, 60e-6, 3, 0.6),
    LayerSpec("lid", 380.0, 3.40e6, 800e-6, 4, 1.0),
)

INTERFACES = (("substrate", "die"), ("die", "tim"), ("tim", "lid"))
AMBIENT = (("lid", "top"), ("substrate", "bottom"))
SOURCE_POWER = np.array([2.0, 1.0])  # W


@dataclass(frozen=True)
class Face:
    """One boundary face set of a component, in local cell indices."""

    label: str
    cells: np.ndarray  # local indices, ordered by (j, i)
    ij: np.ndarray  # (n, 2) the (j, i) of each cell, same order
    half: np.ndarray  # m, centre-to-face distance
    conductivity: np.ndarray  # W/(m K) along the face normal
    area: np.ndarray  # m^2 per cell

    def conductance(self) -> np.ndarray:
        """Half-cell conductance to a shared interface node: ``k A / half``."""
        return self.conductivity * self.area / self.half

    def trim(self, keep: set[tuple[int, int]]) -> "Face":
        mask = np.array([(int(j), int(i)) in keep for j, i in self.ij], dtype=bool)
        return Face(
            label=self.label,
            cells=self.cells[mask],
            ij=self.ij[mask],
            half=self.half[mask],
            conductivity=self.conductivity[mask],
            area=self.area[mask],
        )


@dataclass(frozen=True)
class Component:
    name: str
    cells: np.ndarray  # global cell indices of this component
    K: sp.csc_matrix  # local stiffness, cut faces adiabatic
    C: sp.csc_matrix
    source: np.ndarray  # (n_local, n_src) unit-power shapes
    probes: np.ndarray  # local indices
    interfaces: tuple[Face, ...]
    ambient: tuple[Face, ...]

    @property
    def size(self) -> int:
        return int(self.K.shape[0])

    @property
    def ports(self) -> np.ndarray:
        """Every connection or ambient face cell, in a fixed order."""
        return np.concatenate([face.cells for face in self.interfaces + self.ambient])


@dataclass(frozen=True)
class Assembly:
    K: sp.csc_matrix
    C: sp.csc_matrix
    components: tuple[Component, ...]
    source: np.ndarray
    probes: np.ndarray
    probe_labels: tuple[str, ...]
    grid: int
    footprint: float
    area: float

    @property
    def cell_count(self) -> int:
        return int(self.K.shape[0])


def _span(fraction: float, n: int) -> tuple[int, int]:
    count = max(1, int(round(fraction * n)))
    low = (n - count) // 2
    return low, low + count


def build(nx: int = 10, footprint: float = 10.0e-3, stack=STACK) -> Assembly:
    dx = footprint / nx
    area = dx * dx

    index: dict[tuple[int, int, int, int], int] = {}
    position: list[tuple] = []
    step: list[float] = []
    conductivity: list[float] = []
    capacity: list[float] = []
    z = 0.0
    for layer, spec in enumerate(stack):
        size = spec.thickness / spec.cells
        low, high = _span(spec.fraction, nx)
        for k in range(spec.cells):
            for j in range(low, high):
                for i in range(low, high):
                    index[(layer, k, j, i)] = len(position)
                    position.append((layer, k, j, i))
                    step.append(size)
                    conductivity.append(spec.conductivity)
                    capacity.append(spec.volumetric_capacity * size * area)
            z += size

    step = np.asarray(step)
    conductivity = np.asarray(conductivity)
    n = len(position)

    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []

    def link(a: int, b: int, conductance: float) -> None:
        rows.extend((a, b, a, b))
        cols.extend((a, b, b, a))
        values.extend((conductance, conductance, -conductance, -conductance))

    for key, a in index.items():
        layer, k, j, i = key
        for dj, di in ((0, 1), (1, 0)):
            neighbour = index.get((layer, k, j + dj, i + di))
            if neighbour is not None:
                link(a, neighbour, conductivity[a] * step[a])
        above = index.get((layer, k + 1, j, i))
        if above is None and k == stack[layer].cells - 1:
            above = index.get((layer + 1, 0, j, i))
        if above is not None:
            link(a, above,
                 area / (0.5 * step[a] / conductivity[a] + 0.5 * step[above] / conductivity[above]))

    K = sp.coo_matrix((values, (rows, cols)), shape=(n, n)).tocsc()
    C = sp.diags(np.asarray(capacity), format="csc")

    source = np.zeros((n, SOURCE_POWER.size))
    cell_sets: dict[str, np.ndarray] = {}
    locals_: dict[str, dict[int, int]] = {}
    for layer, spec in enumerate(stack):
        cells = np.array(
            [index[(layer, k, j, i)] for k in range(spec.cells)
             for j in range(nx) for i in range(nx) if (layer, k, j, i) in index],
            dtype=np.int64,
        )
        cell_sets[spec.name] = cells
        locals_[spec.name] = {int(c): p for p, c in enumerate(cells)}

    def build_face(spec: LayerSpec, label: str, through: int) -> Face:
        cells = cell_sets[spec.name]
        local = locals_[spec.name]
        members = np.array([local[int(c)] for c in cells if position[c][1] == through],
                           dtype=np.int64)
        ij = np.array([position[cell_sets[spec.name][m]][2:] for m in members], dtype=np.int64)
        order = np.lexsort((ij[:, 1], ij[:, 0]))
        members, ij = members[order], ij[order]
        return Face(
            label=label,
            cells=members,
            ij=ij,
            half=np.full(members.size, 0.5 * spec.thickness / spec.cells),
            conductivity=np.full(members.size, spec.conductivity),
            area=np.full(members.size, area),
        )

    specs = {spec.name: spec for spec in stack}
    interface_faces: dict[str, dict[str, Face]] = {spec.name: {} for spec in stack}
    for lower, upper in INTERFACES:
        interface_faces[lower][f"to_{upper}"] = build_face(specs[lower], f"to_{upper}",
                                                          specs[lower].cells - 1)
        interface_faces[upper][f"to_{lower}"] = build_face(specs[upper], f"to_{lower}", 0)
        low_face = interface_faces[lower][f"to_{upper}"]
        high_face = interface_faces[upper][f"to_{lower}"]
        common = {(int(j), int(i)) for j, i in low_face.ij} & {
            (int(j), int(i)) for j, i in high_face.ij
        }
        if not common:
            raise RuntimeError(f"interface {lower}-{upper} has no overlapping faces")
        interface_faces[lower][f"to_{upper}"] = low_face.trim(common)
        interface_faces[upper][f"to_{lower}"] = high_face.trim(common)

    ambient_faces: dict[str, dict[str, Face]] = {spec.name: {} for spec in stack}
    for name, side in AMBIENT:
        spec = specs[name]
        ambient_faces[name][side] = build_face(spec, side,
                                               spec.cells - 1 if side == "top" else 0)

    components = []
    probe_globals: list[int] = []
    probe_labels: list[str] = []
    for layer, spec in enumerate(stack):
        cells = cell_sets[spec.name]
        local = locals_[spec.name]
        outside = np.setdiff1d(np.arange(n), cells)
        cross = K[cells][:, outside].tocoo()
        phantom = np.zeros(cells.size)
        for row, value in zip(cross.row, cross.data):
            if value < 0.0:
                phantom[row] -= value

        low, high = _span(spec.fraction, nx)
        middle = (low + high) // 2
        local_source = np.zeros((cells.size, SOURCE_POWER.size))
        local_probes = np.empty(0, dtype=np.int64)
        # Probe layers must be strictly interior: layer 0 and layer cells-1 are
        # port cells (interface or ambient), so a probe there would be a port DOF
        # and blind to the interior enrichment.  Guarding this in code rather
        # than by inspection is what keeps the metric meaningful.
        interior_layers = list(range(1, spec.cells - 1))
        if spec.name == "die":
            for s in range(SOURCE_POWER.size):
                cell = index[(layer, 1 + s, middle + 2 * s - 1, middle + 2 * s - 1)]
                local_source[local[cell], s] = 1.0
                source[cell, s] = 1.0
        if spec.name in ("die", "substrate", "lid", "tim") and interior_layers:
            middle_layer = interior_layers[len(interior_layers) // 2]
            local_probes = np.array(
                [local[index[(layer, middle_layer, middle, middle)]]], dtype=np.int64
            )
        if spec.name == "die" and len(interior_layers) > 1:
            local_probes = np.array(
                [local[index[(layer, interior_layers[0], middle, middle)]],
                 local[index[(layer, interior_layers[-1], low + 1, high - 2)]]],
                dtype=np.int64,
            )

        for p, value in enumerate(local_probes):
            probe_globals.append(int(cells[value]))
            probe_labels.append(f"{spec.name}_{p}")

        components.append(
            Component(
                name=spec.name,
                cells=cells,
                K=(K[cells][:, cells] - sp.diags(phantom)).tocsc(),
                C=C[cells][:, cells].tocsc(),
                source=local_source,
                probes=local_probes,
                interfaces=tuple(interface_faces[spec.name][label]
                                 for label in sorted(interface_faces[spec.name])),
                ambient=tuple(ambient_faces[spec.name][label]
                              for label in sorted(ambient_faces[spec.name])),
            )
        )

    return Assembly(
        K=K,
        C=C,
        components=tuple(components),
        source=source,
        probes=np.asarray(probe_globals, dtype=np.int64),
        probe_labels=tuple(probe_labels),
        grid=nx,
        footprint=footprint,
        area=area,
    )
