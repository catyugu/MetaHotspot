#!/usr/bin/env python3
"""A reusable package component: layered 3-D FVM stack with two physical ports.

Scenario: an IC package (lid / TIM / die / underfill / substrate) is extracted
once and then embedded in assemblies whose *external* coupling is not known at
extraction time.  Two surfaces are real connection surfaces:

* ``top``    - lid surface to a heat sink / ambient (radiation + convection),
* ``bottom`` - substrate surface to the board (contact conductance).

The interior is linear and time invariant.  Both coupling laws are nonlinear
functions of the *port cell temperature only*, so the nonlinearity is diagonal
and confined to the interface.  That is the structural assumption the whole
methodology rests on, and it is stated here rather than hidden in the solver.

Units are SI throughout: W/(m K), J/(m^3 K), m, K, W.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

STEFAN_BOLTZMANN = 5.670374419e-8


@dataclass(frozen=True)
class Layer:
    name: str
    conductivity: float  # W/(m K)
    volumetric_capacity: float  # J/(m^3 K)
    thickness: float  # m
    cells: int  # through-thickness cells


STACK = (
    Layer("lid", 380.0, 3.40e6, 600e-6, 2),
    Layer("tim", 3.5, 2.00e6, 40e-6, 1),
    Layer("die", 130.0, 1.65e6, 120e-6, 2),
    Layer("underfill", 0.55, 1.80e6, 80e-6, 1),
    Layer("substrate", 12.0, 2.20e6, 500e-6, 2),
)

FOOTPRINT = 10.0e-3  # m, square package
GRID = 14  # in-plane cells per side


@dataclass(frozen=True)
class Port:
    """One physical connection surface: its cells and their half-cell data."""

    label: str
    cells: np.ndarray  # cell indices carrying the interface flux
    area: np.ndarray  # m^2, face area per cell
    half: np.ndarray  # m, centre-to-face distance
    conductivity: np.ndarray  # W/(m K) along the face normal


@dataclass(frozen=True)
class Package:
    K: sp.csc_matrix
    C: sp.csc_matrix
    ports: tuple[Port, ...]
    source: np.ndarray  # (n, n_src) unit-power shapes, W
    probes: np.ndarray  # interior temperature observables
    shape: tuple[int, int, int]  # (nx, ny, nz)

    @property
    def size(self) -> int:
        return int(self.K.shape[0])

    @property
    def interface(self) -> np.ndarray:
        return np.concatenate([port.cells for port in self.ports])


def build_package(grid: int = GRID, footprint: float = FOOTPRINT) -> Package:
    """Assemble the layered stack with TPFA and identify both ports."""
    nx = ny = grid
    nz = sum(layer.cells for layer in STACK)
    dx = footprint / nx
    dy = footprint / ny
    area_xy = dx * dy

    thickness = np.empty(nz)
    conductivity = np.empty(nz)
    capacity = np.empty(nz)
    start = 0
    for layer in STACK:
        step = layer.thickness / layer.cells
        thickness[start : start + layer.cells] = step
        conductivity[start : start + layer.cells] = layer.conductivity
        capacity[start : start + layer.cells] = layer.volumetric_capacity * step
        start += layer.cells
    centres = np.cumsum(thickness) - 0.5 * thickness

    n = nx * ny * nz
    index = lambda i, j, k: (k * ny + j) * nx + i  # noqa: E731

    rows, cols, values = [], [], []

    def link(a: int, b: int, conductance: float) -> None:
        rows.extend((a, b, a, b))
        cols.extend((a, b, b, a))
        values.extend((conductance, conductance, -conductance, -conductance))

    for k in range(nz):
        in_plane_x = conductivity[k] * thickness[k] * dy / dx
        in_plane_y = conductivity[k] * thickness[k] * dx / dy
        for j in range(ny):
            for i in range(nx):
                a = index(i, j, k)
                if i + 1 < nx:
                    link(a, index(i + 1, j, k), in_plane_x)
                if j + 1 < ny:
                    link(a, index(i, j + 1, k), in_plane_y)

    for k in range(nz - 1):
        # Two half cells in series between adjacent centres.
        conductance = area_xy / (
            0.5 * thickness[k] / conductivity[k] + 0.5 * thickness[k + 1] / conductivity[k + 1]
        )
        for j in range(ny):
            for i in range(nx):
                link(index(i, j, k), index(i, j, k + 1), conductance)

    K = sp.coo_matrix((values, (rows, cols)), shape=(n, n)).tocsc()
    C = sp.diags(np.repeat(capacity, nx * ny) * area_xy, format="csc")

    top_cells = np.array([index(i, j, nz - 1) for j in range(ny) for i in range(nx)])
    bottom_cells = np.array([index(i, j, 0) for j in range(ny) for i in range(nx)])
    top = Port(
        label="top",
        cells=top_cells,
        area=np.full(top_cells.size, area_xy),
        half=np.full(top_cells.size, 0.5 * thickness[nz - 1]),
        conductivity=np.full(top_cells.size, conductivity[nz - 1]),
    )
    bottom = Port(
        label="bottom",
        cells=bottom_cells,
        area=np.full(bottom_cells.size, area_xy),
        half=np.full(bottom_cells.size, 0.5 * thickness[0]),
        conductivity=np.full(bottom_cells.size, conductivity[0]),
    )

    die_start = STACK[0].cells + STACK[1].cells
    source = np.zeros((n, 2))
    source[index(nx // 3, ny // 3, die_start), 0] = 1.0
    source[index(2 * nx // 3, 2 * ny // 3, die_start + 1), 1] = 1.0

    probes = np.array(
        [
            index(nx // 2, ny // 2, die_start),  # junction
            index(0, 0, die_start),  # die corner
            index(nx // 2, ny // 2, nz - 1),  # lid centre
            index(nx // 2, ny // 2, 0),  # substrate centre
        ]
    )
    return Package(K=K, C=C, ports=(top, bottom), source=source, probes=probes, shape=(nx, ny, nz))


# ---------------------------------------------------------------------------
# nonlinear interface laws
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RadiationConvection:
    """Lid surface to ambient: q = A [eps*sigma (T^4 - Ta^4) + h (T - Ta)]."""

    port: Port
    emissivity: float = 0.85
    coefficient: float = 6.0  # W/(m^2 K), natural convection
    ambient: float = 300.0  # K

    def flux(self, temperature: np.ndarray) -> np.ndarray:
        t = temperature[self.port.cells]
        return self.port.area * (
            self.emissivity * STEFAN_BOLTZMANN * (t**4 - self.ambient**4)
            + self.coefficient * (t - self.ambient)
        )

    def slope(self, temperature: np.ndarray) -> np.ndarray:
        t = temperature[self.port.cells]
        return self.port.area * (
            4.0 * self.emissivity * STEFAN_BOLTZMANN * t**3 + self.coefficient
        )

    def linearized_conductance(self, reference: float) -> np.ndarray:
        t = np.full(self.port.cells.size, reference)
        return self.port.area * (
            4.0 * self.emissivity * STEFAN_BOLTZMANN * t**3 + self.coefficient
        )


@dataclass(frozen=True)
class ContactConductance:
    """Substrate surface to board through a temperature-dependent contact film.

    ``h(T) = h0 (1 + beta (T - reference))`` in series with the half cell, so the
    per-cell conductance is ``A / (1/h + half/k)``.
    """

    port: Port
    reference_h: float = 1500.0  # W/(m^2 K)
    beta: float = 4.0e-3  # 1/K
    reference: float = 320.0  # K
    board: float = 320.0  # K

    def conductance(self, temperature: np.ndarray) -> np.ndarray:
        h = self.reference_h * (1.0 + self.beta * (temperature[self.port.cells] - self.reference))
        return self.port.area / (1.0 / h + self.port.half / self.port.conductivity)

    def flux(self, temperature: np.ndarray) -> np.ndarray:
        t = temperature[self.port.cells]
        return self.conductance(t) * (t - self.board)

    def slope(self, temperature: np.ndarray) -> np.ndarray:
        t = temperature[self.port.cells]
        h = self.reference_h * (1.0 + self.beta * (t - self.reference))
        resistance = 1.0 / h + self.port.half / self.port.conductivity
        g = self.port.area / resistance
        dg_dh = self.port.area * (1.0 / h**2) / resistance**2
        return dg_dh * self.reference_h * self.beta * (t - self.board) + g

    def linearized_conductance(self, reference: float) -> np.ndarray:
        t = np.full(self.port.cells.size, reference)
        return self.conductance(t)


@dataclass(frozen=True)
class Interface:
    """The declared external coupling: one law per port."""

    laws: tuple[RadiationConvection | ContactConductance, ...]

    def flux(self, temperature: np.ndarray, size: int) -> np.ndarray:
        out = np.zeros(size)
        for law in self.laws:
            np.add.at(out, law.port.cells, law.flux(temperature))
        return out

    def slope(self, temperature: np.ndarray, size: int) -> np.ndarray:
        out = np.zeros(size)
        for law in self.laws:
            np.add.at(out, law.port.cells, law.slope(temperature))
        return out

    def linear_conductance(self, reference: float, size: int) -> np.ndarray:
        out = np.zeros(size)
        for law in self.laws:
            np.add.at(out, law.port.cells, law.linearized_conductance(reference))
        return out


def default_interface(package: Package) -> Interface:
    top, bottom = package.ports
    return Interface(laws=(RadiationConvection(top), ContactConductance(bottom)))
