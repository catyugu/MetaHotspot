#!/usr/bin/env python3
"""A lid bonded to a smaller die on a larger base -- two layers of different footprint.

This is the realistic feature the previous die-on-a-big-Cu-block geometry lacked: the
contact window is a *footprint*, so the smaller layer's faces are only partly covered by
common patches and the lid rim overhangs into the cavity.  One physics, several meshes:

* :func:`build_reference` meshes the whole stack on one conforming grid (error reference);
* :func:`build_blocks` meshes it as a *component* block (base + die) and an *envelope*
  block (lid) whose interface grids may deliberately differ -- the repository's
  non-conforming coupling.

Stack (rise coordinates, ambient = 0 K, SI units), bottom to top:

    z = 0.0 mm    base bottom      h_board on the whole 24 x 24 footprint
    z = 1.0 mm    die attach plane: die 10 x 10 on the base (base top ring is exposed)
    z = 1.4 mm    INTERFACE        die top 10 x 10  <->  lid bottom (lid is 16 x 16)
    z = 2.9 mm    lid top          h_lid (cold plate)

The die is 10 x 10 x 0.4 mm, the lid is 16 x 16 x 1.5 mm, so the lid overhangs the die by
3 mm on every side: its bottom faces outside the die see the cavity air at ``h_cavity``,
and the cells that straddle the die edge are only partly covered by common patches
(``xi < 1``).  Power is one 2 x 2 mm hotspot plus a uniform background, both in the die.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from nc_interface import Box, FacePort, Grid, System, assemble, pad_sources, window_port

MM = 1.0e-3


def _constant(k: float, rho_cp: float):
    return lambda x, y: (np.full(x.shape, k), np.full(x.shape, rho_cp))


def _square(half: float):
    return lambda x, y: (np.abs(x) <= half + 1.0e-12) & (np.abs(y) <= half + 1.0e-12)


def _everywhere(x, y):
    return np.ones(x.shape, dtype=bool)


@dataclass(frozen=True)
class LayerSpec:
    """One z-slab: a vectorized material map and the footprint it occupies."""

    name: str
    dz: float
    cells: int
    material: object  # (x, y) -> (k array, rho_cp array)
    footprint: object  # (x, y) -> bool array


@dataclass(frozen=True)
class Physics:
    """Geometry, materials, source and ambient coefficients of the die-on-base case."""

    base_xy: float = 24.0 * MM
    base_t: float = 1.0 * MM
    base_cells: int = 2
    k_base: float = 2.0
    cp_base: float = 2.00e6
    die_xy: float = 10.0 * MM
    die_t: float = 0.4 * MM
    die_cells: int = 2
    k_die: float = 130.0
    cp_die: float = 1.63e6
    lid_xy: float = 16.0 * MM
    lid_t: float = 1.5 * MM
    lid_cells: int = 3
    k_lid: float = 385.0
    cp_lid: float = 3.44e6
    source_xy: float = 2.0 * MM
    power_W: float = 6.0
    hotspot_W: float = 4.0
    h_board: float = 10.0
    h_lid: float = 3000.0
    h_side: float = 10.0
    h_cavity: float = 8.0
    tol: float = 1.0e-9

    # ---------------------------------------------------------------- derived
    @property
    def interface_z(self) -> float:
        return self.base_t + self.die_t

    @property
    def top_z(self) -> float:
        return self.interface_z + self.lid_t

    @property
    def background_W(self) -> float:
        return self.power_W - self.hotspot_W

    @property
    def component_z_cells(self) -> int:
        return self.base_cells + self.die_cells

    # ---------------------------------------------------------------- layers
    def component_layers(self) -> list[LayerSpec]:
        """Base then die (bottom to top); the die sits on the base's top face."""
        return [
            LayerSpec(
                "base",
                self.base_t,
                self.base_cells,
                _constant(self.k_base, self.cp_base),
                _everywhere,
            ),
            LayerSpec(
                "die",
                self.die_t,
                self.die_cells,
                _constant(self.k_die, self.cp_die),
                _square(self.die_xy / 2),
            ),
        ]

    def envelope_layers(self) -> list[LayerSpec]:
        return [
            LayerSpec(
                "lid",
                self.lid_t,
                self.lid_cells,
                _constant(self.k_lid, self.cp_lid),
                _square(self.lid_xy / 2),
            ),
        ]

    def full_layers(self) -> list[LayerSpec]:
        return self.component_layers() + self.envelope_layers()

    # ---------------------------------------------------------------- source
    def sources(self) -> list[Box]:
        half = self.source_xy / 2
        z0, z1 = self.base_t, self.interface_z
        return [
            Box("hotspot", -half, half, -half, half, z0, z1, self.hotspot_W),
            Box(
                "background",
                -self.die_xy / 2,
                self.die_xy / 2,
                -self.die_xy / 2,
                self.die_xy / 2,
                z0,
                z1,
                self.background_W,
            ),
        ]

    # ---------------------------------------------------------------- ambient
    def _classify(self, axis, x, y, z):
        """Ambient HTC of an exposed face at the given position."""
        if axis == 2:
            return np.where(
                z <= self.tol,
                self.h_board,
                np.where(z >= self.top_z - self.tol, self.h_lid, self.h_cavity),
            )
        return np.full_like(z, self.h_side)

    def ambient(self, axis, direction, x, y, z):
        """Reference: the lid rim and the base top ring are exposed faces."""
        return self._classify(axis, x, y, z)

    def _port_mask(self, x, y, z):
        half = self.die_xy / 2
        return (
            (np.abs(x) <= half + self.tol)
            & (np.abs(y) <= half + self.tol)
            & (np.abs(z - self.interface_z) <= self.tol)
        )

    def component_ambient(self, axis, direction, x, y, z):
        """Component block: the die top is the port, not an ambient face."""
        return np.where(self._port_mask(x, y, z), 0.0, self._classify(axis, x, y, z))

    def envelope_ambient(self, axis, direction, x, y, z):
        """Envelope block: the lid bottom is the port where it meets the die."""
        return np.where(self._port_mask(x, y, z), 0.0, self._classify(axis, x, y, z))

    def homogeneous(self, k: float = 130.0, rho_cp: float = 1.6e6) -> "Physics":
        """Same geometry with one material everywhere (analytic checks only)."""
        return replace(
            self,
            k_base=k,
            k_die=k,
            k_lid=k,
            cp_base=rho_cp,
            cp_die=rho_cp,
            cp_lid=rho_cp,
        )


# ---------------------------------------------------------------------------
# grids
# ---------------------------------------------------------------------------


def graded_axis(half: float, fine: float, window_half: float) -> np.ndarray:
    """Symmetric 1-D edges: uniform ``fine`` inside +-``window_half``, 1.25x growth out."""
    n = int(round(2.0 * window_half / fine))
    inner = np.linspace(-window_half, window_half, n + 1)

    def walk(start: float, stop: float, first: float) -> np.ndarray:
        sign = 1.0 if stop > start else -1.0
        edges = [start]
        step = first
        while abs(stop - edges[-1]) > 1.0e-12:
            step = min(step * 1.2, 1.0 * MM)
            edges.append(edges[-1] + sign * min(step, abs(stop - edges[-1])))
        return np.asarray(edges)

    return np.unique(
        np.r_[
            walk(-window_half, -half, inner[1] - inner[0])[::-1],
            inner,
            walk(window_half, half, inner[-1] - inner[-2]),
        ]
    )


def _graded_lid(half: float) -> np.ndarray:
    sizes = 0.4 * MM * 1.2 ** np.arange(16)
    sizes *= half / sizes.sum()
    edges = np.r_[0.0, np.cumsum(sizes)]
    return np.r_[-edges[::-1], edges[1:]]


def envelope_axis(phys: Physics, pattern: str) -> np.ndarray:
    """The lid's in-plane edges over its own footprint."""
    half = phys.lid_xy / 2
    table = {
        "env_0p5": lambda: np.linspace(-half, half, 33),
        "env_1p0": lambda: np.linspace(-half, half, 17),
        "env_2p0": lambda: np.linspace(-half, half, 9),
        "env_1p2_offset": lambda: np.r_[
            -half, -half + 0.6 * MM + 1.2 * MM * np.arange(13), half
        ],
        "env_graded": lambda: _graded_lid(half),
        "env_0p25": lambda: np.linspace(-half, half, 65),
        "env_0p25_offset": lambda: np.r_[
            -half, -half + 0.125 * MM + 0.25 * MM * np.arange(64), half
        ],
        "env_1p0_offset": lambda: np.r_[
            -half, -half + 0.5 * MM + 1.0 * MM * np.arange(15), half
        ],
        "env_0p5_offset": lambda: np.r_[
            -half, -half + 0.25 * MM + 0.5 * MM * np.arange(31), half
        ],
    }
    return table[pattern]()


# Interface-grid variants.  The key spells out the two in-plane spacings and how the
# lid's grid lines sit relative to the die's:
#   same      both sides share one grid (control)
#   aligned   every lid cell covers a whole number of die faces (lid lines are a subset
#             of the die lines)
#   shifted   the lid grid is offset by half a cell, so no lines coincide
#   graded    the lid grid is 0.4 mm at the centre growing geometrically to the rim
#   fine      the lid is much finer than the die (the coarse-component / real EROM case)
VARIANTS: dict[str, tuple[float, str]] = {
    "die0p5_lid0p5_same": (0.5 * MM, "env_0p5"),
    "die0p5_lid1p0_aligned": (0.5 * MM, "env_1p0"),
    "die0p5_lid2p0_aligned": (0.5 * MM, "env_2p0"),
    "die0p5_lid1p2_shifted": (0.5 * MM, "env_1p2_offset"),
    "die0p5_lid_graded": (0.5 * MM, "env_graded"),
    "die0p25_lid0p25_shifted": (0.25 * MM, "env_0p25_offset"),
    "die1p0_lid0p25_fine": (1.0 * MM, "env_0p25"),
    "die2p0_lid0p25_fine": (2.0 * MM, "env_0p25"),
}

VARIANT_NOTES: dict[str, str] = {
    "die0p5_lid0p5_same": "0.5 mm on both sides, same lines (control)",
    "die0p5_lid1p0_aligned": "lid 1.0 mm: each lid cell = 2x2 die faces",
    "die0p5_lid2p0_aligned": "lid 2.0 mm: each lid cell = 4x4 die faces",
    "die0p5_lid1p2_shifted": "lid 1.2 mm, lines shifted off the die lines",
    "die0p5_lid_graded": "lid 0.4->2.3 mm, graded from the centre",
    "die0p25_lid0p25_shifted": "both 0.25 mm, lid shifted by half a cell",
    "die1p0_lid0p25_fine": "die 1.0 mm against a 0.25 mm lid",
    "die2p0_lid0p25_fine": "die 2.0 mm against a 0.25 mm lid",
}


def grid_of_layers(
    name: str, xs: np.ndarray, ys: np.ndarray, layers: list[LayerSpec], *, z0: float
) -> tuple[Grid, np.ndarray]:
    """Build a grid from layer specs, returning the per-cell material and the mask."""
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    z = [z0]
    for layer in layers:
        for step in range(1, layer.cells + 1):
            z.append(z[-1] + layer.dz / layer.cells)
    z = np.asarray(z)
    nx, ny = xs.size - 1, ys.size - 1
    nz = sum(layer.cells for layer in layers)
    xc = 0.5 * (xs[:-1] + xs[1:])
    yc = 0.5 * (ys[:-1] + ys[1:])
    xg, yg = np.meshgrid(xc, yc, indexing="ij")
    k = np.empty((nx, ny, nz))
    rho_cp = np.empty((nx, ny, nz))
    active = np.zeros((nx, ny, nz), dtype=bool)
    start = 0
    for layer in layers:
        kk, cc = layer.material(xg, yg)
        footprint = layer.footprint(xg, yg)
        for index in range(start, start + layer.cells):
            k[:, :, index] = kk
            rho_cp[:, :, index] = cc
            active[:, :, index] = footprint
        start += layer.cells
    return (
        Grid(
            name,
            xs,
            ys,
            z,
            np.ascontiguousarray(k.reshape(-1)),
            np.ascontiguousarray(rho_cp.reshape(-1)),
        ),
        active,
    )


def build_reference(phys: Physics, h_xy: float) -> System:
    """One conforming mesh of the whole stack: the error reference."""
    half = phys.base_xy / 2
    n = int(round(phys.base_xy / h_xy))
    edges = np.linspace(-half, half, n + 1)
    grid, active = grid_of_layers("reference", edges, edges, phys.full_layers(), z0=0.0)
    return assemble(
        grid,
        active=active,
        ambient=phys.ambient,
        sources=phys.sources(),
        name="reference",
    )


def with_contact(port: FacePort, contact_h: float) -> FacePort:
    """Add a series surface film ``1/contact_h`` to a port's per-face conductance."""
    return replace(port, half=port.half + port.k / contact_h)


def build_blocks(
    phys: Physics,
    variant: str,
    *,
    component_h: float | None = None,
    contact_h: float | None = None,
) -> tuple[System, System, FacePort, FacePort]:
    """Assemble the component and envelope blocks with their interface ports."""
    default_h, pattern = VARIANTS[variant]
    fine = default_h if component_h is None else component_h

    xs = graded_axis(phys.base_xy / 2, fine, phys.die_xy / 2)
    comp_grid, comp_active = grid_of_layers(
        "component", xs, xs, phys.component_layers(), z0=0.0
    )
    comp = assemble(
        comp_grid,
        active=comp_active,
        ambient=phys.component_ambient,
        sources=phys.sources(),
        name="component",
    )

    ys = envelope_axis(phys, pattern)
    env_grid, env_active = grid_of_layers(
        "envelope", ys, ys, phys.envelope_layers(), z0=phys.interface_z
    )
    env = assemble(
        env_grid, active=env_active, ambient=phys.envelope_ambient, name="envelope"
    )

    half = phys.die_xy / 2
    window = (-half, half, -half, half)
    comp_port = window_port(comp, window, axis=2, direction=+1, label="z+")
    env_port = window_port(env, window, axis=2, direction=-1, label="z-")
    if contact_h is not None:
        env_port = with_contact(env_port, contact_h)
    pad_sources(comp, env)
    return comp, env, comp_port, env_port


def apply_contact_film(system: System, phys: Physics, contact_h: float | None) -> None:
    """Add a series surface film on the reference's interface plane, in place."""
    if contact_h is None:
        return
    n_below = phys.component_z_cells
    halves = system.halves()
    k = system.conductivity()
    spacing = system.grid.spacings()
    area = np.outer(spacing[0], spacing[1])
    ok = (system.ids[:, :, n_below - 1] >= 0) & (system.ids[:, :, n_below] >= 0)
    below = np.where(ok, system.ids[:, :, n_below - 1], 0)
    above = np.where(ok, system.ids[:, :, n_below], 0)
    g_old = area / (halves[below, 2] / k[below] + halves[above, 2] / k[above])
    g_new = area / (
        halves[below, 2] / k[below] + 1.0 / contact_h + halves[above, 2] / k[above]
    )
    delta = (g_old - g_new).ravel()
    K = system.K.tolil()
    for cb, ca, d, use in zip(below.ravel(), above.ravel(), delta, ok.ravel()):
        if not use or d == 0.0:
            continue
        K[cb, cb] -= d
        K[ca, ca] -= d
        K[cb, ca] += d
        K[ca, cb] += d
    system.K = K.tocsc()
