"""Concrete affine parametric model: an analytic 1-D Fourier rod (the decoupling proof).

This module is *private* — reachable only through the factory under the
registered name ``"toy_1d"``.  It is built entirely from numpy/scipy and
**never imports ``metahotspot``**, so constructing it does not load the C++ DLL.
It exists to prove the experiment scripts are genuinely model-agnostic: the same
Krylov / column-localized experiment code that runs against ``"chiplet_stack"``
runs end-to-end against this rod, unchanged.

Structure mirrors the chiplet stack exactly:

* **detail** = the leftmost rod cell, carrying the volumetric heat source
  (analogous to the die);
* **interface port** = the boundary between the source cell and the rest of the
  rod (analogous to the die-top interface);
* **macro** = the source-free remainder of the rod (cells 1..N-1);
* **boundary group** = the right end of the rod (analogous to the top face).

The DtN assembly mirrors ``mhs::macro::assemble_dtn`` (``modal_port.cpp``): the
base thermal operators ``[FVM cells]`` are embedded at offset ``port_count`` and
each port face couples its cell through conductance ``g = k*A/half = 2k/dx``
(with face area ``A = 1``).  The saturation closure ``g*h*A/(g+h*A)`` the
reduction math builds from the per-group ``(cells, g, areas)`` data then
reproduces the native Robin term ``k*h*A/(k + h*dx/2)`` exactly: ``g = k*A/half``
with ``half = dx/2`` gives ``g*h*A/(g+h*A) = k*h*A/(k + h*dx/2)``.

Because the macro core is source-free, the ambient-balance invariant required by
the Krylov reduction holds (uniform ambient across the ports ⇒ zero residual),
just as it does for the chiplet stack.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import scipy.sparse as sp
from metahotspot.compiled import Operators, SolveOptions

from affine_parametric_models._interfaces import (
    AffineParametricModel,
    AffineSolveResult,
    BoundaryGroup,
    StateLayout,
)
from utils import (
    extract_boundary_groups,
    normalized_operators,
)


@dataclass(frozen=True)
class Toy1DConfig:
    ambient_K: float = 300.0
    n: int = 40  # rod cells (1 detail source cell + n-1 macro cells)
    length_m: float = 1.0e-2  # rod length (m)
    k: float = 100.0  # thermal conductivity (W/m/K)
    rho_c: float = 1.0e6  # volumetric heat capacity (J/m^3/K)
    q_source: float = 1.0e5  # volumetric heat source in the detail cell (W/m^3)
    duration_s: float = 10.0
    dt_s: float = 0.5
    top_h: float = 2500.0  # W/m^2 K on the right-end boundary group
    # reduced-order experiment knobs reused by the scripts via overrides.
    # The toy ignores the geometry knobs but accepts them so the shared
    # QUICK_OVERRIDES dicts (substrate_cells, max_xy_cell_mm, ...) don't crash.
    max_xy_cell_mm: float = 0.0
    substrate_cells: int = 0
    bump_cells: int = 0
    die_cells: int = 0
    tim_cells: int = 0
    spreader_cells: int = 0
    cold_plate_cells: int = 0
    bump_rows: int = 0
    bump_columns: int = 0

    @property
    def dx(self) -> float:
        return self.length_m / self.n

    @property
    def nx(self) -> int:
        return self.n

    @property
    def nz(self) -> int:
        return self.n

    def report_dict(self) -> dict:
        return {**self.__dict__, "nx": self.n, "nz": self.n, "dx_m": self.dx}


def _face_area(cfg: Toy1DConfig) -> float:
    return 1.0  # unit-depth 1-D rod, cross-section 1 x 1


def _conductance(cfg: Toy1DConfig) -> float:
    """k*A/half = k*A/(dx/2) = 2k/dx  (matches ``interface_conductance``)."""
    return cfg.k * _face_area(cfg) / (cfg.dx / 2.0)


def _macro_core(cfg: Toy1DConfig) -> tuple[sp.csc_matrix, sp.csc_matrix, np.ndarray]:
    """Source-free macro rod ``[interface port | cells 1..N-1]``.

    Mirrors ``assemble_dtn``: the rod cells are embedded at offset 1 and the
    interface port couples the left rod cell through conductance ``g``.
    """
    n_macro = cfg.n - 1
    k, dx = cfg.k, cfg.dx
    diag = np.full(n_macro, 2.0 * k / dx)
    off = np.full(n_macro - 1, -k / dx)
    K_rod = sp.diags([off, diag, off], [-1, 0, 1], format="lil")
    K_rod[0, 0] = k / dx  # left macro cell: right neighbor only
    K_rod[n_macro - 1, n_macro - 1] = k / dx  # right macro cell: left neighbor only

    g = _conductance(cfg)
    K = sp.lil_matrix((n_macro + 1, n_macro + 1))
    K[0, 0] = g
    K[0, 1] = -g
    K[1, 0] = -g
    K[1, 1] += g
    K[1:, 1:] += K_rod
    C = sp.diags([0.0] + [cfg.rho_c * dx] * n_macro, format="lil")
    f = np.zeros(n_macro + 1)  # source-free macro

    return K.tocsc(), C.tocsc(), f


def _merged_core(cfg: Toy1DConfig) -> tuple[sp.csc_matrix, sp.csc_matrix, np.ndarray]:
    """``[interface port | boundary port | macro cells]`` for boundary extraction.

    The single boundary group is the right macro cell; both ports couple their
    cell through ``g``, producing exactly the ``+g/-g`` pattern
    :func:`utils.extract_boundary_groups` recovers.
    """
    n_macro = cfg.n - 1
    k, dx = cfg.k, cfg.dx
    diag = np.full(n_macro, 2.0 * k / dx)
    off = np.full(n_macro - 1, -k / dx)
    K_rod = sp.diags([off, diag, off], [-1, 0, 1], format="lil")
    K_rod[0, 0] = k / dx
    K_rod[n_macro - 1, n_macro - 1] = k / dx

    g = _conductance(cfg)
    K = sp.lil_matrix((n_macro + 2, n_macro + 2))
    # interface port 0 <-> left macro cell (merged index 2)
    K[0, 0] = g
    K[0, 2] = -g
    K[2, 0] = -g
    K[2, 2] += g
    # boundary port 1 <-> right macro cell (merged index n_macro+1)
    K[1, 1] = g
    K[1, n_macro + 1] = -g
    K[n_macro + 1, 1] = -g
    K[n_macro + 1, n_macro + 1] += g
    K[2:, 2:] += K_rod

    C = sp.diags([0.0, 0.0] + [cfg.rho_c * dx] * n_macro, format="lil")
    f = np.zeros(n_macro + 2)
    return K.tocsc(), C.tocsc(), f


def _reference_operators(cfg: Toy1DConfig, h: float):
    """Full N-cell rod with source at cell 0 and Robin at cell N-1."""
    n, k, dx = cfg.n, cfg.k, cfg.dx
    diag = np.full(n, 2.0 * k / dx)
    off = np.full(n - 1, -k / dx)
    K = sp.diags([off, diag, off], [-1, 0, 1], format="lil")
    K[0, 0] = k / dx  # left end: right neighbor only
    K[n - 1, n - 1] = k / dx  # right end: left neighbor only
    robin = k * h * _face_area(cfg) / (k + h * dx / 2.0)
    K[n - 1, n - 1] += robin

    f = np.zeros(n)
    f[0] = cfg.q_source * (dx * _face_area(cfg))
    f[n - 1] += robin * cfg.ambient_K
    C = sp.diags([cfg.rho_c * dx] * n, format="csc")
    return K.tocsc(), C.tocsc(), f


class _Toy1D(AffineParametricModel):
    """Private concrete implementation registered as ``"toy_1d"``."""

    def __init__(self, cfg: Toy1DConfig):
        self._cfg = cfg

    # ------------------------------------------------------------- identity

    @property
    def name(self) -> str:
        return "toy_1d"

    @property
    def ambient_K(self) -> float:
        return self._cfg.ambient_K

    # --------------------------------------------------- DtN core extraction

    @property
    def port_count(self) -> int:
        return 1

    @cached_property
    def _core(self) -> Operators:
        K, C, f = _macro_core(self._cfg)
        return normalized_operators(K, C, f)

    def core_operators(self) -> Operators:
        return self._core

    def merged_operators(self, group_sizes) -> Operators:
        K, C, f = _merged_core(self._cfg)
        return normalized_operators(K, C, f)

    @cached_property
    def _boundary_groups(self) -> tuple[BoundaryGroup, ...]:
        merged = self.merged_operators([1])
        ((cells, g),) = extract_boundary_groups(merged, 1, [1])
        return (
            BoundaryGroup(
                name="right",
                cells=cells,
                g=g,
                areas=np.asarray([_face_area(self._cfg)], dtype=np.float64),
                h_default=self._cfg.top_h,
            ),
        )

    def boundary_groups(self) -> tuple[BoundaryGroup, ...]:
        return self._boundary_groups

    @property
    def macro_cell_count(self) -> int:
        return self._cfg.n - 1

    @property
    def macro_nx(self) -> int:
        return 1

    @property
    def macro_ny(self) -> int:
        return 1

    @cached_property
    def macro_grid(self) -> np.ndarray:
        # The macro rod (cells 1..N-1) is a single column; ids are 0..N-2.
        return np.arange(self._cfg.n - 1, dtype=np.int64).reshape(1, 1, self._cfg.n - 1)

    @property
    def dt(self) -> float:
        return self._cfg.dt_s

    @cached_property
    def port_lookup(self) -> dict[tuple[int, int], int]:
        return {(0, 0): 0}

    # --------------------------------------- native reference + recovery

    def full_reference(self, h_vec) -> AffineSolveResult:
        if len(h_vec) != 1:
            raise ValueError("toy_1d has exactly one boundary group")
        h = float(h_vec[0])
        n = self._cfg.n
        ambient = self._cfg.ambient_K

        K, C, f = _reference_operators(self._cfg, h)

        started = time.perf_counter()
        steady = np.asarray(sp.linalg.spsolve(K, f))
        steady_s = time.perf_counter() - started

        n_steps = int(round(self._cfg.duration_s / self._cfg.dt_s))
        times = self._cfg.dt_s * np.arange(1, n_steps + 1)
        A = sp.csc_matrix(C / self._cfg.dt_s + K)
        factor = sp.linalg.factorized(A)
        state = np.full(n, ambient)
        history = np.empty((n_steps, n))
        started = time.perf_counter()
        rhs = C @ state / self._cfg.dt_s + f
        for k in range(n_steps):
            state = factor(rhs)
            history[k] = state
            rhs = C @ state / self._cfg.dt_s + f
        transient_s = time.perf_counter() - started

        return AffineSolveResult(
            steady_temperature=steady,
            times=times,
            history=history,
            compile_s=0.0,
            steady_s=steady_s,
            transient_s=transient_s,
            full_order=n,
        )

    def state_layout(self, internal_count: int) -> StateLayout:
        return StateLayout(
            detail_count=1, port_count=self.port_count, internal_count=internal_count
        )

    def solve_reduced(self, operators: Operators, state, transient: bool):
        """Solve ``operators`` coupled to the source-bearing detail cell.

        Mirrors ``mhs::macro::assemble_coupled`` + ``solve_system`` in pure
        numpy: state = ``[detail source cell | port | internal]``.  The detail
        cell carries the volumetric source and couples to the port (leading
        state of the reduced macro operator) through conductance ``g``; the
        port in turn couples to the macro internal modes through ``operators``.
        The series of the two ``g`` gives the physical cell-to-cell conduction
        ``k/dx``, exactly as in the C++ coupling.
        """
        K, C, f = operators
        n_internal = K.shape[0] - 1  # port_count == 1
        dx = self._cfg.dx
        g = _conductance(self._cfg)

        # [detail | port | internal]
        total = 1 + 1 + n_internal
        K_c = sp.lil_matrix((total, total))
        K_c[0, 0] = g
        K_c[0, 1] = -g
        K_c[1, 0] = -g
        K_c[1, 1] += g
        K_c[1:, 1:] += K
        C_c = sp.diags([self._cfg.rho_c * dx, 0.0] + [0.0] * n_internal, format="lil")
        C_c[1:, 1:] += C
        f_c = np.zeros(total)
        f_c[0] = self._cfg.q_source * (dx * _face_area(self._cfg))
        f_c[1:] += f
        K_c = K_c.tocsc()
        C_c = C_c.tocsc()

        if transient:
            n_steps = int(round(self._cfg.duration_s / self._cfg.dt_s))
            times = self._cfg.dt_s * np.arange(1, n_steps + 1)
            A = sp.csc_matrix(C_c / self._cfg.dt_s + K_c)
            factor = sp.linalg.factorized(A)
            state = np.asarray(state, dtype=np.float64)
            history = np.empty((n_steps, total))
            started = time.perf_counter()
            rhs = C_c @ state / self._cfg.dt_s + f_c
            for k in range(n_steps):
                state = factor(rhs)
                history[k] = state
                rhs = C_c @ state / self._cfg.dt_s + f_c
            elapsed = time.perf_counter() - started
            return times, history, elapsed
        started = time.perf_counter()
        steady = np.asarray(sp.linalg.spsolve(K_c, f_c))
        return steady, time.perf_counter() - started

    @cached_property
    def _detail_to_full(self) -> np.ndarray:
        # Detail = cell 0 (the source cell) of the full rod.
        return np.asarray([0], dtype=np.int64)

    @cached_property
    def _macro_to_full(self) -> np.ndarray:
        # Macro = rod cells 1..N-1.
        return np.arange(1, self._cfg.n, dtype=np.int64)

    def detail_to_full(self) -> np.ndarray:
        return self._detail_to_full

    def macro_to_full(self) -> np.ndarray:
        return self._macro_to_full

    @property
    def full_cell_count(self) -> int:
        return self._cfg.n

    @property
    def detail_cell_count(self) -> int:
        return 1

    def recover_temperature(
        self, states, *, basis, ports: int, ambient_K: float | None
    ) -> np.ndarray:
        states = np.atleast_2d(states)
        temperature = np.empty((states.shape[0], self.full_cell_count))
        temperature[:, self.detail_to_full()] = states[:, : self.detail_cell_count]
        internal = (basis @ states[:, self.detail_cell_count + ports :].T).T
        temperature[:, self.macro_to_full()] = (
            internal if ambient_K is None else ambient_K + internal
        )
        return temperature

    def monitor_cells(self) -> np.ndarray:
        return np.asarray([0], dtype=np.int64)

    def monitor_full(self, detail_cells: np.ndarray) -> np.ndarray:
        return np.asarray(detail_cells, dtype=np.int64)

    def report_dict(self) -> dict:
        return self._cfg.report_dict()

    def solver_options(self, transient: bool):

        dt = self._cfg.dt_s if transient else 1.0
        return SolveOptions(
            linear_solver="EigenSparseLU",
            linear_tolerance=1.0e-12,
            linear_max_iterations=5000,
            nonlinear_max_iterations=30,
            nonlinear_relative_tolerance=1.0e-11,
            nonlinear_absolute_tolerance=1.0e-11,
            integrator="Bdf1",
            step_strategy="Fixed",
            error_rel_tol=1.0e-3,
            min_dt=dt,
            max_dt=dt,
            fixed_dt=dt,
        )


def _builder(overrides: dict | None = None, **_kwargs) -> AffineParametricModel:
    cfg = Toy1DConfig(**(overrides or {}))
    return _Toy1D(cfg)
