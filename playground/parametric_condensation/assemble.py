#!/usr/bin/env python3
"""Assembly of extracted components, plus the BCI parameterisation.

Two facts make this design work, and both are checked in ``test_assembly.py``:

1. Each component is extracted with its exact static constraint modes, so
   eliminating a component's interior is exact at steady state.
2. An interface couples through one independent massless node per face pair,
   each side contributing only its own half-cell conductance.  Eliminating those
   nodes reproduces the monolithic face conductance
   ``A / (half_l/k_l + half_r/k_r)``, so identity coupling reproduces the
   monolithic solve.

**The BCI parameter is a port-diagonal term, not an extraction parameter.**  The
ambient coefficient ``h`` enters as ``h A`` on the ambient face cells, which are
port DOFs.  A component's reduced interface block is exactly ``S_r(s)``, so the
assembled operator is ``S_r(s) + diag(h A)`` for every ``h``: no re-extraction is
needed across the BCI box.  ``experiment.py`` measures this instead of assuming
it.

Every component block carries a **port map** ``M`` with ``T_ports = M theta``.
Without port compression ``M = I`` and the interface relation is exact; with
compression ``M = P`` the interface law can only be evaluated at ``P theta``,
which is exactly the cost this study wants to expose.  The ambient groups keep
their own ``trace`` matrix so both cases run through one code path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

import package_assembly as pa
import reduction as rd
from linalg import Cost

AMBIENT_LAW = {
    "lid": dict(emissivity=0.85, coefficient=25.0, temperature=300.0),
    "substrate": dict(emissivity=0.0, coefficient=3000.0, temperature=320.0),
}
BCI_BOX = {"lid": (5.0, 5.0e3), "substrate": (50.0, 2.0e4)}


@dataclass(frozen=True)
class Operators:
    K: sp.csc_matrix
    C: sp.csc_matrix


@dataclass(frozen=True)
class AmbientGroup:
    """One ambient face set: linear ``h A (T - Ta)`` plus fixed radiation.

    ``trace`` is ``(n_face x n_state)``; without port compression it is a
    selection matrix and the group acts diagonally.
    """

    label: str
    trace: sp.csr_matrix
    area: np.ndarray
    emissivity: float
    temperature: float
    coefficient: float

    def _face_temperature(self, state):
        return self.trace @ state

    def flux(self, state: np.ndarray) -> np.ndarray:
        t = self._face_temperature(state)
        radiated = self.emissivity * pa.STEFAN_BOLTZMANN * (t**4 - self.temperature**4)
        return self.trace.T @ (self.area * (radiated + self.coefficient * (t - self.temperature)))

    def jacobian(self, state: np.ndarray) -> sp.csc_matrix:
        t = self._face_temperature(state)
        slope = self.area * (
            4.0 * self.emissivity * pa.STEFAN_BOLTZMANN * t**3 + self.coefficient
        )
        return (self.trace.T @ sp.diags(slope) @ self.trace).tocsc()

    def linear_term(self) -> sp.csc_matrix:
        """The ``h A`` contribution to the operator (no radiation)."""
        weight = self.area * self.coefficient
        return (self.trace.T @ sp.diags(weight) @ self.trace).tocsc()


@dataclass
class System:
    """A thermal system whose only nonlinearity sits on the ambient ports."""

    label: str
    K: sp.csc_matrix
    C: sp.csc_matrix
    source: np.ndarray
    probe_map: sp.csr_matrix
    ambient: tuple[AmbientGroup, ...]
    constant: np.ndarray  # state corresponding to a uniform temperature of 1 K
    port_groups: dict[str, sp.csr_matrix] = field(default_factory=dict)
    dimension: int = 0

    def __post_init__(self):
        self.dimension = int(self.K.shape[0])

    def initial(self, ambient: float) -> np.ndarray:
        """State for a uniform ``ambient`` field.

        Enrichment coordinates are modal amplitudes, not temperatures: a uniform
        field is ``Psi @ ambient`` in the interior, so they must start at zero.
        Starting them at ``ambient`` injects a spurious jump of ``ambient`` times
        the enrichment block and ruins the early transient.
        """
        return ambient * self.constant

    def ports(self, state: np.ndarray) -> dict[str, np.ndarray]:
        """Physical port temperature per declared face."""
        return {label: trace @ state for label, trace in self.port_groups.items()}

    def flux(self, state: np.ndarray) -> np.ndarray:
        out = np.zeros(self.dimension)
        for group in self.ambient:
            out += group.flux(state)
        return out

    def jacobian(self, state: np.ndarray) -> sp.csc_matrix:
        operator = self.K.tocsc()
        for group in self.ambient:
            operator = operator + group.jacobian(state)
        return operator.tocsc()

    def linear_operator(self, coefficients: dict[str, float], shift: float = 0.0):
        """``K + sum_g h_g A_g + shift C``: the BCI-parameterised operator."""
        operator = self.K.tocsc()
        for group in self.ambient:
            h = coefficients.get(group.label, group.coefficient)
            weight = group.area * h
            operator = operator + (group.trace.T @ sp.diags(weight) @ group.trace).tocsc()
        if shift:
            operator = operator + shift * self.C
        return operator.tocsc()


def _ambient_groups(assembly, coefficients, build_trace) -> tuple[AmbientGroup, ...]:
    groups = []
    for component in assembly.components:
        if not component.ambient:
            continue
        law = AMBIENT_LAW[component.name]
        for face in component.ambient:
            groups.append(
                AmbientGroup(
                    label=component.name,
                    trace=build_trace(component, face),
                    area=face.area,
                    emissivity=law["emissivity"],
                    temperature=law["temperature"],
                    coefficient=coefficients.get(component.name, law["coefficient"]),
                )
            )
    return tuple(groups)


# ---------------------------------------------------------------------------
# monolithic reference
# ---------------------------------------------------------------------------


def monolithic(assembly: pa.Assembly, coefficients: dict[str, float]) -> System:
    """The whole package solved as one system; the reference for everything."""
    def build_trace(component, face):
        rows = np.arange(face.ij.shape[0])
        cols = np.asarray(component.cells[face.cells], dtype=np.int64)
        return sp.csr_matrix((np.ones(rows.size), (rows, cols)),
                             shape=(rows.size, assembly.cell_count))

    probe_map = sp.lil_matrix((assembly.probes.size, assembly.cell_count))
    for p, cell in enumerate(assembly.probes):
        probe_map[p, int(cell)] = 1.0

    port_groups = {}
    for component in assembly.components:
        for face in component.interfaces + component.ambient:
            rows = np.arange(face.ij.shape[0])
            cols = np.asarray(component.cells[face.cells], dtype=np.int64)
            port_groups[f"{component.name}_{face.label}"] = sp.csr_matrix(
                (np.ones(rows.size), (rows, cols)),
                shape=(rows.size, assembly.cell_count),
            )

    return System(
        label="monolithic",
        K=assembly.K.tocsc(),
        C=assembly.C.tocsc(),
        source=assembly.source,
        probe_map=probe_map.tocsr(),
        ambient=_ambient_groups(assembly, coefficients, build_trace),
        constant=np.ones(assembly.cell_count),
        port_groups=port_groups,
    )


# ---------------------------------------------------------------------------
# per-component extraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReducedComponent:
    name: str
    K: np.ndarray
    C: np.ndarray
    V: np.ndarray
    port_map: np.ndarray  # (n_physical_port, m)
    port_face: dict[str, slice]
    m: int
    q: int

    @property
    def dimension(self) -> int:
        return self.m + self.q

    def port_index(self, label: str, k: int) -> int:
        """Position of face cell ``k`` in the component's physical port list."""
        return self.port_face[label].start + k


def build_plans(component: pa.Component, shifts, cost: Cost | None = None):
    """Response plans for one component: interface coupling only, and with sources.

    ``C`` is diagonal, so ``C_iG = 0`` and the interface coupling block
    ``A_iG(s) = K_iG`` carries no frequency dependence.  The plan can therefore
    pass one fixed right-hand side for all shifts, which makes the
    source-augmented plan exact rather than approximate.
    """
    operators = Operators(K=component.K.tocsc(), C=component.C.tocsc())
    part = rd.partition(operators, component.ports)
    interface = part.KiG.toarray()
    sources = np.asarray(component.source)[part.interior, :]
    plan_interface = rd.ResponsePlan.build(part, shifts, cost)
    plan_source = rd.ResponsePlan.build(
        part, shifts, cost, inputs=np.column_stack((interface, sources))
    )
    return plan_interface, plan_source


def extract(component: pa.Component, *, method: str, order: int,
            plan: rd.ResponsePlan | None = None,
            plan_source: rd.ResponsePlan | None = None,
            cost: Cost | None = None,
            port_basis: np.ndarray | None = None) -> ReducedComponent:
    """Static condensation plus interior enrichment for one component."""
    operators = Operators(K=component.K.tocsc(), C=component.C.tocsc())
    part = rd.partition(operators, component.ports)
    psi = rd.static_modes(part, cost)

    if method == "guyan":
        enrichment = np.empty((part.interior_count, 0))
    elif method == "craig_bampton":
        enrichment = rd.craig_bampton(part, order)
    elif method == "rational_krylov":
        if plan is None:
            raise ValueError("rational_krylov needs an interface plan")
        enrichment = rd.rational_krylov(part, order, plan)
    elif method == "rk_source":
        if plan_source is None:
            raise ValueError("rk_source needs a source-augmented plan")
        enrichment = rd.rational_krylov(part, order, plan_source)
    else:
        raise ValueError(f"unknown method {method!r}")

    model = rd.assemble(operators, part, enrichment, psi)
    m_physical = part.port_count
    port_map = np.eye(m_physical)

    faces: dict[str, slice] = {}
    offset = 0
    for face in component.interfaces + component.ambient:
        faces[face.label] = slice(offset, offset + face.cells.size)
        offset += face.cells.size

    if port_basis is not None:
        q = model.interior_dimension
        block = np.zeros((model.dimension, port_basis.shape[1] + q))
        block[:m_physical, : port_basis.shape[1]] = port_basis
        block[m_physical:, port_basis.shape[1]:] = np.eye(q)
        model = rd.Reduced(
            K=block.T @ model.K @ block,
            C=block.T @ model.C @ block,
            V=model.V @ block,
            interface=model.interface,
            port_count=port_basis.shape[1],
        )
        port_map = port_basis

    return ReducedComponent(
        name=component.name,
        K=np.asarray(model.K),
        C=np.asarray(model.C),
        V=np.asarray(model.V),
        port_map=np.asarray(port_map),
        port_face=faces,
        m=model.port_count,
        q=model.interior_dimension,
    )


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------


def assemble(assembly: pa.Assembly, models: dict[str, ReducedComponent], *,
             coefficients: dict[str, float], label: str) -> System:
    """Couple reduced components through independent interface nodes."""
    offsets: dict[str, int] = {}
    cursor = 0
    for component in assembly.components:
        offsets[component.name] = cursor
        cursor += models[component.name].dimension
    node_start = cursor

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    values: list[np.ndarray] = []

    def push(block_rows, block_cols, block_values):
        rows.append(np.asarray(block_rows, dtype=np.int64).ravel())
        cols.append(np.asarray(block_cols, dtype=np.int64).ravel())
        values.append(np.asarray(block_values, dtype=np.float64).ravel())

    for component in assembly.components:
        model = models[component.name]
        base = offsets[component.name]
        dense = model.K
        nz = np.flatnonzero(dense != 0.0)
        push(base + nz // dense.shape[0], base + nz % dense.shape[1], dense.ravel()[nz])

    node = node_start
    for lower, upper in pa.INTERFACES:
        low_component = next(c for c in assembly.components if c.name == lower)
        high_component = next(c for c in assembly.components if c.name == upper)
        low_face = next(f for f in low_component.interfaces if f.label == f"to_{upper}")
        high_face = next(f for f in high_component.interfaces if f.label == f"to_{lower}")
        if not np.array_equal(low_face.ij, high_face.ij):
            raise RuntimeError(f"interface {lower}-{upper} is not conforming")

        low_model, high_model = models[lower], models[upper]
        low_base, high_base = offsets[lower], offsets[upper]
        for k in range(low_face.ij.shape[0]):
            g_low = float(low_face.conductance()[k])
            g_high = float(high_face.conductance()[k])
            a_local = low_model.port_index(low_face.label, k)
            b_local = high_model.port_index(high_face.label, k)
            # Each side contributes its own half-cell conductance to its own
            # diagonal; the node carries the sum.  Eliminating the node then
            # gives the series conductance, which equals the monolithic face
            # conductance (checked in test_assembly.py).  With a compressed port
            # the contribution is spread over the retained coordinates.
            row_low = np.flatnonzero(low_model.port_map[a_local])
            row_high = np.flatnonzero(high_model.port_map[b_local])
            p_low = low_model.port_map[a_local, row_low]
            p_high = high_model.port_map[b_local, row_high]
            low_coeff = g_low * p_low
            high_coeff = g_high * p_high
            low_cols = low_base + row_low
            high_cols = high_base + row_high
            # The half-cell conductance lands on the port as the OUTER PRODUCT
            # ``g p p^T`` of its coupling vector.  For a full port set ``p`` is a
            # unit vector and this is a scalar diagonal term; for a compressed
            # port it is a small dense block, and dropping its off-diagonal
            # entries makes the assembled steady state wrong.
            for target_cols, vector, conductance in (
                (low_cols, p_low, g_low), (high_cols, p_high, g_high)
            ):
                k = target_cols.size
                push(np.repeat(target_cols, k), np.tile(target_cols, k),
                     conductance * np.outer(vector, vector).ravel())
            # Symmetric off-diagonal block: one row per retained coordinate on
            # each side, so the row list must repeat the node index.
            push(np.r_[low_cols, high_cols, np.full(row_low.size, node),
                       np.full(row_high.size, node)],
                 np.r_[np.full(row_low.size, node), np.full(row_high.size, node),
                       low_cols, high_cols],
                 np.r_[-low_coeff, -high_coeff, -low_coeff, -high_coeff])
            push([node], [node], [g_low + g_high])
            node += 1

    n = node
    K = sp.coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    ).tocsc()

    capacity_rows, capacity_cols, capacity_values = [], [], []
    for component in assembly.components:
        model = models[component.name]
        base = offsets[component.name]
        dense = model.C
        nz = np.flatnonzero(dense != 0.0)
        capacity_rows.append(base + nz // dense.shape[0])
        capacity_cols.append(base + nz % dense.shape[1])
        capacity_values.append(dense.ravel()[nz])
    C = sp.coo_matrix(
        (np.concatenate(capacity_values),
         (np.concatenate(capacity_rows), np.concatenate(capacity_cols))),
        shape=(n, n),
    ).tocsc()

    def build_trace(component, face):
        model = models[component.name]
        base = offsets[component.name]
        rows, cols, data = [], [], []
        for k in range(face.ij.shape[0]):
            physical = model.port_index(face.label, k)
            for j in np.flatnonzero(model.port_map[physical]):
                rows.append(k)
                cols.append(base + j)
                data.append(model.port_map[physical, j])
        return sp.csr_matrix((data, (rows, cols)), shape=(face.ij.shape[0], n))

    source = np.zeros((n, assembly.source.shape[1]))
    probe_map = sp.lil_matrix((assembly.probes.size, n))
    position = {int(cell): p for p, cell in enumerate(assembly.probes)}
    for component in assembly.components:
        model = models[component.name]
        base = offsets[component.name]
        source[base : base + model.dimension, :] = model.V.T @ component.source
        for local in component.probes:
            p = position[int(component.cells[local])]
            for j, value in enumerate(model.V[local, :]):
                if value != 0.0:
                    probe_map[p, base + j] = value

    port_groups = {}
    for component in assembly.components:
        model = models[component.name]
        for face in component.interfaces + component.ambient:
            port_groups[f"{component.name}_{face.label}"] = build_trace(component, face)

    # A uniform temperature field is 1 on every port and interface node and 0 on
    # every enrichment coordinate.
    constant = np.zeros(n)
    for component in assembly.components:
        base = offsets[component.name]
        model = models[component.name]
        constant[base : base + model.m] = 1.0
    constant[node_start:] = 1.0

    return System(
        label=label,
        K=K,
        C=C,
        source=source,
        probe_map=probe_map.tocsr(),
        ambient=_ambient_groups(assembly, coefficients, build_trace),
        constant=constant,
        port_groups=port_groups,
    )
