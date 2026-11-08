"""High-level wrapper for ``mhs_model_t`` — model construction."""

from __future__ import annotations

import ctypes
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from metahotspot._error import check
from metahotspot._handle import OwnedHandle
from metahotspot._lib import get_dll
from metahotspot.enums import Axis, FluidBC, GeometryOp, StudyType
from metahotspot.types import MhsModel, MhsStudy, MhsSurfaceRegion, Point2D, Rect2D

if TYPE_CHECKING:
    from metahotspot.compiled import Compiled


# ---- low-level ctypes marshalling helpers --------------------------------


def _text(value: str | None) -> bytes | None:
    return None if value is None else value.encode("utf-8")


def _double_ptr(array: np.ndarray):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _create_model(dll):
    handle = ctypes.POINTER(MhsModel)()
    check(dll.mhs_model_create(ctypes.byref(handle)), "create")
    return handle


@dataclass(frozen=True)
class Study:
    type: StudyType = StudyType.STEADY
    initial_temperature: float = 300.0
    duration: float = 0.0
    output_interval: float = 1.0

    def _native(self) -> MhsStudy:
        return MhsStudy(
            int(self.type),
            self.initial_temperature,
            self.duration,
            self.output_interval,
        )


@dataclass(frozen=True)
class SurfaceRegion:
    """Axis-aligned SI rectangle. Tangential axes: X→(y,z), Y→(x,z), Z→(x,y)."""

    axis: Axis
    coordinate: float
    a_min: float
    a_max: float
    b_min: float
    b_max: float


@dataclass(frozen=True)
class RomPortPatch:
    """A physical facet of a meshless ROM port.

    trace maps reduced coordinates to the adjacent physical temperature;
    half_resistance is distance/conductivity in m² K/W. NCC on nonmatching
    interfaces requires both tangential-gradient maps.
    """

    rectangle: tuple[float, float, float, float]
    half_resistance: float
    center: tuple[float, float]
    trace: Sequence[float]
    offset: float = 0.0
    gradient_a: Sequence[float] | None = None
    gradient_b: Sequence[float] | None = None


def _surface_regions(regions: Sequence[SurfaceRegion]) -> tuple[object, int]:
    if not regions:
        raise ValueError("surface regions must not be empty")
    values = [
        MhsSurfaceRegion(
            int(r.axis), r.coordinate, Rect2D(r.a_min, r.a_max, r.b_min, r.b_max)
        )
        for r in regions
    ]
    return (MhsSurfaceRegion * len(values))(*values), len(values)


def _add_piecewise(dll, handle, name, points) -> None:
    native = (Point2D * len(points))(*[Point2D(x, y) for x, y in points])
    check(
        dll.mhs_model_add_function_piecewise(handle, _text(name), native, len(points)),
        "add_function_piecewise",
    )


def _add_periodic_constant(dll, handle, name, values, period) -> None:
    native = (ctypes.c_double * len(values))(*values)
    check(
        dll.mhs_model_add_function_periodic_piecewise_constant(
            handle, _text(name), native, len(values), period
        ),
        "add_function_periodic_piecewise_constant",
    )


# ---- public wrapper ------------------------------------------------------


class Model(OwnedHandle):
    """A mutable MetaHotspot model, wrapping an mhs_model_t handle.

    Usage::

        m = Model()
        # Set SI mesh axes, materials and volumes before compiling.
        c = m.compile()
        sol = c.solve()

    Use the context manager for automatic cleanup::

        with Model() as m:
            ...
    """

    def __init__(self) -> None:
        dll = get_dll()
        handle = _create_model(dll)
        super().__init__(dll, handle, dll.mhs_model_destroy)
        self._rom_orders: dict[int, int] = {}

    # ---- Model construction helpers ----

    def set_study(self, study: Study) -> None:
        """Set the default study for subsequent compiled snapshots."""
        native = study._native()
        self._call("mhs_model_set_study", ctypes.byref(native), ctx="set_study")

    def set_cartesian_mesh(
        self,
        x: Sequence[float] | np.ndarray,
        y: Sequence[float] | np.ndarray,
        z: Sequence[float] | np.ndarray,
        *,
        domain: int = 0,
    ) -> None:
        """Set mesh vertices."""
        arrays = [np.ascontiguousarray(a, dtype=np.float64) for a in (x, y, z)]
        if any(a.ndim != 1 or a.size < 2 for a in arrays):
            raise ValueError(
                "each mesh axis needs a 1D array with at least two vertices"
            )
        self._call(
            "mhs_model_set_cartesian_mesh",
            domain,
            *[v for a in arrays for v in (a.size, _double_ptr(a))],
            ctx="set_cartesian_mesh",
        )

    def add_variable(self, name: str, expression: str) -> None:
        """Add an expression constant."""
        self._call(
            "mhs_model_add_variable", _text(name), _text(expression), ctx="add_variable"
        )

    def add_material(
        self,
        name: str,
        kx: str,
        ky: str,
        kz: str,
        rho: str,
        c: str,
        dynamic_viscosity: str | None = None,
    ) -> int:
        """Register a material and return its ID. Laws use canonical T."""
        identifier = ctypes.c_uint32()
        text_args = tuple(_text(v) for v in (name, kx, ky, kz, rho, c))
        self._call(
            "mhs_model_add_material",
            *text_args,
            _text(dynamic_viscosity),
            ctypes.byref(identifier),
            ctx="add_material",
        )

        return identifier.value

    def add_source(self, expression: str) -> int:
        """Register a volumetric source in W/m³ (canonical x/y/z/T/t)."""
        identifier = ctypes.c_uint32()
        self._call(
            "mhs_model_add_source",
            _text(expression),
            ctypes.byref(identifier),
            ctx="add_source",
        )
        return identifier.value

    def add_volume(
        self, material: int, source: int = 0xFFFFFFFF, *, domain: int = 0
    ) -> int:
        """Return a volume ID. Later volumes override earlier volumes."""
        identifier = ctypes.c_uint32()
        self._call(
            "mhs_model_add_volume",
            domain,
            material,
            source,
            ctypes.byref(identifier),
            ctx="add_volume",
        )
        return identifier.value

    def add_box(
        self,
        volume: int,
        minimum,
        maximum,
        op: GeometryOp = GeometryOp.ADD,
        *,
        domain: int = 0,
    ) -> None:
        """Append an axis-aligned box operation in SI metres."""
        lo = np.ascontiguousarray(minimum, dtype=np.float64)
        hi = np.ascontiguousarray(maximum, dtype=np.float64)
        if lo.shape != (3,) or hi.shape != (3,):
            raise ValueError("box bounds must each contain three coordinates")
        self._call(
            "mhs_model_volume_add_box",
            domain,
            volume,
            int(op),
            _double_ptr(lo),
            _double_ptr(hi),
            ctx="add_box",
        )

    # ---- Atomic boundary conditions ----

    def add_dirichlet(
        self, regions: Sequence[SurfaceRegion], temperature: str, *, domain: int = 0
    ) -> None:
        """Prescribe temperature on nonempty surface regions (later calls win)."""
        native, count = _surface_regions(regions)
        self._call(
            "mhs_model_add_dirichlet",
            domain,
            native,
            count,
            _text(temperature),
            ctx="add_dirichlet",
        )

    def add_neumann(
        self, regions: Sequence[SurfaceRegion], heat_flux: str, *, domain: int = 0
    ) -> None:
        """Prescribe heat flux on nonempty surface regions."""
        native, count = _surface_regions(regions)
        self._call(
            "mhs_model_add_neumann",
            domain,
            native,
            count,
            _text(heat_flux),
            ctx="add_neumann",
        )

    def add_convection(
        self,
        regions: Sequence[SurfaceRegion],
        coefficient: str,
        ambient_temperature: str,
        *,
        domain: int = 0,
    ) -> None:
        """Apply convection on nonempty surface regions."""
        native, count = _surface_regions(regions)
        self._call(
            "mhs_model_add_convection",
            domain,
            native,
            count,
            _text(coefficient),
            _text(ambient_temperature),
            ctx="add_convection",
        )

    def set_default_dirichlet(self, temperature: str, *, domain: int = 0) -> None:
        self._call(
            "mhs_model_set_default_dirichlet",
            domain,
            _text(temperature),
            ctx="set_default_dirichlet",
        )

    def set_default_neumann(self, heat_flux: str, *, domain: int = 0) -> None:
        self._call(
            "mhs_model_set_default_neumann",
            domain,
            _text(heat_flux),
            ctx="set_default_neumann",
        )

    def set_default_convection(
        self, coefficient: str, ambient_temperature: str, *, domain: int = 0
    ) -> None:
        self._call(
            "mhs_model_set_default_convection",
            domain,
            _text(coefficient),
            _text(ambient_temperature),
            ctx="set_default_convection",
        )

    # ---- Functions ----

    def add_function_expr(self, name: str, expression: str) -> None:
        self._call(
            "mhs_model_add_function_expr",
            _text(name),
            _text(expression),
            ctx="add_function_expr",
        )

    def add_function_gauss(
        self, name: str, amplitude: float, tau: float, center: float
    ) -> None:
        self._call(
            "mhs_model_add_function_gauss",
            _text(name),
            amplitude,
            tau,
            center,
            ctx="add_function_gauss",
        )

    def add_function_sine(
        self, name: str, amplitude: float, angular_frequency: float, phase: float
    ) -> None:
        self._call(
            "mhs_model_add_function_sine",
            _text(name),
            amplitude,
            angular_frequency,
            phase,
            ctx="add_function_sine",
        )

    def add_function_double_exponential(
        self, name: str, amplitude: float, alpha: float, beta: float
    ) -> None:
        self._call(
            "mhs_model_add_function_double_exponential",
            _text(name),
            amplitude,
            alpha,
            beta,
            ctx="add_function_double_exponential",
        )

    def add_function_piecewise(self, name: str, points: np.ndarray) -> None:
        """Register a piecewise-linear function.

        Parameters
        ----------
        points : ndarray of shape (N, 2) — columns are (x, y).
        """
        _add_piecewise(self._dll, self._handle, name, points)

    def add_function_periodic_piecewise_constant(
        self, name: str, values: np.ndarray, period: float
    ) -> None:
        """Register a periodic piecewise-constant function."""
        _add_periodic_constant(self._dll, self._handle, name, values, period)

    # ---- Probes & fluid boundaries ----

    def add_point_probe(self, name: str, position: Sequence[float]) -> None:
        """Add a named point probe in SI metres."""
        point = np.ascontiguousarray(position, dtype=np.float64)
        if point.shape != (3,):
            raise ValueError("probe position must contain three coordinates")
        self._call(
            "mhs_model_add_point_probe",
            _text(name),
            _double_ptr(point),
            ctx="add_point_probe",
        )

    def add_fluid_boundary(
        self,
        regions: Sequence[SurfaceRegion],
        kind: FluidBC,
        value: float,
        inlet_temperature: float | None = None,
        *,
        domain: int = 0,
    ) -> None:
        """Apply a hydraulic condition on nonempty surface regions."""
        native, count = _surface_regions(regions)
        temperature = (
            None if inlet_temperature is None else ctypes.c_double(inlet_temperature)
        )
        self._call(
            "mhs_model_add_fluid_boundary",
            domain,
            native,
            count,
            int(kind),
            value,
            None if temperature is None else ctypes.byref(temperature),
            ctx="add_fluid_boundary",
        )

    def add_fom_domain(self) -> int:
        """Add an independent FOM; domain 0 exists from creation."""
        identifier = ctypes.c_uint32()
        self._call(
            "mhs_model_add_fom_domain", ctypes.byref(identifier), ctx="add_fom_domain"
        )
        return identifier.value

    def add_rom_domain(
        self,
        stiffness: np.ndarray,
        capacity: np.ndarray,
        rhs: np.ndarray,
        *,
        initial_state: np.ndarray | None = None,
    ) -> int:
        """Add a meshless ROM using projected K/C/f and optional modal initial state."""
        k = np.ascontiguousarray(stiffness, dtype=np.float64)
        c = np.ascontiguousarray(capacity, dtype=np.float64)
        f = np.ascontiguousarray(rhs, dtype=np.float64)
        if (
            k.ndim != 2
            or k.shape[0] == 0
            or k.shape[0] != k.shape[1]
            or c.shape != k.shape
        ):
            raise ValueError("ROM K/C must be nonempty square matrices")
        if f.shape != (k.shape[0],):
            raise ValueError("ROM rhs must have shape (order,)")
        initial = (
            None
            if initial_state is None
            else np.ascontiguousarray(initial_state, dtype=np.float64)
        )
        if initial is not None and initial.shape != f.shape:
            raise ValueError("ROM modal initial state must have shape (order,)")
        identifier = ctypes.c_uint32()
        self._call(
            "mhs_model_add_rom_domain",
            k.shape[0],
            _double_ptr(k),
            _double_ptr(c),
            _double_ptr(f),
            None if initial is None else _double_ptr(initial),
            ctypes.byref(identifier),
            ctx="add_rom_domain",
        )
        self._rom_orders[identifier.value] = k.shape[0]
        return identifier.value

    def add_rom_affine_term(
        self,
        domain: int,
        name: str,
        value: float,
        stiffness: np.ndarray,
        rhs: np.ndarray,
    ) -> None:
        """Supply mu*K_i and mu*f_i from a reusable affine HTC ROM.

        The parameter value is frozen into each compiled snapshot. Different
        snapshots can reuse the same ROM matrices and physical port mappings.
        """
        order = self._rom_orders.get(domain)
        if order is None:
            raise ValueError("affine parameter requires a ROM domain")
        k = np.ascontiguousarray(stiffness, dtype=np.float64)
        f = np.ascontiguousarray(rhs, dtype=np.float64)
        if k.shape != (order, order) or f.shape != (order,):
            raise ValueError("invalid affine ROM operator dimensions")
        self._call(
            "mhs_model_rom_add_affine_term",
            domain,
            _text(name),
            value,
            _double_ptr(k),
            _double_ptr(f),
            ctx="add_rom_affine_term",
        )

    def add_rom_port(
        self,
        domain: int,
        name: str,
        region: SurfaceRegion,
        outward_sign: int,
        patches: Sequence[RomPortPatch],
    ) -> int:
        """Declare a meshless physical ROM port and its modal facet maps."""
        surface, _ = _surface_regions([region])
        identifier = ctypes.c_uint32()
        self._call(
            "mhs_model_add_rom_port",
            domain,
            _text(name),
            surface,
            outward_sign,
            ctypes.byref(identifier),
            ctx="add_rom_port",
        )
        if domain not in self._rom_orders:
            raise ValueError("ROM port requires a ROM domain created by add_rom_domain")
        for patch in patches:
            if len(patch.rectangle) != 4:
                raise ValueError("ROM patch rectangle needs four bounds")
            rect = Rect2D(*patch.rectangle)
            center = np.ascontiguousarray(patch.center, dtype=np.float64)
            trace = np.ascontiguousarray(patch.trace, dtype=np.float64)
            if center.shape != (2,) or trace.shape != (self._rom_orders[domain],):
                raise ValueError("ROM patch center or trace has invalid dimensions")
            ga = (
                None
                if patch.gradient_a is None
                else np.ascontiguousarray(patch.gradient_a, dtype=np.float64)
            )
            gb = (
                None
                if patch.gradient_b is None
                else np.ascontiguousarray(patch.gradient_b, dtype=np.float64)
            )
            if (ga is not None and ga.shape != trace.shape) or (
                gb is not None and gb.shape != trace.shape
            ):
                raise ValueError("ROM patch tangent-gradient shape must match trace")
            self._call(
                "mhs_model_rom_port_add_patch",
                domain,
                identifier.value,
                ctypes.byref(rect),
                patch.half_resistance,
                _double_ptr(center),
                _double_ptr(trace),
                patch.offset,
                None if ga is None else _double_ptr(ga),
                None if gb is None else _double_ptr(gb),
                ctx="add_rom_port_patch",
            )
        return identifier.value

    def add_rom_observation(
        self,
        domain: int,
        name: str,
        position: Sequence[float],
        weights: Sequence[float],
        offset: float = 0.0,
    ) -> None:
        """Map ROM coordinates to a named physical point-probe temperature."""
        xyz = np.ascontiguousarray(position, dtype=np.float64)
        coefficients = np.ascontiguousarray(weights, dtype=np.float64)
        if xyz.shape != (3,) or coefficients.shape != (
            self._rom_orders.get(domain, -1),
        ):
            raise ValueError(
                "ROM observation requires xyz=(3,) and modal weights=(order,)"
            )
        self._call(
            "mhs_model_add_rom_observation",
            domain,
            _text(name),
            _double_ptr(xyz),
            _double_ptr(coefficients),
            offset,
            ctx="add_rom_observation",
        )

    def add_port(self, domain: int, name: str, region: SurfaceRegion) -> int:
        """Declare a named exterior thermal interface, connected exactly once."""
        identifier = ctypes.c_uint32()
        native, _ = _surface_regions([region])
        self._call(
            "mhs_model_add_port",
            domain,
            _text(name),
            native,
            ctypes.byref(identifier),
            ctx="add_port",
        )
        return identifier.value

    def connect(self, domain_a: int, port_a: int, domain_b: int, port_b: int) -> int:
        """Join coincident ports with perfect thermal contact and automatic NCC."""
        identifier = ctypes.c_uint32()
        self._call(
            "mhs_model_connect",
            domain_a,
            port_a,
            domain_b,
            port_b,
            ctypes.byref(identifier),
            ctx="connect",
        )
        return identifier.value

    # ---- Compile ----

    def compile(self) -> Compiled:
        """Compile the model into a read-only runtime representation."""
        from metahotspot.compiled import Compiled

        return Compiled._from_model(self._dll, self._handle)
