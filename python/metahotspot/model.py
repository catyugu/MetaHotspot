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
from metahotspot.enums import Axis, FluidBC, GeometryOp, StudyType, ThermalBoundaryKind
from metahotspot.types import (
    MhsLibrary,
    MhsModel,
    MhsRomPortPatch,
    MhsStudy,
    MhsSurfaceRegion,
    MhsThermalBoundary,
    Point2D,
    Rect2D,
)

if TYPE_CHECKING:
    from metahotspot.compiled import Compiled


# ---- low-level ctypes marshalling helpers --------------------------------


def _text(value: str | None) -> bytes | None:
    return None if value is None else value.encode("utf-8")


def _double_ptr(array: np.ndarray):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _create_model(dll, library):
    handle = ctypes.POINTER(MhsModel)()
    check(dll.mhs_model_create(library._handle, ctypes.byref(handle)), "create")
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
class Dirichlet:
    temperature: str


@dataclass(frozen=True)
class Neumann:
    heat_flux: str


@dataclass(frozen=True)
class Convection:
    coefficient: str
    ambient_temperature: str


ThermalBoundary = Dirichlet | Neumann | Convection


def _thermal_boundary(condition: ThermalBoundary) -> MhsThermalBoundary:
    if isinstance(condition, Dirichlet):
        return MhsThermalBoundary(
            int(ThermalBoundaryKind.DIRICHLET), _text(condition.temperature), None
        )
    if isinstance(condition, Neumann):
        return MhsThermalBoundary(
            int(ThermalBoundaryKind.NEUMANN), _text(condition.heat_flux), None
        )
    if isinstance(condition, Convection):
        return MhsThermalBoundary(
            int(ThermalBoundaryKind.CONVECTION),
            _text(condition.coefficient),
            _text(condition.ambient_temperature),
        )
    raise TypeError("expected Dirichlet, Neumann or Convection")


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


# ---- public wrapper ------------------------------------------------------


class Library(OwnedHandle):
    """Reusable material, source, expression and boundary definitions."""

    def __init__(self) -> None:
        dll = get_dll()
        handle = ctypes.POINTER(MhsLibrary)()
        check(dll.mhs_library_create(ctypes.byref(handle)), "create_library")
        super().__init__(dll, handle, dll.mhs_library_destroy)

    def _add_id(self, function: str, *args, ctx: str | None = None) -> int:
        identifier = ctypes.c_uint32()
        self._call(function, *args, ctypes.byref(identifier), ctx=ctx)
        return identifier.value

    def add_variable(self, name: str, expression: str) -> int:
        """Add an expression constant."""
        return self._add_id(
            "mhs_library_add_variable",
            _text(name),
            _text(expression),
            ctx="add_variable",
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
        text_args = tuple(_text(v) for v in (name, kx, ky, kz, rho, c))
        return self._add_id(
            "mhs_library_add_material",
            *text_args,
            _text(dynamic_viscosity),
            ctx="add_material",
        )

    def add_source(self, expression: str) -> int:
        """Register a volumetric source in W/m³ (canonical x/y/z/T/t)."""
        return self._add_id(
            "mhs_library_add_source", _text(expression), ctx="add_source"
        )

    # ---- Functions ----

    def add_function_expr(self, name: str, expression: str) -> int:
        return self._add_id(
            "mhs_library_add_function_expr",
            _text(name),
            _text(expression),
            ctx="add_function_expr",
        )

    def add_function_gauss(
        self, name: str, amplitude: float, tau: float, center: float
    ) -> int:
        return self._add_id(
            "mhs_library_add_function_gauss",
            _text(name),
            amplitude,
            tau,
            center,
            ctx="add_function_gauss",
        )

    def add_function_sine(
        self, name: str, amplitude: float, angular_frequency: float, phase: float
    ) -> int:
        return self._add_id(
            "mhs_library_add_function_sine",
            _text(name),
            amplitude,
            angular_frequency,
            phase,
            ctx="add_function_sine",
        )

    def add_function_double_exponential(
        self, name: str, amplitude: float, alpha: float, beta: float
    ) -> int:
        return self._add_id(
            "mhs_library_add_function_double_exponential",
            _text(name),
            amplitude,
            alpha,
            beta,
            ctx="add_function_double_exponential",
        )

    def add_function_piecewise(self, name: str, points: np.ndarray) -> int:
        """Register a piecewise-linear function.

        Parameters
        ----------
        points : ndarray of shape (N, 2) — columns are (x, y).
        """
        native = (Point2D * len(points))(*(Point2D(x, y) for x, y in points))
        return self._add_id(
            "mhs_library_add_function_piecewise",
            _text(name),
            native,
            len(points),
            ctx="add_function_piecewise",
        )

    def add_function_periodic_piecewise_constant(
        self, name: str, values: np.ndarray, period: float
    ) -> int:
        """Register a periodic piecewise-constant function."""
        native = (ctypes.c_double * len(values))(*values)
        return self._add_id(
            "mhs_library_add_function_periodic_piecewise_constant",
            _text(name),
            native,
            len(values),
            period,
            ctx="add_function_periodic_piecewise_constant",
        )

    def add_thermal_boundary(self, condition: ThermalBoundary) -> int:
        native = _thermal_boundary(condition)
        return self._add_id("mhs_library_add_thermal_boundary", ctypes.byref(native))

    def add_fluid_boundary(
        self, kind: FluidBC, value: float, inlet_temperature: float | None = None
    ) -> int:
        temperature = (
            None if inlet_temperature is None else ctypes.c_double(inlet_temperature)
        )
        return self._add_id(
            "mhs_library_add_fluid_boundary",
            int(kind),
            value,
            None if temperature is None else ctypes.byref(temperature),
        )


class Model(OwnedHandle):
    """A mutable MetaHotspot model, wrapping an mhs_model_t handle.

    Usage::

        library = Library()
        m = Model(library)
        m.add_cartesian_component()  # Explicitly create the Cartesian component.
        # Set SI mesh axes, materials and volumes before compiling.
        c = m.compile()
        sol = c.solve()

    Use the context manager for automatic cleanup::

        with Library() as library, Model(library) as m:
            ...
    """

    def __init__(self, library: Library) -> None:
        dll = get_dll()
        handle = _create_model(dll, library)
        super().__init__(dll, handle, dll.mhs_model_destroy)
        self.library = library
        self._reduced_orders: dict[int, int] = {}

    # ---- Model construction helpers ----

    def set_cartesian_mesh(
        self,
        x: Sequence[float] | np.ndarray,
        y: Sequence[float] | np.ndarray,
        z: Sequence[float] | np.ndarray,
        *,
        component: int = 0,
    ) -> None:
        """Set mesh vertices."""
        arrays = [np.ascontiguousarray(a, dtype=np.float64) for a in (x, y, z)]
        if any(a.ndim != 1 or a.size < 2 for a in arrays):
            raise ValueError(
                "each mesh axis needs a 1D array with at least two vertices"
            )
        self._call(
            "mhs_model_set_cartesian_mesh",
            component,
            *[v for a in arrays for v in (a.size, _double_ptr(a))],
            ctx="set_cartesian_mesh",
        )

    def set_flow_mode(self, mode: int, *, component: int = 0) -> None:
        """Select the fluid solution: FROZEN (legacy) or STOKES (coupled)."""
        self._call("mhs_model_set_flow_mode", component, int(mode), ctx="set_flow_mode")

    def add_volume(
        self, material: int, source: int = 0xFFFFFFFF, *, component: int = 0
    ) -> int:
        """Return a volume ID. Later volumes override earlier volumes."""
        identifier = ctypes.c_uint32()
        self._call(
            "mhs_model_add_volume",
            component,
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
        component: int = 0,
    ) -> None:
        """Append an axis-aligned box operation in SI metres."""
        lo = np.ascontiguousarray(minimum, dtype=np.float64)
        hi = np.ascontiguousarray(maximum, dtype=np.float64)
        if lo.shape != (3,) or hi.shape != (3,):
            raise ValueError("box bounds must each contain three coordinates")
        self._call(
            "mhs_model_volume_add_box",
            component,
            volume,
            int(op),
            _double_ptr(lo),
            _double_ptr(hi),
            ctx="add_box",
        )

    # ---- Thermal boundary assignments ----

    def add_thermal_boundary(
        self, regions: Sequence[SurfaceRegion], condition: int, *, component: int = 0
    ) -> None:
        native, count = _surface_regions(regions)
        self._call(
            "mhs_model_add_thermal_boundary", component, native, count, condition
        )

    def set_default_thermal_boundary(
        self, condition: int, *, component: int = 0
    ) -> None:
        self._call("mhs_model_set_default_thermal_boundary", component, condition)

    # ---- Probes & fluid boundaries ----

    def add_fluid_boundary(
        self, regions: Sequence[SurfaceRegion], condition: int, *, component: int = 0
    ) -> None:
        native, count = _surface_regions(regions)
        self._call("mhs_model_add_fluid_boundary", component, native, count, condition)

    def add_cartesian_component(self) -> int:
        """Add a Cartesian finite-volume component; IDs start at zero."""
        identifier = ctypes.c_uint32()
        self._call(
            "mhs_model_add_cartesian_component",
            ctypes.byref(identifier),
            ctx="add_cartesian_component",
        )
        return identifier.value

    def add_reduced_component(
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
            "mhs_model_add_reduced_component",
            k.shape[0],
            _double_ptr(k),
            _double_ptr(c),
            _double_ptr(f),
            None if initial is None else _double_ptr(initial),
            ctypes.byref(identifier),
            ctx="add_reduced_component",
        )
        self._reduced_orders[identifier.value] = k.shape[0]
        return identifier.value

    def add_reduced_affine_term(
        self,
        component: int,
        name: str,
        value: float,
        stiffness: np.ndarray,
        rhs: np.ndarray,
    ) -> None:
        """Supply an intrinsic affine K/f parameter to a ROM.

        Exterior HTC uses add_thermal_boundary(component=...), sharing
        the physical port maps used for inter-component connections.
        """
        order = self._reduced_orders.get(component)
        if order is None:
            raise ValueError("affine parameter requires a ROM component")
        k = np.ascontiguousarray(stiffness, dtype=np.float64)
        f = np.ascontiguousarray(rhs, dtype=np.float64)
        if k.shape != (order, order) or f.shape != (order,):
            raise ValueError("invalid affine ROM operator dimensions")
        self._call(
            "mhs_model_add_reduced_affine_term",
            component,
            _text(name),
            value,
            _double_ptr(k),
            _double_ptr(f),
            ctx="add_reduced_affine_term",
        )

    def add_port(
        self,
        component: int,
        name: str,
        region: SurfaceRegion,
        *,
        patches: Sequence[RomPortPatch] = (),
        outward_sign: int = 0,
    ) -> int:
        """Declare an exterior port with optional reduced trace patches."""
        order = self._reduced_orders.get(component)
        if order is None:
            if patches or outward_sign:
                raise ValueError("Cartesian ports derive their traces from geometry")
        elif not patches or outward_sign not in (-1, 1):
            raise ValueError("reduced ports require patches and outward_sign +/-1")

        native_patches = (MhsRomPortPatch * len(patches))()
        keepalive = []
        for i, patch in enumerate(patches):
            if len(patch.rectangle) != 4:
                raise ValueError("port patch rectangle needs four bounds")
            center = np.ascontiguousarray(patch.center, dtype=np.float64)
            trace = np.ascontiguousarray(patch.trace, dtype=np.float64)
            if center.shape != (2,) or trace.shape != (order,):
                raise ValueError("patch center or trace dimensions are invalid")
            gradients = [
                None if grad is None else np.ascontiguousarray(grad, dtype=np.float64)
                for grad in (patch.gradient_a, patch.gradient_b)
            ]
            if any(
                grad is not None and grad.shape != trace.shape for grad in gradients
            ):
                raise ValueError("tangential gradient shape must match trace")
            keepalive.extend((center, trace, *gradients))
            native_patches[i] = MhsRomPortPatch(
                Rect2D(*patch.rectangle),
                patch.half_resistance,
                (ctypes.c_double * 2)(*center),
                _double_ptr(trace),
                patch.offset,
                *(None if grad is None else _double_ptr(grad) for grad in gradients),
            )

        native, _ = _surface_regions([region])
        identifier = ctypes.c_uint32()
        self._call(
            "mhs_model_add_port",
            component,
            _text(name),
            native,
            outward_sign,
            native_patches if patches else None,
            len(patches),
            ctypes.byref(identifier),
            ctx="add_port",
        )
        return identifier.value

    def add_observation(
        self,
        component: int,
        name: str,
        position: Sequence[float],
        *,
        weights: Sequence[float] | None = None,
        offset: float = 0.0,
    ) -> None:
        """Observe physical temperature in a Cartesian or reduced component."""
        xyz = np.ascontiguousarray(position, dtype=np.float64)
        if xyz.shape != (3,):
            raise ValueError("observation position must have three coordinates")
        order = self._reduced_orders.get(component)
        if order is None:
            if weights is not None or offset != 0.0:
                raise ValueError("Cartesian observations use only a physical point")
            coefficients = None
        else:
            if weights is None:
                raise ValueError("reduced observation requires modal weights")
            coefficients = np.ascontiguousarray(weights, dtype=np.float64)
            if coefficients.shape != (order,):
                raise ValueError("reduced observation weights must have shape (order,)")
        self._call(
            "mhs_model_add_observation",
            component,
            _text(name),
            _double_ptr(xyz),
            None if coefficients is None else _double_ptr(coefficients),
            offset,
            ctx="add_observation",
        )

    def connect(
        self, component_a: int, port_a: int, component_b: int, port_b: int
    ) -> int:
        """Connect coincident thermal ports with nonmatching-interface correction."""
        identifier = ctypes.c_uint32()
        self._call(
            "mhs_model_connect",
            component_a,
            port_a,
            component_b,
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
