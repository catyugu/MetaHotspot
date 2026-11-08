"""High-level wrapper for ``mhs_model_t`` — model construction."""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING

import numpy as np

from metahotspot._error import check
from metahotspot._handle import OwnedHandle
from metahotspot._lib import get_dll
from metahotspot.enums import Axis, FluidBC, GeometryOp
from metahotspot.types import MhsFaceRegion, MhsModel, Point2D, Rect2D

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


def _face_regions(regions) -> tuple[object, int]:
    values = [
        MhsFaceRegion(int(axis), coordinate, Rect2D(a_min, a_max, b_min, b_max))
        for axis, coordinate, a_min, a_max, b_min, b_max in regions or ()
    ]
    return ((MhsFaceRegion * len(values))(*values) if values else None), len(values)


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


def _add_fluid_boundary(
    dll, handle, axis, coordinate, region, kind, value, inlet_temperature
) -> None:
    regions, count = _face_regions([(axis, coordinate, *region)])
    temperature = (
        None if inlet_temperature is None else ctypes.c_double(inlet_temperature)
    )
    check(
        dll.mhs_model_add_fluid_boundary(
            handle,
            regions,
            count,
            int(kind),
            value,
            None if temperature is None else ctypes.byref(temperature),
        ),
        "add_fluid_boundary",
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

    # ---- Model construction helpers ----

    def set_initial_temperature(self, temperature_K: float) -> None:
        """Set the default state and fluid reference temperature in kelvin."""
        self._call(
            "mhs_model_set_initial_temperature",
            temperature_K,
            ctx="set_initial_temperature",
        )

    def set_mesh(
        self,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        z: np.ndarray | None = None,
    ) -> None:
        """Set mesh vertices."""
        arrays = [np.ascontiguousarray(a, dtype=np.float64) for a in (x, y, z)]
        if any(a.ndim != 1 or a.size < 2 for a in arrays):
            raise ValueError(
                "each mesh axis needs a 1D array with at least two vertices"
            )
        self._call(
            "mhs_model_set_mesh",
            *[v for a in arrays for v in (a.size, _double_ptr(a))],
            ctx="set_mesh",
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

    def add_volume(self, material: int, source: int = 0xFFFFFFFF) -> int:
        """Return a volume ID. Later volumes override earlier volumes."""
        identifier = ctypes.c_uint32()
        self._call(
            "mhs_model_add_volume",
            material,
            source,
            ctypes.byref(identifier),
            ctx="add_volume",
        )
        return identifier.value

    def add_box(
        self, volume: int, minimum, maximum, op: GeometryOp = GeometryOp.ADD
    ) -> None:
        """Append an axis-aligned box operation in SI metres."""
        lo = np.ascontiguousarray(minimum, dtype=np.float64)
        hi = np.ascontiguousarray(maximum, dtype=np.float64)
        if lo.shape != (3,) or hi.shape != (3,):
            raise ValueError("box bounds must each contain three coordinates")
        self._call(
            "mhs_model_volume_add_box",
            volume,
            int(op),
            _double_ptr(lo),
            _double_ptr(hi),
            ctx="add_box",
        )

    # ---- Atomic boundary conditions ----

    def add_dirichlet(
        self,
        temperature: str,
        regions: (list[tuple[int, float, float, float, float, float]] | None) = None,
    ) -> None:
        """Add a Dirichlet boundary condition.

        Each region is (axis, coordinate, a_min, a_max, b_min, b_max).
        Pass ``None`` or an empty list to ignore the call.
        """
        native_regions, count = _face_regions(regions)
        self._call(
            "mhs_model_add_dirichlet",
            native_regions,
            count,
            _text(temperature),
            ctx="add_dirichlet",
        )

    def add_neumann(
        self,
        heat_flux: str,
        regions: (list[tuple[int, float, float, float, float, float]] | None) = None,
    ) -> None:
        """Add a Neumann (heat flux) boundary condition."""
        native_regions, count = _face_regions(regions)
        self._call(
            "mhs_model_add_neumann",
            native_regions,
            count,
            _text(heat_flux),
            ctx="add_neumann",
        )

    def add_convection(
        self,
        coefficient: str,
        ambient_temperature: str,
        regions: (list[tuple[int, float, float, float, float, float]] | None) = None,
    ) -> None:
        """Add a convection (Robin) boundary condition."""
        native_regions, count = _face_regions(regions)
        self._call(
            "mhs_model_add_convection",
            native_regions,
            count,
            _text(coefficient),
            _text(ambient_temperature),
            ctx="add_convection",
        )

    def set_default_dirichlet(self, temperature: str) -> None:
        self._call(
            "mhs_model_set_default_dirichlet",
            _text(temperature),
            ctx="set_default_dirichlet",
        )

    def set_default_neumann(self, heat_flux: str) -> None:
        self._call(
            "mhs_model_set_default_neumann", _text(heat_flux), ctx="set_default_neumann"
        )

    def set_default_convection(
        self, coefficient: str, ambient_temperature: str
    ) -> None:
        self._call(
            "mhs_model_set_default_convection",
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

    def add_probe(self, name: str, x: float, y: float, z: float) -> None:
        self._call("mhs_model_add_probe", _text(name), x, y, z, ctx="add_probe")

    def add_fluid_boundary(
        self,
        axis: Axis,
        coordinate: float,
        a_min: float,
        a_max: float,
        b_min: float,
        b_max: float,
        kind: FluidBC,
        value: float,
        inlet_temperature: float | None = None,
    ) -> None:
        _add_fluid_boundary(
            self._dll,
            self._handle,
            axis,
            coordinate,
            (a_min, a_max, b_min, b_max),
            kind,
            value,
            inlet_temperature,
        )

    # ---- Compile ----

    def compile(self) -> Compiled:
        """Compile the model into a read-only runtime representation."""
        from metahotspot.compiled import Compiled

        return Compiled._from_model(self._dll, self._handle)
