"""High-level wrapper for ``mhs_model_t`` — model construction."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

from metahotspot._error import check
from metahotspot._handle import OwnedHandle
from metahotspot._lib import get_dll
from metahotspot.enums import FluidBC, GeometryOp, Study, LengthUnit, Axis
from metahotspot.types import MhsFaceRegion, MhsModel, Point2D, Rect2D


# ---- low-level ctypes marshalling helpers --------------------------------


def _text(value: str | None) -> bytes | None:
    return None if value is None else value.encode("utf-8")


def _double_ptr(array: np.ndarray):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _create_model(dll):
    handle = ctypes.POINTER(MhsModel)()
    check(dll.mhs_model_create(ctypes.byref(handle)), "create")
    return handle


def _set_mesh(dll, handle, x, y, z) -> None:
    args = []
    for array in (x, y, z):
        args.extend(
            (
                0 if array is None else len(array),
                None if array is None else _double_ptr(array),
            )
        )
    check(dll.mhs_model_set_mesh(handle, *args), "set_mesh")


def _add_layer(dll, handle, thickness, x_offset, y_offset) -> int:
    identifier = ctypes.c_uint32()
    check(
        dll.mhs_model_add_layer(
            handle,
            _text(thickness),
            _text(x_offset),
            _text(y_offset),
            ctypes.byref(identifier),
        ),
        "add_layer",
    )
    return identifier.value


def _add_block(
    dll, handle, layer, material_name, heat_source, x_offset, y_offset, thickness
):
    identifier = ctypes.c_uint32()
    check(
        dll.mhs_model_add_block(
            handle,
            layer,
            _text(material_name),
            _text(heat_source),
            _text(x_offset),
            _text(y_offset),
            _text(thickness),
            ctypes.byref(identifier),
        ),
        "add_block",
    )
    return identifier.value


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
    rect = Rect2D(*region)
    check(
        dll.mhs_model_add_fluid_boundary(
            handle, int(axis), coordinate, rect, int(kind), value, inlet_temperature
        ),
        "add_fluid_boundary",
    )


# ---- public wrapper ------------------------------------------------------


class Model(OwnedHandle):
    """A mutable MetaHotspot model, wrapping an mhs_model_t handle.

    Usage::

        m = Model()
        m.read_xml("case.xml")
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

    def read_xml(self, path: str | Path) -> None:
        """Load a MetaHotspot XML case file."""
        self._call("mhs_model_read_xml", _text(str(path)), ctx="read_xml")

    def set_settings(
        self,
        study: Study = Study.STEADY,
        length_unit: LengthUnit = LengthUnit.MILLIMETER,
        initial_temperature_K: float = 300.0,
        duration: float = 0.0,
        output_interval: float = 0.0,
    ) -> None:
        """Set global study parameters."""
        self._call(
            "mhs_model_set_settings",
            int(study),
            int(length_unit),
            initial_temperature_K,
            duration,
            output_interval,
            ctx="set_settings",
        )

    def set_mesh(
        self,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        z: np.ndarray | None = None,
    ) -> None:
        """Set mesh vertices."""
        _set_mesh(self._dll, self._handle, x, y, z)

    def add_variable(self, name: str, expression: str) -> None:
        """Add a geometry variable."""
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
    ) -> None:
        """Register a material."""
        text_args = tuple(_text(v) for v in (name, kx, ky, kz, rho, c))
        self._call(
            "mhs_model_add_material",
            *text_args,
            _text(dynamic_viscosity),
            ctx="add_material",
        )

    def add_layer(
        self, thickness: str, x_offset: str = "0", y_offset: str = "0"
    ) -> int:
        """Add a layer.  Returns the layer ID."""
        return _add_layer(self._dll, self._handle, thickness, x_offset, y_offset)

    def add_block(
        self,
        layer: int,
        material_name: str,
        heat_source: str = "0",
        x_offset: str = "0",
        y_offset: str = "0",
        thickness: str | None = None,
    ) -> int:
        """Add a block to a layer.  Returns the block ID."""
        return _add_block(
            self._dll,
            self._handle,
            layer,
            material_name,
            heat_source,
            x_offset,
            y_offset,
            thickness,
        )

    def add_rect(
        self,
        block: int,
        op: GeometryOp = GeometryOp.ADD,
        x: str = "0",
        y: str = "0",
        width: str = "1",
        height: str = "1",
    ) -> None:
        """Add a rectangular geometry operation (add or subtract)."""
        self._call(
            "mhs_model_add_rect",
            block,
            int(op),
            _text(x),
            _text(y),
            _text(width),
            _text(height),
            ctx="add_rect",
        )

    # ---- Atomic boundary conditions ----

    def add_dirichlet(
        self,
        temperature: str,
        regions: (
            list[tuple[int, float, float, float, float, float, float]] | None
        ) = None,
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
        regions: (
            list[tuple[int, float, float, float, float, float, float]] | None
        ) = None,
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
        regions: (
            list[tuple[int, float, float, float, float, float, float]] | None
        ) = None,
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
        inlet_temperature: float = 0.0,
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
