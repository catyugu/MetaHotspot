"""High-level wrapper for ``mhs_model_t`` — model construction."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

from metahotspot._lib import get_dll as _get_dll
from metahotspot._dll_interface import configure_dll
from metahotspot._error import check
from metahotspot.enums import FluidBC, GeometryOp, Study, LengthUnit, Axis
from metahotspot.types import (
    MhsModel,
    MhsFaceRegion,
    Rect2D,
    Point2D,
    SolverOpts,
    MHS_LAYER_ID_INVALID,
    MHS_BLOCK_ID_INVALID,
    MHS_MATERIAL_ID_INVALID,
    MHS_FUNCTION_ID_INVALID,
    MHS_PROBE_ID_INVALID,
)


class Model:
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
        dll = _get_dll()
        configure_dll(dll)
        self._dll = dll

        pp = ctypes.POINTER(MhsModel)()
        check(dll.mhs_model_create(ctypes.byref(pp)), "create")
        self._handle: MhsModel = pp
        self._owned = True

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        if self._owned and self._handle is not None:
            self._dll.mhs_model_destroy(self._handle)
            self._handle = None

    def __enter__(self) -> Model:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # ---- Model construction helpers ----

    def read_xml(self, path: str | Path) -> None:
        """Load a MetaHotspot XML case file."""
        path_bytes = str(path).encode("utf-8")
        check(self._dll.mhs_model_read_xml(self._handle, path_bytes), "read_xml")

    def set_settings(
        self,
        study: Study = Study.STEADY,
        length_unit: LengthUnit = LengthUnit.MILLIMETER,
        initial_temperature_K: float = 300.0,
        duration: float = 0.0,
        output_interval: float = 0.0,
    ) -> None:
        """Set global study parameters."""
        check(
            self._dll.mhs_model_set_settings(
                self._handle,
                int(study),
                int(length_unit),
                initial_temperature_K,
                duration,
                output_interval,
            ),
            "set_settings",
        )

    def set_mesh(
        self,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        z: np.ndarray | None = None,
    ) -> None:
        """Set mesh vertices."""
        nx = len(x) if x is not None else 0
        ny = len(y) if y is not None else 0
        nz = len(z) if z is not None else 0

        x_ptr = (
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) if x is not None else None
        )
        y_ptr = (
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) if y is not None else None
        )
        z_ptr = (
            z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) if z is not None else None
        )

        check(
            self._dll.mhs_model_set_mesh(self._handle, nx, x_ptr, ny, y_ptr, nz, z_ptr),
            "set_mesh",
        )

    def add_variable(self, name: str, expression: str) -> None:
        """Add a geometry variable."""
        check(
            self._dll.mhs_model_add_variable(
                self._handle, name.encode("utf-8"), expression.encode("utf-8")
            ),
            "add_variable",
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
        """Register a material.  Returns the material ID."""
        dv = (
            dynamic_viscosity.encode("utf-8") if dynamic_viscosity is not None else None
        )
        mid = self._dll.mhs_model_add_material(
            self._handle,
            name.encode("utf-8"),
            kx.encode("utf-8"),
            ky.encode("utf-8"),
            kz.encode("utf-8"),
            rho.encode("utf-8"),
            c.encode("utf-8"),
            dv,
        )
        if mid == MHS_MATERIAL_ID_INVALID:
            err = self._dll.mhs_last_error()
            raise RuntimeError(f"add_material failed: {err}")
        return mid

    def add_layer(
        self, thickness: str, x_offset: str = "0", y_offset: str = "0"
    ) -> int:
        """Add a layer.  Returns the layer ID."""
        lid = self._dll.mhs_model_add_layer(
            self._handle,
            thickness.encode("utf-8"),
            x_offset.encode("utf-8"),
            y_offset.encode("utf-8"),
        )
        if lid == MHS_LAYER_ID_INVALID:
            err = self._dll.mhs_last_error()
            raise RuntimeError(f"add_layer failed: {err}")
        return lid

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
        th = thickness.encode("utf-8") if thickness is not None else None
        bid = self._dll.mhs_model_add_block(
            self._handle,
            layer,
            material_name.encode("utf-8"),
            heat_source.encode("utf-8"),
            x_offset.encode("utf-8"),
            y_offset.encode("utf-8"),
            th,
        )
        if bid == MHS_BLOCK_ID_INVALID:
            err = self._dll.mhs_last_error()
            raise RuntimeError(f"add_block failed: {err}")
        return bid

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
        check(
            self._dll.mhs_model_add_rect(
                self._handle,
                block,
                int(op),
                x.encode("utf-8"),
                y.encode("utf-8"),
                width.encode("utf-8"),
                height.encode("utf-8"),
            ),
            "add_rect",
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
        Pass ``None`` for regions to use an empty list.
        """
        c_regions = []
        if regions:
            for r in regions:
                axis, coord, a_min, a_max, b_min, b_max = r
                rect = Rect2D(a_min, a_max, b_min, b_max)
                c_regions.append(MhsFaceRegion(int(axis), coord, rect))
        arr = (MhsFaceRegion * len(c_regions))(*c_regions) if c_regions else None
        check(
            self._dll.mhs_model_add_dirichlet(
                self._handle, arr, len(c_regions), temperature.encode("utf-8")
            ),
            "add_dirichlet",
        )

    def add_neumann(
        self,
        heat_flux: str,
        regions: (
            list[tuple[int, float, float, float, float, float, float]] | None
        ) = None,
    ) -> None:
        """Add a Neumann (heat flux) boundary condition."""
        c_regions = []
        if regions:
            for r in regions:
                axis, coord, a_min, a_max, b_min, b_max = r
                rect = Rect2D(a_min, a_max, b_min, b_max)
                c_regions.append(MhsFaceRegion(int(axis), coord, rect))
        arr = (MhsFaceRegion * len(c_regions))(*c_regions) if c_regions else None
        check(
            self._dll.mhs_model_add_neumann(
                self._handle, arr, len(c_regions), heat_flux.encode("utf-8")
            ),
            "add_neumann",
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
        c_regions = []
        if regions:
            for r in regions:
                axis, coord, a_min, a_max, b_min, b_max = r
                rect = Rect2D(a_min, a_max, b_min, b_max)
                c_regions.append(MhsFaceRegion(int(axis), coord, rect))
        arr = (MhsFaceRegion * len(c_regions))(*c_regions) if c_regions else None
        check(
            self._dll.mhs_model_add_convection(
                self._handle,
                arr,
                len(c_regions),
                coefficient.encode("utf-8"),
                ambient_temperature.encode("utf-8"),
            ),
            "add_convection",
        )

    def set_default_dirichlet(self, temperature: str) -> None:
        check(
            self._dll.mhs_model_set_default_dirichlet(
                self._handle, temperature.encode("utf-8")
            ),
            "set_default_dirichlet",
        )

    def set_default_neumann(self, heat_flux: str) -> None:
        check(
            self._dll.mhs_model_set_default_neumann(
                self._handle, heat_flux.encode("utf-8")
            ),
            "set_default_neumann",
        )

    def set_default_convection(
        self, coefficient: str, ambient_temperature: str
    ) -> None:
        check(
            self._dll.mhs_model_set_default_convection(
                self._handle,
                coefficient.encode("utf-8"),
                ambient_temperature.encode("utf-8"),
            ),
            "set_default_convection",
        )

    # ---- Functions ----

    def add_function_expr(self, name: str, expression: str) -> int:
        fid = self._dll.mhs_model_add_function_expr(
            self._handle,
            name.encode("utf-8"),
            expression.encode("utf-8"),
        )
        if fid == MHS_FUNCTION_ID_INVALID:
            err = self._dll.mhs_last_error()
            raise RuntimeError(f"add_function_expr failed: {err}")
        return fid

    def add_function_gauss(
        self, name: str, amplitude: float, tau: float, center: float
    ) -> int:
        fid = self._dll.mhs_model_add_function_gauss(
            self._handle,
            name.encode("utf-8"),
            amplitude,
            tau,
            center,
        )
        if fid == MHS_FUNCTION_ID_INVALID:
            err = self._dll.mhs_last_error()
            raise RuntimeError(f"add_function_gauss failed: {err}")
        return fid

    def add_function_sine(
        self, name: str, amplitude: float, angular_frequency: float, phase: float
    ) -> int:
        fid = self._dll.mhs_model_add_function_sine(
            self._handle,
            name.encode("utf-8"),
            amplitude,
            angular_frequency,
            phase,
        )
        if fid == MHS_FUNCTION_ID_INVALID:
            err = self._dll.mhs_last_error()
            raise RuntimeError(f"add_function_sine failed: {err}")
        return fid

    def add_function_double_exponential(
        self, name: str, amplitude: float, alpha: float, beta: float
    ) -> int:
        fid = self._dll.mhs_model_add_function_double_exponential(
            self._handle,
            name.encode("utf-8"),
            amplitude,
            alpha,
            beta,
        )
        if fid == MHS_FUNCTION_ID_INVALID:
            err = self._dll.mhs_last_error()
            raise RuntimeError(f"add_function_double_exponential failed: {err}")
        return fid

    def add_function_piecewise(self, name: str, points: np.ndarray) -> int:
        """Register a piecewise-linear function.

        Parameters
        ----------
        points : ndarray of shape (N, 2) — columns are (x, y).
        """
        n = points.shape[0]
        c_points = (Point2D * n)()
        for i in range(n):
            c_points[i].x = points[i, 0]
            c_points[i].y = points[i, 1]
        fid = self._dll.mhs_model_add_function_piecewise(
            self._handle,
            name.encode("utf-8"),
            c_points,
            n,
        )
        if fid == MHS_FUNCTION_ID_INVALID:
            err = self._dll.mhs_last_error()
            raise RuntimeError(f"add_function_piecewise failed: {err}")
        return fid

    def add_function_periodic_piecewise_constant(
        self, name: str, values: np.ndarray, period: float
    ) -> int:
        """Register a periodic piecewise-constant function."""
        n = values.shape[0]
        c_values = (ctypes.c_double * n)()
        for i in range(n):
            c_values[i] = values[i]
        fid = self._dll.mhs_model_add_function_periodic_piecewise_constant(
            self._handle,
            name.encode("utf-8"),
            c_values,
            n,
            period,
        )
        if fid == MHS_FUNCTION_ID_INVALID:
            err = self._dll.mhs_last_error()
            raise RuntimeError(
                f"add_function_periodic_piecewise_constant failed: {err}"
            )
        return fid

    # ---- Probes & fluid boundaries ----

    def add_probe(self, name: str, x: float, y: float, z: float) -> int:
        pid = self._dll.mhs_model_add_probe(
            self._handle,
            name.encode("utf-8"),
            x,
            y,
            z,
        )
        if pid == MHS_PROBE_ID_INVALID:
            err = self._dll.mhs_last_error()
            raise RuntimeError(f"add_probe failed: {err}")
        return pid

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
        rect = Rect2D(a_min, a_max, b_min, b_max)
        check(
            self._dll.mhs_model_add_fluid_boundary(
                self._handle,
                int(axis),
                coordinate,
                rect,
                int(kind),
                value,
                inlet_temperature,
            ),
            "add_fluid_boundary",
        )

    # ---- Material introspection ----

    def material_count(self) -> int:
        return self._dll.mhs_model_material_count(self._handle)

    def material_name(self, index: int) -> str:
        ptr = self._dll.mhs_model_material_name(self._handle, index)
        if not ptr:
            raise IndexError(f"material index {index} out of range")
        return ptr.decode("utf-8")

    # ---- Compile ----

    def compile(self) -> Compiled:
        """Compile the model into a read-only runtime representation."""
        from metahotspot.compiled import Compiled

        return Compiled._from_model(self._dll, self._handle)
