"""High-level wrapper for ``mhs_model_t`` — model construction."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from metahotspot._lib import get_dll as _get_dll
from metahotspot._handle import OwnedHandle
import metahotspot._native as _native
from metahotspot.enums import FluidBC, GeometryOp, Study, LengthUnit, Axis
from metahotspot.types import MhsModel


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
        dll = _get_dll()
        super().__init__(dll.mhs_model_destroy, dll)

        self._handle: MhsModel = _native.create_model(dll)

    # ---- Model construction helpers ----

    def read_xml(self, path: str | Path) -> None:
        """Load a MetaHotspot XML case file."""
        _native.call(self._dll, "mhs_model_read_xml", self._handle, str(path))

    def set_settings(
        self,
        study: Study = Study.STEADY,
        length_unit: LengthUnit = LengthUnit.MILLIMETER,
        initial_temperature_K: float = 300.0,
        duration: float = 0.0,
        output_interval: float = 0.0,
    ) -> None:
        """Set global study parameters."""
        _native.set_settings(
            self._dll,
            self._handle,
            study,
            length_unit,
            initial_temperature_K,
            duration,
            output_interval,
        )

    def set_mesh(
        self,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        z: np.ndarray | None = None,
    ) -> None:
        """Set mesh vertices."""
        _native.set_mesh(self._dll, self._handle, x, y, z)

    def add_variable(self, name: str, expression: str) -> None:
        """Add a geometry variable."""
        _native.call(
            self._dll, "mhs_model_add_variable", self._handle, name, expression
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
        _native.call(
            self._dll,
            "mhs_model_add_material",
            self._handle,
            name,
            kx,
            ky,
            kz,
            rho,
            c,
            dynamic_viscosity,
        )

    def add_layer(
        self, thickness: str, x_offset: str = "0", y_offset: str = "0"
    ) -> int:
        """Add a layer.  Returns the layer ID."""
        return _native.add_id(
            self._dll,
            "mhs_model_add_layer",
            self._handle,
            thickness,
            x_offset,
            y_offset,
        )

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
        return _native.add_id(
            self._dll,
            "mhs_model_add_block",
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
        _native.call(
            self._dll,
            "mhs_model_add_rect",
            self._handle,
            block,
            int(op),
            x,
            y,
            width,
            height,
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
        _native.add_boundary(
            self._dll, "mhs_model_add_dirichlet", self._handle, regions, temperature
        )

    def add_neumann(
        self,
        heat_flux: str,
        regions: (
            list[tuple[int, float, float, float, float, float, float]] | None
        ) = None,
    ) -> None:
        """Add a Neumann (heat flux) boundary condition."""
        _native.add_boundary(
            self._dll, "mhs_model_add_neumann", self._handle, regions, heat_flux
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
        _native.add_boundary(
            self._dll,
            "mhs_model_add_convection",
            self._handle,
            regions,
            coefficient,
            ambient_temperature,
        )

    def set_default_dirichlet(self, temperature: str) -> None:
        _native.call(
            self._dll, "mhs_model_set_default_dirichlet", self._handle, temperature
        )

    def set_default_neumann(self, heat_flux: str) -> None:
        _native.call(
            self._dll, "mhs_model_set_default_neumann", self._handle, heat_flux
        )

    def set_default_convection(
        self, coefficient: str, ambient_temperature: str
    ) -> None:
        _native.call(
            self._dll,
            "mhs_model_set_default_convection",
            self._handle,
            coefficient,
            ambient_temperature,
        )

    # ---- Functions ----

    def add_function_expr(self, name: str, expression: str) -> None:
        _native.call(
            self._dll, "mhs_model_add_function_expr", self._handle, name, expression
        )

    def add_function_gauss(
        self, name: str, amplitude: float, tau: float, center: float
    ) -> None:
        _native.call(
            self._dll,
            "mhs_model_add_function_gauss",
            self._handle,
            name,
            amplitude,
            tau,
            center,
        )

    def add_function_sine(
        self, name: str, amplitude: float, angular_frequency: float, phase: float
    ) -> None:
        _native.call(
            self._dll,
            "mhs_model_add_function_sine",
            self._handle,
            name,
            amplitude,
            angular_frequency,
            phase,
        )

    def add_function_double_exponential(
        self, name: str, amplitude: float, alpha: float, beta: float
    ) -> None:
        _native.call(
            self._dll,
            "mhs_model_add_function_double_exponential",
            self._handle,
            name,
            amplitude,
            alpha,
            beta,
        )

    def add_function_piecewise(self, name: str, points: np.ndarray) -> None:
        """Register a piecewise-linear function.

        Parameters
        ----------
        points : ndarray of shape (N, 2) — columns are (x, y).
        """
        _native.add_piecewise(self._dll, self._handle, name, points)

    def add_function_periodic_piecewise_constant(
        self, name: str, values: np.ndarray, period: float
    ) -> None:
        """Register a periodic piecewise-constant function."""
        _native.add_periodic_constant(self._dll, self._handle, name, values, period)

    # ---- Probes & fluid boundaries ----

    def add_probe(self, name: str, x: float, y: float, z: float) -> None:
        _native.call(self._dll, "mhs_model_add_probe", self._handle, name, x, y, z)

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
        _native.add_fluid_boundary(
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
