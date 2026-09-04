from __future__ import annotations

import ctypes

import numpy as np

from metahotspot._error import check
from metahotspot.types import (
    MhsFaceRegion,
    MhsModel,
    Point2D,
    Rect2D,
)


def _text(value: str | None) -> bytes | None:
    return None if value is None else value.encode("utf-8")


def _double_ptr(array: np.ndarray):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


# ---------------------------------------------------------------------------


def create_model(dll):
    handle = ctypes.POINTER(MhsModel)()
    check(dll.mhs_model_create(ctypes.byref(handle)), "create")
    return handle


def set_settings(
    dll, handle, study, length_unit, initial_temperature, duration, output_interval
) -> None:
    check(
        dll.mhs_model_set_settings(
            handle,
            int(study),
            int(length_unit),
            initial_temperature,
            duration,
            output_interval,
        ),
        "set_settings",
    )


def read_xml(dll, handle, path: str) -> None:
    check(dll.mhs_model_read_xml(handle, _text(path)), "read_xml")


def add_variable(dll, handle, name: str, expression: str) -> None:
    check(
        dll.mhs_model_add_variable(handle, _text(name), _text(expression)),
        "add_variable",
    )


def add_material(dll, handle, name, kx, ky, kz, rho, specific_heat, dynamic_viscosity):
    check(
        dll.mhs_model_add_material(
            handle,
            *(_text(value) for value in (name, kx, ky, kz, rho, specific_heat)),
            _text(dynamic_viscosity),
        ),
        "add_material",
    )


def set_mesh(dll, handle, x, y, z) -> None:
    arrays = (x, y, z)
    args = []
    for array in arrays:
        args.extend(
            (
                0 if array is None else len(array),
                None if array is None else _double_ptr(array),
            )
        )
    check(dll.mhs_model_set_mesh(handle, *args), "set_mesh")


def add_layer(dll, handle, thickness, x_offset, y_offset) -> int:
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


def add_block(
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


def add_rect(dll, handle, block, op, x, y, width, height) -> None:
    check(
        dll.mhs_model_add_rect(
            handle, block, int(op), _text(x), _text(y), _text(width), _text(height)
        ),
        "add_rect",
    )


def face_regions(regions) -> tuple[object, int]:
    values = []
    for axis, coordinate, a_min, a_max, b_min, b_max in regions or ():
        values.append(
            MhsFaceRegion(int(axis), coordinate, Rect2D(a_min, a_max, b_min, b_max))
        )
    return ((MhsFaceRegion * len(values))(*values) if values else None), len(values)


def add_dirichlet(dll, handle, regions, temperature) -> None:
    native_regions, count = face_regions(regions)
    check(
        dll.mhs_model_add_dirichlet(handle, native_regions, count, _text(temperature)),
        "add_dirichlet",
    )


def add_neumann(dll, handle, regions, heat_flux) -> None:
    native_regions, count = face_regions(regions)
    check(
        dll.mhs_model_add_neumann(handle, native_regions, count, _text(heat_flux)),
        "add_neumann",
    )


def add_convection(dll, handle, regions, coefficient, ambient_temperature) -> None:
    native_regions, count = face_regions(regions)
    check(
        dll.mhs_model_add_convection(
            handle,
            native_regions,
            count,
            _text(coefficient),
            _text(ambient_temperature),
        ),
        "add_convection",
    )


def set_default_dirichlet(dll, handle, temperature) -> None:
    check(
        dll.mhs_model_set_default_dirichlet(handle, _text(temperature)),
        "set_default_dirichlet",
    )


def set_default_neumann(dll, handle, heat_flux) -> None:
    check(
        dll.mhs_model_set_default_neumann(handle, _text(heat_flux)),
        "set_default_neumann",
    )


def set_default_convection(dll, handle, coefficient, ambient_temperature) -> None:
    check(
        dll.mhs_model_set_default_convection(
            handle, _text(coefficient), _text(ambient_temperature)
        ),
        "set_default_convection",
    )


def add_function_expr(dll, handle, name, expression) -> None:
    check(
        dll.mhs_model_add_function_expr(handle, _text(name), _text(expression)),
        "add_function_expr",
    )


def add_function_gauss(dll, handle, name, amplitude, tau, center) -> None:
    check(
        dll.mhs_model_add_function_gauss(handle, _text(name), amplitude, tau, center),
        "add_function_gauss",
    )


def add_function_sine(dll, handle, name, amplitude, angular_frequency, phase) -> None:
    check(
        dll.mhs_model_add_function_sine(
            handle, _text(name), amplitude, angular_frequency, phase
        ),
        "add_function_sine",
    )


def add_function_double_exponential(dll, handle, name, amplitude, alpha, beta) -> None:
    check(
        dll.mhs_model_add_function_double_exponential(
            handle, _text(name), amplitude, alpha, beta
        ),
        "add_function_double_exponential",
    )


def add_probe(dll, handle, name, x, y, z) -> None:
    check(dll.mhs_model_add_probe(handle, _text(name), x, y, z), "add_probe")


def add_piecewise(dll, handle, name: str, points: np.ndarray) -> None:
    native = (Point2D * len(points))(*[Point2D(x, y) for x, y in points])
    check(
        dll.mhs_model_add_function_piecewise(handle, _text(name), native, len(points)),
        "add_function_piecewise",
    )


def add_periodic_constant(
    dll, handle, name: str, values: np.ndarray, period: float
) -> None:
    native = (ctypes.c_double * len(values))(*values)
    check(
        dll.mhs_model_add_function_periodic_piecewise_constant(
            handle, _text(name), native, len(values), period
        ),
        "add_function_periodic_piecewise_constant",
    )


def add_fluid_boundary(
    dll, handle, axis, coordinate, region, kind, value, inlet_temperature
) -> None:
    rect = Rect2D(*region)
    check(
        dll.mhs_model_add_fluid_boundary(
            handle, int(axis), coordinate, rect, int(kind), value, inlet_temperature
        ),
        "add_fluid_boundary",
    )
