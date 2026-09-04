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


def call(dll, name: str, handle, *args) -> None:
    args = tuple(_text(arg) if isinstance(arg, str) else arg for arg in args)
    check(getattr(dll, name)(handle, *args), name.removeprefix("mhs_model_"))


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


def add_id(dll, name: str, handle, *args) -> int:
    identifier = ctypes.c_uint32()
    call_args = tuple(_text(arg) if isinstance(arg, str) else arg for arg in args)
    check(
        getattr(dll, name)(handle, *call_args, ctypes.byref(identifier)),
        name.removeprefix("mhs_model_"),
    )
    return identifier.value


def face_regions(regions) -> tuple[object, int]:
    values = []
    for axis, coordinate, a_min, a_max, b_min, b_max in regions or ():
        values.append(
            MhsFaceRegion(int(axis), coordinate, Rect2D(a_min, a_max, b_min, b_max))
        )
    return ((MhsFaceRegion * len(values))(*values) if values else None), len(values)


def add_boundary(dll, name: str, handle, regions, *expressions) -> None:
    native_regions, count = face_regions(regions)
    args = tuple(_text(expression) for expression in expressions)
    check(
        getattr(dll, name)(handle, native_regions, count, *args),
        name.removeprefix("mhs_model_"),
    )


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
