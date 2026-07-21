"""Python enums matching the MetaHotspot C API constants."""

from __future__ import annotations

import enum


class Status(enum.IntEnum):
    OK = 0
    ERR_NULL_PTR = -1
    ERR_LAYER = -2
    ERR_BLOCK = -3
    ERR_BOUNDARY = -4
    ERR_MATERIAL = -5
    ERR_FUNCTION = -6
    ERR_COMPILE = -7
    ERR_SOLVE = -8
    ERR_IO = -9
    ERR_OOM = -10
    ERR_UNSET = -11
    ERR_FLUID = -12
    ERR_MESH = -13
    ERR_VARIABLE = -14
    ERR_PROBE = -15


class Study(enum.IntEnum):
    STEADY = 0
    TRANSIENT = 1


class LengthUnit(enum.IntEnum):
    METER = 0
    MILLIMETER = 1
    MICROMETER = 2
    NANOMETER = 3
    INCH = 4
    MIL = 5


class Axis(enum.IntEnum):
    X = 0
    Y = 1
    Z = 2


class GeometryOp(enum.IntEnum):
    ADD = 0
    SUB = 1


class SolverType(enum.IntEnum):
    PARDISO = 0
    EIGEN_SPARSE_LU = 1
    EIGEN_BICGSTAB = 2


class FluidBC(enum.IntEnum):
    NONE = 0
    PRESSURE = 1
    MASS_FLOW = 2
    VELOCITY = 3
