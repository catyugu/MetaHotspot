"""Python enums matching the MetaHotspot C API constants."""

from __future__ import annotations

import enum


class Status(enum.IntEnum):
    OK = 0
    ERR_NULL_PTR = -1
    ERR_INVALID_ARG = -2
    ERR_COMPILE = -3
    ERR_ASSEMBLE = -4
    ERR_SOLVE = -5
    ERR_IO = -6
    ERR_OOM = -7
    ERR_UNSET = -8
    ERR_RUNTIME = -9


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


class Operator(enum.IntEnum):
    STIFFNESS = 0
    CAPACITY = 1
