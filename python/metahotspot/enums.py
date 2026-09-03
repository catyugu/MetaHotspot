"""Python enums matching the MetaHotspot C API constants."""

from __future__ import annotations

import enum


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


class Face(enum.IntEnum):
    XM = 0
    XP = 1
    YM = 2
    YP = 3
    ZM = 4
    ZP = 5


class GeometryOp(enum.IntEnum):
    ADD = 0
    SUB = 1


class SolverType(enum.IntEnum):
    PARDISO = 0
    AMG = 1


class FluidBC(enum.IntEnum):
    NONE = 0
    PRESSURE = 1
    MASS_FLOW = 2
    VELOCITY = 3


class IntegratorKind(enum.IntEnum):
    BDF1 = 0
    BDF2 = 1


class StepStrategy(enum.IntEnum):
    ADAPTIVE = 0
    FIXED = 1
