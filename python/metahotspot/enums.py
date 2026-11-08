"""Python enums matching the MetaHotspot C API constants."""

from __future__ import annotations

import enum


class StudyType(enum.IntEnum):
    STEADY = 0
    TRANSIENT = 1


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
    PRESSURE_INLET = 1
    MASS_FLOW_INLET = 2
    OUTLET = 3


class LogLevel(enum.IntEnum):
    TRACE = 0
    DEBUG = 1
    INFO = 2
    WARN = 3
    ERROR = 4
    CRITICAL = 5
    OFF = 6
