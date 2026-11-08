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
    SPARSE_LU = 2


class ComponentKind(enum.IntEnum):
    CARTESIAN = 0
    REDUCED = 1


class ThermalBoundaryKind(enum.IntEnum):
    DIRICHLET = 0
    NEUMANN = 1
    CONVECTION = 2


class FlowMode(enum.IntEnum):
    FROZEN = 0
    STOKES = 1


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


class FieldQuantity(enum.IntEnum):
    TEMPERATURE = 0
    PRESSURE = 1
    VELOCITY = 2
    REDUCED_COORDINATE = 3


class EntitySupport(enum.IntEnum):
    CELL = 0
    FACE = 1
    REDUCED = 2
