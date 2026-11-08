"""Solve-time study configuration, independent of compiled physics."""

from dataclasses import dataclass

from metahotspot.enums import Study


@dataclass(frozen=True)
class StudyOptions:
    """Study type, duration and output interval in seconds.

    A zero output interval records accepted solver steps.
    """

    type: Study = Study.STEADY
    duration: float = 0.0
    output_interval: float = 1.0
