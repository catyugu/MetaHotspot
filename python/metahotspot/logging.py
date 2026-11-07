"""Process-wide MetaHotspot logging configuration."""

from metahotspot._error import check
from metahotspot._lib import get_dll
from metahotspot.enums import LogLevel


def set_log_level(level: LogLevel) -> None:
    """Set MetaHotspot's runtime minimum level, overriding MHS_LOG_LEVEL."""
    check(get_dll().mhs_log_set_level(int(level)), "set_log_level")


def get_log_level() -> LogLevel:
    """Return MetaHotspot's current logging level."""
    return LogLevel(get_dll().mhs_log_get_level())
