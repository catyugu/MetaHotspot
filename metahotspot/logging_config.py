"""
Logging configuration for MetaHotspot.

Usage:
    from metahotspot.logging_config import get_logger
    logger = get_logger(__name__)

Log format: [LEVEL] message
Handlers: console only (stderr)
"""

import logging
import sys

# Module-level log level - can be overridden via set_level()
_log_level = logging.INFO


def set_level(level: int) -> None:
    """Set global log level (e.g., logging.DEBUG, logging.INFO)."""
    global _log_level
    _log_level = level


def get_logger(name: str, level: int | None = None) -> logging.Logger:
    """
    Get or create a logger with consistent formatting.

    Args:
        name: Logger name (typically __name__ from the calling module)
        level: Optional override for this logger's level

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    if level is not None:
        logger.setLevel(level)
    else:
        logger.setLevel(_log_level)

    # Only add handler if logger doesn't already have one
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level if level is not None else _log_level)

        formatter = logging.Formatter(
            fmt="[%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Prevent propagation to root logger (avoids duplicate output)
    logger.propagate = False

    return logger
