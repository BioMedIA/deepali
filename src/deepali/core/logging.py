r"""Utilities for setting up logging in a main scripts."""

from __future__ import annotations

from argparse import Namespace
from enum import Enum
import logging
from logging import Logger
from typing import Optional, Union


class LogLevel(str, Enum):
    r"""Enumeration of logging levels.

    In particular, this enumeration can be used for type annotation of log level argument when using Typer.

    """

    NOTSET = "NOTSET"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    def __str__(self) -> str:
        return self.value

    def __int__(self) -> int:
        r"""Cast enumeration to int logging level."""
        return int(getattr(logging, self.value))

    def __eq__(self, other: Union[LogLevel, int]) -> bool:
        return int(self) == int(other)

    def __lt__(self, other: Union[LogLevel, int]) -> bool:
        return int(self) < int(other)

    def __le__(self, other: Union[LogLevel, int]) -> bool:
        return int(self) <= int(other)

    def __gt__(self, other: Union[LogLevel, int]) -> bool:
        return int(self) > int(other)

    def __ge__(self, other: Union[LogLevel, int]) -> bool:
        return int(self) <= int(other)


LOG_FORMAT = "%(asctime)-15s [%(levelname)s] %(message)s"
LOG_LEVELS = tuple(log_level.value for log_level in LogLevel)


def configure_logging(
    logger: Logger,
    args: Optional[Namespace] = None,
    log_level: Optional[Union[int, str, LogLevel]] = None,
    format: Optional[str] = None,
):
    r"""Initialize logging."""
    logging.basicConfig(format=format or LOG_FORMAT)
    if log_level is None:
        log_level = logging.INFO
    if args is not None:
        log_level = getattr(args, "log_level", log_level)
    if isinstance(log_level, str):
        log_level = LogLevel(log_level)
    logger.setLevel(int(log_level))
    logging.getLogger("botocore").setLevel(logging.ERROR)
    logging.getLogger("s3transfer").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.ERROR)
