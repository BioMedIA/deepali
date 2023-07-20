r"""Utilities for working with temporary files and directories."""

from contextlib import contextmanager
import os
from pathlib import Path
import shutil
from tempfile import mkdtemp, mkstemp
from typing import Generator, Optional, Union

from .pathlib import PathStr, delete


__all__ = (
    "make_temp_file",
    "temp_dir",
    "temp_file",
)


def make_temp_file(
    suffix: Optional[str] = None,
    prefix: Optional[str] = None,
    dir: Optional[PathStr] = None,
    text: bool = False,
) -> Path:
    r"""Make temporary file with mkstemp, but close open file handle immediately."""
    fp, path = mkstemp(suffix=suffix, prefix=prefix, dir=dir, text=text)
    os.close(fp)
    return Path(path)


@contextmanager
def temp_dir(
    suffix: str = None, prefix: str = None, dir: Union[Path, str] = None
) -> Generator[Path, None, None]:
    r"""Create temporary directory within context."""
    path = mkdtemp(suffix=suffix, prefix=prefix, dir=dir)
    try:
        yield Path(path)
    finally:
        shutil.rmtree(path)


@contextmanager
def temp_file(
    suffix: Optional[str] = None,
    prefix: Optional[str] = None,
    dir: PathStr = None,
    text: bool = False,
) -> Generator[Path, None, None]:
    r"""Create temporary file within context."""
    path = make_temp_file(suffix=suffix, prefix=prefix, dir=dir, text=text)
    try:
        yield path
    finally:
        delete(path)
