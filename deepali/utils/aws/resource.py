r"""Interface for resources stored locally, remotely, or in cloud storage."""

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import re
import shutil
from typing import Generator, Optional, TypeVar, Union
from urllib.parse import urlsplit

PathStr = Union[Path, str]


# See https://mypy.readthedocs.io/en/latest/generics.html#generic-methods-and-generic-self for the use
# of `T` to annotate `self`. Many methods of `Resource` return `self` and we want those return values
# to be the type of the subclass, not the looser type of `Resource`.
T = TypeVar("T", bound="Resource")


def to_uri(arg: PathStr) -> str:
    r"""Create valid URI from resource path.

    Args:
        arg: Local path or a valid URI.

    Returns:
        Valid URI.

    """
    # Path
    if isinstance(arg, Path):
        return arg.absolute().as_uri()
    # Windows path with drive letter
    if os.name == "nt":
        match = re.match(r"([a-zA-Z]):[/\\](.*)", arg)
        if match:
            return Path(match.group(1) + ":/" + match.group(2)).as_uri()
    # URI with scheme prefix
    match = re.match(r"([a-zA-Z0-9]+)://(.*)", arg)
    if match:
        scheme = match.group(1).lower()
        # Local file URI
        if scheme == "file":
            return Path("/" + match.group(2)).absolute().as_uri()
        # AWS S3 object URI
        if scheme == "s3":
            return "s3://" + re.sub("^/+", "", re.sub(r"[/\\]{1,}", "/", match.group(2)))
        # Other URI
        return urlsplit(arg, scheme="file").geturl()
    # Unix path or relative path
    return Path(arg).absolute().as_uri()


class Resource(object):
    r"""Interface for storage objects.

    This base class can be used for storage objects that are only stored locally.
    The base implementations of the ``Resource`` interface functions reflect this use
    case, where ``Resource("/path/to/file")`` represents such local path object.
    The factory function ``Resource.from_uri`` is recommended for creating concrete
    instances of ``Resource`` or one of its subclasses. To create a resource instance
    for a local file path, use ``Resource.from_uri("file:///path/to/file")``.
    An S3 object resource is created by ``Resource.from_uri("s3://bucket/key")``.
    By using the ``Resource`` interface when writing tools that read and write
    from either local, remote, or cloud storage, the tool CLI can create these
    resource instances from input argument URIs or local file path strings
    without URI scheme, i.e., without "file://" prefix. The consumer or producer
    of a resource object can either directly read/write the object data using
    the ``read_(bytes|text)`` and/or ``write_(bytes|text)`` functions, or
    download/upload the storage object to/from a local file path using the
    ``pull`` and ``push`` operations. Note that these operations directly interact
    with the local storage if the resource instance is of base type ``Resource``,
    rather than a remote or cloud storage specific subclass. The ``pull`` and ``push``
    operations should be preferred over ``read`` and ``write`` if the resource data
    is accessed multiple times, in order to take advantage of the local temporary
    copy of the resource object. Otherwise, system IO operations can be saved by
    using the direct ``read`` and ``write`` operations instead.

    Additionally, the ``Resource.release`` function should be called by tools when
    a resource is no longer required to indicate that the local copy of this resource
    can be removed. If the resource object itself represents a local ``Resource``,
    the release operation has no effect. To ensure that the ``release`` function is
    called also in case of an exception, the ``Resource`` class implements the context
    manager interface functions ``__enter__`` and ``__exit__``.

    Example usage with resource context:

    .. code-block:: python

        with Resource.from_uri("s3://bucket/key") as res:
            # request download to local storage
            path = res.pull().path
            # use local storage object referenced by path
        # local copy of storage object has been deleted

    The above is equivalent to using a try-finally block:

    .. code-block:: python

        res = Resource.from_uri("s3://bucket/key")
        try:
            path = res.pull().path
            # use local storage object referenced by path
        finally:
            # delete local copy of storage object
            res.release()

    Usage of the ``with`` statement is recommended.

    Nesting of contexts for a resource object is possible and post-pones the
    invocation of the ``release`` operation until the outermost context has
    been left. This is accomplished by using a counter that is increment by
    ``__enter__``, and decremented again by ``__exit__``.

    It should be noted that ``Resource`` operations are generally not thread-safe,
    and actual consumers of resource objects should require the main thread to
    deal with obtaining, downloading (if ``pull`` is used), and releasing a resource.
    For different resources from remote storage (e.g., AWS S3), when using multiple
    processes (threads), the main process (thread) must initialize the default client
    connection (e.g., using ``S3Client.init_default()``) before spawning processes.

    """

    def __init__(self: T, path: PathStr) -> None:
        r"""Initialize storage object.

        Args:
            path (str, pathlib.Path): Local path of storage object.

        """
        self._path = Path(path).absolute()
        self._depth = 0

    def __enter__(self: T) -> T:
        r"""Enter context."""
        self._depth = max(1, self._depth + 1)
        return self

    def __exit__(self: T, *exc) -> None:
        r"""Release resource when leaving outermost context."""
        self._depth = max(0, self._depth - 1)
        if self._depth == 0:
            self.release()

    @staticmethod
    def from_uri(uri: str) -> Resource:
        r"""Create storage object from URI.

        Args:
            uri: URI of storage object.

        Returns:
            obj (Resource): Instance of concrete type representing the referenced storage object.

        """
        res = urlsplit(uri, scheme="file")
        if res.scheme == "file":
            match = re.match(r"/+([a-zA-Z]:.*)", res.path)
            path = match.group(1) if match else res.path
            return Resource(Path("/" + res.netloc + "/" + path if res.netloc else path))
        if res.scheme == "s3":
            # DO NOT import at module level to avoid cyclical import!
            from .s3.object import S3Object

            return S3Object.from_uri(uri)
        raise ValueError("Invalid or unsupported storage object URI: %s", uri)

    @property
    def uri(self: T) -> str:
        r"""
        Returns:
            uri (str): URI of storage object.

        """
        return self.path.as_uri()

    @property
    def path(self: T) -> Path:
        r"""Get absolute local path of storage object."""
        return self._path

    @property
    def name(self: T) -> str:
        r"""Name of storage object including file name extension, excluding directory path."""
        return self.path.name

    def with_path(self: T, path) -> T:
        r"""Create copy of storage object reference with modified ``path``.

        Args:
            path (str, pathlib.Path): New local path of storage object.

        Returns:
            self: New storage object reference with modified ``path`` property.

        """
        obj = deepcopy(self)
        obj._path = Path(path).absolute()
        return obj

    def with_properties(self: T, **kwargs) -> T:
        r"""Create copy of storage object reference with modified properties.

        Args:
            **kwargs: New property values. Only specified properties are changed.

        Returns:
            self: New storage object reference with modified properties.

        """
        obj = deepcopy(self)
        for name, value in kwargs.items():
            setattr(obj, name, value)
        return obj

    def exists(self: T) -> bool:
        r"""Whether object exists in storage."""
        return self.path.exists()

    def is_file(self: T) -> bool:
        r"""Whether storage object represents a file."""
        return self.path.is_file()

    def is_dir(self: T) -> bool:
        r"""Whether storage object represents a directory."""
        return self.path.is_dir()

    def iterdir(self: T, prefix: Optional[str] = None) -> Generator[T, None, None]:
        r"""List storage objects within directory, excluding subfolder contents.

        Args:
            prefix: Name prefix.

        Returns:
            iterable: Generator of storage objects.

        """
        assert type(self) is Resource, "must be implemented by subclass"
        for path in self.path.iterdir():
            if not prefix or path.name.startswith(prefix):
                yield Resource(path)

    def pull(self: T, force: bool = False) -> T:
        r"""Download content of storage object to local path.

        Args:
            force (bool): Whether to force download even if local path already exists.

        Returns:
            self: This storage object.

        """
        return self

    def push(self: T, force: bool = False) -> T:
        r"""Upload content of local path to storage object.

        Args:
            force (bool): Whether to force upload even if storage object already exists.

        Returns:
            self: This storage object.

        """
        return self

    def read_bytes(self: T) -> bytes:
        r"""Read file content from local path if it exists, or referenced storage object otherwise.

        Returns:
            data (bytes): Binary file content of storage object.

        """
        return self.pull().path.read_bytes()

    def write_bytes(self: T, data: bytes) -> T:
        r"""Write bytes to storage object.

        Args:
            data (bytes): Binary data to write.

        Returns:
            self: This storage object.

        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(data)
        return self.push()

    def read_text(self: T, encoding: Optional[str] = None) -> str:
        r"""Read text file content from local path if it exists, or referenced storage object otherwise.

        Args:
            encoding (str): Text encoding.

        Returns:
            text (str): Decoded text file content of storage object.

        """
        return self.pull().path.read_text()

    def write_text(self: T, text: str, encoding: Optional[str] = None) -> T:
        r"""Write text to storage object.

        Args:
            text (str): Text to write.
            encoding (str): Text encoding.

        Returns:
            self: This storage object.

        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(text, encoding=encoding)
        return self.push()

    def rmdir(self: T) -> T:
        r"""Remove directory both locally and from remote storage."""
        try:
            shutil.rmtree(self.path)
        except FileNotFoundError:
            pass
        return self

    def unlink(self: T) -> T:
        r"""Remove file both locally and from remote storage."""
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        return self

    def delete(self: T) -> T:
        r"""Remove object both locally and from remote storage."""
        try:
            self.rmdir()
        except NotADirectoryError:
            self.unlink()
        return self

    def release(self: T) -> T:
        r"""Release local temporary copy of storage object.

        Only remove local copy of storage object. When the storage object
        is only stored locally, i.e., self is not a subclass of Resource,
        but of type ``Resource``, this operation does nothing.

        """
        if type(self) is not Resource:
            try:
                shutil.rmtree(self.path)
            except FileNotFoundError:
                pass
            except NotADirectoryError:
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
        return self

    def __str__(self: T) -> str:
        r"""Get human-readable string representation of storage object reference."""
        return self.uri

    def __repr__(self: T) -> str:
        r"""Get human-readable string representation of storage object reference."""
        return type(self).__name__ + "(path='{}')".format(self.path)
