from typing import NamedTuple, Optional


class S3Config(NamedTuple):
    r"""Configuration of AWS Simple Storage Service."""

    mode: str = "r"
    verify: bool = True
    region: Optional[str] = None
