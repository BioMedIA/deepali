r"""Auxiliary functions for implementing argparse based command line interfaces."""

from .environ import cuda_visible_devices
from .environ import check_cuda_visible_devices
from .environ import init_omp_num_threads

from .logging import LOG_FORMAT
from .logging import LogLevel
from .logging import configure_logging

from .argparse import ArgumentParser
from .argparse import entry_point
from .argparse import main_func
from .argparse import MainCallable
from .argparse import ParserCallable
from .argparse import ParsedArguments
from .argparse import UnknownArguments

from .warnings import filter_warning_of_experimental_named_tensors_feature


Args = ParsedArguments


__all__ = (
    "Args",
    "ArgumentParser",
    "LogLevel",
    "LOG_FORMAT",
    "entry_point",
    "check_cuda_visible_devices",
    "configure_logging",
    "cuda_visible_devices",
    "filter_warning_of_experimental_named_tensors_feature",
    "init_omp_num_threads",
    "main_func",
    "MainCallable",
    "ParsedArguments",
    "ParserCallable",
    "UnknownArguments",
)
