r"""Auxiliary functions to set up logging in main scripts."""

import logging


def configure_logging(log, args):
    r"""Initialize logging."""
    logging.basicConfig(format="%(asctime)-15s [%(levelname)s] %(message)s")
    if hasattr(args, "log_level"):
        log.setLevel(args.log_level)
    else:
        log.setLevel(logging.INFO)
