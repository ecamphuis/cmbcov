"""
Logging configuration for the command-line entry points.

The library never configures logging: its modules log through
``logging.getLogger(__name__)`` and leave handlers and levels to the
application. The command-line scripts call :func:`configure_logging`; a Python
user who wants the run's progress calls it too, or
``logging.basicConfig(level=logging.INFO)``.
"""

import logging

__all__ = ["configure_logging"]


def configure_logging(verbose: bool) -> None:
    """
    Configure the root logger the way the command-line scripts do.

    Parameters
    ----------
    verbose : bool
        ``True`` shows the run's progress (``INFO`` and above); ``False``
        shows warnings and errors only.
    """
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
