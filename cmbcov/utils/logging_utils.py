"""
Logging configuration for the command-line entry points.

The library never configures logging: its modules log through
``logging.getLogger(__name__)`` and leave handlers and levels to the
application. The command-line scripts call :func:`configure_logging`; a Python
user who wants the run's progress calls it too, or
``logging.basicConfig(level=logging.INFO)``, or enables just the progress
lines with ``logging.getLogger("cmbcov.progress").setLevel(logging.INFO)``
(:mod:`cmbcov.utils.progress`).
"""

import logging

__all__ = ["configure_logging"]


def configure_logging(verbose: bool, quiet: bool = False) -> None:
    """
    Configure the root logger the way the command-line scripts do.

    Parameters
    ----------
    verbose : bool
        ``True`` shows the run's progress (``INFO`` and above); ``False``
        shows warnings and errors only.
    quiet : bool, default False
        ``True`` hides the progress lines of the long computations (the
        ``cmbcov.progress`` logger, :class:`~cmbcov.utils.progress.Progress`).
        By default they are shown, with or without ``verbose``.
    """
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("cmbcov.progress").setLevel(
        logging.WARNING if quiet else logging.INFO
    )
