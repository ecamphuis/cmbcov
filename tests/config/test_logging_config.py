"""
The library never configures logging; only the command-line entry points do.

Importing cmbcov and constructing the validator must leave the root logger
and the library loggers as they were, and ``configure_logging`` (what the
scripts call) sets the root level.
"""

import logging

import pytest

import cmbcov  # noqa: F401
from cmbcov.config.validation import ParameterValidator
from cmbcov.utils import configure_logging


@pytest.fixture
def restore_root_logger():
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield root
    for handler in list(root.handlers):
        if handler not in handlers:
            root.removeHandler(handler)
    for handler in handlers:
        if handler not in root.handlers:
            root.addHandler(handler)
    root.setLevel(level)


def test_importing_and_validating_leaves_logging_alone(restore_root_logger):
    root = restore_root_logger
    handlers, level = list(root.handlers), root.level
    ParameterValidator()
    assert root.level == level
    assert root.handlers == handlers
    library = logging.getLogger("cmbcov.config.validation")
    assert library.level == logging.NOTSET
    assert logging.getLogger("cmbcov.config.generator").level == logging.NOTSET


@pytest.mark.parametrize(
    "verbose, expected", [(False, logging.WARNING), (True, logging.INFO)]
)
def test_configure_logging_sets_the_root_level(restore_root_logger, verbose, expected):
    root = restore_root_logger
    # basicConfig does nothing when the root has handlers (pytest's own):
    # start from a bare root, as in a script.
    for handler in list(root.handlers):
        root.removeHandler(handler)
    configure_logging(verbose)
    assert root.level == expected
