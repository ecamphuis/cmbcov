"""
Utility functions for CMB covariance analysis.

This package re-exports only the handful of names imported this way by the
rest of the package (see ``__all__``). Everything else -- file, HEALPix and
threading utilities -- is imported from its submodule directly, e.g.
``from .utils.file_utils import get_git_revision_short_hash`` or
``from .utils.threading_utils import get_optimal_nthreads``.
"""

from .array_utils import (
    SPECTRUM_UNITS,
    add_nested_dicts,
    dl_to_cl,
    mult_nested_dicts,
    read_spectrum_file,
    safe_divide,
)

__all__ = [
    "SPECTRUM_UNITS",
    "safe_divide",
    "read_spectrum_file",
    "dl_to_cl",
    "add_nested_dicts",
    "mult_nested_dicts",
]
