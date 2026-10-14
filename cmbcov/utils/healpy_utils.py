"""
HEALPix and spherical harmonic utility functions.

This module contains utilities for HEALPix map manipulation and
spherical harmonic operations commonly used in CMB analysis.

healpy is imported lazily across the package
---------------------------------------------
``import healpy`` pulls in matplotlib (``healpy.visufunc``) and
``astropy.coordinates`` (``healpy.rotator``), which dominates the import time
of ``cmbcov``. The package uses healpy only for FITS I/O (``read_map``,
``write_cl``, ``read_cl``), the pixel window, ``gauss_beam``, pixel-count
checks and the ``Alm`` index helpers, so no module imports it at top level.
Each function that needs it does ``import healpy as hp`` in its own body.
A process that never reaches one of those functions never loads healpy.
"""

import numpy as np


def get_nside_from_ell(ell: int, safety_factor: int = 100) -> int:
    """
    Get appropriate nside for a given maximum multipole.

    Parameters
    ----------
    ell : int
        Maximum multipole moment
    safety_factor : int, optional
        Additional buffer for reliable computation

    Returns
    -------
    int
        Appropriate nside value
    """
    min_nside = max(2 ** (int(np.log2(max(ell + safety_factor, 2)))), 128)
    return min_nside


def ensure_ducc0_compatible_dtype(
    arr: np.ndarray, target_dtype: str | None = None
) -> np.ndarray:
    """
    Ensure array has compatible dtype for ducc0 operations.

    ducc0 requires arrays to have dtype 'f4' (float32) or 'f8' (float64).

    Parameters
    ----------
    arr : np.ndarray
        Input array
    target_dtype : str, optional
        Target dtype ('f4' or 'f8'). Default: 'f8' (float64)

    Returns
    -------
    np.ndarray
        Array with compatible dtype

    Raises
    ------
    ValueError
        If target_dtype is not 'f4' or 'f8'
    """
    if target_dtype is None:
        target_dtype = "f8"  # Default to float64 for better precision

    if target_dtype not in ["f4", "f8"]:
        raise ValueError(f"target_dtype must be 'f4' or 'f8', got {target_dtype}")

    dtype_map = {"f4": np.float32, "f8": np.float64}
    return np.asarray(arr, dtype=dtype_map[target_dtype])
