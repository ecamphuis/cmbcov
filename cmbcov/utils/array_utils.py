"""
Array manipulation utility functions.

This module contains utilities for array operations commonly used
in covariance matrix computation.
"""

import os
import warnings
from typing import Any

import numpy as np

#: The first ``.fits`` spectrum read in this process, once it has been warned
#: about (a run reads one file per frequency pair).
_WARNED_FITS_READ: list = []

#: Accepted values of the ``spectrum_units`` parameter-file key and of its
#: per-input overrides: the convention a *tabulated* spectrum file is written
#: in. ``"Cl"`` is :math:`C_\ell` itself (the package's internal convention,
#: and the historical behaviour: no conversion); ``"Dl"`` is
#: :math:`D_\ell = \ell(\ell+1)C_\ell/2\pi`, which is what CAMB's
#: ``*_lensedCls.dat`` and most published spectra carry.
SPECTRUM_UNITS = ("Cl", "Dl")


def dl_to_cl(values: np.ndarray) -> np.ndarray:
    """
    Convert :math:`D_\\ell = \\ell(\\ell+1)C_\\ell/2\\pi` to :math:`C_\\ell`.

    Parameters
    ----------
    values : ndarray
        ``(nell,)`` or ``(nell, nspec)``, indexed by multipole from
        ``ell = 0`` -- the layout :func:`read_spectrum_file` returns.

    Returns
    -------
    ndarray
        ``2 pi values / (ell (ell + 1))``, a new array of the same shape and
        of floating dtype.

    Notes
    -----
    ``ell = 0`` and ``ell = 1`` are set to **zero**, not left as they were and
    not extrapolated: :math:`\\ell(\\ell+1)` vanishes at ``ell = 0`` and the
    conversion is undefined there, while at ``ell = 1`` the monopole/dipole
    convention of a :math:`D_\\ell` file is not fixed (CAMB files start at
    ``ell = 2``, so both entries are padding). Neither multipole carries
    signal in this package: the covariance is reported from ``lmin >= 2`` and
    a non-zero ``C_0``/``C_1`` would only add mask-coupled leakage from modes
    that do not exist. Note that the ``Cl`` convention leaves them untouched,
    so the two conventions genuinely differ at ``ell < 2`` for a file whose
    first two rows are not zero.
    """
    values = np.asarray(values, dtype=float)
    ell = np.arange(values.shape[0], dtype=float)
    denominator = ell[2:] * (ell[2:] + 1.0)
    output = np.zeros(values.shape, dtype=float)
    if values.ndim == 1:
        output[2:] = 2.0 * np.pi * values[2:] / denominator
    else:
        # `2 pi D_l / (l(l+1))` as written, one rounding: the equivalent
        # `D_l * (2 pi / (l(l+1)))` differs from it by an ulp, which would
        # make a converted D_l file and the same spectrum tabulated as C_l
        # disagree in the last bit of the covariance.
        output[2:] = 2.0 * np.pi * values[2:] / denominator[:, np.newaxis]
    return output


def read_spectrum_file(file_path: str, lmax: int) -> np.ndarray:
    """
    Read a power spectrum from a file and ensure it has length lmax.

    Parameters
    ----------
    file_path : str
        Path to the spectrum file
    lmax : int
        Desired maximum multipole

    Returns
    -------
    np.ndarray
        Spectrum array of shape (lmax, n_spectra)

    Raises
    ------
    ValueError
        If file format is unsupported or data is invalid
    FileNotFoundError
        If the file does not exist

    Notes
    -----
    The ``.npy`` branch is an intentional unimplemented stub: it always
    raises ``NotImplementedError``.
    """
    import healpy as hp  # lazy, see utils/healpy_utils.py

    if not isinstance(file_path, str):
        raise TypeError("file_path must be a string")
    if not isinstance(lmax, int) or lmax <= 0:
        raise ValueError("lmax must be a positive integer")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.endswith(".npy"):
        raise NotImplementedError("Reading .npy files is not yet implemented")
    elif file_path.endswith(".fits"):
        if not _WARNED_FITS_READ:
            _WARNED_FITS_READ.append(file_path)
            warnings.warn(
                "Reading .fits assumes the file starts at ell=0 and is "
                f"consecutive (first: {file_path!r}). Relies on hp.read_cl. "
                "Warned once per process.",
                stacklevel=2,
            )
        # hp.read_cl gives (nell,) for one spectrum and (nspec, nell) for
        # several; this function returns (lmax, nspec) either way, the layout
        # of the .txt branch below (ell down the rows).
        spectrum = np.asarray(hp.read_cl(file_path))
        if spectrum.ndim == 1:
            spectrum = spectrum[:, np.newaxis]
        elif spectrum.ndim == 2:
            spectrum = spectrum.T
        else:
            raise ValueError(
                f"Spectrum file {file_path!r} has {spectrum.ndim} dimensions; "
                "expected one spectrum or a (nspec, nell) stack"
            )
        if spectrum.shape[1] < 1:
            raise ValueError("Spectrum file must have at least one spectrum")
        if spectrum.shape[0] < lmax:
            raise ValueError(
                f"Spectrum file {file_path!r} must extend to at least lmax: it "
                f"ends at ell = {spectrum.shape[0] - 1}, but {lmax} multipoles "
                f"(ell 0..{lmax - 1}) are needed"
            )
        return spectrum[:lmax]
    elif file_path.endswith(".txt") or file_path.endswith(".dat"):
        spectrum = np.loadtxt(file_path)

        # Validate dimensions
        if spectrum.ndim != 2:
            raise ValueError("Spectrum file must have two dimensions (ell, nspec + 1)")

        ells = spectrum[:, 0].astype(int)
        if np.any(np.diff(ells) != 1):
            raise ValueError("Spectrum file must have consecutive ell values")
        if np.max(ells) < lmax - 1:  # Fixed off-by-one error
            raise ValueError(
                f"Spectrum file {file_path!r} must extend to at least lmax: it "
                f"ends at ell = {int(np.max(ells))}, but {lmax} multipoles "
                f"(ell 0..{lmax - 1}) are needed"
            )

        spec_values = spectrum[:, 1:]
        output = np.zeros((lmax, spectrum.shape[1] - 1))
        valid_ells = ells < lmax
        output[ells[valid_ells]] = spec_values[valid_ells]
        return output
    else:
        raise ValueError(
            "Unsupported file format. Supported formats are .txt, .npy, .fits"
        )


def safe_divide(
    numerator: np.ndarray, denominator: np.ndarray, default: float = 1.0
) -> np.ndarray:
    """
    Safely divide arrays, handling division by zero and NaN values.

    Parameters
    ----------
    numerator : np.ndarray
        Numerator array
    denominator : np.ndarray
        Denominator array
    default : float, optional
        Default value for invalid results

    Returns
    -------
    np.ndarray
        Result of division with invalid values replaced by default
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator

    # Replace invalid values
    result = np.where(np.isfinite(result), result, default)
    return result


#: Valid settings for the ``missing`` policy of :func:`_operate_nested_dicts`
#: and its wrappers, see their docstrings.
_MISSING_KEY_POLICIES = ("identity", "error")


def _operate_nested_dicts(
    dict1: dict[str, Any],
    dict2: dict[str, Any],
    operation: str,
    missing: str = "error",
    _path: str = "",
) -> dict[str, Any]:
    """
    Apply an operation element-wise to two nested dictionaries.

    Parameters
    ----------
    dict1 : Dict[str, Any]
        First dictionary
    dict2 : Dict[str, Any]
        Second dictionary
    operation : str
        Operation to apply ('add' or 'mult')
    missing : {"identity", "error"}, optional
        What to do with a key present on only one side. ``"identity"``
        carries the lone value through unchanged (the identity element for
        the operation: 0 for add, 1 for mult -- the previous, only,
        behaviour). ``"error"`` raises :class:`KeyError` naming the key path
        and which side lacks it. Defaults to ``"error"`` so that a call
        nobody thought the missing-key policy through for fails loudly
        instead of silently combining mismatched dictionaries.
    _path : str, optional
        Key path accumulated by recursive calls, for error messages. Not for
        external callers.

    Returns
    -------
    Dict[str, Any]
        Result of the operation

    Raises
    ------
    ValueError
        If ``operation`` or ``missing`` is not supported, or if a leaf pair
        cannot be combined (mismatched shapes/types, or a dict paired with a
        non-dict).
    KeyError
        If ``missing="error"`` and a key is present on only one side.
    """
    if operation not in ("add", "mult"):
        raise ValueError(f"Unsupported operation: {operation}")
    if missing not in _MISSING_KEY_POLICIES:
        raise ValueError(
            f"Unsupported missing policy: {missing!r}; expected one of "
            f"{_MISSING_KEY_POLICIES}"
        )

    result = {}
    all_keys = set(dict1.keys()) | set(dict2.keys())

    for key in all_keys:
        key_path = f"{_path}[{key!r}]"
        in1, in2 = key in dict1, key in dict2

        if in1 and in2:
            val1, val2 = dict1[key], dict2[key]
            is_dict1, is_dict2 = isinstance(val1, dict), isinstance(val2, dict)

            if is_dict1 and is_dict2:
                result[key] = _operate_nested_dicts(
                    val1, val2, operation, missing, key_path
                )
            elif is_dict1 or is_dict2:
                raise ValueError(
                    f"{key_path}: cannot {operation} a dict with a non-dict "
                    f"({type(val1).__name__} and {type(val2).__name__})"
                )
            else:
                try:
                    if operation == "add":
                        result[key] = val1 + val2
                    else:
                        result[key] = val1 * val2
                except (TypeError, ValueError) as exc:
                    shape1 = getattr(val1, "shape", None)
                    shape2 = getattr(val2, "shape", None)
                    if shape1 is not None or shape2 is not None:
                        what = f"arrays of shape {shape1} and {shape2}"
                    else:
                        what = (
                            f"values of type {type(val1).__name__} and "
                            f"{type(val2).__name__}"
                        )
                    raise ValueError(f"{key_path}: cannot {operation} {what}") from exc
        elif in1 or in2:
            if missing == "error":
                lacking = "dict2" if in1 else "dict1"
                raise KeyError(
                    f"{key_path} is missing from {lacking} " "(missing='error' policy)"
                )
            result[key] = dict1[key] if in1 else dict2[key]

    return result


def add_nested_dicts(
    dict1: dict[str, Any], dict2: dict[str, Any], missing: str = "error"
) -> dict[str, Any]:
    """
    Add two nested dictionaries element-wise.

    Recursively adds corresponding values from two nested dictionaries.
    If a value is a dictionary, recurse; otherwise, add numerically.

    Parameters
    ----------
    dict1 : Dict[str, Any]
        First dictionary to add
    dict2 : Dict[str, Any]
        Second dictionary to add
    missing : {"identity", "error"}, optional
        Policy for a key present on only one side: ``"identity"`` carries it
        through unchanged (0 is the identity element for addition -- the
        previous, only, behaviour); ``"error"`` (the default) raises
        :class:`KeyError` naming the key path and which side lacks it, so
        that a caller which has not thought about the missing-key case fails
        loudly rather than silently combining mismatched dictionaries.

    Returns
    -------
    Dict[str, Any]
        Sum of the two dictionaries

    Raises
    ------
    KeyError
        If ``missing="error"`` and a key is present on only one side.
    ValueError
        If a leaf pair cannot be added (mismatched shapes/types, or a dict
        paired with a non-dict).

    Examples
    --------
    >>> d1 = {'a': {'x': 1, 'y': 2}, 'b': 3}
    >>> d2 = {'a': {'x': 10, 'y': 20}, 'b': 30}
    >>> add_nested_dicts(d1, d2)
    {'a': {'x': 11, 'y': 22}, 'b': 33}
    """
    return _operate_nested_dicts(dict1, dict2, "add", missing=missing)


def mult_nested_dicts(
    dict1: dict[str, Any], dict2: dict[str, Any], missing: str = "error"
) -> dict[str, Any]:
    """
    Multiply two nested dictionaries element-wise.

    Recursively multiplies corresponding values from two nested dictionaries.
    If a value is a dictionary, recurse; otherwise, multiply numerically.

    Parameters
    ----------
    dict1 : Dict[str, Any]
        First dictionary to multiply
    dict2 : Dict[str, Any]
        Second dictionary to multiply
    missing : {"identity", "error"}, optional
        Policy for a key present on only one side: ``"identity"`` carries it
        through unchanged (1 is the identity element for multiplication --
        the previous, only, behaviour); ``"error"`` (the default) raises
        :class:`KeyError` naming the key path and which side lacks it, so
        that a caller which has not thought about the missing-key case fails
        loudly rather than silently combining mismatched dictionaries.

    Returns
    -------
    Dict[str, Any]
        Product of the two dictionaries

    Raises
    ------
    KeyError
        If ``missing="error"`` and a key is present on only one side.
    ValueError
        If a leaf pair cannot be multiplied (mismatched shapes/types, or a
        dict paired with a non-dict).

    Examples
    --------
    >>> d1 = {'a': {'x': 1, 'y': 2}, 'b': 3}
    >>> d2 = {'a': {'x': 10, 'y': 20}, 'b': 30}
    >>> mult_nested_dicts(d1, d2)
    {'a': {'x': 10, 'y': 40}, 'b': 90}
    """
    return _operate_nested_dicts(dict1, dict2, "mult", missing=missing)
