"""Shared definitions of the exact covariance: fields, spectra and the mask coupling width."""

from __future__ import annotations

import sys
import warnings
from collections.abc import Mapping, Sequence

import numpy as np

__all__ = [
    "mask_coupling_width",
    "SPECTRA",
]


#: Fields and the canonical spectrum names of the polarised functions.
FIELDS = ("T", "E", "B")
SPECTRA = ("TT", "EE", "BB", "TE", "TB", "EB")


# --------------------------------------------------------------------------- #
# Expected relative cost of a row (the weights of the progress lines)
# --------------------------------------------------------------------------- #
def row_weights(rows: Sequence[int], use_symmetry: bool = True) -> list[float]:
    """
    Expected relative cost of each row ``l'`` of ``rows`` (arbitrary units):
    the number of orders ``m'`` it sums over, ``l' + 1`` with the symmetry
    and ``2 l' + 1`` without. A row computes one set of correlation columns
    per order (module docstring of :mod:`cmbcov.exact.gl`), each at a cost
    that does not depend on ``l'``, so the cost is linear in ``l'`` (not the
    ``O(l'^4)`` of the per-column reference path); the polarised rows have
    the same structure. Nothing is fitted. A row also has a fixed cost
    independent of ``l'``, not modelled: the estimate runs high early in a
    stage (by about 15% at the 20% line of a one-minute run, 30% in a
    three-second one: the fixed cost is then a larger share) and settles.
    With ``term_selection`` a row drops some of its orders, not known in
    advance; the estimate likewise runs high at first.
    """
    return [
        float(ellp + 1 if use_symmetry else 2 * ellp + 1) for ellp in map(int, rows)
    ]


# --------------------------------------------------------------------------- #
# How much C_L headroom a mask needs: the coupling width
# --------------------------------------------------------------------------- #
#: Fraction of a mask's coupling weight that the margin ``lmax_int - lmax``
#: should cover before the truncation of the ``C_L`` sum is considered safe.
COUPLING_COVERAGE = 0.99


def mask_coupling_width(wl: np.ndarray, coverage: float = COUPLING_COVERAGE) -> int:
    r"""
    Half-width in multipole of a mask's harmonic coupling, from ``W_L`` alone.

    A mask spreads a multipole over its own harmonic content: in the MASTER
    kernel
    :math:`M_{\ell\ell'} = \frac{2\ell'+1}{4\pi} \sum_L (2L+1) W_L
    \begin{pmatrix}\ell & \ell' & L\\0&0&0\end{pmatrix}^2`
    the mask multipole :math:`L` enters with weight :math:`(2L+1) W_L`, and the
    3j symbol vanishes unless :math:`|\ell - \ell'| \le L`.  The weight carried
    by mask multipoles *larger* than some :math:`\Delta` is therefore what a
    band-limit :math:`\ell' + \Delta` throws away, and requiring that fraction
    to be small is what fixes the margin.  This function returns that
    :math:`\Delta`:

    .. math::

        \Delta(p) = \min \Big\{ \Delta :
            \sum_{L \le \Delta} (2L+1) W_L \ \ge\ p \sum_L (2L+1) W_L \Big\}

    with ``p = coverage`` (default 0.99).

    Why this and not :func:`~.mask.mask_spectral_moment`.  The obvious scalar
    from ``W_L`` is :math:`\sqrt{\langle L(L+1)\rangle_W}`, but it is an rms:
    for a smooth apodised mask ``W_L`` is compact and the needed margin is a
    few times the rms, while for a hard-edged mask the rms is dominated by a
    power-law tail that carries very little *total* weight and overstates the
    bulk.  For cosine-apodised caps of very different radii and apodisations,
    this estimator tracks the margin at which the exact diagonal reaches 1e-3
    relative accuracy to within roughly 30%, whereas the rms is off by factors
    of 2-6 and does not even order the masks correctly.  It is still an
    *estimate*: the residual bias at exactly this margin is of order 1e-3 of
    the diagonal, and a tail-heavy mask needs a few times more for 1e-6.

    Parameters
    ----------
    wl : ndarray
        Mask power spectrum ``W_L`` for ``L = 0 .. wl.size - 1``.  For a
        band-limited mask (``grid="gl"``) the result cannot exceed ``Lw``,
        which is the exact bound on the coupling.
    coverage : float
        Fraction of the coupling weight the width must contain, in ``(0, 1)``.

    Returns
    -------
    int
        ``0`` for a full-sky mask (all the weight in the monopole) and for an
        identically zero spectrum.
    """
    wl = np.asarray(wl, dtype=float)
    if wl.ndim != 1:
        raise ValueError(f"wl must be 1-D, got shape {wl.shape}")
    if not 0.0 < coverage < 1.0:
        raise ValueError(f"coverage must lie in (0, 1), got {coverage}")
    if wl.size == 0:
        return 0
    ell = np.arange(wl.size)
    # W_L is a power spectrum, so the clip only removes rounding-level negatives.
    weight = np.clip((2.0 * ell + 1.0) * wl, 0.0, None)
    total = float(weight.sum())
    if total == 0.0:
        return 0
    cum = np.cumsum(weight) / total
    return int(np.searchsorted(cum, coverage))


#: ``(entry point, lmax, lmax_int, width)`` already warned about by
#: :func:`_warn_short_margin` in this process: a row loop over
#: ``exact_covariance_row`` would otherwise repeat it once per row.
_WARNED_SHORT_MARGINS: set = set()


def _warn_short_margin(
    wl: np.ndarray, lmax: int, lmax_int: int, stacklevel: int = 3
) -> int:
    """
    Warn if ``lmax_int - lmax`` is below the mask's coupling width.

    Returns the width so that callers can reuse it; emits a ``UserWarning``
    naming the margin supplied and the width estimated, once per entry
    point, ``lmax``, ``lmax_int`` and width per process.
    """
    width = mask_coupling_width(wl)
    margin = int(lmax_int) - int(lmax)
    if margin >= width:
        return width
    caller = sys._getframe(1).f_code.co_name
    key = (caller, int(lmax), int(lmax_int), int(width))
    if key in _WARNED_SHORT_MARGINS:
        return width
    _WARNED_SHORT_MARGINS.add(key)
    warnings.warn(
        f"internal band-limit margin {margin} (lmax={lmax}, internal band-limit "
        f"{lmax_int}) is below this mask's coupling width {width} "
        f"({100 * COUPLING_COVERAGE:.0f}% of sum_L (2L+1) W_L lies at L <= "
        f"{width}): C_L above the internal band-limit is dropped from the sum "
        "of Eq. (10), which biases the covariance near lmax. Pass "
        f"lmax_int >= {int(lmax) + width} with C_L defined that far. "
        f"(Warned once per process for {caller} at these values.)",
        UserWarning,
        stacklevel=stacklevel,
    )
    return width


# --------------------------------------------------------------------------- #
# Polarisation on the GL grid
# --------------------------------------------------------------------------- #
def _canonical_spec(spec: str) -> str:
    """``"ET" -> "TE"`` etc.; raises on anything outside :data:`SPECTRA`."""
    s = str(spec).upper()
    if s in SPECTRA:
        return s
    if s[::-1] in SPECTRA:
        return s[::-1]
    raise ValueError(f"unknown spectrum {spec!r}; expected one of {SPECTRA}")


def _cl_matrix(cls: Mapping[str, np.ndarray], lmax: int) -> np.ndarray:
    """
    ``C[X, Z, L]`` (shape ``(3, 3, lmax+1)``), the symmetric 3x3 spectrum
    matrix per multipole from a dict with keys among :data:`SPECTRA`
    (missing = zero).  E and B have no ``L < 2`` modes; those entries are
    zeroed.
    """
    cmat = np.zeros((3, 3, lmax + 1))
    for key, cl in cls.items():
        spec = _canonical_spec(key)
        cl = np.asarray(cl, dtype=float)
        if cl.ndim != 1 or cl.size < lmax + 1:
            raise ValueError(f"cls[{key!r}] must have at least lmax+1 entries")
        i, j = FIELDS.index(spec[0]), FIELDS.index(spec[1])
        cmat[i, j] = cl[: lmax + 1]
        cmat[j, i] = cl[: lmax + 1]
    cmat[1:, :, :2] = 0.0
    cmat[:, 1:, :2] = 0.0
    return cmat


def _spectra_fields(spectra: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Canonical spectrum names and the fields they involve, in FIELDS order."""
    specs = tuple(dict.fromkeys(_canonical_spec(s) for s in spectra))
    if not specs:
        raise ValueError("spectra must not be empty")
    used = {c for s in specs for c in s}
    fields = tuple(f for f in FIELDS if f in used)
    return specs, fields
