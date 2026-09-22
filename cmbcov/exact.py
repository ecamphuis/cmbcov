r"""
Exact pseudo-:math:`C_\ell` covariance (Camphuis et al. 2022, Sect. 3).

This module implements the *exact* algorithm of Camphuis et al. (2022,
arXiv:2204.13721, Sect. 3.1 and Fig. 2) for a single real mask :math:`W` and a
real temperature spectrum :math:`C_L`, without any narrow-kernel approximation.

Notation
--------
Let ``S`` be the synthesis operator (alm -> map), ``A`` the analysis operator
used by the estimator (``hp.map2alm(..., iter=iter)``) and ``W`` the
point-wise multiplication by the mask.  The pseudo-alm operator is
``K = A W S`` and the pseudo-alm correlation matrix is

.. math::

    \langle \tilde a \tilde a^\dagger \rangle = K \, \mathrm{diag}(C_L) \, K^\dagger .

The covariance follows from paper Eq. (7),

.. math::

    \tilde\Sigma_{\ell\ell'} = \frac{2}{(2\ell+1)(2\ell'+1)}
        \sum_{m m'} \left| \langle \tilde a_{\ell m} \tilde a^*_{\ell' m'} \rangle \right|^2 .

One column :math:`(\ell', m')` of the correlation matrix is obtained by applying
the operator to the basis vector :math:`e_{\ell' m'}`, right to left
(paper Fig. 2):

1. :math:`I_{LM} = (K^\dagger e_{\ell' m'})_{LM}` -- synthesise the single-mode
   (complex) map :math:`Y_{\ell' m'}`, multiply by ``W``, analyse.
2. :math:`x_{LM} = C_L I_{LM}` (paper Eq. 10).
3. Synthesise :math:`x` into the complex map :math:`X`, multiply by ``W``,
   analyse: :math:`c_{\ell m} = \langle \tilde a_{\ell m} \tilde a^*_{\ell' m'}\rangle`
   for all :math:`\ell, m` (paper Eq. 11).

Two numerical points are handled explicitly here:

* Complex maps.  HEALPix alm arrays assume a real map
  (:math:`a_{L,-M} = (-1)^M a^*_{LM}`).  The maps :math:`Y_{\ell' m'}` and
  :math:`X` are complex, so their real and imaginary parts are transformed
  separately and recombined into full-``M`` coefficients
  (:func:`_full_from_pair` / :func:`_pair_from_full`).
* Adjoint of the iterated analysis.  ``hp.map2alm(iter=n)`` is
  :math:`A = \sum_{k=0}^{n} (1 - S^\dagger S)^k S^\dagger`, which is *not*
  self-adjoint against ``S``.  Step 1 therefore applies
  :math:`A^\dagger = S \sum_k (1 - S^\dagger S)^k` exactly
  (:func:`_adjoint_map2alm`), so that the computed matrix is
  :math:`K C K^\dagger` for the same ``K`` that ``hp.anafast`` uses.  The naive
  variant :math:`K C K` (analyse ``W Y`` with the iterated analysis) is kept
  behind ``exact_adjoint=False`` for comparison.

Complexity: each :math:`(\ell', m')` column costs a fixed number of spherical
harmonic transforms, i.e. :math:`O(\ell'^3)` when ``nside ~ lmax``; a row needs
:math:`2\ell'+1` columns (or :math:`\ell'+1` using the :math:`m' \to -m'`
symmetry), hence :math:`O(\ell'^4)` per row and :math:`O(\ell^5)` for the full
matrix.

Cost of the GL column
---------------------
Since the complex map of a column is only ever a pair of real maps, the GL path
never materialises it (nor the ``(lmax+1, 2 lmax+1)`` full-``M`` arrays): a
column is carried as the two healpy alm sets ``(ra, sa)`` of the real and
imaginary parts (:func:`_correlation_column_pair_gl`).  That leaves six real
transforms per column with ``m' != 0`` and three with ``m' = 0``, plus one
single-order synthesis per part for step 1 (the map of a *single* harmonic has
one azimuthal order, so all but one term of the Legendre sum is zero --
:func:`~.grid.gl_synthesis_single_m`, bit-identical to the full synthesis).
Measured at ``lmax=500``, ``Lw=384`` (grid 692) on 7 threads this is 21 ms per
column against 59 ms for the literal complex-map version, and a row at
``l' = 500`` costs 10.6 s instead of 29.6 s.  The literal version is kept as
``_exact_row_gl(..., reference=True)`` and the two are pinned against each
other in ``tests/test_exact_gl_speed.py``.

Rows are independent, so :func:`exact_covariance` can spread them over
processes (``nprocs``, default 1).  On a 14-core laptop that is *not* the
better lever -- ducc0's own threads rebalance within a transform and share one
copy of the 7.7 MB maps, and beat 14 single-threaded processes by 1.56x.

Grids: ``grid="healpix"`` versus ``grid="gl"``
----------------------------------------------
The public functions take a ``grid`` switch selecting *which estimator's*
covariance is computed; the two answer different questions and neither is
wrong.

``grid="healpix"`` (default)
    ``S`` and ``A`` are ``hp.alm2map`` / ``hp.map2alm(iter=iter)`` on the
    HEALPix pixelisation of the mask.  The result is the covariance of the
    pixelised ``map2alm(iter)`` pseudo-spectrum estimator -- what
    ``hp.anafast`` on a HEALPix map times a pixel mask actually does.  HEALPix
    has no exact quadrature, so this path needs the adjoint correction above
    and is itself only accurate to the level at which the iterated ``map2alm``
    approximates the integral (~4e-7 on the full-sky test).

``grid="gl"``
    The mask enters as a band-limited alm set (band-limit ``Lw``), synthesised
    on a Gauss-Legendre grid (:mod:`cmbcov.grid`) large
    enough that every analysis in the algorithm is an exact integral.  The
    result is the covariance of the *exact-integral* pseudo-spectrum estimator
    of the band-limited mask, to machine precision (2e-15 on the full sky,
    symmetric to 3e-16 with no adjoint machinery).  The only approximation is
    the mask truncation ``Lw`` and the ``map2alm`` that produced its alm.

Measured on the baseline mask at ``nside=32``, ``Lw=48``, ``lmax=32``, the two
differ by ~1e-5 on the diagonal band and up to ~1e-3 far off the diagonal
(where ``Sigma`` is 1e-4 of the diagonal); the difference shrinks as
``nside^-2`` as the HEALPix quadrature improves, and does *not* shrink with
more ``map2alm`` iterations.

Reported band-limit versus internal band-limit
----------------------------------------------
:math:`\tilde\Sigma_{\ell\ell'}` sums :math:`C_L` over the mask's coupling
range *around* :math:`\ell'`, so the algorithm needs :math:`C_L` **above** the
highest multipole it reports.  ``lmax`` alone cannot express that: truncating
the sum at ``lmax`` sets :math:`C_L = 0` over the upper half of the kernel and
biases every row near the top of the matrix, silently and severely.  Measured
on a 4.4% survey mask (nside 512, mask band-limit ``Lw = 512``), row 500
against an internal band-limit of 700: margin 200 -> 1.000000, margin 120 ->
0.999574, margin 0 -> 0.316251 on the diagonal, i.e. a factor 3.2 with no
diagnostic.

``lmax_int`` (default ``None`` = ``lmax``, i.e. the historical behaviour)
therefore separates the two: every harmonic array, the :math:`C_L` sum and the
GL grid are carried to ``lmax_int``, and only rows/columns up to ``lmax`` are
returned.  That is exactly the same arithmetic as computing the whole matrix
at ``lmax_int`` and slicing it, at a fraction of the cost.  The margin
``lmax_int - lmax`` is compared against :func:`mask_coupling_width`, an
estimate of the mask's own coupling half-width built from ``W_L``, and a
``UserWarning`` names both numbers when the margin is too small.

Checked against simulations: 20000 ``healpy`` skies band-limited at 36 and
masked by an apodised cap at nside 32, pseudo-spectra to ``lmax = 12``.  The
diagonal of the HEALPix-path covariance run with ``lmax_int = 36`` matches the
sample covariance to 0.4-1.5% at every multipole (the Monte Carlo's own
1-sigma is 1.0% per element, and the residual moves with the seed), while the
same call with ``lmax_int = None`` falls to 0.49 of it at ``l = 12`` and 0.82
at ``l = 10``.

Polarisation (GL grid only)
---------------------------
:func:`exact_covariance_row_pol` / :func:`exact_covariance_pol` extend the
algorithm to the fields ``X in {T, E, B}`` and the six spectra TT, EE, BB, TE,
TB, EB, on the GL grid only (the HEALPix path stays TT-only).  With
``K = blockdiag(K_0, K_2)``, ``K_0 = A_0 W S_0`` on T and ``K_2 = A_2 W S_2``
on (E, B), and ``C`` the ``(L, M)``-diagonal 3x3 matrix
``[[TT, TE, TB], [TE, EE, EB], [TB, EB, BB]]``, the pseudo-alm correlators are

    R^{XZ}_{lm, l'm'} = <a~^X_lm conj(a~^Z_l'm')> = [K C K^dagger]^{XZ}_{lm, l'm'}

and on GL ``K^dagger = K`` (analysis is the adjoint of synthesis).  One column
``(Z, l'm')`` of ``R`` for every ``(X, lm)`` costs: ``u = K e_{Z, l'm'}`` (a
spin-0 round trip for Z = T, spin-2 for Z in {E, B}; the unit vector at
``(l', +m')`` has no reality partner, hence complex maps), then
``v^X_LM = sum_Z' C^{XZ'}_L u^{Z'}_LM``, then ``R^{.Z} = K v`` (spin-0 on
``v^T``, spin-2 on ``(v^E, v^B)``).  Wick's theorem with the reality condition
``a~_{l,-m} = (-1)^m conj(a~_lm)`` folded onto ``m' -> -m'`` gives

    Cov(C~^{XY}_l, C~^{ZW}_l') = 1 / ((2l+1)(2l'+1)) sum_{m m'}
        [ R^{XW} conj(R^{YZ}) + R^{XZ} conj(R^{YW}) ]_{lm, l'm'}

which reduces to ``2 sum |R^{TT}|^2`` for XY = ZW = TT.  The ``m' < 0``
column contributes the complex conjugate of the ``m' > 0`` one, so with
``use_symmetry`` the sum is ``(m'=0 term) + 2 Re sum_{m'>0}``.  The three
columns Z = T, E, B for one ``(l', m')`` give every block among the six
spectra at once, so the unit of work is "all requested blocks of one row";
only the columns the requested spectra need are computed (T only for TT;
T and E for TT/TE/EE; ...).  ducc0's spin-2 convention is HEALPix's
(gradient, curl) = (E, B), so signs match ``healpy.anafast``.
"""

from __future__ import annotations

import hashlib
import os
import sys
import warnings
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .grid import (
    full_from_pair,
    gl_analysis,
    gl_analysis_complex,
    gl_minimal_lmax,
    gl_synthesis,
    gl_synthesis_complex,
    gl_synthesis_single_m,
    pair_from_full,
)
from .sht import alm2cl, ducc0_alm2map, ducc0_map2alm
from .term_selection import mode_power, pole_rotation, rotate_alm

__all__ = [
    "exact_covariance_row",
    "exact_covariance",
    "exact_covariance_row_pol",
    "exact_covariance_pol",
    "mask_coupling_width",
    "SPECTRA",
]

#: Fields and the canonical spectrum names of the polarised functions.
FIELDS = ("T", "E", "B")
SPECTRA = ("TT", "EE", "BB", "TE", "TB", "EB")

# Full-M <-> real-map-pair alm conversions live in :mod:`.grid` (same layout:
# shape (lmax+1, 2*lmax+1), column index M+lmax); kept under their historical
# private names here.
_full_from_pair = full_from_pair
_pair_from_full = pair_from_full


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
    from ``W_L`` is :math:`\sqrt{\langle L(L+1)\rangle_W}` (that function), but
    it is an rms: for a smooth apodised mask ``W_L`` is compact and the needed
    margin is a few times the rms, while for a hard-edged mask the rms is
    dominated by a power-law tail that carries very little *total* weight and
    it overstates the bulk.  Measured against the truncation error of the exact
    diagonal for cosine-apodised caps (40 deg radius with 40, 10, 2 and 0 deg
    apodisation, and a 10 deg radius cap with 5 deg apodisation; ``Lw`` 48 and
    96, ``l' = 30`` and ``60``), the margin at which the diagonal reaches 1e-3
    relative accuracy is

        mask                     sqrt(<L(L+1)>_W)   this estimator   measured
        cap 40 deg, apod 40 deg          3.8              7            8
        cap 40 deg, apod 10 deg          4.9             19           22-24
        cap 40 deg, apod  2 deg          8.6             40           36
        cap 40 deg, hard edge            9.4             48           40-54
        cap 10 deg, apod  5 deg         15.8             46           40-46

    i.e. this estimator is within a factor 0.8-1.3 of the measured requirement
    across a factor 12 in mask width.  The rms is off by 2.1x to 5.7x over the
    same set and does not even order the masks correctly (the 10 deg cap has
    the largest rms and not the largest requirement).  This is still an
    *estimate*: the residual bias at exactly this margin is ~1e-3 of the
    diagonal (1.4e-4 on the nside=16 cap of ``tests/test_exact_margin.py``),
    and a tail-heavy mask needs a few times more for 1e-6.

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
        "of Eq. (10), which biases the covariance near lmax (on a 4.4% survey "
        "mask a zero margin cost a factor 3.2 on the diagonal). Pass "
        f"lmax_int >= {int(lmax) + width} with C_L defined that far. "
        f"(Warned once per process for {caller} at these values.)",
        UserWarning,
        stacklevel=stacklevel,
    )
    return width


#: Content-keyed memo of ``W_L`` for HEALPix pixel masks: the margin check is
#: per row, the spectrum is per mask, and a full matrix would otherwise pay one
#: ``map2alm`` at ``3 nside - 1`` for every row of it.
#: Keys are ``(nside, blake2b digest of the mask)``, values ``W_L``.
_MASK_WL_CACHE: OrderedDict = OrderedDict()
_MASK_WL_CACHE_SIZE = 4


def _mask_wl_healpix(mask: np.ndarray, nside: int) -> np.ndarray:
    """``W_L`` of a HEALPix pixel mask up to ``3 nside - 1`` (diagnostic only)."""
    mask = np.ascontiguousarray(mask, dtype=float)
    key = (int(nside), hashlib.blake2b(mask, digest_size=16).digest())
    wl = _MASK_WL_CACHE.get(key)
    if wl is None:
        # iter=0: this feeds a 1% quantile of the spectrum, not the covariance.
        wl = alm2cl(ducc0_map2alm(mask, lmax=3 * nside - 1, iter=0))
        _MASK_WL_CACHE[key] = wl
        while len(_MASK_WL_CACHE) > _MASK_WL_CACHE_SIZE:
            _MASK_WL_CACHE.popitem(last=False)
    return wl


# --------------------------------------------------------------------------- #
# Real-map transforms and the adjoint of the iterated analysis
# --------------------------------------------------------------------------- #
def _synth(alm: np.ndarray, nside: int, lmax: int) -> np.ndarray:
    return ducc0_alm2map(alm, nside, lmax=lmax)


def _analyse0(m: np.ndarray, lmax: int) -> np.ndarray:
    """Plain adjoint synthesis ``S^dagger`` (map2alm with iter=0)."""
    return ducc0_map2alm(m, lmax=lmax, pol=False, iter=0)


def _analyse(m: np.ndarray, lmax: int, iter: int) -> np.ndarray:
    """The estimator's analysis ``A`` (map2alm with ``iter`` iterations)."""
    return ducc0_map2alm(m, lmax=lmax, pol=False, iter=iter)


def _adjoint_map2alm(alm: np.ndarray, nside: int, lmax: int, iter: int) -> np.ndarray:
    r"""
    Apply ``A^dagger`` to a real-map alm set, where ``A = hp.map2alm(iter=iter)``.

    healpy iterates ``a <- a + S^dagger (f - S a)`` starting from ``S^dagger f``,
    i.e. ``A = sum_{k=0}^{iter} (1 - S^dagger S)^k S^dagger``.  Hence
    ``A^dagger = S sum_k (1 - S^dagger S)^k`` and the result is a map.
    """
    v = alm.copy()
    acc = alm.copy()
    for _ in range(iter):
        v = v - _analyse0(_synth(v, nside, lmax), lmax)
        acc += v
    return _synth(acc, nside, lmax)


# --------------------------------------------------------------------------- #
# Complex-map transforms
# --------------------------------------------------------------------------- #
def _complex_synth(full: np.ndarray, nside: int, lmax: int) -> np.ndarray:
    r, s = _pair_from_full(full, lmax)
    out = _synth(r, nside, lmax).astype(complex)
    if np.any(s):
        out += 1j * _synth(s, nside, lmax)
    return out


def _complex_analyse(m: np.ndarray, lmax: int, iter: int) -> np.ndarray:
    r = _analyse(m.real, lmax, iter)
    if np.any(m.imag):
        s = _analyse(m.imag, lmax, iter)
    else:
        s = np.zeros_like(r)
    return _full_from_pair(r, s, lmax)


def _complex_analyse0(m: np.ndarray, lmax: int) -> np.ndarray:
    r = _analyse0(m.real, lmax)
    if np.any(m.imag):
        s = _analyse0(m.imag, lmax)
    else:
        s = np.zeros_like(r)
    return _full_from_pair(r, s, lmax)


def _complex_adjoint_map2alm(
    full: np.ndarray, nside: int, lmax: int, iter: int
) -> np.ndarray:
    r, s = _pair_from_full(full, lmax)
    out = _adjoint_map2alm(r, nside, lmax, iter).astype(complex)
    if np.any(s):
        out += 1j * _adjoint_map2alm(s, nside, lmax, iter)
    return out


# --------------------------------------------------------------------------- #
# One column of <a~ a~^dagger>
# --------------------------------------------------------------------------- #
def _correlation_column(
    mask: np.ndarray,
    cl: np.ndarray,
    ellp: int,
    mp: int,
    lmax: int,
    nside: int,
    iter: int,
    exact_adjoint: bool,
) -> np.ndarray:
    r"""
    ``c[l, m] = <a~_{lm} a~*_{l'm'}>`` for all ``(l, m)``, as a full-``M`` array
    of shape ``(lmax+1, 2*lmax+1)``  (paper Eqs. 8-11, Fig. 2).
    """
    # Basis vector e_{l'm'} in full-M form.
    e = np.zeros((lmax + 1, 2 * lmax + 1), dtype=complex)
    e[ellp, lmax + mp] = 1.0

    # Step 1: I = K^dagger e = S^dagger W (A^dagger e)   [or A W S e if naive]
    if exact_adjoint:
        y = _complex_adjoint_map2alm(e, nside, lmax, iter)
        i_lm = _complex_analyse0(mask * y, lmax)
    else:
        y = _complex_synth(e, nside, lmax)
        i_lm = _complex_analyse(mask * y, lmax, iter)

    # Step 2: x_LM = C_L I_LM   (Eq. 10)
    x = cl[:, None] * i_lm

    # Step 3: c = K x = A W S x   (Eq. 11)
    x_map = _complex_synth(x, nside, lmax)
    return _complex_analyse(mask * x_map, lmax, iter)


# --------------------------------------------------------------------------- #
# Gauss-Legendre path
# --------------------------------------------------------------------------- #
def _mask_alm_for_gl(
    mask: np.ndarray, nside: int | None, lw: int | None
) -> tuple[np.ndarray, int]:
    """
    Band-limited mask alm ``(mask_alm, lw)`` for the GL path.

    ``mask`` is either a complex healpy alm array (used as is; ``lw`` is
    inferred from its size unless given) or a real HEALPix map, converted once
    with ``hp.map2alm(mask, lmax=lw, iter=10)`` where ``lw`` defaults to
    ``3 * nside - 1``.
    """
    import healpy as hp  # lazy, see utils/healpy_utils.py

    mask = np.asarray(mask)
    if np.iscomplexobj(mask):
        lw_from_size = hp.Alm.getlmax(mask.size)
        if lw_from_size < 0:
            raise ValueError("mask alm size is not a valid healpy alm size")
        if lw is None:
            lw = lw_from_size
        elif lw != lw_from_size:
            raise ValueError(
                f"mask alm has band-limit {lw_from_size} but lw={lw} was given"
            )
        return np.ascontiguousarray(mask, dtype=np.complex128), int(lw)

    mask = np.asarray(mask, dtype=float)
    if nside is None:
        nside = hp.npix2nside(mask.size)
    elif mask.size != hp.nside2npix(nside):
        raise ValueError("mask size does not match nside")
    if lw is None:
        lw = 3 * nside - 1
    return ducc0_map2alm(mask, lmax=lw, iter=10), int(lw)


def _pole_frame_alm(mask, mask_alm: np.ndarray, lw: int) -> np.ndarray:
    r"""
    The mask alm rotated so that the mask's centroid sits at ``+z``
    (:func:`~cmbcov.term_selection.pole_rotation`).

    The exact covariance is rotation invariant: :math:`\tilde\Sigma_{\ell\ell'}
    = \frac{2}{(2\ell+1)(2\ell'+1)} \sum_{m m'} |\langle \tilde a_{\ell m}
    \tilde a^*_{\ell' m'}\rangle|^2` sums over *all* ``(m, m')`` and a
    rotation acts on each degree by a unitary Wigner-``D`` matrix in ``m``,
    so the row depends on the mask only up to rotation.  Which ``m'`` carry
    it does depend on the frame, and the term selection of
    :func:`_kept_mprime` is only sharp in the pole frame.  ``mask`` is the
    caller's input: a pixel map is used for the moments directly; an alm
    input is synthesised at the smallest ``nside`` with ``3 nside - 1 >= lw``
    (at least 16) for that purpose only.
    """
    import healpy as hp  # lazy, see utils/healpy_utils.py

    mask = np.asarray(mask)
    if np.iscomplexobj(mask):
        nside = 16
        while 3 * nside - 1 < lw:
            nside *= 2
        mask_map = ducc0_alm2map(mask_alm, nside, lmax=lw)
    else:
        mask_map = np.asarray(mask, dtype=float)
        hp.npix2nside(mask_map.size)  # validates a HEALPix map
    return rotate_alm(mask_alm, lw, pole_rotation(mask_map))


def _kept_mprime(
    mask_alm: np.ndarray,
    lw: int,
    ellp: int,
    tolerance: float,
    spins: Sequence[int] = (0,),
    nthreads: int | None = None,
) -> np.ndarray:
    r"""
    Which orders ``m'`` of the row ``l'`` to compute under term selection:
    a bool array of length ``2 l' + 1`` (index ``m' + l'``), True where the
    mask overlap :math:`p_{m'} = \int W^2 |Y_{\ell' m'}|^2`
    (:func:`~cmbcov.term_selection.mode_power`) satisfies
    :math:`p_{m'} \ge \epsilon_m \max_{m'} p_{m'}` with ``eps_m =
    tolerance / 10`` -- the same rule as
    :func:`~cmbcov.term_selection.select_terms`.  Several
    ``spins`` (``(0, 2)`` for a polarised row) are united, each against its
    own maximum; a spin whose power vanishes identically (spin 2 at
    ``l' < 2``) is ignored.  The ``(m, m')`` pair rule of ``select_terms``
    does not apply here: each column's SHT returns every ``(l, m)`` at once.
    """
    if tolerance <= 0:
        raise ValueError("term_selection must be a positive tolerance")
    keep = np.zeros(2 * ellp + 1, dtype=bool)
    for spin in spins:
        p = mode_power(mask_alm, lw, ellp, spin=spin, nthreads=nthreads)
        if p.max() > 0:
            keep |= p >= (tolerance / 10) * p.max()
    if not keep.any():  # cannot happen (the maximum always passes); be safe
        keep[:] = True
    return keep


def _correlation_column_gl(
    mask_map: np.ndarray,
    cl: np.ndarray,
    ellp: int,
    mp: int,
    lmax: int,
    lmax_grid: int,
    nthreads: int | None,
) -> np.ndarray:
    r"""
    GL version of :func:`_correlation_column`: ``c[l, m] = <a~_{lm} a~*_{l'm'}>``
    with ``S``, ``A`` the exact GL synthesis/analysis and ``mask_map`` the
    band-limited mask on the same grid.  No adjoint correction is needed:
    on GL, analysis *is* the adjoint of synthesis.
    """
    e = np.zeros((lmax + 1, 2 * lmax + 1), dtype=complex)
    e[ellp, lmax + mp] = 1.0

    # Step 1: I = K^dagger e = S^dagger W S e
    y = gl_synthesis_complex(e, lmax, lmax_grid, nthreads=nthreads)
    i_lm = gl_analysis_complex(mask_map * y, lmax, lmax_grid, nthreads=nthreads)

    # Step 2: x_LM = C_L I_LM   (Eq. 10)
    x = cl[:, None] * i_lm

    # Step 3: c = K x = S^dagger W S x   (Eq. 11)
    x_map = gl_synthesis_complex(x, lmax, lmax_grid, nthreads=nthreads)
    return gl_analysis_complex(mask_map * x_map, lmax, lmax_grid, nthreads=nthreads)


def _unit_pair_alm(
    lmax: int, ellp: int, mp: int
) -> tuple[np.ndarray, np.ndarray, bool]:
    r"""
    The real-map pair of the full-``M`` unit vector at ``(ellp, mp)``.

    ``pair_from_full`` of an array with a single 1 at ``(l', m')`` has one
    nonzero entry each, at healpy index ``(l', |m'|)``:

        m' = 0:  r = 1,                s = 0
        m' > 0:  r = 1/2,              s = -i/2
        m' < 0:  r = (-1)^{m'} / 2,    s = i (-1)^{m'} / 2

    Returns ``(r, s, has_s)`` with ``has_s`` False only for ``m' = 0``, where
    the map is real and the second transform of every step is skipped (exactly
    as ``np.any(s)`` does in :func:`~.grid.gl_synthesis_complex`).
    """
    import healpy as hp  # lazy, see utils/healpy_utils.py

    nalm = hp.Alm.getsize(lmax)
    r = np.zeros(nalm, dtype=complex)
    s = np.zeros(nalm, dtype=complex)
    idx = hp.Alm.getidx(lmax, ellp, abs(mp))
    if mp == 0:
        r[idx] = 1.0
        return r, s, False
    sign = 1.0 if mp % 2 == 0 else -1.0
    if mp > 0:
        r[idx] = 0.5
        s[idx] = -0.5j
    else:
        r[idx] = 0.5 * sign
        s[idx] = 0.5j * sign
    return r, s, True


def _correlation_column_pair_gl(
    mask_map: np.ndarray,
    cl_alm: np.ndarray,
    ellp: int,
    mp: int,
    lmax: int,
    lmax_grid: int,
    nthreads: int | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    r"""
    :func:`_correlation_column_gl` in the real-map-pair representation.

    Returns the healpy alms ``(ra, sa)`` of the real and imaginary parts of the
    complex map behind the column, i.e. exactly the pair that
    :func:`~.grid.full_from_pair` would turn into the full-``M`` column ``c``;
    ``sa`` is ``None`` when it is identically zero (``m' = 0``).

    This is the same arithmetic as :func:`_correlation_column_gl` with three
    allocations removed (module docstring, "Cost"):

    * the complex maps.  ``W * (f + i g)`` with real ``W`` is
      ``(W f) + i (W g)`` bit-for-bit, so the two real maps are masked in
      place and never combined.
    * the full-``M`` round trip around the ``C_L`` multiply.  ``C_L`` is real
      and ``pair_from_full(C_L full_from_pair(r, s))`` differs from
      ``(C_L r, C_L s)`` only by the rounding of ``(a-b) + (a+b)`` against
      ``2a``, i.e. by an ulp; the row is unchanged to ~1e-14 relative
      (asserted in ``tests/test_exact_gl_speed.py``).
    * the full ``0..lmax`` order range of the step-1 synthesis, which acts on
      a single harmonic (:func:`~.grid.gl_synthesis_single_m`, bit-identical).
    """
    r, s, has_s = _unit_pair_alm(lmax, ellp, mp)
    am = abs(mp)

    def k_c_k(unit: np.ndarray) -> np.ndarray:
        """``K C K`` on one part: steps 1, 2 and 3 with ``K = S^dagger W S``."""
        field = gl_synthesis_single_m(unit, lmax, am, lmax_grid, nthreads=nthreads)
        field *= mask_map
        coeff = gl_analysis(field, lmax, lmax_grid, nthreads=nthreads)  # step 1
        coeff *= cl_alm  # step 2
        field = gl_synthesis(coeff, lmax, lmax_grid, nthreads=nthreads)
        field *= mask_map
        return gl_analysis(field, lmax, lmax_grid, nthreads=nthreads)  # step 3

    return k_c_k(r), (k_c_k(s) if has_s else None)


def _abs2_sum_over_m(
    ra: np.ndarray, sa: np.ndarray | None, ell_alm: np.ndarray, lmax: int
) -> np.ndarray:
    r"""
    ``sum_m |c[l, m]|^2`` for ``m = -l..l`` from the pair ``(ra, sa)``.

    With ``c[l, m] = ra + i sa`` and ``c[l, -m] = (-1)^m (conj ra + i conj sa)``
    (:func:`~.grid.full_from_pair`) the modulus of the ``-m`` entry is
    ``|conj(ra) + i conj(sa)|``, so the sum only needs the healpy half of the
    coefficients.  This replaces building the ``(lmax+1, 2 lmax+1)`` column and
    summing it, which cost a quarter of the per-column time.
    """
    up = ra if sa is None else ra + 1j * sa
    w = up.real**2 + up.imag**2
    nm0 = lmax + 1  # healpy stores the m = 0 block first, then m >= 1
    if ra.size > nm0:
        lo = np.conj(ra[nm0:])
        if sa is not None:
            lo = lo + 1j * np.conj(sa[nm0:])
        w[nm0:] += lo.real**2 + lo.imag**2
    return np.bincount(ell_alm, weights=w, minlength=lmax + 1)


def _exact_row_gl(
    mask_alm: np.ndarray,
    lw: int,
    cl: np.ndarray,
    ellp: int,
    lmax: int,
    lmax_grid: int | None = None,
    use_symmetry: bool = True,
    nthreads: int | None = None,
    mask_map: np.ndarray | None = None,
    reference: bool = False,
    lmax_int: int | None = None,
    term_selection: float | None = None,
) -> np.ndarray:
    r"""
    ``Sigma[:, ellp]`` of the exact-integral estimator for a band-limited mask.

    Parameters
    ----------
    mask_alm : ndarray
        healpy alm of the mask, truncated at ``lw``.
    lw : int
        Band-limit of the mask.
    cl, ellp, lmax, use_symmetry, lmax_int
        As in :func:`exact_covariance_row`.  The internal band-limit
        ``lmax_i = lmax if lmax_int is None else lmax_int`` drives every array
        here; the returned row is sliced back to ``lmax``.
    lmax_grid : int, optional
        GL grid band-limit.  ``None`` uses :func:`~.grid.gl_minimal_lmax`
        ``(lmax_i, lw)`` -- the *internal* band-limit, not the reported one --
        the smallest grid on which every transform in the algorithm is exact;
        a smaller grid aliases (error ~1e-2 one step below the minimum), a
        larger one only costs time.
    nthreads : int, optional
        Threads for ducc0.
    mask_map : ndarray, optional
        ``gl_synthesis(mask_alm, lw, lmax_grid)`` precomputed by the caller to
        share it across rows.
    reference : bool
        If True, use the literal complex-map implementation
        (:func:`_correlation_column_gl`) instead of the real-pair one.  Same
        answer to ~1e-14 relative and ~2.4x slower; kept so that
        ``tests/test_exact_gl_speed.py`` can pin the equivalence.
    term_selection : float, optional
        Skip the orders ``m'`` that :func:`_kept_mprime` drops at this
        tolerance.  ``mask_alm`` must already be in the pole frame
        (:func:`_pole_frame_alm`; the public entry points do this once) for
        the selection to be effective; it is correct in any frame.

    Returns
    -------
    ndarray
        ``Sigma[l, ellp]`` for ``l = 0..lmax``.
    """
    import healpy as hp  # lazy, see utils/healpy_utils.py

    lmax_i = int(lmax if lmax_int is None else lmax_int)
    mask_alm = np.asarray(mask_alm)
    cl = np.asarray(cl, dtype=float)[: lmax_i + 1]
    if mask_alm.size != hp.Alm.getsize(lw):
        raise ValueError("mask_alm size does not match lw")
    if lmax_i < lmax:
        raise ValueError("lmax_int must be at least lmax")
    if cl.size != lmax_i + 1:
        raise ValueError("cl must have at least lmax_int+1 entries")
    if not 0 <= ellp <= lmax:
        raise ValueError("ellp must lie in [0, lmax]")
    if lmax_grid is None:
        lmax_grid = gl_minimal_lmax(lmax_i, lw)
    if lmax_grid < lmax_i:
        raise ValueError("lmax_grid must be at least the internal band-limit")
    if mask_map is None:
        mask_map = gl_synthesis(mask_alm, lw, lmax_grid, nthreads=nthreads)

    ell = np.arange(lmax_i + 1)
    total = np.zeros(lmax_i + 1)
    ell_alm = hp.Alm.getlm(lmax_i)[0]
    cl_alm = cl[ell_alm]

    if use_symmetry:
        m_values = range(0, ellp + 1)
    else:
        m_values = range(-ellp, ellp + 1)
    keep = (
        None
        if term_selection is None
        else _kept_mprime(mask_alm, lw, ellp, term_selection, (0,), nthreads)
    )

    for mp in m_values:
        if keep is not None and not keep[mp + ellp]:
            continue
        if reference:
            c = _correlation_column_gl(
                mask_map, cl, ellp, mp, lmax_i, lmax_grid, nthreads
            )
            contrib = np.sum(np.abs(c) ** 2, axis=1)
        else:
            ra, sa = _correlation_column_pair_gl(
                mask_map, cl_alm, ellp, mp, lmax_i, lmax_grid, nthreads
            )
            contrib = _abs2_sum_over_m(ra, sa, ell_alm, lmax_i)
        if use_symmetry and mp > 0:
            contrib *= 2.0
        total += contrib

    row = 2.0 / ((2 * ell + 1) * (2 * ellp + 1)) * total
    return row[: lmax + 1]


# --------------------------------------------------------------------------- #
# Row-level parallelism (GL grid): rows are independent
# --------------------------------------------------------------------------- #
_WORKER: dict[str, object] = {}


def _row_worker_init(
    mask_alm,
    lw,
    cl,
    lmax,
    lmax_grid,
    use_symmetry,
    nthreads,
    lmax_int=None,
    term_selection=None,
):
    """Per-process setup: keep the row arguments and build the GL mask map once."""
    _WORKER.clear()
    _WORKER.update(
        mask_alm=mask_alm,
        lw=lw,
        cl=cl,
        lmax=lmax,
        lmax_grid=lmax_grid,
        use_symmetry=use_symmetry,
        nthreads=nthreads,
        lmax_int=lmax_int,
        term_selection=term_selection,
        mask_map=gl_synthesis(mask_alm, lw, lmax_grid, nthreads=nthreads),
    )


def _row_worker(ellp: int):
    """One row in a worker process; returns ``(ellp, Sigma[:, ellp])``."""
    return ellp, _exact_row_gl(
        _WORKER["mask_alm"],
        _WORKER["lw"],
        _WORKER["cl"],
        ellp,
        _WORKER["lmax"],
        lmax_grid=_WORKER["lmax_grid"],
        use_symmetry=_WORKER["use_symmetry"],
        nthreads=_WORKER["nthreads"],
        mask_map=_WORKER["mask_map"],
        lmax_int=_WORKER["lmax_int"],
        term_selection=_WORKER["term_selection"],
    )


def _rows_gl_parallel(
    mask_alm,
    lw,
    cl,
    rows,
    lmax,
    lmax_grid,
    use_symmetry,
    nthreads,
    nprocs,
    lmax_int=None,
    term_selection=None,
):
    """
    ``[(ellp, row), ...]`` for every requested row, computed in ``nprocs``
    processes.  Each row is bit-identical to the serial one for the same
    ``nthreads`` (ducc0 splits a transform deterministically over rings, and
    every row is computed by a single call chain).
    """
    rows = list(rows)
    nprocs = min(nprocs, len(rows)) or 1
    if nthreads is None:
        nthreads = max(1, (os.cpu_count() or 1) // nprocs)
    # O(l'^4) per row: start the expensive ones first so the tail is short.
    order = sorted(rows, key=lambda lp: -lp)
    with ProcessPoolExecutor(
        max_workers=nprocs,
        initializer=_row_worker_init,
        initargs=(
            mask_alm,
            lw,
            cl,
            lmax,
            lmax_grid,
            use_symmetry,
            nthreads,
            lmax_int,
            term_selection,
        ),
    ) as pool:
        return list(pool.map(_row_worker, order))


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


def _apply_k_gl(
    x_full: np.ndarray,
    spin: int,
    mask_map: np.ndarray,
    lmax: int,
    lmax_grid: int,
    nthreads: int | None,
) -> np.ndarray:
    """``K x = S^dagger W S x`` on full-``M`` coefficients (spin 0 or 2)."""
    y = gl_synthesis_complex(x_full, lmax, lmax_grid, spin=spin, nthreads=nthreads)
    return gl_analysis_complex(
        mask_map * y, lmax, lmax_grid, spin=spin, nthreads=nthreads
    )


def _correlation_columns_gl(
    mask_map: np.ndarray,
    cmat: np.ndarray,
    ellp: int,
    mp: int,
    lmax: int,
    lmax_grid: int,
    nthreads: int | None,
    fields: Sequence[str],
) -> np.ndarray:
    r"""
    ``R[X, Z, l, m] = <a~^X_lm conj(a~^Z_l'm')>`` for the columns
    ``Z in fields`` and every ``X in fields`` (other entries left at zero),
    shape ``(3, 3, lmax+1, 2*lmax+1)``.  Generalises
    :func:`_correlation_column_gl` (its result is ``R[0, 0]``).
    """
    nf = len(FIELDS)
    r = np.zeros((nf, nf, lmax + 1, 2 * lmax + 1), dtype=complex)
    want_t = "T" in fields
    want_p = "E" in fields or "B" in fields

    for z in fields:
        iz = FIELDS.index(z)
        # Step 1: u = K e_{Z, l'm'}
        u = np.zeros((nf, lmax + 1, 2 * lmax + 1), dtype=complex)
        if iz == 0:
            e = np.zeros((lmax + 1, 2 * lmax + 1), dtype=complex)
            e[ellp, lmax + mp] = 1.0
            u[0] = _apply_k_gl(e, 0, mask_map, lmax, lmax_grid, nthreads)
        else:
            if ellp < 2:
                continue  # E and B have no l' < 2 modes
            e = np.zeros((2, lmax + 1, 2 * lmax + 1), dtype=complex)
            e[iz - 1, ellp, lmax + mp] = 1.0
            u[1:] = _apply_k_gl(e, 2, mask_map, lmax, lmax_grid, nthreads)

        # Step 2: v^X_LM = sum_Z' C^{XZ'}_L u^{Z'}_LM
        v = np.einsum("xzl,zlm->xlm", cmat, u)

        # Step 3: R^{.Z} = K v
        if want_t and np.any(v[0]):
            r[0, iz] = _apply_k_gl(v[0], 0, mask_map, lmax, lmax_grid, nthreads)
        if want_p and np.any(v[1:]):
            r[1:, iz] = _apply_k_gl(v[1:], 2, mask_map, lmax, lmax_grid, nthreads)
    return r


def _exact_row_pol_gl(
    mask_alm: np.ndarray,
    lw: int,
    cmat: np.ndarray,
    ellp: int,
    lmax: int,
    spectra: Sequence[str],
    lmax_grid: int | None = None,
    use_symmetry: bool = True,
    nthreads: int | None = None,
    mask_map: np.ndarray | None = None,
    lmax_int: int | None = None,
    term_selection: float | None = None,
) -> dict[tuple[str, str], np.ndarray]:
    r"""
    Row ``l'`` of every block ``Cov(C~^{s1}_l, C~^{s2}_l')`` for ``s1, s2`` in
    ``spectra``, on the GL grid.  Arguments as in :func:`_exact_row_gl` with
    ``cmat`` from :func:`_cl_matrix`, whose last axis must run to the internal
    band-limit ``lmax_i = lmax if lmax_int is None else lmax_int``.  Returns
    ``{(s1, s2): array over l = 0..lmax}``.  ``term_selection`` skips the
    ``m'`` of :func:`_kept_mprime`, with the spin-0 and spin-2 mode powers
    united when a polarised field is requested.
    """
    import healpy as hp  # lazy, see utils/healpy_utils.py

    lmax_i = int(lmax if lmax_int is None else lmax_int)
    mask_alm = np.asarray(mask_alm)
    if mask_alm.size != hp.Alm.getsize(lw):
        raise ValueError("mask_alm size does not match lw")
    if lmax_i < lmax:
        raise ValueError("lmax_int must be at least lmax")
    if cmat.shape != (3, 3, lmax_i + 1):
        raise ValueError("cmat must have shape (3, 3, lmax_int+1)")
    if not 0 <= ellp <= lmax:
        raise ValueError("ellp must lie in [0, lmax]")
    if lmax_grid is None:
        lmax_grid = gl_minimal_lmax(lmax_i, lw)
    if lmax_grid < lmax_i:
        raise ValueError("lmax_grid must be at least the internal band-limit")
    if mask_map is None:
        mask_map = gl_synthesis(mask_alm, lw, lmax_grid, nthreads=nthreads)

    specs, fields = _spectra_fields(spectra)
    pairs = [(s1, s2) for s1 in specs for s2 in specs]
    idx = {s: (FIELDS.index(s[0]), FIELDS.index(s[1])) for s in specs}
    total = {pair: np.zeros(lmax_i + 1, dtype=complex) for pair in pairs}

    if use_symmetry:
        m_values = range(0, ellp + 1)
    else:
        m_values = range(-ellp, ellp + 1)
    keep = None
    if term_selection is not None:
        spins = (0,) if fields == ("T",) else (0, 2)
        keep = _kept_mprime(mask_alm, lw, ellp, term_selection, spins, nthreads)

    for mp in m_values:
        if keep is not None and not keep[mp + ellp]:
            continue
        r = _correlation_columns_gl(
            mask_map, cmat, ellp, mp, lmax_i, lmax_grid, nthreads, fields
        )
        weight = 2.0 if (use_symmetry and mp > 0) else 1.0
        for s1, s2 in pairs:
            x, y = idx[s1]
            z, w = idx[s2]
            contrib = np.sum(
                r[x, w] * np.conj(r[y, z]) + r[x, z] * np.conj(r[y, w]), axis=1
            )
            total[(s1, s2)] += weight * contrib

    ell = np.arange(lmax_i + 1)
    norm = 1.0 / ((2 * ell + 1) * (2 * ellp + 1))
    return {pair: (norm * total[pair].real)[: lmax + 1] for pair in pairs}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def exact_covariance_row(
    mask: np.ndarray,
    cl: np.ndarray,
    ellp: int,
    lmax: int,
    nside: int | None = None,
    iter: int = 3,
    use_symmetry: bool = True,
    exact_adjoint: bool = True,
    grid: str = "healpix",
    lw: int | None = None,
    lmax_grid: int | None = None,
    nthreads: int | None = None,
    lmax_int: int | None = None,
    term_selection: float | None = None,
) -> np.ndarray:
    r"""
    One row/column ``Sigma[:, ellp]`` of the exact TT pseudo-``C_l`` covariance.

    Implements paper Eq. (7) with the correlations of Eqs. (8)-(11) obtained
    by spherical harmonic transforms as in Fig. 2 of Camphuis et al. (2022).

    Parameters
    ----------
    mask : ndarray
        ``grid="healpix"``: real HEALPix mask (RING ordering) at resolution
        ``nside``.  ``grid="gl"``: either such a map (converted once with
        ``hp.map2alm(mask, lmax=lw, iter=10)``) or a complex healpy alm array
        of the mask already truncated at ``lw`` (used as is).
    cl : ndarray
        Real temperature power spectrum, at least ``lmax_int+1`` long (i.e.
        ``lmax+1`` when ``lmax_int`` is not given).
    ellp : int
        Multipole :math:`\ell'` of the row, in ``[0, lmax]``.
    lmax : int
        Maximum multipole *reported*.  Use ``nside >= lmax_int/2``.
    nside : int, optional
        HEALPix resolution of ``mask``.  Required for ``grid="healpix"``; for
        ``grid="gl"`` it is inferred from a map's size and unused for an alm.
    iter : int
        Number of map2alm iterations of the estimator (healpy default 3).
        HEALPix grid only.
    use_symmetry : bool
        If True, only ``m' >= 0`` is computed and the ``m' > 0`` terms are
        counted twice.  For real ``W`` and real ``C_L`` one has
        ``c_{-m'}[l, m] = (-1)^{m+m'} conj c_{m'}[l, -m]``, so the sum over
        ``m`` of ``|c|^2`` is identical for ``m'`` and ``-m'``.
    exact_adjoint : bool
        If True (default), step 1 uses the exact adjoint of the iterated
        analysis so that the result is :math:`K C K^\dagger`.  If False, the
        naive :math:`K C K` (analyse ``W Y_{l'm'}`` directly) is computed.
        HEALPix grid only; on GL analysis is the adjoint of synthesis.
    grid : {"healpix", "gl"}
        Which estimator's covariance to compute (module docstring).
        ``"healpix"``: the covariance of the pixelised ``map2alm(iter)``
        estimator, i.e. of ``hp.anafast`` applied to a HEALPix map times the
        pixel mask.  ``"gl"``: the covariance of the exact-integral estimator
        for the band-limited mask, computed on a Gauss-Legendre grid to
        machine precision.  They differ at ~1e-5 (diagonal band) / ~1e-3
        (far off-diagonal) at ``nside=32`` and converge as ``nside^-2``.
    lw : int, optional
        GL grid only: band-limit of the mask.  Defaults to ``3*nside - 1``
        for a map input and to the array's own band-limit for an alm input.
    lmax_grid : int, optional
        GL grid only: GL grid band-limit.  ``None`` selects the minimal exact
        grid :func:`~.grid.gl_minimal_lmax` ``(lmax_int, lw)`` -- sized from
        the *internal* band-limit, since that is what the transforms carry.
    nthreads : int, optional
        GL grid only: threads for ducc0.
    lmax_int : int, optional
        Internal band-limit.  ``C_L`` is summed and every harmonic array is
        carried to ``lmax_int`` while only ``l = 0..lmax`` is returned, so the
        row is identical to the one computed with ``lmax=lmax_int`` and sliced.
        ``None`` (default) means ``lmax_int = lmax``, the historical behaviour.
        Because the mask couples ``l'`` to ``L`` over its own width, a row at
        ``l'`` needs ``C_L`` above ``l'``; a margin ``lmax_int - lmax`` below
        :func:`mask_coupling_width` raises a ``UserWarning`` naming both
        numbers (module docstring, "Reported band-limit versus internal
        band-limit").
    term_selection : float, optional
        GL grid only.  ``None`` (default): every ``m'`` is computed.  A
        positive tolerance rotates the mask alm to the pole frame once (the
        row is rotation invariant, see :func:`_pole_frame_alm`) and skips
        the orders ``m'`` whose mask overlap is below ``tolerance / 10`` of
        the largest (:func:`_kept_mprime`, the ``eps_m`` rule of
        :func:`~cmbcov.term_selection.select_terms`).
        For a 20-degree cap about 60% of the ``m'`` are skipped.  Raises
        ``ValueError`` with ``grid="healpix"``.

    Returns
    -------
    ndarray
        ``Sigma[l, ellp]`` for ``l = 0..lmax``.

    Notes
    -----
    Cost is :math:`O(\ell'^4)`: :math:`\ell'+1` (or :math:`2\ell'+1`) columns,
    each requiring a fixed number of :math:`O(\ell_{\rm int}^3)` transforms.
    """
    import healpy as hp  # lazy, see utils/healpy_utils.py

    lmax_i = int(lmax if lmax_int is None else lmax_int)
    if lmax_i < lmax:
        raise ValueError("lmax_int must be at least lmax")
    if term_selection is not None and grid != "gl":
        raise ValueError("term_selection is implemented for grid='gl' only")

    if grid == "gl":
        mask_alm, lw = _mask_alm_for_gl(mask, nside, lw)
        _warn_short_margin(alm2cl(mask_alm), lmax, lmax_i)
        if term_selection is not None:
            mask_alm = _pole_frame_alm(mask, mask_alm, lw)
        return _exact_row_gl(
            mask_alm,
            lw,
            cl,
            ellp,
            lmax,
            lmax_grid=lmax_grid,
            use_symmetry=use_symmetry,
            nthreads=nthreads,
            lmax_int=lmax_int,
            term_selection=term_selection,
        )
    if grid != "healpix":
        raise ValueError(f"grid must be 'healpix' or 'gl', got {grid!r}")
    if nside is None:
        raise ValueError("nside is required for grid='healpix'")

    mask = np.asarray(mask, dtype=float)
    cl = np.asarray(cl, dtype=float)[: lmax_i + 1]
    if mask.size != hp.nside2npix(nside):
        raise ValueError("mask size does not match nside")
    if cl.size != lmax_i + 1:
        raise ValueError("cl must have at least lmax_int+1 entries")
    if not 0 <= ellp <= lmax:
        raise ValueError("ellp must lie in [0, lmax]")
    _warn_short_margin(_mask_wl_healpix(mask, nside), lmax, lmax_i)

    ell = np.arange(lmax_i + 1)
    total = np.zeros(lmax_i + 1)

    if use_symmetry:
        m_values = range(0, ellp + 1)
    else:
        m_values = range(-ellp, ellp + 1)

    for mp in m_values:
        c = _correlation_column(mask, cl, ellp, mp, lmax_i, nside, iter, exact_adjoint)
        contrib = np.sum(np.abs(c) ** 2, axis=1)
        if use_symmetry and mp > 0:
            contrib *= 2.0
        total += contrib

    row = 2.0 / ((2 * ell + 1) * (2 * ellp + 1)) * total
    return row[: lmax + 1]


def exact_covariance(
    mask: np.ndarray,
    cl: np.ndarray,
    lmax: int,
    nside: int | None = None,
    rows: Iterable[int] | None = None,
    grid: str = "healpix",
    lw: int | None = None,
    lmax_grid: int | None = None,
    nprocs: int = 1,
    lmax_int: int | None = None,
    term_selection: float | None = None,
    **kw,
) -> np.ndarray:
    r"""
    Exact TT pseudo-``C_l`` covariance matrix ``Sigma[l, l']`` (paper Eq. 7).

    Parameters
    ----------
    mask, cl, lmax, nside, grid, lw, lmax_grid, lmax_int
        As in :func:`exact_covariance_row`.  With ``grid="gl"`` the mask is
        converted to its band-limited alm and synthesised on the GL grid once,
        shared by all rows.  ``lmax_int`` gives every row the same internal
        band-limit, so the returned ``(lmax+1, lmax+1)`` matrix equals the
        ``(lmax_int+1, lmax_int+1)`` one sliced to ``lmax``, without computing
        the rows above ``lmax``.
    rows : iterable of int, optional
        Multipoles :math:`\ell'` whose columns ``Sigma[:, l']`` are computed.
        Defaults to all ``0..lmax``.  Columns not requested are left at zero.
    nprocs : int
        ``grid="gl"`` only: number of worker *processes* over which the rows
        are distributed (they are independent).  The default 1 keeps the
        calculation in this process and is bit-identical to the serial loop.
        Above 1, each worker rebuilds the GL mask map (a few MB) and runs
        ducc0 with ``nthreads`` threads; if ``nthreads`` is not given it
        defaults to ``max(1, cpu_count // nprocs)`` inside the workers so that
        the two levels of parallelism do not oversubscribe the machine.  Rows
        have very different costs (``O(l'^4)``), so they are submitted
        longest-first.  On platforms whose default start method is ``spawn``
        (macOS, Windows) the caller must be import-safe, i.e. inside
        ``if __name__ == "__main__":``.

        Measure before using it.  On a 14-core laptop at ``lmax=500``,
        ``Lw=384`` every combination gives bit-identical numbers but
        ``nprocs=14`` is 1.56x *slower* than one process on 14 threads: the
        performance/efficiency core split caps static per-row parallelism at
        ~8x, and 14 copies of the 7.7 MB maps then run into memory bandwidth.
        It should win where ducc0's
        thread scaling runs out first -- many homogeneous cores, or a grid
        small enough that a transform does not thread well.
    term_selection : float, optional
        ``grid="gl"`` only; as in :func:`exact_covariance_row`.  The mask alm
        is rotated to the pole frame once here and shared by every row (and
        every worker process).
    **kw
        Passed to :func:`exact_covariance_row` (``iter``, ``use_symmetry``,
        ``exact_adjoint``, ``nthreads``).

    Returns
    -------
    ndarray
        Shape ``(lmax+1, lmax+1)``.  Cost is :math:`O(\ell^5)` for all rows.
    """
    if rows is None:
        rows = range(lmax + 1)
    sigma = np.zeros((lmax + 1, lmax + 1))
    nprocs = int(nprocs)
    if nprocs < 1:
        raise ValueError("nprocs must be at least 1")
    lmax_i = int(lmax if lmax_int is None else lmax_int)
    if lmax_i < lmax:
        raise ValueError("lmax_int must be at least lmax")
    if term_selection is not None and grid != "gl":
        raise ValueError("term_selection is implemented for grid='gl' only")

    if grid == "gl":
        mask_alm, lw = _mask_alm_for_gl(mask, nside, lw)
        _warn_short_margin(alm2cl(mask_alm), lmax, lmax_i)
        if term_selection is not None:
            mask_alm = _pole_frame_alm(mask, mask_alm, lw)
        if lmax_grid is None:
            lmax_grid = gl_minimal_lmax(lmax_i, lw)
        nthreads = kw.get("nthreads")
        use_symmetry = kw.get("use_symmetry", True)
        cl = np.asarray(cl, dtype=float)
        rows = list(rows)
        if nprocs > 1:
            for ellp, row in _rows_gl_parallel(
                mask_alm,
                lw,
                cl,
                rows,
                lmax,
                lmax_grid,
                use_symmetry,
                nthreads,
                nprocs,
                lmax_int=lmax_int,
                term_selection=term_selection,
            ):
                sigma[:, ellp] = row
            return sigma
        mask_map = gl_synthesis(mask_alm, lw, lmax_grid, nthreads=nthreads)
        for ellp in rows:
            sigma[:, ellp] = _exact_row_gl(
                mask_alm,
                lw,
                cl,
                ellp,
                lmax,
                lmax_grid=lmax_grid,
                use_symmetry=use_symmetry,
                nthreads=nthreads,
                mask_map=mask_map,
                lmax_int=lmax_int,
                term_selection=term_selection,
            )
        return sigma
    if nprocs > 1:
        raise ValueError("nprocs > 1 is implemented for grid='gl' only")

    for ellp in rows:
        sigma[:, ellp] = exact_covariance_row(
            mask, cl, ellp, lmax, nside, grid=grid, lmax_int=lmax_int, **kw
        )
    return sigma


def exact_covariance_row_pol(
    mask: np.ndarray,
    cls: Mapping[str, np.ndarray],
    ellp: int,
    lmax: int,
    spectra: Sequence[str] = ("TT", "TE", "EE", "BB"),
    grid: str = "gl",
    nside: int | None = None,
    lw: int | None = None,
    lmax_grid: int | None = None,
    use_symmetry: bool = True,
    nthreads: int | None = None,
    lmax_int: int | None = None,
    term_selection: float | None = None,
    **kw,
) -> dict[tuple[str, str], np.ndarray]:
    r"""
    One row ``l'`` of the exact polarised pseudo-``C_l`` covariance.

    Returns ``Cov(C~^{s1}_l, C~^{s2}_{l'})`` for ``l = 0..lmax`` and every
    ordered pair ``(s1, s2)`` of the requested ``spectra`` (module docstring,
    "Polarisation").  GL grid only; the HEALPix path is TT-only.

    Parameters
    ----------
    mask : ndarray
        As in :func:`exact_covariance_row` with ``grid="gl"``: a real HEALPix
        map (converted once with ``hp.map2alm(mask, lmax=lw, iter=10)``) or a
        complex healpy alm array truncated at ``lw``.
    cls : mapping
        Power spectra keyed by name among ``TT, EE, BB, TE, TB, EB`` (``"ET"``
        etc. are accepted); missing spectra are zero.  Each at least
        ``lmax_int+1`` long (``lmax+1`` when ``lmax_int`` is not given).
    ellp : int
        Multipole :math:`\ell'` of the row.
    lmax : int
        Maximum multipole of the pseudo-spectra.
    spectra : sequence of str
        Spectra whose covariance blocks are wanted.  Only the correlator
        columns their fields need are computed: ``("TT",)`` costs the same as
        :func:`exact_covariance_row`; ``("TT", "TE", "EE")`` needs the T and E
        columns; anything with B needs all three.
    grid : {"gl", "healpix"}
        ``"gl"`` (default) computes the covariance of the exact-integral
        estimator of the band-limited mask.  ``"healpix"`` is only accepted
        for ``spectra=("TT",)``, where it delegates to
        :func:`exact_covariance_row` (``nside`` required; ``iter`` and
        ``exact_adjoint`` may be passed through ``**kw``); any polarised
        spectrum raises ``NotImplementedError``.
    nside, lw, lmax_grid, use_symmetry, nthreads, lmax_int
        As in :func:`exact_covariance_row`.  ``use_symmetry`` sums
        ``m' >= 0`` only and doubles the real part of the ``m' > 0`` terms.
        ``lmax_int`` carries the spectra, the harmonic arrays and the GL grid
        to the internal band-limit and returns ``l = 0..lmax`` only.
    term_selection : float, optional
        GL grid only; as in :func:`exact_covariance_row`, with the spin-0
        and spin-2 mode powers united when a polarised field is requested.

    Returns
    -------
    dict
        ``{(s1, s2): ndarray of shape (lmax+1,)}`` with
        ``result[(s1, s2)][l] = Cov(C~^{s1}_l, C~^{s2}_{l'})`` for all ordered
        pairs of the canonical spectrum names; ``(s2, s1)`` is the transposed
        block, i.e. a different array of the same row.

    Notes
    -----
    Per ``(l', m')`` the T column costs one spin-0 round trip plus one spin-0
    and one spin-2 round trip (the latter only if a polarised field is
    requested); each of the E and B columns costs one spin-2 round trip plus
    the same two, so the full six-spectrum row is ~7x the TT row in
    transforms.  Rows with ``l' < 2`` have zero E and B columns.
    """
    specs, fields = _spectra_fields(spectra)
    lmax_i = int(lmax if lmax_int is None else lmax_int)
    if lmax_i < lmax:
        raise ValueError("lmax_int must be at least lmax")
    if term_selection is not None and grid != "gl":
        raise ValueError("term_selection is implemented for grid='gl' only")
    if grid == "healpix":
        if fields != ("T",):
            raise NotImplementedError(
                "polarised exact covariance is implemented on the Gauss-Legendre "
                "grid only (grid='gl'); the HEALPix path is TT-only"
            )
        cl_tt = np.asarray(cls.get("TT", np.zeros(lmax_i + 1)), dtype=float)
        row = exact_covariance_row(
            mask,
            cl_tt,
            ellp,
            lmax,
            nside=nside,
            use_symmetry=use_symmetry,
            grid="healpix",
            lmax_int=lmax_int,
            **kw,
        )
        return {("TT", "TT"): row}
    if grid != "gl":
        raise ValueError(f"grid must be 'gl' or 'healpix', got {grid!r}")
    if kw:
        raise TypeError(f"unexpected keyword arguments for grid='gl': {sorted(kw)}")

    mask_alm, lw = _mask_alm_for_gl(mask, nside, lw)
    _warn_short_margin(alm2cl(mask_alm), lmax, lmax_i)
    if term_selection is not None:
        mask_alm = _pole_frame_alm(mask, mask_alm, lw)
    cmat = _cl_matrix(cls, lmax_i)
    return _exact_row_pol_gl(
        mask_alm,
        lw,
        cmat,
        ellp,
        lmax,
        specs,
        lmax_grid=lmax_grid,
        use_symmetry=use_symmetry,
        nthreads=nthreads,
        lmax_int=lmax_int,
        term_selection=term_selection,
    )


def exact_covariance_pol(
    mask: np.ndarray,
    cls: Mapping[str, np.ndarray],
    lmax: int,
    spectra: Sequence[str] = ("TT", "TE", "EE", "BB"),
    rows: Iterable[int] | None = None,
    grid: str = "gl",
    nside: int | None = None,
    lw: int | None = None,
    lmax_grid: int | None = None,
    use_symmetry: bool = True,
    nthreads: int | None = None,
    lmax_int: int | None = None,
    term_selection: float | None = None,
    **kw,
) -> dict[tuple[str, str], np.ndarray]:
    r"""
    Exact polarised pseudo-``C_l`` covariance blocks ``Sigma^{s1 s2}[l, l']``.

    Parameters
    ----------
    mask, cls, lmax, spectra, grid, nside, lw, lmax_grid, use_symmetry, nthreads, lmax_int
        As in :func:`exact_covariance_row_pol`.  With ``grid="gl"`` the mask
        is converted to its band-limited alm and synthesised on the GL grid
        once, shared by all rows.
    rows : iterable of int, optional
        Multipoles :math:`\ell'` whose columns ``Sigma[:, l']`` are computed.
        Defaults to all ``0..lmax``.  Columns not requested are left at zero.
    term_selection : float, optional
        GL grid only; as in :func:`exact_covariance_row_pol`, the mask alm
        rotated to the pole frame once and shared by every row.
    **kw
        HEALPix-only options (``iter``, ``exact_adjoint``) forwarded to
        :func:`exact_covariance_row`; only valid with ``spectra=("TT",)``.

    Returns
    -------
    dict
        ``{(s1, s2): ndarray of shape (lmax+1, lmax+1)}`` with
        ``result[(s1, s2)][l, l'] = Cov(C~^{s1}_l, C~^{s2}_{l'})``, so that
        ``result[(s2, s1)] == result[(s1, s2)].T`` once all rows are computed.
    """
    specs, fields = _spectra_fields(spectra)
    lmax_i = int(lmax if lmax_int is None else lmax_int)
    if lmax_i < lmax:
        raise ValueError("lmax_int must be at least lmax")
    if rows is None:
        rows = range(lmax + 1)
    pairs = [(s1, s2) for s1 in specs for s2 in specs]
    sigma = {pair: np.zeros((lmax + 1, lmax + 1)) for pair in pairs}
    if term_selection is not None and grid != "gl":
        raise ValueError("term_selection is implemented for grid='gl' only")

    if grid == "healpix":
        if fields != ("T",):
            raise NotImplementedError(
                "polarised exact covariance is implemented on the Gauss-Legendre "
                "grid only (grid='gl'); the HEALPix path is TT-only"
            )
        cl_tt = np.asarray(cls.get("TT", np.zeros(lmax_i + 1)), dtype=float)
        sigma[("TT", "TT")] = exact_covariance(
            mask,
            cl_tt,
            lmax,
            nside=nside,
            rows=rows,
            grid="healpix",
            use_symmetry=use_symmetry,
            lmax_int=lmax_int,
            **kw,
        )
        return sigma
    if grid != "gl":
        raise ValueError(f"grid must be 'gl' or 'healpix', got {grid!r}")
    if kw:
        raise TypeError(f"unexpected keyword arguments for grid='gl': {sorted(kw)}")

    mask_alm, lw = _mask_alm_for_gl(mask, nside, lw)
    _warn_short_margin(alm2cl(mask_alm), lmax, lmax_i)
    if term_selection is not None:
        mask_alm = _pole_frame_alm(mask, mask_alm, lw)
    if lmax_grid is None:
        lmax_grid = gl_minimal_lmax(lmax_i, lw)
    cmat = _cl_matrix(cls, lmax_i)
    mask_map = gl_synthesis(mask_alm, lw, lmax_grid, nthreads=nthreads)
    for ellp in rows:
        row = _exact_row_pol_gl(
            mask_alm,
            lw,
            cmat,
            ellp,
            lmax,
            specs,
            lmax_grid=lmax_grid,
            use_symmetry=use_symmetry,
            nthreads=nthreads,
            mask_map=mask_map,
            lmax_int=lmax_int,
            term_selection=term_selection,
        )
        for pair in pairs:
            sigma[pair][:, ellp] = row[pair]
    return sigma
