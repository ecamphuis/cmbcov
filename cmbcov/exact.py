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
**Per column** (``_exact_row_gl(..., batched=False)``, the reference).
Since the complex map of a column is only ever a pair of real maps, this path
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

**Batched** (the default).  Step 3 applies one linear operator to every
column of the row, and every transform of a column is a Legendre sum per
azimuthal order plus an FFT per ring.  A chunk of columns therefore goes
through the Legendre sums as matrix products, ``(L x rings) @ (rings x
columns)`` per order with a table of :math:`\lambda_{LM}(\theta)`
(:func:`~.grid.gl_legendre_table`, northern rings only, split by the parity
of ``L + M``), and through one batched FFT per ring
(:func:`_batched_chunk_gl`).  Step 1 needs no transform at all: the
masked single harmonic has the ring profile
:math:`W_{M - m'}(\theta) \lambda_{\ell' m'}(\theta)` in order ``M``, so
its analysis is one more matrix product.  Only what can be non-zero is
computed: the orders ``|M - m'| <= Lw`` after step 1 and ``<= 2 Lw`` after
step 3, and the degrees ``|L - l'| <= Lw`` of the step-1 coefficients (and
so of the synthesis) and ``|L - l'| <= 2 Lw`` of the analysis -- the Gaunt
selection rule of a mask of band-limit ``Lw`` -- so the matrix products run
over those rows only and the table is only generated up to the top of the
window.  The FFT is only as long as the order band needs
(:func:`_chunk_fft_size`).  The chunks are sized by ``max_memory_gb``
(:func:`_batched_row_plan`), which also decides whether the Legendre table
is held for every row or regenerated per chunk into one reused buffer; ducc0
takes one field per call and recomputes its recursion per transform, so it
cannot share that work between columns itself, and the table is read off
``leg2alm`` two rings per call (:func:`~.grid.gl_legendre_table`, straight
into ring-contiguous rows).  Nothing in the plan depends on the number of
threads, and the ring stage (:func:`_ring_stage`) hands fixed blocks of
rings to its workers, so a batched row is the same to the bit for any
``nthreads`` (asserted for 1 to 8 threads).  The result is the per-column
row to a few 1e-15 of its largest entry (asserted to 1e-13 in
``tests/test_exact_batched.py``, on masks with no symmetry axis: on an
azimuthally symmetric one, such as the baseline cap of ``tests/data``, a
column only couples ``M = m'`` and the order bands are not exercised).  On
the two-patch test mask at ``lmax`` 32 the per-column row itself moves by
up to 3e-14 of its maximum when the grid is enlarged.

Measured back to back on a 14-core laptop shared with other jobs (load
about 10-15), ``Lw = 384``, a two-patch mask, row ``l' = lmax``, default
budget of 2 GiB, batched against one column at a time: ``lmax`` 500, 3.7-10.1
s against 24.8 s; 1000, 28-36 s against 130 s; 1500, 88-90 against 245 ms
per column; 2000, 112-179 against 521 ms per column (1500 and 2000 estimated
from sampled columns).  So the batched row is faster at every size, roughly
2.5-4x (the ranges are the machine's load).  The same holds for the
polarised rows below.

**Polarised rows**, two kernels; the per-column reference is
:func:`_correlation_columns_gl` (``_exact_row_pol_gl(..., batched=False)``).
An ``m'`` carries up to nine ring profiles (the T, E and B columns on T, Q
and U), so at the same budget a chunk holds up to nine times fewer ``m'``
than a TT one, and the tables are up to three times larger (spin 0 for T,
spin 2 at twice the size).

*Tables held* (they fit in half the budget: up to ``lmax`` ~ 500 at 2 GiB
with ``Lw = 384``): :func:`_batched_chunk_pol_gl`, the TT scheme in a basis
where ducc0's spin-2 transforms carry no complex factor.  On the
coefficients (T, E, B' = iB) and on the rings (T, Q, U'' = iU), the spin-2
synthesis and analysis of order ``M >= 0`` are the real symmetric block
``[[lambda+, lambda-], [lambda-, lambda+]]`` of the two spin-weighted
Legendre functions (:func:`~.grid.gl_legendre_table` with ``spin=2``, whose
convention is asserted against ``alm2leg`` and
:func:`~.grid.gl_synthesis_complex`); at ``-M`` the table of ``|M|`` serves
with :math:`\lambda^-` negated.  :math:`\lambda^+` has the mirror parity
:math:`(-1)^{L+M}` and :math:`\lambda^-` the opposite one, so each of the
four products runs on the northern rings against ``f_N + f_S`` or ``f_N -
f_S`` as for spin 0.  The unit vector of a B column is ``i`` times that of
B', whose step-1 coefficients are those of the E unit vector with E and B'
swapped, so a B column costs no step-1 work.  The ``C`` mixing is ``D C
D^{-1}`` with ``D = diag(1, 1, i)``, applied per ``L``; the factors of ``i``
are restored in the contraction (:func:`_pol_terms`), which is, per ``(L,
M)``, the Gram matrix of the (output, column) pairs summed over the chunk's
columns -- a batched matrix product on a staging buffer of about 16 orders.

*Tables that would be streamed*: :func:`_batched_chunk_pol_direct`, the
same bands, windows, ring stage and contraction, with every Legendre sum a
ducc0 transform (``leg2alm`` / ``alm2leg``) on the column's own orders and
degrees -- no table at all, and the functions ducc0 evaluates are the ones
of the per-column path.  With a few ``m'`` per chunk a table cannot pay
for itself: it is regenerated for every chunk and every pass, and shared by
only a few columns, so transforming them directly is cheaper.  Which kernel runs follows from the memory plan alone
(``hold_table``), not from timings.  (The tables must be ducc0's own values, not those of a numpy
recursion: near the poles at ``lmax`` 2000 the functions are
ill-conditioned in :math:`\cos\theta`, ducc0's values are 3e-10 from an
extended-precision evaluation there and a numpy recursion's 5e-10, and they
differ from each other by as much, so the batched row would no longer be
the per-column one to rounding.)

Both give the per-column row to a few 1e-15 of its maximum (asserted to
1e-13 of each block's Wick scale -- the Cauchy-Schwarz bound from the auto
blocks of the row -- in ``tests/test_exact_pol_batched.py``); on the columns
of one chunk at ``lmax`` 500 the per-column row itself moves 20-40 times
more when the grid is enlarged.

Measured the same way, all six spectra, row ``l' = lmax``, default budget of
2 GiB, batched against one column at a time: ``lmax`` 500, 59-122 s against
309 s; 1000, 591-749 s against 2241 s; 1500, 1.42 against 4.40 s per column;
2000, 1.55-1.99 against 5.73 s per column (1500 and 2000 estimated from
sampled columns), again roughly 2.5-4x.

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
``use_symmetry`` the sum is ``(m'=0 term) + 2 Re sum_{m'>0}``.  On a grid
at least :func:`~.grid.gl_minimal_lmax` the columns are computed in batches
within ``max_memory_gb`` ("Cost of the GL column", polarised rows).  The three
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

import ducc0
import numpy as np

from .grid import (
    _gl_geometry,
    full_from_pair,
    gl_analysis,
    gl_analysis_complex,
    gl_legendre_table,
    gl_legendre_table_bytes,
    gl_minimal_lmax,
    gl_north_rings,
    gl_ring_modes,
    gl_shape,
    gl_synthesis,
    gl_synthesis_complex,
    gl_synthesis_single_m,
    pair_from_full,
)
from .grid import _nthreads as _gl_nthreads
from .sht import alm2cl, ducc0_alm2map, ducc0_map2alm
from .term_selection import mode_power, pole_rotation, rotate_alm
from .utils.threading_utils import row_block_workers, run_row_blocks

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


# --------------------------------------------------------------------------- #
# Batched GL columns: Legendre GEMMs over a chunk of columns
# --------------------------------------------------------------------------- #
#: Default memory budget of a batched exact GL row, in GiB (``max_memory_gb``
#: of :func:`exact_covariance_row` / :func:`exact_covariance`); what it covers
#: is itemised in :func:`_batched_row_plan`.
DEFAULT_EXACT_MEMORY_GB = 2.0

#: Target size of one block of the Legendre table when the whole table is not
#: held (:class:`_BatchedGL`): large enough that generating it keeps every
#: worker busy, small next to the chunk working set.  Per spin-0 table: the
#: spin-2 table is twice as large per order, and a streamed block holding
#: spin 0 and spin 2 targets three times this, so that it spans as many
#: orders as a TT block (the generation cost is dominated by a fixed cost
#: per ducc0 call, one call per ring pair and block: at ``lmax`` 2000 on 14
#: threads, 1, 4 and 16 orders per call ran at 2.5, 0.9 and 0.4 ns per entry
#: for spin 0).  A held table is generated once, in blocks of this size
#: whatever its spins.
_TABLE_BLOCK_BYTES = 64 * 2**20

#: Bytes of the ring block whose two FFTs and mask product run together in
#: :func:`_ring_stage` (cache-sized, so the three passes hit cache).
_FFT_BLOCK_BYTES = 4 * 2**20

#: Most workers of :func:`_ring_stage` (and at most ``nthreads``).  Each
#: holds numpy temporaries of about one ring block (elementwise operations
#: on non-contiguous views buffer), which the plan charges for this many
#: workers whatever ``nthreads`` is, so that the chunks do not depend on it.
_RING_WORKERS = 16


def _ring_blocks(n_fft: int, n_north: int, width: int) -> tuple[int, int]:
    """``(rings per block, blocks)`` of :func:`_ring_stage`: fixed blocks
    of about half of :data:`_FFT_BLOCK_BYTES` (at least one ring) tiling
    the northern rings, whatever the number of workers."""
    step = min(n_north, max(1, _FFT_BLOCK_BYTES // (2 * n_fft * width * 16)))
    return step, -(-n_north // step)


def _ring_scratch(n_fft: int, ntheta: int, width: int) -> int:
    """Bytes charged for the ring stage's workers: one block each, for as
    many workers as can run (:data:`_RING_WORKERS`, at most one per
    block), independently of ``nthreads``."""
    step, nblock = _ring_blocks(n_fft, (ntheta + 1) // 2, width)
    return min(_RING_WORKERS, nblock) * step * n_fft * width * 16


#: ``(what, budget)`` already warned about by :func:`_batched_row_plan`.
_WARNED_MEMORY_FLOOR: set = set()


def _band(mps: np.ndarray, width: int, lmax: int) -> tuple[int, int]:
    """``[min m' - width, max m' + width]`` clipped to ``[-lmax, lmax]``."""
    return max(-lmax, int(mps.min()) - width), min(lmax, int(mps.max()) + width)


def _chunk_fft_size(mps: np.ndarray, lmax: int, lw: int) -> int:
    r"""
    Azimuthal FFT length of a chunk (:func:`_batched_chunk_gl`).

    Step 3 multiplies ring profiles with orders in the band
    ``B = [min m' - lw, max m' + lw]`` (the orders step 1 can reach) by a
    mask of orders ``|k| <= lw``; the product occupies ``|B| + 2 lw``
    consecutive orders, so any ``N`` at least that long resolves it without
    aliasing.  Only the outputs ``|M| <= lmax`` are kept, and a product order
    ``p`` and an output ``M`` differ by at most ``2 lmax + lw``, so
    ``N >= 2 lmax + lw + 1`` is enough too (the grid's ``nphi`` is such an
    ``N``).  The smaller of the two, rounded up to a fast FFT length.
    """
    lo, hi = _band(mps, lw, lmax)
    return int(ducc0.fft.good_size(min(hi - lo + 1 + 2 * lw, 2 * lmax + lw + 1)))


def _chunk_bytes(nc: int, n_fft: int, ntheta: int, lmax: int) -> dict[str, int]:
    """Working set of one chunk of ``nc`` columns (:func:`_batched_row_plan`),
    in bytes, as allocated by :func:`_batched_chunk_gl`."""
    terms = {
        # the chunk's ring profiles, complex, (n_fft, ntheta, nc)
        "fields": n_fft * ntheta * nc * 16,
        # the mask on the n_fft points of every ring, and its FFT input
        "ring_mask": n_fft * ntheta * (8 + 16),
        # per order (both signs of M): the step-1 legs g_N +- g_S, a gathered
        # mask block, the GEMM outputs in L and on the rings; the step-1
        # profiles and their alm2leg output
        "scratch": (3 * ntheta + (lmax + 1)) * 2 * nc * 16 + ntheta * nc * 24,
        # the ring stage's workers
        "ring": _ring_scratch(n_fft, ntheta, nc),
    }
    terms["total"] = sum(terms.values())
    return terms


def _table_bytes(
    ms: np.ndarray, lmax: int, lmax_grid: int, spins: Sequence[int]
) -> int:
    """Bytes of the Legendre tables of ``spins`` for the orders ``ms``."""
    return sum(gl_legendre_table_bytes(ms, lmax, lmax_grid, spin=s) for s in spins)


def _table_block_target(spins: Sequence[int], hold_table: bool) -> int:
    """Target bytes of a table block: :data:`_TABLE_BLOCK_BYTES` for a held
    table, and that per spin-0-sized table (1 for spin 0, 2 for spin 2) for
    a streamed one."""
    if hold_table:
        return _TABLE_BLOCK_BYTES
    return _TABLE_BLOCK_BYTES * sum(1 if s == 0 else 2 for s in spins)


def _chunk_bytes_pol(
    nc: int,
    n_fft: int,
    ntheta: int,
    lmax: int,
    want_t: bool = True,
    nz: int = 3,
    n_out: int = 3,
    nkeys: int = 45,
) -> dict[str, int]:
    """Working set of one chunk of ``nc`` orders ``m'`` of a polarised row
    (:func:`_batched_chunk_pol_gl`), in bytes: ``nz`` column types per
    ``m'`` (T, E, B), each carried on the rings as T (if ``want_t``), Q and
    iU, i.e. ``width = (want_t + 2) nz nc`` complex ring columns, and
    ``n_out`` output fields contracted into ``nkeys`` sums.  An upper bound
    of what the kernel allocates at once (``tests/test_exact_pol_batched.py``
    traces it)."""
    n_north = (ntheta + 1) // 2
    w = nz * nc
    width = (int(want_t) + 2) * w
    rows = lmax + 2  # even + odd degrees of one order, both signs of M below
    terms = {
        # the chunk's ring profiles, complex, (n_fft, ntheta, width)
        "fields": n_fft * ntheta * width * 16,
        # the mask on the n_fft points of every ring, and its FFT input
        "ring_mask": n_fft * ntheta * (8 + 16),
        # the step-1 unit profiles (T, lambda^+, lambda^-) with ducc0's
        # output and its input alm, and the legs of both signs of M
        "profiles": (lmax + 1) * nc * 48 + n_north * nc * 112,
        "legs": ntheta * 2 * nc * 16 * 3,
        # per order, both signs of M: step-1 products and sources, the mixed
        # coefficients, the synthesis products; then per signed order the
        # analysis products, the outputs and the product scratch
        "step1": rows * 2 * nc * 16 * 8 + rows * 2 * nc * 16,
        "mix": rows * 2 * w * 16 * 3,
        "synthesis": ntheta * 2 * w * 16 * 5 + n_north * w * 16,
        "analysis": rows * w * 16 * 4,
        # the staging buffer while it is contracted, and the accumulators
        "stage": _stage_rows(lmax, n_out * nz, nc, nkeys)
        * _stage_row_bytes(n_out * nz, nc, nkeys)
        + (lmax + 1) * nkeys * 8,
        # the ring stage's workers
        "ring": _ring_scratch(n_fft, ntheta, width),
    }
    terms["total"] = sum(terms.values())
    return terms


def _stream_elements(spins: Sequence[int], lmax: int, lmax_grid: int) -> dict[int, int]:
    """``complex128`` elements of the reused buffer of each spin that
    streamed table blocks are built in (:meth:`_BatchedGL.tables`): the
    largest block's share of that spin (a block grows to
    :func:`_table_block_target` and overshoots by less than one order, and
    never exceeds the whole table; spin 0 and spin 2 split it 1 : 2) plus
    the front part :func:`~.grid.gl_legendre_table` addresses below the
    lowest order, ``(lmax + 1) npair`` per component."""
    units = {0: 1, 2: 2}
    total_units = sum(units[sp] for sp in spins)
    block = min(
        _table_block_target(spins, False)
        + _table_bytes(np.array([0]), lmax, lmax_grid, spins),
        _table_bytes(np.arange(lmax + 1), lmax, lmax_grid, spins),
    )
    npair = (gl_north_rings(lmax_grid) + 1) // 2
    return {
        sp: -(-block * units[sp] // total_units) // 16 + units[sp] * (lmax + 1) * npair
        for sp in spins
    }


def _plane_blocks(n_fft: int, ntheta: int, nplane: int) -> tuple[int, int]:
    """``(rings per block, blocks per plane)`` of :func:`_plane_ring_stage`:
    fixed blocks of about half of :data:`_FFT_BLOCK_BYTES` of one plane."""
    per = max(1, min(ntheta, _FFT_BLOCK_BYTES // (2 * n_fft * 16)))
    return per, -(-ntheta // per)


def _plane_ring_scratch(n_fft: int, ntheta: int, nplane: int) -> int:
    """Bytes charged for the workers of :func:`_plane_ring_stage` (one block
    each, :data:`_RING_WORKERS` at most), independently of ``nthreads``."""
    per, nper = _plane_blocks(n_fft, ntheta, nplane)
    return min(_RING_WORKERS, nplane * nper) * per * n_fft * 16


def _chunk_bytes_pol_direct(
    nc: int,
    n_fft: int,
    ntheta: int,
    lmax: int,
    lw: int,
    want_t: bool = True,
    nz: int = 3,
    n_out: int = 3,
    nkeys: int = 45,
) -> dict[str, int]:
    """Working set of one chunk of :func:`_batched_chunk_pol_direct`, in
    bytes (an upper bound of what it allocates at once, traced in
    ``tests/test_exact_pol_batched.py``)."""
    width = (int(want_t) + 2) * nz * nc
    n1 = min(2 * lw + 1, 2 * lmax + 1)  # orders of a column in step 1
    alm1 = n1 * (lmax + 1)  # their coefficients, degrees |M| .. lmax
    terms = {
        # the ring profiles, complex planes (width, ntheta, n_fft)
        "fields": n_fft * ntheta * width * 16,
        # the mask on the n_fft points of every ring: its FFT input, real
        # part and the repeated (real, imaginary) copy
        "ring_mask": n_fft * ntheta * (16 + 8 + 16),
        # the unit profiles of the columns (T, Q, U) and ducc0's output
        "profiles": 2 * 3 * ntheta * nc * 16 + 3 * (lmax + 1) * n1 * 16,
        # one column of step 1 and the synthesis: the legs and their
        # product, the step-1 coefficients, the mixed ones and scratch
        "column": 4 * ntheta * n1 * 16 + 14 * alm1 * 16,
        # the staging buffer (its front part and the rows) while contracted,
        # the accumulators, and a pair of scratch coefficients
        "stage": (lmax + 1) * n_out * nz * nc * 16
        + _stage_rows(lmax, n_out * nz, nc, nkeys)
        * (_stage_row_bytes(n_out * nz, nc, nkeys) + 2 * 16)
        + (lmax + 1) * nkeys * 8,
        # the ring stage's workers
        "ring": _plane_ring_scratch(n_fft, ntheta, width),
    }
    terms["total"] = sum(terms.values())
    return terms


def _batched_row_plan(
    mps: Sequence[int],
    lmax: int,
    lw: int,
    lmax_grid: int,
    max_memory_bytes: int,
    hold_table: bool | None = None,
    spins: Sequence[int] = (0,),
    layout: tuple[bool, int, int, int] | None = None,
) -> dict:
    r"""
    The deterministic memory plan of a batched exact GL row: which columns
    go together, and what is held.

    Everything the batched row allocates that scales with the problem is
    charged against ``max_memory_bytes``:

    ``modes``
        the mask's ring modes :math:`W_k(\theta)`, ``|k| <= lw``
        (:func:`~.grid.gl_ring_modes`), ``ntheta (2 lw + 1)`` complex, and
        their sums and differences over mirrored rings (as much again);
    ``table``
        the Legendre table :math:`\lambda_{LM}(\theta)` on the northern rings
        (:func:`~.grid.gl_legendre_table`): the whole table for
        ``M = 0 .. lmax`` if it fits in half the budget (``hold_table``;
        it is then kept for every row sharing the plan, built in place);
        otherwise streamed in blocks of about :func:`_table_block_target`
        bytes, re-generated for each chunk and each of the two passes over
        the orders (up to the degree each pass needs), all in one reused
        buffer per spin (:func:`_stream_elements`).  A polarised row holds
        spin-2 tables too (``spins``);
    per chunk (:func:`_chunk_bytes`)
        the ring profiles of its columns, ``N ntheta nc`` complex with ``N``
        the chunk's FFT length (:func:`_chunk_fft_size`) -- the dominant
        term -- the mask sampled on ``N`` points per ring, and per-order
        scratch.

    Nothing in the plan depends on the number of threads, so the chunks, and
    with them the summation order, are the same for any ``nthreads``: the
    rows are bit-identical across thread counts.

    A polarised row (:func:`_batched_chunk_pol_gl`) passes ``spins`` (the
    tables it holds: spin 2, and spin 0 if T is requested) and ``layout =
    (want_t, nz, n_out, nkeys)`` (T among the outputs, number of column
    types and of output fields, sums contracted), and its chunks are charged
    by :func:`_chunk_bytes_pol`.

    Columns are taken in the order given and a chunk grows while its
    working set fits in what the budget leaves after ``modes`` and
    ``table``.  A budget too small for a single column is warned about
    (once per budget) and one column per chunk is used.  Not charged:
    the fixed baseline -- the Python process, the mask as loaded, the
    healpy-ordered mask alm, and the row itself -- nor ducc0's and the BLAS
    library's internal buffers, nor freed memory the system allocator keeps.
    Measured with a 2 GiB budget, one row at ``lmax`` 500, 1000 and 2000
    (``Lw`` 384, macOS): numpy's allocations peak at 1.93, 1.93 and 1.91 GiB
    (``tracemalloc``) and the process's maximum resident size grows by 2.0,
    2.3 and 2.3 GiB over a 0.1 GiB start.  Polarised rows, a few chunks
    each: 1.85, 1.86 and 1.64 GiB traced, 2.3, 2.9 and 2.6 GiB resident;
    over a whole row macOS's allocator keeps more freed blocks (3.1 GiB at
    500, 3.3-4.4 GiB at 1000), which ``MallocLargeCache=0`` avoids (2.0 GiB)
    at the cost of faulting the per-order temporaries in afresh (93 s
    instead of 55 s at 500).

    Returns
    -------
    dict
        ``chunks`` (list of int arrays of ``m'``), ``hold_table``, and the
        byte terms ``modes``, ``table``, ``chunk`` (the largest chunk's
        :func:`_chunk_bytes` total) and ``total`` (their sum, the peak).
    """
    ntheta, _ = gl_shape(lmax_grid)
    mps = np.asarray(list(mps), dtype=np.int64)
    modes = 2 * ntheta * (2 * lw + 1) * 16
    full_table = _table_bytes(np.arange(lmax + 1), lmax, lmax_grid, spins)
    if hold_table is None:
        hold_table = full_table <= max_memory_bytes // 2
    # Built in place, with no second buffer: a held table block by block
    # into its own memory, a streamed one into one reused buffer per spin
    # (_stream_elements, charged whole although its front part is never
    # written).
    direct = layout is not None and len(layout) > 4 and bool(layout[4])
    if hold_table:
        table = full_table
    elif direct:  # the table-free polarised kernel holds no table
        table = 0
    else:
        table = 16 * sum(_stream_elements(spins, lmax, lmax_grid).values())
    left = max_memory_bytes - modes - table

    def chunk_bytes(nc: int, n_fft: int) -> int:
        if layout is None:
            return _chunk_bytes(nc, n_fft, ntheta, lmax)["total"]
        if direct:
            return _chunk_bytes_pol_direct(nc, n_fft, ntheta, lmax, lw, *layout[:4])[
                "total"
            ]
        return _chunk_bytes_pol(nc, n_fft, ntheta, lmax, *layout[:4])["total"]

    chunks, largest, start, floor = [], 0, 0, False
    while start < mps.size:
        stop = start + 1
        while stop < mps.size:
            cand = mps[start : stop + 1]
            n_fft = _chunk_fft_size(cand, lmax, lw)
            if chunk_bytes(cand.size, n_fft) > left:
                break
            stop += 1
        chunk = mps[start:stop]
        n_fft = _chunk_fft_size(chunk, lmax, lw)
        size = chunk_bytes(chunk.size, n_fft)
        floor |= size > left
        largest = max(largest, size)
        chunks.append(chunk)
        start = stop
    if floor:
        key = (
            "exact GL row",
            int(max_memory_bytes),
            int(lmax),
            int(lmax_grid),
            tuple(spins),
            layout,
        )
        if key not in _WARNED_MEMORY_FLOOR:
            _WARNED_MEMORY_FLOOR.add(key)
            warnings.warn(
                f"max_memory_gb = {max_memory_bytes / 1024**3:.3g} is below the "
                f"floor of the batched exact row at lmax_int={lmax}, grid "
                f"{lmax_grid} ({(modes + table + largest) / 1024**3:.3g} GiB for "
                "one column per chunk); using one column per chunk.",
                UserWarning,
                stacklevel=3,
            )
    return {
        "chunks": chunks,
        "hold_table": bool(hold_table),
        "modes": modes,
        "table": table,
        "chunk": largest,
        "total": modes + table + largest,
    }


def _truncate(entry, n: int):
    """The tables of one order (``(even, odd)``, or ``(spin-0 pair or
    None, spin-2 quadruple)``) cut to their first ``n`` degrees."""
    if entry is None:
        return None
    if isinstance(entry[0], np.ndarray):
        return tuple(t[: (n + 1 - k % 2) // 2] for k, t in enumerate(entry))
    return tuple(_truncate(part, n) for part in entry)


def _first_row(first: int, lo: int) -> int:
    """Index of the first row with degree ``>= lo`` in a parity table whose
    row 0 has degree ``first`` (rows two degrees apart)."""
    return max(0, (lo - first + 1) // 2)


def _window(ellp: int, width: int, lmax: int) -> tuple[int, int]:
    """Degrees ``|L - ellp| <= width`` within ``[0, lmax]``."""
    return max(0, ellp - width), min(lmax, ellp + width)


class _BatchedGL:
    r"""
    What a batched exact GL row needs besides the column list, built once
    and shared by every row of a matrix (:func:`exact_covariance` and each
    of its worker processes, :func:`exact_covariance_pol`): the ring
    geometry, the mask's ring modes and the Legendre tables of ``spins``
    (``(0,)`` for TT; spin 2, and spin 0 if T is requested, for a polarised
    row), held whole or streamed as :func:`_batched_row_plan` decides.
    """

    def __init__(
        self,
        mask_alm: np.ndarray,
        lw: int,
        lmax: int,
        lmax_grid: int,
        nthreads: int | None,
        max_memory_gb: float | None,
        spins: Sequence[int] = (0,),
    ):
        if max_memory_gb is None:
            max_memory_gb = DEFAULT_EXACT_MEMORY_GB
        if not max_memory_gb > 0:
            raise ValueError(f"max_memory_gb must be positive, got {max_memory_gb}")
        self.lw, self.lmax, self.lmax_grid = int(lw), int(lmax), int(lmax_grid)
        self.max_memory_bytes = int(max_memory_gb * 1024**3)
        self.nthreads = _gl_nthreads(nthreads, lmax_grid)
        geom = _gl_geometry(lmax_grid)
        self.ntheta = geom["ntheta"]
        self.theta = geom["theta"]
        self.wbar = geom["wbar"]
        self.n_north = gl_north_rings(lmax_grid)
        self.n_south = self.ntheta // 2
        # Ring order of the chunk arrays: the northern rings, then the southern
        # ones in mirror order (ring ntheta-1-j at position n_north+j), so that
        # a ring and its mirror image sit n_north apart.
        self.rings = np.concatenate(
            [
                np.arange(self.n_north),
                self.ntheta - 1 - np.arange(self.n_south),
            ]
        )
        self.modes = gl_ring_modes(mask_alm, lw, lmax_grid, nthreads=self.nthreads)
        # W_N + W_S and W_N - W_S on the mirrored ring pairs (the equator, if
        # any, is its own mirror: W_eq in both), for the step-1 legs.
        north, south = (
            self.modes[: self.n_north],
            self.modes[self.rings[self.n_north :]],
        )
        self.modes_sum = north.copy()
        self.modes_sum[: self.n_south] += south
        self.modes_diff = north.copy()
        self.modes_diff[: self.n_south] -= south
        self.spins = tuple(spins)
        if not self.spins or any(sp not in (0, 2) for sp in self.spins):
            raise ValueError(f"spins must be a subset of (0, 2), got {spins}")
        full_table = _table_bytes(np.arange(lmax + 1), lmax, lmax_grid, self.spins)
        self.hold_table = full_table <= self.max_memory_bytes // 2
        self._table: dict[int, tuple] = {}
        self._stream: dict[int, np.ndarray] = {}
        self._ring_mask: tuple[int, np.ndarray] | None = None
        self._plane_mask: tuple[int, np.ndarray] | None = None

    def plan(
        self, mps: Sequence[int], layout: tuple[bool, int, int, int] | None = None
    ) -> dict:
        return _batched_row_plan(
            mps,
            self.lmax,
            self.lw,
            self.lmax_grid,
            self.max_memory_bytes,
            hold_table=self.hold_table,
            spins=self.spins,
            layout=layout,
        )

    def _generate(self, block: list[int], hi: int, stream: bool) -> list:
        """The tables of the orders ``block`` for ``L <= hi``: ``(even, odd)``
        per order for ``spins == (0,)``, else ``(spin-0 pair or None, spin-2
        quadruple)`` (:func:`~.grid.gl_legendre_table`).  ``stream``: built
        in the reused buffers of :meth:`_stream_buffer`, overwritten by the
        next block."""
        ms = np.array(block)

        def table(spin: int) -> list:
            out = self._stream_buffer(spin) if stream else None
            return gl_legendre_table(
                ms, hi, self.lmax_grid, self.nthreads, spin=spin, out=out
            )

        if self.spins == (0,):
            return table(0)
        t0 = table(0) if 0 in self.spins else [None] * ms.size
        return list(zip(t0, table(2)))

    def _stream_buffer(self, spin: int) -> np.ndarray:
        """The buffer streamed blocks of ``spin`` are built in, allocated
        once: room for the largest block (:func:`_table_block_target`) plus
        the front part :func:`~.grid.gl_legendre_table` addresses but never
        writes, so only the block itself is ever committed."""
        if spin not in self._stream:
            size = _stream_elements(self.spins, self.lmax, self.lmax_grid)[spin]
            self._stream[spin] = np.empty(size, dtype=np.complex128)
        return self._stream[spin]

    def tables(self, orders: Sequence[int], hi: int | None = None):
        """``(M, tables)`` for ``M`` in ``orders`` (increasing) with ``M <=
        hi``, rows ``L = M .. hi`` (default ``lmax``), from the held table or
        generated block by block into one reused buffer: ``(even, odd)``
        parity tables of lambda_{LM} (:func:`~.grid.gl_legendre_table`) for
        TT, the pair ``(spin 0 or None, spin 2)`` of :meth:`_generate`
        otherwise.  A streamed block overwrites the previous one, so the
        tables of an order are only valid until the next is requested."""
        hi = self.lmax if hi is None else int(hi)
        orders = [m for m in orders if m <= hi]
        if self.hold_table:
            missing = [m for m in orders if m not in self._table]
            for block in self._blocks(missing, self.lmax):
                self._table.update(zip(block, self._generate(block, self.lmax, False)))
            for m in orders:
                yield m, _truncate(self._table[m], hi - m + 1)
            return
        pending: dict = {}
        for block in self._blocks(orders, hi):
            pending = dict(zip(block, self._generate(block, hi, True)))
            for m in block:
                yield m, pending[m]

    def _blocks(self, orders: list[int], hi: int) -> list[list[int]]:
        """``orders`` in consecutive groups of about
        :func:`_table_block_target` bytes of tables up to ``hi``."""
        per_ell = _table_bytes(np.array([hi]), hi, self.lmax_grid, self.spins)
        target = _table_block_target(self.spins, self.hold_table)
        blocks, block, size = [], [], 0
        for m in orders:
            block.append(m)
            size += (hi - m + 1) * per_ell
            if size >= target:
                blocks.append(block)
                block, size = [], 0
        if block:
            blocks.append(block)
        return blocks

    def plane_ring_mask(self, n_fft: int) -> np.ndarray:
        """:meth:`ring_mask` for the planes of
        :func:`_batched_chunk_pol_direct`: rings in natural order, the value
        repeated for the real and imaginary parts, ``(ntheta, 2 n_fft)``."""
        if self._plane_mask is None or self._plane_mask[0] != n_fft:
            spec = np.zeros((self.ntheta, n_fft), dtype=np.complex128)
            ks = np.arange(-self.lw, self.lw + 1)
            np.add.at(spec.T, ks % n_fft, self.modes.T)
            ducc0.fft.c2c(
                spec, axes=(1,), forward=False, out=spec, nthreads=self.nthreads
            )
            real = spec.real * self.wbar[:, None]
            del spec
            self._plane_mask = (n_fft, np.repeat(real, 2, axis=1))
        return self._plane_mask[1]

    def ring_mask(self, n_fft: int) -> np.ndarray:
        """The mask on ``n_fft`` equispaced points of every ring times the
        ring weight ``wbar(theta)``, ``(n_fft, ntheta)`` with the rings in the
        order of :attr:`rings`: ``wbar sum_k W_k(theta) e^{2 pi i k j /
        n_fft}`` (modes that coincide modulo ``n_fft`` are added: the samples
        are exact)."""
        if self._ring_mask is None or self._ring_mask[0] != n_fft:
            spec = np.zeros((n_fft, self.ntheta), dtype=np.complex128)
            ks = np.arange(-self.lw, self.lw + 1)
            np.add.at(spec, ks % n_fft, self.modes[self.rings].T)
            ducc0.fft.c2c(
                spec, axes=(0,), forward=False, out=spec, nthreads=self.nthreads
            )
            self._ring_mask = (n_fft, spec.real * self.wbar[None, self.rings])
        return self._ring_mask[1]


def _step1_profiles(ctx: _BatchedGL, ellp: int, mps: np.ndarray) -> np.ndarray:
    r"""
    ``wbar(theta) lambda_{l' m'}(theta)`` on the northern rings for the
    columns ``mps``, shape ``(n_north, nc)``: the ring profile of the weighted
    unit map of step 1 (:math:`\lambda_{l',-m} = (-1)^m \lambda_{l'm}`; on the
    mirror ring it is multiplied by ``(-1)^(l' + m')``).  One ``alm2leg`` call
    over the distinct ``|m'|``, each entry at ``L = l'`` (``lstride`` = number
    of orders, so nothing else is read).
    """
    am = np.abs(mps)
    orders = np.unique(am)
    alm = np.zeros((1, (ellp + 1) * orders.size), dtype=np.complex128)
    alm[0, ellp * orders.size + np.arange(orders.size)] = 1.0
    leg = ducc0.sht.alm2leg(
        alm=alm,
        lmax=ellp,
        theta=ctx.theta[: ctx.n_north],
        spin=0,
        mval=orders.astype(np.int64),
        mstart=np.arange(orders.size, dtype=np.int64),
        lstride=orders.size,
        nthreads=ctx.nthreads,
    )[0].real
    prof = leg[:, np.searchsorted(orders, am)]
    prof[:, (mps < 0) & (am % 2 == 1)] *= -1.0
    prof *= ctx.wbar[: ctx.n_north, None]
    return prof


def _step1_groups(
    ctx: _BatchedGL, mps: np.ndarray, n_pos: int, swap: bool = False
) -> list:
    r"""
    The two parity groups of a chunk's columns for :func:`_step1_legs`:
    ``(lo, hi, to_plus, to_minus, stride2)`` with ``mps[:n_pos]`` the
    columns whose unit profile is even about the equator (``l' + m'`` even
    for :math:`\lambda_{l'm'}` and :math:`\lambda^+_{l'm'}`), which take
    ``W_N + W_S`` into ``g_N + g_S``; the others take ``W_N - W_S``.
    ``swap`` exchanges the two, for :math:`\lambda^-_{l'm'}`, whose mirror
    parity is the opposite.  ``stride2``: the group's ``m'`` run in steps of
    2 (a run of consecutive ``m'``), so that its mask columns are a strided
    slice rather than a gather.
    """
    groups = []
    nc = mps.size
    sums = (ctx.modes_sum, ctx.modes_diff)
    for lo, hi, (to_plus, to_minus) in (
        (0, n_pos, sums[::-1] if swap else sums),
        (n_pos, nc, sums if swap else sums[::-1]),
    ):
        if hi > lo:
            steps = np.diff(mps[lo:hi])
            groups.append((lo, hi, to_plus, to_minus, bool(np.all(steps == 2))))
    return groups


def _step1_legs(
    plus: np.ndarray,
    minus: np.ndarray,
    m: int,
    mps: np.ndarray,
    prof: np.ndarray,
    groups: list,
    lw: int,
) -> None:
    """``plus = g_N + g_S`` (``(n_north, nc)``) and ``minus = g_N - g_S``
    (``(n_south, nc)``) for ``g = W_{m - m'} prof`` (zero for ``|m - m'| >
    lw``), the step-1 legs of order ``m`` (:func:`_batched_chunk_gl`), with
    ``prof`` the columns' weighted unit profiles on the northern rings and
    ``groups`` from :func:`_step1_groups`."""
    n_south = minus.shape[0]
    for lo, hi, to_plus, to_minus, stride2 in groups:
        k = m - mps[lo:hi]
        if stride2:  # k decreases by 2: |k| <= lw on a contiguous range
            a = lo + min(hi - lo, max(0, (int(k[0]) - lw + 1) // 2))
            b = lo + min(hi - lo, (int(k[0]) + lw) // 2 + 1)
            plus[:, lo:a] = 0.0
            plus[:, max(a, b) : hi] = 0.0
            minus[:, lo:a] = 0.0
            minus[:, max(a, b) : hi] = 0.0
            if a < b:
                first = m - int(mps[a]) + lw
                np.multiply(
                    to_plus[:, first::-2][:, : b - a],
                    prof[:, a:b],
                    out=plus[:, a:b],
                )
                np.multiply(
                    to_minus[:n_south, first::-2][:, : b - a],
                    prof[:n_south, a:b],
                    out=minus[:, a:b],
                )
            continue
        inside = np.abs(k) <= lw
        cols = np.where(inside, k + lw, 0)
        np.multiply(to_plus[:, cols], prof[:, lo:hi], out=plus[:, lo:hi])
        np.multiply(
            to_minus[:n_south, cols], prof[:n_south, lo:hi], out=minus[:, lo:hi]
        )
        if not inside.all():
            plus[:, lo:hi][:, ~inside] = 0.0
            minus[:, lo:hi][:, ~inside] = 0.0


def _signed(am: int, band: tuple[int, int]) -> list[int]:
    """The orders ``+am`` and ``-am`` (one of them for ``am = 0``) in ``band``."""
    return [m for m in ((am, -am) if am else (0,)) if band[0] <= m <= band[1]]


def _ring_stage(ctx: _BatchedGL, fields: np.ndarray) -> None:
    """
    The ring stage of step 3 in place on ``fields`` (``(n_fft, ntheta,
    width)``, rings in the order of :attr:`_BatchedGL.rings`): each ring's
    orders to its ``n_fft`` points, times the mask and the ring weight of
    the analysis, back to orders; then ``f_N + f_S`` into the northern ring
    and ``f_N - f_S`` into its mirror.  The northern rings are cut into
    fixed blocks of a few rings, sized so that the passes over a block run
    in cache, and whole blocks are handed to the workers, each FFT
    single-threaded: which rings share an FFT call does not depend on the
    number of workers, so neither does the result, to the bit.  (ducc0's
    batched FFT rounds a column differently depending on its place in the
    batch, and splitting the rings by worker first made the last bit depend
    on ``nthreads``.)
    """
    n_fft, _, width = fields.shape
    n_north, n_south = ctx.n_north, ctx.n_south
    ring_mask = ctx.ring_mask(n_fft)
    step, nblock = _ring_blocks(n_fft, n_north, width)

    def fft_mask_fft(block: np.ndarray, mask: np.ndarray) -> None:
        ducc0.fft.c2c(block, axes=(0,), forward=False, inorm=0, out=block, nthreads=1)
        block *= mask[:, :, None]
        ducc0.fft.c2c(block, axes=(0,), forward=True, inorm=2, out=block, nthreads=1)

    def blocks(b0: int, b1: int) -> None:
        for b in range(b0, b1):
            q0, q1 = b * step, min((b + 1) * step, n_north)
            north = fields[:, q0:q1]
            fft_mask_fft(north, ring_mask[:, q0:q1])
            p1 = min(q1, n_south)
            if q0 < p1:
                south = fields[:, n_north + q0 : n_north + p1]
                fft_mask_fft(south, ring_mask[:, n_north + q0 : n_north + p1])
                paired = north[:, : p1 - q0]
                # f_N - f_S, then f_N + f_S = 2 f_N - (f_N - f_S): no buffer
                np.subtract(paired, south, out=south)
                paired *= 2.0
                paired -= south

    workers = min(nblock, _RING_WORKERS, row_block_workers(fields.size, ctx.nthreads))
    run_row_blocks(blocks, nblock, workers)


def _orders_of(band: tuple[int, int]) -> range:
    """The distinct ``|M|`` of the orders ``M`` in ``band``."""
    lo, hi = band
    if lo <= 0 <= hi:
        return range(0, max(-lo, hi) + 1)
    return range(min(abs(lo), abs(hi)), max(abs(lo), abs(hi)) + 1)


def _batched_chunk_gl(
    ctx: _BatchedGL,
    cl: np.ndarray,
    ellp: int,
    mps: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    r"""
    ``sum_c weights[c] sum_M |c_c[L, M]|^2`` over the columns ``c`` of the
    chunk ``mps`` (orders ``m'`` of row ``ellp``), for ``L = 0 .. lmax``.

    The same linear algebra as :func:`_correlation_column_pair_gl`, with the
    spherical-harmonic transforms of every column of the chunk done together:

    1. Step 1, :math:`I = S^\dagger W Y_{\ell' m'}`.  The ring profile of
       mode ``M`` of :math:`W Y_{\ell' m'}` is
       :math:`W_{M - m'}(\theta) \lambda_{\ell' m'}(\theta)`, non-zero only
       for ``|M - m'| <= lw``, so the analysis is, per order,
       :math:`I_{LM} = \sum_\theta \lambda_{LM} \bar w W_{M-m'}
       \lambda_{\ell' m'}` -- one matrix product with the Legendre table
       for all columns (no FFT, no map).
    2. :math:`x_{LM} = C_L I_{LM}`.
    3. Synthesis per order, :math:`X_M(\theta) = \sum_L x_{LM}
       \lambda_{LM}(\theta)` (a matrix product), for the orders of the band
       ``|M - m'| <= lw``; FFT to ``N`` points per ring
       (:func:`_chunk_fft_size`), multiply by the mask and the ring weight
       :math:`\bar w`, FFT back; analysis per order, :math:`c_{LM} =
       \sum_\theta \lambda_{LM} f_M(\theta)`, for the orders
       ``|M - m'| <= 2 lw`` that can be non-zero, and ``|c|^2`` summed at
       once.

    Each matrix product runs on the northern rings only, the degrees split
    by the parity of ``L + M`` (:func:`~.grid.gl_legendre_table`): a sum over
    all rings of :math:`\lambda_{LM} g` is ``even @ (g_N + g_S) + odd @ (g_N
    - g_S)``, and a synthesis gives ``even x + odd x`` on a northern ring and
    ``even x - odd x`` on its mirror.  The rings of the chunk arrays are in
    the order of :attr:`_BatchedGL.rings` (mirror pairs ``n_north`` apart),
    the step-1 legs are built directly as ``g_N +- g_S`` from
    :math:`W_N \pm W_S` (columns grouped by the parity of ``l' + m'``, which
    decides the sign of the mirror profile), and the FFT workers leave
    ``f_N +- f_S`` behind, so no pass over the arrays is spent on the
    mirror sums.  The odd functions vanish on the equator, whose term is
    dropped from the odd products.  ``+M`` and ``-M`` share the table of
    ``|M|`` (their sign ``(-1)^M`` enters twice in steps 1-3 and not at all
    in ``|c|^2``).  Orders outside the bands are exact zeros of the
    per-column path (up to its rounding); skipping them removes the most
    work at ``lmax >> lw``.

    Only ``matmul``, element-wise products and FFTs along one axis are used,
    so the per-chunk arithmetic could move to another array namespace; the
    table and ring modes come from ducc0 (CPU).
    """
    lmax, lw = ctx.lmax, ctx.lw
    ntheta, n_north, n_south = ctx.ntheta, ctx.n_north, ctx.n_south
    # Columns grouped by the parity of l' + m' (the sign of the mirror
    # profile); the result is a sum over columns, so their order is free.
    even_first = np.argsort((ellp + mps) % 2, kind="stable")
    mps = np.asarray(mps)[even_first]
    weights = np.asarray(weights, dtype=float)[even_first]
    nc = mps.size
    n_pos = int(np.count_nonzero((ellp + mps) % 2 == 0))
    band_a = _band(mps, lw, lmax)
    band_c = _band(mps, 2 * lw, lmax)
    n_fft = _chunk_fft_size(mps, lmax, lw)
    prof = _step1_profiles(ctx, ellp, mps)  # (n_north, nc)

    fields = np.zeros((n_fft, ntheta, nc), dtype=np.complex128)
    plus = np.empty((n_north, 2, nc), dtype=np.complex128)
    minus = np.empty((n_south, 2, nc), dtype=np.complex128)
    groups = _step1_groups(ctx, mps, n_pos)

    # Steps 1-3 up to the synthesis, order by order, on the degrees
    # |L - l'| <= lw where the step-1 coefficients can be non-zero.
    lo, hi = _window(ellp, lw, lmax)
    for am, (even, odd) in ctx.tables(_orders_of(band_a), hi):
        ms = _signed(am, band_a)
        if not ms:
            continue
        nm = len(ms)
        ke, ko = _first_row(am, lo), _first_row(am + 1, lo)
        even, odd = even[ke:], odd[ko:]
        for i, m in enumerate(ms):
            _step1_legs(plus[:, i], minus[:, i], m, mps, prof, groups, lw)
        u = (even @ plus[:, :nm].reshape(n_north, -1).view(np.float64)).view(
            np.complex128
        )  # (n_even, nm nc)
        u *= cl[am + 2 * ke :: 2][: even.shape[0], None]
        x_even = (even.T @ u.view(np.float64)).view(np.complex128)
        x_even = x_even.reshape(n_north, nm, nc)
        if odd.shape[0] and n_south:
            odd = odd[:, :n_south]
            u = (odd @ minus[:, :nm].reshape(n_south, -1).view(np.float64)).view(
                np.complex128
            )
            u *= cl[am + 1 + 2 * ko :: 2][: odd.shape[0], None]
            x_odd = (odd.T @ u.view(np.float64)).view(np.complex128)
            x_odd = x_odd.reshape(n_south, nm, nc)
            for i, m in enumerate(ms):
                row = fields[m % n_fft]
                np.add(x_even[:n_south, i], x_odd[:, i], out=row[:n_south])
                row[n_south:n_north] = x_even[n_south:, i]
                np.subtract(x_even[:n_south, i], x_odd[:, i], out=row[n_north:])
        else:
            for i, m in enumerate(ms):
                row = fields[m % n_fft]
                row[:n_north] = x_even[:, i]
                row[n_north:] = x_even[:n_south, i]

    _ring_stage(ctx, fields)

    # Step-3 analysis and |c|^2, order by order, straight from the rows, on
    # the degrees |L - l'| <= 2 lw where c can be non-zero.
    total = np.zeros(lmax + 1)
    w_rep = np.repeat(weights, 2)
    lo, hi = _window(ellp, 2 * lw, lmax)
    for am, (even, odd) in ctx.tables(_orders_of(band_c), hi):
        ke, ko = _first_row(am, lo), _first_row(am + 1, lo)
        even, odd = even[ke:], odd[ko:]
        for m in _signed(am, band_c):
            row = fields[m % n_fft].view(np.float64)
            c = even @ row[:n_north]
            c *= c
            total[am + 2 * ke :: 2][: even.shape[0]] += c @ w_rep
            if odd.shape[0] and n_south:
                c = odd[:, :n_south] @ row[n_north:]
                c *= c
                total[am + 1 + 2 * ko :: 2][: odd.shape[0]] += c @ w_rep
    return total


def _exact_row_gl_batched(
    ctx: _BatchedGL,
    cl: np.ndarray,
    ellp: int,
    m_values: Sequence[int],
    use_symmetry: bool,
) -> np.ndarray:
    """``sum_{m'} weight(m') sum_m |c[l, m]|^2`` over ``m_values``, chunked
    by :meth:`_BatchedGL.plan` (the un-normalised row of :func:`_exact_row_gl`)."""
    total = np.zeros(ctx.lmax + 1)
    for mps in ctx.plan(m_values)["chunks"]:
        weights = np.where((mps > 0) & use_symmetry, 2.0, 1.0)
        total += _batched_chunk_gl(ctx, cl, ellp, mps, weights)
    return total


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
    batched: bool = True,
    max_memory_gb: float | None = None,
    batch: _BatchedGL | None = None,
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
        (:func:`_correlation_column_gl`) instead of the real-pair one (and
        not the batched one).  Same answer to ~1e-14 relative and ~2.4x
        slower than the real-pair columns; kept so that
        ``tests/test_exact_gl_speed.py`` can pin the equivalence.
    term_selection : float, optional
        Skip the orders ``m'`` that :func:`_kept_mprime` drops at this
        tolerance.  ``mask_alm`` must already be in the pole frame
        (:func:`_pole_frame_alm`; the public entry points do this once) for
        the selection to be effective; it is correct in any frame.
    batched : bool
        If True (default), the columns are computed in chunks by Legendre
        matrix products (:func:`_batched_chunk_gl`, module docstring "Cost
        of the GL column"); if False, one column at a time through ducc0
        transforms (:func:`_correlation_column_pair_gl`), the reference the
        batched path is tested against.  The two agree to a few 1e-15 of the
        largest entry.  A grid below :func:`~.grid.gl_minimal_lmax`
        (inexact on purpose) always takes the per-column path, whose
        aliasing it then reproduces.
    max_memory_gb : float, optional
        Memory budget of the batched path in GiB (default
        :data:`DEFAULT_EXACT_MEMORY_GB`), see :func:`_batched_row_plan`.
    batch : _BatchedGL, optional
        Row-independent state of the batched path built by the caller
        (:func:`exact_covariance`) to share the ring modes and the Legendre
        table across rows; built here when not given.  Its ``max_memory_gb``
        then applies and the argument above is ignored.

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

    ell = np.arange(lmax_i + 1)
    if use_symmetry:
        m_values = range(0, ellp + 1)
    else:
        m_values = range(-ellp, ellp + 1)
    keep = (
        None
        if term_selection is None
        else _kept_mprime(mask_alm, lw, ellp, term_selection, (0,), nthreads)
    )

    if batched and not reference and lmax_grid >= gl_minimal_lmax(lmax_i, lw):
        if batch is None:
            batch = _BatchedGL(mask_alm, lw, lmax_i, lmax_grid, nthreads, max_memory_gb)
        elif (batch.lw, batch.lmax, batch.lmax_grid) != (lw, lmax_i, lmax_grid):
            raise ValueError("batch was built for another lw, lmax_int or grid")
        kept = [mp for mp in m_values if keep is None or keep[mp + ellp]]
        total = _exact_row_gl_batched(batch, cl, ellp, kept, use_symmetry)
        row = 2.0 / ((2 * ell + 1) * (2 * ellp + 1)) * total
        return row[: lmax + 1]

    if mask_map is None:
        mask_map = gl_synthesis(mask_alm, lw, lmax_grid, nthreads=nthreads)
    total = np.zeros(lmax_i + 1)
    ell_alm = hp.Alm.getlm(lmax_i)[0]
    cl_alm = cl[ell_alm]

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
    max_memory_gb=None,
):
    """Per-process setup: keep the row arguments and build what the rows share
    (the batched path's ring modes and table policy, or the GL mask map)."""
    _WORKER.clear()
    lmax_i = int(lmax if lmax_int is None else lmax_int)
    batch = None
    mask_map = None
    if lmax_grid >= gl_minimal_lmax(lmax_i, lw):
        batch = _BatchedGL(mask_alm, lw, lmax_i, lmax_grid, nthreads, max_memory_gb)
    else:
        mask_map = gl_synthesis(mask_alm, lw, lmax_grid, nthreads=nthreads)
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
        mask_map=mask_map,
        batch=batch,
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
        batch=_WORKER["batch"],
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
    max_memory_gb=None,
):
    """
    ``[(ellp, row), ...]`` for every requested row, computed in ``nprocs``
    processes.  Each row is bit-identical to the serial one for the same
    ``nthreads`` (ducc0 splits a transform deterministically over rings, and
    every row is computed by a single call chain; ``max_memory_gb`` is per
    process, so every worker plans the same chunks as the serial loop).
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
            max_memory_gb,
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


# --------------------------------------------------------------------------- #
# Batched polarised columns
# --------------------------------------------------------------------------- #
#: Power of ``i`` in ``d_X`` of the rotated basis (T, E, B' = iB) of
#: :func:`_batched_chunk_pol_gl`: ``d = (1, 1, i)``.
_ROT = (0, 0, 1)


def _step1_profiles_spin2(
    ctx: _BatchedGL, ellp: int, mps: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    ``wbar lambda^+_{l'm'}`` and ``wbar lambda^-_{l'm'}`` on the northern
    rings for the columns ``mps``, each ``(n_north, nc)``: the ring profiles
    of the weighted (Q, iU) map of the E unit vector at ``(l', m')``
    (:func:`~.grid.gl_legendre_table`, spin 2).  ``alm2leg`` of an E unit
    vector gives ``(lambda^+, -i lambda^-)``; for ``m' < 0``,
    :math:`\lambda^\pm_{l',-m} = \pm(-1)^m \lambda^\pm_{l'm}`.  On the
    mirror ring the two are multiplied by ``(-1)^(l'+m')`` and
    ``-(-1)^(l'+m')``.
    """
    am = np.abs(mps)
    orders = np.unique(am)
    alm = np.zeros((2, (ellp + 1) * orders.size), dtype=np.complex128)
    alm[0, ellp * orders.size + np.arange(orders.size)] = 1.0
    leg = ducc0.sht.alm2leg(
        alm=alm,
        lmax=ellp,
        theta=ctx.theta[: ctx.n_north],
        spin=2,
        mval=orders.astype(np.int64),
        mstart=np.arange(orders.size, dtype=np.int64),
        lstride=orders.size,
        nthreads=ctx.nthreads,
    )
    idx = np.searchsorted(orders, am)
    plus = leg[0].real[:, idx]
    minus = -leg[1].imag[:, idx]
    del leg, alm
    neg, odd = mps < 0, am % 2 == 1
    plus[:, neg & odd] *= -1.0
    minus[:, neg & ~odd] *= -1.0
    plus *= ctx.wbar[: ctx.n_north, None]
    minus *= ctx.wbar[: ctx.n_north, None]
    return plus, minus


def _pol_layout(fields: Sequence[str], ellp: int) -> tuple[list[int], list[int]]:
    """``(cols, outs)`` of a polarised batched row: the column types (0, 1,
    2 = T, E, B; E and B only for ``l' >= 2``) and output fields among
    ``fields``."""
    cols = [z for z in range(3) if FIELDS[z] in fields and (z == 0 or ellp >= 2)]
    outs = [x for x in range(3) if FIELDS[x] in fields]
    return cols, outs


def _pol_terms(
    specs: Sequence[str], cols: Sequence[int], rot: Sequence[int] = _ROT
) -> dict:
    r"""
    The Wick contraction of every block in the rotated basis of
    :func:`_batched_chunk_pol_gl`: ``{(s1, s2): [(sign, kind, p, q), ...]}``
    with ``Cov^{s1 s2} = sum sign * {Re, Im} S(p, q)`` and ``S(p, q) =
    sum_{m, m'} w_{m'} G'_p conj(G'_q)`` over the rotated correlators
    ``G'_{(x, z)}`` (output ``x``, column ``z``).

    With ``R^{XZ} = conj(d_X) d_Z G'^{XZ}`` and ``d = (1, 1, i)``, the term
    ``R^{ab} conj(R^{ce})`` is ``i^k G'^{ab} conj(G'^{ce})`` with ``k = -[a
    = B] + [b = B] + [c = B] - [e = B]``, whose real part is ``Re S``,
    ``-Im S``, ``-Re S`` or ``Im S`` for ``k = 0, 1, 2, 3``.  ``(p, q)`` is
    put in increasing order (``Im S(q, p) = -Im S(p, q)``), and ``Im
    S(p, p) = 0`` is dropped, as are terms with a column absent from ``cols``
    (exactly zero).  ``rot = (0, 0, 0)`` gives the contraction in the
    original basis (:func:`_batched_chunk_pol_direct`): real parts only.
    """
    blocks = {}
    for s1 in specs:
        for s2 in specs:
            x, y = FIELDS.index(s1[0]), FIELDS.index(s1[1])
            z, w = FIELDS.index(s2[0]), FIELDS.index(s2[1])
            terms = []
            for a, b, c, e in ((x, w, y, z), (x, z, y, w)):
                if b not in cols or e not in cols:
                    continue
                k = (-rot[a] + rot[b] + rot[c] - rot[e]) % 4
                kind = "re" if k % 2 == 0 else "im"
                sign = (1, -1, -1, 1)[k]
                p, q = (a, b), (c, e)
                if p > q:
                    p, q = q, p
                    if kind == "im":
                        sign = -sign
                if kind == "im" and p == q:
                    continue
                terms.append((sign, kind, p, q))
            blocks[(s1, s2)] = terms
    return blocks


def _pol_mixing(cmat: np.ndarray, cols: Sequence[int], want_t: bool) -> dict:
    r"""
    Step 2 in the rotated basis: ``{(o, z): [(coef, source), ...]}`` with
    ``v'^o = sum coef_L * source`` for the output component ``o`` (0, 1, 2 =
    T, E, B' = iB) of the column type ``z``, ``source`` among ``"I"`` (the
    step-1 T coefficients of a T column) and ``"uE"``, ``"uB"`` (the E and
    B' coefficients of an E unit vector).  ``v' = D C D^{-1} u'`` with
    ``D = diag(1, 1, i)``: a T column gives ``d_o C^{oT} I``; an E column
    ``d_o (C^{oE} u_E - i C^{oB} u_B)``; a B column is ``i`` times a B'
    unit vector, whose step-1 coefficients are those of the E unit vector
    swapped (``K'_2`` is symmetric in E and B'), so it gives ``d_o (C^{oE}
    u_B - i C^{oB} u_E)``, the ``i`` restored in the contraction
    (:func:`_pol_terms`).  Coefficients that vanish for every ``L`` are
    dropped; the T component only if ``want_t``.
    """
    rot = (1.0, 1.0, 1.0j)
    mix = {}
    for z in cols:
        for o in (0, 1, 2) if want_t else (1, 2):
            if z == 0:
                terms = [(rot[o] * cmat[o, 0], "I")]
            else:
                first, second = ("uE", "uB") if z == 1 else ("uB", "uE")
                terms = [
                    (rot[o] * cmat[o, 1], first),
                    (-1.0j * rot[o] * cmat[o, 2], second),
                ]
            kept = []
            for coef, source in terms:
                if np.any(coef):
                    if np.iscomplexobj(coef) and not np.any(coef.imag):
                        coef = np.ascontiguousarray(coef.real)
                    kept.append((coef, source))
            mix[(o, z)] = kept
    return mix


def _windowed(t0, t2, am: int, lo: int):
    """The spin-0 pair (or None) and spin-2 quadruple of order ``am`` cut
    to the degrees ``>= lo``, and the first degree of each parity."""
    ke, ko = _first_row(am, lo), _first_row(am + 1, lo)
    if t0 is not None:
        t0 = (t0[0][ke:], t0[1][ko:])
    t2 = (t2[0][ke:], t2[1][ko:], t2[2][ke:], t2[3][ko:])
    return t0, t2, (am + 2 * ke, am + 1 + 2 * ko)


def _rgemm(table: np.ndarray, x: np.ndarray) -> np.ndarray:
    """``table @ x`` for a real table and a complex ``x`` with contiguous
    rows, as one real matrix product (the real and imaginary parts ride
    along as columns)."""
    return (table @ x.view(np.float64)).view(np.complex128)


def _batched_chunk_pol_gl(
    ctx: _BatchedGL,
    mix: dict,
    ellp: int,
    mps: np.ndarray,
    weights: np.ndarray,
    cols: Sequence[int],
    outs: Sequence[int],
    keys: Sequence[tuple],
) -> dict:
    r"""
    ``S(p, q)`` (:func:`_pol_terms`) summed over the columns of the chunk
    ``mps`` of row ``ellp`` with ``weights``, for ``(kind, p, q)`` in
    ``keys``: ``{key: array over L = 0 .. lmax}``.

    The polarised counterpart of :func:`_batched_chunk_gl`, the same
    arithmetic as :func:`_correlation_columns_gl` with the transforms of all
    columns of the chunk done together.  It works in the rotated basis
    ``(T, E, B' = iB)`` on the coefficients and ``(T, Q, U'' = iU)`` on the
    rings, in which the spin-2 synthesis and analysis of order ``M >= 0``
    are the real symmetric blocks

    .. math::

        \begin{pmatrix} Q \\ U'' \end{pmatrix}_M = \sum_L
        \begin{pmatrix} \lambda^+ & \lambda^- \\ \lambda^- & \lambda^+
        \end{pmatrix}_{LM} \begin{pmatrix} E \\ B' \end{pmatrix}_{LM},
        \qquad
        \begin{pmatrix} E \\ B' \end{pmatrix}_{LM} = \sum_\theta \bar w
        \begin{pmatrix} \lambda^+ & \lambda^- \\ \lambda^- & \lambda^+
        \end{pmatrix}_{LM} \begin{pmatrix} Q \\ U'' \end{pmatrix}_M

    (ducc0's convention, :func:`~.grid.gl_legendre_table`); at ``-M`` the
    table of ``|M|`` is used with :math:`\lambda^-` negated (the common
    :math:`(-1)^M` cancels as in the TT kernel).  So every Legendre sum is a
    real matrix product on complex data, split by the mirror parity of the
    functions: :math:`\lambda^+` rows with ``L - M`` even and
    :math:`\lambda^-` rows with ``L - M`` odd are even about the equator
    (paired with ``f_N + f_S``), the other two odd (paired with ``f_N -
    f_S``, equator dropped).  Per order:

    1. Step 1: the T columns as in the TT kernel; for E, one set of legs of
       the E unit vector, ``wbar W_{M-m'} lambda^\pm_{l'm'}``
       (:func:`_step1_profiles_spin2`), four products giving its E and B'
       coefficients.  A B' unit vector has the same coefficients swapped,
       so a B column costs no step-1 work.
    2. The ``C`` mixing between fields per ``L`` (:func:`_pol_mixing`).
    3. Synthesis of T (spin 0) and (Q, U'') for every column type; the ring
       stage of the TT kernel (:func:`_ring_stage`) on all of them at once;
       analysis of T, E and B' per order, and the products ``G'_p
       conj(G'_q)`` summed over the columns with their weights, per
       ``L`` -- the six-spectrum contraction instead of ``|c|^2``.

    Orders are restricted to the bands of the TT kernel.
    """
    lmax, lw = ctx.lmax, ctx.lw
    ntheta, n_north, n_south = ctx.ntheta, ctx.n_north, ctx.n_south
    even_first = np.argsort((ellp + mps) % 2, kind="stable")
    mps = np.asarray(mps)[even_first]
    weights = np.asarray(weights, dtype=float)[even_first]
    nc = mps.size
    n_pos = int(np.count_nonzero((ellp + mps) % 2 == 0))
    band_a = _band(mps, lw, lmax)
    band_c = _band(mps, 2 * lw, lmax)
    n_fft = _chunk_fft_size(mps, lmax, lw)
    want_t = 0 in outs
    t_col = 0 in cols
    p_col = 1 in cols or 2 in cols
    nz = len(cols)
    w = nz * nc
    zpos = {z: k for k, z in enumerate(cols)}
    q0 = w if want_t else 0  # ring columns: [T][Q][U''], w each
    width = q0 + 2 * w
    fields = np.zeros((n_fft, ntheta, width), dtype=np.complex128)

    if t_col:
        prof = _step1_profiles(ctx, ellp, mps)
        groups = _step1_groups(ctx, mps, n_pos)
        plus = np.empty((n_north, 2, nc), dtype=np.complex128)
        minus = np.empty((n_south, 2, nc), dtype=np.complex128)
    if p_col:
        prof_p, prof_m = _step1_profiles_spin2(ctx, ellp, mps)
        groups_p = _step1_groups(ctx, mps, n_pos)
        groups_m = _step1_groups(ctx, mps, n_pos, swap=True)
        # [ring, sign of M, (lambda^+ leg, lambda^- leg), column]
        leg_s = np.empty((n_north, 2, 2, nc), dtype=np.complex128)
        leg_d = np.empty((n_south, 2, 2, nc), dtype=np.complex128)

    def combine(op, a, b, out=None):
        """``a + b`` (``op`` = np.add) or ``a - b`` (np.subtract, -M)."""
        return op(a, b) if out is None else op(a, b, out=out)

    def mixed(dst, lstart, n, src):
        """``dst[:, :, k] = v'`` of every (output, column) for one parity,
        rows of degree ``lstart, lstart + 2, ...``."""
        tmp = None
        for (o, z), terms in mix.items():
            k = zpos[z]
            if o == 0:
                target = dst["T"][:, :, k]
            else:
                target = dst["P"][:, :, o - 1, k]
            if not terms:
                target[...] = 0.0
                continue
            for j, (coef, source) in enumerate(terms):
                c = coef[lstart::2][:n, None, None]
                if j == 0:
                    np.multiply(src[source], c, out=target)
                else:
                    if tmp is None:
                        tmp = np.empty(src[source].shape, dtype=np.complex128)
                    np.multiply(src[source], c, out=tmp)
                    target += tmp

    # Steps 1-3 up to the synthesis, order by order (both signs together),
    # on the degrees |L - l'| <= lw where the step-1 coefficients can be
    # non-zero.
    lo, hi = _window(ellp, lw, lmax)
    for am, (t0, t2) in ctx.tables(_orders_of(band_a), hi):
        ms = _signed(am, band_a)
        if not ms:
            continue
        nm = len(ms)
        ops = [np.add if m >= 0 else np.subtract for m in ms]
        t0, t2, lstarts = _windowed(t0, t2, am, lo)
        pe, po, me, mo = t2
        n_e, n_o = pe.shape[0], po.shape[0]
        me, po = me[:, :n_south], po[:, :n_south]
        src = [{}, {}]  # per parity: step-1 coefficients by source
        if t_col:
            even, odd = t0
            for i, m in enumerate(ms):
                _step1_legs(plus[:, i], minus[:, i], m, mps, prof, groups, lw)
            src[0]["I"] = _rgemm(even, plus[:, :nm].reshape(n_north, -1))
            src[0]["I"] = src[0]["I"].reshape(n_e, nm, nc)
            if n_o:
                src[1]["I"] = _rgemm(
                    odd[:, :n_south], minus[:, :nm].reshape(n_south, -1)
                )
                src[1]["I"] = src[1]["I"].reshape(n_o, nm, nc)
        if p_col:
            for i, m in enumerate(ms):
                _step1_legs(
                    leg_s[:, i, 0], leg_d[:, i, 0], m, mps, prof_p, groups_p, lw
                )
                _step1_legs(
                    leg_s[:, i, 1], leg_d[:, i, 1], m, mps, prof_m, groups_m, lw
                )
            sums = leg_s[:, :nm].reshape(n_north, -1)
            diffs = leg_d[:, :nm].reshape(n_south, -1)
            # E = sum lambda^+ g^+ + lambda^- g^-, B' = sum lambda^- g^+ +
            # lambda^+ g^-, each function with the legs of its parity.
            for parity, (sym, anti, x_sym, x_anti) in enumerate(
                ((pe, me, sums, diffs), (mo, po, sums, diffs))
            ):
                n = (n_e, n_o)[parity]
                if not n:
                    continue
                a = _rgemm(sym, x_sym).reshape(n, nm, 2, nc)
                b = _rgemm(anti, x_anti).reshape(n, nm, 2, nc)
                if parity == 0:  # lambda^+ even: a = +, b = -
                    plus_part, minus_part = a, b
                else:  # lambda^+ odd: b = +, a = -
                    plus_part, minus_part = b, a
                u_e = np.empty((n, nm, nc), dtype=np.complex128)
                u_b = np.empty((n, nm, nc), dtype=np.complex128)
                for i in range(nm):
                    combine(ops[i], plus_part[:, i, 0], minus_part[:, i, 1], u_e[:, i])
                    combine(ops[i], plus_part[:, i, 1], minus_part[:, i, 0], u_b[:, i])
                del a, b, plus_part, minus_part
                src[parity]["uE"], src[parity]["uB"] = u_e, u_b

        # Step 2: v' = C' u' per parity, (n, nm, nz, nc) for T and
        # (n, nm, 2, nz, nc) for (E, B').
        v = [{}, {}]
        for parity in (0, 1):  # a parity may have no degree in the window
            n = (n_e, n_o)[parity]
            if want_t:
                v[parity]["T"] = np.empty((n, nm, nz, nc), dtype=np.complex128)
            v[parity]["P"] = np.empty((n, nm, 2, nz, nc), dtype=np.complex128)
            if n:
                mixed(v[parity], lstarts[parity], n, src[parity])
        del src

        # Step 3, synthesis: T as in the TT kernel, (Q, U'') from the four
        # products Pe^T, Mo^T (all northern rings) and Po^T, Me^T (odd
        # about the equator: rings with a mirror only).
        rows = [fields[m % n_fft] for m in ms]
        if want_t:
            x_e = _rgemm(even.T, v[0]["T"].reshape(n_e, nm * w))
            x_e = x_e.reshape(n_north, nm, w)
            if n_o:
                x_o = _rgemm(odd[:, :n_south].T, v[1]["T"].reshape(n_o, nm * w))
                x_o = x_o.reshape(n_south, nm, w)
            for i, row in enumerate(rows):
                if n_o:
                    np.add(x_e[:n_south, i], x_o[:, i], out=row[:n_south, :w])
                    row[n_south:n_north, :w] = x_e[n_south:, i]
                    np.subtract(x_e[:n_south, i], x_o[:, i], out=row[n_north:, :w])
                else:
                    row[:n_north, :w] = x_e[:, i]
                    row[n_north:, :w] = x_e[:n_south, i]
            del x_e
            if n_o:
                del x_o
        vp_e = v[0]["P"].reshape(n_e, nm * 2 * w)
        pe_t = _rgemm(pe.T, vp_e).reshape(n_north, nm, 2, w)
        me_t = _rgemm(me.T, vp_e).reshape(n_south, nm, 2, w)
        if n_o:
            vp_o = v[1]["P"].reshape(n_o, nm * 2 * w)
            mo_t = _rgemm(mo.T, vp_o).reshape(n_north, nm, 2, w)
            po_t = _rgemm(po.T, vp_o).reshape(n_south, nm, 2, w)
        del v
        anti = np.empty((n_south, w), dtype=np.complex128)
        for i, row in enumerate(rows):
            for c in (0, 1):  # Q, U''
                cs = slice(q0 + c * w, q0 + (c + 1) * w)
                north = row[:n_north, cs]
                if n_o:
                    combine(ops[i], pe_t[:, i, c], mo_t[:, i, 1 - c], north)
                    combine(ops[i], po_t[:, i, c], me_t[:, i, 1 - c], anti)
                else:
                    north[...] = pe_t[:, i, c]
                    if ops[i] is np.add:
                        anti[...] = me_t[:, i, 1 - c]
                    else:
                        np.negative(me_t[:, i, 1 - c], out=anti)
                np.subtract(north[:n_south], anti, out=row[n_north:, cs])
                north[:n_south] += anti
        del pe_t, me_t, anti
        if n_o:
            del mo_t, po_t

    _ring_stage(ctx, fields)

    # Step-3 analysis order by order into a staging buffer of (L, M) rows,
    # contracted over the columns whenever it is full: per row, the Gram
    # matrix G[p, q] = sum_c w_c G'_p conj(G'_q) of the n_out x nz outputs,
    # one batched matrix product for all the pairs of the six-spectrum
    # contraction (which would otherwise be dozens of small calls per order).
    n_out = len(outs)
    npair = n_out * nz
    opos = {x: k for k, x in enumerate(outs)}
    sel = np.array(
        [
            2 * ((opos[a] * nz + zpos[b]) * npair + opos[c] * nz + zpos[e])
            + (kind == "im")
            for kind, (a, b), (c, e) in keys
        ],
        dtype=np.int64,
    )
    acc = np.zeros((lmax + 1, len(keys)))
    n_stage = _stage_rows(lmax, npair, nc, len(keys))
    stage = np.empty((n_stage, n_out, nz, nc), dtype=np.complex128)
    segments: list[tuple[int, int, int]] = []  # (first row, rows, first L)

    def contract(r: int) -> None:
        """Add the staged rows ``:r`` to ``acc``, per ``L``."""
        y = stage[:r].reshape(r, npair, nc)
        gram = np.matmul(y * weights, np.conj(y).transpose(0, 2, 1))
        vals = gram.reshape(r, npair * npair).view(np.float64)[:, sel]
        del gram
        for r0, n, ell0 in segments:
            acc[ell0 : ell0 + 2 * n - 1 : 2] += vals[r0 : r0 + n]
        segments.clear()

    filled = 0
    lo, hi = _window(ellp, 2 * lw, lmax)
    for am, (t0, t2) in ctx.tables(_orders_of(band_c), hi):
        t0, t2, lstarts = _windowed(t0, t2, am, lo)
        pe, po, me, mo = t2
        n_e, n_o = pe.shape[0], po.shape[0]
        me, po = me[:, :n_south], po[:, :n_south]
        for m in _signed(am, band_c):
            if filled + n_e + n_o > n_stage:
                contract(filled)
                filled = 0
            op = np.add if m >= 0 else np.subtract
            rr = fields[m % n_fft].view(np.float64)
            north, south = rr[:n_north], rr[n_north:]
            qu = slice(2 * q0, 2 * width)
            for parity, (sym, anti, x_sym, x_anti) in enumerate(
                ((pe, me, north, south), (mo, po, north, south))
            ):
                n = (n_e, n_o)[parity]
                if not n:
                    continue
                rows = stage[filled : filled + n]
                segments.append((filled, n, lstarts[parity]))
                if want_t:
                    even, odd = t0
                    table, x = (
                        (even, north) if parity == 0 else (odd[:, :n_south], south)
                    )
                    out_t = rows[:, opos[0]].reshape(n, w).view(np.float64)
                    np.matmul(table, x[:, : 2 * w], out=out_t)
                a = _rgemm(sym, x_sym[:, qu])  # (n, 2w): [from Q, from U'']
                b = _rgemm(anti, x_anti[:, qu])
                plus_part, minus_part = (a, b) if parity == 0 else (b, a)
                if 1 in outs:  # E = lambda^+ Q + lambda^- U''
                    out_e = rows[:, opos[1]].reshape(n, w)
                    op(plus_part[:, :w], minus_part[:, w:], out=out_e)
                if 2 in outs:  # B' = lambda^- Q + lambda^+ U''
                    out_b = rows[:, opos[2]].reshape(n, w)
                    op(plus_part[:, w:], minus_part[:, :w], out=out_b)
                del a, b, plus_part, minus_part
                filled += n
    if filled:
        contract(filled)
    return {key: acc[:, j] for j, key in enumerate(keys)}


#: Target bytes of the staging buffer of :func:`_batched_chunk_pol_gl`.
_STAGE_BYTES = 16 * 2**20


def _stage_rows(lmax: int, npair: int, nc: int, nkeys: int) -> int:
    """Rows of the staging buffer of :func:`_batched_chunk_pol_gl`: about
    :data:`_STAGE_BYTES` but at most 16 orders' worth of rows, and at least
    the ``lmax + 1`` rows of one order."""
    per_row = _stage_row_bytes(npair, nc, nkeys)
    return max(lmax + 1, min(_STAGE_BYTES // per_row, 16 * (lmax + 1)))


def _stage_row_bytes(npair: int, nc: int, nkeys: int) -> int:
    """Bytes per staged row while it is contracted: the outputs, their
    weighted and conjugated copies, the Gram matrix and the selected
    values."""
    return 3 * npair * nc * 16 + npair * npair * 16 + nkeys * 8


# --------------------------------------------------------------------------- #
# Batched polarised columns without tables (streamed regime)
# --------------------------------------------------------------------------- #
def _unit_legs_all(
    ctx: _BatchedGL, ellp: int, mps: np.ndarray, spin: int
) -> np.ndarray:
    r"""
    ``wbar`` times the ring profiles on every ring (natural order) of the
    unit vector at ``(l', m')`` for each column: ``(1, ntheta, nc)``
    (:math:`\lambda_{l'm'}`, spin 0) or ``(2, ntheta, nc)`` (the (Q, U)
    profile of the E unit vector, spin 2), complex.  ``alm2leg`` at
    ``|m'|``; for ``m' < 0`` the complex-linear rule ``leg(l', -|m|) =
    (-1)^m conj(leg(l', |m|))`` (:func:`~.grid.full_from_pair`).
    """
    am = np.abs(mps)
    orders = np.unique(am)
    ncomp = 1 if spin == 0 else 2
    alm = np.zeros((ncomp, (ellp + 1) * orders.size), dtype=np.complex128)
    alm[0, ellp * orders.size + np.arange(orders.size)] = 1.0
    leg = ducc0.sht.alm2leg(
        alm=alm,
        lmax=ellp,
        theta=ctx.theta,
        spin=spin,
        mval=orders.astype(np.int64),
        mstart=np.arange(orders.size, dtype=np.int64),
        lstride=orders.size,
        nthreads=ctx.nthreads,
    )
    prof = leg[:, :, np.searchsorted(orders, am)]
    neg = mps < 0
    if neg.any():
        sign = np.where(am[neg] % 2 == 1, -1.0, 1.0)
        prof[:, :, neg] = sign * np.conj(prof[:, :, neg])
    prof *= ctx.wbar[None, :, None]
    return prof


def _order_runs(orders: np.ndarray, n_fft: int) -> list[tuple[int, int, int]]:
    """``(first index, last index + 1, first slot)`` of the runs of
    consecutive ``orders`` whose FFT slots ``M % n_fft`` are consecutive."""
    slots = orders % n_fft
    cut = np.flatnonzero(np.diff(slots) != 1) + 1
    bounds = np.concatenate([[0], cut, [orders.size]])
    return [(int(a), int(b), int(slots[a])) for a, b in zip(bounds[:-1], bounds[1:])]


def _signed_groups(orders: np.ndarray) -> list[tuple[np.ndarray, bool]]:
    """``orders`` (increasing, signed) as the ``M >= 0`` group and the ``M
    < 0`` group, each ``(orders, negative)``: ducc0 takes ``|M|``, unique
    per call."""
    out = []
    pos, neg = orders[orders >= 0], orders[orders < 0]
    if pos.size:
        out.append((pos, False))
    if neg.size:
        out.append((neg, True))
    return out


def _alm_layout(orders: np.ndarray, hi: int) -> tuple[np.ndarray, int]:
    """``mstart`` and size of a compact alm array holding, for each (signed)
    order, degrees ``|M| .. hi`` (ducc0 addresses ``mstart + L``)."""
    n = hi - np.abs(orders) + 1
    first = np.concatenate([[0], np.cumsum(n)[:-1]])
    return (first - np.abs(orders)).astype(np.int64), int(n.sum())


def _batched_chunk_pol_direct(
    ctx: _BatchedGL,
    cmat: np.ndarray,
    ellp: int,
    mps: np.ndarray,
    weights: np.ndarray,
    cols: Sequence[int],
    outs: Sequence[int],
    keys: Sequence[tuple],
) -> dict:
    r"""
    :func:`_batched_chunk_pol_gl` without Legendre tables: every Legendre
    sum is a ducc0 transform (``leg2alm`` / ``alm2leg``, which evaluate the
    same functions as the per-column transforms, by their own recursion) on
    the orders of the column's band only, and on the degrees up to the top
    of the window ``|L - l'| <= lw`` (step 1, synthesis) or ``<= 2 lw``
    (analysis).  The chunk's columns still share the ring stage (one FFT
    and mask product per ring for all of them) and the contraction.

    Used when the tables would have to be streamed: a polarised ``m'``
    carries up to nine ring profiles, so a chunk holds few columns, and a
    table regenerated for each chunk and shared by only a few columns costs
    more than transforming them directly.
    It works in the original basis (T, E, B) on the coefficients and (T,
    Q, U) on the rings, the contraction taking real parts only.  ducc0 takes
    orders ``M >= 0``: at ``-M`` the spin-0 functions carry ``(-1)^M``,
    dropped as in the table kernel (it enters twice in steps 1-3 and not in
    the products of pass 2), and the spin-2 ones swap the sign of
    :math:`\lambda^-`, i.e. synthesis at ``|M|`` of ``(E, -B)`` gives ``(Q,
    -U)`` and analysis at ``|M|`` of ``(Q, -U)`` gives ``(E, -B)``
    (:func:`~.grid.banded_integrals_gl`).  The ring profiles live on planes
    ``(component, ring, order slot)``, rings in natural order, the layout
    ducc0's legs have.
    """
    lmax, lw, ntheta = ctx.lmax, ctx.lw, ctx.ntheta
    mps = np.asarray(mps)
    weights = np.asarray(weights, dtype=float)
    nc = mps.size
    n_fft = _chunk_fft_size(mps, lmax, lw)
    want_t = 0 in outs
    t_col = 0 in cols
    p_col = 1 in cols or 2 in cols
    nz = len(cols)
    zpos = {z: k for k, z in enumerate(cols)}
    ncp = int(want_t) + 2  # planes per column: [T], Q, U
    qoff = int(want_t)
    fields = np.zeros((nz * nc * ncp, ntheta, n_fft), dtype=np.complex128)
    nt = ctx.nthreads
    theta = ctx.theta

    def plane(k: int, c: int) -> int:
        return (k * nc + c) * ncp

    lo1, hi1 = _window(ellp, lw, lmax)
    prof0 = _unit_legs_all(ctx, ellp, mps, 0) if t_col else None
    prof2 = _unit_legs_all(ctx, ellp, mps, 2) if p_col else None

    # Steps 1-3 up to the synthesis, column by column.
    for c, mp in enumerate(mps):
        mp = int(mp)
        orders = np.arange(max(-hi1, mp - lw), min(hi1, mp + lw) + 1)
        if not orders.size:
            continue
        wcols = ctx.modes[:, orders[0] - mp + lw : orders[-1] - mp + lw + 1]
        for grp, negative in _signed_groups(orders):
            i0 = int(np.searchsorted(orders, grp[0]))
            w = wcols[:, i0 : i0 + grp.size]
            mval = np.abs(grp).astype(np.int64)
            mstart, size = _alm_layout(grp, hi1)
            src = {}
            if t_col:
                legs = (w * prof0[0, :, c][:, None])[None]
                src["T"] = ducc0.sht.leg2alm(
                    leg=legs,
                    lmax=hi1,
                    theta=theta,
                    spin=0,
                    mval=mval,
                    mstart=mstart,
                    nthreads=nt,
                    alm=np.empty((1, size), np.complex128),
                )[0]
                del legs
            if p_col:
                legs = w[None] * prof2[:, :, c][:, :, None]
                if negative:
                    legs[1] *= -1.0
                ueb = ducc0.sht.leg2alm(
                    leg=legs,
                    lmax=hi1,
                    theta=theta,
                    spin=2,
                    mval=mval,
                    mstart=mstart,
                    nthreads=nt,
                    alm=np.empty((2, size), np.complex128),
                )
                del legs
                if negative:
                    ueb[1] *= -1.0
                src["E"], src["B"] = ueb[0], ueb[1]
            ell = np.concatenate([np.arange(abs(int(m)), hi1 + 1) for m in grp])
            runs = _order_runs(grp, n_fft)
            for z in cols:
                k = zpos[z]
                if z == 0:
                    u = {"T": src["T"]}
                elif z == 1:
                    u = {"E": src["E"], "B": src["B"]}
                else:  # a B unit vector: (u_E, u_B) = (-u_B, u_E) of an E one
                    u = {"E": -src["B"], "B": src["E"]}
                v = {}
                for x in ((0, 1, 2) if want_t else (1, 2)):
                    acc = None
                    for zz, name in enumerate("TEB"):
                        if name not in u or not np.any(cmat[x, zz]):
                            continue
                        term = cmat[x, zz][ell] * u[name]
                        acc = term if acc is None else acc + term
                    v[x] = acc if acc is not None else np.zeros(size, np.complex128)
                p0 = plane(k, c)
                for a, b, slot in runs:
                    mv, ms_ = mval[a:b], mstart[a:b]
                    if want_t:
                        ducc0.sht.alm2leg(
                            alm=v[0][None],
                            lmax=hi1,
                            theta=theta,
                            spin=0,
                            mval=mv,
                            mstart=ms_,
                            nthreads=nt,
                            leg=fields[p0 : p0 + 1, :, slot : slot + b - a],
                        )
                    eb = np.stack([v[1], -v[2] if negative else v[2]])
                    qu = fields[p0 + qoff : p0 + qoff + 2, :, slot : slot + b - a]
                    ducc0.sht.alm2leg(
                        alm=eb,
                        lmax=hi1,
                        theta=theta,
                        spin=2,
                        mval=mv,
                        mstart=ms_,
                        nthreads=nt,
                        leg=qu,
                    )
                    if negative:
                        qu[1] *= -1.0
                    del eb
                del u, v
            del src

    _plane_ring_stage(ctx, fields)

    # Step-3 analysis into a staging buffer of (M, L) rows, contracted over
    # the columns whenever it is full (as in the table kernel).
    lo2, hi2 = _window(ellp, 2 * lw, lmax)
    band_c = _band(mps, 2 * lw, lmax)
    all_orders = np.arange(max(band_c[0], -hi2), min(band_c[1], hi2) + 1)
    n_out = len(outs)
    npair = n_out * nz
    opos = {x: j for j, x in enumerate(outs)}
    sel = np.array(
        [
            2 * ((opos[a] * nz + zpos[b]) * npair + opos[c_] * nz + zpos[e])
            + (kind == "im")
            for kind, (a, b), (c_, e) in keys
        ],
        dtype=np.int64,
    )
    acc = np.zeros((lmax + 1, len(keys)))
    n_stage = _stage_rows(lmax, npair, nc, len(keys))
    # rows 0 .. lmax are a front part ducc0 addresses (mstart is the slot
    # of L = 0) but never writes; the data rows follow
    front = lmax + 1
    stage = np.empty((front + n_stage, n_out, nz, nc), dtype=np.complex128)
    order_rows = hi2 - np.abs(all_orders) + 1
    blocks, start, rows = [], 0, 0
    for j, n in enumerate(order_rows):
        if rows + n > n_stage and j > start:
            blocks.append((start, j))
            start, rows = j, 0
        rows += n
    blocks.append((start, all_orders.size))

    for b0, b1 in blocks:
        border = all_orders[b0:b1]
        first = np.concatenate([[0], np.cumsum(order_rows[b0:b1])[:-1]])
        filled = int(order_rows[b0:b1].sum())
        view = stage[front : front + filled]
        view[...] = 0.0
        flat = stage.reshape(-1)
        for c, mp in enumerate(mps):
            mp = int(mp)
            mine = (border >= mp - 2 * lw) & (border <= mp + 2 * lw)
            if not mine.any():
                continue
            for grp, negative in _signed_groups(border[mine]):
                idx = np.searchsorted(border, grp)
                mval = np.abs(grp).astype(np.int64)
                row0 = front + first[idx] - mval  # stage row of (M, L = 0)
                for z in cols:
                    k = zpos[z]
                    p0 = plane(k, c)
                    for a, b, slot in _order_runs(grp, n_fft):
                        mv = mval[a:b]
                        if want_t:
                            col = (opos[0] * nz + k) * nc + c
                            ducc0.sht.leg2alm(
                                leg=fields[p0 : p0 + 1, :, slot : slot + b - a],
                                lmax=hi2,
                                theta=theta,
                                spin=0,
                                mval=mv,
                                mstart=(row0[a:b] * npair * nc + col).astype(np.int64),
                                lstride=npair * nc,
                                nthreads=nt,
                                alm=flat[None],
                            )
                        out_e = 1 in outs
                        out_b = 2 in outs
                        if not (out_e or out_b):
                            continue
                        qu = fields[p0 + qoff : p0 + qoff + 2, :, slot : slot + b - a]
                        if negative:
                            qu[1] *= -1.0
                        pe = opos[1] if out_e else opos[2]
                        pb = opos[2] if out_b else opos[1]
                        col = (pe * nz + k) * nc + c
                        step = (pb - pe) * nz * nc
                        if out_e and out_b:
                            alm = np.lib.stride_tricks.as_strided(
                                flat,
                                shape=(2, flat.size - step),
                                strides=(step * 16, 16),
                            )
                            ducc0.sht.leg2alm(
                                leg=qu,
                                lmax=hi2,
                                theta=theta,
                                spin=2,
                                mval=mv,
                                mstart=(row0[a:b] * npair * nc + col).astype(np.int64),
                                lstride=npair * nc,
                                nthreads=nt,
                                alm=alm,
                            )
                            if negative:
                                bcol = (pb * nz + k) * nc + c
                                for m_, r0 in zip(mv, front + first[idx[a:b]]):
                                    n = hi2 - int(m_) + 1
                                    sl = slice(
                                        r0 * npair * nc + bcol,
                                        (r0 + n) * npair * nc + bcol,
                                        npair * nc,
                                    )
                                    flat[sl] *= -1.0
                        else:  # only one of E, B wanted: a scratch pair
                            tmp_start, tmp_size = _alm_layout(grp[a:b], hi2)
                            eb = ducc0.sht.leg2alm(
                                leg=qu,
                                lmax=hi2,
                                theta=theta,
                                spin=2,
                                mval=mv,
                                mstart=tmp_start,
                                nthreads=nt,
                                alm=np.empty((2, tmp_size), np.complex128),
                            )
                            want = eb[0] if out_e else (-eb[1] if negative else eb[1])
                            dst = view[:, opos[1] if out_e else opos[2], k, c]
                            for i, (m_, r0) in enumerate(zip(mv, first[idx[a:b]])):
                                n = hi2 - int(m_) + 1
                                t0 = tmp_start[i] + int(m_)
                                dst[r0 : r0 + n] = want[t0 : t0 + n]
                            del eb
        # contract the window rows
        y = view.reshape(filled, npair, nc)
        gram = np.matmul(y * weights, np.conj(y).transpose(0, 2, 1))
        vals = gram.reshape(filled, npair * npair).view(np.float64)[:, sel]
        del gram
        for j, m_ in enumerate(border):
            am = abs(int(m_))
            lstart = max(am, lo2)
            n = hi2 - lstart + 1
            if n <= 0:
                continue
            r0 = first[j] + (lstart - am)
            acc[lstart : hi2 + 1] += vals[r0 : r0 + n]
    return {key: acc[:, j] for j, key in enumerate(keys)}


def _plane_ring_stage(ctx: _BatchedGL, fields: np.ndarray) -> None:
    """The ring stage on ``(plane, ring, order slot)`` arrays (natural ring
    order): each plane's rings to ``n_fft`` points, times the mask and the
    ring weight of the analysis, back to orders; fixed blocks of rings per
    plane handed to the workers (one FFT call each, single-threaded), so the
    result does not depend on the number of workers."""
    nplane, ntheta, n_fft = fields.shape
    mask = ctx.plane_ring_mask(n_fft)
    per, nper = _plane_blocks(n_fft, ntheta, nplane)
    nblock = nplane * nper

    def blocks(b0: int, b1: int) -> None:
        for b in range(b0, b1):
            pl, r0 = divmod(b, nper)
            r0 *= per
            r1 = min(r0 + per, ntheta)
            blk = fields[pl, r0:r1]
            ducc0.fft.c2c(blk, axes=(1,), forward=False, inorm=0, out=blk, nthreads=1)
            np.multiply(blk.view(np.float64), mask[r0:r1], out=blk.view(np.float64))
            ducc0.fft.c2c(blk, axes=(1,), forward=True, inorm=2, out=blk, nthreads=1)

    workers = min(nblock, _RING_WORKERS, row_block_workers(fields.size, ctx.nthreads))
    run_row_blocks(blocks, nblock, workers)


def _exact_row_pol_gl_batched(
    ctx: _BatchedGL,
    cmat: np.ndarray,
    ellp: int,
    m_values: Sequence[int],
    use_symmetry: bool,
    specs: Sequence[str],
    fields: Sequence[str],
) -> dict[tuple[str, str], np.ndarray]:
    """The un-normalised blocks ``sum_{m'} weight(m') sum_m [R^{XW}
    conj(R^{YZ}) + R^{XZ} conj(R^{YW})]`` (real part) of
    :func:`_exact_row_pol_gl` over ``m_values``, chunked by
    :meth:`_BatchedGL.plan`: with the Legendre tables held,
    :func:`_batched_chunk_pol_gl`; when they would have to be streamed,
    :func:`_batched_chunk_pol_direct`."""
    cols, outs = _pol_layout(fields, ellp)
    direct = not ctx.hold_table
    blocks = _pol_terms(specs, cols, (0, 0, 0) if direct else _ROT)
    keys = sorted(
        {(kind, p, q) for terms in blocks.values() for _, kind, p, q in terms}
    )
    acc = {key: np.zeros(ctx.lmax + 1) for key in keys}
    if keys:
        layout = (0 in outs, len(cols), len(outs), len(keys), direct)
        if not direct:
            mix = _pol_mixing(cmat, cols, 0 in outs)
        for mps in ctx.plan(m_values, layout=layout)["chunks"]:
            weights = np.where((mps > 0) & use_symmetry, 2.0, 1.0)
            if direct:
                part = _batched_chunk_pol_direct(
                    ctx, cmat, ellp, mps, weights, cols, outs, keys
                )
            else:
                part = _batched_chunk_pol_gl(
                    ctx, mix, ellp, mps, weights, cols, outs, keys
                )
            for key in keys:
                acc[key] += part[key]
    total = {}
    for pair, terms in blocks.items():
        total[pair] = np.zeros(ctx.lmax + 1)
        for sign, kind, p, q in terms:
            total[pair] += sign * acc[(kind, p, q)]
    return total


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
    batched: bool = True,
    max_memory_gb: float | None = None,
    batch: _BatchedGL | None = None,
) -> dict[tuple[str, str], np.ndarray]:
    r"""
    Row ``l'`` of every block ``Cov(C~^{s1}_l, C~^{s2}_l')`` for ``s1, s2`` in
    ``spectra``, on the GL grid.  Arguments as in :func:`_exact_row_gl` with
    ``cmat`` from :func:`_cl_matrix`, whose last axis must run to the internal
    band-limit ``lmax_i = lmax if lmax_int is None else lmax_int``.  Returns
    ``{(s1, s2): array over l = 0..lmax}``.  ``term_selection`` skips the
    ``m'`` of :func:`_kept_mprime`, with the spin-0 and spin-2 mode powers
    united when a polarised field is requested.

    ``batched`` (default True) computes the columns in chunks by Legendre
    matrix products (:func:`_batched_chunk_pol_gl`; for ``spectra`` with T
    only, the TT kernel :func:`_batched_chunk_gl`), within ``max_memory_gb``;
    False takes one column at a time through ducc0 transforms
    (:func:`_correlation_columns_gl`), the reference.  ``batch`` is the
    row-independent state (:func:`_polarised_batch`) shared by the rows of
    :func:`exact_covariance_pol`.  A grid below
    :func:`~.grid.gl_minimal_lmax` always takes the per-column path.
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

    specs, fields = _spectra_fields(spectra)
    pairs = [(s1, s2) for s1 in specs for s2 in specs]
    idx = {s: (FIELDS.index(s[0]), FIELDS.index(s[1])) for s in specs}
    ell = np.arange(lmax_i + 1)
    norm = 1.0 / ((2 * ell + 1) * (2 * ellp + 1))

    if use_symmetry:
        m_values = range(0, ellp + 1)
    else:
        m_values = range(-ellp, ellp + 1)
    keep = None
    if term_selection is not None:
        spins = (0,) if fields == ("T",) else (0, 2)
        keep = _kept_mprime(mask_alm, lw, ellp, term_selection, spins, nthreads)

    if batched and lmax_grid >= gl_minimal_lmax(lmax_i, lw):
        if batch is None:
            batch = _polarised_batch(
                mask_alm, lw, lmax_i, lmax_grid, nthreads, max_memory_gb, fields
            )
        elif (batch.lw, batch.lmax, batch.lmax_grid) != (lw, lmax_i, lmax_grid):
            raise ValueError("batch was built for another lw, lmax_int or grid")
        elif batch.spins != _polarised_spins(fields):
            raise ValueError("batch was built for other spectra")
        kept = [mp for mp in m_values if keep is None or keep[mp + ellp]]
        if fields == ("T",):
            # 2 sum |R^TT|^2: the TT kernel, with the Wick factor 2
            tt = _exact_row_gl_batched(batch, cmat[0, 0], ellp, kept, use_symmetry)
            return {("TT", "TT"): (norm * 2.0 * tt)[: lmax + 1]}
        total = _exact_row_pol_gl_batched(
            batch, cmat, ellp, kept, use_symmetry, specs, fields
        )
        return {pair: (norm * total[pair])[: lmax + 1] for pair in pairs}

    if mask_map is None:
        mask_map = gl_synthesis(mask_alm, lw, lmax_grid, nthreads=nthreads)
    total = {pair: np.zeros(lmax_i + 1, dtype=complex) for pair in pairs}

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

    return {pair: (norm * total[pair].real)[: lmax + 1] for pair in pairs}


def _polarised_spins(fields: Sequence[str]) -> tuple[int, ...]:
    """Legendre tables a polarised row needs: spin 0 for T alone (the TT
    kernel), else spin 2 and spin 0 if T is among ``fields``."""
    if tuple(fields) == ("T",):
        return (0,)
    return ((0,) if "T" in fields else ()) + (2,)


def _polarised_batch(
    mask_alm: np.ndarray,
    lw: int,
    lmax: int,
    lmax_grid: int,
    nthreads: int | None,
    max_memory_gb: float | None,
    fields: Sequence[str],
) -> _BatchedGL:
    """The :class:`_BatchedGL` of a polarised row for ``fields``."""
    return _BatchedGL(
        mask_alm,
        lw,
        lmax,
        lmax_grid,
        nthreads,
        max_memory_gb,
        spins=_polarised_spins(fields),
    )


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
    max_memory_gb: float | None = None,
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
        GL grid only: threads for ducc0.  The batched rows are bit-identical
        for any value.
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
    max_memory_gb : float, optional
        GL grid only: memory budget in GiB (default
        :data:`DEFAULT_EXACT_MEMORY_GB`, 2) for everything the row holds that
        scales with the problem.  The columns ``m'`` of a GL row are computed
        in chunks, their spherical-harmonic transforms done together as
        matrix products (module docstring, "Cost of the GL column"); the
        budget covers the chunk's ring profiles (the dominant term,
        ``16 N ntheta`` bytes per column with ``ntheta`` the GL rings and
        ``N <= 2 lmax_int + lw + 1`` the chunk's azimuthal FFT length), the
        mask sampled on the rings, per-order scratch, the mask's ring modes
        and the Legendre table -- held whole if it fits in half the budget,
        streamed in blocks otherwise.  It sets how many columns share a
        chunk, not the result (which agrees to a few 1e-15 of the largest
        entry whatever the chunking).  Not covered: the mask as loaded, its
        alm, the returned row, ducc0's and the BLAS library's internal
        buffers and freed memory the system allocator keeps; with 2 GiB the
        process's resident size was measured to grow by 2.0-2.3 GiB
        (:func:`_batched_row_plan`).  Raises ``ValueError`` with
        ``grid="healpix"``.

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
    if max_memory_gb is not None and grid != "gl":
        raise ValueError("max_memory_gb applies to grid='gl' only")

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
            max_memory_gb=max_memory_gb,
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
    max_memory_gb: float | None = None,
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
        ``nprocs=14`` was 1.56x *slower* than one process on 14 threads with
        the per-column transforms: the performance/efficiency core split caps
        static per-row parallelism at ~8x, and 14 copies of the 7.7 MB maps
        then run into memory bandwidth.  It should win where the thread
        scaling runs out first -- many homogeneous cores, or a grid small
        enough that a transform does not thread well.  The batched rows'
        matrix products run on the BLAS library's own threads, which
        ``nthreads`` does not set: with ``nprocs > 1`` limit them too
        (``OMP_NUM_THREADS``, ``OPENBLAS_NUM_THREADS`` or
        ``VECLIB_MAXIMUM_THREADS``, set before numpy is imported).
    term_selection : float, optional
        ``grid="gl"`` only; as in :func:`exact_covariance_row`.  The mask alm
        is rotated to the pole frame once here and shared by every row (and
        every worker process).
    max_memory_gb : float, optional
        ``grid="gl"`` only; as in :func:`exact_covariance_row`, per process.
        The mask's ring modes and, when it fits in half the budget, the
        Legendre table are built once and shared by every row (by every row
        of a worker with ``nprocs > 1``, each of which holds its own copy:
        the peak is then about ``nprocs`` times the budget).
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
    if max_memory_gb is not None and grid != "gl":
        raise ValueError("max_memory_gb applies to grid='gl' only")

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
                max_memory_gb=max_memory_gb,
            ):
                sigma[:, ellp] = row
            return sigma
        batch = mask_map = None
        if lmax_grid >= gl_minimal_lmax(lmax_i, lw):
            batch = _BatchedGL(mask_alm, lw, lmax_i, lmax_grid, nthreads, max_memory_gb)
        else:
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
                batch=batch,
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
    max_memory_gb: float | None = None,
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
    max_memory_gb : float, optional
        GL grid only; as in :func:`exact_covariance_row` (default
        :data:`DEFAULT_EXACT_MEMORY_GB`, 2): the columns ``m'`` of the row
        are computed in chunks, and the budget sets the chunk size and
        whether the Legendre tables (spin 0 and spin 2, up to three times
        the TT table) are held.  A polarised column carries up to nine ring
        profiles (the T, E and B columns, each on T, Q and U), so a chunk
        holds up to nine times fewer ``m'`` than a TT one at the same
        budget.  With the tables held the chunk's Legendre sums are matrix
        products against them; otherwise ducc0 transforms on the chunk's
        bands, with no table (module docstring, "Cost of the GL column").
        It changes the result only at the level of rounding.  Raises
        ``ValueError`` with ``grid="healpix"``.

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
    transforms.  Rows with ``l' < 2`` have zero E and B columns.  The
    batched columns (module docstring, "Cost of the GL column") share the
    step-1 work of the E and B columns, which a rotation of the B axis makes
    identical.
    """
    specs, fields = _spectra_fields(spectra)
    lmax_i = int(lmax if lmax_int is None else lmax_int)
    if lmax_i < lmax:
        raise ValueError("lmax_int must be at least lmax")
    if term_selection is not None and grid != "gl":
        raise ValueError("term_selection is implemented for grid='gl' only")
    if max_memory_gb is not None and grid != "gl":
        raise ValueError("max_memory_gb applies to grid='gl' only")
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
        max_memory_gb=max_memory_gb,
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
    max_memory_gb: float | None = None,
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
    max_memory_gb : float, optional
        GL grid only; as in :func:`exact_covariance_row_pol`.  The mask's
        ring modes and, when they fit in half the budget, the Legendre
        tables are built once and shared by every row.
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
    if max_memory_gb is not None and grid != "gl":
        raise ValueError("max_memory_gb applies to grid='gl' only")

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
    batch = mask_map = None
    if lmax_grid >= gl_minimal_lmax(lmax_i, lw):
        batch = _polarised_batch(
            mask_alm, lw, lmax_i, lmax_grid, nthreads, max_memory_gb, fields
        )
    else:
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
            batch=batch,
        )
        for pair in pairs:
            sigma[pair][:, ellp] = row[pair]
    return sigma
