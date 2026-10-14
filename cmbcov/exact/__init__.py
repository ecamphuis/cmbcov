r"""
Exact pseudo-:math:`C_\ell` covariance (Camphuis et al. 2022, Sect. 3).

This package implements the *exact* algorithm of Camphuis et al. (2022,
arXiv:2204.13721, Sect. 3.1 and Fig. 2) for a single real mask :math:`W` and a
real temperature spectrum :math:`C_L`, without any narrow-kernel approximation.
It evaluates the full covariance row by row on a HEALPix or Gauss-Legendre
grid, for temperature alone (`exact_covariance`, `exact_covariance_row`) and
with polarisation (`exact_covariance_pol`, `exact_covariance_row_pol`), and is
the reference the approximations in `cmbcov.approximations` are validated
against.

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

Rows are independent, so :func:`exact_covariance` can spread them over
processes (``nprocs``, default 1).  ducc0's own threads (``nthreads``) share
one copy of the maps and are usually the better lever.

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
    and is only as accurate as the iterated ``map2alm`` is an approximation of
    the integral.

``grid="gl"``
    The mask enters as a band-limited alm set (band-limit ``Lw``), synthesised
    on a Gauss-Legendre grid (:mod:`cmbcov.harmonics.grid`) large enough that every
    analysis in the algorithm is an exact integral.  The result is the
    covariance of the *exact-integral* pseudo-spectrum estimator of the
    band-limited mask, to machine precision and with no adjoint machinery.  The
    only approximation is the mask truncation ``Lw`` and the ``map2alm`` that
    produced its alm.

The two differ by an amount that shrinks as ``nside^-2`` as the HEALPix
quadrature improves and does *not* shrink with more ``map2alm`` iterations.

Reported band-limit versus internal band-limit
----------------------------------------------
:math:`\tilde\Sigma_{\ell\ell'}` sums :math:`C_L` over the mask's coupling
range *around* :math:`\ell'`, so the algorithm needs :math:`C_L` **above** the
highest multipole it reports.  Truncating the sum at ``lmax`` sets :math:`C_L =
0` over the upper half of the kernel and biases every row near the top of the
matrix, silently and severely (for a survey mask the diagonal can be off by a
factor of order 3).

``lmax_int`` (default ``None``, meaning ``lmax_int = lmax``) therefore
separates the two: every harmonic array, the :math:`C_L` sum and the GL grid
are carried to ``lmax_int``, and only rows/columns up to ``lmax`` are returned.
This is exactly the arithmetic of computing the whole matrix at ``lmax_int``
and slicing it, at a fraction of the cost.  The margin ``lmax_int - lmax`` is
compared against :func:`mask_coupling_width`, an estimate of the mask's own
coupling half-width built from ``W_L``, and a ``UserWarning`` names both
numbers when the margin is too small.

The code is split by path: `common` (fields, spectra, the mask coupling width),
`healpix` (the HEALPix path), `gl` (the temperature-only Gauss-Legendre column
computation), `pol` (its polarised version), `plan` (the memory plan of the
batched rows), `parallel` (rows spread over processes) and `covariance` (the
public entry points).
"""

from .common import SPECTRA, mask_coupling_width
from .covariance import (
    exact_covariance,
    exact_covariance_pol,
    exact_covariance_row,
    exact_covariance_row_pol,
)

__all__ = [
    "SPECTRA",
    "exact_covariance",
    "exact_covariance_pol",
    "exact_covariance_row",
    "exact_covariance_row_pol",
    "mask_coupling_width",
]
