r"""
Gauss-Legendre (GL) grid primitives: exact spherical-harmonic quadrature.

Why a GL grid instead of HEALPix
--------------------------------
HEALPix has no exact quadrature rule.  ``hp.map2alm`` is an *approximate*
inverse of ``hp.alm2map`` -- about 5e-3 at ``iter=0``, 1e-5 at ``iter=3``, and
never better because the iterated solution converges to the pixel
least-squares alm, not to the integral.  On a Gauss-Legendre grid, ducc0's
``analysis_2d`` is both the exact inverse and the exact adjoint of
``synthesis_2d`` (round trip ~5e-14, spin 0 and spin 2), so once the mask
enters as a band-limited alm set everything downstream is exact to machine
precision, and the Neumann-series adjoint correction that the HEALPix path of
:mod:`cmbcov.exact` needs becomes unnecessary.

Grid definition
---------------
A GL grid of *grid band-limit* ``Lg`` has ``ntheta = Lg + 1`` rings at the
Gauss-Legendre nodes in :math:`\cos\theta` and ``nphi = 2 Lg + 2`` equispaced
points per ring (:func:`gl_shape`).  With ``n = Lg + 1`` nodes the GL rule
integrates polynomials of degree ``2n - 1 = 2 Lg + 1`` in :math:`\cos\theta`
exactly, and ``nphi > 2 Lg`` resolves every azimuthal frequency up to
``2 Lg`` without aliasing.

The minimal exact grid
----------------------
The covariance algorithm analyses products of a mask ``W`` (band-limit
``Lw``) with harmonics of degree up to ``lmax``, against harmonics of degree
up to ``lmax``.  The integrand :math:`W \, Y_{\ell m} \, Y^*_{\ell' m'}` has
degree ``Lw + 2 lmax`` (the azimuthal selection rule makes it a genuine
polynomial in :math:`\cos\theta`), so exactness requires
``2 Lg + 1 >= Lw + 2 lmax``, i.e.

    Lg_min = lmax + ceil((Lw - 1) / 2)             (:func:`gl_minimal_lmax`)

Measured on the package's baseline mask: the error is ~1e-2 (row-normalised,
worst multipole) one below ``Lg_min`` and ~1e-15 at ``Lg_min``.  The a-priori
sufficient rule ``Lg = lmax + Lw`` over-resolves by ``Lw/2``.

When only *one* harmonic of degree ``ell`` is masked (step 1 of the exact
covariance, and :func:`spin_weighted_integrals_gl`) the integrand degree drops
to ``Lw + ell + lmax`` and the rule becomes :func:`gl_step1_minimal_lmax`.  Note
that ``analysis_2d`` cannot produce coefficients above the grid band-limit, so
``Lg >= lmax`` is a floor on both rules -- which is what keeps the saving modest
for the covariance.

Complex maps
------------
ducc0 transforms real maps (``a_{L,-M} = (-1)^M a^*_{LM}``).  A complex map
:math:`f = \mathrm{Re} f + i\,\mathrm{Im} f` is transformed by treating the
two parts as independent real maps with healpy alms ``r`` and ``s``; the
full-``M`` coefficients are then

    x[L,  M] = r[L, M] + i s[L, M]                          (M >= 0)
    x[L, -M] = (-1)^M ( conj r[L, M] + i conj s[L, M] )     (M >  0)

**Full-M layout** used throughout this module and by
:mod:`cmbcov.exact`: a complex array of shape
``(lmax + 1, 2 lmax + 1)`` indexed ``[L, M + lmax]``, so that column ``lmax``
holds ``M = 0`` and unused entries (``|M| > L``) are zero.

The only approximation left
---------------------------
Everything here is exact for a band-limited mask.  The mask's truncation
``Lw`` (and the HEALPix ``map2alm`` that produces its alm from a pixel map) is
therefore the single approximation of the GL pipeline; hard-edged pixel masks
band-limit slowly, apodised masks quickly.
"""

from __future__ import annotations

import functools
import math
import os

import ducc0
import numpy as np

from .utils.threading_utils import (
    _env_nthreads,
    get_optimal_nthreads,
    row_block_workers,
    run_row_blocks,
)

__all__ = [
    "gl_minimal_lmax",
    "gl_step1_minimal_lmax",
    "gl_shape",
    "gl_thetas",
    "gl_quadrature_weights",
    "gl_synthesis",
    "gl_analysis",
    "gl_synthesis_single_m",
    "gl_north_rings",
    "gl_legendre_table",
    "gl_legendre_table_bytes",
    "gl_legendre_table_capacity",
    "gl_ring_modes",
    "gl_synthesis_complex",
    "gl_analysis_complex",
    "full_from_pair",
    "pair_from_full",
    "spin_weighted_integrals_gl",
    "gl_mask_modes",
    "banded_integrals_gl",
    "cross_spectrum_full_m",
]


# --------------------------------------------------------------------------- #
# Grid geometry
# --------------------------------------------------------------------------- #
def gl_minimal_lmax(lmax: int, lw: int) -> int:
    r"""
    Smallest GL grid band-limit on which the covariance algorithm is exact.

    Parameters
    ----------
    lmax : int
        Band-limit of the harmonics being analysed and synthesised.
    lw : int
        Band-limit of the mask.

    Returns
    -------
    int
        ``Lg = lmax + ceil((lw - 1) / 2)`` (never below ``lmax``).

    Notes
    -----
    A GL rule with ``n = Lg + 1`` nodes integrates polynomials of degree
    ``2n - 1 = 2 Lg + 1`` in :math:`\cos\theta` exactly.  The worst integrand
    of the exact covariance is :math:`W \, Y_{\ell m} \, Y^*_{\ell' m'}` with
    :math:`\ell, \ell' \le` ``lmax``; after the :math:`\phi` integration
    (which forces the three azimuthal orders to sum to zero, making the
    :math:`\sin\theta` factors pair up) it is a polynomial of degree
    ``lw + 2 lmax``.  Exactness therefore needs ``2 Lg + 1 >= lw + 2 lmax``,
    i.e. ``Lg >= lmax + (lw - 1) / 2``.  The azimuthal count
    ``nphi = 2 Lg + 2`` then also exceeds the maximal azimuthal frequency
    ``lw + 2 lmax``.  The error is ~1e-2 one grid step below this value and
    ~1e-15 at it.
    """
    if lmax < 0 or lw < 0:
        raise ValueError("lmax and lw must be non-negative")
    return max(lmax, lmax + math.ceil((lw - 1) / 2))


def gl_step1_minimal_lmax(lmax: int, lw: int, ell: int) -> int:
    r"""
    Minimal exact GL grid for *one masked harmonic* (step 1 of the exact row).

    Step 1 of :func:`cmbcov.exact.exact_covariance_row`
    analyses :math:`W Y_{\ell m}` against harmonics of degree up to ``lmax``,
    i.e. the single harmonic has degree ``ell``, not ``lmax``.  The integrand
    has degree ``lw + ell + lmax`` rather than ``lw + 2 lmax``, so

        Lg >= (lw + ell + lmax - 1) / 2 ,

    but ``analysis_2d`` cannot produce coefficients above the grid band-limit,
    so the grid must also satisfy ``Lg >= lmax``:

        Lg_1 = max(lmax, ceil((lw + ell + lmax - 1) / 2))

    (the same rule as :func:`spin_weighted_integrals_gl` with
    ``lmax_out = lmax``).  Verified numerically: exact at ``Lg_1``, aliased
    (error 0.1-0.5) one step below, for several ``(lmax, lw, ell)``.

    It reduces to :func:`gl_minimal_lmax` ``(lmax, lw)`` at ``ell == lmax``
    and saturates at ``lmax`` for ``ell <= lmax - lw + 1``.  **It is not used
    by default**: the ``Lg >= lmax`` floor caps the saving, the rows that
    benefit are the cheap low-``l'`` ones, and using a different grid for
    step 1 perturbs the result at the round-off level.
    """
    if lmax < 0 or lw < 0 or ell < 0:
        raise ValueError("lmax, lw and ell must be non-negative")
    return max(lmax, math.ceil((lw + ell + lmax - 1) / 2))


def gl_shape(lmax_grid: int) -> tuple[int, int]:
    """
    Map dimensions ``(ntheta, nphi)`` of the GL grid of band-limit ``lmax_grid``.

    ``ntheta = lmax_grid + 1`` rings, ``nphi = 2 lmax_grid + 2`` points per ring.
    """
    if lmax_grid < 0:
        raise ValueError("lmax_grid must be non-negative")
    return lmax_grid + 1, 2 * lmax_grid + 2


def gl_thetas(lmax_grid: int) -> np.ndarray:
    """Colatitudes (radians) of the ``lmax_grid + 1`` GL rings, north to south."""
    ntheta, _ = gl_shape(lmax_grid)
    return ducc0.misc.GL_thetas(ntheta)


def gl_quadrature_weights(lmax_grid: int) -> np.ndarray:
    r"""
    Quadrature weight of every grid point, shape ``(ntheta, nphi)``.

    ``sum(weights * f)`` is the exact integral :math:`\int f \, d\Omega` of any
    band-limited ``f`` of degree ``<= 2 lmax_grid + 1``; the weights sum to
    :math:`4\pi`.
    """
    ntheta, nphi = gl_shape(lmax_grid)
    w = ducc0.misc.GL_weights(ntheta, nphi)  # per ring, includes 2 pi / nphi
    return np.repeat(w[:, None], nphi, axis=1)


# --------------------------------------------------------------------------- #
# Real-map transforms
# --------------------------------------------------------------------------- #
#: Grid points below which one more ducc0 thread does not pay for itself.
#: Measured on a 14-core laptop with one exact-covariance row (grid points =
#: ``ntheta * nphi``): grid 56 (6.5e3 points) is fastest on 1 thread and 57%
#: slower on 14; grid 168 (5.7e4) and 264 (1.4e5) peak at 4 threads and are
#: within 5% up to 8; grid 692 (9.6e5) is still improving at 14 (4.31 s versus
#: 5.21 s on 7).  ``get_optimal_nthreads(None)``'s ``min(ncpu // 2, 8)`` = 7 is
#: 21% off the best at the large end and 35% off at the small end, so the GL
#: transforms size their own default.
_GL_POINTS_PER_THREAD = 16384


def _nthreads(nthreads: int | None, lmax_grid: int | None = None) -> int:
    """
    Threads for one GL transform.

    An explicit ``nthreads`` always wins, then ``OMP_NUM_THREADS``, then a
    default scaled with the grid size (:data:`_GL_POINTS_PER_THREAD`) and
    capped at the CPU count.  With no grid size to go on, the package-wide
    :func:`~.utils.threading_utils.get_optimal_nthreads` policy applies.
    """
    if nthreads is not None:
        return int(nthreads)
    env_nthreads = _env_nthreads()
    if env_nthreads is not None:
        return env_nthreads
    ncpu = os.cpu_count() or 1
    if lmax_grid is None:
        return get_optimal_nthreads(None) or ncpu
    ntheta, nphi = gl_shape(lmax_grid)
    return int(min(ncpu, max(1, (ntheta * nphi) // _GL_POINTS_PER_THREAD)))


def _check_alm(alm: np.ndarray, lmax: int, spin: int) -> np.ndarray:
    import healpy as hp  # lazy, see utils/healpy_utils.py

    nalm = hp.Alm.getsize(lmax)
    alm = np.ascontiguousarray(alm, dtype=np.complex128)
    ncomp = 1 if spin == 0 else 2
    if alm.ndim == 1:
        alm = alm[None, :]
    if alm.shape != (ncomp, nalm):
        raise ValueError(
            f"alm must have shape ({ncomp}, {nalm}) for spin {spin}, lmax {lmax}; "
            f"got {alm.shape}"
        )
    return alm


def gl_synthesis(
    alm: np.ndarray,
    lmax: int,
    lmax_grid: int,
    spin: int = 0,
    nthreads: int | None = None,
) -> np.ndarray:
    """
    Synthesise a real map on the GL grid of band-limit ``lmax_grid``.

    Parameters
    ----------
    alm : ndarray
        healpy-ordered coefficients truncated at ``lmax``: shape ``(nalm,)``
        for spin 0, ``(2, nalm)`` (E, B) for ``spin > 0``.
    lmax : int
        Band-limit of ``alm``.  May exceed ``lmax_grid``: synthesis is a
        point-wise evaluation and never needs a large grid.
    lmax_grid : int
        Grid band-limit, see :func:`gl_shape`.
    spin : int
        Spin of the transform (0 for temperature, 2 for Q/U).
    nthreads : int, optional
        Threads for ducc0; defaults to the package's thread policy.

    Returns
    -------
    ndarray
        Map of shape ``(ntheta, nphi)`` for spin 0, ``(2, ntheta, nphi)``
        otherwise.
    """
    ntheta, nphi = gl_shape(lmax_grid)
    out = ducc0.sht.synthesis_2d(
        alm=_check_alm(alm, lmax, spin),
        ntheta=ntheta,
        nphi=nphi,
        lmax=lmax,
        geometry="GL",
        spin=spin,
        nthreads=_nthreads(nthreads, lmax_grid),
    )
    return out[0] if spin == 0 else out


def gl_analysis(
    map: np.ndarray,
    lmax: int,
    lmax_grid: int,
    spin: int = 0,
    nthreads: int | None = None,
) -> np.ndarray:
    """
    Analyse a real GL map into healpy-ordered alm up to ``lmax``.

    On the GL grid this is the exact inverse *and* exact adjoint of
    :func:`gl_synthesis` whenever the map is band-limited within the grid.

    Parameters
    ----------
    map : ndarray
        Shape ``(ntheta, nphi)`` for spin 0, ``(2, ntheta, nphi)`` (Q, U)
        for ``spin > 0``, matching :func:`gl_shape` ``(lmax_grid)``.
    lmax : int
        Output band-limit; must not exceed ``lmax_grid``.
    lmax_grid : int
        Grid band-limit, used to validate the map shape.
    spin, nthreads
        As in :func:`gl_synthesis`.

    Returns
    -------
    ndarray
        Shape ``(nalm,)`` for spin 0, ``(2, nalm)`` (E, B) otherwise.
    """
    if lmax > lmax_grid:
        raise ValueError(
            f"analysis up to lmax={lmax} needs a grid with lmax_grid >= lmax "
            f"(got {lmax_grid})"
        )
    shape = gl_shape(lmax_grid)
    map = np.ascontiguousarray(map, dtype=np.float64)
    ncomp = 1 if spin == 0 else 2
    if map.ndim == 2:
        map = map[None, :, :]
    if map.shape != (ncomp,) + shape:
        raise ValueError(
            f"map must have shape {(ncomp,) + shape} for spin {spin}, "
            f"lmax_grid {lmax_grid}; got {map.shape}"
        )
    out = ducc0.sht.analysis_2d(
        map=map,
        lmax=lmax,
        geometry="GL",
        spin=spin,
        nthreads=_nthreads(nthreads, lmax_grid),
    )
    return out[0] if spin == 0 else out


_GL_GEOMETRY: dict = {}


def _gl_geometry(lmax_grid: int) -> dict:
    """Cached ring description of the GL grid, for the low-level ducc0 calls."""
    geom = _GL_GEOMETRY.get(lmax_grid)
    if geom is None:
        ntheta, nphi = gl_shape(lmax_grid)
        geom = {
            "ntheta": ntheta,
            "nphi": nphi,
            "theta": ducc0.misc.GL_thetas(ntheta),
            # ducc0 wants uint64 ring descriptors but int64 m descriptors.
            "nphi_arr": np.full(ntheta, nphi, dtype=np.uint64),
            "phi0": np.zeros(ntheta),
            "ringstart": (np.arange(ntheta) * nphi).astype(np.uint64),
            # 2 pi w_GL per ring (GL_weights includes 2 pi / nphi), the ring
            # weight of banded_integrals_gl.
            "wbar": ducc0.misc.GL_weights(ntheta, nphi) * nphi,
        }
        for array in geom.values():
            if isinstance(array, np.ndarray):
                array.setflags(write=False)
        _GL_GEOMETRY[lmax_grid] = geom
    return geom


def gl_synthesis_single_m(
    alm: np.ndarray,
    lmax: int,
    m: int,
    lmax_grid: int,
    nthreads: int | None = None,
) -> np.ndarray:
    r"""
    :func:`gl_synthesis` for an alm whose only nonzero entries have order ``m``.

    A full synthesis evaluates the Legendre sum for every order ``0..lmax``,
    which costs :math:`O(\ell_{max}^2 n_\theta / 2)`; if only one order is
    populated -- as for the single-harmonic map :math:`Y_{\ell' m'}` of step 1
    of the exact covariance -- all but ``lmax - m + 1`` of those terms are
    zero.  This routine calls ``ducc0.sht.alm2leg`` for that one order and then
    ``leg2map`` for the azimuthal FFT, which is what ``synthesis_2d`` does
    internally with the full order range.  The omitted orders contribute exact
    zeros, so the result is **bit-identical** to
    ``gl_synthesis(alm, lmax, lmax_grid)`` (asserted in
    ``tests/test_exact_gl_speed.py``), and only the FFT cost remains.

    Parameters
    ----------
    alm : ndarray
        healpy-ordered spin-0 coefficients truncated at ``lmax``, shape
        ``(nalm,)``.  Entries of order other than ``m`` are ignored, not
        checked.
    lmax : int
        Band-limit of ``alm``.
    m : int
        The single azimuthal order present, ``0 <= m <= lmax``.
    lmax_grid : int
        Grid band-limit, see :func:`gl_shape`.
    nthreads : int, optional
        Threads for ducc0.

    Returns
    -------
    ndarray
        Real map of shape ``(ntheta, nphi)``.
    """
    import healpy as hp  # lazy, see utils/healpy_utils.py

    if not 0 <= m <= lmax:
        raise ValueError(f"m must lie in [0, lmax]; got m={m}, lmax={lmax}")
    geom = _gl_geometry(lmax_grid)
    nt = _nthreads(nthreads, lmax_grid)
    alm = np.ascontiguousarray(alm, dtype=np.complex128)
    if alm.ndim == 1:
        alm = alm[None, :]
    if alm.shape != (1, hp.Alm.getsize(lmax)):
        raise ValueError(
            f"alm must have shape (1, {hp.Alm.getsize(lmax)}) for lmax {lmax}; "
            f"got {alm.shape}"
        )
    leg1 = ducc0.sht.alm2leg(
        alm=alm,
        lmax=lmax,
        theta=geom["theta"],
        spin=0,
        mval=np.array([m], dtype=np.int64),
        # index at which (l=0, m) would sit in the healpy layout
        mstart=np.array([hp.Alm.getidx(lmax, 0, m)], dtype=np.int64),
        nthreads=nt,
    )
    # leg2map wants a complete order range 0..m; the lower orders are zero.
    leg = np.zeros((1, geom["ntheta"], m + 1), dtype=np.complex128)
    leg[0, :, m] = leg1[0, :, 0]
    out = ducc0.sht.leg2map(
        leg=leg,
        nphi=geom["nphi_arr"],
        phi0=geom["phi0"],
        ringstart=geom["ringstart"],
        nthreads=nt,
    )
    return out.reshape(geom["ntheta"], geom["nphi"])


# --------------------------------------------------------------------------- #
# Legendre tables and ring modes (the batched exact covariance rows)
# --------------------------------------------------------------------------- #
def gl_north_rings(lmax_grid: int) -> int:
    r"""
    Number of rings of the northern half of the GL grid, equator included.

    The GL nodes are symmetric about the equator: ring ``ntheta - 1 - j``
    sits at :math:`\pi - \theta_j`, where
    :math:`\lambda_{LM}(\pi - \theta) = (-1)^{L+M} \lambda_{LM}(\theta)`.
    Rings ``0 .. n - 1`` with ``n = ceil(ntheta / 2)`` therefore carry every
    Legendre value of the grid; for odd ``ntheta`` the last of them is the
    equator, its own mirror image.
    """
    ntheta, _ = gl_shape(lmax_grid)
    return (ntheta + 1) // 2


def gl_legendre_table(
    ms: np.ndarray,
    lmax: int,
    lmax_grid: int,
    nthreads: int | None = None,
    spin: int = 0,
    out: np.ndarray | None = None,
) -> list[tuple[np.ndarray, ...]]:
    r"""
    Normalised associated Legendre functions on the northern GL rings, split
    by the parity of ``L - M`` (``spin=0``), or the two spin-2 functions
    :math:`\lambda^\pm_{LM}` split the same way (``spin=2``).

    For every order ``M`` in ``ms`` returns the pair ``(even, odd)`` with
    ``even[i, j] = lambda_{M + 2i, M}(theta_j)`` and
    ``odd[i, j] = lambda_{M + 2i + 1, M}(theta_j)``, ``L <= lmax``,
    ``j < gl_north_rings(lmax_grid)``, in the convention of ducc0 and
    healpy: :math:`Y_{LM}(\theta, \phi) = \lambda_{LM}(\theta) e^{i M \phi}`
    (Condon-Shortley phase included).  The southern rings follow from
    :math:`\lambda_{LM}(\pi - \theta) = (-1)^{L+M} \lambda_{LM}(\theta)`,
    which is why the two parities are kept apart: a sum over all rings of
    :math:`\lambda_{LM} f` is ``even @ (f_N + f_S)`` and
    ``odd @ (f_N - f_S)`` on the northern rings.

    The values are ducc0's own, bit for bit the ones its transforms use.
    ``ducc0.sht.leg2alm`` computes
    :math:`a_{LM} = \sum_j \mathrm{leg}_j(M)\, \lambda_{LM}(\theta_j)` with no
    weights, so a call on two rings with legs ``(1, i)`` returns
    :math:`\lambda_{LM}(\theta_1) + i \lambda_{LM}(\theta_2)` for every
    ``(L, M)`` at once; ducc0's scaled recursion handles the underflow of
    :math:`\sin^M\theta` near the poles.  Each call writes its ring pair
    straight into place (``lstride`` = one row of rings), so every order's
    table is ring-contiguous rows ``L = M .. lmax`` and the two parities are
    views of every other row; no transposition or second buffer is needed.
    A ducc0 call recomputes its recursion coefficients, a fixed cost per
    ``(L, M)`` whatever the number of rings, so the cost per entry falls with
    the number of orders per call: at ``lmax`` 2000 on 14 threads, 1, 4 and
    16 orders ran at 2.5, 0.9 and 0.4 ns per entry (spin 0; spin 2 2.0, 0.7
    and 0.6), against 3.4, 1.2 and 0.6 with the former transposed layout.
    (No numpy recursion reproduces these values: at ``lmax`` 2000 the
    functions near the poles are ill-conditioned in :math:`\cos\theta`, and
    ducc0 itself is 3e-10 away from an extended-precision evaluation at
    :math:`\theta = 0.0011`, ``L = 2000``, a numpy recursion 5e-10.)  The
    ring pairs are spread over ``nthreads`` workers
    (:func:`~.utils.threading_utils.run_row_blocks`), each ducc0 call
    single-threaded, so the table is **bit-identical for any thread count**
    and any grouping of the orders into calls.

    **Spin 2.**  ducc0's spin-2 transforms are HEALPix's: for ``M >= 0`` the
    (Q, U) ring profile of the (E, B) coefficients of order ``M`` is

    .. math::

        Q_M(\theta) = \sum_L \lambda^+_{LM} E_{LM} + i \lambda^-_{LM} B_{LM},
        \qquad
        U_M(\theta) = \sum_L -i \lambda^-_{LM} E_{LM} + \lambda^+_{LM} B_{LM}

    with real :math:`\lambda^\pm_{LM}(\theta)` (``alm2leg`` of an E unit
    vector returns ``(lambda^+, -i lambda^-)``, of a B unit vector
    ``(i lambda^-, lambda^+)``), zero for ``L < 2``.  ``leg2alm`` is the
    adjoint, :math:`E = \sum_\theta \lambda^+ Q + i\lambda^- U`,
    :math:`B = \sum_\theta -i\lambda^- Q + \lambda^+ U`, so the call with
    Q legs ``(1, i)`` on two rings and U legs 0 returns
    :math:`\lambda^+(\theta_1) + i\lambda^+(\theta_2)` in E and ``-i``
    times the same combination of :math:`\lambda^-` in B (multiplied by
    ``i`` in place).  The mirror rule
    is :math:`\lambda^+_{LM}(\pi - \theta) = (-1)^{L+M}
    \lambda^+_{LM}(\theta)` and :math:`\lambda^-_{LM}(\pi - \theta) =
    -(-1)^{L+M} \lambda^-_{LM}(\theta)`: the ``L - M``-even rows of
    :math:`\lambda^-` are odd functions about the equator.  (All of this is
    asserted against ``alm2leg`` and :func:`gl_synthesis_complex` in
    ``tests/test_exact_pol_batched.py``.)

    Parameters
    ----------
    ms : array_like of int
        Orders, strictly increasing, in ``[0, lmax]``.
    lmax : int
        Highest degree.
    lmax_grid : int
        GL grid band-limit (:func:`gl_shape`).
    nthreads : int, optional
        Workers; the package thread policy by default.
    spin : {0, 2}
        Which functions to tabulate.
    out : ndarray, optional
        A one-dimensional ``complex128`` buffer of at least
        :func:`gl_legendre_table_capacity` elements to build the tables in
        (they are then views of it, overwritten by the next call on the same
        buffer); only its last :func:`gl_legendre_table_bytes` bytes are
        written.  A new buffer by default.

    Returns
    -------
    list of tuple of ndarray
        Per order, in the order of ``ms``: ``(even, odd)`` for ``spin=0``,
        ``(plus_even, plus_odd, minus_even, minus_odd)`` for ``spin=2``;
        read-only ``float64`` arrays of shapes ``(ceil(n / 2), n_north)``
        (even) and ``(floor(n / 2), n_north)`` (odd), ``n = lmax - M + 1``,
        rows with a constant stride (every other row of the order's
        ring-contiguous table), i.e. BLAS-compatible as they are.  They
        share one buffer, of which :func:`gl_legendre_table_bytes` bytes are
        written.
    """
    if spin not in (0, 2):
        raise ValueError(f"spin must be 0 or 2, got {spin}")
    ms = np.asarray(ms, dtype=np.int64).ravel()
    if ms.size == 0:
        return []
    if ms[0] < 0 or ms[-1] > lmax or np.any(np.diff(ms) <= 0):
        raise ValueError("ms must be strictly increasing orders in [0, lmax]")
    geom = _gl_geometry(lmax_grid)
    n_north = gl_north_rings(lmax_grid)
    npair = (n_north + 1) // 2
    nt = _nthreads(nthreads, lmax_grid)
    theta = np.empty(2 * npair)
    theta[:n_north] = geom["theta"][:n_north]
    theta[n_north:] = geom["theta"][n_north - 1]  # padding ring, discarded
    n_ell = lmax - ms + 1
    offset = np.concatenate([[0], np.cumsum(n_ell * npair)])  # complex entries
    ncomp = 1 if spin == 0 else 2
    total = int(offset[-1])

    # Layout per component: order after order, rows L = M .. lmax of npair
    # complex entries (ring pairs), i.e. 2 npair float64 rings per row.
    # leg2alm addresses (L, M) of ring pair p at mstart + L * lstride with
    # lstride = npair, and mstart (the hypothetical L = 0 slot) must not
    # fall before the buffer: the data sit at the end of the buffer, after a
    # front part that is never written (so np.empty does not commit it).
    size = gl_legendre_table_capacity(ms, lmax, lmax_grid, spin) // ncomp
    if out is None:
        buf = np.empty((ncomp, size), dtype=np.complex128)
    else:
        if out.dtype != np.complex128 or out.ndim != 1 or out.size < ncomp * size:
            raise ValueError(
                f"out must be a 1-D complex128 array of at least {ncomp * size} "
                "elements (gl_legendre_table_capacity)"
            )
        # one fixed row per component, data at its end: however the blocks
        # vary, only the last written-size part of each row is ever touched
        size = out.size // ncomp
        buf = out[: ncomp * size].reshape(ncomp, size)
    start = size - total
    base = start + offset[:-1] - ms * npair
    # Q legs (1, i) on the ring pair; for spin 2 the U legs are zero.
    leg = np.zeros((ncomp, 2, ms.size), dtype=np.complex128)
    leg[0, 0] = 1.0
    leg[0, 1] = 1.0j

    def pairs(p0: int, p1: int) -> None:
        for p in range(p0, p1):
            ducc0.sht.leg2alm(
                leg=leg,
                lmax=lmax,
                theta=theta[2 * p : 2 * p + 2],
                spin=spin,
                mval=ms,
                mstart=base + p,
                lstride=npair,
                nthreads=1,
                alm=buf,
            )

    run_row_blocks(pairs, npair, nt)
    if spin == 2:  # B = -i (lambda^-(theta_1) + i lambda^-(theta_2))
        minus = buf[1, start:]

        def times_i(r0: int, r1: int) -> None:
            seg = minus[r0 * npair : r1 * npair]
            np.multiply(seg, 1.0j, out=seg)

        run_row_blocks(times_i, total // npair, nt)

    tables = []
    for i in range(ms.size):
        entry = []
        for c in range(ncomp):
            rows = buf[c, start + offset[i] : start + offset[i + 1]].view(np.float64)
            rows = rows.reshape(int(n_ell[i]), 2 * npair)[:, :n_north]
            even, odd = rows[0::2], rows[1::2]
            even.flags.writeable = False
            odd.flags.writeable = False
            entry += [even, odd]
        tables.append(tuple(entry))
    return tables


def gl_legendre_table_bytes(
    ms: np.ndarray, lmax: int, lmax_grid: int, spin: int = 0
) -> int:
    """Bytes of the tables :func:`gl_legendre_table` returns for ``ms`` (all
    it writes); spin 2 holds two functions and twice the bytes."""
    ms = np.asarray(ms, dtype=np.int64).ravel()
    npair = (gl_north_rings(lmax_grid) + 1) // 2
    ncomp = 1 if spin == 0 else 2
    return ncomp * int(np.sum(lmax - ms + 1)) * npair * 16


def gl_legendre_table_capacity(
    ms: np.ndarray, lmax: int, lmax_grid: int, spin: int = 0
) -> int:
    """``complex128`` elements of the buffer :func:`gl_legendre_table`
    builds the tables of ``ms`` in: the written tables and, in front of
    them, ``min(ms) (n_north + 1) // 2`` elements per component that are
    addressed but never written.  A reused buffer (``out``) must hold the
    capacity of every block given to it; per component, the largest
    written size plus ``lmax (n_north + 1) // 2`` suffices, and only the
    written part at the end of each component's share is ever touched."""
    ms = np.asarray(ms, dtype=np.int64).ravel()
    npair = (gl_north_rings(lmax_grid) + 1) // 2
    ncomp = 1 if spin == 0 else 2
    written = gl_legendre_table_bytes(ms, lmax, lmax_grid, spin) // 16 // ncomp
    return ncomp * (written + int(ms.min(initial=0)) * npair)


def gl_ring_modes(
    mask_alm: np.ndarray,
    lw: int,
    lmax_grid: int,
    nthreads: int | None = None,
) -> np.ndarray:
    r"""
    Azimuthal modes :math:`W_k(\theta_j)`, ``|k| <= lw``, of a real
    band-limited map on every ring of the GL grid, from its alm directly.

    ``W(theta_j, phi) = sum_k W_k(theta_j) e^{i k phi}`` exactly: the Legendre
    stage of the synthesis (``ducc0.sht.alm2leg``) for ``k >= 0``, and
    :math:`W_{-k} = \overline{W_k}` for a real map.  Unlike
    :func:`gl_mask_modes` (an FFT of the synthesised map), the modes do not
    alias when ``2 lw + 1 > nphi``, and only the ``2 lw + 1`` populated
    columns are stored.  ``W_0`` is the real part of the ``k = 0`` leg, as in
    a real-map synthesis, which ignores ``Im a_{L0}``.

    Returns
    -------
    ndarray
        Complex array of shape ``(ntheta, 2 lw + 1)``; column ``k + lw`` holds
        :math:`W_k`.  Agrees with ``gl_mask_modes`` to ~1e-16 where the latter
        does not alias.
    """
    import healpy as hp  # lazy, see utils/healpy_utils.py

    geom = _gl_geometry(lmax_grid)
    ks = np.arange(lw + 1, dtype=np.int64)
    leg = ducc0.sht.alm2leg(
        alm=_check_alm(mask_alm, lw, 0),
        lmax=lw,
        theta=geom["theta"],
        spin=0,
        mval=ks,
        mstart=hp.Alm.getidx(lw, 0, ks).astype(np.int64),
        nthreads=_nthreads(nthreads, lmax_grid),
    )[0]
    modes = np.empty((geom["ntheta"], 2 * lw + 1), dtype=np.complex128)
    modes[:, lw:] = leg
    modes[:, lw] = leg[:, 0].real
    modes[:, :lw] = np.conj(leg[:, :0:-1])
    return modes


# --------------------------------------------------------------------------- #
# Full-M <-> real-map-pair conversions (any spin; leading dims preserved)
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=16)
def _lm_tables(lmax: int) -> dict:
    """
    Index tables of :func:`full_from_pair` / :func:`pair_from_full` for one
    ``lmax``, computed once per process (they depend on nothing else; the
    ACC precompute used to rebuild them with ``healpy.Alm.getlm`` on every
    call, thousands of times per kernel pair).  All arrays are read-only.

    ``ell``, ``m``, ``sign``
        healpy's alm ordering and ``(-1)^m`` (:func:`_lm_index`).
    ``pos``
        indices (into the healpy ordering) of the ``m > 0`` entries.
    ``flat_p``, ``flat_m``
        flat index into a C-ordered ``(lmax+1, 2lmax+1)`` full-``M`` array
        of ``(L, +M)`` and ``(L, -M)``.
    ``col_p``, ``col_m_pos``, ``ell_pos``, ``sign_pos``
        the same as ``(row, column)`` pairs (for strided destinations), the
        ``-M`` ones restricted to ``pos``.
    """
    import healpy as hp  # lazy, see utils/healpy_utils.py

    ell, m = hp.Alm.getlm(lmax)
    sign = np.where(m % 2 == 0, 1.0, -1.0)
    pos = np.flatnonzero(m > 0)
    ncol = 2 * lmax + 1
    tables = {
        "ell": ell,
        "m": m,
        "sign": sign,
        "pos": pos,
        "flat_p": ell * ncol + lmax + m,
        "flat_m": ell * ncol + lmax - m,
        "col_p": lmax + m,
        "col_m_pos": lmax - m[pos],
        "ell_pos": ell[pos],
        "sign_pos": sign[pos],
    }
    tables["flat_m_pos"] = tables["flat_m"][pos]
    for array in tables.values():
        array.setflags(write=False)
    return tables


def _lm_index(lmax: int):
    """(l, m) arrays for healpy's alm ordering and the (-1)^m signs (cached,
    read-only; see :func:`_lm_tables`)."""
    tables = _lm_tables(lmax)
    return tables["ell"], tables["m"], tables["sign"]


def full_from_pair(
    r: np.ndarray, s: np.ndarray, lmax: int, out: np.ndarray | None = None
) -> np.ndarray:
    r"""
    Full-``M`` coefficients of the complex map ``Re + i Im`` from the healpy
    alms ``r`` (of ``Re``) and ``s`` (of ``Im``).

        x[L,  M] = r[L, M] + i s[L, M]                        (M >= 0)
        x[L, -M] = (-1)^M ( conj r[L, M] + i conj s[L, M] )   (M > 0)

    ``r`` and ``s`` may carry leading dimensions (e.g. ``(2, nalm)`` for an
    (E, B) pair), which are preserved.  Returns an array of shape
    ``(..., lmax+1, 2*lmax+1)`` indexed ``[..., L, M+lmax]``.

    ``out``, if given, is a complex128 array (or a strided view, e.g. one
    ``m`` of a packed block) of exactly that shape; it is overwritten --
    zeros where ``L < |M|`` -- and returned, so a caller that needs the
    coefficients inside a larger array pays no intermediate copy.  The
    values are the same either way.
    """
    r = np.asarray(r)
    s = np.asarray(s)
    t = _lm_tables(lmax)
    shape = r.shape[:-1] + (lmax + 1, 2 * lmax + 1)
    upper = r + 1j * s
    lower = t["sign_pos"] * (
        np.conj(np.take(r, t["pos"], axis=-1))
        + 1j * np.conj(np.take(s, t["pos"], axis=-1))
    )
    if out is None:
        full = np.zeros(shape, dtype=complex)
        flat = full.reshape(r.shape[:-1] + (-1,))
        flat[..., t["flat_p"]] = upper
        flat[..., t["flat_m_pos"]] = lower
        return full
    if out.shape != shape:
        raise ValueError(f"out must have shape {shape}; got {out.shape}")
    out[...] = 0
    out[..., t["ell"], t["col_p"]] = upper
    out[..., t["ell_pos"], t["col_m_pos"]] = lower
    return out


def pair_from_full(full: np.ndarray, lmax: int):
    """Inverse of :func:`full_from_pair`: healpy alms of ``Re`` and ``Im``
    (leading dimensions preserved)."""
    full = np.asarray(full)
    t = _lm_tables(lmax)
    if full.flags.c_contiguous and full.shape[-2:] == (lmax + 1, 2 * lmax + 1):
        flat = full.reshape(full.shape[:-2] + (-1,))
        xp = np.take(flat, t["flat_p"], axis=-1)
        xm = t["sign"] * np.conj(np.take(flat, t["flat_m"], axis=-1))
    else:
        ell, m = t["ell"], t["m"]
        xp = full[..., ell, lmax + m]
        xm = t["sign"] * np.conj(full[..., ell, lmax - m])
    r = 0.5 * (xp + xm)
    s = -0.5j * (xp - xm)
    return r, s


# --------------------------------------------------------------------------- #
# Complex-map transforms (spin 0 and spin 2)
# --------------------------------------------------------------------------- #
def _full_shape(lmax: int, spin: int) -> tuple[int, ...]:
    """Full-``M`` array shape: ``(lmax+1, 2lmax+1)`` for spin 0, with a
    leading (E, B) axis of length 2 for ``spin > 0``."""
    shape: tuple[int, ...] = (lmax + 1, 2 * lmax + 1)
    return shape if spin == 0 else (2,) + shape


def gl_synthesis_complex(
    alm_full: np.ndarray,
    lmax: int,
    lmax_grid: int,
    spin: int = 0,
    nthreads: int | None = None,
) -> np.ndarray:
    """
    Synthesise a complex map from full-``M`` coefficients.

    The complex-linear extension of :func:`gl_synthesis`: the coefficients are
    split with :func:`pair_from_full` into the healpy alms of two real maps,
    which are synthesised and recombined as ``Re + i Im``.  For ``spin=2`` the
    two real maps are (Q, U) pairs and the coefficients are (E, B); a complex
    (Q, U) pair is one real pair plus ``i`` times another, and (E, B) is linear
    in (Q, U), so the same split applies component-wise.

    Parameters
    ----------
    alm_full : ndarray
        Complex array of shape ``(lmax+1, 2*lmax+1)`` for spin 0, or
        ``(2, lmax+1, 2*lmax+1)`` (E, B) for spin 2, indexed
        ``[..., L, M+lmax]`` (module docstring); ``+M`` and ``-M`` entries are
        independent.
    lmax, lmax_grid, spin, nthreads
        As in :func:`gl_synthesis`.

    Returns
    -------
    ndarray
        Complex map of shape ``(ntheta, nphi)`` for spin 0, ``(2, ntheta,
        nphi)`` (Q, U) for spin 2.
    """
    alm_full = np.asarray(alm_full)
    shape = _full_shape(lmax, spin)
    if alm_full.shape != shape:
        raise ValueError(
            f"alm_full must have shape {shape} for spin {spin}; got {alm_full.shape}"
        )
    r, s = pair_from_full(alm_full, lmax)
    out = gl_synthesis(r, lmax, lmax_grid, spin=spin, nthreads=nthreads).astype(complex)
    if np.any(s):
        out += 1j * gl_synthesis(s, lmax, lmax_grid, spin=spin, nthreads=nthreads)
    return out


def gl_analysis_complex(
    map_complex: np.ndarray,
    lmax: int,
    lmax_grid: int,
    spin: int = 0,
    nthreads: int | None = None,
) -> np.ndarray:
    """
    Analyse a complex GL map into full-``M`` coefficients up to ``lmax``.

    Real and imaginary parts are analysed as two real maps (Q/U pairs for
    ``spin=2``) and recombined with :func:`full_from_pair`.  Exact inverse of
    :func:`gl_synthesis_complex` on a sufficiently large grid, and its exact
    adjoint for the inner products ``sum_{L, M} x conj(y)`` on coefficients and
    ``sum w f conj(g)`` (GL weights, summed over Q and U for spin 2) on maps.

    Parameters
    ----------
    map_complex : ndarray
        Complex map of shape ``(ntheta, nphi)`` for spin 0, ``(2, ntheta,
        nphi)`` (Q, U) for spin 2, on the grid ``lmax_grid``.
    lmax, lmax_grid, spin, nthreads
        As in :func:`gl_analysis`.

    Returns
    -------
    ndarray
        Complex array of shape ``(lmax+1, 2*lmax+1)`` for spin 0,
        ``(2, lmax+1, 2*lmax+1)`` (E, B) for spin 2, indexed ``[..., L, M+lmax]``.
    """
    map_complex = np.asarray(map_complex)
    r = gl_analysis(map_complex.real, lmax, lmax_grid, spin=spin, nthreads=nthreads)
    if np.iscomplexobj(map_complex) and np.any(map_complex.imag):
        s = gl_analysis(map_complex.imag, lmax, lmax_grid, spin=spin, nthreads=nthreads)
    else:
        s = np.zeros_like(r)
    return full_from_pair(r, s, lmax)


# --------------------------------------------------------------------------- #
# ACC mode-coupling integrals (spin 0 and spin 2)
# --------------------------------------------------------------------------- #
def spin_weighted_integrals_gl(
    mask_alm: np.ndarray,
    lw: int,
    ell: int,
    m: int,
    lmax_out: int,
    lmax_grid: int | None = None,
    nthreads: int | None = None,
    spin: int = 0,
) -> np.ndarray:
    r"""
    Full-``M`` coefficients of the masked-harmonic integral :math:`I_{\ell m, LM}`
    on a Gauss-Legendre grid (the ACC precompute's central quantity, Camphuis et
    al. 2022 Fig. 8):

    .. math::

        I_{\ell m, LM} = \int W(\hat n)\, Y_{\ell m}(\hat n)\, Y^*_{LM}(\hat n)
        \, d\Omega ,

    computed exactly for a band-limited mask by synthesising :math:`W` and the
    single mode :math:`Y_{\ell m}` on a shared GL grid, multiplying pointwise,
    and analysing the product.  In operator language this is one column of
    the pseudo-alm operator ``K = S^dagger W S``: the result is ``K e`` for
    the unit vector ``e`` at ``(ell, m)``.

    This is the *two-SHT reference route*: about five real transforms per
    call (mask synthesis, two for the complex single mode, two analyses),
    the mask re-synthesised every time.  The ACC precompute produces the
    same numbers (to rounding, same grid and same quadrature) with
    :func:`banded_integrals_gl` at full band and mask modes shared per grid
    (:data:`cmbcov.approximations.acc._GL_PRODUCER`); this function is kept
    as the reference that route is tested against, and for callers that
    need a single integral.

    For ``spin=2`` the unit vector is an **E-mode** unit vector (``B = 0``) and
    ``K_2 = S_2^dagger W S_2`` acts on (E, B) pairs: the result holds both the
    E-mode response and the E->B leakage of the mask,

    .. math::

        u^E_{LM} = (K_2 e^E_{\ell m})^E_{LM}, \qquad
        u^B_{LM} = (K_2 e^E_{\ell m})^B_{LM} ,

    which are the ``E`` and ``B`` components of the HEALPix path's
    :func:`cmbcov.mask.MaskWlm.compute_spin_weighted_integrals`
    (that path builds one (T, Q, U) map from a unit alm with ``T = E = 1`` and
    scales its Q/U maps by ``sqrt(2)``; this function carries no such
    factor).

    Parameters
    ----------
    mask_alm : ndarray
        healpy-ordered alm of the mask, truncated at ``lw`` (e.g. from
        ``hp.map2alm(mask, lmax=lw, iter=10)``).
    lw : int
        Band-limit of ``mask_alm``.
    ell, m : int
        Degree and order of the harmonic being integrated against; ``|m| <= ell``.
    lmax_out : int
        Output band-limit ``L = 0..lmax_out`` of the returned integrals.
    lmax_grid : int, optional
        GL grid band-limit.  ``None`` uses the minimal exact grid derived
        below; a smaller grid aliases, a larger one only costs time.
    nthreads : int, optional
        Threads for ducc0.
    spin : int
        Spin of the harmonic being integrated: 0 (temperature, the unit
        vector is :math:`Y_{\ell m}`) or 2 (polarisation, the unit vector is
        the E-mode :math:`e^E_{\ell m}`).

    Returns
    -------
    ndarray
        ``spin=0``: complex array of shape ``(lmax_out + 1, 2 lmax_out + 1)``,
        full-``M`` layout (module docstring), column ``lmax_out + M``.
        ``spin=2``: shape ``(2, lmax_out + 1, 2 lmax_out + 1)`` holding the
        (E, B) coefficients ``(u^E, u^B)``; all zero for ``ell < 2`` (no E
        modes exist there).

    Notes
    -----
    **Grid size.**  The integrand :math:`W \, Y_{\ell m} \, Y^*_{LM}` has
    degree ``lw`` (mask) ``+ ell`` (one harmonic) ``+ lmax_out`` (the other,
    worst case ``L = lmax_out``) in :math:`\cos\theta` after the :math:`\phi`
    integration collapses the azimuthal factors (same argument as
    :func:`gl_minimal_lmax`, generalised from two equal harmonic degrees to
    two different ones ``ell`` and ``lmax_out``).  Exactness needs
    ``2 Lg + 1 >= lw + ell + lmax_out``, i.e.

        Lg_min = ceil((lw + ell + lmax_out - 1) / 2)

    which reduces to :func:`gl_minimal_lmax` ``(lmax_out, lw)`` when
    ``ell == lmax_out``.  The grid must also be at least as large as either
    output band-limit (``analysis_2d`` requires ``lmax_grid >= lmax`` and
    synthesis of :math:`Y_{\ell m}` needs ``lmax_grid >= ell``), so the
    default is ``max(Lg_min, ell, lmax_out)``.  The same rule holds for spin
    2: a spin-weighted harmonic :math:`{}_sY_{\ell m}` is
    :math:`d^\ell_{m,-s}(\theta) e^{im\phi}` up to a constant, and the product
    of two Wigner-d functions with equal lower indices,
    :math:`d^\ell_{m,-s} d^L_{m,-s}`, is a polynomial of degree ``ell + L``
    in :math:`\cos\theta` (the half-angle prefactors pair up into
    ``(1 -+ cos theta)`` powers), exactly as for spin 0.
    """
    if spin not in (0, 2):
        raise ValueError(f"spin must be 0 or 2, got {spin}")
    if ell < 0 or lmax_out < 0:
        raise ValueError("ell and lmax_out must be non-negative")
    if abs(m) > ell:
        raise ValueError(f"|m| must be <= ell, got m={m}, ell={ell}")

    if lmax_grid is None:
        lg_min = math.ceil((lw + ell + lmax_out - 1) / 2)
        lmax_grid = max(lg_min, ell, lmax_out)
    elif lmax_grid < max(ell, lmax_out):
        raise ValueError(
            f"lmax_grid={lmax_grid} must be >= max(ell, lmax_out) = "
            f"{max(ell, lmax_out)}"
        )

    if spin == 2 and ell < 2:
        return np.zeros(_full_shape(lmax_out, 2), dtype=complex)

    w_map = gl_synthesis(mask_alm, lw, lmax_grid, nthreads=nthreads)

    # Single-mode complex map: full-M alm with one unit entry (the E entry
    # for spin 2), synthesised on the shared grid (same construction as
    # cmbcov.exact._correlation_columns_gl's basis vector).
    e = np.zeros(_full_shape(ell, spin), dtype=complex)
    if spin == 0:
        e[ell, ell + m] = 1.0
    else:
        e[0, ell, ell + m] = 1.0
    y = gl_synthesis_complex(e, ell, lmax_grid, spin=spin, nthreads=nthreads)

    return gl_analysis_complex(
        w_map * y, lmax_out, lmax_grid, spin=spin, nthreads=nthreads
    )


# --------------------------------------------------------------------------- #
# Banded ACC integrals through Legendre transforms (spin 0 and spin 2)
# --------------------------------------------------------------------------- #
def gl_mask_modes(
    mask_alm: np.ndarray,
    lw: int,
    lmax_grid: int,
    nthreads: int | None = None,
) -> np.ndarray:
    r"""
    Azimuthal Fourier modes of the mask on the rings of a GL grid.

    .. math::

        W_{m_3}(\theta_j) = \frac{1}{n_\phi} \sum_{k=0}^{n_\phi - 1}
        W(\theta_j, \phi_k)\, e^{-i m_3 \phi_k}
        = \frac{1}{2\pi} \int_0^{2\pi} W(\theta_j, \phi)\, e^{-i m_3 \phi}\, d\phi ,

    the coefficient of :math:`e^{i m_3 \phi}` on ring ``j`` (exact for
    ``|m_3| < n_\phi / 2``, i.e. always for a mask of band-limit ``lw`` when
    ``lmax_grid >= lw``; on the smaller grids that
    :func:`spin_weighted_integrals_gl` may use, mode ``m_3`` and
    ``m_3 - n_\phi`` alias onto the same column, exactly as they do inside
    ``analysis_2d`` -- the banded integrals reproduce the two-SHT result mode
    by mode either way).

    This is the input :func:`banded_integrals_gl` needs from the mask: one
    synthesis and one FFT per grid, to be shared by every ``(ell, m)`` on it
    (the ACC precompute keeps one per grid for the whole run, for the
    central and the primed side alike).

    Parameters
    ----------
    mask_alm, lw
        As in :func:`spin_weighted_integrals_gl`.
    lmax_grid : int
        Grid band-limit, see :func:`gl_shape`.
    nthreads : int, optional
        Threads for ducc0.

    Returns
    -------
    ndarray
        Complex array of shape ``(ntheta, nphi)``; column ``m_3 mod nphi``
        holds :math:`W_{m_3}(\theta)`.
    """
    w_map = gl_synthesis(mask_alm, lw, lmax_grid, nthreads=nthreads)
    return np.fft.fft(w_map, axis=1) / w_map.shape[1]


def _unit_mode_legs(
    ell: int, m: int, spin: int, theta: np.ndarray, nthreads: int
) -> np.ndarray:
    r"""
    Ring profiles of the single-mode complex map of :func:`spin_weighted_integrals_gl`.

    For spin 0 the unit map is :math:`Y_{\ell m} = \lambda_{\ell m}(\theta)
    e^{i m \phi}` and this returns :math:`\lambda_{\ell m}(\theta)`; for
    spin 2 the unit E-mode vector synthesises to a complex (Q, U) pair whose
    only azimuthal mode is ``m``, :math:`(Q, U) = (q_{\ell m}(\theta),
    u_{\ell m}(\theta)) e^{i m \phi}`, and this returns ``(q, u)``.

    ``ducc0.sht.alm2leg`` evaluates the ``m >= 0`` profile of a *real* map
    with unit coefficient at ``(ell, |m|)``; the complex unit map at ``m < 0``
    is obtained from the ``+|m|`` profile through
    :func:`pair_from_full` / :func:`full_from_pair`, which give

        leg(ell, -|m|) = (-1)^m conj(leg(ell, |m|))

    (for spin 0 this is the usual :math:`\lambda_{\ell,-m} = (-1)^m
    \lambda_{\ell m}`).  Checked against the FFT of
    :func:`gl_synthesis_complex` to 1e-16 for both spins.

    Returns an array of shape ``(ncomp, ntheta)``, ``ncomp = 1`` for spin 0
    and ``2`` (Q, U) for spin 2.
    """
    import healpy as hp  # lazy, see utils/healpy_utils.py

    am = abs(m)
    ncomp = 1 if spin == 0 else 2
    alm = np.zeros((ncomp, hp.Alm.getsize(ell)), dtype=np.complex128)
    alm[0, hp.Alm.getidx(ell, ell, am)] = 1.0  # E entry for spin 2
    leg = ducc0.sht.alm2leg(
        alm=alm,
        lmax=ell,
        theta=theta,
        spin=spin,
        mval=np.array([am], dtype=np.int64),
        mstart=np.array([hp.Alm.getidx(ell, 0, am)], dtype=np.int64),
        nthreads=nthreads,
    )[:, :, 0]
    if m < 0:
        leg = (-1.0) ** am * np.conj(leg)
    return leg


def _fill_legs(
    legs: np.ndarray,
    wu: np.ndarray,
    mask_modes: np.ndarray,
    first: int,
    phase: np.ndarray | None,
    nthreads: int,
) -> None:
    r"""
    ``legs[c, j, i] = wu[c, j] * mask_modes[j, (first + i) mod nphi]``, then
    for the ``M < 0`` block (``phase`` given) ``legs[0] *= phase`` and, for
    spin 2, ``legs[1] *= -phase``: the legs of :func:`banded_integrals_gl`.

    The mask columns of a run of consecutive orders are at most two slices
    (``nphi > 2 lmax_out``), multiplied straight into place without a
    gather.  The rows (rings) are split into contiguous blocks over up to
    ``nthreads`` workers (:func:`~.utils.threading_utils.run_row_blocks`);
    every entry is the same one or two products as on one thread, so the
    result is bit-identical for any thread count.
    """
    ncomp, ntheta, size = legs.shape
    nphi = mask_modes.shape[1]
    neg_phase = None if phase is None or ncomp == 1 else -phase

    def rows(r0: int, r1: int) -> None:
        j = 0
        while j < size:
            k = (first + j) % nphi
            n = min(size - j, nphi - k)
            for c in range(ncomp):
                np.multiply(
                    wu[c, r0:r1, None],
                    mask_modes[r0:r1, k : k + n],
                    out=legs[c, r0:r1, j : j + n],
                )
            j += n
        if phase is not None:
            legs[0, r0:r1] *= phase
            if neg_phase is not None:
                legs[1, r0:r1] *= neg_phase

    run_row_blocks(rows, ntheta, row_block_workers(legs.size, nthreads))


def banded_integrals_gl(
    mask_alm: np.ndarray,
    lw: int,
    ell: int,
    m: int,
    lmax_out: int,
    m_band: int,
    lmax_grid: int | None = None,
    nthreads: int | None = None,
    spin: int = 0,
    mask_modes: np.ndarray | None = None,
) -> np.ndarray:
    r"""
    :func:`spin_weighted_integrals_gl` restricted to the azimuthal band
    ``|M - m| <= m_band``, computed by Legendre transforms instead of SHTs.

    The two-SHT route synthesises :math:`W Y_{\ell m}` on the grid and
    analyses the product for *every* order ``M``.  But the product has a
    single azimuthal structure: the mode-``M`` ring profile of
    :math:`W Y_{\ell m}` is :math:`W_{M-m}(\theta)\, \lambda_{\ell m}(\theta)`
    (:func:`gl_mask_modes` times :func:`_unit_mode_legs`), so with the ring
    weights :math:`\bar w(\theta) = 2\pi w^{GL}(\theta)`

    .. math::

        I_{\ell m, LM} = \sum_\theta \bar w(\theta)\, W_{M-m}(\theta)\,
        \lambda_{\ell m}(\theta)\, \lambda_{LM}(\theta) ,

    which is the Gaunt sum :math:`\sum_{\ell_3} w_{\ell_3, M-m}
    \int Y_{\ell_3, M-m} Y_{\ell m} Y^*_{LM}`
    (``gaunt_sum_integrals`` in ``tests/reference/gaunt.py``)
    evaluated by quadrature -- exact on the same grid as the two-SHT route,
    since it is the same product and the same GL rule, only with the
    :math:`\phi` sum done analytically.  The sum over :math:`\theta` is one
    Legendre transform per ``M`` (``ducc0.sht.leg2alm``, the adjoint of
    ``alm2leg``: it applies :math:`\lambda_{LM}` with no weights, hence the
    :math:`\bar w` in the leg), so restricting ``M`` to a band restricts the
    cost proportionally.  When the mask is rotated so that its centroid sits
    at the pole, :math:`W_{m_3}` is concentrated at small :math:`|m_3|` and a
    band ``m_band << lmax_out`` carries all of :math:`I` (see
    :mod:`cmbcov.term_selection`).

    **Negative M.**  ``leg2alm`` takes orders ``M >= 0`` only; the ``M < 0``
    entries of the full-``M`` layout follow from
    :math:`\lambda_{L,-M} = (-1)^M \lambda_{LM}` (spin 0) and, for spin 2,
    from :math:`{}_{s}\lambda_{L,-M} = (-1)^M {}_{-s}\lambda_{LM}`, which
    swaps the roles of the even and odd spin-weighted Legendre combinations:
    the ``M < 0`` block is ``leg2alm`` at ``|M|`` applied to the legs
    ``(G_Q, -G_U)``, with the E output multiplied by :math:`(-1)^{|M|}` and
    the B output by :math:`-(-1)^{|M|}`.  This is the same pairing that
    :func:`full_from_pair` encodes for complex maps, written out for one
    order at a time; verified numerically against
    :func:`spin_weighted_integrals_gl` to 2e-16 for both spins.

    **Layout.**  Each ``leg2alm`` call (one for ``M >= 0``, one for
    ``M < 0``) writes its orders straight into their columns of the
    full-``M`` output (``mstart = M + lmax_out``, ``lstride = 2 lmax_out +
    1``), the common sign :math:`(-1)^{|M|}` applied to the legs; no
    intermediate ``(M, L)`` array, transpose or :func:`full_from_pair` is
    involved.  The legs themselves are multiplied into place from at most two
    column slices of ``mask_modes`` per sign (no gather).  Bit-identical to
    the earlier gather-and-transpose form of this function.

    **Default ACC producer.**  At full band this is how
    :mod:`cmbcov.approximations.acc` produces every GL coefficient when no
    term selection is asked for; :func:`spin_weighted_integrals_gl` is the
    reference it is tested against (``tests/test_acc_gl_producer.py``).

    Parameters
    ----------
    mask_alm, lw, ell, m, lmax_out, lmax_grid, nthreads, spin
        As in :func:`spin_weighted_integrals_gl`; the default grid rule is
        identical, so the two functions agree to round-off in band.
    m_band : int
        Half-width of the azimuthal band: entries with ``|M - m| > m_band``
        are returned as exact zeros.  ``m_band >= lmax_out + |m|`` (in
        particular ``m_band >= 2 lmax_out`` whenever ``|m| <= lmax_out``)
        reproduces the full result.
    mask_modes : ndarray, optional
        :func:`gl_mask_modes` ``(mask_alm, lw, lmax_grid)`` on the *same*
        grid, precomputed once per ``(ell, lmax_out)`` and shared across
        ``m``; when given, ``mask_alm`` is not used.

    Returns
    -------
    ndarray
        Same shape and layout as :func:`spin_weighted_integrals_gl`:
        ``(lmax_out + 1, 2 lmax_out + 1)`` for spin 0, ``(2, ...)`` (E, B)
        for spin 2; all zero for ``spin=2, ell < 2``.

    Notes
    -----
    **Cost.**  ``leg2alm`` over ``n_M`` orders is
    :math:`O(n_M\, n_\theta\, \ell_{max})`, against
    :math:`O(n_\theta \ell_{max}^2 + n_\theta n_\phi \log n_\phi)` for
    each transform of the two-SHT route (about five per call: mask, the
    complex single mode, the complex product), and the band restricts
    ``n_M`` to ``2 m_band + 1``.  At full band, ``nside`` 256
    (``lmax_out = 511``, ``lw = 875``, grid 818), measured on a 14-core
    laptop inside the ACC precompute: 11.9 ms per call (spin 0 and 2
    averaged) against 49.7 ms for :func:`spin_weighted_integrals_gl`, 4.2x;
    single-threaded 3.3x (80-87 ms against 267-286 ms per ``m``, both
    spins).  About 60% of it was ``leg2alm`` (threaded), the rest the
    then single-threaded construction of the legs, which
    :func:`_fill_legs` now splits over rings on the same ``nthreads``
    (3-4x faster, bit-identical).  The mask
    synthesis is hoisted out through ``mask_modes``; without it, it is
    repeated per call as in the two-SHT route.
    """
    if spin not in (0, 2):
        raise ValueError(f"spin must be 0 or 2, got {spin}")
    if ell < 0 or lmax_out < 0:
        raise ValueError("ell and lmax_out must be non-negative")
    if abs(m) > ell:
        raise ValueError(f"|m| must be <= ell, got m={m}, ell={ell}")
    if m_band < 0:
        raise ValueError(f"m_band must be non-negative, got {m_band}")

    if lmax_grid is None:
        lg_min = math.ceil((lw + ell + lmax_out - 1) / 2)
        lmax_grid = max(lg_min, ell, lmax_out)
    elif lmax_grid < max(ell, lmax_out):
        raise ValueError(
            f"lmax_grid={lmax_grid} must be >= max(ell, lmax_out) = "
            f"{max(ell, lmax_out)}"
        )

    ncomp = 1 if spin == 0 else 2
    out = np.zeros((ncomp, lmax_out + 1, 2 * lmax_out + 1), dtype=complex)
    if spin == 2 and ell < 2:
        return out.reshape(_full_shape(lmax_out, spin))

    geom = _gl_geometry(lmax_grid)
    ntheta, nphi = geom["ntheta"], geom["nphi"]
    nt = _nthreads(nthreads, lmax_grid)
    if mask_modes is None:
        mask_modes = gl_mask_modes(mask_alm, lw, lmax_grid, nthreads=nthreads)
    elif mask_modes.shape != (ntheta, nphi):
        raise ValueError(
            f"mask_modes must have shape {(ntheta, nphi)} for lmax_grid "
            f"{lmax_grid}; got {mask_modes.shape}"
        )
    # wbar(theta) unit_c(theta): the ring profile of the weighted single mode.
    wu = geom["wbar"] * _unit_mode_legs(ell, m, spin, geom["theta"], nt)
    m_lo, m_hi = max(-lmax_out, m - m_band), min(lmax_out, m + m_band)
    ncol = 2 * lmax_out + 1
    flat = out.reshape(ncomp, (lmax_out + 1) * ncol)  # a view: out is contiguous

    for lo, hi, negative in ((max(m_lo, 0), m_hi, False), (m_lo, min(m_hi, -1), True)):
        if lo > hi:
            continue
        Ms = np.arange(lo, hi + 1)
        # legs[c, theta, M] = wbar W_{M-m} unit_c -- the mode-M ring profile of
        # the weighted product map W * y (_fill_legs, threaded over rings).
        # The M < 0 rule (docstring): legs (G_Q, -G_U) at |M|, E output times
        # (-1)^|M|, B output times -(-1)^|M|.  The common (-1)^|M| is applied
        # to the legs (leg2alm is linear in them, and a sign is exact), the B
        # sign to the output below.
        legs = np.empty((ncomp, ntheta, Ms.size), dtype=np.complex128)
        _fill_legs(
            legs,
            wu,
            mask_modes,
            lo - m,
            (-1.0) ** np.abs(Ms) if negative else None,
            nt,
        )
        # leg2alm writes each order straight into its full-M column
        # (index mstart + L * lstride = L * ncol + M + lmax_out); entries with
        # L < |M| are not touched and stay zero.
        ducc0.sht.leg2alm(
            leg=legs,
            lmax=lmax_out,
            theta=geom["theta"],
            spin=spin,
            mval=np.abs(Ms).astype(np.int64),
            mstart=(Ms + lmax_out).astype(np.int64),
            lstride=ncol,
            nthreads=nt,
            alm=flat,
        )
        if negative and spin == 2:
            out[1][:, lo + lmax_out : hi + lmax_out + 1] *= -1.0
    return out.reshape(_full_shape(lmax_out, spin))


def cross_spectrum_full_m(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    r"""
    Full complex cross-spectrum :math:`\sum_M x_{LM} \, \bar y_{LM}` per ``L``.

    Unlike the brute-force cross-spectrum ``alm2cl(x[0], y[0]) + alm2cl(x[1],
    y[1])`` on the ``(2, 3, nalm)`` integral sets of
    :meth:`~cmbcov.mask.MaskWlm.compute_spin_weighted_integrals`
    (which returns only the real part, and silently truncates a 1-D T-only
    input to its first ``nspec`` multipoles rather than ``nspec`` channels),
    this operates directly on the full-``M`` layout (module docstring) and
    keeps the complex value.  The ACC precompute expands those HEALPix sets
    with :func:`full_from_pair` and contracts them in this form too.

    Parameters
    ----------
    x, y : ndarray
        Full-``M`` coefficient arrays of identical shape ``(lmax+1, 2 lmax+1)``.

    Returns
    -------
    ndarray
        Complex array of shape ``(lmax+1,)``.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape != y.shape:
        raise ValueError(
            f"x and y must have the same shape; got {x.shape} and {y.shape}"
        )
    return np.sum(x * np.conj(y), axis=1)
