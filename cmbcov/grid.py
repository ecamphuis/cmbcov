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

import math
import os

import ducc0
import numpy as np

from .utils.threading_utils import _env_nthreads, get_optimal_nthreads

__all__ = [
    "gl_minimal_lmax",
    "gl_step1_minimal_lmax",
    "gl_shape",
    "gl_thetas",
    "gl_quadrature_weights",
    "gl_synthesis",
    "gl_analysis",
    "gl_synthesis_single_m",
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
        }
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
# Full-M <-> real-map-pair conversions (any spin; leading dims preserved)
# --------------------------------------------------------------------------- #
def _lm_index(lmax: int):
    """(l, m) arrays for healpy's alm ordering and the (-1)^m signs."""
    import healpy as hp  # lazy, see utils/healpy_utils.py

    ell, m = hp.Alm.getlm(lmax)
    sign = np.where(m % 2 == 0, 1.0, -1.0)
    return ell, m, sign


def full_from_pair(r: np.ndarray, s: np.ndarray, lmax: int) -> np.ndarray:
    r"""
    Full-``M`` coefficients of the complex map ``Re + i Im`` from the healpy
    alms ``r`` (of ``Re``) and ``s`` (of ``Im``).

        x[L,  M] = r[L, M] + i s[L, M]                        (M >= 0)
        x[L, -M] = (-1)^M ( conj r[L, M] + i conj s[L, M] )   (M > 0)

    ``r`` and ``s`` may carry leading dimensions (e.g. ``(2, nalm)`` for an
    (E, B) pair), which are preserved.  Returns an array of shape
    ``(..., lmax+1, 2*lmax+1)`` indexed ``[..., L, M+lmax]``.
    """
    r = np.asarray(r)
    s = np.asarray(s)
    ell, m, sign = _lm_index(lmax)
    full = np.zeros(r.shape[:-1] + (lmax + 1, 2 * lmax + 1), dtype=complex)
    full[..., ell, lmax + m] = r + 1j * s
    pos = m > 0
    full[..., ell[pos], lmax - m[pos]] = sign[pos] * (
        np.conj(r[..., pos]) + 1j * np.conj(s[..., pos])
    )
    return full


def pair_from_full(full: np.ndarray, lmax: int):
    """Inverse of :func:`full_from_pair`: healpy alms of ``Re`` and ``Im``
    (leading dimensions preserved)."""
    full = np.asarray(full)
    ell, m, sign = _lm_index(lmax)
    xp = full[..., ell, lmax + m]
    xm = sign * np.conj(full[..., ell, lmax - m])
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
    synthesis and one FFT per grid, to be shared by every ``(ell, m)`` on it.

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
    each of the two SHTs, so the saving is roughly
    ``(2 m_band + 1) / (2 lmax_out + 1)`` (measured 5x at ``nside`` 256 for
    the compiled full band, more in proportion for narrow bands).  The mask
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
    wbar = ducc0.misc.GL_weights(ntheta, nphi) * nphi  # 2 pi w_GL per ring

    unit = _unit_mode_legs(ell, m, spin, geom["theta"], nt)  # (ncomp, ntheta)
    Ms = np.arange(max(-lmax_out, m - m_band), min(lmax_out, m + m_band) + 1)
    if Ms.size == 0:
        return out.reshape(_full_shape(lmax_out, spin))
    # G[c, theta, M] = wbar W_{M-m} unit_c  -- the mode-M ring profile of the
    # weighted product map W * y.
    G = (wbar * unit)[:, :, None] * mask_modes[:, (Ms - m) % nphi][None, :, :]

    for negative in (False, True):
        sel = Ms < 0 if negative else Ms >= 0
        if not sel.any():
            continue
        Mabs = np.abs(Ms[sel]).astype(np.int64)
        n_m = Mabs.size
        legs = np.ascontiguousarray(G[:, :, sel])
        if negative and spin == 2:
            legs[1] *= -1.0
        alm = np.zeros((ncomp, n_m * (lmax_out + 1)), dtype=np.complex128)
        ducc0.sht.leg2alm(
            leg=legs,
            lmax=lmax_out,
            theta=geom["theta"],
            spin=spin,
            mval=Mabs,
            mstart=(np.arange(n_m) * (lmax_out + 1)).astype(np.int64),
            lstride=1,
            nthreads=nt,
            alm=alm,
        )
        alm = alm.reshape(ncomp, n_m, lmax_out + 1)  # [c, M, L]
        phase = (-1.0) ** Mabs if negative else np.ones(n_m)
        cols = Ms[sel] + lmax_out
        out[0][:, cols] = (alm[0] * phase[:, None]).T
        if spin == 2:
            out[1][:, cols] = (alm[1] * (-phase if negative else phase)[:, None]).T
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
