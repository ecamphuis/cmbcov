"""
Spherical-harmonic primitives.

Low-level transforms shared by the mask handling and the alm containers:
a ducc0-backed map-to-alm transform (spin-0 for temperature, genuine spin-2 for
Q/U to E/B) and the complex spin-weighted spherical harmonics used to build the
mode-coupling integrals of Camphuis et al. (2022) Eq. (2).
"""

import functools
import warnings

import ducc0
import numpy as np

from .utils.healpy_utils import ensure_ducc0_compatible_dtype, get_nside_from_ell
from .utils.threading_utils import get_optimal_nthreads

__all__ = [
    "DEFAULT_MAP2ALM_ITER",
    "ducc0_map2alm",
    "ducc0_alm2map",
    "alm2cl",
    "almxfl",
    "cplx_spin_weighted_ylm",
]


#: Jacobi refinement iterations of :func:`ducc0_map2alm` when ``iter`` is not
#: given: healpy's own ``hp.map2alm`` default. The ACC coupling cache
#: records this value for ``grid="healpix"`` kernels (``acc_cache``).
DEFAULT_MAP2ALM_ITER = 3


@functools.lru_cache(maxsize=32)
def _sht_geometry(nside: int) -> dict:
    """HEALPix RING-scheme ring geometry, as used by ``ducc0.sht``."""
    return ducc0.healpix.Healpix_Base(nside=nside, scheme="RING").sht_info()


def _pixel_volume(nside: int) -> float:
    return 4 * np.pi / (12 * nside**2)


def ducc0_map2alm(
    maps: np.ndarray,
    lmax: int,
    nside: int | None = None,
    pol: bool = True,
    nthreads: int | None = None,
    iter: int = DEFAULT_MAP2ALM_ITER,
) -> np.ndarray:
    """
    Convert HEALPix maps to spherical harmonic coefficients using ducc0.

    This function replaces ``hp.map2alm`` with ducc0.sht.adjoint_synthesis
    for better performance and data type compatibility, including healpy's
    Jacobi refinement (``iter``).

    Parameters
    ----------
    maps : np.ndarray
        Input HEALPix map(s). Shape (npix,) for single map or (3, npix) for TQU
    lmax : int
        Maximum multipole moment
    nside : int, optional
        HEALPix resolution parameter (auto-detected if None)
    pol : bool, optional
        Whether to treat as polarization maps (default: True)
    nthreads : int, optional
        Number of threads for computation (default: auto-detect)
    iter : int, optional
        Number of Jacobi refinement iterations, matching
        ``hp.map2alm(..., iter=iter)``: starting from the plain (pixel-volume
        weighted) adjoint synthesis, each iteration adds the adjoint of the
        residual ``map - synthesis(alm)`` (default: 3, healpy's default;
        ``iter=0`` is the plain adjoint).

    Returns
    -------
    np.ndarray
        Spherical harmonic coefficients
    """
    import healpy as hp  # lazy, see utils/healpy_utils.py

    # Ensure proper data type
    maps = ensure_ducc0_compatible_dtype(maps)

    # Handle input shape
    if maps.ndim == 1:
        maps = maps[np.newaxis, :]

    if nside is None:
        nside = hp.get_nside(maps[0])

    if nthreads is None:
        nthreads = get_optimal_nthreads(nside)

    # Get ducc0 geometry
    geom = _sht_geometry(nside)
    pixel_volume = _pixel_volume(nside)

    def _adjoint_spin0(m: np.ndarray) -> np.ndarray:
        a = ducc0.sht.adjoint_synthesis(
            map=m,
            theta=geom["theta"],
            nphi=geom["nphi"],
            phi0=geom["phi0"],
            lmax=lmax,
            mmax=lmax,
            ringstart=geom["ringstart"],
            nthreads=nthreads,
            spin=0,
        )
        return a[0] * pixel_volume

    def _synth_spin0(a: np.ndarray) -> np.ndarray:
        out = ducc0.sht.synthesis(
            alm=a[np.newaxis, :],
            theta=geom["theta"],
            nphi=geom["nphi"],
            phi0=geom["phi0"],
            lmax=lmax,
            mmax=lmax,
            ringstart=geom["ringstart"],
            nthreads=nthreads,
            spin=0,
        )
        return out[0]

    def _adjoint_spin2(qu: np.ndarray) -> np.ndarray:
        a = ducc0.sht.adjoint_synthesis(
            map=qu.reshape(1, 2, -1),
            theta=geom["theta"],
            nphi=geom["nphi"],
            phi0=geom["phi0"],
            lmax=lmax,
            mmax=lmax,
            ringstart=geom["ringstart"],
            nthreads=nthreads,
            spin=2,
        )
        return a[0] * pixel_volume

    def _synth_spin2(eb: np.ndarray) -> np.ndarray:
        out = ducc0.sht.synthesis(
            alm=eb.reshape(1, 2, -1),
            theta=geom["theta"],
            nphi=geom["nphi"],
            phi0=geom["phi0"],
            lmax=lmax,
            mmax=lmax,
            ringstart=geom["ringstart"],
            nthreads=nthreads,
            spin=2,
        )
        return out[0]

    if pol and maps.shape[0] == 3:
        # Temperature (spin-0)
        temp_map = maps[0:1]
        temp_alm = _adjoint_spin0(temp_map)
        for _ in range(iter):
            resid = temp_map[0] - _synth_spin0(temp_alm)
            temp_alm = temp_alm + _adjoint_spin0(resid[np.newaxis, :])

        # Polarization Q, U -> E, B (spin-2)
        pol_maps = np.array([maps[1], maps[2]])  # [Q, U]
        pol_alm = _adjoint_spin2(pol_maps)
        for _ in range(iter):
            resid = pol_maps - _synth_spin2(pol_alm)
            pol_alm = pol_alm + _adjoint_spin2(resid)

        # ducc0 returns [E, B] for spin-2
        return np.array([temp_alm, pol_alm[0], pol_alm[1]])

    else:
        # Single map (temperature only)
        if maps.shape[0] > 1:
            maps = maps[0:1]  # Take first map only

        alm = _adjoint_spin0(maps)
        for _ in range(iter):
            resid = maps[0] - _synth_spin0(alm)
            alm = alm + _adjoint_spin0(resid[np.newaxis, :])

        return alm  # Return single alm array


def ducc0_alm2map(
    alm: np.ndarray,
    nside: int,
    lmax: int | None = None,
    mmax: int | None = None,
    pol: bool = False,
    nthreads: int | None = None,
) -> np.ndarray:
    """
    Convert spherical harmonic coefficients to a HEALPix map using ducc0.

    This function replaces ``hp.alm2map`` with ``ducc0.sht.synthesis`` on the
    same RING-scheme ring geometry used by :func:`ducc0_map2alm`.

    Parameters
    ----------
    alm : np.ndarray
        Spherical harmonic coefficients in healpy's m-major layout. Shape
        ``(nalm,)`` for a single spin-0 field (synthesised regardless of
        ``pol``), or ``(3, nalm)`` for ``pol=True`` -- interpreted as
        (T, E, B) and synthesised to (T, Q, U) using a spin-0 transform for T
        and a genuine spin-2 transform for (E, B) -> (Q, U).
    nside : int
        HEALPix resolution parameter of the output map.
    lmax : int, optional
        Maximum multipole moment. Inferred from the alm array size if
        omitted (using ``mmax`` if given, else assuming ``mmax = lmax``).
    mmax : int, optional
        Maximum m moment. Defaults to ``lmax``.
    pol : bool, optional
        If True and ``alm`` has shape ``(3, nalm)``, synthesise (T, Q, U)
        from (T, E, B) -- matches ``hp.alm2map(alm, nside, pol=True)``.
    nthreads : int, optional
        Number of threads for computation (default: auto-detect)

    Returns
    -------
    np.ndarray
        ``(npix,)`` map for a 1-D input, or ``(3, npix)`` (T, Q, U) map for
        ``pol=True`` with a ``(3, nalm)`` input.
    """
    import healpy as hp  # lazy, see utils/healpy_utils.py

    alm = np.ascontiguousarray(alm, dtype=np.complex128)

    if nthreads is None:
        nthreads = get_optimal_nthreads(nside)

    geom = _sht_geometry(nside)
    npix = 12 * nside**2

    if pol and alm.ndim == 2 and alm.shape[0] == 3:
        if lmax is None:
            lmax = hp.Alm.getlmax(alm.shape[-1], mmax=mmax)
        mmax_use = lmax if mmax is None else mmax

        t_map = ducc0.sht.synthesis(
            alm=alm[0:1],
            theta=geom["theta"],
            nphi=geom["nphi"],
            phi0=geom["phi0"],
            lmax=lmax,
            mmax=mmax_use,
            ringstart=geom["ringstart"],
            nthreads=nthreads,
            spin=0,
        )

        eb_alm = np.array([alm[1], alm[2]])
        qu_map = ducc0.sht.synthesis(
            alm=eb_alm.reshape(1, 2, -1),
            theta=geom["theta"],
            nphi=geom["nphi"],
            phi0=geom["phi0"],
            lmax=lmax,
            mmax=mmax_use,
            ringstart=geom["ringstart"],
            nthreads=nthreads,
            spin=2,
        )

        out = np.empty((3, npix), dtype=np.float64)
        out[0] = t_map[0]
        out[1] = qu_map[0, 0]
        out[2] = qu_map[0, 1]
        return out

    if alm.ndim != 1:
        raise ValueError(
            "ducc0_alm2map only supports a 1-D spin-0 alm array, or a "
            "(3, nalm) array with pol=True"
        )

    if lmax is None:
        lmax = hp.Alm.getlmax(alm.size, mmax=mmax)
    mmax_use = lmax if mmax is None else mmax

    t_map = ducc0.sht.synthesis(
        alm=alm[np.newaxis, :],
        theta=geom["theta"],
        nphi=geom["nphi"],
        phi0=geom["phi0"],
        lmax=lmax,
        mmax=mmax_use,
        ringstart=geom["ringstart"],
        nthreads=nthreads,
        spin=0,
    )
    return t_map[0]


def alm2cl(
    alm1: np.ndarray,
    alm2: np.ndarray | None = None,
    lmax: int | None = None,
    lmax_out: int | None = None,
    nspec: int | None = None,
) -> np.ndarray:
    r"""
    (Cross-)power spectrum from healpy-layout alm(s), replacing ``hp.alm2cl``.

    .. math::

        C_\ell = \frac{1}{2\ell+1}\left[ a_{\ell 0} b^*_{\ell 0}
            + 2 \sum_{m>0} \mathrm{Re}(a_{\ell m} b^*_{\ell m}) \right]

    Parameters
    ----------
    alm1 : np.ndarray
        Spherical harmonic coefficients, healpy m-major layout. Shape
        ``(nalm,)`` for a single field, or ``(n, nalm)`` for ``n`` fields.
    alm2 : np.ndarray, optional
        Second set of coefficients, same shape convention as ``alm1``.
        Default: ``alm2 = alm1`` (auto-spectra).
    lmax : int, optional
        Band-limit shared by both inputs used to align (l, m) indices when
        ``alm1`` and ``alm2`` were packed at different band-limits. Default:
        the smaller of the two arrays' own band-limits.
    lmax_out : int, optional
        Maximum l of the returned spectra. Default: ``lmax``.
    nspec : int, optional
        Number of spectra to return for multi-field input, taken in
        healpy's "ordered by diagonal" order -- for 3 fields (T, E, B):
        TT, EE, BB, TE, EB, TB. Default: all ``n * (n + 1) / 2`` spectra.

    Returns
    -------
    np.ndarray
        ``Cl`` of shape ``(lmax_out + 1,)`` for a single field, or
        ``(nspec, lmax_out + 1)`` for multi-field input.
    """
    import healpy as hp  # lazy, see utils/healpy_utils.py

    alm1 = np.asarray(alm1)
    a1 = alm1[np.newaxis, :] if alm1.ndim == 1 else alm1
    single = alm1.ndim == 1
    if a1.ndim != 2:
        raise NotImplementedError("alm2cl only supports 1-D or 2-D (n, nalm) inputs")
    n1 = a1.shape[0]
    lmax1 = hp.Alm.getlmax(a1.shape[-1])

    if alm2 is None:
        a2 = a1
        n2 = n1
        lmax2 = lmax1
    else:
        alm2 = np.asarray(alm2)
        a2 = alm2[np.newaxis, :] if alm2.ndim == 1 else alm2
        if a2.ndim != 2:
            raise NotImplementedError(
                "alm2cl only supports 1-D or 2-D (n, nalm) inputs"
            )
        n2 = a2.shape[0]
        lmax2 = hp.Alm.getlmax(a2.shape[-1])

    if n1 != n2:
        raise NotImplementedError(
            "alm2cl requires alm1 and alm2 to hold the same number of fields"
        )

    lmax_common = lmax if lmax is not None else min(lmax1, lmax2)
    out_lmax = lmax_out if lmax_out is not None else lmax_common

    idx = np.arange(hp.Alm.getsize(lmax_common))
    l_idx, m_idx = hp.Alm.getlm(lmax_common, idx)
    idx1 = hp.Alm.getidx(lmax1, l_idx, m_idx)
    idx2 = hp.Alm.getidx(lmax2, l_idx, m_idx)
    weight = np.where(m_idx == 0, 1.0, 2.0)
    denom = 2 * np.arange(lmax_common + 1) + 1

    n = n1
    if n == 1:
        pairs = [(0, 0)]
    else:
        pairs = [(i, i + d) for d in range(n) for i in range(n - d)]
    if nspec is not None:
        pairs = pairs[:nspec]

    out = np.zeros((len(pairs), out_lmax + 1))
    for k, (i, j) in enumerate(pairs):
        x = a1[i][idx1]
        y = a2[j][idx2]
        contrib = weight * (x * np.conj(y)).real
        cl = np.bincount(l_idx, weights=contrib, minlength=lmax_common + 1) / denom
        n_copy = min(out_lmax + 1, cl.size)
        out[k, :n_copy] = cl[:n_copy]

    if single and len(pairs) == 1:
        return out[0]
    return out


def almxfl(alm: np.ndarray, fl: np.ndarray) -> np.ndarray:
    r"""
    Multiply ``a_{\ell m}`` by :math:`f_\ell`, replacing ``hp.almxfl``.

    Parameters
    ----------
    alm : np.ndarray
        Spherical harmonic coefficients, healpy m-major layout, shape
        ``(nalm,)``.
    fl : np.ndarray
        Function of :math:`\ell`, defined for ``l = 0 .. fl.size - 1`` and
        assumed to be zero elsewhere (matching ``hp.almxfl``).

    Returns
    -------
    np.ndarray
        ``alm`` multiplied elementwise by ``fl[l]``, same shape as ``alm``.
    """
    import healpy as hp  # lazy, see utils/healpy_utils.py

    alm = np.asarray(alm)
    fl = np.asarray(fl)
    lmax = hp.Alm.getlmax(alm.size)
    l_idx, _ = hp.Alm.getlm(lmax, np.arange(alm.size))

    factor = np.zeros(alm.size, dtype=np.result_type(fl.dtype, np.float64))
    valid = l_idx < fl.size
    factor[valid] = fl[l_idx[valid]]

    return alm * factor


def cplx_spin_weighted_ylm(ell, m, nside=None):
    """This script returns the real and imaginary part of the
    spin0 spherical harmonic of order l, m

    Parameters
    ----------
    nside: int
        input resolution
    ell: int
        degree
    m: int
        order

    Returns
    -------
    np.ndarray
    Spin-Weighted Spherical Harmonic maps
    """
    import healpy as hp  # lazy, see utils/healpy_utils.py

    if nside is None:
        nside = get_nside_from_ell(ell)
    try:
        assert ell > 1 and abs(m) <= ell
    except AssertionError:
        print(f"Invalid parameters: ell={ell}, m={m}. Requires ell > 1 and |m| <= ell")
        return -1
    # set up
    nalm = hp.Alm.getsize(ell, np.abs(m))
    alms = np.zeros((3, nalm), dtype=complex)
    alms[0, -1] = 1.0
    alms[1, -1] = 1.0

    # Let us discriminate on  m
    output = np.zeros((3, 12 * nside**2), dtype=np.complex128)
    if m == 0:
        output[:] = (
            ducc0_alm2map(alms, nside=nside, pol=True, lmax=ell, mmax=np.abs(m))
            + 0.0 * 1.0j
        )
    elif m != 0:
        alms /= 2.0
        output[:] = (
            ducc0_alm2map(alms, nside=nside, pol=True, lmax=ell, mmax=np.abs(m))
            + 0.0 * 1.0j
        )
        output[:] += 1.0j * ducc0_alm2map(
            -1.0j * alms, nside=nside, pol=True, lmax=ell, mmax=np.abs(m)
        )
        if m < 0:
            output[:] = (-1.0) ** np.abs(m) * np.conj(output[:])

    for indexpol, ylm in enumerate(
        [output[0], output[1] + output[2], output[1] - output[2]]
    ):
        if not np.isclose(
            4 * np.pi * np.mean(np.abs(ylm) ** 2, dtype=float), 1.0, atol=1e-2
        ):
            warnings.warn(
                "Spherical harmonics not correctly normalized, \n "
                f"with {4 * np.pi * np.mean(np.abs(ylm) ** 2, dtype=float)} at {indexpol}",
                UserWarning,
                stacklevel=2,
            )
            return -1
    output[1:] *= np.sqrt(2.0)
    return output
