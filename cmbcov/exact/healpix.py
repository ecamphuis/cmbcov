"""The HEALPix path of the exact covariance: transforms and the correlation column."""

from __future__ import annotations

import hashlib
from collections import OrderedDict

import numpy as np

from ..harmonics.grid import full_from_pair, pair_from_full
from ..harmonics.sht import alm2cl, ducc0_alm2map, ducc0_map2alm

# Full-M <-> real-map-pair alm conversions live in :mod:`.grid` (same layout:
# shape (lmax+1, 2*lmax+1), column index M+lmax); aliased here.
_full_from_pair = full_from_pair
_pair_from_full = pair_from_full


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
