r"""
A-priori selection of the ``(m, m', M)`` terms that carry the ACC kernel.

The ACC coupling kernel at multipoles :math:`(\ell, \ell')` is (Camphuis et
al. 2022, Sect. 4; ``sparsity.py`` in the exploratory notes)

.. math::

    K(L_1, L_2) = \mathrm{Re} \sum_{m m'} \Theta(m, m', L_1)\,
    \overline{\Theta(m, m', L_2)}, \qquad
    \Theta(m, m', L) = \sum_M I_{\ell m, LM}\, \overline{I_{\ell' m', LM}} ,

with :math:`I_{\ell m, LM} = \int W Y_{\ell m} Y^*_{LM}`.  Its cost is
:math:`(2\ell+1)(2\ell'+1)` pairs times :math:`(2 L_{max}+1)` orders ``M``
per ``L``.  ``K`` is invariant under rotations of the mask, but which terms
carry it is not: in a frame where the mask's centroid (or, for a belt, its
symmetry axis) sits at ``+z``, the mask's azimuthal spectrum
:math:`P(m_3) \propto \sum_\ell |w_{\ell m_3}|^2` is concentrated at small
:math:`|m_3|`, and three selection rules become sharp:

1. **orders.** :math:`I_{\ell m, LM}` vanishes for every ``M`` unless the
   mask overlaps :math:`Y_{\ell m}`: :math:`p_m = \sum_{LM} |I_{\ell m,LM}|^2
   = \int W^2 |Y_{\ell m}|^2` (Parseval, :func:`mode_power`).  Keep ``m``
   with :math:`p_m \ge \epsilon_m \max_m p_m`.
2. **pairs.** :math:`\Theta(m, m', L)` couples ``m`` and ``m'`` through
   the mask orders :math:`m_3 = M - m` and :math:`m_3' = M - m'`, so it is
   controlled by :math:`|m - m'|` through the autocorrelation
   :math:`A(d) = \sum_{m_3} P(m_3) P(m_3 + d)` of the azimuthal spectrum.
   Keep pairs with :math:`p_m p_{m'} A(m - m') \ge \epsilon_{pair} \max`.
3. **band.** :math:`I_{\ell m, LM} = \sum_{\ell_3} w_{\ell_3, M-m}
   G(\ldots)` involves only the mask order :math:`M - m`, so ``M`` can be
   restricted to :math:`|M - m| \le k` with :math:`\sum_{|m_3| \le k} P(m_3)
   \ge 1 - \delta_{band}`
   (:func:`~cmbcov.grid.banded_integrals_gl`).

The defaults ``eps_m = tol/10``, ``eps_pair = tol/4``, ``delta_band =
tol/10`` were validated on the survey mask (nside 64 and 128, ``ell`` 64 and
128): the kernel built from the selected terms is within a Frobenius error
``tol`` of the full one, with 3-25% of the pairs and 5-15% of the ``M``
orders (``criterion_validate.py``).  The selection is only worth it in the
pole frame -- in an arbitrary frame :math:`P(m_3)` is broad and all three
rules keep almost everything -- so :func:`select_terms` expects a mask alm
that has already been rotated with :func:`pole_rotation` /
:func:`rotate_alm`; ``K`` itself does not depend on the frame (verified to
6e-15 in ``sparsity.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid import (
    _full_shape,
    gl_quadrature_weights,
    gl_shape,
    gl_synthesis,
    gl_synthesis_complex,
)

__all__ = [
    "pole_rotation",
    "rotate_alm",
    "mode_power",
    "azimuthal_spectrum",
    "TermSelection",
    "select_terms",
]

#: Below this centroid length the mask is treated as belt-like and its
#: symmetry axis (smallest second moment) is used instead of the centroid.
_CENTROID_MIN = 0.2


def mask_moments(mask_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""
    First and second moments of a HEALPix mask,
    :math:`c = \sum_p W_p \hat n_p / \sum_p W_p` and
    :math:`T = \sum_p W_p \hat n_p \hat n_p^T / \sum_p W_p`.
    """
    import healpy as hp  # lazy, see utils/healpy_utils.py

    mask_map = np.asarray(mask_map, dtype=float)
    nside = hp.npix2nside(mask_map.size)
    v = np.array(hp.pix2vec(nside, np.arange(mask_map.size)))
    total = mask_map.sum()
    if total <= 0:
        raise ValueError("mask has no positive weight")
    c = (v * mask_map).sum(axis=1) / total
    T = (v * mask_map) @ v.T / total
    return c, T


def pole_rotation(mask_map: np.ndarray):
    r"""
    healpy ``Rotator`` taking the mask's centroid to ``+z``.

    The axis is the centroid :math:`c` of :func:`mask_moments` when
    :math:`|c| \ge 0.2`; a mask symmetric under :math:`\hat n \to -\hat n`
    (galactic cut, equatorial belt, pair of antipodal patches) has
    :math:`c \approx 0` and its symmetry axis is read off the second moment
    :math:`T` instead.  For an axisymmetric weight :math:`T` has one
    isolated eigenvalue and a degenerate pair, and the axis is the
    eigenvector of the *isolated* one: the smallest for an equatorial belt
    (little weight along the axis), the **largest** for a galactic cut,
    which keeps the two polar caps.  Taking the smallest eigenvalue
    unconditionally -- as the exploratory ``sparsity.rotator_to_pole`` did
    -- sends a galactic cut's in-plane direction to the pole and spreads
    its azimuthal spectrum from :math:`P(0) = 1 - 10^{-9}` to
    :math:`P(0) = 0.76` (``log_galcut_n64.txt``); the isolated eigenvalue
    is chosen by the larger of the two eigenvalue gaps.

    **Convention.**  With the axis at ``(lon, lat)``, ``hp.Rotator(rot=[lon,
    lat, 0], deg=True)`` maps the axis to ``+x`` (not ``+z``: healpy's
    ``rot`` applies a rotation about ``z`` by ``lon``, then about ``y`` by
    ``lat``, in the passive sense), and a further ``Rotator(rot=[0, -90,
    0])`` takes ``+x`` to ``+z``.  ``Rotator.rotate_alm`` applies
    ``r.mat`` to the field, so the returned rotator ``r`` satisfies
    ``r.mat @ axis == [0, 0, 1]`` -- this is asserted, so a change in
    healpy's convention would fail loudly rather than silently select the
    wrong terms.

    Parameters
    ----------
    mask_map : ndarray
        HEALPix mask (RING ordering), pixel weights >= 0.

    Returns
    -------
    healpy.Rotator
    """
    import healpy as hp  # lazy, see utils/healpy_utils.py

    c, T = mask_moments(mask_map)
    norm = np.linalg.norm(c)
    if norm >= _CENTROID_MIN:
        axis = c / norm
    else:
        lam, e = np.linalg.eigh(T)  # ascending eigenvalues
        # the isolated eigenvalue: smallest (belt) or largest (cap pair)
        axis = e[:, 0] if lam[1] - lam[0] >= lam[2] - lam[1] else e[:, 2]
    lon, lat = hp.vec2ang(axis, lonlat=True)
    r1 = hp.Rotator(rot=[float(lon[0]), float(lat[0]), 0.0], deg=True)
    r2 = hp.Rotator(rot=[0.0, -90.0, 0.0], deg=True)
    r = r2 * r1
    assert np.allclose(
        r.mat @ axis, [0.0, 0.0, 1.0], atol=1e-10
    ), "healpy Rotator convention changed: axis not mapped to +z"
    return r


def rotate_alm(alm: np.ndarray, lw: int, rot) -> np.ndarray:
    """Rotated copy of a healpy alm set (``rot.rotate_alm``, band-limit ``lw``)."""
    out = np.array(alm, dtype=np.complex128, copy=True)
    rot.rotate_alm(out, lmax=lw, inplace=True)
    return out


def _ring_weighted_mask_squared(
    mask_alm: np.ndarray, lw: int, lmax_grid: int, nthreads
) -> np.ndarray:
    """``sum_phi w W^2`` per ring on the GL grid ``lmax_grid`` (GL weights included)."""
    w_map = gl_synthesis(mask_alm, lw, lmax_grid, nthreads=nthreads)
    wts = gl_quadrature_weights(lmax_grid)
    return (wts * w_map**2).sum(axis=1)


def mode_power(
    mask_alm: np.ndarray,
    lw: int,
    ell: int,
    spin: int = 0,
    nthreads: int | None = None,
) -> np.ndarray:
    r"""
    Mask overlap :math:`p_m` of every order ``m`` of degree ``ell``.

    .. math::

        p_m = \int W^2\, |Y_{\ell m}|^2\, d\Omega
            = \sum_{L M} |I_{\ell m, LM}|^2
        \qquad (L \le \ell_w + \ell) ,

    the second equality by Parseval, since :math:`W Y_{\ell m}` is
    band-limited at :math:`\ell_w + \ell` and the GL analysis is unitary
    there.  For ``spin=2`` the unit vector is the E-mode one of
    :func:`~cmbcov.grid.spin_weighted_integrals_gl`,
    :math:`|Y|^2` becomes :math:`|Q|^2 + |U|^2` of its complex (Q, U) map,
    and the sum on the right runs over both the E and B outputs.

    Because :math:`|Y_{\ell m}|^2` does not depend on :math:`\phi`, all
    ``2 ell + 1`` orders come from one mask synthesis, one synthesis of the
    single-degree vector with **all** orders set, and an FFT along the rings
    (which separates the orders, ``nphi > 2 ell``):
    :math:`p_m = \sum_\theta \big(\sum_\phi w W^2\big)(\theta)\,
    |y_m(\theta)|^2`.  The grid ``lmax_grid = lw + ell`` makes the
    integrand (degree ``2 lw + 2 ell``) exact.

    Parameters
    ----------
    mask_alm : ndarray
        healpy-ordered alm of the mask, truncated at ``lw`` (rotated to the
        pole frame if the result feeds :func:`select_terms`).
    lw, ell : int
        Band-limit of the mask; degree of the harmonic.
    spin : int
        0 or 2.
    nthreads : int, optional
        Threads for ducc0.

    Returns
    -------
    ndarray
        Shape ``(2 ell + 1,)``, index ``m + ell``.
    """
    if spin not in (0, 2):
        raise ValueError(f"spin must be 0 or 2, got {spin}")
    lg = lw + ell
    w2 = _ring_weighted_mask_squared(mask_alm, lw, lg, nthreads)
    ms = np.arange(-ell, ell + 1)
    if spin == 2 and ell < 2:
        return np.zeros(ms.size)
    e = np.zeros(_full_shape(ell, spin), dtype=complex)
    if spin == 0:
        e[ell, :] = 1.0
    else:
        e[0, ell, :] = 1.0
    y = gl_synthesis_complex(e, ell, lg, spin=spin, nthreads=nthreads)
    y = y.reshape(-1, *gl_shape(lg))  # (ncomp, ntheta, nphi)
    nphi = y.shape[-1]
    f = np.fft.fft(y, axis=-1) / nphi  # coefficient of e^{i m phi}
    ym = f[:, :, ms % nphi]  # (ncomp, ntheta, 2 ell + 1)
    return np.einsum("t,ctm->m", w2, np.abs(ym) ** 2)


def azimuthal_spectrum(mask_alm: np.ndarray, lw: int) -> np.ndarray:
    r"""
    Azimuthal spectrum of the mask, :math:`P(m_3) = \sum_\ell |w_{\ell m_3}|^2`
    for signed :math:`m_3 \in [-\ell_w, \ell_w]` (index ``m3 + lw``),
    normalised to :math:`\sum_{m_3} P = 1`.  :math:`P(-m_3) = P(m_3)` for a
    real mask.  In the pole frame this is the distribution of the mask's
    coupling in azimuth, which sets the ``M`` band of :func:`select_terms`.
    """
    import healpy as hp  # lazy, see utils/healpy_utils.py

    _, ms = hp.Alm.getlm(lw)
    p = np.zeros(lw + 1)
    np.add.at(p, ms, np.abs(np.asarray(mask_alm)) ** 2)
    full = np.concatenate([p[:0:-1], [p[0]], p[1:]])
    return full / full.sum()


@dataclass
class TermSelection:
    """
    Which ``(m, m', M)`` terms of the ACC kernel at ``(ell, ellp)`` to compute.

    Attributes
    ----------
    ell, ellp : int
    keep_m, keep_mp : ndarray of bool
        Orders kept at ``ell`` (shape ``2 ell + 1``, index ``m + ell``) and
        at ``ellp``.
    keep_pair : ndarray of bool
        Shape ``(2 ell + 1, 2 ellp + 1)``; True only where both orders are kept.
    m_band : int
        Half-width ``k`` of the ``|M - m| <= k`` band.
    tolerance, eps_m, eps_pair, delta_band : float
        The thresholds that produced the selection.
    """

    ell: int
    ellp: int
    keep_m: np.ndarray
    keep_mp: np.ndarray
    keep_pair: np.ndarray
    m_band: int
    tolerance: float
    eps_m: float
    eps_pair: float
    delta_band: float

    @property
    def frac_m(self) -> float:
        """Fraction of the ``2 ell + 1`` orders kept."""
        return float(self.keep_m.mean())

    @property
    def frac_mp(self) -> float:
        """Fraction of the ``2 ellp + 1`` orders kept."""
        return float(self.keep_mp.mean())

    @property
    def frac_pairs(self) -> float:
        """Fraction of the ``(2 ell + 1)(2 ellp + 1)`` pairs kept."""
        return float(self.keep_pair.mean())

    def frac_band(self, lmax_out: int) -> float:
        """Fraction ``(2 m_band + 1) / (2 lmax_out + 1)`` of the orders ``M``."""
        return min(1.0, (2 * self.m_band + 1) / (2 * lmax_out + 1))

    def summary(self) -> str:
        return (
            f"TermSelection(ell={self.ell}, ellp={self.ellp}, tol={self.tolerance:g}): "
            f"m {self.keep_m.sum()}/{self.keep_m.size} ({self.frac_m:.3f}), "
            f"m' {self.keep_mp.sum()}/{self.keep_mp.size} ({self.frac_mp:.3f}), "
            f"pairs {self.keep_pair.sum()}/{self.keep_pair.size} ({self.frac_pairs:.3f}), "
            f"M band |M - m| <= {self.m_band}"
        )


def select_terms(
    mask_alm: np.ndarray,
    lw: int,
    ell: int,
    ellp: int,
    tolerance: float,
    *,
    eps_m: float | None = None,
    eps_pair: float | None = None,
    delta_band: float | None = None,
    spin: int = 0,
    nthreads: int | None = None,
) -> TermSelection:
    r"""
    Apply the three selection rules of the module docstring.

    Parameters
    ----------
    mask_alm : ndarray
        healpy-ordered alm of the mask, truncated at ``lw``, **already
        rotated to the pole frame** (:func:`pole_rotation`,
        :func:`rotate_alm`).  The rules are correct in any frame, but only
        selective in that one.
    lw, ell, ellp : int
        Mask band-limit and the two multipoles of the kernel.
    tolerance : float
        Target relative Frobenius error of the kernel; sets the defaults
        ``eps_m = tolerance / 10``, ``eps_pair = tolerance / 4``,
        ``delta_band = tolerance / 10``.
    eps_m, eps_pair, delta_band : float, optional
        Override the individual thresholds.
    spin : int
        0 or 2; selects the :func:`mode_power` used for :math:`p_m`.
    nthreads : int, optional
        Threads for ducc0.

    Returns
    -------
    TermSelection
    """
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    eps_m = tolerance / 10 if eps_m is None else eps_m
    eps_pair = tolerance / 4 if eps_pair is None else eps_pair
    delta_band = tolerance / 10 if delta_band is None else delta_band

    P = azimuthal_spectrum(mask_alm, lw)  # index m3 + lw
    A = np.correlate(P, P, mode="full")  # index d + 2 lw
    A /= A.max()

    pm = mode_power(mask_alm, lw, ell, spin=spin, nthreads=nthreads)
    pmp = (
        pm
        if ellp == ell
        else mode_power(mask_alm, lw, ellp, spin=spin, nthreads=nthreads)
    )
    keep_m = pm >= eps_m * pm.max()
    keep_mp = pmp >= eps_m * pmp.max()

    ms = np.arange(-ell, ell + 1)
    mps = np.arange(-ellp, ellp + 1)
    est = np.outer(pm, pmp) * A[(ms[:, None] - mps[None, :]) + 2 * lw]
    keep_pair = (est >= eps_pair * est.max()) & keep_m[:, None] & keep_mp[None, :]

    # M band: smallest k with sum_{|m3| <= k} P(m3) >= 1 - delta_band.
    cum = np.cumsum(P[lw:] * np.where(np.arange(lw + 1) == 0, 1.0, 2.0))
    m_band = int(min(np.searchsorted(cum, 1.0 - delta_band), lw))

    return TermSelection(
        ell=ell,
        ellp=ellp,
        keep_m=keep_m,
        keep_mp=keep_mp,
        keep_pair=keep_pair,
        m_band=m_band,
        tolerance=tolerance,
        eps_m=eps_m,
        eps_pair=eps_pair,
        delta_band=delta_band,
    )
