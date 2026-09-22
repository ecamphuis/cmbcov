r"""
Wigner 3j symbols, Gaunt coefficients and the literal Gaunt-sum form of the
ACC mode-coupling integrals -- a reference implementation.

The ACC precompute needs

.. math::

    I_{\ell m, LM} = \int W\, Y_{\ell m}\, Y^*_{LM}\, d\Omega ,

which :func:`cmbcov.grid.spin_weighted_integrals_gl`
evaluates by quadrature (synthesise, multiply, analyse) and
:func:`cmbcov.grid.banded_integrals_gl` by Legendre
transforms over a band of ``M``.  Expanding the mask,
:math:`W = \sum_{\ell_3 m_3} w_{\ell_3 m_3} Y_{\ell_3 m_3}`, and using
:math:`Y^*_{LM} = (-1)^M Y_{L,-M}` gives the closed form

.. math::

    I_{\ell m, LM} = (-1)^M \sum_{\ell_3} w_{\ell_3, M-m}\,
    G(\ell_3, \ell, L;\; M-m, m, -M) ,

with :math:`G(l_1, l_2, l_3; m_1, m_2, m_3) = \int Y_{l_1 m_1} Y_{l_2 m_2}
Y_{l_3 m_3}\, d\Omega` the Gaunt coefficient (:func:`gaunt`); the azimuthal
selection rule :math:`m_1 + m_2 + m_3 = 0` fixes :math:`m_3 = M - m`, which
is why only the mask orders in the band ``|M - m| <= m_band`` enter.  The
negative-order mask coefficients that healpy does not store follow from the
mask being real, :math:`w_{\ell,-m} = (-1)^m \bar w_{\ell m}`.  Both phase
conventions are verified numerically against ``banded_integrals_gl`` in
``tests/test_gaunt.py`` rather than trusted from the algebra.

This is :math:`O(\ell_w)` per ``(L, M)`` (one 3j symbol per :math:`\ell_3`,
each :math:`O(\min \ell)` through the Racah sum), i.e. far slower than the
quadrature routes and, being a sum of alternating terms evaluated with
``gammaln``, only good to ~1e-9 relative for multipoles of a few tens.  It
exists as an independent check and as documentation of what the banded
integrals compute; use it for small ``ell`` only.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln

__all__ = ["wigner_3j", "gaunt", "gaunt_sum_integrals"]


def wigner_3j(l1, l2, l3, m1: int, m2: int, m3: int) -> np.ndarray:
    r"""
    Wigner 3j symbol :math:`\begin{pmatrix} l_1 & l_2 & l_3 \\ m_1 & m_2 & m_3
    \end{pmatrix}` by the Racah formula, vectorised over ``l3``.

    .. math::

        \begin{pmatrix} l_1 & l_2 & l_3 \\ m_1 & m_2 & m_3 \end{pmatrix}
        = (-1)^{l_1 - l_2 - m_3} \sqrt{\Delta(l_1 l_2 l_3)}\,
        \sqrt{\prod_i (l_i + m_i)!\,(l_i - m_i)!}\;
        \sum_k \frac{(-1)^k}{k!\,(l_1 + l_2 - l_3 - k)!\,(l_1 - m_1 - k)!\,
        (l_2 + m_2 - k)!\,(l_3 - l_2 + m_1 + k)!\,(l_3 - l_1 - m_2 + k)!}

    with :math:`\Delta = \frac{(l_1 + l_2 - l_3)!\,(l_1 - l_2 + l_3)!\,
    (-l_1 + l_2 + l_3)!}{(l_1 + l_2 + l_3 + 1)!}`, the sum running over all
    ``k`` for which every factorial argument is non-negative.  Factorials
    are taken through ``scipy.special.gammaln``; the alternating sum loses
    digits as the multipoles grow (measured against sympy: relative error
    1e-15 for ``l <= 20``, 4e-12 at ``(20, 20, 21)``, 2e-9 at ``(30, 25,
    40)``, 9e-8 at ``(50, 40, 60)``), which is fine for a reference at
    small ``ell`` but not for a production kernel.

    Parameters
    ----------
    l1, l2 : int
        Two of the degrees.
    l3 : int or array of int
        The third degree; an array gives one symbol per entry.
    m1, m2, m3 : int
        Orders.  Symbols violating ``m1 + m2 + m3 = 0``, the triangle
        inequality or ``|m_i| <= l_i`` are returned as exact zeros.

    Returns
    -------
    ndarray
        Shape of ``np.asarray(l3)``, dtype float.
    """
    l3 = np.asarray(l3)
    scalar = l3.ndim == 0
    l3 = np.atleast_1d(l3).astype(np.int64)
    out = np.zeros(l3.shape)
    ok = (
        (m1 + m2 + m3 == 0)
        & (abs(m1) <= l1)
        & (abs(m2) <= l2)
        & (np.abs(m3) <= l3)
        & (l3 >= abs(l1 - l2))
        & (l3 <= l1 + l2)
    )
    if not np.any(ok):
        return out[0] if scalar else out
    L3 = l3[ok].astype(float)
    log_delta = (
        gammaln(l1 + l2 - L3 + 1)
        + gammaln(l1 - l2 + L3 + 1)
        + gammaln(-l1 + l2 + L3 + 1)
        - gammaln(l1 + l2 + L3 + 2)
    )
    log_pref = 0.5 * (
        log_delta
        + gammaln(l1 + m1 + 1)
        + gammaln(l1 - m1 + 1)
        + gammaln(l2 + m2 + 1)
        + gammaln(l2 - m2 + 1)
        + gammaln(L3 + m3 + 1)
        + gammaln(L3 - m3 + 1)
    )
    # Racah sum: k from max(0, l2 - l3 - m1, l1 - l3 + m2)
    #            to   min(l1 + l2 - l3, l1 - m1, l2 + m2).
    k_lo = np.maximum.reduce([np.zeros_like(L3), l2 - L3 - m1, l1 - L3 + m2])
    k_hi = np.minimum.reduce(
        [l1 + l2 - L3, np.full_like(L3, l1 - m1), np.full_like(L3, l2 + m2)]
    )
    ks = np.arange(0, int(k_hi.max()) + 1, dtype=float)
    K = ks[None, :]
    valid = (K >= k_lo[:, None]) & (K <= k_hi[:, None])
    Kc = np.where(valid, K, 0.0)  # keep gammaln arguments positive
    L3c = L3[:, None]
    log_terms = -(
        gammaln(Kc + 1)
        + gammaln(np.where(valid, l1 + l2 - L3c - Kc, 0.0) + 1)
        + gammaln(np.where(valid, l1 - m1 - Kc, 0.0) + 1)
        + gammaln(np.where(valid, l2 + m2 - Kc, 0.0) + 1)
        + gammaln(np.where(valid, L3c - l2 + m1 + Kc, 0.0) + 1)
        + gammaln(np.where(valid, L3c - l1 - m2 + Kc, 0.0) + 1)
    )
    # Scale the sum by its largest term to avoid overflow, then restore.
    ref = np.where(valid, log_terms, -np.inf).max(axis=1, keepdims=True)
    terms = np.where(valid, (-1.0) ** ks[None, :] * np.exp(log_terms - ref), 0.0)
    total = terms.sum(axis=1) * np.exp(ref[:, 0] + log_pref)
    sign = (-1.0) ** (l1 - l2 - m3)
    out[ok] = sign * total
    return out[0] if scalar else out


def gaunt(l1, l2, l3, m1: int, m2: int, m3: int) -> np.ndarray:
    r"""
    Gaunt coefficient :math:`\int Y_{l_1 m_1} Y_{l_2 m_2} Y_{l_3 m_3}\, d\Omega`,
    vectorised over ``l3``:

    .. math::

        G = \sqrt{\frac{(2l_1+1)(2l_2+1)(2l_3+1)}{4\pi}}
        \begin{pmatrix} l_1 & l_2 & l_3 \\ 0 & 0 & 0 \end{pmatrix}
        \begin{pmatrix} l_1 & l_2 & l_3 \\ m_1 & m_2 & m_3 \end{pmatrix} .

    Zero unless :math:`m_1 + m_2 + m_3 = 0`, the triangle inequality holds
    and :math:`l_1 + l_2 + l_3` is even.
    """
    l3 = np.asarray(l3)
    pref = np.sqrt((2 * l1 + 1) * (2 * l2 + 1) * (2 * l3 + 1) / (4 * np.pi))
    return pref * wigner_3j(l1, l2, l3, 0, 0, 0) * wigner_3j(l1, l2, l3, m1, m2, m3)


def gaunt_sum_integrals(
    mask_alm: np.ndarray,
    lw: int,
    ell: int,
    m: int,
    lmax_out: int,
    m_band: int,
) -> np.ndarray:
    r"""
    :func:`cmbcov.grid.banded_integrals_gl` (spin 0) by
    the literal Gaunt sum,

    .. math::

        I_{\ell m, LM} = (-1)^M \sum_{\ell_3 = |M - m|}^{\ell_w}
        w_{\ell_3, M-m}\; G(\ell_3, \ell, L;\; M - m,\, m,\, -M) ,

    for ``|M - m| <= m_band``, zero elsewhere.  Same shape and layout as the
    quadrature routines: ``(lmax_out + 1, 2 lmax_out + 1)``, column
    ``lmax_out + M``.  Cost :math:`O(\ell_w \min(\ell, L))` per ``(L, M)``;
    meant for tests and small ``ell`` only (module docstring).

    Parameters
    ----------
    mask_alm : ndarray
        healpy-ordered alm of the (real) mask, truncated at ``lw``.
    lw, ell, m, lmax_out, m_band
        As in :func:`~cmbcov.grid.banded_integrals_gl`.
    """
    import healpy as hp  # lazy, see utils/healpy_utils.py

    if abs(m) > ell:
        raise ValueError(f"|m| must be <= ell, got m={m}, ell={ell}")
    mask_alm = np.asarray(mask_alm)
    out = np.zeros((lmax_out + 1, 2 * lmax_out + 1), dtype=complex)
    ls = np.arange(lw + 1)
    for M in range(max(-lmax_out, m - m_band), min(lmax_out, m + m_band) + 1):
        m3 = M - m
        if abs(m3) > lw:
            continue
        # w_{l3, m3} for all l3 (zero below |m3|), with w_{l,-m} = (-1)^m conj w_{lm}.
        idx = hp.Alm.getidx(lw, ls[ls >= abs(m3)], abs(m3))
        w = np.zeros(lw + 1, dtype=complex)
        w[abs(m3) :] = mask_alm[idx]
        if m3 < 0:
            w = (-1.0) ** abs(m3) * np.conj(w)
        for L in range(abs(M), lmax_out + 1):
            g = gaunt(ell, L, ls, m, -M, m3)  # symmetric in the three columns
            out[L, lmax_out + M] = (-1.0) ** M * np.sum(w * g)
    return out
