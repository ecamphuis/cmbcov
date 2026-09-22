r"""
Native computation of the MASTER / PolSpice mode-coupling kernels.

This module replaces the external Fortran ``master_kernels`` executable, which
required a HEALPix-f90 + cfitsio installation and scaled as :math:`O(\ell_{max}^3)`
in scalar Wigner-3j recursions.

Mathematical background
-----------------------
Camphuis et al. (2022), Eq. (3), defines the symmetric coupling operator

.. math::
    \Xi^{ss'}_{\ell\ell'}[A] = \sum_L \frac{2L+1}{4\pi} A_L
        \begin{pmatrix}\ell & \ell' & L\\ s & -s & 0\end{pmatrix}
        \begin{pmatrix}\ell & \ell' & L\\ s' & -s' & 0\end{pmatrix}

and the MASTER coupling matrix (Eq. 5) is :math:`M_{\ell\ell'} = (2\ell'+1)\Xi_{\ell\ell'}`.

Evaluating that sum directly costs :math:`O(\ell_{max}^3)` Wigner-3j evaluations.
Instead we use the equivalent real-space form. With the mask correlation function

.. math::  w(\mu) = \sum_L \frac{2L+1}{4\pi} W_L P_L(\mu),

the coupling matrix is the Legendre transform of :math:`w(\mu)\xi(\mu)`
(the PolSpice relation :math:`\langle\tilde\xi\rangle = w\,\xi`, Eq. 38):

.. math::
    M^{ss'}_{\ell\ell'} = \frac{2\ell'+1}{2}\int_{-1}^{1} d\mu\;
        w(\mu)\, d^{\ell}_{ss}(\mu)\, d^{\ell'}_{s's'}(\mu).

Discretised on Gauss-Legendre nodes this is *exact* (the integrand is a
polynomial in :math:`\mu`) and becomes a single dense matrix product

.. math::  M = \tfrac{2\ell'+1}{2}\; D_s^\top\, \mathrm{diag}(w\,\omega)\, D_{s'},

which maps directly onto BLAS/GPU rather than a scalar recursion.

The four kernel channels match the Fortran layout of ``master_kernels.F90``:

======  ==========================  ==========================================
index   name                        relation
======  ==========================  ==========================================
0       ``KERNEL_TT``               <C_TT>       = sum_l2 K[0] C_TT
1       ``KERNEL_EEpBB``            <C_EE+C_BB>  = sum_l2 K[1] (C_EE+C_BB)
2       ``KERNEL_EEmBB``            <C_EE-C_BB>  = sum_l2 K[2] (C_EE-C_BB)
3       ``KERNEL_TE``               <C_TE>       = sum_l2 K[3] C_TE
======  ==========================  ==========================================
"""

from __future__ import annotations

import numpy as np

KERNEL_TT = 0
KERNEL_EEpBB = 1
KERNEL_EEmBB = 2
KERNEL_TE = 3

# (m, m') pairs and their Wigner-d seed values at l = max(|m|, |m'|).
_SPIN_PAIRS = {
    (0, 0): 0,
    (2, 2): 2,
    (2, -2): 2,
    (2, 0): 2,
}


def _wigner_d_seed(m: int, mp: int, mu: np.ndarray) -> np.ndarray:
    """Closed-form ``d^{l0}_{m m'}(mu)`` at ``l0 = max(|m|, |m'|)``."""
    if (m, mp) == (0, 0):
        return np.ones_like(mu)
    if (m, mp) == (2, 2):
        return 0.25 * (1.0 + mu) ** 2
    if (m, mp) == (2, -2):
        return 0.25 * (1.0 - mu) ** 2
    if (m, mp) == (2, 0):
        return np.sqrt(3.0 / 8.0) * (1.0 - mu**2)
    raise ValueError(f"unsupported spin pair ({m}, {mp})")


def wigner_d_table(lmax: int, mu: np.ndarray, m: int, mp: int) -> np.ndarray:
    r"""
    Tabulate the reduced Wigner d-functions :math:`d^{\ell}_{m m'}(\mu)`.

    Uses the standard three-term recursion in :math:`\ell`

    .. math::
        d^{\ell+1}_{mm'} = \frac{(2\ell+1)\big[\ell(\ell+1)\mu - m m'\big] d^{\ell}_{mm'}
            - (\ell+1)\sqrt{(\ell^2-m^2)(\ell^2-m'^2)}\, d^{\ell-1}_{mm'}}
            {\ell\sqrt{\big((\ell+1)^2-m^2\big)\big((\ell+1)^2-m'^2\big)}}

    Parameters
    ----------
    lmax : int
        Maximum multipole (inclusive).
    mu : ndarray
        Values of :math:`\cos\theta`, shape ``(n_mu,)``.
    m, mp : int
        Spin indices; one of ``(0,0)``, ``(2,2)``, ``(2,-2)``, ``(2,0)``.

    Returns
    -------
    ndarray
        Shape ``(n_mu, lmax+1)``. Entries with ``l < max(|m|,|mp|)`` are zero.
    """
    if (m, mp) not in _SPIN_PAIRS:
        raise ValueError(f"unsupported spin pair ({m}, {mp})")
    mu = np.asarray(mu, dtype=np.float64)
    l0 = _SPIN_PAIRS[(m, mp)]
    d = np.zeros((mu.size, lmax + 1), dtype=np.float64)
    if lmax < l0:
        return d

    d[:, l0] = _wigner_d_seed(m, mp, mu)
    if lmax == l0:
        return d

    # First step upward: the l-1 term vanishes at l = l0.
    ell = l0
    if ell == 0:
        d[:, 1] = mu
    else:
        num = (2 * ell + 1) * (ell * (ell + 1) * mu - m * mp) * d[:, ell]
        den = ell * np.sqrt(((ell + 1) ** 2 - m**2) * ((ell + 1) ** 2 - mp**2))
        d[:, ell + 1] = num / den

    for ell in range(l0 + 1, lmax):
        a = np.sqrt((ell**2 - m**2) * (ell**2 - mp**2))
        b = np.sqrt(((ell + 1) ** 2 - m**2) * ((ell + 1) ** 2 - mp**2))
        num = (2 * ell + 1) * (ell * (ell + 1) * mu - m * mp) * d[:, ell] - (
            ell + 1
        ) * a * d[:, ell - 1]
        d[:, ell + 1] = num / (ell * b)
    return d


def gauss_legendre_nodes(
    n_theta: int, mu_min: float = -1.0
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Gauss-Legendre nodes and weights for :math:`\int_{\mu_{min}}^{1} d\mu`.

    The default ``mu_min = -1`` is the full sphere. Restricting the range is
    what the PolSpice kernels need: their integrands are supported only on
    :math:`\theta < \theta_{max}`, i.e. :math:`\mu > \cos\theta_{max}`, and are
    analytic there but only :math:`C^1` across the cut. Integrating just the
    support therefore restores spectral convergence. This mirrors
    ``do_legendre.f90``, which calls ``gauleg_double(mumin, 1, ...)`` with
    ``mumin = cos(thetamax)``.

    Parameters
    ----------
    n_theta : int
        Number of quadrature nodes.
    mu_min : float
        Lower bound of the :math:`\mu = \cos\theta` integration range.

    Returns
    -------
    mu : ndarray
        Nodes, shape ``(n_theta,)``, all strictly inside ``(mu_min, 1)``.
    weights : ndarray
        Quadrature weights, summing to ``1 - mu_min``.
    """
    x, w = np.polynomial.legendre.leggauss(n_theta)
    half = 0.5 * (1.0 - mu_min)
    return half * x + 0.5 * (1.0 + mu_min), half * w


def xi_operator(
    a_mu: np.ndarray,
    weights: np.ndarray,
    d_left: np.ndarray,
    d_right: np.ndarray,
) -> np.ndarray:
    r"""
    Quadrature core: the coupling operator :math:`\Xi` acting in real space.

    Computes

    .. math::
        \Xi_{\ell\ell'}[a] = \frac{1}{2}\int d\mu\; a(\mu)\,
            d^{\ell}_{\text{left}}(\mu)\, d^{\ell'}_{\text{right}}(\mu)

    as a single dense matrix product. This is the extended definition of
    :math:`\Xi` of Camphuis et al. (2022) Eq. (C.7): the operator acts directly
    on a *correlation function* ``a(mu)``, which is equivalent to acting on its
    Legendre transform :math:`A_L` in the harmonic form of Eq. (3).

    The two tables are independent, so that a kernel whose left factor is not
    a plain Wigner-d table can be built with the same core. The decoupling
    kernel :math:`^{\mathrm{dec}}G` in `kernels.polspice` is one: its left
    factor is a non-local transform of :math:`d^{\ell}_{2-2}`, and its right
    factor is :math:`d^{\ell'}_{22}`.

    Parameters
    ----------
    a_mu : ndarray
        The function ``a`` sampled at the quadrature nodes, shape ``(n_mu,)``.
        Note this is a *real-space* function, not a power spectrum.
    weights : ndarray
        Quadrature weights matching the nodes, shape ``(n_mu,)``.
    d_left, d_right : ndarray
        Wigner-d tables from `wigner_d_table`, shape ``(n_mu, lmax+1)``. They
        index the first (``l``) and second (``l'``) axis of the result
        respectively and need not use the same spin pair.

    Returns
    -------
    ndarray
        :math:`\Xi_{\ell\ell'}`, shape ``(d_left.shape[1], d_right.shape[1])``.
        Multiply by ``2*l'+1`` along the last axis to obtain a coupling matrix.
    """
    aw = (
        0.5 * np.asarray(weights, dtype=np.float64) * np.asarray(a_mu, dtype=np.float64)
    )
    return (d_left * aw[:, None]).T @ d_right


def two_lp1(lmax: int) -> np.ndarray:
    r"""Row vector :math:`2\ell'+1` used to turn :math:`\Xi` into a coupling matrix."""
    return 2.0 * np.arange(lmax + 1) + 1.0


def mask_correlation_function(wl: np.ndarray, mu: np.ndarray) -> np.ndarray:
    r"""
    Mask angular correlation function :math:`w(\mu)=\sum_L \frac{2L+1}{4\pi}W_L P_L(\mu)`.

    This is the :math:`w(\theta)` of Camphuis et al. (2022) Eq. (38).
    """
    wl = np.asarray(wl, dtype=np.float64)
    lmax_mask = wl.size - 1
    pl = wigner_d_table(lmax_mask, mu, 0, 0)  # d^L_00 = P_L
    fl = (2.0 * np.arange(lmax_mask + 1) + 1.0) * wl / (4.0 * np.pi)
    return pl @ fl


def coupling_kernels(
    wl: np.ndarray,
    lmax: int,
    n_theta: int | None = None,
    dtype: type = np.float64,
) -> np.ndarray:
    r"""
    Compute the four MASTER coupling kernels from a mask power spectrum.

    Equivalent to the Fortran ``master_kernels`` executable, but with no
    external dependency and using a single BLAS matrix product.

    Parameters
    ----------
    wl : ndarray
        Mask power spectrum :math:`W_L`, shape ``(lmax_mask+1,)``.
    lmax : int
        Maximum multipole of the returned kernels (inclusive).
    n_theta : int, optional
        Number of Gauss-Legendre nodes. Defaults to the exactness requirement
        ``lmax + (lmax_mask+1)//2 + 2``; the integrand is a polynomial of degree
        ``2*lmax + lmax_mask``, so ``n_theta >= (2*lmax+lmax_mask)/2 + 1`` is exact.
    dtype : type
        Output dtype.

    Returns
    -------
    ndarray
        Shape ``(4, lmax+1, lmax+1)``, indexed ``[channel, l1, l2]``, matching
        ``KERNEL_TT``/``KERNEL_EEpBB``/``KERNEL_EEmBB``/``KERNEL_TE``.
    """
    wl = np.asarray(wl, dtype=np.float64)
    lmax_mask = wl.size - 1
    if n_theta is None:
        n_theta = (2 * lmax + lmax_mask) // 2 + 2

    mu, weights = gauss_legendre_nodes(n_theta)
    w_mu = mask_correlation_function(wl, mu)

    d00 = wigner_d_table(lmax, mu, 0, 0)
    d22p = wigner_d_table(lmax, mu, 2, 2)
    d22m = wigner_d_table(lmax, mu, 2, -2)
    d20 = wigner_d_table(lmax, mu, 2, 0)

    kernels = np.empty((4, lmax + 1, lmax + 1), dtype=dtype)
    kernels[KERNEL_TT] = xi_operator(w_mu, weights, d00, d00)
    kernels[KERNEL_EEpBB] = xi_operator(w_mu, weights, d22p, d22p)
    kernels[KERNEL_EEmBB] = xi_operator(w_mu, weights, d22m, d22m)
    kernels[KERNEL_TE] = xi_operator(w_mu, weights, d20, d20)
    kernels *= two_lp1(lmax)[None, None, :]
    return kernels
