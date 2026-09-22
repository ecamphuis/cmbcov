r"""
Native computation of the PolSpice apodization kernels.

This module replaces the external Fortran ``cor2cl`` executable (and the
``master_kernels`` run that consumed its output), which required a
HEALPix-f90 + cfitsio installation.

Mathematical background
-----------------------
On a small footprint the MASTER mode-coupling matrix is singular, because the
mask correlation function :math:`w(\theta)` vanishes for
:math:`\theta \gtrsim \theta_{max}`. PolSpice (Szapudi et al. 2001; Chon et al.
2004) regularises this in *real* space: the pseudo correlation function
:math:`\tilde\xi` is divided by :math:`w` and apodized with a scalar function
:math:`f_{apo}`, which cuts the unobserved large angular scales smoothly.
Following Camphuis et al. (2022), Sect. 6.2 and Eqs. (39)-(40),

.. math::
    \hat\xi(\theta) = g(\theta)\,\tilde\xi(\theta), \qquad
    g(\theta) = \begin{cases} f_{apo}(\theta)/w(\theta) & \theta<\theta_{max}\\
                              0 & \theta\ge\theta_{max}\end{cases}

so that :math:`\langle\hat\xi\rangle = f_{apo}\,\xi` (Eq. 41), using the MASTER
real-space relation :math:`\langle\tilde\xi\rangle = w\,\xi` (Eq. 38).

Because a multiplication in real space is a convolution in harmonic space, both
operations are coupling matrices built with the extended :math:`\Xi` operator of
Eq. (C.7), which acts on a *correlation function* sampled in
:math:`\mu=\cos\theta` rather than on a power spectrum:

.. math::
    \Xi^{ss'}_{\ell\ell'}[a] = \frac{1}{2}\int_{-1}^{1} d\mu\; a(\mu)\,
        d^{\ell}_{ss}(\mu)\, d^{\ell'}_{s's'}(\mu).

The kernels computed here are

======================  ==========  ==================================================
symbol                  paper Eq.   definition
======================  ==========  ==================================================
:math:`^{0}K`           (43)        :math:`(2\ell'+1)\,\Xi^{00}_{\ell\ell'}[f_{apo}]`
:math:`^{-2}K`          (Sect 6.2)  :math:`(2\ell'+1)\,\Xi^{2-2}_{\ell\ell'}[f_{apo}]`
:math:`^{+2}K`          (C.23)      :math:`(2\ell'+1)\,\Xi^{22}_{\ell\ell'}[f_{apo}]`
:math:`^{\times}K`      (46)        :math:`(2\ell'+1)\,\Xi^{20}_{\ell\ell'}[f_{apo}]`
:math:`^{0}G`           (51)        :math:`(2\ell'+1)\,\Xi^{00}_{\ell\ell'}[g]`
:math:`^{-2}G`          (52)        :math:`(2\ell'+1)\,\Xi^{2-2}_{\ell\ell'}[g]`
:math:`^{+2}G`          (C.23)      :math:`(2\ell'+1)\,\Xi^{22}_{\ell\ell'}[g]`
:math:`^{\times}G`      (53)        :math:`(2\ell'+1)\,\Xi^{20}_{\ell\ell'}[g]`
:math:`^{dec}G`         (49)        Chon et al. (2004) Sect. 5, see below
======================  ==========  ==================================================

The ``K`` kernels relate the PolSpice spectrum to the *true* spectrum
(Eqs. 42, 44, 45); the ``G`` kernels relate it to the *pseudo*-spectrum
(Eqs. 47-50) and are the ones used to propagate the pseudo-:math:`C_\ell`
covariance, via :math:`^{\pm}G \equiv \frac{1}{2}(^{dec}G \pm {}^{-2}G)`
(Eqs. 55-56).

The decoupling kernel
---------------------
:math:`^{dec}G` maps :math:`\tilde C^{EE}+\tilde C^{BB}` to the decoupled
PolSpice :math:`\hat C^{EE}+\hat C^{BB}` (Eq. 49). The single integral printed
as Eq. (54), :math:`\frac{2\ell'+1}{2}\int d\mu\, g\, d^{\ell}_{22}
d^{\ell'}_{2-2}`, is *not* that kernel. It is not the identity on the full
sky without apodization (1/6 at :math:`\ell=\ell'=2`), and it leaves E-to-B
leakage as large as the B-to-B window. Chon et al. (2004) instead build
:math:`\xi = \sum_L \frac{2L+1}{4\pi}(C^E_L+C^B_L)\, d^L_{2-2}` from
:math:`\xi_+` (their Eqs. 86, 90) and read :math:`C^E+C^B` off against
:math:`f_{apo}\, d^\ell_{2-2}` (their Eq. 85). In harmonic space that is

.. math::
    ^{dec}G_{\ell\ell'} = \sum_L {}^{-2}K_{\ell L}[f_{apo}]\;
        {}^{+2}G_{L\ell'}[1/w],

and taking the adjoint of their Eq. (90) gives the same kernel as one
real-space integral, with no sum over :math:`L`:

.. math::
    ^{dec}G_{\ell\ell'} = \frac{2\ell'+1}{2}\int_{\cos\theta_{max}}^{1} d\mu'\,
        \frac{R_\ell(\mu')\, d^{\ell'}_{22}(\mu')}{w(\mu')},

.. math::
    R_\ell(\mu') = h_\ell(\mu') + \frac{8}{(1+\mu')^2}\left[
        \int_{\cos\theta_{max}}^{\mu'} \frac{h_\ell(\mu)}{1-\mu}\,d\mu
        - (1-\mu') \int_{\cos\theta_{max}}^{\mu'}
          \frac{(2+\mu)\, h_\ell(\mu)}{(1-\mu)^2}\,d\mu \right],
    \qquad h_\ell = f_{apo}\, d^\ell_{2-2}.

The inner integrals run over :math:`\theta\in[\theta',\theta_{max}]`, which
is Chon et al.'s causality property, so :math:`1/w` is never needed beyond
:math:`\theta_{max}`. With :math:`^{+2}M[W]` the MASTER kernel, this kernel
satisfies :math:`^{dec}G\,{}^{+2}M = {}^{-2}K[f_{apo}]` (their Eq. 91). That
is exactly the condition for no E-B leakage in the mean.

Conventions and numerics
------------------------
* All angles are in **radians**. The Fortran parameter files (and
  ``CovarianceConfig``) use degrees, so convert with ``np.deg2rad``.
* Quadrature is Gauss-Legendre restricted to :math:`\mu\in[\cos\theta_{max},1]`,
  exactly as ``cor2cl.F90`` does via ``do_legendre(..., z1=cos(thetamax), ...)``.
  Outside that range every integrand is identically zero, and inside it they are
  analytic, so the quadrature converges spectrally (empirically to
  :math:`10^{-13}` already at ``n_theta = lmax + 2``).
* ``cor2cl`` computes the Legendre transform :math:`G_L = 2\pi\int d\mu\,g\,P_L`
  and feeds it to ``master_kernels``, which evaluates the Wigner-3j form of
  :math:`\Xi[G]`. Since :math:`\sum_L \frac{2L+1}{4\pi} G_L P_L(\mu) = g(\mu)`,
  that is the same operator as the direct real-space quadrature used here --
  except that the Fortran round trip truncates the :math:`L` sum at ``lmax``
  whereas the 3j sum extends to :math:`\ell+\ell'`. The direct quadrature is
  therefore the more accurate of the two.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .coupling import (
    gauss_legendre_nodes,
    mask_correlation_function,
    two_lp1,
    wigner_d_table,
    xi_operator,
)

#: No apodization, :math:`f_{apo}\equiv 1` (``apodizetype = -1`` in ``cor2cl``).
APODIZE_NONE = -1
#: Gaussian apodization (``apodizetype = 0``).
APODIZE_GAUSSIAN = 0
#: Half-cosine apodization, Camphuis et al. (2022) Eq. (57) (``apodizetype = 1``).
APODIZE_COSINE = 1


def apodization_function(
    theta: np.ndarray,
    theta_max: float,
    apodize_sigma: float | None = None,
    apodize_type: int = APODIZE_COSINE,
) -> np.ndarray:
    r"""
    PolSpice scalar apodizing function :math:`f_{apo}(\theta)`.

    Reproduces ``apodize_mod.f90::apodizefunction``, with all angles in radians
    instead of degrees.

    For ``apodize_type = APODIZE_COSINE`` and ``apodize_sigma >= theta_max``
    (in particular the default ``apodize_sigma = None``) this is exactly
    Camphuis et al. (2022) Eq. (57),

    .. math::
        f_{apo}(\theta) = \begin{cases}
            \frac{1}{2}\left(1+\cos\frac{\pi\theta}{\theta_{max}}\right)
                & \theta<\theta_{max},\\
            0 & \text{otherwise},\end{cases}

    which goes smoothly from :math:`f_{apo}(0)=1` to
    :math:`f_{apo}(\theta_{max})=0` with vanishing first derivative, avoiding
    Fourier ringing.

    Parameters
    ----------
    theta : ndarray
        Angular separations in radians.
    theta_max : float
        Largest lag kept in the correlation function, in radians. The paper uses
        :math:`\theta_{max}=\pi/6` (30 deg) for SPT-3G.
    apodize_sigma : float, optional
        Apodization scale in radians. ``None`` (default) means "use
        ``theta_max``", i.e. Eq. (57). A value ``<= 0`` disables apodization
        entirely (:math:`f_{apo}\equiv 1` everywhere, including
        :math:`\theta>\theta_{max}`), matching the ``fwhm_degrees <= 0``
        shortcut of the Fortran routine.

        - ``APODIZE_COSINE``: half-cosine width; the cut-off is the *tighter*
          of ``theta_max`` and ``apodize_sigma``.
        - ``APODIZE_GAUSSIAN``: FWHM of the Gaussian, converted internally to
          :math:`\sigma = \mathrm{FWHM}/\sqrt{8\ln 2}`.
    apodize_type : int
        One of `APODIZE_NONE`, `APODIZE_GAUSSIAN`, `APODIZE_COSINE`.

    Returns
    -------
    ndarray
        :math:`f_{apo}(\theta)`, same shape as ``theta``, exactly zero for
        :math:`\theta \ge \theta_{max}` (unless apodization is disabled).
    """
    theta = np.asarray(theta, dtype=np.float64)
    if apodize_type == APODIZE_NONE:
        # cor2cl.F90: do_apod = (apodizetype >= 0); otherwise fcor = 1 over the
        # whole quadrature range, which is [0, theta_max].
        return np.where(theta < theta_max, 1.0, 0.0)

    if apodize_sigma is None:
        apodize_sigma = theta_max
    if apodize_sigma <= 0.0:
        # apodize_mod.f90 special case: no apodization at all.
        return np.ones_like(theta)

    if apodize_type == APODIZE_GAUSSIAN:
        sigma = apodize_sigma / np.sqrt(8.0 * np.log(2.0))
        return np.where(theta < theta_max, np.exp(-0.5 * (theta / sigma) ** 2), 0.0)

    if apodize_type == APODIZE_COSINE:
        argument = np.maximum(theta / theta_max, theta / apodize_sigma)
        return np.where(argument < 1.0, 0.5 * (1.0 + np.cos(np.pi * argument)), 0.0)

    raise ValueError(
        f"unknown apodize_type {apodize_type}; expected one of "
        f"{APODIZE_NONE}, {APODIZE_GAUSSIAN}, {APODIZE_COSINE}"
    )


def polspice_g_function(
    mu: np.ndarray,
    theta_max: float,
    wl: np.ndarray | None = None,
    apodize_sigma: float | None = None,
    apodize_type: int = APODIZE_COSINE,
) -> np.ndarray:
    r"""
    The PolSpice correction function :math:`g = f_{apo}/w`, Eq. (40).

    .. math::
        g(\theta) = \begin{cases} f_{apo}(\theta)/w(\theta)
                        & \theta\in[0,\theta_{max}),\\
                    0 & \theta\in[\theta_{max},\pi].\end{cases}

    Parameters
    ----------
    mu : ndarray
        :math:`\mu=\cos\theta`.
    theta_max : float
        Apodization cut-off in radians.
    wl : ndarray, optional
        Mask power spectrum :math:`W_L`. ``None`` means full sky,
        :math:`w\equiv 1`, in which case :math:`g = f_{apo}`.
    apodize_sigma, apodize_type
        Passed to `apodization_function`.

    Returns
    -------
    ndarray
        :math:`g(\mu)`, exactly ``0`` wherever :math:`\theta\ge\theta_{max}`
        (the division is never evaluated there, so a vanishing :math:`w` at
        large lag cannot produce a NaN).

    Raises
    ------
    ValueError
        If :math:`w(\theta)\le 0` somewhere in :math:`[0,\theta_{max})`, where
        the ratio is genuinely needed. That signals a ``theta_max`` larger than
        the angular size of the mask, for which PolSpice cannot be regularised.
    """
    mu = np.asarray(mu, dtype=np.float64)
    theta = np.arccos(np.clip(mu, -1.0, 1.0))
    f_apo = apodization_function(theta, theta_max, apodize_sigma, apodize_type)

    if wl is None:
        return f_apo

    inside = theta < theta_max
    w_mu = mask_correlation_function(wl, mu)
    if np.any(w_mu[inside] <= 0.0):
        bad = theta[inside][w_mu[inside] <= 0.0]
        raise ValueError(
            "mask correlation function w(theta) <= 0 for theta in "
            f"[{bad.min():.4f}, {bad.max():.4f}] rad, inside theta_max="
            f"{theta_max:.4f} rad: g = f_apo/w is not defined. Reduce theta_max "
            "or use a larger mask."
        )
    g = np.zeros_like(mu)
    np.divide(f_apo, w_mu, out=g, where=inside)
    return g


def _decoupling_table(
    lmax: int,
    mu: np.ndarray,
    mu_min: float,
    f_apo_of_mu,
    points: int = 6,
    block: int = 512,
) -> np.ndarray:
    r"""
    :math:`R_\ell(\mu')` of the decoupling kernel at the nodes ``mu``.

    See the module docstring. The two cumulative integrals are accumulated
    between consecutive nodes with a ``points``-point Gauss-Legendre rule on
    each gap. Their integrands are regular up to :math:`\mu = 1`, because
    :math:`d^\ell_{2-2} \propto (1-\mu)^2`. A gap is narrower than one
    oscillation of :math:`d^{\ell_{max}}`, so a few points per gap are already
    exact to rounding.

    Returns
    -------
    ndarray
        Shape ``(mu.size, lmax+1)``, in the order of ``mu``.
    """
    order = np.argsort(mu)
    nodes = mu[order]
    edges = np.concatenate(([mu_min], nodes))
    t, tw = np.polynomial.legendre.leggauss(points)

    first = np.zeros((nodes.size, lmax + 1))
    second = np.zeros((nodes.size, lmax + 1))
    for start in range(0, nodes.size, block):
        stop = min(start + block, nodes.size)
        lo, hi = edges[start:stop], edges[start + 1 : stop + 1]
        half = 0.5 * (hi - lo)
        x = (half[:, None] * t + 0.5 * (hi + lo)[:, None]).ravel()
        wx = (half[:, None] * tw).ravel()
        h = f_apo_of_mu(x)[:, None] * wigner_d_table(lmax, x, 2, -2)
        shape = (stop - start, points, lmax + 1)
        first[start:stop] = ((wx / (1.0 - x))[:, None] * h).reshape(shape).sum(1)
        second[start:stop] = (
            ((wx * (2.0 + x) / (1.0 - x) ** 2)[:, None] * h).reshape(shape).sum(1)
        )
    first = np.cumsum(first, axis=0)
    second = np.cumsum(second, axis=0)

    h_nodes = f_apo_of_mu(nodes)[:, None] * wigner_d_table(lmax, nodes, 2, -2)
    table = h_nodes + (8.0 / (1.0 + nodes) ** 2)[:, None] * (
        first - (1.0 - nodes)[:, None] * second
    )
    out = np.empty_like(table)
    out[order] = table
    return out


@dataclass(frozen=True)
class PolSpiceKernels:
    r"""
    Container for the PolSpice kernels of Camphuis et al. (2022) Sect. 6.

    All matrices are indexed ``[l, l']`` with ``l, l' in [0, lmax]`` and carry
    the :math:`(2\ell'+1)` normalisation on the second index, i.e. they act as
    :math:`\hat C_\ell = \sum_{\ell'} \mathcal{K}_{\ell\ell'} C_{\ell'}`.

    Attributes
    ----------
    K0, Km2, Kp2, Kx : ndarray
        :math:`^{0}K,\,^{-2}K,\,^{+2}K,\,^{\times}K`; built from
        :math:`f_{apo}` (Eqs. 43, 46 and Sect. 6.2.2). They map the true
        spectrum to the PolSpice spectrum, Eqs. (42), (44), (45), (C.22).
    G0, Gm2, Gp2, Gx : ndarray
        :math:`^{0}G,\,^{-2}G,\,^{+2}G,\,^{\times}G`; built from
        :math:`g=f_{apo}/w` (Eqs. 51-53). They map the pseudo-spectrum to the
        PolSpice spectrum, Eqs. (47), (48), (50).
    Gdec : ndarray
        :math:`^{dec}G`, the decoupling kernel of Eq. (49), built as in Chon
        et al. (2004) Sect. 5 (see the module docstring), not from the
        printed Eq. (54).
    theta_max : float
        Apodization cut-off used, in radians.
    n_theta : int
        Number of Gauss-Legendre nodes used.
    """

    K0: np.ndarray
    Km2: np.ndarray
    Kp2: np.ndarray
    Kx: np.ndarray
    G0: np.ndarray
    Gm2: np.ndarray
    Gp2: np.ndarray
    Gx: np.ndarray
    Gdec: np.ndarray
    theta_max: float
    n_theta: int

    @property
    def lmax(self) -> int:
        """Maximum multipole (inclusive)."""
        return self.K0.shape[0] - 1

    @property
    def Gplus(self) -> np.ndarray:
        r""":math:`^{+}G = \tfrac{1}{2}({}^{dec}G + {}^{-2}G)`, Eq. (56)."""
        return 0.5 * (self.Gdec + self.Gm2)

    @property
    def Gminus(self) -> np.ndarray:
        r""":math:`^{-}G = \tfrac{1}{2}({}^{dec}G - {}^{-2}G)`, Eq. (56)."""
        return 0.5 * (self.Gdec - self.Gm2)

    def as_master_array(self, which: str = "G") -> np.ndarray:
        r"""
        Pack the kernels in the legacy ``master_kernels`` channel layout.

        The Fortran pipeline obtained these matrices by running ``cor2cl``
        followed by ``master_kernels``, whose FITS output has four channels
        ordered ``[Xi^00, Xi^22, Xi^2-2, Xi^20]`` -- i.e. the same layout as
        `cmbcov.kernels.coupling.coupling_kernels`.

        Parameters
        ----------
        which : {"G", "K"}
            Which family to pack.

        Returns
        -------
        ndarray
            Shape ``(4, lmax+1, lmax+1)``.

        Notes
        -----
        This layout has no slot for :math:`^{dec}G`: the legacy code used
        :math:`^{+2}G` (Eq. C.23, the *non*-decoupled estimator) in the EE+BB
        channel. Use `Gplus` / `Gminus` for the decoupled Eq. (56) form.
        """
        if which == "G":
            parts = (self.G0, self.Gp2, self.Gm2, self.Gx)
        elif which == "K":
            parts = (self.K0, self.Kp2, self.Km2, self.Kx)
        else:
            raise ValueError(f"which must be 'G' or 'K', got {which!r}")
        return np.stack(parts, axis=0)

    #: Channel index of the covariance layout returned by
    #: :meth:`as_covariance_array`.
    COV_TT = 0
    COV_PLUS = 1
    COV_MINUS = 2
    COV_TE = 3

    def as_covariance_array(self) -> np.ndarray:
        r"""
        Pack the ``G`` kernels in the layout the covariance propagation needs.

        Unlike :meth:`as_master_array`, which reproduces the legacy Fortran
        channel order and has no slot for :math:`^{dec}G`, this is the
        *decoupled* Eq. (56) form:

        ======  ==================  ==================================
        index   name                kernel
        ======  ==================  ==================================
        0       ``COV_TT``          :math:`^{0}G`
        1       ``COV_PLUS``        :math:`^{+}G = \tfrac12(^{dec}G + {}^{-2}G)`
        2       ``COV_MINUS``       :math:`^{-}G = \tfrac12(^{dec}G - {}^{-2}G)`
        3       ``COV_TE``          :math:`^{\times}G`
        ======  ==================  ==================================

        Channel 1 is *both* the ``EE <- EE`` and the ``BB <- BB`` kernel, and
        channel 2 is both mixing kernels, because the PolSpice decoupling
        acts on :math:`(E, B)` as

        .. math::
            \hat C^{EE} = {}^{+}G\,\tilde C^{EE} + {}^{-}G\,\tilde C^{BB},
            \qquad
            \hat C^{BB} = {}^{-}G\,\tilde C^{EE} + {}^{+}G\,\tilde C^{BB}.

        The legacy layout instead put :math:`^{+2}G` (Eq. C.23, the
        *non*-decoupled estimator) where :math:`^{dec}G` belongs. The two are
        far apart at low :math:`\ell`: on the survey footprint (nside 512,
        :math:`f_{sky}` 0.0404, :math:`\theta_{max}` 30 deg) binned
        :math:`\sigma(EE)` built from the legacy pair is 86.3 times the
        correct one over :math:`2 \le \ell < 27`, 1.062 over
        :math:`27 \le \ell < 52`, and within 0.6% above :math:`\ell = 100`.

        Returns
        -------
        ndarray
            Shape ``(4, lmax+1, lmax+1)``.
        """
        return np.stack((self.G0, self.Gplus, self.Gminus, self.Gx), axis=0)


def polspice_kernels(
    lmax: int,
    theta_max: float,
    wl: np.ndarray | None = None,
    apodize_sigma: float | None = None,
    apodize_type: int = APODIZE_COSINE,
    n_theta: int | None = None,
    dtype: type = np.float64,
) -> PolSpiceKernels:
    r"""
    Compute all PolSpice kernels of Camphuis et al. (2022) Sect. 6 natively.

    Equivalent to running the Fortran ``cor2cl`` followed by
    ``master_kernels``, but with no external dependency, no truncation of the
    intermediate Legendre transform, and using dense matrix products.

    Parameters
    ----------
    lmax : int
        Maximum multipole of the returned kernels (inclusive).
    theta_max : float
        PolSpice apodization cut-off in **radians** (paper: :math:`\pi/6`).
    wl : ndarray, optional
        Mask power spectrum :math:`W_L`, shape ``(lmax_mask+1,)``, used to build
        :math:`w(\theta)` (Eq. 38) and hence :math:`g` (Eq. 40). ``None`` means
        full sky, :math:`w\equiv 1`, for which the ``G`` kernels coincide with
        the ``K`` kernels and :math:`^{dec}G = {}^{-2}K`.
    apodize_sigma : float, optional
        Apodization scale in radians; ``None`` reproduces Eq. (57). See
        `apodization_function`.
    apodize_type : int
        `APODIZE_NONE`, `APODIZE_GAUSSIAN` or `APODIZE_COSINE`.
    n_theta : int, optional
        Number of Gauss-Legendre nodes on :math:`[\cos\theta_{max}, 1]`.
        Defaults to ``lmax + lmax_mask//2 + 32``. Exactness of the polynomial
        factor :math:`d^{\ell}d^{\ell'}` (degree :math:`2\ell_{max}`) alone
        needs ``lmax+1``; the extra margin covers the analytic, non-polynomial
        factors :math:`f_{apo}` and :math:`1/w`.
    dtype : type
        Output dtype.

    Returns
    -------
    PolSpiceKernels
        See the dataclass for the individual matrices.

    Notes
    -----
    Normalisation is fixed by the :math:`(2\ell'+1)\Xi` convention of
    Eqs. (43), (46), (51)-(53), identical to the MASTER convention of Eq. (5).
    A useful check is that each row of :math:`^{0}K` sums to
    :math:`f_{apo}(0)=1`, because
    :math:`\sum_{\ell'}\frac{2\ell'+1}{2}P_{\ell'}(\mu)\to\delta(\mu-1)`.
    """
    if lmax < 0:
        raise ValueError(f"lmax must be non-negative, got {lmax}")
    if not 0.0 < theta_max <= np.pi:
        raise ValueError(f"theta_max must lie in (0, pi] radians, got {theta_max}")

    lmax_mask = 0 if wl is None else np.asarray(wl).size - 1
    if n_theta is None:
        n_theta = lmax + lmax_mask // 2 + 32

    # Integrate only over the support of f_apo: [cos(theta_max), 1].
    mu_min = np.cos(theta_max) if theta_max < np.pi else -1.0
    mu, weights = gauss_legendre_nodes(n_theta, mu_min)
    theta = np.arccos(np.clip(mu, -1.0, 1.0))

    f_apo = apodization_function(theta, theta_max, apodize_sigma, apodize_type)
    g = polspice_g_function(mu, theta_max, wl, apodize_sigma, apodize_type)
    # Every node lies inside theta_max, where polspice_g_function has already
    # checked that w > 0.
    inv_w = np.ones_like(mu) if wl is None else 1.0 / mask_correlation_function(wl, mu)

    def f_apo_of_mu(x: np.ndarray) -> np.ndarray:
        return apodization_function(
            np.arccos(np.clip(x, -1.0, 1.0)), theta_max, apodize_sigma, apodize_type
        )

    d00 = wigner_d_table(lmax, mu, 0, 0)
    d22p = wigner_d_table(lmax, mu, 2, 2)
    d22m = wigner_d_table(lmax, mu, 2, -2)
    d20 = wigner_d_table(lmax, mu, 2, 0)

    norm = two_lp1(lmax)[None, :]

    def _kernel(a_mu: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.asarray(norm * xi_operator(a_mu, weights, left, right), dtype=dtype)

    return PolSpiceKernels(
        K0=_kernel(f_apo, d00, d00),
        Km2=_kernel(f_apo, d22m, d22m),
        Kp2=_kernel(f_apo, d22p, d22p),
        Kx=_kernel(f_apo, d20, d20),
        G0=_kernel(g, d00, d00),
        Gm2=_kernel(g, d22m, d22m),
        Gp2=_kernel(g, d22p, d22p),
        Gx=_kernel(g, d20, d20),
        # Eq. (49) via Chon et al. (2004) Sect. 5, not the printed Eq. (54).
        Gdec=_kernel(inv_w, _decoupling_table(lmax, mu, mu_min, f_apo_of_mu), d22p),
        theta_max=float(theta_max),
        n_theta=int(n_theta),
    )
