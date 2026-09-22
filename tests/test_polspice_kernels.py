"""
Regression tests for the native PolSpice kernels.

These pin the numerics of `cmbcov.kernels.polspice` against

1. The analytic trivial limit (f_apo == 1, w == 1  =>  0K = 0G = identity).
2. The normalisation identity sum_l' 0K_ll' = f_apo(0) = 1.
3. The harmonic route actually taken by the Fortran pipeline
   (`cor2cl` -> Legendre transform of g -> `master_kernels`), which is itself
   validated against exact Wigner-3j sums in `test_coupling_kernels.py`.
4. Exactness/stability of the Gauss-Legendre quadrature.

All angles are in radians.
"""

import numpy as np
import pytest

from cmbcov.kernels.coupling import (
    KERNEL_TE,
    KERNEL_TT,
    KERNEL_EEmBB,
    KERNEL_EEpBB,
    coupling_kernels,
    gauss_legendre_nodes,
    mask_correlation_function,
    wigner_d_table,
    xi_operator,
)
from cmbcov.kernels.polspice import (
    APODIZE_COSINE,
    APODIZE_GAUSSIAN,
    APODIZE_NONE,
    apodization_function,
    polspice_g_function,
    polspice_kernels,
)

THETA_MAX = np.pi / 6.0  # paper value, 30 deg


def _full_sky_wl(lmax_mask: int = 8) -> np.ndarray:
    """Mask identically 1  =>  W_0 = 4*pi, all other W_L = 0, i.e. w(theta) = 1."""
    wl = np.zeros(lmax_mask + 1)
    wl[0] = 4.0 * np.pi
    return wl


def _legendre_coefficients(fun, lmax_mask: int, mu_min: float = -1.0) -> np.ndarray:
    """W_L = 2*pi int dmu fun(mu) P_L(mu), the inverse of `mask_correlation_function`."""
    mu, w = gauss_legendre_nodes(8 * lmax_mask + 128, mu_min)
    pl = wigner_d_table(lmax_mask, mu, 0, 0)
    return 2.0 * np.pi * (pl * w[:, None]).T @ np.asarray(fun(mu))


def _small_patch_wl(lmax_mask: int = 64, scale: float = 0.02, fsky: float = 0.04):
    """
    A realistic small-footprint mask correlation function.

    ``w(theta) = fsky * exp(-(1-cos theta)/scale)`` falls off well before
    ``THETA_MAX``, so ``1/w`` is huge (and the truncated Legendre series even
    goes negative) at large lag -- exactly the situation PolSpice regularises.
    """
    return _legendre_coefficients(
        lambda mu: fsky * np.exp(-(1.0 - mu) / scale), lmax_mask
    )


# --------------------------------------------------------------------------
# Apodizing function, Eq. (57) / apodize_mod.f90
# --------------------------------------------------------------------------


def test_apodization_matches_eq57():
    """Default cosine apodization is exactly Camphuis et al. (2022) Eq. (57)."""
    theta = np.linspace(0.0, np.pi, 501)
    f = apodization_function(theta, THETA_MAX)
    inside = theta < THETA_MAX
    expected = np.where(inside, 0.5 * (1.0 + np.cos(np.pi * theta / THETA_MAX)), 0.0)
    np.testing.assert_allclose(f, expected, atol=1e-15)
    assert f[0] == pytest.approx(1.0)


def test_apodization_is_exactly_zero_beyond_theta_max():
    """f_apo, and hence g, must vanish identically for theta >= theta_max."""
    theta = np.concatenate(
        [np.array([THETA_MAX]), np.linspace(THETA_MAX + 1e-12, np.pi, 200)]
    )
    for apodize_type in (APODIZE_NONE, APODIZE_GAUSSIAN, APODIZE_COSINE):
        f = apodization_function(theta, THETA_MAX, apodize_type=apodize_type)
        assert np.all(f == 0.0), apodize_type


def test_apodization_smooth_at_the_cut():
    """
    Eq. (57) reaches zero with a vanishing first derivative, so there is no
    Fourier ringing: approaching theta_max the function must fall off
    quadratically, f ~ (pi eps / 2 theta_max)^2 / 2.
    """
    for eps in (1e-4, 1e-5, 1e-6):
        below = apodization_function(np.array([THETA_MAX - eps]), THETA_MAX)[0]
        assert 0.0 < below
        # One-sided derivative -> 0, i.e. C^1 across the cut.
        assert below / eps < 1e-3
        assert below == pytest.approx(0.25 * (np.pi * eps / THETA_MAX) ** 2, rel=1e-4)


def test_apodization_gaussian_matches_fortran():
    """apodize_mod.f90 type 0: exp(-(theta/sigma)^2/2), sigma = FWHM/sqrt(8 ln2)."""
    fwhm = 0.1
    theta = np.linspace(0.0, 0.5 * THETA_MAX, 17)
    sigma = fwhm / np.sqrt(8.0 * np.log(2.0))
    np.testing.assert_allclose(
        apodization_function(theta, THETA_MAX, fwhm, APODIZE_GAUSSIAN),
        np.exp(-0.5 * (theta / sigma) ** 2),
        rtol=1e-14,
    )


def test_apodization_cosine_uses_the_tighter_scale():
    """apodize_mod.f90 type 1 takes max(theta/thetamax, theta/sigma)."""
    sigma = 0.5 * THETA_MAX
    theta = np.linspace(0.0, np.pi, 301)
    f = apodization_function(theta, THETA_MAX, sigma, APODIZE_COSINE)
    assert np.all(f[theta >= sigma] == 0.0)
    np.testing.assert_allclose(
        f[theta < sigma],
        0.5 * (1.0 + np.cos(np.pi * theta[theta < sigma] / sigma)),
        atol=1e-15,
    )


def test_apodization_disabled():
    """apodizetype = -1 or a non-positive sigma disables apodization."""
    theta = np.linspace(0.0, np.pi, 51)
    inside = apodization_function(theta, THETA_MAX, apodize_type=APODIZE_NONE)
    np.testing.assert_array_equal(inside, (theta < THETA_MAX).astype(float))
    # apodize_mod.f90 returns 1 for every angle when fwhm <= 0.
    np.testing.assert_array_equal(
        apodization_function(theta, THETA_MAX, 0.0, APODIZE_COSINE), np.ones_like(theta)
    )


# --------------------------------------------------------------------------
# g = f_apo / w, Eq. (40)
# --------------------------------------------------------------------------


def test_g_reduces_to_f_apo_on_the_full_sky():
    """With w == 1 the correction function is just the apodizing function."""
    mu = np.linspace(-1.0, 1.0, 101)
    theta = np.arccos(mu)
    g_nomask = polspice_g_function(mu, THETA_MAX)
    g_unitmask = polspice_g_function(mu, THETA_MAX, _full_sky_wl())
    np.testing.assert_allclose(
        g_nomask, apodization_function(theta, THETA_MAX), atol=1e-15
    )
    np.testing.assert_allclose(g_unitmask, g_nomask, atol=1e-12)


def test_g_is_finite_and_cut_off_exactly():
    """
    Eq. (40): g must be finite where it is used and *exactly* zero beyond
    theta_max, even though 1/w blows up (and changes sign) there.
    """
    wl = _small_patch_wl()
    mu = np.linspace(-1.0, 1.0, 4001)
    theta = np.arccos(np.clip(mu, -1.0, 1.0))
    w = mask_correlation_function(wl, mu)

    # Precondition: this really is a pathological mask at large lag.
    assert np.min(np.abs(w[theta > 2.0 * THETA_MAX])) < 1e-10

    g = polspice_g_function(mu, THETA_MAX, wl)
    assert np.all(np.isfinite(g))
    assert np.all(g[theta >= THETA_MAX] == 0.0)
    assert np.all(g[theta < THETA_MAX] > 0.0)
    # Where it is used, g is exactly f_apo/w.
    inside = theta < THETA_MAX
    np.testing.assert_allclose(
        g[inside],
        apodization_function(theta[inside], THETA_MAX) / w[inside],
        rtol=1e-12,
    )


def test_g_raises_when_the_mask_is_too_small_for_theta_max():
    """w <= 0 inside theta_max means PolSpice cannot be regularised there."""
    wl = _small_patch_wl()
    with pytest.raises(ValueError, match="w\\(theta\\) <= 0"):
        polspice_g_function(np.linspace(-1.0, 1.0, 401), np.pi, wl)


# --------------------------------------------------------------------------
# Trivial limit: no apodization, full sky
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["K0", "Km2", "Kp2", "Kx", "G0", "Gm2", "Gp2", "Gx", "Gdec"]
)
def test_no_apodization_full_sky_is_identity(name):
    """
    Paper Sect. 6.3: with f_apo == 1 the K/G kernels reduce to (the inverse of)
    the master kernels; on the full sky (w == 1) both are the identity.

    f_apo == 1 *identically* requires theta_max = pi as well as no apodization.
    """
    lmax = 16
    kernels = polspice_kernels(lmax, np.pi, apodize_type=APODIZE_NONE)
    block = getattr(kernels, name)
    l0 = 0 if name in ("K0", "G0") else 2
    np.testing.assert_allclose(block[l0:, l0:], np.eye(lmax + 1 - l0), atol=1e-11)


def test_g_equals_k_on_the_full_sky():
    """With w == 1, Eq. (40) gives g = f_apo, so the G and K kernels coincide."""
    lmax = 20
    a = polspice_kernels(lmax, THETA_MAX)
    b = polspice_kernels(lmax, THETA_MAX, wl=_full_sky_wl())
    for name in ("K0", "Km2", "Kp2", "Kx", "Gdec"):
        np.testing.assert_allclose(
            getattr(a, name), getattr(b, name), rtol=1e-10, atol=1e-12
        )
    np.testing.assert_allclose(a.G0, a.K0, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(a.Gm2, a.Km2, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(a.Gx, a.Kx, rtol=1e-12, atol=1e-14)


def test_G_inverts_the_master_kernel_without_apodization():
    """
    Paper Sect. 6.3: "the first three equations reduce to the inverse of the
    master kernels when the PolSpice apodization function is set to 1".

    With f_apo == 1, g = 1/w, and multiplying by w then by 1/w in real space is
    the identity. Verified with a strictly positive, band-limited w so that
    both operators are exactly representable; the last few rows are lost to the
    lmax truncation of the intermediate product.
    """
    lmax = 40
    # w(mu) = 1 + 0.3 mu, strictly positive on [-1, 1].
    wl = np.array([4.0 * np.pi, 4.0 * np.pi / 3.0 * 0.3])
    master = coupling_kernels(wl, lmax)[KERNEL_TT]
    kernels = polspice_kernels(lmax, np.pi, wl=wl, apodize_type=APODIZE_NONE)
    product = master @ kernels.G0
    np.testing.assert_allclose(
        product[: lmax - 3], np.eye(lmax + 1)[: lmax - 3], atol=1e-9
    )


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name,l0", [("K0", 0), ("Kp2", 2)])
def test_K_rows_are_normalised_to_one(name, l0):
    """
    The paper writes "sum_ll' 0K_ll' = 1"; what actually holds is the *row*
    normalisation sum_l' 0K_ll' = f_apo(0) = 1, since
    sum_l' (2l'+1)/2 d^l'_ss(mu) d^l'_ss(mu') -> delta(mu - mu') and
    d^l_00(1) = d^l_22(1) = 1, so the delta lands on f_apo(theta = 0) = 1.

    The relation is exact only as lmax -> infinity, and rows within ~lmax of
    the truncation edge lose part of the delta (the last row keeps about half
    of it), so only interior rows are tested.
    """
    lmax = 120
    kernels = polspice_kernels(lmax, THETA_MAX)
    row_sums = getattr(kernels, name).sum(axis=1)
    np.testing.assert_allclose(row_sums[l0 : lmax - 20], 1.0, atol=1e-3)
    # The naive double sum of the paper is ~lmax+1, not 1.
    assert getattr(kernels, name).sum() == pytest.approx(lmax + 1 - l0, rel=0.05)


@pytest.mark.parametrize("name", ["Km2", "Kx"])
def test_mixed_and_negative_spin_K_rows_are_not_normalised(name):
    """
    The row normalisation does *not* extend to -2K and xK: their Wigner-d
    tables vanish at theta = 0 (d^l_{2-2}(1) = d^l_{20}(1) = 0), so the
    completeness delta picks up no weight from f_apo(0). Their row sums
    converge to l-dependent values far below 1 at low l, and only approach 1
    at l >> 1. This is checked here so that the normalisation claim is not
    over-generalised.
    """
    coarse = getattr(polspice_kernels(120, THETA_MAX), name).sum(axis=1)
    fine = getattr(polspice_kernels(240, THETA_MAX), name).sum(axis=1)
    # Converged (lmax-independent) away from the truncation edge ...
    np.testing.assert_allclose(coarse[2:100], fine[2:100], atol=1e-3)
    # ... but nowhere near 1 at low multipoles.
    assert coarse[2] < 0.1
    assert coarse[5] < 0.3


# --------------------------------------------------------------------------
# Quadrature
# --------------------------------------------------------------------------


def test_kernels_are_independent_of_quadrature_resolution():
    """The Gauss-Legendre rule on [cos theta_max, 1] converges spectrally."""
    lmax = 32
    wl = _small_patch_wl()
    a = polspice_kernels(lmax, THETA_MAX, wl=wl)
    b = polspice_kernels(lmax, THETA_MAX, wl=wl, n_theta=2 * a.n_theta)
    for name in ("K0", "Km2", "Kp2", "Kx", "G0", "Gm2", "Gp2", "Gx", "Gdec"):
        x, y = getattr(a, name), getattr(b, name)
        np.testing.assert_allclose(
            x, y, rtol=1e-9, atol=1e-10 * np.abs(y).max(), err_msg=name
        )


def test_default_quadrature_is_already_converged():
    """The default n_theta must not be the reason two runs agree."""
    lmax = 24
    wl = _small_patch_wl()
    ref = polspice_kernels(lmax, THETA_MAX, wl=wl, n_theta=8 * lmax + 512)
    for n_theta in (lmax + 2, lmax + 40, 3 * lmax):
        got = polspice_kernels(lmax, THETA_MAX, wl=wl, n_theta=n_theta)
        np.testing.assert_allclose(
            got.G0, ref.G0, rtol=1e-9, atol=1e-10 * np.abs(ref.G0).max()
        )


# --------------------------------------------------------------------------
# Equivalence with the Fortran route (cor2cl -> master_kernels)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,channel",
    [
        ("G0", KERNEL_TT),
        ("Gp2", KERNEL_EEpBB),
        ("Gm2", KERNEL_EEmBB),
        ("Gx", KERNEL_TE),
    ],
)
def test_matches_harmonic_route_through_cor2cl(name, channel):
    """
    `cor2cl` computes G_L = 2*pi int dmu g(mu) P_L(mu) and `master_kernels`
    then evaluates the Wigner-3j form of Xi[G]. Because
    sum_L (2L+1)/(4pi) G_L P_L = g, that is the same operator as the direct
    real-space quadrature -- provided the L sum is carried to l+l' <= 2*lmax
    rather than stopping at lmax as the Fortran pipeline does.
    """
    lmax = 24
    wl = _small_patch_wl()
    kernels = polspice_kernels(lmax, THETA_MAX, wl=wl)
    gl = _legendre_coefficients(
        lambda mu: polspice_g_function(mu, THETA_MAX, wl),
        2 * lmax,
        mu_min=np.cos(THETA_MAX),
    )
    reference = coupling_kernels(gl, lmax)[channel]
    got = getattr(kernels, name)
    np.testing.assert_allclose(
        got, reference, rtol=1e-8, atol=1e-9 * np.abs(reference).max()
    )


def test_fortran_lmax_truncation_is_the_only_difference():
    """
    Document the one genuine numerical difference with the Fortran pipeline:
    `cor2cl` only exports G_L up to lmax, so the 3j sum, which needs L up to
    l+l', is truncated. The effect is a sub-percent bias that the native
    implementation does not have.
    """
    lmax = 24
    wl = _small_patch_wl()
    exact = polspice_kernels(lmax, THETA_MAX, wl=wl).G0
    truncated = coupling_kernels(
        _legendre_coefficients(
            lambda mu: polspice_g_function(mu, THETA_MAX, wl),
            lmax,
            mu_min=np.cos(THETA_MAX),
        ),
        lmax,
    )[KERNEL_TT]
    relative = np.abs(truncated - exact).max() / np.abs(exact).max()
    assert 1e-3 < relative < 1.0


# --------------------------------------------------------------------------
# Decoupling kernel, Eq. (54)
# --------------------------------------------------------------------------


def _moderate_patch_wl(lmax_mask: int = 64):
    """w(theta) = 0.3 exp(-(1-cos theta)/0.2): masked, but w stays O(1) inside THETA_MAX."""
    return _legendre_coefficients(lambda mu: 0.3 * np.exp(-(1.0 - mu) / 0.2), lmax_mask)


def test_printed_eq54_is_not_the_decoupling_kernel():
    """
    Erratum, kept as a record. The single integral printed as Eq. (54),
    (2l'+1)/2 int g d^l_22 d^l'_{2-2}, is not the identity on the full sky
    without apodization: its l = l' = 2 entry is analytically
    (2*2+1)/2 * int (1-mu^2)^2/16 dmu = 1/6. Chon et al. (2004) say the
    decoupled window is then a Kronecker delta. `Gdec` is.
    """
    lmax = 10
    mu, w = gauss_legendre_nodes(64)
    printed = (2.0 * np.arange(lmax + 1) + 1.0)[None, :] * xi_operator(
        np.ones_like(mu),
        w,
        wigner_d_table(lmax, mu, 2, 2),
        wigner_d_table(lmax, mu, 2, -2),
    )
    assert printed[2, 2] == pytest.approx(1.0 / 6.0, rel=1e-10)
    kernels = polspice_kernels(lmax, np.pi, apodize_type=APODIZE_NONE)
    np.testing.assert_allclose(kernels.Gdec[2:, 2:], np.eye(lmax - 1), atol=1e-11)


@pytest.mark.parametrize(
    "apodize_type", [APODIZE_COSINE, APODIZE_GAUSSIAN, APODIZE_NONE]
)
def test_Gdec_is_minus2K_on_the_full_sky(apodize_type):
    """
    Chon et al. Eq. (91): the decoupled estimator has window -2K[f] for both
    C^E and C^B. On the full sky +2M = 1, so dec-G itself must equal -2K. This
    holds even though the quadrature stops at theta_max, because R_l only ever
    reaches inwards from theta_max (causality of Chon Eq. 90).
    """
    kernels = polspice_kernels(32, THETA_MAX, apodize_type=apodize_type)
    scale = np.abs(kernels.Km2).max()
    np.testing.assert_allclose(kernels.Gdec, kernels.Km2, rtol=0, atol=1e-11 * scale)


def test_Gdec_removes_E_to_B_leakage_in_the_mean():
    """
    Eqs. (48)-(49) with (A.18)-(A.21): <C_hat^E +- C_hat^B> =
    {dec-G +2M, -2G -2M} (C^E +- C^B). Decoupling (Chon Eq. 91) requires
    dec-G +2M = -2K[f] = -2G -2M, so that the E->B coefficient
    (dec-G +2M - -2G -2M)/2 vanishes. The product needs l' well past lmax, so
    the kernels are built to lmax_big and compared on the first rows.
    """
    lmax, lmax_big = 32, 320
    wl = _moderate_patch_wl()
    kernels = polspice_kernels(lmax_big, THETA_MAX, wl=wl)
    master = coupling_kernels(wl, lmax_big)
    rows = slice(2, lmax + 1)

    decoupled = (kernels.Gdec @ master[KERNEL_EEpBB])[rows, rows]
    difference = (kernels.Gm2 @ master[KERNEL_EEmBB])[rows, rows]
    target = kernels.Km2[rows, rows]
    scale = np.abs(target).max()
    np.testing.assert_allclose(
        difference, target, rtol=0, atol=1e-10 * scale
    )  # control
    np.testing.assert_allclose(decoupled, target, rtol=0, atol=1e-9 * scale)

    e_to_b = 0.5 * (decoupled - difference)
    assert np.abs(e_to_b).max() < 1e-9 * scale


def test_Gdec_matches_the_harmonic_product_form():
    """
    The real-space form against the independent harmonic form
    dec-G = sum_L -2K[f]_lL +2G[1/w]_Ll'. The L sum converges slowly, so the
    agreement is at the ~1e-6 level set by L_big, not rounding.
    """
    lmax, lmax_big = 24, 400
    wl = _moderate_patch_wl()
    mu, w = gauss_legendre_nodes(lmax_big + 64, np.cos(THETA_MAX))
    two_l = 2.0 * np.arange(lmax_big + 1) + 1.0
    f_apo = apodization_function(np.arccos(mu), THETA_MAX)
    d2m2 = wigner_d_table(lmax_big, mu, 2, -2)
    d22 = wigner_d_table(lmax_big, mu, 2, 2)
    minus2k = two_l[None, :] * xi_operator(f_apo, w, d2m2[:, : lmax + 1], d2m2)
    plus2g = two_l[None, : lmax + 1] * xi_operator(
        1.0 / mask_correlation_function(wl, mu), w, d22, d22[:, : lmax + 1]
    )
    product = minus2k @ plus2g

    gdec = polspice_kernels(lmax, THETA_MAX, wl=wl).Gdec
    rows = slice(2, lmax + 1)
    scale = np.abs(product[rows, rows]).max()
    np.testing.assert_allclose(
        gdec[rows, rows], product[rows, rows], rtol=0, atol=1e-5 * scale
    )


@pytest.mark.parametrize(
    "apodize_type", [APODIZE_COSINE, APODIZE_GAUSSIAN, APODIZE_NONE]
)
def test_Gdec_quadrature_is_converged_for_every_apodization(apodize_type):
    """Gaussian and no apodization are discontinuous at theta_max; the cumulative rule copes."""
    lmax = 40
    wl = _moderate_patch_wl()
    a = polspice_kernels(lmax, THETA_MAX, wl=wl, apodize_type=apodize_type)
    b = polspice_kernels(
        lmax, THETA_MAX, wl=wl, apodize_type=apodize_type, n_theta=3 * a.n_theta
    )
    scale = np.abs(b.Gdec).max()
    np.testing.assert_allclose(a.Gdec, b.Gdec, rtol=0, atol=1e-11 * scale)


def test_Gplus_Gminus_definition():
    """Eq. (56): +-G = (dec G +- (-2)G)/2."""
    kernels = polspice_kernels(12, THETA_MAX, wl=_small_patch_wl())
    scale = max(np.abs(kernels.Gdec).max(), np.abs(kernels.Gm2).max())
    np.testing.assert_allclose(
        kernels.Gplus + kernels.Gminus, kernels.Gdec, rtol=1e-12, atol=1e-13 * scale
    )
    np.testing.assert_allclose(
        kernels.Gplus - kernels.Gminus, kernels.Gm2, rtol=1e-12, atol=1e-13 * scale
    )


# --------------------------------------------------------------------------
# Packaging / API
# --------------------------------------------------------------------------


def test_as_master_array_layout():
    """Legacy layout is [Xi^00, Xi^22, Xi^2-2, Xi^20], as in master_kernels.F90."""
    kernels = polspice_kernels(10, THETA_MAX, wl=_small_patch_wl())
    arr = kernels.as_master_array("G")
    assert arr.shape == (4, 11, 11)
    np.testing.assert_array_equal(arr[KERNEL_TT], kernels.G0)
    np.testing.assert_array_equal(arr[KERNEL_EEpBB], kernels.Gp2)
    np.testing.assert_array_equal(arr[KERNEL_EEmBB], kernels.Gm2)
    np.testing.assert_array_equal(arr[KERNEL_TE], kernels.Gx)
    np.testing.assert_array_equal(kernels.as_master_array("K")[KERNEL_TT], kernels.K0)
    with pytest.raises(ValueError):
        kernels.as_master_array("bogus")


def test_as_covariance_array_layout():
    """
    The covariance layout is the decoupled Eq. (56) form (G0, G+, G-, Gx),
    and it is NOT the legacy layout: channel 1 must carry decG, not +2G.
    """
    kernels = polspice_kernels(10, THETA_MAX, wl=_small_patch_wl())
    arr = kernels.as_covariance_array()
    assert arr.shape == (4, 11, 11)
    np.testing.assert_array_equal(arr[kernels.COV_TT], kernels.G0)
    np.testing.assert_array_equal(arr[kernels.COV_PLUS], kernels.Gplus)
    np.testing.assert_array_equal(arr[kernels.COV_MINUS], kernels.Gminus)
    np.testing.assert_array_equal(arr[kernels.COV_TE], kernels.Gx)

    # G+ and G- carry decG and -2G between them (Eq. 56), to rounding: the
    # half-sum/half-difference round trip is not bit-exact.
    np.testing.assert_allclose(
        arr[1] + arr[2],
        kernels.Gdec,
        rtol=1e-12,
        atol=1e-14 * np.abs(kernels.Gdec).max(),
    )
    np.testing.assert_allclose(
        arr[1] - arr[2],
        kernels.Gm2,
        rtol=1e-12,
        atol=1e-14 * np.abs(kernels.Gm2).max(),
    )

    # The regression this layout exists to prevent: the legacy EE kernel,
    # 0.5 * (Gp2 + Gm2), is a genuinely different operator from G+.
    legacy_ee = 0.5 * (kernels.Gp2 + kernels.Gm2)
    assert np.abs(legacy_ee - arr[1]).max() > 0.1 * np.abs(arr[1]).max()


def test_invalid_arguments():
    with pytest.raises(ValueError):
        polspice_kernels(10, 0.0)
    with pytest.raises(ValueError):
        polspice_kernels(10, 4.0)
    with pytest.raises(ValueError):
        polspice_kernels(-1, THETA_MAX)
    with pytest.raises(ValueError):
        apodization_function(np.zeros(3), THETA_MAX, apodize_type=7)


# --------------------------------------------------------------------------
# Refactored quadrature core
# --------------------------------------------------------------------------


def test_xi_operator_accepts_two_different_d_tables():
    """The core must not assume the left and right tables share a spin pair."""
    lmax = 8
    mu, w = gauss_legendre_nodes(4 * lmax + 32)
    d22p = wigner_d_table(lmax, mu, 2, 2)
    d22m = wigner_d_table(lmax, mu, 2, -2)
    mixed = xi_operator(np.ones_like(mu), w, d22p, d22m)
    same = xi_operator(np.ones_like(mu), w, d22m, d22m)
    # Same spin pair: orthogonality gives 1/(2l+1) on the diagonal.
    np.testing.assert_allclose(
        np.diag(same)[2:], 1.0 / (2.0 * np.arange(2, lmax + 1) + 1.0), atol=1e-12
    )
    assert not np.allclose(mixed, same)


def test_gauss_legendre_nodes_on_a_subinterval():
    """Nodes stay inside (mu_min, 1) and the weights integrate constants."""
    mu, w = gauss_legendre_nodes(24, np.cos(THETA_MAX))
    assert mu.min() > np.cos(THETA_MAX) and mu.max() < 1.0
    assert w.sum() == pytest.approx(1.0 - np.cos(THETA_MAX), rel=1e-14)
    # A polynomial of degree 2n-1 is integrated exactly.
    assert (w * mu**7).sum() == pytest.approx((1.0 - np.cos(THETA_MAX) ** 8) / 8.0)
