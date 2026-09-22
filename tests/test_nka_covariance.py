"""
Validation of the NKA pseudo-Cl covariance against Monte Carlo simulations.

The narrow-kernel approximation (Efstathiou 2004; Camphuis et al. 2022 Eq. 26)
predicts

    Sigma_ll' = 2 C_l C_l' Xi^00_ll'[W^2]

This module checks that prediction against the sample covariance of simulated
masked skies. It exercises the whole analytic chain -- mask power spectrum,
coupling kernel, and the NKA formula -- against an independent estimate, and is
the only test that constrains the *covariance* rather than the mean spectrum.
"""

import numpy as np
import pytest

healpy = pytest.importorskip("healpy")

from cmbcov.kernels.coupling import (  # noqa: E402
    KERNEL_TT,
    coupling_kernels,
)

NSIDE = 64
LMAX = 120
N_SIMS = 1500
BAND = (25, 110)


@pytest.fixture(scope="module")
def setup():
    npix = healpy.nside2npix(NSIDE)
    theta, _ = healpy.pix2ang(NSIDE, np.arange(npix))
    edge = np.radians(40)
    mask = np.zeros(npix)
    cap = theta < edge
    mask[cap] = 0.5 * (1.0 + np.cos(np.pi * theta[cap] / edge))

    ell = np.arange(LMAX + 1)
    cl = np.zeros(LMAX + 1)
    cl[2:] = 1e-3 / (ell[2:] * (ell[2:] + 1)) ** 0.9
    return mask, cl


def nka_covariance(mask, cl):
    """Sigma_ll' = 2 C_l C_l' Xi^00_ll'[W^2]  (Camphuis et al. 2022, Eq. 26)."""
    wl_squared = healpy.anafast(mask**2, lmax=LMAX)
    kernel = coupling_kernels(wl_squared, LMAX)[KERNEL_TT]
    # coupling_kernels returns M = (2l'+1) Xi; strip the factor to recover Xi.
    xi = kernel / (2 * np.arange(LMAX + 1) + 1)[None, :]
    return 2.0 * np.outer(cl, cl) * xi


def monte_carlo_covariance(mask, cl, n_sims=N_SIMS, seed=3):
    np.random.seed(seed)
    spectra = np.empty((LMAX + 1, n_sims))
    for i in range(n_sims):
        sky = healpy.synfast(cl, NSIDE, lmax=LMAX, new=True)
        spectra[:, i] = healpy.anafast(sky * mask, lmax=LMAX)
    return np.cov(spectra)


@pytest.mark.slow
def test_nka_diagonal_matches_simulations(setup):
    """
    The NKA diagonal should agree with simulations at the few-percent level.

    NKA is a ~4-5% approximation (Camphuis et al. 2022, Table 1), so this is a
    loose bound by design; it is here to catch gross normalisation errors such
    as a missing factor of 2 or fsky used in place of <W^2>.
    """
    mask, cl = setup
    lo, hi = BAND
    ratio = (
        np.diag(monte_carlo_covariance(mask, cl))[lo:hi]
        / np.diag(nka_covariance(mask, cl))[lo:hi]
    )
    assert 0.85 < ratio.mean() < 1.15, f"NKA diagonal off by {ratio.mean():.3f}"


@pytest.mark.slow
def test_nka_reproduces_mode_correlation_structure(setup):
    """
    The correlation matrix is the sharp test.

    Mask-induced correlation between neighbouring multipoles is the physical
    effect this package exists to model, and unlike the covariance amplitude it
    is insensitive to the overall NKA normalisation. A transposed kernel, a
    wrong mask spectrum (W instead of W^2) or a broken quadrature would all
    destroy this structure.
    """
    mask, cl = setup
    lo, hi = BAND
    mc = monte_carlo_covariance(mask, cl)
    an = nka_covariance(mask, cl)

    def correlation(matrix):
        d = np.sqrt(np.diag(matrix))
        return matrix / np.outer(d, d)

    cor_mc, cor_an = correlation(mc), correlation(an)
    for offset in (1, 2, 3, 5):
        got = np.mean(np.diag(cor_mc, offset)[lo:hi])
        want = np.mean(np.diag(cor_an, offset)[lo:hi])
        assert (
            abs(got - want) < 0.05
        ), f"correlation at |l-l'|={offset}: MC {got:+.3f} vs NKA {want:+.3f}"


def test_nka_uses_squared_mask_not_mask(setup):
    """
    Guard against the classic slip of feeding W_l where W^2_l is required.

    The covariance is quartic in the mask, so it couples to the spectrum of the
    SQUARED mask. Using W_l instead inflates the diagonal by ~2.3x here.

    Note the explicit atol=0.0: these covariances are of order 1e-18, far below
    numpy's default atol of 1e-8, so a comparison without it would call any two
    values equal and the test would be vacuous.
    """
    mask, cl = setup
    lo, hi = BAND
    correct = np.diag(nka_covariance(mask, cl))[lo:hi]

    wl_plain = healpy.anafast(mask, lmax=LMAX)
    kernel = coupling_kernels(wl_plain, LMAX)[KERNEL_TT]
    xi = kernel / (2 * np.arange(LMAX + 1) + 1)[None, :]
    wrong = np.diag(2.0 * np.outer(cl, cl) * xi)[lo:hi]

    assert not np.allclose(correct, wrong, rtol=0.2, atol=0.0)
    assert (wrong / correct).mean() > 1.5, "W vs W^2 must differ substantially"


def test_full_sky_nka_reduces_to_cosmic_variance(setup):
    """
    Exact analytic limit: with no mask, Xi^00 = delta_ll' / (2l'+1), so

        Sigma_ll = 2 C_l^2 / (2l + 1)

    which is textbook full-sky cosmic variance. This pins the absolute
    normalisation of the NKA formula -- including the factor of 2 and the
    (2l+1) mode count -- without any Monte Carlo noise.
    """
    _, cl = setup
    wl_full = np.zeros(LMAX + 1)
    wl_full[0] = 4.0 * np.pi  # mask identically 1
    kernel = coupling_kernels(wl_full, LMAX)[KERNEL_TT]
    xi = kernel / (2 * np.arange(LMAX + 1) + 1)[None, :]
    sigma = 2.0 * np.outer(cl, cl) * xi

    ell = np.arange(LMAX + 1)
    expected = 2.0 * cl**2 / (2 * ell + 1)

    lo, hi = BAND
    np.testing.assert_allclose(
        np.diag(sigma)[lo:hi], expected[lo:hi], rtol=1e-10, atol=0.0
    )

    # And no off-diagonal correlation at all on the full sky.
    off = sigma - np.diag(np.diag(sigma))
    assert np.abs(off[lo:hi, lo:hi]).max() < 1e-12 * np.abs(sigma[lo:hi, lo:hi]).max()
