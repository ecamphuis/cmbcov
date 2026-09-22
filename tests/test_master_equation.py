"""
End-to-end validation of the MASTER equation against Monte Carlo simulations.

The algebraic tests in ``test_coupling_kernels.py`` verify the coupling kernel
against exact Wigner-3j symbols, but they cannot catch a transposed matrix
(Xi is symmetric) or a wrong physical convention. This module closes that gap
by simulating masked skies and checking

    <pseudo-C_l>  ==  sum_l' M_ll' C_l'                (Hivon et al. 2002)

which is Eq. (4)/(34) of Camphuis et al. (2022).
"""

import numpy as np
import pytest

healpy = pytest.importorskip("healpy")

from cmbcov.kernels.coupling import (  # noqa: E402
    KERNEL_TT,
    coupling_kernels,
    mask_correlation_function,
)

NSIDE = 64
LMAX = 150  # must stay <= 3*nside-1 so anafast/synfast are trustworthy
N_SIMS = 200


@pytest.fixture(scope="module")
def apodised_mask():
    """Apodised polar cap, fsky ~ 3.5% (SPT-3G-like small footprint)."""
    npix = healpy.nside2npix(NSIDE)
    theta, _ = healpy.pix2ang(NSIDE, np.arange(npix))
    edge = np.radians(40)
    mask = np.zeros(npix)
    cap = theta < edge
    mask[cap] = 0.5 * (1.0 + np.cos(np.pi * theta[cap] / edge))
    return mask


@pytest.fixture(scope="module")
def fiducial_cl():
    ell = np.arange(LMAX + 1)
    cl = np.zeros(LMAX + 1)
    cl[2:] = 1e-3 / (ell[2:] * (ell[2:] + 1)) ** 0.9
    return cl


def test_mask_spectrum_is_band_limited_enough(apodised_mask):
    """
    Guard against the trap that broke the first version of this test: asking
    healpy for a mask spectrum beyond ~4*nside returns silent garbage. Here we
    confirm w(0) reproduces <W^2> from the pixels directly.
    """
    wl = healpy.anafast(apodised_mask, lmax=LMAX)
    w_zero = mask_correlation_function(wl, np.array([1.0]))[0]
    np.testing.assert_allclose(w_zero, np.mean(apodised_mask**2), rtol=2e-3)


def test_kernel_row_sum_is_mean_square_mask(apodised_mask):
    """
    sum_l' M_ll' = <W^2>, not fsky: the mask multiplies the field, and the
    spectrum is quadratic in the field.
    """
    wl = healpy.anafast(apodised_mask, lmax=LMAX)
    kernel = coupling_kernels(wl, LMAX)[KERNEL_TT]
    np.testing.assert_allclose(
        kernel[20:100].sum(axis=1), np.mean(apodised_mask**2), rtol=5e-3
    )


@pytest.mark.slow
def test_master_equation_against_monte_carlo(apodised_mask, fiducial_cl):
    """
    <pseudo-C_l> from simulations must equal M @ C_l.

    Tolerances note. It is tempting to compare against the full-sky error on
    the mean, sqrt(2 / ((2l+1) N_sims) / N_multipoles), but that assumes the
    multipoles are independent. On a 3.5% footprint they are strongly
    correlated -- Camphuis et al. (2022) measure a correlation length of
    |l - l'| ~ 25 -- so a 110-multipole band holds only ~4 independent modes
    and the true error on the mean is roughly 5x larger than the naive value.
    Empirically the agreement is at the few-per-mille level and is unchanged
    when nside is doubled (i.e. it is not a pixelisation artefact), so we test
    against an absolute tolerance instead of a fake sigma.
    """
    np.random.seed(1234)

    wl = healpy.anafast(apodised_mask, lmax=LMAX)
    kernel = coupling_kernels(wl, LMAX)[KERNEL_TT]
    predicted = kernel @ fiducial_cl

    total = np.zeros(LMAX + 1)
    for _ in range(N_SIMS):
        sky = healpy.synfast(fiducial_cl, NSIDE, lmax=LMAX, new=True)
        total += healpy.anafast(sky * apodised_mask, lmax=LMAX)
    measured = total / N_SIMS

    lo, hi = 20, 130
    ratio = measured[lo:hi] / predicted[lo:hi]

    # Band-averaged agreement: this is the statement that the MASTER equation
    # holds. A transposed kernel, a wrong (2l+1) factor or an fsky-vs-<W^2>
    # normalisation error would all show up here as a large, coherent offset.
    assert (
        abs(ratio.mean() - 1.0) < 0.02
    ), f"MC/analytic band mean is {ratio.mean():.4f}, expected 1 +- 0.02"

    # Per-multipole sanity: dominated by genuine MC scatter at low l.
    assert (
        np.abs(ratio - 1.0).max() < 0.15
    ), f"worst multipole deviates by {np.abs(ratio - 1.0).max():.3f}"

    # A coherent tilt across the band would indicate a spin/convention error
    # rather than noise; allow the few-per-mille level seen empirically.
    first_half = ratio[: (hi - lo) // 2].mean()
    second_half = ratio[(hi - lo) // 2 :].mean()
    assert (
        abs(first_half - second_half) < 0.03
    ), f"MC/analytic tilts across the band: {first_half:.4f} -> {second_half:.4f}"
