"""
Shared fixtures for the ``test_exact_*`` modules.

``test_exact_covariance.py`` (HEALPix grid) and ``test_exact_gl.py`` (GL grid)
each validate the exact covariance against the *same* Monte Carlo oracle: 10000
``healpy.synfast``/``anafast`` simulations of the same apodised cap and power
spectrum at NSIDE=32, LMAX=60. healpy is kept as the simulator on purpose --
these tests exist to check the ducc0-based exact covariance against an
independent implementation, so the simulation itself must not use any
``cmbcov`` code. Building that sample covariance once
per session (instead of once per module) avoids paying for it twice.
"""

import numpy as np
import pytest

from cmbcov.sht import alm2cl

healpy = pytest.importorskip("healpy")

# Monte Carlo comparison, shared by tests/test_exact_covariance.py and
# tests/test_exact_gl.py.
NSIDE = 32
LMAX = 60
N_SIMS = 10000


def power_law_cl(lmax):
    ell = np.arange(lmax + 1)
    cl = np.zeros(lmax + 1)
    cl[2:] = 1e-3 / (ell[2:] * (ell[2:] + 1)) ** 0.9
    return cl


def apodised_cap(nside):
    npix = healpy.nside2npix(nside)
    theta, _ = healpy.pix2ang(nside, np.arange(npix))
    edge = np.radians(40)
    mask = np.zeros(npix)
    cap = theta < edge
    mask[cap] = 0.5 * (1.0 + np.cos(np.pi * theta[cap] / edge))
    return mask


def monte_carlo_covariance(mask, cl, nside, lmax, n_sims, seed=3):
    np.random.seed(seed)
    spectra = np.empty((lmax + 1, n_sims))
    for i in range(n_sims):
        sky = healpy.synfast(cl, nside, lmax=lmax, new=True)
        spectra[:, i] = healpy.anafast(sky * mask, lmax=lmax)
    return np.cov(spectra)


def compute_cross_spec_cplxmapalm(alm1, alm2, nspec=4, lmax=None):
    """
    Brute-force cross-power spectrum between two ``(2, 3, nalm)`` integral
    sets, the array format returned by
    ``MaskWlm.compute_spin_weighted_integrals``: axis 0 selects the alms of
    the real part (index 0) or the imaginary part (index 1) of the weighted
    map, axis 1 the (T, E, B) field.  Adds the cross-spectra of the two
    parts, ``alm2cl(alm1[0], alm2[0]) + alm2cl(alm1[1], alm2[1])`` -- the real
    part of the complex cross-spectrum the ACC HEALPix branch contracts on
    full-``M`` coefficients (``_healpix_full_m``), so it is used in tests as an
    independent reference for that real part.

    ``lmax`` defaults to the smaller of the two inputs' effective band-limit
    (``healpy.Alm.getlmax`` of the trailing, packed-alm axis).
    """
    if lmax is None:
        lmax = min(
            healpy.Alm.getlmax(alm1.shape[-1]), healpy.Alm.getlmax(alm2.shape[-1])
        )
    return alm2cl(alm1[0], alm2=alm2[0], lmax_out=lmax, nspec=nspec) + alm2cl(
        alm1[1], alm2=alm2[1], lmax_out=lmax, nspec=nspec
    )


@pytest.fixture(scope="session")
def mc_covariance_nside32():
    """Mask, spectrum, and sample covariance of N_SIMS masked skies at
    NSIDE=32, LMAX=60, seed=3 -- the Monte Carlo oracle shared by the
    HEALPix (test_exact_covariance.py) and GL (test_exact_gl.py) exact
    covariance tests."""
    mask = apodised_cap(NSIDE)
    cl = power_law_cl(LMAX)
    cov = monte_carlo_covariance(mask, cl, NSIDE, LMAX, N_SIMS)
    return mask, cl, cov
