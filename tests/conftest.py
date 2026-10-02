"""
Shared fixtures and masks for the test suite.

``test_exact_covariance.py`` (HEALPix grid) and ``test_exact_gl.py`` (GL grid)
each validate the exact covariance against the *same* Monte Carlo oracle: 10000
``healpy.synfast``/``anafast`` simulations of the same apodised cap and power
spectrum at NSIDE=32, LMAX=60. healpy is kept as the simulator on purpose --
these tests exist to check the ducc0-based exact covariance against an
independent implementation, so the simulation itself must not use any
``cmbcov`` code. Building that sample covariance once
per session (instead of once per module) avoids paying for it twice.

Masks without azimuthal symmetry.  The baseline mask of ``tests/data`` is an
apodised polar cap, azimuthally symmetric to 1e-18: on it every coefficient
``I_{lm,LM}`` of ``W Y_lm`` sits at ``M = m``, every exact column couples
``M = m'`` only and every ACC cross-spectrum ``X_{mm'}`` is diagonal in
``(m, m')``.  An error in how azimuthal orders are mixed -- an off-by-one in
an ``M - m`` band, a wrong sign on the ``-m`` reflection, a transposed
``(m, m')`` index, a wrong ``(-1)^M`` phase -- is then invisible.  Every
comparison between two code paths that mix orders therefore also runs on
:func:`patchy_mask` (every order populated in any frame) or, where the
code path rotates the mask to its pole frame first (term selection),
:func:`two_blob_mask` (compact, so the selection drops terms, but not
symmetric about its centroid).
"""

import os

import numpy as np
import pytest

from cmbcov.sht import alm2cl, ducc0_alm2map, ducc0_map2alm
from cmbcov.term_selection import rotate_alm

healpy = pytest.importorskip("healpy")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

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


# --------------------------------------------------------------------------- #
# Masks without azimuthal symmetry (module docstring)
# --------------------------------------------------------------------------- #
def patchy_mask(nside=32):
    """
    A mask with no symmetry axis: the baseline mask (an apodised polar cap,
    azimuthally symmetric, so every column of it couples ``M = m'`` only)
    moved to colatitude 55 deg, plus a second cosine-apodised patch.  Every
    azimuthal order of the mask is then populated, which is what exercises
    the coupling across orders (and its band edges).  Clipped to ``[0, 1]``.
    """
    cap = healpy.read_map(os.path.join(DATA_DIR, "baseline_mask.fits"))
    if nside != healpy.npix2nside(cap.size):
        cap = healpy.ud_grade(cap, nside)
    lmax = 3 * nside - 1
    alm = rotate_alm(
        ducc0_map2alm(cap, lmax=lmax, iter=10),
        lmax,
        healpy.Rotator(rot=(40.0, 55.0, 0.0), deg=True),
    )
    moved = ducc0_alm2map(alm, nside, lmax=lmax)
    vec = np.array(healpy.pix2vec(nside, np.arange(healpy.nside2npix(nside))))
    centre = healpy.ang2vec(np.radians(115.0), np.radians(250.0))
    dist = np.arccos(np.clip(centre @ vec, -1.0, 1.0))
    ramp = np.clip((np.radians(22.0) - dist) / np.radians(12.0), 0.0, 1.0)
    return np.clip(moved + 0.5 - 0.5 * np.cos(np.pi * ramp), 0.0, 1.0)


def cosine_cap(nside, lon, lat, radius_deg, apod_deg):
    """Azimuthally symmetric cosine-apodised cap centred at ``(lon, lat)``."""
    v = healpy.ang2vec(lon, lat, lonlat=True)
    pv = np.array(healpy.pix2vec(nside, np.arange(healpy.nside2npix(nside))))
    dist = np.degrees(np.arccos(np.clip(v @ pv, -1, 1)))
    x = np.clip((dist - (radius_deg - apod_deg)) / apod_deg, 0, 1)
    return 0.5 * (1 + np.cos(np.pi * x))


def two_blob_mask(nside):
    """
    A large cap plus a smaller one 50 deg away, clipped to ``[0, 1]``.
    Compact enough that the term selection, which works in the mask's pole
    frame, drops terms -- unlike :func:`patchy_mask`, whose patches are too
    far apart -- but not symmetric about its centroid, so the pole-frame mask
    still populates every azimuthal order (an off-axis cap rotated to the
    pole is azimuthally symmetric again).
    """
    return np.clip(
        cosine_cap(nside, 47.0, 23.0, 30, 12)
        + 0.6 * cosine_cap(nside, 95.0, 5.0, 18, 8),
        0,
        1,
    )


def _write_mask(directory, name, mask):
    healpy.write_map(str(directory / name), mask, dtype=np.float64, overwrite=True)
    return str(directory)


@pytest.fixture(scope="session")
def patchy_mask_dir(tmp_path_factory):
    """Directory holding ``patchy_mask.fits`` (:func:`patchy_mask`, nside 32,
    the baseline mask's resolution), for the ``MaskWlm`` / ``Cov`` inputs."""
    return _write_mask(
        tmp_path_factory.mktemp("patchy_mask"), "patchy_mask.fits", patchy_mask(32)
    )


@pytest.fixture(scope="session")
def two_blob_mask_dir(tmp_path_factory):
    """Directory holding ``two_blob_mask.fits`` (:func:`two_blob_mask`) at
    nside 32 and ``two_blob_mask_16.fits`` at nside 16."""
    directory = tmp_path_factory.mktemp("two_blob_mask")
    _write_mask(directory, "two_blob_mask.fits", two_blob_mask(32))
    return _write_mask(directory, "two_blob_mask_16.fits", two_blob_mask(16))
