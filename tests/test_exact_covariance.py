"""
Validation of the exact TT pseudo-Cl covariance (Camphuis et al. 2022, Sect. 3).

``cmbcov.exact`` computes rows of

    Sigma_ll' = 2 / ((2l+1)(2l'+1)) sum_{m m'} |<a~_lm a~*_l'm'>|^2      (Eq. 7)

by spherical harmonic transforms (Fig. 2 of the paper).  The tests pin the
normalisation on the full sky, the two symmetries the implementation relies on,
and (slow) the agreement with Monte Carlo simulations, where the exact result
must beat the narrow-kernel approximation 2 C_l C_l' Xi[W^2].
"""

import numpy as np
import pytest

healpy = pytest.importorskip("healpy")

from conftest import apodised_cap, power_law_cl  # noqa: E402

from cmbcov.exact import (  # noqa: E402
    exact_covariance,
    exact_covariance_row,
)
from cmbcov.kernels.coupling import (  # noqa: E402
    KERNEL_TT,
    coupling_kernels,
)

# Small sizes for the exact/symmetry tests.
NSIDE_SMALL = 16
LMAX_SMALL = 24

# Sizes for the Monte Carlo comparison (same mask and spectrum as
# tests/test_nka_covariance.py, at lower resolution for speed). The sample
# covariance itself comes from the session-scoped mc_covariance_nside32
# fixture in tests/conftest.py, shared with tests/test_exact_gl.py.
NSIDE = 32
LMAX = 60
N_SIMS = 10000
BAND = (20, 50)


@pytest.fixture(scope="module")
def small_setup():
    return apodised_cap(NSIDE_SMALL), power_law_cl(LMAX_SMALL)


@pytest.fixture(scope="module")
def setup():
    return apodised_cap(NSIDE), power_law_cl(LMAX)


def nka_covariance(mask, cl, lmax):
    """Sigma_ll' = 2 C_l C_l' Xi^00_ll'[W^2]  (Camphuis et al. 2022, Eq. 26)."""
    wl_squared = healpy.anafast(mask**2, lmax=lmax)
    kernel = coupling_kernels(wl_squared, lmax)[KERNEL_TT]
    xi = kernel / (2 * np.arange(lmax + 1) + 1)[None, :]
    return 2.0 * np.outer(cl, cl) * xi


def test_full_sky_is_cosmic_variance():
    """
    With W == 1 the pseudo-alm are the alm, so

        Sigma_ll' = 2 C_l^2 / (2l+1) delta_ll'.

    This pins the factor 2 and the (2l+1)(2l'+1) mode counting of Eq. (7).
    Iterated HEALPix transforms are not exact to machine precision, hence
    rtol 1e-6 (the measured error is ~4e-7).
    """
    lmax, nside = LMAX_SMALL, NSIDE_SMALL
    cl = power_law_cl(lmax)
    mask = np.ones(healpy.nside2npix(nside))
    sigma = exact_covariance(mask, cl, lmax, nside)

    ell = np.arange(lmax + 1)
    expected = np.diag(2.0 * cl**2 / (2 * ell + 1))
    np.testing.assert_allclose(sigma, expected, rtol=1e-6, atol=1e-6 * expected.max())


def test_row_column_symmetry(small_setup):
    """Sigma[l, l'] computed from row l' must equal Sigma[l', l] from row l."""
    mask, cl = small_setup
    lmax, nside = LMAX_SMALL, NSIDE_SMALL
    rows = (3, 8, 13, 20)
    computed = {lp: exact_covariance_row(mask, cl, lp, lmax, nside) for lp in rows}
    for a in rows:
        for b in rows:
            np.testing.assert_allclose(computed[a][b], computed[b][a], rtol=1e-8)


def test_m_prime_symmetry(small_setup):
    """
    For real W and real C_L the m' < 0 columns are conjugates of the m' > 0
    ones, so summing m' >= 0 with weight 2 for m' > 0 must reproduce the full
    m' in [-l', l'] loop.
    """
    mask, cl = small_setup
    lmax, nside = LMAX_SMALL, NSIDE_SMALL
    for lp in (0, 1, 7, 18):
        fast = exact_covariance_row(mask, cl, lp, lmax, nside, use_symmetry=True)
        full = exact_covariance_row(mask, cl, lp, lmax, nside, use_symmetry=False)
        np.testing.assert_allclose(fast, full, rtol=1e-10)


@pytest.mark.slow
def test_exact_matches_simulations_and_beats_nka(setup, mc_covariance_nside32):
    """
    The exact diagonal must agree with the sample covariance of simulated
    masked skies within Monte Carlo noise, and must be closer to it than the
    narrow-kernel approximation.

    The relative noise of one sample-variance element is sqrt(2/N); the band
    mean of the ratio is compared against 2 sigma of that *per-element* value,
    which is conservative because neighbouring multipoles of the MC estimate
    are correlated by the mask.
    """
    mask, cl = setup
    lo, hi = BAND
    _, _, mc = mc_covariance_nside32
    exact = exact_covariance(mask, cl, LMAX, NSIDE, rows=range(lo, hi))
    nka = nka_covariance(mask, cl, LMAX)

    ratio_exact = (np.diag(exact) / np.diag(mc))[lo:hi]
    ratio_nka = (np.diag(nka) / np.diag(mc))[lo:hi]
    sigma = np.sqrt(2.0 / N_SIMS)

    dev_exact = abs(ratio_exact.mean() - 1.0)
    dev_nka = abs(ratio_nka.mean() - 1.0)
    assert dev_exact < 2.0 * sigma, (
        f"exact/MC band mean {ratio_exact.mean():.4f} " f"(sigma/element {sigma:.4f})"
    )
    assert dev_exact < dev_nka, (
        f"exact/MC {ratio_exact.mean():.4f} is not closer to 1 than "
        f"NKA/MC {ratio_nka.mean():.4f}"
    )


@pytest.mark.slow
def test_exact_handles_band_limit_where_nka_fails(setup, mc_covariance_nside32):
    """
    At l' -> lmax the simulated skies contain no power above lmax; the exact
    calculation knows this (it uses the truncated C_L), the NKA does not and
    overshoots by a factor of a few.  This is the sharpest available check that
    the exact result really is exact.
    """
    mask, cl = setup
    _, _, mc = mc_covariance_nside32
    row = exact_covariance_row(mask, cl, LMAX, LMAX, NSIDE)
    nka = nka_covariance(mask, cl, LMAX)

    ratio_exact = row[LMAX] / mc[LMAX, LMAX]
    ratio_nka = nka[LMAX, LMAX] / mc[LMAX, LMAX]
    sigma = np.sqrt(2.0 / N_SIMS)
    assert abs(ratio_exact - 1.0) < 3.0 * sigma, f"exact/MC at lmax: {ratio_exact:.4f}"
    assert ratio_nka > 1.5, f"NKA/MC at lmax: {ratio_nka:.3f}"
