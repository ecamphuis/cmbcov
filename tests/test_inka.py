"""
INKA (Nicola et al. 2021; Camphuis et al. 2022 Eqs. 30-31, A.24-A.25).

Dividing row l of M by the sum of *column* l and smoothing with ``M.T @ C``
breaks Eq. (A.25): on the baseline mask a constant spectrum would come out
as 1.93 at l = 3, and against the exact covariance the diagonal would be
3-4x further off than the Eq. (A.24) form, and worse than NKA across each
row. These tests pin the renormalisation, the smoothing direction, the
wiring into the strategy, and the accuracy against the exact covariance.
"""

import os
import tempfile
import warnings

import numpy as np
import pytest

healpy = pytest.importorskip("healpy")

from cmbcov.approximations.inka import (  # noqa: E402
    INKAStrategy,
    renormalised_kernel,
)
from cmbcov.covariance import (  # noqa: E402
    Cov,
    CovarianceConfig,
    CovarianceMethod,
)
from cmbcov.exact import exact_covariance_row  # noqa: E402
from cmbcov.grid import gl_analysis, gl_synthesis  # noqa: E402
from cmbcov.kernels.coupling import (  # noqa: E402
    KERNEL_TT,
    coupling_kernels,
)
from cmbcov.keys import CovKey  # noqa: E402
from cmbcov.sht import alm2cl  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "data")


def _power_law(size: int) -> np.ndarray:
    ell = np.arange(size)
    cl = np.zeros(size)
    cl[2:] = 1e-3 / (ell[2:] * (ell[2:] + 1)) ** 0.9
    return cl


@pytest.fixture(scope="module")
def band_limited_mask():
    """Baseline mask band-limited to LW, with W_l and W^2_l exact on a GL grid."""
    lw = 16
    mask_alm = healpy.map2alm(
        healpy.read_map(os.path.join(DATA, "baseline_mask.fits")), lmax=lw, iter=10
    )
    lmax_grid = 2 * lw + 8
    w_map = gl_synthesis(mask_alm, lw, lmax_grid)
    wl = alm2cl(mask_alm)
    wsql = alm2cl(gl_analysis(w_map**2, 2 * lw, lmax_grid))
    return lw, mask_alm, wl, wsql


# --------------------------------------------------------------------------
# Eq. (A.24)-(A.25)
# --------------------------------------------------------------------------


def test_renormalised_kernel_rows_sum_to_one(band_limited_mask):
    """Eq. (A.25), for every channel, including the l < 2 rows of spin 2."""
    _, _, wl, _ = band_limited_mask
    kernels = coupling_kernels(wl, 64)
    for channel in (kernels[0], 0.5 * (kernels[1] + kernels[2]), kernels[3]):
        bar = renormalised_kernel(channel)
        sums = bar.sum(axis=1)
        nonzero = channel.sum(axis=1) != 0
        np.testing.assert_allclose(sums[nonzero], 1.0, rtol=1e-12)
        np.testing.assert_array_equal(bar[~nonzero], channel[~nonzero])


def test_constant_spectrum_is_left_unchanged(band_limited_mask):
    """The property the old column-sum form broke (1.93 at l = 3)."""
    _, _, wl, _ = band_limited_mask
    m = coupling_kernels(wl, 80)[KERNEL_TT]
    np.testing.assert_allclose(renormalised_kernel(m) @ np.ones(81), 1.0, rtol=1e-12)


# --------------------------------------------------------------------------
# the strategy uses it, in the Eq. (31) direction
# --------------------------------------------------------------------------


def test_strategy_is_nka_on_the_renormalised_spectrum():
    lmax = 48
    cov = Cov(
        "baseline_mask.fits",
        config=CovarianceConfig(method=CovarianceMethod.INKA, lmax=lmax),
        mask_path=os.path.abspath(DATA),
        save_dir=tempfile.mkdtemp(),
    )
    cl = _power_law(lmax)
    key = CovKey(("T", "T", "T", "T"), ("090GHz",) * 4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sigma = INKAStrategy(cov).compute_covariance_term(
            key, {"090GHz090GHz": {"TT": cl}}
        )
        m = cov.M[KERNEL_TT]
        xi = cov.norm_Xi["TT", "TT"]

    cbar = renormalised_kernel(m) @ cl
    expected = 2.0 * np.outer(cbar, cbar) * xi  # two identical Wick terms, Eq. (31)
    np.testing.assert_allclose(
        sigma, expected, rtol=1e-12, atol=1e-14 * np.abs(expected).max()
    )
    transposed = renormalised_kernel(m).T @ cl
    assert np.abs(transposed / cbar - 1.0)[2:].max() > 1e-2


# --------------------------------------------------------------------------
# against the exact covariance
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ellp", [24, 32, 40])
def test_inka_beats_nka_against_the_exact_covariance(band_limited_mask, ellp):
    """
    Smooth power law, exact TT covariance on the GL grid (Eq. 7).

    Measured when this was written (lw = 16, l' = 24/32/40):
    INKA diagonal -1.05/-0.58/-0.37%, row rms 2.40/1.24/0.77%;
    NKA diagonal -4.95/-2.80/-1.80%;
    old column-sum INKA diagonal +3.48/+1.94/+1.24%, row rms 6.6/3.4/2.1%.
    """
    lw, mask_alm, wl, wsql = band_limited_mask
    lmax, lmax_int = 64, 64 + lw
    n = lmax_int + 1
    cl = _power_law(n)
    ell = np.arange(n)
    m = coupling_kernels(wl, lmax_int)[KERNEL_TT]
    xi = coupling_kernels(wsql, lmax_int)[KERNEL_TT] / (2 * ell + 1)[None, :]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exact = exact_covariance_row(
            mask_alm, cl, ellp, lmax, grid="gl", lw=lw, lmax_int=lmax_int
        )

    def relative_error(spectrum):
        approx = 2.0 * spectrum[: lmax + 1] * spectrum[ellp] * xi[: lmax + 1, ellp]
        return approx / exact - 1.0

    window = slice(ellp - 6, ellp + 7)
    inka = relative_error(renormalised_kernel(m) @ cl)
    nka = relative_error(cl)

    assert abs(inka[ellp]) < 1.5e-2
    assert np.sqrt(np.mean(inka[window] ** 2)) < 3e-2
    assert abs(inka[ellp]) < 0.5 * abs(nka[ellp])
