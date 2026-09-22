"""
Orientation of the PolSpice kernels and of the covariance transform, Eq. (55).

The package holds every kernel as ``K[l, l'] = (2l'+1) Xi[l, l']``, acting as
``C_hat_l = sum_l' K_ll' C_l'``. The PolSpice covariance is then
``G Sigma G^T``. ``G^T Sigma G`` would be right for kernels read straight
from a Fortran FITS file, which come back transposed, but is wrong for the
native kernels, which are the default: on the end-to-end baseline the two
transforms differ by 14% (median) on the diagonal.

These tests pin both halves of the fix: the transform follows Eq. (55), and a
Fortran-ordered file is brought to the package orientation when it is loaded.
(A Fortran G is no longer reused by the kernel cache at all, see the last
test; `_load_kernel_fits` still reads legacy Fortran M/Msq files.)
"""

import json
import os
import shutil

import numpy as np
import pytest
from astropy.io import fits
from numpy.polynomial import legendre

from cmbcov.kernels.coupling import (
    coupling_kernels,
    gauss_legendre_nodes,
    wigner_d_table,
)
from cmbcov.kernels.polspice import polspice_kernels
from cmbcov.postprocess import CovariancePostProcessor

DATA = os.path.join(os.path.dirname(__file__), "data")
THETA_MAX = np.pi / 6.0


def _small_patch_wl(lmax_mask: int = 64, scale: float = 0.02, fsky: float = 0.04):
    """``w(theta) = fsky exp(-(1-cos theta)/scale)`` as a Legendre spectrum."""
    mu, w = gauss_legendre_nodes(8 * lmax_mask + 128, -1.0)
    pl = wigner_d_table(lmax_mask, mu, 0, 0)
    return 2.0 * np.pi * (pl * w[:, None]).T @ (fsky * np.exp(-(1.0 - mu) / scale))


def _write_like_fortran(path: str, kernel: np.ndarray) -> None:
    """
    Write ``kernel[c, l1, l2]`` the way ``master_kernels`` does.

    The Fortran array is ``kernels(l1, l2, c)``, and ``image3D2fits`` writes it
    column-major with ``NAXIS = (l1, l2, c)``. For a Fortran-ordered array
    ``F``, ``F.T`` has exactly those bytes in C order and exactly that header
    once astropy reverses the axes, so this reproduces the file bit for bit.
    """
    fortran = np.asfortranarray(np.transpose(kernel, (1, 2, 0)))
    fits.PrimaryHDU(np.ascontiguousarray(fortran.T)).writeto(path, overwrite=True)


# --------------------------------------------------------------------------
# which way round the native kernels act
# --------------------------------------------------------------------------


def test_native_kernel_acts_on_its_second_index():
    """
    Eq. (39) in real space: xi_hat = f_apo xi_tilde. With theta_max = pi the
    cosine window is (1+mu)/2, a polynomial, so the real-space route is exact
    and decides the orientation independently of any docstring.
    """
    lmax = 30
    kernels = polspice_kernels(lmax, np.pi)
    ell = np.arange(lmax + 1)
    spectrum = np.random.default_rng(1).uniform(1.0, 2.0, lmax + 1)

    mu, w = legendre.leggauss(200)
    xi = legendre.legval(mu, (2 * ell + 1) / (4 * np.pi) * spectrum)
    pl = np.array([legendre.legval(mu, row) for row in np.eye(lmax + 1)])
    polspice = 2 * np.pi * pl @ (w * 0.5 * (1 + mu) * xi)

    np.testing.assert_allclose(kernels.K0 @ spectrum, polspice, rtol=1e-10)
    assert np.abs(kernels.K0.T @ spectrum - polspice).max() > 0.1 * polspice.max()


# --------------------------------------------------------------------------
# Eq. (55)
# --------------------------------------------------------------------------


def test_pseudo_to_spice_is_the_covariance_of_the_transformed_spectra():
    """
    Draw correlated T and E pseudo-spectra, apply the PolSpice kernels to each
    draw, and compare the sample covariance of the result with the transform
    applied to the sample covariance of the input. Both are linear in the
    draws, so they agree to rounding, for auto and cross blocks alike.
    """
    lmax = 24
    n = lmax + 1
    G = polspice_kernels(lmax, THETA_MAX, wl=_small_patch_wl()).as_covariance_array()
    post = CovariancePostProcessor(G)

    rng = np.random.default_rng(7)
    mixing = rng.normal(size=(2 * n, 2 * n))
    draws = rng.normal(size=(4000, 2 * n)) @ mixing.T
    pseudo_t, pseudo_e = draws[:, :n], draws[:, n:]
    spice_t = pseudo_t @ post.G_kernels["TT"].T
    spice_e = pseudo_e @ post.G_kernels["EE"].T

    def cross_cov(a, b):
        a = a - a.mean(0)
        b = b - b.mean(0)
        return a.T @ b / (len(a) - 1)

    for key, (pa, pb), (sa, sb) in (
        ("TTxTT", (pseudo_t, pseudo_t), (spice_t, spice_t)),
        ("TTxEE", (pseudo_t, pseudo_e), (spice_t, spice_e)),
    ):
        expected = cross_cov(sa, sb)
        got = post.pseudo_to_spice(cross_cov(pa, pb), key)
        np.testing.assert_allclose(
            got, expected, rtol=1e-8, atol=1e-10 * np.abs(expected).max()
        )

        left, right = key.split("x")
        transposed = post.G_kernels[left].T @ cross_cov(pa, pb) @ post.G_kernels[right]
        assert np.abs(transposed - expected).max() > 0.1 * np.abs(expected).max()


# --------------------------------------------------------------------------
# Fortran-ordered files are normalised on load
# --------------------------------------------------------------------------


@pytest.mark.parametrize("family", ["M", "G"])
def test_fortran_ordered_file_is_read_in_package_orientation(tmp_path, family):
    from cmbcov.mask import _load_kernel_fits

    lmax = 20
    wl = _small_patch_wl()
    if family == "M":
        kernel = coupling_kernels(wl, lmax)
    else:
        kernel = polspice_kernels(lmax, THETA_MAX, wl=wl).as_master_array("G")

    fortran_file = str(tmp_path / "fortran.fits")
    _write_like_fortran(fortran_file, kernel)
    assert not np.allclose(fits.getdata(fortran_file), kernel)  # really is transposed
    np.testing.assert_array_equal(_load_kernel_fits(fortran_file), kernel)

    native_file = str(tmp_path / "native.fits")
    fits.PrimaryHDU(kernel).writeto(native_file)
    np.testing.assert_array_equal(_load_kernel_fits(native_file), kernel)


def test_cached_fortran_G_file_does_not_reach_the_pipeline(tmp_path):
    """
    A legacy Fortran G file left in the kernel directory must not flip the
    transform -- and is not used at all.

    The orientation of such a file is handled by `_load_kernel_fits` (pinned
    above), but a Fortran G is also numerically different: cor2cl truncates
    the Legendre transform of g at lmax, while the Wigner-3j sum needs
    ``L <= l + l'`` (23% max element error at lmax 24 on a small patch). So
    `get_polspice_master_kernel` reuses G only from a native manifest; a
    file whose manifest says ``backend="fortran"`` is left untouched and G
    is recomputed natively. The native kernel is computed in a separate
    directory here, so the lookup cannot find a native cache instead and
    pass vacuously.
    """
    pytest.importorskip("healpy")
    from cmbcov.mask import KERNEL_CACHE_VERSION, MaskWlm

    shutil.copy(os.path.join(DATA, "baseline_mask.fits"), tmp_path / "m.fits")
    wlm = MaskWlm("m.fits", load_path=str(tmp_path))
    ell_max = 32
    native = wlm.compute_polspice_master_kernel(
        ell_max - 1, ell_max - 1, str(tmp_path / "native"), 1, 30.0, 30.0
    )
    kernel_dir = tmp_path / "kernels"
    kernel_dir.mkdir()
    g_file = str(kernel_dir / "G_m_typ1_sig30.0_the30.0.fits")
    _write_like_fortran(g_file, native)
    with open(g_file + ".manifest.json", "w") as f:
        json.dump(
            {
                "cache_version": KERNEL_CACHE_VERSION,
                "mask_digest": wlm.mask_digest,
                "backend": "fortran",
                "l1max": ell_max - 1,
                "l2max": ell_max - 1,
                "apodization_type": 1,
                "apodization_sigma": 30.0,
                "theta_max": 30.0,
            },
            f,
        )
    fortran_bytes = open(g_file, "rb").read()

    with pytest.warns(UserWarning, match="backend"):
        loaded = wlm.get_polspice_master_kernel(ell_max, str(kernel_dir), 1, 30.0, 30.0)
    np.testing.assert_array_equal(loaded, native)
    assert open(g_file, "rb").read() == fortran_bytes
