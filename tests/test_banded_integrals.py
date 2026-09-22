"""
Banded ACC integrals through Legendre transforms
(``cmbcov.grid.banded_integrals_gl``).

``banded_integrals_gl`` must reproduce ``spin_weighted_integrals_gl`` entry
by entry inside the band ``|M - m| <= m_band`` -- to round-off, since both
evaluate the same product on the same GL grid -- and return exact zeros
outside it.  These tests pin that for spin 0 and spin 2, positive and
negative ``m`` (the ``M < 0`` block goes through a separate sign rule), the
full-band limit, the precomputed ``mask_modes`` path, and the layout of
``gl_mask_modes``.
"""

import numpy as np
import pytest

healpy = pytest.importorskip("healpy")

from cmbcov.grid import (  # noqa: E402
    banded_integrals_gl,
    gl_mask_modes,
    gl_shape,
    gl_synthesis,
    spin_weighted_integrals_gl,
)

TOL = 1e-12  # relative to the largest entry, both spins


def smooth_mask_alm(nside, lw, seed=0):
    """Apodised, off-axis, non-symmetric mask as a band-limited alm."""
    rng = np.random.default_rng(seed)
    theta, phi = healpy.pix2ang(nside, np.arange(healpy.nside2npix(nside)))
    c1, c2 = rng.uniform(0.6, 1.4), rng.uniform(1.0, 3.0)
    mask = np.exp(-((theta - c1) ** 2 / 0.3 + (phi - c2) ** 2 / 0.6))
    mask += 0.3 * np.exp(-((theta - 2.2) ** 2 / 0.1 + (phi - 5.0) ** 2 / 0.2))
    return healpy.map2alm(mask, lmax=lw, iter=3)


def band_mask(m, lmax_out, m_band):
    Ms = np.arange(-lmax_out, lmax_out + 1)
    return np.abs(Ms - m) <= m_band


@pytest.mark.parametrize("spin", [0, 2])
@pytest.mark.parametrize(
    "nside, lw, ell, lmax_out", [(16, 20, 9, 12), (32, 47, 30, 31)]
)
def test_in_band_matches_two_sht_route_and_out_of_band_is_zero(
    spin, nside, lw, ell, lmax_out
):
    alm = smooth_mask_alm(nside, lw)
    for m in (-ell, -ell // 2, -1, 0, 2, ell // 2, ell):
        ref = spin_weighted_integrals_gl(alm, lw, ell, m, lmax_out, spin=spin)
        for m_band in (0, 3, 11):
            got = banded_integrals_gl(alm, lw, ell, m, lmax_out, m_band, spin=spin)
            assert got.shape == ref.shape
            band = band_mask(m, lmax_out, m_band)
            err = np.abs(got - ref)[..., band].max() / np.abs(ref).max()
            assert err <= TOL, (spin, m, m_band, err)
            assert np.all(got[..., ~band] == 0)
            # something must actually be in band (the band is never empty)
            assert np.abs(got[..., band]).max() > 0


@pytest.mark.parametrize("spin", [0, 2])
def test_full_band_reproduces_full_result(spin):
    nside, lw, ell, lmax_out = 16, 20, 9, 12
    alm = smooth_mask_alm(nside, lw)
    for m in (-9, -4, 0, 5, 9):
        ref = spin_weighted_integrals_gl(alm, lw, ell, m, lmax_out, spin=spin)
        got = banded_integrals_gl(alm, lw, ell, m, lmax_out, 2 * lmax_out, spin=spin)
        assert np.abs(got - ref).max() / np.abs(ref).max() <= TOL


def test_precomputed_mask_modes_give_identical_result():
    nside, lw, ell, lmax_out, m_band = 16, 20, 9, 12, 4
    alm = smooth_mask_alm(nside, lw)
    # same default grid rule as spin_weighted_integrals_gl
    lg = max(int(np.ceil((lw + ell + lmax_out - 1) / 2)), ell, lmax_out)
    modes = gl_mask_modes(alm, lw, lg)
    assert modes.shape == gl_shape(lg)
    for m in (-3, 0, 7):
        a = banded_integrals_gl(alm, lw, ell, m, lmax_out, m_band)
        b = banded_integrals_gl(
            alm, lw, ell, m, lmax_out, m_band, lmax_grid=lg, mask_modes=modes
        )
        assert np.array_equal(a, b)
    with pytest.raises(ValueError, match="mask_modes"):
        banded_integrals_gl(
            alm, lw, ell, 0, lmax_out, m_band, lmax_grid=lg + 1, mask_modes=modes
        )


def test_gl_mask_modes_layout():
    """Column m3 mod nphi holds (1/nphi) sum_phi W e^{-i m3 phi}; W real so
    W_{-m3} = conj W_{m3}, and the m3 = 0 column is the ring mean."""
    lw, lg = 20, 30
    alm = smooth_mask_alm(16, lw)
    modes = gl_mask_modes(alm, lw, lg)
    w_map = gl_synthesis(alm, lw, lg)
    ntheta, nphi = gl_shape(lg)
    assert modes.shape == (ntheta, nphi)
    np.testing.assert_allclose(modes[:, 0], w_map.mean(axis=1), rtol=0, atol=1e-14)
    phi = 2 * np.pi * np.arange(nphi) / nphi
    for m3 in (1, 5, -7):
        direct = (w_map * np.exp(-1j * m3 * phi)).sum(axis=1) / nphi
        np.testing.assert_allclose(modes[:, m3 % nphi], direct, rtol=0, atol=1e-14)
    np.testing.assert_allclose(
        modes[:, (-5) % nphi], np.conj(modes[:, 5]), rtol=0, atol=1e-14
    )
    # modes beyond the band-limit are (numerically) empty
    assert np.abs(modes[:, lw + 1 : nphi - lw]).max() < 1e-13 * np.abs(modes).max()


def test_spin2_below_ell2_is_zero_and_argument_checks():
    alm = smooth_mask_alm(16, 20)
    out = banded_integrals_gl(alm, 20, 1, 0, 12, 3, spin=2)
    assert out.shape == (2, 13, 25) and not np.any(out)
    with pytest.raises(ValueError):
        banded_integrals_gl(alm, 20, 5, 6, 12, 3)
    with pytest.raises(ValueError):
        banded_integrals_gl(alm, 20, 5, 0, 12, -1)
    with pytest.raises(ValueError):
        banded_integrals_gl(alm, 20, 5, 0, 12, 3, spin=1)
    with pytest.raises(ValueError):
        banded_integrals_gl(alm, 20, 5, 0, 12, 3, lmax_grid=11)
