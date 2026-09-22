"""
A-priori term selection for the ACC kernel
(``cmbcov.term_selection``).

Pins: the Parseval identity behind ``mode_power`` (spin 0 and 2), the
``pole_rotation`` convention, the sharp selection an azimuthally symmetric
cap must give once rotated to the pole (pairs only on ``m == m'``, band
``k <= 1``), and -- on the package's baseline mask -- that the kernel built
from the selected terms is within ``tolerance`` of the full one.  The kernels
are assembled here from the integrals directly (Theta and K as in Camphuis
et al. 2022, Sect. 4), independently of the ACC precompute.
"""

import os

import numpy as np
import pytest

healpy = pytest.importorskip("healpy")

from cmbcov.grid import (  # noqa: E402
    banded_integrals_gl,
    gl_mask_modes,
    spin_weighted_integrals_gl,
)
from cmbcov.term_selection import (  # noqa: E402
    azimuthal_spectrum,
    mask_moments,
    mode_power,
    pole_rotation,
    rotate_alm,
    select_terms,
)

DATA = os.path.join(os.path.dirname(__file__), "data")


def cap_mask(nside, lon, lat, radius_deg, apod_deg):
    """Azimuthally symmetric cosine-apodised cap centred at (lon, lat)."""
    v = healpy.ang2vec(lon, lat, lonlat=True)
    pv = np.array(healpy.pix2vec(nside, np.arange(healpy.nside2npix(nside))))
    dist = np.degrees(np.arccos(np.clip(v @ pv, -1, 1)))
    x = np.clip((dist - (radius_deg - apod_deg)) / apod_deg, 0, 1)
    return 0.5 * (1 + np.cos(np.pi * x))


def frob(a):
    return np.sqrt(np.sum(a * a))


def kernel_from_integrals(X, Y, keep_pair=None):
    """K(L1, L2) = Re sum_{m m'} Theta(m, m', L1) conj Theta(m, m', L2),
    Theta = sum_M X[m, L, M] conj Y[m', L, M]; X, Y of shape (n_m, L, M)."""
    th = np.einsum("mLM,nLM->mnL", X, np.conj(Y), optimize=True)
    if keep_pair is not None:
        th = th * keep_pair[:, :, None]
    flat = th.reshape(-1, th.shape[-1])
    return (flat.T @ np.conj(flat)).real


@pytest.mark.parametrize("spin", [0, 2])
def test_mode_power_equals_parseval_sum(spin):
    nside, lw, ell = 16, 12, 6
    theta, phi = healpy.pix2ang(nside, np.arange(healpy.nside2npix(nside)))
    mask = np.exp(-((theta - 1.0) ** 2 / 0.3 + (phi - 2.0) ** 2 / 0.6))
    alm = healpy.map2alm(mask, lmax=lw, iter=3)
    pm = mode_power(alm, lw, ell, spin=spin)
    assert pm.shape == (2 * ell + 1,)
    lmax_out = lw + ell  # W Y_lm is band-limited here: Parseval is exact
    ref = np.array(
        [
            np.sum(
                np.abs(spin_weighted_integrals_gl(alm, lw, ell, m, lmax_out, spin=spin))
                ** 2
            )
            for m in range(-ell, ell + 1)
        ]
    )
    assert np.abs(pm - ref).max() / ref.max() <= 1e-12


def test_azimuthal_spectrum_is_symmetric_and_normalised():
    lw = 12
    alm = healpy.map2alm(cap_mask(16, 30.0, 40.0, 40, 15), lmax=lw, iter=3)
    P = azimuthal_spectrum(alm, lw)
    assert P.shape == (2 * lw + 1,)
    assert abs(P.sum() - 1) < 1e-14
    np.testing.assert_allclose(P, P[::-1], rtol=0, atol=1e-16)


def test_pole_rotation_takes_centroid_to_z():
    nside, lw = 16, 47
    mask = cap_mask(nside, 47.0, 23.0, 30, 12)
    c, _ = mask_moments(mask)
    rot = pole_rotation(mask)
    np.testing.assert_allclose(rot.mat @ (c / np.linalg.norm(c)), [0, 0, 1], atol=1e-10)
    # the rotated field really is centred on +z: synthesise and re-measure
    alm = rotate_alm(healpy.map2alm(mask, lmax=lw, iter=10), lw, rot)
    rotated = healpy.alm2map(alm, nside, lmax=lw)
    c1, _ = mask_moments(np.clip(rotated, 0, None))
    assert np.linalg.norm(c1[:2]) < 2e-3 and c1[2] > 0.5
    # |c| ~ 0: an equatorial belt (smallest second moment along the axis)
    # and a galactic cut keeping two polar caps (largest) must both map
    # their symmetry axis to +z; put the axis off the z direction first.
    # Apodised edges, so that the pixel sums behind the moments are accurate:
    # the axis residual measured here is 1.9e-4 (belt) / 2.6e-4 (caps) at
    # nside 16 and 4.8e-5 / 6.5e-5 at nside 32, against ~3e-3 for binary
    # masks at either resolution.
    nside_sym = 32
    axis = healpy.ang2vec(70.0, 35.0, lonlat=True)
    pv = np.array(healpy.pix2vec(nside_sym, np.arange(healpy.nside2npix(nside_sym))))
    cos_d = np.abs(axis @ pv)
    taper = 0.5 * (1 + np.cos(np.pi * np.clip((cos_d - 0.4) / 0.3, 0, 1)))
    for symmetric in (taper, 1.0 - taper):
        c, _ = mask_moments(symmetric)
        assert np.linalg.norm(c) < 1e-12
        r_sym = pole_rotation(symmetric)
        assert np.linalg.norm((r_sym.mat @ axis)[:2]) < 1e-4


@pytest.mark.parametrize("tolerance", [1e-2, 1e-3])
def test_symmetric_cap_selects_only_diagonal_pairs_in_pole_frame(tolerance):
    nside = 32
    lw = 3 * nside - 1
    ell, ellp = nside, nside + 3
    mask = cap_mask(nside, 47.0, 23.0, 30, 12)
    alm0 = healpy.map2alm(mask, lmax=lw, iter=10)
    alm1 = rotate_alm(alm0, lw, pole_rotation(mask))
    sel = select_terms(alm1, lw, ell, ellp, tolerance)
    assert sel.ell == ell and sel.ellp == ellp
    assert sel.keep_m.shape == (2 * ell + 1,) and sel.keep_mp.shape == (2 * ellp + 1,)
    assert sel.keep_pair.shape == (2 * ell + 1, 2 * ellp + 1)
    assert sel.m_band <= 1
    # pair (m, m') sits at [m + ell, m' + ellp]; m == m' is the diagonal offset by ellp - ell
    diagonal = np.eye(2 * ell + 1, 2 * ellp + 1, k=ellp - ell, dtype=bool)
    assert not np.any(sel.keep_pair & ~diagonal)
    assert np.all(sel.keep_pair <= (sel.keep_m[:, None] & sel.keep_mp[None, :]))
    assert 0 < sel.frac_pairs < 0.02 and 0 < sel.frac_m < 1
    assert "pairs" in sel.summary()
    # in the original frame the same rules are far less selective
    sel0 = select_terms(alm0, lw, ell, ellp, tolerance)
    assert sel0.m_band > 10 and sel0.frac_pairs > 5 * sel.frac_pairs


@pytest.mark.parametrize("ellp_offset", [0, 3])
def test_selected_terms_reproduce_baseline_kernel_within_tolerance(ellp_offset):
    """Kernel from the selected (m, m', |M - m| <= k) terms, built with
    banded_integrals_gl, against the full kernel from spin_weighted_integrals_gl."""
    nside = 16
    lw, lmax_out = 3 * nside - 1, 2 * nside - 1
    ell, ellp = nside, nside + ellp_offset
    tolerance = 1e-3
    mask = healpy.read_map(os.path.join(DATA, "baseline_mask.fits"))
    alm = rotate_alm(healpy.map2alm(mask, lmax=lw, iter=10), lw, pole_rotation(mask))
    sel = select_terms(alm, lw, ell, ellp, tolerance)

    def full(ell_):
        return np.array(
            [
                spin_weighted_integrals_gl(alm, lw, ell_, m, lmax_out)
                for m in range(-ell_, ell_ + 1)
            ]
        )

    X = full(ell)
    Y = X if ellp == ell else full(ellp)
    K = kernel_from_integrals(X, Y)

    def banded(ell_, keep):
        lg = max(int(np.ceil((lw + ell_ + lmax_out - 1) / 2)), ell_, lmax_out)
        modes = gl_mask_modes(alm, lw, lg)
        out = np.zeros((2 * ell_ + 1, lmax_out + 1, 2 * lmax_out + 1), dtype=complex)
        for i in np.flatnonzero(keep):
            out[i] = banded_integrals_gl(
                alm,
                lw,
                ell_,
                i - ell_,
                lmax_out,
                sel.m_band,
                lmax_grid=lg,
                mask_modes=modes,
            )
        return out

    Xb = banded(ell, sel.keep_m)
    Yb = Xb if ellp == ell else banded(ellp, sel.keep_mp)
    Ks = kernel_from_integrals(Xb, Yb, sel.keep_pair)
    err = frob(K - Ks) / frob(K)
    assert err <= tolerance, (sel.summary(), err)
    assert sel.frac_pairs < 1.0  # the selection did drop something


def two_blob_alm(nside, lw):
    """Genuinely asymmetric mask: a large cap plus a smaller one 50 deg away.
    In the pole frame P(0) = 0.74, P(1) = 0.077, P(2) = 0.025 (nside 16)."""
    mask = np.clip(
        cap_mask(nside, 47.0, 23.0, 30, 12) + 0.6 * cap_mask(nside, 95.0, 5.0, 18, 8),
        0,
        1,
    )
    return rotate_alm(healpy.map2alm(mask, lmax=lw, iter=10), lw, pole_rotation(mask))


@pytest.mark.xfail(
    strict=True,
    reason=(
        "the default thresholds (eps_m = tol/10, eps_pair = tol/4, delta_band = tol/10) were "
        "validated on the survey mask only; on this asymmetric two-blob mask the kernel error "
        "at tol 1e-3 is 1.36e-3 (ell = ell') and 3.03e-3 (ell' = ell + 3), entirely from the "
        "pair rule (band-only 9e-6 / 1.4e-5, m-only 3e-7); eps_pair = tol/40 still leaves "
        "2.4e-4 / 5.3e-4.  Kept strict so a change of defaults that fixes it shows up here."
    ),
)
@pytest.mark.parametrize("ellp_offset", [0, 3])
def test_default_thresholds_on_asymmetric_mask(ellp_offset):
    nside = 16
    lw, lmax_out = 3 * nside - 1, 2 * nside - 1
    ell, ellp = nside, nside + ellp_offset
    tolerance = 1e-3
    alm = two_blob_alm(nside, lw)
    sel = select_terms(alm, lw, ell, ellp, tolerance)
    X = np.array(
        [
            spin_weighted_integrals_gl(alm, lw, ell, m, lmax_out)
            for m in range(-ell, ell + 1)
        ]
    )
    Y = (
        X
        if ellp == ell
        else np.array(
            [
                spin_weighted_integrals_gl(alm, lw, ellp, m, lmax_out)
                for m in range(-ellp, ellp + 1)
            ]
        )
    )
    K = kernel_from_integrals(X, Y)
    Ms = np.arange(-lmax_out, lmax_out + 1)
    band_x = (
        np.abs(Ms[None, None, :] - np.arange(-ell, ell + 1)[:, None, None])
        <= sel.m_band
    )
    band_y = (
        np.abs(Ms[None, None, :] - np.arange(-ellp, ellp + 1)[:, None, None])
        <= sel.m_band
    )
    Ks = kernel_from_integrals(X * band_x, Y * band_y, sel.keep_pair)
    err = frob(K - Ks) / frob(K)
    assert (
        0.2 < sel.frac_pairs < 0.35 and 20 <= sel.m_band <= 28
    )  # the selection is real
    assert err <= tolerance, (sel.summary(), err)
