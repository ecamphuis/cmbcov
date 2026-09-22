"""
Unit tests for ``cmbcov.mask.mask_spectral_moment``.

``mask_spectral_moment`` is a pure function of a spectrum array (see its
docstring): it computes the power-weighted mean of ``L(L+1)``,

    sum_L (2L+1) L(L+1) wl_L  /  sum_L (2L+1) wl_L ,

which, evaluated on a squared-mask spectrum ``W^2_l``, is the mask-spectral
coefficient that sets the ACC polarisation leakage floor.

These tests check the function against analytically-derived values, not
against numbers read off a run of the code, and check the direction of a
physical effect (sharper mask -> larger moment) rather than a magnitude.
"""

import healpy as hp
import numpy as np
import pytest

from cmbcov.mask import mask_spectral_moment


def test_delta_spectrum_gives_exactly_l_times_lplus1():
    """
    A spectrum with all its power at a single multipole L0 collapses the
    weighted mean to L0(L0+1) exactly, whatever the (nonzero) amplitude at
    L0: the amplitude appears in both the numerator and denominator sums and
    cancels.
    """
    lmax = 40
    for l0 in (0, 1, 17, 40):
        wl = np.zeros(lmax + 1)
        wl[l0] = 3.7  # arbitrary nonzero amplitude; should not matter
        moment = mask_spectral_moment(wl)
        assert moment == pytest.approx(l0 * (l0 + 1), rel=0, abs=1e-10)


def test_flat_spectrum_matches_analytic_weighted_mean():
    """
    For a flat spectrum wl_L = 1 over L = 0 .. N, the weighted mean has a
    closed form derived here (not read off a run of the code):

        (2L+1) L(L+1) = 2L^3 + 3L^2 + L

        sum_{L=0}^{N} 2L^3 = N^2 (N+1)^2 / 2         [2 * (N(N+1)/2)^2]
        sum_{L=0}^{N} 3L^2 = N (N+1) (2N+1) / 2       [3 * N(N+1)(2N+1)/6]
        sum_{L=0}^{N} L    = N (N+1) / 2

        sum_{L=0}^{N} (2L+1) L(L+1)
            = N(N+1)/2 * [ N(N+1) + (2N+1) + 1 ]
            = N(N+1)/2 * (N+1)(N+2)
            = N (N+1)^2 (N+2) / 2

        sum_{L=0}^{N} (2L+1) = (N+1)^2

        weighted mean = [N (N+1)^2 (N+2) / 2] / (N+1)^2 = N (N+2) / 2
    """
    for lmax in (0, 1, 2, 5, 12, 47):
        wl = np.ones(lmax + 1)
        moment = mask_spectral_moment(wl)
        analytic = lmax * (lmax + 2) / 2.0
        assert moment == pytest.approx(analytic, rel=1e-12, abs=1e-12)


def test_lmax_argument_restricts_the_sum():
    """Passing lmax < wl.size - 1 must match evaluating on the truncated array."""
    wl = np.ones(50)
    restricted = mask_spectral_moment(wl, lmax=9)
    truncated_array = mask_spectral_moment(wl[:10])
    assert restricted == pytest.approx(9 * 11 / 2.0, rel=1e-12)
    assert restricted == pytest.approx(truncated_array, rel=1e-12)


def test_lmax_beyond_array_size_raises():
    wl = np.ones(5)
    with pytest.raises(ValueError):
        mask_spectral_moment(wl, lmax=10)


def test_all_zero_spectrum_raises():
    with pytest.raises(ValueError):
        mask_spectral_moment(np.zeros(10))


def _cosine_apodized_cap(nside, radius_deg, width_deg):
    """
    A disc of radius ``radius_deg`` centred on the north pole, tapered to
    zero with a cosine (Hann) profile over the outer ``width_deg`` of the
    radius. Smaller ``width_deg`` means a sharper edge -- more power at high
    L -- for the same disc size.
    """
    npix = hp.nside2npix(nside)
    theta, _ = hp.pix2ang(nside, np.arange(npix))
    radius = np.radians(radius_deg)
    width = np.radians(width_deg)
    inner = radius - width

    mask = np.zeros(npix)
    mask[theta <= inner] = 1.0
    taper = (theta > inner) & (theta <= radius)
    x = (theta[taper] - inner) / width
    mask[taper] = 0.5 * (1.0 + np.cos(np.pi * x))
    return mask


def test_moment_grows_when_mask_is_made_sharper():
    """
    Two cosine-apodised caps of the same size but different taper widths: the
    narrower taper puts more power at high L, so mask_spectral_moment of its
    power spectrum must be larger. Uses the mask's own W_l (not W^2_l) --
    the function is agnostic to which spectrum it is handed, and W_l is
    cheaper to obtain in a test.
    """
    nside = 64
    lmax = 3 * nside - 1
    sharp = _cosine_apodized_cap(nside, radius_deg=30.0, width_deg=2.0)
    smooth = _cosine_apodized_cap(nside, radius_deg=30.0, width_deg=15.0)

    wl_sharp = hp.anafast(sharp, lmax=lmax)
    wl_smooth = hp.anafast(smooth, lmax=lmax)

    moment_sharp = mask_spectral_moment(wl_sharp)
    moment_smooth = mask_spectral_moment(wl_smooth)

    assert moment_sharp > moment_smooth
