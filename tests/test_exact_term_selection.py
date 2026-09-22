r"""
Term selection in the exact GL covariance
(``exact_covariance_row(..., term_selection=)`` and friends).

Only the ``m'`` rule applies: the row is a sum over all ``m'`` of the
mask-weighted single-harmonic columns, so the ``m'`` whose mask overlap
``p_{m'}`` is below ``tolerance / 10`` of the largest are skipped after the
mask has been rotated to the pole frame (the row is rotation invariant).
Measured when written, on a 20 deg cosine-apodised cap at (45, 30), nside 16,
lmax 32, tolerance 1e-3: max relative error on elements above 1e-6 of the
diagonal 1.5e-5 (l' = 10), 6.3e-8 (20), 1.7e-6 (32); fraction of ``m'``
skipped 0.48 / 0.54 / 0.59 (the geometric expectation for a cap of radius
``r`` is ``1 - sin r`` = 0.66).
"""

import warnings

import numpy as np
import pytest

healpy = pytest.importorskip("healpy")

from cmbcov import exact  # noqa: E402
from cmbcov.exact import (  # noqa: E402
    _kept_mprime,
    _mask_alm_for_gl,
    _pole_frame_alm,
    exact_covariance,
    exact_covariance_row,
    exact_covariance_row_pol,
)

NSIDE, LMAX, TOL = 16, 32, 1e-3


def cap_mask(nside, lon, lat, radius_deg, apod_deg):
    v = healpy.ang2vec(lon, lat, lonlat=True)
    pv = np.array(healpy.pix2vec(nside, np.arange(healpy.nside2npix(nside))))
    dist = np.degrees(np.arccos(np.clip(v @ pv, -1, 1)))
    x = np.clip((dist - (radius_deg - apod_deg)) / apod_deg, 0, 1)
    return 0.5 * (1 + np.cos(np.pi * x))


@pytest.fixture(scope="module")
def setup():
    mask = cap_mask(NSIDE, 45.0, 30.0, 20.0, 8.0)
    ell = np.arange(LMAX + 1)
    cl = np.zeros(LMAX + 1)
    cl[2:] = 1e-3 / (ell[2:] * (ell[2:] + 1)) ** 0.9
    return mask, cl


def _rel_err(sel, full):
    big = np.abs(full) > 1e-6 * np.abs(full).max()
    return (np.abs(sel - full)[big] / np.abs(full)[big]).max()


@pytest.mark.parametrize("ellp", [10, 20, 32])
def test_row_with_term_selection_matches_the_full_row(setup, ellp, capsys):
    mask, cl = setup
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # band-limit margin at l' = lmax
        full = exact_covariance_row(mask, cl, ellp, LMAX, grid="gl")
        sel = exact_covariance_row(mask, cl, ellp, LMAX, grid="gl", term_selection=TOL)
    err = _rel_err(sel, full)
    alm, lw = _mask_alm_for_gl(mask, None, None)
    keep = _kept_mprime(_pole_frame_alm(mask, alm, lw), lw, ellp, TOL)
    skipped = 1.0 - keep.mean()
    with capsys.disabled():
        print(
            f"\n  exact row l'={ellp}: max rel err {err:.2e}, "
            f"fraction of m' skipped {skipped:.3f} ({keep.size - keep.sum()}/{keep.size})"
        )
    assert err <= TOL, err
    assert skipped > 0.3, skipped


def test_selection_is_correct_in_any_frame_and_symmetric_in_m(setup):
    """The row does not depend on the frame; the kept set is even in m'."""
    mask, cl = setup
    alm, lw = _mask_alm_for_gl(mask, None, None)
    rotated = _pole_frame_alm(mask, alm, lw)
    plain = exact.__dict__["_exact_row_gl"](alm, lw, cl, 20, LMAX)
    in_pole = exact.__dict__["_exact_row_gl"](rotated, lw, cl, 20, LMAX)
    np.testing.assert_allclose(in_pole, plain, rtol=1e-11)
    keep = _kept_mprime(rotated, lw, 20, TOL)
    assert np.array_equal(keep, keep[::-1])
    # the alm-input entry point rotates through a synthesised map
    from_alm = exact_covariance_row(alm, cl, 20, LMAX, grid="gl", term_selection=TOL)
    from_map = exact_covariance_row(mask, cl, 20, LMAX, grid="gl", term_selection=TOL)
    np.testing.assert_allclose(from_alm, from_map, rtol=1e-9)


def test_matrix_and_worker_plumbing_carry_the_selection(setup):
    mask, cl = setup
    row = exact_covariance_row(mask, cl, 20, LMAX, grid="gl", term_selection=TOL)
    matrix = exact_covariance(mask, cl, LMAX, grid="gl", rows=(20,), term_selection=TOL)
    np.testing.assert_allclose(matrix[:, 20], row, rtol=1e-13)
    # the process-pool worker path, exercised in this process
    alm, lw = _mask_alm_for_gl(mask, None, None)
    rotated = _pole_frame_alm(mask, alm, lw)
    lmax_grid = exact.gl_minimal_lmax(LMAX, lw)
    exact._row_worker_init(rotated, lw, cl, LMAX, lmax_grid, True, 1, None, TOL)
    ellp, worker_row = exact._row_worker(20)
    assert ellp == 20
    np.testing.assert_allclose(worker_row, row, rtol=1e-13)
    exact._row_worker_init(rotated, lw, cl, LMAX, lmax_grid, True, 1, None, None)
    _, worker_full = exact._row_worker(20)
    assert _rel_err(worker_row, worker_full) <= TOL


def test_polarised_row_with_term_selection(setup):
    mask, cl = setup
    cls = {"TT": cl, "EE": 0.1 * cl, "BB": 0.01 * cl, "TE": 0.3 * cl}
    spectra = ("TT", "TE", "EE")
    full = exact_covariance_row_pol(mask, cls, 20, LMAX, spectra=spectra)
    sel = exact_covariance_row_pol(
        mask, cls, 20, LMAX, spectra=spectra, term_selection=TOL
    )
    for pair in full:
        assert _rel_err(sel[pair], full[pair]) <= TOL, pair


def test_term_selection_is_gl_only(setup):
    mask, cl = setup
    with pytest.raises(ValueError, match="grid='gl'"):
        exact_covariance_row(
            mask, cl, 5, LMAX, NSIDE, grid="healpix", term_selection=TOL
        )
    with pytest.raises(ValueError, match="grid='gl'"):
        exact_covariance(mask, cl, LMAX, NSIDE, grid="healpix", term_selection=TOL)
    with pytest.raises(ValueError, match="grid='gl'"):
        exact_covariance_row_pol(
            mask,
            {"TT": cl},
            5,
            LMAX,
            spectra=("TT",),
            grid="healpix",
            nside=NSIDE,
            term_selection=TOL,
        )
