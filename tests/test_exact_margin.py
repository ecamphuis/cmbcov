"""
Internal band-limit (``lmax_int``) of the exact covariance.

``Sigma_ll'`` sums ``C_L`` over the mask's coupling range *around* ``l'``, so
the algorithm needs ``C_L`` above the highest multipole it reports: truncating
the sum at the reported ``lmax`` sets ``C_L = 0`` over the upper half of the
kernel and biases the top rows of the matrix.  ``lmax_int`` separates the two
band-limits.  This module pins

* that ``lmax_int=None`` reproduces the historical behaviour bit-for-bit (and
  that that behaviour really is biased, so the parameter is not cosmetic);
* the contract: a row computed to ``lmax`` with ``lmax_int=L`` is the row of
  the whole matrix computed at ``L``, sliced -- TT on both grids and the
  polarised blocks on the GL grid;
* the margin warning: it fires when ``lmax_int - lmax`` is below the mask's
  own coupling width and stays quiet when the margin is comfortable.

Everything runs at NSIDE=16, Lw=24, lmax=6, lmax_int=14 so that the whole
module costs a couple of seconds.
"""

import warnings

import numpy as np
import pytest

healpy = pytest.importorskip("healpy")

from conftest import apodised_cap  # noqa: E402

from cmbcov.exact import (  # noqa: E402
    exact_covariance,
    exact_covariance_pol,
    exact_covariance_row,
    exact_covariance_row_pol,
    mask_coupling_width,
)
from cmbcov.sht import alm2cl, ducc0_map2alm  # noqa: E402

NSIDE = 16
LW = 24
LMAX = 6
LMAX_INT = 14
# The mask is band-limited at Lw, so the coupling is strictly zero beyond
# |dl| > Lw: lmax_int = LMAX + LW is the fully converged reference.
LMAX_CONVERGED = LMAX + LW


@pytest.fixture(scope="module")
def setup():
    """Apodised-cap mask (map and band-limited alm) and a red spectrum."""
    mask = apodised_cap(NSIDE)
    mask_alm = ducc0_map2alm(mask, lmax=LW, iter=10)
    cl = 1000.0 / (np.arange(LMAX_CONVERGED + 1) + 1.0) ** 2
    return mask, mask_alm, cl


@pytest.fixture(autouse=True)
def _fresh_margin_warnings():
    """
    The margin warning fires once per entry point, lmax, lmax_int and width
    per process; start every test from an empty record so that earlier
    (silenced) calls do not swallow the warnings a test expects.
    """
    import cmbcov.exact as exact_module

    exact_module._WARNED_SHORT_MARGINS.clear()
    yield
    exact_module._WARNED_SHORT_MARGINS.clear()


def _quiet(fn, *args, **kw):
    """Call ``fn`` with the margin warning silenced (tested separately)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return fn(*args, **kw)


# --------------------------------------------------------------------------- #
# (a) the default is exactly today's behaviour
# --------------------------------------------------------------------------- #
def test_default_matches_lmax_int_equal_lmax(setup):
    """``lmax_int=None`` is ``lmax_int=lmax``, bit for bit, on both grids."""
    mask, mask_alm, cl = setup
    cl_short = cl[: LMAX + 1]

    gl_default = _quiet(exact_covariance, mask_alm, cl_short, LMAX, grid="gl", lw=LW)
    gl_explicit = _quiet(
        exact_covariance, mask_alm, cl_short, LMAX, grid="gl", lw=LW, lmax_int=LMAX
    )
    assert np.array_equal(gl_default, gl_explicit)

    hp_default = _quiet(exact_covariance_row, mask, cl_short, LMAX, LMAX, nside=NSIDE)
    hp_explicit = _quiet(
        exact_covariance_row, mask, cl_short, LMAX, LMAX, nside=NSIDE, lmax_int=LMAX
    )
    assert np.array_equal(hp_default, hp_explicit)


def test_zero_margin_is_biased(setup):
    """The trap the parameter exists for: no margin, no accuracy."""
    mask, mask_alm, cl = setup
    truncated = _quiet(
        exact_covariance_row, mask_alm, cl[: LMAX + 1], LMAX, LMAX, grid="gl", lw=LW
    )
    converged = _quiet(
        exact_covariance_row,
        mask_alm,
        cl,
        LMAX,
        LMAX,
        grid="gl",
        lw=LW,
        lmax_int=LMAX_CONVERGED,
    )
    ratio = truncated[LMAX] / converged[LMAX]
    # Measured 0.76837 on this mask (the survey mask of the report lost a
    # factor 3.2 the same way); the point is that it is nowhere near 1.
    assert 0.75 < ratio < 0.79


# --------------------------------------------------------------------------- #
# (b) the contract: same numbers as computing at lmax_int and slicing
# --------------------------------------------------------------------------- #
def _assert_same(part, whole_sliced):
    """Tight agreement, plus the bit-identity that currently holds."""
    scale = float(np.abs(whole_sliced).max())
    assert np.allclose(part, whole_sliced, rtol=1e-14, atol=1e-14 * scale)
    # The two paths run the same arithmetic on the same arrays, so they agree
    # exactly today.  The rtol above is the contract; if an optimisation ever
    # breaks the bit-identity, relax this line only, not the one above.
    assert np.array_equal(part, whole_sliced)


def test_gl_rows_match_full_matrix_sliced(setup):
    _, mask_alm, cl = setup
    whole = _quiet(exact_covariance, mask_alm, cl, LMAX_INT, grid="gl", lw=LW)
    part = _quiet(
        exact_covariance, mask_alm, cl, LMAX, grid="gl", lw=LW, lmax_int=LMAX_INT
    )
    _assert_same(part, whole[: LMAX + 1, : LMAX + 1])


def test_gl_single_row_matches_full_matrix_sliced(setup):
    _, mask_alm, cl = setup
    whole_row = _quiet(
        exact_covariance_row, mask_alm, cl, LMAX, LMAX_INT, grid="gl", lw=LW
    )
    part_row = _quiet(
        exact_covariance_row,
        mask_alm,
        cl,
        LMAX,
        LMAX,
        grid="gl",
        lw=LW,
        lmax_int=LMAX_INT,
    )
    _assert_same(part_row, whole_row[: LMAX + 1])


def test_healpix_rows_match_full_matrix_sliced(setup):
    mask, _, cl = setup
    whole = _quiet(exact_covariance, mask, cl, LMAX_INT, nside=NSIDE)
    part = _quiet(exact_covariance, mask, cl, LMAX, nside=NSIDE, lmax_int=LMAX_INT)
    _assert_same(part, whole[: LMAX + 1, : LMAX + 1])


def test_pol_blocks_match_full_matrix_sliced(setup):
    _, mask_alm, cl = setup
    cls = {"TT": cl, "EE": 0.1 * cl, "TE": 0.05 * cl}
    spectra = ("TT", "EE", "TE")
    whole = _quiet(
        exact_covariance_pol, mask_alm, cls, LMAX_INT, spectra=spectra, lw=LW
    )
    part = _quiet(
        exact_covariance_pol,
        mask_alm,
        cls,
        LMAX,
        spectra=spectra,
        lw=LW,
        lmax_int=LMAX_INT,
    )
    assert set(part) == set(whole)
    for pair in part:
        _assert_same(part[pair], whole[pair][: LMAX + 1, : LMAX + 1])


def test_pol_row_matches_full_matrix_sliced(setup):
    _, mask_alm, cl = setup
    cls = {"TT": cl, "EE": 0.1 * cl, "TE": 0.05 * cl}
    spectra = ("TT", "TE")
    whole = _quiet(
        exact_covariance_row_pol, mask_alm, cls, LMAX, LMAX_INT, spectra=spectra, lw=LW
    )
    part = _quiet(
        exact_covariance_row_pol,
        mask_alm,
        cls,
        LMAX,
        LMAX,
        spectra=spectra,
        lw=LW,
        lmax_int=LMAX_INT,
    )
    for pair in part:
        _assert_same(part[pair], whole[pair][: LMAX + 1])


def test_grid_is_sized_from_the_internal_band_limit(setup):
    """
    The GL grid must follow ``lmax_int``, not ``lmax``.

    ``gl_minimal_lmax(lmax, lw)`` would be too small for the ``lmax_int``
    transforms and alias; the explicit minimal grid for ``lmax_int`` must
    reproduce the default exactly.
    """
    from cmbcov.grid import gl_minimal_lmax

    _, mask_alm, cl = setup
    auto = _quiet(
        exact_covariance_row,
        mask_alm,
        cl,
        LMAX,
        LMAX,
        grid="gl",
        lw=LW,
        lmax_int=LMAX_INT,
    )
    explicit = _quiet(
        exact_covariance_row,
        mask_alm,
        cl,
        LMAX,
        LMAX,
        grid="gl",
        lw=LW,
        lmax_int=LMAX_INT,
        lmax_grid=gl_minimal_lmax(LMAX_INT, LW),
    )
    assert np.array_equal(auto, explicit)


def test_lmax_int_below_lmax_raises(setup):
    _, mask_alm, cl = setup
    with pytest.raises(ValueError, match="lmax_int"):
        exact_covariance_row(
            mask_alm, cl, LMAX, LMAX, grid="gl", lw=LW, lmax_int=LMAX - 1
        )


def test_short_cl_raises(setup):
    """``cl`` must reach the internal band-limit, not just the reported one."""
    _, mask_alm, cl = setup
    with pytest.raises(ValueError, match="cl must have at least"):
        _quiet(
            exact_covariance_row,
            mask_alm,
            cl[: LMAX + 1],
            LMAX,
            LMAX,
            grid="gl",
            lw=LW,
            lmax_int=LMAX_INT,
        )


# --------------------------------------------------------------------------- #
# (c) the margin warning, and the width estimator behind it
# --------------------------------------------------------------------------- #
def test_coupling_width_of_this_mask(setup):
    """The width is a property of the mask: positive, and bounded by Lw."""
    _, mask_alm, _ = setup
    width = mask_coupling_width(alm2cl(mask_alm))
    assert 0 < width <= LW
    assert width == 7  # measured on the nside=16 apodised cap at Lw=24


def test_coupling_width_full_sky_is_zero():
    """All the power in the monopole: a full-sky mask couples nothing."""
    wl = np.zeros(LW + 1)
    wl[0] = 4 * np.pi
    assert mask_coupling_width(wl) == 0
    assert mask_coupling_width(np.zeros(LW + 1)) == 0


def test_coupling_width_grows_with_coverage(setup):
    _, mask_alm, _ = setup
    wl = alm2cl(mask_alm)
    widths = [mask_coupling_width(wl, c) for c in (0.5, 0.9, 0.99, 0.999)]
    assert widths == sorted(widths)


def test_width_margin_is_enough(setup):
    """
    Calibration of the estimator: a margin of one coupling width brings the
    diagonal to ~1e-4 of its converged value (measured 1.4e-4 here; the
    converged reference is lmax_int = lmax + Lw, beyond which the coupling of
    a band-limited mask is exactly zero).
    """
    _, mask_alm, cl = setup
    width = mask_coupling_width(alm2cl(mask_alm))
    at_width = _quiet(
        exact_covariance_row,
        mask_alm,
        cl,
        LMAX,
        LMAX,
        grid="gl",
        lw=LW,
        lmax_int=LMAX + width,
    )
    converged = _quiet(
        exact_covariance_row,
        mask_alm,
        cl,
        LMAX,
        LMAX,
        grid="gl",
        lw=LW,
        lmax_int=LMAX_CONVERGED,
    )
    assert abs(at_width[LMAX] / converged[LMAX] - 1.0) < 1e-3


@pytest.mark.parametrize("grid", ["gl", "healpix"])
def test_warns_when_margin_too_small(setup, grid):
    mask, mask_alm, cl = setup
    m = mask_alm if grid == "gl" else mask
    kw = {"lw": LW} if grid == "gl" else {"nside": NSIDE}
    with pytest.warns(UserWarning, match="coupling width 7"):
        exact_covariance_row(m, cl[: LMAX + 1], LMAX, LMAX, grid=grid, **kw)
    # the message names the margin supplied as well as the width estimated
    with pytest.warns(UserWarning, match="margin 3"):
        exact_covariance_row(m, cl, LMAX, LMAX, grid=grid, lmax_int=LMAX + 3, **kw)


@pytest.mark.parametrize("grid", ["gl", "healpix"])
def test_no_warning_when_margin_is_comfortable(setup, grid):
    mask, mask_alm, cl = setup
    m = mask_alm if grid == "gl" else mask
    kw = {"lw": LW} if grid == "gl" else {"nside": NSIDE}
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        exact_covariance_row(m, cl, LMAX, LMAX, grid=grid, lmax_int=LMAX_INT, **kw)


def test_full_sky_mask_never_warns():
    """A mask with no coupling width must not produce a spurious warning."""
    mask = np.ones(healpy.nside2npix(NSIDE))
    cl = 1000.0 / (np.arange(LMAX + 1) + 1.0) ** 2
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        exact_covariance_row(mask, cl, LMAX, LMAX, nside=NSIDE)


def test_matrix_entry_points_warn(setup):
    """``exact_covariance`` and the polarised entry points check too."""
    _, mask_alm, cl = setup
    cls = {"TT": cl[: LMAX + 1]}
    with pytest.warns(UserWarning, match="coupling width"):
        exact_covariance(mask_alm, cl[: LMAX + 1], LMAX, grid="gl", lw=LW, rows=[0])
    with pytest.warns(UserWarning, match="coupling width"):
        exact_covariance_pol(mask_alm, cls, LMAX, spectra=("TT",), lw=LW, rows=[0])
    with pytest.warns(UserWarning, match="coupling width"):
        exact_covariance_row_pol(mask_alm, cls, 0, LMAX, spectra=("TT",), lw=LW)


def test_row_loop_warns_once(setup):
    """A loop over rows warns once per process, not once per row."""
    _, mask_alm, cl = setup
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for ellp in range(LMAX + 1):
            exact_covariance_row(mask_alm, cl, ellp, LMAX, grid="gl", lw=LW)
    margin = [w for w in caught if "coupling width" in str(w.message)]
    assert len(margin) == 1
    assert "Warned once per process" in str(margin[0].message)
