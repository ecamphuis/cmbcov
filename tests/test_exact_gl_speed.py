"""
Invariance of the fast GL row against the literal implementation.

``exact.py``'s literal GL reference (``_exact_row_gl(..., reference=True)``)
builds, per ``(l', m')`` column, two complex maps and two
``(lmax+1, 2 lmax+1)`` full-``M`` coefficient arrays.  Profiling one column
at ``lmax=500``, ``Lw=384`` (grid 692) showed only 55% of the time inside
ducc0; 35% went into ``full_from_pair`` / ``pair_from_full`` and 5% into the
complex maps.  The fast path instead carries the column as the pair of
real-map alms
(:func:`~cmbcov.exact._correlation_column_pair_gl`), the
single-harmonic synthesis of step 1 uses one azimuthal order only
(:func:`~cmbcov.grid.gl_synthesis_single_m`), and the
``sum_m |c|^2`` is done on the healpy half of the coefficients.

These tests assert *no timings* -- they are flaky -- but they pin every step
of that rewrite against the
literal version kept behind ``_exact_row_gl(..., reference=True)``:

* the single-order synthesis is bit-identical to the full one;
* the unit-vector pair is bit-identical to ``pair_from_full``;
* the ``|c|^2`` accumulation is the full-``M`` sum;
* the row is the reference row to 1e-13 relative, element by element,
  with and without the ``m' -> -m'`` symmetry;
* ``nprocs > 1`` is bit-identical to the serial loop.
"""

import os

import numpy as np
import pytest

healpy = pytest.importorskip("healpy")

from cmbcov.exact import (  # noqa: E402
    _abs2_sum_over_m,
    _exact_row_gl,
    _unit_pair_alm,
    exact_covariance,
)
from cmbcov.grid import (  # noqa: E402
    full_from_pair,
    gl_analysis_complex,
    gl_minimal_lmax,
    gl_step1_minimal_lmax,
    gl_synthesis,
    gl_synthesis_complex,
    gl_synthesis_single_m,
    pair_from_full,
)
from cmbcov.sht import ducc0_map2alm  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

NSIDE = 32
LW = 48
LMAX = 32


def red_cl(lmax):
    return 1000.0 / (np.arange(lmax + 1) + 1.0) ** 2


@pytest.fixture(scope="module")
def baseline():
    mask = healpy.read_map(os.path.join(DATA_DIR, "baseline_mask.fits"))
    return ducc0_map2alm(mask, lmax=LW, iter=10), red_cl(LMAX)


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("lmax,lmax_grid", [(32, 56), (40, 40)])
def test_single_m_synthesis_is_bit_identical(lmax, lmax_grid):
    """
    ``gl_synthesis_single_m`` drops every azimuthal order but one from the
    Legendre stage.  The dropped orders contribute exact zeros when the alm has
    none, so the map must agree with the full synthesis to the last bit -- not
    merely to round-off.
    """
    rng = np.random.default_rng(lmax)
    nalm = healpy.Alm.getsize(lmax)
    ell, m = healpy.Alm.getlm(lmax)
    for mm in (0, 1, 7, lmax):
        alm = np.zeros(nalm, dtype=complex)
        sel = m == mm
        alm[sel] = rng.normal(size=sel.sum()) + 1j * rng.normal(size=sel.sum())
        if mm == 0:
            alm[sel] = alm[sel].real
        got = gl_synthesis_single_m(alm, lmax, mm, lmax_grid, nthreads=2)
        ref = gl_synthesis(alm, lmax, lmax_grid, nthreads=2)
        np.testing.assert_array_equal(got, ref)


def test_single_m_synthesis_validation():
    alm = np.zeros(healpy.Alm.getsize(8), dtype=complex)
    with pytest.raises(ValueError):
        gl_synthesis_single_m(alm, 8, 9, 12)
    with pytest.raises(ValueError):
        gl_synthesis_single_m(alm, 8, -1, 12)
    with pytest.raises(ValueError):
        gl_synthesis_single_m(np.zeros(3, dtype=complex), 8, 2, 12)


@pytest.mark.parametrize("mp", [0, 1, 4, -1, -4])
def test_unit_pair_alm_matches_pair_from_full(mp):
    """``_unit_pair_alm`` is ``pair_from_full`` of the full-M unit vector."""
    lmax, ellp = 12, 9
    e = np.zeros((lmax + 1, 2 * lmax + 1), dtype=complex)
    e[ellp, lmax + mp] = 1.0
    r_ref, s_ref = pair_from_full(e, lmax)
    r, s, has_s = _unit_pair_alm(lmax, ellp, mp)
    np.testing.assert_array_equal(r, r_ref)
    np.testing.assert_array_equal(s, s_ref)
    assert has_s == bool(np.any(s_ref))


def test_abs2_sum_over_m_matches_full_m_sum():
    """
    ``sum_m |c[l, m]|^2`` from the healpy pair equals the sum over the full-M
    column, with ``|c[l, -m]| = |conj(ra) + i conj(sa)|``.
    """
    lmax = 20
    rng = np.random.default_rng(0)
    nalm = healpy.Alm.getsize(lmax)
    ell = healpy.Alm.getlm(lmax)[0]
    ra = rng.normal(size=nalm) + 1j * rng.normal(size=nalm)
    sa = rng.normal(size=nalm) + 1j * rng.normal(size=nalm)
    ref = np.sum(np.abs(full_from_pair(ra, sa, lmax)) ** 2, axis=1)
    np.testing.assert_allclose(_abs2_sum_over_m(ra, sa, ell, lmax), ref, rtol=1e-14)
    ref0 = np.sum(np.abs(full_from_pair(ra, np.zeros_like(ra), lmax)) ** 2, axis=1)
    np.testing.assert_allclose(_abs2_sum_over_m(ra, None, ell, lmax), ref0, rtol=1e-14)


# --------------------------------------------------------------------------- #
# The row itself
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ellp", [0, 1, 7, 20, 32])
def test_fast_row_equals_reference_row(baseline, ellp):
    """
    The real-pair row and the literal complex-map row agree element by element
    to 1e-13 relative.  They are not bit-identical: skipping the full-M round
    trip around the ``C_L`` multiply replaces ``(a-b) + (a+b)`` by ``2a``, an
    ulp.  The rows here span up to ten decades, so this is a per-element bound
    on the smallest entries too, not a row-normalised one.
    """
    mask_alm, cl = baseline
    fast = _exact_row_gl(mask_alm, LW, cl, ellp, LMAX, nthreads=2)
    ref = _exact_row_gl(mask_alm, LW, cl, ellp, LMAX, nthreads=2, reference=True)
    np.testing.assert_allclose(fast, ref, rtol=1e-13)


def test_fast_row_equals_reference_without_symmetry(baseline):
    """The same for the full ``m' = -l'..l'`` loop, which exercises ``m' < 0``."""
    mask_alm, cl = baseline
    kw = {"nthreads": 2, "use_symmetry": False}
    fast = _exact_row_gl(mask_alm, LW, cl, 11, LMAX, **kw)
    ref = _exact_row_gl(mask_alm, LW, cl, 11, LMAX, reference=True, **kw)
    np.testing.assert_allclose(fast, ref, rtol=1e-13)


def test_nprocs_is_bit_identical(baseline):
    """
    Rows are independent and each is computed by one call chain, so spreading
    them over processes must not move a single bit.  ``nprocs`` is rejected on
    the HEALPix path.
    """
    mask_alm, cl = baseline
    rows = (3, 9, 16)
    serial = exact_covariance(mask_alm, cl, LMAX, rows=rows, grid="gl", nthreads=2)
    parallel = exact_covariance(
        mask_alm, cl, LMAX, rows=rows, grid="gl", nthreads=2, nprocs=3
    )
    np.testing.assert_array_equal(parallel, serial)
    with pytest.raises(ValueError):
        exact_covariance(mask_alm, cl, LMAX, rows=rows, grid="gl", nprocs=0)
    with pytest.raises(ValueError):
        exact_covariance(np.zeros(12), cl, LMAX, nside=1, rows=rows, nprocs=2)


# --------------------------------------------------------------------------- #
# The per-row (step 1) grid rule -- derived and verified, not used by default
# --------------------------------------------------------------------------- #
def test_gl_step1_minimal_lmax_rule():
    """
    ``Lg_1 = max(lmax, ceil((lw + ell + lmax - 1) / 2))``: step 1 integrates the
    mask against a *single* harmonic of degree ``ell``, not against the whole
    band, but ``analysis_2d`` still cannot resolve coefficients above the grid
    band-limit, which is what makes the ``lmax`` floor binding.
    """
    assert gl_step1_minimal_lmax(500, 384, 500) == gl_minimal_lmax(500, 384) == 692
    assert gl_step1_minimal_lmax(500, 384, 50) == 500  # floor, not 467
    assert gl_step1_minimal_lmax(500, 384, 250) == 567
    assert gl_step1_minimal_lmax(32, 48, 5) == 42
    with pytest.raises(ValueError):
        gl_step1_minimal_lmax(-1, 0, 0)


@pytest.mark.parametrize("ellp", [5, 16, 32])
def test_step1_is_exact_on_its_own_minimal_grid(ellp):
    """
    Step 1 (``analysis(W Y_{l'm'})``) on ``Lg_1`` reproduces the result on a
    generously large grid to 1e-13, and aliases one step below whenever the
    ``lmax`` floor is not what is binding.

    The mask is white up to ``LW`` (as in ``tests/test_grid.py``), not the
    baseline mask: a real apodised mask carries almost no power at ``l ~ Lw``,
    so the aliasing it produces one step below the minimum is ~1e-3 of the
    result rather than the O(0.1) cliff the rule is about.
    """
    rng = np.random.default_rng(ellp)
    m = healpy.Alm.getlm(LW)[1]
    mask_alm = rng.normal(size=m.size) + 1j * rng.normal(size=m.size)
    mask_alm[m == 0] = mask_alm[m == 0].real
    lg1 = gl_step1_minimal_lmax(LMAX, LW, ellp)

    def step1(lg):
        w = gl_synthesis(mask_alm, LW, lg)
        e = np.zeros((LMAX + 1, 2 * LMAX + 1), dtype=complex)
        e[ellp, LMAX + 3] = 1.0
        return gl_analysis_complex(w * gl_synthesis_complex(e, LMAX, lg), LMAX, lg)

    ref = step1(LMAX + LW + 5)
    scale = np.abs(ref).max()
    assert np.abs(step1(lg1) - ref).max() < 1e-13 * scale
    assert 2 * lg1 + 1 >= LW + ellp + LMAX
    if lg1 > LMAX:
        assert np.abs(step1(lg1 - 1) - ref).max() > 1e-2 * scale
