"""
Gauss-Legendre grid primitives (``cmbcov.grid``).

On a GL grid ducc0's ``analysis_2d`` is the exact inverse and exact adjoint of
``synthesis_2d``, which is what lets the exact covariance and the ACC
precomputation be exact to machine precision once the mask is band-limited.
These tests pin the three facts the package relies on: the round trip (real,
spin-2 and complex maps), the minimal-grid rule ``Lg = lmax + ceil((Lw-1)/2)``
including its failure one step below, and the quadrature weights.
"""

import numpy as np
import pytest

healpy = pytest.importorskip("healpy")

from cmbcov.grid import (  # noqa: E402
    full_from_pair,
    gl_analysis,
    gl_analysis_complex,
    gl_minimal_lmax,
    gl_quadrature_weights,
    gl_shape,
    gl_synthesis,
    gl_synthesis_complex,
)

LMAX = 40


def random_alm(rng, lmax, ncomp=None, lmin=0):
    """Random healpy alm set (real m=0, zero below lmin)."""
    ell, m = healpy.Alm.getlm(lmax)
    shape = (healpy.Alm.getsize(lmax),) if ncomp is None else (ncomp, len(ell))
    alm = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    alm[..., m == 0] = alm[..., m == 0].real
    alm[..., ell < lmin] = 0.0
    return alm


def random_full_m(rng, lmax):
    """Full-M coefficients with independent +M and -M entries (a complex map)."""
    full = np.zeros((lmax + 1, 2 * lmax + 1), dtype=complex)
    ell, m = np.indices(full.shape)
    ok = np.abs(m - lmax) <= ell
    full[ok] = rng.normal(size=ok.sum()) + 1j * rng.normal(size=ok.sum())
    return full


def rel_err(a, b):
    return np.abs(a - b).max() / np.abs(b).max()


def test_gl_shape_and_quadrature_weights():
    """Lg+1 rings, 2Lg+2 points per ring; weights sum to 4 pi and integrate
    |Y_lm|^2 (degree 2 Lg <= 2 Lg + 1) to exactly 1."""
    lg = 25
    assert gl_shape(lg) == (26, 52)
    w = gl_quadrature_weights(lg)
    assert w.shape == (26, 52)
    np.testing.assert_allclose(w.sum(), 4 * np.pi, rtol=1e-14)
    for ell, m in ((lg, 0), (lg, 7), (lg, lg)):
        e = np.zeros((ell + 1, 2 * ell + 1), dtype=complex)
        e[ell, ell + m] = 1.0
        y = gl_synthesis_complex(e, ell, lg)
        np.testing.assert_allclose(np.sum(w * np.abs(y) ** 2), 1.0, rtol=1e-13)


def test_gl_minimal_lmax_rule():
    """Lg = lmax + ceil((Lw-1)/2), never below lmax (values from the spike)."""
    assert gl_minimal_lmax(32, 48) == 56
    assert gl_minimal_lmax(32, 47) == 55
    assert gl_minimal_lmax(32, 24) == 44
    assert gl_minimal_lmax(32, 64) == 64
    assert gl_minimal_lmax(32, 0) == 32
    assert gl_minimal_lmax(32, 1) == 32
    assert gl_minimal_lmax(1000, 500) == 1250
    with pytest.raises(ValueError):
        gl_minimal_lmax(-1, 0)


@pytest.mark.parametrize("spin", [0, 2])
def test_round_trip_is_exact(spin):
    """analysis(synthesis(a)) == a to 1e-13 on the grid Lg = lmax."""
    rng = np.random.default_rng(1)
    if spin == 0:
        alm = random_alm(rng, LMAX)
    else:
        alm = random_alm(rng, LMAX, ncomp=2, lmin=spin)
    mp = gl_synthesis(alm, LMAX, LMAX, spin=spin)
    assert mp.shape == ((2,) if spin else ()) + gl_shape(LMAX)
    back = gl_analysis(mp, LMAX, LMAX, spin=spin)
    assert rel_err(back, alm) < 1e-13


def test_complex_round_trip_with_independent_pm_m():
    """
    A complex map has independent +M and -M coefficients.  Transforming Re and
    Im separately recovers them to 1e-13; assuming a real map (analysing only
    the real part and mirroring) is wrong by O(1).
    """
    rng = np.random.default_rng(2)
    full = random_full_m(rng, LMAX)
    mp = gl_synthesis_complex(full, LMAX, LMAX)
    assert np.iscomplexobj(mp) and np.abs(mp.imag).max() > 0.1 * np.abs(mp).max()
    back = gl_analysis_complex(mp, LMAX, LMAX)
    assert rel_err(back, full) < 1e-13

    r = gl_analysis(mp.real, LMAX, LMAX)
    naive = full_from_pair(r, np.zeros_like(r), LMAX)
    assert rel_err(naive, full) > 0.1


@pytest.mark.parametrize("lw", [47, 48])
def test_minimal_grid_cliff_on_masked_single_mode(lw):
    """
    Analysing W * Y_lm (W white up to Lw, l = lmax) up to lmax is exact at
    Lg = gl_minimal_lmax and aliased one step below: the integrand
    W Y_lm Y*_l'm' has degree Lw + 2 lmax, and GL with Lg+1 nodes integrates
    degree 2 Lg + 1.  Both parities of Lw are covered (ceil matters).
    Measured: 0.2-0.9 at Lg-1, 1e-14 at Lg.
    """
    lmax = 32
    rng = np.random.default_rng(lw)
    mask_alm = random_alm(rng, lw)
    lg = gl_minimal_lmax(lmax, lw)
    assert 2 * lg + 1 >= lw + 2 * lmax > 2 * (lg - 1) + 1

    e = np.zeros((lmax + 1, 2 * lmax + 1), dtype=complex)
    e[lmax, lmax + 5] = 1.0

    def product_alm(grid):
        w = gl_synthesis(mask_alm, lw, grid)
        y = gl_synthesis_complex(e, lmax, grid)
        return gl_analysis_complex(w * y, lmax, grid)

    ref = product_alm(lmax + lw + 5)
    assert rel_err(product_alm(lg), ref) < 1e-13
    assert rel_err(product_alm(lg + 1), ref) < 1e-13
    assert rel_err(product_alm(lg - 1), ref) > 1e-2


def test_synthesis_allows_lmax_above_grid_but_analysis_does_not():
    """Synthesis is point-wise evaluation (any grid); analysis needs Lg >= lmax."""
    rng = np.random.default_rng(3)
    alm = random_alm(rng, 60)
    mp = gl_synthesis(alm, 60, 40)
    assert mp.shape == gl_shape(40)
    with pytest.raises(ValueError):
        gl_analysis(mp, 60, 40)
    with pytest.raises(ValueError):
        gl_analysis(mp, 30, 41)  # shape does not match the declared grid
