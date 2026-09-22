"""
Spin-2 complex transforms on the Gauss-Legendre grid
(``cmbcov.grid`` with ``spin=2``).

The polarised exact covariance applies ``K_2 = A_2 W S_2`` to unit vectors
``e_{E, l'm'}`` / ``e_{B, l'm'}`` at a single ``(l', +m')``, which have no
reality partner: the (Q, U) maps are complex and their (E, B) coefficients
live in the full-``M`` layout.  These tests pin the two facts the covariance
relies on: the spin-2 complex round trip is exact, and the complex analysis is
the exact adjoint of the complex synthesis under the GL quadrature weights,
which is what makes ``K`` Hermitian with no adjoint correction.
"""

import numpy as np
import pytest

healpy = pytest.importorskip("healpy")

from cmbcov.grid import (  # noqa: E402
    full_from_pair,
    gl_analysis,
    gl_analysis_complex,
    gl_quadrature_weights,
    gl_shape,
    gl_synthesis,
    gl_synthesis_complex,
    pair_from_full,
)

LMAX = 40


def random_full_m(rng, lmax, ncomp=None, lmin=0):
    """Full-M coefficients with independent +M and -M entries (a complex map);
    ``ncomp=2`` gives an (E, B) pair, zero below ``lmin``."""
    shape = (lmax + 1, 2 * lmax + 1)
    if ncomp is not None:
        shape = (ncomp,) + shape
    full = np.zeros(shape, dtype=complex)
    ell, m = np.indices(shape[-2:])
    ok = (np.abs(m - lmax) <= ell) & (ell >= lmin)
    n = ok.sum()
    for comp in full.reshape(-1, *shape[-2:]):
        comp[ok] = rng.normal(size=n) + 1j * rng.normal(size=n)
    return full


def random_complex_map(rng, lmax_grid, ncomp=None):
    shape = gl_shape(lmax_grid)
    if ncomp is not None:
        shape = (ncomp,) + shape
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def rel_err(a, b):
    return np.abs(a - b).max() / np.abs(b).max()


def test_pair_conversions_preserve_leading_dimensions():
    """full_from_pair / pair_from_full act component-wise on an (E, B) pair
    exactly as on each component alone, and invert each other."""
    rng = np.random.default_rng(0)
    nalm = healpy.Alm.getsize(LMAX)
    _, m = healpy.Alm.getlm(LMAX)
    r = rng.normal(size=(2, nalm)) + 1j * rng.normal(size=(2, nalm))
    s = rng.normal(size=(2, nalm)) + 1j * rng.normal(size=(2, nalm))
    r[:, m == 0] = r[:, m == 0].real  # real-map alms have real m = 0
    s[:, m == 0] = s[:, m == 0].real
    full = full_from_pair(r, s, LMAX)
    assert full.shape == (2, LMAX + 1, 2 * LMAX + 1)
    for i in range(2):
        np.testing.assert_array_equal(full[i], full_from_pair(r[i], s[i], LMAX))
    r2, s2 = pair_from_full(full, LMAX)
    np.testing.assert_allclose(r2, r, rtol=1e-14)
    np.testing.assert_allclose(s2, s, rtol=1e-14)


def test_spin2_complex_round_trip_is_exact():
    """analysis(synthesis(x)) == x to 1e-13 for an (E, B) full-M set with
    independent +M and -M entries, on the grid Lg = lmax; the (Q, U) maps are
    genuinely complex.  Measured 1.6e-14."""
    rng = np.random.default_rng(1)
    full = random_full_m(rng, LMAX, ncomp=2, lmin=2)
    qu = gl_synthesis_complex(full, LMAX, LMAX, spin=2)
    assert qu.shape == (2,) + gl_shape(LMAX)
    assert np.iscomplexobj(qu) and np.abs(qu.imag).max() > 0.1 * np.abs(qu).max()
    back = gl_analysis_complex(qu, LMAX, LMAX, spin=2)
    assert back.shape == full.shape
    assert rel_err(back, full) < 1e-13


def test_spin2_complex_extends_real_transform_linearly():
    """A complex (Q, U) pair is a real pair plus i times another: the spin-2
    complex synthesis of full_from_pair(r, s) is S_2 r + i S_2 s exactly."""
    rng = np.random.default_rng(2)
    nalm = healpy.Alm.getsize(LMAX)
    ell, m = healpy.Alm.getlm(LMAX)

    def real_eb():
        a = rng.normal(size=(2, nalm)) + 1j * rng.normal(size=(2, nalm))
        a[:, m == 0] = a[:, m == 0].real
        a[:, ell < 2] = 0.0
        return a

    r, s = real_eb(), real_eb()
    expected = gl_synthesis(r, LMAX, LMAX, spin=2) + 1j * gl_synthesis(
        s, LMAX, LMAX, spin=2
    )
    got = gl_synthesis_complex(full_from_pair(r, s, LMAX), LMAX, LMAX, spin=2)
    assert rel_err(got, expected) < 1e-14


@pytest.mark.parametrize("spin", [0, 2])
def test_complex_analysis_is_adjoint_of_complex_synthesis(spin):
    """
    <S x, y>_map == <x, A y>_alm with the GL quadrature weights, for complex
    x (full-M) and complex y (maps), summed over Q and U for spin 2.  This is
    the identity behind K^dagger = K in the exact covariance.  Measured 2e-16
    for both spins.
    """
    rng = np.random.default_rng(10 + spin)
    ncomp = None if spin == 0 else 2
    x = random_full_m(rng, LMAX, ncomp=ncomp, lmin=spin)
    y = random_complex_map(rng, LMAX, ncomp=ncomp)
    w = gl_quadrature_weights(LMAX)
    lhs = np.sum(w * gl_synthesis_complex(x, LMAX, LMAX, spin=spin) * np.conj(y))
    rhs = np.sum(x * np.conj(gl_analysis_complex(y, LMAX, LMAX, spin=spin)))
    assert abs(lhs - rhs) < 1e-13 * abs(lhs)


def test_spin0_signature_unchanged():
    """Existing callers pass (alm_full, lmax, lmax_grid[, nthreads=]) and get
    the 2-D spin-0 result; ``spin`` is a keyword defaulting to 0."""
    rng = np.random.default_rng(3)
    full = random_full_m(rng, LMAX)
    mp = gl_synthesis_complex(full, LMAX, LMAX)
    assert mp.shape == gl_shape(LMAX)
    np.testing.assert_array_equal(mp, gl_synthesis_complex(full, LMAX, LMAX, spin=0))
    back = gl_analysis_complex(mp, LMAX, LMAX, nthreads=1)
    assert back.shape == full.shape
    assert rel_err(back, full) < 1e-13


def test_spin2_shape_validation():
    rng = np.random.default_rng(4)
    with pytest.raises(ValueError):
        gl_synthesis_complex(random_full_m(rng, LMAX), LMAX, LMAX, spin=2)
    with pytest.raises(ValueError):
        gl_synthesis_complex(random_full_m(rng, LMAX, ncomp=2), LMAX, LMAX, spin=0)
    with pytest.raises(ValueError):
        gl_analysis(random_complex_map(rng, LMAX).real, LMAX, LMAX, spin=2)
