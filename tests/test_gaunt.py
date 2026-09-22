"""
Wigner 3j symbols, Gaunt coefficients and the Gaunt-sum form of the ACC
integrals (``tests/reference/gaunt.py``).

The 3j symbols are checked against sympy's exact rational arithmetic
(including ``m != 0`` and the large-``l`` cancellation of the Racah sum);
the Gaunt-sum integrals -- with their ``(-1)^M`` and ``w_{l,-m} = (-1)^m
conj w_{lm}`` phases -- against the quadrature route
``grid.banded_integrals_gl`` on a small mask.
"""

import numpy as np
import pytest

healpy = pytest.importorskip("healpy")
sympy_wigner = pytest.importorskip("sympy.physics.wigner")

from reference.gaunt import (  # noqa: E402
    gaunt,
    gaunt_sum_integrals,
    wigner_3j,
)

from cmbcov.grid import banded_integrals_gl  # noqa: E402

CASES = [
    # (l1, l2, l3, m1, m2, m3, tolerance)
    (2, 3, 4, 1, -2, 1, 1e-14),
    (5, 5, 6, 0, 0, 0, 1e-14),
    (10, 7, 9, 3, -5, 2, 1e-14),
    (1, 1, 2, 1, 1, -2, 1e-14),
    (4, 6, 3, -2, 2, 0, 1e-14),
    (12, 12, 12, 5, 5, -10, 1e-13),
    (3, 2, 1, 0, 0, 0, 1e-14),
    (7, 8, 15, 7, 8, -15, 1e-14),
    (20, 20, 21, 1, 0, -1, 1e-11),
    (30, 25, 40, 10, -3, -7, 1e-8),  # Racah cancellation: measured 1.8e-9
    (50, 40, 60, -20, 15, 5, 5e-7),  # measured 9.3e-8
]


@pytest.mark.parametrize("l1, l2, l3, m1, m2, m3, tol", CASES)
def test_wigner_3j_against_sympy(l1, l2, l3, m1, m2, m3, tol):
    ref = float(sympy_wigner.wigner_3j(l1, l2, l3, m1, m2, m3).evalf(30))
    got = float(wigner_3j(l1, l2, l3, m1, m2, m3))
    assert abs(got - ref) <= tol * abs(ref)


def test_wigner_3j_vectorised_over_l3_and_selection_rules():
    l3 = np.arange(0, 12)
    got = wigner_3j(4, 5, l3, 2, -3, 1)
    ref = np.array(
        [float(sympy_wigner.wigner_3j(4, 5, int(ell), 2, -3, 1)) for ell in l3]
    )
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-14)
    assert np.all(got[l3 < 1] == 0) and np.all(got[l3 > 9] == 0)
    assert wigner_3j(4, 5, 6, 2, -3, 2) == 0  # m1 + m2 + m3 != 0
    assert wigner_3j(4, 5, 6, 5, -3, -2) == 0  # |m1| > l1


def test_gaunt_matches_direct_integral_for_low_l():
    """G(1,1,2; 1,1,-2) = sqrt(3/(10 pi)) * ... checked via sympy's gaunt."""
    from sympy.physics.wigner import gaunt as sympy_gaunt

    for args in [
        (1, 1, 2, 1, 1, -2),
        (2, 3, 5, 0, 1, -1),
        (4, 4, 0, 2, -2, 0),
        (3, 2, 5, -1, -1, 2),
    ]:
        ref = float(sympy_gaunt(*args).evalf(30))
        assert abs(float(gaunt(*args)) - ref) <= 1e-14 * max(abs(ref), 1e-3)
    assert gaunt(2, 2, 3, 0, 0, 0) == 0  # odd l1 + l2 + l3


def test_gaunt_sum_integrals_match_banded_integrals_gl():
    nside, lw, lmax_out, m_band = 16, 12, 10, 4
    theta, phi = healpy.pix2ang(nside, np.arange(healpy.nside2npix(nside)))
    mask = np.exp(-((theta - 1.0) ** 2 / 0.3 + (phi - 2.0) ** 2 / 0.6))
    alm = healpy.map2alm(mask, lmax=lw, iter=3)
    for ell in (0, 3, 8):
        for m in sorted({-ell, -1, 0, ell // 2, ell}):
            if abs(m) > ell:
                continue
            ref = banded_integrals_gl(alm, lw, ell, m, lmax_out, m_band)
            got = gaunt_sum_integrals(alm, lw, ell, m, lmax_out, m_band)
            assert got.shape == ref.shape
            assert np.abs(got - ref).max() / np.abs(ref).max() <= 1e-10, (ell, m)
