"""
Regression tests for the native MASTER coupling kernels.

These pin the numerics of `cmbcov.kernels.coupling` against
two independent references:

1. The analytic full-sky limit (W_L = 4*pi*delta_L0  =>  M = identity).
2. A brute-force O(lmax^3) evaluation of Camphuis et al. (2022) Eq. (3) using
   exact Wigner-3j symbols.

Reference (2) requires `sympy` and is skipped if unavailable.
"""

import numpy as np
import pytest

from cmbcov.kernels.coupling import (
    KERNEL_TE,
    KERNEL_TT,
    KERNEL_EEmBB,
    KERNEL_EEpBB,
    coupling_kernels,
    mask_correlation_function,
    wigner_d_table,
)

CHANNELS = [
    ("TT", KERNEL_TT),
    ("EE+BB", KERNEL_EEpBB),
    ("EE-BB", KERNEL_EEmBB),
    ("TE", KERNEL_TE),
]


def _full_sky_wl(lmax_mask: int = 16) -> np.ndarray:
    """Mask identically 1  =>  W_0 = 4*pi, all other W_L = 0."""
    wl = np.zeros(lmax_mask + 1)
    wl[0] = 4.0 * np.pi
    return wl


# --------------------------------------------------------------------------
# Wigner d-functions
# --------------------------------------------------------------------------


def test_wigner_d00_matches_legendre():
    """d^l_00 is the Legendre polynomial P_l."""
    mu = np.linspace(-1.0, 1.0, 41)
    lmax = 20
    d = wigner_d_table(lmax, mu, 0, 0)
    for ell in range(lmax + 1):
        coeffs = np.zeros(ell + 1)
        coeffs[ell] = 1.0
        expected = np.polynomial.legendre.legval(mu, coeffs)
        np.testing.assert_allclose(d[:, ell], expected, atol=1e-12)


@pytest.mark.parametrize("m,mp", [(0, 0), (2, 2), (2, -2), (2, 0)])
def test_wigner_d_orthogonality(m, mp):
    """int_-1^1 d^l_mm' d^l'_mm' dmu = 2/(2l+1) delta_ll'."""
    lmax = 24
    mu, w = np.polynomial.legendre.leggauss(2 * lmax + 8)
    d = wigner_d_table(lmax, mu, m, mp)
    gram = (d * w[:, None]).T @ d
    l0 = max(abs(m), abs(mp))
    expected = np.diag(2.0 / (2.0 * np.arange(lmax + 1) + 1.0))
    np.testing.assert_allclose(gram[l0:, l0:], expected[l0:, l0:], atol=1e-11)


def test_wigner_d_bounded_by_one():
    """|d^l_mm'(mu)| <= 1 for all real rotations."""
    mu = np.linspace(-1.0, 1.0, 101)
    for m, mp in [(0, 0), (2, 2), (2, -2), (2, 0)]:
        d = wigner_d_table(60, mu, m, mp)
        assert np.abs(d).max() <= 1.0 + 1e-10


# --------------------------------------------------------------------------
# Mask correlation function
# --------------------------------------------------------------------------


def test_mask_correlation_full_sky_is_unity():
    """A mask equal to 1 everywhere has w(theta) == 1 for all theta."""
    mu = np.linspace(-1.0, 1.0, 51)
    w = mask_correlation_function(_full_sky_wl(), mu)
    np.testing.assert_allclose(w, 1.0, atol=1e-12)


# --------------------------------------------------------------------------
# Coupling kernels
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name,idx", CHANNELS)
def test_full_sky_kernel_is_identity(name, idx):
    """Camphuis et al. (2022) Eq. (5): unmasked sky must not couple modes."""
    lmax = 12
    kernels = coupling_kernels(_full_sky_wl(), lmax)
    l0 = 0 if idx == KERNEL_TT else 2
    block = kernels[idx][l0:, l0:]
    np.testing.assert_allclose(block, np.eye(block.shape[0]), atol=1e-10)


def test_kernel_row_sum_equals_mask_correlation_at_zero():
    """sum_l' M_ll' = w(theta=0) = <W^2>, by Legendre completeness."""
    lmax, lmax_mask = 40, 20
    wl = np.exp(-np.arange(lmax_mask + 1) / 5.0)
    wl[0] = 4.0 * np.pi * 0.04
    kernels = coupling_kernels(wl, lmax)
    w0 = mask_correlation_function(wl, np.array([1.0]))[0]
    # Interior rows only: rows near lmax lose support truncated by lmax.
    np.testing.assert_allclose(
        kernels[KERNEL_TT][: lmax - lmax_mask].sum(axis=1), w0, rtol=1e-9
    )


def test_kernel_is_independent_of_quadrature_resolution():
    """Gauss-Legendre is exact, so extra nodes must not change the answer."""
    lmax, lmax_mask = 24, 12
    wl = np.exp(-np.arange(lmax_mask + 1) / 4.0)
    wl[0] = 4.0 * np.pi * 0.1
    a = coupling_kernels(wl, lmax)
    b = coupling_kernels(wl, lmax, n_theta=4 * lmax + 64)
    # Tolerance must be relative to the matrix scale: most entries are
    # legitimately zero to machine precision.
    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-11 * np.abs(b).max())


def _brute_force_kernels(wl: np.ndarray, lmax: int) -> np.ndarray:
    """Direct O(lmax^3) evaluation of Eq. (3) with exact Wigner-3j symbols."""
    from sympy.physics.wigner import wigner_3j

    lmax_mask = wl.size - 1
    fl = (2 * np.arange(lmax_mask + 1) + 1) * wl / (4.0 * np.pi)
    out = np.zeros((4, lmax + 1, lmax + 1))
    for l1 in range(lmax + 1):
        for l2 in range(lmax + 1):
            t00 = t22e = t22o = t02 = 0.0
            for big_l in range(abs(l1 - l2), min(l1 + l2, lmax_mask) + 1):
                w000 = float(wigner_3j(l1, l2, big_l, 0, 0, 0))
                w220 = float(wigner_3j(l1, l2, big_l, 2, -2, 0))
                if (l1 + l2 + big_l) % 2 == 0:
                    t00 += w000 * w000 * fl[big_l]
                    t22e += w220 * w220 * fl[big_l]
                    t02 += w000 * w220 * fl[big_l]
                else:
                    t22o += w220 * w220 * fl[big_l]
            norm = 2 * l2 + 1
            out[KERNEL_TT, l1, l2] = t00 * norm
            out[KERNEL_EEpBB, l1, l2] = (t22e + t22o) * norm
            out[KERNEL_EEmBB, l1, l2] = (t22e - t22o) * norm
            out[KERNEL_TE, l1, l2] = t02 * norm
    return out


@pytest.mark.parametrize("name,idx", CHANNELS)
def test_matches_exact_wigner3j_sum(name, idx):
    """The quadrature form must reproduce the defining Wigner-3j sum exactly."""
    pytest.importorskip("sympy")
    lmax, lmax_mask = 10, 8
    rng = np.random.default_rng(0)
    wl = np.exp(-np.arange(lmax_mask + 1) / 3.0) * (1 + 0.3 * rng.random(lmax_mask + 1))
    wl[0] = 4.0 * np.pi * 0.04  # SPT-3G-like fsky
    fast = coupling_kernels(wl, lmax)[idx]
    ref = _brute_force_kernels(wl, lmax)[idx]
    np.testing.assert_allclose(fast, ref, rtol=1e-10, atol=1e-14 * np.abs(ref).max())
