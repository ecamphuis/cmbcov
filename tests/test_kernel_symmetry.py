"""
Regression tests for the recovery of the symmetric operator Xi from M.

The MASTER kernel carries the (2l'+1) factor on its SECOND index,

    M_ll' = (2l' + 1) Xi_ll' ,

so recovering Xi means dividing along axis 1. The original implementation used
``(K.T / f).T``, which divides by the FIRST index and therefore leaves
``(2l2+1)/(2l1+1) * Xi``. That is not symmetric, and it propagated all the way
into an asymmetric, non-positive-definite covariance matrix -- max|C - C.T| was
1.5% of the matrix scale and the smallest eigenvalue was large and negative.

Symmetry of Xi is a mathematical identity, so a violation is now an error
rather than a warning.
"""

import numpy as np
import pytest

from cmbcov.mask import MaskWlm


def _reference_xi(n=6, seed=0):
    """A symmetric Xi and the corresponding M = (2l'+1) Xi."""
    rng = np.random.default_rng(seed)
    xi = rng.random((n, n))
    xi = xi + xi.T
    ell_factors = 2.0 * np.arange(n) + 1.0
    kernel = xi * ell_factors[None, :]
    return xi, kernel


def test_symmetrisation_recovers_xi_exactly():
    xi, kernel = _reference_xi()
    stacked = np.stack([kernel] * 4)
    recovered = MaskWlm._make_kernel_symmetric(None, stacked)
    for channel in range(4):
        np.testing.assert_allclose(recovered[channel], xi, rtol=1e-13, atol=0.0)


def test_symmetrisation_output_is_symmetric():
    _, kernel = _reference_xi()
    recovered = MaskWlm._make_kernel_symmetric(None, np.stack([kernel] * 4))
    for channel in range(4):
        block = recovered[channel]
        asymmetry = np.abs(block - block.T).max() / np.abs(block).max()
        assert asymmetry < 1e-12, f"channel {channel} asymmetry {asymmetry:.3e}"


def test_dividing_by_the_wrong_index_is_detectably_asymmetric():
    """
    Documents the original defect: the two idioms differ, and the wrong one is
    grossly asymmetric rather than subtly so.
    """
    xi, kernel = _reference_xi()
    ell_factors = 2.0 * np.arange(kernel.shape[1]) + 1.0
    wrong = (kernel.T / ell_factors).T  # divides by the FIRST index
    right = kernel / ell_factors[None, :]  # divides by the SECOND index
    asym = lambda a: np.abs(a - a.T).max() / np.abs(a).max()  # noqa: E731
    assert asym(wrong) > 0.1
    assert asym(right) < 1e-12
    np.testing.assert_allclose(right, xi, rtol=1e-13, atol=0.0)


def test_asymmetric_kernel_raises_rather_than_warns():
    """
    An asymmetric Xi yields a non-positive-definite covariance, so it must stop
    the computation rather than emit a warning that can be filtered away.
    """
    rng = np.random.default_rng(1)
    n = 6
    asymmetric = rng.random((n, n))  # not of the form (2l'+1) Xi
    with pytest.raises(ValueError, match="not symmetric"):
        MaskWlm._make_kernel_symmetric(None, np.stack([asymmetric] * 4))
