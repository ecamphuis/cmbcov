"""
The ACC *assembly*: sliding the central kernel along the diagonals (Eq. 33).

``tests/test_acc_coupling.py`` and ``tests/test_acc_vs_exact.py`` pin the
precomputed kernel Theta-bar; nothing pinned what ``ACCStrategy`` then does
with it.  Two checks:

- ``compute_acc_term`` equals the standard window formula with a
  zero-padded spectrum, on every diagonal and at both edges of the
  multipole range, the spectra reaching ``lmax_int = lmax + max(0, S - 1 -
  l*)``.  An earlier implementation re-anchored the window on ell_max near
  the upper edge and was off by Delta - 1 there.
- ``compute_covariance_term`` for TTxTT equals an explicit-loop evaluation
  of Eq. 25 with the Eq. 33 translation, written here without any code
  shared with acc.py.
"""

import os
import tempfile
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from cmbcov.approximations import StrategyFactory
from cmbcov.approximations.acc import (
    ACCStrategy,
    precompute_acc_kernels,
)
from cmbcov.covariance import Cov, CovarianceConfig, CovarianceMethod
from cmbcov.keys import CovKey

DATA = os.path.join(os.path.dirname(__file__), "data")
NSIDE = 16
ELL = 16  # centralell; the ACC kernel is (2 NSIDE)^2, so L runs 0..31
LMAX = 48
DMAX = 4
#: ACC reads the spectra to lmax + max(0, S - 1 - ELL), S = 2 NSIDE.
LMAX_INT = LMAX + 2 * NSIDE - 1 - ELL


def test_window_matches_zero_padded_reference():
    rng = np.random.default_rng(0)
    lmax, central, size = 120, 50, 80
    lmax_int = lmax + size - 1 - central  # 149, see acc_window_pad
    kernel = rng.random((size, size))
    kernel /= kernel.sum()
    cl1, cl2 = rng.random(lmax_int), rng.random(lmax_int)
    fake_self = SimpleNamespace(cov=SimpleNamespace(lmax=lmax))
    coupling = {("TT", "TT"): kernel}

    padded1 = np.zeros(lmax_int + 2 * size)
    padded2 = np.zeros(lmax_int + 2 * size)
    padded1[size : size + lmax_int] = cl1
    padded2[size : size + lmax_int] = cl2

    for delta in range(6):
        for ell1 in range(lmax - delta):
            ell2 = ell1 + delta
            got = ACCStrategy.compute_acc_term(
                fake_self,
                {"TT": cl1},
                {"TT": cl2},
                ("TT", "TT"),
                central,
                ell1,
                ell2,
                covariance_coupling=coupling,
            )
            start = size + ell1 - central  # window start in padded coordinates
            ref = padded1[start : start + size] @ kernel @ padded2[start : start + size]
            assert got == pytest.approx(ref, rel=1e-13, abs=0.0), (delta, ell1)
            # symmetric in (ell1, ell2)
            swapped = ACCStrategy.compute_acc_term(
                fake_self,
                {"TT": cl1},
                {"TT": cl2},
                ("TT", "TT"),
                central,
                ell2,
                ell1,
                covariance_coupling=coupling,
            )
            assert swapped == got


@pytest.fixture(scope="module")
def acc_setup():
    config = CovarianceConfig(
        method=CovarianceMethod.ACC, lmax=LMAX, dmax=DMAX, centralell=ELL
    )
    cov = Cov(
        "baseline_mask.fits",
        config=config,
        mask_path=os.path.abspath(DATA),
        save_dir=tempfile.mkdtemp(),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # low centralell warning
        precompute_acc_kernels(
            cov.wlm,
            cov.save_dir,
            centralell=ELL,
            dmax=DMAX,
            nside=NSIDE,
            grid="healpix",
        )
        strategy = StrategyFactory.create_strategy(cov)
    ell = np.arange(LMAX_INT)
    cl = np.zeros(LMAX_INT)
    cl[2:] = 1e-3 / (ell[2:] * (ell[2:] + 1)) ** 0.9
    return cov, strategy, cl


def test_assembled_tt_covariance_matches_explicit_eq25(acc_setup):
    cov, strategy, cl = acc_setup
    key = CovKey(("T", "T", "T", "T"), ("090GHz",) * 4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sigma = strategy.compute_covariance_term(key, {"090GHz090GHz": {"TT": cl}})

    xi = cov.norm_Xi["TT", "TT"]
    expected = np.zeros((LMAX, LMAX))
    for delta in range(DMAX):
        theta = strategy.get_covariance_coupling(ELL, ELL + delta)[("TT", "TT")]
        theta = theta / theta.sum()  # Eq. 23
        size = theta.shape[0]
        for ell1 in range(LMAX - delta):
            ell2 = ell1 + delta
            # Eq. 25 with Eq. 33: Theta-bar_{ell1 ell2}(L1, L2)
            #   = Theta-bar_{ELL, ELL+delta}(L1 - ell1 + ELL, L2 - ell1 + ELL)
            total = 0.0
            for i in range(size):
                L1 = i + ell1 - ELL
                if not 0 <= L1 < LMAX_INT:
                    continue
                for j in range(size):
                    L2 = j + ell1 - ELL
                    if not 0 <= L2 < LMAX_INT:
                        continue
                    total += cl[L1] * theta[i, j] * cl[L2]
            value = 2.0 * xi[ell1, ell2] * total  # two identical Wick terms
            expected[ell1, ell2] = value
            expected[ell2, ell1] = value

    scale = np.abs(expected).max()
    np.testing.assert_allclose(sigma, expected, rtol=1e-12, atol=1e-14 * scale)
    # every stored diagonal is populated, nothing beyond dmax is
    assert np.count_nonzero(sigma) == np.count_nonzero(expected) > 0
    assert np.all(sigma[np.triu_indices(LMAX, DMAX)] == 0)


def _diag_sum_unflatten(lmax, flattened_cov):
    """Reference ``np.diag``-sum form that ``_unflatten_cov`` is compared against."""
    dmax = (flattened_cov.shape[0] - 1) // 2 + 1
    output = np.zeros((lmax, lmax))
    for d in range(dmax):
        output += np.diag(flattened_cov[dmax - 1 + d, : lmax - d], d)
        if d > 0:
            output += np.diag(flattened_cov[dmax - 1 - d, : lmax - d], -d)
    return output


@pytest.mark.parametrize("lmax, dmax", [(1, 1), (5, 1), (48, 4), (300, 20), (20, 21)])
def test_unflatten_cov_is_bit_identical_to_the_diag_sum(lmax, dmax):
    """
    ``_unflatten_cov`` fills the bands in place instead of summing
    ``2 dmax - 1`` dense ``np.diag`` matrices; every entry is still
    ``0.0 + x``, so the bytes (signed zeros and NaN included) must not move.
    ``dmax = lmax + 1`` is the largest offset count the old form accepted.
    """
    rng = np.random.default_rng(lmax * 100 + dmax)
    flat = rng.standard_normal((2 * dmax - 1, lmax))
    flat[0, : min(3, lmax)] = -0.0
    flat[-1, -1] = np.nan
    fake_self = SimpleNamespace(cov=SimpleNamespace(lmax=lmax))
    got = ACCStrategy._unflatten_cov(fake_self, flat)
    expected = _diag_sum_unflatten(lmax, flat)
    assert got.shape == (lmax, lmax)
    assert got.tobytes() == expected.tobytes()
