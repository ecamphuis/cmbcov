"""
The normalisation-consistency guard of the ACC error budget
(``acc_budget.normalisation_consistency``).

Eq. 22 for the TT x TT kernel of a pair, ``sum Theta = (2l+1)(2l'+1)
Xi^00[W^2]``, is exact when the kernel and Xi come from the same band-limited
mask. ``Cov.Xi`` uses the mask at its own band limit and the kernels use
``acc_precompute.lw``; the guard measures how far the identity is from holding
and warns when it is off by more than 1e-2.

* the identity holds to solver round-off when both see the same mask;
* a deliberately low ``lw`` on a mask with sharp edges makes it fail badly, and
  the WARNING fires;
* the standard test configuration does not warn;
* the check never raises, and says so when there is no TT x TT kernel.
"""

import logging
import os
import warnings

import healpy as hp
import numpy as np
import pytest

from cmbcov.approximations import StrategyFactory, acc_budget
from cmbcov.approximations.acc import precompute_acc_kernels
from cmbcov.covariance import Cov, CovarianceConfig, CovarianceMethod
from cmbcov.kernels.coupling import KERNEL_TT
from cmbcov.keys import CovKeys

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
MASK = "baseline_mask.fits"  # nside 32: Cov.Xi is built out to l = 64
ELL = 16
DMAX = 3
LMAX = 60
LOGGER = "cmbcov.approximations.acc_budget"


def _strategy(kernel_dir, mask=MASK, mask_path=DATA, lmax=LMAX, dmax=DMAX):
    config = CovarianceConfig(
        method=CovarianceMethod.ACC,
        lmax=lmax,
        lmin=2,
        dmax=dmax,
        centralell=ELL,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # low centralell
        cov = Cov(mask, config=config, mask_path=mask_path, save_dir=str(kernel_dir))
        return StrategyFactory.create_strategy(cov)


def _precompute(kernel_dir, mask=MASK, mask_path=DATA, dmax=DMAX, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        precompute_acc_kernels(
            mask,
            str(kernel_dir),
            centralell=ELL,
            dmax=dmax,
            mask_path=mask_path,
            **kwargs,
        )


@pytest.fixture(scope="module")
def consistent(tmp_path_factory):
    """GL kernels at lw = 2 nside of the mask, summed out to the full support."""
    directory = tmp_path_factory.mktemp("xi_consistent")
    # lw = 64 is the mask's own band limit (Cov.Xi); ell + lw + dmax < 2 * 48
    # so the (L1, L2) sum is not truncated either.
    _precompute(directory, nside=48, grid="gl", lw=64)
    return _strategy(directory)


@pytest.fixture(scope="module")
def sharp_mask(tmp_path_factory):
    """A hard-edged cap: power far beyond any low lw."""
    directory = tmp_path_factory.mktemp("xi_sharp_mask")
    nside = 32
    mask = np.zeros(hp.nside2npix(nside))
    mask[hp.query_disc(nside, hp.ang2vec(np.pi / 3, 1.0), np.radians(50))] = 1.0
    hp.write_map(str(directory / "cap.fits"), mask, overwrite=True)
    return str(directory)


def test_identity_holds_to_roundoff_when_both_see_the_same_mask(consistent):
    """The raw sum rule, without the guard's own code in the comparison."""
    xi00 = np.asarray(consistent.cov.Xi)[KERNEL_TT]
    for d in range(DMAX):
        ell_prime = ELL + d
        theta = consistent.get_covariance_coupling(ELL, ell_prime, pairs=[("TT", "TT")])
        n = (2 * ELL + 1) * (2 * ell_prime + 1)
        ratio = theta[("TT", "TT")].sum() / n / xi00[ELL, ell_prime]
        assert abs(ratio - 1.0) < 1e-9


def test_the_guard_reports_that_roundoff_mismatch(consistent, caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER):
        result = acc_budget.normalisation_consistency(consistent)
    assert result["max_relative_mismatch"] < 1e-9
    assert result["exceeds"] is False
    assert result["diagonals_checked"] == DMAX
    assert result["lw"] == 64 and result["grid"] == "gl"
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("agree to" in r.getMessage() for r in caplog.records)


def test_low_lw_on_a_sharp_mask_is_flagged(tmp_path, sharp_mask, caplog):
    _precompute(
        tmp_path,
        mask="cap.fits",
        mask_path=sharp_mask,
        nside=48,
        grid="gl",
        lw=8,
        spectra=("TT", "DD", "TD", "DT"),
    )
    strategy = _strategy(tmp_path, mask="cap.fits", mask_path=sharp_mask)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        result = acc_budget.normalisation_consistency(strategy)
    assert result["max_relative_mismatch"] > 0.05
    assert result["mismatch_first"] > 0.05  # a missing-lw problem shows at d = 0
    assert result["exceeds"] is True
    assert result["lw"] == 8
    warnings_logged = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings_logged) == 1
    text = warnings_logged[0].getMessage()
    assert "beyond the kernels' lw" in text and "raise acc_precompute.lw" in text

    # and the report beside the covariance carries the one clear line
    budget = strategy.error_budget(CovKeys(["T", "E"], ["090GHz"]))
    report = acc_budget.format_error_budget(budget)
    assert "max relative mismatch" in report
    assert "EXCEEDED" in report and "lw = 8" in report


def test_raising_lw_on_the_same_mask_clears_it(tmp_path, sharp_mask):
    _precompute(
        tmp_path, mask="cap.fits", mask_path=sharp_mask, nside=48, grid="gl", lw=64
    )
    strategy = _strategy(tmp_path, mask="cap.fits", mask_path=sharp_mask)
    result = acc_budget.normalisation_consistency(strategy)
    assert result["max_relative_mismatch"] < 1e-9


@pytest.mark.parametrize(
    "options",
    [
        {"nside": 16, "grid": "healpix"},  # tests/test_acc_error_budget.py
        {"nside": 16, "grid": "gl"},  # default lw = 3 nside - 1, L1, L2 < 32
    ],
    ids=["healpix", "gl"],
)
def test_no_warning_on_the_standard_test_configuration(tmp_path, options, caplog):
    _precompute(tmp_path, **options)
    strategy = _strategy(tmp_path, lmax=48, dmax=2)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        budget = strategy.error_budget(CovKeys(["T", "E"], ["090GHz"]))
    norm = budget["normalisation"]
    assert 0.0 < norm["max_relative_mismatch"] < acc_budget.XI_MISMATCH_WARNING / 5
    assert norm["exceeds"] is False
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert "warning threshold 1e-02: ok" in acc_budget.format_error_budget(budget)


def test_no_tt_kernel_is_skipped_with_a_note(tmp_path, caplog):
    _precompute(tmp_path, nside=16, grid="healpix", spectra=("EE",))
    strategy = _strategy(tmp_path, lmax=48, dmax=2)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        result = acc_budget.normalisation_consistency(strategy)
    assert result["max_relative_mismatch"] is None
    assert "no TT x TT coupling kernel" in result["unavailable"]
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    text = acc_budget._format_normalisation(result)
    assert any("NOT AVAILABLE" in line for line in text)


def test_the_check_never_raises(consistent, monkeypatch, caplog):
    def broken(*args, **kwargs):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(consistent, "get_covariance_coupling", broken)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        result = acc_budget.normalisation_consistency(consistent)
    assert result["max_relative_mismatch"] is None
    assert "disk on fire" in result["unavailable"]
    assert any(r.levelno == logging.WARNING for r in caplog.records)

    # a strategy that is not one at all
    result = acc_budget.normalisation_consistency(object())
    assert result["max_relative_mismatch"] is None
    assert "unavailable" in result
