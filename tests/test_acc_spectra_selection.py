r"""
The ``spectra`` selection on the ACC coupling precompute, its manifest, and
the matching restriction of the loader.

The ACC precompute (``precompute_acc_kernels``) can restrict its output to a
subset of the ``5 x 5 = 25`` ordered pairs of :data:`COUPLING_SPECTRA`, and
``ACCStrategy.get_covariance_coupling`` matches that restriction rather than
always loading the sixteen ``{TT,EE,TE,ET}^2`` pairs, so a caller that only
needs ``TTxTT`` (a TT-only study, e.g. comparing ACC against the exact
covariance) is not forced to pay for the rest. On the GL grid computing the
spin-2 (E, B) integrals for every ``m`` is ~92% of the precompute wall time
at production sizes, so a caller that only needs T should never pay for it.

This file pins, at the small size used throughout ``tests/test_acc_gl*.py``
(``nside=16``, ``centralell=16``, ``lmax=48``, ``lw=10``):

- a ``spectra=("TT",)`` precompute produces exactly ``{"TTxTT"}`` and its
  values are bit-identical to the ``TTxTT`` entry of a full (``spectra=None``)
  precompute;
- the default (``spectra=None``) precompute is unchanged: still all 25 pairs;
- the manifest written alongside the kernels round-trips (``spectra``,
  ``grid``, ``lw``, ``nside``, ``centralell``);
- ``get_covariance_coupling`` asked for a pair absent from a restricted cache
  raises ``OSError`` naming both the missing pair and the manifest, rather
  than substituting anything;
- ``compute_covariance_term`` for a TT-only ``CovKey`` gives the same answer
  whether it is pointed at a TT-only cache or a full one.
"""

import os
import tempfile
import warnings

import numpy as np
import pytest

from cmbcov.approximations import StrategyFactory
from cmbcov.approximations.acc import (
    COUPLING_SPECTRA,
    precompute_acc_kernels,
)
from cmbcov.covariance import Cov, CovarianceConfig, CovarianceMethod
from cmbcov.keys import CovKey

DATA = os.path.join(os.path.dirname(__file__), "data")
NSIDE = 16
ELL = 16
LMAX = 48
DMAX = 2
LW = 10

#: ACC reads the spectra to lmax_int = lmax + max(0, S - 1 - centralell),
#: S = 2 NSIDE the kernel size (acc_window_pad).
LMAX_INT = LMAX + 2 * NSIDE - 1 - ELL
ELLP_RANGE = [ELL, ELL + 1]
FREQ = "090GHz"


def _cov(save_dir, dmax=DMAX):
    config = CovarianceConfig(
        method=CovarianceMethod.ACC, lmax=LMAX, dmax=dmax, centralell=ELL
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # low centralell warning
        return Cov(
            "baseline_mask.fits",
            config=config,
            mask_path=os.path.abspath(DATA),
            save_dir=save_dir,
        )


def _tt_cl(size):
    ell = np.arange(size)
    tt = np.zeros(size)
    tt[2:] = 1e-3 / (ell[2:] * (ell[2:] + 1)) ** 0.9
    return tt


# --------------------------------------------------------------------------- #
# (A) the precompute: spectra=("TT",) vs spectra=None (dryrun)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def full_dryrun():
    cov = _cov(tempfile.mkdtemp())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return precompute_acc_kernels(
            cov.wlm,
            None,
            centralell=ELL,
            dmax=DMAX,  # ELLP_RANGE
            nside=NSIDE,
            dryrun=True,
            grid="gl",
            lw=LW,
        )


@pytest.fixture(scope="module")
def tt_only_dryrun():
    cov = _cov(tempfile.mkdtemp())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return precompute_acc_kernels(
            cov.wlm,
            None,
            centralell=ELL,
            dmax=DMAX,  # ELLP_RANGE
            nside=NSIDE,
            dryrun=True,
            grid="gl",
            lw=LW,
            spectra=("TT",),
        )


def test_full_precompute_still_produces_all_twenty_five(full_dryrun):
    for ellp, inner in full_dryrun.items():
        assert len(inner) == 25, f"ellp={ellp} produced {len(inner)} pairs"
        assert set(inner) == {
            f"{a}x{b}" for a in COUPLING_SPECTRA for b in COUPLING_SPECTRA
        }


def test_tt_only_precompute_produces_a_single_kernel(tt_only_dryrun):
    for inner in tt_only_dryrun.values():
        assert set(inner) == {"TTxTT"}


def test_tt_only_kernel_is_bit_identical_to_the_full_precompute(
    full_dryrun, tt_only_dryrun
):
    """
    Both runs cover the whole (m, m') range in a single contraction block at
    this problem size, so the TT-only path -- a different code path (a
    1-field coefficient stack instead of 3) -- must reproduce the TT kernel
    of the full 25-kernel run to round-off, not just approximately.
    """
    for ellp in ELLP_RANGE:
        full = full_dryrun[ellp]["TTxTT"]
        tt_only = tt_only_dryrun[ellp]["TTxTT"]
        np.testing.assert_allclose(
            tt_only, full, rtol=0, atol=1e-15 * np.abs(full).max(), err_msg=str(ellp)
        )


def test_tt_only_also_works_on_the_healpix_grid():
    """``spectra`` is grid-agnostic: healpix does not skip the spin-2
    synthesis (T, E, B come from one joint transform), but it must still
    restrict the kernels computed and returned."""
    cov = _cov(tempfile.mkdtemp())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = precompute_acc_kernels(
            cov.wlm,
            None,
            centralell=ELL,
            dmax=1,
            nside=NSIDE,
            dryrun=True,
            grid="healpix",
            spectra=("TT",),
        )
    assert set(result[ELL]) == {"TTxTT"}


@pytest.mark.parametrize("spectra", [(), ("XX",), ("TT", "XX")])
def test_invalid_spectra_selection_is_rejected(spectra):
    cov = _cov(tempfile.mkdtemp())
    with pytest.raises(ValueError):
        precompute_acc_kernels(
            cov.wlm,
            None,
            centralell=ELL,
            dmax=1,
            nside=NSIDE,
            dryrun=True,
            grid="gl",
            lw=LW,
            spectra=spectra,
        )


# --------------------------------------------------------------------------- #
# (B) the manifest
# --------------------------------------------------------------------------- #
def test_manifest_round_trips_for_a_full_precompute():
    cov = _cov(tempfile.mkdtemp())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        precompute_acc_kernels(
            cov.wlm,
            cov.save_dir,
            centralell=ELL,
            dmax=DMAX,  # ELLP_RANGE
            nside=NSIDE,
            grid="gl",
            lw=LW,
        )
    strategy = StrategyFactory.create_strategy(cov)

    manifest = strategy.get_covariance_coupling_manifest(ELL, ELL)
    assert manifest is not None
    assert sorted(manifest["spectra"]) == sorted(COUPLING_SPECTRA)
    assert manifest["grid"] == "gl"
    assert manifest["lw"] == LW
    assert manifest["nside"] == NSIDE
    assert manifest["ell"] == ELL
    assert manifest["ellp"] == ELL
    assert manifest["centralell"] == ELL
    assert manifest["git_hash"] is None or isinstance(manifest["git_hash"], str)

    # A pair never computed has no manifest.
    assert strategy.get_covariance_coupling_manifest(ELL, ELL + 50) is None


def test_manifest_round_trips_for_a_tt_only_precompute():
    cov = _cov(tempfile.mkdtemp())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        precompute_acc_kernels(
            cov.wlm,
            cov.save_dir,
            centralell=ELL,
            dmax=DMAX,  # ELLP_RANGE
            nside=NSIDE,
            grid="gl",
            lw=LW,
            spectra=("TT",),
        )
    strategy = StrategyFactory.create_strategy(cov)

    manifest = strategy.get_covariance_coupling_manifest(ELL, ELL + 1)
    assert manifest["spectra"] == ["TT"]
    assert manifest["ell"] == ELL
    assert manifest["ellp"] == ELL + 1


# --------------------------------------------------------------------------- #
# (C) the loader: a missing pair, with and without an explanatory manifest
# --------------------------------------------------------------------------- #
def test_loader_names_the_missing_pair_and_the_manifest_for_a_restricted_cache():
    cov = _cov(tempfile.mkdtemp())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        precompute_acc_kernels(
            cov.wlm,
            cov.save_dir,
            centralell=ELL,
            dmax=1,
            nside=NSIDE,
            grid="gl",
            lw=LW,
            spectra=("TT",),
        )
    strategy = StrategyFactory.create_strategy(cov)

    # TT alone loads fine.
    assert set(strategy.get_covariance_coupling(ELL, ELL, pairs=[("TT", "TT")])) == {
        ("TT", "TT")
    }

    with pytest.raises(OSError) as excinfo:
        strategy.get_covariance_coupling(ELL, ELL, pairs=[("TT", "DD")])
    message = str(excinfo.value)
    assert "TTxDD_16x16.npy" in message  # names the missing pair
    assert strategy.get_covariance_coupling_manifest_path(ELL, ELL) in message
    assert "['TT']" in message  # what the cache was actually built for


def test_loader_behaves_as_before_when_there_is_no_manifest():
    """
    A cache with no manifest at all (files placed there by hand, or written
    before this feature) must fall back to the plain "not found" error, not
    claim anything about spectra it cannot know.
    """
    cov = _cov(tempfile.mkdtemp())
    strategy = StrategyFactory.create_strategy(cov)
    with pytest.raises(OSError) as excinfo:
        strategy.get_covariance_coupling(ELL, ELL, pairs=[("TT", "TT")])
    message = str(excinfo.value)
    assert "TTxTT_16x16.npy" in message
    assert "spectra" not in message


# --------------------------------------------------------------------------- #
# (D) end to end: compute_covariance_term against a TT-only vs a full cache
# --------------------------------------------------------------------------- #
def test_compute_covariance_term_matches_between_tt_only_and_full_cache():
    cov_full = _cov(tempfile.mkdtemp())
    cov_tt = _cov(tempfile.mkdtemp())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        precompute_acc_kernels(
            cov_full.wlm,
            cov_full.save_dir,
            centralell=ELL,
            dmax=DMAX,
            nside=NSIDE,
            grid="gl",
            lw=LW,
        )
        precompute_acc_kernels(
            cov_tt.wlm,
            cov_tt.save_dir,
            centralell=ELL,
            dmax=DMAX,
            nside=NSIDE,
            grid="gl",
            lw=LW,
            spectra=("TT",),
        )
        strategy_full = StrategyFactory.create_strategy(cov_full)
        strategy_tt = StrategyFactory.create_strategy(cov_tt)

        key = CovKey(("T", "T", "T", "T"), (FREQ,) * 4)
        cl = {FREQ + FREQ: {"TT": _tt_cl(LMAX_INT)}}

        sigma_full = strategy_full.compute_covariance_term(key, cl)
        sigma_tt = strategy_tt.compute_covariance_term(key, cl)

    assert np.isfinite(sigma_full).all() and np.abs(sigma_full).max() > 0
    np.testing.assert_allclose(
        sigma_tt, sigma_full, rtol=0, atol=1e-15 * np.abs(sigma_full).max()
    )
