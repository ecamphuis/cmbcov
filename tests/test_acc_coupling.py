"""
Regression tests for the ACC covariance-coupling precomputation.

This is the path implementing the central approximation of Camphuis et al.
(2022): the reduced coupling kernel Theta-bar is computed once at a reference
multipole and reused along each diagonal (Eq. 33).

It had no test coverage at all, which matters because it is the part of the
package about to be restructured: the alm containers it depends on exist only
to serve this code. The golden file here is what makes that refactor provable.
"""

import os

import numpy as np
import pytest

healpy = pytest.importorskip("healpy")

from cmbcov.approximations.acc import (  # noqa: E402
    CHANNEL_ALIASES,
    COUPLING_SPECTRA,
    precompute_acc_kernels,
)

DATA = os.path.join(os.path.dirname(__file__), "data")
ELL = 16
DMAX = 2
ELLP_RANGE = [16, 17]  # = coupling_ellprange(ELL, DMAX)
NSIDE = 16

#: The golden ``tests/data/baseline_acc_coupling.npz`` was written under the
#: old channel names (pre-B-mode); the precompute now returns the new ones
#: (:data:`COUPLING_SPECTRA`).  This is the fixed legacy label set the golden
#: keys use -- unrelated to the package's channel constants, which have
#: since been renamed -- with the equivalent new label for every comparison.
LEGACY_SPECTRA = ("TT", "EE", "BB", "TE", "ET")


@pytest.fixture(scope="module")
def coupling():
    return precompute_acc_kernels(
        "baseline_mask.fits",
        None,
        centralell=ELL,
        dmax=DMAX,
        mask_path=os.path.abspath(DATA),
        nside=NSIDE,
        grid="healpix",
        dryrun=True,
    )


def test_returns_a_kernel_for_every_requested_ellp(coupling):
    assert sorted(coupling) == sorted(ELLP_RANGE)


def test_all_twenty_five_stokes_pairs_are_produced(coupling):
    """Five spectra (TT, EE, BB, TE, ET) give twenty-five ordered pairs."""
    for ellp, inner in coupling.items():
        assert len(inner) == 25, f"ellp={ellp} produced {len(inner)} pairs"
        assert set(inner) == {
            f"{a}x{b}" for a in COUPLING_SPECTRA for b in COUPLING_SPECTRA
        }


def test_matches_recorded_baseline(coupling):
    """
    The coupling kernels must not change under refactoring.

    The golden records all twenty-five ordered pairs of
    ``COUPLING_SPECTRA``.

    Regenerating it when the HEALPix ``map2alm`` default went from
    ``iter=0`` to healpy's ``iter=3`` moved it by: global L2-relative change
    2.25%; entries above 1% of each kernel's maximum median 3.6%, max 13%;
    diagonal median 3.2%. Most of it is the ~2.5% Eq. 22 amplitude deficit of
    ``iter=0`` going away (sum ratio 0.974 -> 0.9999); after Eq. 23
    normalisation the kernels agree with the exact GL ones to 8e-4 (was 8e-3).
    """
    expected = np.load(os.path.join(DATA, "baseline_acc_coupling.npz"))
    recorded = sorted(expected.files)
    assert recorded == sorted(
        f"{ellp}|{a}x{b}"
        for ellp in ELLP_RANGE
        for a in LEGACY_SPECTRA
        for b in LEGACY_SPECTRA
    )
    for name in recorded:
        ellp, legacy_key = name.split("|")
        legacy_a, legacy_b = legacy_key.split("x")
        stokes_key = f"{CHANNEL_ALIASES[legacy_a]}x{CHANNEL_ALIASES[legacy_b]}"
        reference = expected[name]
        array = coupling[int(ellp)][stokes_key]
        assert array.shape == reference.shape
        np.testing.assert_allclose(
            array,
            reference,
            rtol=1e-10,
            atol=1e-14 * np.abs(reference).max(),
            err_msg=f"ellp={ellp} {legacy_key} ({stokes_key})",
        )


def test_kernels_are_finite_and_non_trivial(coupling):
    for ellp, inner in coupling.items():
        for stokes_key, array in inner.items():
            assert np.isfinite(array).all(), f"{ellp} {stokes_key} not finite"
            assert np.abs(array).max() > 0, f"{ellp} {stokes_key} is all zero"


def test_bb_couplings_are_computed_but_unused_by_the_covariance(coupling):
    """
    The precomputation produces LL (old name BB) couplings, but no
    covariance path loads them: ACCStrategy.get_covariance_coupling defaults
    to TT/DD/TD/DT. Documenting that here so the wasted work is visible, and
    so enabling B modes has an obvious starting point. One reader does
    exist -- the error budget (``approximations/acc_budget.py``,
    ``tests/test_acc_error_budget.py``) measures the E->B leakage from
    ``TTxLL`` -- but it only reports; the covariance still never sees LL.
    """
    inner = coupling[ELL]
    assert "LLxLL" in inner
    assert np.abs(inner["LLxLL"]).max() > 0
