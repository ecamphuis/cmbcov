"""
Which Xi channel normalises each pair of Wick kernels.

`Cov._compute_norm_xi` supplies the geometric prefactor for every covariance
term. The correct channel is fixed by the paper's Eq. 22 generalised to
polarisation: `sum Theta^{s1 x s2} = (2l+1)(2l'+1) Xi^{ss'}[W^2]`. The
channel follows a spin-weight
rule: with w(TT) = 0, w(EE) = w(BB) = 1, w(TE) = w(ET) = 1/2, the exact
normalisation of Theta^{s1 x s2} is Xi^00 (Xi^20 / Xi^00) ** (w1 + w2), so

    w = 0    TT x TT                    Xi^00
    w = 1    TT x EE, TE x TE, TE x ET  Xi^20
    w = 2    EE x EE                    Xi^{EE->EE}
    w = 1/2  TT x TE                    no channel exists; Xi^20 is nearest
    w = 3/2  EE x TE                    no channel exists; Xi^{EE->EE} is nearest

These tests pin the assignment itself, not a number downstream of it: eight
entries using Xi^00 instead would be wrong by a further factor Xi^00/Xi^20
(1.10 at l = 16 on the test mask, more at low l).

The identity underlying the w = 1 and w = 2 rows is verified directly against
the covariance coupling kernels in tests/test_acc_gl.py and
tests/test_acc_gl_pol.py; here we only guard the table.
"""

import os
import tempfile
import warnings

import numpy as np
import pytest

from cmbcov.covariance import Cov, CovarianceConfig, CovarianceMethod
from cmbcov.kernels.coupling import (
    KERNEL_TE,
    KERNEL_TT,
    KERNEL_EEmBB,
    KERNEL_EEpBB,
)

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
STOKES = ("TT", "EE", "TE", "ET")

#: Total spin weight of each kernel pair, from the rule in the module docstring.
WEIGHT = {"TT": 0.0, "EE": 1.0, "BB": 1.0, "TE": 0.5, "ET": 0.5}

#: Expected channel per entry: "00", "20" or "EE".
EXPECTED = {}
for _a in STOKES:
    for _b in STOKES:
        _w = WEIGHT[_a] + WEIGHT[_b]
        # w = 1/2 and w = 3/2 have no exact channel; the nearest rung is used.
        EXPECTED[(_a, _b)] = {0.0: "00", 0.5: "20", 1.0: "20", 1.5: "EE", 2.0: "EE"}[_w]


@pytest.fixture(scope="module")
def channels():
    """norm_Xi and the three Xi channels it can draw on, for one small mask."""
    config = CovarianceConfig(method=CovarianceMethod.NKA, lmax=48)
    cov = Cov(
        "baseline_mask.fits",
        config=config,
        mask_path=DATA,
        save_dir=tempfile.mkdtemp(),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xi = cov.Xi
        norm = cov.norm_Xi
    available = {
        "00": xi[KERNEL_TT],
        "20": xi[KERNEL_TE],
        "EE": 0.5 * (xi[KERNEL_EEpBB] + xi[KERNEL_EEmBB]),
    }
    return norm, available


def test_every_stokes_pair_is_present(channels):
    norm, _ = channels
    assert set(norm) == {(a, b) for a in STOKES for b in STOKES}


@pytest.mark.parametrize("pair", sorted(EXPECTED))
def test_channel_matches_the_spin_weight_rule(channels, pair):
    norm, available = channels
    want = EXPECTED[pair]
    np.testing.assert_array_equal(
        norm[pair],
        available[want],
        err_msg=f"norm_Xi{pair} should be the Xi^{want} channel",
    )


def test_the_three_channels_are_genuinely_different(channels):
    """Guard against the test passing because two channels coincide."""
    _, available = channels
    scale = np.abs(available["00"]).max()
    assert np.abs(available["00"] - available["20"]).max() > 1e-3 * scale
    assert np.abs(available["20"] - available["EE"]).max() > 1e-3 * scale


def test_xi00_over_xi20_is_the_size_the_analysis_measured(channels):
    """
    The eight entries normalised by Xi^20 instead of Xi^00 move by this
    ratio. On the nside=16 baseline mask the analysis note measures
    Xi^00/Xi^20 = 1.0989 at
    (16, 16), rising steeply below l ~ 10 where the spin-2 3j symbol departs
    most from the spin-0 one. A regression here means the kernels themselves
    changed, not the table above.
    """
    _, available = channels
    ratio = available["00"][16, 16] / available["20"][16, 16]
    assert ratio == pytest.approx(1.0989, abs=5e-3)
    assert available["00"][4, 4] / available["20"][4, 4] > 1.5


def test_norm_xi_is_symmetric_in_ell(channels):
    """Xi^{ss'}_{l l'} is symmetric; the covariance assembly relies on it."""
    norm, _ = channels
    for pair, kernel in norm.items():
        scale = np.abs(kernel).max()
        assert np.abs(kernel - kernel.T).max() < 1e-12 * scale, pair
