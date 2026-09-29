"""
``dmax`` vs the output binning (docs/theory/acc.md, "dmax").

ACC computes the covariance only on the diagonals ``|l - l'| < dmax`` and
zeroes the rest. A bandpower variance sums multipole-pair separations
``0..w-1`` (``w`` = bin width), and the covariance of two adjacent
bandpowers sums separations ``1..2w-1``. So ``dmax >= w`` is needed for the
error bars and ``dmax >= 2w`` for adjacent-bandpower correlations; below
either threshold, ACC silently drops pairs rather than raising, so
:meth:`~cmbcov.generator.parameter_validation.ParameterValidator._check_parameter_consistency`
warns instead.
"""

import os

from cmbcov.generator.parameter_validation import ParameterValidator

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))


def _base_params(**overrides):
    params = {
        "cov_path": "/tmp/does-not-matter",
        "cov_name": "cov.dat",
        "frequencies": ["090GHz"],
        "lmax": 200,
        "lmin": 2,
        "bins": [[2, 200, 50]],
        "mask_name": "baseline_mask.fits",
        "mask_path": DATA,
        "covariance_approximation": "acc",
        "dmax": 100,
    }
    params.update(overrides)
    return params


def _warnings(params):
    validator = ParameterValidator()
    validator.validate(params)
    return validator.warnings


def _dmax_warnings(params):
    return [w for w in _warnings(params) if "dmax" in w]


def test_no_warning_when_dmax_covers_adjacent_correlations():
    # w = 50, dmax = 2w = 100: both thresholds met.
    params = _base_params(dmax=100)
    assert _dmax_warnings(params) == []


def test_no_warning_when_dmax_exceeds_two_w():
    params = _base_params(dmax=150)
    assert _dmax_warnings(params) == []


def test_neighbour_warning_when_w_le_dmax_lt_2w():
    # w = 50 <= dmax = 60 < 2w = 100: variances complete, correlations not.
    params = _base_params(dmax=60)
    warnings = _dmax_warnings(params)
    assert len(warnings) == 1
    message = warnings[0]
    assert "60" in message
    assert "50" in message
    assert "100" in message
    assert "adjacent" in message or "neighbouring" in message


def test_variance_warning_when_dmax_lt_w():
    # w = 50, dmax = 20 < w: even the variances are underestimated.
    params = _base_params(dmax=20)
    warnings = _dmax_warnings(params)
    assert len(warnings) == 1
    message = warnings[0]
    assert "20" in message
    assert "50" in message
    assert "100" in message
    assert "underestimated" in message


def test_widest_bin_segment_used_when_several_differ():
    # Two bin segments, widths 20 and 50: the rule uses the widest (50).
    params = _base_params(
        bins=[[2, 100, 20], [100, 200, 50]],
        dmax=30,
    )
    warnings = _dmax_warnings(params)
    assert len(warnings) == 1
    # dmax=30 < w=50: variance-underestimate warning, not the neighbour one.
    assert "underestimated" in warnings[0]
    assert "50" in warnings[0]


def test_no_warning_for_non_acc_methods():
    for method in ("nka", "inka"):
        params = _base_params(covariance_approximation=method, dmax=1)
        assert _dmax_warnings(params) == []


def test_no_warning_when_dmax_not_set():
    params = _base_params()
    del params["dmax"]
    assert _dmax_warnings(params) == []
