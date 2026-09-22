"""
``add_tf_uncertainty`` inflates the covariance, and ``fl`` must lie in [0, 1].

``fl`` is the fraction of the signal the pipeline keeps. The covariance is
computed for spectra multiplied by ``data_model = B B pixwin fl`` and divided
back by it, and with ``add_tf_uncertainty`` each leg factor is multiplied by
``1 + sqrt((1 - fl) / 3999) >= 1``: not knowing the transfer function exactly
can only add variance, and the more signal was filtered out (smaller ``fl``)
the larger the inflation; this module pins that direction.

Outside [0, 1] both uses of ``fl`` degrade silently -- ``1 / data_model``
becomes infinite (``safe_divide`` then yields 1, i.e. no debiasing) and
``sqrt(1 - fl)`` becomes NaN (the final ``nan_to_num`` yields 1, i.e. no
uncertainty) -- so a file outside the range is refused.
"""

import glob
import os
import warnings

import numpy as np
import pytest

healpy = pytest.importorskip("healpy")

from cmbcov import CovarianceMatrixGenerator  # noqa: E402
from cmbcov.keys import CovKeys  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "data")
LMAX = 60
FREQS = ["090GHz", "150GHz"]
STOKES = ["T", "E"]
PAIRS = CovKeys(STOKES, FREQS, exclude_asymmetric_stokes=False).combined_frequencies()
KEYS = CovKeys(STOKES, FREQS, exclude_asymmetric_stokes=False)


def _write_fl(directory, values):
    os.makedirs(directory, exist_ok=True)
    template = os.path.join(directory, "fl_{}.dat")
    for pair in PAIRS:
        table = np.column_stack([np.arange(LMAX, dtype=float)] + [values] * 4)
        np.savetxt(template.format(pair), table)
    return template


def _run(tmp_path, name, extra):
    out = tmp_path / name
    template = open(os.path.join(DATA, "baseline_params.yml")).read()
    text = (
        template.replace("PLACEHOLDER_OUT", str(out))
        .replace("PLACEHOLDER_DATA", os.path.abspath(DATA))
        .replace("  - [2, 20, 6]\n  - [20, 60, 10]\n", "  - [2, 60, 1]\n")
    )
    params = tmp_path / f"{name}.yml"
    params.write_text(text + extra)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        CovarianceMatrixGenerator(str(params)).run_full_analysis()
    version = sorted(glob.glob(str(out / "v*")))[-1]
    return np.loadtxt(os.path.join(version, "covariance_matrix.dat"))


def test_tf_uncertainty_inflates_the_covariance_by_the_leg_factors(tmp_path):
    rng = np.random.default_rng(11)
    fl = rng.uniform(0.2, 0.95, LMAX)
    template = _write_fl(tmp_path / "fl", fl)
    off = _run(tmp_path, "off", f"\nfl: {template}\n")
    on = _run(tmp_path, "on", f"\nfl: {template}\nadd_tf_uncertainty: true\n")

    tf = 1 + np.sqrt((1 - fl) / 3999)
    assert np.all(tf > 1)
    lmin, n = 2, LMAX - 2
    expected = np.zeros_like(off)
    for cov_key in KEYS.keys():
        i, j = KEYS[cov_key]
        block = np.outer(tf[lmin:LMAX], tf[lmin:LMAX])
        expected[i * n : (i + 1) * n, j * n : (j + 1) * n] = block
        expected[j * n : (j + 1) * n, i * n : (i + 1) * n] = block.T

    nonzero = off != 0
    assert nonzero.mean() > 0.5
    np.testing.assert_allclose(
        on[nonzero], off[nonzero] * expected[nonzero], rtol=1e-12
    )
    # the direction: every nonzero entry grows in magnitude
    assert np.all(np.abs(on[nonzero]) > np.abs(off[nonzero]))


def test_tf_uncertainty_is_a_no_op_for_a_perfect_transfer_function(tmp_path):
    template = _write_fl(tmp_path / "unit", np.ones(LMAX))
    off = _run(tmp_path, "unit_off", f"\nfl: {template}\n")
    on = _run(tmp_path, "unit_on", f"\nfl: {template}\nadd_tf_uncertainty: true\n")
    assert on.tobytes() == off.tobytes()


@pytest.mark.parametrize(
    "values, first_bad",
    [
        (np.linspace(0.5, 1.5, LMAX), 30),  # above 1
        (np.linspace(-0.2, 0.8, LMAX), 0),  # below 0
    ],
)
def test_fl_outside_zero_one_is_refused(tmp_path, values, first_bad):
    template = _write_fl(tmp_path / "bad", values)
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        _run(tmp_path, "bad", f"\nfl: {template}\n")


def test_fl_at_the_endpoints_is_accepted(tmp_path):
    values = np.linspace(0.0, 1.0, LMAX)
    template = _write_fl(tmp_path / "edge", values)
    covariance = _run(tmp_path, "edge", f"\nfl: {template}\n")
    assert np.isfinite(covariance).all()
