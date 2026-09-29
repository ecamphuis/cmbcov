"""
``cmbcov.conditioning``: whether a delivered covariance is usable in a
Gaussian likelihood, and the report ``CovarianceMatrixGenerator`` writes
beside every covariance (``conditioning.txt``).
"""

import glob
import os

import numpy as np
import pytest

from cmbcov.conditioning import conditioning_report, format_conditioning_report

healpy = pytest.importorskip("healpy")

DATA = os.path.join(os.path.dirname(__file__), "data")


def test_identity_is_perfectly_conditioned():
    report = conditioning_report(np.eye(5))
    assert report["n"] == 5
    assert report["min_eigenvalue"] == pytest.approx(1.0)
    assert report["max_eigenvalue"] == pytest.approx(1.0)
    assert report["condition_number"] == pytest.approx(1.0)
    assert report["positive_definite"]
    assert report["n_negative"] == 0
    assert report["n_bad_diagonal"] == 0


def test_a_negative_eigenvalue_is_reported_and_not_positive_definite():
    # [[1, 2], [2, 1]] has eigenvalues -1 and 3.
    cov = np.array([[1.0, 2.0], [2.0, 1.0]])
    report = conditioning_report(cov)
    assert not report["positive_definite"]
    assert report["n_negative"] == 1
    assert report["condition_number"] is None
    assert report["min_eigenvalue"] == pytest.approx(-1.0)
    assert report["max_eigenvalue"] == pytest.approx(3.0)


def test_the_formatter_warns_that_a_negative_eigenvalue_is_unusable():
    cov = np.array([[1.0, 2.0], [2.0, 1.0]])
    text = format_conditioning_report(conditioning_report(cov))
    assert "not usable" in text
    assert "condition_number = None" in text


def test_diagonal_rescaling_leaves_the_report_unchanged():
    """
    The report is built on the correlation matrix R = D^-1/2 C D^-1/2, so
    it must not depend on the units of each row/column: diag(s) C diag(s)
    reports the same conditioning as C itself.
    """
    rng = np.random.default_rng(0)
    a = rng.normal(size=(6, 6))
    cov = a @ a.T + 6 * np.eye(6)  # symmetric positive definite
    scale = np.diag(rng.uniform(0.1, 10.0, size=6))
    rescaled = scale @ cov @ scale

    report = conditioning_report(cov)
    rescaled_report = conditioning_report(rescaled)

    assert rescaled_report["min_eigenvalue"] == pytest.approx(report["min_eigenvalue"])
    assert rescaled_report["max_eigenvalue"] == pytest.approx(report["max_eigenvalue"])
    assert rescaled_report["condition_number"] == pytest.approx(
        report["condition_number"]
    )


def test_a_zero_diagonal_entry_is_reported_not_a_division_by_zero():
    cov = np.array([[0.0, 0.0], [0.0, 1.0]])
    report = conditioning_report(cov)
    assert report["n_bad_diagonal"] == 1
    assert np.isfinite(report["min_eigenvalue"])
    assert np.isfinite(report["max_eigenvalue"])
    assert "n_bad_diagonal" in format_conditioning_report(report)


def test_a_negative_diagonal_entry_is_reported_not_a_crash():
    cov = np.array([[-1.0, 0.0], [0.0, 1.0]])
    report = conditioning_report(cov)
    assert report["n_bad_diagonal"] == 1
    assert np.isfinite(report["min_eigenvalue"])
    assert np.isfinite(report["max_eigenvalue"])


# --------------------------------------------------------- end to end


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    """
    A small end-to-end run, reusing the suite's own baseline parameter
    file (``tests/data/baseline_params.yml``, the fixture behind
    ``tests/test_end_to_end_baseline.py``) rather than assembling a new
    pipeline just to check that a report file appears.
    """
    from cmbcov import CovarianceMatrixGenerator

    out = tmp_path_factory.mktemp("conditioning_run")
    template = open(os.path.join(DATA, "baseline_params.yml")).read()
    params = out / "params.yml"
    params.write_text(
        template.replace("PLACEHOLDER_OUT", str(out)).replace(
            "PLACEHOLDER_DATA", os.path.abspath(DATA)
        )
    )
    CovarianceMatrixGenerator(str(params)).run_full_analysis()
    version = sorted(glob.glob(os.path.join(str(out), "v*")))[-1]
    return version


def test_a_run_writes_the_conditioning_report(small_run):
    path = os.path.join(small_run, "conditioning.txt")
    assert os.path.exists(path)
    text = open(path).read()
    assert "min_eigenvalue" in text
    assert "condition_number" in text
    # the covariance itself is there and untouched by the report
    assert os.path.exists(os.path.join(small_run, "covariance_matrix.dat"))


def test_a_failure_inside_the_report_does_not_fail_the_run(
    tmp_path, monkeypatch, caplog
):
    """
    Diagnostics must never fail a run (like ``_save_error_budget``): a
    broken ``conditioning_report`` only logs a warning.
    """
    import logging

    import cmbcov.generator.generator as generator_module
    from cmbcov import CovarianceMatrixGenerator

    def _broken(_cov):
        raise RuntimeError("boom")

    monkeypatch.setattr(generator_module, "conditioning_report", _broken)

    template = open(os.path.join(DATA, "baseline_params.yml")).read()
    params = tmp_path / "params.yml"
    params.write_text(
        template.replace("PLACEHOLDER_OUT", str(tmp_path)).replace(
            "PLACEHOLDER_DATA", os.path.abspath(DATA)
        )
    )
    with caplog.at_level(logging.WARNING):
        CovarianceMatrixGenerator(str(params)).run_full_analysis()
    version = sorted(glob.glob(os.path.join(str(tmp_path), "v*")))[-1]
    assert os.path.exists(os.path.join(version, "covariance_matrix.dat"))
    assert not os.path.exists(os.path.join(version, "conditioning.txt"))
    assert any(
        "Could not build the conditioning report" in record.message
        for record in caplog.records
    )
