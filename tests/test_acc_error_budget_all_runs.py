"""
Every ACC run writes ``error_budget.txt`` with the normalisation section
(``acc_budget.error_budget``).

The normalisation check (Eq. 22 sum rule, kernels against ``Cov.Xi``) needs
only the ``TT x TT`` kernel, so it applies to TT-only runs and runs with a B
observable as well as to the T/E runs that always had a budget. For those two
kinds the leakage and translation sections are reported as not applicable.

* a T/E run with a polarised leg is unchanged: same dict keys, same file text
  (``tests/reference/acc_error_budget_te.txt``, recorded before this change,
  numbers compared to 1e-6), same log line;
* a TT-only run and a run with a B observable write the file, with a finite
  normalisation mismatch and the "NOT APPLICABLE" sections, and log the
  normalisation line only;
* an EE-only or BB-only run, whose cache has no ``TT x TT`` kernel, writes the
  file with the skip note.
"""

import glob
import logging
import os
import re
import shutil
import warnings

import numpy as np
import pytest
import yaml

pytest.importorskip("healpy")

from test_acc_bmode_assembly import _write_cls  # noqa: E402
from test_acc_error_budget import _params_file  # noqa: E402

from cmbcov import CovarianceMatrixGenerator  # noqa: E402

REFERENCE = os.path.join(os.path.dirname(__file__), "reference")
NUMBER = re.compile(r"[-+]?\d+\.\d+(?:e[-+]?\d+)?")

pytestmark = pytest.mark.filterwarnings(
    "ignore:internal band-limit margin",
    "ignore:centralell=",
    "ignore::UserWarning",
)


def _assert_same_text(text, reference, rel=1e-6):
    """Same text; floats (written with a decimal point) equal to ``rel``."""
    assert NUMBER.sub("#", text) == NUMBER.sub("#", reference)
    for got, expected in zip(NUMBER.findall(text), NUMBER.findall(reference)):
        assert float(got) == pytest.approx(float(expected), rel=rel, abs=1e-12)


def test_te_run_budget_is_unchanged(tmp_path, caplog):
    params = _params_file(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        CovarianceMatrixGenerator(params).precompute_acc_kernels()
        generator = CovarianceMatrixGenerator(params)
        with caplog.at_level(logging.INFO):
            generator.run_full_analysis()
    path = tmp_path / "out" / "v0" / "error_budget.txt"

    with open(os.path.join(REFERENCE, "acc_error_budget_te.txt")) as handle:
        _assert_same_text(path.read_text(), handle.read())

    budget = generator.covariance_instance.error_budget(
        generator.cov_keys, band_edges=[20, 48]
    )
    assert set(budget) == {
        "method",
        "centralell",
        "dmax",
        "lmin",
        "lmax",
        "leakage",
        "normalisation",
        "blocks",
        "translation",
        "measured_reference",
    }
    assert "applicable" not in budget
    assert budget["leakage"]["lambda"] is not None

    lines = [r.message for r in caplog.records if "ACC error budget" in r.message]
    assert len(lines) == 1
    _assert_same_text(
        lines[0],
        "ACC error budget: E->B leakage lambda = 5.112e-02, largest polarised "
        "block bias +1.022e-01; the Eq. 33 translation error is NOT bounded "
        f"(max |l - l*| = 32). Report: {path}",
        rel=1e-3,
    )
    normalisation = [r for r in caplog.records if "ACC normalisation" in r.message]
    assert normalisation and all(r.levelno == logging.INFO for r in normalisation)


DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))


def _run(tmp_path, observables, caplog, acc_precompute=None):
    """A one-frequency ACC run on the baseline mask, kernels precomputed for it."""
    work = str(tmp_path)
    shutil.copy(os.path.join(DATA, "baseline_mask.fits"), work)
    _write_cls(os.path.join(work, "cls.dat"), False)
    params = {
        "cov_path": os.path.join(work, "out"),
        "cov_name": "cov.dat",
        "frequencies": ["090GHz"],
        "observables": observables,
        "lmax": 20,
        "lmin": 2,
        "bins": [[2, 20, 3]],
        "mask_name": "baseline_mask.fits",
        "mask_path": work,
        "covariance_approximation": "acc",
        "dmax": 3,
        "centralell": 10,
        "cmb_spectrum": os.path.join(work, "cls.dat"),
        "beams": {"090GHz": 5.0},
        "pixwin": 32,
        "nl": {"090GHz": 10.0},
        "polspice_postprocess": False,
        "acc_precompute": {
            "nside": 16,
            "grid": "gl",
            "lw": 10,
            **(acc_precompute or {}),
        },
    }
    path = os.path.join(work, "params.yml")
    with open(path, "w") as handle:
        yaml.dump(params, handle)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        CovarianceMatrixGenerator(path).precompute_acc_kernels()
        generator = CovarianceMatrixGenerator(path)
        with caplog.at_level(logging.INFO):
            generator.run_full_analysis()
    version = sorted(glob.glob(os.path.join(work, "out", "v*")))[-1]
    with open(os.path.join(version, "error_budget.txt")) as handle:
        text = handle.read()
    budget = generator.covariance_instance.error_budget(generator.cov_keys)
    return text, budget


@pytest.mark.parametrize(
    "observables, reason",
    [
        (["TT"], "tt_only"),
        (["TT", "EE", "TE", "BB"], "b_run"),
    ],
    ids=["tt-only", "b-run"],
)
def test_tt_only_and_b_runs_write_the_normalisation_check(
    tmp_path, caplog, observables, reason
):
    text, budget = _run(tmp_path, observables, caplog)

    assert budget["applicable"] is False
    assert budget["reason"] == reason
    assert budget["leakage"]["applicable"] is False
    assert budget["translation"]["applicable"] is False
    assert "lambda" not in budget["leakage"]  # not "could not be bounded"
    assert budget["blocks"] == {}
    mismatch = budget["normalisation"]["max_relative_mismatch"]
    assert mismatch is not None and np.isfinite(mismatch)

    assert "1. E->B leakage of the kernel set: NOT APPLICABLE" in text
    assert "2. Eq. 33 translation error: NOT APPLICABLE" in text
    assert "3. Normalisation: kernels against Cov.Xi (Eq. 22 sum rule)" in text
    assert f"max relative mismatch = {mismatch:.3e}" in text
    if reason == "b_run":
        assert "per-Wick-term" in text and "Known limits" in text
    else:
        assert "no polarised leg" in text
    assert "NOT AVAILABLE" not in text

    messages = [r for r in caplog.records if "ACC error budget" in r.message]
    assert not messages  # no leakage headline for these runs
    norm = [r for r in caplog.records if "ACC normalisation" in r.message]
    assert len(norm) >= 1
    expected = logging.WARNING if budget["normalisation"]["exceeds"] else logging.INFO
    assert all(r.levelno == expected for r in norm)


@pytest.mark.parametrize(
    "observables, acc_precompute",
    [(["EE"], {"spectra": ["EE"]}), (["EE", "BB"], {})],
    ids=["ee-only", "ee-bb"],
)
def test_runs_without_a_tt_kernel_write_the_skip_note(
    tmp_path, caplog, observables, acc_precompute
):
    text, budget = _run(tmp_path, observables, caplog, acc_precompute)

    assert budget["normalisation"]["max_relative_mismatch"] is None
    assert "3. Normalisation: kernels against Cov.Xi" in text
    flat = " ".join(text.split())  # the note is wrapped
    assert "NOT AVAILABLE: no TT x TT coupling kernel in the cache" in flat
    assert "nothing is compared for this run" in flat
