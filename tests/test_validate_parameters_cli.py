"""
Tests for the ``validate-parameters`` console script.

pyproject.toml declared this entry point while the module did not exist, so
installing the package created a command that failed on import. These tests
pin that it exists, is importable under the name the entry point uses, and
reports the right exit status.
"""

import textwrap

import pytest

from cmbcov.scripts.validate_parameters import main

GOOD = """
cov_path: ./out
cov_name: covariance_matrix.dat
frequencies: ['090GHz', '150GHz']
stokes: ['T', 'E']
lmax: 200
lmin: 2
bins:
  - [2, 30, 1]
  - [30, 200, 10]
mask_name: mask.fits
mask_path: ./
covariance_approximation: nka
"""


def _write(tmp_path, text, name="params.yml"):
    path = tmp_path / name
    path.write_text(textwrap.dedent(text))
    return str(path)


def test_entry_point_target_is_importable():
    """The exact symbol pyproject.toml points at must exist."""
    from cmbcov.scripts import validate_parameters

    assert callable(validate_parameters.main)


def test_valid_file_returns_zero(tmp_path):
    assert main(["-p", _write(tmp_path, GOOD)]) == 0


def test_missing_required_parameter_fails(tmp_path):
    broken = GOOD.replace("lmax: 200\n", "")
    assert main(["-p", _write(tmp_path, broken)]) == 1


def test_strict_mode_turns_warnings_into_failure(tmp_path):
    """A missing mask file is only a warning by default."""
    path = _write(tmp_path, GOOD)
    assert main(["-p", path]) == 0
    assert main(["-p", path, "--strict"]) == 1


def test_missing_file_exits_cleanly(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        main(["-p", str(tmp_path / "nope.yml")])
    assert "not found" in str(excinfo.value)


def test_malformed_yaml_reports_location(tmp_path):
    """yaml's mark names the line/column; it must not be swallowed."""
    path = _write(tmp_path, "cov_path: ./out\n  bad indent: [1,\n")
    with pytest.raises(SystemExit) as excinfo:
        main(["-p", path])
    assert "could not parse" in str(excinfo.value)
