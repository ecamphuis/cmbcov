r"""
BB-only NKA/INKA escape hatch (docs/theory/bmode_kernels.md,
"BB-only NKA/INKA escape hatch"; ``generator.parameter_validation.APPROXIMATIONS_SUPPORTING_B``).

NKA and INKA cannot compute a B-mode block in general (they collapse EE and
BB by construction), except for ``observables: [BB]`` alone: a
leakage-neglected shortcut for a high-``ell`` ``Cov(BB, BB)`` without an ACC
precompute. This file pins:

* the validation accept/reject boundary (``[BB]`` alone is allowed for
  NKA/INKA; anything else with a B observable is still rejected, for both
  the parameter-file validator and the ``Cov`` runtime guard);
* NKA's ``Cov(BB, BB)`` equals the explicit ``2 (C^BB)^2 Xi^{EE->EE}``
  formula;
* INKA smooths ``C^BB`` with the same kernel as ``C^EE``;
* the PolSpice path is the diagonal level-1-style ``G+`` transform, not
  ``pseudo_to_spice_bmode``;
* the leakage warning fires with a sensible ``ell_safe``, which decreases
  as ``C^BB`` grows relative to ``C^EE``;
* an end-to-end ``CovarianceMatrixGenerator`` run with
  ``observables: [BB]``, both methods, PolSpice on and off.

The
``(1 + leak/signal)^2`` estimator behind the warning is only a rough guide,
not a correction: measured against ``exact_covariance_row_pol`` on a small
config, it tracks the ell dependence and is conservative (over- rather than
under-estimates the missed variance) everywhere tested, agreeing within
about 20-40% for ell above ~15 and over-estimating by a larger factor at the
very lowest multipoles (see the docstring of ``Cov.bb_leakage_ell_safe`` and
the module docstring below for the numbers).
"""

import os
import warnings

import numpy as np
import pytest

from cmbcov.approximations.inka import (
    INKAStrategy,
    renormalised_kernel,
)
from cmbcov.approximations.nka import NKAStrategy
from cmbcov.covariance import (
    Cov,
    CovarianceConfig,
    CovarianceMethod,
    bb_only_observable,
)
from cmbcov.generator.parameter_validation import (
    ParameterValidator,
)
from cmbcov.keys import CovKey, CovKeys
from cmbcov.postprocess import CovariancePostProcessor

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))


def _base_params(**overrides):
    params = {
        "cov_path": "/tmp/does-not-matter",
        "cov_name": "cov.dat",
        "frequencies": ["090GHz"],
        "lmax": 20,
        "lmin": 2,
        "bins": [[2, 20, 6]],
        "mask_name": "baseline_mask.fits",
        "mask_path": DATA,
        "covariance_approximation": "nka",
    }
    params.update(overrides)
    return params


def _errors(params):
    validator = ParameterValidator()
    validator.validate(params)
    return validator.errors


def _cov(lmax=40, method=CovarianceMethod.NKA, polspice_postprocess=False):
    config = CovarianceConfig(
        method=method, lmax=lmax, lmin=2, polspice_postprocess=polspice_postprocess
    )
    return Cov("baseline_mask.fits", config=config, mask_path=DATA, save_dir="/tmp")


def _bb_spectra(lmax, bb_amplitude=1.0):
    ell = np.arange(lmax)
    cl_ee = np.zeros(lmax)
    cl_ee[2:] = 1.0 / (ell[2:] * (ell[2:] + 1)) ** 0.9
    cl_bb = bb_amplitude * cl_ee
    return cl_bb, cl_ee


# --------------------------------------------------------------------------- #
# Validation: accept/reject boundary
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("approximation", ["nka", "inka"])
def test_bb_only_nka_inka_accepted(approximation):
    params = _base_params(observables=["BB"], covariance_approximation=approximation)
    errors = _errors(params)
    assert not any("observables" in e or "cannot compute" in e for e in errors), errors


@pytest.mark.parametrize("approximation", ["nka", "inka"])
@pytest.mark.parametrize(
    "observables", [["TT", "BB"], ["EE", "BB"], ["TB"], ["EB"], ["TT", "EE", "BB"]]
)
def test_other_b_observable_sets_still_rejected(approximation, observables):
    params = _base_params(
        observables=observables, covariance_approximation=approximation
    )
    errors = _errors(params)
    assert any("cannot compute" in e for e in errors), errors


def test_rejection_message_names_bb_only_and_acc():
    params = _base_params(observables=["TT", "BB"], covariance_approximation="nka")
    (error,) = [e for e in _errors(params) if "cannot compute" in e]
    assert "BB" in error and "escape hatch" in error
    assert "ACC" in error or "acc" in error


def test_bb_only_acc_still_accepted():
    """The pre-existing ACC path over 'observables: [BB]' is untouched."""
    params = _base_params(
        observables=["BB"],
        covariance_approximation="acc",
        dmax=1,
        centralell=10,
    )
    errors = _errors(params)
    assert not any("observables" in e or "cannot compute" in e for e in errors), errors


def test_cov_runtime_guard_accepts_bb_only_nka():
    cov = _cov()
    keys = CovKeys([], ["090GHz", "150GHz"], observables=["BB"])
    cl_bb, _ = _bb_spectra(cov.lmax)
    cl = {
        freq_key: {"BB": cl_bb}
        for freq_key in ("090GHz090GHz", "090GHz150GHz", "150GHz150GHz")
    }
    _, _, matrix = cov.compute_covariance_matrix([2, 10, 20, 30, 39], keys, cl)
    assert np.isfinite(matrix).all()


def test_cov_runtime_guard_still_rejects_mixed_b_nka():
    cov = _cov()
    keys = CovKeys([], ["090GHz"], observables=["TT", "BB"])
    with pytest.raises(ValueError, match="needs the ACC method"):
        cov.compute_covariance_matrix([2, 10], keys, cl={})


def test_bb_only_observable_helper():
    assert bb_only_observable(CovKeys([], ["a"], observables=["BB"]))
    assert not bb_only_observable(CovKeys([], ["a"], observables=["TT", "BB"]))
    assert not bb_only_observable(CovKeys(["T", "E"], ["a"]))


# --------------------------------------------------------------------------- #
# NKA: the explicit 2 (C^BB)^2 Xi^{EE->EE} formula
# --------------------------------------------------------------------------- #


def test_nka_bb_only_matches_explicit_formula():
    cov = _cov(lmax=40)
    lmax = cov.lmax
    cl_bb, _ = _bb_spectra(lmax)
    cl = {"090GHz090GHz": {"BB": cl_bb}}
    cov_key = CovKey(("B", "B", "B", "B"), ("090GHz",) * 4)
    strategy = NKAStrategy(cov)
    got = strategy.compute_covariance_term(cov_key, cl)

    xi_ee = cov.norm_Xi[("EE", "EE")]
    expected = 2.0 * np.outer(cl_bb, cl_bb) * xi_ee
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_nka_bb_only_cross_frequency_matches_ee_with_same_numbers():
    """Cross-frequency BB is fine (the brief: 'any number of frequencies').
    nka.py treats 'BB' exactly like 'EE' in both the geometric channel and
    the cl-to-average conversion, so feeding the same numbers in as 'EE'
    (the pre-existing, well-tested T/E-only path) must give the same
    two-Wick-contraction result as 'BB', for a genuinely cross-frequency
    key (frequencies interleaved, not a auto-auto pair)."""
    cov = _cov(lmax=30)
    lmax = cov.lmax
    cl_bb_1, _ = _bb_spectra(lmax, bb_amplitude=1.0)
    cl_bb_2, _ = _bb_spectra(lmax, bb_amplitude=2.0)
    cl_cross, _ = _bb_spectra(lmax, bb_amplitude=1.5)
    cl_bb = {
        "090GHz090GHz": {"BB": cl_bb_1},
        "150GHz150GHz": {"BB": cl_bb_2},
        "090GHz150GHz": {"BB": cl_cross},
        "150GHz090GHz": {"BB": cl_cross},
    }
    cl_ee = {freq: {"EE": spectra["BB"]} for freq, spectra in cl_bb.items()}

    freq = ("090GHz", "150GHz", "090GHz", "150GHz")
    bb_key = CovKey(("B", "B", "B", "B"), freq)
    ee_key = CovKey(("E", "E", "E", "E"), freq)
    strategy = NKAStrategy(cov)

    got_bb = strategy.compute_covariance_term(bb_key, cl_bb)
    got_ee = strategy.compute_covariance_term(ee_key, cl_ee)
    np.testing.assert_allclose(got_bb, got_ee, rtol=1e-12)
    assert np.any(got_bb != 0)


# --------------------------------------------------------------------------- #
# INKA: smoothed with the EE (M+) kernel
# --------------------------------------------------------------------------- #


def test_inka_bb_only_uses_ee_kernel():
    cov = _cov(lmax=40, method=CovarianceMethod.INKA)
    lmax = cov.lmax
    cl_bb, _ = _bb_spectra(lmax)
    cl = {"090GHz090GHz": {"BB": cl_bb}}
    cov_key = CovKey(("B", "B", "B", "B"), ("090GHz",) * 4)

    got = INKAStrategy(cov).compute_covariance_term(cov_key, cl)

    m_plus = renormalised_kernel((cov.M[1] + cov.M[2]) / 2)
    smoothed_bb = m_plus @ cl_bb
    xi_ee = cov.norm_Xi[("EE", "EE")]
    expected = 2.0 * np.outer(smoothed_bb, smoothed_bb) * xi_ee
    np.testing.assert_allclose(got, expected, rtol=1e-12)


# --------------------------------------------------------------------------- #
# PolSpice: diagonal G+ transform, not pseudo_to_spice_bmode
# --------------------------------------------------------------------------- #


def test_polspice_bb_only_is_diagonal_gplus():
    cov = _cov(lmax=30, polspice_postprocess=True)
    g_kernels = cov.G  # (G0, Gplus, Gminus, Gx)
    post = CovariancePostProcessor(g_kernels)
    rng = np.random.default_rng(0)
    mat = rng.normal(size=(cov.lmax, cov.lmax))
    mat = mat + mat.T

    got = post.pseudo_to_spice(mat, "BBxBB")
    expected = g_kernels[1] @ mat @ g_kernels[1].T
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_polspice_bb_only_end_to_end_matches_diagonal_transform():
    """The full compute_covariance_matrix path (polspice on) for a BB-only
    NKA run equals the raw NKA block put through the diagonal transform by
    hand -- i.e. it never goes through pseudo_to_spice_bmode (which would
    need a pseudo EE block this run never computes)."""
    lmax = 30
    cov_polspice = _cov(lmax=lmax, polspice_postprocess=True)
    cov_raw = _cov(lmax=lmax, polspice_postprocess=False)
    keys = CovKeys([], ["090GHz"], observables=["BB"])
    cl_bb, _ = _bb_spectra(lmax)
    cl = {"090GHz090GHz": {"BB": cl_bb}}

    lbins = [2, 8, 14, 20, 29]
    _, bin_matrix, matrix_polspice = cov_polspice.compute_covariance_matrix(
        lbins, keys, cl
    )
    _, _, matrix_raw = cov_raw.compute_covariance_matrix(lbins, keys, cl)

    raw_strategy = NKAStrategy(cov_raw)
    cov_key = next(iter(keys.keys()))
    raw_block = raw_strategy.compute_covariance_term(cov_key, cl)
    gplus = cov_polspice.G[1]
    transformed = gplus @ raw_block @ gplus.T
    expected_binned = bin_matrix @ transformed @ bin_matrix.T

    np.testing.assert_allclose(matrix_polspice, expected_binned, rtol=1e-10)
    assert not np.allclose(matrix_polspice, matrix_raw)


# --------------------------------------------------------------------------- #
# The leakage warning
# --------------------------------------------------------------------------- #


def test_bb_leakage_ell_safe_shape_and_bounds():
    cov = _cov(lmax=40)
    cl_bb, cl_ee = _bb_spectra(40, bb_amplitude=1.0)
    report = cov.bb_leakage_ell_safe(cl_bb, cl_ee)
    assert report["lmax"] == 39
    assert 2 <= report["ell_safe"] <= report["lmax"] + 1
    assert report["lowest_bin_factor"] >= 1.0
    assert report["reliable"] == (report["ell_safe"] <= report["lmax"])
    assert report["inflation"].shape == (40,)


def test_bb_leakage_ell_safe_decreases_as_cbb_grows():
    cov = _cov(lmax=40)
    _, cl_ee = _bb_spectra(40)
    ell_safe_values = []
    for amplitude in (0.01, 1.0, 10.0, 100.0):
        cl_bb = amplitude * cl_ee
        report = cov.bb_leakage_ell_safe(cl_bb, cl_ee)
        ell_safe_values.append(report["ell_safe"])
    assert ell_safe_values == sorted(ell_safe_values, reverse=True)
    assert ell_safe_values[0] > ell_safe_values[-1]


def test_warn_bb_only_leakage_fires_and_reports_ell_safe():
    cov = _cov(lmax=40)
    cl_bb, cl_ee = _bb_spectra(40, bb_amplitude=1.0)
    with pytest.warns(UserWarning, match="ell_safe|ell >="):
        report = cov.warn_bb_only_leakage(cl_bb, cl_ee, CovarianceMethod.NKA)
    assert isinstance(report["ell_safe"], int)


def test_warn_bb_only_leakage_unreliable_everywhere_message():
    cov = _cov(lmax=20)
    cl_bb, cl_ee = _bb_spectra(20, bb_amplitude=1e-4)
    with pytest.warns(UserWarning, match="unreliable"):
        report = cov.warn_bb_only_leakage(cl_bb, cl_ee, CovarianceMethod.INKA)
    assert not report["reliable"]


# --------------------------------------------------------------------------- #
# End to end through CovarianceMatrixGenerator
# --------------------------------------------------------------------------- #


def _write_bb_cls(path, lmax=80):
    ell = np.arange(lmax)
    tt = np.zeros(lmax)
    tt[2:] = 1e3 / (ell[2:] * (ell[2:] + 1)) ** 0.9
    ee = 0.1 * tt
    bb = 0.05 * tt
    te = 0.5 * np.sqrt(tt * ee)
    zero = np.zeros(lmax)
    # 9-spectrum layout: TT EE BB TE TB EB ET BT BE
    np.savetxt(path, np.column_stack([ell, tt, ee, bb, te, zero, zero, te, zero, zero]))


@pytest.mark.parametrize("approximation", ["nka", "inka"])
@pytest.mark.parametrize("polspice_postprocess", [False, True])
def test_generator_end_to_end_bb_only(tmp_path, approximation, polspice_postprocess):
    import shutil

    import yaml

    from cmbcov import CovarianceMatrixGenerator

    work = str(tmp_path)
    shutil.copy(os.path.join(DATA, "baseline_mask.fits"), work)
    _write_bb_cls(os.path.join(work, "cls.dat"))
    params = {
        "cov_path": os.path.join(work, "out"),
        "cov_name": "cov.dat",
        "frequencies": ["090GHz"],
        "observables": ["BB"],
        "lmax": 20,
        "lmin": 2,
        "bins": [[2, 20, 6]],
        "mask_name": "baseline_mask.fits",
        "mask_path": work,
        "covariance_approximation": approximation,
        "cmb_spectrum": os.path.join(work, "cls.dat"),
        "beams": {"090GHz": 5.0},
        "pixwin": 32,
        "nl": {"090GHz090GHz": 10.0},
        "polspice_postprocess": polspice_postprocess,
    }
    path = os.path.join(work, "params.yml")
    with open(path, "w") as handle:
        yaml.dump(params, handle)

    generator = CovarianceMatrixGenerator(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        generator.run_full_analysis()

    import glob

    version = sorted(glob.glob(os.path.join(work, "out", "v*")))[-1]
    matrix = np.loadtxt(os.path.join(version, "cov.dat"))
    assert np.isfinite(matrix).all()
    assert np.allclose(matrix, matrix.T, atol=1e-12 * np.max(np.abs(matrix)))

    report_path = os.path.join(version, generator.BB_LEAKAGE_WARNING_NAME)
    assert os.path.exists(report_path)
    with open(report_path) as handle:
        content = handle.read()
    assert "ell_safe" in content
    assert generator._bb_leakage_report is not None
