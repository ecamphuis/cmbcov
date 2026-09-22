"""
The configuration and kernel-pair layer for B-mode observables
(docs/theory/bmode_kernels.md).

At this layer a user can *ask* for B-mode observables and get a
validated configuration, the right kernel pairs, and a clear
"not implemented yet" error at assembly for methods that cannot compute a
B block; a default (T/E-only) run is untouched. This file pins:

* ``stokes``/``observables`` equivalence and every rejection
  (:func:`~cmbcov.generator.parameter_validation.resolve_observables`,
  :func:`~cmbcov.generator.parameter_validation.ParameterValidator._check_observables`);
* ``CovKeys`` block lists built from ``observables``, per level, and that
  ``parity_mixed_blocks`` gates the parity-mixed ones;
* :func:`~cmbcov.bmode_wick.required_kernel_pairs` counts
  against the derivation, and that the T/E-only default equals today's pairs;
* the spectra-file TB/EB reading (zero fallback, non-zero detection) and the
  ``pix``/noise dict fix for BB/TB/EB;
* the ``NotImplementedError`` assembly gate.

Per-Wick-term ACC assembly is out of scope and not tested here (see
tests/test_acc_bmode_assembly.py).
"""

import os
import warnings

import numpy as np
import pytest
import yaml

healpy = pytest.importorskip("healpy")

from cmbcov import CovarianceMatrixGenerator  # noqa: E402
from cmbcov.approximations.acc import (  # noqa: E402
    precompute_acc_kernels,
    require_acc_cache_pairs,
)
from cmbcov.bmode_wick import (  # noqa: E402
    required_kernel_pairs,
)
from cmbcov.covariance import (  # noqa: E402
    Cov,
    CovarianceConfig,
    CovarianceMethod,
)
from cmbcov.generator.parameter_validation import (  # noqa: E402
    ParameterValidator,
    resolve_observables,
)
from cmbcov.keys import CovKeys, SpecKey  # noqa: E402
from cmbcov.spectra import (  # noqa: E402
    SpectraLoader,
    detect_parity_odd_nonzero,
    stokes_columns,
)

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


# --------------------------------------------------------------------------- #
# resolve_observables: shorthand, aliases, rejections
# --------------------------------------------------------------------------- #


def test_stokes_t_is_shorthand_for_tt():
    stokes, observables = resolve_observables({"stokes": ["T"]})
    assert stokes == ["T"]
    assert observables == ["TT"]


def test_stokes_t_e_is_shorthand_for_tt_ee_te():
    stokes, observables = resolve_observables({"stokes": ["T", "E"]})
    assert stokes == ["T", "E"]
    assert set(observables) == {"TT", "EE", "TE"}


def test_observables_equivalent_to_stokes_shorthand():
    _, from_stokes = resolve_observables({"stokes": ["T", "E"]})
    _, from_observables = resolve_observables({"observables": ["TT", "EE", "TE"]})
    assert set(from_stokes) == set(from_observables)


def test_observables_accepts_et_bt_be_aliases():
    _, observables = resolve_observables({"observables": ["TT", "ET", "BT", "BE"]})
    assert set(observables) == {"TT", "TE", "TB", "EB"}


def test_both_stokes_and_observables_is_an_error():
    with pytest.raises(ValueError, match="both 'stokes' and 'observables'"):
        resolve_observables({"stokes": ["T"], "observables": ["TT"]})


def test_neither_stokes_nor_observables_is_an_error():
    with pytest.raises(ValueError, match="neither 'stokes' nor 'observables'"):
        resolve_observables({})


def test_b_in_stokes_without_observables_is_an_error():
    with pytest.raises(ValueError, match="use 'observables'"):
        resolve_observables({"stokes": ["T", "E", "B"]})


def test_unknown_observable_is_an_error():
    with pytest.raises(ValueError, match="unknown spectra"):
        resolve_observables({"observables": ["TT", "XY"]})


def test_unknown_stokes_letter_is_an_error():
    with pytest.raises(ValueError, match="Invalid Stokes"):
        resolve_observables({"stokes": ["T", "Q"]})


# --------------------------------------------------------------------------- #
# ParameterValidator: parity_mixed_blocks and approximation rejections
# --------------------------------------------------------------------------- #


def _errors(params):
    validator = ParameterValidator()
    validator.validate(params)
    return validator.errors


def test_valid_level1_params_pass():
    assert _errors(_base_params(stokes=["T", "E"])) == []


def test_valid_bb_params_with_acc_pass_observables_check():
    params = _base_params(
        observables=["TT", "EE", "TE", "BB"],
        covariance_approximation="acc",
        dmax=1,
        centralell=10,
    )
    errors = _errors(params)
    assert not any("observables" in e or "parity" in e for e in errors)


def test_bb_with_nka_is_rejected():
    params = _base_params(observables=["TT", "EE", "BB"])
    errors = _errors(params)
    assert any("cannot compute" in e for e in errors)


def test_bb_with_inka_is_rejected():
    params = _base_params(
        observables=["TT", "EE", "BB"], covariance_approximation="inka"
    )
    errors = _errors(params)
    assert any("cannot compute" in e for e in errors)


def test_parity_mixed_without_tb_eb_is_rejected():
    params = _base_params(observables=["TT", "EE", "TE"], parity_mixed_blocks=True)
    errors = _errors(params)
    assert any("parity_mixed_blocks" in e and "TB" in e for e in errors)


def test_parity_mixed_with_no_parity_even_observable_is_rejected():
    params = _base_params(
        observables=["TB", "EB"],
        parity_mixed_blocks=True,
        covariance_approximation="acc",
        dmax=1,
        centralell=10,
    )
    errors = _errors(params)
    assert any("parity-even observable" in e for e in errors)


def test_parity_mixed_valid_combination_passes():
    params = _base_params(
        observables=["TT", "EE", "TE", "TB", "EB"],
        parity_mixed_blocks=True,
        covariance_approximation="acc",
        dmax=1,
        centralell=10,
    )
    errors = _errors(params)
    assert not any("parity_mixed_blocks" in e for e in errors)


# --------------------------------------------------------------------------- #
# CovKeys: blocks built from observables, not a Stokes alphabet
# --------------------------------------------------------------------------- #


def test_covkeys_observables_bb_does_not_create_tb_eb():
    keys = CovKeys([], ["090GHz"], observables=["TT", "EE", "TE", "BB"])
    stokekeys = {sk.stokekey() for sk in keys.spec_keys}
    assert stokekeys == {"TT", "EE", "TE", "BB"} or stokekeys == {
        "TT",
        "EE",
        "TE",
        "ET",
        "BB",
    }
    for cov_key in keys.keys():
        left, right = cov_key.left.stokekey(), cov_key.right.stokekey()
        assert left in stokekeys and right in stokekeys
        assert left not in ("TB", "EB") and right not in ("TB", "EB")


def test_covkeys_parity_mixed_blocks_default_excludes_cross_parity():
    keys = CovKeys([], ["090GHz"], observables=["TT", "EE", "TE", "TB", "EB"])
    for cov_key in keys.keys():
        left_odd = "B" in cov_key.left.stokekey() and cov_key.left.stokekey() != "BB"
        right_odd = "B" in cov_key.right.stokekey() and cov_key.right.stokekey() != "BB"
        assert left_odd == right_odd, cov_key


def test_covkeys_parity_mixed_blocks_true_includes_cross_parity():
    keys = CovKeys(
        [],
        ["090GHz"],
        observables=["TT", "TB"],
        parity_mixed_blocks=True,
    )
    stokekeys = {cov_key.stokekey() for cov_key in keys.keys()}
    assert "TTxTB" in stokekeys


def test_speckey_accepts_b():
    sk = SpecKey(("B", "B"), ("090GHz", "090GHz"))
    assert sk.stokekey() == "BB"


def test_kernel_stokekey_raises_for_b():
    sk = SpecKey(("B", "B"), ("090GHz", "090GHz"))
    with pytest.raises(ValueError, match="B-mode observable"):
        sk.kernel_stokekey()


def test_key_to_cross_kernel_raises_for_b():
    keys = CovKeys([], ["090GHz"], observables=["TT", "BB"])
    (cov_key,) = [k for k in keys.keys() if k.stokekey() == "TTxBB"]
    with pytest.raises(ValueError, match="does not support a B letter"):
        cov_key.key_to_cross_kernel()


# --------------------------------------------------------------------------- #
# required_kernel_pairs: counts against the derivation
# --------------------------------------------------------------------------- #


def _today_pairs():
    """Today's literal kernel pairs for stokes=['T', 'E'] (>=2 frequencies,
    the frequency-independent superset)."""
    pairs = set()
    for cov_key in CovKeys(["T", "E"], ["a", "b"]).keys():
        for contraction in cov_key.key_to_cross_kernel():
            a, b = (spec.kernel_stokekey() for spec in contraction)
            pairs.add(min((a, b), (b, a)))
    return pairs


def test_level1_equals_todays_pairs():
    level1 = required_kernel_pairs(["TT", "EE", "TE"])
    assert level1 == _today_pairs()


def test_level2_bb_pair_count():
    """16 four-spectrum blocks, one orientation (Sect. 9.6): 17 pairs."""
    assert len(required_kernel_pairs(["TT", "EE", "TE", "BB"])) == 17


def test_level3_parity_respecting_pair_count():
    """13 parity-respecting six-spectrum blocks, best orientation: 18 pairs."""
    observables = ["TT", "EE", "TE", "BB", "TB", "EB"]
    assert len(required_kernel_pairs(observables)) == 18


def test_level4_parity_mixed_pair_count():
    """All 21 blocks, best orientation: 31 pairs."""
    observables = ["TT", "EE", "TE", "BB", "TB", "EB"]
    assert len(required_kernel_pairs(observables, parity_mixed_blocks=True)) == 31


def test_nonzero_tb_eb_pair_count():
    """C^TB, C^EB != 0: 40 pairs, whether or not parity_mixed_blocks."""
    observables = ["TT", "EE", "TE", "BB", "TB", "EB"]
    assert len(required_kernel_pairs(observables, parity_odd_nonzero=True)) == 40
    assert (
        len(
            required_kernel_pairs(
                observables, parity_odd_nonzero=True, parity_mixed_blocks=True
            )
        )
        == 40
    )


def test_parity_odd_nonzero_irrelevant_without_tb_eb():
    """Sect. 9's C^TB, C^EB terms only enter once TB or EB is itself an
    observable; with neither, the flag changes nothing."""
    observables = ["TT", "EE", "TE", "BB"]
    assert required_kernel_pairs(observables) == required_kernel_pairs(
        observables, parity_odd_nonzero=True
    )


# --------------------------------------------------------------------------- #
# Spectra reading: TB/EB zero fallback, BB pix/noise, non-zero detection
# --------------------------------------------------------------------------- #


def test_tb_eb_default_to_zero_on_a_four_column_file():
    values = np.random.default_rng(0).uniform(0.1, 0.9, (10, 4))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = stokes_columns(values, ["TT", "TB", "EB"], "f.dat", "cmb_spectrum")
    np.testing.assert_array_equal(got["TB"], np.zeros(10))
    np.testing.assert_array_equal(got["EB"], np.zeros(10))
    assert any("treating C^TB as zero" in str(w.message) for w in caught)


def test_tb_eb_read_from_a_nine_column_file():
    values = np.zeros((10, 9))
    values[:, 4] = 1.0  # TB
    values[:, 5] = 2.0  # EB
    got = stokes_columns(values, ["TT", "TB", "EB"], "f.dat", "cmb_spectrum")
    np.testing.assert_array_equal(got["TB"], np.ones(10))
    np.testing.assert_array_equal(got["EB"], 2 * np.ones(10))


def test_bb_missing_from_four_column_file_still_raises():
    """Unlike TB/EB, BB is a real requested spectrum with no lenient
    fallback: missing it from a too-short file is still an error."""
    values = np.random.default_rng(0).uniform(0.1, 0.9, (10, 1))
    with pytest.raises(ValueError, match="cannot supply 'BB'"):
        stokes_columns(values, ["BB"], "f.dat", "cmb_spectrum")


def _write_cmb_spectrum_file(path, n_columns, nonzero_tb=False, nonzero_eb=False):
    """A tiny CAMB-style (4-column) or full (9-column) spectrum file."""
    lmax_file = 20
    ell = np.arange(lmax_file)
    cols = np.zeros((lmax_file, n_columns))
    cols[2:, 0] = 1.0  # TT
    if n_columns >= 4:
        cols[2:, 1] = 0.5  # EE
        cols[2:, 2] = 0.1  # BB
        cols[2:, 3] = 0.2  # TE
    if n_columns == 9:
        if nonzero_tb:
            cols[10, 4] = 1e-3  # TB, one multipole
        if nonzero_eb:
            cols[15, 5] = 2e-3  # EB, one multipole
    table = np.column_stack([ell, cols])
    np.savetxt(str(path), table)


def test_detect_parity_odd_nonzero_absent_columns(tmp_path):
    """TB/EB requested, but the file has no such columns at all (4-column,
    CAMB-style): False, not an error."""
    path = tmp_path / "cl_4col.dat"
    _write_cmb_spectrum_file(path, 4)
    assert detect_parity_odd_nonzero(str(path), None, ["090GHz090GHz"]) is False


def test_detect_parity_odd_nonzero_all_zero_columns(tmp_path):
    """The usual EB null test: TB/EB columns present (9-column) but all
    zero -- False."""
    path = tmp_path / "cl_9col_zero.dat"
    _write_cmb_spectrum_file(path, 9)
    assert detect_parity_odd_nonzero(str(path), None, ["090GHz090GHz"]) is False


def test_detect_parity_odd_nonzero_one_nonzero_value(tmp_path):
    """One non-zero TB value anywhere in the file (not just below some
    lmax) is enough: True."""
    path = tmp_path / "cl_9col_nonzero.dat"
    _write_cmb_spectrum_file(path, 9, nonzero_tb=True)
    assert detect_parity_odd_nonzero(str(path), None, ["090GHz090GHz"]) is True


def test_detect_parity_odd_nonzero_checks_foregrounds_too(tmp_path):
    """A non-zero EB in the foregrounds file (one per frequency pair) is
    also detected, with the cmb_spectrum itself carrying none."""
    cmb = tmp_path / "cl.dat"
    _write_cmb_spectrum_file(cmb, 9)
    fg_template = str(tmp_path / "fg_{}.dat")
    _write_cmb_spectrum_file(fg_template.format("090GHz090GHz"), 9, nonzero_eb=True)
    assert detect_parity_odd_nonzero(str(cmb), fg_template, ["090GHz090GHz"]) is True


def test_detect_parity_odd_nonzero_constant_cmb_spectrum():
    """A constant cmb_spectrum applies to every requested Stokes key alike,
    including TB/EB."""
    assert detect_parity_odd_nonzero(1e-3, None, []) is True
    assert detect_parity_odd_nonzero(0.0, None, []) is False


def test_detect_parity_odd_nonzero_dict_cmb_spectrum_without_tb_eb():
    """A per-Stokes-key dict with no TB/EB entry is zero by construction."""
    assert detect_parity_odd_nonzero({"TT": 1e-3, "EE": 5e-4}, None, []) is False
    assert detect_parity_odd_nonzero({"TT": 1e-3, "TB": 2e-5}, None, []) is True


def test_loader_parity_odd_nonzero_reads_the_configured_file(tmp_path):
    path = tmp_path / "cl_9col_nonzero.dat"
    _write_cmb_spectrum_file(path, 9, nonzero_tb=True)
    loader = SpectraLoader.__new__(SpectraLoader)
    loader.combined_stokes = ["TT", "TB", "EB"]
    loader.combined_frequencies = ["090GHz090GHz"]
    loader.config = type("Cfg", (), {"cmb_spectrum": str(path), "foregrounds": None})()
    assert loader.parity_odd_nonzero is True


def test_loader_parity_odd_nonzero_false_when_not_requested():
    """TB/EB absent from combined_stokes: not read at all, so nothing to
    detect. No config needed
    -- the short-circuit must happen before self.config is touched."""
    loader = SpectraLoader.__new__(SpectraLoader)
    loader.combined_stokes = ["TT", "EE", "TE"]
    assert loader.parity_odd_nonzero is False


# --------------------------------------------------------------------------- #
# Precompute pair selection: EB null test (absent/zero) -> 18, non-zero -> 40
# --------------------------------------------------------------------------- #


def _acc_bmode_params(tmp_path, cmb_spectrum_path):
    return {
        "cov_path": str(tmp_path / "out"),
        "cov_name": "cov.dat",
        "frequencies": ["090GHz"],
        "observables": ["TT", "EE", "TE", "BB", "TB", "EB"],
        "lmax": 5,
        "lmin": 2,
        "bins": [[2, 5, 1]],
        "mask_name": "baseline_mask.fits",
        "mask_path": DATA,
        "covariance_approximation": "acc",
        "dmax": 1,
        "centralell": 10,
        "cmb_spectrum": cmb_spectrum_path,
    }


def _precompute_plan(tmp_path, params, name):
    path = tmp_path / name
    path.write_text(yaml.dump(params))
    generator = CovarianceMatrixGenerator(str(path))
    return generator.precompute_acc_kernels(dryrun=True)


def test_precompute_pairs_absent_tb_eb_columns(tmp_path):
    cmb = tmp_path / "cl_4col.dat"
    _write_cmb_spectrum_file(cmb, 4)
    plan = _precompute_plan(
        tmp_path, _acc_bmode_params(tmp_path, str(cmb)), "p_absent.yml"
    )
    assert len(plan["pairs"]) == 18


def test_precompute_pairs_all_zero_tb_eb(tmp_path):
    cmb = tmp_path / "cl_9col_zero.dat"
    _write_cmb_spectrum_file(cmb, 9)
    plan = _precompute_plan(
        tmp_path, _acc_bmode_params(tmp_path, str(cmb)), "p_zero.yml"
    )
    assert len(plan["pairs"]) == 18


def test_precompute_pairs_nonzero_tb(tmp_path):
    cmb = tmp_path / "cl_9col_nonzero.dat"
    _write_cmb_spectrum_file(cmb, 9, nonzero_tb=True)
    plan = _precompute_plan(
        tmp_path, _acc_bmode_params(tmp_path, str(cmb)), "p_nonzero.yml"
    )
    assert len(plan["pairs"]) == 40


# --------------------------------------------------------------------------- #
# The recompute guard: non-zero TB/EB against an 18-pair cache must refuse
# --------------------------------------------------------------------------- #


def test_recompute_error_against_18_pair_cache():
    """A cache built for the 18-pair (EB-null-test) set does not cover the
    40-pair (non-zero C^TB/C^EB) set: require_acc_cache_pairs must raise the
    existing "recompute" error naming the missing pairs, not silently give a
    partial answer. The 18-pair set itself, which the cache was built for,
    still passes."""
    import shutil
    import tempfile

    observables = ["TT", "EE", "TE", "BB", "TB", "EB"]
    pairs_18 = sorted(required_kernel_pairs(observables))
    pairs_40 = sorted(required_kernel_pairs(observables, parity_odd_nonzero=True))
    assert len(pairs_18) == 18 and len(pairs_40) == 40

    kernel_dir = tempfile.mkdtemp()
    try:
        precompute_acc_kernels(
            "baseline_mask.fits",
            kernel_dir,
            centralell=16,
            dmax=1,
            mask_path=DATA,
            nside=16,
            grid="healpix",
            pairs=pairs_18,
        )
        require_acc_cache_pairs(kernel_dir, 16, 1, pairs_18)  # does not raise
        missing = sorted(set(pairs_40) - set(pairs_18))
        assert missing
        with pytest.raises(OSError, match="not found") as excinfo:
            require_acc_cache_pairs(kernel_dir, 16, 1, pairs_40)
        # The missing pair the error names is genuinely one of the 22 pairs
        # the 40-pair set adds over the 18-pair one (not e.g. a mistaken
        # complaint about a pair both sets share).
        assert any(f"{a}x{b}" in str(excinfo.value) for a, b in missing)
    finally:
        shutil.rmtree(kernel_dir, ignore_errors=True)


def test_bb_pixel_window_matches_ee():
    loader = SpectraLoader.__new__(SpectraLoader)
    loader.config = type("Cfg", (), {"pixwin": None})()
    loader._spectra_lmax = 5
    loader.logger = __import__("logging").getLogger(__name__)
    loader._load_pixel_window()
    np.testing.assert_array_equal(loader.pix["BB"], loader.pix["EE"])
    np.testing.assert_array_equal(loader.pix["TB"], loader.pix["TE"])
    np.testing.assert_array_equal(loader.pix["EB"], loader.pix["EE"])


# --------------------------------------------------------------------------- #
# Assembly gate: NotImplementedError for any block involving B
# --------------------------------------------------------------------------- #


def test_b_block_is_refused_for_nka():
    """NKA and INKA collapse EE and BB, so they cannot compute a B block:
    this is refused before anything is computed. The ACC assembly itself is
    tested in tests/test_acc_bmode_assembly.py."""
    config = CovarianceConfig(
        method=CovarianceMethod.NKA,
        lmax=10,
        lmin=2,
    )
    cov = Cov("baseline_mask.fits", config=config, mask_path=DATA)
    covariance_keys = CovKeys([], ["090GHz"], observables=["TT", "BB"])
    with pytest.raises(ValueError, match="needs the ACC method"):
        cov.compute_covariance_matrix([2, 10], covariance_keys, cl={})


def test_t_e_only_block_does_not_hit_the_b_gate():
    """A T/E-only covariance_keys never reaches the B check (regression
    guard: it must not raise NotImplementedError for a default run)."""
    config = CovarianceConfig(
        method=CovarianceMethod.NKA,
        lmax=10,
        lmin=2,
    )
    cov = Cov("baseline_mask.fits", config=config, mask_path=DATA)
    covariance_keys = CovKeys(["T"], ["090GHz"])
    # cl={} will fail input validation instead (a different, expected error),
    # proving the B gate is not what stops this run.
    with pytest.raises(ValueError, match="Missing power spectrum"):
        cov.compute_covariance_matrix([2, 10], covariance_keys, cl={})
