"""
The per-run polarised ACC error budget (``approximations/acc_budget.py``).

It reports two things and changes nothing:

* the E->B leakage of the kernel set in use, ``lambda = sum Theta^{TT x BB} /
  sum Theta^{TT x EE}`` at ``(l*, l*)``, and ``(n_E / 2) lambda`` per block
  with ``n_E`` the number of polarised legs among the block's four (derived
  from the ``CovKey``, not tabulated). The ``BB`` kernels it reads are written
  by every default precompute and read by nothing else in the package, so the
  budget costs one ``np.load`` and no new computation -- pinned below;
* the Eq. 33 translation error, which is *not* bounded: only ``|l - l*|`` at
  the band edges, with the measured survey-mask calibration quoted as text.

The golden ``tests/data/baseline_acc_coupling.npz`` was produced with exactly
the settings of the fixture here (baseline mask, ``centralell = 16``,
``nside = 16``, HEALPix grid; ``tests/data/regenerate_baselines.py``), so the
hand computation below is against a file, independent of the loader.
"""

import glob
import json
import os
import textwrap
import warnings

import numpy as np
import pytest

healpy = pytest.importorskip("healpy")

from cmbcov import CovarianceMatrixGenerator  # noqa: E402
from cmbcov.approximations import (  # noqa: E402
    StrategyFactory,
    acc_budget,
)
from cmbcov.approximations.acc import (  # noqa: E402
    _DEFAULT_KERNEL_STOKES,
    COUPLING_SPECTRA,
    precompute_acc_kernels,
)
from cmbcov.covariance import (  # noqa: E402
    Cov,
    CovarianceConfig,
    CovarianceMethod,
)
from cmbcov.keys import CovKey, CovKeys  # noqa: E402

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
MASK = "baseline_mask.fits"
NSIDE = 16
ELL = 16
DMAX = 2
LMAX = 48
FREQ = "090GHz"


def _cov(kernel_dir, lmin=2):
    config = CovarianceConfig(
        method=CovarianceMethod.ACC,
        lmax=LMAX,
        lmin=lmin,
        dmax=DMAX,
        centralell=ELL,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # low centralell
        return Cov(MASK, config=config, mask_path=DATA, save_dir=str(kernel_dir))


@pytest.fixture(scope="module")
def acc_cache(tmp_path_factory):
    """A kernel cache with all twenty-five pairs, as the default precompute writes."""
    directory = tmp_path_factory.mktemp("acc_budget_cache")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        precompute_acc_kernels(
            MASK,
            str(directory),
            centralell=ELL,
            dmax=DMAX,
            mask_path=DATA,
            nside=NSIDE,
            grid="healpix",
        )
    return directory


@pytest.fixture(scope="module")
def strategy(acc_cache):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return StrategyFactory.create_strategy(_cov(acc_cache))


def _keys(stokes=("T", "E"), freqs=(FREQ,)):
    return CovKeys(list(stokes), list(freqs), exclude_asymmetric_stokes=False)


def _lambda_from_golden():
    """lambda straight out of the recorded kernel file, no package code."""
    with np.load(os.path.join(DATA, "baseline_acc_coupling.npz")) as golden:
        return float(golden[f"{ELL}|TTxBB"].sum() / golden[f"{ELL}|TTxEE"].sum())


def _lambda_from_cache(acc_cache):
    """lambda from the .npy files of the cache, read without the loader."""
    directory = os.path.join(str(acc_cache), "covariance_coupling")
    sums = {}
    for pair in ("TTxLL", "TTxDD"):
        sums[pair] = float(
            np.load(os.path.join(directory, f"{pair}_{ELL}x{ELL}.npy")).sum()
        )
    return sums["TTxLL"] / sums["TTxDD"]


# ------------------------------------------------------- the leakage itself


def test_leakage_matches_a_direct_computation_from_the_kernel_files(
    strategy, acc_cache
):
    budget = strategy.error_budget(_keys())
    reported = budget["leakage"]["lambda"]
    from_files = _lambda_from_cache(acc_cache)
    from_golden = _lambda_from_golden()
    # Same floats, not merely close: the budget sums the very arrays the
    # cache holds.
    assert reported == from_files
    assert reported == pytest.approx(from_golden, rel=1e-10, abs=0.0)
    # and the sums it reports are those sums
    assert budget["leakage"]["sum_TTxBB"] / budget["leakage"]["sum_TTxEE"] == reported
    assert budget["leakage"]["kernel_pair"] == (ELL, ELL)


def test_the_hand_computed_numbers_on_the_baseline_mask(strategy):
    """
    Every reported number, recomputed here from the golden kernel file and
    the block's Stokes legs.
    """
    lam = _lambda_from_golden()
    budget = strategy.error_budget(_keys())
    assert budget["leakage"]["lambda"] == pytest.approx(lam, rel=1e-10, abs=0.0)
    expected = {
        "TTxTE": 0.5 * lam,  # legs T,T,T,E -> n_E = 1
        "TTxEE": 1.0 * lam,  # T,T,E,E     -> 2
        "TExTE": 1.0 * lam,  # T,E,T,E     -> 2
        "TExEE": 1.5 * lam,  # T,E,E,E     -> 3
        "EExEE": 2.0 * lam,  # E,E,E,E     -> 4
    }
    got = {
        entry["stokes"]: entry["leakage_bias"] for entry in budget["blocks"].values()
    }
    assert set(got) == set(expected)
    for name, value in expected.items():
        assert got[name] == pytest.approx(value, rel=1e-10, abs=0.0)
    # TTxTT carries no polarised leg and is not listed at all.
    assert "TTxTT" not in got


def test_n_e_is_read_off_the_key():
    assert acc_budget.polarised_leg_count(CovKey(("T",) * 4, (FREQ,) * 4)) == 0
    assert (
        acc_budget.polarised_leg_count(CovKey(("T", "T", "T", "E"), (FREQ,) * 4)) == 1
    )
    assert (
        acc_budget.polarised_leg_count(CovKey(("T", "E", "T", "E"), (FREQ,) * 4)) == 2
    )
    assert acc_budget.polarised_leg_count(CovKey(("E",) * 4, (FREQ,) * 4)) == 4
    # B counts with E: it is the other half of the same completeness
    # relation. `SpecKey` still refuses B, so this is checked on a key-shaped
    # object -- the count is ready for B when CovKeys admits it.
    from types import SimpleNamespace

    assert (
        acc_budget.polarised_leg_count(SimpleNamespace(stoke=("B", "B", "T", "T"))) == 2
    )


def test_every_polarised_block_of_a_multi_frequency_run_is_listed(strategy):
    keys = _keys(freqs=(FREQ, "150GHz"))
    budget = strategy.error_budget(keys)
    listed = set(budget["blocks"])
    expected = {
        f"{key.stokekey()} {key.freqkey()}"
        for key in keys.keys()
        if acc_budget.polarised_leg_count(key)
    }
    assert listed == expected
    assert len(listed) > 5  # frequencies multiply the blocks, not the bias
    lam = budget["leakage"]["lambda"]
    for entry in budget["blocks"].values():
        assert entry["leakage_bias"] == pytest.approx(0.5 * entry["n_E"] * lam)


# ----------------------------------------------------------- when it is absent


def test_a_tt_only_run_has_no_budget(strategy):
    assert strategy.error_budget(_keys(stokes=("T",))) is None


def test_nka_and_inka_have_no_budget(tmp_path):
    for method in (CovarianceMethod.NKA, CovarianceMethod.INKA):
        cov = Cov(
            MASK,
            config=CovarianceConfig(method=method, lmax=LMAX),
            mask_path=DATA,
            save_dir=str(tmp_path),
        )
        assert cov.error_budget(_keys()) is None


def test_missing_bb_kernels_are_reported_not_guessed(tmp_path):
    """
    A cache built with an explicit ``spectra`` list without BB cannot bound
    the leakage. The budget says so, naming what to do; it does not fall back
    to EE or to a number from the docs.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        precompute_acc_kernels(
            MASK,
            str(tmp_path),
            centralell=ELL,
            dmax=DMAX,
            mask_path=DATA,
            nside=NSIDE,
            grid="healpix",
            spectra=("TT", "EE", "TE", "ET"),
        )
        budget = StrategyFactory.create_strategy(_cov(tmp_path)).error_budget(_keys())
    assert budget is not None
    assert budget["leakage"]["lambda"] is None
    assert "acc_precompute.spectra" in budget["leakage"]["unavailable"]
    assert all(entry["leakage_bias"] is None for entry in budget["blocks"].values())
    text = acc_budget.format_error_budget(budget)
    assert "NOT AVAILABLE" in text


# ------------------------------------- it costs nothing and changes nothing


def test_bb_kernels_are_in_the_cache_and_read_by_nothing_else(strategy, acc_cache):
    """
    The budget is free: LL (old name BB) is one of the five COUPLING_SPECTRA
    every default precompute writes, and no covariance path loads it
    (``_DEFAULT_KERNEL_STOKES``), so the file is already on disk and unread.
    """
    assert "LL" in COUPLING_SPECTRA
    assert "LL" not in _DEFAULT_KERNEL_STOKES
    directory = os.path.join(str(acc_cache), "covariance_coupling")
    assert os.path.exists(os.path.join(directory, f"TTxLL_{ELL}x{ELL}.npy"))

    before = {
        p: os.stat(p).st_mtime_ns for p in glob.glob(os.path.join(directory, "*"))
    }
    strategy.error_budget(_keys())
    after = {p: os.stat(p).st_mtime_ns for p in glob.glob(os.path.join(directory, "*"))}
    assert after == before  # nothing written, nothing recomputed


def test_the_budget_does_not_change_the_covariance(strategy):
    cl = {FREQ + FREQ: {key: _spectrum() for key in ("TT", "EE", "TE", "ET")}}
    key = CovKey(("E",) * 4, (FREQ,) * 4)
    before = strategy.compute_covariance_term(key, cl)
    strategy.error_budget(_keys())
    after = strategy.compute_covariance_term(key, cl)
    np.testing.assert_array_equal(after, before)


def _spectrum():
    size = LMAX + 2 * NSIDE - 1 - ELL
    ell = np.arange(size)
    cl = np.zeros(size)
    cl[2:] = 1e-3 / (ell[2:] * (ell[2:] + 1)) ** 0.9
    return cl


# ------------------------------------------------------ the translation part


def test_translation_is_reported_as_unbounded(strategy):
    budget = strategy.error_budget(_keys(), band_edges=[2, 20, 48])
    translation = budget["translation"]
    assert translation["bounded"] is False
    assert translation["proxy"] is None
    assert translation["band_edges"] == [2, 48]
    assert translation["max_offset_from_centralell"] == 32  # |48 - 16|
    assert "not bounded" in translation["note"]
    # the measured survey numbers are quoted, not extrapolated from
    assert translation["measured_reference"][500] == -5.6e-3


def test_band_edges_default_to_the_reported_range(strategy):
    budget = strategy.error_budget(_keys())
    assert budget["translation"]["band_edges"] == [2, LMAX - 1]
    assert budget["translation"]["max_offset_from_centralell"] == LMAX - 1 - ELL


# ------------------------------------------------------------- the run output


def _params_file(tmp_path, method="acc"):
    template = open(os.path.join(DATA, "baseline_params.yml")).read()
    text = (
        template.replace("PLACEHOLDER_OUT", str(tmp_path / "out"))
        .replace("PLACEHOLDER_DATA", DATA)
        .replace("lmax: 60", f"lmax: {LMAX}")
        .replace("[20, 60, 10]", f"[20, {LMAX}, 10]")
    )
    if method == "acc":
        text = text.replace(
            "covariance_approximation: nka",
            f"covariance_approximation: acc\ndmax: {DMAX}\ncentralell: {ELL}",
        )
        text += textwrap.dedent(
            """
            acc_precompute:
              nside: 16
              grid: healpix
            """
        )
    path = tmp_path / "params.yml"
    path.write_text(text)
    return str(path)


@pytest.fixture(scope="module")
def acc_run(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("acc_budget_run")
    params = _params_file(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        generator = CovarianceMatrixGenerator(params)
        generator.precompute_acc_kernels()
        CovarianceMatrixGenerator(params).run_full_analysis()
    return tmp_path / "out" / "v0"


def test_the_report_is_written_beside_the_covariance(acc_run):
    path = acc_run / "error_budget.txt"
    assert path.exists()
    text = path.read_text()
    lam = _lambda_from_golden()
    assert f"lambda = {lam:.3e}" in text
    assert "EExEE" in text
    assert "NOT BOUNDED" in text
    # the covariance itself is there and untouched by the report
    assert (acc_run / "covariance_matrix.dat").exists()


def test_the_run_logs_the_headline_numbers(tmp_path, caplog):
    import logging

    params = _params_file(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        generator = CovarianceMatrixGenerator(params)
        generator.precompute_acc_kernels()
        with caplog.at_level(logging.INFO):
            CovarianceMatrixGenerator(params).run_full_analysis()
    messages = [record.message for record in caplog.records]
    budget_lines = [m for m in messages if "ACC error budget" in m]
    assert len(budget_lines) == 1
    assert "E->B leakage lambda" in budget_lines[0]
    assert "NOT bounded" in budget_lines[0]


def test_a_non_acc_run_writes_no_report(tmp_path):
    params = _params_file(tmp_path, method="nka")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        CovarianceMatrixGenerator(params).run_full_analysis()
    version = tmp_path / "out" / "v0"
    assert (version / "covariance_matrix.dat").exists()
    assert not (version / "error_budget.txt").exists()


def test_the_report_is_json_serialisable(strategy):
    """The dict is a report: it must survive being written out as it is."""
    budget = strategy.error_budget(_keys())
    round_trip = json.loads(json.dumps(budget, default=list))
    assert round_trip["leakage"]["lambda"] == budget["leakage"]["lambda"]
