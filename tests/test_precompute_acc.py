"""
The standalone ACC precompute and its parameter-file driver.

The intended ACC workflow is: the mask is fixed, the coupling kernels are
precomputed once, and the covariance is recomputed many times. This file pins
the three pieces that make that work without a ``Cov`` in the precompute:

- :func:`~cmbcov.approximations.acc.precompute_acc_kernels`
  writes exactly the kernel set the recompute loop
  (``ACCStrategy.compute_covariance_term``) loads for a ``centralell`` /
  ``dmax`` -- checked by recording the loader's requests, not by restating the
  formula -- with a manifest a later ``Cov`` validates against;
- the ``acc_precompute`` block of the parameter file is validated;
- ``precompute-acc`` on a parameter file writes the kernels where every later
  ``compute-covariance`` run of the same file looks for them, which is *not*
  the per-run versioned ``save_dir``.

Everything runs at ``nside = 16``, ``centralell = 16`` on the test mask
(``tests/data/baseline_mask.fits``, nside 32), in well under a second per test.
"""

import glob
import json
import os
import textwrap
import warnings

import numpy as np
import pytest

from cmbcov import CovarianceMatrixGenerator, ParameterManager
from cmbcov.approximations import StrategyFactory
from cmbcov.approximations.acc import (
    COUPLING_SPECTRA,
    coupling_ellprange,
    precompute_acc_kernels,
)
from cmbcov.covariance import Cov, CovarianceConfig, CovarianceMethod
from cmbcov.generator.parameter_validation import (
    AccPrecomputeConfig,
    ParameterValidator,
)
from cmbcov.keys import CovKey, CovKeys
from cmbcov.mask import MaskWlm
from cmbcov.scripts import precompute_acc
from cmbcov.sht import DEFAULT_MAP2ALM_ITER

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
MASK = "baseline_mask.fits"
NSIDE = 16
ELL = 16
DMAX = 3
LMAX = 48
LW = 10

#: ACC reads the spectra to lmax_int = lmax + max(0, S - 1 - centralell),
#: S = 2 NSIDE the kernel size (acc_window_pad).
LMAX_INT = LMAX + 2 * NSIDE - 1 - ELL
FREQ = "090GHz"


def _cov(kernel_dir, dmax=DMAX, centralell=ELL):
    config = CovarianceConfig(
        method=CovarianceMethod.ACC, lmax=LMAX, dmax=dmax, centralell=centralell
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # low centralell
        return Cov(MASK, config=config, mask_path=DATA, save_dir=str(kernel_dir))


def _cl(size):
    ell = np.arange(size)
    cl = np.zeros(size)
    cl[2:] = 1e-3 / (ell[2:] * (ell[2:] + 1)) ** 0.9
    return cl


def _precompute(save_dir, **kwargs):
    options = {"centralell": ELL, "dmax": DMAX, "mask_path": DATA, "nside": NSIDE}
    options.update(kwargs)
    return precompute_acc_kernels(MASK, save_dir, **options)


def _no_unverifiable_warning(caught):
    messages = [str(w.message) for w in caught]
    assert not [m for m in messages if "cannot be checked" in m], messages


# --------------------------------------------------------------------------- #
# precompute_acc_kernels: the kernel set, the manifest, the validated reload
# --------------------------------------------------------------------------- #
def test_coupling_ellprange_is_the_diagonal_offsets():
    assert coupling_ellprange(16, 1) == [16]
    assert coupling_ellprange(250, 4) == [250, 251, 252, 253]


def test_default_grid_is_gl(tmp_path):
    """The user-facing default grid is gl (exact, and already what
    production runs use): an unspecified grid must still produce a valid,
    loadable cache, recorded as 'gl'."""
    assert (
        precompute_acc_kernels(
            MASK, str(tmp_path), centralell=ELL, dmax=1, mask_path=DATA, nside=NSIDE
        )
        is None
    )
    manifest = json.load(
        open(tmp_path / "covariance_coupling" / f"manifest_{ELL}x{ELL}.json")
    )
    assert manifest["grid"] == "gl"
    assert manifest["lw"] == 3 * NSIDE - 1
    assert "map2alm_iter" not in manifest

    cov = _cov(tmp_path, dmax=1)
    loaded = cov.get_acc_coupling_kernels(
        ELL, ELL, COUPLING_SPECTRA, [("TT", "TT")], pairs=[("TT", "TT")]
    )
    assert np.isfinite(loaded[("TT", "TT")]).all()


def test_yaml_driven_default_grid_is_gl(tmp_path, capsys):
    """The ``acc_precompute`` block's default follows the same switch: a
    parameter file with no ``acc_precompute`` block precomputes on 'gl'."""
    params = _params_file(tmp_path, block="")
    generator = CovarianceMatrixGenerator(params)
    plan = generator.precompute_acc_kernels(dryrun=True)
    assert plan["grid"] == "gl"

    assert generator.precompute_acc_kernels() is not None
    kernel_dir = tmp_path / "out" / "covariance_coupling"
    manifest = json.load(open(kernel_dir / f"manifest_{ELL}x{ELL}.json"))
    assert manifest["grid"] == "gl"


@pytest.mark.parametrize("grid", ["healpix", "gl"])
def test_writes_exactly_what_the_recompute_loop_loads(tmp_path, grid, monkeypatch):
    extra = {"grid": grid, "lw": LW} if grid == "gl" else {"grid": grid}
    assert _precompute(str(tmp_path), **extra) is None

    manifests = sorted(glob.glob(str(tmp_path / "covariance_coupling" / "*.json")))
    written = {
        tuple(int(x) for x in os.path.basename(p)[9:-5].split("x")) for p in manifests
    }
    kernels = glob.glob(str(tmp_path / "covariance_coupling" / "*.npy"))
    assert len(kernels) == len(written) * len(COUPLING_SPECTRA) ** 2

    cov = _cov(tmp_path)
    requested = set()
    loader = cov.get_acc_coupling_kernels

    def recording(ell, ell_prime, *args, **kwargs):
        requested.add((ell, ell_prime))
        return loader(ell, ell_prime, *args, **kwargs)

    monkeypatch.setattr(cov, "get_acc_coupling_kernels", recording)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        strategy = StrategyFactory.create_strategy(cov)
        for stokes in (("T", "T", "T", "T"), ("T", "E", "T", "E"), ("E",) * 4):
            sigma = strategy.compute_covariance_term(
                CovKey(stokes, (FREQ,) * 4),
                {
                    FREQ
                    + FREQ: {
                        "TT": _cl(LMAX_INT),
                        "EE": _cl(LMAX_INT),
                        "TE": _cl(LMAX_INT),
                    }
                },
            )
            assert np.isfinite(sigma).all() and np.abs(sigma).max() > 0
    _no_unverifiable_warning(caught)

    # The loader asked for exactly the pairs that were written: none missing
    # (the run succeeded) and none computed for nothing.
    assert requested == written == {(ELL, p) for p in coupling_ellprange(ELL, DMAX)}

    digest = MaskWlm(MASK, load_path=DATA).mask_digest
    for path in manifests:
        manifest = json.load(open(path))
        assert manifest["mask_digest"] == digest
        assert manifest["centralell"] == ELL
        assert manifest["grid"] == grid
        assert manifest["nside"] == NSIDE
        assert sorted(manifest["spectra"]) == sorted(COUPLING_SPECTRA)
        if grid == "healpix":
            assert manifest["map2alm_iter"] == DEFAULT_MAP2ALM_ITER
            assert manifest["lw"] is None
        else:
            assert "map2alm_iter" not in manifest
            assert manifest["lw"] == LW


def test_written_kernels_are_the_dryrun_kernels(tmp_path):
    """The disk round trip changes no bit, and dryrun writes nothing."""
    dry = _precompute(str(tmp_path / "dry"), dmax=2, grid="gl", lw=LW, dryrun=True)
    assert not os.path.exists(tmp_path / "dry")
    assert sorted(dry) == [ELL, ELL + 1]

    _precompute(str(tmp_path / "wet"), dmax=2, grid="gl", lw=LW)
    cov = _cov(tmp_path / "wet", dmax=2)
    pairs = [(a, b) for a in COUPLING_SPECTRA for b in COUPLING_SPECTRA]
    for ellp, inner in dry.items():
        loaded = cov.get_acc_coupling_kernels(
            ELL, ellp, COUPLING_SPECTRA, pairs, pairs=pairs
        )
        for (a, b), kernel in loaded.items():
            assert np.array_equal(kernel, inner[f"{a}x{b}"]), (ellp, a, b)


def test_a_run_with_a_different_mask_refuses_the_cache(tmp_path):
    """The realistic drift: the mask in the file changes, the directory does not."""
    import healpy as hp

    _precompute(str(tmp_path / "kernels"), dmax=1, grid="healpix", spectra=("TT",))
    other = hp.read_map(os.path.join(DATA, MASK))
    other[: other.size // 2] = 0.0
    (tmp_path / "other").mkdir()
    hp.write_map(str(tmp_path / "other" / MASK), other)

    config = CovarianceConfig(
        method=CovarianceMethod.ACC, lmax=LMAX, dmax=1, centralell=ELL
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cov = Cov(
            MASK,
            config=config,
            mask_path=str(tmp_path / "other"),
            save_dir=str(tmp_path / "run"),
            acc_kernel_dir=str(tmp_path / "kernels"),
        )
    with pytest.raises(ValueError, match="mask_digest"):
        cov.get_acc_coupling_kernels(
            ELL, ELL, COUPLING_SPECTRA, [("TT", "TT")], pairs=[("TT", "TT")]
        )


def test_a_larger_dmax_run_does_not_find_its_kernels(tmp_path):
    _precompute(str(tmp_path), dmax=2, grid="healpix", spectra=("TT",))
    strategy = StrategyFactory.create_strategy(_cov(tmp_path, dmax=3))
    with pytest.raises(OSError, match="16x18"):
        strategy.compute_covariance_term(
            CovKey(("T",) * 4, (FREQ,) * 4), {FREQ + FREQ: {"TT": _cl(LMAX_INT)}}
        )


def test_in_process_rewrite_is_not_shadowed_by_the_cov_memo(tmp_path):
    """
    ``Cov`` memoises kernel loads. A precompute has no reference to that
    memo, so a rewrite by it in the same process must still be seen.
    """
    pair = [("TT", "TT")]
    _precompute(str(tmp_path), dmax=1, grid="gl", lw=LW, spectra=("TT",))
    cov = _cov(tmp_path, dmax=1)
    first = cov.get_acc_coupling_kernels(ELL, ELL, COUPLING_SPECTRA, pair, pairs=pair)
    again = cov.get_acc_coupling_kernels(ELL, ELL, COUPLING_SPECTRA, pair, pairs=pair)
    assert again[pair[0]] is first[pair[0]]  # served from the memo

    path = str(tmp_path / "covariance_coupling" / f"TTxTT_{ELL}x{ELL}.npy")
    stat = os.stat(path)
    _precompute(str(tmp_path), dmax=1, grid="gl", lw=LW - 2, spectra=("TT",))
    # Restore the old mtime, so only the in-process write counter can tell.
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    expected = _precompute(
        None, dmax=1, grid="gl", lw=LW - 2, spectra=("TT",), dryrun=True
    )[ELL]["TTxTT"]
    after = cov.get_acc_coupling_kernels(ELL, ELL, COUPLING_SPECTRA, pair, pairs=pair)
    assert not np.array_equal(first[pair[0]], expected)
    assert np.array_equal(after[pair[0]], expected)


def test_out_of_process_rewrite_is_not_shadowed_by_the_cov_memo(tmp_path):
    """A file replaced behind the package's back (new mtime) is reloaded."""
    pair = [("TT", "TT")]
    _precompute(str(tmp_path), dmax=1, grid="healpix", spectra=("TT",))
    cov = _cov(tmp_path, dmax=1)
    first = cov.get_acc_coupling_kernels(ELL, ELL, COUPLING_SPECTRA, pair, pairs=pair)
    path = str(tmp_path / "covariance_coupling" / f"TTxTT_{ELL}x{ELL}.npy")
    replacement = np.asarray(first[pair[0]]) * 2.0
    stat = os.stat(path)
    np.save(path, replacement)  # not through write_coupling_kernels
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    after = cov.get_acc_coupling_kernels(ELL, ELL, COUPLING_SPECTRA, pair, pairs=pair)
    assert np.array_equal(after[pair[0]], replacement)


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({}, ValueError),  # neither dmax nor ellprange
        ({"dmax": 2, "ellprange": [16, 17]}, ValueError),  # both
        ({"dmax": 0}, ValueError),
        ({"dmax": True}, TypeError),
        ({"dmax": 2.0}, TypeError),
        ({"dmax": 2, "centralell": -1}, ValueError),
        ({"ellprange": 16}, TypeError),
    ],
)
def test_bad_arguments_are_rejected_before_any_work(kwargs, error):
    options = {"centralell": ELL, "mask_path": DATA, "dryrun": True}
    options.update(kwargs)
    with pytest.raises(error):
        precompute_acc_kernels(MASK, None, **options)


def test_save_dir_is_required_unless_dryrun():
    with pytest.raises(ValueError, match="save_dir"):
        precompute_acc_kernels(MASK, None, centralell=ELL, dmax=1, mask_path=DATA)


def test_ellprange_override_matches_dmax(tmp_path):
    by_dmax = _precompute(None, dmax=2, grid="healpix", spectra=("TT",), dryrun=True)
    by_range = precompute_acc_kernels(
        MaskWlm(MASK, load_path=DATA),
        None,
        centralell=ELL,
        ellprange=[ELL, ELL + 1],
        nside=NSIDE,
        grid="healpix",
        spectra=("TT",),
        dryrun=True,
    )
    for ellp in (ELL, ELL + 1):
        assert np.array_equal(by_dmax[ellp]["TTxTT"], by_range[ellp]["TTxTT"])


# --------------------------------------------------------------------------- #
# The acc_precompute block of the parameter file
# --------------------------------------------------------------------------- #
BASE = {
    "cov_path": "./out",
    "cov_name": "covariance_matrix.dat",
    "frequencies": ["090GHz"],
    "stokes": ["T", "E"],
    "lmax": 60,
    "lmin": 2,
    "bins": [[2, 60, 10]],
    "mask_name": MASK,
    "mask_path": DATA,
    "covariance_approximation": "acc",
    "dmax": 3,
    "centralell": 16,
}


def _validate(block, **overrides):
    params = dict(BASE, **overrides)
    if block is not ...:
        params["acc_precompute"] = block
    validator = ParameterValidator()
    ok = validator.validate(params)
    return ok, validator.errors, validator.warnings


@pytest.mark.parametrize(
    "block",
    [
        ...,  # absent
        None,  # `acc_precompute:` with an empty body: all defaults
        {},
        {"grid": "healpix", "nside": 256, "max_memory_gb": 6},
        {"grid": "gl", "lw": 512, "nside": 256, "max_memory_gb": 0.5},
        {"grid": "gl", "nside": 100},  # GL does not need a power of two
        {"spectra": ["TT", "EE", "TE", "ET"]},
        {"spectra": list(COUPLING_SPECTRA)},
    ],
)
def test_good_blocks_validate_cleanly(block):
    ok, errors, warns = _validate(block)
    assert ok, errors
    assert not [w for w in warns if "acc_precompute" in w], warns


def test_tt_only_spectra_are_fine_for_a_temperature_only_run():
    ok, errors, _ = _validate({"spectra": ["TT"]}, stokes=["T"])
    assert ok, errors


@pytest.mark.parametrize(
    "block, fragment",
    [
        ([1, 2], "must be a mapping"),
        ("gl", "must be a mapping"),
        ({"grids": "gl"}, "Unknown key"),
        ({"save_dir": "/tmp"}, "Unknown key"),
        ({"grid": "GL"}, "grid"),
        ({"grid": "gauss"}, "grid"),
        ({"grid": "healpix", "lw": 512}, "only used with grid 'gl'"),
        ({"grid": "gl", "lw": 0}, "lw"),
        ({"grid": "gl", "lw": 51.2}, "lw"),
        ({"nside": 0}, "nside"),
        ({"nside": True}, "nside"),
        ({"nside": "256"}, "nside"),
        ({"grid": "healpix", "nside": 100}, "power of two"),
        ({"spectra": []}, "non-empty list"),
        ({"spectra": "TT"}, "non-empty list"),
        ({"spectra": ["TT", "XX"]}, "unknown spectra"),
        ({"spectra": ["TT"]}, "lacks ['DD', 'TD', 'DT']"),
        ({"spectra": ["TT", "EE", "TE"]}, "lacks ['DT']"),
        ({"max_memory_gb": 0}, "max_memory_gb"),
        ({"max_memory_gb": -2}, "max_memory_gb"),
        ({"max_memory_gb": True}, "max_memory_gb"),
        ({"max_memory_gb": "6"}, "max_memory_gb"),
    ],
)
def test_bad_blocks_are_errors(block, fragment):
    ok, errors, _ = _validate(block)
    assert not ok
    assert any(fragment in e for e in errors), errors


def test_block_under_another_method_warns_but_validates():
    ok, errors, warns = _validate({"grid": "gl"}, covariance_approximation="nka")
    assert ok, errors
    assert any("acc_precompute" in w and "ignored" in w for w in warns), warns


def test_needed_spectra_agree_with_the_covariance_keys():
    """The coverage check derives its answer from the keys a run computes."""
    from cmbcov.generator.parameter_validation import (
        acc_kernel_spectra_needed,
    )

    assert acc_kernel_spectra_needed(["T"]) == ("TT",)
    assert acc_kernel_spectra_needed(["E"]) == ("DD",)
    assert acc_kernel_spectra_needed(["T", "E"]) == ("TT", "DD", "TD", "DT")
    stokes = set()
    for key in CovKeys(["T", "E"], ["a", "b"]).keys():
        for contraction in key.key_to_cross_kernel():
            stokes.update(s.kernel_stokekey() for s in contraction)
    assert stokes == {"TT", "DD", "TD", "DT"}


# --------------------------------------------------------------------------- #
# precompute-acc end to end on a parameter file
# --------------------------------------------------------------------------- #
BLOCK = """
acc_precompute:
  nside: 16
  grid: gl
  lw: 10
"""


def _params_file(tmp_path, method="acc", block=BLOCK, dmax=3):
    template = open(os.path.join(DATA, "baseline_params.yml")).read()
    text = template.replace("PLACEHOLDER_OUT", str(tmp_path / "out")).replace(
        "PLACEHOLDER_DATA", DATA
    )
    text = text.replace(
        "covariance_approximation: nka",
        f"covariance_approximation: {method}\ndmax: {dmax}\ncentralell: {ELL}",
    )
    # ACC reads the spectra to lmax_int = lmax + (2 NSIDE - 1 - ELL) = lmax +
    # 15, which must stay within baseline_cls.dat (ell <= 64): report to lmax
    # 48 instead of 60.
    text = text.replace("lmax: 60", f"lmax: {LMAX}").replace(
        "[20, 60, 10]", f"[20, {LMAX}, 10]"
    )
    path = tmp_path / "params.yml"
    path.write_text(text + textwrap.dedent(block))
    return str(path)


def test_pipeline_config_exposes_the_block(tmp_path):
    config = ParameterManager(_params_file(tmp_path)).load_and_validate(
        setup_output=False
    )
    assert config.acc_precompute == AccPrecomputeConfig(nside=16, grid="gl", lw=10)
    assert config.save_dir is None
    assert config.acc_kernel_dir == str(tmp_path / "out")
    assert not os.path.exists(tmp_path / "out")  # no version directory made
    with pytest.raises(AttributeError):
        config.acc_precompute.grid = "healpix"

    plain = ParameterManager(_params_file(tmp_path, block="")).load_and_validate(
        setup_output=False
    )
    assert plain.acc_precompute is None


def test_precompute_acc_script_then_every_run_reuses_the_kernels(tmp_path, capsys):
    params = _params_file(tmp_path)

    assert precompute_acc.main([params, "--dryrun"]) == 0
    assert not os.path.exists(tmp_path / "out")

    assert precompute_acc.main([params]) == 0
    kernel_dir = tmp_path / "out" / "covariance_coupling"
    manifests = sorted(os.listdir(kernel_dir))
    assert [m for m in manifests if m.endswith(".json")] == [
        f"manifest_{ELL}x{p}.json" for p in coupling_ellprange(ELL, 3)
    ]
    manifest = json.load(open(kernel_dir / f"manifest_{ELL}x{ELL}.json"))
    assert (manifest["grid"], manifest["lw"], manifest["nside"]) == ("gl", 10, 16)
    assert "Wrote 3 kernel pair(s)" in capsys.readouterr().out
    # The precompute did not create a version directory for the next run to
    # silently reuse.
    assert glob.glob(str(tmp_path / "out" / "v*")) == []

    # Two runs of the same file: each gets a fresh versioned save_dir, and
    # both find the one set of kernels.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # low centralell
        CovarianceMatrixGenerator(params).run_full_analysis()
        CovarianceMatrixGenerator(params).run_full_analysis()
    first = np.loadtxt(tmp_path / "out" / "v0" / "covariance_matrix.dat")
    second = np.loadtxt(tmp_path / "out" / "v1" / "covariance_matrix.dat")
    assert np.isfinite(first).all() and np.abs(np.diag(first)).min() > 0
    assert np.array_equal(first, second)
    assert not os.path.exists(tmp_path / "out" / "v0" / "covariance_coupling")


def test_precompute_acc_script_refuses_a_non_acc_file(tmp_path, capsys):
    params = _params_file(tmp_path, method="nka")
    assert precompute_acc.main(["-p", params]) == 1
    assert "only applies to 'acc'" in capsys.readouterr().err
    assert not os.path.exists(tmp_path / "out" / "covariance_coupling")


def test_precompute_acc_script_reports_an_invalid_block(tmp_path, capsys):
    params = _params_file(
        tmp_path, block="acc_precompute:\n  grid: healpix\n  lw: 10\n"
    )
    assert precompute_acc.main([params]) == 1
    assert "only used with grid 'gl'" in capsys.readouterr().err


def test_entry_point_is_declared():
    text = open(os.path.join(os.path.dirname(DATA), "..", "pyproject.toml")).read()
    assert 'precompute-acc = "cmbcov.scripts.precompute_acc:main"' in text
    assert callable(precompute_acc.main)
