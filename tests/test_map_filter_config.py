"""
Tests of the ``map_filter`` (filter-and-bin) plumbing: the parameter block,
the per-channel T/E hook in ``Cov.compute_covariance_matrix`` (B blocks left
alone) and the generator wiring.

The numerics live in ``cmbcov.filtering``; nothing here runs them. The
module is replaced by a stand-in (``fake_filtering``), so these tests check
only how the correction is configured, cached and applied.
"""

import os
import shutil
import sys
import types
import warnings

import numpy as np
import pytest
import yaml

import cmbcov
from cmbcov.covariance import Cov, CovarianceConfig, CovarianceMethod
from cmbcov.generator import CovarianceMatrixGenerator
from cmbcov.generator.parameter_validation import (
    MapFilterConfig,
    ParameterManager,
    ParameterValidator,
    PipelineConfig,
)
from cmbcov.keys import CovKeys

DATA = os.path.join(os.path.dirname(__file__), "data")
MASK = "baseline_mask.fits"
FREQ = "090GHz"
LBINS = [2, 10, 20, 30]

BASE = {
    "cov_path": "./out",
    "cov_name": "covariance_matrix.dat",
    "frequencies": [FREQ],
    "stokes": ["T"],
    "lmax": 60,
    "lmin": 2,
    "bins": [[2, 60, 10]],
    "mask_name": MASK,
    "mask_path": DATA,
    "covariance_approximation": "nka",
}


# --------------------------------------------------------------------------- #
# A stand-in for cmbcov.filtering (the real one is tested in test_filtering.py)
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_filtering(monkeypatch):
    """
    Install stand-ins for the ``cmbcov.filtering`` functions the plumbing
    calls; the stand-ins record their arguments in ``module.calls``.
    """
    try:
        import cmbcov.filtering as module
    except ImportError:
        module = types.ModuleType("cmbcov.filtering")
        monkeypatch.setitem(sys.modules, "cmbcov.filtering", module)
        monkeypatch.setattr(cmbcov, "filtering", module, raising=False)

    module_calls = {"probe": [], "cache": []}
    installed = {}
    if not hasattr(module, "BLOCK_CHANNEL"):
        # Only for a stand-in module: the real table is data, not numerics,
        # and is used as is (it is tested in test_filtering.py).
        te = ("TT", "EE", "TE", "ET")
        installed["CHANNELS"] = ("00", "20", "EE")
        installed["BLOCK_CHANNEL"] = {
            (a, b): (
                "00"
                if a == b == "TT"
                else "EE" if "EE" in (a, b) and "TT" not in (a, b) else "20"
            )
            for a in te
            for b in te
        }

        def channels_for(stokes_pairs):
            table = installed["BLOCK_CHANNEL"]
            needed = {table.get(tuple(p.split("x"))) for p in stokes_pairs}
            return tuple(ch for ch in installed["CHANNELS"] if ch in needed)

        installed["channels_for"] = channels_for

    def default_nodes(lmax, node_min=32, ratio=1.25):
        nodes = [node_min]
        while nodes[-1] < lmax - 1:
            nodes.append(min(int(np.ceil(nodes[-1] * ratio)), lmax - 1))
        return nodes

    def highpass_profile(lmax_grid, lx, shape="sharp", power=6.0):
        return np.ones(lmax_grid + 1)

    def filtered_sum_rule_ratio(mask_alm, lw, profile, nodes, dmax, **kwargs):
        module_calls["probe"].append(
            {"lw": lw, "nodes": list(nodes), "dmax": dmax, "kwargs": kwargs}
        )
        return {
            ch: np.full((len(nodes), dmax + 1), 0.5)
            for ch in kwargs.get("channels", ("00",))
        }

    def rho_matrix(nodes, g, lmax):
        return np.full((lmax, lmax), 0.5)

    def transfer_correction(rho, fl_left, fl_right):
        return rho / np.outer(fl_left, fl_right)

    def cached_sum_rule_ratio(cache_dir, identity, compute):
        module_calls["cache"].append({"dir": cache_dir, "identity": identity})
        return compute()

    for name, function in {
        **installed,
        "default_nodes": default_nodes,
        "highpass_profile": highpass_profile,
        "filtered_sum_rule_ratio": filtered_sum_rule_ratio,
        "rho_matrix": rho_matrix,
        "transfer_correction": transfer_correction,
        "cached_sum_rule_ratio": cached_sum_rule_ratio,
    }.items():
        monkeypatch.setattr(module, name, function, raising=False)
    module.calls = module_calls
    return module


# --------------------------------------------------------------------------- #
# The parameter block
# --------------------------------------------------------------------------- #
def _validate(block, **overrides):
    params = dict(BASE, **overrides)
    if block is not ...:
        params["map_filter"] = block
    validator = ParameterValidator()
    ok = validator.validate(params)
    return ok, validator.errors


@pytest.mark.parametrize(
    "block",
    [
        ...,  # absent
        {"type": "fourier_highpass", "lx": 30},
        {"type": "fourier_highpass", "lx": 30.5, "shape": "exp", "power": 4},
        {"type": "fourier_highpass", "lx": 30, "nprobe": 1, "seed": -3},
        {"type": "fourier_highpass", "lx": 30, "node_min": 2, "node_ratio": 1.01},
        {"type": "fourier_highpass", "lx": 30, "dmax": 0, "lw": 100},
        {"type": "fourier_highpass", "lx": 30, "dmax": None, "lw": None},
    ],
)
def test_good_blocks_validate(block):
    ok, errors = _validate(block)
    assert ok, errors


@pytest.mark.parametrize(
    "block, fragment",
    [
        (None, "must be a mapping"),
        ({}, "'map_filter.type' is required"),
        ([1, 2], "must be a mapping"),
        ({"lx": 30}, "'map_filter.type' is required"),
        ({"type": "gaussian", "lx": 30}, "map_filter.type"),
        ({"type": "fourier_highpass"}, "'map_filter.lx' is required"),
        ({"type": "fourier_highpass", "lx": 0}, "map_filter.lx"),
        ({"type": "fourier_highpass", "lx": -5}, "map_filter.lx"),
        ({"type": "fourier_highpass", "lx": "30"}, "map_filter.lx"),
        ({"type": "fourier_highpass", "lx": True}, "map_filter.lx"),
        ({"type": "fourier_highpass", "lx": 30, "shape": "box"}, "map_filter.shape"),
        ({"type": "fourier_highpass", "lx": 30, "power": 0}, "map_filter.power"),
        ({"type": "fourier_highpass", "lx": 30, "nprobe": 0}, "map_filter.nprobe"),
        ({"type": "fourier_highpass", "lx": 30, "nprobe": 2.5}, "map_filter.nprobe"),
        ({"type": "fourier_highpass", "lx": 30, "seed": 1.5}, "map_filter.seed"),
        ({"type": "fourier_highpass", "lx": 30, "node_min": 1}, "map_filter.node_min"),
        (
            {"type": "fourier_highpass", "lx": 30, "node_ratio": 1.0},
            "map_filter.node_ratio",
        ),
        ({"type": "fourier_highpass", "lx": 30, "dmax": -1}, "map_filter.dmax"),
        ({"type": "fourier_highpass", "lx": 30, "lw": 0}, "map_filter.lw"),
        ({"type": "fourier_highpass", "lx": 30, "bogus": 1}, "Unknown key"),
    ],
)
def test_bad_blocks_are_reported(block, fragment):
    ok, errors = _validate(block)
    assert not ok
    assert any(fragment in error for error in errors), errors


def test_defaults_and_absent_block():
    config = MapFilterConfig.from_block({"type": "fourier_highpass", "lx": 30})
    assert config == MapFilterConfig(type="fourier_highpass", lx=30)
    assert (config.shape, config.power, config.nprobe, config.seed) == (
        "sharp",
        6.0,
        32,
        0,
    )
    assert (config.node_min, config.node_ratio) == (32, 1.25)
    assert config.dmax is None and config.lw is None
    params = dict(BASE, save_dir="x", git_hash="y")
    # no block: None
    validator = ParameterValidator()
    assert validator.validate(dict(BASE))
    assert PipelineConfig.from_params(_effective(params)).map_filter is None
    with_block = dict(params, map_filter={"type": "fourier_highpass", "lx": 30})
    assert PipelineConfig.from_params(_effective(with_block)).map_filter == config


def _effective(params):
    """The parameters as load_and_validate assembles them: defaults applied."""
    effective = dict(ParameterValidator.OPTIONAL_PARAMS)
    effective.update(params)
    return effective


def test_resolved_dmax_and_lw():
    config = MapFilterConfig(type="fourier_highpass", lx=30)
    assert config.resolved_dmax(2000, CovarianceMethod.ACC, 11) == 10
    assert config.resolved_dmax(2000, CovarianceMethod.NKA, 0) == 64
    assert config.resolved_dmax(40, CovarianceMethod.NKA, 0) == 39
    explicit = MapFilterConfig(type="fourier_highpass", lx=30, dmax=7, lw=99)
    assert explicit.resolved_dmax(2000, CovarianceMethod.ACC, 11) == 7
    assert explicit.resolved_lw(512, 100) == 99
    assert config.resolved_lw(512, 2000) == 3 * 512 - 1
    assert config.resolved_lw(512, 100) == 200


def _manager(tmp_path, params):
    manager = ParameterManager(str(tmp_path / "p.yml"))
    manager.config = types.SimpleNamespace(params=params)
    return manager


def test_changed_or_dropped_map_filter_is_incompatible(tmp_path):
    block = {"type": "fourier_highpass", "lx": 30}
    manager = _manager(tmp_path, dict(BASE, map_filter=block))
    assert manager._incompatible_parameters(dict(BASE, map_filter=dict(block))) == []
    assert manager._incompatible_parameters(
        dict(BASE, map_filter=dict(block, lx=40))
    ) == ["map_filter"]
    assert manager._incompatible_parameters(dict(BASE)) == ["map_filter"]
    # a saved file that has the block where the new one has none
    manager = _manager(tmp_path, dict(BASE))
    assert manager._incompatible_parameters(dict(BASE, map_filter=block)) == [
        "map_filter"
    ]


# --------------------------------------------------------------------------- #
# The Cov hook
# --------------------------------------------------------------------------- #
def _cl(size, freqs=(FREQ,), stokes=("TT",)):
    ell = np.arange(size)
    base = np.zeros(size)
    base[2:] = 1e-3 / (ell[2:] * (ell[2:] + 1)) ** 0.9
    out = {}
    for i, f1 in enumerate(freqs):
        for f2 in freqs[i:]:
            out[f1 + f2] = {
                s: base * (1 + 0.1 * i) * (0.3 if s != "TT" else 1.0) for s in stokes
            }
    return out


def _cov(save_dir, lmax=32, **config):
    config = CovarianceConfig(
        method=CovarianceMethod.NKA,
        lmax=lmax,
        polspice_postprocess=False,
        **config,
    )
    return Cov(
        MASK,
        config=config,
        mask_path=os.path.abspath(DATA),
        save_dir=str(save_dir),
    )


class _Spy:
    """Record the raw block that reaches the post-processing of each key."""

    def __init__(self, monkeypatch):
        self.terms = {}
        real = Cov._add_to_output_matrix

        def spy(cov, output_matrix, covariance_term, cov_key, *args, **kwargs):
            self.terms[cov_key.stokekey(), cov_key.freqkey()] = np.array(
                covariance_term
            )
            return real(cov, output_matrix, covariance_term, cov_key, *args, **kwargs)

        monkeypatch.setattr(Cov, "_add_to_output_matrix", spy)


def _run(cov, keys, cl):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = cov.compute_covariance_matrix(LBINS, keys, cl)
    return result, [w for w in caught if "map_filter" in str(w.message)]


def _factor(rho, fl_left, fl_right):
    return rho / np.outer(fl_left, fl_right)


def test_hook_multiplies_tt_blocks_by_rho_over_fl(
    tmp_path, monkeypatch, fake_filtering
):
    lmax = 32
    freqs = ["090GHz", "150GHz"]
    keys = CovKeys(["T"], freqs)
    cl = _cl(lmax, freqs)
    rng = np.random.default_rng(3)
    rho = rng.uniform(0.5, 1.5, (lmax, lmax))
    fl = {
        pair: {"TT": rng.uniform(0.3, 1.0, lmax + 5)}
        for pair in keys.combined_frequencies()
    }

    spy = _Spy(monkeypatch)
    (_, bin_matrix, plain), warned = _run(_cov(tmp_path / "a"), keys, cl)
    assert warned == []
    raw = dict(spy.terms)

    spy.terms.clear()
    (_, _, filtered), warned = _run(
        _cov(tmp_path / "b", filter_rho={"00": rho}, filter_fl=fl), keys, cl
    )
    assert warned == []  # TT only: no warning
    assert set(spy.terms) == set(raw)
    for (stokes, freq), term in spy.terms.items():
        left, right = freq.split("x")
        expected = raw[stokes, freq] * _factor(
            rho, fl[left]["TT"][:lmax], fl[right]["TT"][:lmax]
        )
        np.testing.assert_allclose(term, expected, rtol=1e-13, atol=0)
        assert not np.allclose(term, raw[stokes, freq], rtol=1e-3, atol=0)

    # the binned result is the binned corrected block (no PolSpice transform,
    # no debiasing, no D_ell here)
    n = bin_matrix.shape[0]
    for cov_key in keys.keys():
        i, j = keys[cov_key]
        block = filtered[i * n : (i + 1) * n, j * n : (j + 1) * n]
        expected = (
            bin_matrix @ spy.terms[cov_key.stokekey(), cov_key.freqkey()] @ bin_matrix.T
        )
        np.testing.assert_allclose(block, expected, rtol=1e-12, atol=0)


def test_filter_fl_none_means_rho_alone(tmp_path, monkeypatch, fake_filtering):
    lmax = 32
    keys = CovKeys(["T"], [FREQ])
    rho = np.full((lmax, lmax), 0.7)
    spy = _Spy(monkeypatch)
    _run(_cov(tmp_path / "a"), keys, _cl(lmax))
    raw = next(iter(spy.terms.values()))
    _run(_cov(tmp_path / "b", filter_rho={"00": rho}), keys, _cl(lmax))
    np.testing.assert_allclose(next(iter(spy.terms.values())), 0.7 * raw, rtol=1e-13)


def test_unset_filter_is_bit_identical_and_never_imports_filtering(tmp_path):
    keys = CovKeys(["T"], [FREQ])
    cl = _cl(32)
    first, _ = _run(_cov(tmp_path / "a"), keys, cl)
    second, warned = _run(
        _cov(tmp_path / "b", filter_rho=None, filter_fl=None), keys, cl
    )
    np.testing.assert_array_equal(first[2], second[2])
    assert warned == []


#: The channel of each T/E block of a one-frequency T+E run, written out
#: (not read from cmbcov.filtering.BLOCK_CHANNEL).
TE_RUN_CHANNELS = {
    "TTxTT": "00",
    "TTxEE": "20",
    "EExTT": "20",
    "TTxTE": "20",
    "TExTT": "20",
    "TExTE": "20",
    "EExEE": "EE",
    "EExTE": "EE",
    "TExEE": "EE",
}


def test_te_blocks_are_corrected_by_their_channel(
    tmp_path, monkeypatch, fake_filtering
):
    """
    T+E run: every block is multiplied by ``rho[ch] / (fl[s1] fl[s2])`` with
    its own channel and the transfer functions of its two spectra; no
    warning (no B block).
    """
    lmax = 32
    keys = CovKeys(["T", "E"], [FREQ])
    cl = _cl(lmax, stokes=("TT", "EE", "TE", "ET"))
    rng = np.random.default_rng(5)
    rho = {ch: rng.uniform(0.5, 1.5, (lmax, lmax)) for ch in ("00", "20", "EE")}
    pair = FREQ + FREQ
    fl = {pair: {s: rng.uniform(0.3, 1.0, lmax) for s in ("TT", "EE", "TE")}}
    spy = _Spy(monkeypatch)
    _run(_cov(tmp_path / "a"), keys, cl)
    raw = dict(spy.terms)
    spy.terms.clear()
    _, warned = _run(_cov(tmp_path / "b", filter_rho=rho, filter_fl=fl), keys, cl)
    assert warned == []
    seen = {stokes for stokes, _ in spy.terms}
    assert seen == {"TTxTT", "TTxTE", "TTxEE", "TExTE", "TExEE", "EExEE"}
    for (stokes, freq), term in spy.terms.items():
        channel = TE_RUN_CHANNELS[stokes]
        s1, s2 = (s if s != "ET" else "TE" for s in stokes.split("x"))
        expected = raw[stokes, freq] * _factor(rho[channel], fl[pair][s1], fl[pair][s2])
        np.testing.assert_allclose(term, expected, rtol=1e-13, atol=0)


def test_missing_channel_is_refused(tmp_path, fake_filtering):
    keys = CovKeys(["T", "E"], [FREQ])
    cl = _cl(32, stokes=("TT", "EE", "TE", "ET"))
    cov = _cov(tmp_path, filter_rho={"00": np.ones((32, 32))})
    with pytest.raises(ValueError, match="channel '20'"):
        cov.compute_covariance_matrix(LBINS, keys, cl)


def test_missing_transfer_function_is_refused(tmp_path, fake_filtering):
    keys = CovKeys(["T", "E"], [FREQ])
    cl = _cl(32, stokes=("TT", "EE", "TE", "ET"))
    rho = {ch: np.ones((32, 32)) for ch in ("00", "20", "EE")}
    fl = {FREQ + FREQ: {"TT": np.ones(32), "TE": np.ones(32)}}
    cov = _cov(tmp_path, filter_rho=rho, filter_fl=fl)
    with pytest.raises(ValueError, match="no 'EE' transfer function"):
        cov.compute_covariance_matrix(LBINS, keys, cl)


def test_b_blocks_are_unchanged_with_one_warning(tmp_path, monkeypatch, fake_filtering):
    """BB-only NKA run (the cheap B-capable configuration): left as it is."""
    lmax = 32
    keys = CovKeys([], [FREQ], observables=["BB"])
    cl = _cl(lmax, stokes=("TT", "EE", "BB", "TE", "ET"))
    spy = _Spy(monkeypatch)
    _run(_cov(tmp_path / "a"), keys, cl)
    raw = dict(spy.terms)
    assert [stokes for stokes, _ in raw] == ["BBxBB"]
    spy.terms.clear()
    rho = {"00": np.full((lmax, lmax), 0.5), "EE": np.full((lmax, lmax), 0.5)}
    _, warned = _run(_cov(tmp_path / "b", filter_rho=rho), keys, cl)
    assert len(warned) == 1
    assert issubclass(warned[0].category, UserWarning)
    message = str(warned[0].message)
    assert "BBxBB" in message and "left uncorrected" in message
    for key, term in spy.terms.items():
        np.testing.assert_array_equal(term, raw[key])


def test_raw_block_cache_stays_unfiltered(tmp_path, monkeypatch, fake_filtering):
    lmax = 32
    keys = CovKeys(["T"], [FREQ])
    cl = _cl(lmax)
    rho = {"00": np.full((lmax, lmax), 0.5)}
    plain_dir, filtered_dir = tmp_path / "a", tmp_path / "b"
    plain_dir.mkdir()
    filtered_dir.mkdir()
    (_, _, plain), _ = _run(_cov(plain_dir, save_raw_blocks=True), keys, cl)
    (_, _, fresh), _ = _run(
        _cov(filtered_dir, save_raw_blocks=True, filter_rho=rho), keys, cl
    )
    name = "cov_TTxTT_090GHz090GHzx090GHz090GHz.npy"
    np.testing.assert_array_equal(
        np.load(plain_dir / name), np.load(filtered_dir / name)
    )
    assert not np.allclose(plain, fresh, rtol=1e-3, atol=0)

    # a second run reads the cached (unfiltered) block and still corrects it
    (_, _, cached), _ = _run(
        _cov(filtered_dir, save_raw_blocks=True, filter_rho=rho), keys, cl
    )
    np.testing.assert_array_equal(cached, fresh)
    np.testing.assert_array_equal(
        np.load(plain_dir / name), np.load(filtered_dir / name)
    )


def test_hook_does_not_modify_the_block_in_place(tmp_path, monkeypatch, fake_filtering):
    """The factor is applied to a new array: the strategy's block survives."""
    from cmbcov.approximations import StrategyFactory

    keys = CovKeys(["T"], [FREQ])
    produced = []
    real = StrategyFactory.create_strategy

    def create(cov):
        strategy = real(cov)
        inner = strategy.compute_covariance_terms

        def recording(*args, **kwargs):
            for cov_key, term in inner(*args, **kwargs):
                produced.append((term, term.copy()))
                yield cov_key, term

        strategy.compute_covariance_terms = recording
        return strategy

    monkeypatch.setattr(StrategyFactory, "create_strategy", staticmethod(create))
    _run(_cov(tmp_path, filter_rho={"00": np.full((32, 32), 0.5)}), keys, _cl(32))
    assert produced
    for term, snapshot in produced:
        np.testing.assert_array_equal(term, snapshot)


PAIR = FREQ + FREQ


@pytest.mark.parametrize(
    "rho, fl, fragment",
    [
        (np.ones((32, 32)), None, "must be a dict"),
        ({"00": np.ones((31, 31))}, None, "shape"),
        ({"00": np.ones(32)}, None, "shape"),
        ({"BB": np.ones((32, 32))}, None, "unknown channel"),
        ({"00": np.ones((32, 32))}, {PAIR: np.ones(40)}, "must be a dict"),
        ({"00": np.ones((32, 32))}, {PAIR: {"TT": np.ones(10)}}, "length"),
        (None, {PAIR: {"TT": np.ones(40)}}, "without filter_rho"),
    ],
)
def test_bad_filter_inputs_are_refused(tmp_path, rho, fl, fragment):
    with pytest.raises(ValueError, match=fragment):
        _cov(tmp_path, filter_rho=rho, filter_fl=fl)


# --------------------------------------------------------------------------- #
# The generator wiring
# --------------------------------------------------------------------------- #
def _write_inputs(workdir, with_fl, polarised=False):
    shutil.copy(os.path.join(DATA, MASK), workdir)
    ell = np.arange(80)
    tt = np.zeros(80)
    tt[2:] = 1e3 / (ell[2:] * (ell[2:] + 1)) ** 0.9
    # 4-column layout TT EE BB TE for a polarised run
    columns = [tt, 0.05 * tt, 0.001 * tt, 0.1 * tt] if polarised else [tt]
    np.savetxt(os.path.join(workdir, "cls.dat"), np.column_stack([ell, *columns]))
    extra = {}
    if with_fl:
        values = [0.8, 0.7, 0.6, 0.75] if polarised else [0.8]
        fl = np.column_stack([ell] + [np.full(80, v) for v in values])
        extra["fl"] = os.path.join(workdir, "fl_{}.dat")
        np.savetxt(os.path.join(workdir, f"fl_{FREQ}{FREQ}.dat"), fl)
    return extra


def _params(workdir, with_fl=True, polarised=False, **extra):
    params = {
        "cov_path": os.path.join(workdir, "out"),
        "cov_name": "cov.dat",
        "frequencies": [FREQ],
        "stokes": ["T", "E"] if polarised else ["T"],
        "lmax": 20,
        "lmin": 2,
        "bins": [[2, 20, 3]],
        "mask_name": MASK,
        "mask_path": workdir,
        "covariance_approximation": "nka",
        "cmb_spectrum": os.path.join(workdir, "cls.dat"),
        "beams": {FREQ: 5.0},
        "pixwin": 32,
        "nl": {FREQ: 10.0},
        "polspice_postprocess": False,
        "map_filter": {"type": "fourier_highpass", "lx": 12, "nprobe": 5, "seed": 4},
    }
    params.update(_write_inputs(workdir, with_fl, polarised))
    params.update(extra)
    path = os.path.join(workdir, "params.yml")
    with open(path, "w") as handle:
        yaml.dump(params, handle)
    return path


def _capture_filter(monkeypatch):
    seen = {}
    real = Cov.compute_covariance_matrix

    def capture(cov, *args, **kwargs):
        seen["rho"] = cov.config.filter_rho
        seen["fl"] = cov.config.filter_fl
        seen["mask_digest"] = cov.wlm.mask_digest
        return real(cov, *args, **kwargs)

    monkeypatch.setattr(Cov, "compute_covariance_matrix", capture)
    return seen


def test_generator_builds_rho_and_hands_it_to_cov(
    tmp_path, monkeypatch, fake_filtering
):
    path = _params(str(tmp_path))
    seen = _capture_filter(monkeypatch)
    generator = CovarianceMatrixGenerator(path)
    generator.run_full_analysis()

    calls = fake_filtering.calls
    assert len(calls["cache"]) == 1
    cache = calls["cache"][0]
    assert cache["dir"] == generator.config.acc_kernel_dir
    identity = cache["identity"]
    assert identity["mask_digest"] == seen["mask_digest"]
    lw = min(3 * generator.covariance_instance.wlm.nside - 1, 2 * 20)
    assert identity["lw"] == lw
    assert identity["lmax"] == 20
    assert identity["lx"] == 12
    assert identity["shape"] == "sharp" and identity["power"] == 6.0
    assert identity["nprobe"] == 5 and identity["seed"] == 4
    assert identity["dmax"] == 19  # NKA: 64, capped at lmax - 1
    assert identity["nodes"] == fake_filtering.default_nodes(20, 32, 1.25)
    assert identity["type"] == "fourier_highpass"
    assert identity["channels"] == ["00"]  # TT-only run: the spin-0 channel

    assert len(calls["probe"]) == 1
    probe = calls["probe"][0]
    assert probe["lw"] == lw and probe["dmax"] == 19
    assert probe["kwargs"]["nprobe"] == 5 and probe["kwargs"]["seed"] == 4
    assert tuple(probe["kwargs"]["channels"]) == ("00",)

    assert set(seen["rho"]) == {"00"}
    assert seen["rho"]["00"].shape == (20, 20)
    assert set(seen["fl"]) == {FREQ + FREQ}
    np.testing.assert_allclose(seen["fl"][FREQ + FREQ]["TT"][:20], 0.8)

    saved = np.load(os.path.join(generator.config.save_dir, "map_filter_ratio.npz"))
    assert saved["nodes"].tolist() == identity["nodes"]
    assert saved["channels"].tolist() == ["00"]
    assert saved["g_00"].shape == (len(identity["nodes"]), 20)


def test_generator_te_run_probes_every_channel(
    tmp_path, monkeypatch, fake_filtering, caplog
):
    path = _params(str(tmp_path), polarised=True)
    seen = _capture_filter(monkeypatch)
    generator = CovarianceMatrixGenerator(path)
    with caplog.at_level("WARNING"):
        generator.run_full_analysis()
    assert not any("uncorrected" in r.getMessage() for r in caplog.records)

    calls = fake_filtering.calls
    assert len(calls["cache"]) == 1 and len(calls["probe"]) == 1
    assert calls["cache"][0]["identity"]["channels"] == ["00", "20", "EE"]
    assert tuple(calls["probe"][0]["kwargs"]["channels"]) == ("00", "20", "EE")

    assert set(seen["rho"]) == {"00", "20", "EE"}
    for rho in seen["rho"].values():
        assert rho.shape == (20, 20)
    fl = seen["fl"][FREQ + FREQ]
    for stokes, value in (("TT", 0.8), ("EE", 0.7), ("TE", 0.75)):
        np.testing.assert_allclose(fl[stokes][:20], value)

    saved = np.load(os.path.join(generator.config.save_dir, "map_filter_ratio.npz"))
    assert saved["channels"].tolist() == ["00", "20", "EE"]
    for ch in ("00", "20", "EE"):
        assert saved[f"g_{ch}"].shape == (len(saved["nodes"]), 20)


def test_generator_without_fl_uses_rho_alone_and_warns(
    tmp_path, monkeypatch, fake_filtering, caplog
):
    path = _params(str(tmp_path), with_fl=False)
    generator = CovarianceMatrixGenerator(path)
    with caplog.at_level("WARNING"):
        generator.run_full_analysis()
    assert any("without fl" in r.getMessage() for r in caplog.records)
    fl = generator.covariance_instance.config.filter_fl[FREQ + FREQ]["TT"]
    np.testing.assert_array_equal(fl[:20], np.ones(20))


@pytest.mark.filterwarnings("ignore:BB-only nka run")  # its own leakage warning
def test_generator_bb_only_run_probes_nothing_and_warns(
    tmp_path, fake_filtering, caplog
):
    params = _params(str(tmp_path), polarised=True)
    with open(params) as handle:
        loaded = yaml.safe_load(handle)
    loaded.pop("stokes")
    loaded["observables"] = ["BB"]
    with open(params, "w") as handle:
        yaml.dump(loaded, handle)
    generator = CovarianceMatrixGenerator(params)
    with caplog.at_level("WARNING"):
        generator.run_full_analysis()
    messages = [r.getMessage() for r in caplog.records]
    assert any("BBxBB" in m and "left uncorrected" in m for m in messages)
    assert any("no T/E block" in m for m in messages)
    assert fake_filtering.calls == {"probe": [], "cache": []}
    assert generator.covariance_instance.config.filter_rho is None


def test_generator_dryrun_computes_no_rho(tmp_path, fake_filtering):
    generator = CovarianceMatrixGenerator(_params(str(tmp_path)))
    generator.run_full_analysis(dryrun=True)
    assert fake_filtering.calls == {"probe": [], "cache": []}
    assert generator.covariance_instance.config.filter_rho is None


def test_generator_without_block_never_touches_filtering(tmp_path, fake_filtering):
    path = _params(str(tmp_path))
    params = yaml.safe_load(open(path))
    del params["map_filter"]
    with open(path, "w") as handle:
        yaml.dump(params, handle)
    generator = CovarianceMatrixGenerator(path)
    generator.run_full_analysis()
    assert fake_filtering.calls == {"probe": [], "cache": []}
    assert generator.covariance_instance.config.filter_rho is None
    assert not os.path.exists(
        os.path.join(generator.config.save_dir, "map_filter_ratio.npz")
    )
