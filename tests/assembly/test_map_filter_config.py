"""
Tests of the ``map_filter`` (filter-and-bin) plumbing: the parameter block,
the per-channel T/E hook in ``Cov.compute_covariance_matrix`` (B blocks left
alone) and the generator wiring.

The numerics live in ``cmbcov.assembly.filtering``; nothing here runs them. The
module is replaced by a stand-in (``fake_filtering``), so these tests check
only how the correction is configured, cached and applied.
"""

import json
import os
import shutil
import sys
import types
import warnings

import numpy as np
import pytest
import yaml
from conftest import DATA_DIR  # noqa: E402

import cmbcov
from cmbcov.assembly.covariance import Cov, CovarianceConfig, CovarianceMethod
from cmbcov.assembly.keys import CovKeys
from cmbcov.config import CovarianceMatrixGenerator
from cmbcov.config.validation import (
    LowpassConfig,
    MapFilterConfig,
    ParameterManager,
    ParameterValidator,
    PipelineConfig,
)

DATA = DATA_DIR
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
# A stand-in for cmbcov.assembly.filtering (the real one is tested in test_filtering.py)
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_filtering(monkeypatch):
    """
    Install stand-ins for the ``cmbcov.assembly.filtering`` functions the plumbing
    calls; the stand-ins record their arguments in ``module.calls``.
    """
    try:
        import cmbcov.assembly.filtering as module
    except ImportError:
        module = types.ModuleType("cmbcov.assembly.filtering")
        monkeypatch.setitem(sys.modules, "cmbcov.assembly.filtering", module)
        monkeypatch.setattr(cmbcov.assembly, "filtering", module, raising=False)

    module_calls = {"probe": [], "cache": []}
    plan_calls = []
    real_node_plan = getattr(module, "node_plan", None)
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

    def node_plan(lmax, node_min, ratio, lx, **kwargs):
        # pure Python (no numerics): the real plan, recorded
        plan_calls.append(
            {"lmax": lmax, "node_min": node_min, "ratio": ratio, "lx": lx, **kwargs}
        )
        if real_node_plan is not None:
            return real_node_plan(lmax, node_min, ratio, lx, **kwargs)
        nodes = default_nodes(lmax, node_min, ratio)
        return types.SimpleNamespace(
            nodes=nodes,
            nprobe=[kwargs["nprobe"]] * len(nodes),
            lmax_int_floor=[0] * len(nodes),
        )

    def highpass_profile(lmax_grid, lx, shape="sharp", power=6.0):
        return np.ones(lmax_grid + 1)

    def filtered_sum_rule_ratio(mask_alm, lw, profile, nodes, dmax, **kwargs):
        module_calls["probe"].append(
            {
                "lw": lw,
                "profile": profile,
                "nodes": list(nodes),
                "dmax": dmax,
                "kwargs": kwargs,
            }
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

    sa_calls = []

    def spectrum_aware_ratio(
        mask_alm, lw, profile, nodes, dmax, cl_pre, cl_used, terms, **kwargs
    ):
        sa_calls.append(
            {
                "lw": lw,
                "profile": profile,
                "nodes": list(nodes),
                "dmax": dmax,
                "cl_pre": {p: np.array(v) for p, v in cl_pre.items()},
                "cl_used": {p: np.array(v) for p, v in cl_used.items()},
                "terms": {b: list(t) for b, t in terms.items()},
                "kwargs": kwargs,
            }
        )
        return {
            b: np.full((len(nodes), dmax + 1), 0.25 + 0.01 * i)
            for i, b in enumerate(terms)
        }

    for name, function in {
        **installed,
        "node_plan": node_plan,
        "highpass_profile": highpass_profile,
        "filtered_sum_rule_ratio": filtered_sum_rule_ratio,
        "rho_matrix": rho_matrix,
        "transfer_correction": transfer_correction,
        "cached_sum_rule_ratio": cached_sum_rule_ratio,
        "spectrum_aware_ratio": spectrum_aware_ratio,
    }.items():
        monkeypatch.setattr(module, name, function, raising=False)
    module.calls = module_calls
    module.plan_calls = plan_calls
    # kept apart from ``calls`` so the white-probe assertions stay exact
    module.sa_calls = sa_calls
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
        {"type": "fourier_highpass", "lx": 30, "spectrum_aware": True},
        {"type": "fourier_highpass", "lx": 30, "spectrum_aware": False},
        {"type": "fourier_highpass", "lx": 30, "near_cut_step": 0},
        {"type": "fourier_highpass", "lx": 30, "near_cut_step": 2.5},
        {"type": "fourier_highpass", "lx": 30, "near_cut_step": None},
        {"type": "fourier_highpass", "lx": 30, "nprobe_near_cut": 1},
        {"type": "fourier_highpass", "lx": 30, "nprobe_near_cut": None},
        {"type": "fourier_highpass", "lx": 30, "lowpass": None},
        {"type": "fourier_highpass", "lx": 30, "lowpass": {"lx": 614}},
        {"type": "fourier_highpass", "lx": 30, "lowpass": {"lx": 30.5}},
        {
            "type": "fourier_highpass",
            "lx": 30,
            "lowpass": {"lx": 614.4, "shape": "exp", "power": 6},
        },
        {
            "type": "fourier_highpass",
            "lx": 30,
            "shape": "exp",
            "lowpass": {"lx": 100, "shape": "sharp"},
        },
        {"type": "fourier_highpass", "lx": 30, "lowpass": {"lx": 90, "power": 2.5}},
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
        (
            {"type": "fourier_highpass", "lx": 30, "spectrum_aware": "yes"},
            "map_filter.spectrum_aware",
        ),
        (
            {"type": "fourier_highpass", "lx": 30, "spectrum_aware": 1},
            "map_filter.spectrum_aware",
        ),
        (
            {"type": "fourier_highpass", "lx": 30, "near_cut_step": -1},
            "map_filter.near_cut_step",
        ),
        (
            {"type": "fourier_highpass", "lx": 30, "near_cut_step": "20"},
            "map_filter.near_cut_step",
        ),
        (
            {"type": "fourier_highpass", "lx": 30, "near_cut_step": True},
            "map_filter.near_cut_step",
        ),
        (
            {"type": "fourier_highpass", "lx": 30, "near_cut_step": float("nan")},
            "map_filter.near_cut_step",
        ),
        (
            {"type": "fourier_highpass", "lx": 30, "nprobe_near_cut": 0},
            "map_filter.nprobe_near_cut",
        ),
        (
            {"type": "fourier_highpass", "lx": 30, "nprobe_near_cut": 2.5},
            "map_filter.nprobe_near_cut",
        ),
        (
            {"type": "fourier_highpass", "lx": 30, "nprobe_near_cut": True},
            "map_filter.nprobe_near_cut",
        ),
        (
            {"type": "fourier_highpass", "lx": 30, "lowpass": 614},
            "'map_filter.lowpass' must be null or a mapping",
        ),
        (
            {"type": "fourier_highpass", "lx": 30, "lowpass": [614]},
            "'map_filter.lowpass' must be null or a mapping",
        ),
        (
            {"type": "fourier_highpass", "lx": 30, "lowpass": {}},
            "'map_filter.lowpass.lx' is required",
        ),
        (
            {"type": "fourier_highpass", "lx": 30, "lowpass": {"shape": "exp"}},
            "'map_filter.lowpass.lx' is required",
        ),
        (
            {"type": "fourier_highpass", "lx": 30, "lowpass": {"lx": 30}},
            "must exceed the high-pass",
        ),
        (
            {"type": "fourier_highpass", "lx": 30, "lowpass": {"lx": 12}},
            "must exceed the high-pass",
        ),
        (
            {"type": "fourier_highpass", "lx": 30, "lowpass": {"lx": 0}},
            "map_filter.lowpass.lx",
        ),
        (
            {"type": "fourier_highpass", "lx": 30, "lowpass": {"lx": -600}},
            "map_filter.lowpass.lx",
        ),
        (
            {"type": "fourier_highpass", "lx": 30, "lowpass": {"lx": "614"}},
            "map_filter.lowpass.lx",
        ),
        (
            {"type": "fourier_highpass", "lx": 30, "lowpass": {"lx": True}},
            "map_filter.lowpass.lx",
        ),
        (
            {"type": "fourier_highpass", "lx": 30, "lowpass": {"lx": float("inf")}},
            "map_filter.lowpass.lx",
        ),
        (
            {"type": "fourier_highpass", "lx": 30, "lowpass": {"lx": float("nan")}},
            "map_filter.lowpass.lx",
        ),
        (
            {
                "type": "fourier_highpass",
                "lx": 30,
                "lowpass": {"lx": 614, "shape": "box"},
            },
            "map_filter.lowpass.shape",
        ),
        (
            {"type": "fourier_highpass", "lx": 30, "lowpass": {"lx": 614, "power": 0}},
            "map_filter.lowpass.power",
        ),
        (
            {"type": "fourier_highpass", "lx": 30, "lowpass": {"lx": 614, "power": -1}},
            "map_filter.lowpass.power",
        ),
        (
            {
                "type": "fourier_highpass",
                "lx": 30,
                "lowpass": {"lx": 614, "power": float("inf")},
            },
            "map_filter.lowpass.power",
        ),
        (
            {
                "type": "fourier_highpass",
                "lx": 30,
                "lowpass": {"lx": 614, "power": "6"},
            },
            "map_filter.lowpass.power",
        ),
        (
            {"type": "fourier_highpass", "lx": 30, "lowpass": {"lx": 614, "width": 3}},
            "Unknown key(s) ['width'] in 'map_filter.lowpass'",
        ),
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
    assert config.spectrum_aware is False
    assert config.near_cut_step is None and config.nprobe_near_cut == 128
    assert config.lowpass is None  # no low-pass unless asked for
    explicit = MapFilterConfig.from_block(
        {
            "type": "fourier_highpass",
            "lx": 30,
            "near_cut_step": 0,
            "nprobe_near_cut": 64,
        }
    )
    assert (explicit.near_cut_step, explicit.nprobe_near_cut) == (0, 64)
    assert MapFilterConfig.from_block(
        {"type": "fourier_highpass", "lx": 30, "spectrum_aware": True}
    ).spectrum_aware
    params = dict(BASE, save_dir="x", git_hash="y")
    # no block: None
    validator = ParameterValidator()
    assert validator.validate(dict(BASE))
    assert PipelineConfig.from_params(_effective(params)).map_filter is None
    with_block = dict(params, map_filter={"type": "fourier_highpass", "lx": 30})
    assert PipelineConfig.from_params(_effective(with_block)).map_filter == config


def test_lowpass_block_parsing():
    base = {"type": "fourier_highpass", "lx": 30}
    config = MapFilterConfig.from_block({**base, "lowpass": {"lx": 614.4}})
    assert config.lowpass == LowpassConfig(lx=614.4)  # exp, power 6
    assert (config.lowpass.shape, config.lowpass.power) == ("exp", 6.0)
    assert config.lowpass.as_dict() == {"lx": 614.4, "shape": "exp", "power": 6.0}
    # null is absent, and equal to an absent key
    assert MapFilterConfig.from_block({**base, "lowpass": None}) == (
        MapFilterConfig.from_block(base)
    )
    sharp = MapFilterConfig.from_block(
        {**base, "lowpass": {"lx": 100, "shape": "sharp", "power": 3}}
    ).lowpass
    assert (sharp.lx, sharp.shape) == (100, "sharp")
    # the power of the sharp shape is unused: it does not enter the identity
    assert sharp.as_dict() == {"lx": 100.0, "shape": "sharp", "power": None}
    assert LowpassConfig(100, "sharp", 9.0).as_dict() == sharp.as_dict()
    # integers are recorded as floats
    assert LowpassConfig(lx=614, power=6).as_dict() == {
        "lx": 614.0,
        "shape": "exp",
        "power": 6.0,
    }
    # through the validator and PipelineConfig
    params = dict(BASE, save_dir="x", git_hash="y")
    block = {**base, "lowpass": {"lx": 614.4, "shape": "sharp"}}
    assert ParameterValidator().validate(dict(BASE, map_filter=block))
    parsed = PipelineConfig.from_params(_effective(dict(params, map_filter=block)))
    assert parsed.map_filter.lowpass == LowpassConfig(614.4, "sharp")
    assert parsed.map_filter.lx == 30
    assert "lowpass" in MapFilterConfig.KEYS


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


def test_changed_or_dropped_lowpass_is_incompatible(tmp_path):
    block = {"type": "fourier_highpass", "lx": 30}
    banded = dict(block, lowpass={"lx": 614, "shape": "exp", "power": 6})
    manager = _manager(tmp_path, dict(BASE, map_filter=banded))
    same = dict(banded, lowpass=dict(banded["lowpass"]))
    assert manager._incompatible_parameters(dict(BASE, map_filter=same)) == []
    # the low-pass dropped, added, or changed in any key
    assert manager._incompatible_parameters(dict(BASE, map_filter=block)) == [
        "map_filter"
    ]
    for change in ({"lx": 700}, {"shape": "sharp"}, {"power": 4}):
        other = dict(block, lowpass={**banded["lowpass"], **change})
        assert manager._incompatible_parameters(dict(BASE, map_filter=other)) == [
            "map_filter"
        ], change
    manager = _manager(tmp_path, dict(BASE, map_filter=block))
    assert manager._incompatible_parameters(dict(BASE, map_filter=banded)) == [
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
#: (not read from cmbcov.assembly.filtering.BLOCK_CHANNEL).
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


#: The keys of the ratio-cache identity of a run without a low-pass.
OLD_IDENTITY_KEYS = {
    "mask_digest",
    "type",
    "lx",
    "shape",
    "power",
    "channels",
    "nodes",
    "dmax",
    "nprobe",
    "lmax_int_floor",
    "seed",
    "lw",
    "lmax",
}


def probe_profile_is_highpass_only(module, profile, lx, shape, power):
    """``profile`` is ``partial(highpass_profile, lx=, shape=, power=)``."""
    return (
        profile.func is module.highpass_profile
        and profile.args == ()
        and profile.keywords == {"lx": lx, "shape": shape, "power": power}
    )


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
    # the node plan of the cut lx = 12: nodes, per-node probes and floors
    assert fake_filtering.plan_calls == [
        {
            "lmax": 20,
            "node_min": 32,
            "ratio": 1.25,
            "lx": 12,
            "near_cut_step": None,
            "nprobe": 5,
            "nprobe_near_cut": 128,
        }
    ]
    plan = fake_filtering.node_plan(20, 32, 1.25, 12, nprobe=5, nprobe_near_cut=128)
    assert identity["nodes"] == plan.nodes == list(range(11, 20))
    assert identity["nprobe"] == plan.nprobe == [128] * len(plan.nodes)
    assert identity["lmax_int_floor"] == plan.lmax_int_floor
    assert any(plan.lmax_int_floor) and not all(plan.lmax_int_floor)
    assert identity["seed"] == 4
    assert identity["dmax"] == 19  # NKA: 64, capped at lmax - 1
    assert identity["type"] == "fourier_highpass"
    assert identity["channels"] == ["00"]  # TT-only run: the spin-0 channel
    # no low-pass: exactly the keys of the identity before the key existed, so
    # the saved ratio caches stay valid, and the high-pass profile alone
    assert set(identity) == OLD_IDENTITY_KEYS
    assert probe_profile_is_highpass_only(
        fake_filtering, fake_filtering.calls["probe"][0]["profile"], 12, "sharp", 6.0
    )

    assert len(calls["probe"]) == 1
    probe = calls["probe"][0]
    assert probe["lw"] == lw and probe["dmax"] == 19
    assert probe["nodes"] == plan.nodes
    assert probe["kwargs"]["nprobe"] == plan.nprobe
    assert probe["kwargs"]["lmax_int_floor"] == plan.lmax_int_floor
    assert probe["kwargs"]["seed"] == 4
    assert tuple(probe["kwargs"]["channels"]) == ("00",)

    assert set(seen["rho"]) == {"00"}
    assert seen["rho"]["00"].shape == (20, 20)
    assert set(seen["fl"]) == {FREQ + FREQ}
    np.testing.assert_allclose(seen["fl"][FREQ + FREQ]["TT"][:20], 0.8)

    saved = np.load(os.path.join(generator.config.save_dir, "map_filter_ratio.npz"))
    assert saved["nodes"].tolist() == identity["nodes"]
    assert saved["nprobe"].tolist() == identity["nprobe"]
    assert saved["lmax_int_floor"].tolist() == identity["lmax_int_floor"]
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


# --------------------------------------------------------------------------- #
# Spectrum-aware TT x TT (filter_rho_blocks)
# --------------------------------------------------------------------------- #
def _block_factor_inputs(keys, rng, nodes=(5, 9, 17, 31)):
    from cmbcov.assembly.filtering import block_id

    return {
        block_id(k): (list(nodes), rng.uniform(0.5, 1.5, (len(nodes), 4)))
        for k in keys.keys()
        if k.stokekey() == "TTxTT"
    }


def test_hook_tt_blocks_use_their_own_factor_without_fl(tmp_path, monkeypatch):
    """
    T+E run with ``filter_rho_blocks``: TT x TT is multiplied by
    ``rho_matrix(nodes, g, lmax)`` of its block (the real interpolation, no
    ``fl``), every other block by ``rho[ch] / (fl fl)`` exactly as without
    the block factors. ``filter_rho`` has no ``"00"``: none is needed.
    """
    from cmbcov.assembly.filtering import block_id, rho_matrix

    lmax = 32
    keys = CovKeys(["T", "E"], [FREQ])
    cl = _cl(lmax, stokes=("TT", "EE", "TE", "ET"))
    rng = np.random.default_rng(11)
    rho = {ch: rng.uniform(0.5, 1.5, (lmax, lmax)) for ch in ("20", "EE")}
    pair = FREQ + FREQ
    fl = {pair: {s: rng.uniform(0.3, 1.0, lmax) for s in ("TT", "EE", "TE")}}
    blocks = _block_factor_inputs(keys, rng)
    assert list(blocks) == [f"TTxTT_{pair}x{pair}"]

    spy = _Spy(monkeypatch)
    _run(_cov(tmp_path / "a"), keys, cl)
    raw = dict(spy.terms)
    spy.terms.clear()
    _, warned = _run(
        _cov(tmp_path / "b", filter_rho=rho, filter_fl=fl, filter_rho_blocks=blocks),
        keys,
        cl,
    )
    assert warned == []
    for (stokes, freq), term in spy.terms.items():
        if stokes == "TTxTT":
            nodes, g = blocks[f"TTxTT_{freq}"]
            expected = raw[stokes, freq] * rho_matrix(nodes, g, lmax)
        else:
            channel = TE_RUN_CHANNELS[stokes]
            s1, s2 = (x if x != "ET" else "TE" for x in stokes.split("x"))
            expected = raw[stokes, freq] * _factor(
                rho[channel], fl[pair][s1], fl[pair][s2]
            )
        np.testing.assert_allclose(term, expected, rtol=1e-13, atol=0)
        assert not np.allclose(term, raw[stokes, freq], rtol=1e-3, atol=0)

    # the white-only result for the non-TT blocks is unchanged by the block
    # factors (same raw block, same factor)
    spy.terms.clear()
    rho_all = dict(rho, **{"00": np.ones((lmax, lmax))})
    _run(_cov(tmp_path / "c", filter_rho=rho_all, filter_fl=fl), keys, cl)
    white_only = dict(spy.terms)
    spy.terms.clear()
    _run(
        _cov(tmp_path / "d", filter_rho=rho, filter_fl=fl, filter_rho_blocks=blocks),
        keys,
        cl,
    )
    for key, term in spy.terms.items():
        if key[0] != "TTxTT":
            np.testing.assert_array_equal(term, white_only[key])

    # TT-only run: the block factors alone, no filter_rho, no fl needed
    keys_t = CovKeys(["T"], ["090GHz", "150GHz"])
    cl_t = _cl(lmax, ("090GHz", "150GHz"))
    blocks_t = _block_factor_inputs(keys_t, rng)
    assert len(blocks_t) == 6
    spy.terms.clear()
    _run(_cov(tmp_path / "e"), keys_t, cl_t)
    raw_t = dict(spy.terms)
    spy.terms.clear()
    _run(_cov(tmp_path / "f", filter_rho_blocks=blocks_t), keys_t, cl_t)
    for k in keys_t.keys():
        nodes, g = blocks_t[block_id(k)]
        key = (k.stokekey(), k.freqkey())
        np.testing.assert_allclose(
            spy.terms[key], raw_t[key] * rho_matrix(nodes, g, lmax), rtol=1e-13
        )


def test_hook_partial_block_factors_need_the_white_channel(tmp_path):
    keys = CovKeys(["T"], ["090GHz", "150GHz"])
    cl = _cl(32, ("090GHz", "150GHz"))
    blocks = _block_factor_inputs(keys, np.random.default_rng(1))
    blocks.pop(next(iter(blocks)))
    cov = _cov(tmp_path, filter_rho_blocks=blocks)
    with pytest.raises(ValueError, match="does not cover every TT x TT block"):
        cov.compute_covariance_matrix(LBINS, keys, cl)


@pytest.mark.parametrize(
    "blocks, fragment",
    [
        ([1, 2], "must be a dict"),
        ({"EExEE_aaxaa": ([5], np.ones((1, 3)))}, "not a TT x TT block id"),
        ({"TTxTT_aaxaa": np.ones((1, 3))}, "pair"),
        ({"TTxTT_aaxaa": ([5, 5], np.ones((2, 3)))}, "strictly increasing"),
        ({"TTxTT_aaxaa": ([0, 5], np.ones((2, 3)))}, "strictly increasing"),
        ({"TTxTT_aaxaa": ([5.5], np.ones((1, 3)))}, "strictly increasing"),
        ({"TTxTT_aaxaa": ([5, 9], np.ones((1, 3)))}, "shape"),
        ({"TTxTT_aaxaa": ([5], np.full((1, 3), np.nan))}, "finite"),
    ],
)
def test_bad_filter_rho_blocks_are_refused(tmp_path, blocks, fragment):
    with pytest.raises(ValueError, match=fragment):
        _cov(tmp_path, filter_rho_blocks=blocks)


def test_filter_fl_with_only_block_factors_is_accepted(tmp_path):
    blocks = {"TTxTT_aaxaa": ([5], np.ones((1, 3)))}
    _cov(tmp_path, filter_rho_blocks=blocks, filter_fl={"aa": {"TT": np.ones(40)}})


def _spectrum_aware_params(workdir, polarised=False):
    path = _params(workdir, polarised=polarised)
    params = yaml.safe_load(open(path))
    params["map_filter"]["spectrum_aware"] = True
    with open(path, "w") as handle:
        yaml.dump(params, handle)
    return path


def test_generator_spectrum_aware_tt_run(tmp_path, monkeypatch, fake_filtering):
    """
    TT-only run with ``spectrum_aware``: no white probe at all, one
    spectrum-aware probe with the run's Wick terms and spectra
    (``C_pre = C B^2 pixwin + nl / fl``, ``C_used`` the covariance's
    spectrum), cached under an identity with the terms and the spectra
    digest, and handed over as ``filter_rho_blocks``.
    """
    from cmbcov.assembly.filtering import NOISE_FL_FLOOR

    path = _spectrum_aware_params(str(tmp_path))
    seen = {}
    real = Cov.compute_covariance_matrix

    def capture(cov, lbins, keys, cl, *args, **kwargs):
        seen["rho"] = cov.config.filter_rho
        seen["blocks"] = cov.config.filter_rho_blocks
        seen["cl"] = cl
        return real(cov, lbins, keys, cl, *args, **kwargs)

    monkeypatch.setattr(Cov, "compute_covariance_matrix", capture)
    generator = CovarianceMatrixGenerator(path)
    generator.run_full_analysis()

    assert fake_filtering.calls["probe"] == []  # "00" is not probed
    assert len(fake_filtering.sa_calls) == 1
    call = fake_filtering.sa_calls[0]
    pair = FREQ + FREQ
    block = f"TTxTT_{pair}x{pair}"
    assert call["terms"] == {block: [(pair, pair), (pair, pair)]}
    plan = fake_filtering.node_plan(20, 32, 1.25, 12, nprobe=5, nprobe_near_cut=128)
    assert call["nodes"] == plan.nodes
    assert call["kwargs"]["nprobe"] == plan.nprobe
    assert call["kwargs"]["lmax_int_floor"] == plan.lmax_int_floor
    assert call["kwargs"]["seed"] == 4
    assert call["dmax"] == 19

    # the spectra: pre-filter (no fl on the signal, nl / fl) and used
    sp = generator.spectra
    signal = sp.cl_dict[pair]["TT"] * sp.beams[FREQ] ** 2 * sp.pix["TT"]
    nl = sp.nl_dict_biased[pair]["TT"]
    np.testing.assert_allclose(call["cl_pre"][pair], signal + nl / 0.8, rtol=1e-14)
    assert 0.8 > NOISE_FL_FLOOR
    np.testing.assert_allclose(
        call["cl_used"][pair], seen["cl"][pair]["TT"], rtol=0, atol=0
    )
    np.testing.assert_allclose(
        call["cl_used"][pair], 0.8 * signal + nl, rtol=1e-14, atol=0
    )

    caches = fake_filtering.calls["cache"]
    assert len(caches) == 1
    identity = caches[0]["identity"]
    assert identity["kind"] == "spectrum_aware_TT"
    assert identity["terms"] == {block: [[pair, pair], [pair, pair]]}
    assert identity["channels"] == []
    assert identity["spectra_digest"] == Cov._spectra_digest(
        {f"pre_{pair}": call["cl_pre"][pair], f"used_{pair}": call["cl_used"][pair]}
    )
    for key in ("mask_digest", "lx", "shape", "power", "dmax", "seed", "lw"):
        assert key in identity
    assert identity["nodes"] == plan.nodes
    assert identity["nprobe"] == plan.nprobe
    assert identity["lmax_int_floor"] == plan.lmax_int_floor
    assert identity["noise_fl_floor"] == NOISE_FL_FLOOR
    assert identity["oversample"] == call["kwargs"]["oversample"] == 2.0

    assert seen["rho"] is None
    assert list(seen["blocks"]) == [block]
    nodes, g = seen["blocks"][block]
    assert nodes == identity["nodes"] and np.all(g == 0.25)

    saved = np.load(os.path.join(generator.config.save_dir, "map_filter_ratio.npz"))
    assert saved["channels"].tolist() == []
    assert saved["spectrum_aware_blocks"].tolist() == [block]
    np.testing.assert_array_equal(saved["g_block_0"], g)


def test_generator_spectrum_aware_te_run_probes_white_without_00(
    tmp_path, monkeypatch, fake_filtering
):
    path = _spectrum_aware_params(str(tmp_path), polarised=True)
    seen = _capture_filter(monkeypatch)
    generator = CovarianceMatrixGenerator(path)
    generator.run_full_analysis()
    probe = fake_filtering.calls["probe"]
    assert len(probe) == 1
    assert tuple(probe[0]["kwargs"]["channels"]) == ("20", "EE")
    caches = fake_filtering.calls["cache"]
    assert [c["identity"]["channels"] for c in caches] == [["20", "EE"], []]
    assert "terms" not in caches[0]["identity"]
    assert set(seen["rho"]) == {"20", "EE"}
    assert len(fake_filtering.sa_calls) == 1
    assert list(fake_filtering.sa_calls[0]["cl_pre"]) == [FREQ + FREQ]
    saved = np.load(os.path.join(generator.config.save_dir, "map_filter_ratio.npz"))
    assert saved["channels"].tolist() == ["20", "EE"]
    assert saved["spectrum_aware_blocks"].tolist() == [
        f"TTxTT_{FREQ}{FREQ}x{FREQ}{FREQ}"
    ]


def test_generator_spectra_change_changes_the_identity(tmp_path, fake_filtering):
    digests = []
    for scale in (1.0, 1.1):
        workdir = tmp_path / str(scale)
        workdir.mkdir()
        path = _spectrum_aware_params(str(workdir))
        cls = np.loadtxt(os.path.join(str(workdir), "cls.dat"))
        cls[:, 1:] *= scale
        np.savetxt(os.path.join(str(workdir), "cls.dat"), cls)
        CovarianceMatrixGenerator(path).run_full_analysis()
        digests.append(fake_filtering.calls["cache"][-1]["identity"]["spectra_digest"])
    assert digests[0] != digests[1]


def test_generator_lowpass_reaches_the_profile_and_the_identity(
    tmp_path, fake_filtering
):
    """
    ``map_filter.lowpass`` makes the run's profile the combined one and adds
    a ``lowpass`` entry to the cache identity; every key of the low-pass
    changes the identity (the power of the sharp shape excepted, it is
    unused), and none of them changes the nodes or the probe settings.
    """
    cases = (
        None,
        {"lx": 18},
        {"lx": 18, "shape": "exp", "power": 6},  # = the previous one
        {"lx": 18, "shape": "exp", "power": 4},
        {"lx": 24},
        {"lx": 18, "shape": "sharp"},
        {"lx": 18, "shape": "sharp", "power": 3},  # power unused: = the previous
    )
    identities, profiles = [], []
    for i, lowpass in enumerate(cases):
        workdir = tmp_path / str(i)
        workdir.mkdir()
        path = _params(str(workdir))
        if lowpass is not None:
            _edit_params(path, lowpass=lowpass)
        CovarianceMatrixGenerator(path).run_full_analysis()
        identities.append(fake_filtering.calls["cache"][-1]["identity"])
        profiles.append(fake_filtering.calls["probe"][-1]["profile"])

    # absent: the old identity, the high-pass profile, the old plan call
    assert set(identities[0]) == OLD_IDENTITY_KEYS
    assert set(fake_filtering.plan_calls[0]) == {
        "lmax",
        "node_min",
        "ratio",
        "lx",
        "near_cut_step",
        "nprobe",
        "nprobe_near_cut",
    }
    assert probe_profile_is_highpass_only(fake_filtering, profiles[0], 12, "sharp", 6.0)
    # present: the extra entry (floats and a string, plain JSON values), the
    # old entries unchanged, the combined profile with the same high-pass
    for i, (identity, profile, lowpass) in enumerate(
        zip(identities[1:], profiles[1:], cases[1:]), start=1
    ):
        assert set(identity) == OLD_IDENTITY_KEYS | {"lowpass"}
        assert {k: v for k, v in identity.items() if k != "lowpass"} == identities[0]
        shape = lowpass.get("shape", "exp")
        power = float(lowpass.get("power", 6)) if shape == "exp" else None
        assert identity["lowpass"] == {
            "lx": float(lowpass["lx"]),
            "shape": shape,
            "power": power,
        }
        assert profile.func is fake_filtering.combined_profile
        assert profile.args == ()
        assert profile.keywords == {
            "lx": 12,
            "shape": "sharp",
            "power": 6.0,
            "lowpass": identity["lowpass"],
        }
        # the node plan was asked for the low-pass
        assert fake_filtering.plan_calls[i]["lx_lowpass"] == float(lowpass["lx"])
        assert fake_filtering.plan_calls[i]["lowpass_shape"] == shape
    # the identities (hence the cache files) differ exactly when the filter does
    keys = [json.dumps(i, sort_keys=True) for i in identities]
    assert keys[2] == keys[1] and keys[6] == keys[5]
    assert len({keys[0], keys[1], keys[3], keys[4], keys[5]}) == 5
    # dmax, lw and the channels do not depend on the low-pass; this run's
    # lmax of 20 is below every low-pass range here, so neither do the nodes
    for identity in identities[1:]:
        for key in ("nodes", "nprobe", "lmax_int_floor", "dmax", "lw", "channels"):
            assert identity[key] == identities[0][key]


def test_generator_lowpass_reaches_the_spectrum_aware_probes(tmp_path, fake_filtering):
    """The spectrum-aware identity extends the white one, so it carries the
    low-pass, and its probes get the same combined profile."""
    path = _params(str(tmp_path))
    _edit_params(path, spectrum_aware=True, lowpass={"lx": 18, "shape": "sharp"})
    CovarianceMatrixGenerator(path).run_full_analysis()
    identity = fake_filtering.calls["cache"][-1]["identity"]
    assert identity["kind"] == "spectrum_aware_TT"
    assert identity["lowpass"] == {"lx": 18.0, "shape": "sharp", "power": None}
    profile = fake_filtering.sa_calls[-1]["profile"]
    assert profile.func is fake_filtering.combined_profile
    assert profile.keywords["lowpass"] == identity["lowpass"]
    # without the low-pass the spectrum-aware identity has no such entry
    other = tmp_path / "plain"
    other.mkdir()
    path = _params(str(other))
    _edit_params(path, spectrum_aware=True)
    CovarianceMatrixGenerator(path).run_full_analysis()
    assert "lowpass" not in fake_filtering.calls["cache"][-1]["identity"]
    assert probe_profile_is_highpass_only(
        fake_filtering, fake_filtering.sa_calls[-1]["profile"], 12, "sharp", 6.0
    )


@pytest.mark.parametrize(
    "lowpass, step, sharp_warning, step_warning",
    [
        (None, None, False, False),
        (None, 0, False, False),  # no low-pass: nothing to warn about
        ({"lx": 18, "shape": "sharp"}, None, True, False),  # cut inside lmax = 20
        ({"lx": 20, "shape": "sharp"}, None, True, False),  # lx <= lmax
        ({"lx": 100, "shape": "sharp"}, None, False, False),  # cut beyond lmax
        ({"lx": 18, "shape": "exp"}, None, False, False),
        ({"lx": 18, "shape": "exp"}, 0, False, True),  # 0.6 * 18 < 20
        ({"lx": 18, "shape": "sharp"}, 0, True, True),
        ({"lx": 100, "shape": "exp"}, 0, False, False),  # 0.6 * 100 > 20
        ({"lx": 18, "shape": "exp"}, 2, False, False),  # only step 0 switches off
    ],
)
def test_lowpass_log_warnings(
    tmp_path, fake_filtering, caplog, lowpass, step, sharp_warning, step_warning
):
    """The sharp low-pass inside the covariance, and ``near_cut_step: 0`` with
    the rolloff inside it, each log one warning per run."""
    path = _params(str(tmp_path))
    extra = {} if lowpass is None else {"lowpass": lowpass}
    if step is not None:
        extra["near_cut_step"] = step
    _edit_params(path, **extra)
    with caplog.at_level("WARNING"):
        CovarianceMatrixGenerator(path).run_full_analysis()
    messages = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    sharp = [m for m in messages if "sharp low-pass cut" in m]
    nodes = [m for m in messages if "near_cut_step is 0" in m]
    assert len(sharp) == int(sharp_warning), messages
    assert len(nodes) == int(step_warning), messages
    if sharp:
        assert "3-4% off" in sharp[0] and "kink" in sharp[0]
    if nodes:
        assert "geometric nodes alone" in nodes[0]


def _real_run(workdir, lowpass, kernel_dir=None):
    """A real (unfaked) lmax-20 run; the ratio file it saves."""
    path = _params(str(workdir), polarised=True)
    _edit_params(path, nprobe=7, **({} if lowpass is None else {"lowpass": lowpass}))
    if kernel_dir is not None:
        params = yaml.safe_load(open(path))
        params["cov_path"] = str(kernel_dir)  # the ratio cache lives there
        with open(path, "w") as handle:
            yaml.dump(params, handle)
    generator = CovarianceMatrixGenerator(path)
    generator.run_full_analysis()
    with np.load(os.path.join(generator.config.save_dir, "map_filter_ratio.npz")) as z:
        return generator, {k: z[k] for k in z.files}


def test_real_run_lowpass(tmp_path):
    """
    End to end with the real probing (T and E blocks, all three channels):

    * no low-pass, and a low-pass so far above the band that it is 1 in
      float64, give the same ratios bit for bit, but a different cache file
      (the identity says what was asked for);
    * a low-pass inside the band lowers the ratios at the high multipoles;
    * a repeated run reads its ratios from the cache (no new file).
    """
    kernels = tmp_path / "kernels"
    runs = {}
    for name, lowpass in (
        ("plain", None),
        ("huge", {"lx": 1e9, "shape": "sharp"}),
        ("band", {"lx": 16, "shape": "exp"}),
        ("band again", {"lx": 16, "shape": "exp"}),
    ):
        workdir = tmp_path / name.replace(" ", "_")
        workdir.mkdir()
        runs[name] = _real_run(workdir, lowpass, kernels)[1]
    files = os.listdir(kernels / "filtered_sum_rule")
    assert len(files) == 3  # plain, huge, band; the repeat was a cache hit
    plain, huge, band = runs["plain"], runs["huge"], runs["band"]
    for ch in plain["channels"]:
        assert np.array_equal(huge[f"g_{ch}"], plain[f"g_{ch}"]), ch
        assert np.array_equal(runs["band again"][f"g_{ch}"], band[f"g_{ch}"]), ch
        assert np.all(band[f"g_{ch}"][-1] < plain[f"g_{ch}"][-1]), ch
        assert np.all(band[f"g_{ch}"] > 0)


# --------------------------------------------------------------------------- #
# Near-cut settings and the low-transfer-function warning
# --------------------------------------------------------------------------- #
def _edit_params(path, **block):
    params = yaml.safe_load(open(path))
    params["map_filter"].update(block)
    with open(path, "w") as handle:
        yaml.dump(params, handle)
    return path


@pytest.mark.parametrize("spectrum_aware", [False, True])
def test_generator_honours_near_cut_settings(tmp_path, fake_filtering, spectrum_aware):
    """``near_cut_step: 0`` and ``nprobe_near_cut`` reach the node plan, and
    its nodes, probes and floors reach the probing call and the identity,
    on the white and the spectrum-aware path."""
    identities, calls = [], []
    for i, block in enumerate(
        ({}, {"near_cut_step": 0}, {"nprobe_near_cut": 7}, {"near_cut_step": 2})
    ):
        workdir = tmp_path / str(i)
        workdir.mkdir()
        path = _params(str(workdir))
        _edit_params(path, spectrum_aware=spectrum_aware, **block)
        CovarianceMatrixGenerator(path).run_full_analysis()
        plan_call = fake_filtering.plan_calls[-1]
        assert plan_call["near_cut_step"] == block.get("near_cut_step")
        assert plan_call["nprobe_near_cut"] == block.get("nprobe_near_cut", 128)
        identities.append(fake_filtering.calls["cache"][-1]["identity"])
        calls.append(
            fake_filtering.sa_calls[-1]
            if spectrum_aware
            else fake_filtering.calls["probe"][-1]
        )
    default, old, seven, step2 = identities
    assert old["nodes"] == [19]  # the geometric list of lmax 20
    assert old["nprobe"] == [128]  # 19 is below 2 lx: the near-cut default
    assert default["nodes"] == list(range(11, 20))
    assert seven["nodes"] == default["nodes"] and seven["nprobe"] == [7] * 9
    assert step2["nodes"] == [11, 13, 15, 17, 19]
    for identity, call in zip(identities, calls):
        assert call["nodes"] == identity["nodes"]
        assert call["kwargs"]["nprobe"] == identity["nprobe"]
        assert call["kwargs"]["lmax_int_floor"] == identity["lmax_int_floor"]
    # every setting changes the cache identity
    keys = [yaml.safe_dump(i, sort_keys=True) for i in identities]
    assert len(set(keys)) == 4


def _low_fl_params(workdir, map_filter=True, lmin=2, polarised=False):
    path = _params(workdir, polarised=polarised, lmin=lmin)
    ell = np.arange(80)
    fl = np.where(ell < 6, 0.05, np.where(ell < 9, 0.2, 0.9))
    columns = [fl, np.full(80, 0.9), fl, np.full(80, 0.9)] if polarised else [fl]
    if polarised:  # EE: 0.25 from 14 to 15 only
        columns[1] = np.where((ell >= 14) & (ell < 16), 0.25, 0.9)
    np.savetxt(
        os.path.join(workdir, f"fl_{FREQ}{FREQ}.dat"), np.column_stack([ell, *columns])
    )
    if not map_filter:
        params = yaml.safe_load(open(path))
        del params["map_filter"]
        with open(path, "w") as handle:
            yaml.dump(params, handle)
    return path


def _fl_warnings(caplog):
    messages = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    degraded = [m for m in messages if "accuracy is degraded" in m]
    cut = [m for m in messages if "cut these bins" in m]
    return degraded, cut


def test_low_transfer_function_warnings(tmp_path, fake_filtering, caplog):
    """Bins [2, 20, 3]: 2-4, 5-7, 8-10, ...; fl 0.05 below 6, 0.2 below 9."""
    from cmbcov.assembly.filtering import FL_CUT, FL_DEGRADED

    assert (FL_DEGRADED, FL_CUT) == (0.3, 0.1)
    path = _low_fl_params(str(tmp_path))
    with caplog.at_level("WARNING"):
        CovarianceMatrixGenerator(path).run_full_analysis()
    degraded, cut = _fl_warnings(caplog)
    assert len(degraded) == 1 and len(cut) == 1
    pair = FREQ + FREQ
    assert f"TT {pair}: 2-4 (0.050), 5-7 (0.050), 8-10 (0.200)" in degraded[0]
    assert "11-13" not in degraded[0]
    assert f"TT {pair}: 2-4 (0.050), 5-7 (0.050)" in cut[0]
    assert "8-10" not in cut[0]

    # lmin truncates the first bin; the polarised run names each spectrum
    caplog.clear()
    os.makedirs(tmp_path / "p")
    path = _low_fl_params(str(tmp_path / "p"), lmin=3, polarised=True)
    with caplog.at_level("WARNING"):
        CovarianceMatrixGenerator(path).run_full_analysis()
    degraded, cut = _fl_warnings(caplog)
    assert f"TT {pair}: 3-4 (0.050), 5-7 (0.050), 8-10 (0.200)" in degraded[0]
    assert f"EE {pair}: 14-16 (0.250)" in degraded[0]
    assert "EE" not in cut[0] and "TE" not in degraded[0]
    assert "BB" not in degraded[0]  # low BB fl: B blocks are not corrected


def test_no_transfer_function_warning_when_fl_is_high_or_no_filter(
    tmp_path, fake_filtering, caplog
):
    with caplog.at_level("WARNING"):
        CovarianceMatrixGenerator(_params(str(tmp_path))).run_full_analysis()
    assert _fl_warnings(caplog) == ([], [])
    caplog.clear()
    workdir = tmp_path / "nofilter"
    workdir.mkdir()
    with caplog.at_level("WARNING"):
        CovarianceMatrixGenerator(
            _low_fl_params(str(workdir), map_filter=False)
        ).run_full_analysis()
    assert _fl_warnings(caplog) == ([], [])


def test_bin_transfer_minimum():
    from cmbcov.assembly.filtering import bin_transfer_minimum

    fl = np.arange(11) / 10.0  # fl[l] = l / 10
    fl[7] = np.nan
    assert bin_transfer_minimum([0, 3, 6, 9, 12], fl) == [
        (0, 2, 0.0),
        (3, 5, 0.3),
        (6, 8, 0.6),
        (9, 10, 0.9),
    ]
    assert bin_transfer_minimum([0, 3, 6, 9], fl, lmin=4, lmax=8) == [
        (4, 5, 0.4),
        (6, 7, 0.6),
    ]
    assert bin_transfer_minimum([0, 3], fl, lmin=5) == []
