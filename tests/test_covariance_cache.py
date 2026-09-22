"""
Tests for the raw per-block covariance cache (``CovarianceConfig.save_raw_blocks``).

The cache is opt-in and keyed: with ``save_raw_blocks`` each block is
written into ``save_dir`` as ``cov_<stokes>_<freq>.npy`` with a manifest of
everything it depends on, and reused only on an exact manifest match, so a
recompute with new spectra and the same ``save_dir`` (default ``"./"``)
cannot silently return a stale covariance.
"""

import json
import os
import warnings

import numpy as np
import pytest

from cmbcov.approximations import StrategyFactory
from cmbcov.approximations.acc import precompute_acc_kernels
from cmbcov.covariance import Cov, CovarianceConfig, CovarianceMethod
from cmbcov.keys import CovKeys

DATA = os.path.join(os.path.dirname(__file__), "data")
FREQ = "090GHz"
LBINS = [2, 10, 20, 30]


def _cl(size, scale=1.0):
    ell = np.arange(size)
    tt = np.zeros(size)
    tt[2:] = scale * 1e-3 / (ell[2:] * (ell[2:] + 1)) ** 0.9
    return {FREQ + FREQ: {"TT": tt}}


def _cov(save_dir, save_raw_blocks, lmax=32):
    config = CovarianceConfig(
        method=CovarianceMethod.NKA, lmax=lmax, save_raw_blocks=save_raw_blocks
    )
    return Cov(
        "baseline_mask.fits",
        config=config,
        mask_path=os.path.abspath(DATA),
        save_dir=str(save_dir),
    )


def _counting(monkeypatch):
    """Count calls to every strategy's compute_covariance_term."""
    calls = []
    real = StrategyFactory.create_strategy

    def create(cov):
        strategy = real(cov)
        inner = strategy.compute_covariance_term

        def counted(*args, **kwargs):
            calls.append(1)
            return inner(*args, **kwargs)

        strategy.compute_covariance_term = counted
        return strategy

    monkeypatch.setattr(StrategyFactory, "create_strategy", staticmethod(create))
    return calls


def _run(cov, cl):
    keys = CovKeys(["T"], [FREQ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cov.compute_covariance_matrix(LBINS, keys, cl)[2]


def _block_files(directory):
    return sorted(n for n in os.listdir(directory) if n.startswith("cov_"))


def test_default_writes_and_reads_nothing(tmp_path, monkeypatch):
    first = _run(_cov(tmp_path, False), _cl(32))
    assert _block_files(tmp_path) == []

    # A stale, manifest-less block under the old name is not reloaded either.
    stale = tmp_path / "cov_TTxTT_090GHz090GHzx090GHz090GHz.npy"
    np.save(stale, np.full((32, 32), 3.0))
    calls = _counting(monkeypatch)
    again = _run(_cov(tmp_path, False), _cl(32))
    assert calls == [1]
    np.testing.assert_array_equal(again, first)


def test_opt_in_writes_a_block_with_its_manifest_and_reuses_it(tmp_path, monkeypatch):
    first = _run(_cov(tmp_path, True), _cl(32))
    names = _block_files(tmp_path)
    assert names == [
        "cov_TTxTT_090GHz090GHzx090GHz090GHz.npy",
        "cov_TTxTT_090GHz090GHzx090GHz090GHz.npy.manifest.json",
    ]
    manifest = json.load(open(tmp_path / names[1]))
    assert manifest["method"] == "nka"
    assert manifest["lmax"] == 32
    assert len(manifest["spectra_digest"]) == 32

    calls = _counting(monkeypatch)
    again = _run(_cov(tmp_path, True), _cl(32))
    assert calls == []  # reused
    np.testing.assert_array_equal(again, first)


def test_changed_spectrum_value_is_recomputed_not_reused(tmp_path, monkeypatch):
    _run(_cov(tmp_path, True), _cl(32))
    cl = _cl(32)
    cl[FREQ + FREQ]["TT"][17] *= 1.0 + 1e-9  # one value, barely
    calls = _counting(monkeypatch)
    changed = _run(_cov(tmp_path, True), cl)
    assert calls == [1]
    # and the block on disk now describes the new spectra
    calls.clear()
    np.testing.assert_array_equal(_run(_cov(tmp_path, True), cl), changed)
    assert calls == []


def test_changed_lmax_is_recomputed(tmp_path, monkeypatch):
    _run(_cov(tmp_path, True, lmax=32), _cl(32))
    calls = _counting(monkeypatch)
    _run(_cov(tmp_path, True, lmax=40), _cl(40))
    assert calls == [1]


def test_manifest_less_block_is_not_reused_and_is_overwritten(tmp_path, monkeypatch):
    reference = _run(_cov(tmp_path / "ref", False), _cl(32))
    stale = tmp_path / "cov_TTxTT_090GHz090GHzx090GHz090GHz.npy"
    np.save(stale, np.full((32, 32), 3.0))
    calls = _counting(monkeypatch)
    got = _run(_cov(tmp_path, True), _cl(32))
    assert calls == [1]
    np.testing.assert_array_equal(got, reference)
    assert not np.all(np.load(stale) == 3.0)
    assert os.path.exists(str(stale) + ".manifest.json")


def test_binary_cache_round_trips_exactly(tmp_path):
    """np.save preserves the bits; entries are ~1e-18."""
    rng = np.random.default_rng(0)
    block = rng.standard_normal((64, 64)) * 1e-18
    cov = Cov.__new__(Cov)
    cov.save_dir = str(tmp_path)

    class _Key:
        def stokekey(self):
            return "TTxTT"

        def freqkey(self):
            return "090GHz090GHzx090GHz090GHz"

    manifest = {"x": 1}
    cov._save_raw_block(_Key(), block, manifest)
    assert np.load(cov._covariance_cache_path(_Key())).tobytes() == block.tobytes()


# --------------------------------------------------------------------------- #
# ACC: dmax and the kernel cache are part of the key
# --------------------------------------------------------------------------- #
ELL = 16
NSIDE = 16
LMAX_ACC = 48
#: ACC reads the spectra to lmax + max(0, 2 NSIDE - 1 - ELL).
LMAX_ACC_INT = LMAX_ACC + 2 * NSIDE - 1 - ELL


@pytest.fixture(scope="module")
def acc_kernels(tmp_path_factory):
    kernel_dir = tmp_path_factory.mktemp("acc_kernels")
    cov = _acc_cov(kernel_dir, kernel_dir, dmax=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        precompute_acc_kernels(
            cov.wlm,
            str(kernel_dir),
            centralell=ELL,
            dmax=3,
            nside=NSIDE,
            grid="gl",
            lw=10,
            spectra=("TT",),
        )
    return kernel_dir


def _acc_cov(save_dir, kernel_dir, dmax):
    config = CovarianceConfig(
        method=CovarianceMethod.ACC,
        lmax=LMAX_ACC,
        dmax=dmax,
        centralell=ELL,
        save_raw_blocks=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Cov(
            "baseline_mask.fits",
            config=config,
            mask_path=os.path.abspath(DATA),
            save_dir=str(save_dir),
            acc_kernel_dir=str(kernel_dir),
        )


def test_acc_changed_dmax_is_recomputed(tmp_path, acc_kernels, monkeypatch):
    cl = _cl(LMAX_ACC_INT)
    _run(_acc_cov(tmp_path, acc_kernels, dmax=2), cl)
    calls = _counting(monkeypatch)
    _run(_acc_cov(tmp_path, acc_kernels, dmax=2), cl)
    assert calls == []
    _run(_acc_cov(tmp_path, acc_kernels, dmax=3), cl)
    assert calls == [1]


def test_acc_rewritten_kernel_file_is_recomputed(tmp_path, acc_kernels, monkeypatch):
    cl = _cl(LMAX_ACC_INT)
    _run(_acc_cov(tmp_path, acc_kernels, dmax=2), cl)
    kernel = os.path.join(acc_kernels, "covariance_coupling", f"TTxTT_{ELL}x{ELL}.npy")
    stat = os.stat(kernel)
    os.utime(kernel, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    calls = _counting(monkeypatch)
    _run(_acc_cov(tmp_path, acc_kernels, dmax=2), cl)
    assert calls == [1]


def test_acc_changed_spectrum_value_past_lmax_is_recomputed(
    tmp_path, acc_kernels, monkeypatch
):
    """The manifest digests the spectra as ACC reads them, i.e. to lmax_int."""
    cl = _cl(LMAX_ACC_INT)
    _run(_acc_cov(tmp_path, acc_kernels, dmax=2), cl)
    manifest = json.load(
        open(tmp_path / "cov_TTxTT_090GHz090GHzx090GHz090GHz.npy.manifest.json")
    )
    assert manifest["lmax_int"] == LMAX_ACC_INT
    cl[FREQ + FREQ]["TT"][LMAX_ACC_INT - 1] *= 1.5  # in the padding, read by ACC
    calls = _counting(monkeypatch)
    _run(_acc_cov(tmp_path, acc_kernels, dmax=2), cl)
    assert calls == [1]
