"""
The batched ACC assembly (``ACCStrategy.compute_covariance_terms``, used by
``Cov.compute_covariance_matrix``) against the per-block assembly it
replaced (``tests/reference/acc_unbatched.py``, the code of 0990d00): bit for
bit, not to a tolerance.

The batched form computes every distinct kernel product of a batch of
blocks once per diagonal, and every left factor ``w1 @ Theta`` once, with the
same operations on the same operands as the per-block form, and sums each
block's terms in the same order; so the numbers must be identical, whatever
the batch size, the grouping of blocks, the chunking of multipoles or the
order in which blocks are yielded. Checked on

- random non-symmetric kernels (``Theta^{qp} = (Theta^{pq})^T``, one of them
  unit-sum), a non-symmetric ``norm_Xi``, three frequencies, with ``TE`` and
  ``ET`` of a pair equal in value but distinct arrays, and one array shared
  by two keys (content identity, not ``id()``, decides what is shared);
- the rippled cap of ``tests/reference/acc_level1.py`` (GL kernels, ``l* =
  8``, ``dmax = 4``): a two-frequency T/E run, and two-frequency runs with the
  six observables on a 45-pair cache (both orientations of every block,
  ``C^TB = C^EB = 0`` and not);
- ``Cov.compute_covariance_matrix`` end to end, PolSpice transform on and
  off, ``sum_asymmetric_stokes`` on and off.

Nothing is kept between calls: a second call with spectra changed in place
(same arrays, new values), or a second run with other spectra, gives the
per-block result for the new spectra.
"""

import itertools
import os
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
from reference.acc_unbatched import reference_block  # noqa: E402

from cmbcov.approximations import StrategyFactory
from cmbcov.approximations import acc as acc_module
from cmbcov.approximations.acc import (
    COUPLING_SPECTRA,
    ACCStrategy,
    _group_blocks,
    acc_internal_lmax,
)
from cmbcov.approximations.base import CovarianceStrategy
from cmbcov.covariance import Cov, CovarianceConfig, CovarianceMethod
from cmbcov.keys import CovKeys

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def _batched(strategy, keys, cl):
    out = {}
    for key, block in _quiet(lambda: list(strategy.compute_covariance_terms(keys, cl))):
        assert key not in out, f"{key} yielded twice"
        out[key] = block
    assert set(out) == set(keys)
    return out


def _assert_bit_identical(strategy, keys, cl, batched):
    for key in keys:
        reference = _quiet(reference_block, strategy, key, cl)
        assert np.array_equal(
            batched[key], reference
        ), f"{key}: max |diff| {np.max(np.abs(batched[key] - reference)):.3e}"


# --------------------------------------------------------------------------- #
# Random kernels, three frequencies
# --------------------------------------------------------------------------- #

FAKE_FREQS = ["f1", "f2", "f3"]
OBSERVABLE_SPECTRA = ("TT", "TE", "ET", "EE")


class _FakeKernelStrategy(ACCStrategy):
    """ACCStrategy with in-memory kernels instead of the on-disk cache."""

    def __init__(self, cov, kernels):
        super().__init__(cov)
        self._fake_kernels = kernels

    def get_covariance_coupling(self, ell, ell_prime, pairs=None):
        table = self._fake_kernels[ell_prime - ell]
        return {pair: table[pair] for pair in pairs}


def _fake_inputs(seed, lmax=70, centralell=30, size=48, dmax=5):
    rng = np.random.default_rng(seed)
    kernels = {}
    for offset in range(dmax):
        table = {}
        for s1, s2 in itertools.product(COUPLING_SPECTRA, repeat=2):
            table[s1, s2] = rng.standard_normal((size, size)) + 0.3
        for s1, s2 in itertools.combinations_with_replacement(COUPLING_SPECTRA, 2):
            if s1 == s2:
                table[s1, s1] = 0.5 * (table[s1, s1] + table[s1, s1].T)
            else:
                table[s2, s1] = table[s1, s2].T
        kernels[offset] = table
    kernels[0]["TT", "TT"] = kernels[0]["TT", "TT"] / kernels[0]["TT", "TT"].sum()
    norm_xi = {
        pair: 0.5 + rng.random((lmax, lmax))
        for pair in itertools.product(OBSERVABLE_SPECTRA, repeat=2)
    }
    n = acc_internal_lmax(lmax, size, centralell) + 3
    cl = {}
    for fa, fb in itertools.product(FAKE_FREQS, repeat=2):
        cl[fa + fb] = {s: rng.standard_normal(n) for s in OBSERVABLE_SPECTRA}
    # TE and ET of one pair equal in value, distinct arrays (as a survey run's
    # data model gives them); and one array under two keys.
    cl["f1f2"]["ET"] = cl["f1f2"]["TE"].copy()
    cl["f2f3"]["EE"] = cl["f1f3"]["EE"]
    config = SimpleNamespace(
        method=CovarianceMethod.ACC, dmax=dmax, centralell=centralell
    )
    cov = SimpleNamespace(lmax=lmax, config=config, norm_Xi=norm_xi)
    return _FakeKernelStrategy(cov, kernels), cl


@pytest.mark.parametrize("batch_blocks", [1, 5, 1000])
@pytest.mark.parametrize("chunk", [None, 7])
def test_random_kernels_three_frequencies(monkeypatch, batch_blocks, chunk):
    strategy, cl = _fake_inputs(11)
    keys = list(CovKeys(["T", "E"], FAKE_FREQS).keys())
    lmax, dmax = strategy.cov.lmax, strategy.cov.config.dmax
    # the budget in blocks: (2 dmax - 1 + products) x lmax doubles each, <= 4
    # products per T/E block
    per_block = (2 * dmax - 1 + 4) * lmax * 8
    monkeypatch.setattr(acc_module, "_ASSEMBLY_BATCH_BYTES", batch_blocks * per_block)
    if chunk is not None:
        size = strategy._fake_kernels[0]["TT", "TT"].shape[0]
        monkeypatch.setattr(acc_module, "_ASSEMBLY_CHUNK_BYTES", chunk * 24 * size)
    batches = []
    real = ACCStrategy._assemble_batch

    def spy(self, batch, kind):
        batches.append(len(batch))
        return real(self, batch, kind)

    monkeypatch.setattr(ACCStrategy, "_assemble_batch", spy)
    batched = _batched(strategy, keys, cl)
    assert sum(batches) == len(keys)
    if batch_blocks == 5:
        assert len(batches) > 1
    if batch_blocks == 1000:
        assert batches == [len(keys)]
    _assert_bit_identical(strategy, keys, cl, batched)


def test_single_block_call_is_the_reference():
    strategy, cl = _fake_inputs(12)
    for key in CovKeys(["T", "E"], FAKE_FREQS).keys():
        got = _quiet(strategy.compute_covariance_term, key, cl)
        assert np.array_equal(got, _quiet(reference_block, strategy, key, cl))


def test_spectra_changed_in_place_are_not_served_from_a_previous_call():
    strategy, cl = _fake_inputs(13)
    keys = list(CovKeys(["T", "E"], FAKE_FREQS).keys())
    first = _batched(strategy, keys, cl)
    for spectra in cl.values():
        for array in spectra.values():
            array *= 1.5
            array += 0.25
    second = _batched(strategy, keys, cl)
    _assert_bit_identical(strategy, keys, cl, second)
    assert not any(np.array_equal(first[k], second[k]) for k in keys)


def test_group_blocks_partitions_and_shares_left_factors():
    factors = [{("a", 1), ("b", 1)}, {("c", 1)}, {("a", 1)}, {("c", 1), ("d", 1)}]
    assert _group_blocks(factors, 10) == [[0, 1, 2, 3]]
    groups = _group_blocks(factors, 2)
    assert sorted(i for g in groups for i in g) == [0, 1, 2, 3]
    assert all(len(g) <= 2 for g in groups)
    # blocks sharing factors land together, independent of listing order
    assert sorted(map(sorted, groups)) == [[0, 2], [1, 3]]
    assert _group_blocks(factors, 2) == groups  # deterministic
    assert _group_blocks([], 3) == []


# --------------------------------------------------------------------------- #
# Real GL kernels on the rippled cap
# --------------------------------------------------------------------------- #

hp = pytest.importorskip("healpy")

from reference.acc_level1 import (  # noqa: E402
    CENTRALELL,
    DMAX,
    LMAX,
    LW,
    NSIDE_ACC,
    level1_spectra,
    rippled_cap_bandlimited,
)

B_OBS = ["TT", "EE", "BB", "TE", "TB", "EB"]
FREQS = ["f1", "f2"]


def _precompute(work, pairs=None):
    from cmbcov.approximations.acc import precompute_acc_kernels

    hp.write_map(
        os.path.join(work, "mask.fits"),
        rippled_cap_bandlimited(),
        overwrite=True,
        dtype=np.float64,
    )
    precompute_acc_kernels(
        "mask.fits",
        work,
        centralell=CENTRALELL,
        dmax=DMAX,
        mask_path=work,
        nside=NSIDE_ACC,
        grid="gl",
        lw=LW,
        pairs=pairs,
    )


def _cov(work, **config):
    config = CovarianceConfig(
        method=CovarianceMethod.ACC,
        lmax=LMAX,
        lmin=2,
        dmax=DMAX,
        centralell=CENTRALELL,
        **config,
    )
    return _quiet(Cov, "mask.fits", config=config, mask_path=work, save_dir=work)


@pytest.fixture(scope="module")
def te_work(tmp_path_factory):
    work = str(tmp_path_factory.mktemp("batched_te"))
    _quiet(_precompute, work)
    return work


@pytest.fixture(scope="module")
def b_work(tmp_path_factory):
    from cmbcov.bmode_wick import required_kernel_pairs

    work = str(tmp_path_factory.mktemp("batched_b"))
    pairs = sorted(required_kernel_pairs(B_OBS, parity_odd_nonzero=True))
    assert len(pairs) == 45
    _quiet(_precompute, work, pairs)
    return work


def _b_spectra(odd, scale=1.0, n=80):
    ell = np.arange(n)
    tt, ee, bb = np.zeros(n), np.zeros(n), np.zeros(n)
    tt[2:] = 1.0 / (ell[2:] * (ell[2:] + 1)) ** 0.8
    ee[2:] = 0.3 / (ell[2:] + 3.0) ** 1.7 * (1 + 0.5 * np.cos(ell[2:] / 2.0))
    bb[2:] = 0.05 / (ell[2:] + 1.0) ** 0.9 * (1 + 0.4 * np.sin(ell[2:] / 1.3))
    te = 0.6 * np.sqrt(tt * ee) * np.cos(ell / 3.0)
    tb = 0.1 * np.sqrt(tt * bb) * np.sin(ell / 2.5) if odd else 0 * tt
    eb = 0.2 * np.sqrt(ee * bb) * np.cos(ell / 4.0 + 0.3) if odd else 0 * tt
    cl = {}
    for i, f1 in enumerate(FREQS):
        for j, f2 in enumerate(FREQS):
            white = 1e-4 * (i + 1) if i == j else 0.0
            s = scale * (1.0 + 0.1 * i + 0.2 * j)
            ab, ba = 1.0 + 0.05 * i, 1.0 + 0.05 * j
            cl[f1 + f2] = {
                "TT": s * tt + white,
                "EE": s * ee + 2 * white,
                "BB": s * bb + 2 * white,
                "TE": s * ab * te,
                "ET": s * ba * te,
                "TB": s * ab * tb,
                "BT": s * ba * tb,
                "EB": s * ab * eb,
                "BE": s * ba * eb,
            }
    return cl


def _run_blocks(cov, keys, cl):
    strategy = StrategyFactory.create_strategy(cov)
    strategy.configure_run(keys, cl)
    return strategy, _batched(strategy, list(keys.keys()), cl)


@pytest.mark.parametrize("batch_bytes", [1, None])
def test_level1_two_frequencies(te_work, monkeypatch, batch_bytes):
    if batch_bytes is not None:
        monkeypatch.setattr(acc_module, "_ASSEMBLY_BATCH_BYTES", batch_bytes)
    cov = _cov(te_work)
    keys = CovKeys(["T", "E"], FREQS)
    cl = level1_spectra()
    strategy, batched = _run_blocks(cov, keys, cl)
    _assert_bit_identical(strategy, list(keys.keys()), cl, batched)


@pytest.mark.parametrize("odd", [False, True])
@pytest.mark.parametrize("batch_bytes", [1, None])
def test_bmode_six_observables_two_frequencies(b_work, monkeypatch, odd, batch_bytes):
    if batch_bytes is not None:
        monkeypatch.setattr(acc_module, "_ASSEMBLY_BATCH_BYTES", batch_bytes)
    cov = _cov(b_work, polspice_postprocess=False)
    keys = CovKeys(["T", "E", "B"], FREQS, observables=B_OBS)
    cl = _b_spectra(odd)
    strategy, batched = _run_blocks(cov, keys, cl)
    _assert_bit_identical(strategy, list(keys.keys()), cl, batched)


def _per_block_reference(monkeypatch):
    """Make Cov.compute_covariance_matrix run the per-block reference."""

    def terms(self, cov_keys, cl):
        for key in cov_keys:
            yield key, reference_block(self, key, cl)

    monkeypatch.setattr(ACCStrategy, "compute_covariance_terms", terms)


@pytest.mark.parametrize(
    "which, config",
    [
        ("te", {}),
        ("te", {"polspice_postprocess": False, "sum_asymmetric_stokes": True}),
        ("b", {}),
        ("b", {"sum_asymmetric_stokes": True}),
        ("b", {"polspice_postprocess": False}),
    ],
)
def test_covariance_matrix_is_bit_identical(
    te_work, b_work, monkeypatch, which, config
):
    if which == "te":
        work, keys, cl = te_work, CovKeys(["T", "E"], FREQS), level1_spectra()
    else:
        work = b_work
        keys = CovKeys(["T", "E", "B"], FREQS, observables=B_OBS)
        cl = _b_spectra(odd=False)
    batched = _quiet(_cov(work, **config).compute_covariance_matrix, 5, keys, cl)
    with monkeypatch.context() as patch:
        _per_block_reference(patch)
        reference = _quiet(_cov(work, **config).compute_covariance_matrix, 5, keys, cl)
    for got, want in zip(batched, reference):
        assert np.array_equal(got, want)


def test_successive_runs_do_not_share_products(b_work, monkeypatch):
    """One Cov, runs with spectra A, B, A: each equals the per-block result
    for its own spectra, and the two A runs are identical."""
    cov = _cov(b_work, polspice_postprocess=False)
    keys = CovKeys(["T", "E", "B"], FREQS, observables=B_OBS)
    cl_a, cl_b = _b_spectra(odd=False), _b_spectra(odd=True, scale=1.3)
    runs = [
        _quiet(cov.compute_covariance_matrix, 5, keys, cl)[2]
        for cl in (cl_a, cl_b, cl_a)
    ]
    assert np.array_equal(runs[0], runs[2])
    assert not np.array_equal(runs[0], runs[1])
    with monkeypatch.context() as patch:
        _per_block_reference(patch)
        for cl, got in ((cl_a, runs[0]), (cl_b, runs[1])):
            want = _quiet(cov.compute_covariance_matrix, 5, keys, cl)[2]
            assert np.array_equal(got, want)


def test_base_strategy_default_yields_every_key_in_order():
    calls = []

    class Stub(CovarianceStrategy):
        def compute_covariance_term(self, cov_key, cl):
            calls.append(cov_key)
            return cov_key

    keys = ["a", "b", "c"]
    assert list(Stub(None).compute_covariance_terms(keys, {})) == [(k, k) for k in keys]
    assert calls == keys
