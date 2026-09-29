r"""
Extending an ACC kernel cache with more pairs, and which files a cache
serves.

Before, every precompute into a directory overwrote the per-``(l, l')``
manifest with the channels of its own pairs only: after adding, say, the
pairs ``(DT, TT)`` to an 18-pair B-mode cache, the manifest of every
diagonal claimed ``spectra: [TT, DT]``, so the record no longer described
the kernels on disk (and a later "missing pair" error quoted the wrong
spectra). And a precompute with another identity (another mask, ``lw``,
term selection, ...) into the same directory left the kernels it did not
rewrite certified by the new manifest.

Now (:func:`~cmbcov.approximations.acc_cache.write_coupling_kernels`) the
manifest records the kernel files written (``pairs``); a write with the
same identity merges its ``pairs`` and ``spectra`` into the record, one
with another identity starts a new record, and the loader never serves a
file the record does not list. This pins:

(a) a cache built in steps (the 18 pairs of cmbcov 0.3.0, then the 7 the
    both-orientation requirement adds; or the pairs without an L leg, then
    the rest) has the manifests and kernel files of a one-shot precompute
    of the union, bit for bit, and a two-frequency B run on it is
    bit-identical;
(b) a pair stored only as its transpose counts as on disk for the B-mode
    orientation plan (``ACCStrategy._pairs_on_disk``), as it does for the
    loader: a cache precomputed with every pair transposed gives the same
    blocks, computed in both orientations;
(c) the record rules at the ``acc_cache`` level: merge, new record on
    another identity (left-over files refused, by name), a manifest without
    a record (cmbcov 0.3.0), and an interrupted write with another
    identity leaving no manifest rather than a wrong one.
"""

import json
import os
import warnings

import numpy as np
import pytest

hp = pytest.importorskip("healpy")

from reference.acc_level1 import (  # noqa: E402
    CENTRALELL,
    DMAX,
    LMAX,
    LW,
    NSIDE_ACC,
    rippled_cap_bandlimited,
)

from cmbcov.approximations import StrategyFactory, acc_cache  # noqa: E402
from cmbcov.approximations.acc import (  # noqa: E402
    COUPLING_CHANNELS,
    precompute_acc_kernels,
    require_acc_cache_pairs,
)
from cmbcov.bmode_wick import required_kernel_pairs  # noqa: E402
from cmbcov.covariance import Cov, CovarianceConfig, CovarianceMethod  # noqa: E402
from cmbcov.keys import CovKeys  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

B_OBS = ["TT", "EE", "BB", "TE", "TB", "EB"]
FREQS = ["f1", "f2"]
PAIRS = sorted(required_kernel_pairs(B_OBS))
#: What the both-orientation requirement adds to the 18 pairs of cmbcov 0.3.0.
ADDED = [
    ("DD", "TD"),
    ("DL", "TL"),
    ("DT", "DT"),
    ("DT", "TT"),
    ("LD", "TL"),
    ("LL", "TD"),
    ("LT", "LT"),
]


def _precompute(work, *pair_lists):
    hp.write_map(
        os.path.join(work, "mask.fits"),
        rippled_cap_bandlimited(),
        overwrite=True,
        dtype=np.float64,
    )
    for pairs in pair_lists:
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
    return work


def _cov(work):
    config = CovarianceConfig(
        method=CovarianceMethod.ACC,
        lmax=LMAX,
        lmin=2,
        dmax=DMAX,
        centralell=CENTRALELL,
        polspice_postprocess=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Cov("mask.fits", config=config, mask_path=work, save_dir=work)


def _spectra(n=80):
    """Two frequencies, every letter pair, XY != YX across frequencies,
    C^TB = C^EB = 0."""
    ell = np.arange(n)
    tt, ee, bb = np.zeros(n), np.zeros(n), np.zeros(n)
    tt[2:] = 1.0 / (ell[2:] * (ell[2:] + 1)) ** 0.8
    ee[2:] = 0.3 / (ell[2:] + 3.0) ** 1.7 * (1 + 0.5 * np.cos(ell[2:] / 2.0))
    bb[2:] = 0.05 / (ell[2:] + 1.0) ** 0.9 * (1 + 0.4 * np.sin(ell[2:] / 1.3))
    te = 0.6 * np.sqrt(tt * ee) * np.cos(ell / 3.0)
    cl = {}
    for i, f1 in enumerate(FREQS):
        for j, f2 in enumerate(FREQS):
            white = 0.01 * (i + 1) if i == j else 0.0
            scale = 1.0 + 0.1 * i + 0.2 * j
            cl[f1 + f2] = {
                "TT": scale * tt + white,
                "EE": scale * ee + 2 * white,
                "BB": scale * bb + 2 * white,
                "TE": scale * (1.0 + 0.05 * i) * te,
                "ET": scale * (1.0 + 0.05 * j) * te,
                "TB": 0 * tt,
                "BT": 0 * tt,
                "EB": 0 * tt,
                "BE": 0 * tt,
            }
    return cl


def _blocks(work):
    """Every raw block of a two-frequency B run, and its orientation."""
    cl = _spectra()
    keys = CovKeys([], FREQS, observables=B_OBS)
    strategy = StrategyFactory.create_strategy(_cov(work))
    strategy.configure_run(keys, cl)
    out = {}
    for key in keys.keys():
        orientation = strategy.raw_block_inputs(key, cl)[1]["acc_block_orientation"]
        out[key.stokekey(), key.freqkey()] = (
            strategy.compute_covariance_term(key, cl),
            orientation,
        )
    return out


def _manifests(work):
    out = {}
    for d in range(DMAX):
        manifest = acc_cache.read_coupling_manifest(work, CENTRALELL, CENTRALELL + d)
        manifest.pop("git_hash")
        out[d] = manifest
    return out


def _kernel_files(work):
    folder = os.path.join(work, "covariance_coupling")
    return {
        name: np.load(os.path.join(folder, name))
        for name in sorted(os.listdir(folder))
        if name.endswith(".npy")
    }


@pytest.fixture(scope="module")
def one_shot(tmp_path_factory):
    work = _precompute(str(tmp_path_factory.mktemp("one_shot")), PAIRS)
    return work, _blocks(work)


# --------------------------------------------------------------------------- #
# (a) extending a cache in steps
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "steps",
    [
        [sorted(set(PAIRS) - set(ADDED)), ADDED],
        [
            [p for p in PAIRS if "L" not in "".join(p)],
            [p for p in PAIRS if "L" in "".join(p)],
        ],
    ],
    ids=["0.3.0-then-added", "no-L-then-L"],
)
def test_extending_a_cache_in_steps_matches_a_one_shot_precompute(
    one_shot, tmp_path, steps
):
    reference_dir, reference_blocks = one_shot
    assert sorted(p for step in steps for p in step) == PAIRS
    work = _precompute(str(tmp_path), *steps)

    # The manifests record the union, exactly as the one-shot precompute's.
    manifests = _manifests(work)
    assert manifests == _manifests(reference_dir)
    for manifest in manifests.values():
        assert manifest["pairs"] == sorted("x".join(p) for p in PAIRS)
        assert manifest["spectra"] == list(COUPLING_CHANNELS)

    kernels = _kernel_files(work)
    reference_kernels = _kernel_files(reference_dir)
    assert sorted(kernels) == sorted(reference_kernels)
    for name, array in kernels.items():
        np.testing.assert_array_equal(array, reference_kernels[name], err_msg=name)

    require_acc_cache_pairs(work, CENTRALELL, DMAX, PAIRS)  # does not raise
    blocks = _blocks(work)
    assert blocks.keys() == reference_blocks.keys()
    for label, (block, orientation) in blocks.items():
        np.testing.assert_array_equal(block, reference_blocks[label][0])
        assert orientation == reference_blocks[label][1]
    assert {o for _, o in blocks.values()} == {"natural", "both"}


# --------------------------------------------------------------------------- #
# (b) a pair stored only as its transpose
# --------------------------------------------------------------------------- #


def test_a_pair_stored_only_as_its_transpose_is_on_disk(one_shot, tmp_path):
    reference_dir, reference_blocks = one_shot
    work = _precompute(str(tmp_path), [(b, a) for a, b in PAIRS])
    names = set(_kernel_files(work))
    asymmetric = [(a, b) for a, b in PAIRS if a != b]
    assert all(
        f"{a}x{b}_{CENTRALELL}x{CENTRALELL}.npy" not in names for a, b in asymmetric
    )

    strategy = StrategyFactory.create_strategy(_cov(work))
    assert strategy._pairs_on_disk(PAIRS)
    require_acc_cache_pairs(work, CENTRALELL, DMAX, PAIRS)  # does not raise
    blocks = _blocks(work)
    for label, (block, orientation) in blocks.items():
        np.testing.assert_array_equal(block, reference_blocks[label][0])
        assert orientation == reference_blocks[label][1]

    # One transposed file gone: that pair is no longer on disk.
    a, b = asymmetric[0]
    os.remove(
        os.path.join(
            work, "covariance_coupling", f"{b}x{a}_{CENTRALELL}x{CENTRALELL}.npy"
        )
    )
    assert not strategy._pairs_on_disk([(a, b)])
    assert strategy._pairs_on_disk([p for p in PAIRS if p != (a, b)])


# --------------------------------------------------------------------------- #
# (c) the record rules, on plain arrays
# --------------------------------------------------------------------------- #

KNOWN = ("TT", "DD", "LL", "TD", "DT")


def _kernel(seed):
    return np.random.default_rng(seed).normal(size=(4, 4))


def _write(save_dir, kernels, spectra, **identity):
    fields = {"grid": "gl", "lw": 6, "nside": 16, "mask_digest": "mask-a"}
    fields.update(identity)
    acc_cache.write_coupling_kernels(
        save_dir, kernels, 8, 9, KNOWN, 8, spectra=spectra, **fields
    )


def _load(save_dir, pairs):
    return acc_cache.load_coupling_kernels(save_dir, 8, 9, KNOWN, pairs, pairs=pairs)


def test_same_identity_merges_the_record(tmp_path):
    save_dir = str(tmp_path)
    _write(save_dir, {("TD", "TD"): _kernel(0)}, ("TD",))
    _write(save_dir, {"TTxTT": _kernel(1), ("DD", "TT"): _kernel(2)}, ("TT", "DD"))
    manifest = acc_cache.read_coupling_manifest(save_dir, 8, 9)
    assert manifest["pairs"] == ["DDxTT", "TDxTD", "TTxTT"]
    assert manifest["spectra"] == ["TT", "DD", "TD"]  # KNOWN order
    loaded = _load(save_dir, [("TD", "TD"), ("TT", "TT"), ("TT", "DD")])
    np.testing.assert_array_equal(loaded["TD", "TD"], _kernel(0))
    np.testing.assert_array_equal(loaded["TT", "DD"], _kernel(2).T)


@pytest.mark.parametrize(
    "change",
    [
        {"mask_digest": "mask-b"},
        {"lw": 8},
        {"nside": 32},
        {"grid": "healpix"},
        {"term_selection": 1e-3},
    ],
    ids=lambda c: next(iter(c)),
)
def test_another_identity_starts_a_new_record(tmp_path, change):
    """Kernels an earlier cache left behind are never served under the new
    manifest's identity: the loader and the orientation check refuse them,
    naming the file."""
    save_dir = str(tmp_path)
    _write(save_dir, {"TTxTT": _kernel(0), "DDxDD": _kernel(1)}, ("TT", "DD"))
    _write(save_dir, {"TTxTT": _kernel(2)}, ("TT",), **change)
    manifest = acc_cache.read_coupling_manifest(save_dir, 8, 9)
    assert manifest["pairs"] == ["TTxTT"] and manifest["spectra"] == ["TT"]
    np.testing.assert_array_equal(
        _load(save_dir, [("TT", "TT")])["TT", "TT"], _kernel(2)
    )
    left_over = os.path.join(save_dir, "covariance_coupling", "DDxDD_8x9.npy")
    assert os.path.exists(left_over)
    found, unrecorded = acc_cache.locate_coupling_kernel(
        save_dir, ("DD", "DD"), 8, 9, KNOWN, manifest=manifest
    )
    assert found is None and unrecorded == [left_over]
    with pytest.raises(OSError, match="left over") as excinfo:
        _load(save_dir, [("DD", "DD")])
    assert left_over in str(excinfo.value)


def test_a_manifest_without_a_record_merges_without_one(tmp_path):
    """A cmbcov 0.3.0 manifest (no ``pairs``): extending it with the same
    identity merges ``spectra`` but cannot know the full file list, so the
    record stays absent and every file is served, as before."""
    save_dir = str(tmp_path)
    _write(save_dir, {"TTxTT": _kernel(0), "DDxDD": _kernel(1)}, ("TT", "DD"))
    path = acc_cache.coupling_manifest_path(save_dir, 8, 9)
    with open(path) as handle:
        manifest = json.load(handle)
    del manifest["pairs"]
    with open(path, "w") as handle:
        json.dump(manifest, handle)
    _write(save_dir, {"TDxTD": _kernel(2)}, ("TD",))
    merged = acc_cache.read_coupling_manifest(save_dir, 8, 9)
    assert "pairs" not in merged
    assert merged["spectra"] == ["TT", "DD", "TD"]
    assert set(_load(save_dir, [("TT", "TT"), ("DD", "DD"), ("TD", "TD")])) == {
        ("TT", "TT"),
        ("DD", "DD"),
        ("TD", "TD"),
    }


def test_an_interrupted_write_with_another_identity_leaves_no_manifest(
    tmp_path, monkeypatch
):
    """The old manifest is removed before a write with another identity
    touches any kernel file, so a failure part-way leaves an unverifiable
    cache (warned about), never kernels certified by the wrong identity. An
    interrupted extension (same identity) keeps the old record, which does
    not list the new files."""
    save_dir = str(tmp_path)
    _write(save_dir, {"TTxTT": _kernel(0)}, ("TT",))

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(np, "save", fail)
    with pytest.raises(OSError, match="disk full"):
        _write(save_dir, {"DDxDD": _kernel(1)}, ("DD",))
    assert acc_cache.read_coupling_manifest(save_dir, 8, 9)["pairs"] == ["TTxTT"]
    with pytest.raises(OSError, match="disk full"):
        _write(save_dir, {"TTxTT": _kernel(2)}, ("TT",), mask_digest="mask-b")
    assert acc_cache.read_coupling_manifest(save_dir, 8, 9) is None
