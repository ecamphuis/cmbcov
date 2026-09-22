r"""
Identity validation and in-memory memoisation of the ACC coupling-kernel
cache.

This file guards against pointing a run at a stale ``save_dir`` silently
loading kernels built from a different mask or a different resolution:

- ``mask_digest`` (a content digest of the mask map, written into the
  manifest at :func:`~cmbcov.approximations.acc_cache.write_coupling_kernels`
  time) and ``centralell`` are checked on *every* load against ``current``
  values, which a real run draws from ``Cov`` / its config -- these tests
  pass them directly, the way a caller with no ``Cov`` (or a caller testing
  ``acc_cache`` in isolation, as ``tests/test_acc_cache.py`` does) would.
- ``nside``, ``lw`` and ``grid`` are precompute-call choices with no
  persistent record on ``Cov``, so they cannot be compared against an
  independently known "current" value the way mask/centralell can. Instead
  they are checked for *session* consistency: the first manifest a
  ``save_dir`` shows this process (including one just written) establishes
  a reference, and every later manifest for that ``save_dir`` must match
  it. The tests below exercise this the same way a real bug would trigger
  it -- writing kernels for two different (ell, ellp) pairs into the same
  ``save_dir`` with a mismatched field, then reloading the first pair.
- an unverifiable cache (no manifest, or a manifest predating mask-identity
  recording) warns once per ``save_dir`` per process rather than raising,
  and still loads.
- ``Cov.get_acc_coupling_kernels``/``write_acc_coupling_kernels`` memoise
  validated loads in memory, so the same kernel file is not re-parsed with
  ``np.load`` on every diagonal offset of every block.

Kernels here are tiny synthetic arrays written directly through
``write_coupling_kernels``, never computed -- this file never builds an
``ACCStrategy`` precompute, only ``acc_cache`` calls and a cheap ``Cov``
(mask load only, no spherical-harmonic transforms).
"""

import os
import warnings

import numpy as np
import pytest

from cmbcov.approximations import acc_cache
from cmbcov.approximations.acc import COUPLING_SPECTRA
from cmbcov.covariance import Cov, CovarianceConfig, CovarianceMethod

DATA = os.path.join(os.path.dirname(__file__), "data")
KNOWN_SPECTRA = ("TT", "EE", "BB", "TE", "ET")


def _kernel(seed: int, size: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(size, size))


def _write(
    save_dir,
    ell,
    ellp,
    *,
    nside=16,
    lw=10,
    grid="gl",
    centralell=16,
    mask_digest="digestA",
    seed=0,
):
    kernel = _kernel(seed)
    acc_cache.write_coupling_kernels(
        save_dir,
        {("TT", "TT"): kernel},
        ell=ell,
        ellp=ellp,
        known_spectra=KNOWN_SPECTRA,
        centralell=centralell,
        spectra=("TT",),
        grid=grid,
        lw=lw,
        nside=nside,
        mask_digest=mask_digest,
    )
    return kernel


def _small_cov(
    save_dir: str, centralell: int = 16, dmax: int = 1, lmax: int = 32
) -> Cov:
    config = CovarianceConfig(
        method=CovarianceMethod.ACC, lmax=lmax, dmax=dmax, centralell=centralell
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # low centralell warning
        return Cov(
            "baseline_mask.fits", config=config, mask_path=DATA, save_dir=save_dir
        )


# --------------------------------------------------------------------------- #
# mask_digest / centralell: checked directly against `current`
# --------------------------------------------------------------------------- #


def test_loads_cleanly_for_the_mask_it_was_built_for(tmp_path):
    save_dir = str(tmp_path)
    kernel = _write(save_dir, 16, 16, mask_digest="digestA")

    loaded = acc_cache.load_coupling_kernels(
        save_dir,
        16,
        16,
        KNOWN_SPECTRA,
        default_pairs=[("TT", "TT")],
        current={"mask_digest": "digestA", "centralell": 16},
    )
    np.testing.assert_array_equal(loaded[("TT", "TT")], kernel)


def test_raises_for_a_different_mask(tmp_path):
    save_dir = str(tmp_path)
    _write(save_dir, 16, 16, mask_digest="digestA")

    with pytest.raises(ValueError) as excinfo:
        acc_cache.load_coupling_kernels(
            save_dir,
            16,
            16,
            KNOWN_SPECTRA,
            default_pairs=[("TT", "TT")],
            current={"mask_digest": "digestB", "centralell": 16},
        )
    message = str(excinfo.value)
    assert "mask_digest" in message
    assert "digestA" in message
    assert "digestB" in message
    assert save_dir in message


def test_raises_for_a_different_centralell(tmp_path):
    save_dir = str(tmp_path)
    _write(save_dir, 16, 16, centralell=16, mask_digest="digestA")

    with pytest.raises(ValueError) as excinfo:
        acc_cache.load_coupling_kernels(
            save_dir,
            16,
            16,
            KNOWN_SPECTRA,
            default_pairs=[("TT", "TT")],
            current={"mask_digest": "digestA", "centralell": 200},
        )
    message = str(excinfo.value)
    assert "centralell" in message
    assert "16" in message
    assert "200" in message


# --------------------------------------------------------------------------- #
# nside / lw / grid: no independent "current" value exists on Cov for these
# (they are precompute-call choices, not config), so they are checked for
# session consistency instead -- writing two (ell, ellp) pairs into the same
# save_dir with a mismatched field, then reloading the first, is what
# actually triggers a mix-up in practice.
# --------------------------------------------------------------------------- #


def test_raises_for_a_different_nside(tmp_path):
    save_dir = str(tmp_path)
    kernel = _write(save_dir, 16, 16, nside=16)
    current = {"mask_digest": "digestA", "centralell": 16}

    loaded = acc_cache.load_coupling_kernels(
        save_dir, 16, 16, KNOWN_SPECTRA, default_pairs=[("TT", "TT")], current=current
    )
    np.testing.assert_array_equal(loaded[("TT", "TT")], kernel)

    # A second pair written into the same save_dir at a different nside --
    # a cache directory mixing two resolutions.
    _write(save_dir, 16, 17, nside=32, seed=1)

    with pytest.raises(ValueError, match="nside") as excinfo:
        acc_cache.load_coupling_kernels(
            save_dir,
            16,
            16,
            KNOWN_SPECTRA,
            default_pairs=[("TT", "TT")],
            current=current,
        )
    assert "16" in str(excinfo.value)
    assert "32" in str(excinfo.value)


def test_raises_for_a_different_lw(tmp_path):
    save_dir = str(tmp_path)
    _write(save_dir, 16, 16, lw=10)
    current = {"mask_digest": "digestA", "centralell": 16}
    acc_cache.load_coupling_kernels(
        save_dir, 16, 16, KNOWN_SPECTRA, default_pairs=[("TT", "TT")], current=current
    )

    _write(save_dir, 16, 17, lw=47, seed=1)

    with pytest.raises(ValueError, match="lw"):
        acc_cache.load_coupling_kernels(
            save_dir,
            16,
            16,
            KNOWN_SPECTRA,
            default_pairs=[("TT", "TT")],
            current=current,
        )


def test_raises_for_a_different_grid(tmp_path):
    save_dir = str(tmp_path)
    _write(save_dir, 16, 16, grid="gl")
    current = {"mask_digest": "digestA", "centralell": 16}
    acc_cache.load_coupling_kernels(
        save_dir, 16, 16, KNOWN_SPECTRA, default_pairs=[("TT", "TT")], current=current
    )

    _write(save_dir, 16, 17, grid="healpix", seed=1)

    with pytest.raises(ValueError, match="grid"):
        acc_cache.load_coupling_kernels(
            save_dir,
            16,
            16,
            KNOWN_SPECTRA,
            default_pairs=[("TT", "TT")],
            current=current,
        )


# --------------------------------------------------------------------------- #
# map2alm_iter: HEALPix-grid caches only; missing means the old iter=0
# --------------------------------------------------------------------------- #


def _load_tt(save_dir, current):
    return acc_cache.load_coupling_kernels(
        save_dir,
        16,
        16,
        KNOWN_SPECTRA,
        default_pairs=[("TT", "TT")],
        current=current,
    )


CURRENT_ITER3 = {"mask_digest": "digestA", "centralell": 16, "map2alm_iter": 3}


def test_healpix_cache_without_map2alm_iter_is_refused(tmp_path):
    """A HEALPix cache with no recorded ``map2alm_iter`` defaults to iter=0."""
    save_dir = str(tmp_path)
    _write(save_dir, 16, 16, grid="healpix")

    with pytest.raises(ValueError, match="map2alm_iter"):
        _load_tt(save_dir, CURRENT_ITER3)


def test_healpix_cache_with_the_current_map2alm_iter_loads(tmp_path):
    save_dir = str(tmp_path)
    kernel = _kernel(0)
    acc_cache.write_coupling_kernels(
        save_dir,
        {("TT", "TT"): kernel},
        ell=16,
        ellp=16,
        known_spectra=KNOWN_SPECTRA,
        centralell=16,
        spectra=("TT",),
        grid="healpix",
        lw=10,
        nside=16,
        mask_digest="digestA",
        map2alm_iter=3,
    )
    manifest = acc_cache.read_coupling_manifest(save_dir, 16, 16)
    assert manifest["map2alm_iter"] == 3
    np.testing.assert_array_equal(
        _load_tt(save_dir, CURRENT_ITER3)[("TT", "TT")], kernel
    )


def test_gl_cache_ignores_map2alm_iter(tmp_path):
    """The GL precompute is exact and never uses the HEALPix map2alm."""
    save_dir = str(tmp_path)
    kernel = _write(save_dir, 16, 16, grid="gl")
    assert "map2alm_iter" not in acc_cache.read_coupling_manifest(save_dir, 16, 16)
    np.testing.assert_array_equal(
        _load_tt(save_dir, CURRENT_ITER3)[("TT", "TT")], kernel
    )


# --------------------------------------------------------------------------- #
# healpix_kernel_version: HEALPix-grid caches only; missing means version 1
# (Theta from Re X only), refused just like a stale map2alm_iter -- see
# acc_cache.HEALPIX_KERNEL_VERSION.
# --------------------------------------------------------------------------- #

CURRENT_KERNEL_VERSION = {
    "mask_digest": "digestA",
    "centralell": 16,
    "healpix_kernel_version": acc_cache.HEALPIX_KERNEL_VERSION,
}


def test_healpix_cache_without_kernel_version_is_refused(tmp_path):
    """A HEALPix cache with no recorded ``healpix_kernel_version`` is version 1."""
    save_dir = str(tmp_path)
    _write(save_dir, 16, 16, grid="healpix")

    with pytest.raises(ValueError, match="healpix_kernel_version") as excinfo:
        _load_tt(save_dir, CURRENT_KERNEL_VERSION)
    assert "complex-Theta fix" in str(excinfo.value)


def test_healpix_cache_at_version_1_is_refused(tmp_path):
    save_dir = str(tmp_path)
    kernel = _kernel(0)
    acc_cache.write_coupling_kernels(
        save_dir,
        {("TT", "TT"): kernel},
        ell=16,
        ellp=16,
        known_spectra=KNOWN_SPECTRA,
        centralell=16,
        spectra=("TT",),
        grid="healpix",
        lw=10,
        nside=16,
        mask_digest="digestA",
        map2alm_iter=3,
        healpix_kernel_version=1,
    )
    manifest = acc_cache.read_coupling_manifest(save_dir, 16, 16)
    assert manifest["healpix_kernel_version"] == 1

    with pytest.raises(ValueError, match="healpix_kernel_version") as excinfo:
        _load_tt(save_dir, CURRENT_KERNEL_VERSION)
    assert "complex-Theta fix" in str(excinfo.value)


def test_healpix_cache_with_the_current_kernel_version_loads(tmp_path):
    save_dir = str(tmp_path)
    kernel = _kernel(0)
    acc_cache.write_coupling_kernels(
        save_dir,
        {("TT", "TT"): kernel},
        ell=16,
        ellp=16,
        known_spectra=KNOWN_SPECTRA,
        centralell=16,
        spectra=("TT",),
        grid="healpix",
        lw=10,
        nside=16,
        mask_digest="digestA",
        map2alm_iter=3,
        healpix_kernel_version=acc_cache.HEALPIX_KERNEL_VERSION,
    )
    manifest = acc_cache.read_coupling_manifest(save_dir, 16, 16)
    assert manifest["healpix_kernel_version"] == acc_cache.HEALPIX_KERNEL_VERSION
    np.testing.assert_array_equal(
        _load_tt(save_dir, CURRENT_KERNEL_VERSION)[("TT", "TT")], kernel
    )


def test_gl_cache_ignores_healpix_kernel_version(tmp_path):
    """The GL precompute is exact and records no kernel version at all."""
    save_dir = str(tmp_path)
    kernel = _write(save_dir, 16, 16, grid="gl")
    assert "healpix_kernel_version" not in acc_cache.read_coupling_manifest(
        save_dir, 16, 16
    )
    np.testing.assert_array_equal(
        _load_tt(save_dir, CURRENT_KERNEL_VERSION)[("TT", "TT")], kernel
    )


def test_cov_writes_and_accepts_the_package_default(tmp_path):
    from cmbcov.sht import DEFAULT_MAP2ALM_ITER

    save_dir = str(tmp_path)
    cov = _small_cov(save_dir)
    kernel = _kernel(1)
    cov.write_acc_coupling_kernels(
        {("TT", "TT"): kernel},
        16,
        16,
        KNOWN_SPECTRA,
        centralell=16,
        spectra=("TT",),
        grid="healpix",
        lw=10,
        nside=16,
    )
    manifest = acc_cache.read_coupling_manifest(save_dir, 16, 16)
    assert manifest["map2alm_iter"] == DEFAULT_MAP2ALM_ITER == 3
    loaded = cov.get_acc_coupling_kernels(
        16, 16, KNOWN_SPECTRA, [("TT", "TT")], pairs=[("TT", "TT")]
    )
    np.testing.assert_array_equal(loaded[("TT", "TT")], kernel)


# --------------------------------------------------------------------------- #
# Unverifiable caches: warn once per save_dir per process, still load.
# --------------------------------------------------------------------------- #


def test_manifest_without_mask_digest_warns_once_and_still_loads(tmp_path):
    save_dir = str(tmp_path)
    kernel = _kernel(0)
    # No mask_digest passed: simulates a manifest written before mask
    # identity was recorded.
    acc_cache.write_coupling_kernels(
        save_dir,
        {("TT", "TT"): kernel},
        ell=16,
        ellp=16,
        known_spectra=KNOWN_SPECTRA,
        centralell=16,
        spectra=("TT",),
        grid="gl",
        lw=10,
        nside=16,
    )
    current = {"mask_digest": "digestA", "centralell": 16}

    with pytest.warns(UserWarning) as record:
        for _ in range(5):
            loaded = acc_cache.load_coupling_kernels(
                save_dir,
                16,
                16,
                KNOWN_SPECTRA,
                default_pairs=[("TT", "TT")],
                current=current,
            )
            np.testing.assert_array_equal(loaded[("TT", "TT")], kernel)

    mask_warnings = [
        w
        for w in record
        if "mask_digest" in str(w.message) and issubclass(w.category, UserWarning)
    ]
    assert len(mask_warnings) == 1


def test_pair_with_no_manifest_warns_once_and_still_loads(tmp_path):
    save_dir = str(tmp_path)
    kernel = _kernel(0)
    # Write the kernel file by hand, bypassing write_coupling_kernels, so
    # there is no manifest at all -- an older cache, or files placed by
    # hand.
    path = acc_cache.coupling_save_path(save_dir, ("TT", "TT"), 16, 16, KNOWN_SPECTRA)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, kernel)
    current = {"mask_digest": "digestA", "centralell": 16}

    with pytest.warns(UserWarning) as record:
        for _ in range(5):
            loaded = acc_cache.load_coupling_kernels(
                save_dir,
                16,
                16,
                KNOWN_SPECTRA,
                default_pairs=[("TT", "TT")],
                current=current,
            )
            np.testing.assert_array_equal(loaded[("TT", "TT")], kernel)

    no_manifest_warnings = [
        w
        for w in record
        if "no manifest" in str(w.message) and issubclass(w.category, UserWarning)
    ]
    assert len(no_manifest_warnings) == 1


def test_unverifiable_cache_warns_once_with_a_count_and_range(tmp_path):
    """
    A run loads every (centralell, centralell + d) pair; the unverifiable
    condition is the same for all of them, so it is reported once per cache
    directory, with how many pairs it concerns and their range.
    """
    save_dir = str(tmp_path)
    for ellp in (16, 17, 18):
        acc_cache.write_coupling_kernels(
            save_dir,
            {("TT", "TT"): _kernel(ellp)},
            ell=16,
            ellp=ellp,
            known_spectra=KNOWN_SPECTRA,
            centralell=16,
            spectra=("TT",),
            grid="gl",
            lw=10,
            nside=16,
        )
    # and one pair with no manifest at all
    path = acc_cache.coupling_save_path(save_dir, ("TT", "TT"), 16, 19, KNOWN_SPECTRA)
    np.save(path, _kernel(19))
    current = {"mask_digest": "digestA", "centralell": 16}

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        for _ in range(3):
            for ellp in (16, 17, 18, 19):
                acc_cache.load_coupling_kernels(
                    save_dir,
                    16,
                    ellp,
                    KNOWN_SPECTRA,
                    default_pairs=[("TT", "TT")],
                    current=current,
                )
    messages = [str(w.message) for w in record if save_dir in str(w.message)]
    assert len(messages) == 1, messages
    assert "3 of 3 manifests have no mask_digest ((16, 16)..(16, 18))" in messages[0]
    assert "1 of 4 kernel pairs have no manifest ((16, 19))" in messages[0]


# --------------------------------------------------------------------------- #
# git_hash is recorded but never validated.
# --------------------------------------------------------------------------- #


def test_git_hash_mismatch_is_not_checked(tmp_path):
    save_dir = str(tmp_path)
    manifest_path = acc_cache.coupling_manifest_path(save_dir, 16, 16)
    _write(save_dir, 16, 16, mask_digest="digestA")

    # Corrupt the recorded git_hash directly; this must not affect loading.
    import json

    with open(manifest_path) as f:
        manifest = json.load(f)
    manifest["git_hash"] = "not-a-real-hash"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    loaded = acc_cache.load_coupling_kernels(
        save_dir,
        16,
        16,
        KNOWN_SPECTRA,
        default_pairs=[("TT", "TT")],
        current={"mask_digest": "digestA", "centralell": 16},
    )
    assert ("TT", "TT") in loaded


# --------------------------------------------------------------------------- #
# Cov-level memo: repeated loads hit disk once, a write invalidates, a
# mutated return value cannot corrupt a later load.
# --------------------------------------------------------------------------- #


def test_repeated_loads_hit_disk_once(tmp_path, monkeypatch):
    cov = _small_cov(str(tmp_path))
    kernel = _kernel(0)
    cov.write_acc_coupling_kernels(
        {("TT", "TT"): kernel},
        ell=16,
        ellp=16,
        known_spectra=COUPLING_SPECTRA,
        centralell=16,
        spectra=("TT",),
        grid="gl",
        lw=10,
        nside=16,
    )

    calls = []
    real_load = acc_cache.np.load

    def counting_load(*args, **kwargs):
        calls.append(1)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(acc_cache.np, "load", counting_load)

    for _ in range(5):
        loaded = cov.get_acc_coupling_kernels(
            16, 16, COUPLING_SPECTRA, [("TT", "TT")], pairs=[("TT", "TT")]
        )
        np.testing.assert_array_equal(loaded[("TT", "TT")], kernel)

    assert len(calls) == 1


def test_write_in_between_invalidates_the_memo(tmp_path, monkeypatch):
    cov = _small_cov(str(tmp_path))
    kernel1 = _kernel(0)
    cov.write_acc_coupling_kernels(
        {("TT", "TT"): kernel1},
        ell=16,
        ellp=16,
        known_spectra=COUPLING_SPECTRA,
        centralell=16,
        spectra=("TT",),
        grid="gl",
        lw=10,
        nside=16,
    )

    calls = []
    real_load = acc_cache.np.load

    def counting_load(*args, **kwargs):
        calls.append(1)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(acc_cache.np, "load", counting_load)

    loaded1 = cov.get_acc_coupling_kernels(
        16, 16, COUPLING_SPECTRA, [("TT", "TT")], pairs=[("TT", "TT")]
    )[("TT", "TT")]
    assert len(calls) == 1
    np.testing.assert_array_equal(loaded1, kernel1)

    kernel2 = _kernel(1)
    cov.write_acc_coupling_kernels(
        {("TT", "TT"): kernel2},
        ell=16,
        ellp=16,
        known_spectra=COUPLING_SPECTRA,
        centralell=16,
        spectra=("TT",),
        grid="gl",
        lw=10,
        nside=16,
    )

    loaded2 = cov.get_acc_coupling_kernels(
        16, 16, COUPLING_SPECTRA, [("TT", "TT")], pairs=[("TT", "TT")]
    )[("TT", "TT")]
    assert len(calls) == 2
    np.testing.assert_array_equal(loaded2, kernel2)


def test_mutating_a_returned_kernel_does_not_corrupt_the_next_load(tmp_path):
    cov = _small_cov(str(tmp_path))
    kernel = _kernel(0)
    cov.write_acc_coupling_kernels(
        {("TT", "TT"): kernel},
        ell=16,
        ellp=16,
        known_spectra=COUPLING_SPECTRA,
        centralell=16,
        spectra=("TT",),
        grid="gl",
        lw=10,
        nside=16,
    )

    loaded1 = cov.get_acc_coupling_kernels(
        16, 16, COUPLING_SPECTRA, [("TT", "TT")], pairs=[("TT", "TT")]
    )[("TT", "TT")]

    with pytest.raises(ValueError, match="read-only"):
        loaded1[0, 0] = 999.0

    loaded2 = cov.get_acc_coupling_kernels(
        16, 16, COUPLING_SPECTRA, [("TT", "TT")], pairs=[("TT", "TT")]
    )[("TT", "TT")]
    np.testing.assert_array_equal(loaded2, kernel)
