"""
Regression tests for on-disk kernel cache keying (`cmbcov.mask`,
"Kernel cache keying").

Reusing `M_<mask>.fits`/`Msq_<mask>.fits` by file name and size alone risks
a stale cache silently passing while holding kernel values that differ from
a fresh computation by up to 2.0e-3. The MASTER (M/Msq), PolSpice-Master (G)
and PolSpice (K) caches instead carry a sidecar JSON manifest
(`<fits path>.manifest.json`) and are reused only when it matches:

- a kernel is written under a parameter-keyed name
  (`<legacy stem>.k<12 hex digits>.fits`), found again by that name;
- a file with no manifest, or a mismatching one, is never modified: the fresh
  kernel goes to the keyed name next to it (the tests below compare bytes);
- the legacy name is still read when its manifest matches;
- G and K are reused only from a native manifest: a Fortran-written G/K is
  recomputed (cor2cl truncates the Legendre transform at lmax);
- G and K are written and reused rather than recomputed on every run.

Every test uses a private `tmp_path` for its kernel cache -- never
`tests/data/utils_baseline_mask/`.
"""

import json
import os
import re
import shutil
import warnings

import numpy as np
import pytest

healpy = pytest.importorskip("healpy")

from cmbcov.mask import (  # noqa: E402
    KERNEL_CACHE_VERSION,
    MaskWlm,
    _kernel_manifest_path,
    keyed_kernel_path,
)

DATA = os.path.join(os.path.dirname(__file__), "data")

# nside=32 mask, kept to a low band limit throughout: this file is only
# exercising cache bookkeeping, not kernel accuracy (that is
# tests/test_coupling_kernels.py).
ELL_MAX = 17  # -> l1max = l2max = 16
ELL_MAX_WIDE = 33  # -> l1max = l2max = 32, strictly wider than ELL_MAX
APOD = (1, 30.0, 30.0)  # apodization type, sigma, theta_max

KEYED = re.compile(r".*\.k[0-9a-f]{12}\.fits$")


@pytest.fixture
def wlm(tmp_path):
    shutil.copy(os.path.join(DATA, "baseline_mask.fits"), tmp_path / "m.fits")
    return MaskWlm("m.fits", load_path=str(tmp_path))


@pytest.fixture
def other_wlm(tmp_path):
    """
    A different mask under the *same* file/mask name ("m.fits"), in a
    separate directory -- lets a test plant "the wrong mask's kernel" under
    exactly the path `wlm`'s getter would look for.
    """
    mask = np.asarray(healpy.read_map(os.path.join(DATA, "baseline_mask.fits")))
    # Zero out a contiguous quarter of the good pixels: a different mask
    # (different digest, different kernel), same nside and file name.
    good = np.flatnonzero(mask > 0)
    mask = mask.copy()
    mask[good[: len(good) // 4]] = 0.0
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    healpy.write_map(str(other_dir / "m.fits"), mask, overwrite=True, dtype=np.float64)
    return MaskWlm("m.fits", load_path=str(other_dir))


def _manifest(fits_path):
    with open(_kernel_manifest_path(str(fits_path))) as f:
        return json.load(f)


def _write_manifest(fits_path, manifest):
    with open(_kernel_manifest_path(str(fits_path)), "w") as f:
        json.dump(manifest, f)


def _snapshot(directory):
    """``{name: bytes}`` of every file in ``directory``."""
    return {
        name: open(os.path.join(directory, name), "rb").read()
        for name in sorted(os.listdir(directory))
    }


def _keyed_files(directory, prefix):
    return sorted(
        n for n in os.listdir(directory) if n.startswith(prefix) and KEYED.match(n)
    )


def _plant_legacy_msq(wlm_source, cache_dir, scratch):
    """
    Compute `wlm_source`'s MASTER kernels into `scratch` and copy the Msq
    file and its manifest to the *legacy* name `cache_dir/Msq_m.fits`,
    i.e. what a cache written before keyed names looks like.
    """
    wlm_source.compute_master_coupling_kernels(ELL_MAX - 1, ELL_MAX - 1, str(scratch))
    (keyed,) = _keyed_files(scratch, "Msq_m.")
    os.makedirs(cache_dir, exist_ok=True)
    legacy = cache_dir / "Msq_m.fits"
    shutil.copy(scratch / keyed, legacy)
    shutil.copy(
        _kernel_manifest_path(str(scratch / keyed)), _kernel_manifest_path(str(legacy))
    )
    return legacy


def _get_msq(wlm, cache_dir, ell_max=ELL_MAX):
    return wlm.get_master_coupling_kernels(
        ell_max=ell_max,
        kernel_dir=str(cache_dir),
        use_squared=True,
        make_symmetric=False,
    )


def _direct_msq(wlm, tmp_path):
    return wlm.compute_master_coupling_kernels(
        ELL_MAX - 1, ELL_MAX - 1, str(tmp_path / "reference")
    )[1]


# ---------------------------------------------------------------------------
# Keyed file names
# ---------------------------------------------------------------------------


def test_keyed_name_is_deterministic_and_changes_with_every_field():
    manifest = {"cache_version": 1, "l1max": 16, "mask_digest": "ab", "x": 1.5}
    path = keyed_kernel_path("/d/Msq_m.fits", manifest)
    assert re.fullmatch(r"/d/Msq_m\.k[0-9a-f]{12}\.fits", path)
    # Key order does not matter; every value does.
    assert keyed_kernel_path("/d/Msq_m.fits", dict(reversed(manifest.items()))) == path
    for field in manifest:
        changed = dict(manifest)
        changed[field] = "other"
        assert keyed_kernel_path("/d/Msq_m.fits", changed) != path


def test_compute_writes_keyed_files_only(wlm, tmp_path):
    cache_dir = tmp_path / "cache"
    wlm.compute_master_coupling_kernels(ELL_MAX - 1, ELL_MAX - 1, str(cache_dir))
    wlm.compute_polspice_master_kernel(ELL_MAX - 1, ELL_MAX - 1, str(cache_dir), *APOD)
    wlm.compute_polspice_kernel(ELL_MAX - 1, ELL_MAX - 1, str(cache_dir), *APOD)
    names = sorted(os.listdir(cache_dir))
    fits_names = [n for n in names if n.endswith(".fits")]
    assert len(fits_names) == 4 and all(KEYED.match(n) for n in fits_names), names
    for name in fits_names:
        manifest = _manifest(cache_dir / name)
        assert manifest["backend"] == "native"
        assert str(cache_dir / name) == keyed_kernel_path(
            str(cache_dir / re.sub(r"\.k[0-9a-f]{12}\.fits$", ".fits", name)), manifest
        )


# ---------------------------------------------------------------------------
# The incident itself: a planted, wrong file under the expected legacy name
# ---------------------------------------------------------------------------


def test_planted_kernel_from_a_different_mask_is_not_reused(wlm, other_wlm, tmp_path):
    """
    Guards against a kernel file sitting at the legacy path `wlm` looks for,
    with a fully formed manifest, but built from another mask: the loader
    must detect the mask_digest mismatch, leave that file alone, and return
    `wlm`'s own kernel.
    """
    cache_dir = tmp_path / "cache"
    legacy = _plant_legacy_msq(other_wlm, cache_dir, tmp_path / "scratch")
    assert _manifest(legacy)["mask_digest"] == other_wlm.mask_digest != wlm.mask_digest
    before = _snapshot(cache_dir)

    with pytest.warns(UserWarning, match="mask_digest.*left untouched"):
        loaded = _get_msq(wlm, cache_dir)

    np.testing.assert_array_equal(loaded, _direct_msq(wlm, tmp_path))
    after = _snapshot(cache_dir)
    assert {n: after[n] for n in before} == before  # legacy bytes unchanged
    (keyed,) = _keyed_files(cache_dir, "Msq_m.")
    assert _manifest(cache_dir / keyed)["mask_digest"] == wlm.mask_digest


def test_planted_kernel_with_a_stale_cache_version_is_not_reused(wlm, tmp_path):
    """The other incident shape: same mask, but written by older code."""
    cache_dir = tmp_path / "cache"
    legacy = _plant_legacy_msq(wlm, cache_dir, tmp_path / "scratch")
    manifest = _manifest(legacy)
    manifest["cache_version"] = KERNEL_CACHE_VERSION - 1
    _write_manifest(legacy, manifest)
    # Corrupt the on-disk kernel values too, so a silent reuse would be
    # caught even if the manifest were (wrongly) ignored.
    from astropy.io import fits as astropy_fits

    data = astropy_fits.getdata(str(legacy))
    astropy_fits.PrimaryHDU(np.asarray(data) * 0.0).writeto(str(legacy), overwrite=True)
    before = _snapshot(cache_dir)

    with pytest.warns(UserWarning, match="cache_version"):
        loaded = _get_msq(wlm, cache_dir)

    assert not np.allclose(loaded, 0.0)
    after = _snapshot(cache_dir)
    assert {n: after[n] for n in before} == before
    (keyed,) = _keyed_files(cache_dir, "Msq_m.")
    assert _manifest(cache_dir / keyed)["cache_version"] == KERNEL_CACHE_VERSION


# ---------------------------------------------------------------------------
# Never overwrite: manifest-less legacy files (every cache before keying)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", ["M", "Msq", "G"])
def test_manifest_less_legacy_file_is_left_byte_identical_and_keyed_file_reused(
    wlm, tmp_path, monkeypatch, family
):
    cache_dir = tmp_path / "cache"
    os.makedirs(cache_dir)
    # A manifest-less legacy file with garbage content under the legacy name.
    legacy_name = {
        "M": "M_m.fits",
        "Msq": "Msq_m.fits",
        "G": "G_m_typ1_sig30.0_the30.0.fits",
    }[family]
    from astropy.io import fits as astropy_fits

    astropy_fits.PrimaryHDU(np.full((4, ELL_MAX, ELL_MAX), 7.0)).writeto(
        str(cache_dir / legacy_name)
    )
    before = _snapshot(cache_dir)

    def get():
        if family == "G":
            return wlm.get_polspice_master_kernel(ELL_MAX, str(cache_dir), *APOD)
        return wlm.get_master_coupling_kernels(
            ell_max=ELL_MAX,
            kernel_dir=str(cache_dir),
            use_squared=family == "Msq",
            make_symmetric=False,
        )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first = get()
    assert [w for w in caught if "no manifest" in str(w.message)], [
        str(w.message) for w in caught
    ]
    assert not np.any(first == 7.0)
    after = _snapshot(cache_dir)
    assert {n: after[n] for n in before} == before  # byte-identical
    keyed = _keyed_files(cache_dir, legacy_name[: -len("fits")])
    assert len(keyed) == 1

    # Second call: found under the keyed name, nothing recomputed, no warning.
    import cmbcov.kernels.coupling as coupling_module
    import cmbcov.kernels.polspice as polspice_module

    calls = []

    def counting(real):
        def inner(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)

        return inner

    monkeypatch.setattr(
        coupling_module,
        "coupling_kernels",
        counting(coupling_module.coupling_kernels),
    )
    monkeypatch.setattr(
        polspice_module,
        "polspice_kernels",
        counting(polspice_module.polspice_kernels),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        second = get()
    assert calls == []
    np.testing.assert_array_equal(second, first)
    assert second.dtype == np.float64 and second.dtype.isnative
    assert _snapshot(cache_dir) == after


def test_no_manifest_warns_once_per_file(wlm, tmp_path):
    cache_dir = tmp_path / "cache"
    legacy = _plant_legacy_msq(wlm, cache_dir, tmp_path / "scratch")
    os.remove(_kernel_manifest_path(str(legacy)))
    other_dir = tmp_path / "cache_other"
    shutil.copytree(cache_dir, other_dir)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _get_msq(wlm, cache_dir)
        _get_msq(wlm, cache_dir)  # keyed file now found: no second warning
        _get_msq(wlm, other_dir)  # a different legacy file: warned on its own

    manifest_warnings = [w for w in caught if "no manifest" in str(w.message)]
    assert len(manifest_warnings) == 2


# ---------------------------------------------------------------------------
# Every keyed field, one at a time, on a legacy-name file
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field, corrupt",
    [
        ("cache_version", lambda m: KERNEL_CACHE_VERSION - 1),
        ("mask_digest", lambda m: "0" * len(m["mask_digest"])),
        ("l1max", lambda m: m["l1max"] - 1),
        ("l2max", lambda m: m["l2max"] - 1),
        ("mask_alm_lmax", lambda m: m["mask_alm_lmax"] - 1),
        ("mask_alm_maxiter", lambda m: m["mask_alm_maxiter"] + 1),
        ("mask_alm_epsilon", lambda m: m["mask_alm_epsilon"] * 10),
    ],
)
def test_each_keyed_field_triggers_a_named_recompute(wlm, tmp_path, field, corrupt):
    cache_dir = tmp_path / "cache"
    legacy = _plant_legacy_msq(wlm, cache_dir, tmp_path / "scratch")
    manifest = _manifest(legacy)
    manifest[field] = corrupt(manifest)
    _write_manifest(legacy, manifest)
    before = _snapshot(cache_dir)

    with pytest.warns(UserWarning, match=field):
        loaded = _get_msq(wlm, cache_dir)

    np.testing.assert_array_equal(loaded, _direct_msq(wlm, tmp_path))
    after = _snapshot(cache_dir)
    # The mismatching file and its manifest are left exactly as they were.
    assert {n: after[n] for n in before} == before
    assert _manifest(legacy)[field] == manifest[field]
    (keyed,) = _keyed_files(cache_dir, "Msq_m.")
    assert _manifest(cache_dir / keyed)[field] != manifest[field]


# ---------------------------------------------------------------------------
# Reuse without recomputing: keyed name, legacy name with a matching manifest
# ---------------------------------------------------------------------------


def _count_coupling_calls(monkeypatch):
    import cmbcov.kernels.coupling as coupling_module

    calls = []
    real = coupling_module.coupling_kernels

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(coupling_module, "coupling_kernels", counting)
    return calls


def test_matching_keyed_cache_is_reused_without_recomputing(wlm, tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    first = _get_msq(wlm, cache_dir)
    calls = _count_coupling_calls(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a cache hit must not warn either
        second = _get_msq(wlm, cache_dir)
    assert calls == []
    np.testing.assert_array_equal(second, first)


def test_legacy_name_with_a_matching_manifest_is_still_reused(
    wlm, tmp_path, monkeypatch
):
    """Backward compatibility: caches written before keyed names."""
    cache_dir = tmp_path / "cache"
    legacy = _plant_legacy_msq(wlm, cache_dir, tmp_path / "scratch")
    before = _snapshot(cache_dir)
    calls = _count_coupling_calls(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loaded = _get_msq(wlm, cache_dir)
    assert calls == []
    assert _snapshot(cache_dir) == before  # nothing written next to it
    from astropy.io import fits as astropy_fits

    np.testing.assert_array_equal(loaded, astropy_fits.getdata(str(legacy)))


def test_keyed_file_takes_precedence_over_a_matching_legacy_file(wlm, tmp_path):
    cache_dir = tmp_path / "cache"
    scratch = tmp_path / "scratch"
    legacy = _plant_legacy_msq(wlm, cache_dir, scratch)
    from astropy.io import fits as astropy_fits

    # Same manifest, marker values under the legacy name.
    astropy_fits.PrimaryHDU(np.full((4, ELL_MAX, ELL_MAX), 7.0)).writeto(
        str(legacy), overwrite=True
    )
    (keyed,) = _keyed_files(scratch, "Msq_m.")
    for suffix in ("", ".manifest.json"):
        shutil.copy(str(scratch / keyed) + suffix, str(cache_dir / keyed) + suffix)
    loaded = _get_msq(wlm, cache_dir)
    np.testing.assert_array_equal(loaded, astropy_fits.getdata(str(cache_dir / keyed)))
    assert not np.any(loaded == 7.0)


def test_unrelated_file_at_the_keyed_name_is_not_overwritten(wlm, tmp_path):
    """A keyed name occupied by a manifest-less file (an interrupted write)."""
    cache_dir = tmp_path / "cache"
    _get_msq(wlm, tmp_path / "scratch")
    (keyed,) = _keyed_files(tmp_path / "scratch", "Msq_m.")
    os.makedirs(cache_dir)
    (cache_dir / keyed).write_bytes(b"partial")
    before = _snapshot(cache_dir)
    with pytest.warns(UserWarning, match="Not writing kernel cache"):
        loaded = _get_msq(wlm, cache_dir)
    np.testing.assert_array_equal(loaded, _direct_msq(wlm, tmp_path))
    after = _snapshot(cache_dir)
    assert after[keyed] == b"partial"
    assert {n: after[n] for n in before} == before


# ---------------------------------------------------------------------------
# Slicing a wider cache reproduces the smaller kernel
# ---------------------------------------------------------------------------


def test_kernel_sliced_from_a_wider_cache_matches_a_direct_computation(wlm, tmp_path):
    """
    Load-path prerequisite for treating `l1max`/`l2max` as "cached >=
    requested" rather than equality: verify that is actually true for the
    native path before relying on it in the manifest comparison.
    """
    wide_cache = tmp_path / "wide"
    _get_msq(wlm, wide_cache, ell_max=ELL_MAX_WIDE)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # sliced reuse must not warn
        sliced = _get_msq(wlm, wide_cache)
    assert len(_keyed_files(wide_cache, "Msq_m.")) == 1  # reused, not rewritten

    # Not bit-identical: computing directly at lmax=16 uses a differently
    # sized Gauss-Legendre grid than computing at lmax=32 and slicing, so the
    # two accumulate floating-point rounding differently even though both
    # are exact quadratures of the same polynomial integrand.
    np.testing.assert_allclose(
        sliced, _direct_msq(wlm, tmp_path), rtol=1e-6, atol=1e-12
    )


# ---------------------------------------------------------------------------
# A failed FITS write must never leave a manifest behind
# ---------------------------------------------------------------------------


def test_manifest_is_not_written_if_the_fits_write_fails(wlm, tmp_path, monkeypatch):
    from astropy.io import fits as astropy_fits

    def failing_writeto(self, *args, **kwargs):
        raise OSError("disk full (simulated)")

    monkeypatch.setattr(astropy_fits.PrimaryHDU, "writeto", failing_writeto)

    cache_dir = tmp_path / "cache"
    with pytest.raises(OSError, match="disk full"):
        wlm.compute_master_coupling_kernels(ELL_MAX - 1, ELL_MAX - 1, str(cache_dir))

    assert [n for n in os.listdir(cache_dir) if "manifest" in n] == []


# ---------------------------------------------------------------------------
# G and K: written, reused natively, a Fortran cache recomputed
# ---------------------------------------------------------------------------


def _write_like_fortran(path: str, kernel: np.ndarray) -> None:
    """Write ``kernel[c, l1, l2]`` the way the legacy ``master_kernels`` did.

    Mirrors ``tests/test_polspice_orientation.py::_write_like_fortran``: see
    that module for why this reproduces the file bit for bit.
    """
    from astropy.io import fits as astropy_fits

    fortran = np.asfortranarray(np.transpose(kernel, (1, 2, 0)))
    astropy_fits.PrimaryHDU(np.ascontiguousarray(fortran.T)).writeto(
        path, overwrite=True
    )


def test_polspice_master_kernel_is_cached_and_reused(wlm, tmp_path, monkeypatch):
    """G is written and reused rather than recomputed on every run."""
    cache_dir = tmp_path / "cache"
    first = wlm.get_polspice_master_kernel(ELL_MAX, str(cache_dir), *APOD)
    (keyed,) = _keyed_files(cache_dir, "G_m_typ1_sig30.0_the30.0.")
    manifest = _manifest(cache_dir / keyed)
    assert manifest["backend"] == "native"
    assert manifest["mask_digest"] == wlm.mask_digest
    assert (manifest["l1max"], manifest["l2max"]) == (ELL_MAX - 1, ELL_MAX - 1)
    assert {"mask_alm_lmax", "mask_alm_maxiter", "mask_alm_epsilon"} <= set(manifest)

    import cmbcov.kernels.polspice as polspice_module

    calls = []
    real = polspice_module.polspice_kernels
    monkeypatch.setattr(
        polspice_module,
        "polspice_kernels",
        lambda *a, **k: calls.append(1) or real(*a, **k),
    )
    fresh = MaskWlm("m.fits", load_path=wlm.load_path)  # a new process, in effect
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        second = fresh.get_polspice_master_kernel(ELL_MAX, str(cache_dir), *APOD)
    assert calls == []
    assert fresh.mask_alm is None  # not even the alm solve ran
    np.testing.assert_array_equal(second, first)
    assert second.dtype == np.float64 and second.dtype.isnative


def test_polspice_master_kernel_from_a_fortran_cache_is_recomputed_not_reused(
    wlm, tmp_path
):
    """
    A G file left over from the legacy Fortran route -- Fortran-ordered
    bytes, manifest ``backend="fortran"`` -- is not numerically equivalent
    (cor2cl truncates the Legendre transform of g at lmax: 23% max element
    error at lmax 24 on a small patch). It must be recomputed natively,
    written under the keyed name, and left untouched itself.
    """
    cache_dir = tmp_path / "cache"
    os.makedirs(cache_dir)
    fits_path = str(cache_dir / "G_m_typ1_sig30.0_the30.0.fits")
    native_kernel = wlm.compute_polspice_master_kernel(
        ELL_MAX - 1, ELL_MAX - 1, str(tmp_path / "scratch"), *APOD
    )
    # Distinguishable from the native kernel, so a reuse would be caught.
    _write_like_fortran(fits_path, native_kernel * 1.01)
    _write_manifest(
        fits_path,
        {
            "cache_version": KERNEL_CACHE_VERSION,
            "mask_digest": wlm.mask_digest,
            "backend": "fortran",
            "l1max": ELL_MAX - 1,
            "l2max": ELL_MAX - 1,
            "apodization_type": 1,
            "apodization_sigma": 30.0,
            "theta_max": 30.0,
        },
    )
    before = _snapshot(cache_dir)

    with pytest.warns(UserWarning, match="backend"):
        loaded = wlm.get_polspice_master_kernel(ELL_MAX, str(cache_dir), *APOD)

    np.testing.assert_array_equal(loaded, native_kernel)
    after = _snapshot(cache_dir)
    assert {n: after[n] for n in before} == before
    (keyed,) = _keyed_files(cache_dir, "G_m_typ1_sig30.0_the30.0.")
    assert _manifest(cache_dir / keyed)["backend"] == "native"

    # And the next call finds the native keyed file without warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        again = wlm.get_polspice_master_kernel(ELL_MAX, str(cache_dir), *APOD)
    np.testing.assert_array_equal(again, native_kernel)


def test_polspice_kernel_from_a_fortran_cache_is_recomputed_not_reused(wlm, tmp_path):
    cache_dir = tmp_path / "cache"
    os.makedirs(cache_dir)
    fits_path = str(cache_dir / "K_typ1_sig30.0_the30.0.fits")
    native_kernel = wlm.compute_polspice_kernel(
        ELL_MAX - 1, ELL_MAX - 1, str(tmp_path / "scratch"), *APOD
    )
    _write_like_fortran(fits_path, native_kernel * 1.01)
    _write_manifest(
        fits_path,
        {
            "cache_version": KERNEL_CACHE_VERSION,
            "backend": "fortran",
            "l1max": ELL_MAX - 1,
            "l2max": ELL_MAX - 1,
            "apodization_type": 1,
            "apodization_sigma": 30.0,
            "theta_max": 30.0,
        },
    )
    before = _snapshot(cache_dir)
    with pytest.warns(UserWarning, match="backend"):
        loaded = wlm.get_polspice_kernel(ELL_MAX, str(cache_dir), *APOD)
    np.testing.assert_array_equal(loaded, native_kernel)
    after = _snapshot(cache_dir)
    assert {n: after[n] for n in before} == before
    assert len(_keyed_files(cache_dir, "K_typ1_sig30.0_the30.0.")) == 1


def test_master_kernel_from_a_fortran_cache_is_still_reused_orientation_corrected(
    wlm, tmp_path, monkeypatch
):
    """
    M/Msq keep reusing a Fortran-written file with a matching manifest (the
    Wigner-3j MASTER kernel is the same operator): `_load_kernel_fits`
    corrects its transposed orientation from the data.
    """
    cache_dir = tmp_path / "cache"
    legacy = _plant_legacy_msq(wlm, cache_dir, tmp_path / "scratch")
    from astropy.io import fits as astropy_fits

    native = np.asarray(astropy_fits.getdata(str(legacy)), dtype=np.float64)
    _write_like_fortran(str(legacy), native)
    manifest = _manifest(legacy)
    manifest["backend"] = "fortran"
    _write_manifest(legacy, manifest)
    calls = _count_coupling_calls(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loaded = _get_msq(wlm, cache_dir)
    assert calls == []
    np.testing.assert_array_equal(loaded, native)


def test_polspice_kernel_k_has_no_mask_digest_in_its_key(wlm, other_wlm, tmp_path):
    """
    K is the full-sky PolSpice kernel (computed with an empty mask), so its
    key must not include mask_digest -- two MaskWlm objects with different
    masks compute the identical K kernel and share one cache file.
    """
    cache_dir = tmp_path / "shared"
    k1 = wlm.get_polspice_kernel(ELL_MAX, str(cache_dir), *APOD)
    k2 = other_wlm.get_polspice_kernel(ELL_MAX, str(cache_dir), *APOD)
    np.testing.assert_array_equal(k1, k2)
    (keyed,) = _keyed_files(cache_dir, "K_typ1_sig30.0_the30.0.")
    assert "mask_digest" not in _manifest(cache_dir / keyed)
