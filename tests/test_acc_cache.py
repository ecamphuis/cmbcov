r"""
``acc_cache`` is the on-disk kernel cache extracted out of ``acc.py``: paths,
filenames, ``np.load``/``np.save`` and the manifest JSON, with no
knowledge of Wigner symbols, spherical-harmonic integrals or their
contractions.  This file exercises it standalone, against plain arrays,
without building an ``ACCStrategy`` or a mask at all -- the point being that
none of this needs the physics.

It pins:

- a round trip through :func:`write_coupling_kernels` /
  :func:`load_coupling_kernels` preserves both the kernel arrays and the
  manifest fields;
- the four distinct error cases :func:`load_coupling_kernels` raises for a
  missing kernel file, each with its distinguishing wording: a restricted
  cache (named alongside the manifest path), an outright missing file, a
  pre-ET-fix cache (named alongside the aliased TE path), and a legacy
  ``.txt``-only cache (naming the ``convert_acc_cache`` command).

``tests/test_acc_cache_convert.py`` covers :func:`convert_text_cache`
(``.txt`` -> ``.npy``) separately.
"""

import numpy as np
import pytest

from cmbcov.approximations import acc_cache

KNOWN_SPECTRA = ("TT", "EE", "BB", "TE", "ET")


def _kernel(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(4, 4))


def test_round_trip_preserves_kernels_and_manifest(tmp_path):
    save_dir = str(tmp_path)
    kernels = {
        ("TT", "TT"): _kernel(0),
        ("EE", "EE"): _kernel(1),
        ("TE", "TE"): _kernel(2),
        ("ET", "ET"): _kernel(3),
    }

    acc_cache.write_coupling_kernels(
        save_dir,
        kernels,
        ell=16,
        ellp=17,
        known_spectra=KNOWN_SPECTRA,
        centralell=16,
        spectra=("TT", "EE", "TE", "ET"),
        grid="gl",
        lw=47,
        nside=16,
    )

    loaded = acc_cache.load_coupling_kernels(
        save_dir,
        16,
        17,
        KNOWN_SPECTRA,
        default_pairs=list(kernels.keys()),
    )
    for key, value in kernels.items():
        np.testing.assert_array_equal(loaded[key], value)

    manifest = acc_cache.read_coupling_manifest(save_dir, 16, 17)
    assert manifest["ell"] == 16
    assert manifest["ellp"] == 17
    assert manifest["spectra"] == ["TT", "EE", "TE", "ET"]
    assert manifest["grid"] == "gl"
    assert manifest["lw"] == 47
    assert manifest["nside"] == 16
    assert manifest["centralell"] == 16
    assert "git_hash" in manifest


def test_missing_manifest_is_none(tmp_path):
    assert acc_cache.read_coupling_manifest(str(tmp_path), 16, 17) is None


def test_load_raises_for_pair_outside_restricted_cache(tmp_path):
    save_dir = str(tmp_path)
    acc_cache.write_coupling_kernels(
        save_dir,
        {("TT", "TT"): _kernel(0)},
        ell=16,
        ellp=17,
        known_spectra=KNOWN_SPECTRA,
        centralell=16,
        spectra=("TT",),
        grid="gl",
        lw=47,
        nside=16,
    )

    manifest_path = acc_cache.coupling_manifest_path(save_dir, 16, 17)
    with pytest.raises(OSError) as excinfo:
        acc_cache.load_coupling_kernels(
            save_dir,
            16,
            17,
            KNOWN_SPECTRA,
            default_pairs=[("EE", "EE")],
        )
    message = str(excinfo.value)
    assert manifest_path in message
    assert "('EE', 'EE')" in message
    assert "computed for spectra" in message


def test_load_raises_plain_missing_file_with_no_manifest(tmp_path):
    save_dir = str(tmp_path)
    with pytest.raises(OSError) as excinfo:
        acc_cache.load_coupling_kernels(
            save_dir,
            16,
            17,
            KNOWN_SPECTRA,
            default_pairs=[("TT", "TT")],
        )
    message = str(excinfo.value)
    assert "Covariance coupling not found at" in message
    assert "computed for spectra" not in message
    assert "predate" not in message


def test_load_raises_for_pre_et_fix_cache(tmp_path):
    save_dir = str(tmp_path)
    # Write only the TE file directly, bypassing write_coupling_kernels, to
    # simulate a cache from before ET kernels were computed separately.
    te_path = acc_cache.coupling_save_path(
        save_dir, ("TE", "TE"), 16, 17, KNOWN_SPECTRA
    )
    import os

    os.makedirs(os.path.dirname(te_path), exist_ok=True)
    np.save(te_path, _kernel(0))

    with pytest.raises(OSError) as excinfo:
        acc_cache.load_coupling_kernels(
            save_dir,
            16,
            17,
            KNOWN_SPECTRA,
            default_pairs=[("ET", "ET")],
        )
    message = str(excinfo.value)
    assert te_path in message
    assert "predate the ET fix" in message


def test_load_raises_for_legacy_text_only_cache(tmp_path):
    """
    A ``.txt``-only cache (the on-disk format before the switch to ``.npy``)
    must raise, naming the ``.txt`` file and the ``convert_acc_cache``
    command, rather than being silently reinterpreted or upgraded in place.
    """
    save_dir = str(tmp_path)
    npy_path = acc_cache.coupling_save_path(
        save_dir, ("TT", "TT"), 16, 17, KNOWN_SPECTRA
    )
    import os

    legacy_path = os.path.splitext(npy_path)[0] + ".txt"
    os.makedirs(os.path.dirname(legacy_path), exist_ok=True)
    np.savetxt(legacy_path, _kernel(0))

    with pytest.raises(FileNotFoundError) as excinfo:
        acc_cache.load_coupling_kernels(
            save_dir,
            16,
            17,
            KNOWN_SPECTRA,
            default_pairs=[("TT", "TT")],
        )
    message = str(excinfo.value)
    assert legacy_path in message
    assert "convert_acc_cache" in message
    assert not os.path.exists(npy_path)  # never silently written at load time


def test_coupling_save_path_validates_spectra(tmp_path):
    with pytest.raises(ValueError):
        acc_cache.coupling_save_path(str(tmp_path), ("TE", "TB"), 16, 17, KNOWN_SPECTRA)

    with pytest.raises(TypeError):
        acc_cache.coupling_save_path(str(tmp_path), 123, 16, 17, KNOWN_SPECTRA)

    assert acc_cache.coupling_save_path(
        str(tmp_path), "ETxTT", 16, 17, KNOWN_SPECTRA
    ).endswith("covariance_coupling/ETxTT_16x17.npy")
