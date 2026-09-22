r"""
``acc_cache.convert_text_cache`` upgrades a legacy ``.txt`` ACC
coupling-kernel cache (the on-disk format before the switch to ``.npy``) to
``.npy`` in place.

It pins:

- every ``.txt`` kernel under ``<save_dir>/covariance_coupling/`` is
  converted, and the conversion round-trips bit-exactly against
  ``np.loadtxt`` of the ``.txt`` source;
- manifests (``manifest_*.json``) are left untouched;
- a second run is idempotent: already-converted files are skipped and not
  recounted;
- ``--remove-text`` (``remove_text=True``) only deletes ``.txt`` files once
  their ``.npy`` counterpart is verified;
- the converted cache then loads and validates through
  :func:`~cmbcov.approximations.acc_cache.load_coupling_kernels`
  exactly like a cache that was always ``.npy``.
"""

import json
import os

import numpy as np
import pytest

from cmbcov.approximations import acc_cache

KNOWN_SPECTRA = ("TT", "EE", "BB", "TE", "ET")


def _kernel(seed: int, size: int = 6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(size, size))


def _write_legacy_cache(save_dir: str) -> dict:
    """
    A cache written the way it was before the ``.npy`` switch: kernel files
    as ``.txt`` (``np.savetxt``), manifest as JSON (unaffected by the format
    change -- manifests were always JSON).
    """
    kernels = {("TT", "TT"): _kernel(0), ("EE", "EE"): _kernel(1)}
    coupling_dir = os.path.join(save_dir, "covariance_coupling")
    os.makedirs(coupling_dir, exist_ok=True)
    for (s1, s2), kernel in kernels.items():
        npy_path = acc_cache.coupling_save_path(
            save_dir, (s1, s2), 16, 17, KNOWN_SPECTRA
        )
        txt_path = os.path.splitext(npy_path)[0] + ".txt"
        np.savetxt(txt_path, kernel)
    manifest = {
        "ell": 16,
        "ellp": 17,
        "spectra": ["TT", "EE"],
        "grid": "gl",
        "lw": 47,
        "nside": 16,
        "centralell": 16,
        "git_hash": None,
    }
    manifest_path = acc_cache.coupling_manifest_path(save_dir, 16, 17)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return kernels


def test_convert_round_trips_bit_exactly(tmp_path):
    save_dir = str(tmp_path)
    kernels = _write_legacy_cache(save_dir)

    converted = acc_cache.convert_text_cache(save_dir, verbose=False)
    assert converted == len(kernels)

    for (s1, s2), kernel in kernels.items():
        npy_path = acc_cache.coupling_save_path(
            save_dir, (s1, s2), 16, 17, KNOWN_SPECTRA
        )
        assert os.path.exists(npy_path)
        np.testing.assert_array_equal(np.load(npy_path), kernel)
        # np.savetxt's default fmt='%.18e' round-trips float64 exactly.
        txt_path = os.path.splitext(npy_path)[0] + ".txt"
        np.testing.assert_array_equal(np.load(npy_path), np.loadtxt(txt_path))


def test_convert_leaves_manifests_untouched(tmp_path):
    save_dir = str(tmp_path)
    _write_legacy_cache(save_dir)
    manifest_path = acc_cache.coupling_manifest_path(save_dir, 16, 17)
    before = json.load(open(manifest_path))

    acc_cache.convert_text_cache(save_dir, verbose=False)

    after = json.load(open(manifest_path))
    assert before == after


def test_convert_is_idempotent(tmp_path):
    save_dir = str(tmp_path)
    _write_legacy_cache(save_dir)

    first = acc_cache.convert_text_cache(save_dir, verbose=False)
    assert first == 2

    second = acc_cache.convert_text_cache(save_dir, verbose=False)
    assert second == 0  # already converted, nothing new done

    # The .txt files are untouched (remove_text was never passed).
    coupling_dir = os.path.join(save_dir, "covariance_coupling")
    txt_files = [f for f in os.listdir(coupling_dir) if f.endswith(".txt")]
    assert len(txt_files) == 2


def test_convert_returns_zero_for_a_directory_with_no_legacy_cache(tmp_path):
    save_dir = str(tmp_path)
    assert acc_cache.convert_text_cache(save_dir, verbose=False) == 0


def test_remove_text_only_deletes_after_verification(tmp_path):
    save_dir = str(tmp_path)
    _write_legacy_cache(save_dir)
    coupling_dir = os.path.join(save_dir, "covariance_coupling")

    converted = acc_cache.convert_text_cache(save_dir, remove_text=True, verbose=False)
    assert converted == 2

    remaining = os.listdir(coupling_dir)
    assert not any(f.endswith(".txt") for f in remaining)
    assert sum(f.endswith(".npy") for f in remaining) == 2
    assert "manifest_16x17.json" in remaining


def test_remove_text_false_keeps_the_text_files_by_default(tmp_path):
    save_dir = str(tmp_path)
    _write_legacy_cache(save_dir)
    coupling_dir = os.path.join(save_dir, "covariance_coupling")

    acc_cache.convert_text_cache(save_dir, verbose=False)

    remaining = os.listdir(coupling_dir)
    assert sum(f.endswith(".txt") for f in remaining) == 2


def test_converted_cache_loads_and_validates(tmp_path):
    save_dir = str(tmp_path)
    kernels = _write_legacy_cache(save_dir)
    acc_cache.convert_text_cache(save_dir, remove_text=True, verbose=False)

    loaded = acc_cache.load_coupling_kernels(
        save_dir,
        16,
        17,
        KNOWN_SPECTRA,
        default_pairs=list(kernels.keys()),
        current={"mask_digest": None, "centralell": 16},
    )
    for key, kernel in kernels.items():
        np.testing.assert_array_equal(loaded[key], kernel)


def test_mismatched_existing_npy_raises_loudly(tmp_path):
    """
    A ``.npy`` that already exists but does not match its ``.txt`` source is
    a genuine inconsistency (e.g. a hand-edited file); the converter must
    refuse to silently paper over it.
    """
    save_dir = str(tmp_path)
    _write_legacy_cache(save_dir)
    npy_path = acc_cache.coupling_save_path(
        save_dir, ("TT", "TT"), 16, 17, KNOWN_SPECTRA
    )
    np.save(npy_path, _kernel(99))  # deliberately wrong content

    with pytest.raises(ValueError, match="does not match"):
        acc_cache.convert_text_cache(save_dir, verbose=False)
