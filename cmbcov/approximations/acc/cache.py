"""
On-disk cache for ACC coupling kernels.

This module owns ``save_dir``, filenames, ``np.load``/``np.save`` and
the manifest JSON for the coupling kernels computed by
:mod:`cmbcov.approximations.acc`.  It knows nothing about the underlying
maths: callers pass in the set of valid single-spectrum labels (e.g.
``acc.COUPLING_SPECTRA``) purely as strings to validate against.
"""

import glob
import json
import os
import re
import subprocess
import warnings
from collections.abc import Iterable, Sequence

import numpy as np

__all__ = [
    "git_hash",
    "HEALPIX_KERNEL_VERSION",
    "coupling_save_path",
    "coupling_manifest_path",
    "coupling_kernel_candidates",
    "kernel_file_pair",
    "locate_coupling_kernel",
    "read_coupling_manifest",
    "write_coupling_kernels",
    "write_generation",
    "load_coupling_kernels",
    "convert_text_cache",
]

#: Version of the ``grid="healpix"`` coupling-kernel convention recorded in the
#: manifest as ``healpix_kernel_version`` (the GL path records nothing, like
#: ``map2alm_iter``). Bumped whenever a change alters the numbers a
#: HEALPix-grid precompute writes, so an incompatible cache is refused rather
#: than silently reused:
#:
#: 1. Theta built from ``Re X(L1, L2)`` only (biased for a mask without
#:    azimuthal symmetry); a manifest without the field counts as this.
#: 2. Theta keeps ``X(L1)``, ``X(L2)`` complex through the contraction
#:    (Eq. 20).
HEALPIX_KERNEL_VERSION = 2

#: Manifest fields with no record on ``Cov`` (they are choices made at
#: precompute call time, ``precompute_acc_kernels(nside=, lw=, grid=)``, not
#: part of ``CovarianceConfig``).  They are validated for *session*
#: consistency: the first manifest read (or kernel write) for a given
#: ``save_dir`` in this process establishes the reference, and every later
#: manifest for that ``save_dir`` is compared against it.  See
#: :func:`_validate_cache_identity`.
_REFERENCE_FIELDS = ("nside", "lw", "grid")

#: ``{abspath(save_dir): {"nside":, "lw":, "grid":}}`` -- the session
#: reference described above.  Module-level so that it survives across the
#: many :func:`load_coupling_kernels` calls one covariance computation makes
#: (one per diagonal offset per block).
_CACHE_REFERENCE: dict[str, dict] = {}

#: ``{abspath(save_dir): n}`` -- how many :func:`write_coupling_kernels` calls
#: this process has made into each ``save_dir``.  An in-memory kernel memo
#: (``Cov.get_acc_coupling_kernels``) records the count at load time and
#: reloads when it has moved, so a write by any caller in this process (such
#: as :func:`~cmbcov.approximations.acc.precompute_acc_kernels`) is never
#: shadowed by a stale memo entry.
_WRITE_GENERATION: dict[str, int] = {}

#: ``save_dir`` (abspath) already warned about an unverifiable cache identity
#: in this process. The loader runs inside a loop over every block and
#: offset, so this limits the warning to once.
_WARNED_SAVE_DIRS: set = set()


def git_hash() -> str | None:
    """
    The current commit hash, for the coupling-cache manifest.  ``None`` (not
    an exception) if this is not a git checkout or ``git`` is unavailable.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(__file__),
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def write_generation(save_dir: str) -> int:
    """Number of :func:`write_coupling_kernels` calls into ``save_dir`` in this process."""
    return _WRITE_GENERATION.get(os.path.abspath(save_dir), 0)


def coupling_save_path(
    save_dir: str,
    stokes_key: str | tuple,
    ell: int,
    ell_prime: int,
    known_spectra: Sequence[str],
) -> str:
    """
    Path for saving one coupling-kernel file under ``save_dir``.

    ``known_spectra`` is the caller's set of valid single-spectrum labels
    (e.g. :data:`~cmbcov.approximations.acc.COUPLING_SPECTRA`);
    this module only checks membership in it.
    """
    if isinstance(stokes_key, str):
        stokes_key = stokes_key.split("x")
    elif not isinstance(stokes_key, (list, tuple)):
        raise TypeError(f"stokes_key should be str or tuple, got {type(stokes_key)}")

    stokes_1, stokes_2 = stokes_key
    for spec in (stokes_1, stokes_2):
        if spec not in known_spectra:
            raise ValueError(
                f"unknown coupling spectrum {spec!r}; expected one of "
                f"{known_spectra}"
            )
    # The letters are NOT sorted: TD (T at l, D at l') and DT (D at l, T
    # at l') are different kernels and live in different files.
    filename = f"covariance_coupling/{stokes_1}x{stokes_2}_{ell}x{ell_prime}.npy"
    return os.path.join(save_dir, filename)


def _legacy_text_path(npy_path: str) -> str:
    """The legacy ``.txt`` path for a ``.npy`` kernel path."""
    return os.path.splitext(npy_path)[0] + ".txt"


def coupling_manifest_path(save_dir: str, ell: int, ell_prime: int) -> str:
    """Path of the manifest written alongside the kernels of one ``(ell, ell_prime)`` pair."""
    filename = f"covariance_coupling/manifest_{ell}x{ell_prime}.json"
    return os.path.join(save_dir, filename)


def _legacy_npy_path(
    save_dir: str, stokes_key: tuple[str, str], ell: int, ell_prime: int
) -> str:
    """
    The on-disk path a legacy-named pair would be saved under, built like
    :func:`coupling_save_path` but without the ``known_spectra`` membership
    check (legacy labels are not in the current label set).
    """
    stokes_1, stokes_2 = stokes_key
    return os.path.join(
        save_dir, "covariance_coupling/" f"{stokes_1}x{stokes_2}_{ell}x{ell_prime}.npy"
    )


def coupling_kernel_candidates(
    save_dir: str,
    stokes_key: tuple[str, str],
    ell: int,
    ell_prime: int,
    known_spectra: Sequence[str],
    legacy_names: dict[str, str] | None = None,
) -> list[tuple[str, bool]]:
    """
    Every on-disk path that can serve the kernel of ``stokes_key`` at
    ``(ell, ell_prime)``, in fallback order: the pair's own name, its legacy
    name, the transposed pair's own name, the transposed pair's legacy name.

    Each entry is ``(path, transposed)``. ``transposed`` is ``True`` when
    the file holds the kernel of ``(stokes_2, stokes_1)`` rather than
    ``(stokes_1, stokes_2)``; by the kernel-transpose identity
    (docs/theory/bmode_kernels.md,
    :math:`\\Theta^{ab\\times cd}(L_1, L_2) = \\Theta^{cd\\times ab}(L_2, L_1)`),
    the array at that path must then be transposed to serve as this pair's
    kernel. A candidate may appear more than once (e.g. for a diagonal pair,
    whose transpose is itself); callers should skip a path already ruled out.

    ``legacy_names`` is the same canonical-label -> legacy-on-disk-label map
    :func:`load_coupling_kernels` takes; ``None`` skips the legacy
    candidates entirely, leaving only the pair's own name and its
    transpose.
    """
    stokes_1, stokes_2 = stokes_key
    swapped_key = (stokes_2, stokes_1)
    candidates = [
        (coupling_save_path(save_dir, stokes_key, ell, ell_prime, known_spectra), False)
    ]
    if legacy_names is not None:
        legacy_key = tuple(legacy_names.get(s, s) for s in stokes_key)
        if legacy_key != stokes_key:
            candidates.append(
                (_legacy_npy_path(save_dir, legacy_key, ell, ell_prime), False)
            )
    if swapped_key != stokes_key:
        candidates.append(
            (
                coupling_save_path(
                    save_dir, swapped_key, ell, ell_prime, known_spectra
                ),
                True,
            )
        )
        if legacy_names is not None:
            legacy_swapped = tuple(legacy_names.get(s, s) for s in swapped_key)
            if legacy_swapped != swapped_key:
                candidates.append(
                    (_legacy_npy_path(save_dir, legacy_swapped, ell, ell_prime), True)
                )
    return candidates


def kernel_file_pair(path: str) -> str:
    """The ``"s1xs2"`` label of a kernel file, from its name
    (``.../covariance_coupling/s1xs2_<ell>x<ell_prime>.npy``) -- the pair as
    stored on disk, whatever orientation or name a caller asked for."""
    return os.path.basename(path).split("_", 1)[0]


def _pair_label(key) -> str:
    """``"s1xs2"`` for a kernel key given as ``"s1xs2"`` or ``(s1, s2)``."""
    return key if isinstance(key, str) else "x".join(key)


def locate_coupling_kernel(
    save_dir: str,
    stokes_key: tuple[str, str],
    ell: int,
    ell_prime: int,
    known_spectra: Sequence[str],
    legacy_names: dict[str, str] | None = None,
    manifest: dict | bool | None = None,
) -> tuple[tuple[str, bool] | None, list[str]]:
    """
    The file that serves the kernel of ``stokes_key`` at ``(ell,
    ell_prime)``, exactly as :func:`load_coupling_kernels` picks it.

    Returns ``(found, unrecorded)``. ``found`` is ``(path, transposed)`` of
    the first of :func:`coupling_kernel_candidates` (own name, legacy name,
    transposed own name, transposed legacy name) that exists and that the
    manifest's pair record lists, or ``None``. ``unrecorded`` lists the
    candidates that exist but are NOT in that record: files left over from
    an earlier cache with a different identity
    (:func:`write_coupling_kernels`), which are never served. A manifest with
    no ``pairs`` record, or a falsy ``manifest`` (none), accepts every file.
    """
    recorded = None
    if manifest and manifest.get("pairs") is not None:
        recorded = set(manifest["pairs"])
    stokes_key = tuple(stokes_key)
    unrecorded = []
    for path, transposed in coupling_kernel_candidates(
        save_dir, stokes_key, ell, ell_prime, known_spectra, legacy_names
    ):
        if not os.path.exists(path):
            continue
        # A legacy-named file holds the kernel of the pair it stands for,
        # so either its own name or that pair's may be the recorded one.
        represents = "x".join(stokes_key[::-1] if transposed else stokes_key)
        if recorded is not None and not (
            kernel_file_pair(path) in recorded or represents in recorded
        ):
            if path not in unrecorded:
                unrecorded.append(path)
            continue
        return (path, transposed), unrecorded
    return None, unrecorded


def read_coupling_manifest(save_dir: str, ell: int, ell_prime: int) -> dict | None:
    """
    Read the manifest :func:`write_coupling_kernels` writes next to the
    kernels of one ``(ell, ell_prime)`` pair, or ``None`` if there is none.

    The manifest records ``spectra`` (the channels that pair was computed
    for), ``pairs`` (the kernel files written, as ``"s1xs2"``; absent in
    manifests written by older versions), ``grid``, ``lw``, ``nside``,
    ``centralell`` and ``git_hash``, plus the optional identity fields of
    :func:`write_coupling_kernels`. This lets :func:`load_coupling_kernels`
    tell "the cache was built for a smaller spectrum set" apart from "this
    file is missing". ``spectra`` and ``pairs`` cover every write into the
    directory with the same identity.
    """
    path = coupling_manifest_path(save_dir, ell, ell_prime)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def write_coupling_kernels(
    save_dir: str,
    coupling_kernels: dict,
    ell: int,
    ellp: int,
    known_spectra: Sequence[str],
    centralell: int | None,
    verbose: bool = False,
    spectra: Sequence[str] = (),
    grid: str = "gl",
    lw: int | None = None,
    nside: int | None = None,
    mask_digest: str | None = None,
    map2alm_iter: int | None = None,
    healpix_kernel_version: int | None = None,
    term_selection: float | None = None,
    term_selection_stats: dict | None = None,
) -> None:
    """
    Save coupling kernels to disk, plus a manifest recording what was
    computed (see :func:`read_coupling_manifest`).

    Parameters
    ----------
    save_dir : str
        Cache root; files go to ``<save_dir>/covariance_coupling/``.
    coupling_kernels : dict
        Kernel arrays keyed by ``"s1xs2"`` or ``(s1, s2)``.
    ell, ellp : int
        Multipole pair of the kernels.
    known_spectra : sequence of str
        Valid labels (see :func:`coupling_save_path`).
    centralell : int or None
        Reference multipole recorded in the manifest.
    verbose : bool
        Print each file written.
    spectra : sequence of str
        Channels the precompute was asked for, recorded as ``spectra``.
    grid : str
        ``"gl"`` or ``"healpix"``.
    lw, nside : int, optional
        Mask band-limit and ACC resolution, recorded in the manifest.
    mask_digest : str, optional
        Content digest of the mask map the precompute consumed
        (:attr:`~cmbcov.mask.MaskWlm.mask_digest`), so
        :func:`load_coupling_kernels` can tell a cache built for a different
        mask. Left out of the manifest when ``None``.
    map2alm_iter : int, optional
        Jacobi refinement count of the HEALPix ``map2alm`` that built the
        integrals (``grid="healpix"`` only). Left out when ``None``.
    healpix_kernel_version : int, optional
        :data:`HEALPIX_KERNEL_VERSION` (``grid="healpix"`` only). Left out
        when ``None``.
    term_selection : float, optional
        A-priori term-selection tolerance the kernels were built with
        (:mod:`cmbcov.approximations.acc.term_selection`), ``None`` for the full contraction.
        Always written (``null`` when ``None``); a manifest without the
        field is read as ``None`` (:func:`_validate_cache_identity`).
    term_selection_stats : dict, optional
        Fractions of orders, pairs and ``M`` band kept at this ``(ell,
        ellp)``; recorded when given.

    Notes
    -----
    This also updates this ``save_dir``'s session reference for ``nside``,
    ``lw`` and ``grid`` (see :func:`_validate_cache_identity`).

    **Extending a cache.** When the directory already holds a manifest for
    ``(ell, ellp)`` with the same identity (every field of
    :data:`_MANIFEST_IDENTITY_FIELDS`), the new write extends that cache:
    ``pairs`` and ``spectra`` become the union of both records (``spectra``
    in ``known_spectra`` order), so adding pairs in several precomputes gives
    the manifest of a one-shot precompute of their union (``git_hash`` is the
    latest write's). Otherwise the manifest records this write alone, and the
    old manifest is removed before any kernel file is written, so an
    interrupted write never leaves kernels certified by the wrong identity.
    :func:`load_coupling_kernels` never serves a file the ``pairs`` record
    does not list, so kernels left by an earlier cache are refused rather
    than validated against this write's identity.
    """
    os.makedirs(
        os.path.dirname(
            coupling_save_path(save_dir, "TTxTT", ell, ellp, known_spectra)
        ),
        exist_ok=True,
    )
    abs_dir = os.path.abspath(save_dir)
    # Bumped before any file is touched, so a memo can never record a
    # generation that predates a partially completed write.
    _WRITE_GENERATION[abs_dir] = _WRITE_GENERATION.get(abs_dir, 0) + 1

    manifest = {
        "ell": ell,
        "ellp": ellp,
        "spectra": list(spectra),
        "pairs": sorted({_pair_label(key) for key in coupling_kernels}),
        "grid": grid,
        "lw": lw,
        "nside": nside,
        "centralell": centralell,
        "git_hash": git_hash(),
        "term_selection": None if term_selection is None else float(term_selection),
    }
    if term_selection_stats is not None:
        manifest["term_selection_stats"] = {
            k: (int(v) if isinstance(v, (int, np.integer)) else float(v))
            for k, v in term_selection_stats.items()
        }
    if mask_digest is not None:
        manifest["mask_digest"] = mask_digest
    if map2alm_iter is not None:
        manifest["map2alm_iter"] = map2alm_iter
    if healpix_kernel_version is not None:
        manifest["healpix_kernel_version"] = healpix_kernel_version
    manifest_path = coupling_manifest_path(save_dir, ell, ellp)
    existing = read_coupling_manifest(save_dir, ell, ellp)
    if existing is not None and not _same_identity(existing, manifest):
        os.remove(manifest_path)
        existing = None
    manifest = _merged_manifest(existing, manifest, known_spectra)

    for key in coupling_kernels.keys():
        save_path = coupling_save_path(save_dir, key, ell, ellp, known_spectra)
        try:
            # save_path already ends in .npy; np.save would append another
            # .npy if it didn't, but never doubles an existing one.
            np.save(save_path, coupling_kernels[key])
            if verbose:
                print(f"  Saved {key} coupling to {save_path}")
        except Exception as e:
            print(f"  Error saving {key}: {e}")
            raise

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    if verbose:
        print(f"  Saved manifest to {manifest_path}")

    _CACHE_REFERENCE[os.path.abspath(save_dir)] = {
        field: manifest.get(field) for field in _REFERENCE_FIELDS
    }


#: Manifest fields that identify how the kernels of one ``(ell, ellp)``
#: were computed: a write whose manifest agrees with the one on disk on all
#: of them extends that cache (:func:`write_coupling_kernels`); any
#: difference, a field present in one and absent in the other included,
#: starts a new one. ``git_hash`` is deliberately not one of them (a code
#: change alone does not invalidate a cache).
_MANIFEST_IDENTITY_FIELDS = (
    "ell",
    "ellp",
    "centralell",
    "mask_digest",
    "grid",
    "lw",
    "nside",
    "term_selection",
    "term_selection_stats",
    "map2alm_iter",
    "healpix_kernel_version",
)

_ABSENT = object()


def _same_identity(existing: dict, new: dict) -> bool:
    """Whether two manifests agree on every :data:`_MANIFEST_IDENTITY_FIELDS`
    field (absent counts as a value of its own)."""
    return all(
        existing.get(field, _ABSENT) == new.get(field, _ABSENT)
        for field in _MANIFEST_IDENTITY_FIELDS
    )


def _merged_manifest(
    existing: dict | None, new: dict, known_spectra: Sequence[str]
) -> dict:
    """
    The manifest to write for ``new`` given the one on disk: ``new`` itself
    unless ``existing`` has the same identity (:func:`_same_identity`), in
    which case ``spectra`` and ``pairs`` are the unions of both records.
    ``spectra`` is then in ``known_spectra`` order (labels outside it after,
    in first-seen order), as a one-shot precompute of the union writes it.
    ``pairs`` stays absent if ``existing`` has no ``pairs`` record (a
    manifest from an older version: which files it covered is not known, so
    no record would be complete).
    """
    if existing is None or not _same_identity(existing, new):
        return new
    merged = dict(new)
    seen = list(existing.get("spectra", []))
    seen += [s for s in new.get("spectra", []) if s not in seen]
    merged["spectra"] = [s for s in known_spectra if s in seen] + [
        s for s in seen if s not in known_spectra
    ]
    if existing.get("pairs") is None:
        merged.pop("pairs", None)
    else:
        merged["pairs"] = sorted(set(existing["pairs"]) | set(new["pairs"]))
    return merged


def _unverifiable_summary(save_dir: str) -> str:
    """
    How much of the cache under ``save_dir`` cannot be verified: the kernel
    pairs with no manifest and the manifests with no ``mask_digest``, each
    counted with the ``(ell, ell_prime)`` range they span.
    """
    directory = os.path.join(save_dir, "covariance_coupling")
    try:
        names = os.listdir(directory)
    except OSError:
        return ""
    pair_re = re.compile(r"^[A-Z]{2}x[A-Z]{2}_(\d+)x(\d+)\.(?:npy|txt)$")
    manifest_re = re.compile(r"^manifest_(\d+)x(\d+)\.json$")
    kernel_pairs = set()
    manifests = {}
    for name in names:
        match = pair_re.match(name)
        if match:
            kernel_pairs.add((int(match.group(1)), int(match.group(2))))
        match = manifest_re.match(name)
        if match:
            manifests[(int(match.group(1)), int(match.group(2)))] = name
    no_manifest = sorted(kernel_pairs - set(manifests))
    no_digest = []
    for pair, name in sorted(manifests.items()):
        try:
            with open(os.path.join(directory, name)) as f:
                if "mask_digest" not in json.load(f):
                    no_digest.append(pair)
        except (OSError, ValueError):
            no_digest.append(pair)

    def span(pairs):
        return f"{pairs[0]}..{pairs[-1]}" if len(pairs) > 1 else f"{pairs[0]}"

    parts = []
    if no_manifest:
        parts.append(
            f"{len(no_manifest)} of {len(kernel_pairs)} kernel pairs have no "
            f"manifest ({span(no_manifest)})"
        )
    if no_digest:
        parts.append(
            f"{len(no_digest)} of {len(manifests)} manifests have no mask_digest "
            f"({span(no_digest)})"
        )
    return "; ".join(parts)


def _warn_unverifiable_once(save_dir: str, message: str) -> None:
    """
    Emit ``message`` as a ``UserWarning``, at most once per ``save_dir`` per
    process, followed by a count and ``(ell, ell_prime)`` range of everything
    in that cache that cannot be verified (the same condition holds for every
    pair a run loads).
    """
    abs_dir = os.path.abspath(save_dir)
    if abs_dir in _WARNED_SAVE_DIRS:
        return
    _WARNED_SAVE_DIRS.add(abs_dir)
    summary = _unverifiable_summary(save_dir)
    if summary:
        message = (
            f"{message} In this cache: {summary}. (Warned once per cache directory.)"
        )
    warnings.warn(message, UserWarning, stacklevel=4)


def _raise_identity_mismatch(
    field: str,
    cached_value,
    run_value,
    save_dir: str,
    manifest_path: str,
    detail: str | None = None,
) -> None:
    raise ValueError(
        f"ACC coupling cache mismatch in {field!r}: the cache at "
        f"{manifest_path} (save_dir={save_dir!r}) was built with "
        f"{field}={cached_value!r}, but this run has {field}={run_value!r}. "
        + (f"{detail} " if detail else "")
        + "Recompute the coupling cache for this run (precompute_acc_kernels or "
        "the cmbcov-precompute command) "
        "or point save_dir at the cache that matches it."
    )


def _validate_cache_identity(
    manifest: dict | bool | None,
    current: dict,
    save_dir: str,
    ell: int,
    ell_prime: int,
) -> None:
    """
    Compare a manifest's recorded identity against ``current`` (this run's
    values), raising on a mismatch and warning once per ``save_dir`` when a
    field cannot be checked at all.

    ``current`` carries ``mask_digest`` (a content digest of the mask map,
    see :attr:`~cmbcov.mask.MaskWlm.mask_digest`),
    ``centralell`` and, optionally, ``map2alm_iter`` and
    ``healpix_kernel_version`` (both checked for ``grid="healpix"`` caches
    only; a missing ``map2alm_iter`` means 0, a missing
    ``healpix_kernel_version`` means 1, see :data:`HEALPIX_KERNEL_VERSION`)
    and ``term_selection`` (the term-selection tolerance the run expects its
    kernels to have been built with, ``None`` for the full kernels; checked
    whenever the key is present in ``current``, a manifest without the field
    meaning ``None``, so a run asking for full kernels never loads selected
    ones and vice versa). All are known independently of the cache.
    ``nside``, ``lw`` and ``grid`` have no such independent value, so they
    are checked for *session* consistency: the first manifest seen for this
    ``save_dir`` in this process (or a kernel write, see
    :func:`write_coupling_kernels`) establishes the reference, and every
    later manifest for the same ``save_dir`` must match it.

    A manifest missing entirely, or missing the ``mask_digest`` field, is
    unverifiable rather than wrong: this warns instead of raising, naming
    the fields that could not be checked.
    """
    manifest_path = coupling_manifest_path(save_dir, ell, ell_prime)

    if not manifest:
        _warn_unverifiable_once(
            save_dir,
            f"ACC coupling cache at {save_dir!r} has no manifest for "
            f"(ell, ell_prime) = ({ell}, {ell_prime}) (expected at "
            f"{manifest_path!r}); mask_digest, nside, lw, grid and "
            "centralell cannot be checked against this run. Recomputing "
            "the precompute (precompute_acc_kernels) adds "
            "verification.",
        )
        return

    unverifiable_fields = []
    if "mask_digest" not in manifest:
        unverifiable_fields.append("mask_digest")
    elif manifest["mask_digest"] != current.get("mask_digest"):
        _raise_identity_mismatch(
            "mask_digest",
            manifest["mask_digest"],
            current.get("mask_digest"),
            save_dir,
            manifest_path,
        )

    if manifest.get("centralell") != current.get("centralell"):
        _raise_identity_mismatch(
            "centralell",
            manifest.get("centralell"),
            current.get("centralell"),
            save_dir,
            manifest_path,
        )

    if "term_selection" in current:
        cached_ts = manifest.get("term_selection")  # absent: None
        run_ts = current["term_selection"]
        same = (cached_ts is None and run_ts is None) or (
            cached_ts is not None
            and run_ts is not None
            and float(cached_ts) == float(run_ts)
        )
        if not same:
            if cached_ts is None:
                detail = (
                    "The cache holds the full kernels (no term selection) but "
                    "this run asks for kernels built with a-priori term "
                    "selection."
                )
            elif run_ts is None:
                detail = (
                    "The cache holds kernels built with a-priori term selection "
                    "(an approximation at that tolerance) but this run asks for "
                    "the full kernels."
                )
            else:
                detail = "The term-selection tolerances differ."
            _raise_identity_mismatch(
                "term_selection", cached_ts, run_ts, save_dir, manifest_path, detail
            )

    # A HEALPix-grid cache with no recorded map2alm_iter counts as iter=0
    # (a plain adjoint, which differs from refined kernels), so a mismatch
    # is refused, not warned on.
    if manifest.get("grid") == "healpix" and current.get("map2alm_iter") is not None:
        cached_iter = manifest.get("map2alm_iter", 0)
        if cached_iter != current["map2alm_iter"]:
            _raise_identity_mismatch(
                "map2alm_iter",
                cached_iter,
                current["map2alm_iter"],
                save_dir,
                manifest_path,
            )

    # A HEALPix-grid cache with no recorded healpix_kernel_version counts as
    # version 1 (Theta built from Re X only, biased for a mask without
    # azimuthal symmetry), so a mismatch is refused, not warned on. A cache
    # with a *higher* version than this package knows was built by a newer
    # package and is refused too.
    if (
        manifest.get("grid") == "healpix"
        and current.get("healpix_kernel_version") is not None
    ):
        cached_version = manifest.get("healpix_kernel_version", 1)
        current_version = current["healpix_kernel_version"]
        if cached_version != current_version:
            if cached_version < current_version:
                detail = (
                    "This cache was written with an older kernel definition "
                    "and must be recomputed."
                )
            else:
                detail = (
                    "This cache was built by a newer version of the package "
                    "than this one knows how to validate."
                )
            _raise_identity_mismatch(
                "healpix_kernel_version",
                cached_version,
                current_version,
                save_dir,
                manifest_path,
                detail=detail,
            )

    abs_dir = os.path.abspath(save_dir)
    reference = _CACHE_REFERENCE.get(abs_dir)
    if reference is None:
        _CACHE_REFERENCE[abs_dir] = {
            field: manifest.get(field) for field in _REFERENCE_FIELDS
        }
    else:
        for field in _REFERENCE_FIELDS:
            if manifest.get(field) != reference.get(field):
                _raise_identity_mismatch(
                    field,
                    manifest.get(field),
                    reference.get(field),
                    save_dir,
                    manifest_path,
                )

    if unverifiable_fields:
        _warn_unverifiable_once(
            save_dir,
            f"ACC coupling cache manifest at {manifest_path!r} (cache at "
            f"{save_dir!r}) does not record {unverifiable_fields}, so this "
            "cannot be checked against this run's mask. Recomputing the "
            "precompute (precompute_acc_kernels) adds verification.",
        )


def _spectra_covered(
    stokes_key: tuple[str, str],
    manifest_spectra: Sequence[str],
    legacy_names: dict[str, str] | None,
) -> bool:
    """
    Whether both labels of ``stokes_key`` are covered by ``manifest_spectra``
    -- directly, or (when ``legacy_names`` is given) under their legacy
    on-disk name, so a manifest recorded under legacy channel names still
    covers a request made with the canonical ones.
    """

    def one(label: str) -> bool:
        if label in manifest_spectra:
            return True
        return legacy_names is not None and legacy_names.get(label) in manifest_spectra

    return all(one(label) for label in stokes_key)


def load_coupling_kernels(
    save_dir: str,
    ell: int,
    ell_prime: int,
    known_spectra: Sequence[str],
    default_pairs: Sequence[tuple[str, str]],
    pairs: Iterable[tuple[str, str]] | None = None,
    current: dict | None = None,
    legacy_names: dict[str, str] | None = None,
) -> dict:
    """
    Load covariance coupling kernels from disk.

    Parameters
    ----------
    known_spectra : sequence of str
        Valid single-spectrum labels (membership only; no meaning attached).
    default_pairs : sequence of (str, str)
        The ``(stokes_1, stokes_2)`` pairs to load when ``pairs`` is ``None``.
    pairs : iterable of (str, str), optional
        The ``(stokes_1, stokes_2)`` kernel pairs to load, each built from
        ``known_spectra``.  Defaults to ``default_pairs``.
    legacy_names : dict of str to str, optional
        Canonical label -> legacy on-disk label (e.g. ``{"DD": "EE", "LL":
        "BB", "TD": "TE", "DT": "ET"}``).  A string-to-string fallback: when
        the file for a requested pair's canonical labels is absent, the file
        for the pair with every label passed through this map is tried next,
        so a cache written under legacy labels loads without recompute.
        ``None`` (default) skips this fallback.
    current : dict, optional
        This run's identity: ``mask_digest`` and ``centralell``, and
        optionally ``map2alm_iter``, ``healpix_kernel_version`` and
        ``term_selection`` (see :func:`_validate_cache_identity`).  ``None``
        (default) skips identity validation, for a plain load with no
        ``Cov`` to draw values from.  Otherwise, on every load: a mismatched
        field raises (naming the field, the cached and the run's value, and
        ``save_dir``); a field that cannot be checked at all (no manifest,
        or none with a ``mask_digest``) warns once per ``save_dir`` per
        process instead.

    Raises
    ------
    ValueError
        If ``current`` is given and a manifest's recorded ``mask_digest``,
        ``nside``, ``lw``, ``grid``, ``centralell`` or ``term_selection``
        does not match this run (see ``current`` above and
        :func:`_validate_cache_identity`).
        ``git_hash`` is recorded but never validated -- a code change alone
        must not invalidate a cache.
    FileNotFoundError
        If the ``.npy`` kernel is missing but a legacy ``.txt`` file is
        present.  The message names the ``.txt`` file and the
        ``convert_acc_cache`` command that converts it; nothing is converted
        at load time.
    OSError
        If a kernel file is missing outright.  The message names the missing
        pair. If a manifest exists for ``(ell, ell_prime)`` (written by
        :func:`write_coupling_kernels`) and it shows the cache was built for
        a spectrum set not covering this pair, the message says so and that
        the cache must be recomputed; there is never a silent substitution.
        Likewise a missing ``DT`` (or ``ET``) file whose ``TD`` (``TE``)
        counterpart exists is an error: they are different kernels. A file
        that exists but that the manifest's ``pairs`` record does not list
        (left over from an earlier cache with another identity,
        :func:`write_coupling_kernels`) is not served either: the message
        names it.
    """
    if pairs is None:
        pairs = default_pairs
    else:
        pairs = list(pairs)

    coupling_kernels = {}
    # False when there is none. Read up front: its pair record decides which
    # files may be served (locate_coupling_kernel).
    manifest: dict | bool = read_coupling_manifest(save_dir, ell, ell_prime) or False

    if current is not None:
        _validate_cache_identity(manifest, current, save_dir, ell, ell_prime)

    for stokes_1, stokes_2 in pairs:
        stokes_key = (stokes_1, stokes_2)
        for spec in stokes_key:
            if spec not in known_spectra:
                raise ValueError(
                    f"unknown coupling spectrum {spec!r} in pair {stokes_key!r}; "
                    f"expected one of {known_spectra}"
                )
        coupling_path = coupling_save_path(
            save_dir, stokes_key, ell, ell_prime, known_spectra
        )

        # Own name, legacy name, transposed own name, transposed legacy
        # name (docs/theory/bmode_kernels.md): a cache that only stores the
        # other orientation of this pair still serves it, exactly.
        found, unrecorded = locate_coupling_kernel(
            save_dir,
            stokes_key,
            ell,
            ell_prime,
            known_spectra,
            legacy_names,
            manifest,
        )
        if found is not None:
            path, transposed = found
            array = np.load(path)
            coupling_kernels[stokes_key] = (
                np.ascontiguousarray(array.T) if transposed else array
            )
            continue
        if unrecorded:
            manifest_path = coupling_manifest_path(save_dir, ell, ell_prime)
            raise OSError(
                f"Covariance coupling of {stokes_key!r} not served: "
                f"{', '.join(unrecorded)} exists but the manifest "
                f"{manifest_path} does not list it (its record: "
                f"{manifest['pairs']}). The last precompute into this "
                "directory had a different identity (mask, centralell, grid, "
                "lw, nside or term selection) and did not rewrite this pair, "
                "so the file is left over from an earlier cache. Recompute it "
                "with precompute_acc_kernels(..., pairs=[...]) using this "
                "run's settings, or remove it."
            )

        legacy_path = _legacy_text_path(coupling_path)
        if os.path.exists(legacy_path):
            raise FileNotFoundError(
                f"Legacy text-format ACC coupling cache found at "
                f"{legacy_path!r}: the on-disk cache format changed from "
                ".txt to .npy. Convert the cache directory once with:\n"
                "    python -m cmbcov.scripts."
                f"convert_acc_cache {save_dir!r}\n"
                "(add --remove-text to delete the .txt files after "
                "verification), then rerun."
            )

        covered = manifest and _spectra_covered(
            (stokes_1, stokes_2), manifest["spectra"], legacy_names
        )
        if manifest and not covered:
            manifest_path = coupling_manifest_path(save_dir, ell, ell_prime)
            raise OSError(
                f"Covariance coupling not found at {coupling_path}: the "
                f"cache at {manifest_path} was computed for spectra "
                f"{manifest['spectra']} only (grid={manifest['grid']!r}), "
                f"not {stokes_key!r}. Recompute with "
                "precompute_acc_kernels(..., spectra=...) covering it."
            )

        # The reversed-order channel, under both its legacy name ("ET") and
        # its canonical one ("DT", see CHANNEL_ALIASES in acc.py): it is a
        # different kernel from the direct-order ("TE"/"TD") one, so the
        # direct-order file is never substituted for it.
        for reversed_label, direct_label in (("ET", "TE"), ("DT", "TD")):
            if reversed_label not in stokes_key:
                continue
            aliased_key = tuple(
                direct_label if s == reversed_label else s for s in stokes_key
            )
            aliased_path = os.path.join(
                save_dir,
                "covariance_coupling/"
                f"{aliased_key[0]}x{aliased_key[1]}_{ell}x{ell_prime}.npy",
            )
            if os.path.exists(aliased_path):
                raise OSError(
                    f"Covariance coupling not found at {coupling_path}, "
                    f"but {aliased_path} exists: the cache stores ET under "
                    "the TE name, which is a different kernel, so it must "
                    "be recomputed with precompute_acc_kernels."
                )
        raise OSError(f"Covariance coupling not found at {coupling_path}")

    return coupling_kernels


def _max_rel_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Elementwise max relative difference between two arrays of equal shape."""
    denom = np.maximum(np.abs(a), np.abs(b))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(denom > 0, np.abs(a - b) / denom, 0.0)
    return float(np.max(rel)) if rel.size else 0.0


def convert_text_cache(
    save_dir: str, *, remove_text: bool = False, verbose: bool = True
) -> int:
    """
    Convert every legacy ``*.txt`` ACC coupling kernel under
    ``<save_dir>/covariance_coupling/`` to ``.npy``.

    Each conversion is verified bit-exactly against ``np.loadtxt`` of the
    ``.txt`` source (``np.savetxt``'s default ``fmt='%.18e'`` round-trips a
    float64 exactly); a mismatch raises, reporting the max relative
    difference.

    Idempotent: a ``.npy`` file that already exists and matches its ``.txt``
    source is left alone and not recounted. Manifests (``manifest_*.json``)
    are untouched -- only kernel files are converted.

    Parameters
    ----------
    save_dir : str
        The cache directory (the ``save_dir`` passed to ``Cov`` /
        :func:`~cmbcov.approximations.acc.precompute_acc_kernels`).
    remove_text : bool
        Delete the ``.txt`` files after their ``.npy`` counterpart has been
        written and verified (or was already there and verified to match).
        Never deletes a ``.txt`` file whose conversion could not be
        verified -- conversion failures raise before this runs.
    verbose : bool
        Print progress, one line per file plus a summary.

    Returns
    -------
    int
        The number of ``.txt`` files newly converted to ``.npy`` (files
        already converted are skipped and not counted).
    """
    coupling_dir = os.path.join(save_dir, "covariance_coupling")
    if not os.path.isdir(coupling_dir):
        if verbose:
            print(
                f"No coupling cache directory at {coupling_dir!r}; nothing to convert."
            )
        return 0

    txt_paths = sorted(glob.glob(os.path.join(coupling_dir, "*.txt")))
    converted = 0
    verified_txt_paths = []

    for txt_path in txt_paths:
        npy_path = os.path.splitext(txt_path)[0] + ".npy"
        text_array = np.loadtxt(txt_path)

        if os.path.exists(npy_path):
            existing = np.load(npy_path)
            if existing.shape == text_array.shape and np.array_equal(
                existing, text_array
            ):
                verified_txt_paths.append(txt_path)
                if verbose:
                    print(f"  Already converted: {npy_path}")
                continue
            raise ValueError(
                f"{npy_path} already exists and does not match {txt_path} "
                f"(max relative difference {_max_rel_diff(existing, text_array):.3e}); "
                "refusing to overwrite. Remove or investigate the mismatched "
                ".npy by hand."
            )

        np.save(npy_path, text_array)
        round_trip = np.load(npy_path)
        if not np.array_equal(round_trip, text_array):
            raise ValueError(
                f"Round trip for {txt_path} -> {npy_path} was not bit-exact "
                f"(max relative difference {_max_rel_diff(round_trip, text_array):.3e})."
            )

        converted += 1
        verified_txt_paths.append(txt_path)
        if verbose:
            print(f"  Converted {txt_path} -> {npy_path}")

    if remove_text:
        for txt_path in verified_txt_paths:
            os.remove(txt_path)
            if verbose:
                print(f"  Removed {txt_path}")

    if verbose:
        print(f"Converted {converted} file(s) in {coupling_dir!r}.")
    return converted
