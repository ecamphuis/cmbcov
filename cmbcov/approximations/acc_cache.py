"""
On-disk cache for ACC coupling kernels.

This module owns ``save_dir``, filenames, ``np.load``/``np.save`` and
the manifest JSON for the coupling kernels computed by
:mod:`cmbcov.approximations.acc`.  It knows nothing about
Wigner symbols, spherical-harmonic integrals or their contractions -- callers
pass in the set of valid single-spectrum labels (e.g.
``acc.COUPLING_SPECTRA``) purely as strings to validate against, not as
physics.
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
    "read_coupling_manifest",
    "write_coupling_kernels",
    "write_generation",
    "load_coupling_kernels",
    "convert_text_cache",
]

#: Version of the ``grid="healpix"`` coupling-kernel physics recorded in the
#: manifest as ``healpix_kernel_version`` (the GL path is exact and records
#: nothing, like ``map2alm_iter``). Bumped whenever a change alters the
#: numbers a HEALPix-grid precompute writes, so an old cache is refused
#: rather than silently reused with the wrong physics:
#:
#: 1. Theta built from ``Re X(L1, L2)`` only -- biased for a mask without
#:    azimuthal symmetry. Manifests at this version have no
#:    ``healpix_kernel_version`` field at all (the field did not exist yet).
#: 2. Theta keeps ``X(L1)``, ``X(L2)`` complex through the contraction
#:    (Eq. 20), fixing the bias above.
HEALPIX_KERNEL_VERSION = 2

#: Manifest fields with no persistent record on ``Cov`` (they are choices
#: made at precompute call time -- ``precompute_acc_kernels(nside=,
#: lw=, grid=)`` -- not part of ``CovarianceConfig``).  They are validated
#: for *session* consistency instead of against an independently known
#: "current" value: the first manifest read (or kernel write) for a given
#: ``save_dir`` in this process establishes the reference, and every later
#: manifest for that ``save_dir`` is compared against it.  See
#: :func:`_validate_cache_identity`.
_REFERENCE_FIELDS = ("nside", "lw", "grid")

#: ``{abspath(save_dir): {"nside":, "lw":, "grid":}}`` -- the session
#: reference described above.  Module-level (like ``exact._MASK_WL_CACHE``)
#: because the whole point is that it survives across the many
#: :func:`load_coupling_kernels` calls one covariance computation makes
#: (one per diagonal offset per block), not just within one call.
_CACHE_REFERENCE: dict[str, dict] = {}

#: ``{abspath(save_dir): n}`` -- how many :func:`write_coupling_kernels` calls
#: this process has made into each ``save_dir``.  An in-memory kernel memo
#: (``Cov.get_acc_coupling_kernels``) records the count at load time and
#: reloads when it has moved, so a write by any caller in this process --
#: including one with no reference to that memo, such as
#: :func:`~cmbcov.approximations.acc.precompute_acc_kernels`
#: -- can never be shadowed by a stale memo entry.
_WRITE_GENERATION: dict[str, int] = {}

#: ``save_dir`` (abspath) already warned about an unverifiable cache identity
#: in this process. The loader runs inside a loop over every block and
#: offset, so without this the same warning would fire on every call.
_WARNED_SAVE_DIRS: set = set()


def git_hash() -> str | None:
    """
    The current commit hash, for the coupling-cache manifest.  ``None`` (not
    an exception) if this is not a git checkout or ``git`` is unavailable --
    the manifest is a diagnostic aid, not something the precompute should
    fail over.
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
    this module only checks membership in it, it attaches no meaning to the
    labels themselves.
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
    # The letters are NOT sorted: TE (T at l, E at l') and ET (E at l, T
    # at l') are different kernels and live in different files.
    filename = f"covariance_coupling/{stokes_1}x{stokes_2}_{ell}x{ell_prime}.npy"
    return os.path.join(save_dir, filename)


def _legacy_text_path(npy_path: str) -> str:
    """The pre-conversion ``.txt`` path for a ``.npy`` kernel path."""
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
    check (the old labels are not necessarily in the current label set).
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
    kernel. The list is not deduplicated against the caller's own checks --
    a caller that has already tried a candidate may see it again (e.g. a
    diagonal pair, whose transpose is itself) and should simply skip a path
    it already ruled out.

    ``legacy_names`` is the same canonical-label -> old-on-disk-label map
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


def read_coupling_manifest(save_dir: str, ell: int, ell_prime: int) -> dict | None:
    """
    Read the manifest :func:`write_coupling_kernels` writes next to the
    kernels of one ``(ell, ell_prime)`` pair, or ``None`` if there is none (a
    cache written before the manifest existed, or the pair was never
    computed).

    The manifest records ``spectra`` (the requested subset of the caller's
    spectra set that pair was computed for), ``grid``, ``lw``, ``nside``,
    ``centralell`` and ``git_hash`` -- enough for :func:`load_coupling_kernels`
    to tell "this cache was built for a smaller spectrum set" apart from
    "this file is missing because something went wrong".
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

    ``term_selection`` is the a-priori term-selection tolerance the kernels
    were built with (:mod:`cmbcov.term_selection`), or
    ``None`` for the full contraction; it is **always** written, as
    ``null`` when ``None``, and a manifest predating the field is read as
    ``None`` (:func:`_validate_cache_identity`).  ``term_selection_stats``
    (the fractions of orders, pairs and ``M`` band kept at this ``(ell,
    ellp)``) is recorded when given.

    ``map2alm_iter`` is the Jacobi refinement count of the HEALPix
    ``map2alm`` that built the spin-weighted integrals (``grid="healpix"``
    only; the GL path is exact and records nothing).  Like ``mask_digest``
    it is left out of the manifest when ``None``.

    ``healpix_kernel_version`` is :data:`HEALPIX_KERNEL_VERSION` at the time
    of the write (``grid="healpix"`` only, ``None`` for GL -- same rule as
    ``map2alm_iter``), so a cache built by the pre-Eq.-20 HEALPix kernel
    (version 1, biased for a mask without azimuthal symmetry) is
    distinguishable from one built after the fix.

    ``mask_digest`` is a content digest of the mask map the precompute
    consumed (see :attr:`~cmbcov.mask.MaskWlm.mask_digest`),
    recorded in the manifest so :func:`load_coupling_kernels` can tell a
    cache built for a different mask from one built for this one.  Omitted
    (``None``, the default) it is left out of the manifest entirely rather
    than written as ``null`` -- distinguishing "not recorded" (an
    old-format cache, or a caller that has not opted into mask-identity
    checking) from "recorded and empty", which never legitimately happens.

    This also updates this ``save_dir``'s session reference for ``nside``,
    ``lw`` and ``grid`` (see :func:`_validate_cache_identity`), so a cache
    read back right after being written in this process always validates
    cleanly against it.
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

    manifest = {
        "ell": ell,
        "ellp": ellp,
        "spectra": list(spectra),
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
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    if verbose:
        print(f"  Saved manifest to {manifest_path}")

    _CACHE_REFERENCE[os.path.abspath(save_dir)] = {
        field: manifest.get(field) for field in _REFERENCE_FIELDS
    }


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
    in that cache that cannot be verified -- the same condition holds for
    every pair a run loads, and was once reported pair by pair.
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
        "the precompute-acc command) "
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
    ``healpix_kernel_version`` means 1 -- see :data:`HEALPIX_KERNEL_VERSION`)
    and ``term_selection`` (the a-priori term-selection tolerance the run
    expects its kernels to have been built with, ``None`` for the full
    kernels; checked whenever the key is present in ``current``, a manifest
    without the field meaning ``None`` -- so a run asking for full kernels
    never loads selected ones and vice versa),
    all known independently of the cache from ``Cov`` / its config / the
    package. ``nside``, ``lw`` and ``grid`` have no such independent
    "current" value -- they are precompute-call choices, not part of
    ``CovarianceConfig`` -- so they are checked for *session* consistency
    instead: the first manifest seen for this ``save_dir`` in this process
    (or a kernel write, see :func:`write_coupling_kernels`) establishes the
    reference, and every later manifest for the same ``save_dir`` must
    match it.

    A manifest missing entirely, or missing the ``mask_digest`` field (an
    older cache, written before mask identity was recorded), is
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
        cached_ts = manifest.get("term_selection")  # absent in old manifests: None
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

    # A HEALPix-grid cache with no recorded map2alm_iter predates that field
    # and defaults to iter=0 (a plain adjoint, ~2.5% low on the Eq. 22 sum);
    # it is known to differ, so it is refused, not warned on.
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

    # A HEALPix-grid cache with no recorded healpix_kernel_version predates
    # that field, i.e. version 1 (Theta built from Re X only, biased for a
    # mask without azimuthal symmetry); it is known to differ from the
    # current version, so it is refused, not warned on. A cache recorded
    # with a *higher* version than this package knows about was built by a
    # newer package and is refused too, rather than assuming it is
    # compatible.
    if (
        manifest.get("grid") == "healpix"
        and current.get("healpix_kernel_version") is not None
    ):
        cached_version = manifest.get("healpix_kernel_version", 1)
        current_version = current["healpix_kernel_version"]
        if cached_version != current_version:
            if cached_version < current_version:
                detail = (
                    "This cache predates the complex-Theta fix and must be "
                    "recomputed."
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
            f"ACC coupling cache at {save_dir!r} predates mask-identity "
            f"recording (manifest at {manifest_path!r} has no "
            f"{unverifiable_fields}); this field cannot be checked "
            "against this run's mask. Recomputing the precompute "
            "(precompute_acc_kernels) adds verification.",
        )


def _spectra_covered(
    stokes_key: tuple[str, str],
    manifest_spectra: Sequence[str],
    legacy_names: dict[str, str] | None,
) -> bool:
    """
    Whether both labels of ``stokes_key`` are covered by ``manifest_spectra``
    -- directly, or (when ``legacy_names`` is given) under their old on-disk
    name, so a manifest recorded under the old channel names still covers a
    request made with the new ones.
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
        Canonical label -> an older on-disk label it used to be called (e.g.
        ``{"DD": "EE", "LL": "BB", "TD": "TE", "DT": "ET"}``).  Purely a
        string-to-string fallback, still no meaning attached: when the file
        for a requested pair's canonical labels is absent, the file for the
        pair with every label passed through this map is tried next (before
        the ``.txt``/"not found" fallbacks below), and loaded in its place if
        present -- so a cache written under old labels loads without
        recompute and gives bit-identical results.  ``None`` (default) skips
        this fallback entirely (unchanged behaviour).
    current : dict, optional
        This run's ``mask_digest`` and ``centralell`` (see
        :func:`_validate_cache_identity`).  ``None`` (the default) skips
        identity validation entirely -- unchanged behaviour for a caller
        that only wants the plain load (e.g. a direct ``acc_cache`` call
        with no ``Cov`` to draw "current" values from).  A caller that
        passes it gets, on every load (not only when a file is missing):
        a mismatched field raises (naming the field, the cached and the
        run's value, and ``save_dir``); a field that cannot be checked at
        all (no manifest, or a manifest predating mask-identity recording)
        warns once per ``save_dir`` per process instead.

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
        If the ``.npy`` kernel is missing but the legacy ``.txt`` file (the
        on-disk format before kernels were switched to ``.npy``) is present.
        The message names the ``.txt`` file and the ``convert_acc_cache``
        command that converts it; the cache is never silently written to or
        read from at load time.
    OSError
        If a kernel file is missing outright.  The message names the missing
        pair. If a manifest exists for ``(ell, ell_prime)`` (written by
        :func:`write_coupling_kernels`) and it shows the cache was built for
        a spectrum set not covering this pair, the message says so and that
        the cache must be recomputed -- there is never a silent
        substitution. Otherwise, an ET file that is missing while its TE
        counterpart exists means the cache predates ET kernels being
        computed separately from TE; such kernels must also be recomputed,
        with no fallback.
    """
    if pairs is None:
        pairs = default_pairs
    else:
        pairs = list(pairs)

    coupling_kernels = {}
    manifest: dict | bool | None = None  # False once looked up and absent

    if current is not None:
        manifest = read_coupling_manifest(save_dir, ell, ell_prime)
        if manifest is None:
            manifest = False
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

        if os.path.exists(coupling_path):
            coupling_kernels[stokes_key] = np.load(coupling_path)
            continue

        # Own name, legacy name, transposed own name, transposed legacy
        # name (docs/theory/bmode_kernels.md): a cache that only
        # stores the other orientation of this pair -- e.g. a B-mode
        # precompute's minimal pair set -- still serves it, exactly
        # (transposing a real array is exact; no recompute needed).
        found = False
        for candidate_path, transposed in coupling_kernel_candidates(
            save_dir, stokes_key, ell, ell_prime, known_spectra, legacy_names
        ):
            if candidate_path == coupling_path:
                continue  # already tried above
            if os.path.exists(candidate_path):
                array = np.load(candidate_path)
                coupling_kernels[stokes_key] = (
                    np.ascontiguousarray(array.T) if transposed else array
                )
                found = True
                break
        if found:
            continue

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

        if manifest is None:
            manifest = read_coupling_manifest(save_dir, ell, ell_prime)
            if manifest is None:
                manifest = False

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

        # The reversed-order (central D/T at l', primed at l) channel, in
        # both the old naming ("ET") and the new one ("DT" -- the canonical
        # name of old "ET", see CHANNEL_ALIASES in acc.py); a cache missing
        # it but holding the direct-order ("TE"/"TD") file predates the fix
        # that computes them separately.
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
                    f"but {aliased_path} exists: these kernels predate "
                    "the ET fix (ET is a different kernel from TE) and "
                    "must be recomputed with precompute_acc_kernels."
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
    float64 exactly, so ``np.array_equal`` must hold -- a mismatch raises,
    reporting the max relative difference, rather than silently keeping a
    kernel that does not match its text source).

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
