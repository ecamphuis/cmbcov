"""
Mask handling and coupling-kernel orchestration.

`MaskWlm` owns everything derived from the survey footprint: the map itself,
its harmonic coefficients, the mask power spectra W_l and W^2_l that drive the
coupling operators of Camphuis et al. (2022) Eq. (3), degraded copies of the
mask for the ACC precomputation, and the on-disk cache of the four coupling
kernel families (M, Msq, K, G).

Kernels are computed natively by `cmbcov.kernels`. A
legacy Fortran backend (``master_kernels``/``cor2cl``) is kept internally as
an independent cross-check the package no longer runs; it is not part of
this repository. A cached M/Msq FITS file written by that
Fortran route is still read correctly: `_load_kernel_fits` detects and
corrects its orientation from the array itself, not from which tool wrote it.
A Fortran-written G or K is not reused (see "Kernel cache keying" below).
"""

import hashlib
import json
import os
import re
import warnings
from typing import Any

import ducc0
import numpy as np
from astropy.io import fits

from .grid import gl_analysis, gl_synthesis
from .sht import alm2cl, almxfl, cplx_spin_weighted_ylm, ducc0_alm2map, ducc0_map2alm
from .utils.healpy_utils import get_nside_from_ell
from .utils.threading_utils import get_optimal_nthreads

__all__ = ["MaskWlm", "mask_spectral_moment"]

# Default settings of the iterative least-squares alm solve. Module-level so
# that MaskWlm.compute_spherical_harmonics's signature and the "ensure the alm
# exists" helper below cannot drift apart: the helper has to reason about what
# the defaults are in order to avoid downgrading a stricter solve.
DEFAULT_ALM_MAXITER = 20
DEFAULT_ALM_EPSILON = 1e-10

#: Bump this whenever a change to this module, `kernels.coupling` or
#: `kernels.polspice` alters the *values* a MASTER/PolSpice kernel cache
#: (`M_*.fits`, `Msq_*.fits`, `G_*.fits`, `K_*.fits`) is written with -- never
#: the git hash, which would invalidate every on-disk cache on every commit
#: regardless of whether it touched kernel computation at all. Keying the
#: cache on what produced it this way: a stale cache written under an older
#: version of this constant is silently recomputed rather than reused, even
#: though the mask itself did not change.
#:
#: Version 2: ``get_polspice_master_kernel`` returns the decoupled Eq. (56)
#: layout ``(G0, Gplus, Gminus, Gx)`` instead of the legacy
#: ``(G0, Gp2, Gm2, Gx)``. A cached file from version 1 holds a different
#: operator in channels 1 and 2, so it must not be reused.
KERNEL_CACHE_VERSION = 2


def mask_spectral_moment(wl: np.ndarray, lmax: int | None = None) -> float:
    r"""
    Power-weighted mean of :math:`L(L+1)` for a mask power spectrum.

    .. math::

        \langle L(L+1) \rangle_{W} = \frac{\sum_L (2L+1)\, L(L+1)\, W_L}
                                           {\sum_L (2L+1)\, W_L}

    A pure function of a spectrum array: it takes no ``MaskWlm`` and performs
    no transform, so it can be evaluated on any ``W_l`` or ``W^2_l`` already
    in hand.

    Passed the mask's own spectrum ``W_l`` this is a measure of the mask's
    angular scale (large for a mask with sharp features or small apodisation,
    small for a smooth, weakly-apodised one). Passed the **squared**-mask
    spectrum ``W^2_l`` (e.g. ``MaskWlm.compute_power_spectra()[1]``), this is
    the :math:`\langle L(L+1) \rangle_{W^2}` that sets the size of the ACC
    polarisation leakage floor: the fractional bias of a
    two-E-leg covariance block at multipole :math:`\ell` is measured as

    .. math::

        \lambda(\ell) \approx 0.5\ \text{to}\ 0.65 \times
        \frac{\langle L(L+1) \rangle_{W^2}}{\ell^2} ,

    good to about 20% across a factor 18 in mask width. Evaluating this
    function on ``W^2_l`` therefore lets a caller predict the leakage bias at
    their multipole of interest before running anything.

    Parameters
    ----------
    wl : np.ndarray
        Power spectrum, ``wl[L]`` for ``L = 0 .. wl.size - 1``.
    lmax : int, optional
        Largest ``L`` to include in the sums (default: ``wl.size - 1``, i.e.
        the whole array). Must not exceed ``wl.size - 1``.

    Returns
    -------
    float
        The power-weighted mean of ``L(L+1)`` over ``L = 0 .. lmax``.

    Raises
    ------
    ValueError
        If ``lmax`` is negative, exceeds the size of ``wl``, or the spectrum
        is identically zero over the requested range (the mean is undefined).
    """
    wl = np.asarray(wl, dtype=np.float64)
    if wl.ndim != 1:
        raise ValueError(f"wl must be 1-D, got shape {wl.shape}")

    if lmax is None:
        lmax = wl.size - 1
    if lmax < 0:
        raise ValueError(f"lmax must be non-negative, got {lmax}")
    if lmax > wl.size - 1:
        raise ValueError(
            f"lmax={lmax} exceeds the spectrum's own range (wl has size {wl.size})"
        )

    ell = np.arange(lmax + 1)
    weight = (2.0 * ell + 1.0) * wl[: lmax + 1]
    denom = float(np.sum(weight))
    if denom == 0.0:
        raise ValueError("wl is identically zero over the requested range")

    numer = float(np.sum(weight * ell * (ell + 1.0)))
    return numer / denom


def _load_kernel_fits(path: str) -> np.ndarray:
    r"""
    Read a coupling-kernel FITS file in the package orientation ``[c, l, l']``.

    Every kernel is held in memory as :math:`\mathcal{K}_{\ell\ell'} =
    (2\ell'+1)\,\Xi_{\ell\ell'}`, acting as :math:`C_\ell = \sum_{\ell'}
    \mathcal{K}_{\ell\ell'} C_{\ell'}` (Camphuis et al. 2022, Eqs. 5, 43, 51).
    Files written by this package with astropy already are. Files written by
    the Fortran ``master_kernels`` are not: it fills ``kernels(l1, l2, c)``
    with the :math:`(2\ell_2+1)` factor on ``l2`` and writes it column-major,
    so astropy reads it back as ``[c, l2, l1]``, the transpose.

    Both kinds share one file name (``M_<mask>.fits``), so the orientation is
    read from the data instead of from where the file came from: channel 0 is
    :math:`(2\ell'+1)\Xi^{00}` with :math:`\Xi^{00}` exactly symmetric, so the
    ``(2l+1)`` factor can only be removed cleanly along the axis that carries
    it. A diagonal kernel is the same either way and is returned unchanged.
    """
    kernel = np.asarray(fits.getdata(path))
    channel0 = kernel[0] if kernel.ndim == 3 else kernel
    two_l_plus_1 = 2.0 * np.arange(channel0.shape[-1]) + 1.0

    def asymmetry(xi: np.ndarray) -> float:
        scale = np.abs(xi).max()
        return np.abs(xi - xi.T).max() / scale if scale > 0 else 0.0

    factor_on_last = asymmetry(channel0 / two_l_plus_1[None, :])
    factor_on_first = asymmetry(channel0 / two_l_plus_1[:, None])
    if factor_on_first < factor_on_last:
        kernel = np.swapaxes(kernel, -1, -2)
    return np.ascontiguousarray(kernel)


# ============================================================================
# Kernel cache keying
#
# The MASTER (M/Msq), PolSpice-Master (G) and PolSpice (K) kernel getters
# below cache their FITS files under MaskWlm.save_dir (``<mask_path>/
# utils_<mask>/``). Each file has a sidecar JSON manifest recording the
# inputs that determine its values, and a file is reused only when its
# manifest matches this run (keying the cache on what produced it, after
# an incident where an unrelated code change silently invalidated every
# cache).
#
# File names. A kernel is written under a parameter-keyed name,
# ``<legacy stem>.k<digest>.fits``, e.g. ``Msq_border_apod_mask_512.
# k1f2e3d4c5b6a.fits``, where the digest is a 12-hex-digit blake2b of the
# canonical JSON of the whole manifest (`keyed_kernel_path`). The legacy stems
# are ``M_<mask>``, ``Msq_<mask>``, ``G_<mask>_typ<t>_sig<s>_the<th>`` and
# ``K_typ<t>_sig<s>_the<th>``. A lookup tries, in order: the keyed name of
# this exact request; any other keyed file of the same stem whose manifest
# matches (a wider band limit, sliced -- the smallest sufficient one first);
# then the legacy name ``<legacy stem>.fits`` if its manifest matches, for
# caches written before keyed names existed.
#
# Never overwrite. A file with no manifest, or whose manifest does not match,
# is left byte-for-byte untouched -- legacy name or keyed name alike -- and the
# fresh kernel goes to this request's keyed name, where the next run finds it.
# Nothing here deletes or rewrites a file it did not write under that exact
# key.
#
# Backend. M/Msq manifests may carry ``backend="fortran"`` from the legacy
# master_kernels route; that field is not checked for them (the Wigner-3j
# MASTER kernel of the same W_L is the same operator). G and K are reused
# only from a native manifest: cor2cl hands master_kernels the Legendre
# transform of g (or f_apo) truncated at L <= lmax where the 3j sum needs
# L <= l + l', which on a small patch is a 23% max element error of G0 at
# lmax 24 and 0.18% at lmax 256. A Fortran G/K is
# therefore recomputed natively, never reused.
# ============================================================================

#: File-name marker of a parameter-keyed kernel cache, see `keyed_kernel_path`.
_KEYED_DIGEST_HEX = 12


def _kernel_manifest_path(fits_path: str) -> str:
    """Sidecar manifest path for a kernel FITS file."""
    return fits_path + ".manifest.json"


def _read_kernel_manifest(fits_path: str) -> dict | None:
    """
    Read the sidecar manifest next to ``fits_path``, or ``None`` if there is
    none -- either a cache written before manifests existed or a manifest that
    failed to parse. Both are treated identically ("unverifiable, do not
    reuse") rather than raised: a corrupt manifest must never block a
    recompute of a cache that is cheap to rebuild.
    """
    path = _kernel_manifest_path(fits_path)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write_kernel_manifest(fits_path: str, manifest: dict) -> None:
    """
    Write the sidecar manifest for ``fits_path``, atomically (temporary file
    then ``os.replace``).

    Callers must call this only *after* the FITS file itself has been
    written successfully (`_write_kernel_cache` does), so a crash or
    exception mid-write of the kernel file never leaves a valid-looking
    manifest beside a partial or missing kernel.
    """
    path = _kernel_manifest_path(fits_path)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _kernel_cache_mismatch(
    manifest: dict | None, expected: dict, monotonic_fields: tuple[str, ...] = ()
) -> str | None:
    """
    Compare a loaded manifest against this run's ``expected`` key values.

    Returns the name of the first field that differs, ``"manifest"`` if
    ``manifest`` is ``None`` (missing or unparseable), or ``None`` if the
    cache is valid to reuse as-is.

    Fields named in ``monotonic_fields`` (the band limits ``l1max``/``l2max``
    a kernel was computed at) are checked as "cached >= requested" rather
    than equality: a kernel computed at a larger band limit and sliced down
    is the same kernel up to rounding for M/Msq (exact quadrature, verified
    in tests/test_kernel_cache_keying.py) and up to the spectrally convergent
    quadrature of the analytic factors f_apo and 1/w for G/K, so a wider
    cache must not be rejected just because it was not computed at exactly
    this band limit.
    """
    if manifest is None:
        return "manifest"
    for field, value in expected.items():
        if field in monotonic_fields:
            if manifest.get(field, -1) < value:
                return field
        elif manifest.get(field) != value:
            return field
    return None


def keyed_kernel_path(legacy_path: str, manifest: dict) -> str:
    """
    The parameter-keyed file name a kernel with this ``manifest`` is written
    under, next to its legacy name: ``<legacy stem>.k<digest>.fits``.

    The digest is the first 12 hex digits of blake2b over the manifest as
    canonical JSON (sorted keys, no whitespace), so it is deterministic
    across processes and changes whenever any recorded input does.
    """
    stem = (
        legacy_path[: -len(".fits")] if legacy_path.endswith(".fits") else legacy_path
    )
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    digest = hashlib.blake2b(
        payload.encode(), digest_size=_KEYED_DIGEST_HEX // 2
    ).hexdigest()
    return f"{stem}.k{digest}.fits"


def _keyed_kernel_siblings(legacy_path: str) -> list:
    """Every keyed kernel file of ``legacy_path``'s stem in its directory."""
    directory, name = os.path.split(legacy_path)
    stem = name[: -len(".fits")] if name.endswith(".fits") else name
    pattern = re.compile(re.escape(stem) + rf"\.k[0-9a-f]{{{_KEYED_DIGEST_HEX}}}\.fits")
    try:
        names = os.listdir(directory or ".")
    except OSError:
        return []
    return sorted(os.path.join(directory, n) for n in names if pattern.fullmatch(n))


#: Kernel cache file paths (absolute) already warned about in this process.
#: Mirrors approximations/acc_cache.py's `_WARNED_SAVE_DIRS`: these getters
#: run inside loops over many (ell_max, ...) requests, so without this the
#: same warning would fire on every call rather than once per file.
_WARNED_KERNEL_PATHS: set = set()


def _warn_kernel_cache_mismatch_once(path: str, field: str, written_to: str) -> None:
    """Emit one ``UserWarning`` per ``path`` per process naming why it was not reused."""
    abs_path = os.path.abspath(path)
    if abs_path in _WARNED_KERNEL_PATHS:
        return
    _WARNED_KERNEL_PATHS.add(abs_path)
    if field == "manifest":
        reason = "has no manifest (predates cache keying, or a prior manifest was unreadable)"
    else:
        reason = f"manifest field {field!r} does not match this run"
    warnings.warn(
        f"Kernel cache at {path!r} {reason}; it is left untouched, and the "
        f"kernel is recomputed and written to {os.path.basename(written_to)!r}.",
        UserWarning,
        stacklevel=4,
    )


def _read_cached_kernel(path: str, manifest: dict) -> np.ndarray:
    """
    Read a cached kernel as contiguous native-endian float64.

    A native manifest means the file was written by this package in its own
    orientation, so it is read as is. Any other (a legacy Fortran-written
    M/Msq) goes through `_load_kernel_fits`, which corrects the orientation
    from the data. The byte-order conversion matters beyond tidiness: FITS
    is big-endian, and numpy does not hand ``>f8`` operands to BLAS, so a
    kernel left big-endian would be multiplied by a slow generic loop and
    round differently from a freshly computed one. The values are exact
    either way.
    """
    if manifest.get("backend") == "native":
        kernel = fits.getdata(path)
    else:
        kernel = _load_kernel_fits(path)
    return np.ascontiguousarray(kernel, dtype=np.float64)


def _find_kernel_cache(
    legacy_path: str,
    manifest: dict,
    ell_max: int,
    monotonic_fields: tuple[str, ...] = ("l1max", "l2max"),
    keyed_fields: tuple[str, ...] | None = None,
) -> np.ndarray | None:
    """
    The cached kernel for this request, sliced to ``ell_max``, or ``None``.

    ``manifest`` is the manifest a fresh computation of this request would be
    written with; ``keyed_fields`` (default: all of its fields) are the ones
    a cached manifest must match, ``monotonic_fields`` as "cached >=
    requested". Lookup order and the never-overwrite rule: see "Kernel cache
    keying" above. A legacy-name file that is present but not reusable is
    warned about once per process.
    """
    expected = (
        dict(manifest)
        if keyed_fields is None
        else {k: manifest[k] for k in keyed_fields}
    )
    exact = keyed_kernel_path(legacy_path, manifest)

    matching = []
    for path in [exact] + [
        p for p in _keyed_kernel_siblings(legacy_path) if p != exact
    ]:
        if not os.path.exists(path):
            continue
        cached = _read_kernel_manifest(path)
        if _kernel_cache_mismatch(cached, expected, monotonic_fields) is None:
            rank = 0 if path == exact else 1
            matching.append(
                (rank, tuple(cached.get(f, 0) for f in monotonic_fields), path, cached)
            )
    legacy_manifest = None
    if os.path.exists(legacy_path):
        legacy_manifest = _read_kernel_manifest(legacy_path)
        mismatch = _kernel_cache_mismatch(legacy_manifest, expected, monotonic_fields)
        if mismatch is None:
            matching.append((2, (), legacy_path, legacy_manifest))
        elif not matching:
            _warn_kernel_cache_mismatch_once(legacy_path, mismatch, exact)

    for _, _, path, cached in sorted(matching, key=lambda item: item[:3]):
        try:
            kernel = _read_cached_kernel(path, cached)
        except OSError as e:
            warnings.warn(f"Failed to load kernel cache {path!r}: {e}", stacklevel=3)
            continue
        if (
            kernel.ndim == 3
            and kernel.shape[1] >= ell_max
            and kernel.shape[2] >= ell_max
        ):
            return kernel[:, :ell_max, :ell_max]
        # The manifest matched but the array itself is smaller -- an
        # inconsistent cache (e.g. hand-edited) rather than a keying miss.
        warnings.warn(
            f"Kernel cache {path!r} is smaller than its manifest says "
            f"({kernel.shape} for ell_max {ell_max}); not reused",
            stacklevel=3,
        )
    return None


def _write_kernel_cache(
    legacy_path: str, array: np.ndarray, manifest: dict, overwrite: bool = True
) -> str | None:
    """
    Write ``array`` and its manifest under the keyed name of ``manifest``
    (`keyed_kernel_path`), never over a file whose manifest differs.

    Returns the path written (or already present), or ``None`` when an
    unrelated file occupies the keyed name -- a keyed file with no manifest
    (an interrupted write) or with a different one (hand-edited); it is left
    untouched, with a warning, and the caller keeps its in-memory kernel.
    With ``overwrite=False`` an existing matching file is kept as is. Both
    files are written to a temporary name and moved into place, so a
    concurrent run never reads a partial kernel.
    """
    path = keyed_kernel_path(legacy_path, manifest)
    if os.path.exists(path) or os.path.exists(_kernel_manifest_path(path)):
        existing = _read_kernel_manifest(path)
        if existing != manifest or not os.path.exists(path):
            abs_path = os.path.abspath(path)
            if abs_path not in _WARNED_KERNEL_PATHS:
                _WARNED_KERNEL_PATHS.add(abs_path)
                warnings.warn(
                    f"Not writing kernel cache {path!r}: a file with that name "
                    "but without a matching manifest is already there and is "
                    "left untouched (remove it to let the cache be written).",
                    UserWarning,
                    stacklevel=3,
                )
            return None
        if not overwrite:
            return path
    tmp = f"{path}.tmp{os.getpid()}"
    fits.PrimaryHDU(array).writeto(tmp, overwrite=True)
    os.replace(tmp, path)
    # Only after the FITS file is in place, so a failed write can never leave
    # a valid-looking manifest beside a missing or partial kernel.
    _write_kernel_manifest(path, manifest)
    return path


class MaskWlm:
    """
    Spherical mask analysis class for CMB covariance matrix computation.

    This class handles spherical masks for CMB analysis, computing their
    harmonic coefficients, power spectra, coupling matrices, and associated
    kernels for analytical covariance matrix estimation.

    The class provides methods for:
    - Mask manipulation and degradation
    - Spherical harmonic analysis
    - Power spectrum computation
    - Coupling matrix computation (native MASTER/PolSpice kernels)
    - PolSpice and Master method kernels

    Attributes
    ----------
    mask_name : str
        Name of the mask file (without .fits extension)
    load_path : str
        Directory path where mask file is located
    mask : np.ndarray
        HEALPix mask map data
    nside : int
        HEALPix resolution parameter (nside)
    mask_alm : np.ndarray, optional
        Spherical harmonic coefficients of the mask
    mask_power_spectrum : np.ndarray, optional
        Angular power spectrum of the mask
    mask_squared_power_spectrum : np.ndarray, optional
        Angular power spectrum of the squared mask

    Raises
    ------
    FileNotFoundError
        If mask file cannot be found
    ValueError
        If mask data is invalid or nside is not valid
    """

    def __init__(
        self,
        mask_name: str,
        load_path: str | None = None,
        precompute_alm: bool = False,
    ):
        """
        Initialize mask analysis object.

        Parameters
        ----------
        mask_name : str
            Name of mask file (with or without .fits extension)
        load_path : str, optional
            Path to directory containing mask file (default: current directory)
        precompute_alm : bool, optional
            Whether to precompute spherical harmonic coefficients (default: False)

        Raises
        ------
        FileNotFoundError
            If mask file doesn't exist
        ValueError
            If mask data is invalid
        """
        # Handle paths and filenames
        self.load_path = load_path if load_path is not None else "./"
        self.mask_name = mask_name.replace(".fits", "")

        # Load and validate mask
        self._load_mask()

        # Initialize computed quantities
        self.mask_alm: np.ndarray | None = None
        self._mask_alm_lmax: int = -1
        self._mask_alm_maxiter: int | None = None
        self._mask_alm_epsilon: float | None = None
        # Convergence info from the iterative least-squares alm solve
        # (ducc0.sht.pseudo_analysis), populated by compute_spherical_harmonics.
        self._mask_alm_residual: float | None = None
        self._mask_alm_iterations: int | None = None
        self._mask_alm_stopreason: int | None = None
        self._mask_alm_quality: float | None = None
        self.mask_power_spectrum: np.ndarray | None = None
        self.mask_squared_power_spectrum: np.ndarray | None = None
        self._mask_digest: str | None = None

        # Precompute if requested
        if precompute_alm:
            self.compute_spherical_harmonics()

    @property
    def mask_digest(self) -> str:
        """
        Content digest of the mask map, for identifying which mask an
        on-disk cache (e.g. the ACC coupling kernels, see
        :mod:`cmbcov.approximations.acc_cache`) was built
        from.

        Same algorithm (blake2b, 16-byte digest, hex-encoded) as
        :func:`cmbcov.exact._mask_wl_healpix`'s ``W_L``
        memo key, so the package has one way of fingerprinting a mask
        rather than two. Computed once and cached: ``self.mask`` is set only
        by :meth:`_load_mask`, during ``__init__``, and never reassigned
        afterwards.
        """
        if self._mask_digest is None:
            mask = np.ascontiguousarray(self.mask, dtype=np.float64)
            self._mask_digest = hashlib.blake2b(mask, digest_size=16).hexdigest()
        return self._mask_digest

    def _load_mask(self) -> None:
        """Load and validate mask from file."""
        import healpy as hp  # lazy, see utils/healpy_utils.py

        mask_path = os.path.join(self.load_path, self.mask_name + ".fits")

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        try:
            self.mask = hp.read_map(mask_path)
            # Ensure mask has proper dtype for ducc0 compatibility
            self.mask = np.asarray(self.mask, dtype=np.float64)
        except Exception as e:
            raise ValueError(f"Failed to read mask file {mask_path}: {e}") from e

        # Clean bad pixels
        if np.any(hp.mask_bad(self.mask)):
            warnings.warn(
                f"Found bad pixels in mask {self.mask_name}, setting to 0.0",
                stacklevel=3,
            )
            self.mask[hp.mask_bad(self.mask)] = 0.0

        self.nside = hp.get_nside(self.mask)

    @property
    def mask_file_path(self) -> str:
        """Get full path to mask file."""
        return os.path.join(self.load_path, self.mask_name + ".fits")

    @property
    def save_dir(self) -> str:
        """Get the default save directory for this mask."""
        dir_name = f"utils_{self.mask_name}"
        save_path = os.path.join(self.load_path, dir_name)
        os.makedirs(save_path, exist_ok=True)
        return save_path

    def __str__(self) -> str:
        """String representation of the mask object."""
        return f"MaskWlm(name='{self.mask_name}', nside={self.nside}, path='{self.mask_file_path}')"

    def __repr__(self) -> str:
        """Detailed string representation."""
        alm_status = "computed" if self.mask_alm is not None else "not computed"
        cl_status = (
            "computed" if self.mask_power_spectrum is not None else "not computed"
        )
        return (
            f"MaskWlm(mask_name='{self.mask_name}', nside={self.nside}, "
            f"alm_status='{alm_status}', cl_status='{cl_status}')"
        )

    # ==================== Spherical Harmonic Methods ====================

    def compute_spherical_harmonics(
        self,
        lmax: int | None = None,
        nthreads: int | None = None,
        maxiter: int = DEFAULT_ALM_MAXITER,
        epsilon: float = DEFAULT_ALM_EPSILON,
    ) -> np.ndarray:
        """
        Compute spherical harmonic coefficients of the mask.

        HEALPix has no exact quadrature, so this is the one inherently
        approximate step of the branch's otherwise-exact GL pipeline (see
        :mod:`cmbcov.grid`). A single adjoint synthesis
        (``ducc0_map2alm``, i.e. ``map2alm`` at ``iter=0``) carries ~5e-3
        relative alm error and biases the MASTER kernel by ~0.2% at coarse
        nside. Instead this method solves for the alm as a converged
        least-squares fit of the band-limited synthesis to the pixel mask,
        via ``ducc0.sht.pseudo_analysis`` (iterative LSMR) on the HEALPix
        RING geometry -- the same geometry ``ducc0_map2alm`` uses, obtained
        from ``ducc0.healpix.Healpix_Base(nside, "RING").sht_info()``.

        Parameters
        ----------
        lmax : int, optional
            Maximum multipole (default: 3 * nside)
        nthreads : int, optional
            Number of threads to use for computation (default: auto-detect)
        maxiter : int, optional
            Maximum LSMR iterations (default: 20)
        epsilon : float, optional
            Relative residual tolerance used as the solver's stopping
            criterion (default: 1e-10)

        Returns
        -------
        np.ndarray
            Spherical harmonic coefficients

        Raises
        ------
        ValueError
            If lmax is invalid
        """
        if lmax is None:
            # 2*nside, not the historical 3*nside: the least-squares solve
            # converges in ~8 iterations at 2*nside but hits maxiter at 3*nside
            # at every resolution tested (nside 32 and 128 alike). Above
            # ~2*nside a HEALPix map no longer determines its alm, so those
            # coefficients are fitted noise, not a band-limited mask.
            lmax = 2 * self.nside

        if lmax <= 0:
            raise ValueError(f"lmax must be positive, got {lmax}")

        # The cache must be keyed on lmax. The previous guard returned early
        # whenever mask_alm existed, without ever looking at lmax, so the first
        # caller silently fixed the band limit for the lifetime of the object:
        # degrade_mask(16) followed by compute_power_spectra() returned a mask
        # spectrum truncated to 3*16 instead of 3*nside, with no warning. The
        # solver parameters are folded into the same guard: a request with
        # different maxiter/epsilon also triggers a recompute.
        if (
            self.mask_alm is not None
            and self._mask_alm_lmax >= lmax
            and self._mask_alm_maxiter == maxiter
            and self._mask_alm_epsilon == epsilon
        ):
            return self.mask_alm

        geom = ducc0.healpix.Healpix_Base(self.nside, "RING").sht_info()
        pixel_map = np.ascontiguousarray(self.mask, dtype=np.float64)[np.newaxis, :]

        alm, stopreason, iterations, residual, quality = ducc0.sht.pseudo_analysis(
            map=pixel_map,
            theta=geom["theta"],
            nphi=geom["nphi"],
            phi0=geom["phi0"],
            ringstart=geom["ringstart"],
            lmax=lmax,
            mmax=lmax,
            spin=0,
            nthreads=nthreads or get_optimal_nthreads(self.nside),
            maxiter=maxiter,
            epsilon=epsilon,
        )

        self.mask_alm = alm[0]
        self._mask_alm_lmax = lmax
        self._mask_alm_maxiter = maxiter
        self._mask_alm_epsilon = epsilon
        # W_l and W^2_l are functions of the alm just replaced, so any cached
        # copies now describe a different band-limited mask. Drop them rather
        # than let compute_power_spectra return spectra from the previous solve.
        self.mask_power_spectrum = None
        self.mask_squared_power_spectrum = None
        # Convergence diagnostics: stopreason 1/2 means the tolerance was met,
        # 7 means maxiter was reached first (common here -- a hard-edged or
        # steeply apodised mask's misfit floor is set by HEALPix's lack of an
        # exact quadrature, not by the iteration count, so the residual can
        # plateau above epsilon while the least-squares solution itself has
        # already converged; `quality` tracks that inner convergence).
        self._mask_alm_stopreason = int(stopreason)
        self._mask_alm_iterations = int(iterations)
        self._mask_alm_residual = float(residual)
        self._mask_alm_quality = float(quality)
        if self._mask_alm_iterations >= maxiter:
            warnings.warn(
                f"Mask alm solve did not converge in {maxiter} iterations at "
                f"lmax={lmax} (residual {self._mask_alm_residual:.2e}). "
                "Coefficients above ~2*nside are not determined by a HEALPix map; "
                "lower lmax or raise maxiter.",
                RuntimeWarning,
                stacklevel=2,
            )

        return self.mask_alm

    def _pending_alm_solve_params(
        self, min_lmax: int | None = None
    ) -> tuple[int, int, float]:
        """
        The ``(lmax, maxiter, epsilon)`` that :meth:`_ensure_mask_alm` would
        solve at for ``min_lmax``, without performing the solve.

        Used to validate a kernel cache's manifest (see "Kernel cache
        keying" above) against what this run's mask alm solve *would* be,
        on the fast path where the on-disk cache turns out to match and the
        solve should be skipped entirely -- forcing it just to check would
        defeat the point of caching the kernel in the first place. Must stay
        in lockstep with :meth:`_ensure_mask_alm`'s own arithmetic below.
        """
        if min_lmax is None:
            min_lmax = 2 * self.nside

        if self.mask_alm is None:
            return min_lmax, DEFAULT_ALM_MAXITER, DEFAULT_ALM_EPSILON

        cached_maxiter = (
            DEFAULT_ALM_MAXITER
            if self._mask_alm_maxiter is None
            else self._mask_alm_maxiter
        )
        cached_epsilon = (
            DEFAULT_ALM_EPSILON
            if self._mask_alm_epsilon is None
            else self._mask_alm_epsilon
        )
        return (
            max(min_lmax, self._mask_alm_lmax),
            max(DEFAULT_ALM_MAXITER, cached_maxiter),
            min(DEFAULT_ALM_EPSILON, cached_epsilon),
        )

    def _ensure_mask_alm(self, min_lmax: int | None = None) -> np.ndarray:
        """
        Make sure ``self.mask_alm`` exists, without weakening an existing solve.

        Callers that only need *some* alm (compute_power_spectra, the kernel
        entry points) must not call ``compute_spherical_harmonics()`` bare:
        that requests the defaults in every dimension at once, and the cache
        guard compares ``maxiter``/``epsilon`` by equality, so a caller who
        had just run ``compute_spherical_harmonics(lmax=200, maxiter=60)``
        would fail the guard and have its solve silently replaced by a fresh
        one at ``lmax=2*nside, maxiter=20`` -- a downgrade with no warning,
        since the narrower solve converges happily.

        The reverse mistake is just as easy: simply skipping the call when
        ``mask_alm`` is not None would let ``degrade_mask(nside//2)``, which
        deliberately solves at a *low* band limit, leak that truncation into
        W_l and W^2_l (the bug ``tests/test_mask_alm_cache.py`` guards).

        So ask for the element-wise strictest of the requirement and whatever
        is cached: the widest band limit, the largest iteration budget, the
        tightest tolerance. That can only ever upgrade the stored alm, never
        reduce it, and it reproduces the plain-default request exactly when
        nothing has been computed yet.

        Parameters
        ----------
        min_lmax : int, optional
            Band limit the caller needs at minimum (default: ``2 * nside``,
            i.e. ``compute_spherical_harmonics``'s own default).
        """
        lmax, maxiter, epsilon = self._pending_alm_solve_params(min_lmax)
        return self.compute_spherical_harmonics(
            lmax=lmax, maxiter=maxiter, epsilon=epsilon
        )

    def degrade_mask(self, target_nside: int) -> np.ndarray:
        """
        Create a degraded version of the mask at lower resolution.

        Uses spherical harmonic filtering to properly degrade the mask
        while preserving its statistical properties.

        Parameters
        ----------
        target_nside : int
            Target HEALPix resolution parameter

        Returns
        -------
        np.ndarray, int
            Degraded mask at target_nside resolution, and the nside used

        Raises
        ------
        ValueError
            If target_nside is invalid or larger than current nside
        """
        import healpy as hp  # lazy, see utils/healpy_utils.py

        if not hp.isnsideok(target_nside):
            raise ValueError(f"Invalid target nside: {target_nside}")

        if target_nside >= self.nside:
            warnings.warn(
                f"Target nside ({target_nside}) >= current nside ({self.nside}), "
                "returning original mask",
                stacklevel=2,
            )
            return self.mask.copy(), self.nside

        lmax = 3 * target_nside
        # Only a *lower bound*: the low-pass below already handles a cached alm
        # that is wider than lmax, so there is no reason to discard one (and a
        # bare compute_spherical_harmonics(lmax=...) would discard any solve
        # made with non-default maxiter/epsilon -- see _ensure_mask_alm).
        self._ensure_mask_alm(min_lmax=lmax)

        # Create low-pass filter
        current_lmax = hp.Alm.getlmax(self.mask_alm.size)
        filter_spectrum = np.zeros(current_lmax + 1)
        filter_spectrum[: min(lmax + 1, current_lmax + 1)] = 1.0

        try:
            filtered_alm = almxfl(self.mask_alm, filter_spectrum)
            degraded_mask = ducc0_alm2map(filtered_alm, nside=target_nside)
            degraded_mask = np.clip(degraded_mask, 0.0, 1.0)
        except Exception as e:
            raise ValueError(f"Failed to degrade mask: {e}") from e

        return degraded_mask, target_nside

    # ==================== Power Spectrum Methods ====================

    def compute_power_spectra(
        self,
        save: bool = False,
        output_dir: str | None = None,
        nthreads: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute power spectra of the mask and squared mask.

        Under this branch's philosophy the mask IS its band-limited alm, so
        ``W^2_l`` is computed as the exact power spectrum of the *band-limited*
        mask squared, not (as ``hp.anafast`` would give) of the pixel mask
        squared. Concretely: synthesise ``W`` from ``self.mask_alm`` (band-limit
        ``Lw``) onto a Gauss-Legendre grid, square it pointwise, and analyse
        the result exactly up to ``2 Lw``.

        Grid size derivation: the integrand of the analysis step is
        ``W^2 * Y*_LM``, of degree ``2 Lw`` (from ``W^2``) ``+ L`` with
        ``L <= 2 Lw``, i.e. degree ``<= 4 Lw``. A GL grid of band-limit ``Lg``
        integrates polynomials in cos(theta) exactly up to degree
        ``2 Lg + 1`` (see :mod:`cmbcov.grid`), so exactness
        needs ``2 Lg + 1 >= 4 Lw``, i.e. ``Lg >= 2 Lw - 1/2`` and therefore
        ``Lg = ceil((4 Lw - 1) / 2) = 2 Lw`` for integer ``Lw``; this also
        satisfies ``gl_analysis``'s requirement ``Lg >= lmax`` for the
        ``lmax = 2 Lw`` analysis. :func:`cmbcov.grid.gl_minimal_lmax`
        does not fit this case directly (it is derived for ``W * Y_lmax *
        Y*_lmax``, not ``W^2 * Y_L``), so ``Lg`` is computed directly here.

        Cost: the grid has ``(Lg+1) * (2 Lg+2) = (2 Lw+1)(4 Lw+2)`` points; at
        ``nside=1024`` (``Lw = 3*nside = 3072``) that is ~75e6 points, ~600 MB
        of float64 -- fine as a one-off but worth knowing before calling this
        at high resolution.

        Parameters
        ----------
        save : bool, optional
            Whether to save power spectra to files (default: False)
        output_dir : str, optional
            Directory to save files (default: self.load_path)
        nthreads : int, optional
            Number of threads for the GL transforms (default: auto-detect)

        Returns
        -------
        tuple
            (mask_power_spectrum, mask_squared_power_spectrum)

        Raises
        ------
        ValueError
            If computation fails
        """
        # Ensure spherical harmonics are computed
        self._ensure_mask_alm()

        # Compute mask power spectrum if needed
        if self.mask_power_spectrum is None:
            try:
                self.mask_power_spectrum = alm2cl(self.mask_alm)
            except Exception as e:
                raise ValueError(f"Failed to compute mask power spectrum: {e}") from e

        # Compute squared mask power spectrum if needed
        if self.mask_squared_power_spectrum is None:
            try:
                nthreads = nthreads or get_optimal_nthreads(self.nside)
                lw = self._mask_alm_lmax
                lg = 2 * lw

                w_grid = gl_synthesis(
                    self.mask_alm, lmax=lw, lmax_grid=lg, spin=0, nthreads=nthreads
                )
                w_squared_grid = w_grid * w_grid
                w_squared_alm = gl_analysis(
                    w_squared_grid,
                    lmax=2 * lw,
                    lmax_grid=lg,
                    spin=0,
                    nthreads=nthreads,
                )
                self.mask_squared_power_spectrum = alm2cl(w_squared_alm)
            except Exception as e:
                raise ValueError(
                    f"Failed to compute squared mask power spectrum: {e}"
                ) from e

        # Save if requested
        if save:
            save_dir = output_dir if output_dir is not None else self.save_dir
            try:
                self._save_power_spectra(save_dir)
            except Exception as e:
                warnings.warn(f"Failed to save power spectra: {e}", stacklevel=2)

        return self.mask_power_spectrum, self.mask_squared_power_spectrum

    def _save_power_spectra(self, output_dir: str | None = None) -> None:
        """Save power spectra to FITS files."""
        import healpy as hp  # lazy, see utils/healpy_utils.py

        if output_dir is None:
            output_dir = self.save_dir
        else:
            os.makedirs(output_dir, exist_ok=True)

        if self.mask_power_spectrum is not None:
            cl_file = os.path.join(output_dir, f"wl_{self.mask_name}.fits")
            hp.write_cl(cl_file, self.mask_power_spectrum, overwrite=True)

        if self.mask_squared_power_spectrum is not None:
            cl_sq_file = os.path.join(output_dir, f"wlsq_{self.mask_name}.fits")
            hp.write_cl(cl_sq_file, self.mask_squared_power_spectrum, overwrite=True)

    def compute_spin_weighted_integrals(
        self,
        ell: int,
        m: int,
        mask_for_ell: np.ndarray | None = None,
        target_nside: int | None = None,
    ) -> np.ndarray:
        """
        Compute spin-weighted spherical harmonic integrals I_ell_m.

        This method computes the integrals needed for analytical covariance
        matrix estimation using spin-weighted spherical harmonics.

        Parameters
        ----------
        ell : int
            Multipole degree
        m : int
            Multipole order
        mask_for_ell : np.ndarray, optional
            Pre-computed mask at appropriate resolution
        target_nside : int, optional
            Target resolution for computation

        Returns
        -------
        np.ndarray
            Complex array of shape ``(2, 3, nalm)``, ``nalm =
            hp.Alm.getsize(2 * target_nside - 1)``: axis 0 selects the alms
            of the real part (index 0) or the imaginary part (index 1) of
            the mask-weighted spin-weighted (T, Q, U) maps, axis 1 is the
            (T, Q, U) field, axis 2 the packed ``(l, m)`` alm index at
            ``lmax = 2 * target_nside - 1``, ``mmax = lmax`` (``pol=True``
            convention of :func:`~cmbcov.sht.ducc0_map2alm`).

        Raises
        ------
        ValueError
            If ell or m values are invalid, or if the spin-weighted harmonics
            could not be computed for the given ``ell``, ``m``.
        """
        import healpy as hp  # lazy, see utils/healpy_utils.py

        if ell < 0:
            raise ValueError(f"ell must be non-negative, got {ell}")
        if abs(m) > ell:
            raise ValueError(f"|m| must be <= ell, got m={m}, ell={ell}")

        # Determine resolution and mask
        if mask_for_ell is None:
            if target_nside is None:
                target_nside = get_nside_from_ell(ell)
            mask_for_ell, _ = self.degrade_mask(target_nside)
        else:
            target_nside = hp.get_nside(mask_for_ell)

        # Compute spin-weighted spherical harmonics
        spin_weighted_ylm = cplx_spin_weighted_ylm(ell, m, nside=target_nside)

        # Check if function returned error code
        if isinstance(spin_weighted_ylm, int) and spin_weighted_ylm == -1:
            raise ValueError(
                f"Invalid parameters for spin-weighted harmonics: ell={ell}, m={m}"
            )
        # Multiply mask by spin-weighted harmonics
        weighted_maps = mask_for_ell * spin_weighted_ylm

        lmax = 2 * target_nside
        integrals = np.empty((2, 3, hp.Alm.getsize(lmax - 1)), dtype=complex)
        integrals[0] = ducc0_map2alm(
            weighted_maps.real, lmax=lmax - 1, nside=target_nside, pol=True
        )
        integrals[1] = ducc0_map2alm(
            weighted_maps.imag, lmax=lmax - 1, nside=target_nside, pol=True
        )
        return integrals

    # ==================== Coupling Matrix Methods ====================

    # ---- kernel cache names and manifests (see "Kernel cache keying") ----

    def _master_kernel_paths(self, kernel_dir: str) -> tuple[str, str]:
        """Legacy file names of the MASTER kernels, ``(M, Msq)``."""
        return (
            os.path.join(kernel_dir, f"M_{self.mask_name}.fits"),
            os.path.join(kernel_dir, f"Msq_{self.mask_name}.fits"),
        )

    def _polspice_master_kernel_path(
        self,
        kernel_dir: str,
        apodization_type: int,
        apodization_sigma: float,
        theta_max: float,
    ) -> str:
        """Legacy file name of the PolSpice-Master kernel G."""
        return os.path.join(
            kernel_dir,
            f"G_{self.mask_name}_typ{apodization_type}_sig{apodization_sigma}"
            f"_the{theta_max}.fits",
        )

    @staticmethod
    def _polspice_kernel_path(
        kernel_dir: str,
        apodization_type: int,
        apodization_sigma: float,
        theta_max: float,
    ) -> str:
        """Legacy file name of the full-sky PolSpice kernel K (no mask in it)."""
        return os.path.join(
            kernel_dir,
            f"K_typ{apodization_type}_sig{apodization_sigma}_the{theta_max}.fits",
        )

    def _mask_kernel_manifest(
        self, l1max: int, l2max: int, alm_params: tuple[int, int, float]
    ) -> dict:
        """
        Manifest of a kernel built from the band-limited mask power spectrum
        W_L (M/Msq, and G on top of the apodisation fields): the mask, the
        alm solve that produced W_L, the band limit and the cache format.
        """
        alm_lmax, alm_maxiter, alm_epsilon = alm_params
        return {
            "cache_version": KERNEL_CACHE_VERSION,
            "backend": "native",
            "mask_digest": self.mask_digest,
            "l1max": int(l1max),
            "l2max": int(l2max),
            "mask_alm_lmax": int(alm_lmax),
            "mask_alm_maxiter": int(alm_maxiter),
            "mask_alm_epsilon": float(alm_epsilon),
        }

    @staticmethod
    def _apodization_manifest(
        apodization_type: int, apodization_sigma: float, theta_max: float
    ) -> dict:
        """The PolSpice apodisation fields of a G/K manifest."""
        return {
            "apodization_type": int(apodization_type),
            "apodization_sigma": float(apodization_sigma),
            "theta_max": float(theta_max),
        }

    def _current_alm_solve_params(self) -> tuple[int, int, float]:
        """``(lmax, maxiter, epsilon)`` of the alm solve held right now."""
        return self._mask_alm_lmax, self._mask_alm_maxiter, self._mask_alm_epsilon

    def compute_master_coupling_kernels(
        self,
        l1max: int,
        l2max: int,
        output_dir: str | None = None,
        overwrite: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute master method coupling kernels natively.

        This method computes the coupling matrices needed for the Master method
        of power spectrum estimation, which corrects for the effects of incomplete
        sky coverage.

        Both kernels are written to ``output_dir`` under their parameter-keyed
        names (``M_<mask>.k<digest>.fits``, ``Msq_<mask>.k<digest>.fits``,
        see "Kernel cache keying" at the top of this module), each with its
        manifest. A file already there under another name -- including the
        legacy ``M_<mask>.fits`` -- is never modified.

        Parameters
        ----------
        l1max : int
            Maximum multipole for first dimension
        l2max : int
            Maximum multipole for second dimension
        output_dir : str
            Directory to save coupling kernel files
        overwrite : bool
            Whether to rewrite a keyed file that already exists with this
            exact manifest.

        Returns
        -------
        tuple
            (master_kernel, master_kernel_squared) - coupling kernels for
            mask and squared mask respectively, each of shape
            ``(4, lmax+1, lmax+1)`` indexed ``[channel, l1, l2]`` such that
            ``<C_l1> = sum_l2 K[c, l1, l2] C_l2``.

        Raises
        ------
        ValueError
            If input parameters are invalid
        """
        if l1max <= 0 or l2max <= 0:
            raise ValueError(f"l1max and l2max must be positive, got {l1max}, {l2max}")

        if output_dir is None:
            output_dir = self.save_dir

        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
            except Exception as e:
                raise ValueError(
                    f"Cannot create output directory {output_dir}: {e}"
                ) from e

        # Ensure spherical harmonics are computed
        self._ensure_mask_alm()

        # The MASTER kernel is a function of the band-limited mask power
        # spectrum W_L, i.e. of the mask, the alm solve that produced it and
        # the requested band limit -- all in the manifest. Read only after
        # _ensure_mask_alm() above so the solve parameters are those used.
        manifest = self._mask_kernel_manifest(
            l1max, l2max, self._current_alm_solve_params()
        )

        # ~1000x faster than the scalar Wigner-3j recursion the legacy
        # Fortran master_kernels executable used, and validated against exact
        # Wigner-3j symbols in tests/test_coupling_kernels.py.
        from .kernels.coupling import coupling_kernels

        wl, wlsq = self.compute_power_spectra(save=False)
        lmax = max(l1max, l2max)
        master_kernel = coupling_kernels(wl, lmax)
        master_kernel_squared = coupling_kernels(wlsq, lmax)

        for array, legacy_path in zip(
            (master_kernel, master_kernel_squared),
            self._master_kernel_paths(output_dir),
        ):
            _write_kernel_cache(legacy_path, array, manifest, overwrite=overwrite)

        return master_kernel, master_kernel_squared

    def compute_polspice_master_kernel(
        self,
        l1max: int,
        l2max: int,
        output_dir: str | None = None,
        apodization_type: int = 1,
        apodization_sigma: float = 30.0,
        theta_max: float = 30.0,
    ) -> np.ndarray:
        """
        Compute combined PolSpice + Master method coupling kernel (G matrix).

        This combines PolSpice apodization with Master method coupling matrix
        computation for improved power spectrum estimation.

        The kernel is written to ``output_dir`` under its parameter-keyed name
        (``G_<mask>_typ<t>_sig<s>_the<th>.k<digest>.fits``, see "Kernel cache
        keying" at the top of this module) with its manifest, so that
        :meth:`get_polspice_master_kernel` reuses it on later runs instead of
        redoing the mask alm solve, W_L and ``polspice_kernels`` on every run
        (0.46 s at lmax 700 on the survey mask, 16 s at lmax 3000 on an
        nside-2048 mask).

        Parameters
        ----------
        l1max : int
            Maximum multipole for first dimension
        l2max : int
            Maximum multipole for second dimension
        output_dir : str
            Directory to save G matrix file
        apodization_type : int, optional
            PolSpice apodization type (default: 1)
        apodization_sigma : float, optional
            PolSpice apodization parameter (default: 30.0)
        theta_max : float, optional
            PolSpice maximum angle parameter (default: 30.0)

        Returns
        -------
        np.ndarray
            Shape ``(4, lmax+1, lmax+1)`` in the covariance layout
            ``(G0, Gplus, Gminus, Gx)`` of
            :meth:`~cmbcov.kernels.polspice.PolSpiceKernels.as_covariance_array`.

        Raises
        ------
        ValueError
            If parameters are invalid
        """
        if l1max <= 0 or l2max <= 0:
            raise ValueError(f"l1max and l2max must be positive, got {l1max}, {l2max}")

        # Validate apodization parameters
        if apodization_type <= 0:
            raise ValueError(
                f"apodization_type must be positive, got {apodization_type}"
            )
        if apodization_sigma <= 0:
            raise ValueError(
                f"apodization_sigma must be positive, got {apodization_sigma}"
            )
        if theta_max <= 0:
            raise ValueError(f"theta_max must be positive, got {theta_max}")

        if output_dir is None:
            output_dir = self.save_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Ensure prerequisites are computed
        self._ensure_mask_alm()

        # This API takes DEGREES, matching the legacy cor2cl parameter file,
        # while polspice_kernels works in radians. Convert at the boundary.
        from .kernels.polspice import polspice_kernels

        wl, _ = self.compute_power_spectra(save=False)
        kernels = polspice_kernels(
            lmax=max(l1max, l2max),
            theta_max=np.radians(theta_max),
            wl=wl,
            apodize_sigma=np.radians(apodization_sigma),
            apodize_type=apodization_type,
        )
        # The covariance layout, not the legacy master_kernels one: the
        # latter has no slot for decG and puts the non-decoupled +2G in the
        # EE+BB channel (see PolSpiceKernels.as_covariance_array).
        kernel = kernels.as_covariance_array()

        # G depends on W_L (so on the mask and its alm solve), the band limit
        # and the apodisation.
        manifest = self._mask_kernel_manifest(
            l1max, l2max, self._current_alm_solve_params()
        )
        manifest.update(
            self._apodization_manifest(apodization_type, apodization_sigma, theta_max)
        )
        _write_kernel_cache(
            self._polspice_master_kernel_path(
                output_dir, apodization_type, apodization_sigma, theta_max
            ),
            kernel,
            manifest,
        )
        return kernel

    def compute_polspice_kernel(
        self,
        l1max: int,
        l2max: int,
        output_dir: str | None = None,
        apodization_type: int = 1,
        apodization_sigma: float = 30.0,
        theta_max: float = 30.0,
    ) -> np.ndarray:
        """
        Compute PolSpice coupling kernel (K matrix) for full-sky computation.

        This computes the coupling kernel for full-sky (no mask) PolSpice
        analysis, used as a reference or for simulations.

        The kernel is written to ``output_dir`` under its parameter-keyed name
        (``K_typ<t>_sig<s>_the<th>.k<digest>.fits``) with its manifest.

        Parameters
        ----------
        l1max : int
            Maximum multipole for first dimension
        l2max : int
            Maximum multipole for second dimension
        output_dir : str
            Directory to save K matrix file
        apodization_type : int, optional
            PolSpice apodization type (default: 1)
        apodization_sigma : float, optional
            PolSpice apodization parameter (default: 30.0)
        theta_max : float, optional
            PolSpice maximum angle parameter (default: 30.0)

        Returns
        -------
        np.ndarray
            PolSpice coupling kernel (K matrix)

        Raises
        ------
        ValueError
            If parameters are invalid
        """
        if l1max <= 0 or l2max <= 0:
            raise ValueError(f"l1max and l2max must be positive, got {l1max}, {l2max}")

        # Validate apodization parameters
        if apodization_type <= 0:
            raise ValueError(
                f"apodization_type must be positive, got {apodization_type}"
            )
        if apodization_sigma <= 0:
            raise ValueError(
                f"apodization_sigma must be positive, got {apodization_sigma}"
            )
        if theta_max <= 0:
            raise ValueError(f"theta_max must be positive, got {theta_max}")

        if output_dir is None:
            output_dir = self.save_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Full-sky K kernels: built from f_apo alone (no mask), so no wl.
        # This API takes DEGREES; polspice_kernels works in radians.
        from .kernels.polspice import polspice_kernels

        kernels = polspice_kernels(
            lmax=max(l1max, l2max),
            theta_max=np.radians(theta_max),
            wl=None,
            apodize_sigma=np.radians(apodization_sigma),
            apodize_type=apodization_type,
        )
        kernel = kernels.as_master_array("K")
        _write_kernel_cache(
            self._polspice_kernel_path(
                output_dir, apodization_type, apodization_sigma, theta_max
            ),
            kernel,
            self._polspice_kernel_manifest(
                l1max, l2max, apodization_type, apodization_sigma, theta_max
            ),
        )
        return kernel

    @classmethod
    def _polspice_kernel_manifest(
        cls,
        l1max: int,
        l2max: int,
        apodization_type: int,
        apodization_sigma: float,
        theta_max: float,
    ) -> dict:
        """
        Manifest of the full-sky K kernel: no mask digest and no alm solve --
        it is computed with an empty mask (:meth:`compute_polspice_kernel`),
        so it does not depend on which mask this object holds.
        """
        manifest = {
            "cache_version": KERNEL_CACHE_VERSION,
            "backend": "native",
            "l1max": int(l1max),
            "l2max": int(l2max),
        }
        manifest.update(
            cls._apodization_manifest(apodization_type, apodization_sigma, theta_max)
        )
        return manifest

    # ==================== High-Level Kernel Access Methods ====================

    def get_master_coupling_kernels(
        self,
        ell_max: int,
        kernel_dir: str | None = None,
        use_squared: bool = True,
        make_symmetric: bool = True,
    ) -> np.ndarray:
        """
        Get master method coupling kernels with automatic computation if needed.

        A cached file is reused only when its manifest matches this mask,
        this code version, this run's alm solve and a band limit at least
        this wide -- never by file name and size alone. Lookup order and the
        never-overwrite rule: "Kernel cache keying" at the top of this module.
        The manifest's ``backend`` is not checked for M/Msq: a legacy
        Fortran-written MASTER kernel is the same Wigner-3j operator and
        `_load_kernel_fits` corrects its orientation on load.

        Parameters
        ----------
        ell_max : int
            Maximum multipole for the kernel
        kernel_dir : str
            Directory containing or for saving kernel files
        use_squared : bool, optional
            Whether to use squared mask kernel (default: True)
        make_symmetric : bool, optional
            Whether to make kernel symmetric (default: True)

        Returns
        -------
        np.ndarray
            Master coupling kernel matrix

        Raises
        ------
        ValueError
            If kernel cannot be computed or loaded
        """
        if ell_max <= 0:
            raise ValueError(f"ell_max must be positive, got {ell_max}")

        if kernel_dir is None:
            kernel_dir = self.save_dir

        master_file, master_sq_file = self._master_kernel_paths(kernel_dir)
        target_file = master_sq_file if use_squared else master_file

        # _pending_alm_solve_params reports what this run's alm solve *would*
        # be without performing it, so a cache hit skips the solve entirely.
        manifest = self._mask_kernel_manifest(
            ell_max - 1, ell_max - 1, self._pending_alm_solve_params()
        )
        kernel = _find_kernel_cache(
            target_file,
            manifest,
            ell_max,
            keyed_fields=tuple(k for k in manifest if k != "backend"),
        )

        # Compute kernel if not available or wrong size
        if kernel is None:
            try:
                master_kernel, master_sq_kernel = self.compute_master_coupling_kernels(
                    ell_max - 1, ell_max - 1, kernel_dir
                )
                kernel = master_sq_kernel if use_squared else master_kernel
            except Exception as e:
                raise ValueError(
                    f"Failed to compute master coupling kernels: {e}"
                ) from e

        # Make symmetric if requested
        if make_symmetric:
            try:
                kernel = self._make_kernel_symmetric(kernel)
            except Exception as e:
                warnings.warn(f"Failed to make kernel symmetric: {e}", stacklevel=2)

        return kernel

    def get_polspice_kernel(
        self,
        ell_max: int,
        kernel_dir: str | None = None,
        apodization_type: int = 1,
        apodization_sigma: float = 30.0,
        theta_max: float = 30.0,
    ) -> np.ndarray:
        """
        Get PolSpice coupling kernel with automatic computation if needed.

        A cached file is reused only from a native manifest matching this
        code version, the apodisation and a band limit at least this wide
        (K has no mask in its key: it is the full-sky kernel). A
        Fortran-written K is not reused: cor2cl truncates the Legendre
        transform of f_apo at ``lmax``. Lookup order and the never-overwrite
        rule: "Kernel cache keying" at the top of this module.

        Parameters
        ----------
        ell_max : int
            Maximum multipole for the kernel
        kernel_dir : str
            Directory containing or for saving kernel files
        apodization_type : int, optional
            PolSpice apodization type (default: 1)
        apodization_sigma : float, optional
            PolSpice apodization parameter (default: 30.0)
        theta_max : float, optional
            PolSpice maximum angle parameter (default: 30.0)

        Returns
        -------
        np.ndarray
            PolSpice coupling kernel (K matrix)
        """
        if ell_max <= 0:
            raise ValueError(f"ell_max must be positive, got {ell_max}")

        if kernel_dir is None:
            kernel_dir = self.save_dir

        kernel = _find_kernel_cache(
            self._polspice_kernel_path(
                kernel_dir, apodization_type, apodization_sigma, theta_max
            ),
            self._polspice_kernel_manifest(
                ell_max - 1, ell_max - 1, apodization_type, apodization_sigma, theta_max
            ),
            ell_max,
        )

        # Compute if needed
        if kernel is None:
            try:
                kernel = self.compute_polspice_kernel(
                    ell_max - 1,
                    ell_max - 1,
                    kernel_dir,
                    apodization_type,
                    apodization_sigma,
                    theta_max,
                )
            except Exception as e:
                raise ValueError(f"Failed to compute PolSpice kernel: {e}") from e

        return kernel

    def get_polspice_master_kernel(
        self,
        ell_max: int,
        kernel_dir: str | None = None,
        apodization_type: int = 1,
        apodization_sigma: float = 30.0,
        theta_max: float = 30.0,
    ) -> np.ndarray:
        """
        Get combined PolSpice + Master coupling kernel with automatic computation if needed.

        A cached file is reused only from a native manifest matching this
        mask, code version, alm solve, apodisation and a band limit at least
        this wide. Lookup order and the never-overwrite rule: "Kernel cache
        keying" at the top of this module.

        A Fortran-written G (manifest ``backend`` other than ``"native"``) is
        *not* numerically equivalent and is never reused: cor2cl hands
        master_kernels the Legendre transform of g truncated at ``L <= lmax``
        while the Wigner-3j sum needs ``L <= l + l'`` -- a 23% max element
        error of G0 at lmax 24 and 0.18% at lmax 256 on a small patch.
        Such a file is left untouched and G is
        recomputed natively and written under its keyed name.

        Parameters
        ----------
        ell_max : int
            Maximum multipole for the kernel
        kernel_dir : str
            Directory containing or for saving kernel files
        apodization_type : int, optional
            PolSpice apodization type (default: 1)
        apodization_sigma : float, optional
            PolSpice apodization parameter (default: 30.0)
        theta_max : float, optional
            PolSpice maximum angle parameter (default: 30.0)

        Returns
        -------
        np.ndarray
            Shape ``(ell_max, ell_max)`` per channel, in the covariance
            layout ``(G0, Gplus, Gminus, Gx)`` -- the decoupled Eq. (56)
            form, not the legacy ``master_kernels`` channel order.
        """
        if ell_max <= 0:
            raise ValueError(f"ell_max must be positive, got {ell_max}")

        if kernel_dir is None:
            kernel_dir = self.save_dir

        manifest = self._mask_kernel_manifest(
            ell_max - 1, ell_max - 1, self._pending_alm_solve_params()
        )
        manifest.update(
            self._apodization_manifest(apodization_type, apodization_sigma, theta_max)
        )
        kernel = _find_kernel_cache(
            self._polspice_master_kernel_path(
                kernel_dir, apodization_type, apodization_sigma, theta_max
            ),
            manifest,
            ell_max,
        )

        # Compute if needed
        if kernel is None:
            try:
                kernel = self.compute_polspice_master_kernel(
                    ell_max - 1,
                    ell_max - 1,
                    kernel_dir,
                    apodization_type,
                    apodization_sigma,
                    theta_max,
                )
            except Exception as e:
                raise ValueError(
                    f"Failed to compute PolSpice-Master kernel: {e}"
                ) from e

        return kernel

    def _make_kernel_symmetric(self, kernel: np.ndarray) -> np.ndarray:
        """
        Make coupling kernel symmetric using proper normalization.

        Parameters
        ----------
        kernel : np.ndarray
            Input coupling kernel of shape (n_spec, n_ell, n_ell)

        Returns
        -------
        np.ndarray
            Symmetrized kernel
        """
        if kernel.ndim != 3:
            raise ValueError(f"Kernel must be 3D, got shape {kernel.shape}")

        symmetric_kernel = np.zeros_like(kernel)

        for spec_idx in range(kernel.shape[0]):
            # Recover the symmetric operator Xi from M_ll' = (2l'+1) Xi_ll'.
            # The (2l'+1) factor sits on the SECOND index, so it must be divided
            # out along axis 1. The previous form, (K.T / f).T, divides by the
            # FIRST index instead: it leaves (2l2+1)/(2l1+1) * Xi, which is not
            # symmetric, and propagates into an asymmetric, non-positive-definite
            # covariance matrix. It also only broadcasts at all because the
            # kernels happen to be square.
            ell_factors = 2.0 * np.arange(kernel.shape[2]) + 1
            normalized = kernel[spec_idx] / ell_factors[None, :]
            symmetric_kernel[spec_idx] = normalized

            # Verify symmetry for first spectrum (TT).
            # atol must be tied to the matrix scale: these entries are ~1e-5 and
            # far-off-diagonal ones are ~1e-12, so numpy's default atol of 1e-8
            # would declare a genuinely asymmetric kernel symmetric.
            scale = float(np.abs(normalized).max())
            if spec_idx == 0 and not np.allclose(
                normalized, normalized.T, rtol=1e-8, atol=1e-12 * scale
            ):
                # Symmetry of Xi is a mathematical identity, not a diagnostic:
                # an asymmetric Xi yields a non-positive-definite covariance.
                raise ValueError(
                    "Xi is not symmetric after normalisation "
                    f"(max asymmetry {np.abs(normalized - normalized.T).max() / scale:.3e}). "
                    "This indicates a kernel convention mismatch."
                )

        return symmetric_kernel

    # ==================== Utility Methods ====================

    def get_mask_statistics(self) -> dict[str, Any]:
        """
        Get comprehensive statistics about the mask.

        Returns
        -------
        dict
            Dictionary containing mask statistics
        """
        import healpy as hp  # lazy, see utils/healpy_utils.py

        good_pixels = hp.mask_good(self.mask)

        stats = {
            "mask_name": self.mask_name,
            "nside": self.nside,
            "npix_total": len(self.mask),
            "npix_good": np.sum(good_pixels),
            "npix_bad": np.sum(~good_pixels),
            "sky_fraction": (
                np.mean(self.mask[good_pixels]) if np.any(good_pixels) else 0.0
            ),
            "mask_min": np.min(self.mask[good_pixels]) if np.any(good_pixels) else 0.0,
            "mask_max": np.max(self.mask[good_pixels]) if np.any(good_pixels) else 0.0,
            "mask_mean": (
                np.mean(self.mask[good_pixels]) if np.any(good_pixels) else 0.0
            ),
            "mask_std": np.std(self.mask[good_pixels]) if np.any(good_pixels) else 0.0,
            "alm_computed": self.mask_alm is not None,
            "power_spectra_computed": self.mask_power_spectrum is not None,
        }

        if self.mask_alm is not None:
            stats["lmax"] = hp.Alm.getlmax(self.mask_alm.size)

        return stats

    def validate_mask(self) -> bool:
        """
        Validate mask data integrity.

        Returns
        -------
        bool
            True if mask passes validation checks

        Raises
        ------
        ValueError
            If mask fails validation with details
        """
        import healpy as hp  # lazy, see utils/healpy_utils.py

        issues = []

        # Check for NaN or infinite values
        if not np.all(np.isfinite(self.mask[hp.mask_good(self.mask)])):
            issues.append("Mask contains NaN or infinite values")

        # Check for negative values
        good_pixels = hp.mask_good(self.mask)
        if np.any(self.mask[good_pixels] < 0):
            issues.append("Mask contains negative values")

        # Check if mask is all zeros
        if np.all(self.mask[good_pixels] == 0):
            issues.append("Mask is all zeros")

        # Check sky fraction
        sky_fraction = np.mean(self.mask[good_pixels]) if np.any(good_pixels) else 0.0
        if sky_fraction < 0.01:
            issues.append(f"Very low sky fraction: {sky_fraction:.4f}")
        elif sky_fraction > 0.99:
            issues.append(
                f"Very high sky fraction: {sky_fraction:.4f} (nearly full sky)"
            )

        # Report issues
        if issues:
            raise ValueError(f"Mask validation failed: {'; '.join(issues)}")

        return True
