"""Recompute-side helpers of the ACC: windows, cache requirements and band assembly.

Everything :class:`~cmbcov.approximations.acc.strategy.ACCStrategy` needs to
turn cached kernels and a spectrum into a covariance block: the window padding
and internal ``lmax``, the check that a cache holds the pairs a covariance
needs, the padded spectra, the banded sums and the grouping of blocks.
"""

import hashlib
from collections.abc import Callable, Iterable

import numpy as np

from . import cache as acc_cache
from .channels import COUPLING_CHANNELS, COUPLING_SPECTRA, LEGACY_CHANNEL_NAMES

__all__ = [
    "acc_cached_kernel_size",
    "acc_internal_lmax",
    "acc_window_pad",
    "require_acc_cache_pairs",
]


#: Working-set budget, in bytes, of one ``ell1`` chunk of the batched ACC
#: assembly (:func:`_acc_band`).  A chunk of ``n`` multipoles holds three
#: ``(n, size)`` float64 arrays -- the two spectrum windows and the
#: window-times-kernel product -- so ``n = budget // (24 * size)``.  Fixed,
#: not a caller knob: ``compute_covariance_term`` has no memory argument, and
#: this is far below the default :data:`DEFAULT_COUPLING_MEMORY_GB` of the
#: precompute.
_ASSEMBLY_CHUNK_BYTES = 64 * 1024**2


def acc_window_pad(kernel_size: int, centralell: int) -> int:
    r"""
    How far past ``lmax`` the spectra must extend so that no reported ACC
    element loses kernel weight: ``max(0, kernel_size - 1 - centralell)``.

    Element ``(ell1, ell2)`` uses the kernel of ``(l*, l* + |ell2 - ell1|)``
    translated by ``shift = min(ell1, ell2) - l*`` (Eq. 33): kernel index
    ``i = 0 .. S-1`` stands for ``L = shift + i``. The largest reported
    ``min(ell1, ell2)`` is ``lmax - 1`` (on the diagonal), whose window ends
    at ``L = lmax - 1 - l* + S - 1``. The spectra must therefore hold
    ``lmax + S - 1 - l*`` multipoles, independent of ``dmax``. The low end
    (``L < 0`` for ``min(ell1, ell2) < l*``) is not a clip: there is no
    spectrum there.

    For ``S = 2 nside_acc = 512`` and ``l* = 250``, this is 261, not
    ``nside_acc = 256``: padding by 256 would still cut the last kernel
    columns from the top rows.
    """
    return max(0, int(kernel_size) - 1 - int(centralell))


def acc_internal_lmax(lmax: int, kernel_size: int, centralell: int) -> int:
    """
    ``lmax_int = lmax + acc_window_pad(kernel_size, centralell)``: the number
    of multipoles the spectra of an ACC computation must hold so that no
    element with ``ell1, ell2 < lmax`` has its coupling window cut (see
    :func:`acc_window_pad`). The covariance is still reported to ``lmax``.
    """
    return int(lmax) + acc_window_pad(kernel_size, centralell)


def acc_cached_kernel_size(
    kernel_dir: str,
    centralell: int,
    dmax: int,
    pairs: Iterable[tuple[str, str]] | None = None,
) -> int | None:
    """
    Largest side length of the ACC kernel files present under
    ``<kernel_dir>/covariance_coupling/`` for the pairs ``(centralell,
    centralell + d)``, ``d < dmax``, read from the ``.npy`` headers alone
    (no array is loaded). ``pairs`` restricts the Stokes pairs looked at
    (default: every ordered pair of :data:`COUPLING_SPECTRA`). A pair whose
    canonically named file is absent is also tried under its legacy letter
    name (:data:`LEGACY_CHANNEL_NAMES`) and under the transposed pair's own
    and legacy names (transposing does not change a square kernel's side
    length; docs/theory/bmode_kernels.md), so a cache written under letter
    names, or holding only the other orientation of a pair, reports the same
    size. ``None`` when no such file exists at all; the kernel load then
    reports what is missing.
    """
    if pairs is None:
        pairs = [(a, b) for a in COUPLING_SPECTRA for b in COUPLING_SPECTRA]
    pairs = list(pairs)
    size = None

    def header_side(path: str) -> int | None:
        try:
            with open(path, "rb") as handle:
                version = np.lib.format.read_magic(handle)
                if version == (1, 0):
                    header = np.lib.format.read_array_header_1_0(handle)
                else:
                    header = np.lib.format.read_array_header_2_0(handle)
                shape = header[0]
        except (OSError, ValueError):
            return None
        return max(shape) if shape else 0

    for diagonal_offset in range(int(dmax)):
        for pair in pairs:
            side = None
            for candidate_path, _transposed in acc_cache.coupling_kernel_candidates(
                kernel_dir,
                pair,
                centralell,
                centralell + diagonal_offset,
                COUPLING_CHANNELS,
                LEGACY_CHANNEL_NAMES,
            ):
                side = header_side(candidate_path)
                if side is not None:
                    break
            if side is None:
                continue
            size = side if size is None else max(size, side)
    return size


def _missing_kernel_pairs(
    kernel_dir: str,
    centralell: int,
    ell_prime: int,
    pairs: Iterable[tuple[str, str]],
) -> list[tuple[str, str]]:
    """
    The pairs of ``pairs`` that :func:`~cmbcov.approximations.acc.cache.load_coupling_kernels`
    cannot serve at ``(centralell, ell_prime)``: no file under the pair's own
    name, its legacy name or those of its transpose, or only files the
    manifest's pair record does not list
    (:func:`~cmbcov.approximations.acc.cache.locate_coupling_kernel`, the
    loader's own rule). File existence only; nothing is loaded.
    """
    manifest = acc_cache.read_coupling_manifest(kernel_dir, centralell, ell_prime)
    missing = []
    for pair in pairs:
        found, _ = acc_cache.locate_coupling_kernel(
            kernel_dir,
            tuple(pair),
            centralell,
            ell_prime,
            COUPLING_CHANNELS,
            LEGACY_CHANNEL_NAMES,
            manifest or False,
        )
        if found is None:
            missing.append(tuple(pair))
    return missing


#: Why a B-mode cache can lack pairs a run needs, and what to run: shared by
#: :func:`require_acc_cache_pairs` and :meth:`ACCStrategy._wick_plan`.
_BMODE_PAIRS_HELP = (
    "A run with a B observable needs, for every off-diagonal block, the "
    "kernel pairs of both of its orientations, Cov(a, b) and Cov(b, a), so "
    "that both of its triangles are exact at centralell "
    "(bmode_wick.required_kernel_pairs: 24, 25 or 45 pairs; "
    "docs/theory/bmode_kernels.md, Sect. 6). A cache may lack some of them: "
    "it may have been built with one orientation per block, or for "
    "C^TB = C^EB = 0 when this run's TB or EB spectra are non-zero. Rerun "
    "`cmbcov-precompute` on the parameter file, or add only the missing "
    "pairs to this cache with precompute_acc_kernels(<mask>, <kernel_dir>, "
    "centralell=..., dmax=..., pairs=[...]) and the nside, lw and grid it "
    "was built with: the kernels already there are kept, and the manifests "
    "merge (acc_cache.write_coupling_kernels)."
)


def require_acc_cache_pairs(
    kernel_dir: str,
    centralell: int,
    dmax: int,
    pairs: Iterable[tuple[str, str]],
) -> None:
    """
    Raise ``OSError`` if the on-disk cache does not cover every one of
    ``pairs``, for every diagonal ``(centralell, centralell + d)``,
    ``d < dmax``, naming every missing pair.

    Run early (:class:`~cmbcov.spectra.SpectraLoader`) against the pairs a
    B-mode ``observables`` list needs
    (:func:`~cmbcov.approximations.acc.bmode_wick.required_kernel_pairs`), so that a run is
    refused before anything else rather than failing on its first block. A
    pair counts as covered when the loader would serve it
    (:func:`_missing_kernel_pairs`): under its own or legacy name, or
    stored only as its transpose. The message lists the missing pairs (the
    union over the diagonals, and the diagonals affected), explains why a
    B-mode cache can lack pairs and what to run, and ends with the loader's
    own error for the first missing pair (which names its file and, where
    it applies, the manifest's spectra or a left-over file). A legacy
    text-format cache raises the loader's ``FileNotFoundError`` unchanged. A
    no-op for an empty ``pairs``.
    """
    pairs = sorted({tuple(pair) for pair in pairs})
    if not pairs:
        return
    missing: dict[tuple[str, str], list[int]] = {}
    for diagonal_offset in range(int(dmax)):
        ell_prime = centralell + diagonal_offset
        for pair in _missing_kernel_pairs(kernel_dir, centralell, ell_prime, pairs):
            missing.setdefault(pair, []).append(ell_prime)
    if not missing:
        return
    first_pair = min(missing, key=lambda p: (missing[p][0], p))
    try:
        acc_cache.load_coupling_kernels(
            kernel_dir,
            centralell,
            missing[first_pair][0],
            COUPLING_CHANNELS,
            [first_pair],
            pairs=[first_pair],
            legacy_names=LEGACY_CHANNEL_NAMES,
        )
        loader_error = None
    except FileNotFoundError:
        raise
    except OSError as err:
        loader_error = str(err)
    affected = sorted({lp for lps in missing.values() for lp in lps})
    where = (
        f"on every diagonal (centralell, centralell + d), d < {int(dmax)}"
        if len(affected) == int(dmax)
        else f"at (centralell, l') for l' in {affected}"
    )
    listed = ", ".join("x".join(pair) for pair in sorted(missing))
    raise OSError(
        f"ACC coupling kernels not found under {kernel_dir!r} (centralell "
        f"{centralell}): this run needs {len(pairs)} kernel pairs, and "
        f"{len(missing)} of them are missing {where}: {listed}. "
        f"{_BMODE_PAIRS_HELP}"
        + (f" First missing file: {loader_error}" if loader_error else "")
    )


def _unit_sum_kernel(coupling_matrix: np.ndarray) -> np.ndarray:
    """
    Eq. 23 normalisation as in :meth:`ACCStrategy.compute_acc_term`: the
    kernel divided by its sum, unless the sum is already 1 (``np.isclose``).
    """
    if not np.isclose(np.sum(coupling_matrix), 1.0):
        coupling_matrix = coupling_matrix / np.sum(coupling_matrix)
    return coupling_matrix


def _padded_spectrum(
    spectrum: np.ndarray, central_ell: int, n_read: int, size: int
) -> np.ndarray:
    """
    ``padded[central_ell + L] = spectrum[L]`` for ``0 <= L < n_read``, zero
    elsewhere, of length ``central_ell + n_read + size``.

    In :meth:`ACCStrategy.compute_acc_term` kernel index ``i`` stands for
    ``L = i + shift`` with ``shift = min(ell1, ell2) - central_ell``. In
    padded coordinates that is index ``min(ell1, ell2) + i``, so the whole
    window is ``padded[m : m + size]`` with ``m = min(ell1, ell2)``; the
    leading zeros stand in for the low-end runoff (``L < 0``). With
    ``n_read = lmax + acc_window_pad(size, central_ell)`` (:func:`acc_window_pad`)
    no window of an ``m < lmax`` reaches past ``spectrum[n_read - 1]``, so
    nothing is clipped at the high end.
    """
    padded = np.zeros(central_ell + n_read + size)
    padded[central_ell : central_ell + n_read] = spectrum[:n_read]
    return padded


def _acc_band(
    kernel: np.ndarray,
    padded_1: np.ndarray,
    padded_2: np.ndarray,
    n_ell: int,
    chunk: int,
) -> np.ndarray:
    """
    ``out[m] = padded_1[m:m+S] @ kernel @ padded_2[m:m+S]`` for
    ``m = 0 .. n_ell - 1``, i.e. :meth:`ACCStrategy.compute_acc_term` for every
    ``min(ell1, ell2) = m`` of one diagonal (it depends on ``(ell1, ell2)``
    only through ``m``, the kernel and the spectra), as one BLAS product per
    chunk of ``chunk`` multipoles instead of one Python call per element.

    Rows of the window matrices multiply the kernel's rows from the left and
    its columns from the right, preserving the ``cl1 @ K @ cl2`` index order
    of the scalar path for a non-symmetric kernel.
    """
    size = kernel.shape[0]
    if kernel.shape != (size, size):
        raise ValueError(f"ACC coupling kernel must be square, got {kernel.shape}")
    windows_1 = np.lib.stride_tricks.sliding_window_view(padded_1, size)
    same = padded_2 is padded_1
    windows_2 = (
        windows_1 if same else np.lib.stride_tricks.sliding_window_view(padded_2, size)
    )
    out = np.empty(n_ell)
    for start in range(0, n_ell, chunk):
        stop = min(n_ell, start + chunk)
        w1 = np.ascontiguousarray(windows_1[start:stop])
        w2 = w1 if same else np.ascontiguousarray(windows_2[start:stop])
        out[start:stop] = np.einsum("ij,ij->i", w1 @ kernel, w2)
    return out


def _spectrum_identity(spectrum: np.ndarray) -> tuple:
    """
    Content identity of a spectrum array: dtype, shape and a blake2b digest
    of its bytes. Two arrays with the same identity give the same padded
    spectrum (:func:`_padded_spectrum`) and so bit-identical kernel products;
    the batched assembly shares products between blocks by this identity, not
    by ``id()`` (a run's ``cl`` often holds equal arrays under different keys,
    e.g. ``TE`` and ``ET`` of one frequency pair).
    """
    array = np.ascontiguousarray(spectrum)
    digest = hashlib.blake2b(memoryview(array).cast("B"), digest_size=16)
    return (array.dtype.str, array.shape, digest.hexdigest())


def _identity_memo() -> Callable[[np.ndarray], tuple]:
    """
    :func:`_spectrum_identity`, memoised by ``id()`` for the duration of one
    assembly call: the arrays are referenced by the call's ``cl`` throughout,
    so an ``id`` cannot be reused within it. Never kept between calls, so a
    spectrum changed in place between two calls is hashed afresh.
    """
    identities: dict[int, tuple] = {}

    def identity(array: np.ndarray) -> tuple:
        key = id(array)
        if key not in identities:
            identities[key] = _spectrum_identity(array)
        return identities[key]

    return identity


def _acc_bands(
    products: dict,
    n_ell: int,
) -> dict:
    """
    :func:`_acc_band` for many products on one diagonal:
    ``products[key] = (kernel, padded_1, padded_2)`` gives
    ``bands[key] = _acc_band(kernel, padded_1, padded_2, n_ell, chunk)``,
    bit for bit, with the left factor ``w1 @ kernel`` of each chunk (the
    ``n_ell x S x S`` matrix product, nearly all of the cost) computed once
    per distinct ``(padded_1, kernel)`` pair of arrays and shared by every
    product with that left factor. Each product is otherwise evaluated
    exactly as :func:`_acc_band` does: same chunks (``chunk`` from
    :data:`_ASSEMBLY_CHUNK_BYTES` and the kernel size), same contiguous
    window copies, same ``np.einsum`` on the same operands.
    """
    groups: dict[tuple[int, int], list] = {}
    for key, (kernel, padded_1, _) in products.items():
        groups.setdefault((id(padded_1), id(kernel)), []).append(key)
    bands = {}
    for keys in groups.values():
        kernel, padded_1, _ = products[keys[0]]
        size = kernel.shape[0]
        if kernel.shape != (size, size):
            raise ValueError(f"ACC coupling kernel must be square, got {kernel.shape}")
        chunk = max(1, _ASSEMBLY_CHUNK_BYTES // (24 * size))
        windows_1 = np.lib.stride_tricks.sliding_window_view(padded_1, size)
        rights = []
        for key in keys:
            padded_2 = products[key][2]
            same = padded_2 is padded_1
            windows_2 = (
                windows_1
                if same
                else np.lib.stride_tricks.sliding_window_view(padded_2, size)
            )
            out = np.empty(n_ell)
            bands[key] = out
            rights.append((out, same, windows_2))
        for start in range(0, n_ell, chunk):
            stop = min(n_ell, start + chunk)
            w1 = np.ascontiguousarray(windows_1[start:stop])
            left = w1 @ kernel
            for out, same, windows_2 in rights:
                w2 = w1 if same else np.ascontiguousarray(windows_2[start:stop])
                out[start:stop] = np.einsum("ij,ij->i", left, w2)
    return bands


def _group_blocks(factors: list[set], batch_size: int) -> list[list[int]]:
    """
    Split blocks ``0 .. len(factors) - 1`` into batches of at most
    ``batch_size``, greedily, so that blocks sharing left factors
    (``factors[i]``, the ``(left spectrum, kernel)`` pairs of block ``i``'s
    kernel products) land in the same batch: each batch is grown by the
    block adding the fewest left factors not already in it, among those the
    one sharing the most (ties: the lowest index). Deterministic; every
    block appears exactly once, and a batch lists its blocks in increasing
    index. The grouping only decides which products are computed together,
    never a value.
    """
    n = len(factors)
    if n <= batch_size:
        return [list(range(n))] if n else []
    users: dict = {}
    for i, block_factors in enumerate(factors):
        for factor in block_factors:
            users.setdefault(factor, []).append(i)
    sizes = np.array([len(f) for f in factors], dtype=np.int64)
    weight = int(sizes.max()) + 1  # new factors first, then shared ones
    remaining = np.ones(n, dtype=bool)
    unused = np.iinfo(np.int64).max
    batches = []
    while remaining.any():
        new = sizes.copy()
        shared = np.zeros(n, dtype=np.int64)
        covered: set = set()
        batch = []
        while len(batch) < batch_size and remaining.any():
            score = np.where(remaining, new * weight - shared, unused)
            j = int(np.argmin(score))
            batch.append(j)
            remaining[j] = False
            for factor in factors[j]:
                if factor not in covered:
                    covered.add(factor)
                    for i in users[factor]:
                        new[i] -= 1
                        shared[i] += 1
        batches.append(sorted(batch))
    return batches


class _BlockPlan:
    """
    How one block of the batched ACC assembly is computed, fixed before any
    kernel is read (:meth:`ACCStrategy._plan_te_block`,
    :meth:`ACCStrategy._plan_wick_block`):

    - ``kind``: ``"te"`` (Eq. 23 per contraction, T/E-only run), ``"wick"``
      (per-Wick-term normalisation, run with a B observable) or ``"zero"``
      (a parity-mixed block the run does not ask for);
    - ``upper``, ``lower``: ``(product key, weight)`` per contraction or
      Wick term, in summation order; the
      weight is the ``norm_Xi`` matrix (``"te"``) or the
      :class:`~cmbcov.approximations.acc.bmode_wick.WickTerm` (``"wick"``). ``lower`` is
      ``None`` when the lower triangle is not computed from terms of its
      own (auto blocks; for ``"wick"`` also one-orientation blocks, see
      ``same_band_lower``);
    - ``products``: ``product key -> (spectrum_1, label_1, spectrum_2,
      label_2, kernel pair)``; a product key is ``(identity_1, identity_2,
      kernel pair)`` with the spectra's content identities
      (:func:`_spectrum_identity`), so equal products of different blocks
      share one key;
    - ``pairs``: the kernel pairs the block reads from the cache (for
      ``"te"`` also that of a contraction whose band is a reversed direct
      one).
    """

    __slots__ = ("cov_key", "kind", "auto", "upper", "lower", "products", "pairs")

    def __init__(
        self,
        cov_key,
        kind,
        auto=False,
        upper=(),
        lower=None,
        products=None,
        pairs=None,
    ):
        self.cov_key = cov_key
        self.kind = kind
        self.auto = auto
        self.upper = list(upper)
        self.lower = None if lower is None else list(lower)
        self.products = products or {}
        self.pairs = set(pairs or ())

    def left_factors(self) -> set:
        """The ``(left spectrum identity, kernel pair)`` of every product."""
        return {(key[0], key[2]) for key in self.products}


#: Version tag of the ACC normalisation used by a run with a B observable,
#: recorded in every raw-block manifest of such a run
#: (:meth:`ACCStrategy.raw_block_inputs`), so a block saved under another
#: rule -- Eq. 23 per Wick contraction, what a T/E-only run uses -- is never
#: reused for it. Change it whenever the per-term rule changes.
ACC_NORMALISATION_RULE = "per-wick-term-v1"


#: Spectra keys that are parity-odd (``C^TB``, ``C^EB`` in either order).
_PARITY_ODD_CL_KEYS = ("TB", "BT", "EB", "BE")


def _contraction_multiset(contractions) -> list:
    """
    The Wick contractions ``((spectrum_1, spectrum_2), kernel_key, norm)`` of
    a T/E block as a sorted list of identities, each taken up to the
    identity ``C1 . Theta^{pq} . C2 = C2 . Theta^{qp} . C1`` (the definition
    gives ``Theta^{qp} = (Theta^{pq})^T``): two blocks with the same list
    have the same ACC value on every element.
    """
    out = []
    for (spec_1, spec_2), kernel_key, norm in contractions:
        forward = (id(spec_1), id(spec_2), tuple(kernel_key), id(norm))
        reverse = (id(spec_2), id(spec_1), tuple(kernel_key[::-1]), id(norm))
        out.append(min(forward, reverse))
    return sorted(out)


def _parity_odd_nonzero(cl: dict[str, dict[str, np.ndarray]]) -> bool:
    """Whether any ``TB``/``BT``/``EB``/``BE`` spectrum in ``cl`` has a
    non-zero entry (at any frequency pair)."""
    for spectra in cl.values():
        for key in _PARITY_ODD_CL_KEYS:
            array = spectra.get(key) if hasattr(spectra, "get") else None
            if array is not None and np.any(np.asarray(array) != 0):
                return True
    return False


def _check_spectrum_length(
    spectrum: np.ndarray, name: str, lmax: int, kernel_size: int, centralell: int
) -> None:
    """Raise if ``spectrum`` is shorter than the ``lmax_int`` the ACC term reads."""
    needed = acc_internal_lmax(lmax, kernel_size, centralell)
    if len(spectrum) < needed:
        raise ValueError(
            f"ACC spectrum {name} has {len(spectrum)} multipoles (ell 0.."
            f"{len(spectrum) - 1}); the ACC term needs {needed} (lmax_int = lmax "
            f"{lmax} + pad {needed - lmax}, the pad being kernel size "
            f"{kernel_size} - 1 - centralell {centralell}) so that no coupling "
            "window is cut at lmax. Supply the spectra to ell = "
            f"{needed - 1}; the covariance is still reported to lmax."
        )
