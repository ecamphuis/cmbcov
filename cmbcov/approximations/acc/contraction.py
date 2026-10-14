"""Blocked contraction of the ACC coupling kernels.

The working-set and block-size model of the precompute, the per-block
contraction (:func:`contract_coupling_block`) and the accumulation of the
kernels over all ``(ell, ell')`` pairs (:func:`_accumulate_coupling_kernels`).
"""

import warnings
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .channels import _CENTRAL_FIELD, _PRIME_FIELD, COUPLING_SPECTRA
from .producers import _pack_block, _paired_m_order

__all__ = [
    "DEFAULT_COUPLING_MEMORY_GB",
    "contract_coupling_block",
]


#: Default memory budget of the ACC precompute, in GiB: everything it holds
#: that scales with the problem -- coefficient sets held in RAM and the
#: blocked working set of :func:`_accumulate_coupling_kernels`
#: (:func:`_coupling_working_set`) -- on top of a fixed baseline (see
#: :func:`precompute_acc_kernels`, ``max_memory_gb``).
DEFAULT_COUPLING_MEMORY_GB = 2.0


#: Inner-block width aimed at when the whole inner set does not fit in the
#: budget (:func:`_coupling_block_sizes`).  The inner side is re-visited once
#: per outer block -- re-synthesised on the streamed path, re-read from RAM or
#: from the disk store otherwise -- so the budget is spent on the outer block
#: to keep the number of passes down, while this floor keeps the per-``L``
#: GEMMs wide enough to run near BLAS peak.
_INNER_BLOCK_TARGET = 32


#: ``(max_bytes, floor)`` pairs already warned about by
#: :func:`_coupling_block_sizes`: the precompute asks once per ``(ell, ellp)``
#: pair, always with the same numbers.
_WARNED_MEMORY_FLOORS: set = set()


def _coupling_working_set(
    n_outer: int,
    n_inner: int,
    lmax: int,
    per_m: int,
    per_pair: int,
    nspec: int,
    inner_copies: int = 1,
    producer_bytes: int | None = None,
) -> dict[str, int]:
    """
    The peak resident bytes the blocked contraction
    (:func:`_accumulate_coupling_kernels`) is charged for, term by term, with
    outer blocks of ``n_outer`` orders and inner blocks of ``n_inner``:

    ``kernels``
        the ``(nspec, nspec, lmax, lmax)`` float64 accumulator, plus the one
        ``lmax x lmax`` gram being added into it
        (:func:`contract_coupling_block` with ``out``).
    ``outer``
        the outer coefficient block, ``n_outer * per_m``.
    ``inner``
        ``inner_copies`` inner blocks (2 when the next one is read ahead),
        ``inner_copies * n_inner * per_m``.
    ``theta``
        ``n_outer * n_inner * per_pair``: ``per_pair`` is the ``Theta`` bytes
        of one ``(m, m')`` pair, i.e. ``(nchannels + 1) * lmax * 16`` -- the
        float ``(Re, Im)`` slab of every channel plus the complex channel
        being converted into it.
    ``producer``
        what the coefficient producer holds beyond the block it fills
        (:func:`_gl_producer_bytes`, :func:`_healpix_producer_bytes`);
        ``None`` charges three per-``m`` sets, the GL figure without its
        legs and mask modes.
    ``total``
        the sum.  An upper bound: the producer only runs between
        contractions, and while an outer block is built at most one inner
        block is resident, but every term is charged as if all were
        resident at once.

    Not charged here, because the caller charges or excludes them: a central
    set held in RAM, and primed sets kept for a later ``ellp`` (subtracted
    from the budget before the contraction is planned,
    :meth:`_CouplingPrecompute._compute_ellp_coupling`), and the fixed
    baseline of :func:`precompute_acc_kernels` (``max_memory_gb``).
    """
    if producer_bytes is None:
        producer_bytes = 3 * per_m
    terms = {
        "kernels": (nspec * nspec + 1) * lmax * lmax * 8,
        "outer": n_outer * per_m,
        "inner": inner_copies * n_inner * per_m,
        "theta": n_outer * n_inner * per_pair,
        "producer": int(producer_bytes),
    }
    terms["total"] = sum(terms.values())
    return terms


def _coupling_block_sizes(
    n_m: int,
    n_mp: int,
    lmax: int,
    per_m: int,
    per_pair: int,
    max_bytes: int,
    nspec: int | None = None,
    inner_copies: int = 1,
    producer_bytes: int | None = None,
) -> tuple[int, int]:
    """
    Block widths ``(nb, nbp)`` over the outer and the inner side of the
    contraction, ``n_m`` and ``n_mp`` orders long: the blocking with the
    fewest passes whose :func:`_coupling_working_set` fits ``max_bytes``.

    ``inner_copies`` is how many inner blocks are resident at once: 1, or 2
    when the next inner block is read ahead while the current one is
    contracted (:func:`_accumulate_coupling_kernels`, primed-outer order).
    ``per_pair`` and ``producer_bytes`` are as in
    :func:`_coupling_working_set`; ``nspec`` (the side of the output
    accumulator) defaults to all of :data:`COUPLING_SPECTRA`, i.e. 5.

    The function is symmetric in what the two sides hold; which one is
    outer is the caller's choice (:func:`_accumulate_coupling_kernels`): the
    central ``m`` on the streamed path, the primed ``m'`` when the central
    set can be re-read (held in RAM or in the disk store).  Below, "central"
    means outer and "primed" inner.

    The primed block is the inner loop and is revisited (rebuilt or re-read)
    once per central block, so the budget is spent on ``nb`` first and
    ``nbp`` is held at :data:`_INNER_BLOCK_TARGET` (halved only if even that
    will not fit, or if the halving buys enough central width to cut the
    number of block passes; choosing by passes keeps the blocking monotonic
    in the budget).  When everything fits, both blocks cover the whole range
    and the contraction is a single pass.

    There is a hard floor -- the working set with one ``m`` on each side.
    A budget below it cannot be honoured; the smallest blocking is used anyway
    and a :class:`UserWarning` reports what the run will actually need.
    """
    if nspec is None:
        nspec = len(COUPLING_SPECTRA)

    def working_set(nb, nbp):
        return _coupling_working_set(
            nb, nbp, lmax, per_m, per_pair, nspec, inner_copies, producer_bytes
        )

    floor = working_set(1, 1)["total"]
    if max_bytes < floor:
        if (max_bytes, floor) in _WARNED_MEMORY_FLOORS:
            return 1, 1
        _WARNED_MEMORY_FLOORS.add((max_bytes, floor))
        warnings.warn(
            f"coupling memory budget of {max_bytes / 1024**2:.1f} MiB is below "
            f"the {floor / 1024**2:.1f} MiB this (ell, ellp, lmax) needs for "
            "the output kernels, the coefficient producer and one m on each "
            "side; proceeding with single-m blocks (warned once per process "
            "for this budget)",
            UserWarning,
            stacklevel=2,
        )
        return 1, 1
    # Everything that does not scale with the block widths.
    avail = max_bytes - working_set(0, 0)["total"]

    def widest_nb(nbp):
        # The largest nb with working_set(nb, nbp) <= max_bytes: the terms
        # are linear in nb at fixed nbp.
        rest = avail - inner_copies * nbp * per_m
        return min(n_m, int(rest // (per_m + nbp * per_pair))) if rest > 0 else 0

    # Candidate primed widths: the target and its halvings, plus -- only with
    # the whole central range in one block -- its doublings up to n_mp.  Each
    # candidate's passes can only fall as the budget grows, so taking the one
    # with the fewest passes (ties: fewest central blocks, then the widest
    # nbp) is monotonic in the budget.
    nbp = min(n_mp, _INNER_BLOCK_TARGET)
    candidates = []
    trial = nbp
    while True:
        candidates.append((trial, widest_nb(trial)))
        if trial == 1:
            break
        trial = max(1, trial // 2)
    trial = nbp
    while trial < n_mp:
        trial = min(n_mp, 2 * trial)
        if working_set(n_m, trial)["total"] > max_bytes:
            break
        candidates.append((trial, n_m))

    best = None
    for nbp, nb in candidates:
        if nb < 1:
            continue
        central_blocks = -(-n_m // nb)
        key = (central_blocks * -(-n_mp // nbp), central_blocks, -nbp)
        if best is None or key < best[0]:
            best = (key, nb, nbp)
    if best is None:
        return 1, 1
    _, nb, nbp = best
    return nb, nbp


def contract_coupling_block(
    left,
    right,
    central_field: Sequence[int],
    prime_field: Sequence[int],
    pair_mask=None,
    xp=np,
    output_pairs: Sequence[tuple[int, int]] | None = None,
    out=None,
):
    r"""
    The contraction of one ``(m, m')`` block of the ACC precompute: the
    per-``L`` matmul that builds ``Theta`` and the real gram that turns it
    into the kernel increment.

    .. math::

        \Theta^{s}(m, m', L) = \sum_M X^{a(s)}_{m L M}\, \bar Y^{b(s)}_{m' L M},
        \qquad
        \Delta K^{s_1 s_2}(L_1, L_2) = \mathrm{Re} \sum_{(m, m') \in \text{block}}
            \Theta^{s_1}(m, m', L_1)\, \overline{\Theta^{s_2}(m, m', L_2)} .

    Parameters
    ----------
    left : array
        ``(nfields, lmax, na, ncoef)`` complex, the central coefficients of
        the block (:func:`_pack_block` with ``conjugate=False``).
    right : array
        ``(nfields, lmax, ncoef, nbj)`` complex, the **conjugated** primed
        coefficients (:func:`_pack_block` with ``conjugate=True``, a
        transposed view; any strides ``matmul`` accepts).
    central_field, prime_field : sequence of int
        Field index ``(T, D, L) = (0, 1, 2)`` of the central and primed leg
        of every output channel (:data:`_CENTRAL_FIELD` / :data:`_PRIME_FIELD`
        restricted to the requested spectra); ``nspec = len(central_field)``.
    pair_mask : array of bool, optional
        ``(na, nbj)``: the ``(m, m')`` pairs of the block that contribute
        (term selection).  ``None`` keeps every pair, at no cost.
    output_pairs : sequence of (int, int), optional
        Restrict the output to these ``(k1, k2)`` channel-index pairs
        (indices into ``central_field``/``prime_field``) instead of the full
        ``nspec x nspec`` square.  ``Theta`` (``flats``) is then built only
        for the channels that actually appear in ``output_pairs``, and the
        gram (``dgemm``) only for the pairs themselves -- no symmetric
        mirroring, since an explicit pair list need not be symmetric.
        ``None`` (default) is the full square.
    xp : module
        Array namespace, ``numpy`` by default.  Only operations that exist
        identically in ``numpy`` and ``jax.numpy`` are used (``matmul``,
        ``reshape``, ``real``/``imag``, ``stack``, ``where``, ``.T``): no
        in-place writes, no ``out=``.  Passing ``xp=jax.numpy`` with
        ``left``/``right`` (and ``pair_mask``) already on the device is the
        intended GPU path; the producers of the coefficients (the SHTs and
        banded Legendre transforms of :mod:`cmbcov.harmonics.grid`)
        stay on the CPU and the blocks are moved over one at a time.
    out : ndarray, optional
        numpy only: a ``(nspec, nspec, lmax, lmax)`` float64 accumulator the
        increment is added into, one gram at a time, instead of being
        returned as a new array -- what :func:`_accumulate_coupling_kernels`
        uses, so that the only kernel-sized transient is one ``lmax x lmax``
        gram rather than the stacked increment and its rows.

    Returns
    -------
    array
        ``(nspec, nspec, lmax, lmax)`` float64 kernel increment, ``[s1, s2]``
        for ``s2 < s1`` the transpose of ``[s2, s1]``; with ``out``, ``out``
        after the increment has been added.

    Notes
    -----
    The ``Re`` of the gram needs no complex GEMM: with ``(Re, Im)``
    interleaved along the flattened ``(m, m')`` axis,

        ``sum_f (Re a Re b + Im a Im b) = Re sum_f a conj(b)``

    is a real dot product over that doubled axis, one ``dgemm`` per channel
    pair at half the flops of a ``zgemm``.  ``stack((real, imag), -1)``
    builds this layout without in-place writes, so the same code runs under
    ``numpy`` and ``jax.numpy``.  The cost is one transient complex ``Theta``
    of one channel next to the float slab.
    """
    nspec = len(central_field)
    if len(prime_field) != nspec:
        raise ValueError("central_field and prime_field must have the same length")
    lmax, na = left.shape[1], left.shape[2]
    nbj = right.shape[3]
    npair = na * nbj
    mask_flat = None
    if pair_mask is not None:
        mask_flat = xp.asarray(pair_mask, dtype=bool).reshape(1, npair)

    def theta_flat(k: int):
        a, b = central_field[k], prime_field[k]
        theta = xp.matmul(left[a], right[b]).reshape(lmax, npair)
        if mask_flat is not None:
            theta = xp.where(mask_flat, theta, xp.zeros((), dtype=theta.dtype))
        # (Re, Im) interleaved along the flattened (m, m') axis.
        return xp.stack((theta.real, theta.imag), axis=-1).reshape(lmax, 2 * npair)

    if out is not None:
        # numpy accumulation: the same grams, added into out as they come.
        if output_pairs is None:
            flats = [theta_flat(k) for k in range(nspec)]
            for k1 in range(nspec):
                for k2 in range(k1, nspec):
                    gram = xp.matmul(flats[k1], flats[k2].T)
                    out[k1, k2] += gram
                    if k2 != k1:
                        out[k2, k1] += gram.T
                    del gram
            return out
        needed = sorted({k for pair in output_pairs for k in pair})
        flat_of = {k: theta_flat(k) for k in needed}
        for k1, k2 in dict.fromkeys(output_pairs):
            out[k1, k2] += xp.matmul(flat_of[k1], flat_of[k2].T)
        return out

    if output_pairs is None:
        flats = [theta_flat(k) for k in range(nspec)]
        blocks: list[list] = [[None] * nspec for _ in range(nspec)]
        for k1 in range(nspec):
            for k2 in range(k1, nspec):
                gram = xp.matmul(flats[k1], flats[k2].T)
                blocks[k1][k2] = gram
                if k2 != k1:
                    blocks[k2][k1] = gram.T
        return xp.stack([xp.stack(row) for row in blocks])

    # Explicit pair list: build Theta only for the channels referenced (as
    # either leg) by output_pairs, and the gram only for the pairs
    # themselves -- no assumption of symmetry.
    needed = sorted({k for pair in output_pairs for k in pair})
    flat_of = {k: theta_flat(k) for k in needed}
    out = np.zeros((nspec, nspec, lmax, lmax))
    for k1, k2 in output_pairs:
        out[k1, k2] = np.asarray(xp.matmul(flat_of[k1], flat_of[k2].T))
    return out


def _contiguous_runs(positions: Sequence[int]) -> Iterator[tuple[int, int, int]]:
    """``(source start, destination start, length)`` of the maximal runs of
    consecutive ascending values in ``positions``."""
    start = 0
    for k in range(1, len(positions) + 1):
        if k == len(positions) or positions[k] != positions[k - 1] + 1:
            yield positions[start], start, k - start
            start = k


def _accumulate_coupling_kernels(
    central_fn: Callable[[int], np.ndarray],
    n_m: int,
    prime_fn: Callable[[int], np.ndarray],
    n_mp: int,
    lmax: int,
    max_bytes: int,
    spec_indices: Sequence[int] | None = None,
    nfields: int = 3,
    verbose: bool = False,
    keep_m: np.ndarray | None = None,
    keep_mp: np.ndarray | None = None,
    pair_mask: np.ndarray | None = None,
    output_pairs: Sequence[tuple[int, int]] | None = None,
    central_reader: Callable[[Sequence[int], np.ndarray], np.ndarray] | None = None,
    prime_reader: Callable[[Sequence[int], np.ndarray], np.ndarray] | None = None,
    producer_bytes: int | None = None,
) -> np.ndarray:
    r"""
    The whole ACC precompute for one ``(ell, ellp)`` pair, as blocked BLAS.

    Both grids reduce to the same two contractions on complex full-``M``
    coefficients (:func:`_coefficient_provider`).  With ``X`` the central
    coefficient stack and ``Y`` the primed one (indices ``[field, L, m, M]``,
    complex128, ``2 * lmax - 1`` values of ``M``),

    .. math::

        \Theta^{s}(m, m', L) = \sum_M X^{a(s)}_{m L M}\, \bar Y^{b(s)}_{m' L M}
        \qquad\text{(one GEMM per } L\text{)}

        K^{s_1 s_2}(L_1, L_2) = \mathrm{Re} \sum_{m m'}
            \Theta^{s_1}(m, m', L_1)\, \overline{\Theta^{s_2}(m, m', L_2)}
        \qquad\text{(one GEMM per channel pair)} ,

    with ``(a, b)`` given by :data:`_CENTRAL_FIELD` / :data:`_PRIME_FIELD`.
    ``Theta`` is kept complex on both grids, as Eq. 20 requires.  The ``(m,
    m')`` plane is blocked (:func:`_coupling_block_sizes`) and the kernels are
    accumulated block by block, so the full ``Theta`` (``nchannels (2 ell + 1)
    (2 ellp + 1) lmax`` complex values) is never materialised.

    The per-block arithmetic, including the real-``dgemm`` evaluation of the
    ``Re`` in the second line, lives in :func:`contract_coupling_block`; this
    function owns the blocking, the iteration order and the accumulation.

    ``spec_indices`` restricts the output to a subset of
    :data:`COUPLING_SPECTRA` (indices into it); ``None`` (the default) means
    all five.  ``nfields`` is the number of fields (T, D, L) that
    ``central_fn``/``prime_fn`` actually provide per ``m`` -- 3 normally, or 1
    when the caller has determined (:func:`_resolve_spectra`) that every
    requested kernel needs only T on both sides, in which case the
    coefficient providers were built with ``t_only=True`` and never expand
    fields 1, 2.

    **Term selection** (:mod:`cmbcov.approximations.acc.term_selection`).
    ``keep_m`` / ``keep_mp`` (bool, index ``m + ell`` / ``m' + ellp``)
    restrict the orders iterated over -- the paired ``+m, -m`` order of
    :func:`_paired_m_order` is kept, filtered to the kept indices, so a
    reflecting provider still serves ``-m`` from the transform of ``+m``
    whenever both are kept (the mode power is even in ``m``, so in practice
    they always are) -- and ``pair_mask`` (bool, ``(2 ell + 1, 2 ellp +
    1)``) zeroes the ``Theta`` entries of the pairs that are not kept before
    the gram; a block pair whose mask is entirely False is skipped without
    packing its primed side.  All three default to ``None``: every order and
    every pair.

    ``output_pairs`` (indices into ``spec_indices``, i.e. positions 0 ..
    ``nspec - 1``) restricts both the ``Theta`` built and the kernels
    accumulated to an explicit list of ordered channel pairs -- see
    :func:`contract_coupling_block`.  ``None`` (default) is the full square;
    an explicit list also narrows the ``Theta`` term of the memory-budget
    accounting (:func:`_coupling_block_sizes`) to the channels actually built
    (the accumulator is the full ``nspec x nspec`` square either way).

    **Memory.**  ``max_bytes`` bounds :func:`_coupling_working_set` of the
    chosen blocking: the accumulator and one gram, the resident coefficient
    blocks, the ``Theta`` slab and ``producer_bytes`` (what the coefficient
    producer holds beyond the block it fills; ``None`` charges three
    per-``m`` sets).  Each block's increment is added into the accumulator
    gram by gram (:func:`contract_coupling_block`, ``out``).

    **Loop order.**  With ``central_reader=None`` (the streamed path: every
    coefficient comes from a provider that synthesises it) the central
    blocks are the outer loop and every primed block is rebuilt once per
    central block.  A ``central_reader`` -- ``(rows, out) -> left block``
    (:func:`_pack_block`'s ``conjugate=False`` layout, written into the flat
    buffer ``out``), for a central set
    that is cheap to re-read (held in RAM, or :class:`_CentralStore` on
    disk) -- swaps the loops: the primed blocks are the outer loop, so every
    primed coefficient is synthesised exactly once, and the central blocks
    are re-read inside, alternately forwards and backwards, one block read
    ahead in a helper thread (the budget is charged for two central blocks,
    :func:`_coupling_block_sizes` ``inner_copies=2``).  ``prime_reader``
    (``(cols, out) -> right block``, ``conjugate=True`` layout) likewise
    serves the primed side, for ``ell == ellp`` from the store.  The loop
    order changes only the summation order of the kernels, i.e. the result
    agrees to rounding.

    Returns
    -------
    ndarray
        ``(nspec, nspec, lmax, lmax)`` float64, ``[s1, s2]`` being the kernel
        for ``COUPLING_SPECTRA[spec_indices[s1]] x
        COUPLING_SPECTRA[spec_indices[s2]]``, ``nspec = len(spec_indices)``
        (25 = 5x5 when ``spec_indices`` is ``None``).
    """
    if spec_indices is None:
        spec_indices = list(range(len(COUPLING_SPECTRA)))
    nspec = len(spec_indices)
    ncoef = 2 * lmax - 1
    itemsize = np.dtype(np.complex128).itemsize

    # The kernels sum over every (m, m') pair, so the iteration order is free;
    # a reflecting (GL) provider exploits a +m / -m pairing (see
    # _coefficient_provider).  With a selection the order is filtered, not
    # rebuilt, so kept +m / -m stay adjacent.
    m_order = _paired_m_order(n_m)
    mp_order = _paired_m_order(n_mp)
    if keep_m is not None:
        m_order = [i for i in m_order if keep_m[i]]
    if keep_mp is not None:
        mp_order = [j for j in mp_order if keep_mp[j]]
    n_kept, n_kept_p = len(m_order), len(mp_order)
    if pair_mask is not None:
        pair_mask = np.asarray(pair_mask, dtype=bool)
        if pair_mask.shape != (n_m, n_mp):
            raise ValueError(
                f"pair_mask must have shape {(n_m, n_mp)}; got {pair_mask.shape}"
            )

    # Theta is built only for the channels an explicit pair list references
    # (all of them otherwise); the accumulator is always the full square.
    if output_pairs is not None:
        n_channels_used = len({k for pair in output_pairs for k in pair})
    else:
        n_channels_used = nspec

    per_m = nfields * lmax * ncoef * itemsize
    # The float (Re, Im) slab of Theta, (n_channels_used, lmax, 2 nb nbp)
    # float64 -- the same bytes as the complex slab -- plus the one complex
    # channel of Theta being converted into it (contract_coupling_block).
    per_pair = (n_channels_used + 1) * lmax * itemsize
    primed_outer = central_reader is not None
    # _coupling_block_sizes returns (outer width, inner width).  Primed
    # outer, the next central block is read ahead: two resident at once.
    inner_copies = 2 if primed_outer else 1
    outer, inner = _coupling_block_sizes(
        n_kept_p if primed_outer else n_kept,
        n_kept if primed_outer else n_kept_p,
        lmax,
        per_m,
        per_pair,
        max_bytes,
        nspec=nspec,
        inner_copies=inner_copies,
        producer_bytes=producer_bytes,
    )
    nb, nbp = (inner, outer) if primed_outer else (outer, inner)

    if verbose:
        plan = _coupling_working_set(
            outer, inner, lmax, per_m, per_pair, nspec, inner_copies, producer_bytes
        )
        detail = ", ".join(
            f"{name} {plan[name] / 1024**3:.2f}"
            for name in ("outer", "inner", "theta", "kernels", "producer")
        )
        if primed_outer:
            print(
                f"  contraction blocking: m' in {nbp}-blocks "
                f"({-(-n_kept_p // nbp)} of them, outer, built once), "
                f"m in {nb}-blocks ({-(-n_kept // nb)} of them, re-read)"
            )
        else:
            print(
                f"  contraction blocking: m in {nb}-blocks "
                f"({-(-n_kept // nb)} of them), "
                f"m' in {nbp}-blocks ({-(-n_kept_p // nbp)} of them)"
            )
        print(
            f"  planned working set {plan['total'] / 1024**3:.2f} of "
            f"{max_bytes / 1024**3:.2f} GiB ({detail} GiB)"
        )

    central_fields = [_CENTRAL_FIELD[k] for k in spec_indices]
    prime_fields = [_PRIME_FIELD[k] for k in spec_indices]

    # Every block is written into one of these buffers, allocated once here
    # (the plan's "outer" and "inner" terms) rather than per block: the
    # system allocator may retain freed multi-GiB blocks, which would push
    # the footprint well past the budget.
    entries_per_m = nfields * lmax * ncoef
    outer_buffer = np.empty(outer * entries_per_m, dtype=np.complex128)
    inner_buffers = [
        np.empty(inner * entries_per_m, dtype=np.complex128)
        for _ in range(inner_copies)
    ]
    right_buffer = outer_buffer if primed_outer else inner_buffers[0]

    def pack_right(cols):
        if prime_reader is not None:
            return prime_reader(cols, right_buffer)
        return _pack_block(
            prime_fn, cols, lmax, conjugate=True, nfields=nfields, out=right_buffer
        )

    def block_mask_of(rows, cols):
        """``(skip, mask)`` of one block pair under term selection."""
        if pair_mask is None:
            return False, None
        mask = pair_mask[np.ix_(rows, cols)]
        if not mask.any():
            return True, None
        return False, (None if mask.all() else mask)

    out = np.zeros((nspec, nspec, lmax, lmax))
    skipped = 0
    if not primed_outer:
        for i0 in range(0, n_kept, nb):
            rows = m_order[i0 : i0 + nb]
            left = _pack_block(
                central_fn,
                rows,
                lmax,
                conjugate=False,
                nfields=nfields,
                out=outer_buffer,
            )

            for j0 in range(0, n_kept_p, nbp):
                cols = mp_order[j0 : j0 + nbp]
                skip, block_mask = block_mask_of(rows, cols)
                if skip:
                    skipped += 1
                    continue
                right = pack_right(cols)
                contract_coupling_block(
                    left,
                    right,
                    central_fields,
                    prime_fields,
                    block_mask,
                    output_pairs=output_pairs,
                    out=out,
                )
                del right
            del left
    else:
        inner_starts = list(range(0, n_kept, nb))
        reads = 0  # central blocks read so far: they alternate between buffers

        def read_next(rows):
            nonlocal reads
            buffer = inner_buffers[reads % inner_copies]
            reads += 1
            return reader.submit(central_reader, rows, buffer)

        # One reader thread keeps the next central block coming while the
        # current one is contracted (preadv, the copies and the BLAS calls
        # all release the GIL); the first read of each pass also overlaps
        # the synthesis of the primed block.
        with ThreadPoolExecutor(max_workers=1) as reader:
            for jb, j0 in enumerate(range(0, n_kept_p, nbp)):
                cols = mp_order[j0 : j0 + nbp]
                # Serpentine over the central blocks: the last block read in
                # one pass is the first of the next.
                starts = inner_starts[::-1] if jb % 2 else inner_starts
                tasks = []
                for i0 in starts:
                    rows = m_order[i0 : i0 + nb]
                    skip, block_mask = block_mask_of(rows, cols)
                    if skip:
                        skipped += 1
                    else:
                        tasks.append((rows, block_mask))
                if not tasks:
                    continue  # the primed block is never packed
                pending = read_next(tasks[0][0])
                right = pack_right(cols)
                for k, (_, block_mask) in enumerate(tasks):
                    left = pending.result()
                    # Into the other buffer: the one read two blocks ago,
                    # whose contraction has returned.
                    pending = read_next(tasks[k + 1][0]) if k + 1 < len(tasks) else None
                    contract_coupling_block(
                        left,
                        right,
                        central_fields,
                        prime_fields,
                        block_mask,
                        output_pairs=output_pairs,
                        out=out,
                    )
                    del left
                del right

    if verbose and pair_mask is not None:
        n_blocks = -(-n_kept // nb) * -(-n_kept_p // nbp)
        print(f"  term selection: {skipped} of {n_blocks} block pairs skipped")
    return out
