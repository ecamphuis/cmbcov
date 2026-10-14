"""Memory plan of the batched Gauss-Legendre rows.

Sizes the chunks of columns a batched row processes at once from ``max_memory_gb``
and decides whether the Legendre table is held or regenerated per chunk
(:func:`_batched_row_plan`).
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import ducc0
import numpy as np

from ..harmonics.grid import gl_legendre_table_bytes, gl_north_rings, gl_shape

# --------------------------------------------------------------------------- #
# Batched GL columns: Legendre GEMMs over a chunk of columns
# --------------------------------------------------------------------------- #
#: Default memory budget of a batched exact GL row, in GiB (``max_memory_gb``
#: of :func:`exact_covariance_row` / :func:`exact_covariance`); what it covers
#: is itemised in :func:`_batched_row_plan`.
DEFAULT_EXACT_MEMORY_GB = 2.0


#: Target size of one block of the Legendre table when the whole table is not
#: held (:class:`_BatchedGL`): large enough that generating it keeps every
#: worker busy, small next to the chunk working set.  Per spin-0 table: the
#: spin-2 table is twice as large per order, and a streamed block holding
#: spin 0 and spin 2 targets three times this, so that it spans as many
#: orders as a TT block (the generation cost is dominated by a fixed cost
#: per ducc0 call, one call per ring pair and block, so more orders per call
#: is cheaper per entry).  A held table is generated once, in blocks of this
#: size whatever its spins.
_TABLE_BLOCK_BYTES = 64 * 2**20


#: Bytes of the ring block whose two FFTs and mask product run together in
#: :func:`_ring_stage` (cache-sized, so the three passes hit cache).
_FFT_BLOCK_BYTES = 4 * 2**20


#: Most workers of :func:`_ring_stage` (and at most ``nthreads``).  Each
#: holds numpy temporaries of about one ring block (elementwise operations
#: on non-contiguous views buffer), which the plan charges for this many
#: workers whatever ``nthreads`` is, so that the chunks do not depend on it.
_RING_WORKERS = 16


def _ring_blocks(n_fft: int, n_north: int, width: int) -> tuple[int, int]:
    """``(rings per block, blocks)`` of :func:`_ring_stage`: fixed blocks
    of about half of :data:`_FFT_BLOCK_BYTES` (at least one ring) tiling
    the northern rings, whatever the number of workers."""
    step = min(n_north, max(1, _FFT_BLOCK_BYTES // (2 * n_fft * width * 16)))
    return step, -(-n_north // step)


def _ring_scratch(n_fft: int, ntheta: int, width: int) -> int:
    """Bytes charged for the ring stage's workers: one block each, for as
    many workers as can run (:data:`_RING_WORKERS`, at most one per
    block), independently of ``nthreads``."""
    step, nblock = _ring_blocks(n_fft, (ntheta + 1) // 2, width)
    return min(_RING_WORKERS, nblock) * step * n_fft * width * 16


#: ``(what, budget)`` already warned about by :func:`_batched_row_plan`.
_WARNED_MEMORY_FLOOR: set = set()


def _band(mps: np.ndarray, width: int, lmax: int) -> tuple[int, int]:
    """``[min m' - width, max m' + width]`` clipped to ``[-lmax, lmax]``."""
    return max(-lmax, int(mps.min()) - width), min(lmax, int(mps.max()) + width)


def _chunk_fft_size(mps: np.ndarray, lmax: int, lw: int) -> int:
    r"""
    Azimuthal FFT length of a chunk (:func:`_batched_chunk_gl`).

    Step 3 multiplies ring profiles with orders in the band
    ``B = [min m' - lw, max m' + lw]`` (the orders step 1 can reach) by a
    mask of orders ``|k| <= lw``; the product occupies ``|B| + 2 lw``
    consecutive orders, so any ``N`` at least that long resolves it without
    aliasing.  Only the outputs ``|M| <= lmax`` are kept, and a product order
    ``p`` and an output ``M`` differ by at most ``2 lmax + lw``, so
    ``N >= 2 lmax + lw + 1`` is enough too (the grid's ``nphi`` is such an
    ``N``).  The smaller of the two, rounded up to a fast FFT length.
    """
    lo, hi = _band(mps, lw, lmax)
    return int(ducc0.fft.good_size(min(hi - lo + 1 + 2 * lw, 2 * lmax + lw + 1)))


def _chunk_bytes(nc: int, n_fft: int, ntheta: int, lmax: int) -> dict[str, int]:
    """Working set of one chunk of ``nc`` columns (:func:`_batched_row_plan`),
    in bytes, as allocated by :func:`_batched_chunk_gl`."""
    terms = {
        # the chunk's ring profiles, complex, (n_fft, ntheta, nc)
        "fields": n_fft * ntheta * nc * 16,
        # the mask on the n_fft points of every ring, and its FFT input
        "ring_mask": n_fft * ntheta * (8 + 16),
        # per order (both signs of M): the step-1 legs g_N +- g_S, a gathered
        # mask block, the GEMM outputs in L and on the rings; the step-1
        # profiles and their alm2leg output
        "scratch": (3 * ntheta + (lmax + 1)) * 2 * nc * 16 + ntheta * nc * 24,
        # the ring stage's workers
        "ring": _ring_scratch(n_fft, ntheta, nc),
    }
    terms["total"] = sum(terms.values())
    return terms


def _table_bytes(
    ms: np.ndarray, lmax: int, lmax_grid: int, spins: Sequence[int]
) -> int:
    """Bytes of the Legendre tables of ``spins`` for the orders ``ms``."""
    return sum(gl_legendre_table_bytes(ms, lmax, lmax_grid, spin=s) for s in spins)


def _table_block_target(spins: Sequence[int], hold_table: bool) -> int:
    """Target bytes of a table block: :data:`_TABLE_BLOCK_BYTES` for a held
    table, and that per spin-0-sized table (1 for spin 0, 2 for spin 2) for
    a streamed one."""
    if hold_table:
        return _TABLE_BLOCK_BYTES
    return _TABLE_BLOCK_BYTES * sum(1 if s == 0 else 2 for s in spins)


def _chunk_bytes_pol(
    nc: int,
    n_fft: int,
    ntheta: int,
    lmax: int,
    want_t: bool = True,
    nz: int = 3,
    n_out: int = 3,
    nkeys: int = 45,
) -> dict[str, int]:
    """Working set of one chunk of ``nc`` orders ``m'`` of a polarised row
    (:func:`_batched_chunk_pol_gl`), in bytes: ``nz`` column types per
    ``m'`` (T, E, B), each carried on the rings as T (if ``want_t``), Q and
    iU, i.e. ``width = (want_t + 2) nz nc`` complex ring columns, and
    ``n_out`` output fields contracted into ``nkeys`` sums.  An upper bound
    of what the kernel allocates at once."""
    n_north = (ntheta + 1) // 2
    w = nz * nc
    width = (int(want_t) + 2) * w
    rows = lmax + 2  # even + odd degrees of one order, both signs of M below
    terms = {
        # the chunk's ring profiles, complex, (n_fft, ntheta, width)
        "fields": n_fft * ntheta * width * 16,
        # the mask on the n_fft points of every ring, and its FFT input
        "ring_mask": n_fft * ntheta * (8 + 16),
        # the step-1 unit profiles (T, lambda^+, lambda^-) with ducc0's
        # output and its input alm, and the legs of both signs of M
        "profiles": (lmax + 1) * nc * 48 + n_north * nc * 112,
        "legs": ntheta * 2 * nc * 16 * 3,
        # per order, both signs of M: step-1 products and sources, the mixed
        # coefficients, the synthesis products; then per signed order the
        # analysis products, the outputs and the product scratch
        "step1": rows * 2 * nc * 16 * 8 + rows * 2 * nc * 16,
        "mix": rows * 2 * w * 16 * 3,
        "synthesis": ntheta * 2 * w * 16 * 5 + n_north * w * 16,
        "analysis": rows * w * 16 * 4,
        # the staging buffer while it is contracted, and the accumulators
        "stage": _stage_rows(lmax, n_out * nz, nc, nkeys)
        * _stage_row_bytes(n_out * nz, nc, nkeys)
        + (lmax + 1) * nkeys * 8,
        # the ring stage's workers
        "ring": _ring_scratch(n_fft, ntheta, width),
    }
    terms["total"] = sum(terms.values())
    return terms


def _stream_elements(spins: Sequence[int], lmax: int, lmax_grid: int) -> dict[int, int]:
    """``complex128`` elements of the reused buffer of each spin that
    streamed table blocks are built in (:meth:`_BatchedGL.tables`): the
    largest block's share of that spin (a block grows to
    :func:`_table_block_target` and overshoots by less than one order, and
    never exceeds the whole table; spin 0 and spin 2 split it 1 : 2) plus
    the front part :func:`~.grid.gl_legendre_table` addresses below the
    lowest order, ``(lmax + 1) npair`` per component."""
    units = {0: 1, 2: 2}
    total_units = sum(units[sp] for sp in spins)
    block = min(
        _table_block_target(spins, False)
        + _table_bytes(np.array([0]), lmax, lmax_grid, spins),
        _table_bytes(np.arange(lmax + 1), lmax, lmax_grid, spins),
    )
    npair = (gl_north_rings(lmax_grid) + 1) // 2
    return {
        sp: -(-block * units[sp] // total_units) // 16 + units[sp] * (lmax + 1) * npair
        for sp in spins
    }


def _plane_blocks(n_fft: int, ntheta: int, nplane: int) -> tuple[int, int]:
    """``(rings per block, blocks per plane)`` of :func:`_plane_ring_stage`:
    fixed blocks of about half of :data:`_FFT_BLOCK_BYTES` of one plane."""
    per = max(1, min(ntheta, _FFT_BLOCK_BYTES // (2 * n_fft * 16)))
    return per, -(-ntheta // per)


def _plane_ring_scratch(n_fft: int, ntheta: int, nplane: int) -> int:
    """Bytes charged for the workers of :func:`_plane_ring_stage` (one block
    each, :data:`_RING_WORKERS` at most), independently of ``nthreads``."""
    per, nper = _plane_blocks(n_fft, ntheta, nplane)
    return min(_RING_WORKERS, nplane * nper) * per * n_fft * 16


def _chunk_bytes_pol_direct(
    nc: int,
    n_fft: int,
    ntheta: int,
    lmax: int,
    lw: int,
    want_t: bool = True,
    nz: int = 3,
    n_out: int = 3,
    nkeys: int = 45,
) -> dict[str, int]:
    """Working set of one chunk of :func:`_batched_chunk_pol_direct`, in
    bytes (an upper bound of what it allocates at once)."""
    width = (int(want_t) + 2) * nz * nc
    n1 = min(2 * lw + 1, 2 * lmax + 1)  # orders of a column in step 1
    alm1 = n1 * (lmax + 1)  # their coefficients, degrees |M| .. lmax
    terms = {
        # the ring profiles, complex planes (width, ntheta, n_fft)
        "fields": n_fft * ntheta * width * 16,
        # the mask on the n_fft points of every ring: its FFT input, real
        # part and the repeated (real, imaginary) copy
        "ring_mask": n_fft * ntheta * (16 + 8 + 16),
        # the unit profiles of the columns (T, Q, U) and ducc0's output
        "profiles": 2 * 3 * ntheta * nc * 16 + 3 * (lmax + 1) * n1 * 16,
        # one column of step 1 and the synthesis: the legs and their
        # product, the step-1 coefficients, the mixed ones and scratch
        "column": 4 * ntheta * n1 * 16 + 14 * alm1 * 16,
        # the staging buffer (its front part and the rows) while contracted,
        # the accumulators, and a pair of scratch coefficients
        "stage": (lmax + 1) * n_out * nz * nc * 16
        + _stage_rows(lmax, n_out * nz, nc, nkeys)
        * (_stage_row_bytes(n_out * nz, nc, nkeys) + 2 * 16)
        + (lmax + 1) * nkeys * 8,
        # the ring stage's workers
        "ring": _plane_ring_scratch(n_fft, ntheta, width),
    }
    terms["total"] = sum(terms.values())
    return terms


def _batched_row_plan(
    mps: Sequence[int],
    lmax: int,
    lw: int,
    lmax_grid: int,
    max_memory_bytes: int,
    hold_table: bool | None = None,
    spins: Sequence[int] = (0,),
    layout: tuple[bool, int, int, int] | None = None,
) -> dict:
    r"""
    The deterministic memory plan of a batched exact GL row: which columns
    go together, and what is held.

    Everything the batched row allocates that scales with the problem is
    charged against ``max_memory_bytes``:

    ``modes``
        the mask's ring modes :math:`W_k(\theta)`, ``|k| <= lw``
        (:func:`~.grid.gl_ring_modes`), ``ntheta (2 lw + 1)`` complex, and
        their sums and differences over mirrored rings (as much again);
    ``table``
        the Legendre table :math:`\lambda_{LM}(\theta)` on the northern rings
        (:func:`~.grid.gl_legendre_table`): the whole table for
        ``M = 0 .. lmax`` if it fits in half the budget (``hold_table``;
        it is then kept for every row sharing the plan, built in place);
        otherwise streamed in blocks of about :func:`_table_block_target`
        bytes, re-generated for each chunk and each of the two passes over
        the orders (up to the degree each pass needs), all in one reused
        buffer per spin (:func:`_stream_elements`).  A polarised row holds
        spin-2 tables too (``spins``);
    per chunk (:func:`_chunk_bytes`)
        the ring profiles of its columns, ``N ntheta nc`` complex with ``N``
        the chunk's FFT length (:func:`_chunk_fft_size`) -- the dominant
        term -- the mask sampled on ``N`` points per ring, and per-order
        scratch.

    Nothing in the plan depends on the number of threads, so the chunks, and
    with them the summation order, are the same for any ``nthreads``: the
    rows are bit-identical across thread counts.

    A polarised row (:func:`_batched_chunk_pol_gl`) passes ``spins`` (the
    tables it holds: spin 2, and spin 0 if T is requested) and ``layout =
    (want_t, nz, n_out, nkeys)`` (T among the outputs, number of column
    types and of output fields, sums contracted), and its chunks are charged
    by :func:`_chunk_bytes_pol`.

    Columns are taken in the order given and a chunk grows while its
    working set fits in what the budget leaves after ``modes`` and
    ``table``.  A budget too small for a single column is warned about
    (once per budget) and one column per chunk is used.  Not charged:
    the fixed baseline -- the Python process, the mask as loaded, the
    healpy-ordered mask alm, and the row itself -- nor ducc0's and the BLAS
    library's internal buffers, nor freed memory the system allocator keeps,
    so the resident size of the process can exceed the budget.

    Returns
    -------
    dict
        ``chunks`` (list of int arrays of ``m'``), ``hold_table``, and the
        byte terms ``modes``, ``table``, ``chunk`` (the largest chunk's
        :func:`_chunk_bytes` total) and ``total`` (their sum, the peak).
    """
    ntheta, _ = gl_shape(lmax_grid)
    mps = np.asarray(list(mps), dtype=np.int64)
    modes = 2 * ntheta * (2 * lw + 1) * 16
    full_table = _table_bytes(np.arange(lmax + 1), lmax, lmax_grid, spins)
    if hold_table is None:
        hold_table = full_table <= max_memory_bytes // 2
    # Built in place, with no second buffer: a held table block by block
    # into its own memory, a streamed one into one reused buffer per spin
    # (_stream_elements, charged whole although its front part is never
    # written).
    direct = layout is not None and len(layout) > 4 and bool(layout[4])
    if hold_table:
        table = full_table
    elif direct:  # the table-free polarised kernel holds no table
        table = 0
    else:
        table = 16 * sum(_stream_elements(spins, lmax, lmax_grid).values())
    left = max_memory_bytes - modes - table

    def chunk_bytes(nc: int, n_fft: int) -> int:
        if layout is None:
            return _chunk_bytes(nc, n_fft, ntheta, lmax)["total"]
        if direct:
            return _chunk_bytes_pol_direct(nc, n_fft, ntheta, lmax, lw, *layout[:4])[
                "total"
            ]
        return _chunk_bytes_pol(nc, n_fft, ntheta, lmax, *layout[:4])["total"]

    chunks, largest, start, floor = [], 0, 0, False
    while start < mps.size:
        stop = start + 1
        while stop < mps.size:
            cand = mps[start : stop + 1]
            n_fft = _chunk_fft_size(cand, lmax, lw)
            if chunk_bytes(cand.size, n_fft) > left:
                break
            stop += 1
        chunk = mps[start:stop]
        n_fft = _chunk_fft_size(chunk, lmax, lw)
        size = chunk_bytes(chunk.size, n_fft)
        floor |= size > left
        largest = max(largest, size)
        chunks.append(chunk)
        start = stop
    if floor:
        key = (
            "exact GL row",
            int(max_memory_bytes),
            int(lmax),
            int(lmax_grid),
            tuple(spins),
            layout,
        )
        if key not in _WARNED_MEMORY_FLOOR:
            _WARNED_MEMORY_FLOOR.add(key)
            warnings.warn(
                f"max_memory_gb = {max_memory_bytes / 1024**3:.3g} is below the "
                f"floor of the batched exact row at lmax_int={lmax}, grid "
                f"{lmax_grid} ({(modes + table + largest) / 1024**3:.3g} GiB for "
                "one column per chunk); using one column per chunk.",
                UserWarning,
                stacklevel=3,
            )
    return {
        "chunks": chunks,
        "hold_table": bool(hold_table),
        "modes": modes,
        "table": table,
        "chunk": largest,
        "total": modes + table + largest,
    }


def _truncate(entry, n: int):
    """The tables of one order (``(even, odd)``, or ``(spin-0 pair or
    None, spin-2 quadruple)``) cut to their first ``n`` degrees."""
    if entry is None:
        return None
    if isinstance(entry[0], np.ndarray):
        return tuple(t[: (n + 1 - k % 2) // 2] for k, t in enumerate(entry))
    return tuple(_truncate(part, n) for part in entry)


def _first_row(first: int, lo: int) -> int:
    """Index of the first row with degree ``>= lo`` in a parity table whose
    row 0 has degree ``first`` (rows two degrees apart)."""
    return max(0, (lo - first + 1) // 2)


def _window(ellp: int, width: int, lmax: int) -> tuple[int, int]:
    """Degrees ``|L - ellp| <= width`` within ``[0, lmax]``."""
    return max(0, ellp - width), min(lmax, ellp + width)


#: Target bytes of the staging buffer of :func:`_batched_chunk_pol_gl`.
_STAGE_BYTES = 16 * 2**20


def _stage_rows(lmax: int, npair: int, nc: int, nkeys: int) -> int:
    """Rows of the staging buffer of :func:`_batched_chunk_pol_gl`: about
    :data:`_STAGE_BYTES` but at most 16 orders' worth of rows, and at least
    the ``lmax + 1`` rows of one order."""
    per_row = _stage_row_bytes(npair, nc, nkeys)
    return max(lmax + 1, min(_STAGE_BYTES // per_row, 16 * (lmax + 1)))


def _stage_row_bytes(npair: int, nc: int, nkeys: int) -> int:
    """Bytes per staged row while it is contracted: the outputs, their
    weighted and conjugated copies, the Gram matrix and the selected
    values."""
    return 3 * npair * nc * 16 + npair * npair * 16 + nkeys * 8
