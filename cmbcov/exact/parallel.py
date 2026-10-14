"""Process-parallel evaluation of the rows of the exact covariance.

The workers are top-level functions of this module so that they can be pickled
by module path under the ``spawn`` start method.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

from ..harmonics.grid import gl_minimal_lmax, gl_synthesis
from .gl import _BatchedGL, _exact_row_gl
from .pol import _exact_row_pol_gl, _polarised_batch

# --------------------------------------------------------------------------- #
# Row-level parallelism (GL grid): rows are independent
# --------------------------------------------------------------------------- #
_WORKER: dict[str, object] = {}


def _row_worker_init(
    mask_alm,
    lw,
    cl,
    lmax,
    lmax_grid,
    use_symmetry,
    nthreads,
    lmax_int=None,
    term_selection=None,
    max_memory_gb=None,
    pol=None,
):
    """Per-process setup: keep the row arguments and build what the rows share
    (the batched path's ring modes and table policy, or the GL mask map).
    ``pol = (specs, fields)`` makes the rows polarised (``cl`` is then the
    spectra matrix of :func:`_cl_matrix`)."""
    _WORKER.clear()
    lmax_i = int(lmax if lmax_int is None else lmax_int)
    batch = None
    mask_map = None
    if lmax_grid >= gl_minimal_lmax(lmax_i, lw):
        if pol is None:
            batch = _BatchedGL(mask_alm, lw, lmax_i, lmax_grid, nthreads, max_memory_gb)
        else:
            batch = _polarised_batch(
                mask_alm, lw, lmax_i, lmax_grid, nthreads, max_memory_gb, pol[1]
            )
    else:
        mask_map = gl_synthesis(mask_alm, lw, lmax_grid, nthreads=nthreads)
    _WORKER.update(
        mask_alm=mask_alm,
        lw=lw,
        cl=cl,
        lmax=lmax,
        lmax_grid=lmax_grid,
        use_symmetry=use_symmetry,
        nthreads=nthreads,
        lmax_int=lmax_int,
        term_selection=term_selection,
        mask_map=mask_map,
        batch=batch,
        pol=pol,
    )


def _row_worker(ellp: int):
    """One row in a worker process; returns ``(ellp, Sigma[:, ellp])`` (for a
    polarised worker, the dict of blocks of the row)."""
    if _WORKER["pol"] is not None:
        return ellp, _exact_row_pol_gl(
            _WORKER["mask_alm"],
            _WORKER["lw"],
            _WORKER["cl"],
            ellp,
            _WORKER["lmax"],
            _WORKER["pol"][0],
            lmax_grid=_WORKER["lmax_grid"],
            use_symmetry=_WORKER["use_symmetry"],
            nthreads=_WORKER["nthreads"],
            mask_map=_WORKER["mask_map"],
            lmax_int=_WORKER["lmax_int"],
            term_selection=_WORKER["term_selection"],
            batch=_WORKER["batch"],
        )
    return ellp, _exact_row_gl(
        _WORKER["mask_alm"],
        _WORKER["lw"],
        _WORKER["cl"],
        ellp,
        _WORKER["lmax"],
        lmax_grid=_WORKER["lmax_grid"],
        use_symmetry=_WORKER["use_symmetry"],
        nthreads=_WORKER["nthreads"],
        mask_map=_WORKER["mask_map"],
        lmax_int=_WORKER["lmax_int"],
        term_selection=_WORKER["term_selection"],
        batch=_WORKER["batch"],
    )


def _rows_gl_parallel(
    mask_alm,
    lw,
    cl,
    rows,
    lmax,
    lmax_grid,
    use_symmetry,
    nthreads,
    nprocs,
    lmax_int=None,
    term_selection=None,
    max_memory_gb=None,
    pol=None,
):
    """
    ``[(ellp, row), ...]`` for every requested row, computed in ``nprocs``
    processes (``pol = (specs, fields)`` for the polarised rows, each a dict
    of blocks, with ``cl`` the spectra matrix).  Each row is bit-identical to
    the serial one for the same ``nthreads`` (ducc0 splits a transform
    deterministically over rings, and every row is computed by a single call
    chain; ``max_memory_gb`` is per process, so every worker plans the same
    chunks as the serial loop).
    """
    rows = list(rows)
    nprocs = min(nprocs, len(rows)) or 1
    if nthreads is None:
        nthreads = max(1, (os.cpu_count() or 1) // nprocs)
    # O(l'^4) per row: start the expensive ones first so the tail is short.
    order = sorted(rows, key=lambda lp: -lp)
    with ProcessPoolExecutor(
        max_workers=nprocs,
        initializer=_row_worker_init,
        initargs=(
            mask_alm,
            lw,
            cl,
            lmax,
            lmax_grid,
            use_symmetry,
            nthreads,
            lmax_int,
            term_selection,
            max_memory_gb,
            pol,
        ),
    ) as pool:
        return list(pool.map(_row_worker, order))
