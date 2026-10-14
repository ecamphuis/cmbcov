"""
Threading and performance utilities.

This module contains utilities for threading optimization and
performance tuning commonly used in numerical computations.
"""

import multiprocessing
import os
import warnings
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

#: Non-empty once an invalid ``OMP_NUM_THREADS`` has been warned about (the
#: warning is issued once per process; the helpers run on every SHT/grid call).
_WARNED_INVALID_OMP_NUM_THREADS: list = []


def _env_nthreads() -> int | None:
    """
    Parse ``OMP_NUM_THREADS`` from the environment.

    Shared by the thread-count policies of :func:`get_optimal_nthreads` and
    ``cmbcov.harmonics.grid._nthreads``, so both honour it the same way.

    Returns
    -------
    int or None
        The environment value clamped to at least 1 (``"0"`` and negative
        values are not a usable thread count), or ``None`` if the variable is
        unset or not an integer -- callers should fall back to their own
        auto-detection in that case. A non-integer value is warned about
        once per process.
    """
    env_threads = os.environ.get("OMP_NUM_THREADS")
    if env_threads is None:
        return None
    try:
        return max(1, int(env_threads))
    except ValueError:
        if not _WARNED_INVALID_OMP_NUM_THREADS:
            _WARNED_INVALID_OMP_NUM_THREADS.append(env_threads)
            warnings.warn(
                f"OMP_NUM_THREADS={env_threads!r} is not an integer; "
                "falling back to automatic thread-count detection. "
                "Warned once per process.",
                stacklevel=2,
            )
        return None


#: Environment variables that set the size of the BLAS, OpenMP and numexpr
#: thread pools of a process (:func:`child_thread_env`).
_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@contextmanager
def child_thread_env(nthreads: int) -> Iterator[None]:
    """
    Set every thread-pool variable of :data:`_THREAD_ENV_VARS` to
    ``max(1, nthreads)`` for the duration of the context, and restore the
    previous values (or remove the variables that were unset) on exit.

    Processes started while the context is active (``spawn``) inherit the
    limits, so their BLAS and OpenMP pools get that many threads.  The
    calling process's already-initialised BLAS is not affected, but anything
    in the caller that reads ``OMP_NUM_THREADS`` while the context is active
    (:func:`get_optimal_nthreads`) sees the child value: resolve the caller's
    own thread count before entering.
    """
    value = str(max(1, int(nthreads)))
    saved = {name: os.environ.get(name) for name in _THREAD_ENV_VARS}
    try:
        for name in _THREAD_ENV_VARS:
            os.environ[name] = value
        yield
    finally:
        for name, old in saved.items():
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old


def get_optimal_nthreads(nside: int | None = None) -> int:
    """
    Determine optimal number of threads for spherical harmonic computations.

    This function provides intelligent thread selection based on:
    1. OMP_NUM_THREADS environment variable (if set), clamped to at least 1
    2. Problem size (nside) and CPU count for optimal performance

    Parameters
    ----------
    nside : int, optional
        HEALPix resolution parameter for problem size estimation.
        If None, uses general-purpose thread count.

    Returns
    -------
    int
        Optimal number of threads for computation
    """
    # Check environment variable first (commonly used for OpenMP)
    env_nthreads = _env_nthreads()
    if env_nthreads is not None:
        return env_nthreads

    # Auto-detect based on CPU count and problem size
    n_cpu = multiprocessing.cpu_count()

    if nside is None:
        # General-purpose default
        return min(n_cpu // 2, 8)

    # For CMB analysis, threading efficiency depends on resolution
    # Higher nside benefits more from parallelization
    if nside <= 64:
        # For small problems, use fewer threads to avoid overhead
        return min(4, n_cpu)
    elif nside <= 512:
        # Medium problems can use more threads efficiently
        return min(n_cpu // 2, 8)
    else:
        # Large problems benefit from all available cores
        return min(n_cpu, 16)  # Cap at 16 to avoid diminishing returns


#: Complex128 entries per worker below which :func:`run_row_blocks` callers
#: do not split an elementwise array operation further
#: (:func:`row_block_workers`). The operations are memory-bound products and
#: copies; blocks of only a few rows (many small numpy calls contending for
#: the GIL) are slower than serial, hence one worker per 128 Ki entries (2 MiB).
ROW_BLOCK_ENTRIES_PER_THREAD = 131072

#: The shared worker pool of :func:`run_row_blocks`, created on first use and
#: dropped in a forked child, whose copy of it would have no threads.
_ROW_BLOCK_POOL: list = []

if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_ROW_BLOCK_POOL.clear)


def row_block_workers(entries: int, nthreads: int) -> int:
    """Workers for an elementwise operation over ``entries`` array entries:
    at most ``nthreads``, and one per :data:`ROW_BLOCK_ENTRIES_PER_THREAD`."""
    return max(1, min(int(nthreads), entries // ROW_BLOCK_ENTRIES_PER_THREAD))


def run_row_blocks(fn: Callable[[int, int], None], nrows: int, workers: int) -> None:
    """
    Call ``fn(r0, r1)`` on ``workers`` contiguous blocks ``[r0, r1)`` that
    tile ``range(nrows)``, on a shared thread pool, and wait for all of them
    (re-raising the first exception).  With one worker it is a plain call.

    Meant for numpy array operations split over the leading axis: numpy
    releases the GIL inside each ufunc loop, and an elementwise operation
    gives the same bits whatever the blocking, so callers stay
    bit-identical to the serial form.  ``fn`` must write disjoint rows and
    must not itself call :func:`run_row_blocks` (a worker waiting on the
    pool it runs in could deadlock it).
    """
    workers = max(1, min(int(workers), int(nrows)))
    if workers == 1:
        fn(0, nrows)
        return
    if not _ROW_BLOCK_POOL:
        _ROW_BLOCK_POOL.append(
            ThreadPoolExecutor(
                max_workers=os.cpu_count() or 1, thread_name_prefix="cmbcov-rows"
            )
        )
    bounds = [nrows * i // workers for i in range(workers + 1)]
    futures = [
        _ROW_BLOCK_POOL[0].submit(fn, bounds[i], bounds[i + 1]) for i in range(workers)
    ]
    for future in futures:
        future.result()
