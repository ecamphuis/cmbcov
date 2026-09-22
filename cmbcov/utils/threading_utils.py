"""
Threading and performance utilities.

This module contains utilities for threading optimization and
performance tuning commonly used in numerical computations.
"""

import multiprocessing
import os
import warnings

#: Whether an invalid ``OMP_NUM_THREADS`` has already been warned about in
#: this process (warn once, since threading helpers run on every SHT/grid
#: call).
_WARNED_INVALID_OMP_NUM_THREADS: list = []


def _env_nthreads() -> int | None:
    """
    Parse ``OMP_NUM_THREADS`` from the environment, shared by every
    thread-count policy in this package (:func:`get_optimal_nthreads` and
    :func:`~.grid._nthreads`'s grid-size-aware branch), so both honour it the
    same way instead of re-parsing it independently.

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
