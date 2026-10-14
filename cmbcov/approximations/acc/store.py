"""On-disk store of the central-multipole coefficients of the ACC precompute."""

import os
import shutil
from collections.abc import Callable, Sequence

import numpy as np

from ...utils.threading_utils import (
    get_optimal_nthreads,
    row_block_workers,
    run_row_blocks,
)
from .contraction import _contiguous_runs
from .producers import _block_array, _pack_block

try:  # uncached I/O for the central store (_CentralStore); POSIX-only
    import fcntl

    _F_NOCACHE = getattr(fcntl, "F_NOCACHE", None)  # macOS
except ImportError:  # pragma: no cover - Windows
    fcntl = None
    _F_NOCACHE = None


class _CentralStore:
    r"""
    The central (``ell``) full-``M`` coefficient set of one precompute,
    built once and kept in a file on disk, for a set that does not fit the
    memory budget (:meth:`_CouplingPrecompute.compute`).

    The file holds a complex128 ``(nfields, lmax, n, ncoef)`` array -- the
    left-operand layout of :func:`_pack_block` -- whose third axis runs over
    ``order``, the kept orders in the paired ``+m, -m`` order the contraction
    iterates in, so that reading a block of consecutive orders is
    ``nfields * lmax`` contiguous reads of ``len(block) * ncoef`` values each,
    straight into the destination.  Plain ``pread``/``pwrite`` rather than a
    ``np.memmap``, so what the process holds is only the blocks it reads, and
    the file is kept out of the OS file cache (``F_NOCACHE`` on macOS,
    ``POSIX_FADV_DONTNEED`` after each fill and read elsewhere): every
    ``ellp`` reads it whole, once per outer block, so caching it buys little
    and it competes with the working set for memory (on a memory-limited
    machine it can trigger compression or swapping).

    ``directory`` is created by the caller (a fresh temporary directory) and
    is removed, with the file, by :meth:`close`.
    """

    FILENAME = "central_full_m.bin"

    def __init__(
        self,
        directory: str,
        order: Sequence[int],
        nfields: int,
        lmax: int,
    ) -> None:
        self.directory = directory
        self.path = os.path.join(directory, self.FILENAME)
        self.order = list(order)
        self.position = {i: p for p, i in enumerate(self.order)}
        self.n = len(self.order)
        self.nfields = int(nfields)
        self.lmax = int(lmax)
        self.ncoef = 2 * self.lmax - 1
        self.itemsize = np.dtype(np.complex128).itemsize
        self.nbytes = self.nfields * self.lmax * self.n * self.ncoef * self.itemsize
        self._fd: int | None = os.open(
            self.path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600
        )
        os.ftruncate(self._fd, self.nbytes)
        if _F_NOCACHE is not None:
            fcntl.fcntl(self._fd, _F_NOCACHE, 1)

    def _drop_cache(self) -> None:
        """Advise the OS to drop the file's cached pages (not macOS, where
        ``F_NOCACHE`` keeps them out in the first place)."""
        if _F_NOCACHE is None and hasattr(os, "posix_fadvise"):
            os.posix_fadvise(self._fd, 0, 0, os.POSIX_FADV_DONTNEED)

    def _offset(self, f: int, L: int, p: int) -> int:
        return ((f * self.lmax + L) * self.n + p) * self.ncoef * self.itemsize

    def _write(self, buf: np.ndarray, offset: int) -> None:
        view = memoryview(buf.reshape(-1).view(np.uint8))
        while len(view):
            done = os.pwrite(self._fd, view, offset)
            view = view[done:]
            offset += done

    def _read_into(self, buf: np.ndarray, offset: int) -> None:
        if not buf.flags.c_contiguous:  # reshape would read into a copy
            raise ValueError("the central store reads into contiguous buffers only")
        view = memoryview(buf.reshape(-1).view(np.uint8))
        while len(view):
            if hasattr(os, "preadv"):
                done = os.preadv(self._fd, [view], offset)
            else:  # pragma: no cover - platforms without preadv
                data = os.pread(self._fd, len(view), offset)
                done = len(data)
                view[:done] = data
            if done == 0:
                raise OSError(f"short read from the central store {self.path}")
            view = view[done:]
            offset += done

    def fill(self, provider: Callable[[int], np.ndarray], chunk: int) -> None:
        """Synthesise every order once, ``chunk`` at a time
        (:func:`_pack_block`, into one buffer reused for every chunk), and
        write it to the file."""
        chunk = max(1, min(int(chunk), self.n))
        buffer = np.empty(
            chunk * self.nfields * self.lmax * self.ncoef, dtype=np.complex128
        )
        for p0 in range(0, self.n, chunk):
            block = _pack_block(
                provider,
                self.order[p0 : p0 + chunk],
                self.lmax,
                conjugate=False,
                nfields=self.nfields,
                out=buffer,
            )
            for f in range(self.nfields):
                for L in range(self.lmax):
                    self._write(block[f, L], self._offset(f, L, p0))
            del block
        if _F_NOCACHE is None and hasattr(os, "posix_fadvise"):
            os.fsync(self._fd)  # dirty pages are not dropped
            self._drop_cache()

    def left(self, rows: Sequence[int], out: np.ndarray | None = None) -> np.ndarray:
        """The ``(nfields, lmax, len(rows), ncoef)`` block of the orders
        ``rows`` (indices ``m + ell``) -- :func:`_pack_block`'s
        ``conjugate=False`` layout -- read from the file, into the leading
        entries of the flat buffer ``out`` when given."""
        positions = [self.position[i] for i in rows]
        out = _block_array(out, self.nfields, self.lmax, len(rows))
        for p0, q0, k in _contiguous_runs(positions):
            for f in range(self.nfields):
                for L in range(self.lmax):
                    self._read_into(out[f, L, q0 : q0 + k], self._offset(f, L, p0))
        self._drop_cache()
        return out

    def right(self, cols: Sequence[int], out: np.ndarray | None = None) -> np.ndarray:
        """The conjugated ``(nfields, lmax, ncoef, len(cols))`` block of the
        orders ``cols`` -- :func:`_pack_block`'s ``conjugate=True`` form, a
        transposed view -- for ``ell == ellp``: :meth:`left`, conjugated in
        place."""
        out = self.left(cols, out)
        rows = out.reshape(self.nfields * self.lmax, -1)  # a view

        def conjugate(r0: int, r1: int) -> None:
            np.conjugate(rows[r0:r1], out=rows[r0:r1])

        run_row_blocks(
            conjugate,
            rows.shape[0],
            row_block_workers(out.size, get_optimal_nthreads()),
        )
        return out.transpose(0, 1, 3, 2)

    def close(self) -> None:
        """Close and delete the file and its directory (idempotent)."""
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        shutil.rmtree(self.directory, ignore_errors=True)
