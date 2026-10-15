"""
Progress reports for the long computations.

:class:`Progress` counts the finished items of one stage (a multipole of the
ACC precompute, a covariance block, a row of the exact covariance, ...) and
logs plain lines to the ``cmbcov.progress`` logger at ``INFO``::

    ACC precompute: 12/36 l' values (33%), 4m12s elapsed, ~8m20s left

The library never configures logging. The command-line scripts show these
lines by default and hide them with ``--quiet``
(:func:`~cmbcov.utils.logging_utils.configure_logging`); a Python user sees
them after ``configure_logging(...)`` or by enabling the logger::

    logging.getLogger("cmbcov.progress").setLevel(logging.INFO)

The percentage and the time left follow the expected *work*, not the item
count: a stage whose items differ in cost passes a weight per item (its
expected relative cost, from the cost model next to the loop), see
:class:`Progress`. Without weights the items count alike.

The lines are plain text (no terminal progress bars), so they read well in
cluster job logs.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Sequence
from types import TracebackType

import numpy as np

__all__ = ["Progress", "format_duration", "PROGRESS_LOGGER"]

#: Name of the logger the progress lines go to.
PROGRESS_LOGGER = "cmbcov.progress"

_logger = logging.getLogger(PROGRESS_LOGGER)

#: Seconds between two progress lines, at most (a new 10% step is reported
#: whatever the time since the last line).
DEFAULT_INTERVAL = 30.0


def format_duration(seconds: float) -> str:
    """
    A duration for a progress line: ``"35s"``, ``"4m12s"`` or ``"1h05m12s"``
    (rounded to whole seconds, never negative).
    """
    total = max(0, int(round(float(seconds))))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


class Progress:
    """
    Progress of one stage of ``total`` items, optionally of unequal cost.

    Call :meth:`update` once per finished item and :meth:`close` at the end
    (or use the instance as a context manager). A line is logged when a new
    10% step is reached, or when :data:`DEFAULT_INTERVAL` seconds (``interval``)
    have passed since the last line; :meth:`close` logs a final line with the
    total time. The 100% step itself is left to that final line.

    **Percentage and time left.** Each item has a weight, its expected
    relative cost (all equal without ``weights``). With ``W`` the total
    weight and ``w`` the weight done, the line shows ``100 w / W`` per cent,
    and the time left is the elapsed time scaled by the work left:
    ``elapsed * (W - w) / w``. The estimate is therefore adaptive (the
    seconds per unit of weight are measured on the items already done) and
    the weights only need to be right *relative to each other*; the cost
    models that supply them are in the stage's own module (the probing nodes
    of the map filter, the rows of the exact covariance). It remains an
    estimate: the items of the next stretch can cost more or less than the
    model says, and a stage that stalls (swap, a busy node) stretches it.
    The item count shown (``17/34 nodes``) is the plain count.

    The counter is guarded by a lock, so threads (or the parent process
    counting completions of its workers) may call :meth:`update` freely. The
    cost of a call without a line is a lock and one clock read.

    Parameters
    ----------
    stage : str
        Name at the start of every line, e.g. ``"ACC precompute"``.
    total : int
        Number of items. With ``total <= 0`` nothing is ever logged.
    unit : str, default "item"
        What an item is, singular, e.g. ``"block"``.
    plural : str, optional
        The plural of ``unit`` (``unit + "s"`` by default), e.g.
        ``"l' values"``. A stage of one item uses ``unit``.
    weights : sequence of float, optional
        Expected relative cost of each item, in the order the items are
        counted: ``total`` finite non-negative numbers. Equal items (the
        default) when omitted, or when they sum to zero. :meth:`update`
        counts items in this order unless it is told which item finished.
    interval : float, default 30
        Minimum seconds between two lines that are not 10% steps.
    clock : callable, optional
        A monotonic clock returning seconds (``time.monotonic``); injectable
        for tests.
    logger : logging.Logger, optional
        Where the lines go (the ``cmbcov.progress`` logger).
    """

    def __init__(
        self,
        stage: str,
        total: int,
        unit: str = "item",
        *,
        plural: str | None = None,
        weights: Sequence[float] | None = None,
        interval: float = DEFAULT_INTERVAL,
        clock: Callable[[], float] = time.monotonic,
        logger: logging.Logger | None = None,
    ) -> None:
        self.stage = stage
        self.total = int(total)
        self.unit = unit
        self.plural = unit + "s" if plural is None else plural
        self.interval = float(interval)
        self._weights: np.ndarray | None = None
        self._total_weight = float(max(self.total, 0))
        if weights is not None:
            w = np.asarray(weights, dtype=float)
            if w.shape != (max(self.total, 0),):
                raise ValueError(
                    f"weights must have one entry per item ({self.total}), "
                    f"got shape {w.shape}"
                )
            if not np.all(np.isfinite(w)) or np.any(w < 0):
                raise ValueError("weights must be finite and non-negative")
            if w.sum() > 0:
                self._weights = w
                self._total_weight = float(w.sum())
        self._clock = clock
        self._logger = _logger if logger is None else logger
        self._lock = threading.Lock()
        self._start = clock()
        self._last_line = self._start
        self._step = 0  # the 10% steps (0..10) already reported
        self.done = 0
        self.done_weight = 0.0
        self._closed = False

    @property
    def fraction(self) -> float:
        """Share of the total weight done, in ``[0, 1]``."""
        if self._total_weight <= 0:
            return 0.0
        return min(1.0, self.done_weight / self._total_weight)

    def _label(self) -> str:
        return self.unit if self.total == 1 else self.plural

    # ------------------------------------------------------------------
    def update(self, n: int = 1, *, item: int | None = None) -> None:
        """
        Count ``n`` more finished items, logging a line when one is due.

        With weights the items are counted in order (the next ``n`` weights
        are done), or, for stages that finish out of order (the workers of
        the exact rows), ``item`` names the index of the one item that
        finished.
        """
        with self._lock:
            n = int(n)
            if self._weights is None:
                self.done_weight += n
            elif item is not None:
                self.done_weight += self._weights[int(item)]
            else:
                self.done_weight += self._weights[self.done : self.done + n].sum()
            self.done += n
            if self._closed or self.total <= 0:
                return
            if self.done >= self.total:
                return  # the final line is close()'s
            now = self._clock()
            step = int(10 * self.fraction + 1e-9)
            if step <= self._step and now - self._last_line < self.interval:
                return
            self._step = max(self._step, step)
            self._last_line = now
            self._log(now)

    def close(self) -> None:
        """
        Log the final line (the total time) and stop; calling it again does
        nothing. A stage left before its last item (an exception) is reported
        as stopped, with the count it reached.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self.total <= 0 or not self._logger.isEnabledFor(logging.INFO):
                return
            elapsed = format_duration(self._clock() - self._start)
            if self.done >= self.total:
                self._logger.info(
                    "%s: %d/%d %s (100%%), done in %s",
                    self.stage,
                    self.total,
                    self.total,
                    self._label(),
                    elapsed,
                )
            else:
                self._logger.info(
                    "%s: stopped at %d/%d %s after %s",
                    self.stage,
                    self.done,
                    self.total,
                    self._label(),
                    elapsed,
                )

    def __enter__(self) -> Progress:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # ------------------------------------------------------------------
    def _log(self, now: float) -> None:
        """One intermediate line (called with the lock held)."""
        if not self._logger.isEnabledFor(logging.INFO):
            return
        elapsed = now - self._start
        fraction = self.fraction
        percent = min(99, int(100 * fraction + 1e-9))
        if fraction > 0 and elapsed > 0:
            left_s = elapsed * (1.0 - fraction) / fraction
            left = f", ~{format_duration(left_s)} left"
        else:
            left = ""
        self._logger.info(
            "%s: %d/%d %s (%d%%), %s elapsed%s",
            self.stage,
            self.done,
            self.total,
            self._label(),
            percent,
            format_duration(elapsed),
            left,
        )
