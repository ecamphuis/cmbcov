"""
The progress helper (:mod:`cmbcov.utils.progress`) with a fake clock, and the
visibility switch of :func:`cmbcov.utils.configure_logging`.
"""

import logging
import threading

import numpy as np
import pytest

from cmbcov.utils import configure_logging
from cmbcov.utils.progress import PROGRESS_LOGGER, Progress, format_duration


class FakeClock:
    """A monotonic clock the test moves by hand."""

    def __init__(self, start=1000.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def lines(caplog):
    """The messages logged on ``cmbcov.progress`` so far."""
    caplog.set_level(logging.INFO, logger=PROGRESS_LOGGER)

    def get():
        return [r.getMessage() for r in caplog.records if r.name == PROGRESS_LOGGER]

    return get


@pytest.mark.parametrize(
    "seconds, text",
    [
        (0, "0s"),
        (0.4, "0s"),
        (35, "35s"),
        (59.6, "1m00s"),
        (252, "4m12s"),
        (3599, "59m59s"),
        (3912, "1h05m12s"),
        (-5, "0s"),
    ],
)
def test_format_duration(seconds, text):
    assert format_duration(seconds) == text


def test_ten_percent_steps_and_final_line(lines):
    clock = FakeClock()
    with Progress("stage", 10, "item", clock=clock) as progress:
        for _ in range(10):
            clock.advance(10.0)  # fast enough for no 30 s line in between
            progress.update()
    out = lines()
    # one line per 10% step before the end, then the final line
    assert out[:2] == [
        "stage: 1/10 items (10%), 10s elapsed, ~1m30s left",
        "stage: 2/10 items (20%), 20s elapsed, ~1m20s left",
    ]
    assert len(out) == 10
    assert out[8] == "stage: 9/10 items (90%), 1m30s elapsed, ~10s left"
    assert out[-1] == "stage: 10/10 items (100%), done in 1m40s"


def test_format_of_the_example_line(lines):
    clock = FakeClock()
    with Progress("ACC precompute", 36, "l'", plural="l' values", clock=clock) as p:
        clock.advance(252.0)
        p.update(12)
    assert lines()[0] == (
        "ACC precompute: 12/36 l' values (33%), 4m12s elapsed, ~8m24s left"
    )


def test_rate_limit_between_steps(lines):
    """Within one 10% step a line needs ``interval`` seconds since the last."""
    clock = FakeClock()
    progress = Progress("stage", 1000, "item", interval=30.0, clock=clock)
    for _ in range(50):  # 5%: no step reached, and no time passes
        progress.update()
    assert lines() == []
    clock.advance(29.0)
    progress.update()
    assert lines() == []
    clock.advance(2.0)  # 31 s since the start: a time line, at 5.1%
    progress.update()
    assert len(lines()) == 1
    assert "(5%)" in lines()[0]
    for _ in range(10):  # 31 s... no new line: 0 s since the last one
        progress.update()
    assert len(lines()) == 1
    clock.advance(31.0)
    progress.update()
    assert len(lines()) == 2
    progress.close()


def test_a_slow_item_logs_each_time_interval(lines):
    clock = FakeClock()
    with Progress("stage", 3, "item", interval=30.0, clock=clock) as progress:
        clock.advance(100.0)
        progress.update()  # 33%: a step
        clock.advance(10.0)
        progress.update()  # 66%: a step again (6 > 3), whatever the time
    assert len(lines()) == 3  # two intermediate lines and the final one


def test_final_line_and_close_is_idempotent(lines):
    clock = FakeClock()
    progress = Progress("stage", 2, "block", clock=clock)
    clock.advance(5.0)
    progress.update(2)
    assert lines() == []  # the 100% line is close()'s
    progress.close()
    progress.close()
    assert lines() == ["stage: 2/2 blocks (100%), done in 5s"]
    progress.update()  # after close: counted, never logged
    assert len(lines()) == 1


def test_stopped_stage_is_reported(lines):
    clock = FakeClock()
    with pytest.raises(RuntimeError):
        with Progress("stage", 5, "row", interval=1e9, clock=clock) as progress:
            clock.advance(3.0)
            progress.update()
            raise RuntimeError("boom")
    assert lines()[-1] == "stage: stopped at 1/5 rows after 3s"


def test_empty_stage_logs_nothing(lines):
    with Progress("stage", 0, "row") as progress:
        progress.update()
    assert lines() == []


def test_eta_is_linear_in_the_items_done(lines):
    clock = FakeClock()
    progress = Progress("stage", 100, "row", clock=clock)
    clock.advance(60.0)
    progress.update(25)
    assert lines()[0] == "stage: 25/100 rows (25%), 1m00s elapsed, ~3m00s left"
    progress.close()


def test_a_single_item_uses_the_singular(lines):
    with Progress("stage", 1, "node", clock=FakeClock()) as progress:
        progress.update()
    assert lines() == ["stage: 1/1 node (100%), done in 0s"]


def test_plural_defaults_to_s_and_can_be_given(lines):
    clock = FakeClock()
    with Progress("a", 2, "block", clock=clock) as a:
        a.update(2)
    with Progress("b", 2, "l'", plural="l' values", clock=clock) as b:
        b.update(2)
    assert lines() == [
        "a: 2/2 blocks (100%), done in 0s",
        "b: 2/2 l' values (100%), done in 0s",
    ]


# --------------------------------------------------------------------------- #
# weights: the percentage and the time left follow the expected work
# --------------------------------------------------------------------------- #
def test_weighted_percentage_and_eta(lines):
    """Two cheap items then one that costs eight times as much: after the
    two, 20% of the work is done, so the work left takes 4x the time spent."""
    clock = FakeClock()
    progress = Progress("stage", 3, "item", weights=[1, 1, 8], clock=clock)
    clock.advance(10.0)
    progress.update(2)
    assert lines() == ["stage: 2/3 items (20%), 10s elapsed, ~40s left"]
    clock.advance(40.0)
    progress.update()
    progress.close()
    assert lines()[-1] == "stage: 3/3 items (100%), done in 50s"


def test_weighted_steps_follow_the_work(lines):
    """A 10% step is a tenth of the weight, not of the count: the heavy last
    item gives no line of its own before the final one, and the first of
    ten equal-looking items passes a step only once it has weighed enough."""
    clock = FakeClock()
    progress = Progress(
        "stage", 4, "item", weights=[1, 1, 1, 97], interval=1e9, clock=clock
    )
    clock.advance(1.0)
    progress.update()  # 1% of the work: no new step
    assert lines() == []
    progress.update()  # 2%
    progress.update()  # 3%
    assert lines() == []
    clock.advance(97.0)
    progress.update()  # done
    progress.close()
    assert lines() == ["stage: 4/4 items (100%), done in 1m38s"]


def test_unweighted_is_the_equal_weight_special_case(lines):
    clock = FakeClock()
    plain = Progress("p", 10, "item", clock=clock)
    equal = Progress("p", 10, "item", weights=[3.0] * 10, clock=clock)
    clock.advance(7.0)
    plain.update(4)
    equal.update(4)
    a, b = lines()
    assert a == b == "p: 4/10 items (40%), 7s elapsed, ~10s left"


def test_update_by_item_counts_the_weight_of_that_item(lines):
    """Out-of-order completion (the processes of the exact rows)."""
    clock = FakeClock()
    progress = Progress("stage", 4, "row", weights=[1, 1, 1, 7], clock=clock)
    clock.advance(10.0)
    progress.update(item=3)  # the heavy one finished first
    assert lines() == ["stage: 1/4 rows (70%), 10s elapsed, ~4s left"]
    assert progress.done == 1
    progress.close()


def test_weighted_update_in_order_sums_the_next_weights():
    progress = Progress("stage", 4, weights=[1, 2, 3, 4], clock=FakeClock())
    progress.update(2)
    assert progress.done_weight == 3
    progress.update()
    assert progress.done_weight == 6
    assert progress.fraction == pytest.approx(0.6)


def test_zero_total_weight_falls_back_to_equal_items(lines):
    clock = FakeClock()
    progress = Progress("stage", 4, "item", weights=[0, 0, 0, 0], clock=clock)
    clock.advance(8.0)
    progress.update(2)
    assert lines() == ["stage: 2/4 items (50%), 8s elapsed, ~8s left"]


@pytest.mark.parametrize(
    "weights", [[1, 2], [1, 2, 3, 4], [1, -1, 1], [1, np.nan, 1], [1, np.inf, 1]]
)
def test_bad_weights_are_refused(weights):
    with pytest.raises(ValueError, match="weights"):
        Progress("stage", 3, weights=weights)


def test_threads_count_every_update(lines):
    progress = Progress("stage", 4000, "row")

    def work():
        for _ in range(1000):
            progress.update()

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert progress.done == 4000
    progress.close()
    assert lines()[-1].startswith("stage: 4000/4000 rows (100%), done in ")


def test_nothing_is_formatted_when_the_logger_is_off(caplog):
    caplog.set_level(logging.WARNING, logger=PROGRESS_LOGGER)
    clock = FakeClock()
    with Progress("stage", 4, "row", clock=clock) as progress:
        for _ in range(4):
            clock.advance(100.0)
            progress.update()
    assert [r for r in caplog.records if r.name == PROGRESS_LOGGER] == []


# --------------------------------------------------------------------------- #
# configure_logging
# --------------------------------------------------------------------------- #
@pytest.fixture
def bare_logging():
    """Restores the root logger and the progress logger after the test."""
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    progress_level = logging.getLogger(PROGRESS_LOGGER).level
    yield
    for handler in list(root.handlers):
        root.removeHandler(handler)
    for handler in handlers:
        root.addHandler(handler)
    root.setLevel(level)
    logging.getLogger(PROGRESS_LOGGER).setLevel(progress_level)


@pytest.mark.parametrize("verbose", [False, True])
def test_progress_is_shown_by_default(bare_logging, verbose):
    configure_logging(verbose)
    assert logging.getLogger(PROGRESS_LOGGER).isEnabledFor(logging.INFO)


@pytest.mark.parametrize("verbose", [False, True])
def test_quiet_hides_progress(bare_logging, verbose):
    configure_logging(verbose, quiet=True)
    progress_logger = logging.getLogger(PROGRESS_LOGGER)
    assert not progress_logger.isEnabledFor(logging.INFO)
    assert progress_logger.isEnabledFor(logging.WARNING)


def test_progress_reaches_a_script_handler(bare_logging, capsys):
    """Without --verbose the root is at WARNING, yet progress lines are shown."""
    # basicConfig does nothing when the root has handlers (pytest's own):
    # start from a bare root, as in a script.
    for handler in list(logging.getLogger().handlers):
        logging.getLogger().removeHandler(handler)
    configure_logging(False)
    clock = FakeClock()
    with Progress("stage", 1, "row", clock=clock) as progress:
        clock.advance(2.0)
        progress.update()
    assert "stage: 1/1 row (100%), done in 2s" in capsys.readouterr().err
    configure_logging(False, quiet=True)
    with Progress("stage", 1, "row", clock=clock) as progress:
        progress.update()
    assert capsys.readouterr().err == ""
