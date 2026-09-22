"""
``OMP_NUM_THREADS`` handling agrees between ``threading_utils`` and ``grid``.

Returning ``int(os.environ["OMP_NUM_THREADS"])`` unclamped would give a
``"0"``/``"-4"`` value 0 or negative thread counts, so ``get_optimal_nthreads``
and ``grid.py``'s ``_nthreads`` both clamp with ``max(1, int(env))`` by
sharing one policy,
:func:`cmbcov.utils.threading_utils._env_nthreads`.
"""

import warnings

import pytest

from cmbcov.grid import _nthreads
from cmbcov.utils.threading_utils import (
    get_optimal_nthreads,
)


@pytest.mark.parametrize("value", ["0", "-4"])
def test_get_optimal_nthreads_clamps_to_one(monkeypatch, value):
    monkeypatch.setenv("OMP_NUM_THREADS", value)
    assert get_optimal_nthreads() == 1


@pytest.mark.parametrize("value", ["0", "-4"])
def test_grid_nthreads_clamps_to_one(monkeypatch, value):
    monkeypatch.setenv("OMP_NUM_THREADS", value)
    assert _nthreads(None) == 1
    assert _nthreads(None, lmax_grid=32) == 1


def test_get_optimal_nthreads_falls_back_on_garbage(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "garbage")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = get_optimal_nthreads()
    assert result >= 1


def test_grid_nthreads_falls_back_on_garbage(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "garbage")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = _nthreads(None, lmax_grid=32)
    assert result >= 1


def test_get_optimal_nthreads_honours_valid_value(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    assert get_optimal_nthreads() == 3
    assert get_optimal_nthreads(nside=1024) == 3


def test_grid_nthreads_honours_valid_value(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    assert _nthreads(None) == 3
    assert _nthreads(None, lmax_grid=4096) == 3


def test_explicit_nthreads_still_wins_over_env(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    assert _nthreads(7, lmax_grid=4096) == 7
