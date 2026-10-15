"""
The probing-node cost model behind the weights of the progress lines
(:mod:`cmbcov.assembly.filtering`). It only weights the progress estimate;
nothing computed depends on it.
"""

import numpy as np
import pytest

from cmbcov.assembly import filtering


def test_white_probe_work_counts_the_transforms_of_node_sums():
    # "00": seven spin-0 transforms and two filters
    assert filtering._white_probe_work(("00",), 40) == (2, 7)
    # a spin-2 channel at l' < 2 has no probe: nothing to count
    assert filtering._white_probe_work(("20", "EE"), 1) == (0, 0)
    f, t = filtering._white_probe_work(("00", "20", "EE"), 40)
    assert f == 7 and t == pytest.approx(7 + 2.5 * 3 + 4 + 2.5 * 4)


def test_spectra_probe_work_one_pair_is_the_white_channel():
    assert filtering._spectra_probe_work(1) == filtering._white_probe_work(("00",), 9)
    assert filtering._spectra_probe_work(3) == (4, 15)


def test_node_weights_cap_the_probes_at_the_complete_basis(monkeypatch):
    """A node with 2 l' + 1 <= nprobe uses the complete basis (fewer probes);
    the unit costs are measured, here replaced by simple ones."""
    monkeypatch.setattr(filtering, "_filter_seconds", lambda lg: 1e-6 * lg**2)
    monkeypatch.setattr(
        filtering, "_transform_seconds", lambda lmax, lg, nt: 1e-9 * lmax**2 * lg
    )
    geometry = [(180, 0, 359), (180, 0, 359), (360, 0, 719)]
    work = lambda ellp: (2.0, 7.0)  # noqa: E731
    w = filtering._node_weights([2, 30, 30], [16, 16, 16], geometry, work)
    assert w[0] / w[1] == pytest.approx(5 / 16)  # 5 probes of the basis vs 16
    assert w[2] > 4 * w[1]  # the grid is twice as big


def test_node_weights_with_the_measured_unit_costs():
    """The real timings: positive, finite, and growing with the grid."""
    geometry = [(40, 0, 80), (120, 0, 240)]
    w = filtering._node_weights([20, 20], [8, 8], geometry, lambda ellp: (2.0, 7.0))
    assert np.all(np.isfinite(w)) and w[0] > 0
    assert w[1] > w[0]
