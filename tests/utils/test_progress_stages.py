"""
Every long stage reports its progress on the ``cmbcov.progress`` logger.

One small run per hooked stage, asserting that the final line (the item count
and the total time) is logged, and that the numbers are unchanged by the
reporting: the same call with the logger off gives the same result.
"""

import logging
import os
import shutil

import numpy as np
import pytest
import yaml
from conftest import DATA_DIR  # noqa: E402

from cmbcov import CovarianceMatrixGenerator
from cmbcov.approximations.acc.precompute import precompute_acc_kernels
from cmbcov.exact.covariance import exact_covariance, exact_covariance_pol
from cmbcov.harmonics.sht import ducc0_map2alm
from cmbcov.scripts import compute_covariance
from cmbcov.utils.progress import PROGRESS_LOGGER, Progress

healpy = pytest.importorskip("healpy")

MASK = "baseline_mask.fits"
FREQ = "090GHz"


@pytest.fixture
def weights_seen(monkeypatch):
    """``{stage: (total, weights)}`` of every :class:`Progress` built from here
    on (the weights are ``None`` for a stage that passes none)."""
    seen = {}
    original = Progress.__init__

    def spy(self, stage, total, *args, **kwargs):
        original(self, stage, total, *args, **kwargs)
        seen[stage] = (total, kwargs.get("weights"))

    monkeypatch.setattr(Progress, "__init__", spy)
    return seen


@pytest.fixture
def lines(caplog):
    """The progress lines logged so far."""
    caplog.set_level(logging.INFO, logger=PROGRESS_LOGGER)

    def get(stage=""):
        return [
            r.getMessage()
            for r in caplog.records
            if r.name == PROGRESS_LOGGER and r.getMessage().startswith(stage)
        ]

    return get


def _final(out, stage, count, unit):
    """The one final line of ``stage``: ``count`` items of ``unit``, done."""
    final = [m for m in out if m.startswith(f"{stage}: {count}/{count} {unit} (100%)")]
    assert len(final) == 1, out
    assert ", done in " in final[0]


# --------------------------------------------------------------------------- #
# ACC precompute
# --------------------------------------------------------------------------- #
def test_acc_precompute_reports_each_ellp(tmp_path, lines):
    precompute_acc_kernels(
        MASK, str(tmp_path), centralell=16, dmax=3, mask_path=DATA_DIR, nside=16
    )
    _final(lines("ACC precompute"), "ACC precompute", 3, "l' values")


def test_acc_precompute_passes_one_weight_per_ellp(tmp_path, weights_seen):
    precompute_acc_kernels(
        MASK, str(tmp_path), centralell=16, dmax=3, mask_path=DATA_DIR, nside=16
    )
    total, weights = weights_seen["ACC precompute"]
    assert total == 3 and len(weights) == 3
    assert list(weights) == [17.0, 18.0, 19.0]  # ellp + 1


# --------------------------------------------------------------------------- #
# exact covariance rows, serial and with processes
# --------------------------------------------------------------------------- #
LW, LMAX = 48, 16


def _cl(lmax):
    return {
        "TT": 1000.0 / (np.arange(lmax + 1) + 1.0) ** 2,
        "EE": 10.0 / (np.arange(lmax + 1) + 1.0) ** 2,
        "TE": 3.0 / (np.arange(lmax + 1) + 1.0) ** 2,
    }


@pytest.fixture(scope="module")
def mask_alm():
    mask = healpy.read_map(os.path.join(DATA_DIR, MASK))
    return ducc0_map2alm(mask, lmax=LW, iter=10)


@pytest.mark.parametrize("nprocs", [1, 2])
def test_exact_covariance_reports_each_row(mask_alm, lines, nprocs, caplog):
    rows = (3, 9, 16)
    cl = _cl(LMAX)["TT"]
    out = exact_covariance(
        mask_alm, cl, LMAX, rows=rows, grid="gl", nthreads=1, nprocs=nprocs
    )
    _final(lines("exact covariance"), "exact covariance", 3, "rows")
    # the reporting changes nothing
    caplog.set_level(logging.WARNING, logger=PROGRESS_LOGGER)
    quiet = exact_covariance(
        mask_alm, cl, LMAX, rows=rows, grid="gl", nthreads=1, nprocs=nprocs
    )
    np.testing.assert_array_equal(out, quiet)


@pytest.mark.parametrize("nprocs", [1, 2])
def test_exact_covariance_pol_reports_each_row(mask_alm, lines, nprocs):
    exact_covariance_pol(
        mask_alm,
        _cl(LMAX),
        LMAX,
        spectra=("TT", "EE", "TE"),
        rows=(2, 9),
        nthreads=1,
        nprocs=nprocs,
    )
    _final(
        lines("exact covariance (polarised)"),
        "exact covariance (polarised)",
        2,
        "rows",
    )


@pytest.mark.parametrize("nprocs", [1, 2])
def test_exact_rows_pass_weights_growing_with_ellp(mask_alm, weights_seen, nprocs):
    rows = (3, 9, 16)
    exact_covariance(
        mask_alm, _cl(LMAX)["TT"], LMAX, rows=rows, grid="gl", nthreads=1, nprocs=nprocs
    )
    exact_covariance_pol(
        mask_alm,
        _cl(LMAX),
        LMAX,
        spectra=("TT", "EE", "TE"),
        rows=rows,
        nthreads=1,
        nprocs=nprocs,
    )
    for stage in ("exact covariance", "exact covariance (polarised)"):
        total, weights = weights_seen[stage]
        assert total == 3 and len(weights) == 3
        # the parallel stage counts in the order the rows are submitted (the
        # largest first); the serial one in the order given
        assert len(set(weights)) == 3
        assert (weights[0] > weights[-1]) == (nprocs > 1)


def test_row_weights_are_the_number_of_orders():
    from cmbcov.exact.common import row_weights

    assert row_weights([0, 9]) == [1.0, 10.0]
    assert row_weights([4], use_symmetry=False) == [9.0]


def test_the_parallel_rows_come_back_in_the_old_order(mask_alm):
    """``pool.map`` returned the rows most expensive first; so does the
    submit-and-count loop."""
    from cmbcov.exact.parallel import _rows_gl_parallel

    cl = np.asarray(_cl(LMAX)["TT"])
    result = _rows_gl_parallel(
        mask_alm, LW, cl, [3, 16, 9], LMAX, 56, True, 1, 2, lmax_int=None
    )
    assert [ellp for ellp, _ in result] == [16, 9, 3]


# --------------------------------------------------------------------------- #
# the generator: map_filter probing and covariance assembly
# --------------------------------------------------------------------------- #
def _params(workdir, **block):
    workdir = str(workdir)
    shutil.copy(os.path.join(DATA_DIR, MASK), workdir)
    ell = np.arange(80)
    tt = np.zeros(80)
    tt[2:] = 1e3 / (ell[2:] * (ell[2:] + 1)) ** 0.9
    columns = [tt, 0.05 * tt, 0.001 * tt, 0.1 * tt]
    np.savetxt(os.path.join(workdir, "cls.dat"), np.column_stack([ell, *columns]))
    params = {
        "cov_path": os.path.join(workdir, "out"),
        "cov_name": "cov.dat",
        "frequencies": [FREQ],
        "stokes": ["T", "E"],
        "lmax": 20,
        "lmin": 2,
        "bins": [[2, 20, 3]],
        "mask_name": MASK,
        "mask_path": workdir,
        "covariance_approximation": "nka",
        "cmb_spectrum": os.path.join(workdir, "cls.dat"),
        "beams": {FREQ: 5.0},
        "pixwin": 32,
        "nl": {FREQ: 10.0},
        "polspice_postprocess": False,
        "map_filter": {
            "type": "fourier_highpass",
            "lx": 12,
            "nprobe": 5,
            "seed": 4,
            **block,
        },
    }
    path = os.path.join(workdir, "params.yml")
    with open(path, "w") as handle:
        yaml.dump(params, handle)
    return path


def test_map_filter_probing_and_assembly_report(tmp_path, lines):
    path = _params(tmp_path, spectrum_aware=True)
    generator = CovarianceMatrixGenerator(path)
    generator.run_full_analysis()
    nodes = len(generator._map_filter_ratio["nodes"])
    out = lines()
    _final(out, "map_filter probing (white channels)", nodes, "nodes")
    _final(out, "map_filter probing (spectrum-aware TT)", nodes, "nodes")
    # T and E, one frequency: TT, TE, ET and EE blocks
    n_blocks = len(generator.cov_keys.keys())
    _final(out, "covariance assembly", n_blocks, "blocks")


def test_probing_passes_one_weight_per_node(tmp_path, weights_seen):
    path = _params(tmp_path, spectrum_aware=True)
    generator = CovarianceMatrixGenerator(path)
    generator.run_full_analysis()
    nodes = len(generator._map_filter_ratio["nodes"])
    for stage in (
        "map_filter probing (white channels)",
        "map_filter probing (spectrum-aware TT)",
    ):
        total, weights = weights_seen[stage]
        assert total == nodes and len(weights) == nodes
        assert all(w > 0 for w in weights)
        assert len(set(weights)) > 1  # the nodes are not alike


def test_assembly_report_does_not_change_the_covariance(tmp_path, caplog):
    results = []
    for level in (logging.INFO, logging.WARNING):
        caplog.set_level(level, logger=PROGRESS_LOGGER)
        workdir = tmp_path / str(level)
        workdir.mkdir()
        generator = CovarianceMatrixGenerator(_params(workdir))
        generator.run_full_analysis()
        results.append(np.loadtxt(os.path.join(generator.config.save_dir, "cov.dat")))
    np.testing.assert_array_equal(results[0], results[1])


# --------------------------------------------------------------------------- #
# the command-line scripts
# --------------------------------------------------------------------------- #
def _plain_params(workdir):
    path = _params(workdir)
    params = yaml.safe_load(open(path))
    del params["map_filter"]
    with open(path, "w") as handle:
        yaml.dump(params, handle)
    return path


def test_cov_script_shows_progress_by_default_and_quiet_hides_it(tmp_path, caplog):
    for name, flags, shown in (("default", [], True), ("quiet", ["--quiet"], False)):
        workdir = tmp_path / name
        workdir.mkdir()
        caplog.clear()
        assert compute_covariance.main([_plain_params(workdir), *flags]) == 0
        got = [r for r in caplog.records if r.name == PROGRESS_LOGGER]
        assert bool(got) is shown, (name, [r.getMessage() for r in got])
    # leave the logger as the library has it
    logging.getLogger(PROGRESS_LOGGER).setLevel(logging.NOTSET)


def test_acc_b_mode_run_reports_the_second_pass(tmp_path, lines):
    """An ACC run with a B observable under ``polspice_postprocess`` keeps the
    raw blocks for a second pass, which is reported as its own stage; the
    kernels are precomputed through the generator on the way."""
    shutil.copy(os.path.join(DATA_DIR, MASK), tmp_path)
    ell = np.arange(80)
    tt = np.zeros(80)
    tt[2:] = 1e3 / (ell[2:] * (ell[2:] + 1)) ** 0.9
    ee, bb = 0.1 * tt, 0.02 * tt
    te = 0.5 * np.sqrt(tt * ee)
    zero = 0 * tt
    np.savetxt(
        tmp_path / "cls.dat",
        np.column_stack([ell, tt, ee, bb, te, zero, zero, te, zero, zero]),
    )
    params = {
        "cov_path": str(tmp_path / "out"),
        "cov_name": "cov.dat",
        "frequencies": [FREQ],
        "observables": ["TT", "EE", "TE", "BB"],
        "lmax": 20,
        "lmin": 2,
        "bins": [[2, 20, 3]],
        "mask_name": MASK,
        "mask_path": str(tmp_path),
        "covariance_approximation": "acc",
        "dmax": 3,
        "centralell": 10,
        "cmb_spectrum": str(tmp_path / "cls.dat"),
        "beams": {FREQ: 5.0},
        "pixwin": 32,
        "nl": {FREQ: 10.0},
        "polspice_postprocess": True,
        "acc_precompute": {"nside": 16, "grid": "gl", "lw": 10},
    }
    path = tmp_path / "params.yml"
    path.write_text(yaml.dump(params))
    CovarianceMatrixGenerator(str(path)).precompute_acc_kernels()
    generator = CovarianceMatrixGenerator(str(path))
    generator.run_full_analysis()
    n_blocks = len(generator.cov_keys.keys())
    out = lines()
    _final(out, "ACC precompute", 3, "l' values")
    _final(out, "covariance assembly", n_blocks, "blocks")
    _final(out, "covariance post-processing", n_blocks, "blocks")
