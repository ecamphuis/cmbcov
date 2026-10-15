"""
``params_resolved.yml``: the record of what a run actually used.

It sits beside ``params.yml`` in the version directory, holds every parameter
with its effective value (the defaults the file did not set included), a
``derived`` section of values the run computed and a ``provenance`` section,
and leaves ``params.yml`` -- which the next run compares itself with -- and
that comparison untouched.
"""

import datetime
import glob
import os
import shutil
import socket

import numpy as np
import pytest
import yaml
from conftest import DATA_DIR  # noqa: E402

from cmbcov import CovarianceMatrixGenerator
from cmbcov.assembly.covariance import Cov
from cmbcov.config.run_record import (
    MAX_ITEMS,
    PRECOMPUTE_RESOLVED_NAME,
    RESOLVED_NAME,
    RunRecord,
    resolved_parameters,
    yaml_safe,
)
from cmbcov.config.validation import ParameterManager

DATA = DATA_DIR
MASK = "baseline_mask.fits"
FREQ = "090GHz"


def _baseline_params(tmp_path, extra=""):
    """The parameter file of the end-to-end baseline run, in ``tmp_path``."""
    template = open(os.path.join(DATA, "baseline_params.yml")).read()
    text = template.replace("PLACEHOLDER_OUT", str(tmp_path)).replace(
        "PLACEHOLDER_DATA", os.path.abspath(DATA)
    )
    path = tmp_path / "params.yml"
    path.write_text(text + extra)
    return str(path)


def _version_dir(out):
    return sorted(glob.glob(os.path.join(out, "v*")))[-1]


def _record(save_dir):
    with open(os.path.join(save_dir, RESOLVED_NAME)) as handle:
        return yaml.safe_load(handle)


@pytest.fixture(scope="module")
def baseline_run(tmp_path_factory):
    """One small end-to-end run (the baseline of test_end_to_end_baseline)."""
    tmp_path = tmp_path_factory.mktemp("record")
    path = _baseline_params(tmp_path)
    generator = CovarianceMatrixGenerator(path)
    generator.run_full_analysis()
    return generator, path


# --------------------------------------------------------------------------- #
# The file of a run
# --------------------------------------------------------------------------- #
def test_the_record_is_written_and_loads_back(baseline_run):
    generator, _ = baseline_run
    save_dir = generator.config.save_dir
    assert os.path.exists(os.path.join(save_dir, RESOLVED_NAME))
    record = _record(save_dir)
    assert record["covariance_approximation"] == "nka"
    assert record["lmax"] == 60
    assert record["save_dir"] == save_dir
    assert record["git_hash"] == generator.config.git_hash


def test_defaults_the_file_did_not_set_are_there(baseline_run):
    generator, path = baseline_run
    asked = yaml.safe_load(open(path))
    record = _record(generator.config.save_dir)
    for key, default in (
        ("polspice_postprocess", True),
        ("save_raw_blocks", False),
        ("parity_mixed_blocks", False),
        ("add_tf_uncertainty", False),
        ("inpainting_rescaling", 1.0),
        ("spectrum_units", "Cl"),
        ("sum_assymetric_stokes", False),
    ):
        assert key not in asked
        assert record[key] == default, key
    # dataclass defaults that are no OPTIONAL_PARAMS key: present, null
    for key in ("dmax", "centralell", "post_process_correction", "fl", "hl"):
        assert key in record and record[key] is None, key
    assert record["observables"] == ["EE", "TE", "TT"]


def test_every_parameter_of_params_yml_is_there_unchanged(baseline_run):
    """The two files share their keys, so they can be diffed."""
    generator, _ = baseline_run
    save_dir = generator.config.save_dir
    saved = yaml.safe_load(open(os.path.join(save_dir, "params.yml")))
    record = _record(save_dir)
    assert {key: record[key] for key in saved} == saved


def test_derived_section(baseline_run):
    generator, _ = baseline_run
    derived = _record(generator.config.save_dir)["derived"]
    assert derived["mask_nside"] == 32
    assert derived["bins"]["lbins"][0] == 2 and derived["bins"]["lbins"][-1] == 60
    assert derived["bins"]["n_bins"] == len(derived["bins"]["lbins"]) - 1
    assert derived["n_blocks"] == len(generator.cov_keys.keys())
    assert derived["spectra_lmax"] == 60
    assert derived["nthreads"]["sht"] >= 1
    assert derived["spectrum_units"] == {
        "cmb_spectrum": "Cl",
        "foregrounds": "Cl",
        "nl": "Cl",
    }
    assert "acc" not in derived and "map_filter" not in derived


def test_provenance_section(baseline_run):
    generator, _ = baseline_run
    text = open(os.path.join(generator.config.save_dir, RESOLVED_NAME)).read()
    provenance = yaml.safe_load(text)["provenance"]
    for key in (
        "cmbcov_version",
        "git_revision",
        "python",
        "numpy",
        "scipy",
        "ducc0",
        "healpy",
        "cpu_count",
        "thread_environment",
        "start_time",
        "end_time",
        "wall_time_s",
        "status",
    ):
        assert key in provenance, key
    assert provenance["status"] == "completed"
    assert provenance["numpy"] == np.__version__
    assert provenance["git_revision"] == generator.config.git_hash
    start = datetime.datetime.fromisoformat(provenance["start_time"])
    end = datetime.datetime.fromisoformat(provenance["end_time"])
    assert start.tzinfo is not None and end >= start
    assert provenance["wall_time_s"] >= 0
    # no host name, no user name key
    assert socket.gethostname() not in text or len(socket.gethostname()) < 4
    assert not {"host", "hostname", "user", "username"} & set(provenance)


def test_params_yml_is_what_it_was_byte_for_byte(baseline_run):
    """``params.yml`` is the dump of the effective record, as before."""
    generator, _ = baseline_run
    expected = yaml.dump(
        dict(generator.params), default_flow_style=False, sort_keys=False
    )
    with open(os.path.join(generator.config.save_dir, "params.yml")) as handle:
        assert handle.read() == expected


def test_a_second_run_into_the_same_directory_is_not_confused(tmp_path):
    """``params_resolved.yml`` is a ``.yml`` file in the version directory but
    not a parameter file: ``--overwrite 0`` still finds one parameter file,
    compares against it, and leaves it byte for byte."""
    path = _baseline_params(tmp_path)
    first = CovarianceMatrixGenerator(path)
    first.run_full_analysis()
    save_dir = first.config.save_dir
    before = open(os.path.join(save_dir, "params.yml"), "rb").read()
    second = CovarianceMatrixGenerator(path)
    second.run_full_analysis(overwrite=0)
    assert second.config.save_dir == save_dir
    assert open(os.path.join(save_dir, "params.yml"), "rb").read() == before
    assert _record(save_dir)["provenance"]["status"] == "completed"
    # a changed parameter is still refused, record or not
    text = open(path).read().replace("apodizesigma: 30.0", "apodizesigma: 20.0")
    changed = tmp_path / "changed.yml"
    changed.write_text(text)
    with pytest.raises(ValueError, match="apodizesigma"):
        CovarianceMatrixGenerator(str(changed)).run_full_analysis(overwrite=0)


def test_a_failed_run_still_has_its_record(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("no luck")

    monkeypatch.setattr(Cov, "compute_covariance_matrix", boom)
    generator = CovarianceMatrixGenerator(_baseline_params(tmp_path))
    with pytest.raises(RuntimeError, match="no luck"):
        generator.run_full_analysis()
    record = _record(generator.config.save_dir)
    assert record["lmax"] == 60
    assert record["derived"]["mask_nside"] == 32  # known before the computation
    assert record["provenance"]["status"] == "failed"
    assert record["provenance"]["error"] == "RuntimeError"
    assert "end_time" in record["provenance"]


def test_a_dry_run_is_marked(tmp_path):
    generator = CovarianceMatrixGenerator(_baseline_params(tmp_path))
    generator.run_full_analysis(dryrun=True)
    assert _record(generator.config.save_dir)["provenance"]["status"] == "dry run"


def test_the_record_is_written_before_the_computation(tmp_path, monkeypatch):
    seen = {}
    real = Cov.compute_covariance_matrix

    def look(cov, *args, **kwargs):
        seen["record"] = _record(os.path.join(str(tmp_path), "v0"))
        return real(cov, *args, **kwargs)

    monkeypatch.setattr(Cov, "compute_covariance_matrix", look)
    CovarianceMatrixGenerator(_baseline_params(tmp_path)).run_full_analysis()
    assert seen["record"]["provenance"]["status"] == "running"
    assert "end_time" not in seen["record"]["provenance"]
    assert seen["record"]["derived"]["bins"]["n_bins"] > 0


# --------------------------------------------------------------------------- #
# map_filter and ACC
# --------------------------------------------------------------------------- #
def _filter_params(workdir):
    shutil.copy(os.path.join(DATA, MASK), workdir)
    ell = np.arange(80)
    tt = np.zeros(80)
    tt[2:] = 1e3 / (ell[2:] * (ell[2:] + 1)) ** 0.9
    np.savetxt(
        os.path.join(workdir, "cls.dat"),
        np.column_stack([ell, tt, 0.05 * tt, 0.001 * tt, 0.1 * tt]),
    )
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
        "map_filter": {"type": "fourier_highpass", "lx": 12, "nprobe": 5, "seed": 4},
    }
    path = os.path.join(workdir, "params.yml")
    with open(path, "w") as handle:
        yaml.dump(params, handle)
    return path


def test_map_filter_block_is_expanded_and_its_plan_recorded(tmp_path):
    generator = CovarianceMatrixGenerator(_filter_params(str(tmp_path)))
    generator.run_full_analysis()
    record = _record(generator.config.save_dir)
    block = record["map_filter"]
    # the file set type, lx, nprobe and seed; the rest are the defaults
    assert (block["type"], block["lx"], block["nprobe"], block["seed"]) == (
        "fourier_highpass",
        12,
        5,
        4,
    )
    assert block["shape"] == "sharp" and block["node_ratio"] == 1.25
    assert block["nprobe_near_cut"] == 128 and block["lowpass"] is None
    assert block["dmax"] is None and block["lw"] is None  # as asked
    plan = record["derived"]["map_filter"]
    settings = generator.config.map_filter
    assert plan["dmax"] == settings.resolved_dmax(20, generator.config.method, 0)
    assert plan["lw"] == settings.resolved_lw(32, 20)
    nodes = [int(n) for n in generator._map_filter_ratio["nodes"]]
    assert plan["nodes"] == nodes
    assert plan["nprobe"] == [int(n) for n in generator._map_filter_ratio["nprobe"]]
    assert len(plan["lmax_int_floor"]) == len(nodes)
    assert plan["channels"] == [str(c) for c in generator._map_filter_ratio["channels"]]
    assert plan["spectrum_aware"] is False


def test_acc_precompute_and_run_records(tmp_path):
    template = open(os.path.join(DATA, "baseline_params.yml")).read()
    text = template.replace("PLACEHOLDER_OUT", str(tmp_path / "out")).replace(
        "PLACEHOLDER_DATA", os.path.abspath(DATA)
    )
    text = text.replace(
        "covariance_approximation: nka",
        "covariance_approximation: acc\ndmax: 3\ncentralell: 16",
    )
    text = text.replace("lmax: 60", "lmax: 48").replace("[20, 60, 10]", "[20, 48, 10]")
    path = tmp_path / "params.yml"
    path.write_text(text + "acc_precompute: {nside: 16, grid: gl, lw: 10}\n")
    kernel_dir = tmp_path / "out"

    # a dry run writes nothing
    CovarianceMatrixGenerator(str(path)).precompute_acc_kernels(dryrun=True)
    assert not kernel_dir.exists()

    CovarianceMatrixGenerator(str(path)).precompute_acc_kernels()
    pre = yaml.safe_load(open(kernel_dir / PRECOMPUTE_RESOLVED_NAME))
    assert pre["acc_precompute"] == {
        "nside": 16,
        "grid": "gl",
        "lw": 10,
        "spectra": None,
        "max_memory_gb": None,
        "scratch_dir": None,
    }
    assert pre["dmax"] == 3 and pre["centralell"] == 16
    assert pre["save_dir"] is None
    assert pre["derived"]["acc"]["ellprange"] == [16, 17, 18]
    assert pre["derived"]["acc"]["kernel_dir"] == str(
        kernel_dir / "covariance_coupling"
    )
    assert pre["derived"]["kernels"] == {"grid": "gl", "nside": 16, "lw": 10}
    assert pre["provenance"]["status"] == "completed"

    # without an acc_precompute block, the precompute record has the defaults
    path.write_text(text)
    config = ParameterManager(str(path)).load_and_validate(setup_output=False)
    assert "acc_precompute" not in resolved_parameters(config)
    block = resolved_parameters(config, acc_precompute_defaults=True)["acc_precompute"]
    assert block["grid"] == "gl" and block["nside"] is None and block["lw"] is None

    # the run reads the kernels; its record names where and the internal limit
    path.write_text(text + "acc_precompute: {nside: 16, grid: gl, lw: 10}\n")
    generator = CovarianceMatrixGenerator(str(path))
    generator.run_full_analysis()
    record = _record(generator.config.save_dir)
    acc = record["derived"]["acc"]
    assert acc["dmax"] == 3 and acc["centralell"] == 16
    assert acc["kernel_dir"] == str(kernel_dir / "covariance_coupling")
    assert acc["lmax_int"] == 48 + 2 * 16 - 1 - 16
    assert record["derived"]["spectra_lmax"] == acc["lmax_int"]
    assert record["acc_precompute"]["lw"] == 10


# --------------------------------------------------------------------------- #
# The helpers
# --------------------------------------------------------------------------- #
def test_yaml_safe():
    value = {
        "i": np.int64(3),
        "f": np.float32(0.5),
        "b": np.bool_(True),
        "t": (1, 2),
        "a": np.arange(3),
        "p": os.path.join("a", "b"),
        1: {"nested": [np.float64(1.5), None]},
        "inf": float("inf"),
        "obj": object,
    }
    safe = yaml_safe(value)
    assert safe["i"] == 3 and type(safe["i"]) is int
    assert type(safe["f"]) is float and type(safe["b"]) is bool
    assert safe["t"] == [1, 2] and safe["a"] == [0, 1, 2]
    assert safe["1"] == {"nested": [1.5, None]}
    assert safe["inf"] == "inf"
    assert isinstance(safe["obj"], str)
    assert yaml.safe_load(yaml.safe_dump(safe)) == safe


def test_long_lists_and_arrays_are_not_dumped():
    assert yaml_safe(list(range(MAX_ITEMS))) == list(range(MAX_ITEMS))
    assert yaml_safe(list(range(MAX_ITEMS + 1))) == f"<{MAX_ITEMS + 1} values omitted>"
    assert yaml_safe(np.zeros((MAX_ITEMS, 2))) == f"<{2 * MAX_ITEMS} values omitted>"


def test_the_record_class_survives_an_unwritable_path(tmp_path, caplog):
    record = RunRecord(str(tmp_path / "missing" / RESOLVED_NAME), {"lmax": 3})
    record.write()  # a warning, not an exception
    record.finish("completed")
    assert "could not write" in caplog.text
    ok = RunRecord(
        str(tmp_path / RESOLVED_NAME), {"lmax": 3}, clock=iter([0, 2.5]).__next__
    )
    ok.finish("completed")
    assert (
        yaml.safe_load(open(tmp_path / RESOLVED_NAME))["provenance"]["wall_time_s"]
        == 2.5
    )
    assert not os.path.exists(tmp_path / f"{RESOLVED_NAME}.tmp")
