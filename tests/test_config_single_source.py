"""
One value, one representation.

`PipelineConfig` is the single source of truth: a plain dictionary parsed
from the file and mutated after validation, alongside a `CovarianceConfig`
rebuilt from it at the last moment, would let two copies of one value drift
apart.

These tests pin the closure: `PipelineConfig` is built once by the validator,
every consumer reads it, the parameter record that remains is read-only and
derived, and both spellings of the asymmetric-Stokes key reach the same field.
"""

import glob
import os

import pytest

healpy = pytest.importorskip("healpy")

from cmbcov import (  # noqa: E402
    CovarianceMatrixGenerator,
    ParameterManager,
)
from cmbcov.covariance import CovarianceMethod  # noqa: E402
from cmbcov.generator.parameter_validation import (  # noqa: E402
    PipelineConfig,
    normalise_param_aliases,
)

DATA = os.path.join(os.path.dirname(__file__), "data")


def write_params(tmp_path, extra=""):
    """Write the baseline parameter file into `tmp_path`, plus `extra` lines."""
    template = open(os.path.join(DATA, "baseline_params.yml")).read()
    path = tmp_path / "params.yml"
    path.write_text(
        template.replace("PLACEHOLDER_OUT", str(tmp_path)).replace(
            "PLACEHOLDER_DATA", os.path.abspath(DATA)
        )
        + extra
    )
    return str(path)


def loaded_generator(tmp_path, extra=""):
    """A generator with parameters loaded and the data model built."""
    generator = CovarianceMatrixGenerator(write_params(tmp_path, extra))
    generator.load_and_save_parameters()
    generator.setup_frequency_stokes_mapping()
    return generator


# ----------------------------------------------------------------------------
# The config is the single source of truth
# ----------------------------------------------------------------------------


def test_validation_returns_the_config(tmp_path):
    """Validation produces the object, not a dictionary to build one from."""
    config = ParameterManager(write_params(tmp_path)).load_and_validate()
    assert isinstance(config, PipelineConfig)


def test_every_holder_of_params_shares_one_object(tmp_path):
    """
    The generator, the manager and the spectra loader must not each hold their
    own copy: there is one record, hanging off one config.
    """
    generator = loaded_generator(tmp_path)

    assert generator.config is generator.param_manager.config
    assert generator.spectra.config is generator.config
    assert generator.params is generator.config.params
    assert generator.param_manager.params is generator.config.params
    assert generator.spectra.params is generator.config.params


def test_params_record_cannot_be_written(tmp_path):
    """
    The failure this closes: change a value in one representation, read the
    stale one somewhere else. The record is immutable, so there is nothing to
    change.
    """
    generator = loaded_generator(tmp_path)

    with pytest.raises(TypeError):
        generator.params["lmax"] = 9999
    with pytest.raises(TypeError):
        generator.params["sum_assymetric_stokes"] = True
    with pytest.raises(TypeError):
        del generator.params["lmax"]

    # And the config itself is frozen: neither a field nor a delegating
    # property can be reassigned. (dataclasses.FrozenInstanceError is an
    # AttributeError.)
    with pytest.raises(AttributeError):
        generator.config.cov_name = "somewhere_else.dat"
    with pytest.raises(AttributeError):
        generator.config.lmax = 9999

    assert generator.config.lmax == generator.params["lmax"] == 60
    assert generator.lmax == 60


def test_generator_params_cannot_be_rebound(tmp_path):
    """Swapping in a whole new dictionary would recreate the second copy."""
    generator = loaded_generator(tmp_path)
    with pytest.raises(AttributeError):
        generator.params = {"lmax": 9999}
    with pytest.raises(AttributeError):
        generator.save_dir = "/somewhere/else"


def test_record_agrees_with_the_fields_consumers_read(tmp_path):
    """
    The record is derived from the same values the pipeline reads. If the two
    ever disagreed, the saved parameter file would not describe the run.
    """
    config = loaded_generator(tmp_path).config

    assert config.lmax == config.params["lmax"]
    assert config.lmin == config.params["lmin"]
    assert config.Dl == config.params["Dl"]
    assert config.thetamax == config.params["thetamax"]
    assert config.cov_name == config.params["cov_name"]
    assert config.mask_name == config.params["mask_name"]
    assert config.bins == config.params["bins"]
    assert config.frequencies == config.params["frequencies"]
    assert config.sum_asymmetric_stokes == config.params["sum_assymetric_stokes"]
    assert config.covariance_approximation == config.params["covariance_approximation"]
    assert config.method is CovarianceMethod.NKA

    # The mapping interface reads the same record.
    assert config["lmax"] == config.lmax
    assert config.get("nothing_here") is None


def test_save_dir_and_git_hash_are_set_at_construction(tmp_path):
    """
    These two were the write-back: `_setup_directories` put them into the
    dictionary after validation. They are fields now, resolved once.
    """
    generator = loaded_generator(tmp_path)
    config = generator.config

    assert config.save_dir == sorted(glob.glob(os.path.join(str(tmp_path), "v*")))[-1]
    assert config.save_dir == config.params["save_dir"]
    assert generator.save_dir == config.save_dir
    assert generator.param_manager.save_dir == config.save_dir
    assert config.git_hash and config.git_hash == config.params["git_hash"]


def test_covariance_config_is_not_rebuilt_downstream(tmp_path):
    """
    `Cov` must receive the CovarianceConfig the validator built, not a second
    one assembled from the dictionary at setup time.
    """
    generator = loaded_generator(tmp_path)
    generator.load_pre_process()
    generator.load_post_process()
    generator.setup_analysis_object()

    assert generator.covariance_instance.config is generator.config.covariance
    assert generator.covariance_instance.lmax == generator.config.lmax


def test_noise_bias_flag_is_not_written_back_into_the_record(tmp_path):
    """
    Loading white-noise levels is loader state, not written back into the
    parameter record: the record still says what the file said.
    """
    generator = loaded_generator(tmp_path)
    assert generator.config.nl_is_biased is True  # the default
    assert generator.spectra.noise_is_biased is True

    generator.load_pre_process()

    assert generator.spectra.noise_is_biased is False  # white noise is unbiased
    assert generator.params["nl_is_biased"] is True
    assert generator.config.nl_is_biased is True


# ----------------------------------------------------------------------------
# Both spellings of the asymmetric-Stokes key
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["sum_assymetric_stokes", "sum_asymmetric_stokes"])
def test_both_spellings_reach_the_same_field(tmp_path, key):
    """
    `sum_assymetric_stokes` (double s) is what real parameter files contain and
    must keep working; the correct spelling is accepted as well. Both set the
    one field `CovarianceConfig.sum_asymmetric_stokes`.
    """
    config = loaded_generator(tmp_path, extra=f"\n{key}: True\n").config

    assert config.sum_asymmetric_stokes is True
    assert config.covariance.sum_asymmetric_stokes is True
    # The record is normalised onto the canonical file key, so the parameter
    # file written beside the results always has one spelling.
    assert config.params["sum_assymetric_stokes"] is True
    assert "sum_asymmetric_stokes" not in config.params


def test_neither_spelling_keeps_the_documented_default(tmp_path):
    config = loaded_generator(tmp_path).config
    assert config.sum_asymmetric_stokes is False
    assert config.params["sum_assymetric_stokes"] is False


def test_agreeing_spellings_are_accepted(tmp_path):
    config = loaded_generator(
        tmp_path,
        extra="\nsum_assymetric_stokes: True\nsum_asymmetric_stokes: True\n",
    ).config
    assert config.sum_asymmetric_stokes is True


def test_conflicting_spellings_are_an_error(tmp_path):
    """Preferring one silently is the substitution failure, not a convenience."""
    path = write_params(
        tmp_path,
        extra="\nsum_assymetric_stokes: False\nsum_asymmetric_stokes: True\n",
    )
    with pytest.raises(ValueError, match="sum_assymetric_stokes"):
        ParameterManager(path).load_and_validate()


def test_alias_normalisation_is_pure():
    """The caller's dictionary is not edited underneath it."""
    original = {"sum_asymmetric_stokes": True, "lmax": 60}
    normalised = normalise_param_aliases(original)
    assert normalised == {"sum_assymetric_stokes": True, "lmax": 60}
    assert original == {"sum_asymmetric_stokes": True, "lmax": 60}


# ----------------------------------------------------------------------------
# Absent values are reported, never guessed
# ----------------------------------------------------------------------------


def test_missing_required_parameter_raises_rather_than_defaulting():
    with pytest.raises(ValueError, match="Required parameter 'lmax'"):
        PipelineConfig.from_params(
            {
                "cov_path": "./out",
                "cov_name": "c.dat",
                "covariance_approximation": "nka",
                "save_dir": "./out/v0",
                "git_hash": "abc1234",
            }
        )


def test_unknown_method_names_the_valid_ones():
    with pytest.raises(ValueError, match="Unknown covariance_approximation"):
        PipelineConfig.from_params({"covariance_approximation": "not_a_method"})


# ----------------------------------------------------------------------------
# save_raw_blocks: validated, default off, one field
# ----------------------------------------------------------------------------


def test_save_raw_blocks_defaults_to_false_and_reaches_the_covariance_config(tmp_path):
    config = loaded_generator(tmp_path).config
    assert config.save_raw_blocks is False
    assert config.covariance.save_raw_blocks is False
    assert config.params["save_raw_blocks"] is False

    (tmp_path / "on").mkdir()
    on = loaded_generator(tmp_path / "on", extra="\nsave_raw_blocks: true\n").config
    assert on.save_raw_blocks is True and on.covariance.save_raw_blocks is True


def test_save_raw_blocks_must_be_a_boolean(tmp_path):
    path = write_params(tmp_path, extra="\nsave_raw_blocks: 'yes please'\n")
    with pytest.raises(ValueError, match="save_raw_blocks must be true or false"):
        ParameterManager(path).load_and_validate()


def test_pipeline_run_writes_no_raw_blocks_by_default(tmp_path):
    CovarianceMatrixGenerator(write_params(tmp_path)).run_full_analysis()
    version = sorted(glob.glob(os.path.join(str(tmp_path), "v*")))[-1]
    assert [n for n in os.listdir(version) if n.startswith("cov_")] == []


# ----------------------------------------------------------------------------
# --overwrite N against a parameter file saved by an older version
# ----------------------------------------------------------------------------


def _saved_params(version_dir):
    import yaml

    path = os.path.join(version_dir, "params.yml")
    with open(path) as handle:
        return path, yaml.safe_load(handle)


def _rewrite(path, params):
    import yaml

    with open(path, "w") as handle:
        yaml.dump(params, handle, default_flow_style=False, sort_keys=False)


def test_overwrite_accepts_an_old_saved_file_without_an_optional_key(tmp_path):
    params_file = write_params(tmp_path)
    CovarianceMatrixGenerator(params_file).run_full_analysis()
    version = os.path.join(str(tmp_path), "v0")
    saved_path, saved = _saved_params(version)
    # What a version directory written before save_raw_blocks existed holds.
    del saved["save_raw_blocks"]
    _rewrite(saved_path, saved)

    CovarianceMatrixGenerator(params_file).run_full_analysis(overwrite=0)
    assert sorted(glob.glob(os.path.join(str(tmp_path), "v*"))) == [version]
    assert _saved_params(version)[1]["save_raw_blocks"] is False


def test_overwrite_still_refuses_a_saved_file_whose_value_differs(tmp_path):
    params_file = write_params(tmp_path)
    CovarianceMatrixGenerator(params_file).run_full_analysis()
    saved_path, saved = _saved_params(os.path.join(str(tmp_path), "v0"))
    saved["save_raw_blocks"] = True
    _rewrite(saved_path, saved)

    with pytest.raises(ValueError, match="not compatible"):
        CovarianceMatrixGenerator(params_file).run_full_analysis(overwrite=0)


def test_overwrite_still_refuses_a_saved_file_missing_a_required_key(tmp_path):
    """Only keys with a documented default are filled in."""
    params_file = write_params(tmp_path)
    CovarianceMatrixGenerator(params_file).run_full_analysis()
    saved_path, saved = _saved_params(os.path.join(str(tmp_path), "v0"))
    del saved["cov_name"]
    _rewrite(saved_path, saved)

    with pytest.raises(ValueError, match="not compatible"):
        CovarianceMatrixGenerator(params_file).run_full_analysis(overwrite=0)


def test_overwrite_ignores_provenance_keys(tmp_path):
    """
    git_hash and save_dir record the run, not the request: a version
    directory written by another revision (or moved) is still overwritable.
    """
    from cmbcov.generator.parameter_validation import (
        ParameterValidator,
    )

    assert ParameterValidator.PROVENANCE_PARAMS == ["save_dir", "git_hash"]
    params_file = write_params(tmp_path)
    CovarianceMatrixGenerator(params_file).run_full_analysis()
    saved_path, saved = _saved_params(os.path.join(str(tmp_path), "v0"))
    saved["git_hash"] = "0000000"
    saved["save_dir"] = "/somewhere/else/v0"
    _rewrite(saved_path, saved)

    CovarianceMatrixGenerator(params_file).run_full_analysis(overwrite=0)
    assert sorted(glob.glob(os.path.join(str(tmp_path), "v*"))) == [
        os.path.join(str(tmp_path), "v0")
    ]
    assert _saved_params(saved_path[: -len("/params.yml")])[1]["git_hash"] != "0000000"


def _variant(tmp_path, base_params, replace=None, extra=""):
    """
    The same parameter file, under the same name, in its own directory,
    still pointing at ``tmp_path`` as cov_path. (A renamed file is checked
    too: see the tests below.)
    """
    text = open(base_params).read()
    for old, new_text in (replace or {}).items():
        assert old in text
        text = text.replace(old, new_text)
    directory = tmp_path / f"variant{abs(hash((str(replace), extra))) % 10**6}"
    directory.mkdir()
    path = directory / os.path.basename(base_params)
    path.write_text(text + extra)
    return str(path)


def test_overwrite_allows_a_different_binning(tmp_path):
    params_file = write_params(tmp_path)
    CovarianceMatrixGenerator(params_file).run_full_analysis()
    rebinned = _variant(
        tmp_path,
        params_file,
        replace={
            "  - [2, 20, 6]\n  - [20, 60, 10]\n": "  - [2, 60, 4]\n",
            "lmin: 2": "lmin: 4",
        },
    )
    CovarianceMatrixGenerator(rebinned).run_full_analysis(overwrite=0)
    assert _saved_params(os.path.join(str(tmp_path), "v0"))[1]["lmin"] == 4


@pytest.mark.parametrize(
    "replace, extra, key",
    [
        (None, "\ncentralell: 16\n", "centralell"),
        (None, "\ndmax: 3\n", "dmax"),
        (
            {"mask_name: baseline_mask.fits": "mask_name: other_mask.fits"},
            "",
            "mask_name",
        ),
        (
            {"frequencies: ['090GHz', '150GHz']": "frequencies: ['090GHz']"},
            "",
            "frequencies",
        ),
    ],
)
def test_overwrite_refuses_a_numerical_parameter_change(tmp_path, replace, extra, key):
    params_file = write_params(tmp_path)
    CovarianceMatrixGenerator(params_file).run_full_analysis()
    changed = _variant(tmp_path, params_file, replace=replace, extra=extra)
    with pytest.raises(ValueError, match=f"not compatible.*{key}"):
        CovarianceMatrixGenerator(changed).run_full_analysis(overwrite=0)


# ----------------------------------------------------------------------------
# --overwrite N after the parameter file was renamed
# ----------------------------------------------------------------------------


def _renamed(tmp_path, base_params, name, replace=None):
    """``base_params`` (with ``replace`` applied) saved under another name."""
    text = open(base_params).read()
    for old, new_text in (replace or {}).items():
        assert old in text
        text = text.replace(old, new_text)
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def _yaml_files(version_dir):
    return sorted(n for n in os.listdir(version_dir) if n.endswith((".yml", ".yaml")))


def test_overwrite_checks_a_renamed_parameter_file(tmp_path):
    """The saved file under its old name is still what the run is checked against."""
    params_file = write_params(tmp_path)
    CovarianceMatrixGenerator(params_file).run_full_analysis()
    renamed = _renamed(
        tmp_path,
        params_file,
        "renamed.yml",
        replace={"mask_name: baseline_mask.fits": "mask_name: other_mask.fits"},
    )
    with pytest.raises(ValueError, match="not compatible.*mask_name"):
        CovarianceMatrixGenerator(renamed).run_full_analysis(overwrite=0)


def test_overwrite_with_a_renamed_parameter_file_leaves_one_parameter_file(tmp_path):
    params_file = write_params(tmp_path)
    CovarianceMatrixGenerator(params_file).run_full_analysis()
    renamed = _renamed(tmp_path, params_file, "renamed.yaml")
    CovarianceMatrixGenerator(renamed).run_full_analysis(overwrite=0)
    assert _yaml_files(os.path.join(str(tmp_path), "v0")) == ["renamed.yaml"]


def test_overwrite_refuses_a_version_dir_with_several_parameter_files(tmp_path):
    params_file = write_params(tmp_path)
    CovarianceMatrixGenerator(params_file).run_full_analysis()
    version = os.path.join(str(tmp_path), "v0")
    with open(os.path.join(version, "stray.yml"), "w") as handle:
        handle.write("lmax: 60\n")
    with pytest.raises(ValueError, match="more than one parameter file"):
        CovarianceMatrixGenerator(params_file).run_full_analysis(overwrite=0)
