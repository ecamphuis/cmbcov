"""
Column layouts of the tabulated spectrum inputs.

Every file parameter (``cmb_spectrum``, ``fl``, ``foregrounds``, ``nl``,
``post_process_correction``) is read with :func:`read_spectrum_file` and then
one column is picked per Stokes key by
:func:`cmbcov.spectra.stokes_columns`. A 4-column file
(TT, EE, BB, TE) has no ET column: ``ET`` is the same spectrum as ``TE`` with
the fields swapped, so every loader reads it from the TE column. A 9-column
file is read as it is, so a file that distinguishes ET from TE is honoured.
"""

import os
import warnings

import numpy as np
import pytest

healpy = pytest.importorskip("healpy")

from cmbcov import CovarianceMatrixGenerator  # noqa: E402
from cmbcov.spectra import (  # noqa: E402
    SPECTRUM_FILE_LAYOUTS,
    stokes_columns,
)

DATA = os.path.join(os.path.dirname(__file__), "data")
LMAX = 60
PAIRS = ["090GHz090GHz", "090GHz150GHz", "150GHz150GHz"]
STOKES_KEYS = ["TT", "TE", "EE", "ET"]


def _columns(n_columns, seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.1, 0.9, (LMAX, n_columns))


def _write(directory, name, columns):
    os.makedirs(directory, exist_ok=True)
    template = os.path.join(directory, name + "_{}.dat")
    for pair in PAIRS:
        table = np.column_stack([np.arange(LMAX, dtype=float), columns])
        np.savetxt(template.format(pair), table)
    return template


def _loader(tmp_path, extra):
    template = open(os.path.join(DATA, "baseline_params.yml")).read()
    params = tmp_path / f"p{abs(hash(extra)) % 10**6}.yml"
    # One output directory per parameter set: a run with no covariance yet
    # leaves its version directory reusable, and a later run with other
    # parameters would be refused there.
    params.write_text(
        template.replace(
            "PLACEHOLDER_OUT", str(tmp_path / f"out_{params.stem}")
        ).replace("PLACEHOLDER_DATA", os.path.abspath(DATA))
        + extra
    )
    generator = CovarianceMatrixGenerator(str(params))
    generator.load_and_save_parameters()
    generator.setup_frequency_stokes_mapping()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        generator.load_pre_process()
        generator.load_post_process()
    return generator


# --------------------------------------------------------------------------- #
# stokes_columns itself
# --------------------------------------------------------------------------- #


def test_four_and_nine_column_files_agree_when_et_equals_te():
    four = _columns(4, 1)
    nine = np.zeros((LMAX, 9))
    nine[:, :4] = four
    nine[:, 6] = four[:, 3]  # ET = TE
    got_four = stokes_columns(four, STOKES_KEYS, "f.dat", "fl")
    got_nine = stokes_columns(nine, STOKES_KEYS, "f.dat", "fl")
    for key in STOKES_KEYS:
        np.testing.assert_array_equal(got_four[key], got_nine[key])
    np.testing.assert_array_equal(got_four["ET"], four[:, 3])


def test_nine_column_file_keeps_a_distinct_et():
    nine = _columns(9, 2)
    got = stokes_columns(nine, STOKES_KEYS, "f.dat", "fl")
    np.testing.assert_array_equal(got["ET"], nine[:, 6])
    assert not np.allclose(got["ET"], got["TE"])


def test_one_column_file_is_tt_only():
    one = _columns(1, 3)
    np.testing.assert_array_equal(
        stokes_columns(one, ["TT"], "f.dat", "nl")["TT"], one[:, 0]
    )
    with pytest.raises(ValueError, match="cannot supply 'EE'"):
        stokes_columns(one, ["TT", "EE"], "f.dat", "nl")


@pytest.mark.parametrize("n_columns", [2, 3, 5, 8, 10])
def test_unsupported_column_count_names_what_is_supported(n_columns):
    with pytest.raises(ValueError, match=f"found {n_columns} spectrum columns"):
        stokes_columns(_columns(n_columns, 4), ["TT"], "f.dat", "foregrounds")
    for supported in SPECTRUM_FILE_LAYOUTS:
        assert stokes_columns(_columns(supported, 5), ["TT"], "f.dat", "foregrounds")[
            "TT"
        ].shape == (LMAX,)


# --------------------------------------------------------------------------- #
# through the loaders: every file parameter accepts a 4-column file
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "param, attribute",
    [
        ("fl", "fl_dict"),
        ("foregrounds", None),
        ("nl", "nl_dict"),
        ("post_process_correction", "post_process_corrections"),
    ],
)
def test_four_column_file_works_for_every_file_parameter(tmp_path, param, attribute):
    columns = _columns(4, 6)
    template = _write(tmp_path / param, param, columns)
    generator = _loader(tmp_path, f"\n{param}: {template}\n")
    if attribute is None:  # foregrounds are folded into the signal spectra
        loaded = generator.cl_dict
        reference = _loader(tmp_path, "").cl_dict
        for pair in PAIRS:
            for i, key in zip((0, 3, 1, 3), ("TT", "TE", "EE", "ET")):
                np.testing.assert_allclose(
                    loaded[pair][key] - reference[pair][key],
                    columns[:, i],
                    rtol=1e-12,
                )
        return
    loaded = getattr(generator.spectra, attribute)
    for pair in PAIRS:
        for i, key in zip((0, 3, 1, 3), ("TT", "TE", "EE", "ET")):
            np.testing.assert_array_equal(loaded[pair][key], columns[:, i])


def test_four_and_nine_column_inputs_give_the_same_covariance(tmp_path):
    columns = _columns(4, 7)
    nine = np.zeros((LMAX, 9))
    nine[:, :4] = columns
    nine[:, 6] = columns[:, 3]
    four_template = _write(tmp_path / "four", "ppc", columns)
    nine_template = _write(tmp_path / "nine", "ppc", nine)
    a = _loader(tmp_path, f"\npost_process_correction: {four_template}\n")
    b = _loader(tmp_path, f"\npost_process_correction: {nine_template}\n")
    for pair in PAIRS:
        for key in STOKES_KEYS:
            np.testing.assert_array_equal(
                a.post_process_corrections[pair][key],
                b.post_process_corrections[pair][key],
            )


def test_unsupported_cmb_spectrum_column_count_still_names_the_file(tmp_path):
    bad = tmp_path / "bad_cls.dat"
    np.savetxt(bad, np.column_stack([np.arange(LMAX, dtype=float), _columns(3, 8)]))
    with pytest.raises(ValueError, match="bad_cls.dat"):
        _loader(tmp_path, f"\ncmb_spectrum: {bad}\n")
