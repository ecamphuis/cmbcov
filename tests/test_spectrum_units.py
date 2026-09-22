"""
``spectrum_units``: the convention a tabulated input is written in.

Reading a spectrum file as ``C_l`` with no conversion when it in fact holds
``D_l = l(l+1) C_l / 2pi`` -- the file shipped in ``tests/data`` (and most
published spectra, CAMB's ``*_lensedCls.dat`` among them, are ``D_l``) --
biases the covariance by ``(l(l+1)/2pi)^2`` -- a factor 1.2e6 at l = 2000 --
silently.

The key is ``spectrum_units: Cl | Dl``, global, with a per-input override
``<input>_units`` for each of the three inputs whose columns are power
spectra: ``cmb_spectrum``, ``foregrounds`` and ``nl``. The default is ``Cl``
everywhere. Beams, the pixel window, ``fl``/``hl`` and
``post_process_correction`` are dimensionless and are never converted,
whatever the key says.
"""

import glob
import os
import warnings

import numpy as np
import pytest

healpy = pytest.importorskip("healpy")

from cmbcov import CovarianceMatrixGenerator  # noqa: E402
from cmbcov.generator.parameter_validation import (  # noqa: E402
    ParameterValidator,
    canonical_spectrum_units,
)
from cmbcov.utils.array_utils import dl_to_cl  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "data")
LMAX = 60
FREQ_PAIRS = ("090GHz090GHz", "090GHz150GHz", "150GHz150GHz")


def _write_params(tmp_path, name, extra="", cmb=None):
    """A baseline parameter file in ``tmp_path``, with ``extra`` appended."""
    out = tmp_path / name
    template = open(os.path.join(DATA, "baseline_params.yml")).read()
    text = template.replace("PLACEHOLDER_OUT", str(out)).replace(
        "PLACEHOLDER_DATA", os.path.abspath(DATA)
    )
    if cmb is not None:
        text = text.replace(
            f"cmb_spectrum: {os.path.abspath(DATA)}/baseline_cls.dat",
            f"cmb_spectrum: {cmb}",
        )
        assert f"cmb_spectrum: {cmb}" in text
    params = tmp_path / f"{name}.yml"
    params.write_text(text + extra)
    return str(params), out


def _run(tmp_path, name, extra="", cmb=None):
    """Run the full pipeline and return the saved covariance."""
    params, out = _write_params(tmp_path, name, extra=extra, cmb=cmb)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        CovarianceMatrixGenerator(params).run_full_analysis()
    version = sorted(glob.glob(str(out / "v*")))[-1]
    return np.loadtxt(os.path.join(version, "covariance_matrix.dat"))


def _load(tmp_path, name, extra="", cmb=None):
    """Run the pipeline only as far as the inputs; return the SpectraLoader."""
    params, _ = _write_params(tmp_path, name, extra=extra, cmb=cmb)
    generator = CovarianceMatrixGenerator(params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        generator.load_and_save_parameters()
        generator.setup_frequency_stokes_mapping()
        generator.load_pre_process()
        generator.load_post_process()
    return generator.spectra


def _table(values, start=0):
    """``[ell, columns...]`` text-file rows for a (nell, nspec) array."""
    ell = np.arange(start, start + values.shape[0])
    return np.column_stack([ell, values])


def _write_table(path, values, start=0):
    # %.17g round-trips an IEEE double exactly, so a file written here and
    # read back holds the same floats: the comparisons below are then exact
    # and test the conversion, not the text format.
    np.savetxt(path, _table(values, start=start), fmt="%.17g")
    return str(path)


def _cl_and_dl_files(tmp_path, seed=7, nspec=4, start=2, nell=LMAX):
    """
    A pair of files holding the same spectra in the two conventions.

    The ``D_l`` table is written first; the ``C_l`` one is built from the
    values *as read back from it*, so the only difference between the two
    runs is the conversion itself.
    """
    rng = np.random.default_rng(seed)
    ell = np.arange(start, nell)
    dl = np.abs(rng.normal(size=(ell.size, nspec))) * 1e3
    dl_path = _write_table(tmp_path / "spectrum_dl.dat", dl, start=start)
    dl_read = np.loadtxt(dl_path)[:, 1:]
    cl = 2.0 * np.pi * dl_read / (ell * (ell + 1.0))[:, None]
    cl_path = _write_table(tmp_path / "spectrum_cl.dat", cl, start=start)
    return cl_path, dl_path


# ----------------------------------------------------------------- the maths


def test_dl_to_cl_is_the_textbook_conversion():
    dl = np.arange(1.0, 11.0)
    cl = dl_to_cl(dl)
    ell = np.arange(dl.size)
    np.testing.assert_allclose(
        cl[2:], 2 * np.pi * dl[2:] / (ell[2:] * (ell[2:] + 1)), rtol=0, atol=0
    )


def test_dl_to_cl_zeroes_the_monopole_and_dipole():
    """
    l(l+1) vanishes at l = 0 and the l = 1 convention of a D_l file is not
    fixed; both are set to zero rather than left alone or extrapolated.
    """
    dl = np.full(6, 123.0)
    cl = dl_to_cl(dl)
    assert cl[0] == 0.0
    assert cl[1] == 0.0
    assert (cl[2:] > 0).all()


def test_dl_to_cl_handles_several_columns():
    dl = np.arange(24.0).reshape(6, 4)
    cl = dl_to_cl(dl)
    assert cl.shape == dl.shape
    assert (cl[:2] == 0).all()
    for column in range(4):
        np.testing.assert_allclose(
            cl[:, column], dl_to_cl(dl[:, column]), rtol=0, atol=0
        )


# --------------------------------------------------------- the key end to end


def test_dl_file_and_equivalent_cl_file_give_identical_covariances(tmp_path):
    cl_path, dl_path = _cl_and_dl_files(tmp_path)
    from_cl = _run(tmp_path, "cl", extra="\ncmb_spectrum_units: Cl\n", cmb=cl_path)
    from_dl = _run(tmp_path, "dl", extra="\ncmb_spectrum_units: Dl\n", cmb=dl_path)
    np.testing.assert_array_equal(from_dl, from_cl)


def test_the_default_is_cl_and_changes_nothing(tmp_path):
    """
    No key at all, the global key set to Cl, and the per-input override set to
    Cl must all give the same covariance as before the key existed. (The
    recorded golden of tests/test_end_to_end_baseline.py pins the last part.)
    """
    plain = _run(tmp_path, "plain")
    global_cl = _run(tmp_path, "global", extra="\nspectrum_units: Cl\n")
    per_input = _run(tmp_path, "per_input", extra="\ncmb_spectrum_units: Cl\n")
    np.testing.assert_array_equal(global_cl, plain)
    np.testing.assert_array_equal(per_input, plain)


def test_cmb_spectrum_file_is_converted(tmp_path):
    cl_path, dl_path = _cl_and_dl_files(tmp_path)
    expected = np.zeros((LMAX, 4))
    expected[2:] = np.loadtxt(cl_path)[:, 1:]
    loader = _load(
        tmp_path, "cmb", extra="\nspectrum_units: Dl\nnl_units: Cl\n", cmb=dl_path
    )
    # cl_dict holds CMB + foregrounds (none here), before the data model.
    for index, stokes in enumerate(("TT", "EE")):
        np.testing.assert_allclose(
            loader.cl_dict[FREQ_PAIRS[0]][stokes], expected[:, index], rtol=1e-15
        )


def test_foregrounds_file_is_converted(tmp_path):
    cl_path, dl_path = _cl_and_dl_files(tmp_path, seed=11)
    # One file per frequency pair: the same table under every name.
    for pair in FREQ_PAIRS:
        for path, tag in ((cl_path, "cl"), (dl_path, "dl")):
            table = np.loadtxt(path)
            np.savetxt(tmp_path / f"fg_{tag}_{pair}.dat", table, fmt="%.17g")
    extra_dl = f"\nforegrounds: {tmp_path}/fg_dl_{{}}.dat\nforegrounds_units: Dl\n"
    extra_cl = f"\nforegrounds: {tmp_path}/fg_cl_{{}}.dat\n"
    from_dl = _load(tmp_path, "fg_dl", extra=extra_dl)
    from_cl = _load(tmp_path, "fg_cl", extra=extra_cl)
    for pair in FREQ_PAIRS:
        for stokes in ("TT", "EE", "TE"):
            np.testing.assert_array_equal(
                from_dl.cl_dict[pair][stokes], from_cl.cl_dict[pair][stokes]
            )
    # and it really is the D_l file that moved
    assert not np.allclose(
        from_dl.cl_dict[FREQ_PAIRS[0]]["TT"][2:],
        np.loadtxt(tmp_path / f"fg_dl_{FREQ_PAIRS[0]}.dat")[:, 1],
    )


def test_noise_file_is_converted(tmp_path):
    cl_path, dl_path = _cl_and_dl_files(tmp_path, seed=13)
    for pair in FREQ_PAIRS:
        for path, tag in ((cl_path, "cl"), (dl_path, "dl")):
            np.savetxt(tmp_path / f"nl_{tag}_{pair}.dat", np.loadtxt(path), fmt="%.17g")
    # `nl` as a filename template replaces the white-noise dict of the
    # baseline file; YAML takes the last value of a repeated key.
    from_dl = _load(
        tmp_path, "nl_dl", extra=f"\nnl: {tmp_path}/nl_dl_{{}}.dat\nnl_units: Dl\n"
    )
    from_cl = _load(tmp_path, "nl_cl", extra=f"\nnl: {tmp_path}/nl_cl_{{}}.dat\n")
    for pair in FREQ_PAIRS:
        for stokes in ("TT", "EE", "TE"):
            np.testing.assert_array_equal(
                from_dl.nl_dict[pair][stokes], from_cl.nl_dict[pair][stokes]
            )


def test_per_input_override_beats_the_global_key(tmp_path):
    """
    ``spectrum_units: Dl`` with ``cmb_spectrum_units: Cl`` must read the CMB
    file as C_l -- the common case of a D_l foreground template beside a C_l
    CMB spectrum (or the reverse).
    """
    cl_path, _ = _cl_and_dl_files(tmp_path, seed=17)
    override = _load(
        tmp_path,
        "override",
        extra="\nspectrum_units: Dl\ncmb_spectrum_units: Cl\nnl_units: Cl\n",
        cmb=cl_path,
    )
    plain = _load(tmp_path, "plain_units", cmb=cl_path)
    for stokes in ("TT", "EE", "TE"):
        np.testing.assert_array_equal(
            override.cl_dict[FREQ_PAIRS[0]][stokes],
            plain.cl_dict[FREQ_PAIRS[0]][stokes],
        )


def test_monopole_and_dipole_of_a_dl_file_are_zeroed_in_the_pipeline(tmp_path):
    """A file that starts at l = 0 with non-zero rows: they must not survive."""
    rng = np.random.default_rng(3)
    dl = np.abs(rng.normal(size=(LMAX, 4))) * 1e3 + 1.0
    path = _write_table(tmp_path / "from_zero_dl.dat", dl, start=0)
    loader = _load(
        tmp_path,
        "from_zero",
        extra="\ncmb_spectrum_units: Dl\n",
        cmb=path,
    )
    for stokes in ("TT", "EE", "TE"):
        assert loader.cl_dict[FREQ_PAIRS[0]][stokes][0] == 0.0
        assert loader.cl_dict[FREQ_PAIRS[0]][stokes][1] == 0.0
        assert loader.cl_dict[FREQ_PAIRS[0]][stokes][2] != 0.0
    # Under the Cl convention the same file keeps them, so the two
    # conventions genuinely differ at l < 2.
    kept = _load(tmp_path, "from_zero_cl", cmb=path)
    assert kept.cl_dict[FREQ_PAIRS[0]]["TT"][0] != 0.0


# ------------------------------------------------- what the key must not touch


def test_dimensionless_inputs_are_untouched_by_the_key(tmp_path):
    """
    Beams, the pixel window, the transfer function and the post-processing
    corrections are dimensionless: ``spectrum_units: Dl`` must leave them
    exactly as the file has them.
    """
    rng = np.random.default_rng(5)
    beams = rng.uniform(0.3, 1.0, size=(LMAX, 2))
    beams_path = _write_table(tmp_path / "beams.dat", beams)
    pixwin = rng.uniform(0.3, 1.0, size=(LMAX, 2))
    pixwin_path = _write_table(tmp_path / "pixwin.dat", pixwin)
    fl = rng.uniform(0.2, 1.0, size=(LMAX, 4))
    ppc = rng.uniform(0.5, 2.0, size=(LMAX, 4))
    for pair in FREQ_PAIRS:
        np.savetxt(tmp_path / f"fl_{pair}.dat", _table(fl), fmt="%.17g")
        np.savetxt(tmp_path / f"ppc_{pair}.dat", _table(ppc), fmt="%.17g")
    _, dl_path = _cl_and_dl_files(tmp_path, seed=19)

    extra = (
        f"\nbeams: {beams_path}"
        f"\npixwin: {pixwin_path}"
        f"\nfl: {tmp_path}/fl_{{}}.dat"
        f"\npost_process_correction: {tmp_path}/ppc_{{}}.dat"
        "\nspectrum_units: Dl"
        "\nnl_units: Cl\n"
    )
    loader = _load(tmp_path, "dimensionless", extra=extra, cmb=dl_path)

    np.testing.assert_array_equal(loader.beams["090GHz"], beams[:, 0])
    np.testing.assert_array_equal(loader.beams["150GHz"], beams[:, 1])
    np.testing.assert_array_equal(loader.pix["TT"], pixwin[:, 0] ** 2)
    np.testing.assert_array_equal(loader.pix["EE"], pixwin[:, 1] ** 2)
    for pair in FREQ_PAIRS:
        np.testing.assert_array_equal(loader.fl_dict[pair]["TT"], fl[:, 0])
        np.testing.assert_array_equal(loader.fl_dict[pair]["EE"], fl[:, 1])
        np.testing.assert_array_equal(
            loader.post_process_corrections[pair]["TT"], ppc[:, 0]
        )
        np.testing.assert_array_equal(
            loader.post_process_corrections[pair]["EE"], ppc[:, 1]
        )
    # the CMB spectrum next to them did move
    assert not np.allclose(
        loader.cl_dict[FREQ_PAIRS[0]]["TT"][2:], np.loadtxt(dl_path)[:, 1]
    )


# ------------------------------------------------------------------ refusals


@pytest.mark.parametrize("value", ["Dell", "D_l", "cl_", "", 1, None, True])
def test_an_unknown_units_value_raises_naming_the_allowed_ones(value):
    with pytest.raises(ValueError) as error:
        canonical_spectrum_units(value, "spectrum_units")
    message = str(error.value)
    assert "spectrum_units" in message
    assert "'Cl'" in message and "'Dl'" in message


@pytest.mark.parametrize("spelling", ["Cl", "cl", "CL", "Dl", "dl", "DL", " Dl "])
def test_case_does_not_matter(spelling):
    assert (
        canonical_spectrum_units(spelling, "spectrum_units")
        == spelling.strip().capitalize()
    )


def test_validation_reports_an_unknown_value(tmp_path):
    params, _ = _write_params(tmp_path, "bad", extra="\nspectrum_units: Dell\n")
    import yaml

    with open(params) as handle:
        loaded = yaml.safe_load(handle)
    validator = ParameterValidator()
    assert not validator.validate(loaded)
    assert any("spectrum_units" in e and "'Dl'" in e for e in validator.errors)


@pytest.mark.parametrize(
    "extra, needle",
    [
        ("\nspectrum_units: Dl\n", "uK*arcmin"),  # the baseline nl is a dict
        ("\nnl_units: Dl\n", "uK*arcmin"),
    ],
)
def test_dl_on_white_noise_levels_is_refused(tmp_path, extra, needle):
    """
    White-noise levels in uK*arcmin are a flat C_l by construction. Silently
    ignoring the key for them is how conventions get mixed; the run stops and
    says to write ``nl_units: Cl``.
    """
    params, _ = _write_params(tmp_path, "whitenoise", extra=extra)
    with pytest.raises(ValueError) as error:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            CovarianceMatrixGenerator(params).run_full_analysis()
    assert needle in str(error.value)
    assert "nl_units: Cl" in str(error.value)


def test_dl_on_a_constant_cmb_spectrum_is_refused(tmp_path):
    params, _ = _write_params(
        tmp_path,
        "constant",
        extra="\ncmb_spectrum: 1.0e-4\ncmb_spectrum_units: Dl\n",
    )
    with pytest.raises(ValueError) as error:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            CovarianceMatrixGenerator(params).run_full_analysis()
    assert "cmb_spectrum_units: Cl" in str(error.value)


def test_a_dimensionless_input_has_no_units_key(tmp_path):
    """``fl_units`` and friends do not exist; asking for them raises."""
    params, _ = _write_params(tmp_path, "nofl")
    generator = CovarianceMatrixGenerator(params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        generator.load_and_save_parameters()
    for name in ("fl", "hl", "beams", "pixwin", "post_process_correction"):
        with pytest.raises(ValueError, match="not a spectrum input"):
            generator.config.spectrum_units_of(name)
