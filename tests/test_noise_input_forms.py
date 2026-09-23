"""
The three user-facing forms of ``nl``, and the convention behind them.

``nl`` is the noise power spectrum of the map *as delivered to the estimator*:
no beam, no pixel window, no transfer function, and never multiplied by the
data model. This is the MASTER convention (Hivon et al. 2002,
astro-ph/0105302, Eqs. (15)-(16)),

    <Ctilde_l> = M_ll' F_l' B^2_l' <C_l'> + <Ntilde_l>
    Delta C_l ~ (C_l + N_l / B^2_l) sqrt(2 / nu_l)

so the beam reaches the noise only through the debiasing, which divides each
leg by ``B1 B2 pix fl``.

The forms are:

1. ``{freq: sigma}``           -- temperature white-noise level, uK*arcmin
2. ``{freq: [sigma_T, sigma_P]}``
3. ``"path/nl_{}.txt"``        -- a tabulated noise power spectrum

Dict keys are *single* frequencies. The important test here is the last one:
a white-noise level and an equivalent flat tabulated file must now produce the
same ``nl_dict_biased``. Before this change the dict path multiplied the flat
level by the data model and the file path did not.
"""

import os
import warnings

import numpy as np
import pytest

healpy = pytest.importorskip("healpy")

from cmbcov import CovarianceMatrixGenerator, ParameterManager  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "data")

#: uK*arcmin -> uK*rad, the conversion a white-noise level goes through.
ARCMIN = np.pi / 10800.0

FREQS = ("090GHz", "150GHz")
AUTO_PAIRS = ("090GHz090GHz", "150GHz150GHz")
CROSS_PAIR = "090GHz150GHz"


def _write_params(tmp_path, name, nl_block):
    """
    The baseline parameter file with its ``nl`` block replaced by `nl_block`.

    Repeating the key in YAML is not enough here: the baseline block is a
    mapping, so the replacement is textual.
    """
    template = open(os.path.join(DATA, "baseline_params.yml")).read()
    text = template.replace("PLACEHOLDER_OUT", str(tmp_path / name)).replace(
        "PLACEHOLDER_DATA", os.path.abspath(DATA)
    )
    start = text.index("nl:\n")
    end = text.index("apodizetype:")
    text = text[:start] + nl_block.rstrip("\n") + "\n" + text[end:]
    path = tmp_path / f"{name}.yml"
    path.write_text(text)
    return str(path)


def _load(tmp_path, name, nl_block):
    """Run as far as the data model; return the SpectraLoader."""
    params = _write_params(tmp_path, name, nl_block)
    generator = CovarianceMatrixGenerator(params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        generator.load_and_save_parameters()
        generator.setup_frequency_stokes_mapping()
        generator.load_pre_process()
        generator.load_post_process()
        generator.spectra.prepare_workflow()
    return generator.spectra


# ---------------------------------------------------------------------------
# Form 1 vs form 2
# ---------------------------------------------------------------------------


def test_one_number_equals_two_numbers_with_sigma_p_root_two(tmp_path):
    """
    Form 1 is the ``sigma_P = sqrt(2) sigma_T`` special case of form 2: the
    factor 2 in power is the usual sqrt(2) in amplitude for Q/U.
    """
    one = _load(tmp_path, "one", "nl:\n  090GHz: 20.0\n  150GHz: 9.0\n")
    two = _load(
        tmp_path,
        "two",
        "nl:\n"
        f"  090GHz: [20.0, {20.0 * np.sqrt(2):.17g}]\n"
        f"  150GHz: [9.0, {9.0 * np.sqrt(2):.17g}]\n",
    )
    for pair in AUTO_PAIRS:
        for stokes in one.nl_dict[pair]:
            np.testing.assert_allclose(
                two.nl_dict[pair][stokes],
                one.nl_dict[pair][stokes],
                rtol=1e-12,
                atol=0.0,
                err_msg=f"{pair} {stokes}",
            )


def test_two_numbers_set_T_and_P_independently(tmp_path):
    """The two levels are squared on their own; TE stays zero."""
    loader = _load(tmp_path, "indep", "nl:\n  090GHz: [20.0, 5.0]\n  150GHz: 9.0\n")
    nl = loader.nl_dict["090GHz090GHz"]
    np.testing.assert_allclose(nl["TT"], (20.0 * ARCMIN) ** 2, rtol=1e-13)
    np.testing.assert_allclose(nl["EE"], (5.0 * ARCMIN) ** 2, rtol=1e-13)
    np.testing.assert_array_equal(nl["TE"], np.zeros_like(nl["TE"]))
    # and the other frequency, given as a single number, keeps the factor 2
    other = loader.nl_dict["150GHz150GHz"]
    np.testing.assert_allclose(other["EE"], 2 * (9.0 * ARCMIN) ** 2, rtol=1e-13)


def test_one_number_uses_the_polarisation_factor_two(tmp_path):
    loader = _load(tmp_path, "form1", "nl:\n  090GHz: 20.0\n  150GHz: 9.0\n")
    nl = loader.nl_dict["090GHz090GHz"]
    np.testing.assert_allclose(nl["TT"], (20.0 * ARCMIN) ** 2, rtol=1e-13)
    np.testing.assert_allclose(nl["EE"], 2 * (20.0 * ARCMIN) ** 2, rtol=1e-13)


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------


def test_cross_frequency_noise_is_zero_and_silent(tmp_path):
    """
    Map noise is uncorrelated between bands, so a cross pair is zero: the
    normal case, not a missing input, and therefore no warning.
    """
    params = _write_params(tmp_path, "cross", "nl:\n  090GHz: 20.0\n  150GHz: 9.0\n")
    generator = CovarianceMatrixGenerator(params)
    generator.load_and_save_parameters()
    generator.setup_frequency_stokes_mapping()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        generator.load_pre_process()
    noise_warnings = [
        w for w in caught if "noiseless" in str(w.message) or "nl" in str(w.message)
    ]
    assert not noise_warnings, [str(w.message) for w in noise_warnings]

    cross = generator.spectra.nl_dict[CROSS_PAIR]
    for stokes, values in cross.items():
        np.testing.assert_array_equal(values, np.zeros_like(values), err_msg=stokes)


def test_pair_spelled_key_raises_with_a_helpful_message(tmp_path):
    """The pre-change spelling must say what replaced it, not just 'unknown'."""
    params = _write_params(
        tmp_path, "pairkey", "nl:\n  090GHz090GHz: 20.0\n  150GHz150GHz: 9.0\n"
    )
    generator = CovarianceMatrixGenerator(params)
    generator.load_and_save_parameters()
    generator.setup_frequency_stokes_mapping()
    with pytest.raises(ValueError, match="frequency \\*pair\\*"):
        generator.load_pre_process()
    try:
        generator.load_pre_process()
    except ValueError as exc:
        assert "090GHz" in str(exc)
        assert "single frequency" in str(exc)


def test_unknown_key_names_the_expected_frequencies(tmp_path):
    params = _write_params(tmp_path, "unknown", "nl:\n  353GHz: 20.0\n")
    generator = CovarianceMatrixGenerator(params)
    generator.load_and_save_parameters()
    generator.setup_frequency_stokes_mapping()
    with pytest.raises(ValueError, match="Unrecognised"):
        generator.load_pre_process()
    try:
        generator.load_pre_process()
    except ValueError as exc:
        for freq in FREQS:
            assert freq in str(exc)


def test_missing_frequency_still_warns(tmp_path):
    params = _write_params(tmp_path, "missing", "nl:\n  090GHz: 20.0\n")
    generator = CovarianceMatrixGenerator(params)
    generator.load_and_save_parameters()
    generator.setup_frequency_stokes_mapping()
    with pytest.warns(UserWarning, match="noiseless"):
        generator.load_pre_process()
    nl = generator.spectra.nl_dict["150GHz150GHz"]
    np.testing.assert_array_equal(nl["TT"], np.zeros_like(nl["TT"]))


def test_three_entries_is_an_error(tmp_path):
    params = _write_params(tmp_path, "three", "nl:\n  090GHz: [1.0, 2.0, 3.0]\n")
    generator = CovarianceMatrixGenerator(params)
    generator.load_and_save_parameters()
    generator.setup_frequency_stokes_mapping()
    with pytest.raises(ValueError, match="3 entries"):
        generator.load_pre_process()


def test_dl_units_with_a_dict_still_raises(tmp_path):
    """Levels are a flat C_l by construction; the validator says so first."""
    params = _write_params(tmp_path, "dlunits", "nl:\n  090GHz: 20.0\nnl_units: Dl\n")
    with pytest.raises(ValueError, match="C_l by construction"):
        ParameterManager(params).load_and_validate()


# ---------------------------------------------------------------------------
# The convention: a level and an equivalent flat file agree
# ---------------------------------------------------------------------------


def test_level_and_equivalent_flat_file_give_the_same_biased_noise(tmp_path):
    """
    The convention fix, and the one that matters.

    A white-noise level of sigma uK*arcmin is a flat ``C_l = (sigma pi/10800)^2``
    with ``N^EE = N^BB = 2 N^TT``. Written out as a tabulated ``nl`` file, it
    must reach the covariance as the same ``nl_dict_biased``: neither path
    multiplies the noise by the beam/pixel-window data model.

    Before this change the dict path set ``noise_is_biased = False`` and so
    multiplied the flat level by ``B^2 pix fl``, while the file path did not --
    the two disagreed by exactly that factor.
    """
    sigma = {"090GHz": 20.0, "150GHz": 9.0}
    lmax = 60
    ell = np.arange(lmax + 1)
    for pair in (*AUTO_PAIRS, CROSS_PAIR):
        f1, f2 = pair[: len(pair) // 2], pair[len(pair) // 2 :]
        n_tt = (sigma[f1] * ARCMIN) ** 2 if f1 == f2 else 0.0
        table = np.column_stack(
            [
                ell,
                np.full(ell.shape, n_tt),  # TT
                np.full(ell.shape, 2 * n_tt),  # EE
                np.full(ell.shape, 2 * n_tt),  # BB
                np.zeros(ell.shape),  # TE
            ]
        )
        np.savetxt(tmp_path / f"nlflat_{pair}.dat", table, fmt="%.17g")

    from_level = _load(tmp_path, "level", "nl:\n  090GHz: 20.0\n  150GHz: 9.0\n")
    from_file = _load(tmp_path, "flatfile", f"nl: {tmp_path}/nlflat_{{}}.dat\n")

    for pair in (*AUTO_PAIRS, CROSS_PAIR):
        for stokes in from_level.nl_dict_biased[pair]:
            np.testing.assert_allclose(
                from_file.nl_dict_biased[pair][stokes],
                from_level.nl_dict_biased[pair][stokes],
                rtol=1e-12,
                atol=0.0,
                err_msg=f"{pair} {stokes}",
            )


def test_noise_is_never_multiplied_by_the_data_model(tmp_path):
    """
    ``nl_dict_biased`` is ``nl_dict`` unchanged -- the beam reaches the noise
    only through the debiasing (MASTER Eq. (16): N_l / B^2_l).
    """
    loader = _load(tmp_path, "nomodel", "nl:\n  090GHz: 20.0\n  150GHz: 9.0\n")
    for pair, spectra in loader.nl_dict.items():
        for stokes, values in spectra.items():
            np.testing.assert_array_equal(
                loader.nl_dict_biased[pair][stokes],
                values,
                err_msg=f"{pair} {stokes}",
            )
    # and the data model really is non-trivial here (5' and 2' beams, nside 32)
    model = loader.data_model["090GHz090GHz"]["TT"]
    assert not np.allclose(model, np.ones_like(model))


# ---------------------------------------------------------------------------
# The removed key
# ---------------------------------------------------------------------------


def test_nl_is_biased_is_refused_by_validation(tmp_path):
    params = _write_params(
        tmp_path, "removed", "nl:\n  090GHz: 20.0\nnl_is_biased: true\n"
    )
    with pytest.raises(ValueError) as excinfo:
        ParameterManager(params).load_and_validate()
    message = str(excinfo.value)
    assert "nl_is_biased" in message
    assert "removed" in message
    assert "debiasing" in message
