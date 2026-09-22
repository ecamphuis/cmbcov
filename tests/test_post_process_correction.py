"""
``post_process_correction`` factors must reach the covariance.

Each covariance block ``Cov(C^{s1}_{f1}, C^{s2}_{f2})`` is multiplied, after the
PolSpice transform, by ``np.outer(d[f1][s1], d[f2][s2])``
(``CovariancePostProcessor.apply_debiasing``), where ``f1``/``f2`` are
frequency-pair keys (``"090GHz150GHz"``) and ``s1``/``s2`` Stokes-pair keys
(``"TE"``) -- the same keys ``SpectraLoader`` builds its dictionaries with.
The per-leg factor is

    d[f][s] = post_process_correction[f][s] / data_model[f][s] * tf_uncertainty[f][s]

(``data_model = B_f1 B_f2 pixwin_s fl``; ``tf_uncertainty`` only with
``add_tf_uncertainty``).

With width-1 bins the binning, ``D_ell`` scaling and the transform act
entrywise on the corrected block, so the covariance with corrections is the
covariance without them times the outer product of the leg factors, entry by
entry.
"""

import glob
import os
import warnings

import numpy as np
import pytest

healpy = pytest.importorskip("healpy")

from cmbcov import CovarianceMatrixGenerator  # noqa: E402
from cmbcov.keys import CovKeys  # noqa: E402
from cmbcov.spectra import SPECTRUM_FILE_LAYOUTS  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "data")
LMAX = 60
FREQS = ["090GHz", "150GHz"]
STOKES = ["T", "E"]
#: 9-column spectrum-file layout, as an {stokes key: column index} lookup --
#: pins the same TT,EE,BB,TE,TB,EB,ET,BT,BE column order the package reads.
_NINE_COLUMN_INDEX = {s: i for i, s in enumerate(SPECTRUM_FILE_LAYOUTS[9])}


def _run(tmp_path, name, extra=""):
    out = tmp_path / name
    template = open(os.path.join(DATA, "baseline_params.yml")).read()
    text = (
        template.replace("PLACEHOLDER_OUT", str(out))
        .replace("PLACEHOLDER_DATA", os.path.abspath(DATA))
        .replace("  - [2, 20, 6]\n  - [20, 60, 10]\n", "  - [2, 60, 1]\n")
    )
    params = tmp_path / f"{name}.yml"
    params.write_text(text + extra)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        CovarianceMatrixGenerator(str(params)).run_full_analysis()
    version = sorted(glob.glob(str(out / "v*")))[-1]
    return np.loadtxt(os.path.join(version, "covariance_matrix.dat"))


def _write_factor_files(directory, seed, constant=None):
    """One 9-column file per frequency pair; returns {pair: {stokes: factor}}."""
    rng = np.random.default_rng(seed)
    keys = CovKeys(STOKES, FREQS, exclude_asymmetric_stokes=False)
    factors = {}
    os.makedirs(directory, exist_ok=True)
    for pair in keys.combined_frequencies():
        table = np.empty((LMAX, 10))
        table[:, 0] = np.arange(LMAX)
        columns = (
            np.full((LMAX, 9), constant)
            if constant is not None
            else rng.uniform(0.5, 2.0, (LMAX, 9))
        )
        table[:, 1:] = columns
        np.savetxt(os.path.join(directory, f"ppc_{pair}.dat"), table)
        factors[pair] = {
            s: columns[:, _NINE_COLUMN_INDEX[s]] for s in keys.combined_stokes()
        }
    return os.path.join(directory, "ppc_{}.dat"), factors


def _leg_outer(factors, lmax=LMAX):
    """Outer product of the leg factors, laid out like the binned covariance."""
    keys = CovKeys(STOKES, FREQS, exclude_asymmetric_stokes=False)
    lmin = 2
    n = lmax - lmin
    expected = np.zeros((len(keys) * n, len(keys) * n))
    for cov_key in keys.keys():
        i, j = keys[cov_key]
        f1, f2 = cov_key.freqkey().split("x")
        p1, p2 = cov_key.stokekey().split("x")
        block = np.outer(factors[f1][p1][lmin:lmax], factors[f2][p2][lmin:lmax])
        expected[i * n : (i + 1) * n, j * n : (j + 1) * n] = block
        expected[j * n : (j + 1) * n, i * n : (i + 1) * n] = block.T
    return expected


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ppc_reference")
    return _run(tmp, "none")


def test_correction_multiplies_each_block_by_the_outer_product_of_leg_factors(
    tmp_path, reference
):
    template, factors = _write_factor_files(tmp_path / "ppc", seed=4)
    corrected = _run(tmp_path, "ppc", f"\npost_process_correction: {template}\n")
    expected = _leg_outer(factors)
    assert corrected.shape == reference.shape == expected.shape
    nonzero = reference != 0
    assert nonzero.mean() > 0.5
    np.testing.assert_allclose(
        corrected[nonzero], reference[nonzero] * expected[nonzero], rtol=1e-12
    )
    assert np.all(corrected[~nonzero] == 0)


def test_named_corrections_multiply_together(tmp_path, reference):
    first, f1 = _write_factor_files(tmp_path / "a", seed=5)
    second, f2 = _write_factor_files(tmp_path / "b", seed=6)
    combined = {pair: {s: f1[pair][s] * f2[pair][s] for s in f1[pair]} for pair in f1}
    corrected = _run(
        tmp_path,
        "two",
        f"\npost_process_correction:\n  hl: {first}\n  inpainting: {second}\n",
    )
    nonzero = reference != 0
    np.testing.assert_allclose(
        corrected[nonzero],
        reference[nonzero] * _leg_outer(combined)[nonzero],
        rtol=1e-12,
    )


def test_unit_correction_leaves_the_covariance_bit_identical(tmp_path, reference):
    template, _ = _write_factor_files(tmp_path / "ones", seed=0, constant=1.0)
    ones = _run(tmp_path, "ones", f"\npost_process_correction: {template}\n")
    assert ones.tobytes() == reference.tobytes()


def test_debiasing_factor_is_correction_over_data_model_times_tf_uncertainty(
    tmp_path,
):
    """Every factor once per leg, keyed by frequency pair and Stokes pair."""
    ppc_template, ppc = _write_factor_files(tmp_path / "ppc", seed=7)
    fl_template, fl = _write_factor_files(tmp_path / "fl", seed=8)
    for pair in fl:  # a transfer function lies in (0, 1]
        for s in fl[pair]:
            fl[pair][s] = fl[pair][s] / 2.0
    keys = CovKeys(STOKES, FREQS, exclude_asymmetric_stokes=False)
    for pair in keys.combined_frequencies():
        table = np.loadtxt(fl_template.format(pair))
        table[:, 1:] /= 2.0
        np.savetxt(fl_template.format(pair), table)

    out = tmp_path / "out"
    template = open(os.path.join(DATA, "baseline_params.yml")).read()
    params = tmp_path / "p.yml"
    params.write_text(
        template.replace("PLACEHOLDER_OUT", str(out)).replace(
            "PLACEHOLDER_DATA", os.path.abspath(DATA)
        )
        + f"\npost_process_correction: {ppc_template}\nfl: {fl_template}\n"
        + "add_tf_uncertainty: true\n"
    )
    generator = CovarianceMatrixGenerator(str(params))
    generator.load_and_save_parameters()
    generator.setup_frequency_stokes_mapping()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        generator.load_pre_process()
    generator.load_post_process()
    _, _, data_model, debiasing = generator.prepare_workflow()

    assert set(debiasing) == set(keys.combined_frequencies())
    for pair in keys.combined_frequencies():
        f1, f2 = pair[: len(pair) // 2], pair[len(pair) // 2 :]
        assert set(debiasing[pair]) == set(keys.combined_stokes())
        for s in keys.combined_stokes():
            beam = generator.beams[f1] * generator.beams[f2]
            dm = beam * generator.pix[s] * fl[pair][s]
            np.testing.assert_allclose(data_model[pair][s], dm, rtol=1e-14)
            tf = 1 + np.sqrt((1 - fl[pair][s]) / 3999)
            # 1 / data_model is taken as 1 where data_model vanishes (the
            # polarised pixel window at l < 2): safe_divide's default.
            with np.errstate(divide="ignore"):
                inverse = np.where(dm != 0, 1.0 / dm, 1.0)
            np.testing.assert_allclose(
                debiasing[pair][s], ppc[pair][s] * inverse * tf, rtol=1e-12
            )
