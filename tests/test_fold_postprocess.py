r"""
Post-processing through leg projections
(:meth:`~cmbcov.covariance.Cov._process_covariance_term`).

PolSpice transform, D_ell scaling, debiasing and binning are all linear on
each leg of a covariance block, so a block is post-processed as
$P_L C P_R^T$ with $P = B\,\mathrm{diag}(\delta d)\,G$ built once per leg,
rather than by two dense $\ell_{\max}^3$ products per block. This file pins that the two
agree to round-off (the per-block reference,
``Cov._process_covariance_term_reference``, is what the projections
replace), for every kind of run that reaches the post-processing:

* NKA and INKA over T and E, TT alone and BB alone (the BB-only escape hatch);
* ACC over T and E, TT alone, and the runs with a B observable, whose PolSpice
  transform mixes EE and BB;
* with the PolSpice transform on and off, ``Dl`` on and off, per-multipole
  debiasing (``post_process_correction`` and ``add_tf_uncertainty`` reach the
  covariance this way) and none, ``sum_asymmetric_stokes`` on and off, binned
  and unbinned output;
* that the projections of one call are not reused by the next, and that
  a run only takes them when they pay.

"Agree" is the largest difference over the largest element of the matrix,
$10^{-12}$ here; measured, it is a few $10^{-15}$.
"""

import glob
import os
import shutil
import warnings

import numpy as np
import pytest
import yaml

healpy = pytest.importorskip("healpy")

from cmbcov import CovarianceMatrixGenerator  # noqa: E402
from cmbcov.covariance import Cov, CovarianceConfig, CovarianceMethod  # noqa: E402
from cmbcov.keys import CovKeys  # noqa: E402
from cmbcov.postprocess import CovariancePostProcessor  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
LMAX = 60
FREQS = ["090GHz", "150GHz"]
TOLERANCE = 1e-12


def _difference(folded, reference):
    """Largest difference relative to the largest element of the reference."""
    return np.abs(folded - reference).max() / np.abs(reference).max()


# --------------------------------------------------------------------------- #
# NKA and INKA, directly on Cov
# --------------------------------------------------------------------------- #


def _spectra(keys, size, seed):
    """Positive, smooth, random spectra for every (frequency pair, spectrum)."""
    rng = np.random.default_rng(seed)
    ell = np.arange(size)
    base = np.zeros(size)
    base[2:] = 1e-2 / (ell[2:] * (ell[2:] + 1)) ** 0.9
    cl = {}
    for freq in keys.combined_frequencies():
        cl[freq] = {
            stokes: base * rng.uniform(0.5, 2.0) * (1 + 0.1 * rng.random(size))
            for stokes in ("TT", "EE", "BB", "TE", "ET")
        }
    return cl


def _debiasing(keys, size, seed):
    """A different factor at every multipole for every leg."""
    rng = np.random.default_rng(seed)
    return {
        freq: {
            stokes: rng.uniform(0.5, 2.0, size)
            for stokes in ("TT", "EE", "BB", "TE", "ET")
        }
        for freq in keys.combined_frequencies()
    }


def _cov(method=CovarianceMethod.NKA, lmax=LMAX, **config):
    config = CovarianceConfig(method=method, lmax=lmax, lmin=2, **config)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Cov("baseline_mask.fits", config=config, mask_path=DATA, save_dir="/tmp")


def _matrix(cov, fold, lbins, keys, cl):
    cov._fold_postprocess = fold
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cov.compute_covariance_matrix(lbins, keys, cl)[2]


BINNED = [2, 12, 22, 40, 60]


@pytest.mark.parametrize("method", [CovarianceMethod.NKA, CovarianceMethod.INKA])
@pytest.mark.parametrize("polspice", [True, False], ids=["polspice", "no_polspice"])
@pytest.mark.parametrize("Dl", [True, False], ids=["Dl", "Cl"])
@pytest.mark.parametrize("debias", [True, False], ids=["debiased", "plain"])
@pytest.mark.parametrize("lbins", [BINNED, 1], ids=["binned", "unbinned"])
def test_te_projections_match_the_per_block_reference(
    method, polspice, Dl, debias, lbins
):
    keys = CovKeys(["T", "E"], FREQS)
    cl = _spectra(keys, LMAX, seed=1)
    cov = _cov(
        method,
        polspice_postprocess=polspice,
        Dl=Dl,
        debiasing_dict=_debiasing(keys, LMAX, seed=2) if debias else None,
    )
    folded = _matrix(cov, True, lbins, keys, cl)
    reference = _matrix(cov, False, lbins, keys, cl)
    assert folded.shape == reference.shape
    assert _difference(folded, reference) < TOLERANCE
    # the reference is not the identity: the two really are different code
    assert np.abs(reference).max() > 0


@pytest.mark.parametrize("sum_asymmetric", [True, False], ids=["summed", "TE_and_ET"])
def test_asymmetric_stokes_sum_is_unchanged(sum_asymmetric):
    keys = CovKeys(["T", "E"], FREQS, exclude_asymmetric_stokes=False)
    cl = _spectra(keys, LMAX, seed=3)
    cov = _cov(
        Dl=True,
        sum_asymmetric_stokes=sum_asymmetric,
        debiasing_dict=_debiasing(keys, LMAX, seed=4),
    )
    folded = _matrix(cov, True, BINNED, keys, cl)
    reference = _matrix(cov, False, BINNED, keys, cl)
    assert _difference(folded, reference) < TOLERANCE


def test_tt_only_projections_match_the_per_block_reference():
    keys = CovKeys(["T"], FREQS)
    cl = _spectra(keys, LMAX, seed=5)
    cov = _cov(Dl=True, debiasing_dict=_debiasing(keys, LMAX, seed=6))
    folded = _matrix(cov, True, BINNED, keys, cl)
    reference = _matrix(cov, False, BINNED, keys, cl)
    assert _difference(folded, reference) < TOLERANCE


@pytest.mark.parametrize("method", [CovarianceMethod.NKA, CovarianceMethod.INKA])
@pytest.mark.parametrize("polspice", [True, False], ids=["polspice", "no_polspice"])
def test_bb_only_escape_hatch_matches_the_per_block_reference(method, polspice):
    """BB alone under NKA/INKA takes the diagonal transform, G+ on BB."""
    keys = CovKeys([], ["090GHz"], observables=["BB"])
    cl = _spectra(keys, LMAX, seed=7)
    cov = _cov(
        method,
        polspice_postprocess=polspice,
        Dl=True,
        debiasing_dict=_debiasing(keys, LMAX, seed=8),
    )
    folded = _matrix(cov, True, BINNED, keys, cl)
    reference = _matrix(cov, False, BINNED, keys, cl)
    assert _difference(folded, reference) < TOLERANCE


# --------------------------------------------------------------------------- #
# The projection of a leg
# --------------------------------------------------------------------------- #


def test_projection_is_bin_scaling_and_kernel_applied_in_order():
    rng = np.random.default_rng(9)
    n, nb = 30, 5
    g = rng.normal(size=(n, n))
    b = rng.uniform(size=(nb, n))
    d = rng.uniform(0.5, 2.0, n)
    c = rng.normal(size=(n, n))
    g2 = rng.normal(size=(n, n))
    d2 = rng.uniform(0.5, 2.0, n)
    post = CovariancePostProcessor({})

    scaling = post.leg_scaling("f", "TT", n, True, {"f": {"TT": d}})
    ell = np.arange(n)
    np.testing.assert_allclose(scaling, ell * (ell + 1) / (2 * np.pi) * d)
    assert post.leg_scaling("f", "TT", n, False, None) is None
    assert post.leg_scaling("f", "TT", n, False, {}) is None
    np.testing.assert_array_equal(
        post.leg_scaling("f", "TT", n, False, {"f": {"TT": d}}), d
    )

    left = post.leg_projection(g, d, b, n)
    right = post.leg_projection(g2, d2, b, n)
    by_hand = b @ np.diag(d) @ (g @ c @ g2.T) @ np.diag(d2) @ b.T
    np.testing.assert_allclose(post.project(left, c, right), by_hand, rtol=1e-12)

    # no kernel, no scaling, no bins: the identity
    np.testing.assert_array_equal(post.leg_projection(None, None, None, n), np.eye(n))
    # unbinned with a kernel and a scaling: rows scaled, no bin matrix
    np.testing.assert_allclose(
        post.leg_projection(g, d, None, n), np.diag(d) @ g, rtol=1e-14
    )


# --------------------------------------------------------------------------- #
# The projections of one call stay with that call
# --------------------------------------------------------------------------- #


def test_projections_do_not_leak_between_calls():
    """
    One ``Cov`` computes twice, with different spectra, different debiasing
    (of the same shape and keys) and a different binning; each call must
    equal the reference on its own inputs. A projection kept from the first
    call would reproduce the first debiasing in the second.
    """
    keys = CovKeys(["T", "E"], FREQS)
    cov = _cov(Dl=True)

    cl_one, cl_two = _spectra(keys, LMAX, seed=10), _spectra(keys, LMAX, seed=11)
    cov.config.debiasing_dict = _debiasing(keys, LMAX, seed=12)
    first = _matrix(cov, True, BINNED, keys, cl_one)
    first_reference = _matrix(cov, False, BINNED, keys, cl_one)

    cov.config.debiasing_dict = _debiasing(keys, LMAX, seed=13)
    second = _matrix(cov, True, BINNED, keys, cl_two)
    second_reference = _matrix(cov, False, BINNED, keys, cl_two)

    cov.config.debiasing_dict = None
    other_bins = [2, 30, 60]
    third = _matrix(cov, True, other_bins, keys, cl_two)
    third_reference = _matrix(cov, False, other_bins, keys, cl_two)

    assert _difference(first, first_reference) < TOLERANCE
    assert _difference(second, second_reference) < TOLERANCE
    assert _difference(third, third_reference) < TOLERANCE
    # and the three really differ, so agreement is not agreement on one thing
    assert _difference(second[:, :], first) > 1e-3


def test_a_projection_is_built_once_per_leg(monkeypatch):
    """Nine blocks of a T/E run, two frequencies: one projection per leg."""
    built = []
    real = CovariancePostProcessor.leg_projection

    def counting(kernel, scaling, bin_matrix, lmax):
        built.append(1)
        return real(kernel, scaling, bin_matrix, lmax)

    monkeypatch.setattr(
        CovariancePostProcessor, "leg_projection", staticmethod(counting)
    )
    keys = CovKeys(["T", "E"], FREQS)
    cl = _spectra(keys, LMAX, seed=14)
    cov = _cov(Dl=True, debiasing_dict=_debiasing(keys, LMAX, seed=15))
    _matrix(cov, True, BINNED, keys, cl)
    legs = {
        (freq, stokes)
        for cov_key in keys.keys()
        for freq, stokes in zip(
            cov_key.freqkey().split("x"), cov_key.stokekey().split("x"), strict=True
        )
    }
    assert len(built) == len(legs) < 2 * len(list(keys.keys()))


# --------------------------------------------------------------------------- #
# When a run takes the projections
# --------------------------------------------------------------------------- #


def test_a_run_takes_the_projections_only_when_they_pay(monkeypatch):
    calls = []
    real = Cov._process_covariance_term

    def spy(self, *args, **kwargs):
        calls.append(1)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Cov, "_process_covariance_term", spy)
    keys = CovKeys(["T"], ["090GHz"])
    cl = _spectra(keys, LMAX, seed=16)

    def run(lbins, **config):
        calls.clear()
        _matrix(_cov(**config), None, lbins, keys, cl)
        return len(calls)

    assert run(BINNED) == 1  # 4 bins over 60 multipoles: 15 per bin
    assert run([2, 30, 60]) == 1  # 2 bins
    assert run(1) == 0  # unbinned: the projections would be as large as the blocks
    # 19 bins of 3 multipoles: fewer than the 4 per bin the rule asks for
    assert run(list(range(2, 61, 3))) == 0
    assert run(BINNED, polspice_postprocess=False) == 0  # nothing to fold


def test_automatic_choice_is_the_projections_when_binned_and_the_reference_otherwise():
    keys = CovKeys(["T", "E"], FREQS)
    cl = _spectra(keys, LMAX, seed=17)
    cov = _cov(Dl=True, debiasing_dict=_debiasing(keys, LMAX, seed=18))
    automatic = _matrix(cov, None, BINNED, keys, cl)
    np.testing.assert_array_equal(automatic, _matrix(cov, True, BINNED, keys, cl))
    automatic = _matrix(cov, None, 1, keys, cl)
    np.testing.assert_array_equal(automatic, _matrix(cov, False, 1, keys, cl))


# --------------------------------------------------------------------------- #
# ACC: T/E, TT alone and the B-mode runs (EE/BB mixing)
# --------------------------------------------------------------------------- #

SIX = ["TT", "EE", "BB", "TE", "TB", "EB"]


def _write_cls(path, nonzero_odd):
    n = 80
    ell = np.arange(n)
    tt = np.zeros(n)
    tt[2:] = 1e3 / (ell[2:] * (ell[2:] + 1)) ** 0.9
    ee, bb = 0.1 * tt, 0.02 * tt
    te = 0.5 * np.sqrt(tt * ee)
    tb = 0.05 * np.sqrt(tt * bb) if nonzero_odd else 0 * tt
    eb = 0.05 * np.sqrt(ee * bb) if nonzero_odd else 0 * tt
    np.savetxt(path, np.column_stack([ell, tt, ee, bb, te, tb, eb, te, tb, eb]))


def _write_ppc(directory, freqs, size=80):
    """Per-multipole ``post_process_correction`` files, one per frequency pair."""
    rng = np.random.default_rng(19)
    keys = CovKeys([], freqs, observables=["TT"])
    os.makedirs(directory, exist_ok=True)
    for pair in keys.combined_frequencies():
        table = np.column_stack(
            [np.arange(size)] + [rng.uniform(0.5, 2.0, size) for _ in range(9)]
        )
        np.savetxt(os.path.join(directory, f"ppc_{pair}.dat"), table)
    return os.path.join(directory, "ppc_{}.dat")


def _acc_pair(workdir, freqs, polspice=True, sum_asymmetric=False, **extra):
    """Run one ACC parameter file with the projections and with the reference."""
    shutil.copy(os.path.join(DATA, "baseline_mask.fits"), workdir)
    _write_cls(os.path.join(workdir, "cls.dat"), nonzero_odd=True)
    params = {
        "cov_path": os.path.join(workdir, "out"),
        "cov_name": "cov.dat",
        "frequencies": freqs,
        "lmax": 20,
        "lmin": 2,
        "bins": [[2, 20, 3]],
        "mask_name": "baseline_mask.fits",
        "mask_path": workdir,
        "covariance_approximation": "acc",
        "dmax": 3,
        "centralell": 10,
        "cmb_spectrum": os.path.join(workdir, "cls.dat"),
        "beams": dict.fromkeys(freqs, 5.0),
        "pixwin": 32,
        "nl": dict.fromkeys(freqs, 10.0),
        "polspice_postprocess": polspice,
        "sum_asymmetric_stokes": sum_asymmetric,
        "Dl": True,
        "post_process_correction": _write_ppc(os.path.join(workdir, "ppc"), freqs),
        "acc_precompute": {"nside": 16, "grid": "gl", "lw": 10},
    }
    params.update(extra)
    path = os.path.join(workdir, "params.yml")
    with open(path, "w") as handle:
        yaml.dump(params, handle)
    CovarianceMatrixGenerator(path).precompute_acc_kernels()

    matrices = {}
    for fold in (True, False):
        Cov._fold_postprocess = fold
        try:
            CovarianceMatrixGenerator(path).run_full_analysis()
        finally:
            Cov._fold_postprocess = None
        newest = max(
            glob.glob(os.path.join(workdir, "out", "v*")),
            key=lambda d: int(os.path.basename(d)[1:]),
        )
        matrices[fold] = np.loadtxt(os.path.join(newest, "cov.dat"))
    return matrices[True], matrices[False]


@pytest.mark.parametrize(
    "label, extra, freqs, polspice, sum_asymmetric",
    [
        ("te_two_frequencies", {"stokes": ["T", "E"]}, FREQS, True, False),
        ("te_summed_te_et", {"stokes": ["T", "E"]}, FREQS, True, True),
        ("te_no_polspice", {"stokes": ["T", "E"]}, FREQS[:1], False, False),
        ("tt_only", {"stokes": ["T"]}, FREQS, True, False),
        ("b_level2", {"observables": ["TT", "EE", "TE", "BB"]}, FREQS[:1], True, False),
        ("b_level3_tb_eb", {"observables": SIX}, FREQS[:1], True, False),
        (
            "b_level4_parity_mixed",
            {"observables": SIX, "parity_mixed_blocks": True},
            FREQS[:1],
            True,
            False,
        ),
        (
            "b_two_frequencies_summed",
            {"observables": ["TT", "EE", "TE", "BB"]},
            FREQS,
            True,
            True,
        ),
        ("b_no_polspice", {"observables": SIX}, FREQS[:1], False, False),
    ],
)
def test_acc_runs_match_the_per_block_reference(
    tmp_path, label, extra, freqs, polspice, sum_asymmetric
):
    folded, reference = _acc_pair(
        str(tmp_path),
        freqs,
        polspice=polspice,
        sum_asymmetric=sum_asymmetric,
        **extra,
    )
    assert folded.shape == reference.shape
    assert np.isfinite(folded).all()
    difference = _difference(folded, reference)
    print(f"{label}: max difference / max element = {difference:.2e}")
    assert difference < TOLERANCE
