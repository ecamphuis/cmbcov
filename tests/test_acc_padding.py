"""
The ACC term is computed with the spectra read past ``lmax``, to
``lmax_int = lmax + max(0, S - 1 - centralell)`` (``acc_window_pad``), and
reported to ``lmax``.

Element ``(ell1, ell2)`` uses the kernel of ``(l*, l* + |ell2 - ell1|)``
translated by ``shift = min(ell1, ell2) - l*`` (Eq. 33): kernel index ``i``
stands for ``L = shift + i``. Padding the spectra to ``lmax_int`` avoids
losing kernel weight for elements within ``S - 1 - l*`` of ``lmax`` (up to
92% of the kernel weight at ``ell = lmax - 1`` on the survey ``l* = 250``,
``S = 512`` cache).

Pinned here:

- no reported element's window reaches past ``lmax_int``, and
  ``lmax_int - 1`` would cut the top element: the pad is exact, not merely
  sufficient;
- with unit spectra and unit ``norm_Xi`` every element whose window does not
  run below ``L = 0`` sums its whole unit-normalised kernel, up to ``lmax - 1``;
- the reported covariance is the truncation of the ACC term computed on
  ``lmax_int``.

``tests/test_acc_vs_exact.py`` checks an entry next to ``lmax`` against the
exact covariance.
"""

import itertools
import os
import textwrap
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from cmbcov.approximations.acc import (
    ACCStrategy,
    acc_internal_lmax,
    acc_window_pad,
    precompute_acc_kernels,
)
from cmbcov.covariance import Cov, CovarianceConfig, CovarianceMethod
from cmbcov.keys import CovKey, CovKeys

FREQ = "090GHz"


class _FakeKernelStrategy(ACCStrategy):
    """ACCStrategy with in-memory kernels instead of the on-disk cache."""

    def __init__(self, cov, kernels):
        super().__init__(cov)
        self._fake_kernels = kernels

    def get_covariance_coupling(self, ell, ell_prime, pairs=None):
        table = self._fake_kernels[ell_prime - ell]
        return dict.fromkeys(pairs or [("TT", "TT")], table)


def _cov(lmax, centralell, dmax, norm):
    config = SimpleNamespace(
        method=CovarianceMethod.ACC, dmax=dmax, centralell=centralell
    )
    return SimpleNamespace(lmax=lmax, config=config, norm_Xi={("TT", "TT"): norm})


@pytest.mark.parametrize(
    "lmax, size, centralell",
    [(700, 512, 250), (3000, 512, 250), (48, 32, 16), (90, 64, 40), (60, 16, 30)],
)
def test_pad_is_exactly_what_the_top_window_needs(lmax, size, centralell):
    pad = acc_window_pad(size, centralell)
    lmax_int = acc_internal_lmax(lmax, size, centralell)
    assert lmax_int == lmax + pad == lmax + max(0, size - 1 - centralell)
    # Window of min(ell1, ell2) = m covers L = m - l* .. m - l* + S - 1.
    ms = np.arange(lmax)
    last_index = ms - centralell + size - 1
    assert last_index.max() == lmax_int - 1 or pad == 0
    assert np.all(last_index < lmax_int)  # nothing reported is cut
    if pad > 0:
        assert last_index.max() >= lmax_int - 1  # one multipole fewer would cut
    # Survey cache: the pad is 261, not nside_acc = 256.
    if (size, centralell) == (512, 250):
        assert pad == 261


@pytest.mark.parametrize(
    "lmax, size, centralell, dmax", [(90, 64, 40, 5), (40, 64, 30, 3), (70, 16, 0, 2)]
)
def test_no_reported_element_is_clipped_at_lmax_int(lmax, size, centralell, dmax):
    """
    Unit spectra of exactly ``lmax_int`` multipoles and unit ``norm_Xi``: an
    element is the kernel weight inside its window, so every element whose
    window starts at ``L >= 0`` must sum the whole unit kernel -- including
    the last row, which the pre-padding clip cut hardest.
    """
    rng = np.random.default_rng(size + centralell)
    kernels = {d: rng.random((size, size)) for d in range(dmax)}
    lmax_int = acc_internal_lmax(lmax, size, centralell)
    cl = {FREQ + FREQ: {"TT": np.ones(lmax_int)}}
    cov = _cov(lmax, centralell, dmax, np.ones((lmax, lmax)))
    key = CovKey(("T",) * 4, (FREQ,) * 4)
    sigma = _FakeKernelStrategy(cov, kernels).compute_covariance_term(key, cl)

    for d in range(dmax):
        for ell1 in range(max(0, centralell), lmax - d):
            # two identical Wick contractions, each sums to 1
            assert sigma[ell1, ell1 + d] == pytest.approx(2.0, rel=1e-12), (d, ell1)
            assert sigma[ell1 + d, ell1] == sigma[ell1, ell1 + d]

    # The same run with the spectra cut at lmax (the old behaviour) loses
    # weight at the top: this is the bias the padding removes.
    cut = {FREQ + FREQ: {"TT": np.r_[np.ones(lmax), np.zeros(lmax_int - lmax)]}}
    clipped = _FakeKernelStrategy(cov, kernels).compute_covariance_term(key, cut)
    if acc_window_pad(size, centralell) > 0:
        assert clipped[lmax - 1, lmax - 1] < 2.0 - 1e-3


def test_reported_covariance_is_the_truncation_of_the_term_at_lmax_int():
    lmax, size, centralell, dmax = 60, 32, 12, 4
    lmax_int = acc_internal_lmax(lmax, size, centralell)  # 79
    rng = np.random.default_rng(5)
    kernels = {
        d: {
            pair: rng.random((size, size))
            for pair in itertools.product(("TT", "DD", "TD", "DT"), repeat=2)
        }
        for d in range(dmax)
    }
    # norm_Xi of the reported run is the lmax corner of the lmax_int one:
    # the ACC term reads norm_Xi only for ell1, ell2 < lmax.
    norm_int = {
        pair: 0.5 + rng.random((lmax_int, lmax_int))
        for pair in itertools.product(("TT", "EE", "TE", "ET"), repeat=2)
    }
    norm = {pair: array[:lmax, :lmax].copy() for pair, array in norm_int.items()}
    # Spectra long enough for the lmax_int run's own padding.
    n = acc_internal_lmax(lmax_int, size, centralell)
    freqs = (FREQ, "150GHz")
    cl = {
        a + b: {s: rng.random(n) for s in ("TT", "EE", "TE", "ET")}
        for a, b in itertools.product(freqs, repeat=2)
    }

    class Strategy(ACCStrategy):
        def get_covariance_coupling(self, ell, ell_prime, pairs=None):
            table = kernels[ell_prime - ell]
            return {pair: table[pair] for pair in pairs}

    def run(report_lmax, norm_xi, key):
        config = SimpleNamespace(
            method=CovarianceMethod.ACC, dmax=dmax, centralell=centralell
        )
        cov = SimpleNamespace(lmax=report_lmax, config=config, norm_Xi=norm_xi)
        return Strategy(cov).compute_covariance_term(key, cl)

    for key in (
        CovKey(("T",) * 4, (FREQ,) * 4),
        CovKey(("T", "E", "T", "E"), (FREQ, FREQ, "150GHz", FREQ)),
        CovKey(("E", "E", "T", "E"), (FREQ, "150GHz", "150GHz", FREQ)),
    ):
        reported = run(lmax, norm, key)
        full = run(lmax_int, norm_int, key)
        assert reported.shape == (lmax, lmax)
        np.testing.assert_allclose(
            reported, full[:lmax, :lmax], rtol=1e-14, atol=0.0, err_msg=str(key)
        )


# --------------------------------------------------------------------------- #
# Cov and the parameter-file pipeline: lmax_int is required, and bounded
# --------------------------------------------------------------------------- #
DATA = os.path.join(os.path.dirname(__file__), "data")
NSIDE, ELL = 16, 16  # kernels 32 x 32 -> pad 15; the nside-32 mask allows 64


@pytest.fixture(scope="module")
def kernel_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("kernels")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        precompute_acc_kernels(
            "baseline_mask.fits",
            str(directory),
            centralell=ELL,
            dmax=2,
            mask_path=DATA,
            nside=NSIDE,
            grid="gl",
            lw=10,
            spectra=("TT",),
        )
    return str(directory)


def _acc(kernel_dir, lmax):
    config = CovarianceConfig(
        method=CovarianceMethod.ACC, lmax=lmax, dmax=2, centralell=ELL
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Cov(
            "baseline_mask.fits",
            config=config,
            mask_path=DATA,
            save_dir=kernel_dir,
            acc_kernel_dir=kernel_dir,
        )


def _tt(size):
    ell = np.arange(size)
    tt = np.zeros(size)
    tt[2:] = 1e-3 / (ell[2:] * (ell[2:] + 1)) ** 0.9
    return {FREQ + FREQ: {"TT": tt}}


def test_cov_reports_lmax_int_and_rejects_shorter_spectra(kernel_dir):
    cov = _acc(kernel_dir, 48)
    assert cov.acc_internal_lmax() == 48 + 15
    keys = CovKeys(["T"], [FREQ])
    with pytest.raises(ValueError, match="has 62 multipoles; the ACC method needs 63"):
        cov.compute_covariance_matrix([2, 20, 48], keys, _tt(62))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ells, _, sigma = cov.compute_covariance_matrix([2, 20, 48], keys, _tt(63))
    assert sigma.shape == (2, 2) and np.all(np.isfinite(sigma))


def test_lmax_int_may_exceed_two_nside_of_the_mask(kernel_dir):
    """
    Only the reported lmax is bounded by 2 * nside (64 here); lmax_int is
    bounded only by how far the spectra reach.
    """
    cov = _acc(kernel_dir, 64)
    assert cov.acc_internal_lmax() == 64 + 15 > 2 * cov.wlm.nside
    keys = CovKeys(["T"], [FREQ])
    with pytest.raises(ValueError, match="needs 79"):
        cov.compute_covariance_matrix([2, 20, 64], keys, _tt(78))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, sigma = cov.compute_covariance_matrix([2, 20, 64], keys, _tt(79))
    assert np.all(np.isfinite(sigma)) and np.all(np.diag(sigma) > 0)
    with pytest.raises(ValueError, match="cannot exceed 2\\*nside"):
        _acc(kernel_dir, 65)


def test_nka_still_takes_spectra_of_length_lmax(kernel_dir):
    config = CovarianceConfig(method=CovarianceMethod.NKA, lmax=48)
    cov = Cov("baseline_mask.fits", config=config, mask_path=DATA, save_dir=kernel_dir)
    _, _, sigma = cov.compute_covariance_matrix(
        [2, 20, 48], CovKeys(["T"], [FREQ]), _tt(48)
    )
    assert np.all(np.isfinite(sigma))
    with pytest.raises(ValueError, match="acc_internal_lmax applies to the ACC"):
        cov.acc_internal_lmax()


def test_pipeline_names_a_spectrum_file_that_stops_short_of_lmax_int(
    kernel_dir, tmp_path
):
    """baseline_cls.dat ends at ell = 64; lmax 60 + pad 15 needs ell 74."""
    from cmbcov import CovarianceMatrixGenerator

    template = open(os.path.join(DATA, "baseline_params.yml")).read()
    text = (
        template.replace("PLACEHOLDER_OUT", kernel_dir)
        .replace("PLACEHOLDER_DATA", DATA)
        .replace("stokes: ['T', 'E']", "stokes: ['T']")
        .replace(
            "covariance_approximation: nka",
            f"covariance_approximation: acc\ndmax: 2\ncentralell: {ELL}",
        )
    )
    params = tmp_path / "params.yml"
    params.write_text(text)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError) as excinfo:
            CovarianceMatrixGenerator(str(params)).run_full_analysis()
    message = str(excinfo.value)
    assert "baseline_cls.dat" in message
    assert "ends at ell = 64" in message
    assert "lmax_int = 75" in message


def test_pipeline_without_kernels_says_to_precompute(tmp_path):
    from cmbcov import CovarianceMatrixGenerator

    template = open(os.path.join(DATA, "baseline_params.yml")).read()
    text = (
        template.replace("PLACEHOLDER_OUT", str(tmp_path / "out"))
        .replace("PLACEHOLDER_DATA", DATA)
        .replace(
            "covariance_approximation: nka",
            f"covariance_approximation: acc\ndmax: 2\ncentralell: {ELL}",
        )
    )
    params = tmp_path / "params.yml"
    params.write_text(textwrap.dedent(text))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="precompute-acc"):
            CovarianceMatrixGenerator(str(params)).run_full_analysis()


def test_low_centralell_warns_once_per_value(kernel_dir):
    """The warning fires once per ``centralell`` value per process, not on
    every ``Cov``, computation or ACC block."""
    import cmbcov.covariance as covariance_module

    covariance_module._WARNED_LOW_CENTRALELL.discard(ELL)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(2):
            cov = Cov(
                "baseline_mask.fits",
                config=CovarianceConfig(
                    method=CovarianceMethod.ACC, lmax=48, dmax=2, centralell=ELL
                ),
                mask_path=DATA,
                save_dir=kernel_dir,
                acc_kernel_dir=kernel_dir,
            )
            cov.compute_covariance_matrix(
                [2, 20, 48], CovKeys(["T"], [FREQ, "150GHz"]), _two_freq(63)
            )
    low = [w for w in caught if "centralell" in str(w.message)]
    assert len(low) == 1
    assert "centralell=16" in str(low[0].message)


def _two_freq(size):
    tt = _tt(size)[FREQ + FREQ]["TT"]
    return {a + b: {"TT": tt} for a in (FREQ, "150GHz") for b in (FREQ, "150GHz")}
