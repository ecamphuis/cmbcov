r"""
Orientation of the T/E-only ACC blocks (docs/theory/acc.md, Sect. 3).

The exact covariance of two different spectra is not symmetric within its
block, ``Cov(a_l, b_l') != Cov(a_l', b_l)``, and the ACC kernel of
``Cov(a, b)`` at ``(l*, l* + Delta)`` is exact only with ``a`` on the lower
multipole. ``ACCStrategy.compute_covariance_term`` therefore computes the
lower triangle of a non-auto block from the Wick contractions of the
transposed key. Before, it reused the upper value, so the element
``(a at l + Delta, b at l)`` was taken from the kernel with ``a`` at ``l``;
which element was exact then depended on the order in which the run lists
its spectra, not on the spectra themselves. In a multi-frequency T/E run
that broke the cancellation of the CMB in the frequency differences, and the
matrix was not positive definite (-1.3e-4 and -1.8e-4 in correlation units
on a 4% survey footprint with 2 and 3 frequencies).

On the level-1 configuration of ``tests/reference/acc_level1.py`` (rippled
cap, ``l* = 8``, ``dmax = 4``, ``lmax = 20``, two frequencies), measured
before / after the change:

(a) the lower triangle of every non-auto block is the transposed upper
    triangle of the transposed block (round-off after; up to 1.2e-3 of the
    block's largest element off before);
(b) with the same spectra at every frequency the two maps are one map, and
    the exact covariance vanishes on the frequency differences; the raw ACC
    matrix there: 2.0e-3 (correlation units) before, 7e-16 after;
(c) listing the frequencies in the other order changes the matrix (beyond
    the permutation) by 2.4e-5 of its largest element before, 2e-16 after;
(d) two frequencies with white noise 1e-4 and 2e-4 of the TT scale: the
    binned (three bins per spectrum) correlation matrix had a -2.0e-4
    eigenvalue before; its lowest is +1.8e-6 after.
"""

import os
import warnings

import numpy as np
import pytest

hp = pytest.importorskip("healpy")

from reference.acc_level1 import (  # noqa: E402
    CENTRALELL,
    DMAX,
    LMAX,
    LW,
    NSIDE_ACC,
    rippled_cap_bandlimited,
)

from cmbcov.approximations import StrategyFactory  # noqa: E402
from cmbcov.approximations.acc import precompute_acc_kernels  # noqa: E402
from cmbcov.covariance import Cov, CovarianceConfig, CovarianceMethod  # noqa: E402
from cmbcov.keys import CovKeys  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

FREQS = ["f1", "f2"]


@pytest.fixture(scope="module")
def level1_cov(tmp_path_factory):
    work = str(tmp_path_factory.mktemp("acc_orientation"))
    hp.write_map(
        os.path.join(work, "mask.fits"),
        rippled_cap_bandlimited(),
        overwrite=True,
        dtype=np.float64,
    )
    precompute_acc_kernels(
        "mask.fits",
        work,
        centralell=CENTRALELL,
        dmax=DMAX,
        mask_path=work,
        nside=NSIDE_ACC,
        grid="gl",
        lw=LW,
    )
    config = CovarianceConfig(
        method=CovarianceMethod.ACC,
        lmax=LMAX,
        lmin=2,
        dmax=DMAX,
        centralell=CENTRALELL,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Cov("mask.fits", config=config, mask_path=work, save_dir=work)


def _spectra(noise=(0.0, 0.0), n=80):
    """Two-frequency T/E spectra: one sky, white noise ``noise[i]`` in the
    auto-spectra of frequency ``i`` (twice that in EE)."""
    ell = np.arange(n)
    tt = np.zeros(n)
    ee = np.zeros(n)
    tt[2:] = 1.0 / (ell[2:] * (ell[2:] + 1)) ** 0.8
    ee[2:] = 0.3 / (ell[2:] + 3.0) ** 1.7 * (1 + 0.5 * np.cos(ell[2:] / 2.0))
    te = 0.6 * np.sqrt(tt * ee) * np.cos(ell / 3.0)
    cl = {}
    for i, f1 in enumerate(FREQS):
        for j, f2 in enumerate(FREQS):
            white = noise[i] if i == j else 0.0
            cl[f1 + f2] = {
                "TT": tt + white,
                "EE": ee + 2 * white,
                "TE": te.copy(),
                "ET": te.copy(),
            }
    return cl


def _raw_matrix(cov, keys, cl):
    """The full raw (pseudo-C_l) matrix of ``keys``, lower blocks by
    transposition as ``Cov.compute_covariance_matrix`` fills them."""
    strategy = StrategyFactory.create_strategy(cov)
    strategy.configure_run(keys, cl)
    size = LMAX
    n = len(keys.spec_keys)
    out = np.zeros((n * size, n * size))
    for key in keys.keys():
        i, j = keys[key]
        block = strategy.compute_covariance_term(key, cl)
        out[i * size : (i + 1) * size, j * size : (j + 1) * size] = block
        if i != j:
            out[j * size : (j + 1) * size, i * size : (i + 1) * size] = block.T
    return out


def _correlation(matrix):
    d = np.sqrt(np.abs(np.diag(matrix)))
    d[d == 0] = 1.0
    return matrix / np.outer(d, d)


def test_lower_triangle_is_the_transposed_block(level1_cov):
    cl = _spectra((0.01, 0.02))
    keys = CovKeys(["T", "E"], FREQS)
    strategy = StrategyFactory.create_strategy(level1_cov)
    strategy.configure_run(keys, cl)
    checked = 0
    for key in keys.keys():
        if key.auto():
            continue
        block = strategy.compute_covariance_term(key, cl)
        transposed = strategy.compute_covariance_term(key.transpose(), cl)
        scale = np.max(np.abs(block))
        np.testing.assert_allclose(
            np.tril(block, -1), np.tril(transposed.T, -1), rtol=0, atol=1e-13 * scale
        )
        # the upper triangle is the block's own orientation
        np.testing.assert_allclose(
            np.triu(transposed, 1), np.tril(block, -1).T, rtol=0, atol=1e-13 * scale
        )
        checked += 1
    assert checked == 45


def test_identical_frequencies_keep_the_exact_null_space(level1_cov):
    """Same spectra at every frequency pair: the pseudo spectra of the same
    letters are one random variable, so every difference of two of them
    (TT, EE, and TE with ET across frequencies) has zero variance."""
    cl = _spectra()
    for freq in cl:
        cl[freq] = {s: a.copy() for s, a in cl["f1f1"].items()}
    keys = CovKeys(["T", "E"], FREQS)
    corr = _correlation(_raw_matrix(level1_cov, keys, cl))
    groups: dict[str, list[int]] = {}
    for i, spec in enumerate(keys.spec_keys):
        letters = spec.stokekey()
        groups.setdefault("TE" if letters in ("TE", "ET") else letters, []).append(i)
    assert sorted(len(g) for g in groups.values()) == [3, 3, 4]
    rows = []
    for members in groups.values():
        for other in members[1:]:
            for ell in range(2, LMAX):
                v = np.zeros(corr.shape[0])
                v[members[0] * LMAX + ell] = 1.0
                v[other * LMAX + ell] = -1.0
                rows.append(v)
    null = np.array(rows)
    assert np.max(np.abs(null @ corr @ null.T)) <= 1e-12


def test_frequency_order_does_not_change_the_matrix(level1_cov):
    cl = _spectra((0.01, 0.02))
    keys = CovKeys(["T", "E"], FREQS)
    keys_swapped = CovKeys(["T", "E"], FREQS[::-1])
    labels = [(s.stokekey(), s.freqkey()) for s in keys.spec_keys]

    def as_listed_in_keys(label):
        # XY at (f2, f1) is YX at (f1, f2)
        letters, freqs = label
        if freqs == "f2f1":
            return letters[::-1], "f1f2"
        return label

    order = [
        labels.index(as_listed_in_keys((s.stokekey(), s.freqkey())))
        for s in keys_swapped.spec_keys
    ]
    index = np.concatenate([np.arange(k * LMAX, (k + 1) * LMAX) for k in order])
    direct = _raw_matrix(level1_cov, keys, cl)
    swapped = _raw_matrix(level1_cov, keys_swapped, cl)
    np.testing.assert_allclose(
        direct[np.ix_(index, index)],
        swapped,
        rtol=0,
        atol=1e-13 * np.max(np.abs(direct)),
    )


def test_near_degenerate_two_frequency_matrix_is_positive_definite(level1_cov):
    """Low noise: the frequency differences have a variance ~1e-6 of the
    CMB's, which the old orientation rule overshot into a negative
    eigenvalue (-2.0e-4). Binning is a congruence, so this checks the raw
    matrix on the smooth directions."""
    cl = _spectra((1e-4, 2e-4))
    keys = CovKeys(["T", "E"], FREQS)
    raw = _raw_matrix(level1_cov, keys, cl)
    n_bins, width, first = 3, 6, 2
    binning = np.zeros((n_bins, LMAX))
    for b in range(n_bins):
        binning[b, first + width * b : first + width * (b + 1)] = 1.0 / width
    project = np.kron(np.eye(len(keys.spec_keys)), binning)
    eigenvalues = np.linalg.eigvalsh(_correlation(project @ raw @ project.T))
    assert eigenvalues[0] > 1e-7, eigenvalues[:3]
