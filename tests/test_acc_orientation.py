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


# --------------------------------------------------------------------------- #
# Runs with a B observable (per-Wick-term assembly,
# docs/theory/bmode_kernels.md, Sect. 6)
# --------------------------------------------------------------------------- #
#
# ACCStrategy._compute_covariance_term_wick had the same orientation rule:
# the band of Cov(a, b) served both triangles. With a cache holding the
# kernel pairs of both orientations (e.g. all 81 ordered pairs) that was the
# listed orientation, so the matrix depended on the order of the spectra;
# with the one-orientation cache cmbcov 0.3.0 precomputed (18 pairs), the
# blocks whose two orientations were both on disk (EE x BB, TE x TE and
# EB x EB across frequency pairs, ...) still did. Measured on this
# configuration (two frequencies, observables TT EE BB TE TB EB,
# C^TB = C^EB = 0), before / after the fix:
#
#   identical maps, raw matrix on the frequency differences (correlation
#   units): 81-pair cache 1.5e-1 / 1.3e-15, 18-pair cache 7.4e-2 / 1.3e-15;
#   frequency order reversed, correlation change beyond the permutation:
#   6.8e-2 / 6.7e-16 (81 pairs), 2.1e-2 / 6.7e-16 (18 pairs);
#   white noise 1e-4, 2e-4, binned correlation matrix: lowest eigenvalue
#   -3.4e-2 (9 negative) / +1.8e-6 (81 pairs), -2.3e-2 (3 negative) /
#   +1.8e-6 (18 pairs).
#
# required_kernel_pairs now includes both orientations of every block (25
# pairs here), so the "minimal" cache below computes every non-symmetric
# block in both orientations, like the 81-pair one: the same null space,
# order independence and PD, and both triangles exact at l*.

B_OBS = ["TT", "EE", "BB", "TE", "TB", "EB"]
B_FREQS = ["f1", "f2"]


@pytest.fixture(scope="module")
def bmode_covs(tmp_path_factory):
    """The rippled cap with GL kernels for all 81 ordered channel pairs, and
    with the 25-pair set required_kernel_pairs gives for B_OBS (both
    orientations of every block)."""
    from cmbcov.approximations.acc import COUPLING_CHANNELS
    from cmbcov.bmode_wick import required_kernel_pairs

    all_pairs = [(a, b) for a in COUPLING_CHANNELS for b in COUPLING_CHANNELS]
    covs = {}
    for name, pairs in (
        ("full", all_pairs),
        ("minimal", sorted(required_kernel_pairs(B_OBS))),
    ):
        work = str(tmp_path_factory.mktemp(f"acc_orientation_b_{name}"))
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
            pairs=pairs,
        )
        config = CovarianceConfig(
            method=CovarianceMethod.ACC,
            lmax=LMAX,
            lmin=2,
            dmax=DMAX,
            centralell=CENTRALELL,
            polspice_postprocess=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            covs[name] = Cov("mask.fits", config=config, mask_path=work, save_dir=work)
    return covs


def _b_single_spectra(odd=False, n=80):
    ell = np.arange(n)
    tt, ee, bb = np.zeros(n), np.zeros(n), np.zeros(n)
    tt[2:] = 1.0 / (ell[2:] * (ell[2:] + 1)) ** 0.8
    ee[2:] = 0.3 / (ell[2:] + 3.0) ** 1.7 * (1 + 0.5 * np.cos(ell[2:] / 2.0))
    bb[2:] = 0.05 / (ell[2:] + 1.0) ** 0.9 * (1 + 0.4 * np.sin(ell[2:] / 1.3))
    te = 0.6 * np.sqrt(tt * ee) * np.cos(ell / 3.0)
    tb = 0.1 * np.sqrt(tt * bb) * np.sin(ell / 2.5) if odd else 0 * tt
    eb = 0.2 * np.sqrt(ee * bb) * np.cos(ell / 4.0 + 0.3) if odd else 0 * tt
    return {"TT": tt, "EE": ee, "BB": bb, "TE": te, "TB": tb, "EB": eb}


def _b_spectra(noise=None, distinct=False, odd=False):
    """Two-frequency spectra of every letter pair: one sky, white noise
    ``noise[i]`` in the auto-spectra of frequency ``i`` (twice that in EE
    and BB); ``distinct`` scales each frequency pair differently and makes
    XY and YX differ across frequencies."""
    base = _b_single_spectra(odd)
    cl = {}
    for i, f1 in enumerate(B_FREQS):
        for j, f2 in enumerate(B_FREQS):
            white = 0.0 if noise is None or i != j else noise[i]
            scale = 1.0 + 0.1 * i + 0.2 * j if distinct else 1.0
            ab = 1.0 + 0.05 * i if distinct else 1.0
            ba = 1.0 + 0.05 * j if distinct else 1.0
            cl[f1 + f2] = {
                "TT": scale * base["TT"] + white,
                "EE": scale * base["EE"] + 2 * white,
                "BB": scale * base["BB"] + 2 * white,
                "TE": scale * ab * base["TE"],
                "ET": scale * ba * base["TE"],
                "TB": scale * ab * base["TB"],
                "BT": scale * ba * base["TB"],
                "EB": scale * ab * base["EB"],
                "BE": scale * ba * base["EB"],
            }
    return cl


@pytest.mark.parametrize("cache", ["full", "minimal"])
def test_b_run_identical_frequencies_keep_the_exact_null_space(bmode_covs, cache):
    """Same spectra at every frequency pair: every spectrum with the same two
    letters (in either order) is one random variable, so their differences
    have zero variance, B blocks included."""
    keys = CovKeys([], B_FREQS, observables=B_OBS)
    corr = _correlation(_raw_matrix(bmode_covs[cache], keys, _b_spectra()))
    groups: dict[str, list[int]] = {}
    for i, spec in enumerate(keys.spec_keys):
        groups.setdefault("".join(sorted(spec.stokekey())), []).append(i)
    assert sorted(len(g) for g in groups.values()) == [3, 3, 3, 4, 4, 4]
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


def _position_in(keys):
    """Index in ``keys.spec_keys`` of a spectrum of another listing
    (``XY`` at ``(f2, f1)`` is ``YX`` at ``(f1, f2)``)."""
    labels = [(s.stokekey(), s.freqkey()) for s in keys.spec_keys]

    def index(spec):
        letters, freqs = spec.stokekey(), spec.freqkey()
        if (letters, freqs) not in labels:
            half = len(freqs) // 2
            letters, freqs = letters[::-1], freqs[half:] + freqs[:half]
        return labels.index((letters, freqs))

    return index


@pytest.mark.parametrize("cache", ["full", "minimal"])
@pytest.mark.parametrize("odd", [False, True], ids=["TB=EB=0", "TB,EB!=0"])
def test_b_run_spectrum_order_does_not_change_the_matrix(bmode_covs, cache, odd):
    """Listing the frequencies, or the observables, in another order
    permutes the matrix and changes nothing else."""
    if odd and cache == "minimal":
        pytest.skip("the 25-pair cache does not cover non-zero C^TB, C^EB")
    cov = bmode_covs[cache]
    cl = _b_spectra((0.01, 0.02), distinct=True, odd=odd)
    keys = CovKeys([], B_FREQS, observables=B_OBS)
    direct = _raw_matrix(cov, keys, cl)
    index = _position_in(keys)
    for other in (
        CovKeys([], B_FREQS[::-1], observables=B_OBS),
        CovKeys([], B_FREQS, observables=B_OBS[::-1]),
    ):
        order = [index(s) for s in other.spec_keys]
        rows = np.concatenate([np.arange(k * LMAX, (k + 1) * LMAX) for k in order])
        np.testing.assert_allclose(
            direct[np.ix_(rows, rows)],
            _raw_matrix(cov, other, cl),
            rtol=0,
            atol=1e-13 * np.max(np.abs(direct)),
        )


def test_b_run_lower_triangle_is_the_transposed_block(bmode_covs):
    """With both orientations on disk, the lower triangle of every non-auto
    block is the transposed upper triangle of the transposed block, and the
    raw-block identity records the orientation."""
    cl = _b_spectra((0.01, 0.02), distinct=True)
    keys = CovKeys([], B_FREQS, observables=B_OBS, parity_mixed_blocks=True)
    strategy = StrategyFactory.create_strategy(bmode_covs["full"])
    strategy.configure_run(keys, cl)
    orientations: dict[str, int] = {}
    for key in keys.keys():
        if key.auto():
            continue
        block = strategy.compute_covariance_term(key, cl)
        transposed = strategy.compute_covariance_term(key.transpose(), cl)
        scale = np.max(np.abs(block)) or 1.0
        np.testing.assert_allclose(
            np.tril(block, -1), np.tril(transposed.T, -1), rtol=0, atol=1e-13 * scale
        )
        orientation = strategy.raw_block_inputs(key, cl)[1]["acc_block_orientation"]
        orientations[orientation] = orientations.get(orientation, 0) + 1
    # "natural": blocks whose transposed key has the same Wick terms (two
    # TT, two EE or two BB spectra, TB x TB-type ...), computed once
    assert orientations.get("transposed", 0) == 0
    assert orientations["both"] > 0 and orientations["natural"] > 0


def test_b_run_both_triangles_exact_at_lstar(bmode_covs):
    """Single frequency, every ordered pair of different parity-matched
    spectra: the lower triangle is exact at l* too (``Cov(s1 at l* + Delta,
    s2 at l*)``), with all 81 pairs on disk and with the 25-pair set of
    required_kernel_pairs. With the one-orientation 18-pair set of cmbcov
    0.3.0, TT x EE, TT x TE, EE x TE, TT x BB, BB x TE and EB x TB were off
    there by their exact asymmetry: 1.2 to 3.8% (T/E), 6.7% (TB x EB), 29%
    (TE x BB) and 49% (TT x BB) on this cap."""
    from cmbcov.exact import exact_covariance_row_pol
    from cmbcov.keys import CovKey
    from cmbcov.sht import ducc0_map2alm

    cls = _b_single_spectra()
    cl = {"ff": {**cls, "ET": cls["TE"], "BT": cls["TB"], "BE": cls["EB"]}}
    keys = CovKeys([], ["f"], observables=B_OBS)
    alm = ducc0_map2alm(bmode_covs["full"].wlm.mask, lmax=LW, pol=False, iter=10)
    row = exact_covariance_row_pol(
        alm, cls, CENTRALELL, LMAX, spectra=B_OBS, lw=LW, lmax_int=40
    )
    worst = {}
    for cache in ("full", "minimal"):
        strategy = StrategyFactory.create_strategy(bmode_covs[cache])
        strategy.configure_run(keys, cl)
        for s1 in B_OBS:
            for s2 in B_OBS:
                if s1 == s2 or (s1 in ("TB", "EB")) != (s2 in ("TB", "EB")):
                    continue
                key = CovKey((s1[0], s1[1], s2[0], s2[1]), ("f",) * 4)
                block = strategy.compute_covariance_term(key, cl)
                ref = row[(s1, s2)]  # ref[l] = Cov(s1 at l, s2 at l*)
                worst[cache, s1, s2] = max(
                    abs(block[CENTRALELL + d, CENTRALELL] / ref[CENTRALELL + d] - 1)
                    for d in range(1, DMAX)
                )
    assert len(worst) == 2 * 14
    assert max(worst.values()) <= 1e-12, max(worst.items(), key=lambda kv: kv[1])


def test_b_run_near_degenerate_two_frequency_matrix_is_positive_definite(bmode_covs):
    """Low noise, all 81 pairs: the old orientation rule gave the binned
    correlation matrix 9 negative eigenvalues, the lowest -3.4e-2."""
    cl = _b_spectra((1e-4, 2e-4))
    keys = CovKeys([], B_FREQS, observables=B_OBS)
    raw = _raw_matrix(bmode_covs["full"], keys, cl)
    n_bins, width, first = 3, 6, 2
    binning = np.zeros((n_bins, LMAX))
    for b in range(n_bins):
        binning[b, first + width * b : first + width * (b + 1)] = 1.0 / width
    project = np.kron(np.eye(len(keys.spec_keys)), binning)
    eigenvalues = np.linalg.eigvalsh(_correlation(project @ raw @ project.T))
    assert eigenvalues[0] > 1e-7, eigenvalues[:3]
