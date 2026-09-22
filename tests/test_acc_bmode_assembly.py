r"""
The ACC covariance assembly of a run with a B observable, with the
per-Wick-term normalisation of docs/theory/bmode_kernels.md (:meth:`ACCStrategy._compute_covariance_term_wick`,
:meth:`Cov.acc_term_scale`).

(a) **Exact at l*.** On the derivation test's rippled cap (band-limited at
    ``LW = 6``), every one of the 21 blocks among TT, EE, BB, TE, TB, EB,
    assembled by the production path at ``l = l*`` for ``Delta = 0..3``,
    equals ``exact_covariance_row_pol``: measured <= 1.1e-12 relative with
    ``C^TB = C^EB = 0`` and with both non-zero (the TT x TT floor, the
    identity ``sum Theta^{TTxTT} = n Xi^00`` with ``Cov.Xi``; every other
    term is exact by construction, <= 3e-15). The parity-mixed blocks with
    ``C^TB = C^EB = 0`` are ~1e-5 of their neighbours, so they are compared
    in units of ``sqrt(Cov(a,a) Cov(b,b))``.
(b) **Off l*.** The B16 configuration (docs/theory/bmode_kernels.md, Sect. 8; the appendix script:
    baseline mask band-limited at ``L_w = 10``, ``l* = 16``, shifts 5, 10, 20,
    30, ``Delta = 0, 1, 3``), production ACC against kernels built at
    ``(l, l')`` themselves. Measured worst ``|ACC/exact - 1|``, recommended
    rule (documented in brackets): C^BB = 0.2 C^EE: TT x TT 0.56% (0.6%),
    EE x EE 1.44% (1.4%), EE x BB 0.90%, BB x BB 1.01%, TB x TB 0.80%
    (<= 1.4%), TB x EB 0.93% (0.9%), EB x EB 1.41% (1.4%); level-1 EE x EE
    4.11% (4.1%). C^BB = 0.01 C^EE: EE x EE 1.46% (1.5%), EE x BB 1.32%,
    BB x BB 1.15%, TB x TB 0.82% (<= 1.5%), TB x EB 1.06% (1.1%), EB x EB
    0.90% (0.9%); level-1 EE x EE 5.38% (5.4%). C^BB = 0 (docs/theory/bmode_kernels.md, Sect. 8):
    TT x EE 2.21% -> 0.89% (2.2% -> 0.9%), EE x EE 5.45% -> 1.46%
    (5.4% -> 1.5%), TE x TE 2.20% -> 0.89% (2.2% -> 0.9%).
(c) **Level 1 unchanged.** A two-frequency ``stokes: [T, E]`` ACC run
    matches a T/E-only reference (stored by
    ``tests/reference/acc_level1.py``), and its raw-block manifest has no
    new field.
(d) **End to end.** ``CovarianceMatrixGenerator`` runs of levels 2, 3 and 4
    (``polspice_postprocess: false``) give finite matrices, symmetric to
    <= 1e-12 of their largest entry, with the expected block layout.
"""

import glob
import os
import shutil
import warnings

import numpy as np
import pytest
import yaml

hp = pytest.importorskip("healpy")

from reference.acc_level1 import (  # noqa: E402
    build_level1_reference,
    rippled_cap_bandlimited,
)

from cmbcov import CovarianceMatrixGenerator  # noqa: E402
from cmbcov.approximations import StrategyFactory  # noqa: E402
from cmbcov.approximations.acc import (  # noqa: E402
    ACC_NORMALISATION_RULE,
    CHANNEL_ALIASES,
    COUPLING_CHANNELS,
    precompute_acc_kernels,
)
from cmbcov.bmode_wick import (  # noqa: E402
    block_wick_terms,
    canonical_pair,
    covkey_wick_terms,
    required_kernel_pairs,
    term_class,
)
from cmbcov.covariance import (  # noqa: E402
    Cov,
    CovarianceConfig,
    CovarianceMethod,
)
from cmbcov.exact import exact_covariance_row_pol  # noqa: E402
from cmbcov.keys import CovKey, CovKeys  # noqa: E402
from cmbcov.sht import ducc0_map2alm  # noqa: E402

pytestmark = pytest.mark.filterwarnings(
    "ignore:internal band-limit margin",
    "ignore:centralell=",
    "ignore::UserWarning",
)

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
REFERENCE = os.path.join(os.path.dirname(__file__), "reference")
SIX = ["TT", "EE", "BB", "TE", "TB", "EB"]
ALL_PAIRS = [(a, b) for a in COUPLING_CHANNELS for b in COUPLING_CHANNELS]


def _key(s1, s2, freq="f"):
    return CovKey((s1[0], s1[1], s2[0], s2[1]), (freq,) * 4)


def _acc_cov(workdir, lmax, dmax, centralell):
    config = CovarianceConfig(
        method=CovarianceMethod.ACC,
        lmax=lmax,
        lmin=2,
        dmax=dmax,
        centralell=centralell,
        polspice_postprocess=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Cov("mask.fits", config=config, mask_path=workdir, save_dir=workdir)


# --------------------------------------------------------------------------- #
# Unit checks of the generator and the term classes
# --------------------------------------------------------------------------- #


def test_covkey_wick_terms_match_block_wick_terms_up_to_transposition():
    """Single frequency: the frequency-aware expansion is the
    docs/theory/bmode_kernels.md, Sect. 4, generator with every term
    transposed into its stored orientation."""
    for s1 in SIX:
        for s2 in SIX:
            for odd in (False, True):
                expected = {}
                for (a, c1, c2, b), coef in block_wick_terms(s1, s2, odd).items():
                    n1, n2 = CHANNEL_ALIASES[c1], CHANNEL_ALIASES[c2]
                    if canonical_pair(n1, n2) != (n1, n2):
                        a, n1, n2, b = b, n2, n1, a
                    key = (a, n1, n2, b)
                    expected[key] = expected.get(key, 0) + coef
                expected = {k: v for k, v in expected.items() if v}
                got = {
                    (t.left[0], t.channel_1, t.channel_2, t.right[0]): t.coefficient
                    for t in covkey_wick_terms(tuple(s1 + s2), ("f",) * 4, odd)
                }
                assert got == expected, (s1, s2, odd)


def test_wick_term_pairs_are_the_precompute_orientation():
    """Every pair a term asks for is in the orientation required_kernel_pairs
    (hence the precompute) uses, so a full-orientation cache always has it."""
    stored = required_kernel_pairs(SIX, parity_odd_nonzero=True)
    for s1 in SIX:
        for s2 in SIX:
            for t in covkey_wick_terms(tuple(s1 + s2), ("f",) * 4, True):
                assert (t.channel_1, t.channel_2) == canonical_pair(
                    t.channel_1, t.channel_2
                )
    assert all(canonical_pair(*p) == p for p in stored)


def test_term_classes():
    assert term_class("TT", "TT") == "eq23"
    for pair in [("TT", "DD"), ("TD", "DT"), ("DD", "DD"), ("DD", "DT")]:
        assert term_class(*pair) == "deficit"
    for pair in [("TT", "LL"), ("DD", "LL"), ("LL", "DT"), ("LL", "LL")]:
        assert term_class(*pair) == "auto"
    for pair in [("TL", "TL"), ("TL", "LT"), ("DL", "LD"), ("LD", "LD")]:
        assert term_class(*pair) == "cross"
    for pair in [("TT", "TL"), ("DD", "DL"), ("TD", "TL")]:
        assert term_class(*pair) == "frozen"


def test_multi_frequency_terms_carry_leg_order_frequencies():
    """Cov(T_1 B_2, T_1 B_2), the docs/theory/bmode_kernels.md, Sect. 4,
    TB x TB row
    ``TT.Theta^{TTxDD}.BB + TT.Theta^{TTxLL}.EE + TE.Theta^{TLxLT}.TE`` with
    frequencies, each term transposed into the stored orientation
    (the old-letter minimum: EExTT, BBxTT, BTxTB): the TL x LT term reads
    C^{T_1 E_2} on both sides, as ET_{21} and TE_{12} (Sect. 2: the
    leading term of R^{T_nu B_mu} is -C^{T_nu E_mu} X^{TL})."""
    terms = covkey_wick_terms(("T", "B", "T", "B"), ("1", "2", "1", "2"))
    assert [tuple(t) for t in terms] == [
        (1, ("BB", ("2", "2")), "DD", "TT", ("TT", ("1", "1"))),
        (1, ("EE", ("2", "2")), "LL", "TT", ("TT", ("1", "1"))),
        (1, ("ET", ("2", "1")), "LT", "TL", ("TE", ("1", "2"))),
    ]


# --------------------------------------------------------------------------- #
# (a) exact at l* on the derivation's rippled cap
# --------------------------------------------------------------------------- #

A_LW, A_LSTAR, A_DMAX, A_NSIDE_ACC, A_LMAX = 6, 8, 4, 16, 20


def _a_spectra(odd):
    n = 80
    ell = np.arange(n)
    tt, ee, bb = np.zeros(n), np.zeros(n), np.zeros(n)
    tt[2:] = 1.0 / (ell[2:] * (ell[2:] + 1)) ** 0.8
    ee[2:] = 0.3 / (ell[2:] + 3.0) ** 1.7 * (1 + 0.5 * np.cos(ell[2:] / 2.0))
    bb[2:] = 0.05 / (ell[2:] + 1.0) ** 0.9 * (1 + 0.4 * np.sin(ell[2:] / 1.3))
    te = 0.6 * np.sqrt(tt * ee) * np.cos(ell / 3.0)
    tb = 0.1 * np.sqrt(tt * bb) * np.sin(ell / 2.5) if odd else 0 * tt
    eb = 0.2 * np.sqrt(ee * bb) * np.cos(ell / 4.0 + 0.3) if odd else 0 * tt
    return {"TT": tt, "EE": ee, "BB": bb, "TE": te, "TB": tb, "EB": eb}


@pytest.fixture(scope="module")
def rippled(tmp_path_factory):
    """The rippled cap as a band-limited map, with GL kernels for every one
    of the 81 ordered pairs (so every block is computed in its natural
    orientation) and, in a second directory, only the 18-pair minimal set."""
    full = str(tmp_path_factory.mktemp("rippled_full"))
    minimal = str(tmp_path_factory.mktemp("rippled_18"))
    for work, pairs in (
        (full, ALL_PAIRS),
        (minimal, sorted(required_kernel_pairs(SIX))),
    ):
        hp.write_map(
            os.path.join(work, "mask.fits"),
            rippled_cap_bandlimited(32, A_LW),
            overwrite=True,
            dtype=np.float64,
        )
        precompute_acc_kernels(
            "mask.fits",
            work,
            centralell=A_LSTAR,
            dmax=A_DMAX,
            mask_path=work,
            nside=A_NSIDE_ACC,
            grid="gl",
            lw=A_LW,
            pairs=pairs,
        )
    cov = _acc_cov(full, A_LMAX, A_DMAX, A_LSTAR)
    alm = ducc0_map2alm(cov.wlm.mask, lmax=A_LW, pol=False, iter=10)
    return {"full": full, "minimal": minimal, "cov": cov, "alm": alm}


def _exact_rows(alm, cls):
    return {
        d: exact_covariance_row_pol(
            alm, cls, A_LSTAR + d, A_LMAX, spectra=SIX, lw=A_LW, lmax_int=40
        )
        for d in range(A_DMAX)
    }


@pytest.mark.parametrize("odd", [False, True], ids=["TB=EB=0", "TB,EB!=0"])
def test_every_block_is_exact_at_lstar(rippled, odd):
    cls = _a_spectra(odd)
    cl = {"ff": {**cls, "ET": cls["TE"], "BT": cls["TB"], "BE": cls["EB"]}}
    keys = CovKeys([], ["f"], observables=SIX, parity_mixed_blocks=True)
    strategy = StrategyFactory.create_strategy(rippled["cov"])
    strategy.configure_run(keys, cl)
    assert strategy._run_context["parity_odd"] is odd
    rows = _exact_rows(rippled["alm"], cls)
    worst_plain = worst_natural = 0.0
    assert len(keys.keys()) == 21
    for key in keys.keys():
        s1, s2 = key.stokekey().split("x")
        block = strategy.compute_covariance_term(key, cl)
        for d in range(A_DMAX):
            got = block[A_LSTAR, A_LSTAR + d]
            ref = rows[d][(s1, s2)][A_LSTAR]
            natural = np.sqrt(
                abs(rows[0][(s1, s1)][A_LSTAR]) * abs(rows[0][(s2, s2)][A_LSTAR])
            )
            mixed = (s1 in ("TB", "EB")) != (s2 in ("TB", "EB"))
            if mixed and not odd:
                err = abs(got - ref) / natural
                worst_natural = max(worst_natural, err)
            else:
                err = abs(got / ref - 1)
                worst_plain = max(worst_plain, err)
            assert err <= 1e-10, (s1, s2, d, err)
    assert worst_plain <= 1e-10 and worst_natural <= 1e-10


def test_transposed_orientation_from_the_minimal_cache(rippled):
    """With only the 18-pair cache (best orientation,
    docs/theory/bmode_kernels.md, Sect. 6), Cov(TE, BB)
    is computed as Cov(BB, TE)^T: its exact element is then (l*+Delta, l*),
    and the raw-block identity records the orientation."""
    cls = _a_spectra(False)
    cl = {"ff": {**cls, "ET": cls["TE"], "BT": cls["TB"], "BE": cls["EB"]}}
    cov = _acc_cov(rippled["minimal"], A_LMAX, A_DMAX, A_LSTAR)
    strategy = StrategyFactory.create_strategy(cov)
    strategy.configure_run(CovKeys([], ["f"], observables=SIX), cl)
    orientation = {
        blk: strategy.raw_block_inputs(_key(*blk), cl)[1]["acc_block_orientation"]
        for blk in [("TE", "BB"), ("BB", "TE"), ("TB", "EB"), ("EB", "TB")]
    }
    assert orientation == {
        ("TE", "BB"): "transposed",
        ("BB", "TE"): "natural",
        ("TB", "EB"): "transposed",
        ("EB", "TB"): "natural",
    }
    block = strategy.compute_covariance_term(_key("TE", "BB"), cl)
    rows = {
        0: exact_covariance_row_pol(
            rippled["alm"], cls, A_LSTAR, A_LMAX, spectra=SIX, lw=A_LW, lmax_int=40
        )
    }
    for d in range(A_DMAX):
        ref = rows[0][("TE", "BB")][A_LSTAR + d]  # Cov(TE_{l*+d}, BB_{l*})
        assert abs(block[A_LSTAR + d, A_LSTAR] / ref - 1) <= 1e-10, d


def test_parity_mixed_block_is_zero_unless_requested(rippled):
    cls = _a_spectra(False)
    cl = {"ff": {**cls, "ET": cls["TE"], "BT": cls["TB"], "BE": cls["EB"]}}
    strategy = StrategyFactory.create_strategy(rippled["cov"])
    strategy.configure_run(CovKeys([], ["f"], observables=SIX), cl)
    assert not np.any(strategy.compute_covariance_term(_key("TT", "TB"), cl))
    strategy.configure_run(
        CovKeys([], ["f"], observables=SIX, parity_mixed_blocks=True), cl
    )
    assert np.any(strategy.compute_covariance_term(_key("TT", "TB"), cl))


def test_healpix_grid_kernels_are_refused_for_a_b_run(tmp_path):
    """The per-term rule reads raw kernel sums, which carry a sqrt(2) per
    spin-2 integral on the HEALPix grid: refused, not silently rescaled."""
    work = str(tmp_path)
    shutil.copy(
        os.path.join(DATA, "baseline_mask.fits"), os.path.join(work, "mask.fits")
    )
    precompute_acc_kernels(
        "mask.fits",
        work,
        centralell=10,
        dmax=1,
        mask_path=work,
        nside=16,
        grid="healpix",
        pairs=sorted(required_kernel_pairs(["TT", "EE", "TE", "BB"])),
    )
    cov = _acc_cov(work, 20, 1, 10)
    cls = _a_spectra(False)
    cl = {"ff": {**cls, "ET": cls["TE"]}}
    strategy = StrategyFactory.create_strategy(cov)
    with pytest.raises(ValueError, match="GL grid"):
        strategy.compute_covariance_term(_key("BB", "BB"), cl)


# --------------------------------------------------------------------------- #
# (b) off l*: the B16 row (docs/theory/bmode_kernels.md, Sect. 8)
# --------------------------------------------------------------------------- #

B_LW, B_LSTAR, B_SHIFTS, B_DELTAS, B_LMAX, B_NSIDE_ACC = (
    10,
    16,
    [5, 10, 20, 30],
    [0, 1, 3],
    50,
    32,
)

#: Documented worst |ACC/exact - 1| (docs/theory/bmode_kernels.md, Sect. 8,
#: B16 summary row), recommended rule; ``("<=", x)`` for blocks
#: the note bounds only through "worst other block". C^BB/C^EE -> block -> value.
DOCUMENTED = {
    0.2: {
        ("TT", "TT"): 0.006,
        ("EE", "EE"): 0.014,
        ("EE", "BB"): ("<=", 0.014),
        ("BB", "BB"): ("<=", 0.014),
        ("TB", "TB"): ("<=", 0.014),
        ("TB", "EB"): 0.009,
        ("EB", "EB"): 0.014,
    },
    0.01: {
        ("TT", "TT"): 0.006,
        ("EE", "EE"): 0.015,
        ("EE", "BB"): ("<=", 0.015),
        ("BB", "BB"): ("<=", 0.015),
        ("TB", "TB"): ("<=", 0.015),
        ("TB", "EB"): 0.011,
        ("EB", "EB"): 0.009,
    },
    0.0: {
        ("TT", "EE"): 0.009,
        ("EE", "EE"): 0.015,
        ("TE", "TE"): 0.009,
    },
}

#: Documented level-1 ("today") worst error, where docs/theory/bmode_kernels.md,
#: Sect. 8, says it is worse than the recommended rule.
DOCUMENTED_LEVEL1 = {
    0.2: {("EE", "EE"): 0.041},
    0.01: {("EE", "EE"): 0.054},
    0.0: {("TT", "EE"): 0.022, ("EE", "EE"): 0.054, ("TE", "TE"): 0.022},
}


@pytest.fixture(scope="module")
def b16(tmp_path_factory):
    """The appendix harness's configuration through the production path: the
    baseline mask band-limited at L_w = 10 (so Cov.Xi is that of the mask the
    kernels see), GL kernels at l* = 16, and exact kernels at (l, l + Delta)."""
    work = str(tmp_path_factory.mktemp("b16"))
    mask = hp.read_map(os.path.join(DATA, "baseline_mask.fits"))
    nside = hp.get_nside(mask)
    mask = hp.alm2map(hp.map2alm(mask, lmax=B_LW, iter=10), nside, lmax=B_LW)
    hp.write_map(os.path.join(work, "mask.fits"), mask, dtype=np.float64)
    dmax = max(B_DELTAS) + 1
    precompute_acc_kernels(
        "mask.fits",
        work,
        centralell=B_LSTAR,
        dmax=dmax,
        mask_path=work,
        nside=B_NSIDE_ACC,
        grid="gl",
        lw=B_LW,
        pairs=ALL_PAIRS,
    )
    exact = {}
    for s in B_SHIFTS:
        ell = B_LSTAR + s
        result = precompute_acc_kernels(
            "mask.fits",
            None,
            centralell=ell,
            ellprange=[ell + d for d in B_DELTAS],
            mask_path=work,
            nside=B_NSIDE_ACC,
            grid="gl",
            lw=B_LW,
            pairs=ALL_PAIRS,
            dryrun=True,
        )
        for d in B_DELTAS:
            exact[ell, d] = result[ell + d]
    return {"cov": _acc_cov(work, B_LMAX, dmax, B_LSTAR), "exact": exact}


def _b16_spectra(bb_frac):
    n = 200
    ell = np.arange(n)
    tt = np.zeros(n)
    tt[2:] = 1e-3 / (ell[2:] * (ell[2:] + 1)) ** 0.9
    cls = {
        "TT": tt,
        "EE": 0.1 * tt,
        "BB": bb_frac * 0.1 * tt,
        "TE": 0.5 * np.sqrt(0.1) * tt,
    }
    zero = np.zeros(n)
    cls.update({"ET": cls["TE"], "TB": zero, "BT": zero, "EB": zero, "BE": zero})
    return {"ff": cls}


def _worst(block, key, cl, exact):
    cls = cl["ff"]
    size = next(iter(exact.values()))["TTxTT"].shape[0]
    worst = 0.0
    for s in B_SHIFTS:
        l1 = B_LSTAR + s
        for d in B_DELTAS:
            l2 = l1 + d
            n = (2 * l1 + 1) * (2 * l2 + 1)
            ref = (
                sum(
                    t.coefficient
                    * cls[t.left[0]][:size]
                    @ exact[l1, d][f"{t.channel_1}x{t.channel_2}"]
                    @ cls[t.right[0]][:size]
                    for t in covkey_wick_terms(key.stoke, key.freq)
                )
                / n
            )
            worst = max(worst, abs(block[l1, l2] / ref - 1))
    return worst


@pytest.mark.parametrize("bb_frac", [0.2, 0.01, 0.0])
def test_off_lstar_accuracy_reproduces_section_10_5(b16, bb_frac):
    cl = _b16_spectra(bb_frac)
    wick = StrategyFactory.create_strategy(b16["cov"])
    wick.configure_run(CovKeys([], ["f"], observables=SIX), cl)
    level1 = StrategyFactory.create_strategy(b16["cov"])
    level1.configure_run(CovKeys(["T", "E"], ["f"]), cl)
    measured = {}
    for blk, documented in DOCUMENTED[bb_frac].items():
        key = _key(*blk)
        worst = _worst(wick.compute_covariance_term(key, cl), key, cl, b16["exact"])
        measured[blk] = worst
        if isinstance(documented, tuple):
            assert worst <= 1.5 * documented[1], (blk, worst, documented)
        else:
            assert documented / 1.5 <= worst <= 1.5 * documented, (
                blk,
                worst,
                documented,
            )
    for blk, documented in DOCUMENTED_LEVEL1[bb_frac].items():
        key = _key(*blk)
        worst = _worst(level1.compute_covariance_term(key, cl), key, cl, b16["exact"])
        assert documented / 1.5 <= worst <= 1.5 * documented, (blk, worst, documented)
        assert worst > 2 * measured[blk], (blk, worst, measured[blk])


def test_eb_blocks_fail_at_zero_cbb_as_documented(b16):
    """docs/theory/bmode_kernels.md, Sect. 8: with C^BB = 0 exactly,
    Cov(EB, EB) is second order and no
    per-term rule reaches it (measured 212% here, TB x EB 94%)."""
    cl = _b16_spectra(0.0)
    wick = StrategyFactory.create_strategy(b16["cov"])
    wick.configure_run(CovKeys([], ["f"], observables=SIX), cl)
    key = _key("EB", "EB")
    assert _worst(wick.compute_covariance_term(key, cl), key, cl, b16["exact"]) > 0.1


# --------------------------------------------------------------------------- #
# (c) level 1 unchanged, and the raw-block digest
# --------------------------------------------------------------------------- #


def test_level1_run_matches_the_recorded_reference(tmp_path):
    """
    A two-frequency T/E-only ACC run must reproduce
    ``tests/reference/acc_level1_reference.npz`` up to the rounding
    difference of the BLAS/SHT libraries it is built on, not bit-for-bit:
    the reference was recorded on a different machine. On Linux x86
    (different BLAS from the macOS/Accelerate one this was recorded with)
    the worst relative difference measured was 3.6e-13; rtol=1e-10 leaves a
    ~280x margin while still catching a structural change to the assembly.
    """
    reference = np.load(os.path.join(REFERENCE, "acc_level1_reference.npz"))
    current = build_level1_reference(str(tmp_path))
    assert sorted(reference.files) == sorted(current)
    for name in reference.files:
        np.testing.assert_allclose(
            current[name],
            reference[name],
            rtol=1e-10,
            atol=1e-12 * np.abs(reference[name]).max(),
            err_msg=name,
        )


def _level1_cov(workdir):
    hp.write_map(
        os.path.join(workdir, "mask.fits"),
        rippled_cap_bandlimited(),
        overwrite=True,
        dtype=np.float64,
    )
    precompute_acc_kernels(
        "mask.fits",
        workdir,
        centralell=8,
        dmax=2,
        mask_path=workdir,
        nside=16,
        grid="gl",
        lw=6,
        pairs=sorted(required_kernel_pairs(["TT", "EE", "TE", "BB"])),
    )
    return _acc_cov(workdir, 20, 2, 8)


def test_raw_block_manifest_level1_unchanged_b_run_versioned(tmp_path):
    """A T/E-only run's manifest has exactly the T/E-only fields (no B-mode
    fields); the same TT x TT block in a B run carries the normalisation
    rule, so a block saved by the level-1 run is never reused for the B
    run."""
    cov = _level1_cov(str(tmp_path))
    cls = _a_spectra(False)
    cl = {"ff": {**cls, "ET": cls["TE"]}}
    key = _key("TT", "TT")
    level1 = StrategyFactory.create_strategy(cov)
    level1.configure_run(CovKeys(["T", "E"], ["f"]), cl)
    brun = StrategyFactory.create_strategy(cov)
    brun.configure_run(CovKeys([], ["f"], observables=["TT", "EE", "TE", "BB"]), cl)
    manifest_1 = cov._raw_block_manifest(level1, key, cl)
    manifest_b = cov._raw_block_manifest(brun, key, cl)
    assert set(manifest_1) == {
        "raw_block_cache_version",
        "method",
        "stokes",
        "frequencies",
        "lmax",
        "mask_digest",
        "mask_alm_lmax",
        "mask_alm_maxiter",
        "mask_alm_epsilon",
        "kernel_cache_version",
        "spectra_digest",
        "lmax_int",
        "dmax",
        "centralell",
        "acc_kernel_dir",
        "acc_kernels_digest",
    }
    assert manifest_b["acc_normalisation"] == ACC_NORMALISATION_RULE
    assert manifest_1 != manifest_b
    # The level-1 manifest does not depend on whether run context was set.
    bare = StrategyFactory.create_strategy(cov)
    assert cov._raw_block_manifest(bare, key, cl) == manifest_1


def test_saved_level1_block_is_recomputed_for_a_b_run(tmp_path):
    cov = _level1_cov(str(tmp_path))
    cov.config.save_raw_blocks = True
    cls = _a_spectra(False)
    cl = {"ff": {**cls, "ET": cls["TE"]}}
    cov.compute_covariance_matrix([2, 8, 14, 20], CovKeys(["T"], ["f"]), cl)
    level1_block = np.load(os.path.join(str(tmp_path), "cov_TTxTT_ffxff.npy"))
    cov.compute_covariance_matrix(
        [2, 8, 14, 20], CovKeys([], ["f"], observables=["TT", "BB"]), cl
    )
    b_block = np.load(os.path.join(str(tmp_path), "cov_TTxTT_ffxff.npy"))
    # TT x TT under the per-term rule differs from Eq. 23 only by the
    # summation order of the band, so compare the manifests, not values.
    with open(os.path.join(str(tmp_path), "cov_TTxTT_ffxff.npy.manifest.json")) as f:
        assert '"acc_normalisation"' in f.read()
    np.testing.assert_allclose(b_block, level1_block, rtol=1e-12, atol=0)


def test_single_frequency_run_with_white_noise_levels(tmp_path):
    """Guards against ``_load_noise`` aliasing ET onto TE unconditionally
    while ``prepare_workflow`` refuses a noise key the data model lacks:
    together these would make every single-frequency run with a white-noise
    ``nl`` dict (where the run has TE but no ET) fail with ``KeyError``.
    Found while building the single-frequency B-mode runs of (d)."""
    work = str(tmp_path)
    shutil.copy(os.path.join(DATA, "baseline_mask.fits"), work)
    params = {
        "cov_path": os.path.join(work, "out"),
        "cov_name": "cov.dat",
        "frequencies": ["090GHz"],
        "stokes": ["T", "E"],
        "lmax": 20,
        "lmin": 2,
        "bins": [[2, 20, 3]],
        "mask_name": "baseline_mask.fits",
        "mask_path": work,
        "covariance_approximation": "nka",
        "cmb_spectrum": os.path.join(DATA, "baseline_cls.dat"),
        "beams": {"090GHz": 5.0},
        "pixwin": 32,
        "nl": {"090GHz090GHz": 10.0},
    }
    path = os.path.join(work, "params.yml")
    with open(path, "w") as handle:
        yaml.dump(params, handle)
    CovarianceMatrixGenerator(path).run_full_analysis()
    version = sorted(glob.glob(os.path.join(work, "out", "v*")))[-1]
    assert np.isfinite(np.loadtxt(os.path.join(version, "cov.dat"))).all()


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #


def test_b_run_with_polspice_postprocessing_completes(tmp_path):
    """
    PolSpice post-processing of a B-mode run
    (``tests/test_polspice_bmode.py``) runs this configuration to
    completion and returns a finite matrix.
    """
    cov = _level1_cov(str(tmp_path))
    cov.config.polspice_postprocess = True
    cls = _a_spectra(False)
    cl = {"ff": {**cls, "ET": cls["TE"]}}
    keys = CovKeys([], ["f"], observables=["TT", "EE", "TE", "BB"])
    _, _, matrix = cov.compute_covariance_matrix([2, 8, 14, 20], keys, cl)
    assert np.isfinite(matrix).all()


def test_b_run_with_nka_is_refused(tmp_path):
    shutil.copy(os.path.join(DATA, "baseline_mask.fits"), str(tmp_path))
    config = CovarianceConfig(method=CovarianceMethod.NKA, lmax=10, lmin=2)
    cov = Cov(
        "baseline_mask.fits",
        config=config,
        mask_path=str(tmp_path),
        save_dir=str(tmp_path),
    )
    keys = CovKeys([], ["090GHz"], observables=["TT", "BB"])
    with pytest.raises(ValueError, match="needs the ACC method"):
        cov.compute_covariance_matrix([2, 10], keys, cl={})


# --------------------------------------------------------------------------- #
# (d) end to end through CovarianceMatrixGenerator
# --------------------------------------------------------------------------- #

E2E_NBINS = 6


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


def _generator_run(workdir, observables, parity_mixed, nonzero_odd, freqs):
    shutil.copy(os.path.join(DATA, "baseline_mask.fits"), workdir)
    _write_cls(os.path.join(workdir, "cls.dat"), nonzero_odd)
    params = {
        "cov_path": os.path.join(workdir, "out"),
        "cov_name": "cov.dat",
        "frequencies": freqs,
        "observables": observables,
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
        "nl": {f + f: 10.0 for f in freqs},
        "parity_mixed_blocks": parity_mixed,
        "polspice_postprocess": False,
        "acc_precompute": {"nside": 16, "grid": "gl", "lw": 10},
    }
    path = os.path.join(workdir, "params.yml")
    with open(path, "w") as handle:
        yaml.dump(params, handle)
    plan = CovarianceMatrixGenerator(path).precompute_acc_kernels()
    generator = CovarianceMatrixGenerator(path)
    generator.run_full_analysis()
    version = sorted(glob.glob(os.path.join(workdir, "out", "v*")))[-1]
    matrix = np.loadtxt(os.path.join(version, "cov.dat"))
    return plan, generator, matrix


@pytest.mark.parametrize(
    "level, observables, parity_mixed, nonzero_odd, n_pairs",
    [
        (2, ["TT", "EE", "TE", "BB"], False, False, 17),
        (3, SIX, False, False, 18),
        (3, SIX, False, True, 40),
        (4, SIX, True, False, 31),
    ],
    ids=["level2", "level3", "level3-nonzero-TB-EB", "level4"],
)
@pytest.mark.parametrize("freqs", [["090GHz"], ["090GHz", "150GHz"]], ids=["1f", "2f"])
def test_generator_end_to_end(
    tmp_path, level, observables, parity_mixed, nonzero_odd, n_pairs, freqs
):
    plan, generator, matrix = _generator_run(
        str(tmp_path), observables, parity_mixed, nonzero_odd, freqs
    )
    assert len(plan["pairs"]) == n_pairs
    specs = generator.cov_keys.spec_keys
    assert matrix.shape == (len(specs) * E2E_NBINS,) * 2
    assert np.isfinite(matrix).all()
    scale = np.abs(matrix).max()
    assert np.abs(matrix - matrix.T).max() <= 1e-12 * scale
    assert (np.diag(matrix) > 0).all()

    def parity_odd(spec):
        return spec.stokekey().count("B") % 2 == 1

    for i, a in enumerate(specs):
        for j, b in enumerate(specs):
            block = matrix[
                i * E2E_NBINS : (i + 1) * E2E_NBINS, j * E2E_NBINS : (j + 1) * E2E_NBINS
            ]
            if parity_odd(a) != parity_odd(b):
                assert np.any(block) == parity_mixed, (a, b)
            elif i == j:
                assert np.all(np.diag(block) > 0), a
    # Reported, not asserted: ACC is not guaranteed positive semi-definite.
    eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    print(
        f"level {level} {len(freqs)}f: smallest eigenvalue {eigenvalues.min():.3e}, "
        f"largest {eigenvalues.max():.3e}, ratio {eigenvalues.min() / eigenvalues.max():.2e}"
    )
