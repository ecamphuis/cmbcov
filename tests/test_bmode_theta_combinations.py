r"""
Which combinations of the ACC ``Theta`` products the B-mode covariance
blocks need (derivation: docs/theory/bmode_kernels.md).

The ACC kernels are exact at the multipoles they are built at, so the algebra
is tested there: the per-``(m, m')`` cross-spectra
``X^{ab}_{mm'}(L) = sum_M u^a_{lm,LM} conj(u^b_{l'm',LM})`` are built from
:func:`spin_weighted_integrals_gl` at ``l`` and ``l'`` themselves (no
translation), contracted into kernels by the production
:func:`contract_coupling_block` with the flat ``(central, prime)`` field
indexing extended by four channels (TB, BT, EB, BE), and combined with the
true spectra by the correlator table below.  The oracle is
:func:`exact_covariance_row_pol` (``K C K^dagger`` applied to unit vectors
of T, E *and* B, then Wick), which shares no bookkeeping with the table.

Configuration: an off-centre, azimuthally rippled apodised cap (no
reflection symmetry, so ``X`` is genuinely complex and ``|u^B| ~ |u^E|``),
band-limited at ``LW = 6``; ``LMAX = 24``; ``C^TT, C^EE, C^BB, C^TE`` with
different shapes (``C^BB`` not proportional to ``C^EE``, ``C^TE`` changing
sign), ``C^TB = C^EB = 0``.  Measured:

- B-input column of ``K_2``: ``(K e^B)^E = -u^B``, ``(K e^B)^B = +u^E``
  bit-for-bit (max difference 0.0) at every ``m``; the alternatives ``+u^B``
  and ``-conj(u^B)`` are off by O(1).
- The nine correlators ``R^{XZ}`` match the brute-force columns element-wise
  to <= 6e-16 relative, and the table's ``sum C u^X conj(u^Z)`` form is the
  complex conjugate of ``<a~^X conj(a~^Z)>``, not the correlator itself.
- All 16 ordered blocks among TT, EE, BB, TE at (7,7), (7,9), (9,7), (8,11):
  <= 4.4e-16 relative.
- The current literal rule (spectrum and kernel both labelled by the
  Stokes letters) gives Cov(BB,BB) low by a factor 20 to 70, Cov(EE,EE)
  0.58 to 0.77 of the truth, and has no channel at all for EE x BB, TT x BB,
  TE x BB.

Six spectra (TT, EE, BB, TE, TB, EB; the note's Sects. 9-11), measured
on the same configuration:

- The correlator table generated from the three columns of ``K`` (with the
  true spectrum labelled in leg order, so ``ET`` and ``TE`` stay distinct for
  cross-frequency use) reduces to :data:`CORRELATORS` when ``C^TB = C^EB = 0``.
- All 36 ordered blocks match the oracle with ``C^TB = C^EB = 0`` and with
  ``C^TB, C^EB != 0`` (<= 8.2e-16 of the block's natural scale, <= 1.1e-15
  plain relative on every block that is not a near-zero parity-mixed one);
  the ``BT``, ``BE``, ``ET`` orientations of the observables give the same
  blocks.
- Blocks pairing a parity-even with a parity-odd spectrum (e.g. TT x TB) are
  nonzero with ``C^TB = C^EB = 0`` on a mask without reflection symmetry
  (correlation coefficient 2.7e-5 here) and vanish (4e-13, roundoff) on a
  reflection-symmetric one.
- Kernel sums obey exactly six identities; two of them, parity-odd, are new.
- Leakage law behind the normalisation recommendation: at ``l = l'`` every
  kernel with one ``u^B`` leg on each ``X`` has ``sum Theta = +-1/4 n
  Xi^(w) rho``, ``rho = Xi^{EE->BB} / Xi^{EE->EE}``.
- PolSpice/MASTER channels: ``<C~^TB> = M^TE C^TB``, ``<C~^EB> = M^- C^EB``,
  and ``^x G M^TE = ^x K``, ``^-2 G M^- = ^-2 K``.
"""

import itertools
import math

import numpy as np
import pytest

hp = pytest.importorskip("healpy")

from cmbcov.approximations.acc import (  # noqa: E402
    _CENTRAL_FIELD,
    _PRIME_FIELD,
    CHANNEL_ALIASES,
    COUPLING_SPECTRA,
    contract_coupling_block,
)
from cmbcov.exact import (  # noqa: E402
    _cl_matrix,
    _correlation_columns_gl,
    exact_covariance_row_pol,
)
from cmbcov.grid import (  # noqa: E402
    gl_analysis,
    gl_analysis_complex,
    gl_minimal_lmax,
    gl_synthesis,
    gl_synthesis_complex,
    spin_weighted_integrals_gl,
)
from cmbcov.kernels.coupling import (  # noqa: E402
    KERNEL_TE,
    KERNEL_TT,
    KERNEL_EEmBB,
    KERNEL_EEpBB,
    coupling_kernels,
)
from cmbcov.kernels.polspice import polspice_kernels  # noqa: E402

# Both sides truncate the true field at LMAX identically (the kernels have no
# support above l' + LW = 17), so exact.py's band-limit margin warning, which
# is about approximating an infinite-band sky, does not apply here.
pytestmark = pytest.mark.filterwarnings("ignore:internal band-limit margin")

NSIDE, LW, LMAX = 32, 6, 24
# One GL grid for both sides: different grid sizes move the m = 0 integrals by
# SHT roundoff (up to 1.3e-13 on the baseline mask; docs/theory/bmode_kernels.md).
LG = gl_minimal_lmax(LMAX, LW)
FIELDS = "TEB"
PAIRS = [(7, 7), (7, 9), (9, 7), (8, 11)]
BLOCK_SPECTRA = ("TT", "EE", "BB", "TE")

#: Coupling channels ``ab`` of ``X^{ab}``, in this file's own old-style field
#: letters (``TT, EE, BB, TE, ET`` for today's five, then ``TB, BT, EB, BE``
#: for the four the B-mode blocks add) -- every ``CORRELATORS`` /
#: ``block_terms`` entry below and the ``k[...]`` dict :func:`_kernels`
#: builds are keyed by these, entirely independent of whichever letters the
#: package's own channel names use (``COUPLING_SPECTRA`` uses
#: ``TT, DD, LL, TD, DT``; see :data:`_TODAYS_FIVE_OLD_NAMES` for the one
#: place that still needs the correspondence).  ``a`` is the field of the
#: central (``l``) integral set,
#: ``b`` of the primed (``l'``) one.
CHANNELS = ("TT", "EE", "BB", "TE", "ET", "TB", "BT", "EB", "BE")
CENTRAL = tuple(FIELDS.index(c[0]) for c in CHANNELS)
PRIME = tuple(FIELDS.index(c[1]) for c in CHANNELS)
#: Today's five (:data:`COUPLING_SPECTRA`) under this file's old-style names
#: -- what ``old = set(...)`` below filters against when deciding which
#: kernel pairs are "new" (see :func:`test_extended_indexing_keeps_todays_channels`
#: for the cross-check that they really are the same five channels as the
#: package's, renamed, ``COUPLING_SPECTRA``).
_TODAYS_FIVE_OLD_NAMES = CHANNELS[:5]

#: R^{XZ}_{lm,l'm'} = sum over entries (sign, spectrum, channel) of
#: sign * sum_L C^{spectrum}_L X^{channel}_{mm'}(L)   (C^TB = C^EB = 0).
#: Follows from K e^E = (u^E, u^B), K e^B = (-u^B, u^E) and K Hermitian.
CORRELATORS = {
    "TT": ((+1, "TT", "TT"),),
    "TE": ((+1, "TE", "TE"),),
    "ET": ((+1, "TE", "ET"),),
    "EE": ((+1, "EE", "EE"), (+1, "BB", "BB")),
    "BB": ((+1, "EE", "BB"), (+1, "BB", "EE")),
    "TB": ((-1, "TE", "TB"),),
    "BT": ((-1, "TE", "BT"),),
    "EB": ((-1, "EE", "EB"), (+1, "BB", "BE")),
    "BE": ((-1, "EE", "BE"), (+1, "BB", "EB")),
}


def block_terms(s1, s2):
    r"""
    Cov(C~^{XY}_l, C~^{ZW}_l') = (1/n) sum_{mm'} [R^{XZ} conj R^{YW} +
    R^{XW} conj R^{YZ}], expanded with :data:`CORRELATORS` into
    ``{(spec1, ch1, ch2, spec2): coefficient}`` meaning
    ``coefficient * C^{spec1} . Re Theta^{ch1 x ch2} . C^{spec2} / n``.
    """
    x, y = s1
    z, w = s2
    out = {}
    for left, right in ((x + z, y + w), (x + w, y + z)):
        for c1, sp1, ch1 in CORRELATORS[left]:
            for c2, sp2, ch2 in CORRELATORS[right]:
                key = (sp1, ch1, ch2, sp2)
                out[key] = out.get(key, 0) + c1 * c2
    return {k: v for k, v in out.items() if v}


def _mask_alm():
    npix = hp.nside2npix(NSIDE)
    theta, phi = hp.pix2ang(NSIDE, np.arange(npix))
    vec = np.array(hp.pix2vec(NSIDE, np.arange(npix))).T
    dist = np.arccos(np.clip(vec @ hp.ang2vec(np.radians(50), np.radians(30)), -1, 1))
    edge = np.radians(60)
    cap = np.where(dist < edge, 0.5 * (1 + np.cos(np.pi * dist / edge)), 0.0)
    mask = cap * (1 + 0.3 * np.sin(3 * phi + 1.0) * np.sin(theta))
    return hp.map2alm(mask, lmax=LW, iter=10)


def _spectra():
    ell = np.arange(LMAX + 1)
    tt = np.zeros(LMAX + 1)
    ee = np.zeros(LMAX + 1)
    bb = np.zeros(LMAX + 1)
    tt[2:] = 1.0 / (ell[2:] * (ell[2:] + 1)) ** 0.8
    ee[2:] = 0.3 / (ell[2:] + 3.0) ** 1.7 * (1 + 0.5 * np.cos(ell[2:] / 2.0))
    bb[2:] = 0.05 / (ell[2:] + 1.0) ** 0.9 * (1 + 0.4 * np.sin(ell[2:] / 1.3))
    te = 0.6 * np.sqrt(tt * ee) * np.cos(ell / 3.0)
    return {"TT": tt, "EE": ee, "BB": bb, "TE": te}


@pytest.fixture(scope="module")
def setup():
    alm = _mask_alm()
    integrals = {}

    def u(ell):
        """(3, 2l+1, LMAX+1, 2 LMAX+1): (T, E, B) for every m."""
        if ell not in integrals:
            out = np.zeros((3, 2 * ell + 1, LMAX + 1, 2 * LMAX + 1), complex)
            for i, m in enumerate(range(-ell, ell + 1)):
                out[0, i] = spin_weighted_integrals_gl(
                    alm, LW, ell, m, LMAX, lmax_grid=LG
                )
                out[1:, i] = spin_weighted_integrals_gl(
                    alm, LW, ell, m, LMAX, lmax_grid=LG, spin=2
                )
            integrals[ell] = out
        return integrals[ell]

    return alm, _spectra(), u


def _kernels(u, ell, ellp):
    """{(ch1, ch2): Re sum_{mm'} X^{ch1}(L1) conj X^{ch2}(L2)} through the
    production contraction with the extended flat indexing."""
    left = np.ascontiguousarray(np.transpose(u(ell), (0, 2, 1, 3)))
    right = np.ascontiguousarray(np.transpose(np.conj(u(ellp)), (0, 2, 3, 1)))
    k = contract_coupling_block(left, right, CENTRAL, PRIME)
    return {
        (a, b): k[i, j] for i, a in enumerate(CHANNELS) for j, b in enumerate(CHANNELS)
    }


def _evaluate(terms, kernels, cls, n):
    return (
        sum(c * cls[a] @ kernels[(b, d)] @ cls[e] for (a, b, d, e), c in terms.items())
        / n
    )


def _k2_b_column(alm, ell, m, lmax_out):
    """(K_2 e^B_{lm}) on the grid spin_weighted_integrals_gl would pick."""
    lg = max(math.ceil((LW + ell + lmax_out - 1) / 2), ell, lmax_out)
    w = gl_synthesis(alm, LW, lg)
    e = np.zeros((2, ell + 1, 2 * ell + 1), complex)
    e[1, ell, ell + m] = 1.0
    y = gl_synthesis_complex(e, ell, lg, spin=2)
    return gl_analysis_complex(w * y, lmax_out, lg, spin=2)


# --------------------------------------------------------------------------- #
def test_extended_indexing_keeps_todays_channels():
    """
    The four new channels append to the flat indexing; nothing moves.

    ``COUPLING_SPECTRA`` was renamed (``TT, EE, BB, TE, ET``
    -> ``TT, DD, LL, TD, DT``); ``CHANNEL_ALIASES`` maps this file's own
    old-style names for today's five (:data:`_TODAYS_FIVE_OLD_NAMES`) onto
    it, channel for channel.  ``_CENTRAL_FIELD``/``_PRIME_FIELD`` (``acc.py``)
    now have nine entries themselves -- the rename extended them in place, in
    ``COUPLING_CHANNELS`` order, so their own first five already are today's
    five unchanged; this checks this file's independently-derived ``CENTRAL``/
    ``PRIME`` against that first five.
    """
    assert tuple(CHANNEL_ALIASES[c] for c in _TODAYS_FIVE_OLD_NAMES) == COUPLING_SPECTRA
    assert CENTRAL[:5] == tuple(_CENTRAL_FIELD[:5])
    assert PRIME[:5] == tuple(_PRIME_FIELD[:5])


@pytest.mark.parametrize("ell", [2, 5, 9])
def test_b_input_column_is_the_rotated_e_input(setup, ell):
    """
    ``K_2 = [[W+, -W^B], [W^B, W+]]`` on (E, B): the B-input column is
    ``(-u^B, u^E)`` with no conjugation, for every ``m`` (measured
    difference exactly 0.0).  The wrong-sign and conjugated alternatives
    are off by O(1) of ``|u^E|`` (0.41 to 1.92).
    """
    alm, _, _ = setup
    for m in range(-ell, ell + 1):
        ue, ub = spin_weighted_integrals_gl(alm, LW, ell, m, 16, spin=2)
        be, bb = _k2_b_column(alm, ell, m, 16)
        scale = np.abs(ue).max()
        assert np.abs(be + ub).max() <= 1e-14 * scale
        assert np.abs(bb - ue).max() <= 1e-14 * scale
        assert np.abs(be - ub).max() > 0.3 * scale
        assert np.abs(be + np.conj(ub)).max() > 0.3 * scale


def test_correlators_match_brute_force_columns(setup):
    r"""
    Element-wise check of :data:`CORRELATORS` against
    ``R[X, Z] = <a~^X_{lm} conj(a~^Z_{l'm'})>`` of the exact code
    (``K C K^dagger`` on unit vectors) at ``(l, l') = (7, 9)``.  The table's
    ``sum_L C_L X(L)`` equals ``conj(R)`` (measured <= 6e-16), and differs
    from ``R`` itself by >= 0.85 of max|R|: the ``R = sum C u^X conj(u^Z)`` of
    docs/theory/bmode_kernels.md, Sect. 2, is the conjugated correlator,
    immaterial for the (real) covariance.
    """
    alm, cls, u = setup
    ell, ellp = 7, 9
    wmap = gl_synthesis(alm, LW, LG)
    cmat = _cl_matrix(cls, LMAX)
    for mp in (-4, 0, 3):
        r = _correlation_columns_gl(wmap, cmat, ellp, mp, LMAX, LG, None, tuple(FIELDS))
        for xz, entries in CORRELATORS.items():
            val = np.zeros(2 * ell + 1, complex)
            for sign, spec, (a, b) in entries:
                x = np.einsum(
                    "mLM,LM->mL",
                    u(ell)[FIELDS.index(a)],
                    np.conj(u(ellp)[FIELDS.index(b), mp + ellp]),
                )
                val += sign * x @ cls[spec]
            ref = r[FIELDS.index(xz[0]), FIELDS.index(xz[1]), ell]
            ref = ref[LMAX - ell : LMAX + ell + 1]
            scale = np.abs(ref).max()
            assert np.abs(val - np.conj(ref)).max() < 1e-13 * scale, (mp, xz)
            assert np.abs(val - ref).max() > 0.1 * scale, (mp, xz)


@pytest.mark.parametrize("ell, ellp", PAIRS)
def test_every_block_matches_exact_covariance(setup, ell, ellp):
    """All 16 ordered blocks among TT, EE, BB, TE, with C^BB != 0:
    measured <= 4.4e-16 relative (unshared grids: <= 1.3e-13 on the
    azimuthally symmetric baseline mask, the SHT roundoff at m = 0)."""
    alm, cls, u = setup
    kernels = _kernels(u, ell, ellp)
    n = (2 * ell + 1) * (2 * ellp + 1)
    exact = exact_covariance_row_pol(
        alm, cls, ellp, LMAX, spectra=BLOCK_SPECTRA, lw=LW, lmax_grid=LG
    )
    for s1 in BLOCK_SPECTRA:
        for s2 in BLOCK_SPECTRA:
            got = _evaluate(block_terms(s1, s2), kernels, cls, n)
            ref = exact[(s1, s2)][ell]
            assert (
                abs(got / ref - 1) < 1e-12
            ), f"{s1}x{s2} ({ell},{ellp}): {got / ref - 1:.2e}"


def test_documented_combination_tables():
    """The tables of the derivation note, written out; the transposition
    ``C . Theta^{axb} . C' = C' . Theta^{bxa} . C`` is not applied here."""
    assert block_terms("BB", "BB") == {
        ("EE", "BB", "BB", "EE"): 2,
        ("EE", "BB", "EE", "BB"): 2,
        ("BB", "EE", "BB", "EE"): 2,
        ("BB", "EE", "EE", "BB"): 2,
    }
    assert block_terms("EE", "BB") == {
        ("EE", "EB", "EB", "EE"): 2,
        ("EE", "EB", "BE", "BB"): -2,
        ("BB", "BE", "EB", "EE"): -2,
        ("BB", "BE", "BE", "BB"): 2,
    }
    assert block_terms("TT", "BB") == {("TE", "TB", "TB", "TE"): 2}
    assert block_terms("TE", "BB") == {
        ("TE", "TB", "EB", "EE"): 2,
        ("TE", "TB", "BE", "BB"): -2,
    }
    assert block_terms("EE", "EE") == {
        ("EE", "EE", "EE", "EE"): 2,
        ("EE", "EE", "BB", "BB"): 2,
        ("BB", "BB", "EE", "EE"): 2,
        ("BB", "BB", "BB", "BB"): 2,
    }
    assert block_terms("TE", "TE") == {
        ("TT", "TT", "EE", "EE"): 1,
        ("TT", "TT", "BB", "BB"): 1,
        ("TE", "TE", "ET", "TE"): 1,
    }
    assert block_terms("TE", "EE") == {
        ("TE", "TE", "EE", "EE"): 2,
        ("TE", "TE", "BB", "BB"): 2,
    }


def _canonical_pairs(blocks):
    need = set()
    for s1, s2 in blocks:
        for _, ch1, ch2, _ in block_terms(s1, s2):
            need.add(min((ch1, ch2), (ch2, ch1)))  # Re Theta^{bxa} = (Re Theta^{axb})^T
    return need


def test_minimal_kernel_set():
    """
    Distinct kernel pairs, up to transposition, that the four-spectrum
    covariance needs.  All 16 ordered blocks: 15 pairs of today's channels
    plus 9 new ones.  One block per unordered pair with BB on the right
    (the orientation a symmetric ACC assembly would compute): 6 new ones,
    over the three new channels TB, EB, BE only.
    """
    old = set(_TODAYS_FIVE_OLD_NAMES)
    ordered = _canonical_pairs(itertools.product(BLOCK_SPECTRA, repeat=2))
    new = {p for p in ordered if not set(p) <= old}
    assert len(ordered - new) == 15
    assert new == {
        ("TB", "TB"), ("BT", "BT"),
        ("EB", "TB"), ("BE", "TB"), ("BE", "BT"), ("BT", "EB"),
        ("EB", "EB"), ("BE", "BE"), ("BE", "EB"),
    }  # fmt: skip
    order = ("TT", "EE", "TE", "BB")
    upper = _canonical_pairs((a, b) for i, a in enumerate(order) for b in order[i:])
    new_upper = {p for p in upper if not set(p) <= old}
    assert new_upper == {
        ("TB", "TB"), ("EB", "TB"), ("BE", "TB"),
        ("EB", "EB"), ("BE", "BE"), ("BE", "EB"),
    }  # fmt: skip
    assert {c for p in upper for c in p} - old == {"TB", "EB", "BE"}


def test_literal_stokes_rule_fails(setup):
    """
    Today's rule labels spectrum and kernel by the same Stokes letters
    (``key_to_cross``): Cov(BB,BB) = (2/n) C^BB . Theta^{BBxBB} . C^BB,
    Cov(EE,EE) = (2/n) C^EE . Theta^{EExEE} . C^EE, etc.  Measured
    literal/exact at (7,7), (7,9), (8,11): BBxBB 0.051, 0.028, 0.015;
    EExEE 0.577, 0.670, 0.768; TExTE 0.833, 0.881, 0.920; TExEE 0.784,
    0.846, 0.896.  EExBB, TTxBB and TExBB need the EB and TB channels, which
    do not exist, with C^EB = C^TB = 0: the literal value would be 0.
    """
    alm, cls, u = setup
    literal = {
        ("BB", "BB"): ({("BB", "BB", "BB", "BB"): 2}, 0.06),
        ("EE", "EE"): ({("EE", "EE", "EE", "EE"): 2}, 0.8),
        ("TE", "TE"): (
            {("TT", "TT", "EE", "EE"): 1, ("TE", "TE", "ET", "TE"): 1},
            0.93,
        ),
        ("TE", "EE"): ({("TE", "TE", "EE", "EE"): 2}, 0.9),
    }
    for ell, ellp in [(7, 7), (7, 9), (8, 11)]:
        kernels = _kernels(u, ell, ellp)
        n = (2 * ell + 1) * (2 * ellp + 1)
        exact = exact_covariance_row_pol(
            alm, cls, ellp, LMAX, spectra=BLOCK_SPECTRA, lw=LW, lmax_grid=LG
        )
        for block, (terms, bound) in literal.items():
            ratio = _evaluate(terms, kernels, cls, n) / exact[block][ell]
            assert 0 < ratio < bound, f"{block} ({ell},{ellp}): {ratio:.4f}"
        for block in [("EE", "BB"), ("TT", "BB"), ("TE", "BB")]:
            assert abs(exact[block][ell]) > 1e-3 * abs(exact[("BB", "BB")][ell])
    assert not {"EB", "TB", "BE", "BT"} & set(_TODAYS_FIVE_OLD_NAMES)


@pytest.mark.parametrize("ell, ellp", [(7, 7), (7, 9), (8, 11)])
def test_block_completeness_identities(setup, ell, ellp):
    r"""
    With ``C^EE = C^BB = 1`` the B-mode block combinations collapse onto the
    MASTER channels of ``W^2`` (``l' + 2 LW <= LMAX`` so that completeness is
    exact):

        sum [BBxBB + 2 BBxEE + EExEE]           = n Xi^{EE->EE}[W^2]
        sum [EBxEB - EBxBE - BExEB + BExBE]     = n Xi^{EE->BB}[W^2]

    Measured 1.6e-14 to 8.0e-14 (Xi from the Wigner-3j code, an independent
    route).  These are block-level identities: no single kernel satisfies
    one (EBxEB alone is 0.18 to 0.25 of the second).
    """
    alm, _, u = setup
    assert ellp + 2 * LW <= LMAX
    g = 2 * LW + 5
    w2 = hp.alm2cl(gl_analysis(gl_synthesis(alm, LW, g) ** 2, 2 * LW, g))
    ells = np.arange(LMAX + 1)
    xi = coupling_kernels(w2, LMAX) / (2 * ells + 1)[None, :]
    xi_ee = 0.5 * (xi[KERNEL_EEpBB] + xi[KERNEL_EEmBB])[ell, ellp]
    xi_eb = 0.5 * (xi[KERNEL_EEpBB] - xi[KERNEL_EEmBB])[ell, ellp]
    k = {key: v.sum() for key, v in _kernels(u, ell, ellp).items()}
    n = (2 * ell + 1) * (2 * ellp + 1)
    bbbb = k[("BB", "BB")] + 2 * k[("BB", "EE")] + k[("EE", "EE")]
    eebb = k[("EB", "EB")] - k[("EB", "BE")] - k[("BE", "EB")] + k[("BE", "BE")]
    assert abs(bbbb / (n * xi_ee) - 1) < 1e-11
    assert abs(eebb / (n * xi_eb) - 1) < 1e-11
    assert 0.1 < k[("EB", "EB")] / (n * xi_eb) < 0.5


# =========================================================================== #
# Six spectra: TT, EE, BB, TE, TB, EB  (docs/theory/bmode_kernels.md)
# =========================================================================== #

SIX = ("TT", "EE", "BB", "TE", "TB", "EB")
PARITY_ODD = ("TB", "EB")

#: The columns of ``K`` on unit inputs in terms of the three integral sets
#: (Sect. 2): ``(K e^X)^Y = sign * u^f`` for every ``(Y, f, sign)`` in
#: ``COLUMNS[X]``.  :data:`CORRELATORS` is what this generates when
#: ``C^TB = C^EB = 0``.
COLUMNS = {
    "T": (("T", "T", +1),),
    "E": (("E", "E", +1), ("B", "B", +1)),
    "B": (("E", "B", -1), ("B", "E", +1)),
}


def canonical(spec):
    """``ET -> TE``, ``BT -> TB``, ``BE -> EB`` (single-frequency lookup)."""
    return spec if spec in SIX else spec[::-1]


def correlator(xz, parity_odd=True):
    r"""
    ``R^{XZ} = sum_{Y Y'} C^{Y Y'} (K e^X)^Y conj((K e^Z)^{Y'})`` as
    ``{(true spectrum, channel): coefficient}``.  The true spectrum is
    labelled in leg order (``Y`` from the ``X`` leg, ``Y'`` from the ``Z``
    leg), so ``ET`` and ``TE`` stay distinct, as they must across two
    frequencies.  ``parity_odd=False`` drops the ``C^TB``, ``C^EB`` terms.
    """
    x, z = xz
    out = {}
    for y, f, s in COLUMNS[x]:
        for yp, fp, sp in COLUMNS[z]:
            if not parity_odd and canonical(y + yp) in PARITY_ODD:
                continue
            key = (y + yp, f + fp)
            out[key] = out.get(key, 0) + s * sp
    return {k: v for k, v in out.items() if v}


def block_terms_six(s1, s2, parity_odd=True):
    """:func:`block_terms` with the generated correlators (any observable
    orientation, spectra in leg order)."""
    x, y = s1
    z, w = s2
    out = {}
    for left, right in ((x + z, y + w), (x + w, y + z)):
        for (sp1, ch1), c1 in correlator(left, parity_odd).items():
            for (sp2, ch2), c2 in correlator(right, parity_odd).items():
                key = (sp1, ch1, ch2, sp2)
                out[key] = out.get(key, 0) + c1 * c2
    return {k: v for k, v in out.items() if v}


def _value_and_scale(terms, kernels, cls, n):
    """The block and the sum of the absolute values of its terms."""
    vals = [
        c * cls[canonical(a)] @ kernels[(b, d)] @ cls[canonical(e)] / n
        for (a, b, d, e), c in terms.items()
    ]
    return sum(vals), sum(abs(v) for v in vals)


def _odd_spectra(cls):
    """``cls`` plus ``C^TB``, ``C^EB`` of their own shapes (3x3 matrix
    positive definite at every ``L >= 2``)."""
    ell = np.arange(LMAX + 1)
    out = dict(cls)
    out["TB"] = 0.25 * np.sqrt(cls["TT"] * cls["BB"]) * np.sin(ell / 2.5 + 0.3)
    out["EB"] = 0.3 * np.sqrt(cls["EE"] * cls["BB"]) * np.cos(ell / 1.7)
    return out


def _xi_w2(alm):
    """Xi^{ss'}_{ll'}[W^2] channels of the band-limited mask (as in
    :func:`test_block_completeness_identities`)."""
    g = 2 * LW + 5
    w2 = hp.alm2cl(gl_analysis(gl_synthesis(alm, LW, g) ** 2, 2 * LW, g))
    ells = np.arange(LMAX + 1)
    xi = coupling_kernels(w2, LMAX) / (2 * ells + 1)[None, :]
    return {
        "00": xi[KERNEL_TT],
        "20": xi[KERNEL_TE],
        "EE": 0.5 * (xi[KERNEL_EEpBB] + xi[KERNEL_EEmBB]),
        "EB": 0.5 * (xi[KERNEL_EEpBB] - xi[KERNEL_EEmBB]),
    }


def test_general_correlators_reduce_to_the_table():
    """The generated correlators are :data:`CORRELATORS` when C^TB = C^EB = 0;
    with them non-zero every correlator gains the terms below; and the
    generated 16 four-spectrum blocks are :func:`block_terms`."""
    for xz, entries in CORRELATORS.items():
        reduced = {}
        for (spec, ch), c in correlator(xz, parity_odd=False).items():
            reduced[(canonical(spec), ch)] = reduced.get((canonical(spec), ch), 0) + c
        assert reduced == {(s, ch): c for c, s, ch in entries}, xz
    assert correlator("TE") == {("TE", "TE"): 1, ("TB", "TB"): 1}
    assert correlator("TB") == {("TE", "TB"): -1, ("TB", "TE"): 1}
    assert correlator("EE") == {
        ("EE", "EE"): 1, ("EB", "EB"): 1, ("BE", "BE"): 1, ("BB", "BB"): 1,
    }  # fmt: skip
    assert correlator("BB") == {
        ("EE", "BB"): 1, ("EB", "BE"): -1, ("BE", "EB"): -1, ("BB", "EE"): 1,
    }  # fmt: skip
    assert correlator("EB") == {
        ("EE", "EB"): -1, ("EB", "EE"): 1, ("BE", "BB"): -1, ("BB", "BE"): 1,
    }  # fmt: skip
    for s1 in BLOCK_SPECTRA:
        for s2 in BLOCK_SPECTRA:
            got = {}
            for (a, b, d, e), c in block_terms_six(s1, s2, parity_odd=False).items():
                key = (canonical(a), b, d, canonical(e))
                got[key] = got.get(key, 0) + c
            assert {k: v for k, v in got.items() if v} == block_terms(s1, s2)


@pytest.mark.parametrize("parity_odd", [False, True])
@pytest.mark.parametrize("ell, ellp", PAIRS)
def test_six_spectrum_blocks_match_exact(setup, ell, ellp, parity_odd):
    """
    All 36 ordered blocks among TT, EE, BB, TE, TB, EB, in every observable
    orientation (TE/ET, TB/BT, EB/BE), against the oracle; with C^TB = C^EB
    = 0 and with both non-zero.  Measured <= 8.2e-16 of the larger of the
    block's sum of absolute terms and ``sqrt(Cov(s1,s1) Cov(s2,s2))``, and
    <= 1.1e-15 plain relative on every block except the parity-mixed ones
    with C^TB = C^EB = 0: those are ~1e-5 of their neighbours, and their
    roundoff (2e-26 absolute, from the large ``X`` they are built of) is not
    small relative to themselves.
    """
    alm, cls, u = setup
    cc = _odd_spectra(cls) if parity_odd else cls
    kernels = _kernels(u, ell, ellp)
    n = (2 * ell + 1) * (2 * ellp + 1)
    exact = exact_covariance_row_pol(
        alm, cc, ellp, LMAX, spectra=SIX, lw=LW, lmax_grid=LG
    )
    for s1 in SIX:
        for s2 in SIX:
            ref = exact[(s1, s2)][ell]
            natural = math.sqrt(abs(exact[(s1, s1)][ell] * exact[(s2, s2)][ell]))
            for o1 in {s1, s1[::-1]}:
                for o2 in {s2, s2[::-1]}:
                    terms = block_terms_six(o1, o2, parity_odd)
                    got, scale = _value_and_scale(terms, kernels, cc, n)
                    assert scale > 0
                    tol = 1e-13 * max(scale, natural)
                    assert abs(got - ref) <= tol, (o1, o2, got, ref)


def _cap_only_alm():
    npix = hp.nside2npix(NSIDE)
    vec = np.array(hp.pix2vec(NSIDE, np.arange(npix))).T
    dist = np.arccos(np.clip(vec @ hp.ang2vec(np.radians(50), np.radians(30)), -1, 1))
    edge = np.radians(60)
    cap = np.where(dist < edge, 0.5 * (1 + np.cos(np.pi * dist / edge)), 0.0)
    return hp.map2alm(cap, lmax=LW, iter=10)


def test_parity_mixed_blocks(setup):
    """
    Cov(parity-even, parity-odd) with C^TB = C^EB = 0 is odd under a
    reflection of the mask.  It vanishes for a reflection-symmetric mask
    (the bare cap: measured 4e-13 in correlation units, roundoff) and is
    small but real otherwise (the rippled cap: 2.7e-5; a generic
    two-cap-and-hole mask reached 6e-4).
    """
    alm, cls, _ = setup
    for mask, lo, hi in ((alm, 1e-6, 1e-3), (_cap_only_alm(), 0.0, 1e-11)):
        ex = exact_covariance_row_pol(
            mask, cls, 7, LMAX, spectra=SIX, lw=LW, lmax_grid=LG
        )
        corr = max(
            abs(ex[(a, b)][7]) / math.sqrt(ex[(a, a)][7] * ex[(b, b)][7])
            for a in ("TT", "EE", "BB", "TE")
            for b in PARITY_ODD
        )
        assert lo <= corr < hi, corr


@pytest.mark.parametrize("ell, ellp", [(7, 7), (7, 9), (8, 11)])
def test_all_kernel_sum_identities(setup, ell, ellp):
    r"""
    The complete set of linear identities among the 45 kernel sums and the
    ``n Xi[W^2]`` channels (an SVD over two unrelated masks and 64 ``(l, l')``
    each found exactly this six-dimensional null space, gap 2e-6 -> 5e-14).
    Closed forms exist only for products of the per-``(m, m')`` sums of
    ``X^TT``, ``X^EE + X^BB`` and ``X^BE - X^EB`` (completeness of the
    spin-0 and of the full spin-2 basis); no identity involves a TB or BT
    channel, and none holds for a single leakage kernel.
    """
    alm, _, u = setup
    xi = _xi_w2(alm)
    k = {key: v.sum() for key, v in _kernels(u, ell, ellp).items()}
    n = (2 * ell + 1) * (2 * ellp + 1)

    def check(combo, target):
        value = sum(c * k[p] for p, c in combo.items())
        scale = sum(abs(c * k[p]) for p, c in combo.items())
        assert abs(value - target) <= 1e-11 * max(scale, abs(target)), (
            combo,
            value,
            target,
        )

    check({("TT", "TT"): 1}, n * xi["00"][ell, ellp])
    check({("TT", "EE"): 1, ("TT", "BB"): 1}, n * xi["20"][ell, ellp])
    check({("EE", "EE"): 1, ("EE", "BB"): 2, ("BB", "BB"): 1}, n * xi["EE"][ell, ellp])
    check({("EB", "EB"): 1, ("EB", "BE"): -2, ("BE", "BE"): 1}, n * xi["EB"][ell, ellp])
    # parity-odd, new: TT x (BE - EB) and (EE + BB) x (BE - EB) sum to zero
    check({("TT", "BE"): 1, ("TT", "EB"): -1}, 0.0)
    check({("EE", "BE"): 1, ("EE", "EB"): -1, ("BB", "BE"): 1, ("BB", "EB"): -1}, 0.0)


def _pairs_of(blocks, parity_odd=False):
    return {
        min((ch1, ch2), (ch2, ch1))
        for s1, s2 in blocks
        for _, ch1, ch2, _ in block_terms_six(s1, s2, parity_odd)
    }


def _minimal_over_orientation(blocks):
    """(new pairs, total pairs) minimised over which orientation of each
    off-diagonal block is computed (a symmetric assembly needs one)."""
    old = set(_TODAYS_FIVE_OLD_NAMES)
    diag = [b for b in blocks if b[0] == b[1]]
    off = [b for b in blocks if b[0] != b[1]]
    base = _pairs_of(diag)
    options = [(_pairs_of([b]), _pairs_of([b[::-1]])) for b in off]
    best = None
    for choice in itertools.product((0, 1), repeat=len(off)):
        pairs = set(base).union(*(opt[c] for opt, c in zip(options, choice)))
        new = {p for p in pairs if not set(p) <= old}
        key = (len(new), len(pairs))
        best = key if best is None or key < best else best
    return best


def test_minimal_kernel_set_six_spectra():
    """
    Kernel pairs (up to transposition) for the six spectra, C^TB = C^EB = 0.
    The 13 parity-respecting unordered blocks need one pair beyond the six of
    Sect. 6, TB x BT (the TE x ET analogue in Cov(TB,TB)), so the BT channel
    is required once TB is an observable: 7 new pairs, 18 in all at the best
    orientation.  Adding the 8 parity-mixed blocks: 20 new, 31 in all.  All
    nine channels are ordered pairs of (T, E, B): flat indexing suffices.
    """
    old = set(_TODAYS_FIVE_OLD_NAMES)
    order = ("TT", "EE", "TE", "BB", "TB", "EB")
    upper = [(a, b) for i, a in enumerate(order) for b in order[i:]]
    respecting = [(a, b) for a, b in upper if (a in PARITY_ODD) == (b in PARITY_ODD)]
    assert len(upper) == 21 and len(respecting) == 13
    new = {p for p in _pairs_of(respecting) if not set(p) <= old}
    assert new == {
        ("TB", "TB"), ("EB", "TB"), ("BE", "TB"),
        ("EB", "EB"), ("BE", "BE"), ("BE", "EB"), ("BT", "TB"),
    }  # fmt: skip
    assert _minimal_over_orientation(respecting) == (7, 18)
    assert _minimal_over_orientation(upper) == (20, 31)
    channels = {c for p in _pairs_of(upper) for c in p}
    assert channels == set(CHANNELS)
    # with C^TB, C^EB != 0 every block couples to the parity-mixed kernels
    assert len(_pairs_of(respecting, parity_odd=True)) == 40


@pytest.mark.parametrize("ell", [12, 16])
def test_cross_leakage_quarter_law(setup, ell):
    r"""
    The law behind the normalisation recommendation (Sect. 10).  At
    ``l = l'`` every kernel with one ``u^B`` leg on each ``X`` (channels TB,
    BT, EB, BE on both sides) has ``sum Theta = +-(1/4) n Xi^(w) rho``,
    ``rho = Xi^{EE->BB}/Xi^{EE->EE}``, with ``Xi^(w)`` the spin-weight ladder
    (``Xi^20``, ``Xi^00 (Xi^20/Xi^00)^{3/2}``, ``Xi^{EE->EE}`` for w = 1, 3/2, 2).
    Measured at l = 12, 16: the EB family is +-0.2500 to four digits, the TB
    family 0.2434 to 0.2483 (still approaching 1/4, as ~1/l).  Off the
    diagonal the coefficient departs from 1/4 by ~Delta/l in opposite
    directions for EB x EB and BE x BE (0.2128 and 0.2903 at (12, 14)).
    """
    alm, _, u = setup
    xi = _xi_w2(alm)
    expected = {
        ("TB", "TB"): (+1, "20"), ("TB", "BT"): (-1, "20"),
        ("TB", "EB"): (+1, 1.5), ("TB", "BE"): (-1, 1.5),
        ("EB", "EB"): (+1, "EE"), ("BE", "BE"): (+1, "EE"), ("EB", "BE"): (-1, "EE"),
    }  # fmt: skip

    def coefficient(k, pair, rung, a, b):
        n = (2 * a + 1) * (2 * b + 1)
        rho = xi["EB"][a, b] / xi["EE"][a, b]
        if rung == 1.5:
            x_w = xi["00"][a, b] * (xi["20"][a, b] / xi["00"][a, b]) ** 1.5
        else:
            x_w = xi[rung][a, b]
        return k[pair].sum() / (n * x_w * rho)

    k = _kernels(u, ell, ell)
    for pair, (sign, rung) in expected.items():
        c = coefficient(k, pair, rung, ell, ell)
        assert abs(c - 0.25 * sign) < 0.01, (pair, c)
    k = _kernels(u, ell, ell + 2)
    c_ebeb = coefficient(k, ("EB", "EB"), "EE", ell, ell + 2)
    c_bebe = coefficient(k, ("BE", "BE"), "EE", ell, ell + 2)
    assert c_ebeb < 0.235 < 0.265 < c_bebe
    assert abs(c_ebeb + c_bebe - 0.5) < 0.01


def test_tb_eb_mean_and_polspice_channels(setup):
    r"""
    Part C of the note.  The mean pseudo-spectra: ``<C~^TB> = M^TE C^TB``
    and ``<C~^EB> = M^- C^EB`` (the EEmBB channel, not EE->EE or ``M^+``),
    with no leakage of TE into TB or of EE, BB into EB in the mean (measured
    <= 4e-14, and <= 4e-18 of the auto spectra with C^TB = C^EB = 0).  The
    PolSpice kernels those channels call for deconvolve them onto the
    apodisation kernels: ``^x G M^TE = ^x K``, ``^-2 G M^- = ^-2 K`` (<= 2e-13),
    as the decoupled ``^dec G M^+ = ^-2 K`` does for EE + BB.
    """
    alm, cls, _ = setup
    cc = _odd_spectra(cls)
    wmap = gl_synthesis(alm, LW, LG)
    m_w = coupling_kernels(hp.alm2cl(alm), LMAX)
    for spectra, check in ((cc, True), (cls, False)):
        cmat = _cl_matrix(spectra, LMAX)
        for ell in (5, 9):
            mean = np.zeros((3, 3))
            for mp in range(-ell, ell + 1):
                r = _correlation_columns_gl(
                    wmap, cmat, ell, mp, LMAX, LG, None, tuple(FIELDS)
                )
                mean += r[:, :, ell, LMAX + mp].real
            mean /= 2 * ell + 1
            if check:
                tb = m_w[KERNEL_TE][ell] @ spectra["TB"]
                eb = m_w[KERNEL_EEmBB][ell] @ spectra["EB"]
                assert abs(mean[0, 2] / tb - 1) < 1e-12
                assert abs(mean[1, 2] / eb - 1) < 1e-12
                assert (
                    abs(mean[1, 2] / (m_w[KERNEL_EEpBB][ell] @ spectra["EB"]) - 1) > 0.1
                )
            else:
                assert abs(mean[0, 2]) < 1e-14 * math.sqrt(mean[0, 0] * mean[2, 2])
                assert abs(mean[1, 2]) < 1e-14 * math.sqrt(mean[1, 1] * mean[2, 2])
    lbig, hi = 100, 40
    m_big = coupling_kernels(hp.alm2cl(alm), lbig)
    pk = polspice_kernels(lbig, np.deg2rad(40.0), wl=hp.alm2cl(alm))

    def rel(a, b):
        a, b = a[2 : hi + 1, : hi + 1], b[2 : hi + 1, : hi + 1]
        return np.abs(a - b).max() / np.abs(b).max()

    assert rel(pk.Gx @ m_big[KERNEL_TE], pk.Kx) < 1e-11
    assert rel(pk.Gm2 @ m_big[KERNEL_EEmBB], pk.Km2) < 1e-11
    assert rel(pk.Gdec @ m_big[KERNEL_EEpBB], pk.Km2) < 1e-11
    assert rel(pk.Gplus @ m_big[KERNEL_EEmBB], pk.Km2) > 1e-2
