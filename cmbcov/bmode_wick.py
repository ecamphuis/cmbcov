"""
Wick-term generator for the B-mode covariance blocks
(docs/theory/bmode_kernels.md), and the kernel-pair
requirement derived from it.

The generator (:data:`_COLUMNS`, :func:`canonical_spectrum`,
:func:`block_wick_terms`) is ported from
``tests/test_bmode_theta_combinations.py`` (``COLUMNS``, ``canonical``,
``correlator``, ``block_terms_six``), which is where it is checked against
the exact oracle (``exact_covariance_row_pol``); this module is the small,
pure generator itself, so both the tests and the runtime configuration layer
(:mod:`cmbcov.keys`) share one implementation.

The generator decides which ACC coupling-kernel *pairs* a requested set of
observables needs (:func:`required_kernel_pairs`), and is used for the
assembly itself: :func:`covkey_wick_terms` expands one covariance block,
with its frequencies, into the terms the ACC strategy sums, and
:func:`term_class` / :func:`spin_weight` / :func:`leakage_legs` classify
each term for the per-term normalisation rule of
docs/theory/bmode_kernels.md, Sect. 5.
"""

from __future__ import annotations

import itertools
from typing import NamedTuple

from .approximations.acc import CHANNEL_ALIASES

__all__ = [
    "PARITY_ODD_SPECTRA",
    "WickTerm",
    "block_wick_terms",
    "canonical_pair",
    "canonical_spectrum",
    "covkey_wick_terms",
    "leakage_legs",
    "required_kernel_pairs",
    "spin_weight",
    "term_class",
]

#: The six single-frequency spectra this module covers, in "leg
#: order" canonical form (``ET -> TE``, ``BT -> TB``, ``BE -> EB``).
_SIX = ("TT", "EE", "BB", "TE", "TB", "EB")

#: Spectra that change sign under a mask reflection (parity-odd).
PARITY_ODD_SPECTRA = ("TB", "EB")

#: ``(K e^X)^Y = sign * u^f`` for ``X`` in ``{T, E, B}``
#: (docs/theory/bmode_kernels.md): the columns of the mixing
#: matrix ``K``, in the fields ``{T, E, B}`` that
#: :data:`cmbcov.approximations.acc.COUPLING_CHANNELS`
#: renamed to ``{T, D, L}``. This module keeps the old T/E/B letters
#: internally (as the derivation and its test do) and translates to the new
#: channel names only in :func:`required_kernel_pairs`, via
#: :data:`~cmbcov.approximations.acc.CHANNEL_ALIASES`.
_COLUMNS: dict[str, tuple[tuple[str, str, int], ...]] = {
    "T": (("T", "T", +1),),
    "E": (("E", "E", +1), ("B", "B", +1)),
    "B": (("E", "B", -1), ("B", "E", +1)),
}


def canonical_spectrum(spec: str) -> str:
    """``ET -> TE``, ``BT -> TB``, ``BE -> EB`` (single-frequency lookup)."""
    return spec if spec in _SIX else spec[::-1]


def _correlator(xz: str, parity_odd: bool = True) -> dict[tuple[str, str], int]:
    """
    ``R^{XZ} = sum_{Y Y'} C^{Y Y'} (K e^X)^Y conj((K e^Z)^{Y'})`` as
    ``{(true spectrum, channel): coefficient}``. The true spectrum is
    labelled in leg order (``Y`` from the ``X`` leg, ``Y'`` from the ``Z``
    leg), so ``ET`` and ``TE`` stay distinct, as they must across two
    frequencies. ``parity_odd=False`` drops every term whose true spectrum
    is ``TB`` or ``EB`` (``C^TB = C^EB = 0``).
    """
    x, z = xz
    out: dict[tuple[str, str], int] = {}
    for y, f, s in _COLUMNS[x]:
        for yp, fp, sp in _COLUMNS[z]:
            if not parity_odd and canonical_spectrum(y + yp) in PARITY_ODD_SPECTRA:
                continue
            key = (y + yp, f + fp)
            out[key] = out.get(key, 0) + s * sp
    return {k: v for k, v in out.items() if v}


def block_wick_terms(
    s1: str, s2: str, parity_odd: bool = True
) -> dict[tuple[str, str, str, str], int]:
    """
    The Wick terms of ``n Cov(C~^{s1}_l, C~^{s2}_l')``
    (docs/theory/bmode_kernels.md), as ``{(left_spectrum,
    left_channel, right_channel, right_spectrum): coefficient}`` meaning
    ``coefficient * C^{left_spectrum} . Re Theta^{left_channel x
    right_channel} . C^{right_spectrum} / n``.

    ``s1``, ``s2`` and the channels are in this module's OLD-style letters
    (``TT, EE, BB, TE, ET, TB, BT, EB, BE``); a caller that indexes
    :data:`cmbcov.approximations.acc.COUPLING_CHANNELS`
    must translate the channels through
    :data:`~cmbcov.approximations.acc.CHANNEL_ALIASES`
    (:func:`required_kernel_pairs` does this). ``parity_odd=False`` drops
    every term whose spectrum is ``TB`` or ``EB``.
    """
    x, y = s1
    z, w = s2
    out: dict[tuple[str, str, str, str], int] = {}
    for left, right in ((x + z, y + w), (x + w, y + z)):
        for (sp1, ch1), c1 in _correlator(left, parity_odd).items():
            for (sp2, ch2), c2 in _correlator(right, parity_odd).items():
                key = (sp1, ch1, ch2, sp2)
                out[key] = out.get(key, 0) + c1 * c2
    return {k: v for k, v in out.items() if v}


def _is_parity_odd(spec: str) -> bool:
    return canonical_spectrum(spec) in PARITY_ODD_SPECTRA


def _blocks(observables: list[str], parity_mixed_blocks: bool) -> list[tuple[str, str]]:
    """Unordered pairs of ``observables`` (including self-pairs), dropping
    pairs between a parity-even and a parity-odd observable unless
    ``parity_mixed_blocks``."""
    blocks = []
    for i, s1 in enumerate(observables):
        for s2 in observables[i:]:
            if not parity_mixed_blocks and _is_parity_odd(s1) != _is_parity_odd(s2):
                continue
            blocks.append((s1, s2))
    return blocks


def _pairs_of(
    blocks: list[tuple[str, str]], parity_odd: bool, bb_needed: bool
) -> set[tuple[str, str]]:
    """Channel pairs (old letters, canonicalised up to transposition:
    ``Theta^{cd x ab}(L2, L1) = Theta^{ab x cd}(L1, L2)``) the given blocks'
    Wick terms need. Terms whose true spectrum is ``BB`` are dropped unless
    ``bb_needed`` (by default, the C^BB terms of EE x EE, TE x TE, TE x EE
    ... are neglected until an observable contains B)."""
    pairs: set[tuple[str, str]] = set()
    for s1, s2 in blocks:
        for sp1, ch1, ch2, sp2 in block_wick_terms(s1, s2, parity_odd):
            if not bb_needed and (sp1 == "BB" or sp2 == "BB"):
                continue
            pairs.add(min((ch1, ch2), (ch2, ch1)))
    return pairs


def _minimal_pairs(
    blocks: list[tuple[str, str]], parity_odd: bool, bb_needed: bool
) -> set[tuple[str, str]]:
    """
    The channel-pair set of :func:`_pairs_of`, minimised over which
    orientation of each off-diagonal block is used (``Cov(a, b)`` or
    ``Cov(b, a)``): docs/theory/bmode_kernels.md, "best
    orientation". Ported from the test's ``_minimal_over_orientation``, which
    the derivation's 18/31/40 pair counts come from; brute force over the
    off-diagonal blocks (``2**n_off``), fine for the handful of blocks six
    observables give (``n_off <= 15``).
    """
    diag = [b for b in blocks if b[0] == b[1]]
    off = [b for b in blocks if b[0] != b[1]]
    base = _pairs_of(diag, parity_odd, bb_needed)
    options = [
        (
            _pairs_of([b], parity_odd, bb_needed),
            _pairs_of([b[::-1]], parity_odd, bb_needed),
        )
        for b in off
    ]
    best: set[tuple[str, str]] | None = None
    for choice in itertools.product((0, 1), repeat=len(off)):
        pairs = set(base).union(*(opt[c] for opt, c in zip(options, choice)))
        if best is None or len(pairs) < len(best):
            best = pairs
    return best if best is not None else base


def required_kernel_pairs(
    observables: list[str],
    parity_odd_nonzero: bool = False,
    parity_mixed_blocks: bool = False,
) -> set[tuple[str, str]]:
    """
    The ACC coupling-kernel channel pairs (new
    :data:`~cmbcov.approximations.acc.COUPLING_CHANNELS`
    names) a covariance run over ``observables`` needs, from the Wick-term
    generator above.

    Parameters
    ----------
    observables : list of str
        Requested spectra, e.g. ``["TT", "EE", "TE"]`` or ``["TT", "EE",
        "TE", "BB", "TB", "EB"]``. ``ET``/``BT``/``BE`` are accepted as
        aliases of ``TE``/``TB``/``EB``.
    parity_odd_nonzero : bool, default False
        Whether ``C^TB``/``C^EB`` are (possibly) non-zero. Only changes the
        result when ``TB`` or ``EB`` is itself in ``observables`` -- with
        neither requested, those spectrum columns are not read at all,
        so whether they happen to be non-zero in the file cannot
        matter.
    parity_mixed_blocks : bool, default False
        Whether blocks between a parity-even spectrum (``TT, EE, TE, BB``)
        and a parity-odd one (``TB, EB``) are included.

    Returns
    -------
    set of (str, str)
        Unordered channel pairs (each ``min(pair)`` up to swap), suitable
        for :func:`~cmbcov.approximations.acc.precompute_acc_kernels`
        ``pairs=``.

    Notes
    -----
    For an ``observables`` set with no B letter at all, this delegates to
    the pre-existing, unminimised mechanism
    (:meth:`~cmbcov.keys.SpecKey.kernel_stokekey` via
    :meth:`~cmbcov.keys.CovKey.key_to_cross_kernel`),
    which is what an ACC run over ``stokes: [T, E]`` has always loaded --
    the point being that this does not change a single pair a default
    run reads, byte for byte. Level >= 2 (any B observable) instead uses the
    Wick generator above with :func:`_minimal_pairs`; the two mechanisms
    are not required to agree pair-for-pair on the T/E-only sub-blocks of a
    B-mode run, only on their overall counts against the derivation
    (docs/theory/bmode_kernels.md): a design choice available
    because :func:`required_kernel_pairs` does not itself implement the
    assembly that consumes these pairs, so nothing depends on which
    minimisation is used.
    """
    canon = [canonical_spectrum(o) for o in observables]
    if not any("B" in o for o in canon):
        return _te_only_pairs(canon)

    bb_needed = any("B" in o for o in canon)
    parity_odd = parity_odd_nonzero and any(o in PARITY_ODD_SPECTRA for o in canon)
    blocks = _blocks(canon, parity_mixed_blocks)
    if parity_odd:
        # With C^TB, C^EB possibly non-zero every block couples to the
        # parity-mixed kernels; the documented pair count (40,
        # docs/theory/bmode_kernels.md, Sect. 6) is the plain, unminimised
        # one, so this is kept unminimised here to match it exactly.
        # Orientation minimisation (_minimal_pairs) does find a smaller,
        # valid 32-pair cover in this case -- a possible later optimisation,
        # not applied here since it would not match the documented 40.
        old_pairs = _pairs_of(blocks, parity_odd, bb_needed)
    else:
        old_pairs = _minimal_pairs(blocks, parity_odd, bb_needed)
    return {(CHANNEL_ALIASES[a], CHANNEL_ALIASES[b]) for a, b in old_pairs}


def _te_only_pairs(observables: list[str]) -> set[tuple[str, str]]:
    """
    Today's kernel pairs for a T/E-only ``observables`` list: exactly what
    :meth:`~cmbcov.keys.CovKey.key_to_cross_kernel`
    (via :meth:`~cmbcov.keys.SpecKey.kernel_stokekey`)
    already gives every existing ACC run, reusing that code path directly
    rather than re-deriving it, so a default run's kernel pairs cannot
    drift from what that code path already computes. Frequency
    does not change which pairs are needed (:meth:`kernel_stokekey` reads
    only the Stokes letters), but at least two frequencies are needed to
    reach the full pair set a real multi-frequency run uses (an
    ``ET``-labelled spectrum only appears as its own
    :class:`~cmbcov.keys.SpecKey` once two different
    frequencies are involved) -- two dummy frequency labels are used here,
    since three make no further difference (checked).
    """
    from .keys import CovKeys

    letters = sorted({c for obs in observables for c in obs})
    cov_keys = CovKeys(letters, ["f1", "f2"], exclude_asymmetric_stokes=False)
    pairs: set[tuple[str, str]] = set()
    for cov_key in cov_keys.keys():
        for contraction in cov_key.key_to_cross_kernel():
            a, b = (spec.kernel_stokekey() for spec in contraction)
            pairs.add(min((a, b), (b, a)))
    return pairs


# --------------------------------------------------------------------------- #
# Per-block expansion with frequencies, and the term classes of the
# per-Wick-term normalisation (docs/theory/bmode_kernels.md)
# --------------------------------------------------------------------------- #

#: Old-letter name of every new ``COUPLING_CHANNELS`` name (the inverse of
#: :data:`CHANNEL_ALIASES`), used to canonicalise a pair exactly the way
#: :func:`required_kernel_pairs` does: the minimum, in OLD letters, of the
#: pair and its transpose.
_OLD_OF_NEW = {new: old for old, new in CHANNEL_ALIASES.items()}

#: The four cross-leakage channels (one ``u^L`` leg each).
_CROSS_CHANNELS = ("TL", "LT", "DL", "LD")


class WickTerm(NamedTuple):
    """
    One expanded Wick term of ``n Cov(C~^{s1}_l, C~^{s2}_{l'})``:
    ``coefficient * C^{left} . Theta^{channel_1 x channel_2} . C^{right}``.

    ``left`` and ``right`` are ``(stokes letters, (freq_a, freq_b))`` of the
    true spectrum in leg order; ``channel_1``/``channel_2`` are
    :data:`~cmbcov.approximations.acc.COUPLING_CHANNELS`
    names, in the orientation :func:`canonical_pair` stores on disk.
    """

    coefficient: int
    left: tuple[str, tuple[str, str]]
    channel_1: str
    channel_2: str
    right: tuple[str, tuple[str, str]]


def canonical_pair(channel_1: str, channel_2: str) -> tuple[str, str]:
    """
    The orientation of the kernel pair ``(channel_1, channel_2)`` (new names)
    that :func:`required_kernel_pairs` asks the precompute for: the smaller,
    in the OLD channel letters, of the pair and its transpose
    (``Theta^{cd x ab}(L2, L1) = Theta^{ab x cd}(L1, L2)``, derivation
    Eq. 1.1).
    """
    a, b = _OLD_OF_NEW[channel_1], _OLD_OF_NEW[channel_2]
    if (b, a) < (a, b):
        return channel_2, channel_1
    return channel_1, channel_2


def covkey_wick_terms(
    stoke: tuple[str, str, str, str],
    freq: tuple[str, str, str, str],
    parity_odd: bool = False,
) -> list[WickTerm]:
    """
    The expanded Wick terms of one covariance block
    ``Cov(C~^{X_a Y_b}_l, C~^{Z_c W_d}_{l'})``, with frequencies, each in the
    kernel orientation the cache stores.

    ``n Cov = sum_{mm'} [R^{X_a Z_c} conj R^{Y_b W_d} + R^{X_a W_d} conj
    R^{Y_b Z_c}]``, with every correlator expanded by the table of
    docs/theory/bmode_kernels.md (:data:`_COLUMNS`); the
    true spectrum of a correlator ``R^{X_a Z_c}`` carries the frequencies
    ``(a, c)`` in leg order. A term whose pair is not in
    :func:`canonical_pair` orientation is transposed by (1.1),
    ``C^A . Theta^{c1 x c2} . C^B = C^B . Theta^{c2 x c1} . C^A``, which is
    exact. Identical terms are merged, zero coefficients dropped, and the
    result is sorted, so it is deterministic.

    Parameters
    ----------
    stoke, freq : 4-tuples
        As :class:`~cmbcov.keys.CovKey` stores them.
    parity_odd : bool
        Keep the terms whose true spectrum is ``TB``/``EB`` (in either
        order); ``False`` drops them (``C^TB = C^EB = 0``).
    """
    x, y, z, w = stoke
    fx, fy, fz, fw = freq
    merged: dict[tuple, int] = {}
    for (a1, fa1, b1, fb1), (a2, fa2, b2, fb2) in (
        ((x, fx, z, fz), (y, fy, w, fw)),
        ((x, fx, w, fw), (y, fy, z, fz)),
    ):
        left_terms = _correlator(a1 + b1, parity_odd)
        right_terms = _correlator(a2 + b2, parity_odd)
        for (sp1, ch1), c1 in left_terms.items():
            for (sp2, ch2), c2 in right_terms.items():
                n1, n2 = CHANNEL_ALIASES[ch1], CHANNEL_ALIASES[ch2]
                left, right = (sp1, (fa1, fb1)), (sp2, (fa2, fb2))
                if canonical_pair(n1, n2) != (n1, n2):
                    key = (right, n2, n1, left)
                else:
                    key = (left, n1, n2, right)
                merged[key] = merged.get(key, 0) + c1 * c2
    return [
        WickTerm(coef, left, ch1, ch2, right)
        for (left, ch1, ch2, right), coef in sorted(merged.items())
        if coef
    ]


def spin_weight(channel_1: str, channel_2: str) -> float:
    """
    Spin weight ``w`` of a kernel pair (docs/theory/bmode_kernels.md,
    Sect. 5): every non-``T`` field letter weighs 1/2, so ``w(TT) = 0``,
    ``w(TD) = w(DT) = w(TL) = w(LT) = 1/2``, ``w(DD) = w(LL) = w(DL) =
    w(LD) = 1``; ``w`` of the pair is the sum over its two channels.
    """
    return sum(0.5 for letter in channel_1 + channel_2 if letter != "T")


def leakage_legs(channel_1: str, channel_2: str) -> int:
    """``k``, the number of ``u^L`` legs of a kernel pair (its letters ``L``)."""
    return (channel_1 + channel_2).count("L")


def term_class(channel_1: str, channel_2: str) -> str:
    """
    Normalisation class of a kernel pair (docs/theory/bmode_kernels.md), from its two channels alone:

    - ``"eq23"``: ``TT x TT``, Eq. 23 with ``Xi^00``, unchanged;
    - ``"deficit"``: ``k = 0`` with a spin-2 leg; the deficit ``1 - c``
      decays as ``rho`` (Eq. 23 as ``l -> infinity``);
    - ``"auto"``: an ``LL`` channel (auto leakage): ``c`` frozen at ``l*``;
    - ``"cross"``: both channels in ``{TL, LT, DL, LD}``: ``c -> +-1/4``;
    - ``"frozen"``: anything else, i.e. the odd-``k`` pairs without an
      ``LL`` channel (``TT x TL``, ``DD x DL``, ``TD x TL``, ...), which only
      parity-mixed blocks and the non-zero ``C^TB``/``C^EB`` terms use.
      This class has not been separately validated; it gets the
      auto-leakage treatment (``c`` frozen at ``l*``,
      ``Xi^(w) rho^(k/2)`` in ``l``), which keeps them exact at ``l*``.
    """
    if (channel_1, channel_2) == ("TT", "TT"):
        return "eq23"
    if leakage_legs(channel_1, channel_2) == 0:
        return "deficit"
    if "LL" in (channel_1, channel_2):
        return "auto"
    if channel_1 in _CROSS_CHANNELS and channel_2 in _CROSS_CHANNELS:
        return "cross"
    return "frozen"
