"""
Per-run error budget of the polarised ACC blocks.

This module only *reports*. Nothing here enters the covariance, and every
number is read off what a run already has in memory or in its kernel cache --
the precomputed coupling kernels and ``norm_Xi``. Two error terms are known to
sit on the polarised blocks:

**E -> B leakage (bounded here).** ``norm_Xi`` normalises the kernel of a pair
with the Xi channel that satisfies Eq. 22, and for the spin-2 channels that
identity holds only for E and B *summed*: what is missing is the ``LL`` (old
name ``BB``) part of the completeness relation. Its size is measured directly
from the kernel set in use,

    lambda = sum Theta^{TT x LL} / sum Theta^{TT x DD}   at (l*, l*),

(old channel names ``TT x BB`` / ``TT x EE``) and a block with ``n_E``
polarised legs among its four is biased high by ``(n_E / 2) lambda`` (measured
on an apodised test mask at l* = 250: predicted Cov(EE,EE) +0.98% against +0.99%
observed at l = l*, Cov(TE,TE) +0.49% against +0.51%, Cov(TT,TE) +0.25%
against +0.25%). The ``LL`` kernels this needs are already written by
``precompute_acc_kernels`` -- ``COUPLING_SPECTRA`` has five entries and the
precompute computes all twenty-five ordered pairs by default -- and no
covariance path reads them back (``acc._DEFAULT_KERNEL_STOKES`` is ``TT, DD,
TD, DT``), so the budget costs one extra ``np.load`` of a file the cache
already holds and no new computation. A cache built with an explicit
``acc_precompute.spectra`` list that omits ``LL`` has nothing to read, and the
budget then says so instead of guessing.

**Eq. 33 translation (NOT bounded here).** The error of reusing the ``l*``
kernel at ``l`` grows with ``|l - l*|``; on an apodised test mask it is -0.56% on
Cov(EE,EE) at l = 500 and +3.8% at l = 100, for l* = 250. No cheap quantity a
run can evaluate was found to track it, so none is reported: the budget gives
``|l - l*|`` at the band edges and the measured calibration above, and says
plainly that the translation term is not bounded. Two candidates were measured
against the measured numbers above and both failed:

* the MASTER coupling row translated from ``l*`` to ``l``, ``[sum_d
  Xi_{l*,l*+d} C_{l+d}] / [sum_d Xi_{l,l+d} C_{l+d}] - 1``, squared for the two
  spectrum legs: it gives -0.08% at l = 100 and +0.07% at l = 500 for
  Cov(EE,EE) where the measurement says +3.8% and -0.56% -- wrong sign at
  l = 100, an order of magnitude too small at l = 500 -- and it depends
  strongly on where the row is truncated (TT at l = 100 swings from -0.2% to
  -18% as the half-width goes 30 -> 250);
* the assembled term's own sensitivity to the kernel, ``S(l) = ACC(l,l) /
  [sum_contractions norm_Xi C C] - 1`` (what a locally flat spectrum would
  give, which is kernel-shape independent because Eq. 23 makes the kernel
  sum to one): it does not vanish at ``l = l*``, where the translation error
  is exactly zero by construction (S = +3.9% for Cov(EE,EE) there), and the
  measured error exceeds it at l = 100 (3.8% against S = 1.8%) and at
  l = 600 (0.33% against 0.11%), so it is neither an estimate nor a bound.

**Normalisation consistency (checked here).** ``norm_Xi`` comes from
``Cov.Xi``, the MASTER operator of the mask at its own band limit, while the
kernels are built from the mask truncated at ``acc_precompute.lw``. Eq. 22 ties
the two together exactly when they come from the same mask, so
:func:`normalisation_consistency` compares them (see there).

**Which runs.** Every ACC run gets a budget. The normalisation check needs
only the ``TT x TT`` kernel and ``Xi``, so it applies to all of them (it is
skipped with a note when the cache has no ``TT x TT`` kernel, as for an
EE-only or BB-only run). The leakage and translation terms above describe
polarised blocks made with the Eq. 23 normalisation, so for a TT-only run
and for a run with a B observable (per-Wick-term normalisation, exact at
``l*``; docs/theory/bmode_kernels.md, "Known limits") they are reported as
not applicable, with ``"applicable": False`` in the dict.

Scripts: ad hoc, not included in this repository.
"""

import logging
import textwrap

import numpy as np

from ..kernels.coupling import KERNEL_TT

logger = logging.getLogger(__name__)

__all__ = [
    "SURVEY_MEASUREMENT",
    "error_budget",
    "XI_MISMATCH_WARNING",
    "format_error_budget",
    "leakage_ratio",
    "normalisation_consistency",
    "polarised_leg_count",
]

#: Stokes labels that carry the spin-2 completeness deficit. ``B`` is here for
#: when it joins ``CovKeys``: a B leg is E's partner in the same relation, so
#: it counts the same way.
POLARISED_LABELS = ("E", "B")

#: The kernel pair whose sums define ``lambda``. ``TT x LL`` over ``TT x DD``
#: (old names ``TT x BB`` / ``TT x EE``) isolates the spin-2 deficit with one
#: spin-0 leg held fixed, which is the ratio measured for this budget.
LEAKAGE_PAIRS = (("TT", "LL"), ("TT", "DD"))

#: Relative mismatch between the kernels' own normalisation and ``Cov.Xi`` above
#: which :func:`normalisation_consistency` logs a WARNING. The mismatch is the
#: fraction of the mask's weight that the kernel shape ``Theta / sum Theta``
#: does not carry while ``Xi`` does, so it bounds (it does not equal) the error
#: it puts on an element: the bound is reached only for spectra that vary
#: strongly over the missing part. 1e-2 is where that bound reaches the size of
#: the errors ACC already carries and reports (E->B leakage 0.5-1% per block;
#: Eq. 33 translation up to a few percent), so a smaller mismatch is never the
#: dominant error. Measured, for reference: 1e-4 on the HEALPix nside-16 test
#: configuration, 5e-5 on the same with GL, and 3.4e-3 on the survey
#: configuration (lw = 875, kernels 2 nside = 512), where it is the kernel's
#: L range that cuts the tail, not lw (the mismatch there is 2.7e-5 at
#: d = 0 and grows with d) -- so none of these warn, while a mask with power
#: beyond lw does (1e-2 to 1e-1 for a sharp-edged cap at lw = 8 to 40).
XI_MISMATCH_WARNING = 1e-2

#: What the leakage and translation terms were measured to be on an apodised
#: test mask (nside 512, about 4% of the sky, GL, l* = 250, dmax 5), quoted in the
#: report so the numbers a run prints can be read against something. Purely
#: descriptive: nothing extrapolates from it.
SURVEY_MEASUREMENT = {
    "centralell": 250,
    "lambda": 4.92e-3,
    "leakage_at_lstar": {"EExEE": 9.9e-3, "TExTE": 5.1e-3, "TTxTE": 2.5e-3},
    "translation_EExEE": {100: 3.8e-2, 500: -5.6e-3},
}


def polarised_leg_count(cov_key) -> int:
    """
    ``n_E``: how many of a block's four Stokes legs are polarised.

    A covariance block ``Cov(C^{s1 s2}, C^{s3 s4})`` has four legs, one per
    field entering the two spectra (``CovKey.stoke``). Counting the ``E`` (and
    ``B``) labels among them gives ``n_E = 0`` for TTxTT, 1 for TTxTE, 2 for
    TTxEE and TExTE, 3 for EExTE and 4 for EExEE -- the same weights measured
    block by block on the apodised test mask, where the bias was ``(n_E / 2) lambda``
    to two significant figures. It is derived here, not tabulated.

    Parameters
    ----------
    cov_key : CovKey

    Returns
    -------
    int
    """
    return sum(1 for label in cov_key.stoke if label in POLARISED_LABELS)


def leakage_ratio(strategy, ell: int | None = None, ell_prime: int | None = None):
    """
    ``lambda = sum Theta^{TT x BB} / sum Theta^{TT x EE}`` for the kernel set
    an ACC run is using.

    Parameters
    ----------
    strategy : ACCStrategy
        The strategy whose kernel cache is read, through its own
        ``get_covariance_coupling`` -- the same validated, memoised path the
        covariance itself uses, so nothing is recomputed and nothing is
        written.
    ell, ell_prime : int, optional
        The kernel pair to measure it on. Both default to ``centralell``: at
        the central pair the Eq. 33 translation is the identity, so the ratio
        there is the leakage alone.

    Returns
    -------
    dict
        ``{"lambda": float, "sum_TTxBB": float, "sum_TTxEE": float,
        "kernel_pair": (ell, ell_prime)}``, or ``{"lambda": None,
        "unavailable": reason}`` when the cache holds no ``BB`` kernel (an
        ``acc_precompute.spectra`` list without ``BB``). A missing kernel is
        reported, never substituted.
    """
    central = strategy.cov.config.centralell
    ell = central if ell is None else ell
    ell_prime = ell if ell_prime is None else ell_prime
    try:
        kernels = strategy.get_covariance_coupling(
            ell, ell_prime, pairs=list(LEAKAGE_PAIRS)
        )
    except (OSError, ValueError) as error:
        return {
            "lambda": None,
            "kernel_pair": (ell, ell_prime),
            "unavailable": (
                f"no LL coupling kernel in the cache: {error}. The E->B "
                "leakage cannot be bounded for this run; recompute the "
                "kernels with 'LL' in acc_precompute.spectra (the default "
                "computes all five channels)."
            ),
        }
    sum_bb = float(np.sum(kernels[("TT", "LL")]))
    sum_ee = float(np.sum(kernels[("TT", "DD")]))
    if sum_ee == 0.0:
        return {
            "lambda": None,
            "kernel_pair": (ell, ell_prime),
            "unavailable": "sum Theta^{TT x EE} is zero; lambda is undefined",
        }
    return {
        "lambda": sum_bb / sum_ee,
        "sum_TTxBB": sum_bb,
        "sum_TTxEE": sum_ee,
        "kernel_pair": (ell, ell_prime),
    }


def normalisation_consistency(strategy, threshold: float = XI_MISMATCH_WARNING) -> dict:
    r"""
    Check that the kernels and ``Cov.Xi`` normalise the same mask (Eq. 22).

    The sum rule of Eq. 22 for the ``TT x TT`` kernel of the pair ``(l, l')``
    is ``sum_{L1 L2} Theta_{l l'}(L1, L2) = n Xi^{00}_{l l'}[W^2]`` with
    ``n = (2l+1)(2l'+1)``. It holds to round-off when ``Theta`` and ``Xi`` are
    built from one and the same band-limited mask, and this function measures
    how far it is from holding, at ``(l*, l* + d)`` for every diagonal
    ``d`` in ``range(dmax)`` -- the pairs the run loads:

        max_d | sum Theta^{TT x TT}_{l*, l*+d} / n  /  Xi^{00}_{l*, l*+d} - 1 | .

    Conventions (checked numerically to 1e-11 on a test mask, see
    ``tests/test_acc_xi_consistency.py``): a kernel is stored as the plain
    ``Theta(L1, L2)`` array of ``precompute_acc_kernels``, with no
    normalisation applied on disk, so its full sum is the left side; ``Cov.Xi``
    is ``(4, lmax, lmax)`` indexed ``[channel, l, l']`` with channel
    :data:`~cmbcov.kernels.coupling.KERNEL_TT` = ``Xi^{00}[W^2]``, already
    without the ``(2l'+1)`` of the MASTER matrix.

    What the number means. ``Cov.Xi`` is built from the mask at its own band
    limit (``2 nside`` of the mask), the kernels from the mask truncated at
    ``acc_precompute.lw`` and stored only for ``L1, L2 < 2 nside_acc``. ACC
    takes the amplitude from ``Xi`` and only the unit-sum *shape* from the
    kernel, so a mismatch ``eps`` is not an ``eps`` error on the covariance:
    it is the share of the mask's weight that the shape lacks, and bounds the
    element error (reached only if the spectra vary strongly over that missing
    part). Two things make it non-zero, told apart by the diagonal: a mask
    with power beyond ``lw`` shows at every ``d`` (including ``d = 0``, which
    is returned as ``mismatch_first``); a kernel range ``2 nside_acc`` that
    cuts the tail of ``Theta`` grows with ``d``. On the HEALPix grid the
    number also holds the quadrature error of the kernels; with
    ``term_selection`` set, the dropped terms. The manifest's ``grid``,
    ``lw``, ``nside`` and ``term_selection`` are returned so the number can be
    read against them.

    The cost is one load of each ``TT x TT`` kernel of the run (through the
    strategy's memoised, validated loader, so no new read when the covariance
    has just been computed from them) and one array sum each.

    Parameters
    ----------
    strategy : ACCStrategy
    threshold : float
        Relative mismatch above which a WARNING is logged.

    Returns
    -------
    dict
        ``{"max_relative_mismatch": float, "worst_pair": (l, l'),
        "mismatch_first": float, "diagonals_checked": int,
        "threshold": float, "exceeds": bool,
        "lw", "grid", "nside", "term_selection"}``; or
        ``{"max_relative_mismatch": None, "unavailable": reason}`` when the
        cache has no ``TT x TT`` kernel (an EE-only or BB-only run) or
        anything else prevents the check. Never raises: this is a
        diagnostic, a failure is logged and reported as unavailable.
    """
    try:
        return _normalisation_consistency(strategy, threshold)
    except Exception as error:  # noqa: BLE001 - diagnostics must not fail a run
        reason = f"{type(error).__name__}: {error}"
        logger.warning(
            "ACC normalisation check (Cov.Xi against kernels) failed: %s", reason
        )
        return {
            "max_relative_mismatch": None,
            "threshold": threshold,
            "unavailable": f"the check failed ({reason}).",
        }


def _normalisation_consistency(strategy, threshold: float) -> dict:
    cov = strategy.cov
    central = cov.config.centralell
    xi00 = np.asarray(cov.Xi)[KERNEL_TT]
    diagonals = [d for d in range(cov.config.dmax) if central + d < xi00.shape[1]]
    if not diagonals or central >= xi00.shape[0]:
        return {
            "max_relative_mismatch": None,
            "threshold": threshold,
            "unavailable": "Cov.Xi does not reach the run's kernel pairs.",
        }

    worst, worst_pair, first = -1.0, None, None
    for d in diagonals:
        ell_prime = central + d
        try:
            kernels = strategy.get_covariance_coupling(
                central, ell_prime, pairs=[("TT", "TT")]
            )
        except (OSError, ValueError) as error:
            reason = f"no TT x TT coupling kernel in the cache ({error}). "
            logger.info("ACC normalisation check skipped: %s", reason)
            return {
                "max_relative_mismatch": None,
                "threshold": threshold,
                "unavailable": (
                    reason + "Eq. 22 is only checked on TT x TT, so nothing "
                    "is compared for this run."
                ),
            }
        n = (2 * central + 1) * (2 * ell_prime + 1)
        xi = float(xi00[central, ell_prime])
        if xi == 0.0:
            continue
        mismatch = abs(float(np.sum(kernels[("TT", "TT")])) / n / xi - 1.0)
        if first is None:
            first = mismatch
        if mismatch > worst:
            worst, worst_pair = mismatch, (central, ell_prime)
    if worst_pair is None:
        return {
            "max_relative_mismatch": None,
            "threshold": threshold,
            "unavailable": "Xi^00 vanishes at every kernel pair of the run.",
        }

    manifest = strategy.get_covariance_coupling_manifest(central, central) or {}
    result = {
        "max_relative_mismatch": worst,
        "worst_pair": worst_pair,
        "mismatch_first": first,
        "diagonals_checked": len(diagonals),
        "threshold": threshold,
        "exceeds": worst > threshold,
        "lw": manifest.get("lw"),
        "grid": manifest.get("grid"),
        "nside": manifest.get("nside"),
        "term_selection": manifest.get("term_selection"),
    }
    if result["exceeds"]:
        logger.warning(
            "ACC normalisation: the coupling kernels and Cov.Xi disagree by "
            "%.2e (worst at (l, l') = %s; %.2e at d = 0; threshold %.0e). "
            "Cov.Xi carries mask weight that the kernel shapes lack, so ACC "
            "elements are off by up to about this fraction. If the mismatch "
            "is already large at d = 0, the mask has power beyond the "
            "kernels' lw (%s): raise acc_precompute.lw and recompute the "
            "kernels. If it grows with d, the kernel range 2 nside (nside "
            "%s) cuts the tail: raise acc_precompute.nside. (Kernel grid %s "
            "and term_selection %s can also contribute.)",
            worst,
            worst_pair,
            first,
            threshold,
            result["lw"],
            result["nside"],
            result["grid"],
            result["term_selection"],
        )
    else:
        logger.info(
            "ACC normalisation: kernels and Cov.Xi agree to %.2e (max over %d "
            "diagonals, worst at %s; threshold %.0e).",
            worst,
            len(diagonals),
            worst_pair,
            threshold,
        )
    return result


def error_budget(strategy, covariance_keys, band_edges=None) -> dict:
    """
    The error budget of one ACC run.

    Every ACC run gets one, because the normalisation check (section 3)
    needs only the ``TT x TT`` kernel and ``Cov.Xi``. The leakage and
    translation sections describe polarised blocks made with the Eq. 23
    normalisation, so they apply to a T/E run with a polarised leg only.

    Parameters
    ----------
    strategy : ACCStrategy
    covariance_keys : CovKeys
        The blocks the run computes.
    band_edges : sequence of int, optional
        Bin edges of the reported bandpowers, used only for the ``|l - l*|``
        the translation term is quoted at. Defaults to ``[lmin, lmax - 1]``
        of the run.

    Returns
    -------
    dict
        For a T/E run with a polarised leg: ``method``, ``centralell``,
        ``dmax``, ``lmin``, ``lmax``, ``leakage`` (``lambda`` is ``None``,
        with ``unavailable``, when it could not be bounded), ``normalisation``,
        ``blocks``, ``translation`` and ``measured_reference``.

        For a run to which those two terms do not apply -- TT-only, or with
        a B observable -- the dict has ``"applicable": False`` and
        ``"reason"`` (``"tt_only"`` or ``"b_run"``), the same run header, the
        ``normalisation`` entry, an empty ``blocks``, and ``leakage`` and
        ``translation`` reduced to ``{"applicable": False, "reason": text}``.
        (``leakage["lambda"] is None`` keeps its other meaning, "could not be
        bounded", for the first kind only.)
    """
    from ..covariance import has_b_observable

    blocks = {}
    for cov_key in covariance_keys.keys():
        n_polarised = polarised_leg_count(cov_key)
        if n_polarised:
            blocks[f"{cov_key.stokekey()} {cov_key.freqkey()}"] = {
                "stokes": cov_key.stokekey(),
                "frequencies": cov_key.freqkey(),
                "n_E": n_polarised,
            }
    if has_b_observable(covariance_keys):
        return _budget_without_bias_terms(strategy, "b_run", B_RUN_REASONS)
    if not blocks:
        return _budget_without_bias_terms(strategy, "tt_only", TT_ONLY_REASONS)

    config = strategy.cov.config
    leakage = leakage_ratio(strategy)
    for entry in blocks.values():
        entry["leakage_bias"] = (
            None
            if leakage["lambda"] is None
            else 0.5 * entry["n_E"] * leakage["lambda"]
        )

    lmax = strategy.cov.lmax
    edges = list(band_edges) if band_edges else [config.lmin, lmax - 1]
    central = config.centralell
    offsets = [abs(int(edge) - central) for edge in edges]

    return {
        "method": config.method.value,
        "centralell": central,
        "dmax": config.dmax,
        "lmin": config.lmin,
        "lmax": lmax,
        "leakage": leakage,
        "normalisation": normalisation_consistency(strategy),
        "blocks": blocks,
        "translation": {
            "bounded": False,
            "band_edges": [int(edges[0]), int(edges[-1])],
            "max_offset_from_centralell": max(offsets),
            "proxy": None,
            "note": (
                "The Eq. 33 translation error grows with |l - l*| and is not "
                "bounded per run: no quantity a run can cheaply evaluate was "
                "found to track the measured error (see the module docstring "
                "for the two that were tried and how they failed). Use the "
                "measured mask calibration below as the only guide."
            ),
            "measured_reference": SURVEY_MEASUREMENT["translation_EExEE"],
        },
        "measured_reference": SURVEY_MEASUREMENT,
    }


#: Why the leakage and translation sections do not apply to a TT-only run.
TT_ONLY_REASONS = {
    "leakage": (
        "This run has no polarised leg. The E->B leakage of the Eq. 23 "
        "normalisation biases only blocks with a polarised leg, and TT-only "
        "blocks are unaffected."
    ),
    "translation": (
        "Not reported: the calibration this report quotes for the Eq. 33 "
        "translation was measured on polarised blocks, and this run has no "
        "polarised leg. The translation also acts on TT blocks and is not "
        "bounded per run either."
    ),
}

#: Why they do not apply to a run with a B observable (the reasoning of the
#: former ``ACCStrategy.error_budget``).
B_RUN_REASONS = {
    "leakage": (
        "This run has a B observable, so its blocks use the per-Wick-term "
        "normalisation (docs/theory/bmode_kernels.md), which is exact at l* "
        "and carries no Eq. 23 leakage bias of the (n_E / 2) lambda form "
        "quoted for T/E runs."
    ),
    "translation": (
        "Not bounded per run. The measured accuracy of a run with a B "
        "observable off l* is given in docs/theory/bmode_kernels.md, Sect. 8 "
        "'Known limits'; the translation calibration quoted for T/E runs does "
        "not describe the per-Wick-term rule."
    ),
}


def _budget_without_bias_terms(strategy, kind: str, reasons: dict) -> dict:
    """The budget of a run to which the leakage and translation terms do not
    apply: the run header and the normalisation check, which does."""
    config = strategy.cov.config
    return {
        "applicable": False,
        "reason": kind,
        "method": config.method.value,
        "centralell": config.centralell,
        "dmax": config.dmax,
        "lmin": config.lmin,
        "lmax": strategy.cov.lmax,
        "leakage": {"applicable": False, "reason": reasons["leakage"]},
        "translation": {"applicable": False, "reason": reasons["translation"]},
        "normalisation": normalisation_consistency(strategy),
        "blocks": {},
    }


def format_error_budget(budget: dict) -> str:
    """
    The budget as the text written beside the covariance. Reports; changes
    nothing.
    """
    if budget.get("applicable", True) is False:
        return _format_without_bias_terms(budget)
    leakage = budget["leakage"]
    lines = [
        "ACC polarised error budget",
        "==========================",
        "",
        "Reported, not applied: the covariance is unchanged by this file.",
        "",
        f"centralell (l*) = {budget['centralell']}, dmax = {budget['dmax']}, "
        f"lmin = {budget['lmin']}, lmax = {budget['lmax']}",
        "",
        "1. E->B leakage of the kernel set",
        "   lambda = sum Theta^{TT x BB} / sum Theta^{TT x EE} at "
        f"(l*, l*) = {tuple(leakage['kernel_pair'])}",
    ]
    if leakage["lambda"] is None:
        lines += [f"   NOT AVAILABLE: {leakage['unavailable']}"]
    else:
        lines += [
            f"   lambda = {leakage['lambda']:.3e}"
            f"   (sum TTxBB = {leakage['sum_TTxBB']:.6e}, "
            f"sum TTxEE = {leakage['sum_TTxEE']:.6e})",
            "",
            "   A block with n_E polarised legs among its four is biased HIGH",
            "   by (n_E / 2) lambda; TT-only blocks are unaffected.",
            "",
            f"   {'block':<34}{'n_E':>5}{'relative bias':>16}",
        ]
        for name, entry in sorted(budget["blocks"].items()):
            lines.append(
                f"   {name:<34}{entry['n_E']:>5}{entry['leakage_bias']:>+16.3e}"
            )
    translation = budget["translation"]
    lines += [
        "",
        "2. Eq. 33 translation error: NOT BOUNDED",
        f"   band edges {translation['band_edges']}, "
        f"max |l - l*| = {translation['max_offset_from_centralell']}",
        textwrap.fill(
            translation["note"], width=76, initial_indent="   ", subsequent_indent="   "
        ),
        "",
        "   Measured on an apodised test mask (nside 512, about 4% of the",
        "   sky, GL, l* = 250, dmax 5): lambda = 4.92e-03, Cov(EE,EE) ACC/exact",
        "   +0.99% at l = l*, of which the translation part is -0.56% at",
        "   l = 500 and +3.8% at l = 100.",
        "",
    ]
    lines += _format_normalisation(budget.get("normalisation"))
    return "\n".join(lines)


def _format_without_bias_terms(budget: dict) -> str:
    """The report of a run to which sections 1 and 2 do not apply."""
    lines = [
        "ACC error budget",
        "================",
        "",
        "Reported, not applied: the covariance is unchanged by this file.",
        "",
        f"centralell (l*) = {budget['centralell']}, dmax = {budget['dmax']}, "
        f"lmin = {budget['lmin']}, lmax = {budget['lmax']}",
        "",
    ]
    for number, title, key in (
        (1, "E->B leakage of the kernel set", "leakage"),
        (2, "Eq. 33 translation error", "translation"),
    ):
        lines += [
            f"{number}. {title}: NOT APPLICABLE",
            textwrap.fill(
                budget[key]["reason"],
                width=76,
                initial_indent="   ",
                subsequent_indent="   ",
            ),
            "",
        ]
    return "\n".join(lines + _format_normalisation(budget.get("normalisation")))


def _format_normalisation(norm: dict | None) -> list[str]:
    """Section 3 of the report: the Eq. 22 consistency of kernels and Xi."""
    lines = [
        "3. Normalisation: kernels against Cov.Xi (Eq. 22 sum rule)",
        "   sum Theta^{TT x TT}_{l*,l*+d} / n  vs  Xi^{00}_{l*,l*+d},"
        "  n = (2l+1)(2l'+1)",
    ]
    if norm is None:
        return lines + ["   NOT AVAILABLE: not computed.", ""]
    if norm["max_relative_mismatch"] is None:
        return lines + [
            textwrap.fill(
                "NOT AVAILABLE: " + norm["unavailable"],
                width=76,
                initial_indent="   ",
                subsequent_indent="   ",
            ),
            "",
        ]
    lines.append(
        f"   max relative mismatch = {norm['max_relative_mismatch']:.3e} over "
        f"{norm['diagonals_checked']} diagonals (worst at (l, l') = "
        f"{tuple(norm['worst_pair'])}); warning threshold "
        f"{norm['threshold']:.0e}: " + ("EXCEEDED" if norm["exceeds"] else "ok")
    )
    lines.append(
        f"   at d = 0: {norm['mismatch_first']:.3e}; kernels built with "
        f"lw = {norm['lw']}, nside = {norm['nside']}, grid = {norm['grid']}, "
        f"term_selection = {norm['term_selection']}"
    )
    if norm["exceeds"]:
        lines.append(
            textwrap.fill(
                "Cov.Xi carries mask weight that the kernel shapes lack, so ACC "
                "elements are off by up to about this fraction. Large already "
                "at d = 0: the mask has power beyond the kernels' lw; raise "
                "acc_precompute.lw and recompute the kernels. Growing with d: "
                "the kernel range 2 nside cuts the tail; raise "
                "acc_precompute.nside.",
                width=76,
                initial_indent="   ",
                subsequent_indent="   ",
            )
        )
    lines.append("")
    return lines
