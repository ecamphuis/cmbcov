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

Scripts: ad hoc, not included in this repository.
"""

import textwrap

import numpy as np

__all__ = [
    "SURVEY_MEASUREMENT",
    "error_budget",
    "format_error_budget",
    "leakage_ratio",
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


def error_budget(strategy, covariance_keys, band_edges=None) -> dict | None:
    """
    The polarised error budget of one ACC run.

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
    dict or None
        ``None`` if no block of the run has a polarised leg -- a TT-only run
        has no polarised error budget, and reporting an empty one would only
        invite reading a bound into it.
    """
    blocks = {}
    for cov_key in covariance_keys.keys():
        n_polarised = polarised_leg_count(cov_key)
        if n_polarised:
            blocks[f"{cov_key.stokekey()} {cov_key.freqkey()}"] = {
                "stokes": cov_key.stokekey(),
                "frequencies": cov_key.freqkey(),
                "n_E": n_polarised,
            }
    if not blocks:
        return None

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


def format_error_budget(budget: dict) -> str:
    """
    The budget as the text written beside the covariance. Reports; changes
    nothing.
    """
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
    return "\n".join(lines)
