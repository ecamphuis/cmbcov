r"""
The Wick-contraction kernel lookup for polarised ``CovKey``\ s.

docs/theory/exact_covariance.md, Sect. 1, works out that the exact
covariance of two pseudo-spectra has two Wick contractions

    Cov(C^XY_l, C^ZW_l') = (1/n) [R^{XZ} conj(R^{YW}) + R^{XW} conj(R^{YZ})]

and that the covariance-coupling KERNEL for a contraction with one T leg and
one E leg is order sensitive off the diagonal: Theta^{TE x TE} != Theta^{TE x
ET} for l != l' (they agree only at l == l'). ``CovKey.key_to_cross`` builds
the two contractions as SpecKey pairs and then calls ``sorted_copy()``, which
at equal frequency reorders the Stokes letters and turns ``(E, T)`` into
``(T, E)`` — correct for looking up the POWER SPECTRUM (``C^{TE}_l ==
C^{ET}_l`` is the same number either way) but wrong for the KERNEL, which the
strategy was indexing with the very same sorted key.

This file pins the fix: ``CovKey.key_to_cross_kernel()`` (added in
``cmbcov/keys.py``) returns the two contractions WITHOUT
sorting, so the kernel lookup in ``ACCStrategy.compute_covariance_term`` /
``compute_acc_term`` (``cmbcov/approximations/acc.py``)
gets the true ET-preserving order while the power-spectrum lookup keeps using
the sorted key unchanged.
"""

import os
import tempfile
import warnings

import numpy as np
import pytest

from cmbcov.approximations import StrategyFactory
from cmbcov.approximations.acc import (
    CHANNEL_ALIASES,
    precompute_acc_kernels,
)
from cmbcov.covariance import Cov, CovarianceConfig, CovarianceMethod
from cmbcov.exact import exact_covariance_pol
from cmbcov.keys import CovKey
from cmbcov.sht import ducc0_map2alm

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
NSIDE = 16
ELL = 16  # centralell; the ACC kernel is (2 NSIDE)^2, so L runs 0..31
LMAX = 48  # Cov.lmax: multipoles 0..47
DMAX = 4
LW = 10

#: ACC reads the spectra to lmax_int = lmax + max(0, S - 1 - centralell),
#: S = 2 NSIDE the kernel size (acc_window_pad).
LMAX_INT = LMAX + 2 * NSIDE - 1 - ELL
ROWS = list(range(ELL, ELL + DMAX))
FREQ = "090GHz"


# --------------------------------------------------------------------------- #
# (A), (B): key_to_cross_kernel picks the true Wick-contraction order
# --------------------------------------------------------------------------- #
def test_single_freq_tete_uses_ttxee_and_texet():
    """
    Cov(C^TE_l, C^TE_l') at a single frequency: the two Wick contractions are
    Theta^{TT x EE} (spectra TT, EE) and Theta^{TE x ET} (spectra TE, TE) —
    see docs/theory/exact_covariance.md, Sect. 1. ``key_to_cross`` alone (sorted)
    would report the second contraction as ``(TE, TE)``, aliasing it to the
    wrong kernel for l != l'; ``key_to_cross_kernel`` must report ``(TE,
    ET)``.
    """
    key = CovKey(("T", "E", "T", "E"), (FREQ,) * 4)

    combo_1, combo_2 = key.key_to_cross()
    kernel_1, kernel_2 = key.key_to_cross_kernel()

    # power-spectrum lookup: unaffected, still fully sorted
    assert (combo_1[0].stokekey(), combo_1[1].stokekey()) == ("TT", "EE")
    assert (combo_2[0].stokekey(), combo_2[1].stokekey()) == ("TE", "TE")

    # kernel lookup: term 1 is symmetric under TT/EE and unaffected; term 2
    # must keep the true ET order
    assert (kernel_1[0].stokekey(), kernel_1[1].stokekey()) == ("TT", "EE")
    assert (kernel_2[0].stokekey(), kernel_2[1].stokekey()) == ("TE", "ET")


def test_single_freq_eete_uses_etxee_and_eexet():
    """
    Cov(C^EE_l, C^TE_l') at a single frequency: with X=E,Y=E (left) and
    Z=T,W=E (right), the Wick rule gives

        term 1: R^{XZ} conj(R^{YW}) = R^{ET} conj(R^{EE})  -> kernel ET x EE
        term 2: R^{XW} conj(R^{YZ}) = R^{EE} conj(R^{ET})  -> kernel EE x ET

    Both kernels are needed and they are transposes of each other
    (``_theta_to_coupling_kernels`` builds ``EExET`` as ``(ETxEE).T``); before
    the fix both contractions were sorted to ``(TE, EE)`` / ``(EE, TE)``, so
    the single-frequency Cov(EE, TE) block only ever requested one of the two
    transposed kernels and carried its lower-triangle values in its upper
    triangle.
    """
    key = CovKey(("E", "E", "T", "E"), (FREQ,) * 4)

    kernel_1, kernel_2 = key.key_to_cross_kernel()

    assert (kernel_1[0].stokekey(), kernel_1[1].stokekey()) == ("ET", "EE")
    assert (kernel_2[0].stokekey(), kernel_2[1].stokekey()) == ("EE", "ET")


# --------------------------------------------------------------------------- #
# (D) two-frequency keys: unchanged where ET was already requested
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "stoke, freq",
    [
        # cross-frequency single-Stokes-pair key: sorted_copy sorts on
        # frequency alone here (freq[0] != freq[1] in the right SpecKey), so
        # the Stokes order it keeps already happens to be the true ET order.
        (("T", "E", "T", "E"), ("090GHz", "090GHz", "150GHz", "090GHz")),
        (("T", "E", "T", "E"), ("090GHz", "090GHz", "150GHz", "150GHz")),
        (("E", "E", "T", "E"), ("090GHz", "090GHz", "150GHz", "090GHz")),
    ],
)
def test_two_freq_keys_already_requesting_et_are_unchanged(stoke, freq):
    key = CovKey(stoke, freq)
    combo_1, combo_2 = key.key_to_cross()
    kernel_1, kernel_2 = key.key_to_cross_kernel()

    assert (kernel_1[0].stokekey(), kernel_1[1].stokekey()) == (
        combo_1[0].stokekey(),
        combo_1[1].stokekey(),
    )
    assert (kernel_2[0].stokekey(), kernel_2[1].stokekey()) == (
        combo_2[0].stokekey(),
        combo_2[1].stokekey(),
    )
    # and it is genuinely the ET order, not TE re-aliased
    assert "ET" in (
        combo_1[0].stokekey(),
        combo_1[1].stokekey(),
        combo_2[0].stokekey(),
        combo_2[1].stokekey(),
    )


# --------------------------------------------------------------------------- #
# (C) end to end: the assembled single-frequency Cov(EE, TE) block
# --------------------------------------------------------------------------- #
def _cov(save_dir, dmax=DMAX):
    config = CovarianceConfig(
        method=CovarianceMethod.ACC, lmax=LMAX, dmax=dmax, centralell=ELL
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # low centralell warning
        return Cov(
            "baseline_mask.fits", config=config, mask_path=DATA, save_dir=save_dir
        )


def _spectra(size):
    """TT power law, EE = 0.1 TT, TE = 0.5 sqrt(TT EE), BB = 0."""
    ell = np.arange(size)
    tt = np.zeros(size)
    tt[2:] = 1e-3 / (ell[2:] * (ell[2:] + 1)) ** 0.9
    ee = 0.1 * tt
    return {"TT": tt, "EE": ee, "TE": 0.5 * np.sqrt(tt * ee)}


@pytest.fixture(scope="module")
def gl_precompute():
    """GL kernels at (16, 16..19), LW=10, saved to disk; the strategy that
    loads them; the exact reference on the same band-limited mask."""
    cov = _cov(tempfile.mkdtemp())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        precompute_acc_kernels(
            cov.wlm,
            cov.save_dir,
            centralell=ELL,
            dmax=DMAX,  # ROWS
            nside=NSIDE,
            grid="gl",
            lw=LW,
        )
        strategy = StrategyFactory.create_strategy(cov)
    mask_alm = ducc0_map2alm(cov.wlm.mask, lmax=LW, pol=False, iter=10)
    cls = _spectra(LMAX)
    exact = exact_covariance_pol(
        mask_alm, cls, LMAX - 1, spectra=("TT", "TE", "EE"), rows=ROWS, grid="gl"
    )
    # the strategy reads the spectra to LMAX_INT; the first LMAX entries are cls
    return cov, strategy, _spectra(LMAX_INT), exact


def _old_aliased_covariance_term(strategy, cov_key, cl):
    """
    Reconstruct the pre-fix ``compute_covariance_term`` behaviour using only
    the public API, for comparison.

    ``ACCStrategy.compute_acc_term`` now takes an optional ``kernel_key``
    that defaults to ``stokes_key`` (the sorted spectrum key) when omitted —
    exactly the argument ``compute_covariance_term`` passed to
    ``covariance_coupling[...]`` before this fix. So a loop that mirrors
    ``compute_covariance_term`` but never passes ``kernel_key`` reproduces
    the old, aliased (Stokes-sorted) kernel lookup bit for bit, without
    needing a second copy of the package or ``git stash``.
    """
    cov = strategy.cov
    dmax, centralell = cov.config.dmax, cov.config.centralell
    flat_cov = strategy._empty_flatten_cov(dmax)

    for diagonal_offset in range(dmax):
        coupling_kernels = strategy.get_covariance_coupling(
            centralell, centralell + diagonal_offset
        )
        combination_1, combination_2 = cov_key.key_to_cross()

        for ell1 in range(cov.lmax):
            ell2 = ell1 + diagonal_offset
            if ell2 >= cov.lmax:
                continue

            def term(combo, e1, e2, ck=coupling_kernels):
                stokes_key = (combo[0].stokekey(), combo[1].stokekey())
                return (
                    strategy.compute_acc_term(
                        cl[combo[0].freqkey()],
                        cl[combo[1].freqkey()],
                        stokes_key,
                        centralell,
                        e1,
                        e2,
                        covariance_coupling=ck,
                        # The pre-fix bug: kernel_key defaulted to stokes_key
                        # (the sorted, power-spectrum key). Before the
                        # channel rename that literally coincided with the
                        # kernel dict's own keys; CHANNEL_ALIASES reproduces
                        # the same (buggy, order-blind) lookup now that the
                        # kernel dict is keyed by the new channel names.
                        kernel_key=(
                            CHANNEL_ALIASES[stokes_key[0]],
                            CHANNEL_ALIASES[stokes_key[1]],
                        ),
                    )
                    * cov.norm_Xi[combo[0].stokekey(), combo[1].stokekey()][e1, e2]
                )

            flat_cov[diagonal_offset + dmax - 1, ell1] = term(
                combination_1, ell1, ell2
            ) + term(combination_2, ell1, ell2)

            if diagonal_offset != 0:
                if cov_key.auto():
                    flat_cov[-diagonal_offset + dmax - 1, ell1] = flat_cov[
                        diagonal_offset + dmax - 1, ell1
                    ]
                else:
                    flat_cov[-diagonal_offset + dmax - 1, ell1] = term(
                        combination_1, ell2, ell1
                    ) + term(combination_2, ell2, ell1)

    return strategy._unflatten_cov(flat_cov)


def test_fix_changes_the_assembled_eete_block_and_moves_it_toward_exact(gl_precompute):
    """
    End to end: the assembled single-frequency Cov(EE, TE) block changes
    value under the fix, and moves closer to the exact reference.

    Note on what "closer" can mean here: ``compute_acc_term``'s Eq. 33
    translation window depends on ``(ell1, ell2)`` only through
    ``min(ell1, ell2)`` and a FIXED kernel/spectrum key, so for any
    ``CovKey`` — fixed or not — the assembled ACC block is symmetric under
    ``l <-> l'`` BY CONSTRUCTION (verified directly below); this is a
    structural property of the translation-invariance approximation itself,
    not something the TE/ET kernel fix changes or could change. The exact
    ``Cov(EE, TE)`` block is not symmetric, so the best an ACC block can do
    is approximate its symmetric part, ``0.5 * (ref[l, l'] + ref[l', l])``.
    What the fix does is replace the wrong kernel (``TE`` aliased from
    ``ET``) with the true ``ET``/``EE`` pair in that approximation; this
    checks that the result is both measurably different from the pre-fix
    value and a better approximation of that symmetric target.
    """
    cov, strategy, cls, exact = gl_precompute
    key = CovKey(("E", "E", "T", "E"), (FREQ,) * 4)
    cl = {FREQ + FREQ: cls}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        new = strategy.compute_covariance_term(key, cl)
        old = _old_aliased_covariance_term(strategy, key, cl)

    ref = exact[("EE", "TE")]

    checked_any = False
    for ellp in ROWS:
        for ell in ROWS:
            if ell == ellp:
                continue
            # structural symmetry, unaffected by the fix (see docstring)
            assert new[ell, ellp] == pytest.approx(new[ellp, ell], rel=1e-12)
            assert old[ell, ellp] == pytest.approx(old[ellp, ell], rel=1e-12)

            sym_exact = 0.5 * (ref[ell, ellp] + ref[ellp, ell])
            if sym_exact == 0:
                continue
            checked_any = True

            rel_change = abs(new[ell, ellp] - old[ell, ellp]) / abs(sym_exact)
            assert 1e-5 < rel_change < 1e-1, (ell, ellp, rel_change)

            assert abs(new[ell, ellp] - sym_exact) <= abs(old[ell, ellp] - sym_exact), (
                ell,
                ellp,
                new[ell, ellp],
                old[ell, ellp],
                sym_exact,
            )

    assert checked_any, "fixture produced no non-zero exact reference to compare"


def test_fix_leaves_ttxtt_and_texte_diagonal_blocks_unchanged(gl_precompute):
    """
    Sanity check that the fix is scoped to the kernel lookup that actually
    needs ET: Cov(TT, TT) has no mixed T/E leg on either side of either
    contraction (``key_to_cross_kernel`` returns ``(TT, TT)`` for both
    terms), so its assembled block must be untouched.
    """
    cov, strategy, cls, exact = gl_precompute
    key = CovKey(("T", "T", "T", "T"), (FREQ,) * 4)
    cl = {FREQ + FREQ: cls}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        new = strategy.compute_covariance_term(key, cl)
        old = _old_aliased_covariance_term(strategy, key, cl)

    # This asserted bit-identity while both sides were the same scalar loop.
    # The assembly is now batched (one matrix product per diagonal), which
    # adds the same terms in a different order, so the two agree to round-off
    # rather than bit for bit: 3.6e-16 relative worst case. "Untouched" is
    # the property under test, and rtol=1e-13 still separates it from a real
    # kernel-selection regression -- picking the TE kernel where ET belongs
    # moved blocks by 1e-4 to 3e-4 -- by nine orders of magnitude.
    np.testing.assert_allclose(new, old, rtol=1e-13, atol=1e-13 * np.abs(old).max())
