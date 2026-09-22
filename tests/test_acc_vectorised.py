r"""
Old-vs-new equivalence for the vectorised ACC coupling precompute.

A Python double loop over ``(m, m')`` -- one ``cross_spectrum_full_m`` (GL)
or complex full-``M`` ``cross_spectrum_full_m`` (HEALPix) call per channel
per pair, 1.25 million calls at ``l = l' = 250`` -- fills a
``(5, 2l+1, 2l'+1, lmax)`` ``Theta`` table and reduces it with
``_theta_to_coupling_kernels``; that naive double loop is the reference this
module checks the shipped code against.  Both HEALPix and GL keep ``Theta``
complex, so the HEALPix reference here is the complex double loop on the
full-``M`` expansion of the HEALPix integrals, with an ``alm2cl``-based loop
kept as an independent check of its real part.  The whole object is a
single batched contraction, and the shipped code evaluates it as one GEMM
per ``L``, accumulating straight into the kernels so ``Theta`` is never
materialised
(:func:`~cmbcov.approximations.acc._accumulate_coupling_kernels`).

That is a pure rewrite of the arithmetic, so it must reproduce the old result
to round-off.  ``tests/test_acc_coupling.py`` pins the HEALPix branch against a
golden file; this file keeps the naive double loop itself as the reference and
asserts the two agree directly, on a mask that is **not** azimuthally symmetric
so the imaginary part of the GL cross-spectrum is non-zero and actually
contributes (see
``tests/test_acc_gl.py::test_gl_kernel_keeps_the_imaginary_cross_spectrum``).
It also pins the blocking: the kernels must not depend on the memory budget.
"""

import os

import numpy as np
import pytest

healpy = pytest.importorskip("healpy")
import healpy as hp  # noqa: E402
from conftest import compute_cross_spec_cplxmapalm  # noqa: E402

from cmbcov.approximations.acc import (  # noqa: E402
    COUPLING_SPECTRA,
    _coupling_block_sizes,
    _CouplingPrecompute,
    _gl_integrals,
    _paired_m_order,
    precompute_acc_kernels,
)
from cmbcov.grid import (  # noqa: E402
    cross_spectrum_full_m,
    full_from_pair,
    spin_weighted_integrals_gl,
)
from cmbcov.mask import MaskWlm  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "data")

NSIDE = 16
ELL = 16
ELLP = ELL + 1
LMAX = 2 * NSIDE  # acc.py's array-length convention; band-limit LMAX - 1
LW = 10
ROTATION = (37.0, 21.0, 11.0)  # breaks the test mask's azimuthal symmetry


# --------------------------------------------------------------------------
# Reference implementations: the naive double loops, verbatim.
# --------------------------------------------------------------------------


def _naive_gl_theta(integrals, integrals_p, lmax):
    """``(5, 2l+1, 2l'+1, lmax)`` complex Theta, one call per (m, m')."""
    theta = np.zeros(
        (len(COUPLING_SPECTRA), len(integrals), len(integrals_p), lmax),
        dtype=np.complex128,
    )
    for m, (t, eb) in enumerate(integrals):
        for mp, (tp, ebp) in enumerate(integrals_p):
            theta[0, m, mp, :] = cross_spectrum_full_m(t, tp)
            theta[1, m, mp, :] = cross_spectrum_full_m(eb[0], ebp[0])
            theta[2, m, mp, :] = cross_spectrum_full_m(eb[1], ebp[1])
            theta[3, m, mp, :] = cross_spectrum_full_m(t, ebp[0])
            theta[4, m, mp, :] = cross_spectrum_full_m(eb[0], tp)
    return theta


def _naive_healpix_theta(integrals, integrals_p, lmax):
    """
    ``(5, 2l+1, 2l'+1, lmax)`` complex Theta from the ``(2, 3, nalm)`` HEALPix
    integral sets: each set expanded to full-``M`` with ``full_from_pair``
    (T, E, B), then one ``cross_spectrum_full_m`` per channel per (m, m').

    The real part is checked, pair by pair, against an independent
    ``alm2cl``-based brute force -- ``(2L+1) * (alm2cl(x_re, y_re) +
    alm2cl(x_im, y_im))`` (``compute_cross_spec_cplxmapalm``) for TT, EE, BB,
    TE, and the same with the two sets swapped for ET -- which ties the
    full-``M`` expansion, including its ``M < 0`` half and conjugation, to
    healpy's ``alm2cl``.
    """
    theta = np.zeros(
        (len(COUPLING_SPECTRA), len(integrals), len(integrals_p), lmax),
        dtype=np.complex128,
    )
    ls = 2 * np.arange(lmax) + 1
    lmax_out = lmax - 1
    full = [full_from_pair(c[0], c[1], lmax_out) for c in integrals]
    full_p = [full_from_pair(c[0], c[1], lmax_out) for c in integrals_p]

    def magnitude(c):
        # |r| + |s| at +-M: the rounding scale of r + i s and of the M < 0 half,
        # whose Re/Im parts can cancel far below |r| + |s|.
        return np.abs(full_from_pair(np.abs(c[0]) + np.abs(c[1]), 0 * c[0], lmax_out))

    fields = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 0))  # COUPLING_SPECTRA order
    worst = 0.0
    for m, x in enumerate(full):
        mag_x = magnitude(integrals[m])
        for mp, y in enumerate(full_p):
            mag_y = magnitude(integrals_p[mp])
            for k, (a, b) in enumerate(fields):
                theta[k, m, mp, :] = cross_spectrum_full_m(x[a], y[b])

            real = np.empty((len(COUPLING_SPECTRA), lmax))
            real[:4] = ls * compute_cross_spec_cplxmapalm(
                integrals[m], integrals_p[mp], nspec=4
            )
            real[4] = (
                ls * compute_cross_spec_cplxmapalm(integrals_p[mp], integrals[m])[3]
            )
            bound = np.array([np.sum(mag_x[a] * mag_y[b], axis=1) for a, b in fields])
            diff = np.abs(theta[:, m, mp].real - real)
            assert np.all(diff[bound == 0] == 0)
            worst = max(worst, float(np.max(diff[bound > 0] / bound[bound > 0])))
    # Rounding only: measured 6.6e-16 on the rotated test mask at (16, 16) and
    # (16, 17).  (Relative to |Re Theta| itself it reaches 2e-9 on entries at
    # 1e-6 of the maximum, from cancellation, hence this scale.)
    assert worst < 1e-13, f"Re Theta vs alm2cl: max |diff| / bound = {worst:.2e}"
    return theta


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rotated_mask_alm():
    """The test mask, band-limited and rotated off the pole."""
    mask = hp.read_map(os.path.join(DATA, "baseline_mask.fits"))
    alm = hp.map2alm(mask, lmax=LW, iter=10)
    return hp.Rotator(rot=ROTATION).rotate_alm(alm.copy(), lmax=LW)


@pytest.fixture(scope="module")
def rotated_mask_dir(tmp_path_factory):
    """A directory holding a rotated copy of the test mask as a map."""
    out = tmp_path_factory.mktemp("rotated_mask")
    mask = hp.read_map(os.path.join(DATA, "baseline_mask.fits"))
    rotated = hp.Rotator(rot=ROTATION).rotate_map_pixel(mask)
    hp.write_map(str(out / "baseline_mask.fits"), rotated, overwrite=True)
    return str(out)


def _precompute(mask_dir=DATA):
    return _CouplingPrecompute(
        MaskWlm("baseline_mask.fits", load_path=os.path.abspath(mask_dir))
    )


def _assert_all_kernels_match(shipped, reference, rtol=1e-12):
    for sp1 in COUPLING_SPECTRA:
        for sp2 in COUPLING_SPECTRA:
            key = f"{sp1}x{sp2}"
            ref = reference[key]
            np.testing.assert_allclose(
                shipped[key],
                ref,
                rtol=rtol,
                atol=1e-14 * np.abs(ref).max(),
                err_msg=key,
            )


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ellp", [ELL, ELLP], ids=["diagonal", "off-diagonal"])
def test_gl_vectorised_matches_the_naive_double_loop(rotated_mask_alm, ellp):
    """
    All twenty-five GL kernels, against the double loop, on a rotated mask.

    The imaginary part of the ``(m, m')`` cross-spectrum is ~0.5 of the real
    part here (unlike the shipped azimuthally symmetric test mask, where it is
    ~1e-17), so a vectorisation that silently dropped it -- the one real risk
    in this rewrite -- would fail this rather than pass by accident.
    """
    precompute = _precompute()
    integrals = _gl_integrals(rotated_mask_alm, LW, ELL, LMAX)
    integrals_p = (
        integrals if ellp == ELL else _gl_integrals(rotated_mask_alm, LW, ellp, LMAX)
    )

    theta = _naive_gl_theta(integrals, integrals_p, LMAX)
    assert np.abs(theta.imag).sum() > 0.3 * np.abs(theta.real).sum()
    reference = precompute._theta_to_coupling_kernels(theta, ELL, ellp, LMAX)

    shipped = precompute._compute_ellp_coupling(
        ELL,
        ellp,
        integrals,
        {},
        np.array([]),
        None,
        NSIDE,
        LMAX,
        grid="gl",
        mask_alm=rotated_mask_alm,
        lw=LW,
    )
    _assert_all_kernels_match(shipped, reference)


@pytest.mark.parametrize("ellp", [ELL, ELLP], ids=["diagonal", "off-diagonal"])
def test_healpix_vectorised_matches_the_naive_double_loop(rotated_mask_dir, ellp):
    """
    Same for the HEALPix branch, whose Theta is complex too.  On the rotated
    mask its imaginary part is ~0.5 of the real part, so a HEALPix branch that
    kept only ``Re Theta`` would fail this.
    """
    precompute = _precompute(rotated_mask_dir)
    wn_for_ell, nside = precompute.wlm.degrade_mask(NSIDE)

    integrals = precompute._compute_central_integrals(
        ELL, wn_for_ell, nside, grid="healpix"
    )
    if ellp == ELL:
        integrals_p = integrals
    else:
        integrals_p = precompute._compute_central_integrals(
            ellp, wn_for_ell, nside, grid="healpix"
        )

    theta = _naive_healpix_theta(
        [integrals[m] for m in range(2 * ELL + 1)],
        [integrals_p[mp] for mp in range(2 * ellp + 1)],
        LMAX,
    )
    assert np.abs(theta.imag).sum() > 0.3 * np.abs(theta.real).sum()
    reference = precompute._theta_to_coupling_kernels(theta, ELL, ellp, LMAX)

    shipped = precompute._compute_ellp_coupling(
        ELL, ellp, integrals, {}, np.array([]), wn_for_ell, nside, LMAX, grid="healpix"
    )
    _assert_all_kernels_match(shipped, reference)


@pytest.mark.parametrize("grid", ["gl", "healpix"])
def test_kernels_do_not_depend_on_the_memory_budget(
    rotated_mask_alm, rotated_mask_dir, grid
):
    """
    The blocking is an implementation detail: a budget so small that every
    ``(m, m')`` block is a single pair must give the same kernels, bit for bit
    up to the reordering of the accumulation, as a budget large enough for one
    pass.  Rotated mask on both grids, so ``Im Theta`` is exercised.
    """
    precompute = _precompute(DATA if grid == "gl" else rotated_mask_dir)
    kwargs = {"grid": grid, "lw": LW, "mask_alm": rotated_mask_alm}
    if grid == "gl":
        integrals = _gl_integrals(rotated_mask_alm, LW, ELL, LMAX)
        wn_for_ell, nside = None, NSIDE
    else:
        wn_for_ell, nside = precompute.wlm.degrade_mask(NSIDE)
        integrals = precompute._compute_central_integrals(
            ELL, wn_for_ell, nside, grid="healpix"
        )

    def run(max_memory_bytes):
        return precompute._compute_ellp_coupling(
            ELL,
            ELLP,
            integrals,
            {},
            np.array([]),
            wn_for_ell,
            nside,
            LMAX,
            max_memory_bytes=max_memory_bytes,
            **kwargs,
        )

    _assert_all_kernels_match(run(1), run(4 * 1024**3), rtol=1e-13)


def test_gl_streams_integrals_when_they_do_not_fit(rotated_mask_alm):
    """
    Over budget, ``_compute_central_integrals`` returns ``None`` and the
    coefficients are synthesised block by block from the mask alm instead.
    The kernels must be identical either way.
    """
    precompute = _precompute()
    held = precompute._compute_central_integrals(
        ELL,
        None,
        NSIDE,
        grid="gl",
        mask_alm=rotated_mask_alm,
        lw=LW,
        lmax=LMAX,
        max_memory_bytes=4 * 1024**3,
    )
    streamed = precompute._compute_central_integrals(
        ELL,
        None,
        NSIDE,
        grid="gl",
        mask_alm=rotated_mask_alm,
        lw=LW,
        lmax=LMAX,
        max_memory_bytes=1024,
    )
    assert held is not None and streamed is None

    common = {
        "grid": "gl",
        "mask_alm": rotated_mask_alm,
        "lw": LW,
        "max_memory_bytes": 4 * 1024**3,
    }
    from_ram = precompute._compute_ellp_coupling(
        ELL, ELLP, held, {}, np.array([]), None, NSIDE, LMAX, **common
    )
    from_mask = precompute._compute_ellp_coupling(
        ELL, ELLP, None, {}, np.array([]), None, NSIDE, LMAX, **common
    )
    _assert_all_kernels_match(from_mask, from_ram, rtol=1e-13)


def test_healpix_streams_integrals_when_they_do_not_fit(rotated_mask_dir):
    """
    The HEALPix branch follows the GL memory rule: the compact central set
    ``(2l+1, 2, 3, nalm)`` complex128 is held only when it fits in half the
    budget, else ``None`` and every ``m`` is synthesised per block.  The
    kernels must be identical either way, with and without a held primed set.
    """
    precompute = _precompute(rotated_mask_dir)
    wn_for_ell, nside = precompute.wlm.degrade_mask(NSIDE)
    compact = (2 * ELL + 1) * 2 * 3 * hp.Alm.getsize(2 * nside - 1) * 16

    held = precompute._compute_central_integrals(
        ELL, wn_for_ell, nside, grid="healpix", max_memory_bytes=2 * compact
    )
    streamed = precompute._compute_central_integrals(
        ELL, wn_for_ell, nside, grid="healpix", max_memory_bytes=2 * compact - 2
    )
    assert streamed is None
    assert held.shape == (2 * ELL + 1, 2, 3, hp.Alm.getsize(2 * nside - 1))

    budget = 4 * 1024**3
    from_ram = precompute._compute_ellp_coupling(
        ELL,
        ELLP,
        held,
        {},
        np.array([]),
        wn_for_ell,
        nside,
        LMAX,
        grid="healpix",
        max_memory_bytes=budget,
    )
    cache = {}
    from_mask = precompute._compute_ellp_coupling(
        ELL,
        ELLP,
        None,
        cache,
        np.array([]),
        wn_for_ell,
        nside,
        LMAX,
        grid="healpix",
        max_memory_bytes=budget,
    )
    assert cache == {}  # not reused later: the primed set is streamed too
    _assert_all_kernels_match(from_mask, from_ram, rtol=1e-13)

    # A primed set a later ellp reuses is materialised, cached and reused.
    from_cache = precompute._compute_ellp_coupling(
        ELL,
        ELLP,
        None,
        cache,
        np.array([ELLP]),
        wn_for_ell,
        nside,
        LMAX,
        grid="healpix",
        max_memory_bytes=budget,
    )
    assert list(cache) == [ELLP]
    _assert_all_kernels_match(from_cache, from_ram, rtol=1e-13)
    reused = precompute._compute_ellp_coupling(
        ELL,
        ELLP,
        None,
        cache,
        np.array([]),
        wn_for_ell,
        nside,
        LMAX,
        grid="healpix",
        max_memory_bytes=budget,
    )
    _assert_all_kernels_match(reused, from_ram, rtol=1e-13)


@pytest.mark.parametrize("spectra", [None, ("TT",)], ids=["all", "TT-only"])
def test_healpix_precompute_does_not_depend_on_max_memory_gb(rotated_mask_dir, spectra):
    """
    End to end through ``precompute_acc_kernels``: a ``max_memory_gb`` below
    the compact central set (so it is streamed, and the contraction is split
    into many blocks) gives the same HEALPix kernels as a budget that holds
    everything, including a repeated ``ellp`` that exercises the primed cache.
    """
    wlm = MaskWlm("baseline_mask.fits", load_path=rotated_mask_dir)
    common = {
        "centralell": ELL,
        "ellprange": [ELL, ELLP, ELL + 3, ELLP],
        "nside": NSIDE,
        "dryrun": True,
        "grid": "healpix",
        "spectra": spectra,
    }
    compact = (2 * ELL + 1) * 2 * 3 * hp.Alm.getsize(2 * NSIDE - 1) * 16
    tiny_gb = 1.5 * compact / 1024**3  # > the contraction floor, < 2 * compact
    streamed = precompute_acc_kernels(wlm, None, max_memory_gb=tiny_gb, **common)
    in_ram = precompute_acc_kernels(wlm, None, max_memory_gb=4.0, **common)
    for ellp in (ELL, ELLP, ELL + 3):
        assert streamed[ellp].keys() == in_ram[ellp].keys()
        for key, ref in in_ram[ellp].items():
            np.testing.assert_allclose(
                streamed[ellp][key],
                ref,
                rtol=1e-13,
                atol=1e-14 * np.abs(ref).max(),
                err_msg=f"{ellp} {key}",
            )


@pytest.mark.parametrize("spin", [0, 2])
@pytest.mark.parametrize("m", [0, 3, 7, 12])
def test_negative_m_integrals_are_an_exact_reflection(rotated_mask_alm, spin, m):
    r"""
    ``_coefficient_provider`` (GL, ``reflect=True``) synthesises only ``|m|``
    and reflects the ``m < 0`` half,

        ``I_{l,-m,LM} = (-1)^(m+M) conj(I_{l,m,L,-M})`` ,

    which halves the spherical-harmonic transforms -- by far the dominant cost
    of the precompute now the contraction is BLAS.  That identity holds for a
    *real* mask; it is asserted here bit-for-bit, for both spins and on a mask
    with no azimuthal symmetry, because a silent failure would corrupt exactly
    half of every kernel.
    """
    ell, lmax_out = 12, LMAX - 1
    up = spin_weighted_integrals_gl(rotated_mask_alm, LW, ell, m, lmax_out, spin=spin)
    down = spin_weighted_integrals_gl(
        rotated_mask_alm, LW, ell, -m, lmax_out, spin=spin
    )
    order = np.arange(-lmax_out, lmax_out + 1)
    predicted = ((-1.0) ** (m + order)) * np.conj(up[..., ::-1])
    np.testing.assert_array_equal(predicted, down)


def test_paired_m_order_is_a_permutation_putting_plus_m_before_minus_m():
    for ell in (0, 1, 5, 16):
        order = _paired_m_order(2 * ell + 1)
        assert sorted(order) == list(range(2 * ell + 1))
        for d in range(1, ell + 1):
            assert order.index(ell + d) + 1 == order.index(ell - d)


def _target_block_sizes(max_bytes):
    """Blocking at the author's target size: ell = ell' = 250, lmax = 512."""
    lmax, n_m = 512, 501
    per_m = 3 * lmax * (2 * lmax - 1) * 16
    per_pair = len(COUPLING_SPECTRA) * lmax * 16  # complex Theta slab
    # output kernels + the one-entry +m cache each provider keeps
    kernels = len(COUPLING_SPECTRA) ** 2 * lmax * lmax * 8 + 2 * per_m
    nb, nbp = _coupling_block_sizes(n_m, n_m, lmax, per_m, per_pair, max_bytes)
    return nb, nbp, n_m, per_m, per_pair, kernels


@pytest.mark.parametrize("max_bytes", [2 * 1024**3, 8 * 1024**3, 512 * 1024**2])
def test_block_sizes_respect_the_budget(max_bytes):
    """
    The chosen blocking must fit the stated budget, and must not leave room for
    a wider central block (the one that sets how often the primed integrals are
    re-synthesised).
    """
    nb, nbp, n_m, per_m, per_pair, kernels = _target_block_sizes(max_bytes)

    assert 1 <= nb <= n_m and 1 <= nbp <= n_m
    assert (nb + nbp) * per_m + nb * nbp * per_pair + kernels <= max_bytes
    if nb < n_m:
        assert (nb + 1 + nbp) * per_m + (nb + 1) * nbp * per_pair + kernels > max_bytes


def test_a_budget_below_the_hard_floor_warns_rather_than_lying():
    """
    The 25 output kernels alone are 52 MB at lmax=512 and one m of GL integrals
    is 25 MB, so no blocking can honour a 64 MB budget.  Say so instead of
    silently overshooting.
    """
    with pytest.warns(UserWarning, match="below"):
        nb, nbp, *_ = _target_block_sizes(64 * 1024**2)
    assert (nb, nbp) == (1, 1)


@pytest.mark.parametrize("nspec", [5, 9])
@pytest.mark.parametrize("ellp", [250, 269])
def test_block_passes_never_increase_with_the_budget(nspec, ellp):
    """
    A larger budget must never give a worse blocking.  Realistic arguments:
    ``centralell = 250``, ``dmax = 20`` (so ``ellp`` up to 269), ``nside = 256``
    (``lmax = 512``), all three fields; ``nspec = 9`` is the planned B-mode
    extension.  The old two-phase search kept ``nbp = 32`` with ``nb = 1`` at
    ``nspec = 9``, 1.0 GiB (8016 passes) where ``nbp = 16`` gives ``nb = 17``.
    """
    lmax, n_m, n_mp = 512, 501, 2 * ellp + 1
    per_m = 3 * lmax * (2 * lmax - 1) * 16
    per_pair = nspec * lmax * 16
    kernels = nspec**2 * lmax * lmax * 8 + 2 * per_m
    floor = kernels + 2 * per_m + per_pair

    budgets = np.unique(
        np.concatenate(
            [
                np.linspace(floor, 4 * 1024**3, 4000),
                np.linspace(0.9 * 1024**3, 1.1 * 1024**3, 400),
            ]
        ).astype(np.int64)
    )
    previous = None
    for max_bytes in budgets:
        max_bytes = int(max_bytes)
        nb, nbp = _coupling_block_sizes(
            n_m, n_mp, lmax, per_m, per_pair, max_bytes, nspec=nspec
        )
        assert 1 <= nb <= n_m and 1 <= nbp <= n_mp
        assert (nb + nbp) * per_m + nb * nbp * per_pair + kernels <= max_bytes
        passes = -(-n_m // nb) * -(-n_mp // nbp)
        if previous is not None:
            assert passes <= previous[0], (
                f"{max_bytes / 1024**3:.4f} GiB: {passes} passes (nb={nb}, "
                f"nbp={nbp}) after {previous[0]} at "
                f"{previous[1] / 1024**3:.4f} GiB"
            )
        previous = (passes, max_bytes)
