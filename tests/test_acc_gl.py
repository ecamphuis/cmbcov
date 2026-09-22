r"""
Gauss-Legendre backend for the ACC mode-coupling precompute.

Covers the exact-quadrature path
:func:`cmbcov.approximations.acc.precompute_acc_kernels`
gains with ``grid="gl"``:

- **Parseval**: :func:`~cmbcov.grid.spin_weighted_integrals_gl`
  reproduces the GL quadrature of :math:`W^2 |Y_{\ell m}|^2` to machine
  precision once its output band-limit covers the full support of the
  integrand :math:`W Y_{\ell m}` (band-limit ``lw + ell``).
- **Eq. 22** (Camphuis et al. 2022): ``sum_{l1,l2} Theta = (2l+1)(2l'+1) Xi[W^2]``
  becomes exact (not just ~0.976, as pinned on HEALPix by
  ``tests/test_acc_vs_exact.py::test_theta_normalisation_matches_eq22_up_to_transform_bias``),
  *provided* the ``l1, l2`` sum is not itself truncated below the true
  support ``ell + lw`` of the integrand -- see the module-level ``LW``
  constant below for why this suite uses a smaller mask band-limit than
  ``tests/test_acc_vs_exact.py``'s implicit one.
- **ACC(GL) vs exact(GL)**: two independent routes (the diagonal
  approximation's own machinery vs :func:`exact_covariance_row`) agree to
  ~1e-9, an order of magnitude tighter than the HEALPix ACC-vs-exact
  agreement of ~1e-4 (``tests/test_acc_vs_exact.py``), because both are now
  exact for the same band-limited mask.

The HEALPix default path is untouched by this feature; that is pinned by
``tests/test_acc_coupling.py`` (rerun alongside this file, not duplicated
here).
"""

import os

import healpy as hp
import numpy as np
import pytest

from cmbcov.approximations.acc import (
    _CouplingPrecompute,
    precompute_acc_kernels,
)
from cmbcov.exact import exact_covariance_row
from cmbcov.grid import (
    cross_spectrum_full_m,
    gl_analysis,
    gl_minimal_lmax,
    gl_quadrature_weights,
    gl_synthesis,
    gl_synthesis_complex,
    spin_weighted_integrals_gl,
)
from cmbcov.kernels.coupling import KERNEL_TT, coupling_kernels
from cmbcov.mask import MaskWlm

DATA = os.path.join(os.path.dirname(__file__), "data")

# ACC's working resolution and central/off multipoles, matching
# tests/test_acc_vs_exact.py so the two suites are directly comparable.
NSIDE = 16
ELL = 16
ELLPS = [ELL, ELL + 1, ELL + 3]
LMAX = 2 * NSIDE - 1  # ACC's theta_before_summing has lmax_out = 2*NSIDE - 1

# Mask band-limit for the Eq. 22 / ACC(GL)-vs-exact(GL) tests below.
#
# ACC's Theta only sums l1, l2 up to LMAX = 2*NSIDE - 1 = 31 (fixed by the
# nside chosen for the precompute); the integrand W * Y_{ell m} that builds
# each I_lm is genuinely band-limited at `ell + lw`, with real power out to
# that degree. Eq. 22 is an identity for the l1,l2 sum running over the
# *full* support of that integrand: if `ell + lw > LMAX` the l1,l2 sum
# truncates real power and the ratio settles at 1 - O(5e-5) for this mask,
# not 1 (verified: at the default-ish lw = 3*NSIDE-1 = 47, ell+lw = 63 >> 31
# and the ratio is ~1.00005, not 1 to 1e-8). Choosing LW small enough that
# `max(ELLPS) + LW <= LMAX` removes that truncation entirely, and the ratio
# then reaches ~1e-13 (see test below) -- this is *not* cheating the
# tolerance, it isolates the GL quadrature's exactness (the thing this test
# is actually about) from the separate, expected effect of ACC's finite
# l1,l2 reach.
LW = 10


@pytest.fixture(scope="module")
def small_mask_alm():
    """A short mask alm, reused across the Parseval checks below."""
    mask = hp.read_map(os.path.join(DATA, "baseline_mask.fits"))
    lw = 15
    return hp.map2alm(mask, lmax=lw, iter=10), lw


@pytest.mark.parametrize(
    "ell, m",
    [(5, 0), (5, 3), (8, -4), (8, 8), (12, -12), (12, 0)],
    ids=["l5m0", "l5m3", "l8m-4", "l8m8", "l12m-12", "l12m0"],
)
def test_spin_weighted_integrals_gl_satisfies_parseval(small_mask_alm, ell, m):
    r"""
    :math:`\sum_{LM} |I_{\ell m, LM}|^2 = \int W^2 |Y_{\ell m}|^2 d\Omega`
    exactly, once the output band-limit covers the integrand's true
    band-limit ``lw + ell`` (m=0 is included: it is the mode HEALPix's
    non-iterated ``map2alm`` gets worst, per the module docstring of
    :mod:`cmbcov.grid`).
    """
    mask_alm, lw = small_mask_alm
    lmax_out = lw + ell  # full support of W * Y_{ell m}, nothing truncated
    integrals = spin_weighted_integrals_gl(mask_alm, lw, ell, m, lmax_out)
    lhs = np.sum(np.abs(integrals) ** 2)

    # Independent right-hand side: pointwise GL quadrature of W^2 |Y_lm|^2 on
    # a grid with a safety margin over the minimal exact grid, built from
    # primitives other than the function under test.
    lg_check = gl_minimal_lmax(lmax_out, lw) + 5
    w_map = gl_synthesis(mask_alm, lw, lg_check)
    e = np.zeros((ell + 1, 2 * ell + 1), dtype=complex)
    e[ell, ell + m] = 1.0
    y = gl_synthesis_complex(e, ell, lg_check)
    weights = gl_quadrature_weights(lg_check)
    rhs = np.sum(weights * w_map**2 * np.abs(y) ** 2)

    assert lhs == pytest.approx(rhs, rel=1e-12, abs=1e-16)


@pytest.fixture(scope="module")
def gl_setup():
    """ACC(GL) Theta, an exact Xi[W^2] from the same mask alm, and a toy cl."""
    wlm = MaskWlm("baseline_mask.fits", load_path=os.path.abspath(DATA))
    theta = precompute_acc_kernels(
        wlm,
        None,
        centralell=ELL,
        ellprange=ELLPS,
        nside=NSIDE,
        dryrun=True,
        grid="gl",
        lw=LW,
    )
    # ACC's own mask alm, exactly as _setup_coupling_computation builds it.
    mask_alm = hp.map2alm(wlm.mask, lmax=LW, iter=10)

    # Xi computed EXACTLY from that same band-limited mask alm: synthesise
    # W^2 on a GL grid large enough for the true degree-4*LW integrand
    # (W has degree LW, so W^2 has degree 2*LW, and analysing it up to
    # degree 2*LW needs 2*Lg+1 >= 2*(2*LW)), analyse to get the mask power
    # spectrum, and feed the *exact* MASTER kernel. Unlike
    # tests/test_acc_vs_exact.py's ``hp.anafast(wn**2)``, nothing here goes
    # through a HEALPix quadrature.
    grid_lmax = 2 * LW + 5
    w_map = gl_synthesis(mask_alm, LW, grid_lmax)
    w2_alm = gl_analysis(w_map**2, 2 * LW, grid_lmax)
    w2_cl = hp.alm2cl(w2_alm)
    xi = (
        coupling_kernels(w2_cl, LMAX)[KERNEL_TT]
        / (2 * np.arange(LMAX + 1) + 1)[None, :]
    )

    ell_arr = np.arange(LMAX + 1)
    cl = np.zeros(LMAX + 1)
    cl[2:] = 1e-3 / (ell_arr[2:] * (ell_arr[2:] + 1)) ** 0.9

    return mask_alm, theta, xi, cl


@pytest.mark.parametrize("ellp", ELLPS)
def test_theta_normalisation_matches_eq22_exactly_on_gl(gl_setup, ellp):
    """
    Eq. 22 on the GL path: unlike the HEALPix ratio (~0.95-1.0, pinned by
    ``tests/test_acc_vs_exact.py``), this must reach 1 to 1e-8 once the mask
    band-limit ``LW`` is chosen so ACC's finite l1,l2 reach does not truncate
    the integrand (see the ``LW`` comment above).
    """
    _, theta, xi, _ = gl_setup
    ratio = theta[ellp]["TTxTT"].sum() / (
        (2 * ELL + 1) * (2 * ellp + 1) * xi[ELL, ellp]
    )
    assert abs(ratio - 1.0) < 1e-8, f"sum(Theta)/Eq.22 - 1 = {ratio - 1:.3e}"


@pytest.mark.parametrize("ellp", ELLPS)
def test_acc_gl_matches_exact_gl(gl_setup, ellp):
    """
    ACC(GL) (paper Eq. 23/25, built from Theta) vs exact(GL)
    (:func:`exact_covariance_row`, an independent code path -- see
    ``tests/test_acc_vs_exact.py``'s module docstring). Both are exact for
    this band-limited mask, so they should agree far more tightly than the
    HEALPix pair's ~1e-4 (``tests/test_acc_vs_exact.py``).
    """
    mask_alm, theta, xi, cl = gl_setup
    theta_bar = theta[ellp]["TTxTT"] / theta[ellp]["TTxTT"].sum()  # Eq. 23
    acc = 2.0 * xi[ELL, ellp] * (cl @ theta_bar @ cl)  # Eq. 25
    exact = exact_covariance_row(mask_alm, cl, ellp, LMAX, grid="gl", lw=LW)[ELL]
    assert (
        abs(acc / exact - 1.0) < 1e-9
    ), f"ACC(GL)/exact(GL) - 1 = {acc / exact - 1:.3e}"


def test_gl_kernel_keeps_the_imaginary_cross_spectrum():
    r"""
    The GL branch of ``_compute_ellp_coupling`` keeps the per-``(m, m')``
    cross-spectra ``X_{mm'}(L) = sum_M I_{lm,LM} conj(I_{l'm',LM})`` complex
    and forms ``Theta(L1, L2) = Re sum_{mm'} X(L1) conj(X(L2))`` (paper
    Eq. 20); the HEALPix branch keeps only ``Re X`` and so drops the
    ``Im X(L1) Im X(L2)`` term.  On this suite's test mask the two agree
    because the mask is azimuthally symmetric (``X`` is real to ~1e-17); on a
    rotated copy of the same band-limited mask ``Im X`` is ~0.56 of ``Re X``
    and the dropped term matters.  Measured against :func:`exact_covariance_row`
    on the rotated mask at ``(16, 17)``: ``2/n C Theta C`` with the full
    kernel reproduces the exact covariance to 1e-10, with the Re-only kernel
    it is 17% low, and after the Eq. 23 normalisation (which cancels most of
    it) still 0.6% low -- the same 0.5-0.7% bias the HEALPix branch has for
    a generic mask.  This pins the GL branch's choice.
    """
    from cmbcov.approximations.acc import _gl_integrals

    ell, ellp = ELL, ELL + 1
    mask = hp.read_map(os.path.join(DATA, "baseline_mask.fits"))
    mask_alm = hp.map2alm(mask, lmax=LW, iter=10)
    rotated_alm = hp.Rotator(rot=(37.0, 21.0, 11.0)).rotate_alm(
        mask_alm.copy(), lmax=LW
    )
    ell_arr = np.arange(LMAX + 1)
    cl = np.zeros(LMAX + 1)
    cl[2:] = 1e-3 / (ell_arr[2:] * (ell_arr[2:] + 1)) ** 0.9

    precompute = _CouplingPrecompute(
        MaskWlm("baseline_mask.fits", load_path=os.path.abspath(DATA))
    )

    for label, alm, expected_imag in [
        ("azimuthal", mask_alm, 1e-10),
        ("rotated", rotated_alm, None),
    ]:
        integrals = _gl_integrals(alm, LW, ell, LMAX + 1)
        integrals_p = _gl_integrals(alm, LW, ellp, LMAX + 1)
        x = np.array(
            [
                [cross_spectrum_full_m(t, tp) for tp, _ in integrals_p]
                for t, _ in integrals
            ]
        ).reshape(-1, LMAX + 1)
        imag_ratio = np.abs(x.imag).sum() / np.abs(x.real).sum()
        if expected_imag is not None:
            assert imag_ratio < expected_imag, f"{label}: {imag_ratio:.3e}"
        else:
            assert imag_ratio > 0.3, f"{label}: |Im X| only {imag_ratio:.3e} of |Re X|"

        shipped = precompute._compute_ellp_coupling(
            ell,
            ellp,
            integrals,
            {},
            np.array([]),
            None,
            NSIDE,
            LMAX + 1,
            grid="gl",
            mask_alm=alm,
            lw=LW,
        )["TTxTT"]
        full = (x.T @ x.conj()).real
        re_only = x.real.T @ x.real
        np.testing.assert_allclose(
            shipped, full, rtol=1e-12, atol=1e-14 * np.abs(full).max()
        )

        exact = exact_covariance_row(alm, cl, ellp, LMAX, grid="gl", lw=LW)[ell]
        n = (2 * ell + 1) * (2 * ellp + 1)
        assert abs(2 / n * (cl @ full @ cl) / exact - 1) < 1e-9, label
        raw_re = 2 / n * (cl @ re_only @ cl) / exact
        if label == "rotated":
            assert raw_re < 0.9, f"Re-only kernel should be far off, got {raw_re:.4f}"
