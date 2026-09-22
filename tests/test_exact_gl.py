"""
Exact TT pseudo-Cl covariance on a Gauss-Legendre grid (``grid="gl"``).

``cmbcov.exact`` with ``grid="gl"`` computes the covariance
of the exact-integral pseudo-spectrum estimator for a band-limited mask.  On a
GL grid analysis is the exact inverse and adjoint of synthesis, so the result
is machine-exact and symmetric with no adjoint correction, unlike the HEALPix
path which computes the covariance of the pixelised ``map2alm(iter)``
estimator.  The tests pin the normalisation (full sky), the two symmetries,
the Parseval identity behind the ACC precomputation, the measured
GL-vs-HEALPix difference on the baseline mask, and (slow) agreement with
Monte Carlo simulations.
"""

import os

import numpy as np
import pytest

from cmbcov.sht import ducc0_map2alm

healpy = pytest.importorskip("healpy")

from conftest import apodised_cap, power_law_cl  # noqa: E402

from cmbcov.exact import (  # noqa: E402
    exact_covariance,
    exact_covariance_row,
)
from cmbcov.grid import (  # noqa: E402
    gl_analysis_complex,
    gl_minimal_lmax,
    gl_quadrature_weights,
    gl_synthesis,
    gl_synthesis_complex,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Baseline mask (nside=32) band-limited at Lw=48, pseudo-spectrum up to 32.
NSIDE_BASE = 32
LW = 48
LMAX_BASE = 32

# Monte Carlo comparison, same mask and spectrum as tests/test_exact_covariance.py.
# The sample covariance itself comes from the session-scoped
# mc_covariance_nside32 fixture in tests/conftest.py, shared with that module.
NSIDE = 32
LMAX = 60
N_SIMS = 10000
BAND = (20, 50)


def red_cl(lmax):
    return 1000.0 / (np.arange(lmax + 1) + 1.0) ** 2


@pytest.fixture(scope="module")
def baseline():
    mask = healpy.read_map(os.path.join(DATA_DIR, "baseline_mask.fits"))
    mask_alm = ducc0_map2alm(mask, lmax=LW, iter=10)
    return mask, mask_alm, red_cl(LMAX_BASE)


@pytest.fixture(scope="module")
def setup():
    return apodised_cap(NSIDE), power_law_cl(LMAX)


def test_full_sky_is_cosmic_variance_to_machine_precision():
    """
    With W == 1 (alm = sqrt(4 pi) delta_00, Lw = 0)

        Sigma_ll' = 2 C_l^2 / (2l+1) delta_ll'

    to 1e-12: GL transforms are exact, so unlike the HEALPix path (rtol 1e-6
    in tests/test_exact_covariance.py) nothing but round-off is left.
    """
    lmax = 24
    cl = red_cl(lmax)
    full_sky = np.array([np.sqrt(4 * np.pi)], dtype=complex)
    sigma = exact_covariance(full_sky, cl, lmax, grid="gl")

    ell = np.arange(lmax + 1)
    expected = np.diag(2.0 * cl**2 / (2 * ell + 1))
    np.testing.assert_allclose(sigma, expected, rtol=1e-12, atol=1e-12 * expected.max())


def test_map_and_alm_inputs_are_the_same_estimator(baseline):
    """A HEALPix map is converted with map2alm(iter=10, lmax=lw); passing that
    alm directly must give the identical row, and the matrix the same column."""
    mask, mask_alm, cl = baseline
    from_map = exact_covariance_row(mask, cl, 20, LMAX_BASE, grid="gl", lw=LW)
    from_alm = exact_covariance_row(mask_alm, cl, 20, LMAX_BASE, grid="gl")
    with_nside = exact_covariance_row(
        mask, cl, 20, LMAX_BASE, NSIDE_BASE, grid="gl", lw=LW
    )
    np.testing.assert_allclose(from_map, from_alm, rtol=1e-14)
    np.testing.assert_allclose(with_nside, from_alm, rtol=1e-14)
    matrix = exact_covariance(mask_alm, cl, LMAX_BASE, grid="gl", rows=(20,))
    np.testing.assert_allclose(matrix[:, 20], from_alm, rtol=1e-14)
    assert np.all(matrix[:, 19] == 0)


def test_input_validation(baseline):
    mask, mask_alm, cl = baseline
    with pytest.raises(ValueError):
        exact_covariance_row(mask, cl, 20, LMAX_BASE, grid="gl", lw=LW, nside=16)
    with pytest.raises(ValueError):
        exact_covariance_row(mask_alm, cl, 20, LMAX_BASE, grid="gl", lw=LW + 1)
    with pytest.raises(ValueError):
        exact_covariance_row(mask, cl, 20, LMAX_BASE, NSIDE_BASE, grid="bogus")
    with pytest.raises(ValueError):
        exact_covariance_row(mask, cl, 20, LMAX_BASE)  # healpix needs nside
    with pytest.raises(ValueError):
        exact_covariance_row(mask_alm, cl, 20, LMAX_BASE, grid="gl", lmax_grid=31)


def test_row_column_symmetry_without_adjoint_correction(baseline):
    """
    On GL the quadrature-weighted analysis is the true adjoint of synthesis,
    so K C K^dagger is symmetric by construction and no Neumann-series adjoint
    (the HEALPix ``exact_adjoint``) is needed.  Measured max|S - S^T| / max|S|
    is 3e-18; element-wise the far off-diagonal (1e-21 of the maximum) is
    round-off limited at ~3e-13.
    """
    _, mask_alm, cl = baseline
    rows = (3, 8, 13, 20, 32)
    computed = {
        lp: exact_covariance_row(mask_alm, cl, lp, LMAX_BASE, grid="gl") for lp in rows
    }
    scale = max(np.abs(r).max() for r in computed.values())
    for a in rows:
        for b in rows:
            assert abs(computed[a][b] - computed[b][a]) <= 1e-13 * scale
            np.testing.assert_allclose(computed[a][b], computed[b][a], rtol=1e-11)


def test_m_prime_symmetry(baseline):
    """Summing m' >= 0 with weight 2 must reproduce the full m' loop."""
    _, mask_alm, cl = baseline
    for lp in (0, 1, 7, 18):
        fast = exact_covariance_row(
            mask_alm, cl, lp, LMAX_BASE, grid="gl", use_symmetry=True
        )
        full = exact_covariance_row(
            mask_alm, cl, lp, LMAX_BASE, grid="gl", use_symmetry=False
        )
        np.testing.assert_allclose(fast, full, rtol=1e-12)


def test_minimal_grid_equals_sufficient_grid(baseline):
    """
    The default grid gl_minimal_lmax(lmax, lw) gives the same row as the
    a-priori sufficient grid lmax + lw, which over-resolves by lw/2.

    Row-normalised (max|diff| / max|row|, the spike's metric) the two agree to
    2e-13; the row spans ten decades, so its smallest elements are round-off
    limited at ~3e-11 relative.
    """
    _, mask_alm, cl = baseline
    lp = LMAX_BASE
    assert gl_minimal_lmax(LMAX_BASE, LW) == 56
    minimal = exact_covariance_row(mask_alm, cl, lp, LMAX_BASE, grid="gl")
    large = exact_covariance_row(
        mask_alm, cl, lp, LMAX_BASE, grid="gl", lmax_grid=LMAX_BASE + LW
    )
    assert np.abs(minimal - large).max() < 1e-12 * np.abs(large).max()
    np.testing.assert_allclose(minimal, large, rtol=1e-9)


def test_gl_matches_healpix_estimator_on_diagonal_band(baseline):
    """
    The two grids compute the covariance of two different estimators: HEALPix
    that of the pixelised map2alm(iter) pseudo-spectrum, GL that of the
    exact-integral pseudo-spectrum of the band-limited mask.  On the baseline
    mask at nside=32, Lw=48 they differ by 1.0e-5 across the diagonal band
    (l = 15..22, where Sigma/Sigma_diag ~ 0.7-1.1) and by up to 7e-3 at
    l = 32 where Sigma is 2e-4 of the diagonal.  The 2e-5 bound is that
    measured difference, converging as nside^-2, not a tolerance to loosen.
    """
    mask, mask_alm, cl = baseline
    lp = 20
    gl = exact_covariance_row(mask_alm, cl, lp, LMAX_BASE, grid="gl")
    hpx = exact_covariance_row(mask, cl, lp, LMAX_BASE, NSIDE_BASE)
    rel = np.abs(hpx - gl) / np.abs(gl)
    assert rel[15:23].max() < 2e-5, f"diagonal band: {rel[15:23].max():.2e}"
    assert rel[lp] < 2e-5
    assert rel.max() < 1e-2


def test_parseval_identity_for_acc_precompute(baseline):
    """
    sum_{LM} |I_{lm,LM}|^2 = int W^2 |Y_lm|^2 dOmega  with I = analysis(W Y_lm).

    This is the ACC precomputation primitive.  On the grid Lg = Lw + l (the
    product's band-limit) the identity holds to 1e-15 for every mode; the
    HEALPix precompute at iter=0 loses up to 1e-2 on m=0 modes at nside=32.
    """
    _, mask_alm, _ = baseline
    for ell, m in ((2, 0), (10, 0), (10, 7), (15, 15), (30, 12)):
        lg = LW + ell
        w = gl_synthesis(mask_alm, LW, lg)
        e = np.zeros((ell + 1, 2 * ell + 1), dtype=complex)
        e[ell, ell + m] = 1.0
        product = w * gl_synthesis_complex(e, ell, lg)
        lhs = np.sum(np.abs(gl_analysis_complex(product, lg, lg)) ** 2)
        rhs = np.sum(gl_quadrature_weights(lg) * np.abs(product) ** 2)
        assert abs(lhs - rhs) < 1e-13 * rhs, f"(l, m) = ({ell}, {m}): {lhs - rhs}"


@pytest.mark.slow
def test_gl_exact_matches_simulations(setup, mc_covariance_nside32):
    """
    The GL diagonal must agree with the sample covariance of simulated masked
    skies (hp.anafast on the pixel mask at nside=32) within Monte Carlo noise.
    The GL result is the covariance of the band-limited exact-integral
    estimator (Lw = 3 nside - 1 = 95 here); its 1e-5 difference from the
    pixelised estimator is far below the per-element MC sigma sqrt(2/N).
    """
    mask, cl = setup
    lo, hi = BAND
    _, _, mc = mc_covariance_nside32
    gl = exact_covariance(mask, cl, LMAX, NSIDE, rows=range(lo, hi), grid="gl")

    ratio = (np.diag(gl) / np.diag(mc))[lo:hi]
    sigma = np.sqrt(2.0 / N_SIMS)
    assert abs(ratio.mean() - 1.0) < 2.0 * sigma, (
        f"GL exact/MC band mean {ratio.mean():.4f} " f"(sigma/element {sigma:.4f})"
    )


@pytest.mark.slow
def test_gl_exact_handles_band_limit(setup, mc_covariance_nside32):
    """At l' = lmax the simulated skies have no power above lmax; the GL
    calculation uses the truncated C_L and must still match the MC."""
    mask, cl = setup
    _, _, mc = mc_covariance_nside32
    row = exact_covariance_row(mask, cl, LMAX, LMAX, NSIDE, grid="gl")
    ratio = row[LMAX] / mc[LMAX, LMAX]
    sigma = np.sqrt(2.0 / N_SIMS)
    assert abs(ratio - 1.0) < 3.0 * sigma, f"GL exact/MC at lmax: {ratio:.4f}"
