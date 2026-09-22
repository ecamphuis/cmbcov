"""
Exact polarised pseudo-Cl covariance on the Gauss-Legendre grid
(``cmbcov.exact.exact_covariance_row_pol`` /
``exact_covariance_pol``).

The polarised algorithm applies ``K = blockdiag(K_0, K_2)`` to unit vectors
in T, E and B, multiplies by the 3x3 spectrum matrix and applies ``K`` again,
then contracts the correlators ``R^{XZ}`` with Wick's theorem.  The tests pin
the normalisation and the Wick contraction (full-sky identities for every
block), the two symmetries (row/column and m' -> -m'), the minimal-grid rule
for spin 2, agreement with the TT-only implementation, the HEALPix-path
guard, and (slow) agreement with ``healpy.synfast``/``anafast`` Monte Carlo
simulations of masked polarised skies -- including the pure E -> B leakage
variance, which is the sharpest test of the spin-2 bookkeeping.

The fixtures are local on purpose: healpy is the independent oracle, and this
module must not depend on the shared ``tests/conftest.py`` fixtures.
"""

import time

import numpy as np
import pytest

healpy = pytest.importorskip("healpy")

from cmbcov.exact import (  # noqa: E402
    SPECTRA,
    exact_covariance,
    exact_covariance_pol,
    exact_covariance_row,
    exact_covariance_row_pol,
)
from cmbcov.grid import gl_minimal_lmax  # noqa: E402

NSIDE = 32
LMAX = 60
LW = 3 * NSIDE - 1
ALL_SPECTRA = ("TT", "TE", "EE", "BB", "TB", "EB")

# Monte Carlo comparison (slow).
N_SIMS = 10000
MC_ROWS = (20, 40)
MC_NOFF = 4  # compare l' - 4 .. l' + 4 of each row
MC_ORDER = ("TT", "EE", "BB", "TE")  # synfast input / anafast output[:4]
MC_BLOCKS = (
    ("EE", "EE"),
    ("TE", "TE"),
    ("TT", "EE"),
    ("TE", "EE"),
    ("TE", "TT"),
    ("BB", "BB"),
    ("TT", "TT"),
)
MC_SIGMA_BOUND = 4.0


def apodised_cap(nside):
    """40 degree cosine-apodised polar cap (same as the TT MC tests)."""
    npix = healpy.nside2npix(nside)
    theta, _ = healpy.pix2ang(nside, np.arange(npix))
    edge = np.radians(40)
    mask = np.zeros(npix)
    cap = theta < edge
    mask[cap] = 0.5 * (1.0 + np.cos(np.pi * theta[cap] / edge))
    return mask


def spectra(lmax, bb=0.02):
    """Power-law TT, EE = 0.1 TT, TE = 0.5 sqrt(TT EE), BB = bb * TT."""
    ell = np.arange(lmax + 1)
    tt = np.zeros(lmax + 1)
    tt[2:] = 1e-3 / (ell[2:] * (ell[2:] + 1)) ** 0.9
    ee = 0.1 * tt
    return {"TT": tt, "EE": ee, "BB": bb * tt, "TE": 0.5 * np.sqrt(tt * ee)}


def rel_err(a, b):
    return np.abs(a - b).max() / np.abs(b).max()


@pytest.fixture(scope="module")
def cap():
    return apodised_cap(NSIDE), spectra(LMAX)


@pytest.fixture(scope="module")
def cap_rows(cap):
    """Rows l' = 20 and 40 of all four-spectrum blocks on the cap mask."""
    mask, cls = cap
    return {lp: exact_covariance_row_pol(mask, cls, lp, LMAX) for lp in (20, 40)}


# --------------------------------------------------------------------------- #
# Full sky: every block is diagonal with the cosmic-variance value
# --------------------------------------------------------------------------- #
def test_full_sky_identities_for_every_block():
    """
    With W == 1 (alm = sqrt(4 pi) delta_00, Lw = 0), R^{XZ} = C^{XZ}_l
    delta delta and the Wick contraction gives, per multipole,

        cov(XY, ZW) = (C^{XZ} C^{YW} + C^{XW} C^{YZ}) / (2l+1)

    for all six spectra, with nonzero TE and BB (TB = EB = 0).  Measured:
    diagonal to 2e-15 relative, off-diagonal 1e-31 of the diagonal.
    """
    lmax = 16
    cls = spectra(lmax)
    full_sky = np.array([np.sqrt(4 * np.pi)], dtype=complex)
    sigma = exact_covariance_pol(full_sky, cls, lmax, spectra=ALL_SPECTRA)

    ell = np.arange(lmax + 1)
    zero = np.zeros(lmax + 1)
    c = {**{s: cls.get(s, zero) for s in SPECTRA}}
    c.update({s[::-1]: c[s] for s in SPECTRA})

    assert set(sigma) == {(a, b) for a in ALL_SPECTRA for b in ALL_SPECTRA}
    scale = max(np.abs(v).max() for v in sigma.values())
    for (xy, zw), block in sigma.items():
        x, y = xy
        z, w = zw
        expected = (c[x + z] * c[y + w] + c[x + w] * c[y + z]) / (2 * ell + 1)
        assert block.shape == (lmax + 1, lmax + 1)
        off = block - np.diag(np.diag(block))
        assert np.abs(off).max() < 1e-14 * scale, (xy, zw)
        if np.any(expected):
            np.testing.assert_allclose(
                np.diag(block)[2:], expected[2:], rtol=1e-12, err_msg=f"{xy},{zw}"
            )
        else:
            assert np.abs(np.diag(block)).max() < 1e-14 * scale, (xy, zw)

    # The named identities of the specification, spelled out.
    np.testing.assert_allclose(
        np.diag(sigma[("EE", "EE")])[2:],
        (2 * cls["EE"] ** 2 / (2 * ell + 1))[2:],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        np.diag(sigma[("TE", "TE")])[2:],
        ((cls["TT"] * cls["EE"] + cls["TE"] ** 2) / (2 * ell + 1))[2:],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        np.diag(sigma[("TT", "EE")])[2:],
        (2 * cls["TE"] ** 2 / (2 * ell + 1))[2:],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        np.diag(sigma[("BB", "BB")])[2:],
        (2 * cls["BB"] ** 2 / (2 * ell + 1))[2:],
        rtol=1e-12,
    )


# --------------------------------------------------------------------------- #
# Symmetries and grid rule on the apodised cap
# --------------------------------------------------------------------------- #
def test_row_column_symmetry_between_blocks(cap_rows):
    """
    cov(XY, ZW)_{l l'} == cov(ZW, XY)_{l' l}: element (40) of the l' = 20 row
    of block (XY, ZW) equals element (20) of the l' = 40 row of block
    (ZW, XY).  K is Hermitian on GL, so this holds at round-off: measured
    3e-13 element-wise, 1e-16 of the block scale.
    """
    r20, r40 = cap_rows[20], cap_rows[40]
    specs = ("TT", "TE", "EE", "BB")
    scale = max(np.abs(r20[k]).max() for k in r20)
    for a in specs:
        for b in specs:
            x, y = r20[(a, b)][40], r40[(b, a)][20]
            assert abs(x - y) < 1e-14 * scale, (a, b)
            np.testing.assert_allclose(x, y, rtol=1e-11, err_msg=f"{a},{b}")


def test_m_prime_symmetry(cap):
    """m' >= 0 with weight 2 Re must reproduce the full m' loop (measured 6e-16)."""
    mask, cls = cap
    fast = exact_covariance_row_pol(mask, cls, 12, LMAX, use_symmetry=True)
    full = exact_covariance_row_pol(mask, cls, 12, LMAX, use_symmetry=False)
    for k in fast:
        assert rel_err(fast[k], full[k]) < 1e-13, k


def test_minimal_grid_is_exact_for_spin2(cap):
    """The spin-0 rule Lg = lmax + ceil((Lw-1)/2) also holds for the spin-2
    round trips: the minimal grid and the over-resolved grid lmax + Lw agree
    to 1e-13 on every block (measured 1e-14)."""
    mask, cls = cap
    assert gl_minimal_lmax(LMAX, LW) == 107
    minimal = exact_covariance_row_pol(mask, cls, 12, LMAX)
    large = exact_covariance_row_pol(mask, cls, 12, LMAX, lmax_grid=LMAX + LW)
    for k in minimal:
        assert rel_err(minimal[k], large[k]) < 1e-13, k


# --------------------------------------------------------------------------- #
# Agreement with the TT-only implementation, API behaviour
# --------------------------------------------------------------------------- #
def test_tt_block_matches_exact_covariance_row(cap, cap_rows):
    """
    spectra=("TT",) must equal exact_covariance_row(grid="gl") on the cap
    mask, with nonzero TE in ``cls`` (the TT block depends on C^TT only).
    The TT block of the four-spectrum row is the same number.

    The two are no longer the same arithmetic: the TT path carries a column as
    the pair of real-map alms and multiplies ``C_L`` there, while the polarised
    path still builds the ``(lmax+1, 2 lmax+1)`` full-``M`` array and contracts
    ``cmat`` into it (see ``tests/test_exact_gl_speed.py``).  Those differ by an
    ulp on the intermediates, so the rows agree to 1.4e-16 of the row maximum
    but only to 1.2e-13 on the single element at l = 49, which is 3e-9 of the
    peak and round-off limited.  The bounds below are those measured values.
    """
    mask, cls = cap
    ref = exact_covariance_row(mask, cls["TT"], 20, LMAX, grid="gl")
    tt_only = exact_covariance_row_pol(mask, cls, 20, LMAX, spectra=("TT",))
    assert set(tt_only) == {("TT", "TT")}
    for got in (tt_only[("TT", "TT")], cap_rows[20][("TT", "TT")]):
        assert np.abs(got - ref).max() < 1e-15 * np.abs(ref).max()
        np.testing.assert_allclose(got, ref, rtol=1e-12)


def test_matrix_builder_and_key_handling(cap, cap_rows):
    """exact_covariance_pol fills the requested columns of every ordered
    block, (s2, s1) is the transpose, "ET" is canonicalised to "TE", and
    spectra absent from ``cls`` are zero."""
    mask, cls = cap
    sigma = exact_covariance_pol(mask, cls, LMAX, rows=(20,))
    for k, block in sigma.items():
        np.testing.assert_allclose(block[:, 20], cap_rows[20][k], rtol=1e-14)
        assert np.all(block[:, 19] == 0)
    both = exact_covariance_pol(mask, cls, LMAX, rows=(20, 40), spectra=("TT", "EE"))
    tt_ee = both[("TT", "EE")]
    ee_tt = both[("EE", "TT")]
    assert abs(tt_ee[20, 40] - ee_tt[40, 20]) < 1e-14 * np.abs(tt_ee).max()

    # Same spectra with the reversed key; BB must stay, since B -> E leakage
    # feeds C^BB into R^EE and hence into cov(TE, TE).
    cls_reversed = {"TT": cls["TT"], "EE": cls["EE"], "BB": cls["BB"], "ET": cls["TE"]}
    row = exact_covariance_row_pol(
        mask, cls_reversed, 20, LMAX, spectra=("TT", "ET", "EE")
    )
    assert ("TE", "TE") in row and ("ET", "ET") not in row
    np.testing.assert_allclose(
        row[("TE", "TE")], cap_rows[20][("TE", "TE")], rtol=1e-14
    )

    no_te = exact_covariance_row_pol(
        mask, {"TT": cls["TT"], "EE": cls["EE"]}, 20, LMAX, spectra=("TT", "EE")
    )
    assert np.all(no_te[("TT", "EE")] == 0)
    assert no_te[("EE", "EE")].max() > 0

    with pytest.raises(ValueError):
        exact_covariance_row_pol(mask, cls, 20, LMAX, spectra=("TT", "XX"))
    with pytest.raises(ValueError):
        exact_covariance_row_pol(mask, cls, 20, LMAX, spectra=())
    with pytest.raises(ValueError):
        exact_covariance_row_pol(mask, cls, 20, LMAX, grid="bogus")
    with pytest.raises(TypeError):
        exact_covariance_row_pol(mask, cls, 20, LMAX, iter=3)


def test_low_multipole_rows_have_no_polarisation():
    """E and B have no l < 2 modes: the E/B columns of rows l' = 0, 1 are zero
    and the TT block is still the TT answer."""
    lmax = 8
    cls = spectra(lmax)
    full_sky = np.array([np.sqrt(4 * np.pi)], dtype=complex)
    for lp in (0, 1):
        row = exact_covariance_row_pol(full_sky, cls, lp, lmax)
        assert np.all(row[("EE", "EE")] == 0)
        assert np.all(row[("TE", "TE")] == 0)
        assert np.all(row[("BB", "BB")] == 0)


def test_healpix_path_is_tt_only():
    """grid="healpix" raises NotImplementedError for any polarised spectrum
    and delegates to exact_covariance_row / exact_covariance for TT."""
    nside, lmax = 16, 12
    mask = apodised_cap(nside)
    cls = spectra(lmax)
    for spec in (("TT", "TE"), ("EE",), ("BB", "BB")):
        with pytest.raises(NotImplementedError, match="TT-only"):
            exact_covariance_row_pol(
                mask, cls, 6, lmax, spectra=spec, grid="healpix", nside=nside
            )
        with pytest.raises(NotImplementedError, match="TT-only"):
            exact_covariance_pol(
                mask, cls, lmax, spectra=spec, grid="healpix", nside=nside
            )
    row = exact_covariance_row_pol(
        mask, cls, 6, lmax, spectra=("TT",), grid="healpix", nside=nside, iter=1
    )
    ref = exact_covariance_row(mask, cls["TT"], 6, lmax, nside, iter=1)
    np.testing.assert_allclose(row[("TT", "TT")], ref, rtol=1e-14)
    matrix = exact_covariance_pol(
        mask, cls, lmax, spectra=("TT",), grid="healpix", nside=nside, rows=(6,)
    )
    ref_matrix = exact_covariance(mask, cls["TT"], lmax, nside, rows=(6,))
    np.testing.assert_allclose(matrix[("TT", "TT")], ref_matrix, rtol=1e-14)


# --------------------------------------------------------------------------- #
# Monte Carlo
# --------------------------------------------------------------------------- #
def monte_carlo_pol_covariance(mask, cls, nside, lmax, n_sims, seed=3):
    """Sample covariance of the (TT, EE, BB, TE) pseudo-spectra of masked
    healpy.synfast(pol=True) skies, as ``cov[i, l, j, l']`` over MC_ORDER."""
    np.random.seed(seed)
    spec = np.empty((4, lmax + 1, n_sims))
    for i in range(n_sims):
        t, q, u = healpy.synfast(
            [cls[k] for k in MC_ORDER], nside, lmax=lmax, new=True, pol=True
        )
        out = healpy.anafast([t * mask, q * mask, u * mask], lmax=lmax, pol=True)
        spec[:, :, i] = out[:4]
    cov = np.cov(spec.reshape(4 * (lmax + 1), n_sims))
    return cov.reshape(4, lmax + 1, 4, lmax + 1)


@pytest.mark.slow
def test_exact_pol_matches_simulations():
    """
    Rows l' = 20 and 40 of cov(EE,EE), cov(TE,TE), cov(TT,EE), cov(TE,EE),
    cov(TE,TT), cov(BB,BB) and cov(TT,TT) against the sample covariance of
    N_SIMS masked polarised skies (nside=32, lmax=60, C^BB = 0, so the BB
    block is pure E -> B leakage through the mask).  Two statistics per
    block, both in units of the Monte Carlo standard deviation of the sample
    covariance (paper Eq. 18, Isserlis form, evaluated with the analytic
    values):

    * element-wise: the diagonal and the first MC_NOFF off-diagonals of each
      row, sigma^2 = (Sigma_ll'^2 + Sigma_ll Sigma_l'l') / N;
    * band mean of the ratio MC / analytic over the diagonal l = 16..44, with
      the correlated sigma  Cov(S_ll, S_l'l') = (Sigma^aa_ll' Sigma^bb_ll' +
      Sigma^ab_ll' Sigma^ab_l'l) / N summed over the band.  This resolves a
      ~2.5% normalisation error at 4 sigma; an E/B sign mismatch would give
      z ~ 100 on cov(TE,EE) and cov(TE,TT), which are odd in E.

    Measured (max |z| element-wise / z of the band mean), N = 10000 seed 3
    (this test): EE,EE 2.8/+2.6, TE,TE 1.5/-1.4, TT,EE 1.2/0.0, TE,EE
    1.2/+0.9, TE,TT 2.0/-1.9, BB,BB 2.5/-1.1, TT,TT 1.0/-1.9; seed 11: 1.6-2.6
    / -0.5..+1.4; N = 5000 seeds 3 and 7: 0.4-3.2 / -2.3..+2.4.  The band
    mean flips sign between seeds (EE,EE: +2.6, +1.4, +2.4, -2.3), so there
    is no bias, but its scatter is ~1.5-2x the Gaussian sigma: the
    pseudo-spectra are chi^2-like with few effective degrees of freedom on
    this 40 degree cap, and the elements are correlated.  The 4 sigma bound
    is one sigma above the largest observed value.  MC takes ~40 s.
    """
    mask = apodised_cap(NSIDE)
    cls = spectra(LMAX, bb=0.0)
    t0 = time.time()
    mc = monte_carlo_pol_covariance(mask, cls, NSIDE, LMAX, N_SIMS)
    t_mc = time.time() - t0

    band = list(range(min(MC_ROWS) - MC_NOFF, max(MC_ROWS) + MC_NOFF + 1))
    sigma = exact_covariance_pol(mask, cls, LMAX, spectra=MC_ORDER, rows=band)

    report = []
    for a, b in MC_BLOCKS:
        ia, ib = MC_ORDER.index(a), MC_ORDER.index(b)
        s_ab, s_aa, s_bb = sigma[(a, b)], sigma[(a, a)], sigma[(b, b)]

        z = []
        for lp in MC_ROWS:
            for ell in range(lp - MC_NOFF, lp + MC_NOFF + 1):
                an = s_ab[ell, lp]
                var = (an**2 + s_aa[ell, ell] * s_bb[lp, lp]) / N_SIMS
                z.append((mc[ia, ell, ib, lp] - an) / np.sqrt(var))
        z_max = np.abs(z).max()

        diag = np.array([s_ab[ell, ell] for ell in band])
        assert np.all(diag > 0) or (a != b)
        ratio = np.array([mc[ia, ell, ib, ell] for ell in band]) / diag
        cov_sum = 0.0
        for i, ell in enumerate(band):
            for j, lq in enumerate(band):
                c = (
                    s_aa[ell, lq] * s_bb[ell, lq] + s_ab[ell, lq] * s_ab[lq, ell]
                ) / N_SIMS
                cov_sum += c / (diag[i] * diag[j])
        z_mean = (ratio.mean() - 1.0) / (np.sqrt(cov_sum) / len(band))

        report.append(f"{a},{b}: max|z| = {z_max:.2f}, band mean z = {z_mean:+.2f}")
        assert (
            z_max < MC_SIGMA_BOUND
        ), f"{a},{b}: max |z| = {z_max:.2f} (MC {t_mc:.0f} s)"
        assert abs(z_mean) < MC_SIGMA_BOUND, (
            f"{a},{b}: band mean MC/analytic = {ratio.mean():.4f}, "
            f"z = {z_mean:+.2f} (MC {t_mc:.0f} s)"
        )
    print("; ".join(report), f"(MC {t_mc:.0f} s)")
