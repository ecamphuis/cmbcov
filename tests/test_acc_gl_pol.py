r"""
Polarisation on the Gauss-Legendre ACC precompute, and the ET kernel.

What is pinned here, all measured on ``tests/data/baseline_mask.fits``
band-limited at ``LW = 10`` (as ``tests/test_acc_gl.py``) unless stated:

- ``spin_weighted_integrals_gl(spin=2)`` returns ``K_2 e^E``: on the full
  sky the E component is the unit vector itself and the B component is zero
  (1e-15); against the HEALPix path's ``compute_spin_weighted_integrals``
  (ducc0 HEALPix spin-2 ``map2alm`` at healpy's default ``iter=3``, an
  independent route) it agrees to 5.5e-4 at worst at ``nside=16`` (6.1e-3
  at ``iter=0``).
- The ET kernel: ``Theta^{ET}(m, m') = conj(Theta^{TE}(m', m))`` at
  ``l == l'``, so ``ETxET == TExTE``, ``TTxET == TTxTE``, ``EExET == EExTE``
  there (1e-16), while at ``l != l'`` the ET kernels differ from the TE ones
  by 3e-3 to 1.6e-2 of the kernel maximum.
- Every polarised Wick term of the exact covariance
  (:func:`exact_covariance_pol`) is reproduced by the corresponding GL
  kernel, ``(1/n) C1 . Theta^{s1xs2} . C2`` with ``n = (2l+1)(2l'+1)``, to
  1e-13 at ``(16, 16..19)``; for the ``(C^TE)^2`` term of Cov(TE, TE) and
  for Cov(EE_l, TE_l') this requires the ET kernel: substituting the TE
  kernel for it would be off by 3.3e-3 (16, 17) to 8.4e-3 (16, 19).
- The loader refuses a cache without ET files.
- GL and HEALPix polarised kernels agree to 2e-3 after Eq. 23
  normalisation (measured 1.1e-4 to 8.1e-4, the same level as TT).
- Through ``ACCStrategy.compute_covariance_term``, TT with the Xi of the same
  band-limited mask reproduces the exact covariance to 1e-9 where the
  central kernel is used unshifted (``min(l, l') = centralell``) and to 5e-3
  (measured 0.9969..1.0048) on the other entries of the band, which use
  the Eq. 33 translation.

The polarised *blocks* of the strategy are deliberately not asserted: with
the package's ``norm_Xi`` they are 3-16% high on this mask, entirely from
the normalisation (the kernels are exact, see above).
"""

import os
import tempfile
import warnings

import healpy as hp
import numpy as np
import pytest
from conftest import compute_cross_spec_cplxmapalm

from cmbcov.approximations import StrategyFactory
from cmbcov.approximations.acc import (
    _CENTRAL_FIELD,
    _PRIME_FIELD,
    COUPLING_SPECTRA,
    _healpix_full_m,
    precompute_acc_kernels,
)
from cmbcov.covariance import Cov, CovarianceConfig, CovarianceMethod
from cmbcov.exact import exact_covariance_pol
from cmbcov.grid import (
    cross_spectrum_full_m,
    full_from_pair,
    gl_analysis,
    gl_synthesis,
    spin_weighted_integrals_gl,
)
from cmbcov.kernels.coupling import KERNEL_TT, coupling_kernels
from cmbcov.keys import CovKey
from cmbcov.mask import MaskWlm
from cmbcov.sht import ducc0_map2alm

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
NSIDE = 16
ELL = 16
LMAX = 48  # Cov.lmax: multipoles 0..47
DMAX = 4
LW = 10

#: ACC reads the spectra to lmax_int = lmax + max(0, S - 1 - centralell),
#: S = 2 NSIDE the kernel size (acc_window_pad).
LMAX_INT = LMAX + 2 * NSIDE - 1 - ELL
ROWS = list(range(ELL, ELL + DMAX))
KSIZE = 2 * NSIDE  # kernel size: L = 0..31
FREQ = "090GHz"


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


def _rel(a, b):
    return np.abs(a - b).max() / np.abs(b).max()


# --------------------------------------------------------------------------- #
# (A) spin-2 integrals on the GL grid
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ell, m", [(2, 0), (5, 3), (8, -4), (12, 12)])
def test_spin2_integrals_full_sky_are_the_unit_vector(ell, m):
    """W = 1: u^E = e^E_{ell m}, u^B = 0 (measured 9e-16 and 1.2e-15)."""
    lmax_out = 20
    full_sky = np.array([np.sqrt(4 * np.pi)], dtype=complex)
    eb = spin_weighted_integrals_gl(full_sky, 0, ell, m, lmax_out, spin=2)
    assert eb.shape == (2, lmax_out + 1, 2 * lmax_out + 1)
    e = np.zeros((lmax_out + 1, 2 * lmax_out + 1), dtype=complex)
    e[ell, lmax_out + m] = 1.0
    assert np.abs(eb[0] - e).max() < 1e-14
    assert np.abs(eb[1]).max() < 1e-14


def test_spin2_integrals_vanish_below_ell_2():
    fs = np.array([np.sqrt(4 * np.pi)], dtype=complex)
    assert not np.any(spin_weighted_integrals_gl(fs, 0, 1, 0, 8, spin=2))
    with pytest.raises(ValueError):
        spin_weighted_integrals_gl(fs, 0, 4, 0, 8, spin=1)


@pytest.fixture(scope="module")
def healpix_integral_setup():
    cov = _cov(tempfile.mkdtemp())
    wn, _ = cov.wlm.degrade_mask(NSIDE)
    lw = 3 * NSIDE - 1
    # The alm of the very pixel mask the HEALPix path multiplies by, so the
    # comparison isolates the HEALPix quadrature from the mask difference.
    mask_alm = ducc0_map2alm(wn, lmax=lw, pol=False, iter=10)
    return cov, wn, mask_alm, lw


@pytest.mark.parametrize(
    "ell, m", [(5, 0), (5, 3), (12, 0), (12, -7), (16, 0), (16, 16), (16, -9)]
)
def test_spin2_integrals_match_healpix_path(healpix_integral_setup, ell, m):
    """
    GL (E, B) vs the HEALPix ``compute_spin_weighted_integrals`` E and B at
    ``nside=16`` (``lmax_out = 31``), both from the same degraded pixel mask.
    The HEALPix E/B carry the ``sqrt(2)`` of ``cplx_spin_weighted_ylm``.
    Measured ``max|diff| / max|E_GL|`` with the HEALPix ``map2alm`` at
    ``iter=3``: E 9.3e-5 .. 5.5e-4, B 7.7e-5 .. 5.1e-4; T is 6.1e-5 .. 2.0e-4
    at ``m = 0`` and 2.0e-4 .. 1.0e-3 otherwise.  (At ``iter=0``, E and B
    reached 6.1e-3 and T 4.9e-2 at ``m = 0``.)  B is normalised by the E scale because the azimuthally
    symmetric mask has no ``m = 0`` E->B leakage (|B|/|E| ~ 1e-8 on GL).
    """
    cov, wn, mask_alm, lw = healpix_integral_setup
    lmax_out = 2 * NSIDE - 1
    cma = cov.wlm.compute_spin_weighted_integrals(
        ell, m, mask_for_ell=wn, target_nside=NSIDE
    )
    hpx = full_from_pair(cma[0], cma[1], lmax_out)
    gl_t = spin_weighted_integrals_gl(mask_alm, lw, ell, m, lmax_out)
    gl_eb = spin_weighted_integrals_gl(mask_alm, lw, ell, m, lmax_out, spin=2)
    scale = np.abs(gl_eb[0]).max()
    err_e = np.abs(hpx[1] / np.sqrt(2) - gl_eb[0]).max() / scale
    err_b = np.abs(hpx[2] / np.sqrt(2) - gl_eb[1]).max() / scale
    err_t = _rel(hpx[0], gl_t)
    assert err_e < 1.5e-3, f"E: {err_e:.3e}"
    assert err_b < 1.5e-3, f"B: {err_b:.3e}"
    assert err_t < 2e-3, f"T: {err_t:.3e}"


# --------------------------------------------------------------------------- #
# (B) the precompute: ET kernels, loader, GL vs HEALPix
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def gl_precompute():
    """GL kernels at (16, 16..19), LW=10, saved to disk; the strategy that
    loads them; the mask alm they were built from; the exact reference."""
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
    exact_no_te = exact_covariance_pol(
        mask_alm,
        {"TT": cls["TT"], "EE": cls["EE"]},
        LMAX - 1,
        spectra=("TE",),
        rows=ROWS,
        grid="gl",
    )
    return cov, strategy, mask_alm, cls, exact, exact_no_te


def test_loader_returns_distinct_et_kernels(gl_precompute):
    _, strategy, _, _, _, _ = gl_precompute
    same = strategy.get_covariance_coupling(ELL, ELL)
    assert set(same) == {
        (a, b) for a in ("TT", "DD", "TD", "DT") for b in ("TT", "DD", "TD", "DT")
    }
    # l == l': every DT kernel equals its TD counterpart (old ET / TE)
    for a in ("TT", "DD", "TD"):
        assert _rel(same[(a, "DT")], same[(a, "TD")]) < 1e-12
        assert _rel(same[("DT", a)], same[("TD", a)]) < 1e-12
    assert _rel(same[("DT", "DT")], same[("TD", "TD")]) < 1e-12
    # l != l': they differ, and DTxTD is the transpose of TDxDT
    for ellp, floor in [(ELL + 1, 2e-3), (ELL + 3, 5e-3)]:
        k = strategy.get_covariance_coupling(ELL, ellp)
        assert _rel(k[("TD", "DT")], k[("TD", "TD")]) > floor
        assert _rel(k[("TT", "DT")], k[("TT", "TD")]) > floor
        np.testing.assert_allclose(k[("DT", "TD")], k[("TD", "DT")].T, rtol=0, atol=0)


def test_save_path_does_not_alias_et_to_te(gl_precompute):
    _, strategy, _, _, _, _ = gl_precompute
    te = strategy.get_covariance_coupling_save_path(("TD", "TD"), 16, 17)
    et = strategy.get_covariance_coupling_save_path(("TD", "DT"), 16, 17)
    assert te != et and et.endswith("TDxDT_16x17.npy")
    assert strategy.get_covariance_coupling_save_path("DTxTT", 16, 17).endswith(
        "DTxTT_16x17.npy"
    )
    with pytest.raises(ValueError):
        strategy.get_covariance_coupling_save_path(("TD", "TB"), 16, 17)


def test_loader_refuses_a_cache_without_et_files():
    """A pre-fix cache (TD files only) must raise, not silently read TD."""
    cov = _cov(tempfile.mkdtemp(), dmax=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        precompute_acc_kernels(
            cov.wlm, cov.save_dir, centralell=ELL, dmax=1, nside=NSIDE, grid="gl", lw=LW
        )
        strategy = StrategyFactory.create_strategy(cov)
    assert len(strategy.get_covariance_coupling(ELL, ELL)) == 16
    for a in COUPLING_SPECTRA:
        for b in COUPLING_SPECTRA:
            if "DT" in (a, b):
                os.remove(strategy.get_covariance_coupling_save_path((a, b), ELL, ELL))
    with pytest.raises(OSError, match="predate the ET fix"):
        strategy.get_covariance_coupling(ELL, ELL)


def test_polarised_kernels_reproduce_exact_wick_terms(gl_precompute):
    r"""
    Each Wick term of the exact polarised covariance, isolated by the choice
    of spectra, equals ``(1/n) C1 . Theta^{s1xs2}_{16,l'} . C2`` with the
    raw GL kernel (no Xi, no Eq. 23 normalisation), ``n = (2l+1)(2l'+1)``,
    for the spectra truncated to the kernel's ``L < 32`` (the kernel has no
    support beyond ``l' + LW = 29``).  Measured 1e-15 .. 1e-14.  The two
    terms needing ET are also evaluated with the TE kernel substituted for
    it: off by 3.3e-3 / 8.4e-3 at ``(16, 17)`` / ``(16, 19)``.
    """
    _, strategy, _, cls, exact, exact_no_te = gl_precompute
    tt, ee, te = (cls[s][:KSIZE] for s in ("TT", "EE", "TE"))
    for ellp in ROWS:
        n = (2 * ELL + 1) * (2 * ellp + 1)
        k = strategy.get_covariance_coupling(ELL, ellp)
        checks = {
            "Cov(EE,EE) = 2/n EE.DDxDD.EE": (
                2 / n * ee @ k[("DD", "DD")] @ ee,
                exact[("EE", "EE")][ELL, ellp],
            ),
            "Cov(TE,TE)|TE=0 = 1/n TT.TTxDD.EE": (
                1 / n * tt @ k[("TT", "DD")] @ ee,
                exact_no_te[("TE", "TE")][ELL, ellp],
            ),
            "Cov(TE,TE) TE^2 term = 1/n TE.TDxDT.TE": (
                1 / n * te @ k[("TD", "DT")] @ te,
                exact[("TE", "TE")][ELL, ellp] - exact_no_te[("TE", "TE")][ELL, ellp],
            ),
            "Cov(TT,EE) = 2/n TE.TDxTD.TE": (
                2 / n * te @ k[("TD", "TD")] @ te,
                exact[("TT", "EE")][ELL, ellp],
            ),
            "Cov(TT,TE) = 2/n TT.TTxTD.TE": (
                2 / n * tt @ k[("TT", "TD")] @ te,
                exact[("TT", "TE")][ELL, ellp],
            ),
            "Cov(EE_l,TE_l') = 2/n EE.DDxDT.TE": (
                2 / n * ee @ k[("DD", "DT")] @ te,
                exact[("EE", "TE")][ELL, ellp],
            ),
            "Cov(TE_l,EE_l') = 2/n TE.TDxDD.EE": (
                2 / n * te @ k[("TD", "DD")] @ ee,
                exact[("TE", "EE")][ELL, ellp],
            ),
        }
        for label, (acc, ref) in checks.items():
            assert (
                abs(acc / ref - 1) < 1e-12
            ), f"(16,{ellp}) {label}: {acc / ref - 1:.3e}"
        if ellp != ELL:
            aliased = 1 / n * te @ k[("TD", "TD")] @ te
            ref = exact[("TE", "TE")][ELL, ellp] - exact_no_te[("TE", "TE")][ELL, ellp]
            assert (
                abs(aliased / ref - 1) > 2e-3
            ), "TE kernel should not pass for the TE^2 term"
            aliased = 2 / n * ee @ k[("DD", "TD")] @ te
            assert abs(aliased / exact[("EE", "TE")][ELL, ellp] - 1) > 2e-3


def test_healpix_et_cross_spectrum_is_the_swapped_te_entry(tmp_path):
    r"""
    On the HEALPix branch's complex full-``M`` path (:func:`_healpix_full_m`
    and the ``_CENTRAL_FIELD`` / ``_PRIME_FIELD`` channel map),

        ``Theta^{ET}(a, b) = conj(Theta^{TE}(b, a))`` ,

    i.e. ET is the TE entry with the two integral sets -- ``(l, m)`` and
    ``(l', m')`` -- swapped and conjugated.  Its real part must equal the
    independent healpy brute force ``(2L+1) compute_cross_spec_cplxmapalm(b,
    a)[TE]``, and ET is not TE at ``(l, l') = (16, 17)``.

    The mask is the test mask rotated off the pole, so ``Im Theta`` is a
    sizeable fraction of ``Re Theta`` and the conjugation is actually tested;
    through the shipped precompute on the same mask, ``ETxET == TExTE`` and
    ``TTxET == TTxTE`` at ``l == l'``.
    """
    rotated = hp.Rotator(rot=(37.0, 21.0, 11.0)).rotate_map_pixel(
        hp.read_map(os.path.join(DATA, "baseline_mask.fits"))
    )
    hp.write_map(str(tmp_path / "rotated.fits"), rotated, overwrite=True)
    wlm = MaskWlm("rotated.fits", load_path=str(tmp_path))
    wn, _ = wlm.degrade_mask(NSIDE)
    a = wlm.compute_spin_weighted_integrals(16, 3, mask_for_ell=wn, target_nside=NSIDE)
    b = wlm.compute_spin_weighted_integrals(17, -2, mask_for_ell=wn, target_nside=NSIDE)
    lmax = 2 * NSIDE
    full_a, full_b = _healpix_full_m(a, lmax), _healpix_full_m(b, lmax)

    def theta(spec, x, y):
        k = COUPLING_SPECTRA.index(spec)
        return cross_spectrum_full_m(x[_CENTRAL_FIELD[k]], y[_PRIME_FIELD[k]])

    got = theta("DT", full_a, full_b)
    swapped = theta("TD", full_b, full_a)
    direct = theta("TD", full_a, full_b)
    scale = np.abs(got).max()
    assert np.abs(got.imag).sum() > 0.1 * np.abs(got.real).sum()
    np.testing.assert_allclose(got, np.conj(swapped), rtol=1e-12, atol=1e-16 * scale)
    brute = (2 * np.arange(lmax) + 1) * compute_cross_spec_cplxmapalm(b, a, nspec=4)[3]
    np.testing.assert_allclose(got.real, brute, rtol=1e-12, atol=1e-14 * scale)
    assert _rel(got, direct) > 1e-2  # DT is not TD for (l, l') = (16, 17)

    kernels = precompute_acc_kernels(
        wlm,
        None,
        centralell=ELL,
        ellprange=[ELL],
        nside=NSIDE,
        dryrun=True,
        grid="healpix",
        spectra=("TT", "TD", "DT"),
    )[ELL]
    for et_key, te_key in (("DTxDT", "TDxTD"), ("TTxDT", "TTxTD")):
        ref = kernels[te_key]
        np.testing.assert_allclose(
            kernels[et_key], ref, rtol=1e-12, atol=1e-14 * np.abs(ref).max()
        )


def test_gl_and_healpix_polarised_kernels_agree_after_normalisation():
    """
    Normalised kernels (Eq. 23) from the two backends at ``nside=16``, GL
    with the ``lw = 3 nside - 1 = 47`` default: measured 1.1e-4 .. 8.1e-4 for
    every pair (TT 4.9e-4 .. 7.4e-4) with the HEALPix ``map2alm`` at
    ``iter=3``.  The raw sums differ by ``sqrt(2)^k`` for ``k`` E/B legs (the
    HEALPix E/B convention), times 0.9998 .. 0.9999.  At ``iter=0``, the
    normalised kernels agreed to 9e-3 and the sums carried a 0.969 .. 0.993
    deficit.
    """
    cov = _cov(tempfile.mkdtemp(), dmax=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hpx = precompute_acc_kernels(
            cov.wlm,
            None,
            centralell=ELL,
            dmax=2,
            nside=NSIDE,
            dryrun=True,
            grid="healpix",
        )
        gl = precompute_acc_kernels(
            cov.wlm, None, centralell=ELL, dmax=2, nside=NSIDE, dryrun=True, grid="gl"
        )
    legs = {"TT": 0, "DD": 2, "LL": 2, "TD": 1, "DT": 1}
    for ellp in (ELL, ELL + 1):
        for a in COUPLING_SPECTRA:
            for b in COUPLING_SPECTRA:
                key = f"{a}x{b}"
                g, h = gl[ellp][key], hpx[ellp][key]
                assert _rel(h / h.sum(), g / g.sum()) < 2e-3, f"{ellp} {key}"
                sum_ratio = h.sum() / g.sum() / np.sqrt(2) ** (legs[a] + legs[b])
                assert 0.999 < sum_ratio < 1.001, f"{ellp} {key}: {sum_ratio:.4f}"


# --------------------------------------------------------------------------- #
# (D) TT through the strategy vs exact on the GL grid
# --------------------------------------------------------------------------- #
def test_assembled_tt_matches_exact_with_band_limited_xi(gl_precompute):
    """
    ``compute_covariance_term`` for TTxTT, rescaled entrywise from the
    strategy's ``norm_Xi`` (pixel-mask ``W^2``) to the Xi of the LW=10 mask
    the kernels were built from (the term is linear in ``Xi[l, l']``), vs
    :func:`exact_covariance_pol` on that mask.  Measured: 1 - O(1e-12)
    wherever ``min(l, l') = centralell`` (unshifted kernel), 0.9969 ..
    1.0048 on the rest of the band ``|l - l'| <= 3`` of rows 16..19 (the
    Eq. 33 translation).  With the strategy's own ``norm_Xi`` the same
    entries are 0.984 .. 0.993, entirely the ``Xi`` mismatch.
    """
    cov, strategy, mask_alm, cls, exact, _ = gl_precompute
    key = CovKey(("T", "T", "T", "T"), (FREQ,) * 4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sigma = strategy.compute_covariance_term(key, {FREQ + FREQ: _spectra(LMAX_INT)})

    grid_lmax = 2 * LW + 5
    w2 = hp.alm2cl(
        gl_analysis(gl_synthesis(mask_alm, LW, grid_lmax) ** 2, 2 * LW, grid_lmax)
    )
    xi_band = (
        coupling_kernels(w2, LMAX - 1)[KERNEL_TT] / (2 * np.arange(LMAX) + 1)[None, :]
    )
    xi_strategy = cov.norm_Xi[("TT", "TT")]
    ref = exact[("TT", "TT")]

    for ellp in ROWS:
        for ell in range(ellp - DMAX + 1, ellp + DMAX):
            r_band = (
                sigma[ell, ellp]
                * xi_band[ell, ellp]
                / xi_strategy[ell, ellp]
                / ref[ell, ellp]
            )
            r_strategy = sigma[ell, ellp] / ref[ell, ellp]
            if min(ell, ellp) == ELL:
                assert abs(r_band - 1) < 1e-9, (ell, ellp, r_band)
            else:
                assert abs(r_band - 1) < 5e-3, (ell, ellp, r_band)
            assert 0.98 < r_strategy < 0.995, (ell, ellp, r_strategy)
