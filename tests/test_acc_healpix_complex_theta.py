r"""
The HEALPix ACC precompute must keep the per-``(m, m')`` cross-spectra complex.

``_CouplingPrecompute._compute_ellp_coupling`` forms, for each spectrum ``s``,

.. math::

    X^s_{mm'}(L) = \sum_M u^a_{\ell m, LM}\, \overline{u^b_{\ell' m', LM}}, \qquad
    K^{s s}(L_1, L_2) = \mathrm{Re} \sum_{mm'} X^s_{mm'}(L_1)\,
        \overline{X^s_{mm'}(L_2)}

(paper Eq. 20).  The GL branch always did exactly that.  A HEALPix branch
that instead built ``X`` from ``alm2cl`` of the alms of the real and
imaginary parts of the complex integral map would keep ``Re X`` only, so
its kernel would be ``sum Re X(L1) Re X(L2)``: the ``Im X(L1) Im X(L2)``
term would be dropped.  For a mask symmetric under ``phi -> -phi`` (the test
mask is a polar cap) ``Im X`` vanishes; for a generic mask it does not, so a
``Re``-only kernel would be biased.  Both grids share one complex
contraction (the HEALPix integrals expanded to full-``M`` with
``full_from_pair``); the two checks below pin that.

What is compared
----------------
The masks are the test mask band-limited at ``LW = 10``
(``hp.map2alm(..., iter=10)``), as-is ("azimuthal") and rotated by
``hp.Rotator(rot=(37, 21, 11)).rotate_alm`` ("rotated"), each synthesised
with ``hp.alm2map`` straight onto an ``NSIDE = 16`` pixel map and written to
a FITS file.  ``NSIDE`` is the working resolution, so
``MaskWlm.degrade_mask`` returns the map unchanged: the HEALPix branch sees
the samples of exactly the band-limited function whose alm the GL branch
recovers from the same map (``map2alm(iter=10)`` at ``lmax = LW``; measured
alm error 3e-15).  Synthesising at nside 32 instead would not do: the
band-limited map dips to -0.012, ``degrade_mask`` clips it to ``[0, 1]``,
and the two branches then see different masks (measured 1e-4 to 8e-4 on
the ACC term below, 1-6% on kernel entries, all from the clip).

The observable is the ACC term (Eq. 23 then Eq. 25 without its ``Xi``
prefactor) ``c @ Kbar @ c``, ``Kbar = K / K.sum()``, with the toy spectrum
of ``tests/test_acc_gl.py``, for TTxTT, EExEE and TExTE at
``(ell, ellp) = (16, 16), (16, 17), (16, 19)``.  The unit-sum normalisation
removes the HEALPix branch's ``sqrt(2)`` on the E/B integrals (see
``precompute_acc_kernels``) and any overall sign of E.

1. **HEALPix vs GL**, rotated mask: relative difference of the ACC term.
2. **Orientation invariance on HEALPix**: rotated vs as-is.  The full Eq. 20
   kernel is invariant under a rotation of the mask (``X -> D^l X D^{l'}``
   with unitary Wigner-D blocks, and the kernel is a trace); the ``Re``-only
   kernel is not.

"Fixed HEALPix" is the complex-``X`` reference built here, independently of
the package's contraction, from the branch's own integrals:
``full_from_pair(cma[0], cma[1])`` for T, E and B, then an ``einsum`` of the
formula above.  The shipped HEALPix kernels must equal its full contraction
to rounding (rtol 1e-12) on the rotated mask (measured relative difference
of the ACC term, max over TT/EE/TE, is 3.3e-7 to 4.7e-7 against GL and
1.7e-6 to 3.4e-6 between the rotated and as-is masks; per-spectrum values
are in the constants' comments).  ``|Im X| / |Re X|`` on the rotated mask
is 0.51-0.69 over ``ellp = 16, 17, 19``, so the reference's ``Re``-only
contraction misses these tolerances by orders of magnitude (at least
2000x ``TOL_VS_GL`` and 400x ``TOL_INVARIANCE``), which is why the full
complex contraction is required.  The golden of
``tests/test_acc_coupling.py`` is not affected by this: on the
(unrotated, nside-32) test mask ``|Im X| / |Re X| <= 4.4e-16`` for all five
spectra at (16, 16) and (16, 17), and the ``Im Im`` term is at most 7e-33
of each kernel's maximum (9e-30 of any entry above 1e-6 of the maximum),
against the golden's ``rtol = 1e-10``.
"""

import os

import healpy as hp
import numpy as np
import pytest

from cmbcov.approximations.acc import (
    _CENTRAL_FIELD,
    _PRIME_FIELD,
    COUPLING_SPECTRA,
    precompute_acc_kernels,
)
from cmbcov.grid import full_from_pair
from cmbcov.mask import MaskWlm

# degrade_mask warns that nside 16 >= the map's nside and returns the map
# as-is; that is exactly what this suite wants (see module docstring).
pytestmark = pytest.mark.filterwarnings("ignore:Target nside")

DATA = os.path.join(os.path.dirname(__file__), "data")

NSIDE = 16
ELL = 16
ELLPS = [ELL, ELL + 1, ELL + 3]
LMAX_OUT = 2 * NSIDE - 1  # kernels are (2 NSIDE) x (2 NSIDE)
LW = 10  # mask band-limit, as in tests/test_acc_gl.py
ROTATION = (37.0, 21.0, 11.0)
SPECTRA = ("TT", "DD", "TD")

#: Check 1, fixed HEALPix vs GL on the rotated mask.  Measured max 4.7e-7
#: (TT 3.3e-7/3.3e-7/4.7e-7, EE 2.0e-7/2.2e-7/3.5e-7, TE 2.6e-7/2.6e-7/3.7e-7
#: at ellp 16/17/19); a Re-only HEALPix contraction misses this by 4.1e-3 to
#: 6.8e-3.
TOL_VS_GL = 2e-6

#: Check 2, HEALPix rotated vs as-is.  Measured max 3.4e-6 for fixed HEALPix
#: (TT 1.65e-6/1.75e-6/3.43e-6, EE 7.6e-7/8.8e-7/2.1e-6,
#: TE 1.1e-6/9.7e-7/1.8e-6);
#: the shipped HEALPix vs GL gap on the as-is mask, where ``Im X`` = 0 and
#: the gap is pure transform error, is of the same size (max 3.0e-6).
#: A Re-only HEALPix contraction misses this, rotated vs as-is, by 4.1e-3 to
#: 6.8e-3.
TOL_INVARIANCE = 1e-5


def _toy_cl() -> np.ndarray:
    ell = np.arange(LMAX_OUT + 1)
    cl = np.zeros(LMAX_OUT + 1)
    cl[2:] = 1e-3 / (ell[2:] * (ell[2:] + 1)) ** 0.9
    return cl


def _acc_term(kernel: np.ndarray) -> float:
    """``c @ Kbar @ c`` with the Eq. 23 unit-sum kernel."""
    cl = _toy_cl()
    return float(cl @ (kernel / kernel.sum()) @ cl)


def _reference_healpix_kernels(wlm: MaskWlm) -> dict:
    r"""
    Kernels from the shipped HEALPix integrals with ``X`` kept complex.

    Returns ``{ellp: {s: (full, re_only, imag_ratio)}}``: the Eq. 20 kernel
    ``Re sum X(L1) conj X(L2)``, the ``Re X``-only kernel the shipped branch
    builds, and ``sum |Im X| / sum |Re X|``.
    """
    mask_map, _ = wlm.degrade_mask(NSIDE)

    def integrals(ell: int) -> np.ndarray:
        # (2 ell + 1, 3, L + 1, 2 L + 1): per m, full-M (T, E, B) integrals.
        return np.array(
            [
                full_from_pair(cma[0], cma[1], LMAX_OUT)
                for cma in (
                    wlm.compute_spin_weighted_integrals(
                        ell, m, mask_for_ell=mask_map, target_nside=NSIDE
                    )
                    for m in range(-ell, ell + 1)
                )
            ]
        )

    central = integrals(ELL)
    out = {}
    for ellp in ELLPS:
        prime = central if ellp == ELL else integrals(ellp)
        out[ellp] = {}
        for s in SPECTRA:
            k = COUPLING_SPECTRA.index(s)
            x = np.einsum(
                "mLM,nLM->mnL",
                central[:, _CENTRAL_FIELD[k]],
                prime[:, _PRIME_FIELD[k]].conj(),
            ).reshape(-1, LMAX_OUT + 1)
            out[ellp][s] = (
                (x.T @ x.conj()).real,
                x.real.T @ x.real,
                np.abs(x.imag).sum() / np.abs(x.real).sum(),
            )
    return out


@pytest.fixture(scope="module")
def kernels(tmp_path_factory):
    """
    For the as-is and rotated masks: shipped HEALPix kernels, GL kernels, and
    the reference complex-``X`` HEALPix kernels, all at ``NSIDE``, ``ELL``,
    ``ELLPS``.
    """
    base_alm = hp.map2alm(
        hp.read_map(os.path.join(DATA, "baseline_mask.fits")), lmax=LW, iter=10
    )
    rotated_alm = hp.Rotator(rot=ROTATION).rotate_alm(base_alm.copy(), lmax=LW)

    tmp = tmp_path_factory.mktemp("rotated_mask")
    result = {}
    for label, alm in (("azimuthal", base_alm), ("rotated", rotated_alm)):
        hp.write_map(
            str(tmp / f"{label}.fits"),
            hp.alm2map(alm, nside=NSIDE, lmax=LW),
            dtype=np.float64,
            overwrite=True,
        )
        wlm = MaskWlm(f"{label}.fits", load_path=str(tmp))
        common = {
            "centralell": ELL,
            "ellprange": ELLPS,
            "nside": NSIDE,
            "dryrun": True,
            "spectra": SPECTRA,
        }
        result[label] = {
            "healpix": precompute_acc_kernels(wlm, None, grid="healpix", **common),
            "gl": precompute_acc_kernels(wlm, None, grid="gl", lw=LW, **common),
            "reference": _reference_healpix_kernels(wlm),
        }
    return result


def _cases():
    return [(s, ellp) for s in SPECTRA for ellp in ELLPS]


def _relative_gaps(kernels, healpix_term) -> dict:
    """Per ``(s, ellp)``: (HEALPix rotated / GL rotated - 1, HEALPix rotated /
    HEALPix as-is - 1), with ``healpix_term(label, s, ellp)`` the HEALPix ACC
    term under test."""
    gaps = {}
    for s, ellp in _cases():
        gl_rot = _acc_term(kernels["rotated"]["gl"][ellp][f"{s}x{s}"])
        rot = healpix_term("rotated", s, ellp)
        az = healpix_term("azimuthal", s, ellp)
        gaps[(s, ellp)] = (rot / gl_rot - 1.0, rot / az - 1.0)
    return gaps


def _report(gaps: dict, which: int) -> str:
    return "; ".join(f"{s}({ellp}) {g[which]:+.2e}" for (s, ellp), g in gaps.items())


def test_reference_complex_theta_meets_both_tolerances(kernels):
    """
    The complex-``X`` reference, computed from the HEALPix integrals, meets
    both tolerances -- so they are achievable -- and the shipped kernel equals
    it to rounding.

    Written to hold for either a ``Re``-only or a full complex contraction:
    on the rotated mask the shipped kernel must equal the reference's
    ``Re``-only contraction or its full one, nothing else; the two tests
    below then decide which.
    """

    def close(a, b):
        return np.allclose(a, b, rtol=1e-12, atol=1e-14 * np.abs(b).max())

    for s, ellp in _cases():
        key = f"{s}x{s}"
        for label, imag_bound in (("azimuthal", (0.0, 1e-12)), ("rotated", (0.3, 1.0))):
            full, re_only, imag_ratio = kernels[label]["reference"][ellp][s]
            shipped = kernels[label]["healpix"][ellp][key]
            assert (
                imag_bound[0] <= imag_ratio <= imag_bound[1]
            ), f"{label} {s} ellp={ellp}: |Im X|/|Re X| = {imag_ratio:.3e}"
            if label == "azimuthal":
                assert close(full, shipped) and close(
                    re_only, shipped
                ), f"{label} {key} ellp={ellp}: reference != shipped"
            else:
                assert close(re_only, shipped) or close(full, shipped), (
                    f"{label} {key} ellp={ellp}: shipped kernel is neither the "
                    "Re-only nor the full reference contraction"
                )

        # As-is mask: Im X = 0, so the shipped HEALPix vs GL gap is transform
        # error alone, and must sit inside the invariance tolerance.
        as_is = _acc_term(kernels["azimuthal"]["healpix"][ellp][key])
        gl_as_is = _acc_term(kernels["azimuthal"]["gl"][ellp][key])
        assert (
            abs(as_is / gl_as_is - 1.0) < TOL_INVARIANCE
        ), f"{key} ellp={ellp}: as-is HEALPix/GL - 1 = {as_is / gl_as_is - 1:.3e}"

    gaps = _relative_gaps(
        kernels,
        lambda label, s, ellp: _acc_term(kernels[label]["reference"][ellp][s][0]),
    )
    assert all(abs(g[0]) < TOL_VS_GL for g in gaps.values()), _report(gaps, 0)
    assert all(abs(g[1]) < TOL_INVARIANCE for g in gaps.values()), _report(gaps, 1)


def test_healpix_kernel_matches_gl_on_rotated_mask(kernels):
    """Check 1 on the shipped HEALPix branch: ACC term within ``TOL_VS_GL`` of
    GL on the rotated mask, for every spectrum and ``ellp`` (a Re-only
    contraction would miss this by 4.1e-3 to 6.8e-3)."""
    gaps = _relative_gaps(
        kernels,
        lambda label, s, ellp: _acc_term(kernels[label]["healpix"][ellp][f"{s}x{s}"]),
    )
    assert all(abs(g[0]) < TOL_VS_GL for g in gaps.values()), _report(gaps, 0)


def test_healpix_kernel_is_orientation_invariant(kernels):
    """Check 2 on the shipped HEALPix branch: ACC term of the rotated mask
    within ``TOL_INVARIANCE`` of the as-is mask, for every spectrum and
    ``ellp`` (a Re-only contraction would miss this by 4.1e-3 to 6.8e-3)."""
    gaps = _relative_gaps(
        kernels,
        lambda label, s, ellp: _acc_term(kernels[label]["healpix"][ellp][f"{s}x{s}"]),
    )
    assert all(abs(g[1]) < TOL_INVARIANCE for g in gaps.values()), _report(gaps, 1)
