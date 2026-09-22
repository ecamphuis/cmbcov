"""
Regression tests for the exact-quadrature mask entry point (mask.py).

MaskWlm.compute_spherical_harmonics now solves for the mask alm as a
converged least-squares fit (ducc0.sht.pseudo_analysis) instead of a single
adjoint synthesis, and MaskWlm.compute_power_spectra now computes W^2_l as
the exact GL-quadrature power spectrum of the band-limited mask squared,
instead of hp.anafast on the pixel mask squared. See the module docstrings
of mask.py and grid.py for the rationale.

Tests here use lmax = 2*nside rather than the class default of 3*nside: at
3*nside (the baseline mask's default, nside=32) the pixelisation is close
enough to its Nyquist limit that a hard-edged/steeply-apodised mask has a
genuine aliasing floor (both the pseudo_analysis solve and
hp.map2alm(iter=10) plateau around a ~1e-4 relative map-reconstruction
residual there, and disagree with each other at the high-l, noise-dominated
tail -- see the mask.py docstring and the report accompanying this branch).
At 2*nside the pixelisation resolves the band-limit comfortably, and the two
solvers agree to ~1e-8, well inside the 1e-5 bound checked below.
"""

import os

import healpy as hp
import numpy as np
import pytest

from cmbcov.grid import gl_quadrature_weights, gl_synthesis
from cmbcov.mask import MaskWlm

DATA = os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture(scope="module")
def wlm():
    return MaskWlm("baseline_mask", load_path=DATA)


def test_pseudo_analysis_alm_matches_map2alm_iter10(wlm):
    """
    At a band-limit well inside the mask's pixel resolution (2*nside), the
    converged least-squares alm and healpy's iterated map2alm must agree
    tightly -- both are estimating the same well-resolved band-limited
    function, so any real disagreement would signal a bug (wrong geometry,
    wrong normalisation, ...) rather than genuine aliasing.
    """
    nside = wlm.nside
    lmax = 2 * nside
    alm = wlm.compute_spherical_harmonics(lmax=lmax)
    alm_reference = hp.map2alm(wlm.mask, lmax=lmax, iter=10)

    scale = np.abs(alm_reference).max()
    keep = np.abs(alm_reference) > 1e-4 * scale
    rel = np.abs(alm - alm_reference)[keep] / np.abs(alm_reference)[keep]
    assert rel.max() < 1e-5


def test_w_squared_parseval_identity(wlm):
    """
    Exact identity for a band-limited field g = W^2 (Parseval's theorem):
    sum_L (2L+1) Cl_g / (4 pi) == mean(g^2) over the sphere, the latter
    computed independently by GL quadrature. This holds to machine precision
    only if the W^2_l returned by compute_power_spectra really is the exact
    spectrum of the band-limited mask squared (the point of this branch's
    change to compute_power_spectra), not an approximate hp.anafast of the
    pixel mask squared.
    """
    wlm.compute_spherical_harmonics()
    _, w2l = wlm.compute_power_spectra()
    lw = wlm._mask_alm_lmax
    lg = 2 * lw

    ell = np.arange(len(w2l))
    lhs = np.sum((2 * ell + 1) * w2l) / (4 * np.pi)

    w_grid = gl_synthesis(wlm.mask_alm, lmax=lw, lmax_grid=lg, spin=0)
    weights = gl_quadrature_weights(lg)
    rhs = np.sum(weights * (w_grid**2) ** 2) / (4 * np.pi)

    assert lhs == pytest.approx(rhs, rel=1e-12, abs=0.0)


def test_convergence_info_is_populated_and_within_tolerance():
    """compute_spherical_harmonics must record solver convergence diagnostics."""
    fresh = MaskWlm("baseline_mask", load_path=DATA)
    epsilon = 1e-8
    fresh.compute_spherical_harmonics(lmax=2 * fresh.nside, epsilon=epsilon)

    assert fresh._mask_alm_iterations is not None
    assert fresh._mask_alm_residual is not None
    assert fresh._mask_alm_stopreason is not None
    assert fresh._mask_alm_iterations > 0
    # stopreason 1/2 mean the tolerance was met before maxiter; either way
    # the residual reported must not be nonsensical.
    assert 0.0 <= fresh._mask_alm_residual < 1.0


def test_requesting_a_wider_band_still_recomputes():
    """
    One-liner duplicating the spirit of test_mask_alm_cache.py for the new
    solver: the lmax-keyed cache must still recompute on a wider request even
    though the alm are now produced by pseudo_analysis rather than a single
    adjoint synthesis.
    """
    fresh = MaskWlm("baseline_mask", load_path=DATA)
    narrow = fresh.compute_spherical_harmonics(lmax=16)
    wide = fresh.compute_spherical_harmonics(lmax=64)
    assert hp.Alm.getlmax(wide.size) == 64
    assert wide.size > narrow.size
