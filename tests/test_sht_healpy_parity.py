"""
Parity checks: ``cmbcov.sht`` ducc0-backed primitives
against their healpy counterparts, on random alm/maps.

healpy stays an independent reference here (as everywhere else in the test
suite) -- these tests are exactly what license the removal of the forbidden
``hp.map2alm`` / ``hp.alm2map`` / ``hp.alm2cl`` / ``hp.almxfl`` calls from the
package itself in favour of the functions tested here.
"""

import healpy as hp
import numpy as np
import pytest

from cmbcov.sht import alm2cl, almxfl, ducc0_alm2map, ducc0_map2alm

NSIDE = 16
LMAX = 3 * NSIDE - 1


def random_alm(rng, lmax, ncomp=None):
    """Random healpy-layout alm set (real m=0 to represent a real map)."""
    ell, m = hp.Alm.getlm(lmax)
    shape = (hp.Alm.getsize(lmax),) if ncomp is None else (ncomp, len(ell))
    alm = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    alm[..., m == 0] = alm[..., m == 0].real
    return alm


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


# --------------------------------------------------------------------------- #
# ducc0_alm2map vs hp.alm2map
# --------------------------------------------------------------------------- #
def test_alm2map_spin0(rng):
    alm = random_alm(rng, LMAX)
    expected = hp.alm2map(alm, NSIDE, lmax=LMAX)
    got = ducc0_alm2map(alm, NSIDE, lmax=LMAX)
    diff = np.abs(got - expected)
    assert diff.max() < 1e-12, f"max abs diff {diff.max():.3e}"


def test_alm2map_pol(rng):
    alm = random_alm(rng, LMAX, ncomp=3)
    expected = hp.alm2map(alm, NSIDE, lmax=LMAX, pol=True)
    got = ducc0_alm2map(alm, NSIDE, lmax=LMAX, pol=True)
    diff = np.abs(got - np.asarray(expected))
    assert diff.max() < 1e-12, f"max abs diff {diff.max():.3e}"


def test_alm2map_restricted_mmax(rng):
    # Mirrors cmbcov.sht.cplx_spin_weighted_ylm's usage:
    # lmax != mmax.
    ell, m = 20, 5
    nalm = hp.Alm.getsize(ell, m)
    alms = np.zeros((3, nalm), dtype=complex)
    alms[0, -1] = 1.0
    alms[1, -1] = 1.0
    expected = hp.alm2map(alms, nside=NSIDE, pol=True, lmax=ell, mmax=m)
    got = ducc0_alm2map(alms, nside=NSIDE, pol=True, lmax=ell, mmax=m)
    diff = np.abs(got - np.asarray(expected))
    assert diff.max() < 1e-12, f"max abs diff {diff.max():.3e}"


# --------------------------------------------------------------------------- #
# ducc0_map2alm(iter=...) vs hp.map2alm(iter=...)
# --------------------------------------------------------------------------- #
def test_map2alm_iter3_spin0(rng):
    alm_in = random_alm(rng, LMAX)
    m = hp.alm2map(alm_in, NSIDE, lmax=LMAX)
    expected = hp.map2alm(m, lmax=LMAX, iter=3)
    got = ducc0_map2alm(m, LMAX, nside=NSIDE, pol=False, iter=3)
    diff = np.abs(got - expected)
    assert diff.max() < 1e-12, f"max abs diff {diff.max():.3e}"


def test_map2alm_iter3_pol(rng):
    alm_in = random_alm(rng, LMAX, ncomp=3)
    maps = hp.alm2map(alm_in, NSIDE, lmax=LMAX, pol=True)
    expected = hp.map2alm(maps, lmax=LMAX, iter=3, pol=True)
    got = ducc0_map2alm(np.asarray(maps), LMAX, nside=NSIDE, pol=True, iter=3)
    diff = np.abs(got - np.asarray(expected))
    assert diff.max() < 1e-12, f"max abs diff {diff.max():.3e}"


def test_map2alm_default_is_healpy_default(rng):
    """Without ``iter`` both transforms use healpy's default, iter=3."""
    alm_in = random_alm(rng, LMAX)
    m = hp.alm2map(alm_in, NSIDE, lmax=LMAX)
    got_default = ducc0_map2alm(m, LMAX, nside=NSIDE, pol=False)
    got_iter3 = ducc0_map2alm(m, LMAX, nside=NSIDE, pol=False, iter=3)
    assert np.array_equal(got_default, got_iter3)
    expected = hp.map2alm(m, lmax=LMAX)
    assert np.abs(got_default - expected).max() < 1e-12


# --------------------------------------------------------------------------- #
# alm2cl vs hp.alm2cl
# --------------------------------------------------------------------------- #
def test_alm2cl_single(rng):
    alm = random_alm(rng, LMAX)
    expected = hp.alm2cl(alm)
    got = alm2cl(alm)
    diff = np.abs(got - expected)
    assert diff.max() < 1e-14, f"max abs diff {diff.max():.3e}"


def test_alm2cl_cross_single(rng):
    alm1 = random_alm(rng, LMAX)
    alm2 = random_alm(rng, LMAX)
    expected = hp.alm2cl(alm1, alms2=alm2)
    got = alm2cl(alm1, alm2=alm2)
    diff = np.abs(got - expected)
    assert diff.max() < 1e-14, f"max abs diff {diff.max():.3e}"


def test_alm2cl_teb_nspec4(rng):
    # mirrors tests.conftest.compute_cross_spec_cplxmapalm's call pattern:
    # three-field (T, E, B) alm sets, nspec=4, lmax_out < lmax.
    alm1 = random_alm(rng, LMAX, ncomp=3)
    alm2 = random_alm(rng, LMAX, ncomp=3)
    lmax_out = LMAX - 5
    expected = hp.alm2cl(alm1, alms2=alm2, lmax_out=lmax_out, nspec=4)
    got = alm2cl(alm1, alm2=alm2, lmax_out=lmax_out, nspec=4)
    diff = np.abs(got - np.asarray(expected))
    assert diff.max() < 1e-14, f"max abs diff {diff.max():.3e}"


def test_alm2cl_teb_full(rng):
    alm1 = random_alm(rng, LMAX, ncomp=3)
    expected = hp.alm2cl(alm1)
    got = alm2cl(alm1)
    diff = np.abs(got - np.asarray(expected))
    assert diff.max() < 1e-14, f"max abs diff {diff.max():.3e}"


def test_alm2cl_restricted_lmax_out(rng):
    """lmax_out below the array's own band-limit truncates the output only."""
    alm1 = random_alm(rng, LMAX)
    alm2 = random_alm(rng, LMAX)
    lmax_out = LMAX - 10
    expected = hp.alm2cl(alm1, alms2=alm2, lmax_out=lmax_out)
    got = alm2cl(alm1, alm2=alm2, lmax_out=lmax_out)
    diff = np.abs(got - expected)
    assert diff.max() < 1e-14, f"max abs diff {diff.max():.3e}"


# --------------------------------------------------------------------------- #
# almxfl vs hp.almxfl
# --------------------------------------------------------------------------- #
def test_almxfl(rng):
    alm = random_alm(rng, LMAX)
    fl = rng.normal(size=LMAX + 1)
    expected = hp.almxfl(alm, fl)
    got = almxfl(alm, fl)
    diff = np.abs(got - expected)
    assert diff.max() < 1e-14, f"max abs diff {diff.max():.3e}"


def test_almxfl_short_fl(rng):
    """fl shorter than lmax+1 is treated as zero beyond its length."""
    alm = random_alm(rng, LMAX)
    fl = rng.normal(size=LMAX // 2)
    expected = hp.almxfl(alm, fl)
    got = almxfl(alm, fl)
    diff = np.abs(got - expected)
    assert diff.max() < 1e-14, f"max abs diff {diff.max():.3e}"


def test_spin_weighted_ylm_defaults_its_resolution():
    """
    Regression: cplx_spin_weighted_ylm derives nside from ell when the caller
    omits it, but the import that branch needs was removed as "unused" by an
    automated lint fix. Every existing caller passes nside explicitly, so the
    whole suite stayed green while the default path raised NameError.
    """
    from cmbcov.sht import cplx_spin_weighted_ylm

    maps = cplx_spin_weighted_ylm(8, 3)
    assert maps.shape[0] == 3  # spin 0, +2, -2
    assert np.iscomplexobj(maps)
