"""
Guards against the band-limit cache in MaskWlm.

If compute_spherical_harmonics memoised mask_alm behind a guard that
returned before ever reading lmax, the first caller would fix the band
limit for the lifetime of the object: a degrade_mask(16) followed by
compute_power_spectra() would then silently return a mask spectrum
truncated to 3*16 rather than 3*nside. That matters most for masks with
small-scale structure (point-source masks), whose power extends to high L.
"""

import numpy as np
import pytest

healpy = pytest.importorskip("healpy")

from cmbcov.mask import MaskWlm  # noqa: E402

NSIDE = 32


@pytest.fixture
def mask_file(tmp_path):
    npix = healpy.nside2npix(NSIDE)
    theta, _ = healpy.pix2ang(NSIDE, np.arange(npix))
    edge = np.radians(40.0)
    mask = np.zeros(npix)
    cap = theta < edge
    mask[cap] = 0.5 * (1.0 + np.cos(np.pi * theta[cap] / edge))
    path = tmp_path / "m.fits"
    healpy.write_map(str(path), mask, overwrite=True, dtype=np.float64)
    return str(tmp_path), "m.fits"


def test_power_spectrum_is_independent_of_call_order(mask_file):
    load_path, name = mask_file
    direct, _ = MaskWlm(name, load_path=load_path).compute_power_spectra(save=False)

    after_degrade = MaskWlm(name, load_path=load_path)
    after_degrade.degrade_mask(NSIDE // 2)
    delayed, _ = after_degrade.compute_power_spectra(save=False)

    assert len(direct) == len(delayed)
    np.testing.assert_array_equal(direct, delayed)


def test_requesting_a_larger_lmax_recomputes(mask_file):
    load_path, name = mask_file
    wlm = MaskWlm(name, load_path=load_path)
    small = wlm.compute_spherical_harmonics(lmax=16)
    large = wlm.compute_spherical_harmonics(lmax=64)
    assert healpy.Alm.getlmax(large.size) == 64
    assert large.size > small.size


def test_requesting_a_smaller_lmax_reuses_the_cache(mask_file):
    """A wider band already contains the narrower one, so do not recompute."""
    load_path, name = mask_file
    wlm = MaskWlm(name, load_path=load_path)
    wide = wlm.compute_spherical_harmonics(lmax=64)
    narrow = wlm.compute_spherical_harmonics(lmax=16)
    assert narrow is wide


# --- The mirror-image bug: "ensure the alm exists" downgrading a prior solve ---
#
# compute_power_spectra (and the kernel entry points) must not ensure the alm
# by calling compute_spherical_harmonics() bare, i.e. requesting lmax=2*nside,
# maxiter=20, epsilon=1e-10 all at once: the cache guard compares maxiter and
# epsilon by equality, so a caller who had deliberately solved at a wider band
# limit or with a bigger iteration budget would fail the guard and have that
# solve silently replaced by the narrower default one -- no warning, since the
# default solve converges perfectly well. The tests below pin both directions:
# an existing solve is never weakened, and a deliberately narrow one (what
# degrade_mask leaves behind) is still widened back to the default.

STRICT_LMAX = 80  # != 2*NSIDE, and converges well inside maxiter=60
STRICT_MAXITER = 60


def test_compute_power_spectra_preserves_a_stricter_prior_solve(mask_file):
    load_path, name = mask_file
    wlm = MaskWlm(name, load_path=load_path)
    wlm.compute_spherical_harmonics(
        lmax=STRICT_LMAX, maxiter=STRICT_MAXITER, epsilon=1e-10
    )

    wl, wlsq = wlm.compute_power_spectra(save=False)

    assert healpy.Alm.getlmax(wlm.mask_alm.size) == STRICT_LMAX
    assert wlm._mask_alm_maxiter == STRICT_MAXITER
    # The spectra must be the ones implied by that alm, not by a re-solve.
    assert len(wl) == STRICT_LMAX + 1
    assert len(wlsq) == 2 * STRICT_LMAX + 1


def test_kernel_entry_points_preserve_a_stricter_prior_solve(mask_file):
    """The same "ensure the alm" call sites feed the coupling kernels."""
    load_path, name = mask_file
    wlm = MaskWlm(name, load_path=load_path)
    wlm.compute_spherical_harmonics(
        lmax=STRICT_LMAX, maxiter=STRICT_MAXITER, epsilon=1e-10
    )

    wlm.compute_master_coupling_kernels(l1max=16, l2max=16, output_dir=load_path)

    assert healpy.Alm.getlmax(wlm.mask_alm.size) == STRICT_LMAX
    assert wlm._mask_alm_maxiter == STRICT_MAXITER


def test_degrade_mask_preserves_a_wider_prior_solve(mask_file):
    """degrade_mask low-passes the alm, so a wider cached one is usable as-is."""
    load_path, name = mask_file
    wlm = MaskWlm(name, load_path=load_path)
    wlm.compute_spherical_harmonics(
        lmax=STRICT_LMAX, maxiter=STRICT_MAXITER, epsilon=1e-10
    )

    degraded, target = wlm.degrade_mask(NSIDE // 2)

    assert target == NSIDE // 2
    assert len(degraded) == healpy.nside2npix(NSIDE // 2)
    assert healpy.Alm.getlmax(wlm.mask_alm.size) == STRICT_LMAX


def test_compute_power_spectra_still_widens_a_narrow_solve(mask_file):
    """
    The fix must not become "solve only if mask_alm is None": degrade_mask
    deliberately solves at a *low* band limit (3 * target_nside), and letting
    that truncation leak into W_l is the original bug this module guards.
    """
    load_path, name = mask_file
    wlm = MaskWlm(name, load_path=load_path)
    wlm.degrade_mask(NSIDE // 2)
    assert healpy.Alm.getlmax(wlm.mask_alm.size) == 3 * (NSIDE // 2)

    wl, _ = wlm.compute_power_spectra(save=False)

    assert healpy.Alm.getlmax(wlm.mask_alm.size) == 2 * NSIDE
    assert len(wl) == 2 * NSIDE + 1


def test_resolving_the_alm_invalidates_cached_spectra(mask_file):
    """W_l and W^2_l are functions of the alm; they must not outlive it."""
    load_path, name = mask_file
    wlm = MaskWlm(name, load_path=load_path)
    wl_default, _ = wlm.compute_power_spectra(save=False)
    assert len(wl_default) == 2 * NSIDE + 1

    wlm.compute_spherical_harmonics(
        lmax=STRICT_LMAX, maxiter=STRICT_MAXITER, epsilon=1e-10
    )
    wl_strict, wlsq_strict = wlm.compute_power_spectra(save=False)

    assert len(wl_strict) == STRICT_LMAX + 1
    assert len(wlsq_strict) == 2 * STRICT_LMAX + 1
