r"""
The default producer of the GL ACC coefficients: Legendre transforms at full
band (``cmbcov.approximations.acc._GL_PRODUCER = "legendre"``).

Without term selection the per-``m`` integrals
:math:`I_{\ell m, LM} = \int W Y_{\ell m} Y^*_{LM}` come from
:func:`~cmbcov.grid.banded_integrals_gl` with ``m_band = lmax_out + |m|``
(every ``M``), the mask's ring Fourier modes computed once per GL grid
(:class:`~cmbcov.approximations.acc._GLMaskModes`); the reference route
(``"two_sht"``) is :func:`~cmbcov.grid.spin_weighted_integrals_gl`, which
synthesises the mask and ``Y_lm``, multiplies and analyses.  Both evaluate
the same product with the same Gauss-Legendre rule on the same grid, so they
must agree to rounding: per ``m`` for both spins (and on the ``m < 0`` half,
which a held set now builds by reflection), and through the whole precompute
-- all 25 kernels and the T-only path, at ``ell == ellp`` and
``ell != ellp``, held and streamed.  Different code paths are not
bit-identical across BLAS/FFT libraries (macOS Accelerate vs Linux
OpenBLAS), so the comparisons use ``max |a - b| <= 1e-13 * max |reference|``
(measured: 1e-15 on macOS).

The mask is the baseline mask rotated off the pole (no azimuthal symmetry,
so ``Im Theta`` and every ``m``-``M`` coupling contribute) at ``lw = 24``,
which puts ``ell`` and each ``ellp`` on different grids.
"""

import os

import numpy as np
import pytest

healpy = pytest.importorskip("healpy")
import healpy as hp  # noqa: E402

from cmbcov.approximations import acc  # noqa: E402
from cmbcov.approximations.acc import (  # noqa: E402
    _gl_banded_grid,
    _gl_integrals,
    _gl_integrals_m,
    _GLMaskModes,
    precompute_acc_kernels,
)
from cmbcov.grid import gl_mask_modes, spin_weighted_integrals_gl  # noqa: E402
from cmbcov.mask import MaskWlm  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "data")

NSIDE = 16
LMAX = 2 * NSIDE  # acc.py convention: output band-limit LMAX - 1
ELL = 16
ELLPRANGE = [ELL, ELL + 1, ELL + 3]
LW = 24
ROTATION = (37.0, 21.0, 11.0)  # breaks the test mask's azimuthal symmetry

#: Cross-path tolerance, relative to the largest entry of the reference.
REL = 1e-13


@pytest.fixture(scope="module")
def rotated_mask_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("rotated_mask_producer")
    mask = hp.read_map(os.path.join(DATA, "baseline_mask.fits"))
    rotated = hp.Rotator(rot=ROTATION).rotate_map_pixel(mask)
    hp.write_map(str(out / "baseline_mask.fits"), rotated, overwrite=True)
    return str(out)


@pytest.fixture(scope="module")
def wlm(rotated_mask_dir):
    return MaskWlm("baseline_mask.fits", load_path=rotated_mask_dir)


@pytest.fixture(scope="module")
def mask_alm():
    mask = hp.read_map(os.path.join(DATA, "baseline_mask.fits"))
    alm = hp.map2alm(mask, lmax=LW, iter=10)
    return hp.Rotator(rot=ROTATION).rotate_alm(alm.copy(), lmax=LW)


def _rel(got, ref) -> float:
    return float(np.abs(got - ref).max() / np.abs(ref).max())


def test_the_default_producer_is_legendre():
    assert acc._GL_PRODUCER == "legendre"


def test_grids_of_the_test_setup_differ():
    """``ell`` and every ``ellp != ell`` sit on different grids here, so the
    comparisons below exercise distinct mask modes on the two sides."""
    grids = [_gl_banded_grid(LW, l_val, LMAX - 1) for l_val in ELLPRANGE]
    assert len(set(grids)) == len(grids), grids


@pytest.mark.parametrize("ell", [2, ELL, ELL + 3])
def test_per_m_coefficients_match_the_two_sht_route(mask_alm, ell):
    """``_gl_integrals_m`` (Legendre, shared mask modes) against
    :func:`spin_weighted_integrals_gl`, spin 0 and the (E, B) pair of spin
    2, for positive, zero and negative ``m``."""
    lmax_out = LMAX - 1
    modes = gl_mask_modes(mask_alm, LW, _gl_banded_grid(LW, ell, lmax_out))
    for m in sorted({-ell, -ell // 2, -1, 0, 1, ell // 2, ell}):
        u_t, u_eb = _gl_integrals_m(mask_alm, LW, ell, m, LMAX, mask_modes=modes)
        ref_t = spin_weighted_integrals_gl(mask_alm, LW, ell, m, lmax_out)
        ref_eb = spin_weighted_integrals_gl(mask_alm, LW, ell, m, lmax_out, spin=2)
        assert u_t.shape == ref_t.shape and u_eb.shape == ref_eb.shape
        assert _rel(u_t, ref_t) <= REL, (ell, m, "T")
        assert _rel(u_eb, ref_eb) <= REL, (ell, m, "EB")
        # entries with L < |M| are exact zeros on both routes
        lower = np.arange(lmax_out + 1)[:, None] < np.abs(
            np.arange(-lmax_out, lmax_out + 1)
        )
        assert not u_t[lower].any() and not u_eb[:, lower].any()
        # and the mask modes default (computed per call) changes nothing
        again = _gl_integrals_m(mask_alm, LW, ell, m, LMAX)
        np.testing.assert_array_equal(again[0], u_t)
        np.testing.assert_array_equal(again[1], u_eb)


def test_held_set_matches_the_two_sht_route_on_both_halves(mask_alm):
    """``_gl_integrals`` transforms ``m >= 0`` and reflects ``m < 0``; every
    order must still match the direct two-SHT integral, and the T-only set
    must be the T part of the full one."""
    lmax_out = LMAX - 1
    held = _gl_integrals(mask_alm, LW, ELL, LMAX)
    t_only = _gl_integrals(mask_alm, LW, ELL, LMAX, t_only=True)
    assert len(held) == 2 * ELL + 1
    for i, m in enumerate(range(-ELL, ELL + 1)):
        ref_t = spin_weighted_integrals_gl(mask_alm, LW, ELL, m, lmax_out)
        ref_eb = spin_weighted_integrals_gl(mask_alm, LW, ELL, m, lmax_out, spin=2)
        assert _rel(held[i][0], ref_t) <= REL, (m, "T")
        assert _rel(held[i][1], ref_eb) <= REL, (m, "EB")
        np.testing.assert_array_equal(t_only[i][0], held[i][0])
        assert t_only[i][1] is None


def _precompute(wlm, producer, monkeypatch, **kwargs):
    monkeypatch.setattr(acc, "_GL_PRODUCER", producer)
    return precompute_acc_kernels(
        wlm,
        None,
        centralell=ELL,
        ellprange=ELLPRANGE,
        nside=NSIDE,
        lw=LW,
        dryrun=True,
        **kwargs,
    )


@pytest.mark.parametrize(
    "spectra, n_kernels", [(None, 25), (("TT",), 1)], ids=["all-25", "T-only"]
)
@pytest.mark.parametrize(
    "max_memory_gb", [4.0, 2.0 / 1024], ids=["held", "streamed-or-stored"]
)
def test_precompute_kernels_match_the_two_sht_route(
    wlm, monkeypatch, tmp_path, spectra, n_kernels, max_memory_gb
):
    """All kernels of ``(ELL, ellp)`` for ``ellp`` in ``ELLPRANGE`` (``ell ==
    ellp`` and two ``ell != ellp``), with the central set held in RAM and
    with a budget that forces the store / streamed paths and small blocks."""
    common = {
        "spectra": spectra,
        "max_memory_gb": max_memory_gb,
        "scratch_dir": str(tmp_path),
    }
    new = _precompute(wlm, "legendre", monkeypatch, **common)
    ref = _precompute(wlm, "two_sht", monkeypatch, **common)
    assert sorted(new) == sorted(ref) == ELLPRANGE
    for ellp in ELLPRANGE:
        assert new[ellp].keys() == ref[ellp].keys()
        assert len(new[ellp]) == n_kernels
        for key, r in ref[ellp].items():
            assert np.abs(r).max() > 0
            diff = np.abs(new[ellp][key] - r).max()
            assert (
                diff <= REL * np.abs(r).max()
            ), f"ellp={ellp} {key}: {diff / np.abs(r).max():.2e}"


@pytest.mark.parametrize(
    "max_memory_gb", [4.0, 2.0 / 1024], ids=["held", "streamed-or-stored"]
)
def test_mask_modes_are_computed_once_per_grid(
    wlm, monkeypatch, tmp_path, max_memory_gb
):
    """One :func:`gl_mask_modes` per distinct grid over a whole precompute,
    whether the central set is held (built once) or re-synthesised per
    block alongside the primed side."""
    calls = []
    original = acc.gl_mask_modes

    def counting(mask_alm, lw, lmax_grid, *a, **k):
        calls.append(lmax_grid)
        return original(mask_alm, lw, lmax_grid, *a, **k)

    monkeypatch.setattr(acc, "gl_mask_modes", counting)
    _precompute(
        wlm,
        "legendre",
        monkeypatch,
        spectra=("TT",),
        max_memory_gb=max_memory_gb,
        scratch_dir=str(tmp_path),
    )
    grids = {_gl_banded_grid(LW, l_val, LMAX - 1) for l_val in ELLPRANGE}
    assert sorted(calls) == sorted(grids)


def test_mask_modes_cache_is_bounded_and_keyed_by_grid(mask_alm):
    cache = _GLMaskModes(mask_alm, LW, LMAX - 1, max_grids=2)
    a = cache(ELL)
    assert cache(ELL) is a
    b = cache(ELL + 1)
    c = cache(ELL + 3)  # evicts the least recently used grid, ELL's
    assert len(cache._modes) == 2
    assert cache(ELL + 1) is b and cache(ELL + 3) is c
    assert cache(ELL) is not a
    np.testing.assert_array_equal(cache(ELL), a)
    assert cache.matches(mask_alm, LW, LMAX - 1)
    assert not cache.matches(mask_alm.copy(), LW, LMAX - 1)
    assert not a.flags.writeable
