"""
Filter-and-bin map-making: the filtered sum rule ratio of ``cmbcov.assembly.filtering``.

The tests pin the ring-filter profiles and the self-adjointness of the ring
filter, the no-filter limit (``rho == 1`` bit for bit), the exact-basis mode
against an independent closed form (azimuthally symmetric mask about the scan
pole with a theta-independent cut, where ``K K^dagger`` is diagonal in ``m``),
the random-probe mode against the exact basis, the interpolation to a full
block, the transfer-function correction, the on-disk cache, and the physics:
the filtered covariance model against exact filtered covariance rows of an
off-pole cap with a ``m_c = lx sin(theta)`` cut.

Section 8 covers the polarisation channels ``"20"`` and ``"EE"``: the
block-to-channel table against ``Cov._compute_norm_xi`` and the
complete-basis probes against exact white-spectrum polarised rows, unfiltered
(``exact_covariance_row_pol``) and filtered (an in-test row over the complex
basis).

Section 9 covers the spectrum-aware TT x TT ratio (``spectrum_aware_ratio``):
complete-basis probes against exact filtered and unfiltered rows with
different spectra on the two sides, one and two frequencies; the no-filter
limit; seeds, nodes, the spectrum-length cap and the grid oversampling.

Masks are built analytically on a GL grid and band-limited with
``gl_analysis``; nothing is read from disk.
"""

import functools
import hashlib
import json
import os
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

healpy = pytest.importorskip("healpy")

from cmbcov.assembly.filtering import (  # noqa: E402
    BLOCK_CHANNEL,
    CACHE_SUBDIR,
    CHANNELS,
    FourierRingFilter,
    cached_sum_rule_ratio,
    channels_for,
    combined_profile,
    default_nodes,
    filtered_sum_rule_ratio,
    highpass_profile,
    lowpass_profile,
    rho_matrix,
    transfer_correction,
)
from cmbcov.exact.gl import _abs2_sum_over_m, _unit_pair_alm  # noqa: E402
from cmbcov.harmonics.grid import (  # noqa: E402
    gl_analysis,
    gl_minimal_lmax,
    gl_shape,
    gl_synthesis,
    gl_synthesis_single_m,
    gl_thetas,
    spin_weighted_integrals_gl,
)

DEG = np.pi / 180.0


# --------------------------------------------------------------------------- #
# Masks
# --------------------------------------------------------------------------- #
def _taper(x):
    """C^1 step: 0 for x <= 0, 1 for x >= 1."""
    x = np.clip(x, 0.0, 1.0)
    return np.sin(0.5 * np.pi * x) ** 2


def _cap_alm(lw, theta0, phi0, radius, apod, lmax_grid=128):
    """alm (band-limit ``lw``) of an apodised cap centred at (theta0, phi0)."""
    ntheta, nphi = gl_shape(lmax_grid)
    th = gl_thetas(lmax_grid)[:, None]
    ph = 2 * np.pi * np.arange(nphi)[None, :] / nphi
    cosg = np.cos(th) * np.cos(theta0) + np.sin(th) * np.sin(theta0) * np.cos(ph - phi0)
    gamma = np.arccos(np.clip(cosg, -1.0, 1.0))
    return gl_analysis(_taper((radius - gamma) / apod), lw, lmax_grid)


def _theta_independent(fn):
    """``profile`` callable returning ``fn(M)`` on every ring."""

    def profile(lmax_grid):
        ntheta, nphi = gl_shape(lmax_grid)
        M = np.arange(nphi // 2 + 1, dtype=float)
        return np.broadcast_to(fn(M), (ntheta, M.size)).copy()

    return profile


# Off-pole cap shared by tests 4 and 7: coupling width 17 at lw = 24.
OFF_LW = 24


@pytest.fixture(scope="module")
def off_pole_cap():
    return _cap_alm(OFF_LW, 55 * DEG, 1.0, 25 * DEG, 12 * DEG, lmax_grid=200)


# --------------------------------------------------------------------------- #
# 1. Profiles and the ring filter
# --------------------------------------------------------------------------- #
def test_highpass_profile_values():
    lg, lx = 20, 7.5
    th = gl_thetas(lg)
    ntheta, nphi = gl_shape(lg)

    sharp = highpass_profile(lg, lx, "sharp")
    assert sharp.shape == (ntheta, nphi // 2 + 1)
    assert sharp.dtype == np.float64
    for j in (0, 5, ntheta // 2, ntheta - 1):
        mc = lx * np.sin(th[j])
        for M in (0, 1, 3, 7, 8, 20):
            assert sharp[j, M] == (1.0 if M > mc else 0.0)
    assert np.all(sharp[:, 0] == 0.0)

    exp = highpass_profile(lg, lx, "exp", power=4.0)
    assert np.all(exp[:, 0] == 0.0)
    for j, M in ((0, 1), (3, 2), (ntheta // 2, 7), (ntheta // 2, 21)):
        assert exp[j, M] == pytest.approx(np.exp(-((lx * np.sin(th[j]) / M) ** 4)))
    # the exponential cut is 1/e at m_c(theta)
    j = ntheta // 2
    mc = lx * np.sin(th[j])
    M = np.arange(exp.shape[1])
    assert np.all(exp[j, M > mc] > np.exp(-1.0))
    assert np.all(exp[j, M < mc] < np.exp(-1.0))

    # lx = 0 means no filter, for both shapes (M = 0 kept)
    assert np.all(highpass_profile(lg, 0.0, "sharp") == 1.0)
    assert np.all(highpass_profile(lg, 0.0, "exp") == 1.0)

    with pytest.raises(ValueError):
        highpass_profile(lg, lx, "gaussian")
    with pytest.raises(ValueError):
        highpass_profile(lg, -1.0)
    with pytest.raises(ValueError):
        highpass_profile(lg, lx, "exp", power=0.0)
    with pytest.raises(ValueError):
        highpass_profile(-1, lx)


def test_lowpass_profile_values():
    lg, lx = 20, 7.5
    th = gl_thetas(lg)
    ntheta, nphi = gl_shape(lg)

    exp = lowpass_profile(lg, lx, "exp", power=4.0)
    assert exp.shape == (ntheta, nphi // 2 + 1)
    assert exp.dtype == np.float64
    # exp[-(M / (lx sin theta))^p], 1 at M = 0, 1/e at M = m_c(theta)
    assert np.all(exp[:, 0] == 1.0)
    for j, M in ((0, 1), (3, 2), (ntheta // 2, 7), (ntheta // 2, 21)):
        assert exp[j, M] == pytest.approx(np.exp(-((M / (lx * np.sin(th[j]))) ** 4)))
    j = ntheta // 2
    mc = lx * np.sin(th[j])
    M = np.arange(exp.shape[1])
    assert np.all(exp[j, M < mc] > np.exp(-1.0))
    assert np.all(exp[j, M > mc] < np.exp(-1.0))
    assert np.all(np.diff(exp, axis=1) <= 0)  # a low-pass: decreasing in M
    # the default is the exponential shape with power 6
    np.testing.assert_array_equal(
        lowpass_profile(lg, lx), lowpass_profile(lg, lx, "exp", 6.0)
    )

    sharp = lowpass_profile(lg, lx, "sharp")
    assert sharp.shape == exp.shape and sharp.dtype == np.float64
    for j in (0, 5, ntheta // 2, ntheta - 1):
        mc = lx * np.sin(th[j])
        for M in (0, 1, 3, 7, 8, 20):
            assert sharp[j, M] == (1.0 if M <= mc else 0.0)
    assert np.all(sharp[:, 0] == 1.0)  # M = 0 passes a low-pass

    # same m_c convention as the high-pass: at one lx the sharp pair is
    # complementary
    np.testing.assert_array_equal(
        sharp + highpass_profile(lg, lx, "sharp"), np.ones_like(sharp)
    )

    # a huge power overflows in the exponent: still 0, no warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        big = lowpass_profile(lg, lx, "exp", power=200.0)
    assert np.all(np.isfinite(big)) and big[:, 0].min() == 1.0

    for bad in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="lx"):
            lowpass_profile(lg, bad)
    with pytest.raises(ValueError, match="shape"):
        lowpass_profile(lg, lx, "gaussian")
    for bad in (0.0, -2.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="power"):
            lowpass_profile(lg, lx, "exp", power=bad)
    with pytest.raises(ValueError, match="lmax_grid"):
        lowpass_profile(-1, lx)


@pytest.mark.parametrize("high_shape", ["sharp", "exp"])
@pytest.mark.parametrize("low_shape", ["exp", "sharp"])
def test_combined_profile_is_the_product(high_shape, low_shape):
    lg, lx, lx_lp = 24, 2.0, 6.0
    low = {"lx": lx_lp, "shape": low_shape, "power": 3.0}
    h = combined_profile(lg, lx, high_shape, 4.0, low)
    expect = highpass_profile(lg, lx, high_shape, 4.0) * lowpass_profile(
        lg, lx_lp, low_shape, 3.0
    )
    np.testing.assert_array_equal(h, expect)
    assert h.shape == expect.shape and h.dtype == np.float64
    # a band: zero at M = 0 (high-pass) and far above the low-pass cut
    assert np.all(h[:, 0] == 0.0)
    far = np.arange(h.shape[1]) > 3 * lx_lp
    assert far.any() and np.all(h[:, far] < 1e-9)
    assert 0.0 <= h.min() and h.max() <= 1.0
    # shape and power default as for lowpass_profile ("exp", 6); power None is 6
    np.testing.assert_array_equal(
        combined_profile(lg, lx, high_shape, 4.0, {"lx": lx_lp, "power": None}),
        highpass_profile(lg, lx, high_shape, 4.0) * lowpass_profile(lg, lx_lp),
    )
    np.testing.assert_array_equal(
        combined_profile(lg, lx, high_shape, 4.0, {"lx": lx_lp}),
        highpass_profile(lg, lx, high_shape, 4.0) * lowpass_profile(lg, lx_lp),
    )
    # the sharp pair is the indicator of the band
    if high_shape == "sharp" and low_shape == "sharp":
        th = gl_thetas(lg)
        M = np.arange(h.shape[1])[None, :]
        mc = np.sin(th)[:, None]
        np.testing.assert_array_equal(
            h, ((M > lx * mc) & (M <= lx_lp * mc)).astype(float)
        )


def test_combined_profile_without_lowpass_is_the_highpass():
    """No low-pass: the very array of ``highpass_profile``, and a ``ratio``
    computed from it identical to that of ``highpass_profile`` itself."""
    for lg, lx, shape, power in ((20, 7.5, "sharp", 6.0), (30, 3.0, "exp", 4.0)):
        np.testing.assert_array_equal(
            combined_profile(lg, lx, shape, power),
            highpass_profile(lg, lx, shape, power),
        )
        np.testing.assert_array_equal(
            combined_profile(lg, lx, shape, power, None),
            highpass_profile(lg, lx, shape, power),
        )
    # the defaults are those of highpass_profile
    np.testing.assert_array_equal(combined_profile(20, 7.5), highpass_profile(20, 7.5))
    # lx = 0 (no high-pass) with a low-pass alone is not a band: refused
    with pytest.raises(ValueError, match="exceed"):
        combined_profile(20, 0.0, "sharp", 6.0, {"lx": 0.0})
    # ... but a positive low-pass cut over a null high-pass cut is accepted
    h = combined_profile(20, 0.0, "sharp", 6.0, {"lx": 5.0, "shape": "sharp"})
    np.testing.assert_array_equal(h, lowpass_profile(20, 5.0, "sharp"))


def test_combined_profile_errors():
    lg = 20
    with pytest.raises(ValueError, match="exceed"):
        combined_profile(lg, 5.0, "sharp", 6.0, {"lx": 5.0})
    with pytest.raises(ValueError, match="exceed"):
        combined_profile(lg, 5.0, "sharp", 6.0, {"lx": 4.0})
    with pytest.raises(ValueError, match="lowpass"):
        combined_profile(lg, 5.0, "sharp", 6.0, {"shape": "exp"})  # no lx
    with pytest.raises(ValueError, match="lowpass"):
        combined_profile(lg, 5.0, "sharp", 6.0, {"lx": 9.0, "width": 2.0})
    with pytest.raises(ValueError, match="shape"):
        combined_profile(lg, 5.0, "sharp", 6.0, {"lx": 9.0, "shape": "box"})
    with pytest.raises(ValueError, match="power"):
        combined_profile(lg, 5.0, "sharp", 6.0, {"lx": 9.0, "power": 0.0})
    with pytest.raises(ValueError, match="lx"):
        combined_profile(lg, 5.0, "sharp", 6.0, {"lx": float("inf")})


def test_fourier_ring_filter_self_adjoint_and_identity():
    lg = 24
    ntheta, nphi = gl_shape(lg)
    rng = np.random.default_rng(3)
    f = rng.standard_normal((ntheta, nphi))
    g = rng.standard_normal((ntheta, nphi))
    F = FourierRingFilter(highpass_profile(lg, 9.0, "exp"))
    lhs = np.sum(f * F(g))
    rhs = np.sum(F(f) * g)
    assert abs(lhs - rhs) <= 1e-13 * np.sum(np.abs(f * g))
    # the filter acts ring by ring and the GL weight is constant along a
    # ring, so it is self-adjoint for the weighted inner product too
    wq = np.repeat(np.linspace(0.1, 1.0, ntheta)[:, None], nphi, axis=1)
    lhs = np.sum(wq * f * F(g))
    rhs = np.sum(wq * F(f) * g)
    assert abs(lhs - rhs) <= 1e-13 * np.sum(np.abs(wq * f * g))

    ident = FourierRingFilter(np.ones((ntheta, nphi // 2 + 1)))
    out = ident(f)
    assert out is not f
    assert np.array_equal(out, f)

    # on a single azimuthal order a theta-independent cut is a multiplication
    lmax = 12
    alm = np.zeros(healpy.Alm.getsize(lmax), dtype=complex)
    alm[healpy.Alm.getidx(lmax, 9, 5)] = 0.3 - 0.7j
    y = gl_synthesis_single_m(alm, lmax, 5, lg)
    hfun = _theta_independent(lambda M: 1.0 - np.exp(-((M / 3.0) ** 2)))
    np.testing.assert_allclose(
        FourierRingFilter(hfun(lg))(y),
        (1.0 - np.exp(-((5 / 3.0) ** 2))) * y,
        rtol=0,
        atol=1e-14 * np.abs(y).max(),
    )

    # a (Q, U) pair is filtered map by map
    pair = np.stack([f, g])
    np.testing.assert_array_equal(F(pair), np.stack([F(f), F(g)]))
    np.testing.assert_array_equal(ident(pair), pair)

    with pytest.raises(ValueError):
        F(f[:, :-2])
    with pytest.raises(ValueError):
        F(f[0])
    with pytest.raises(ValueError):
        FourierRingFilter(np.ones(5))
    with pytest.raises(ValueError):
        FourierRingFilter(np.full((ntheta, nphi // 2 + 1), np.nan))


def test_default_nodes():
    nodes = default_nodes(4000)
    assert nodes[0] == 32 and nodes[-1] == 3999
    assert all(b > a for a, b in zip(nodes, nodes[1:]))
    ratios = np.array(nodes[1:]) / np.array(nodes[:-1])
    assert ratios.max() <= 1.25 * np.sqrt(1.25) + 1e-12
    assert default_nodes(20) == [19]
    assert default_nodes(2) == [1]
    assert default_nodes(100, node_min=10, ratio=2.0)[-1] == 99
    with pytest.raises(ValueError):
        default_nodes(100, ratio=1.0)
    with pytest.raises(ValueError):
        default_nodes(1)


def _check_merge(nodes, lmax, node_min, ratio, lx, step):
    """The merge rule of ``default_nodes`` with a cut, checked against the
    geometric list: dense nodes every ``step`` from ``round(0.95 lx)`` to
    ``2 lx``; a geometric node between the first and the last dense node is
    dropped exactly when it is closer than ``step / 2`` to a dense node."""
    geometric = default_nodes(lmax, node_min, ratio)
    start = int(np.floor(0.95 * lx + 0.5))
    dense = [n for n in range(start, int(np.floor(2 * lx)) + 1, step) if n <= lmax - 1]
    assert set(dense) <= set(nodes)
    for n in geometric:
        inside = dense and dense[0] <= n <= dense[-1] and n != lmax - 1
        near = inside and min(abs(n - k) for k in dense) < step / 2
        assert (n in nodes) == (not near or n in dense), n
    assert set(nodes) == set(dense) | {n for n in geometric if n in nodes}
    assert nodes == sorted(set(nodes)) and nodes[-1] == lmax - 1


def test_default_nodes_near_the_cut():
    # the survey case: lx = 300, dense every 20 from 285 to 585
    old = default_nodes(1001)
    assert old == [32, 40, 50, 62, 78, 98, 122, 152, 190, 238] + [
        298,
        372,
        465,
        581,
        726,
        1000,
    ]
    nodes = default_nodes(1001, lx=300)
    assert nodes == old[:10] + list(range(285, 586, 20)) + [726, 1000]
    # 298, 372 and 581 sit within 10 of a dense node; 465 is one
    assert default_nodes(1001, lx=300, near_cut_step=None) == nodes
    assert default_nodes(1001, lx=300.0, near_cut_step=20) == nodes
    # near_cut_step = 0: the old list; no lx: the old list
    assert default_nodes(1001, lx=300, near_cut_step=0) == default_nodes(1001)
    assert default_nodes(1001, 32, 1.25) == default_nodes(1001, 32, 1.25, lx=None)
    # the merge rule, for several spacings and cuts
    for lmax, node_min, ratio, lx, step in (
        (1001, 32, 1.25, 300, None),
        (1001, 32, 1.25, 300, 37),
        (4000, 32, 1.25, 300, 60),
        (4000, 20, 1.1, 150, 9.6),  # 9.6 rounds to 10
        (2000, 32, 1.5, 100, None),  # 100 / 15 = 6.67 -> 7
        (3000, 32, 1.25, 1000, 200),
    ):
        nodes = default_nodes(lmax, node_min, ratio, lx=lx, near_cut_step=step)
        expect_step = max(1, int(np.floor((lx / 15 if step is None else step) + 0.5)))
        _check_merge(nodes, lmax, node_min, ratio, lx, expect_step)
    # a geometric node exactly half a step from its dense neighbours stays
    nodes = default_nodes(1001, lx=300, near_cut_step=24)
    assert 465 in nodes and 453 in nodes and 477 in nodes
    assert 298 not in nodes and 372 not in nodes and 581 not in nodes
    # spacing at least 1: a tiny step or cut gives every multipole
    assert default_nodes(100, lx=10)[:11] == list(range(10, 21))
    assert default_nodes(100, lx=300, near_cut_step=0.4) == default_nodes(100)
    # the cut above lmax: no dense node fits, the old list
    assert default_nodes(1000, lx=2000) == default_nodes(1000)
    # the cut partly above lmax: dense nodes up to lmax - 1, which stays
    nodes = default_nodes(1000, lx=600, near_cut_step=40)
    assert [n for n in nodes if n >= 570] == list(range(570, 991, 40)) + [999]
    # the cut below node_min: the dense nodes are kept below node_min
    nodes = default_nodes(500, node_min=32, lx=10)
    assert nodes[:11] == list(range(10, 21)) and 32 in nodes
    # lmax tiny
    assert default_nodes(2, lx=300) == [1]
    assert default_nodes(2, lx=1) == [1]
    assert default_nodes(3, lx=1) == [1, 2]
    with pytest.raises(ValueError, match="near_cut_step"):
        default_nodes(100, lx=30, near_cut_step=-1)
    with pytest.raises(ValueError, match="lx"):
        default_nodes(100, lx=0)


def test_node_plan():
    from cmbcov.assembly.filtering import (
        BANDLIMIT_FLOOR,
        BANDLIMIT_NODE_LIMIT,
        NodePlan,
        node_plan,
    )

    plan = node_plan(3500, 32, 1.25, 300)
    assert isinstance(plan, NodePlan)
    assert plan.nodes == default_nodes(3500, 32, 1.25, lx=300)
    assert len(plan.nprobe) == len(plan.lmax_int_floor) == len(plan.nodes)
    floor = int(np.ceil(BANDLIMIT_FLOOR * 300))
    for n, k, f in zip(*plan):
        near = 285 <= n < 600
        # complete basis from the first dense node to below 2 lx
        assert k == (2 * n + 1 if near else 32), n
        assert f == (floor if 285 <= n < BANDLIMIT_NODE_LIMIT * 300 else 0), n
    assert [n for n, f in zip(plan.nodes, plan.lmax_int_floor) if f] == list(
        range(285, 390, 20)
    )
    # an explicit near-cut count, and the plain count elsewhere
    plan = node_plan(3500, 32, 1.25, 300, nprobe=16, nprobe_near_cut=128)
    assert {k for n, k in zip(plan.nodes, plan.nprobe) if 285 <= n < 600} == {128}
    assert {k for n, k in zip(plan.nodes, plan.nprobe) if not 285 <= n < 600} == {16}
    # near_cut_step = 0: the old nodes, the per-node settings still apply
    plan = node_plan(1001, 32, 1.25, 300, near_cut_step=0)
    assert plan.nodes == default_nodes(1001)
    assert dict(zip(plan.nodes, plan.nprobe))[298] == 597
    assert dict(zip(plan.nodes, plan.nprobe))[238] == 32
    assert dict(zip(plan.nodes, plan.lmax_int_floor))[372] == floor
    assert dict(zip(plan.nodes, plan.lmax_int_floor))[465] == 0
    with pytest.raises(ValueError, match="nprobe_near_cut"):
        node_plan(1001, 32, 1.25, 300, nprobe_near_cut=0)
    with pytest.raises(ValueError, match="nprobe"):
        node_plan(1001, 32, 1.25, 300, nprobe=0)


def test_default_nodes_through_a_lowpass():
    # the toy of the validation: lmax 321, high-pass 30, low-pass 200
    base = default_nodes(321, lx=30)
    exp = default_nodes(321, lx=30, lx_lowpass=200)
    assert exp == default_nodes(321, lx=30, lx_lowpass=200, lowpass_shape="exp")
    # dense nodes every 200 / 15 = 13.3 -> 13 from round(0.6 * 200) = 120 to 400
    # (cut at lmax - 1 = 320); the geometric nodes 122, 152, 190, 238 within 6.5
    # of one are dropped, 98 and the end stay
    dense = list(range(120, 321, 13))
    assert exp == sorted({n for n in base if n < 120 or n == 320} | set(dense))
    assert exp[-3:] == [302, 315, 320] and 98 in exp and 122 not in exp
    assert all(b > a for a, b in zip(exp, exp[1:]))
    # the high-pass nodes are those of the high-pass alone
    assert [n for n in exp if n < 98] == [n for n in base if n < 98]

    # sharp: also every 200 / 50 = 4 from 190 to 230; they replace the coarse
    # nodes of that stretch (211, 224)
    sharp = default_nodes(321, lx=30, lx_lowpass=200, lowpass_shape="sharp")
    onset = list(range(190, 231, 4))
    assert set(onset) <= set(sharp)
    assert not {211, 224} & set(sharp)  # coarse nodes inside the onset stretch
    assert [n for n in sharp if n < 185] == [n for n in exp if n < 185]
    assert [n for n in sharp if n > 230] == [n for n in exp if n > 230]
    assert len(sharp) > len(exp) and sharp == sorted(set(sharp))
    assert 185 in sharp and 237 in sharp  # the coarse nodes around the onset

    # the survey case: SPT-3G's low-pass at 6144 is above 0.6 lmax-ish: no node
    assert default_nodes(3500, lx=300, lx_lowpass=6144) == default_nodes(3500, lx=300)
    assert default_nodes(3500, lx=300, lx_lowpass=6144, lowpass_shape="sharp") == (
        default_nodes(3500, lx=300)
    )
    # the range starts at 0.6 lx_lowpass: 3686 > 3499 here, 3600 just fits
    assert default_nodes(3500, lx=300, lx_lowpass=5800) != default_nodes(3500, lx=300)
    # near_cut_step = 0 switches every extra node off
    assert default_nodes(321, lx=30, near_cut_step=0, lx_lowpass=200) == (
        default_nodes(321)
    )
    # a range that only just fits: 0.6 * 1500 = 900, then 1000 = lmax - 1
    only = default_nodes(1001, lx=300, lx_lowpass=1500)
    assert only == sorted(set(default_nodes(1001, lx=300)) | {900})
    with pytest.raises(ValueError, match="lx_lowpass"):
        default_nodes(321, lx=30, lx_lowpass=30)
    with pytest.raises(ValueError, match="lx_lowpass"):
        default_nodes(321, lx=30, lx_lowpass=20)
    with pytest.raises(ValueError, match="lx_lowpass"):
        default_nodes(321, lx=30, lx_lowpass=float("nan"))
    with pytest.raises(ValueError, match="lx_lowpass"):
        default_nodes(321, lx=30, lx_lowpass=float("inf"))
    with pytest.raises(ValueError, match="lx_lowpass needs lx"):
        default_nodes(321, lx_lowpass=200)
    with pytest.raises(ValueError, match="lowpass_shape"):
        default_nodes(321, lx=30, lx_lowpass=200, lowpass_shape="box")


def test_node_plan_through_a_lowpass():
    from cmbcov.assembly.filtering import node_plan

    plan = node_plan(321, 32, 1.25, 30, nprobe=32, nprobe_near_cut=128, lx_lowpass=200)
    assert plan.nodes == default_nodes(321, lx=30, lx_lowpass=200)
    base = node_plan(321, 32, 1.25, 30, nprobe=32, nprobe_near_cut=128)
    for n, k, f in zip(*plan):
        in_rolloff = 120 <= n < 400
        in_cut = 29 <= n < 60
        assert k == (128 if in_rolloff or in_cut else 32), n
        # the band-limit floor is that of the high-pass cut alone
        assert f == (105 if 29 <= n < 39 else 0), n
    # complete basis in the rolloff with nprobe_near_cut=None
    plan = node_plan(321, 32, 1.25, 30, nprobe_near_cut=None, lx_lowpass=200)
    assert [k for n, k in zip(plan.nodes, plan.nprobe) if n >= 120] == [
        2 * n + 1 for n in plan.nodes if n >= 120
    ]
    assert dict(zip(plan.nodes, plan.nprobe))[98] == 32
    # no low-pass argument, or one that adds no node: the plan of the high-pass
    spt = node_plan(3500, 32, 1.25, 300, lx_lowpass=6144)
    assert spt == node_plan(3500, 32, 1.25, 300)
    assert node_plan(3500, 32, 1.25, 300, lx_lowpass=None) == spt
    assert base == node_plan(321, 32, 1.25, 30, nprobe=32, nprobe_near_cut=128)
    # the sharp low-pass has the same probe rule on its extra nodes
    sharp = node_plan(
        321, 32, 1.25, 30, nprobe_near_cut=128, lx_lowpass=200, lowpass_shape="sharp"
    )
    assert {k for n, k in zip(sharp.nodes, sharp.nprobe) if n >= 120} == {128}
    # ... and the floors are those of the high-pass cut alone
    assert [n for n, f in zip(sharp.nodes, sharp.lmax_int_floor) if f] == [
        29,
        31,
        32,
        33,
        35,
        37,
    ]
    with pytest.raises(ValueError, match="lx_lowpass"):
        node_plan(321, 32, 1.25, 30, lx_lowpass=10)


# --------------------------------------------------------------------------- #
# 2. No filter: rho == 1 exactly
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "profile",
    [
        functools.partial(highpass_profile, lx=0.0, shape="sharp"),
        _theta_independent(np.ones_like),
    ],
)
def test_no_filter_gives_one(profile, off_pole_cap):
    # node 5 with 32 probes: exact basis (11 vectors); node 30: 4 random probes;
    # node 1: no E-mode probe exists (spin-2 channels are 1 by definition).
    # The identity filter returns a copy, so both operators do the same
    # arithmetic and the paired ratio is 1 bit for bit, in every channel.
    g = filtered_sum_rule_ratio(
        off_pole_cap,
        OFF_LW,
        profile,
        [5, 30, 1],
        7,
        channels=CHANNELS,
        nprobe=32,
        nthreads=1,
    )
    assert list(g) == list(CHANNELS)
    for ch in CHANNELS:
        assert g[ch].shape == (3, 8)
        assert np.all(g[ch] == 1.0), ch
    g = filtered_sum_rule_ratio(
        off_pole_cap, OFF_LW, profile, [30], 3, nprobe=4, nthreads=1
    )
    assert list(g) == ["00"]  # the default: TT x TT only
    assert np.all(g["00"] == 1.0)


def _ratio(cap, profile, nodes, nprobe, dmax=4):
    return filtered_sum_rule_ratio(
        cap,
        OFF_LW,
        profile,
        nodes,
        dmax,
        channels=CHANNELS,
        nprobe=nprobe,
        seed=3,
        nthreads=1,
    )


def test_no_lowpass_ratio_is_the_highpass_ratio_bit_for_bit(off_pole_cap):
    """
    The ratio computed with ``combined_profile`` and no low-pass, and with a
    low-pass so far above the grid that it is 1 in float64 (sharp: every
    mode passes; exp: ``exp(-1e-48) == 1``), equals the ratio of
    ``highpass_profile`` itself, every channel and entry (the same arithmetic
    on the same arrays).  Node 6: the complete basis; node 14: 3 random
    probes.
    """
    nodes, nprobe = [6, 14], [13, 3]
    for shape in ("sharp", "exp"):
        old = _ratio(
            off_pole_cap,
            functools.partial(highpass_profile, lx=5.0, shape=shape),
            nodes,
            nprobe,
        )
        for lowpass in (
            None,
            {"lx": 1e9, "shape": "sharp"},
            {"lx": 1e9, "shape": "exp", "power": 6.0},
        ):
            new = _ratio(
                off_pole_cap,
                functools.partial(
                    combined_profile, lx=5.0, shape=shape, lowpass=lowpass
                ),
                nodes,
                nprobe,
            )
            for ch in CHANNELS:
                assert np.array_equal(new[ch], old[ch]), (shape, lowpass, ch)


def test_lowpass_lowers_the_ratio_and_is_monotonic_in_its_cut(off_pole_cap):
    """Removing the modes above a cut lowers the ratio: for the complete
    basis at the node 20 every channel's ratio is below the high-pass-alone
    one at every distance, and a lower low-pass cut lowers it further (cut
    at 24, then 12), for both low-pass shapes.  (The cap sits at colatitude
    55 deg, so the cut ``lx sin(theta)`` bites on the modes of a degree
    well below ``lx``.)"""
    nodes = [20]
    nprobe = [41]
    hp = _ratio(
        off_pole_cap,
        functools.partial(highpass_profile, lx=5.0, shape="sharp"),
        nodes,
        nprobe,
    )
    for shape in ("exp", "sharp"):
        g = {}
        for cut in (24.0, 12.0):
            g[cut] = _ratio(
                off_pole_cap,
                functools.partial(
                    combined_profile,
                    lx=5.0,
                    shape="sharp",
                    lowpass={"lx": cut, "shape": shape},
                ),
                nodes,
                nprobe,
            )
        for ch in CHANNELS:
            assert np.all(g[24.0][ch] > 0.0)
            assert np.all(g[24.0][ch] < hp[ch]), (shape, ch)
            assert np.all(g[12.0][ch] < g[24.0][ch]), (shape, ch)
            assert np.all(g[12.0][ch] < 0.97 * hp[ch]), (shape, ch)


# --------------------------------------------------------------------------- #
# 3. Exact basis against the closed form of a symmetric mask
# --------------------------------------------------------------------------- #
def test_exact_basis_matches_closed_form_for_symmetric_mask():
    r"""
    Polar cap (azimuthally symmetric about the scan pole) and a cut that does
    not depend on theta.  Then ``W`` and ``F`` both conserve ``m``, ``K`` is
    diagonal in ``m`` with ``K_{lm,Lm} = h(m) W_{lm,Lm}``, and

        Xi^F(l, l') = 1/(n n') sum_m h(m)^4 |(W^2)_{lm, l'm}|^2 ,

    ``W^2`` as a coupling matrix, computed here from the alm of the map W^2
    with ``spin_weighted_integrals_gl``.

    What makes the comparison exact (not just close):

    * ``margin = lw``, so ``lmax_int >= l' + lw``: the middle sum over ``L``
      in ``(W P W)_{lm,l'm} = sum_{L <= lmax_int} W_{lm,Lm} W_{Lm,l'm}`` is
      complete (``W_{Lm,l'm} = 0`` for ``L > l' + lw``), so ``W P W = W^2``
      on the columns of degree ``l'``;
    * ``lw_node = min(lw, 2 lmax_int) = lw``: no mask truncation;
    * a theta-independent ``h(M)`` maps a band-limited map to a band-limited
      map (on a ring ``Y_lm`` is ``e^{i m phi}`` times a constant), so the
      filtered operator is exact on the minimal GL grid like the unfiltered
      one;
    * ``W^2`` has band-limit ``2 lw`` and its alm is analysed exactly on the
      grid ``2 lw``; ``spin_weighted_integrals_gl`` picks its own exact grid;
    * ``2 l' + 1 <= nprobe``: the exact orthonormal basis, no noise.

    Measured: 2.6e-15.  ``h(M)^4`` (not ``h^2``) is tested because ``h`` is
    not 0/1-valued.
    """
    lw = 16
    alm = _cap_alm(lw, 0.0, 0.0, 40 * DEG, 20 * DEG, lmax_grid=64)

    def hfun(M):
        return 1.0 - np.exp(-((M / 2.0) ** 2))

    dmax = 6
    nodes = [3, 12]  # node 3: d = 4..6 fall below l = 0 (one-sided estimate)
    g = filtered_sum_rule_ratio(
        alm,
        lw,
        _theta_independent(hfun),
        nodes,
        dmax,
        nprobe=32,
        margin=lw,
        nthreads=1,
    )["00"]

    w_map = gl_synthesis(alm, lw, 2 * lw)
    w2_alm = gl_analysis(w_map**2, 2 * lw, 2 * lw)
    for i, lp in enumerate(nodes):
        lo = lp + dmax
        num = np.zeros(lo + 1)
        den = np.zeros(lo + 1)
        for m in range(-lp, lp + 1):
            integ = spin_weighted_integrals_gl(w2_alm, 2 * lw, lp, m, lo)
            a2 = np.abs(integ[:, lo + m]) ** 2
            num += hfun(abs(m)) ** 4 * a2
            den += a2
        r = num / den
        ref = [r[lp]] + [
            0.5 * (r[lp + d] + r[lp - d]) if lp - d >= 0 else r[lp + d]
            for d in range(1, dmax + 1)
        ]
        np.testing.assert_allclose(g[i], ref, rtol=0, atol=1e-10)
        assert 0.0 < g[i].min() < 1.0  # a genuine, non-trivial filter


# --------------------------------------------------------------------------- #
# 4. Random probes against the exact basis
# --------------------------------------------------------------------------- #
def test_random_probes_converge_and_are_reproducible(off_pole_cap):
    """
    Off-pole cap, sharp cut at ``8 sin(theta)``, node ``l' = 40``.

    Measured over 40 seeds, rms relative error per ``d`` (``d = 0..6``) of
    the random-probe ratio against the exact basis (81 vectors): 3.9-4.0%
    with 8 probes, 2.5-2.6% with 16, 1.3-1.5% with 64 (the ``1/sqrt(N)``
    law); worst single seed 2.2 times the rms; mean over seeds within 1.5
    standard errors of zero.  The errors of different ``d`` are strongly
    correlated (one common factor per seed).  With 12 seeds the rms is known
    to about +-20%, hence the bands below.
    """
    prof = functools.partial(highpass_profile, lx=8.0, shape="sharp")
    lp, dmax = 40, 6

    def run(nprobe, seed, nodes=(lp,)):
        return filtered_sum_rule_ratio(
            off_pole_cap,
            OFF_LW,
            prof,
            list(nodes),
            dmax,
            nprobe=nprobe,
            seed=seed,
            nthreads=1,
        )["00"]

    exact = run(2 * lp + 1, 0)[0]
    assert np.array_equal(exact, run(200, 7)[0])  # exact basis ignores the seed
    err16 = np.array([run(16, s)[0] / exact - 1 for s in range(12)])
    err64 = np.array([run(64, s)[0] / exact - 1 for s in range(12)])
    rms16 = np.sqrt(np.mean(err16**2))
    rms64 = np.sqrt(np.mean(err64**2))
    assert 0.6 * 0.026 < rms16 < 1.6 * 0.026
    assert 0.6 * 0.014 < rms64 < 1.6 * 0.014
    assert rms64 < 0.8 * rms16
    assert np.abs(err16).max() < 4 * 0.026
    assert np.abs(err64).max() < 4 * 0.014
    assert abs(err64.mean()) < 3 * 0.014 / np.sqrt(12)

    # Reproducible, seed-dependent, and independent of the other nodes.
    a = run(16, 5)
    assert np.array_equal(a, run(16, 5))
    assert not np.array_equal(a, run(16, 6))
    both = run(16, 5, nodes=(20, lp))
    assert np.array_equal(both[1], a[0])
    assert np.array_equal(both[0], run(16, 5, nodes=(20,))[0])

    # ... and of the other channels requested: every channel sees the same
    # probes, and "00" is computed exactly as when it is alone.
    many = filtered_sum_rule_ratio(
        off_pole_cap,
        OFF_LW,
        prof,
        [lp],
        dmax,
        channels=("EE", "00", "20"),
        nprobe=16,
        seed=5,
        nthreads=1,
    )
    assert list(many) == ["EE", "00", "20"]
    assert np.array_equal(many["00"], a)
    ee_alone = filtered_sum_rule_ratio(
        off_pole_cap,
        OFF_LW,
        prof,
        [lp],
        dmax,
        channels=("EE",),
        nprobe=16,
        seed=5,
        nthreads=1,
    )
    assert np.array_equal(many["EE"], ee_alone["EE"])


# --------------------------------------------------------------------------- #
# 5. Interpolation and the transfer correction
# --------------------------------------------------------------------------- #
def test_rho_matrix():
    from scipy.interpolate import PchipInterpolator

    nodes = [10, 20, 40]
    dmax = 3
    d = np.arange(dmax + 1)
    g = np.array([0.5 + 0.05 * d, 0.7 + 0.02 * d, 0.9 - 0.01 * d])
    lmax = 60
    rho = rho_matrix(nodes, g, lmax)
    assert rho.shape == (lmax, lmax) and rho.dtype == np.float64
    assert np.array_equal(rho, rho.T)

    # at the nodes, on every even diagonal (lbar integer)
    for i, node in enumerate(nodes):
        for k in range(0, 4):
            l1, l2 = node - k, node + k
            assert rho[l1, l2] == pytest.approx(g[i, min(2 * k, dmax)], abs=1e-14)
    # held constant outside the node range
    assert rho[3, 3] == g[0, 0] and rho[2, 5] == g[0, 3] and rho[0, 1] == g[0, 1]
    assert rho[50, 50] == g[-1, 0] and rho[45, 59] == g[-1, dmax]
    # in between: PCHIP in ln(lbar); d > dmax uses the dmax column
    interp = PchipInterpolator(np.log(nodes), g, axis=0)
    assert rho[25, 28] == pytest.approx(interp(np.log(26.5))[3], abs=1e-14)
    assert rho[14, 16] == pytest.approx(interp(np.log(15.0))[2], abs=1e-14)
    assert rho[10, 30] == pytest.approx(interp(np.log(20.0))[dmax], abs=1e-14)
    assert rho[12, 31] == pytest.approx(interp(np.log(21.5))[dmax], abs=1e-14)

    # one node: a function of d only
    one = rho_matrix([15], g[:1], 30)
    ll = np.arange(30)
    dd = np.minimum(np.abs(ll[:, None] - ll[None, :]), dmax)
    assert np.array_equal(one, g[0][dd])

    rho_matrix(default_nodes(1000), np.full((len(default_nodes(1000)), 5), 0.9), 1000)
    with pytest.raises(ValueError):
        rho_matrix([10, 10, 40], g, lmax)
    with pytest.raises(ValueError):
        rho_matrix([0, 20, 40], g, lmax)
    with pytest.raises(ValueError):
        rho_matrix(nodes, g[:2], lmax)


def test_transfer_correction():
    rng = np.random.default_rng(1)
    rho = rng.uniform(0.5, 1.0, (6, 6))
    keep = rho.copy()
    fl = np.array([0.0, -0.2, np.nan, 0.5, np.inf, 0.8, 0.3, 0.1])
    fr = np.array([0.9, 0.7, 0.6, 0.0, 0.5, 0.4])
    out = transfer_correction(rho, fl, fr)
    left = np.array([1.0, 1.0, 1.0, 0.5, 1.0, 0.8])
    right = np.array([0.9, 0.7, 0.6, 1.0, 0.5, 0.4])
    np.testing.assert_allclose(out, rho / np.outer(left, right), rtol=1e-15)
    assert np.array_equal(rho, keep) and out is not rho
    with pytest.raises(ValueError):
        transfer_correction(rho, fl[:3], fr)


# --------------------------------------------------------------------------- #
# 6. Cache
# --------------------------------------------------------------------------- #
def test_cached_sum_rule_ratio(tmp_path):
    calls = []

    def compute():
        calls.append(1)
        value = 0.25 * len(calls)
        return {"00": np.full((2, 3), value), "EE": np.full((2, 3), -value)}

    identity = {
        "mask": "abc",
        "lx": 300.0,
        "nodes": [32, 40],
        "dmax": 2,
        "channels": ["00", "EE"],
    }
    g1 = cached_sum_rule_ratio(tmp_path, identity, compute)
    assert len(calls) == 1 and list(g1) == ["00", "EE"]
    assert np.all(g1["00"] == 0.25) and np.all(g1["EE"] == -0.25)
    key = json.dumps(identity, sort_keys=True)
    digest = hashlib.blake2b(key.encode(), digest_size=8).hexdigest()
    path = tmp_path / CACHE_SUBDIR / f"rho_{digest}.npz"
    assert path.exists()
    assert os.listdir(tmp_path / CACHE_SUBDIR) == [path.name]  # no temp left
    with np.load(path) as data:
        assert sorted(data.files) == ["channels", "g_00", "g_EE", "identity"]

    # hit: same identity, key order irrelevant, numpy scalars accepted
    reordered = {
        "channels": ["00", "EE"],
        "dmax": np.int64(2),
        "nodes": np.array([32, 40]),
        "lx": 300.0,
        "mask": "abc",
    }
    g2 = cached_sum_rule_ratio(tmp_path, reordered, compute)
    assert len(calls) == 1 and list(g2) == ["00", "EE"]
    for ch in g1:
        assert np.array_equal(g2[ch], g1[ch])

    # changed identity: recompute, separate file
    g3 = cached_sum_rule_ratio(tmp_path, {**identity, "lx": 200.0}, compute)
    assert len(calls) == 2 and np.all(g3["00"] == 0.5)
    assert len(os.listdir(tmp_path / CACHE_SUBDIR)) == 2

    # a file under the right name with a different identity (a hash
    # collision, or a stale file) is a miss and is overwritten
    with open(path, "wb") as f:
        np.savez(
            f, channels=np.array(["00"]), g_00=np.zeros((2, 3)), identity=np.array("{}")
        )
    g4 = cached_sum_rule_ratio(tmp_path, identity, compute)
    assert len(calls) == 3 and np.all(g4["00"] == 0.75)
    assert np.all(cached_sum_rule_ratio(tmp_path, identity, compute)["00"] == 0.75)
    assert len(calls) == 3

    # the TT-only layout of earlier versions (a single ``g``), even under the
    # same identity, is a miss
    with open(path, "wb") as f:
        np.savez(f, g=np.zeros((2, 3)), identity=np.array(key))
    g5 = cached_sum_rule_ratio(tmp_path, identity, compute)
    assert len(calls) == 4 and np.all(g5["00"] == 1.0)

    with pytest.raises(ValueError, match="2-D"):
        cached_sum_rule_ratio(tmp_path / "x", identity, lambda: {"00": np.ones(3)})
    with pytest.raises(ValueError, match="mapping"):
        cached_sum_rule_ratio(tmp_path / "y", identity, lambda: np.ones((2, 3)))


# --------------------------------------------------------------------------- #
# 7. Physics: the filtered covariance model against exact filtered rows
# --------------------------------------------------------------------------- #
class _Operator:
    """``K = A W F S`` on the GL grid ``lg`` (a minimal reference
    implementation, independent of :mod:`cmbcov.assembly.filtering`); ``F = None`` is
    no filter."""

    def __init__(self, lg, lmax, w_map, F=None):
        self.lg, self.lmax, self.w, self.F = lg, lmax, w_map, F

    def apply(self, alm):
        f = gl_synthesis(alm, self.lmax, self.lg, nthreads=1)
        if self.F is not None:
            f = self.F(f)
        return gl_analysis(f * self.w, self.lmax, self.lg, nthreads=1)

    def adjoint_unit(self, unit, m):
        f = gl_synthesis_single_m(unit, self.lmax, m, self.lg, nthreads=1) * self.w
        if self.F is not None:
            f = self.F(f)
        return gl_analysis(f, self.lmax, self.lg, nthreads=1)


def _threaded(fn, items):
    """``[fn(x) for x in items]`` on threads: ducc releases the GIL, and every
    operator call is single-threaded, so the results are those of the loop."""
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
        return list(pool.map(fn, items))


def _exact_rows(op, cl, rows):
    """``Cov(C~_l, C~_l')`` for every ``l`` and each ``l'`` in ``rows``."""
    lmax = op.lmax
    ell_alm = healpy.Alm.getlm(lmax)[0]
    cl_alm = cl[ell_alm]
    ell = np.arange(lmax + 1)

    def one_m(job):
        lp, mp = job
        r, s, has_s = _unit_pair_alm(lmax, lp, mp)
        ra = op.apply(op.adjoint_unit(r, mp) * cl_alm)
        sa = op.apply(op.adjoint_unit(s, mp) * cl_alm) if has_s else None
        return _abs2_sum_over_m(ra, sa, ell_alm, lmax)

    jobs = [(lp, mp) for lp in rows for mp in range(lp + 1)]
    terms = iter(_threaded(one_m, jobs))
    out = np.zeros((len(rows), lmax + 1))
    for i, lp in enumerate(rows):
        for mp in range(lp + 1):
            out[i] += (1.0 if mp == 0 else 2.0) * next(terms)
        out[i] *= 2.0 / ((2 * ell + 1) * (2 * lp + 1))
    return out


def _coupling(op):
    """``M[l, L] = 1/(2l+1) sum_{m M} |K_{lm,LM}|^2``: ``<C~_l> = sum_L M C_L``.

    Row ``l`` from ``K^dagger e_{lm}``, whose ``(L, M)`` entry is
    ``conj K_{lm,LM}``."""
    lmax = op.lmax
    ell_alm = healpy.Alm.getlm(lmax)[0]

    def row(ell):
        out = np.zeros(lmax + 1)
        for m in range(ell + 1):
            r, s, has_s = _unit_pair_alm(lmax, ell, m)
            ra = op.adjoint_unit(r, m)
            sa = op.adjoint_unit(s, m) if has_s else None
            out += (1.0 if m == 0 else 2.0) * _abs2_sum_over_m(ra, sa, ell_alm, lmax)
        return out / (2 * ell + 1)

    return np.array(_threaded(row, range(lmax + 1)))


PHYS_LMAX = 80  # internal band-limit of the reference
PHYS_LX = 8.0
PHYS_ROWS = [24, 40, 56]  # 3, 5 and 7 times the cut
PHYS_DMAX = 6


@pytest.fixture(scope="module")
def unfiltered_reference(off_pole_cap):
    lg = gl_minimal_lmax(PHYS_LMAX, OFF_LW)
    w_map = gl_synthesis(off_pole_cap, OFF_LW, lg, nthreads=1)
    ell = np.arange(PHYS_LMAX + 1.0)
    cl = np.zeros(PHYS_LMAX + 1)
    cl[2:] = (
        (1000.0 + 4500.0 * np.exp(-((ell[2:] - 60.0) ** 2) / (2 * 25.0**2)))
        * 2
        * np.pi
        / (ell[2:] * (ell[2:] + 1))
    )
    op_u = _Operator(lg, PHYS_LMAX, w_map)
    return {
        "lg": lg,
        "w_map": w_map,
        "cl": cl,
        "mean_u": _coupling(op_u) @ cl,
        "cov_u": _exact_rows(op_u, cl, PHYS_ROWS),
        "op_u": op_u,
    }


# Measured errors of the filtered covariance model, in per cent of the exact
# filtered diagonal, over |l - l'| <= 6 of the rows l' = 24, 40, 56 (lx = 8,
# lmax 80, lw 24, coupling width 17), rho from rho_matrix on default_nodes
# (exact basis at every node):
#
#                     form B: cov_u[F C] rho/(F F')   form A: cov_u[C] rho
#   sharp  l' = 24    diag -5.88  max 6.26            diag +5.42  max 5.66
#          l' = 40    diag -0.82  max 0.83            diag -0.01  max 0.25
#          l' = 56    diag -0.25  max 0.25            diag +0.04  max 0.06
#   exp    l' = 24    diag -6.12  max 6.59            diag +7.31  max 7.59
#          l' = 40    diag -0.76  max 0.77            diag +0.37  max 0.37
#          l' = 56    diag -0.24  max 0.24            diag +0.16  max 0.16
#
# Form B is systematically low: Cov[F C] is not F_l F_l' Cov[C] when F_L
# varies across the mask kernel, and at this toy scale it does (sharp:
# F = 0.76, 0.87, 0.91 at 24, 40, 56).  Naive C -> F C is 24%, 10%, 7% low
# (sharp).  At 3 lx (l' = 24) the model is only good to ~6-8% at this
# scale; the research (lx = 30, l up to 232) found <= 0.25% at >= 3 lx, but
# there l / (mask coupling width) was several times larger.  The same
# numbers without interpolation (rho of the exact white rows element by
# element) differ by <= 0.1% from these; the reference grid (+20) moves them
# by <= 0.02%.  Tolerances: the measured maximum plus 20-40%.
TOLERANCES = {
    # shape: {row: (form B max, form A max)} in per cent
    "sharp": {24: (7.5, 7.0), 40: (1.1, 0.35), 56: (0.35, 0.1)},
    "exp": {24: (8.0, 9.0), 40: (1.0, 0.5), 56: (0.32, 0.22)},
}


@pytest.mark.parametrize("shape", ["sharp", "exp"])
def test_filtered_covariance_model(shape, off_pole_cap, unfiltered_reference):
    ref = unfiltered_reference
    lg = ref["lg"]
    op_f = _Operator(
        lg,
        PHYS_LMAX,
        ref["w_map"],
        FourierRingFilter(highpass_profile(lg, PHYS_LX, shape)),
    )
    cl = ref["cl"]
    fl = (_coupling(op_f) @ cl) / ref["mean_u"]
    fl[:2] = 1.0  # C_0 = C_1 = 0: F undefined there and never used
    cov_f = _exact_rows(op_f, cl, PHYS_ROWS)
    cov_uf = _exact_rows(ref["op_u"], fl * cl, PHYS_ROWS)

    nodes = default_nodes(PHYS_LMAX - PHYS_DMAX, node_min=16)
    g = filtered_sum_rule_ratio(
        off_pole_cap,
        OFF_LW,
        functools.partial(highpass_profile, lx=PHYS_LX, shape=shape),
        nodes,
        PHYS_DMAX,
        nprobe=2 * nodes[-1] + 1,  # exact basis everywhere
        nthreads=1,
    )["00"]
    rho = rho_matrix(nodes, g, PHYS_LMAX + 1)
    corr = transfer_correction(rho, fl, fl)

    ds = np.arange(-PHYS_DMAX, PHYS_DMAX + 1)
    for i, lp in enumerate(PHYS_ROWS):
        ls = lp + ds
        truth = cov_f[i, ls]
        diag = truth[PHYS_DMAX]
        err_b = 100 * (cov_uf[i, ls] * corr[ls, lp] - truth) / diag
        err_a = 100 * (ref["cov_u"][i, ls] * rho[ls, lp] - truth) / diag
        naive = 100 * (cov_uf[i, lp] / diag - 1)
        print(
            f"{shape} l'={lp} F={fl[lp]:.3f} rho={rho[lp, lp]:.3f} "
            f"B diag {err_b[PHYS_DMAX]:+.3f} max {np.abs(err_b).max():.3f} | "
            f"A diag {err_a[PHYS_DMAX]:+.3f} max {np.abs(err_a).max():.3f} | "
            f"C->FC {naive:+.1f}"
        )
        tol_b, tol_a = TOLERANCES[shape][lp]
        assert np.abs(err_b).max() < tol_b
        assert np.abs(err_a).max() < tol_a
        assert np.abs(err_b).max() < abs(naive) / 3


# --------------------------------------------------------------------------- #
# 8. Polarisation channels
# --------------------------------------------------------------------------- #
def test_block_channel_is_the_norm_xi_table():
    """``BLOCK_CHANNEL`` maps every T/E block exactly as ``_compute_norm_xi``."""
    import types

    from cmbcov.assembly.covariance import Cov

    # Xi[0] = 00, Xi[3] = 20, EE = (Xi[1] + Xi[2]) / 2: distinct sentinels
    xi = [np.array([1.0]), np.array([10.0]), np.array([30.0]), np.array([7.0])]
    table = Cov._compute_norm_xi(types.SimpleNamespace(Xi=xi))
    name = {1.0: "00", 7.0: "20", 20.0: "EE"}
    assert {pair: name[float(v[0])] for pair, v in table.items()} == BLOCK_CHANNEL
    assert len(BLOCK_CHANNEL) == 16
    assert all("B" not in s1 + s2 for s1, s2 in BLOCK_CHANNEL)

    assert channels_for(["TTxTT"]) == ("00",)
    assert channels_for(["EExTE", "TTxTT"]) == ("00", "EE")
    assert channels_for([("ET", "TT"), "BBxBB"]) == ("20",)
    assert channels_for(["BBxBB", "TBxEB"]) == ()
    assert channels_for(["TTxTT", "TTxEE", "EExEE", "TExTE"]) == CHANNELS
    with pytest.raises(ValueError):
        channels_for(["TTxEExBB"])


def test_channel_arguments_are_checked(off_pole_cap):
    prof = functools.partial(highpass_profile, lx=0.0)
    for bad in ((), ("BB",), ("00", "TE")):
        with pytest.raises(ValueError, match="channel"):
            filtered_sum_rule_ratio(
                off_pole_cap, OFF_LW, prof, [5], 2, channels=bad, nthreads=1
            )
    # duplicates are dropped, a bare string is one channel
    g = filtered_sum_rule_ratio(
        off_pole_cap, OFF_LW, prof, [5], 2, channels="EE", nthreads=1
    )
    assert list(g) == ["EE"]
    g = filtered_sum_rule_ratio(
        off_pole_cap, OFF_LW, prof, [5], 2, channels=("20", "20"), nthreads=1
    )
    assert list(g) == ["20"]


_FIELDS = "TEB"


def _white_row_pol(w_map, F, lg, lmax, ellp, cls, block):
    r"""
    Exact row ``Cov(C~^{s1}_l, C~^{s2}_{l'})``, ``l = 0 .. lmax``, of the
    filtered polarised estimator ``K_0 = A W F S`` on T, ``K_2`` on (E, B),
    ``F = None`` for no filter: the algorithm of ``cmbcov.exact``
    ("Polarisation") with the filter inserted, over the complete *complex*
    basis ``e_{l'm'}``, ``m' = -l' .. l'`` (no folding), in the full-M
    representation.  ``u = K^dagger e_Z``, ``v = C u``, ``R^{.Z} = K v`` and

        Cov = 1/(n n') sum_{m m'} [R^{XW} conj R^{YZ} + R^{XZ} conj R^{YW}].

    Independent of ``filtered_sum_rule_ratio`` in basis, representation and
    algebra (a general ``C`` matrix, the Wick sum); shares only the GL
    transforms and the ring filter.
    """
    from cmbcov.harmonics.grid import full_from_pair, pair_from_full

    filt = F if F is not None else (lambda f: f)

    def S(a, spin):
        return gl_synthesis(a, lmax, lg, spin=spin, nthreads=1)

    def A(f, spin):
        return gl_analysis(f, lmax, lg, spin=spin, nthreads=1)

    def on_full(op, x):
        r, s = pair_from_full(x, lmax)
        return full_from_pair(op(r), op(s), lmax)

    def adj(spin):
        return lambda a: A(filt(w_map * S(a, spin)), spin)

    def fwd(spin):
        return lambda a: A(w_map * filt(S(a, spin)), spin)

    cmat = np.zeros((3, 3, lmax + 1))
    for key, cl in cls.items():
        i, j = _FIELDS.index(key[0]), _FIELDS.index(key[1])
        cmat[i, j] = cmat[j, i] = cl[: lmax + 1]
    cmat[1:, :, :2] = 0.0  # E and B have no L < 2
    cmat[:, 1:, :2] = 0.0

    (x, y), (z, wf) = ([_FIELDS.index(c) for c in s] for s in block)
    nm = 2 * lmax + 1
    cov = np.zeros(lmax + 1)
    for mp in range(-ellp, ellp + 1):
        R = np.zeros((3, 3, lmax + 1, nm), dtype=complex)
        for iz in {z, wf}:
            u = np.zeros((3, lmax + 1, nm), dtype=complex)
            if iz == 0:
                e = np.zeros((lmax + 1, nm), dtype=complex)
                e[ellp, lmax + mp] = 1.0
                u[0] = on_full(adj(0), e)
            else:
                e = np.zeros((2, lmax + 1, nm), dtype=complex)
                e[iz - 1, ellp, lmax + mp] = 1.0
                u[1:] = on_full(adj(2), e)
            v = np.einsum("xzl,zlm->xlm", cmat, u)
            R[0, iz] = on_full(fwd(0), v[0])
            R[1:, iz] = on_full(fwd(2), v[1:])
        cov += np.sum(
            R[x, wf] * np.conj(R[y, z]) + R[x, z] * np.conj(R[y, wf]), axis=1
        ).real
    ell = np.arange(lmax + 1)
    return cov / ((2 * ell + 1) * (2 * ellp + 1))


# channel -> (white spectrum, block): 2 Xi^{F,ch} is that exact block.
_WHITE = {
    "00": ("TT", ("TT", "TT")),
    "20": ("TE", ("TT", "EE")),
    "EE": ("EE", ("EE", "EE")),
}


@pytest.mark.parametrize(
    "shape, lowpass",
    [
        ("sharp", None),
        ("exp", None),
        # the band: a high-pass at 5 and a low-pass at 7.5 inside the rows
        # l = 0 .. 25 of the node (the low-pass bites from l = 7.5 up)
        pytest.param("sharp", {"lx": 7.5, "shape": "exp"}, id="sharp-lowpass-exp"),
        pytest.param("exp", {"lx": 7.5, "shape": "sharp"}, id="exp-lowpass-sharp"),
    ],
)
def test_channel_probes_are_the_exact_white_blocks(shape, lowpass, off_pole_cap):
    r"""
    Complete-basis probes of every channel against exact white-spectrum rows
    (module docstring of ``cmbcov.assembly.filtering``, "Polarisation channels"):
    ``2 S^{ch}_l / (n n')`` is the TT x TT row with only ``C^TT = 1``
    (``"00"``), the TT x EE row with only ``C^TE = 1`` (``"20"``) and the
    EE x EE row with only ``C^EE = 1`` (``"EE"``).

    * unfiltered sums against ``exact_covariance_row_pol(grid="gl")``, the
      package's exact reference (this pins the probe algebra);
    * the in-test row ``_white_row_pol`` without filter against the same
      (this pins the in-test reference);
    * filtered sums against ``_white_row_pol`` with the theta-dependent cut
      ``m_c = 5 sin(theta)`` on the off-pole cap;
    * ``filtered_sum_rule_ratio`` end to end against the ratio of the two
      reference rows.

    Same grid and internal band-limit everywhere, so the agreement is at
    round-off.  Measured (relative to the largest entry of each row):
    2e-16 to 8e-16 for all three channels, both shapes; the ratio of the
    end-to-end check agrees to ``rtol = 1e-12`` (asserted).
    """
    from cmbcov.assembly.filtering import _node_sums
    from cmbcov.exact.covariance import exact_covariance_row_pol

    ellp, dmax, margin = 9, 4, 12
    lmax = ellp + dmax + margin
    lg = gl_minimal_lmax(lmax, OFF_LW)
    w_map = gl_synthesis(off_pole_cap, OFF_LW, lg, nthreads=1)
    profile = functools.partial(combined_profile, lx=5.0, shape=shape, lowpass=lowpass)
    F = FourierRingFilter(profile(lg))

    exact, count, sums = _node_sums(
        w_map, F, ellp, lmax, lg, CHANNELS, 2 * ellp + 1, 0, 1
    )
    assert exact and count == 2 * ellp + 1
    ell = np.arange(lmax + 1)
    norm = 2.0 / ((2 * ell + 1) * (2 * ellp + 1))
    one = np.ones(lmax + 1)
    worst = 0.0
    refs = {}
    for ch in CHANNELS:
        spec, block = _WHITE[ch]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # margin 0 of the row: intended
            package = exact_covariance_row_pol(
                off_pole_cap,
                {spec: one},
                ellp,
                lmax,
                spectra=("TT", "EE"),
                grid="gl",
                lw=OFF_LW,
                lmax_grid=lg,
                lmax_int=lmax,
                nthreads=1,
            )[block]
        ref_u = _white_row_pol(w_map, None, lg, lmax, ellp, {spec: one}, block)
        ref_f = _white_row_pol(w_map, F, lg, lmax, ellp, {spec: one}, block)
        refs[ch] = (ref_f, ref_u)
        s_f, s_u = (norm * a for a in sums[ch])
        scale = np.abs(package).max()
        errs = [
            np.abs(s_u - package).max() / scale,
            np.abs(ref_u - package).max() / scale,
            np.abs(s_f - ref_f).max() / np.abs(ref_f).max(),
        ]
        print(f"{shape} {ch}: " + " ".join(f"{e:.1e}" for e in errs))
        worst = max(worst, *errs)
        # a genuine filter effect, not a trivial identity
        assert np.all(s_f[ellp - 2 : ellp + 3] < 0.9 * s_u[ellp - 2 : ellp + 3])
    assert worst < 1e-13

    g = filtered_sum_rule_ratio(
        off_pole_cap,
        OFF_LW,
        profile,
        [ellp],
        dmax,
        channels=CHANNELS,
        nprobe=2 * ellp + 1,
        margin=margin,
        nthreads=1,
    )
    for ch in CHANNELS:
        ref_f, ref_u = refs[ch]
        with np.errstate(invalid="ignore"):  # E has no l < 2: 0 / 0 there
            r = ref_f / ref_u
        ref = [r[ellp]] + [
            0.5 * (r[ellp + d] + r[ellp - d]) for d in range(1, dmax + 1)
        ]
        np.testing.assert_allclose(g[ch][0], ref, rtol=1e-12, atol=0)


def test_per_node_probe_counts(off_pole_cap):
    """
    ``nprobe`` one per node: a list of equal counts is the int, bit for bit,
    in every channel and for the spectrum-aware ratio; in a mixed list each
    node is computed exactly as alone with its own count, a node with
    ``2 l' + 1`` probes being the complete basis (the exact white TT row of
    ``_white_row_pol``).
    """
    from cmbcov.assembly.filtering import spectrum_aware_ratio

    prof = functools.partial(highpass_profile, lx=5.0, shape="sharp")
    nodes, dmax, margin = [9, 30], 4, 12

    def white(nprobe, nodes=nodes, **kw):
        return filtered_sum_rule_ratio(
            off_pole_cap,
            OFF_LW,
            prof,
            nodes,
            dmax,
            channels=CHANNELS,
            nprobe=nprobe,
            seed=2,
            margin=margin,
            nthreads=1,
            **kw,
        )

    a = white(6)
    for same in ([6, 6], (6, 6), np.array([6, 6])):
        b = white(same, lmax_int_floor=[0, 0])
        for ch in CHANNELS:
            assert np.array_equal(a[ch], b[ch]), ch
    mixed = white([19, 6])
    exact9 = white(19, nodes=[9])
    for ch in CHANNELS:
        assert np.array_equal(mixed[ch][0], exact9[ch][0]), ch
        assert np.array_equal(mixed[ch][1], a[ch][1]), ch
        assert not np.array_equal(mixed[ch][0], a[ch][0]), ch
    # the complete-basis node is the exact white TT x TT row
    lmax = 9 + dmax + margin
    lg = gl_minimal_lmax(lmax, OFF_LW)
    w_map = gl_synthesis(off_pole_cap, OFF_LW, lg, nthreads=1)
    one = {"TT": np.ones(lmax + 1)}
    r = _white_row_pol(
        w_map, FourierRingFilter(prof(lg)), lg, lmax, 9, one, ("TT", "TT")
    ) / _white_row_pol(w_map, None, lg, lmax, 9, one, ("TT", "TT"))
    ref = [r[9]] + [0.5 * (r[9 + d] + r[9 - d]) for d in range(1, dmax + 1)]
    np.testing.assert_allclose(mixed["00"][0], ref, rtol=1e-12, atol=0)

    cl, fl = _steep_spectra(120)
    args = ({"aa": cl}, {"aa": fl * cl}, {"t": [("aa", "aa"), ("aa", "aa")]})

    def sa(nprobe, nodes=nodes, **kw):
        return spectrum_aware_ratio(
            off_pole_cap,
            OFF_LW,
            prof,
            nodes,
            dmax,
            *args,
            nprobe=nprobe,
            seed=2,
            margin=margin,
            nthreads=1,
            **kw,
        )["t"]

    s6 = sa(6)
    assert np.array_equal(s6, sa([6, 6], lmax_int_floor=0))
    s_mixed = sa([6, 61])  # node 30: 61 = 2 * 30 + 1, the complete basis
    assert np.array_equal(s_mixed[0], s6[0])
    assert np.array_equal(s_mixed[1], sa(61, nodes=[30])[0])
    assert np.array_equal(s_mixed[1], sa(200, nodes=[30])[0])

    for bad, fragment in (
        ([6], "one per node"),
        ([6, 0], "at least 1"),
        ([6, 2.5], "integers"),
        (True, "integer"),
    ):
        with pytest.raises(ValueError, match=fragment):
            white(bad)
    for bad, fragment in (([0], "one per node"), ([0, -1], "at least 0")):
        with pytest.raises(ValueError, match=fragment):
            white(6, lmax_int_floor=bad)


def test_bandlimit_floor_at_the_cut(off_pole_cap):
    r"""
    The node at the cut (``l' = lx = 8``) with a short internal band-limit
    (``margin = 2``: ``lmax_int = 14 = 1.75 lx``, as the survey's default
    ``1.57 lx`` at its node 300), without and with the floor
    ``ceil(BANDLIMIT_FLOOR lx) = 28``, against the in-test exact white row
    ratio (``_white_row_pol``, complete complex basis) at the band-limit 160
    on its minimal grid.

    Measured (max over ``d <= 4`` of the relative error):

    * exp cut (power 6): no floor -22.4 to -23.1% over ``d`` (all low),
      floor 0.478%; the reference is converged (60, 100 and 160 agree to
      1e-8);
    * sharp cut: no floor -22.4 to -24.9% against the reference at 160 (21
      to 26% low against those at 60 and 100, which differ by up to 5% among
      themselves: the sharp cut's white ratio at the cut keeps moving with
      the band-limit and the grid), floor 1.55% (3.6% against the reference
      at 60).

    And the floor is just a band-limit: ``lmax_int_floor = L`` (any margin
    below it) reproduces the reference at ``L`` to round-off.
    """
    from cmbcov.assembly.filtering import BANDLIMIT_FLOOR

    lx, ellp, dmax = 8.0, 8, 4
    floor = int(np.ceil(BANDLIMIT_FLOOR * lx))
    assert floor == 28

    def ref_row(prof, L):
        lg = gl_minimal_lmax(L, OFF_LW)
        w_map = gl_synthesis(off_pole_cap, OFF_LW, lg, nthreads=1)
        one = {"TT": np.ones(L + 1)}
        r = _white_row_pol(
            w_map, FourierRingFilter(prof(lg)), lg, L, ellp, one, ("TT", "TT")
        ) / _white_row_pol(w_map, None, lg, L, ellp, one, ("TT", "TT"))
        return np.array(
            [r[ellp]] + [0.5 * (r[ellp + d] + r[ellp - d]) for d in range(1, dmax + 1)]
        )

    for shape, tol_floor, low in (("exp", 0.0055, (0.22, 0.24)), ("sharp", 0.02, None)):
        prof = functools.partial(highpass_profile, lx=lx, shape=shape)

        def g(floor, prof=prof):
            return filtered_sum_rule_ratio(
                off_pole_cap,
                OFF_LW,
                prof,
                [ellp],
                dmax,
                nprobe=2 * ellp + 1,
                margin=2,
                lmax_int_floor=floor,
                nthreads=1,
            )["00"][0]

        ref = ref_row(prof, 160)
        no_floor = g(None) / ref - 1
        with_floor = g(floor) / ref - 1
        print(
            f"{shape}: no floor {100 * no_floor.min():+.2f}..{100 * no_floor.max():+.2f}%"
            f", floor {100 * np.abs(with_floor).max():.3f}%"
        )
        assert np.all(no_floor < -0.15)  # the missing scattered power: too low
        if low is not None:
            assert np.all((-no_floor > low[0]) & (-no_floor < low[1]))
        assert np.abs(with_floor).max() < tol_floor
        assert np.abs(with_floor).max() < 0.2 * np.abs(no_floor).min()
        if shape == "exp":
            # the floor is the band-limit: equal to the reference at that one
            np.testing.assert_allclose(g(60), ref_row(prof, 60), rtol=1e-12, atol=0)


# --------------------------------------------------------------------------- #
# 9. Spectrum-aware TT x TT probing
# --------------------------------------------------------------------------- #
def _tt_rows(w_map, F, lg, lmax, ellp, spectra, terms):
    r"""
    Exact rows ``Sigma(l, l')``, ``l = 0 .. lmax``, of TT blocks with
    several frequency pairs, filtered (``F``) or not (``F = None``):

        Sigma_b = 1/(n n') sum_{m m'} sum_{(p, q) in terms[b]} R^p conj R^q,
        R^p = K diag(C^p) K^dagger,  K = A W F S,

    over the complete *complex* basis ``e_{l'm'}``, ``m' = -l' .. l'``, in
    the full-M representation (as ``_white_row_pol``).  Independent of the
    probing code in basis, representation and algebra; shares only the GL
    transforms and the ring filter.
    """
    from cmbcov.harmonics.grid import full_from_pair, pair_from_full

    filt = F if F is not None else (lambda f: f)

    def on_full(op, x):
        r, s = pair_from_full(x, lmax)
        return full_from_pair(op(r), op(s), lmax)

    def adj(a):
        return gl_analysis(
            filt(w_map * gl_synthesis(a, lmax, lg, nthreads=1)), lmax, lg, nthreads=1
        )

    def fwd(a):
        return gl_analysis(
            w_map * filt(gl_synthesis(a, lmax, lg, nthreads=1)), lmax, lg, nthreads=1
        )

    pairs = sorted({p for wick in terms.values() for t in wick for p in t})
    rows = {b: np.zeros(lmax + 1) for b in terms}
    for mp in range(-ellp, ellp + 1):
        e = np.zeros((lmax + 1, 2 * lmax + 1), dtype=complex)
        e[ellp, lmax + mp] = 1.0
        u = on_full(adj, e)
        R = {p: on_full(fwd, spectra[p][: lmax + 1, None] * u) for p in pairs}
        for b, wick in terms.items():
            for p, q in wick:
                rows[b] += np.sum(R[p] * np.conj(R[q]), axis=1).real
    ell = np.arange(lmax + 1)
    return {b: r / ((2 * ell + 1) * (2 * ellp + 1)) for b, r in rows.items()}


def _steep_spectra(size):
    """A steep, non-white 'pre-filter' spectrum and a made-up smooth
    transfer function ``F_L`` (``C_used = F C_pre`` is not ``const C_pre``)."""
    ell = np.arange(size, dtype=float)
    cl_pre = 100.0 / (ell + 2.0) ** 2.5 + 2e-3
    fl = 0.3 + 0.7 * (1.0 - np.exp(-((ell / 8.0) ** 2)))
    return cl_pre, fl


SA_ELLP, SA_DMAX, SA_MARGIN = 9, 4, 12


@pytest.fixture(scope="module")
def sa_setup(off_pole_cap):
    lmax = SA_ELLP + SA_DMAX + SA_MARGIN
    lg = gl_minimal_lmax(lmax, OFF_LW)
    w_map = gl_synthesis(off_pole_cap, OFF_LW, lg, nthreads=1)
    profile = functools.partial(highpass_profile, lx=5.0, shape="sharp")
    return {
        "lmax": lmax,
        "lg": lg,
        "w_map": w_map,
        "profile": profile,
        "F": FourierRingFilter(profile(lg)),
    }


def _sa_node(setup, cl_pre, cl_used, products):
    from cmbcov.assembly.filtering import _node_sums_spectra

    ellp = SA_ELLP
    return _node_sums_spectra(
        setup["w_map"],
        setup["F"],
        ellp,
        setup["lmax"],
        setup["lg"],
        cl_pre,
        cl_used,
        products,
        2 * ellp + 1,
        0,
        1,
    )


def _pm_average(r, ellp, dmax):
    return [r[ellp]] + [0.5 * (r[ellp + d] + r[ellp - d]) for d in range(1, dmax + 1)]


def test_spectrum_aware_exact_single_pair(off_pole_cap, sa_setup):
    r"""
    (a) Exactness, one frequency.  Off-pole cap, sharp cut ``5 sin(theta)``,
    a steep spectrum ``C_pre`` and ``C_used = F_L C_pre``, exact basis at
    ``l' = 9``:

    * unfiltered sums / (n n') = ``exact_covariance_row(grid="gl")`` with
      ``C_used`` (pins the probe algebra and which spectrum is unfiltered);
    * filtered sums / (n n') = the in-test exact filtered row with
      ``C_pre`` (``_white_row_pol``, a general spectrum);
    * ``exact unfiltered row x g_C`` = exact filtered row for
      ``|l - l'| <= dmax`` (the claim of the method);
    * ``spectrum_aware_ratio`` end to end = the +-d average of the ratio of
      the two reference rows.

    Measured (relative to the row maximum, and to the filtered diagonal
    for the model): unfiltered 1.1e-15, filtered 5.2e-16, model 1.5e-15.
    """
    from cmbcov.assembly.filtering import spectrum_aware_ratio
    from cmbcov.exact.covariance import exact_covariance_row

    s = sa_setup
    lmax, ellp, dmax = s["lmax"], SA_ELLP, SA_DMAX
    cl_pre, fl = _steep_spectra(lmax + 1)
    cl_used = fl * cl_pre
    pre, used = {"aa": cl_pre}, {"aa": cl_used}

    exact, count, sums = _sa_node(s, pre, used, [("aa", "aa")])
    assert exact and count == 2 * ellp + 1
    ell = np.arange(lmax + 1)
    norm = 2.0 / ((2 * ell + 1) * (2 * ellp + 1))  # two equal Wick terms
    s_f, s_u = (norm * a for a in sums[("aa", "aa")])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # margin 0 of the row: intended
        row_u = exact_covariance_row(
            off_pole_cap,
            cl_used,
            ellp,
            lmax,
            grid="gl",
            lw=OFF_LW,
            lmax_grid=s["lg"],
            lmax_int=lmax,
            nthreads=1,
        )
    row_f = _white_row_pol(
        s["w_map"], s["F"], s["lg"], lmax, ellp, {"TT": cl_pre}, ("TT", "TT")
    )
    err_u = np.abs(s_u - row_u).max() / np.abs(row_u).max()
    err_f = np.abs(s_f - row_f).max() / np.abs(row_f).max()
    band = slice(ellp - dmax, ellp + dmax + 1)
    model = row_u[band] * (s_f / s_u)[band]
    err_model = np.abs(model - row_f[band]).max() / row_f[ellp]
    print(f"unfiltered {err_u:.1e} filtered {err_f:.1e} model {err_model:.1e}")
    assert max(err_u, err_f, err_model) < 1e-12
    # a genuine effect: the filter changes the block, and C_used is not
    # C_pre times a constant, so the ratio is not rho_white / (F F')
    assert np.all(row_f[band] < 0.9 * row_u[band])

    g = spectrum_aware_ratio(
        off_pole_cap,
        OFF_LW,
        s["profile"],
        [ellp],
        dmax,
        pre,
        used,
        {"blk": [("aa", "aa"), ("aa", "aa")]},
        nprobe=2 * ellp + 1,
        margin=SA_MARGIN,
        oversample=None,  # the minimal grid of the references
        nthreads=1,
    )
    assert list(g) == ["blk"] and g["blk"].shape == (1, dmax + 1)
    np.testing.assert_allclose(
        g["blk"][0], _pm_average(row_f / row_u, ellp, dmax), rtol=1e-12, atol=0
    )


def test_spectrum_aware_exact_two_frequencies(off_pole_cap, sa_setup):
    r"""
    (b) Two frequencies ``a``, ``b`` with different spectra (``C^bb`` a
    different shape, ``C^ab = 0.8 sqrt(C^aa C^bb)``), each with its own
    made-up transfer function.  Blocks with two different Wick terms
    (``ab x ab``: ``aa.bb + ab.ab``; ``aa x ab``: ``aa.ab + ab.aa``) and a
    single repeated one (``aa x bb``: ``2 ab.ab``) against the in-test
    exact rows built directly from the correlators ``R^p``, filtered with
    ``C_pre`` and unfiltered with ``C_used``.

    Measured: sums against the reference rows 3.0e-16 to 7.3e-16, model
    rows (``unfiltered x g_C`` against filtered) 5.2e-16 to 7.8e-16 of the
    diagonal.
    """
    from cmbcov.assembly.filtering import spectrum_aware_ratio

    s = sa_setup
    lmax, ellp, dmax = s["lmax"], SA_ELLP, SA_DMAX
    ell = np.arange(lmax + 1, dtype=float)
    base, fl = _steep_spectra(lmax + 1)
    pre = {
        "aa": base,
        "bb": 1.5 * base * (1.0 + 0.05 * ell),
    }
    pre["ab"] = 0.8 * np.sqrt(pre["aa"] * pre["bb"])
    fls = {"aa": fl, "bb": fl**1.5, "ab": fl**1.25}
    used = {p: fls[p] * pre[p] + (0.01 if p != "ab" else 0.0) for p in pre}
    terms = {
        "abxab": [("aa", "bb"), ("ab", "ab")],
        "aaxab": [("aa", "ab"), ("ab", "aa")],
        "aaxbb": [("ab", "ab"), ("ab", "ab")],
    }
    from cmbcov.assembly.filtering import _product_key

    products = list(
        dict.fromkeys(_product_key(*t) for wick in terms.values() for t in wick)
    )
    exact, _, sums = _sa_node(s, pre, used, products)
    assert exact
    ref_f = _tt_rows(s["w_map"], s["F"], s["lg"], lmax, ellp, pre, terms)
    ref_u = _tt_rows(s["w_map"], None, s["lg"], lmax, ellp, used, terms)
    norm = 1.0 / ((2 * ell + 1) * (2 * ellp + 1))
    band = slice(ellp - dmax, ellp + dmax + 1)
    worst = 0.0
    for b, wick in terms.items():
        s_f = norm * sum(sums[_product_key(*t)][0] for t in wick)
        s_u = norm * sum(sums[_product_key(*t)][1] for t in wick)
        errs = [
            np.abs(s_f - ref_f[b]).max() / np.abs(ref_f[b]).max(),
            np.abs(s_u - ref_u[b]).max() / np.abs(ref_u[b]).max(),
            np.abs(ref_u[b][band] * (s_f / s_u)[band] - ref_f[b][band]).max()
            / abs(ref_f[b][ellp]),
        ]
        print(f"{b}: " + " ".join(f"{e:.1e}" for e in errs))
        worst = max(worst, *errs)
    assert worst < 1e-12
    # the two Wick terms of ab x ab differ, so the block is not one product
    assert not np.allclose(ref_f["abxab"], ref_f["aaxbb"], rtol=1e-3)

    g = spectrum_aware_ratio(
        off_pole_cap,
        OFF_LW,
        s["profile"],
        [ellp],
        dmax,
        pre,
        used,
        terms,
        nprobe=2 * ellp + 1,
        margin=SA_MARGIN,
        oversample=None,  # the minimal grid of the references
        nthreads=1,
    )
    assert list(g) == list(terms)
    for b in terms:
        np.testing.assert_allclose(
            g[b][0], _pm_average(ref_f[b] / ref_u[b], ellp, dmax), rtol=1e-12, atol=0
        )


@pytest.mark.parametrize(
    "profile",
    [
        functools.partial(highpass_profile, lx=0.0, shape="sharp"),
        _theta_independent(np.ones_like),
    ],
)
def test_spectrum_aware_no_filter_same_spectra_gives_one(profile, off_pole_cap):
    """(c) ``h = 1`` and ``C_used = C_pre``: both operators do the same
    arithmetic and ``g`` is 1 bit for bit, exact basis and random probes."""
    from cmbcov.assembly.filtering import spectrum_aware_ratio

    cl, _ = _steep_spectra(200)
    spectra = {"aa": cl, "ab": 0.5 * cl, "bb": 2 * cl}
    terms = {"x": [("aa", "bb"), ("ab", "ab")], "y": [("aa", "aa")] * 2}
    g = spectrum_aware_ratio(
        off_pole_cap,
        OFF_LW,
        profile,
        [5, 30, 1],
        7,
        spectra,
        spectra,
        terms,
        nprobe=12,
        nthreads=1,
    )
    assert list(g) == ["x", "y"]
    for b in g:
        assert g[b].shape == (3, 8)
        assert np.all(g[b] == 1.0), b


def test_spectrum_aware_seed_nodes_cap_and_errors(off_pole_cap, sa_setup):
    """(d) Seed reproducibility, node independence, the spectrum-length
    cap of ``lmax_int``, and the argument checks."""
    from cmbcov.assembly.filtering import _node_sums_spectra, spectrum_aware_ratio

    prof = functools.partial(highpass_profile, lx=8.0, shape="sharp")
    cl, fl = _steep_spectra(120)
    pre, used = {"aa": cl}, {"aa": fl * cl}
    terms = {"t": [("aa", "aa"), ("aa", "aa")]}

    def run(nodes, seed, nprobe=6, **kw):
        return spectrum_aware_ratio(
            off_pole_cap,
            OFF_LW,
            prof,
            list(nodes),
            5,
            pre,
            used,
            terms,
            nprobe=nprobe,
            seed=seed,
            nthreads=1,
            **kw,
        )["t"]

    a = run([40], 3)
    assert np.array_equal(a, run([40], 3))
    assert not np.array_equal(a, run([40], 4))
    both = run([20, 40], 3)
    assert np.array_equal(both[1], a[0])
    assert np.array_equal(both[0], run([20], 3)[0])
    # the white "00" channel is the special case C_pre = C_used = 1
    ones = {"aa": np.ones(120)}
    white = filtered_sum_rule_ratio(
        off_pole_cap, OFF_LW, prof, [40], 5, nprobe=6, seed=3, nthreads=1
    )["00"]
    sa_white = spectrum_aware_ratio(
        off_pole_cap,
        OFF_LW,
        prof,
        [40],
        5,
        ones,
        ones,
        {"w": [("aa", "aa")]},
        nprobe=6,
        seed=3,
        oversample=None,  # the white ratio's minimal grid
        nthreads=1,
    )["w"]
    np.testing.assert_allclose(sa_white, white, rtol=1e-14, atol=0)

    # Spectra shorter than l' + dmax + margin + 1 cap lmax_int for both
    # operators: identical to full-length spectra with the margin that
    # gives that lmax_int.
    ellp, dmax = 30, 5
    short = {"aa": cl[:41]}, {"aa": (fl * cl)[:41]}  # lmax_int capped at 40
    capped = spectrum_aware_ratio(
        off_pole_cap,
        OFF_LW,
        prof,
        [ellp],
        dmax,
        *short,
        terms,
        nprobe=4,
        seed=1,
        nthreads=1,
    )["t"]
    full = spectrum_aware_ratio(
        off_pole_cap,
        OFF_LW,
        prof,
        [ellp],
        dmax,
        pre,
        used,
        terms,
        nprobe=4,
        seed=1,
        margin=40 - ellp - dmax,
        nthreads=1,
    )["t"]
    assert np.array_equal(capped, full)
    # capped below l' + dmax: the missing l' + d use l' - d alone
    tight = {"aa": cl[:33]}, {"aa": (fl * cl)[:33]}  # lmax_int 32 = l' + 2
    g = spectrum_aware_ratio(
        off_pole_cap,
        OFF_LW,
        prof,
        [ellp],
        dmax,
        *tight,
        terms,
        nprobe=4,
        seed=1,
        nthreads=1,
    )["t"][0]
    # the default grid: at least 2 lmax_int, never below the minimal one
    lg = max(gl_minimal_lmax(32, OFF_LW), 64)
    assert lg == 64
    w_map = gl_synthesis(off_pole_cap, OFF_LW, lg, nthreads=1)
    _, _, sums = _node_sums_spectra(
        w_map,
        FourierRingFilter(prof(lg)),
        ellp,
        32,
        lg,
        *tight,
        [("aa", "aa")],
        4,
        1,
        1,
    )
    r = sums[("aa", "aa")][0] / sums[("aa", "aa")][1]
    assert g[1] == pytest.approx(0.5 * (r[ellp + 1] + r[ellp - 1]), rel=1e-14)
    for d in (3, 4, 5):
        assert g[d] == pytest.approx(r[ellp - d], rel=1e-14)

    with pytest.raises(ValueError, match="oversample"):
        spectrum_aware_ratio(
            off_pole_cap, OFF_LW, prof, [10], 2, pre, used, terms, oversample=0.5
        )
    with pytest.raises(ValueError, match="no spectrum for frequency pair 'bb'"):
        spectrum_aware_ratio(
            off_pole_cap, OFF_LW, prof, [10], 2, pre, used, {"t": [("aa", "bb")]}
        )
    with pytest.raises(ValueError, match="below the node"):
        spectrum_aware_ratio(
            off_pole_cap, OFF_LW, prof, [50], 2, short[0], short[1], terms
        )
    with pytest.raises(ValueError, match="terms"):
        spectrum_aware_ratio(off_pole_cap, OFF_LW, prof, [10], 2, pre, used, {})
    with pytest.raises(ValueError, match="terms"):
        spectrum_aware_ratio(
            off_pole_cap, OFF_LW, prof, [10], 2, pre, used, {"t": [("aa",)]}
        )
    with pytest.raises(ValueError, match="finite"):
        spectrum_aware_ratio(
            off_pole_cap, OFF_LW, prof, [10], 2, {"aa": cl * np.nan}, used, terms
        )


def test_pre_filter_noise_and_block_id():
    from cmbcov.assembly.filtering import NOISE_FL_FLOOR, block_id, pre_filter_noise
    from cmbcov.assembly.keys import CovKey

    nl = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    fl = np.array([0.0, NOISE_FL_FLOOR, 0.5, np.nan, 0.8, 0.9])
    out = pre_filter_noise(nl, fl)
    np.testing.assert_array_equal(out, [1.0, 2.0, 6.0, 4.0, 6.25])
    with pytest.raises(ValueError):
        pre_filter_noise(nl, fl[:3])
    key = CovKey(("T", "T", "T", "T"), ("090GHz", "150GHz", "090GHz", "090GHz"))
    assert block_id(key) == "TTxTT_090GHz150GHzx090GHz090GHz"
