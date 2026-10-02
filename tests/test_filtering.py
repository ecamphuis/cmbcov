"""
Filter-and-bin map-making: the filtered sum rule ratio of ``cmbcov.filtering``.

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

Masks are built analytically on a GL grid and band-limited with
``gl_analysis``; nothing is read from disk.
"""

import functools
import hashlib
import json
import os
import warnings

import numpy as np
import pytest

healpy = pytest.importorskip("healpy")

from cmbcov.exact import _abs2_sum_over_m, _unit_pair_alm  # noqa: E402
from cmbcov.filtering import (  # noqa: E402
    BLOCK_CHANNEL,
    CACHE_SUBDIR,
    CHANNELS,
    FourierRingFilter,
    cached_sum_rule_ratio,
    channels_for,
    default_nodes,
    filtered_sum_rule_ratio,
    highpass_profile,
    rho_matrix,
    transfer_correction,
)
from cmbcov.grid import (  # noqa: E402
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
    implementation, independent of :mod:`cmbcov.filtering`); ``F = None`` is
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


def _exact_rows(op, cl, rows):
    """``Cov(C~_l, C~_l')`` for every ``l`` and each ``l'`` in ``rows``."""
    lmax = op.lmax
    ell_alm = healpy.Alm.getlm(lmax)[0]
    cl_alm = cl[ell_alm]
    ell = np.arange(lmax + 1)
    out = np.zeros((len(rows), lmax + 1))
    for i, lp in enumerate(rows):
        for mp in range(lp + 1):
            r, s, has_s = _unit_pair_alm(lmax, lp, mp)
            ra = op.apply(op.adjoint_unit(r, mp) * cl_alm)
            sa = op.apply(op.adjoint_unit(s, mp) * cl_alm) if has_s else None
            out[i] += (1.0 if mp == 0 else 2.0) * _abs2_sum_over_m(
                ra, sa, ell_alm, lmax
            )
        out[i] *= 2.0 / ((2 * ell + 1) * (2 * lp + 1))
    return out


def _coupling(op):
    """``M[l, L] = 1/(2l+1) sum_{m M} |K_{lm,LM}|^2``: ``<C~_l> = sum_L M C_L``.

    Row ``l`` from ``K^dagger e_{lm}``, whose ``(L, M)`` entry is
    ``conj K_{lm,LM}``."""
    lmax = op.lmax
    ell_alm = healpy.Alm.getlm(lmax)[0]
    out = np.zeros((lmax + 1, lmax + 1))
    for ell in range(lmax + 1):
        for m in range(ell + 1):
            r, s, has_s = _unit_pair_alm(lmax, ell, m)
            ra = op.adjoint_unit(r, m)
            sa = op.adjoint_unit(s, m) if has_s else None
            out[ell] += (1.0 if m == 0 else 2.0) * _abs2_sum_over_m(
                ra, sa, ell_alm, lmax
            )
        out[ell] /= 2 * ell + 1
    return out


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

    from cmbcov.covariance import Cov

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
    from cmbcov.grid import full_from_pair, pair_from_full

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


@pytest.mark.parametrize("shape", ["sharp", "exp"])
def test_channel_probes_are_the_exact_white_blocks(shape, off_pole_cap):
    r"""
    Complete-basis probes of every channel against exact white-spectrum rows
    (module docstring of ``cmbcov.filtering``, "Polarisation channels"):
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
    from cmbcov.exact import exact_covariance_row_pol
    from cmbcov.filtering import _node_sums

    ellp, dmax, margin = 9, 4, 12
    lmax = ellp + dmax + margin
    lg = gl_minimal_lmax(lmax, OFF_LW)
    w_map = gl_synthesis(off_pole_cap, OFF_LW, lg, nthreads=1)
    profile = functools.partial(highpass_profile, lx=5.0, shape=shape)
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
