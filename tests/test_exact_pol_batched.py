"""
The batched polarised exact GL row against the per-column one.

``exact_covariance_row_pol`` / ``exact_covariance_pol`` compute the columns
``m'`` of a row in chunks, the spherical-harmonic transforms of a chunk done
together as Legendre matrix products and one batched FFT
(:func:`cmbcov.exact._batched_chunk_pol_gl`; for T alone the TT kernel).
The per-column path (``_exact_row_pol_gl(..., batched=False)``, one column at
a time through ducc0 transforms) is the reference.  The two are different
arithmetic, so they are compared with a tolerance and not bit for bit:
``1e-13`` of each block's Wick scale (:func:`wick_scale`, the Cauchy-Schwarz
bound of the block from the auto blocks of the same row, which is also the
size of the products whose rounding makes up the error), well above what is
measured and allowing for another BLAS library.  Measured on the two masks
below (``LMAX`` 32, ``LW`` 48, every spectrum subset with non-zero TB and
EB, rows 0 to 32, with and without the ``m'`` symmetry): at most 2.5e-15 of
the Wick scale, and 1.2e-15 of the row maximum.  (A block's own maximum is
not a usable scale: at l' = 2 the (BB, EB) block of the two-blob mask is
1e-5 of the row, and 1.3e-13 of its own maximum.)

The kernel works in a rotated basis (E, iB) on the coefficients and (Q, iU)
on the rings, where ducc0's spin-2 transforms are real symmetric blocks of
the two spin-weighted Legendre functions; the tests below pin that
convention against ``alm2leg`` and :func:`~cmbcov.grid.gl_synthesis_complex`
directly, then the rows.  As for TT, every row comparison uses masks with
no symmetry axis (``patchy_mask``, ``two_blob_mask`` of
``tests/conftest.py``): on an azimuthally symmetric mask every column
couples ``M = m'`` only and an order-mixing error goes unnoticed.

Also pinned: every spectrum subset (their fields decide which column types
and outputs are computed), rows with ``l' < 2`` (no E or B columns), the
``m' -> -m'`` symmetry, the chunking (one column, uneven, whole), the
internal band-limit and grid size, masks wider and narrower than the
harmonics, held and streamed tables, term selection, the memory plan against
traced allocations, and the public entry points.
"""

import os
import warnings

import numpy as np
import pytest

healpy = pytest.importorskip("healpy")
ducc0 = pytest.importorskip("ducc0")

from conftest import patchy_mask, two_blob_mask  # noqa: E402

from cmbcov import exact  # noqa: E402
from cmbcov.exact import (  # noqa: E402
    _batched_chunk_pol_direct,
    _batched_chunk_pol_gl,
    _batched_row_plan,
    _BatchedGL,
    _cl_matrix,
    _exact_row_pol_gl,
    _kept_mprime,
    _pol_layout,
    _pol_mixing,
    _pol_terms,
    _polarised_batch,
    _spectra_fields,
    _step1_profiles_spin2,
    exact_covariance_pol,
    exact_covariance_row_pol,
)
from cmbcov.grid import (  # noqa: E402
    _gl_geometry,
    gl_legendre_table,
    gl_legendre_table_bytes,
    gl_minimal_lmax,
    gl_north_rings,
    gl_synthesis_complex,
    gl_thetas,
)
from cmbcov.sht import ducc0_map2alm  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

LW = 48
LMAX = 32
#: Batched against per-column, as a fraction of each block's maximum.
TOL = 1e-13
ALL = ("TT", "EE", "BB", "TE", "TB", "EB")
SUBSETS = [ALL, ("TT", "TE", "EE"), ("EE", "BB", "EB"), ("TB",), ("BB",), ("TT",)]


def spectra(lmax):
    """Power laws with every cross-spectrum non-zero (TB and EB included, so
    that the imaginary mixing coefficients of the rotated basis are used)."""
    ell = np.arange(lmax + 1)
    tt = np.zeros(lmax + 1)
    tt[2:] = 1e-3 / (ell[2:] * (ell[2:] + 1)) ** 0.9
    return {
        "TT": tt,
        "EE": 0.1 * tt,
        "BB": 0.02 * tt,
        "TE": 0.15 * tt,
        "TB": 0.03 * tt,
        "EB": 0.01 * tt,
    }


@pytest.fixture(scope="module")
def patchy():
    mask = patchy_mask()
    return mask, ducc0_map2alm(mask, lmax=LW, iter=10)


@pytest.fixture(scope="module")
def patchy_reference(patchy):
    """Per-column rows of all six spectra on ``patchy``, by ``(l', use_symmetry)``."""
    _, mask_alm = patchy
    cache = {}

    def get(ellp, use_symmetry=True):
        key = (ellp, use_symmetry)
        if key not in cache:
            cache[key] = row(
                mask_alm,
                spectra(LMAX),
                ellp,
                ALL,
                batched=False,
                use_symmetry=use_symmetry,
            )
        return cache[key]

    return get


@pytest.fixture(scope="module")
def two_blob():
    mask = two_blob_mask(32)
    return mask, ducc0_map2alm(mask, lmax=LW, iter=10)


def wick_scale(full, s1, s2):
    r"""
    The largest Cauchy-Schwarz bound of the block ``(s1, s2) = (XY, ZW)``
    over ``l``: ``Cov(XY, ZW) = sum R^{XW} conj(R^{YZ}) + R^{XZ}
    conj(R^{YW})`` and ``Cov(XX, WW) = 2 sum |R^{XW}|^2``, so

        |Cov(XY, ZW)_l| <= [sqrt(Cov(XX, WW)_l Cov(YY, ZZ)_l)
                            + sqrt(Cov(XX, ZZ)_l Cov(YY, WW)_l)] / 2 .

    ``full`` is a row with all six spectra (the auto blocks are needed).
    """
    (x, y), (z, w) = s1, s2

    def auto(p, q):
        return np.abs(full[(p + p, q + q)])

    bound = np.sqrt(auto(x, w) * auto(y, z)) + np.sqrt(auto(x, z) * auto(y, w))
    return 0.5 * bound.max()


def assert_blocks_close(got, full, tol=TOL):
    """Every block of ``got`` within ``tol`` of its Wick scale of the
    reference ``full`` (all six spectra); exact zeros where it is zero."""
    for (s1, s2), block in got.items():
        ref = full[(s1, s2)]
        scale = wick_scale(full, s1, s2)
        assert np.abs(ref).max() <= scale * (1 + 1e-12)
        err = np.abs(block - ref).max()
        assert err <= tol * scale, ((s1, s2), err, scale)


#: The two batched kernels: with the Legendre tables held (the table GEMMs of
#: ``_batched_chunk_pol_gl``) and when they would be streamed (the ducc0
#: transforms of ``_batched_chunk_pol_direct``).
KERNELS = ("table", "direct")


def row(mask_alm, cls, ellp, specs, lw=LW, lmax=LMAX, kernel=None, **kw):
    """The row through ``_exact_row_pol_gl``; ``kernel`` forces one of the
    two batched kernels (by the table policy of the shared state)."""
    lmax_i = kw.get("lmax_int") or lmax
    cmat = _cl_matrix(cls, lmax_i)
    kw.setdefault("nthreads", 2)
    if kernel is not None:
        grid = kw.get("lmax_grid") or gl_minimal_lmax(lmax_i, lw)
        fields = _spectra_fields(specs)[1]
        ctx = _polarised_batch(mask_alm, lw, lmax_i, grid, kw["nthreads"], None, fields)
        ctx.hold_table = kernel == "table"
        kw["batch"] = ctx
    return _exact_row_pol_gl(mask_alm, lw, cmat, ellp, lmax, specs, **kw)


def chunked_row(ctx, cls, ellp, specs, parts, use_symmetry=True):
    """The row assembled from the chunk kernel over the partition ``parts``
    of the columns, as :func:`exact._exact_row_pol_gl_batched` does."""
    specs, fields = _spectra_fields(specs)
    cmat = _cl_matrix(cls, ctx.lmax)
    cols, outs = _pol_layout(fields, ellp)
    direct = not ctx.hold_table
    blocks = _pol_terms(specs, cols, (0, 0, 0) if direct else exact._ROT)
    keys = sorted({(k, p, q) for t in blocks.values() for _, k, p, q in t})
    mix = _pol_mixing(cmat, cols, 0 in outs)
    acc = {key: np.zeros(ctx.lmax + 1) for key in keys}
    for mps in parts:
        weights = np.where((mps > 0) & use_symmetry, 2.0, 1.0)
        if direct:
            part = _batched_chunk_pol_direct(
                ctx, cmat, ellp, mps, weights, cols, outs, keys
            )
        else:
            part = _batched_chunk_pol_gl(ctx, mix, ellp, mps, weights, cols, outs, keys)
        for key in keys:
            acc[key] += part[key]
    ell = np.arange(ctx.lmax + 1)
    norm = 1.0 / ((2 * ell + 1) * (2 * ellp + 1))
    return {
        pair: norm * sum((s * acc[(k, p, q)] for s, k, p, q in terms), 0.0 * ell)
        for pair, terms in blocks.items()
    }


# --------------------------------------------------------------------------- #
# Building blocks: ducc0's spin-2 convention
# --------------------------------------------------------------------------- #
def spin2_legs(lmax, ell, m, theta, comp):
    """``alm2leg`` of the unit vector ``comp`` (0 = E, 1 = B) at ``(ell,
    m)``, ``m >= 0``: the (Q, U) ring profiles of order ``m``."""
    alm = np.zeros((2, healpy.Alm.getsize(lmax)), dtype=complex)
    alm[comp, healpy.Alm.getidx(lmax, ell, m)] = 1.0
    return ducc0.sht.alm2leg(
        alm=alm,
        lmax=lmax,
        theta=theta,
        spin=2,
        mval=np.array([m]),
        mstart=np.array([healpy.Alm.getidx(lmax, 0, m)]),
    )[:, :, 0]


def test_spin2_table_is_ducc0s_and_thread_independent():
    """
    ``gl_legendre_table(spin=2)`` returns lambda^+ and lambda^- split by the
    parity of ``L - M`` with ``alm2leg`` of an E unit vector equal to
    ``(lambda^+, -i lambda^-)`` and of a B unit vector to ``(i lambda^-,
    lambda^+)`` (1e-15 absolute), zero for ``L < 2``; the same bits for any
    number of workers and any grouping of the orders; odd ring count.
    """
    lmax, lmax_grid = 40, 70
    ms = np.array([0, 1, 2, 3, 17, 39, 40])
    one = gl_legendre_table(ms, lmax, lmax_grid, nthreads=1, spin=2)
    four = gl_legendre_table(ms, lmax, lmax_grid, nthreads=4, spin=2)
    late = gl_legendre_table(ms[2:], lmax, lmax_grid, nthreads=3, spin=2)
    theta = gl_thetas(lmax_grid)[: gl_north_rings(lmax_grid)]
    assert theta.size == 36
    for i, m in enumerate(ms):
        for j, table in enumerate(one[i]):
            np.testing.assert_array_equal(table, four[i][j])
            if i >= 2:
                np.testing.assert_array_equal(table, late[i - 2][j])
        pe, po, me, mo = one[i]
        n = lmax - m + 1
        assert pe.shape == me.shape == ((n + 1) // 2, theta.size)
        assert po.shape == mo.shape == (n // 2, theta.size)
        for ell in range(m, lmax + 1):
            k, parity = (ell - m) // 2, (ell - m) % 2
            plus, minus = (pe, po)[parity][k], (me, mo)[parity][k]
            e_leg = spin2_legs(lmax, ell, m, theta, 0)
            b_leg = spin2_legs(lmax, ell, m, theta, 1)
            assert np.abs(e_leg[0] - plus).max() < 1e-15
            assert np.abs(e_leg[1] + 1j * minus).max() < 1e-15
            assert np.abs(b_leg[0] - 1j * minus).max() < 1e-15
            assert np.abs(b_leg[1] - plus).max() < 1e-15
            if ell < 2:
                assert not plus.any() and not minus.any()
    assert gl_legendre_table_bytes(ms, lmax, lmax_grid, spin=2) == (
        2 * gl_legendre_table_bytes(ms, lmax, lmax_grid)
    )
    with pytest.raises(ValueError):
        gl_legendre_table(ms, lmax, lmax_grid, spin=1)


@pytest.mark.parametrize("ell, m", [(2, 0), (7, 3), (12, 12), (9, 1)])
def test_spin2_convention_against_complex_synthesis(ell, m):
    """
    The convention of the kernel, pinned against
    :func:`~cmbcov.grid.gl_synthesis_complex` (the per-column path's
    transform) on every ring: the complex (Q, U) map of the full-``M`` unit
    vector at ``(ell, +-m)`` has the single azimuthal order ``+-m`` with
    profiles ``(lambda^+, -i lambda^-)`` for E and ``(i lambda^-,
    lambda^+)`` for B, where ``lambda^+-_{l,-m} = +-(-1)^m lambda^+-_{lm}``,
    and on the southern rings ``lambda^+(pi - theta) = (-1)^(l+m)
    lambda^+(theta)``, ``lambda^-(pi - theta) = -(-1)^(l+m)
    lambda^-(theta)``.
    """
    lmax, lmax_grid = 14, 23
    geom = _gl_geometry(lmax_grid)
    ntheta, nphi = geom["ntheta"], geom["nphi"]
    n_north = gl_north_rings(lmax_grid)
    pe, po, me, mo = gl_legendre_table(np.array([m]), lmax, lmax_grid, spin=2)[0]
    k, parity = (ell - m) // 2, (ell - m) % 2
    plus_n, minus_n = (pe, po)[parity][k], (me, mo)[parity][k]
    mirror = (-1.0) ** (ell + m)
    plus = np.empty(ntheta)
    minus = np.empty(ntheta)
    plus[:n_north], minus[:n_north] = plus_n, minus_n
    plus[n_north:] = mirror * plus_n[: ntheta - n_north][::-1]
    minus[n_north:] = -mirror * minus_n[: ntheta - n_north][::-1]
    for sign in (1, -1) if m else (1,):
        big_m = sign * m
        s_plus = 1.0 if sign > 0 else (-1.0) ** m
        s_minus = 1.0 if sign > 0 else -((-1.0) ** m)
        for comp in (0, 1):
            full = np.zeros((2, lmax + 1, 2 * lmax + 1), dtype=complex)
            full[comp, ell, lmax + big_m] = 1.0
            maps = gl_synthesis_complex(full, lmax, lmax_grid, spin=2)
            modes = np.fft.fft(maps, axis=-1) / nphi
            q, u = modes[0, :, big_m % nphi], modes[1, :, big_m % nphi]
            if comp == 0:
                want_q, want_u = s_plus * plus, -1j * s_minus * minus
            else:
                want_q, want_u = 1j * s_minus * minus, s_plus * plus
            assert np.abs(q - want_q).max() < 1e-15
            assert np.abs(u - want_u).max() < 1e-15
            others = np.delete(modes, big_m % nphi, axis=-1)
            assert np.abs(others).max() < 1e-15


def test_step1_profiles_spin2(patchy):
    """The weighted unit profiles of the E columns (``m'`` of both signs)
    against the FFT of the complex synthesis of the unit vector."""
    _, mask_alm = patchy
    lmax_grid = gl_minimal_lmax(LMAX, LW)
    ctx = _BatchedGL(mask_alm, LW, LMAX, lmax_grid, 2, None, spins=(2,))
    ellp = 11
    mps = np.array([-7, -2, 0, 3, 11])
    plus, minus = _step1_profiles_spin2(ctx, ellp, mps)
    for c, mp in enumerate(mps):
        full = np.zeros((2, LMAX + 1, 2 * LMAX + 1), dtype=complex)
        full[0, ellp, LMAX + mp] = 1.0
        maps = gl_synthesis_complex(full, LMAX, lmax_grid, spin=2)
        modes = np.fft.fft(maps[:, : ctx.n_north], axis=-1) / maps.shape[-1]
        q, u = modes[0, :, mp % maps.shape[-1]], modes[1, :, mp % maps.shape[-1]]
        wbar = ctx.wbar[: ctx.n_north]
        assert np.abs(plus[:, c] - wbar * q.real).max() < 1e-15 * wbar.max()
        assert np.abs(minus[:, c] - wbar * (1j * u).real).max() < 1e-15 * wbar.max()


def test_contraction_phases():
    """
    ``_pol_terms`` in the rotated basis reproduces the original Wick
    contraction: with random complex correlators ``R`` and ``G' = D R
    D^{-1}``, ``D = diag(1, 1, i)``, the signed Re/Im sums of ``G'`` equal
    ``Re [R^{XW} conj R^{YZ} + R^{XZ} conj R^{YW}]`` for all 36 blocks.
    """
    rng = np.random.default_rng(1)
    r = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    d = np.array([1.0, 1.0, 1.0j])
    g = d[:, None] * r * np.conj(d)[None, :]
    for pair, terms in _pol_terms(ALL, [0, 1, 2]).items():
        x, y = ("TEB".index(c) for c in pair[0])
        z, w = ("TEB".index(c) for c in pair[1])
        want = (r[x, w] * np.conj(r[y, z]) + r[x, z] * np.conj(r[y, w])).real
        got = 0.0
        for sign, kind, (a, b), (c, e) in terms:
            s = g[a, b] * np.conj(g[c, e])
            got += sign * (s.real if kind == "re" else s.imag)
        assert abs(got - want) < 1e-14, pair


# --------------------------------------------------------------------------- #
# The row
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("specs", SUBSETS, ids=[",".join(s) for s in SUBSETS])
@pytest.mark.parametrize("ellp", [0, 1, 2, 20, 32])
@pytest.mark.parametrize("use_symmetry", [True, False])
def test_batched_row_equals_per_column_row(
    patchy, patchy_reference, specs, ellp, use_symmetry, kernel
):
    """Every block of every spectrum subset (which decides the column types
    and outputs computed), rows without E and B columns (l' < 2) included,
    through both kernels."""
    _, mask_alm = patchy
    got = row(
        mask_alm, spectra(LMAX), ellp, specs, use_symmetry=use_symmetry, kernel=kernel
    )
    names = _spectra_fields(specs)[0]
    assert set(got) == {(a, b) for a in names for b in names}
    assert_blocks_close(got, patchy_reference(ellp, use_symmetry))


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("ellp", [2, 13, 32])
def test_batched_row_on_the_two_blob_mask(two_blob, ellp, kernel):
    _, mask_alm = two_blob
    cls = spectra(LMAX)
    got = row(mask_alm, cls, ellp, ALL, kernel=kernel)
    ref = row(mask_alm, cls, ellp, ALL, batched=False)
    assert_blocks_close(got, ref)


@pytest.mark.parametrize("kernel", KERNELS)
def test_batched_row_with_internal_margin_and_large_grid(patchy, kernel):
    """``lmax_int > lmax`` and a grid above the minimal one."""
    _, mask_alm = patchy
    lmax_int = LMAX + 20
    cls = spectra(lmax_int)
    for lmax_grid in (None, gl_minimal_lmax(lmax_int, LW) + 13):
        kw = {"lmax_int": lmax_int, "lmax_grid": lmax_grid}
        got = row(mask_alm, cls, 25, ALL, kernel=kernel, **kw)
        ref = row(mask_alm, cls, 25, ALL, batched=False, **kw)
        assert got[("EE", "EE")].shape == (LMAX + 1,)
        assert_blocks_close(got, ref)


@pytest.mark.parametrize("kernel", KERNELS)
def test_mask_wider_than_the_harmonics(patchy, kernel):
    """``lw > 2 lmax``: the mask's modes coincide on the chunk's short FFT."""
    _, mask_alm = patchy
    lmax = 10
    cls = spectra(lmax)
    for ellp in (3, 10):
        got = row(mask_alm, cls, ellp, ALL, lmax=lmax, kernel=kernel)
        ref = row(mask_alm, cls, ellp, ALL, lmax=lmax, batched=False)
        assert_blocks_close(got, ref)


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("use_symmetry", [True, False])
def test_mask_narrower_than_the_harmonics(patchy, use_symmetry, kernel):
    """
    ``lw << lmax``: the band edges of the four step-1 legs of the E columns
    (both parity groups, the lambda^- group with the mirror rule swapped),
    in whole rows and in chunks of 5 columns, where orders far from a
    chunk's ``m'`` leave whole groups outside the band.
    """
    mask, _ = patchy
    lw = 6
    mask_alm = ducc0_map2alm(mask, lmax=lw, iter=10)
    cls = spectra(LMAX)
    fields = _spectra_fields(ALL)[1]
    ctx = _polarised_batch(
        mask_alm, lw, LMAX, gl_minimal_lmax(LMAX, lw), 2, None, fields
    )
    ctx.hold_table = kernel == "table"
    for ellp in (9, 32):
        kw = {"lw": lw, "use_symmetry": use_symmetry}
        got = row(mask_alm, cls, ellp, ALL, kernel=kernel, **kw)
        ref = row(mask_alm, cls, ellp, ALL, batched=False, **kw)
        assert_blocks_close(got, ref)
        mps = np.arange(0 if use_symmetry else -ellp, ellp + 1)
        parts = [mps[i : i + 5] for i in range(0, mps.size, 5)]
        chunked = chunked_row(ctx, cls, ellp, ALL, parts, use_symmetry)
        assert_blocks_close(chunked, ref)


#: Mask band-limit below ``LMAX``: with ``LW`` 48 > ``LMAX`` 32 the order bands
#: of the batched kernel cover every order and their edges are never reached.
LW_NARROW = 10


@pytest.fixture(scope="module", params=["patchy", "two_blob"])
def narrow(request):
    """Both asymmetric masks truncated at ``LW_NARROW``, with a cache of
    per-column rows of all six spectra by ``(l', use_symmetry)``."""
    mask = patchy_mask() if request.param == "patchy" else two_blob_mask(32)
    mask_alm = ducc0_map2alm(mask, lmax=LW_NARROW, iter=10)
    cache = {}

    def reference(ellp, use_symmetry):
        key = (ellp, use_symmetry)
        if key not in cache:
            cache[key] = row(
                mask_alm,
                spectra(LMAX),
                ellp,
                ALL,
                lw=LW_NARROW,
                batched=False,
                use_symmetry=use_symmetry,
            )
        return cache[key]

    return mask_alm, reference


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("specs", SUBSETS[:3], ids=[",".join(s) for s in SUBSETS[:3]])
@pytest.mark.parametrize("ellp", [5, 20, 32])
@pytest.mark.parametrize("use_symmetry", [True, False])
def test_band_edges_with_a_narrow_mask(narrow, specs, ellp, use_symmetry, kernel):
    """
    ``lw < lmax`` on both asymmetric masks: the orders ``|M - m'| <= lw``
    of step 1 and ``<= 2 lw`` of step 3 are strict subsets, so an error at a
    band edge changes the row (an off-by-one planted in either band fails
    here, and passes the tests at ``LW`` 48).
    """
    mask_alm, reference = narrow
    got = row(
        mask_alm,
        spectra(LMAX),
        ellp,
        specs,
        lw=LW_NARROW,
        use_symmetry=use_symmetry,
        kernel=kernel,
    )
    assert_blocks_close(got, reference(ellp, use_symmetry))


def test_sub_minimal_grid_takes_the_per_column_path(patchy):
    """An aliasing grid is the per-column path's business: bit-identical."""
    _, mask_alm = patchy
    cls = spectra(LMAX)
    lmax_grid = gl_minimal_lmax(LMAX, LW) - 3
    got = row(mask_alm, cls, 20, ALL, lmax_grid=lmax_grid)
    ref = row(mask_alm, cls, 20, ALL, lmax_grid=lmax_grid, batched=False)
    for key in ref:
        np.testing.assert_array_equal(got[key], ref[key])


@pytest.mark.parametrize(
    "sizes",
    [(1,) * 21, (7, 7, 7), (5, 11, 5), (21,), (3, 17, 21)],
    ids=["ones", "even", "uneven", "whole", "negative"],
)
@pytest.mark.parametrize("kernel", KERNELS)
def test_chunking_does_not_matter(patchy, patchy_reference, sizes, kernel):
    """
    The chunk kernel over any partition of the columns sums to the
    per-column row: chunks of one column, chunks that do not divide the 21
    columns, one chunk, and (last case) the 41 columns ``m' = -20..20``
    without the ``m'`` symmetry.
    """
    _, mask_alm = patchy
    cls = spectra(LMAX)
    ellp = 20
    lmax_grid = gl_minimal_lmax(LMAX, LW)
    ctx = _polarised_batch(mask_alm, LW, LMAX, lmax_grid, 2, None, "TEB")
    ctx.hold_table = kernel == "table"
    use_symmetry = sum(sizes) == ellp + 1
    mps = np.arange(0 if use_symmetry else -ellp, ellp + 1)
    assert sum(sizes) == mps.size
    parts = np.split(mps, np.cumsum(sizes)[:-1])
    got = chunked_row(ctx, cls, ellp, ALL, parts, use_symmetry)
    assert_blocks_close(got, patchy_reference(ellp, use_symmetry))


def test_held_and_streamed_tables_give_the_same_bits(patchy, monkeypatch):
    """
    The Legendre tables held whole and those generated block by block into
    one reused buffer, up to a window's top degree, are the same numbers,
    spin 0 and spin 2 (the TT kernel streams them; a polarised row that
    cannot hold its tables uses the table-free kernel instead).
    """
    _, mask_alm = patchy
    lmax_grid = gl_minimal_lmax(LMAX, LW)
    held = _polarised_batch(mask_alm, LW, LMAX, lmax_grid, 3, 1.0, "TEB")
    streamed = _polarised_batch(mask_alm, LW, LMAX, lmax_grid, 2, 1.0, "TEB")
    streamed.hold_table = False
    monkeypatch.setattr(exact, "_TABLE_BLOCK_BYTES", 2**14)  # several blocks
    for hi in (LMAX, 20):
        orders = list(range(0, hi + 1, 3))
        # a streamed block is overwritten by the next: compare as they come
        for (m1, t1), (m2, t2) in zip(
            held.tables(orders, hi), streamed.tables(orders, hi)
        ):
            assert m1 == m2
            for k, (x, y) in enumerate(zip(t1[0] + t1[1], t2[0] + t2[1])):
                assert x.shape[0] == (hi - m1 + 2 - k % 2) // 2
                np.testing.assert_array_equal(x, y)
    assert not streamed._table and sorted(held._table) == list(range(0, LMAX + 1, 3))


@pytest.mark.parametrize("hold_table", [True, False])
def test_rows_are_bit_identical_across_thread_counts(patchy, monkeypatch, hold_table):
    """
    A polarised row is the same to the bit for 1, 2, 3 and 8 threads, with
    several chunks and, with a small ring block, several FFT calls per
    worker of the ring stage (nine profiles per ``m'``: before the ring
    blocks were fixed, the grouping of rings into FFT calls followed the
    split over workers and moved the last bit).
    """
    _, mask_alm = patchy
    monkeypatch.setattr(exact, "_FFT_BLOCK_BYTES", 64 * 2**10)
    lmax = 64
    cmat = _cl_matrix(spectra(lmax), lmax)
    lmax_grid = gl_minimal_lmax(lmax, LW)
    rows, chunks = [], []
    for nthreads in (1, 2, 3, 8):
        ctx = _polarised_batch(mask_alm, LW, lmax, lmax_grid, nthreads, 0.03, "TEB")
        ctx.hold_table = hold_table
        layout, _ = pol_layout(ALL, 60, direct=not hold_table)
        chunks.append(len(ctx.plan(range(61), layout=layout)["chunks"]))
        rows.append(
            _exact_row_pol_gl(
                mask_alm, LW, cmat, 60, lmax, ALL, nthreads=nthreads, batch=ctx
            )
        )
    assert chunks[0] > 1 and len(set(chunks)) == 1
    for row in rows[1:]:
        for key in row:
            np.testing.assert_array_equal(row[key], rows[0][key])


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("which", ["cap", "two_blob"])
def test_term_selection_skips_the_same_columns(monkeypatch, which, kernel):
    """
    With ``term_selection`` the batched row computes exactly the ``m'`` of
    ``_kept_mprime`` (spin-0 and spin-2 mode powers united) and agrees with
    the per-column row under the same selection, through the public entry
    point too.  The compact masks of ``tests/test_exact_batched.py``: the
    baseline cap (symmetric in its pole frame) and the two-blob mask (not).
    """
    if which == "cap":
        mask = healpy.read_map(os.path.join(DATA_DIR, "baseline_mask.fits"))
    else:
        mask = two_blob_mask(32)
    cls = spectra(LMAX)
    tol = 1e-3
    mask_alm, lw = exact._mask_alm_for_gl(mask, None, LW)
    mask_alm = exact._pole_frame_alm(mask, mask_alm, lw)
    seen = []
    name = "_batched_chunk_pol_gl" if kernel == "table" else "_batched_chunk_pol_direct"
    chunk = getattr(exact, name)

    def spy(ctx, mix, ellp, mps, *args):
        seen.extend(int(m) for m in mps)
        return chunk(ctx, mix, ellp, mps, *args)

    monkeypatch.setattr(exact, name, spy)
    ellp = 30
    kw = {"term_selection": tol}
    got = row(mask_alm, cls, ellp, ALL, lw=lw, kernel=kernel, **kw)
    keep = _kept_mprime(mask_alm, lw, ellp, tol, (0, 2), 2)
    kept = [m for m in range(ellp + 1) if keep[m + ellp]]
    assert seen == kept and len(kept) < ellp + 1
    ref = row(mask_alm, cls, ellp, ALL, lw=lw, batched=False, **kw)
    assert_blocks_close(got, ref)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # short margin
        public = exact_covariance_row_pol(
            mask, cls, ellp, LMAX, spectra=ALL, lw=LW, term_selection=tol
        )
    assert_blocks_close(public, ref)


# --------------------------------------------------------------------------- #
# The memory plan
# --------------------------------------------------------------------------- #
def pol_layout(specs, ellp, direct=False):
    specs, fields = _spectra_fields(specs)
    cols, outs = _pol_layout(fields, ellp)
    rot = (0, 0, 0) if direct else exact._ROT
    keys = {
        (k, p, q) for t in _pol_terms(specs, cols, rot).values() for _, k, p, q in t
    }
    layout = (0 in outs, len(cols), len(outs), len(keys), direct)
    return layout, exact._polarised_spins(fields)


@pytest.mark.parametrize("lmax,lw", [(32, 48), (500, 384), (2000, 384)])
@pytest.mark.parametrize("specs", [ALL, ("EE", "BB")], ids=["all", "EE,BB"])
def test_memory_plan_within_budget(lmax, lw, specs):
    """
    Every chunk of the plan fits: ``total`` (ring modes + tables + the
    largest chunk) is within the budget, the chunks cover the columns once
    and in order, and the tables are held exactly when they fit in half the
    budget.  Plans only -- nothing is allocated.
    """
    lmax_grid = gl_minimal_lmax(lmax, lw)
    mps = np.arange(lmax + 1)
    _, spins = pol_layout(specs, lmax)
    full_table = sum(gl_legendre_table_bytes(mps, lmax, lmax_grid, s) for s in spins)
    for gb in (2.0, 8.0, 32.0):
        budget = int(gb * 1024**3)
        # the table-free kernel when the tables do not fit in half the budget
        layout, _ = pol_layout(specs, lmax, direct=full_table > budget // 2)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            plan = _batched_row_plan(
                mps, lmax, lw, lmax_grid, budget, spins=spins, layout=layout
            )
        assert plan["total"] <= budget
        assert plan["hold_table"] == (full_table <= budget // 2)
        np.testing.assert_array_equal(np.concatenate(plan["chunks"]), mps)


@pytest.mark.parametrize("hold_table", [True, False])
@pytest.mark.parametrize("specs", [ALL, ("EE", "BB")], ids=["all", "EE,BB"])
def test_allocations_stay_within_the_plan(patchy, hold_table, specs):
    """
    What a batched polarised row allocates (numpy allocations, traced with
    ``tracemalloc`` from before the ring modes are built to the end of the
    row) stays within the plan's ``total``, for held and streamed tables and
    a budget that forces several chunks.  Measured: 0.80-0.91 of the plan.
    """
    import tracemalloc

    _, mask_alm = patchy
    lmax = 64
    cls = spectra(lmax)
    specs, fields = _spectra_fields(specs)
    cmat = _cl_matrix(cls, lmax)
    lmax_grid = gl_minimal_lmax(lmax, LW)
    layout, _ = pol_layout(specs, lmax, direct=not hold_table)
    budget_gb = 0.05
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        ctx = _polarised_batch(mask_alm, LW, lmax, lmax_grid, 2, budget_gb, fields)
        ctx.hold_table = hold_table
        plan = ctx.plan(range(lmax + 1), layout=layout)
        exact._exact_row_pol_gl_batched(
            ctx, cmat, lmax, range(lmax + 1), True, specs, fields
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(plan["chunks"]) > 1
    assert plan["total"] <= budget_gb * 1024**3
    assert peak <= plan["total"], (peak, plan)


def test_memory_plan_floor_warns():
    """A budget below one ``m'`` per chunk warns once and still covers all."""
    lmax, lw = 500, 384
    lmax_grid = gl_minimal_lmax(lmax, lw)
    layout, spins = pol_layout(ALL, lmax)
    exact._WARNED_MEMORY_FLOOR.clear()
    kw = {"spins": spins, "layout": layout}
    with pytest.warns(UserWarning, match="below the floor"):
        plan = _batched_row_plan(range(lmax + 1), lmax, lw, lmax_grid, 2**28, **kw)
    assert all(c.size == 1 for c in plan["chunks"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _batched_row_plan(range(lmax + 1), lmax, lw, lmax_grid, 2**28, **kw)


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #
def test_public_entry_points_take_the_budget(patchy):
    mask, mask_alm = patchy
    lmax_int = LMAX + 18  # the mask's coupling width is 18
    cls = spectra(lmax_int)
    rows = (1, 20, 32)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # not below the floor
        matrix = exact_covariance_pol(
            mask_alm,
            cls,
            LMAX,
            spectra=ALL,
            lmax_int=lmax_int,
            rows=rows,
            nthreads=2,
            max_memory_gb=0.05,
        )
    for ellp in rows:
        ref = row(mask_alm, cls, ellp, ALL, lmax_int=lmax_int, batched=False)
        assert_blocks_close({k: v[:, ellp] for k, v in matrix.items()}, ref)
        one = exact_covariance_row_pol(
            mask_alm, cls, ellp, LMAX, spectra=ALL, lmax_int=lmax_int, nthreads=2
        )
        assert_blocks_close(one, ref)
    with pytest.raises(ValueError, match="grid='gl' only"):
        exact_covariance_row_pol(
            mask,
            cls,
            5,
            LMAX,
            spectra=("TT",),
            grid="healpix",
            nside=32,
            max_memory_gb=1.0,
        )
    with pytest.raises(ValueError, match="grid='gl' only"):
        exact_covariance_pol(
            mask,
            cls,
            LMAX,
            spectra=("TT",),
            grid="healpix",
            nside=32,
            rows=(5,),
            max_memory_gb=1.0,
        )
    with pytest.raises(ValueError, match="positive"):
        exact_covariance_row_pol(mask_alm, cls, 5, LMAX, max_memory_gb=0)
    cmat = _cl_matrix(cls, LMAX)
    # built for E and B only: no spin-0 table for the T column
    other = _polarised_batch(
        mask_alm, LW, LMAX, gl_minimal_lmax(LMAX, LW), 2, None, "EB"
    )
    assert other.spins == (2,)
    with pytest.raises(ValueError, match="other spectra"):
        _exact_row_pol_gl(mask_alm, LW, cmat, 5, LMAX, ALL, batch=other)
    with pytest.raises(ValueError, match="another lw"):
        _exact_row_pol_gl(
            mask_alm, LW, cmat, 5, LMAX, ("EE", "BB"), lmax_grid=90, batch=other
        )
