"""
The batched exact GL row against the per-column one.

``exact_covariance_row(grid="gl")`` computes the columns ``m'`` of a row in
chunks, the spherical-harmonic transforms of a chunk done together as
Legendre matrix products and one batched FFT
(:func:`cmbcov.exact._batched_chunk_gl`).  The per-column path
(``_exact_row_gl(..., batched=False)``, one column at a time through ducc0
transforms) is the reference.  The two are different arithmetic, so they are
compared with a tolerance -- ``1e-13`` of the row maximum, far enough above
what is measured to allow for another BLAS library (Linux/OpenBLAS not
measured) -- and not bit for bit.  Measured on the two-patch mask below
(``LMAX`` 32, ``LW`` 48, rows 0, 1, 7, 20, 32): 1e-16 to 1.1e-15 of the
maximum, at most 5e-15 relative element by element, while the per-column
row itself moves by 8e-16 to 3e-14 of its maximum when the grid is
enlarged by 37, so the batched row is inside the round-off of the
reference.

The baseline mask of ``tests/data`` is an apodised polar cap, azimuthally
symmetric to 1e-18: on it every column couples ``M = m'`` only, and a
band-edge error in the batched legs goes unnoticed (checked: an off-by-one
in the step-1 band passed the row and chunking tests on the cap, and fails
ten of them on the two-patch mask).  Hence ``patchy_mask`` (shared, in
``tests/conftest.py``).

Also pinned: the chunking does not matter (chunks of one column, chunks that
do not divide the column count, one chunk); a held and a streamed Legendre
table give the same bits; term selection skips the same ``m'``; the memory
plan stays within its budget and warns below its floor; and the building
blocks (Legendre table, ring modes) against ducc0 directly.
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
    _batched_chunk_gl,
    _batched_row_plan,
    _BatchedGL,
    _exact_row_gl,
    _kept_mprime,
    exact_covariance,
    exact_covariance_row,
)
from cmbcov.grid import (  # noqa: E402
    gl_legendre_table,
    gl_legendre_table_bytes,
    gl_legendre_table_capacity,
    gl_mask_modes,
    gl_minimal_lmax,
    gl_north_rings,
    gl_ring_modes,
    gl_shape,
    gl_thetas,
)
from cmbcov.sht import ducc0_map2alm  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

LW = 48
LMAX = 32
#: Batched against per-column, as a fraction of the row maximum.
TOL = 1e-13


def red_cl(lmax):
    return 1000.0 / (np.arange(lmax + 1) + 1.0) ** 2


@pytest.fixture(scope="module")
def patchy():
    mask = patchy_mask()
    return mask, ducc0_map2alm(mask, lmax=LW, iter=10)


def test_patchy_mask_has_no_symmetry_axis(patchy):
    """Every order ``|k| <= 12`` of the test mask carries >= 1e-3 of the
    largest ring mode (the baseline cap alone: 1e-18 outside ``k = 0``)."""
    _, mask_alm = patchy
    modes = gl_ring_modes(mask_alm, LW, gl_minimal_lmax(LMAX, LW))
    power = np.abs(modes).max(axis=0) / np.abs(modes).max()
    assert power[LW - 12 : LW + 13].min() > 1e-3


def assert_rows_close(got, ref, tol=TOL):
    scale = np.abs(ref).max()
    assert np.abs(got - ref).max() <= tol * scale, np.abs(got - ref).max() / scale


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def test_legendre_table_is_ducc0s_and_thread_independent():
    """
    ``gl_legendre_table`` reads lambda_{LM} off ``leg2alm`` two rings at a
    time; the values must be the Legendre functions ``alm2leg`` evaluates
    (1e-15 absolute, the functions are O(1)), split by the parity of
    ``L - M``, and the same bits for any number of workers and any grouping
    of the orders, since each ring pair is one single-threaded call.  Orders
    need not start at 0 (the block of a streamed table).  Odd ring count
    (grid 70: 71 rings, 36 northern) exercises the padding ring.
    """
    lmax, lmax_grid = 40, 70
    ms = np.array([0, 3, 4, 17, 39, 40])
    one = gl_legendre_table(ms, lmax, lmax_grid, nthreads=1)
    four = gl_legendre_table(ms, lmax, lmax_grid, nthreads=4)
    late = gl_legendre_table(ms[2:], lmax, lmax_grid, nthreads=3)
    theta = gl_thetas(lmax_grid)[: gl_north_rings(lmax_grid)]
    assert theta.size == 36
    for i, (m, (even, odd)) in enumerate(zip(ms, one)):
        for j, table in enumerate((even, odd)):
            np.testing.assert_array_equal(table, four[i][j])
            if i >= 2:
                np.testing.assert_array_equal(table, late[i - 2][j])
            assert table.flags.c_contiguous or table.strides[1] == 8
        n = lmax - m + 1
        assert even.shape == ((n + 1) // 2, theta.size)
        assert odd.shape == (n // 2, theta.size)
        for ell in range(m, lmax + 1):
            alm = np.zeros((1, healpy.Alm.getsize(lmax)), dtype=complex)
            alm[0, healpy.Alm.getidx(lmax, ell, m)] = 1.0
            ref = ducc0.sht.alm2leg(
                alm=alm,
                lmax=lmax,
                theta=theta,
                spin=0,
                mval=np.array([m]),
                mstart=np.array([healpy.Alm.getidx(lmax, 0, m)]),
            )[0, :, 0].real
            got = (even, odd)[(ell - m) % 2][(ell - m) // 2]
            assert np.abs(got - ref).max() < 1e-15
    assert gl_legendre_table_bytes(ms, lmax, lmax_grid) == (
        np.sum(lmax - ms + 1) * ((theta.size + 1) // 2) * 16
    )
    with pytest.raises(ValueError):
        gl_legendre_table(np.array([3, 3]), lmax, lmax_grid)
    with pytest.raises(ValueError):
        gl_legendre_table(np.array([41]), lmax, lmax_grid)


def test_ring_modes_match_the_fft_of_the_map(patchy):
    """``gl_ring_modes`` (Legendre stage only) against ``gl_mask_modes`` (FFT
    of the synthesised map) on a grid where the latter does not alias."""
    _, mask_alm = patchy
    lmax_grid = gl_minimal_lmax(LMAX, LW)
    modes = gl_ring_modes(mask_alm, LW, lmax_grid)
    fft = gl_mask_modes(mask_alm, LW, lmax_grid)
    ks = np.arange(-LW, LW + 1)
    assert modes.shape == (gl_shape(lmax_grid)[0], 2 * LW + 1)
    assert np.abs(modes - fft[:, ks % fft.shape[1]]).max() < 1e-15
    assert np.all(modes[:, LW].imag == 0)


# --------------------------------------------------------------------------- #
# The row
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ellp", [0, 1, 7, 20, 32])
@pytest.mark.parametrize("use_symmetry", [True, False])
def test_batched_row_equals_per_column_row(patchy, ellp, use_symmetry):
    _, mask_alm = patchy
    cl = red_cl(LMAX)
    kw = {"nthreads": 2, "use_symmetry": use_symmetry}
    got = _exact_row_gl(mask_alm, LW, cl, ellp, LMAX, **kw)
    ref = _exact_row_gl(mask_alm, LW, cl, ellp, LMAX, batched=False, **kw)
    assert_rows_close(got, ref)


def test_batched_row_with_internal_margin_and_large_grid(patchy):
    """``lmax_int > lmax`` and a grid above the minimal one."""
    _, mask_alm = patchy
    lmax_int = LMAX + 20
    cl = red_cl(lmax_int)
    for lmax_grid in (None, gl_minimal_lmax(lmax_int, LW) + 13):
        kw = {"nthreads": 2, "lmax_int": lmax_int, "lmax_grid": lmax_grid}
        got = _exact_row_gl(mask_alm, LW, cl, 25, LMAX, **kw)
        ref = _exact_row_gl(mask_alm, LW, cl, 25, LMAX, batched=False, **kw)
        assert got.shape == (LMAX + 1,)
        assert_rows_close(got, ref)


def test_mask_wider_than_the_harmonics(patchy):
    """
    ``lw > 2 lmax``: the chunk's FFT length ``2 lmax + lw + 1`` is shorter
    than the mask's ``2 lw + 1`` modes, which then coincide on the ring
    samples (and are added, as sampling the mask does).
    """
    _, mask_alm = patchy
    lmax = 10
    cl = red_cl(lmax)
    for ellp in (3, 10):
        got = _exact_row_gl(mask_alm, LW, cl, ellp, lmax, nthreads=2)
        ref = _exact_row_gl(mask_alm, LW, cl, ellp, lmax, nthreads=2, batched=False)
        assert_rows_close(got, ref)


@pytest.mark.parametrize("use_symmetry", [True, False])
def test_mask_narrower_than_the_harmonics(patchy, use_symmetry):
    """
    ``lw << lmax``: for most orders only part of a chunk's columns are within
    ``|M - m'| <= lw``, which exercises the band edges of the step-1 legs
    (both parity groups, consecutive and full ``m'`` ranges) and a short FFT.
    """
    mask, _ = patchy
    lw = 6
    mask_alm = ducc0_map2alm(mask, lmax=lw, iter=10)
    cl = red_cl(LMAX)
    batch = _BatchedGL(mask_alm, lw, LMAX, gl_minimal_lmax(LMAX, lw), 2, None)
    ell = np.arange(LMAX + 1)
    for ellp in (9, 32):
        kw = {"nthreads": 2, "use_symmetry": use_symmetry}
        got = _exact_row_gl(mask_alm, lw, cl, ellp, LMAX, **kw)
        ref = _exact_row_gl(mask_alm, lw, cl, ellp, LMAX, batched=False, **kw)
        assert_rows_close(got, ref)
        # chunks of 5 columns: orders far from a chunk's m' leave a whole
        # parity group outside the band
        mps = np.arange(0 if use_symmetry else -ellp, ellp + 1)
        weights = np.where((mps > 0) & use_symmetry, 2.0, 1.0)
        total = sum(
            _batched_chunk_gl(batch, cl, ellp, mps[i : i + 5], weights[i : i + 5])
            for i in range(0, mps.size, 5)
        )
        chunked = 2.0 / ((2 * ell + 1) * (2 * ellp + 1)) * total
        assert_rows_close(chunked, ref)


@pytest.mark.parametrize("which", ["patchy", "two_blob"])
@pytest.mark.parametrize("ellp", [5, 21, 32])
def test_windows_and_bands_with_a_narrow_mask(which, ellp):
    """
    ``lw = 10 < lmax``: the orders ``|M - m'| <= lw`` (step 1) and ``<=
    2 lw`` (step 3) and the degrees ``|L - l'| <= lw`` and ``<= 2 lw`` the
    batched row restricts itself to are strict subsets, on both asymmetric
    masks, so an error at any edge changes the row.
    """
    mask = patchy_mask() if which == "patchy" else two_blob_mask(32)
    lw = 10
    mask_alm = ducc0_map2alm(mask, lmax=lw, iter=10)
    cl = red_cl(LMAX)
    for use_symmetry in (True, False):
        kw = {"nthreads": 2, "use_symmetry": use_symmetry}
        got = _exact_row_gl(mask_alm, lw, cl, ellp, LMAX, **kw)
        ref = _exact_row_gl(mask_alm, lw, cl, ellp, LMAX, batched=False, **kw)
        assert_rows_close(got, ref)


def test_sub_minimal_grid_takes_the_per_column_path(patchy):
    """An aliasing grid is the per-column path's business: bit-identical."""
    _, mask_alm = patchy
    cl = red_cl(LMAX)
    lmax_grid = gl_minimal_lmax(LMAX, LW) - 3
    kw = {"nthreads": 2, "lmax_grid": lmax_grid}
    got = _exact_row_gl(mask_alm, LW, cl, 20, LMAX, **kw)
    ref = _exact_row_gl(mask_alm, LW, cl, 20, LMAX, batched=False, **kw)
    np.testing.assert_array_equal(got, ref)


@pytest.mark.parametrize("sizes", [(1,) * 21, (7, 7, 7), (5, 11, 5), (21,)])
def test_chunking_does_not_matter(patchy, sizes):
    """
    The chunk kernel over any partition of the columns sums to the
    per-column row: chunks of one column, chunks that do not divide the 21
    columns, one chunk.
    """
    _, mask_alm = patchy
    cl = red_cl(LMAX)
    ellp = 20
    lmax_grid = gl_minimal_lmax(LMAX, LW)
    batch = _BatchedGL(mask_alm, LW, LMAX, lmax_grid, 2, None)
    mps = np.arange(ellp + 1)
    total = np.zeros(LMAX + 1)
    start = 0
    for size in sizes:
        chunk = mps[start : start + size]
        total += _batched_chunk_gl(
            batch, cl, ellp, chunk, np.where(chunk > 0, 2.0, 1.0)
        )
        start += size
    assert start == mps.size
    ell = np.arange(LMAX + 1)
    got = 2.0 / ((2 * ell + 1) * (2 * ellp + 1)) * total
    ref = _exact_row_gl(mask_alm, LW, cl, ellp, LMAX, nthreads=2, batched=False)
    assert_rows_close(got, ref)


def test_budget_sets_the_chunks_not_the_row(patchy):
    """
    Small budgets split the row into several chunks (a count that does not
    divide the columns included); the row agrees with the per-column one
    either way.
    """
    _, mask_alm = patchy
    cl = red_cl(LMAX)
    ellp = 32
    lmax_grid = gl_minimal_lmax(LMAX, LW)
    ref = _exact_row_gl(mask_alm, LW, cl, ellp, LMAX, nthreads=2, batched=False)
    # the smallest budget without the floor warning: one column per chunk
    exact._WARNED_MEMORY_FLOOR.clear()
    with pytest.warns(UserWarning, match="below the floor"):
        one = _batched_row_plan(range(ellp + 1), LMAX, LW, lmax_grid, 1, True)
    whole = _batched_row_plan(range(ellp + 1), LMAX, LW, lmax_grid, 2**30, True)
    middle = (one["total"] + whole["total"]) // 2
    sizes = []
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for budget in (one["total"], middle, 2**30):
            batch = _BatchedGL(mask_alm, LW, LMAX, lmax_grid, 2, budget / 1024**3)
            assert batch.hold_table
            sizes.append([c.size for c in batch.plan(range(ellp + 1))["chunks"]])
            got = _exact_row_gl(mask_alm, LW, cl, ellp, LMAX, nthreads=2, batch=batch)
            assert_rows_close(got, ref)
    assert sizes[0] == [1] * (ellp + 1)
    assert len(sizes[1]) > 1 and len(set(sizes[1])) > 1  # uneven chunks
    assert sizes[2] == [ellp + 1]


def test_held_and_streamed_tables_give_the_same_bits(patchy):
    """
    The Legendre table held whole and the one generated block by block for
    each pass are the same numbers (each ring pair is one ducc0 call), so
    a chunk gives the same bits either way, with the held table filled on
    the first pass and reused on the second.
    """
    _, mask_alm = patchy
    cl = red_cl(LMAX)
    lmax_grid = gl_minimal_lmax(LMAX, LW)
    held = _BatchedGL(mask_alm, LW, LMAX, lmax_grid, 2, 1.0)
    streamed = _BatchedGL(mask_alm, LW, LMAX, lmax_grid, 3, 1.0)
    streamed.hold_table = False
    assert held.hold_table
    for mps in (np.arange(0, 9), np.arange(20, 33), np.array([-4, 2, 11])):
        weights = np.ones(mps.size)
        a = _batched_chunk_gl(held, cl, 32, mps, weights)
        b = _batched_chunk_gl(streamed, cl, 32, mps, weights)
        np.testing.assert_array_equal(a, b)
    assert not streamed._table and len(held._table) == LMAX + 1


@pytest.mark.parametrize("hold_table", [True, False])
def test_rows_are_bit_identical_across_thread_counts(patchy, monkeypatch, hold_table):
    """
    The plan does not depend on ``nthreads`` and the ring stage hands fixed
    blocks of rings to its workers, so a row is the same to the bit for 1,
    2, 3 and 8 threads, with several chunks and (with a small ring block)
    several FFT calls per worker; held and streamed tables alike.
    """
    _, mask_alm = patchy
    monkeypatch.setattr(exact, "_FFT_BLOCK_BYTES", 16 * 2**10)
    lmax = 64
    cl = red_cl(lmax)
    lmax_grid = gl_minimal_lmax(lmax, LW)
    rows, chunks = [], []
    for nthreads in (1, 2, 3, 8):
        batch = _BatchedGL(mask_alm, LW, lmax, lmax_grid, nthreads, 0.005)
        batch.hold_table = hold_table
        chunks.append(len(batch.plan(range(61))["chunks"]))
        rows.append(
            _exact_row_gl(mask_alm, LW, cl, 60, lmax, nthreads=nthreads, batch=batch)
        )
    assert chunks[0] > 1 and len(set(chunks)) == 1
    for row in rows[1:]:
        np.testing.assert_array_equal(row, rows[0])


@pytest.mark.parametrize("width", [117, 45])
def test_ring_stages_are_bit_identical_across_thread_counts(patchy, width):
    """
    The ring stage of the table kernels (``_ring_stage``, mirror-paired
    rings; blocks of 7 and 19 rings here, 5 and 2 of them) and of the
    table-free polarised kernel (``_plane_ring_stage``, one block per plane)
    on random data give the same bits for 1 to 8 workers.  ducc0's batched
    FFT rounds a column differently in the SIMD tail of a call, which needs
    an odd number of columns: with the rings split by worker first (as
    before), 3 workers moved the last bit at width 117.
    """
    _, mask_alm = patchy
    lmax_grid = gl_minimal_lmax(LMAX, LW)
    rng = np.random.default_rng(3)
    n_fft = 150
    ntheta = gl_shape(lmax_grid)[0]
    ring = rng.standard_normal((n_fft, ntheta, 2 * width)).view(complex)
    planes = rng.standard_normal((width, ntheta, 2 * n_fft)).view(complex)
    outs = []
    for nthreads in (1, 2, 3, 5, 8):
        batch = _BatchedGL(mask_alm, LW, LMAX, lmax_grid, nthreads, None)
        a, b = ring.copy(), planes.copy()
        exact._ring_stage(batch, a)
        exact._plane_ring_stage(batch, b)
        outs.append((a, b))
    for a, b in outs[1:]:
        np.testing.assert_array_equal(a, outs[0][0])
        np.testing.assert_array_equal(b, outs[0][1])


def test_legendre_table_in_a_reused_buffer(patchy):
    """
    Tables built in a caller's buffer (``out``, as the streamed blocks
    are) are the same bits as in their own, for blocks of different sizes
    and first orders one after the other, and only the end of the buffer
    is written.
    """
    lmax, lmax_grid = 40, 70
    npair = (gl_north_rings(lmax_grid) + 1) // 2
    size = gl_legendre_table_bytes(np.arange(3), lmax, lmax_grid) // 16
    buf = np.full(size + (lmax + 1) * npair, np.nan, dtype=complex)
    for ms in (np.array([0, 1, 2]), np.array([7]), np.array([30, 31, 35])):
        assert buf.size >= gl_legendre_table_capacity(ms, lmax, lmax_grid)
        got = gl_legendre_table(ms, lmax, lmax_grid, 2, out=buf)
        ref = gl_legendre_table(ms, lmax, lmax_grid, 2)
        for a, b in zip(got, ref):
            for x, y in zip(a, b):
                np.testing.assert_array_equal(x, y)
    assert np.isnan(buf[: (lmax + 1) * npair - 3 * npair]).all()


@pytest.mark.parametrize(
    "which, ellps", [("cap", (12, 30)), ("two_blob", (20, 30))], ids=["cap", "two_blob"]
)
def test_term_selection_skips_the_same_columns(monkeypatch, which, ellps):
    """
    With ``term_selection`` the batched row computes exactly the ``m'`` of
    ``_kept_mprime`` (gaps included) and agrees with the per-column row
    under the same selection.  The masks are compact, so that the selection
    skips orders (on the two-patch mask it keeps them all up to ``l' = 30``
    even at a tolerance of 0.1): the baseline cap, azimuthally symmetric in
    its pole frame (where the selected columns are computed), and the
    two-blob mask of ``tests/conftest.py``, which is not (it skips the top
    two and three ``m'`` of ``l' = 20`` and ``30``).
    """
    if which == "cap":
        mask = healpy.read_map(os.path.join(DATA_DIR, "baseline_mask.fits"))
    else:
        mask = two_blob_mask(32)
    cl = red_cl(LMAX)
    tol = 1e-3
    mask_alm, lw = exact._mask_alm_for_gl(mask, None, LW)
    mask_alm = exact._pole_frame_alm(mask, mask_alm, lw)
    seen = []
    chunk = exact._batched_chunk_gl

    def spy(batch, cl_, ellp, mps, weights):
        seen.extend(int(m) for m in mps)
        return chunk(batch, cl_, ellp, mps, weights)

    monkeypatch.setattr(exact, "_batched_chunk_gl", spy)
    for ellp in ellps:
        seen.clear()
        kw = {"nthreads": 2, "term_selection": tol}
        got = _exact_row_gl(mask_alm, lw, cl, ellp, LMAX, **kw)
        keep = _kept_mprime(mask_alm, lw, ellp, tol, (0,), 2)
        kept = [m for m in range(ellp + 1) if keep[m + ellp]]
        assert seen == kept and len(kept) < ellp + 1
        ref = _exact_row_gl(mask_alm, lw, cl, ellp, LMAX, batched=False, **kw)
        assert_rows_close(got, ref)
    public = exact_covariance_row(
        mask, cl, 30, LMAX, grid="gl", lw=LW, term_selection=tol
    )
    assert_rows_close(public, ref)


# --------------------------------------------------------------------------- #
# The memory plan
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("lmax,lw", [(32, 48), (500, 384), (2000, 384)])
def test_memory_plan_within_budget(lmax, lw):
    """
    Every chunk of the plan fits: ``total`` (ring modes + table + the largest
    chunk) is within the budget, the chunks cover the columns once and in
    order, and the table is held exactly when it fits in half the budget.
    Plans only -- nothing is allocated -- so realistic sizes are cheap.
    """
    lmax_grid = gl_minimal_lmax(lmax, lw)
    mps = np.arange(lmax + 1)
    full_table = gl_legendre_table_bytes(mps, lmax, lmax_grid)
    for gb in (0.5, 2.0, 8.0, 32.0):
        budget = int(gb * 1024**3)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            plan = _batched_row_plan(mps, lmax, lw, lmax_grid, budget)
        assert plan["total"] <= budget
        assert plan["hold_table"] == (full_table <= budget // 2)
        np.testing.assert_array_equal(np.concatenate(plan["chunks"]), mps)
        sizes = [c.size for c in plan["chunks"]]
        assert max(sizes) >= 1


@pytest.mark.parametrize("hold_table", [True, False])
def test_allocations_stay_within_the_plan(patchy, hold_table):
    """
    What a batched row actually allocates (numpy allocations, traced with
    ``tracemalloc`` from before the ring modes are built to the end of the
    row) stays within the plan's ``total``, for a held and a streamed table
    and a budget that forces several chunks.  ducc0's internal buffers are
    not traced (nor charged).
    """
    import tracemalloc

    _, mask_alm = patchy
    lmax = 64
    cl = red_cl(lmax)
    lmax_grid = gl_minimal_lmax(lmax, LW)
    budget_gb = 0.006
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        batch = _BatchedGL(mask_alm, LW, lmax, lmax_grid, 2, budget_gb)
        batch.hold_table = hold_table
        plan = batch.plan(range(lmax + 1))
        exact._exact_row_gl_batched(batch, cl, lmax, range(lmax + 1), True)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(plan["chunks"]) > 1
    assert plan["total"] <= budget_gb * 1024**3
    assert peak <= plan["total"], (peak, plan)


def test_memory_plan_floor_warns():
    """A budget below one column per chunk warns once and still covers all."""
    lmax, lw = 500, 384
    lmax_grid = gl_minimal_lmax(lmax, lw)
    exact._WARNED_MEMORY_FLOOR.clear()
    with pytest.warns(UserWarning, match="below the floor"):
        plan = _batched_row_plan(range(lmax + 1), lmax, lw, lmax_grid, 10 * 2**20)
    assert all(c.size == 1 for c in plan["chunks"])
    assert len(plan["chunks"]) == lmax + 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _batched_row_plan(range(lmax + 1), lmax, lw, lmax_grid, 10 * 2**20)


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #
def test_public_entry_points_take_the_budget(patchy):
    mask, mask_alm = patchy
    lmax_int = LMAX + 18  # the mask's coupling width is 18
    cl = red_cl(lmax_int)
    rows = (5, 20, 32)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # not below the floor
        matrix = exact_covariance(
            mask_alm,
            cl,
            LMAX,
            lmax_int=lmax_int,
            rows=rows,
            grid="gl",
            nthreads=2,
            max_memory_gb=8e-3,
        )
    for ellp in rows:
        row = exact_covariance_row(
            mask_alm, cl, ellp, LMAX, grid="gl", nthreads=2, lmax_int=lmax_int
        )
        assert_rows_close(matrix[:, ellp], row)
    with pytest.raises(ValueError):
        exact_covariance_row(mask, cl, 5, LMAX, nside=32, max_memory_gb=1.0)
    with pytest.raises(ValueError):
        exact_covariance(mask, cl, LMAX, nside=32, rows=(5,), max_memory_gb=1.0)
    with pytest.raises(ValueError):
        exact_covariance_row(mask_alm, cl, 5, LMAX, grid="gl", max_memory_gb=0)
