r"""
The memory budget of the ACC precompute (``max_memory_gb``) and the threaded
array copies of its coefficient producer.

``max_memory_gb`` bounds everything the precompute holds that scales with
the problem (:func:`~cmbcov.approximations.acc._coupling_working_set`: the
output accumulator, the coefficient blocks, the ``Theta`` slab, the
producer's transients and caches; whole coefficient sets held in RAM come
out of the budget first).  These tests check the *plan*, which is plain
arithmetic and the same on every platform; the resident set on a real
survey mask is measured once and documented rather than asserted here
(RSS depends on the allocator and the operating system).

The threaded leg building (:func:`~cmbcov.grid._fill_legs`) and copies
(:meth:`~cmbcov.approximations.acc._CoefficientProvider.into`,
:meth:`~cmbcov.approximations.acc._CentralStore.right`) split elementwise
operations over rows, so they are compared bit for bit with one thread (the
same code path); comparisons with an independent reference use
``max |a - b| <= 1e-13 * max |reference|``.
"""

import os

import numpy as np
import pytest

healpy = pytest.importorskip("healpy")
import healpy as hp  # noqa: E402

from cmbcov import grid  # noqa: E402
from cmbcov.approximations import acc  # noqa: E402
from cmbcov.approximations.acc import (  # noqa: E402
    _CentralStore,
    _coefficient_provider,
    _coupling_block_sizes,
    _coupling_working_set,
    _CouplingPrecompute,
    _gl_banded_grid,
    _gl_producer_bytes,
    _healpix_producer_bytes,
    _pack_block,
    _paired_m_order,
    _TermSelectionPlan,
    contract_coupling_block,
)
from cmbcov.mask import MaskWlm  # noqa: E402
from cmbcov.utils import threading_utils  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "data")
NSIDE = 16
LMAX = 2 * NSIDE
ELL = 16
LW = 24
REL = 1e-13
ITEM = 16  # complex128


def _per_m(lmax, nfields=3):
    return nfields * lmax * (2 * lmax - 1) * ITEM


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------


CASES = [
    # (lmax, n_m, n_mp, nspec, nfields, inner_copies, lw)
    (32, 33, 35, 5, 3, 1, 24),
    (32, 33, 33, 1, 1, 2, 24),
    (512, 501, 501, 4, 3, 2, 875),  # the survey precompute, primed outer
    (512, 501, 699, 4, 3, 2, 875),
    (512, 501, 539, 9, 3, 1, 875),  # B-mode channels, streamed
    (2048, 1001, 1201, 5, 3, 2, 3071),
]


@pytest.mark.parametrize("case", CASES, ids=[str(c[:5]) for c in CASES])
def test_planned_working_set_stays_within_every_budget(case):
    """
    For budgets from the floor to far above the whole problem, the chosen
    blocking's planned working set -- every term, including the producer's
    legs and mask modes -- fits the budget, and one more outer order would
    not (the budget is used, not just respected).  The number of block
    passes never increases with the budget.
    """
    lmax, n_m, n_mp, nspec, nfields, inner_copies, lw = case
    per_m = _per_m(lmax, nfields)
    per_pair = (nspec + 1) * lmax * ITEM
    ellp = (n_mp - 1) // 2
    producer = _gl_producer_bytes(lw, lmax, ((n_m - 1) // 2, ellp), nfields)

    def plan(n_outer, n_inner):
        return _coupling_working_set(
            n_outer, n_inner, lmax, per_m, per_pair, nspec, inner_copies, producer
        )["total"]

    floor = plan(1, 1)
    whole = plan(n_m, n_mp)
    budgets = np.unique(
        np.concatenate(
            [np.geomspace(floor, 2 * whole, 600), np.linspace(floor, whole, 600)]
        ).astype(np.int64)
    )
    previous = None
    for budget in budgets:
        budget = int(budget)
        nb, nbp = _coupling_block_sizes(
            n_m,
            n_mp,
            lmax,
            per_m,
            per_pair,
            budget,
            nspec=nspec,
            inner_copies=inner_copies,
            producer_bytes=producer,
        )
        assert 1 <= nb <= n_m and 1 <= nbp <= n_mp
        assert plan(nb, nbp) <= budget, (budget, nb, nbp)
        if nb < n_m:
            assert plan(nb + 1, nbp) > budget, (budget, nb, nbp)
        passes = -(-n_m // nb) * -(-n_mp // nbp)
        if previous is not None:
            assert passes <= previous, (budget, nb, nbp)
        previous = passes
    assert (nb, nbp) == (n_m, n_mp)  # 2 * whole: a single pass


def test_the_plan_charges_every_resident_term():
    """The terms of the working set, written out for the survey
    precompute (``ell = 250``, ``ellp = 349``, ``nside = 256``, ``lw = 875``,
    four channels, primed outer with 419 orders, two 32-order central
    blocks)."""
    lmax, nspec = 512, 4
    per_m = _per_m(lmax)
    per_pair = (nspec + 1) * lmax * ITEM
    producer = _gl_producer_bytes(875, lmax, (250, 349), 3)
    lg = _gl_banded_grid(875, 349, lmax - 1)
    assert producer == (
        3 * per_m + 2 * (lg + 1) * lmax * ITEM + 2 * (lg + 1) * (2 * lg + 2) * ITEM
    )
    plan = _coupling_working_set(419, 32, lmax, per_m, per_pair, nspec, 2, producer)
    assert plan["kernels"] == (nspec**2 + 1) * lmax**2 * 8
    assert plan["outer"] == 419 * per_m
    assert plan["inner"] == 2 * 32 * per_m
    assert plan["theta"] == 419 * 32 * (nspec + 1) * lmax * ITEM
    assert plan["producer"] == producer
    assert plan["total"] == sum(v for k, v in plan.items() if k != "total")
    assert plan["total"] <= 12 * 1024**3
    npix = 12 * 256**2
    assert _healpix_producer_bytes(256, lmax, 3) == (
        acc._HEALPIX_PRODUCER_BYTES_PER_PIXEL * npix + per_m
    )


def test_a_budget_below_the_floor_warns_and_uses_single_orders():
    lmax, nspec = 512, 5
    per_m = _per_m(lmax)
    per_pair = (nspec + 1) * lmax * ITEM
    floor = _coupling_working_set(1, 1, lmax, per_m, per_pair, nspec, 2)["total"]
    with pytest.warns(UserWarning, match="below"):
        blocks = _coupling_block_sizes(
            501, 501, lmax, per_m, per_pair, floor - 1, nspec=nspec, inner_copies=2
        )
    assert blocks == (1, 1)
    assert _coupling_block_sizes(
        501, 501, lmax, per_m, per_pair, floor, nspec=nspec, inner_copies=2
    ) == (1, 1)


# ---------------------------------------------------------------------------
# Whole sets held in RAM come out of the budget
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mask_alm():
    mask = hp.read_map(os.path.join(DATA, "baseline_mask.fits"))
    return hp.map2alm(mask, lmax=LW, iter=10)


@pytest.fixture(scope="module")
def precompute():
    return _CouplingPrecompute(MaskWlm("baseline_mask.fits", load_path=DATA))


def _budgets_seen(monkeypatch):
    seen = []
    real = acc._accumulate_coupling_kernels

    def spy(*args, **kwargs):
        seen.append((args[5], kwargs.get("producer_bytes")))
        return real(*args, **kwargs)

    monkeypatch.setattr(acc, "_accumulate_coupling_kernels", spy)
    return seen


def test_a_held_central_set_comes_out_of_the_contraction_budget(
    precompute, mask_alm, monkeypatch
):
    budget = 64 * 1024**2
    held = precompute._compute_central_integrals(
        ELL,
        None,
        NSIDE,
        grid="gl",
        mask_alm=mask_alm,
        lw=LW,
        lmax=LMAX,
        max_memory_bytes=budget,
    )
    assert held is not None
    held_bytes = (2 * ELL + 1) * _per_m(LMAX)
    seen = _budgets_seen(monkeypatch)
    common = {"grid": "gl", "mask_alm": mask_alm, "lw": LW, "max_memory_bytes": budget}
    precompute._compute_ellp_coupling(
        ELL, ELL + 1, held, {}, np.array([]), None, NSIDE, LMAX, **common
    )
    precompute._compute_ellp_coupling(
        ELL, ELL + 1, None, {}, np.array([]), None, NSIDE, LMAX, **common
    )
    (with_held, producer), (streamed, _) = seen
    assert with_held == budget - held_bytes
    assert streamed == budget
    assert producer == _gl_producer_bytes(LW, LMAX, (ELL, ELL + 1), 3)


def test_a_primed_set_is_kept_only_within_half_the_budget(
    precompute, mask_alm, monkeypatch
):
    """A primed set reused by a later ``ellp`` is materialised only while
    the sets held stay within half the budget, is charged to the budget
    while kept, and is dropped at its last use."""
    ellp = ELL + 1
    primed_bytes = (2 * ellp + 1) * _per_m(LMAX)
    common = {"grid": "gl", "mask_alm": mask_alm, "lw": LW}
    seen = _budgets_seen(monkeypatch)

    cache = {}
    budget = 2 * primed_bytes
    first = precompute._compute_ellp_coupling(
        ELL,
        ellp,
        None,
        cache,
        np.array([ellp]),
        None,
        NSIDE,
        LMAX,
        max_memory_bytes=budget,
        **common,
    )
    assert list(cache) == [ellp]
    assert seen[-1][0] == budget - primed_bytes
    again = precompute._compute_ellp_coupling(
        ELL,
        ellp,
        None,
        cache,
        np.array([]),
        None,
        NSIDE,
        LMAX,
        max_memory_bytes=budget,
        **common,
    )
    assert cache == {}  # last use
    assert seen[-1][0] == budget - primed_bytes

    small = {}
    streamed = precompute._compute_ellp_coupling(
        ELL,
        ellp,
        None,
        small,
        np.array([ellp]),
        None,
        NSIDE,
        LMAX,
        max_memory_bytes=2 * primed_bytes - 2,
        **common,
    )
    assert small == {}  # would not fit in half: streamed
    assert seen[-1][0] == 2 * primed_bytes - 2
    for key, ref in first.items():
        scale = np.abs(ref).max()
        assert np.abs(again[key] - ref).max() <= REL * scale, key
        assert np.abs(streamed[key] - ref).max() <= REL * scale, key


def test_term_selection_keeps_two_mask_mode_grids(mask_alm):
    plan = _TermSelectionPlan(mask_alm, LW, LMAX, ELL, 1e-3, t_only=False)
    grids = set()
    for ell in range(ELL, ELL + 12, 2):
        modes = plan.mask_modes(ell)
        lg = _gl_banded_grid(LW, ell, LMAX - 1)
        grids.add(lg)
        np.testing.assert_array_equal(modes, grid.gl_mask_modes(plan.mask_alm, LW, lg))
        assert len(plan._modes._modes) <= acc._GL_MODE_GRIDS
    assert len(grids) > acc._GL_MODE_GRIDS


# ---------------------------------------------------------------------------
# Accumulating into the kernels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pairs", [None, [(0, 1), (2, 2), (1, 0)]])
def test_accumulating_into_out_matches_the_returned_increment(pairs):
    rng = np.random.default_rng(7)
    lmax, na, nbj, ncoef = 6, 3, 4, 11

    def cplx(*shape):
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    left = cplx(3, lmax, na, ncoef)
    right = cplx(3, lmax, nbj, ncoef).transpose(0, 1, 3, 2)
    cf, pf = [0, 1, 2], [0, 2, 1]
    start = rng.standard_normal((3, 3, lmax, lmax))
    increment = contract_coupling_block(left, right, cf, pf, output_pairs=pairs)
    reference = start + increment
    out = start.copy()
    returned = contract_coupling_block(left, right, cf, pf, output_pairs=pairs, out=out)
    assert returned is out
    assert np.abs(out - reference).max() <= REL * np.abs(reference).max()


# ---------------------------------------------------------------------------
# Threaded legs and copies
# ---------------------------------------------------------------------------


@pytest.fixture
def split_everything(monkeypatch):
    """One worker per entry block of 64 entries, so even the test sizes run
    on several threads."""
    monkeypatch.setattr(threading_utils, "ROW_BLOCK_ENTRIES_PER_THREAD", 64)


@pytest.mark.parametrize("ncomp", [1, 2])
@pytest.mark.parametrize("negative", [False, True])
def test_fill_legs_is_the_same_on_any_number_of_threads(
    split_everything, ncomp, negative
):
    rng = np.random.default_rng(ncomp + 2 * negative)
    ntheta, nphi, size, first = 37, 64, 29, 50  # the run wraps round nphi
    modes = rng.standard_normal((ntheta, nphi)) + 1j * rng.standard_normal(
        (ntheta, nphi)
    )
    wu = rng.standard_normal((ncomp, ntheta)) + 1j * rng.standard_normal(
        (ncomp, ntheta)
    )
    phase = (-1.0) ** np.arange(size) if negative else None

    def legs(nthreads):
        out = np.empty((ncomp, ntheta, size), dtype=np.complex128)
        grid._fill_legs(out, wu, modes, first, phase, nthreads)
        return out

    serial = legs(1)
    for nthreads in (2, 3, 8):
        np.testing.assert_array_equal(legs(nthreads), serial)

    columns = (first + np.arange(size)) % nphi
    reference = wu[:, :, None] * modes[None, :, columns]
    if negative:
        reference[0] *= phase
        if ncomp == 2:
            reference[1] *= -phase
    assert np.abs(serial - reference).max() <= REL * np.abs(reference).max()


def test_banded_integrals_do_not_depend_on_the_thread_count(split_everything, mask_alm):
    for spin in (0, 2):
        for m in (-5, 0, 7):
            one = grid.banded_integrals_gl(
                mask_alm,
                LW,
                ELL,
                m,
                LMAX - 1,
                LMAX - 1 + abs(m),
                spin=spin,
                nthreads=1,
            )
            four = grid.banded_integrals_gl(
                mask_alm,
                LW,
                ELL,
                m,
                LMAX - 1,
                LMAX - 1 + abs(m),
                spin=spin,
                nthreads=4,
            )
            np.testing.assert_array_equal(four, one)


def test_threaded_copies_into_packed_blocks_and_the_store(
    split_everything, mask_alm, monkeypatch, tmp_path
):
    """The provider's copies, conjugated copies and reflections, and the
    store's in-place conjugation, on four threads and on one."""
    source = _CouplingPrecompute(
        MaskWlm("baseline_mask.fits", load_path=DATA)
    )._integral_source("gl", None, NSIDE, mask_alm, LW, LMAX, t_only=False)

    def provider():
        return _coefficient_provider(
            None,
            lambda m: source["synthesise"](ELL, m),
            source["to_full_m"],
            ELL,
            reflect=True,
            fields=source["fields"],
        )

    order = _paired_m_order(2 * ELL + 1)

    def blocks(nthreads):
        monkeypatch.setattr(acc, "get_optimal_nthreads", lambda: nthreads)
        left = _pack_block(provider(), order, LMAX, conjugate=False)
        right = _pack_block(provider(), order, LMAX, conjugate=True)
        directory = tmp_path / f"store{nthreads}"
        directory.mkdir()
        store = _CentralStore(str(directory), order, 3, LMAX)
        try:
            store.fill(provider(), 8)
            stored = store.right(order[:9])
        finally:
            store.close()
        return left, right, stored

    one = blocks(1)
    four = blocks(4)
    for a, b in zip(four, one):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(one[1], np.conj(one[0]).transpose(0, 1, 3, 2))
    np.testing.assert_array_equal(
        one[2], np.conj(one[0][:, :, :9]).transpose(0, 1, 3, 2)
    )


def test_blocks_written_into_a_reused_buffer(mask_alm, tmp_path):
    """``_pack_block`` and the store write a block into the leading entries
    of a flat buffer (how the contraction reuses one allocation per side)
    with the same numbers as into a new array; a short buffer is refused."""
    source = _CouplingPrecompute(
        MaskWlm("baseline_mask.fits", load_path=DATA)
    )._integral_source("gl", None, NSIDE, mask_alm, LW, LMAX, t_only=False)
    provider = _coefficient_provider(
        None,
        lambda m: source["synthesise"](ELL, m),
        source["to_full_m"],
        ELL,
        reflect=True,
        fields=source["fields"],
    )
    order = _paired_m_order(2 * ELL + 1)
    per_m_entries = 3 * LMAX * (2 * LMAX - 1)
    buffer = np.full(12 * per_m_entries, np.nan, dtype=np.complex128)
    for conjugate in (False, True):
        new = _pack_block(provider, order[:7], LMAX, conjugate)
        reused = _pack_block(provider, order[:7], LMAX, conjugate, out=buffer)
        assert np.shares_memory(reused, buffer)
        np.testing.assert_array_equal(reused, new)
    with pytest.raises(ValueError, match="at least"):
        _pack_block(provider, order[:13], LMAX, False, out=buffer)

    directory = tmp_path / "store"
    directory.mkdir()
    store = _CentralStore(str(directory), order, 3, LMAX)
    try:
        store.fill(provider, 5)
        for rows in (order[:7], order[3:10]):
            np.testing.assert_array_equal(store.left(rows, buffer), store.left(rows))
            np.testing.assert_array_equal(store.right(rows, buffer), store.right(rows))
    finally:
        store.close()
