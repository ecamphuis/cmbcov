r"""
The on-disk central coefficient store of the ACC precompute
(:class:`~cmbcov.approximations.acc._CentralStore`) and the primed-outer
loop order it enables.

When the central (``ell``) full-``M`` set does not fit in half the memory
budget, the precompute builds it once into a file and reads blocks back for
every ``ellp``; the contraction then loops over primed blocks outside
(synthesised once) and over central blocks inside (re-read).  Only the
summation order changes, so the three routes to a kernel -- held in RAM,
stored on disk, streamed (re-synthesised per block, the pre-store path) --
must agree to rounding.  Different code paths are not bit-identical across
BLAS libraries (macOS Accelerate vs Linux OpenBLAS), so the comparisons use
``max |a - b| <= 1e-13 * max |reference|``.

Checked on the rotated test mask (no azimuthal symmetry, so ``Im Theta``
contributes), with budgets small enough to force several blocks on both
sides, at ``ell == ellp`` and ``ell != ellp``, with and without term
selection; and the store is removed after a run and after an exception.
Term selection works in the mask's pole frame, where the rotated test mask
(a cap) is azimuthally symmetric again, so the selected runs are also
checked on the two-blob mask of ``tests/conftest.py``, which is not.
"""

import os
import shutil

import numpy as np
import pytest

healpy = pytest.importorskip("healpy")
import healpy as hp  # noqa: E402

from cmbcov.approximations import acc  # noqa: E402
from cmbcov.approximations.acc import (  # noqa: E402
    _CentralStore,
    _coefficient_provider,
    _coupling_block_sizes,
    _CouplingPrecompute,
    _gl_integrals,
    _pack_block,
    _paired_m_order,
    _TermSelectionPlan,
    precompute_acc_kernels,
)
from cmbcov.mask import MaskWlm  # noqa: E402
from cmbcov.term_selection import pole_rotation, rotate_alm  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "data")

NSIDE = 16
ELL = 16
LMAX = 2 * NSIDE
LW = 10
ROTATION = (37.0, 21.0, 11.0)  # breaks the test mask's azimuthal symmetry
TOL_TS = 1e-3  # term-selection tolerance
#: Mask band-limit of the term-selection tests: select_terms indexes the
#: mask's azimuthal autocorrelation at m - m', so it needs 2 lw >= ell + ellp.
LW_TS = 24

#: Cross-path tolerance, relative to the largest entry of the reference.
REL = 1e-13


@pytest.fixture(scope="module")
def rotated_mask_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("rotated_mask_store")
    mask = hp.read_map(os.path.join(DATA, "baseline_mask.fits"))
    rotated = hp.Rotator(rot=ROTATION).rotate_map_pixel(mask)
    hp.write_map(str(out / "baseline_mask.fits"), rotated, overwrite=True)
    return str(out)


@pytest.fixture(scope="module")
def wlm(rotated_mask_dir):
    return MaskWlm("baseline_mask.fits", load_path=rotated_mask_dir)


@pytest.fixture(scope="module", params=["rotated", "two_blob"])
def ts_wlm(request, wlm, two_blob_mask_dir):
    """The mask of the term-selection runs: the rotated test mask, and the
    two-blob mask (asymmetric in its pole frame)."""
    if request.param == "rotated":
        return wlm
    return MaskWlm("two_blob_mask.fits", load_path=two_blob_mask_dir)


@pytest.fixture(scope="module")
def rotated_mask_alm():
    mask = hp.read_map(os.path.join(DATA, "baseline_mask.fits"))
    alm = hp.map2alm(mask, lmax=LW, iter=10)
    return hp.Rotator(rot=ROTATION).rotate_alm(alm.copy(), lmax=LW)


def _assert_close(got: dict, ref: dict, label: str) -> float:
    """Every kernel within ``REL * max|ref|``; returns the worst ratio."""
    assert got.keys() == ref.keys()
    worst = 0.0
    for key, r in ref.items():
        scale = np.abs(r).max()
        diff = np.abs(got[key] - r).max()
        assert diff <= REL * scale, f"{label} {key}: {diff / scale:.2e}"
        worst = max(worst, diff / scale)
    return worst


def _per_m(nfields=3):
    return nfields * LMAX * (2 * LMAX - 1) * 16


# ---------------------------------------------------------------------------
# The store itself
# ---------------------------------------------------------------------------


def test_store_blocks_equal_packed_blocks(tmp_path, rotated_mask_alm):
    """``left`` / ``right`` read back exactly what :func:`_pack_block` packs,
    for contiguous and non-contiguous row sets."""
    integrals = _gl_integrals(rotated_mask_alm, LW, ELL, LMAX)
    provider = _coefficient_provider(
        integrals, None, lambda item: acc._gl_full_m(item), ELL, reflect=True
    )
    order = _paired_m_order(2 * ELL + 1)
    os.makedirs(tmp_path / "s")
    store = _CentralStore(str(tmp_path / "s"), order, 3, LMAX)
    try:
        store.fill(provider, chunk=5)
        for rows in (order[:7], order[3:4], order[10:], [order[2], order[9], order[4]]):
            np.testing.assert_array_equal(
                store.left(rows), _pack_block(provider, rows, LMAX, conjugate=False)
            )
            np.testing.assert_array_equal(
                store.right(rows), _pack_block(provider, rows, LMAX, conjugate=True)
            )
    finally:
        store.close()
    assert not os.path.exists(tmp_path / "s")


def test_provider_into_matches_the_returned_stack(rotated_mask_alm):
    """``into`` (straight into a block slot, conjugated or not) writes the
    same numbers the provider returns, on the reflected ``-m`` half too."""
    precompute = _CouplingPrecompute.__new__(_CouplingPrecompute)
    source = precompute._integral_source(
        "gl", None, NSIDE, rotated_mask_alm, LW, LMAX, t_only=False
    )
    held = _gl_integrals(rotated_mask_alm, LW, ELL, LMAX)
    for fields in (None, source["fields"]):
        provider = _coefficient_provider(
            None,
            lambda m: source["synthesise"](ELL, m),
            source["to_full_m"],
            ELL,
            reflect=True,
            fields=fields,
        )
        for i in _paired_m_order(2 * ELL + 1)[:9]:
            ref = acc._gl_full_m(held[i])
            np.testing.assert_array_equal(provider(i), ref)
            out = np.empty_like(ref)
            provider.into(i, out)
            np.testing.assert_array_equal(out, ref)
            provider.into(i, out, conjugate=True)
            np.testing.assert_array_equal(out, np.conj(ref))


# ---------------------------------------------------------------------------
# Held vs stored vs streamed
# ---------------------------------------------------------------------------


def _three_paths(
    precompute, ellp, budget, selection=None, mask_alm=None, tmp=None, lw=LW
):
    """The kernels of ``(ELL, ellp)`` from the held, stored and streamed
    central set, all at the same contraction budget."""
    kwargs = {
        "grid": "gl",
        "mask_alm": mask_alm,
        "lw": lw,
        "max_memory_bytes": budget,
        "selection": selection,
    }
    held = precompute._compute_central_integrals(
        ELL,
        None,
        NSIDE,
        grid="gl",
        mask_alm=mask_alm,
        lw=lw,
        lmax=LMAX,
        max_memory_bytes=1024**3,
        selection=selection,
    )
    assert held is not None
    store = precompute._central_store(
        ELL,
        grid="gl",
        wn_for_ell=None,
        nside=NSIDE,
        mask_alm=mask_alm,
        lw=lw,
        lmax=LMAX,
        t_only=False,
        selection=selection,
        max_memory_bytes=budget,
        scratch_dir=str(tmp),
    )
    assert store is not None
    try:
        results = {
            name: precompute._compute_ellp_coupling(
                ELL, ellp, central, {}, np.array([]), None, NSIDE, LMAX, **kwargs
            )
            for name, central in (("held", held), ("store", store), ("streamed", None))
        }
    finally:
        store.close()
    assert not os.path.exists(store.directory)
    return results


def _small_budget(n_m, n_mp):
    """A budget forcing several blocks on both sides on every path: the
    primed-outer order (held, stored; two central blocks resident) and the
    central-outer one (streamed)."""
    budget = 2 * 1024**2
    per_pair = 5 * LMAX * 16
    outer, inner = _coupling_block_sizes(
        n_mp, n_m, LMAX, _per_m(), per_pair, budget, inner_copies=2
    )
    assert 1 < outer < n_mp and 1 < inner < n_m, (outer, inner)
    outer, inner = _coupling_block_sizes(n_m, n_mp, LMAX, _per_m(), per_pair, budget)
    assert 1 < outer < n_m and 1 < inner < n_mp, (outer, inner)
    return budget


@pytest.mark.parametrize("ellp", [ELL, ELL + 1], ids=["diagonal", "off-diagonal"])
def test_held_stored_and_streamed_agree(wlm, rotated_mask_alm, tmp_path, ellp):
    precompute = _CouplingPrecompute(wlm)
    budget = _small_budget(2 * ELL + 1, 2 * ellp + 1)
    results = _three_paths(
        precompute, ellp, budget, mask_alm=rotated_mask_alm, tmp=tmp_path
    )
    # One pass, everything in RAM, as the reference.
    reference = precompute._compute_ellp_coupling(
        ELL,
        ellp,
        None,
        {},
        np.array([]),
        None,
        NSIDE,
        LMAX,
        grid="gl",
        mask_alm=rotated_mask_alm,
        lw=LW,
        max_memory_bytes=4 * 1024**3,
    )
    for name, got in results.items():
        _assert_close(got, reference, name)


@pytest.mark.parametrize("ellp", [ELL, ELL + 3], ids=["diagonal", "off-diagonal"])
def test_held_stored_and_streamed_agree_under_term_selection(ts_wlm, tmp_path, ellp):
    precompute = _CouplingPrecompute(ts_wlm)
    alm = hp.map2alm(ts_wlm.mask, lmax=LW_TS, iter=10)
    alm = rotate_alm(alm, LW_TS, pole_rotation(ts_wlm.mask))
    plan = _TermSelectionPlan(alm, LW_TS, LMAX, ELL, TOL_TS, t_only=False)
    sel = plan.selection(ELL, ellp)
    n_m, n_mp = int(sel.keep_m.sum()), int(sel.keep_mp.sum())
    budget = _small_budget(n_m, n_mp)
    results = _three_paths(
        precompute, ellp, budget, selection=plan, mask_alm=alm, tmp=tmp_path, lw=LW_TS
    )
    reference = results.pop("streamed")
    for name, got in results.items():
        _assert_close(got, reference, name)


@pytest.mark.parametrize(
    "term_selection, mask",
    [(None, "rotated"), (TOL_TS, "rotated"), (TOL_TS, "two_blob")],
    ids=["full", "selected", "selected-two_blob"],
)
def test_precompute_store_matches_in_ram_end_to_end(
    wlm, two_blob_mask_dir, tmp_path, term_selection, mask
):
    """Through :func:`precompute_acc_kernels`: a budget below half the
    central set (so the store is built, and blocks are small on both sides)
    against one that holds everything, over several ``ellp`` including
    ``ell`` itself; and the store directory is gone afterwards."""
    common = {
        "centralell": ELL,
        "ellprange": [ELL, ELL + 1, ELL + 3],
        "nside": NSIDE,
        "lw": LW if term_selection is None else LW_TS,
        "dryrun": True,
        "term_selection": term_selection,
    }
    if mask == "two_blob":
        wlm = MaskWlm("two_blob_mask.fits", load_path=two_blob_mask_dir)
    scratch = tmp_path / "scratch"
    small = precompute_acc_kernels(
        wlm, None, max_memory_gb=2.0 / 1024, scratch_dir=str(scratch), **common
    )
    big = precompute_acc_kernels(wlm, None, max_memory_gb=4.0, **common)
    for ellp in common["ellprange"]:
        _assert_close(small[ellp], big[ellp], f"ellp={ellp}")
    assert os.listdir(scratch) == []


def test_no_space_falls_back_to_streaming(wlm, tmp_path, monkeypatch):
    """Too little free space: a warning and the streamed path, same kernels."""
    common = {
        "centralell": ELL,
        "ellprange": [ELL, ELL + 1],
        "nside": NSIDE,
        "lw": LW,
        "dryrun": True,
    }
    stored = precompute_acc_kernels(
        wlm, None, max_memory_gb=2.0 / 1024, scratch_dir=str(tmp_path), **common
    )
    usage = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(
        acc.shutil, "disk_usage", lambda path: usage._replace(free=1024)
    )
    with pytest.warns(UserWarning, match="streaming the central set"):
        streamed = precompute_acc_kernels(
            wlm, None, max_memory_gb=2.0 / 1024, scratch_dir=str(tmp_path), **common
        )
    for ellp in common["ellprange"]:
        _assert_close(streamed[ellp], stored[ellp], f"ellp={ellp}")


# ---------------------------------------------------------------------------
# Location and clean-up
# ---------------------------------------------------------------------------


def _record_store_dirs(monkeypatch):
    seen = []
    original = _CentralStore.__init__

    def recording(self, directory, *a, **k):
        seen.append(directory)
        original(self, directory, *a, **k)

    monkeypatch.setattr(_CentralStore, "__init__", recording)
    return seen


def test_store_defaults_to_the_kernel_directory_and_is_removed(
    wlm, tmp_path, monkeypatch
):
    seen = _record_store_dirs(monkeypatch)
    save_dir = tmp_path / "kernels"
    precompute_acc_kernels(
        wlm,
        str(save_dir),
        centralell=ELL,
        ellprange=[ELL, ELL + 1],
        nside=NSIDE,
        lw=LW,
        max_memory_gb=2.0 / 1024,
    )
    assert len(seen) == 1
    assert os.path.dirname(seen[0]) == str(save_dir)
    assert os.path.basename(seen[0]).startswith("acc_central_store_")
    assert not os.path.exists(seen[0])
    assert sorted(os.listdir(save_dir)) == ["covariance_coupling"]


def test_store_is_removed_after_an_exception(wlm, tmp_path, monkeypatch):
    seen = _record_store_dirs(monkeypatch)

    def failing(self, *a, **k):
        assert os.path.exists(os.path.join(seen[-1], _CentralStore.FILENAME))
        raise RuntimeError("boom")

    monkeypatch.setattr(_CouplingPrecompute, "_compute_ellp_coupling", failing)
    with pytest.raises(RuntimeError, match="boom"):
        precompute_acc_kernels(
            wlm,
            None,
            centralell=ELL,
            ellprange=[ELL + 1],
            nside=NSIDE,
            lw=LW,
            max_memory_gb=2.0 / 1024,
            dryrun=True,
            scratch_dir=str(tmp_path),
        )
    assert len(seen) == 1 and not os.path.exists(seen[0])
    assert os.listdir(tmp_path) == []


def test_store_is_removed_when_its_build_fails(wlm, tmp_path, monkeypatch):
    seen = _record_store_dirs(monkeypatch)
    original = _CentralStore.fill

    def failing(self, *a, **k):
        original(self, *a, **k)
        raise KeyboardInterrupt

    monkeypatch.setattr(_CentralStore, "fill", failing)
    with pytest.raises(KeyboardInterrupt):
        precompute_acc_kernels(
            wlm,
            None,
            centralell=ELL,
            ellprange=[ELL + 1],
            nside=NSIDE,
            lw=LW,
            max_memory_gb=2.0 / 1024,
            dryrun=True,
            scratch_dir=str(tmp_path),
        )
    assert len(seen) == 1 and not os.path.exists(seen[0])
    assert os.listdir(tmp_path) == []
