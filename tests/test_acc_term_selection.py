r"""
A-priori term selection in the ACC precompute
(``precompute_acc_kernels(..., term_selection=)``).

Pins four things:

1. ``term_selection=None`` changes nothing: the GL kernels with and without
   the keyword are bit-identical, and the HEALPix golden of
   ``tests/test_acc_coupling.py`` is still met with ``term_selection=None``
   spelled out.  (The refactor of the block contraction into
   :func:`contract_coupling_block` was also checked bit-for-bit against the
   pre-refactor code on 206 kernels -- GL and HEALPix, all spectra and
   T-only, single and multi-block -- when it was written.)
2. With ``term_selection=1e-3`` the kernels agree with the full ones to
   within the tolerance in relative Frobenius norm, on an off-axis
   cosine-apodised cap and on the survey-like baseline mask, for ``ellp =
   ell`` and ``ell + 3``.  Measured when the tests were written (all
   spectra): cap ``nside 32`` TTxTT 2.3e-7 / 8.7e-8, EExEE 6.9e-6 / 1.3e-6,
   TExTE 6.7e-7 / 8.8e-8, worst pair BBxBB 5.3e-4 / 1.2e-4 with 0.5% of the
   ``(m, m')`` pairs and ``m_band = 0``; baseline ``nside 16`` TTxTT
   6.9e-7 / 3.3e-7, EExEE 1.2e-5 / 4.9e-6, TExTE 2.3e-6 / 8.3e-7, worst
   BBxBB 8.7e-4 / 5.4e-4.  T-only precomputes (spin-0 selection alone) are
   looser: TTxTT 5.9e-6 / 2.3e-5 (cap), 1.7e-5 / 8.1e-6 (baseline).
3. The manifest round trip: ``term_selection`` and the fractions kept are
   written and read back, an old manifest without the field means ``None``,
   and a run must ask for the same ``term_selection`` to load the cache --
   in both directions.
4. :func:`contract_coupling_block` reproduces the in-place contraction it
   replaced bit-for-bit (no pair mask) and the naive masked gram to
   round-off (with a pair mask), and the ``+m / -m`` reflection shortcut
   still halves the transforms under a selection.
"""

import json
import os
import warnings

import numpy as np
import pytest

healpy = pytest.importorskip("healpy")

from cmbcov.approximations import acc, acc_cache  # noqa: E402
from cmbcov.approximations.acc import (  # noqa: E402
    _CENTRAL_FIELD,
    _PRIME_FIELD,
    CHANNEL_ALIASES,
    COUPLING_SPECTRA,
    contract_coupling_block,
    precompute_acc_kernels,
)
from cmbcov.covariance import (  # noqa: E402
    Cov,
    CovarianceConfig,
    CovarianceMethod,
)
from cmbcov.mask import MaskWlm  # noqa: E402

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
TOL = 1e-3
LW = 47
PAIRS = [(a, b) for a in COUPLING_SPECTRA for b in COUPLING_SPECTRA]


def cap_mask(nside, lon, lat, radius_deg, apod_deg):
    """Azimuthally symmetric cosine-apodised cap centred at (lon, lat)."""
    v = healpy.ang2vec(lon, lat, lonlat=True)
    pv = np.array(healpy.pix2vec(nside, np.arange(healpy.nside2npix(nside))))
    dist = np.degrees(np.arccos(np.clip(v @ pv, -1, 1)))
    x = np.clip((dist - (radius_deg - apod_deg)) / apod_deg, 0, 1)
    return 0.5 * (1 + np.cos(np.pi * x))


def frob(a):
    return np.sqrt(np.sum(a * a))


@pytest.fixture(scope="module")
def cap_wlm(tmp_path_factory):
    """20 deg cosine-apodised cap at (lon, lat) = (45, 30), nside 32."""
    path = tmp_path_factory.mktemp("cap")
    healpy.write_map(str(path / "cap.fits"), cap_mask(32, 45.0, 30.0, 20.0, 8.0))
    return MaskWlm("cap.fits", load_path=str(path))


@pytest.fixture(scope="module")
def baseline_wlm():
    return MaskWlm("baseline_mask.fits", load_path=DATA)


def _precompute(wlm, ell, nside, **kw):
    options = {
        "centralell": ell,
        "ellprange": [ell, ell + 3],
        "nside": nside,
        "grid": "gl",
        "lw": LW,
        "dryrun": True,
    }
    options.update(kw)
    return precompute_acc_kernels(wlm, None, **options)


# --------------------------------------------------------------------------- #
# 1. the default path is untouched
# --------------------------------------------------------------------------- #
def test_term_selection_none_is_bit_identical_to_the_plain_call(baseline_wlm):
    plain = _precompute(baseline_wlm, 16, 16)
    explicit = _precompute(baseline_wlm, 16, 16, term_selection=None)
    assert plain.keys() == explicit.keys()
    for ellp in plain:
        assert plain[ellp].keys() == explicit[ellp].keys()
        for key in plain[ellp]:
            assert np.array_equal(plain[ellp][key], explicit[ellp][key]), (ellp, key)


def test_term_selection_none_still_meets_the_healpix_golden():
    """The golden of tests/test_acc_coupling.py, with the keyword spelled out."""
    coupling = precompute_acc_kernels(
        "baseline_mask.fits",
        None,
        centralell=16,
        dmax=2,
        mask_path=DATA,
        nside=16,
        grid="healpix",
        dryrun=True,
        term_selection=None,
    )
    expected = np.load(os.path.join(DATA, "baseline_acc_coupling.npz"))
    for name in expected.files:
        ellp, legacy_key = name.split("|")
        legacy_a, legacy_b = legacy_key.split("x")
        key = f"{CHANNEL_ALIASES[legacy_a]}x{CHANNEL_ALIASES[legacy_b]}"
        ref = expected[name]
        np.testing.assert_allclose(
            coupling[int(ellp)][key], ref, rtol=1e-10, atol=1e-14 * np.abs(ref).max()
        )


def test_term_selection_is_gl_only(baseline_wlm):
    with pytest.raises(ValueError, match="grid='gl'"):
        precompute_acc_kernels(
            baseline_wlm,
            None,
            centralell=16,
            dmax=1,
            nside=16,
            grid="healpix",
            dryrun=True,
            term_selection=TOL,
        )
    with pytest.raises(ValueError):
        _precompute(baseline_wlm, 16, 16, term_selection=-1.0)


# --------------------------------------------------------------------------- #
# 2. the selected kernels are within tolerance of the full ones
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spectra", [None, ("TT",)], ids=["all", "TT"])
@pytest.mark.parametrize("which", ["cap", "baseline"])
def test_selected_kernels_are_within_tolerance(
    request, which, spectra, cap_wlm, baseline_wlm, capsys
):
    wlm, ell, nside = (cap_wlm, 32, 32) if which == "cap" else (baseline_wlm, 16, 16)
    full = _precompute(wlm, ell, nside, spectra=spectra)
    with capsys.disabled():
        print()
    selected = _precompute(wlm, ell, nside, spectra=spectra, term_selection=TOL)
    checked = ["TTxTT"] if spectra == ("TT",) else ["TTxTT", "DDxDD", "TDxTD"]
    for ellp in (ell, ell + 3):
        assert full[ellp].keys() == selected[ellp].keys()
        errs = {
            key: frob(full[ellp][key] - selected[ellp][key]) / frob(full[ellp][key])
            for key in full[ellp]
        }
        worst = max(errs, key=errs.get)
        with capsys.disabled():
            print(
                f"  {which} nside {nside} spectra {spectra} ellp {ellp}: "
                + ", ".join(f"{k} {errs[k]:.2e}" for k in checked)
                + f"; worst {worst} {errs[worst]:.2e}"
            )
        for key in checked:
            assert errs[key] <= TOL, (which, spectra, ellp, key, errs[key])
        # every kernel of the set, not only the three named ones
        assert errs[worst] <= TOL, (which, spectra, ellp, worst, errs[worst])
        assert errs["TTxTT"] > 0  # the selection did drop something


def test_selection_actually_drops_terms(cap_wlm):
    """The plan behind the cap run keeps a fraction of the orders and pairs."""
    from cmbcov.approximations.acc import _TermSelectionPlan
    from cmbcov.sht import ducc0_map2alm
    from cmbcov.term_selection import pole_rotation, rotate_alm

    alm = ducc0_map2alm(cap_wlm.mask, lmax=LW, pol=False, iter=10)
    alm = rotate_alm(alm, LW, pole_rotation(cap_wlm.mask))
    plan = _TermSelectionPlan(alm, LW, 64, 32, TOL, t_only=False)
    stats = plan.stats(32, 35)
    assert 0 < stats["frac_m"] < 0.6 and 0 < stats["frac_mp"] < 0.6
    assert 0 < stats["frac_pairs"] < 0.02
    assert stats["m_band"] <= 1 and 0 < stats["frac_band"] < 0.05
    # keep_m of a degree does not depend on the partner multipole
    assert np.array_equal(plan.selection(32, 32).keep_m, plan.selection(32, 35).keep_m)


# --------------------------------------------------------------------------- #
# 3. the manifest round trip and the loader's refusal
# --------------------------------------------------------------------------- #
def _cov(kernel_dir, term_selection):
    config = CovarianceConfig(
        method=CovarianceMethod.ACC,
        lmax=32,
        dmax=1,
        centralell=16,
        acc_term_selection=term_selection,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # low centralell
        return Cov(
            "baseline_mask.fits",
            config=config,
            mask_path=DATA,
            save_dir=str(kernel_dir),
        )


def _write(kernel_dir, term_selection):
    precompute_acc_kernels(
        "baseline_mask.fits",
        str(kernel_dir),
        centralell=16,
        dmax=1,
        mask_path=DATA,
        nside=16,
        grid="gl",
        lw=LW,
        spectra=("TT",),
        term_selection=term_selection,
    )


def test_manifest_records_the_selection_and_its_fractions(tmp_path):
    _write(tmp_path / "sel", TOL)
    manifest = acc_cache.read_coupling_manifest(str(tmp_path / "sel"), 16, 16)
    assert manifest["term_selection"] == TOL
    stats = manifest["term_selection_stats"]
    assert set(stats) == {"frac_m", "frac_mp", "frac_pairs", "m_band", "frac_band"}
    assert 0 < stats["frac_pairs"] < 1 and isinstance(stats["m_band"], int)

    _write(tmp_path / "full", None)
    manifest = acc_cache.read_coupling_manifest(str(tmp_path / "full"), 16, 16)
    assert "term_selection" in manifest and manifest["term_selection"] is None
    assert "term_selection_stats" not in manifest


def test_loader_refuses_a_cache_with_a_different_term_selection(tmp_path):
    _write(tmp_path / "sel", TOL)
    _write(tmp_path / "full", None)
    pair = [("TT", "TT")]

    # the matching runs load
    for kernel_dir, ts in ((tmp_path / "sel", TOL), (tmp_path / "full", None)):
        loaded = _cov(kernel_dir, ts).get_acc_coupling_kernels(
            16, 16, COUPLING_SPECTRA, pair, pairs=pair
        )
        assert loaded[("TT", "TT")].shape == (32, 32)

    # a full-kernel run must not load selected kernels ...
    with pytest.raises(ValueError, match="term_selection") as excinfo:
        _cov(tmp_path / "sel", None).get_acc_coupling_kernels(
            16, 16, COUPLING_SPECTRA, pair, pairs=pair
        )
    assert "term selection" in str(excinfo.value)
    # ... nor a selected run the full ones, nor a different tolerance
    with pytest.raises(ValueError, match="term_selection"):
        _cov(tmp_path / "full", TOL).get_acc_coupling_kernels(
            16, 16, COUPLING_SPECTRA, pair, pairs=pair
        )
    with pytest.raises(ValueError, match="term_selection"):
        _cov(tmp_path / "sel", 1e-2).get_acc_coupling_kernels(
            16, 16, COUPLING_SPECTRA, pair, pairs=pair
        )


def test_old_manifest_without_the_field_means_none(tmp_path):
    _write(tmp_path / "full", None)
    path = acc_cache.coupling_manifest_path(str(tmp_path / "full"), 16, 16)
    with open(path) as f:
        manifest = json.load(f)
    del manifest["term_selection"]
    with open(path, "w") as f:
        json.dump(manifest, f)
    pair = [("TT", "TT")]
    _cov(tmp_path / "full", None).get_acc_coupling_kernels(
        16, 16, COUPLING_SPECTRA, pair, pairs=pair
    )
    with pytest.raises(ValueError, match="term_selection"):
        _cov(tmp_path / "full", TOL).get_acc_coupling_kernels(
            16, 16, COUPLING_SPECTRA, pair, pairs=pair
        )


def test_config_rejects_a_non_positive_tolerance():
    config = CovarianceConfig(
        method=CovarianceMethod.ACC,
        lmax=32,
        dmax=1,
        centralell=200,
        acc_term_selection=0.0,
    )
    with pytest.raises(ValueError, match="acc_term_selection"):
        config.validate()


# --------------------------------------------------------------------------- #
# 4. the block contraction and the reflection shortcut
# --------------------------------------------------------------------------- #
def _inline_reference(left, right, spec_indices):
    """The pre-refactor in-place contraction of one block, verbatim."""
    nspec = len(spec_indices)
    lmax, na = left.shape[1], left.shape[2]
    nbj = right.shape[3]
    theta = np.empty((nspec, lmax, na * nbj), dtype=np.complex128)
    for out_k, k in enumerate(spec_indices):
        np.matmul(
            left[_CENTRAL_FIELD[k]],
            right[_PRIME_FIELD[k]],
            out=theta[out_k].reshape(lmax, na, nbj),
        )
    flat = theta.view(np.float64)
    out = np.zeros((nspec, nspec, lmax, lmax))
    for k1 in range(nspec):
        for k2 in range(k1, nspec):
            out[k1, k2] += flat[k1] @ flat[k2].T
    for k1 in range(nspec):
        for k2 in range(k1):
            out[k1, k2] = out[k2, k1].T
    return out


@pytest.mark.parametrize("spec_indices", [list(range(5)), [0], [0, 3, 4]])
def test_contract_coupling_block_equals_the_inline_computation(spec_indices):
    rng = np.random.default_rng(7)
    lmax, na, nbj, ncoef = 9, 4, 5, 2 * 9 - 1
    nfields = 3
    left = rng.normal(size=(nfields, lmax, na, ncoef)) + 1j * rng.normal(
        size=(nfields, lmax, na, ncoef)
    )
    right = rng.normal(size=(nfields, lmax, ncoef, nbj)) + 1j * rng.normal(
        size=(nfields, lmax, ncoef, nbj)
    )
    central = [_CENTRAL_FIELD[k] for k in spec_indices]
    prime = [_PRIME_FIELD[k] for k in spec_indices]
    got = contract_coupling_block(left, right, central, prime)
    ref = _inline_reference(left, right, spec_indices)
    assert got.shape == ref.shape
    assert np.array_equal(got, ref)  # bit-identical

    # with a pair mask: the naive masked gram
    mask = rng.random((na, nbj)) < 0.4
    got = contract_coupling_block(left, right, central, prime, pair_mask=mask)
    theta = np.array([np.matmul(left[a], right[b]) for a, b in zip(central, prime)])
    theta = theta * mask[None, None, :, :]
    flat = theta.reshape(len(spec_indices), lmax, na * nbj)
    naive = np.einsum("aLp,bKp->abLK", flat, np.conj(flat)).real
    np.testing.assert_allclose(got, naive, rtol=1e-12, atol=1e-12 * np.abs(naive).max())
    assert not np.allclose(got, ref)  # the mask did something


def test_contract_coupling_block_accepts_an_array_namespace():
    """The ``xp`` seam: a numpy-compatible namespace gives the same result."""
    import types

    rng = np.random.default_rng(3)
    lmax, na, nbj, ncoef = 5, 2, 3, 9
    left = rng.normal(size=(1, lmax, na, ncoef)) + 0j
    right = rng.normal(size=(1, lmax, ncoef, nbj)) + 0j
    calls = []

    def logged(name):
        f = getattr(np, name)

        def wrapper(*a, **k):
            calls.append(name)
            return f(*a, **k)

        return wrapper

    xp = types.SimpleNamespace(
        **{n: logged(n) for n in ("matmul", "stack", "where", "asarray", "zeros")}
    )
    got = contract_coupling_block(
        left, right, [0], [0], pair_mask=np.ones((na, nbj), bool), xp=xp
    )
    ref = contract_coupling_block(left, right, [0], [0])
    assert np.array_equal(got, ref)
    assert {"matmul", "stack", "where"} <= set(calls)


def test_reflection_shortcut_survives_the_selection(cap_wlm, monkeypatch):
    """
    Streamed (not held) central set: the provider synthesises ``|m|`` once
    and reflects ``-m``; with a selection the kept ``+m / -m`` stay adjacent
    in the iteration order, so the number of banded transforms on the
    central side is the number of kept ``|m|`` -- about half the kept orders
    (15 of 29 at ``ell = 32`` on the cap).  The primed side is a different
    multipole so that the central calls can be told apart.
    """
    from cmbcov.approximations.acc import _TermSelectionPlan
    from cmbcov.sht import ducc0_map2alm
    from cmbcov.term_selection import pole_rotation, rotate_alm

    ell, nside = 32, 32
    calls = []
    original = acc.banded_integrals_gl

    def counting(mask_alm, lw, l_val, m, *a, **k):
        calls.append((l_val, m))
        return original(mask_alm, lw, l_val, m, *a, **k)

    monkeypatch.setattr(acc, "banded_integrals_gl", counting)
    # 2 MB: below half the T-only selected central set (3.8 MB, so it is
    # streamed) but above the contraction floor (~0.6 MB, no warning)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _precompute(
            cap_wlm,
            ell,
            nside,
            spectra=("TT",),
            ellprange=[ell + 3],
            term_selection=TOL,
            max_memory_gb=0.002,
        )
    alm = ducc0_map2alm(cap_wlm.mask, lmax=LW, pol=False, iter=10)
    alm = rotate_alm(alm, LW, pole_rotation(cap_wlm.mask))
    keep = _TermSelectionPlan(alm, LW, 2 * nside, ell, TOL, t_only=True).keep_m(ell)
    ms = np.arange(-ell, ell + 1)[keep]
    n_abs = len(set(np.abs(ms)))
    central = [m for (l_val, m) in calls if l_val == ell]
    assert central and all(m >= 0 for m in central)  # -m always reflected
    assert sorted(central) == sorted(set(np.abs(ms)))  # each kept |m| once
    assert len(central) == n_abs < keep.sum()  # the halving is kept
