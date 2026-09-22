r"""
The nine-channel :data:`COUPLING_CHANNELS`, old-name aliases,
and the explicit ``pairs`` selection of :func:`precompute_acc_kernels`.

This is deliberately independent of ``tests/test_acc_gl_pol.py`` and
``tests/test_acc_spectra_selection.py`` (which pin the *default* five
channels, renamed but otherwise unchanged): it pins the four B-mode-related
things --

1. :func:`normalise_channel` / :data:`CHANNEL_ALIASES` accept both the old
   and the new channel strings and agree on the canonical name.
2. A kernel cache written under the old channel names (``EE, BB, TE, ET``)
   loads through :meth:`ACCStrategy.get_covariance_coupling` without a
   recompute, bit-identical to the same cache under its new names.
3. ``precompute_acc_kernels(..., pairs=[...])`` restricts the output to an
   explicit list of ordered pairs, bit-identical to the corresponding entries
   of the full-square (``pairs=None``) run on the same tiny config.
4. A precompute requesting the new ``TL`` channel (not computed by any
   default run) reproduces the naive ``(m, m')`` double-loop reference
   (the same reference form as ``tests/test_acc_vectorised.py``), for one
   pair on a tiny config.
5. Every place that probes an ACC kernel file *by name* on disk -- not only
   the kernel loader itself -- accepts an old-name-only cache: the survey
   cache has no new-named file at all, so :meth:`Cov.acc_internal_lmax` and
   :meth:`ACCStrategy.raw_block_inputs` (which read the ``.npy`` header size
   and the file's ``stat``, not its array) and the assembled covariance
   itself must all agree, old-name-only cache against new-name cache.
"""

import os
import shutil
import warnings

import numpy as np
import pytest

healpy = pytest.importorskip("healpy")

from cmbcov.approximations import StrategyFactory  # noqa: E402
from cmbcov.approximations.acc import (  # noqa: E402
    _CENTRAL_FIELD,
    _PRIME_FIELD,
    CHANNEL_ALIASES,
    COUPLING_CHANNELS,
    _gl_integrals,
    normalise_channel,
    precompute_acc_kernels,
)
from cmbcov.covariance import (  # noqa: E402
    Cov,
    CovarianceConfig,
    CovarianceMethod,
)
from cmbcov.grid import cross_spectrum_full_m  # noqa: E402
from cmbcov.keys import CovKey  # noqa: E402
from cmbcov.mask import MaskWlm  # noqa: E402
from cmbcov.sht import ducc0_map2alm  # noqa: E402

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
NSIDE = 16
ELL = 16
LMAX = 48
LW = 10
FREQ = "090GHz"


def _cov(save_dir, dmax=1):
    config = CovarianceConfig(
        method=CovarianceMethod.ACC, lmax=LMAX, dmax=dmax, centralell=ELL
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # low centralell warning
        return Cov(
            "baseline_mask.fits", config=config, mask_path=DATA, save_dir=save_dir
        )


# --------------------------------------------------------------------------- #
# 1. Alias normalisation
# --------------------------------------------------------------------------- #


def test_normalise_channel_accepts_old_and_new_names():
    for old, new in CHANNEL_ALIASES.items():
        assert normalise_channel(old) == new
        assert new in COUPLING_CHANNELS
    for new in COUPLING_CHANNELS:
        assert normalise_channel(new) == new


def test_normalise_channel_rejects_unknown_strings():
    with pytest.raises(ValueError):
        normalise_channel("XX")


# --------------------------------------------------------------------------- #
# 2. Old-name cache loads without recompute, bit-identical
# --------------------------------------------------------------------------- #


def _rename_to_legacy(coupling_dir: str) -> None:
    """
    Rewrite every ``.npy`` kernel file under ``coupling_dir`` from its new
    channel name to the old one (``CHANNEL_ALIASES`` inverted), simulating a
    cache written before B-mode support.
    """
    legacy_of = {new: old for old, new in CHANNEL_ALIASES.items() if old != new}
    for name in os.listdir(coupling_dir):
        if not name.endswith(".npy"):
            continue
        pair_part, rest = name.split("_", 1)
        a, b = pair_part.split("x")
        legacy_a, legacy_b = legacy_of.get(a, a), legacy_of.get(b, b)
        if (legacy_a, legacy_b) != (a, b):
            os.rename(
                os.path.join(coupling_dir, name),
                os.path.join(coupling_dir, f"{legacy_a}x{legacy_b}_{rest}"),
            )


def test_old_name_cache_loads_bit_identically(tmp_path):
    new_dir = tmp_path / "new"
    legacy_dir = tmp_path / "legacy"
    new_dir.mkdir()

    precompute_acc_kernels(
        "baseline_mask.fits",
        str(new_dir),
        centralell=ELL,
        dmax=1,
        mask_path=DATA,
        nside=NSIDE,
        grid="healpix",
    )
    shutil.copytree(new_dir, legacy_dir)
    _rename_to_legacy(str(legacy_dir / "covariance_coupling"))

    strategy_new = StrategyFactory.create_strategy(_cov(str(new_dir)))
    strategy_legacy = StrategyFactory.create_strategy(_cov(str(legacy_dir)))

    kernels_new = strategy_new.get_covariance_coupling(ELL, ELL)
    kernels_legacy = strategy_legacy.get_covariance_coupling(ELL, ELL)

    assert set(kernels_new) == set(kernels_legacy)
    for key in kernels_new:
        np.testing.assert_array_equal(kernels_new[key], kernels_legacy[key])


def _te_cl(size):
    """TT power law, EE = 0.1 TT, TE = 0.5 sqrt(TT EE) -- as in test_acc_gl_pol.py."""
    ell = np.arange(size)
    tt = np.zeros(size)
    tt[2:] = 1e-3 / (ell[2:] * (ell[2:] + 1)) ** 0.9
    ee = 0.1 * tt
    te = 0.5 * np.sqrt(tt * ee)
    return {"TT": tt, "EE": ee, "TE": te}


def test_legacy_only_cache_gives_the_same_lmax_int_and_covariance_block(tmp_path):
    """
    Every by-name disk probe an old-name-only cache exercises --
    ``acc_cached_kernel_size`` (via ``Cov.acc_internal_lmax``), the
    ``raw_block_inputs`` file stat digest, and the kernel loader itself --
    must agree with the new-named copy of the same cache: same ``lmax_int``,
    same assembled covariance block. The user's real survey cache has no
    new-named file at all, so this is end to end, not just the loader.
    """
    dmax = 2
    new_dir = tmp_path / "new"
    legacy_dir = tmp_path / "legacy"
    new_dir.mkdir()

    precompute_acc_kernels(
        "baseline_mask.fits",
        str(new_dir),
        centralell=ELL,
        dmax=dmax,
        mask_path=DATA,
        nside=NSIDE,
        grid="healpix",
    )
    shutil.copytree(new_dir, legacy_dir)
    _rename_to_legacy(str(legacy_dir / "covariance_coupling"))
    # Confirm this is genuinely an old-name-only cache: no new-named file
    # survives under legacy_dir.
    legacy_names_on_disk = os.listdir(legacy_dir / "covariance_coupling")
    assert not any(
        name.startswith(("DD", "LL", "TD", "DT")) for name in legacy_names_on_disk
    )
    assert any(
        name.startswith(("EE", "BB", "TE", "ET")) for name in legacy_names_on_disk
    )

    cov_new = _cov(str(new_dir), dmax=dmax)
    cov_legacy = _cov(str(legacy_dir), dmax=dmax)

    lmax_int_new = cov_new.acc_internal_lmax()
    lmax_int_legacy = cov_legacy.acc_internal_lmax()
    assert lmax_int_new is not None
    assert lmax_int_new == lmax_int_legacy

    key = CovKey(("T", "E", "T", "E"), (FREQ,) * 4)
    cl = {FREQ + FREQ: _te_cl(lmax_int_new)}

    strategy_new = StrategyFactory.create_strategy(cov_new)
    strategy_legacy = StrategyFactory.create_strategy(cov_legacy)

    spectra_new, identity_new = strategy_new.raw_block_inputs(key, cl)
    spectra_legacy, identity_legacy = strategy_legacy.raw_block_inputs(key, cl)
    assert identity_new["lmax_int"] == identity_legacy["lmax_int"]
    assert identity_new["dmax"] == identity_legacy["dmax"]
    assert identity_new["centralell"] == identity_legacy["centralell"]
    assert set(spectra_new) == set(spectra_legacy)
    for label in spectra_new:
        np.testing.assert_array_equal(spectra_new[label], spectra_legacy[label])

    block_new = strategy_new.compute_covariance_term(key, cl)
    block_legacy = strategy_legacy.compute_covariance_term(key, cl)
    assert np.abs(block_new).max() > 0
    np.testing.assert_array_equal(block_new, block_legacy)


# --------------------------------------------------------------------------- #
# 3. Explicit pairs vs the full square
# --------------------------------------------------------------------------- #


def test_explicit_pairs_match_the_full_square():
    common = {
        "centralell": ELL,
        "dmax": 1,
        "mask_path": DATA,
        "nside": NSIDE,
        "dryrun": True,
        "grid": "healpix",
    }
    full = precompute_acc_kernels("baseline_mask.fits", None, **common)[ELL]

    pairs = [("TT", "TT"), ("DD", "LL"), ("TD", "DT"), ("DT", "TD")]
    selected = precompute_acc_kernels(
        "baseline_mask.fits", None, pairs=pairs, **common
    )[ELL]

    assert set(selected) == {f"{a}x{b}" for a, b in pairs}
    for a, b in pairs:
        key = f"{a}x{b}"
        np.testing.assert_array_equal(selected[key], full[key])


def test_explicit_pairs_accept_old_names_and_infer_the_channel_set():
    """``pairs`` normalises old names too, and (with ``spectra=None``) the
    channel set computed is exactly what ``pairs`` references."""
    common = {
        "centralell": ELL,
        "dmax": 1,
        "mask_path": DATA,
        "nside": NSIDE,
        "dryrun": True,
        "grid": "healpix",
    }
    full = precompute_acc_kernels("baseline_mask.fits", None, **common)[ELL]

    selected = precompute_acc_kernels(
        "baseline_mask.fits", None, pairs=[("TT", "EE")], **common
    )[ELL]

    assert set(selected) == {"TTxDD"}
    np.testing.assert_array_equal(selected["TTxDD"], full["TTxDD"])


# --------------------------------------------------------------------------- #
# 4. A new B-mode channel (TL) against the naive (m, m') double loop
# --------------------------------------------------------------------------- #


def _naive_kernel(mask_alm, lw, ell, ellp, lmax, central_field, prime_field):
    """
    ``Re sum_{m m'} Theta(m, m', L1) conj(Theta(m, m', L2))`` for one channel
    pair, ``Theta(m, m', L) = sum_M X^a_{m L M} conj(Y^b_{m' L M})``, computed
    with the plain ``(m, m')`` double loop and :func:`cross_spectrum_full_m`
    -- the same reference form ``tests/test_acc_vectorised.py`` checks the
    blocked contraction against, restricted here to a single channel.
    """
    central = _gl_integrals(mask_alm, lw, ell, lmax)
    primed = central if ellp == ell else _gl_integrals(mask_alm, lw, ellp, lmax)

    def field(item, f):
        u_t, u_eb = item
        return u_t if f == 0 else u_eb[f - 1]

    theta = np.empty((len(central), len(primed), lmax), dtype=np.complex128)
    for m, item in enumerate(central):
        x = field(item, central_field)
        for mp, itemp in enumerate(primed):
            y = field(itemp, prime_field)
            theta[m, mp, :] = cross_spectrum_full_m(x, y)

    flat = theta.reshape(-1, lmax)
    return (flat.T @ flat.conj()).real


def test_new_bmode_channel_matches_the_naive_double_loop():
    """``TL`` (T central, L=B primed -- new, not part of any default
    precompute) reproduces the naive double-loop reference on a tiny config."""
    wlm = MaskWlm("baseline_mask.fits", load_path=DATA)
    mask_alm = ducc0_map2alm(wlm.mask, lmax=LW, pol=False, iter=10)
    lmax = 2 * NSIDE

    k = COUPLING_CHANNELS.index("TL")
    reference = _naive_kernel(
        mask_alm, LW, ELL, ELL, lmax, _CENTRAL_FIELD[k], _PRIME_FIELD[k]
    )

    kernels = precompute_acc_kernels(
        wlm,
        None,
        centralell=ELL,
        ellprange=[ELL],
        nside=NSIDE,
        dryrun=True,
        grid="gl",
        lw=LW,
        pairs=[("TL", "TL")],
    )[ELL]

    assert set(kernels) == {"TLxTL"}
    np.testing.assert_allclose(
        kernels["TLxTL"],
        reference,
        rtol=1e-11,
        atol=1e-14 * np.abs(reference).max(),
    )


@pytest.mark.parametrize("pair", [("DL", "DL"), ("LD", "LD"), ("LT", "LT")])
def test_other_new_bmode_channels_match_the_naive_double_loop(pair):
    """The remaining three new channels (``DL``, ``LD``, ``LT``), same check."""
    wlm = MaskWlm("baseline_mask.fits", load_path=DATA)
    mask_alm = ducc0_map2alm(wlm.mask, lmax=LW, pol=False, iter=10)
    lmax = 2 * NSIDE

    a, b = pair
    reference = _naive_kernel(
        mask_alm,
        LW,
        ELL,
        ELL,
        lmax,
        _CENTRAL_FIELD[COUPLING_CHANNELS.index(a)],
        _PRIME_FIELD[COUPLING_CHANNELS.index(b)],
    )

    kernels = precompute_acc_kernels(
        wlm,
        None,
        centralell=ELL,
        ellprange=[ELL],
        nside=NSIDE,
        dryrun=True,
        grid="gl",
        lw=LW,
        pairs=[pair],
    )[ELL]

    assert set(kernels) == {f"{a}x{b}"}
    np.testing.assert_allclose(
        kernels[f"{a}x{b}"],
        reference,
        rtol=1e-11,
        atol=1e-14 * np.abs(reference).max(),
    )
