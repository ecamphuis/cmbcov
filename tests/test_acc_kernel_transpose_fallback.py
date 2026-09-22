r"""
Kernel-pair transpose fallback.

A level-1 (T/E-only) ACC run's Eq. 23 path loads exact ``(channel_1,
channel_2)`` kernel pairs (``Cov.get_acc_coupling_kernels`` via
``ACCStrategy.compute_covariance_term``). A level-3 (B-mode) precompute's
minimal pair set (``bmode_wick.required_kernel_pairs``) stores only one
orientation of several channel pairs a level-1 run needs -- e.g. it has
``DTxTD`` but not ``TDxDT`` -- so without a fallback, loading a B-mode
cache from a T/E-only run would raise ``OSError``.

By the kernel-transpose identity (docs/theory/bmode_kernels.md, :math:`\Theta^{ab\times cd}(L_1, L_2) = \Theta^{cd\times ab}(L_2,
L_1)`), the missing orientation is exactly the transpose of the one on disk,
at the same ``(ell, ell_prime)``. This pins that fallback: the array-level
identity (unit tests against a hand-written cache), and the end-to-end
result (a level-1 run on a level-3 precompute cache, and on a cache missing
only the natural orientation of the pairs it needs).
"""

import os
import shutil
import warnings

import numpy as np
import pytest

healpy = pytest.importorskip("healpy")

from cmbcov.approximations import (  # noqa: E402
    StrategyFactory,
    acc_cache,
)
from cmbcov.approximations.acc import (  # noqa: E402
    COUPLING_CHANNELS,
    acc_cached_kernel_size,
    precompute_acc_kernels,
)
from cmbcov.bmode_wick import required_kernel_pairs  # noqa: E402
from cmbcov.covariance import (  # noqa: E402
    Cov,
    CovarianceConfig,
    CovarianceMethod,
)
from cmbcov.keys import CovKey  # noqa: E402

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
NSIDE = 16
ELL = 16
LMAX = 48
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


def _te_cl(size):
    """TT power law, EE = 0.1 TT, TE = 0.5 sqrt(TT EE), as in test_acc_bmode_channels.py."""
    ell = np.arange(size)
    tt = np.zeros(size)
    tt[2:] = 1e-3 / (ell[2:] * (ell[2:] + 1)) ** 0.9
    ee = 0.1 * tt
    te = 0.5 * np.sqrt(tt * ee)
    return {"TT": tt, "EE": ee, "TE": te}


# --------------------------------------------------------------------------- #
# Array-level fallback, against a hand-written tiny cache
# --------------------------------------------------------------------------- #


def test_transposed_pair_loads_as_the_exact_transpose(tmp_path):
    save_dir = str(tmp_path)
    known = ["TT", "TD"]
    array = np.array([[1.0, 2.0, 5.0], [3.0, 4.0, 6.0], [7.0, 8.0, 9.0]])
    acc_cache.write_coupling_kernels(
        save_dir, {"TDxTT": array}, 5, 5, known, centralell=5, spectra=known, grid="gl"
    )

    loaded = acc_cache.load_coupling_kernels(
        save_dir, 5, 5, known, default_pairs=[("TT", "TD")], pairs=[("TT", "TD")]
    )
    np.testing.assert_array_equal(loaded[("TT", "TD")], array.T)


def test_natural_orientation_preferred_when_both_are_on_disk(tmp_path):
    """The natural file is used as is, not derived from the other's
    transpose, when both are on disk."""
    save_dir = str(tmp_path)
    known = ["TT", "TD"]
    natural = np.array([[1.0, 2.0], [3.0, 4.0]])
    # Deliberately not the transpose of `natural`, so a wrong fallback would
    # be caught by the comparison below.
    other = np.array([[9.0, 9.0], [9.0, 9.0]])
    acc_cache.write_coupling_kernels(
        save_dir,
        {"TTxTD": natural, "TDxTT": other},
        5,
        5,
        known,
        centralell=5,
        spectra=known,
        grid="gl",
    )

    loaded = acc_cache.load_coupling_kernels(
        save_dir, 5, 5, known, default_pairs=[("TT", "TD")], pairs=[("TT", "TD")]
    )
    np.testing.assert_array_equal(loaded[("TT", "TD")], natural)


def test_legacy_name_and_transpose_combine(tmp_path):
    """Fallback order is new name, legacy name, transposed new name,
    transposed legacy name -- exercised here with only the transposed-legacy
    file present."""
    save_dir = str(tmp_path)
    legacy_names = {"TD": "TE"}
    array = np.array([[1.0, 2.0], [3.0, 4.0]])
    # Written under the legacy label ("TE"), transposed orientation.
    acc_cache.write_coupling_kernels(
        save_dir,
        {"TExTT": array},
        5,
        5,
        ["TT", "TE"],
        centralell=5,
        spectra=["TT", "TE"],
        grid="gl",
    )

    loaded = acc_cache.load_coupling_kernels(
        save_dir,
        5,
        5,
        ["TT", "TD"],
        default_pairs=[("TT", "TD")],
        pairs=[("TT", "TD")],
        legacy_names=legacy_names,
    )
    np.testing.assert_array_equal(loaded[("TT", "TD")], array.T)


def test_missing_pair_error_names_it_when_neither_orientation_exists(tmp_path):
    save_dir = str(tmp_path)
    known = ["TT", "TD"]
    with pytest.raises(OSError, match="TTxTD"):
        acc_cache.load_coupling_kernels(
            save_dir, 5, 5, known, default_pairs=[("TT", "TD")], pairs=[("TT", "TD")]
        )


def test_acc_cached_kernel_size_accepts_transposed_file(tmp_path):
    save_dir = str(tmp_path)
    array = np.zeros((7, 7))
    acc_cache.write_coupling_kernels(
        save_dir,
        {"TDxTT": array},
        5,
        5,
        COUPLING_CHANNELS,
        centralell=5,
        spectra=COUPLING_CHANNELS,
        grid="gl",
    )

    size = acc_cached_kernel_size(save_dir, centralell=5, dmax=1, pairs=[("TT", "TD")])
    assert size == 7


# --------------------------------------------------------------------------- #
# End to end: a level-1 (T/E-only) run on a level-3 (B-mode) precompute cache
# --------------------------------------------------------------------------- #


def test_level1_run_on_bmode_precompute_cache_matches_full_square(tmp_path):
    full_dir = tmp_path / "full"
    b_dir = tmp_path / "bcache"
    full_dir.mkdir()

    precompute_acc_kernels(
        "baseline_mask.fits",
        str(full_dir),
        centralell=ELL,
        dmax=2,
        mask_path=DATA,
        nside=NSIDE,
        grid="healpix",
    )
    pairs18 = sorted(required_kernel_pairs(["TT", "EE", "TE", "BB", "TB", "EB"]))
    precompute_acc_kernels(
        "baseline_mask.fits",
        str(b_dir),
        centralell=ELL,
        dmax=2,
        mask_path=DATA,
        nside=NSIDE,
        grid="healpix",
        pairs=pairs18,
    )

    cov_full = _cov(str(full_dir), dmax=2)
    cov_b = _cov(str(b_dir), dmax=2)
    strategy_full = StrategyFactory.create_strategy(cov_full)
    strategy_b = StrategyFactory.create_strategy(cov_b)

    lmax_int_full = cov_full.acc_internal_lmax()
    assert lmax_int_full is not None
    cl = {FREQ + FREQ: _te_cl(lmax_int_full)}

    for stokes in (("T", "T", "T", "T"), ("E", "E", "E", "E"), ("T", "E", "T", "E")):
        key = CovKey(stokes, (FREQ,) * 4)

        spectra_full, identity_full = strategy_full.raw_block_inputs(key, cl)
        spectra_b, identity_b = strategy_b.raw_block_inputs(key, cl)
        assert identity_full["lmax_int"] == identity_b["lmax_int"]
        for label in spectra_full:
            np.testing.assert_array_equal(spectra_full[label], spectra_b[label])

        block_full = strategy_full.compute_covariance_term(key, cl)
        block_b = strategy_b.compute_covariance_term(key, cl)
        assert np.abs(block_full).max() > 0
        np.testing.assert_array_equal(block_full, block_b)


def test_level1_run_on_transposed_only_cache_matches_natural(tmp_path):
    """Delete the exact files a T/E-only TE,TE block's Eq. 23 path requests
    (``TTxDD``, ``TDxDT`` -- ``CovKey.key_to_cross_kernel`` new-name form),
    leaving only their transposes (``DDxTT``, ``DTxTD``) on disk, and check
    the assembled block is unchanged."""
    natural_dir = tmp_path / "natural"
    transposed_dir = tmp_path / "transposed"
    natural_dir.mkdir()

    precompute_acc_kernels(
        "baseline_mask.fits",
        str(natural_dir),
        centralell=ELL,
        dmax=1,
        mask_path=DATA,
        nside=NSIDE,
        grid="healpix",
    )
    shutil.copytree(natural_dir, transposed_dir)

    coupling_dir = transposed_dir / "covariance_coupling"
    removed = [
        name
        for name in os.listdir(coupling_dir)
        if name.startswith(("TTxDD_", "TDxDT_"))
    ]
    assert removed  # sanity: the natural files exist before removal
    for name in removed:
        os.remove(coupling_dir / name)
    # The transposes must still be there to fall back to.
    for name in os.listdir(coupling_dir):
        assert not name.startswith(("TTxDD_", "TDxDT_"))
    assert any(n.startswith("DDxTT_") for n in os.listdir(coupling_dir))
    assert any(n.startswith("DTxTD_") for n in os.listdir(coupling_dir))

    cov_natural = _cov(str(natural_dir), dmax=1)
    cov_transposed = _cov(str(transposed_dir), dmax=1)
    strategy_natural = StrategyFactory.create_strategy(cov_natural)
    strategy_transposed = StrategyFactory.create_strategy(cov_transposed)

    lmax_int = cov_natural.acc_internal_lmax()
    cl = {FREQ + FREQ: _te_cl(lmax_int)}
    key = CovKey(("T", "E", "T", "E"), (FREQ,) * 4)

    block_natural = strategy_natural.compute_covariance_term(key, cl)
    block_transposed = strategy_transposed.compute_covariance_term(key, cl)
    assert np.abs(block_natural).max() > 0
    np.testing.assert_array_equal(block_natural, block_transposed)
