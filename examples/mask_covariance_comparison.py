#!/usr/bin/env python3
"""
Exact vs. approximate pseudo-:math:`C_\\ell` covariance on a real footprint.

This script answers a single question: on a realistic mask, how accurate is
each implemented approximation (NKA, INKA, and optionally ACC) compared with
the exact pseudo-:math:`C_\\ell` covariance of Camphuis et al. (2022, Sect. 3)?
Every accuracy number quoted for this package before now comes from a
40-degree toy cap at ``lmax < 60`` (:mod:`cmbcov.exact`);
this script validates against a realistic, survey-shaped footprint instead.

Pipeline
--------
1. Load a CMB TT spectrum from a CAMB ``lensedCls`` file and add an
   atmosphere-like TT noise model to form ``C_l^total``.
2. Compute the *exact* GL-grid pseudo-``C_l`` covariance
   (:func:`cmbcov.exact.exact_covariance_row`) for the
   given mask, row by row, checkpointed so a partial or interrupted run can
   be resumed.
3. Compute the same covariance with NKA, INKA, and (optionally) ACC through
   the package's own strategy objects (:mod:`cmbcov.
   approximations`), calling ``compute_covariance_term`` directly so the
   result is the same raw pseudo-``C_l`` quantity the exact code produces --
   *not* the PolSpice-deconvolved, binned output of
   ``Cov.compute_covariance_matrix``, which answers a different question.
4. Compare: diagonal and near-diagonal ratios, a bandpower-binned comparison
   (the number that matters for a likelihood), and the correlation-matrix
   difference (an approximation can get the diagonal right and the
   off-diagonal structure wrong).

Usage
-----
::

    .venv/bin/python examples/mask_covariance_comparison.py /path/to/mask.fits \\
        --lmax 500 --lw 250 --out results

Develop and test against ``tests/data/baseline_mask.fits`` at small ``lmax``
(seconds, not the ~1.5 hours a realistic mask needs at ``lmax=500``)::

    .venv/bin/python examples/mask_covariance_comparison.py \\
        tests/data/baseline_mask.fits --lmax 48 --lw 24 --out /tmp/results
"""

import argparse
import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass

import healpy as hp
import numpy as np

from cmbcov.approximations import StrategyFactory
from cmbcov.approximations.acc import (
    coupling_ellprange,
    precompute_acc_kernels,
)
from cmbcov.binning import BinningManager
from cmbcov.covariance import Cov, CovarianceConfig, CovarianceMethod
from cmbcov.exact import exact_covariance_row
from cmbcov.grid import gl_minimal_lmax
from cmbcov.keys import CovKey, CovKeys
from cmbcov.sht import ducc0_map2alm
from cmbcov.utils.file_utils import get_git_revision_short_hash

#: Single frequency label used throughout; the CovKeys/Cov machinery is
#: multi-frequency but this comparison is explicitly single-frequency TT.
FREQ = "SURVEY"

DEFAULT_SPECTRUM_FILE = os.path.join(
    os.path.dirname(__file__),
    "..",
    "tests",
    "data",
    "planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lensedCls.dat",
)

__all__ = [
    "load_cmb_cl",
    "noise_cl",
    "exact_covariance_path",
    "compute_exact_covariance",
    "self_consistency_checks",
    "build_cov",
    "compute_nka_inka_term",
    "acc_precompute",
    "compute_acc_term",
    "diagonal_ratio_bands",
    "offdiagonal_ratio_bands",
    "binned_comparison",
    "correlation_difference",
]


# --------------------------------------------------------------------------- #
# 1. Spectrum loading
# --------------------------------------------------------------------------- #
def load_cmb_cl(path: str, lmax: int) -> np.ndarray:
    """
    Read a CAMB ``lensedCls``-format spectrum file into ``C_l^TT``.

    Parameters
    ----------
    path : str
        Path to a CAMB file with whitespace-separated columns
        ``L TT EE BB TE``, starting at ``L=2``, holding
        :math:`D_\\ell = \\ell(\\ell+1) C_\\ell / 2\\pi` in :math:`\\mu K^2`.
    lmax : int
        Multipole up to which the returned array is truncated or zero-padded.

    Returns
    -------
    ndarray
        ``C_l^TT`` in :math:`\\mu K^2`, shape ``(lmax + 1,)``, with
        ``C_0 = C_1 = 0``.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist, naming the missing file explicitly.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"CMB spectrum file not found: {path!r}. This must be a CAMB "
            "lensedCls file with columns 'L TT EE BB TE' (D_l in uK^2)."
        )

    data = np.loadtxt(path)
    ell_file = data[:, 0].astype(int)
    dl_tt = data[:, 1]

    cl = np.zeros(lmax + 1)
    # D_l -> C_l = 2 pi D_l / (l (l+1)); only fill multipoles present both in
    # the file and within [0, lmax] -- higher multipoles are zero-padded,
    # lower ones (l=0,1, no monopole/dipole in a CAMB file) stay zero.
    keep = (ell_file >= 2) & (ell_file <= lmax)
    ell_keep = ell_file[keep]
    cl[ell_keep] = 2.0 * np.pi * dl_tt[keep] / (ell_keep * (ell_keep + 1))

    print(f"Loaded CMB TT spectrum from {path}")
    for ell_check in (2, 100, 220, 500, lmax):
        if 0 <= ell_check <= lmax:
            print(f"    C_l^TT(l={ell_check}) = {cl[ell_check]:.6e} uK^2")
    dl_loaded = np.zeros(lmax + 1)
    dl_loaded[ell_keep] = dl_tt[keep]
    peak = int(np.argmax(dl_loaded))
    print(f"    D_l^TT peak at l = {peak} (expect near l ~ 220)")
    return cl


# --------------------------------------------------------------------------- #
# 2. Noise model
# --------------------------------------------------------------------------- #
def noise_cl(lmax: int, depth: float, lknee: float, alpha: float) -> np.ndarray:
    """
    Atmosphere-like TT noise spectrum, white at high ``l``.

    .. math::

        N_\\ell = N_{\\rm white} \\left(1 + (\\ell_{\\rm knee}/\\ell)^\\alpha\\right),
        \\qquad
        N_{\\rm white} = \\left(\\frac{\\Delta_T [\\mu K\\text{-arcmin}] \\, \\pi}
        {10800}\\right)^2 \\; \\mu K^2 .

    Parameters
    ----------
    lmax : int
        Maximum multipole.
    depth : float
        Map depth :math:`\\Delta_T` in :math:`\\mu K`-arcmin.
    lknee : float
        Knee multipole of the ``1/f``-like atmospheric term.
    alpha : float
        Power-law index of the atmospheric term.

    Returns
    -------
    ndarray
        ``N_l``, shape ``(lmax + 1,)``, with ``N_0 = N_1 = 0``.
    """
    n_white = (depth * np.pi / 10800.0) ** 2
    nl = np.zeros(lmax + 1)
    ell = np.arange(2, lmax + 1)
    nl[2:] = n_white * (1.0 + (lknee / ell) ** alpha)
    return nl


# --------------------------------------------------------------------------- #
# 3. Exact covariance, GL path, resumable
# --------------------------------------------------------------------------- #
def exact_covariance_path(out_dir: str, lmax: int, lw: int) -> str:
    """Path of the cached exact covariance matrix for ``(lmax, lw)``."""
    return os.path.join(out_dir, f"exact_lmax{lmax}_lw{lw}.npy")


def _checkpoint_path(out_dir: str, lmax: int, lw: int) -> str:
    return exact_covariance_path(out_dir, lmax, lw) + ".partial.npz"


def _sidecar_path(out_dir: str, lmax: int, lw: int) -> str:
    return exact_covariance_path(out_dir, lmax, lw).replace(".npy", ".json")


def compute_exact_covariance(
    mask_map: np.ndarray,
    nside: int,
    cl_total: np.ndarray,
    lmax: int,
    lw: int,
    rows: Sequence[int],
    out_dir: str,
    mask_file: str,
    noise_params: dict[str, float],
    overwrite: bool = False,
    nthreads: int | None = None,
    checkpoint_every: int = 10,
) -> np.ndarray:
    """
    Exact GL-grid pseudo-``C_l`` covariance, row by row, resumable.

    Skips recomputation entirely if the final ``.npy`` file already exists
    (unless ``overwrite``); otherwise resumes from a ``.partial.npz``
    checkpoint written every ``checkpoint_every`` rows, so an interrupted
    ~1.5 hour run does not have to restart from zero.

    Parameters
    ----------
    mask_map : ndarray
        HEALPix mask map (RING).
    nside : int
        Mask resolution (unused for the GL path once the mask alm is built,
        kept for the JSON sidecar).
    cl_total : ndarray
        ``C_l^CMB + N_l``, shape ``(lmax + 1,)``.
    lmax : int
        Maximum multipole of the pseudo-spectrum.
    lw : int
        Band-limit of the mask on the GL grid.
    rows : sequence of int
        Multipoles :math:`\\ell'` whose columns are computed. Columns not
        requested are left at zero (see
        :func:`cmbcov.exact.exact_covariance`).
    out_dir : str
        Directory for the cached matrix, checkpoint and JSON sidecar.
    mask_file, noise_params : misc
        Recorded in the JSON sidecar.
    overwrite : bool
        Recompute even if a finished result is cached.
    nthreads : int, optional
        Threads for ducc0.
    checkpoint_every : int
        Rows between checkpoint writes.

    Returns
    -------
    ndarray
        ``Sigma[l, l']``, shape ``(lmax + 1, lmax + 1)``.
    """
    final_path = exact_covariance_path(out_dir, lmax, lw)
    if os.path.exists(final_path) and not overwrite:
        print(f"Exact covariance already cached at {final_path}; skipping.")
        return np.load(final_path)

    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = _checkpoint_path(out_dir, lmax, lw)

    sigma = np.zeros((lmax + 1, lmax + 1))
    done: set = set()
    if os.path.exists(ckpt_path) and not overwrite:
        chk = np.load(ckpt_path)
        if chk["sigma"].shape == sigma.shape:
            sigma = chk["sigma"]
            done = {int(r) for r in chk["done_rows"]}
            print(f"Resuming exact covariance from checkpoint: {len(done)} rows done.")

    # Band-limited mask alm, built once and reused for every row: passing it
    # (rather than the pixel map) to exact_covariance_row avoids repeating the
    # iter=10 map2alm that builds it (module docstring of exact.py).
    mask_alm = ducc0_map2alm(
        np.asarray(mask_map, dtype=float), lmax=lw, pol=False, iter=10
    )

    todo = [r for r in rows if r not in done]
    n_total = len(todo)
    t_start = time.time()
    for i, ellp in enumerate(todo):
        sigma[:, ellp] = exact_covariance_row(
            mask_alm, cl_total, ellp, lmax, grid="gl", lw=lw, nthreads=nthreads
        )
        done.add(ellp)
        n_done = i + 1
        elapsed = time.time() - t_start
        rate = elapsed / n_done
        remaining = rate * (n_total - n_done)
        print(
            f"  row l'={ellp:5d}  ({n_done}/{n_total})  "
            f"elapsed {elapsed:8.1f}s  est. remaining {remaining:8.1f}s"
        )
        if n_done % checkpoint_every == 0 or n_done == n_total:
            np.savez(
                ckpt_path, sigma=sigma, done_rows=np.array(sorted(done), dtype=int)
            )

    # Only a run over every l' = 0..lmax produces the canonical, complete
    # matrix; that is the only case allowed to write the final .npy (which a
    # later run treats as "already done, skip"). A --rows subset run (a
    # deliberately cheap partial pass, item 3 of the brief) must not leave
    # an incomplete matrix at that path, or a later full run would silently
    # load it and think it was finished.
    is_complete = len(done) == lmax + 1
    if is_complete:
        np.save(final_path, sigma)
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
    else:
        np.savez(ckpt_path, sigma=sigma, done_rows=np.array(sorted(done), dtype=int))
        print(
            f"Partial run: {len(done)}/{lmax + 1} rows computed and checkpointed at "
            f"{ckpt_path}. Re-run with the default (full) --rows to complete and "
            "cache the final matrix."
        )

    meta = {
        "lmax": lmax,
        "lw": lw,
        "lmax_grid_bandlimit": gl_minimal_lmax(lmax, lw),
        "noise": noise_params,
        "mask_file": os.path.abspath(mask_file),
        "nside": int(nside),
        "rows_computed": sorted(done),
        "complete": is_complete,
        "git_hash": get_git_revision_short_hash(),
        "wall_time_seconds": time.time() - t_start,
    }
    with open(_sidecar_path(out_dir, lmax, lw), "w") as handle:
        json.dump(meta, handle, indent=2)

    return sigma


# --------------------------------------------------------------------------- #
# 4. Self-consistency checks
# --------------------------------------------------------------------------- #
def self_consistency_checks(sigma: np.ndarray, rows: Sequence[int]) -> None:
    """
    Print symmetry, positivity and diagonal-sign checks on ``sigma``.

    These cost nothing and catch a broken run before any comparison is
    trusted. If ``rows`` is a proper subset of ``0..lmax`` the matrix is only
    complete on the ``rows x rows`` sub-block (columns not requested are left
    at zero; see :func:`cmbcov.exact.exact_covariance`),
    so the checks are restricted to that sub-block.
    """
    lmax = sigma.shape[0] - 1
    idx = np.array(sorted(rows))
    partial = idx.size != lmax + 1
    block = sigma[np.ix_(idx, idx)]
    scale = np.abs(block).max()

    sym_err = np.abs(block - block.T).max() / scale if scale > 0 else 0.0
    eigenvalues = np.linalg.eigvalsh((block + block.T) / 2)
    min_over_max = eigenvalues.min() / eigenvalues.max()
    diag = np.diag(block)
    all_diag_positive = bool(np.all(diag > 0))

    tag = " (restricted to computed rows x rows sub-block)" if partial else ""
    print(f"Self-consistency checks on the exact covariance{tag}:")
    print(f"    symmetry  |S - S.T|.max() / |S|.max()      = {sym_err:.3e}")
    print(
        f"    positivity  lambda_min / lambda_max          = {min_over_max:.3e} "
        "(legitimately near-singular; only meaningfully-negative values are a bug)"
    )
    print(f"    diagonal strictly positive everywhere         = {all_diag_positive}")


# --------------------------------------------------------------------------- #
# 5. The approximations
# --------------------------------------------------------------------------- #
def build_cov(
    mask_name: str,
    mask_path: str,
    method: CovarianceMethod,
    lmax: int,
    save_dir: str | None = None,
    dmax: int | None = None,
    centralell: int | None = None,
) -> Cov:
    """
    Build a ``Cov`` object whose arrays cover ``l = 0..lmax`` inclusive.

    The package's own convention is that ``CovarianceConfig.lmax`` is an
    array *length* (spectra and kernels run ``0..lmax-1``); to match
    ``exact_covariance``'s inclusive convention (arrays of length
    ``lmax + 1``, indices ``0..lmax``) this passes ``lmax + 1`` to the
    package.
    """
    config = CovarianceConfig(
        method=method,
        lmax=lmax + 1,
        dmax=dmax,
        centralell=centralell,
    )
    return Cov(mask_name, config, save_dir=save_dir, mask_path=mask_path)


def _single_tt_key(cov: Cov) -> tuple[CovKeys, CovKey]:
    """The one CovKey for single-frequency, TT-only, auto-covariance."""
    keys = CovKeys(["T"], [FREQ])
    (cov_key,) = keys.positions.keys()
    return keys, cov_key


def compute_nka_inka_term(
    mask_name: str,
    mask_path: str,
    method: CovarianceMethod,
    lmax: int,
    cl_total: np.ndarray,
) -> np.ndarray:
    """
    Raw pseudo-``C_l`` covariance term for NKA or INKA, shape
    ``(lmax + 1, lmax + 1)``, directly comparable to
    :func:`~cmbcov.exact.exact_covariance`.

    Calls ``strategy.compute_covariance_term`` directly rather than
    ``Cov.compute_covariance_matrix``: the latter also applies the PolSpice
    deconvolution and bandpower binning (``CovariancePostProcessor``), which
    answers "what does the likelihood see" rather than "how good is this
    approximation of the same pseudo-``C_l`` quantity the exact code
    computes" -- the question this script asks.
    """
    cov = build_cov(mask_name, mask_path, method, lmax)
    strategy = StrategyFactory.create_strategy(cov)
    _, cov_key = _single_tt_key(cov)
    cl_dict = {f"{FREQ}{FREQ}": {"TT": cl_total[: cov.lmax]}}
    return strategy.compute_covariance_term(cov_key, cl_dict)


def acc_precompute(
    mask_name: str,
    mask_path: str,
    lmax: int,
    lw: int,
    centralell: int,
    dmax: int,
    save_dir: str,
    overwrite: bool = False,
    verbose: bool = True,
) -> float:
    """
    Precompute and cache the ACC covariance-coupling kernels.

    Skipped (cost 0) if every expected kernel file already exists under
    ``save_dir/covariance_coupling`` and ``overwrite`` is not set.

    Returns
    -------
    float
        Wall-clock seconds spent (0.0 if the precompute was skipped).
    """
    ellprange = coupling_ellprange(centralell, dmax)
    stokes_specs = ["TT", "EE", "TE", "ET"]
    expected = [
        os.path.join(
            save_dir, "covariance_coupling", f"{s1}x{s2}_{centralell}x{ellp}.npy"
        )
        for ellp in ellprange
        for s1 in stokes_specs
        for s2 in stokes_specs
    ]
    if not overwrite and all(os.path.exists(p) for p in expected):
        print("ACC coupling kernels already cached; skipping precompute.")
        return 0.0

    print(
        f"Precomputing ACC coupling kernels for centralell={centralell}, "
        f"dmax={dmax} (grid=gl, lw={lw}); this is the expensive ACC step."
    )
    t0 = time.time()
    precompute_acc_kernels(
        mask_name,
        save_dir,
        centralell=centralell,
        dmax=dmax,
        mask_path=mask_path,
        grid="gl",
        lw=lw,
        verbose=verbose,
    )
    dt = time.time() - t0
    print(f"ACC precompute finished in {dt:.1f}s.")
    return dt


def compute_acc_term(
    mask_name: str,
    mask_path: str,
    lmax: int,
    centralell: int,
    dmax: int,
    save_dir: str,
    cl_total: np.ndarray,
) -> np.ndarray:
    """
    Raw ACC pseudo-``C_l`` covariance term, shape ``(lmax + 1, lmax + 1)``.

    Requires :func:`acc_precompute` to have already cached the coupling
    kernels under ``save_dir``. Entries with ``|l - l'| >= dmax`` are exactly
    zero by construction (module docstring of
    :mod:`cmbcov.approximations.acc`); the comparison
    functions below must be restricted to the computed band.
    """
    cov = build_cov(
        mask_name,
        mask_path,
        CovarianceMethod.ACC,
        lmax,
        save_dir=save_dir,
        dmax=dmax,
        centralell=centralell,
    )
    strategy = StrategyFactory.create_strategy(cov)
    _, cov_key = _single_tt_key(cov)
    # ACC reads the spectra to lmax_int = lmax + max(0, S - 1 - centralell)
    # (Cov.acc_internal_lmax) so that no coupling window is cut at lmax. The
    # exact reference is the covariance of a sky band-limited at lmax, so the
    # matching spectrum is zero past lmax.
    lmax_int = cov.acc_internal_lmax() or cov.lmax
    cl_read = np.zeros(max(lmax_int, cov.lmax))
    cl_read[: cov.lmax] = cl_total[: cov.lmax]
    cl_dict = {f"{FREQ}{FREQ}": {"TT": cl_read}}
    return strategy.compute_covariance_term(cov_key, cl_dict)


# --------------------------------------------------------------------------- #
# 6. The comparison
# --------------------------------------------------------------------------- #
def _band_mask(lmax: int, dmax: int | None) -> np.ndarray | None:
    """Boolean ``|l - l'| < dmax`` mask, or ``None`` if ``dmax`` is not set."""
    if dmax is None:
        return None
    ell = np.arange(lmax + 1)
    return np.abs(ell[:, None] - ell[None, :]) < dmax


def diagonal_ratio_bands(
    approx: np.ndarray, exact: np.ndarray, band_width: int = 50
) -> list[tuple[int, int, float, float, float]]:
    """
    Ratio approx/exact on the diagonal, summarised in bands of ``band_width``
    multipoles as (band_lo, band_hi, median, p16, p84).
    """
    lmax = exact.shape[0] - 1
    diag_exact = np.diag(exact)
    diag_approx = np.diag(approx)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(diag_exact != 0, diag_approx / diag_exact, np.nan)

    rows = []
    for lo in range(0, lmax + 1, band_width):
        hi = min(lo + band_width, lmax + 1)
        band = ratio[lo:hi]
        band = band[np.isfinite(band)]
        if band.size == 0:
            continue
        rows.append(
            (
                lo,
                hi - 1,
                float(np.median(band)),
                float(np.percentile(band, 16)),
                float(np.percentile(band, 84)),
            )
        )
    return rows


def offdiagonal_ratio_bands(
    approx: np.ndarray,
    exact: np.ndarray,
    lags: Sequence[int] = (1, 2, 5, 10),
    band_width: int = 50,
    dmax: int | None = None,
) -> list[tuple[int, int, int, float, float, float]]:
    """
    Same as :func:`diagonal_ratio_bands` for the off-diagonals ``|l - l'| =
    lag``. Lags at or beyond ``dmax`` (ACC's zero band) are skipped, since
    ACC is identically zero there by construction, not "inaccurate".
    """
    rows = []
    for lag in lags:
        if dmax is not None and lag >= dmax:
            continue
        diag_exact = np.diag(exact, lag)
        diag_approx = np.diag(approx, lag)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(diag_exact != 0, diag_approx / diag_exact, np.nan)
        n = ratio.size  # indexed by the smaller multipole l = 0..lmax-lag
        for lo in range(0, n, band_width):
            hi = min(lo + band_width, n)
            band = ratio[lo:hi]
            band = band[np.isfinite(band)]
            if band.size == 0:
                continue
            rows.append(
                (
                    lag,
                    lo,
                    hi - 1,
                    float(np.median(band)),
                    float(np.percentile(band, 16)),
                    float(np.percentile(band, 84)),
                )
            )
    return rows


def binned_comparison(
    approx: np.ndarray, exact: np.ndarray, bin_width: int = 50
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bandpower-binned ratio: bin ``Sigma`` with ``BinningManager`` into
    ``B Sigma B.T`` and report the ratio approx/exact on the binned diagonal
    and first off-diagonal -- the number that matters for a likelihood.

    Returns
    -------
    ell_centers, diag_ratio, offdiag1_ratio : ndarray
    """
    lmax = exact.shape[0] - 1
    binning = BinningManager(lmax + 1, lmin=0)
    bin_matrix, num_bins = binning.create_bin_matrix(
        bin_width, flatten_with_ell_factor=False
    )
    ell_centers = bin_matrix @ np.arange(lmax + 1)

    binned_exact = bin_matrix @ exact @ bin_matrix.T
    binned_approx = bin_matrix @ approx @ bin_matrix.T

    with np.errstate(divide="ignore", invalid="ignore"):
        diag_ratio = np.where(
            np.diag(binned_exact) != 0,
            np.diag(binned_approx) / np.diag(binned_exact),
            np.nan,
        )
        off_exact = np.diag(binned_exact, 1)
        off_approx = np.diag(binned_approx, 1)
        offdiag1_ratio = np.where(off_exact != 0, off_approx / off_exact, np.nan)

    return ell_centers[:num_bins], diag_ratio, offdiag1_ratio


def correlation_difference(
    approx: np.ndarray, exact: np.ndarray, dmax: int | None = None
) -> tuple[float, float]:
    """
    Max and median absolute difference of the correlation matrices.

    An approximation can reproduce the diagonal (amplitude) correctly while
    getting the mask-induced mode correlations wrong; this is the check that
    catches that. Restricted to ``|l - l'| < dmax`` when given (ACC's
    identically-zero band would otherwise dominate the statistic with a
    structural, not an accuracy, effect). Uses ``atol=0`` throughout --
    these covariances are ~1e-18 in raw units, where the default
    ``np.allclose`` atol makes any comparison vacuous.
    """

    def _corr(matrix: np.ndarray) -> np.ndarray:
        d = np.diag(matrix)
        safe = np.where(d > 0, d, 1.0)
        norm = np.sqrt(np.outer(safe, safe))
        corr = matrix / norm
        corr[(d <= 0)[:, None] | (d <= 0)[None, :]] = np.nan
        return corr

    diff = np.abs(_corr(approx) - _corr(exact))
    mask = _band_mask(exact.shape[0] - 1, dmax)
    if mask is not None:
        diff = np.where(mask, diff, np.nan)

    finite = diff[np.isfinite(diff)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.max(finite)), float(np.median(finite))


# --------------------------------------------------------------------------- #
# CSV / print helpers
# --------------------------------------------------------------------------- #
def _print_and_save_diagonal(rows: list[tuple], out_csv: str, label: str) -> None:
    print(f"  {label} diagonal ratio (approx/exact), by band of 50 multipoles:")
    print("    band            median      p16       p84")
    with open(out_csv, "w") as handle:
        handle.write("band_lo,band_hi,median,p16,p84\n")
        for lo, hi, med, p16, p84 in rows:
            print(f"    [{lo:4d},{hi:4d}]   {med:8.4f}  {p16:8.4f}  {p84:8.4f}")
            handle.write(f"{lo},{hi},{med},{p16},{p84}\n")


def _print_and_save_offdiag(rows: list[tuple], out_csv: str, label: str) -> None:
    print(f"  {label} off-diagonal ratio (approx/exact):")
    print("    lag  band            median      p16       p84")
    with open(out_csv, "w") as handle:
        handle.write("lag,band_lo,band_hi,median,p16,p84\n")
        for lag, lo, hi, med, p16, p84 in rows:
            print(
                f"    {lag:3d}  [{lo:4d},{hi:4d}]   {med:8.4f}  {p16:8.4f}  {p84:8.4f}"
            )
            handle.write(f"{lag},{lo},{hi},{med},{p16},{p84}\n")


def _print_and_save_binned(
    ell_centers: np.ndarray,
    diag_ratio: np.ndarray,
    offdiag1_ratio: np.ndarray,
    out_csv: str,
    label: str,
) -> None:
    print(f"  {label} binned (width 50) diagonal and first-off-diagonal ratio:")
    print("    ell_center      diag ratio   offdiag1 ratio")
    with open(out_csv, "w") as handle:
        handle.write("ell_center,diag_ratio,offdiag1_ratio\n")
        n = len(ell_centers)
        for i in range(n):
            off = offdiag1_ratio[i] if i < len(offdiag1_ratio) else np.nan
            print(f"    {ell_centers[i]:10.1f}    {diag_ratio[i]:10.4f}   {off:10.4f}")
            handle.write(f"{ell_centers[i]},{diag_ratio[i]},{off}\n")


def compare_and_report(
    name: str,
    approx: np.ndarray,
    exact: np.ndarray,
    out_dir: str,
    dmax: int | None = None,
) -> None:
    """Run every comparison in item 6 of the brief for one approximation."""
    print(f"\n=== {name} vs. exact ===")
    if dmax is not None:
        print(
            f"    ACC is identically zero for |l - l'| >= {dmax}; off-diagonal "
            "and correlation comparisons are restricted to that band. The "
            "binned (width 50) comparison mixes many zeroed entries into each "
            "bandpower and is not expected to be accurate for this reason."
        )

    diag_rows = diagonal_ratio_bands(approx, exact)
    _print_and_save_diagonal(
        diag_rows, os.path.join(out_dir, f"diagonal_ratio_{name}.csv"), name
    )

    offdiag_rows = offdiagonal_ratio_bands(approx, exact, dmax=dmax)
    _print_and_save_offdiag(
        offdiag_rows, os.path.join(out_dir, f"offdiag_ratio_{name}.csv"), name
    )

    ell_centers, diag_ratio, offdiag1_ratio = binned_comparison(approx, exact)
    _print_and_save_binned(
        ell_centers,
        diag_ratio,
        offdiag1_ratio,
        os.path.join(out_dir, f"binned_ratio_{name}.csv"),
        name,
    )

    corr_max, corr_median = correlation_difference(approx, exact, dmax=dmax)
    print(
        f"  {name} correlation-matrix |difference|: max={corr_max:.4f}  median={corr_median:.4f}"
    )
    with open(os.path.join(out_dir, f"correlation_diff_{name}.csv"), "w") as handle:
        handle.write("max_abs_diff,median_abs_diff\n")
        handle.write(f"{corr_max},{corr_median}\n")


# --------------------------------------------------------------------------- #
# Optional plotting
# --------------------------------------------------------------------------- #
def make_plot(
    results: dict[str, tuple[np.ndarray, int | None]],
    exact: np.ndarray,
    out_dir: str,
) -> None:
    """Save a small PNG with per-approximation diagonal ratios, if matplotlib
    is importable; skip cleanly otherwise."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not importable; skipping --plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ell = np.arange(exact.shape[0])

    for name, (approx, _dmax) in results.items():
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(
                np.diag(exact) != 0, np.diag(approx) / np.diag(exact), np.nan
            )
        axes[0].plot(ell, ratio, label=name)

        ell_c, diag_ratio, _ = binned_comparison(approx, exact)
        axes[1].plot(ell_c, diag_ratio, "o-", label=name)

    axes[0].axhline(1.0, color="k", lw=0.8, ls="--")
    axes[0].set_xlabel(r"$\ell$")
    axes[0].set_ylabel("diagonal ratio (approx/exact)")
    axes[0].legend()
    axes[0].set_title("Per-multipole diagonal ratio")

    axes[1].axhline(1.0, color="k", lw=0.8, ls="--")
    axes[1].set_xlabel(r"$\ell$ (bandpower centre)")
    axes[1].set_ylabel("binned diagonal ratio (approx/exact)")
    axes[1].legend()
    axes[1].set_title("Bandpower (width 50) diagonal ratio")

    fig.tight_layout()
    png_path = os.path.join(out_dir, "diagonal_ratio_comparison.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {png_path}")


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
def _parse_rows(spec: str | None, lmax: int) -> list[int]:
    """Parse ``--rows`` as a comma-separated list of ints and/or ``a-b``
    inclusive ranges. ``None`` means every multipole ``0..lmax``."""
    if spec is None:
        return list(range(lmax + 1))
    rows: set = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo, hi = token.split("-")
            rows.update(range(int(lo), int(hi) + 1))
        else:
            rows.add(int(token))
    return sorted(r for r in rows if 0 <= r <= lmax)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the exact pseudo-Cl TT covariance against NKA/INKA/ACC "
            "on a realistic mask."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("mask", help="Path to a HEALPix mask FITS file.")
    parser.add_argument(
        "--lmax", type=int, default=500, help="Maximum multipole of the comparison."
    )
    parser.add_argument(
        "--lw", type=int, required=True, help="Band-limit of the mask on the GL grid."
    )
    parser.add_argument(
        "--out", required=True, help="Output directory for cached results and CSVs."
    )
    parser.add_argument(
        "--rows",
        type=str,
        default=None,
        help=(
            "Subset of l' columns to compute for the exact covariance, e.g. "
            "'0-100,200,300-310'. Default: all 0..lmax. Lets the author run a "
            "cheap partial pass before the full ~1.5 hour computation."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute the exact covariance / ACC precompute even if cached.",
    )
    parser.add_argument(
        "--spectrum-file",
        type=str,
        default=DEFAULT_SPECTRUM_FILE,
        help="CAMB lensedCls file with columns 'L TT EE BB TE' (D_l, uK^2).",
    )
    parser.add_argument(
        "--depth", type=float, default=10.0, help="Map depth in uK-arcmin."
    )
    parser.add_argument(
        "--lknee", type=float, default=200.0, help="Atmospheric noise knee multipole."
    )
    parser.add_argument(
        "--alpha", type=float, default=2.0, help="Atmospheric noise power-law index."
    )
    parser.add_argument(
        "--acc", action="store_true", help="Also compute the ACC approximation."
    )
    parser.add_argument(
        "--centralell",
        type=int,
        default=None,
        help="ACC central multipole. Defaults to lmax // 2.",
    )
    parser.add_argument(
        "--dmax",
        type=int,
        default=10,
        help=(
            "ACC diagonal-band half-width. ACC returns exactly zero for "
            "|l - l'| >= dmax, so the comparison is restricted to that band."
        ),
    )
    parser.add_argument(
        "--nthreads", type=int, default=None, help="Threads for ducc0 transforms."
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save a PNG of the diagonal ratios if matplotlib is available.",
    )
    return parser


@dataclass
class NoiseParams:
    depth: float
    lknee: float
    alpha: float

    def as_dict(self) -> dict[str, float]:
        return {"depth_uk_arcmin": self.depth, "lknee": self.lknee, "alpha": self.alpha}


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    centralell = args.centralell if args.centralell is not None else args.lmax // 2

    # --- 1 & 2: spectrum + noise -> C_l^total -----------------------------
    cl_cmb = load_cmb_cl(args.spectrum_file, args.lmax)
    noise = NoiseParams(args.depth, args.lknee, args.alpha)
    nl = noise_cl(args.lmax, noise.depth, noise.lknee, noise.alpha)
    # C_l^total assumes homogeneous (isotropic) instrument noise masked by
    # the same W as the signal -- the single-mask limitation of Cov, which
    # has no separate noise mask/hit-count map.
    cl_total = cl_cmb + nl
    print(f"White noise level: {nl[-1]:.6e} uK^2 (Delta_T={noise.depth} uK-arcmin)")

    # --- 3: exact covariance ------------------------------------------------
    mask_map = hp.read_map(args.mask)
    nside = hp.get_nside(mask_map)
    rows = _parse_rows(args.rows, args.lmax)
    print(f"\nComputing exact covariance for {len(rows)}/{args.lmax + 1} rows...")
    exact = compute_exact_covariance(
        mask_map,
        nside,
        cl_total,
        args.lmax,
        args.lw,
        rows,
        args.out,
        args.mask,
        noise.as_dict(),
        overwrite=args.overwrite,
        nthreads=args.nthreads,
    )

    # --- 4: self-consistency -------------------------------------------------
    self_consistency_checks(exact, rows)

    if len(rows) != args.lmax + 1:
        print(
            "\nPartial run (--rows subset): skipping the NKA/INKA/ACC "
            "comparison, which needs the full matrix. Re-run with the full "
            "row range (the default) for the comparison."
        )
        return

    # --- 5 & 6: approximations and comparison --------------------------------
    mask_basename = os.path.basename(args.mask)
    mask_dir = os.path.dirname(args.mask) or "."

    results: dict[str, tuple[np.ndarray, int | None]] = {}

    print("\nComputing NKA covariance term...")
    nka = compute_nka_inka_term(
        mask_basename, mask_dir, CovarianceMethod.NKA, args.lmax, cl_total
    )
    results["NKA"] = (nka, None)

    print("Computing INKA covariance term...")
    inka = compute_nka_inka_term(
        mask_basename, mask_dir, CovarianceMethod.INKA, args.lmax, cl_total
    )
    results["INKA"] = (inka, None)

    if args.acc:
        acc_dt = acc_precompute(
            mask_basename,
            mask_dir,
            args.lmax,
            args.lw,
            centralell,
            args.dmax,
            args.out,
            overwrite=args.overwrite,
        )
        print(f"ACC precompute cost: {acc_dt:.1f}s (0.0 means it was cached).")
        acc = compute_acc_term(
            mask_basename,
            mask_dir,
            args.lmax,
            centralell,
            args.dmax,
            args.out,
            cl_total,
        )
        results["ACC"] = (acc, args.dmax)

    for name, (approx, dmax) in results.items():
        compare_and_report(name, approx, exact, args.out, dmax=dmax)

    if args.plot:
        make_plot(results, exact, args.out)

    print(f"\nAll CSV comparisons written to {args.out}")


if __name__ == "__main__":
    main()
