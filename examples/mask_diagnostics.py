#!/usr/bin/env python3
"""
Characterise a mask before running the exact GL covariance on it.

Context (see the module docstrings of
``cmbcov.mask``/``grid``/``sht``): on the GL branch the
mask enters the covariance exactly once, as an alm truncated at a band-limit
``Lw``. Everything downstream of that truncation -- the coupling kernels, the
covariance itself -- is then exact to machine precision. So ``Lw`` is the one
place accuracy is spent, and the cost of the whole run (GL grid band-limit
``gl_minimal_lmax(lmax, Lw)``) scales with it. This script measures, on a
given FITS mask, how the truncation error falls as ``Lw`` grows, so that
``Lw`` can be chosen by evidence rather than guesswork.

Usage
-----
    .venv/bin/python examples/mask_diagnostics.py /path/to/mask.fits --lmax 500

Runs on any HEALPix mask FITS file, e.g., ``tests/data/baseline_mask.fits``
for development.

Discipline: only ``healpy`` I/O and nside/index arithmetic are used directly
(``hp.read_map``); every spherical-harmonic transform goes through
``cmbcov.mask``/``grid``/``sht``, exactly as
``tests/test_no_healpy_transforms.py`` requires of the package itself.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from collections.abc import Sequence

import healpy as hp
import numpy as np

from cmbcov.grid import (
    gl_minimal_lmax,
    gl_quadrature_weights,
    gl_shape,
    gl_synthesis,
)
from cmbcov.mask import MaskWlm, mask_spectral_moment
from cmbcov.sht import almxfl

# Measured on this machine: one m' column of the ACC
# precompute at lmax=500, Lw=250 (GL grid band-limit 625, i.e. 626 rings)
# took 40.5 ms. Cost scales as the cube of the grid band-limit ratio (the
# quadrature is a 2-D transform over ntheta*nphi ~ Lg^2 points, dominated by
# an O(Lg) per-ring Legendre transform -> O(Lg^3) total per column).
_REFERENCE_MS_PER_COLUMN = 40.5
_REFERENCE_LG = 625


def _fmt_pct(x: float) -> str:
    return f"{x:.3e}"


def _load_mask_or_die(mask_path: str) -> None:
    if not os.path.exists(mask_path):
        hint = (
            "\n(Runs on any HEALPix mask FITS file, e.g. "
            "tests/data/baseline_mask.fits, for development.)"
        )
        sys.exit(f"error: mask file not found: {mask_path}{hint}")


def _mask_ladder(lmax: int, lmax_full: int) -> list[int]:
    """
    The Lw ladder: lmax/4, lmax/2, lmax, 1.5 lmax, 2 lmax, clipped to
    [0, lmax_full] (lmax_full = 3*nside - 1, the finest band-limit a HEALPix
    map of this resolution can determine), deduplicated and sorted.
    """
    raw = [lmax / 4.0, lmax / 2.0, lmax, 1.5 * lmax, 2.0 * lmax]
    candidates = {max(0, min(lmax_full, int(round(x)))) for x in raw}
    return sorted(candidates)


def _print_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    widths = [
        (
            max(len(headers[i]), *(len(row[i]) for row in rows))
            if rows
            else len(headers[i])
        )
        for i in range(len(headers))
    ]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(c.ljust(w) for c, w in zip(row, widths)))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose a mask's harmonic band-limit convergence before "
            "running the exact GL covariance on it."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("mask_file", help="Path to the mask FITS file")
    parser.add_argument(
        "--lmax",
        type=int,
        required=True,
        help="Target covariance band-limit (the lmax the run will be validated at)",
    )
    parser.add_argument(
        "--target",
        type=float,
        default=1e-4,
        help="Truncation-error target used for the final Lw recommendation",
    )
    parser.add_argument(
        "--nthreads", type=int, default=None, help="Threads for ducc0 (default: auto)"
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=60,
        help="Max LSMR iterations for the mask alm least-squares solve",
    )
    args = parser.parse_args(argv)

    _load_mask_or_die(args.mask_file)

    load_path = os.path.dirname(args.mask_file) or "."
    mask_name = os.path.basename(args.mask_file)

    print("=" * 78)
    print(f"Mask diagnostics: {args.mask_file}")
    print(
        f"Target covariance lmax: {args.lmax}    truncation target: {args.target:.1e}"
    )
    print("=" * 78)

    # ------------------------------------------------------------------ #
    # 1. Basic geometry
    # ------------------------------------------------------------------ #
    raw_mask = np.asarray(hp.read_map(args.mask_file), dtype=np.float64)
    nside = hp.get_nside(raw_mask)
    lmax_full = 3 * nside - 1

    finite = np.isfinite(raw_mask)
    has_nan = bool(np.any(~finite))
    in_range = finite & (raw_mask >= 0.0) & (raw_mask <= 1.0)
    has_out_of_range = bool(np.any(finite & ~in_range))

    good = raw_mask[finite]
    fsky = float(np.mean(good))
    mean_w2 = float(np.mean(good**2))
    mean_w4 = float(np.mean(good**4))
    eff_fsky = (mean_w2**2 / mean_w4) if mean_w4 > 0 else float("nan")

    print("\n1. BASIC GEOMETRY")
    print(f"   nside                    : {nside}")
    print(f"   npix                     : {raw_mask.size}")
    print(f"   f_sky = mean(W)          : {fsky:.6f}")
    print(f"   mean(W^2)                : {mean_w2:.6f}")
    print(f"   mean(W^4)                : {mean_w4:.6f}")
    print(f"   effective f_sky W2^2/W4  : {eff_fsky:.6f}")
    print(f"   values outside [0, 1]    : {'YES' if has_out_of_range else 'no'}")
    print(f"   NaN / non-finite values  : {'YES' if has_nan else 'no'}")
    print(f"   3*nside - 1 (finest Lw)  : {lmax_full}")

    # ------------------------------------------------------------------ #
    # Reference: the finest mask alm this HEALPix map can determine.
    # ------------------------------------------------------------------ #
    ladder = _mask_ladder(args.lmax, lmax_full)
    lw_ladder_max = ladder[-1]

    print(
        f"\nSolving for the reference mask alm at lmax={lmax_full} "
        "(this is the largest band-limit used below; smaller Lw are "
        "truncations of it, not independent solves) ..."
    )
    t0 = time.time()
    wlm = MaskWlm(mask_name, load_path=load_path, precompute_alm=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        wlm.compute_spherical_harmonics(
            lmax=lmax_full, nthreads=args.nthreads, maxiter=args.maxiter, epsilon=1e-10
        )
    solve_s = time.time() - t0
    print(
        f"   done in {solve_s:.1f} s "
        f"(residual {wlm._mask_alm_residual:.2e}, {wlm._mask_alm_iterations} iterations)"
    )
    for w in caught:
        print(f"   note: {w.message}")

    # compute_power_spectra reuses the lmax_full solve above rather than
    # re-solving at its own defaults: it ensures the alm via _ensure_mask_alm,
    # which asks for the strictest of (default, cached) in each of lmax /
    # maxiter / epsilon. Before that helper existed it requested the bare
    # defaults, and the cache guard's maxiter/epsilon equality check made it
    # silently re-solve at lmax=2*nside whenever --maxiter != 20 -- discarding
    # the solve this script just paid for. See tests/test_mask_alm_cache.py.
    mask_alm_ref = wlm.mask_alm
    print(
        f"Computing the exact mask and W^2 power spectra "
        f"(GL grid band-limit {2 * lmax_full} for W^2) ..."
    )
    t0 = time.time()
    wl_full, w2l_full = wlm.compute_power_spectra(save=False, nthreads=args.nthreads)
    assert wlm.mask_alm is mask_alm_ref, (
        "compute_power_spectra replaced the reference alm; every Lw below is a "
        "truncation of it, so the ladder would no longer be self-consistent"
    )
    spectra_s = time.time() - t0
    print(f"   done in {spectra_s:.1f} s")
    ell_full = np.arange(lmax_full + 1)
    denom_wl = float(np.sum((2 * ell_full + 1) * wl_full))

    ell_w2 = np.arange(w2l_full.size)
    denom_w2l = float(np.sum((2 * ell_w2 + 1) * w2l_full))

    # Shared comparison grid for the synthesis-residual test: large enough
    # (band-limit lmax_full) to synthesise the reference alm exactly.
    lmax_grid_cmp = lmax_full
    w_ref_grid = gl_synthesis(
        mask_alm_ref,
        lmax=lmax_full,
        lmax_grid=lmax_grid_cmp,
        spin=0,
        nthreads=args.nthreads,
    )
    weights_cmp = gl_quadrature_weights(lmax_grid_cmp)
    max_w = float(np.max(np.abs(w_ref_grid)))

    # ------------------------------------------------------------------ #
    # 2 & 6. Band-limit convergence of W, and the recommendation
    # ------------------------------------------------------------------ #
    print("\n2. BAND-LIMIT CONVERGENCE OF W (the important one)")
    print(
        "   Reference field: the mask alm at the largest Lw in the ladder "
        f"({lw_ladder_max}), synthesised on a GL grid of band-limit "
        f"{lmax_grid_cmp}. Each row below truncates that SAME alm at Lw and "
        "compares."
    )
    print(
        "   Two residuals are reported: max|dW| and rms|dW| (both relative "
        "to max(W)) are the MAP-SPACE synthesis residual against the "
        "reference; 'power above Lw' is the HARMONIC-SPACE residual, the "
        f"fraction of W's own power (measured to lmax_full={lmax_full}) "
        "left above Lw. Whichever of the two is larger at a given Lw is the "
        "conservative estimate of the truncation error there; both are "
        "printed so the reader can see which."
    )

    rows_2 = []
    conservative = {}
    for lw in ladder:
        filt = np.ones(lw + 1)
        truncated_alm = almxfl(mask_alm_ref, filt)
        w_trunc_grid = gl_synthesis(
            truncated_alm,
            lmax=lmax_full,
            lmax_grid=lmax_grid_cmp,
            spin=0,
            nthreads=args.nthreads,
        )
        diff = w_trunc_grid - w_ref_grid
        max_dw = float(np.max(np.abs(diff))) / max_w if max_w > 0 else float("nan")
        rms_dw = (
            float(np.sqrt(np.sum(weights_cmp * diff**2) / (4 * np.pi))) / max_w
            if max_w > 0
            else float("nan")
        )

        power_above = float(
            np.sum((2 * ell_full[ell_full > lw] + 1) * wl_full[ell_full > lw])
        )
        frac_power_above = power_above / denom_wl if denom_wl > 0 else float("nan")

        conservative_value = max(rms_dw, frac_power_above)
        conservative[lw] = conservative_value
        which = "rms|dW|/max(W)" if rms_dw >= frac_power_above else "power above Lw"

        rows_2.append(
            [
                str(lw),
                _fmt_pct(max_dw),
                _fmt_pct(rms_dw),
                _fmt_pct(frac_power_above),
                which,
            ]
        )

    _print_table(
        ["Lw", "max|dW|/max(W)", "rms|dW|/max(W)", "power above Lw", "conservative"],
        rows_2,
    )

    # ------------------------------------------------------------------ #
    # 3. Band-limit convergence of W^2
    # ------------------------------------------------------------------ #
    print("\n3. BAND-LIMIT CONVERGENCE OF W^2 (what Xi[W^2] needs)")
    print(
        "   Squaring a mask band-limited at Lw doubles its true band-limit "
        "to 2 Lw, so the coupling-kernel spectrum needs power measured out "
        f"to 2*Lw. Fractions below are of W^2's total power, measured to "
        f"2*lmax_full={2 * lmax_full}."
    )
    rows_3 = []
    for lw in ladder:
        threshold = 2 * lw
        keep = ell_w2 > threshold
        power_above = float(np.sum((2 * ell_w2[keep] + 1) * w2l_full[keep]))
        frac = power_above / denom_w2l if denom_w2l > 0 else float("nan")
        rows_3.append([str(lw), str(threshold), _fmt_pct(frac)])
    _print_table(["Lw", "2*Lw", "power of W^2 above 2*Lw"], rows_3)

    # ------------------------------------------------------------------ #
    # 4. ACC polarisation leakage floor
    # ------------------------------------------------------------------ #
    moment_w2 = mask_spectral_moment(w2l_full)
    print("\n4. ACC POLARISATION LEAKAGE FLOOR")
    print(f"   <L(L+1)>_W2 (mask_spectral_moment of W^2) = {moment_w2:.4f}")
    print(
        "   Predicted leakage bias lambda(l) ~ (0.5 to 0.65) * <L(L+1)>_W2 / l^2 " ":"
    )
    rows_4 = []
    for ell_star in (100, 500, 1000):
        low = 0.5 * moment_w2 / ell_star**2
        high = 0.65 * moment_w2 / ell_star**2
        rows_4.append([str(ell_star), _fmt_pct(low), _fmt_pct(high)])
    _print_table(["l", "bias (0.5x, low)", "bias (0.65x, high)"], rows_4)

    # ------------------------------------------------------------------ #
    # 5. GL grid size and cost estimate
    # ------------------------------------------------------------------ #
    print("\n5. GL GRID SIZE AND EXACT-COVARIANCE COST ESTIMATE")
    print(
        f"   Reference: {_REFERENCE_MS_PER_COLUMN} ms/column measured at "
        f"lmax=500, Lw=250 (grid band-limit {_REFERENCE_LG}, "
        f"{_REFERENCE_LG + 1} rings). Cost is assumed to scale as the cube "
        "of the grid band-limit ratio. Total columns for the full 0..lmax "
        f"matrix assumed to be (lmax+1)^2 = {(args.lmax + 1) ** 2} "
        "(every (ell, m) pair, ell = 0..lmax); this is a single-process "
        "figure and rows (ell) are independent, so it parallelises "
        "trivially across ell."
    )
    rows_5 = []
    total_columns = (args.lmax + 1) ** 2
    for lw in ladder:
        lg = gl_minimal_lmax(args.lmax, lw)
        ntheta, nphi = gl_shape(lg)
        ms_per_column = _REFERENCE_MS_PER_COLUMN * (lg / _REFERENCE_LG) ** 3
        total_hours = total_columns * ms_per_column / 1000.0 / 3600.0
        rows_5.append(
            [
                str(lw),
                str(lg),
                f"{ntheta} x {nphi}",
                f"{ms_per_column:.2f}",
                f"{total_hours:.2f}",
            ]
        )
    _print_table(
        ["Lw", "GL grid Lg", "ntheta x nphi", "ms/column", "est. hours (full matrix)"],
        rows_5,
    )

    # ------------------------------------------------------------------ #
    # 6. Recommendation
    # ------------------------------------------------------------------ #
    recommended = None
    for lw in ladder:
        if conservative[lw] < args.target:
            recommended = lw
            break

    print("\n6. RECOMMENDATION")
    if recommended is not None:
        print(
            f"   Lw = {recommended} is the smallest ladder value whose "
            f"conservative truncation residual ({conservative[recommended]:.3e}) "
            f"is below the target ({args.target:.1e})."
        )
    else:
        print(
            f"   No Lw in the ladder ({ladder}) reaches the target "
            f"{args.target:.1e}; the largest, Lw={lw_ladder_max}, leaves a "
            f"residual of {conservative[lw_ladder_max]:.3e}. Re-run with a "
            "ladder extended beyond 2*lmax, or accept a looser target."
        )

    print("\n" + "=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
