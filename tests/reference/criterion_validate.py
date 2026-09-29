"""Validate the term-selection thresholds, as ``cmbcov.term_selection`` cites.

The module docstring of ``cmbcov/term_selection.py`` used to cite a
``criterion_validate.py`` that was never in the repository -- it was a scratch
file from the original research -- so the numbers it quoted could not be
re-checked.  This is a replacement, written to reproduce them.

For each (mask, nside, ell, ell') it builds the full kernel K from the
spin-weighted integrals, then the kernel from the selected terms at a given
``eps_pair``, and reports the relative Frobenius error alongside the fraction
of (m, m') pairs and of M orders kept.  ``OFFSETS`` sets the ell' - ell values
swept; the point of sweeping them is that the accuracy degrades with
separation (see ``docs/theory/term_selection.md`` Sect. 6).

Two masks: a synthetic two-blob mask (a large cap plus a smaller one 50
degrees away, always available) and, with ``--mask``, a footprint of your own
as a HEALPix FITS file, degraded to each nside.  The documented numbers were
measured on a 4% apodised survey footprint.

This is a research script, not a test: it is slow and nothing imports it.

    .venv/bin/python tests/reference/criterion_validate.py 64
    .venv/bin/python tests/reference/criterion_validate.py 32 64 128 --mask my_mask.fits
"""

import os
import sys

import healpy
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

from cmbcov.grid import spin_weighted_integrals_gl  # noqa: E402
from cmbcov.term_selection import (  # noqa: E402
    pole_rotation,
    rotate_alm,
    select_terms,
)

TOL = 1e-3
OFFSETS = (0, 5, 10, 19)


def frob(a):
    return np.sqrt(np.sum(a * a))


def kernel_from_integrals(X, Y, keep_pair=None):
    th = np.einsum("mLM,nLM->mnL", X, np.conj(Y), optimize=True)
    if keep_pair is not None:
        th = th * keep_pair[:, :, None]
    flat = th.reshape(-1, th.shape[-1])
    return (flat.T @ np.conj(flat)).real


def cap_mask(nside, lon, lat, radius_deg, apod_deg):
    v = healpy.ang2vec(lon, lat, lonlat=True)
    pv = np.array(healpy.pix2vec(nside, np.arange(healpy.nside2npix(nside))))
    dist = np.degrees(np.arccos(np.clip(v @ pv, -1, 1)))
    x = np.clip((dist - (radius_deg - apod_deg)) / apod_deg, 0, 1)
    return 0.5 * (1 + np.cos(np.pi * x))


def footprint_mask(path, nside):
    """A footprint read from a HEALPix FITS file, degraded to `nside`."""
    m = np.asarray(healpy.read_map(path), dtype=float)
    m[~np.isfinite(m)] = 0.0
    m[m < -1e29] = 0.0
    return np.clip(healpy.ud_grade(m, nside), 0.0, None)


def two_blob_mask(nside):
    return np.clip(
        cap_mask(nside, 47.0, 23.0, 30, 12) + 0.6 * cap_mask(nside, 95.0, 5.0, 18, 8),
        0,
        1,
    )


def pole_alm(mask, lw):
    return rotate_alm(healpy.map2alm(mask, lmax=lw, iter=10), lw, pole_rotation(mask))


def run(name, mask, nside):
    lw, lmax_out = 3 * nside - 1, 2 * nside - 1
    ell = nside
    alm = pole_alm(mask, lw)
    Ms = np.arange(-lmax_out, lmax_out + 1)

    def integrals(ell_):
        return np.array(
            [
                spin_weighted_integrals_gl(alm, lw, ell_, m, lmax_out)
                for m in range(-ell_, ell_ + 1)
            ]
        )

    X = integrals(ell)
    for off in OFFSETS:
        ellp = ell + off
        Y = X if off == 0 else integrals(ellp)
        K = kernel_from_integrals(X, Y)
        nK = frob(K)
        for label, eps in (("tol/4  (old)", TOL / 4), ("tol/40 (new)", TOL / 40)):
            sel = select_terms(alm, lw, ell, ellp, TOL, eps_pair=eps)
            bx = (
                np.abs(Ms[None, None, :] - np.arange(-ell, ell + 1)[:, None, None])
                <= sel.m_band
            )
            by = (
                np.abs(Ms[None, None, :] - np.arange(-ellp, ellp + 1)[:, None, None])
                <= sel.m_band
            )
            err = frob(K - kernel_from_integrals(X * bx, Y * by, sel.keep_pair)) / nK
            frac_M = (2 * sel.m_band + 1) / (2 * lmax_out + 1)
            print(
                f"{name:22s} nside {nside:4d}  l={ell} l'={ellp}  {label}  "
                f"err {err:.3e} {'ok ' if err <= TOL else 'MISS'}  "
                f"pairs {sel.frac_pairs * 100:6.2f}%  M {min(frac_M, 1.0) * 100:6.2f}%",
                flush=True,
            )
        del Y


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("nside", type=int, nargs="*", default=[64])
    parser.add_argument(
        "--mask", help="HEALPix FITS footprint to add to the two-blob mask"
    )
    args = parser.parse_args()
    for nside in args.nside:
        if args.mask:
            run("footprint", footprint_mask(args.mask, nside), nside)
        run("two blobs", two_blob_mask(nside), nside)
