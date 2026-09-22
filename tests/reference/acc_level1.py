"""
A small level-1 (T/E-only) ACC run whose numbers must not move when B-mode
support changes the assembly.

:func:`build_level1_reference` uses only API that predates B-mode support
(``precompute_acc_kernels``, ``Cov``, ``CovarianceConfig``, ``CovKeys``,
``StrategyFactory``), so the same function produces the stored reference
``acc_level1_reference.npz`` and is re-run by
``tests/test_acc_bmode_assembly.py`` against the current one. To
regenerate (deliberately, and say why in the commit message), run
from a checkout of the code that should define the reference::

    python tests/reference/acc_level1.py tests/reference/acc_level1_reference.npz

Configuration: the derivation test's rippled cap (docs/theory/bmode_kernels.md), band-limited at ``LW = 6`` and written at ``nside = 32``; GL
kernels at ``l* = 8``, ``dmax = 4``, ``nside_acc = 16``; ``lmax = 20``; two
frequencies with different spectra, so TE and ET differ across frequencies.
"""

import os
import sys
import tempfile
import warnings

import numpy as np

NSIDE_MASK = 32
LW = 6
CENTRALELL = 8
DMAX = 4
NSIDE_ACC = 16
LMAX = 20
FREQS = ["f1", "f2"]


def rippled_cap_bandlimited(nside: int = NSIDE_MASK, lw: int = LW) -> np.ndarray:
    """The derivation test's mask, band-limited at ``lw`` (a map)."""
    import healpy as hp

    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    vec = np.array(hp.pix2vec(nside, np.arange(npix))).T
    centre = hp.ang2vec(np.radians(50), np.radians(30))
    dist = np.arccos(np.clip(vec @ centre, -1, 1))
    edge = np.radians(60)
    cap = np.where(dist < edge, 0.5 * (1 + np.cos(np.pi * dist / edge)), 0.0)
    mask = cap * (1 + 0.3 * np.sin(3 * phi + 1.0) * np.sin(theta))
    alm = hp.map2alm(mask, lmax=lw, iter=10)
    return hp.alm2map(alm, nside, lmax=lw)


def level1_spectra(n: int = 80) -> dict:
    """Two-frequency T/E spectra, ``n`` multipoles (past ``lmax_int``)."""
    ell = np.arange(n)
    tt = np.zeros(n)
    ee = np.zeros(n)
    tt[2:] = 1.0 / (ell[2:] * (ell[2:] + 1)) ** 0.8
    ee[2:] = 0.3 / (ell[2:] + 3.0) ** 1.7 * (1 + 0.5 * np.cos(ell[2:] / 2.0))
    te = 0.6 * np.sqrt(tt * ee) * np.cos(ell / 3.0)
    cl = {}
    for i, f1 in enumerate(FREQS):
        for j, f2 in enumerate(FREQS):
            scale = 1.0 + 0.1 * i + 0.2 * j
            cl[f1 + f2] = {
                "TT": scale * tt,
                "EE": scale * ee,
                "TE": scale * te * (1.0 + 0.05 * i),
                "ET": scale * te * (1.0 + 0.05 * j),
            }
    return cl


def build_level1_reference(workdir: str) -> dict:
    """
    Every raw block of a two-frequency ``stokes: [T, E]`` ACC run, plus the
    binned PolSpice-transformed matrix of ``Cov.compute_covariance_matrix``.
    Keys: ``raw <stokes> <freqs>`` and ``matrix``.
    """
    import healpy as hp

    from cmbcov.approximations import StrategyFactory
    from cmbcov.approximations.acc import precompute_acc_kernels
    from cmbcov.covariance import (
        Cov,
        CovarianceConfig,
        CovarianceMethod,
    )
    from cmbcov.keys import CovKeys

    hp.write_map(
        os.path.join(workdir, "mask.fits"),
        rippled_cap_bandlimited(),
        overwrite=True,
        dtype=np.float64,
    )
    precompute_acc_kernels(
        "mask.fits",
        workdir,
        centralell=CENTRALELL,
        dmax=DMAX,
        mask_path=workdir,
        nside=NSIDE_ACC,
        grid="gl",
        lw=LW,
    )
    config = CovarianceConfig(
        method=CovarianceMethod.ACC, lmax=LMAX, lmin=2, dmax=DMAX, centralell=CENTRALELL
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # low-centralell warning
        cov = Cov("mask.fits", config=config, mask_path=workdir, save_dir=workdir)
        keys = CovKeys(["T", "E"], FREQS)
        cl = level1_spectra()
        out = {}
        strategy = StrategyFactory.create_strategy(cov)
        for key in keys.keys():
            out[f"raw {key.stokekey()} {key.freqkey()}"] = (
                strategy.compute_covariance_term(key, cl)
            )
        _, _, matrix = cov.compute_covariance_matrix([2, 8, 14, 20], keys, cl)
    out["matrix"] = matrix
    return out


if __name__ == "__main__":  # pragma: no cover - reference generation
    target = sys.argv[1]
    with tempfile.TemporaryDirectory() as work:
        np.savez(target, **build_level1_reference(work))
    print(f"wrote {target}")
