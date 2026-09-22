"""
Cross-validation of the ACC coupling kernel against the exact covariance.

This is the only test that says the ACC precomputation is *correct* rather
than merely *stable*. Two routes with no shared code reach the same number:

- ACC (paper Fig. 8): Theta is built from products of four masked-harmonic
  integrals I summed over m, m', m1, m2, normalised to Theta-bar (Eq. 23) and
  contracted with the spectrum, Sigma = 2 Xi[W^2] * C . Theta-bar . C (Eq. 25).
- Exact (paper Fig. 2): the operator K diag(C) K^dagger is applied to each
  basis vector with three spherical harmonic transforms, and the (l, l') block
  Frobenius norm gives Sigma directly (Eq. 7).

Agreement at the 1e-4 level at nside=16 pins the ACC kernel's shape and the
Eq. 22-25 normalisation together.
"""

import os
import tempfile
import warnings
from types import SimpleNamespace

import healpy as hp
import numpy as np
import pytest

from cmbcov.approximations.acc import (
    ACCStrategy,
    acc_internal_lmax,
    precompute_acc_kernels,
)
from cmbcov.covariance import Cov, CovarianceConfig, CovarianceMethod
from cmbcov.exact import exact_covariance_row
from cmbcov.kernels.coupling import KERNEL_TT, coupling_kernels
from cmbcov.keys import CovKey

DATA = os.path.join(os.path.dirname(__file__), "data")
NSIDE = 16
ELL = 16
ELLPS = [ELL, ELL + 1, ELL + 2, ELL + 3]


@pytest.fixture(scope="module")
def setup():
    config = CovarianceConfig(
        method=CovarianceMethod.ACC, lmax=48, dmax=4, centralell=ELL
    )
    cov = Cov(
        "baseline_mask.fits",
        config=config,
        mask_path=os.path.abspath(DATA),
        save_dir=tempfile.mkdtemp(),
    )
    # ACC's own degraded mask, exactly as _setup_coupling_computation builds it.
    wn, nside = cov.wlm.degrade_mask(NSIDE)
    lmax = 2 * nside - 1  # ACC kernels are (2 nside) square, indices 0..2nside-1
    theta = precompute_acc_kernels(
        cov.wlm,
        None,
        centralell=ELL,
        ellprange=ELLPS,
        nside=NSIDE,
        grid="healpix",
        dryrun=True,
    )

    ell = np.arange(lmax + 1)
    cl = np.zeros(lmax + 1)
    cl[2:] = 1e-3 / (ell[2:] * (ell[2:] + 1)) ** 0.9
    xi = (
        coupling_kernels(hp.anafast(wn**2, lmax=lmax), lmax)[KERNEL_TT]
        / (2 * ell + 1)[None, :]
    )
    return wn, nside, lmax, theta, cl, xi


@pytest.mark.parametrize("ellp", ELLPS)
def test_acc_kernel_reproduces_exact_covariance(setup, ellp):
    wn, nside, lmax, theta, cl, xi = setup
    theta_bar = theta[ellp]["TTxTT"] / theta[ellp]["TTxTT"].sum()  # Eq. 23
    acc = 2.0 * xi[ELL, ellp] * (cl @ theta_bar @ cl)  # Eq. 25
    exact = exact_covariance_row(wn, cl, ellp, lmax, nside, iter=3)[ELL]
    assert (
        abs(acc / exact - 1.0) < 1e-3
    ), f"ACC/exact = {acc / exact:.5f} at ({ELL},{ellp})"


@pytest.mark.parametrize("ellp", ELLPS)
def test_theta_normalisation_matches_eq22_up_to_transform_bias(setup, ellp):
    """
    Eq. 22: sum_{l1 l2} Theta = (2l+1)(2l'+1) Xi[W^2]. The ACC precompute
    builds Theta with healpy's default map2alm (iter=3): measured ratio
    0.99988 .. 0.99995. At single-adjoint (iter=0) transforms this ratio
    would carry a ~2.5% amplitude deficit (0.9705 .. 0.9744); the Eq. 23
    normalisation cancels such a deficit, which is why
    ``test_acc_kernel_reproduces_exact_covariance`` above agrees to 1e-4
    regardless.
    """
    wn, nside, lmax, theta, cl, xi = setup
    ratio = theta[ellp]["TTxTT"].sum() / (
        (2 * ELL + 1) * (2 * ellp + 1) * xi[ELL, ellp]
    )
    assert 0.999 < ratio < 1.001, f"sum(Theta)/Eq.22 = {ratio:.5f}"


@pytest.mark.parametrize("ellp", ELLPS)
def test_entry_next_to_lmax_is_not_biased_by_the_window_clip(setup, ellp):
    """
    The entry ``(ELL, ellp)`` reported at ``lmax = ellp + 1``, so
    ``min(l, l') = ELL`` sits 1..4 multipoles below ``lmax``, through
    ``ACCStrategy.compute_covariance_term``.

    With the spectra read to ``lmax_int = lmax + (2 NSIDE - 1 - ELL)`` it
    matches the exact covariance to the same 1e-3 as the unclipped kernel
    above (measured 1 - 6.4e-5 .. 1 + 2.7e-5). The same computation with the
    spectra cut at ``lmax`` instead drops the kernel's ``L >= lmax`` rows and
    columns: measured 0.452, 0.536, 0.619, 0.701 of exact for
    ``ellp = 16 .. 19``.
    """
    wn, nside, lmax, theta, cl, xi = setup
    report_lmax = ellp + 1
    size = theta[ellp]["TTxTT"].shape[0]
    lmax_int = acc_internal_lmax(report_lmax, size, ELL)
    # cl is band-limited at 2 nside - 1: zeros past it are the true spectrum.
    padded = np.zeros(max(lmax_int, cl.size))
    padded[: cl.size] = cl
    cut = padded.copy()
    cut[report_lmax:] = 0.0

    class Strategy(ACCStrategy):
        def get_covariance_coupling(self, ell, ell_prime, pairs=None):
            return {("TT", "TT"): theta[ell_prime]["TTxTT"]}

    config = SimpleNamespace(
        method=CovarianceMethod.ACC, dmax=ellp - ELL + 1, centralell=ELL
    )
    cov = SimpleNamespace(
        lmax=report_lmax,
        config=config,
        norm_Xi={("TT", "TT"): xi[:report_lmax, :report_lmax]},
    )
    key = CovKey(("T",) * 4, ("090GHz",) * 4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # low centralell
        new = Strategy(cov).compute_covariance_term(
            key, {"090GHz090GHz": {"TT": padded}}
        )[ELL, ellp]
        old = Strategy(cov).compute_covariance_term(key, {"090GHz090GHz": {"TT": cut}})[
            ELL, ellp
        ]
    exact = exact_covariance_row(wn, cl, ellp, lmax, nside, iter=3)[ELL]

    assert abs(new / exact - 1.0) < 1e-3, f"padded ACC/exact = {new / exact:.5f}"
    assert old / exact < 0.75, f"clipped ACC/exact = {old / exact:.5f}"
