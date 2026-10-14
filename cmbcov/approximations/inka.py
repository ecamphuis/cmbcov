"""
Improved narrow-kernel approximation (Nicola et al. 2021).

The delta functions of the NKA are replaced by the renormalised MASTER kernel
(Eq. 30), which is equivalent to smoothing the spectrum first: with
C-bar = M-bar C, the covariance is 2 C-bar_l C-bar_l' Xi[W^2] (Eq. 31).
"""

import numpy as np

from ..assembly.keys import CovKey
from .base import CovarianceStrategy
from .nka import NKAStrategy

__all__ = ["INKAStrategy", "renormalised_kernel"]


def renormalised_kernel(kernel: np.ndarray) -> np.ndarray:
    r"""
    The renormalised MASTER kernel of Camphuis et al. (2022) Eq. (A.24).

    .. math::
        \bar M_{\ell\ell'} = M_{\ell\ell'} \Big/ \sum_{\ell'} M_{\ell\ell'},

    for a kernel indexed ``[l, l']`` that acts as ``M @ C``, so that each row
    sums to one (Eq. A.25) and a constant spectrum is left unchanged. A row
    that sums to zero (``l < 2`` in the spin-2 channels) is left as it is.
    """
    row_sums = kernel.sum(axis=-1, keepdims=True)
    return kernel / np.where(row_sums == 0, 1.0, row_sums)


class INKAStrategy(CovarianceStrategy):
    """
    Improved Narrow Kernel Approximation (INKA) strategy.

    The spectra are first smoothed with the renormalised MASTER kernels, then
    the NKA is applied to the smoothed spectra.
    """

    def compute_covariance_term(
        self, cov_key: CovKey, cl: dict[str, dict[str, np.ndarray]]
    ) -> np.ndarray:
        """
        Compute the INKA covariance term of ``cov_key``.

        Parameters
        ----------
        cov_key : CovKey
            Covariance key specifying which cross-spectra to compute
        cl : Dict[str, Dict[str, np.ndarray]]
            Power spectra, ``cl[freq_combination][stokes_combination]``.

        Returns
        -------
        np.ndarray
            Covariance matrix term for this key
        """
        # Smooth cl with the master kernels. There is no dedicated BB kernel
        # (mask/coupling.py KERNEL_EEpBB/KERNEL_EEmBB), so in the BB-only
        # mode (docs/theory/bmode_kernels.md) C^BB is smoothed like EE,
        # neglecting the EE->BB leakage term M^- C^EE.
        master = self.cov.M_footprint
        master_kernels_dict = {
            "TT": master[0],
            "EE": (master[1] + master[2]) / 2,
            "BB": (master[1] + master[2]) / 2,
            "TE": master[3],
            "ET": master[3],
        }

        master_kernels_dict = {
            key: renormalised_kernel(kernel)
            for key, kernel in master_kernels_dict.items()
        }

        # C-bar = M-bar C (Eq. 31)
        inka_cl = {}
        for freq_key, spectrum_dict in cl.items():
            inka_cl[freq_key] = {}
            for stokes_key, spectrum in spectrum_dict.items():
                inka_cl[freq_key][stokes_key] = (
                    master_kernels_dict[stokes_key] @ spectrum
                )

        # NKA on the smoothed spectra
        nka_strategy = NKAStrategy(self.cov)
        return nka_strategy.compute_covariance_term(cov_key, inka_cl)
