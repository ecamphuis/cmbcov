"""
Narrow-kernel approximation (Efstathiou 2004).

Theta-bar is replaced by delta functions (Eq. 27), giving
Sigma = 2 C_l C_l' Xi[W^2] (Eq. 26). Exact on the full sky, and accurate
whenever the spectrum varies slowly compared with the width of the coupling.
"""

import numpy as np

from ..keys import CovKey
from .base import CovarianceStrategy

__all__ = ["NKAStrategy"]


class NKAStrategy(CovarianceStrategy):
    """
    Narrow Kernel Approximation (NKA) strategy.

    This strategy implements the simplest analytical covariance approximation
    where coupling kernels are approximated as narrow (delta-function-like).
    This assumes minimal coupling between different multipoles due to masking.
    """

    def compute_covariance_term(
        self, cov_key: CovKey, cl: dict[str, dict[str, np.ndarray]]
    ) -> np.ndarray:
        """
        Compute NKA covariance term using narrow kernel approximation.

        Parameters
        ----------
        cov_key : CovKey
            Covariance key specifying which cross-spectra to compute
        cl : Dict[str, Dict[str, np.ndarray]]
            Power spectra organized as cl[freq_combination][stokes_combination]
        config : CovarianceConfig
            Configuration parameters

        Returns
        -------
        np.ndarray
            Covariance matrix term for this key
        """
        combination_1, combination_2 = cov_key.key_to_cross()

        term_1 = (
            self.cl_and_key_to_average(
                cl[combination_1[0].freqkey()][combination_1[0].stokekey()],
                combination_1[0].stokekey(),
            )
            * self.cl_and_key_to_average(
                cl[combination_1[1].freqkey()][combination_1[1].stokekey()],
                combination_1[1].stokekey(),
            )
            * self._norm_xi(combination_1[0].stokekey(), combination_1[1].stokekey())
        )

        term_2 = (
            self.cl_and_key_to_average(
                cl[combination_2[0].freqkey()][combination_2[0].stokekey()],
                combination_2[0].stokekey(),
            )
            * self.cl_and_key_to_average(
                cl[combination_2[1].freqkey()][combination_2[1].stokekey()],
                combination_2[1].stokekey(),
            )
            * self._norm_xi(combination_2[0].stokekey(), combination_2[1].stokekey())
        )

        return term_1 + term_2

    def _norm_xi(self, left: str, right: str) -> np.ndarray:
        """
        ``self.cov.norm_Xi[left, right]``, with "BB" looked up as "EE".

        ``Cov._compute_norm_xi`` stays the exact T/E alphabet
        (``tests/test_norm_xi_channels.py`` pins it): the BB-only NKA/INKA
        escape hatch (``generator.parameter_validation.APPROXIMATIONS_SUPPORTING_B``,
        docs/theory/bmode_kernels.md) instead maps "BB" onto
        "EE" here. The spin-weight rule gives ``BB`` the same weight as
        ``EE`` (``w(BB) = 1``), so ``BB x BB`` is exactly the ``Xi^{EE->EE}``
        channel, up to the E->B leakage this mode neglects throughout.
        """
        norm_xi = self.cov.norm_Xi
        key = (
            left if left != "BB" else "EE",
            right if right != "BB" else "EE",
        )
        return norm_xi[key]

    @staticmethod
    def cl_and_key_to_average(
        power_spectrum: np.ndarray, stokes_key: str
    ) -> np.ndarray:
        """
        Convert power spectrum to covariance format.

        Parameters
        ----------
        power_spectrum : np.ndarray
            Input power spectrum array
        stokes_key : str
            Stokes parameter combination ("TT", "EE", "BB", "TE", or "ET").
            "BB" only reaches here through the BB-only NKA/INKA escape
            hatch (``generator.parameter_validation.APPROXIMATIONS_SUPPORTING_B``,
            docs/theory/bmode_kernels.md): it is treated
            exactly like "EE", leakage from C^EE neglected.

        Returns
        -------
        np.ndarray
            Converted power spectrum in covariance format

        Raises
        ------
        KeyError
            If stokes_key is not one of the supported values
        """
        if stokes_key in ["TT", "EE", "BB"]:
            return np.outer(
                np.sqrt(np.abs(power_spectrum)), np.sqrt(np.abs(power_spectrum))
            )
        elif stokes_key in ["TE", "ET"]:
            return 0.5 * (
                np.outer(np.ones(power_spectrum.shape[0]), power_spectrum)
                + np.outer(power_spectrum, np.ones(power_spectrum.shape[0]))
            )
        else:
            raise KeyError(
                f"stokes_key should be one of ['TT', 'EE', 'BB', 'TE', 'ET'], "
                f"got '{stokes_key}'"
            )
