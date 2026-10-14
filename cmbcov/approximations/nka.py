"""
Narrow-kernel approximation (Efstathiou 2004).

Theta-bar is replaced by delta functions (Eq. 27), giving
Sigma = 2 C_l C_l' Xi[W^2] (Eq. 26). Exact on the full sky, and accurate
whenever the spectrum varies slowly compared with the width of the coupling.
"""

import numpy as np

from ..assembly.keys import CovKey
from .base import CovarianceStrategy

__all__ = ["NKAStrategy"]


class NKAStrategy(CovarianceStrategy):
    """
    Narrow Kernel Approximation (NKA) strategy.

    The coupling kernels are approximated by delta functions, which neglects
    the coupling between different multipoles introduced by the mask.
    """

    def compute_covariance_term(
        self, cov_key: CovKey, cl: dict[str, dict[str, np.ndarray]]
    ) -> np.ndarray:
        """
        Compute the NKA covariance term of ``cov_key``.

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

        ``Cov._compute_norm_xi`` only has the T/E alphabet; the BB-only
        NKA/INKA mode (see ``BB_ONLY_OBSERVABLES`` in
        ``config.validation`` and docs/theory/bmode_kernels.md)
        maps "BB" onto "EE" here. ``BB`` has the same spin-weight as ``EE``
        (``w(BB) = 1``), so ``BB x BB`` is the ``Xi^{EE->EE}`` channel, up to
        the E->B leakage this mode neglects.
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
        Convert a power spectrum to the 2-D array multiplying ``Xi``.

        For TT, EE, BB this is ``sqrt(|C_l C_l'|)``; for TE, ET it is
        ``(C_l + C_l') / 2``.

        Parameters
        ----------
        power_spectrum : np.ndarray
            Input power spectrum array
        stokes_key : str
            One of "TT", "EE", "BB", "TE", "ET". "BB" (BB-only NKA/INKA mode,
            docs/theory/bmode_kernels.md) is treated like "EE", neglecting
            the leakage from C^EE.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_ell, n_ell)``.

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
