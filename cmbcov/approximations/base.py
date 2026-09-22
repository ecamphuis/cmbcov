"""
The strategy interface.

Every approximation in Camphuis et al. (2022) is one choice of the reduced
covariance coupling kernel Theta-bar (Eq. 25). Subclasses differ only in that
choice; everything else -- the mask, the kernels, the block layout -- is shared
and reached through ``self.cov``.
"""

from abc import ABC, abstractmethod

import numpy as np

from ..covariance import Cov
from ..keys import CovKey

__all__ = ["CovarianceStrategy"]


class CovarianceStrategy(ABC):
    """Abstract base class for covariance computation strategies."""

    def __init__(self, covariance_instance: Cov) -> None:
        """
        Initialize strategy with covariance instance.

        Parameters
        ----------
        covariance_instance : Cov
            The main covariance computation instance containing mask and kernels
        """
        self.cov = covariance_instance

    @abstractmethod
    def compute_covariance_term(
        self, cov_key: CovKey, cl: dict[str, dict[str, np.ndarray]]
    ) -> np.ndarray:
        """
        Compute covariance term for a specific key.

        Parameters
        ----------
        cov_key : CovKey
            Covariance key specifying which cross-spectra to compute
        cl : Dict[str, Dict[str, np.ndarray]]
            Power spectra organized as cl[freq_combintation][stokes_combination] where:
            - freq_combination: e.g., "090GHz090GHz", "090GHz150GHz"
            - stokes_combination: e.g., "TT", "EE", "TE", "ET"

        Returns
        -------
        np.ndarray
            Covariance matrix term for this key
        """
        pass

    def configure_run(self, covariance_keys, cl) -> None:
        """
        Called once by :meth:`~cmbcov.covariance.Cov.compute_covariance_matrix`
        before any block, with the run's keys and spectra, for strategies
        whose blocks depend on the run as a whole and not on one key alone
        (ACC: whether any observable has a B letter). The default does
        nothing.
        """
        return None

    def raw_block_inputs(
        self, cov_key: CovKey, cl: dict[str, dict[str, np.ndarray]]
    ) -> tuple[dict[str, np.ndarray], dict]:
        """
        What the raw block of ``cov_key`` reads, for the ``save_raw_blocks``
        cache manifest (:meth:`~cmbcov.covariance.Cov._raw_block_manifest`).

        Returns ``(spectra, identity)``: ``spectra`` maps a label to the exact
        array the strategy reads for this key, ``identity`` holds any further
        JSON-serialisable inputs specific to the strategy. The default -- NKA
        and INKA -- is the four whole spectra of the two Wick contractions
        (:meth:`~cmbcov.keys.CovKey.key_to_cross`) and no
        extra identity (their kernels, Xi and M, are keyed by the mask fields
        ``Cov`` records itself).
        """
        spectra = {}
        for combination in cov_key.key_to_cross():
            for spec in combination:
                label = f"{spec.freqkey()}/{spec.stokekey()}"
                spectra[label] = cl[spec.freqkey()][spec.stokekey()]
        return spectra, {}

    def validate_config(self) -> None:
        """
        Validate configuration for this strategy.
        """
        # Base validation - subclasses can override for specific requirements
        return True

    def error_budget(self, covariance_keys, band_edges=None) -> dict | None:
        """
        A report on the known errors of this method's blocks, or ``None``.

        Only ACC has one (:meth:`~cmbcov.approximations.acc.ACCStrategy.error_budget`);
        NKA and INKA return ``None``, as does an ACC run with no polarised
        leg. Reporting only: no strategy's numbers depend on this.
        """
        return None
