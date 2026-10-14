"""
The strategy interface.

Every approximation in Camphuis et al. (2022) is one choice of the reduced
covariance coupling kernel Theta-bar (Eq. 25). Subclasses differ only in that
choice; everything else -- the mask, the kernels, the block layout -- is shared
and reached through ``self.cov``.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator

import numpy as np

from ..assembly.covariance import Cov
from ..assembly.keys import CovKey

__all__ = ["CovarianceStrategy"]


class CovarianceStrategy(ABC):
    """Abstract base class for covariance computation strategies."""

    #: Lowest multipole with a binning weight in the current run, set by
    #: :meth:`~cmbcov.assembly.covariance.Cov.compute_covariance_matrix` before
    #: :meth:`configure_run` (``None`` outside a run). Informational only.
    binned_ell_min: int | None = None

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
        cl : dict[str, dict[str, np.ndarray]]
            Power spectra, ``cl[freq_combination][stokes_combination]``, with
            ``freq_combination`` e.g. ``"090GHz090GHz"`` and
            ``stokes_combination`` e.g. ``"TT"``, ``"EE"``, ``"TE"``, ``"ET"``.

        Returns
        -------
        np.ndarray
            Covariance matrix term for this key
        """
        pass

    def compute_covariance_terms(
        self, cov_keys: Iterable[CovKey], cl: dict[str, dict[str, np.ndarray]]
    ) -> Iterator[tuple[CovKey, np.ndarray]]:
        """
        Yield ``(cov_key, compute_covariance_term(cov_key, cl))`` for each key.

        A strategy may override this to compute keys together and yield them
        in another order (ACC does). Used by
        :meth:`~cmbcov.assembly.covariance.Cov.compute_covariance_matrix`.
        """
        for cov_key in cov_keys:
            yield cov_key, self.compute_covariance_term(cov_key, cl)

    def configure_run(self, covariance_keys, cl) -> None:
        """
        Hook called once by
        :meth:`~cmbcov.assembly.covariance.Cov.compute_covariance_matrix` before any
        block, with the run's keys and spectra, for strategies whose blocks
        depend on the whole run (ACC: whether any observable has a B letter).
        The default does nothing.
        """
        return None

    def raw_block_inputs(
        self, cov_key: CovKey, cl: dict[str, dict[str, np.ndarray]]
    ) -> tuple[dict[str, np.ndarray], dict]:
        """
        Inputs read by the raw block of ``cov_key``, for the ``save_raw_blocks``
        cache manifest (:meth:`~cmbcov.assembly.covariance.Cov._raw_block_manifest`).

        Returns ``(spectra, identity)``: ``spectra`` maps a label to the array
        the strategy reads for this key; ``identity`` holds any further
        JSON-serialisable, strategy-specific inputs. The default (NKA, INKA)
        returns the four spectra of the two Wick contractions
        (:meth:`~cmbcov.assembly.keys.CovKey.key_to_cross`) and an empty identity, since
        their kernels are keyed by the mask fields ``Cov`` records itself.
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

        Only ACC has one
        (:meth:`~cmbcov.approximations.acc.ACCStrategy.error_budget`); NKA and
        INKA return ``None``, as does an ACC run with no polarised leg.
        Reporting only: the covariance does not depend on it.
        """
        return None
