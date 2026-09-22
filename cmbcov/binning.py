"""
Bandpower binning.

Spectra on a small footprint are correlated over many multipoles, so they are
reported as bandpowers: weighted means over multipole ranges. The binning
matrix B built here turns a spectrum into bandpowers as ``B @ C`` and a
covariance into a bandpower covariance as ``B @ Sigma @ B.T``.

Note that the weights therefore enter the covariance quadratically, so a
weighting error biases the covariance at roughly twice the rate it biases a
spectrum.
"""

import numpy as np

__all__ = ["BinningManager"]


class BinningManager:
    """
    Handles all binning operations for covariance matrices.

    This class provides methods to create binning matrices that average
    power spectra over multipole ranges, supporting both custom bin edges
    and uniform step sizes.

    Parameters
    ----------
    lmax : int
        Maximum multipole for binning operations
    """

    def __init__(self, lmax: int, lmin: int = 0) -> None:
        if lmin < 0:
            raise ValueError(f"lmin must be non-negative, got {lmin}")
        if lmin >= lmax:
            raise ValueError(f"lmin ({lmin}) must be smaller than lmax ({lmax})")
        self.lmax = lmax
        self.lmin = lmin

    def create_bin_matrix(
        self, ell_bins: list[int] | int, flatten_with_ell_factor: bool = False
    ) -> tuple[np.ndarray, int]:
        """
        Create binning matrix from either list of cuts or step size.

        Parameters
        ----------
        ell_bins : Union[List[int], int]
            Either a list of bin edges or a uniform step size
        flatten_with_ell_factor : bool, default False
            Whether the bin weights must supply the ell(ell+1)/2pi flattening
            factor. See :meth:`bin_weight`.

        Returns
        -------
        Tuple[np.ndarray, int]
            Binning matrix and number of bins

        Raises
        ------
        TypeError
            If ell_bins is neither list nor int
        """
        if isinstance(ell_bins, list):
            return self._get_bin_matrix_from_range(ell_bins, flatten_with_ell_factor)
        elif isinstance(ell_bins, int):
            return self._get_bin_matrix_from_step(ell_bins, flatten_with_ell_factor)
        else:
            raise TypeError("ell_bins must be list or int")

    @staticmethod
    def bin_weight(ell: int, flatten_with_ell_factor: bool) -> float:
        r"""
        Weight given to multipole ``ell`` when averaging it into a bandpower.

        Bandpowers are weighted means over the multipoles in a bin. Because the
        CMB spectrum varies steeply across a wide bin, the quantity being
        averaged should be as flat as possible in ``ell``, otherwise the
        bandpower is dominated by the low-``ell`` edge of the bin.

        There are two ways to achieve that flatness, and the pipeline picks
        exactly one of them (see ``compute_covariance_matrix``):

        * the covariance has already been converted to
          :math:`D_\ell = \ell(\ell+1)C_\ell/2\pi` by
          :meth:`CovariancePostProcessor.apply_Dl_scaling`, so the binning
          weights must be **uniform**; or
        * the covariance is still in :math:`C_\ell`, so the binning weights
          must carry the :math:`\ell(\ell+1)/2\pi` factor themselves.

        Applying both, or neither, biases the bandpowers.

        The :math:`\ell(\ell+1)/2\pi` weighting is a deliberate choice:
        mode-counting :math:`(2\ell+1)` weights would redefine the
        bandpowers.

        Parameters
        ----------
        ell : int
            The multipole being weighted.
        flatten_with_ell_factor : bool
            ``True`` when this function must supply the flattening factor,
            i.e. when the covariance is still in :math:`C_\ell` units.

        Returns
        -------
        float
            Unnormalised weight for this multipole. Rows of the bin matrix are
            normalised to sum to one afterwards.
        """
        # NOTE: behaviour preserved exactly as before the refactor, so that
        # existing results remain bit-identical. Two caveats worth revisiting
        # deliberately rather than by accident:
        #   * at ell = 0 the flattening weight is exactly 0, so a bin
        #     containing only ell = 0 normalises to 0/0; the callers patch that
        #     up afterwards with an isnan() check.
        #   * the 2*pi denominator cancels in the row normalisation and is
        #     therefore cosmetic; it is kept so the weights read as D_ell.
        if flatten_with_ell_factor:
            return ell * (ell + 1) / (2 * np.pi)
        return 1.0

    def _drop_empty_bins(self, bin_matrix: np.ndarray) -> tuple[np.ndarray, int]:
        """
        Remove bins that carry no weight.

        A bin lying entirely below ``lmin`` contributes nothing, and keeping it
        would divide by a zero row sum during normalisation and emit a NaN
        bandpower.
        """
        keep = bin_matrix.sum(axis=1) > 0
        return bin_matrix[keep], int(keep.sum())

    def _get_bin_matrix_from_range(
        self, ell_cuts: list, Dl: bool = False
    ) -> tuple[np.ndarray, int]:
        """Create bin matrix from list of ell cuts."""
        # Bins are half-open [start, stop), so an edge equal to lmax is a valid
        # exclusive upper bound. Using '<' here silently discarded the final
        # bandpower whenever the last edge coincided with lmax -- which is the
        # case in examples/parameters_example.yml.
        num_bins = np.sum(np.array(ell_cuts) <= self.lmax) - 1
        bin_matrix = np.zeros((num_bins, self.lmax))

        for bin_index in range(num_bins):
            # Multipoles below lmin carry no weight. A bin straddling lmin is
            # truncated rather than dropped, so its usable modes are kept and
            # its reported centre moves to reflect what was actually averaged.
            for ell in range(
                max(ell_cuts[bin_index], self.lmin), ell_cuts[bin_index + 1]
            ):
                bin_matrix[bin_index, ell] = self.bin_weight(ell, Dl)

        bin_matrix, num_bins = self._drop_empty_bins(bin_matrix)

        # Normalize
        bin_matrix /= np.outer(np.sum(bin_matrix, axis=1), np.ones(self.lmax))

        # Handle edge case for first bin
        if num_bins and np.all(np.isnan(bin_matrix[0])):
            bin_matrix[0] = np.zeros(self.lmax)
            bin_matrix[0, 0] = 1.0

        assert np.isclose(
            np.sum(bin_matrix), num_bins, rtol=0, atol=1e-9
        ), f"Expected {num_bins}, got {np.sum(bin_matrix)}"
        return bin_matrix, num_bins

    def _get_bin_matrix_from_step(
        self, ell_step: int, Dl: bool = False
    ) -> tuple[np.ndarray, int]:
        """Create bin matrix from step size."""
        num_bins = int(np.ceil(self.lmax / ell_step))
        bin_matrix = np.zeros((num_bins, self.lmax))

        for bin_index in range(num_bins):
            for ell in range(
                max(bin_index * ell_step, self.lmin),
                min((bin_index + 1) * ell_step, self.lmax),
            ):
                bin_matrix[bin_index, ell] = self.bin_weight(ell, Dl)

        bin_matrix, num_bins = self._drop_empty_bins(bin_matrix)

        # Normalize
        bin_matrix /= np.outer(np.sum(bin_matrix, axis=1), np.ones(self.lmax))

        assert np.isclose(
            np.sum(bin_matrix), num_bins, rtol=0, atol=1e-9
        ), f"Expected {num_bins}, got {np.sum(bin_matrix)}"
        return bin_matrix, num_bins
