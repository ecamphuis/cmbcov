"""
Post-processing of a pseudo-spectrum covariance.

Turns the pseudo-Cl covariance into the covariance of the estimator actually
reported: the PolSpice convolution of Camphuis et al. (2022) Eq. (55),
Sigma_hat = G Sigma G^T, plus optional D_ell scaling and calibration
debiasing. A run with a B-mode observable instead goes through `pseudo_to_spice_bmode`,
which is Eq. (55) plus the EE/BB decoupling mixing of
docs/theory/bmode_kernels.md.
"""

from collections.abc import Callable

import numpy as np

__all__ = ["CovariancePostProcessor"]


class CovariancePostProcessor:
    r"""
    Handles post-processing operations like PolSpice transformation.

    This class provides methods to transform covariance matrices from
    pseudo-Cl to PolSpice format, apply calibration corrections, and
    handle D_ell scaling.

    Parameters
    ----------
    G_kernels : dict or array-like
        PolSpice transformation kernels. A dict is used as given, keyed by
        Stokes pair. An array must be in the layout of
        :meth:`~cmbcov.kernels.polspice.PolSpiceKernels.as_covariance_array`,
        ``(G0, Gplus, Gminus, Gx)`` -- **not** the legacy
        ``as_master_array("G")`` layout ``(G0, Gp2, Gm2, Gx)``, which puts the
        non-decoupled :math:`^{+2}G` where :math:`^{dec}G` belongs and is
        wrong by a factor 86 in the lowest EE bandpower of the survey mask.

    Notes
    -----
    Channel 2, :math:`^{-}G`, is the ``EE <-> BB`` mixing kernel. `pseudo_to_spice`
    stays diagonal in the spectrum -- it is the level-1 (T/E-only) path and
    must stay bit-identical -- so a level-1 ``Cov(EE)`` omits the
    :math:`^{-}G\,\mathrm{Cov}(\tilde C^{BB})\,{}^{-}G^{T}` term (that term is
    omitted at level 1, since it is negligible for a lensing-level
    ``C^{BB}``). A run with a B-mode observable applies it, and
    every other EE/BB mixing term, through `pseudo_to_spice_bmode`.
    """

    #: Two-letter spectra whose PolSpice decoupling mixes them with each
    #: other (docs/theory/bmode_kernels.md): every other spectrum
    #: (TT, TE/ET, TB/BT, EB/BE) is its own single source.
    _MIXED_SOURCES = ("EE", "BB")

    def __init__(self, G_kernels: dict[str, np.ndarray] | np.ndarray) -> None:
        # Handle both dict and array inputs for G_kernels
        if isinstance(G_kernels, dict):
            self.G_kernels = G_kernels
        else:
            # Covariance layout (G0, Gplus, Gminus, Gx): Gplus is already the
            # EE <- EE kernel, so unlike the legacy layout there is nothing to
            # average here.
            #
            # "BB" is only used by `pseudo_to_spice` (never
            # `pseudo_to_spice_bmode`), for the BB-only NKA/INKA escape hatch
            # (generator.parameter_validation.APPROXIMATIONS_SUPPORTING_B,
            # docs/theory/bmode_kernels.md): the diagonal,
            # level-1-style transform Cov_hat(BB) = Gplus Cov(BB) Gplus^T,
            # neglecting the Gminus Cov(..., EE) mixing term that
            # `pseudo_to_spice_bmode` would add (that path needs pseudo EE
            # blocks this mode never computes).
            self.G_kernels = {
                "TT": G_kernels[0],
                "EE": G_kernels[1],
                "BB": G_kernels[1],
                "TE": G_kernels[3],
                "ET": G_kernels[3],
            }
            self.G_mixing = G_kernels[2]

    def pseudo_to_spice(self, cov_matrix: np.ndarray, stokes_key: str) -> np.ndarray:
        """
        Transform pseudo-Cl covariance to PolSpice format.

        Parameters
        ----------
        cov_matrix : np.ndarray
            Input covariance matrix in pseudo-Cl format
        stokes_key : str
            Stokes parameter combination (e.g., "TTxTT", "EExEE", "TTxEE")

        Returns
        -------
        np.ndarray
            Transformed covariance matrix in PolSpice format

        Notes
        -----
        The kernels are indexed ``[l, l']`` and act as
        ``C_hat_l = sum_l' G_ll' C_tilde_l'`` (Eqs. 47-50), so the covariance
        of the PolSpice spectra is ``G_left Sigma G_right^T`` (Eq. 55). The
        transposed form ``G^T Sigma G`` is only right for kernels read
        straight from a Fortran FITS file, which `MaskWlm` now normalises on
        load.
        """
        left, right = stokes_key.split("x")
        return self.G_kernels[left] @ cov_matrix @ self.G_kernels[right].T

    def sources(self, stokekey: str) -> tuple[str, ...]:
        """
        The two-letter pseudo spectra ``stokekey`` decouples into.

        ``"EE"`` and ``"BB"`` both return ``("EE", "BB")``: PolSpice
        decoupling mixes them (docs/theory/bmode_kernels.md).
        Every other spectrum returns ``(stokekey,)``, itself alone --
        including ``"TB"``/``"BT"`` and ``"EB"``/``"BE"``, which are read off
        a single real-space column and never mix with ``EE``/``BB`` or with
        each other in the mean.
        """
        if stokekey in self._MIXED_SOURCES:
            return self._MIXED_SOURCES
        return (stokekey,)

    def _bmode_kernel(self, output_key: str, source_key: str) -> np.ndarray:
        r"""
        The kernel :math:`G_{X\leftarrow a}` of
        docs/theory/bmode_kernels.md, for output spectrum
        ``output_key`` and pseudo source spectrum ``source_key``.

        ======================  ==============================================
        ``(output, source)``    kernel
        ======================  ==============================================
        ``TT, TT``               :math:`^{0}G`
        ``TE/ET, TE/ET``         :math:`^{\times}G`
        ``TB/BT, TB/BT``         :math:`^{\times}G` (same real-space step as TE)
        ``EE, EE`` / ``BB, BB``  :math:`^{+}G`
        ``EE, BB`` / ``BB, EE``  :math:`^{-}G`
        ``EB/BE, EB/BE``         :math:`^{+}G-{}^{-}G={}^{-2}G` (never decoupled)
        ======================  ==============================================
        """
        if output_key in self._MIXED_SOURCES:
            if source_key not in self._MIXED_SOURCES:
                raise ValueError(
                    f"{output_key!r} only mixes with {self._MIXED_SOURCES}, "
                    f"got source {source_key!r}"
                )
            return self.G_kernels["EE"] if output_key == source_key else self.G_mixing
        if source_key != output_key:
            raise ValueError(
                f"{output_key!r} has no cross-spectrum PolSpice source "
                f"{source_key!r}: only EE and BB mix"
            )
        if output_key == "TT":
            return self.G_kernels["TT"]
        if output_key in ("TE", "ET", "TB", "BT"):
            return self.G_kernels["TE"]
        if output_key in ("EB", "BE"):
            return self.G_kernels["EE"] - self.G_mixing
        raise ValueError(f"no PolSpice kernel for spectrum {output_key!r}")

    def pseudo_to_spice_bmode(
        self,
        stokes_key: str,
        block_provider: Callable[[str, str], np.ndarray],
    ) -> np.ndarray:
        r"""
        Transform a pseudo-:math:`C_\ell` covariance block to PolSpice format
        for a run with a B-mode observable.

        Implements docs/theory/bmode_kernels.md:
        :math:`\hat C^{EE} = {}^{+}G\,\tilde C^{EE} + {}^{-}G\,\tilde C^{BB}`,
        :math:`\hat C^{BB} = {}^{-}G\,\tilde C^{EE} + {}^{+}G\,\tilde C^{BB}`
        (EE and BB mix); TE/ET and TB/BT each take :math:`^{\times}G` on
        themselves alone; EB/BE takes :math:`^{+}G-{}^{-}G` on itself alone,
        never decoupled. So

        .. math::
            \operatorname{Cov}(\hat C^X, \hat C^Y) = \sum_{a\in\mathrm{src}(X)}
                \sum_{b\in\mathrm{src}(Y)} G_{X\leftarrow a}\,
                \operatorname{Cov}(\tilde C^a, \tilde C^b)\, G_{Y\leftarrow b}^{T},

        with :math:`\mathrm{src}(EE) = \mathrm{src}(BB) = \{EE, BB\}` (see
        `sources`) and :math:`\mathrm{src}(X) = \{X\}` otherwise, so a block
        needs at most 4 pseudo blocks and exactly 1 when neither ``X`` nor
        ``Y`` is EE or BB.

        Parameters
        ----------
        stokes_key : str
            e.g. ``"EExEE"``, ``"EExBB"``, ``"TBxTB"``.
        block_provider : callable
            ``block_provider(a, b)`` returns the pseudo-:math:`C_\ell`
            covariance :math:`\operatorname{Cov}(\tilde C^a, \tilde C^b)` for
            two-letter source spectra ``a``, ``b`` (each either one half of
            ``stokes_key`` or, when that half is ``EE``/``BB``, the other of
            the two). Called at most 4 times, and reused when a is/isn't a
            direct block (the caller is expected to memoise pseudo blocks
            across output blocks, since several outputs share them).

        Returns
        -------
        np.ndarray
        """
        left, right = stokes_key.split("x")
        total = None
        for a in self.sources(left):
            g_left = self._bmode_kernel(left, a)
            for b in self.sources(right):
                g_right = self._bmode_kernel(right, b)
                term = g_left @ block_provider(a, b) @ g_right.T
                total = term if total is None else total + term
        return total

    def apply_debiasing(
        self,
        cov_matrix: np.ndarray,
        freq_key: str,
        stokes_key: str,
        debiasing_dict: dict,
        lmax: int,
    ) -> np.ndarray:
        """Apply calibration corrections to covariance matrix."""
        # An empty dict is 'no corrections', not 'corrections that happen to be
        # missing'; without this it slips past a None check straight into a
        # KeyError on the first frequency pair.
        if not debiasing_dict:
            return cov_matrix

        f1, f2 = freq_key.split("x")
        p1, p2 = stokes_key.split("x")

        correction_matrix = np.outer(
            debiasing_dict[f1][p1][:lmax], debiasing_dict[f2][p2][:lmax]
        )

        return cov_matrix * correction_matrix

    def apply_Dl_scaling(self, cov_matrix: np.ndarray, lmax: int) -> np.ndarray:
        """Apply D_ell scaling to covariance matrix."""
        ell_values = np.arange(lmax)
        scaling_factor = ell_values * (ell_values + 1) / (2 * np.pi)
        return np.outer(scaling_factor, scaling_factor) * cov_matrix
