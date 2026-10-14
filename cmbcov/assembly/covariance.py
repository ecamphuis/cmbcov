"""
The covariance operator.

`Cov` owns the mask, caches the coupling kernels, selects an approximation
strategy, and assembles the per-spectrum blocks into the bandpower covariance
matrix.
"""

import hashlib
import json
import os
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..harmonics.sht import DEFAULT_MAP2ALM_ITER
from ..mask.footprint import KERNEL_CACHE_VERSION, MaskWlm
from .binning import BinningManager
from .keys import CovKey, CovKeys, SpecKey
from .postprocess import CovariancePostProcessor

__all__ = ["Cov", "CovarianceConfig", "CovarianceMethod", "RAW_BLOCK_CACHE_VERSION"]

#: Bump whenever a change alters the values of a raw (pseudo-C_l, unbinned)
#: covariance block for the same recorded inputs, so that blocks cached with
#: ``save_raw_blocks`` by older code are recomputed rather than reused.
RAW_BLOCK_CACHE_VERSION = 2


#: ``centralell`` values already warned about in this process, see
#: :func:`warn_low_centralell_once`.
_WARNED_LOW_CENTRALELL: set = set()


def warn_low_centralell_once(centralell: int, stacklevel: int = 3) -> None:
    """
    Warn that ``centralell <= 150`` may cost ACC accuracy, once per value per
    process.
    """
    if centralell in _WARNED_LOW_CENTRALELL:
        return
    _WARNED_LOW_CENTRALELL.add(centralell)
    warnings.warn(
        f"centralell={centralell} is quite low, consider using a higher value "
        "for better accuracy (warned once per process)",
        UserWarning,
        stacklevel=stacklevel,
    )


def has_b_observable(covariance_keys: CovKeys) -> bool:
    """
    Whether a run over ``covariance_keys`` has an observable with a B letter
    (``BB``, ``TB``, ``EB``). Such a run assembles every block, its
    T/E blocks included, with the per-Wick-term ACC normalisation.
    """
    return any("B" in spec.stokekey() for spec in covariance_keys.spec_keys)


def bb_only_observable(covariance_keys: CovKeys) -> bool:
    """
    Whether every spectrum in ``covariance_keys`` is ``BB``, the one B
    observable NKA and INKA can compute
    (``config.validation.BB_ONLY_OBSERVABLES``,
    docs/theory/bmode_kernels.md): a leakage-neglected mode for a high-``ell``
    ``Cov(BB, BB)`` without an ACC precompute, cross-frequency BB included.
    Any other B observable (``TB``, ``EB``), or ``BB`` alongside a T/E one,
    needs ACC.
    """
    return bool(covariance_keys.spec_keys) and all(
        spec.stokekey() == "BB" for spec in covariance_keys.spec_keys
    )


class CovarianceMethod(Enum):
    """Enumeration of available covariance computation methods."""

    ACC = "acc"
    NKA = "nka"
    INKA = "inka"


@dataclass
class CovarianceConfig:
    """
    Base configuration for covariance computation.

    Parameters
    ----------
    method : CovarianceMethod, default CovarianceMethod.NKA
        Covariance computation method.
    lmax : int, default 2048
        Maximum multipole for computation.
    lmin : int, default 0
        Lowest multipole retained in the bandpowers. Multipoles below it carry
        no binning weight; a bin straddling ``lmin`` is truncated, not dropped.
    Dl : bool, default False
        Whether to use D_ell = ell(ell+1)/(2π) scaling instead of C_ell.
    sum_asymmetric_stokes : bool, default False
        Whether to combine TE and ET cross-spectra (TE + ET).
    verbose : bool, default False
        Whether to print detailed progress information.
    debiasing_dict : dict, optional
        Per-leg correction factors, ``debiasing_dict[freq][stokes][ell]``.
    apodizetype : int, default 1
        Apodization type: 0 for Gaussian, 1 for cosine (preferred)
    apodizesigma : float, default 30.0
        Apodization scale in DEGREES. The underlying PolSpice code
        (``cor2cl.F90``) parses this as degrees and bounds it at 180; the
        default of 30 corresponds to the theta_max = pi/6 used in Camphuis
        et al. (2022), not to 30 arcminutes.
    thetamax : float, default 30.0
        Maximum angular separation retained in the correlation function, in
        DEGREES. Should be the maximum angular size of the weighted mask.
    dmax : int, optional
        Number of diagonal bands of the covariance matrix computed by ACC.
        Higher values are more accurate and more expensive. Required for ACC.
    centralell : int, optional
        Central multipole of the ACC coupling kernels. Should be low enough
        for efficiency but high enough for signal support. Required for ACC;
        a value of 150 or below warns once.
    acc_term_selection : float, optional
        ACC only: the term-selection tolerance the coupling kernels were
        precomputed with (``precompute_acc_kernels(..., term_selection=)``),
        or ``None`` for the full kernels. It is compared with the kernel-cache
        manifest on every load, so full and selected kernels are never mixed.
    polspice_postprocess : bool, default True
        Whether the pseudo-C_l covariance is turned into the covariance of the
        PolSpice estimator (Eq. 55, ``G Sigma G^T``) before D_ell scaling,
        debiasing and binning. ``False`` returns the binned pseudo-C_l
        covariance. With ``True``, decoupling EE also needs the pseudo BB
        block (docs/theory/bmode_kernels.md).
    save_raw_blocks : bool, default False
        Cache each raw covariance block (the pseudo-C_l block a strategy
        returns, before the PolSpice transform, D_ell scaling, debiasing and
        binning) in ``Cov.save_dir`` as ``cov_<stokes>_<freq>.npy`` with a
        manifest of everything the block depends on (method, lmax, the ACC
        ``lmax_int``, ``dmax`` and ``centralell``, the mask and its alm solve,
        the ACC kernel cache identity and a digest of the exact spectra the
        strategy reads). A later computation reuses a block only on an exact
        manifest match and otherwise recomputes and overwrites it. Off by
        default: nothing is read or written.
    filter_rho : dict, optional
        Filtered sum-rule ratios ``{channel: rho}``, each ``rho(l1, l2)`` of
        shape ``(lmax, lmax)`` (from :func:`cmbcov.assembly.filtering.rho_matrix`),
        keyed by the channels of :data:`cmbcov.assembly.filtering.CHANNELS`
        (``"00"``, ``"20"``, ``"EE"``). Every raw T/E block with Stokes pair
        ``(s1, s2)`` is multiplied elementwise by
        ``rho[ch] / (fl_left[s1][l1] fl_right[s2][l2])``, ``ch =
        filtering.BLOCK_CHANNEL[(s1, s2)]``, before the PolSpice transform,
        D_ell scaling, debiasing and binning; a T/E block whose channel is
        missing raises ``ValueError``. Blocks with a B leg are left unchanged
        and the run warns once. The ``save_raw_blocks`` cache keeps the
        uncorrected block. ``None`` (default): no correction.
    filter_fl : dict, optional
        ``filter_fl[freq_pair][stokes]``: transfer function (length at least
        ``lmax``) of each frequency pair of the run and each spectrum
        (``"TT"``, ``"TE"``, ``"EE"``; ``"ET"`` falls back to ``"TE"``), for
        ``filter_rho``. ``None`` means ``fl = 1`` (the factor is ``rho``
        alone). ``fl["TE"]`` should be a smooth positive reference such as
        ``sqrt(fl["TT"] fl["EE"])`` (docs/theory/filter_and_bin.md).
    filter_rho_blocks : dict, optional
        Spectrum-aware factors of TT x TT blocks, ``{block: (nodes, g)}``
        keyed by :func:`cmbcov.assembly.filtering.block_id` (``"TTxTT_<freqkey>"``),
        ``g`` of shape ``(len(nodes), dmax + 1)`` from
        :func:`cmbcov.assembly.filtering.spectrum_aware_ratio`. A TT x TT block
        present here is multiplied by ``rho_matrix(nodes, g, lmax)`` alone,
        with no ``fl`` division (the ratio was probed with the spectra the
        block is computed with, transfer function included); the matrix is
        built when the block is consumed and not kept. Every other T/E block
        keeps the channel factor of ``filter_rho``. ``None`` (default): no
        block factor.
    """

    method: str = CovarianceMethod.NKA
    lmax: int = 2048
    lmin: int = 0
    Dl: bool = False
    sum_asymmetric_stokes: bool = False
    verbose: bool = False
    debiasing_dict: dict | None = None
    apodizetype: int | None = 1
    apodizesigma: float | None = 30.0
    thetamax: float | None = 30.0
    dmax: int | None = None
    centralell: int | None = None
    save_raw_blocks: bool = False
    acc_term_selection: float | None = None
    polspice_postprocess: bool = True
    filter_rho: dict[str, np.ndarray] | None = None
    filter_fl: dict[str, dict[str, np.ndarray]] | None = None
    filter_rho_blocks: dict[str, tuple[Sequence[int], np.ndarray]] | None = None

    def _validate_filter_rho_blocks(self) -> None:
        """``filter_rho_blocks``: TT x TT block ids mapped to ``(nodes, g)``
        that :func:`cmbcov.assembly.filtering.rho_matrix` accepts."""
        blocks = self.filter_rho_blocks
        if not isinstance(blocks, Mapping):
            raise ValueError(
                "filter_rho_blocks must be a dict {block_id: (nodes, g)}, got "
                f"{type(blocks).__name__}"
            )
        for block, value in blocks.items():
            if not isinstance(block, str) or not block.startswith("TTxTT_"):
                raise ValueError(
                    f"filter_rho_blocks key {block!r} is not a TT x TT block id "
                    "('TTxTT_<freqkey>', cmbcov.assembly.filtering.block_id); only the "
                    "TT x TT blocks have a spectrum-aware factor"
                )
            if not isinstance(value, (tuple, list)) or len(value) != 2:
                raise ValueError(
                    f"filter_rho_blocks[{block!r}] must be a pair (nodes, g)"
                )
            nodes = np.asarray(value[0])
            g = np.asarray(value[1])
            if (
                nodes.ndim != 1
                or nodes.size == 0
                or not np.issubdtype(nodes.dtype, np.integer)
                or np.any(nodes <= 0)
                or np.any(np.diff(nodes) <= 0)
            ):
                raise ValueError(
                    f"filter_rho_blocks[{block!r}]: nodes must be strictly "
                    f"increasing positive integers, got {value[0]!r}"
                )
            if g.ndim != 2 or g.shape[0] != nodes.size or g.shape[1] < 1:
                raise ValueError(
                    f"filter_rho_blocks[{block!r}]: g must have shape "
                    f"(len(nodes), dmax + 1) = ({nodes.size}, ...), got {g.shape}"
                )
            if not np.all(np.isfinite(g)):
                raise ValueError(f"filter_rho_blocks[{block!r}]: g must be finite")

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.filter_rho is not None:
            from .filtering import CHANNELS

            if not isinstance(self.filter_rho, Mapping):
                raise ValueError(
                    "filter_rho must be a dict {channel: rho} with channels "
                    f"among {CHANNELS} (e.g. {{'00': rho}} for TT only), got "
                    f"{type(self.filter_rho).__name__}"
                )
            for channel, rho in self.filter_rho.items():
                if channel not in CHANNELS:
                    raise ValueError(
                        f"filter_rho has unknown channel {channel!r}; known: "
                        f"{CHANNELS}"
                    )
                rho = np.asarray(rho)
                if rho.ndim != 2 or rho.shape != (self.lmax, self.lmax):
                    raise ValueError(
                        f"filter_rho[{channel!r}] must have shape (lmax, lmax) "
                        f"= ({self.lmax}, {self.lmax}), got {rho.shape}"
                    )
        if self.filter_rho_blocks is not None:
            self._validate_filter_rho_blocks()
        if self.filter_fl is not None:
            if self.filter_rho is None and self.filter_rho_blocks is None:
                raise ValueError(
                    "filter_fl is set without filter_rho or filter_rho_blocks"
                )
            for pair, by_stokes in self.filter_fl.items():
                if not isinstance(by_stokes, Mapping):
                    raise ValueError(
                        f"filter_fl[{pair!r}] must be a dict {{stokes: fl}} "
                        f"(e.g. {{'TT': fl}}), got {type(by_stokes).__name__}"
                    )
                for stokes, fl in by_stokes.items():
                    if np.ndim(fl) != 1 or len(fl) < self.lmax:
                        raise ValueError(
                            f"filter_fl[{pair!r}][{stokes!r}] must be a 1-d "
                            f"array of length at least lmax = {self.lmax}"
                        )
        if not isinstance(self.polspice_postprocess, (bool, np.bool_)):
            raise ValueError(
                "polspice_postprocess must be true or false, got "
                f"{self.polspice_postprocess!r}"
            )
        if self.apodizetype not in [0, 1]:
            raise ValueError(
                "apodizetype must be 0 (gaussian) or 1 (cos, preferred in general)"
            )

        if self.apodizesigma <= 0:
            raise ValueError("apodizesigma must be positive")
        if self.thetamax <= 0:
            raise ValueError("thetamax must be positive")
        if self.method not in CovarianceMethod:
            raise ValueError(f"method must be one of {list(CovarianceMethod)}")
        if self.method == CovarianceMethod.ACC:
            if self.dmax is None or self.dmax < 1:
                raise ValueError("dmax must be set and at least 1 for ACC method")
            if self.centralell is None or self.centralell < 0:
                raise ValueError(
                    "centralell must be set and non-negative for ACC method"
                )
            if self.centralell <= 150:
                warn_low_centralell_once(self.centralell)
        if self.acc_term_selection is not None and not (
            float(self.acc_term_selection) > 0
        ):
            raise ValueError("acc_term_selection must be a positive tolerance or None")


class Cov:
    """
    Analytical covariance of CMB pseudo-C_l power spectra (ACC, NKA, INKA).

    Handles mask loading, kernel computation and caching, binning, and
    post-processing.
    """

    #: How blocks are post-processed. ``None``: through the projection of
    #: each leg (:meth:`_process_covariance_term`) when that pays, see
    #: :meth:`_folds_postprocess`, and block by block otherwise. Tests set it
    #: true or false to force either path
    #: (:meth:`_process_covariance_term_reference` is the per-block one).
    _fold_postprocess: bool | None = None
    #: Projections pay when a bandpower spans at least this many multipoles
    #: on average (``lmax / n_bins``): a block then costs ``n_bins lmax^2``
    #: instead of ``2 lmax^3 + n_bins lmax^2``, at least 9 times less.
    _FOLD_MIN_ELL_PER_BIN = 4

    def __init__(
        self,
        mask: str,
        config: CovarianceConfig,
        save_dir: str | None = None,
        mask_path: str | None = None,
        acc_kernel_dir: str | None = None,
    ) -> None:
        """
        Initialize covariance computation object.

        Parameters
        ----------
        mask : str
            Filename or full path to the HEALPix mask file.
        config : CovarianceConfig
            Computation parameters, used for all subsequent covariance
            computations. For ACC set ``method=CovarianceMethod.ACC`` and give
            ``dmax`` and ``centralell``.
        save_dir : str, optional
            Directory of the raw per-block covariance cache (default
            ``"./"``), read and written only with
            ``config.save_raw_blocks``. Also the ACC kernel directory unless
            ``acc_kernel_dir`` is given.
        mask_path : str, optional
            Directory containing ``mask`` (default ``"./"``).
        acc_kernel_dir : str, optional
            Directory under which the ACC coupling kernels are read from
            ``covariance_coupling/`` (the ``save_dir`` given to
            :func:`~cmbcov.approximations.acc.precompute_acc_kernels`).
            Defaults to ``save_dir``. Separate because the kernels depend only
            on the mask and ``centralell``, while ``save_dir`` holds per-run
            results
            (:attr:`~cmbcov.config.validation.PipelineConfig.acc_kernel_dir`).
        """
        # Initialize mask and parameters
        self.mask = mask
        self.save_dir = save_dir if save_dir is not None else "./"
        self.acc_kernel_dir = (
            acc_kernel_dir if acc_kernel_dir is not None else self.save_dir
        )
        self.load_path = mask_path if mask_path is not None else "./"
        self.wlm = MaskWlm(self.mask, load_path=self.load_path)
        self.config = config
        self.config.validate()
        self._lmax = config.lmax
        if self._lmax > 2 * self.wlm.nside:
            raise ValueError(
                f"lmax ({self._lmax}) cannot exceed 2*nside ({2*self.wlm.nside})"
            )

        # The estimator's mask W, from which the PolSpice kernels G/K and the
        # MASTER matrix M are built: `wlm` itself.
        self.wlm_full = self.wlm

        # Initialize cached objects
        self._kernels = {}
        #: In-memory memo of validated ACC coupling-kernel loads, keyed by
        #: ``(abspath(acc_kernel_dir), ell, ell_prime, stokes_key,
        #: mask_digest, centralell)``, each holding ``(kernel, file mtime_ns,
        #: acc_cache.write_generation)``; see :meth:`get_acc_coupling_kernels`.
        #: It lives here, not on the strategy, because it depends only on the
        #: mask and must survive the strategy being recreated for every
        #: ``compute_covariance_matrix`` call.
        self._acc_kernel_memo: dict[tuple, tuple[np.ndarray, int | None, int]] = {}

        # Initialize managers
        self.binning_manager = BinningManager(self.lmax, config.lmin)

    def __str__(self) -> str:
        return f"Cov operator based on mask = {self.mask}, save_dir = {self.save_dir}"

    # ================== Properties ==================

    @property
    def lmax(self) -> int:
        return self._lmax

    @lmax.setter
    def lmax(self, ell: int) -> None:
        """
        Lower the maximum multipole and crop the cached kernels.

        Parameters
        ----------
        ell : int
            New maximum multipole value

        Warns
        -----
        UserWarning
            If ``ell`` is not smaller than the current lmax (no action taken).
        """
        if ell >= self._lmax:
            warnings.warn(
                f"New lmax ({ell}) is larger than current lmax ({self._lmax}). "
                "No action taken to avoid data loss.",
                UserWarning,
                stacklevel=2,
            )
            return

        self.config.lmax = ell
        self._lmax = ell
        self.binning_manager = BinningManager(self._lmax, self.config.lmin)

        # Update cached kernels to new size
        for key in list(self._kernels.keys()):
            if key.startswith(("M", "G", "Xi")):
                kernel = self._kernels[key]
                if kernel.ndim == 3:
                    self._kernels[key] = kernel[:, :ell, :ell]
                elif kernel.ndim == 2:
                    self._kernels[key] = kernel[:ell, :ell]

        # Clear norm_Xi cache as it depends on lmax
        self._kernels.pop("norm_Xi", None)

    @property
    def M(self) -> np.ndarray:
        return self._get_kernel("M")

    @property
    def M_footprint(self) -> np.ndarray:
        """
        The MASTER matrix of the footprint :math:`W_s` (INKA's smoothing of
        the base term).
        """
        return self.M

    @property
    def G(self) -> np.ndarray:
        return self._get_kernel("G")

    @property
    def K(self) -> np.ndarray:
        return self._get_kernel("K")

    @property
    def Xi(self) -> np.ndarray:
        return self._get_kernel("Xi")

    @property
    def norm_Xi(self) -> dict:
        return self._get_kernel("norm_Xi")

    # ================== Kernel Management ==================

    def _get_kernel(self, kernel_type: str, **kwargs) -> np.ndarray | dict:
        """Unified kernel getter with caching."""
        cache_key = f"{kernel_type}_{hash(str(sorted(kwargs.items())))}"

        if cache_key not in self._kernels:
            if kernel_type in ("M", "M_footprint"):
                wlm = self.wlm_full if kernel_type == "M" else self.wlm
                self._kernels[cache_key] = wlm.get_master_coupling_kernels(
                    ell_max=self.lmax,
                    use_squared=False,
                    make_symmetric=False,
                )
            elif kernel_type == "G":
                self._kernels[cache_key] = self.wlm_full.get_polspice_master_kernel(
                    ell_max=self.lmax,
                    apodization_type=self.config.apodizetype,
                    apodization_sigma=self.config.apodizesigma,
                    theta_max=self.config.thetamax,
                )
            elif kernel_type == "K":
                self._kernels[cache_key] = self.wlm_full.get_polspice_kernel(
                    ell_max=self.lmax,
                    apodization_type=self.config.apodizetype,
                    apodization_sigma=self.config.apodizesigma,
                    theta_max=self.config.thetamax,
                )
            elif kernel_type == "Xi":
                self._kernels[cache_key] = self.wlm.get_master_coupling_kernels(
                    ell_max=self.lmax,
                )
            elif kernel_type == "norm_Xi":
                self._kernels[cache_key] = self._compute_norm_xi()
            else:
                raise ValueError(f"Unknown kernel type: {kernel_type}")

        return self._kernels[cache_key]

    def _compute_norm_xi(self) -> dict:
        r"""
        The geometric prefactor Xi^{ss'}[W^2] for each pair of Wick kernels.

        The covariance of two pseudo-spectra is a sum of two Wick
        contractions (`CovKey.key_to_cross`), each of which pairs two
        covariance coupling kernels Theta^{s1 x s2}.  Every approximation
        writes that term as

            Xi^{ss'}_{l l'}[W^2]  *  C . (Theta / sum Theta) . C ,

        so the entry below must be the Xi channel that satisfies the paper's
        Eq. 22 for that kernel pair, ``sum Theta^{s1 x s2} = (2l+1)(2l'+1)
        Xi^{ss'}``.  Which channel that is follows a spin-weight rule:

            give each leg a weight  w(TT) = 0, w(EE) = w(BB) = 1,
            w(TE) = w(ET) = 1/2,  and the correct normalisation of
            Theta^{s1 x s2} is  Xi^00 * (Xi^20 / Xi^00) ** (w1 + w2).

        Only integer total weights land on a channel we have:

            w = 0   TT x TT                      Xi^00          exact
            w = 1   TT x EE, TE x TE, TE x ET    Xi^20          exact
            w = 2   EE x EE                      Xi^{EE->EE}    exact
            w = 1/2 TT x TE                      no channel exists
            w = 3/2 EE x TE                      no channel exists

        "Exact" means up to the E->B leakage deficit of the spin-2
        completeness relation, a low-multipole, mask-dependent effect that
        falls roughly as ell^-2.
        For the two half-integer weights the nearest rung is used: Xi^20 for
        the TT x TE family (which slightly overshoots, leaving those blocks
        marginally *below* the exact value) and Xi^{EE->EE} for the EE x TE
        family, the better of the available channels.

        Using Xi^00 for the eight entries carrying TE or ET together with
        TT or TE/ET would be off by a further factor Xi^00/Xi^20.

        There is no ``"BB"`` entry: the dict is the T/E alphabet. The BB-only
        NKA/INKA mode (``config.validation.BB_ONLY_OBSERVABLES``,
        docs/theory/bmode_kernels.md) looks its ``("BB", "BB")`` prefactor up
        as ``("EE", "EE")``, since ``w(BB) = 1`` makes ``BB x BB`` the
        ``Xi^{EE->EE}`` channel, up to the E->B leakage that mode neglects
        (:meth:`~cmbcov.approximations.nka.NKAStrategy.compute_covariance_term`).
        """
        xi_kernels = self.Xi
        xi00 = xi_kernels[0]  # Xi^{00}, the KERNEL_TT channel
        xi20 = xi_kernels[3]  # Xi^{20}, the KERNEL_TE channel
        xi_ee = 0.5 * (xi_kernels[1] + xi_kernels[2])  # Xi^{EE->EE}

        return {
            ("TT", "TT"): xi00,
            ("TT", "EE"): xi20,
            ("TT", "TE"): xi20,
            ("TT", "ET"): xi20,
            ("EE", "TT"): xi20,
            ("EE", "EE"): xi_ee,
            ("EE", "TE"): xi_ee,
            ("EE", "ET"): xi_ee,
            ("TE", "TT"): xi20,
            ("TE", "EE"): xi_ee,
            ("TE", "TE"): xi20,
            ("TE", "ET"): xi20,
            ("ET", "TT"): xi20,
            ("ET", "EE"): xi_ee,
            ("ET", "TE"): xi20,
            ("ET", "ET"): xi20,
        }

    # ============ BB-only NKA/INKA leakage warning ============

    def bb_leakage_ell_safe(
        self,
        cl_bb: np.ndarray,
        cl_ee: np.ndarray,
        lmin: int = 2,
        inflation_threshold: float = 1.05,
    ) -> dict[str, "int | float | bool | np.ndarray"]:
        r"""
        The mask- and spectrum-specific validity threshold of the BB-only
        NKA/INKA mode (docs/theory/bmode_kernels.md,
        ``config.validation.BB_ONLY_OBSERVABLES``).

        NKA/INKA compute ``Cov(BB, BB)`` from ``C^BB`` alone, missing the
        leaked-E term the true pseudo-BB mean has:

        .. math::
            \langle\tilde C^{BB}_\ell\rangle = ({}^{+}M\,C^{BB})_\ell +
                ({}^{-}M\,C^{EE})_\ell,

        with :math:`{}^{\pm}M = \tfrac12(M_1 \pm M_2)` the MASTER
        ``KERNEL_EEpBB``/``KERNEL_EEmBB`` channels of :attr:`M`
        (``mask/coupling.py``; :math:`M_1 = M[1]`, :math:`M_2 = M[2]`).
        Since the pseudo covariance scales as the mean squared, the variance
        this approximation misses is roughly ``(1 + leak/signal)^2``, with

        .. math::
            \mathrm{signal}_\ell = ({}^{+}M\,C^{BB})_\ell, \qquad
            \mathrm{leak}_\ell = ({}^{-}M\,C^{EE})_\ell.

        This tracks the :math:`\ell` dependence of the true exact-to-NKA
        ratio to within a factor of order unity: adequate for a warning, not
        a correction.

        Parameters
        ----------
        cl_bb, cl_ee : ndarray
            The fiducial ``C^BB``, ``C^EE`` this run's mask sees. ``C^EE``
            is read only for this diagnostic
            (:meth:`~cmbcov.spectra.SpectraLoader.read_extra_stokes_spectrum`).
        lmin : int, default 2
            Multipoles below this are excluded from the search: the spin-2
            kernel rows are exactly zero there, so the ratio is undefined.
        inflation_threshold : float, default 1.05
            The largest tolerated ``(1 + leak/signal)^2``: 5% of missed
            variance (about 2.5% on sigma).

        Returns
        -------
        dict
            ``ell_safe`` (int): the smallest multipole above which the
            inflation stays below ``inflation_threshold`` out to ``lmax``
            (``lmax + 1`` if it never does -- unreliable everywhere).
            ``lmax`` (int): the last multipole examined,
            ``min(len(cl_bb), len(cl_ee), M.shape[-1]) - 1``.
            ``lowest_bin_factor`` (float): the inflation at ``lmin``, i.e.
            how many times the true variance of the lowest multipole this
            mode reports exceeds what NKA/INKA compute for it.
            ``reliable`` (bool): whether ``ell_safe <= lmax``.
            ``inflation`` (ndarray, length ``lmax + 1``): the full
            ``(1 + leak/signal)^2`` curve.
        """
        m_plus = 0.5 * (self.M[1] + self.M[2])
        m_minus = 0.5 * (self.M[1] - self.M[2])
        size = min(len(cl_bb), len(cl_ee), m_plus.shape[-1])
        if size <= lmin:
            raise ValueError(
                f"cl_bb/cl_ee/kernel too short ({size} multipoles) to search "
                f"above lmin={lmin}"
            )
        cl_bb = np.asarray(cl_bb[:size])
        cl_ee = np.asarray(cl_ee[:size])
        signal = m_plus[:size, :size] @ cl_bb
        leak = m_minus[:size, :size] @ cl_ee

        safe_signal = np.where(signal != 0, signal, 1.0)
        ratio = np.where(signal != 0, leak / safe_signal, np.inf)
        inflation = (1.0 + ratio) ** 2

        lmax_examined = size - 1
        unsafe = inflation[lmin:] > inflation_threshold
        if not np.any(unsafe):
            ell_safe = lmin
        else:
            last_unsafe = lmin + int(np.max(np.nonzero(unsafe)[0]))
            ell_safe = last_unsafe + 1

        return {
            "ell_safe": int(ell_safe),
            "lmax": int(lmax_examined),
            "lowest_bin_factor": float(inflation[lmin]),
            "reliable": bool(ell_safe <= lmax_examined),
            "inflation": inflation,
        }

    def warn_bb_only_leakage(
        self,
        cl_bb: np.ndarray,
        cl_ee: np.ndarray,
        method: "CovarianceMethod",
        logger=None,
        lmin: int = 2,
        inflation_threshold: float = 1.05,
        stacklevel: int = 3,
    ) -> dict[str, "int | float | bool | np.ndarray"]:
        """
        :meth:`bb_leakage_ell_safe`, plus the ``UserWarning``/log message
        the BB-only NKA/INKA mode always emits
        (docs/theory/bmode_kernels.md). Never raises.
        """
        report = self.bb_leakage_ell_safe(
            cl_bb, cl_ee, lmin=lmin, inflation_threshold=inflation_threshold
        )
        if report["reliable"]:
            message = (
                f"BB-only {method.value} run: leakage from C^EE into the "
                "pseudo C^BB mean is neglected throughout "
                "(docs/theory/bmode_kernels.md). The estimated "
                "variance inflation this misses, (1 + leak/signal)^2, stays "
                f"below {inflation_threshold} for ell >= "
                f"{report['ell_safe']}; bins below that are underestimated "
                f"(by a factor {report['lowest_bin_factor']:.3g} at "
                f"ell={lmin})."
            )
        else:
            message = (
                f"BB-only {method.value} run: leakage from C^EE into the "
                "pseudo C^BB mean is neglected throughout "
                "(docs/theory/bmode_kernels.md), and the "
                f"estimated variance inflation this misses exceeds "
                f"{inflation_threshold} everywhere up to lmax="
                f"{report['lmax']} (ell_safe would be past lmax): this "
                "run's covariance is unreliable at every multipole it "
                f"reports (lowest-bin factor {report['lowest_bin_factor']:.3g} "
                f"at ell={lmin})."
            )
        warnings.warn(message, UserWarning, stacklevel=stacklevel)
        if logger is not None:
            logger.warning(message)
        return report

    # ============ Per-Wick-term normalisation (runs with a B observable) ============

    def acc_xi_channels(self, ell_max: int | None = None) -> dict[str, np.ndarray]:
        r"""
        The four MASTER channels of :math:`W^2` and :math:`\rho` that the
        per-Wick-term ACC normalisation reads
        (docs/theory/bmode_kernels.md), from :attr:`Xi`:

        - ``"00"``: :math:`\Xi^{00}` (``KERNEL_TT``);
        - ``"20"``: :math:`\Xi^{20}` (``KERNEL_TE``);
        - ``"EE"``: :math:`\Xi^{EE\to EE} = \tfrac12(\Xi^{EE+BB} + \Xi^{EE-BB})`;
        - ``"EB"``: :math:`\Xi^{EE\to BB} = \tfrac12(\Xi^{EE+BB} - \Xi^{EE-BB})`;
        - ``"rho"``: :math:`\rho = \Xi^{EE\to BB} / \Xi^{EE\to EE}`, set to 0
          where :math:`\Xi^{EE\to EE} = 0` (the rows and columns
          :math:`\ell < 2`) and clipped at 0 from below (roundoff).

        ``ell_max`` (default ``lmax``) is the array size; a size above
        ``lmax`` is computed separately (the ACC normaliser needs
        :math:`\Xi` at :math:`(\ell_*, \ell_* + \Delta)`, which can lie past
        ``lmax``). NKA, INKA and T/E-only ACC runs use
        :meth:`_compute_norm_xi` instead.
        """
        size = self.lmax if ell_max is None else max(int(ell_max), self.lmax)
        cache_key = f"acc_xi_channels_{size}"
        if cache_key not in self._kernels:
            if size == self.lmax:
                xi = self.Xi
            else:
                xi = self.wlm.get_master_coupling_kernels(ell_max=size)
            xi00 = np.asarray(xi[0])
            xi20 = np.asarray(xi[3])
            xi_ee = 0.5 * (xi[1] + xi[2])
            xi_eb = 0.5 * (xi[1] - xi[2])
            rho = np.divide(xi_eb, xi_ee, out=np.zeros_like(xi_ee), where=xi_ee != 0)
            self._kernels[cache_key] = {
                "00": xi00,
                "20": xi20,
                "EE": xi_ee,
                "EB": xi_eb,
                "rho": np.clip(rho, 0.0, None),
            }
        return self._kernels[cache_key]

    @staticmethod
    def _xi_ladder(channels: dict[str, np.ndarray], weight: float) -> np.ndarray:
        r"""
        :math:`\Xi^{(w)}`: :math:`\Xi^{00}`, :math:`\Xi^{20}` and
        :math:`\Xi^{EE\to EE}` at ``w`` = 0, 1, 2, and
        :math:`\Xi^{00}(\Xi^{20}/\Xi^{00})^{w}` in between
        (docs/theory/bmode_kernels.md, Sect. 5).
        """
        if weight == 0:
            return channels["00"]
        if weight == 1:
            return channels["20"]
        if weight == 2:
            return channels["EE"]
        xi00, xi20 = channels["00"], channels["20"]
        ratio = np.divide(xi20, xi00, out=np.zeros_like(xi20), where=xi00 != 0)
        return xi00 * np.clip(ratio, 0.0, None) ** weight

    def acc_term_scale(
        self,
        channel_1: str,
        channel_2: str,
        delta: int,
        kernel_sum: float,
        ell_1: np.ndarray,
        ell_2: np.ndarray,
    ) -> np.ndarray:
        r"""
        The factor ``F_t = N_t / S_*`` by which the ACC value of one expanded
        Wick term is multiplied, with its kernel NOT normalised: the term is
        ``coefficient * (C_{+s} . Theta_* . C_{+s}) * F_t`` at every
        ``(ell_1, ell_2)`` of diagonal ``delta``
        (docs/theory/bmode_kernels.md).

        :math:`N_t = \Xi^{(w)}_{\ell\ell'}\,\rho_{\ell\ell'}^{k/2}\,c_t(\ell)`
        with :math:`c_t^* = S_*/(n_*\Xi^{(w)}_*\rho_*^{k/2})` and, by the
        term's class (:func:`~cmbcov.approximations.acc.bmode_wick.term_class`):

        - ``eq23`` (TT x TT): :math:`N = \Xi^{00}` (Eq. 23);
        - ``deficit`` (:math:`k = 0`): :math:`c = 1 - (1 - c^*)\rho/\rho_*`;
        - ``auto`` (an LL channel) and ``frozen`` (odd :math:`k`, no LL):
          :math:`c = c^*`, so :math:`F = \Xi^{(w)}\rho^{k/2} / (n_*\Xi^{(w)}_*\rho_*^{k/2})`,
          independent of :math:`S_*`;
        - ``cross`` (both channels in TL, LT, DL, LD):
          :math:`c = \pm\tfrac14 + (c^* \mp \tfrac14)(2\ell_*+\Delta+1)/(2\ell+\Delta+1)`,
          :math:`\pm` the sign of :math:`c^*`, :math:`\ell = \min(\ell_1, \ell_2)`,
          then clipped to :math:`[0, \tfrac12]` in magnitude with the sign of
          :math:`c^*` kept (docs/theory/bmode_kernels.md, Sect. 5).

        Every class gives :math:`N_t = S_*/n_*` at :math:`\ell = \ell_*` (the
        term is then exact) except ``eq23``, which is exact there only to the
        accuracy of the identity :math:`\sum\Theta^{TT\times TT} = n\Xi^{00}`.

        Parameters
        ----------
        channel_1, channel_2 : str
            The kernel pair (``COUPLING_CHANNELS`` names).
        delta : int
            Diagonal offset: the kernel is the one at
            :math:`(\ell_*, \ell_* + \Delta)`.
        kernel_sum : float
            :math:`S_* = \sum\Theta_*` of the raw kernel, in the GL
            convention (no :math:`\sqrt2` per spin-2 integral).
        ell_1, ell_2 : ndarray of int
            Multipoles at which :math:`F` is wanted (both below ``lmax``).
        """
        from ..approximations.acc.bmode_wick import (
            leakage_legs,
            spin_weight,
            term_class,
        )

        ls = int(self.config.centralell)
        lsp = ls + int(delta)
        if ls < 2:
            raise ValueError(
                "the per-Wick-term ACC normalisation needs centralell >= 2 "
                f"(got {ls}): rho = Xi^(EE->BB)/Xi^(EE->EE) is undefined below"
            )
        channels = self.acc_xi_channels(lsp + 1)
        weight = spin_weight(channel_1, channel_2)
        k = leakage_legs(channel_1, channel_2)
        kind = term_class(channel_1, channel_2)
        ladder = self._xi_ladder(channels, weight)
        rho = channels["rho"]
        ell_1 = np.asarray(ell_1)
        ell_2 = np.asarray(ell_2)

        if kind == "eq23":
            return ladder[ell_1, ell_2] / kernel_sum

        m_here = ladder[ell_1, ell_2] * rho[ell_1, ell_2] ** (k / 2)
        m_star = ladder[ls, lsp] * rho[ls, lsp] ** (k / 2)
        n_star = (2 * ls + 1) * (2 * lsp + 1)
        if not np.isfinite(m_star) or m_star <= 0:
            raise ValueError(
                f"Xi^(w) rho^(k/2) at (l*, l*+Delta) = ({ls}, {lsp}) is "
                f"{m_star!r} for the kernel pair {channel_1}x{channel_2}; the "
                "per-Wick-term normalisation needs it positive"
            )
        if kind in ("auto", "frozen"):
            return m_here / (n_star * m_star)

        c_star = kernel_sum / (n_star * m_star)
        if kind == "deficit":
            if not rho[ls, lsp] > 0:
                raise ValueError(
                    f"rho at (l*, l*+Delta) = ({ls}, {lsp}) is {rho[ls, lsp]!r}; "
                    "the k = 0 deficit rule needs it positive"
                )
            c = 1.0 - (1.0 - c_star) * rho[ell_1, ell_2] / rho[ls, lsp]
        elif kind == "cross":
            sign = 1.0 if c_star >= 0 else -1.0
            quarter = 0.25 * sign
            ell_min = np.minimum(ell_1, ell_2)
            relax = (2 * ls + delta + 1) / (2 * ell_min + delta + 1)
            c = quarter + (c_star - quarter) * relax
            c = sign * np.clip(sign * c, 0.0, 0.5)
        else:  # pragma: no cover - term_class returns one of the five
            raise ValueError(f"unknown term class {kind!r}")
        return m_here * c / kernel_sum

    def acc_term_normaliser(
        self,
        channel_1: str,
        channel_2: str,
        delta: int,
        kernel_sum: float,
        ell_1: np.ndarray,
        ell_2: np.ndarray,
    ) -> np.ndarray:
        r"""
        :math:`N_t(\ell_1, \ell_2)`, i.e. :meth:`acc_term_scale` times
        :math:`S_*`: the normalisation applied to the unit-sum translated
        kernel of one expanded Wick term (docs/theory/bmode_kernels.md,
        Sect. 5).
        """
        return kernel_sum * self.acc_term_scale(
            channel_1, channel_2, delta, kernel_sum, ell_1, ell_2
        )

    # ================== ACC internal band limit ==================

    def acc_internal_lmax(
        self, pairs: Iterable[tuple[str, str]] | None = None
    ) -> int | None:
        """
        ``lmax_int``, the number of multipoles the spectra of an ACC
        computation must hold: ``lmax + max(0, S - 1 - centralell)``, with
        ``S`` the largest kernel of this run's cache (the pairs
        ``(centralell, centralell + d)``, ``d < dmax``, restricted to
        ``pairs`` if given). See
        :func:`~cmbcov.approximations.acc.acc_window_pad` for why this is
        the exact requirement. ``None`` when the cache holds no such kernel
        file. The covariance is still reported and binned to ``lmax``.

        ``lmax_int`` is not bounded by the mask (the ACC term reads the mask
        only through ``Xi[l1, l2]`` with ``l1, l2 < lmax``, and through the
        precomputed kernels); it is limited by how far the spectra reach.

        Raises
        ------
        ValueError
            If the method is not ACC.
        """
        from ..approximations.acc.bands import acc_cached_kernel_size, acc_internal_lmax

        if self.config.method != CovarianceMethod.ACC:
            raise ValueError(
                "acc_internal_lmax applies to the ACC method only, not "
                f"{self.config.method}"
            )
        size = acc_cached_kernel_size(
            self.acc_kernel_dir, self.config.centralell, self.config.dmax, pairs
        )
        if size is None:
            return None
        lmax_int = acc_internal_lmax(self.lmax, size, self.config.centralell)
        return lmax_int

    @staticmethod
    def _acc_kernel_pairs(covariance_keys: CovKeys) -> list[tuple[str, str]]:
        """
        The ACC kernel channel pairs the keys' Wick contractions load.

        For a run with a B observable (:func:`has_b_observable`) this is the
        union, over every block and both of its orientations, of the pairs
        of its expanded Wick terms (with the ``C^TB``/``C^EB`` terms): a
        superset of what the run loads, which is enough for
        :meth:`acc_internal_lmax` (it takes the largest kernel file present).
        """
        pairs = set()
        if has_b_observable(covariance_keys):
            from ..approximations.acc.bmode_wick import covkey_wick_terms

            for cov_key in covariance_keys.keys():
                for key in (cov_key, cov_key.transpose()):
                    for term in covkey_wick_terms(key.stoke, key.freq, parity_odd=True):
                        pairs.add((term.channel_1, term.channel_2))
            return sorted(pairs)
        for cov_key in covariance_keys.keys():
            for contraction in cov_key.key_to_cross_kernel():
                pairs.add(
                    (contraction[0].kernel_stokekey(), contraction[1].kernel_stokekey())
                )
        return sorted(pairs)

    # ================== Input Validation ==================

    def _validate_inputs(
        self, covariance_keys: CovKeys, cl: dict[str, dict[str, np.ndarray]]
    ) -> None:
        """
        Check that ``cl[freqkey][stokekey]`` exists and is non-empty for
        every spectrum of ``covariance_keys.spec_keys`` (and, for ACC, is at
        least ``lmax_int`` long).

        Raises
        ------
        ValueError
            If required power spectra are missing or invalid.
        """
        # Get all required frequency and stokes combinations from the spec_keys
        required_combinations = set()
        for spec_key in covariance_keys.spec_keys:
            freq_key = spec_key.freqkey()
            stokes_key = spec_key.stokekey()
            required_combinations.add((freq_key, stokes_key))

        # Validate that all required power spectra are present in cl
        for freq_key, stokes_key in required_combinations:
            if freq_key not in cl:
                raise ValueError(
                    f"Missing power spectrum for frequency combination: {freq_key}"
                )

            if stokes_key not in cl[freq_key]:
                raise ValueError(
                    f"Missing {stokes_key} spectrum for frequency {freq_key}"
                )

            # Validate that the spectrum array is not None and has reasonable shape
            spectrum = cl[freq_key][stokes_key]
            if spectrum is None:
                raise ValueError(f"Spectrum cl['{freq_key}']['{stokes_key}'] is None")

            if not hasattr(spectrum, "__len__") or len(spectrum) == 0:
                raise ValueError(
                    f"Spectrum cl['{freq_key}']['{stokes_key}'] appears to be empty or invalid"
                )

        # ACC reads the spectra past lmax, to lmax_int (see acc_internal_lmax).
        if self.config.method == CovarianceMethod.ACC:
            lmax_int = self.acc_internal_lmax(self._acc_kernel_pairs(covariance_keys))
            if lmax_int is not None:
                for freq_key, stokes_key in sorted(required_combinations):
                    length = len(cl[freq_key][stokes_key])
                    if length < lmax_int:
                        raise ValueError(
                            f"Spectrum cl['{freq_key}']['{stokes_key}'] has {length} "
                            f"multipoles; the ACC method needs {lmax_int} "
                            f"(lmax_int = lmax {self.lmax} + pad "
                            f"{lmax_int - self.lmax}, see Cov.acc_internal_lmax) so "
                            "that no coupling window is cut at lmax. The covariance "
                            "is still reported to lmax."
                        )

    # ================== Utility Methods ==================

    def _init_binned_output(
        self, covariance_keys: CovKeys, num_bins: int, dtype: str = "float64"
    ) -> np.ndarray:
        """
        Initialize binned output array for covariance matrix.

        Parameters
        ----------
        covariance_keys : CovKeys
            Keys defining the structure of the covariance matrix
        num_bins : int
            Number of bins per covariance block
        dtype : str, default "float64"
            Data type for the output array

        Returns
        -------
        np.ndarray
            Initialized zero array of shape (len(keys)*num_bins, len(keys)*num_bins)
        """
        return np.zeros(
            (len(covariance_keys) * num_bins, len(covariance_keys) * num_bins),
            dtype=dtype,
        )

    def _sum_asymmetric_stokes(
        self, cov_matrix: np.ndarray, cov_keys: CovKeys, num_bins: int
    ) -> np.ndarray:
        """Sum asymmetrical Stokes parameters (TE + ET)."""
        output_matrix, new_cov_keys, shift_mapping = cov_keys.get_asymmetrical_stokes(
            num_bins
        )

        for cov_key, value_list in shift_mapping.items():
            left_index, right_index = new_cov_keys[cov_key]

            for shift_value in value_list:
                try:
                    old_left_index, old_right_index = cov_keys[shift_value]
                    output_matrix[
                        left_index * num_bins : (left_index + 1) * num_bins,
                        right_index * num_bins : (right_index + 1) * num_bins,
                    ] += (
                        0.25
                        * cov_matrix[
                            old_left_index * num_bins : (old_left_index + 1) * num_bins,
                            old_right_index
                            * num_bins : (old_right_index + 1)
                            * num_bins,
                        ]
                    )
                except KeyError:
                    old_left_index, old_right_index = cov_keys[shift_value.transpose()]
                    output_matrix[
                        left_index * num_bins : (left_index + 1) * num_bins,
                        right_index * num_bins : (right_index + 1) * num_bins,
                    ] += (
                        0.25
                        * cov_matrix[
                            old_left_index * num_bins : (old_left_index + 1) * num_bins,
                            old_right_index
                            * num_bins : (old_right_index + 1)
                            * num_bins,
                        ].T
                    )

            # Symmetrize
            if left_index != right_index:
                output_matrix[
                    right_index * num_bins : (right_index + 1) * num_bins,
                    left_index * num_bins : (left_index + 1) * num_bins,
                ] = output_matrix[
                    left_index * num_bins : (left_index + 1) * num_bins,
                    right_index * num_bins : (right_index + 1) * num_bins,
                ].T

        return output_matrix

    # ============== ACC coupling-kernel cache (validated, memoised) ==============

    @staticmethod
    def _normalize_stokes_key(key: str | tuple) -> tuple[str, str]:
        """``"TTxTT"`` or ``("TT", "TT")`` -> ``("TT", "TT")``, for memo keys."""
        if isinstance(key, str):
            stokes_1, stokes_2 = key.split("x")
            return (stokes_1, stokes_2)
        stokes_1, stokes_2 = key
        return (stokes_1, stokes_2)

    def get_acc_coupling_kernels(
        self,
        ell: int,
        ell_prime: int,
        known_spectra: Sequence[str],
        default_pairs: Sequence[tuple[str, str]],
        pairs: Iterable[tuple[str, str]] | None = None,
        legacy_names: dict[str, str] | None = None,
    ) -> dict:
        """
        Load ACC coupling kernels for one ``(ell, ell_prime)`` pair,
        validated against this ``Cov``'s mask and ``centralell``, and served
        from an in-memory memo on repeat requests.

        ``ACCStrategy.compute_covariance_term`` calls this once per diagonal
        offset inside a loop over every block, and a fresh strategy is created
        for every ``compute_covariance_matrix`` call. The kernel files depend
        only on the mask, so the memo lives here to avoid re-reading them
        for every block of every recompute.

        Parameters
        ----------
        ell, ell_prime : int
            The multipole pair the kernels were computed for.
        known_spectra : sequence of str
            Valid kernel channel labels (``COUPLING_CHANNELS``).
        default_pairs : sequence of (str, str)
            The channel pairs loaded when ``pairs`` is ``None``.
        pairs : iterable of (str, str), optional
            The channel pairs to load; defaults to ``default_pairs``.
        legacy_names : dict of str to str, optional
            Canonical channel label to legacy on-disk label, used as a file
            name fallback (see
            :func:`~cmbcov.approximations.acc.cache.load_coupling_kernels`).

        Returns
        -------
        dict
            ``{pair: kernel}``. The arrays are read-only and shared with the
            memo (a copy per access would cost more than it saves); mutating
            one in place raises instead of corrupting later calls.

        Notes
        -----
        A memo entry is written only after a validated
        :func:`~cmbcov.approximations.acc.cache.load_coupling_kernels` call
        and is keyed by the resolved ``acc_kernel_dir``, ``(ell,
        ell_prime)``, the channel pair and this run's identity fields (mask
        digest, ``centralell``, ``config.acc_term_selection``). A hit is
        still checked against the modification time of its file (one
        ``os.stat``) and the in-process write counter of its directory
        (:func:`~cmbcov.approximations.acc.cache.write_generation`): a missing
        or modified file, or any kernel write to that directory in this
        process (e.g. :func:`~cmbcov.approximations.acc.precompute_acc_kernels`)
        drops the entry and reloads with validation. An out-of-process
        rewrite within the filesystem's modification-time resolution is not
        detected.
        """
        from ..approximations.acc import cache as acc_cache

        pairs_list = list(pairs) if pairs is not None else list(default_pairs)
        mask_digest = self.wlm.mask_digest
        centralell = self.config.centralell
        term_selection = getattr(self.config, "acc_term_selection", None)
        kernel_dir = self.acc_kernel_dir
        kernel_dir_abs = os.path.abspath(kernel_dir)
        generation = acc_cache.write_generation(kernel_dir)

        def file_mtime(pair) -> int | None:
            # Own name, legacy name, transposed own name, transposed legacy
            # name: the fallback order load_coupling_kernels loads through, so
            # a memo entry is staleness-checked against the file it came from.
            for candidate_path, _transposed in acc_cache.coupling_kernel_candidates(
                kernel_dir, pair, ell, ell_prime, known_spectra, legacy_names
            ):
                try:
                    return os.stat(candidate_path).st_mtime_ns
                except OSError:
                    continue
            return None

        result = {}
        missing = []
        stamps = {}
        for pair in pairs_list:
            memo_key = (
                kernel_dir_abs,
                ell,
                ell_prime,
                pair,
                mask_digest,
                centralell,
                term_selection,
            )
            cached = self._acc_kernel_memo.get(memo_key)
            mtime = file_mtime(pair)
            if cached is not None and (
                mtime is None or (mtime, generation) != cached[1:]
            ):
                del self._acc_kernel_memo[memo_key]
                cached = None
            if cached is None:
                missing.append(pair)
                stamps[pair] = mtime
            else:
                result[pair] = cached[0]

        if missing:
            loaded = acc_cache.load_coupling_kernels(
                kernel_dir,
                ell,
                ell_prime,
                known_spectra,
                default_pairs,
                pairs=missing,
                current={
                    "mask_digest": mask_digest,
                    "centralell": centralell,
                    "map2alm_iter": DEFAULT_MAP2ALM_ITER,
                    "healpix_kernel_version": acc_cache.HEALPIX_KERNEL_VERSION,
                    "term_selection": term_selection,
                },
                legacy_names=legacy_names,
            )
            for pair, kernel in loaded.items():
                kernel.flags.writeable = False
                memo_key = (
                    kernel_dir_abs,
                    ell,
                    ell_prime,
                    pair,
                    mask_digest,
                    centralell,
                    term_selection,
                )
                # The stamp was taken before the load, so a rewrite racing
                # the load leaves a stale stamp and forces a reload next time
                # rather than pinning the old kernel to the new file.
                self._acc_kernel_memo[memo_key] = (kernel, stamps[pair], generation)
                result[pair] = kernel

        return result

    def write_acc_coupling_kernels(
        self,
        coupling_kernels: dict,
        ell: int,
        ellp: int,
        known_spectra: Sequence[str],
        centralell: int | None,
        verbose: bool = False,
        spectra: Sequence[str] = (),
        grid: str = "gl",
        lw: int | None = None,
        nside: int | None = None,
    ) -> None:
        """
        Save ACC coupling kernels under ``acc_kernel_dir``, recording this
        ``Cov``'s mask digest (see
        :func:`~cmbcov.approximations.acc.cache.write_coupling_kernels`),
        and drop any memo entries :meth:`get_acc_coupling_kernels` holds for
        the ``(ell, ellp)`` pairs this write touches.
        """
        from ..approximations.acc import cache as acc_cache

        acc_cache.write_coupling_kernels(
            self.acc_kernel_dir,
            coupling_kernels,
            ell,
            ellp,
            known_spectra,
            centralell,
            verbose=verbose,
            spectra=spectra,
            grid=grid,
            lw=lw,
            nside=nside,
            mask_digest=self.wlm.mask_digest,
            map2alm_iter=DEFAULT_MAP2ALM_ITER if grid == "healpix" else None,
            healpix_kernel_version=(
                acc_cache.HEALPIX_KERNEL_VERSION if grid == "healpix" else None
            ),
        )

        kernel_dir_abs = os.path.abspath(self.acc_kernel_dir)
        written_pairs = {self._normalize_stokes_key(k) for k in coupling_kernels}
        for memo_key in [
            k
            for k in self._acc_kernel_memo
            if k[0] == kernel_dir_abs
            and k[1] == ell
            and k[2] == ellp
            and k[3] in written_pairs
        ]:
            del self._acc_kernel_memo[memo_key]

    # ================== Main Computation Method ==================

    def compute_covariance_matrix(
        self,
        lbins: list[int] | int,
        covariance_keys: CovKeys,
        cl: dict[str, dict[str, np.ndarray]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the bandpower covariance matrix with the method of
        ``self.config`` (for ACC, ``dmax`` and ``centralell`` must be set).

        Parameters
        ----------
        lbins : list or int
            Binning scheme: a list of bin edges or a step size.
        covariance_keys : CovKeys
            The blocks of the covariance matrix.
        cl : dict
            Power spectra, ``cl[freq_combination][stokes_combination]``, with
            the keys of ``SpecKey.freqkey()`` (e.g. ``"090GHz150GHz"``) and
            ``SpecKey.stokekey()`` (e.g. ``"TT"``, ``"EE"``, ``"TE"``,
            ``"ET"``) of ``covariance_keys.spec_keys``. Length ``lmax`` for
            NKA and INKA; at least ``lmax_int`` for ACC
            (:meth:`acc_internal_lmax`), since ACC reads the spectra past
            ``lmax`` so that no coupling window is cut there.

        Returns
        -------
        tuple of np.ndarray
            ``(binned_ells, binning_matrix, covariance_matrix)``: the mean
            multipole of each bin, the ``(n_bins, lmax)`` binning matrix and
            the covariance of shape ``(len(covariance_keys) * n_bins,) * 2``.

        Raises
        ------
        ValueError
            If ACC method is requested but required ACC parameters are not
            set, or an ACC spectrum is shorter than ``lmax_int``.
        ValueError
            If ``covariance_keys`` has a B observable (``BB``, ``TB``,
            ``EB``) and the method is not ACC: NKA and INKA collapse EE and
            BB by construction. The one exception is ``covariance_keys`` all
            ``BB`` (:func:`bb_only_observable`), NKA/INKA's leakage-neglected
            BB-only mode (docs/theory/bmode_kernels.md).

        Notes
        -----
        A run with a B observable assembles every block, its T/E blocks
        included, as a sum over expanded Wick terms with the per-term
        normalisation of docs/theory/bmode_kernels.md (see
        :meth:`~cmbcov.approximations.acc.ACCStrategy.compute_covariance_term`).
        A T/E-only run uses Eq. 23.

        With ``config.polspice_postprocess`` true, an ACC B run's PolSpice
        transform additionally mixes the EE and BB pseudo blocks
        (docs/theory/bmode_kernels.md, Sect. 7,
        :meth:`~cmbcov.assembly.postprocess.CovariancePostProcessor.pseudo_to_spice_bmode`):
        every raw block is computed once and combined by
        :meth:`_pseudo_block_provider` after the loop. A BB-only NKA/INKA run
        instead uses the diagonal
        :meth:`~cmbcov.assembly.postprocess.CovariancePostProcessor.pseudo_to_spice`
        transform, since it has no pseudo EE block to mix in.

        The PolSpice transform, D_ell scaling, debiasing and binning of a
        block are applied as one matrix per leg, built once per call and
        shared by every block with that leg
        (:meth:`_process_covariance_term`), when
        ``polspice_postprocess`` is true and a bandpower spans at least
        ``_FOLD_MIN_ELL_PER_BIN`` multipoles on average; the result equals
        applying the steps one after the other to round-off.
        """
        self.config.validate()

        b_run = has_b_observable(covariance_keys)
        bb_only = bb_only_observable(covariance_keys)
        # Only NKA/INKA take the BB-only mode; an ACC run with
        # 'observables: [BB]' alone takes the ordinary B-run path.
        bb_only_nka = bb_only and self.config.method != CovarianceMethod.ACC
        if b_run:
            if self.config.method != CovarianceMethod.ACC and not bb_only:
                raise ValueError(
                    "a B-mode observable needs the ACC method: "
                    f"{self.config.method.value} collapses EE and BB by "
                    "construction and cannot compute a block involving B "
                    f"({', '.join(covariance_keys.stokekey())}); the "
                    "exception is 'observables: [BB]' alone, NKA/INKA's "
                    "leakage-neglected escape hatch (see the UserWarning "
                    "this run emits, docs/theory/bmode_kernels.md)."
                )

        # A B run under polspice_postprocess needs every raw pseudo block (up
        # to four per output block: EE and BB legs mix, see
        # postprocess.CovariancePostProcessor.pseudo_to_spice_bmode) before
        # any can be transformed, so the raw blocks are kept for a second
        # pass. A T/E-only run, a B run with polspice_postprocess false and a
        # BB-only NKA/INKA run transform and bin each block as it arrives.
        needs_bmode_mixing = (
            b_run and self.config.polspice_postprocess and not bb_only_nka
        )
        if needs_bmode_mixing:
            stokes_present = set(covariance_keys.combined_stokes())
            if (
                "EE" in stokes_present
                and "BB" not in stokes_present
                and any("B" in s for s in stokes_present)
            ):
                raise ValueError(
                    "covariance_keys has 'EE' and a B observable but not "
                    "'BB': polspice_postprocess needs the pseudo BB block "
                    "to decouple EE (docs/theory/bmode_kernels.md); "
                    "add BB to observables or set "
                    "polspice_postprocess to False "
                    "(config.validation rejects this "
                    "combination earlier for a parameter-file run)."
                )

        # Validate inputs using the covariance keys
        self._validate_inputs(covariance_keys, cl)
        # If the covariance is being converted to D_ell units below, the bin
        # weights must stay uniform; if it stays in C_ell, the bin weights must
        # supply the ell(ell+1)/2pi flattening themselves. Exactly one of the
        # two, hence the negation. See BinningManager.bin_weight.
        bin_matrix, num_bins = self.binning_manager.create_bin_matrix(
            lbins, flatten_with_ell_factor=not self.config.Dl
        )

        # Create computation strategy
        from ..approximations import StrategyFactory

        computation_strategy = StrategyFactory.create_strategy(self)
        weighted = np.flatnonzero(np.any(bin_matrix != 0, axis=0))
        computation_strategy.binned_ell_min = (
            int(weighted[0]) if weighted.size else None
        )
        computation_strategy.configure_run(covariance_keys, cl)

        output_matrix = self._init_binned_output(covariance_keys, num_bins)
        # G is only computed when it is applied; without the PolSpice
        # transform the post-processor only scales, debiases and bins.
        post_processor = CovariancePostProcessor(
            self.G if self.config.polspice_postprocess else {}
        )

        raw_blocks: dict[CovKey, np.ndarray] = {}
        # One projection matrix per spectrum leg, built on first use and
        # dropped with this call (see _leg_projection).
        projections: dict[tuple, np.ndarray] = {}

        # map_filter: each T/E block is multiplied by the factor of its
        # sum-rule channel (cmbcov.assembly.filtering.BLOCK_CHANNEL), or, for a
        # TT x TT block in filter_rho_blocks, by its own spectrum-aware
        # factor; blocks with a B leg are left as they are. The factors are
        # checked here, before anything is computed.
        filter_channel: dict[str, str] = {}
        filter_blocks = self.config.filter_rho_blocks or {}
        if self.config.filter_rho is not None or self.config.filter_rho_blocks:
            from .filtering import BLOCK_CHANNEL, block_id

            white = self.config.filter_rho or {}
            uncorrected = []
            for stokes in covariance_keys.stokekey():
                channel = BLOCK_CHANNEL.get(tuple(stokes.split("x")))
                if channel is None:
                    uncorrected.append(stokes)
                    continue
                if stokes == "TTxTT" and all(
                    block_id(k) in filter_blocks
                    for k in covariance_keys.keys()
                    if k.stokekey() == stokes
                ):
                    continue
                if channel not in white:
                    raise ValueError(
                        f"map_filter: block {stokes} needs the sum-rule "
                        f"channel {channel!r}, which filter_rho does not have "
                        f"(it has {sorted(white)}"
                        + (
                            "; filter_rho_blocks does not cover every "
                            "TT x TT block either"
                            if stokes == "TTxTT" and filter_blocks
                            else ""
                        )
                        + "); compute it with "
                        "filtering.filtered_sum_rule_ratio(..., "
                        "channels=filtering.channels_for(...))."
                    )
                filter_channel[stokes] = channel
            if uncorrected:
                warnings.warn(
                    "map_filter: the filter-and-bin correction covers the T/E "
                    "blocks only; the blocks with a B leg "
                    f"({', '.join(uncorrected)}) are left uncorrected "
                    "(docs/theory/filter_and_bin.md).",
                    UserWarning,
                    stacklevel=2,
                )
        filter_factors: dict[tuple[str, str, str, str], np.ndarray] = {}

        def consume(cov_key: CovKey, covariance_term: np.ndarray) -> None:
            channel = filter_channel.get(cov_key.stokekey())
            block_factor = (
                filter_blocks.get(block_id(cov_key)) if filter_blocks else None
            )
            if block_factor is not None:
                # Spectrum-aware TT x TT: rho_matrix of the block's own
                # ratio, no fl division; built here and dropped with the
                # block (a 3-frequency run has 21 such blocks).
                from .filtering import rho_matrix

                covariance_term = covariance_term * rho_matrix(
                    block_factor[0], block_factor[1], self.lmax
                )
            elif channel is not None:
                covariance_term = covariance_term * self._filter_factor(
                    cov_key, channel, filter_factors
                )
            if needs_bmode_mixing:
                raw_blocks[cov_key] = covariance_term
            else:
                # Post-process and add to output matrix
                self._add_to_output_matrix(
                    output_matrix,
                    covariance_term,
                    cov_key,
                    covariance_keys,
                    bin_matrix,
                    num_bins,
                    post_processor,
                    projections,
                )

        # Blocks read back from the save_raw_blocks cache are used as they
        # come; the others are computed by the strategy's
        # compute_covariance_terms, which may compute blocks together and
        # yield them in its own order (ACC shares kernel products between
        # blocks). Each block lands at its own place, so the order is
        # irrelevant.
        manifests: dict[CovKey, dict] = {}
        to_compute: list[CovKey] = []
        for cov_key in covariance_keys.keys():
            covariance_term = None
            if self.config.save_raw_blocks:
                manifest = self._raw_block_manifest(computation_strategy, cov_key, cl)
                manifests[cov_key] = manifest
                covariance_term = self._load_raw_block(cov_key, manifest)
            if covariance_term is None:
                to_compute.append(cov_key)
            else:
                consume(cov_key, covariance_term)

        for cov_key, covariance_term in computation_strategy.compute_covariance_terms(
            to_compute, cl
        ):
            if self.config.verbose:
                print(f"Computed {cov_key}")
            if cov_key in manifests:
                self._save_raw_block(cov_key, covariance_term, manifests[cov_key])
            consume(cov_key, covariance_term)

        if needs_bmode_mixing:
            spec_index = {spec: i for i, spec in enumerate(covariance_keys.spec_keys)}
            for cov_key in covariance_keys.keys():
                block_provider = self._pseudo_block_provider(
                    cov_key, raw_blocks, spec_index, covariance_keys.spec_keys
                )
                self._add_to_output_matrix_bmode(
                    output_matrix,
                    cov_key,
                    covariance_keys,
                    bin_matrix,
                    num_bins,
                    post_processor,
                    block_provider,
                    projections,
                )

        # Handle asymmetric Stokes if requested
        if self.config.sum_asymmetric_stokes:
            output_matrix = self._sum_asymmetric_stokes(
                output_matrix, covariance_keys, num_bins
            )

        # Return results
        binned_ells = bin_matrix @ np.arange(self.lmax)
        return binned_ells, bin_matrix, output_matrix

    def error_budget(
        self,
        covariance_keys: CovKeys,
        band_edges: Sequence[int] | None = None,
    ) -> dict | None:
        """
        The known-error report of this run's method, or ``None`` if it has
        none.

        Only ACC has one (for a TT-only run or a run with a B observable,
        the leakage and translation sections are reported as not
        applicable); see
        :meth:`~cmbcov.approximations.acc.ACCStrategy.error_budget` and
        :mod:`~cmbcov.approximations.acc.budget`. It reads the coupling
        kernels this ``Cov`` already loads and ``norm_Xi``; it computes no new
        kernel and does not touch the covariance.

        Parameters
        ----------
        covariance_keys : CovKeys
            The blocks the run computes.
        band_edges : sequence of int, optional
            Bandpower edges, quoted with ``|l - l*|``; defaults to
            ``[lmin, lmax - 1]``.
        """
        from ..approximations import StrategyFactory

        strategy = StrategyFactory.create_strategy(self)
        return strategy.error_budget(covariance_keys, band_edges=band_edges)

    def _filter_factor(
        self,
        cov_key: CovKey,
        channel: str,
        cache: dict[tuple[str, str, str, str], np.ndarray],
    ) -> np.ndarray:
        """
        The ``(lmax, lmax)`` factor ``rho[channel] / (fl_left[s1]
        fl_right[s2])`` of a T/E block (``config.filter_rho`` /
        ``filter_fl``), kept in ``cache`` for the blocks with the same
        frequency pairs and Stokes pair. ``fl`` of ``"ET"`` falls back to
        that of ``"TE"``.
        """
        from .filtering import transfer_correction

        left, right = cov_key.freqkey().split("x")
        s1, s2 = cov_key.stokekey().split("x")
        key = (left, right, s1, s2)
        factor = cache.get(key)
        if factor is None:
            fl = self.config.filter_fl
            if fl is None:
                fl_left = fl_right = np.ones(self.lmax)
            else:
                fl_left = self._filter_fl(fl, left, s1)
                fl_right = self._filter_fl(fl, right, s2)
            factor = transfer_correction(
                self.config.filter_rho[channel], fl_left, fl_right
            )
            cache[key] = factor
        return factor

    def _filter_fl(self, fl: dict, pair: str, stokes: str) -> np.ndarray:
        """``fl[pair][stokes]`` (``"ET"`` falling back to ``"TE"``), cut to lmax."""
        if pair not in fl:
            raise ValueError(
                f"map_filter: filter_fl has no entry for frequency pair {pair!r}"
            )
        by_stokes = fl[pair]
        name = stokes
        if name not in by_stokes and name == "ET":
            name = "TE"
        if name not in by_stokes:
            raise ValueError(
                f"map_filter: filter_fl[{pair!r}] has no {stokes!r} transfer "
                f"function (it has {sorted(by_stokes)})"
            )
        return np.asarray(by_stokes[name], dtype=float)[: self.lmax]

    # ================== Raw per-block cache (save_raw_blocks) ==================

    def _covariance_cache_path(self, cov_key: CovKey) -> str:
        """
        Path of the cached raw covariance block of ``cov_key``, a binary
        ``.npy`` (exact round trip), with its manifest at ``<path>.manifest.json``.
        """
        return os.path.join(
            self.save_dir,
            f"cov_{cov_key.stokekey()}_{cov_key.freqkey()}.npy",
        )

    @staticmethod
    def _spectra_digest(spectra: dict[str, np.ndarray]) -> str:
        """blake2b digest of labelled arrays: label, dtype, shape and bytes."""
        digest = hashlib.blake2b(digest_size=16)
        for label in sorted(spectra):
            array = np.ascontiguousarray(spectra[label])
            digest.update(label.encode())
            digest.update(str(array.dtype.str).encode())
            digest.update(str(array.shape).encode())
            digest.update(memoryview(array).cast("B"))
        return digest.hexdigest()

    def _raw_block_manifest(
        self,
        strategy,
        cov_key: CovKey,
        cl: dict[str, dict[str, np.ndarray]],
    ) -> dict:
        """
        Everything the raw block of ``cov_key`` depends on, as JSON-normalised
        data: the cache format, the method, the key, ``lmax``, the mask (its
        digest, the alm solve behind Xi/M and the kernel cache version), a
        digest of the exact spectra the strategy reads for this key, and the
        strategy's own identity fields
        (:meth:`~cmbcov.approximations.base.CovarianceStrategy.raw_block_inputs`;
        for ACC ``lmax_int``, ``dmax``, ``centralell`` and the kernel cache).
        Not in it: binning, ``lmin``, ``Dl``, debiasing and the PolSpice
        apodisation, which act only after the raw block.
        """
        spectra, identity = strategy.raw_block_inputs(cov_key, cl)
        alm_lmax, alm_maxiter, alm_epsilon = self.wlm._pending_alm_solve_params()
        manifest = {
            "raw_block_cache_version": RAW_BLOCK_CACHE_VERSION,
            "method": self.config.method.value,
            "stokes": cov_key.stokekey(),
            "frequencies": cov_key.freqkey(),
            "lmax": int(self.lmax),
            "mask_digest": self.wlm.mask_digest,
            "mask_alm_lmax": int(alm_lmax),
            "mask_alm_maxiter": int(alm_maxiter),
            "mask_alm_epsilon": float(alm_epsilon),
            "kernel_cache_version": KERNEL_CACHE_VERSION,
            "spectra_digest": self._spectra_digest(spectra),
        }
        manifest.update(identity)
        return json.loads(json.dumps(manifest, sort_keys=True))

    def _load_raw_block(self, cov_key: CovKey, manifest: dict) -> np.ndarray | None:
        """The cached raw block of ``cov_key`` if its manifest equals ``manifest``."""
        path = self._covariance_cache_path(cov_key)
        try:
            with open(path + ".manifest.json") as f:
                cached = json.load(f)
        except (OSError, ValueError):
            return None
        if cached != manifest:
            return None
        try:
            block = np.load(path)
        except (OSError, ValueError):
            return None
        if block.shape != (self.lmax, self.lmax):
            return None
        if self.config.verbose:
            print(f"Loaded raw block {cov_key} from {path}")
        return block

    def _save_raw_block(
        self, cov_key: CovKey, block: np.ndarray, manifest: dict
    ) -> None:
        """
        Write the raw block of ``cov_key`` and its manifest, replacing both.

        The old manifest is removed first and the new one written last, each
        file through a temporary name, so an interrupted write never leaves a
        valid manifest beside a block it does not describe.
        """
        path = self._covariance_cache_path(cov_key)
        manifest_path = path + ".manifest.json"
        os.makedirs(self.save_dir, exist_ok=True)
        if os.path.exists(manifest_path):
            os.remove(manifest_path)
        tmp = f"{path}.tmp{os.getpid()}.npy"
        np.save(tmp, block)
        os.replace(tmp, path)
        tmp = f"{manifest_path}.tmp{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        os.replace(tmp, manifest_path)

    def _add_to_output_matrix(
        self,
        output_matrix: np.ndarray,
        covariance_term: np.ndarray,
        cov_key: CovKey,
        covariance_keys: CovKeys,
        bin_matrix: np.ndarray,
        num_bins: int,
        post_processor: CovariancePostProcessor,
        projections: dict[tuple, np.ndarray],
    ):
        """Add processed covariance term to output matrix."""
        # Process the covariance term
        if self._folds_postprocess(bin_matrix):
            binned_term = self._process_covariance_term(
                covariance_term, cov_key, post_processor, bin_matrix, projections
            )
        else:
            binned_term = self._process_covariance_term_reference(
                covariance_term, cov_key, post_processor, bin_matrix
            )

        # Add to output matrix
        left_index, right_index = covariance_keys[cov_key]
        output_matrix[
            left_index * num_bins : (left_index + 1) * num_bins,
            right_index * num_bins : (right_index + 1) * num_bins,
        ] = binned_term

        # Symmetrize if needed
        if left_index != right_index:
            output_matrix[
                right_index * num_bins : (right_index + 1) * num_bins,
                left_index * num_bins : (left_index + 1) * num_bins,
            ] = binned_term.T

    def _add_to_output_matrix_bmode(
        self,
        output_matrix: np.ndarray,
        cov_key: CovKey,
        covariance_keys: CovKeys,
        bin_matrix: np.ndarray,
        num_bins: int,
        post_processor: CovariancePostProcessor,
        block_provider,
        projections: dict[tuple, np.ndarray],
    ):
        """
        Same as :meth:`_add_to_output_matrix`, for a B run under
        ``config.polspice_postprocess``: the PolSpice transform comes from
        :meth:`~cmbcov.assembly.postprocess.CovariancePostProcessor.pseudo_to_spice_bmode`
        and ``block_provider`` (built by :meth:`_pseudo_block_provider`)
        rather than from a single raw block.
        """
        if self._folds_postprocess(bin_matrix):
            binned_term = post_processor.project_bmode(
                cov_key.stokekey(),
                block_provider,
                lambda side, source: self._leg_projection(
                    projections, post_processor, bin_matrix, cov_key, side, source
                ),
            )
        else:
            processed_term = post_processor.pseudo_to_spice_bmode(
                cov_key.stokekey(), block_provider
            )
            binned_term = self._finish_processing_reference(
                processed_term, cov_key, post_processor, bin_matrix
            )

        left_index, right_index = covariance_keys[cov_key]
        output_matrix[
            left_index * num_bins : (left_index + 1) * num_bins,
            right_index * num_bins : (right_index + 1) * num_bins,
        ] = binned_term

        if left_index != right_index:
            output_matrix[
                right_index * num_bins : (right_index + 1) * num_bins,
                left_index * num_bins : (left_index + 1) * num_bins,
            ] = binned_term.T

    def _pseudo_block_provider(
        self,
        cov_key: CovKey,
        raw_blocks: dict[CovKey, np.ndarray],
        spec_index: dict[SpecKey, int],
        spec_keys: list[SpecKey],
    ):
        """
        A ``block_provider(a, b) -> Cov(C_tilde^a, C_tilde^b)`` callable for
        ``pseudo_to_spice_bmode``, closed over ``cov_key``'s own left and
        right frequencies: ``a`` is asked for at ``cov_key.left``'s
        frequencies, ``b`` at ``cov_key.right``'s. ``a``/``b`` equal
        ``cov_key.left``/``right`` except when that side is ``EE`` or ``BB``,
        in which case the other of the two is also asked for
        (:meth:`~cmbcov.assembly.postprocess.CovariancePostProcessor.sources`).
        Every block it can return was already computed in the raw-block loop
        of :meth:`compute_covariance_matrix` (``BB`` is present whenever
        ``EE`` is, in a ``polspice_postprocess: true`` run), so none is
        recomputed.
        """
        left, right = cov_key.left, cov_key.right

        def provider(a: str, b: str) -> np.ndarray:
            a_spec = SpecKey((a[0], a[1]), left.freq)
            b_spec = SpecKey((b[0], b[1]), right.freq)
            return self._pseudo_block(raw_blocks, spec_index, spec_keys, a_spec, b_spec)

        return provider

    @staticmethod
    def _pseudo_block(
        raw_blocks: dict[CovKey, np.ndarray],
        spec_index: dict[SpecKey, int],
        spec_keys: list[SpecKey],
        a: SpecKey,
        b: SpecKey,
    ) -> np.ndarray:
        """
        ``Cov(C_tilde^a, C_tilde^b)`` from the already-computed ``raw_blocks``
        of :meth:`compute_covariance_matrix`, which only holds the upper
        triangle (``ind(left) <= ind(right)`` in ``covariance_keys.spec_keys``):
        the lower triangle is the transpose of the stored block,
        ``Cov(b, a)^T``, since ``Cov(A, B)_{ij} = Cov(B, A)_{ji}``.
        """
        ia, ib = spec_index[a], spec_index[b]
        if ia <= ib:
            key = CovKey.from_spec_keys(spec_keys[ia], spec_keys[ib])
            return raw_blocks[key]
        key = CovKey.from_spec_keys(spec_keys[ib], spec_keys[ia])
        return raw_blocks[key].T

    def _folds_postprocess(self, bin_matrix: np.ndarray | None) -> bool:
        """
        Whether blocks are post-processed through leg projections.

        Without the PolSpice transform there is nothing to save (binning
        alone costs the same either way), and with (nearly) unbinned output
        the projections are as large as the blocks: both use the per-block
        path.
        """
        if self._fold_postprocess is not None:
            return self._fold_postprocess
        return (
            self.config.polspice_postprocess
            and bin_matrix is not None
            and bin_matrix.shape[0] * self._FOLD_MIN_ELL_PER_BIN <= self.lmax
        )

    def _leg_projection(
        self,
        projections: dict[tuple, np.ndarray],
        post_processor: CovariancePostProcessor,
        bin_matrix: np.ndarray | None,
        cov_key: CovKey,
        side: str,
        source: str | None,
    ) -> np.ndarray:
        r"""
        The matrix :math:`P = B\,\mathrm{diag}(\delta d)\,G` of one leg of
        ``cov_key`` (``side`` ``"left"`` or ``"right"``), built on first use
        and kept in ``projections`` for every other block with the same leg.

        A leg is fixed by its frequency pair, its output spectrum (which
        selects the debiasing factor) and the pseudo spectrum ``source`` its
        kernel :math:`G` acts on: ``None`` for the diagonal transform of a
        T/E-only run, the source of ``pseudo_to_spice_bmode`` for a B run.
        ``projections`` belongs to one ``compute_covariance_matrix`` call,
        so it never outlives the spectra and debiasing it was built with.
        """
        index = 0 if side == "left" else 1
        freq_pair = cov_key.freqkey().split("x")[index]
        stokes = cov_key.stokekey().split("x")[index]
        key = (freq_pair, stokes, source)
        projection = projections.get(key)
        if projection is None:
            if not self.config.polspice_postprocess:
                kernel = None
            elif source is None:
                kernel = post_processor.G_kernels[stokes]
            else:
                kernel = post_processor._bmode_kernel(stokes, source)
            scaling = post_processor.leg_scaling(
                freq_pair,
                stokes,
                self.lmax,
                self.config.Dl,
                self.config.debiasing_dict,
            )
            projection = post_processor.leg_projection(
                kernel, scaling, bin_matrix, self.lmax
            )
            projections[key] = projection
        return projection

    def _process_covariance_term(
        self,
        covariance_term: np.ndarray,
        cov_key: CovKey,
        post_processor: CovariancePostProcessor,
        bin_matrix: np.ndarray | None,
        projections: dict[tuple, np.ndarray],
    ) -> np.ndarray:
        r"""
        Process a covariance term (transform, scale, debias, bin) as
        :math:`P_L\,C\,P_R^{T}`, with :math:`P` the projection of each leg
        (:meth:`_leg_projection`). Each step is linear on both legs, so this
        is :meth:`_process_covariance_term_reference` to round-off, at a
        cost of ``n_bins lmax^2`` rather than ``lmax^3`` per block. With
        ``config.polspice_postprocess`` false the PolSpice transform is
        skipped; D_ell scaling, debiasing and binning are applied as usual.
        """
        return post_processor.project(
            self._leg_projection(
                projections, post_processor, bin_matrix, cov_key, "left", None
            ),
            covariance_term,
            self._leg_projection(
                projections, post_processor, bin_matrix, cov_key, "right", None
            ),
        )

    def _process_covariance_term_reference(
        self,
        covariance_term: np.ndarray,
        cov_key: CovKey,
        post_processor: CovariancePostProcessor,
        bin_matrix: np.ndarray | None,
    ) -> np.ndarray:
        """
        The per-block reference of :meth:`_process_covariance_term`: PolSpice
        transform, D_ell scaling, debiasing, then binning, each applied to
        the whole ``lmax x lmax`` block. Used when projections do not pay
        (:meth:`_folds_postprocess`) and as the test reference.
        """
        if self.config.polspice_postprocess:
            # Transform to PolSpice format (a new array: the raw block the
            # strategy returned, which may be cached, is not modified).
            processed_term = post_processor.pseudo_to_spice(
                covariance_term, cov_key.stokekey()
            )
        else:
            processed_term = np.array(covariance_term, copy=True)

        return self._finish_processing_reference(
            processed_term, cov_key, post_processor, bin_matrix
        )

    def _finish_processing_reference(
        self,
        processed_term: np.ndarray,
        cov_key: CovKey,
        post_processor: CovariancePostProcessor,
        bin_matrix: np.ndarray | None,
    ) -> np.ndarray:
        """D_ell scaling, debiasing and binning of the per-block reference path (T/E-only and B-mode)."""
        # Apply D_ell scaling if requested
        if self.config.Dl:
            processed_term = post_processor.apply_Dl_scaling(processed_term, self.lmax)

        # Apply corrections if provided
        if self.config.debiasing_dict is not None:
            processed_term = post_processor.apply_debiasing(
                processed_term,
                cov_key.freqkey(),
                cov_key.stokekey(),
                self.config.debiasing_dict,
                self.lmax,
            )

        # Bin the result
        if bin_matrix is not None:
            return bin_matrix @ processed_term @ bin_matrix.T
        else:
            return processed_term
