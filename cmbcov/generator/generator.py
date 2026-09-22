"""
Main covariance matrix generator class.

This module contains the high-level interface for computing TTTEEE covariance
matrices. It orchestrates all the steps: parameter loading, data preparation,
and covariance computation.
"""

import logging
import os
from collections.abc import Mapping
from typing import Any

import numpy as np

from cmbcov.bmode_wick import required_kernel_pairs
from cmbcov.keys import CovKeys
from cmbcov.utils import (
    add_nested_dicts,
)

from ..approximations.acc import coupling_ellprange, precompute_acc_kernels
from ..covariance import Cov, CovarianceMethod
from ..spectra import SpectraLoader, detect_parity_odd_nonzero
from ..windows import build_window_functions
from .parameter_validation import (
    EMPTY_PARAMS,
    AccPrecomputeConfig,
    ParameterManager,
    PipelineConfig,
)


def _short_freq_label(freq: str) -> str:
    """
    The owner's short frequency label for a window-function file name (e.g.
    ``"090GHz" -> "90"``): the ``GHz`` suffix and any leading zeros stripped.
    A label that does not fit that pattern (no ``GHz`` suffix, or a
    non-numeric prefix) is returned unchanged rather than raising -- the
    header comment inside the file still carries the run's own frequency
    name in full, so nothing is lost, only the file name stays verbose.
    """
    if freq.endswith("GHz") and freq[:-3].isdigit():
        return str(int(freq[:-3]))
    return freq


class CovarianceMatrixGenerator:
    """
    High-level interface for TTTEEE covariance matrix computation.

    This class orchestrates the complete workflow:
    1. Parameter validation and loading
    2. Data loading (beams, masks, spectra, etc.)
    3. Covariance matrix computation
    4. Results saving

    Example:
        >>> gen = CovarianceMatrixGenerator('parameters.yml')
        >>> gen.run_full_analysis()
    """

    def __init__(self, parameter_file: str, verbose: bool = False):
        """
        Initialize the covariance matrix generator.

        Parameters
        ----------
        parameter_file : str
            Path to the YAML parameter file
        verbose : bool, optional
            Enable verbose logging
        """
        self.parameter_file = parameter_file
        self.verbose_mode = verbose
        self.logger = self._setup_logging()

        # Initialize containers. The parameters live in a single validated
        # PipelineConfig, built by the ParameterManager; `params` below is a
        # read-only view of the same object, kept for callers that read
        # parameters by name.
        self.config: PipelineConfig | None = None
        self.param_manager: ParameterManager | None = None
        self.cov_keys: CovKeys | None = None
        self.combined_frequencies: list[str] = []
        self.combined_stokes: list[str] = []

        # The spectra and everything derived from them live in a SpectraLoader,
        # built by setup_frequency_stokes_mapping once the frequency and Stokes
        # layout is known. The dictionaries are exposed here as read-only
        # properties so existing callers keep working.
        self.spectra: SpectraLoader | None = None

        # Analysis object
        self.covariance_instance: Cov | None = None

        # The BB-only NKA/INKA leakage-warning report
        # (docs/theory/bmode_kernels.md), if this run's observables/method
        # combination triggers it; None otherwise. Filled in by
        # compute_covariance_matrix,
        # written beside the covariance by _save_bb_leakage_report.
        self._bb_leakage_report: dict[str, Any] | None = None

    @property
    def params(self) -> Mapping[str, Any]:
        """
        Read-only record of the parameter file.

        NOT the source of truth: :attr:`config` is. This is the effective
        parameter record -- what the file said, plus applied defaults, plus the
        resolved ``save_dir`` and ``git_hash`` -- and it is what gets written
        back beside the results. It cannot be written to, and nothing in the
        pipeline reads it, so it cannot drift away from the configuration.
        """
        return self.param_manager.params if self.param_manager else EMPTY_PARAMS

    @property
    def save_dir(self) -> str | None:
        """Versioned output directory, or None before parameters are loaded."""
        return self.config.save_dir if self.config is not None else None

    @property
    def lmax(self) -> int:
        """Get the maximum multipole from the run configuration."""
        return self.config.lmax if self.config is not None else 0

    # ----------------------------------------------------------------------------------
    # Setup and parameter loading methods
    # ----------------------------------------------------------------------------------

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO if self.verbose_mode else logging.WARNING,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        return logging.getLogger(__name__)

    def load_and_save_parameters(self, overwrite: int | None = None) -> None:
        """
        Load and validate parameters using ParameterManager.

        Parameters
        ----------
        overwrite : int, optional
            Existing output version to write into instead of creating a new
            one. An output-location choice only: nothing of the run that
            wrote that directory is reused, and the parameters already there
            must match in everything but the binning.
        """
        self.logger.info("Loading parameter file...")

        # Use ParameterManager for loading and validation
        self.param_manager = ParameterManager(
            self.parameter_file,
            parent_logger=self.logger,
            overwrite=overwrite,
        )
        self.config = self.param_manager.load_and_validate()
        self.param_manager.save_parameters()

        self.logger.info("Parameters loaded, validated, and saved successfully")

    def setup_frequency_stokes_mapping(self) -> None:
        """Setup frequency and Stokes parameter mappings."""
        frequencies = self.config.frequencies
        stokes = self.config.stokes
        observables = self.config.observables

        if any("B" in obs for obs in observables):
            # A B observable: build from the explicit observables list, not
            # the Stokes alphabet, so e.g. [TT, EE, TE, BB] does not also
            # create TB/EB. Not bit-identical with the alphabet path even for
            # the T/E-only sub-blocks of such a run: a run with a B
            # observable assembles every block per Wick term with the
            # per-term normalisation (docs/theory/bmode_kernels.md,
            # ACCStrategy.compute_covariance_term).
            self.cov_keys = CovKeys(
                [],
                frequencies,
                exclude_asymmetric_stokes=False,
                observables=observables,
                parity_mixed_blocks=self.config.parity_mixed_blocks,
            )
        else:
            # No B observable: the pre-existing alphabet path, unchanged, so
            # a default (or any T/E-only) run stays bit-identical.
            self.cov_keys = CovKeys(
                stokes, frequencies, exclude_asymmetric_stokes=False
            )
        self.combined_frequencies = self.cov_keys.combined_frequencies()
        self.combined_stokes = self.cov_keys.combined_stokes()

        self.logger.info(f"Combined stokes: {self.combined_stokes}")
        self.logger.info(f"Spectrum frequencies: {self.combined_frequencies}")

        # The data model can only be built once the layout is known.
        self.spectra = SpectraLoader(
            self.config,
            self.combined_frequencies,
            self.combined_stokes,
            logger=self.logger,
        )

    # ----------------------------------------------------------------------------------
    # Data loading -- delegated to SpectraLoader
    #
    # Everything from the parameter file to the biased spectra lives in
    # cmbcov.spectra. This class keeps only the workflow:
    # parameter versioning, the covariance object, and saving results.
    # ----------------------------------------------------------------------------------

    @property
    def beams(self):
        return self.spectra.beams

    @property
    def pix(self):
        return self.spectra.pix

    @property
    def fl_dict(self):
        return self.spectra.fl_dict

    @property
    def cl_dict(self):
        return self.spectra.cl_dict

    @property
    def nl_dict(self):
        return self.spectra.nl_dict

    @property
    def post_process_corrections(self):
        return self.spectra.post_process_corrections

    @property
    def data_model(self):
        return self.spectra.data_model

    @property
    def cl_dict_biased(self):
        return self.spectra.cl_dict_biased

    @property
    def nl_dict_biased(self):
        return self.spectra.nl_dict_biased

    @property
    def debiasing_dict(self):
        return self.spectra.debiasing_dict

    def load_pre_process(self) -> None:
        """Load beams, pixel window, transfer functions, signal and noise."""
        self.spectra.load_pre_process()

    def load_post_process(self) -> None:
        """Load the multiplicative post-processing corrections."""
        self.spectra.load_post_process()

    def prepare_workflow(self) -> tuple[dict, dict, dict, dict]:
        """Build the data model, the biased spectra and the debiasing factors."""
        return self.spectra.prepare_workflow()

    def setup_analysis_object(self) -> None:
        """Setup the covariance analysis object."""
        self.logger.info("Setting up analysis object...")
        # The CovarianceConfig was built once, by the validator, from the
        # parameter file; it is not rebuilt here from a second copy of the
        # values. Only the two runtime fields -- which are not parameter-file
        # values at all -- are filled in.
        config = self.config.covariance
        config.verbose = self.verbose_mode
        config.debiasing_dict = self.debiasing_dict

        # Initialize the Cov object. ACC kernels are read from the stable
        # acc_kernel_dir (cov_path), where precompute-acc writes them, not
        # from this run's versioned save_dir.
        self.covariance_instance = Cov(
            self.config.mask_name,
            config=config,
            mask_path=self.config.mask_path,
            save_dir=self.save_dir,
            acc_kernel_dir=self.config.acc_kernel_dir,
        )

    def precompute_acc_kernels(self, dryrun: bool = False) -> dict[str, Any]:
        """
        The one-off ACC step: compute the coupling kernels this parameter
        file's runs will load, and write them to
        :attr:`PipelineConfig.acc_kernel_dir`.

        Mask, ``centralell`` and ``dmax`` come from the file itself, the
        backend settings from its ``acc_precompute`` block (all defaults if
        the block is absent), so the kernels match what every later
        :meth:`run_full_analysis` of the file asks for. No versioned output
        directory is created.

        Parameters
        ----------
        dryrun : bool
            Validate and resolve the plan, but compute nothing.

        Returns
        -------
        dict
            The resolved plan: ``kernel_dir``, ``mask``, ``centralell``,
            ``dmax``, ``ellprange`` and the ``acc_precompute`` settings.

        Raises
        ------
        ValueError
            If the file fails validation, ``covariance_approximation`` is not
            ``acc``, or ``dmax``/``centralell`` are missing or invalid.
        """
        self.param_manager = ParameterManager(
            self.parameter_file, parent_logger=self.logger
        )
        self.config = self.param_manager.load_and_validate(setup_output=False)
        config = self.config
        if config.method != CovarianceMethod.ACC:
            raise ValueError(
                f"{self.parameter_file}: covariance_approximation is "
                f"{config.covariance_approximation!r}; the ACC precompute only "
                "applies to 'acc'"
            )
        # The checks Cov would apply at run time (dmax, centralell present).
        config.covariance.validate()

        settings = config.acc_precompute or AccPrecomputeConfig()

        # A B observable and no manual 'spectra' override: derive the
        # channel pairs from the observables list instead of the default
        # five-channel square. A manual 'spectra' is left untouched (an
        # explicit, advanced override). 'parity_odd_nonzero' is decided by
        # reading the raw TB/EB columns of the spectra files themselves
        # (detect_parity_odd_nonzero -- needs no lmax, so no circularity
        # with the kernel cache this precompute is about to create; shared
        # with SpectraLoader.parity_odd_nonzero, so a covariance run later
        # cannot disagree with what was precomputed here), over every
        # frequency pair this run uses. A run whose C^TB/C^EB nonetheless
        # turn out non-zero (a file changed after this precompute, say)
        # against an 18-pair cache is refused by the existing "recompute"
        # error, naming the missing pairs (SpectraLoader._resolve_spectra_lmax).
        pairs = None
        if settings.spectra is None and any("B" in obs for obs in config.observables):
            combined_frequencies = CovKeys(
                [],
                config.frequencies,
                observables=config.observables,
                parity_mixed_blocks=config.parity_mixed_blocks,
            ).combined_frequencies()
            parity_odd_nonzero = detect_parity_odd_nonzero(
                config.cmb_spectrum, config.foregrounds, combined_frequencies
            )
            pairs = sorted(
                required_kernel_pairs(
                    config.observables,
                    parity_odd_nonzero=parity_odd_nonzero,
                    parity_mixed_blocks=config.parity_mixed_blocks,
                )
            )

        plan = {
            "kernel_dir": config.acc_kernel_dir,
            "mask": os.path.join(config.mask_path or "./", config.mask_name),
            "centralell": config.centralell,
            "dmax": config.dmax,
            "ellprange": coupling_ellprange(config.centralell, config.dmax),
            "nside": settings.nside,
            "grid": settings.grid,
            "lw": settings.lw,
            "spectra": settings.spectra,
            "pairs": pairs,
            "max_memory_gb": settings.max_memory_gb,
        }
        self.logger.info(f"ACC precompute plan: {plan}")
        if dryrun:
            return plan

        precompute_acc_kernels(
            config.mask_name,
            config.acc_kernel_dir,
            centralell=config.centralell,
            dmax=config.dmax,
            mask_path=config.mask_path,
            nside=settings.nside,
            grid=settings.grid,
            lw=settings.lw,
            spectra=settings.spectra,
            pairs=pairs,
            max_memory_gb=settings.max_memory_gb,
            verbose=self.verbose_mode,
        )
        return plan

    def compute_covariance_matrix(
        self, dryrun: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """
        Compute the covariance matrix.

        Parameters
        ----------
        dryrun : bool
            If True, skip actual computation

        Returns
        -------
        tuple
            covariance_matrix or None if dryrun
        """
        if dryrun:
            self.logger.info("Dry run - skipping covariance computation")
            return None

        self.logger.info("Computing covariance matrix...")

        # Setup bins
        lbins = self._setup_bins()

        # input dict
        _ = self.prepare_workflow()

        # setup_analysis_object() ran before prepare_workflow(), so the config
        # captured self.debiasing_dict while it was still the empty dict from
        # __init__. prepare_workflow rebinds that attribute with a deepcopy
        # rather than mutating it, so the config still points at the stale
        # object. Refresh it before computing.
        self.covariance_instance.config.debiasing_dict = self.debiasing_dict

        # Compute main covariance matrix
        # missing="identity": a frequency pair absent from the noise (e.g.
        # `nl: {}`) means no noise for it, not a bug -- keep it that way.
        cl_used = add_nested_dicts(
            self.cl_dict_biased, self.nl_dict_biased, missing="identity"
        )
        final_cov = self.covariance_instance.compute_covariance_matrix(
            lbins, self.cov_keys, cl_used
        )

        self._warn_bb_only_leakage(cl_used)

        # Compute additional matrices if requested
        if self.config.compute_noise_only:
            raise NotImplementedError(
                "Noise-only covariance computation not implemented yet"
            )

        if self.config.compute_signal_only:
            raise NotImplementedError(
                "Signal-only covariance computation not implemented yet"
            )

        return final_cov

    def _warn_bb_only_leakage(self, cl_used: dict[str, dict[str, np.ndarray]]) -> None:
        """
        Fire the BB-only NKA/INKA leakage warning (docs/theory/bmode_kernels.md),
        if this run is one: ``observables: [BB]``
        alone with ``covariance_approximation: nka`` or ``inka``
        (``generator.parameter_validation.APPROXIMATIONS_SUPPORTING_B``).

        Reads ``C^BB`` from ``cl_used`` (the same spectrum the covariance
        was just computed from -- the first frequency pair, representative
        of every other one on the same mask) and ``C^EE`` from the
        ``cmb_spectrum`` source via
        :meth:`~cmbcov.spectra.SpectraLoader.read_extra_stokes_spectrum`,
        which this run never otherwise loads. Sets :attr:`_bb_leakage_report`
        for :meth:`_save_bb_leakage_report`; leaves it ``None`` for every
        other run.
        """
        self._bb_leakage_report = None
        if self.config.observables != ["BB"] or self.config.method not in (
            CovarianceMethod.NKA,
            CovarianceMethod.INKA,
        ):
            return
        freq_key = next(iter(cl_used))
        cl_bb = cl_used[freq_key]["BB"]
        cl_ee = self.spectra.read_extra_stokes_spectrum("EE")
        self._bb_leakage_report = self.covariance_instance.warn_bb_only_leakage(
            cl_bb, cl_ee, self.config.method, logger=self.logger
        )

    def _setup_bins(self) -> list[int]:
        """Setup multipole bins from parameter file."""
        lbins = []
        for bin_spec in self.config.bins:
            lbins.extend(range(bin_spec[0], bin_spec[1], bin_spec[2]))
        lbins.append(self.config.bins[-1][1])  # Add final bin edge

        self.logger.info(f"Bins: {lbins}")
        return lbins

    def save_results(
        self,
        base_dir: str,
        ls: np.ndarray | None,
        final_cov: np.ndarray | None,
        save_windows: bool = False,
    ) -> None:
        """
        Save computation results to disk.

        Parameters
        ----------
        base_dir : str
            Base directory for saving
        ls : np.ndarray or None
            Multipole centers
        final_cov : np.ndarray or None
            Covariance matrix
        save_windows : bool
            Whether to save window functions
        """
        if ls is None or final_cov is None:
            self.logger.info("No results to save (dry run)")
            return

        self.logger.info("Saving results...")

        # Save main results
        path_to_lbins = os.path.join(base_dir, "lbins.dat")
        path_to_final_cov = os.path.join(base_dir, self.config.cov_name)

        np.savetxt(path_to_lbins, ls)
        np.savetxt(path_to_final_cov, final_cov)

        self._save_error_budget(base_dir)
        self._save_bb_leakage_report(base_dir)

        # Save window functions if requested
        if save_windows:
            self._save_window_functions(base_dir, ls)

        self.logger.info("Results saved successfully")

    #: Name of the error-budget report written beside the covariance.
    ERROR_BUDGET_NAME = "error_budget.txt"

    #: Name of the BB-only NKA/INKA leakage-warning report written beside
    #: the covariance (docs/theory/bmode_kernels.md).
    BB_LEAKAGE_WARNING_NAME = "bb_leakage_warning.txt"

    def _save_bb_leakage_report(self, base_dir: str) -> None:
        """
        Write :attr:`_bb_leakage_report` (set by
        :meth:`_warn_bb_only_leakage`) beside the covariance, as the run's
        record of the warning it already logged -- nothing is written for a
        run that is not the BB-only NKA/INKA escape hatch.
        """
        report = self._bb_leakage_report
        if report is None:
            return
        path = os.path.join(base_dir, self.BB_LEAKAGE_WARNING_NAME)
        with open(path, "w") as handle:
            handle.write(
                "BB-only NKA/INKA leakage warning "
                "(docs/theory/bmode_kernels.md)\n"
                f"covariance_approximation: {self.config.covariance_approximation}\n"
                f"ell_safe: {report['ell_safe']}\n"
                f"lmax: {report['lmax']}\n"
                f"reliable: {report['reliable']}\n"
                f"lowest_bin_factor: {report['lowest_bin_factor']:.6g}\n"
                "\n"
                "Leakage from C^EE into the pseudo C^BB mean is neglected "
                "throughout. Below ell_safe, the missed variance inflation "
                "(1 + leak/signal)^2 exceeds 5% (the lowest bin is "
                "underestimated by the factor above); reliable=False means "
                "it exceeds 5% at every multipole up to lmax.\n"
            )

    def _save_error_budget(self, base_dir: str) -> None:
        """
        Write the method's known-error report beside the covariance, and log
        its headline number.

        Only ACC has one, and only when the run has a polarised leg
        (:mod:`~cmbcov.approximations.acc_budget`); nothing
        is written otherwise. This is a report: it reads the kernels already
        in the cache, computes nothing new, and cannot change the covariance
        that was just saved -- so a failure here is logged, not raised.
        """
        if self.covariance_instance is None or self.cov_keys is None:
            return
        from ..approximations import acc_budget

        # The extreme bandpower edges: |l - l*| is largest at one of them, so
        # the intermediate edges cannot change the reported maximum.
        bins = self.config.bins
        edges = [bins[0][0], bins[-1][1]] if bins else None
        try:
            budget = self.covariance_instance.error_budget(
                self.cov_keys, band_edges=edges
            )
        except Exception as error:  # noqa: BLE001 - diagnostics must not fail a run
            self.logger.warning(f"Could not build the ACC error budget: {error}")
            return
        if budget is None:
            return
        path = os.path.join(base_dir, self.ERROR_BUDGET_NAME)
        with open(path, "w") as handle:
            handle.write(acc_budget.format_error_budget(budget))
        leakage = budget["leakage"]["lambda"]
        if leakage is None:
            self.logger.warning(
                "ACC error budget: the E->B leakage could not be bounded "
                f"({budget['leakage']['unavailable']}). Report: {path}"
            )
        else:
            worst = max(entry["leakage_bias"] for entry in budget["blocks"].values())
            self.logger.info(
                f"ACC error budget: E->B leakage lambda = {leakage:.3e}, "
                f"largest polarised block bias {worst:+.3e}; the Eq. 33 "
                "translation error is NOT bounded (max |l - l*| = "
                f"{budget['translation']['max_offset_from_centralell']}). "
                f"Report: {path}"
            )

    #: Subdirectory, beside the covariance, holding the bandpower
    #: window-function text files (:mod:`cmbcov.windows`):
    #: one plain-text file per two-letter spectrum and frequency pair, the
    #: owner's existing convention (``<X>_<f1>x<f2>_window_functions.txt``).
    WINDOW_FUNCTIONS_DIRNAME = "windows"

    def _save_window_functions(self, base_dir: str, ls: np.ndarray) -> None:
        r"""
        Save the bandpower window functions beside the covariance, one
        plain-text file per two-letter spectrum and frequency pair this run
        reports (the owner's existing convention), into
        :attr:`WINDOW_FUNCTIONS_DIRNAME`.

        Each file is the linear map from the fiducial theory spectrum to
        that pair's expected binned output spectrum
        (:func:`~cmbcov.windows.build_window_functions`),
        including the per-pair beam, pixel window, transfer function and
        calibration debiasing this run's own ``debiasing_dict`` applies to
        the covariance
        (:meth:`~cmbcov.postprocess.CovariancePostProcessor.apply_debiasing`).
        The spectra and frequency pairs, and their order, come from
        :attr:`cov_keys`'s own ``spec_keys`` -- the same list the covariance
        itself is built from.

        Raises
        ------
        ValueError
            If ``polspice_postprocess`` is false and both ``EE`` and ``BB``
            are among the observables: the reported ``EE`` (``BB``)
            bandpower then genuinely mixes in :math:`C^{BB}`
            (:math:`C^{EE}`) (docs/theory/bmode_kernels.md Sect. 7), a
            mixing this one-spectrum-per-file text format cannot represent.
            Enable ``polspice_postprocess`` or drop ``--save-windows`` for
            that run instead.
        """
        self.logger.info("Saving window functions...")

        if self.covariance_instance is None or self.cov_keys is None:
            return

        # polspice_postprocess lives on the covariance sub-config
        # (`config.covariance`), not exposed as a `PipelineConfig` property
        # the way `Dl` is.
        polspice_postprocess = self.config.covariance.polspice_postprocess

        combined_stokes = self.cov_keys.combined_stokes()
        if (
            not polspice_postprocess
            and "EE" in combined_stokes
            and "BB" in combined_stokes
        ):
            raise ValueError(
                "Cannot write per-spectrum window-function text files for "
                "this run: polspice_postprocess is false and both EE and BB "
                "are among the observables, so the reported EE bandpower "
                "mixes in C^BB, and the reported BB bandpower mixes in "
                "C^EE (docs/theory/bmode_kernels.md, Sect. 7) -- a mixing "
                "this one-spectrum-per-file text format cannot represent. "
                "Enable polspice_postprocess, or drop --save-windows, for "
                "this run."
            )

        lbins = self._setup_bins()
        lmin = self.config.lmin
        debiasing_dict = self.debiasing_dict or {}

        windows_dir = os.path.join(base_dir, self.WINDOW_FUNCTIONS_DIRNAME)
        os.makedirs(windows_dir, exist_ok=True)

        for spec in self.cov_keys.spec_keys:
            stoke_key = spec.stokekey()
            freq_key = spec.freqkey()

            leg_factor = debiasing_dict.get(freq_key, {}).get(stoke_key)
            row_scale = {stoke_key: leg_factor} if leg_factor is not None else None

            windows = build_window_functions(
                [stoke_key],
                self.covariance_instance,
                lbins,
                Dl=self.config.Dl,
                polspice_postprocess=polspice_postprocess,
                row_scale=row_scale,
            )
            ell = windows["ell"]
            w = windows[f"W_{stoke_key}"]

            keep = ell >= lmin
            rows = np.column_stack([ell[keep], w[:, keep].T])

            file_freq = "x".join(_short_freq_label(f) for f in spec.freq)
            filename = f"{stoke_key}_{file_freq}_window_functions.txt"
            header = (
                f"Band powers window functions for {stoke_key} {freq_key}.\n"
                "<C_hat_b> = sum_l W[b, l] * C_l, with C_l the fiducial "
                "theory spectrum (cmb_spectrum); includes the beam, pixel "
                "window, transfer function and calibration debiasing this "
                "run applies to the covariance "
                "(SpectraLoader.data_model/debiasing_dict)."
            )
            np.savetxt(os.path.join(windows_dir, filename), rows, header=header)

        self.logger.info(f"Window functions saved to {windows_dir}")

    def run_full_analysis(
        self,
        save: bool = True,
        save_corrections: bool = False,
        save_windows: bool = False,
        dryrun: bool = False,
        overwrite: int | None = None,
    ) -> None:
        """
        Run the complete covariance matrix analysis.

        Parameters
        ----------
        overwrite : int, optional
            Version number to overwrite
        save : bool
            Whether to save results
        save_corrections : bool
            Whether to save correction factors
        save_windows : bool
            Whether to save window functions
        dryrun : bool
            Whether to do a dry run
        overwrite : int, optional
            Existing output version to overwrite instead of creating a new one.
        """
        try:
            # Setup and validation
            self.load_and_save_parameters(overwrite=overwrite)
            self.setup_frequency_stokes_mapping()

            # Load all data
            self.load_pre_process()
            self.load_post_process()

            # Prepare for computation
            self.setup_analysis_object()

            # Compute covariance matrix
            final_cov = self.compute_covariance_matrix(dryrun=dryrun)

            # Save results. Cov.compute_covariance_matrix returns the triple
            # (binned_ells, bin_matrix, covariance); save_results takes the
            # multipole centres and the covariance separately.
            if save and not dryrun:
                if final_cov is None:
                    self.logger.warning("No covariance produced; nothing to save")
                else:
                    binned_ells, _bin_matrix, covariance = final_cov
                    self.save_results(
                        self.config.save_dir,
                        binned_ells,
                        covariance,
                        save_windows=save_windows,
                    )

            self.logger.info("Analysis completed successfully")

        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            raise
