"""
Parameter-file driver of the covariance computation.

``CovarianceMatrixGenerator`` loads and validates the parameter file, prepares
the spectra, runs :class:`~cmbcov.assembly.covariance.Cov` and saves the results.
"""

import functools
import logging
import os
from collections.abc import Mapping
from typing import Any

import numpy as np

from ..approximations.acc.bmode_wick import required_kernel_pairs
from ..approximations.acc.precompute import coupling_ellprange, precompute_acc_kernels
from ..assembly.conditioning import conditioning_report, format_conditioning_report
from ..assembly.covariance import Cov, CovarianceMethod
from ..assembly.keys import CovKeys
from ..assembly.windows import build_window_functions
from ..spectra import SpectraLoader, detect_parity_odd_nonzero
from ..utils import (
    add_nested_dicts,
    safe_divide,
)
from .validation import (
    EMPTY_PARAMS,
    AccPrecomputeConfig,
    ParameterManager,
    PipelineConfig,
)

logger = logging.getLogger(__name__)


def _short_freq_label(freq: str) -> str:
    """
    Short frequency label for a window-function file name (e.g.
    ``"090GHz" -> "90"``): the ``GHz`` suffix and any leading zeros stripped.
    A label that does not fit that pattern is returned unchanged (the file
    header carries the full frequency name).
    """
    if freq.endswith("GHz") and freq[:-3].isdigit():
        return str(int(freq[:-3]))
    return freq


class CovarianceMatrixGenerator:
    """
    High-level interface for the covariance matrix computation.

    The workflow is:

    1. parameter validation and loading,
    2. data loading (beams, masks, spectra, ...),
    3. covariance matrix computation,
    4. saving the results.

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
        self.logger = logger

        # The parameters live in a single validated PipelineConfig, built by
        # the ParameterManager; `params` below is a read-only record of them.
        self.config: PipelineConfig | None = None
        self.param_manager: ParameterManager | None = None
        self.cov_keys: CovKeys | None = None
        self.combined_frequencies: list[str] = []
        self.combined_stokes: list[str] = []

        # The spectra and everything derived from them live in a SpectraLoader,
        # built by setup_frequency_stokes_mapping once the frequency and Stokes
        # layout is known. Its dictionaries are exposed here as properties.
        self.spectra: SpectraLoader | None = None

        # Analysis object
        self.covariance_instance: Cov | None = None

        # The BB-only NKA/INKA leakage-warning report
        # (docs/theory/bmode_kernels.md) of this run, or None. Filled in by
        # compute_covariance_matrix, written by _save_bb_leakage_report.
        self._bb_leakage_report: dict[str, Any] | None = None

        # The filter-and-bin sum-rule ratio (``nodes``, ``g``) of a run with a
        # ``map_filter`` block, filled in by setup_map_filter, written beside
        # the covariance by _save_map_filter; None otherwise.
        self._map_filter_ratio: dict[str, np.ndarray] | None = None

    @property
    def params(self) -> Mapping[str, Any]:
        """
        Read-only record of the parameter file.

        Not the source of truth: :attr:`config` is. This is the effective
        parameter record (the file's values, applied defaults, the resolved
        ``save_dir`` and ``git_hash``) that is written beside the results.
        It is read-only and nothing in the pipeline reads it.
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

    def load_and_save_parameters(self, overwrite: int | None = None) -> None:
        """
        Load and validate parameters using ParameterManager.

        Parameters
        ----------
        overwrite : int, optional
            Existing output version to write into instead of creating a new
            one. Nothing of the run that wrote that directory is reused, and
            the parameters already there must match in everything but the
            binning.
        """
        self.logger.info("Loading parameter file...")

        # Use ParameterManager for loading and validation
        self.param_manager = ParameterManager(
            self.parameter_file,
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
            # create TB/EB. Every block of such a run, T/E blocks included,
            # is assembled per Wick term (docs/theory/bmode_kernels.md,
            # ACCStrategy.compute_covariance_term).
            self.cov_keys = CovKeys(
                [],
                frequencies,
                exclude_asymmetric_stokes=False,
                observables=observables,
                parity_mixed_blocks=self.config.parity_mixed_blocks,
            )
        else:
            # No B observable: the Stokes alphabet path.
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
    # cmbcov.spectra. This class keeps the workflow: parameter versioning,
    # the covariance object, and saving results.
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
        # The CovarianceConfig is built once, by the validator; only the two
        # runtime fields, which are not parameter-file values, are set here.
        config = self.config.covariance
        config.verbose = self.verbose_mode
        config.debiasing_dict = self.debiasing_dict

        # Initialize the Cov object. ACC kernels are read from the stable
        # acc_kernel_dir (cov_path), where cmbcov-precompute writes them, not
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
        self.param_manager = ParameterManager(self.parameter_file)
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

        # A B observable and no manual 'spectra' override: derive the channel
        # pairs from the observables list instead of the default five-channel
        # square (a manual 'spectra' is left as given). 'parity_odd_nonzero'
        # is decided from the raw TB/EB columns of the spectra files over
        # every frequency pair of the run (detect_parity_odd_nonzero needs no
        # lmax, and is shared with SpectraLoader.parity_odd_nonzero so a later
        # covariance run agrees). If C^TB/C^EB turn out non-zero against a
        # cache without those pairs, the run is refused with a "recompute"
        # error naming the missing pairs (SpectraLoader._resolve_spectra_lmax).
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
            "scratch_dir": settings.scratch_dir,
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
            scratch_dir=settings.scratch_dir,
            verbose=self.verbose_mode,
        )
        return plan

    def _prepare_spectra(self) -> tuple[list[int], dict[str, dict[str, np.ndarray]]]:
        """
        The multipole bins and the spectra (signal plus noise, as the estimator
        sees them) the covariance is computed with; needs the spectra loaded
        (:meth:`load_pre_process`, :meth:`load_post_process`) and
        :attr:`covariance_instance` set.
        """
        # Setup bins
        lbins = self._setup_bins()

        # input dict
        _ = self.prepare_workflow()

        # setup_analysis_object() ran before prepare_workflow(), which rebinds
        # debiasing_dict, so the config holds the stale object: refresh it.
        self.covariance_instance.config.debiasing_dict = self.debiasing_dict

        # missing="identity": a frequency pair absent from the noise (e.g.
        # `nl: {}`) has no noise.
        cl_used = add_nested_dicts(
            self.cl_dict_biased, self.nl_dict_biased, missing="identity"
        )
        return lbins, cl_used

    def compute_covariance_matrix(
        self, dryrun: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """
        Compute the covariance matrix.

        Parameters
        ----------
        dryrun : bool
            If True, skip the computation.

        Returns
        -------
        tuple or None
            ``(binned_ells, binning_matrix, covariance_matrix)`` as returned
            by :meth:`~cmbcov.assembly.covariance.Cov.compute_covariance_matrix`, or
            ``None`` if ``dryrun``.
        """
        if dryrun:
            self.logger.info("Dry run - skipping covariance computation")
            return None

        self.logger.info("Computing covariance matrix...")

        lbins, cl_used = self._prepare_spectra()
        self._warn_low_transfer(lbins)
        # The spectrum-aware map_filter probes need the spectra the
        # covariance is computed with.
        self.setup_map_filter(cl_used)

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

    #: Name of the filter sum-rule ratio written beside the covariance.
    MAP_FILTER_NAME = "map_filter_ratio.npz"

    def setup_map_filter(
        self, cl_used: dict[str, dict[str, np.ndarray]] | None = None
    ) -> None:
        """
        Build the filter-and-bin correction of a ``map_filter`` run and hand
        it to the covariance object (``config.filter_rho``,
        ``config.filter_rho_blocks`` and ``config.filter_fl``); does nothing
        without the block.

        ``rho`` is the filtered sum-rule ratio of the mask, one per sum-rule
        channel the run's T/E blocks need
        (:func:`cmbcov.assembly.filtering.channels_for`;
        :func:`cmbcov.assembly.filtering.filtered_sum_rule_ratio`, probed at the nodes
        of :func:`~cmbcov.assembly.filtering.node_plan` -- geometric plus dense near
        the cut, with their per-node probe counts and band-limit floors --
        and interpolated by :func:`~cmbcov.assembly.filtering.rho_matrix`). The probes
        are the expensive part, so the ratios are cached in the kernel
        directory under the mask digest, the channels, the nodes with their
        per-node settings and every parameter they depend on
        (:func:`~cmbcov.assembly.filtering.cached_sum_rule_ratio`). The transfer
        functions are ``fl_dict[pair]`` (every Stokes pair the run loads;
        ``ET`` falls back to ``TE`` in the covariance): all ones without
        ``fl`` (the factor is then ``rho`` alone, with a warning). Blocks
        with a B leg are left uncorrected, with a warning.

        With ``spectrum_aware: true`` every TT x TT block is probed with the
        spectra instead (:meth:`_spectrum_aware_inputs`,
        :func:`~cmbcov.assembly.filtering.spectrum_aware_ratio`; cached under the
        same identity plus the Wick terms and a digest of the spectra), and
        the white ``00`` channel is not probed.

        Parameters
        ----------
        cl_used : dict, optional
            The spectra the covariance is computed with,
            ``cl_used[freq_pair][stokes]`` (biased signal plus noise); built
            from ``cl_dict_biased`` and ``nl_dict_biased`` as in
            :meth:`compute_covariance_matrix` when not given. Only read with
            ``spectrum_aware``.
        """
        settings = self.config.map_filter
        self._map_filter_ratio = None
        if settings is None:
            return
        from ..assembly import filtering
        from ..exact.gl import _mask_alm_for_gl

        cov = self.covariance_instance
        lmax = self.config.lmax
        stokes_pairs = self.cov_keys.stokekey()
        spectrum_aware = bool(settings.spectrum_aware) and "TTxTT" in stokes_pairs
        channels = filtering.channels_for(
            [s for s in stokes_pairs if not (spectrum_aware and s == "TTxTT")]
        )
        uncorrected = [
            s
            for s in stokes_pairs
            if tuple(s.split("x")) not in filtering.BLOCK_CHANNEL
        ]
        if uncorrected:
            self.logger.warning(
                "map_filter: the blocks with a B leg (%s) are left uncorrected "
                "(docs/theory/filter_and_bin.md)",
                ", ".join(uncorrected),
            )
        if not channels and not spectrum_aware:
            self.logger.warning(
                "map_filter is set but the run has no T/E block: nothing to correct"
            )
            return
        if self.config.fl is None:
            self.logger.warning(
                "map_filter is set without fl: the correction is rho alone "
                "(transfer function taken as 1)"
            )

        self._warn_lowpass(settings, lmax, filtering)

        lw = settings.resolved_lw(cov.wlm.nside, lmax)
        dmax = settings.resolved_dmax(lmax, self.config.method, self.config.dmax or 0)
        # The low-pass arguments only when there is one: the call, hence the
        # plan, of a run without it is what it was before they existed.
        low_args = (
            {}
            if settings.lowpass is None
            else {
                "lx_lowpass": settings.lowpass.lx,
                "lowpass_shape": settings.lowpass.shape,
            }
        )
        plan = filtering.node_plan(
            lmax,
            settings.node_min,
            settings.node_ratio,
            settings.lx,
            near_cut_step=settings.near_cut_step,
            nprobe=settings.nprobe,
            nprobe_near_cut=settings.nprobe_near_cut,
            **low_args,
        )
        nodes = [int(n) for n in plan.nodes]
        nprobe = [int(n) for n in plan.nprobe]
        floors = [int(n) for n in plan.lmax_int_floor]
        if settings.lowpass is None:
            profile = functools.partial(
                filtering.highpass_profile,
                lx=settings.lx,
                shape=settings.shape,
                power=settings.power,
            )
        else:
            profile = functools.partial(
                filtering.combined_profile,
                lx=settings.lx,
                shape=settings.shape,
                power=settings.power,
                lowpass=settings.lowpass.as_dict(),
            )
        identity = {
            "mask_digest": cov.wlm.mask_digest,
            "type": settings.type,
            "lx": settings.lx,
            "shape": settings.shape,
            "power": settings.power,
            "channels": list(channels),
            "nodes": nodes,
            "dmax": dmax,
            "nprobe": nprobe,
            "lmax_int_floor": floors,
            "seed": settings.seed,
            "lw": lw,
            "lmax": lmax,
        }
        if settings.lowpass is not None:
            # Only when present: the identity (hence the cache file name) of a
            # run without a low-pass is what it was before the key existed.
            identity["lowpass"] = settings.lowpass.as_dict()
        mask_alm = []

        def get_mask_alm() -> np.ndarray:
            if not mask_alm:
                mask_alm.append(_mask_alm_for_gl(cov.wlm.mask, cov.wlm.nside, lw)[0])
            return mask_alm[0]

        self._map_filter_ratio = {
            "nodes": np.asarray(nodes),
            "nprobe": np.asarray(nprobe),
            "lmax_int_floor": np.asarray(floors),
            "channels": np.array(list(channels), dtype=str),
        }
        cov.config.filter_rho = None
        if channels:
            computed = []

            def compute() -> dict[str, np.ndarray]:
                computed.append(True)
                return filtering.filtered_sum_rule_ratio(
                    get_mask_alm(),
                    lw,
                    profile,
                    nodes,
                    dmax,
                    channels=channels,
                    nprobe=nprobe,
                    seed=settings.seed,
                    lmax_int_floor=floors,
                    verbose=self.verbose_mode,
                )

            self.logger.info(
                f"map_filter: channels {', '.join(channels)}, {len(nodes)} nodes "
                f"({nodes[0]}..{nodes[-1]}), dmax={dmax}, lw={lw}, "
                f"{self._probe_summary(nodes, nprobe, floors)}"
            )
            g = filtering.cached_sum_rule_ratio(
                self.config.acc_kernel_dir, identity, compute
            )
            self.logger.info(
                "map_filter: sum-rule ratio "
                + ("computed" if computed else "read from the cache")
            )
            self._map_filter_ratio.update(
                {f"g_{ch}": np.asarray(g[ch]) for ch in channels}
            )
            cov.config.filter_rho = {
                ch: filtering.rho_matrix(nodes, g[ch], lmax) for ch in channels
            }

        cov.config.filter_rho_blocks = None
        if spectrum_aware:
            terms, cl_pre, cl_tt = self._spectrum_aware_inputs(cl_used)
            sa_identity = {
                **identity,
                "channels": [],
                "kind": "spectrum_aware_TT",
                "terms": {b: [list(t) for t in wick] for b, wick in terms.items()},
                "spectra_digest": Cov._spectra_digest(
                    {
                        **{f"pre_{p}": v for p, v in cl_pre.items()},
                        **{f"used_{p}": v for p, v in cl_tt.items()},
                    }
                ),
                "noise_fl_floor": filtering.NOISE_FL_FLOOR,
                "oversample": filtering.DEFAULT_OVERSAMPLE,
            }
            computed = []

            def compute_blocks() -> dict[str, np.ndarray]:
                computed.append(True)
                return filtering.spectrum_aware_ratio(
                    get_mask_alm(),
                    lw,
                    profile,
                    nodes,
                    dmax,
                    cl_pre,
                    cl_tt,
                    terms,
                    nprobe=nprobe,
                    seed=settings.seed,
                    lmax_int_floor=floors,
                    oversample=filtering.DEFAULT_OVERSAMPLE,
                    verbose=self.verbose_mode,
                )

            self.logger.info(
                f"map_filter: spectrum-aware TT x TT, {len(terms)} blocks, "
                f"{len(cl_pre)} frequency pairs, {len(nodes)} nodes, dmax={dmax}, "
                f"lw={lw}, {self._probe_summary(nodes, nprobe, floors)}"
            )
            g_blocks = filtering.cached_sum_rule_ratio(
                self.config.acc_kernel_dir, sa_identity, compute_blocks
            )
            self.logger.info(
                "map_filter: spectrum-aware ratio "
                + ("computed" if computed else "read from the cache")
            )
            self._map_filter_ratio["spectrum_aware_blocks"] = np.array(
                list(terms), dtype=str
            )
            for i, block in enumerate(terms):
                self._map_filter_ratio[f"g_block_{i}"] = np.asarray(g_blocks[block])
            cov.config.filter_rho_blocks = {
                block: (list(nodes), np.asarray(g_blocks[block])) for block in terms
            }

        cov.config.filter_fl = {
            pair: dict(self.fl_dict[pair]) for pair in self.combined_frequencies
        }

    def _warn_lowpass(self, settings, lmax: int, filtering) -> None:
        """
        Log (once per run, from :meth:`setup_map_filter`) the two limits of a
        ``map_filter.lowpass``: the sharp shape with its cut inside the
        covariance (``lx <= lmax``), whose ratio has off-diagonal structure at
        the kink that the (mean, distance) interpolation cannot carry; and
        ``near_cut_step: 0`` with the rolloff inside the covariance
        (``0.6 lx < lmax``), which leaves the rolloff to the geometric nodes.
        """
        low = settings.lowpass
        if low is None:
            return
        if low.shape == "sharp" and low.lx <= lmax:
            self.logger.warning(
                "map_filter.lowpass: the sharp low-pass cut (lx=%s) is inside "
                "the covariance (lmax=%s). The ratio has off-diagonal structure "
                "at the kink that the (mean, distance) interpolation cannot "
                "carry: blocks within a few multipoles of lx are about 3-4%% off "
                "even with the fine nodes (docs/theory/filter_and_bin.md, 'The "
                "low-pass'). The exp shape has no such limit.",
                low.lx,
                lmax,
            )
        start = filtering.LOWPASS_RANGE[0] * float(low.lx)
        if (
            settings.near_cut_step is not None
            and float(settings.near_cut_step) == 0
            and start < lmax
        ):
            self.logger.warning(
                "map_filter: near_cut_step is 0 and the low-pass rolloff "
                "(from %.0f, 0.6 lx) starts below lmax=%s: the rolloff ratio is "
                "interpolated from the geometric nodes alone (up to about 4%% "
                "off for exp, more for sharp, at toy scale)",
                start,
                lmax,
            )

    @staticmethod
    def _probe_summary(nodes: list[int], nprobe: list[int], floors: list[int]) -> str:
        """One log fragment for the per-node probe settings of setup_map_filter."""
        exact = [n for n, k in zip(nodes, nprobe) if 2 * n + 1 <= k]
        random = sorted({k for n, k in zip(nodes, nprobe) if 2 * n + 1 > k})
        floored = [n for n, f in zip(nodes, floors) if f > 0]
        parts = [f"nprobe={','.join(str(k) for k in random) or '-'}"]
        if exact:
            parts.append(
                f"complete basis at {len(exact)} nodes ({exact[0]}..{exact[-1]})"
            )
        if floored:
            parts.append(
                f"band-limit floor {max(floors)} at {len(floored)} nodes "
                f"({floored[0]}..{floored[-1]})"
            )
        return ", ".join(parts)

    def _warn_low_transfer(self, lbins: list[int]) -> None:
        """
        With ``map_filter``: one log warning naming the bins whose lowest
        transfer function (any T/E spectrum the run uses, any frequency pair;
        multipoles from ``lmin`` to ``lmax - 1``) is below
        :data:`cmbcov.assembly.filtering.FL_DEGRADED` ("accuracy degraded"), and a
        stronger one for those below :data:`cmbcov.assembly.filtering.FL_CUT` ("cut
        these bins"). The numbers are not changed.
        """
        if self.config.map_filter is None:
            return
        from ..assembly import filtering

        spectra = []
        for stokekey in self.cov_keys.stokekey():
            for s in stokekey.split("x"):
                s = "TE" if s == "ET" else s
                if s in ("TT", "EE", "TE") and s not in spectra:
                    spectra.append(s)
        degraded, cut = [], []
        for pair in self.combined_frequencies:
            for s in spectra:
                fl = (self.fl_dict.get(pair) or {}).get(s)
                if fl is None:
                    continue
                minima = filtering.bin_transfer_minimum(
                    lbins, fl, self.config.lmin, self.config.lmax
                )
                for threshold, found in (
                    (filtering.FL_DEGRADED, degraded),
                    (filtering.FL_CUT, cut),
                ):
                    low = [f"{a}-{b} ({v:.3f})" for a, b, v in minima if v < threshold]
                    if low:
                        found.append(f"{s} {pair}: {', '.join(low)}")
        if degraded:
            self.logger.warning(
                "map_filter: the transfer function falls below %s inside these "
                "bins, so the filter correction's accuracy is degraded there "
                "(lowest fl in the bin; docs/theory/filter_and_bin.md): %s",
                filtering.FL_DEGRADED,
                "; ".join(degraded),
            )
        if cut:
            self.logger.warning(
                "map_filter: the transfer function falls below %s inside these "
                "bins; their covariance is not reliable: cut these bins (lowest "
                "fl in the bin): %s",
                filtering.FL_CUT,
                "; ".join(cut),
            )

    def _spectrum_aware_inputs(
        self, cl_used: dict[str, dict[str, np.ndarray]] | None
    ) -> tuple[
        dict[str, list[tuple[str, str]]], dict[str, np.ndarray], dict[str, np.ndarray]
    ]:
        """
        The Wick terms of every TT x TT block of the run and the two TT
        spectrum sets of :func:`cmbcov.assembly.filtering.spectrum_aware_ratio`:
        ``(terms, cl_pre, cl_used_tt)``, keyed by the frequency pairs the
        terms name.

        * ``terms[block_id] = [(ac, bd), (ad, bc)]`` from
          :meth:`~cmbcov.assembly.keys.CovKey.key_to_cross`, the pairs the strategies
          read the spectra with.
        * ``cl_pre[pair]``: the sky before the filter, ``cl_dict`` times the
          data model without ``fl`` (``B_1 B_2 pixwin``, from the loader's
          beams and pixel window) plus
          :func:`~cmbcov.assembly.filtering.pre_filter_noise` of the noise
          (``nl / fl`` where ``fl`` exceeds
          :data:`~cmbcov.assembly.filtering.NOISE_FL_FLOOR`, ``nl`` elsewhere; the
          noise transfer function is taken equal to ``fl``). A pair with no
          noise entry has no noise, as in ``cl_used``.
        * ``cl_used_tt[pair] = cl_used[pair]["TT"]``.
        """
        from ..assembly import filtering

        if cl_used is None:
            cl_used = add_nested_dicts(
                self.cl_dict_biased, self.nl_dict_biased, missing="identity"
            )
        terms: dict[str, list[tuple[str, str]]] = {}
        for cov_key in self.cov_keys.keys():
            if cov_key.stokekey() != "TTxTT":
                continue
            (a1, a2), (b1, b2) = cov_key.key_to_cross()
            terms[filtering.block_id(cov_key)] = [
                (a1.freqkey(), a2.freqkey()),
                (b1.freqkey(), b2.freqkey()),
            ]
        pairs = list(
            dict.fromkeys(p for wick in terms.values() for t in wick for p in t)
        )
        cl_pre: dict[str, np.ndarray] = {}
        cl_tt: dict[str, np.ndarray] = {}
        for pair in pairs:
            f1, f2 = pair[: len(pair) // 2], pair[len(pair) // 2 :]
            signal = (
                np.asarray(self.cl_dict[pair]["TT"], dtype=float)
                * self.beams[f1]
                * self.beams[f2]
                * self.pix["TT"]
            )
            nl = (self.nl_dict_biased.get(pair) or {}).get("TT")
            if nl is not None:
                signal = signal + filtering.pre_filter_noise(
                    nl, self.fl_dict[pair]["TT"]
                )
            cl_pre[pair] = signal
            cl_tt[pair] = np.asarray(cl_used[pair]["TT"], dtype=float)
        return terms, cl_pre, cl_tt

    def _save_map_filter(self, base_dir: str) -> None:
        """
        Write the filter sum-rule ratios beside the covariance: ``nodes``,
        ``channels`` and one ``g_<channel>`` per channel; with
        ``spectrum_aware``, also ``spectrum_aware_blocks`` (the block ids)
        and one ``g_block_<i>`` per block, in that order.
        """
        if self._map_filter_ratio is None:
            return
        np.savez(os.path.join(base_dir, self.MAP_FILTER_NAME), **self._map_filter_ratio)

    def _warn_bb_only_leakage(self, cl_used: dict[str, dict[str, np.ndarray]]) -> None:
        """
        Fire the BB-only NKA/INKA leakage warning (docs/theory/bmode_kernels.md)
        if this run is one: ``observables: [BB]`` alone with
        ``covariance_approximation: nka`` or ``inka``
        (``config.validation.BB_ONLY_OBSERVABLES``).

        Reads ``C^BB`` from ``cl_used`` (the first frequency pair, taken as
        representative of the others on the same mask) and ``C^EE`` from the
        ``cmb_spectrum`` source via
        :meth:`~cmbcov.spectra.SpectraLoader.read_extra_stokes_spectrum`.
        Sets :attr:`_bb_leakage_report` for :meth:`_save_bb_leakage_report`;
        leaves it ``None`` for every other run.
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
            Output directory.
        ls : np.ndarray or None
            Bandpower centres (``lbins.dat``).
        final_cov : np.ndarray or None
            Binned covariance matrix (``cov_name``). With ``None`` for either
            (a dry run) nothing is saved.
        save_windows : bool
            Whether to save the window functions.
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

        self._save_conditioning_report(base_dir, final_cov)
        self._save_error_budget(base_dir)
        self._save_bb_leakage_report(base_dir)
        self._save_map_filter(base_dir)

        # Save window functions if requested
        if save_windows:
            self._save_window_functions(base_dir, ls)

        self.logger.info("Results saved successfully")

    #: Name of the conditioning report written beside the covariance.
    CONDITIONING_REPORT_NAME = "conditioning.txt"

    def _save_conditioning_report(self, base_dir: str, final_cov: np.ndarray) -> None:
        """
        Write how well-conditioned the delivered covariance is beside it,
        and log its headline numbers.

        This is a report only: it changes nothing, so a failure here is
        logged, not raised.
        """
        try:
            report = conditioning_report(final_cov)
            path = os.path.join(base_dir, self.CONDITIONING_REPORT_NAME)
            with open(path, "w") as handle:
                handle.write(format_conditioning_report(report))
        except Exception as error:  # noqa: BLE001 - diagnostics must not fail a run
            self.logger.warning(f"Could not build the conditioning report: {error}")
            return
        if report["positive_definite"]:
            self.logger.info(
                f"Conditioning: min eigenvalue = {report['min_eigenvalue']:.3e}, "
                f"condition number = {report['condition_number']:.3e}. "
                f"Report: {path}"
            )
        else:
            self.logger.warning(
                "Conditioning: the covariance is not positive definite "
                f"(min eigenvalue = {report['min_eigenvalue']:.3e}, "
                f"{report['n_negative']} negative eigenvalue(s)). "
                f"Report: {path}"
            )

    #: Name of the error-budget report written beside the covariance.
    ERROR_BUDGET_NAME = "error_budget.txt"

    #: Name of the BB-only NKA/INKA leakage-warning report written beside
    #: the covariance (docs/theory/bmode_kernels.md).
    BB_LEAKAGE_WARNING_NAME = "bb_leakage_warning.txt"

    def _save_bb_leakage_report(self, base_dir: str) -> None:
        """
        Write :attr:`_bb_leakage_report` (set by
        :meth:`_warn_bb_only_leakage`) beside the covariance, as the run's
        record of the warning it already logged. Nothing is written for a
        run that is not BB-only NKA/INKA.
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

        Only ACC has one (:mod:`~cmbcov.approximations.acc.budget`); nothing
        is written otherwise. This is a report: it reads the kernels already
        in the cache and cannot change the covariance, so a failure here is
        logged, not raised.
        """
        if self.covariance_instance is None or self.cov_keys is None:
            return
        from ..approximations.acc import budget as acc_budget

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
        if budget.get("applicable", True) is False:
            # TT-only or B run: no leakage message. The normalisation line
            # (INFO, or WARNING above the threshold) was logged by the check.
            return
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
    #: window-function text files (:mod:`cmbcov.assembly.windows`): one plain-text file
    #: per two-letter spectrum and frequency pair
    #: (``<X>_<f1>x<f2>_window_functions.txt``).
    WINDOW_FUNCTIONS_DIRNAME = "windows"

    def _window_debiasing(self, freq_key: str, stoke_key: str) -> np.ndarray | None:
        r"""
        The output-axis factor of the saved windows for one spectrum: the
        debiasing of the *data* bandpower, ``1 / data_model`` exactly
        (reported multipoles only; 1 where the data model vanishes, as in
        ``debiasing_dict``).

        It is :attr:`debiasing_dict` without ``post_process_correction`` and
        without the ``add_tf_uncertainty`` factor
        :math:`1 + \sqrt{(1 - F_\ell)/3999}`: both multiply the covariance
        only, never the data. With neither set it equals
        ``debiasing_dict[freq_key][stoke_key]``. ``None`` when the run has no
        data model for this spectrum.
        """
        model = (self.data_model or {}).get(freq_key, {}).get(stoke_key)
        if model is None:
            return None
        lmax = self.spectra.lmax
        factor = safe_divide(1.0, np.asarray(model)[:lmax], default=1.0)
        return np.nan_to_num(factor, nan=1.0, posinf=1.0, neginf=1.0)

    def _save_window_functions(self, base_dir: str, ls: np.ndarray) -> None:
        r"""
        Save the bandpower window functions beside the covariance, one
        plain-text file per two-letter spectrum and frequency pair this run
        reports, into :attr:`WINDOW_FUNCTIONS_DIRNAME`.

        Each file is the linear map from the fiducial (sky, unbeamed) theory
        spectrum to the mean of that pair's binned, debiased *data*
        bandpower (:func:`~cmbcov.assembly.windows.build_window_functions`). The
        instrument enters on both axes, as it does in the covariance: the
        pair's data model ``data_model`` (beams of both frequencies, pixel
        window, transfer function; the factor ``cl_dict_biased`` carries) on
        the input axis, and the data's debiasing ``1 / data_model``
        (:meth:`_window_debiasing`) on the output axis. This is the
        covariance's ``debiasing_dict``
        (:meth:`~cmbcov.assembly.postprocess.CovariancePostProcessor.apply_debiasing`)
        without ``post_process_correction`` and the ``add_tf_uncertainty``
        inflation, which multiply the covariance only.
        The spectra and frequency pairs, and their order, are those of
        :attr:`cov_keys`'s ``spec_keys``.

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

        # polspice_postprocess is on the covariance sub-config, not a
        # `PipelineConfig` property.
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
        data_model = self.data_model or {}

        windows_dir = os.path.join(base_dir, self.WINDOW_FUNCTIONS_DIRNAME)
        os.makedirs(windows_dir, exist_ok=True)

        for spec in self.cov_keys.spec_keys:
            stoke_key = spec.stokekey()
            freq_key = spec.freqkey()

            leg_factor = self._window_debiasing(freq_key, stoke_key)
            row_scale = {stoke_key: leg_factor} if leg_factor is not None else None
            # The covariance is computed for cl_dict_biased = data_model * C
            # (SpectraLoader.prepare_workflow), so the mean bandpower responds
            # to the sky spectrum C through data_model on the input axis.
            model = data_model.get(freq_key, {}).get(stoke_key)
            column_scale = {stoke_key: model} if model is not None else None

            windows = build_window_functions(
                [stoke_key],
                self.covariance_instance,
                lbins,
                Dl=self.config.Dl,
                polspice_postprocess=polspice_postprocess,
                row_scale=row_scale,
                column_scale=column_scale,
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
                "sky theory spectrum (cmb_spectrum, no beam) and <C_hat_b> "
                "the mean of the debiased data bandpower; W = Bin "
                "diag(Dl / data_model) K diag(data_model) (K the mean "
                "kernel), the beam, pixel window and transfer function "
                "entering on both axes as in the covariance "
                "(SpectraLoader.data_model); no post_process_correction or "
                "add_tf_uncertainty factor (covariance only)."
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
        save : bool
            Whether to save results.
        save_corrections : bool
            Accepted but currently unused.
        save_windows : bool
            Whether to save window functions.
        dryrun : bool
            Whether to do a dry run (nothing is computed or saved).
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

            # compute_covariance_matrix returns (binned_ells, bin_matrix,
            # covariance); save_results takes the centres and the covariance.
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
