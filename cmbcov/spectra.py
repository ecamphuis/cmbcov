"""
The data model: parameters to physical inputs.

Turns a validated run configuration into the quantities the covariance
actually needs -- beams, pixel window, transfer functions, signal and noise
spectra, and the post-processing corrections -- and then assembles the
"biased" spectra that a real instrument would measure:

    C_biased = C_true * B_l1 * B_l2 * pixwin * F_l

together with the inverse factors used to debias the covariance afterwards.

This is deliberately separate from the workflow that drives it: everything
here is a function of the parameter file and the frequency/Stokes layout, and
nothing here knows about output directories, versioning or the covariance
object.
"""

import logging
import os
import warnings
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np

from .utils import (
    dl_to_cl,
    mult_nested_dicts,
    read_spectrum_file,
    safe_divide,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from .generator.parameter_validation import PipelineConfig

__all__ = ["SpectraLoader", "detect_parity_odd_nonzero", "stokes_columns"]

#: Spectrum-file column layouts this package reads, as the number of spectrum
#: columns (after the leading ``ell`` column) mapped to the spectra they hold,
#: in order. A 4-column file has no ``ET``/``BT``/``BE`` column: those are the
#: same spectra as ``TE``/``TB``/``EB`` with the two fields swapped, so they
#: are taken from those columns (``C_l^{ET} = C_l^{TE}``, exact for any
#: statistically isotropic sky). A 9-column file gives them explicitly and
#: they are read as they are, so a file that distinguishes them is honoured.
SPECTRUM_FILE_LAYOUTS = {
    1: ("TT",),
    4: ("TT", "EE", "BB", "TE"),
    9: ("TT", "EE", "BB", "TE", "TB", "EB", "ET", "BT", "BE"),
}

#: Parity-odd spectra: absent from a file (a 1- or 4-column layout, or a
#: 9-column one written without them), they default to zero rather than
#: raising -- the usual EB null-test convention -- unlike every other
#: spectrum :func:`stokes_columns` reads.
_PARITY_ODD_KEYS = ("TB", "BT", "EB", "BE")


def _read_full_spectrum_columns(file_path: str) -> np.ndarray | None:
    """
    Every row of a spectrum file's columns (no leading ``ell`` column, no
    ``lmax`` truncation or padding), for :func:`detect_parity_odd_nonzero`.

    Deliberately independent of :func:`~cmbcov.utils.array_utils.read_spectrum_file`,
    which requires the file to reach a given ``lmax`` and returns exactly
    ``lmax`` rows -- checking C^TB/C^EB for zero needs the opposite: every
    row the file has, however many that is, with no length requirement.

    Returns
    -------
    ndarray or None
        ``(nrows, nspec)``, or ``None`` if the file does not exist or is not
        one of the supported formats (``.txt``, ``.dat``, ``.fits``); the
        caller then has nothing to check, not an error to raise (the real
        reader, used elsewhere, is what validates the file).
    """
    if not os.path.exists(file_path):
        return None
    if file_path.endswith(".fits"):
        import healpy as hp  # lazy, see utils/healpy_utils.py

        spectrum = np.asarray(hp.read_cl(file_path))
        if spectrum.ndim == 1:
            return spectrum[:, np.newaxis]
        if spectrum.ndim == 2:
            return spectrum.T
        return None
    if file_path.endswith(".txt") or file_path.endswith(".dat"):
        spectrum = np.loadtxt(file_path)
        if spectrum.ndim != 2 or spectrum.shape[1] < 2:
            return None
        return spectrum[:, 1:]
    return None


def _file_has_nonzero_parity_odd(file_path: str) -> bool:
    """Whether any TB/BT/EB/BE column of ``file_path`` is non-zero anywhere
    (every row present, not just up to some ``lmax``)."""
    values = _read_full_spectrum_columns(file_path)
    if values is None:
        return False
    layout = SPECTRUM_FILE_LAYOUTS.get(values.shape[1])
    if layout is None:
        return False
    for key in _PARITY_ODD_KEYS:
        if key in layout and np.any(values[:, layout.index(key)] != 0):
            return True
    return False


def _value_has_nonzero_parity_odd(value: Any, freq_key: str | None) -> bool:
    """
    One ``cmb_spectrum``/``foregrounds``-shaped parameter value: whether it
    supplies a non-zero TB or EB anywhere.

    A constant (``cmb_spectrum`` only) applies to every requested Stokes
    key alike, including TB/EB if they are requested, so a non-zero
    constant counts; a per-Stokes-key dict with no ``TB``/``EB`` entry is
    zero by construction, matching :meth:`SpectraLoader._load_cmb_spectrum`.
    A file path is read with ``freq_key`` substituted in if given
    (``foregrounds`` is one file per frequency pair; ``cmb_spectrum`` is
    read once, ``freq_key=None``).
    """
    if value is None:
        return False
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, dict):
        return any(value.get(key, 0.0) for key in _PARITY_ODD_KEYS)
    if isinstance(value, str):
        path = value.format(freq_key) if freq_key is not None else value
        return _file_has_nonzero_parity_odd(path)
    return False


def detect_parity_odd_nonzero(
    cmb_spectrum: Any, foregrounds: Any, combined_frequencies: Sequence[str]
) -> bool:
    """
    Whether ``C^TB`` or ``C^EB`` is non-zero anywhere in ``cmb_spectrum`` or
    ``foregrounds``, over every row present in the file(s) actually used --
    not merely up to ``lmax``: a non-zero value past ``lmax`` only costs extra
    ACC kernel pairs, so being conservative over the whole file is fine, and
    an EB null test (TB/EB requested but zero or absent) must get the
    smaller, 18-pair kernel set, not 40.

    Needs no ``lmax`` and therefore no ACC kernel cache -- unlike
    :attr:`SpectraLoader.spectra_lmax` -- which is what lets both the ACC
    precompute (before any cache exists) and :attr:`SpectraLoader.parity_odd_nonzero`
    share this one function, so the two decisions cannot disagree.

    Parameters
    ----------
    cmb_spectrum : float, dict, str or None
        The ``cmb_spectrum`` parameter value, read once (no frequency
        substitution).
    foregrounds : str or None
        The ``foregrounds`` parameter value (a filename template over
        frequency pairs, or ``None``).
    combined_frequencies : sequence of str
        Frequency-pair keys ``foregrounds`` is checked at, e.g. from
        :meth:`~cmbcov.keys.CovKeys.combined_frequencies`.
    """
    if _value_has_nonzero_parity_odd(cmb_spectrum, None):
        return True
    return any(
        _value_has_nonzero_parity_odd(foregrounds, cf) for cf in combined_frequencies
    )


def stokes_columns(
    values: np.ndarray, stokes_keys: Sequence[str], file_path: str, what: str
) -> dict[str, np.ndarray]:
    """
    Pick one column per Stokes key out of a spectrum file's columns.

    Parameters
    ----------
    values : ndarray
        ``(lmax, nspec)``, as :func:`read_spectrum_file` returns it: the
        spectrum columns of the file, without the leading ``ell`` column.
    stokes_keys : sequence of str
        The Stokes pairs wanted, e.g. ``("TT", "TE", "ET")``.
    file_path, what : str
        Named in error messages: the file read and the parameter it is for.

    Returns
    -------
    dict
        ``{stokes key: column}``.

    Raises
    ------
    ValueError
        If the file has a column count outside
        :data:`SPECTRUM_FILE_LAYOUTS`, or holds none of the columns a
        requested spectrum could come from (e.g. ``EE`` from a 1-column
        file).

    Notes
    -----
    ``ET``, ``BT`` and ``BE`` fall back to ``TE``, ``TB`` and ``EB`` when the
    file does not carry them (4-column files); see
    :data:`SPECTRUM_FILE_LAYOUTS`.
    """
    n_columns = values.shape[1]
    layout = SPECTRUM_FILE_LAYOUTS.get(n_columns)
    if layout is None:
        raise ValueError(
            f"Unsupported {what} file {file_path!r}: found {n_columns} spectrum "
            "columns (after the leading ell column). Expected "
            + ", ".join(
                f"{n} ({', '.join(spectra)})"
                for n, spectra in sorted(SPECTRUM_FILE_LAYOUTS.items())
            )
            + "."
        )
    columns = {spectrum: index for index, spectrum in enumerate(layout)}
    output = {}
    for key in stokes_keys:
        index = columns.get(key)
        if index is None:
            # ET/BT/BE are TE/TB/EB with the fields swapped.
            index = columns.get(key[::-1])
        if index is None:
            if key in _PARITY_ODD_KEYS:
                # C^TB, C^EB default to zero when the file does not carry
                # them: the usual EB null-test convention, unlike every
                # other spectrum this function reads, which is an error when
                # missing.
                warnings.warn(
                    f"{what} file {file_path!r} has {n_columns} spectrum "
                    f"columns ({', '.join(layout)}) and no {key!r} (or "
                    f"{key[::-1]!r}) column; treating C^{key} as zero.",
                    UserWarning,
                    stacklevel=2,
                )
                output[key] = np.zeros(values.shape[0])
                continue
            raise ValueError(
                f"{what} file {file_path!r} has {n_columns} spectrum columns "
                f"({', '.join(layout)}) and so cannot supply {key!r}"
            )
        output[key] = values.T[index]
    return output


class SpectraLoader:
    """
    Builds the physical inputs to the covariance from a parameter file.

    Parameters
    ----------
    config : PipelineConfig
        Validated run configuration, as built by
        :meth:`ParameterManager.load_and_validate`. It is the only
        representation of the parameters this class reads.
    combined_frequencies : list of str
        Frequency-pair keys, e.g. ``"090GHz150GHz"``, as produced by
        :meth:`CovKeys.combined_frequencies`.
    combined_stokes : list of str
        Stokes-pair keys, e.g. ``"TE"``.
    logger : logging.Logger, optional
        Logger to report progress to.
    """

    def __init__(
        self,
        config: "PipelineConfig",
        combined_frequencies: list,
        combined_stokes: list,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.combined_frequencies = combined_frequencies
        self.combined_stokes = combined_stokes
        self.logger = logger or logging.getLogger(__name__)

        # instrument and sky inputs
        self.beams: dict[str, np.ndarray] = {}
        self.pix: dict[str, np.ndarray] = {}
        self.fl_dict: dict[str, dict[str, np.ndarray]] = {}
        self.cl_dict: dict[str, dict[str, np.ndarray]] = {}
        self.nl_dict: dict[str, dict[str, np.ndarray]] = {}
        self.post_process_corrections: dict[str, dict[str, np.ndarray]] = {}

        # derived
        self.data_model: dict[str, dict[str, np.ndarray]] = {}
        self.cl_dict_biased: dict[str, dict[str, np.ndarray]] = {}
        self.nl_dict_biased: dict[str, dict[str, np.ndarray]] = {}
        self.debiasing_dict: dict[str, dict[str, np.ndarray]] = {}

        # Whether the noise spectra carry the instrumental response. It starts
        # from the parameter file and is overridden by _load_noise_spectra when
        # the user supplies white-noise levels, which are unbiased by
        # construction.
        self.noise_is_biased: bool = bool(config.nl_is_biased)
        self._spectra_lmax: int | None = None

    @property
    def params(self) -> Mapping[str, Any]:
        """
        Read-only record of the parameter file. See :attr:`config`, which is
        the source of truth this class actually reads.
        """
        return self.config.params

    @property
    def lmax(self) -> int:
        """Maximum multipole, from the run configuration."""
        return self.config.lmax

    @property
    def spectra_lmax(self) -> int:
        """
        Number of multipoles the signal, noise, beams, pixel window and
        transfer functions are built with: ``lmax``, except for ACC, which
        reads the spectra to ``lmax_int = lmax + max(0, S - 1 - centralell)``
        so that no reported element's coupling window is cut at ``lmax``
        (:func:`~cmbcov.approximations.acc.acc_window_pad`,
        ``S`` the size of the kernels in ``acc_kernel_dir``). The covariance
        itself, the post-processing corrections and the debiasing are
        applied to ``lmax``.

        Raises
        ------
        ValueError
            For ACC, if no kernel for this run is found in ``acc_kernel_dir``
            (the kernel size, and hence ``lmax_int``, is unknown).
        """
        if self._spectra_lmax is None:
            self._spectra_lmax = self._resolve_spectra_lmax()
        return self._spectra_lmax

    def _resolve_spectra_lmax(self) -> int:
        from .approximations.acc import (
            acc_cached_kernel_size,
            acc_internal_lmax,
            require_acc_cache_pairs,
        )
        from .covariance import CovarianceMethod

        config = self.config
        if config.method != CovarianceMethod.ACC:
            return self.lmax

        if any("B" in obs for obs in config.observables):
            # A B observable: derive the needed pairs from the observables
            # list, not the plain T/E channel square, and -- unlike the
            # plain path below -- fail loudly (the existing "recompute"
            # error, naming the missing pairs) if the on-disk cache does
            # not cover them, rather than silently reporting whatever size
            # the pairs it does have imply.
            from .bmode_wick import required_kernel_pairs

            pairs = sorted(
                required_kernel_pairs(
                    config.observables,
                    parity_odd_nonzero=self.parity_odd_nonzero,
                    parity_mixed_blocks=config.parity_mixed_blocks,
                )
            )
            require_acc_cache_pairs(
                config.acc_kernel_dir, config.centralell, config.dmax, pairs
            )
        else:
            from .generator.parameter_validation import acc_kernel_spectra_needed

            needed = acc_kernel_spectra_needed(config.stokes)
            pairs = [(a, b) for a in needed for b in needed]

        size = acc_cached_kernel_size(
            config.acc_kernel_dir, config.centralell, config.dmax, pairs=pairs
        )
        if size is None:
            raise ValueError(
                "No ACC coupling kernel found under "
                f"{config.acc_kernel_dir!r}/covariance_coupling for (centralell, "
                f"centralell + d) = ({config.centralell}, {config.centralell}+d), "
                f"d < {config.dmax}. Run `precompute-acc` on this parameter file "
                "first: the spectra are loaded to lmax + max(0, S - 1 - "
                "centralell), which depends on the kernel size S."
            )
        return acc_internal_lmax(self.lmax, size, config.centralell)

    def _units_of(self, input_name: str) -> str:
        """
        ``"Cl"`` or ``"Dl"`` for one of the three spectrum inputs
        (``cmb_spectrum``, ``foregrounds``, ``nl``), from
        :meth:`~cmbcov.generator.parameter_validation.PipelineConfig.spectrum_units_of`.

        A configuration object that predates the key (any object duck-typing
        as a ``PipelineConfig``) has no such method; it then means ``"Cl"``,
        which is what the package did before the key existed.
        """
        resolve = getattr(self.config, "spectrum_units_of", None)
        if resolve is None:
            return "Cl"
        return resolve(input_name)

    def _read_file(
        self,
        path: str,
        what: str,
        units: str | None = None,
        lmax: int | None = None,
    ) -> np.ndarray:
        """
        :func:`read_spectrum_file` to ``lmax`` (:attr:`spectra_lmax` by
        default), naming the ACC padding when a file stops short of it, and
        applying the input's units convention.

        Parameters
        ----------
        units : {"Cl", "Dl", None}, optional
            The convention the file's columns are written in, for a file
            that holds power spectra. ``"Dl"`` converts every column with
            :func:`~cmbcov.utils.array_utils.dl_to_cl`
            *here*, before anything else sees the array, so the rest of the
            package only ever handles ``C_l``. ``None`` (the default) marks a
            **dimensionless** input -- beams, the pixel window, transfer
            functions, post-processing corrections -- which carries no units
            key and is never scaled; ``"Cl"`` is a spectrum that needs no
            conversion. ``None`` and ``"Cl"`` do exactly the same thing to
            the array; they are distinguished so that every call site says
            which kind of input it is reading.
        """
        lmax = self.spectra_lmax if lmax is None else lmax
        try:
            values = read_spectrum_file(path, lmax)
        except ValueError as e:
            if lmax > self.lmax and "extend" in str(e):
                raise ValueError(
                    f"{e} -- {what}: the ACC method reads spectra to lmax_int = "
                    f"{lmax} (lmax {self.lmax} + pad "
                    f"{lmax - self.lmax}), so the file must reach "
                    f"ell = {lmax - 1}. Nothing is extrapolated."
                ) from e
            raise
        if units == "Dl":
            self.logger.info(
                f"Converting {what} from D_l to C_l (2 pi D_l / (l(l+1)); "
                "ell 0 and 1 set to zero)"
            )
            values = dl_to_cl(values)
        return values

    def _load_beams(self) -> None:
        """Load and process beam data."""
        import healpy as hp  # lazy, see utils/healpy_utils.py

        self.logger.info("Loading beams...")

        beam_param = self.config.beams
        if isinstance(beam_param, str):
            self.logger.info(f"Reading beams from: {beam_param}")
            # units=None: B_l is a dimensionless window, not a power
            # spectrum; spectrum_units does not apply to it.
            beams_from_file = self._read_file(beam_param, "beams", units=None)
            self.beams = {
                freq: beams_from_file[:, i]
                for i, freq in enumerate(self.config.frequencies)
            }
        elif isinstance(beam_param, dict):
            self.logger.info("Generating Gaussian beams from FWHM values in arcmin")
            self.beams = {
                freq: hp.gauss_beam(
                    beam_param[freq] * np.pi / (180.0 * 60.0), self.spectra_lmax - 1
                )
                for freq in self.config.frequencies
            }
        elif beam_param is None:
            self.logger.info("No beams specified, setting to 1")
            self.beams = {
                freq: np.ones(self.spectra_lmax) for freq in self.config.frequencies
            }
        else:
            raise TypeError("Beam parameter type not supported")

    def _load_pixel_window(self) -> None:
        """Load pixel window function."""
        import healpy as hp  # lazy, see utils/healpy_utils.py

        self.logger.info("Loading pixel window function...")

        pixwin_param = self.config.pixwin
        if isinstance(pixwin_param, int):
            self.logger.info(
                f"Generating HEALPix pixel window for nside={pixwin_param}"
            )
            pixt, pixp = hp.pixwin(pixwin_param, pol=True, lmax=self.spectra_lmax - 1)
            if len(pixt) < self.spectra_lmax:
                raise ValueError(
                    f"The HEALPix pixel window of nside {pixwin_param} ends at ell = "
                    f"{len(pixt) - 1}, but {self.spectra_lmax} multipoles are needed "
                    f"(lmax {self.lmax}"
                    + (
                        f", ACC lmax_int {self.spectra_lmax}"
                        if self.spectra_lmax > self.lmax
                        else ""
                    )
                    + "). Nothing is extrapolated."
                )
        elif isinstance(pixwin_param, str):
            self.logger.info(f"Reading pixel window from file: {pixwin_param}")
            # Load from file
            # units=None: the pixel window is dimensionless, as for the beams.
            pix_from_file = self._read_file(pixwin_param, "pixwin", units=None)
            pixt = pix_from_file[:, 0]
            try:
                pixp = pix_from_file[:, 1]
            except IndexError:
                pixp = pixt  # Assume same for T and P if only one column
        elif pixwin_param is None:
            self.logger.info("No pixel window specified, setting to 1")
            pixt = np.ones(self.spectra_lmax)
            pixp = np.ones(self.spectra_lmax)
        else:
            raise TypeError("Pixel window parameter type not supported")

        # BB is spin-2 polarisation like EE (same pixel window); TB/BT are
        # T x polarisation like TE/ET; EB/BE are polarisation x polarisation
        # like EE.
        self.pix = {
            "TT": pixt**2,
            "EE": pixp**2,
            "ET": pixt * pixp,
            "TE": pixt * pixp,
            "BB": pixp**2,
            "TB": pixt * pixp,
            "BT": pixt * pixp,
            "EB": pixp**2,
            "BE": pixp**2,
        }

    def _load_fl(self) -> None:
        """Load transfer functions (fl and hl)."""
        self.logger.info("Loading transfer functions...")

        # units=None: fl (and hl, which nothing reads yet) is a transfer
        # function -- the dimensionless fraction of the signal the pipeline
        # keeps, in [0, 1] -- not a power spectrum. The spectrum_units key
        # does not apply to it and must never scale it.
        self.fl_dict = self._load_multiple(
            self.config.fl, "fl", self.spectra_lmax, default=1.0, units=None
        )
        if isinstance(self.config.fl, str):
            self._check_transfer_function_range(self.fl_dict, self.config.fl)

    @staticmethod
    def _check_transfer_function_range(
        fl_dict: dict[str, dict[str, np.ndarray]], template: str
    ) -> None:
        """
        Refuse a transfer function outside ``[0, 1]``.

        ``fl`` is the fraction of the signal the pipeline keeps, so it lies in
        ``[0, 1]``, and both of its uses break silently outside that range:
        the covariance is computed for spectra multiplied by
        ``data_model = B B pixwin fl`` and divided back by it afterwards, so
        ``fl <= 0`` makes that division infinite (and ``safe_divide`` then
        substitutes 1, i.e. no debiasing at all); and the transfer-function
        uncertainty ``1 + sqrt((1 - fl) / 3999)`` is NaN for ``fl > 1``, which
        the final ``nan_to_num`` turns into 1, i.e. silently no uncertainty.
        Both are exactly the silent substitutions this package raises on
        instead. No tolerance is applied: 0 and 1 themselves are fine, and
        anything beyond them has one of the two effects above.
        """
        for key_freq, spectra in fl_dict.items():
            for key_pol, values in spectra.items():
                bad = np.flatnonzero((values < 0.0) | (values > 1.0))
                if bad.size:
                    raise ValueError(
                        f"Transfer function fl from {template.format(key_freq)!r} "
                        f"({key_freq} {key_pol}) has {bad.size} multipole(s) "
                        f"outside [0, 1] (first ell = {int(bad[0])}, value "
                        f"{values[bad[0]]:.6g}; range "
                        f"{values.min():.6g} .. {values.max():.6g}). fl is the "
                        "fraction of the signal the pipeline keeps: outside "
                        "[0, 1] the debiasing and the add_tf_uncertainty factor "
                        "silently degrade to 1."
                    )

    def _load_signal_spectra(self) -> None:
        """Load CMB and foreground signal spectra."""
        self.logger.info("Loading signal spectra...")

        # Load CMB power spectrum
        cmb_spectrum = self._load_cmb_spectrum(self.spectra_lmax)

        # Load foreground spectra. These are power spectra in uK^2, added to
        # the CMB spectrum, so they carry the units key.
        fg = self._load_multiple(
            self.config.foregrounds,
            "foregrounds",
            self.spectra_lmax,
            default=0.0,
            units=self._units_of("foregrounds"),
        )

        # Compute combined signal spectra and correction factors
        self._compute_signal(cmb_spectrum, fg)

    def _load_cmb_spectrum(
        self, lmax: int, stokes_keys: list[str] | None = None
    ) -> dict[str, np.ndarray]:
        """
        Load CMB power spectrum.

        A tabulated file is read in the units
        ``config.spectrum_units_of("cmb_spectrum")`` gives and converted to
        ``C_l`` here. A *constant* spectrum (a float, or a dict of per-Stokes
        floats) is a ``C_l`` by construction -- there is no multipole to
        divide by -- so asking for ``Dl`` with one raises rather than being
        quietly ignored.

        Parameters
        ----------
        stokes_keys : list of str, optional
            Which Stokes keys to return, default :attr:`combined_stokes`.
            :meth:`read_extra_stokes_spectrum` passes a key outside
            ``combined_stokes`` -- e.g. ``EE`` for a BB-only run's leakage
            warning -- through this.
        """
        stokes_keys = self.combined_stokes if stokes_keys is None else stokes_keys
        cl_param = self.config.cmb_spectrum
        units = self._units_of("cmb_spectrum")
        if units == "Dl" and not isinstance(cl_param, str) and cl_param is not None:
            raise ValueError(
                "cmb_spectrum units are 'Dl' but cmb_spectrum is a constant "
                f"({type(cl_param).__name__}), which is a C_l by "
                "construction and cannot be converted. Set "
                "'cmb_spectrum_units: Cl' to say so explicitly."
            )

        if isinstance(cl_param, float):
            self.logger.info("Using constant CMB spectrum")
            return {key_pol: cl_param * np.ones(lmax) for key_pol in stokes_keys}
        elif isinstance(cl_param, dict):
            self.logger.info("Using constant CMB spectrum from dictionary values")
            return {
                key_pol: cl_param.get(key_pol, 0.0) * np.ones(lmax)
                for key_pol in stokes_keys
            }
        elif isinstance(cl_param, str):
            self.logger.info(f"Reading CMB spectrum from file: {cl_param}")
            cmb_from_file = self._read_file(
                cl_param, "cmb_spectrum", units=units, lmax=lmax
            )
            self.logger.info(
                "Assuming "
                + ", ".join(SPECTRUM_FILE_LAYOUTS[cmb_from_file.shape[1]])
                + " ordering"
                if cmb_from_file.shape[1] in SPECTRUM_FILE_LAYOUTS
                else "Unsupported column count"
            )
            # An unsupported column count raises here naming the file and the
            # counts supported.
            return stokes_columns(cmb_from_file, stokes_keys, cl_param, "CMB spectrum")
        else:
            raise TypeError(
                f"CMB spectrum parameter not correctly defined, cl is {cl_param}"
            )

    def read_extra_stokes_spectrum(
        self, key_pol: str, lmax: int | None = None
    ) -> np.ndarray:
        """
        Read one Stokes spectrum from ``config.cmb_spectrum``, outside
        :attr:`combined_stokes`.

        Used by the BB-only NKA/INKA leakage warning
        (``generator.generator.CovarianceMatrixGenerator.compute_covariance_matrix``,
        docs/theory/bmode_kernels.md) to read ``C^EE`` for a
        run whose ``combined_stokes`` is ``["BB"]`` alone: the same
        ``cmb_spectrum`` source the run already has, just a column
        :meth:`_load_cmb_spectrum` does not otherwise pick. Not cached and
        not part of :attr:`cl_dict`: this is a diagnostic read for a
        warning, not a covariance input -- the covariance itself never sees
        this spectrum.

        Parameters
        ----------
        key_pol : str
            The Stokes key to read, e.g. ``"EE"``.
        lmax : int, optional
            Number of multipoles, default :attr:`spectra_lmax`.
        """
        lmax = self.spectra_lmax if lmax is None else lmax
        return self._load_cmb_spectrum(lmax, stokes_keys=[key_pol])[key_pol]

    def _load_multiple(
        self,
        param: Any,
        param_name: str,
        lmax: int,
        default: float = 0.0,
        units: str | None = None,
    ) -> dict[str, dict[str, np.ndarray]]:
        """
        Load one spectrum per frequency pair, for every Stokes combination.

        Parameters
        ----------
        param : str or None
            The parameter *value*: a filename template containing ``{}``, which
            is formatted with the frequency-pair key, or None for a constant
            ``default``.
        param_name : str
            The parameter's name, used only in log and error messages.
        lmax : int
            Number of multipoles to read.
        default : float
            Value used when ``param`` is None.
        units : {"Cl", "Dl", None}, optional
            Units convention of the file, see :meth:`_read_file`. ``None``
            for a dimensionless input.
        """
        output = {}

        if param is None:
            self.logger.info(f"No {param_name} specified, setting to 0")
            for cf in self.combined_frequencies:
                output[cf] = {
                    key_pol: default * np.ones(lmax) for key_pol in self.combined_stokes
                }
        elif isinstance(param, str):
            self.logger.info(f"Reading spectra for {param_name} from file: {param}")
            for cf in self.combined_frequencies:
                spectra_from_file = self._read_file(
                    param.format(cf), param_name, units=units, lmax=lmax
                )
                output[cf] = stokes_columns(
                    spectra_from_file,
                    self.combined_stokes,
                    param.format(cf),
                    param_name,
                )
        else:
            raise TypeError(
                f"{param_name} parameter not correctly defined - must be None or string"
            )
        return output

    def _compute_signal(
        self,
        cmb_spectrum: dict[str, np.ndarray],
        fg: dict[str, dict[str, np.ndarray]],
    ) -> None:
        """Compute combined signal spectra and correction factors."""

        for key_freq in self.combined_frequencies:
            self.cl_dict[key_freq] = {}
            for key_pol in self.combined_stokes:
                # Handle potential key reversal for asymmetric cases
                try:
                    signal_spectrum = cmb_spectrum[key_pol] + fg[key_freq][key_pol]
                except KeyError:
                    self.logger.warning(
                        f"Key {key_pol} not found in foregrounds for frequency {key_freq}, trying reversed key"
                    )
                    signal_spectrum = (
                        cmb_spectrum[key_pol[::-1]] + fg[key_freq][key_pol[::-1]]
                    )

                # Apply units and correction
                self.cl_dict[key_freq][key_pol] = np.nan_to_num(
                    signal_spectrum, nan=0.0, posinf=0.0, neginf=0.0
                )

    #: Accepted spellings of the noise parameter, in order of precedence.
    #: ``nl`` is the documented name; ``noise`` is kept for backward compatibility.
    NOISE_PARAM_ALIASES = ("nl", "noise")

    def _resolve_noise_param(self) -> tuple[str | None, object]:
        """
        Find which noise parameter the user actually supplied.

        Both ``nl`` and ``noise`` are accepted as aliases for the same
        parameter, so a user-specified noise level is never silently
        discarded regardless of which spelling the parameter file uses.

        Returns
        -------
        tuple
            ``(name, value)`` of the first alias that is present and non-empty,
            or ``(None, None)`` if the user supplied no noise at all.
        """
        for name in self.NOISE_PARAM_ALIASES:
            value = getattr(self.config, name, None)
            # An empty dict/string is treated as "not supplied" so that the
            # default value from OPTIONAL_PARAMS does not masquerade as input.
            if value is None or (isinstance(value, (dict, str)) and len(value) == 0):
                continue
            return name, value
        return None, None

    def _load_noise_spectra(self) -> None:
        """
        Load noise spectra.

        A tabulated ``nl`` file is a power spectrum and is read in the units
        ``config.spectrum_units_of("nl")`` gives. White-noise *levels* (a
        dict of uK*arcmin) are not: they are turned into a flat ``C_l`` here
        by ``(sigma pi / 10800)^2``, so ``Dl`` for them raises instead of
        being ignored.
        """
        self.logger.info("Loading noise spectra...")

        self.nl_dict = {}
        param_name, noise = self._resolve_noise_param()
        units = self._units_of("nl")
        if units == "Dl" and isinstance(noise, dict):
            raise ValueError(
                f"Noise units are 'Dl' but '{param_name}' holds white-noise "
                "levels in uK*arcmin, not a tabulated spectrum; they define a "
                "flat C_l by construction. Set 'nl_units: Cl' to say so "
                "explicitly."
            )

        if noise is None:
            # Loud, not silent: a noiseless covariance underestimates the
            # error bars.
            warnings.warn(
                "No noise power spectrum supplied (looked for "
                f"{' and '.join(self.NOISE_PARAM_ALIASES)}); the covariance will "
                "be computed for signal only. Set 'nl' if this is not intended.",
                UserWarning,
                stacklevel=2,
            )
            self.nl_dict = self._load_multiple(
                None, param_name or self.NOISE_PARAM_ALIASES[0], self.spectra_lmax
            )
        elif isinstance(noise, str):
            self.nl_dict = self._load_multiple(
                noise, param_name, self.spectra_lmax, units=units
            )
        elif isinstance(noise, dict):
            self.logger.info(
                f"Using white-noise levels from '{param_name}' (uK*arcmin)"
            )
            # dict.get(key, 0.0) silently yields zero noise for any key that does
            # not match the expected frequency-pair spelling, which is how a
            # typo turns into a signal-only covariance with no diagnostic. Check
            # the keys explicitly before using them.
            expected = set(self.combined_frequencies)
            unknown = set(noise) - expected
            if unknown:
                raise ValueError(
                    f"Unrecognised {param_name} frequency-pair keys: "
                    f"{sorted(unknown)}. Expected keys of the form "
                    f"{sorted(expected)[:3]}... (no 'x' separator)."
                )
            missing = expected - set(noise)
            if missing:
                warnings.warn(
                    f"No {param_name} value given for {sorted(missing)}; "
                    "these frequency pairs will be treated as noiseless.",
                    UserWarning,
                    stacklevel=2,
                )
            # White-noise levels are given at map level, so they carry no beam
            # or transfer function. Recorded on the loader, not written back
            # into a parameter representation.
            self.noise_is_biased = False
            self.nl_dict = {}
            for cf in self.combined_frequencies:
                # translate from uk*arcmin to uK^2
                val = noise.get(cf, 0.0) ** 2 / 10800**2 * np.pi**2
                # BB gets the same white-noise level as EE (same polarised
                # detector noise); every cross spectrum -- TE/ET as before,
                # and now TB/BT/EB/BE -- is noiseless, since map noise is
                # uncorrelated between Stokes parameters.
                temp_dict = {
                    "TT": val,
                    "EE": 2 * val,
                    "BB": 2 * val,
                    "TE": 0.0,
                    "ET": 0.0,
                    "TB": 0.0,
                    "BT": 0.0,
                    "EB": 0.0,
                    "BE": 0.0,
                }
                self.nl_dict[cf] = {
                    key_pol: temp_dict[key_pol] * np.ones(self.spectra_lmax)
                    for key_pol in self.combined_stokes
                }
        else:
            raise TypeError(
                f"Noise parameter '{param_name}' must be a filename or a dict of "
                f"white-noise levels, got {type(noise).__name__}"
            )

        for noise in self.nl_dict.values():
            for key_pol in self.combined_stokes:
                noise[key_pol] = np.nan_to_num(
                    noise[key_pol], nan=0.0, posinf=0.0, neginf=0.0
                )
            # Only where ET is one of this run's spectra: prepare_workflow
            # multiplies nl_dict by a data model built over exactly
            # combined_stokes and refuses any extra key, so an unconditional
            # alias broke every single-frequency run with white-noise `nl`
            # (combined_stokes has TE but no ET there).
            if (
                ("TE" in noise)
                and ("ET" not in noise)
                and ("ET" in self.combined_stokes)
            ):
                noise["ET"] = noise["TE"]

        self.logger.info(f"Loaded noise for frequencies: {list(self.nl_dict.keys())}")

    @property
    def parity_odd_nonzero(self) -> bool:
        """
        Whether ``C^TB`` or ``C^EB`` is non-zero anywhere it was requested:
        the usual EB null test has both exactly zero, which is the smaller
        (18-pair) ACC kernel set; any non-zero value needs the larger
        (40-pair) one (:func:`~cmbcov.bmode_wick.required_kernel_pairs`).
        False (not an error) if TB/EB were not requested at all
        (``"TB"``/``"EB"`` absent from :attr:`combined_stokes`) -- they are
        then not even read (:func:`stokes_columns`), so there is nothing to
        check.

        Delegates to :func:`detect_parity_odd_nonzero`, over the raw
        ``cmb_spectrum``/``foregrounds`` inputs -- not :attr:`cl_dict`, which
        is truncated to ``spectra_lmax`` -- so this needs no ``spectra_lmax``
        (hence no ACC kernel cache) and gives the same answer the ACC
        precompute's own check does, before either loads any spectra.
        """
        if "TB" not in self.combined_stokes and "EB" not in self.combined_stokes:
            return False
        return detect_parity_odd_nonzero(
            self.config.cmb_spectrum, self.config.foregrounds, self.combined_frequencies
        )

    def load_pre_process(self) -> None:
        """Load all preprocessing data: beams, pixel windows, transfer functions, and spectra."""
        self._load_beams()
        self._load_pixel_window()
        self._load_fl()
        self._load_signal_spectra()
        self._load_noise_spectra()

    def load_post_process(self) -> None:
        """Load post-processing correction factors.
        All post-processing corrections are in form of mutiple factors (Hl, inpainting, etc.)

        These are dimensionless multiplicative factors applied to the
        covariance afterwards, not power spectra, so ``units=None``
        everywhere below: the ``spectrum_units`` key never scales them.
        """
        self.logger.info("Loading post-processing correction factors...")

        ppc_param = self.config.post_process_correction

        self.post_process_corrections = None
        if ppc_param is None:
            self.logger.info("No post-processing corrections specified, setting to 1")
            self.post_process_corrections = {
                cf: {key_pol: np.ones(self.lmax) for key_pol in self.combined_stokes}
                for cf in self.combined_frequencies
            }
        elif isinstance(ppc_param, str):
            self.logger.info(
                f"Reading post-processing corrections from file: {ppc_param}"
            )
            self.post_process_corrections = self._load_multiple(
                ppc_param, "post_process_correction", self.lmax, units=None
            )
        elif isinstance(ppc_param, dict):
            for ppc_name, ppc in ppc_param.items():
                self.logger.info(f"Loading post-processing correction: {ppc_name}")
                ppc_loaded = self._load_multiple(ppc, ppc_name, self.lmax, units=None)
                if self.post_process_corrections is None:
                    self.post_process_corrections = ppc_loaded
                else:
                    # missing="identity": two correction files may
                    # legitimately cover different spectra/frequencies.
                    self.post_process_corrections = mult_nested_dicts(
                        self.post_process_corrections, ppc_loaded, missing="identity"
                    )
        else:
            raise TypeError("post_process_correction parameter type not supported")

        # if isinstance(inpainting_param, dict):
        #     for key, val in inpainting_param.items():
        #         try:
        #             interp_data = hp.read_cl(val)
        #             self.inpainting_rescaling[key] = 1 + (1 - interp_data[0, :self.lmax])
        #         except Exception as e:
        #             self.logger.warning(f"Error loading inpainting correction for {key}: {e}")
        #             self.inpainting_rescaling[key] = np.ones(self.lmax)

        # elif isinstance(inpainting_param, float):
        #     for cf in self.combined_frequencies:
        #         self.inpainting_rescaling[cf] = inpainting_param * np.ones(self.lmax)

        # else:
        #     raise TypeError("inpainting_rescaling parameter not correctly defined")

    # ----------------------------------------------------------------------------------
    # Main workflow methods
    # ----------------------------------------------------------------------------------

    def prepare_workflow(self) -> tuple[dict, dict, dict, dict]:
        """Prepare spectra, data model, and post processing for covariance computation.

        Returns
        -------
        tuple
            (cl_dict_biased, nl_dict_biased, data_model, debiasing_dict)
        """
        self.logger.info("Preparing final spectra and data model...")

        # start by preparing the correction factors
        self.data_model = {}
        for key_freq in self.combined_frequencies:
            self.logger.info(f"Preparing data model for {key_freq}")
            self.data_model[key_freq] = {}
            kf1, kf2 = key_freq[: len(key_freq) // 2], key_freq[len(key_freq) // 2 :]
            for key_pol in self.combined_stokes:
                self.data_model[key_freq][key_pol] = (
                    self.beams[kf1]
                    * self.beams[kf2]
                    * self.pix[key_pol]
                    * self.fl_dict[key_freq][key_pol]
                )

        # missing="error": cl_dict/nl_dict and data_model are both built over
        # exactly combined_frequencies x combined_stokes (see
        # _compute_signal, _load_multiple and load_pre_process), so a key
        # missing from data_model here means beams/pixwin/fl were silently
        # not applied to it -- a bug, not a legitimate configuration.
        self.cl_dict_biased = mult_nested_dicts(
            self.cl_dict, self.data_model, missing="error"
        )
        if not self.noise_is_biased:
            self.nl_dict_biased = mult_nested_dicts(
                self.nl_dict, self.data_model, missing="error"
            )
        else:
            self.nl_dict_biased = deepcopy(self.nl_dict)

        # The per-leg factor applied to the covariance of each spectrum
        # C^s_f (f a frequency pair, s a Stokes pair -- the keys
        # CovariancePostProcessor.apply_debiasing splits a CovKey into):
        #
        #   d[f][s] = post_process_correction[f][s] / data_model[f][s]
        #             * tf_uncertainty[f][s]
        #
        # Block (C^s1_f1, C^s2_f2) is multiplied by outer(d[f1][s1], d[f2][s2]),
        # so each factor enters exactly once per leg. Only the
        # reported multipoles are kept: the spectra may be longer (ACC reads
        # them to lmax_int), the corrections are loaded to lmax.
        lmax = self.lmax
        self.debiasing_dict = {}
        for key_freq in self.combined_frequencies:
            self.debiasing_dict[key_freq] = {}
            for key_pol in self.combined_stokes:
                self.logger.info(f"Preparing debiasing for {key_freq} {key_pol}")
                factor = safe_divide(
                    1.0, self.data_model[key_freq][key_pol][:lmax], default=1.0
                )

                # Transfer-function uncertainty (if requested). fl is the
                # fraction of the signal the pipeline keeps, in [0, 1]
                # (_check_transfer_function_range refuses a file outside it),
                # so 1 + sqrt((1 - fl) / 3999) >= 1: it MULTIPLIES the leg
                # factor, and the block, which is the point -- not knowing the
                # transfer function exactly can only add variance, and the
                # more signal the filtering removed (small fl) the larger the
                # inflation.
                if self.config.add_tf_uncertainty:
                    self.logger.info("Adding transfer function uncertainty...")
                    tf_uncertainty = 1 + np.sqrt(
                        (1 - self.fl_dict[key_freq][key_pol][:lmax]) / (4000 - 1)
                    )
                    factor *= tf_uncertainty

                correction = np.asarray(
                    self.post_process_corrections[key_freq][key_pol]
                )[:lmax]
                if correction.shape != factor.shape:
                    raise ValueError(
                        f"post_process_correction for {key_freq} {key_pol} has "
                        f"{correction.shape[0]} multipoles, {lmax} are needed"
                    )
                factor = correction * factor

                # Final cleanup
                self.debiasing_dict[key_freq][key_pol] = np.nan_to_num(
                    factor,
                    nan=1.0,
                    posinf=1.0,
                    neginf=1.0,
                )

        return (
            self.cl_dict_biased,
            self.nl_dict_biased,
            self.data_model,
            self.debiasing_dict,
        )
