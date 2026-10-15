"""
Parameter-file loading and validation.

The output of validation is a :class:`PipelineConfig`: one validated, frozen
object, built once, that every downstream consumer reads. The plain parameter
dictionary is only a read-only record of it (:attr:`PipelineConfig.params`).
"""

import logging
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import yaml

from ..approximations.acc.bmode_wick import (
    PARITY_ODD_SPECTRA,
    canonical_spectrum,
    required_kernel_pairs,
)
from ..approximations.acc.channels import COUPLING_CHANNELS, normalise_channel
from ..assembly.covariance import CovarianceConfig, CovarianceMethod
from ..assembly.keys import CovKeys
from ..utils.array_utils import SPECTRUM_UNITS
from ..utils.file_utils import get_git_revision_short_hash, get_versioned_path
from .run_record import RECORD_NAMES

logger = logging.getLogger(__name__)

#: Accepted alternative spellings of parameter-file keys, mapped to the key
#: this package treats as canonical.
#:
#: ``sum_assymetric_stokes`` (double s) is the canonical file key; the
#: correctly spelled ``sum_asymmetric_stokes`` is accepted too. Supplying both
#: with different values is an error.
PARAM_ALIASES: dict[str, str] = {
    "sum_asymmetric_stokes": "sum_assymetric_stokes",
    # `noise` is a legacy spelling of `nl` (resolved by SpectraLoader, which
    # accepts both); its units override follows the same rule.
    "noise_units": "nl_units",
}

#: The tabulated inputs whose file columns are power spectra in
#: :math:`\mu K^2`, and which the ``spectrum_units`` key therefore converts.
#: Each has a per-input override named ``<input>_units``; ``None`` there means
#: "follow ``spectrum_units``".
#:
#: Everything else a run reads from a file is dimensionless and is **never**
#: converted, whatever the units key says: ``beams`` (:math:`B_\ell`),
#: ``pixwin`` (the pixel window), ``fl`` and ``hl`` (transfer functions, the
#: fraction of the signal kept, in ``[0, 1]``) and
#: ``post_process_correction`` (multiplicative correction factors). Scaling
#: any of those by :math:`2\pi/\ell(\ell+1)` would be meaningless.
SPECTRUM_UNITS_INPUTS = ("cmb_spectrum", "foregrounds", "nl")

#: Parameter-file key holding the default units of every input of
#: :data:`SPECTRUM_UNITS_INPUTS`.
SPECTRUM_UNITS_KEY = "spectrum_units"


def spectrum_units_key(input_name: str) -> str:
    """The per-input units override key of ``input_name``, e.g. ``nl_units``."""
    return f"{input_name}_units"


def canonical_spectrum_units(value: Any, where: str) -> str:
    """
    Check one units value and return its canonical spelling.

    Parameters
    ----------
    value : Any
        The value as written in the parameter file. Matched against
        :data:`~cmbcov.utils.array_utils.SPECTRUM_UNITS`
        case-insensitively, so ``Dl``, ``dl`` and ``DL`` are the same
        request; anything else raises.
    where : str
        The key this value came from, quoted in the error message.

    Returns
    -------
    str
        ``"Cl"`` or ``"Dl"``.

    Raises
    ------
    ValueError
        For any other value, naming the allowed ones. Units are never
        guessed: reading a :math:`D_\\ell` file as :math:`C_\\ell` biases the
        covariance by :math:`(\\ell(\\ell+1)/2\\pi)^2`.
    """
    for unit in SPECTRUM_UNITS:
        if isinstance(value, str) and value.strip().lower() == unit.lower():
            return unit
    raise ValueError(
        f"'{where}' must be one of {list(SPECTRUM_UNITS)} "
        "('Cl' = C_l, 'Dl' = l(l+1)C_l/2pi; case-insensitive), got "
        f"{value!r}"
    )


def resolve_spectrum_units(params: Mapping[str, Any], input_name: str) -> str:
    """
    The units one input of :data:`SPECTRUM_UNITS_INPUTS` is written in.

    The per-input override ``<input>_units`` wins when it is set; otherwise
    the global ``spectrum_units``; otherwise the default (``"Cl"``). Both are
    checked by :func:`canonical_spectrum_units`, so an unknown spelling raises
    here even if validation was skipped.

    Parameters
    ----------
    params : mapping
        Parameter mapping (aliases already normalised).
    input_name : str
        One of :data:`SPECTRUM_UNITS_INPUTS`.

    Returns
    -------
    str
        ``"Cl"`` or ``"Dl"``.
    """
    return _resolve_spectrum_units(
        params.get(spectrum_units_key(input_name)),
        params.get(SPECTRUM_UNITS_KEY),
        input_name,
    )


def _resolve_spectrum_units(override: Any, global_units: Any, input_name: str) -> str:
    """
    The rule behind :func:`resolve_spectrum_units` and
    :meth:`PipelineConfig.spectrum_units_of`: per-input override, else the
    global key, else the default.
    """
    if input_name not in SPECTRUM_UNITS_INPUTS:
        raise ValueError(
            f"{input_name!r} is not a spectrum input; expected one of "
            f"{list(SPECTRUM_UNITS_INPUTS)}. Beams, the pixel window, "
            "transfer functions and post-processing corrections are "
            "dimensionless and carry no units key."
        )
    if override is not None:
        return canonical_spectrum_units(override, spectrum_units_key(input_name))
    if global_units is not None:
        return canonical_spectrum_units(global_units, SPECTRUM_UNITS_KEY)
    return ParameterValidator.OPTIONAL_PARAMS[SPECTRUM_UNITS_KEY]


EMPTY_PARAMS: Mapping[str, Any] = MappingProxyType({})

#: Parameter-file key of the ACC precompute block, see :class:`AccPrecomputeConfig`.
ACC_PRECOMPUTE_KEY = "acc_precompute"


@dataclass(frozen=True)
class AccPrecomputeConfig:
    """
    The ``acc_precompute`` block of a parameter file: how the one-off ACC
    coupling-kernel precompute (``precompute-acc``) is run.

    Only the *how* lives here; the mask, ``centralell``, ``dmax`` and the
    kernel directory are read from the file's top level, as for the
    covariance runs. Every field is the keyword of the same name of
    :func:`~cmbcov.approximations.acc.precompute_acc_kernels`, with the same
    default.

    Attributes
    ----------
    nside : int, optional
        ACC working resolution (kernels are ``2 nside`` square). ``None``
        picks it from ``centralell + dmax``.
    grid : {"healpix", "gl"}, default "gl"
        Quadrature backend of the mode-coupling integrals.
    lw : int, optional
        Mask band-limit, ``grid: gl`` only. ``None`` means ``3 nside - 1``.
    spectra : tuple of str, optional
        Kernel channels to compute, a subset of ``COUPLING_CHANNELS``. The
        observable names (``TT, EE, BB, TE, ET``) are accepted and normalised
        to the channel names (``CHANNEL_ALIASES``). ``None`` means
        ``COUPLING_SPECTRA``, the five T/E channels.
    max_memory_gb : float, optional
        Peak-memory budget of the contraction. ``None`` means the package
        default.
    scratch_dir : str, optional
        Directory for the temporary on-disk store of the central
        coefficients, used when they do not fit in half of
        ``max_memory_gb``. ``None`` means the kernel directory.
    """

    nside: int | None = None
    grid: str = "gl"
    lw: int | None = None
    spectra: tuple[str, ...] | None = None
    max_memory_gb: float | None = None
    scratch_dir: str | None = None

    #: Keys accepted in the block.
    KEYS = ("nside", "grid", "lw", "spectra", "max_memory_gb", "scratch_dir")
    #: Accepted values of ``grid``.
    GRIDS = ("healpix", "gl")

    @classmethod
    def from_block(cls, block: Mapping[str, Any] | None) -> "AccPrecomputeConfig":
        """
        Build from an already validated block (``None`` means all defaults).

        ``spectra`` is normalised to the channel names here (observable names
        are accepted, see :func:`~cmbcov.approximations.acc.normalise_channel`);
        malformed values are left as given, ``_check_acc_precompute`` having
        already rejected them.
        """
        block = dict(block or {})
        spectra = block.get("spectra")
        if (
            spectra is not None
            and isinstance(spectra, list)
            and all(isinstance(s, str) for s in spectra)
        ):
            try:
                spectra = tuple(normalise_channel(s) for s in spectra)
            except ValueError:
                spectra = tuple(spectra)
        return cls(
            nside=block.get("nside"),
            grid=block.get("grid", "gl"),
            lw=block.get("lw"),
            spectra=tuple(spectra) if spectra is not None else None,
            max_memory_gb=block.get("max_memory_gb"),
            scratch_dir=block.get("scratch_dir"),
        )


def _is_int(value: Any) -> bool:
    """A YAML integer: ``int`` but not ``bool`` (``True`` is an ``int`` in Python)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    """A YAML number: ``int`` or ``float`` but not ``bool``."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


#: Parameter-file key of the map-filter block, see :class:`MapFilterConfig`.
MAP_FILTER_KEY = "map_filter"


@dataclass(frozen=True)
class LowpassConfig:
    """
    The optional ``map_filter.lowpass`` sub-block: a scan-direction low-pass
    multiplying the high-pass of :class:`MapFilterConfig`, ``h = h_highpass
    h_lowpass`` (:func:`cmbcov.assembly.filtering.lowpass_profile`).

    Attributes
    ----------
    lx : float
        Cut of the low-pass, ``m_c = lx sin(theta)`` like the high-pass; must
        exceed the high-pass ``lx``.
    shape : {"exp", "sharp"}, default "exp"
        ``exp``: ``h = exp[-(M / (lx sin(theta)))^power]``; ``sharp``:
        ``h = 1`` where ``M <= lx sin(theta)``, else 0.
    power : float, default 6
        Steepness of the ``exp`` profile (unused by ``sharp``).
    """

    lx: float
    shape: str = "exp"
    power: float = 6.0

    #: Keys accepted in the sub-block.
    KEYS = ("lx", "shape", "power")
    #: Accepted values of ``shape``.
    SHAPES = ("exp", "sharp")

    @classmethod
    def from_block(cls, block: Mapping[str, Any]) -> "LowpassConfig":
        """Build from an already validated sub-block (absent keys default)."""
        block = dict(block)
        return cls(
            lx=block.get("lx"),
            shape=block.get("shape", "exp"),
            power=block.get("power", 6.0),
        )

    def as_dict(self) -> dict[str, Any]:
        """``{"lx", "shape", "power"}`` as plain floats and a string, the form
        :func:`cmbcov.assembly.filtering.combined_profile` takes and the cache
        identity records.  ``power`` is ``None`` for ``sharp`` (unused, so a
        change of it does not probe again)."""
        return {
            "lx": float(self.lx),
            "shape": self.shape,
            "power": float(self.power) if self.shape == "exp" else None,
        }


@dataclass(frozen=True)
class MapFilterConfig:
    """
    The ``map_filter`` block of a parameter file: the optional filter-and-bin
    correction of the T/E covariance blocks
    (:mod:`cmbcov.assembly.filtering`, docs/theory/filter_and_bin.md).

    Each raw T/E pseudo-C_l block with Stokes pair ``(s1, s2)`` is
    multiplied, elementwise, by
    ``rho^ch(l1, l2) / (fl_left[s1][l1] fl_right[s2][l2])``, where
    ``rho^ch`` is the filtered sum-rule ratio of the mask for the filter
    described here, in the block's sum-rule channel ``ch`` (``00``, ``20``
    or ``EE``). Blocks with a B leg are left uncorrected.

    Attributes
    ----------
    type : {"fourier_highpass"}
        Kind of filter. Only the scan-direction Fourier high-pass exists.
    lx : float
        Cut of the filter, ``m_c = lx sin(theta)`` (``m`` the azimuthal
        Fourier index around the map's polar axis, which must be the scan
        axis).
    shape : {"sharp", "exp"}, default "sharp"
        Filter profile around the cut.
    power : float, default 6
        Steepness of the ``exp`` profile (unused by ``sharp``).
    nprobe : int, default 32
        Number of random probes of the stochastic sum-rule estimate at the
        nodes outside the near-cut range (below ``0.95 lx`` and from
        ``2 lx`` up).
    seed : int, default 0
        Seed of those probes.
    node_min : int, default 32
        Smallest multipole of the interpolation nodes.
    node_ratio : float, default 1.25
        Geometric spacing of the nodes, ``> 1``.
    dmax : int, optional
        Largest ``|l1 - l2|`` the sum-rule ratio is probed to. ``None``
        means the run's ACC ``dmax - 1`` for ACC and 64 otherwise, capped at
        ``lmax - 1`` (:meth:`resolved_dmax`).
    lw : int, optional
        Mask band-limit of the probes. ``None`` means
        ``min(3 nside - 1, 2 lmax)`` (:meth:`resolved_lw`).
    spectrum_aware : bool, default False
        Probe every TT x TT block with the run's fiducial spectra
        (:func:`cmbcov.assembly.filtering.spectrum_aware_ratio`) instead of the white
        ``00`` channel: the block is then multiplied by its own ratio, with
        no ``fl`` division. It captures the power a sharp
        ``theta``-dependent cut scatters across multipoles; the other T/E
        blocks are unchanged. The ratio depends on the spectra, so a change
        of spectra probes again.
    near_cut_step : float, optional
        Spacing of the extra nodes between ``0.95 lx`` and ``2 lx``
        (:func:`cmbcov.assembly.filtering.default_nodes`). ``None`` means ``lx / 15``;
        ``0`` adds none (the geometric nodes alone): neither the high-pass
        near-cut nodes nor the low-pass rolloff nodes. With a ``lowpass``
        whose rolloff starts inside the covariance (``0.6 lx_lowpass <
        lmax``) the run logs a warning, since the rolloff is then
        interpolated from the geometric nodes alone.
    nprobe_near_cut : int, optional
        Probes of the nodes from ``0.95 lx`` to below ``2 lx``; default
        128. ``None`` (an explicit ``null``) means the complete basis of
        each node's degree (exact ratio, :func:`cmbcov.assembly.filtering.node_plan`),
        several times more expensive at survey scale. Those of the nodes below
        ``1.3 lx`` also carry the probes to a band-limit of at least
        ``3.5 lx`` (not a parameter).
    lowpass : LowpassConfig, optional
        A scan-direction low-pass multiplying the high-pass (``h = h_highpass
        h_lowpass``). ``None`` (the sub-block absent or ``null``): the
        high-pass alone, and every result and cache key is that of a file
        without the key. ``fl`` must include the low-pass (nothing checks
        it). The sharp shape with ``lx <= lmax`` logs a warning: the ratio
        has off-diagonal structure at the kink that the (mean, distance)
        interpolation cannot carry, so blocks within a few multipoles of
        ``lx`` are about 3-4% off; the exp shape has no such limit.
    """

    type: str
    lx: float
    shape: str = "sharp"
    power: float = 6.0
    nprobe: int = 32
    seed: int = 0
    node_min: int = 32
    node_ratio: float = 1.25
    dmax: int | None = None
    lw: int | None = None
    spectrum_aware: bool = False
    near_cut_step: float | None = None
    nprobe_near_cut: int | None = 128
    lowpass: LowpassConfig | None = None

    #: Keys accepted in the block.
    KEYS = (
        "type",
        "lx",
        "shape",
        "power",
        "nprobe",
        "seed",
        "node_min",
        "node_ratio",
        "dmax",
        "lw",
        "spectrum_aware",
        "near_cut_step",
        "nprobe_near_cut",
        "lowpass",
    )
    #: Accepted values of ``type``.
    TYPES = ("fourier_highpass",)
    #: Accepted values of ``shape``.
    SHAPES = ("sharp", "exp")
    #: Default ``dmax`` for the methods without an ACC ``dmax``.
    DEFAULT_DMAX = 64

    @classmethod
    def from_block(cls, block: Mapping[str, Any]) -> "MapFilterConfig":
        """
        Build from an already validated block (``_check_map_filter`` has
        rejected malformed values; absent optional keys take the defaults).
        """
        block = dict(block)
        return cls(
            type=block.get("type"),
            lx=block.get("lx"),
            shape=block.get("shape", "sharp"),
            power=block.get("power", 6.0),
            nprobe=block.get("nprobe", 32),
            seed=block.get("seed", 0),
            node_min=block.get("node_min", 32),
            node_ratio=block.get("node_ratio", 1.25),
            dmax=block.get("dmax"),
            lw=block.get("lw"),
            spectrum_aware=block.get("spectrum_aware", False),
            near_cut_step=block.get("near_cut_step"),
            nprobe_near_cut=block.get("nprobe_near_cut", 128),
            lowpass=(
                None
                if block.get("lowpass") is None
                else LowpassConfig.from_block(block["lowpass"])
            ),
        )

    def resolved_dmax(self, lmax: int, method: CovarianceMethod, acc_dmax: int) -> int:
        """
        The ``dmax`` the probes use: the block's, else ``acc_dmax - 1`` for
        ACC (its kernels carry ``|l1 - l2| < dmax``) and
        :attr:`DEFAULT_DMAX` otherwise; capped at ``lmax - 1``, and never
        below 0.
        """
        if self.dmax is not None:
            dmax = self.dmax
        elif method == CovarianceMethod.ACC:
            dmax = acc_dmax - 1
        else:
            dmax = self.DEFAULT_DMAX
        return max(0, min(int(dmax), lmax - 1))

    def resolved_lw(self, nside: int, lmax: int) -> int:
        """The mask band-limit of the probes: the block's, else ``min(3 nside - 1, 2 lmax)``."""
        if self.lw is not None:
            return int(self.lw)
        return min(3 * nside - 1, 2 * lmax)


#: Every valid single-frequency observable spectrum, canonical ("leg order")
#: form (docs/theory/bmode_kernels.md).
VALID_OBSERVABLES = ("TT", "EE", "BB", "TE", "TB", "EB")

#: Accepted aliases of the six observables above -- the reversed-leg
#: spelling, e.g. ``ET`` for ``TE``.
OBSERVABLE_ALIASES = {"ET": "TE", "BT": "TB", "BE": "EB"}

#: Covariance approximations that can compute a B-mode block. NKA and INKA
#: collapse ``EE`` and ``BB`` by construction (the spin-weight rule cannot
#: tell them apart), so they are degenerate for BB. Of
#: :class:`~cmbcov.assembly.covariance.CovarianceMethod`'s three members only ACC
#: qualifies (``cmbcov.exact`` is a test oracle, not a selectable
#: ``covariance_approximation``).
#:
#: The one exception is ``observables: [BB]`` alone (``BB_ONLY_OBSERVABLES``
#: below): a leakage-neglected mode for a high-``ell`` ``Cov(BB, BB)`` without
#: an ACC precompute. NKA gives ``Cov(BB, BB) = 2 (C^BB)^2 Xi^{EE->EE}``,
#: missing the leaked-E term that dominates the pseudo-BB covariance over a
#: wide ell range on realistic masks (docs/theory/bmode_kernels.md); the run
#: warns with a mask- and spectrum-specific ``ell_safe`` below which that is a
#: real underestimate (:meth:`~cmbcov.assembly.covariance.Cov.bb_leakage_ell_safe`).
#: Any other observable with a B letter (``TB``, ``EB``, or ``BB`` alongside
#: a T/E one) needs ACC.
APPROXIMATIONS_SUPPORTING_B = (CovarianceMethod.ACC,)

#: Approximations that accept ``observables: [BB]`` alone (all three); see
#: :data:`APPROXIMATIONS_SUPPORTING_B`.
APPROXIMATIONS_SUPPORTING_BB_ONLY = (
    CovarianceMethod.ACC,
    CovarianceMethod.NKA,
    CovarianceMethod.INKA,
)

#: The one ``observables`` value NKA/INKA are allowed alongside a B letter
#: (canonical form; ``BB`` has no alias). See
#: :data:`APPROXIMATIONS_SUPPORTING_B`.
BB_ONLY_OBSERVABLES = ["BB"]


def _letters_from_observables(observables: list[str]) -> list[str]:
    """The Stokes letters an ``observables`` list spans, in ``T, E, B`` order."""
    letters = {c for obs in observables for c in obs}
    return [c for c in "TEB" if c in letters]


def _observables_from_stokes(stokes: list[str]) -> list[str]:
    """
    The ``observables`` a plain ``stokes`` alphabet implies: the full
    Cartesian square :class:`~cmbcov.assembly.keys.CovKeys` builds without
    ``observables=``, canonicalised. Only called for a T/E-only ``stokes``
    (a ``B`` in ``stokes`` is rejected before).
    """
    return sorted({canonical_spectrum(a + b) for a in stokes for b in stokes})


def resolve_observables(params: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """
    The Stokes letters (for the T/E-only :class:`~cmbcov.assembly.keys.CovKeys`
    alphabet path) and the canonical ``observables`` list a parameter file
    asks for.

    Exactly one of ``stokes`` or ``observables`` must be given.
    ``stokes: [T]`` / ``[T, E]`` are accepted as shorthand for
    ``observables: [TT]`` / ``[TT, EE, TE]``; any other ``stokes`` value
    must not contain ``B`` (use ``observables`` instead, which supports
    it). ``observables`` entries may be given as ``ET``, ``BT`` or ``BE``,
    accepted as aliases of ``TE``, ``TB``, ``EB``.

    Returns
    -------
    (list of str, list of str)
        ``(stokes, observables)``. ``stokes`` is the letters the alphabet
        path needs; it is only used when ``observables`` has no ``B``.

    Raises
    ------
    ValueError
        If both or neither of ``stokes``/``observables`` are given, if
        ``stokes`` holds an invalid letter or a ``B`` without
        ``observables``, or if ``observables`` is malformed or holds an
        unknown name.
    """
    has_stokes = "stokes" in params
    has_observables = "observables" in params
    if has_stokes and has_observables:
        raise ValueError(
            "Parameter file sets both 'stokes' and 'observables'; give only "
            "one. 'observables' is the general form ('stokes: [T]' / "
            "'[T, E]' are shorthand for 'observables: [TT]' / "
            "'[TT, EE, TE]')."
        )
    if not has_stokes and not has_observables:
        raise ValueError(
            "Parameter file sets neither 'stokes' nor 'observables'; one is required."
        )

    if has_observables:
        observables = params["observables"]
        if (
            not isinstance(observables, list)
            or not observables
            or not all(isinstance(o, str) for o in observables)
        ):
            raise ValueError(
                f"'observables' must be a non-empty list of strings, got "
                f"{observables!r}"
            )
        canonical: list[str] = []
        unknown = []
        for obs in observables:
            mapped = OBSERVABLE_ALIASES.get(obs, obs)
            if mapped not in VALID_OBSERVABLES:
                unknown.append(obs)
            elif mapped not in canonical:
                canonical.append(mapped)
        if unknown:
            raise ValueError(
                f"'observables' has unknown spectra {sorted(unknown)}; "
                f"expected a subset of {list(VALID_OBSERVABLES)} (ET, BT, "
                "BE accepted as aliases of TE, TB, EB)"
            )
        return _letters_from_observables(canonical), canonical

    stokes = params["stokes"]
    if not isinstance(stokes, list) or not all(isinstance(s, str) for s in stokes):
        raise ValueError(f"'stokes' must be a list of strings, got {stokes!r}")
    if any(s not in ("T", "E", "B") for s in stokes):
        raise ValueError(
            f"Invalid Stokes parameters {stokes!r}; valid letters are 'T', 'E', 'B'"
        )
    if "B" in stokes:
        raise ValueError(
            "'stokes' contains 'B', which only works as a shorthand for "
            "'T'/'E' ('stokes: [T]' / '[T, E]' for 'observables: [TT]' / "
            "'[TT, EE, TE]'); use 'observables' to ask for B-mode spectra "
        )
    return list(stokes), _observables_from_stokes(stokes)


def _check_parity_mixed_and_approximation(
    observables: list[str], parity_mixed_blocks: bool, approximation: CovarianceMethod
) -> list[str]:
    """
    The cross-checks between ``observables``, ``parity_mixed_blocks`` and
    ``covariance_approximation`` that must be rejected, never auto-enabled.

    Returns
    -------
    list of str
        Error messages (empty if everything is consistent).
    """
    errors = []
    parity_odd = [o for o in observables if o in PARITY_ODD_SPECTRA]
    parity_even = [o for o in observables if o not in PARITY_ODD_SPECTRA]
    if parity_mixed_blocks:
        if not parity_odd:
            errors.append(
                "'parity_mixed_blocks: true' requires 'TB' or 'EB' in "
                f"'observables' (got {observables})"
            )
        if not parity_even:
            errors.append(
                "'parity_mixed_blocks: true' requires at least one "
                f"parity-even observable (TT, EE, TE, BB; got {observables})"
            )
    b_observables = [o for o in observables if "B" in o]
    if b_observables and approximation not in APPROXIMATIONS_SUPPORTING_B:
        if observables == BB_ONLY_OBSERVABLES:
            # BB-only mode: any of ACC/NKA/INKA works here.
            pass
        else:
            supported = [m.value for m in APPROXIMATIONS_SUPPORTING_B]
            errors.append(
                f"observables {b_observables} need a B-mode block, but "
                f"'covariance_approximation: {approximation.value}' cannot "
                f"compute one (NKA and INKA are degenerate for BB by "
                f"construction); use {supported}, or set 'observables: "
                f"{BB_ONLY_OBSERVABLES}' alone for NKA/INKA's "
                "leakage-neglected BB-only escape hatch "
                "(docs/theory/bmode_kernels.md)"
            )
    return errors


def _check_polspice_postprocess_and_observables(
    observables: list[str], polspice_postprocess: bool
) -> list[str]:
    """
    With ``polspice_postprocess: true`` the decoupled ``EE`` output needs
    the pseudo ``BB`` block, since PolSpice decoupling mixes them
    (docs/theory/bmode_kernels.md,
    :meth:`~cmbcov.assembly.postprocess.CovariancePostProcessor.pseudo_to_spice_bmode`).
    An ``observables`` list with ``EE`` and a B observable other than ``BB``
    but not ``BB`` itself (e.g. ``[TT, EE, TE, EB]``) cannot supply that
    block, so it is rejected rather than silently dropping the mixing term.

    Returns
    -------
    list of str
        Error messages (empty if everything is consistent).
    """
    if not polspice_postprocess:
        return []
    if "EE" not in observables:
        return []
    missing_bb_source = [o for o in observables if "B" in o and o != "BB"]
    if missing_bb_source and "BB" not in observables:
        return [
            "'observables' has 'EE' and a B observable "
            f"({sorted(missing_bb_source)}) but not 'BB': "
            "'polspice_postprocess: true' needs the pseudo BB block to "
            "decouple EE (docs/theory/bmode_kernels.md); add "
            "'BB' to 'observables' or set 'polspice_postprocess: false'"
        ]
    return []


def acc_kernel_spectra_needed(stokes: list[str]) -> tuple[str, ...]:
    """
    The coupling-kernel channels
    (:data:`~cmbcov.approximations.acc.COUPLING_CHANNELS`) an ACC run over the
    T/E letters ``stokes`` loads, read off the Wick contractions of every
    covariance key (:meth:`~cmbcov.assembly.keys.CovKey.key_to_cross_kernel`, mapped
    with :meth:`~cmbcov.assembly.keys.SpecKey.kernel_stokekey`), in ``COUPLING_CHANNELS``
    order.
    """
    needed = set()
    for cov_key in CovKeys(list(stokes), ["f"], exclude_asymmetric_stokes=False).keys():
        for contraction in cov_key.key_to_cross_kernel():
            needed.update(spec.kernel_stokekey() for spec in contraction)
    return tuple(s for s in COUPLING_CHANNELS if s in needed)


def acc_kernel_channels_needed(
    observables: list[str], parity_mixed_blocks: bool = False
) -> tuple[str, ...]:
    """
    The coupling-kernel channels (not pairs) a run over ``observables``
    needs at minimum: the pairs of
    :func:`~cmbcov.approximations.acc.bmode_wick.required_kernel_pairs` flattened onto their
    channels, in :data:`~cmbcov.approximations.acc.COUPLING_CHANNELS` order.

    ``parity_odd_nonzero=False`` is assumed, as the spectra are not read at
    validation time; a run whose ``C^TB``/``C^EB`` turn out non-zero needs
    more pairs, which is caught at kernel-load time (the "computed for
    spectra ... only; recompute" error).
    """
    pairs = required_kernel_pairs(
        observables, parity_odd_nonzero=False, parity_mixed_blocks=parity_mixed_blocks
    )
    needed = {c for pair in pairs for c in pair}
    return tuple(s for s in COUPLING_CHANNELS if s in needed)


def normalise_param_aliases(params: Mapping[str, Any]) -> dict[str, Any]:
    """
    Rewrite accepted alternative spellings onto their canonical keys.

    Parameters
    ----------
    params : mapping
        Parameters exactly as read from the parameter file.

    Returns
    -------
    dict
        A new dictionary with every alias in :data:`PARAM_ALIASES` replaced by
        its canonical key.

    Raises
    ------
    ValueError
        If both spellings are present with different values.
    """
    normalised = dict(params)
    for alias, canonical in PARAM_ALIASES.items():
        if alias not in normalised:
            continue
        value = normalised.pop(alias)
        if canonical in normalised and normalised[canonical] != value:
            raise ValueError(
                f"Parameter file sets both '{canonical}' and its accepted "
                f"alternative spelling '{alias}' to different values "
                f"({normalised[canonical]!r} and {value!r}). Keep only one."
            )
        normalised[canonical] = value
    return normalised


class ParameterValidator:
    """
    Validates parameter files for covariance matrix computation.
    """

    # Define required parameters and their expected types.
    #
    # 'stokes' is not listed here: exactly one of 'stokes' or 'observables'
    # is required, checked (with a clearer, combined message) by
    # `_check_observables` -> `resolve_observables`, not by the generic
    # "required parameter is missing" loop below.
    REQUIRED_PARAMS = {
        "cov_path": str,
        "cov_name": str,
        "frequencies": list,
        "lmax": int,
        "lmin": int,
        "bins": list,
        "mask_name": str,
        "covariance_approximation": str,
        "mask_path": str,
    }

    # Define optional parameters with defaults
    OPTIONAL_PARAMS = {
        "beams": None,
        "pixwin": 8192,
        "fl": None,
        "hl": None,
        "cmb_spectrum": None,
        "nl": {},
        "foregrounds": None,
        # Units the tabulated spectrum inputs are written in. With the
        # default "Cl" the file columns are used as C_l with no conversion.
        # Each per-input override defaults to None, meaning "follow
        # spectrum_units".
        "spectrum_units": "Cl",
        "cmb_spectrum_units": None,
        "foregrounds_units": None,
        "nl_units": None,
        "inpainting_rescaling": 1.0,
        "apodizetype": 1,
        "apodizesigma": 30.0,
        "thetamax": 30.0,
        "Dl": False,
        "sum_assymetric_stokes": False,
        "compute_noise_only": False,
        "compute_signal_only": False,
        "add_tf_uncertainty": False,
        "save_raw_blocks": False,
        # Whether a block between a parity-even and a parity-odd observable
        # is included. Meaningless without TB/EB in observables/stokes.
        "parity_mixed_blocks": False,
        # Whether the pseudo-C_l covariance goes through the PolSpice
        # transform (Eq. 55) before scaling, debiasing and binning.
        "polspice_postprocess": True,
    }

    REBINNING_PARAMS = ["lmin", "lmax", "bins"]

    #: Removed parameter keys. A file that still carries one is refused, not
    #: silently ignored, since dropping it would change the numbers. Maps
    #: key -> explanation.
    REMOVED_PARAMS = {
        "nl_is_biased": (
            "'nl' is now always the noise power spectrum of the map as "
            "delivered to the estimator: it carries no beam, no pixel window "
            "and no transfer function, and it is never multiplied by the data "
            "model (MASTER, Hivon et al. 2002, Eqs. (15)-(16)). Beam "
            "deconvolution reaches the noise through the debiasing, which "
            "divides each leg by B1 B2 pix fl, so the noise enters the error "
            "bars as N_l / B_l^2. Delete the key."
        ),
    }

    #: Keys of the saved parameter record that describe the run rather than
    #: what was asked for, and so cannot make two runs incompatible:
    #: ``save_dir`` (the versioned directory this run resolved to) and
    #: ``git_hash`` (the package revision it ran at). Both are added by
    #: :meth:`ParameterManager.load_and_validate`, not read from the file.
    #: Everything else in the record comes from the parameter file.
    #: ``cov_path`` and ``cov_name`` are *not* here: the user asks for them.
    PROVENANCE_PARAMS = ["save_dir", "git_hash"]

    def __init__(self):
        """Initialize the parameter validator."""
        self.errors = []
        self.warnings = []
        self.logger = logger

    def validate(self, params: dict[str, Any]) -> bool:
        """
        Validate a parameter dictionary.

        Parameters
        ----------
        params : dict
            Parameter dictionary to validate

        Returns
        -------
        bool
            True if validation passes, False otherwise
        """
        self.logger.info("Starting parameter validation...")
        self.errors = []
        self.warnings = []

        # Refuse removed parameters before anything else reads a key
        self._check_removed_params(params)

        # Check required parameters
        self.logger.debug("Checking required parameters...")
        self._check_required_params(params)

        # Check parameter value consistency
        self.logger.debug("Checking parameter consistency...")
        self._check_parameter_consistency(params)

        # Check file paths exist
        self.logger.debug("Checking file paths...")
        self._check_file_paths(params)

        # Check 'stokes'/'observables' (B-mode configuration)
        self._check_observables(params)

        # Check the ACC precompute block, if any
        self._check_acc_precompute(params)

        # Check the map-filter block, if any
        self._check_map_filter(params)

        # Check the spectrum-units keys
        self._check_spectrum_units(params)

        # Log validation results
        if self.errors:
            self.logger.error(f"Validation failed with {len(self.errors)} errors")
        if self.warnings:
            self.logger.warning(
                f"Validation completed with {len(self.warnings)} warnings"
            )
        if not self.errors and not self.warnings:
            self.logger.info("Parameter validation passed successfully")

        return len(self.errors) == 0

    def _check_removed_params(self, params: dict[str, Any]) -> None:
        """
        Refuse any key in :attr:`REMOVED_PARAMS`.

        Each one is a validation error naming the key and what replaced it.
        """
        for param, explanation in self.REMOVED_PARAMS.items():
            if param in params:
                error_msg = f"Parameter '{param}' has been removed. {explanation}"
                self.logger.error(error_msg)
                self.errors.append(error_msg)

    def _check_required_params(self, params: dict[str, Any]) -> None:
        """Check that all required parameters are present."""
        self.logger.debug(f"Checking {len(self.REQUIRED_PARAMS)} required parameters")

        for param, expected_type in self.REQUIRED_PARAMS.items():
            if param not in params:
                error_msg = f"Required parameter '{param}' is missing"
                self.logger.error(error_msg)
                self.errors.append(error_msg)
            elif not isinstance(params[param], expected_type):
                error_msg = (
                    f"Parameter '{param}' should be of type {expected_type.__name__}, "
                    f"got {type(params[param]).__name__}"
                )
                self.logger.error(error_msg)
                self.errors.append(error_msg)
            else:
                self.logger.debug(f"Parameter '{param}' validated successfully")

        for param, value in params.items():
            # Special type checks for complex parameters
            if param == "frequencies":
                if not all(isinstance(f, str) for f in value):
                    self.errors.append("All frequencies must be strings")

            elif param == "stokes":
                valid_stokes = ["T", "E", "B"]
                if not all(s in valid_stokes for s in value):
                    self.errors.append(
                        f"Invalid Stokes parameters. Valid: {valid_stokes}"
                    )

            elif param == "bins":
                if not all(isinstance(b, list) and len(b) == 3 for b in value):
                    self.errors.append(
                        "Each bin specification should be [start, stop, step]"
                    )

            elif param == "lmax":
                if value <= 0:
                    self.errors.append("lmax must be positive")

            elif param == "lmin":
                if value < 0:
                    self.errors.append("lmin must be non-negative")
                if "lmax" in params and value >= params["lmax"]:
                    self.errors.append("lmin must be less than lmax")

            elif param in ("save_raw_blocks", "polspice_postprocess"):
                if not isinstance(value, bool):
                    self.errors.append(
                        f"{param} must be true or false, got "
                        f"{value!r} ({type(value).__name__})"
                    )

    def _check_parameter_consistency(self, params: dict[str, Any]) -> None:
        """Check internal consistency of parameters."""

        # Check frequency format
        if "frequencies" in params:
            for freq in params["frequencies"]:
                if not freq.endswith("GHz"):
                    self.warnings.append(f"Frequency '{freq}' doesn't end with 'GHz'")

        # Check bins don't exceed lmax
        if "bins" in params and "lmax" in params:
            max_bin = max(b[1] for b in params["bins"])
            if max_bin > params["lmax"]:
                self.errors.append(
                    f"Bin maximum ({max_bin}) exceeds lmax ({params['lmax']})"
                )

        # Check that mask_path exists if mask_name doesn't include full path
        if "mask_name" in params and "mask_path" in params:
            if not os.path.isabs(params["mask_name"]) and params["mask_path"] is None:
                self.warnings.append(
                    "Relative mask_name provided but no mask_path specified"
                )

        # dmax vs the binning: a bandpower variance sums multipole-pair
        # separations 0..w-1 (w = widest bin width), and the covariance of
        # two adjacent bandpowers sums separations 1..2w-1. ACC computes
        # only |l - l'| < dmax and zeroes the rest, so a too-small dmax
        # silently drops pairs instead of raising -- warn instead of erroring,
        # since a smaller dmax is a legitimate precompute-cost trade-off.
        approximation = params.get("covariance_approximation")
        dmax = params.get("dmax")
        bins = params.get("bins")
        if (
            str(approximation).lower() == CovarianceMethod.ACC.value
            and dmax is not None
            and isinstance(bins, list)
            and bins
            and all(isinstance(b, list) and len(b) == 3 for b in bins)
        ):
            w = max(b[2] for b in bins)
            if dmax < w:
                self.warnings.append(
                    f"dmax ({dmax}) is below the widest bin width ({w}): "
                    f"multipole pairs separated by {dmax}..{w - 1} within a "
                    "bin are dropped, so every bandpower variance is "
                    f"underestimated. Use dmax >= {w} for the error bars, "
                    f"dmax >= {2 * w} for adjacent-bandpower correlations."
                )
            elif dmax < 2 * w:
                self.warnings.append(
                    f"dmax ({dmax}) is at least the widest bin width ({w}), "
                    "so bandpower variances are complete, but the "
                    "correlation between adjacent bandpowers is "
                    f"underestimated: pairs separated by {dmax}..{2 * w - 1} "
                    f"across neighbouring bins are dropped. Use dmax >= "
                    f"{2 * w} to capture them."
                )

    def _check_observables(self, params: dict[str, Any]) -> None:
        """
        Validate ``stokes``/``observables`` and ``parity_mixed_blocks``:
        exactly one of ``stokes``/``observables``, the shorthand and alias
        rules, and that ``parity_mixed_blocks`` and B observables are only
        asked of a covariance approximation and configuration that support
        them. Every rejection names the offending keys.
        """
        try:
            _, observables = resolve_observables(params)
        except ValueError as e:
            self.errors.append(str(e))
            return

        parity_mixed_blocks = params.get("parity_mixed_blocks", False)
        if not isinstance(parity_mixed_blocks, bool):
            self.errors.append(
                "'parity_mixed_blocks' must be true or false, got "
                f"{parity_mixed_blocks!r}"
            )
            parity_mixed_blocks = bool(parity_mixed_blocks)

        approximation = params.get("covariance_approximation")
        try:
            method = CovarianceMethod[str(approximation).upper()]
        except KeyError:
            # Reported by _check_required_params / PipelineConfig.from_params;
            # nothing more to check against an unknown approximation.
            return

        self.errors.extend(
            _check_parity_mixed_and_approximation(
                observables, parity_mixed_blocks, method
            )
        )

        polspice_postprocess = params.get("polspice_postprocess", True)
        if not isinstance(polspice_postprocess, bool):
            polspice_postprocess = bool(polspice_postprocess)
        self.errors.extend(
            _check_polspice_postprocess_and_observables(
                observables, polspice_postprocess
            )
        )

    def _check_file_paths(self, params: dict[str, Any]) -> None:
        """Check that specified file paths exist."""
        self.logger.debug("Checking file path parameters...")
        file_params = ["beams", "cmb_spectrum", "pixwin"]

        for param in file_params:
            if param in params and isinstance(params[param], str):
                self.logger.debug(f"Checking file parameter '{param}': {params[param]}")
                if not os.path.exists(params[param]):
                    warning_msg = f"File not found: {params[param]}"
                    self.logger.warning(warning_msg)
                    self.warnings.append(warning_msg)
                else:
                    self.logger.debug(f"File parameter '{param}' exists")

        # Check mask file
        if "mask_name" in params:
            mask_file = params["mask_name"]
            if "mask_path" in params and params["mask_path"]:
                mask_file = os.path.join(params["mask_path"], mask_file)

            self.logger.debug(f"Checking mask file: {mask_file}")
            if not os.path.exists(mask_file):
                warning_msg = f"Mask file not found: {mask_file}"
                self.logger.warning(warning_msg)
                self.warnings.append(warning_msg)
            else:
                self.logger.debug("Mask file exists")

        # Check foreground files
        if "foregrounds" in params and params["foregrounds"]:
            # This is a format string, so we can't check directly
            # but we can warn about the format
            if "{}" not in str(params["foregrounds"]):
                warning_msg = "Foregrounds parameter should contain '{}' placeholders"
                self.logger.warning(warning_msg)
                self.warnings.append(warning_msg)
            else:
                self.logger.debug("Foregrounds parameter format is valid")

    def _check_map_filter(self, params: dict[str, Any]) -> None:
        """
        Validate the optional ``map_filter`` block (:class:`MapFilterConfig`).

        Errors: the block is not a mapping; an unknown key; ``type`` missing
        or not ``fourier_highpass``; ``lx`` missing or not a positive number;
        ``shape`` not ``sharp`` or ``exp``; ``power`` not a positive number;
        ``nprobe`` not an integer >= 1; ``seed`` not an integer;
        ``node_min`` not an integer >= 2; ``node_ratio`` not a number > 1;
        ``dmax`` not null or an integer >= 0; ``lw`` not null or a positive
        integer; ``spectrum_aware`` not a boolean; ``near_cut_step`` not
        null or a number >= 0; ``nprobe_near_cut`` not null or an integer
        >= 1; ``lowpass`` not null or a mapping of :class:`LowpassConfig`
        keys with ``lx`` a finite number above ``lx`` of the block, ``shape``
        ``exp`` or ``sharp``, ``power`` a finite positive number.

        A run with B-mode blocks is not refused: the correction covers the
        T/E blocks only and the run warns that the blocks with a B leg are
        uncorrected.
        """
        if MAP_FILTER_KEY not in params:
            return
        block = params[MAP_FILTER_KEY]
        where = MAP_FILTER_KEY
        if not isinstance(block, dict):
            self.errors.append(
                f"'{where}' must be a mapping of {list(MapFilterConfig.KEYS)}, "
                f"got {type(block).__name__}"
            )
            return

        unknown = sorted(set(block) - set(MapFilterConfig.KEYS))
        if unknown:
            self.errors.append(
                f"Unknown key(s) {unknown} in '{where}'; expected a subset of "
                f"{list(MapFilterConfig.KEYS)}"
            )

        kind = block.get("type")
        if kind not in MapFilterConfig.TYPES:
            self.errors.append(
                f"'{where}.type' is required and must be one of "
                f"{list(MapFilterConfig.TYPES)}, got {kind!r}"
            )

        if "lx" not in block:
            self.errors.append(f"'{where}.lx' is required (the filter cut)")
        elif not _is_number(block["lx"]) or not block["lx"] > 0:
            self.errors.append(
                f"'{where}.lx' must be a positive number, got {block['lx']!r}"
            )

        shape = block.get("shape", "sharp")
        if shape not in MapFilterConfig.SHAPES:
            self.errors.append(
                f"'{where}.shape' must be one of {list(MapFilterConfig.SHAPES)}, "
                f"got {shape!r}"
            )

        power = block.get("power", 6.0)
        if not _is_number(power) or not power > 0:
            self.errors.append(
                f"'{where}.power' must be a positive number, got {power!r}"
            )

        for key, low in (("nprobe", 1), ("node_min", 2)):
            value = block.get(key, low)
            if not _is_int(value) or value < low:
                self.errors.append(
                    f"'{where}.{key}' must be an integer >= {low}, got {value!r}"
                )

        seed = block.get("seed", 0)
        if not _is_int(seed):
            self.errors.append(f"'{where}.seed' must be an integer, got {seed!r}")

        ratio = block.get("node_ratio", 1.25)
        if not _is_number(ratio) or not ratio > 1:
            self.errors.append(
                f"'{where}.node_ratio' must be a number > 1, got {ratio!r}"
            )

        dmax = block.get("dmax")
        if dmax is not None and (not _is_int(dmax) or dmax < 0):
            self.errors.append(
                f"'{where}.dmax' must be null or an integer >= 0, got {dmax!r}"
            )

        lw = block.get("lw")
        if lw is not None and (not _is_int(lw) or lw <= 0):
            self.errors.append(
                f"'{where}.lw' must be null or a positive integer, got {lw!r}"
            )

        spectrum_aware = block.get("spectrum_aware", False)
        if not isinstance(spectrum_aware, bool):
            self.errors.append(
                f"'{where}.spectrum_aware' must be true or false, got "
                f"{spectrum_aware!r}"
            )

        step = block.get("near_cut_step")
        if step is not None and (
            not _is_number(step) or not math.isfinite(step) or step < 0
        ):
            self.errors.append(
                f"'{where}.near_cut_step' must be null or a number >= 0, got "
                f"{step!r}"
            )

        near = block.get("nprobe_near_cut")
        if near is not None and (not _is_int(near) or near < 1):
            self.errors.append(
                f"'{where}.nprobe_near_cut' must be null or an integer >= 1, got "
                f"{near!r}"
            )

        self._check_lowpass(block, where)

    def _check_lowpass(self, block: dict[str, Any], where: str) -> None:
        """Validate the optional ``lowpass`` sub-block of ``block``
        (:class:`LowpassConfig`); ``where`` names the parent block."""
        lowpass = block.get("lowpass")
        if lowpass is None:
            return
        where = f"{where}.lowpass"
        if not isinstance(lowpass, dict):
            self.errors.append(
                f"'{where}' must be null or a mapping of "
                f"{list(LowpassConfig.KEYS)}, got {type(lowpass).__name__}"
            )
            return
        unknown = sorted(set(lowpass) - set(LowpassConfig.KEYS))
        if unknown:
            self.errors.append(
                f"Unknown key(s) {unknown} in '{where}'; expected a subset of "
                f"{list(LowpassConfig.KEYS)}"
            )

        if "lx" not in lowpass:
            self.errors.append(f"'{where}.lx' is required (the low-pass cut)")
        else:
            lx = lowpass["lx"]
            if not _is_number(lx) or not math.isfinite(lx) or not lx > 0:
                self.errors.append(
                    f"'{where}.lx' must be a finite positive number, got {lx!r}"
                )
            else:
                high = block.get("lx")
                if _is_number(high) and math.isfinite(high) and not lx > high:
                    self.errors.append(
                        f"'{where}.lx' ({lx!r}) must exceed the high-pass "
                        f"'map_filter.lx' ({high!r})"
                    )

        shape = lowpass.get("shape", "exp")
        if shape not in LowpassConfig.SHAPES:
            self.errors.append(
                f"'{where}.shape' must be one of {list(LowpassConfig.SHAPES)}, "
                f"got {shape!r}"
            )

        power = lowpass.get("power", 6.0)
        if not _is_number(power) or not math.isfinite(power) or not power > 0:
            self.errors.append(
                f"'{where}.power' must be a finite positive number, got {power!r}"
            )

    def _check_acc_precompute(self, params: dict[str, Any]) -> None:
        """
        Validate the optional ``acc_precompute`` block
        (:class:`AccPrecomputeConfig`).

        Errors: the block is not a mapping; an unknown key; ``nside`` or
        ``lw`` not a positive integer; ``nside`` not a power of two with
        ``grid: healpix`` (the mask is degraded to it); ``grid`` not
        ``healpix`` or ``gl``; ``lw`` with ``grid: healpix`` (it would be
        ignored); ``spectra`` not a non-empty list drawn from
        ``COUPLING_CHANNELS`` (observable names accepted), or missing a
        kernel the run's ``observables`` need; ``max_memory_gb`` not a
        positive number; ``scratch_dir`` not a non-empty string.

        Warning, not error: the block with a ``covariance_approximation``
        other than ``acc``. Nothing reads it then, and the same file is
        legitimately switched between methods; ``precompute-acc`` itself
        refuses such a file.
        """
        if ACC_PRECOMPUTE_KEY not in params:
            return
        block = params[ACC_PRECOMPUTE_KEY]
        if block is None:
            block = {}
        if not isinstance(block, dict):
            self.errors.append(
                f"'{ACC_PRECOMPUTE_KEY}' must be a mapping of "
                f"{list(AccPrecomputeConfig.KEYS)}, got {type(block).__name__}"
            )
            return

        approximation = params.get("covariance_approximation")
        if str(approximation).lower() != CovarianceMethod.ACC.value:
            self.warnings.append(
                f"'{ACC_PRECOMPUTE_KEY}' is ignored: covariance_approximation "
                f"is {approximation!r}, not 'acc'"
            )

        where = ACC_PRECOMPUTE_KEY
        unknown = sorted(set(block) - set(AccPrecomputeConfig.KEYS))
        if unknown:
            self.errors.append(
                f"Unknown key(s) {unknown} in '{where}'; expected a subset of "
                f"{list(AccPrecomputeConfig.KEYS)}"
            )

        grid = block.get("grid", "gl")
        if grid not in AccPrecomputeConfig.GRIDS:
            self.errors.append(
                f"'{where}.grid' must be one of {list(AccPrecomputeConfig.GRIDS)}, "
                f"got {grid!r}"
            )

        nside = block.get("nside")
        if nside is not None:
            if not _is_int(nside) or nside <= 0:
                self.errors.append(
                    f"'{where}.nside' must be a positive integer, got {nside!r}"
                )
            elif grid == "healpix" and nside & (nside - 1):
                self.errors.append(
                    f"'{where}.nside' must be a power of two with grid "
                    f"'healpix', got {nside}"
                )

        lw = block.get("lw")
        if lw is not None:
            if not _is_int(lw) or lw <= 0:
                self.errors.append(
                    f"'{where}.lw' must be a positive integer, got {lw!r}"
                )
            if grid != "gl":
                self.errors.append(
                    f"'{where}.lw' is only used with grid 'gl' (grid is {grid!r})"
                )

        memory = block.get("max_memory_gb")
        if memory is not None and (
            isinstance(memory, bool)
            or not isinstance(memory, (int, float))
            or not memory > 0
        ):
            self.errors.append(
                f"'{where}.max_memory_gb' must be a positive number, got {memory!r}"
            )

        scratch = block.get("scratch_dir")
        if scratch is not None and (not isinstance(scratch, str) or not scratch):
            self.errors.append(
                f"'{where}.scratch_dir' must be a non-empty string (a directory), "
                f"got {scratch!r}"
            )

        spectra = block.get("spectra")
        if spectra is not None:
            if (
                not isinstance(spectra, list)
                or not spectra
                or not all(isinstance(s, str) for s in spectra)
            ):
                self.errors.append(
                    f"'{where}.spectra' must be a non-empty list drawn from "
                    f"{list(COUPLING_CHANNELS)} (old names accepted), got {spectra!r}"
                )
                return
            normalised = {}
            unknown_spectra = []
            for s in spectra:
                try:
                    normalised[s] = normalise_channel(s)
                except ValueError:
                    unknown_spectra.append(s)
            if unknown_spectra:
                self.errors.append(
                    f"'{where}.spectra' has unknown spectra {sorted(unknown_spectra)}; "
                    f"expected a subset of {list(COUPLING_CHANNELS)} (old names accepted)"
                )
                return
            normalised_spectra = set(normalised.values())
            try:
                _, observables = resolve_observables(params)
            except ValueError:
                # Already reported by _check_observables; nothing further to
                # check here without a resolved observables list.
                return
            parity_mixed_blocks = bool(params.get("parity_mixed_blocks", False))
            missing = [
                s
                for s in acc_kernel_channels_needed(observables, parity_mixed_blocks)
                if s not in normalised_spectra
            ]
            if missing:
                self.errors.append(
                    f"'{where}.spectra' {spectra} lacks {missing}, which an "
                    f"ACC run over observables {observables} loads"
                )

    def _check_spectrum_units(self, params: dict[str, Any]) -> None:
        """
        Validate ``spectrum_units`` and the per-input ``<input>_units``
        overrides (:data:`SPECTRUM_UNITS_INPUTS`).

        Errors: a value that is not ``Cl`` or ``Dl``
        (:func:`canonical_spectrum_units`); ``Dl`` asked of an input that is
        not a tabulated file -- a constant ``cmb_spectrum`` (float or dict)
        or white-noise levels in uK*arcmin -- which are C_l by construction
        and cannot be converted. This is an error, not a silent no-op, since
        an ignored units key would make a covariance wrong by
        ``(l(l+1)/2pi)^2``; say ``<input>_units: Cl`` to combine a D_l file
        with a constant input.
        """
        if SPECTRUM_UNITS_KEY in params:
            try:
                canonical_spectrum_units(params[SPECTRUM_UNITS_KEY], SPECTRUM_UNITS_KEY)
            except ValueError as error:
                self.errors.append(str(error))

        for input_name in SPECTRUM_UNITS_INPUTS:
            key = spectrum_units_key(input_name)
            value = params.get(key)
            if value is not None:
                try:
                    canonical_spectrum_units(value, key)
                except ValueError as error:
                    self.errors.append(str(error))

            # "Dl" on a non-tabulated input has nothing to convert.
            try:
                units = resolve_spectrum_units(params, input_name)
            except ValueError:
                continue  # already reported above
            supplied = params.get(input_name)
            if input_name == "nl" and not supplied:
                # `nl: {}` (the default) or an absent key is "no noise
                # supplied"; `noise` is its legacy spelling.
                supplied = params.get("noise")
            if units == "Dl" and supplied and not isinstance(supplied, str):
                self.errors.append(
                    f"'{key}' (or '{SPECTRUM_UNITS_KEY}') is 'Dl' but "
                    f"'{input_name}' is not a spectrum file: it is "
                    f"{type(supplied).__name__} "
                    f"({'white-noise levels in uK*arcmin' if input_name == 'nl' else 'a constant'}"
                    "), which is C_l by construction. Set "
                    f"'{key}: Cl' to say so explicitly."
                )

    def get_validation_report(self) -> str:
        """
        Get a formatted validation report.

        Returns
        -------
        str
            Formatted report of errors and warnings
        """
        report = []

        if self.errors:
            report.append("ERRORS:")
            for error in self.errors:
                report.append(f"  - {error}")

        if self.warnings:
            report.append("WARNINGS:")
            for warning in self.warnings:
                report.append(f"  - {warning}")

        if not self.errors and not self.warnings:
            report.append("Parameter validation passed successfully.")

        return "\n".join(report)


@dataclass(frozen=True)
class PipelineConfig(Mapping):
    """
    The validated parameters of one covariance run, built once.

    The single source of truth for everything the pipeline reads from a
    parameter file. It is frozen.

    The multipole/kernel half of the parameters is stored once, in
    :attr:`covariance` (the :class:`~cmbcov.assembly.covariance.CovarianceConfig` that
    :class:`~cmbcov.assembly.covariance.Cov` consumes), and exposed through read-only
    properties (:attr:`lmax`, :attr:`Dl`, :attr:`sum_asymmetric_stokes`, ...).

    The object is also a read-only :class:`~collections.abc.Mapping` over
    :attr:`params`, so parameter-file keys can still be read by name
    (``config["lmax"]``, ``config.get("beams")``).

    Attributes
    ----------
    covariance : CovarianceConfig
        The covariance-computation half of the parameters (method, lmax, lmin,
        Dl, sum_asymmetric_stokes, apodization, dmax, centralell,
        save_raw_blocks, polspice_postprocess). Its
        ``verbose`` and ``debiasing_dict`` fields are runtime state rather
        than parameter-file values and are filled in by the generator.
    save_dir : str or None
        Versioned output directory, resolved at construction. ``None``
        only for a configuration loaded without setting up an output
        directory (``ParameterManager.load_and_validate(setup_output=False)``,
        as ``precompute-acc`` does).
    git_hash : str
        Short git hash of the package at run time, resolved at construction.
    acc_precompute : AccPrecomputeConfig or None
        The ``acc_precompute`` block, or ``None`` if the file has none.
    map_filter : MapFilterConfig or None
        The ``map_filter`` block, or ``None`` if the file has none.
    params : Mapping
        Read-only record of the effective parameters: the file's values with
        alias spellings normalised, defaults applied, and ``save_dir`` and
        ``git_hash`` added. It is what is written next to the results, not
        what the code reads.
    """

    # --- covariance-computation parameters (stored once, in CovarianceConfig)
    covariance: CovarianceConfig

    # --- output and workflow
    cov_path: str
    cov_name: str
    frequencies: list[str]
    #: Stokes letters for the T/E-only :class:`~cmbcov.assembly.keys.CovKeys` alphabet
    #: path (:func:`resolve_observables`); only used when :attr:`observables`
    #: has no B.
    stokes: list[str]
    #: Canonical observables list, always populated -- from the file's
    #: ``observables`` key, or derived from ``stokes`` shorthand
    #: (:func:`resolve_observables`).
    observables: list[str]
    #: Whether a block between a parity-even and a parity-odd observable is
    #: included. Default False; meaningless without TB/EB in ``observables``.
    parity_mixed_blocks: bool
    bins: list[list[int]]
    mask_name: str
    mask_path: str | None
    save_dir: str | None
    git_hash: str

    # --- data-model inputs, consumed by SpectraLoader
    #
    # No field below carries an inline default: their defaults live in
    # ParameterValidator.OPTIONAL_PARAMS only.
    beams: Any
    pixwin: Any
    fl: Any
    hl: Any
    cmb_spectrum: Any
    foregrounds: Any
    nl: Any
    inpainting_rescaling: Any
    add_tf_uncertainty: bool

    # Units of the tabulated spectrum inputs: the global default and the
    # per-input overrides (``None`` = follow the global one). Read through
    # :meth:`spectrum_units_of`, never directly, so one rule resolves them.
    spectrum_units: str
    cmb_spectrum_units: str | None
    foregrounds_units: str | None
    nl_units: str | None

    # --- workflow switches
    compute_noise_only: bool
    compute_signal_only: bool

    # --- read by the pipeline but absent from OPTIONAL_PARAMS, so their
    #     default lives here.
    #: Legacy alias for ``nl``, resolved by SpectraLoader.
    noise: Any = None
    post_process_correction: Any = None
    #: The ``acc_precompute`` block; ``None`` when the file has none.
    acc_precompute: AccPrecomputeConfig | None = None
    #: The ``map_filter`` block; ``None`` when the file has none.
    map_filter: MapFilterConfig | None = None

    #: Read-only record of the effective parameter file (see the class
    #: docstring).
    params: Mapping[str, Any] = field(default_factory=lambda: EMPTY_PARAMS)

    # ------------------------------------------------------------------
    # Delegating properties: the covariance half lives in `covariance`
    # ------------------------------------------------------------------

    @property
    def method(self) -> CovarianceMethod:
        """Covariance approximation, as an enum member."""
        return self.covariance.method

    @property
    def covariance_approximation(self) -> str:
        """Covariance approximation, as spelled in the parameter file."""
        return self.covariance.method.value

    @property
    def lmax(self) -> int:
        return self.covariance.lmax

    @property
    def lmin(self) -> int:
        return self.covariance.lmin

    @property
    def Dl(self) -> bool:  # noqa: N802 - the parameter-file key and the
        # CovarianceConfig field are both spelled `Dl`.
        return self.covariance.Dl

    @property
    def sum_asymmetric_stokes(self) -> bool:
        """
        Whether TE and ET are folded together.

        The parameter-file key is ``sum_assymetric_stokes`` (double s); the
        correct spelling is accepted as well. Both reach this one field.
        """
        return self.covariance.sum_asymmetric_stokes

    @property
    def apodizetype(self) -> int | None:
        return self.covariance.apodizetype

    @property
    def apodizesigma(self) -> float | None:
        return self.covariance.apodizesigma

    @property
    def thetamax(self) -> float | None:
        return self.covariance.thetamax

    @property
    def dmax(self) -> int | None:
        return self.covariance.dmax

    @property
    def centralell(self) -> int | None:
        return self.covariance.centralell

    @property
    def save_raw_blocks(self) -> bool:
        """
        Whether raw covariance blocks are cached, with manifests, in
        :attr:`save_dir` (see ``CovarianceConfig.save_raw_blocks``).
        """
        return self.covariance.save_raw_blocks

    def spectrum_units_of(self, input_name: str) -> str:
        """
        The units a tabulated spectrum input is written in: ``"Cl"`` or
        ``"Dl"``.

        Parameters
        ----------
        input_name : str
            One of :data:`SPECTRUM_UNITS_INPUTS` (``cmb_spectrum``,
            ``foregrounds``, ``nl``). Anything else raises: every other
            tabulated input (beams, pixel window, ``fl``/``hl``,
            ``post_process_correction``) is dimensionless and is never
            converted.

        Returns
        -------
        str
            The per-input override ``<input>_units`` if set, else the global
            ``spectrum_units`` (default ``"Cl"``).
        """
        if input_name not in SPECTRUM_UNITS_INPUTS:
            raise ValueError(
                f"{input_name!r} is not a spectrum input; expected one of "
                f"{list(SPECTRUM_UNITS_INPUTS)}. Beams, the pixel window, "
                "transfer functions and post-processing corrections are "
                "dimensionless and carry no units key."
            )
        return _resolve_spectrum_units(
            getattr(self, spectrum_units_key(input_name)),
            self.spectrum_units,
            input_name,
        )

    @property
    def acc_kernel_dir(self) -> str:
        """
        Directory under which ACC coupling kernels live (in
        ``covariance_coupling/``): ``cov_path`` itself, not the versioned
        :attr:`save_dir` below it.

        The kernels depend only on the mask and ``centralell``, while every
        run gets a fresh ``save_dir`` (``cov_path/v0``, ``v1``, ...).
        ``precompute-acc`` writes here and every ``compute-covariance`` run
        of the same file reads from here; the manifest check refuses the
        cache if the mask or ``centralell`` has changed since.
        """
        return self.cov_path

    # ------------------------------------------------------------------
    # Read-only mapping interface over the effective parameter record
    # ------------------------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        return self.params[key]

    def __iter__(self):
        return iter(self.params)

    def __len__(self) -> int:
        return len(self.params)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"PipelineConfig(method={self.covariance.method.value!r}, "
            f"lmax={self.lmax}, lmin={self.lmin}, "
            f"frequencies={self.frequencies}, stokes={self.stokes}, "
            f"save_dir={self.save_dir!r})"
        )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "PipelineConfig":
        """
        Build the configuration from an effective parameter mapping.

        Parameters
        ----------
        params : mapping
            Parameters after alias normalisation and defaults, including
            ``save_dir`` and ``git_hash``, as assembled by
            :meth:`ParameterManager.load_and_validate`.

        Returns
        -------
        PipelineConfig

        Raises
        ------
        ValueError
            If a required parameter is missing, or if
            ``covariance_approximation`` names no known method. Only
            parameters with a default in
            :attr:`ParameterValidator.OPTIONAL_PARAMS` get one.
        """

        def required(name: str) -> Any:
            if name not in params:
                raise ValueError(
                    f"Required parameter '{name}' is missing; cannot build the "
                    "run configuration."
                )
            return params[name]

        def optional(name: str) -> Any:
            """Value from the file, else the one documented default."""
            if name in params:
                return params[name]
            if name in ParameterValidator.OPTIONAL_PARAMS:
                return ParameterValidator.OPTIONAL_PARAMS[name]
            raise ValueError(
                f"Parameter '{name}' has no value and no documented default."
            )

        approximation = required("covariance_approximation")
        try:
            method = CovarianceMethod[str(approximation).upper()]
        except KeyError:
            raise ValueError(
                f"Unknown covariance_approximation {approximation!r}; expected "
                f"one of {[member.value for member in CovarianceMethod]}"
            ) from None

        covariance = CovarianceConfig(
            method=method,
            lmax=required("lmax"),
            lmin=required("lmin"),
            Dl=optional("Dl"),
            # The file key is `sum_assymetric_stokes`; the correct spelling is
            # normalised onto it before this point.
            sum_asymmetric_stokes=optional("sum_assymetric_stokes"),
            apodizetype=optional("apodizetype"),
            apodizesigma=optional("apodizesigma"),
            thetamax=optional("thetamax"),
            # Only meaningful for ACC; CovarianceConfig.validate() rejects a
            # missing value for that method rather than inventing one.
            dmax=params.get("dmax", None),
            centralell=params.get("centralell", None),
            save_raw_blocks=optional("save_raw_blocks"),
            polspice_postprocess=optional("polspice_postprocess"),
        )

        stokes, observables = resolve_observables(params)

        return cls(
            covariance=covariance,
            cov_path=required("cov_path"),
            cov_name=required("cov_name"),
            frequencies=required("frequencies"),
            stokes=stokes,
            observables=observables,
            parity_mixed_blocks=optional("parity_mixed_blocks"),
            bins=required("bins"),
            mask_name=required("mask_name"),
            mask_path=required("mask_path"),
            save_dir=required("save_dir"),
            git_hash=required("git_hash"),
            beams=optional("beams"),
            pixwin=optional("pixwin"),
            fl=optional("fl"),
            hl=optional("hl"),
            cmb_spectrum=optional("cmb_spectrum"),
            foregrounds=optional("foregrounds"),
            nl=optional("nl"),
            spectrum_units=canonical_spectrum_units(
                optional(SPECTRUM_UNITS_KEY), SPECTRUM_UNITS_KEY
            ),
            cmb_spectrum_units=optional("cmb_spectrum_units"),
            foregrounds_units=optional("foregrounds_units"),
            nl_units=optional("nl_units"),
            # Legacy alias for `nl`, resolved by SpectraLoader.
            noise=params.get("noise", None),
            # Read by SpectraLoader but absent from OPTIONAL_PARAMS.
            post_process_correction=params.get("post_process_correction", None),
            acc_precompute=(
                AccPrecomputeConfig.from_block(params[ACC_PRECOMPUTE_KEY])
                if ACC_PRECOMPUTE_KEY in params
                else None
            ),
            map_filter=(
                MapFilterConfig.from_block(params[MAP_FILTER_KEY])
                if MAP_FILTER_KEY in params
                else None
            ),
            inpainting_rescaling=optional("inpainting_rescaling"),
            add_tf_uncertainty=optional("add_tf_uncertainty"),
            compute_noise_only=optional("compute_noise_only"),
            compute_signal_only=optional("compute_signal_only"),
            params=MappingProxyType(dict(params)),
        )


class ParameterManager:
    """
    Manages parameter loading, validation, and compatibility checking.

    :meth:`load_and_validate` builds the :class:`PipelineConfig`. After it
    returns, :attr:`config` is the run's configuration and :attr:`params` is
    a read-only record of the file.
    """

    def __init__(
        self,
        parameter_file: str,
        overwrite: int | None = None,
    ):
        """
        Initialize the parameter manager.

        Parameters
        ----------
        parameter_file : str
            Path to the parameter file
        overwrite : int, optional
            Existing output version number to write into. If ``None`` a new
            version directory is created. Nothing of the previous run is
            reused, but the parameter file already in that directory must
            match this one in everything but the binning and the provenance
            keys (see :meth:`_check_rebinning`).
        """
        self.parameter_file = parameter_file
        self.config: PipelineConfig | None = None
        self.overwrite = overwrite
        self.logger = logger
        self.validator = ParameterValidator()

    @property
    def params(self) -> Mapping[str, Any]:
        """
        Read-only record of the effective parameters.

        Not the source of truth: :attr:`config` is. The same mapping as
        ``self.config.params`` (the file's values with alias spellings
        normalised, defaults applied, and ``save_dir``/``git_hash`` added); it
        is written beside the results. Read-only.
        """
        return self.config.params if self.config is not None else EMPTY_PARAMS

    @property
    def save_dir(self) -> str | None:
        """Versioned output directory, or None before validation."""
        return self.config.save_dir if self.config is not None else None

    def load_and_validate(self, setup_output: bool = True) -> PipelineConfig:
        """
        Load parameters from file, validate them, and build the run config.

        This is where the :class:`PipelineConfig` is created.

        Parameters
        ----------
        setup_output : bool, default True
            Resolve (and create) the versioned output directory and check
            compatibility with a previous version in it. ``False`` skips
            both and leaves ``save_dir`` as ``None``: for a step that reads
            the file but produces no covariance, such as ``precompute-acc``,
            which would otherwise create an empty version directory that
            the next run then silently reuses.

        Returns
        -------
        PipelineConfig
            The validated configuration. It is also a read-only mapping over
            the effective parameters, so ``config["lmax"]`` and
            ``config.get("beams")`` work like a plain parameter dictionary.

        Raises
        ------
        FileNotFoundError
            If parameter file doesn't exist
        ValueError
            If validation fails
        """
        self.logger.info(f"Loading parameter file: {self.parameter_file}")

        # Load parameters
        if not os.path.exists(self.parameter_file):
            error_msg = f"Parameter file not found: {self.parameter_file}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            with open(self.parameter_file) as f:
                raw_params = yaml.load(f, yaml.CLoader)
            self.logger.info("Parameter file loaded successfully")
        except Exception as e:
            error_msg = f"Error loading parameter file: {e}"
            self.logger.error(error_msg)
            raise ValueError(error_msg) from e

        # Normalise alternative spellings first, so the defaults below cannot
        # fill in the canonical key behind them.
        effective = normalise_param_aliases(raw_params or {})

        # Apply defaults for missing optional parameters
        self.logger.debug("Applying default values for optional parameters")
        self._apply_defaults(effective)

        # Validate
        self.logger.info("Starting parameter validation")
        if not self.validator.validate(effective):
            validation_report = self.validator.get_validation_report()
            self.logger.error("Parameter validation failed")
            raise ValueError(f"Parameter validation failed:\n{validation_report}")

        # Print warnings if any
        if self.validator.warnings:
            self.logger.warning(
                f"Validation completed with {len(self.validator.warnings)} warnings"
            )
            for warning in self.validator.warnings:
                self.logger.warning(f"  - {warning}")

        if not setup_output:
            effective["save_dir"] = None
            effective["git_hash"] = get_git_revision_short_hash()
            self.config = PipelineConfig.from_params(effective)
            self.logger.info("Parameters validated (no output directory set up)")
            return self.config

        # Setup directories and versioning
        self.logger.info("Setting up output directories and versioning")
        base_dir, git_hash_code = self._setup_directories(effective)
        effective["save_dir"] = base_dir
        effective["git_hash"] = git_hash_code

        # `effective` becomes the config's read-only record.
        self.config = PipelineConfig.from_params(effective)

        # Check rebinning compatibility if needed
        self.logger.info("Checking for existing parameter compatibility")
        self.save_path = os.path.join(
            self.config.save_dir, os.path.basename(self.parameter_file)
        )
        # Compared against whatever parameter file the directory holds: the
        # file may have been renamed.
        self._existing_parameter_file = self._find_saved_parameter_file()

        if self._existing_parameter_file is not None:
            with open(self._existing_parameter_file) as f:
                existing_params = yaml.load(f, yaml.CLoader)
            incompatible = self._incompatible_parameters(existing_params)
            if incompatible:
                self.logger.error("Parameters are not compatible with existing version")
                raise ValueError(
                    "Parameters are not compatible with existing version "
                    f"{self._existing_parameter_file!r}: "
                    f"{', '.join(incompatible)} differ(s). "
                    "The binning (lmin, lmax, bins) and the provenance keys "
                    f"({', '.join(ParameterValidator.PROVENANCE_PARAMS)}) may "
                    "differ; everything else must match, since --overwrite "
                    "replaces that directory's results in place."
                )

        self.logger.info("Parameter loading and validation completed successfully")
        return self.config

    def _find_saved_parameter_file(self) -> str | None:
        """
        The parameter file a previous run saved in the version directory.

        That is the ``.yml``/``.yaml`` file there, or a file named like this
        run's parameter file. None for a new (empty) directory. The run
        records ``params_resolved.yml`` and ``precompute_params_resolved.yml``
        (:data:`~cmbcov.config.run_record.RECORD_NAMES`) are not parameter
        files and are not counted.

        Raises
        ------
        ValueError
            If the directory holds more than one: which set of parameters
            its results belong to is then ambiguous.
        """
        save_dir = self.config.save_dir
        found = {
            os.path.join(save_dir, name)
            for name in os.listdir(save_dir)
            if name.endswith((".yml", ".yaml")) and name not in RECORD_NAMES
        }
        if os.path.exists(self.save_path):
            found.add(self.save_path)
        if len(found) > 1:
            raise ValueError(
                f"Version directory {save_dir!r} holds more than one parameter "
                f"file ({', '.join(sorted(map(os.path.basename, found)))}), so "
                "it is ambiguous which parameters its results belong to. Keep "
                "the one that matches them and remove the others."
            )
        return found.pop() if found else None

    def _apply_defaults(self, params: dict[str, Any]) -> None:
        """Apply default values for missing optional parameters, in place."""
        defaults_applied = 0
        for param, default in ParameterValidator.OPTIONAL_PARAMS.items():
            if param not in params:
                params[param] = default
                self.logger.debug(f"Applied default value for '{param}': {default}")
                defaults_applied += 1

        if defaults_applied > 0:
            self.logger.info(
                f"Applied default values for {defaults_applied} optional parameters"
            )
        else:
            self.logger.debug("No default values needed to be applied")

    def _setup_directories(self, params: Mapping[str, Any]) -> tuple:
        """
        Resolve the versioned output directory and the code version.

        Parameters
        ----------
        params : mapping
            Validated parameters, read only.

        Returns
        -------
        tuple
            ``(base_dir, git_hash)``, which become fields of the
            :class:`PipelineConfig`.
        """
        base_dir = get_versioned_path(
            params["cov_path"],
            write=self.overwrite,
            name=params["cov_name"],
        )

        git_hash_code = get_git_revision_short_hash()

        self.logger.info(f"Base directory: {base_dir}")
        self.logger.info(f"Git hash: {git_hash_code}")

        return base_dir, git_hash_code

    def _check_rebinning(self, other_params: dict[str, Any]) -> bool:
        """
        Check if current parameters are compatible with another set for rebinning.

        Every parameter except the binning
        (:attr:`ParameterValidator.REBINNING_PARAMS`) and the provenance keys
        (:attr:`ParameterValidator.PROVENANCE_PARAMS`, which record the run
        rather than the request) must match. A key of
        :attr:`ParameterValidator.OPTIONAL_PARAMS` that ``other_params`` does
        not have counts as its default value; any other missing key is
        incompatible.

        This runs only when a run is directed at a version directory that
        already holds a parameter file (``--overwrite N``), so that a
        directory never holds results from one set of parameters and the
        parameter file of another.

        Parameters
        ----------
        other_params : dict
            Other parameter dictionary to compare against

        Returns
        -------
        bool
            True if compatible, False otherwise
        """
        return not self._incompatible_parameters(other_params)

    def _incompatible_parameters(self, other_params: dict[str, Any]) -> list[str]:
        """The parameters that differ, in the sense of :meth:`_check_rebinning`."""
        self.logger.info("Checking parameter compatibility for rebinning")
        incompatible = []
        other_params = other_params or {}

        # ``map_filter`` changes the numbers, and the loop below walks only
        # this file's keys, so a saved file that has it where this one does
        # not is checked here.
        if MAP_FILTER_KEY in other_params and MAP_FILTER_KEY not in self.params:
            self.logger.warning(
                f"Parameter '{MAP_FILTER_KEY}' is only in the other parameter set"
            )
            incompatible.append(MAP_FILTER_KEY)

        for param in self.params:
            if param in ParameterValidator.REBINNING_PARAMS:
                self.logger.debug(f"Skipping rebinning parameter '{param}'")
                continue  # Skip rebinning parameters
            if param in ParameterValidator.PROVENANCE_PARAMS:
                # Recorded about the run (where it wrote, which revision it
                # ran at), not requested by it: never a reason to refuse.
                self.logger.debug(f"Skipping provenance parameter '{param}'")
                continue
            if param in other_params or param in ParameterValidator.OPTIONAL_PARAMS:
                # An optional key absent from the other parameter file had its
                # default there.
                other_value = other_params.get(
                    param, ParameterValidator.OPTIONAL_PARAMS.get(param)
                )
                if self.params[param] != other_value:
                    self.logger.warning(
                        f"Parameter '{param}' differs between parameter sets"
                    )
                    incompatible.append(param)
                else:
                    self.logger.debug(f"Parameter '{param}' matches")
            else:
                self.logger.warning(
                    f"Parameter '{param}' not found in other parameter set"
                )
                incompatible.append(param)

        if incompatible:
            self.logger.error(
                f"Found {len(incompatible)} incompatible parameters: {incompatible}"
            )
        else:
            self.logger.info("Parameter compatibility check passed")
        return incompatible

    def save_parameters(self) -> None:
        """
        Save the effective parameters (:attr:`params`) to ``save_path``, the
        parameter file's name inside the version directory.

        A previous parameter file of another name in that directory is removed.
        """
        self.logger.info(f"Saving parameters to: {self.save_path}")

        try:
            with open(self.save_path, "w") as f:
                # dict(): the record is a read-only mapping, which yaml has no
                # representer for.
                yaml.dump(
                    dict(self.params), f, default_flow_style=False, sort_keys=False
                )
            self.logger.info("Parameters saved successfully")
            # A renamed parameter file: drop the previous copy.
            previous = getattr(self, "_existing_parameter_file", None)
            if previous is not None and previous != self.save_path:
                os.remove(previous)
                self.logger.info(f"Removed previous parameter file: {previous}")
        except Exception as e:
            error_msg = f"Error saving parameters: {e}"
            self.logger.error(error_msg)
            raise OSError(error_msg) from e
