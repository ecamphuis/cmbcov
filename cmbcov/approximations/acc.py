"""
Approximated covariance coupling (Camphuis et al. 2022, Sect. 5).

Exploits the observation that the reduced coupling kernel depends only on the
multipole separations: Theta-bar at (l, l+Delta) is approximately the same as
at (l*, l*+Delta) for any reference l* (Eq. 33). The expensive exact kernel is
therefore computed once at l* and reused along each diagonal, giving 1%
accuracy for O(dmax * nside^4) work instead of O(lmax^5).
"""

import hashlib
import json
import math
import os
import shutil
import tempfile
import warnings
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..covariance import CovarianceMethod, warn_low_centralell_once
from ..grid import (
    banded_integrals_gl,
    full_from_pair,
    gl_mask_modes,
    spin_weighted_integrals_gl,
)
from ..keys import CovKey, SpecKey
from ..mask import MaskWlm
from ..sht import DEFAULT_MAP2ALM_ITER, ducc0_map2alm
from ..term_selection import TermSelection, pole_rotation, rotate_alm, select_terms
from ..utils.healpy_utils import get_nside_from_ell
from ..utils.threading_utils import (
    get_optimal_nthreads,
    row_block_workers,
    run_row_blocks,
)
from . import acc_cache
from .base import CovarianceStrategy

try:  # uncached I/O for the central store (_CentralStore); POSIX-only
    import fcntl

    _F_NOCACHE = getattr(fcntl, "F_NOCACHE", None)  # macOS
except ImportError:  # pragma: no cover - Windows
    fcntl = None
    _F_NOCACHE = None

__all__ = [
    "ACCStrategy",
    "CHANNEL_ALIASES",
    "COUPLING_CHANNELS",
    "COUPLING_SPECTRA",
    "DEFAULT_COUPLING_MEMORY_GB",
    "LEGACY_CHANNEL_NAMES",
    "acc_cached_kernel_size",
    "acc_internal_lmax",
    "acc_window_pad",
    "contract_coupling_block",
    "coupling_ellprange",
    "normalise_channel",
    "precompute_acc_kernels",
    "require_acc_cache_pairs",
]

#: Channels of the mode-coupling cross-spectra Theta^{s}(m, m', L), in the
#: channel order of ``theta_before_summing``.  Each is
#: ``sum_M u^a_{l m, LM} conj(u^b_{l' m', LM})`` with ``s = ab``, where ``u^T``
#: (field ``T``) is the spin-0 integral, ``u^D`` (field ``D``) the direct
#: spin-2 response and ``u^L`` (field ``L``) the E->B leakage response to an
#: E-mode unit vector -- ``D`` and ``L`` were called ``E`` and ``B`` before
#: B-mode support (docs/theory/bmode_kernels.md); :data:`CHANNEL_ALIASES`
#: accepts the old names.  ``TD`` pairs T at ``l`` with D at ``l'``; ``DT``
#: pairs D at ``l`` with T at ``l'``.  At ``l == l'`` one has ``Theta^{DT}(m,
#: m') = conj(Theta^{TD}(m', m))``, so a kernel whose ``DT`` legs are all
#: swapped for ``TD`` is unchanged (``DTxDT = TDxTD``, ``DTxTT = TDxTT``;
#: the old ``ET``/``TE`` relation) -- but the mixed ``TDxDT`` is not
#: ``TDxTD`` (2e-3 of the maximum on an asymmetric test mask; they agree
#: only on an azimuthally symmetric one, where ``Theta^{TD}`` is diagonal and
#: real).  For ``l != l'`` ``DT`` and ``TD`` differ (measured 3e-3 to 1.6e-2 of the kernel maximum at
#: ``(16, 17)`` and ``(16, 19)`` on the test mask), and ``DT`` is what the
#: second Wick contraction of Cov(TE, TE) needs.  The kernels
#: ``output["s1xs2"]`` are ordered pairs of this tuple.
COUPLING_CHANNELS = ("TT", "DD", "LL", "TD", "DT", "TL", "LT", "DL", "LD")

#: The five channels a precompute computes by default (``spectra=None``),
#: unchanged in meaning from before B-mode support: the old ``TT, EE, BB, TE,
#: ET``, renamed.  Extending this default would silently multiply the
#: precompute's disk and wall cost, so it stays pinned to these five; the four new
#: channels (``TL, LT, DL, LD``) are computed only when requested explicitly
#: via ``spectra`` or ``pairs``.
COUPLING_SPECTRA = COUPLING_CHANNELS[:5]

#: Old (pre-B-mode) channel strings, mapped onto their :data:`COUPLING_CHANNELS`
#: equivalent.  ``TB``/``BT``/``EB``/``BE`` have no old-precompute equivalent
#: on disk (B-mode kernels were never computed before this), but the strings
#: are accepted for symmetry with the T/E aliases.
CHANNEL_ALIASES = {
    "TT": "TT",
    "EE": "DD",
    "BB": "LL",
    "TE": "TD",
    "ET": "DT",
    "TB": "TL",
    "BT": "LT",
    "EB": "DL",
    "BE": "LD",
}


#: The reverse of :data:`CHANNEL_ALIASES`, restricted to the channels that
#: actually changed name: canonical (new) -> the single old on-disk label,
#: e.g. ``{"DD": "EE", "TD": "TE", "DT": "ET", ...}``.  Passed to
#: :func:`~cmbcov.approximations.acc_cache.load_coupling_kernels`
#: (``legacy_names``) so a kernel cache written under the old labels loads
#: without recompute.
LEGACY_CHANNEL_NAMES = {new: old for old, new in CHANNEL_ALIASES.items() if old != new}


def normalise_channel(name: str) -> str:
    """
    An old or new :data:`COUPLING_CHANNELS` string -> its canonical (new)
    name.  Raises ``ValueError`` on anything else.
    """
    if name in COUPLING_CHANNELS:
        return name
    if name in CHANNEL_ALIASES:
        return CHANNEL_ALIASES[name]
    raise ValueError(
        f"unknown coupling channel {name!r}; expected one of "
        f"{COUPLING_CHANNELS} or the old alias {tuple(CHANNEL_ALIASES)}"
    )


#: Field of the *central* (``ell``) integral set used by each entry of
#: :data:`COUPLING_CHANNELS`, indexing ``(T, D, L) = (0, 1, 2)``.
_CENTRAL_FIELD = (0, 1, 2, 0, 1, 0, 2, 1, 2)
#: Field of the *primed* (``ellp``) integral set, same convention.  ``TD``
#: pairs T at ``ell`` with D at ``ellp``, ``DT`` the other way round, and
#: likewise for the ``L`` (leakage) channels.
_PRIME_FIELD = (0, 1, 2, 1, 0, 2, 0, 2, 1)

#: The subset of :data:`COUPLING_CHANNELS` that :meth:`ACCStrategy.get_covariance_coupling`
#: loads by default (unchanged from before the ``spectra`` selection was
#: added): the sixteen ``{TT,DD,TD,DT}^2`` pairs (old ``{TT,EE,TE,ET}^2``).
#: ``LL`` is computed by the precompute but never read back by the strategy.
_DEFAULT_KERNEL_STOKES = ("TT", "DD", "TD", "DT")


#: The :data:`COUPLING_CHANNELS` with an E->B leakage (``L``) leg, for which
#: term selection does not meet its tolerance on every mask.
_LEAKAGE_CHANNELS = frozenset(c for c in COUPLING_CHANNELS if "L" in c)


def _resolve_spectra(
    spectra: Sequence[str] | None,
) -> tuple[tuple[str, ...], list[int], bool]:
    """
    Validate and normalise a ``spectra`` argument to
    :func:`precompute_acc_kernels`.

    Parameters
    ----------
    spectra : sequence of str, optional
        A subset of :data:`COUPLING_CHANNELS`, old or new names (see
        :data:`CHANNEL_ALIASES`).  ``None`` (the default) means
        :data:`COUPLING_SPECTRA`, the five channels computed before B-mode
        support existed.

    Returns
    -------
    spectra_tuple : tuple of str
        The requested subset, normalised to the new names and in
        :data:`COUPLING_CHANNELS` order (so the output is deterministic
        regardless of the order ``spectra`` was given in).
    spec_indices : list of int
        The indices of ``spectra_tuple`` into :data:`COUPLING_CHANNELS`.
    t_only : bool
        Whether every kernel the caller asked for needs only the spin-0 (T)
        field on both the central and primed side -- i.e. ``spectra`` is
        ``("TT",)``.  When true, the GL branch can skip the spin-2 (D, L)
        integrals entirely (two Legendre components against one for T; on
        the earlier two-SHT producer they were ~92% of the precompute wall
        time at ``centralell=250``, ``nside=256``).
    """
    if spectra is None:
        spectra_tuple = COUPLING_SPECTRA
    else:
        requested = {normalise_channel(s) for s in spectra}
        if not requested:
            raise ValueError("spectra must not be empty")
        spectra_tuple = tuple(s for s in COUPLING_CHANNELS if s in requested)

    spec_indices = [COUPLING_CHANNELS.index(s) for s in spectra_tuple]
    fields_needed = {_CENTRAL_FIELD[i] for i in spec_indices} | {
        _PRIME_FIELD[i] for i in spec_indices
    }
    t_only = fields_needed == {0}
    return spectra_tuple, spec_indices, t_only


#: Default memory budget of the ACC precompute, in GiB: everything it holds
#: that scales with the problem -- coefficient sets held in RAM and the
#: blocked working set of :func:`_accumulate_coupling_kernels`
#: (:func:`_coupling_working_set`) -- on top of a fixed baseline (see
#: :func:`precompute_acc_kernels`, ``max_memory_gb``).
DEFAULT_COUPLING_MEMORY_GB = 2.0

#: Inner-block width aimed at when the whole inner set does not fit in the
#: budget (:func:`_coupling_block_sizes`).  The inner side is re-visited once
#: per outer block -- re-synthesised on the streamed path, re-read from RAM or
#: from the disk store otherwise -- so the budget is spent on the outer block
#: to keep the number of passes down, while this floor keeps the per-``L``
#: GEMMs wide enough to run near BLAS peak.
_INNER_BLOCK_TARGET = 32


def _gl_banded_grid(lw: int, ell: int, lmax_out: int) -> int:
    """
    The GL grid band-limit :func:`~cmbcov.grid.banded_integrals_gl`
    (and :func:`~cmbcov.grid.spin_weighted_integrals_gl`)
    choose by default for ``(lw, ell, lmax_out)``: ``max(ceil((lw + ell +
    lmax_out - 1) / 2), ell, lmax_out)``.  Restated here so the mask modes
    can be computed once per ``ell`` on exactly that grid and handed in.
    """
    return max(math.ceil((lw + ell + lmax_out - 1) / 2), ell, lmax_out)


#: Producer of the GL coefficients on the default path (no term selection).
#: ``"legendre"``: :func:`~cmbcov.grid.banded_integrals_gl` at full band
#: (``m_band = lmax_out + |m|``), the azimuthal sum done analytically from
#: the mask's ring Fourier modes (:func:`~cmbcov.grid.gl_mask_modes`,
#: computed once per grid and shared by every ``m``) and one
#: ``ducc0.sht.leg2alm`` per sign of ``M`` and spin.  ``"two_sht"``: the
#: reference route :func:`~cmbcov.grid.spin_weighted_integrals_gl`
#: (synthesise the mask and ``Y_lm``, multiply, analyse: about five
#: transforms per ``m`` and spin).  Both evaluate the same product with the
#: same Gauss-Legendre rule on the same grid and agree to rounding
#: (``tests/test_acc_gl_producer.py``); the switch exists for those tests
#: and is not a user option.
_GL_PRODUCER = "legendre"


def _gl_integrals_m(
    mask_alm: np.ndarray,
    lw: int,
    ell: int,
    m: int,
    lmax: int,
    t_only: bool = False,
    m_band: int | None = None,
    mask_modes: np.ndarray | None = None,
) -> tuple:
    """
    GL mode-coupling integrals for one ``(ell, m)``: ``(u_T, u_EB)``, ``u_T``
    the spin-0 full-M array and ``u_EB`` the ``(2, lmax, 2*lmax-1)`` (E, B)
    response to the E-mode unit vector, or ``(u_T, None)`` with ``t_only``.

    The integrals come from :func:`~cmbcov.grid.banded_integrals_gl` on the
    grid :func:`_gl_banded_grid`, with the mask modes ``mask_modes``
    (:func:`~cmbcov.grid.gl_mask_modes` on that grid, computed here when
    ``None``; callers share one per grid across ``m``).  ``m_band=None``
    (the default path) is the full band ``lmax_out + |m|``, i.e. every
    ``M``: the same numbers as the two-SHT
    :func:`~cmbcov.grid.spin_weighted_integrals_gl` to rounding, which is
    what runs instead when :data:`_GL_PRODUCER` is ``"two_sht"``.  With an
    ``m_band`` (term selection) the integrals are restricted to
    ``|M - m| <= m_band``.
    """
    lmax_out = lmax - 1
    if m_band is None and _GL_PRODUCER == "two_sht":
        u_t = spin_weighted_integrals_gl(mask_alm, lw, ell, m, lmax_out)
        if t_only:
            return u_t, None
        return u_t, spin_weighted_integrals_gl(mask_alm, lw, ell, m, lmax_out, spin=2)
    if m_band is None:
        m_band = lmax_out + abs(m)
    lg = _gl_banded_grid(lw, ell, lmax_out)
    if mask_modes is None:
        mask_modes = gl_mask_modes(mask_alm, lw, lg)
    u_t = banded_integrals_gl(
        mask_alm, lw, ell, m, lmax_out, m_band, lmax_grid=lg, mask_modes=mask_modes
    )
    if t_only:
        return u_t, None
    u_eb = banded_integrals_gl(
        mask_alm,
        lw,
        ell,
        m,
        lmax_out,
        m_band,
        lmax_grid=lg,
        spin=2,
        mask_modes=mask_modes,
    )
    return u_t, u_eb


def _reflect_into(
    src: np.ndarray, m: int, out: np.ndarray, conjugate: bool = False
) -> None:
    r"""
    Write the coefficients of order ``m < 0`` from those of ``|m|`` (one
    field, full-``M`` layout ``[..., L, M + lmax_out]``) into ``out``:

        ``I_{l,m,LM} = (-1)^(m+M) conj(I_{l,|m|,L,-M})``

    (the reflection of :func:`_coefficient_provider`), or its complex
    conjugate with ``conjugate=True``.  The one implementation of the
    identity, so a set built with it (:func:`_gl_integrals`) and a provider
    reflecting on the fly give the same bits.
    """
    lmax_out = (src.shape[-1] - 1) // 2
    factor = ((-1.0) ** m) * (-1.0) ** np.arange(-lmax_out, lmax_out + 1)
    if conjugate:
        np.multiply(src[..., ::-1], factor, out=out)
    else:
        np.conjugate(src[..., ::-1], out=out)
        out *= factor


def _reflected_item(item: tuple, m: int) -> tuple:
    """The ``(u_T, u_EB)`` item of order ``m < 0`` from the item of ``|m|``
    (:func:`_reflect_into`, field by field)."""
    reflected = []
    for part in item:
        if part is None:
            reflected.append(None)
            continue
        out = np.empty_like(part)
        _reflect_into(part, m, out)
        reflected.append(out)
    return tuple(reflected)


def _gl_integrals(
    mask_alm: np.ndarray,
    lw: int,
    ell: int,
    lmax: int,
    t_only: bool = False,
    keep_m: np.ndarray | None = None,
    m_band: int | None = None,
    mask_modes: np.ndarray | None = None,
) -> list[tuple | None]:
    """
    GL mode-coupling integrals for every ``m`` of one ``ell``: a list of
    ``(u_T, u_EB)`` tuples (:func:`_gl_integrals_m`), index ``m + ell``.

    ``lmax`` is the acc.py convention ``2*nside`` (an array length, not a
    band-limit); the output band-limit is ``lmax - 1``.

    ``t_only=True`` skips the spin-2 synthesis and returns ``(u_T, None)``
    tuples instead -- the coefficient provider (:func:`_gl_full_m`) then never
    touches the second element.  Spin 2 has two Legendre components against
    one for T (on the two-SHT route it was ~92% of the precompute wall time
    at ``centralell=250``, ``nside=256``), so a caller that only needs
    T-field kernels (``spectra=("TT",)``) skips it entirely.

    Default path (no ``keep_m``, no ``m_band``): the mask modes are computed
    once for the grid of ``ell`` (unless given) and shared by every ``m``,
    and only ``m >= 0`` is transformed -- the ``m < 0`` half is the
    reflection :func:`_reflected_item`, exactly as a reflecting
    :func:`_coefficient_provider` serves it, so a set held in RAM and one
    synthesised on demand hold the same bits.

    ``keep_m`` (term selection, index ``m + ell``) leaves ``None`` at the
    orders that are not kept; the contraction never asks for them.
    ``m_band`` / ``mask_modes`` are forwarded to :func:`_gl_integrals_m`;
    with a selection every kept ``m`` is transformed directly, as before.
    """
    if keep_m is not None or m_band is not None:
        return [
            (
                _gl_integrals_m(
                    mask_alm,
                    lw,
                    ell,
                    m,
                    lmax,
                    t_only=t_only,
                    m_band=m_band,
                    mask_modes=mask_modes,
                )
                if keep_m is None or keep_m[m + ell]
                else None
            )
            for m in range(-ell, ell + 1)
        ]
    if mask_modes is None and _GL_PRODUCER != "two_sht":
        mask_modes = gl_mask_modes(mask_alm, lw, _gl_banded_grid(lw, ell, lmax - 1))
    items: list[tuple | None] = [None] * (2 * ell + 1)
    for m in range(ell + 1):
        item = _gl_integrals_m(
            mask_alm, lw, ell, m, lmax, t_only=t_only, mask_modes=mask_modes
        )
        items[ell + m] = item
        if m > 0:
            items[ell - m] = _reflected_item(item, -m)
    return items


#: Mask-mode grids a GL precompute keeps at once (:class:`_GLMaskModes`, and
#: the term-selection plan's own cache): the central grid and the current
#: primed one.
_GL_MODE_GRIDS = 2


class _GLMaskModes:
    """
    :func:`~cmbcov.grid.gl_mask_modes` of one mask alm, per GL grid, for the
    default GL producer (:func:`_gl_integrals_m`).

    ``modes(ell)`` returns the modes on the grid :func:`_gl_banded_grid`
    ``(lw, ell, lmax_out)``, computing them on first use.  Grids depend on
    ``ell`` only through ``ceil((lw + ell + lmax_out - 1) / 2)``, so the
    central and a neighbouring primed multipole often share one; the cache
    keeps the ``max_grids`` (default :data:`_GL_MODE_GRIDS`) most recently
    used grids -- the central one and the current primed one on the
    streamed path -- rather than one per ``ellp`` of a long ``ellprange``.
    One grid is a complex ``(Lg + 1, 2 Lg + 2)`` array: 21.5 MB at ``Lg =
    818`` (``nside = 256``, ``lw = 875``, ``ell = 250``), about 0.3 GB at
    ``Lg = 3060``; the memory budget charges :data:`_GL_MODE_GRIDS` of the
    largest (:func:`_gl_producer_bytes`).

    The term-selection path keeps its own instance
    (:meth:`_TermSelectionPlan.mask_modes`, pole-frame mask).
    """

    def __init__(
        self,
        mask_alm: np.ndarray,
        lw: int,
        lmax_out: int,
        max_grids: int | None = None,
    ) -> None:
        self.mask_alm = mask_alm
        self.lw = int(lw)
        self.lmax_out = int(lmax_out)
        self.max_grids = _GL_MODE_GRIDS if max_grids is None else int(max_grids)
        self._modes: dict[int, np.ndarray] = {}

    def matches(self, mask_alm: np.ndarray, lw: int, lmax_out: int) -> bool:
        return (
            self.mask_alm is mask_alm
            and self.lw == int(lw)
            and self.lmax_out == int(lmax_out)
        )

    def __call__(self, ell: int) -> np.ndarray:
        lg = _gl_banded_grid(self.lw, ell, self.lmax_out)
        modes = self._modes.pop(lg, None)
        if modes is None:
            modes = gl_mask_modes(self.mask_alm, self.lw, lg)
            modes.setflags(write=False)
        self._modes[lg] = modes  # most recently used last
        while len(self._modes) > self.max_grids:
            del self._modes[next(iter(self._modes))]
        return modes


def _union_selection(selections: Sequence[TermSelection]) -> TermSelection:
    """
    The union of several :class:`~cmbcov.term_selection.TermSelection`
    of the same ``(ell, ellp)`` (one per spin): ``keep_m`` / ``keep_mp`` /
    ``keep_pair`` OR-ed, ``m_band`` the maximum, so that no requested field
    loses a term its own spin would have kept.  The thresholds recorded are
    those of the first selection (they are the same for every spin).
    """
    first = selections[0]
    keep_m = np.logical_or.reduce([s.keep_m for s in selections])
    keep_mp = np.logical_or.reduce([s.keep_mp for s in selections])
    keep_pair = np.logical_or.reduce([s.keep_pair for s in selections])
    return TermSelection(
        ell=first.ell,
        ellp=first.ellp,
        keep_m=keep_m,
        keep_mp=keep_mp,
        keep_pair=keep_pair,
        m_band=max(s.m_band for s in selections),
        tolerance=first.tolerance,
        eps_m=first.eps_m,
        eps_pair=first.eps_pair,
        delta_band=first.delta_band,
    )


class _TermSelectionPlan:
    r"""
    The a-priori term selection of one precompute (``grid="gl"`` only).

    Holds the mask alm **in the pole frame** (see
    :mod:`cmbcov.term_selection`: the kernels are
    rotation invariant, the selection is only sharp in that frame), the
    tolerance, and caches of everything that is shared across ``ellp``:

    * ``selection(ell, ellp)``: the union over ``spins`` of
      :func:`~cmbcov.term_selection.select_terms`
      (spin 0 alone for a T-only precompute, spin 0 and 2 otherwise).
    * ``keep_m(l)``: the kept orders of degree ``l`` -- ``select_terms``
      derives them from :func:`~cmbcov.term_selection.mode_power`
      at ``l`` alone, so they do not depend on the partner multipole and the
      central integral set can be built once and reused for every ``ellp``.
    * ``m_band``: the ``|M - m|`` band, which depends only on the mask's
      azimuthal spectrum and is therefore one number for the whole run.
    * ``mask_modes(l)``: :func:`~cmbcov.grid.gl_mask_modes`
      on the grid :func:`_gl_banded_grid` of ``l`` (a :class:`_GLMaskModes`
      of the pole-frame alm: the :data:`_GL_MODE_GRIDS` most recently used
      grids are kept, as on the default path, instead of one per ``ellp``).
    """

    def __init__(
        self,
        mask_alm_pole: np.ndarray,
        lw: int,
        lmax: int,
        ell: int,
        tolerance: float,
        t_only: bool,
    ) -> None:
        if tolerance <= 0:
            raise ValueError("term_selection must be a positive tolerance")
        self.mask_alm = np.ascontiguousarray(mask_alm_pole, dtype=np.complex128)
        self.lw = int(lw)
        self.lmax_out = int(lmax) - 1
        self.tolerance = float(tolerance)
        self.spins: tuple[int, ...] = (0,) if t_only else (0, 2)
        self._selections: dict[tuple[int, int], TermSelection] = {}
        self._modes = _GLMaskModes(
            self.mask_alm, self.lw, self.lmax_out, max_grids=_GL_MODE_GRIDS
        )
        self.m_band = self.selection(ell, ell).m_band

    def selection(self, ell: int, ellp: int) -> TermSelection:
        key = (int(ell), int(ellp))
        sel = self._selections.get(key)
        if sel is None:
            sel = _union_selection(
                [
                    select_terms(
                        self.mask_alm, self.lw, key[0], key[1], self.tolerance, spin=s
                    )
                    for s in self.spins
                ]
            )
            self._selections[key] = sel
        return sel

    def keep_m(self, l_val: int) -> np.ndarray:
        return self.selection(l_val, l_val).keep_m

    def mask_modes(self, l_val: int) -> np.ndarray:
        return self._modes(l_val)

    def stats(self, ell: int, ellp: int) -> dict:
        """The fractions kept at ``(ell, ellp)``, for the cache manifest."""
        sel = self.selection(ell, ellp)
        return {
            "frac_m": sel.frac_m,
            "frac_mp": sel.frac_mp,
            "frac_pairs": sel.frac_pairs,
            "m_band": int(sel.m_band),
            "frac_band": sel.frac_band(self.lmax_out),
        }


def _gl_full_m(item: tuple, t_only: bool = False) -> np.ndarray:
    """One GL ``(u_T, u_EB)`` item as the ``(nfields, lmax, 2*lmax-1)``
    complex coefficient stack: ``(T, E, B)``, or ``(T,)`` with ``t_only``."""
    u_t, u_eb = item
    if t_only:
        return u_t[np.newaxis]
    return np.stack((u_t, u_eb[0], u_eb[1]))


def _healpix_integrals(
    wlm: MaskWlm, wn_for_ell: np.ndarray, nside: int, ell: int
) -> np.ndarray:
    """
    HEALPix mode-coupling integrals for every ``m`` of one ``ell``, in the
    compact form: ``(2*ell+1, 2, 3, nalm)`` complex, one
    :meth:`~cmbcov.mask.MaskWlm.compute_spin_weighted_integrals`
    result per ``m`` (index ``m + ell``; axis 1 the healpy alms of the real and
    imaginary parts of the weighted map, axis 2 the (T, E, B) field).
    """
    import healpy as hp  # lazy, see utils/healpy_utils.py

    nalm = hp.Alm.getsize(2 * nside - 1)
    out = np.empty((2 * ell + 1, 2, 3, nalm), dtype=np.complex128)
    for i, m in enumerate(range(-ell, ell + 1)):
        out[i] = wlm.compute_spin_weighted_integrals(
            ell, m, mask_for_ell=wn_for_ell, target_nside=nside
        )
    return out


def _healpix_full_m(cma: np.ndarray, lmax: int, nfields: int = 3) -> np.ndarray:
    r"""
    One compact HEALPix integral set, ``(2, 3, nalm)``, as the
    ``(nfields, lmax, 2*lmax-1)`` complex full-``M`` coefficient stack.

    Field by field this is :func:`~cmbcov.grid.full_from_pair`
    of the alms ``r`` (real part) and ``s`` (imaginary part):
    ``x[L, M] = r + i s`` and ``x[L, -M] = (-1)^M (conj r + i conj s)``, which
    is the harmonic expansion of the complex integral map ``Re + i Im`` for the
    spin-0 T and, component by component, for the (E, B) pair (the same
    reality condition holds for the E/B alms of a real ``(Q, U)``).  Nothing is
    dropped: ``Re sum_M x conj(y)`` is ``alm2cl(x_r, y_r) + alm2cl(x_s, y_s)``
    times ``2L+1`` (the ``M > 0`` pair supplies alm2cl's ``2 - delta_{M0}``
    weight), and ``Im sum_M x conj(y)`` is kept too.  Only the first
    ``nfields`` fields are expanded (``nfields = 1`` for T only).  The E/B
    fields keep the ``sqrt(2)`` of
    :func:`~cmbcov.sht.cplx_spin_weighted_ylm`.
    """
    return full_from_pair(cma[0, :nfields], cma[1, :nfields], lmax - 1)


class _CoefficientProvider:
    """
    Per-``m`` full-``M`` coefficients of one ``ell``; see
    :func:`_coefficient_provider`, which builds it.

    ``provider(i)`` returns the complex128 ``(nfields, lmax, 2*lmax-1)``
    stack of ``m = i - ell`` as a new array; ``provider.into(i, out,
    conjugate)`` writes it (or its complex conjugate) straight into ``out``
    -- typically one ``m`` of a packed block (:func:`_pack_block`), a strided
    view -- so the block is filled without an intermediate full-``M`` copy.
    """

    def __init__(self, held, synthesise, to_full_m, ell, reflect, fields, into):
        self.held = held
        self.synthesise = synthesise
        self.to_full_m = to_full_m
        self.ell = ell
        self.reflect = reflect and held is None
        self.fields = fields
        self.into_fn = into
        # One-entry cache of the +|m| fields, for the reflection (see
        # _coefficient_provider).
        self._cache: dict[int, Sequence[np.ndarray]] = {}

    def _item(self, m: int):
        if self.held is not None:
            return self.held[m + self.ell]
        return self.synthesise(m)

    def _fields_of(self, item) -> Sequence[np.ndarray]:
        """The per-field 2-D full-``M`` arrays of ``item``, without a copy
        when the grid provides ``fields`` (GL); else ``to_full_m(item)``."""
        if self.fields is not None:
            return self.fields(item)
        return self.to_full_m(item)

    def _reflected_source(self, m: int) -> Sequence[np.ndarray]:
        am = abs(m)
        fields = self._cache.get(am)
        if fields is None:
            fields = self._fields_of(self.synthesise(am))
            self._cache.clear()
            self._cache[am] = fields
        return fields

    def into(self, i: int, out: np.ndarray, conjugate: bool = False) -> None:
        """Write the coefficients of ``m = i - ell`` (conjugated if
        ``conjugate``) into ``out``, shape ``(nfields, lmax, 2*lmax-1)``."""
        m = i - self.ell
        if not self.reflect:
            item = self._item(m)
            if self.into_fn is not None:
                self.into_fn(item, out)
                if conjugate:
                    np.conjugate(out, out=out)
                return
            fields = self._fields_of(item)
        else:
            fields = self._reflected_source(m)
        reflected = self.reflect and m < 0

        def rows(r0: int, r1: int) -> None:
            for f, src in enumerate(fields):
                if reflected:
                    # I_{l,-m,LM} = (-1)^(m+M) conj(I_{l,m,L,-M}), factor +-1.
                    _reflect_into(src[r0:r1], m, out[f, r0:r1], conjugate)
                elif conjugate:
                    np.conjugate(src[r0:r1], out=out[f, r0:r1])
                else:
                    np.copyto(out[f, r0:r1], src[r0:r1])

        # Memory-bound copies, split over L on the package's threads; the
        # same elementwise operations, so the same bits as one thread.
        run_row_blocks(
            rows,
            out.shape[1],
            row_block_workers(out[0].size * len(fields), get_optimal_nthreads()),
        )

    def __call__(self, i: int) -> np.ndarray:
        if not self.reflect:
            item = self._item(i - self.ell)
            if self.fields is None:
                return self.to_full_m(item)
            return np.stack(self._fields_of(item))
        source = self._reflected_source(i - self.ell)
        out = np.empty((len(source),) + source[0].shape, dtype=np.complex128)
        self.into(i, out)
        return out


def _coefficient_provider(
    held: Sequence | None,
    synthesise: Callable[[int], object],
    to_full_m: Callable[[object], np.ndarray],
    ell: int,
    reflect: bool,
    fields: Callable[[object], Sequence[np.ndarray]] | None = None,
    into: Callable[[object, np.ndarray], object] | None = None,
) -> _CoefficientProvider:
    """
    Per-``m`` coefficient provider shared by both grids.

    The returned callable maps an index ``i`` (``m = i - ell``) to the
    complex128 ``(nfields, lmax, 2*lmax-1)`` full-``M`` coefficient stack
    ``to_full_m(item)``, where ``item`` is one per-``m`` integral set in the
    grid's own storage form -- ``held[i]`` when the whole set of this ``ell``
    is held in RAM, else ``synthesise(m)``:

    ============  ==========================  ===========================
    grid          item (``synthesise(m)``)    ``to_full_m``
    ============  ==========================  ===========================
    ``"gl"``      ``(u_T, u_EB)`` tuple       :func:`_gl_full_m`
    ``"healpix"`` ``(2, 3, nalm)`` compact    :func:`_healpix_full_m`
    ============  ==========================  ===========================

    so a held HEALPix set stays compact and is expanded (indexing only) when
    an ``m`` is asked for.  Its ``into(i, out, conjugate)`` method writes the
    same numbers straight into a destination (:func:`_pack_block`); the
    optional ``fields`` (``item`` -> the ``nfields`` 2-D full-``M`` arrays it
    already holds, GL) and ``into`` (``(item, out)`` -> fill ``out``,
    HEALPix) let it do so without the intermediate stack ``to_full_m``
    builds.  Neither changes a value.

    ``reflect=True`` (GL only) lets the synthesised case pay one transform per
    ``|m|``.  For a real mask

        ``I_{l,-m,LM} = (-1)^(m+M) conj(I_{l,m,L,-M})``

    (both spins; bit-for-bit on the two-SHT route,
    ``tests/test_acc_vectorised.py``, and to rounding on the default
    Legendre route, ``tests/test_acc_gl_producer.py``), so the ``m < 0`` half
    is an index reflection of the ``m > 0`` half (:func:`_reflect_into`).
    Paired with the ``+m, -m`` iteration order of :func:`_paired_m_order`, a
    one-entry cache turns that into half the transforms; a held GL set
    (:func:`_gl_integrals`) is built with the same reflection, so held and
    synthesised coefficients are the same bits.  The identity has not been verified on
    the HEALPix grid (pixel quadrature, iterated ``map2alm``), so the HEALPix
    branch passes ``reflect=False`` and synthesises every ``m``.
    """
    return _CoefficientProvider(held, synthesise, to_full_m, ell, reflect, fields, into)


def _paired_m_order(n: int) -> list[int]:
    """
    Iteration order over the ``2 ell + 1`` values of ``m`` that puts ``+m``
    immediately before ``-m``.

    The kernels sum over all ``(m, m')`` pairs, so the order is free; this one
    lets a reflecting :func:`_coefficient_provider` (GL) serve ``-m`` from the
    transform it just did for ``+m``.
    """
    ell = (n - 1) // 2
    order = [ell]
    for d in range(1, ell + 1):
        order.extend((ell + d, ell - d))
    return order


def _block_array(
    buffer: np.ndarray | None, nfields: int, lmax: int, n: int
) -> np.ndarray:
    """A C-contiguous complex128 ``(nfields, lmax, n, 2 lmax - 1)`` block:
    new, or the leading entries of the flat ``buffer`` (reused for every
    block of a contraction, :func:`_accumulate_coupling_kernels`)."""
    shape = (nfields, lmax, n, 2 * lmax - 1)
    if buffer is None:
        return np.empty(shape, dtype=np.complex128)
    size = math.prod(shape)
    if buffer.dtype != np.complex128 or buffer.ndim != 1 or buffer.size < size:
        raise ValueError(
            f"block buffer must be flat complex128 with at least {size} entries"
        )
    return buffer[:size].reshape(shape)


def _pack_block(
    coeff_fn: Callable[[int], np.ndarray],
    m_values: Sequence[int],
    lmax: int,
    conjugate: bool,
    nfields: int = 3,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """
    Gather the complex full-``M`` coefficients of a block of ``m`` into one
    complex128 array, ``ncoef = 2 * lmax - 1``.

    ``conjugate=False`` gives the C-contiguous ``(nfields, lmax,
    len(m_values), ncoef)`` -- the left operand of the per-``L`` GEMM.
    ``conjugate=True`` gives the right operand, the conjugated coefficients
    with shape ``(nfields, lmax, ncoef, len(m_values))``, as the transposed
    view of a C-contiguous ``(nfields, lmax, len(m_values), ncoef)`` array.
    Either way every ``m`` is written as ``nfields * lmax`` contiguous rows:
    filling the transposed layout directly scattered each ``m`` over the
    whole block with a stride of ``len(m_values)`` elements, which on a
    wide block (the primed side is the wide outer block once the central
    set is held or stored) touched every page of it per ``m``.  ``np.matmul``
    passes the transposed operand to BLAS as such, without a copy (measured
    bit-identical to the contiguous layout on Accelerate).

    A :class:`_CoefficientProvider` writes each ``m`` straight into its slot
    of the block (:meth:`_CoefficientProvider.into`); a plain callable is
    copied from.  ``nfields`` is 3 (T, E, B) except on
    the T-only path (``spectra=("TT",)``, providers built with
    ``t_only=True``), where it is 1.  ``out``, a flat complex128 buffer,
    receives the block in its leading entries instead of a new array
    (:func:`_block_array`).
    """
    n = len(m_values)
    block = _block_array(out, nfields, lmax, n)
    into = getattr(coeff_fn, "into", None)
    for i, m in enumerate(m_values):
        if into is not None:
            into(m, block[:, :, i, :], conjugate)
            continue
        fields = coeff_fn(m)
        for f in range(nfields):
            if conjugate:
                np.conjugate(fields[f], out=block[f, :, i, :])
            else:
                block[f, :, i, :] = fields[f]
    return block.transpose(0, 1, 3, 2) if conjugate else block


#: ``(max_bytes, floor)`` pairs already warned about by
#: :func:`_coupling_block_sizes`: the precompute asks once per ``(ell, ellp)``
#: pair, always with the same numbers.
_WARNED_MEMORY_FLOORS: set = set()

#: Transient bytes per ``nside`` pixel of one HEALPix integral set
#: (:meth:`~cmbcov.mask.MaskWlm.compute_spin_weighted_integrals`: the
#: spin-weighted ``Y_lm`` maps, their product with the mask, the real and
#: imaginary parts and the ``map2alm`` residuals).  Measured 186 B per pixel
#: with ``tracemalloc`` at ``nside`` 16 and 32; rounded up for the ``ducc0``
#: scratch that ``tracemalloc`` does not see.
_HEALPIX_PRODUCER_BYTES_PER_PIXEL = 256


def _gl_producer_bytes(lw: int, lmax: int, ells: Iterable[int], nfields: int) -> int:
    """
    What the GL coefficient producer holds while it builds a block, beyond
    the block itself (:func:`_coupling_working_set`, ``producer``):

    * three per-``m`` coefficient sets, ``per_m`` each -- the order being
      synthesised (:func:`_gl_integrals_m`), the one-entry ``+|m|`` cache of
      the provider building the block, and the cache of the other side's
      provider on the streamed path (:func:`_coefficient_provider`);
    * the legs of one :func:`~cmbcov.grid.banded_integrals_gl` call,
      ``ncomp * ntheta * lmax`` complex (``ncomp`` 2 for spin 2);
    * the mask modes of :data:`_GL_MODE_GRIDS` grids, ``ntheta * nphi``
      complex each.

    ``ells`` are the multipoles whose grids may be resident (``ell`` and
    ``ellp``); the largest grid is charged.
    """
    lmax_out = lmax - 1
    lg = max(_gl_banded_grid(lw, int(l_val), lmax_out) for l_val in ells)
    ntheta, nphi = lg + 1, 2 * lg + 2  # grid.gl_shape
    itemsize = np.dtype(np.complex128).itemsize
    per_m = nfields * lmax * (2 * lmax - 1) * itemsize
    legs = (1 if nfields == 1 else 2) * ntheta * lmax * itemsize
    modes = _GL_MODE_GRIDS * ntheta * nphi * itemsize
    return 3 * per_m + legs + modes


def _healpix_producer_bytes(nside: int, lmax: int, nfields: int) -> int:
    """The HEALPix counterpart of :func:`_gl_producer_bytes`: the transients
    of one :meth:`~cmbcov.mask.MaskWlm.compute_spin_weighted_integrals`
    call (:data:`_HEALPIX_PRODUCER_BYTES_PER_PIXEL`) plus one full-``M``
    stack (the provider reflects nothing on this grid, so keeps no cache)."""
    per_m = nfields * lmax * (2 * lmax - 1) * np.dtype(np.complex128).itemsize
    return _HEALPIX_PRODUCER_BYTES_PER_PIXEL * 12 * nside**2 + per_m


def _coupling_working_set(
    n_outer: int,
    n_inner: int,
    lmax: int,
    per_m: int,
    per_pair: int,
    nspec: int,
    inner_copies: int = 1,
    producer_bytes: int | None = None,
) -> dict[str, int]:
    """
    The peak resident bytes the blocked contraction
    (:func:`_accumulate_coupling_kernels`) is charged for, term by term, with
    outer blocks of ``n_outer`` orders and inner blocks of ``n_inner``:

    ``kernels``
        the ``(nspec, nspec, lmax, lmax)`` float64 accumulator, plus the one
        ``lmax x lmax`` gram being added into it
        (:func:`contract_coupling_block` with ``out``).
    ``outer``
        the outer coefficient block, ``n_outer * per_m``.
    ``inner``
        ``inner_copies`` inner blocks (2 when the next one is read ahead),
        ``inner_copies * n_inner * per_m``.
    ``theta``
        ``n_outer * n_inner * per_pair``: ``per_pair`` is the ``Theta`` bytes
        of one ``(m, m')`` pair, i.e. ``(nchannels + 1) * lmax * 16`` -- the
        float ``(Re, Im)`` slab of every channel plus the complex channel
        being converted into it.
    ``producer``
        what the coefficient producer holds beyond the block it fills
        (:func:`_gl_producer_bytes`, :func:`_healpix_producer_bytes`);
        ``None`` charges three per-``m`` sets, the GL figure without its
        legs and mask modes.
    ``total``
        the sum.  An upper bound: the producer only runs between
        contractions, and while an outer block is built at most one inner
        block is resident, but every term is charged as if all were
        resident at once.

    Not charged here, because the caller charges or excludes them: a central
    set held in RAM, and primed sets kept for a later ``ellp`` (subtracted
    from the budget before the contraction is planned,
    :meth:`_CouplingPrecompute._compute_ellp_coupling`), and the fixed
    baseline of :func:`precompute_acc_kernels` (``max_memory_gb``).
    """
    if producer_bytes is None:
        producer_bytes = 3 * per_m
    terms = {
        "kernels": (nspec * nspec + 1) * lmax * lmax * 8,
        "outer": n_outer * per_m,
        "inner": inner_copies * n_inner * per_m,
        "theta": n_outer * n_inner * per_pair,
        "producer": int(producer_bytes),
    }
    terms["total"] = sum(terms.values())
    return terms


def _coupling_block_sizes(
    n_m: int,
    n_mp: int,
    lmax: int,
    per_m: int,
    per_pair: int,
    max_bytes: int,
    nspec: int | None = None,
    inner_copies: int = 1,
    producer_bytes: int | None = None,
) -> tuple[int, int]:
    """
    Block widths ``(nb, nbp)`` over the outer and the inner side of the
    contraction, ``n_m`` and ``n_mp`` orders long: the blocking with the
    fewest passes whose :func:`_coupling_working_set` fits ``max_bytes``.

    ``inner_copies`` is how many inner blocks are resident at once: 1, or 2
    when the next inner block is read ahead while the current one is
    contracted (:func:`_accumulate_coupling_kernels`, primed-outer order).
    ``per_pair`` and ``producer_bytes`` are as in
    :func:`_coupling_working_set`; ``nspec`` (the side of the output
    accumulator) defaults to all of :data:`COUPLING_SPECTRA`, i.e. 5.

    The function is symmetric in what the two sides hold; which one is
    outer is the caller's choice (:func:`_accumulate_coupling_kernels`): the
    central ``m`` on the streamed path, the primed ``m'`` when the central
    set can be re-read (held in RAM or in the disk store).  Below, "central"
    means outer and "primed" inner.

    The primed block is the inner loop and is revisited (rebuilt or re-read)
    once per central block, so the budget is spent on ``nb`` first and
    ``nbp`` is held at :data:`_INNER_BLOCK_TARGET` (halved only if even that
    will not fit, or if the halving buys enough central width to cut the
    number of block passes; choosing by passes keeps the blocking monotonic
    in the budget).  When everything fits, both blocks cover the whole range
    and the contraction is a single pass.

    There is a hard floor -- the working set with one ``m`` on each side.
    A budget below it cannot be honoured; the smallest blocking is used anyway
    and a :class:`UserWarning` reports what the run will actually need.
    """
    if nspec is None:
        nspec = len(COUPLING_SPECTRA)

    def working_set(nb, nbp):
        return _coupling_working_set(
            nb, nbp, lmax, per_m, per_pair, nspec, inner_copies, producer_bytes
        )

    floor = working_set(1, 1)["total"]
    if max_bytes < floor:
        if (max_bytes, floor) in _WARNED_MEMORY_FLOORS:
            return 1, 1
        _WARNED_MEMORY_FLOORS.add((max_bytes, floor))
        warnings.warn(
            f"coupling memory budget of {max_bytes / 1024**2:.1f} MiB is below "
            f"the {floor / 1024**2:.1f} MiB this (ell, ellp, lmax) needs for "
            "the output kernels, the coefficient producer and one m on each "
            "side; proceeding with single-m blocks (warned once per process "
            "for this budget)",
            UserWarning,
            stacklevel=2,
        )
        return 1, 1
    # Everything that does not scale with the block widths.
    avail = max_bytes - working_set(0, 0)["total"]

    def widest_nb(nbp):
        # The largest nb with working_set(nb, nbp) <= max_bytes: the terms
        # are linear in nb at fixed nbp.
        rest = avail - inner_copies * nbp * per_m
        return min(n_m, int(rest // (per_m + nbp * per_pair))) if rest > 0 else 0

    # Candidate primed widths: the target and its halvings, plus -- only with
    # the whole central range in one block -- its doublings up to n_mp.  Each
    # candidate's passes can only fall as the budget grows, so taking the one
    # with the fewest passes (ties: fewest central blocks, then the widest
    # nbp) is monotonic in the budget; keeping only the first nbp that fits
    # was not.
    nbp = min(n_mp, _INNER_BLOCK_TARGET)
    candidates = []
    trial = nbp
    while True:
        candidates.append((trial, widest_nb(trial)))
        if trial == 1:
            break
        trial = max(1, trial // 2)
    trial = nbp
    while trial < n_mp:
        trial = min(n_mp, 2 * trial)
        if working_set(n_m, trial)["total"] > max_bytes:
            break
        candidates.append((trial, n_m))

    best = None
    for nbp, nb in candidates:
        if nb < 1:
            continue
        central_blocks = -(-n_m // nb)
        key = (central_blocks * -(-n_mp // nbp), central_blocks, -nbp)
        if best is None or key < best[0]:
            best = (key, nb, nbp)
    if best is None:
        return 1, 1
    _, nb, nbp = best
    return nb, nbp


def contract_coupling_block(
    left,
    right,
    central_field: Sequence[int],
    prime_field: Sequence[int],
    pair_mask=None,
    xp=np,
    output_pairs: Sequence[tuple[int, int]] | None = None,
    out=None,
):
    r"""
    The contraction of one ``(m, m')`` block of the ACC precompute: the
    per-``L`` matmul that builds ``Theta`` and the real gram that turns it
    into the kernel increment.

    .. math::

        \Theta^{s}(m, m', L) = \sum_M X^{a(s)}_{m L M}\, \bar Y^{b(s)}_{m' L M},
        \qquad
        \Delta K^{s_1 s_2}(L_1, L_2) = \mathrm{Re} \sum_{(m, m') \in \text{block}}
            \Theta^{s_1}(m, m', L_1)\, \overline{\Theta^{s_2}(m, m', L_2)} .

    Parameters
    ----------
    left : array
        ``(nfields, lmax, na, ncoef)`` complex, the central coefficients of
        the block (:func:`_pack_block` with ``conjugate=False``).
    right : array
        ``(nfields, lmax, ncoef, nbj)`` complex, the **conjugated** primed
        coefficients (:func:`_pack_block` with ``conjugate=True``, a
        transposed view; any strides ``matmul`` accepts).
    central_field, prime_field : sequence of int
        Field index ``(T, E, B) = (0, 1, 2)`` of the central and primed leg
        of every output channel (:data:`_CENTRAL_FIELD` / :data:`_PRIME_FIELD`
        restricted to the requested spectra); ``nspec = len(central_field)``.
    pair_mask : array of bool, optional
        ``(na, nbj)``: the ``(m, m')`` pairs of the block that contribute
        (term selection).  ``None`` keeps every pair, at no cost.
    output_pairs : sequence of (int, int), optional
        Restrict the output to these ``(k1, k2)`` channel-index pairs
        (indices into ``central_field``/``prime_field``) instead of the full
        ``nspec x nspec`` square.  ``Theta`` (``flats``) is then built only
        for the channels that actually appear in ``output_pairs``, and the
        gram (``dgemm``) only for the pairs themselves -- no symmetric
        mirroring, since an explicit pair list need not be symmetric.
        ``None`` (default) is the full square, bit-identical to before this
        argument existed.
    xp : module
        Array namespace, ``numpy`` by default.  Only operations that exist
        identically in ``numpy`` and ``jax.numpy`` are used (``matmul``,
        ``reshape``, ``real``/``imag``, ``stack``, ``where``, ``.T``): no
        in-place writes, no ``out=``.  Passing ``xp=jax.numpy`` with
        ``left``/``right`` (and ``pair_mask``) already on the device is the
        intended GPU path; the producers of the coefficients (the SHTs and
        banded Legendre transforms of :mod:`cmbcov.grid`)
        stay on the CPU and the blocks are moved over one at a time.
    out : ndarray, optional
        numpy only: a ``(nspec, nspec, lmax, lmax)`` float64 accumulator the
        increment is added into, one gram at a time, instead of being
        returned as a new array -- what :func:`_accumulate_coupling_kernels`
        uses, so that the only kernel-sized transient is one ``lmax x lmax``
        gram rather than the stacked increment and its rows.  ``out +=
        increment`` elementwise either way, so the result is bit-identical.

    Returns
    -------
    array
        ``(nspec, nspec, lmax, lmax)`` float64 kernel increment, ``[s1, s2]``
        for ``s2 < s1`` the transpose of ``[s2, s1]``; with ``out``, ``out``
        after the increment has been added.

    Notes
    -----
    The ``Re`` of the gram needs no complex GEMM: with ``(Re, Im)``
    interleaved along the flattened ``(m, m')`` axis,

        ``sum_f (Re a Re b + Im a Im b) = Re sum_f a conj(b)``

    is a real dot product over that doubled axis, one ``dgemm`` per channel
    pair at half the flops of a ``zgemm``.  ``stack((real, imag), -1)``
    builds exactly the memory layout the in-place version obtained from a
    ``float64`` view of the complex slab, so the numpy default path is
    bit-identical to it (asserted in ``tests/test_acc_term_selection.py``
    and on the golden kernels).  The cost is one transient complex ``Theta``
    of one channel next to the float slab.
    """
    nspec = len(central_field)
    if len(prime_field) != nspec:
        raise ValueError("central_field and prime_field must have the same length")
    lmax, na = left.shape[1], left.shape[2]
    nbj = right.shape[3]
    npair = na * nbj
    mask_flat = None
    if pair_mask is not None:
        mask_flat = xp.asarray(pair_mask, dtype=bool).reshape(1, npair)

    def theta_flat(k: int):
        a, b = central_field[k], prime_field[k]
        theta = xp.matmul(left[a], right[b]).reshape(lmax, npair)
        if mask_flat is not None:
            theta = xp.where(mask_flat, theta, xp.zeros((), dtype=theta.dtype))
        # (Re, Im) interleaved along the flattened (m, m') axis.
        return xp.stack((theta.real, theta.imag), axis=-1).reshape(lmax, 2 * npair)

    if out is not None:
        # numpy accumulation: the same grams, added into out as they come.
        if output_pairs is None:
            flats = [theta_flat(k) for k in range(nspec)]
            for k1 in range(nspec):
                for k2 in range(k1, nspec):
                    gram = xp.matmul(flats[k1], flats[k2].T)
                    out[k1, k2] += gram
                    if k2 != k1:
                        out[k2, k1] += gram.T
                    del gram
            return out
        needed = sorted({k for pair in output_pairs for k in pair})
        flat_of = {k: theta_flat(k) for k in needed}
        for k1, k2 in dict.fromkeys(output_pairs):
            out[k1, k2] += xp.matmul(flat_of[k1], flat_of[k2].T)
        return out

    if output_pairs is None:
        flats = [theta_flat(k) for k in range(nspec)]
        blocks: list[list] = [[None] * nspec for _ in range(nspec)]
        for k1 in range(nspec):
            for k2 in range(k1, nspec):
                gram = xp.matmul(flats[k1], flats[k2].T)
                blocks[k1][k2] = gram
                if k2 != k1:
                    blocks[k2][k1] = gram.T
        return xp.stack([xp.stack(row) for row in blocks])

    # Explicit pair list: build Theta only for the channels referenced (as
    # either leg) by output_pairs, and the gram only for the pairs
    # themselves -- no assumption of symmetry.
    needed = sorted({k for pair in output_pairs for k in pair})
    flat_of = {k: theta_flat(k) for k in needed}
    out = np.zeros((nspec, nspec, lmax, lmax))
    for k1, k2 in output_pairs:
        out[k1, k2] = np.asarray(xp.matmul(flat_of[k1], flat_of[k2].T))
    return out


def _contiguous_runs(positions: Sequence[int]) -> Iterator[tuple[int, int, int]]:
    """``(source start, destination start, length)`` of the maximal runs of
    consecutive ascending values in ``positions``."""
    start = 0
    for k in range(1, len(positions) + 1):
        if k == len(positions) or positions[k] != positions[k - 1] + 1:
            yield positions[start], start, k - start
            start = k


class _CentralStore:
    r"""
    The central (``ell``) full-``M`` coefficient set of one precompute,
    built once and kept in a file on disk, for a set that does not fit the
    memory budget (:meth:`_CouplingPrecompute.compute`).

    The file holds a complex128 ``(nfields, lmax, n, ncoef)`` array -- the
    left-operand layout of :func:`_pack_block` -- whose third axis runs over
    ``order``, the kept orders in the paired ``+m, -m`` order the contraction
    iterates in, so that reading a block of consecutive orders is
    ``nfields * lmax`` contiguous reads of ``len(block) * ncoef`` values each,
    straight into the destination.  Plain ``pread``/``pwrite`` rather than a
    ``np.memmap``, so what the process holds is only the blocks it reads, and
    the file is kept out of the OS file cache (``F_NOCACHE`` on macOS,
    ``POSIX_FADV_DONTNEED`` after each fill and read elsewhere): every
    ``ellp`` reads it whole, once per outer block, so caching it buys little
    and, at 11.7 GiB next to a working set of the same size, pushed the
    machine into memory compression (survey mask, ``centralell = 250``,
    ``nside = 256``, 12 GiB budget, one run each on a shared 36 GB laptop:
    cached, 147 s of system time and ``Theta`` GEMMs at 0.56 s per call;
    uncached, with the read-ahead of :func:`_accumulate_coupling_kernels`,
    37 s and 0.13 s).

    ``directory`` is created by the caller (a fresh temporary directory) and
    is removed, with the file, by :meth:`close`.
    """

    FILENAME = "central_full_m.bin"

    def __init__(
        self,
        directory: str,
        order: Sequence[int],
        nfields: int,
        lmax: int,
    ) -> None:
        self.directory = directory
        self.path = os.path.join(directory, self.FILENAME)
        self.order = list(order)
        self.position = {i: p for p, i in enumerate(self.order)}
        self.n = len(self.order)
        self.nfields = int(nfields)
        self.lmax = int(lmax)
        self.ncoef = 2 * self.lmax - 1
        self.itemsize = np.dtype(np.complex128).itemsize
        self.nbytes = self.nfields * self.lmax * self.n * self.ncoef * self.itemsize
        self._fd: int | None = os.open(
            self.path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600
        )
        os.ftruncate(self._fd, self.nbytes)
        if _F_NOCACHE is not None:
            fcntl.fcntl(self._fd, _F_NOCACHE, 1)

    def _drop_cache(self) -> None:
        """Advise the OS to drop the file's cached pages (not macOS, where
        ``F_NOCACHE`` keeps them out in the first place)."""
        if _F_NOCACHE is None and hasattr(os, "posix_fadvise"):
            os.posix_fadvise(self._fd, 0, 0, os.POSIX_FADV_DONTNEED)

    def _offset(self, f: int, L: int, p: int) -> int:
        return ((f * self.lmax + L) * self.n + p) * self.ncoef * self.itemsize

    def _write(self, buf: np.ndarray, offset: int) -> None:
        view = memoryview(buf.reshape(-1).view(np.uint8))
        while len(view):
            done = os.pwrite(self._fd, view, offset)
            view = view[done:]
            offset += done

    def _read_into(self, buf: np.ndarray, offset: int) -> None:
        if not buf.flags.c_contiguous:  # reshape would read into a copy
            raise ValueError("the central store reads into contiguous buffers only")
        view = memoryview(buf.reshape(-1).view(np.uint8))
        while len(view):
            if hasattr(os, "preadv"):
                done = os.preadv(self._fd, [view], offset)
            else:  # pragma: no cover - platforms without preadv
                data = os.pread(self._fd, len(view), offset)
                done = len(data)
                view[:done] = data
            if done == 0:
                raise OSError(f"short read from the central store {self.path}")
            view = view[done:]
            offset += done

    def fill(self, provider: Callable[[int], np.ndarray], chunk: int) -> None:
        """Synthesise every order once, ``chunk`` at a time
        (:func:`_pack_block`, into one buffer reused for every chunk), and
        write it to the file."""
        chunk = max(1, min(int(chunk), self.n))
        buffer = np.empty(
            chunk * self.nfields * self.lmax * self.ncoef, dtype=np.complex128
        )
        for p0 in range(0, self.n, chunk):
            block = _pack_block(
                provider,
                self.order[p0 : p0 + chunk],
                self.lmax,
                conjugate=False,
                nfields=self.nfields,
                out=buffer,
            )
            for f in range(self.nfields):
                for L in range(self.lmax):
                    self._write(block[f, L], self._offset(f, L, p0))
            del block
        if _F_NOCACHE is None and hasattr(os, "posix_fadvise"):
            os.fsync(self._fd)  # dirty pages are not dropped
            self._drop_cache()

    def left(self, rows: Sequence[int], out: np.ndarray | None = None) -> np.ndarray:
        """The ``(nfields, lmax, len(rows), ncoef)`` block of the orders
        ``rows`` (indices ``m + ell``) -- :func:`_pack_block`'s
        ``conjugate=False`` layout -- read from the file, into the leading
        entries of the flat buffer ``out`` when given."""
        positions = [self.position[i] for i in rows]
        out = _block_array(out, self.nfields, self.lmax, len(rows))
        for p0, q0, k in _contiguous_runs(positions):
            for f in range(self.nfields):
                for L in range(self.lmax):
                    self._read_into(out[f, L, q0 : q0 + k], self._offset(f, L, p0))
        self._drop_cache()
        return out

    def right(self, cols: Sequence[int], out: np.ndarray | None = None) -> np.ndarray:
        """The conjugated ``(nfields, lmax, ncoef, len(cols))`` block of the
        orders ``cols`` -- :func:`_pack_block`'s ``conjugate=True`` form, a
        transposed view -- for ``ell == ellp``: :meth:`left`, conjugated in
        place."""
        out = self.left(cols, out)
        rows = out.reshape(self.nfields * self.lmax, -1)  # a view

        def conjugate(r0: int, r1: int) -> None:
            np.conjugate(rows[r0:r1], out=rows[r0:r1])

        run_row_blocks(
            conjugate,
            rows.shape[0],
            row_block_workers(out.size, get_optimal_nthreads()),
        )
        return out.transpose(0, 1, 3, 2)

    def close(self) -> None:
        """Close and delete the file and its directory (idempotent)."""
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        shutil.rmtree(self.directory, ignore_errors=True)


def _accumulate_coupling_kernels(
    central_fn: Callable[[int], np.ndarray],
    n_m: int,
    prime_fn: Callable[[int], np.ndarray],
    n_mp: int,
    lmax: int,
    max_bytes: int,
    spec_indices: Sequence[int] | None = None,
    nfields: int = 3,
    verbose: bool = False,
    keep_m: np.ndarray | None = None,
    keep_mp: np.ndarray | None = None,
    pair_mask: np.ndarray | None = None,
    output_pairs: Sequence[tuple[int, int]] | None = None,
    central_reader: Callable[[Sequence[int], np.ndarray], np.ndarray] | None = None,
    prime_reader: Callable[[Sequence[int], np.ndarray], np.ndarray] | None = None,
    producer_bytes: int | None = None,
) -> np.ndarray:
    r"""
    The whole ACC precompute for one ``(ell, ellp)`` pair, as blocked BLAS.

    Both grids reduce to the same two contractions on complex full-``M``
    coefficients (:func:`_coefficient_provider`).  With ``X`` the central
    coefficient stack and ``Y`` the primed one (indices ``[field, L, m, M]``,
    complex128, ``2 * lmax - 1`` values of ``M``),

    .. math::

        \Theta^{s}(m, m', L) = \sum_M X^{a(s)}_{m L M}\, \bar Y^{b(s)}_{m' L M}
        \qquad\text{(one GEMM per } L\text{)}

        K^{s_1 s_2}(L_1, L_2) = \mathrm{Re} \sum_{m m'}
            \Theta^{s_1}(m, m', L_1)\, \overline{\Theta^{s_2}(m, m', L_2)}
        \qquad\text{(one GEMM per channel pair)} ,

    with ``(a, b)`` given by :data:`_CENTRAL_FIELD` / :data:`_PRIME_FIELD`.
    ``Theta`` is kept complex on both grids, as Eq. 20 requires.  The naive
    form of the first line is the ``(m, m')`` double loop this replaced; the
    second is :meth:`_CouplingPrecompute._theta_to_coupling_kernels`.

    The ``(m, m')`` plane is blocked (:func:`_coupling_block_sizes`) and the
    kernels are accumulated block by block, so the full ``Theta`` -- 10.3 GB at
    ``ell = ellp = 250``, ``lmax = 512`` -- is never materialised.

    The ``Re`` of the second line needs no complex GEMM and no split into real
    and imaginary parts: viewing the complex ``Theta`` slab as float64
    interleaves ``(Re, Im)`` along the flattened ``(m, m')`` axis, and

        ``sum_f (Re a Re b + Im a Im b) = Re sum_f a conj(b)``

    is then literally the real dot product over that doubled axis.  So one
    ``dgemm`` on the raw buffer gives the kernel exactly, at half the flops of
    a ``zgemm`` and with no copy of ``Theta`` at all.

    ``spec_indices`` restricts the output to a subset of
    :data:`COUPLING_SPECTRA` (indices into it); ``None`` (the default) means
    all five, i.e. the shipped 25-kernel behaviour.  ``nfields`` is the
    number of fields (T, E, B) that ``central_fn``/``prime_fn`` actually
    provide per ``m`` -- 3 normally, or 1 when the caller has determined
    (:func:`_resolve_spectra`) that every requested kernel needs only T on
    both sides, in which case the coefficient providers were built with
    ``t_only=True`` and never expand fields 1, 2.

    The per-block arithmetic lives in :func:`contract_coupling_block`; this
    function owns the blocking, the iteration order and the accumulation.

    **Term selection** (:mod:`cmbcov.term_selection`).
    ``keep_m`` / ``keep_mp`` (bool, index ``m + ell`` / ``m' + ellp``)
    restrict the orders iterated over -- the paired ``+m, -m`` order of
    :func:`_paired_m_order` is kept, filtered to the kept indices, so a
    reflecting provider still serves ``-m`` from the transform of ``+m``
    whenever both are kept (the mode power is even in ``m``, so in practice
    they always are) -- and ``pair_mask`` (bool, ``(2 ell + 1, 2 ellp +
    1)``) zeroes the ``Theta`` entries of the pairs that are not kept before
    the gram; a block pair whose mask is entirely False is skipped without
    packing its primed side.  All three default to ``None``: every order and
    every pair, the unchanged full contraction.

    ``output_pairs`` (indices into ``spec_indices``, i.e. positions 0 ..
    ``nspec - 1``) restricts both the ``Theta`` built and the kernels
    accumulated to an explicit list of ordered channel pairs -- see
    :func:`contract_coupling_block`.  ``None`` (default) is the full square,
    bit-identical to before this argument existed; it also narrows the
    ``Theta`` term of the memory-budget accounting
    (:func:`_coupling_block_sizes`) to the channels actually built (the
    accumulator is the full ``nspec x nspec`` square either way).

    **Memory.**  ``max_bytes`` bounds :func:`_coupling_working_set` of the
    chosen blocking: the accumulator and one gram, the resident coefficient
    blocks, the ``Theta`` slab and ``producer_bytes`` (what the coefficient
    producer holds beyond the block it fills; ``None`` charges three
    per-``m`` sets).  Each block's increment is added into the accumulator
    gram by gram (:func:`contract_coupling_block`, ``out``).

    **Loop order.**  With ``central_reader=None`` (the streamed path: every
    coefficient comes from a provider that synthesises it) the central
    blocks are the outer loop and every primed block is rebuilt once per
    central block.  A ``central_reader`` -- ``(rows, out) -> left block``
    (:func:`_pack_block`'s ``conjugate=False`` layout, written into the flat
    buffer ``out``), for a central set
    that is cheap to re-read (held in RAM, or :class:`_CentralStore` on
    disk) -- swaps the loops: the primed blocks are the outer loop, so every
    primed coefficient is synthesised exactly once, and the central blocks
    are re-read inside, alternately forwards and backwards, one block read
    ahead in a helper thread (the budget is charged for two central blocks,
    :func:`_coupling_block_sizes` ``inner_copies=2``).  ``prime_reader`` (``(cols, out) -> right block``,
    ``conjugate=True`` layout) likewise serves the primed side, for
    ``ell == ellp`` from the store.  Only the summation order of the
    kernels changes, i.e. the result agrees to rounding.

    Returns
    -------
    ndarray
        ``(nspec, nspec, lmax, lmax)`` float64, ``[s1, s2]`` being the kernel
        for ``COUPLING_SPECTRA[spec_indices[s1]] x
        COUPLING_SPECTRA[spec_indices[s2]]``, ``nspec = len(spec_indices)``
        (25 = 5x5 when ``spec_indices`` is ``None``).
    """
    if spec_indices is None:
        spec_indices = list(range(len(COUPLING_SPECTRA)))
    nspec = len(spec_indices)
    ncoef = 2 * lmax - 1
    itemsize = np.dtype(np.complex128).itemsize

    # The kernels sum over every (m, m') pair, so the iteration order is free;
    # a reflecting (GL) provider exploits a +m / -m pairing (see
    # _coefficient_provider).  With a selection the order is filtered, not
    # rebuilt, so kept +m / -m stay adjacent.
    m_order = _paired_m_order(n_m)
    mp_order = _paired_m_order(n_mp)
    if keep_m is not None:
        m_order = [i for i in m_order if keep_m[i]]
    if keep_mp is not None:
        mp_order = [j for j in mp_order if keep_mp[j]]
    n_kept, n_kept_p = len(m_order), len(mp_order)
    if pair_mask is not None:
        pair_mask = np.asarray(pair_mask, dtype=bool)
        if pair_mask.shape != (n_m, n_mp):
            raise ValueError(
                f"pair_mask must have shape {(n_m, n_mp)}; got {pair_mask.shape}"
            )

    # Theta is built only for the channels an explicit pair list references
    # (all of them otherwise); the accumulator is always the full square.
    if output_pairs is not None:
        n_channels_used = len({k for pair in output_pairs for k in pair})
    else:
        n_channels_used = nspec

    per_m = nfields * lmax * ncoef * itemsize
    # The float (Re, Im) slab of Theta, (n_channels_used, lmax, 2 nb nbp)
    # float64 -- the same bytes as the complex slab -- plus the one complex
    # channel of Theta being converted into it (contract_coupling_block).
    per_pair = (n_channels_used + 1) * lmax * itemsize
    primed_outer = central_reader is not None
    # _coupling_block_sizes returns (outer width, inner width).  Primed
    # outer, the next central block is read ahead: two resident at once.
    inner_copies = 2 if primed_outer else 1
    outer, inner = _coupling_block_sizes(
        n_kept_p if primed_outer else n_kept,
        n_kept if primed_outer else n_kept_p,
        lmax,
        per_m,
        per_pair,
        max_bytes,
        nspec=nspec,
        inner_copies=inner_copies,
        producer_bytes=producer_bytes,
    )
    nb, nbp = (inner, outer) if primed_outer else (outer, inner)

    if verbose:
        plan = _coupling_working_set(
            outer, inner, lmax, per_m, per_pair, nspec, inner_copies, producer_bytes
        )
        detail = ", ".join(
            f"{name} {plan[name] / 1024**3:.2f}"
            for name in ("outer", "inner", "theta", "kernels", "producer")
        )
        if primed_outer:
            print(
                f"  contraction blocking: m' in {nbp}-blocks "
                f"({-(-n_kept_p // nbp)} of them, outer, built once), "
                f"m in {nb}-blocks ({-(-n_kept // nb)} of them, re-read)"
            )
        else:
            print(
                f"  contraction blocking: m in {nb}-blocks "
                f"({-(-n_kept // nb)} of them), "
                f"m' in {nbp}-blocks ({-(-n_kept_p // nbp)} of them)"
            )
        print(
            f"  planned working set {plan['total'] / 1024**3:.2f} of "
            f"{max_bytes / 1024**3:.2f} GiB ({detail} GiB)"
        )

    central_fields = [_CENTRAL_FIELD[k] for k in spec_indices]
    prime_fields = [_PRIME_FIELD[k] for k in spec_indices]

    # Every block is written into one of these buffers, allocated once here
    # (the plan's "outer" and "inner" terms) rather than per block: freeing
    # a multi-GiB block per pass and allocating the next let the system
    # allocator keep the freed ones (macOS caches large blocks), which
    # pushed the footprint to about twice a 6 GiB budget on the survey mask.
    entries_per_m = nfields * lmax * ncoef
    outer_buffer = np.empty(outer * entries_per_m, dtype=np.complex128)
    inner_buffers = [
        np.empty(inner * entries_per_m, dtype=np.complex128)
        for _ in range(inner_copies)
    ]
    right_buffer = outer_buffer if primed_outer else inner_buffers[0]

    def pack_right(cols):
        if prime_reader is not None:
            return prime_reader(cols, right_buffer)
        return _pack_block(
            prime_fn, cols, lmax, conjugate=True, nfields=nfields, out=right_buffer
        )

    def block_mask_of(rows, cols):
        """``(skip, mask)`` of one block pair under term selection."""
        if pair_mask is None:
            return False, None
        mask = pair_mask[np.ix_(rows, cols)]
        if not mask.any():
            return True, None
        return False, (None if mask.all() else mask)

    out = np.zeros((nspec, nspec, lmax, lmax))
    skipped = 0
    if not primed_outer:
        for i0 in range(0, n_kept, nb):
            rows = m_order[i0 : i0 + nb]
            left = _pack_block(
                central_fn,
                rows,
                lmax,
                conjugate=False,
                nfields=nfields,
                out=outer_buffer,
            )

            for j0 in range(0, n_kept_p, nbp):
                cols = mp_order[j0 : j0 + nbp]
                skip, block_mask = block_mask_of(rows, cols)
                if skip:
                    skipped += 1
                    continue
                right = pack_right(cols)
                contract_coupling_block(
                    left,
                    right,
                    central_fields,
                    prime_fields,
                    block_mask,
                    output_pairs=output_pairs,
                    out=out,
                )
                del right
            del left
    else:
        inner_starts = list(range(0, n_kept, nb))
        reads = 0  # central blocks read so far: they alternate between buffers

        def read_next(rows):
            nonlocal reads
            buffer = inner_buffers[reads % inner_copies]
            reads += 1
            return reader.submit(central_reader, rows, buffer)

        # One reader thread keeps the next central block coming while the
        # current one is contracted (preadv, the copies and the BLAS calls
        # all release the GIL); the first read of each pass also overlaps
        # the synthesis of the primed block.
        with ThreadPoolExecutor(max_workers=1) as reader:
            for jb, j0 in enumerate(range(0, n_kept_p, nbp)):
                cols = mp_order[j0 : j0 + nbp]
                # Serpentine over the central blocks: the last block read in
                # one pass is the first of the next.
                starts = inner_starts[::-1] if jb % 2 else inner_starts
                tasks = []
                for i0 in starts:
                    rows = m_order[i0 : i0 + nb]
                    skip, block_mask = block_mask_of(rows, cols)
                    if skip:
                        skipped += 1
                    else:
                        tasks.append((rows, block_mask))
                if not tasks:
                    continue  # the primed block is never packed
                pending = read_next(tasks[0][0])
                right = pack_right(cols)
                for k, (_, block_mask) in enumerate(tasks):
                    left = pending.result()
                    # Into the other buffer: the one read two blocks ago,
                    # whose contraction has returned.
                    pending = read_next(tasks[k + 1][0]) if k + 1 < len(tasks) else None
                    contract_coupling_block(
                        left,
                        right,
                        central_fields,
                        prime_fields,
                        block_mask,
                        output_pairs=output_pairs,
                        out=out,
                    )
                    del left
                del right

    if verbose and pair_mask is not None:
        n_blocks = -(-n_kept // nb) * -(-n_kept_p // nbp)
        print(f"  term selection: {skipped} of {n_blocks} block pairs skipped")
    return out


#: Working-set budget, in bytes, of one ``ell1`` chunk of the batched ACC
#: assembly (:func:`_acc_band`).  A chunk of ``n`` multipoles holds three
#: ``(n, size)`` float64 arrays -- the two spectrum windows and the
#: window-times-kernel product -- so ``n = budget // (24 * size)``: 5461
#: multipoles for the ``size = 512`` kernel of ``nside_acc = 256``, 1365 for
#: ``size = 2048``.  Fixed, not a caller knob: ``compute_covariance_term``
#: has no memory argument, and this is two orders of magnitude below the
#: default :data:`DEFAULT_COUPLING_MEMORY_GB` of the precompute.
_ASSEMBLY_CHUNK_BYTES = 64 * 1024**2


def acc_window_pad(kernel_size: int, centralell: int) -> int:
    r"""
    How far past ``lmax`` the spectra must extend so that no reported ACC
    element loses kernel weight: ``max(0, kernel_size - 1 - centralell)``.

    Element ``(ell1, ell2)`` uses the kernel of ``(l*, l* + |ell2 - ell1|)``
    translated by ``shift = min(ell1, ell2) - l*`` (Eq. 33): kernel index
    ``i = 0 .. S-1`` stands for ``L = shift + i``. The largest reported
    ``min(ell1, ell2)`` is ``lmax - 1`` (on the diagonal), whose window ends
    at ``L = lmax - 1 - l* + S - 1``. The spectra must therefore hold
    ``lmax + S - 1 - l*`` multipoles, independent of ``dmax``. The low end
    (``L < 0`` for ``min(ell1, ell2) < l*``) is not a clip: there is no
    spectrum there.

    For the survey cache, ``S = 2 nside_acc = 512`` and ``l* = 250``, this is
    261, not ``nside_acc = 256``. Padding by 256 would still cut kernel
    columns 507..511 from the top five rows (1.4e-5 of the kernel weight at
    most, measured over its 20 diagonals).
    """
    return max(0, int(kernel_size) - 1 - int(centralell))


def acc_internal_lmax(lmax: int, kernel_size: int, centralell: int) -> int:
    """
    ``lmax_int = lmax + acc_window_pad(kernel_size, centralell)``: the number
    of multipoles the spectra of an ACC computation must hold so that no
    element with ``ell1, ell2 < lmax`` has its coupling window cut (see
    :func:`acc_window_pad`). The covariance is still reported to ``lmax``.
    """
    return int(lmax) + acc_window_pad(kernel_size, centralell)


def acc_cached_kernel_size(
    kernel_dir: str,
    centralell: int,
    dmax: int,
    pairs: Iterable[tuple[str, str]] | None = None,
) -> int | None:
    """
    Largest side length of the ACC kernel files present under
    ``<kernel_dir>/covariance_coupling/`` for the pairs ``(centralell,
    centralell + d)``, ``d < dmax``, read from the ``.npy`` headers alone
    (no array is loaded). ``pairs`` restricts the Stokes pairs looked at
    (default: every ordered pair of :data:`COUPLING_SPECTRA`). A pair whose
    new-name file is absent is also tried under its old on-disk name
    (:data:`LEGACY_CHANNEL_NAMES`), and under the transposed pair's own and
    legacy names (docs/theory/bmode_kernels.md -- transposing
    does not change a square kernel's side length), so a cache written
    before B-mode support, or one holding only the other orientation of a
    pair, reports the same size as its new-named/natural-orientation
    equivalent. ``None`` when no such file exists at all; the kernel load
    then reports what is missing.
    """
    if pairs is None:
        pairs = [(a, b) for a in COUPLING_SPECTRA for b in COUPLING_SPECTRA]
    pairs = list(pairs)
    size = None

    def header_side(path: str) -> int | None:
        try:
            with open(path, "rb") as handle:
                version = np.lib.format.read_magic(handle)
                if version == (1, 0):
                    header = np.lib.format.read_array_header_1_0(handle)
                else:
                    header = np.lib.format.read_array_header_2_0(handle)
                shape = header[0]
        except (OSError, ValueError):
            return None
        return max(shape) if shape else 0

    for diagonal_offset in range(int(dmax)):
        for pair in pairs:
            side = None
            for candidate_path, _transposed in acc_cache.coupling_kernel_candidates(
                kernel_dir,
                pair,
                centralell,
                centralell + diagonal_offset,
                COUPLING_CHANNELS,
                LEGACY_CHANNEL_NAMES,
            ):
                side = header_side(candidate_path)
                if side is not None:
                    break
            if side is None:
                continue
            size = side if size is None else max(size, side)
    return size


def _missing_kernel_pairs(
    kernel_dir: str,
    centralell: int,
    ell_prime: int,
    pairs: Iterable[tuple[str, str]],
) -> list[tuple[str, str]]:
    """
    The pairs of ``pairs`` that :func:`~cmbcov.approximations.acc_cache.load_coupling_kernels`
    cannot serve at ``(centralell, ell_prime)``: no file under the pair's own
    name, its legacy name or those of its transpose, or only files the
    manifest's pair record does not list
    (:func:`~cmbcov.approximations.acc_cache.locate_coupling_kernel`, the
    loader's own rule). File existence only; nothing is loaded.
    """
    manifest = acc_cache.read_coupling_manifest(kernel_dir, centralell, ell_prime)
    missing = []
    for pair in pairs:
        found, _ = acc_cache.locate_coupling_kernel(
            kernel_dir,
            tuple(pair),
            centralell,
            ell_prime,
            COUPLING_CHANNELS,
            LEGACY_CHANNEL_NAMES,
            manifest or False,
        )
        if found is None:
            missing.append(tuple(pair))
    return missing


#: Why a B-mode cache can lack pairs a run needs, and what to run: shared by
#: :func:`require_acc_cache_pairs` and :meth:`ACCStrategy._wick_plan`.
_BMODE_PAIRS_HELP = (
    "The kernel pairs a run with a B observable needs changed after cmbcov "
    "0.3.0: every off-diagonal block now needs the pairs of both of its "
    "orientations, Cov(a, b) and Cov(b, a), so that both of its triangles "
    "are exact at centralell (bmode_wick.required_kernel_pairs: 24, 25 or "
    "45 pairs where 0.3.0 asked for 17, 18, 31 or 40; "
    "docs/theory/bmode_kernels.md, Sect. 6). A B-mode cache precomputed by "
    "cmbcov 0.3.0 or earlier therefore lacks some; so does a cache built for "
    "C^TB = C^EB = 0 when this run's TB or EB spectra are non-zero. Rerun "
    "`cmbcov-precompute` on the parameter file, or add only the missing "
    "pairs to this cache with precompute_acc_kernels(<mask>, <kernel_dir>, "
    "centralell=..., dmax=..., pairs=[...]) and the nside, lw and grid it "
    "was built with: the kernels already there are kept, and the manifests "
    "merge (acc_cache.write_coupling_kernels)."
)


def require_acc_cache_pairs(
    kernel_dir: str,
    centralell: int,
    dmax: int,
    pairs: Iterable[tuple[str, str]],
) -> None:
    """
    Raise ``OSError`` if the on-disk cache does not cover every one of
    ``pairs``, for every diagonal ``(centralell, centralell + d)``,
    ``d < dmax``, naming every missing pair.

    Run early (:class:`~cmbcov.spectra.SpectraLoader`) against the pairs a
    B-mode ``observables`` list needs
    (:func:`~cmbcov.bmode_wick.required_kernel_pairs`), so that a run is
    refused before anything else rather than failing on its first block
    (or, before the both-orientation requirement, silently computing one
    triangle of some blocks from the other orientation's kernel). A pair
    counts as covered when the loader would serve it
    (:func:`_missing_kernel_pairs`): under its own or legacy name, or
    stored only as its transpose. The message lists the missing pairs (the
    union over the diagonals, and the diagonals affected), says that the
    B-mode requirement changed after cmbcov 0.3.0 and what to run, and
    ends with the loader's own error for the first missing pair (which
    names its file and, where it applies, the manifest's spectra or a
    left-over file). A legacy text-format cache raises the loader's
    ``FileNotFoundError`` unchanged. A no-op for an empty ``pairs``.
    """
    pairs = sorted({tuple(pair) for pair in pairs})
    if not pairs:
        return
    missing: dict[tuple[str, str], list[int]] = {}
    for diagonal_offset in range(int(dmax)):
        ell_prime = centralell + diagonal_offset
        for pair in _missing_kernel_pairs(kernel_dir, centralell, ell_prime, pairs):
            missing.setdefault(pair, []).append(ell_prime)
    if not missing:
        return
    first_pair = min(missing, key=lambda p: (missing[p][0], p))
    try:
        acc_cache.load_coupling_kernels(
            kernel_dir,
            centralell,
            missing[first_pair][0],
            COUPLING_CHANNELS,
            [first_pair],
            pairs=[first_pair],
            legacy_names=LEGACY_CHANNEL_NAMES,
        )
        loader_error = None
    except FileNotFoundError:
        raise
    except OSError as err:
        loader_error = str(err)
    affected = sorted({lp for lps in missing.values() for lp in lps})
    where = (
        f"on every diagonal (centralell, centralell + d), d < {int(dmax)}"
        if len(affected) == int(dmax)
        else f"at (centralell, l') for l' in {affected}"
    )
    listed = ", ".join("x".join(pair) for pair in sorted(missing))
    raise OSError(
        f"ACC coupling kernels not found under {kernel_dir!r} (centralell "
        f"{centralell}): this run needs {len(pairs)} kernel pairs, and "
        f"{len(missing)} of them are missing {where}: {listed}. "
        f"{_BMODE_PAIRS_HELP}"
        + (f" First missing file: {loader_error}" if loader_error else "")
    )


def _unit_sum_kernel(coupling_matrix: np.ndarray) -> np.ndarray:
    """
    Eq. 23 normalisation, exactly as :meth:`ACCStrategy.compute_acc_term`
    applies it per call (same expression, so bit-identical), hoisted out of
    the element loop.
    """
    if not np.isclose(np.sum(coupling_matrix), 1.0):
        coupling_matrix = coupling_matrix / np.sum(coupling_matrix)
    return coupling_matrix


def _padded_spectrum(
    spectrum: np.ndarray, central_ell: int, n_read: int, size: int
) -> np.ndarray:
    """
    ``padded[central_ell + L] = spectrum[L]`` for ``0 <= L < n_read``, zero
    elsewhere, of length ``central_ell + n_read + size``.

    In :meth:`ACCStrategy.compute_acc_term` kernel index ``i`` stands for
    ``L = i + shift`` with ``shift = min(ell1, ell2) - central_ell``. In
    padded coordinates that is index ``min(ell1, ell2) + i``, so the whole
    window is ``padded[m : m + size]`` with ``m = min(ell1, ell2)``; the
    leading zeros stand in for the low-end runoff (``L < 0``). With
    ``n_read = lmax + acc_window_pad(size, central_ell)`` (:func:`acc_window_pad`)
    no window of an ``m < lmax`` reaches past ``spectrum[n_read - 1]``, so
    nothing is clipped at the high end.
    """
    padded = np.zeros(central_ell + n_read + size)
    padded[central_ell : central_ell + n_read] = spectrum[:n_read]
    return padded


def _acc_band(
    kernel: np.ndarray,
    padded_1: np.ndarray,
    padded_2: np.ndarray,
    n_ell: int,
    chunk: int,
) -> np.ndarray:
    """
    ``out[m] = padded_1[m:m+S] @ kernel @ padded_2[m:m+S]`` for
    ``m = 0 .. n_ell - 1``, i.e. :meth:`ACCStrategy.compute_acc_term` for every
    ``min(ell1, ell2) = m`` of one diagonal (it depends on ``(ell1, ell2)``
    only through ``m``, the kernel and the spectra), as one BLAS product per
    chunk of ``chunk`` multipoles instead of one Python call per element.

    Rows of the window matrices multiply the kernel's rows from the left and
    its columns from the right, preserving the ``cl1 @ K @ cl2`` index order
    of the scalar path for a non-symmetric kernel.
    """
    size = kernel.shape[0]
    if kernel.shape != (size, size):
        raise ValueError(f"ACC coupling kernel must be square, got {kernel.shape}")
    windows_1 = np.lib.stride_tricks.sliding_window_view(padded_1, size)
    same = padded_2 is padded_1
    windows_2 = (
        windows_1 if same else np.lib.stride_tricks.sliding_window_view(padded_2, size)
    )
    out = np.empty(n_ell)
    for start in range(0, n_ell, chunk):
        stop = min(n_ell, start + chunk)
        w1 = np.ascontiguousarray(windows_1[start:stop])
        w2 = w1 if same else np.ascontiguousarray(windows_2[start:stop])
        out[start:stop] = np.einsum("ij,ij->i", w1 @ kernel, w2)
    return out


#: Memory the batched ACC assembly (:meth:`ACCStrategy.compute_covariance_terms`)
#: may hold for one batch of blocks: their flattened diagonals,
#: ``(2 dmax - 1) x lmax`` doubles each, plus the bands of their kernel
#: products on the current diagonal. A block alone is never split, so a
#: batch always holds at least one. At ``lmax = 3500``, ``dmax = 100`` a
#: block takes 5.6 MB, so a batch holds ~90 blocks.
_ASSEMBLY_BATCH_BYTES = 512 * 1024**2


def _spectrum_identity(spectrum: np.ndarray) -> tuple:
    """
    Content identity of a spectrum array: dtype, shape and a blake2b digest
    of its bytes. Two arrays with the same identity give the same padded
    spectrum (:func:`_padded_spectrum`) and so bit-identical kernel products;
    the batched assembly shares products between blocks by this identity, not
    by ``id()`` (a run's ``cl`` often holds equal arrays under different keys,
    e.g. ``TE`` and ``ET`` of one frequency pair).
    """
    array = np.ascontiguousarray(spectrum)
    digest = hashlib.blake2b(memoryview(array).cast("B"), digest_size=16)
    return (array.dtype.str, array.shape, digest.hexdigest())


def _identity_memo() -> Callable[[np.ndarray], tuple]:
    """
    :func:`_spectrum_identity`, memoised by ``id()`` for the duration of one
    assembly call: the arrays are referenced by the call's ``cl`` throughout,
    so an ``id`` cannot be reused within it. Never kept between calls, so a
    spectrum changed in place between two calls is hashed afresh.
    """
    identities: dict[int, tuple] = {}

    def identity(array: np.ndarray) -> tuple:
        key = id(array)
        if key not in identities:
            identities[key] = _spectrum_identity(array)
        return identities[key]

    return identity


def _acc_bands(
    products: dict,
    n_ell: int,
) -> dict:
    """
    :func:`_acc_band` for many products on one diagonal:
    ``products[key] = (kernel, padded_1, padded_2)`` gives
    ``bands[key] = _acc_band(kernel, padded_1, padded_2, n_ell, chunk)``,
    bit for bit, with the left factor ``w1 @ kernel`` of each chunk (the
    ``n_ell x S x S`` matrix product, nearly all of the cost) computed once
    per distinct ``(padded_1, kernel)`` pair of arrays and shared by every
    product with that left factor. Each product is otherwise evaluated
    exactly as :func:`_acc_band` does: same chunks (``chunk`` from
    :data:`_ASSEMBLY_CHUNK_BYTES` and the kernel size), same contiguous
    window copies, same ``np.einsum`` on the same operands.
    """
    groups: dict[tuple[int, int], list] = {}
    for key, (kernel, padded_1, _) in products.items():
        groups.setdefault((id(padded_1), id(kernel)), []).append(key)
    bands = {}
    for keys in groups.values():
        kernel, padded_1, _ = products[keys[0]]
        size = kernel.shape[0]
        if kernel.shape != (size, size):
            raise ValueError(f"ACC coupling kernel must be square, got {kernel.shape}")
        chunk = max(1, _ASSEMBLY_CHUNK_BYTES // (24 * size))
        windows_1 = np.lib.stride_tricks.sliding_window_view(padded_1, size)
        rights = []
        for key in keys:
            padded_2 = products[key][2]
            same = padded_2 is padded_1
            windows_2 = (
                windows_1
                if same
                else np.lib.stride_tricks.sliding_window_view(padded_2, size)
            )
            out = np.empty(n_ell)
            bands[key] = out
            rights.append((out, same, windows_2))
        for start in range(0, n_ell, chunk):
            stop = min(n_ell, start + chunk)
            w1 = np.ascontiguousarray(windows_1[start:stop])
            left = w1 @ kernel
            for out, same, windows_2 in rights:
                w2 = w1 if same else np.ascontiguousarray(windows_2[start:stop])
                out[start:stop] = np.einsum("ij,ij->i", left, w2)
    return bands


def _group_blocks(factors: list[set], batch_size: int) -> list[list[int]]:
    """
    Split blocks ``0 .. len(factors) - 1`` into batches of at most
    ``batch_size``, greedily, so that blocks sharing left factors
    (``factors[i]``, the ``(left spectrum, kernel)`` pairs of block ``i``'s
    kernel products) land in the same batch: each batch is grown by the
    block adding the fewest left factors not already in it, among those the
    one sharing the most (ties: the lowest index). Deterministic; every
    block appears exactly once, and a batch lists its blocks in increasing
    index. The grouping only decides which products are computed together,
    never a value.
    """
    n = len(factors)
    if n <= batch_size:
        return [list(range(n))] if n else []
    users: dict = {}
    for i, block_factors in enumerate(factors):
        for factor in block_factors:
            users.setdefault(factor, []).append(i)
    sizes = np.array([len(f) for f in factors], dtype=np.int64)
    weight = int(sizes.max()) + 1  # new factors first, then shared ones
    remaining = np.ones(n, dtype=bool)
    unused = np.iinfo(np.int64).max
    batches = []
    while remaining.any():
        new = sizes.copy()
        shared = np.zeros(n, dtype=np.int64)
        covered: set = set()
        batch = []
        while len(batch) < batch_size and remaining.any():
            score = np.where(remaining, new * weight - shared, unused)
            j = int(np.argmin(score))
            batch.append(j)
            remaining[j] = False
            for factor in factors[j]:
                if factor not in covered:
                    covered.add(factor)
                    for i in users[factor]:
                        new[i] -= 1
                        shared[i] += 1
        batches.append(sorted(batch))
    return batches


class _BlockPlan:
    """
    How one block of the batched ACC assembly is computed, fixed before any
    kernel is read (:meth:`ACCStrategy._plan_te_block`,
    :meth:`ACCStrategy._plan_wick_block`):

    - ``kind``: ``"te"`` (Eq. 23 per contraction, T/E-only run), ``"wick"``
      (per-Wick-term normalisation, run with a B observable) or ``"zero"``
      (a parity-mixed block the run does not ask for);
    - ``upper``, ``lower``: ``(product key, weight)`` per contraction or
      Wick term, in the order the per-block assembly summed them; the
      weight is the ``norm_Xi`` matrix (``"te"``) or the
      :class:`~cmbcov.bmode_wick.WickTerm` (``"wick"``). ``lower`` is
      ``None`` when the lower triangle is not computed from terms of its
      own (auto blocks; for ``"wick"`` also one-orientation blocks, see
      ``same_band_lower``);
    - ``products``: ``product key -> (spectrum_1, label_1, spectrum_2,
      label_2, kernel pair)``; a product key is ``(identity_1, identity_2,
      kernel pair)`` with the spectra's content identities
      (:func:`_spectrum_identity`), so equal products of different blocks
      share one key;
    - ``pairs``: the kernel pairs the block reads from the cache (for
      ``"te"`` also that of a contraction whose band is a reversed direct
      one, which the per-block assembly asked the loader for too).
    """

    __slots__ = ("cov_key", "kind", "auto", "upper", "lower", "products", "pairs")

    def __init__(
        self,
        cov_key,
        kind,
        auto=False,
        upper=(),
        lower=None,
        products=None,
        pairs=None,
    ):
        self.cov_key = cov_key
        self.kind = kind
        self.auto = auto
        self.upper = list(upper)
        self.lower = None if lower is None else list(lower)
        self.products = products or {}
        self.pairs = set(pairs or ())

    def left_factors(self) -> set:
        """The ``(left spectrum identity, kernel pair)`` of every product."""
        return {(key[0], key[2]) for key in self.products}


#: Version tag of the ACC normalisation used by a run with a B observable,
#: recorded in every raw-block manifest of such a run
#: (:meth:`ACCStrategy.raw_block_inputs`), so a block saved under another
#: rule -- Eq. 23 per Wick contraction, what a T/E-only run uses -- is never
#: reused for it. Change it whenever the per-term rule changes.
ACC_NORMALISATION_RULE = "per-wick-term-v1"

#: Spectra keys that are parity-odd (``C^TB``, ``C^EB`` in either order).
_PARITY_ODD_CL_KEYS = ("TB", "BT", "EB", "BE")


def _contraction_multiset(contractions) -> list:
    """
    The Wick contractions ``((spectrum_1, spectrum_2), kernel_key, norm)`` of
    a T/E block as a sorted list of identities, each taken up to the
    identity ``C1 . Theta^{pq} . C2 = C2 . Theta^{qp} . C1`` (the definition
    gives ``Theta^{qp} = (Theta^{pq})^T``): two blocks with the same list
    have the same ACC value on every element.
    """
    out = []
    for (spec_1, spec_2), kernel_key, norm in contractions:
        forward = (id(spec_1), id(spec_2), tuple(kernel_key), id(norm))
        reverse = (id(spec_2), id(spec_1), tuple(kernel_key[::-1]), id(norm))
        out.append(min(forward, reverse))
    return sorted(out)


def _parity_odd_nonzero(cl: dict[str, dict[str, np.ndarray]]) -> bool:
    """Whether any ``TB``/``BT``/``EB``/``BE`` spectrum in ``cl`` has a
    non-zero entry (at any frequency pair)."""
    for spectra in cl.values():
        for key in _PARITY_ODD_CL_KEYS:
            array = spectra.get(key) if hasattr(spectra, "get") else None
            if array is not None and np.any(np.asarray(array) != 0):
                return True
    return False


def _check_spectrum_length(
    spectrum: np.ndarray, name: str, lmax: int, kernel_size: int, centralell: int
) -> None:
    """Raise if ``spectrum`` is shorter than the ``lmax_int`` the ACC term reads."""
    needed = acc_internal_lmax(lmax, kernel_size, centralell)
    if len(spectrum) < needed:
        raise ValueError(
            f"ACC spectrum {name} has {len(spectrum)} multipoles (ell 0.."
            f"{len(spectrum) - 1}); the ACC term needs {needed} (lmax_int = lmax "
            f"{lmax} + pad {needed - lmax}, the pad being kernel size "
            f"{kernel_size} - 1 - centralell {centralell}) so that no coupling "
            "window is cut at lmax. Supply the spectra to ell = "
            f"{needed - 1}; the covariance is still reported to lmax."
        )


class ACCStrategy(CovarianceStrategy):
    """
    Analytical Covariance Coupling (ACC) strategy.

    This strategy implements the ACC method for computing covariance matrices
    with precomputed coupling kernels. The method accounts for masking effects
    by using coupling kernels computed at a central multipole.

    References
    ----------
    Camphuis et al. (2022), https://arxiv.org/abs/2204.13721
    """

    def compute_covariance_term(
        self, cov_key: CovKey, cl: dict[str, dict[str, np.ndarray]]
    ) -> np.ndarray:
        """
        Compute ACC covariance term using precomputed coupling kernels.

        Parameters
        ----------
        cov_key : CovKey
            Covariance key specifying which cross-spectra to compute
        cl : Dict[str, Dict[str, np.ndarray]]
            Power spectra organized as cl[freq_combination][stokes_combination]

        Returns
        -------
        np.ndarray
            Full covariance matrix for this key, ``(lmax, lmax)``.

        Notes
        -----
        The term is evaluated with the spectra read to ``lmax_int = lmax +
        acc_window_pad(S, centralell)`` (:func:`acc_window_pad`, ``S`` the
        kernel size), so every reported element ``ell1, ell2 < lmax`` sees
        its whole translated kernel. It equals the ACC term computed on
        ``lmax_int`` and truncated to ``lmax``: an element depends on the
        spectra, the kernel and ``norm_Xi[ell1, ell2]`` only, and
        ``norm_Xi`` is read only for ``ell1, ell2 < lmax``. Padding the
        spectra to ``lmax_int`` avoids dropping kernel weight for the last
        elements, which would otherwise bias the entries within
        ``S - 1 - l*`` of ``lmax`` low.

        In a run with a B observable (set by :meth:`configure_run`,
        or, for a direct call, a block with a B letter) every block, T/E
        blocks included, is instead the sum over its expanded Wick terms,
        each translated as above and normalised by its own ``N_t``
        (:meth:`_compute_covariance_term_wick`,
        docs/theory/bmode_kernels.md), with the same orientation rule as
        below when the cache holds the kernel pairs of both orientations
        (:meth:`_wick_plan`). A T/E-only run (level 1) takes the path below.

        Each element is computed in the orientation its kernel is exact in:
        the upper triangle (``ell1 <= ell2``) of a block ``Cov(a, b)`` from
        the Wick contractions of ``cov_key``, the lower triangle from those
        of ``cov_key.transpose()``, so the lower triangle is the transposed
        upper triangle of ``Cov(b, a)`` and the result does not depend on
        the order in which the run lists its spectra
        (docs/theory/acc.md, Sect. 3). A block whose transposed key has the
        same contractions (an auto block, or one between two TT or two EE
        spectra) is symmetric and computed once, bit-identical to the
        single-orientation assembly of cmbcov 0.2.0.

        The block is computed by the batched assembly of
        :meth:`compute_covariance_terms` as a batch of one; a run's
        :meth:`~cmbcov.covariance.Cov.compute_covariance_matrix` computes
        its blocks together there, with the same result bit for bit.

        Raises
        ------
        ValueError
            If required ACC parameters are not properly configured, or a
            spectrum this key reads holds fewer than ``lmax_int`` multipoles.
            In a B run also if a Wick term's true spectrum is missing from
            ``cl``, ``centralell < 2``, or the kernels are not GL-grid ones.
        """
        self.validate_config()
        for _, block in self._assemble([cov_key], cl):
            return block

    def compute_covariance_terms(
        self, cov_keys: Iterable[CovKey], cl: dict[str, dict[str, np.ndarray]]
    ) -> Iterator[tuple[CovKey, np.ndarray]]:
        """
        ``(cov_key, compute_covariance_term(cov_key, cl))`` for every key of
        ``cov_keys``, bit for bit, computed together: the order in which the
        pairs are yielded is not that of ``cov_keys``.

        Many blocks of a run evaluate the same kernel product
        ``C1 . Theta . C2`` on a diagonal, and many more share its left
        factor, the window matrix of ``C1`` times ``Theta`` (nearly all of the
        cost). A block alone computes its own; here the blocks are taken in
        batches of blocks sharing left factors (:func:`_group_blocks`),
        bounded by :data:`_ASSEMBLY_BATCH_BYTES`, and each batch runs
        diagonal by diagonal: every distinct product of the batch once, and
        every left factor once (:func:`_acc_bands`). The value of every
        element is unchanged, since each product is evaluated by the same
        operations on the same operands and each block sums its terms in the
        same order. Nothing is kept between calls.

        Used by :meth:`~cmbcov.covariance.Cov.compute_covariance_matrix`;
        the blocks of one batch are held (flattened) until the batch is done,
        and each is expanded to ``(lmax, lmax)`` only when it is yielded.
        """
        self.validate_config()
        yield from self._assemble(list(cov_keys), cl)

    def _assemble(
        self, cov_keys: list[CovKey], cl: dict[str, dict[str, np.ndarray]]
    ) -> Iterator[tuple[CovKey, np.ndarray]]:
        """Plan every block of ``cov_keys``, group them in batches and yield
        each block when its batch is done (see
        :meth:`compute_covariance_terms`)."""
        identity = _identity_memo()
        plans = []
        for cov_key in cov_keys:
            if self._wick_mode(cov_key):
                plans.append(self._plan_wick_block(cov_key, cl, identity))
            else:
                plans.append(self._plan_te_block(cov_key, cl, identity))

        dmax = self.cov.config.dmax
        lmax = self.cov.lmax
        # T/E and B blocks are never batched together (they only meet in
        # direct calls without run context): one normalises the kernels,
        # the other does not.
        for kind in ("te", "wick"):
            members = [i for i, plan in enumerate(plans) if plan.kind == kind]
            if not members:
                continue
            per_block = max(
                (2 * dmax - 1 + len(plans[i].products)) * lmax * 8 for i in members
            )
            batch_size = max(1, _ASSEMBLY_BATCH_BYTES // max(1, per_block))
            groups = _group_blocks(
                [plans[i].left_factors() for i in members], batch_size
            )
            for group in groups:
                batch = [plans[members[g]] for g in group]
                flats = self._assemble_batch(batch, kind)
                for index, plan in enumerate(batch):
                    # Release each flattened block as it is expanded, and
                    # the whole batch before the next one is computed, so
                    # that at most one batch is held.
                    block = self._unflatten_cov(flats[index])
                    flats[index] = None
                    yield plan.cov_key, block
                    del block
                del flats
        for plan in plans:
            if plan.kind == "zero":
                yield plan.cov_key, self._unflatten_cov(self._empty_flatten_cov(dmax))

    def _plan_te_block(self, cov_key: CovKey, cl, identity) -> _BlockPlan:
        """
        The contractions of a block of a T/E-only run and the kernel
        product each one reads, chosen exactly as the per-block assembly
        of cmbcov 0.3.0 chose them (so the result is bit-identical to it).

        Orientation. The exact covariance is not symmetric within a block
        between two different spectra, Cov(a_l, b_l') != Cov(a_l', b_l),
        and ACC is exact at l* only in the orientation its kernel was
        computed in: spectrum a on the l* leg, b on the l* + Delta leg.
        Every element is therefore computed in that orientation: the upper
        triangle (ell1 <= ell2) from the Wick contractions of cov_key, the
        lower one from those of the transposed key, since
        Cov(a_{l+D}, b_l) = Cov(b_l, a_{l+D}). An element then depends on
        its two spectra and on which of its multipoles is the smaller, never
        on the order in which the run lists its spectra. Using the upper
        value for both triangles, as before, made Cov(TE_l, EE_l') and
        Cov(EE_l, TE_l') of different frequency pairs disagree on which
        element is exact, so the CMB no longer cancelled in the frequency
        differences and a multi-frequency T/E matrix was not positive
        definite (docs/theory/acc.md, Sect. 3). An auto block is
        symmetric and has one orientation.

        Kernel lookup keys (covariance_coupling is keyed by
        COUPLING_CHANNELS, via SpecKey.kernel_stokekey -- e.g. "DT" != "TD"
        off the diagonal) keep the true Wick order of
        CovKey.key_to_cross_kernel; the cl lookup uses key_to_cross, whose
        sorted_copy() order is right for spectra ("TE" and "ET" are the
        same spectrum). For each Wick contraction w:
          spectra  cl[combination_w[k].freqkey()][combination_w[k].stokekey()]
          kernel   coupling_kernels[kernel_w[0].kernel_stokekey(), kernel_w[1].kernel_stokekey()]
          prefactor norm_Xi[combination_w[0].stokekey(), combination_w[1].stokekey()]
        read at the element's own (row, column) multipoles.
        compute_acc_term depends on (ell1, ell2) only through
        min(ell1, ell2): both kernel indices are shifted by the same amount
        and the section is square. So on diagonal Delta one band per
        contraction gives the whole upper (or lower) diagonal.

        Band reuse within the block (what fixes the operands of each band):
        a contraction whose (spectrum_1, spectrum_2, kernel pair) arrays
        already have a band reuses it; a lower-triangle contraction that is
        a direct one read backwards, C1 . Theta^{pq} . C2 = C2 . Theta^{qp} . C1
        (the definition gives Theta^{qp} = (Theta^{pq})^T, and the cache
        stores it so: s2xs1 is written as the transpose of s1xs2, a p x p
        kernel is a symmetric Gram matrix), reuses the direct band.
        """
        names: dict[int, str] = {}
        auto = cov_key.auto()
        upper_contractions = self._te_contractions(cov_key, cl, names)
        lower_contractions = (
            upper_contractions
            if auto
            else self._te_contractions(cov_key.transpose(), cl, names)
        )
        # A block whose transposed key has the same contractions, up to the
        # identity C1 . Theta^{pq} . C2 = C2 . Theta^{qp} . C1 (every block
        # between two TT, or two EE, spectra; so every block of a T-only or
        # E-only run), has one orientation: its lower triangle is its upper
        # band at norm_Xi[ell2, ell1], bit-identical to before.
        one_orientation = auto or _contraction_multiset(
            upper_contractions
        ) == _contraction_multiset(lower_contractions)
        if one_orientation:
            lower_contractions = upper_contractions

        products: dict[tuple, tuple] = {}
        chosen: dict[tuple, tuple] = {}

        def product(spec_1, spec_2, kernel_key) -> tuple:
            key = (identity(spec_1), identity(spec_2), kernel_key)
            if key not in products:
                products[key] = (
                    spec_1,
                    names[id(spec_1)],
                    spec_2,
                    names[id(spec_2)],
                    kernel_key,
                )
            return key

        upper = []
        for (spec_1, spec_2), kernel_key, norm in upper_contractions:
            memo = (id(spec_1), id(spec_2), kernel_key)
            if memo not in chosen:
                chosen[memo] = product(spec_1, spec_2, kernel_key)
            upper.append((chosen[memo], norm))
        lower = None
        if not auto:
            lower = []
            for (spec_1, spec_2), kernel_key, norm in lower_contractions:
                memo = (id(spec_1), id(spec_2), kernel_key)
                reverse = (id(spec_2), id(spec_1), kernel_key[::-1])
                if memo in chosen:
                    key = chosen[memo]
                elif reverse in chosen:
                    key = chosen[reverse]
                else:
                    key = chosen[memo] = product(spec_1, spec_2, kernel_key)
                lower.append((key, norm))
        # The kernels read are those of every contraction, including one
        # whose band is a reversed direct one (as before: the cache must
        # hold them).
        pairs = {
            kernel_key for _, kernel_key, _ in upper_contractions + lower_contractions
        }
        return _BlockPlan(cov_key, "te", auto, upper, lower, products, pairs)

    def _plan_wick_block(self, cov_key: CovKey, cl, identity) -> _BlockPlan:
        """
        The Wick terms of a block of a run with a B observable
        (:meth:`_wick_plan`) and the kernel product each one reads; a
        parity-mixed block the run does not ask for is ``"zero"``. See
        :meth:`_compute_covariance_term_wick`.
        """
        from ..keys import _is_parity_odd_spec

        flags = self._run_flags(cov_key, cl)
        if (
            _is_parity_odd_spec(cov_key.left) != _is_parity_odd_spec(cov_key.right)
            and not flags["parity_mixed_blocks"]
        ):
            return _BlockPlan(cov_key, "zero")

        upper_terms, lower_terms, _ = self._wick_plan(cov_key, cl)
        spectra = {}
        for term in upper_terms + (lower_terms or []):
            for side in (term.left, term.right):
                if side not in spectra:
                    spectra[side] = self._wick_spectrum(cl, side)
        products: dict[tuple, tuple] = {}

        def product(term) -> tuple:
            pair = (term.channel_1, term.channel_2)
            (name_1, spec_1), (name_2, spec_2) = spectra[term.left], spectra[term.right]
            key = (identity(spec_1), identity(spec_2), pair)
            if key not in products:
                products[key] = (spec_1, f"cl[{name_1}]", spec_2, f"cl[{name_2}]", pair)
            return key

        upper = [(product(term), term) for term in upper_terms]
        lower = (
            None
            if lower_terms is None
            else [(product(term), term) for term in lower_terms]
        )
        pairs = {entry[4] for entry in products.values()}
        return _BlockPlan(
            cov_key, "wick", cov_key.auto(), upper, lower, products, pairs
        )

    def _assemble_batch(self, batch: list[_BlockPlan], kind: str) -> list[np.ndarray]:
        """
        The flattened blocks of one batch of plans of one ``kind``,
        diagonal by diagonal: the kernels once per diagonal, every distinct
        product once (:func:`_acc_bands`), then each block's upper and lower
        bands summed exactly as the per-block assembly summed them.
        """
        config = self.cov.config
        dmax = config.dmax
        centralell = config.centralell
        lmax = self.cov.lmax
        flats = [self._empty_flatten_cov(dmax) for _ in batch]
        products: dict[tuple, tuple] = {}
        for plan in batch:
            products.update(plan.products)
        needed_pairs = set().union(*(plan.pairs for plan in batch))
        used_pairs = {entry[4] for entry in products.values()}
        padded: dict[tuple, np.ndarray] = {}

        def padded_spectrum(spectrum, identity, label, size):
            if (identity, size) not in padded:
                _check_spectrum_length(spectrum, label, lmax, size, centralell)
                n_read = acc_internal_lmax(lmax, size, centralell)
                padded[identity, size] = _padded_spectrum(
                    spectrum, centralell, n_read, size
                )
            return padded[identity, size]

        for diagonal_offset in range(dmax):
            n_ell = lmax - diagonal_offset
            ell_prime = centralell + diagonal_offset
            if kind == "te":
                # (the T/E path reads the kernels before the n_ell check)
                coupling_kernels = self.get_covariance_coupling(
                    centralell, ell_prime, pairs=needed_pairs
                )
                if n_ell <= 0:
                    continue
                # Normalise each distinct kernel once per diagonal.
                kernels = {
                    pair: _unit_sum_kernel(coupling_kernels[pair])
                    for pair in used_pairs
                }
            else:
                if n_ell <= 0:
                    continue
                coupling_kernels = self.get_covariance_coupling(
                    centralell, ell_prime, pairs=sorted(needed_pairs)
                )
                self._require_gl_kernels(centralell, ell_prime)
                kernels = {
                    pair: np.asarray(coupling_kernels[pair], dtype=float)
                    for pair in used_pairs
                }
                kernel_sums = {
                    pair: float(np.sum(coupling_kernels[pair])) for pair in used_pairs
                }
            ell1 = np.arange(n_ell)
            ell2 = ell1 + diagonal_offset

            work = {}
            for key, (spec_1, label_1, spec_2, label_2, pair) in products.items():
                kernel = kernels[pair]
                size = kernel.shape[0]
                work[key] = (
                    kernel,
                    padded_spectrum(spec_1, key[0], label_1, size),
                    padded_spectrum(spec_2, key[1], label_2, size),
                )
            bands = _acc_bands(work, n_ell)
            del work

            upper_row = diagonal_offset + dmax - 1
            lower_row = -diagonal_offset + dmax - 1
            if kind == "te":
                self._sum_te_bands(
                    batch,
                    flats,
                    bands,
                    diagonal_offset,
                    ell1,
                    ell2,
                    upper_row,
                    lower_row,
                )
            else:
                self._sum_wick_bands(
                    batch,
                    flats,
                    bands,
                    kernel_sums,
                    diagonal_offset,
                    ell1,
                    ell2,
                    upper_row,
                    lower_row,
                )
        return flats

    @staticmethod
    def _sum_te_bands(
        batch, flats, bands, diagonal_offset, ell1, ell2, upper_row, lower_row
    ):
        """
        One diagonal of every T/E block of a batch from the bands of its
        products: each contraction's band times ``norm_Xi`` read at the
        element's own (row, column), summed in the block's contraction order
        (upper triangle at ``norm[ell1, ell2]``, lower at
        ``norm[ell2, ell1]``); an auto block's lower triangle is its upper
        one. ``norm[...]`` is gathered once per matrix and triangle per
        diagonal (the same values each block read before).
        """
        n_ell = ell1.size
        norms: dict[tuple, np.ndarray] = {}
        for plan, flat in zip(batch, flats):
            upper = np.zeros(n_ell)
            for key, norm in plan.upper:
                memo = (id(norm), False)
                if memo not in norms:
                    norms[memo] = norm[ell1, ell2]
                upper += bands[key] * norms[memo]
            flat[upper_row, :n_ell] = upper
            if diagonal_offset != 0:
                if plan.auto:
                    lower = upper
                else:
                    lower = np.zeros(n_ell)
                    for key, norm in plan.lower:
                        memo = (id(norm), True)
                        if memo not in norms:
                            norms[memo] = norm[ell2, ell1]
                        lower += bands[key] * norms[memo]
                flat[lower_row, :n_ell] = lower

    def _sum_wick_bands(
        self,
        batch,
        flats,
        bands,
        kernel_sums,
        diagonal_offset,
        ell1,
        ell2,
        upper_row,
        lower_row,
    ):
        """
        One diagonal of every block of a batch of a run with a B observable
        from the bands of its products: each Wick term's
        ``coefficient * band * F_t`` (:meth:`~cmbcov.covariance.Cov.acc_term_scale`),
        in the block's term order. ``F_t`` depends on the term's kernel
        pair, the diagonal and the triangle only, so it is computed once per
        pair and triangle per diagonal (the same values each block computed
        before).
        """
        n_ell = ell1.size
        scales: dict[tuple, np.ndarray] = {}

        def scale(term, lower_triangle):
            pair = (term.channel_1, term.channel_2)
            memo = (pair, lower_triangle)
            if memo not in scales:
                scales[memo] = self.cov.acc_term_scale(
                    term.channel_1,
                    term.channel_2,
                    diagonal_offset,
                    kernel_sums[pair],
                    ell2 if lower_triangle else ell1,
                    ell1 if lower_triangle else ell2,
                )
            return scales[memo]

        for plan, flat in zip(batch, flats):
            upper = np.zeros(n_ell)
            lower = np.zeros(n_ell)
            # One orientation (plan.lower is None): the lower triangle is the
            # upper band with N_t read at (l + Delta, l), unchanged.
            same_band_lower = (
                diagonal_offset != 0 and not plan.auto and plan.lower is None
            )
            for key, term in plan.upper:
                band = bands[key]
                upper += term.coefficient * band * scale(term, False)
                if same_band_lower:
                    lower += term.coefficient * band * scale(term, True)
            # Both orientations: the element (a at l + Delta, b at l) is
            # Cov(b_l, a_{l+Delta}), i.e. the terms of the transposed key with
            # b on the l* leg of their kernels, read at the element's own
            # (row, column) multipoles like the level-1 path.
            if diagonal_offset != 0 and plan.lower is not None:
                for key, term in plan.lower:
                    lower += term.coefficient * bands[key] * scale(term, True)
            flat[upper_row, :n_ell] = upper
            if diagonal_offset != 0:
                flat[lower_row, :n_ell] = upper if plan.auto else lower

    def _te_contractions(
        self,
        cov_key: CovKey,
        cl: dict[str, dict[str, np.ndarray]],
        names: dict[int, str],
    ) -> list[tuple[tuple[np.ndarray, np.ndarray], tuple[str, str], np.ndarray]]:
        """
        The two Wick contractions of a T/E ``cov_key``, in its own
        orientation: ``((spectrum_1, spectrum_2), kernel_key, norm_Xi)``
        each, with the spectra looked up through
        :meth:`~cmbcov.keys.CovKey.key_to_cross` and the kernel pair through
        :meth:`~cmbcov.keys.CovKey.key_to_cross_kernel`. ``names`` collects a
        label per spectrum array for the length check's message.
        """
        combination_1, combination_2 = cov_key.key_to_cross()
        kernel_1, kernel_2 = cov_key.key_to_cross_kernel()
        contractions = []
        for combination, kernel in (
            (combination_1, kernel_1),
            (combination_2, kernel_2),
        ):
            spectra_key = (combination[0].stokekey(), combination[1].stokekey())
            for spec in combination:
                spectrum = cl[spec.freqkey()][spec.stokekey()]
                names.setdefault(
                    id(spectrum), f"cl[{spec.freqkey()!r}][{spec.stokekey()!r}]"
                )
            contractions.append(
                (
                    (
                        cl[combination[0].freqkey()][spectra_key[0]],
                        cl[combination[1].freqkey()][spectra_key[1]],
                    ),
                    (kernel[0].kernel_stokekey(), kernel[1].kernel_stokekey()),
                    self.cov.norm_Xi[spectra_key],
                )
            )
        return contractions

    def raw_block_inputs(
        self, cov_key: CovKey, cl: dict[str, dict[str, np.ndarray]]
    ) -> tuple[dict[str, np.ndarray], dict]:
        """
        The ``save_raw_blocks`` manifest inputs of an ACC block: the four
        Wick-contraction spectra as :meth:`compute_covariance_term` reads
        them (``[:lmax_int]``), plus ``lmax_int``, ``dmax``, ``centralell`` and the identity
        of the kernel cache -- the resolved ``acc_kernel_dir`` and a digest
        of, for every diagonal, the pair's manifest and the size and
        modification time of the kernel files this key loads (those of both
        orientations for a non-auto block). A recomputed or replaced kernel
        set therefore invalidates the block, including a legacy cache
        without manifests.

        In a run with a B observable (:meth:`_wick_mode`) the spectra are
        every true spectrum the block's expanded Wick terms read (those of
        both orientations for a block computed in both), the pairs are
        those terms' kernel pairs, and the identity also records the
        normalisation rule (:data:`ACC_NORMALISATION_RULE`), the pair list
        and the orientation the block was computed in (``"natural"`` or
        ``"both"``, :meth:`_wick_plan`). A block saved by a T/E-only run
        (Eq. 23 per contraction) therefore never matches a B-run manifest, a
        T/E-only run's manifest is unchanged, and a block saved before the
        lower triangle of B runs was computed in its own orientation (up to
        cmbcov 0.3.0, or from a one-orientation cache: ``"natural"`` or
        ``"transposed"`` for a block that now has ``"both"``) is recomputed.
        """
        config = self.cov.config
        kernel_dir = os.path.abspath(self.cov.acc_kernel_dir)
        lmax = self.cov.lmax
        if self._wick_mode(cov_key):
            upper_terms, lower_terms, orientation = self._wick_plan(cov_key, cl)
            terms = upper_terms + (lower_terms or [])
            pairs = sorted({(t.channel_1, t.channel_2) for t in terms})
            size = acc_cached_kernel_size(
                kernel_dir, config.centralell, config.dmax, pairs=pairs
            )
            lmax_int = (
                lmax
                if size is None
                else acc_internal_lmax(lmax, size, config.centralell)
            )
            spectra = {}
            for term in terms:
                for side in (term.left, term.right):
                    label, array = self._wick_spectrum(cl, side)
                    spectra[label] = array[:lmax_int]
            identity = self._kernel_cache_identity(kernel_dir, pairs, lmax_int)
            identity.update(
                {
                    "acc_normalisation": ACC_NORMALISATION_RULE,
                    "acc_kernel_pairs": ["x".join(pair) for pair in pairs],
                    "acc_block_orientation": orientation,
                }
            )
            return spectra, identity

        # The kernel pairs of both orientations: the lower triangle of a
        # non-auto block is computed from the transposed key's contractions
        # (compute_covariance_term). Its spectra are the same four.
        pairs = set()
        for key in (cov_key,) if cov_key.auto() else (cov_key, cov_key.transpose()):
            for kernel in key.key_to_cross_kernel():
                pairs.add((kernel[0].kernel_stokekey(), kernel[1].kernel_stokekey()))
        pairs = sorted(pairs)
        size = acc_cached_kernel_size(
            kernel_dir, config.centralell, config.dmax, pairs=pairs
        )
        lmax_int = (
            lmax if size is None else acc_internal_lmax(lmax, size, config.centralell)
        )
        spectra, _ = super().raw_block_inputs(cov_key, cl)
        spectra = {label: array[:lmax_int] for label, array in spectra.items()}
        return spectra, self._kernel_cache_identity(kernel_dir, pairs, lmax_int)

    def _kernel_cache_identity(
        self, kernel_dir: str, pairs: Sequence[tuple[str, str]], lmax_int: int
    ) -> dict:
        """
        The kernel-cache part of :meth:`raw_block_inputs`' identity:
        ``lmax_int``, ``dmax``, ``centralell``, the kernel directory and a
        digest of, per diagonal, the manifest and the size and modification
        time of every kernel file of ``pairs``.
        """
        config = self.cov.config
        per_diagonal = []
        for diagonal_offset in range(config.dmax):
            ell = config.centralell
            ell_prime = ell + diagonal_offset
            files = {}
            for pair in pairs:
                # Own name, legacy name, transposed own name, transposed
                # legacy name (docs/theory/bmode_kernels.md): a
                # legacy-only or transpose-only cache still invalidates the
                # block exactly like its new-named/natural-orientation
                # equivalent would, by stating the file it actually loads
                # through.
                stat = None
                for candidate_path, _transposed in acc_cache.coupling_kernel_candidates(
                    kernel_dir,
                    pair,
                    ell,
                    ell_prime,
                    COUPLING_CHANNELS,
                    LEGACY_CHANNEL_NAMES,
                ):
                    try:
                        stat = os.stat(candidate_path)
                        break
                    except OSError:
                        continue
                files["x".join(pair)] = (
                    None if stat is None else [stat.st_size, stat.st_mtime_ns]
                )
            per_diagonal.append(
                {
                    "manifest": acc_cache.read_coupling_manifest(
                        kernel_dir, ell, ell_prime
                    ),
                    "files": files,
                }
            )
        payload = json.dumps(per_diagonal, sort_keys=True, default=str)
        identity = {
            "lmax_int": int(lmax_int),
            "dmax": int(config.dmax),
            "centralell": int(config.centralell),
            "acc_kernel_dir": kernel_dir,
            "acc_kernels_digest": hashlib.blake2b(
                payload.encode(), digest_size=16
            ).hexdigest(),
        }
        return identity

    # ------------------------------------------------------------------
    # Runs with a B observable: per-Wick-term assembly
    # ------------------------------------------------------------------

    #: Run context set by :meth:`configure_run`; ``None`` until then (a
    #: direct :meth:`compute_covariance_term` call), in which case it is
    #: derived from the block and the spectra alone (:meth:`_run_flags`).
    _run_context: dict | None = None

    def configure_run(self, covariance_keys, cl) -> None:
        """
        Record what the run as a whole decides for every block:

        - ``wick``: any observable has a B letter, so every block (T/E ones
          included) is assembled per Wick term with the per-term
          normalisation of docs/theory/bmode_kernels.md, Sect. 5;
        - ``parity_odd``: the ``C^TB``/``C^EB`` terms are kept -- only when a
          parity-odd observable (``TB``, ``EB``) is requested AND a
          ``TB``/``BT``/``EB``/``BE`` spectrum in ``cl`` is non-zero
          (C^TB and C^EB are read only at level 3 and up);
        - ``parity_mixed_blocks``: from ``covariance_keys``.
        """
        from ..covariance import has_b_observable
        from ..keys import _is_parity_odd_spec

        wick = has_b_observable(covariance_keys)
        parity_odd_requested = any(
            _is_parity_odd_spec(spec) for spec in covariance_keys.spec_keys
        )
        self._run_context = {
            "wick": wick,
            "parity_odd": bool(
                wick and parity_odd_requested and _parity_odd_nonzero(cl)
            ),
            "parity_mixed_blocks": bool(
                getattr(covariance_keys, "parity_mixed_blocks", False)
            ),
        }

    def _run_flags(self, cov_key: CovKey, cl) -> dict:
        """The run context, or, without :meth:`configure_run`, the flags a
        run made of this block alone would have."""
        if self._run_context is not None:
            return self._run_context
        from ..keys import _is_parity_odd_spec

        wick = "B" in cov_key.stoke
        parity_odd_requested = _is_parity_odd_spec(cov_key.left) or _is_parity_odd_spec(
            cov_key.right
        )
        return {
            "wick": wick,
            "parity_odd": bool(
                wick and parity_odd_requested and _parity_odd_nonzero(cl)
            ),
            "parity_mixed_blocks": False,
        }

    def _wick_mode(self, cov_key: CovKey) -> bool:
        """Whether ``cov_key`` is assembled per Wick term: the run has a B
        observable (or, without run context, the block itself has one)."""
        if self._run_context is not None:
            return bool(self._run_context["wick"])
        return "B" in cov_key.stoke

    @staticmethod
    def _wick_spectrum(cl, side) -> tuple[str, np.ndarray]:
        """``(label, array)`` of a Wick term's true spectrum ``(letters,
        (freq_1, freq_2))``, looked up like the level-1 path does
        (:meth:`~cmbcov.keys.SpecKey.sorted_copy`, so
        ``ET`` at one frequency reads ``TE``)."""
        letters, freqs = side
        spec = SpecKey(tuple(letters), tuple(freqs)).sorted_copy()
        freq_key, stokes_key = spec.freqkey(), spec.stokekey()
        try:
            array = cl[freq_key][stokes_key]
        except KeyError:
            raise ValueError(
                f"the ACC B-mode assembly needs the true spectrum "
                f"cl[{freq_key!r}][{stokes_key!r}] (a Wick term of this block "
                "reads it even if it is not an observable); supply it"
            ) from None
        return f"{freq_key}/{stokes_key}", array

    def _pairs_on_disk(self, pairs: Iterable[tuple[str, str]]) -> bool:
        """Whether the kernel loader can serve every pair on the first
        diagonal ``(centralell, centralell)``: under its own or legacy name,
        or stored only as its transpose, and listed by the manifest's pair
        record if it has one (:func:`_missing_kernel_pairs`, the loader's own
        rule, :func:`~cmbcov.approximations.acc_cache.locate_coupling_kernel`)."""
        ell = self.cov.config.centralell
        return not _missing_kernel_pairs(self.cov.acc_kernel_dir, ell, ell, pairs)

    def _wick_term_multiset(self, terms, cl) -> list:
        """
        The Wick terms of a block as a sorted list of ``(identity,
        coefficient)``, each term taken up to the identity
        ``C1 . Theta^{pq} . C2 = C2 . Theta^{qp} . C1`` and with its spectra
        identified by the arrays ``cl`` holds for them: two blocks with the
        same list have the same ACC value on every element (the Wick
        counterpart of :func:`_contraction_multiset`).
        """
        merged: dict[tuple, int] = {}
        for term in terms:
            left = id(self._wick_spectrum(cl, term.left)[1])
            right = id(self._wick_spectrum(cl, term.right)[1])
            forward = (left, term.channel_1, term.channel_2, right)
            reverse = (right, term.channel_2, term.channel_1, left)
            key = min(forward, reverse)
            merged[key] = merged.get(key, 0) + term.coefficient
        return sorted((k, v) for k, v in merged.items() if v)

    def _wick_plan(self, cov_key: CovKey, cl) -> tuple[list, list | None, str]:
        """
        How a block of a run with a B observable is assembled:
        ``(upper_terms, lower_terms, orientation)``.

        The exact covariance of two different spectra is not symmetric
        within its block, and the kernel of ``Cov(a, b)`` at
        ``(l*, l* + Delta)`` is exact only with ``a`` on the lower multipole
        (docs/theory/acc.md, Sect. 3; docs/theory/bmode_kernels.md,
        Sect. 6). So, as in the T/E-only path, every element is computed in
        its own orientation:

        - an auto block, or a block whose transposed key has the same Wick
          terms (:meth:`_wick_term_multiset`; e.g. two TT, EE or BB spectra
          at different frequencies), has one orientation: ``lower_terms`` is
          ``None``, its lower triangle is its upper band, and
          ``orientation`` is ``"natural"``;
        - every other block takes its upper triangle from the terms of
          ``Cov(a, b)`` and its lower one from those of ``Cov(b, a)``
          (``orientation = "both"``), so the lower triangle is the
          transposed upper triangle of ``Cov(b, a)`` and the block does not
          depend on the order in which the run lists its spectra.

        The kernel pairs of both orientations must be on disk
        (:func:`~cmbcov.bmode_wick.required_kernel_pairs` includes them).
        There is no single-orientation fallback: a cache that lacks some
        (e.g. a B-mode cache precomputed by cmbcov 0.3.0 or earlier, which
        stored one orientation per block) raises ``OSError`` naming the
        block and its missing pairs, rather than computing one triangle
        from the other orientation's kernel.
        """
        from ..bmode_wick import covkey_wick_terms

        parity_odd = self._run_flags(cov_key, cl)["parity_odd"]
        natural = covkey_wick_terms(cov_key.stoke, cov_key.freq, parity_odd)
        lower = None
        if cov_key.left != cov_key.right:
            swapped = cov_key.transpose()
            transposed = covkey_wick_terms(swapped.stoke, swapped.freq, parity_odd)
            if self._wick_term_multiset(natural, cl) != self._wick_term_multiset(
                transposed, cl
            ):
                lower = transposed
        needed = sorted({(t.channel_1, t.channel_2) for t in natural + (lower or [])})
        ell = self.cov.config.centralell
        missing = _missing_kernel_pairs(self.cov.acc_kernel_dir, ell, ell, needed)
        if missing:
            raise OSError(
                f"ACC coupling kernels not found under "
                f"{self.cov.acc_kernel_dir!r} for the block "
                f"{cov_key.stokekey()} ({cov_key.freqkey()}): it needs "
                f"{len(needed)} kernel pairs"
                + (" (both orientations)" if lower is not None else "")
                + f", and {', '.join('x'.join(p) for p in missing)} are "
                f"missing at ({ell}, {ell}). {_BMODE_PAIRS_HELP}"
            )
        return natural, lower, "natural" if lower is None else "both"

    def _require_gl_kernels(self, ell: int, ell_prime: int) -> None:
        """
        The per-term normalisation reads the raw kernel sums (``c_t*``), so
        the kernels must be in the GL convention. HEALPix-grid kernels carry
        a ``sqrt(2)`` per spin-2 integral (``precompute_acc_kernels``,
        ``grid``) that Eq. 23 cancels but this rule would not; a cache with
        no manifest cannot say which it is. Both are refused.
        """
        manifest = acc_cache.read_coupling_manifest(
            self.cov.acc_kernel_dir, ell, ell_prime
        )
        grid = None if manifest is None else manifest.get("grid")
        if grid != "gl":
            raise ValueError(
                "the per-Wick-term ACC normalisation of a run with a B "
                "observable needs coupling kernels precomputed on the GL grid "
                f"(grid='gl'); the kernels of ({ell}, {ell_prime}) under "
                f"{self.cov.acc_kernel_dir!r} are "
                + (
                    "from a cache with no manifest"
                    if manifest is None
                    else f"grid={grid!r}"
                )
                + ". Recompute them with precompute_acc_kernels(grid='gl')."
            )

    def _compute_covariance_term_wick(
        self, cov_key: CovKey, cl: dict[str, dict[str, np.ndarray]]
    ) -> np.ndarray:
        r"""
        One block of a run with a B observable, as a sum over its expanded
        Wick terms (docs/theory/bmode_kernels.md):

        .. math::

            \tilde\Sigma(\ell, \ell+\Delta) = \sum_t \sigma_t\,
            C^{a_t}_{+s}\cdot\Theta^{t}_*\cdot C^{b_t}_{+s}\;
            N_t(\ell, \ell+\Delta) / S^t_*,

        with :math:`\Theta^t_*` the raw kernel of the term's channel pair at
        :math:`(\ell_*, \ell_*+\Delta)`, the Eq. 33 translation of today
        (:func:`_acc_band`), and :math:`N_t` the per-term rule
        (:meth:`~cmbcov.covariance.Cov.acc_term_scale`).
        Each element is computed in its own orientation
        (:meth:`_wick_plan`): the upper triangle from the terms of
        ``cov_key``, the lower one from those of ``cov_key.transpose()``,
        both with :math:`N_t` read at the element's own (row, column)
        multipoles, so every element is exact at :math:`\ell_*` and the
        result does not depend on the order in which the run lists its
        spectra (docs/theory/bmode_kernels.md, Sect. 6). An auto block and a
        block whose transposed key has the same terms take one band for both
        triangles, :math:`N_t` read at ``(l + Delta, l)`` for the lower one.
        A cache without the kernel pairs of both orientations is refused
        (``OSError``). A parity-mixed block
        (one parity-odd observable) is zero unless the run asks for
        ``parity_mixed_blocks``.

        :meth:`compute_covariance_term` of such a run computes the block
        through the batched assembly (:meth:`compute_covariance_terms`), with
        the same result; this method forces the per-Wick-term path for one
        block whatever the run context.
        """
        plan = self._plan_wick_block(cov_key, cl, _identity_memo())
        if plan.kind == "zero":
            return self._unflatten_cov(self._empty_flatten_cov(self.cov.config.dmax))
        return self._unflatten_cov(self._assemble_batch([plan], "wick")[0])

    def validate_config(self) -> None:
        """Validate configuration for ACC strategy."""
        super().validate_config()

        config = self.cov.config
        if config.method != CovarianceMethod.ACC:
            raise ValueError(f"ACC method expected but got {config.method}")

        if config.dmax is None or config.dmax < 1:
            raise ValueError("ACC method requires dmax to be set and at least 1")

        if config.centralell is None or config.centralell < 0:
            raise ValueError("centralell must be set and non-negative for ACC method")

        if config.centralell <= 150:
            warn_low_centralell_once(config.centralell)

    def error_budget(self, covariance_keys, band_edges=None) -> dict:
        """
        The error budget of this run, as a dict; every ACC run has one.

        Three parts, from :mod:`~cmbcov.approximations.acc_budget`. For a
        T/E run with a polarised leg: the E->B leakage of the kernel set in
        use, ``lambda = sum Theta^{TT x BB} / sum Theta^{TT x EE}``, giving
        ``(n_E / 2) lambda`` per block (the ``BB`` kernels are already in the
        cache and read by nothing else, so this computes nothing new); and
        the Eq. 33 translation error, which is *not* bounded -- the budget
        reports ``|l - l*|`` at the band edges and says so. For every run:
        the normalisation check of the kernels against ``Cov.Xi`` (Eq. 22).
        See that module for what was tried and measured.

        Parameters
        ----------
        covariance_keys : CovKeys
            The blocks the run computes.
        band_edges : sequence of int, optional
            Bandpower edges, for the ``|l - l*|`` quoted; defaults to
            ``[lmin, lmax - 1]``.

        Notes
        -----
        Reporting only. Nothing here touches the covariance.

        A TT-only run has no polarised leg, and a run with a B observable
        uses the per-Wick-term normalisation (docs/theory/bmode_kernels.md),
        which is exact at ``l*`` and carries no Eq. 23 leakage bias of the
        ``(n_E / 2) lambda`` form; its measured accuracy off ``l*`` is
        docs/theory/bmode_kernels.md, "Known limits". For both the leakage
        and translation sections are reported as not applicable
        (``"applicable": False`` in the dict) and only the normalisation
        check is computed.
        """
        from . import acc_budget

        return acc_budget.error_budget(self, covariance_keys, band_edges=band_edges)

    def get_covariance_coupling_save_path(
        self, stokes_key: str | tuple, ell: int, ell_prime: int
    ) -> str:
        """Get path for saving covariance coupling kernel."""
        return acc_cache.coupling_save_path(
            self.cov.acc_kernel_dir, stokes_key, ell, ell_prime, COUPLING_CHANNELS
        )

    def get_covariance_coupling_manifest_path(self, ell: int, ell_prime: int) -> str:
        """Path of the manifest written alongside the kernels of one ``(ell, ell_prime)`` pair."""
        return acc_cache.coupling_manifest_path(self.cov.acc_kernel_dir, ell, ell_prime)

    def get_covariance_coupling_manifest(self, ell: int, ell_prime: int) -> dict | None:
        """
        Read the manifest :func:`precompute_acc_kernels` writes next to
        the kernels of one ``(ell, ell_prime)`` pair, or ``None`` if there is
        none (a cache written before the manifest existed, or the pair was
        never computed).

        The manifest records ``spectra`` (the requested subset of
        :data:`COUPLING_SPECTRA` that pair was computed for), ``grid``,
        ``lw``, ``nside``, ``centralell`` and ``git_hash`` -- enough for
        :meth:`get_covariance_coupling` to tell "this cache was built for a
        smaller spectrum set" apart from "this file is missing because
        something went wrong".
        """
        return acc_cache.read_coupling_manifest(self.cov.acc_kernel_dir, ell, ell_prime)

    def get_covariance_coupling(
        self,
        ell: int,
        ell_prime: int,
        pairs: Iterable[tuple[str, str]] | None = None,
    ) -> dict:
        """
        Load covariance coupling kernels.

        Every load is validated against this strategy's ``Cov`` (its mask
        and ``config.centralell``) via
        :meth:`~cmbcov.covariance.Cov.get_acc_coupling_kernels`,
        not only when a file is missing -- a mismatch raises ``ValueError``
        naming the field; a cache that predates mask-identity recording (or
        has no manifest at all) cannot be fully checked and instead warns
        once per kernel directory (``Cov.acc_kernel_dir``) per process.
        Repeated loads of the same ``(ell, ell_prime)`` pair are served from
        an in-memory memo on ``Cov`` after the first (validated) read; see
        that method.

        Parameters
        ----------
        pairs : iterable of (str, str), optional
            The ``(stokes_1, stokes_2)`` kernel pairs to load, each a member
            of :data:`COUPLING_CHANNELS`. Defaults to today's sixteen
            ``{TT,DD,TD,DT}^2`` pairs (:data:`_DEFAULT_KERNEL_STOKES`),
            unchanged from before this argument existed. A caller that knows
            it needs only a subset -- :meth:`compute_covariance_term` asks
            for exactly the two pairs its Wick contractions use, via
            :meth:`~cmbcov.keys.CovKey.key_to_cross_kernel`
            -- skips loading the rest.

        Raises
        ------
        ValueError
            If a manifest's recorded mask digest, ``nside``, ``lw``,
            ``grid`` or ``centralell`` does not match this run (see above).
        OSError
            If a kernel file is missing.  The message names the missing pair.
            If a manifest exists for ``(ell, ell_prime)`` (written by
            :func:`precompute_acc_kernels`) and it shows the cache was
            built for a spectrum set not covering this pair, the message says
            so and that the cache must be recomputed -- there is never a
            silent substitution. Otherwise, an ET file that is missing while
            its TE counterpart exists means the cache predates ET kernels
            being computed separately from TE; such kernels must also be
            recomputed, with no fallback.
        """
        stokes_specs = _DEFAULT_KERNEL_STOKES
        default_pairs = [(s1, s2) for s1 in stokes_specs for s2 in stokes_specs]
        return self.cov.get_acc_coupling_kernels(
            ell,
            ell_prime,
            COUPLING_CHANNELS,
            default_pairs,
            pairs=pairs,
            legacy_names=LEGACY_CHANNEL_NAMES,
        )

    def compute_acc_term(
        self,
        cl1_dict: dict,
        cl2_dict: dict,
        stokes_key: tuple,
        central_ell: int,
        ell1: int,
        ell2: int,
        covariance_coupling: dict | None = None,
        kernel_key: tuple | None = None,
    ) -> float:
        """
        Compute single ACC covariance term.

        Parameters
        ----------
        stokes_key : tuple
            ``(stokes_1, stokes_2)`` used to look up the POWER SPECTRA,
            ``cl1_dict[stokes_1]`` and ``cl2_dict[stokes_2]``. Order-
            independent (``"TE"`` and ``"ET"`` are the same spectrum).
        kernel_key : tuple, optional
            ``(stokes_1, stokes_2)`` used to look up the covariance-coupling
            KERNEL, ``covariance_coupling[kernel_key]``. Not order-
            independent away from the diagonal (``"TE"`` and ``"ET"`` are
            different kernels for ``ell1 != ell2``); defaults to
            ``stokes_key`` for backward compatibility, but callers indexing
            a mixed T/E pair should pass the true Wick-contraction order
            here (see ``CovKey.key_to_cross_kernel``).
        """
        assert max(ell1, ell2) < self.cov.lmax

        stokes_1, stokes_2 = stokes_key
        if kernel_key is None:
            kernel_key = stokes_key

        if covariance_coupling is None:
            if not getattr(self, "_warned_scalar_load", False):
                self._warned_scalar_load = True
                warnings.warn(
                    "compute_acc_term loads the coupling kernels itself when "
                    "covariance_coupling is not given (first: "
                    f"{central_ell, central_ell + ell2 - ell1}); warned once per "
                    "strategy",
                    UserWarning,
                    stacklevel=2,
                )
            covariance_coupling = self.get_covariance_coupling(
                central_ell, central_ell + ell2 - ell1, pairs=[kernel_key]
            )

        coupling_matrix = covariance_coupling[kernel_key]
        size = coupling_matrix.shape[0]

        # Normalize coupling if needed (Eq. 23: Theta-bar has unit sum)
        if not np.isclose(np.sum(coupling_matrix), 1.0):
            coupling_matrix = coupling_matrix / np.sum(coupling_matrix)

        # Translation invariance (Eq. 33): the kernel computed at
        # (central_ell, central_ell + Delta) is reused at (ell1, ell2) by
        # shifting its L indices by ell_min - central_ell.  Kernel index i
        # therefore stands for the absolute multipole L = i + shift.  Where
        # that window runs below L = 0 there is no spectrum, which is the
        # same as dropping those rows and columns of the kernel.  At the high
        # end the spectra hold lmax_int = lmax + acc_window_pad(size,
        # central_ell) multipoles, which reaches past every window of an
        # element with ell1, ell2 < lmax: nothing is cut there.
        lmax = self.cov.lmax
        for name, spectrum in (
            (f"cl1_dict[{stokes_1!r}]", cl1_dict[stokes_1]),
            (f"cl2_dict[{stokes_2!r}]", cl2_dict[stokes_2]),
        ):
            _check_spectrum_length(spectrum, name, lmax, size, central_ell)
        shift = min(ell1, ell2) - central_ell
        lo = max(0, shift)
        hi = shift + size
        if hi <= lo:
            return 0.0
        section = coupling_matrix[lo - shift : hi - shift, lo - shift : hi - shift]

        return cl1_dict[stokes_1][lo:hi] @ section @ cl2_dict[stokes_2][lo:hi]

    def _empty_flatten_cov(self, dmax: int) -> np.ndarray:
        """
        Create empty flattened covariance array for ACC computation.

        ACC stores covariance in a flattened diagonal format for efficiency,
        representing only the non-zero diagonal bands of the matrix.
        """
        return np.zeros((2 * (dmax - 1) + 1, self.cov.lmax))

    def _unflatten_cov(self, flattened_cov: np.ndarray) -> np.ndarray:
        """
        Convert flattened covariance to full matrix for ACC computation.

        This reconstructs the full covariance matrix from ACC's flattened
        diagonal representation: row ``dmax - 1 + d`` of ``flattened_cov``
        is diagonal ``+d`` (``out[l, l + d]``), row ``dmax - 1 - d`` is
        diagonal ``-d`` (``out[l + d, l]``), each read over its first
        ``lmax - d`` entries.

        Each band is added in place into one preallocated matrix through a
        strided view of its diagonal (diagonal ``+d`` of the C-contiguous
        ``lmax x lmax`` buffer is ``flat[d::lmax + 1]``, diagonal ``-d`` is
        ``flat[d * lmax::lmax + 1]``).  The form this replaced summed
        ``2 dmax - 1`` full ``np.diag`` matrices (~0.1 s per block at
        ``lmax = 3000``).  The in-place ``+=`` onto zeros is the same IEEE
        operation, ``0.0 + x``, that sum performed for every entry (the
        bands never overlap), so the result is bit-identical, signed zeros
        included.  Offsets ``d >= lmax`` have no entries and are skipped
        (the old form raised a broadcast error for ``d > lmax``).
        """
        dmax = (flattened_cov.shape[0] - 1) // 2 + 1
        lmax = self.cov.lmax
        output_matrix = np.zeros((lmax, lmax))
        flat = output_matrix.reshape(-1)  # a view: output_matrix is contiguous

        for diagonal_offset in range(min(dmax, lmax)):
            n = lmax - diagonal_offset
            flat[diagonal_offset :: lmax + 1][:n] += flattened_cov[
                dmax - 1 + diagonal_offset, :n
            ]
            if diagonal_offset > 0:
                flat[diagonal_offset * lmax :: lmax + 1][:n] += flattened_cov[
                    dmax - 1 - diagonal_offset, :n
                ]

        return output_matrix


# ---------------------------------------------------------------------------
# The one-off precompute: coupling kernels from the mask alone
# ---------------------------------------------------------------------------


def coupling_ellprange(centralell: int, dmax: int) -> list[int]:
    """
    The ``ellp`` values whose kernels :meth:`ACCStrategy.compute_covariance_term`
    loads for a run with this ``centralell`` and ``dmax``.

    The recompute loop asks, for every diagonal offset ``d`` in
    ``range(dmax)``, for the kernels of the pair ``(centralell,
    centralell + d)`` (Eq. 33: the kernel at ``(l*, l* + Delta)`` stands in
    for every ``(l, l + Delta)`` of that diagonal).  So the precompute that
    serves it is ``ell = centralell`` with ``ellprange = [centralell,
    centralell + 1, ..., centralell + dmax - 1]`` -- no more, no fewer.
    """
    return [centralell + d for d in range(dmax)]


class _CouplingPrecompute:
    """
    The ACC coupling-kernel precompute for one mask.

    Needs only the mask (a :class:`~cmbcov.mask.MaskWlm`):
    no ``Cov``, no ``CovarianceConfig``.  :func:`precompute_acc_kernels` is
    the public entry point; the methods below are the stages it runs, kept
    separately addressable for the tests that check the blocked contraction
    against its naive reference.
    """

    def __init__(self, wlm: MaskWlm) -> None:
        self.wlm = wlm
        self._gl_modes_cache: _GLMaskModes | None = None

    def compute(
        self,
        ell: int,
        ellprange: np.ndarray | list[int],
        *,
        save_dir: str | None,
        centralell: int,
        nside: int | None = None,
        dryrun: bool = False,
        verbose: bool = False,
        grid: str = "gl",
        lw: int | None = None,
        max_memory_gb: float | None = None,
        spectra: Sequence[str] | None = None,
        pairs: Sequence[tuple[str, str]] | None = None,
        term_selection: float | None = None,
        scratch_dir: str | None = None,
    ) -> dict | None:
        """
        Compute the coupling kernels of ``(ell, ellp)`` for every ``ellp`` in
        ``ellprange``; write them under ``save_dir`` (with a manifest
        recording ``centralell``) unless ``dryrun``, in which case they are
        returned as ``{ellp: {"s1xs2": kernel}}``.  See
        :func:`precompute_acc_kernels` for the arguments.

        ``term_selection=None`` (default) leaves every code path as it was.
        With a tolerance (``grid="gl"`` only) the mask alm is first rotated
        to the pole frame -- the kernels are rotation invariant (the same
        ``K`` to 1e-14 in either frame, ``sparsity.py`` / ``docs``), the
        selection rules are only sharp there -- and every ``(ell, ellp)``
        pair is computed from the terms
        :func:`~cmbcov.term_selection.select_terms`
        keeps, as described in :class:`_TermSelectionPlan`.

        A central set that does not fit in half the memory budget is built
        once into a :class:`_CentralStore` on disk (:meth:`_central_store`,
        under ``scratch_dir``) and read back for every ``ellp``; the store
        is deleted when this returns or raises.
        """
        spectra_tuple, spec_indices, t_only = _resolve_spectra(spectra)

        pairs_tuple = None
        if pairs is not None:
            pairs_tuple = tuple(
                (normalise_channel(a), normalise_channel(b)) for a, b in pairs
            )
            if not pairs_tuple:
                raise ValueError("pairs must not be empty")
            pair_channels = {c for ab in pairs_tuple for c in ab}
            if spectra is None:
                # No explicit spectra: the channel set is exactly what the
                # pairs reference.
                spectra_tuple = tuple(
                    s for s in COUPLING_CHANNELS if s in pair_channels
                )
                spec_indices = [COUPLING_CHANNELS.index(s) for s in spectra_tuple]
                fields_needed = {_CENTRAL_FIELD[i] for i in spec_indices} | {
                    _PRIME_FIELD[i] for i in spec_indices
                }
                t_only = fields_needed == {0}
            else:
                missing = pair_channels - set(spectra_tuple)
                if missing:
                    raise ValueError(
                        f"pairs references channel(s) {sorted(missing)} not in "
                        f"spectra {spectra_tuple}; add them to spectra or drop "
                        "them from pairs"
                    )

        if term_selection is not None and grid != "gl":
            raise ValueError(
                "term_selection is implemented for grid='gl' only "
                f"(got grid={grid!r})"
            )

        if term_selection is not None:
            requested = (
                {c for ab in pairs_tuple for c in ab}
                if pairs_tuple is not None
                else set(spectra_tuple)
            )
            leaky = sorted(requested & _LEAKAGE_CHANNELS)
            if leaky:
                # Once per precompute: this method runs once per call.
                warnings.warn(
                    f"term_selection={term_selection:g} with E->B leakage kernels "
                    f"({', '.join(leaky)}): the tolerance is not guaranteed for "
                    "the kernels that involve the leakage field on masks that "
                    "are not azimuthally symmetric about their centre; errors "
                    "up to ~10x the tolerance were measured. Use "
                    "term_selection=None to avoid this",
                    UserWarning,
                    stacklevel=3,
                )

        # Validate inputs and setup parameters
        nside, wn_for_ell, lmax, mask_alm, lw = self._setup_coupling_computation(
            ell, ellprange, verbose, nside=nside, grid=grid, lw=lw
        )
        if max_memory_gb is None:
            max_memory_gb = DEFAULT_COUPLING_MEMORY_GB
        max_memory_bytes = int(max_memory_gb * 1024**3)

        selection = None
        if term_selection is not None:
            # Pole frame: the kernels do not depend on it, the selection does.
            mask_alm = rotate_alm(mask_alm, lw, pole_rotation(self.wlm.mask))
            selection = _TermSelectionPlan(
                mask_alm, lw, lmax, ell, float(term_selection), t_only
            )
            if verbose:
                print(
                    f"Term selection at tolerance {term_selection:g}: mask rotated "
                    f"to the pole frame, M band |M - m| <= {selection.m_band}"
                )

        # Compute central I_lm integrals
        integral_mask = self._compute_central_integrals(
            ell,
            wn_for_ell,
            nside,
            grid=grid,
            mask_alm=mask_alm,
            lw=lw,
            lmax=lmax,
            max_memory_bytes=max_memory_bytes,
            t_only=t_only,
            selection=selection,
        )
        store = None
        if integral_mask is None and len(ellprange) > 0:
            if scratch_dir is None and save_dir is not None and not dryrun:
                scratch_dir = save_dir
            store = self._central_store(
                ell,
                grid=grid,
                wn_for_ell=wn_for_ell,
                nside=nside,
                mask_alm=mask_alm,
                lw=lw,
                lmax=lmax,
                t_only=t_only,
                selection=selection,
                max_memory_bytes=max_memory_bytes,
                scratch_dir=scratch_dir,
                verbose=verbose,
            )
        if verbose:
            if integral_mask is not None:
                print("Central I_lm: held in RAM")
            elif store is not None:
                print(
                    f"Central I_lm: disk store of {store.nbytes / 1024**3:.2f} GiB "
                    f"at {store.path}"
                )
            else:
                print("Central I_lm: streamed per block")
        try:
            return self._compute_ellp_loop(
                ell,
                ellprange,
                integral_mask if store is None else store,
                save_dir=save_dir,
                centralell=centralell,
                dryrun=dryrun,
                verbose=verbose,
                wn_for_ell=wn_for_ell,
                nside=nside,
                lmax=lmax,
                grid=grid,
                mask_alm=mask_alm,
                lw=lw,
                max_memory_bytes=max_memory_bytes,
                spec_indices=spec_indices,
                t_only=t_only,
                selection=selection,
                pairs_tuple=pairs_tuple,
                spectra_tuple=spectra_tuple,
                term_selection=term_selection,
            )
        finally:
            if store is not None:
                store.close()
                if verbose:
                    print(f"Removed the central store {store.directory}")

    def _compute_ellp_loop(
        self,
        ell: int,
        ellprange,
        integral_mask,
        *,
        save_dir,
        centralell,
        dryrun,
        verbose,
        wn_for_ell,
        nside,
        lmax,
        grid,
        mask_alm,
        lw,
        max_memory_bytes,
        spec_indices,
        t_only,
        selection,
        pairs_tuple,
        spectra_tuple,
        term_selection,
    ) -> dict | None:
        """The loop over ``ellp`` of :meth:`compute`: one
        :meth:`_compute_ellp_coupling` per value, saved or collected."""
        all_results = {} if dryrun else None
        integral_mask_cache = {}

        for indellp, ellp in enumerate(ellprange):
            if verbose:
                print(f"Processing ellp={ellp} ({indellp+1}/{len(ellprange)})")

            # Compute coupling kernels for this ellp
            coupling_kernels = self._compute_ellp_coupling(
                ell,
                ellp,
                integral_mask,
                integral_mask_cache,
                ellprange[indellp + 1 :],
                wn_for_ell,
                nside,
                lmax,
                verbose,
                grid=grid,
                mask_alm=mask_alm,
                lw=lw,
                max_memory_bytes=max_memory_bytes,
                spec_indices=spec_indices,
                t_only=t_only,
                selection=selection,
                pairs=pairs_tuple,
            )

            # Save or store results
            if not dryrun:
                self._save_coupling_kernels(
                    save_dir,
                    coupling_kernels,
                    ell,
                    ellp,
                    centralell,
                    verbose,
                    spectra=spectra_tuple,
                    grid=grid,
                    lw=lw,
                    nside=nside,
                    term_selection=term_selection,
                    term_selection_stats=(
                        None if selection is None else selection.stats(ell, ellp)
                    ),
                )
            else:
                all_results[ellp] = coupling_kernels.copy()

        # Clean up and return
        del integral_mask_cache
        return all_results if dryrun else None

    def _setup_coupling_computation(
        self,
        ell: int,
        ellprange: np.ndarray | list[int],
        verbose: bool = False,
        nside: int | None = None,
        grid: str = "gl",
        lw: int | None = None,
    ) -> tuple:
        """Setup and validate parameters for coupling computation."""
        # Input validation
        if not isinstance(ellprange, (list, np.ndarray)):
            raise TypeError(f"ellprange must be list or array, got {type(ellprange)}")
        if ell < 0:
            raise ValueError(f"ell must be non-negative, got {ell}")
        if any(ellp < 0 for ellp in ellprange):
            raise ValueError("All ellprange values must be non-negative")
        if grid not in ("healpix", "gl"):
            raise ValueError(f"grid must be 'healpix' or 'gl', got {grid!r}")

        if nside is None:
            nside = get_nside_from_ell(max(ell, np.max(ellprange)))

        mask_alm = None
        if grid == "healpix":
            wn_for_ell, nside = self.wlm.degrade_mask(nside)
        else:
            # GL path: exact for a band-limited mask, so use the mask's own
            # alm (obtained once here, iter=10 for an accurate band-limit
            # truncation) rather than a HEALPix-degraded pixel map.
            wn_for_ell = None
            if lw is None:
                lw = 3 * nside - 1
            mask_alm = ducc0_map2alm(self.wlm.mask, lmax=lw, pol=False, iter=10)

        lmax = 2 * nside

        if verbose:
            print(f"Computing covariance coupling for ell={ell}, ellprange={ellprange}")
            print(f"Using nside={nside}, lmax={lmax}, grid={grid}")
            full_m = 3 * lmax * (2 * lmax - 1) * 16
            print(
                f"Full-M coefficients: {full_m / 1024**2:.1f} MB per m, "
                f"{(2 * ell + 1) * full_m / 1024**3:.2f} GiB for the whole "
                "central set"
            )

        return nside, wn_for_ell, lmax, mask_alm, lw

    def _gl_mask_modes(
        self, mask_alm: np.ndarray, lw: int, lmax_out: int
    ) -> "_GLMaskModes":
        """The per-grid mask-mode cache of this precompute (:class:`_GLMaskModes`),
        kept on the instance so that every :meth:`_integral_source` of one
        run -- central set, store, each ``ellp`` -- shares it; rebuilt when
        called with a different mask alm (by identity), ``lw`` or
        ``lmax_out``."""
        cache = getattr(self, "_gl_modes_cache", None)
        if cache is None or not cache.matches(mask_alm, lw, lmax_out):
            cache = _GLMaskModes(mask_alm, lw, lmax_out)
            self._gl_modes_cache = cache
        return cache

    def _integral_source(
        self,
        grid: str,
        wn_for_ell: np.ndarray | None,
        nside: int,
        mask_alm: np.ndarray | None,
        lw: int | None,
        lmax: int,
        t_only: bool,
        selection: "_TermSelectionPlan | None" = None,
    ) -> dict:
        """
        The grid-specific half of the coefficient pipeline, as plain callables
        consumed by :meth:`_compute_central_integrals`,
        :meth:`_compute_ellp_coupling` and :func:`_coefficient_provider`:

        ``synthesise(ell, m)``
            one per-``m`` integral set in the grid's storage form (GL:
            ``(u_T, u_EB)`` tuple; HEALPix: compact ``(2, 3, nalm)``).
        ``materialise(ell)``
            the whole set of one ``ell``, indexable by ``m + ell``.
        ``nbytes(ell)``
            the size of that whole set, for the memory-budget rule.
        ``to_full_m(item)``
            complex128 ``(nfields, lmax, 2*lmax-1)`` full-``M`` stack.
        ``fields(item)`` (GL) / ``to_full_m_into(item, out)`` (HEALPix)
            optional shortcuts to the same numbers without the intermediate
            stack (:func:`_coefficient_provider`).
        ``reflect``
            whether the ``+-m`` reflection shortcut may be used (GL only).

        With a ``selection`` (GL only) the integrals are the banded ones of
        :func:`~cmbcov.grid.banded_integrals_gl` at the
        plan's ``m_band`` and the plan's per-grid mask modes, and
        ``materialise`` fills only the kept orders (``None`` elsewhere).
        """
        import healpy as hp  # lazy, see utils/healpy_utils.py

        nfields = 1 if t_only else 3
        if grid == "healpix":
            if selection is not None:
                raise ValueError("term_selection is implemented for grid='gl' only")
            nalm = hp.Alm.getsize(2 * nside - 1)
            wlm = self.wlm
            return {
                "synthesise": lambda ell, m: wlm.compute_spin_weighted_integrals(
                    ell, m, mask_for_ell=wn_for_ell, target_nside=nside
                ),
                "materialise": lambda ell: _healpix_integrals(
                    wlm, wn_for_ell, nside, ell
                ),
                "nbytes": lambda ell: (2 * ell + 1) * 2 * 3 * nalm * 16,
                "to_full_m": lambda cma: _healpix_full_m(cma, lmax, nfields),
                "to_full_m_into": lambda cma, out: full_from_pair(
                    cma[0, :nfields], cma[1, :nfields], lmax - 1, out=out
                ),
                "reflect": False,
            }

        def gl_fields(item):
            u_t, u_eb = item
            return (u_t,) if t_only else (u_t, u_eb[0], u_eb[1])

        if selection is None:
            # Default producer (_GL_PRODUCER): the mask modes of each grid are
            # computed once per precompute and shared by every m, both sides
            # and every ellp on that grid (_GLMaskModes); the two-SHT
            # reference route does not use them.
            if _GL_PRODUCER == "two_sht":

                def modes(ell):
                    return None

            else:
                modes = self._gl_mask_modes(mask_alm, lw, lmax - 1)
            return {
                "synthesise": lambda ell, m: _gl_integrals_m(
                    mask_alm, lw, ell, m, lmax, t_only=t_only, mask_modes=modes(ell)
                ),
                "materialise": lambda ell: _gl_integrals(
                    mask_alm, lw, ell, lmax, t_only=t_only, mask_modes=modes(ell)
                ),
                "nbytes": lambda ell: (2 * ell + 1)
                * nfields
                * lmax
                * (2 * lmax - 1)
                * 16,
                "to_full_m": lambda item: _gl_full_m(item, t_only),
                "fields": gl_fields,
                "reflect": True,
            }
        plan = selection
        return {
            "synthesise": lambda ell, m: _gl_integrals_m(
                plan.mask_alm,
                plan.lw,
                ell,
                m,
                lmax,
                t_only=t_only,
                m_band=plan.m_band,
                mask_modes=plan.mask_modes(ell),
            ),
            "materialise": lambda ell: _gl_integrals(
                plan.mask_alm,
                plan.lw,
                ell,
                lmax,
                t_only=t_only,
                keep_m=plan.keep_m(ell),
                m_band=plan.m_band,
                mask_modes=plan.mask_modes(ell),
            ),
            "nbytes": lambda ell: int(plan.keep_m(ell).sum())
            * nfields
            * lmax
            * (2 * lmax - 1)
            * 16,
            "to_full_m": lambda item: _gl_full_m(item, t_only),
            "fields": gl_fields,
            "reflect": True,
        }

    def _compute_central_integrals(
        self,
        ell: int,
        wn_for_ell: np.ndarray | None,
        nside: int,
        grid: str = "gl",
        mask_alm: np.ndarray | None = None,
        lw: int | None = None,
        lmax: int | None = None,
        max_memory_bytes: int | None = None,
        t_only: bool = False,
        selection: "_TermSelectionPlan | None" = None,
    ) -> np.ndarray | list[tuple] | None:
        """
        Compute the central I_lm integrals, to be reused across every
        ``ellp``, or return ``None`` when they do not fit the memory budget.

        For grid="healpix" the set is the compact ``(2*ell+1, 2, 3, nalm)``
        complex array (:func:`_healpix_integrals`), one ``(2, 3, nalm)``
        :meth:`~cmbcov.mask.MaskWlm.compute_spin_weighted_integrals`
        result per ``m``; it is expanded to full-``M`` one ``m`` at a time by
        :func:`_healpix_full_m`.  For grid="gl" it is a list with one
        ``(u_T, u_EB)`` tuple per m (:func:`_gl_integrals`): the spin-0 full-M
        array and the ``(2, lmax, 2*lmax-1)`` (E, B) response to the E-mode
        unit vector, or ``(u_T, None)`` when ``t_only=True``.

        Same rule on both grids: the set is materialised only when its size
        fits in half the memory budget -- GL ``(2l+1) * 3 * lmax * (2 lmax -
        1)`` complex128, 12.6 GB at ``ell = 250``, ``lmax = 512``; HEALPix
        ``(2l+1) * 2 * 3 * nalm`` complex128, about half that.  Above it the
        return value is ``None``; :meth:`compute` then builds the set once
        into a disk store (:meth:`_central_store`), or -- when no store can
        be written -- each block of ``m`` is synthesised on demand inside
        :meth:`_compute_ellp_coupling`.

        ``t_only=True`` means the caller has determined via
        :func:`_resolve_spectra` that every kernel it asked for needs only the
        T field.  On the GL branch it skips the spin-2 (E, B) integrals
        entirely (two Legendre components against one for T; ~92% of the
        precompute wall time at ``centralell=250``, ``nside=256`` on the
        earlier two-SHT producer).  The
        HEALPix branch computes T, E and B in one joint ``map2alm`` per ``m``
        and has no cheaper T-only synthesis; there ``t_only`` only restricts
        the full-``M`` expansion (and the budget of the contraction) to T.
        """
        if max_memory_bytes is None:
            max_memory_bytes = int(DEFAULT_COUPLING_MEMORY_GB * 1024**3)
        if lmax is None:
            lmax = 2 * nside
        source = self._integral_source(
            grid, wn_for_ell, nside, mask_alm, lw, lmax, t_only, selection=selection
        )
        if source["nbytes"](ell) > max_memory_bytes // 2:
            return None
        return source["materialise"](ell)

    def _central_store(
        self,
        ell: int,
        *,
        grid: str,
        wn_for_ell: np.ndarray | None,
        nside: int,
        mask_alm: np.ndarray | None,
        lw: int | None,
        lmax: int,
        t_only: bool,
        selection: "_TermSelectionPlan | None",
        max_memory_bytes: int,
        scratch_dir: str | None,
        verbose: bool = False,
    ) -> "_CentralStore | None":
        """
        Build the central full-``M`` set of ``ell`` once into a
        :class:`_CentralStore` -- a file in a fresh temporary directory under
        ``scratch_dir`` (the system temporary directory when ``None``) --
        for every ``ellp`` to read back, instead of re-synthesising it per
        ``ellp``.  The orders stored are those the contraction iterates
        over: all ``2 ell + 1``, or the kept ones under term selection
        (:meth:`_TermSelectionPlan.keep_m` depends on ``ell`` alone), in the
        paired order of :func:`_paired_m_order` so the reflection of
        :func:`_coefficient_provider` halves the transforms here too.

        Returns ``None``, with a :class:`UserWarning`, when the file system
        of ``scratch_dir`` has less free space than the store plus 1 GiB;
        the caller then streams the central set per block as before.  On any
        exception during the build the directory is removed.
        """
        source = self._integral_source(
            grid, wn_for_ell, nside, mask_alm, lw, lmax, t_only, selection=selection
        )
        nfields = 1 if t_only else 3
        order = _paired_m_order(2 * ell + 1)
        if selection is not None:
            keep = selection.keep_m(ell)
            order = [i for i in order if keep[i]]
        per_m = nfields * lmax * (2 * lmax - 1) * np.dtype(np.complex128).itemsize
        nbytes = len(order) * per_m

        if scratch_dir is not None:
            os.makedirs(scratch_dir, exist_ok=True)
        parent = scratch_dir if scratch_dir is not None else tempfile.gettempdir()
        free = shutil.disk_usage(parent).free
        if free < nbytes + 1024**3:
            warnings.warn(
                f"the central coefficient store needs {nbytes / 1024**3:.2f} GiB "
                f"but {parent} has {free / 1024**3:.2f} GiB free; streaming the "
                "central set per block instead (slower). Pass scratch_dir to "
                "put the store elsewhere.",
                UserWarning,
                stacklevel=3,
            )
            return None

        directory = tempfile.mkdtemp(prefix="acc_central_store_", dir=scratch_dir)
        store = None
        try:
            store = _CentralStore(directory, order, nfields, lmax)
            provider = _coefficient_provider(
                None,
                lambda m: source["synthesise"](ell, m),
                source["to_full_m"],
                ell,
                reflect=source["reflect"],
                fields=source.get("fields"),
                into=source.get("to_full_m_into"),
            )
            chunk = min(len(order), 64, max(1, (max_memory_bytes // 4) // per_m))
            if verbose:
                print(
                    f"Building the central store: {len(order)} orders, "
                    f"{nbytes / 1024**3:.2f} GiB, in {store.path}"
                )
            store.fill(provider, chunk)
        except BaseException:
            if store is not None:
                store.close()
            else:
                shutil.rmtree(directory, ignore_errors=True)
            raise
        return store

    def _compute_ellp_coupling(
        self,
        ell: int,
        ellp: int,
        integral_mask: np.ndarray | list[tuple] | None,
        integral_mask_cache: dict,
        remaining_ellp_values: np.ndarray,
        wn_for_ell: np.ndarray | None,
        nside: int,
        lmax: int,
        verbose: bool = False,
        grid: str = "gl",
        mask_alm: np.ndarray | None = None,
        lw: int | None = None,
        max_memory_bytes: int | None = None,
        spec_indices: Sequence[int] | None = None,
        t_only: bool = False,
        selection: "_TermSelectionPlan | None" = None,
        pairs: Sequence[tuple[str, str]] | None = None,
    ) -> dict:
        r"""
        Compute the coupling kernels for one ``ellp``, restricted to
        ``spec_indices`` (indices into :data:`COUPLING_CHANNELS`; ``None``
        means all five, i.e. all 25 ordered pairs -- the original behaviour).

        ``pairs`` (channel-name pairs, normalised to the new names -- see
        :func:`normalise_channel`) restricts the output further, to an
        explicit list of ordered pairs instead of the full square of
        ``spec_indices``: see :func:`_accumulate_coupling_kernels`,
        ``output_pairs``.

        Both grids go through one code path: a :func:`_coefficient_provider`
        per side hands complex full-``M`` coefficients to
        :func:`_accumulate_coupling_kernels`, which replaces the ``(m, m')``
        double loop with one GEMM per ``L`` and accumulates straight into the
        kernels, so the ``(nspec, 2l+1, 2l'+1, lmax)`` ``Theta`` table is never
        built.  Only the source of the integrals differs
        (:meth:`_integral_source`).

        ``Theta`` is complex on both grids.  For a map split into the
        ``(r, s)`` healpy alm pair of its real and imaginary parts (the
        HEALPix integrals),

            ``Re(sum_M x_LM conj(y_LM)) = sum_M r_x conj(r_y) + s_x conj(s_y)``
            ``= ls[L] * (alm2cl(x[0], y[0]) + alm2cl(x[1], y[1]))[L]`` ,

        which the HEALPix branch builds; the imaginary
        part ``Im(sum_M x conj(y))`` is also kept, via
        :func:`_healpix_full_m`.  It vanishes for a mask symmetric under
        ``phi -> -phi`` (the test mask is azimuthally symmetric,
        ``tests/test_acc_gl.py`` measures 1e-17) but is comparable to the real
        part for a generic mask, and Eq. 20 needs it (it is
        ``|sum_L C_L Theta(L)|^2``, not ``(sum_L C_L Re Theta(L))^2``).

        ``integral_mask`` is the central set held in RAM (as
        :meth:`_compute_central_integrals` returns it), a
        :class:`_CentralStore` (the set on disk, built once per precompute
        by :meth:`compute` when it does not fit the budget), or ``None``,
        meaning "synthesise each ``m`` on demand" (the streamed path, kept
        for when no store can be written).  Held or stored, the central
        blocks are re-read inside the loop over primed blocks, so every
        primed coefficient is synthesised once; streamed, the central blocks
        are the outer loop and the primed ones are rebuilt per central
        block (:func:`_accumulate_coupling_kernels`).  A primed set
        (``ell != ellp``) is materialised only when a later ``ellp`` in
        ``remaining_ellp_values`` reuses it (then kept in
        ``integral_mask_cache``); otherwise it is streamed block by block.  At
        ``ell == ellp`` the central set (held, stored or streamed) serves
        both sides.

        ``max_memory_bytes`` caps the whole-set caches plus the blocked
        working set; ``None`` uses :data:`DEFAULT_COUPLING_MEMORY_GB`.  A
        central set held in RAM and any primed set kept for a later ``ellp``
        are subtracted from it, and the rest bounds
        :func:`_coupling_working_set` (with the producer's transients,
        :func:`_gl_producer_bytes` / :func:`_healpix_producer_bytes`).  A
        primed set is kept only while the sets held stay within half the
        budget, and is dropped at its last use.

        ``selection`` (a :class:`_TermSelectionPlan`, GL only) restricts the
        contraction to the kept orders and pairs of ``(ell, ellp)``.
        """
        if max_memory_bytes is None:
            max_memory_bytes = int(DEFAULT_COUPLING_MEMORY_GB * 1024**3)

        source = self._integral_source(
            grid, wn_for_ell, nside, mask_alm, lw, lmax, t_only, selection=selection
        )
        sel = None if selection is None else selection.selection(ell, ellp)
        if verbose and sel is not None:
            print("  " + sel.summary())
        store = integral_mask if isinstance(integral_mask, _CentralStore) else None
        if store is not None:
            integral_mask = None
        # Whole coefficient sets held in RAM (the central one, primed ones
        # kept for a later ellp) come out of the budget before the
        # contraction is planned.
        held_bytes = source["nbytes"](ell) if integral_mask is not None else 0
        if ell != ellp:
            integral_mask_prime = integral_mask_cache.get(ellp)
            if ellp not in remaining_ellp_values:
                # Last use: dropped from the cache, freed after this pair.
                integral_mask_cache.pop(ellp, None)
            if integral_mask_prime is not None:
                if verbose:
                    print(f"  Reusing cached I_lm for ellp={ellp}")
            elif ellp in remaining_ellp_values:
                # Only materialise the whole primed set when a later ellp
                # will reuse it and the sets held stay within half the
                # budget (the rule of the central set); otherwise it is
                # streamed block by block.
                kept = sum(source["nbytes"](k) for k in integral_mask_cache)
                if held_bytes + kept + source["nbytes"](ellp) <= max_memory_bytes // 2:
                    integral_mask_prime = source["materialise"](ellp)
                    integral_mask_cache[ellp] = integral_mask_prime
                    if verbose:
                        print(f"  Cached I_lm for ellp={ellp} (appears again later)")
            if integral_mask_prime is not None:
                held_bytes += source["nbytes"](ellp)
            held_bytes += sum(
                source["nbytes"](k) for k in integral_mask_cache if k != ellp
            )
        else:
            if verbose:
                print("  Using ell==ellp optimization")
            integral_mask_prime = integral_mask

        nfields = 1 if t_only else 3
        if grid == "gl":
            producer_bytes = _gl_producer_bytes(
                lw if selection is None else selection.lw, lmax, (ell, ellp), nfields
            )
        else:
            producer_bytes = _healpix_producer_bytes(nside, lmax, nfields)

        def provider(held, l_val):
            return _coefficient_provider(
                held,
                lambda m: source["synthesise"](l_val, m),
                source["to_full_m"],
                l_val,
                reflect=source["reflect"],
                fields=source.get("fields"),
                into=source.get("to_full_m_into"),
            )

        central_fn = provider(integral_mask, ell)
        # A central set that is cheap to re-read (disk store or held in RAM)
        # lets the contraction put the primed side in the outer loop, so the
        # primed coefficients are synthesised once (see
        # _accumulate_coupling_kernels, "Loop order").
        central_reader = prime_reader = None
        if store is not None:
            central_reader = store.left
            if ell == ellp:
                prime_reader = store.right
        elif integral_mask is not None:

            def central_reader(rows, out=None):
                return _pack_block(central_fn, rows, lmax, False, nfields, out=out)

        active = (
            list(range(len(COUPLING_SPECTRA))) if spec_indices is None else spec_indices
        )
        local = {global_index: i for i, global_index in enumerate(active)}
        output_pairs = None
        if pairs is not None:
            output_pairs = [
                (local[COUPLING_CHANNELS.index(a)], local[COUPLING_CHANNELS.index(b)])
                for a, b in pairs
            ]
        kernels = _accumulate_coupling_kernels(
            central_fn,
            2 * ell + 1,
            provider(integral_mask_prime, ellp),
            2 * ellp + 1,
            lmax,
            max(0, max_memory_bytes - held_bytes),
            spec_indices=active,
            nfields=nfields,
            verbose=verbose,
            keep_m=None if sel is None else sel.keep_m,
            keep_mp=None if sel is None else sel.keep_mp,
            pair_mask=None if sel is None else sel.keep_pair,
            output_pairs=output_pairs,
            central_reader=central_reader,
            prime_reader=prime_reader,
            producer_bytes=producer_bytes,
        )

        if pairs is not None:
            return {
                f"{a}x{b}": kernels[i1, i2]
                for (a, b), (i1, i2) in zip(pairs, output_pairs)
            }
        return {
            f"{COUPLING_CHANNELS[k1]}x{COUPLING_CHANNELS[k2]}": kernels[i1, i2]
            for i1, k1 in enumerate(active)
            for i2, k2 in enumerate(active)
        }

    def _theta_to_coupling_kernels(
        self,
        theta_before_summing: np.ndarray,
        ell: int,
        ellp: int,
        lmax: int,
        verbose: bool = False,
    ) -> dict:
        """
        Convert theta arrays to coupling kernels.

        ``output["s1xs2"][L1, L2] = Re sum_{m m'} Theta^{s1}(m, m', L1)
        conj(Theta^{s2}(m, m', L2))`` for every ordered pair of
        :data:`COUPLING_SPECTRA`; ``s2xs1`` is the transpose of ``s1xs2``.
        ``Theta`` is complex on both grids.

        .. note::

            No longer on the shipped path: :func:`_accumulate_coupling_kernels`
            performs this reduction block by block, so the full
            ``theta_before_summing`` (5.1 GB at ``ell = ellp = 250``) is never
            materialised.  It is kept as the readable reference form of the
            reduction, and is what ``tests/test_acc_vectorised.py`` checks the
            blocked path against.
        """
        output = {}
        specs = list(COUPLING_SPECTRA)

        # Reshape theta arrays once for efficiency
        reshaped_theta = {}
        flat_size = (2 * ell + 1) * (2 * ellp + 1)
        for k in range(len(specs)):
            reshaped_theta[k] = theta_before_summing[k].reshape((flat_size, lmax))

        def gram(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return np.dot(a.T, np.conj(b)).real

        for k1, sp1 in enumerate(specs):
            for k2, sp2 in enumerate(specs):
                key = f"{sp1}x{sp2}"

                # Use symmetry to reduce computation
                if k2 < k1:
                    output[key] = output[f"{sp2}x{sp1}"].T
                    continue

                if verbose:
                    print(f"  Computing coupling for {key}")

                try:
                    # Use optimized matrix multiplication
                    output[key] = gram(reshaped_theta[k1], reshaped_theta[k2])
                except MemoryError:
                    if verbose:
                        print(
                            f"    Memory error for {key}, trying chunk-wise computation"
                        )
                    # Fallback to chunk-wise computation for large arrays
                    output[key] = np.zeros((lmax, lmax))
                    chunk_size = min(100, flat_size)
                    for i in range(0, flat_size, chunk_size):
                        end_i = min(i + chunk_size, flat_size)
                        output[key] += gram(
                            reshaped_theta[k1][i:end_i], reshaped_theta[k2][i:end_i]
                        )

        # Clean up large arrays to free memory
        del theta_before_summing, reshaped_theta
        return output

    def _save_coupling_kernels(
        self,
        save_dir: str,
        coupling_kernels: dict,
        ell: int,
        ellp: int,
        centralell: int,
        verbose: bool = False,
        spectra: Sequence[str] = COUPLING_SPECTRA,
        grid: str = "gl",
        lw: int | None = None,
        nside: int | None = None,
        term_selection: float | None = None,
        term_selection_stats: dict | None = None,
    ) -> None:
        """
        Save coupling kernels to disk, plus a manifest recording what was
        computed (see :func:`~cmbcov.approximations.acc_cache.read_coupling_manifest`)
        -- including a digest of the mask (see
        :attr:`~cmbcov.mask.MaskWlm.mask_digest`) and, for
        ``grid="healpix"``, the ``map2alm`` iteration count and
        :data:`~cmbcov.approximations.acc_cache.HEALPIX_KERNEL_VERSION`,
        the same fields
        :meth:`~cmbcov.covariance.Cov.get_acc_coupling_kernels`
        validates a later run against.
        """
        acc_cache.write_coupling_kernels(
            save_dir,
            coupling_kernels,
            ell,
            ellp,
            COUPLING_CHANNELS,
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
            term_selection=term_selection,
            term_selection_stats=term_selection_stats,
        )


def precompute_acc_kernels(
    mask: str | MaskWlm,
    save_dir: str | None,
    *,
    centralell: int,
    dmax: int | None = None,
    mask_path: str | None = None,
    nside: int | None = None,
    grid: str = "gl",
    lw: int | None = None,
    spectra: Sequence[str] | None = None,
    pairs: Sequence[tuple[str, str]] | None = None,
    max_memory_gb: float | None = None,
    dryrun: bool = False,
    verbose: bool = False,
    ellprange: Sequence[int] | None = None,
    term_selection: float | None = None,
    scratch_dir: str | None = None,
) -> dict[int, dict[str, np.ndarray]] | None:
    r"""
    Precompute the ACC coupling kernels for one mask -- the one-off step of
    the ACC workflow.  Needs no ``Cov`` and no ``CovarianceConfig``.

    Writes exactly the kernel set a later ACC run with the same mask,
    ``centralell``, ``dmax`` and kernel directory loads
    (:meth:`ACCStrategy.compute_covariance_term`, via
    :meth:`~cmbcov.covariance.Cov.get_acc_coupling_kernels`):
    the pairs ``(centralell, centralell + d)`` for ``d`` in ``range(dmax)``,
    see :func:`coupling_ellprange`.  Files go to
    ``<save_dir>/covariance_coupling/`` through
    :func:`~cmbcov.approximations.acc_cache.write_coupling_kernels`,
    one ``.npy`` per kernel plus a manifest per pair recording the mask
    digest, ``centralell``, ``grid``, ``lw``, ``nside``, ``spectra`` and (for
    ``grid="healpix"``) ``map2alm_iter``, so the run validates the cache
    against itself on every load.

    Parameters
    ----------
    mask : str or MaskWlm
        The mask: a file name looked up in ``mask_path`` (as ``Cov`` does), or
        an already loaded :class:`~cmbcov.mask.MaskWlm`.
    save_dir : str or None
        Directory under which ``covariance_coupling/`` is written.  For a
        direct ``Cov`` run it is ``Cov(save_dir=...)`` (or
        ``Cov(acc_kernel_dir=...)``); for a parameter-file run it is
        :attr:`~cmbcov.generator.parameter_validation.PipelineConfig.acc_kernel_dir`.
        May be ``None`` only with ``dryrun=True``.
    centralell : int
        The reference multipole :math:`\ell_*` of Eq. 33; must match the run's
        ``CovarianceConfig.centralell`` (it is recorded and checked).
    dmax : int
        Number of diagonals the run computes; kernels are built for
        ``ellp = centralell .. centralell + dmax - 1``.  A run with a smaller
        ``dmax`` reuses the cache; a larger one does not find its kernels.
        Exactly one of ``dmax`` and ``ellprange`` must be given.
    mask_path : str, optional
        Directory holding ``mask`` when it is a file name (default ``"./"``).
    nside : int, optional
        ACC working resolution; the kernels are ``(2 nside) x (2 nside)``.
        Defaults to :func:`~cmbcov.utils.healpy_utils.get_nside_from_ell`
        of the largest multipole requested.
    grid : {"healpix", "gl"}, default "gl"
        Quadrature backend for the mode-coupling integrals
        :math:`I_{\ell m, LM}`.  ``"gl"`` (default, since it is already what
        production runs use) computes the integrals exactly (to ~1e-15) on a
        Gauss-Legendre grid for the mask band-limited at ``lw``, for the
        spin-0 (T) and spin-2 (E, B) channels alike.  ``"healpix"`` degrades
        the mask to ``nside`` and uses one ``map2alm`` (``iter=DEFAULT_MAP2ALM_ITER``)
        per integral instead.  Both keep the
        per-``(m, m')`` cross-spectra ``Theta^s(m, m', L)`` complex
        (``output = Re sum_{m m'} Theta^{s1} conj(Theta^{s2})``, Eq. 20)
        through one contraction.  They differ in one convention: the HEALPix
        E/B integrals carry a factor ``sqrt(2)`` (from
        :func:`~cmbcov.sht.cplx_spin_weighted_ylm`), so raw
        HEALPix kernels with ``k`` E/B legs are ``2^(k/2)`` larger; it cancels
        in the Eq. 23 normalisation.  Keeping the full complex ``Theta``
        (rather than only ``Re Theta``) avoids biasing the HEALPix kernels
        for a mask not symmetric under ``phi -> -phi``.
    lw : int, optional
        ``grid="gl"`` only: band-limit of the mask alm used to build the
        integrals.  Defaults to ``3 * nside - 1``.
    spectra : sequence of str, optional
        Subset of :data:`COUPLING_CHANNELS` to compute kernels for (old
        names -- ``TT, EE, BB, TE, ET`` -- are accepted and normalised to
        the new ones, :data:`CHANNEL_ALIASES`); the output is ``spectra x
        spectra`` unless ``pairs`` restricts it further.  ``None`` (default)
        computes :data:`COUPLING_SPECTRA`, the five channels computed before
        B-mode support (25 ordered pairs) -- extending this default would
        silently multiply the precompute's cost, so it stays pinned; the
        four new channels (``TL, LT, DL, LD``) are computed only when
        requested explicitly.  ``("TT",)`` on ``grid="gl"`` skips the spin-2
        integrals entirely (two Legendre components against one for T); on
        ``grid="healpix"`` the synthesis is joint across
        T/D/L, and only the full-``M`` expansion, the contraction and the
        files written shrink.  A run
        that needs a kernel outside the subset fails on load, naming it.
    pairs : sequence of (str, str), optional
        An explicit list of ordered channel pairs (old or new names) to
        compute, instead of the full square of ``spectra`` -- e.g.
        ``[("TT", "TT"), ("DD", "LL")]``.  Restricts both the ``Theta``
        built (only the channels referenced) and the kernels accumulated
        and written (only the pairs themselves).  ``None`` (default) is the
        full square, bit-identical to before this argument existed.  If
        ``spectra`` is also given, every channel ``pairs`` references must
        be in it (``ValueError`` otherwise); if ``spectra`` is ``None``, the
        channel set is inferred from ``pairs`` alone.
    max_memory_gb : float, optional
        Memory budget, in GiB (default :data:`DEFAULT_COUPLING_MEMORY_GB`),
        for everything the precompute holds that scales with the problem:

        * the central coefficient set, when it is held in RAM -- only if it
          fits in half the budget; otherwise it is built once into a
          temporary file (see ``scratch_dir``) and read back for every
          ``ellp`` -- and any primed set kept for a later ``ellp`` (under
          the same half-budget rule);
        * the working set of the blocked contraction, from what is left
          (:func:`_coupling_working_set`): the output kernels, the
          coefficient blocks on both sides including the block read ahead,
          the ``Theta`` slab, and the coefficient producer's transients and
          mask-mode caches.  The ``(m, m')`` block widths are chosen to use
          it.

        Not covered, a fixed baseline on top of it: the Python process and
        its libraries (about 0.1 GiB); the mask map and its alm as loaded
        (8 bytes per mask pixel, 0.4 GiB for an ``nside`` 2048 mask; reading
        and transforming it peaks at about 2 GiB before the contraction
        starts); memory the system allocator keeps after numpy frees it (on
        macOS, whose allocator caches large freed blocks, the survey run
        peaks at 13.1 GiB of footprint (11.4 GiB RSS) at 12 GiB and 8.9
        (7.0) at 6 GiB; ``MallocLargeCache=0`` disables the cache; glibc,
        on Linux, unmaps every allocation above 32 MiB when it is freed);
        and, with
        ``dryrun=True``, the kernels returned, ``nspec**2 * (2 nside)**2 * 8``
        bytes per ``ellp``.  Measured on the survey mask (``centralell``
        250, ``nside`` 256, ``lw`` 875, four spectra, ``MallocLargeCache=0``):
        peak footprint 12.49 GiB at 12 GiB and 6.50 GiB at 6 GiB, i.e. the
        budget plus a 0.5 GiB baseline.  One ``m`` of integrals is 24 MiB
        there (11.7 GiB for the whole central set), and all five channels
        give ``m'`` blocks of 44 orders (12 outer passes over the stored
        central set) at 2 GiB, 173 (3) at 6 GiB and 415 (2) at 12 GiB; at
        24 GiB the central set is held in RAM.  A budget below the floor
        (the output kernels, the producer and one ``m`` on each side) is
        warned about and the smallest blocks are used.
    scratch_dir : str, optional
        Where the temporary central-coefficient store goes when the central
        set does not fit in half of ``max_memory_gb``: a fresh
        ``acc_central_store_*`` subdirectory of it, holding one file of
        ``(2 centralell + 1) * nfields * 2 nside * (4 nside - 1) * 16`` bytes
        (11.7 GiB at ``centralell = 250``, ``nside = 256``, three fields),
        deleted when the precompute returns or raises.  ``None`` (default)
        uses ``save_dir`` (or the system temporary directory with
        ``dryrun=True``).  If the file system lacks the space, a
        :class:`UserWarning` is issued and the set is re-synthesised per
        block instead, as before the store existed.
    dryrun : bool, default False
        Compute but write nothing; return the kernels instead.
    verbose : bool, default False
        Print timing, blocking and progress information.
    ellprange : sequence of int, optional
        Override of the ``ellp`` values, for studies of individual pairs
        (e.g. ``[16, 17, 19]``).  A run loads exactly
        :func:`coupling_ellprange` ``(centralell, dmax)``, so an arbitrary
        ``ellprange`` does not in general produce a usable cache.
    term_selection : float, optional
        ``grid="gl"`` only.  ``None`` (default): the full contraction, every
        code path and every number unchanged.  A positive float is the
        target relative Frobenius error of each kernel and is passed to
        :func:`~cmbcov.term_selection.select_terms`: the
        mask alm is rotated to the pole frame (the kernels are rotation
        invariant -- the same ``K`` to 1e-14 in either frame -- while the
        selection is only sharp there), and per ``(ell, ellp)`` only the
        orders ``m`` with enough mask overlap, the ``(m, m')`` pairs the
        mask's azimuthal autocorrelation couples, and the band ``|M - m|
        <= m_band`` are computed (the integrals through
        :func:`~cmbcov.grid.banded_integrals_gl`).  For
        polarised kernels the spin-0 and spin-2 selections are united so no
        field loses terms.  The tolerance and the fractions kept are
        recorded in the manifest (``term_selection``,
        ``term_selection_stats``) and a run must ask for the same
        ``term_selection`` (``CovarianceConfig.acc_term_selection``) to load
        the cache.  Raises ``ValueError`` with ``grid="healpix"``.

    Returns
    -------
    dict or None
        With ``dryrun=True``, ``{ellp: {"s1xs2": (2 nside, 2 nside) array}}``;
        otherwise ``None``.

    Raises
    ------
    TypeError
        If ``centralell``, ``dmax`` or ``ellprange`` have the wrong type.
    ValueError
        If ``centralell`` is negative, ``dmax < 1``, both or neither of
        ``dmax``/``ellprange`` are given, ``save_dir`` is ``None`` without
        ``dryrun``, ``grid`` is unknown, or ``spectra`` is empty or unknown.
    """
    if isinstance(centralell, (bool, np.bool_)) or not isinstance(
        centralell, (int, np.integer)
    ):
        raise TypeError(f"centralell must be an int, got {type(centralell)}")
    if centralell < 0:
        raise ValueError(f"centralell must be non-negative, got {centralell}")
    if (dmax is None) == (ellprange is None):
        raise ValueError(
            "give exactly one of dmax (the run's diagonal count, the normal "
            "case) and ellprange (an explicit override)"
        )
    if dmax is not None:
        if isinstance(dmax, (bool, np.bool_)) or not isinstance(
            dmax, (int, np.integer)
        ):
            raise TypeError(f"dmax must be an int, got {type(dmax)}")
        if dmax < 1:
            raise ValueError(f"dmax must be at least 1, got {dmax}")
        ellprange = coupling_ellprange(int(centralell), int(dmax))
    elif not isinstance(ellprange, (list, tuple, np.ndarray)):
        raise TypeError(f"ellprange must be list or array, got {type(ellprange)}")
    elif isinstance(ellprange, tuple):
        ellprange = list(ellprange)
    if save_dir is None and not dryrun:
        raise ValueError("save_dir is required unless dryrun=True")

    wlm = mask if isinstance(mask, MaskWlm) else MaskWlm(mask, load_path=mask_path)
    return _CouplingPrecompute(wlm).compute(
        int(centralell),
        ellprange,
        save_dir=save_dir,
        centralell=int(centralell),
        nside=nside,
        dryrun=dryrun,
        verbose=verbose,
        grid=grid,
        lw=lw,
        max_memory_gb=max_memory_gb,
        spectra=spectra,
        pairs=pairs,
        term_selection=term_selection,
        scratch_dir=scratch_dir,
    )
