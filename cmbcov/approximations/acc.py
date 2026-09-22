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
import warnings
from collections.abc import Callable, Iterable, Sequence

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
from . import acc_cache
from .base import CovarianceStrategy

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
#: m') = conj(Theta^{TD}(m', m))`` and every kernel built from ``DT`` equals
#: its ``TD`` counterpart (this was the old ``ET``/``TE`` relation); for
#: ``l != l'`` they differ (measured 3e-3 to 1.6e-2 of the kernel maximum at
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
        integrals entirely; that synthesis is ~92% of the precompute wall
        time at ``centralell=250``, ``nside=256``.
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


#: Default peak-memory budget for the coupling contraction, in GiB.  It caps
#: the blocked working set of :func:`_accumulate_coupling_kernels` (see
#: :func:`precompute_acc_kernels`, ``max_memory_gb``).
DEFAULT_COUPLING_MEMORY_GB = 2.0

#: Primed-block width aimed at when the whole primed set does not fit in the
#: budget.  The primed block is the *inner* loop, so it is rebuilt once per
#: central block; spending the budget on the central block instead keeps the
#: number of rebuilds down, while this floor keeps the per-``L`` GEMMs wide
#: enough to run near BLAS peak.
_PRIME_BLOCK_TARGET = 32


def _gl_banded_grid(lw: int, ell: int, lmax_out: int) -> int:
    """
    The GL grid band-limit :func:`~cmbcov.grid.banded_integrals_gl`
    (and :func:`~cmbcov.grid.spin_weighted_integrals_gl`)
    choose by default for ``(lw, ell, lmax_out)``: ``max(ceil((lw + ell +
    lmax_out - 1) / 2), ell, lmax_out)``.  Restated here so the mask modes
    can be computed once per ``ell`` on exactly that grid and handed in.
    """
    return max(math.ceil((lw + ell + lmax_out - 1) / 2), ell, lmax_out)


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

    ``m_band=None`` (the default path) uses the two-SHT
    :func:`~cmbcov.grid.spin_weighted_integrals_gl`.  With
    an ``m_band`` (term selection) the integrals come from
    :func:`~cmbcov.grid.banded_integrals_gl` restricted to
    ``|M - m| <= m_band`` on the grid :func:`_gl_banded_grid`, with the mask
    modes ``mask_modes`` precomputed on that grid by the caller.
    """
    lmax_out = lmax - 1
    if m_band is None:
        u_t = spin_weighted_integrals_gl(mask_alm, lw, ell, m, lmax_out)
        if t_only:
            return u_t, None
        return u_t, spin_weighted_integrals_gl(mask_alm, lw, ell, m, lmax_out, spin=2)
    lg = _gl_banded_grid(lw, ell, lmax_out)
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
    touches the second element.  This is the dominant cost of the precompute
    (~92% of the wall time at ``centralell=250``, ``nside=256``), so a caller
    that only needs T-field kernels (``spectra=("TT",)``) skips it entirely.

    ``keep_m`` (term selection, index ``m + ell``) leaves ``None`` at the
    orders that are not kept; the contraction never asks for them.
    ``m_band`` / ``mask_modes`` are forwarded to :func:`_gl_integrals_m`.
    """
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
      on the grid :func:`_gl_banded_grid` of ``l``, computed once per grid.
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
        self._modes: dict[int, np.ndarray] = {}
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
        lg = _gl_banded_grid(self.lw, l_val, self.lmax_out)
        modes = self._modes.get(lg)
        if modes is None:
            modes = gl_mask_modes(self.mask_alm, self.lw, lg)
            self._modes[lg] = modes
        return modes

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


def _coefficient_provider(
    held: Sequence | None,
    synthesise: Callable[[int], object],
    to_full_m: Callable[[object], np.ndarray],
    ell: int,
    reflect: bool,
) -> Callable[[int], np.ndarray]:
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
    an ``m`` is asked for.

    ``reflect=True`` (GL only) lets the synthesised case pay one transform per
    ``|m|``.  For a real mask

        ``I_{l,-m,LM} = (-1)^(m+M) conj(I_{l,m,L,-M})``

    (both spins; verified bit-for-bit on the GL grid in
    ``tests/test_acc_vectorised.py``), so the ``m < 0`` half is an index
    reflection of the ``m > 0`` half.  Paired with the ``+m, -m`` iteration
    order of :func:`_paired_m_order`, a one-entry cache turns that into half
    the spherical-harmonic transforms.  The identity has not been verified on
    the HEALPix grid (pixel quadrature, iterated ``map2alm``), so the HEALPix
    branch passes ``reflect=False`` and synthesises every ``m``.
    """
    if held is not None:
        return lambda i: to_full_m(held[i])
    if not reflect:
        return lambda i: to_full_m(synthesise(i - ell))

    cache: dict[int, np.ndarray] = {}

    def from_mask(i: int) -> np.ndarray:
        m = i - ell
        full = cache.get(abs(m))
        if full is None:
            full = to_full_m(synthesise(abs(m)))
            cache.clear()
            cache[abs(m)] = full
        if m >= 0:
            return full
        lmax_out = (full.shape[-1] - 1) // 2
        phase = (-1.0) ** np.arange(-lmax_out, lmax_out + 1)  # (-1)^M
        return ((-1.0) ** m) * phase * np.conj(full[:, :, ::-1])

    return from_mask


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


def _pack_block(
    coeff_fn: Callable[[int], np.ndarray],
    m_values: Sequence[int],
    lmax: int,
    conjugate: bool,
    nfields: int = 3,
) -> np.ndarray:
    """
    Gather the complex full-``M`` coefficients of a block of ``m`` into one
    contiguous complex128 array, ``ncoef = 2 * lmax - 1``.

    ``conjugate=False`` gives ``(nfields, lmax, len(m_values), ncoef)`` -- the
    left operand of the per-``L`` GEMM.  ``conjugate=True`` gives the
    conjugated transpose layout ``(nfields, lmax, ncoef, len(m_values))``, the
    right operand.  Writing the conjugate straight into that layout avoids
    ever holding two copies of a block.  ``nfields`` is 3 (T, E, B) except on
    the T-only path (``spectra=("TT",)``, providers built with
    ``t_only=True``), where it is 1.
    """
    n = len(m_values)
    ncoef = 2 * lmax - 1
    shape = (nfields, lmax, ncoef, n) if conjugate else (nfields, lmax, n, ncoef)
    block = np.empty(shape, dtype=np.complex128)
    for i, m in enumerate(m_values):
        fields = coeff_fn(m)
        for f in range(nfields):
            if conjugate:
                block[f, :, :, i] = np.conj(fields[f])
            else:
                block[f, :, i, :] = fields[f]
    return block


#: ``(max_bytes, floor)`` pairs already warned about by
#: :func:`_coupling_block_sizes`: the precompute asks once per ``(ell, ellp)``
#: pair, always with the same numbers.
_WARNED_MEMORY_FLOORS: set = set()


def _coupling_block_sizes(
    n_m: int,
    n_mp: int,
    lmax: int,
    per_m: int,
    per_pair: int,
    max_bytes: int,
    nspec: int | None = None,
    npairs: int | None = None,
) -> tuple[int, int]:
    """
    Block widths ``(nb, nbp)`` over ``m`` and ``m'`` for the contraction.

    The working set of one ``(m, m')`` block pair is

        ``(nb + nbp) * per_m``    the two coefficient blocks
        ``+ nb * nbp * per_pair`` the Theta slab and its real/imaginary copies

    on top of the ``npairs`` output kernels that are always resident
    (``npairs`` defaults to ``nspec**2``, the full square; a caller that
    restricts the precompute to an explicit list of ordered pairs
    (:func:`precompute_acc_kernels`, ``pairs=``) passes the smaller actual
    count -- what is resident is the requested *pairs*, not ``nspec**2``).
    ``nspec`` defaults to all of :data:`COUPLING_SPECTRA`, i.e. 5; a caller
    that requested a subset via ``spectra`` passes the smaller count. The
    primed block is the inner loop and is rebuilt once per central block, so
    the budget is spent on ``nb`` first and ``nbp`` is held at
    :data:`_PRIME_BLOCK_TARGET` (halved only if even that will not fit, or if
    the halving buys enough central width to cut the number of block passes;
    choosing by passes keeps the blocking monotonic in the budget).  When
    everything fits, both blocks cover the whole range and the contraction is a
    single pass.

    There is a hard floor -- the output kernels plus one ``m`` on each side.
    A budget below it cannot be honoured; the smallest blocking is used anyway
    and a :class:`UserWarning` reports what the run will actually need.
    """
    if nspec is None:
        nspec = len(COUPLING_SPECTRA)
    if npairs is None:
        npairs = nspec**2
    # The output kernels, plus the one-entry +m cache each side's provider
    # may keep (see _coefficient_provider), are resident for the whole
    # contraction.
    reserve = npairs * lmax * lmax * 8 + 2 * per_m
    floor = reserve + 2 * per_m + per_pair
    if max_bytes < floor:
        if (max_bytes, floor) in _WARNED_MEMORY_FLOORS:
            return 1, 1
        _WARNED_MEMORY_FLOORS.add((max_bytes, floor))
        warnings.warn(
            f"coupling memory budget of {max_bytes / 1024**2:.1f} MiB is below "
            f"the {floor / 1024**2:.1f} MiB this (ell, ellp, lmax) needs for "
            "the output kernels and one m on each side; proceeding with "
            "single-m blocks (warned once per process for this budget)",
            UserWarning,
            stacklevel=2,
        )
        return 1, 1
    avail = max_bytes - reserve

    def widest_nb(nbp):
        rest = avail - nbp * per_m
        return min(n_m, int(rest // (per_m + nbp * per_pair))) if rest > 0 else 0

    # Candidate primed widths: the target and its halvings, plus -- only with
    # the whole central range in one block -- its doublings up to n_mp.  Each
    # candidate's passes can only fall as the budget grows, so taking the one
    # with the fewest passes (ties: fewest central blocks, then the widest
    # nbp) is monotonic in the budget; keeping only the first nbp that fits
    # was not.
    nbp = min(n_mp, _PRIME_BLOCK_TARGET)
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
        if (n_m + trial) * per_m + n_m * trial * per_pair > avail:
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
        coefficients (:func:`_pack_block` with ``conjugate=True``).
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

    Returns
    -------
    array
        ``(nspec, nspec, lmax, lmax)`` float64 kernel increment, ``[s1, s2]``
        for ``s2 < s1`` the transpose of ``[s2, s1]``.

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
    memory-budget accounting (:func:`_coupling_block_sizes`) to the channels
    and pairs actually resident, instead of ``nspec`` and ``nspec**2``.

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

    # What is actually resident: with an explicit pair list, only the
    # channels it references (Theta) and the pairs themselves (output
    # kernels); the full square otherwise (unchanged behaviour).
    if output_pairs is not None:
        n_channels_used = len({k for pair in output_pairs for k in pair})
        n_pairs_used = len(output_pairs)
    else:
        n_channels_used = nspec
        n_pairs_used = nspec * nspec

    per_m = nfields * lmax * ncoef * itemsize
    # The float (Re, Im) slab of Theta, (n_channels_used, lmax, 2 nb nbp)
    # float64 -- the same bytes as the complex slab -- plus, transiently, one
    # complex channel of Theta while it is being built
    # (contract_coupling_block).
    per_pair = n_channels_used * lmax * itemsize
    nb, nbp = _coupling_block_sizes(
        n_kept,
        n_kept_p,
        lmax,
        per_m,
        per_pair,
        max_bytes,
        nspec=n_channels_used,
        npairs=n_pairs_used,
    )

    if verbose:
        peak = (nb + nbp + 2) * per_m + nb * nbp * per_pair + n_pairs_used * lmax**2 * 8
        print(
            f"  contraction blocking: m in {nb}-blocks ({-(-n_kept // nb)} of them), "
            f"m' in {nbp}-blocks ({-(-n_kept_p // nbp)} of them); "
            f"estimated peak {peak / 1024**3:.2f} GiB"
        )

    central_fields = [_CENTRAL_FIELD[k] for k in spec_indices]
    prime_fields = [_PRIME_FIELD[k] for k in spec_indices]

    out = np.zeros((nspec, nspec, lmax, lmax))
    skipped = 0
    for i0 in range(0, n_kept, nb):
        rows = m_order[i0 : i0 + nb]
        left = _pack_block(central_fn, rows, lmax, conjugate=False, nfields=nfields)

        for j0 in range(0, n_kept_p, nbp):
            cols = mp_order[j0 : j0 + nbp]
            block_mask = None
            if pair_mask is not None:
                block_mask = pair_mask[np.ix_(rows, cols)]
                if not block_mask.any():
                    skipped += 1
                    continue
                if block_mask.all():
                    block_mask = None
            right = _pack_block(prime_fn, cols, lmax, conjugate=True, nfields=nfields)
            out += contract_coupling_block(
                left,
                right,
                central_fields,
                prime_fields,
                block_mask,
                output_pairs=output_pairs,
            )
            del right
        del left

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


def require_acc_cache_pairs(
    kernel_dir: str,
    centralell: int,
    dmax: int,
    pairs: Iterable[tuple[str, str]],
) -> None:
    """
    Raise the on-disk cache's existing "recompute" error if it does not
    cover every one of ``pairs``, for every diagonal ``(centralell,
    centralell + d)``, ``d < dmax``.

    A thin wrapper around :func:`~cmbcov.approximations.acc_cache.load_coupling_kernels`,
    which already raises ``OSError`` naming the missing pair's file, for the
    plain ACC T/E assembly; this is the same check, run early against the
    pairs a B-mode ``observables`` list needs
    (:func:`~cmbcov.bmode_wick.required_kernel_pairs`),
    so that a run whose spectra turn out to need the 40-pair set is refused
    against an 18-pair cache before anything else, not silently given a
    partial answer. A no-op for an empty ``pairs``.

    The manifest (:func:`~cmbcov.approximations.acc_cache.write_coupling_kernels`)
    records the *channels* a precompute covered, not the exact pairs
    (``pairs=`` can restrict a precompute to fewer pairs than the full
    square of its channels), so an 18-vs-40-pair cache -- both built from
    the same nine :data:`COUPLING_CHANNELS` -- gets the plain "not found"
    form of the error (naming the missing pair's file directly) rather than
    the "computed for spectra ... only" form; both are the one existing
    mechanism, and both name the missing pair.
    """
    pairs = list(pairs)
    if not pairs:
        return
    for diagonal_offset in range(int(dmax)):
        acc_cache.load_coupling_kernels(
            kernel_dir,
            centralell,
            centralell + diagonal_offset,
            COUPLING_CHANNELS,
            pairs,
            pairs=pairs,
            legacy_names=LEGACY_CHANNEL_NAMES,
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


#: Version tag of the ACC normalisation used by a run with a B observable,
#: recorded in every raw-block manifest of such a run
#: (:meth:`ACCStrategy.raw_block_inputs`), so a block saved under another
#: rule -- Eq. 23 per Wick contraction, what a T/E-only run uses -- is never
#: reused for it. Change it whenever the per-term rule changes.
ACC_NORMALISATION_RULE = "per-wick-term-v1"

#: Spectra keys that are parity-odd (``C^TB``, ``C^EB`` in either order).
_PARITY_ODD_CL_KEYS = ("TB", "BT", "EB", "BE")


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
        docs/theory/bmode_kernels.md). A T/E-only run
        (level 1) takes the unchanged path below, bit-identical to before.

        Raises
        ------
        ValueError
            If required ACC parameters are not properly configured, or a
            spectrum this key reads holds fewer than ``lmax_int`` multipoles.
            In a B run also if a Wick term's true spectrum is missing from
            ``cl``, ``centralell < 2``, or the kernels are not GL-grid ones.
        """
        # Validate configuration
        self.validate_config()

        if self._wick_mode(cov_key):
            # A run with a B observable (levels 2-4): every block, T/E ones
            # included, is a sum over expanded Wick terms, each with its own
            # normalisation (docs/theory/bmode_kernels.md).
            # Everything below is the unchanged level-1 path.
            return self._compute_covariance_term_wick(cov_key, cl)

        dmax = self.cov.config.dmax
        centralell = self.cov.config.centralell

        # Initialize flattened covariance
        flat_cov = self._empty_flatten_cov(dmax)

        combination_1, combination_2 = cov_key.key_to_cross()
        # Stokes order for the KERNEL lookup only (covariance_coupling is
        # keyed by COUPLING_CHANNELS, via SpecKey.kernel_stokekey -- e.g.
        # "DT" != "TD" off the diagonal); combination_1/2 above keep their
        # sorted_copy() order for the cl (power spectrum) lookup, where "TE"
        # and "ET" are the same spectrum. See CovKey.key_to_cross_kernel.
        # This CovKey needs at most two kernel pairs (the two Wick
        # contractions), so the loader is asked to load only those -- not
        # the full sixteen-pair default.
        kernel_1, kernel_2 = cov_key.key_to_cross_kernel()
        needed_pairs = {
            (kernel_1[0].kernel_stokekey(), kernel_1[1].kernel_stokekey()),
            (kernel_2[0].kernel_stokekey(), kernel_2[1].kernel_stokekey()),
        }

        # Batched form of the element loop over compute_acc_term (kept below
        # as the scalar reference). For each Wick contraction w:
        #   spectra  cl[combination_w[k].freqkey()][combination_w[k].stokekey()]
        #   kernel   coupling_kernels[kernel_w[0].kernel_stokekey(), kernel_w[1].kernel_stokekey()]
        #   prefactor norm_Xi[combination_w[0].stokekey(), combination_w[1].stokekey()]
        # compute_acc_term depends on (ell1, ell2) only through
        # min(ell1, ell2): both kernel indices are shifted by the same amount
        # and the section is square. So on diagonal Delta the upper element
        # (ell1, ell1 + Delta) and the lower one (ell1 + Delta, ell1) share one
        # ACC value, and only the norm_Xi index order differs between them.
        contractions = []
        names: dict[int, str] = {}
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

        lmax = self.cov.lmax
        auto = cov_key.auto()
        padded: dict[tuple[int, int], np.ndarray] = {}

        for diagonal_offset in range(dmax):
            coupling_kernels = self.get_covariance_coupling(
                centralell, centralell + diagonal_offset, pairs=needed_pairs
            )
            n_ell = lmax - diagonal_offset
            if n_ell <= 0:
                continue
            ell1 = np.arange(n_ell)
            ell2 = ell1 + diagonal_offset

            # Normalise each distinct kernel once per diagonal, not per element.
            kernels = {}
            for _, kernel_key, _ in contractions:
                if kernel_key not in kernels:
                    kernels[kernel_key] = _unit_sum_kernel(coupling_kernels[kernel_key])

            # One band of ACC values per contraction; identical contractions
            # (same spectra arrays, same kernel) are evaluated once.
            bands = {}
            upper = np.zeros(n_ell)
            lower = np.zeros(n_ell)
            for (spec_1, spec_2), kernel_key, norm in contractions:
                matrix = kernels[kernel_key]
                size = matrix.shape[0]
                memo = (id(spec_1), id(spec_2), kernel_key)
                if memo not in bands:
                    n_read = acc_internal_lmax(lmax, size, centralell)
                    for spectrum in (spec_1, spec_2):
                        if (id(spectrum), size) not in padded:
                            _check_spectrum_length(
                                spectrum, names[id(spectrum)], lmax, size, centralell
                            )
                            padded[id(spectrum), size] = _padded_spectrum(
                                spectrum, centralell, n_read, size
                            )
                    chunk = max(1, _ASSEMBLY_CHUNK_BYTES // (24 * size))
                    bands[memo] = _acc_band(
                        matrix,
                        padded[id(spec_1), size],
                        padded[id(spec_2), size],
                        n_ell,
                        chunk,
                    )
                band = bands[memo]
                upper += band * norm[ell1, ell2]
                if diagonal_offset != 0 and not auto:
                    lower += band * norm[ell2, ell1]

            flat_cov[diagonal_offset + dmax - 1, :n_ell] = upper
            if diagonal_offset != 0:
                flat_cov[-diagonal_offset + dmax - 1, :n_ell] = upper if auto else lower

        # Unflatten and return full matrix (consistent with other strategies)
        return self._unflatten_cov(flat_cov)

    def raw_block_inputs(
        self, cov_key: CovKey, cl: dict[str, dict[str, np.ndarray]]
    ) -> tuple[dict[str, np.ndarray], dict]:
        """
        The ``save_raw_blocks`` manifest inputs of an ACC block: the four
        Wick-contraction spectra as :meth:`compute_covariance_term` reads
        them (``[:lmax_int]``), plus ``lmax_int``, ``dmax``, ``centralell`` and the identity
        of the kernel cache -- the resolved ``acc_kernel_dir`` and a digest
        of, for every diagonal, the pair's manifest and the size and
        modification time of the two kernel files this key loads. A
        recomputed or replaced kernel set therefore invalidates the block,
        including a legacy cache without manifests.

        In a run with a B observable (:meth:`_wick_mode`) the spectra are
        every true spectrum the block's expanded Wick terms read, the pairs
        are those terms' kernel pairs, and the identity also records the
        normalisation rule (:data:`ACC_NORMALISATION_RULE`), the pair list
        and the orientation the block was computed in. A block saved by a
        T/E-only run (Eq. 23 per contraction) therefore never matches a
        B-run manifest, and a T/E-only run's manifest is unchanged.
        """
        config = self.cov.config
        kernel_dir = os.path.abspath(self.cov.acc_kernel_dir)
        lmax = self.cov.lmax
        if self._wick_mode(cov_key):
            terms, transposed = self._wick_orientation(cov_key, cl)
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
                    "acc_block_orientation": (
                        "transposed" if transposed else "natural"
                    ),
                }
            )
            return spectra, identity

        kernel_1, kernel_2 = cov_key.key_to_cross_kernel()
        pairs = sorted(
            {
                (kernel_1[0].kernel_stokekey(), kernel_1[1].kernel_stokekey()),
                (kernel_2[0].kernel_stokekey(), kernel_2[1].kernel_stokekey()),
            }
        )
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
        """Whether every pair's kernel file exists (new or legacy name) for
        the first diagonal ``(centralell, centralell)``."""
        ell = self.cov.config.centralell
        kernel_dir = self.cov.acc_kernel_dir
        for pair in pairs:
            path = acc_cache.coupling_save_path(
                kernel_dir, pair, ell, ell, COUPLING_CHANNELS
            )
            if os.path.exists(path):
                continue
            legacy = tuple(LEGACY_CHANNEL_NAMES.get(s, s) for s in pair)
            legacy_path = os.path.join(
                kernel_dir,
                f"covariance_coupling/{legacy[0]}x{legacy[1]}_{ell}x{ell}.npy",
            )
            if legacy == tuple(pair) or not os.path.exists(legacy_path):
                return False
        return True

    def _wick_orientation(self, cov_key: CovKey, cl) -> tuple[list, bool]:
        """
        The expanded Wick terms this block is computed from, and whether they
        are those of the transposed block ``Cov(b, a)``.

        The precompute stores, for every off-diagonal block, the kernel pairs
        of ONE orientation, the one minimising the pair count
        (:func:`~cmbcov.bmode_wick.required_kernel_pairs`,
        docs/theory/bmode_kernels.md): e.g. Cov(BB, TE)
        rather than Cov(TE, BB). The natural orientation is used when its
        kernels are all on disk, else the transposed one when its are, else
        the natural one (whose load then raises the "recompute" error naming
        the missing pair). ``Cov(a, b)_{l l'} = Cov(b, a)_{l' l}`` exactly;
        in ACC the two orientations differ only in which of the two
        elements ``(l*, l*+Delta)`` / ``(l*+Delta, l*)`` is exact.
        """
        from ..bmode_wick import covkey_wick_terms

        parity_odd = self._run_flags(cov_key, cl)["parity_odd"]
        natural = covkey_wick_terms(cov_key.stoke, cov_key.freq, parity_odd)
        if cov_key.left == cov_key.right:
            return natural, False
        natural_pairs = {(t.channel_1, t.channel_2) for t in natural}
        if self._pairs_on_disk(natural_pairs):
            return natural, False
        swapped = cov_key.transpose()
        transposed = covkey_wick_terms(swapped.stoke, swapped.freq, parity_odd)
        if self._pairs_on_disk({(t.channel_1, t.channel_2) for t in transposed}):
            return transposed, True
        return natural, False

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
        The lower triangle is filled as in the level-1 path (same band,
        :math:`N_t` read at ``(l + Delta, l)``). A parity-mixed block
        (one parity-odd observable) is zero unless the run asks for
        ``parity_mixed_blocks``.
        """
        from ..keys import _is_parity_odd_spec

        config = self.cov.config
        dmax = config.dmax
        centralell = config.centralell
        lmax = self.cov.lmax
        flags = self._run_flags(cov_key, cl)
        flat_cov = self._empty_flatten_cov(dmax)

        if (
            _is_parity_odd_spec(cov_key.left) != _is_parity_odd_spec(cov_key.right)
            and not flags["parity_mixed_blocks"]
        ):
            return self._unflatten_cov(flat_cov)

        terms, transposed = self._wick_orientation(cov_key, cl)
        needed_pairs = sorted({(t.channel_1, t.channel_2) for t in terms})
        spectra = {}
        for term in terms:
            for side in (term.left, term.right):
                if side not in spectra:
                    spectra[side] = self._wick_spectrum(cl, side)
        auto = cov_key.auto()
        padded: dict[tuple[int, int], np.ndarray] = {}

        for diagonal_offset in range(dmax):
            n_ell = lmax - diagonal_offset
            if n_ell <= 0:
                continue
            ell_prime = centralell + diagonal_offset
            coupling_kernels = self.get_covariance_coupling(
                centralell, ell_prime, pairs=needed_pairs
            )
            self._require_gl_kernels(centralell, ell_prime)
            ell1 = np.arange(n_ell)
            ell2 = ell1 + diagonal_offset

            bands = {}
            upper = np.zeros(n_ell)
            lower = np.zeros(n_ell)
            for term in terms:
                pair = (term.channel_1, term.channel_2)
                kernel = coupling_kernels[pair]
                size = kernel.shape[0]
                (name_1, spec_1), (name_2, spec_2) = (
                    spectra[term.left],
                    spectra[term.right],
                )
                memo = (id(spec_1), id(spec_2), pair)
                if memo not in bands:
                    n_read = acc_internal_lmax(lmax, size, centralell)
                    for name, spectrum in ((name_1, spec_1), (name_2, spec_2)):
                        if (id(spectrum), size) not in padded:
                            _check_spectrum_length(
                                spectrum, f"cl[{name}]", lmax, size, centralell
                            )
                            padded[id(spectrum), size] = _padded_spectrum(
                                spectrum, centralell, n_read, size
                            )
                    chunk = max(1, _ASSEMBLY_CHUNK_BYTES // (24 * size))
                    bands[memo] = _acc_band(
                        np.asarray(kernel, dtype=float),
                        padded[id(spec_1), size],
                        padded[id(spec_2), size],
                        n_ell,
                        chunk,
                    )
                band = bands[memo]
                kernel_sum = float(np.sum(kernel))
                scale = self.cov.acc_term_scale(
                    term.channel_1,
                    term.channel_2,
                    diagonal_offset,
                    kernel_sum,
                    ell1,
                    ell2,
                )
                upper += term.coefficient * band * scale
                if diagonal_offset != 0 and not auto:
                    scale_lower = self.cov.acc_term_scale(
                        term.channel_1,
                        term.channel_2,
                        diagonal_offset,
                        kernel_sum,
                        ell2,
                        ell1,
                    )
                    lower += term.coefficient * band * scale_lower

            flat_cov[diagonal_offset + dmax - 1, :n_ell] = upper
            if diagonal_offset != 0:
                flat_cov[-diagonal_offset + dmax - 1, :n_ell] = upper if auto else lower

        block = self._unflatten_cov(flat_cov)
        return np.ascontiguousarray(block.T) if transposed else block

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

    def error_budget(self, covariance_keys, band_edges=None) -> dict | None:
        """
        The polarised error budget of this run, as a dict, or ``None`` when no
        block has a polarised leg.

        Two terms, from
        :mod:`~cmbcov.approximations.acc_budget`: the E->B
        leakage of the kernel set in use, ``lambda = sum Theta^{TT x BB} /
        sum Theta^{TT x EE}``, giving ``(n_E / 2) lambda`` per block (the
        ``BB`` kernels are already in the cache and read by nothing else, so
        this computes nothing new); and the Eq. 33 translation error, which is
        *not* bounded -- the budget reports ``|l - l*|`` at the band edges and
        says so. See that module for what was tried and measured.

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

        ``None`` for a run with a B observable: its blocks use the
        per-Wick-term normalisation (docs/theory/bmode_kernels.md), which
        is exact at ``l*`` and carries no Eq. 23 leakage bias of the
        ``(n_E / 2) lambda`` form this report quotes; its measured accuracy
        off ``l*`` is docs/theory/bmode_kernels.md, "Known limits".
        """
        from ..covariance import has_b_observable
        from . import acc_budget

        if has_b_observable(covariance_keys):
            return None
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
        if verbose:
            print(
                "Central I_lm: "
                + ("held in RAM" if integral_mask is not None else "streamed per block")
            )

        # Process each ellp value
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
                "reflect": False,
            }
        if selection is None:
            return {
                "synthesise": lambda ell, m: _gl_integrals_m(
                    mask_alm, lw, ell, m, lmax, t_only=t_only
                ),
                "materialise": lambda ell: _gl_integrals(
                    mask_alm, lw, ell, lmax, t_only=t_only
                ),
                "nbytes": lambda ell: (2 * ell + 1)
                * nfields
                * lmax
                * (2 * lmax - 1)
                * 16,
                "to_full_m": lambda item: _gl_full_m(item, t_only),
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
        return value is ``None`` and each block of ``m`` is synthesised on
        demand inside :meth:`_compute_ellp_coupling`.

        ``t_only=True`` means the caller has determined via
        :func:`_resolve_spectra` that every kernel it asked for needs only the
        T field.  On the GL branch it skips the spin-2 (E, B) synthesis
        entirely (~92% of the precompute wall time at ``centralell=250``,
        ``nside=256``).  The
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

        ``integral_mask`` may be ``None``, meaning "synthesise each ``m`` on
        demand" -- the mode :meth:`_compute_central_integrals` selects when
        the whole set would blow the memory budget.  A primed set
        (``ell != ellp``) is materialised only when a later ``ellp`` in
        ``remaining_ellp_values`` reuses it (then kept in
        ``integral_mask_cache``); otherwise it is streamed block by block.  At
        ``ell == ellp`` the central set serves both sides.

        ``max_memory_bytes`` caps the blocked working set; ``None`` uses
        :data:`DEFAULT_COUPLING_MEMORY_GB`.

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
        if ell != ellp:
            integral_mask_prime = integral_mask_cache.get(ellp)
            if integral_mask_prime is not None:
                if verbose:
                    print(f"  Reusing cached I_lm for ellp={ellp}")
            elif ellp in remaining_ellp_values:
                # Only materialise the whole primed set when a later ellp
                # will reuse it; otherwise it is streamed block by block.
                integral_mask_prime = source["materialise"](ellp)
                integral_mask_cache[ellp] = integral_mask_prime
                if verbose:
                    print(f"  Cached I_lm for ellp={ellp} (appears again later)")
        else:
            if verbose:
                print("  Using ell==ellp optimization")
            integral_mask_prime = integral_mask

        def provider(held, l_val):
            return _coefficient_provider(
                held,
                lambda m: source["synthesise"](l_val, m),
                source["to_full_m"],
                l_val,
                reflect=source["reflect"],
            )

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
            provider(integral_mask, ell),
            2 * ell + 1,
            provider(integral_mask_prime, ellp),
            2 * ellp + 1,
            lmax,
            max_memory_bytes,
            spec_indices=active,
            nfields=1 if t_only else 3,
            verbose=verbose,
            keep_m=None if sel is None else sel.keep_m,
            keep_mp=None if sel is None else sel.keep_mp,
            pair_mask=None if sel is None else sel.keep_pair,
            output_pairs=output_pairs,
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
        integrals entirely (~92% of the wall time at ``centralell=250``,
        ``nside=256``); on ``grid="healpix"`` the synthesis is joint across
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
        Peak-memory budget, in GiB (default :data:`DEFAULT_COUPLING_MEMORY_GB`),
        of the blocked contraction (:func:`_accumulate_coupling_kernels`).  It
        sets the ``(m, m')`` block widths and, on both grids, whether the
        central integral set is held in RAM (when it fits in half the budget)
        or streamed.  At ``ell = ellp =
        250``, ``nside = 256``, ``lw = 512`` one ``m`` of integrals is 24 MiB
        (11.7 GiB for a whole set), so 2 GiB gives 11 central blocks, 12 GiB
        gives 2 and 24 GiB gives 1.  It bounds the contraction working set,
        not process RSS (measured 2.85 GiB RSS for a 2 GiB budget).
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
    )
