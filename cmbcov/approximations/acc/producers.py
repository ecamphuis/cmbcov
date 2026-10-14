"""Coefficient producers of the ACC precompute.

The mask integrals ``u^a_{l m, LM}`` the kernels are contracted from, produced
on a Gauss-Legendre grid (banded or full-``m``) or on a HEALPix map, the
term selection that restricts them to the ``(m, m', M)`` terms that matter,
and the provider that hands them to the contraction block by block.
"""

import math
from collections.abc import Callable, Iterable, Sequence

import numpy as np

from ...harmonics.grid import (
    banded_integrals_gl,
    full_from_pair,
    gl_mask_modes,
    spin_weighted_integrals_gl,
)
from ...mask.footprint import MaskWlm
from ...utils.threading_utils import (
    get_optimal_nthreads,
    row_block_workers,
    run_row_blocks,
)
from .term_selection import TermSelection, select_terms


def _gl_banded_grid(lw: int, ell: int, lmax_out: int) -> int:
    """
    The GL grid band-limit :func:`~cmbcov.harmonics.grid.banded_integrals_gl`
    (and :func:`~cmbcov.harmonics.grid.spin_weighted_integrals_gl`)
    choose by default for ``(lw, ell, lmax_out)``: ``max(ceil((lw + ell +
    lmax_out - 1) / 2), ell, lmax_out)``.  Restated here so the mask modes
    can be computed once per ``ell`` on exactly that grid and handed in.
    """
    return max(math.ceil((lw + ell + lmax_out - 1) / 2), ell, lmax_out)


#: Producer of the GL coefficients on the default path (no term selection).
#: ``"legendre"``: :func:`~cmbcov.harmonics.grid.banded_integrals_gl` at full band
#: (``m_band = lmax_out + |m|``), the azimuthal sum done analytically from
#: the mask's ring Fourier modes (:func:`~cmbcov.harmonics.grid.gl_mask_modes`,
#: computed once per grid and shared by every ``m``) and one
#: ``ducc0.sht.leg2alm`` per sign of ``M`` and spin.  ``"two_sht"``: the
#: reference route :func:`~cmbcov.harmonics.grid.spin_weighted_integrals_gl`
#: (synthesise the mask and ``Y_lm``, multiply, analyse).  Both evaluate the
#: same product with the same Gauss-Legendre rule on the same grid and agree
#: to rounding (``tests/approximations/acc/test_acc_gl_producer.py``); the switch exists for
#: those tests and is not a user option.
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

    The integrals come from :func:`~cmbcov.harmonics.grid.banded_integrals_gl` on the
    grid :func:`_gl_banded_grid`, with the mask modes ``mask_modes``
    (:func:`~cmbcov.harmonics.grid.gl_mask_modes` on that grid, computed here when
    ``None``; callers share one per grid across ``m``).  ``m_band=None``
    (the default path) is the full band ``lmax_out + |m|``, i.e. every
    ``M``: the same numbers as the two-SHT
    :func:`~cmbcov.harmonics.grid.spin_weighted_integrals_gl` to rounding, which is
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
    one for T, so a caller that only needs T-field kernels
    (``spectra=("TT",)``) skips it entirely.

    Default path (no ``keep_m``, no ``m_band``): the mask modes are computed
    once for the grid of ``ell`` (unless given) and shared by every ``m``,
    and only ``m >= 0`` is transformed -- the ``m < 0`` half is the
    reflection :func:`_reflected_item`, exactly as a reflecting
    :func:`_coefficient_provider` serves it, so a set held in RAM and one
    synthesised on demand hold the same bits.

    ``keep_m`` (term selection, index ``m + ell``) leaves ``None`` at the
    orders that are not kept; the contraction never asks for them.
    ``m_band`` / ``mask_modes`` are forwarded to :func:`_gl_integrals_m`;
    with a selection every kept ``m`` is transformed directly.
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
    :func:`~cmbcov.harmonics.grid.gl_mask_modes` of one mask alm, per GL grid, for the
    default GL producer (:func:`_gl_integrals_m`).

    ``modes(ell)`` returns the modes on the grid :func:`_gl_banded_grid`
    ``(lw, ell, lmax_out)``, computing them on first use.  Grids depend on
    ``ell`` only through ``ceil((lw + ell + lmax_out - 1) / 2)``, so the
    central and a neighbouring primed multipole often share one; the cache
    keeps the ``max_grids`` (default :data:`_GL_MODE_GRIDS`) most recently
    used grids -- the central one and the current primed one on the
    streamed path -- rather than one per ``ellp`` of a long ``ellprange``.
    One grid is a complex ``(Lg + 1, 2 Lg + 2)`` array (``16 (Lg + 1)
    (2 Lg + 2)`` bytes); the memory budget charges :data:`_GL_MODE_GRIDS` of
    the largest (:func:`_gl_producer_bytes`).

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
    The union of several :class:`~cmbcov.approximations.acc.term_selection.TermSelection`
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
    :mod:`cmbcov.approximations.acc.term_selection`: the kernels are
    rotation invariant, the selection is only sharp in that frame), the
    tolerance, and caches of everything that is shared across ``ellp``:

    * ``selection(ell, ellp)``: the union over ``spins`` of
      :func:`~cmbcov.approximations.acc.term_selection.select_terms`
      (spin 0 alone for a T-only precompute, spin 0 and 2 otherwise).
    * ``keep_m(l)``: the kept orders of degree ``l`` -- ``select_terms``
      derives them from :func:`~cmbcov.approximations.acc.term_selection.mode_power`
      at ``l`` alone, so they do not depend on the partner multipole and the
      central integral set can be built once and reused for every ``ellp``.
    * ``m_band``: the ``|M - m|`` band, which depends only on the mask's
      azimuthal spectrum and is therefore one number for the whole run.
    * ``mask_modes(l)``: :func:`~cmbcov.harmonics.grid.gl_mask_modes`
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
    complex coefficient stack of fields ``(T, D, L)``, or ``(T,)`` with
    ``t_only``."""
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

    Field by field this is :func:`~cmbcov.harmonics.grid.full_from_pair`
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
    :func:`~cmbcov.harmonics.sht.cplx_spin_weighted_ylm`.
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
    ``tests/approximations/acc/test_acc_vectorised.py``, and to rounding on the default
    Legendre route, ``tests/approximations/acc/test_acc_gl_producer.py``), so the ``m < 0`` half
    is an index reflection of the ``m > 0`` half (:func:`_reflect_into`).
    Paired with the ``+m, -m`` iteration order of :func:`_paired_m_order`, a
    one-entry cache turns that into half the transforms; a held GL set
    (:func:`_gl_integrals`) is built with the same reflection, so held and
    synthesised coefficients are the same bits.  The identity is not assumed
    on the HEALPix grid (pixel quadrature, iterated ``map2alm``), so the
    HEALPix branch passes ``reflect=False`` and synthesises every ``m``.
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
    Either way every ``m`` is written as ``nfields * lmax`` contiguous rows
    (filling the transposed layout directly would scatter each ``m`` over the
    whole block).  ``np.matmul`` passes the transposed operand to BLAS as
    such, without a copy.

    A :class:`_CoefficientProvider` writes each ``m`` straight into its slot
    of the block (:meth:`_CoefficientProvider.into`); a plain callable is
    copied from.  ``nfields`` is 3 (T, D, L) except on
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


#: Transient bytes per ``nside`` pixel of one HEALPix integral set
#: (:meth:`~cmbcov.mask.MaskWlm.compute_spin_weighted_integrals`: the
#: spin-weighted ``Y_lm`` maps, their product with the mask, the real and
#: imaginary parts and the ``map2alm`` residuals).  ``tracemalloc`` gives
#: 186 B per pixel; rounded up for the ``ducc0`` scratch it does not see.
_HEALPIX_PRODUCER_BYTES_PER_PIXEL = 256


def _gl_producer_bytes(lw: int, lmax: int, ells: Iterable[int], nfields: int) -> int:
    """
    What the GL coefficient producer holds while it builds a block, beyond
    the block itself (:func:`_coupling_working_set`, ``producer``):

    * three per-``m`` coefficient sets, ``per_m`` each -- the order being
      synthesised (:func:`_gl_integrals_m`), the one-entry ``+|m|`` cache of
      the provider building the block, and the cache of the other side's
      provider on the streamed path (:func:`_coefficient_provider`);
    * the legs of one :func:`~cmbcov.harmonics.grid.banded_integrals_gl` call,
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
