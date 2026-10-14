"""Precompute of the ACC coupling kernels (Camphuis et al. 2022, Eq. 33).

:func:`precompute_acc_kernels` computes the reduced coupling kernel once at the
central multipole ``l*`` for a mask; the strategy reuses it along each diagonal.
"""

import os
import shutil
import tempfile
import warnings
from collections.abc import Sequence

import numpy as np

from ...harmonics.grid import full_from_pair
from ...harmonics.sht import DEFAULT_MAP2ALM_ITER, ducc0_map2alm
from ...mask.footprint import MaskWlm
from ...utils.healpy_utils import get_nside_from_ell
from . import cache as acc_cache
from .channels import (
    _CENTRAL_FIELD,
    _LEAKAGE_CHANNELS,
    _PRIME_FIELD,
    COUPLING_CHANNELS,
    COUPLING_SPECTRA,
    _resolve_spectra,
    normalise_channel,
)
from .contraction import DEFAULT_COUPLING_MEMORY_GB, _accumulate_coupling_kernels
from .producers import (
    _GL_PRODUCER,
    _coefficient_provider,
    _gl_full_m,
    _gl_integrals,
    _gl_integrals_m,
    _gl_producer_bytes,
    _GLMaskModes,
    _healpix_full_m,
    _healpix_integrals,
    _healpix_producer_bytes,
    _pack_block,
    _paired_m_order,
    _TermSelectionPlan,
)
from .store import _CentralStore
from .term_selection import pole_rotation, rotate_alm

__all__ = [
    "coupling_ellprange",
    "precompute_acc_kernels",
]


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

        ``term_selection=None`` (default) computes the full contraction.
        With a tolerance (``grid="gl"`` only) the mask alm is first rotated
        to the pole frame -- the kernels are rotation invariant, but the
        selection rules are only sharp there -- and every ``(ell, ellp)``
        pair is computed from the terms
        :func:`~cmbcov.approximations.acc.term_selection.select_terms`
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
                    "are not azimuthally symmetric about their centre; the error "
                    "can exceed the tolerance by an order of magnitude. Use "
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
        :func:`~cmbcov.harmonics.grid.banded_integrals_gl` at the
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
        entirely (two Legendre components against one for T).  The HEALPix
        branch computes T, E and B in one joint ``map2alm`` per ``m`` and has
        no cheaper T-only synthesis; there ``t_only`` only restricts the
        full-``M`` expansion (and the budget of the contraction) to T.
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
        the caller then streams the central set per block.  On any
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
        means all five, i.e. all 25 ordered pairs).

        ``pairs`` (channel-name pairs, normalised to canonical names -- see
        :func:`normalise_channel`) restricts the output further, to an
        explicit list of ordered pairs instead of the full square of
        ``spec_indices``: see :func:`_accumulate_coupling_kernels`,
        ``output_pairs``.

        Both grids go through one code path: a :func:`_coefficient_provider`
        per side hands complex full-``M`` coefficients to
        :func:`_accumulate_coupling_kernels`, which uses one GEMM per ``L``
        and accumulates straight into the kernels, so the ``(nspec, 2l+1,
        2l'+1, lmax)`` ``Theta`` table is never built.  Only the source of
        the integrals differs (:meth:`_integral_source`).

        ``Theta`` is complex on both grids.  For a map split into the
        ``(r, s)`` healpy alm pair of its real and imaginary parts (the
        HEALPix integrals),

            ``Re(sum_M x_LM conj(y_LM)) = sum_M r_x conj(r_y) + s_x conj(s_y)``
            ``= ls[L] * (alm2cl(x[0], y[0]) + alm2cl(x[1], y[1]))[L]`` ,

        which the HEALPix branch builds; the imaginary
        part ``Im(sum_M x conj(y))`` is also kept, via
        :func:`_healpix_full_m`.  It vanishes for a mask symmetric under
        ``phi -> -phi`` but is comparable to the real part for a generic
        mask, and Eq. 20 needs it (it is ``|sum_L C_L Theta(L)|^2``, not
        ``(sum_L C_L Re Theta(L))^2``).

        ``integral_mask`` is the central set held in RAM (as
        :meth:`_compute_central_integrals` returns it), a
        :class:`_CentralStore` (the set on disk, built once per precompute
        by :meth:`compute` when it does not fit the budget), or ``None``,
        meaning "synthesise each ``m`` on demand" (the streamed path, used
        when no store can be written).  Held or stored, the central
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

            Reference form of the reduction, not on the production path:
            :func:`_accumulate_coupling_kernels` performs it block by block,
            so the full ``theta_before_summing`` is never materialised.
            ``tests/approximations/acc/test_acc_vectorised.py`` checks the blocked path against
            it.
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
        computed (see :func:`~cmbcov.approximations.acc.cache.read_coupling_manifest`)
        -- including a digest of the mask (see
        :attr:`~cmbcov.mask.MaskWlm.mask_digest`) and, for
        ``grid="healpix"``, the ``map2alm`` iteration count and
        :data:`~cmbcov.approximations.acc.cache.HEALPIX_KERNEL_VERSION`,
        the same fields
        :meth:`~cmbcov.assembly.covariance.Cov.get_acc_coupling_kernels`
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
    :meth:`~cmbcov.assembly.covariance.Cov.get_acc_coupling_kernels`):
    the pairs ``(centralell, centralell + d)`` for ``d`` in ``range(dmax)``,
    see :func:`coupling_ellprange`.  Files go to
    ``<save_dir>/covariance_coupling/`` through
    :func:`~cmbcov.approximations.acc.cache.write_coupling_kernels`,
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
        :attr:`~cmbcov.config.validation.PipelineConfig.acc_kernel_dir`.
        May be ``None`` only with ``dryrun=True``.
    centralell : int
        The reference multipole :math:`\ell_*` of Eq. 33; must match the run's
        ``CovarianceConfig.centralell`` (it is recorded and checked).
    dmax : int, optional
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
        :math:`I_{\ell m, LM}`.  ``"gl"`` computes the integrals exactly (to
        ~1e-15) on a Gauss-Legendre grid for the mask band-limited at ``lw``,
        for the spin-0 (T) and spin-2 (E, B) channels alike.  ``"healpix"``
        degrades the mask to ``nside`` and uses one ``map2alm``
        (``iter=DEFAULT_MAP2ALM_ITER``) per integral instead.  Both keep the
        per-``(m, m')`` cross-spectra ``Theta^s(m, m', L)`` complex
        (``output = Re sum_{m m'} Theta^{s1} conj(Theta^{s2})``, Eq. 20)
        through one contraction, which avoids biasing the kernels for a mask
        not symmetric under ``phi -> -phi``.  They differ in one convention:
        the HEALPix E/B integrals carry a factor ``sqrt(2)`` (from
        :func:`~cmbcov.harmonics.sht.cplx_spin_weighted_ylm`), so raw HEALPix kernels
        with ``k`` E/B legs are ``2^(k/2)`` larger; it cancels in the Eq. 23
        normalisation.
    lw : int, optional
        ``grid="gl"`` only: band-limit of the mask alm used to build the
        integrals.  Defaults to ``3 * nside - 1``.
    spectra : sequence of str, optional
        Subset of :data:`COUPLING_CHANNELS` to compute kernels for (the
        T/E/B-letter names ``TT, EE, BB, TE, ET`` are accepted and normalised,
        :data:`CHANNEL_ALIASES`); the output is ``spectra x spectra`` unless
        ``pairs`` restricts it further.  ``None`` (default) computes
        :data:`COUPLING_SPECTRA`, the five channels ``TT, DD, LL, TD, DT``
        (25 ordered pairs); the four leakage channels (``TL, LT, DL, LD``)
        are computed only when requested explicitly.  ``("TT",)`` on
        ``grid="gl"`` skips the spin-2 integrals entirely (two Legendre
        components against one for T); on ``grid="healpix"`` the synthesis is
        joint across T/D/L, and only the full-``M`` expansion, the
        contraction and the files written shrink.  A run that needs a kernel
        outside the subset fails on load, naming it.
    pairs : sequence of (str, str), optional
        An explicit list of ordered channel pairs (canonical or letter
        names) to compute, instead of the full square of ``spectra`` -- e.g.
        ``[("TT", "TT"), ("DD", "LL")]``.  Restricts both the ``Theta``
        built (only the channels referenced) and the kernels accumulated
        and written (only the pairs themselves).  ``None`` (default) is the
        full square.  If
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
        its libraries; the mask map and its alm as loaded (8 bytes per mask
        pixel; reading and transforming a large mask peaks at a few GiB
        before the contraction starts); memory the system allocator retains
        after numpy frees it (platform-dependent: some allocators cache large
        freed blocks, while glibc unmaps allocations above 32 MiB when they
        are freed); and, with ``dryrun=True``, the kernels returned,
        ``nspec**2 * (2 nside)**2 * 8`` bytes per ``ellp``.  One ``m`` of
        integrals is ``nfields * 2 nside * (4 nside - 1) * 16`` bytes
        (``nfields`` = 3, or 1 for a T-only precompute).  A
        budget below the floor (the output kernels, the producer and one
        ``m`` on each side) is warned about and the smallest blocks are used.
    scratch_dir : str, optional
        Where the temporary central-coefficient store goes when the central
        set does not fit in half of ``max_memory_gb``: a fresh
        ``acc_central_store_*`` subdirectory of it, holding one file of
        ``(2 centralell + 1) * nfields * 2 nside * (4 nside - 1) * 16`` bytes,
        deleted when the precompute returns or raises.  ``None`` (default)
        uses ``save_dir`` (or the system temporary directory with
        ``dryrun=True``).  If the file system lacks the space, a
        :class:`UserWarning` is issued and the set is re-synthesised per
        block instead.
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
        ``grid="gl"`` only.  ``None`` (default): the full contraction.  A
        positive float is the target relative Frobenius error of each kernel
        and is passed to :func:`~cmbcov.approximations.acc.term_selection.select_terms`: the
        mask alm is rotated to the pole frame (the kernels are rotation
        invariant, while the selection is only sharp there), and per
        ``(ell, ellp)`` only the
        orders ``m`` with enough mask overlap, the ``(m, m')`` pairs the
        mask's azimuthal autocorrelation couples, and the band ``|M - m|
        <= m_band`` are computed (the integrals through
        :func:`~cmbcov.harmonics.grid.banded_integrals_gl`).  For
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
