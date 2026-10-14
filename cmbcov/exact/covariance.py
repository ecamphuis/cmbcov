"""The public entry points of the exact covariance: rows and full matrices, TT and polarised."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from ..harmonics.grid import gl_minimal_lmax, gl_synthesis
from ..harmonics.sht import alm2cl
from .common import _cl_matrix, _spectra_fields, _warn_short_margin
from .gl import _BatchedGL, _exact_row_gl, _mask_alm_for_gl, _pole_frame_alm
from .healpix import _correlation_column, _mask_wl_healpix
from .parallel import _rows_gl_parallel
from .pol import _exact_row_pol_gl, _polarised_batch

__all__ = [
    "exact_covariance_row",
    "exact_covariance",
    "exact_covariance_row_pol",
    "exact_covariance_pol",
]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def exact_covariance_row(
    mask: np.ndarray,
    cl: np.ndarray,
    ellp: int,
    lmax: int,
    nside: int | None = None,
    iter: int = 3,
    use_symmetry: bool = True,
    exact_adjoint: bool = True,
    grid: str = "healpix",
    lw: int | None = None,
    lmax_grid: int | None = None,
    nthreads: int | None = None,
    lmax_int: int | None = None,
    term_selection: float | None = None,
    max_memory_gb: float | None = None,
) -> np.ndarray:
    r"""
    One row/column ``Sigma[:, ellp]`` of the exact TT pseudo-``C_l`` covariance.

    Implements paper Eq. (7) with the correlations of Eqs. (8)-(11) obtained
    by spherical harmonic transforms as in Fig. 2 of Camphuis et al. (2022).

    Parameters
    ----------
    mask : ndarray
        ``grid="healpix"``: real HEALPix mask (RING ordering) at resolution
        ``nside``.  ``grid="gl"``: either such a map (converted once with
        ``hp.map2alm(mask, lmax=lw, iter=10)``) or a complex healpy alm array
        of the mask already truncated at ``lw`` (used as is).
    cl : ndarray
        Real temperature power spectrum, at least ``lmax_int+1`` long (i.e.
        ``lmax+1`` when ``lmax_int`` is not given).
    ellp : int
        Multipole :math:`\ell'` of the row, in ``[0, lmax]``.
    lmax : int
        Maximum multipole *reported*.  Use ``nside >= lmax_int/2``.
    nside : int, optional
        HEALPix resolution of ``mask``.  Required for ``grid="healpix"``; for
        ``grid="gl"`` it is inferred from a map's size and unused for an alm.
    iter : int
        Number of map2alm iterations of the estimator (healpy default 3).
        HEALPix grid only.
    use_symmetry : bool
        If True, only ``m' >= 0`` is computed and the ``m' > 0`` terms are
        counted twice.  For real ``W`` and real ``C_L`` one has
        ``c_{-m'}[l, m] = (-1)^{m+m'} conj c_{m'}[l, -m]``, so the sum over
        ``m`` of ``|c|^2`` is identical for ``m'`` and ``-m'``.
    exact_adjoint : bool
        If True (default), step 1 uses the exact adjoint of the iterated
        analysis so that the result is :math:`K C K^\dagger`.  If False, the
        naive :math:`K C K` (analyse ``W Y_{l'm'}`` directly) is computed.
        HEALPix grid only; on GL analysis is the adjoint of synthesis.
    grid : {"healpix", "gl"}
        Which estimator's covariance to compute (docstring of :mod:`cmbcov.exact`).
        ``"healpix"``: the covariance of the pixelised ``map2alm(iter)``
        estimator, i.e. of ``hp.anafast`` applied to a HEALPix map times the
        pixel mask.  ``"gl"``: the covariance of the exact-integral estimator
        for the band-limited mask, computed on a Gauss-Legendre grid to
        machine precision.  They differ by an amount that shrinks as
        ``nside^-2``.
    lw : int, optional
        GL grid only: band-limit of the mask.  Defaults to ``3*nside - 1``
        for a map input and to the array's own band-limit for an alm input.
    lmax_grid : int, optional
        GL grid only: GL grid band-limit.  ``None`` selects the minimal exact
        grid :func:`~.grid.gl_minimal_lmax` ``(lmax_int, lw)`` -- sized from
        the *internal* band-limit, since that is what the transforms carry.
    nthreads : int, optional
        GL grid only: threads for ducc0.  The batched rows are bit-identical
        for any value.
    lmax_int : int, optional
        Internal band-limit.  ``C_L`` is summed and every harmonic array is
        carried to ``lmax_int`` while only ``l = 0..lmax`` is returned, so the
        row is identical to the one computed with ``lmax=lmax_int`` and sliced.
        ``None`` (default) means ``lmax_int = lmax``.
        Because the mask couples ``l'`` to ``L`` over its own width, a row at
        ``l'`` needs ``C_L`` above ``l'``; a margin ``lmax_int - lmax`` below
        :func:`mask_coupling_width` raises a ``UserWarning`` naming both
        numbers (docstring of :mod:`cmbcov.exact`, "Reported band-limit versus
        internal band-limit").
    term_selection : float, optional
        GL grid only.  ``None`` (default): every ``m'`` is computed.  A
        positive tolerance rotates the mask alm to the pole frame once (the
        row is rotation invariant, see :func:`_pole_frame_alm`) and skips
        the orders ``m'`` whose mask overlap is below ``tolerance / 10`` of
        the largest (:func:`_kept_mprime`, the ``eps_m`` rule of
        :func:`~cmbcov.approximations.acc.term_selection.select_terms`).  Raises
        ``ValueError`` with ``grid="healpix"``.
    max_memory_gb : float, optional
        GL grid only: memory budget in GiB (default
        :data:`DEFAULT_EXACT_MEMORY_GB`, 2) for everything the row holds that
        scales with the problem.  The columns ``m'`` of a GL row are computed
        in chunks, their spherical-harmonic transforms done together as
        matrix products (docstring of :mod:`cmbcov.exact.gl`, "Structure of the GL
        column computation"); the budget covers the chunk's ring profiles (the
        dominant term, ``16 N ntheta`` bytes per column with ``ntheta`` the GL
        rings and
        ``N <= 2 lmax_int + lw + 1`` the chunk's azimuthal FFT length), the
        mask sampled on the rings, per-order scratch, the mask's ring modes
        and the Legendre table -- held whole if it fits in half the budget,
        streamed in blocks otherwise.  It sets how many columns share a
        chunk, not the result (which agrees to a few 1e-15 of the largest
        entry whatever the chunking).  Not covered: the mask as loaded, its
        alm, the returned row, ducc0's and the BLAS library's internal
        buffers and freed memory the system allocator keeps, so the process's
        resident size can exceed the budget (:func:`_batched_row_plan`).
        Raises ``ValueError`` with ``grid="healpix"``.

    Returns
    -------
    ndarray
        ``Sigma[l, ellp]`` for ``l = 0..lmax``.

    Notes
    -----
    Cost is :math:`O(\ell'^4)`: :math:`\ell'+1` (or :math:`2\ell'+1`) columns,
    each requiring a fixed number of :math:`O(\ell_{\rm int}^3)` transforms.
    """
    import healpy as hp  # lazy, see utils/healpy_utils.py

    lmax_i = int(lmax if lmax_int is None else lmax_int)
    if lmax_i < lmax:
        raise ValueError("lmax_int must be at least lmax")
    if term_selection is not None and grid != "gl":
        raise ValueError("term_selection is implemented for grid='gl' only")
    if max_memory_gb is not None and grid != "gl":
        raise ValueError("max_memory_gb applies to grid='gl' only")

    if grid == "gl":
        mask_alm, lw = _mask_alm_for_gl(mask, nside, lw)
        _warn_short_margin(alm2cl(mask_alm), lmax, lmax_i)
        if term_selection is not None:
            mask_alm = _pole_frame_alm(mask, mask_alm, lw)
        return _exact_row_gl(
            mask_alm,
            lw,
            cl,
            ellp,
            lmax,
            lmax_grid=lmax_grid,
            use_symmetry=use_symmetry,
            nthreads=nthreads,
            lmax_int=lmax_int,
            term_selection=term_selection,
            max_memory_gb=max_memory_gb,
        )
    if grid != "healpix":
        raise ValueError(f"grid must be 'healpix' or 'gl', got {grid!r}")
    if nside is None:
        raise ValueError("nside is required for grid='healpix'")

    mask = np.asarray(mask, dtype=float)
    cl = np.asarray(cl, dtype=float)[: lmax_i + 1]
    if mask.size != hp.nside2npix(nside):
        raise ValueError("mask size does not match nside")
    if cl.size != lmax_i + 1:
        raise ValueError("cl must have at least lmax_int+1 entries")
    if not 0 <= ellp <= lmax:
        raise ValueError("ellp must lie in [0, lmax]")
    _warn_short_margin(_mask_wl_healpix(mask, nside), lmax, lmax_i)

    ell = np.arange(lmax_i + 1)
    total = np.zeros(lmax_i + 1)

    if use_symmetry:
        m_values = range(0, ellp + 1)
    else:
        m_values = range(-ellp, ellp + 1)

    for mp in m_values:
        c = _correlation_column(mask, cl, ellp, mp, lmax_i, nside, iter, exact_adjoint)
        contrib = np.sum(np.abs(c) ** 2, axis=1)
        if use_symmetry and mp > 0:
            contrib *= 2.0
        total += contrib

    row = 2.0 / ((2 * ell + 1) * (2 * ellp + 1)) * total
    return row[: lmax + 1]


def exact_covariance(
    mask: np.ndarray,
    cl: np.ndarray,
    lmax: int,
    nside: int | None = None,
    rows: Iterable[int] | None = None,
    grid: str = "healpix",
    lw: int | None = None,
    lmax_grid: int | None = None,
    nprocs: int = 1,
    lmax_int: int | None = None,
    term_selection: float | None = None,
    max_memory_gb: float | None = None,
    **kw,
) -> np.ndarray:
    r"""
    Exact TT pseudo-``C_l`` covariance matrix ``Sigma[l, l']`` (paper Eq. 7).

    Parameters
    ----------
    mask, cl, lmax, nside, grid, lw, lmax_grid, lmax_int
        As in :func:`exact_covariance_row`.  With ``grid="gl"`` the mask is
        converted to its band-limited alm and synthesised on the GL grid once,
        shared by all rows.  ``lmax_int`` gives every row the same internal
        band-limit, so the returned ``(lmax+1, lmax+1)`` matrix equals the
        ``(lmax_int+1, lmax_int+1)`` one sliced to ``lmax``, without computing
        the rows above ``lmax``.
    rows : iterable of int, optional
        Multipoles :math:`\ell'` whose columns ``Sigma[:, l']`` are computed.
        Defaults to all ``0..lmax``.  Columns not requested are left at zero.
    nprocs : int
        ``grid="gl"`` only: number of worker *processes* over which the rows
        are distributed (they are independent).  The default 1 keeps the
        calculation in this process and is bit-identical to the serial loop.
        Above 1, each worker builds its own shared state (mask map, or ring
        modes and Legendre table) and runs ducc0 with ``nthreads`` threads; if
        ``nthreads`` is not given it
        defaults to ``max(1, cpu_count // nprocs)`` inside the workers so that
        the two levels of parallelism do not oversubscribe the machine.  Rows
        have very different costs (``O(l'^4)``), so they are submitted
        longest-first.  On platforms whose default start method is ``spawn``
        (macOS, Windows) the caller must be import-safe, i.e. inside
        ``if __name__ == "__main__":``.

        Processes are usually not faster than ducc0's own threads on a laptop
        (the threads share one copy of the maps); they are meant for many-core
        nodes where the thread scaling of one transform runs out.  The batched
        rows' matrix products run on the BLAS library's own threads, which
        ``nthreads`` does not set: with ``nprocs > 1`` limit them too
        (``OMP_NUM_THREADS``, ``OPENBLAS_NUM_THREADS`` or
        ``VECLIB_MAXIMUM_THREADS``, set before numpy is imported).
    term_selection : float, optional
        ``grid="gl"`` only; as in :func:`exact_covariance_row`.  The mask alm
        is rotated to the pole frame once here and shared by every row (and
        every worker process).
    max_memory_gb : float, optional
        ``grid="gl"`` only; as in :func:`exact_covariance_row`, per process.
        The mask's ring modes and, when it fits in half the budget, the
        Legendre table are built once and shared by every row (by every row
        of a worker with ``nprocs > 1``, each of which holds its own copy:
        the peak is then about ``nprocs`` times the budget).
    **kw
        Passed to :func:`exact_covariance_row` (``iter``, ``use_symmetry``,
        ``exact_adjoint``, ``nthreads``).

    Returns
    -------
    ndarray
        Shape ``(lmax+1, lmax+1)``.  Cost is :math:`O(\ell^5)` for all rows.
    """
    if rows is None:
        rows = range(lmax + 1)
    sigma = np.zeros((lmax + 1, lmax + 1))
    nprocs = int(nprocs)
    if nprocs < 1:
        raise ValueError("nprocs must be at least 1")
    lmax_i = int(lmax if lmax_int is None else lmax_int)
    if lmax_i < lmax:
        raise ValueError("lmax_int must be at least lmax")
    if term_selection is not None and grid != "gl":
        raise ValueError("term_selection is implemented for grid='gl' only")
    if max_memory_gb is not None and grid != "gl":
        raise ValueError("max_memory_gb applies to grid='gl' only")

    if grid == "gl":
        mask_alm, lw = _mask_alm_for_gl(mask, nside, lw)
        _warn_short_margin(alm2cl(mask_alm), lmax, lmax_i)
        if term_selection is not None:
            mask_alm = _pole_frame_alm(mask, mask_alm, lw)
        if lmax_grid is None:
            lmax_grid = gl_minimal_lmax(lmax_i, lw)
        nthreads = kw.get("nthreads")
        use_symmetry = kw.get("use_symmetry", True)
        cl = np.asarray(cl, dtype=float)
        rows = list(rows)
        if nprocs > 1:
            for ellp, row in _rows_gl_parallel(
                mask_alm,
                lw,
                cl,
                rows,
                lmax,
                lmax_grid,
                use_symmetry,
                nthreads,
                nprocs,
                lmax_int=lmax_int,
                term_selection=term_selection,
                max_memory_gb=max_memory_gb,
            ):
                sigma[:, ellp] = row
            return sigma
        batch = mask_map = None
        if lmax_grid >= gl_minimal_lmax(lmax_i, lw):
            batch = _BatchedGL(mask_alm, lw, lmax_i, lmax_grid, nthreads, max_memory_gb)
        else:
            mask_map = gl_synthesis(mask_alm, lw, lmax_grid, nthreads=nthreads)
        for ellp in rows:
            sigma[:, ellp] = _exact_row_gl(
                mask_alm,
                lw,
                cl,
                ellp,
                lmax,
                lmax_grid=lmax_grid,
                use_symmetry=use_symmetry,
                nthreads=nthreads,
                mask_map=mask_map,
                lmax_int=lmax_int,
                term_selection=term_selection,
                batch=batch,
            )
        return sigma
    if nprocs > 1:
        raise ValueError("nprocs > 1 is implemented for grid='gl' only")

    for ellp in rows:
        sigma[:, ellp] = exact_covariance_row(
            mask, cl, ellp, lmax, nside, grid=grid, lmax_int=lmax_int, **kw
        )
    return sigma


def exact_covariance_row_pol(
    mask: np.ndarray,
    cls: Mapping[str, np.ndarray],
    ellp: int,
    lmax: int,
    spectra: Sequence[str] = ("TT", "TE", "EE", "BB"),
    grid: str = "gl",
    nside: int | None = None,
    lw: int | None = None,
    lmax_grid: int | None = None,
    use_symmetry: bool = True,
    nthreads: int | None = None,
    lmax_int: int | None = None,
    term_selection: float | None = None,
    max_memory_gb: float | None = None,
    **kw,
) -> dict[tuple[str, str], np.ndarray]:
    r"""
    One row ``l'`` of the exact polarised pseudo-``C_l`` covariance.

    Returns ``Cov(C~^{s1}_l, C~^{s2}_{l'})`` for ``l = 0..lmax`` and every
    ordered pair ``(s1, s2)`` of the requested ``spectra`` (docstring of :mod:`cmbcov.exact.pol`,
    "Polarisation").  GL grid only; the HEALPix path is TT-only.

    Parameters
    ----------
    mask : ndarray
        As in :func:`exact_covariance_row` with ``grid="gl"``: a real HEALPix
        map (converted once with ``hp.map2alm(mask, lmax=lw, iter=10)``) or a
        complex healpy alm array truncated at ``lw``.
    cls : mapping
        Power spectra keyed by name among ``TT, EE, BB, TE, TB, EB`` (``"ET"``
        etc. are accepted); missing spectra are zero.  Each at least
        ``lmax_int+1`` long (``lmax+1`` when ``lmax_int`` is not given).
    ellp : int
        Multipole :math:`\ell'` of the row.
    lmax : int
        Maximum multipole of the pseudo-spectra.
    spectra : sequence of str
        Spectra whose covariance blocks are wanted.  Only the correlator
        columns their fields need are computed: ``("TT",)`` costs the same as
        :func:`exact_covariance_row`; ``("TT", "TE", "EE")`` needs the T and E
        columns; anything with B needs all three.
    grid : {"gl", "healpix"}
        ``"gl"`` (default) computes the covariance of the exact-integral
        estimator of the band-limited mask.  ``"healpix"`` is only accepted
        for ``spectra=("TT",)``, where it delegates to
        :func:`exact_covariance_row` (``nside`` required; ``iter`` and
        ``exact_adjoint`` may be passed through ``**kw``); any polarised
        spectrum raises ``NotImplementedError``.
    nside, lw, lmax_grid, use_symmetry, nthreads, lmax_int
        As in :func:`exact_covariance_row`.  ``use_symmetry`` sums
        ``m' >= 0`` only and doubles the real part of the ``m' > 0`` terms.
        ``lmax_int`` carries the spectra, the harmonic arrays and the GL grid
        to the internal band-limit and returns ``l = 0..lmax`` only.
    term_selection : float, optional
        GL grid only; as in :func:`exact_covariance_row`, with the spin-0
        and spin-2 mode powers united when a polarised field is requested.
    max_memory_gb : float, optional
        GL grid only; as in :func:`exact_covariance_row` (default
        :data:`DEFAULT_EXACT_MEMORY_GB`, 2): the columns ``m'`` of the row
        are computed in chunks, and the budget sets the chunk size and
        whether the Legendre tables (spin 0 and spin 2, up to three times
        the TT table) are held.  A polarised column carries up to nine ring
        profiles (the T, E and B columns, each on T, Q and U), so a chunk
        holds up to nine times fewer ``m'`` than a TT one at the same
        budget.  With the tables held the chunk's Legendre sums are matrix
        products against them; otherwise ducc0 transforms on the chunk's
        bands, with no table (docstring of :mod:`cmbcov.exact.gl`, "Structure of the GL
        column computation").  It changes the result only at the level of rounding.
        Raises ``ValueError`` with ``grid="healpix"``.
    **kw
        ``grid="healpix"`` only: ``iter`` and ``exact_adjoint``, forwarded to
        :func:`exact_covariance_row`.  Any keyword with ``grid="gl"`` raises
        ``TypeError``.

    Returns
    -------
    dict
        ``{(s1, s2): ndarray of shape (lmax+1,)}`` with
        ``result[(s1, s2)][l] = Cov(C~^{s1}_l, C~^{s2}_{l'})`` for all ordered
        pairs of the canonical spectrum names; ``(s2, s1)`` is the transposed
        block, i.e. a different array of the same row.

    Notes
    -----
    Per ``(l', m')`` the T column costs one spin-0 round trip plus one spin-0
    and one spin-2 round trip (the latter only if a polarised field is
    requested); each of the E and B columns costs one spin-2 round trip plus
    the same two, so the full six-spectrum row is ~7x the TT row in
    transforms.  Rows with ``l' < 2`` have zero E and B columns.  The
    batched columns (docstring of :mod:`cmbcov.exact.gl`, "Structure of the GL
    column computation") share the step-1 work of the E and B columns, which a
    rotation of the B axis makes identical.
    """
    specs, fields = _spectra_fields(spectra)
    lmax_i = int(lmax if lmax_int is None else lmax_int)
    if lmax_i < lmax:
        raise ValueError("lmax_int must be at least lmax")
    if term_selection is not None and grid != "gl":
        raise ValueError("term_selection is implemented for grid='gl' only")
    if max_memory_gb is not None and grid != "gl":
        raise ValueError("max_memory_gb applies to grid='gl' only")
    if grid == "healpix":
        if fields != ("T",):
            raise NotImplementedError(
                "polarised exact covariance is implemented on the Gauss-Legendre "
                "grid only (grid='gl'); the HEALPix path is TT-only"
            )
        cl_tt = np.asarray(cls.get("TT", np.zeros(lmax_i + 1)), dtype=float)
        row = exact_covariance_row(
            mask,
            cl_tt,
            ellp,
            lmax,
            nside=nside,
            use_symmetry=use_symmetry,
            grid="healpix",
            lmax_int=lmax_int,
            **kw,
        )
        return {("TT", "TT"): row}
    if grid != "gl":
        raise ValueError(f"grid must be 'gl' or 'healpix', got {grid!r}")
    if kw:
        raise TypeError(f"unexpected keyword arguments for grid='gl': {sorted(kw)}")

    mask_alm, lw = _mask_alm_for_gl(mask, nside, lw)
    _warn_short_margin(alm2cl(mask_alm), lmax, lmax_i)
    if term_selection is not None:
        mask_alm = _pole_frame_alm(mask, mask_alm, lw)
    cmat = _cl_matrix(cls, lmax_i)
    return _exact_row_pol_gl(
        mask_alm,
        lw,
        cmat,
        ellp,
        lmax,
        specs,
        lmax_grid=lmax_grid,
        use_symmetry=use_symmetry,
        nthreads=nthreads,
        lmax_int=lmax_int,
        term_selection=term_selection,
        max_memory_gb=max_memory_gb,
    )


def exact_covariance_pol(
    mask: np.ndarray,
    cls: Mapping[str, np.ndarray],
    lmax: int,
    spectra: Sequence[str] = ("TT", "TE", "EE", "BB"),
    rows: Iterable[int] | None = None,
    grid: str = "gl",
    nside: int | None = None,
    lw: int | None = None,
    lmax_grid: int | None = None,
    use_symmetry: bool = True,
    nthreads: int | None = None,
    lmax_int: int | None = None,
    term_selection: float | None = None,
    max_memory_gb: float | None = None,
    nprocs: int = 1,
    **kw,
) -> dict[tuple[str, str], np.ndarray]:
    r"""
    Exact polarised pseudo-``C_l`` covariance blocks ``Sigma^{s1 s2}[l, l']``.

    Parameters
    ----------
    mask, cls, lmax, spectra, grid, nside, lw, lmax_grid, use_symmetry, nthreads, lmax_int
        As in :func:`exact_covariance_row_pol`.  With ``grid="gl"`` the mask
        is converted to its band-limited alm and synthesised on the GL grid
        once, shared by all rows.
    rows : iterable of int, optional
        Multipoles :math:`\ell'` whose columns ``Sigma[:, l']`` are computed.
        Defaults to all ``0..lmax``.  Columns not requested are left at zero.
    nprocs : int
        GL grid only: number of worker *processes* over which the rows are
        distributed (they are independent), as in :func:`exact_covariance`.
        The default 1 keeps the calculation in this process and is
        bit-identical to the serial loop, and so is every ``nprocs`` for the
        same ``nthreads``.  Above 1, each worker builds its own shared state
        (mask map or ring modes and tables) and runs ducc0 with ``nthreads``
        threads; if ``nthreads`` is not given it defaults to
        ``max(1, cpu_count // nprocs)`` inside the workers.  Rows are
        submitted longest-first.  On platforms whose default start method is
        ``spawn`` (macOS, Windows) the caller must be import-safe, i.e. inside
        ``if __name__ == "__main__":``.  Processes are usually not faster than
        threads on one laptop (see :func:`exact_covariance`); they are meant
        for many-core nodes, where the thread scaling of one transform runs
        out.  The BLAS threads of the batched matrix products are not set by
        ``nthreads``: with ``nprocs > 1`` limit them too (``OMP_NUM_THREADS``,
        ``OPENBLAS_NUM_THREADS`` or ``VECLIB_MAXIMUM_THREADS``, set before
        numpy is imported).  Raises ``ValueError`` if below 1 or with
        ``grid="healpix"``.
    term_selection : float, optional
        GL grid only; as in :func:`exact_covariance_row_pol`, the mask alm
        rotated to the pole frame once and shared by every row (and every
        worker process).
    max_memory_gb : float, optional
        GL grid only; as in :func:`exact_covariance_row_pol`, per process.  The
        mask's ring modes and, when they fit in half the budget, the Legendre
        tables are built once and shared by every row (by every row of a
        worker with ``nprocs > 1``, each of which holds its own copy: the
        peak is then about ``nprocs`` times the budget).
    **kw
        HEALPix-only options (``iter``, ``exact_adjoint``) forwarded to
        :func:`exact_covariance_row`; only valid with ``spectra=("TT",)``.

    Returns
    -------
    dict
        ``{(s1, s2): ndarray of shape (lmax+1, lmax+1)}`` with
        ``result[(s1, s2)][l, l'] = Cov(C~^{s1}_l, C~^{s2}_{l'})``, so that
        ``result[(s2, s1)] == result[(s1, s2)].T`` once all rows are computed.
    """
    specs, fields = _spectra_fields(spectra)
    nprocs = int(nprocs)
    if nprocs < 1:
        raise ValueError("nprocs must be at least 1")
    lmax_i = int(lmax if lmax_int is None else lmax_int)
    if lmax_i < lmax:
        raise ValueError("lmax_int must be at least lmax")
    if rows is None:
        rows = range(lmax + 1)
    pairs = [(s1, s2) for s1 in specs for s2 in specs]
    sigma = {pair: np.zeros((lmax + 1, lmax + 1)) for pair in pairs}
    if term_selection is not None and grid != "gl":
        raise ValueError("term_selection is implemented for grid='gl' only")
    if max_memory_gb is not None and grid != "gl":
        raise ValueError("max_memory_gb applies to grid='gl' only")

    if grid == "healpix":
        if nprocs > 1:
            raise ValueError("nprocs > 1 is implemented for grid='gl' only")
        if fields != ("T",):
            raise NotImplementedError(
                "polarised exact covariance is implemented on the Gauss-Legendre "
                "grid only (grid='gl'); the HEALPix path is TT-only"
            )
        cl_tt = np.asarray(cls.get("TT", np.zeros(lmax_i + 1)), dtype=float)
        sigma[("TT", "TT")] = exact_covariance(
            mask,
            cl_tt,
            lmax,
            nside=nside,
            rows=rows,
            grid="healpix",
            use_symmetry=use_symmetry,
            lmax_int=lmax_int,
            **kw,
        )
        return sigma
    if grid != "gl":
        raise ValueError(f"grid must be 'gl' or 'healpix', got {grid!r}")
    if kw:
        raise TypeError(f"unexpected keyword arguments for grid='gl': {sorted(kw)}")

    mask_alm, lw = _mask_alm_for_gl(mask, nside, lw)
    _warn_short_margin(alm2cl(mask_alm), lmax, lmax_i)
    if term_selection is not None:
        mask_alm = _pole_frame_alm(mask, mask_alm, lw)
    if lmax_grid is None:
        lmax_grid = gl_minimal_lmax(lmax_i, lw)
    cmat = _cl_matrix(cls, lmax_i)
    rows = list(rows)
    if nprocs > 1:
        for ellp, row in _rows_gl_parallel(
            mask_alm,
            lw,
            cmat,
            rows,
            lmax,
            lmax_grid,
            use_symmetry,
            nthreads,
            nprocs,
            lmax_int=lmax_int,
            term_selection=term_selection,
            max_memory_gb=max_memory_gb,
            pol=(specs, fields),
        ):
            for pair in pairs:
                sigma[pair][:, ellp] = row[pair]
        return sigma
    batch = mask_map = None
    if lmax_grid >= gl_minimal_lmax(lmax_i, lw):
        batch = _polarised_batch(
            mask_alm, lw, lmax_i, lmax_grid, nthreads, max_memory_gb, fields
        )
    else:
        mask_map = gl_synthesis(mask_alm, lw, lmax_grid, nthreads=nthreads)
    for ellp in rows:
        row = _exact_row_pol_gl(
            mask_alm,
            lw,
            cmat,
            ellp,
            lmax,
            specs,
            lmax_grid=lmax_grid,
            use_symmetry=use_symmetry,
            nthreads=nthreads,
            mask_map=mask_map,
            lmax_int=lmax_int,
            term_selection=term_selection,
            batch=batch,
        )
        for pair in pairs:
            sigma[pair][:, ellp] = row[pair]
    return sigma
