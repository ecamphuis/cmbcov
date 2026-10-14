r"""The Gauss-Legendre column computation of the exact covariance, temperature only.

Structure of the GL column computation
--------------------------------------
**Per column** (``_exact_row_gl(..., batched=False)``, the reference).  The
complex map of a column is only ever a pair of real maps, so this path never
materialises it (nor the ``(lmax+1, 2 lmax+1)`` full-``M`` arrays): a column
is carried as the two healpy alm sets ``(ra, sa)`` of the real and imaginary
parts (:func:`_correlation_column_pair_gl`).  That takes six real transforms
per column with ``m' != 0`` and three with ``m' = 0``, plus one single-order
synthesis per part for step 1 (the map of a *single* harmonic has one
azimuthal order, so all but one term of the Legendre sum is zero --
:func:`~.grid.gl_synthesis_single_m`, bit-identical to the full synthesis).
The literal complex-map version is kept as ``_exact_row_gl(...,
reference=True)``.

**Batched** (the default).  Step 3 applies one linear operator to every column
of the row, and every transform of a column is a Legendre sum per azimuthal
order plus an FFT per ring.  A chunk of columns therefore goes through the
Legendre sums as matrix products, ``(L x rings) @ (rings x columns)`` per
order with a table of :math:`\lambda_{LM}(\theta)`
(:func:`~.grid.gl_legendre_table`, northern rings only, split by the parity
of ``L + M``), and through one batched FFT per ring
(:func:`_batched_chunk_gl`).  Step 1 needs no transform: the masked single
harmonic has the ring profile :math:`W_{M - m'}(\theta) \lambda_{\ell' m'}
(\theta)` in order ``M``, so its analysis is one more matrix product.  Only
what can be non-zero is computed: the orders ``|M - m'| <= Lw`` after step 1
and ``<= 2 Lw`` after step 3, and the degrees ``|L - l'| <= Lw`` of the
step-1 coefficients and ``|L - l'| <= 2 Lw`` of the analysis (the Gaunt
selection rule of a mask of band-limit ``Lw``), so the matrix products run
over those rows only and the table is generated only up to the top of the
window.  The FFT is only as long as the order band needs
(:func:`_chunk_fft_size`).  The chunks are sized by ``max_memory_gb``
(:func:`_batched_row_plan`), which also decides whether the Legendre table is
held for every row or regenerated per chunk into one reused buffer.  Nothing
in the plan depends on the number of threads, and the ring stage
(:func:`_ring_stage`) hands fixed blocks of rings to its workers, so a batched
row is the same to the bit for any ``nthreads``.  It agrees with the
per-column row to a few 1e-15 of its largest entry.
"""

from __future__ import annotations

from collections.abc import Sequence

import ducc0
import numpy as np

from ..approximations.acc.term_selection import mode_power, pole_rotation, rotate_alm
from ..harmonics.grid import (
    _gl_geometry,
    gl_analysis,
    gl_analysis_complex,
    gl_legendre_table,
    gl_minimal_lmax,
    gl_north_rings,
    gl_ring_modes,
    gl_synthesis,
    gl_synthesis_complex,
    gl_synthesis_single_m,
)
from ..harmonics.grid import _nthreads as _gl_nthreads
from ..harmonics.sht import ducc0_alm2map, ducc0_map2alm
from ..utils.threading_utils import row_block_workers, run_row_blocks
from .common import FIELDS
from .plan import (
    _RING_WORKERS,
    DEFAULT_EXACT_MEMORY_GB,
    _band,
    _batched_row_plan,
    _chunk_fft_size,
    _first_row,
    _ring_blocks,
    _stream_elements,
    _table_block_target,
    _table_bytes,
    _truncate,
    _window,
)


# --------------------------------------------------------------------------- #
# Gauss-Legendre path
# --------------------------------------------------------------------------- #
def _mask_alm_for_gl(
    mask: np.ndarray, nside: int | None, lw: int | None
) -> tuple[np.ndarray, int]:
    """
    Band-limited mask alm ``(mask_alm, lw)`` for the GL path.

    ``mask`` is either a complex healpy alm array (used as is; ``lw`` is
    inferred from its size unless given) or a real HEALPix map, converted once
    with ``hp.map2alm(mask, lmax=lw, iter=10)`` where ``lw`` defaults to
    ``3 * nside - 1``.
    """
    import healpy as hp  # lazy, see utils/healpy_utils.py

    mask = np.asarray(mask)
    if np.iscomplexobj(mask):
        lw_from_size = hp.Alm.getlmax(mask.size)
        if lw_from_size < 0:
            raise ValueError("mask alm size is not a valid healpy alm size")
        if lw is None:
            lw = lw_from_size
        elif lw != lw_from_size:
            raise ValueError(
                f"mask alm has band-limit {lw_from_size} but lw={lw} was given"
            )
        return np.ascontiguousarray(mask, dtype=np.complex128), int(lw)

    mask = np.asarray(mask, dtype=float)
    if nside is None:
        nside = hp.npix2nside(mask.size)
    elif mask.size != hp.nside2npix(nside):
        raise ValueError("mask size does not match nside")
    if lw is None:
        lw = 3 * nside - 1
    return ducc0_map2alm(mask, lmax=lw, iter=10), int(lw)


def _pole_frame_alm(mask, mask_alm: np.ndarray, lw: int) -> np.ndarray:
    r"""
    The mask alm rotated so that the mask's centroid sits at ``+z``
    (:func:`~cmbcov.approximations.acc.term_selection.pole_rotation`).

    The exact covariance is rotation invariant: :math:`\tilde\Sigma_{\ell\ell'}
    = \frac{2}{(2\ell+1)(2\ell'+1)} \sum_{m m'} |\langle \tilde a_{\ell m}
    \tilde a^*_{\ell' m'}\rangle|^2` sums over *all* ``(m, m')`` and a
    rotation acts on each degree by a unitary Wigner-``D`` matrix in ``m``,
    so the row depends on the mask only up to rotation.  Which ``m'`` carry
    it does depend on the frame, and the term selection of
    :func:`_kept_mprime` is only sharp in the pole frame.  ``mask`` is the
    caller's input: a pixel map is used for the moments directly; an alm
    input is synthesised at the smallest ``nside`` with ``3 nside - 1 >= lw``
    (at least 16) for that purpose only.
    """
    import healpy as hp  # lazy, see utils/healpy_utils.py

    mask = np.asarray(mask)
    if np.iscomplexobj(mask):
        nside = 16
        while 3 * nside - 1 < lw:
            nside *= 2
        mask_map = ducc0_alm2map(mask_alm, nside, lmax=lw)
    else:
        mask_map = np.asarray(mask, dtype=float)
        hp.npix2nside(mask_map.size)  # validates a HEALPix map
    return rotate_alm(mask_alm, lw, pole_rotation(mask_map))


def _kept_mprime(
    mask_alm: np.ndarray,
    lw: int,
    ellp: int,
    tolerance: float,
    spins: Sequence[int] = (0,),
    nthreads: int | None = None,
) -> np.ndarray:
    r"""
    Which orders ``m'`` of the row ``l'`` to compute under term selection:
    a bool array of length ``2 l' + 1`` (index ``m' + l'``), True where the
    mask overlap :math:`p_{m'} = \int W^2 |Y_{\ell' m'}|^2`
    (:func:`~cmbcov.approximations.acc.term_selection.mode_power`) satisfies
    :math:`p_{m'} \ge \epsilon_m \max_{m'} p_{m'}` with ``eps_m =
    tolerance / 10`` -- the same rule as
    :func:`~cmbcov.approximations.acc.term_selection.select_terms`.  Several
    ``spins`` (``(0, 2)`` for a polarised row) are united, each against its
    own maximum; a spin whose power vanishes identically (spin 2 at
    ``l' < 2``) is ignored.  The ``(m, m')`` pair rule of ``select_terms``
    does not apply here: each column's SHT returns every ``(l, m)`` at once.
    """
    if tolerance <= 0:
        raise ValueError("term_selection must be a positive tolerance")
    keep = np.zeros(2 * ellp + 1, dtype=bool)
    for spin in spins:
        p = mode_power(mask_alm, lw, ellp, spin=spin, nthreads=nthreads)
        if p.max() > 0:
            keep |= p >= (tolerance / 10) * p.max()
    if not keep.any():  # cannot happen (the maximum always passes); be safe
        keep[:] = True
    return keep


def _correlation_column_gl(
    mask_map: np.ndarray,
    cl: np.ndarray,
    ellp: int,
    mp: int,
    lmax: int,
    lmax_grid: int,
    nthreads: int | None,
) -> np.ndarray:
    r"""
    GL version of :func:`_correlation_column`: ``c[l, m] = <a~_{lm} a~*_{l'm'}>``
    with ``S``, ``A`` the exact GL synthesis/analysis and ``mask_map`` the
    band-limited mask on the same grid.  No adjoint correction is needed:
    on GL, analysis *is* the adjoint of synthesis.
    """
    e = np.zeros((lmax + 1, 2 * lmax + 1), dtype=complex)
    e[ellp, lmax + mp] = 1.0

    # Step 1: I = K^dagger e = S^dagger W S e
    y = gl_synthesis_complex(e, lmax, lmax_grid, nthreads=nthreads)
    i_lm = gl_analysis_complex(mask_map * y, lmax, lmax_grid, nthreads=nthreads)

    # Step 2: x_LM = C_L I_LM   (Eq. 10)
    x = cl[:, None] * i_lm

    # Step 3: c = K x = S^dagger W S x   (Eq. 11)
    x_map = gl_synthesis_complex(x, lmax, lmax_grid, nthreads=nthreads)
    return gl_analysis_complex(mask_map * x_map, lmax, lmax_grid, nthreads=nthreads)


def _unit_pair_alm(
    lmax: int, ellp: int, mp: int
) -> tuple[np.ndarray, np.ndarray, bool]:
    r"""
    The real-map pair of the full-``M`` unit vector at ``(ellp, mp)``.

    ``pair_from_full`` of an array with a single 1 at ``(l', m')`` has one
    nonzero entry each, at healpy index ``(l', |m'|)``:

        m' = 0:  r = 1,                s = 0
        m' > 0:  r = 1/2,              s = -i/2
        m' < 0:  r = (-1)^{m'} / 2,    s = i (-1)^{m'} / 2

    Returns ``(r, s, has_s)`` with ``has_s`` False only for ``m' = 0``, where
    the map is real and the second transform of every step is skipped (exactly
    as ``np.any(s)`` does in :func:`~.grid.gl_synthesis_complex`).
    """
    import healpy as hp  # lazy, see utils/healpy_utils.py

    nalm = hp.Alm.getsize(lmax)
    r = np.zeros(nalm, dtype=complex)
    s = np.zeros(nalm, dtype=complex)
    idx = hp.Alm.getidx(lmax, ellp, abs(mp))
    if mp == 0:
        r[idx] = 1.0
        return r, s, False
    sign = 1.0 if mp % 2 == 0 else -1.0
    if mp > 0:
        r[idx] = 0.5
        s[idx] = -0.5j
    else:
        r[idx] = 0.5 * sign
        s[idx] = 0.5j * sign
    return r, s, True


def _correlation_column_pair_gl(
    mask_map: np.ndarray,
    cl_alm: np.ndarray,
    ellp: int,
    mp: int,
    lmax: int,
    lmax_grid: int,
    nthreads: int | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    r"""
    :func:`_correlation_column_gl` in the real-map-pair representation.

    Returns the healpy alms ``(ra, sa)`` of the real and imaginary parts of the
    complex map behind the column, i.e. exactly the pair that
    :func:`~.grid.full_from_pair` would turn into the full-``M`` column ``c``;
    ``sa`` is ``None`` when it is identically zero (``m' = 0``).

    This is the same arithmetic as :func:`_correlation_column_gl` without three
    allocations:

    * the complex maps.  ``W * (f + i g)`` with real ``W`` is
      ``(W f) + i (W g)`` bit-for-bit, so the two real maps are masked in
      place and never combined.
    * the full-``M`` round trip around the ``C_L`` multiply.  ``C_L`` is real
      and ``pair_from_full(C_L full_from_pair(r, s))`` differs from
      ``(C_L r, C_L s)`` only by the rounding of ``(a-b) + (a+b)`` against
      ``2a``, i.e. by an ulp.
    * the full ``0..lmax`` order range of the step-1 synthesis, which acts on
      a single harmonic (:func:`~.grid.gl_synthesis_single_m`, bit-identical).
    """
    r, s, has_s = _unit_pair_alm(lmax, ellp, mp)
    am = abs(mp)

    def k_c_k(unit: np.ndarray) -> np.ndarray:
        """``K C K`` on one part: steps 1, 2 and 3 with ``K = S^dagger W S``."""
        field = gl_synthesis_single_m(unit, lmax, am, lmax_grid, nthreads=nthreads)
        field *= mask_map
        coeff = gl_analysis(field, lmax, lmax_grid, nthreads=nthreads)  # step 1
        coeff *= cl_alm  # step 2
        field = gl_synthesis(coeff, lmax, lmax_grid, nthreads=nthreads)
        field *= mask_map
        return gl_analysis(field, lmax, lmax_grid, nthreads=nthreads)  # step 3

    return k_c_k(r), (k_c_k(s) if has_s else None)


def _abs2_sum_over_m(
    ra: np.ndarray, sa: np.ndarray | None, ell_alm: np.ndarray, lmax: int
) -> np.ndarray:
    r"""
    ``sum_m |c[l, m]|^2`` for ``m = -l..l`` from the pair ``(ra, sa)``.

    With ``c[l, m] = ra + i sa`` and ``c[l, -m] = (-1)^m (conj ra + i conj sa)``
    (:func:`~.grid.full_from_pair`) the modulus of the ``-m`` entry is
    ``|conj(ra) + i conj(sa)|``, so the sum only needs the healpy half of the
    coefficients, without building the ``(lmax+1, 2 lmax+1)`` column.
    """
    up = ra if sa is None else ra + 1j * sa
    w = up.real**2 + up.imag**2
    nm0 = lmax + 1  # healpy stores the m = 0 block first, then m >= 1
    if ra.size > nm0:
        lo = np.conj(ra[nm0:])
        if sa is not None:
            lo = lo + 1j * np.conj(sa[nm0:])
        w[nm0:] += lo.real**2 + lo.imag**2
    return np.bincount(ell_alm, weights=w, minlength=lmax + 1)


class _BatchedGL:
    r"""
    What a batched exact GL row needs besides the column list, built once
    and shared by every row of a matrix (:func:`exact_covariance` and each
    of its worker processes, :func:`exact_covariance_pol`): the ring
    geometry, the mask's ring modes and the Legendre tables of ``spins``
    (``(0,)`` for TT; spin 2, and spin 0 if T is requested, for a polarised
    row), held whole or streamed as :func:`_batched_row_plan` decides.
    """

    def __init__(
        self,
        mask_alm: np.ndarray,
        lw: int,
        lmax: int,
        lmax_grid: int,
        nthreads: int | None,
        max_memory_gb: float | None,
        spins: Sequence[int] = (0,),
    ):
        if max_memory_gb is None:
            max_memory_gb = DEFAULT_EXACT_MEMORY_GB
        if not max_memory_gb > 0:
            raise ValueError(f"max_memory_gb must be positive, got {max_memory_gb}")
        self.lw, self.lmax, self.lmax_grid = int(lw), int(lmax), int(lmax_grid)
        self.max_memory_bytes = int(max_memory_gb * 1024**3)
        self.nthreads = _gl_nthreads(nthreads, lmax_grid)
        geom = _gl_geometry(lmax_grid)
        self.ntheta = geom["ntheta"]
        self.theta = geom["theta"]
        self.wbar = geom["wbar"]
        self.n_north = gl_north_rings(lmax_grid)
        self.n_south = self.ntheta // 2
        # Ring order of the chunk arrays: the northern rings, then the southern
        # ones in mirror order (ring ntheta-1-j at position n_north+j), so that
        # a ring and its mirror image sit n_north apart.
        self.rings = np.concatenate(
            [
                np.arange(self.n_north),
                self.ntheta - 1 - np.arange(self.n_south),
            ]
        )
        self.modes = gl_ring_modes(mask_alm, lw, lmax_grid, nthreads=self.nthreads)
        # W_N + W_S and W_N - W_S on the mirrored ring pairs (the equator, if
        # any, is its own mirror: W_eq in both), for the step-1 legs.
        north, south = (
            self.modes[: self.n_north],
            self.modes[self.rings[self.n_north :]],
        )
        self.modes_sum = north.copy()
        self.modes_sum[: self.n_south] += south
        self.modes_diff = north.copy()
        self.modes_diff[: self.n_south] -= south
        self.spins = tuple(spins)
        if not self.spins or any(sp not in (0, 2) for sp in self.spins):
            raise ValueError(f"spins must be a subset of (0, 2), got {spins}")
        full_table = _table_bytes(np.arange(lmax + 1), lmax, lmax_grid, self.spins)
        self.hold_table = full_table <= self.max_memory_bytes // 2
        self._table: dict[int, tuple] = {}
        self._stream: dict[int, np.ndarray] = {}
        self._ring_mask: tuple[int, np.ndarray] | None = None
        self._plane_mask: tuple[int, np.ndarray] | None = None

    def plan(
        self, mps: Sequence[int], layout: tuple[bool, int, int, int] | None = None
    ) -> dict:
        return _batched_row_plan(
            mps,
            self.lmax,
            self.lw,
            self.lmax_grid,
            self.max_memory_bytes,
            hold_table=self.hold_table,
            spins=self.spins,
            layout=layout,
        )

    def _generate(self, block: list[int], hi: int, stream: bool) -> list:
        """The tables of the orders ``block`` for ``L <= hi``: ``(even, odd)``
        per order for ``spins == (0,)``, else ``(spin-0 pair or None, spin-2
        quadruple)`` (:func:`~.grid.gl_legendre_table`).  ``stream``: built
        in the reused buffers of :meth:`_stream_buffer`, overwritten by the
        next block."""
        ms = np.array(block)

        def table(spin: int) -> list:
            out = self._stream_buffer(spin) if stream else None
            return gl_legendre_table(
                ms, hi, self.lmax_grid, self.nthreads, spin=spin, out=out
            )

        if self.spins == (0,):
            return table(0)
        t0 = table(0) if 0 in self.spins else [None] * ms.size
        return list(zip(t0, table(2)))

    def _stream_buffer(self, spin: int) -> np.ndarray:
        """The buffer streamed blocks of ``spin`` are built in, allocated
        once: room for the largest block (:func:`_table_block_target`) plus
        the front part :func:`~.grid.gl_legendre_table` addresses but never
        writes, so only the block itself is ever committed."""
        if spin not in self._stream:
            size = _stream_elements(self.spins, self.lmax, self.lmax_grid)[spin]
            self._stream[spin] = np.empty(size, dtype=np.complex128)
        return self._stream[spin]

    def tables(self, orders: Sequence[int], hi: int | None = None):
        """``(M, tables)`` for ``M`` in ``orders`` (increasing) with ``M <=
        hi``, rows ``L = M .. hi`` (default ``lmax``), from the held table or
        generated block by block into one reused buffer: ``(even, odd)``
        parity tables of lambda_{LM} (:func:`~.grid.gl_legendre_table`) for
        TT, the pair ``(spin 0 or None, spin 2)`` of :meth:`_generate`
        otherwise.  A streamed block overwrites the previous one, so the
        tables of an order are only valid until the next is requested."""
        hi = self.lmax if hi is None else int(hi)
        orders = [m for m in orders if m <= hi]
        if self.hold_table:
            missing = [m for m in orders if m not in self._table]
            for block in self._blocks(missing, self.lmax):
                self._table.update(zip(block, self._generate(block, self.lmax, False)))
            for m in orders:
                yield m, _truncate(self._table[m], hi - m + 1)
            return
        pending: dict = {}
        for block in self._blocks(orders, hi):
            pending = dict(zip(block, self._generate(block, hi, True)))
            for m in block:
                yield m, pending[m]

    def _blocks(self, orders: list[int], hi: int) -> list[list[int]]:
        """``orders`` in consecutive groups of about
        :func:`_table_block_target` bytes of tables up to ``hi``."""
        per_ell = _table_bytes(np.array([hi]), hi, self.lmax_grid, self.spins)
        target = _table_block_target(self.spins, self.hold_table)
        blocks, block, size = [], [], 0
        for m in orders:
            block.append(m)
            size += (hi - m + 1) * per_ell
            if size >= target:
                blocks.append(block)
                block, size = [], 0
        if block:
            blocks.append(block)
        return blocks

    def plane_ring_mask(self, n_fft: int) -> np.ndarray:
        """:meth:`ring_mask` for the planes of
        :func:`_batched_chunk_pol_direct`: rings in natural order, the value
        repeated for the real and imaginary parts, ``(ntheta, 2 n_fft)``."""
        if self._plane_mask is None or self._plane_mask[0] != n_fft:
            spec = np.zeros((self.ntheta, n_fft), dtype=np.complex128)
            ks = np.arange(-self.lw, self.lw + 1)
            np.add.at(spec.T, ks % n_fft, self.modes.T)
            ducc0.fft.c2c(
                spec, axes=(1,), forward=False, out=spec, nthreads=self.nthreads
            )
            real = spec.real * self.wbar[:, None]
            del spec
            self._plane_mask = (n_fft, np.repeat(real, 2, axis=1))
        return self._plane_mask[1]

    def ring_mask(self, n_fft: int) -> np.ndarray:
        """The mask on ``n_fft`` equispaced points of every ring times the
        ring weight ``wbar(theta)``, ``(n_fft, ntheta)`` with the rings in the
        order of :attr:`rings`: ``wbar sum_k W_k(theta) e^{2 pi i k j /
        n_fft}`` (modes that coincide modulo ``n_fft`` are added: the samples
        are exact)."""
        if self._ring_mask is None or self._ring_mask[0] != n_fft:
            spec = np.zeros((n_fft, self.ntheta), dtype=np.complex128)
            ks = np.arange(-self.lw, self.lw + 1)
            np.add.at(spec, ks % n_fft, self.modes[self.rings].T)
            ducc0.fft.c2c(
                spec, axes=(0,), forward=False, out=spec, nthreads=self.nthreads
            )
            self._ring_mask = (n_fft, spec.real * self.wbar[None, self.rings])
        return self._ring_mask[1]


def _step1_profiles(ctx: _BatchedGL, ellp: int, mps: np.ndarray) -> np.ndarray:
    r"""
    ``wbar(theta) lambda_{l' m'}(theta)`` on the northern rings for the
    columns ``mps``, shape ``(n_north, nc)``: the ring profile of the weighted
    unit map of step 1 (:math:`\lambda_{l',-m} = (-1)^m \lambda_{l'm}`; on the
    mirror ring it is multiplied by ``(-1)^(l' + m')``).  One ``alm2leg`` call
    over the distinct ``|m'|``, each entry at ``L = l'`` (``lstride`` = number
    of orders, so nothing else is read).
    """
    am = np.abs(mps)
    orders = np.unique(am)
    alm = np.zeros((1, (ellp + 1) * orders.size), dtype=np.complex128)
    alm[0, ellp * orders.size + np.arange(orders.size)] = 1.0
    leg = ducc0.sht.alm2leg(
        alm=alm,
        lmax=ellp,
        theta=ctx.theta[: ctx.n_north],
        spin=0,
        mval=orders.astype(np.int64),
        mstart=np.arange(orders.size, dtype=np.int64),
        lstride=orders.size,
        nthreads=ctx.nthreads,
    )[0].real
    prof = leg[:, np.searchsorted(orders, am)]
    prof[:, (mps < 0) & (am % 2 == 1)] *= -1.0
    prof *= ctx.wbar[: ctx.n_north, None]
    return prof


def _step1_groups(
    ctx: _BatchedGL, mps: np.ndarray, n_pos: int, swap: bool = False
) -> list:
    r"""
    The two parity groups of a chunk's columns for :func:`_step1_legs`:
    ``(lo, hi, to_plus, to_minus, stride2)`` with ``mps[:n_pos]`` the
    columns whose unit profile is even about the equator (``l' + m'`` even
    for :math:`\lambda_{l'm'}` and :math:`\lambda^+_{l'm'}`), which take
    ``W_N + W_S`` into ``g_N + g_S``; the others take ``W_N - W_S``.
    ``swap`` exchanges the two, for :math:`\lambda^-_{l'm'}`, whose mirror
    parity is the opposite.  ``stride2``: the group's ``m'`` run in steps of
    2 (a run of consecutive ``m'``), so that its mask columns are a strided
    slice rather than a gather.
    """
    groups = []
    nc = mps.size
    sums = (ctx.modes_sum, ctx.modes_diff)
    for lo, hi, (to_plus, to_minus) in (
        (0, n_pos, sums[::-1] if swap else sums),
        (n_pos, nc, sums if swap else sums[::-1]),
    ):
        if hi > lo:
            steps = np.diff(mps[lo:hi])
            groups.append((lo, hi, to_plus, to_minus, bool(np.all(steps == 2))))
    return groups


def _step1_legs(
    plus: np.ndarray,
    minus: np.ndarray,
    m: int,
    mps: np.ndarray,
    prof: np.ndarray,
    groups: list,
    lw: int,
) -> None:
    """``plus = g_N + g_S`` (``(n_north, nc)``) and ``minus = g_N - g_S``
    (``(n_south, nc)``) for ``g = W_{m - m'} prof`` (zero for ``|m - m'| >
    lw``), the step-1 legs of order ``m`` (:func:`_batched_chunk_gl`), with
    ``prof`` the columns' weighted unit profiles on the northern rings and
    ``groups`` from :func:`_step1_groups`."""
    n_south = minus.shape[0]
    for lo, hi, to_plus, to_minus, stride2 in groups:
        k = m - mps[lo:hi]
        if stride2:  # k decreases by 2: |k| <= lw on a contiguous range
            a = lo + min(hi - lo, max(0, (int(k[0]) - lw + 1) // 2))
            b = lo + min(hi - lo, (int(k[0]) + lw) // 2 + 1)
            plus[:, lo:a] = 0.0
            plus[:, max(a, b) : hi] = 0.0
            minus[:, lo:a] = 0.0
            minus[:, max(a, b) : hi] = 0.0
            if a < b:
                first = m - int(mps[a]) + lw
                np.multiply(
                    to_plus[:, first::-2][:, : b - a],
                    prof[:, a:b],
                    out=plus[:, a:b],
                )
                np.multiply(
                    to_minus[:n_south, first::-2][:, : b - a],
                    prof[:n_south, a:b],
                    out=minus[:, a:b],
                )
            continue
        inside = np.abs(k) <= lw
        cols = np.where(inside, k + lw, 0)
        np.multiply(to_plus[:, cols], prof[:, lo:hi], out=plus[:, lo:hi])
        np.multiply(
            to_minus[:n_south, cols], prof[:n_south, lo:hi], out=minus[:, lo:hi]
        )
        if not inside.all():
            plus[:, lo:hi][:, ~inside] = 0.0
            minus[:, lo:hi][:, ~inside] = 0.0


def _signed(am: int, band: tuple[int, int]) -> list[int]:
    """The orders ``+am`` and ``-am`` (one of them for ``am = 0``) in ``band``."""
    return [m for m in ((am, -am) if am else (0,)) if band[0] <= m <= band[1]]


def _ring_stage(ctx: _BatchedGL, fields: np.ndarray) -> None:
    """
    The ring stage of step 3 in place on ``fields`` (``(n_fft, ntheta,
    width)``, rings in the order of :attr:`_BatchedGL.rings`): each ring's
    orders to its ``n_fft`` points, times the mask and the ring weight of
    the analysis, back to orders; then ``f_N + f_S`` into the northern ring
    and ``f_N - f_S`` into its mirror.  The northern rings are cut into
    fixed blocks of a few rings, sized so that the passes over a block run
    in cache, and whole blocks are handed to the workers, each FFT
    single-threaded: which rings share an FFT call does not depend on the
    number of workers, so neither does the result, to the bit (ducc0's
    batched FFT rounds a column differently depending on its place in the
    batch).
    """
    n_fft, _, width = fields.shape
    n_north, n_south = ctx.n_north, ctx.n_south
    ring_mask = ctx.ring_mask(n_fft)
    step, nblock = _ring_blocks(n_fft, n_north, width)

    def fft_mask_fft(block: np.ndarray, mask: np.ndarray) -> None:
        ducc0.fft.c2c(block, axes=(0,), forward=False, inorm=0, out=block, nthreads=1)
        block *= mask[:, :, None]
        ducc0.fft.c2c(block, axes=(0,), forward=True, inorm=2, out=block, nthreads=1)

    def blocks(b0: int, b1: int) -> None:
        for b in range(b0, b1):
            q0, q1 = b * step, min((b + 1) * step, n_north)
            north = fields[:, q0:q1]
            fft_mask_fft(north, ring_mask[:, q0:q1])
            p1 = min(q1, n_south)
            if q0 < p1:
                south = fields[:, n_north + q0 : n_north + p1]
                fft_mask_fft(south, ring_mask[:, n_north + q0 : n_north + p1])
                paired = north[:, : p1 - q0]
                # f_N - f_S, then f_N + f_S = 2 f_N - (f_N - f_S): no buffer
                np.subtract(paired, south, out=south)
                paired *= 2.0
                paired -= south

    workers = min(nblock, _RING_WORKERS, row_block_workers(fields.size, ctx.nthreads))
    run_row_blocks(blocks, nblock, workers)


def _orders_of(band: tuple[int, int]) -> range:
    """The distinct ``|M|`` of the orders ``M`` in ``band``."""
    lo, hi = band
    if lo <= 0 <= hi:
        return range(0, max(-lo, hi) + 1)
    return range(min(abs(lo), abs(hi)), max(abs(lo), abs(hi)) + 1)


def _batched_chunk_gl(
    ctx: _BatchedGL,
    cl: np.ndarray,
    ellp: int,
    mps: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    r"""
    ``sum_c weights[c] sum_M |c_c[L, M]|^2`` over the columns ``c`` of the
    chunk ``mps`` (orders ``m'`` of row ``ellp``), for ``L = 0 .. lmax``.

    The same linear algebra as :func:`_correlation_column_pair_gl`, with the
    spherical-harmonic transforms of every column of the chunk done together:

    1. Step 1, :math:`I = S^\dagger W Y_{\ell' m'}`.  The ring profile of
       mode ``M`` of :math:`W Y_{\ell' m'}` is
       :math:`W_{M - m'}(\theta) \lambda_{\ell' m'}(\theta)`, non-zero only
       for ``|M - m'| <= lw``, so the analysis is, per order,
       :math:`I_{LM} = \sum_\theta \lambda_{LM} \bar w W_{M-m'}
       \lambda_{\ell' m'}` -- one matrix product with the Legendre table
       for all columns (no FFT, no map).
    2. :math:`x_{LM} = C_L I_{LM}`.
    3. Synthesis per order, :math:`X_M(\theta) = \sum_L x_{LM}
       \lambda_{LM}(\theta)` (a matrix product), for the orders of the band
       ``|M - m'| <= lw``; FFT to ``N`` points per ring
       (:func:`_chunk_fft_size`), multiply by the mask and the ring weight
       :math:`\bar w`, FFT back; analysis per order, :math:`c_{LM} =
       \sum_\theta \lambda_{LM} f_M(\theta)`, for the orders
       ``|M - m'| <= 2 lw`` that can be non-zero, and ``|c|^2`` summed at
       once.

    Each matrix product runs on the northern rings only, the degrees split
    by the parity of ``L + M`` (:func:`~.grid.gl_legendre_table`): a sum over
    all rings of :math:`\lambda_{LM} g` is ``even @ (g_N + g_S) + odd @ (g_N
    - g_S)``, and a synthesis gives ``even x + odd x`` on a northern ring and
    ``even x - odd x`` on its mirror.  The rings of the chunk arrays are in
    the order of :attr:`_BatchedGL.rings` (mirror pairs ``n_north`` apart),
    the step-1 legs are built directly as ``g_N +- g_S`` from
    :math:`W_N \pm W_S` (columns grouped by the parity of ``l' + m'``, which
    decides the sign of the mirror profile), and the FFT workers leave
    ``f_N +- f_S`` behind, so no pass over the arrays is spent on the
    mirror sums.  The odd functions vanish on the equator, whose term is
    dropped from the odd products.  ``+M`` and ``-M`` share the table of
    ``|M|`` (their sign ``(-1)^M`` enters twice in steps 1-3 and not at all
    in ``|c|^2``).  Orders outside the bands are exact zeros of the
    per-column path (up to its rounding); skipping them removes the most
    work at ``lmax >> lw``.

    Only ``matmul``, element-wise products and FFTs along one axis are used,
    so the per-chunk arithmetic could move to another array namespace; the
    table and ring modes come from ducc0 (CPU).
    """
    lmax, lw = ctx.lmax, ctx.lw
    ntheta, n_north, n_south = ctx.ntheta, ctx.n_north, ctx.n_south
    # Columns grouped by the parity of l' + m' (the sign of the mirror
    # profile); the result is a sum over columns, so their order is free.
    even_first = np.argsort((ellp + mps) % 2, kind="stable")
    mps = np.asarray(mps)[even_first]
    weights = np.asarray(weights, dtype=float)[even_first]
    nc = mps.size
    n_pos = int(np.count_nonzero((ellp + mps) % 2 == 0))
    band_a = _band(mps, lw, lmax)
    band_c = _band(mps, 2 * lw, lmax)
    n_fft = _chunk_fft_size(mps, lmax, lw)
    prof = _step1_profiles(ctx, ellp, mps)  # (n_north, nc)

    fields = np.zeros((n_fft, ntheta, nc), dtype=np.complex128)
    plus = np.empty((n_north, 2, nc), dtype=np.complex128)
    minus = np.empty((n_south, 2, nc), dtype=np.complex128)
    groups = _step1_groups(ctx, mps, n_pos)

    # Steps 1-3 up to the synthesis, order by order, on the degrees
    # |L - l'| <= lw where the step-1 coefficients can be non-zero.
    lo, hi = _window(ellp, lw, lmax)
    for am, (even, odd) in ctx.tables(_orders_of(band_a), hi):
        ms = _signed(am, band_a)
        if not ms:
            continue
        nm = len(ms)
        ke, ko = _first_row(am, lo), _first_row(am + 1, lo)
        even, odd = even[ke:], odd[ko:]
        for i, m in enumerate(ms):
            _step1_legs(plus[:, i], minus[:, i], m, mps, prof, groups, lw)
        u = (even @ plus[:, :nm].reshape(n_north, -1).view(np.float64)).view(
            np.complex128
        )  # (n_even, nm nc)
        u *= cl[am + 2 * ke :: 2][: even.shape[0], None]
        x_even = (even.T @ u.view(np.float64)).view(np.complex128)
        x_even = x_even.reshape(n_north, nm, nc)
        if odd.shape[0] and n_south:
            odd = odd[:, :n_south]
            u = (odd @ minus[:, :nm].reshape(n_south, -1).view(np.float64)).view(
                np.complex128
            )
            u *= cl[am + 1 + 2 * ko :: 2][: odd.shape[0], None]
            x_odd = (odd.T @ u.view(np.float64)).view(np.complex128)
            x_odd = x_odd.reshape(n_south, nm, nc)
            for i, m in enumerate(ms):
                row = fields[m % n_fft]
                np.add(x_even[:n_south, i], x_odd[:, i], out=row[:n_south])
                row[n_south:n_north] = x_even[n_south:, i]
                np.subtract(x_even[:n_south, i], x_odd[:, i], out=row[n_north:])
        else:
            for i, m in enumerate(ms):
                row = fields[m % n_fft]
                row[:n_north] = x_even[:, i]
                row[n_north:] = x_even[:n_south, i]

    _ring_stage(ctx, fields)

    # Step-3 analysis and |c|^2, order by order, straight from the rows, on
    # the degrees |L - l'| <= 2 lw where c can be non-zero.
    total = np.zeros(lmax + 1)
    w_rep = np.repeat(weights, 2)
    lo, hi = _window(ellp, 2 * lw, lmax)
    for am, (even, odd) in ctx.tables(_orders_of(band_c), hi):
        ke, ko = _first_row(am, lo), _first_row(am + 1, lo)
        even, odd = even[ke:], odd[ko:]
        for m in _signed(am, band_c):
            row = fields[m % n_fft].view(np.float64)
            c = even @ row[:n_north]
            c *= c
            total[am + 2 * ke :: 2][: even.shape[0]] += c @ w_rep
            if odd.shape[0] and n_south:
                c = odd[:, :n_south] @ row[n_north:]
                c *= c
                total[am + 1 + 2 * ko :: 2][: odd.shape[0]] += c @ w_rep
    return total


def _exact_row_gl_batched(
    ctx: _BatchedGL,
    cl: np.ndarray,
    ellp: int,
    m_values: Sequence[int],
    use_symmetry: bool,
) -> np.ndarray:
    """``sum_{m'} weight(m') sum_m |c[l, m]|^2`` over ``m_values``, chunked
    by :meth:`_BatchedGL.plan` (the un-normalised row of :func:`_exact_row_gl`)."""
    total = np.zeros(ctx.lmax + 1)
    for mps in ctx.plan(m_values)["chunks"]:
        weights = np.where((mps > 0) & use_symmetry, 2.0, 1.0)
        total += _batched_chunk_gl(ctx, cl, ellp, mps, weights)
    return total


def _exact_row_gl(
    mask_alm: np.ndarray,
    lw: int,
    cl: np.ndarray,
    ellp: int,
    lmax: int,
    lmax_grid: int | None = None,
    use_symmetry: bool = True,
    nthreads: int | None = None,
    mask_map: np.ndarray | None = None,
    reference: bool = False,
    lmax_int: int | None = None,
    term_selection: float | None = None,
    batched: bool = True,
    max_memory_gb: float | None = None,
    batch: _BatchedGL | None = None,
) -> np.ndarray:
    r"""
    ``Sigma[:, ellp]`` of the exact-integral estimator for a band-limited mask.

    Parameters
    ----------
    mask_alm : ndarray
        healpy alm of the mask, truncated at ``lw``.
    lw : int
        Band-limit of the mask.
    cl, ellp, lmax, use_symmetry, lmax_int
        As in :func:`exact_covariance_row`.  The internal band-limit
        ``lmax_i = lmax if lmax_int is None else lmax_int`` drives every array
        here; the returned row is sliced back to ``lmax``.
    lmax_grid : int, optional
        GL grid band-limit.  ``None`` uses :func:`~.grid.gl_minimal_lmax`
        ``(lmax_i, lw)`` -- the *internal* band-limit, not the reported one --
        the smallest grid on which every transform in the algorithm is exact;
        a smaller grid aliases (error ~1e-2 one step below the minimum), a
        larger one only costs time.
    nthreads : int, optional
        Threads for ducc0.
    mask_map : ndarray, optional
        ``gl_synthesis(mask_alm, lw, lmax_grid)`` precomputed by the caller to
        share it across rows.
    reference : bool
        If True, use the literal complex-map implementation
        (:func:`_correlation_column_gl`) instead of the real-pair one (and
        not the batched one).  Same answer to ~1e-14 relative, but slower than
        the real-pair columns; kept as a reference.
    term_selection : float, optional
        Skip the orders ``m'`` that :func:`_kept_mprime` drops at this
        tolerance.  ``mask_alm`` must already be in the pole frame
        (:func:`_pole_frame_alm`; the public entry points do this once) for
        the selection to be effective; it is correct in any frame.
    batched : bool
        If True (default), the columns are computed in chunks by Legendre
        matrix products (:func:`_batched_chunk_gl`, module docstring
        "Structure of the GL column computation"); if False, one column at a
        time through ducc0 transforms (:func:`_correlation_column_pair_gl`).
        The two agree to a few 1e-15 of the largest entry.  A grid below :func:`~.grid.gl_minimal_lmax`
        (inexact on purpose) always takes the per-column path, whose
        aliasing it then reproduces.
    max_memory_gb : float, optional
        Memory budget of the batched path in GiB (default
        :data:`DEFAULT_EXACT_MEMORY_GB`), see :func:`_batched_row_plan`.
    batch : _BatchedGL, optional
        Row-independent state of the batched path built by the caller
        (:func:`exact_covariance`) to share the ring modes and the Legendre
        table across rows; built here when not given.  Its ``max_memory_gb``
        then applies and the argument above is ignored.

    Returns
    -------
    ndarray
        ``Sigma[l, ellp]`` for ``l = 0..lmax``.
    """
    import healpy as hp  # lazy, see utils/healpy_utils.py

    lmax_i = int(lmax if lmax_int is None else lmax_int)
    mask_alm = np.asarray(mask_alm)
    cl = np.asarray(cl, dtype=float)[: lmax_i + 1]
    if mask_alm.size != hp.Alm.getsize(lw):
        raise ValueError("mask_alm size does not match lw")
    if lmax_i < lmax:
        raise ValueError("lmax_int must be at least lmax")
    if cl.size != lmax_i + 1:
        raise ValueError("cl must have at least lmax_int+1 entries")
    if not 0 <= ellp <= lmax:
        raise ValueError("ellp must lie in [0, lmax]")
    if lmax_grid is None:
        lmax_grid = gl_minimal_lmax(lmax_i, lw)
    if lmax_grid < lmax_i:
        raise ValueError("lmax_grid must be at least the internal band-limit")

    ell = np.arange(lmax_i + 1)
    if use_symmetry:
        m_values = range(0, ellp + 1)
    else:
        m_values = range(-ellp, ellp + 1)
    keep = (
        None
        if term_selection is None
        else _kept_mprime(mask_alm, lw, ellp, term_selection, (0,), nthreads)
    )

    if batched and not reference and lmax_grid >= gl_minimal_lmax(lmax_i, lw):
        if batch is None:
            batch = _BatchedGL(mask_alm, lw, lmax_i, lmax_grid, nthreads, max_memory_gb)
        elif (batch.lw, batch.lmax, batch.lmax_grid) != (lw, lmax_i, lmax_grid):
            raise ValueError("batch was built for another lw, lmax_int or grid")
        kept = [mp for mp in m_values if keep is None or keep[mp + ellp]]
        total = _exact_row_gl_batched(batch, cl, ellp, kept, use_symmetry)
        row = 2.0 / ((2 * ell + 1) * (2 * ellp + 1)) * total
        return row[: lmax + 1]

    if mask_map is None:
        mask_map = gl_synthesis(mask_alm, lw, lmax_grid, nthreads=nthreads)
    total = np.zeros(lmax_i + 1)
    ell_alm = hp.Alm.getlm(lmax_i)[0]
    cl_alm = cl[ell_alm]

    for mp in m_values:
        if keep is not None and not keep[mp + ellp]:
            continue
        if reference:
            c = _correlation_column_gl(
                mask_map, cl, ellp, mp, lmax_i, lmax_grid, nthreads
            )
            contrib = np.sum(np.abs(c) ** 2, axis=1)
        else:
            ra, sa = _correlation_column_pair_gl(
                mask_map, cl_alm, ellp, mp, lmax_i, lmax_grid, nthreads
            )
            contrib = _abs2_sum_over_m(ra, sa, ell_alm, lmax_i)
        if use_symmetry and mp > 0:
            contrib *= 2.0
        total += contrib

    row = 2.0 / ((2 * ell + 1) * (2 * ellp + 1)) * total
    return row[: lmax + 1]


def _apply_k_gl(
    x_full: np.ndarray,
    spin: int,
    mask_map: np.ndarray,
    lmax: int,
    lmax_grid: int,
    nthreads: int | None,
) -> np.ndarray:
    """``K x = S^dagger W S x`` on full-``M`` coefficients (spin 0 or 2)."""
    y = gl_synthesis_complex(x_full, lmax, lmax_grid, spin=spin, nthreads=nthreads)
    return gl_analysis_complex(
        mask_map * y, lmax, lmax_grid, spin=spin, nthreads=nthreads
    )


def _correlation_columns_gl(
    mask_map: np.ndarray,
    cmat: np.ndarray,
    ellp: int,
    mp: int,
    lmax: int,
    lmax_grid: int,
    nthreads: int | None,
    fields: Sequence[str],
) -> np.ndarray:
    r"""
    ``R[X, Z, l, m] = <a~^X_lm conj(a~^Z_l'm')>`` for the columns
    ``Z in fields`` and every ``X in fields`` (other entries left at zero),
    shape ``(3, 3, lmax+1, 2*lmax+1)``.  Generalises
    :func:`_correlation_column_gl` (its result is ``R[0, 0]``).
    """
    nf = len(FIELDS)
    r = np.zeros((nf, nf, lmax + 1, 2 * lmax + 1), dtype=complex)
    want_t = "T" in fields
    want_p = "E" in fields or "B" in fields

    for z in fields:
        iz = FIELDS.index(z)
        # Step 1: u = K e_{Z, l'm'}
        u = np.zeros((nf, lmax + 1, 2 * lmax + 1), dtype=complex)
        if iz == 0:
            e = np.zeros((lmax + 1, 2 * lmax + 1), dtype=complex)
            e[ellp, lmax + mp] = 1.0
            u[0] = _apply_k_gl(e, 0, mask_map, lmax, lmax_grid, nthreads)
        else:
            if ellp < 2:
                continue  # E and B have no l' < 2 modes
            e = np.zeros((2, lmax + 1, 2 * lmax + 1), dtype=complex)
            e[iz - 1, ellp, lmax + mp] = 1.0
            u[1:] = _apply_k_gl(e, 2, mask_map, lmax, lmax_grid, nthreads)

        # Step 2: v^X_LM = sum_Z' C^{XZ'}_L u^{Z'}_LM
        v = np.einsum("xzl,zlm->xlm", cmat, u)

        # Step 3: R^{.Z} = K v
        if want_t and np.any(v[0]):
            r[0, iz] = _apply_k_gl(v[0], 0, mask_map, lmax, lmax_grid, nthreads)
        if want_p and np.any(v[1:]):
            r[1:, iz] = _apply_k_gl(v[1:], 2, mask_map, lmax, lmax_grid, nthreads)
    return r
