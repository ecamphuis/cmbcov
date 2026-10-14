r"""The polarised Gauss-Legendre path of the exact covariance.

**Polarised rows** use two kernels; the per-column reference is
:func:`_correlation_columns_gl` (``_exact_row_pol_gl(..., batched=False)``).
An ``m'`` carries up to nine ring profiles (the T, E and B columns on T, Q and
U), so a chunk holds up to nine times fewer ``m'`` than a TT one, and the
tables are up to three times larger (spin 0 for T, spin 2 at twice the size).

*Tables held* (when they fit in half the memory budget):
:func:`_batched_chunk_pol_gl`, the TT scheme in a basis where ducc0's spin-2
transforms carry no complex factor.  On the coefficients (T, E, B' = iB) and
on the rings (T, Q, U'' = iU), the spin-2 synthesis and analysis of order
``M >= 0`` are the real symmetric block ``[[lambda+, lambda-], [lambda-,
lambda+]]`` of the two spin-weighted Legendre functions
(:func:`~.grid.gl_legendre_table` with ``spin=2``); at ``-M`` the table of
``|M|`` serves with :math:`\lambda^-` negated.  :math:`\lambda^+` has the
mirror parity :math:`(-1)^{L+M}` and :math:`\lambda^-` the opposite one, so
each of the four products runs on the northern rings against ``f_N + f_S`` or
``f_N - f_S`` as for spin 0.  The unit vector of a B column is ``i`` times
that of B', whose step-1 coefficients are those of the E unit vector with E
and B' swapped, so a B column costs no step-1 work.  The ``C`` mixing is ``D C
D^{-1}`` with ``D = diag(1, 1, i)``, applied per ``L``; the factors of ``i``
are restored in the contraction (:func:`_pol_terms`), which is, per ``(L,
M)``, the Gram matrix of the (output, column) pairs summed over the chunk's
columns.

*Tables streamed*: :func:`_batched_chunk_pol_direct`, with the same bands,
windows, ring stage and contraction, but every Legendre sum is a ducc0
transform (``leg2alm`` / ``alm2leg``) on the column's own orders and degrees,
so no table is stored.  With only a few ``m'`` per chunk a table cannot pay for
itself (it is regenerated for every chunk and pass and shared by few columns).
The choice follows from the memory plan alone (``hold_table``), not from
timings.  The tables must be ducc0's own values rather than those of a numpy
recursion: near the poles at high ``lmax`` the functions are ill-conditioned in
:math:`\cos\theta`, and a different recursion would no longer reproduce the
per-column row to rounding.

Polarisation (GL grid only)
---------------------------
:func:`exact_covariance_row_pol` / :func:`exact_covariance_pol` extend the
algorithm to the fields ``X in {T, E, B}`` and the six spectra TT, EE, BB, TE,
TB, EB, on the GL grid only (the HEALPix path stays TT-only).  With
``K = blockdiag(K_0, K_2)``, ``K_0 = A_0 W S_0`` on T and ``K_2 = A_2 W S_2``
on (E, B), and ``C`` the ``(L, M)``-diagonal 3x3 matrix
``[[TT, TE, TB], [TE, EE, EB], [TB, EB, BB]]``, the pseudo-alm correlators are

    R^{XZ}_{lm, l'm'} = <a~^X_lm conj(a~^Z_l'm')> = [K C K^dagger]^{XZ}_{lm, l'm'}

and on GL ``K^dagger = K`` (analysis is the adjoint of synthesis).  One column
``(Z, l'm')`` of ``R`` for every ``(X, lm)`` costs: ``u = K e_{Z, l'm'}`` (a
spin-0 round trip for Z = T, spin-2 for Z in {E, B}; the unit vector at
``(l', +m')`` has no reality partner, hence complex maps), then
``v^X_LM = sum_Z' C^{XZ'}_L u^{Z'}_LM``, then ``R^{.Z} = K v`` (spin-0 on
``v^T``, spin-2 on ``(v^E, v^B)``).  Wick's theorem with the reality condition
``a~_{l,-m} = (-1)^m conj(a~_lm)`` folded onto ``m' -> -m'`` gives

    Cov(C~^{XY}_l, C~^{ZW}_l') = 1 / ((2l+1)(2l'+1)) sum_{m m'}
        [ R^{XW} conj(R^{YZ}) + R^{XZ} conj(R^{YW}) ]_{lm, l'm'}

which reduces to ``2 sum |R^{TT}|^2`` for XY = ZW = TT.  The ``m' < 0``
column contributes the complex conjugate of the ``m' > 0`` one, so with
``use_symmetry`` the sum is ``(m'=0 term) + 2 Re sum_{m'>0}``.  On a grid
at least :func:`~.grid.gl_minimal_lmax` the columns are computed in batches
within ``max_memory_gb`` ("Structure of the GL column computation", polarised rows).  The three
columns Z = T, E, B for one ``(l', m')`` give every block among the six
spectra at once, so the unit of work is "all requested blocks of one row";
only the columns the requested spectra need are computed (T only for TT;
T and E for TT/TE/EE; ...).  ducc0's spin-2 convention is HEALPix's
(gradient, curl) = (E, B), so signs match ``healpy.anafast``.
"""

from __future__ import annotations

from collections.abc import Sequence

import ducc0
import numpy as np

from ..harmonics.grid import gl_minimal_lmax, gl_synthesis
from ..utils.threading_utils import row_block_workers, run_row_blocks
from .common import FIELDS, _spectra_fields
from .gl import (
    _BatchedGL,
    _correlation_columns_gl,
    _exact_row_gl_batched,
    _kept_mprime,
    _orders_of,
    _ring_stage,
    _signed,
    _step1_groups,
    _step1_legs,
    _step1_profiles,
)
from .plan import (
    _RING_WORKERS,
    _band,
    _chunk_fft_size,
    _first_row,
    _plane_blocks,
    _stage_rows,
    _window,
)

# --------------------------------------------------------------------------- #
# Batched polarised columns
# --------------------------------------------------------------------------- #
#: Power of ``i`` in ``d_X`` of the rotated basis (T, E, B' = iB) of
#: :func:`_batched_chunk_pol_gl`: ``d = (1, 1, i)``.
_ROT = (0, 0, 1)


def _step1_profiles_spin2(
    ctx: _BatchedGL, ellp: int, mps: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    ``wbar lambda^+_{l'm'}`` and ``wbar lambda^-_{l'm'}`` on the northern
    rings for the columns ``mps``, each ``(n_north, nc)``: the ring profiles
    of the weighted (Q, iU) map of the E unit vector at ``(l', m')``
    (:func:`~.grid.gl_legendre_table`, spin 2).  ``alm2leg`` of an E unit
    vector gives ``(lambda^+, -i lambda^-)``; for ``m' < 0``,
    :math:`\lambda^\pm_{l',-m} = \pm(-1)^m \lambda^\pm_{l'm}`.  On the
    mirror ring the two are multiplied by ``(-1)^(l'+m')`` and
    ``-(-1)^(l'+m')``.
    """
    am = np.abs(mps)
    orders = np.unique(am)
    alm = np.zeros((2, (ellp + 1) * orders.size), dtype=np.complex128)
    alm[0, ellp * orders.size + np.arange(orders.size)] = 1.0
    leg = ducc0.sht.alm2leg(
        alm=alm,
        lmax=ellp,
        theta=ctx.theta[: ctx.n_north],
        spin=2,
        mval=orders.astype(np.int64),
        mstart=np.arange(orders.size, dtype=np.int64),
        lstride=orders.size,
        nthreads=ctx.nthreads,
    )
    idx = np.searchsorted(orders, am)
    plus = leg[0].real[:, idx]
    minus = -leg[1].imag[:, idx]
    del leg, alm
    neg, odd = mps < 0, am % 2 == 1
    plus[:, neg & odd] *= -1.0
    minus[:, neg & ~odd] *= -1.0
    plus *= ctx.wbar[: ctx.n_north, None]
    minus *= ctx.wbar[: ctx.n_north, None]
    return plus, minus


def _pol_layout(fields: Sequence[str], ellp: int) -> tuple[list[int], list[int]]:
    """``(cols, outs)`` of a polarised batched row: the column types (0, 1,
    2 = T, E, B; E and B only for ``l' >= 2``) and output fields among
    ``fields``."""
    cols = [z for z in range(3) if FIELDS[z] in fields and (z == 0 or ellp >= 2)]
    outs = [x for x in range(3) if FIELDS[x] in fields]
    return cols, outs


def _pol_terms(
    specs: Sequence[str], cols: Sequence[int], rot: Sequence[int] = _ROT
) -> dict:
    r"""
    The Wick contraction of every block in the rotated basis of
    :func:`_batched_chunk_pol_gl`: ``{(s1, s2): [(sign, kind, p, q), ...]}``
    with ``Cov^{s1 s2} = sum sign * {Re, Im} S(p, q)`` and ``S(p, q) =
    sum_{m, m'} w_{m'} G'_p conj(G'_q)`` over the rotated correlators
    ``G'_{(x, z)}`` (output ``x``, column ``z``).

    With ``R^{XZ} = conj(d_X) d_Z G'^{XZ}`` and ``d = (1, 1, i)``, the term
    ``R^{ab} conj(R^{ce})`` is ``i^k G'^{ab} conj(G'^{ce})`` with ``k = -[a
    = B] + [b = B] + [c = B] - [e = B]``, whose real part is ``Re S``,
    ``-Im S``, ``-Re S`` or ``Im S`` for ``k = 0, 1, 2, 3``.  ``(p, q)`` is
    put in increasing order (``Im S(q, p) = -Im S(p, q)``), and ``Im
    S(p, p) = 0`` is dropped, as are terms with a column absent from ``cols``
    (exactly zero).  ``rot = (0, 0, 0)`` gives the contraction in the
    original basis (:func:`_batched_chunk_pol_direct`): real parts only.
    """
    blocks = {}
    for s1 in specs:
        for s2 in specs:
            x, y = FIELDS.index(s1[0]), FIELDS.index(s1[1])
            z, w = FIELDS.index(s2[0]), FIELDS.index(s2[1])
            terms = []
            for a, b, c, e in ((x, w, y, z), (x, z, y, w)):
                if b not in cols or e not in cols:
                    continue
                k = (-rot[a] + rot[b] + rot[c] - rot[e]) % 4
                kind = "re" if k % 2 == 0 else "im"
                sign = (1, -1, -1, 1)[k]
                p, q = (a, b), (c, e)
                if p > q:
                    p, q = q, p
                    if kind == "im":
                        sign = -sign
                if kind == "im" and p == q:
                    continue
                terms.append((sign, kind, p, q))
            blocks[(s1, s2)] = terms
    return blocks


def _pol_mixing(cmat: np.ndarray, cols: Sequence[int], want_t: bool) -> dict:
    r"""
    Step 2 in the rotated basis: ``{(o, z): [(coef, source), ...]}`` with
    ``v'^o = sum coef_L * source`` for the output component ``o`` (0, 1, 2 =
    T, E, B' = iB) of the column type ``z``, ``source`` among ``"I"`` (the
    step-1 T coefficients of a T column) and ``"uE"``, ``"uB"`` (the E and
    B' coefficients of an E unit vector).  ``v' = D C D^{-1} u'`` with
    ``D = diag(1, 1, i)``: a T column gives ``d_o C^{oT} I``; an E column
    ``d_o (C^{oE} u_E - i C^{oB} u_B)``; a B column is ``i`` times a B'
    unit vector, whose step-1 coefficients are those of the E unit vector
    swapped (``K'_2`` is symmetric in E and B'), so it gives ``d_o (C^{oE}
    u_B - i C^{oB} u_E)``, the ``i`` restored in the contraction
    (:func:`_pol_terms`).  Coefficients that vanish for every ``L`` are
    dropped; the T component only if ``want_t``.
    """
    rot = (1.0, 1.0, 1.0j)
    mix = {}
    for z in cols:
        for o in (0, 1, 2) if want_t else (1, 2):
            if z == 0:
                terms = [(rot[o] * cmat[o, 0], "I")]
            else:
                first, second = ("uE", "uB") if z == 1 else ("uB", "uE")
                terms = [
                    (rot[o] * cmat[o, 1], first),
                    (-1.0j * rot[o] * cmat[o, 2], second),
                ]
            kept = []
            for coef, source in terms:
                if np.any(coef):
                    if np.iscomplexobj(coef) and not np.any(coef.imag):
                        coef = np.ascontiguousarray(coef.real)
                    kept.append((coef, source))
            mix[(o, z)] = kept
    return mix


def _windowed(t0, t2, am: int, lo: int):
    """The spin-0 pair (or None) and spin-2 quadruple of order ``am`` cut
    to the degrees ``>= lo``, and the first degree of each parity."""
    ke, ko = _first_row(am, lo), _first_row(am + 1, lo)
    if t0 is not None:
        t0 = (t0[0][ke:], t0[1][ko:])
    t2 = (t2[0][ke:], t2[1][ko:], t2[2][ke:], t2[3][ko:])
    return t0, t2, (am + 2 * ke, am + 1 + 2 * ko)


def _rgemm(table: np.ndarray, x: np.ndarray) -> np.ndarray:
    """``table @ x`` for a real table and a complex ``x`` with contiguous
    rows, as one real matrix product (the real and imaginary parts ride
    along as columns)."""
    return (table @ x.view(np.float64)).view(np.complex128)


def _batched_chunk_pol_gl(
    ctx: _BatchedGL,
    mix: dict,
    ellp: int,
    mps: np.ndarray,
    weights: np.ndarray,
    cols: Sequence[int],
    outs: Sequence[int],
    keys: Sequence[tuple],
) -> dict:
    r"""
    ``S(p, q)`` (:func:`_pol_terms`) summed over the columns of the chunk
    ``mps`` of row ``ellp`` with ``weights``, for ``(kind, p, q)`` in
    ``keys``: ``{key: array over L = 0 .. lmax}``.

    The polarised counterpart of :func:`_batched_chunk_gl`, the same
    arithmetic as :func:`_correlation_columns_gl` with the transforms of all
    columns of the chunk done together.  It works in the rotated basis
    ``(T, E, B' = iB)`` on the coefficients and ``(T, Q, U'' = iU)`` on the
    rings, in which the spin-2 synthesis and analysis of order ``M >= 0``
    are the real symmetric blocks

    .. math::

        \begin{pmatrix} Q \\ U'' \end{pmatrix}_M = \sum_L
        \begin{pmatrix} \lambda^+ & \lambda^- \\ \lambda^- & \lambda^+
        \end{pmatrix}_{LM} \begin{pmatrix} E \\ B' \end{pmatrix}_{LM},
        \qquad
        \begin{pmatrix} E \\ B' \end{pmatrix}_{LM} = \sum_\theta \bar w
        \begin{pmatrix} \lambda^+ & \lambda^- \\ \lambda^- & \lambda^+
        \end{pmatrix}_{LM} \begin{pmatrix} Q \\ U'' \end{pmatrix}_M

    (ducc0's convention, :func:`~.grid.gl_legendre_table`); at ``-M`` the
    table of ``|M|`` is used with :math:`\lambda^-` negated (the common
    :math:`(-1)^M` cancels as in the TT kernel).  So every Legendre sum is a
    real matrix product on complex data, split by the mirror parity of the
    functions: :math:`\lambda^+` rows with ``L - M`` even and
    :math:`\lambda^-` rows with ``L - M`` odd are even about the equator
    (paired with ``f_N + f_S``), the other two odd (paired with ``f_N -
    f_S``, equator dropped).  Per order:

    1. Step 1: the T columns as in the TT kernel; for E, one set of legs of
       the E unit vector, ``wbar W_{M-m'} lambda^\pm_{l'm'}``
       (:func:`_step1_profiles_spin2`), four products giving its E and B'
       coefficients.  A B' unit vector has the same coefficients swapped,
       so a B column costs no step-1 work.
    2. The ``C`` mixing between fields per ``L`` (:func:`_pol_mixing`).
    3. Synthesis of T (spin 0) and (Q, U'') for every column type; the ring
       stage of the TT kernel (:func:`_ring_stage`) on all of them at once;
       analysis of T, E and B' per order, and the products ``G'_p
       conj(G'_q)`` summed over the columns with their weights, per
       ``L`` -- the six-spectrum contraction instead of ``|c|^2``.

    Orders are restricted to the bands of the TT kernel.
    """
    lmax, lw = ctx.lmax, ctx.lw
    ntheta, n_north, n_south = ctx.ntheta, ctx.n_north, ctx.n_south
    even_first = np.argsort((ellp + mps) % 2, kind="stable")
    mps = np.asarray(mps)[even_first]
    weights = np.asarray(weights, dtype=float)[even_first]
    nc = mps.size
    n_pos = int(np.count_nonzero((ellp + mps) % 2 == 0))
    band_a = _band(mps, lw, lmax)
    band_c = _band(mps, 2 * lw, lmax)
    n_fft = _chunk_fft_size(mps, lmax, lw)
    want_t = 0 in outs
    t_col = 0 in cols
    p_col = 1 in cols or 2 in cols
    nz = len(cols)
    w = nz * nc
    zpos = {z: k for k, z in enumerate(cols)}
    q0 = w if want_t else 0  # ring columns: [T][Q][U''], w each
    width = q0 + 2 * w
    fields = np.zeros((n_fft, ntheta, width), dtype=np.complex128)

    if t_col:
        prof = _step1_profiles(ctx, ellp, mps)
        groups = _step1_groups(ctx, mps, n_pos)
        plus = np.empty((n_north, 2, nc), dtype=np.complex128)
        minus = np.empty((n_south, 2, nc), dtype=np.complex128)
    if p_col:
        prof_p, prof_m = _step1_profiles_spin2(ctx, ellp, mps)
        groups_p = _step1_groups(ctx, mps, n_pos)
        groups_m = _step1_groups(ctx, mps, n_pos, swap=True)
        # [ring, sign of M, (lambda^+ leg, lambda^- leg), column]
        leg_s = np.empty((n_north, 2, 2, nc), dtype=np.complex128)
        leg_d = np.empty((n_south, 2, 2, nc), dtype=np.complex128)

    def combine(op, a, b, out=None):
        """``a + b`` (``op`` = np.add) or ``a - b`` (np.subtract, -M)."""
        return op(a, b) if out is None else op(a, b, out=out)

    def mixed(dst, lstart, n, src):
        """``dst[:, :, k] = v'`` of every (output, column) for one parity,
        rows of degree ``lstart, lstart + 2, ...``."""
        tmp = None
        for (o, z), terms in mix.items():
            k = zpos[z]
            if o == 0:
                target = dst["T"][:, :, k]
            else:
                target = dst["P"][:, :, o - 1, k]
            if not terms:
                target[...] = 0.0
                continue
            for j, (coef, source) in enumerate(terms):
                c = coef[lstart::2][:n, None, None]
                if j == 0:
                    np.multiply(src[source], c, out=target)
                else:
                    if tmp is None:
                        tmp = np.empty(src[source].shape, dtype=np.complex128)
                    np.multiply(src[source], c, out=tmp)
                    target += tmp

    # Steps 1-3 up to the synthesis, order by order (both signs together),
    # on the degrees |L - l'| <= lw where the step-1 coefficients can be
    # non-zero.
    lo, hi = _window(ellp, lw, lmax)
    for am, (t0, t2) in ctx.tables(_orders_of(band_a), hi):
        ms = _signed(am, band_a)
        if not ms:
            continue
        nm = len(ms)
        ops = [np.add if m >= 0 else np.subtract for m in ms]
        t0, t2, lstarts = _windowed(t0, t2, am, lo)
        pe, po, me, mo = t2
        n_e, n_o = pe.shape[0], po.shape[0]
        me, po = me[:, :n_south], po[:, :n_south]
        src = [{}, {}]  # per parity: step-1 coefficients by source
        if t_col:
            even, odd = t0
            for i, m in enumerate(ms):
                _step1_legs(plus[:, i], minus[:, i], m, mps, prof, groups, lw)
            src[0]["I"] = _rgemm(even, plus[:, :nm].reshape(n_north, -1))
            src[0]["I"] = src[0]["I"].reshape(n_e, nm, nc)
            if n_o:
                src[1]["I"] = _rgemm(
                    odd[:, :n_south], minus[:, :nm].reshape(n_south, -1)
                )
                src[1]["I"] = src[1]["I"].reshape(n_o, nm, nc)
        if p_col:
            for i, m in enumerate(ms):
                _step1_legs(
                    leg_s[:, i, 0], leg_d[:, i, 0], m, mps, prof_p, groups_p, lw
                )
                _step1_legs(
                    leg_s[:, i, 1], leg_d[:, i, 1], m, mps, prof_m, groups_m, lw
                )
            sums = leg_s[:, :nm].reshape(n_north, -1)
            diffs = leg_d[:, :nm].reshape(n_south, -1)
            # E = sum lambda^+ g^+ + lambda^- g^-, B' = sum lambda^- g^+ +
            # lambda^+ g^-, each function with the legs of its parity.
            for parity, (sym, anti, x_sym, x_anti) in enumerate(
                ((pe, me, sums, diffs), (mo, po, sums, diffs))
            ):
                n = (n_e, n_o)[parity]
                if not n:
                    continue
                a = _rgemm(sym, x_sym).reshape(n, nm, 2, nc)
                b = _rgemm(anti, x_anti).reshape(n, nm, 2, nc)
                if parity == 0:  # lambda^+ even: a = +, b = -
                    plus_part, minus_part = a, b
                else:  # lambda^+ odd: b = +, a = -
                    plus_part, minus_part = b, a
                u_e = np.empty((n, nm, nc), dtype=np.complex128)
                u_b = np.empty((n, nm, nc), dtype=np.complex128)
                for i in range(nm):
                    combine(ops[i], plus_part[:, i, 0], minus_part[:, i, 1], u_e[:, i])
                    combine(ops[i], plus_part[:, i, 1], minus_part[:, i, 0], u_b[:, i])
                del a, b, plus_part, minus_part
                src[parity]["uE"], src[parity]["uB"] = u_e, u_b

        # Step 2: v' = C' u' per parity, (n, nm, nz, nc) for T and
        # (n, nm, 2, nz, nc) for (E, B').
        v = [{}, {}]
        for parity in (0, 1):  # a parity may have no degree in the window
            n = (n_e, n_o)[parity]
            if want_t:
                v[parity]["T"] = np.empty((n, nm, nz, nc), dtype=np.complex128)
            v[parity]["P"] = np.empty((n, nm, 2, nz, nc), dtype=np.complex128)
            if n:
                mixed(v[parity], lstarts[parity], n, src[parity])
        del src

        # Step 3, synthesis: T as in the TT kernel, (Q, U'') from the four
        # products Pe^T, Mo^T (all northern rings) and Po^T, Me^T (odd
        # about the equator: rings with a mirror only).
        rows = [fields[m % n_fft] for m in ms]
        if want_t:
            x_e = _rgemm(even.T, v[0]["T"].reshape(n_e, nm * w))
            x_e = x_e.reshape(n_north, nm, w)
            if n_o:
                x_o = _rgemm(odd[:, :n_south].T, v[1]["T"].reshape(n_o, nm * w))
                x_o = x_o.reshape(n_south, nm, w)
            for i, row in enumerate(rows):
                if n_o:
                    np.add(x_e[:n_south, i], x_o[:, i], out=row[:n_south, :w])
                    row[n_south:n_north, :w] = x_e[n_south:, i]
                    np.subtract(x_e[:n_south, i], x_o[:, i], out=row[n_north:, :w])
                else:
                    row[:n_north, :w] = x_e[:, i]
                    row[n_north:, :w] = x_e[:n_south, i]
            del x_e
            if n_o:
                del x_o
        vp_e = v[0]["P"].reshape(n_e, nm * 2 * w)
        pe_t = _rgemm(pe.T, vp_e).reshape(n_north, nm, 2, w)
        me_t = _rgemm(me.T, vp_e).reshape(n_south, nm, 2, w)
        if n_o:
            vp_o = v[1]["P"].reshape(n_o, nm * 2 * w)
            mo_t = _rgemm(mo.T, vp_o).reshape(n_north, nm, 2, w)
            po_t = _rgemm(po.T, vp_o).reshape(n_south, nm, 2, w)
        del v
        anti = np.empty((n_south, w), dtype=np.complex128)
        for i, row in enumerate(rows):
            for c in (0, 1):  # Q, U''
                cs = slice(q0 + c * w, q0 + (c + 1) * w)
                north = row[:n_north, cs]
                if n_o:
                    combine(ops[i], pe_t[:, i, c], mo_t[:, i, 1 - c], north)
                    combine(ops[i], po_t[:, i, c], me_t[:, i, 1 - c], anti)
                else:
                    north[...] = pe_t[:, i, c]
                    if ops[i] is np.add:
                        anti[...] = me_t[:, i, 1 - c]
                    else:
                        np.negative(me_t[:, i, 1 - c], out=anti)
                np.subtract(north[:n_south], anti, out=row[n_north:, cs])
                north[:n_south] += anti
        del pe_t, me_t, anti
        if n_o:
            del mo_t, po_t

    _ring_stage(ctx, fields)

    # Step-3 analysis order by order into a staging buffer of (L, M) rows,
    # contracted over the columns whenever it is full: per row, the Gram
    # matrix G[p, q] = sum_c w_c G'_p conj(G'_q) of the n_out x nz outputs,
    # one batched matrix product for all the pairs of the six-spectrum
    # contraction (which would otherwise be dozens of small calls per order).
    n_out = len(outs)
    npair = n_out * nz
    opos = {x: k for k, x in enumerate(outs)}
    sel = np.array(
        [
            2 * ((opos[a] * nz + zpos[b]) * npair + opos[c] * nz + zpos[e])
            + (kind == "im")
            for kind, (a, b), (c, e) in keys
        ],
        dtype=np.int64,
    )
    acc = np.zeros((lmax + 1, len(keys)))
    n_stage = _stage_rows(lmax, npair, nc, len(keys))
    stage = np.empty((n_stage, n_out, nz, nc), dtype=np.complex128)
    segments: list[tuple[int, int, int]] = []  # (first row, rows, first L)

    def contract(r: int) -> None:
        """Add the staged rows ``:r`` to ``acc``, per ``L``."""
        y = stage[:r].reshape(r, npair, nc)
        gram = np.matmul(y * weights, np.conj(y).transpose(0, 2, 1))
        vals = gram.reshape(r, npair * npair).view(np.float64)[:, sel]
        del gram
        for r0, n, ell0 in segments:
            acc[ell0 : ell0 + 2 * n - 1 : 2] += vals[r0 : r0 + n]
        segments.clear()

    filled = 0
    lo, hi = _window(ellp, 2 * lw, lmax)
    for am, (t0, t2) in ctx.tables(_orders_of(band_c), hi):
        t0, t2, lstarts = _windowed(t0, t2, am, lo)
        pe, po, me, mo = t2
        n_e, n_o = pe.shape[0], po.shape[0]
        me, po = me[:, :n_south], po[:, :n_south]
        for m in _signed(am, band_c):
            if filled + n_e + n_o > n_stage:
                contract(filled)
                filled = 0
            op = np.add if m >= 0 else np.subtract
            rr = fields[m % n_fft].view(np.float64)
            north, south = rr[:n_north], rr[n_north:]
            qu = slice(2 * q0, 2 * width)
            for parity, (sym, anti, x_sym, x_anti) in enumerate(
                ((pe, me, north, south), (mo, po, north, south))
            ):
                n = (n_e, n_o)[parity]
                if not n:
                    continue
                rows = stage[filled : filled + n]
                segments.append((filled, n, lstarts[parity]))
                if want_t:
                    even, odd = t0
                    table, x = (
                        (even, north) if parity == 0 else (odd[:, :n_south], south)
                    )
                    out_t = rows[:, opos[0]].reshape(n, w).view(np.float64)
                    np.matmul(table, x[:, : 2 * w], out=out_t)
                a = _rgemm(sym, x_sym[:, qu])  # (n, 2w): [from Q, from U'']
                b = _rgemm(anti, x_anti[:, qu])
                plus_part, minus_part = (a, b) if parity == 0 else (b, a)
                if 1 in outs:  # E = lambda^+ Q + lambda^- U''
                    out_e = rows[:, opos[1]].reshape(n, w)
                    op(plus_part[:, :w], minus_part[:, w:], out=out_e)
                if 2 in outs:  # B' = lambda^- Q + lambda^+ U''
                    out_b = rows[:, opos[2]].reshape(n, w)
                    op(plus_part[:, w:], minus_part[:, :w], out=out_b)
                del a, b, plus_part, minus_part
                filled += n
    if filled:
        contract(filled)
    return {key: acc[:, j] for j, key in enumerate(keys)}


# --------------------------------------------------------------------------- #
# Batched polarised columns without tables (streamed regime)
# --------------------------------------------------------------------------- #
def _unit_legs_all(
    ctx: _BatchedGL, ellp: int, mps: np.ndarray, spin: int
) -> np.ndarray:
    r"""
    ``wbar`` times the ring profiles on every ring (natural order) of the
    unit vector at ``(l', m')`` for each column: ``(1, ntheta, nc)``
    (:math:`\lambda_{l'm'}`, spin 0) or ``(2, ntheta, nc)`` (the (Q, U)
    profile of the E unit vector, spin 2), complex.  ``alm2leg`` at
    ``|m'|``; for ``m' < 0`` the complex-linear rule ``leg(l', -|m|) =
    (-1)^m conj(leg(l', |m|))`` (:func:`~.grid.full_from_pair`).
    """
    am = np.abs(mps)
    orders = np.unique(am)
    ncomp = 1 if spin == 0 else 2
    alm = np.zeros((ncomp, (ellp + 1) * orders.size), dtype=np.complex128)
    alm[0, ellp * orders.size + np.arange(orders.size)] = 1.0
    leg = ducc0.sht.alm2leg(
        alm=alm,
        lmax=ellp,
        theta=ctx.theta,
        spin=spin,
        mval=orders.astype(np.int64),
        mstart=np.arange(orders.size, dtype=np.int64),
        lstride=orders.size,
        nthreads=ctx.nthreads,
    )
    prof = leg[:, :, np.searchsorted(orders, am)]
    neg = mps < 0
    if neg.any():
        sign = np.where(am[neg] % 2 == 1, -1.0, 1.0)
        prof[:, :, neg] = sign * np.conj(prof[:, :, neg])
    prof *= ctx.wbar[None, :, None]
    return prof


def _order_runs(orders: np.ndarray, n_fft: int) -> list[tuple[int, int, int]]:
    """``(first index, last index + 1, first slot)`` of the runs of
    consecutive ``orders`` whose FFT slots ``M % n_fft`` are consecutive."""
    slots = orders % n_fft
    cut = np.flatnonzero(np.diff(slots) != 1) + 1
    bounds = np.concatenate([[0], cut, [orders.size]])
    return [(int(a), int(b), int(slots[a])) for a, b in zip(bounds[:-1], bounds[1:])]


def _signed_groups(orders: np.ndarray) -> list[tuple[np.ndarray, bool]]:
    """``orders`` (increasing, signed) as the ``M >= 0`` group and the ``M
    < 0`` group, each ``(orders, negative)``: ducc0 takes ``|M|``, unique
    per call."""
    out = []
    pos, neg = orders[orders >= 0], orders[orders < 0]
    if pos.size:
        out.append((pos, False))
    if neg.size:
        out.append((neg, True))
    return out


def _alm_layout(orders: np.ndarray, hi: int) -> tuple[np.ndarray, int]:
    """``mstart`` and size of a compact alm array holding, for each (signed)
    order, degrees ``|M| .. hi`` (ducc0 addresses ``mstart + L``)."""
    n = hi - np.abs(orders) + 1
    first = np.concatenate([[0], np.cumsum(n)[:-1]])
    return (first - np.abs(orders)).astype(np.int64), int(n.sum())


def _batched_chunk_pol_direct(
    ctx: _BatchedGL,
    cmat: np.ndarray,
    ellp: int,
    mps: np.ndarray,
    weights: np.ndarray,
    cols: Sequence[int],
    outs: Sequence[int],
    keys: Sequence[tuple],
) -> dict:
    r"""
    :func:`_batched_chunk_pol_gl` without Legendre tables: every Legendre
    sum is a ducc0 transform (``leg2alm`` / ``alm2leg``, which evaluate the
    same functions as the per-column transforms, by their own recursion) on
    the orders of the column's band only, and on the degrees up to the top
    of the window ``|L - l'| <= lw`` (step 1, synthesis) or ``<= 2 lw``
    (analysis).  The chunk's columns still share the ring stage (one FFT
    and mask product per ring for all of them) and the contraction.

    Used when the tables would have to be streamed: a polarised ``m'``
    carries up to nine ring profiles, so a chunk holds few columns, and a
    table regenerated for each chunk and shared by only a few columns costs
    more than transforming them directly.
    It works in the original basis (T, E, B) on the coefficients and (T,
    Q, U) on the rings, the contraction taking real parts only.  ducc0 takes
    orders ``M >= 0``: at ``-M`` the spin-0 functions carry ``(-1)^M``,
    dropped as in the table kernel (it enters twice in steps 1-3 and not in
    the products of pass 2), and the spin-2 ones swap the sign of
    :math:`\lambda^-`, i.e. synthesis at ``|M|`` of ``(E, -B)`` gives ``(Q,
    -U)`` and analysis at ``|M|`` of ``(Q, -U)`` gives ``(E, -B)``
    (:func:`~.grid.banded_integrals_gl`).  The ring profiles live on planes
    ``(component, ring, order slot)``, rings in natural order, the layout
    ducc0's legs have.
    """
    lmax, lw, ntheta = ctx.lmax, ctx.lw, ctx.ntheta
    mps = np.asarray(mps)
    weights = np.asarray(weights, dtype=float)
    nc = mps.size
    n_fft = _chunk_fft_size(mps, lmax, lw)
    want_t = 0 in outs
    t_col = 0 in cols
    p_col = 1 in cols or 2 in cols
    nz = len(cols)
    zpos = {z: k for k, z in enumerate(cols)}
    ncp = int(want_t) + 2  # planes per column: [T], Q, U
    qoff = int(want_t)
    fields = np.zeros((nz * nc * ncp, ntheta, n_fft), dtype=np.complex128)
    nt = ctx.nthreads
    theta = ctx.theta

    def plane(k: int, c: int) -> int:
        return (k * nc + c) * ncp

    lo1, hi1 = _window(ellp, lw, lmax)
    prof0 = _unit_legs_all(ctx, ellp, mps, 0) if t_col else None
    prof2 = _unit_legs_all(ctx, ellp, mps, 2) if p_col else None

    # Steps 1-3 up to the synthesis, column by column.
    for c, mp in enumerate(mps):
        mp = int(mp)
        orders = np.arange(max(-hi1, mp - lw), min(hi1, mp + lw) + 1)
        if not orders.size:
            continue
        wcols = ctx.modes[:, orders[0] - mp + lw : orders[-1] - mp + lw + 1]
        for grp, negative in _signed_groups(orders):
            i0 = int(np.searchsorted(orders, grp[0]))
            w = wcols[:, i0 : i0 + grp.size]
            mval = np.abs(grp).astype(np.int64)
            mstart, size = _alm_layout(grp, hi1)
            src = {}
            if t_col:
                legs = (w * prof0[0, :, c][:, None])[None]
                src["T"] = ducc0.sht.leg2alm(
                    leg=legs,
                    lmax=hi1,
                    theta=theta,
                    spin=0,
                    mval=mval,
                    mstart=mstart,
                    nthreads=nt,
                    alm=np.empty((1, size), np.complex128),
                )[0]
                del legs
            if p_col:
                legs = w[None] * prof2[:, :, c][:, :, None]
                if negative:
                    legs[1] *= -1.0
                ueb = ducc0.sht.leg2alm(
                    leg=legs,
                    lmax=hi1,
                    theta=theta,
                    spin=2,
                    mval=mval,
                    mstart=mstart,
                    nthreads=nt,
                    alm=np.empty((2, size), np.complex128),
                )
                del legs
                if negative:
                    ueb[1] *= -1.0
                src["E"], src["B"] = ueb[0], ueb[1]
            ell = np.concatenate([np.arange(abs(int(m)), hi1 + 1) for m in grp])
            runs = _order_runs(grp, n_fft)
            for z in cols:
                k = zpos[z]
                if z == 0:
                    u = {"T": src["T"]}
                elif z == 1:
                    u = {"E": src["E"], "B": src["B"]}
                else:  # a B unit vector: (u_E, u_B) = (-u_B, u_E) of an E one
                    u = {"E": -src["B"], "B": src["E"]}
                v = {}
                for x in ((0, 1, 2) if want_t else (1, 2)):
                    acc = None
                    for zz, name in enumerate("TEB"):
                        if name not in u or not np.any(cmat[x, zz]):
                            continue
                        term = cmat[x, zz][ell] * u[name]
                        acc = term if acc is None else acc + term
                    v[x] = acc if acc is not None else np.zeros(size, np.complex128)
                p0 = plane(k, c)
                for a, b, slot in runs:
                    mv, ms_ = mval[a:b], mstart[a:b]
                    if want_t:
                        ducc0.sht.alm2leg(
                            alm=v[0][None],
                            lmax=hi1,
                            theta=theta,
                            spin=0,
                            mval=mv,
                            mstart=ms_,
                            nthreads=nt,
                            leg=fields[p0 : p0 + 1, :, slot : slot + b - a],
                        )
                    eb = np.stack([v[1], -v[2] if negative else v[2]])
                    qu = fields[p0 + qoff : p0 + qoff + 2, :, slot : slot + b - a]
                    ducc0.sht.alm2leg(
                        alm=eb,
                        lmax=hi1,
                        theta=theta,
                        spin=2,
                        mval=mv,
                        mstart=ms_,
                        nthreads=nt,
                        leg=qu,
                    )
                    if negative:
                        qu[1] *= -1.0
                    del eb
                del u, v
            del src

    _plane_ring_stage(ctx, fields)

    # Step-3 analysis into a staging buffer of (M, L) rows, contracted over
    # the columns whenever it is full (as in the table kernel).
    lo2, hi2 = _window(ellp, 2 * lw, lmax)
    band_c = _band(mps, 2 * lw, lmax)
    all_orders = np.arange(max(band_c[0], -hi2), min(band_c[1], hi2) + 1)
    n_out = len(outs)
    npair = n_out * nz
    opos = {x: j for j, x in enumerate(outs)}
    sel = np.array(
        [
            2 * ((opos[a] * nz + zpos[b]) * npair + opos[c_] * nz + zpos[e])
            + (kind == "im")
            for kind, (a, b), (c_, e) in keys
        ],
        dtype=np.int64,
    )
    acc = np.zeros((lmax + 1, len(keys)))
    n_stage = _stage_rows(lmax, npair, nc, len(keys))
    # rows 0 .. lmax are a front part ducc0 addresses (mstart is the slot
    # of L = 0) but never writes; the data rows follow
    front = lmax + 1
    stage = np.empty((front + n_stage, n_out, nz, nc), dtype=np.complex128)
    order_rows = hi2 - np.abs(all_orders) + 1
    blocks, start, rows = [], 0, 0
    for j, n in enumerate(order_rows):
        if rows + n > n_stage and j > start:
            blocks.append((start, j))
            start, rows = j, 0
        rows += n
    blocks.append((start, all_orders.size))

    for b0, b1 in blocks:
        border = all_orders[b0:b1]
        first = np.concatenate([[0], np.cumsum(order_rows[b0:b1])[:-1]])
        filled = int(order_rows[b0:b1].sum())
        view = stage[front : front + filled]
        view[...] = 0.0
        flat = stage.reshape(-1)
        for c, mp in enumerate(mps):
            mp = int(mp)
            mine = (border >= mp - 2 * lw) & (border <= mp + 2 * lw)
            if not mine.any():
                continue
            for grp, negative in _signed_groups(border[mine]):
                idx = np.searchsorted(border, grp)
                mval = np.abs(grp).astype(np.int64)
                row0 = front + first[idx] - mval  # stage row of (M, L = 0)
                for z in cols:
                    k = zpos[z]
                    p0 = plane(k, c)
                    for a, b, slot in _order_runs(grp, n_fft):
                        mv = mval[a:b]
                        if want_t:
                            col = (opos[0] * nz + k) * nc + c
                            ducc0.sht.leg2alm(
                                leg=fields[p0 : p0 + 1, :, slot : slot + b - a],
                                lmax=hi2,
                                theta=theta,
                                spin=0,
                                mval=mv,
                                mstart=(row0[a:b] * npair * nc + col).astype(np.int64),
                                lstride=npair * nc,
                                nthreads=nt,
                                alm=flat[None],
                            )
                        out_e = 1 in outs
                        out_b = 2 in outs
                        if not (out_e or out_b):
                            continue
                        qu = fields[p0 + qoff : p0 + qoff + 2, :, slot : slot + b - a]
                        if negative:
                            qu[1] *= -1.0
                        pe = opos[1] if out_e else opos[2]
                        pb = opos[2] if out_b else opos[1]
                        col = (pe * nz + k) * nc + c
                        step = (pb - pe) * nz * nc
                        if out_e and out_b:
                            alm = np.lib.stride_tricks.as_strided(
                                flat,
                                shape=(2, flat.size - step),
                                strides=(step * 16, 16),
                            )
                            ducc0.sht.leg2alm(
                                leg=qu,
                                lmax=hi2,
                                theta=theta,
                                spin=2,
                                mval=mv,
                                mstart=(row0[a:b] * npair * nc + col).astype(np.int64),
                                lstride=npair * nc,
                                nthreads=nt,
                                alm=alm,
                            )
                            if negative:
                                bcol = (pb * nz + k) * nc + c
                                for m_, r0 in zip(mv, front + first[idx[a:b]]):
                                    n = hi2 - int(m_) + 1
                                    sl = slice(
                                        r0 * npair * nc + bcol,
                                        (r0 + n) * npair * nc + bcol,
                                        npair * nc,
                                    )
                                    flat[sl] *= -1.0
                        else:  # only one of E, B wanted: a scratch pair
                            tmp_start, tmp_size = _alm_layout(grp[a:b], hi2)
                            eb = ducc0.sht.leg2alm(
                                leg=qu,
                                lmax=hi2,
                                theta=theta,
                                spin=2,
                                mval=mv,
                                mstart=tmp_start,
                                nthreads=nt,
                                alm=np.empty((2, tmp_size), np.complex128),
                            )
                            want = eb[0] if out_e else (-eb[1] if negative else eb[1])
                            dst = view[:, opos[1] if out_e else opos[2], k, c]
                            for i, (m_, r0) in enumerate(zip(mv, first[idx[a:b]])):
                                n = hi2 - int(m_) + 1
                                t0 = tmp_start[i] + int(m_)
                                dst[r0 : r0 + n] = want[t0 : t0 + n]
                            del eb
        # contract the window rows
        y = view.reshape(filled, npair, nc)
        gram = np.matmul(y * weights, np.conj(y).transpose(0, 2, 1))
        vals = gram.reshape(filled, npair * npair).view(np.float64)[:, sel]
        del gram
        for j, m_ in enumerate(border):
            am = abs(int(m_))
            lstart = max(am, lo2)
            n = hi2 - lstart + 1
            if n <= 0:
                continue
            r0 = first[j] + (lstart - am)
            acc[lstart : hi2 + 1] += vals[r0 : r0 + n]
    return {key: acc[:, j] for j, key in enumerate(keys)}


def _plane_ring_stage(ctx: _BatchedGL, fields: np.ndarray) -> None:
    """The ring stage on ``(plane, ring, order slot)`` arrays (natural ring
    order): each plane's rings to ``n_fft`` points, times the mask and the
    ring weight of the analysis, back to orders; fixed blocks of rings per
    plane handed to the workers (one FFT call each, single-threaded), so the
    result does not depend on the number of workers."""
    nplane, ntheta, n_fft = fields.shape
    mask = ctx.plane_ring_mask(n_fft)
    per, nper = _plane_blocks(n_fft, ntheta, nplane)
    nblock = nplane * nper

    def blocks(b0: int, b1: int) -> None:
        for b in range(b0, b1):
            pl, r0 = divmod(b, nper)
            r0 *= per
            r1 = min(r0 + per, ntheta)
            blk = fields[pl, r0:r1]
            ducc0.fft.c2c(blk, axes=(1,), forward=False, inorm=0, out=blk, nthreads=1)
            np.multiply(blk.view(np.float64), mask[r0:r1], out=blk.view(np.float64))
            ducc0.fft.c2c(blk, axes=(1,), forward=True, inorm=2, out=blk, nthreads=1)

    workers = min(nblock, _RING_WORKERS, row_block_workers(fields.size, ctx.nthreads))
    run_row_blocks(blocks, nblock, workers)


def _exact_row_pol_gl_batched(
    ctx: _BatchedGL,
    cmat: np.ndarray,
    ellp: int,
    m_values: Sequence[int],
    use_symmetry: bool,
    specs: Sequence[str],
    fields: Sequence[str],
) -> dict[tuple[str, str], np.ndarray]:
    """The un-normalised blocks ``sum_{m'} weight(m') sum_m [R^{XW}
    conj(R^{YZ}) + R^{XZ} conj(R^{YW})]`` (real part) of
    :func:`_exact_row_pol_gl` over ``m_values``, chunked by
    :meth:`_BatchedGL.plan`: with the Legendre tables held,
    :func:`_batched_chunk_pol_gl`; when they would have to be streamed,
    :func:`_batched_chunk_pol_direct`."""
    cols, outs = _pol_layout(fields, ellp)
    direct = not ctx.hold_table
    blocks = _pol_terms(specs, cols, (0, 0, 0) if direct else _ROT)
    keys = sorted(
        {(kind, p, q) for terms in blocks.values() for _, kind, p, q in terms}
    )
    acc = {key: np.zeros(ctx.lmax + 1) for key in keys}
    if keys:
        layout = (0 in outs, len(cols), len(outs), len(keys), direct)
        if not direct:
            mix = _pol_mixing(cmat, cols, 0 in outs)
        for mps in ctx.plan(m_values, layout=layout)["chunks"]:
            weights = np.where((mps > 0) & use_symmetry, 2.0, 1.0)
            if direct:
                part = _batched_chunk_pol_direct(
                    ctx, cmat, ellp, mps, weights, cols, outs, keys
                )
            else:
                part = _batched_chunk_pol_gl(
                    ctx, mix, ellp, mps, weights, cols, outs, keys
                )
            for key in keys:
                acc[key] += part[key]
    total = {}
    for pair, terms in blocks.items():
        total[pair] = np.zeros(ctx.lmax + 1)
        for sign, kind, p, q in terms:
            total[pair] += sign * acc[(kind, p, q)]
    return total


def _exact_row_pol_gl(
    mask_alm: np.ndarray,
    lw: int,
    cmat: np.ndarray,
    ellp: int,
    lmax: int,
    spectra: Sequence[str],
    lmax_grid: int | None = None,
    use_symmetry: bool = True,
    nthreads: int | None = None,
    mask_map: np.ndarray | None = None,
    lmax_int: int | None = None,
    term_selection: float | None = None,
    batched: bool = True,
    max_memory_gb: float | None = None,
    batch: _BatchedGL | None = None,
) -> dict[tuple[str, str], np.ndarray]:
    r"""
    Row ``l'`` of every block ``Cov(C~^{s1}_l, C~^{s2}_l')`` for ``s1, s2`` in
    ``spectra``, on the GL grid.  Arguments as in :func:`_exact_row_gl` with
    ``cmat`` from :func:`_cl_matrix`, whose last axis must run to the internal
    band-limit ``lmax_i = lmax if lmax_int is None else lmax_int``.  Returns
    ``{(s1, s2): array over l = 0..lmax}``.  ``term_selection`` skips the
    ``m'`` of :func:`_kept_mprime`, with the spin-0 and spin-2 mode powers
    united when a polarised field is requested.

    ``batched`` (default True) computes the columns in chunks by Legendre
    matrix products (:func:`_batched_chunk_pol_gl`; for ``spectra`` with T
    only, the TT kernel :func:`_batched_chunk_gl`), within ``max_memory_gb``;
    False takes one column at a time through ducc0 transforms
    (:func:`_correlation_columns_gl`), the reference.  ``batch`` is the
    row-independent state (:func:`_polarised_batch`) shared by the rows of
    :func:`exact_covariance_pol`.  A grid below
    :func:`~.grid.gl_minimal_lmax` always takes the per-column path.
    """
    import healpy as hp  # lazy, see utils/healpy_utils.py

    lmax_i = int(lmax if lmax_int is None else lmax_int)
    mask_alm = np.asarray(mask_alm)
    if mask_alm.size != hp.Alm.getsize(lw):
        raise ValueError("mask_alm size does not match lw")
    if lmax_i < lmax:
        raise ValueError("lmax_int must be at least lmax")
    if cmat.shape != (3, 3, lmax_i + 1):
        raise ValueError("cmat must have shape (3, 3, lmax_int+1)")
    if not 0 <= ellp <= lmax:
        raise ValueError("ellp must lie in [0, lmax]")
    if lmax_grid is None:
        lmax_grid = gl_minimal_lmax(lmax_i, lw)
    if lmax_grid < lmax_i:
        raise ValueError("lmax_grid must be at least the internal band-limit")

    specs, fields = _spectra_fields(spectra)
    pairs = [(s1, s2) for s1 in specs for s2 in specs]
    idx = {s: (FIELDS.index(s[0]), FIELDS.index(s[1])) for s in specs}
    ell = np.arange(lmax_i + 1)
    norm = 1.0 / ((2 * ell + 1) * (2 * ellp + 1))

    if use_symmetry:
        m_values = range(0, ellp + 1)
    else:
        m_values = range(-ellp, ellp + 1)
    keep = None
    if term_selection is not None:
        spins = (0,) if fields == ("T",) else (0, 2)
        keep = _kept_mprime(mask_alm, lw, ellp, term_selection, spins, nthreads)

    if batched and lmax_grid >= gl_minimal_lmax(lmax_i, lw):
        if batch is None:
            batch = _polarised_batch(
                mask_alm, lw, lmax_i, lmax_grid, nthreads, max_memory_gb, fields
            )
        elif (batch.lw, batch.lmax, batch.lmax_grid) != (lw, lmax_i, lmax_grid):
            raise ValueError("batch was built for another lw, lmax_int or grid")
        elif batch.spins != _polarised_spins(fields):
            raise ValueError("batch was built for other spectra")
        kept = [mp for mp in m_values if keep is None or keep[mp + ellp]]
        if fields == ("T",):
            # 2 sum |R^TT|^2: the TT kernel, with the Wick factor 2
            tt = _exact_row_gl_batched(batch, cmat[0, 0], ellp, kept, use_symmetry)
            return {("TT", "TT"): (norm * 2.0 * tt)[: lmax + 1]}
        total = _exact_row_pol_gl_batched(
            batch, cmat, ellp, kept, use_symmetry, specs, fields
        )
        return {pair: (norm * total[pair])[: lmax + 1] for pair in pairs}

    if mask_map is None:
        mask_map = gl_synthesis(mask_alm, lw, lmax_grid, nthreads=nthreads)
    total = {pair: np.zeros(lmax_i + 1, dtype=complex) for pair in pairs}

    for mp in m_values:
        if keep is not None and not keep[mp + ellp]:
            continue
        r = _correlation_columns_gl(
            mask_map, cmat, ellp, mp, lmax_i, lmax_grid, nthreads, fields
        )
        weight = 2.0 if (use_symmetry and mp > 0) else 1.0
        for s1, s2 in pairs:
            x, y = idx[s1]
            z, w = idx[s2]
            contrib = np.sum(
                r[x, w] * np.conj(r[y, z]) + r[x, z] * np.conj(r[y, w]), axis=1
            )
            total[(s1, s2)] += weight * contrib

    return {pair: (norm * total[pair].real)[: lmax + 1] for pair in pairs}


def _polarised_spins(fields: Sequence[str]) -> tuple[int, ...]:
    """Legendre tables a polarised row needs: spin 0 for T alone (the TT
    kernel), else spin 2 and spin 0 if T is among ``fields``."""
    if tuple(fields) == ("T",):
        return (0,)
    return ((0,) if "T" in fields else ()) + (2,)


def _polarised_batch(
    mask_alm: np.ndarray,
    lw: int,
    lmax: int,
    lmax_grid: int,
    nthreads: int | None,
    max_memory_gb: float | None,
    fields: Sequence[str],
) -> _BatchedGL:
    """The :class:`_BatchedGL` of a polarised row for ``fields``."""
    return _BatchedGL(
        mask_alm,
        lw,
        lmax,
        lmax_grid,
        nthreads,
        max_memory_gb,
        spins=_polarised_spins(fields),
    )
