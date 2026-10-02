r"""
Filter-and-bin map-making in the pseudo-:math:`C_\ell` covariance (T and E).

The operator
------------
Filter-and-bin map-making filters each scan of the time stream and bins it
into a map.  For scans along the iso-latitude rings of the map's coordinate
system the result is a ring-local linear operator :math:`F` applied to the
sky *before* the analysis mask, so the pseudo-alm operator of
:mod:`cmbcov.exact` (``K = A W S``) becomes

.. math::

    \tilde a = K a, \qquad K = A\, W F\, S, \qquad K^\dagger = A\, F W\, S ,

with ``S`` the Gauss-Legendre synthesis, ``A = S^dagger`` the GL analysis
(:mod:`cmbcov.grid`), ``W`` the multiplication by the mask and ``F`` real and
self-adjoint.  For polarisation ``K_2 = A_2 W F S_2`` acts on ``(E, B)``,
with ``F`` applied to the Q map and to the U map separately (spin-0 ``K``
is written ``K_0`` below).  A Fourier filter along the ring (:class:`FourierRingFilter`)
multiplies the azimuthal mode ``M`` of the ring at colatitude
:math:`\theta` by :math:`h(\theta, |M|)`; :func:`highpass_profile` gives the
sharp and the exponential high-pass at :math:`m_c = \ell_x \sin\theta`, the
azimuthal order at which a ring of that colatitude sees the flat-sky
wavenumber :math:`\ell_x` along the scan.  A cut that depends on
:math:`\theta` is not band-limited, so the *discrete* operator on the GL grid
is the definition.

The filtered sum rule
---------------------
For a white spectrum the covariance of the filtered estimator is
:math:`2\,\Xi^F(\ell,\ell')` with

.. math::

    \Xi^F(\ell,\ell') = \frac{1}{n n'} \sum_{m m'}
        \left| (K K^\dagger)_{\ell m, \ell' m'} \right|^2 ,
    \qquad n = 2\ell+1,\ n' = 2\ell'+1 ,

and :math:`\Xi` is the same sum with ``F = 1``.  There is no 3j closed form
for :math:`\Xi^F`: that of :math:`\Xi` rests on :math:`K K^\dagger =
S^\dagger W^2 S` being a multiplication operator, and :math:`W F F^\dagger W`
is not one.  The object computed here is the ratio

.. math::

    \rho(\ell, \ell') = \Xi^F(\ell,\ell') / \Xi(\ell,\ell') ,

and the filtered covariance is modelled as :math:`\rho(\ell,\ell')` times the
unfiltered covariance of the signal spectrum :math:`C_L`, or equivalently
(the post-processing form of :func:`transfer_correction`) the unfiltered
covariance of :math:`F_L C_L` times :math:`\rho(\ell,\ell') / (F_\ell
F_{\ell'})`, :math:`F_\ell` being the transfer function of the mean
pseudo-spectrum.

The probing identity
--------------------
For a real Gaussian field :math:`z` of unit variance supported on the single
degree :math:`\ell'` (:math:`\langle z z^\dagger\rangle` the identity on that
degree),

.. math::

    E \sum_m \left| (K K^\dagger z)_{\ell m} \right|^2
        = \sum_{m m'} \left| (K K^\dagger)_{\ell m, \ell' m'} \right|^2
        = n n'\, \Xi^F(\ell, \ell')

for every :math:`\ell` at once.  Each probe costs one synthesis of ``z``
(shared), three transforms and two filter applications for the filtered
operator, three transforms for the unfiltered one.  Pushing the *same* probes
through both operators and taking the ratio of the two sums cancels most of
the noise: on the toy patches of the research notes the rms error of
:math:`\rho` at :math:`\ell' = 160` was 0.9-3.3% with 16 probes and
0.4-1.8% with 64 (against 5% for 16 unpaired probes), falling as
:math:`1/\sqrt{N \ell'}`.  When :math:`2\ell'+1` does not exceed the number
of probes the complete orthonormal real basis of the degree is used instead
and the ratio is exact (:func:`filtered_sum_rule_ratio`).

Polarisation channels
---------------------
Every T/E covariance block is normalised by one of three sum-rule channels
(the table of :meth:`cmbcov.covariance.Cov._compute_norm_xi`, reproduced in
:data:`BLOCK_CHANNEL`): ``"00"`` for TT x TT, ``"20"`` for the blocks of
spin weight 1 (TT x EE, TT x TE, TE x TE, ... and transposes) and ``"EE"``
for EE x EE, EE x TE and transposes.  Each is probed with the same ``z`` as
above (degree :math:`\ell'` of the right-hand field), and
:math:`\rho^{ch} = \Xi^{F,ch} / \Xi^{ch}`:

* ``"00"``: ``z`` spin 0, :math:`y = K_0 K_0^\dagger z`;
* ``"20"``: ``z`` an E mode (``B = 0``), :math:`u = [K_2^\dagger
  (z, 0)]_E` (the B output dropped), :math:`y = K_0 u` (``u`` read as a
  spin-0 alm);
* ``"EE"``: ``z`` an E mode, ``u`` as for ``"20"``, :math:`y = [K_2 (u,
  0)]_E`.

For each channel :math:`2\,\Xi^{F,ch}` is the exact filtered covariance of a
white spectrum: TT x EE with only :math:`C^{TE} = 1` for ``"20"``, EE x EE
with only :math:`C^{EE} = 1` for ``"EE"`` (to round-off, 2e-16 to 8e-16, in
``tests/test_filtering.py``).  The ``"20"`` probe puts T on the left and E
on the right; its transpose agreed with it to 0.02% at :math:`\ell' = 160` on
the toy patch, and :func:`rho_matrix` is symmetric anyway.  A spin-2
transform costs two to three spin-0 ones, so only the channels a run needs
are computed
(:func:`channels_for`).  Blocks with a B leg are not corrected: see
docs/theory/filter_and_bin.md.

Why :math:`\rho` and not :math:`F_\ell`
---------------------------------------
The filter removes modes whose wavevector points across the scan.  Their
weight in the diagonal :math:`d = \ell - \ell'` of the covariance is set by
the extent of the mask's :math:`|\widetilde{W^2}|^2` along the tangent of
the annulus, i.e. by the *shape* of the mask, and it changes with
:math:`d`.  Measured with the same cut (:math:`m_c = 30 \sin\theta`) on three
masks that all have :math:`F_{100} = 0.80`: :math:`\rho(100, 100) = 0.855`
(patch wide along the scan), 0.782 (cap), 0.685 (patch elongated across the
scan); on the patch :math:`\rho` falls from 0.855 at :math:`d = 0` to 0.68
at :math:`d = 8`, on the tall patch it rises from 0.685 to 0.97.  No function
of :math:`F_\ell` alone can reproduce this: :math:`C \to F C` leaves the
variance 11-62% low, and a :math:`1/\sqrt{F_\ell F_{\ell'}}` rescaling of it
is still off by -6%, +2% and +16% at :math:`\ell = 100` on the three masks.
:math:`\rho` is smooth in :math:`\ell`, so it is computed at a few nodes
(:func:`default_nodes`) and interpolated (:func:`rho_matrix`).

The scan frame
--------------
The filter fixes the frame: the rings of the map's coordinate system are the
scan lines, so the mask alm must be given in that frame (its own polar axis
is the scan axis).  In particular the pole-frame rotation of
:mod:`cmbcov.term_selection` must **not** be applied to the mask passed here.
The unfiltered kernels are rotation invariant and unaffected.

Accuracy
--------
Validated at toy scale only (Gauss-Legendre grid 360, :math:`\ell \le 320`,
mask band-limit 64, cut :math:`\ell_x = 30`) against exact filtered
covariance rows.  TT: :math:`\rho(\ell,\ell')` times the unfiltered
covariance is within 0.25% for :math:`\ell \gtrsim 3` to :math:`4\,\ell_x`,
below 1% at :math:`\ell \approx 2.3\,\ell_x` and 3-5% at
:math:`\ell \approx 1.7\,\ell_x`; PCHIP interpolation in :math:`\ln\ell`
between nodes a factor 1.3-1.7 apart adds about 0.1% or less.  Below about
three times the cut the kernel *shape* changes and the model is only good to
a few per cent; those multipoles have :math:`F_\ell < 0.7`.  T and E: with
each block's own channel and the transfer functions of its two spectra
(:math:`F^{TE}` a smooth positive reference such as
:math:`\sqrt{F^{TT} F^{EE}}`), every T/E block was within 0.7% (exact
:math:`\rho`) and 0.5% (probed) of the exact filtered block at
:math:`\ell \ge 3.3\,\ell_x`, 2-3% at :math:`2\,\ell_x`, for both filter
shapes.  Filters that are not diagonal in ``M`` (polynomial subtraction on a
partial ring, point-source masking during filtering) have the same
:math:`\rho` machinery available through any self-adjoint callable, but only
:class:`FourierRingFilter` is provided.  B modes, cross-spectra between maps
with different filters, and realistic :math:`\ell_{max}` are not validated.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from .exact import mask_coupling_width
from .grid import gl_analysis, gl_minimal_lmax, gl_shape, gl_synthesis, gl_thetas
from .sht import alm2cl

__all__ = [
    "CHANNELS",
    "BLOCK_CHANNEL",
    "channels_for",
    "highpass_profile",
    "FourierRingFilter",
    "default_nodes",
    "filtered_sum_rule_ratio",
    "rho_matrix",
    "transfer_correction",
    "cached_sum_rule_ratio",
]

#: Sub-directory of the cache directory holding :func:`cached_sum_rule_ratio`
#: files.
CACHE_SUBDIR = "filtered_sum_rule"

#: The sum-rule channels, in canonical order (module docstring,
#: "Polarisation channels").
CHANNELS = ("00", "20", "EE")

#: Channel of each T/E block ``(left stokes, right stokes)``, exactly as
#: :meth:`cmbcov.covariance.Cov._compute_norm_xi` normalises it.  A pair with
#: a B leg is absent: such blocks are not corrected.
BLOCK_CHANNEL: dict[tuple[str, str], str] = {
    ("TT", "TT"): "00",
    ("TT", "EE"): "20",
    ("TT", "TE"): "20",
    ("TT", "ET"): "20",
    ("EE", "TT"): "20",
    ("EE", "EE"): "EE",
    ("EE", "TE"): "EE",
    ("EE", "ET"): "EE",
    ("TE", "TT"): "20",
    ("TE", "EE"): "EE",
    ("TE", "TE"): "20",
    ("TE", "ET"): "20",
    ("ET", "TT"): "20",
    ("ET", "EE"): "EE",
    ("ET", "TE"): "20",
    ("ET", "ET"): "20",
}


def _stokes_pair(pair: str | Sequence[str]) -> tuple[str, str]:
    """``"TTxEE"`` or ``("TT", "EE")`` as ``("TT", "EE")``."""
    if isinstance(pair, str):
        parts = pair.split("x")
    else:
        parts = list(pair)
    if len(parts) != 2 or not all(isinstance(p, str) for p in parts):
        raise ValueError(
            f"a stokes pair is 'S1xS2' or (S1, S2), e.g. 'TTxEE'; got {pair!r}"
        )
    return parts[0], parts[1]


def channels_for(stokes_pairs) -> tuple[str, ...]:
    """
    The sum-rule channels the blocks ``stokes_pairs`` need.

    Parameters
    ----------
    stokes_pairs : iterable
        Block stokes pairs, as ``"TTxEE"`` strings
        (:meth:`cmbcov.keys.CovKeys.stokekey`) or ``("TT", "EE")`` tuples.
        Pairs with a B leg need no channel (they are not corrected).

    Returns
    -------
    tuple of str
        The distinct channels of :data:`BLOCK_CHANNEL`, in the order of
        :data:`CHANNELS`; empty if no block is a T/E block.
    """
    needed = set()
    for pair in stokes_pairs:
        channel = BLOCK_CHANNEL.get(_stokes_pair(pair))
        if channel is not None:
            needed.add(channel)
    return tuple(ch for ch in CHANNELS if ch in needed)


# --------------------------------------------------------------------------- #
# Ring filters
# --------------------------------------------------------------------------- #
def highpass_profile(
    lmax_grid: int, lx: float, shape: str = "sharp", power: float = 6.0
) -> np.ndarray:
    r"""
    High-pass ring-filter profile :math:`h(\theta_j, M)` on a GL grid.

    The cut follows the scan: a ring at colatitude :math:`\theta` sees the
    flat-sky wavenumber :math:`\ell_x` along the scan at the azimuthal order
    :math:`m_c(\theta) = \ell_x \sin\theta`.

    Parameters
    ----------
    lmax_grid : int
        GL grid band-limit (:func:`~cmbcov.grid.gl_shape`).
    lx : float
        Cut wavenumber :math:`\ell_x \ge 0`.  ``lx = 0`` means *no filter*:
        ``h = 1`` everywhere, ``M = 0`` included, for both shapes.
    shape : {"sharp", "exp"}
        ``"sharp"``: :math:`h = 1` where :math:`M > m_c(\theta)`, else 0 (so
        ``M = 0`` is always removed when ``lx > 0``).  ``"exp"``:
        :math:`h = \exp[-(m_c(\theta)/M)^p]` for ``M > 0`` and 0 at
        ``M = 0``.
    power : float
        Exponent ``p`` of the ``"exp"`` shape (positive; ignored for
        ``"sharp"``).

    Returns
    -------
    ndarray
        float64 array of shape ``(ntheta, nphi // 2 + 1)``, row ``j`` for the
        ring :func:`~cmbcov.grid.gl_thetas` ``(lmax_grid)[j]``, column ``M``.
    """
    if isinstance(lmax_grid, bool) or int(lmax_grid) != lmax_grid or lmax_grid < 0:
        raise ValueError(f"lmax_grid must be a non-negative integer, got {lmax_grid}")
    lmax_grid = int(lmax_grid)
    lx = float(lx)
    if not np.isfinite(lx) or lx < 0:
        raise ValueError(f"lx must be finite and non-negative, got {lx}")
    if shape not in ("sharp", "exp"):
        raise ValueError(f"shape must be 'sharp' or 'exp', got {shape!r}")
    power = float(power)
    if not np.isfinite(power) or power <= 0:
        raise ValueError(f"power must be finite and positive, got {power}")

    ntheta, nphi = gl_shape(lmax_grid)
    if lx == 0.0:
        return np.ones((ntheta, nphi // 2 + 1))
    mc = lx * np.sin(gl_thetas(lmax_grid))[:, None]
    M = np.arange(nphi // 2 + 1, dtype=float)[None, :]
    if shape == "sharp":
        return (M > mc).astype(float)
    h = np.zeros((ntheta, nphi // 2 + 1))
    h[:, 1:] = np.exp(-((mc / M[:, 1:]) ** power))
    return h


class FourierRingFilter:
    r"""
    Multiply the azimuthal mode ``M`` of every ring by ``h[theta_j, |M|]``.

    ``F = irfft(h * rfft(map))`` along :math:`\phi`, ring by ring.  With a
    real ``h`` this is a real symmetric circulant matrix on each ring, and
    since the GL quadrature weight is constant along a ring it is
    self-adjoint for the map inner product :math:`\sum w f g` as well:
    ``F^dagger = F``.  A (Q, U) pair is filtered map by map, which commutes
    with the spin-2 rotation :math:`Q + iU \to e^{i\psi}(Q + iU)`.

    Parameters
    ----------
    h : ndarray
        Real, finite array of shape ``(ntheta, nphi // 2 + 1)`` (e.g. from
        :func:`highpass_profile`).  If ``h`` is identically one the filter is
        the identity and returns a copy of its input without transforming it.
    """

    def __init__(self, h: np.ndarray):
        h = np.array(h, dtype=float, copy=True)
        if h.ndim != 2 or h.shape[1] < 1:
            raise ValueError(
                f"h must be a 2-D (ntheta, nphi//2+1) array, got {h.shape}"
            )
        if not np.all(np.isfinite(h)):
            raise ValueError("h must be finite")
        h.setflags(write=False)
        self.h = h
        self.identity = bool(np.all(h == 1.0))

    def __call__(self, map: np.ndarray) -> np.ndarray:
        """
        Filtered copy of the real map of shape ``(ntheta, nphi)``, or of each
        map of a ``(2, ntheta, nphi)`` (Q, U) pair separately.
        """
        map = np.asarray(map, dtype=float)
        if map.ndim not in (2, 3):
            raise ValueError(
                "map must be (ntheta, nphi) or (ncomp, ntheta, nphi), got shape "
                f"{map.shape}"
            )
        nphi = map.shape[-1]
        if map.shape[-2] != self.h.shape[0] or nphi // 2 + 1 != self.h.shape[1]:
            raise ValueError(
                f"map shape {map.shape} does not match h shape {self.h.shape} "
                "(need ntheta equal and nphi // 2 + 1 == h.shape[1])"
            )
        if self.identity:
            return map.copy()
        return np.fft.irfft(np.fft.rfft(map, axis=-1) * self.h, n=nphi, axis=-1)


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
def default_nodes(lmax: int, node_min: int = 32, ratio: float = 1.25) -> list[int]:
    r"""
    Multipoles at which :math:`\rho` is computed: geometric, ending at
    ``lmax - 1``.

    Parameters
    ----------
    lmax : int
        Size of the covariance blocks (``l = 0 .. lmax - 1``); at least 2.
    node_min : int
        First node, clipped to ``lmax - 1``; at least 1 (nodes enter a
        logarithm in :func:`rho_matrix`).
    ratio : float
        Ratio between successive nodes, ``> 1``.  PCHIP in
        :math:`\ln\ell` between nodes a factor 1.3-1.7 apart added about
        0.1% or less above three times the cut in the research
        measurements; 1.25 leaves headroom.

    Returns
    -------
    list of int
        Strictly increasing, starting at ``min(node_min, lmax - 1)`` and
        ending with ``lmax - 1``.  Consecutive values are rounded from
        ``node_min * ratio**k`` and forced to increase by at least one; a
        value closer to ``lmax - 1`` than a factor ``sqrt(ratio)`` is dropped
        so that the last interval is not a sliver.
    """
    lmax = int(lmax)
    node_min = int(node_min)
    ratio = float(ratio)
    if lmax < 2:
        raise ValueError(f"lmax must be at least 2, got {lmax}")
    if node_min < 1:
        raise ValueError(f"node_min must be at least 1, got {node_min}")
    if not np.isfinite(ratio) or ratio <= 1.0:
        raise ValueError(f"ratio must be finite and > 1, got {ratio}")
    last = lmax - 1
    first = min(node_min, last)
    nodes = [first]
    x = float(first)
    while True:
        x *= ratio
        nxt = max(int(round(x)), nodes[-1] + 1)
        if nxt * np.sqrt(ratio) > last:
            break
        nodes.append(nxt)
        x = float(nxt)
    if nodes[-1] != last:
        nodes.append(last)
    return nodes


# --------------------------------------------------------------------------- #
# The filtered sum rule by probing
# --------------------------------------------------------------------------- #
def _truncate_alm(alm: np.ndarray, lmax_in: int, lmax_out: int) -> np.ndarray:
    """healpy alm band-limited at ``lmax_in`` cut to ``lmax_out <= lmax_in``.

    healpy's layout is m-major with ``l`` ascending inside each ``m``, so the
    entries with ``l <= lmax_out`` are, in their original order, exactly the
    layout of band-limit ``lmax_out``."""
    import healpy as hp  # lazy, see utils/healpy_utils.py

    if lmax_out >= lmax_in:
        return alm
    ell = hp.Alm.getlm(lmax_in)[0]
    return np.ascontiguousarray(alm[ell <= lmax_out])


def _probe_alms(ellp: int, lmax_int: int, nprobe: int, seed: int):
    """
    Probes of degree ``ellp`` as healpy alm arrays at band-limit ``lmax_int``.

    ``2 ellp + 1 <= nprobe``: the complete orthonormal real basis of the
    degree (1 at ``m = 0``; ``1/sqrt 2`` and ``i/sqrt 2`` at each ``m > 0``),
    returned as ``(True, iterator)``.  Otherwise ``nprobe`` unit-variance
    Gaussian fields (``N(0,1)`` at ``m = 0``, ``(N + iN)/sqrt 2`` at
    ``m > 0``) from ``np.random.default_rng([seed, ellp])``.
    """
    import healpy as hp  # lazy, see utils/healpy_utils.py

    nalm = hp.Alm.getsize(lmax_int)
    idx = hp.Alm.getidx(lmax_int, ellp, np.arange(ellp + 1))
    exact = 2 * ellp + 1 <= nprobe

    def gen():
        if exact:
            z = np.zeros(nalm, dtype=complex)
            z[idx[0]] = 1.0
            yield z
            for k in range(1, ellp + 1):
                for value in (1.0, 1.0j):
                    z = np.zeros(nalm, dtype=complex)
                    z[idx[k]] = value / np.sqrt(2.0)
                    yield z
            return
        rng = np.random.default_rng([int(seed), int(ellp)])
        for _ in range(nprobe):
            re = rng.standard_normal(ellp + 1)
            im = rng.standard_normal(ellp + 1)
            z = np.zeros(nalm, dtype=complex)
            z[idx[0]] = re[0]
            z[idx[1:]] = (re[1:] + 1j * im[1:]) / np.sqrt(2.0)
            yield z

    return exact, gen()


#: Fraction of a node's largest unfiltered sum below which a degree's
#: sum is treated as round-off (no mask coupling) and its ratio not used.
_RELIABLE_FRACTION = 1e-9


def _node_sums(
    w_map: np.ndarray,
    filt: FourierRingFilter,
    ellp: int,
    lmax_int: int,
    lg: int,
    channels: Sequence[str],
    nprobe: int,
    seed: int,
    nthreads: int | None,
) -> tuple[bool, int, dict[str, tuple[np.ndarray, np.ndarray]]]:
    r"""
    Probe sums of one node: ``(exact, count, sums)`` with
    ``sums[ch] = (S^F, S)``, the filtered and unfiltered
    :math:`\sum_{\rm probes} \sum_m |y_{\ell m}|^2` of channel ``ch`` for
    ``l = 0 .. lmax_int`` (module docstring, "Polarisation channels").

    Every channel sees the same probes ``z``, and the filtered and unfiltered
    operators see the same ``z``.  The spin-2 channels share the synthesis of
    ``(z, 0)`` and the adjoint ``u``.  A spin-2 channel at ``ellp < 2`` has no
    E-mode probe: its sums are zero.
    """
    spin0 = "00" in channels
    spin2 = ("20" in channels or "EE" in channels) and ellp >= 2
    size = lmax_int + 1
    sums = {ch: (np.zeros(size), np.zeros(size)) for ch in channels}

    def synth(alm, spin=0):
        return gl_synthesis(alm, lmax_int, lg, spin=spin, nthreads=nthreads)

    def anal(f, spin=0):
        return gl_analysis(f, lmax_int, lg, spin=spin, nthreads=nthreads)

    def no_filter(f):
        return f

    exact, probes = _probe_alms(ellp, lmax_int, nprobe, seed)
    count = 0
    for z in probes:
        count += 1
        if spin0:
            wz = w_map * synth(z)  # W S z, shared
            # unfiltered: A W S A W S z
            y_u = anal(w_map * synth(anal(wz)))
            # filtered: K K^dagger z = A W F S A F W S z
            y_f = anal(w_map * filt(synth(anal(filt(wz)))))
            sums["00"][1][:] += alm2cl(y_u)
            sums["00"][0][:] += alm2cl(y_f)
        if spin2:
            zero = np.zeros_like(z)
            wz2 = w_map * synth(np.stack([z, zero]), 2)  # W S_2 (z, 0), shared
            for index, apply in ((0, filt), (1, no_filter)):
                # u = [K_2^dagger (z, 0)]_E = [A_2 F W S_2 (z, 0)]_E
                u = anal(apply(wz2), 2)[0]
                if "20" in channels:
                    # y = K_0 u = A_0 W F S_0 u
                    sums["20"][index][:] += alm2cl(anal(w_map * apply(synth(u))))
                if "EE" in channels:
                    # y = [K_2 (u, 0)]_E = [A_2 W F S_2 (u, 0)]_E
                    y = anal(w_map * apply(synth(np.stack([u, zero]), 2)), 2)[0]
                    sums["EE"][index][:] += alm2cl(y)
    # sum_m |y_lm|^2 = (2l+1) C_l; the factor cancels in the ratio but is
    # kept so that the sums are the ones of the docstring.
    n = 2 * np.arange(size) + 1.0
    for s_f, s_u in sums.values():
        s_f *= n
        s_u *= n
    return exact, count, sums


def _ratio_by_distance(
    sum_f: np.ndarray, sum_u: np.ndarray, ellp: int, dmax: int
) -> np.ndarray:
    r"""
    ``g[d]``, ``d = 0 .. dmax``, of one node and channel from its sums (see
    :func:`filtered_sum_rule_ratio`, "Returns").
    """
    # A degree whose unfiltered sum is at the round-off floor of the node
    # (the mask couples nothing there: beyond its coupling range the sums
    # are exact zeros or rounding noise) carries no information; its ratio
    # would be noise over noise.  Such degrees are not used: the value of
    # the nearest smaller reliable ``d`` is held instead, so that ``g``
    # stays a smooth function of ``d``.  A channel with no probe at all
    # (spin 2 at ``l' < 2``) has no reliable degree and gives 1.
    reliable = sum_u > _RELIABLE_FRACTION * sum_u.max()
    ratio = np.ones(sum_u.size)
    ratio[reliable] = sum_f[reliable] / sum_u[reliable]

    g = np.empty(dmax + 1)
    g[0] = ratio[ellp]
    for d in range(1, dmax + 1):
        hi = ellp + d
        lo = ellp - d
        if lo >= 0 and reliable[lo] and reliable[hi]:
            g[d] = 0.5 * (ratio[hi] + ratio[lo])
        elif reliable[hi]:
            g[d] = ratio[hi]
        elif lo >= 0 and reliable[lo]:
            g[d] = ratio[lo]
        else:
            g[d] = g[d - 1]
    return g


def _check_channels(channels: Sequence[str]) -> tuple[str, ...]:
    """``channels`` validated, duplicates dropped, in the order given."""
    if isinstance(channels, str):
        channels = (channels,)
    channels = tuple(dict.fromkeys(channels))
    if len(channels) == 0:
        raise ValueError("channels must name at least one channel")
    unknown = [ch for ch in channels if ch not in CHANNELS]
    if unknown:
        raise ValueError(f"unknown channel(s) {unknown}; known: {CHANNELS}")
    return channels


def filtered_sum_rule_ratio(
    mask_alm: np.ndarray,
    lw: int,
    profile: Callable[[int], np.ndarray],
    nodes: Sequence[int],
    dmax: int,
    *,
    channels: Sequence[str] = ("00",),
    nprobe: int = 32,
    seed: int = 0,
    margin: int | None = None,
    nthreads: int | None = None,
    verbose: bool = False,
) -> dict[str, np.ndarray]:
    r"""
    :math:`\rho^{ch} = \Xi^{F,ch} / \Xi^{ch}` at the nodes, by paired probing.

    For each node :math:`\ell'` the probes ``z`` of degree :math:`\ell'` are
    pushed through the operator of each channel (module docstring,
    "Polarisation channels"; ``"00"`` is :math:`K K^\dagger`) with and
    without the filter, and :math:`S^F_\ell = \sum_{\rm probes} \sum_m
    |y_{\ell m}|^2` and :math:`S_\ell` (unfiltered) are accumulated for every
    :math:`\ell`; :math:`\rho^{ch}(\ell,\ell') = S^F_\ell / S_\ell` (module
    docstring, "The probing identity").

    Parameters
    ----------
    mask_alm : ndarray
        healpy alm of the analysis mask truncated at ``lw``, **in the scan
        frame** (module docstring).
    lw : int
        Band-limit of ``mask_alm``.
    profile : callable
        ``profile(lmax_grid) -> h`` of shape ``(ntheta, nphi // 2 + 1)`` on
        the GL grid of band-limit ``lmax_grid``, e.g.
        ``functools.partial(highpass_profile, lx=300, shape="sharp")``.  It
        is called once per node, on that node's grid.
    nodes : sequence of int
        Non-negative multipoles :math:`\ell'`, in any order (row ``i`` of the
        result is ``nodes[i]``).
    dmax : int
        Largest distance from the diagonal returned.
    channels : sequence of str
        Channels to compute, among :data:`CHANNELS` (``"00"``, ``"20"``,
        ``"EE"``); :func:`channels_for` gives those a set of blocks needs.
        The default is the TT x TT channel alone.
    nprobe : int
        Number of random probes per node.  A node with ``2 l' + 1 <= nprobe``
        uses the ``2 l' + 1`` vectors of the complete orthonormal real basis
        of its degree instead, which makes its ratio exact.
    seed : int
        Seed of the random probes; node ``l'`` draws from
        ``np.random.default_rng([seed, l'])``, so its result does not depend
        on which other nodes are requested.  All channels of a node use the
        same probes, so a channel's result does not depend on which other
        channels are requested either.
    margin : int, optional
        Internal band-limit headroom: node ``l'`` carries every alm to
        ``lmax_int = l' + dmax + margin``.  ``None`` uses
        :func:`~cmbcov.exact.mask_coupling_width` of the mask's ``W_L``, the
        same headroom rule as the exact covariance.
    nthreads : int, optional
        Threads for the ducc0 transforms.
    verbose : bool
        Print one line per node (grid, probes, time).

    Returns
    -------
    dict
        ``{channel: g}`` for each requested channel, ``g`` a float64 array of
        shape ``(len(nodes), dmax + 1)``: ``g[i, d]`` is :math:`\rho^{ch}` at
        mean multipole ``nodes[i]`` and distance ``d``, the average of
        :math:`\rho(\ell'+d, \ell')` and :math:`\rho(\ell'-d, \ell')` (only
        the first when ``l' - d < 0``; ``d = 0`` is the single diagonal
        estimate).  A degree whose unfiltered sum is below
        :data:`_RELIABLE_FRACTION` of the node's largest one for that
        channel (no mask coupling, rounding noise) is not used: the value of
        the previous ``d`` is held there.  A spin-2 channel at a node
        ``l' < 2`` has no E-mode probe and is 1.

    Notes
    -----
    Per node the mask is truncated to ``lw_node = min(lw, 2 lmax_int)``,
    which is lossless for the unfiltered operator (a mask multipole above
    ``2 lmax_int`` cannot couple two degrees ``<= lmax_int``), and
    synthesised once on the grid :func:`~cmbcov.grid.gl_minimal_lmax`
    ``(lmax_int, lw_node)``, on which the unfiltered operator is exact (spin
    0 and spin 2 alike).  A :math:`\theta`-dependent cut is not
    band-limited: the filtered operator is *defined* by that grid's rings.

    Cost per probe: ``"00"`` seven spin-0 GL transforms at ``lmax_int`` (one
    synthesis of the probe shared by both operators, then analysis,
    synthesis and analysis for each) and two ring FFT filters.  The spin-2
    channels share one spin-2 synthesis of ``(z, 0)`` and one spin-2
    analysis per operator; then ``"20"`` adds a spin-0 synthesis and
    analysis per operator and ``"EE"`` a spin-2 synthesis and analysis.  A
    spin-2 transform costs two to three spin-0 ones; the three channels
    together took 3.8 times ``"00"`` alone at ``l' = 200`` (one thread,
    ``lw = 64``) and 6 times in the research runs.
    """
    import healpy as hp  # lazy, see utils/healpy_utils.py

    lw = int(lw)
    if lw < 0:
        raise ValueError(f"lw must be non-negative, got {lw}")
    mask_alm = np.ascontiguousarray(mask_alm, dtype=np.complex128)
    if mask_alm.ndim != 1 or mask_alm.size != hp.Alm.getsize(lw):
        raise ValueError(
            f"mask_alm must be a 1-D healpy alm of band-limit lw={lw} "
            f"(size {hp.Alm.getsize(lw)}); got shape {mask_alm.shape}"
        )
    if not callable(profile):
        raise TypeError("profile must be a callable lmax_grid -> h")
    channels = _check_channels(channels)
    nodes = [int(n) for n in nodes]
    if len(nodes) == 0 or min(nodes) < 0:
        raise ValueError(
            f"nodes must be a non-empty list of non-negative ints: {nodes}"
        )
    dmax = int(dmax)
    if dmax < 0:
        raise ValueError(f"dmax must be non-negative, got {dmax}")
    nprobe = int(nprobe)
    if nprobe < 1:
        raise ValueError(f"nprobe must be at least 1, got {nprobe}")
    if margin is None:
        margin = mask_coupling_width(alm2cl(mask_alm))
    margin = int(margin)
    if margin < 0:
        raise ValueError(f"margin must be non-negative, got {margin}")

    g = {ch: np.empty((len(nodes), dmax + 1)) for ch in channels}
    for i, ellp in enumerate(nodes):
        t0 = time.perf_counter()
        lmax_int = ellp + dmax + margin
        lw_node = min(lw, 2 * lmax_int)
        lg = gl_minimal_lmax(lmax_int, lw_node)
        w_map = gl_synthesis(
            _truncate_alm(mask_alm, lw, lw_node), lw_node, lg, nthreads=nthreads
        )
        filt = FourierRingFilter(profile(lg))
        if filt.h.shape != (gl_shape(lg)[0], gl_shape(lg)[1] // 2 + 1):
            raise ValueError(
                f"profile({lg}) returned shape {filt.h.shape}; expected "
                f"{(gl_shape(lg)[0], gl_shape(lg)[1] // 2 + 1)}"
            )
        exact, count, sums = _node_sums(
            w_map, filt, ellp, lmax_int, lg, channels, nprobe, seed, nthreads
        )
        for ch in channels:
            g[ch][i] = _ratio_by_distance(*sums[ch], ellp, dmax)
        if verbose:
            rho0 = " ".join(f"rho{ch}(d=0)={g[ch][i, 0]:.4f}" for ch in channels)
            print(
                f"filtered_sum_rule_ratio: node {ellp} lmax_int {lmax_int} "
                f"grid {lg} lw {lw_node} {'exact basis' if exact else 'random'} "
                f"{count} probes {time.perf_counter() - t0:.2f} s {rho0}",
                flush=True,
            )
    return g


# --------------------------------------------------------------------------- #
# Interpolation and application
# --------------------------------------------------------------------------- #
def rho_matrix(nodes: Sequence[int], g: np.ndarray, lmax: int) -> np.ndarray:
    r"""
    :math:`\rho(\ell_1, \ell_2)` on the full ``(lmax, lmax)`` block.

    For :math:`d = |\ell_1 - \ell_2|` and :math:`\bar\ell = (\ell_1 +
    \ell_2)/2` the value is the PCHIP interpolant (scipy) of ``g[:, d]``
    against :math:`\ln` ``nodes``, evaluated at :math:`\ln\bar\ell`; it is
    held constant at ``g[0, d]`` below the first node and at ``g[-1, d]``
    above the last one.  For ``d > dmax`` the ``d = dmax`` column is used.
    With a single node :math:`\rho` depends on ``d`` only.  Being symmetric,
    it serves a block and its transpose alike (for the ``"20"`` channel the
    probe's T-left / E-right orientation is thereby symmetrised).

    Parameters
    ----------
    nodes : sequence of int
        Strictly increasing positive multipoles (row ``i`` of ``g``).
    g : ndarray
        One channel of the output of :func:`filtered_sum_rule_ratio`, shape
        ``(len(nodes), dmax + 1)``.
    lmax : int
        Block size; ``l = 0 .. lmax - 1``.

    Returns
    -------
    ndarray
        Symmetric float64 array of shape ``(lmax, lmax)``.

    Notes
    -----
    :math:`\bar\ell` only takes the ``2 lmax - 1`` half-integer values
    ``k / 2``, so the interpolants are evaluated once on that set for all
    ``dmax + 1`` columns and the matrix is filled row by row by indexing,
    ``rho[l1, l2] = T[l1 + l2, min(|l1 - l2|, dmax)]``.
    """
    from scipy.interpolate import PchipInterpolator

    nodes_arr = np.asarray(nodes, dtype=float)
    g = np.asarray(g, dtype=float)
    lmax = int(lmax)
    if lmax < 1:
        raise ValueError(f"lmax must be at least 1, got {lmax}")
    if nodes_arr.ndim != 1 or nodes_arr.size == 0:
        raise ValueError("nodes must be a non-empty 1-D sequence")
    if np.any(nodes_arr <= 0) or np.any(np.diff(nodes_arr) <= 0):
        raise ValueError(f"nodes must be strictly increasing and positive: {nodes}")
    if g.ndim != 2 or g.shape[0] != nodes_arr.size or g.shape[1] < 1:
        raise ValueError(
            f"g must have shape (len(nodes), dmax + 1) = ({nodes_arr.size}, ...); "
            f"got {g.shape}"
        )
    if not np.all(np.isfinite(g)):
        raise ValueError("g must be finite")
    dmax = g.shape[1] - 1

    lbar = 0.5 * np.arange(2 * lmax - 1)
    if nodes_arr.size == 1:
        table = np.broadcast_to(g[0], (lbar.size, dmax + 1))
    else:
        x = np.log(np.clip(lbar, nodes_arr[0], nodes_arr[-1]))
        table = PchipInterpolator(np.log(nodes_arr), g, axis=0)(x)

    rho = np.empty((lmax, lmax))
    cols = np.arange(lmax)
    for l1 in range(lmax):
        rho[l1] = table[l1 + cols, np.minimum(np.abs(l1 - cols), dmax)]
    return rho


def transfer_correction(
    rho: np.ndarray, fl_left: np.ndarray, fl_right: np.ndarray
) -> np.ndarray:
    r"""
    The post-processing factor :math:`\rho(\ell_1,\ell_2) / (F_{\ell_1}
    F_{\ell_2})`.

    A block computed with the unfiltered kernels and the spectrum
    :math:`F_L C_L` times this factor is the filtered covariance model of the
    module docstring.

    Parameters
    ----------
    rho : ndarray
        Array of shape ``(n1, n2)``, e.g. from :func:`rho_matrix`.
    fl_left, fl_right : ndarray
        Transfer functions of the two legs, at least ``n1`` resp. ``n2``
        long (sliced to that size).  Where :math:`F_\ell \le 0` or is not
        finite the division factor is set to 1, i.e. the block is left as
        :math:`\rho` there.

    Returns
    -------
    ndarray
        New float64 array of the shape of ``rho``.
    """
    rho = np.asarray(rho, dtype=float)
    if rho.ndim != 2:
        raise ValueError(f"rho must be 2-D, got shape {rho.shape}")
    n1, n2 = rho.shape

    def factor(fl, n, name):
        fl = np.asarray(fl, dtype=float)
        if fl.ndim != 1 or fl.size < n:
            raise ValueError(f"{name} must be 1-D with at least {n} entries")
        fl = fl[:n]
        return np.where(np.isfinite(fl) & (fl > 0), fl, 1.0)

    left = factor(fl_left, n1, "fl_left")
    right = factor(fl_right, n2, "fl_right")
    return rho / (left[:, None] * right[None, :])


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
def _json_default(obj):
    """numpy scalars and arrays as their Python equivalents in the identity."""
    if isinstance(obj, np.generic | np.ndarray):
        return obj.tolist()
    raise TypeError(f"identity value of type {type(obj).__name__} is not JSON")


def cached_sum_rule_ratio(
    cache_dir: str | os.PathLike,
    identity: dict,
    compute: Callable[[], Mapping[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    r"""
    :func:`filtered_sum_rule_ratio` output ``{channel: g}``, cached on disk.

    Parameters
    ----------
    cache_dir : str or path
        Root cache directory; files go to ``<cache_dir>/filtered_sum_rule/``
        (created as needed).
    identity : dict
        Everything the result depends on (mask digest, filter parameters,
        channels, nodes, dmax, nprobe, seed, lw, ...), JSON-serialisable
        (numpy scalars and arrays are converted with ``tolist``).  Its
        canonical form is ``json.dumps(identity, sort_keys=True)``.
    compute : callable
        ``compute() -> {channel: g}`` (non-empty, each ``g`` 2-D), called
        only on a miss.

    Returns
    -------
    dict
        The stored ``{channel: g}`` if ``rho_<hash>.npz`` exists and holds an
        identical identity string, otherwise the freshly computed one, which
        is then written (temporary file and ``os.replace``, so a crash never
        leaves a partial file under the final name).  ``<hash>`` is the
        16-hex-digit blake2b digest of the canonical identity.  An unreadable
        file or a mismatching identity (hash collision) is treated as a miss
        and overwritten.

    Notes
    -----
    File layout: ``identity`` (the canonical string), ``channels`` (the
    channel names, in the order ``compute`` returned them) and one array
    ``g_<channel>`` per channel.  This replaced the TT-only layout (a single
    ``g``); a file of that older layout lacks ``channels`` and is simply a
    miss, recomputed and overwritten.
    """
    key = json.dumps(identity, sort_keys=True, default=_json_default)
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).hexdigest()
    directory = os.path.join(os.fspath(cache_dir), CACHE_SUBDIR)
    path = os.path.join(directory, f"rho_{digest}.npz")

    if os.path.exists(path):
        try:
            with np.load(path, allow_pickle=False) as data:
                if str(data["identity"]) == key:
                    return {
                        str(ch): np.array(data[f"g_{ch}"], dtype=float)
                        for ch in data["channels"]
                    }
        except (OSError, ValueError, KeyError):
            pass

    result = compute()
    if not isinstance(result, Mapping) or len(result) == 0:
        raise ValueError("compute() must return a non-empty {channel: g} mapping")
    g = {str(ch): np.asarray(values, dtype=float) for ch, values in result.items()}
    for ch, values in g.items():
        if values.ndim != 2:
            raise ValueError(
                f"compute() must return 2-D arrays, got shape {values.shape} "
                f"for channel {ch!r}"
            )
    os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        np.savez(
            f,
            identity=np.array(key),
            channels=np.array(list(g)),
            **{f"g_{ch}": values for ch, values in g.items()},
        )
    os.replace(tmp, path)
    return g
