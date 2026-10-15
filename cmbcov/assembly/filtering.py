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
(:mod:`cmbcov.harmonics.grid`), ``W`` the multiplication by the mask and ``F`` real and
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

A scan-direction *low-pass* (:func:`lowpass_profile`, cut at its own
:math:`\ell_x` above the high-pass one, same :math:`m_c = \ell_x \sin\theta`
convention) is the same kind of ring filter: the filter of a run is the
product :math:`h = h_{\mathrm{high}} h_{\mathrm{low}}`
(:func:`combined_profile`), and everything below (the probes, the channels,
:math:`\rho`, the correction) is unchanged.  Only the nodes differ, through
its rolloff (:func:`default_nodes`).

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
the noise; the error of :math:`\rho` falls roughly as
:math:`1/\sqrt{N \ell'}` for :math:`N` probes.  When :math:`2\ell'+1` does not exceed the number
of probes the complete orthonormal real basis of the degree is used instead
and the ratio is exact (:func:`filtered_sum_rule_ratio`).

Polarisation channels
---------------------
Every T/E covariance block is normalised by one of three sum-rule channels
(the table of :meth:`cmbcov.assembly.covariance.Cov._compute_norm_xi`, reproduced in
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
with only :math:`C^{EE} = 1` for ``"EE"`` (to round-off, checked in
``tests/assembly/test_filtering.py``).  The ``"20"`` probe puts T on the left and E
on the right; its transpose agrees with it closely, and :func:`rho_matrix`
is symmetric anyway.  A spin-2 transform costs two to three spin-0 ones, so
only the channels a run needs are computed (:func:`channels_for`).  Blocks
with a B leg are not corrected: see docs/theory/filter_and_bin.md.

Spectrum-aware TT
-----------------
:math:`\rho` describes a white spectrum.  A sharp :math:`\theta`-dependent
cut also scatters power across multipoles (low-:math:`\ell` signal to high
:math:`\ell`), which matters for a steep spectrum such as the beam-damped TT
signal and which no white ratio can carry.  :func:`spectrum_aware_ratio`
probes the TT x TT blocks with the fiducial spectra instead: per block the
ratio :math:`g_C` of the exact filtered covariance (spectra before the
filter, :math:`C_{pre}`) to the exact unfiltered one (the spectra the
covariance is computed with, :math:`C_{used}`, which contain the transfer
function), by the same paired probing.  The block times :math:`g_C` is the
filtered block, without any division by :math:`F_\ell`.

Why :math:`\rho` and not :math:`F_\ell`
---------------------------------------
The filter removes modes whose wavevector points across the scan.  Their
weight in the diagonal :math:`d = \ell - \ell'` of the covariance is set by
the extent of the mask's :math:`|\widetilde{W^2}|^2` along the tangent of
the annulus, i.e. by the *shape* of the mask, and it changes with
:math:`d`.  Masks with the same :math:`F_\ell` but different shapes (wide
along the scan, a cap, elongated across the scan) therefore have different
:math:`\rho(\ell,\ell')`, which also varies with :math:`d` at fixed
:math:`\ell`.  No function of :math:`F_\ell` alone reproduces this, neither
:math:`C \to F C` nor a :math:`1/\sqrt{F_\ell F_{\ell'}}` rescaling of it.
:math:`\rho` is smooth in :math:`\ell`, so it is computed at a few nodes
(:func:`default_nodes`) and interpolated (:func:`rho_matrix`).

The scan frame
--------------
The filter fixes the frame: the rings of the map's coordinate system are the
scan lines, so the mask alm must be given in that frame (its own polar axis
is the scan axis).  In particular the pole-frame rotation of
:mod:`cmbcov.approximations.acc.term_selection` must **not** be applied to the mask passed here.
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
a few per cent at that toy scale; those multipoles have :math:`F_\ell <
0.7`.  The kernel-shape error falls fast with the cut measured in mask
coupling widths; on a survey-sized footprint the range near the cut is set
by the node spacing, the probe noise and the probes' band-limit there,
which :func:`node_plan` treats (docs/theory/filter_and_bin.md, "Near the
cut").  T and E: with
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
from typing import NamedTuple

import numpy as np

from ..exact.common import mask_coupling_width
from ..harmonics.grid import (
    gl_analysis,
    gl_minimal_lmax,
    gl_shape,
    gl_synthesis,
    gl_thetas,
)
from ..harmonics.sht import alm2cl
from ..utils.progress import Progress

__all__ = [
    "CHANNELS",
    "BLOCK_CHANNEL",
    "channels_for",
    "highpass_profile",
    "lowpass_profile",
    "combined_profile",
    "FourierRingFilter",
    "default_nodes",
    "NodePlan",
    "node_plan",
    "FL_DEGRADED",
    "FL_CUT",
    "bin_transfer_minimum",
    "filtered_sum_rule_ratio",
    "NOISE_FL_FLOOR",
    "DEFAULT_OVERSAMPLE",
    "pre_filter_noise",
    "block_id",
    "spectrum_aware_ratio",
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
#: :meth:`cmbcov.assembly.covariance.Cov._compute_norm_xi` normalises it.  A pair with
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
        (:meth:`cmbcov.assembly.keys.CovKey.stokekey`) or ``("TT", "EE")`` tuples.
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
        GL grid band-limit (:func:`~cmbcov.harmonics.grid.gl_shape`).
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
        ring :func:`~cmbcov.harmonics.grid.gl_thetas` ``(lmax_grid)[j]``, column ``M``.
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


def lowpass_profile(
    lmax_grid: int, lx: float, shape: str = "exp", power: float = 6.0
) -> np.ndarray:
    r"""
    Low-pass ring-filter profile :math:`h(\theta_j, M)` on a GL grid.

    The same family as :func:`highpass_profile`, with the cut at the same
    azimuthal order :math:`m_c(\theta) = \ell_x \sin\theta` for the
    wavenumber :math:`\ell_x` along the scan, but removing the modes
    *above* it (the scan-direction low-pass of a time-stream filter; the
    product with a high-pass is the filter of a band).

    Parameters
    ----------
    lmax_grid : int
        GL grid band-limit (:func:`~cmbcov.harmonics.grid.gl_shape`).
    lx : float
        Cut wavenumber :math:`\ell_x`, finite and positive (a low-pass has no
        "off" value; leave it out instead).
    shape : {"exp", "sharp"}
        ``"exp"``: :math:`h = \exp[-(M / m_c(\theta))^p]` (1 at ``M = 0``).
        ``"sharp"``: :math:`h = 1` where :math:`M \le m_c(\theta)`, else 0.
    power : float
        Exponent ``p`` of the ``"exp"`` shape (positive; ignored for
        ``"sharp"``).

    Returns
    -------
    ndarray
        float64 array of shape ``(ntheta, nphi // 2 + 1)``, row ``j`` for the
        ring :func:`~cmbcov.harmonics.grid.gl_thetas` ``(lmax_grid)[j]``,
        column ``M``.
    """
    if isinstance(lmax_grid, bool) or int(lmax_grid) != lmax_grid or lmax_grid < 0:
        raise ValueError(f"lmax_grid must be a non-negative integer, got {lmax_grid}")
    lmax_grid = int(lmax_grid)
    lx = float(lx)
    if not np.isfinite(lx) or lx <= 0:
        raise ValueError(f"lx must be finite and positive, got {lx}")
    if shape not in ("sharp", "exp"):
        raise ValueError(f"shape must be 'sharp' or 'exp', got {shape!r}")
    power = float(power)
    if not np.isfinite(power) or power <= 0:
        raise ValueError(f"power must be finite and positive, got {power}")

    ntheta, nphi = gl_shape(lmax_grid)
    mc = lx * np.sin(gl_thetas(lmax_grid))[:, None]  # > 0 on every GL ring
    M = np.arange(nphi // 2 + 1, dtype=float)[None, :]
    if shape == "sharp":
        return (M <= mc).astype(float)
    with np.errstate(over="ignore"):
        return np.exp(-((M / mc) ** power))


def combined_profile(
    lmax_grid: int,
    lx: float,
    shape: str = "sharp",
    power: float = 6.0,
    lowpass: Mapping | None = None,
) -> np.ndarray:
    r"""
    The ring-filter profile of a ``map_filter`` run: the high-pass
    :func:`highpass_profile` ``(lmax_grid, lx, shape, power)``, times the
    low-pass :func:`lowpass_profile` if ``lowpass`` is given,
    :math:`h = h_{\mathrm{high}} h_{\mathrm{low}}`.

    Parameters
    ----------
    lmax_grid, lx, shape, power :
        As for :func:`highpass_profile`.
    lowpass : mapping, optional
        ``{"lx": ..., "shape": ..., "power": ...}`` with ``lx`` required and
        ``shape`` (``"exp"``) and ``power`` (6; ``None`` also means 6, the
        form the cache identity records for ``"sharp"``) optional, as for
        :func:`lowpass_profile`; its ``lx`` must exceed the high-pass ``lx``
        (otherwise the band is empty).  ``None`` (the default): the
        high-pass alone, exactly :func:`highpass_profile`.
    """
    h = highpass_profile(lmax_grid, lx, shape, power)
    if lowpass is None:
        return h
    extra = sorted(set(lowpass) - {"lx", "shape", "power"})
    if extra or "lx" not in lowpass:
        raise ValueError(
            "lowpass must be a mapping with 'lx' and optionally 'shape' and "
            f"'power'; got {dict(lowpass)!r}"
        )
    if not float(lowpass["lx"]) > float(lx):
        raise ValueError(
            f"the low-pass lx ({lowpass['lx']}) must exceed the high-pass lx ({lx})"
        )
    power = lowpass.get("power")
    return h * lowpass_profile(
        lmax_grid,
        lowpass["lx"],
        lowpass.get("shape", "exp"),
        6.0 if power is None else power,
    )


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
#: Range of the dense nodes near the cut, in units of the cut ``lx``
#: (:func:`default_nodes`).
NEAR_CUT_RANGE = (0.95, 2.0)
#: Default spacing of the dense nodes near the cut: ``lx / NEAR_CUT_DIVISOR``
#: (rounded, at least 1).
NEAR_CUT_DIVISOR = 15
#: Nodes from ``round(0.95 lx)`` up to below ``NEAR_CUT_PROBE_LIMIT * lx``
#: use the near-cut probe count (:func:`node_plan`; by default the complete
#: basis of their degree).
NEAR_CUT_PROBE_LIMIT = 2.0
#: Nodes from ``round(0.95 lx)`` up to below ``BANDLIMIT_NODE_LIMIT * lx``
#: carry every alm to at least ``BANDLIMIT_FLOOR * lx`` (:func:`node_plan`).
BANDLIMIT_NODE_LIMIT = 1.3
#: Floor of the internal band-limit of the nodes near the cut, in units of
#: ``lx`` (:func:`node_plan`, Notes).
BANDLIMIT_FLOOR = 3.5
#: Range of the dense nodes through the rolloff of a low-pass, in units of
#: the low-pass cut ``lx_lowpass`` (:func:`default_nodes`); the nodes in it
#: (up to below the upper end) also get the near-cut probe count
#: (:func:`node_plan`).
LOWPASS_RANGE = (0.6, 2.0)
#: Spacing of the dense nodes in that range: ``lx_lowpass / LOWPASS_DIVISOR``.
LOWPASS_DIVISOR = 15
#: A *sharp* low-pass has a square-root onset at ``lx_lowpass`` (the fraction
#: of the modes of a degree that pass falls as ``sqrt(l - lx_lowpass)``): the
#: nodes from ``LOWPASS_ONSET_RANGE[0]`` to ``LOWPASS_ONSET_RANGE[1]`` times
#: ``lx_lowpass`` are spaced ``lx_lowpass / LOWPASS_ONSET_DIVISOR`` apart.
LOWPASS_ONSET_RANGE = (0.95, 1.15)
LOWPASS_ONSET_DIVISOR = 50


def _check_lowpass_cut(lx: float, lx_lowpass: float, shape: str) -> float:
    """``lx_lowpass`` as a float, checked: finite and above the high-pass
    ``lx``; ``shape`` one of the low-pass shapes."""
    lx_lowpass = float(lx_lowpass)
    if not np.isfinite(lx_lowpass) or lx_lowpass <= lx:
        raise ValueError(
            f"lx_lowpass must be finite and above lx ({lx}), got {lx_lowpass}"
        )
    if shape not in ("sharp", "exp"):
        raise ValueError(f"lowpass_shape must be 'sharp' or 'exp', got {shape!r}")
    return lx_lowpass


def _near_cut_step(lx: float, near_cut_step: float | None) -> int:
    """The spacing of the dense nodes: ``near_cut_step`` rounded, or
    ``lx / NEAR_CUT_DIVISOR`` rounded when ``None``; at least 1.  ``0``
    (no dense nodes) is handled by the caller."""
    step = lx / NEAR_CUT_DIVISOR if near_cut_step is None else float(near_cut_step)
    return max(1, _round_half_up(step))


def _round_half_up(x: float) -> int:
    """``x`` rounded to the nearest integer, halves up (``round`` rounds
    halves to even)."""
    return int(np.floor(x + 0.5))


def default_nodes(
    lmax: int,
    node_min: int = 32,
    ratio: float = 1.25,
    *,
    lx: float | None = None,
    near_cut_step: float | None = None,
    lx_lowpass: float | None = None,
    lowpass_shape: str = "exp",
) -> list[int]:
    r"""
    Multipoles at which :math:`\rho` is computed: geometric, ending at
    ``lmax - 1``, plus dense nodes near the cut ``lx`` if it is given, and
    through the rolloff of a low-pass at ``lx_lowpass`` if that is given.

    Parameters
    ----------
    lmax : int
        Size of the covariance blocks (``l = 0 .. lmax - 1``); at least 2.
    node_min : int
        First geometric node, clipped to ``lmax - 1``; at least 1 (nodes
        enter a logarithm in :func:`rho_matrix`).
    ratio : float
        Ratio between successive geometric nodes, ``> 1``.  PCHIP in
        :math:`\ln\ell` between nodes a factor 1.3-1.7 apart added about
        0.1% or less above three times the cut in the research
        measurements; 1.25 leaves headroom.
    lx : float, optional
        The filter cut.  ``None`` (the default): the geometric nodes alone.
    near_cut_step : float, optional
        Spacing of the dense nodes near the cut (rounded, at least 1);
        ``None`` means ``lx / 15`` (:data:`NEAR_CUT_DIVISOR`), ``0`` no dense
        nodes at all (the geometric list alone): neither those near the cut
        nor those through the low-pass rolloff, which is then interpolated
        from the geometric nodes alone (up to about 4% off for the
        exponential shape, more for the sharp one, at toy scale; the
        generator logs a warning).  Ignored without ``lx``.
    lx_lowpass : float, optional
        The cut of a scan-direction low-pass (:func:`lowpass_profile`); needs
        ``lx``.  ``None`` (the default): no low-pass nodes, and the list is
        exactly the one without this argument.
    lowpass_shape : {"exp", "sharp"}
        Shape of that low-pass; the ``"sharp"`` one gets extra nodes at its
        onset.

    Returns
    -------
    list of int
        Strictly increasing, ending with ``lmax - 1``.

        The geometric nodes start at ``min(node_min, lmax - 1)``; consecutive
        values are rounded from ``node_min * ratio**k`` and forced to
        increase by at least one; a value closer to ``lmax - 1`` than a
        factor ``sqrt(ratio)`` is dropped so that the last interval is not a
        sliver.

        With ``lx`` (and ``near_cut_step != 0``) the dense nodes
        ``round(0.95 lx) + k step`` up to ``2 lx`` (:data:`NEAR_CUT_RANGE`;
        rounding is half up)
        are added, those outside ``1 .. lmax - 1`` dropped; a geometric node
        between the first and the last dense node closer than ``step / 2``
        to a dense node is dropped (``lmax - 1`` is always kept).  Dense
        nodes below ``node_min`` are kept: the cut is where they matter.

        With ``lx_lowpass`` the dense nodes ``round(0.6 lx_lowpass) + k
        lx_lowpass / 15`` up to ``2 lx_lowpass`` (:data:`LOWPASS_RANGE`) are
        added, and for ``lowpass_shape="sharp"`` also every ``lx_lowpass /
        50`` from ``0.95`` to ``1.15 lx_lowpass``
        (:data:`LOWPASS_ONSET_RANGE`); the geometric nodes are dropped by the
        same rule as for the cut, for each of these ranges.  A low-pass whose
        range starts above ``lmax - 1`` (the SPT-3G low-pass at 6144 with
        ``lmax`` 3500) adds nothing.

    Notes
    -----
    Why dense nodes: :math:`\rho` rises from near 0 to a third between
    ``lx`` and ``1.25 lx`` and the spectrum-aware ratio peaks near
    ``1.25 lx``; PCHIP across geometric nodes a factor 1.25 apart missed
    that by -2.3% (white) and +5% (spectrum-aware) at ``1.33 lx`` in a
    survey-scale measurement, an error set by the node spacing in units
    of ``lx`` that does not shrink with the cut.  Nodes every ``lx / 15``
    (a factor about 1.05 near the cut) brought it below 0.1% from
    ``1.17 lx`` up.

    **Low-pass nodes.**  Toy scale (GL grid 360, ``lmax`` 320, mask
    band-limit 64, high-pass 30, low-pass 200, exact basis at a node every 5
    multipoles as the reference): through the rolloff the geometric nodes (a
    factor 1.25 apart) left the interpolated :math:`\rho(\ell, \ell)` up to
    4.3% off for the exponential low-pass (power 6) and 35% off for the
    sharp one.  The nodes this function returns (``lx_lowpass / 15`` from
    ``0.6`` to ``2 lx_lowpass``, and for the sharp shape ``lx_lowpass / 50``
    from ``0.95`` to ``1.15``), with the complete basis at every node, bring
    that to 0.43% (exponential; 0.08% from ``0.75 lx_lowpass`` up) and 1.1%
    (sharp, 0.7% below its onset, 0.1% from ``1.25 lx_lowpass`` up).  The
    sharp shape needs the fine stretch because the passing fraction of a
    degree falls as a square root of ``l - lx_lowpass``: :math:`\rho`
    drops from 0.92 to 0.52 over 20 multipoles, a tenth of ``lx_lowpass``
    (nodes ``lx_lowpass / 15`` apart left 9% there).  The range starts at
    ``0.6 lx_lowpass`` rather than ``0.5``: below it the departure from the
    high-pass alone is a few per cent and smooth (the geometric nodes
    interpolate it to 0.6%), and at ``0.6`` the SPT-3G low-pass (6144, with
    ``lmax`` 3500) adds no node, which a survey-scale check supports (its
    shift of :math:`\rho` by 0.2 to 5%, :math:`\propto (\ell/\ell_x^{low})^6`,
    was interpolated between nodes a factor 1.25 apart to 0.16%).
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

    if lx is None:
        if lx_lowpass is not None:
            raise ValueError("lx_lowpass needs lx")
        return nodes
    lx = float(lx)
    if not np.isfinite(lx) or lx <= 0:
        raise ValueError(f"lx must be finite and positive, got {lx}")
    if near_cut_step is not None:
        if isinstance(near_cut_step, bool) or not np.isfinite(float(near_cut_step)):
            raise ValueError(f"near_cut_step must be a number, got {near_cut_step!r}")
        if float(near_cut_step) < 0:
            raise ValueError(f"near_cut_step must be >= 0, got {near_cut_step}")
    groups = []  # (dense nodes, step) of each range of dense nodes
    if lx_lowpass is not None:
        lx_lowpass = _check_lowpass_cut(lx, lx_lowpass, lowpass_shape)
    if near_cut_step is not None and float(near_cut_step) == 0:
        return nodes
    step = _near_cut_step(lx, near_cut_step)
    lo = _round_half_up(NEAR_CUT_RANGE[0] * lx)
    hi = NEAR_CUT_RANGE[1] * lx
    dense = list(range(lo, int(np.floor(hi)) + 1, step))
    dense = [n for n in dense if 1 <= n <= last]
    if dense:
        groups.append((dense, step))
    if lx_lowpass is not None:
        ranges = [(LOWPASS_RANGE, LOWPASS_DIVISOR)]
        if lowpass_shape == "sharp":
            ranges.append((LOWPASS_ONSET_RANGE, LOWPASS_ONSET_DIVISOR))
        lp_groups = []
        for (a, b), divisor in ranges:
            step_lp = max(1, _round_half_up(lx_lowpass / divisor))
            nodes_lp = range(
                _round_half_up(a * lx_lowpass),
                int(np.floor(b * lx_lowpass + 1e-9)) + 1,  # 1.15 * 200 < 230
                step_lp,
            )
            nodes_lp = [n for n in nodes_lp if 1 <= n <= last]
            if nodes_lp:
                lp_groups.append((nodes_lp, step_lp))
        if len(lp_groups) == 2:
            # the finer onset nodes replace the coarse ones in their stretch
            (coarse, step_c), (fine, step_f) = lp_groups
            coarse = [n for n in coarse if not fine[0] <= n <= fine[-1]]
            lp_groups = (
                [(coarse, step_c), (fine, step_f)] if coarse else [(fine, step_f)]
            )
        groups.extend(lp_groups)
    if not groups:
        return nodes
    kept = [
        n
        for n in nodes
        if n == last
        or all(
            not (d[0] <= n <= d[-1]) or np.abs(np.array(d) - n).min() >= 0.5 * st
            for d, st in groups
        )
    ]
    return sorted(set(kept).union(*(set(d) for d, _ in groups)))


class NodePlan(NamedTuple):
    """The nodes of a ``map_filter`` run and their per-node probe settings
    (:func:`node_plan`): ``nodes`` (list of int), ``nprobe`` (list of int,
    one per node) and ``lmax_int_floor`` (list of int, one per node, 0 for
    no floor)."""

    nodes: list[int]
    nprobe: list[int]
    lmax_int_floor: list[int]


def node_plan(
    lmax: int,
    node_min: int,
    ratio: float,
    lx: float,
    *,
    near_cut_step: float | None = None,
    nprobe: int = 32,
    nprobe_near_cut: int | None = None,
    lx_lowpass: float | None = None,
    lowpass_shape: str = "exp",
) -> NodePlan:
    r"""
    Nodes and per-node probe settings of a filter with cut ``lx``, for
    :func:`filtered_sum_rule_ratio` and :func:`spectrum_aware_ratio`
    (``nodes``, ``nprobe=`` and ``lmax_int_floor=``).

    Parameters
    ----------
    lmax, node_min, ratio, lx, near_cut_step, lx_lowpass, lowpass_shape :
        As for :func:`default_nodes` (``lx`` required here).
    nprobe : int
        Random probes of the other nodes.
    nprobe_near_cut : int, optional
        Probes of the near-cut nodes, from ``round(0.95 lx)`` (the first
        dense node) to below ``2 lx`` (:data:`NEAR_CUT_PROBE_LIMIT`), and,
        with a low-pass, of the nodes from ``round(0.6 lx_lowpass)`` to below
        ``2 lx_lowpass`` (:data:`LOWPASS_RANGE`).
        ``None`` (the default): the complete basis of the degree, ``2 l' +
        1`` vectors, which makes the node's ratio exact.  A count at least
        ``2 l' + 1`` also gives the complete basis.

    Returns
    -------
    NodePlan
        ``nodes`` from :func:`default_nodes`; ``nprobe[i]`` as above;
        ``lmax_int_floor[i] = ceil(BANDLIMIT_FLOOR * lx)`` for the nodes from
        ``round(0.95 lx)`` to below ``1.3 lx`` (:data:`BANDLIMIT_NODE_LIMIT`),
        0 for the others.  The nodes below ``0.95 lx`` keep ``nprobe`` and no
        floor: the transfer function is near zero there and their bins are
        not usable anyway (:data:`FL_CUT`), while the complete basis and a
        raised band-limit would cost as much as near the cut.

    Notes
    -----
    **Probes near the cut.**  32 random probes left 2-3% of noise per node
    between ``lx`` and ``1.25 lx`` at survey scale (coherent over ``d``,
    -3.3% in one bin for one seed); the complete basis removes it and costs
    ``(2 l' + 1) / nprobe`` times the random probes of that node.

    **Band-limit floor.**  A sharp cut scatters power from far multipoles
    into the nodes at the cut, and the default internal band-limit
    ``l' + dmax + margin`` leaves it out.  Survey-scale scan (``lx = 300``,
    ``dmax = 99``, mask coupling width 71, the same probes at every
    band-limit, ratio to the band-limit ``5 lx``, mean over ``|d| <= 10``):
    the white ratio of the node 300 (``1.0 lx``) was -10.9% at the default
    470, -3.6% at ``2.5 lx``, -4.6% at ``3 lx`` (the minimal grid changes
    with the band-limit), -0.8% at ``3.5 lx``; the nodes 320 to 400 were
    within 0.04% from ``3.5 lx`` (0.11 to 0.56% at the default).  The
    sharp cut's ratio at ``lx`` does not strictly converge (it still moved
    by 1% between ``3.5`` and ``5 lx``).  The exponential cut had converged
    to 1e-5 at ``2.5 lx``.  The spectrum-aware ratio is less sensitive
    (-1.6% at the node 300 at the default, -0.2% at ``3 lx`` against
    ``3.33 lx``).

    **Low-pass.**  The nodes through its rolloff carry the near-cut probe
    count: with 32 random probes the toy-scale ratio at the nodes in the
    rolloff had a seed-to-seed spread of 2.4 to 2.8% (the exponential and the
    sharp low-pass at 200, node 238, four seeds; 0.4% for the high-pass
    alone at the same node), because few modes survive there; 128 probes
    bring that to 0.1 to 0.4% below ``0.85 lx_lowpass`` and about 1% (0.1 to
    2.1%) through the rolloff (four seeds, nodes 120 to 320).  No band-limit floor: the complete-basis
    ratio at the nodes in the rolloff agreed with exact rows computed on a
    larger grid and band-limit to 0.1-0.3% (rows 240 and 280; 0.9% at the
    sharp onset, 1.3-1.9% at row 300, 20 multipoles below the toy's
    ``lmax``, where the toy's exact rows truncate the sky).
    """
    lx = float(lx)
    nodes = default_nodes(
        lmax,
        node_min,
        ratio,
        lx=lx,
        near_cut_step=near_cut_step,
        lx_lowpass=lx_lowpass,
        lowpass_shape=lowpass_shape,
    )
    nprobe = int(nprobe)
    if nprobe < 1:
        raise ValueError(f"nprobe must be at least 1, got {nprobe}")
    if nprobe_near_cut is not None:
        nprobe_near_cut = int(nprobe_near_cut)
        if nprobe_near_cut < 1:
            raise ValueError(
                f"nprobe_near_cut must be None or at least 1, got {nprobe_near_cut}"
            )
    floor = int(np.ceil(BANDLIMIT_FLOOR * lx))
    start = _round_half_up(NEAR_CUT_RANGE[0] * lx)
    probes, floors = [], []
    lp_start, lp_stop = None, None
    if lx_lowpass is not None:
        lp_start = _round_half_up(LOWPASS_RANGE[0] * float(lx_lowpass))
        lp_stop = LOWPASS_RANGE[1] * float(lx_lowpass)
    for n in nodes:
        near = start <= n < NEAR_CUT_PROBE_LIMIT * lx
        if lp_start is not None:
            near = near or lp_start <= n < lp_stop
        if near:
            probes.append(2 * n + 1 if nprobe_near_cut is None else nprobe_near_cut)
        else:
            probes.append(nprobe)
        floors.append(floor if start <= n < BANDLIMIT_NODE_LIMIT * lx else 0)
    return NodePlan(nodes, probes, floors)


# --------------------------------------------------------------------------- #
# Usable range: bins with a low transfer function
# --------------------------------------------------------------------------- #
#: A bin whose lowest ``fl`` is below this has a degraded correction: its
#: debiased bandpower is dominated by multipoles that keep little power.
FL_DEGRADED = 0.3
#: A bin whose lowest ``fl`` is below this should be cut: in the survey-scale
#: validation such bins (the one containing the cut) were not reproduced,
#: their variance ratio against simulations ranging from 0.46 to 1.35
#: across the variants of the correction.
FL_CUT = 0.1


def bin_transfer_minimum(
    lbins: Sequence[int], fl: np.ndarray, lmin: int = 0, lmax: int | None = None
) -> list[tuple[int, int, float]]:
    """
    The lowest transfer function of every bin.

    Parameters
    ----------
    lbins : sequence of int
        Bin edges, bin ``i`` the half-open ``[lbins[i], lbins[i + 1])`` (the
        binning of :class:`cmbcov.assembly.binning.BinningManager`).
    fl : ndarray
        Transfer function, ``fl[l]``.
    lmin : int
        Multipoles below it are not in any bin.
    lmax : int, optional
        Multipoles at or above it are not in any bin (default: the length
        of ``fl``).

    Returns
    -------
    list of (int, int, float)
        ``(first, last, lowest fl)`` of every bin with at least one
        multipole in ``[lmin, lmax)`` with a finite ``fl``, ``first`` and
        ``last`` the multipoles of the bin's range (inclusive), in the order
        of the bins.
    """
    fl = np.asarray(fl, dtype=float)
    top = fl.size if lmax is None else min(int(lmax), fl.size)
    out = []
    for lo, hi in zip(lbins[:-1], lbins[1:]):
        first, stop = max(int(lo), int(lmin)), min(int(hi), top)
        if stop <= first:
            continue
        values = fl[first:stop]
        values = values[np.isfinite(values)]
        if values.size:
            out.append((first, stop - 1, float(values.min())))
    return out


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
    # sum_m |y_lm|^2 = (2l+1) C_l; the factor cancels in the ratio but makes
    # the sums those of the docstring.
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
    :func:`filtered_sum_rule_ratio`, "Returns").  A degree ``l' + d`` beyond
    the sums (an internal band-limit capped below ``l' + dmax``, see
    :func:`spectrum_aware_ratio`) counts as unreliable.
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
    size = sum_u.size

    g = np.empty(dmax + 1)
    g[0] = ratio[ellp]
    for d in range(1, dmax + 1):
        hi = ellp + d
        lo = ellp - d
        hi_ok = hi < size and reliable[hi]
        lo_ok = lo >= 0 and reliable[lo]
        if lo_ok and hi_ok:
            g[d] = 0.5 * (ratio[hi] + ratio[lo])
        elif hi_ok:
            g[d] = ratio[hi]
        elif lo_ok:
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
    nprobe: int | Sequence[int] = 32,
    seed: int = 0,
    margin: int | None = None,
    lmax_int_floor: int | Sequence[int] | None = None,
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
    nprobe : int or sequence of int
        Number of random probes per node, one count for all nodes or one per
        node (in the order of ``nodes``; :func:`node_plan` builds it).  A
        node with ``2 l' + 1 <= nprobe`` uses the ``2 l' + 1`` vectors of the
        complete orthonormal real basis of its degree instead, which makes
        its ratio exact.
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
    lmax_int_floor : int or sequence of int, optional
        Lower bound of ``lmax_int``, one for all nodes or one per node:
        ``lmax_int = max(l' + dmax + margin, floor)``.  ``None`` (the
        default) or 0: no floor.  A sharp cut scatters power from far
        multipoles into the nodes at the cut, which the default band-limit
        leaves out (:func:`node_plan`).
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
    synthesised once on the grid :func:`~cmbcov.harmonics.grid.gl_minimal_lmax`
    ``(lmax_int, lw_node)``, on which the unfiltered operator is exact (spin
    0 and spin 2 alike).  A :math:`\theta`-dependent cut is not
    band-limited: the filtered operator is *defined* by that grid's rings.

    Cost per probe: ``"00"`` seven spin-0 GL transforms at ``lmax_int`` (one
    synthesis of the probe shared by both operators, then analysis,
    synthesis and analysis for each) and two ring FFT filters.  The spin-2
    channels share one spin-2 synthesis of ``(z, 0)`` and one spin-2
    analysis per operator; then ``"20"`` adds a spin-0 synthesis and
    analysis per operator and ``"EE"`` a spin-2 synthesis and analysis.  A
    spin-2 transform costs two to three spin-0 ones.
    """
    channels = _check_channels(channels)
    mask_alm, lw, nodes, dmax, nprobe, margin, floors = _check_probe_arguments(
        mask_alm, lw, profile, nodes, dmax, nprobe, margin, lmax_int_floor
    )

    def node(w_map, filt, ellp, lmax_int, lg, node_nprobe):
        exact, count, sums = _node_sums(
            w_map, filt, ellp, lmax_int, lg, channels, node_nprobe, seed, nthreads
        )
        return exact, count, {ch: sums[ch] for ch in channels}

    return _over_nodes(
        "filtered_sum_rule_ratio",
        mask_alm,
        lw,
        profile,
        nodes,
        dmax,
        margin,
        nprobe,
        floors,
        None,
        None,
        nthreads,
        verbose,
        channels,
        node,
        lambda ellp: _white_probe_work(channels, ellp),
    )


def _per_node(value, nodes, name, low, default):
    """``value`` (``None``, an int, or one int per node) as a list of
    ``len(nodes)`` ints ``>= low``; ``None`` gives ``default``."""
    if value is None:
        value = default
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer or one per node, got {value!r}")
    if np.ndim(value) == 0:
        values = [value] * len(nodes)
    else:
        values = list(value)
        if len(values) != len(nodes):
            raise ValueError(
                f"{name} must be an integer or one per node ({len(nodes)}), got "
                f"{len(values)} values"
            )
    out = []
    for v in values:
        if isinstance(v, bool) or int(v) != v:
            raise ValueError(f"{name} values must be integers, got {v!r}")
        if int(v) < low:
            raise ValueError(f"{name} must be at least {low}, got {int(v)}")
        out.append(int(v))
    return out


def _check_probe_arguments(
    mask_alm, lw, profile, nodes, dmax, nprobe, margin, lmax_int_floor=None
):
    """The arguments shared by the probing functions, validated and
    normalised: ``(mask_alm, lw, nodes, dmax, nprobe, margin, floors)``,
    ``margin`` resolved from the mask's coupling width when ``None``,
    ``nprobe`` and ``floors`` (from ``lmax_int_floor``) one int per node."""
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
    nodes = [int(n) for n in nodes]
    if len(nodes) == 0 or min(nodes) < 0:
        raise ValueError(
            f"nodes must be a non-empty list of non-negative ints: {nodes}"
        )
    dmax = int(dmax)
    if dmax < 0:
        raise ValueError(f"dmax must be non-negative, got {dmax}")
    nprobe = _per_node(nprobe, nodes, "nprobe", 1, None)
    floors = _per_node(lmax_int_floor, nodes, "lmax_int_floor", 0, 0)
    if margin is None:
        margin = mask_coupling_width(alm2cl(mask_alm))
    margin = int(margin)
    if margin < 0:
        raise ValueError(f"margin must be non-negative, got {margin}")
    return mask_alm, lw, nodes, dmax, nprobe, margin, floors


#: Progress-line names of the two probing loops, keyed by the ``name`` they
#: pass to :func:`_over_nodes`.
_PROBING_STAGE = {
    "filtered_sum_rule_ratio": "map_filter probing (white channels)",
    "spectrum_aware_ratio": "map_filter probing (spectrum-aware TT)",
}


#: A spin-2 GL transform in spin-0 transforms ("Cost per probe" of
#: :func:`filtered_sum_rule_ratio`: two to three).
_SPIN2_COST = 2.5

#: Rings timed to measure the cost of the ring filter (scaled to the grid).
_FILTER_TIMING_RINGS = 8


def _best_time(call: Callable[[], object], repeat: int) -> float:
    """Shortest of ``repeat`` runs of ``call`` (seconds), after one warm-up."""
    call()
    best = np.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        call()
        best = min(best, time.perf_counter() - t0)
    return best


def _filter_seconds(lg: int) -> float:
    """
    Seconds of one ring-filter application (``rfft``, a product, ``irfft``)
    to one map of the GL grid ``lg``, from timing a few of its rings:
    ``(lg + 1)`` rings of ``2 lg + 2`` points.  Measured, not modelled: the
    cost of an FFT depends on its length through the prime factors (a prime
    ``lg + 1`` makes a node 3-7 times dearer than its neighbour) and on the
    numpy version.
    """
    ntheta, nphi = gl_shape(lg)
    rows = min(ntheta, _FILTER_TIMING_RINGS)
    block = np.random.default_rng(0).standard_normal((rows, nphi))
    h = np.ones((rows, nphi // 2 + 1))

    def call():
        return np.fft.irfft(np.fft.rfft(block, axis=-1) * h, n=nphi, axis=-1)

    return _best_time(call, 3) * ntheta / rows


def _transform_seconds(lmax_int: int, lg: int, nthreads: int | None) -> float:
    """Seconds of one spin-0 GL transform at band-limit ``lmax_int`` on the
    grid ``lg`` (half a synthesis plus analysis pair, timed)."""
    import healpy as hp  # lazy, see utils/healpy_utils.py

    alm = np.random.default_rng(0).standard_normal(hp.Alm.getsize(lmax_int)) + 0j

    def call():
        gl_analysis(
            gl_synthesis(alm, lmax_int, lg, nthreads=nthreads),
            lmax_int,
            lg,
            nthreads=nthreads,
        )

    return _best_time(call, 2) / 2


def _white_probe_work(channels: Sequence[str], ellp: int) -> tuple[float, float]:
    """``(filters, transforms)`` per probe of :func:`_node_sums` at node
    ``ellp`` for ``channels``, counted in the code: ``"00"`` 2 filters and 7
    transforms; the spin-2 channels (``ellp >= 2``) share a spin-2 synthesis
    and a spin-2 analysis per operator, ``"20"`` adds a spin-0 synthesis and
    analysis per operator and a filter, ``"EE"`` a spin-2 synthesis and
    analysis per operator and a two-map filter."""
    filters = transforms = 0.0
    if "00" in channels:
        filters += 2
        transforms += 7
    if ("20" in channels or "EE" in channels) and ellp >= 2:
        filters += 2  # filt(W S_2 (z, 0)): Q and U
        transforms += _SPIN2_COST * 3  # one synthesis, two analyses
        if "20" in channels:
            filters += 1
            transforms += 4
        if "EE" in channels:
            filters += 2
            transforms += _SPIN2_COST * 4
    return filters, transforms


def _spectra_probe_work(npairs: int) -> tuple[float, float]:
    """``(filters, transforms)`` per probe of :func:`_node_sums_spectra` for
    ``npairs`` distinct frequency pairs: one filter on ``W S z`` and one per
    pair; a synthesis and two analyses, then two transforms per operator and
    pair."""
    return 1.0 + npairs, 3.0 + 4.0 * npairs


def _node_geometry(
    ellp: int,
    floor: int,
    dmax: int,
    margin: int,
    lw: int,
    lmax_cap: int | None,
    oversample: float | None,
) -> tuple[int, int, int]:
    """``(lmax_int, lw_node, lg)`` of node ``ellp`` (see :func:`_over_nodes`)."""
    lmax_int = max(ellp + dmax + margin, floor)
    if lmax_cap is not None:
        lmax_int = min(lmax_int, lmax_cap)
    lw_node = min(lw, 2 * lmax_int)
    lg = gl_minimal_lmax(lmax_int, lw_node)
    if oversample is not None:
        lg = max(lg, int(np.ceil(oversample * lmax_int)))
    return lmax_int, lw_node, lg


def _node_weights(
    nodes: Sequence[int],
    nprobe: Sequence[int],
    geometry: Sequence[tuple[int, int, int]],
    probe_work: Callable[[int], tuple[float, float]],
    nthreads: int | None = None,
) -> list[float]:
    """
    Expected cost of each node, for the progress lines: its number of probes
    (the complete basis ``2 l' + 1`` when that is at most ``nprobe``, see
    :func:`_probe_alms`) times the cost of one probe, ``filters`` ring-filter
    applications plus ``transforms`` spin-0 GL transforms (:func:`_node_sums`,
    :func:`_node_sums_spectra`) of the node's grid::

        count * (filters * filter_seconds(lg) + transforms * transform_seconds(lmax_int, lg))

    The structure -- what a node does, how often, on which grid -- is
    counted in the code; the two unit costs are *measured* on the machine by
    timing a few rings of the filter and one transform of the node's own
    size (:func:`_filter_seconds`, :func:`_transform_seconds`; 0.4% of the
    stage). Nothing is fitted: the FFT time is irregular in the grid size and
    the transform time depends on the machine's threads, so neither is a
    formula worth tuning. The progress line scales these by the rate
    observed so far, so only their ratios between nodes matter.

    Validated, with no parameter fitted to them, on five probing runs (8-core
    Apple M-series, 2026-10; ``examples/quick_progress``, lmax 500; lmax 300
    with another cut and low-pass; lmax 500 without low-pass and with 12/24
    probes; lmax 400 with the spectrum-aware TT stage too): the predicted
    total time is within 4-19% of the actual one from the 20% line on (the
    plain count gave 50-70% there).
    """
    filter_s: dict[int, float] = {}
    transform_s: dict[tuple[int, int], float] = {}
    weights = []
    for ellp, n, (lmax_int, _, lg) in zip(nodes, nprobe, geometry):
        count = 2 * ellp + 1 if 2 * ellp + 1 <= n else n
        filters, transforms = probe_work(ellp)
        if lg not in filter_s:
            filter_s[lg] = _filter_seconds(lg)
        if (lmax_int, lg) not in transform_s:
            transform_s[(lmax_int, lg)] = _transform_seconds(lmax_int, lg, nthreads)
        weights.append(
            count * (filters * filter_s[lg] + transforms * transform_s[(lmax_int, lg)])
        )
    return weights


def _over_nodes(
    name: str,
    mask_alm: np.ndarray,
    lw: int,
    profile: Callable[[int], np.ndarray],
    nodes: list[int],
    dmax: int,
    margin: int,
    nprobe: list[int],
    floors: list[int],
    lmax_cap: int | None,
    oversample: float | None,
    nthreads: int | None,
    verbose: bool,
    labels: Sequence[str],
    node_sums: Callable,
    probe_work: Callable[[int], tuple[float, float]],
) -> dict[str, np.ndarray]:
    """
    The node loop shared by :func:`filtered_sum_rule_ratio` and
    :func:`spectrum_aware_ratio`.

    For node ``i`` (multipole ``l'``): ``lmax_int = max(l' + dmax + margin,
    floors[i])`` (at most ``lmax_cap``), the mask truncated to ``lw_node =
    min(lw, 2 lmax_int)`` and synthesised on the grid
    ``gl_minimal_lmax(lmax_int, lw_node)`` (at least ``ceil(oversample *
    lmax_int)`` if ``oversample`` is given), the filter ``profile`` on that
    grid; then ``node_sums(w_map, filt, l', lmax_int, lg, nprobe[i]) ->
    (exact, count, {label: (S^F, S)})`` and row ``i`` of ``g[label]`` is
    :func:`_ratio_by_distance` of those sums.

    ``probe_work(l') -> (filters, transforms)`` counts the work of one probe
    of node ``l'`` for the progress weights (:func:`_node_weights`).
    """
    g = {label: np.empty((len(nodes), dmax + 1)) for label in labels}
    geometry = [
        _node_geometry(ellp, floors[i], dmax, margin, lw, lmax_cap, oversample)
        for i, ellp in enumerate(nodes)
    ]
    # One progress step per node, weighted by the expected cost of the node
    # (probes x cost per probe, which grows steeply with the grid).
    weights = _node_weights(nodes, nprobe, geometry, probe_work, nthreads)
    with Progress(
        _PROBING_STAGE.get(name, name), len(nodes), "node", weights=weights
    ) as progress:
        for i, ellp in enumerate(nodes):
            t0 = time.perf_counter()
            lmax_int, lw_node, lg = geometry[i]
            w_map = gl_synthesis(
                _truncate_alm(mask_alm, lw, lw_node), lw_node, lg, nthreads=nthreads
            )
            filt = FourierRingFilter(profile(lg))
            if filt.h.shape != (gl_shape(lg)[0], gl_shape(lg)[1] // 2 + 1):
                raise ValueError(
                    f"profile({lg}) returned shape {filt.h.shape}; expected "
                    f"{(gl_shape(lg)[0], gl_shape(lg)[1] // 2 + 1)}"
                )
            exact, count, sums = node_sums(w_map, filt, ellp, lmax_int, lg, nprobe[i])
            for label in labels:
                g[label][i] = _ratio_by_distance(*sums[label], ellp, dmax)
            if verbose:
                rho0 = " ".join(f"rho{lab}(d=0)={g[lab][i, 0]:.4f}" for lab in labels)
                print(
                    f"{name}: node {ellp} lmax_int {lmax_int} "
                    f"grid {lg} lw {lw_node} {'exact basis' if exact else 'random'} "
                    f"{count} probes {time.perf_counter() - t0:.2f} s {rho0}",
                    flush=True,
                )
            progress.update()
    return g


# --------------------------------------------------------------------------- #
# Spectrum-aware probing of the TT x TT blocks
# --------------------------------------------------------------------------- #
#: Default GL grid oversampling of :func:`spectrum_aware_ratio` (its Notes,
#: "Grid").
DEFAULT_OVERSAMPLE = 2.0

#: Transfer function below which the pre-filter noise is not reconstructed as
#: ``nl / fl`` (see :func:`pre_filter_noise`).
NOISE_FL_FLOOR = 1e-2


def pre_filter_noise(nl: np.ndarray, fl: np.ndarray) -> np.ndarray:
    r"""
    The noise power spectrum before the filter, ``nl / fl``, from the noise
    ``nl`` of the delivered map.

    The model of the delivered noise is ``nl = fl N_pre``: the noise is
    filtered like the signal (its own transfer function is taken equal to
    the signal's).  Where ``fl`` is not finite or does not exceed
    :data:`NOISE_FL_FLOOR` the ratio is meaningless (0/0 below the cut) and
    ``nl`` itself is kept; for a filtered map that is a noise spectrum
    close to zero there, i.e. the pre-filter noise below the cut is
    neglected.

    Parameters
    ----------
    nl, fl : ndarray
        1-D, ``fl`` at least as long as ``nl``.

    Returns
    -------
    ndarray
        New float64 array of the length of ``nl``.
    """
    nl = np.asarray(nl, dtype=float)
    fl = np.asarray(fl, dtype=float)
    if nl.ndim != 1 or fl.ndim != 1 or fl.size < nl.size:
        raise ValueError(
            f"nl and fl must be 1-D with len(fl) >= len(nl); got {nl.shape}, "
            f"{fl.shape}"
        )
    fl = fl[: nl.size]
    ok = np.isfinite(fl) & (fl > NOISE_FL_FLOOR)
    out = nl.copy()
    out[ok] = nl[ok] / fl[ok]
    return out


def block_id(cov_key) -> str:
    """
    Identifier of a covariance block for :func:`spectrum_aware_ratio` and
    ``CovarianceConfig.filter_rho_blocks``: ``"<stokekey>_<freqkey>"``,
    e.g. ``"TTxTT_090GHz090GHzx090GHz150GHz"``.  The stokes key has the
    fixed form ``XXxYY``, so the identifier is unambiguous.
    """
    return f"{cov_key.stokekey()}_{cov_key.freqkey()}"


def _product_key(pair1: str, pair2: str) -> tuple[str, str]:
    """``sum_m Re(y1 conj y2)`` is symmetric: one key for both orders."""
    return (pair1, pair2) if pair1 <= pair2 else (pair2, pair1)


def _node_sums_spectra(
    w_map: np.ndarray,
    filt: FourierRingFilter,
    ellp: int,
    lmax_int: int,
    lg: int,
    cl_pre: Mapping[str, np.ndarray],
    cl_used: Mapping[str, np.ndarray],
    products: Sequence[tuple[str, str]],
    nprobe: int,
    seed: int,
    nthreads: int | None,
) -> tuple[bool, int, dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]]:
    r"""
    Probe sums of one node: ``(exact, count, sums)`` with
    ``sums[(p1, p2)] = (S^F, S)`` for every product of ``products``,
    :math:`S^F_\ell = \sum_{\rm probes} \sum_m {\rm Re}\,(y^{p_1}_{F,\ell m}
    \overline{y^{p_2}_{F,\ell m}})`, :math:`y^p_F = K^F {\rm diag}(C^p_{pre})
    K^{F\dagger} z`, and :math:`S` the same with ``K`` (no filter) and
    :math:`C^p_{used}`, for ``l = 0 .. lmax_int``.

    The synthesis ``S z`` is shared by both operators, ``K^dagger z`` and
    ``K^{F dagger} z`` by all pairs; each pair then costs a multiplication
    by its spectrum, one synthesis and one analysis per operator.
    """
    import healpy as hp  # lazy, see utils/healpy_utils.py

    pairs = list(dict.fromkeys(p for product in products for p in product))
    size = lmax_int + 1
    sums = {product: (np.zeros(size), np.zeros(size)) for product in products}
    # C_l on the healpy alm layout, once per node (the multiplication by a
    # spectrum is then one elementwise product per pair and probe).
    ell_alm = hp.Alm.getlm(lmax_int)[0]
    spec_f = {p: np.asarray(cl_pre[p], dtype=float)[:size][ell_alm] for p in pairs}
    spec_u = {p: np.asarray(cl_used[p], dtype=float)[:size][ell_alm] for p in pairs}

    def synth(alm):
        return gl_synthesis(alm, lmax_int, lg, nthreads=nthreads)

    def anal(f):
        return gl_analysis(f, lmax_int, lg, nthreads=nthreads)

    exact, probes = _probe_alms(ellp, lmax_int, nprobe, seed)
    count = 0
    for z in probes:
        count += 1
        wz = w_map * synth(z)  # W S z, shared
        u_u = anal(wz)  # K^dagger z
        u_f = anal(filt(wz))  # K^{F dagger} z = A F W S z
        y_u = {p: anal(w_map * synth(u_u * spec_u[p])) for p in pairs}
        y_f = {p: anal(w_map * filt(synth(u_f * spec_f[p]))) for p in pairs}
        for p1, p2 in products:
            s_f, s_u = sums[(p1, p2)]
            s_f += alm2cl(y_f[p1], y_f[p2])
            s_u += alm2cl(y_u[p1], y_u[p2])
    n = 2 * np.arange(size) + 1.0
    for s_f, s_u in sums.values():
        s_f *= n
        s_u *= n
    return exact, count, sums


def _check_terms(terms) -> dict[str, list[tuple[str, str]]]:
    """``terms`` validated: ``{block: [(pair1, pair2), ...]}``, non-empty."""
    if not isinstance(terms, Mapping) or len(terms) == 0:
        raise ValueError(
            "terms must be a non-empty dict {block: [(pair1, pair2), ...]}"
        )
    out = {}
    for block, wick in terms.items():
        wick = [tuple(t) for t in wick]
        if len(wick) == 0 or not all(
            len(t) == 2 and all(isinstance(p, str) for p in t) for t in wick
        ):
            raise ValueError(
                f"terms[{block!r}] must be a non-empty list of (pair1, pair2) "
                f"frequency-pair strings, got {terms[block]!r}"
            )
        out[str(block)] = wick
    return out


def spectrum_aware_ratio(
    mask_alm: np.ndarray,
    lw: int,
    profile: Callable[[int], np.ndarray],
    nodes: Sequence[int],
    dmax: int,
    cl_pre: Mapping[str, np.ndarray],
    cl_used: Mapping[str, np.ndarray],
    terms: Mapping[str, Sequence[tuple[str, str]]],
    *,
    nprobe: int | Sequence[int] = 32,
    seed: int = 0,
    margin: int | None = None,
    lmax_int_floor: int | Sequence[int] | None = None,
    oversample: float | None = DEFAULT_OVERSAMPLE,
    nthreads: int | None = None,
    verbose: bool = False,
) -> dict[str, np.ndarray]:
    r"""
    Ratio of the exact filtered to the exact unfiltered TT x TT covariance
    of each block, with the fiducial spectra, by paired probing.

    The filtered covariance of the TT block ``(ab, cd)`` (frequency pairs
    of its two spectra) is

    .. math::

        \Sigma^F(\ell, \ell') = \frac{1}{n n'} \sum_{m m'} \left[
            R^{ac} \overline{R^{bd}} + R^{ad} \overline{R^{bc}}
        \right]_{\ell m, \ell' m'}, \qquad
        R^{pq} = K^F C^{pq}_{pre} K^{F\dagger},

    with :math:`C^{pq}_{pre}` the spectrum of the sky *before* the filter
    for the pair ``pq``; the unfiltered :math:`\Sigma` has ``K`` and the
    spectra :math:`C^{pq}_{used}` the covariance is computed with.  For a
    real unit-variance probe ``z`` on the degree :math:`\ell'`,
    :math:`E \sum_m (R^1 z)_{\ell m} \overline{(R^2 z)_{\ell m}} =
    \sum_{m m'} R^1 \overline{R^2}`, so pushing the same probes through
    both sides gives :math:`g_C(\ell, \ell') = \Sigma^F / \Sigma` for every
    :math:`\ell` at once.  The block computed with :math:`C_{used}` times
    :math:`g_C` is the filtered block: no transfer function enters
    separately (it is inside the ratio).  With the complete basis of the
    degree (``2 l' + 1 <= nprobe``) the ratio is exact, the scattering of
    power across multipoles by a :math:`\theta`-dependent cut included.

    Parameters
    ----------
    mask_alm, lw, profile, nodes, dmax :
        As for :func:`filtered_sum_rule_ratio`.
    cl_pre : dict
        ``{freq_pair: C_pre}``: the TT spectrum before the filter (signal
        with beams and pixel window but no transfer function, plus the
        pre-filter noise, e.g. :func:`pre_filter_noise`) of every frequency
        pair named in ``terms``.
    cl_used : dict
        ``{freq_pair: C_used}``: the TT spectrum the covariance strategy
        receives for that pair (signal times the full data model, ``fl``
        included, plus the noise).
    terms : dict
        ``{block: [(pair1, pair2), ...]}``: the Wick terms of each block,
        e.g. ``{"b": [("aa", "bb"), ("ab", "ab")]}`` for the block of the
        spectra ``ab`` and ``ab`` (:meth:`cmbcov.assembly.keys.CovKey.key_to_cross`
        gives them; :func:`block_id` names a block).  Repeated terms count
        with their multiplicity; the order within a term does not matter.
    nprobe, seed, margin, lmax_int_floor, nthreads, verbose :
        As for :func:`filtered_sum_rule_ratio`.  All blocks and both
        operators see the same probes; node ``l'`` draws from
        ``np.random.default_rng([seed, l'])``.
    oversample : float or None
        The GL grid of a node has band-limit at least ``ceil(oversample *
        lmax_int)`` (and never less than the minimal exact grid); ``None``
        uses the minimal grid, as :func:`filtered_sum_rule_ratio` does.  See
        Notes, "Grid".

    Returns
    -------
    dict
        ``{block: g}`` in the order of ``terms``, ``g`` of shape
        ``(len(nodes), dmax + 1)`` with the layout, the ``+-d`` averaging
        and the :data:`_RELIABLE_FRACTION` guard (on the unfiltered sum of
        the block) of :func:`filtered_sum_rule_ratio`.

    Notes
    -----
    Per node the internal band-limit is ``lmax_int = l' + dmax + margin``
    (or ``lmax_int_floor`` if larger) and every sky multipole ``0 ..
    lmax_int`` enters both operators, so low-``l`` power scattered to the
    node is included.  **The spectra cap it** (floor included): if a spectrum has fewer than ``lmax_int + 1`` entries (NKA and
    INKA load the spectra to ``lmax`` only), ``lmax_int`` is lowered to the
    shortest length minus one, for both operators alike, and the degrees
    ``l' + d`` beyond it are absent (``g`` then uses ``l' - d`` alone).
    Nothing is padded with zeros.  Power above ``lmax_int`` is missing from
    both sides, as for the white ratio.

    **Grid.**  A :math:`\theta`-dependent sharp cut is not band-limited: it
    puts a step in :math:`\theta` into every mode it chops, whose power
    falls slowly with :math:`\ell`, and on a GL grid of band-limit ``lg``
    the part above the grid aliases into the analysed multipoles.  On the
    minimal grid this overstates the scattered power (at survey scale the
    TT x TT variance was about 5% high against simulations).  The unfiltered
    operator is exact on any grid at least the minimal one, so only the
    filtered side changes.  ``oversample = 2`` (the default) puts the grid at
    about twice the internal band-limit, which removes the overstatement at
    roughly 2-3 times the probing time of the minimal grid.

    Cost per probe: one synthesis of the probe, two analyses (``K^dagger
    z`` with and without the filter), then per distinct frequency pair a
    synthesis and an analysis per operator; one frequency (one pair) costs
    the seven spin-0 transforms of the white ``"00"`` channel.  The probes
    see the spectra, so the result has to be recomputed when they change.
    """
    terms = _check_terms(terms)
    if oversample is not None:
        oversample = float(oversample)
        if not np.isfinite(oversample) or oversample < 1.0:
            raise ValueError(f"oversample must be None or >= 1, got {oversample}")
    mask_alm, lw, nodes, dmax, nprobe, margin, floors = _check_probe_arguments(
        mask_alm, lw, profile, nodes, dmax, nprobe, margin, lmax_int_floor
    )
    products = list(
        dict.fromkeys(_product_key(*t) for wick in terms.values() for t in wick)
    )
    pairs = list(dict.fromkeys(p for product in products for p in product))
    lengths = []
    for name, spectra in (("cl_pre", cl_pre), ("cl_used", cl_used)):
        for p in pairs:
            if p not in spectra:
                raise ValueError(f"{name} has no spectrum for frequency pair {p!r}")
            spec = np.asarray(spectra[p], dtype=float)
            if spec.ndim != 1 or not np.all(np.isfinite(spec)):
                raise ValueError(f"{name}[{p!r}] must be a finite 1-D array")
            lengths.append(spec.size)
    lmax_cap = min(lengths) - 1
    if lmax_cap < max(nodes):
        raise ValueError(
            f"the spectra end at l = {lmax_cap}, below the node {max(nodes)}"
        )

    def node(w_map, filt, ellp, lmax_int, lg, node_nprobe):
        exact, count, sums = _node_sums_spectra(
            w_map,
            filt,
            ellp,
            lmax_int,
            lg,
            cl_pre,
            cl_used,
            products,
            node_nprobe,
            seed,
            nthreads,
        )
        size = lmax_int + 1
        per_block = {}
        for block, wick in terms.items():
            s_f, s_u = np.zeros(size), np.zeros(size)
            for t in wick:
                f, u = sums[_product_key(*t)]
                s_f += f
                s_u += u
            per_block[block] = (s_f, s_u)
        return exact, count, per_block

    return _over_nodes(
        "spectrum_aware_ratio",
        mask_alm,
        lw,
        profile,
        nodes,
        dmax,
        margin,
        nprobe,
        floors,
        lmax_cap,
        oversample,
        nthreads,
        verbose,
        list(terms),
        node,
        lambda ellp: _spectra_probe_work(len(pairs)),
    )


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
    :func:`filtered_sum_rule_ratio` output ``{channel: g}`` (or that of
    :func:`spectrum_aware_ratio`, ``{block: g}``), cached on disk.

    Parameters
    ----------
    cache_dir : str or path
        Root cache directory; files go to ``<cache_dir>/filtered_sum_rule/``
        (created as needed).
    identity : dict
        Everything the result depends on (mask digest, filter parameters,
        channels, nodes, dmax, nprobe, seed, lw, ...), JSON-serialisable
        (numpy scalars and arrays are converted with ``tolist``).  Its
        canonical form is ``json.dumps(identity, sort_keys=True)``.  For
        :func:`spectrum_aware_ratio` it must also hold the Wick terms and a
        digest of both sets of spectra, which the result depends on.
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
    ``g_<channel>`` per channel.  A file lacking ``channels`` is a miss.
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
