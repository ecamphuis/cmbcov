"""Coupling-kernel channels of the ACC: names, aliases and the field pairs.

The nine channels ``Theta^{s}(m, m', L)`` of the reduced coupling kernel, the
T/E/B-letter aliases accepted for them, and the resolution of a requested
subset of them (``spectra``) into canonical names.
"""

from collections.abc import Sequence

__all__ = [
    "CHANNEL_ALIASES",
    "COUPLING_CHANNELS",
    "COUPLING_SPECTRA",
    "LEGACY_CHANNEL_NAMES",
    "normalise_channel",
]


#: Channels of the mode-coupling cross-spectra Theta^{s}(m, m', L), in the
#: channel order of ``theta_before_summing``.  Each is
#: ``sum_M u^a_{l m, LM} conj(u^b_{l' m', LM})`` with ``s = ab``, where ``u^T``
#: (field ``T``) is the spin-0 integral, ``u^D`` (field ``D``) the direct
#: spin-2 response and ``u^L`` (field ``L``) the E->B leakage response to an
#: E-mode unit vector (docs/theory/bmode_kernels.md); :data:`CHANNEL_ALIASES`
#: maps the T/E/B-letter names onto them.  ``TD`` pairs T at ``l`` with D at
#: ``l'``; ``DT`` pairs D at ``l`` with T at ``l'``.  At ``l == l'`` one has
#: ``Theta^{DT}(m, m') = conj(Theta^{TD}(m', m))``, so a kernel whose ``DT``
#: legs are all swapped for ``TD`` is unchanged (``DTxDT = TDxTD``,
#: ``DTxTT = TDxTT``) -- but the mixed ``TDxDT`` is not ``TDxTD`` (they agree
#: only for an azimuthally symmetric mask, where ``Theta^{TD}`` is diagonal
#: and real).  For ``l != l'`` ``DT`` and ``TD`` differ, and ``DT`` is what
#: the second Wick contraction of Cov(TE, TE) needs.  The kernels
#: ``output["s1xs2"]`` are ordered pairs of this tuple.
COUPLING_CHANNELS = ("TT", "DD", "LL", "TD", "DT", "TL", "LT", "DL", "LD")


#: The five channels a precompute computes by default (``spectra=None``).
#: Extending this default would multiply the precompute's disk and wall cost,
#: so the four leakage channels (``TL, LT, DL, LD``) are computed only when
#: requested explicitly via ``spectra`` or ``pairs``.
COUPLING_SPECTRA = COUPLING_CHANNELS[:5]


#: T/E/B-letter channel strings, mapped onto their :data:`COUPLING_CHANNELS`
#: equivalent.  Accepted wherever a channel is named (see
#: :func:`normalise_channel`).
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
#: actually changed name: canonical -> T/E/B-letter label, e.g.
#: ``{"DD": "EE", "TD": "TE", "DT": "ET", ...}``.  Passed to
#: :func:`~cmbcov.approximations.acc.cache.load_coupling_kernels`
#: (``legacy_names``) so a kernel cache written under the letter labels
#: loads without recompute.
LEGACY_CHANNEL_NAMES = {new: old for old, new in CHANNEL_ALIASES.items() if old != new}


def normalise_channel(name: str) -> str:
    """
    A :data:`COUPLING_CHANNELS` string or a T/E/B-letter alias
    (:data:`CHANNEL_ALIASES`) -> the canonical channel name.  Raises
    ``ValueError`` on anything else.
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
#: loads by default: the sixteen ``{TT,DD,TD,DT}^2`` pairs.  ``LL`` is
#: computed by the precompute but never read back by the strategy.
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
        A subset of :data:`COUPLING_CHANNELS`, canonical or T/E/B-letter
        names (see :data:`CHANNEL_ALIASES`).  ``None`` (the default) means
        :data:`COUPLING_SPECTRA`.

    Returns
    -------
    spectra_tuple : tuple of str
        The requested subset, normalised to canonical names and in
        :data:`COUPLING_CHANNELS` order (so the output is deterministic
        regardless of the order ``spectra`` was given in).
    spec_indices : list of int
        The indices of ``spectra_tuple`` into :data:`COUPLING_CHANNELS`.
    t_only : bool
        Whether every kernel the caller asked for needs only the spin-0 (T)
        field on both the central and primed side -- i.e. ``spectra`` is
        ``("TT",)``.  When true, the GL branch can skip the spin-2 (D, L)
        integrals entirely (two Legendre components against one for T).
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
