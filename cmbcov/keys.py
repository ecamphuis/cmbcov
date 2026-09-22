"""
Covariance matrix key management.

This module handles the organization of frequency and Stokes parameter
combinations for covariance matrix computation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Explicit map from a T/E observable spectrum's :meth:`SpecKey.stokekey`
#: (e.g. ``"TE"``) to the ACC coupling-kernel channel that covariance code
#: must index with instead
#: (:data:`cmbcov.approximations.acc.COUPLING_CHANNELS`).
#: Before B-mode support the two were the same strings; keeping the mapping
#: explicit here means a rename of the kernel channels (``EE -> DD``, ``BB ->
#: LL``, ``TE -> TD``, ``ET -> DT``) does not silently change which kernel an
#: observable spectrum reads.
#:
#: A B-mode observable (``BB``, ``TB``, ``EB``) has no entry: its covariance
#: is a *sum* of Wick terms over several channel pairs, not one channel
#: (docs/theory/bmode_kernels.md), so
#: :meth:`SpecKey.kernel_stokekey` raises for it -- use
#: :func:`cmbcov.bmode_wick.required_kernel_pairs`
#: instead, which returns the whole pair set a set of observables needs.
_KERNEL_CHANNEL_OF_STOKEKEY: dict[str, str] = {
    "TT": "TT",
    "EE": "DD",
    "TE": "TD",
    "ET": "DT",
}


@dataclass(frozen=True)
class SpecKey:
    """
    Immutable class representing a spectrum key with Stokes and frequency pairs.

    This class handles pairs of Stokes parameters and frequencies for spectrum
    analysis, providing comparison, sorting, and key generation functionality.

    Parameters
    ----------
    stoke : Tuple[str, str]
        Pair of Stokes parameters (e.g., ('T', 'T'), ('T', 'E'))
    freq : Tuple[str, str]
        Pair of frequency identifiers (e.g., ('090GHz', '150GHz'))
    """

    stoke: tuple[str, str]
    freq: tuple[str, str]

    def __post_init__(self):
        """Validate and normalize the SpecKey after initialization."""
        if len(self.stoke) != 2 or len(self.freq) != 2:
            raise ValueError("Both stoke and freq must be tuples of length 2")

        # Validate Stokes parameters
        for s in self.stoke:
            if s not in ["T", "E", "B"]:
                raise ValueError(
                    f"Invalid Stokes parameter: {s}. Must be 'T', 'E' or 'B'"
                )

    def __str__(self) -> str:
        """Return string representation of the SpecKey."""
        stoke_key, freq_key = self.to_keys()
        return f"{stoke_key} {freq_key}"

    def to_keys(self) -> tuple[str, str]:
        """
        Convert to string keys.

        Returns
        -------
        Tuple[str, str]
            (stokes_string, frequency_string)
        """
        return "".join(self.stoke), "".join(self.freq)

    def auto(self) -> bool:
        """
        Check if this is an auto-spectrum (same Stokes and frequency).

        Returns
        -------
        bool
            True if auto-spectrum, False otherwise
        """
        return (self.stoke[0] == self.stoke[1]) and (self.freq[0] == self.freq[1])

    def sorted_copy(self, according_to_freqs: bool = True) -> SpecKey:
        """
        Return a sorted copy of this SpecKey.

        Parameters
        ----------
        according_to_freqs : bool, default True
            If True, sort according to frequencies first, then Stokes.
            If False, sort according to Stokes first, then frequencies.

        Returns
        -------
        SpecKey
            New sorted SpecKey instance
        """
        stoke, freq = self.stoke, self.freq

        if according_to_freqs:
            if freq[0] > freq[1] or (freq[0] == freq[1] and stoke[0] < stoke[1]):
                stoke, freq = stoke[::-1], freq[::-1]
        else:
            if stoke[0] < stoke[1] or (stoke[0] == stoke[1] and freq[0] < freq[1]):
                stoke, freq = stoke[::-1], freq[::-1]

        return SpecKey(stoke, freq)

    def reverse_stoke(self) -> SpecKey:
        """
        Return a copy with reversed Stokes parameters only.

        Returns
        -------
        SpecKey
            New SpecKey with reversed stoke pair
        """
        return SpecKey(self.stoke[::-1], self.freq)

    def __getitem__(self, item: int) -> tuple[str, str]:
        """Get the item-th pair of (stoke, freq)."""
        return self.stoke[item], self.freq[item]

    def __eq__(self, other: SpecKey) -> bool:
        """
        Check equality with another SpecKey, considering both orientations.

        Two SpecKeys are equal if they represent the same spectrum,
        regardless of order.
        """
        if not isinstance(other, SpecKey):
            return False

        # Direct comparison
        if self.stoke == other.stoke and self.freq == other.freq:
            return True

        # Reversed comparison
        if self.stoke == other.stoke[::-1] and self.freq == other.freq[::-1]:
            return True

        return False

    def __hash__(self) -> int:
        """Hash based on canonical (sorted) representation."""
        # Use sorted representation for consistent hashing
        sorted_spec = self.sorted_copy()
        return hash((sorted_spec.stoke, sorted_spec.freq))

    def freqkey(self, separator: str = "") -> str:
        """
        Get frequency key string.

        Parameters
        ----------
        separator : str, default ""
            Separator between frequencies

        Returns
        -------
        str
            Frequency key string
        """
        return separator.join(self.freq)

    def stokekey(self) -> str:
        """
        Get Stokes key string.

        Returns
        -------
        str
            Stokes key string
        """
        return "".join(self.stoke)

    def kernel_stokekey(self) -> str:
        """
        The ACC coupling-kernel channel this spectrum's ``stokekey()`` maps
        onto, via the explicit table :data:`_KERNEL_CHANNEL_OF_STOKEKEY`
        (e.g. ``"EE" -> "DD"``, ``"TE" -> "TD"``).  Use this, not
        :meth:`stokekey`, wherever the result indexes a coupling-kernel
        dictionary keyed by
        :data:`cmbcov.approximations.acc.COUPLING_CHANNELS`;
        use :meth:`stokekey` for indexing power spectra, which are keyed by
        the observable T/E letters and unaffected by the kernel rename.

        Raises
        ------
        ValueError
            If this spectrum's ``stokekey()`` contains a ``B`` (e.g. ``BB``,
            ``TB``, ``EB``): a B-mode observable's covariance is a sum of
            Wick terms over several channel pairs, not one channel
            (docs/theory/bmode_kernels.md), so there is no
            single kernel to return. Use
            :func:`cmbcov.bmode_wick.required_kernel_pairs`
            instead.
        """
        key = self.stokekey()
        if "B" in key:
            raise ValueError(
                f"kernel_stokekey() has no single channel for {key!r}: a "
                "B-mode observable's covariance is a sum over several "
                "channel pairs. Use "
                "cmbcov.bmode_wick.required_kernel_pairs "
                "instead (docs/theory/bmode_kernels.md)."
            )
        return _KERNEL_CHANNEL_OF_STOKEKEY[key]


@dataclass(frozen=True)
class CovKey:
    """
    Immutable class representing a covariance key with 4-tuple of Stokes/frequency pairs.

    This class represents covariance between two spectra, each defined by a pair
    of Stokes parameters and frequencies.

    Parameters
    ----------
    stoke : Tuple[str, str, str, str]
        Four Stokes parameters representing two spectrum pairs
    freq : Tuple[str, str, str, str]
        Four frequency identifiers representing two spectrum pairs
    """

    stoke: tuple[str, str, str, str]
    freq: tuple[str, str, str, str]

    @classmethod
    def from_spec_keys(cls, left: SpecKey, right: SpecKey) -> CovKey:
        """
        Create CovKey from two SpecKey objects.

        Parameters
        ----------
        left : SpecKey
            Left spectrum key
        right : SpecKey
            Right spectrum key

        Returns
        -------
        CovKey
            New CovKey instance
        """
        stoke = left.stoke + right.stoke
        freq = left.freq + right.freq
        return cls(stoke, freq)

    @classmethod
    def from_mixed_input(
        cls, stoke: tuple | list | str, freq: tuple | list | str
    ) -> CovKey:
        """
        Create CovKey from various input formats (for backward compatibility).

        Parameters
        ----------
        stoke : Union[Tuple, List, str]
            Stokes parameters in various formats
        freq : Union[Tuple, List, str]
            Frequency identifiers in various formats

        Returns
        -------
        CovKey
            New CovKey instance
        """
        # Convert stoke to tuple
        if isinstance(stoke, (tuple, list)):
            stoke_tuple = tuple(stoke)
        elif isinstance(stoke, str):
            if len(stoke) % 4 != 0:
                raise ValueError(
                    f"Stoke string length must be divisible by 4, got {len(stoke)}"
                )
            le = len(stoke) // 4
            stoke_tuple = tuple(stoke[i * le : (i + 1) * le] for i in range(4))
        else:
            raise TypeError(f"Invalid stoke type: {type(stoke)}")

        # Convert freq to tuple
        if isinstance(freq, (tuple, list)):
            freq_tuple = tuple(freq)
        elif isinstance(freq, str):
            if len(freq) % 4 != 0:
                raise ValueError(
                    f"Freq string length must be divisible by 4, got {len(freq)}"
                )
            le = len(freq) // 4
            freq_tuple = tuple(freq[i * le : (i + 1) * le] for i in range(4))
        else:
            raise TypeError(f"Invalid freq type: {type(freq)}")

        return cls(stoke_tuple, freq_tuple)

    def __post_init__(self):
        """Validate the CovKey after initialization and normalize if needed."""
        if len(self.stoke) != 4 or len(self.freq) != 4:
            raise ValueError("Both stoke and freq must be tuples of length 4")

        # Create left and right SpecKeys and normalize them
        left = SpecKey((self.stoke[0], self.stoke[1]), (self.freq[0], self.freq[1]))
        right = SpecKey((self.stoke[2], self.stoke[3]), (self.freq[2], self.freq[3]))

        # Sort the SpecKeys
        left_sorted = left.sorted_copy()
        right_sorted = right.sorted_copy()

        # Update the tuples if sorting changed them
        if left != left_sorted or right != right_sorted:
            new_stoke = left_sorted.stoke + right_sorted.stoke
            new_freq = left_sorted.freq + right_sorted.freq

            # Use object.__setattr__ to modify frozen dataclass
            object.__setattr__(self, "stoke", new_stoke)
            object.__setattr__(self, "freq", new_freq)

    @property
    def left(self) -> SpecKey:
        """Get the left SpecKey."""
        return SpecKey((self.stoke[0], self.stoke[1]), (self.freq[0], self.freq[1]))

    @property
    def right(self) -> SpecKey:
        """Get the right SpecKey."""
        return SpecKey((self.stoke[2], self.stoke[3]), (self.freq[2], self.freq[3]))

    def __eq__(self, other: CovKey) -> bool:
        """Check equality with another CovKey."""
        if not isinstance(other, CovKey):
            return False
        return self.left == other.left and self.right == other.right

    def __str__(self) -> str:
        """Return string representation of the CovKey."""
        stoke_key, freq_key = self.to_keys()
        return f"{stoke_key} {freq_key}"

    def to_keys(self) -> tuple[str, str]:
        """
        Convert to string keys.

        Returns
        -------
        Tuple[str, str]
            (stokes_string, frequency_string)
        """
        return "".join(self.stoke), "".join(self.freq)

    def __hash__(self) -> int:
        """Hash based on left and right SpecKeys."""
        return hash((self.left, self.right))

    def _raise_if_b(self, caller: str) -> None:
        """
        Refuse a B letter mixed with a T/E one:
        ``key_to_cross`` and ``key_to_cross_kernel`` stay the
        two-Wick-contraction shortcut NKA, INKA and the ACC T/E assembly
        use, and NKA's ``norm_Xi`` collapses EE and BB by construction (the
        spin-weight rule cannot tell them apart), so silently applying it to
        a mixed block (e.g. ``TB``, ``EB``) would be wrong, not merely
        unimplemented. Such a block's covariance needs the full multi-term
        sum of :func:`cmbcov.bmode_wick.covkey_wick_terms`
        instead, which the ACC strategy assembles.

        A pure ``BBxBB`` key (every letter ``B``) is let through: it is
        exactly the shortcut this method already returns for ``TT`` or
        ``EE``, which is what the leakage-neglected BB-only NKA/INKA escape
        hatch needs (docs/theory/bmode_kernels.md,
        ``generator.parameter_validation.APPROXIMATIONS_SUPPORTING_B``).
        This never affects ACC: a B run there dispatches to the Wick-term
        assembly before either method is called
        (:meth:`~cmbcov.approximations.acc.ACCStrategy._wick_mode`).
        """
        if "B" in self.stoke and not all(s == "B" for s in self.stoke):
            raise ValueError(
                f"{caller} does not support a B letter mixed with T/E "
                f"({self.stokekey()}): a B-mode block is a sum over several "
                "Wick terms, not the two-contraction shortcut this method "
                "returns. See cmbcov.bmode_wick.block_wick_terms "
                "(docs/theory/bmode_kernels.md)."
            )

    def key_to_cross(self) -> tuple[tuple[SpecKey, SpecKey], tuple[SpecKey, SpecKey]]:
        """
        Convert covariance key to cross-spectrum representation.

        Returns two possible cross-spectrum combinations:
        - (s0,s2) x (s1,s3)
        - (s0,s3) x (s1,s2)

        Returns
        -------
        Tuple[Tuple[SpecKey, SpecKey], Tuple[SpecKey, SpecKey]]
            Two tuples of cross-spectrum pairs

        Raises
        ------
        ValueError
            If this key involves a B letter; see :meth:`_raise_if_b`.
        """
        self._raise_if_b("key_to_cross")
        tuple_1 = (
            SpecKey(
                (self.stoke[0], self.stoke[2]), (self.freq[0], self.freq[2])
            ).sorted_copy(),
            SpecKey(
                (self.stoke[1], self.stoke[3]), (self.freq[1], self.freq[3])
            ).sorted_copy(),
        )
        tuple_2 = (
            SpecKey(
                (self.stoke[0], self.stoke[3]), (self.freq[0], self.freq[3])
            ).sorted_copy(),
            SpecKey(
                (self.stoke[1], self.stoke[2]), (self.freq[1], self.freq[2])
            ).sorted_copy(),
        )
        return tuple_1, tuple_2

    def key_to_cross_kernel(
        self,
    ) -> tuple[tuple[SpecKey, SpecKey], tuple[SpecKey, SpecKey]]:
        """
        Same two Wick-contraction pairs as :meth:`key_to_cross`, but WITHOUT
        ``sorted_copy()``: the Stokes letters keep the true order the Wick
        contraction gives them (e.g. ``ET`` stays ``ET``, it is not resorted
        to ``TE``).

        ``key_to_cross`` sorts each pair because its result is also used to
        index the POWER SPECTRUM, ``cl[freqkey][stokekey]``, which is
        order-independent: :math:`C^{TE}_\\ell = C^{ET}_\\ell` is the same
        spectrum, so which of the two equal dict entries gets used does not
        matter (when both exist; callers generally only populate ``"TE"``).
        It matters for the covariance-coupling KERNEL
        (:data:`cmbcov.approximations.acc.COUPLING_CHANNELS`),
        which is *not* order-independent away from the diagonal:
        :math:`\\Theta^{TE\\times TE}_{\\ell\\ell'} \\neq
        \\Theta^{TE\\times ET}_{\\ell\\ell'}` for :math:`\\ell \\neq \\ell'`
        (they coincide only at :math:`\\ell = \\ell'`). Use this method
        (via :meth:`SpecKey.kernel_stokekey`, not :meth:`SpecKey.stokekey`)
        wherever the result indexes a kernel dictionary keyed by
        ``COUPLING_CHANNELS``; use :meth:`key_to_cross` (and
        :meth:`SpecKey.stokekey`) for indexing power spectra.

        Returns
        -------
        Tuple[Tuple[SpecKey, SpecKey], Tuple[SpecKey, SpecKey]]
            Two tuples of cross-spectrum pairs, Stokes order unsorted.

        Raises
        ------
        ValueError
            If this key involves a B letter; see :meth:`_raise_if_b`.
        """
        self._raise_if_b("key_to_cross_kernel")
        tuple_1 = (
            SpecKey((self.stoke[0], self.stoke[2]), (self.freq[0], self.freq[2])),
            SpecKey((self.stoke[1], self.stoke[3]), (self.freq[1], self.freq[3])),
        )
        tuple_2 = (
            SpecKey((self.stoke[0], self.stoke[3]), (self.freq[0], self.freq[3])),
            SpecKey((self.stoke[1], self.stoke[2]), (self.freq[1], self.freq[2])),
        )
        return tuple_1, tuple_2

    def stokekey(self) -> str:
        """
        Get Stokes key in cross format.

        Returns
        -------
        str
            Stokes key like "TTxEE"
        """
        left_key = "".join(self.stoke[:2])
        right_key = "".join(self.stoke[2:])
        return f"{left_key}x{right_key}"

    def freqkey(self) -> str:
        """
        Get frequency key in cross format.

        Returns
        -------
        str
            Frequency key like "090GHzx150GHz"
        """
        left_key = "".join(self.freq[:2])
        right_key = "".join(self.freq[2:])
        return f"{left_key}x{right_key}"

    def auto(self) -> bool:
        """
        Check if this is an auto-covariance.

        Returns
        -------
        bool
            True if left and right SpecKeys are equal
        """
        return self.left == self.right

    def transpose(self) -> CovKey:
        """
        Return transposed CovKey (swap left and right).

        Returns
        -------
        CovKey
            New CovKey with left and right swapped
        """
        return CovKey(
            (self.stoke[2], self.stoke[3], self.stoke[0], self.stoke[1]),
            (self.freq[2], self.freq[3], self.freq[0], self.freq[1]),
        )

    def shift_stokes(self) -> tuple[CovKey, CovKey, CovKey, CovKey]:
        """
        Generate all Stokes parameter permutations.

        Returns
        -------
        Tuple[CovKey, CovKey, CovKey, CovKey]
            Four CovKey instances with different Stokes arrangements
        """
        return (
            CovKey(self.stoke, self.freq),
            CovKey(
                (self.stoke[0], self.stoke[1], self.stoke[3], self.stoke[2]), self.freq
            ),
            CovKey(
                (self.stoke[1], self.stoke[0], self.stoke[3], self.stoke[2]), self.freq
            ),
            CovKey(
                (self.stoke[1], self.stoke[0], self.stoke[2], self.stoke[3]), self.freq
            ),
        )


def _is_parity_odd_spec(spec_key: SpecKey) -> bool:
    """
    Whether ``spec_key`` is a parity-odd spectrum (``TB``, ``EB`` and their
    reversals): an odd number of ``B`` letters. ``BB`` (two) is parity-even,
    like ``TT``, ``EE`` and ``TE``.
    """
    return spec_key.stokekey().count("B") % 2 == 1


class CovKeys:
    """
    Container class for managing all covariance keys for given Stokes and frequency combinations.

    This class generates and manages all possible CovKey objects for the given
    lists of Stokes parameters and frequencies, providing lookup functionality
    and position mapping.

    Parameters
    ----------
    list_key_stokes : List[str]
        List of Stokes parameter strings (e.g., ['T', 'E']). Ignored (may be
        ``[]``) when ``observables`` is given.
    list_key_freqs : List[str]
        List of frequency identifier strings (e.g., ['090GHz', '150GHz'])
    exclude_asymmetric_stokes : bool, default False
        If True, exclude asymmetric Stokes combinations
    observables : list of str, optional
        Explicit list of two-letter spectra (e.g. ``["TT", "EE", "TE",
        "BB"]``), a subset of ``{TT, EE, BB, TE, TB, EB}`` (``ET``, ``BT``,
        ``BE`` accepted). When given, the spectrum keys are built from
        exactly this list -- not from the Cartesian product of a Stokes
        *alphabet* (``list_key_stokes``) -- so e.g. ``["TT", "EE", "TE",
        "BB"]`` does not also create ``TB``/``EB``.
        ``None`` (the default) keeps the pre-existing alphabet behaviour,
        unchanged, so every existing caller is bit-identical.
    parity_mixed_blocks : bool, default False
        Whether a block between a parity-even spectrum (``TT, EE, TE, BB``)
        and a parity-odd one (``TB, EB``) is included. Only matters when
        ``observables`` mixes the two; a default (T/E-only, or T/E/B with a
        single parity) run never has a parity-odd spectrum, so the filter
        never removes anything.
    """

    def __init__(
        self,
        list_key_stokes: list[str],
        list_key_freqs: list[str],
        exclude_asymmetric_stokes: bool = False,
        observables: list[str] | None = None,
        parity_mixed_blocks: bool = False,
    ):
        if not isinstance(list_key_stokes, list):
            raise TypeError("First parameter should be a list of strings")
        if not isinstance(list_key_freqs, list):
            raise TypeError("Second parameter should be a list of strings")

        self.stokes: list[str] = list_key_stokes
        self.freqs: list[str] = list_key_freqs
        self.exclude_asymmetric_stokes: bool = exclude_asymmetric_stokes
        self.observables: list[str] | None = observables
        self.parity_mixed_blocks: bool = parity_mixed_blocks

        # Generate all spectrum keys
        self.spec_keys: list[SpecKey] = self._generate_spec_keys()

        # Generate position mapping for covariance keys
        self.positions: dict[CovKey, tuple[int, int]] = self._generate_positions()

    def _generate_spec_keys(self) -> list[SpecKey]:
        """
        Generate all SpecKey combinations, from ``observables`` if given,
        otherwise from the Cartesian product of the Stokes and frequency
        lists (the pre-existing, unchanged behaviour).

        Returns
        -------
        List[SpecKey]
            List of all unique SpecKey objects
        """
        if self.observables is not None:
            return self._generate_spec_keys_from_observables()

        spec_keys = []

        for fr1 in self.freqs:
            for fr2 in self.freqs:
                for st1 in self.stokes:
                    for st2 in self.stokes:
                        # Validate Stokes parameters
                        if not (st1 in ["T", "E"] and st2 in ["T", "E"]):
                            raise ValueError(
                                "All Stokes parameters should be either T or E, "
                                f"here you have {st1}, {st2}"
                            )

                        spec_key = SpecKey((st1, st2), (fr1, fr2))

                        # Check for duplicates
                        if spec_key not in spec_keys:
                            # Handle asymmetric Stokes exclusion
                            if (
                                self.exclude_asymmetric_stokes
                                and spec_key.reverse_stoke() in spec_keys
                            ):
                                continue
                            spec_keys.append(spec_key)

        return spec_keys

    def _generate_spec_keys_from_observables(self) -> list[SpecKey]:
        """
        Generate SpecKeys from an explicit ``observables`` list: for each
        requested two-letter spectrum, both Stokes orders (e.g. ``TE`` and
        ``ET``) are generated across every frequency pair when the two
        letters differ -- exactly what the alphabet path already does for a
        pair of letters both present in ``list_key_stokes`` -- so that
        cross-frequency orientations keep being available for
        ``sum_asymmetric_stokes`` as today.
        """
        stoke_pairs: list[tuple[str, str]] = []
        seen_pairs: set[tuple[str, str]] = set()
        for obs in self.observables:
            if not isinstance(obs, str) or len(obs) != 2:
                raise ValueError(
                    f"observables entries must be two-letter spectra, got {obs!r}"
                )
            for pair in ((obs[0], obs[1]), (obs[1], obs[0])):
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    stoke_pairs.append(pair)

        spec_keys: list[SpecKey] = []
        for fr1 in self.freqs:
            for fr2 in self.freqs:
                for st1, st2 in stoke_pairs:
                    spec_key = SpecKey((st1, st2), (fr1, fr2))
                    if spec_key not in spec_keys:
                        if (
                            self.exclude_asymmetric_stokes
                            and spec_key.reverse_stoke() in spec_keys
                        ):
                            continue
                        spec_keys.append(spec_key)
        return spec_keys

    def _generate_positions(self) -> dict[CovKey, tuple[int, int]]:
        """
        Generate position mapping for all CovKey combinations.

        A block between a parity-even spectrum (``TT, EE, TE, BB``) and a
        parity-odd one (``TB, EB``) is left out unless
        ``parity_mixed_blocks``, so it is never computed and stays zero in
        the output matrix -- not merely skipped in favour of some other
        value. This never removes anything from a default (T/E-only, or
        single-parity B) run, which has no parity-odd spectrum at all.

        Returns
        -------
        Dict[CovKey, Tuple[int, int]]
            Mapping from CovKey to (row, col) position indices
        """
        positions = {}

        for ind1, k1 in enumerate(self.spec_keys):
            for ind2, k2 in enumerate(self.spec_keys):
                if ind2 < ind1:  # Only upper triangular + diagonal
                    continue

                if not self.parity_mixed_blocks and (
                    _is_parity_odd_spec(k1) != _is_parity_odd_spec(k2)
                ):
                    continue

                cov_key = CovKey.from_spec_keys(k1, k2)
                positions[cov_key] = (ind1, ind2)

        return positions

    def __str__(self) -> str:
        """Return string representation of CovKeys object."""
        return f"CovKeys object made from Stokes: {self.stokes} and Freqs: {self.freqs}"

    def combined_stokes(self) -> list[str]:
        """
        Get list of unique combined Stokes keys from all SpecKeys.

        Returns
        -------
        List[str]
            List of unique Stokes combinations
        """
        return list(dict.fromkeys([sp.stokekey() for sp in self.spec_keys]))

    def combined_frequencies(self) -> list[str]:
        """
        Get list of unique combined frequency keys from all SpecKeys.

        Returns
        -------
        List[str]
            List of unique frequency combinations
        """
        return list(dict.fromkeys([sp.freqkey() for sp in self.spec_keys]))

    def stokekey(self) -> list[str]:
        """
        Get list of unique Stokes keys from all CovKeys.

        Returns
        -------
        List[str]
            List of unique Stokes cross-combinations
        """
        return list(dict.fromkeys([sp.stokekey() for sp in self.positions.keys()]))

    def freqkey(self) -> list[str]:
        """
        Get list of unique frequency keys from all CovKeys.

        Returns
        -------
        List[str]
            List of unique frequency cross-combinations
        """
        return list(dict.fromkeys([sp.freqkey() for sp in self.positions.keys()]))

    def __getitem__(
        self, item: CovKey | tuple[tuple, tuple] | str
    ) -> tuple[int, int] | list[tuple[int, int]]:
        """
        Get position(s) for given item.

        Parameters
        ----------
        item : CovKey, Tuple, or str
            - CovKey: return position tuple
            - Tuple: (stoke_tuple, freq_tuple) to create CovKey
            - str: Stokes or frequency key to find all matching positions

        Returns
        -------
        Union[Tuple[int, int], List[Tuple[int, int]]]
            Position tuple or list of position tuples
        """
        if isinstance(item, CovKey):
            return self.positions[item]

        elif isinstance(item, tuple) and len(item) == 2:
            stoke, freq = item
            cov_key = CovKey.from_mixed_input(stoke, freq)
            return self.positions[cov_key]

        elif isinstance(item, str):
            # Search in Stokes keys
            if item in self.stokekey():
                return [
                    val for key, val in self.positions.items() if item == key.stokekey()
                ]
            # Search in frequency keys
            elif item in self.freqkey():
                return [
                    val for key, val in self.positions.items() if item == key.freqkey()
                ]
            else:
                raise ValueError(
                    f"String '{item}' not found in Stokes or frequency keys"
                )

        else:
            raise ValueError(f"Invalid item type: {type(item)}")

    def values(self) -> dict[CovKey, tuple[int, int]].values:
        """Get all position values."""
        return self.positions.values()

    def items(self) -> dict[CovKey, tuple[int, int]].items:
        """Get all CovKey items with positions."""
        return self.positions.items()

    def keys(self) -> dict[CovKey, tuple[int, int]].keys:
        """Get all CovKey keys."""
        return self.positions.keys()

    def __len__(self) -> int:
        """Return number of SpecKeys."""
        return len(self.spec_keys)

    def init_binned_output(self, nbin: int, dtype: str = "float") -> np.ndarray:
        """
        Initialize binned output array.

        Parameters
        ----------
        nbin : int
            Number of bins
        dtype : str, default "float"
            Data type for array

        Returns
        -------
        np.ndarray
            Initialized array of shape (len(self) * nbin, len(self) * nbin)
        """
        return np.zeros((len(self) * nbin, len(self) * nbin), dtype=dtype)

    def get_asymmetrical_stokes(self, nbin: int) -> tuple[np.ndarray, CovKeys, dict]:
        """
        Get asymmetrical Stokes configuration.

        Parameters
        ----------
        nbin : int
            Number of bins

        Returns
        -------
        Tuple[np.ndarray, CovKeys, Dict]
            (output_array, new_covkeys, shift_mapping)
        """
        new_cov_keys = CovKeys(
            self.stokes,
            self.freqs,
            exclude_asymmetric_stokes=True,
            observables=self.observables,
            parity_mixed_blocks=self.parity_mixed_blocks,
        )
        output = new_cov_keys.init_binned_output(nbin)
        shift_mapping = {}

        for covkey in new_cov_keys.keys():
            shift_mapping[covkey] = covkey.shift_stokes()

        return output, new_cov_keys, shift_mapping

    #: Deprecated misspelling, kept so existing callers keep working.
    #: The parameter-file key `sum_assymetric_stokes` carries the same
    #: historical misspelling.
    get_assymetrical_stokes = get_asymmetrical_stokes
