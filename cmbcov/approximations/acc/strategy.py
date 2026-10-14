"""The ACC covariance strategy."""

import hashlib
import json
import os
import warnings
from collections.abc import Iterable, Iterator, Sequence

import numpy as np

from ...assembly.covariance import CovarianceMethod, warn_low_centralell_once
from ...assembly.keys import CovKey, SpecKey
from ..base import CovarianceStrategy
from . import cache as acc_cache
from .bands import (
    _BMODE_PAIRS_HELP,
    ACC_NORMALISATION_RULE,
    _acc_bands,
    _BlockPlan,
    _check_spectrum_length,
    _contraction_multiset,
    _group_blocks,
    _identity_memo,
    _missing_kernel_pairs,
    _padded_spectrum,
    _parity_odd_nonzero,
    _unit_sum_kernel,
    acc_cached_kernel_size,
    acc_internal_lmax,
)
from .channels import _DEFAULT_KERNEL_STOKES, COUPLING_CHANNELS, LEGACY_CHANNEL_NAMES

__all__ = [
    "ACCStrategy",
]


#: Memory the batched ACC assembly (:meth:`ACCStrategy.compute_covariance_terms`)
#: may hold for one batch of blocks: their flattened diagonals,
#: ``(2 dmax - 1) x lmax`` doubles each, plus the bands of their kernel
#: products on the current diagonal. A block alone is never split, so a
#: batch always holds at least one.
_ASSEMBLY_BATCH_BYTES = 512 * 1024**2


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
        docs/theory/bmode_kernels.md), with the same orientation rule as
        below when the cache holds the kernel pairs of both orientations
        (:meth:`_wick_plan`). A T/E-only run takes the path below.

        Each element is computed in the orientation its kernel is exact in:
        the upper triangle (``ell1 <= ell2``) of a block ``Cov(a, b)`` from
        the Wick contractions of ``cov_key``, the lower triangle from those
        of ``cov_key.transpose()``, so the lower triangle is the transposed
        upper triangle of ``Cov(b, a)`` and the result does not depend on
        the order in which the run lists its spectra
        (docs/theory/acc.md, Sect. 3). A block whose transposed key has the
        same contractions (an auto block, or one between two TT or two EE
        spectra) is symmetric and computed once.

        The block is computed by the batched assembly of
        :meth:`compute_covariance_terms` as a batch of one; a run's
        :meth:`~cmbcov.assembly.covariance.Cov.compute_covariance_matrix` computes
        its blocks together there, with the same result bit for bit.

        Raises
        ------
        ValueError
            If required ACC parameters are not properly configured, or a
            spectrum this key reads holds fewer than ``lmax_int`` multipoles.
            In a B run also if a Wick term's true spectrum is missing from
            ``cl``, ``centralell < 2``, or the kernels are not GL-grid ones.
        """
        self.validate_config()
        for _, block in self._assemble([cov_key], cl):
            return block

    def compute_covariance_terms(
        self, cov_keys: Iterable[CovKey], cl: dict[str, dict[str, np.ndarray]]
    ) -> Iterator[tuple[CovKey, np.ndarray]]:
        """
        ``(cov_key, compute_covariance_term(cov_key, cl))`` for every key of
        ``cov_keys``, bit for bit, computed together: the order in which the
        pairs are yielded is not that of ``cov_keys``.

        Many blocks of a run evaluate the same kernel product
        ``C1 . Theta . C2`` on a diagonal, and many more share its left
        factor, the window matrix of ``C1`` times ``Theta`` (nearly all of the
        cost). A block alone computes its own; here the blocks are taken in
        batches of blocks sharing left factors (:func:`_group_blocks`),
        bounded by :data:`_ASSEMBLY_BATCH_BYTES`, and each batch runs
        diagonal by diagonal: every distinct product of the batch once, and
        every left factor once (:func:`_acc_bands`). The value of every
        element is unchanged, since each product is evaluated by the same
        operations on the same operands and each block sums its terms in the
        same order. Nothing is kept between calls.

        Used by :meth:`~cmbcov.assembly.covariance.Cov.compute_covariance_matrix`;
        the blocks of one batch are held (flattened) until the batch is done,
        and each is expanded to ``(lmax, lmax)`` only when it is yielded.
        """
        self.validate_config()
        yield from self._assemble(list(cov_keys), cl)

    def _assemble(
        self, cov_keys: list[CovKey], cl: dict[str, dict[str, np.ndarray]]
    ) -> Iterator[tuple[CovKey, np.ndarray]]:
        """Plan every block of ``cov_keys``, group them in batches and yield
        each block when its batch is done (see
        :meth:`compute_covariance_terms`)."""
        identity = _identity_memo()
        plans = []
        for cov_key in cov_keys:
            if self._wick_mode(cov_key):
                plans.append(self._plan_wick_block(cov_key, cl, identity))
            else:
                plans.append(self._plan_te_block(cov_key, cl, identity))

        dmax = self.cov.config.dmax
        lmax = self.cov.lmax
        # T/E and B blocks are never batched together (they only meet in
        # direct calls without run context): one normalises the kernels,
        # the other does not.
        for kind in ("te", "wick"):
            members = [i for i, plan in enumerate(plans) if plan.kind == kind]
            if not members:
                continue
            per_block = max(
                (2 * dmax - 1 + len(plans[i].products)) * lmax * 8 for i in members
            )
            batch_size = max(1, _ASSEMBLY_BATCH_BYTES // max(1, per_block))
            groups = _group_blocks(
                [plans[i].left_factors() for i in members], batch_size
            )
            for group in groups:
                batch = [plans[members[g]] for g in group]
                flats = self._assemble_batch(batch, kind)
                for index, plan in enumerate(batch):
                    # Release each flattened block as it is expanded, and
                    # the whole batch before the next one is computed, so
                    # that at most one batch is held.
                    block = self._unflatten_cov(flats[index])
                    flats[index] = None
                    yield plan.cov_key, block
                    del block
                del flats
        for plan in plans:
            if plan.kind == "zero":
                yield plan.cov_key, self._unflatten_cov(self._empty_flatten_cov(dmax))

    def _plan_te_block(self, cov_key: CovKey, cl, identity) -> _BlockPlan:
        """
        The contractions of a block of a T/E-only run and the kernel
        product each one reads.

        Orientation. The exact covariance is not symmetric within a block
        between two different spectra, Cov(a_l, b_l') != Cov(a_l', b_l),
        and ACC is exact at l* only in the orientation its kernel was
        computed in: spectrum a on the l* leg, b on the l* + Delta leg.
        Every element is therefore computed in that orientation: the upper
        triangle (ell1 <= ell2) from the Wick contractions of cov_key, the
        lower one from those of the transposed key, since
        Cov(a_{l+D}, b_l) = Cov(b_l, a_{l+D}). An element then depends on
        its two spectra and on which of its multipoles is the smaller, never
        on the order in which the run lists its spectra. Using the upper
        value for both triangles would make Cov(TE_l, EE_l') and
        Cov(EE_l, TE_l') of different frequency pairs disagree on which
        element is exact, so the CMB would not cancel in the frequency
        differences and a multi-frequency T/E matrix would not be positive
        definite (docs/theory/acc.md, Sect. 3). An auto block is
        symmetric and has one orientation.

        Kernel lookup keys (covariance_coupling is keyed by
        COUPLING_CHANNELS, via SpecKey.kernel_stokekey -- e.g. "DT" != "TD"
        off the diagonal) keep the true Wick order of
        CovKey.key_to_cross_kernel; the cl lookup uses key_to_cross, whose
        sorted_copy() order is right for spectra ("TE" and "ET" are the
        same spectrum). For each Wick contraction w:
          spectra  cl[combination_w[k].freqkey()][combination_w[k].stokekey()]
          kernel   coupling_kernels[kernel_w[0].kernel_stokekey(), kernel_w[1].kernel_stokekey()]
          prefactor norm_Xi[combination_w[0].stokekey(), combination_w[1].stokekey()]
        read at the element's own (row, column) multipoles.
        compute_acc_term depends on (ell1, ell2) only through
        min(ell1, ell2): both kernel indices are shifted by the same amount
        and the section is square. So on diagonal Delta one band per
        contraction gives the whole upper (or lower) diagonal.

        Band reuse within the block (what fixes the operands of each band):
        a contraction whose (spectrum_1, spectrum_2, kernel pair) arrays
        already have a band reuses it; a lower-triangle contraction that is
        a direct one read backwards, C1 . Theta^{pq} . C2 = C2 . Theta^{qp} . C1
        (the definition gives Theta^{qp} = (Theta^{pq})^T, and the cache
        stores it so: s2xs1 is written as the transpose of s1xs2, a p x p
        kernel is a symmetric Gram matrix), reuses the direct band.
        """
        names: dict[int, str] = {}
        auto = cov_key.auto()
        upper_contractions = self._te_contractions(cov_key, cl, names)
        lower_contractions = (
            upper_contractions
            if auto
            else self._te_contractions(cov_key.transpose(), cl, names)
        )
        # A block whose transposed key has the same contractions, up to the
        # identity C1 . Theta^{pq} . C2 = C2 . Theta^{qp} . C1 (every block
        # between two TT, or two EE, spectra; so every block of a T-only or
        # E-only run), has one orientation: its lower triangle is its upper
        # band at norm_Xi[ell2, ell1].
        one_orientation = auto or _contraction_multiset(
            upper_contractions
        ) == _contraction_multiset(lower_contractions)
        if one_orientation:
            lower_contractions = upper_contractions

        products: dict[tuple, tuple] = {}
        chosen: dict[tuple, tuple] = {}

        def product(spec_1, spec_2, kernel_key) -> tuple:
            key = (identity(spec_1), identity(spec_2), kernel_key)
            if key not in products:
                products[key] = (
                    spec_1,
                    names[id(spec_1)],
                    spec_2,
                    names[id(spec_2)],
                    kernel_key,
                )
            return key

        upper = []
        for (spec_1, spec_2), kernel_key, norm in upper_contractions:
            memo = (id(spec_1), id(spec_2), kernel_key)
            if memo not in chosen:
                chosen[memo] = product(spec_1, spec_2, kernel_key)
            upper.append((chosen[memo], norm))
        lower = None
        if not auto:
            lower = []
            for (spec_1, spec_2), kernel_key, norm in lower_contractions:
                memo = (id(spec_1), id(spec_2), kernel_key)
                reverse = (id(spec_2), id(spec_1), kernel_key[::-1])
                if memo in chosen:
                    key = chosen[memo]
                elif reverse in chosen:
                    key = chosen[reverse]
                else:
                    key = chosen[memo] = product(spec_1, spec_2, kernel_key)
                lower.append((key, norm))
        # The kernels read are those of every contraction, including one
        # whose band is a reversed direct one (the cache must hold them).
        pairs = {
            kernel_key for _, kernel_key, _ in upper_contractions + lower_contractions
        }
        return _BlockPlan(cov_key, "te", auto, upper, lower, products, pairs)

    def _plan_wick_block(self, cov_key: CovKey, cl, identity) -> _BlockPlan:
        """
        The Wick terms of a block of a run with a B observable
        (:meth:`_wick_plan`) and the kernel product each one reads; a
        parity-mixed block the run does not ask for is ``"zero"``. See
        :meth:`_compute_covariance_term_wick`.
        """
        from ...assembly.keys import _is_parity_odd_spec

        flags = self._run_flags(cov_key, cl)
        if (
            _is_parity_odd_spec(cov_key.left) != _is_parity_odd_spec(cov_key.right)
            and not flags["parity_mixed_blocks"]
        ):
            return _BlockPlan(cov_key, "zero")

        upper_terms, lower_terms, _ = self._wick_plan(cov_key, cl)
        spectra = {}
        for term in upper_terms + (lower_terms or []):
            for side in (term.left, term.right):
                if side not in spectra:
                    spectra[side] = self._wick_spectrum(cl, side)
        products: dict[tuple, tuple] = {}

        def product(term) -> tuple:
            pair = (term.channel_1, term.channel_2)
            (name_1, spec_1), (name_2, spec_2) = spectra[term.left], spectra[term.right]
            key = (identity(spec_1), identity(spec_2), pair)
            if key not in products:
                products[key] = (spec_1, f"cl[{name_1}]", spec_2, f"cl[{name_2}]", pair)
            return key

        upper = [(product(term), term) for term in upper_terms]
        lower = (
            None
            if lower_terms is None
            else [(product(term), term) for term in lower_terms]
        )
        pairs = {entry[4] for entry in products.values()}
        return _BlockPlan(
            cov_key, "wick", cov_key.auto(), upper, lower, products, pairs
        )

    def _assemble_batch(self, batch: list[_BlockPlan], kind: str) -> list[np.ndarray]:
        """
        The flattened blocks of one batch of plans of one ``kind``,
        diagonal by diagonal: the kernels once per diagonal, every distinct
        product once (:func:`_acc_bands`), then each block's upper and lower
        bands summed in the block's own term order.
        """
        config = self.cov.config
        dmax = config.dmax
        centralell = config.centralell
        lmax = self.cov.lmax
        flats = [self._empty_flatten_cov(dmax) for _ in batch]
        products: dict[tuple, tuple] = {}
        for plan in batch:
            products.update(plan.products)
        needed_pairs = set().union(*(plan.pairs for plan in batch))
        used_pairs = {entry[4] for entry in products.values()}
        padded: dict[tuple, np.ndarray] = {}

        def padded_spectrum(spectrum, identity, label, size):
            if (identity, size) not in padded:
                _check_spectrum_length(spectrum, label, lmax, size, centralell)
                n_read = acc_internal_lmax(lmax, size, centralell)
                padded[identity, size] = _padded_spectrum(
                    spectrum, centralell, n_read, size
                )
            return padded[identity, size]

        for diagonal_offset in range(dmax):
            n_ell = lmax - diagonal_offset
            ell_prime = centralell + diagonal_offset
            if kind == "te":
                # (the T/E path reads the kernels before the n_ell check)
                coupling_kernels = self.get_covariance_coupling(
                    centralell, ell_prime, pairs=needed_pairs
                )
                if n_ell <= 0:
                    continue
                # Normalise each distinct kernel once per diagonal.
                kernels = {
                    pair: _unit_sum_kernel(coupling_kernels[pair])
                    for pair in used_pairs
                }
            else:
                if n_ell <= 0:
                    continue
                coupling_kernels = self.get_covariance_coupling(
                    centralell, ell_prime, pairs=sorted(needed_pairs)
                )
                self._require_gl_kernels(centralell, ell_prime)
                kernels = {
                    pair: np.asarray(coupling_kernels[pair], dtype=float)
                    for pair in used_pairs
                }
                kernel_sums = {
                    pair: float(np.sum(coupling_kernels[pair])) for pair in used_pairs
                }
            ell1 = np.arange(n_ell)
            ell2 = ell1 + diagonal_offset

            work = {}
            for key, (spec_1, label_1, spec_2, label_2, pair) in products.items():
                kernel = kernels[pair]
                size = kernel.shape[0]
                work[key] = (
                    kernel,
                    padded_spectrum(spec_1, key[0], label_1, size),
                    padded_spectrum(spec_2, key[1], label_2, size),
                )
            bands = _acc_bands(work, n_ell)
            del work

            upper_row = diagonal_offset + dmax - 1
            lower_row = -diagonal_offset + dmax - 1
            if kind == "te":
                self._sum_te_bands(
                    batch,
                    flats,
                    bands,
                    diagonal_offset,
                    ell1,
                    ell2,
                    upper_row,
                    lower_row,
                )
            else:
                self._sum_wick_bands(
                    batch,
                    flats,
                    bands,
                    kernel_sums,
                    diagonal_offset,
                    ell1,
                    ell2,
                    upper_row,
                    lower_row,
                )
        return flats

    @staticmethod
    def _sum_te_bands(
        batch, flats, bands, diagonal_offset, ell1, ell2, upper_row, lower_row
    ):
        """
        One diagonal of every T/E block of a batch from the bands of its
        products: each contraction's band times ``norm_Xi`` read at the
        element's own (row, column), summed in the block's contraction order
        (upper triangle at ``norm[ell1, ell2]``, lower at
        ``norm[ell2, ell1]``); an auto block's lower triangle is its upper
        one. ``norm[...]`` is gathered once per matrix and triangle per
        diagonal.
        """
        n_ell = ell1.size
        norms: dict[tuple, np.ndarray] = {}
        for plan, flat in zip(batch, flats):
            upper = np.zeros(n_ell)
            for key, norm in plan.upper:
                memo = (id(norm), False)
                if memo not in norms:
                    norms[memo] = norm[ell1, ell2]
                upper += bands[key] * norms[memo]
            flat[upper_row, :n_ell] = upper
            if diagonal_offset != 0:
                if plan.auto:
                    lower = upper
                else:
                    lower = np.zeros(n_ell)
                    for key, norm in plan.lower:
                        memo = (id(norm), True)
                        if memo not in norms:
                            norms[memo] = norm[ell2, ell1]
                        lower += bands[key] * norms[memo]
                flat[lower_row, :n_ell] = lower

    def _sum_wick_bands(
        self,
        batch,
        flats,
        bands,
        kernel_sums,
        diagonal_offset,
        ell1,
        ell2,
        upper_row,
        lower_row,
    ):
        """
        One diagonal of every block of a batch of a run with a B observable
        from the bands of its products: each Wick term's
        ``coefficient * band * F_t`` (:meth:`~cmbcov.assembly.covariance.Cov.acc_term_scale`),
        in the block's term order. ``F_t`` depends on the term's kernel
        pair, the diagonal and the triangle only, so it is computed once per
        pair and triangle per diagonal.
        """
        n_ell = ell1.size
        scales: dict[tuple, np.ndarray] = {}

        def scale(term, lower_triangle):
            pair = (term.channel_1, term.channel_2)
            memo = (pair, lower_triangle)
            if memo not in scales:
                scales[memo] = self.cov.acc_term_scale(
                    term.channel_1,
                    term.channel_2,
                    diagonal_offset,
                    kernel_sums[pair],
                    ell2 if lower_triangle else ell1,
                    ell1 if lower_triangle else ell2,
                )
            return scales[memo]

        for plan, flat in zip(batch, flats):
            upper = np.zeros(n_ell)
            lower = np.zeros(n_ell)
            # One orientation (plan.lower is None): the lower triangle is the
            # upper band with N_t read at (l + Delta, l).
            same_band_lower = (
                diagonal_offset != 0 and not plan.auto and plan.lower is None
            )
            for key, term in plan.upper:
                band = bands[key]
                upper += term.coefficient * band * scale(term, False)
                if same_band_lower:
                    lower += term.coefficient * band * scale(term, True)
            # Both orientations: the element (a at l + Delta, b at l) is
            # Cov(b_l, a_{l+Delta}), i.e. the terms of the transposed key with
            # b on the l* leg of their kernels, read at the element's own
            # (row, column) multipoles like the T/E path.
            if diagonal_offset != 0 and plan.lower is not None:
                for key, term in plan.lower:
                    lower += term.coefficient * bands[key] * scale(term, True)
            flat[upper_row, :n_ell] = upper
            if diagonal_offset != 0:
                flat[lower_row, :n_ell] = upper if plan.auto else lower

    def _te_contractions(
        self,
        cov_key: CovKey,
        cl: dict[str, dict[str, np.ndarray]],
        names: dict[int, str],
    ) -> list[tuple[tuple[np.ndarray, np.ndarray], tuple[str, str], np.ndarray]]:
        """
        The two Wick contractions of a T/E ``cov_key``, in its own
        orientation: ``((spectrum_1, spectrum_2), kernel_key, norm_Xi)``
        each, with the spectra looked up through
        :meth:`~cmbcov.assembly.keys.CovKey.key_to_cross` and the kernel pair through
        :meth:`~cmbcov.assembly.keys.CovKey.key_to_cross_kernel`. ``names`` collects a
        label per spectrum array for the length check's message.
        """
        combination_1, combination_2 = cov_key.key_to_cross()
        kernel_1, kernel_2 = cov_key.key_to_cross_kernel()
        contractions = []
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
        return contractions

    def raw_block_inputs(
        self, cov_key: CovKey, cl: dict[str, dict[str, np.ndarray]]
    ) -> tuple[dict[str, np.ndarray], dict]:
        """
        The ``save_raw_blocks`` manifest inputs of an ACC block: the four
        Wick-contraction spectra as :meth:`compute_covariance_term` reads
        them (``[:lmax_int]``), plus ``lmax_int``, ``dmax``, ``centralell`` and the identity
        of the kernel cache -- the resolved ``acc_kernel_dir`` and a digest
        of, for every diagonal, the pair's manifest and the size and
        modification time of the kernel files this key loads (those of both
        orientations for a non-auto block). A recomputed or replaced kernel
        set therefore invalidates the block, including a legacy cache
        without manifests.

        In a run with a B observable (:meth:`_wick_mode`) the spectra are
        every true spectrum the block's expanded Wick terms read (those of
        both orientations for a block computed in both), the pairs are
        those terms' kernel pairs, and the identity also records the
        normalisation rule (:data:`ACC_NORMALISATION_RULE`), the pair list
        and the orientation the block was computed in (see
        :meth:`_wick_plan`). A block saved by a T/E-only run (Eq. 23 per
        contraction) therefore never matches a B-run manifest, and a block
        computed in one orientation never matches one computed in both.
        """
        config = self.cov.config
        kernel_dir = os.path.abspath(self.cov.acc_kernel_dir)
        lmax = self.cov.lmax
        if self._wick_mode(cov_key):
            upper_terms, lower_terms, orientation = self._wick_plan(cov_key, cl)
            terms = upper_terms + (lower_terms or [])
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
                    "acc_block_orientation": orientation,
                }
            )
            return spectra, identity

        # The kernel pairs of both orientations: the lower triangle of a
        # non-auto block is computed from the transposed key's contractions
        # (compute_covariance_term). Its spectra are the same four.
        pairs = set()
        for key in (cov_key,) if cov_key.auto() else (cov_key, cov_key.transpose()):
            for kernel in key.key_to_cross_kernel():
                pairs.add((kernel[0].kernel_stokekey(), kernel[1].kernel_stokekey()))
        pairs = sorted(pairs)
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
                # legacy name (docs/theory/bmode_kernels.md): the file the
                # loader would actually read is the one stated, so a
                # legacy-named or transpose-only cache invalidates the block
                # like any other.
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
        from ...assembly.covariance import has_b_observable
        from ...assembly.keys import _is_parity_odd_spec

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
        from ...assembly.keys import _is_parity_odd_spec

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
        (freq_1, freq_2))``, looked up like the T/E path does
        (:meth:`~cmbcov.assembly.keys.SpecKey.sorted_copy`, so
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
        """Whether the kernel loader can serve every pair on the first
        diagonal ``(centralell, centralell)``: under its own or legacy name,
        or stored only as its transpose, and listed by the manifest's pair
        record if it has one (:func:`_missing_kernel_pairs`, the loader's own
        rule, :func:`~cmbcov.approximations.acc.cache.locate_coupling_kernel`)."""
        ell = self.cov.config.centralell
        return not _missing_kernel_pairs(self.cov.acc_kernel_dir, ell, ell, pairs)

    def _wick_term_multiset(self, terms, cl) -> list:
        """
        The Wick terms of a block as a sorted list of ``(identity,
        coefficient)``, each term taken up to the identity
        ``C1 . Theta^{pq} . C2 = C2 . Theta^{qp} . C1`` and with its spectra
        identified by the arrays ``cl`` holds for them: two blocks with the
        same list have the same ACC value on every element (the Wick
        counterpart of :func:`_contraction_multiset`).
        """
        merged: dict[tuple, int] = {}
        for term in terms:
            left = id(self._wick_spectrum(cl, term.left)[1])
            right = id(self._wick_spectrum(cl, term.right)[1])
            forward = (left, term.channel_1, term.channel_2, right)
            reverse = (right, term.channel_2, term.channel_1, left)
            key = min(forward, reverse)
            merged[key] = merged.get(key, 0) + term.coefficient
        return sorted((k, v) for k, v in merged.items() if v)

    def _wick_plan(self, cov_key: CovKey, cl) -> tuple[list, list | None, str]:
        """
        How a block of a run with a B observable is assembled:
        ``(upper_terms, lower_terms, orientation)``.

        The exact covariance of two different spectra is not symmetric
        within its block, and the kernel of ``Cov(a, b)`` at
        ``(l*, l* + Delta)`` is exact only with ``a`` on the lower multipole
        (docs/theory/acc.md, Sect. 3; docs/theory/bmode_kernels.md,
        Sect. 6). So, as in the T/E-only path, every element is computed in
        its own orientation:

        - an auto block, or a block whose transposed key has the same Wick
          terms (:meth:`_wick_term_multiset`; e.g. two TT, EE or BB spectra
          at different frequencies), has one orientation: ``lower_terms`` is
          ``None``, its lower triangle is its upper band, and
          ``orientation`` is ``"natural"``;
        - every other block takes its upper triangle from the terms of
          ``Cov(a, b)`` and its lower one from those of ``Cov(b, a)``
          (``orientation = "both"``), so the lower triangle is the
          transposed upper triangle of ``Cov(b, a)`` and the block does not
          depend on the order in which the run lists its spectra.

        The kernel pairs of both orientations must be on disk
        (:func:`~cmbcov.approximations.acc.bmode_wick.required_kernel_pairs` includes them).
        There is no single-orientation fallback: a cache that lacks some
        raises ``OSError`` naming the block and its missing pairs.
        """
        from .bmode_wick import covkey_wick_terms

        parity_odd = self._run_flags(cov_key, cl)["parity_odd"]
        natural = covkey_wick_terms(cov_key.stoke, cov_key.freq, parity_odd)
        lower = None
        if cov_key.left != cov_key.right:
            swapped = cov_key.transpose()
            transposed = covkey_wick_terms(swapped.stoke, swapped.freq, parity_odd)
            if self._wick_term_multiset(natural, cl) != self._wick_term_multiset(
                transposed, cl
            ):
                lower = transposed
        needed = sorted({(t.channel_1, t.channel_2) for t in natural + (lower or [])})
        ell = self.cov.config.centralell
        missing = _missing_kernel_pairs(self.cov.acc_kernel_dir, ell, ell, needed)
        if missing:
            raise OSError(
                f"ACC coupling kernels not found under "
                f"{self.cov.acc_kernel_dir!r} for the block "
                f"{cov_key.stokekey()} ({cov_key.freqkey()}): it needs "
                f"{len(needed)} kernel pairs"
                + (" (both orientations)" if lower is not None else "")
                + f", and {', '.join('x'.join(p) for p in missing)} are "
                f"missing at ({ell}, {ell}). {_BMODE_PAIRS_HELP}"
            )
        return natural, lower, "natural" if lower is None else "both"

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
        :math:`(\ell_*, \ell_*+\Delta)`, translated as in Eq. 33
        (:func:`_acc_band`), and :math:`N_t` the per-term rule
        (:meth:`~cmbcov.assembly.covariance.Cov.acc_term_scale`).
        Each element is computed in its own orientation
        (:meth:`_wick_plan`): the upper triangle from the terms of
        ``cov_key``, the lower one from those of ``cov_key.transpose()``,
        both with :math:`N_t` read at the element's own (row, column)
        multipoles, so every element is exact at :math:`\ell_*` and the
        result does not depend on the order in which the run lists its
        spectra (docs/theory/bmode_kernels.md, Sect. 6). An auto block and a
        block whose transposed key has the same terms take one band for both
        triangles, :math:`N_t` read at ``(l + Delta, l)`` for the lower one.
        A cache without the kernel pairs of both orientations is refused
        (``OSError``). A parity-mixed block
        (one parity-odd observable) is zero unless the run asks for
        ``parity_mixed_blocks``.

        :meth:`compute_covariance_term` of such a run computes the block
        through the batched assembly (:meth:`compute_covariance_terms`), with
        the same result; this method forces the per-Wick-term path for one
        block whatever the run context.
        """
        plan = self._plan_wick_block(cov_key, cl, _identity_memo())
        if plan.kind == "zero":
            return self._unflatten_cov(self._empty_flatten_cov(self.cov.config.dmax))
        return self._unflatten_cov(self._assemble_batch([plan], "wick")[0])

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

    def error_budget(self, covariance_keys, band_edges=None) -> dict:
        """
        The error budget of this run, as a dict; every ACC run has one.

        Three parts, from :mod:`~cmbcov.approximations.acc.budget`. For a
        T/E run with a polarised leg: the E->B leakage of the kernel set in
        use, ``lambda = sum Theta^{TT x LL} / sum Theta^{TT x DD}``, giving
        ``(n_E / 2) lambda`` per block (the ``LL`` kernels are already in the
        cache, so this computes nothing new); and the Eq. 33 translation
        error, which is *not* bounded -- the budget reports ``|l - l*|`` at
        the band edges and says so. For every run: the normalisation check
        of the kernels against ``Cov.Xi`` (Eq. 22). See that module for
        details.

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

        A TT-only run has no polarised leg, and a run with a B observable
        uses the per-Wick-term normalisation (docs/theory/bmode_kernels.md),
        which is exact at ``l*`` and carries no Eq. 23 leakage bias of the
        ``(n_E / 2) lambda`` form; its accuracy off ``l*`` is given in
        docs/theory/bmode_kernels.md, "Known limits". For both, the leakage
        and translation sections are reported as not applicable
        (``"applicable": False`` in the dict) and only the normalisation
        check is computed.
        """
        from . import budget as acc_budget

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
        none.

        The manifest records ``spectra`` (the channels that pair was computed
        for), ``grid``, ``lw``, ``nside``, ``centralell`` and ``git_hash``
        (see :func:`~cmbcov.approximations.acc.cache.read_coupling_manifest`),
        which lets :meth:`get_covariance_coupling` tell "this cache was
        built for a smaller spectrum set" apart from "this file is missing".
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
        :meth:`~cmbcov.assembly.covariance.Cov.get_acc_coupling_kernels`,
        not only when a file is missing -- a mismatch raises ``ValueError``
        naming the field; a cache whose manifest records no mask identity (or
        that has no manifest at all) cannot be fully checked and instead
        warns once per kernel directory (``Cov.acc_kernel_dir``) per process.
        Repeated loads of the same ``(ell, ell_prime)`` pair are served from
        an in-memory memo on ``Cov`` after the first (validated) read; see
        that method.

        Parameters
        ----------
        ell, ell_prime : int
            The multipole pair of the kernels (``ell = centralell`` in a run).
        pairs : iterable of (str, str), optional
            The ``(stokes_1, stokes_2)`` kernel pairs to load, each a member
            of :data:`COUPLING_CHANNELS`. Defaults to the sixteen
            ``{TT,DD,TD,DT}^2`` pairs (:data:`_DEFAULT_KERNEL_STOKES`). A
            caller that needs only a subset -- :meth:`compute_covariance_term`
            asks for exactly the two pairs its Wick contractions use, via
            :meth:`~cmbcov.assembly.keys.CovKey.key_to_cross_kernel` -- skips loading
            the rest.

        Returns
        -------
        dict
            ``{(stokes_1, stokes_2): ndarray}``, each kernel square.

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
            silent substitution. Likewise a missing ``DT`` (``ET``) file whose
            ``TD`` (``TE``) counterpart exists is an error: they are
            different kernels.
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
        Compute single ACC covariance term (Eq. 33 translation of the kernel
        at ``central_ell``) for the element ``(ell1, ell2)``.

        Parameters
        ----------
        cl1_dict, cl2_dict : dict
            Spectra of the two legs, indexed by Stokes label; each must hold
            at least ``lmax_int`` multipoles (:func:`acc_internal_lmax`).
        stokes_key : tuple
            ``(stokes_1, stokes_2)`` used to look up the POWER SPECTRA,
            ``cl1_dict[stokes_1]`` and ``cl2_dict[stokes_2]``. Order-
            independent (``"TE"`` and ``"ET"`` are the same spectrum).
        central_ell : int
            The reference multipole ``l*`` of the kernels.
        ell1, ell2 : int
            The element, both ``< lmax``.
        covariance_coupling : dict, optional
            Kernels as returned by :meth:`get_covariance_coupling`; loaded
            here (with a one-time warning) when ``None``.
        kernel_key : tuple, optional
            ``(stokes_1, stokes_2)`` used to look up the covariance-coupling
            KERNEL, ``covariance_coupling[kernel_key]``. Not order-
            independent away from the diagonal (``"TE"`` and ``"ET"`` are
            different kernels for ``ell1 != ell2``); defaults to
            ``stokes_key``, but callers indexing a mixed T/E pair should pass
            the true Wick-contraction order here (see
            ``CovKey.key_to_cross_kernel``).

        Returns
        -------
        float
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
        ``flat[d * lmax::lmax + 1]``); the bands never overlap.  Offsets
        ``d >= lmax`` have no entries and are skipped.
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
