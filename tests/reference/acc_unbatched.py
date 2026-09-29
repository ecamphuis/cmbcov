"""
The per-block ACC assembly of cmbcov 0.3.0 + the orientation fixes
(``main`` at 0990d00), kept verbatim as the reference the batched assembly
(``ACCStrategy.compute_covariance_terms``) must reproduce bit for bit.

``te_block`` is the T/E-only path of ``ACCStrategy.compute_covariance_term``
and ``wick_block`` its B-run path (``_compute_covariance_term_wick``), with
``self`` renamed ``strategy``; ``reference_block`` dispatches like
``compute_covariance_term`` did. Test code only: not maintained for speed.
"""

import numpy as np

from cmbcov.approximations.acc import (
    _ASSEMBLY_CHUNK_BYTES,
    _acc_band,
    _check_spectrum_length,
    _contraction_multiset,
    _padded_spectrum,
    _unit_sum_kernel,
    acc_internal_lmax,
)


def reference_block(strategy, cov_key, cl):
    """What ``strategy.compute_covariance_term(cov_key, cl)`` returned at 0990d00."""
    strategy.validate_config()
    if strategy._wick_mode(cov_key):
        return wick_block(strategy, cov_key, cl)
    return te_block(strategy, cov_key, cl)


def te_block(strategy, cov_key, cl):
    """The T/E-only per-block assembly (Eq. 23 per contraction)."""
    dmax = strategy.cov.config.dmax
    centralell = strategy.cov.config.centralell

    # Initialize flattened covariance
    flat_cov = strategy._empty_flatten_cov(dmax)

    # Orientation. The exact covariance is not symmetric within a block
    # between two different spectra, Cov(a_l, b_l') != Cov(a_l', b_l),
    # and ACC is exact at l* only in the orientation its kernel was
    # computed in: spectrum a on the l* leg, b on the l* + Delta leg.
    # Every element is therefore computed in that orientation: the upper
    # triangle (ell1 <= ell2) from the Wick contractions of cov_key, the
    # lower one from those of the transposed key, since
    # Cov(a_{l+D}, b_l) = Cov(b_l, a_{l+D}). An element then depends on
    # its two spectra and on which of its multipoles is the smaller, never
    # on the order in which the run lists its spectra. Using the upper
    # value for both triangles, as before, made Cov(TE_l, EE_l') and
    # Cov(EE_l, TE_l') of different frequency pairs disagree on which
    # element is exact, so the CMB no longer cancelled in the frequency
    # differences and a multi-frequency T/E matrix was not positive
    # definite (docs/theory/acc.md, Sect. 3). An auto block is
    # symmetric and has one orientation.
    #
    # Kernel lookup keys (covariance_coupling is keyed by
    # COUPLING_CHANNELS, via SpecKey.kernel_stokekey -- e.g. "DT" != "TD"
    # off the diagonal) keep the true Wick order of
    # CovKey.key_to_cross_kernel; the cl lookup uses key_to_cross, whose
    # sorted_copy() order is right for spectra ("TE" and "ET" are the
    # same spectrum). The loader is asked for exactly the kernel pairs
    # of these (at most four) contractions.
    #
    # Batched form of the element loop over compute_acc_term (kept below
    # as the scalar reference). For each Wick contraction w:
    #   spectra  cl[combination_w[k].freqkey()][combination_w[k].stokekey()]
    #   kernel   coupling_kernels[kernel_w[0].kernel_stokekey(), kernel_w[1].kernel_stokekey()]
    #   prefactor norm_Xi[combination_w[0].stokekey(), combination_w[1].stokekey()]
    # read at the element's own (row, column) multipoles.
    # compute_acc_term depends on (ell1, ell2) only through
    # min(ell1, ell2): both kernel indices are shifted by the same amount
    # and the section is square. So on diagonal Delta one band per
    # contraction gives the whole upper (or lower) diagonal.
    names: dict[int, str] = {}
    auto = cov_key.auto()
    upper_contractions = strategy._te_contractions(cov_key, cl, names)
    lower_contractions = (
        upper_contractions
        if auto
        else strategy._te_contractions(cov_key.transpose(), cl, names)
    )
    # A block whose transposed key has the same contractions, up to the
    # identity C1 . Theta^{pq} . C2 = C2 . Theta^{qp} . C1 (every block
    # between two TT, or two EE, spectra; so every block of a T-only or
    # E-only run), has one orientation: its lower triangle is its upper
    # band at norm_Xi[ell2, ell1], bit-identical to before.
    one_orientation = auto or _contraction_multiset(
        upper_contractions
    ) == _contraction_multiset(lower_contractions)
    if one_orientation:
        lower_contractions = upper_contractions
    needed_pairs = {
        kernel_key for _, kernel_key, _ in upper_contractions + lower_contractions
    }

    lmax = strategy.cov.lmax
    padded: dict[tuple[int, int], np.ndarray] = {}

    def band(bands, kernels, n_ell, spec_1, spec_2, kernel_key, reuse_reversed):
        """The ACC band of one contraction on the current diagonal,
        memoised in ``bands`` (same spectra arrays, same kernel)."""
        memo = (id(spec_1), id(spec_2), kernel_key)
        if memo in bands:
            return bands[memo]
        if reuse_reversed:
            # C1 . Theta^{pq} . C2 = C2 . Theta^{qp} . C1: the definition
            # gives Theta^{qp} = (Theta^{pq})^T, and the cache stores it
            # so (s2xs1 is written as the transpose of s1xs2; a p x p
            # kernel is a symmetric Gram matrix). A transposed
            # contraction that is a direct one read backwards therefore
            # reuses its band.
            reverse = (id(spec_2), id(spec_1), kernel_key[::-1])
            if reverse in bands:
                return bands[reverse]
        matrix = kernels[kernel_key]
        size = matrix.shape[0]
        n_read = acc_internal_lmax(lmax, size, centralell)
        for spectrum in (spec_1, spec_2):
            if (id(spectrum), size) not in padded:
                _check_spectrum_length(
                    spectrum, names[id(spectrum)], lmax, size, centralell
                )
                padded[id(spectrum), size] = _padded_spectrum(
                    spectrum, centralell, n_read, size
                )
        chunk = max(1, _ASSEMBLY_CHUNK_BYTES // (24 * size))
        bands[memo] = _acc_band(
            matrix,
            padded[id(spec_1), size],
            padded[id(spec_2), size],
            n_ell,
            chunk,
        )
        return bands[memo]

    for diagonal_offset in range(dmax):
        coupling_kernels = strategy.get_covariance_coupling(
            centralell, centralell + diagonal_offset, pairs=needed_pairs
        )
        n_ell = lmax - diagonal_offset
        if n_ell <= 0:
            continue
        ell1 = np.arange(n_ell)
        ell2 = ell1 + diagonal_offset

        # Normalise each distinct kernel once per diagonal, not per element.
        kernels = {}
        for _, kernel_key, _ in upper_contractions + lower_contractions:
            if kernel_key not in kernels:
                kernels[kernel_key] = _unit_sum_kernel(coupling_kernels[kernel_key])

        # One band of ACC values per distinct contraction (same spectra
        # arrays, same kernel), evaluated once.
        bands: dict[tuple, np.ndarray] = {}

        upper = np.zeros(n_ell)
        for (spec_1, spec_2), kernel_key, norm in upper_contractions:
            upper += (
                band(bands, kernels, n_ell, spec_1, spec_2, kernel_key, False)
                * norm[ell1, ell2]
            )
        flat_cov[diagonal_offset + dmax - 1, :n_ell] = upper

        if diagonal_offset != 0:
            if auto:
                lower = upper
            else:
                lower = np.zeros(n_ell)
                for (spec_1, spec_2), kernel_key, norm in lower_contractions:
                    lower += (
                        band(bands, kernels, n_ell, spec_1, spec_2, kernel_key, True)
                        * norm[ell2, ell1]
                    )
            flat_cov[-diagonal_offset + dmax - 1, :n_ell] = lower

    # Unflatten and return full matrix (consistent with other strategies)
    return strategy._unflatten_cov(flat_cov)


def wick_block(strategy, cov_key, cl):
    """The per-Wick-term per-block assembly of a run with a B observable."""
    from cmbcov.keys import _is_parity_odd_spec

    config = strategy.cov.config
    dmax = config.dmax
    centralell = config.centralell
    lmax = strategy.cov.lmax
    flags = strategy._run_flags(cov_key, cl)
    flat_cov = strategy._empty_flatten_cov(dmax)

    if (
        _is_parity_odd_spec(cov_key.left) != _is_parity_odd_spec(cov_key.right)
        and not flags["parity_mixed_blocks"]
    ):
        return strategy._unflatten_cov(flat_cov)

    upper_terms, lower_terms, _ = strategy._wick_plan(cov_key, cl)
    all_terms = upper_terms + (lower_terms or [])
    needed_pairs = sorted({(t.channel_1, t.channel_2) for t in all_terms})
    spectra = {}
    for term in all_terms:
        for side in (term.left, term.right):
            if side not in spectra:
                spectra[side] = strategy._wick_spectrum(cl, side)
    auto = cov_key.auto()
    padded: dict[tuple[int, int], np.ndarray] = {}

    def term_band(term, coupling_kernels, bands, n_ell):
        """The ACC band of one term on the current diagonal (memoised in
        ``bands`` by spectra arrays and pair) and its kernel sum."""
        pair = (term.channel_1, term.channel_2)
        kernel = coupling_kernels[pair]
        size = kernel.shape[0]
        (name_1, spec_1), (name_2, spec_2) = (
            spectra[term.left],
            spectra[term.right],
        )
        memo = (id(spec_1), id(spec_2), pair)
        if memo not in bands:
            n_read = acc_internal_lmax(lmax, size, centralell)
            for name, spectrum in ((name_1, spec_1), (name_2, spec_2)):
                if (id(spectrum), size) not in padded:
                    _check_spectrum_length(
                        spectrum, f"cl[{name}]", lmax, size, centralell
                    )
                    padded[id(spectrum), size] = _padded_spectrum(
                        spectrum, centralell, n_read, size
                    )
            chunk = max(1, _ASSEMBLY_CHUNK_BYTES // (24 * size))
            bands[memo] = _acc_band(
                np.asarray(kernel, dtype=float),
                padded[id(spec_1), size],
                padded[id(spec_2), size],
                n_ell,
                chunk,
            )
        return bands[memo], float(np.sum(kernel))

    for diagonal_offset in range(dmax):
        n_ell = lmax - diagonal_offset
        if n_ell <= 0:
            continue
        ell_prime = centralell + diagonal_offset
        coupling_kernels = strategy.get_covariance_coupling(
            centralell, ell_prime, pairs=needed_pairs
        )
        strategy._require_gl_kernels(centralell, ell_prime)
        ell1 = np.arange(n_ell)
        ell2 = ell1 + diagonal_offset

        bands: dict[tuple, np.ndarray] = {}
        upper = np.zeros(n_ell)
        lower = np.zeros(n_ell)
        # One orientation (lower_terms is None): the lower triangle is
        # the upper band with N_t read at (l + Delta, l), unchanged.
        same_band_lower = diagonal_offset != 0 and not auto and lower_terms is None
        for term in upper_terms:
            band, kernel_sum = term_band(term, coupling_kernels, bands, n_ell)
            scale = strategy.cov.acc_term_scale(
                term.channel_1,
                term.channel_2,
                diagonal_offset,
                kernel_sum,
                ell1,
                ell2,
            )
            upper += term.coefficient * band * scale
            if same_band_lower:
                scale_lower = strategy.cov.acc_term_scale(
                    term.channel_1,
                    term.channel_2,
                    diagonal_offset,
                    kernel_sum,
                    ell2,
                    ell1,
                )
                lower += term.coefficient * band * scale_lower
        # Both orientations: the element (a at l + Delta, b at l) is
        # Cov(b_l, a_{l+Delta}), i.e. the terms of the transposed key
        # with b on the l* leg of their kernels, read at the element's
        # own (row, column) multipoles like the level-1 path.
        if diagonal_offset != 0 and lower_terms is not None:
            for term in lower_terms:
                band, kernel_sum = term_band(term, coupling_kernels, bands, n_ell)
                scale_lower = strategy.cov.acc_term_scale(
                    term.channel_1,
                    term.channel_2,
                    diagonal_offset,
                    kernel_sum,
                    ell2,
                    ell1,
                )
                lower += term.coefficient * band * scale_lower

        flat_cov[diagonal_offset + dmax - 1, :n_ell] = upper
        if diagonal_offset != 0:
            flat_cov[-diagonal_offset + dmax - 1, :n_ell] = upper if auto else lower

    block = strategy._unflatten_cov(flat_cov)
    return block
