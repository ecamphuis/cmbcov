"""
The batched ACC assembly in ``ACCStrategy.compute_covariance_term`` against
the scalar element loop over ``ACCStrategy.compute_acc_term``.

The reference below is the element loop ``compute_covariance_term`` ran before
it was batched, verbatim apart from formatting: one ``compute_acc_term`` call
per element and Wick contraction, the ``norm_Xi[ell1, ell2]`` /
``norm_Xi[ell2, ell1]`` prefactors, and the auto/non-auto lower band.

Everything is chosen so that an index-order bug cannot hide:

- kernels are random and non-symmetric, distinct per offset and per ordered
  Stokes pair (so ``TExET`` != ``ETxTE`` != ``TExTE``), and not unit-sum (so
  the Eq. 23 normalisation runs), except one that is;
- ``norm_Xi`` is random and non-symmetric (the real one is symmetric, which
  would hide an ``[ell1, ell2]`` / ``[ell2, ell1]`` swap);
- the ``ET`` spectrum is a different array from ``TE`` and the two frequency
  orders are different arrays, so a spectrum looked up with the kernel key,
  or with the wrong frequency order, gives a different number;
- the spectra are 7 multipoles longer than the ``lmax_int = lmax + max(0,
  S - 1 - centralell)`` ACC reads (only ``[:lmax_int]`` may be read);
- geometries cover windows reaching past ``lmax`` into that padding, the low
  end running off ``L = 0``, windows entirely below ``L = 0`` (the scalar
  ``hi <= lo`` return), and a kernel wider than ``lmax``
  (``tests/test_acc_padding.py`` pins the padding).
"""

import itertools
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from cmbcov.approximations import acc as acc_module
from cmbcov.approximations.acc import (
    COUPLING_SPECTRA,
    ACCStrategy,
    acc_internal_lmax,
)
from cmbcov.covariance import CovarianceMethod
from cmbcov.keys import CovKey

FREQS = ("090GHz", "150GHz")
F0, F1 = FREQS

#: Observable (power-spectrum / norm_Xi) labels, as ``SpecKey.stokekey()``
#: gives them after ``sorted_copy()`` -- unaffected by the coupling-kernel
#: channel rename (``COUPLING_SPECTRA``), which only relabels the kernel
#: table below.
OBSERVABLE_SPECTRA = ("TT", "TE", "ET", "EE")

KEYS = [
    # auto, single kernel pair TTxTT for both contractions
    CovKey(("T", "T", "T", "T"), (F0,) * 4),
    # auto, contraction 2 needs TExET
    CovKey(("T", "E", "T", "E"), (F0,) * 4),
    # non-auto, ETxEE and EExET (transposes of each other in real kernels)
    CovKey(("E", "E", "T", "E"), (F0,) * 4),
    # non-auto, two frequencies, ET kept by sorted_copy
    CovKey(("T", "E", "T", "E"), (F0, F0, F1, F0)),
    # non-auto, two frequencies, all four frequency slots differ pairwise
    CovKey(("T", "E", "E", "T"), (F0, F1, F1, F0)),
    # non-auto, TT x EE spectra; both contractions use the TExTE kernel
    CovKey(("T", "T", "E", "E"), (F0, F1, F0, F1)),
]

# (lmax, centralell, size, dmax)
GEOMETRIES = [
    (90, 40, 64, 6),  # windows past lmax for ell1 > 66, lo runoff for ell1 < 40
    (120, 100, 64, 5),  # windows entirely below L = 0 for ell1 <= 36
    (40, 30, 64, 4),  # kernel wider than lmax: past lmax and below L = 0 at once
    (70, 0, 16, 1),  # dmax = 1, offset 0 only
]


class _FakeKernelStrategy(ACCStrategy):
    """ACCStrategy with in-memory kernels instead of the on-disk cache."""

    def __init__(self, cov, kernels):
        super().__init__(cov)
        self._fake_kernels = kernels

    def get_covariance_coupling(self, ell, ell_prime, pairs=None):
        table = self._fake_kernels[ell_prime - ell]
        if pairs is None:
            return dict(table)
        return {pair: table[pair] for pair in pairs}


def _make_inputs(seed, lmax, centralell, size, dmax, signed=False):
    rng = np.random.default_rng(seed)
    kernels = {}
    for offset in range(dmax):
        table = {}
        for s1, s2 in itertools.product(COUPLING_SPECTRA, repeat=2):
            if signed:
                # sign-indefinite, like the real TE/ET kernels, but with a
                # sum well away from zero
                kernel = rng.standard_normal((size, size)) + 0.3
            else:
                kernel = rng.random((size, size))
            # break any accidental near-symmetry
            kernel += np.triu(rng.random((size, size)), 1)
            table[s1, s2] = kernel
        kernels[offset] = table
    # one kernel that is already unit-sum, so the normalisation is skipped
    kernels[0]["TT", "TT"] = kernels[0]["TT", "TT"] / kernels[0]["TT", "TT"].sum()

    norm_xi = {
        pair: 0.5 + rng.random((lmax, lmax))
        for pair in itertools.product(OBSERVABLE_SPECTRA, repeat=2)
    }
    n = acc_internal_lmax(lmax, size, centralell) + 7
    cl = {}
    for fa, fb in itertools.product(FREQS, repeat=2):
        cl[fa + fb] = {
            s: rng.standard_normal(n) if signed else rng.random(n)
            for s in OBSERVABLE_SPECTRA
        }
    config = SimpleNamespace(
        method=CovarianceMethod.ACC, dmax=dmax, centralell=centralell
    )
    cov = SimpleNamespace(lmax=lmax, config=config, norm_Xi=norm_xi)
    return cov, kernels, cl


def _scalar_reference(strategy, cov_key, cl):
    """The pre-batching element loop of compute_covariance_term."""
    cov = strategy.cov
    dmax, centralell = cov.config.dmax, cov.config.centralell
    flat_cov = strategy._empty_flatten_cov(dmax)
    combination_1, combination_2 = cov_key.key_to_cross()
    kernel_1, kernel_2 = cov_key.key_to_cross_kernel()
    needed_pairs = {
        (kernel_1[0].kernel_stokekey(), kernel_1[1].kernel_stokekey()),
        (kernel_2[0].kernel_stokekey(), kernel_2[1].kernel_stokekey()),
    }

    def term(combination, kernel, coupling, e1, e2):
        return (
            strategy.compute_acc_term(
                cl[combination[0].freqkey()],
                cl[combination[1].freqkey()],
                (combination[0].stokekey(), combination[1].stokekey()),
                centralell,
                e1,
                e2,
                covariance_coupling=coupling,
                kernel_key=(kernel[0].kernel_stokekey(), kernel[1].kernel_stokekey()),
            )
            * cov.norm_Xi[combination[0].stokekey(), combination[1].stokekey()][e1, e2]
        )

    for diagonal_offset in range(dmax):
        coupling = strategy.get_covariance_coupling(
            centralell, centralell + diagonal_offset, pairs=needed_pairs
        )
        for ell1 in range(cov.lmax):
            ell2 = ell1 + diagonal_offset
            if ell2 >= cov.lmax:
                continue
            flat_cov[diagonal_offset + dmax - 1, ell1] = term(
                combination_1, kernel_1, coupling, ell1, ell2
            ) + term(combination_2, kernel_2, coupling, ell1, ell2)
            if diagonal_offset != 0:
                if cov_key.auto():
                    flat_cov[-diagonal_offset + dmax - 1, ell1] = flat_cov[
                        diagonal_offset + dmax - 1, ell1
                    ]
                else:
                    flat_cov[-diagonal_offset + dmax - 1, ell1] = term(
                        combination_1, kernel_1, coupling, ell2, ell1
                    ) + term(combination_2, kernel_2, coupling, ell2, ell1)
    return strategy._unflatten_cov(flat_cov)


def _quiet(fn, *args):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args)


@pytest.mark.parametrize(
    "geometry", GEOMETRIES, ids=lambda g: "lmax{}-c{}-S{}-d{}".format(*g)
)
@pytest.mark.parametrize("key_index", range(len(KEYS)))
def test_batched_matches_scalar_loop(geometry, key_index):
    lmax, centralell, size, dmax = geometry
    cov_key = KEYS[key_index]
    cov, kernels, cl = _make_inputs(1000 + key_index, lmax, centralell, size, dmax)

    batched = _quiet(
        _FakeKernelStrategy(cov, kernels).compute_covariance_term, cov_key, cl
    )
    reference = _quiet(
        _scalar_reference, _FakeKernelStrategy(cov, kernels), cov_key, cl
    )

    # The band structure is identical, not just close: same zeros.
    assert np.array_equal(batched != 0, reference != 0)
    assert np.count_nonzero(reference) > 0
    # All inputs are positive, so every nonzero element is a sum of positive
    # terms and a relative tolerance is meaningful elementwise.
    np.testing.assert_allclose(batched, reference, rtol=1e-12, atol=0.0)


def test_batched_matches_scalar_loop_with_small_chunks(monkeypatch):
    """Chunking over ell1 (several chunks per diagonal, ragged last chunk)."""
    lmax, centralell, size, dmax = 90, 40, 64, 6
    # 24 * size bytes per multipole -> 7 multipoles per chunk
    monkeypatch.setattr(acc_module, "_ASSEMBLY_CHUNK_BYTES", 7 * 24 * size)
    for key_index in (1, 2):
        cov_key = KEYS[key_index]
        cov, kernels, cl = _make_inputs(7, lmax, centralell, size, dmax)
        batched = _quiet(
            _FakeKernelStrategy(cov, kernels).compute_covariance_term, cov_key, cl
        )
        reference = _quiet(
            _scalar_reference, _FakeKernelStrategy(cov, kernels), cov_key, cl
        )
        np.testing.assert_allclose(batched, reference, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("key_index", (2, 4))
def test_batched_matches_scalar_loop_sign_indefinite_kernels(key_index):
    """
    Sign-indefinite kernels (like TE/ET) and spectra: elements cancel towards zero,
    where a relative tolerance is meaningless.  The error is instead bounded
    by the absolute-value form of the same sum, the scale its rounding error
    is proportional to: ``|diff| <= 1e-12 * sum_w |norm_w| (|cl1| @ |K_w| @ |cl2|)``.
    """
    lmax, centralell, size, dmax = 90, 40, 64, 6
    cov_key = KEYS[key_index]
    cov, kernels, cl = _make_inputs(55, lmax, centralell, size, dmax, signed=True)

    batched = _quiet(
        _FakeKernelStrategy(cov, kernels).compute_covariance_term, cov_key, cl
    )
    reference = _quiet(
        _scalar_reference, _FakeKernelStrategy(cov, kernels), cov_key, cl
    )

    # Explicit per-element loop, written without compute_acc_term (which
    # would renormalise |K| to unit sum).
    scale = np.zeros((lmax, lmax))
    wick = list(zip(cov_key.key_to_cross(), cov_key.key_to_cross_kernel()))
    for d in range(dmax):
        for ell1 in range(lmax - d):
            ell2 = ell1 + d
            multipoles = np.arange(size) + ell1 - centralell  # L of kernel index
            keep = (multipoles >= 0) & (
                multipoles < acc_internal_lmax(lmax, size, centralell)
            )
            kept = multipoles[keep]
            for e1, e2 in ((ell1, ell2), (ell2, ell1)):
                total = 0.0
                for combination, kernel_pair in wick:
                    spectra_key = (combination[0].stokekey(), combination[1].stokekey())
                    k = kernels[d][
                        kernel_pair[0].kernel_stokekey(),
                        kernel_pair[1].kernel_stokekey(),
                    ]
                    k = np.abs(k / k.sum())[np.ix_(keep, keep)]
                    c1 = np.abs(cl[combination[0].freqkey()][spectra_key[0]][kept])
                    c2 = np.abs(cl[combination[1].freqkey()][spectra_key[1]][kept])
                    total += abs(cov.norm_Xi[spectra_key][e1, e2]) * (c1 @ k @ c2)
                scale[e1, e2] = total

    diff = np.abs(batched - reference)
    assert np.any(np.abs(reference) < 1e-2 * scale), "no cancelling elements tested"
    ratio = np.where(scale > 0, diff / np.where(scale > 0, scale, 1.0), 0.0)
    assert np.all(diff[scale == 0] == 0)
    assert ratio.max() <= 1e-12, ratio.max()


# --------------------------------------------------------------------------- #
# spectra shorter than lmax_int
# --------------------------------------------------------------------------- #
def test_spectra_shorter_than_lmax_int_raise_on_both_paths():
    lmax, centralell, size, dmax = 90, 40, 64, 3
    cov_key = KEYS[0]
    cov, kernels, cl = _make_inputs(9, lmax, centralell, size, dmax)
    needed = acc_internal_lmax(lmax, size, centralell)
    assert needed == lmax + size - 1 - centralell == 113
    short = {f: {s: a[: needed - 1] for s, a in d.items()} for f, d in cl.items()}
    exact = {f: {s: a[:needed] for s, a in d.items()} for f, d in cl.items()}

    strategy = _FakeKernelStrategy(cov, kernels)
    with pytest.raises(ValueError, match=r"needs 113 \(lmax_int = lmax 90 \+ pad 23"):
        strategy.compute_covariance_term(cov_key, short)
    with pytest.raises(ValueError, match="needs 113"):
        _scalar_reference(_FakeKernelStrategy(cov, kernels), cov_key, short)

    # exactly lmax_int multipoles is enough, and what lies past them is unread
    np.testing.assert_array_equal(
        strategy.compute_covariance_term(cov_key, exact),
        strategy.compute_covariance_term(cov_key, cl),
    )
