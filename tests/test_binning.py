"""
Regression tests for bandpower binning.

Binning enters the covariance quadratically (``B @ Sigma @ B.T``), so a weight
or bin-count error biases the covariance at roughly twice the rate it biases a
spectrum -- comparable to the 1-5% differences the ACC/INKA comparison is
designed to resolve.
"""

import numpy as np
import pytest

from cmbcov.binning import BinningManager

CUTS = [2, 30, 60, 100, 150, 200]


def _reference_bin_matrix(cuts, lmax, flatten):
    """Independent construction of the expected bin matrix."""
    n = int(np.sum(np.array(cuts) <= lmax)) - 1
    ref = np.zeros((n, lmax))
    for b in range(n):
        for ell in range(cuts[b], cuts[b + 1]):
            ref[b, ell] = ell * (ell + 1) / (2 * np.pi) if flatten else 1.0
    ref /= ref.sum(axis=1)[:, None]
    return ref, n


@pytest.mark.parametrize("flatten", [False, True])
def test_bin_matrix_matches_reference(flatten):
    lmax = 250
    matrix, n = BinningManager(lmax).create_bin_matrix(
        CUTS, flatten_with_ell_factor=flatten
    )
    ref, n_ref = _reference_bin_matrix(CUTS, lmax, flatten)
    assert n == n_ref
    np.testing.assert_allclose(matrix, ref, rtol=1e-14, atol=0.0)


@pytest.mark.parametrize("flatten", [False, True])
def test_bin_rows_are_normalised(flatten):
    """Each bandpower is a weighted mean, so every row must sum to one."""
    matrix, _ = BinningManager(250).create_bin_matrix(
        CUTS, flatten_with_ell_factor=flatten
    )
    np.testing.assert_allclose(matrix.sum(axis=1), 1.0, rtol=1e-13)


def test_final_bin_is_not_dropped_when_edge_equals_lmax():
    """
    Regression: bins are half-open [start, stop), so an edge equal to lmax is a
    valid exclusive upper bound. The previous '<' test silently discarded the
    highest bandpower -- the exact configuration shipped in
    examples/parameters_example.yml (bins ending at 2000 with lmax 2000).
    """
    matrix, n = BinningManager(200).create_bin_matrix(CUTS)
    assert n == len(CUTS) - 1, "final bin [150, 200) must be retained"
    assert int(np.max(np.nonzero(matrix)[1])) == 199


def test_bin_edge_beyond_lmax_is_still_rejected():
    """The guard against out-of-range bins must survive the off-by-one fix."""
    matrix, n = BinningManager(180).create_bin_matrix(CUTS)
    assert n == len(CUTS) - 2
    assert int(np.max(np.nonzero(matrix)[1])) < 180


def test_flattening_weights_are_monotonic_in_ell():
    """The ell(ell+1) factor must up-weight high ell within a bin."""
    w_lo = BinningManager(200).bin_weight(10, True)
    w_hi = BinningManager(200).bin_weight(100, True)
    assert 0 < w_lo < w_hi
    assert BinningManager(200).bin_weight(10, False) == 1.0


def test_uniform_and_flattened_binning_differ():
    """Guards against the two branches silently collapsing into one."""
    a, _ = BinningManager(250).create_bin_matrix(CUTS, flatten_with_ell_factor=False)
    b, _ = BinningManager(250).create_bin_matrix(CUTS, flatten_with_ell_factor=True)
    assert not np.allclose(a, b)


def test_binning_a_constant_spectrum_is_exact():
    """
    A flat C_ell must bin to itself under either weighting, since rows are
    normalised. This is the binning analogue of the normalisation identity
    sum_l1l2 Theta_bar = 1 in Camphuis et al. (2022) Eq. (24).
    """
    lmax = 250
    for flatten in (False, True):
        matrix, _ = BinningManager(lmax).create_bin_matrix(
            CUTS, flatten_with_ell_factor=flatten
        )
        np.testing.assert_allclose(matrix @ np.ones(lmax), 1.0, rtol=1e-13)


# ---------------------------------------------------------------------------
# lmin
#
# lmin was a REQUIRED parameter that no computation ever read: it was validated
# and then ignored, so every user supplied a value that did nothing. It now
# truncates the bandpowers from below.
# ---------------------------------------------------------------------------


def test_lmin_below_first_bin_edge_is_a_no_op():
    """Existing parameter files use lmin=2 with bins starting at 2."""
    plain, n_plain = BinningManager(250).create_bin_matrix(CUTS)
    cut, n_cut = BinningManager(250, lmin=2).create_bin_matrix(CUTS)
    assert n_plain == n_cut
    np.testing.assert_array_equal(plain, cut)


def test_lmin_removes_weight_below_the_cut():
    matrix, _ = BinningManager(250, lmin=45).create_bin_matrix(CUTS)
    assert matrix[:, :45].sum() == 0.0
    assert int(np.min(np.nonzero(matrix)[1])) >= 45


def test_lmin_drops_bins_that_fall_entirely_below_it():
    """CUTS = [2, 30, 60, 100, 150, 200]; lmin=60 kills the first two bins."""
    _, n = BinningManager(250, lmin=60).create_bin_matrix(CUTS)
    assert n == len(CUTS) - 1 - 2


def test_a_bin_straddling_lmin_is_truncated_not_dropped():
    """
    The usable modes above lmin are kept, and the reported centre moves to
    reflect the range actually averaged.
    """
    full, _ = BinningManager(250).create_bin_matrix(CUTS)
    cut, _ = BinningManager(250, lmin=45).create_bin_matrix(CUTS)
    ells = np.arange(250)
    # CUTS[1:2] = [30, 60) straddles 45 and must survive
    centre_full = (full @ ells)[1]
    centre_cut = (cut @ ells)[0]
    assert 30 < centre_full < 60
    assert 45 <= centre_cut < 60
    assert centre_cut > centre_full


def test_lmin_preserves_normalisation():
    """Rows must still sum to one, so a flat spectrum still bins to itself."""
    matrix, _ = BinningManager(250, lmin=45).create_bin_matrix(CUTS)
    np.testing.assert_allclose(matrix.sum(axis=1), 1.0, rtol=1e-13)
    np.testing.assert_allclose(matrix @ np.ones(250), 1.0, rtol=1e-13)


@pytest.mark.parametrize("bad", [-1, 250, 300])
def test_invalid_lmin_is_rejected(bad):
    with pytest.raises(ValueError):
        BinningManager(250, lmin=bad)
