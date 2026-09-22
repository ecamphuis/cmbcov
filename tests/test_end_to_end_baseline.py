"""
Golden-file test of the whole pipeline.

This is the safety net for structural change. Every other test pins one layer;
this one pins the number the package actually produces, from a parameter file
through validation, data loading, native kernels, the NKA strategy, the
PolSpice transform, D_ell scaling, debiasing and binning.

If a refactor changes any of those numbers, this test says so. Regenerate the
baseline deliberately (and say why in the commit message) rather than adjusting
tolerances.
"""

import glob
import os
import shutil
import tempfile

import numpy as np
import pytest

healpy = pytest.importorskip("healpy")

from cmbcov import CovarianceMatrixGenerator  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "data")


def _clear_baseline_mask_kernel_cache():
    """
    Remove any parameter-keyed kernel cache files left in the shared
    ``tests/data/utils_baseline_mask`` directory by other tests.

    The cache lookup in `cmbcov.mask` intentionally reuses
    a matching cache computed at a larger band limit and slices it down
    (`_kernel_cache_mismatch`'s ``monotonic_fields``): exact for M/Msq, but
    only to the precision of the underlying quadrature for G/K, since those
    are built from the spectrally convergent (not exact) integrals of the
    analytic factors f_apo and 1/w. A G kernel left behind at a larger l1max
    by an earlier test therefore reproduces this run's numbers only to
    ~1e-9 relative, not bit-for-bit, which is close enough to this golden
    test's rtol=1e-7 to blur the distinction between "reused a coarser
    kernel" and "regressed". Clearing the cache directory before each run
    makes the test order-independent: it always recomputes its own kernel
    rather than risking reuse of one computed for a different request.
    """
    cache_dir = os.path.join(DATA, "utils_baseline_mask")
    for pattern in ("*.k*.fits", "*.k*.fits.manifest.json"):
        for path in glob.glob(os.path.join(cache_dir, pattern)):
            os.remove(path)


@pytest.fixture(scope="module")
def pipeline_output():
    """Run the full pipeline in a temporary directory."""
    _clear_baseline_mask_kernel_cache()
    out = tempfile.mkdtemp()
    try:
        template = open(os.path.join(DATA, "baseline_params.yml")).read()
        params = os.path.join(out, "params.yml")
        with open(params, "w") as handle:
            handle.write(
                template.replace("PLACEHOLDER_OUT", out).replace(
                    "PLACEHOLDER_DATA", os.path.abspath(DATA)
                )
            )
        CovarianceMatrixGenerator(params).run_full_analysis()
        version = sorted(glob.glob(os.path.join(out, "v*")))[-1]
        covariance = np.loadtxt(os.path.join(version, "covariance_matrix.dat"))
        bandpowers = np.loadtxt(os.path.join(version, "lbins.dat"))
        yield covariance, bandpowers
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_matches_recorded_baseline(pipeline_output):
    """
    The pipeline must reproduce the recorded covariance up to the rounding
    difference of the BLAS/SHT libraries it is built on, not bit-for-bit:
    the baseline was recorded on a different machine. On Linux x86 (a
    different BLAS from the macOS/Accelerate one this was recorded with),
    1.5% of entries differ from the recorded value by ~1e-9 relative (e.g.
    65.11384076 against 65.11384070, 9.2e-10 relative). rtol=1e-7 leaves
    >100x margin over that while still catching a structural change in the
    result.
    """
    covariance, _ = pipeline_output
    expected = np.load(os.path.join(DATA, "baseline_covariance.npy"))
    assert covariance.shape == expected.shape
    np.testing.assert_allclose(
        covariance, expected, rtol=1e-7, atol=1e-12 * np.abs(expected).max()
    )


def test_bandpowers_match_recorded_baseline(pipeline_output):
    _, bandpowers = pipeline_output
    expected = np.load(os.path.join(DATA, "baseline_lbins.npy"))
    np.testing.assert_allclose(bandpowers, expected, rtol=1e-12, atol=0.0)


def test_output_is_a_valid_covariance_matrix(pipeline_output):
    """
    Symmetry and positive-semi-definiteness are physical requirements, not
    stylistic ones. A kernel convention error breaks both, which is how the
    symmetrisation index bug was caught.
    """
    covariance, _ = pipeline_output
    scale = np.abs(covariance).max()
    assert np.isfinite(covariance).all()
    assert np.abs(covariance - covariance.T).max() < 1e-12 * scale
    assert (np.diag(covariance) > 0).all()

    eigenvalues = np.linalg.eigvalsh((covariance + covariance.T) / 2)
    # The matrix is legitimately near-singular: the two channels in the
    # baseline differ only by beam and noise, so several directions are almost
    # degenerate. Require only that no eigenvalue is meaningfully negative.
    assert eigenvalues.min() > -1e-5 * eigenvalues.max()


def test_correlations_do_not_exceed_one(pipeline_output):
    covariance, _ = pipeline_output
    diagonal = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(diagonal, diagonal)
    assert np.abs(correlation).max() <= 1.0 + 1e-10


def test_sum_asymmetric_stokes_folds_te_and_et(tmp_path):
    """
    Guards against a name mismatch between `Cov._sum_asymmetric_stokes`
    (which calls `cov_keys.get_asymmetrical_stokes`) and `CovKeys` (which
    could define `get_assymetrical_stokes`, double s). Folding TE and ET
    together removes one spectrum, so the matrix shrinks by one bin block.
    """
    template = open(os.path.join(DATA, "baseline_params.yml")).read()
    params = tmp_path / "params.yml"
    params.write_text(
        template.replace("PLACEHOLDER_OUT", str(tmp_path)).replace(
            "PLACEHOLDER_DATA", os.path.abspath(DATA)
        )
        + "\nsum_assymetric_stokes: True\n"
    )
    CovarianceMatrixGenerator(str(params)).run_full_analysis()
    version = sorted(glob.glob(os.path.join(str(tmp_path), "v*")))[-1]
    folded = np.loadtxt(os.path.join(version, "covariance_matrix.dat"))

    baseline = np.load(os.path.join(DATA, "baseline_covariance.npy"))
    n_bins = len(np.load(os.path.join(DATA, "baseline_lbins.npy")))
    assert folded.shape[0] == baseline.shape[0] - n_bins
    assert np.abs(folded - folded.T).max() < 1e-12 * np.abs(folded).max()
