r"""
PolSpice post-processing of a covariance that has a B-mode
observable (docs/theory/bmode_kernels.md, Sect. 7).

(a) **Linear algebra.** On random small symmetric pseudo covariances, the
    full transform matrix ``T`` (block rows per output spectrum, including
    the EE/BB mixing) assembled by hand matches
    :meth:`~cmbcov.postprocess.CovariancePostProcessor.pseudo_to_spice_bmode`
    through a block provider, ``T Sigma T^T``, to 1e-12, for the level 2, 3
    and 4 observable sets. Levels 3 and 4 use the same spectrum list here --
    ``parity_mixed_blocks`` only gates which blocks
    ``compute_covariance_matrix`` *asks for*, not what the transform of a
    single block does, so the two are identical at this layer.
(b) **Decoupling property**, the physics check. The full pseudo
    mean-coupling matrix ``M`` (MASTER, including the EE<->BB mixing
    ``M-`` and the TB/EB kernels) is built from the four channels of
    :func:`~cmbcov.kernels.coupling.coupling_kernels` as
    ``1/2 (M1 +/- M2)`` (``M`` with mixing is not available as a single
    array elsewhere in the code, so this is the documented fallback -- see
    docs/theory/bmode_kernels.md, "What ``pseudo_to_spice``
    would need"). ``T @ M`` is checked block-diagonal in (EE, BB): EE<-BB
    and BB<-EE vanish to numerical precision (a real cancellation between
    two nonzero products, not a zero by construction), TB<-TE and
    EB<-(EE, BB) are zero by construction (asserted for documentation), and
    every diagonal block matches the PolSpice kernel Sect. 11 names for it:
    Kern3 = ``^-2K`` for EE, BB and EB, and the TE/TB kernel ``^xK``, the
    same computation ``tests/test_bmode_theta_combinations.py`` uses for
    ``^x G M^TE = ^xK`` and ``^-2 G M^- = ^-2 K``.
(c) **Level 1.** A T/E-only ACC run matches a T/E-only
    reference (the same stored reference
    ``tests/reference/acc_level1_reference.npz`` that
    ``tests/test_acc_bmode_assembly.py`` pins), confirming PolSpice
    post-processing leaves the level-1 code path untouched.
(d) **End to end.** A small ``CovarianceMatrixGenerator`` run with
    ``polspice_postprocess: true`` for levels 2, 3 and 4 (reusing the
    end-to-end setup of ``tests/test_acc_bmode_assembly.py``):
    finite, symmetric to 1e-12, correct block layout. The smallest/largest
    eigenvalue ratio is printed, as in that file.
(e) The validation rule: ``observables`` with ``EE`` and a B observable but
    not ``BB`` is rejected when ``polspice_postprocess: true``
    (``generator/parameter_validation.py``), and accepted otherwise.
"""

import glob
import os
import shutil
import warnings

import numpy as np
import pytest
import yaml

hp = pytest.importorskip("healpy")

from reference.acc_level1 import build_level1_reference  # noqa: E402

from cmbcov import CovarianceMatrixGenerator  # noqa: E402
from cmbcov.generator.parameter_validation import (  # noqa: E402
    ParameterValidator,
)
from cmbcov.kernels.coupling import (  # noqa: E402
    KERNEL_TE,
    KERNEL_TT,
    KERNEL_EEmBB,
    KERNEL_EEpBB,
    coupling_kernels,
    gauss_legendre_nodes,
    wigner_d_table,
)
from cmbcov.kernels.polspice import polspice_kernels  # noqa: E402
from cmbcov.keys import CovKeys  # noqa: E402
from cmbcov.postprocess import CovariancePostProcessor  # noqa: E402

pytestmark = pytest.mark.filterwarnings(
    "ignore:internal band-limit margin",
    "ignore:centralell=",
    "ignore::UserWarning",
)

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
SIX = ["TT", "EE", "BB", "TE", "TB", "EB"]
THETA_MAX_40DEG = np.deg2rad(40.0)

#: Spectra a run over each observable level involves at the postprocess
#: layer. Levels 3
#: and 4 differ only in which cross-parity CovKeys blocks are populated, not
#: in what pseudo_to_spice_bmode does with any one of them.
LEVEL_SPECTRA = {2: ["TT", "EE", "BB", "TE"], 3: SIX, 4: SIX}


# --------------------------------------------------------------------------- #
# Shared: a hand-rolled transform matrix, independent of postprocess.py
# --------------------------------------------------------------------------- #


def _sources(stokekey: str) -> tuple[str, ...]:
    return ("EE", "BB") if stokekey in ("EE", "BB") else (stokekey,)


def _kernel_for(
    output: str,
    source: str,
    tt: np.ndarray,
    cross: np.ndarray,
    plus: np.ndarray,
    minus: np.ndarray,
) -> np.ndarray:
    """
    The kernel ``output <- source`` for the docs/theory/bmode_kernels.md transform, built independently of
    ``CovariancePostProcessor._bmode_kernel`` from four named channels: an
    auto channel (``tt``, used only by TT), a cross channel (``cross``, used
    by TE/ET/TB/BT), and a plus/minus pair (``plus``/``minus``) for the
    EE/BB mix, whose difference is also the EB/BE channel. Passing the
    PolSpice ``G`` channels gives the ``T`` matrix of (a)/(b); passing the
    MASTER mean-coupling channels of ``coupling_kernels`` (combined as
    ``1/2(M1 +/- M2)``, see the module docstring) gives ``M`` for (b).
    """
    if output in ("EE", "BB"):
        if source not in ("EE", "BB"):
            raise ValueError(f"{output} only mixes with EE/BB, got {source}")
        return plus if output == source else minus
    if source != output:
        raise ValueError(f"{output} has no cross-spectrum source {source}")
    if output == "TT":
        return tt
    if output in ("TE", "ET", "TB", "BT"):
        return cross
    if output in ("EB", "BE"):
        return plus - minus
    raise ValueError(f"unknown spectrum {output!r}")


def _build_transform(
    spectra: list[str],
    tt: np.ndarray,
    cross: np.ndarray,
    plus: np.ndarray,
    minus: np.ndarray,
) -> np.ndarray:
    """The block matrix ``T`` of docs/theory/bmode_kernels.md over
    ``spectra``, each block ``n x n`` (``n = tt.shape[0]``), zero unless the
    column spectrum is a source of the row spectrum (:func:`_sources`)."""
    n = tt.shape[0]
    zero = np.zeros((n, n))
    rows = []
    for x in spectra:
        row = [
            _kernel_for(x, a, tt, cross, plus, minus) if a in _sources(x) else zero
            for a in spectra
        ]
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)


def _block(
    matrix: np.ndarray, spectra: list[str], n: int, x: str, y: str
) -> np.ndarray:
    ix, iy = spectra.index(x), spectra.index(y)
    return matrix[ix * n : (ix + 1) * n, iy * n : (iy + 1) * n]


# --------------------------------------------------------------------------- #
# (a) linear algebra: pseudo_to_spice_bmode == T Sigma T^T
# --------------------------------------------------------------------------- #


def _small_patch_wl(lmax_mask: int = 64, scale: float = 0.02, fsky: float = 0.04):
    """``w(theta) = fsky exp(-(1-cos theta)/scale)`` as a Legendre spectrum
    (the same analytic small-patch mask as ``tests/test_polspice_orientation.py``)."""
    mu, w = gauss_legendre_nodes(8 * lmax_mask + 128, -1.0)
    pl = wigner_d_table(lmax_mask, mu, 0, 0)
    return 2.0 * np.pi * (pl * w[:, None]).T @ (fsky * np.exp(-(1.0 - mu) / scale))


@pytest.mark.parametrize("level", [2, 3, 4])
def test_pseudo_to_spice_bmode_matches_hand_assembled_transform(level):
    lmax = 6
    n = lmax + 1
    wl = _small_patch_wl(lmax_mask=32)
    G = polspice_kernels(lmax, THETA_MAX_40DEG, wl=wl).as_covariance_array()
    post = CovariancePostProcessor(G)

    spectra = LEVEL_SPECTRA[level]
    T = _build_transform(spectra, G[0], G[3], G[1], G[2])

    rng = np.random.default_rng(1000 + level)
    nspec = len(spectra)
    mixing = rng.normal(size=(nspec * n, nspec * n))
    sigma = mixing @ mixing.T  # an arbitrary symmetric PSD pseudo covariance

    def provider(a: str, b: str) -> np.ndarray:
        ia, ib = spectra.index(a), spectra.index(b)
        return sigma[ia * n : (ia + 1) * n, ib * n : (ib + 1) * n]

    expected = T @ sigma @ T.T
    # sigma's entries are O(1e6-1e8) (an n*nspec-dimensional Wishart-like
    # draw); "to 1e-12" is relative float64 precision at that scale, not an
    # absolute tolerance.
    for x in spectra:
        for y in spectra:
            got = post.pseudo_to_spice_bmode(f"{x}x{y}", provider)
            np.testing.assert_allclose(
                got,
                _block(expected, spectra, n, x, y),
                rtol=1e-12,
                atol=1e-12 * np.abs(expected).max(),
            )


# --------------------------------------------------------------------------- #
# (b) decoupling: T @ M is block-diagonal in (EE, BB), TB<-TE and
# EB<-(EE, BB) vanish, and every diagonal block is a named PolSpice kernel
# --------------------------------------------------------------------------- #


def test_decoupling_transform_is_block_diagonal_and_matches_named_kernels():
    # Same recipe as tests/test_bmode_theta_combinations.py
    # test_tb_eb_mean_and_polspice_channels: a large lmax for the matrix
    # chain (the mean-coupling identities hold only once the intermediate
    # sum is not itself truncated by lmax), checked on an interior window
    # away from that truncation.
    lbig, lo, hi = 100, 2, 40
    wl = _small_patch_wl(lmax_mask=64)
    pk = polspice_kernels(lbig, THETA_MAX_40DEG, wl=wl)
    G = pk.as_covariance_array()
    M = coupling_kernels(wl, lbig)
    m_plus = 0.5 * (M[KERNEL_EEpBB] + M[KERNEL_EEmBB])
    m_minus = 0.5 * (M[KERNEL_EEpBB] - M[KERNEL_EEmBB])

    T = _build_transform(SIX, G[0], G[3], G[1], G[2])
    Mfull = _build_transform(SIX, M[KERNEL_TT], M[KERNEL_TE], m_plus, m_minus)
    n = lbig + 1
    TM = T @ Mfull

    def window(a: np.ndarray) -> np.ndarray:
        return a[lo : hi + 1, lo : hi + 1]

    def block(x: str, y: str) -> np.ndarray:
        return window(_block(TM, SIX, n, x, y))

    # Block-diagonal in (EE, BB): a real cancellation, not a zero by
    # construction (both G and M are generically dense here).
    scale = np.abs(block("EE", "EE")).max()
    residual_ee_bb = np.abs(block("EE", "BB")).max() / scale
    residual_bb_ee = np.abs(block("BB", "EE")).max() / scale
    assert residual_ee_bb < 1e-9, residual_ee_bb
    assert residual_bb_ee < 1e-9, residual_bb_ee

    # Zero by construction (each side's own single source), asserted for
    # documentation.
    assert np.all(block("TB", "TE") == 0.0)
    assert np.all(block("EB", "EE") == 0.0)
    assert np.all(block("EB", "BB") == 0.0)

    # Diagonal blocks match the named PolSpice kernels of Sect. 11.
    km2, kx, k0 = window(pk.Km2), window(pk.Kx), window(pk.K0)

    def rel(a: np.ndarray, b: np.ndarray) -> float:
        return np.abs(a - b).max() / np.abs(b).max()

    assert rel(block("TT", "TT"), k0) < 1e-11
    assert rel(block("TE", "TE"), kx) < 1e-11
    assert rel(block("TB", "TB"), kx) < 1e-11
    assert rel(block("EE", "EE"), km2) < 1e-11
    assert rel(block("BB", "BB"), km2) < 1e-11
    assert rel(block("EB", "EB"), km2) < 1e-11


# --------------------------------------------------------------------------- #
# (c) level 1 matches the recorded reference
# --------------------------------------------------------------------------- #


def test_level1_run_matches_the_recorded_reference(tmp_path):
    """
    The same reference ``tests/test_acc_bmode_assembly.py`` pins: included
    here too since PolSpice post-processing support changed
    ``compute_covariance_matrix``'s loop, not just the B-mode branch.

    Same tolerance and reasoning as
    ``tests/test_acc_bmode_assembly.py::test_level1_run_matches_the_recorded_reference``:
    the reference was recorded on a different machine, so the comparison
    must survive a different BLAS/SHT rounding path.
    """
    reference = np.load(
        os.path.join(os.path.dirname(__file__), "reference", "acc_level1_reference.npz")
    )
    current = build_level1_reference(str(tmp_path))
    assert sorted(reference.files) == sorted(current)
    for name in reference.files:
        np.testing.assert_allclose(
            current[name],
            reference[name],
            rtol=1e-10,
            atol=1e-12 * np.abs(reference[name]).max(),
            err_msg=name,
        )


# --------------------------------------------------------------------------- #
# (d) end to end, polspice_postprocess: true
# --------------------------------------------------------------------------- #

E2E_NBINS = 6


def _write_cls(path, nonzero_odd):
    n = 80
    ell = np.arange(n)
    tt = np.zeros(n)
    tt[2:] = 1e3 / (ell[2:] * (ell[2:] + 1)) ** 0.9
    ee, bb = 0.1 * tt, 0.02 * tt
    te = 0.5 * np.sqrt(tt * ee)
    tb = 0.05 * np.sqrt(tt * bb) if nonzero_odd else 0 * tt
    eb = 0.05 * np.sqrt(ee * bb) if nonzero_odd else 0 * tt
    np.savetxt(path, np.column_stack([ell, tt, ee, bb, te, tb, eb, te, tb, eb]))


def _generator_run(workdir, observables, parity_mixed, nonzero_odd, freqs):
    shutil.copy(os.path.join(DATA, "baseline_mask.fits"), workdir)
    _write_cls(os.path.join(workdir, "cls.dat"), nonzero_odd)
    params = {
        "cov_path": os.path.join(workdir, "out"),
        "cov_name": "cov.dat",
        "frequencies": freqs,
        "observables": observables,
        "lmax": 20,
        "lmin": 2,
        "bins": [[2, 20, 3]],
        "mask_name": "baseline_mask.fits",
        "mask_path": workdir,
        "covariance_approximation": "acc",
        "dmax": 3,
        "centralell": 10,
        "cmb_spectrum": os.path.join(workdir, "cls.dat"),
        "beams": dict.fromkeys(freqs, 5.0),
        "pixwin": 32,
        "nl": {f + f: 10.0 for f in freqs},
        "parity_mixed_blocks": parity_mixed,
        "polspice_postprocess": True,
        "acc_precompute": {"nside": 16, "grid": "gl", "lw": 10},
    }
    path = os.path.join(workdir, "params.yml")
    with open(path, "w") as handle:
        yaml.dump(params, handle)
    CovarianceMatrixGenerator(path).precompute_acc_kernels()
    generator = CovarianceMatrixGenerator(path)
    generator.run_full_analysis()
    version = sorted(glob.glob(os.path.join(workdir, "out", "v*")))[-1]
    matrix = np.loadtxt(os.path.join(version, "cov.dat"))
    return generator, matrix


@pytest.mark.parametrize(
    "level, observables, parity_mixed, nonzero_odd",
    [
        (2, ["TT", "EE", "TE", "BB"], False, False),
        (3, SIX, False, False),
        (4, SIX, True, False),
    ],
    ids=["level2", "level3", "level4"],
)
def test_generator_end_to_end_with_polspice_postprocess(
    tmp_path, level, observables, parity_mixed, nonzero_odd
):
    generator, matrix = _generator_run(
        str(tmp_path), observables, parity_mixed, nonzero_odd, ["090GHz"]
    )
    specs = generator.cov_keys.spec_keys
    assert matrix.shape == (len(specs) * E2E_NBINS,) * 2
    assert np.isfinite(matrix).all()
    scale = np.abs(matrix).max()
    assert np.abs(matrix - matrix.T).max() <= 1e-12 * scale
    assert (np.diag(matrix) > 0).all()

    # Reported, not asserted: ACC + PolSpice post-processing is not
    # guaranteed positive semi-definite.
    eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    print(
        f"level {level} postprocessed: smallest eigenvalue {eigenvalues.min():.3e}, "
        f"largest {eigenvalues.max():.3e}, ratio {eigenvalues.min() / eigenvalues.max():.2e}"
    )


# --------------------------------------------------------------------------- #
# (e) validation rule
# --------------------------------------------------------------------------- #


def _base_params(**overrides):
    params = {
        "cov_path": "/tmp/does-not-matter",
        "cov_name": "cov.dat",
        "frequencies": ["090GHz"],
        "lmax": 20,
        "lmin": 2,
        "bins": [[2, 20, 6]],
        "mask_name": "baseline_mask.fits",
        "mask_path": DATA,
        "covariance_approximation": "acc",
        "dmax": 1,
        "centralell": 10,
    }
    params.update(overrides)
    return params


def _errors(params):
    validator = ParameterValidator()
    validator.validate(params)
    return validator.errors


def test_ee_and_eb_without_bb_is_rejected_under_postprocess():
    params = _base_params(observables=["TT", "EE", "TE", "EB"])
    errors = _errors(params)
    assert any("BB" in e and "polspice_postprocess" in e for e in errors)


def test_ee_and_tb_without_bb_is_rejected_under_postprocess():
    params = _base_params(observables=["TT", "EE", "TE", "TB"])
    errors = _errors(params)
    assert any("BB" in e and "polspice_postprocess" in e for e in errors)


def test_ee_and_eb_without_bb_passes_with_postprocess_off():
    params = _base_params(
        observables=["TT", "EE", "TE", "EB"], polspice_postprocess=False
    )
    errors = _errors(params)
    assert not any("polspice_postprocess" in e and "BB" in e for e in errors)


def test_ee_bb_and_eb_passes_under_postprocess():
    params = _base_params(observables=["TT", "EE", "TE", "BB", "EB"])
    errors = _errors(params)
    assert not any("polspice_postprocess" in e and "BB" in e for e in errors)


def test_bb_without_ee_is_not_flagged_by_this_rule():
    """Outside the documented levels (BB only ever appears alongside EE),
    so the rule is deliberately one-sided; ``Cov`` itself
    still refuses this at compute time (see the next test)."""
    params = _base_params(observables=["TT", "BB"])
    errors = _errors(params)
    assert not any("polspice_postprocess" in e and "BB" in e for e in errors)


def test_cov_refuses_ee_and_eb_without_bb_at_compute_time(tmp_path):
    """The generator-level rule has a matching guard directly on ``Cov``,
    exercised here without going through a parameter file."""
    from cmbcov.covariance import (
        Cov,
        CovarianceConfig,
        CovarianceMethod,
    )

    shutil.copy(os.path.join(DATA, "baseline_mask.fits"), str(tmp_path))
    config = CovarianceConfig(
        method=CovarianceMethod.ACC,
        lmax=10,
        lmin=2,
        dmax=1,
        centralell=5,
        polspice_postprocess=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cov = Cov(
            "baseline_mask.fits",
            config=config,
            mask_path=str(tmp_path),
            save_dir=str(tmp_path),
        )
    keys = CovKeys([], ["090GHz"], observables=["TT", "EE", "TE", "EB"])
    with pytest.raises(ValueError, match="polspice_postprocess needs the pseudo BB"):
        cov.compute_covariance_matrix([2, 10], keys, cl={})
