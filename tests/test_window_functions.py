r"""
Bandpower window functions, ``--save-windows``
(:mod:`cmbcov.windows`).

(a) **Maths.** :func:`~cmbcov.windows.build_window_functions`
    applied to a smooth input spectrum reproduces the expected binned mean
    computed independently: the pseudo mean is built from the MASTER
    coupling (``kernels/coupling.py``), then the PolSpice transform, D_ell
    scaling, an instrumental (beam/pixel-window) leg factor and binning are
    applied explicitly (``CovariancePostProcessor``/``BinningManager``),
    exactly mirroring docs/theory/bmode_kernels.md Sect. 7 and
    ``CovariancePostProcessor.apply_debiasing``. Covers ``polspice_postprocess``
    true and false, and the EE/BB mixing window a B-mode run (``BB`` among
    the observables) needs. With a data model (beam, pixel window) on the
    input axis and its inverse on the output axis, ``W`` maps the *unbeamed*
    sky spectrum to the debiased bandpower.
(a') **Beams on, end to end.** On the baseline run (5' and 2' beams, nside-32
    pixel window) every written window, applied to the unbeamed spectrum,
    equals the package's mean debiased bandpower
    ``Bin diag(dl d) K (B_A B_B p F C)`` to 1e-12, with the beams and pixel
    window rebuilt independently of ``SpectraLoader``.
(b) **End to end.** ``compute-covariance --save-windows`` writes one
    plain-text file per spectrum and frequency pair under a ``windows/``
    subdirectory beside the covariance -- the owner's existing convention.
(c) **Unaffected default.** A run without ``--save-windows`` writes nothing
    and its covariance matches the recorded golden baseline
    (``tests/test_end_to_end_baseline.py``).
(d) **Refusal.** A run with ``polspice_postprocess: false`` and both ``EE``
    and ``BB`` among the observables raises instead of writing anything: the
    genuine EE<->BB mixing of the undecoupled pseudo-spectra cannot be
    represented one spectrum per file.
"""

import glob
import os
import shutil

import numpy as np
import pytest
import yaml

healpy = pytest.importorskip("healpy")

from cmbcov import CovarianceMatrixGenerator  # noqa: E402
from cmbcov.binning import BinningManager  # noqa: E402
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
from cmbcov.postprocess import CovariancePostProcessor  # noqa: E402
from cmbcov.scripts.compute_covariance import main  # noqa: E402
from cmbcov.windows import build_window_functions  # noqa: E402

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
THETA_MAX_40DEG = np.deg2rad(40.0)


# --------------------------------------------------------------------------- #
# (a) maths: build_window_functions vs. an independently assembled chain
# --------------------------------------------------------------------------- #


def _small_patch_wl(lmax_mask: int = 64, scale: float = 0.02, fsky: float = 0.04):
    """Analytic small-patch mask power spectrum, as in ``test_polspice_bmode.py``."""
    mu, w = gauss_legendre_nodes(8 * lmax_mask + 128, -1.0)
    pl = wigner_d_table(lmax_mask, mu, 0, 0)
    return 2.0 * np.pi * (pl * w[:, None]).T @ (fsky * np.exp(-(1.0 - mu) / scale))


class _FakeCov:
    """
    A minimal stand-in for ``Cov``: just the two attributes
    :func:`~cmbcov.windows.build_window_functions` reads
    (:attr:`Cov.K`, :attr:`Cov.M`, :attr:`Cov.binning_manager`), built
    directly from the native kernel functions rather than a real mask alm
    solve -- the same shortcut ``test_polspice_bmode.py`` uses.
    """

    def __init__(self, lbig: int, lmin: int = 2, lmax_mask: int = 64):
        wl = _small_patch_wl(lmax_mask=lmax_mask)
        pk = polspice_kernels(lbig, THETA_MAX_40DEG, wl=wl)
        self.K = pk.as_master_array("K")
        self.M = coupling_kernels(wl, lbig)
        self.G = pk.as_covariance_array()
        self.binning_manager = BinningManager(lbig + 1, lmin)


#: A large band limit, with the test spectrum's support stopping well short
#: of it (``_smooth_cl``), so that the MASTER coupling's mixing across
#: multipoles is not itself truncated by lbig -- see
#: ``test_polspice_bmode.py``'s ``test_decoupling_transform_is_...`` for the
#: same precaution. Comparing to a spectrum with support all the way to
#: lbig (or a small lbig) leaves a percent-level truncation residual: real
#: physics, not a bug, so it is deliberately avoided here.
LBIG = 140
CL_SUPPORT = 60
LBINS = [2, 10, 20, 30, 40]


def _smooth_cl(lbig: int, support: int = CL_SUPPORT) -> np.ndarray:
    cl = np.zeros(lbig + 1)
    ell = np.arange(lbig + 1)
    cl[2:support] = 1e3 / (ell[2:support] * (ell[2:support] + 1)) ** 0.9
    return cl


def _fake_leg_factor(lbig: int) -> np.ndarray:
    """
    A synthetic per frequency-pair instrumental leg factor standing in for
    ``debiasing_dict[freq][stoke]`` (beam, pixel window, transfer function,
    calibration; ``SpectraLoader.data_model``/``debiasing_dict``): a smooth,
    non-trivial (not just a constant) function of ``ell`` so that omitting it
    would change the answer, built from a Gaussian beam (as
    :meth:`~cmbcov.spectra.SpectraLoader._load_beams`
    would for two different frequencies) times a smooth pixel-window-like
    roll-off.
    """
    ell = np.arange(lbig + 1)
    beam_1 = healpy.gauss_beam(np.deg2rad(30.0 / 60.0), lbig)
    beam_2 = healpy.gauss_beam(np.deg2rad(10.0 / 60.0), lbig)
    pixwin = 1.0 - 0.1 * (ell / lbig) ** 2
    return beam_1 * beam_2 * pixwin


def test_tt_window_matches_master_coupling_then_polspice_chain():
    cov = _FakeCov(LBIG)
    cl = _smooth_cl(LBIG)
    leg = _fake_leg_factor(LBIG)

    windows = build_window_functions(
        ["TT"],
        cov,
        LBINS,
        Dl=True,
        polspice_postprocess=True,
        row_scale={"TT": leg},
    )
    got = windows["W_TT"] @ cl

    post = CovariancePostProcessor(cov.G)
    pseudo_mean = cov.M[KERNEL_TT] @ cl
    polspice_mean = post.G_kernels["TT"] @ pseudo_mean
    ell = np.arange(LBIG + 1)
    dl_factor = ell * (ell + 1) / (2 * np.pi)
    bin_matrix, _ = cov.binning_manager.create_bin_matrix(
        LBINS, flatten_with_ell_factor=False
    )
    expected = bin_matrix @ (leg * dl_factor * polspice_mean)

    np.testing.assert_allclose(got, expected, rtol=1e-8, atol=0.0)

    # The leg factor is not a numerical no-op: dropping it changes the
    # answer (it is not close to a constant, per the beam/pixel-window
    # shape above).
    without_leg = (
        build_window_functions(["TT"], cov, LBINS, Dl=True, polspice_postprocess=True)[
            "W_TT"
        ]
        @ cl
    )
    assert np.abs(got - without_leg).max() > 1e-3 * np.abs(got).max()


def test_te_window_matches_master_coupling_then_polspice_chain():
    cov = _FakeCov(LBIG)
    te = 0.3 * _smooth_cl(LBIG)
    leg = _fake_leg_factor(LBIG)

    windows = build_window_functions(
        ["TE"],
        cov,
        LBINS,
        Dl=False,
        polspice_postprocess=True,
        row_scale={"TE": leg},
    )
    got = windows["W_TE"] @ te

    post = CovariancePostProcessor(cov.G)
    pseudo_mean = cov.M[KERNEL_TE] @ te
    polspice_mean = post.G_kernels["TE"] @ pseudo_mean
    bin_matrix, _ = cov.binning_manager.create_bin_matrix(
        LBINS, flatten_with_ell_factor=True
    )
    expected = bin_matrix @ (leg * polspice_mean)

    np.testing.assert_allclose(got, expected, rtol=1e-8, atol=0.0)


def test_decoupled_ee_bb_window_has_no_mixing():
    """
    ``polspice_postprocess: true``: EE and BB (and EB) all respond to the
    true spectrum through the same -2K, with no EE<->BB mixing in the mean
    (docs/theory/bmode_kernels.md Sect. 7) -- so no ``W_EE_from_BB`` key is
    produced, and the EE window built from the MASTER-coupling-then-PolSpice
    chain agrees with the direct -2K one even when C^BB is non-zero.
    """
    cov = _FakeCov(LBIG)
    cl_ee = _smooth_cl(LBIG)
    cl_bb = 0.3 * cl_ee

    windows = build_window_functions(
        ["EE", "BB"], cov, LBINS, Dl=False, polspice_postprocess=True
    )
    assert "W_EE_from_BB" not in windows
    assert "W_BB_from_EE" not in windows
    got_ee = windows["W_EE"] @ cl_ee

    post = CovariancePostProcessor(cov.G)
    m_plus = 0.5 * (cov.M[KERNEL_EEpBB] + cov.M[KERNEL_EEmBB])
    m_minus = 0.5 * (cov.M[KERNEL_EEpBB] - cov.M[KERNEL_EEmBB])
    pseudo_ee_mean = m_plus @ cl_ee + m_minus @ cl_bb
    pseudo_bb_mean = m_plus @ cl_bb + m_minus @ cl_ee
    # The mean-level identity is linear, not a covariance transform, so
    # apply the (+, -) G kernels directly rather than through
    # pseudo_to_spice_bmode (a covariance-only API): hat_C^EE = +G C~^EE +
    # -G C~^BB (docs/theory/bmode_kernels.md Sect. 7).
    hat_ee = post.G_kernels["EE"] @ pseudo_ee_mean + post.G_mixing @ pseudo_bb_mean
    bin_matrix, _ = cov.binning_manager.create_bin_matrix(
        LBINS, flatten_with_ell_factor=True
    )
    expected_ee = bin_matrix @ hat_ee

    np.testing.assert_allclose(got_ee, expected_ee, rtol=1e-8, atol=0.0)


def test_undecoupled_ee_bb_window_has_master_mixing():
    """``polspice_postprocess: false``: the reported spectrum is the pseudo
    C_l itself, whose EE mean genuinely mixes in C^BB through M- (and vice
    versa) -- ``W_EE_from_BB``/``W_BB_from_EE`` reproduce that mixing
    exactly (both built from the same M, M- machine-precision consistent)."""
    cov = _FakeCov(LBIG)
    cl_ee = _smooth_cl(LBIG)
    cl_bb = 0.3 * cl_ee

    windows = build_window_functions(
        ["EE", "BB"], cov, LBINS, Dl=False, polspice_postprocess=False
    )
    got_ee = windows["W_EE"] @ cl_ee + windows["W_EE_from_BB"] @ cl_bb
    got_bb = windows["W_BB"] @ cl_bb + windows["W_BB_from_EE"] @ cl_ee

    m_plus = 0.5 * (cov.M[KERNEL_EEpBB] + cov.M[KERNEL_EEmBB])
    m_minus = 0.5 * (cov.M[KERNEL_EEpBB] - cov.M[KERNEL_EEmBB])
    bin_matrix, _ = cov.binning_manager.create_bin_matrix(
        LBINS, flatten_with_ell_factor=True
    )
    expected_ee = bin_matrix @ (m_plus @ cl_ee + m_minus @ cl_bb)
    expected_bb = bin_matrix @ (m_plus @ cl_bb + m_minus @ cl_ee)

    np.testing.assert_allclose(got_ee, expected_ee, rtol=1e-10, atol=0.0)
    np.testing.assert_allclose(got_bb, expected_bb, rtol=1e-10, atol=0.0)

    # And the mixing is not a numerical no-op: it changes the answer.
    ee_without_mixing = windows["W_EE"] @ cl_ee
    assert np.abs(got_ee - ee_without_mixing).max() > 1e-6 * np.abs(got_ee).max()


def test_undecoupled_ee_bb_mixing_window_uses_the_output_legs_own_row_scale():
    """
    ``row_scale`` for the mixing term is keyed by the *output* spectrum
    (``"EE"`` for ``W_EE_from_BB``, ``"BB"`` for ``W_BB_from_EE``), matching
    ``CovariancePostProcessor.apply_debiasing``, which scales a covariance
    block by its output leg alone.
    """
    cov = _FakeCov(LBIG)
    leg_ee = _fake_leg_factor(LBIG)
    leg_bb = 0.5 * leg_ee

    windows = build_window_functions(
        ["EE", "BB"],
        cov,
        LBINS,
        Dl=False,
        polspice_postprocess=False,
        row_scale={"EE": leg_ee, "BB": leg_bb},
    )

    bin_matrix, _ = cov.binning_manager.create_bin_matrix(
        LBINS, flatten_with_ell_factor=True
    )
    m_minus = 0.5 * (cov.M[KERNEL_EEpBB] - cov.M[KERNEL_EEmBB])
    expected_ee_from_bb = bin_matrix @ (leg_ee[:, None] * m_minus)
    expected_bb_from_ee = bin_matrix @ (leg_bb[:, None] * m_minus)

    np.testing.assert_allclose(
        windows["W_EE_from_BB"], expected_ee_from_bb, rtol=1e-10, atol=0.0
    )
    np.testing.assert_allclose(
        windows["W_BB_from_EE"], expected_bb_from_ee, rtol=1e-10, atol=0.0
    )


def test_eb_window_uses_the_minus_two_m_channel():
    """EB never decouples and never mixes with EE/BB
    (docs/theory/bmode_kernels.md Sect. 7): its mean kernel is M- alone
    under ``polspice_postprocess: false``."""
    cov = _FakeCov(LBIG)
    cl_eb = 0.1 * _smooth_cl(LBIG)

    windows = build_window_functions(
        ["EB"], cov, LBINS, Dl=False, polspice_postprocess=False
    )
    got = windows["W_EB"] @ cl_eb

    bin_matrix, _ = cov.binning_manager.create_bin_matrix(
        LBINS, flatten_with_ell_factor=True
    )
    expected = bin_matrix @ (cov.M[KERNEL_EEmBB] @ cl_eb)

    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=0.0)


def _fake_data_model(lbig: int, stokes: str = "TT") -> np.ndarray:
    """
    A synthetic data model ``B_A B_B p F`` (``SpectraLoader.data_model``):
    two Gaussian beams, the HEALPix nside-64 pixel window of the Stokes pair
    (``p_T^2``, ``p_T p_P`` or ``p_P^2``, as ``_load_pixel_window``), and a
    smooth transfer function.
    """
    ell = np.arange(lbig + 1)
    beam_1 = healpy.gauss_beam(np.deg2rad(30.0 / 60.0), lbig)
    beam_2 = healpy.gauss_beam(np.deg2rad(10.0 / 60.0), lbig)
    pix_t, pix_p = healpy.pixwin(64, pol=True, lmax=lbig)
    pix = {"TT": pix_t**2, "TE": pix_t * pix_p, "EE": pix_p**2, "BB": pix_p**2}
    fl = 1.0 - 0.2 * (ell / lbig)
    return beam_1 * beam_2 * pix[stokes] * fl


def _debias(model: np.ndarray) -> np.ndarray:
    """``1/model`` with the package's ``safe_divide`` default of 1 at zeros
    (the pixel window ``p_P`` vanishes at ``ell < 2``)."""
    out = np.ones_like(model)
    np.divide(1.0, model, out=out, where=model != 0.0)
    return out


@pytest.mark.parametrize("stokes", ["TT", "TE", "EE"])
def test_beamed_window_maps_the_unbeamed_spectrum_to_the_debiased_bandpower(
    stokes,
):
    """
    With ``column_scale = data_model`` and ``row_scale = 1/data_model``,
    ``W`` applied to the *unbeamed* spectrum equals the debiased bandpower of
    the beamed sky, assembled independently: MASTER coupling of the beamed
    spectrum ``data_model * C``, the PolSpice transform, debiasing,
    ``D_ell`` scaling and binning. Without ``column_scale`` it is off by
    about the binned ``1/data_model`` (the pre-2026-09-28 windows).
    """
    cov = _FakeCov(LBIG)
    cl = _smooth_cl(LBIG) * (0.3 if stokes == "TE" else 1.0)
    model = _fake_data_model(LBIG, stokes)
    debias = _debias(model)
    kernel = {"TT": KERNEL_TT, "TE": KERNEL_TE}.get(stokes)

    windows = build_window_functions(
        [stokes],
        cov,
        LBINS,
        Dl=True,
        polspice_postprocess=True,
        row_scale={stokes: debias},
        column_scale={stokes: model},
    )
    got = windows[f"W_{stokes}"] @ cl

    post = CovariancePostProcessor(cov.G)
    beamed = model * cl
    if stokes == "EE":
        m_plus = 0.5 * (cov.M[KERNEL_EEpBB] + cov.M[KERNEL_EEmBB])
        # C^BB = 0: the -G mixing of the (zero) pseudo-BB mean drops out,
        # but the pseudo-BB mean itself is M- (model C^EE), not zero.
        m_minus = 0.5 * (cov.M[KERNEL_EEpBB] - cov.M[KERNEL_EEmBB])
        decoupled = post.G_kernels["EE"] @ (m_plus @ beamed) + post.G_mixing @ (
            m_minus @ beamed
        )
    else:
        decoupled = post.G_kernels[stokes] @ (cov.M[kernel] @ beamed)
    ell = np.arange(LBIG + 1)
    dl_factor = ell * (ell + 1) / (2 * np.pi)
    bin_matrix, _ = cov.binning_manager.create_bin_matrix(
        LBINS, flatten_with_ell_factor=False
    )
    expected = bin_matrix @ (debias * dl_factor * decoupled)

    np.testing.assert_allclose(got, expected, rtol=1e-8, atol=0.0)

    # The input-axis factor is not a numerical no-op here.
    row_only = (
        build_window_functions(
            [stokes],
            cov,
            LBINS,
            Dl=True,
            polspice_postprocess=True,
            row_scale={stokes: debias},
        )[f"W_{stokes}"]
        @ cl
    )
    assert np.abs(row_only / expected - 1.0).max() > 1e-2


def test_undecoupled_mixing_window_uses_the_input_spectrums_column_scale():
    """
    ``column_scale`` for the EE<->BB mixing term is keyed by the *input*
    spectrum (``"BB"`` for ``W_EE_from_BB``, ``"EE"`` for ``W_BB_from_EE``):
    the pseudo-EE mean is ``M+ (D_EE C^EE) + M- (D_BB C^BB)``.
    """
    cov = _FakeCov(LBIG)
    cl_ee = _smooth_cl(LBIG)
    cl_bb = 0.3 * cl_ee
    model_ee = _fake_data_model(LBIG, "EE")
    model_bb = 0.7 * model_ee  # distinct, so a swapped key would show
    debias_ee, debias_bb = _debias(model_ee), _debias(model_bb)

    windows = build_window_functions(
        ["EE", "BB"],
        cov,
        LBINS,
        Dl=False,
        polspice_postprocess=False,
        row_scale={"EE": debias_ee, "BB": debias_bb},
        column_scale={"EE": model_ee, "BB": model_bb},
    )
    got_ee = windows["W_EE"] @ cl_ee + windows["W_EE_from_BB"] @ cl_bb
    got_bb = windows["W_BB"] @ cl_bb + windows["W_BB_from_EE"] @ cl_ee

    m_plus = 0.5 * (cov.M[KERNEL_EEpBB] + cov.M[KERNEL_EEmBB])
    m_minus = 0.5 * (cov.M[KERNEL_EEpBB] - cov.M[KERNEL_EEmBB])
    bin_matrix, _ = cov.binning_manager.create_bin_matrix(
        LBINS, flatten_with_ell_factor=True
    )
    pseudo_ee = m_plus @ (model_ee * cl_ee) + m_minus @ (model_bb * cl_bb)
    pseudo_bb = m_plus @ (model_bb * cl_bb) + m_minus @ (model_ee * cl_ee)
    expected_ee = bin_matrix @ (debias_ee * pseudo_ee)
    expected_bb = bin_matrix @ (debias_bb * pseudo_bb)

    np.testing.assert_allclose(got_ee, expected_ee, rtol=1e-10, atol=0.0)
    np.testing.assert_allclose(got_bb, expected_bb, rtol=1e-10, atol=0.0)


def test_column_scale_shorter_than_lmax_is_refused():
    cov = _FakeCov(LBIG)
    with pytest.raises(ValueError, match="column_scale"):
        build_window_functions(
            ["TT"],
            cov,
            LBINS,
            Dl=True,
            polspice_postprocess=True,
            column_scale={"TT": np.ones(LBIG)},
        )


# --------------------------------------------------------------------------- #
# (b)/(c) end to end: the CLI flag, the file convention, and the unaffected
# default
# --------------------------------------------------------------------------- #


def _write_baseline_params(tmp_path):
    template = open(os.path.join(DATA, "baseline_params.yml")).read()
    params_path = tmp_path / "params.yml"
    params_path.write_text(
        template.replace("PLACEHOLDER_OUT", str(tmp_path)).replace(
            "PLACEHOLDER_DATA", DATA
        )
    )
    return str(params_path)


def _latest_version_dir(tmp_path):
    return sorted(glob.glob(os.path.join(str(tmp_path), "v*")))[-1]


#: baseline_params.yml is a T/E-only NKA run at 090GHz/150GHz: TT, TE, EE at
#: each auto-frequency pair, plus ET at the (only) cross-frequency pair
#: (CovKeys.spec_keys collapses TE/ET at equal frequencies -- see
#: keys.SpecKey.__eq__).
_BASELINE_EXPECTED_FILES = {
    "TT_90x90_window_functions.txt",
    "TE_90x90_window_functions.txt",
    "EE_90x90_window_functions.txt",
    "TT_150x150_window_functions.txt",
    "TE_150x150_window_functions.txt",
    "EE_150x150_window_functions.txt",
    "TT_90x150_window_functions.txt",
    "TE_90x150_window_functions.txt",
    "ET_90x150_window_functions.txt",
    "EE_90x150_window_functions.txt",
}


def test_cli_save_windows_writes_one_file_per_spectrum_and_frequency_pair(tmp_path):
    params = _write_baseline_params(tmp_path)
    rc = main(["--parameter-file", params, "--save-windows"])
    assert rc == 0

    version = _latest_version_dir(tmp_path)
    windows_dir = os.path.join(version, "windows")
    assert os.path.isdir(windows_dir)
    assert set(os.listdir(windows_dir)) == _BASELINE_EXPECTED_FILES

    n_bins = len(np.loadtxt(os.path.join(version, "lbins.dat")))
    lmax = 60  # baseline_params.yml
    lmin = 2

    path = os.path.join(windows_dir, "EE_90x150_window_functions.txt")
    with open(path) as handle:
        first_line = handle.readline()
    assert first_line == "# Band powers window functions for EE 090GHz150GHz.\n"

    table = np.loadtxt(path)
    assert table.shape == (lmax - lmin, 1 + n_bins)
    np.testing.assert_array_equal(table[:, 0], np.arange(lmin, lmax))
    assert np.isfinite(table).all()


def test_cli_save_windows_reproduces_the_chain_for_one_frequency_pair(tmp_path):
    """
    The written ``TT_90x150`` window, applied to a true-sky spectrum,
    reproduces the same mask+PolSpice+beam/pixel-window+debiasing chain
    :func:`~cmbcov.windows.build_window_functions`
    documents, assembled here directly from the generator's own public
    attributes (``covariance_instance.K``, ``debiasing_dict``,
    ``data_model``) -- an independent re-derivation, not a re-run of the
    library call the generator itself made. The data model sits on the
    input axis (the covariance is computed for ``data_model * C``), the
    debiasing on the output axis.
    """
    params = _write_baseline_params(tmp_path)
    generator = CovarianceMatrixGenerator(params)
    generator.run_full_analysis(save_windows=True)

    version = _latest_version_dir(tmp_path)
    path = os.path.join(version, "windows", "TT_90x150_window_functions.txt")
    table = np.loadtxt(path)
    ell_file = table[:, 0].astype(int)
    w_file = table[:, 1:]  # (n_ell, n_bins)

    cov = generator.covariance_instance
    lmax = cov.lmax
    lmin = generator.config.lmin
    leg = generator.debiasing_dict["090GHz150GHz"]["TT"]
    model = np.asarray(generator.data_model["090GHz150GHz"]["TT"])[:lmax]
    ell = np.arange(lmax)
    dl = ell * (ell + 1) / (2 * np.pi)  # baseline_params.yml has Dl: True
    bin_matrix, _ = cov.binning_manager.create_bin_matrix(
        generator._setup_bins(), flatten_with_ell_factor=False
    )
    w_expected = (
        bin_matrix @ (leg[:, None] * dl[:, None] * cov.K[0] * model[None, :])
    ).T

    np.testing.assert_array_equal(ell_file, np.arange(lmin, lmax))
    np.testing.assert_allclose(w_file, w_expected[lmin:], rtol=1e-10, atol=0.0)

    cl_true = generator.cl_dict["090GHz150GHz"]["TT"][:lmax]
    # baseline_cls.dat starts at ell=2 (lmin); the loader zero-pads ell < 2,
    # so restricting the dot product's input axis to what the file stores
    # (ell >= lmin) drops only exact zeros and reproduces the full-axis
    # bandpower exactly.
    np.testing.assert_array_equal(cl_true[:lmin], 0.0)
    reported = w_expected.T @ cl_true
    got = w_file.T @ cl_true[lmin:]
    np.testing.assert_allclose(got, reported, rtol=1e-8, atol=0.0)


#: baseline_params.yml's instrument, restated here so that the beamed test
#: below rebuilds the data model without going through SpectraLoader.
_BASELINE_FWHM_ARCMIN = {"090GHz": 5.0, "150GHz": 2.0}
_BASELINE_PIXWIN_NSIDE = 32
_BASELINE_PAIRS = ("090GHz090GHz", "090GHz150GHz", "150GHz150GHz")


def _independent_data_model(freq_1: str, freq_2: str, stokes: str, lmax: int):
    """``B_A B_B p`` of the baseline run from healpy directly (``fl`` = 1)."""
    beam_1, beam_2 = (
        healpy.gauss_beam(np.deg2rad(_BASELINE_FWHM_ARCMIN[f] / 60.0), lmax - 1)
        for f in (freq_1, freq_2)
    )
    pix_t, pix_p = healpy.pixwin(_BASELINE_PIXWIN_NSIDE, pol=True, lmax=lmax - 1)
    pix = {"T": pix_t, "E": pix_p}
    return beam_1 * beam_2 * pix[stokes[0]] * pix[stokes[1]]


def _tf_and_correction_columns(pair_index: int, n: int):
    """
    Synthetic ``fl`` and ``post_process_correction`` columns (TT, EE, BB, TE,
    the 4-column file layout) for one frequency pair: smooth, distinct per
    pair and per Stokes pair, ``fl`` well inside ``[0, 1]`` so that the
    ``add_tf_uncertainty`` inflation ``1 + sqrt((1 - fl)/3999)`` is up to
    1.1 %, far above the test tolerance.
    """
    x = np.arange(n) / 60.0
    fl = {
        "TT": 0.55 + 0.4 * np.cos(1.3 * x + 0.2 * pair_index) ** 2,
        "EE": 0.6 + 0.3 * np.exp(-x - 0.1 * pair_index),
        "BB": 0.7 * np.ones(n),
        "TE": 0.5 + 0.45 * np.sin(0.9 * x + 0.3 + 0.2 * pair_index) ** 2,
    }
    ppc = {
        "TT": 1.0 + 0.05 * np.sin(2.0 * x + pair_index),
        "EE": 1.0 - 0.04 * x,
        "BB": np.ones(n),
        "TE": 1.0 + 0.03 * np.cos(3.0 * x + 0.5 * pair_index),
    }
    return fl, ppc


def _write_tf_and_correction_files(directory) -> tuple[str, str]:
    """Write one ``fl`` and one ``post_process_correction`` file per pair,
    ``ell`` 0 to 99; return the two ``{}`` filename templates."""
    n = 100
    ell = np.arange(n)
    for index, pair in enumerate(_BASELINE_PAIRS):
        fl, ppc = _tf_and_correction_columns(index, n)
        for name, table in (("fl", fl), ("ppc", ppc)):
            np.savetxt(
                os.path.join(str(directory), f"{name}_{pair}.dat"),
                np.column_stack([ell] + [table[k] for k in ("TT", "EE", "BB", "TE")]),
            )
    return (
        os.path.join(str(directory), "fl_{}.dat"),
        os.path.join(str(directory), "ppc_{}.dat"),
    )


@pytest.mark.parametrize("covariance_only_factors", [False, True])
def test_cli_save_windows_with_beams_map_the_unbeamed_spectrum_to_the_mean(
    tmp_path, covariance_only_factors
):
    """
    Beams (5' and 2') and the nside-32 pixel window on: every written window,
    applied to the *unbeamed* fiducial spectrum, equals the mean of the
    debiased *data* bandpower

        Bin diag(dl / D) K (D C),   D = B_A B_B p F,

    to 1e-12 of its largest element.
    The covariance is computed for ``cl_dict_biased = D C``. ``B_A B_B p`` is
    rebuilt here from healpy, not read from ``SpectraLoader``; the TE/ET/EE
    pixel windows (``p_T p_P``, ``p_P^2``) and the cross-frequency beams
    ``B_90 B_150`` are checked. Before 2026-09-28 the windows lacked the
    input-axis factor and ``W C`` exceeded this mean by 1.003 to 1.30 per bin.

    ``covariance_only_factors=True`` adds a transfer function ``F``, a
    post-processing correction ``c`` (both per pair and per Stokes pair) and
    ``add_tf_uncertainty: true``. ``c`` and the inflation
    ``1 + sqrt((1 - F)/3999)`` multiply the covariance's ``debiasing_dict``
    but are never applied to the data, so the windows leave both out
    (owner's decision, 2026-09-29), while the covariance legs keep both
    (checked against an independent ``c (1 + sqrt((1 - F)/3999)) / D``).
    With neither set, the windows' output factor is bit for bit
    ``debiasing_dict``.
    """
    params = _write_baseline_params(tmp_path)
    extra = {}
    if covariance_only_factors:
        fl_template, ppc_template = _write_tf_and_correction_files(tmp_path)
        with open(params, "a") as handle:
            handle.write(
                f"fl: {fl_template}\n"
                f"post_process_correction: {ppc_template}\n"
                "add_tf_uncertainty: true\n"
            )
        for index, pair in enumerate(_BASELINE_PAIRS):
            extra[pair] = _tf_and_correction_columns(index, 100)
    generator = CovarianceMatrixGenerator(params)
    generator.run_full_analysis(save_windows=True)
    assert generator.config.add_tf_uncertainty is covariance_only_factors
    windows_dir = os.path.join(_latest_version_dir(tmp_path), "windows")

    cov = generator.covariance_instance
    lmax = cov.lmax
    lmin = generator.config.lmin
    assert generator.config.Dl
    ell = np.arange(lmax)
    dl = ell * (ell + 1) / (2 * np.pi)
    bin_matrix, _ = cov.binning_manager.create_bin_matrix(
        generator._setup_bins(), flatten_with_ell_factor=False
    )
    kernels = {"TT": cov.K[0], "TE": cov.K[3], "ET": cov.K[3], "EE": cov.K[2]}

    checked = 0
    for spec in generator.cov_keys.spec_keys:
        stokes, freq_key = spec.stokekey(), spec.freqkey()
        freq_1, freq_2 = spec.freq
        model = _independent_data_model(freq_1, freq_2, stokes, lmax)
        correction = np.ones(lmax)
        inflation = np.ones(lmax)
        if covariance_only_factors:
            fl, ppc = extra[freq_key]
            column = "TE" if stokes == "ET" else stokes  # 4-column fallback
            model = model * fl[column][:lmax]
            correction = ppc[column][:lmax]
            inflation = 1.0 + np.sqrt((1.0 - fl[column][:lmax]) / 3999.0)
        debias = np.ones(lmax)
        np.divide(1.0, model, out=debias, where=model != 0.0)
        cl = np.asarray(generator.cl_dict[freq_key][stokes])[:lmax]
        np.testing.assert_array_equal(cl[:lmin], 0.0)

        mean = bin_matrix @ (dl * debias * (kernels[stokes] @ (model * cl)))

        label = "x".join(f.replace("GHz", "").lstrip("0") for f in spec.freq)
        table = np.loadtxt(
            os.path.join(windows_dir, f"{stokes}_{label}_window_functions.txt")
        )
        np.testing.assert_array_equal(table[:, 0], np.arange(lmin, lmax))
        got = table[:, 1:].T @ cl[lmin:]

        np.testing.assert_allclose(
            got, mean, rtol=0.0, atol=1e-12 * np.abs(mean).max(), err_msg=spec
        )

        # The covariance keeps c and the inflation on its legs; the window
        # (1/D exactly) does not.
        covariance_leg = np.asarray(generator.debiasing_dict[freq_key][stokes])
        window_leg = generator._window_debiasing(freq_key, stokes)
        np.testing.assert_allclose(window_leg, debias, rtol=1e-15, atol=0.0)
        np.testing.assert_allclose(
            covariance_leg, correction * inflation * debias, rtol=1e-14, atol=0.0
        )
        if covariance_only_factors:
            # Not vacuous: c (up to 5 %) and the inflation (up to ~1 %).
            ratio = covariance_leg[lmin:] / window_leg[lmin:]
            assert np.abs(ratio - 1.0).max() > 1e-2
        else:
            np.testing.assert_array_equal(window_leg, covariance_leg)
        checked += 1
    assert checked == len(_BASELINE_EXPECTED_FILES)


def test_default_run_writes_no_window_directory_and_covariance_is_unchanged(tmp_path):
    """
    Same golden file and same tolerance as
    ``tests/test_end_to_end_baseline.py::test_matches_recorded_baseline``:
    the recorded covariance was produced on a different machine, so the
    comparison must survive a different BLAS/SHT rounding path, not just
    this one. See that test for the measured Linux x86 difference (~1e-9
    relative on 1.5% of entries) and the margin behind rtol=1e-7.
    """
    params = _write_baseline_params(tmp_path)
    rc = main(["--parameter-file", params])
    assert rc == 0

    version = _latest_version_dir(tmp_path)
    assert not os.path.exists(os.path.join(version, "windows"))

    covariance = np.loadtxt(os.path.join(version, "covariance_matrix.dat"))
    expected = np.load(os.path.join(DATA, "baseline_covariance.npy"))
    np.testing.assert_allclose(
        covariance, expected, rtol=1e-7, atol=1e-12 * np.abs(expected).max()
    )


def test_generator_api_default_omits_window_directory(tmp_path):
    """``run_full_analysis()`` with no ``save_windows`` argument (the
    library default, not just the CLI's) writes nothing either."""
    params = _write_baseline_params(tmp_path)
    CovarianceMatrixGenerator(params).run_full_analysis()
    version = _latest_version_dir(tmp_path)
    assert not os.path.exists(os.path.join(version, "windows"))


# --------------------------------------------------------------------------- #
# (d) refusal: EE<->BB mixing under polspice_postprocess: false
# --------------------------------------------------------------------------- #


def _write_cls(path, n=80):
    ell = np.arange(n)
    tt = np.zeros(n)
    tt[2:] = 1e3 / (ell[2:] * (ell[2:] + 1)) ** 0.9
    ee, bb = 0.1 * tt, 0.02 * tt
    te = 0.5 * np.sqrt(tt * ee)
    zero = 0 * tt
    np.savetxt(path, np.column_stack([ell, tt, ee, bb, te, zero, zero, te, zero, zero]))


def _bmode_generator(workdir, polspice_postprocess):
    os.makedirs(workdir, exist_ok=True)
    shutil.copy(os.path.join(DATA, "baseline_mask.fits"), workdir)
    _write_cls(os.path.join(workdir, "cls.dat"))
    params = {
        "cov_path": os.path.join(workdir, "out"),
        "cov_name": "cov.dat",
        "frequencies": ["090GHz"],
        "observables": ["TT", "EE", "TE", "BB"],
        "lmax": 20,
        "lmin": 2,
        "bins": [[2, 20, 3]],
        "mask_name": "baseline_mask.fits",
        "mask_path": workdir,
        "covariance_approximation": "acc",
        "dmax": 3,
        "centralell": 10,
        "cmb_spectrum": os.path.join(workdir, "cls.dat"),
        "beams": {"090GHz": 5.0},
        "pixwin": 32,
        "nl": {"090GHz": 10.0},
        "polspice_postprocess": polspice_postprocess,
        "acc_precompute": {"nside": 16, "grid": "gl", "lw": 10},
    }
    path = os.path.join(workdir, "params.yml")
    with open(path, "w") as handle:
        yaml.dump(params, handle)
    CovarianceMatrixGenerator(path).precompute_acc_kernels()
    return CovarianceMatrixGenerator(path)


@pytest.mark.filterwarnings("ignore:internal band-limit margin", "ignore:centralell=")
def test_bmode_run_refuses_windows_when_ee_bb_mix_undecoupled(tmp_path):
    generator = _bmode_generator(str(tmp_path / "off"), polspice_postprocess=False)
    with pytest.raises(ValueError, match="EE") as excinfo:
        generator.run_full_analysis(save_windows=True)
    message = str(excinfo.value)
    assert "BB" in message
    assert "polspice_postprocess" in message

    # Refusing the windows does not leave a half-written directory behind.
    version = sorted(glob.glob(os.path.join(str(tmp_path / "off"), "out", "v*")))[-1]
    assert not os.path.exists(os.path.join(version, "windows"))


@pytest.mark.filterwarnings("ignore:internal band-limit margin", "ignore:centralell=")
def test_bmode_run_writes_windows_when_decoupled(tmp_path):
    generator = _bmode_generator(str(tmp_path / "on"), polspice_postprocess=True)
    generator.run_full_analysis(save_windows=True)
    version = sorted(glob.glob(os.path.join(str(tmp_path / "on"), "out", "v*")))[-1]
    windows_dir = os.path.join(version, "windows")
    assert set(os.listdir(windows_dir)) == {
        "TT_90x90_window_functions.txt",
        "TE_90x90_window_functions.txt",
        "EE_90x90_window_functions.txt",
        "BB_90x90_window_functions.txt",
    }
