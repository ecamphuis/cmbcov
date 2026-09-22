r"""
Bandpower window functions.

The linear map from the fiducial theory spectrum (``cmb_spectrum``) to the
*expected* value of the reported bandpower,

.. math::
    \langle\hat C^X_b\rangle = \sum_\ell W^X_{b\ell}\, C^X_\ell

built from the same mean-coupling kernels and binning matrix
``Cov.compute_covariance_matrix`` uses to propagate the *covariance*
(docs/theory/bmode_kernels.md Sect. 7, docs/polspice.md):

* ``polspice_postprocess: true`` (default): the reported spectrum is the
  decoupled PolSpice bandpower. Its mean response to the true spectrum is
  mask-independent (docs/theory/bmode_kernels.md Sect. 7): TT responds
  through :math:`^{0}K`, TE/ET/TB/BT through :math:`^{\times}K`, and EE, BB
  and EB all through the same :math:`^{-2}K`, with **no** EE<->BB mixing in
  the mean -- the whole point of the decoupling kernel
  :math:`^{\mathrm{dec}}G` (``kernels/polspice.py``, Eq. 91 of Camphuis et
  al. 2022).
* ``polspice_postprocess: false``: the reported spectrum is the binned
  pseudo-:math:`C_\ell`. Its mean uses the MASTER coupling (``Cov.M``,
  channels ``(M0, Mp2, Mm2, Mx)``): TT through ``M0``, TE/ET/TB/BT through
  ``Mx``, EE and BB self-response through :math:`M^{+} = (M_{p2}+M_{m2})/2`
  with an EE<->BB mixing term :math:`M^{-} = (M_{p2}-M_{m2})/2`, and EB
  through ``Mm2`` (the :math:`M^{+}-M^{-}` identity of
  docs/theory/bmode_kernels.md, mirroring the PolSpice EB channel).

:math:`D_\ell` scaling (``config.Dl``) is applied to the *output* (row) axis
only, exactly as :meth:`~cmbcov.postprocess.CovariancePostProcessor.apply_Dl_scaling`
does for the covariance; the input axis stays in :math:`C_\ell` units, so
``W`` maps a :math:`C_\ell` theory spectrum to whatever units this run
reports (:math:`C_\ell` or :math:`D_\ell`).

Beams, the pixel window, the transfer function and the run's calibration
debiasing are per frequency-pair, per-Stokes multiplicative factors
(``SpectraLoader.data_model``/``debiasing_dict``) applied to the *output*
axis, after the :math:`D_\ell` scaling and before binning -- exactly where
:meth:`~cmbcov.postprocess.CovariancePostProcessor.apply_debiasing`
applies ``debiasing_dict`` to each covariance leg. ``build_window_functions``
takes them through the optional ``row_scale`` argument, one array per
two-letter spectrum; the caller (:mod:`cmbcov.generator.generator`,
one call per frequency pair) is responsible for looking the right
``debiasing_dict`` entry up. Leaving ``row_scale`` out (or a key out of it)
maps the fiducial theory spectrum to the mask+PolSpice (or mask-only)
response alone, with no instrumental factor.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

__all__ = ["build_window_functions"]

#: Two-letter Stokes pairs that couple only through the "cross" (TE-like)
#: mean-coupling channel: TE/ET are read off the same real-space step as
#: TB/BT (docs/theory/bmode_kernels.md Sect. 7).
_CROSS_PAIRS = ("TE", "ET", "TB", "BT")
#: Two-letter Stokes pairs whose mean response is a spin-2 channel: EE, BB
#: (with EE<->BB mixing when not decoupled) and EB/BE (never mixed, see the
#: module docstring).
_SPIN2_PAIRS = ("EE", "BB", "EB", "BE")


def _mean_channels(
    covariance_instance, polspice_postprocess: bool
) -> dict[str, np.ndarray]:
    """
    The named mean-coupling channels this run's ``polspice_postprocess``
    setting needs, built from :attr:`Cov.K` (``polspice_postprocess: true``)
    or :attr:`Cov.M` (``polspice_postprocess: false``), both in the
    ``(channel_0, channel_+2, channel_-2, channel_x)`` layout of
    :meth:`~cmbcov.kernels.polspice.PolSpiceKernels.as_master_array`.
    """
    family = covariance_instance.K if polspice_postprocess else covariance_instance.M
    k0, kp2, km2, kx = family
    return {
        "0": k0,
        "x": kx,
        "m2": km2,
        "plus": 0.5 * (kp2 + km2),
        "minus": 0.5 * (kp2 - km2),
    }


def _self_kernel(
    stokekey: str, channels: dict[str, np.ndarray], polspice_postprocess: bool
) -> np.ndarray:
    """The diagonal (no cross-spectrum) mean kernel for ``stokekey``."""
    if stokekey == "TT":
        return channels["0"]
    if stokekey in _CROSS_PAIRS:
        return channels["x"]
    if stokekey in _SPIN2_PAIRS:
        if stokekey in ("EE", "BB") and not polspice_postprocess:
            return channels["plus"]
        return channels["m2"]
    raise ValueError(f"no mean kernel for spectrum {stokekey!r}")


def build_window_functions(
    stokekeys: Iterable[str],
    covariance_instance,
    lbins: list[int] | int,
    Dl: bool,
    polspice_postprocess: bool,
    row_scale: Mapping[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    r"""
    Build the bandpower window functions of every two-letter spectrum in
    ``stokekeys``.

    Parameters
    ----------
    stokekeys : iterable of str
        Two-letter Stokes pairs this run reports (e.g.
        ``CovKeys.combined_stokes()``); only these get a window, deduplicated
        and in the order given.
    covariance_instance : Cov
        Supplies the mean-coupling kernels (:attr:`Cov.K`, :attr:`Cov.M`) and
        the binning manager (:attr:`Cov.binning_manager`).
    lbins : list of int, or int
        The same binning scheme passed to
        :meth:`Cov.compute_covariance_matrix`.
    Dl : bool
        ``config.Dl``: whether the reported bandpower (and hence the
        window's output axis) is in :math:`D_\ell` units.
    polspice_postprocess : bool
        ``config.polspice_postprocess``.
    row_scale : mapping of str to array, optional
        Extra multiplicative factor on the *output* axis (length ``lmax``,
        same indexing as ``"ell"``), applied after the :math:`D_\ell` scaling
        and before binning -- the per frequency-pair beam, pixel window,
        transfer function and calibration debiasing a real run folds into
        its reported bandpower (``SpectraLoader.data_model``/
        ``debiasing_dict``,
        :meth:`~cmbcov.postprocess.CovariancePostProcessor.apply_debiasing`).
        A key missing from ``row_scale`` (or ``row_scale`` left as ``None``)
        applies no extra factor for that spectrum. The EE<->BB mixing term
        (see below) is scaled by the *output* spectrum's own entry (``"EE"``
        for ``W_EE_from_BB``, ``"BB"`` for ``W_BB_from_EE``), matching
        ``apply_debiasing``, which scales a covariance block by its output
        leg alone, regardless of which pseudo source contributed to it.

    Returns
    -------
    dict
        ``"ell"`` (the unbinned multipole axis, length ``lmax``),
        ``"bin_edges"`` (``lbins`` as an array), one ``"W_<XY>"`` array of
        shape ``(n_bins, lmax)`` per requested spectrum, and
        ``"W_EE_from_BB"``/``"W_BB_from_EE"`` for the EE<->BB mixing term
        (only under ``polspice_postprocess: false``, when both EE and BB are
        requested -- zero, and so omitted, under ``polspice_postprocess:
        true``, docs/theory/bmode_kernels.md Sect. 7).
    """
    bin_matrix, _num_bins = covariance_instance.binning_manager.create_bin_matrix(
        lbins, flatten_with_ell_factor=not Dl
    )
    lmax = bin_matrix.shape[1]
    ell = np.arange(lmax)
    dl_row = (ell * (ell + 1) / (2 * np.pi))[:, None] if Dl else 1.0

    channels = _mean_channels(covariance_instance, polspice_postprocess)

    def _scaled(key: str, base: np.ndarray) -> np.ndarray:
        if row_scale is not None and key in row_scale:
            return np.asarray(row_scale[key])[:, None] * base
        return base

    unique_keys = list(dict.fromkeys(stokekeys))
    result: dict[str, np.ndarray] = {"ell": ell, "bin_edges": np.asarray(lbins)}

    for key in unique_keys:
        kernel = _self_kernel(key, channels, polspice_postprocess)
        result[f"W_{key}"] = bin_matrix @ _scaled(key, dl_row * kernel)

    if not polspice_postprocess:
        present = set(unique_keys)
        if "EE" in present and "BB" in present:
            mixing = dl_row * channels["minus"]
            result["W_EE_from_BB"] = bin_matrix @ _scaled("EE", mixing)
            result["W_BB_from_EE"] = bin_matrix @ _scaled("BB", mixing)

    return result
