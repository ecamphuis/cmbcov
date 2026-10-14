"""
Assembling the covariance matrix.

`covariance.Cov` ties the mask, the spectra and the chosen approximation
together and assembles the block matrix. `keys` names the spectra and the
Wick contractions between them, `binning` bins to bandpowers, `postprocess`
applies the PolSpice convolution, the D_ell scaling and the debiasing,
`windows` builds the bandpower window functions, `conditioning` reports the
conditioning of the result and `filtering` handles the filter-and-bin
(map-space filter) correction.
"""

from .binning import BinningManager
from .conditioning import conditioning_report, format_conditioning_report
from .covariance import Cov, CovarianceConfig, CovarianceMethod
from .filtering import (
    FourierRingFilter,
    NodePlan,
    combined_profile,
    default_nodes,
    filtered_sum_rule_ratio,
    highpass_profile,
    lowpass_profile,
    node_plan,
    rho_matrix,
    spectrum_aware_ratio,
    transfer_correction,
)
from .keys import CovKey, CovKeys, SpecKey
from .postprocess import CovariancePostProcessor
from .windows import build_window_functions

__all__ = [
    "Cov",
    "CovarianceConfig",
    "CovarianceMethod",
    "CovKey",
    "CovKeys",
    "SpecKey",
    "BinningManager",
    "CovariancePostProcessor",
    "build_window_functions",
    "conditioning_report",
    "format_conditioning_report",
    "FourierRingFilter",
    "NodePlan",
    "default_nodes",
    "filtered_sum_rule_ratio",
    "highpass_profile",
    "lowpass_profile",
    "combined_profile",
    "node_plan",
    "rho_matrix",
    "spectrum_aware_ratio",
    "transfer_correction",
]
