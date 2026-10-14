"""
The survey mask: footprint, coupling kernels and the PolSpice operators.

`footprint.MaskWlm` owns everything derived from the survey footprint (the
map, its harmonic coefficients, the power spectra W_l and W^2_l, degraded
copies and the on-disk kernel cache). `coupling` computes the MASTER coupling
kernels natively (Wigner-d recursions and the operators Xi, M, K, G, with no
Fortran), and `polspice` the PolSpice apodised-correlation kernels.
"""

from .coupling import (
    KERNEL_TE,
    KERNEL_TT,
    KERNEL_EEmBB,
    KERNEL_EEpBB,
    coupling_kernels,
    gauss_legendre_nodes,
    mask_correlation_function,
    wigner_d_table,
    xi_operator,
)
from .footprint import KERNEL_CACHE_VERSION, MaskWlm, mask_spectral_moment
from .polspice import (
    APODIZE_COSINE,
    APODIZE_GAUSSIAN,
    APODIZE_NONE,
    PolSpiceKernels,
    apodization_function,
    polspice_g_function,
    polspice_kernels,
)

__all__ = [
    "MaskWlm",
    "mask_spectral_moment",
    "KERNEL_CACHE_VERSION",
    "coupling_kernels",
    "gauss_legendre_nodes",
    "wigner_d_table",
    "mask_correlation_function",
    "xi_operator",
    "KERNEL_TT",
    "KERNEL_EEpBB",
    "KERNEL_EEmBB",
    "KERNEL_TE",
    "APODIZE_NONE",
    "APODIZE_GAUSSIAN",
    "APODIZE_COSINE",
    "PolSpiceKernels",
    "apodization_function",
    "polspice_g_function",
    "polspice_kernels",
]
