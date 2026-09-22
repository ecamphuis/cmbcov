"""Native (Fortran-free) computation of MASTER/PolSpice coupling kernels."""

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
