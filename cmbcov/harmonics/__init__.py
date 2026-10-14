"""
Spherical-harmonic and quadrature primitives.

`sht` holds the HEALPix transforms (ducc0 backed; healpy is only used for
I/O) and the small harmonic-space helpers built on them. `grid` holds the
Gauss-Legendre ring grids: exact synthesis and analysis, and the banded
integrals over products of harmonic modes that the exact covariance and the
ACC kernels are assembled from.
"""

from .grid import (
    banded_integrals_gl,
    cross_spectrum_full_m,
    full_from_pair,
    gl_analysis,
    gl_analysis_complex,
    gl_legendre_table,
    gl_legendre_table_bytes,
    gl_legendre_table_capacity,
    gl_mask_modes,
    gl_minimal_lmax,
    gl_north_rings,
    gl_quadrature_weights,
    gl_ring_modes,
    gl_shape,
    gl_step1_minimal_lmax,
    gl_synthesis,
    gl_synthesis_complex,
    gl_synthesis_single_m,
    gl_thetas,
    pair_from_full,
    spin_weighted_integrals_gl,
)
from .sht import (
    DEFAULT_MAP2ALM_ITER,
    alm2cl,
    almxfl,
    cplx_spin_weighted_ylm,
    ducc0_alm2map,
    ducc0_map2alm,
)

__all__ = [
    "DEFAULT_MAP2ALM_ITER",
    "alm2cl",
    "almxfl",
    "cplx_spin_weighted_ylm",
    "ducc0_alm2map",
    "ducc0_map2alm",
    "banded_integrals_gl",
    "cross_spectrum_full_m",
    "full_from_pair",
    "gl_analysis",
    "gl_analysis_complex",
    "gl_legendre_table",
    "gl_legendre_table_bytes",
    "gl_legendre_table_capacity",
    "gl_mask_modes",
    "gl_minimal_lmax",
    "gl_north_rings",
    "gl_quadrature_weights",
    "gl_ring_modes",
    "gl_shape",
    "gl_step1_minimal_lmax",
    "gl_synthesis",
    "gl_synthesis_complex",
    "gl_synthesis_single_m",
    "gl_thetas",
    "pair_from_full",
    "spin_weighted_integrals_gl",
]
