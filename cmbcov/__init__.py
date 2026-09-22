"""
Analytical covariance matrices for CMB power spectra.

Implements the pseudo-Cl covariance of Camphuis et al. (2022),
arXiv:2204.13721. Each module owns one part of the calculation:

    kernels/        Wigner-d recursions and the coupling operators Xi, M, K, G
    mask            the survey footprint: W_l, W^2_l, degraded copies, kernel cache
    sht             HEALPix spherical-harmonic primitives (ducc0; healpy is I/O only)
    grid            Gauss-Legendre grids: exact synthesis/analysis and integrals
    term_selection  which (m, m', M) terms carry the ACC kernel, from the mask alone
    exact           the exact covariance of Sect. 3, row by row (HEALPix or GL)
    keys            which spectra exist, and the Wick contractions between them
    approximations  the choice of reduced coupling kernel: NKA, INKA, ACC
    covariance      Cov: ties it together and assembles the block matrix
    binning         bandpower binning
    postprocess     PolSpice convolution, D_ell scaling, debiasing
    spectra         loading and combining the input spectra, beams and noise
    generator/      parameter files, data model, the end-to-end workflow

Example
-------
    >>> from cmbcov import CovarianceMatrixGenerator
    >>> generator = CovarianceMatrixGenerator("parameters.yml")
    >>> generator.run_full_analysis()

`run_full_analysis` validates the parameter file, loads beams, spectra and
noise, computes the covariance and writes it to a versioned output directory.
To keep the matrix in memory instead, run the setup chain yourself and call
`compute_covariance_matrix`, which returns
``(binned_ells, bin_matrix, covariance)``.
"""

from importlib.metadata import PackageNotFoundError, version

from .covariance import Cov

# Main classes - import from appropriate modules
from .generator.generator import CovarianceMatrixGenerator
from .generator.parameter_validation import (
    ParameterManager,
    ParameterValidator,
    PipelineConfig,
)

try:
    __version__ = version("cmbcov")
except PackageNotFoundError:  # not installed, e.g. running from a source tree
    __version__ = "0.0.0.dev0"

__author__ = "Etienne Camphuis"

__all__ = [
    # Core classes
    "CovarianceMatrixGenerator",
    "Cov",
    "ParameterValidator",
    "ParameterManager",
    "PipelineConfig",
]
