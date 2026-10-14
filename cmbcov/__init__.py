"""
Analytical covariance matrices for CMB power spectra.

Implements the pseudo-Cl covariance of Camphuis et al. (2022),
arXiv:2204.13721. Each subpackage owns one part of the calculation:

    harmonics/      HEALPix spherical-harmonic primitives (sht; ducc0, healpy is I/O
                    only) and Gauss-Legendre grids (grid: exact synthesis/analysis
                    and integrals)
    mask/           the survey footprint (footprint: W_l, W^2_l, degraded copies,
                    kernel cache) and the coupling operators Xi, M, K, G from
                    Wigner-d recursions (coupling, polspice)
    exact/          the exact covariance of Sect. 3, row by row (HEALPix or GL)
    approximations/ the choice of reduced coupling kernel: NKA, INKA, and ACC
                    (approximations/acc/: kernels, cache, error budget, term
                    selection of the (m, m', M) terms from the mask alone, B-mode
                    Wick contractions)
    assembly/       Cov ties it together and assembles the block matrix; keys (which
                    spectra exist, and the Wick contractions between them), binning,
                    postprocess (PolSpice convolution, D_ell scaling, debiasing),
                    windows, conditioning, filtering (filter-and-bin)
    spectra         loading and combining the input spectra, beams and noise
    config/         parameter files, data model, the end-to-end workflow

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

from .assembly.covariance import Cov

# Main classes - import from appropriate modules
from .config.generator import CovarianceMatrixGenerator
from .config.validation import (
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
