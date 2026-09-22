"""High-level driver: parameter handling and covariance matrix generation."""

from .generator import CovarianceMatrixGenerator
from .parameter_validation import (
    ParameterManager,
    ParameterValidator,
    PipelineConfig,
)

__all__ = [
    "CovarianceMatrixGenerator",
    "ParameterManager",
    "ParameterValidator",
    "PipelineConfig",
]
