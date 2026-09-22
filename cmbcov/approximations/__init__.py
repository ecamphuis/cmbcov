"""
Covariance approximations.

Each module here implements one choice of the reduced coupling kernel
Theta-bar of Camphuis et al. (2022) Eq. (25):

    nka   delta functions (Eq. 27)          -- exact on the full sky
    inka  renormalised MASTER kernel (Eq. 30)
    acc   translation-invariant exact kernel (Eq. 33) -- this work, 1% accurate

`StrategyFactory` selects one from a `CovarianceConfig`.
"""

import os
import warnings
from abc import ABC, abstractmethod

import numpy as np

from ..covariance import Cov, CovarianceConfig, CovarianceMethod
from ..keys import CovKey
from ..utils.healpy_utils import get_nside_from_ell
from .acc import ACCStrategy
from .base import CovarianceStrategy
from .inka import INKAStrategy
from .nka import NKAStrategy

__all__ = [
    "StrategyFactory",
    "CovarianceStrategy",
    "ACCStrategy",
    "NKAStrategy",
    "INKAStrategy",
]


class StrategyFactory:
    """Factory class for creating covariance computation strategies."""

    @staticmethod
    def create_strategy(covariance_instance: Cov) -> "CovarianceStrategy":
        """
        Create a covariance computation strategy.

        Parameters
        ----------
        method : CovarianceMethod
            The covariance computation method to use
        covariance_instance : Cov
            The main covariance computation instance

        Returns
        -------
        CovarianceStrategy
            Strategy instance for the requested method

        Raises
        ------
        ValueError
            If method is not supported
        """
        strategy_map = {
            CovarianceMethod.ACC: ACCStrategy,
            CovarianceMethod.NKA: NKAStrategy,
            CovarianceMethod.INKA: INKAStrategy,
        }

        if covariance_instance.config.method not in strategy_map:
            raise ValueError(
                f"Unknown covariance method: {covariance_instance.config.method}"
            )

        return strategy_map[covariance_instance.config.method](covariance_instance)
