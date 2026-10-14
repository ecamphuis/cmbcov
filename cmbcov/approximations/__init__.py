"""
Covariance approximations.

Each module here implements one choice of the reduced coupling kernel
Theta-bar of Camphuis et al. (2022) Eq. (25):

    nka   delta functions (Eq. 27)          -- exact on the full sky
    inka  renormalised MASTER kernel (Eq. 30)
    acc   translation-invariant exact kernel (Eq. 33)

`StrategyFactory` selects one from a `CovarianceConfig`.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotations only: assembly imports this package (see __getattr__)
    from ..assembly.covariance import Cov
    from .acc.strategy import ACCStrategy
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

# The strategies import `cmbcov.assembly` (their `Cov`, keys and method enum),
# and `cmbcov.assembly` imports `cmbcov.exact`, which imports
# `cmbcov.approximations.acc.term_selection` and so this package. They are
# therefore imported on first use, so that importing this package (or
# `approximations.acc.term_selection`) does not import `cmbcov.assembly`.
_LAZY = {
    "ACCStrategy": ".acc.strategy",
    "CovarianceStrategy": ".base",
    "NKAStrategy": ".nka",
    "INKAStrategy": ".inka",
}


def __getattr__(name):
    if name in _LAZY:
        return getattr(import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class StrategyFactory:
    """Factory class for creating covariance computation strategies."""

    @staticmethod
    def create_strategy(covariance_instance: "Cov") -> "CovarianceStrategy":
        """
        Create the strategy selected by ``covariance_instance.config.method``.

        Parameters
        ----------
        covariance_instance : Cov
            The covariance instance (mask, kernels, configuration).

        Returns
        -------
        CovarianceStrategy
            An ``ACCStrategy``, ``NKAStrategy`` or ``INKAStrategy``.

        Raises
        ------
        ValueError
            If the method is not supported.
        """
        from ..assembly.covariance import CovarianceMethod
        from .acc.strategy import ACCStrategy
        from .inka import INKAStrategy
        from .nka import NKAStrategy

        strategy_map = {
            CovarianceMethod.ACC: ACCStrategy,
            CovarianceMethod.NKA: NKAStrategy,
            CovarianceMethod.INKA: INKAStrategy,
        }

        if covariance_instance.config.method not in strategy_map:
            raise ValueError(
                f"Unknown covariance method: {covariance_instance.config.method}"
            )

        base = strategy_map[covariance_instance.config.method](covariance_instance)
        return base
