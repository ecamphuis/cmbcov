"""
The ACC approximation: translation-invariant exact kernel (Eq. 33).

The reduced coupling kernel depends only on the multipole separations:
Theta-bar at (l, l+Delta) is approximately the same as at (l*, l*+Delta) for
any reference l* (Eq. 33). The expensive exact kernel is therefore computed
once at l* and reused along each diagonal, giving 1% accuracy for
O(dmax * nside^4) work instead of O(lmax^5).

The precompute side: `channels` names the kernel channels and their aliases;
`producers` makes the mask integrals the kernels are contracted from (on a
Gauss-Legendre grid or a HEALPix map); `contraction` contracts them block by
block under a memory budget; `store` is the on-disk store of the central set;
`precompute` runs it all (`precompute_acc_kernels`); `cache` is the on-disk
coupling-kernel store; `term_selection` chooses which (m, m', M) terms carry
the kernel, from the mask alone.

The recompute side: `strategy` holds `ACCStrategy`, which assembles a
covariance from the cached kernels, with `bands` its window, cache-requirement
and band-assembly helpers; `budget` is the error budget of a configuration and
`bmode_wick` the Wick contractions of the B-mode blocks.
"""

from .bands import (
    acc_cached_kernel_size,
    acc_internal_lmax,
    acc_window_pad,
    require_acc_cache_pairs,
)
from .bmode_wick import (
    PARITY_ODD_SPECTRA,
    WickTerm,
    canonical_spectrum,
    covkey_wick_terms,
    required_kernel_pairs,
)
from .budget import error_budget, format_error_budget
from .cache import (
    convert_text_cache,
    load_coupling_kernels,
    locate_coupling_kernel,
    read_coupling_manifest,
    write_coupling_kernels,
)
from .channels import (
    CHANNEL_ALIASES,
    COUPLING_CHANNELS,
    COUPLING_SPECTRA,
    LEGACY_CHANNEL_NAMES,
    normalise_channel,
)
from .contraction import DEFAULT_COUPLING_MEMORY_GB, contract_coupling_block
from .precompute import coupling_ellprange, precompute_acc_kernels
from .term_selection import TermSelection, select_terms


def __getattr__(name):
    # ACCStrategy imports `cmbcov.assembly`, which imports `cmbcov.exact`, which
    # imports `term_selection` from this package: it is imported on first use so
    # that importing this package does not import `cmbcov.assembly`.
    if name == "ACCStrategy":
        from .strategy import ACCStrategy

        return ACCStrategy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ACCStrategy",
    "precompute_acc_kernels",
    "CHANNEL_ALIASES",
    "COUPLING_CHANNELS",
    "COUPLING_SPECTRA",
    "DEFAULT_COUPLING_MEMORY_GB",
    "LEGACY_CHANNEL_NAMES",
    "acc_cached_kernel_size",
    "acc_internal_lmax",
    "acc_window_pad",
    "contract_coupling_block",
    "coupling_ellprange",
    "normalise_channel",
    "require_acc_cache_pairs",
    "convert_text_cache",
    "load_coupling_kernels",
    "locate_coupling_kernel",
    "read_coupling_manifest",
    "write_coupling_kernels",
    "error_budget",
    "format_error_budget",
    "TermSelection",
    "select_terms",
    "PARITY_ODD_SPECTRA",
    "WickTerm",
    "canonical_spectrum",
    "covkey_wick_terms",
    "required_kernel_pairs",
]
