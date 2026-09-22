"""
Command-line converter for legacy text-format ACC coupling-kernel caches.

Exposed as the ``convert-acc-cache`` console script. The on-disk coupling
cache used to be written with ``np.savetxt`` (``.txt``); it is now
``np.save`` (``.npy``) -- ``np.load`` is ~400x faster than ``np.loadtxt`` on
a realistic kernel, and at ``dmax=20`` with 25 polarised pairs that
difference is worth tens of seconds on every run that re-reads a cache from
a previous session.
:func:`~cmbcov.approximations.acc_cache.load_coupling_kernels`
refuses to read a ``.txt``-only cache (naming this command in the error) so
that a stale format is never silently reinterpreted; this script does the
one-time conversion, verifying every kernel round-trips bit-exactly before
touching anything.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from ..approximations.acc_cache import convert_text_cache


def main(argv: Sequence[str] | None = None) -> int:
    """Convert a ``save_dir``'s legacy ``.txt`` coupling cache to ``.npy``."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert a legacy .txt ACC coupling-kernel cache under "
            "<save_dir>/covariance_coupling/ to .npy, verifying each "
            "conversion bit-exactly against its .txt source."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "save_dir",
        help=(
            "Cache directory (the save_dir passed to precompute_acc_kernels; "
            "Cov's acc_kernel_dir, by default its save_dir)"
        ),
    )
    parser.add_argument(
        "--remove-text",
        action="store_true",
        help="Delete the .txt files once their .npy conversion is verified",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-file progress output",
    )
    args = parser.parse_args(argv)

    converted = convert_text_cache(
        args.save_dir, remove_text=args.remove_text, verbose=not args.quiet
    )
    print(f"{converted} file(s) converted under {args.save_dir!r}.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
