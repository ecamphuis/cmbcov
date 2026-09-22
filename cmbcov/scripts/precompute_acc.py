"""
Command-line ACC coupling-kernel precompute, driven by a run's parameter file.

Exposed as the ``precompute-acc`` console script. The ACC workflow has two
steps with very different costs: the coupling kernels depend only on the mask
(and ``centralell``, ``dmax``) and are computed once, while the covariance is
recomputed many times as spectra, noise, beams and binning change. This is
step one; ``compute-covariance`` on the *same* parameter file is step two.

Mask, ``centralell``, ``dmax`` and the kernel directory are read from the
parameter file the runs use, so they cannot drift between the two steps; the
backend settings come from its ``acc_precompute`` block::

    acc_precompute:
      nside: 256        # optional
      grid: gl          # healpix | gl
      lw: 512           # optional, gl only
      spectra: [TT]     # optional, default all
      max_memory_gb: 6  # optional

The kernels are written to ``<cov_path>/covariance_coupling/``
(:attr:`~cmbcov.generator.parameter_validation.PipelineConfig.acc_kernel_dir`).
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence

from ..generator.generator import CovarianceMatrixGenerator


def main(argv: Sequence[str] | None = None) -> int:
    """Precompute the ACC coupling kernels of a parameter file. Returns 0 on success."""
    parser = argparse.ArgumentParser(
        description=(
            "Precompute the ACC coupling kernels for a covariance parameter "
            "file (once per mask), for later compute-covariance runs of the "
            "same file."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "parameter_file",
        nargs="?",
        default=None,
        help="Path to the YAML parameter file (same as --parameter-file)",
    )
    parser.add_argument(
        "--parameter-file",
        "-p",
        dest="parameter_file_option",
        default=None,
        help="Path to the YAML parameter file (default: parameters.yml)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Validate the file and print the plan without computing",
    )
    args = parser.parse_args(argv)

    if (
        args.parameter_file is not None
        and args.parameter_file_option is not None
        and args.parameter_file != args.parameter_file_option
    ):
        parser.error("give the parameter file once, positionally or with -p")
    parameter_file = args.parameter_file or args.parameter_file_option
    if parameter_file is None:
        parameter_file = "parameters.yml"

    generator = CovarianceMatrixGenerator(parameter_file, verbose=args.verbose)
    start = time.time()
    try:
        plan = generator.precompute_acc_kernels(dryrun=args.dryrun)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    ellprange = plan["ellprange"]
    print(
        f"ACC kernels: mask {plan['mask']}, centralell={plan['centralell']}, "
        f"dmax={plan['dmax']} (ellp {ellprange[0]}..{ellprange[-1]}), "
        f"grid={plan['grid']}, nside={plan['nside'] or 'auto'}, "
        f"lw={plan['lw'] or ('auto' if plan['grid'] == 'gl' else '-')}, "
        f"spectra={list(plan['spectra']) if plan['spectra'] else 'all'}"
    )
    if args.dryrun:
        print(f"Dry run: nothing computed; kernels would go to {plan['kernel_dir']}")
        return 0
    print(
        f"Wrote {len(ellprange)} kernel pair(s) to {plan['kernel_dir']} "
        f"in {time.time() - start:.1f}s."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
