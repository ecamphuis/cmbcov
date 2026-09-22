#!/usr/bin/env python3
"""
Command-line interface for covariance matrix computation.

This script provides the main command-line interface for computing
TTTEEE covariance matrices using the cmbcov package.
"""

import argparse
import os
import sys
from collections.abc import Sequence

from ..generator.generator import CovarianceMatrixGenerator


def main(argv: Sequence[str] | None = None) -> int:
    """Main function for command line interface."""

    parser = argparse.ArgumentParser(
        description="Generate TTTEEE covariance matrices",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "parameter_file_positional",
        nargs="?",
        default=None,
        metavar="parameter_file",
        help="Path to parameter file (same as --parameter-file)",
    )

    parser.add_argument(
        "--parameter-file",
        type=str,
        default=None,
        help="Path to parameter file (default: parameters.yml)",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    parser.add_argument(
        "--overwrite",
        type=int,
        default=None,
        help=(
            "Write into output version N instead of creating a new one. This "
            "chooses where the results go and nothing else: the covariance is "
            "recomputed, and the run is refused if the parameter file already "
            "in that directory differs in anything but the binning"
        ),
    )

    parser.add_argument(
        "--save", action="store_true", default=True, help="Save results to disk"
    )

    parser.add_argument(
        "--save-corrections", action="store_true", help="Save correction factors"
    )

    parser.add_argument(
        "--save-windows",
        action="store_true",
        help=(
            "Save the bandpower window functions beside the covariance, one "
            "plain-text file per spectrum and frequency pair under a "
            "'windows/' subdirectory; see docs/getting_started.md, "
            "'Bandpower window functions'. Refused when polspice_postprocess "
            "is false and both EE and BB are among the observables (the "
            "EE<->BB mixing this file format cannot represent)"
        ),
    )

    parser.add_argument(
        "--dryrun", action="store_true", help="Perform dry run without computation"
    )

    args = parser.parse_args(argv)

    if (
        args.parameter_file_positional is not None
        and args.parameter_file is not None
        and args.parameter_file_positional != args.parameter_file
    ):
        parser.error(
            "give the parameter file once, positionally or with --parameter-file"
        )
    parameter_file = (
        args.parameter_file_positional or args.parameter_file or "parameters.yml"
    )

    # Create and run analysis
    generator = CovarianceMatrixGenerator(
        parameter_file=parameter_file, verbose=args.verbose
    )

    generator.run_full_analysis(
        overwrite=args.overwrite,
        save=args.save,
        save_corrections=args.save_corrections,
        save_windows=args.save_windows,
        dryrun=args.dryrun,
    )

    # Without --verbose the library logger sits at WARNING and the command
    # prints nothing at all, so a run that takes minutes gives no sign that it
    # succeeded or where it put anything. Always report the outcome.
    if args.dryrun:
        print("Dry run complete; no covariance computed.")
        return 0

    config = generator.config
    save_dir = config.save_dir if config is not None else None
    if args.save and save_dir:
        print(f"Covariance written to {os.path.join(save_dir, config.cov_name)}")
    else:
        print("Run complete (results not saved).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
