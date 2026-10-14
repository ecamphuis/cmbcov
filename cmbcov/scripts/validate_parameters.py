"""
Command-line validation of a covariance parameter file.

Exposed as the ``validate-parameters`` console script. Validating separately
from computing is worthwhile: a covariance run reads large masks and computes
coupling kernels before it touches most parameters, so a typo in a frequency
label or a missing beam file would otherwise surface many minutes in.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import yaml

from ..config.validation import ParameterValidator
from ..utils.logging_utils import configure_logging


def _load(path: str) -> dict:
    try:
        with open(path) as handle:
            return yaml.safe_load(handle) or {}
    except FileNotFoundError:
        raise SystemExit(f"error: parameter file not found: {path}") from None
    except yaml.YAMLError as exc:
        # The message includes the line and column of the problem.
        raise SystemExit(f"error: could not parse {path}: {exc}") from None


def main(argv: Sequence[str] | None = None) -> int:
    """Validate a parameter file. Returns 0 if it is usable, 1 otherwise."""
    parser = argparse.ArgumentParser(
        description="Validate a covariance parameter file without running the computation.",
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
        "--strict",
        action="store_true",
        help="Treat warnings as failures",
    )
    args = parser.parse_args(argv)
    configure_logging(verbose=False)
    if (
        args.parameter_file is not None
        and args.parameter_file_option is not None
        and args.parameter_file != args.parameter_file_option
    ):
        parser.error("give the parameter file once, positionally or with -p")
    args.parameter_file = (
        args.parameter_file or args.parameter_file_option or "parameters.yml"
    )

    params = _load(args.parameter_file)
    validator = ParameterValidator()
    ok = validator.validate(params)

    for warning in validator.warnings:
        print(f"warning: {warning}")
    for error in validator.errors:
        print(f"error: {error}")

    if not ok:
        print(f"\n{args.parameter_file}: FAILED ({len(validator.errors)} errors)")
        return 1
    if args.strict and validator.warnings:
        print(
            f"\n{args.parameter_file}: FAILED (strict: {len(validator.warnings)} warnings)"
        )
        return 1

    print(f"\n{args.parameter_file}: OK ({len(validator.warnings)} warnings)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
