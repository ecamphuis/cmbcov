"""
Command-line validation of a covariance parameter file.

Exposed as the ``validate-parameters`` console script. ``pyproject.toml``
declared this entry point but the module did not exist, so installing the
package produced a command that failed immediately on import.

Validating separately from computing is worth doing: a covariance run reads
large masks and computes coupling kernels before it ever touches most
parameters, so a typo in a frequency label or a missing beam file would
otherwise surface many minutes in.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import yaml

from ..generator.parameter_validation import ParameterValidator


def _load(path: str) -> dict:
    try:
        with open(path) as handle:
            return yaml.safe_load(handle) or {}
    except FileNotFoundError:
        raise SystemExit(f"error: parameter file not found: {path}") from None
    except yaml.YAMLError as exc:
        # Keep the mark: it names the line and column of the problem.
        raise SystemExit(f"error: could not parse {path}: {exc}") from None


def main(argv: Sequence[str] | None = None) -> int:
    """Validate a parameter file. Returns 0 if it is usable, 1 otherwise."""
    parser = argparse.ArgumentParser(
        description="Validate a covariance parameter file without running the computation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--parameter-file",
        "-p",
        default="parameters.yml",
        help="Path to the YAML parameter file",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as failures",
    )
    args = parser.parse_args(argv)

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
