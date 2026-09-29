"""
How well-conditioned the delivered covariance is.

A likelihood needs a positive-definite, invertible matrix, and
multi-frequency covariances are genuinely near-singular: on a real
3-frequency T/E run the smallest eigenvalue of the correlation matrix is
~1.5e-10. This module reports the conditioning; it changes nothing.

The report is built on the correlation matrix R = D^-1/2 C D^-1/2 (D the
diagonal of C), not the covariance itself, so that blocks in different
units (e.g. TT against a noise-dominated high-l bin) compare on the same
scale-free footing.
"""

import numpy as np

__all__ = ["conditioning_report", "format_conditioning_report"]

#: Above this condition number, the interpretation line warns that some
#: combination of bandpowers is almost unconstrained.
LARGE_CONDITION_NUMBER = 1e8


def conditioning_report(cov: np.ndarray) -> dict:
    """
    Conditioning of ``cov``, measured on its correlation matrix.

    Parameters
    ----------
    cov : np.ndarray
        A square, (nominally) symmetric covariance matrix.

    Returns
    -------
    dict
        ``n`` (matrix side), ``asymmetry`` (``max|C - C^T| / max|C|``),
        ``n_bad_diagonal`` (number of zero or negative diagonal entries --
        those rows/columns cannot be turned into a correlation and are
        zeroed out instead of dividing by zero, which shows up as an exact
        zero eigenvalue), ``min_eigenvalue``, ``max_eigenvalue`` of the
        symmetrised correlation matrix, ``n_negative`` (eigenvalues below
        zero), ``positive_definite`` (``min_eigenvalue > 0``), and
        ``condition_number`` (``max/min`` when positive definite, else
        ``None``).
    """
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    scale = np.abs(cov).max()
    asymmetry = float(np.abs(cov - cov.T).max() / scale) if scale > 0 else 0.0

    diag = np.diag(cov)
    good = diag > 0
    n_bad_diagonal = int(np.sum(~good))

    # R = D^-1/2 C D^-1/2; a non-positive diagonal entry has no square root,
    # so its row/column is zeroed instead -- it then contributes an exact
    # zero eigenvalue rather than an inf or a nan, correctly marking the
    # matrix as not invertible.
    inv_sqrt_diag = np.zeros(n)
    inv_sqrt_diag[good] = 1.0 / np.sqrt(diag[good])
    corr = cov * inv_sqrt_diag[:, None] * inv_sqrt_diag[None, :]
    corr = (corr + corr.T) / 2

    eigenvalues = np.linalg.eigvalsh(corr)
    min_eigenvalue = float(eigenvalues[0])
    max_eigenvalue = float(eigenvalues[-1])
    n_negative = int(np.sum(eigenvalues < 0))
    positive_definite = bool(min_eigenvalue > 0)
    condition_number = (
        float(max_eigenvalue / min_eigenvalue) if positive_definite else None
    )

    return {
        "n": n,
        "asymmetry": asymmetry,
        "n_bad_diagonal": n_bad_diagonal,
        "min_eigenvalue": min_eigenvalue,
        "max_eigenvalue": max_eigenvalue,
        "n_negative": n_negative,
        "positive_definite": positive_definite,
        "condition_number": condition_number,
    }


def format_conditioning_report(report: dict) -> str:
    """
    The conditioning report as the text written beside the covariance.
    """
    lines = [
        "Covariance conditioning report",
        "===============================",
        "",
        "Computed on the correlation matrix (scale-free); reported, not",
        "applied: the covariance itself is unchanged.",
        "",
        f"n = {report['n']}",
        f"asymmetry = {report['asymmetry']:.3e}",
    ]
    if report["n_bad_diagonal"]:
        lines.append(
            f"n_bad_diagonal = {report['n_bad_diagonal']} "
            "(zero or negative variance; that row/column was zeroed rather "
            "than divided by zero)"
        )
    lines += [
        f"min_eigenvalue = {report['min_eigenvalue']:.3e}",
        f"max_eigenvalue = {report['max_eigenvalue']:.3e}",
        f"n_negative = {report['n_negative']}",
        f"positive_definite = {report['positive_definite']}",
    ]
    if report["condition_number"] is not None:
        lines.append(f"condition_number = {report['condition_number']:.3e}")
    else:
        lines.append("condition_number = None (matrix is not positive definite)")
    lines.append("")

    if report["n_negative"] > 0:
        lines.append(
            "Interpretation: at least one eigenvalue is negative, so this "
            "matrix is not usable as is in a Gaussian likelihood; "
            "regularise it or diagnose the run that produced it."
        )
    elif (
        report["condition_number"] is not None
        and report["condition_number"] > LARGE_CONDITION_NUMBER
    ):
        lines.append(
            "Interpretation: a very large condition number means some "
            "combination of the reported bandpowers is almost unconstrained "
            "by this covariance."
        )
    else:
        lines.append(
            "Interpretation: the matrix is positive definite and reasonably "
            "well conditioned."
        )
    return "\n".join(lines) + "\n"
