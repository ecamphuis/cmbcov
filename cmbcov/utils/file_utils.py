"""
File I/O utility functions.

This module contains utilities for reading/writing files used in
covariance matrix computation: transfer functions, spectra, etc.
"""

import glob
import os
import subprocess
from hashlib import blake2b

#: Timeout for the ``git`` subprocess call below, in seconds. This runs on
#: every pipeline run, so it must not hang if ``git`` itself stalls (e.g. a
#: network-mounted or locked repository).
_GIT_HASH_TIMEOUT = 5


def get_git_revision_short_hash() -> str:
    """
    Get the short Git revision hash for the current repository.
    This is useful for tracking code versions in outputs.

    Parameters
    ----------
    repository_path : str, optional
        Path to the Git repository. If None, uses the cmbcov package default path.

    Returns
    -------
    str
        Short Git revision hash, or a deterministic ``"unknown-XXXXXXXX"``
        fallback (a blake2b digest of the repository path) if ``git`` is
        unavailable, times out, or is not a repository. Deterministic so
        that repeated calls -- in the same process or in separate ones, e.g.
        parallel pipeline runs on the same path -- agree, unlike the
        previous fallback which used Python's randomised ``hash()``.
    """
    # this file
    repository_path = os.path.dirname(os.path.abspath(__file__))

    try:
        result = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repository_path,
            stderr=subprocess.DEVNULL,
            timeout=_GIT_HASH_TIMEOUT,
        )
        return result.decode("ascii").strip()
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        FileNotFoundError,
        OSError,
    ):
        digest = blake2b(repository_path.encode(), digest_size=8).hexdigest()[:8]
        return f"unknown-{digest}"


def get_versioned_path(
    base_dir: str,
    write: int | None = None,
    name: str | None = None,
) -> str:
    """
    Get or create a versioned directory path.

    This function manages version numbering for output directories,
    allowing for overwriting existing versions or creating new ones.

    Parameters
    ----------
    base_dir : str
        Base directory for versioned subdirectories
    write : int, optional
        Specific version number to write to
    name : str, optional
        Name of file to check for existence in previous versions

    Returns
    -------
    str
        Path to the versioned directory

    Examples
    --------
    >>> get_versioned_path("/data/output", write=3)
    '/data/output/v3'
    >>> get_versioned_path("/data/output")
    '/data/output/v0'  # if no versions exist
    >>> get_versioned_path("/data/output", name="results.txt")
    '/data/output/v0'  # if v0 exists but results.txt is missing
    """
    os.makedirs(base_dir, exist_ok=True)

    # Handle specific write version
    if write is not None:
        version_dir = os.path.join(base_dir, f"v{write}")
        os.makedirs(version_dir, exist_ok=True)
        return version_dir

    # Find existing versions
    existing_versions = glob.glob(os.path.join(base_dir, "v*"))

    if not existing_versions:
        # No existing versions, create v0
        new_version = 0
    else:
        # Get highest version number
        version_numbers = []
        for version_path in existing_versions:
            version_name = os.path.basename(version_path)
            if version_name.startswith("v") and version_name[1:].isdigit():
                version_numbers.append(int(version_name[1:]))

        if not version_numbers:
            new_version = 0
        else:
            max_version = max(version_numbers)

            # Check if we can reuse the previous version
            if name is not None:
                prev_version_dir = os.path.join(base_dir, f"v{max_version}")
                target_file = os.path.join(prev_version_dir, name)
                if not os.path.exists(target_file):
                    return prev_version_dir

            new_version = max_version + 1

    # Create new version directory
    version_dir = os.path.join(base_dir, f"v{new_version}")
    os.makedirs(version_dir, exist_ok=True)
    return version_dir
