"""
``get_git_revision_short_hash`` must be deterministic outside a repository.

Python's randomised ``hash()`` would differ between processes for the same
path -- so two pipeline runs writing to the same output directory would
disagree on the "unknown" tag. The fallback is instead a blake2b digest of
the repository path, which agrees within a process, across processes, and
when the ``git`` executable itself is missing (caught via
``FileNotFoundError``/``OSError``, not just ``CalledProcessError``).
"""

import subprocess
import sys

from cmbcov.utils.file_utils import get_git_revision_short_hash


def test_repo_directory_returns_real_short_hash():
    result = get_git_revision_short_hash()
    assert not result.startswith("unknown-")
    # `git rev-parse --short HEAD` is hex.
    int(result, 16)


def test_non_git_directory_returns_deterministic_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "cmbcov.utils.file_utils.os.path.dirname",
        lambda *_: str(tmp_path),
    )
    first = get_git_revision_short_hash()
    second = get_git_revision_short_hash()
    assert first.startswith("unknown-")
    assert first == second


def test_fallback_agrees_across_processes(tmp_path):
    script = (
        "from cmbcov.utils.file_utils import "
        "get_git_revision_short_hash as f\n"
        "from cmbcov.utils import file_utils\n"
        f"file_utils.os.path.dirname = lambda *_: {str(tmp_path)!r}\n"
        "print(f())\n"
    )
    results = []
    for _ in range(2):
        out = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True
        )
        results.append(out.stdout.strip())
    assert results[0] == results[1]
    assert results[0].startswith("unknown-")


def test_missing_git_executable_falls_back(monkeypatch):
    def raise_missing(*args, **kwargs):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(
        "cmbcov.utils.file_utils.subprocess.check_output",
        raise_missing,
    )
    result = get_git_revision_short_hash()
    assert result.startswith("unknown-")
