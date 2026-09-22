"""
Structural test: the package must not call healpy's own spherical-harmonic
transforms.

``cmbcov`` routes every SHT (map <-> alm, and the power
spectra / filters built on top of alm arrays) through ``ducc0`` -- either
directly, or via the pure-numpy/ducc0 helpers in
:mod:`cmbcov.sht` (``ducc0_map2alm``, ``ducc0_alm2map``,
``alm2cl``, ``almxfl``) and :mod:`cmbcov.grid` (the
Gauss-Legendre path). healpy stays a real dependency for I/O and pure index
arithmetic (``hp.read_map``, ``hp.Alm.getlmax``, etc.), and the test suite
itself keeps using ``hp.synfast``/``hp.anafast``/... as an independent
reference -- neither of those is affected by this check.

This test parses the source (rather than grepping raw text) so that mentions
of ``hp.map2alm(...)`` etc. in docstrings and comments -- which document the
very functions this module replaces -- do not trip it; only genuine call
expressions count.
"""

from __future__ import annotations

import ast
import pathlib

import cmbcov

PACKAGE_ROOT = pathlib.Path(cmbcov.__file__).parent

FORBIDDEN = {
    "map2alm",
    "alm2map",
    "anafast",
    "synfast",
    "synalm",
    "alm2cl",
    "almxfl",
    "smoothing",
    "ud_grade",
}

HEALPY_ALIASES = {"hp", "healpy"}

# Deliberate, documented exception -- NOT a bug. See the "Known exception"

ALLOWED_HITS = {}


def _find_forbidden_calls(tree: ast.AST):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr not in FORBIDDEN:
            continue
        value = func.value
        if isinstance(value, ast.Name) and value.id in HEALPY_ALIASES:
            yield func.attr, node.lineno


def test_no_forbidden_healpy_sht_calls():
    hits = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        rel = path.relative_to(PACKAGE_ROOT)
        tree = ast.parse(path.read_text(), filename=str(path))
        for name, lineno in _find_forbidden_calls(tree):
            if (rel.name, name) in ALLOWED_HITS:
                continue
            hits.append(f"{rel}:{lineno}: hp.{name}(...)")

    assert not hits, (
        "Forbidden healpy SHT call(s) found -- route through "
        "cmbcov.sht or cmbcov.grid "
        "instead:\n" + "\n".join(hits)
    )


def test_importing_the_package_does_not_import_healpy():
    """
    healpy is imported lazily, inside the functions that use it (see
    ``cmbcov/utils/healpy_utils.py``): its ``__init__``
    loads matplotlib and ``astropy.coordinates``, which cost 0.24 s of the
    0.27 s package import. A module-level ``import healpy`` anywhere in the
    package would bring that back, so check a fresh interpreter.
    """
    import subprocess
    import sys

    code = (
        "import sys, cmbcov, "
        "cmbcov.scripts.compute_covariance; "
        "print(sorted(m for m in ('healpy', 'matplotlib', 'astropy.coordinates') "
        "if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]", out.stdout + out.stderr


def test_no_module_level_healpy_import():
    """Source-level form of the check above: no top-level ``import healpy``."""
    hits = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in tree.body:
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            if any(n == "healpy" or n.startswith("healpy.") for n in names):
                hits.append(f"{path.relative_to(PACKAGE_ROOT)}:{node.lineno}")
    assert not hits, "module-level healpy import(s): " + ", ".join(hits)
