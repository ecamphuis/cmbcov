# Installation

## Requirements

From `pyproject.toml`:

- Python >= 3.11
- `numpy` >= 1.23.2
- `scipy` >= 1.9.2
- `astropy` >= 5.2
- `healpy` >= 1.16.2
- `pyyaml` >= 6.0
- `ducc0`

`ducc0` and `healpy` are required, not optional: `ducc0` provides every
spherical-harmonic transform the package uses, and `healpy` is used for FITS
I/O, the pixel window table and alm index arithmetic (never for the
transforms themselves).

## Install from source

```bash
git clone https://github.com/ecamphuis/cmbcov
cd cmbcov
pip install -e .
```

This installs the package and its console scripts: `cmbcov-cov` and
`cmbcov-precompute` (with `compute-covariance` and `precompute-acc` kept as
aliases), plus `validate-parameters` and `convert-acc-cache`.

If `ducc0` fails to install from a wheel on your platform, build it from
source:

```bash
pip install --no-binary ducc0 ducc0
```

## Optional dependencies

`pyproject.toml` declares three extras:

```bash
pip install -e ".[test]"       # pytest, sympy: what the test suite needs
pip install -e ".[dev]"        # test + black, ruff, isort, pre-commit, sphinx
pip install -e ".[plotting]"   # matplotlib, seaborn
```

`test` is enough to run the full test suite (`pytest`) and its Wigner-3j
cross-checks (`sympy`); `dev` adds the formatting and linting tools used by
the pre-commit hooks; `plotting` is only used by example scripts that
produce figures.

## Running the tests

```bash
.venv/bin/python -m pytest tests/ -q
```

`tests/data/` holds golden fixtures tracked in git (a small nside-16 mask and
the recorded end-to-end covariance and ACC coupling kernels it produces).
`tests/test_coupling_kernels.py` needs `sympy` (the `dev` extra) and is
skipped otherwise.
