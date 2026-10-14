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

`test` is enough to run the test suite (`pytest`) and its Wigner-3j
cross-checks (`sympy`); `dev` adds the formatting and linting tools used by
the pre-commit hooks; `plotting` is only used by example scripts that
produce figures.

## Running the tests

```bash
.venv/bin/python -m pytest tests/ -q                     # the fast suite (default)
.venv/bin/python -m pytest tests/ -q -m slow             # only the slow tests
.venv/bin/python -m pytest tests/ -q -m "slow or not slow"   # everything
```

A plain `pytest` leaves out the tests marked `slow` (more than about 5 s each:
the Monte Carlo comparisons and the largest identities); `pyproject.toml` sets `-m 'not slow'` by default, and an `-m`
on the command line replaces it. A slow test named by its node id is
deselected too, unless the command carries `-m "slow or not slow"`. CI runs
the fast suite on every push and pull request, and the slow tests in a
separate job after a merge to main (or on request).

`tests/data/` holds golden fixtures tracked in git (a small nside-16 mask and
the recorded end-to-end covariance and ACC coupling kernels it produces).
`tests/mask/test_coupling_kernels.py` needs `sympy` (the `dev` extra) and is
skipped otherwise.
