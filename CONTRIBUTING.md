# Contributing

## Development install

```bash
git clone https://github.com/ecamphuis/cmbcov
cd cmbcov
pip install -e ".[test]"
```

`pip install -e ".[dev]"` additionally pulls in the formatter/linter/docs
tooling used below (`black`, `ruff`, `isort`, `pre-commit`, `sphinx`).

## Running the tests

```bash
python -m pytest -q
```

See [docs/installation.md](docs/installation.md) for the full requirement
list and notes on optional dependencies (`sympy` for the Wigner-3j
cross-checks, `matplotlib`/`seaborn` for example plots).

## Pre-commit and style

```bash
pip install -e ".[dev]"
pre-commit install
pre-commit run --all-files
```

`.pre-commit-config.yaml` runs `black` (formatting) and `ruff` (linting,
including import sorting) at the versions pinned there; CI runs the same
two checks (`black --check`, `ruff check`) so a PR that hasn't run
`pre-commit` will still be caught. `legacy/fortran/` is excluded from both
(vendored cross-check code, not reviewed against this project's style).

Please run `black` and `ruff check --fix` on changed files before opening a
PR, and keep the test suite green (`python -m pytest -q`).

## Opening a pull request

- Keep PRs focused: one change, with tests, is easier to review than a
  large mixed one.
- Add or update a test for any behaviour change.
- If your change affects the documented API or workflow, update the
  relevant page under `docs/` in the same PR.

## Note on this repository

Development happens in a private repository; this public repository is a
curated export of it. Pull requests opened here are welcome and are ported
back to the development repository by the maintainer -- expect your change
to land as a new commit there (possibly reworded or squashed) rather than
as a direct merge of this PR.
