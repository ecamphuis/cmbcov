# Documentation index

- [`installation.md`](installation.md) — requirements, installing from
  source, optional dependencies, running the tests.
- [`getting_started.md`](getting_started.md) — end-to-end walkthrough:
  inputs, the ACC precompute/recompute workflow, running from a parameter
  file and from the Python API, outputs, caching.
- [`parameters.md`](parameters.md) — full reference of every parameter-file
  key.
- [`approximations.md`](approximations.md) — NKA, INKA, ACC and the exact
  covariance: assumptions, cost, accuracy and which spectra each supports.
- [`acc_precomputation.md`](acc_precomputation.md) — user guide to the ACC
  coupling-kernel precompute: sizing `centralell`, `dmax`, `nside`, memory,
  caching.
- [`polarisation.md`](polarisation.md) — T/E and B-mode spectra: the four
  observable levels, kernel channels, parity-mixed blocks, the BB-only
  escape hatch, and known accuracy limitations.
- [`polspice.md`](polspice.md) — the PolSpice post-processing transform.
- [`package_structure.md`](package_structure.md) — module map, for
  contributors.
- [`theory/acc.md`](theory/acc.md), [`theory/exact_covariance.md`](theory/exact_covariance.md),
  [`theory/bmode_kernels.md`](theory/bmode_kernels.md),
  [`theory/term_selection.md`](theory/term_selection.md) — the methods
  themselves.

See also [`../examples/`](../examples/) for runnable scripts and an
annotated parameter file, and the paper,
[arXiv:2204.13721](https://arxiv.org/abs/2204.13721).
