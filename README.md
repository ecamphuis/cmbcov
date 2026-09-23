# cmbcov

cmbcov computes the covariance of CMB power spectra measured on a cut sky,
either exactly or with controlled approximations that are fast enough to
recompute as your spectra, noise and binning change. It implements Camphuis
et al. (2022), [arXiv:2204.13721](https://arxiv.org/abs/2204.13721), providing
the exact pseudo $C_\ell$ covariance of the paper's Sect. 3 and three
approximations to it, computed natively in NumPy with no external toolchain.

## Features

- **NKA** (Narrow Kernel Approximation) — the fastest approximation; exact on
  the full sky, degrades at low multipole.
- **INKA** (Improved NKA) — same cost as NKA, renormalised kernel, more
  accurate.
- **ACC** (Analytical Covariance Coupling) — a one-off precompute per mask,
  then a cheap, accurate recompute for every change of spectra, noise, beams
  or binning.
- **Exact covariance** — the brute-force reference of the paper's Sect. 3,
  for validation.
- **T/E and B-mode spectra** — TT, EE, TE by default; BB, TB, EB on request
  (ACC only, with a fast BB-only NKA/INKA escape hatch).
- **PolSpice post-processing** — the decoupled pseudo-to-full $C_\ell$
  transform of the paper's Eq. (55).
- **Binning** — arbitrary bandpower binning with an `lmin` cut.

## Installation

```bash
git clone https://github.com/ecamphuis/cmbcov
cd cmbcov
pip install -e .
```

See [`docs/installation.md`](docs/installation.md) for requirements, optional
dependencies and running the tests.

## Quickstart

In a typical analysis the survey mask is fixed early: the ACC coupling
kernels for that mask are computed **once**, and the covariance is
**recomputed many times** afterwards as the spectra, noise, beams and binning
change. One parameter file drives both steps:

```yaml
cov_path: ./covariance_output
cov_name: covariance_matrix.dat
frequencies: ['090GHz']
stokes: ['T']
lmax: 48
lmin: 2
bins: [[2, 48, 6]]
mask_name: baseline_mask.fits
mask_path: ./tests/data
covariance_approximation: acc
dmax: 4
centralell: 16
acc_precompute: {nside: 16, grid: gl, lw: 10, spectra: [TT]}
cmb_spectrum: ./tests/data/baseline_cls.dat
beams: {090GHz: 5.0}
pixwin: 16
nl: {090GHz: 20.0}
Dl: true
```

```bash
cmbcov-precompute parameters.yml    # once per mask (the expensive step)
cmbcov-cov parameters.yml           # every run: spectra, noise, beams, binning
```

`centralell` and `dmax` above are toy values for a quick run against the
small nside-16 mask shipped in `tests/data/`; see
[`docs/acc_precomputation.md`](docs/acc_precomputation.md) for how to size
them for a real survey mask. Full walkthrough, including the Python API:
[`docs/getting_started.md`](docs/getting_started.md).

## Which approximation should I use?

- **ACC** for a real analysis: sub-percent accuracy far from `centralell`
  once it is high enough, and the only method that supports BB, TB or EB.
- **NKA** for a quick, full-sky-exact estimate, or when the spectra are
  slowly varying and the mask coupling is weak.
- **INKA** when NKA's low-multipole accuracy is not good enough but an ACC
  precompute is not worth it.
- **Exact** (`cmbcov.exact`) to validate any of the above
  on your own mask, not as a production method (expensive at high `lmax`).
  Details, costs and accuracy numbers: [`docs/approximations.md`](docs/approximations.md).

## Documentation

[`docs/README.md`](docs/README.md) indexes every page: installation,
getting started, the parameter-file reference, approximations, ACC
precomputation, polarisation, PolSpice, and the module map for
contributors. See also [`examples/`](examples/) for runnable scripts and an
annotated parameter file.

## Citation and license

If you use this package, please cite Camphuis et al. (2022),
[arXiv:2204.13721](https://arxiv.org/abs/2204.13721), which describes the
method it implements. MIT License — see [`LICENSE`](LICENSE).
