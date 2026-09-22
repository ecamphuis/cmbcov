# Changelog

## 0.1.0 — initial public release

- Analytical pseudo $C_\ell$ covariance for CMB power spectra, implementing
  Camphuis et al. (2022), [arXiv:2204.13721](https://arxiv.org/abs/2204.13721).
- Three approximations — NKA, INKA and ACC — plus the exact covariance of
  the paper's Sect. 3 for validation.
- ACC: a one-off coupling-kernel precompute per mask, then a cheap
  covariance recompute for every change of spectra, noise, beams or
  binning.
- T/E spectra (TT, EE, TE) by default; BB, TB and EB on request through
  `observables` (ACC), with a fast leakage-neglected BB-only NKA/INKA
  escape hatch.
- PolSpice post-processing (the decoupled estimator of the paper's Eq. 55),
  including the EE/BB mixing for B-mode runs.
- Native (NumPy/`ducc0`) computation of the MASTER and PolSpice coupling
  kernels — no external toolchain.
- Arbitrary bandpower binning with an `lmin` cut.
- Parameter-file driven workflow (`cmbcov-cov`, `cmbcov-precompute`,
  `validate-parameters`, `convert-acc-cache`; `compute-covariance` and
  `precompute-acc` remain as aliases) and an equivalent Python API.
- Bandpower window functions (`cmbcov-cov --save-windows`): the
  linear map from the fiducial theory spectrum to each reported bandpower's
  expectation value, for the decoupled PolSpice output or the pseudo
  $C_\ell$ one, including each pair's beam, pixel window, transfer function
  and calibration debiasing. One plain-text file per spectrum and frequency
  pair, under a `windows/` subdirectory beside the covariance.
