# Changelog

## 0.2.0 — 2026-09-23

### Breaking changes

- **`nl_is_biased` removed.** A parameter file that still contains the key is
  now refused with a validation error rather than silently ignored. `nl` is
  always the noise power spectrum of the map *as delivered to the estimator*:
  it carries no beam, no pixel window and no transfer function, and it is never
  multiplied by the data model. This is the MASTER convention (Hivon et al.
  2002, [astro-ph/0105302](https://arxiv.org/abs/astro-ph/0105302), their
  Eqs. (15)-(16)); the debiasing already divides each leg by
  $B_1 B_2 w^{\rm pix} F_\ell$, so the noise enters the reported error bars as
  $N_\ell / B^2_\ell$.

- **White-noise levels now follow that convention too, which changes numbers.**
  A dict of white-noise levels used to set `noise_is_biased = False`, which
  multiplied the flat level by $B^2 w^{\rm pix} F_\ell$ and so effectively
  treated it as *already* beam-deconvolved — a flat noise contribution to the
  error bars at every $\ell$. That contradicted both the code's own comment and
  the MASTER convention. The noise contribution now grows as $1/B^2_\ell$ at
  high $\ell$ for white-noise levels, exactly as it always did for tabulated
  `nl` files. Error bars at high $\ell$ in noise-dominated runs therefore grow;
  the previous numbers were optimistic.

- **`nl` dict keys are single frequencies, not frequency pairs.** Write
  `nl: {'090GHz': 5.4}`, not `nl: {'090GHz090GHz': 5.4}`; the old spelling
  raises with a message pointing at the new one. An auto pair `f+f` takes that
  frequency's level; a cross pair `f1+f2` is zero, silently, since map noise is
  uncorrelated between bands.

### Added

- **Two-number `nl` entries.** `nl: {'090GHz': [sigma_T, sigma_P]}` gives
  $N^{TT} = (\sigma_T \pi/10800)^2$ and
  $N^{EE} = N^{BB} = (\sigma_P \pi/10800)^2$ independently. A single number
  keeps the old rule $N^{EE} = N^{BB} = 2 N^{TT}$, i.e. it is the
  $\sigma_P = \sqrt 2 \, \sigma_T$ special case. One dict may mix the two forms.

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
