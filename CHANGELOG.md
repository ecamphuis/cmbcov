# Changelog

## 0.7.0 — 2026-10-14

### Breaking changes

- **Package layout reorganised (breaking).** The flat `cmbcov/` modules now
  live in subpackages. The top-level `cmbcov` API (`Cov`,
  `CovarianceMatrixGenerator`, `ParameterValidator`, `ParameterManager`,
  `PipelineConfig`) is unchanged; the old module paths are removed, with no
  compatibility shims. Each new subpackage re-exports its public names
  (`cmbcov.mask.MaskWlm`, `cmbcov.exact.exact_covariance`,
  `cmbcov.approximations.acc.precompute_acc_kernels`, ...), and the defining
  submodule path works too. `cmbcov.approximations.ACCStrategy` and
  `cmbcov.approximations.acc.ACCStrategy` are resolved on first use, so that
  importing `cmbcov.exact` or `cmbcov.approximations.acc.term_selection` does
  not import `cmbcov.assembly`.

  | Old module | New module |
  | --- | --- |
  | `cmbcov.sht`, `cmbcov.grid` | `cmbcov.harmonics.sht`, `cmbcov.harmonics.grid` |
  | `cmbcov.mask` | `cmbcov.mask.footprint` (`cmbcov.mask` re-exports `MaskWlm`) |
  | `cmbcov.kernels.coupling`, `cmbcov.kernels.polspice` | `cmbcov.mask.coupling`, `cmbcov.mask.polspice` |
  | `cmbcov.exact` | `cmbcov.exact`, now a subpackage split by path: `common`, `healpix`, `gl`, `pol`, `plan`, `parallel`, `covariance` (the public entry points; `cmbcov.exact` re-exports them) |
  | `cmbcov.approximations.acc` | `cmbcov.approximations.acc`, now split into `channels`, `producers`, `contraction`, `store`, `precompute` (the kernel precompute) and `strategy`, `bands` (`ACCStrategy` and its helpers); `cmbcov.approximations.acc` re-exports its public names |
  | `cmbcov.approximations.acc_cache`, `cmbcov.approximations.acc_budget` | `cmbcov.approximations.acc.cache`, `cmbcov.approximations.acc.budget` |
  | `cmbcov.term_selection`, `cmbcov.bmode_wick` | `cmbcov.approximations.acc.term_selection`, `cmbcov.approximations.acc.bmode_wick` |
  | `cmbcov.covariance`, `cmbcov.keys`, `cmbcov.binning`, `cmbcov.postprocess`, `cmbcov.windows`, `cmbcov.conditioning`, `cmbcov.filtering` | the same names under `cmbcov.assembly` |
  | `cmbcov.generator.generator`, `cmbcov.generator.parameter_validation` | `cmbcov.config.generator`, `cmbcov.config.validation` |

- **`ParameterManager` and `ParameterValidator` no longer take
  `parent_logger`, and the library no longer configures logging**: a Python
  user who wants the progress lines calls `cmbcov.utils.configure_logging(verbose=True)`
  or `logging.basicConfig(level=logging.INFO)` (see Changed).

### Added

- **`map_filter.lowpass`: a scan-direction low-pass multiplying the
  high-pass.** A time stream that is also low-passed along the scan
  (SPT-3G: cut 300, low-pass 6144) is the same ring filter with
  $h = h_{\mathrm{high}}\thinspace h_{\mathrm{low}}$. The optional sub-block
  `lowpass: {lx, shape, power}` (`shape` `exp`, the default, or `sharp`;
  `power` 6) takes the same cut convention $m_c = l_x \sin\theta$ as the
  high-pass, with `lx` above the high-pass one. `filtering.lowpass_profile`
  and `filtering.combined_profile` build the profiles; `default_nodes` and
  `node_plan` take `lx_lowpass` and `lowpass_shape` and add nodes through the
  rolloff (every $l_x^{\mathrm{low}}/15$ from $0.6$ to
  $2\thinspace l_x^{\mathrm{low}}$, every $l_x^{\mathrm{low}}/50$ at the onset of a sharp
  one) with the near-cut probe count; a low-pass above $\ell_{\max}/0.6$ adds
  none. Without the key nothing changes, bit for bit, and the saved ratio
  caches stay valid. At toy scale (low-pass at 200, inside the band) the
  corrected T/E blocks are within 1.2% of the exact filtered covariance down to
  $F \approx 0.4$, where leaving the low-pass out of $\rho$ gives +50% to
  +450%; at the survey scale the SPT-3G low-pass lowers $\rho$ by 0.2%, 1.4%
  and 5.1% at $\ell$ = 2000, 2800 and 3499 (probe noise 0.1%). See
  `docs/theory/filter_and_bin.md`, "The low-pass".

- **`nprocs` for `exact_covariance_pol`.** The polarised exact covariance
  can now spread its rows over worker processes, as `exact_covariance`
  already could (Gauss-Legendre grid only, default 1). The results are
  bit-identical to the serial loop; processes mostly pay off on many-core
  nodes rather than on one laptop.

- **`map_filter.spectrum_aware`: spectrum-aware correction of the TT x TT
  blocks.** With `spectrum_aware: true` every TT x TT block is probed with
  the run's fiducial spectra instead of a white one
  (`filtering.spectrum_aware_ratio`): the factor is the ratio of the exact
  filtered covariance (spectra before the filter) to the exact unfiltered
  one (the spectra the block is computed with). It multiplies the block
  with no `fl` division and includes the power a sharp latitude-dependent
  cut scatters across multipoles, which the white-spectrum ratio misses: on
  a survey mask with the sharp cut at $l_x = 300$, the TT variance at
  multipoles 2000 to 3500 goes from 5.5% low to within the Monte Carlo noise
  (0.996 $\pm$ 0.006). The other T/E blocks keep the white channels, and
  `00` is then not probed. The probes use a Gauss-Legendre grid oversampled
  by 2. The ratio is cached under a digest of the spectra, so changing them
  probes again. `CovarianceConfig` gains `filter_rho_blocks`
  (`{block_id: (nodes, g)}`, `filtering.block_id`); `filtering` also
  provides `pre_filter_noise`. Default `false`: nothing changes.

- **`map_filter.near_cut_step` and `map_filter.nprobe_near_cut`; the probing
  functions take per-node settings.** `near_cut_step` is the spacing of the
  extra nodes between $0.95\thinspace l_x$ and $2\thinspace l_x$ (`null`:
  $l_x / 15$; `0`: none), `nprobe_near_cut` the probes of the nodes from
  $0.95\thinspace l_x$ to below $2\thinspace l_x$ (default 128; `null`:
  the complete basis, exact ratio). `filtering.filtered_sum_rule_ratio` and
  `filtering.spectrum_aware_ratio` accept `nprobe` as one count per node and
  a new `lmax_int_floor` (lower bound of the internal band-limit, one for
  all nodes or one per node); an integer `nprobe` and no floor give exactly
  the previous results. `filtering.default_nodes` takes `lx` and
  `near_cut_step` (without `lx` the list is unchanged), and
  `filtering.node_plan` builds the nodes with their per-node probes and
  floors. With `map_filter` the run logs a warning naming the bins whose
  lowest `fl` is below 0.3 (accuracy degraded) and one naming those below
  0.1 (cut them); `filtering.bin_transfer_minimum` gives the values.


### Changed

- **`map_filter` near the cut: denser nodes, more probes and a higher
  band-limit by default.** The nodes now include one every $l_x / 15$
  multipoles from $0.95\thinspace l_x$ to $2\thinspace l_x$; those use 128
  random probes instead of 32 (`nprobe_near_cut: null` for the complete
  basis of their degree, an exact ratio); and those
  below $1.3\thinspace l_x$ carry the probes to at least
  $3.5\thinspace l_x$ (the sharp cut scatters power into them from far
  multipoles). This changes the results and the cache keys of every
  `map_filter` run relative to 0.6.0 (`near_cut_step: 0` with
  `nprobe_near_cut: 32` restores the 0.6.0 nodes and probes, but not the
  band-limit of the nodes at the cut). Measured with the complete basis in
  a survey-like test (nside-512 version of a 4% survey mask, sharp cut at
  $l_x = 300$, 13200 paired simulations, bins of 50, Monte Carlo error 1.1
  to 1.4%), the
  paired variance ratio of the bin 350 to 399 went from 0.939 to 0.997 (TT,
  white), 0.967 to 1.011 (TT, `spectrum_aware`), 0.957 to 1.003 (EE) and
  0.924 to 0.975 (TE), that of 400 to 449 from 0.974, 1.007, 0.981, 0.975
  to 0.978, 0.986, 0.986, 0.980; above 450 the bins moved by 0.7% or less.
  With a smooth cut every T/E block is now within 2.1% from 350 up (it was
  4.5% off in 400 to 449). The bin containing the cut is not usable either
  way. The probing costs more: at survey scale (mask band-limit 875,
  `dmax` 99, 8 threads) the near-cut nodes add about 24 minutes for the
  three white channels and 13 minutes for the spectrum-aware TT to the 33
  and 12 to 17 minutes the whole probing took before, and the binned
  variances of that test differ from the complete-basis ones by at most
  0.4%; the complete basis (`nprobe_near_cut: null`) adds about 2.6 and
  1.3 hours.

- **The library no longer configures logging.** `CovarianceMatrixGenerator`
  used to call `logging.basicConfig` and set levels on import-time loggers;
  now every module logs through `logging.getLogger(__name__)` and only the
  command-line scripts configure levels. A Python user who wants the run's
  progress calls `logging.basicConfig(level=logging.INFO)` or
  `cmbcov.utils.configure_logging(verbose=True)` (the function the scripts
  use: it sets the root level and format).
  `ParameterManager` and `ParameterValidator` no longer take `parent_logger`.

- **`CovariancePostProcessor` accepts an optional `mixing` kernel in its dict
  form.** The dict layout now takes a `"mixing"` entry with the EE <-> BB
  kernel, so `pseudo_to_spice_bmode` works with it as with the array layout.
  Without it, the B-mode methods raise a `ValueError` naming the missing
  kernel (they raised `AttributeError`).


- **The default test run deselects the tests marked `slow`** (more than about
  5 s each; `addopts = "-m 'not slow'"`). `pytest -m slow` runs only them and
  `pytest -m "slow or not slow"` everything. CI runs the fast suite on every
  push and pull request, and the slow tests in a separate job on Python 3.12
  after a merge to main (or on request).


## 0.6.0 — 2026-10-01

### Added

- **`map_filter` block: filter-and-bin correction of the T/E
  covariance.** For a map made with a scan-direction Fourier high-pass, each
  raw T/E pseudo-spectrum block is multiplied by
  $\rho^{ch} / (F^{L} F^{R})$, with $\rho^{ch}$ the filtered sum-rule ratio of the mask in the block's
  channel (`00` for TT x TT, `20` for TT x EE, TT x TE and TE x TE, `EE` for
  EE x EE and EE x TE; probed stochastically at a few multipoles, only for
  the channels the run needs, cached in the kernel directory) and $F^{s}$ the
  `fl` of each leg's spectrum, for NKA, INKA and ACC. On toy patches every
  T/E block was within 0.7% of the exact filtered covariance at
  $\ell \ge 3.3\thinspace l_x$. Blocks with a B leg are left uncorrected (one
  `UserWarning` naming them): a single factor was up to 22% off for BB x BB.
  Without the block nothing changes. `CovarianceConfig` gains `filter_rho`
  (`{channel: rho}`) and `filter_fl` (`{freq_pair: {stokes: fl}}`);
  `cmbcov.filtering` provides the probes (`filtered_sum_rule_ratio(...,
  channels=)`, which returns `{channel: g}`), `CHANNELS`, `BLOCK_CHANNEL`
  and `channels_for`. See `docs/theory/filter_and_bin.md`.

- **`max_memory_gb`** (`exact_covariance_row`, `exact_covariance`,
  `exact_covariance_row_pol`, `exact_covariance_pol`, GL grid only; a
  `ValueError` with `grid="healpix"`): memory budget of a row in GiB,
  default 2. It covers the arrays the row holds that grow with the problem,
  and sets how many columns are computed together and whether the Legendre
  tables are kept for all rows; it changes the result only at the level of
  rounding. A polarised column needs up to nine times the memory of a
  temperature one, and its tables up to three times, so polarised rows
  compute fewer columns together at the same budget. Not covered: the mask
  as loaded and the internal buffers of ducc0 and the BLAS library; with
  the default budget the process grew by 2.0-2.3 GiB (temperature) and
  2.3-2.9 GiB (polarisation, a few chunks) in our measurements, and over a
  whole polarised row up to 4.4 GiB on macOS, whose allocator keeps freed
  memory (`MallocLargeCache=0` keeps it at 2 GiB, more slowly). With
  `nprocs > 1` (`exact_covariance`) the budget applies to each worker
  process.

### Performance

- **Exact covariance rows on the Gauss-Legendre grid are faster, for
  temperature and polarisation.** `exact_covariance_row`,
  `exact_covariance`, `exact_covariance_row_pol` and `exact_covariance_pol`
  with `grid="gl"` now compute the columns of a row together, as matrix
  products with tables of Legendre functions and one batched FFT, instead
  of separate spherical-harmonic transforms for every column. Only the
  azimuthal orders and degrees the mask can reach are computed. At the
  default memory budget the rows are roughly 2.5 to 4 times faster than
  before, from lmax 500 to 2000, for both temperature and polarised rows
  (measured on a 14-core laptop under load, mask band-limit 384, one row at
  the highest multipole; the range reflects the load): temperature, 3.7-10 s
  instead of 25 s at lmax 500 and 28-36 s instead of 130 s at lmax 1000;
  all six polarised spectra, 1-2 minutes instead of 5 at lmax 500 and
  10-12.5 minutes instead of 37 at lmax 1000 (lmax 1500 and 2000 were
  estimated from sampled columns and show the same gain). The rows agree
  with the previous code to a few 1e-15 of their largest entry, and are
  identical for any number of threads.

### Changed

- **Term selection now warns when combined with E to B leakage kernels.**
  `precompute_acc_kernels(..., term_selection=<tol>)` emits a `UserWarning`
  (once per call) when a requested kernel has a leakage leg (`LL`, `TL`,
  `LT`, `DL`, `LD`; the default `spectra` includes `LL`). On a mask that is not
  azimuthally symmetric about its centre the tolerance is not guaranteed for
  these kernels: errors up to about 10 times the tolerance were measured
  (`LLxLL` 1.2e-2 at tolerance 1e-3 on a two-blob mask). Kernels without a
  leakage leg stay within the tolerance. No computed number changes.

### Fixed

- **A transfer function with a NaN is now refused.** `fl` files are checked
  for `[0, 1]`; a NaN (the 0/0 of a measured ratio at l = 0, 1) passed both
  comparisons and spread through every EE and TE element of the covariance
  with no error. `SpectraLoader._check_transfer_function_range` now also
  rejects non-finite values.

## 0.5.0 — 2026-09-30

### Changed

- **`max_memory_gb` now bounds the precompute's real working set.** It
  covers everything that grows with the problem: the coefficient blocks,
  the contraction's intermediate arrays, and any coefficient set held in
  memory between kernel pairs. Before, some of these were not counted, and
  a run whose central set fitted in memory could use up to 1.5× the budget.
  Not covered, and documented: a fixed baseline for the Python process and
  the mask as loaded (about 0.5 GiB; about 2 GiB briefly while an
  nside-2048 mask is read). At the same budget the blocks are slightly
  narrower. On macOS the memory footprint can still exceed the budget,
  because the system allocator keeps freed memory; set
  `MallocLargeCache=0` if a tight limit matters.

### Fixed

- **ACC term selection with a low mask band-limit.** When `lw` was below
  half of ℓ + ℓ′, the pair rule crashed with an `IndexError`, or silently
  used a wrong coupling estimate for some pairs. Those pairs do not couple
  (the mask has no azimuthal power beyond `lw`) and are now dropped. Runs
  with `lw` of at least ℓ + ℓ′ over 2, which includes the usual settings,
  are unchanged.

### Added

- **`scratch_dir`** (`acc_precompute` block, `precompute_acc_kernels`):
  where the precompute keeps its temporary on-disk copy of the central
  coefficients when they do not fit in half of `max_memory_gb`. Default:
  the kernel directory. The copy is deleted at the end of the run, also on
  failure.

### Performance

- **The ACC kernel precompute is about 2.5× faster.** Each coefficient set
  is now computed once instead of once per kernel pair, and the Gauss-Legendre
  coefficients use Legendre transforms instead of full spherical-harmonic
  transforms. Measured back to back on a survey mask: about 2.5× per
  kernel pair. A full `dmax = 100` precompute took 1.5 hours on a busy
  machine, against about 2.5 hours before. Kernels are identical up to
  floating-point rounding (2e-14 of the largest element); existing kernel
  caches remain valid.


## 0.4.1 — 2026-09-29

### Fixed

- **The test suite failed on Linux.** Three tests required the batched ACC
  assembly of 0.4.0 to be bit-for-bit identical to the previous per-block
  code. The two do the same arithmetic on matrices of different shapes, and
  OpenBLAS (Linux) rounds such products differently in the last bit, where
  Accelerate (macOS) does not: one element differed by 7e-17. The tests now
  allow floating-point rounding (1e-13 of the block), which still catches
  any real error. No change to the package code.
- **Correction to the 0.4.0 notes:** the batched assembly is identical to
  the previous code up to floating-point rounding, not bit-for-bit on every
  platform. Bit-for-bit with the same BLAS library was checked on macOS.

## 0.4.0 — 2026-09-29

### Breaking changes

- **B-mode runs need a larger kernel cache.** Runs with a B observable now
  need the kernels of both orientations of every block: 24 or 25 kernel
  pairs instead of 17 or 18 for a typical run, 45 instead of 31 or 40 for
  the larger sets. T/E-only and BB-only runs are unchanged. **An existing
  B-mode cache is refused** with a message naming the missing pairs; add
  them with `precompute_acc_kernels(..., pairs=[...])` (now safe, see
  below) or rerun `cmbcov-precompute`. The precompute takes 2–5% longer and
  about 40% more disk.

### Fixed

- **Multi-frequency B-mode covariances depended on the order of the
  spectra and could fail to be positive-definite**, the same bug 0.3.0
  fixed for T/E runs. Every element is now computed in its own
  orientation, so the result no longer depends on how frequencies or
  observables are listed, and is exact at the central multipole.
- **Adding kernel pairs to an existing cache was unsafe.** It could leave
  a wrong record of what the cache holds, and a precompute with different
  settings into the same directory could leave stale kernels that were
  then used. Each cache record now lists its kernel files and is merged on
  extension; kernel files not in the record are refused. Caches written by
  earlier versions still load as before.

### Added

- **`conditioning.txt`** beside every covariance: the eigenvalues of the
  correlation matrix, whether it is positive-definite, and its condition
  number, with a warning in the log if it is not positive-definite.
- **A normalisation check in `error_budget.txt`**, which ACC now writes for
  every run (TT-only and B-mode runs included, with the sections that do
  not apply marked as such). It checks the kernels against the MASTER
  normalisation on every diagonal and warns above 1%.

### Performance

- **A multi-frequency run is about 4× faster.** The ACC assembly now shares
  kernel products between blocks (identical up to floating-point rounding;
  see 0.4.1), and the PolSpice
  transform, D_ℓ scaling, debiasing and binning are folded into one
  projection per spectrum. On a three-frequency T/E run at ℓmax = 3500 the
  whole computation goes from about 10 to 2.5 minutes, with about 1 GB
  less memory. B-mode runs gain 4–6× on the assembly.
- The folding changes results at round-off level only: at most 2e-15 of
  the largest element on that run. Where it would not pay (nearly unbinned
  output, PolSpice off), the previous code runs and results are
  bit-identical.

## 0.3.0 — 2026-09-29

### Fixed

- **Band-power window functions** (`--save-windows`) were wrong whenever a
  beam, pixel window or transfer function was set, which includes every run
  using the default `pixwin`. They left the beam and pixel window off the
  input side, so applied to a theory spectrum they were off by the binned
  inverse of those factors: about 1.3% at ℓ = 3000 from the default pixel
  window alone, and up to a factor of 2 near ℓ = 3500 with a 90 GHz beam.
  They also no longer include `post_process_correction` or the
  transfer-function uncertainty inflation, which apply to the covariance
  only. **Regenerate any saved windows.** Covariance matrices were never
  affected.
- **Multi-frequency T/E ACC covariances could fail to be positive-definite.**
  In blocks between two different spectra (e.g. TE × EE), the lower triangle
  took an orientation that depended on the order in which the run listed
  its spectra. The CMB then no longer cancelled exactly in differences
  between frequency pairs, and a few near-null directions went negative.
  Each element is now computed in its own orientation, so the result no
  longer depends on spectrum order. Variances are unchanged and correlations
  move by at most ~1e-4; T-only and E-only runs are bit-identical. The T/E
  assembly does 37–61% more kernel products. Runs with a B observable are
  not yet covered.

### Changed

- **`term_selection`**: the default pair threshold is now `tolerance / 40`
  (was `tolerance / 4`), about 5× more accurate kernels for 1.5–1.8× the
  pairs kept. The feature is opt-in; the default `term_selection=None` is
  unchanged.

### Added

- **Validator warning for `dmax` against the binning.** ACC drops multipole
  pairs further apart than `dmax`. Correct bandpower error bars need
  `dmax` ≥ the widest bin width w; correct correlations between adjacent
  bandpowers need `dmax` ≥ 2w. The validator warns below either.

### Documented

- How to choose `dmax` for binned output (`docs/theory/acc.md` §4), with the
  measured effect on a 4% footprint.
- Where `term_selection`'s tolerance holds (kernels near the diagonal) and
  what it costs on the assembled covariance: a small low bias on TT
  variances at the default (`docs/theory/term_selection.md` §6).

### Development

- pytest collects only `tests/`.
  `tests/reference/criterion_validate.py` reproduces the term-selection
  numbers (`--mask` for your own footprint).

## 0.2.0 — 2026-09-23

### Breaking changes

- **`nl_is_biased` removed.** A parameter file that still contains the key is
  now refused with a validation error rather than silently ignored. `nl` is
  always the noise power spectrum of the map *as delivered to the estimator*:
  it carries no beam, no pixel window and no transfer function, and it is never
  multiplied by the data model. This is the MASTER convention (Hivon et al.
  2002, [astro-ph/0105302](https://arxiv.org/abs/astro-ph/0105302), their
  Eqs. (15)-(16)); the debiasing already divides each leg by
  $B_1 B_2 w^{\mathrm{pix}} F_\ell$, so the noise enters the reported error bars as
  $N_\ell / B^2_\ell$.

- **White-noise levels now follow that convention too, which changes numbers.**
  A dict of white-noise levels used to set `noise_is_biased = False`, which
  multiplied the flat level by $B^2 w^{\mathrm{pix}} F_\ell$ and so effectively
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
  $\sigma_P = \sqrt 2 \thinspace \sigma_T$ special case. One dict may mix the two forms.

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
