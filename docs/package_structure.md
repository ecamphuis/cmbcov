# Package structure

For contributors: where each concept lives. The same short-form mapping is
in the package docstring (`cmbcov/__init__.py`); this
page adds the file listing and per-module detail.

## Layout

```
cmbcov/
├── pyproject.toml
├── README.md
├── LICENSE
│
├── cmbcov/           # Main package
│   ├── __init__.py                      # Exports CovarianceMatrixGenerator, Cov,
│   │                                     #   ParameterValidator, ParameterManager, PipelineConfig
│   │
│   ├── spectra.py                       # SpectraLoader
│   │
│   ├── harmonics/                       # Spherical-harmonic and quadrature primitives
│   │   ├── __init__.py                  # Re-exports of sht and grid
│   │   ├── sht.py                       # ducc0_map2alm, ducc0_alm2map, alm2cl,
│   │   │                                 #   almxfl, cplx_spin_weighted_ylm
│   │   └── grid.py                      # Gauss-Legendre grid primitives
│   │
│   ├── mask/                            # The survey mask and its coupling kernels
│   │   ├── __init__.py                  # Re-exports: MaskWlm, coupling_kernels, polspice_kernels, ...
│   │   ├── footprint.py                 # MaskWlm
│   │   ├── coupling.py                  # coupling_kernels, xi_operator,
│   │   │                                 #   wigner_d_table, mask_correlation_function
│   │   └── polspice.py                  # polspice_kernels, polspice_g_function,
│   │                                     #   apodization_function, PolSpiceKernels
│   │
│   ├── exact/                           # The exact covariance (validator, not a strategy)
│   │   ├── __init__.py                  # Re-exports; the algorithm and the grid choice
│   │   ├── common.py                    # FIELDS, SPECTRA, mask_coupling_width, shared helpers
│   │   ├── healpix.py                   # The HEALPix path: transforms, correlation column
│   │   ├── gl.py                        # The TT Gauss-Legendre path (per-column and batched)
│   │   ├── pol.py                       # The polarised GL path (T/E/B)
│   │   ├── plan.py                      # Memory plan of the batched rows (max_memory_gb)
│   │   ├── parallel.py                  # Rows over processes (nprocs): the worker functions
│   │   └── covariance.py                # exact_covariance, exact_covariance_row (+ _pol)
│   │
│   ├── approximations/                  # One module per reduced coupling kernel
│   │   ├── __init__.py                  # StrategyFactory + re-exports
│   │   ├── base.py                      # CovarianceStrategy (the interface)
│   │   ├── nka.py                       # NKAStrategy
│   │   ├── inka.py                      # INKAStrategy
│   │   └── acc/                         # The ACC approximation
│   │       ├── __init__.py              # Re-exports: ACCStrategy, precompute_acc_kernels, ...
│   │       ├── channels.py              # Kernel channels, aliases, spectra selection
│   │       ├── producers.py             # Mask integrals (GL, HEALPix), term-selection plan
│   │       ├── contraction.py           # Blocked kernel contraction and its memory model
│   │       ├── store.py                 # On-disk store of the central coefficient set
│   │       ├── precompute.py            # precompute_acc_kernels, coupling_ellprange
│   │       ├── bands.py                 # Recompute helpers: windows, cache needs, band sums
│   │       ├── strategy.py              # ACCStrategy
│   │       ├── cache.py                 # On-disk kernel cache: paths, manifest, load/save
│   │       ├── budget.py                # Per-run ACC known-error report
│   │       ├── term_selection.py        # A-priori (m, m', M) term selection for ACC
│   │       └── bmode_wick.py            # B-mode Wick-term generator, required_kernel_pairs
│   │
│   ├── assembly/                        # Assembling the covariance matrix
│   │   ├── __init__.py                  # Re-exports: Cov, CovKey, BinningManager, ...
│   │   ├── covariance.py                # Cov, CovarianceConfig, CovarianceMethod
│   │   ├── keys.py                      # SpecKey, CovKey, CovKeys
│   │   ├── binning.py                   # BinningManager
│   │   ├── postprocess.py               # CovariancePostProcessor
│   │   ├── windows.py                   # build_window_functions (bandpower windows)
│   │   ├── conditioning.py              # conditioning_report
│   │   └── filtering.py                 # Filter-and-bin sum-rule ratio rho for TT blocks
│   │
│   ├── config/                          # High-level driver
│   │   ├── __init__.py
│   │   ├── generator.py                 # CovarianceMatrixGenerator
│   │   └── validation.py                # ParameterValidator, ParameterManager, PipelineConfig
│   │
│   ├── scripts/                         # Console-script entry points
│   │   ├── compute_covariance.py        # `cmbcov-cov`
│   │   ├── validate_parameters.py       # `validate-parameters`
│   │   ├── precompute_acc.py            # `cmbcov-precompute`
│   │   └── convert_acc_cache.py         # `convert-acc-cache`
│   │
│   └── utils/
│       ├── file_utils.py                # get_git_revision_short_hash, get_versioned_path
│       ├── array_utils.py               # dl_to_cl, read_spectrum_file, safe_divide,
│       │                                 #   add_nested_dicts, mult_nested_dicts
│       ├── healpy_utils.py              # get_nside_from_ell, ensure_ducc0_compatible_dtype
│       └── threading_utils.py           # get_optimal_nthreads
│
├── legacy/fortran/                      # Outside the package; not shipped; manual cross-check
│
├── tests/                               # Test suite (pytest)
│   ├── reference/                       # Test-only reference implementations (e.g. gaunt.py)
│   └── data/                            # Golden fixtures tracked in git
│
└── examples/                            # Top-level usage examples
```

## Module responsibilities

### Assembly (`assembly/`)

- **`assembly/covariance.py`**: `Cov` owns the mask, caches the coupling kernels,
  selects an approximation strategy and assembles the per-spectrum blocks
  into the final bandpower covariance matrix; `CovarianceConfig`
  (dataclass) and `CovarianceMethod` (`ACC`/`NKA`/`INKA` enum). `Cov.__init__`
  rejects `lmax > 2 * nside` of the mask; `Cov.acc_internal_lmax()` gives the
  ACC internal band limit (see [`getting_started.md`](getting_started.md)).
- **`assembly/keys.py`**: `SpecKey`/`CovKey`/`CovKeys` — which spectra exist, the
  Wick contractions between them, and the covariance matrix block layout.
- **`assembly/binning.py`**: `BinningManager` builds the binning matrix `B` so that
  `B @ C` gives bandpowers and `B @ Sigma @ B.T` the bandpower covariance;
  honours `lmin` (a bin straddling it is truncated).
- **`assembly/postprocess.py`**: `CovariancePostProcessor`, the PolSpice convolution
  ($\hat\Sigma = G \Sigma G^T$), $D_\ell$ scaling and calibration debiasing.
  See [`polspice.md`](polspice.md).
- **`assembly/filtering.py`**: the optional filter-and-bin correction of TT x TT blocks
  (`map_filter` block): the high-pass and low-pass profiles, the stochastic filtered
  sum-rule ratio `rho` of the mask at a few nodes, its interpolation and
  cache, and the transfer-function correction `rho / (fl_left fl_right)`
  that `Cov` applies to the raw block. See
  [`theory/filter_and_bin.md`](theory/filter_and_bin.md).
- **`assembly/windows.py`**: `build_window_functions`, the bandpower window functions: the
  linear map from the fiducial theory spectrum to the expected reported
  bandpower.
- **`assembly/conditioning.py`**: `conditioning_report`, how well conditioned the delivered
  covariance is (on its correlation matrix); it changes nothing.

### Approximations (`approximations/`)

One module per choice of the reduced coupling kernel $\bar\Theta$.
`StrategyFactory` (in `approximations/__init__.py`) selects one from a
`CovarianceConfig`.

- **`base.py`**: `CovarianceStrategy`, the abstract interface. Subclasses
  differ only in the choice of $\bar\Theta$; the mask, the kernels and the
  block layout are shared, reached through `self.cov`.
- **`nka.py`**: `NKAStrategy` — delta functions,
  $\Sigma = 2\thinspace C_\ell C_{\ell'}\thinspace \Xi[W^2]$.
- **`inka.py`**: `INKAStrategy` — the renormalised MASTER kernel.
- **`acc/strategy.py`**: `ACCStrategy`, the recompute side (reached through a
  `Cov`). The largest module of the subpackage.
- **`acc/bands.py`**: what `ACCStrategy` needs to turn cached kernels and a
  spectrum into a block: `acc_window_pad`, `acc_internal_lmax`,
  `acc_cached_kernel_size`, `require_acc_cache_pairs`, the padded spectra and
  the banded sums.
- **`acc/precompute.py`**: `precompute_acc_kernels` (the one-off precompute,
  needing only the mask) and `coupling_ellprange`.
- **`acc/channels.py`**: `COUPLING_CHANNELS`, `COUPLING_SPECTRA`,
  `CHANNEL_ALIASES`, `normalise_channel` and the resolution of a requested
  subset of channels.
- **`acc/producers.py`**: the mask integrals the kernels are contracted from,
  on a GL grid or a HEALPix map, the term-selection plan and the coefficient
  provider.
- **`acc/contraction.py`**: `contract_coupling_block` (the per-block kernel
  contraction, written as a NumPy/JAX-agnostic seam), the working-set and
  block-size model and the accumulation over all `(ell, ell')` pairs.
- **`acc/store.py`**: the on-disk store of the central coefficient set used
  when it does not fit in the memory budget.
- **`acc/cache.py`**: the on-disk kernel cache — paths, manifest,
  load/save, legacy-name handling.
- **`acc/budget.py`**: the per-run ACC known-error report
  (`format_error_budget`; see [`approximations.md`](approximations.md)).
- **`acc/term_selection.py`**: `select_terms` and its building blocks
  (`pole_rotation`, `rotate_alm`, `mode_power`, `azimuthal_spectrum`) — the
  a-priori rule deciding which $(m, m', M)$ ACC-kernel terms carry weight,
  from the mask's harmonic content alone. See
  [`theory/term_selection.md`](theory/term_selection.md).
- **`acc/bmode_wick.py`**: the B-mode Wick-term generator (`block_wick_terms`,
  `covkey_wick_terms`) and `required_kernel_pairs`, the ACC coupling-kernel
  channel pairs a set of observables needs; `term_class`, `spin_weight`,
  `leakage_legs` feed the per-term normalisation. See
  [`theory/bmode_kernels.md`](theory/bmode_kernels.md).

The exact (non-approximate) covariance of the paper's Sect. 3 lives in
`exact/`, not here: it is a validator, not a strategy `Cov` selects.

### Harmonics (`harmonics/`)

- **`harmonics/sht.py`**: every spherical-harmonic transform the package uses, on
  `ducc0` (`ducc0_map2alm`, `ducc0_alm2map`, `alm2cl`, `almxfl`,
  `cplx_spin_weighted_ylm`); `healpy` itself is reduced to FITS I/O, the
  pixel window table and alm index arithmetic.
- **`harmonics/grid.py`**: Gauss-Legendre grids — exact synthesis/analysis
  (`gl_synthesis`/`gl_analysis`, real and complex, spin 0 and 2), the
  minimal band-limit for a product of band-limited fields
  (`gl_minimal_lmax`), the ACC-precompute integrals
  (`spin_weighted_integrals_gl`, `banded_integrals_gl`), and the Legendre
  tables (spin 0 and spin 2) and ring modes of the batched exact rows
  (`gl_legendre_table`, `gl_ring_modes`).

### Mask (`mask/`)

- **`footprint.py`**: `MaskWlm` — everything derived from the survey footprint:
  the map, its harmonic coefficients, the mask power spectra $W_\ell$ and
  $W^2_\ell$, degraded copies for the ACC precompute, and the on-disk cache
  of the M/Msq/K/G kernel families under `<mask_path>/utils_<mask>/`.
- **`coupling.py`**: native (Fortran-free) MASTER/PolSpice coupling
  kernels via Gauss-Legendre quadrature of the real-space Legendre form,
  plus `wigner_d_table`, `mask_correlation_function`, `xi_operator`.
- **`polspice.py`**: native PolSpice kernels (`polspice_kernels`,
  `polspice_g_function`, `apodization_function`).

Both kernel modules are written to be JAX-friendly: fixed shapes, no
data-dependent control flow.

### Exact covariance (`exact/`)

- **`covariance.py`**: `exact_covariance_row`/`exact_covariance` (matrix-free
  exact pseudo $C_\ell$ covariance, TT, HEALPix or GL grid) and
  `exact_covariance_row_pol`/`exact_covariance_pol` (GL grid, full T/E/B);
  on GL the columns of a row are batched within `max_memory_gb`.
- **`gl.py`**, **`pol.py`**: the TT and the polarised Gauss-Legendre column
  computations, per column and batched; **`healpix.py`**: the HEALPix path;
  **`plan.py`**: the memory plan of the batched rows; **`parallel.py`**: the
  worker functions of `nprocs > 1`; **`common.py`**: fields, spectra,
  `mask_coupling_width`. The algorithm is described in the docstring of
  `exact/__init__.py`.

### Configuration and data (`config/`, `spectra.py`)

- **`generator.py`**: `CovarianceMatrixGenerator`, the high-level driver
  (parameter loading, beams/pixel-window/spectra/noise loading,
  computation, saving).
- **`validation.py`**: `ParameterValidator` (required/optional
  parameter checks, consistency and file-path checks), `ParameterManager`
  (YAML loading, defaults, output-directory versioning) and
  `PipelineConfig` (the single validated object every consumer reads);
  `AccPrecomputeConfig` for the `acc_precompute` block. See
  [`parameters.md`](parameters.md).
- **`spectra.py`**: `SpectraLoader` — the data model: CMB spectra, beams,
  pixel window and noise for each frequency/Stokes combination.

### Scripts (`scripts/`)

- **`compute_covariance.py`** — `cmbcov-cov` (parameter file
  positionally or with `--parameter-file`).
- **`validate_parameters.py`** — `validate-parameters`, validates a
  parameter file (positionally or with `-p`) without running the
  computation.
- **`precompute_acc.py`** — `cmbcov-precompute`, the one-off ACC
  coupling-kernel precompute for a parameter file.
- **`convert_acc_cache.py`** — `convert-acc-cache`, converts a legacy
  `.txt` ACC coupling-kernel cache to `.npy` in place.

### Utilities (`utils/`)

- **`file_utils.py`**: git revision hash and versioned output paths.
- **`array_utils.py`**: `dl_to_cl`, `read_spectrum_file`, `safe_divide`,
  nested-dict arithmetic (`add_nested_dicts`, `mult_nested_dicts`).
- **`healpy_utils.py`**: nside selection from a multipole
  (`get_nside_from_ell`), `ducc0` dtype coercion
  (`ensure_ducc0_compatible_dtype`).
- **`threading_utils.py`**: thread-count selection (`get_optimal_nthreads`).

### Legacy Fortran cross-check (`legacy/fortran/`)

Outside the package, not shipped, not imported by anything under
`cmbcov/`. A manual cross-check of the native kernel
code against the original Fortran `master_kernels`/`cor2cl` executables;
building it needs a HEALPix-f90 + cfitsio toolchain. See
`legacy/fortran/README.md`.

### Tests (`tests/`)

Run with `pytest tests/` (the fast suite; `-m slow` runs only the slow tests,
`-m "slow or not slow"` everything). `tests/data/` holds golden fixtures tracked in
git: a small mask, and the recorded end-to-end covariance and ACC coupling
kernels it produces — the safety net for structural change; regenerate them
deliberately (`tests/data/regenerate_baselines.py`) if a refactor changes
either. `tests/reference/` holds independent implementations the tests use
(e.g. `gaunt.py`, a literal Gaunt-sum evaluation of the ACC mode-coupling
integrals against `sympy`).
