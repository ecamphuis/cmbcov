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
│   ├── covariance.py                    # Cov, CovarianceConfig, CovarianceMethod
│   ├── binning.py                       # BinningManager
│   ├── postprocess.py                   # CovariancePostProcessor
│   ├── mask.py                          # MaskWlm
│   ├── sht.py                           # ducc0_map2alm, ducc0_alm2map, alm2cl,
│   │                                     #   almxfl, cplx_spin_weighted_ylm
│   ├── grid.py                          # Gauss-Legendre grid primitives
│   ├── term_selection.py                # A-priori (m, m', M) term selection for ACC
│   ├── exact.py                         # exact_covariance, exact_covariance_row (+ _pol)
│   ├── spectra.py                       # SpectraLoader
│   ├── keys.py                          # SpecKey, CovKey, CovKeys
│   ├── bmode_wick.py                    # B-mode Wick-term generator, required_kernel_pairs
│   │
│   ├── approximations/                  # One module per reduced coupling kernel
│   │   ├── __init__.py                  # StrategyFactory + re-exports
│   │   ├── base.py                      # CovarianceStrategy (the interface)
│   │   ├── nka.py                       # NKAStrategy
│   │   ├── inka.py                      # INKAStrategy
│   │   ├── acc.py                       # ACCStrategy + precompute_acc_kernels
│   │   ├── acc_cache.py                 # On-disk kernel cache: paths, manifest, load/save
│   │   └── acc_budget.py                # Per-run ACC known-error report
│   │
│   ├── kernels/                         # Native MASTER/PolSpice coupling kernels
│   │   ├── coupling.py                  # coupling_kernels, xi_operator,
│   │   │                                #   wigner_d_table, mask_correlation_function
│   │   └── polspice.py                  # polspice_kernels, polspice_g_function,
│   │                                     #   apodization_function, PolSpiceKernels
│   │
│   ├── generator/                       # High-level driver
│   │   ├── generator.py                 # CovarianceMatrixGenerator
│   │   └── parameter_validation.py      # ParameterValidator, ParameterManager, PipelineConfig
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

### The covariance itself

- **`covariance.py`**: `Cov` owns the mask, caches the coupling kernels,
  selects an approximation strategy and assembles the per-spectrum blocks
  into the final bandpower covariance matrix; `CovarianceConfig`
  (dataclass) and `CovarianceMethod` (`ACC`/`NKA`/`INKA` enum). `Cov.__init__`
  rejects `lmax > 2 * nside` of the mask; `Cov.acc_internal_lmax()` gives the
  ACC internal band limit (see [`getting_started.md`](getting_started.md)).
- **`binning.py`**: `BinningManager` builds the binning matrix `B` so that
  `B @ C` gives bandpowers and `B @ Sigma @ B.T` the bandpower covariance;
  honours `lmin` (a bin straddling it is truncated).
- **`postprocess.py`**: `CovariancePostProcessor`, the PolSpice convolution
  ($\hat\Sigma = G \Sigma G^T$), $D_\ell$ scaling and calibration debiasing.
  See [`polspice.md`](polspice.md).

### Approximations (`approximations/`)

One module per choice of the reduced coupling kernel $\bar\Theta$.
`StrategyFactory` (in `approximations/__init__.py`) selects one from a
`CovarianceConfig`.

- **`base.py`**: `CovarianceStrategy`, the abstract interface. Subclasses
  differ only in the choice of $\bar\Theta$; the mask, the kernels and the
  block layout are shared, reached through `self.cov`.
- **`nka.py`**: `NKAStrategy` — delta functions,
  $\Sigma = 2\, C_\ell C_{\ell'}\, \Xi[W^2]$.
- **`inka.py`**: `INKAStrategy` — the renormalised MASTER kernel.
- **`acc.py`**: `ACCStrategy` (the recompute side, reached through a `Cov`)
  and `precompute_acc_kernels` (the one-off precompute, needing only the
  mask). `coupling_ellprange`, `COUPLING_CHANNELS`, `COUPLING_SPECTRA`,
  `CHANNEL_ALIASES`, `normalise_channel`, `acc_internal_lmax`,
  `acc_cached_kernel_size` and `contract_coupling_block` (the per-block
  kernel contraction, written as a NumPy/JAX-agnostic seam) also live here.
  The largest module in the package.
- **`acc_cache.py`**: the on-disk kernel cache — paths, manifest,
  load/save, legacy-name handling.
- **`acc_budget.py`**: the per-run ACC known-error report
  (`format_error_budget`; see [`approximations.md`](approximations.md)).

The exact (non-approximate) covariance of the paper's Sect. 3 lives in
`exact.py`, not here: it is a validator, not a strategy `Cov` selects.

### Mask, harmonics and keys

- **`mask.py`**: `MaskWlm` — everything derived from the survey footprint:
  the map, its harmonic coefficients, the mask power spectra $W_\ell$ and
  $W^2_\ell$, degraded copies for the ACC precompute, and the on-disk cache
  of the M/Msq/K/G kernel families under `<mask_path>/utils_<mask>/`.
- **`sht.py`**: every spherical-harmonic transform the package uses, on
  `ducc0` (`ducc0_map2alm`, `ducc0_alm2map`, `alm2cl`, `almxfl`,
  `cplx_spin_weighted_ylm`); `healpy` itself is reduced to FITS I/O, the
  pixel window table and alm index arithmetic.
- **`grid.py`**: Gauss-Legendre grids — exact synthesis/analysis
  (`gl_synthesis`/`gl_analysis`, real and complex, spin 0 and 2), the
  minimal band-limit for a product of band-limited fields
  (`gl_minimal_lmax`), and the ACC-precompute integrals
  (`spin_weighted_integrals_gl`, `banded_integrals_gl`).
- **`term_selection.py`**: `select_terms` and its building blocks
  (`pole_rotation`, `rotate_alm`, `mode_power`, `azimuthal_spectrum`) — the
  a-priori rule deciding which $(m, m', M)$ ACC-kernel terms carry weight,
  from the mask's harmonic content alone. See
  [`theory/term_selection.md`](theory/term_selection.md).
- **`exact.py`**: `exact_covariance_row`/`exact_covariance` (matrix-free
  exact pseudo $C_\ell$ covariance, TT, HEALPix or GL grid) and
  `exact_covariance_row_pol`/`exact_covariance_pol` (GL grid, full T/E/B).
- **`spectra.py`**: `SpectraLoader` — the data model: CMB spectra, beams,
  pixel window and noise for each frequency/Stokes combination.
- **`keys.py`**: `SpecKey`/`CovKey`/`CovKeys` — which spectra exist, the
  Wick contractions between them, and the covariance matrix block layout.
- **`bmode_wick.py`**: the B-mode Wick-term generator (`block_wick_terms`,
  `covkey_wick_terms`) and `required_kernel_pairs`, the ACC coupling-kernel
  channel pairs a set of observables needs; `term_class`, `spin_weight`,
  `leakage_legs` feed the per-term normalisation. See
  [`theory/bmode_kernels.md`](theory/bmode_kernels.md).

### Kernels (`kernels/`)

- **`coupling.py`**: native (Fortran-free) MASTER/PolSpice coupling
  kernels via Gauss-Legendre quadrature of the real-space Legendre form,
  plus `wigner_d_table`, `mask_correlation_function`, `xi_operator`.
- **`polspice.py`**: native PolSpice kernels (`polspice_kernels`,
  `polspice_g_function`, `apodization_function`).

Both are written to be JAX-friendly: fixed shapes, no data-dependent
control flow.

### Generator (`generator/`)

- **`generator.py`**: `CovarianceMatrixGenerator`, the high-level driver
  (parameter loading, beams/pixel-window/spectra/noise loading,
  computation, saving).
- **`parameter_validation.py`**: `ParameterValidator` (required/optional
  parameter checks, consistency and file-path checks), `ParameterManager`
  (YAML loading, defaults, output-directory versioning) and
  `PipelineConfig` (the single validated object every consumer reads);
  `AccPrecomputeConfig` for the `acc_precompute` block. See
  [`parameters.md`](parameters.md).

### Scripts (`scripts/`)

- **`compute_covariance.py`** — `cmbcov-cov` (parameter file
  positionally or with `--parameter-file`).
- **`validate_parameters.py`** — `validate-parameters`, validates a
  parameter file without running the computation.
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

Run with `pytest tests/`. `tests/data/` holds golden fixtures tracked in
git: a small mask, and the recorded end-to-end covariance and ACC coupling
kernels it produces — the safety net for structural change; regenerate them
deliberately (`tests/data/regenerate_baselines.py`) if a refactor changes
either. `tests/reference/` holds independent implementations the tests use
(e.g. `gaunt.py`, a literal Gaunt-sum evaluation of the ACC mode-coupling
integrals against `sympy`).
