# Test data

Small, fast-to-load fixtures used by the test suite and by the example
scripts. Files that are cache artefacts (reproducible from the fixtures
below) are not committed and are listed only for completeness.

## Inputs (committed)

- **`baseline_mask.fits`** -- a small (`nside=32`, ~3.6% sky fraction)
  apodised HEALPix mask used as the standard test fixture. It is a
  synthetic development mask, not real survey data, chosen only for its
  size (fast tests) and smooth apodisation (non-trivial mask spectrum).
- **`baseline_cls.dat`** -- a lensed CMB power spectrum table (columns
  `ell TT EE BB TE`, `D_ell` in `uK^2`) restricted to low multipoles
  (`ell <= 65`), used with `baseline_mask.fits` for fast, deterministic
  tests. A toy table, not a specific external data release.
- **`baseline_params.yml`** -- the parameter file `regenerate_baselines.py`
  and `test_end_to_end_baseline.py` run to produce a small, deterministic
  covariance from the two fixtures above (`PLACEHOLDER_OUT` /
  `PLACEHOLDER_DATA` are substituted by the test fixture / script at run
  time).

## Golden outputs (committed)

Generated from the inputs above by `regenerate_baselines.py`; see that
script's docstring for when it is legitimate to regenerate them and how to
record the shift when you do:

- **`baseline_covariance.npy`**, **`baseline_lbins.npy`** -- pinned by
  `tests/test_end_to_end_baseline.py`.
- **`baseline_acc_coupling.npz`** -- pinned by `tests/test_acc_coupling.py`.

## Cache artefacts (not committed, safe to delete)

- **`utils_baseline_mask/`** -- MASTER/PolSpice coupling-kernel caches for
  `baseline_mask.fits`, written automatically by
  `cmbcov.mask.MaskWlm` the first time a test needs them
  (see the "Kernel cache keying" note in `cmbcov/mask.py`).
  Regenerated on demand; deleting this directory just makes the next test
  run recompute it.

## Public external data (not committed, optional)

- **`planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lensedCls.dat`** --
  the Planck 2018 `base_plikHM_TTTEEE_lowl_lowE_lensing` best-fit lensed
  `C_ell` file, public data from the Planck Legacy Archive (as distributed
  with the Planck chains / CAMB `lensedCls` output). Not required by the
  test suite; used only as the default CMB spectrum by
  `examples/mask_covariance_comparison.py` when present locally
  (`.gitignore`d; pass `--spectrum-file` to use a different one, e.g.
  `baseline_cls.dat`, when it is absent).
