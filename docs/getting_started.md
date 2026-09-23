# Getting started

This walks through one run of the package end to end: what it reads, how it
is driven, and what it writes.

## Inputs

- **Mask.** A HEALPix FITS map, `mask_name` looked up in `mask_path` (both
  required). Its resolution bounds `lmax`: `Cov.__init__` rejects
  `lmax > 2 * nside` of the mask.
- **Spectra.** `cmb_spectrum` is a tabulated file (or `.fits`, read with
  `healpy.read_cl`) with a leading `ell` column and then:
  - 1 column: TT only;
  - 4 columns: TT, EE, BB, TE (CAMB style) — ET is taken equal to TE, and
    TB/EB default to zero (with a warning) if asked for;
  - 9 columns: TT, EE, BB, TE, TB, EB, ET, BT, BE, read as given.

  `foregrounds` (added to the CMB spectrum) and `nl` (noise, see below)
  follow the same column rules. Units are `C_\ell` by default; set
  `spectrum_units: Dl` (or a per-input `<input>_units`) if the file is
  $D_\ell = \ell(\ell+1) C_\ell / 2\pi$, the convention of CAMB's
  `*_lensedCls.dat`. See [`parameters.md`](parameters.md) for the full rule.
- **Beams.** `beams`: a `{frequency: FWHM_arcmin}` dict (Gaussian beams are
  generated) or a file path.
- **Noise.** `nl` is the noise power spectrum of the map *as delivered to the
  estimator*: no beam, no pixel window, no transfer function (the MASTER
  convention, Hivon et al. 2002, Eqs. (15)-(16)). The debiasing divides each
  leg by the data model, so the noise reaches the error bars as
  $N_\ell / B^2_\ell$. Three forms: one number per frequency (temperature
  white-noise level in $\mu K \cdot \mathrm{arcmin}$, with
  $N^{EE} = N^{BB} = 2 N^{TT}$), `[sigma_T, sigma_P]` per frequency, or a
  tabulated file such as `nl_{}.txt`. Dict keys are *single* frequencies;
  cross-frequency noise is zero. See [`parameters.md`](parameters.md).

## The ACC workflow

The survey mask is normally fixed early in an analysis. The ACC coupling
kernels for that mask are the expensive part of ACC, and depend only on the
mask and `centralell` — not on the spectra, noise, beams or binning — so they
are computed **once** and the covariance is **recomputed many times**
afterwards. One parameter file drives both steps: the kernels go to
`<cov_path>/covariance_coupling/`, and every later `cmbcov-cov` run
of that file finds them there.

```bash
cmbcov-precompute parameters.yml    # once per mask (the expensive step)
cmbcov-cov parameters.yml           # every run: spectra, noise, beams, binning
```

NKA and INKA need no precompute step.

## Running from a parameter file

```bash
validate-parameters -p parameters.yml   # cheap: checks the file, touches nothing else
cmbcov-precompute parameters.yml           # ACC only
cmbcov-cov parameters.yml
```

`cmbcov-precompute --dryrun parameters.yml` validates the file and prints the
computation plan without touching the mask. A minimal ACC parameter file,
runnable against the mask shipped with the repository:

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

`centralell: 16` and `dmax: 4` are toy values for this small mask, chosen so
the example runs in seconds; see [`acc_precomputation.md`](acc_precomputation.md)
for sizing them on a real footprint. See [`parameters.md`](parameters.md) for
every key.

## Running from the Python API

The same two steps, in Python, on the small mask above:

```python
import numpy as np
from cmbcov.approximations.acc import precompute_acc_kernels
from cmbcov.covariance import Cov, CovarianceConfig, CovarianceMethod
from cmbcov.keys import CovKeys

# Step 1: precompute -- once per mask. No Cov needed.
precompute_acc_kernels(
    "baseline_mask.fits",
    "./coupling_kernels/",
    centralell=16,
    dmax=4,
    mask_path="./tests/data/",
    nside=16,
    grid="gl",
    lw=10,
    spectra=("TT",),   # restrict to the TT kernel; default computes all 25
)

# Step 2: recompute -- rerun for every new set of spectra, reusing the cache.
config = CovarianceConfig(method=CovarianceMethod.ACC, lmax=48, dmax=4, centralell=16)
cov = Cov("baseline_mask.fits", config=config, mask_path="./tests/data/", save_dir="./coupling_kernels/")
cov_keys = CovKeys(["T"], ["090GHz"])

# ACC reads the spectra past lmax, to lmax_int (see "The ACC internal band limit" below).
lmax_int = cov.acc_internal_lmax()
cl = {
    fkey: {skey: np.ones(lmax_int) for skey in cov_keys.combined_stokes()}
    for fkey in cov_keys.combined_frequencies()
}
binned_ells, bin_matrix, covariance = cov.compute_covariance_matrix(
    [2, 12, 24, 36, 48], cov_keys, cl
)
```

`CovarianceMatrixGenerator("parameters.yml").run_full_analysis()` is what
`cmbcov-cov` wraps: it validates the file, loads the data model
(spectra, beams, noise), computes the covariance and writes it to disk.
Building `Cov` directly, as above, keeps `(binned_ells, bin_matrix,
covariance)` in memory instead.

For NKA and INKA, drop the precompute step and the ACC-only `dmax`/
`centralell`: `CovarianceConfig(method=CovarianceMethod.NKA, lmax=..., Dl=True)`,
then `cov.compute_covariance_matrix(lbins, cov_keys, cl)` directly (spectra of
exactly `lmax` multipoles, not `lmax_int`).

### The ACC internal band limit

ACC evaluates entry $(\ell, \ell')$ with the `centralell` kernel translated to
$\min(\ell, \ell')$, so that no entry near `lmax` loses part of its
translated kernel, every spectrum must reach

```math
\ell_{\mathrm{int}} = \ell_{\max} + \max(0,\, S - 1 - \ell_\ast)
```

with $S$ the kernel size (`2 * nside` of the ACC precompute) and $\ell_\ast$
`centralell`. `Cov.acc_internal_lmax()` returns this. The covariance is still
reported, transformed and binned to `lmax`. From a parameter file,
`cmbcov-cov` builds every spectrum, beam, noise and transfer-function
input to $\ell_{\mathrm{int}}$ itself; a tabulated input that stops short
raises, naming the file, its last multipole and $\ell_{\mathrm{int}}$.

## Outputs

Each `cmbcov-cov` run creates a fresh versioned directory under
`cov_path` (`v0/`, `v1/`, ...) holding:

- `<cov_name>` — the covariance matrix, `np.savetxt`, in $C_\ell$ or
  $D_\ell$ units (`Dl: true`). Block layout follows `covariance_keys`
  (frequency x Stokes combinations), each block `n_bins x n_bins`.
- `lbins.dat` — the bandpower centres, one per output row/column block.
- a copy of the parameter file that produced the run.
- `error_budget.txt` — only for an ACC run with a polarised leg: the E→B
  leakage bound and the (unbounded) Eq. 33 translation-error range; see
  [`approximations.md`](approximations.md).
- `bb_leakage_warning.txt` — only for the BB-only NKA/INKA escape hatch; see
  [`polarisation.md`](polarisation.md).
- `windows/` — only with `cmbcov-cov --save-windows`: one
  plain-text file per spectrum and frequency pair; see "Bandpower window
  functions" below.

The ACC coupling kernels themselves live one level up, in
`<cov_path>/covariance_coupling/` (`.npy` files plus a JSON manifest per
multipole pair), so they are shared by every version directory of the file.

## Bandpower window functions

`cmbcov-cov --save-windows` writes the linear map from the fiducial
theory spectrum (`cmb_spectrum`) to the *expected* value of each reported
bandpower,

```math
\langle \hat C^X_b \rangle = \sum_\ell W^X_{b\ell}\, C^X_\ell ,
```

as one plain-text file per two-letter spectrum and frequency pair among
this run's observables, under a `windows/` subdirectory beside the
covariance — the owner's existing convention. A file is named
`<X>_<f1>x<f2>_window_functions.txt`, with the frequency labels stripped of
their `GHz` suffix and leading zeros (e.g. `EE_90x150_window_functions.txt`
for the `090GHz` x `150GHz` pair; a frequency label that does not fit that
pattern is kept verbatim instead). Its first line is a comment naming the
spectrum and the full frequency labels (e.g. `# Band powers window
functions for EE 090GHz150GHz.`), followed by one row per multipole from
`lmin` to `lmax - 1`: the first column is $\ell$ itself, the remaining
columns one per bandpower — a plain `numpy.savetxt`/`numpy.loadtxt` round
trip. Nothing is written without the flag, and a run's covariance is
unaffected either way.

With `polspice_postprocess: true` (the default) the reported spectrum is
the decoupled PolSpice bandpower, and `W` uses the mask-independent PolSpice
mean kernels of [`polspice.md`](polspice.md): $^0K$ for `TT`,
$^{\times}K$ for `TE`, `ET`, `TB` and `BT`, and the same $^{-2}K$ for `EE`,
`BB` and `EB` — with no `EE` <-> `BB` mixing in the mean; see
[`theory/bmode_kernels.md`](theory/bmode_kernels.md), Sect. 7. With
`polspice_postprocess: false` the reported spectrum is the pseudo
$C_\ell$ itself, and `W` uses the MASTER coupling instead, with a genuine
`EE` <-> `BB` mixing term in the mean. A one-spectrum-per-file text format
cannot represent that mixing, so a run with both `EE` and `BB` among the
observables and `polspice_postprocess: false` makes `--save-windows` raise
instead of writing anything; enable `polspice_postprocess`, or drop
`--save-windows`, for that run.

$D_\ell$ scaling (`Dl: true`) is folded into `W`'s output axis, matching the
covariance's own convention; the input axis stays $C_\ell$. Each file also
includes that frequency pair's beam, pixel window, transfer function and
calibration debiasing — the same per-leg factor
`CovariancePostProcessor.apply_debiasing` applies to the covariance
(`SpectraLoader.data_model`/`debiasing_dict`) — so `W` maps the fiducial
theory spectrum straight to the reported bandpower, every multiplicative
factor included.

## Caching and reusing results

- **ACC coupling kernels** (`<cov_path>/covariance_coupling/`): written once
  by `cmbcov-precompute`, read by every later `cmbcov-cov` run of the
  same file. A manifest records the mask digest, `centralell`, grid and
  quadrature settings; a mismatch (different mask or `centralell`) is
  refused rather than silently reused.
- **Raw pseudo $C_\ell$ blocks** (`save_raw_blocks: true`, off by default):
  caches each block in the run's own output directory, keyed by a manifest
  of everything it depends on (method, `lmax`, the ACC kernel cache, the
  mask, a digest of the spectra read). Reused only on an exact match.
- **Native coupling kernels** (M, Msq, G, K, used by NKA/INKA/PolSpice):
  cached under `<mask_path>/utils_<mask>/`, keyed by a manifest, shared
  across runs and methods on the same mask.
