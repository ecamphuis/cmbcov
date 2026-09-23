# Parameter-file reference

The authoritative source of every key is
[`cmbcov/generator/parameter_validation.py`](../cmbcov/generator/parameter_validation.py);
[`examples/parameters_example.yml`](../examples/parameters_example.yml) is a
runnable, commented template. This page groups the keys and states their
type, default and constraints.

## Required keys

Ten keys are always required.

| Key | Type | Meaning |
| --- | --- | --- |
| `cov_path` | str | Output root; a versioned subdirectory (`v0/`, `v1/`, ...) is created under it. |
| `cov_name` | str | File name of the output covariance matrix. |
| `frequencies` | list of str | Frequency channel labels, e.g. `['090GHz', '150GHz']`. |
| `lmax` | int | Maximum multipole. Must satisfy `lmax <= 2 * nside` of the mask. |
| `lmin` | int | Minimum multipole; must be `< lmax`. Multipoles below it carry no binning weight; a bin straddling it is truncated, not dropped. |
| `bins` | list of `[start, stop, step]` | Binning scheme; the largest bin edge must not exceed `lmax`. |
| `mask_name` | str | Mask FITS file name, looked up in `mask_path`. |
| `mask_path` | str | Directory containing `mask_name`. Required even as `./`. |
| `covariance_approximation` | str | One of `acc`, `nka`, `inka`. |

Exactly one of `stokes` or `observables` (below) is also required; the
validator raises a combined error if both or neither are given.

## Observables

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `stokes` | list of `T`/`E` | — | Shorthand: `[T]` -> `observables: [TT]`; `[T, E]` -> `observables: [TT, EE, TE]`. A `B` here is rejected — use `observables` instead. |
| `observables` | list of str | — | Explicit spectra, a subset of `TT, EE, BB, TE, TB, EB` (`ET`, `BT`, `BE` accepted as aliases). General form; needed for any B-mode spectrum. |
| `parity_mixed_blocks` | bool | `false` | Include a block between a parity-even (`TT, EE, TE, BB`) and a parity-odd (`TB, EB`) spectrum. Needs `TB` or `EB` plus at least one parity-even observable in `observables`; rejected otherwise. |

`BB`, `TB` and `EB` need `covariance_approximation: acc` (NKA and INKA are
degenerate for BB by construction), except `observables: [BB]` alone, which
NKA and INKA also accept as a leakage-neglected escape hatch (cross-frequency
BB included). See [`polarisation.md`](polarisation.md).

## ACC-only keys

Required when `covariance_approximation: acc`; ignored (with a warning if
`acc_precompute` is also set) otherwise.

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `dmax` | int | `None` | Number of diagonal bands `\|l - l'\| < dmax` ACC computes. Must be `>= 1`. |
| `centralell` | int | `None` | Central multipole of the coupling-kernel computation. Must be `>= 0`; `<= 150` warns about accuracy. |

### `acc_precompute` block

Optional; every key inside it is optional. Governs how `cmbcov-precompute`
computes the kernels (see [`acc_precomputation.md`](acc_precomputation.md)).

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `nside` | int | derived from `centralell + dmax` | ACC working resolution; kernels are `2 * nside` square. Must be a power of two with `grid: healpix`. |
| `grid` | `gl` \| `healpix` | `gl` | Quadrature backend. `gl` is exact for a band-limited mask and supports T, E and B; `healpix` is the pixelised alternative. |
| `lw` | int | `3 * nside - 1` | Mask band-limit, `grid: gl` only. Error if given with `grid: healpix`. |
| `spectra` | list of str | all five (`TT, DD, LL, TD, DT`) | Kernel channels to compute, a subset of `TT, DD, LL, TD, DT, TL, LT, DL, LD` (old names `TT, EE, BB, TE, ET` accepted). Must cover what `observables` needs — a run that would fail its first kernel load is rejected at validation instead. |
| `max_memory_gb` | float | 2 | Peak-memory budget of the contraction; must be `> 0`. |

`grid: gl` is required (not merely default) with any B-mode observable: the
per-Wick-term normalisation reads raw kernel sums, which only the GL grid
provides.

## Data-model inputs

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `beams` | dict or str | `None` | `{frequency: FWHM_arcmin}` (Gaussian beams generated), or a file path. |
| `pixwin` | int or str | `8192` | HEALPix nside for the pixel window, or a file path. |
| `cmb_spectrum` | str, float or dict | `None` | Tabulated spectrum file (or `.fits`, read with `healpy.read_cl`); column layout in [`getting_started.md`](getting_started.md). |
| `foregrounds` | str | `None` | Foreground spectrum file (format string over frequency pairs), added to `cmb_spectrum`. |
| `nl` | dict or str | `{}` | Noise of the map **as delivered to the estimator**, in one of three forms (below). `noise` is accepted as a legacy alias. |
| `fl` | str | `None` | Transfer function (format string over frequency pairs), multiplying the signal spectra. Values must lie in `[0, 1]`. |
| `hl` | str | `None` | Accepted and stored, but not currently read or applied anywhere. |
| `post_process_correction` | str or dict | `None` | Multiplicative correction factor(s) applied after the covariance is computed, as a filename or `{name: filename}`; entries are multiplied together. |
| `inpainting_rescaling` | float | `1.0` | Reserved multiplicative correction; the code path that would apply it is currently disabled, so setting this has no effect. |

### The three forms of `nl`

`nl` is the noise power spectrum of the map as delivered to the estimator. It
carries **no** beam, **no** pixel window and **no** transfer function, and it
is never multiplied by the data model. This is the MASTER convention (Hivon et
al. 2002, [astro-ph/0105302](https://arxiv.org/abs/astro-ph/0105302), their
Eqs. (15)-(16)),

$$\langle \tilde C_\ell \rangle = M_{\ell\ell'} F_{\ell'} B^2_{\ell'} \langle C_{\ell'} \rangle + \langle \tilde N_\ell \rangle ,
\qquad
\Delta C_\ell \simeq \left( C_\ell + \frac{N_\ell}{B^2_\ell} \right) \sqrt{\frac{2}{\nu_\ell}} .$$

The debiasing divides each leg by `data_model` $= B_1 B_2 \, w^{\rm pix} F_\ell$,
so the noise enters the reported error bars as $N_\ell / B^2_\ell$ on its own —
it grows at high $\ell$, as it should.

**Form 1 — one number per frequency** (temperature white-noise level,
$\mu K \cdot \mathrm{arcmin}$):

```yaml
nl: {'090GHz': 5.4, '150GHz': 4.4, '220GHz': 16.2}
```

$N^{TT} = (\sigma \pi / 10800)^2$ and $N^{EE} = N^{BB} = 2 N^{TT}$ (the usual
$\sqrt 2$ in amplitude for polarisation).

**Form 2 — two numbers per frequency**, `[sigma_T, sigma_P]`, both
$\mu K \cdot \mathrm{arcmin}$:

```yaml
nl: {'090GHz': [5.4, 7.6], '150GHz': [4.4, 6.2]}
```

$N^{TT} = (\sigma_T \pi / 10800)^2$ and $N^{EE} = N^{BB} = (\sigma_P \pi / 10800)^2$.
A list or tuple of exactly two numbers; any other length raises. One dict may
mix forms 1 and 2 across frequencies.

**Form 3 — a tabulated noise power spectrum file**:

```yaml
nl: 'path/nl_{}.txt'
```

`{}` is filled with the frequency-pair key. Columns follow the layout rules of
[`getting_started.md`](getting_started.md) (the 4-column layout is TT, EE, BB,
TE); units follow `nl_units` / `spectrum_units`.

In forms 1 and 2 every cross-Stokes spectrum (TE/ET, TB/BT, EB/BE) is zero —
map noise is uncorrelated between Stokes parameters. Dict keys are **single
frequencies**, members of `frequencies`, not frequency pairs:

- an auto pair `f+f` takes that frequency's level;
- a cross pair `f1+f2`, `f1 != f2`, is zero, with no warning — map noise is
  uncorrelated between bands, so this is the normal case, not a missing input;
- an unknown key raises, naming it and the expected frequency names;
- a key spelled as a frequency *pair* (e.g. `090GHz090GHz`, the spelling
  accepted before this change) raises with a message saying to use the single
  frequency name;
- a frequency in `frequencies` with no entry warns and is treated as noiseless.

`nl_units: Dl` with a dict raises: levels are a flat $C_\ell$ by construction.

> **Removed:** `nl_is_biased`. A parameter file that still carries it is
> refused. `nl` is now always the map-level noise power spectrum, and beam
> deconvolution is applied by the debiasing.

### `spectrum_units` and per-input overrides

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `spectrum_units` | `Cl` \| `Dl` | `Cl` | Convention of the tabulated inputs. `Dl` reads columns as $D_\ell = \ell(\ell+1) C_\ell / 2\pi$ and converts to $C_\ell$ at read time (zeroing $\ell = 0, 1$); nothing downstream ever sees a $D_\ell$. |
| `cmb_spectrum_units`, `foregrounds_units`, `nl_units` | `Cl` \| `Dl` | `None` (follow `spectrum_units`) | Per-input override, for e.g. a CAMB $D_\ell$ spectrum alongside a $C_\ell$ noise curve. `noise_units` is accepted as an alias of `nl_units`. |

Applies only to `cmb_spectrum`, `foregrounds` and `nl` — the three inputs
whose columns are power spectra in $\mu K^2$. `beams`, `pixwin`, `fl`, `hl`,
`post_process_correction` and `inpainting_rescaling` are dimensionless and
never scaled. Asking for `Dl` on an input that is not a tabulated file (a
constant `cmb_spectrum`, or `nl` given as levels rather than a file) is an
error, not a silent no-op.

Unrelated to the `Dl` key below, which scales the *output* covariance.

## Post-processing and output shape

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `polspice_postprocess` | bool | `true` | Apply the PolSpice transform (Eq. 55, $G \Sigma G^T$) before scaling, debiasing and binning. `false` returns the binned pseudo $C_\ell$ covariance. A run with `EE` and a B observable (`BB`, `TB` or `EB`) but not `BB` itself is rejected when this is `true` — decoupling `EE` needs the pseudo `BB` block. See [`polspice.md`](polspice.md). |
| `apodizetype` | 0 or 1 | `1` | PolSpice apodisation: 0 Gaussian, 1 cosine (preferred). |
| `apodizesigma` | float, degrees | `30.0` | Apodisation scale. |
| `thetamax` | float, degrees | `30.0` | Maximum angular separation retained in the correlation function; should be the maximum angular size of the weighted mask. |
| `Dl` | bool | `false` | Return $D_\ell = \ell(\ell+1) C_\ell / 2\pi$ covariance instead of $C_\ell$. |
| `sum_assymetric_stokes` | bool | `false` | Fold TE and ET together (`sum_asymmetric_stokes`, single-s spelling, is accepted too; giving both with different values is an error). |
| `add_tf_uncertainty` | bool | `false` | Propagate transfer-function uncertainty: each spectrum leg is multiplied by $1 + \sqrt{(1 - f_\ell)/3999} \geq 1$. Needs `fl`. |

## Workflow switches

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `save_raw_blocks` | bool | `false` | Cache each raw (unbinned) pseudo $C_\ell$ block in the run's output directory as `cov_<stokes>_<freq>.npy`, with a manifest; reused only on an exact match. See [`getting_started.md`](getting_started.md), "Caching and reusing results". |
| `compute_noise_only` | bool | `false` | Accepted but raises `NotImplementedError` if set to `true`. |
| `compute_signal_only` | bool | `false` | Accepted but raises `NotImplementedError` if set to `true`. |

`--save-windows` is a `cmbcov-cov` command-line flag, not a
parameter-file key: see [`getting_started.md`](getting_started.md),
"Bandpower window functions".

## Alias spellings

`sum_asymmetric_stokes` -> `sum_assymetric_stokes`, `noise` -> `nl`,
`noise_units` -> `nl_units`. Each accepted alias is folded onto its canonical
key before validation; supplying both spellings with different values is an
error.
