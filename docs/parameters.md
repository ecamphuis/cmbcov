# Parameter-file reference

The authoritative source of every key is
[`cmbcov/config/validation.py`](../cmbcov/config/validation.py);
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
| `max_memory_gb` | float | 2 | Memory budget of the precompute (held integrals and the contraction), on top of a fixed baseline for the process and the mask; must be `> 0`. See [`acc_precomputation.md`](acc_precomputation.md), "Memory budget". |
| `scratch_dir` | str | the kernel directory | Where the temporary on-disk store of the central coefficients goes when they do not fit in half of `max_memory_gb` (a file of `(2 centralell + 1) * 3 * 2 nside * (4 nside - 1) * 16` bytes, 11.7 GiB at `centralell` 250, `nside` 256); deleted when the precompute ends. |

`grid: gl` is required (not merely default) with any B-mode observable: the
per-Wick-term normalisation reads raw kernel sums, which only the GL grid
provides.

### `map_filter` block

Optional. Corrects the **T/E** covariance blocks (TT, TE, EE and their
cross-blocks) for a map made by a scan-direction Fourier high-pass filter
(filter-and-bin map-making); see
[`theory/filter_and_bin.md`](theory/filter_and_bin.md). Each raw T/E
pseudo-spectrum block of spectra $s_1$, $s_2$ is multiplied elementwise by

```math
\frac{\rho^{ch}(\ell_1, \ell_2)}{F^{L, s_1}_{\ell_1}\thinspace F^{R, s_2}_{\ell_2}}
```

for every method (NKA, INKA, ACC), where $ch$ is the block's sum-rule
channel (`00` for TT x TT, `20` for TT x EE, TT x TE and TE x TE, `EE` for
EE x EE and EE x TE, transposes alike) and $F^{L, s_1}$, $F^{R, s_2}$ are
the `fl` of the left and right spectra for their frequency pairs (ET uses
the TE `fl`, which should be a smooth positive reference such as
$\sqrt{F^{TT} F^{EE}}$). Only the channels the run's blocks need are
probed. Blocks with a B leg are left **uncorrected** and the run emits one
`UserWarning` naming them. Without the block the run is unchanged. The map's
polar axis (the mask's coordinate system) must be the scan axis:
iso-latitude rings are the scan lines. Without `fl` the factor is $\rho$
alone, with a warning.

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `type` | `fourier_highpass` | required | Kind of filter; only `fourier_highpass` exists. |
| `lx` | float | required | Cut $m_c = l_x \sin\theta$ of the azimuthal Fourier index. Must be `> 0`. |
| `shape` | `sharp` \| `exp` | `sharp` | Filter profile around the cut. |
| `power` | float | 6 | Steepness of the `exp` profile; must be `> 0`, unused by `sharp`. |
| `nprobe` | int | 32 | Random probes per node of the stochastic sum-rule ratio, at the nodes outside the near-cut range (below $0.95\thinspace l_x$ and from $2\thinspace l_x$ up); `>= 1`. The error falls as the inverse square root. |
| `seed` | int | 0 | Seed of the probes. |
| `node_min` | int | 32 | Lowest multipole of the geometric nodes at which $\rho$ is probed (interpolated in between); `>= 2`. |
| `node_ratio` | float | 1.25 | Geometric spacing of the nodes; `> 1`. |
| `near_cut_step` | number or null | null | Spacing of the extra nodes between $0.95\thinspace l_x$ and $2\thinspace l_x$ (rounded, at least 1); a geometric node closer than half a step to one of them is dropped. `null`: $l_x / 15$ (20 at $l_x = 300$). `0`: no extra nodes at all (the geometric list of v0.6.0), neither near the cut nor through the rolloff of a `lowpass`; with a `lowpass` whose rolloff starts below `lmax` ($0.6\thinspace l_x^{\mathrm{low}} < \ell_{\max}$) the run logs a warning, because that rolloff is then interpolated from the geometric nodes alone (up to about 4% off for `exp`, more for `sharp`, at toy scale). |
| `nprobe_near_cut` | int or null | 128 | Probes of the nodes from $0.95\thinspace l_x$ to below $2\thinspace l_x$, `>= 1`. `null`: the complete basis of each node's degree ($2\ell' + 1$ probes), which makes the ratio exact there. It is expensive at survey scale: with $l_x = 300$, mask band-limit 875 and 8 threads, about 2.6 hours for the three white channels and 1.3 hours for spectrum-aware TT; `128` took about 24 and 13 minutes and changed the binned variances by at most 0.4% in a survey-like test. |
| `lowpass` | mapping or null | null | A scan-direction low-pass multiplying the high-pass, $h = h_{\mathrm{high}}\thinspace h_{\mathrm{low}}$, for a map whose time stream is band-limited (SPT-3G: cut 300, low-pass 6144). Absent or `null`: no low-pass, and the results and cache keys are exactly those of a file without the key. Keys below; see [`theory/filter_and_bin.md`](theory/filter_and_bin.md), "The low-pass". `fl` must include the low-pass, as it must include the high-pass; nothing checks it. |
| `lowpass.lx` | float | required | Cut $m_c = l_x \sin\theta$ of the low-pass, the same convention as the high-pass `lx`. Finite, `> 0`, and above the high-pass `lx`. |
| `lowpass.shape` | `exp` \| `sharp` | `exp` | `exp`: $h_{\mathrm{low}} = \exp[-(\lvert M \rvert / (l_x \sin\theta))^p]$; `sharp`: 1 where $\lvert M \rvert \le l_x \sin\theta$, else 0. The ratio of the `sharp` shape has off-diagonal structure at its kink that the (mean, distance) interpolation cannot carry: blocks within a few multipoles of `lx` are about 3-4% off even with the fine nodes, and the run logs a warning when `lx <= lmax`. The `exp` shape has no such limit. |
| `lowpass.power` | float | 6 | Exponent $p$ of the `exp` shape; finite, `> 0`; unused by `sharp`. |
| `dmax` | int or null | null | Largest $\lvert \ell_1 - \ell_2 \rvert$ probed, `>= 0`. `null`: the run's ACC `dmax - 1` with ACC, 64 otherwise, capped at `lmax - 1`. |
| `lw` | int or null | null | Mask band-limit of the probes. `null`: `min(3 * nside - 1, 2 * lmax)`. |
| `spectrum_aware` | bool | false | Probe every TT x TT block with the run's own spectra instead of the white `00` channel: the block is multiplied by its own ratio of filtered to unfiltered covariance, with no `fl` division. Needed for a sharp cut, which scatters low $\ell$ TT power to high $\ell$ (TT variance 5.5% low at $\ell$ = 2000-3500 otherwise). The other T/E blocks are unchanged. The ratio depends on the spectra: changing them probes again. |

Near the cut the nodes also carry the probes to a higher internal
band-limit (at least $3.5\thinspace l_x$ for the nodes from
$0.95\thinspace l_x$ to below $1.3\thinspace l_x$; not a parameter). The
three near-cut settings are explained in
[`theory/filter_and_bin.md`](theory/filter_and_bin.md), Sect. 2, "Near the
cut". With
`map_filter` the run logs a warning naming the bins whose lowest `fl`
(any T/E spectrum, any frequency pair) is below 0.3, and a stronger one for
those below 0.1: cut those.

The probed ratios are cached in the kernel directory (`cov_path`), keyed on
the mask, the channels, the nodes with their probe counts and band-limit
floors, and every key above (the `lowpass` entry only when there is one, so
the caches of files without it stay valid), so a rerun with the same filter
and blocks does not probe again; with `spectrum_aware` the TT x TT ratios
are keyed on the blocks' Wick terms and a digest of the spectra as well, so
a change of `cmb_spectrum`, `foregrounds`, `nl`, `beams`, `pixwin` or `fl`
probes them again (the white channels are not affected). They are also
saved as `map_filter_ratio.npz` (`nodes`, `nprobe` and `lmax_int_floor`
per node, `channels` and one `g_<channel>` per channel; with `spectrum_aware` also
`spectrum_aware_blocks`, the TT x TT block names, and one `g_block_<i>`
per block in that order) beside the covariance. A saved parameter file that differs in `map_filter` (or has it
where the new one has none) makes `--overwrite` refuse, like any other
change that alters the numbers.


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

The debiasing divides each leg by `data_model` $= B_1 B_2 \thinspace w^{\mathrm{pix}} F_\ell$,
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
