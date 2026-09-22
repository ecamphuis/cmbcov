# ACC coupling-kernel precomputation

ACC needs coupling kernels that capture the effect of masking on the
covariance. They are assumed identical along each diagonal
$|\ell - \ell'| = \Delta$ (the paper's Eq. 33), so ACC computes them for
only one multipole of each diagonal, `centralell`, and reuses that kernel —
translated — for every entry of the diagonal. This is the expensive step of ACC, paid **once per
mask**; see [`getting_started.md`](getting_started.md) for how it fits into
the wider precompute-once, recompute-many workflow, and
[`approximations.md`](approximations.md) for how the choice of `centralell`
and `dmax` trades off against accuracy.

## Parameters

| Name | Meaning | Guidance |
| --- | --- | --- |
| `centralell` | Central multipole $\ell_\ast$ of the coupling computation; recorded in the manifest and checked against every run. | Higher is more accurate and more expensive. `<= 150` triggers an accuracy warning; a few hundred is typically needed for sub-percent accuracy on a few-percent-sky-fraction mask. A larger sky fraction (a less aggressively cut mask) can often use a lower value. |
| `dmax` | Number of diagonal bands `\|l - l'\| < dmax` ACC computes (zero outside the band). | Set by how fast the *true* mask-induced correlation between $\ell$ and $\ell'$ decays for your footprint. Size it for your own mask by computing one exact row (`cmbcov.exact.exact_covariance_row`) at a representative $\ell$ and reading off where the correlation with its neighbours falls below your target; the cost of ACC is one kernel pair per value of `dmax`, so raising it beyond where the true correlation is already small buys little. |
| `nside` | ACC working resolution; kernels are `S = 2 * nside` square. | `None` derives it from `centralell + dmax` (at least 128); higher is more accurate and slower. It also sets how far past `lmax` every run must supply spectra: `lmax_int = lmax + max(0, S - 1 - centralell)` (see [`getting_started.md`](getting_started.md), "The ACC internal band limit"). |
| `grid` | Quadrature backend: `gl` (default) or `healpix`. | `gl` is exact for a band-limited mask and the only grid that supports B-mode observables; `healpix` computes T, E and B from one joint transform but has no term-selection support. |
| `lw` | Mask band-limit, `grid: gl` only. | `None` means `3 * nside - 1`. Larger is more accurate (up to the mask's true band-limit) and more expensive. |

The secondary multipoles are not a free choice: for every offset `d` in
`range(dmax)`, ACC loads the kernel of `(centralell, centralell + d)` and
uses it, translated, for the whole diagonal $\Delta = d$.

## Precomputing the kernels

From a parameter file, the precompute and every later run share one file, so
the mask, `centralell`, `dmax` and the kernel directory cannot drift apart:

```yaml
covariance_approximation: acc
dmax: 20
centralell: 200

acc_precompute:     # optional: how the kernels are computed
  nside: 256        # optional; a power of two with grid: healpix
  grid: gl          # gl (default) | healpix
  lw: 512           # optional, gl only
  spectra: [TT]     # optional, default all; must cover what `observables` needs
  max_memory_gb: 6  # optional
```

```bash
cmbcov-precompute parameters.yml       # once per mask
cmbcov-cov parameters.yml   # every run
```

`cmbcov-precompute --dryrun parameters.yml` validates the file and prints the
plan without computing. From Python, `precompute_acc_kernels` is standalone
— it needs only the mask, no `Cov`, no `CovarianceConfig`:

```python
from cmbcov.approximations.acc import precompute_acc_kernels

precompute_acc_kernels(
    "your_mask.fits", "./coupling_kernels/",
    centralell=200, dmax=20, mask_path="./masks/",
    nside=256, grid="gl", lw=512,
    spectra=("TT",),   # restrict to what you need; default computes all 25
)
```

### Computing only the spectra you need

`spectra` restricts which kernel channels are computed. A TT-only study
needs just `spectra=("TT",)` — on `grid="gl"`, this also skips the spin-2
(E, B) mode-coupling integrals, the dominant cost of a `gl` precompute. Any
other subset that includes E or B still needs the spin-2 integrals. An empty
or unknown `spectra` raises.

## Memory budget

`max_memory_gb` (default 2) is the peak-memory budget of the coupling
contraction, and on both grids also decides whether the central integrals
are kept in memory or re-synthesised per block. Raising it is the single
biggest lever on wall time at high `centralell`, at the cost of more RAM;
lowering it trades time for memory. If memory still forces a split, the
secondary multipoles can be computed in consecutive chunks (passing
`ellprange` instead of `dmax`, sharing one loaded mask) — prefer few, large
chunks, since the central integrals are recomputed once per chunk.

## The cache and its validation

Kernels are saved as `.npy` files under `{save_dir}/covariance_coupling/`,
named `{stokes1}x{stokes2}_{ell}x{ellp}.npy`. A manifest,
`manifest_{ell}x{ellp}.json`, is written next to them, recording `spectra`,
`grid`, `lw`, `nside`, `centralell`, the mask's digest and the package
revision. `Cov` (and `cmbcov-precompute`) load them automatically as long as
they point at the same directory, the same mask, and a `centralell`/`dmax`
no larger than what was precomputed — checked against the manifest on every
load. Changing the mask or `centralell` without rerunning `cmbcov-precompute`
stops the next run with a mismatch error rather than silently using stale
kernels. A cache built with a restricted `spectra` set is refused, naming
what it lacks, for a run that needs a wider one — there is no partial-cache
fallback; recompute with the wider set (or omit `spectra` for the default
five channels, or the full set a B-mode run needs).

A pre-`.npy` (`.txt`-only) cache is refused too, naming the
`convert-acc-cache` script: `convert-acc-cache /path/to/save_dir` converts
it in place, checking every kernel bit-exactly against its `.txt` source
first (`--remove-text` deletes the originals once verified).

## Term selection

`precompute_acc_kernels(..., term_selection=<tol>)` (`grid="gl"` only)
restricts the kernel sum to the $(m, m', M)$ terms the mask's own harmonic
content predicts matter, at a chosen relative-error tolerance, cutting the
number of terms summed to a small fraction of the full set. The default,
`term_selection=None`, is bit-identical to leaving the keyword out. See
[`theory/term_selection.md`](theory/term_selection.md) for the selection
rule and its accuracy/cost trade-off.

## Cost guidance

- The precompute cost grows with `centralell`, `dmax` and `nside`
  (kernels are `2 * nside` square); the recompute cost that follows does
  not depend on any of these once the cache exists.
- A TT-only `spectra` selection is markedly cheaper than the default on
  `grid="gl"`, since it skips the spin-2 integrals.
- `term_selection` reduces the precompute cost further, at a chosen
  accuracy trade-off, on `grid="gl"`.
- Batch the whole `dmax` range in one call when memory allows: the central
  integrals are computed once per call and reused across every secondary
  multipole, so splitting into chunks recomputes them once per chunk.

## Troubleshooting

- **"Covariance coupling not found"**: kernels are missing for some
  `(centralell, centralell + d)` pair, `centralell`/`dmax` do not match what
  was precomputed, or `save_dir`/`acc_kernel_dir` does not point at the
  directory holding `covariance_coupling/`.
- **"cache mismatch in 'mask_digest'" or `'centralell'`**: the cache was
  built for a different mask or `centralell`; rerun `cmbcov-precompute`.
- **A spectrum-length error naming `lmax_int`**: the spectra must reach
  `lmax_int`, not just `lmax`; see [`getting_started.md`](getting_started.md),
  "The ACC internal band limit".
- **High memory usage**: reduce `nside`, lower `max_memory_gb` (slower), or
  split `dmax` into chunks.
- **Long computation times**: normal for the first computation; kernels are
  reused afterwards. If only TT is needed, `spectra=("TT",)` is the single
  biggest lever on `grid="gl"`.
