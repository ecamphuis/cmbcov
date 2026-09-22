# Approximations

The package computes the exact pseudo $C_\ell$ covariance of Camphuis et al.
(2022) Sect. 3, and three approximations to its reduced coupling kernel
$\bar\Theta$: NKA, INKA and ACC. This page is about choosing between them;
the derivations are in [`theory/acc.md`](theory/acc.md) and
[`theory/exact_covariance.md`](theory/exact_covariance.md).

## NKA — Narrow Kernel Approximation

**Assumes** the coupling kernel is narrow enough to replace by delta
functions (Eq. 27 of the paper), giving $\Sigma = 2\, C_\ell C_{\ell'}\, \Xi[W^2]$
(Eq. 26) directly from the mask's power spectrum $W^2$. Exact on the full
sky; accurate wherever the spectrum varies slowly compared with the width of
the true coupling kernel.

**Cost.** No precompute; the cheapest method, dominated by the mask's own
coupling kernel (needed by every method).

**Accuracy.** Degrades where the true coupling kernel is genuinely broad
compared with the multipole scale — typically the lowest multipoles of a
partial-sky mask, where the mode-coupling width is comparable to $\ell$
itself. On a few-percent-sky-fraction footprint the deviation from the exact
covariance is largest below $\ell \sim 20$ and falls with increasing $\ell$.

**Use it** for a quick, cheap estimate, on a large or full-sky-like mask, or
when only high $\ell$ bins matter.

**Spectra.** TT, EE, TE. BB only through the `observables: [BB]`-only escape
hatch (see [`polarisation.md`](polarisation.md)); no TB/EB.

## INKA — Improved NKA

**Assumes** the same delta-function structure as NKA, but with the spectrum
first smoothed by the renormalised MASTER kernel $\bar M$ (each row summing
to one), $\Sigma = 2\, \bar C_\ell \bar C_{\ell'}\, \Xi[W^2]$ (Eq. 31). This
captures some of the coupling NKA discards, at the same cost.

**Cost.** Same as NKA: no precompute, one extra matrix-vector product
($\bar M C$) per spectrum.

**Accuracy.** Consistently closer to the exact covariance than NKA at every
multipole, most noticeably where NKA is worst (low $\ell$ on a partial-sky
mask); it still degrades at the very lowest multipoles, where the true
coupling kernel is broadest.

**Use it** in place of NKA whenever the extra accuracy is worth the (small)
extra cost — essentially always, since INKA is never slower than NKA.

**Spectra.** Same restriction as NKA: EE and BB are degenerate by
construction (the spin-weight rule that builds $\Xi$ cannot tell them
apart), so only the `[BB]`-only escape hatch gives a BB covariance, and no
TB/EB.

## ACC — Analytical Covariance Coupling

**Assumes** the coupling kernel is translation-invariant along each
diagonal: the kernel computed once at a reference multipole $\ell_\ast$
(`centralell`) is reused, translated, for every entry $(\ell, \ell')$ with
$|\ell - \ell'|$ up to `dmax` (Eq. 33). This needs a one-off precompute of
the kernel at $\ell_\ast$ and its neighbours, but nothing further once that
is done.

**Cost.** The precompute is the expensive step, scaling with `centralell`,
`dmax` and the working `nside` (see
[`acc_precomputation.md`](acc_precomputation.md)); it is paid once per mask.
The recompute — everything `compute_covariance_matrix` does afterwards, for
new spectra, noise, beams or binning — reuses the cached kernels and is
comparable in cost to NKA/INKA plus the PolSpice transform.

**Accuracy.** Two effects, both reported by the per-run error budget
(`error_budget.txt`, written for any ACC run with a polarised leg):

- the translation itself gets less accurate the further $\ell$ is from
  `centralell` — this error is **not bounded** by the budget, only its
  extent (the largest $|\ell - \ell_\ast|$ at the band edges) is reported;
- a polarised block additionally carries a small, **bounded** high bias from
  E→B leakage in the kernel normalisation, proportional to the number of
  polarised legs in the block.

Both shrink as `centralell` is raised; `centralell <= 150` triggers an
accuracy warning. Chosen high enough (a few hundred, for a few-percent sky
fraction), ACC is the most accurate approximation implemented, sub-percent
close to `centralell` and growing worse away from it.

**Use it** for a production covariance at large `lmax` that will be
recomputed many times, and for any BB, TB or EB block (with the one
exception below).

**Spectra.** TT, EE, TE, BB, TB, EB, and parity-mixed blocks — the only
method that supports a B-mode observable other than the BB-only escape
hatch. See [`polarisation.md`](polarisation.md) and
[`theory/bmode_kernels.md`](theory/bmode_kernels.md) for the per-spectrum
kernel channels and accuracy off `centralell`.

## Exact

`cmbcov.exact.exact_covariance` / `exact_covariance_row`
(and their `_pol` variants) evaluate the paper's Sect. 3 directly, row by
row, with no approximation. They are standalone functions, not a
`covariance_approximation` choice — there is no `Cov` wrapper for them.

**Cost.** Expensive at high `lmax`: each row is its own spherical-harmonic
transform. Not a production method.

**Accuracy.** The reference: every approximation above is measured against
it.

**Spectra.** TT on both the HEALPix and Gauss-Legendre (`grid="gl"`) grids;
all of TT, EE, BB, TE, TB, EB on the GL grid only.

## Which spectra each method supports

| | TT, EE, TE | BB (alone) | BB (with T/E) | TB, EB |
| --- | --- | --- | --- | --- |
| NKA | yes | yes, leakage-neglected escape hatch, warns | no | no |
| INKA | yes | yes, same escape hatch | no | no |
| ACC | yes | yes | yes | yes |
| Exact (GL grid) | yes | yes | yes | yes |

The `observables: [BB]`-only NKA/INKA mode neglects E→B leakage into the
pseudo-BB mean throughout, and always emits a `UserWarning` giving the
multipole above which the missed variance inflation is estimated to stay
under 5%. See [`polarisation.md`](polarisation.md) for when that is
acceptable and its known limitations.
