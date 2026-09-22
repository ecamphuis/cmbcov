# T/E and B-mode spectra

By default the package computes TT, EE and TE. BB, TB and EB are available
on request, through `observables` (see [`parameters.md`](parameters.md)).
This page covers what changes when they are asked for.

## The four observable levels

| Level | How to ask for it | Spectra used | Blocks returned |
| --- | --- | --- | --- |
| 1 (default) | `stokes: [T, E]`, or `observables: [TT, EE, TE]` | TT, EE, TE | TT, EE, TE — bit-identical to a run that never touches B |
| 2 | `observables: [..., BB]` | + $C^{BB}$ | + BB |
| 3 | `observables: [..., TB]` and/or `[..., EB]` | + $C^{TB}$, $C^{EB}$ | + TB, EB |
| 4 | level 3 plus `parity_mixed_blocks: true` | as level 3 | + the parity-mixed blocks (TT×TB, EE×EB, TE×TB, ...) |

Level 1 is untouched by everything below it: $C^{BB}$ is simply never read,
and the code path is the same as before B-mode support existed. From level 2
up, every block — the T/E ones included — is assembled per Wick term with a
normalisation that also picks up their (small) $C^{BB}$ contributions (see
[`theory/bmode_kernels.md`](theory/bmode_kernels.md)), so a level-2+ run's
TT/EE/TE blocks can differ very slightly from a level-1 run of the same
spectra.

Levels 2 to 4 need `covariance_approximation: acc` and `acc_precompute.grid:
gl` (the default grid) — see the escape hatch below for the one exception.
$C^{BB}$ is read whenever `observables` contains a B; $C^{TB}$ and $C^{EB}$
only when `TB`/`EB` are themselves requested.

## Kernel channels vs. spectra

The ACC coupling kernels are computed in three fields, renamed T, D and L
(rather than the Stokes T, E, B) to keep the kernel *channel* names distinct
from the *spectrum* names they combine to build: a B-mode spectrum has no
single coupling-kernel channel of its own, only combinations of the T/D/L
ones. See [`theory/bmode_kernels.md`](theory/bmode_kernels.md) for the
correlator table and the per-Wick-term normalisation this builds on.

## TB and EB from the spectra file

$C^{TB}$ and $C^{EB}$ are read from the spectrum file's TB/EB columns when
present; the usual case (an EB/TB null test) is that they are absent or
zero, and the code treats a missing column as zero. The ACC kernel-pair set
needed differs slightly depending on whether they are genuinely zero or not,
so a run whose spectra turn out non-zero against a cache built assuming zero
is refused with the usual "recompute the kernels" error rather than
silently substituting the wrong pairs.

## Parity-mixed blocks

`parity_mixed_blocks: true` (level 4) adds the blocks between a parity-even
spectrum (TT, EE, TE, BB) and a parity-odd one (TB, EB) — for instance
TT×TB. They are zero for a mask with a mirror-symmetry plane and small
otherwise, so they default to off; enabling them needs `TB` or `EB` in
`observables` plus at least one parity-even observable, checked at
validation.

## The BB-only NKA/INKA escape hatch

The one exception to "a B observable needs ACC" is `observables: [BB]`
alone (cross-frequency BB is fine): NKA and INKA also accept it, with no ACC
precompute, as a fast route to $\mathrm{Cov}(BB, BB)$ at high $\ell$. It
computes $\mathrm{Cov}(BB, BB) = 2\, (C^{BB})^2\, \Xi^{EE\to EE}$ — the same
formula as an EE covariance, since NKA/INKA cannot distinguish EE and BB by
construction — which neglects the leakage of $C^{EE}$ power into the
pseudo-BB mean entirely.

Because that leakage dominates at low $\ell$, every such run emits a
`UserWarning` (and, from a parameter file, writes `bb_leakage_warning.txt`
beside the covariance) giving `ell_safe`: the smallest multipole above which
the estimated missed variance inflation stays under 5%. Bins below
`ell_safe` are underestimated, potentially by a large factor — treat this
mode as valid only for the high $\ell$ bins above the reported `ell_safe`,
and use full ACC (level 2) if you need the low $\ell$ BB variance.

## Known limitations

Away from `centralell`, the leakage-dominated blocks — BB×BB, TB×TB and
EB×EB — are the least accurate part of ACC's B-mode support: on a realistic
partial-sky mask they can be off by roughly 10-50% far from `centralell`,
because the auto-leakage coefficient the per-Wick-term normalisation freezes
at its `centralell` value is not actually flat in $\ell$. The T/E blocks
(TT, EE, TE) and the blocks with exactly one leaked leg stay close to
sub-percent across the same range. Two practical consequences:

- **Validate a B-mode run's leakage-dominated blocks against the exact
  covariance** (`cmbcov.exact`, GL grid) on your own mask
  before trusting them far from `centralell`.
- **Prefer bins close to `centralell`** for BB, TB and EB if you cannot
  validate, or raise `centralell` so that the band of interest sits nearer
  it.

See [`approximations.md`](approximations.md) for how this fits into ACC's
overall accuracy, and [`theory/bmode_kernels.md`](theory/bmode_kernels.md)
for the underlying derivation.
