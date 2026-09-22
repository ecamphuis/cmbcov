# PolSpice post-processing

`polspice_postprocess: true` (the default) turns the pseudo $C_\ell$
covariance the strategy computes into the covariance of the **decoupled**
PolSpice estimator — the paper's Eq. (55) — before $D_\ell$ scaling,
debiasing and binning. `false` skips this and returns the binned
pseudo $C_\ell$ covariance directly.

## The decoupled estimator

PolSpice's real-space correlation-function estimator (`decouple=YES`) turns
a pseudo spectrum $\tilde C_\ell$ into a decoupled one $\hat C_\ell$ through
a kernel $G$, indexed $[\ell, \ell']$ and acting as
$\hat C_\ell = \sum_{\ell'} G_{\ell\ell'} \tilde C_{\ell'}$. At the level of
the covariance this is a congruence transform,

```math
\hat\Sigma = G_{\mathrm{left}}\, \tilde\Sigma\, G_{\mathrm{right}}^{T},
```

`CovariancePostProcessor.pseudo_to_spice` for a T/E-only block (`G_left`,
`G_right` from the pair of spectra the block correlates). The apodisation
settings `apodizetype`, `apodizesigma` and `thetamax` (see
[`parameters.md`](parameters.md)) control the real-space window PolSpice's
kernel is built from.

## Kernels per spectrum

Each two-letter spectrum takes one of a small set of kernels, built from the
underlying MASTER/PolSpice quadrature (`kernels/coupling.py`,
`kernels/polspice.py`):

| Spectrum | Kernel |
| --- | --- |
| TT | $^{0}G$ |
| TE, ET, TB, BT | $^{\times}G$ |
| EE, BB (each on itself) | $^{+}G$ |
| EE ↔ BB (mixing) | $^{-}G$ |
| EB, BE | $^{+}G - {}^{-}G$ — never decoupled |

## E/B mixing

At level 1 (T/E-only, no B observable — see
[`polarisation.md`](polarisation.md)), `pseudo_to_spice` stays diagonal in
the spectrum: an EE output uses only the pseudo EE block and $^{+}G$. This
is the pre-existing, bit-identical behaviour, and it omits a small term:
PolSpice decoupling genuinely mixes EE and BB,

```math
\hat C^{EE} = {}^{+}G\,\tilde C^{EE} + {}^{-}G\,\tilde C^{BB}, \qquad
\hat C^{BB} = {}^{-}G\,\tilde C^{EE} + {}^{+}G\,\tilde C^{BB},
```

and at the covariance level every EE or BB output block can receive a
contribution from up to four pseudo blocks (EE×EE, EE×BB, BB×EE, BB×BB).
This full mixing is only applied once a B observable is present
(`CovariancePostProcessor.pseudo_to_spice_bmode`), because it needs the
pseudo BB block:

```math
\mathrm{Cov}(\hat C^X, \hat C^Y) = \sum_{a \in \mathrm{src}(X)}
\sum_{b \in \mathrm{src}(Y)} G_{X\leftarrow a}\,
\mathrm{Cov}(\tilde C^a, \tilde C^b)\, G_{Y\leftarrow b}^{T},
```

with $\mathrm{src}(EE) = \mathrm{src}(BB) = \{EE, BB\}$ and
$\mathrm{src}(X) = \{X\}$ for every other spectrum. This is why the parameter validator
rejects `observables` that include `EE` and a B observable (`BB`, `TB` or
`EB`) without `BB` itself when `polspice_postprocess: true`: decoupling `EE`
needs the pseudo `BB` block, and there is nowhere else to get it from. The
`observables: [BB]`-only NKA/INKA escape hatch (see
[`polarisation.md`](polarisation.md)) is the one exception: it applies the
diagonal, level-1-style $^{+}G$ transform instead, since it never computes a
pseudo EE block.

See [`theory/bmode_kernels.md`](theory/bmode_kernels.md) for the derivation
behind this table.
