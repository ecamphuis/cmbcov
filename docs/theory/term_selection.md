# A-priori term selection

Most terms of the sums that build an [ACC](acc.md) coupling kernel, or a row
of the [exact covariance](exact_covariance.md), are negligible, and which
ones can be told in advance from the mask's own harmonic coefficients once
the mask is rotated so that it sits at the north pole. With
`term_selection=<tolerance>` the precompute skips those terms, for a kernel
whose relative error is of the order of the tolerance and an order of
magnitude less work on a survey footprint. The default, `None`, computes
every term.

## 1. The sums, and why most terms vanish

A kernel at $(\ell, \ell')$ is a sum over the orders $m$, $m'$ and $M$:

```math
\Theta(L_1, L_2) = \mathrm{Re}\sum_{mm'} X_{mm'}(L_1)\thinspace\overline{X_{mm'}(L_2)},
\qquad
X_{mm'}(L) = \sum_{M} I_{\ell m, LM}\thinspace\overline{I_{\ell' m', LM}},
```

with the mode-coupling integrals $I_{\ell m, LM} = \int W\thinspace Y_{\ell m}\thinspace Y_{LM}^{\ast}\thinspace d\Omega$
(and their spin-2 analogues for the D and L fields). Its cost is
$(2\ell+1)(2\ell'+1)$ pairs $(m, m')$ times up to $2L_{\max}+1$ orders $M$
per $L$. Expanding the mask as $W = \sum_{\ell_3 m_3} w_{\ell_3 m_3} Y_{\ell_3 m_3}$
turns each integral into a sum of Gaunt coefficients,

```math
I_{\ell m, LM} = (-1)^{M}\sum_{\ell_3} w_{\ell_3, M-m}\thinspace G(\ell_3, \ell, L;\thinspace M-m, m, -M),
\qquad
G(l_1, l_2, l_3; m_1, m_2, m_3) = \int Y_{l_1 m_1} Y_{l_2 m_2} Y_{l_3 m_3}\thinspace d\Omega ,
```

and the azimuthal selection rule $m_1 + m_2 + m_3 = 0$ means that
$I_{\ell m, LM}$ sees only the mask coefficients of order $m_3 = M - m$.
Everything below follows from that.

## 2. The pole frame

The kernel is invariant under rotations of the mask: it sums over all
$(m, m')$, and a rotation acts on each degree by a unitary Wigner-D matrix
in $m$. The same holds for an exact covariance row. Which terms carry the
sum is not invariant. In an arbitrary frame the mask's azimuthal spectrum

```math
P(m_3) \propto \sum_{\ell} \vert w_{\ell m_3}\vert^2, \qquad \sum_{m_3} P(m_3) = 1,
```

is broad and nothing can be skipped. Rotated so that its centroid sits at
$+z$, a compact footprint has $P(m_3)$ concentrated at small $\vert m_3\vert$,
and three selection rules become sharp. `pole_rotation` finds the rotation
from the mask map: the centroid when it is well defined, or, for a mask
symmetric under $\hat n \to -\hat n$ such as a Galactic cut or an equatorial
belt, the symmetry axis read off the second moments. The kernels are the
same in either frame to $10^{-14}$.

## 3. The orders that carry mask power

In the pole frame, for each kernel pair $(\ell, \ell')$ (`select_terms`):

1. **Orders $m$.** By Parseval, the total coupling of a mode is its overlap
   with the squared mask,

   ```math
   p_m = \int W^2\thinspace\vert Y_{\ell m}\vert^2\thinspace d\Omega = \sum_{LM}\vert I_{\ell m, LM}\vert^2
   ```

   (`mode_power`). $Y_{\ell m}$ is small wherever
   $\sin\theta < \vert m\vert/\ell$, so a mask within an angle $r$ of the
   pole overlaps only orders up to about $\vert m\vert = \ell\sin r$. Keep
   $m$ with $p_m \ge \epsilon_m\max_m p_m$, and likewise $m'$.
2. **Pairs $(m, m')$.** $X_{mm'}$ couples $m$ and $m'$ through the mask
   orders $M - m$ and $M - m'$, so its size is set by $\vert m - m'\vert$
   through the autocorrelation $A(d) = \sum_{m_3} P(m_3)\thinspace P(m_3 + d)$
   (`azimuthal_spectrum`). Keep the pairs with
   $p_m\thinspace p_{m'}\thinspace A(m - m') \ge \epsilon_{\mathrm{pair}}\max$.
3. **Band in $M$.** Since $I_{\ell m, LM}$ involves only the mask order
   $M - m$, restrict $M$ to $\vert M - m\vert \le k$, with $k$ the smallest
   value for which $\sum_{\vert m_3\vert \le k} P(m_3) \ge 1 - \delta_{\mathrm{band}}$.
   The integrals of the band are computed directly by a banded Legendre
   transform (`banded_integrals_gl`), which agrees with the two-transform
   route to $2.5\times10^{-13}$ relative in the band.

For polarised kernels the spin-0 and spin-2 selections are made separately,
each with its own $p_m$, and united, so that no field loses a term another
field would have kept. For an azimuthally symmetric mask in the pole frame,
$P(m_3)$ is a delta function at zero and the pair rule keeps only
$m = m'$, a fraction $1/(2\ell+1)$ of the pairs.

## 4. What the tolerance means

`term_selection` is the target relative error of each kernel in the
Frobenius norm,

```math
\frac{\Vert\Theta_{\mathrm{selected}} - \Theta\Vert}{\Vert\Theta\Vert} \lesssim \mathrm{tol} .
```

It sets the three thresholds

```math
\epsilon_m = \mathrm{tol}/10, \qquad \epsilon_{\mathrm{pair}} = \mathrm{tol}/40, \qquad \delta_{\mathrm{band}} = \mathrm{tol}/10,
```

calibrated on a survey footprint (Sect. 5). It is a target, not a
guarantee, and it is met only for $\ell'$ close to $\ell$: Sect. 6 has the
measured errors and the separation at which they leave the tolerance. It is
also not the error of the covariance: dropping terms always lowers the
covariance, and at the default thresholds the binned TT variance comes out
about $0.6\%$ low (Sect. 6, "What it costs on the covariance"). The tolerance and the fractions of $m$, pairs and
$M$ kept are recorded in the cache manifest, and a run must ask for the same
value through `CovarianceConfig.acc_term_selection`; a cache built with a
different value, or with `None` against a tolerance, is refused. Term
selection is available on the Gauss-Legendre grid only.

For an exact row (`exact_covariance_row`, `exact_covariance_row_pol` and the
matrix versions, with `term_selection=`), each column $(\ell', m')$ is one
set of transforms that returns every $(\ell, m)$ at once, so the pair and
band rules have nothing to act on. Only the $m'$ rule applies: columns whose
overlap $p_{m'}$ is below $\mathrm{tol}/10$ of the largest are skipped,
united over spins for a polarised row. On a 20° cosine-apodised cap
at $n_{\mathrm{side}} = 16$, with $\ell_{\max} = 32$ and tolerance $10^{-3}$, this
skips 48 to 59% of the $m'$, and the largest relative error, over the
elements above $10^{-6}$ of the diagonal, is $1.5\times10^{-5}$ at
$\ell' = 10$ and $1.7\times10^{-6}$ at $\ell' = 32$.

## 5. Expected speed-up

Measured on an apodised survey footprint covering about 4% of the sky, in the
pole frame:

- At $\ell_\ast = 250$ with kernels at $n_{\mathrm{side}} = 256$ and tolerance
  $10^{-3}$, the rules keep 60% of the orders $m$, 5.1% of the pairs
  $(m, m')$, about 26 per $m$, and an $M$ band of 15% of the full range. At
  tolerance $10^{-4}$ they keep 61% of $m$, 9.8% of the pairs and 22% of the
  $M$ range.
- At $\ell_\ast = 128$, against the full precompute, the kernel error is
  $4.4\times10^{-4}$ and the error of the assembled ACC diagonal band
  $2.1\times10^{-4}$, from 9.3% of the pairs.
- End to end, one TT kernel pair at $(250, 250)$ with $n_{\mathrm{side}} = 256$
  and $L_w = 767$ on a 14-core laptop took 50.8 s in full and 4.6 s at
  tolerance $10^{-3}$, an 11-fold speed-up, with a Frobenius difference of
  $3.4\times10^{-4}$ on the kernel. The full polarised set was not timed.

The smallest pair fraction that reaches a $10^{-3}$ kernel error, found by
keeping the largest pairs after the fact, is 9.1% at $\ell_\ast = 64$ and
4.3% at $\ell_\ast = 128$ on the same footprint, about 11 pairs per $m$ in
both cases: the sparsity grows roughly as $\ell_\ast$. For exact rows the
gain is only the fraction of columns skipped; with about 60% of the orders
kept, as for the kernels above, the expected speed-up is about 1.6-fold. It
has not been timed. The speed-up applies to the
one-off precompute; the recompute of a covariance from cached kernels is
unchanged.

## 6. The pair rule, and how far it is validated

The three rules are not equally forgiving. The $m$ and band rules are
comfortably inside the tolerance everywhere measured; the pair rule carries
essentially the whole error, and it is the only one whose accuracy depends
on how far apart $\ell$ and $\ell'$ are.

Measured on the survey footprint at $n_{\mathrm{side}} = 64$,
$\ell = 64$, tolerance $10^{-3}$, decomposing the rules:

| $\ell' - \ell$ | all three | band only | pair only | $m$ only |
| --- | --- | --- | --- | --- |
| 0 | $1.3\times10^{-4}$ | $1.1\times10^{-5}$ | $1.3\times10^{-4}$ | $1.7\times10^{-7}$ |
| 19 | $6.6\times10^{-3}$ | $9.7\times10^{-5}$ | $6.7\times10^{-3}$ | $4.5\times10^{-6}$ |

The pair criterion $p_m\thinspace p_{m'} A(m - m')\ge\epsilon_{\mathrm{pair}}\max$
carries no dependence on $\ell' - \ell$, while the kernel itself shrinks as
the two multipoles separate, so the discarded terms become a larger fraction
of a smaller kernel.

### What the tolerance actually buys

At $\mathrm{tol} = 10^{-3}$, relative Frobenius error of the selected
kernel, at the current $\epsilon_{\mathrm{pair}} = \mathrm{tol}/40$ and at
the $\mathrm{tol}/4$ it replaced:

| mask | $n_{\mathrm{side}}$ | $\ell' - \ell$ | $\mathrm{tol}/4$ | $\mathrm{tol}/40$ |
| --- | --- | --- | --- | --- |
| survey | 32 | 0 | $9.6\times10^{-4}$ | $1.1\times10^{-4}$ |
| survey | 32 | 3 | $1.2\times10^{-3}$ ✗ | $1.4\times10^{-4}$ |
| survey | 64 | 0 | $7.3\times10^{-4}$ | $1.3\times10^{-4}$ |
| survey | 64 | 5 | $1.5\times10^{-3}$ ✗ | $2.5\times10^{-4}$ |
| survey | 64 | 10 | $7.3\times10^{-3}$ ✗ | $1.1\times10^{-3}$ ✗ |
| survey | 64 | 19 | $3.4\times10^{-2}$ ✗ | $6.6\times10^{-3}$ ✗ |
| survey | 128 | 0 | $4.3\times10^{-4}$ | $7.7\times10^{-5}$ |
| survey | 128 | 3 | $6.0\times10^{-4}$ | $9.3\times10^{-5}$ |
| two blobs | 64 | 0 | $9.6\times10^{-4}$ | $1.4\times10^{-4}$ |
| two blobs | 64 | 19 | $3.6\times10^{-2}$ ✗ | $2.7\times10^{-2}$ ✗ |
| two blobs | 128 | 0 | $1.0\times10^{-3}$ ✗ | $1.3\times10^{-4}$ |
| two blobs | 128 | 3 | $3.1\times10^{-3}$ ✗ | $2.4\times10^{-4}$ |

✗ marks a result outside the requested tolerance. Two things follow.

**The tolerance holds only for small $\ell' - \ell$.** Up to a separation of
about 5 it is met at $\mathrm{tol}/40$; by a separation of 10 it is missed,
and by 19 — which a `dmax = 20` cache uses — it is missed by a factor of
about 7 on the survey footprint and 27 on the two-blob mask. **If you
precompute with `term_selection` over a wide `dmax`, the far kernels are
much less accurate than the tolerance you asked for** — relative to
themselves. They are also small, and on the assembled covariance this does
not show (next subsection).

**Cost.** The tighter threshold keeps 1.5 to 1.8 times more pairs: on the
survey footprint 14.8% → 23.1% at $n_{\mathrm{side}} = 64$ and 9.1% → 16.4%
at 128. The $m$ and band selections are untouched by it.

### What it costs on the covariance

The kernel error above is not what a run sees. Measured end to end — full
and selected kernel caches on the survey footprint at $n_{\mathrm{side}} = 64$,
$\ell_\ast = 64$, `dmax = 20`, assembled into the ACC covariance with Planck
spectra and $10\thinspace\mu\mathrm{K}\thinspace\mathrm{arcmin}$ noise over $10 \le \ell < 128$,
tolerance $10^{-3}$:

| $\epsilon_{\mathrm{pair}}$ | pairs kept | TT error per element, relative to $\sqrt{C_{\ell\ell}C_{\ell'\ell'}}$ | TT variance of $\Delta\ell = 20$ bandpowers |
| --- | --- | --- | --- |
| $\mathrm{tol}/40$ (default) | 22% | $2$ to $4\times10^{-3}$ | $-0.6\%$ |
| $\mathrm{tol}/10$ | 17% | $5$ to $7\times10^{-3}$ | $-1.3\%$ |
| $\mathrm{tol}/4$ | 13% | $6$ to $10\times10^{-3}$ | $-2.0\%$ |
| $\mathrm{tol}$ | 9% | $0.9$ to $1.6\times10^{-2}$ | $-3.0\%$ |

In short: **the fewer pairs kept, the lower the covariance.** Three things
to know:

- **It is a bias, not scatter.** Every dropped term is a positive
  semi-definite contribution, so the TT covariance is underestimated at every
  $\ell$. EE and TE errors are of the same size with mixed sign (EE binned
  variance $-0.3\%$ at the default).
- **Relative to the diagonal, it does not grow with $\ell' - \ell$.** The
  far kernels are less accurate relative to themselves (above) but they are
  small, so the far-diagonal elements are no worse than the near ones.
- **It is larger than the kernel's Frobenius error**, because the dropped
  pairs sit in the kernel's wings, $L$ far from $\ell_\ast$, where the
  Frobenius norm hardly looks but a red spectrum, translated to
  $\ell < \ell_\ast$, weights heavily. The TT diagonal error peaks near
  $\ell_\ast/2$ ($-2.3\times10^{-3}$ at $\ell = 30$, against
  $-3.9\times10^{-4}$ at $\ell_\ast$). The $m$ and band rules contribute
  nothing noticeable; it is the pair rule.

For scale, the ACC translation error itself is of the same order
($-5.6\times10^{-3}$ at $\ell = 500$ on the survey footprint). The default
was kept for that reason. The production configuration ($\ell_\ast = 250$,
$n_{\mathrm{side}} = 256$) has not been measured this way.

### Limitation: the E to B leakage kernels

The selection rules estimate which terms matter from the mode power of the
response of the E field, and that power does not bound the response of the
leakage field $L$. On a mask that is not azimuthally symmetric about its centre
(in the pole frame), the kernels with an $L$ leg (`LL`, `TL`, `LT`, `DL`, `LD`)
can therefore miss the requested tolerance: measured on a two-blob mask at
$n_{\mathrm{side}} = 32$ and tolerance $10^{-3}$, `LLxLL` is off by up to
$1.2\times10^{-2}$ and the other leakage-leg kernels by $1.0$ to $2.9\times10^{-3}$,
while the kernels without an $L$ leg stay inside the tolerance. The
implementation itself is exact (it reproduces a naive selection of the full
integrals to $10^{-13}$); the rules are the cause. `precompute_acc_kernels`
warns when `term_selection` is combined with an leakage-leg channel, and
`term_selection=None` avoids the problem.

### Checking your own mask

What matters is the covariance, so compare covariances: precompute a small
cache (low $n_{\mathrm{side}}$ and $\ell_\ast$, a few seconds to a minute)
once with `term_selection=None` and once with your tolerance, assemble the
covariance from each with your spectra, and compare the binned variances.
