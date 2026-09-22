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
\epsilon_m = \mathrm{tol}/10, \qquad \epsilon_{\mathrm{pair}} = \mathrm{tol}/4, \qquad \delta_{\mathrm{band}} = \mathrm{tol}/10,
```

which were calibrated on a survey footprint (Sect. 5). It is a target, not
a guarantee (Sect. 6). The tolerance and the fractions of $m$, pairs and
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

## 6. Known limitation: asymmetric masks

The default thresholds were calibrated on a single compact footprint. On a
mask with no dominant centre, the pair rule under-selects. Measured on a
two-blob mask, a large cap plus a smaller cap 50° away, at tolerance
$10^{-3}$: the selected kernel is off by $1.4\times10^{-3}$ at
$\ell' = \ell$ and $3.0\times10^{-3}$ at $\ell' = \ell + 3$, above the
tolerance. The $m$ rule alone leaves $3\times10^{-7}$ and the band rule
alone about $10^{-5}$, so the pair rule is responsible. A ten times
tighter $\epsilon_{\mathrm{pair}}$ leaves $2.4\times10^{-4}$ and
$5.3\times10^{-4}$, within the $10^{-3}$ asked for.

For a mask made of several separated patches, or any mask far from a single
cap:

- ask for a tolerance ten times tighter than the accuracy you need, which
  tightens all three thresholds by ten, and
- check two kernel pairs: compute $(\ell_\ast, \ell_\ast)$ and
  $(\ell_\ast, \ell_\ast + \mathrm{dmax} - 1)$ once with
  `term_selection=None` and once with your tolerance, using
  `precompute_acc_kernels(..., ellprange=[...], dryrun=True)`, which returns
  the kernels without writing them, and compare them in the Frobenius norm.
