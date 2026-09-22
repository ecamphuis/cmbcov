# B-mode covariance kernels

A mask leaks E modes into pseudo-B, and for a realistic B-mode spectrum that
leaked E power, not the B power itself, dominates the covariance of every
block with a B leg. This page shows how the package writes each covariance
block among TT, EE, BB, TE, TB and EB as a sum of Wick terms built from
three coupling fields, how each term is normalised and translated by
[ACC](acc.md), how the PolSpice transform mixes EE and BB, and where the
method is and is not accurate.

Prerequisites: [ACC](acc.md) (kernels, Eqs. 22, 23 and 33) and
[exact covariance](exact_covariance.md) (the reference). Configuration:
[polarisation](../polarisation.md), [parameters](../parameters.md),
[PolSpice](../polspice.md).

## 1. Coupling fields and channels

Masking acts on a unit input mode as follows, with $K$ the pseudo-coefficient
operator of [ACC](acc.md), Sect. 1:

```math
K e_{\ell m}^{T} = u_{\ell m}^{T},
\qquad
K e_{\ell m}^{E} = \big(u_{\ell m}^{D},\ +u_{\ell m}^{L}\big),
\qquad
K e_{\ell m}^{B} = \big(-u_{\ell m}^{L},\ +u_{\ell m}^{D}\big),
```

the pairs being the (E, B) components of the output. There are only three
**coupling fields**: T ($u^T$, spin 0), D ($u^D$, the direct spin-2
response) and L ($u^L$, the E-to-B leakage). A unit B mode needs no field of
its own: its response is the E response with the roles of D and L exchanged
and one sign flipped. This holds bit for bit, with no conjugation, on the
Gauss-Legendre grid.

A **channel** pairs the field of the $\ell$ leg with the field of the
$\ell'$ leg, $X_{mm'}^{ab}(L) = \sum_M u_{\ell m, LM}^{a}\thinspace\overline{u_{\ell' m', LM}^{b}}$.
The nine channels are all ordered pairs of $(T, D, L)$:

| channel | $\ell$ leg | $\ell'$ leg | meaning |
|---|---|---|---|
| TT | T | T | spin-0 response |
| DD | D | D | direct spin-2 response |
| LL | L | L | leakage against leakage |
| TD, DT | T, D | D, T | temperature against direct spin-2 |
| TL, LT | T, L | L, T | temperature against leakage |
| DL, LD | D, L | L, D | direct spin-2 against leakage |

TD and DT, and likewise TL and LT, DL and LD, are different kernels off the
diagonal $\ell = \ell'$. TT, EE, BB, TE, TB and EB are spectra and covariance
blocks, never channels: $\Theta^{LL\times LL}$ is the leakage kernel, not a
kernel of $C^{BB}$.

## 2. Correlators

With $v^{XY} = (K e^X)^Y$ read off the three columns above, the correlator
$R^{XZ} = \sum_{YY'} C^{YY'}\thinspace v^{XY}\thinspace\overline{v^{ZY'}}$ is a signed sum
of channels weighted by true spectra. The true spectrum is written in leg
order, first letter from the $X$ leg:

| $R^{XZ}$ | terms (sign, true spectrum, channel) |
|---|---|
| TT | $(+, TT, TT)$ |
| TE | $(+, TE, TD)$, $(+, TB, TL)$ |
| ET | $(+, ET, DT)$, $(+, BT, LT)$ |
| TB | $(-, TE, TL)$, $(+, TB, TD)$ |
| BT | $(-, ET, LT)$, $(+, BT, DT)$ |
| EE | $(+, EE, DD)$, $(+, EB, DL)$, $(+, BE, LD)$, $(+, BB, LL)$ |
| BB | $(+, EE, LL)$, $(-, EB, LD)$, $(-, BE, DL)$, $(+, BB, DD)$ |
| EB | $(-, EE, DL)$, $(+, EB, DD)$, $(-, BE, LL)$, $(+, BB, LD)$ |
| BE | $(-, EE, LD)$, $(-, EB, LL)$, $(+, BE, DD)$, $(+, BB, DL)$ |

Every coefficient is a real $\pm1$. When $C^{TB} = C^{EB} = 0$ the terms with
a TB, BT, EB or BE spectrum drop, and two readings follow:

- $R^{BB} = C^{EE} X^{LL} + C^{BB} X^{DD}$: pseudo-B correlations contain the
  leaked E power $C^{EE} X^{LL}$, which dominates whenever $C^{BB}$ is much
  smaller than $C^{EE}$.
- $R^{TB} = -C^{TE} X^{TL}$ and $R^{EB} = -C^{EE} X^{DL} + C^{BB} X^{LD}$
  are non-zero although the sky has no TB or EB correlation: the mask
  correlates pseudo-B with T and E.

## 3. Why BB is dominated by leaked E power

Wick's theorem ([ACC](acc.md), Sect. 1) with the table above gives

```math
n\thinspace\mathrm{Cov}(BB, BB) = 2\Big[EE\cdot\Theta^{LL\times LL}\cdot EE + 2\thinspace EE\cdot\Theta^{LL\times DD}\cdot BB + BB\cdot\Theta^{DD\times DD}\cdot BB\Big],
```

with $n = (2\ell+1)(2\ell'+1)$ and $EE$ standing for $C^{EE}$ inside the
products. The assignment is crossed: the leakage kernel carries
$(C^{EE})^2$, and $(C^{BB})^2$ rides on the direct kernel, because a true B
mode reaches pseudo-B through $u^D$. Two other blocks are pure leakage and
survive with $C^{BB} = 0$: EE×BB, through its term
$2\thinspace EE\cdot\Theta^{DL\times DL}\cdot EE$, and TT×BB, whose only term is
$2\thinspace TE\cdot\Theta^{TL\times TL}\cdot TE$ (Sect. 4).

How large this is on a real footprint: on an apodised survey mask covering
about 4% of the sky, with Planck 2018 lensed spectra including lensing B
modes and no noise, the leakage terms make up 78% of the exact pseudo
Cov(BB,BB) at $\ell = 250$ and 74 to 93% over $150 \le \ell \le 400$. After
the PolSpice transform, $\sigma(BB)$ is 1.9 to 3.3 times the value obtained
with the leakage terms removed, over $200 \le \ell \le 380$.

This is why NKA and INKA cannot produce B blocks. Both build the covariance
from products of two (smoothed) spectra with one $\Xi$, so $C^{BB}$ never
sees $C^{EE}$, and TT×BB, EE×BB and TE×BB, which are pure leakage, would
come out exactly zero whatever normalisation is chosen. The package
therefore requires ACC for any B observable. The one exception is
`observables: [BB]` alone with NKA or INKA, a shortcut for high multipoles
that computes $2\thinspace(C^{BB})^2\thinspace\Xi^{EE\to EE}$ and ignores
the leakage. Such a run warns with a validity threshold $\ell_{\mathrm{safe}}$
(`Cov.bb_leakage_ell_safe`): the smallest multipole above which the
estimated missed variance $(1 + \mathrm{leak}/\mathrm{signal})^2$ stays below
1.05, where $\mathrm{signal} = ({}^{+}M\thinspace C^{BB})\mathstrut_{\ell}$,
$\mathrm{leak} = ({}^{-}M\thinspace C^{EE})\mathstrut_{\ell}$ and
${}^{\pm}M = \frac{1}{2}(M_1 \pm M_2)$ are the MASTER EE+BB and EE−BB
channels. On the small test configuration where it was checked against the
exact covariance, the estimate is conservative everywhere and agrees with
the true missed variance within about 20 to 40% for $\ell \gtrsim 15$; it
places the warning, it does not correct the covariance.

## 4. Covariance blocks as sums of Wick terms

Inserting the correlators into Wick's theorem writes every block as

```math
n\thinspace\mathrm{Cov}\big(\tilde C_{\ell}^{s_1}, \tilde C_{\ell'}^{s_2}\big) = \sum_{t}\sigma_t\thinspace C^{a_t}\cdot\Theta_{\ell\ell'}^{c_t\times c_t'}\cdot C^{b_t},
```

with integer coefficients $\sigma_t$ and the symmetry
$A\cdot\Theta^{ab\times cd}\cdot B = B\cdot\Theta^{cd\times ab}\cdot A$ used to
merge mirror terms. With $C^{TB} = C^{EB} = 0$, one orientation of each
parity-respecting block:

| block | $n\thinspace\mathrm{Cov}$ |
|---|---|
| TT×TT | $2\thinspace TT\cdot\Theta^{TT\times TT}\cdot TT$ |
| TT×EE | $2\thinspace TE\cdot\Theta^{TD\times TD}\cdot TE$ |
| TT×TE | $2\thinspace TT\cdot\Theta^{TT\times TD}\cdot TE$ |
| TT×BB | $2\thinspace TE\cdot\Theta^{TL\times TL}\cdot TE$ |
| EE×EE | $2\thinspace EE\cdot\Theta^{DD\times DD}\cdot EE + 4\thinspace EE\cdot\Theta^{DD\times LL}\cdot BB + 2\thinspace BB\cdot\Theta^{LL\times LL}\cdot BB$ |
| EE×TE | $2\thinspace EE\cdot\Theta^{DD\times DT}\cdot TE + 2\thinspace BB\cdot\Theta^{LL\times DT}\cdot TE$ |
| EE×BB | $2\thinspace EE\cdot\Theta^{DL\times DL}\cdot EE - 4\thinspace EE\cdot\Theta^{DL\times LD}\cdot BB + 2\thinspace BB\cdot\Theta^{LD\times LD}\cdot BB$ |
| TE×TE | $TT\cdot\Theta^{TT\times DD}\cdot EE + TE\cdot\Theta^{TD\times DT}\cdot TE + TT\cdot\Theta^{TT\times LL}\cdot BB$ |
| TE×BB | $2\thinspace TE\cdot\Theta^{TL\times DL}\cdot EE - 2\thinspace TE\cdot\Theta^{TL\times LD}\cdot BB$ |
| BB×BB | $2\thinspace BB\cdot\Theta^{DD\times DD}\cdot BB + 4\thinspace BB\cdot\Theta^{DD\times LL}\cdot EE + 2\thinspace EE\cdot\Theta^{LL\times LL}\cdot EE$ |
| TB×TB | $TT\cdot\Theta^{TT\times DD}\cdot BB + TT\cdot\Theta^{TT\times LL}\cdot EE + TE\cdot\Theta^{TL\times LT}\cdot TE$ |
| TB×EB | $BB\cdot\Theta^{DD\times TD}\cdot TE + EE\cdot\Theta^{LL\times TD}\cdot TE - TE\cdot\Theta^{TL\times DL}\cdot BB + TE\cdot\Theta^{TL\times LD}\cdot EE$ |
| EB×EB | $EE\cdot\Theta^{DD\times DD}\cdot BB + EE\cdot\Theta^{DD\times LL}\cdot EE + BB\cdot\Theta^{DD\times LL}\cdot BB - EE\cdot\Theta^{DL\times DL}\cdot BB$ |
| | $\quad + EE\cdot\Theta^{DL\times LD}\cdot EE + BB\cdot\Theta^{DL\times LD}\cdot BB - EE\cdot\Theta^{LD\times LD}\cdot BB + EE\cdot\Theta^{LL\times LL}\cdot BB$ |

Readings:

- The T/E blocks EE×EE, EE×TE and TE×TE gain $C^{BB}$ terms, all of the
  auto-leakage type (an LL channel). They are small for lensing-level
  $C^{BB}$, but they are part of the exact result, so a run with a B
  observable includes them.
- $\Theta^{TL\times LT}$ in TB×TB plays the role that $\Theta^{TD\times DT}$
  plays in TE×TE, and differs from $\Theta^{TL\times TL}$ off the diagonal.
- As $C^{BB} \to 0$ the two surviving terms of EB×EB,
  $EE\cdot\Theta^{DD\times LL}\cdot EE$ and $EE\cdot\Theta^{DL\times LD}\cdot EE$,
  nearly cancel (Sect. 8).
- With $C^{TB}$ or $C^{EB}$ non-zero every block gains terms carrying them,
  up to 30 merged terms for EB×EB, and needs additional kernel pairs
  (Sect. 6).
- The other orientation of an off-diagonal block, for example Cov(BB,TE)
  against Cov(TE,BB), is a different expansion; it agrees at $\ell = \ell'$
  and differs off it, like the TT×EE asymmetry of [ACC](acc.md), Sect. 3.

**Parity-mixed blocks.** Blocks between a parity-even spectrum (TT, EE, BB,
TE) and a parity-odd one (TB, EB), such as TT×TB, change sign under a
reflection of the mask when $C^{TB} = C^{EB} = 0$. They vanish exactly for a
mask with a mirror plane and are small but non-zero otherwise: on the 4%
survey footprint of Sect. 3 the exact rows put them at most
$2.6\times10^{-4}$ of $\sqrt{\mathrm{Cov}\mathstrut_{aa}\mathrm{Cov}\mathstrut_{bb}}$.
They are set to zero unless `parity_mixed_blocks` is requested.

The expansion is generated, not hand-written: `bmode_wick.block_wick_terms`
for a pair of spectra, `bmode_wick.covkey_wick_terms` for one block with
its frequencies. With kernels built at the true $(\ell, \ell')$ it reproduces
`exact_covariance_row_pol` to $1.1\times10^{-15}$ relative for all 36 ordered
blocks, with and without $C^{TB}$ and $C^{EB}$.

## 5. Per-term normalisation

ACC evaluates each term with its kernel $\Theta_\ast$ computed at
$(\ell_\ast, \ell_\ast + \Delta)$, with sum $S_\ast = \sum\Theta_\ast$ and
$n_\ast = (2\ell_\ast+1)(2\ell_\ast+2\Delta+1)$, translated by Eq. 33 and
multiplied by its own normalisation $N_t$:

```math
\tilde\Sigma_t(\ell, \ell+\Delta) = \sigma_t\thinspace N_t(\ell, \ell+\Delta)\thinspace C_{+s}^{a}\cdot\frac{\Theta_\ast}{S_\ast}\cdot C_{+s}^{b},
\qquad C_{+s}(L) = C(L+s),\quad s = \ell - \ell_\ast .
```

**Why not Eq. 23 per kernel.** Completeness gives exactly six identities
among kernel sums:

```math
\begin{aligned}
\sum\Theta^{TT\times TT} &= n\thinspace\Xi^{00}, &
\sum\big(\Theta^{TT\times DD}+\Theta^{TT\times LL}\big) &= n\thinspace\Xi^{20},\\
\sum\big(\Theta^{DD\times DD}+2\thinspace\Theta^{DD\times LL}+\Theta^{LL\times LL}\big) &= n\thinspace\Xi^{EE\to EE}, &
\sum\big(\Theta^{DL\times DL}-2\thinspace\Theta^{DL\times LD}+\Theta^{LD\times LD}\big) &= n\thinspace\Xi^{EE\to BB},\\
\sum\big(\Theta^{TT\times LD}-\Theta^{TT\times DL}\big) &= 0, &
\sum\big(\Theta^{DD\times LD}-\Theta^{DD\times DL}+\Theta^{LL\times LD}-\Theta^{LL\times DL}\big) &= 0,
\end{aligned}
```

all $\Xi$ being of $W^2$ at $(\ell, \ell')$. None of them fixes the sum of a
single leakage kernel, and none contains a TL or LT channel, because a spin-0
and a spin-2 function have no joint completeness relation. Eq. 23 applied
to each kernel gives a leakage kernel the full amplitude of its non-leakage
partner; on the test masks used to derive the rule, that puts every block
with an L leg off by a factor of 7 to 9500. Normalising a whole block by its block-level
identity instead freezes the leakage share of the block at its $\ell_\ast$
value, although that share changes with $\ell$ and with the spectra.

**The rule.** Each term is normalised on its own, from three properties of
its kernel pair: the spin weight $w$ (each D or L letter weighs
$\frac{1}{2}$, T weighs 0), the number $k$ of L letters, and the leakage
ratio $\rho = \Xi^{EE\to BB}/\Xi^{EE\to EE}$ of the MASTER channels, which
falls roughly as $\ell^{-2}$ along a diagonal. With
$\Xi^{(w)} = \Xi^{00}(\Xi^{20}/\Xi^{00})^{w}$ as in [ACC](acc.md), Sect. 5,

```math
N_t(\ell, \ell') = \Xi_{\ell\ell'}^{(w)}\thinspace\rho_{\ell\ell'}^{k/2}\thinspace c_t(\ell),
\qquad
c_t^{\ast} = \frac{S_\ast}{n_\ast\thinspace\Xi_{\ast}^{(w)}\thinspace\rho_{\ast}^{k/2}},
```

where the subscript $\ast$ means evaluation at $(\ell_\ast, \ell_\ast+\Delta)$.
Every term is exact at $\ell_\ast$, because $c_t(\ell_\ast) = c_t^{\ast}$. The
coefficient $c_t(\ell)$ depends on the class of the pair, which depends on
its two channels only (`bmode_wick.term_class`):

| class | kernel pairs | $c_t(\ell)$ |
|---|---|---|
| no leakage, $k = 0$ | TT×TT | 1: Eq. 23 with $\Xi^{00}$, unchanged |
| no leakage, $k = 0$ | TT×DD, TT×TD, TD×TD, TD×DT, DD×DD, DD×DT, ... | $1 - (1 - c_t^{\ast})\thinspace\rho_{\ell\ell'}/\rho_{\ast}$ |
| auto leakage | an LL channel: TT×LL, DD×LL, TD×LL, LL×DT, LL×LL | $c_t^{\ast}$, frozen |
| cross leakage | both channels in TL, LT, DL, LD | $\pm\frac{1}{4} + (c_t^{\ast} \mp \frac{1}{4})\thinspace\dfrac{2\ell_\ast+\Delta+1}{2\ell+\Delta+1}$ |

In the cross-leakage row, $\pm$ is the sign of $c_t^{\ast}$,
$\ell = \min(\ell, \ell')$, and $\vert c_t\vert$ is clipped to
$[0, \frac{1}{2}]$ with its sign kept. Kernel pairs with an odd number of L
letters and no LL channel, such as TT×TL, occur only with non-zero
$C^{TB}$, $C^{EB}$ or in parity-mixed blocks; they are treated like auto
leakage.

Where the classes come from. For $k = 0$, the second identity gives
$1 - c_{TT\times DD} = c_{TT\times LL}\thinspace\rho$ exactly, so the
deficit of a no-leakage term decays like $\rho$ and Eq. 23 is recovered as
$\ell \to \infty$. For cross leakage, one L leg on each side, the measured
$c$ tends to $\pm\frac{1}{4}$ along every diagonal on the test masks, almost
independently of the mask. For auto leakage, $c$ depends on the mask and on $\Delta$, and on
the small test masks it was nearly constant along a diagonal; on a realistic
footprint it is not (Sect. 8).

The rule applies to every block of a run with any B observable, the T/E
blocks included; a run whose observables are only TT, EE and TE keeps
Eq. 23 per Wick contraction ([ACC](acc.md), Sect. 5). Implementation:
`Cov.acc_term_scale` returns $N_t/S_\ast$, `Cov.acc_term_normaliser` returns
$N_t$, and `Cov.acc_xi_channels` supplies $\Xi^{00}$, $\Xi^{20}$,
$\Xi^{EE\to EE}$, $\Xi^{EE\to BB}$ and $\rho$. Requirements that follow:

- The rule reads raw kernel sums, so the kernels must be computed on the
  Gauss-Legendre grid; HEALPix-grid kernels carry a factor $\sqrt{2}$ per
  spin-2 integral and are refused for B runs.
- $\ell_\ast \ge 2$, since $\rho$ is undefined below.
- TT×TT is exact at $\ell_\ast$ only to the accuracy of Eq. 22 with the
  package's $\Xi$, of order $10^{-12}$ relative.

## 6. Kernel-pair sets

The kernel pairs a run needs follow from the expansion of its blocks. Pairs
are counted up to transposition, and for an off-diagonal block the
orientation that minimises the total is stored; the other orientation is
then computed as the transpose. `bmode_wick.required_kernel_pairs` returns
the set, and a parameter-file precompute (`cmbcov-precompute`) requests it
automatically unless an explicit `spectra` list is given, reading the TB
and EB columns of the spectra files to decide whether they are zero.

| observables | TB, EB spectra | kernel pairs | channels |
|---|---|---|---|
| TT, EE, TE | not read | 10 | TT, DD, TD, DT |
| TT, EE, TE, BB | not read | 17 | + LL, TL, DL, LD |
| TT, EE, TE, BB, TB, EB | zero | 18 | all nine |
| TT, EE, TE, BB, TB, EB | non-zero | 40 | all nine |
| as above, with `parity_mixed_blocks` | zero / non-zero | 31 / 40 | all nine |
| BB only | not read | 3 | DD, LL |

A cache built for the zero case is refused, with a message naming the
missing pairs, by a run whose TB or EB spectra turn out non-zero.

## 7. PolSpice post-processing

The PolSpice estimator is a linear map of the pseudo-spectra, so its
covariance is $\hat\Sigma = G\thinspace\tilde\Sigma\thinspace G^T$ (the paper's
Eq. 55), with the decoupled kernels ${}^{\pm}G = \frac{1}{2}({}^{\mathrm{dec}}G \pm {}^{-2}G)$ of
Eq. 56. In the mean:

```math
\hat C^{EE} = {}^{+}G\thinspace\tilde C^{EE} + {}^{-}G\thinspace\tilde C^{BB},
\qquad
\hat C^{BB} = {}^{-}G\thinspace\tilde C^{EE} + {}^{+}G\thinspace\tilde C^{BB},
\qquad
\hat C^{TB} = {}^{\times}G\thinspace\tilde C^{TB},
\qquad
\hat C^{EB} = {}^{-2}G\thinspace\tilde C^{EB} .
```

TB is read off the same real-space step as TE, so it takes ${}^{\times}G$.
EB lives in the imaginary part of $\xi_{-}$ only, which the decoupling does not
touch, so it takes ${}^{-2}G = {}^{+}G - {}^{-}G$ with or without
decoupling. In the mean, TB does not mix with TE, and EB does not mix with
EE or BB. The covariance of the PolSpice spectra is then

```math
\mathrm{Cov}\big(\hat C^{X}, \hat C^{Y}\big) = \sum_{a\in\mathrm{src}(X)}\sum_{b\in\mathrm{src}(Y)} G_{X\leftarrow a}\thinspace\mathrm{Cov}\big(\tilde C^{a}, \tilde C^{b}\big)\thinspace G_{Y\leftarrow b}^{T},
```

with $\mathrm{src}(EE) = \mathrm{src}(BB) = \lbrace EE, BB\rbrace$ and
$\mathrm{src}(X) = \lbrace X\rbrace$ otherwise
(`CovariancePostProcessor.pseudo_to_spice_bmode`). A block with an EE or BB
leg therefore needs up to four pseudo blocks, which is why a run with EE and
a B observable must also have BB among its observables when
`polspice_postprocess` is on. A T/E-only run uses the diagonal transform
(`CovariancePostProcessor.pseudo_to_spice`) and so omits the
${}^{-}G\thinspace\mathrm{Cov}(\tilde C^{BB})\thinspace{}^{-}G^{T}$ term of
its EE output, negligible for lensing-level $C^{BB}$. The BB-only NKA/INKA
shortcut uses $\hat C^{BB} = {}^{+}G\thinspace\tilde C^{BB}$, consistent
with ignoring leakage.

The decoupling kernel ${}^{\mathrm{dec}}G$ is built following Chon et al. (2004),
Sect. 5; the single integral printed as the paper's Eq. 54 is not that
kernel (see [PolSpice](../polspice.md)). With it, the transformed mean
response is block-diagonal in (EE, BB) to $10^{-12}$: EE, BB and EB all
respond to the true spectra through the same ${}^{-2}K$, and TE and TB
through ${}^{\times}K$.

## 8. Known limits

Validated on the apodised survey footprint of Sect. 3 (about 4% of the sky,
$\ell_\ast = 250$, 18 kernel pairs, $\mathrm{dmax} = 20$, Planck 2018 lensed
spectra with lensing B modes, $C^{TB} = C^{EB} = 0$, no noise), against exact
rows at $\ell' = 150$ to 400. Errors below are in units of
$\sqrt{\mathrm{Cov}\mathstrut_{aa}\mathrm{Cov}\mathstrut_{bb}}$, the largest
within 19 multipoles of each row, unless stated otherwise.

- **At $\ell_\ast$** every block is exact to the precision of the kernels:
  T/E blocks to $8\times10^{-5}$, leakage blocks to $2$ to $4\times10^{-4}$.
- **T/E blocks, and the B blocks made only of cross-leakage terms**
  (TT×BB, EE×BB, TE×BB), are sub-percent away from $\ell_\ast$: at most 0.44% for $\ell' \ge 200$, and
  at most 0.47% at $\ell' = 150$ apart from TT×TT, which is at 1.5%, the
  translation floor of [ACC](acc.md).
- **BB×BB, TB×TB and EB×EB, whose leading terms are auto leakage, are not
  sub-percent away from $\ell_\ast$.** The
  diagonal of BB×BB is off by +46% at $\ell = 150$, +12% at 200, −6.5% at
  300 and −13% at 400, relative to the exact diagonal; so the pseudo
  $\sigma(BB)$ is 21% high at 150 and 7% low at 400. TB×TB and EB×EB are
  15 to 16% high at 150, 5% at 200 and −2% at 400.

  The cause is the auto-leakage class: its coefficient $c$ is frozen at
  $\ell_\ast$, but on this footprint it is not constant along a diagonal.
  At $\Delta = 0$, $c$ of $TT\times LL$ rises from 1.9 at $\ell = 62$ to 3.3
  at $\ell = 250$ and 3.5 at $\ell = 500$. The frozen value predicts +47% on
  the BB×BB diagonal at $\ell = 150$; +46% is measured. The cross-leakage
  coefficients, by contrast, are within 9% of $\pm\frac{1}{4}$ at every
  $\Delta$ at $\ell_\ast$, as their law assumes.
- **Near-cancelling EB blocks.** When $C^{BB}$ is well below
  $\rho\thinspace C^{EE}$, EB×EB and TB×EB are small differences of large
  auto- and cross-leakage terms, and percent-level errors on each term are
  amplified. With $C^{BB} = 0$ exactly no per-term or per-block rule reaches
  them: on a small test mask at $\ell_\ast = 16$ the errors are 94% on
  TB×EB and 212% on EB×EB. Use exact rows in that regime.
- **Parity-mixed blocks** with $C^{TB} = C^{EB} = 0$ are exact at
  $\ell_\ast$ but their shape away from $\ell_\ast$ is not modelled: the
  relative error is of order one, although the blocks themselves are of
  order $10^{-4}$ of the diagonal.

What to do:

- Put $\ell_\ast$ near the bins where the BB, TB or EB covariance matters
  most. On a footprint like this one, treat BB×BB, TB×TB and EB×EB as
  uncertain at the 5 to 15% level 50 multipoles from $\ell_\ast$, and up to
  about 50% at 100 multipoles below it.
- Or validate on your own mask: compute exact rows with
  `exact_covariance_row_pol` at two or three $\ell'$ across the range you
  use, with the same band-limited mask and an `lmax_int` margin
  ([exact covariance](exact_covariance.md), Sect. 6), and compare the BB×BB,
  TB×TB and EB×EB rows with the ACC pseudo-Cℓ blocks before the PolSpice
  transform. Each row covers every block at once.
